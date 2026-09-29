"""gemm: C = A @ B, bf16 operands, fp32 accumulate/output — cube-bound family.

The seed choreography (per K-tile: double-buffered A/B tiles, per-tile
back-pressure events, per-row-tile C handoff) is the Ascend analog of the
paper's producer/consumer warp-specialization pattern.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .. import dtypes as dt
from . import Workload, default_compare, gen_uniform, register


@dataclass
class GemmParams:
    bm: int = 64
    bn: int = 64
    bk: int = 64
    stages: int = 2
    block_dim: int = 2


SPEC = {
    "bm": [16, 32, 48, 64, 96, 128],
    "bn": [16, 32, 48, 64, 96, 128],
    "bk": [16, 32, 48, 64, 128],
    "stages": [1, 2, 3, 4],
    "block_dim": [1, 2, 4],
}

NAIVE = GemmParams(bm=32, bn=32, bk=32, stages=1, block_dim=1)


def _kern(shape, p):
    M, N, K = shape["M"], shape["N"], shape["K"]
    BM, BN, BK, S = p.bm, p.bn, p.bk, p.stages

    def kern(m):
        A = m.gm_param("A", "bf16", (M, K))
        B = m.gm_param("B", "bf16", (K, N))
        C = m.gm_param("C", "fp32", (M, N))

        def align32(x):
            return (x + 31) // 32 * 32

        ub = m.ub_pool("ub", 232 * 1024)
        off = 0
        a_off = off; off += align32(BM * BK * 2 * S)
        b_off = off; off += align32(BK * BN * 2 * S)
        c_off = off; off += align32(BM * BN * 4)
        bufA = ub.view("A", a_off, (BM, BK), "bf16", S)
        bufB = ub.view("B", b_off, (BK, BN), "bf16", S)
        bufC = ub.view("C", c_off, (BM, BN), "fp32", 1)
        acc = m.l0c("acc", (BM, BN))

        ld = m.role("ld", "MTE2")
        cu = m.role("cu", "CUBE")
        st = m.role("st", "MTE3")
        pipe = m.pipeline("main", S)
        a_rdy = m.event("a_rdy", ld, cu, pipe)
        b_rdy = m.event("b_rdy", ld, cu, pipe)
        a_free = m.event("a_free", cu, ld, pipe)
        b_free = m.event("b_free", cu, ld, pipe)
        c_rdy = m.event("c_rdy", cu, st)
        c_free = m.event("c_free", st, cu)

        IT = m.num_tiles(M, BM)
        JT = m.num_tiles(N, BN)
        KT = m.num_tiles(K, BK)
        per = m.num_tiles(IT, m.core_count())
        my0 = m.core_id() * per
        cnt = max(0, min(per, IT - my0))

        with ld:
            for i in m.tile_loop("i", cnt):
                for j in m.tile_loop("j", JT):
                    for kt in m.tile_loop("kt", KT):
                        k = (i * JT + j) * KT + kt
                        s = k % S
                        if k >= S:
                            m.wait(a_free, stage=s)
                            m.wait(b_free, stage=s)
                        m.gm2ub(bufA[s], A, ((my0 + i) * BM, kt * BK))
                        m.commit(a_rdy, stage=s)
                        m.gm2ub(bufB[s], B, (kt * BK, j * BN))
                        m.commit(b_rdy, stage=s)

        with cu:
            for i in m.tile_loop("i", cnt):
                for j in m.tile_loop("j", JT):
                    for kt in m.tile_loop("kt", KT):
                        k = (i * JT + j) * KT + kt
                        s = k % S
                        m.wait(a_rdy, stage=s)
                        m.wait(b_rdy, stage=s)
                        m.matmul(acc, bufA[s], bufB[s], clear=(kt == 0))
                        m.commit(a_free, stage=s)
                        m.commit(b_free, stage=s)
                    if (i * JT + j) > 0:
                        m.wait(c_free, stage=0)
                    m.l0c2ub(bufC[0], acc)
                    m.commit(c_rdy, stage=0)

        with st:
            for i in m.tile_loop("i", cnt):
                for j in m.tile_loop("j", JT):
                    m.wait(c_rdy, stage=0)
                    m.ub2gm(C, ((my0 + i) * BM, j * BN), bufC[0])
                    m.commit(c_free, stage=0)

    return kern


def _make_inputs(shape, seed):
    M, N, K = shape["M"], shape["N"], shape["K"]
    rng = random.Random(seed)
    a = dt.quantize_list(gen_uniform(rng, M * K), "bf16")
    b = dt.quantize_list(gen_uniform(rng, K * N), "bf16")
    return {"A": a, "B": b, "C": [0.0] * (M * N)}


def _oracle(shape, inputs):
    M, N, K = shape["M"], shape["N"], shape["K"]
    a, b = inputs["A"], inputs["B"]
    out = [0.0] * (M * N)
    for i in range(M):
        ai = a[i * K:(i + 1) * K]
        for j in range(N):
            out[i * N + j] = sum(ai[k] * b[k * N + j] for k in range(K))
    return {"C": out}


wl = register(Workload(
    name="gemm",
    description="C = A@B (bf16 in, fp32 out), cube-bound; tile/stage/partition search",
    shape={"M": 128, "N": 128, "K": 128},
    domain=[{"M": m, "N": n, "K": k}
            for m in (64, 128) for n in (64, 128) for k in (64, 128, 256)],
    io=[("A", "bf16", ("M", "K"), "in"),
        ("B", "bf16", ("K", "N"), "in"),
        ("C", "fp32", ("M", "N"), "out")],
    tolerance=(1e-3, 1e-3),
    make_inputs=_make_inputs,
    oracle=_oracle,
    compare=default_compare((1e-3, 1e-3)),
    params_class=GemmParams,
    default_params=lambda: GemmParams(**NAIVE.__dict__),
    kernel_fn=_kern,
    block_dim=lambda shape, p: p.block_dim,
    domain_guard=lambda shape, p: (shape["M"] % p.bm == 0 and shape["N"] % p.bn == 0
                                   and shape["K"] % p.bk == 0),
))
