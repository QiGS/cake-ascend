"""kmeans_assign: idx[t] = argmin_k ||x_t - c_k||^2 — the paper's workload.

GEMM formulation (as in the paper: "assign is a compute-bound BF16
GEMM-and-reduction kernel"):
    cross = X @ Ct^T-chunked   (cube, bf16 x bf16 -> fp32 l0c)
    dist  = x_norm[t] + c_norm[k] - 2*cross[t,k]   (vector epilogue)
    idx   = argmin_k dist       (vector)

Ct is stored (D, K) so the B operand loads directly as (DB, KB) tiles.
Four roles with explicit handoffs in both directions: MTE2 (load), CUBE
(cross), V (norms + epilogue + argmin), MTE3 (store indices).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .. import dtypes as dt
from . import Workload, gen_uniform, register


@dataclass
class KMeansParams:
    tb: int = 32
    db: int = 32
    stages: int = 2
    block_dim: int = 2


SPEC = {
    "tb": [16, 32, 64, 128],
    "db": [16, 32, 64],
    "stages": [1, 2, 3],
    "block_dim": [1, 2, 4],
}

NAIVE = KMeansParams(tb=16, db=16, stages=1, block_dim=1)


def _kern(shape, p):
    NT, K, D = shape["N"], shape["K"], shape["D"]
    TB, DB, S = p.tb, p.db, p.stages
    KB = K   # full center set in one tile (argmin needs the whole row)

    def kern(m):
        X = m.gm_param("X", "bf16", (NT, D))
        Ct = m.gm_param("Ct", "bf16", (D, K))     # centers transposed host-side
        IDX = m.gm_param("IDX", "int32", (NT,))

        def align32(x):
            return (x + 31) // 32 * 32

        ub = m.ub_pool("ub", 232 * 1024)
        off = 0
        x_off = off; off += align32(TB * DB * 2 * S)
        c_off = off; off += align32(DB * KB * 2 * S)
        x2_off = off; off += align32(TB * DB * 4)
        xrow_off = off; off += align32(TB * 4)
        xn_off = off; off += align32(TB * 4)
        ct_off = off; off += align32(KB * DB * 4)
        c2_off = off; off += align32(KB * DB * 4)
        crow_off = off; off += align32(KB * 4)
        cn_off = off; off += align32(KB * 4)
        cross_off = off; off += align32(TB * KB * 4)
        t1_off = off; off += align32(TB * KB * 4)
        t2_off = off; off += align32(TB * KB * 4)
        d1_off = off; off += align32(TB * KB * 4)
        d2_off = off; off += align32(TB * KB * 4)
        idx_off = off; off += align32(TB * 4)

        xT = ub.view("xT", x_off, (TB, DB), "bf16", S)
        cT = ub.view("cT", c_off, (DB, KB), "bf16", S)
        x2 = ub.view("x2", x2_off, (TB, DB), "fp32", 1)
        xrow = ub.view("xrow", xrow_off, (TB,), "fp32", 1)
        xnorm = ub.view("xnorm", xn_off, (TB,), "fp32", 1)
        ctt = ub.view("ctt", ct_off, (KB, DB), "fp32", 1)
        c2 = ub.view("c2", c2_off, (KB, DB), "fp32", 1)
        crow = ub.view("crow", crow_off, (KB,), "fp32", 1)
        cnorm = ub.view("cnorm", cn_off, (KB,), "fp32", 1)
        cross = ub.view("cross", cross_off, (TB, KB), "fp32", 1)
        t1 = ub.view("t1", t1_off, (TB, KB), "fp32", 1)
        t2 = ub.view("t2", t2_off, (TB, KB), "fp32", 1)
        d1 = ub.view("d1", d1_off, (TB, KB), "fp32", 1)
        d2 = ub.view("d2", d2_off, (TB, KB), "fp32", 1)
        idx = ub.view("idx", idx_off, (TB,), "int32", 1)
        acc = m.l0c("acc", (TB, KB))

        ld = m.role("ld", "MTE2")
        cu = m.role("cu", "CUBE")
        v = m.role("v", "V")
        st = m.role("st", "MTE3")
        pipe = m.pipeline("main", S)
        x_rdy_m = m.event("x_rdy_m", ld, cu, pipe)
        c_rdy_m = m.event("c_rdy_m", ld, cu, pipe)
        x_free_m = m.event("x_free_m", cu, ld, pipe)
        c_free_m = m.event("c_free_m", cu, ld, pipe)
        x_rdy_v = m.event("x_rdy_v", ld, v, pipe)
        c_rdy_v = m.event("c_rdy_v", ld, v, pipe)
        x_free_v = m.event("x_free_v", v, ld, pipe)
        c_free_v = m.event("c_free_v", v, ld, pipe)
        cross_rdy = m.event("cross_rdy", cu, v)
        cross_free = m.event("cross_free", v, cu)
        idx_rdy = m.event("idx_rdy", v, st)
        idx_free = m.event("idx_free", st, v)

        IT = m.num_tiles(NT, TB)
        DC = m.num_tiles(D, DB)
        per = m.num_tiles(IT, m.core_count())
        my0 = m.core_id() * per
        cnt = max(0, min(per, IT - my0))

        with ld:
            for i in m.tile_loop("i", cnt):
                for dch in m.tile_loop("dch", DC):
                    k = i * DC + dch
                    s = k % S
                    if k >= S:
                        m.wait(x_free_m, stage=s)
                        m.wait(x_free_v, stage=s)
                        m.wait(c_free_m, stage=s)
                        m.wait(c_free_v, stage=s)
                    m.gm2ub(xT[s], X, ((my0 + i) * TB, dch * DB))
                    m.commit(x_rdy_m, stage=s)
                    m.commit(x_rdy_v, stage=s)
                    m.gm2ub(cT[s], Ct, (dch * DB, 0))
                    m.commit(c_rdy_m, stage=s)
                    m.commit(c_rdy_v, stage=s)

        with cu:
            for i in m.tile_loop("i", cnt):
                for dch in m.tile_loop("dch", DC):
                    k = i * DC + dch
                    s = k % S
                    m.wait(x_rdy_m, stage=s)
                    m.wait(c_rdy_m, stage=s)
                    m.matmul(acc, xT[s], cT[s], clear=(dch == 0))
                    m.commit(x_free_m, stage=s)
                    m.commit(c_free_m, stage=s)
                if i > 0:
                    m.wait(cross_free, stage=0)
                m.l0c2ub(cross[0], acc)
                m.commit(cross_rdy, stage=0)

        with v:
            for i in m.tile_loop("i", cnt):
                for dch in m.tile_loop("dch", DC):
                    k = i * DC + dch
                    s = k % S
                    m.wait(x_rdy_v, stage=s)
                    m.wait(c_rdy_v, stage=s)
                    # xnorm[t] += sum_d x[t,d]^2
                    m.v_cast(x2[0], xT[s])
                    m.v_binary("mul", x2[0], x2[0], x2[0])
                    m.v_reduce("sum", xrow[0], x2[0])
                    if dch == 0:
                        m.v_unary("copy", xnorm[0], xrow[0])
                    else:
                        m.v_binary("add", xnorm[0], xnorm[0], xrow[0])
                    # cnorm[k] += sum_d c[k,d]^2   (transpose (DB,KB)->(KB,DB))
                    m.v_transpose(ctt[0], cT[s])
                    m.v_binary("mul", c2[0], ctt[0], ctt[0])
                    m.v_reduce("sum", crow[0], c2[0])
                    if dch == 0:
                        m.v_unary("copy", cnorm[0], crow[0])
                    else:
                        m.v_binary("add", cnorm[0], cnorm[0], crow[0])
                    m.commit(x_free_v, stage=s)
                    m.commit(c_free_v, stage=s)
                m.wait(cross_rdy, stage=0)
                if i > 0:
                    m.wait(idx_free, stage=0)
                # dist = xnorm[:,None] + cnorm[None,:] - 2*cross
                m.v_bcast("row", t1[0], xnorm[0])
                m.v_bcast("col", t2[0], cnorm[0])
                m.v_binary("mul", d1[0], cross[0], -2.0)
                m.v_binary("add", d1[0], t1[0], d1[0])
                m.v_binary("add", d2[0], d1[0], t2[0])
                m.v_argmin(idx[0], d2[0])
                m.commit(cross_free, stage=0)
                m.commit(idx_rdy, stage=0)

        with st:
            for i in m.tile_loop("i", cnt):
                m.wait(idx_rdy, stage=0)
                m.ub2gm(IDX, ((my0 + i) * TB,), idx[0])
                m.commit(idx_free, stage=0)

    return kern


def _make_inputs(shape, seed):
    n, k, d = shape["N"], shape["K"], shape["D"]
    rng = random.Random(seed)
    x = dt.quantize_list(gen_uniform(rng, n * d), "bf16")
    c = dt.quantize_list(gen_uniform(rng, k * d), "bf16")
    # Ct (D, K): transposed centers, matching the GEMM operand layout
    ct = [0.0] * (d * k)
    for dd in range(d):
        for kk in range(k):
            ct[dd * k + kk] = c[kk * d + dd]
    return {"X": x, "Ct": dt.quantize_list(ct, "bf16"), "IDX": [0.0] * n}


def _oracle(shape, inputs):
    n, k, d = shape["N"], shape["K"], shape["D"]
    x, ct = inputs["X"], inputs["Ct"]
    # reconstruct centers from Ct for the reference
    centers = [[ct[dd * k + kk] for dd in range(d)] for kk in range(k)]
    dists = []
    idx = []
    for t in range(n):
        row = x[t * d:(t + 1) * d]
        best, bestd = 0, None
        for kk in range(k):
            s = 0.0
            for dd in range(d):
                diff = row[dd] - centers[kk][dd]
                s += diff * diff
            if bestd is None or s < bestd:
                best, bestd = kk, s
        idx.append(float(best))
        dists.append(bestd)
    return {"IDX": idx, "_dist": dists}


def _compare(shape, expected, got, inputs=None):
    n, k, d = shape["N"], shape["K"], shape["D"]
    exp_idx = expected["IDX"]
    got_idx = got.get("IDX")
    if got_idx is None:
        return False, "missing output 'IDX'"
    if len(got_idx) != n:
        return False, f"IDX length {len(got_idx)} != {n}"
    x = inputs["X"]
    ct = inputs["Ct"]

    def center(kk):
        return [ct[dd * k + kk] for dd in range(d)]

    flips = 0
    for t in range(n):
        e, g = int(exp_idx[t]), int(got_idx[t])
        if e == g:
            continue
        # a flip is legal only for a near-tie under bf16 quantization
        row = x[t * d:(t + 1) * d]
        ce, cg = center(e), center(g)
        de = sum((a - b) ** 2 for a, b in zip(row, ce))
        dg = sum((a - b) ** 2 for a, b in zip(row, cg))
        if abs(de - dg) > 5e-3 * max(1.0, de):
            return False, (f"IDX[{t}] = {g}, expected {e} "
                           f"(dist {dg:.4f} vs {de:.4f}, not a near-tie)")
        flips += 1
    note = "exact" if flips == 0 else f"{flips}/{n} near-tie flips"
    return True, note


wl = register(Workload(
    name="kmeans_assign",
    description="argmin_k ||x-c||^2 via GEMM formulation (paper's Flash-KMeans assign)",
    shape={"N": 128, "K": 32, "D": 64},
    domain=[{"N": n, "K": k, "D": d}
            for n in (64, 128, 256) for k in (16, 32) for d in (32, 64)],
    io=[("X", "bf16", ("N", "D"), "in"),
        ("Ct", "bf16", ("D", "K"), "in"),
        ("IDX", "int32", ("N",), "out")],
    tolerance=(0.0, 0.0),
    make_inputs=_make_inputs,
    oracle=_oracle,
    compare=_compare,
    params_class=KMeansParams,
    default_params=lambda: KMeansParams(**NAIVE.__dict__),
    kernel_fn=_kern,
    block_dim=lambda shape, p: p.block_dim,
    domain_guard=lambda shape, p: (shape["N"] % p.tb == 0
                                   and shape["D"] % p.db == 0),
))
