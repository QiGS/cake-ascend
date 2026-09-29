"""vec_add: Z = X + Y, bf16 in/out — memory-bound warm-up family.

Exposes the speed/accuracy dial: `prec="fp32"` rounds through fp32
(2 casts + fp32 add + cast), `prec="bf16"` adds in bf16 directly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .. import dtypes as dt
from . import (Workload, argmin_rows, default_compare, gen_uniform, register,
               row_sum, shape_dims)


@dataclass
class VecAddParams:
    tile: int = 1024
    stages: int = 1
    block_dim: int = 2
    prec: str = "fp32"


SPEC = {
    "tile": [128, 256, 512, 1024, 2048, 4096],
    "stages": [1, 2, 3, 4],
    "block_dim": [1, 2, 4, 8],
    "prec": ["fp32", "bf16"],
}

NAIVE = VecAddParams(tile=256, stages=1, block_dim=1, prec="fp32")


def _kern(shape, p):
    n = shape["N"]

    def kern(m):
        X = m.gm_param("X", "bf16", (n,))
        Y = m.gm_param("Y", "bf16", (n,))
        Z = m.gm_param("Z", "bf16", (n,))
        T, S = p.tile, p.stages
        ub = m.ub_pool("ub", 232 * 1024)

        def nxt(b):
            return b  # offsets assigned below via cumulative layout

        # concrete UB layout (agent-authored commitments, checker-verified)
        off = 0
        def align32(x):
            return (x + 31) // 32 * 32

        x_off = off; off += align32(T * 2 * S)
        y_off = off; off += align32(T * 2 * S)
        z_off = off; off += align32(T * 2)
        fx_off = off; off += align32(T * 4 * S) if p.prec == "fp32" else 0
        fy_off = off; off += align32(T * 4 * S) if p.prec == "fp32" else 0
        fz_off = off; off += align32(T * 4 * S) if p.prec == "fp32" else 0

        bufX = ub.view("X", x_off, (T,), "bf16", S)
        bufY = ub.view("Y", y_off, (T,), "bf16", S)
        bufZ = ub.view("Z", z_off, (T,), "bf16", 1)
        if p.prec == "fp32":
            bufFX = ub.view("FX", fx_off, (T,), "fp32", S)
            bufFY = ub.view("FY", fy_off, (T,), "fp32", S)
            bufFZ = ub.view("FZ", fz_off, (T,), "fp32", S)

        ld = m.role("ld", "MTE2")
        v = m.role("v", "V")
        st = m.role("st", "MTE3")
        pipe = m.pipeline("main", S)
        x_rdy = m.event("x_rdy", ld, v, pipe)
        y_rdy = m.event("y_rdy", ld, v, pipe)
        x_free = m.event("x_free", v, ld, pipe)
        y_free = m.event("y_free", v, ld, pipe)
        z_rdy = m.event("z_rdy", v, st)
        z_free = m.event("z_free", st, v)

        total = m.num_tiles(n, T)
        per = m.num_tiles(total, m.core_count())
        my0 = m.core_id() * per
        cnt = max(0, min(per, total - my0))

        with ld:
            for t in m.tile_loop("t", cnt):
                s = t % S
                if t >= S:
                    m.wait(x_free, stage=s)
                    m.wait(y_free, stage=s)
                m.gm2ub(bufX[s], X, (my0 * T + t * T,))
                m.commit(x_rdy, stage=s)
                m.gm2ub(bufY[s], Y, (my0 * T + t * T,))
                m.commit(y_rdy, stage=s)

        with v:
            for t in m.tile_loop("t", cnt):
                s = t % S
                m.wait(x_rdy, stage=s)
                m.wait(y_rdy, stage=s)
                if t > 0:
                    m.wait(z_free, stage=0)
                if p.prec == "fp32":
                    m.v_cast(bufFX[s], bufX[s])
                    m.v_cast(bufFY[s], bufY[s])
                    m.v_binary("add", bufFZ[s], bufFX[s], bufFY[s])
                    m.v_cast(bufZ[0], bufFZ[s])
                else:
                    m.v_binary("add", bufZ[0], bufX[s], bufY[s])
                m.commit(x_free, stage=s)
                m.commit(y_free, stage=s)
                m.commit(z_rdy, stage=0)

        with st:
            for t in m.tile_loop("t", cnt):
                m.wait(z_rdy, stage=0)
                m.ub2gm(Z, (my0 * T + t * T,), bufZ[0])
                m.commit(z_free, stage=0)

    return kern


def _make_inputs(shape, seed):
    n = shape["N"]
    rng = random.Random(seed)
    x = dt.quantize_list(gen_uniform(rng, n), "bf16")
    y = dt.quantize_list(gen_uniform(rng, n), "bf16")
    return {"X": x, "Y": y, "Z": [0.0] * n}


def _oracle(shape, inputs):
    n = shape["N"]
    return {"Z": [a + b for a, b in zip(inputs["X"], inputs["Y"])]}


wl = register(Workload(
    name="vec_add",
    description="Z = X + Y (bf16), memory-bound; fp32 vs bf16 compute dial",
    shape={"N": 8192},
    domain=[{"N": v} for v in (1024, 2048, 4096, 8192, 16384)],
    io=[("X", "bf16", ("N",), "in"),
        ("Y", "bf16", ("N",), "in"),
        ("Z", "bf16", ("N",), "out")],
    tolerance=(2e-2, 1e-3),
    make_inputs=_make_inputs,
    oracle=_oracle,
    compare=default_compare((2e-2, 1e-3)),
    params_class=VecAddParams,
    default_params=lambda: VecAddParams(**NAIVE.__dict__),
    kernel_fn=_kern,
    block_dim=lambda shape, p: p.block_dim,
    domain_guard=lambda shape, p: shape["N"] % p.tile == 0,
))
