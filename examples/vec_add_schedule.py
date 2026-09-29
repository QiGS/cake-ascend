"""Hand-authored example: fused bf16 vec-add schedule for CAKE-Ascend.

Try it with:
    python -m cakeasc verify examples/vec_add_schedule.py
    python -m cakeasc codegen examples/vec_add_schedule.py -o out/kernel.cpp
"""
from cakeasc import builder as asc

N = 8192
TILE = 1024
STAGES = 2


@asc.schedule(name="vec_add_manual", block_dim=2)
def kern(m):
    X = m.gm_param("X", "bf16", (N,))
    Y = m.gm_param("Y", "bf16", (N,))
    Z = m.gm_param("Z", "bf16", (N,))

    ub = m.ub_pool("ub", 232 * 1024)

    def align32(x):
        return (x + 31) // 32 * 32

    off = 0
    x_off = off; off += align32(TILE * 2 * STAGES)
    y_off = off; off += align32(TILE * 2 * STAGES)
    z_off = off; off += align32(TILE * 2)

    bufX = ub.view("X", x_off, (TILE,), "bf16", STAGES)
    bufY = ub.view("Y", y_off, (TILE,), "bf16", STAGES)
    bufZ = ub.view("Z", z_off, (TILE,), "bf16", 1)

    ld = m.role("ld", "MTE2")
    v = m.role("v", "V")
    st = m.role("st", "MTE3")

    pipe = m.pipeline("main", STAGES)
    x_rdy = m.event("x_rdy", ld, v, pipe)
    y_rdy = m.event("y_rdy", ld, v, pipe)
    x_free = m.event("x_free", v, ld, pipe)
    y_free = m.event("y_free", v, ld, pipe)
    z_rdy = m.event("z_rdy", v, st)
    z_free = m.event("z_free", st, v)

    total = m.num_tiles(N, TILE)
    per = m.num_tiles(total, m.core_count())
    my0 = m.core_id() * per
    cnt = max(0, min(per, total - my0))

    with ld:
        for t in m.tile_loop("t", cnt):
            s = t % STAGES
            if t >= STAGES:
                m.wait(x_free, stage=s)
                m.wait(y_free, stage=s)
            m.gm2ub(bufX[s], X, (my0 * TILE + t * TILE,))
            m.commit(x_rdy, stage=s)
            m.gm2ub(bufY[s], Y, (my0 * TILE + t * TILE,))
            m.commit(y_rdy, stage=s)

    with v:
        for t in m.tile_loop("t", cnt):
            s = t % STAGES
            m.wait(x_rdy, stage=s)
            m.wait(y_rdy, stage=s)
            if t > 0:
                m.wait(z_free, stage=0)
            m.v_binary("add", bufZ[0], bufX[s], bufY[s])
            m.commit(x_free, stage=s)
            m.commit(y_free, stage=s)
            m.commit(z_rdy, stage=0)

    with st:
        for t in m.tile_loop("t", cnt):
            m.wait(z_rdy, stage=0)
            m.ub2gm(Z, (my0 * TILE + t * TILE,), bufZ[0])
            m.commit(z_free, stage=0)
