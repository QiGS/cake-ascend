"""Tracing builder: the `m` context that schedule code runs against.

Style mirrors the paper's Fig. 3 (`lm.*`) adapted to Ascend resources:

    def kern(m):
        A = m.gm_param("A", "bf16", (128, 256))
        ub = m.ub_pool("ub", 232*1024)
        bufA = ub.view("A", offset=0, shape=(64, 64), dtype="bf16", stages=2)
        acc = m.l0c("acc", shape=(64, 64))
        ld = m.role("ld", "MTE2")
        cu = m.role("cu", "CUBE")
        pipe = m.pipeline("main", stages=2)
        rdy = m.event("a_rdy", prod=ld, cons=cu, pipeline=pipe)
        with ld:
            for kt in m.tile_loop("kt", 4):
                m.gm2ub(bufA[kt % 2], A, (0, kt*64))
                m.commit(rdy, stage=kt % 2)
        with cu:
            for kt in m.tile_loop("kt", 4):
                m.wait(rdy, stage=kt % 2)
                m.matmul(acc, bufA[kt % 2], bufB[kt % 2], clear=(kt == 0))

Loops unroll at trace time (counts are Python ints), so every op in the
resulting Program is concrete — statically analyzable per paper P4/P5 —
while LoopMarker ops retain structure for codegen.
"""
from __future__ import annotations

from . import dtypes as dt
from .arch import get_arch
from .ir import (
    Commit,
    Event,
    Gm2L1,
    Gm2Ub,
    GmParam,
    HardBarrier,
    IRConstructionError,
    L0c2Ub,
    L0cRegion,
    LoopMarker,
    Matmul,
    Op,
    Pipeline,
    Program,
    Role,
    Slot,
    Ub2Gm,
    UbPool,
    UbView,
    VArgmin,
    VBcast,
    VBinary,
    VCast,
    VReduce,
    VTranspose,
    VUnary,
    Wait,
    VALID_ROLE_KINDS,
    check_matmul,
    check_same,
    shape_numel,
    V_BINARY_OPS,
    V_REDUCE_OPS,
    V_UNARY_OPS,
)

__all__ = ["KernelContext", "schedule", "build", "build_all_cores"]


def _slot(x) -> Slot:
    if isinstance(x, Slot):
        return x
    if isinstance(x, UbView):
        return Slot(x, 0)
    raise IRConstructionError(
        f"expected a buffer slot like buf[stage], got {type(x).__name__}",
        region="operand",
        hint="address UB tensors as buf[stage] (stage index selects the pipeline slot)",
    )


class KernelContext:
    """The `m` object: declares resources, records ops, type-checks on the fly."""

    def __init__(self, program: Program, core_id: int, core_count: int):
        self.prog = program
        self.core_id_ = core_id
        self.core_count_ = core_count
        self._role = None
        self._loop_stack = []      # list[(var, value)] for op tags
        self._loop_depth = 0

    # ---------------------------------------------------------------- misc

    def core_id(self) -> int:
        return self.core_id_

    def core_count(self) -> int:
        return self.core_count_

    @staticmethod
    def num_tiles(total: int, tile: int) -> int:
        if tile <= 0:
            raise IRConstructionError(f"tile size must be > 0, got {tile}", region="tiling")
        return -(-int(total) // int(tile))

    @staticmethod
    def tdiv(a: int, b: int) -> int:
        return int(a) // int(b)

    # ------------------------------------------------------------ resources

    def gm_param(self, name: str, dtype: str, shape, desc: str = "") -> GmParam:
        if not dt.is_dtype(dtype):
            raise IRConstructionError(f"unknown dtype {dtype!r}", region=f"param:{name}")
        if any(p.name == name for p in self.prog.params):
            raise IRConstructionError(f"duplicate GM param {name!r}", region=f"param:{name}")
        p = GmParam(name, dtype, tuple(int(d) for d in shape), desc)
        self.prog.params.append(p)
        return p

    def ub_pool(self, name: str, size: int) -> UbPool:
        """Unified Buffer pool (vector path, VECCALC)."""
        return self._mem_pool(name, size, tier="ub")

    def l1_pool(self, name: str, size: int) -> UbPool:
        """L1 / local buffer pool (cube operand path, A1/A2 -> L0A/L0B)."""
        return self._mem_pool(name, size, tier="l1")

    def _mem_pool(self, name: str, size: int, tier: str) -> UbPool:
        if size <= 0:
            raise IRConstructionError(
                f"{tier} pool size must be positive, got {size}",
                region=f"{tier}:{name}")
        pool = UbPool(name, int(size), tier=tier)
        pool._ctx = self
        self.prog.ub_pools.append(pool)
        return pool

    def _declare_view(self, pool: UbPool, name: str, offset: int, shape, dtype: str,
                      stages: int) -> UbView:
        if not dt.is_dtype(dtype):
            raise IRConstructionError(f"unknown dtype {dtype!r}", region=f"view:{name}")
        if stages < 1:
            raise IRConstructionError(
                f"view '{name}' needs stages >= 1, got {stages}", region=f"view:{name}")
        if any(v.name == name for v in self.prog.ub_views):
            raise IRConstructionError(f"duplicate view {name!r}", region=f"view:{name}")
        v = UbView(name, pool, int(offset), tuple(int(d) for d in shape), dtype, int(stages))
        self.prog.ub_views.append(v)
        return v
    def l0c(self, name: str, shape, dtype: str = dt.FP32) -> L0cRegion:
        if dtype not in (dt.FP32,):
            raise IRConstructionError(
                f"l0c region '{name}' must accumulate fp32, got {dtype}",
                region=f"l0c:{name}",
                hint="cube accumulators are fp32 on this target",
            )
        r = L0cRegion(name, tuple(int(d) for d in shape), dtype)
        self.prog.l0c_regions.append(r)
        return r

    def role(self, name: str, kind: str) -> Role:
        if kind not in VALID_ROLE_KINDS:
            raise IRConstructionError(
                f"role '{name}' has invalid kind {kind!r} (valid: {VALID_ROLE_KINDS})",
                region=f"role:{name}",
                hint="roles are execution agents: MTE2 (GM->UB), MTE3 (UB->GM), CUBE (matmul), V (vector), SCALAR",
            )
        r = Role(name, kind)
        if any(x.name == name for x in self.prog.roles):
            raise IRConstructionError(f"duplicate role {name!r}", region=f"role:{name}")
        r._ctx = self
        self.prog.roles.append(r)
        return r

    def pipeline(self, name: str, stages: int) -> Pipeline:
        if stages < 1:
            raise IRConstructionError(f"pipeline '{name}' needs stages >= 1", region=f"pipe:{name}")
        p = Pipeline(name, int(stages))
        self.prog.pipelines.append(p)
        return p

    def event(self, name: str, prod: Role, cons: Role, pipeline: Pipeline | None = None) -> "EventRef":
        from .ir import Event
        if prod not in self.prog.roles or cons not in self.prog.roles:
            raise IRConstructionError(
                f"event '{name}' references undeclared role(s)",
                region=f"event:{name}",
                hint="declare roles with m.role(...) before events",
            )
        e = Event(name, prod, cons, pipeline)
        if any(x.name == name for x in self.prog.events):
            raise IRConstructionError(f"duplicate event {name!r}", region=f"event:{name}")
        self.prog.events.append(e)
        return EventRef(self, e)

    # ------------------------------------------------------------- control

    def tile_loop(self, var: str, count: int):
        return _Loop(self, var, int(count), kind="tile")

    def stages(self, pipe: Pipeline):
        return _Loop(self, f"stage@{pipe.name}", pipe.stages, kind="stage")

    def _current_loop_tags(self) -> dict:
        return {var: val for var, val in self._loop_stack}

    # ------------------------------------------------------------ memory ops

    def _need_role(self, what: str):
        if self._role is None:
            raise IRConstructionError(
                f"{what} issued outside any role block",
                region=what,
                hint="wrap ops in `with m.role(...)` blocks",
            )
        return self._role

    @staticmethod
    def _check_tier(view: UbView, tier: str, what: str):
        if view.pool.tier != tier:
            raise IRConstructionError(
                f"{what}: view '{view.name}' lives in a {view.pool.tier.upper()} pool; "
                f"this operation consumes {tier.upper()} tier",
                region=f"view:{view.name}",
                hint={"ub": "vector ops read/write Unified Buffer (m.ub_pool)",
                      "l1": "cube operands stage through L1 (m.l1_pool): "
                            "GM -> L1 -> L0A/L0B -> L0C"}[tier])

    def gm2ub(self, dst, src: GmParam, gm_off):
        return self._gm_copy(dst, src, gm_off, "Gm2Ub", want_tier="ub")

    def gm2l1(self, dst, src: GmParam, gm_off):
        """DataCopy GM -> L1, feeding the cube operand path."""
        return self._gm_copy(dst, src, gm_off, "Gm2L1", want_tier="l1")

    def _gm_copy(self, dst, src: GmParam, gm_off, kind: str, want_tier: str):
        role = self._need_role(kind.lower())
        d = _slot(dst)
        if not isinstance(src, GmParam):
            raise IRConstructionError(f"{kind.lower()} src must be a GM param",
                                      region=kind.lower())
        off = tuple(int(o) for o in gm_off)
        if len(off) != len(d.view.shape) or len(off) != len(src.shape):
            raise IRConstructionError(
                f"{kind.lower()} offset arity {len(off)} does not match "
                f"src {src.shape} / view {d.view.shape}",
                region=f"{kind.lower()}:{src.name}->{d.view.name}",
                hint="provide one element offset per dimension",
            )
        self._check_tier(d.view, want_tier, kind)
        if d.view.dtype != src.dtype:
            raise IRConstructionError(
                f"{kind.lower()} dtype mismatch: {src.name} is {src.dtype}, "
                f"view '{d.view.name}' is {d.view.dtype}",
                region=f"{kind.lower()}:{src.name}->{d.view.name}",
                hint="DataCopy performs no conversion — load into a same-dtype view, then v_cast",
            )
        if len(d.view.shape) != len(src.shape):
            raise IRConstructionError(
                f"{kind.lower()} rank mismatch: view '{d.view.name}' {d.view.shape} "
                f"vs param {src.shape}",
                region=f"{kind.lower()}:{src.name}->{d.view.name}",
            )
        op_cls = Gm2Ub if kind == "Gm2Ub" else Gm2L1
        self._emit(op_cls(kind, role, dst=d.view, stage=d.stage, src=src, gm_off=off))

    def ub2gm(self, dst: GmParam, gm_off, src):
        role = self._need_role("ub2gm")
        s = _slot(src)
        if not isinstance(dst, GmParam):
            raise IRConstructionError("ub2gm dst must be a GM param", region="ub2gm")
        self._check_tier(s.view, "ub", "ub2gm")
        off = tuple(int(o) for o in gm_off)
        if len(off) != len(s.view.shape) or len(off) != len(dst.shape):
            raise IRConstructionError(
                f"ub2gm offset arity {len(off)} does not match dst {dst.shape} / view {s.view.shape}",
                region=f"ub2gm:{s.view.name}->{dst.name}",
            )
        if s.view.dtype != dst.dtype:
            raise IRConstructionError(
                f"ub2gm dtype mismatch: view '{s.view.name}' is {s.view.dtype}, param {dst.name} is {dst.dtype}",
                region=f"ub2gm:{s.view.name}->{dst.name}",
                hint="insert v_cast into a same-dtype view before storing",
            )
        self._emit(Ub2Gm("Ub2Gm", role, dst=dst, gm_off=off, src=s.view, stage=s.stage))

    def l0c2ub(self, dst, src: L0cRegion):
        role = self._need_role("l0c2ub")
        d = _slot(dst)
        if not isinstance(src, L0cRegion):
            raise IRConstructionError("l0c2ub src must be an l0c region", region="l0c2ub")
        self._check_tier(d.view, "ub", "l0c2ub")
        check_same(d.view.shape, src.shape, f"l0c2ub:{src.name}->{d.view.name}")
        self._emit(L0c2Ub("L0c2Ub", role, dst=d.view, stage=d.stage, src=src))

    # ----------------------------------------------------------- compute ops

    def matmul(self, acc: L0cRegion, a, b, clear: bool):
        role = self._need_role("matmul")
        sa, sb = _slot(a), _slot(b)
        if not isinstance(acc, L0cRegion):
            raise IRConstructionError("matmul acc must be an l0c region", region="matmul")
        # hardware contract: cube operands stage through L1 (A1/A2 -> L0A/L0B),
        # not UB — the vector path and cube path read different memory tiers
        self._check_tier(sa.view, "l1", "matmul")
        self._check_tier(sb.view, "l1", "matmul")
        check_matmul(sa.view.shape, sb.view.shape, acc.shape)
        for name, s in (("a", sa), ("b", sb)):
            if s.view.dtype not in (dt.BF16, dt.FP16):
                raise IRConstructionError(
                    f"matmul operand '{s.view.name}' must be bf16/fp16, got {s.view.dtype}",
                    region=f"matmul:{s.view.name}",
                    hint="cube consumes low-precision operands with fp32 accumulate",
                )
        self._emit(Matmul("Matmul", role, acc=acc, a=sa.view, a_stage=sa.stage,
                          b=sb.view, b_stage=sb.stage, clear=bool(clear)))

    def _vdst(self, op_name: str, dst, x_slots):
        role = self._need_role(op_name)
        d = _slot(dst)
        xs = [_slot(x) for x in x_slots]
        # vector units read/write Unified Buffer only (L1 is cube-only)
        self._check_tier(d.view, "ub", op_name)
        for s in xs:
            self._check_tier(s.view, "ub", op_name)
        return role, d, xs

    def v_binary(self, op: str, dst, x, y):
        role, d, xs = self._vdst("v_binary", dst, [x])
        sx = xs[0]
        if op not in V_BINARY_OPS:
            raise IRConstructionError(f"unknown binary op {op!r}", region="v_binary")
        if isinstance(y, (int, float)):
            ys, yv, yst, ysc = None, None, 0, float(y)
        else:
            sy = _slot(y)
            check_same(sx.view.shape, sy.view.shape, f"v_binary:{op}")
            ys, yv, yst, ysc = sy.view, sy.view, sy.stage, None
        check_same(d.view.shape, sx.view.shape, f"v_binary:{op}:dst")
        self._emit(VBinary("VBinary", role, op=op, dst=d.view, dst_stage=d.stage,
                           x=sx.view, x_stage=sx.stage, y=ys, y_stage=yst, y_scalar=ysc))

    def v_unary(self, op: str, dst, x):
        role, d, xs = self._vdst("v_unary", dst, [x])
        sx = xs[0]
        if op not in V_UNARY_OPS:
            raise IRConstructionError(f"unknown unary op {op!r}", region="v_unary")
        check_same(d.view.shape, sx.view.shape, f"v_unary:{op}:dst")
        self._emit(VUnary("VUnary", role, op=op, dst=d.view, dst_stage=d.stage,
                          x=sx.view, x_stage=sx.stage))

    def v_reduce(self, op: str, dst, x):
        role, d, xs = self._vdst("v_reduce", dst, [x])
        sx = xs[0]
        if op not in V_REDUCE_OPS:
            raise IRConstructionError(f"unknown reduce op {op!r}", region="v_reduce")
        if len(sx.view.shape) != 2 or len(d.view.shape) != 1:
            raise IRConstructionError(
                f"v_reduce reduces (R,C) -> (R,); got {sx.view.shape} -> {d.view.shape}",
                region="v_reduce",
            )
        if d.view.shape[0] != sx.view.shape[0]:
            raise IRConstructionError(
                f"v_reduce row count mismatch {d.view.shape} vs {sx.view.shape}", region="v_reduce")
        self._emit(VReduce("VReduce", role, op=op, dst=d.view, dst_stage=d.stage,
                           x=sx.view, x_stage=sx.stage))

    def v_argmin(self, dst, x):
        role, d, xs = self._vdst("v_argmin", dst, [x])
        sx = xs[0]
        if len(sx.view.shape) != 2 or len(d.view.shape) != 1:
            raise IRConstructionError(
                f"v_argmin reduces (R,C) -> (R,); got {sx.view.shape} -> {d.view.shape}",
                region="v_argmin",
            )
        if d.view.dtype != dt.INT32:
            raise IRConstructionError(
                f"v_argmin dst must be int32, got {d.view.dtype}", region="v_argmin")
        if d.view.shape[0] != sx.view.shape[0]:
            raise IRConstructionError(
                f"v_argmin row count mismatch {d.view.shape} vs {sx.view.shape}", region="v_argmin")
        self._emit(VArgmin("VArgmin", role, dst=d.view, dst_stage=d.stage,
                           x=sx.view, x_stage=sx.stage))

    def v_transpose(self, dst, x):
        role, d, xs = self._vdst("v_transpose", dst, [x])
        sx = xs[0]
        if len(sx.view.shape) != 2 or len(d.view.shape) != 2:
            raise IRConstructionError("v_transpose operates on 2D tiles", region="v_transpose")
        if d.view.shape != sx.view.shape[::-1]:
            raise IRConstructionError(
                f"v_transpose {sx.view.shape} -> {d.view.shape} (expected {sx.view.shape[::-1]})",
                region="v_transpose")
        self._emit(VTranspose("VTranspose", role, dst=d.view, dst_stage=d.stage,
                              x=sx.view, x_stage=sx.stage))

    def v_bcast(self, mode: str, dst, x):
        role, d, xs = self._vdst("v_bcast", dst, [x])
        sx = xs[0]
        if mode not in ("row", "col"):
            raise IRConstructionError(f"v_bcast mode must be row|col, got {mode!r}", region="v_bcast")
        if len(d.view.shape) != 2 or len(sx.view.shape) != 1:
            raise IRConstructionError("v_bcast broadcasts (R,)->(R,C) or (C,)->(R,C)", region="v_bcast")
        r, c = d.view.shape
        if mode == "row" and sx.view.shape[0] != r:
            raise IRConstructionError(f"v_bcast row: src {sx.view.shape} vs dst {d.view.shape}", region="v_bcast")
        if mode == "col" and sx.view.shape[0] != c:
            raise IRConstructionError(f"v_bcast col: src {sx.view.shape} vs dst {d.view.shape}", region="v_bcast")
        self._emit(VBcast("VBcast", role, mode=mode, dst=d.view, dst_stage=d.stage,
                          x=sx.view, x_stage=sx.stage))

    def v_cast(self, dst, x):
        role, d, xs = self._vdst("v_cast", dst, [x])
        sx = xs[0]
        check_same(d.view.shape, sx.view.shape, "v_cast:dst")
        if d.view.dtype == sx.view.dtype:
            raise IRConstructionError(
                "v_cast between identical dtypes is a no-op (use v_unary copy)",
                region="v_cast",
            )
        self._emit(VCast("VCast", role, dst=d.view, dst_stage=d.stage,
                         x=sx.view, x_stage=sx.stage))

    # -------------------------------------------------------------- sync ops

    def commit(self, event, stage: int = 0):
        role = self._need_role("commit")
        ev = _event_of(event)
        if role != ev.prod:
            raise IRConstructionError(
                f"commit('{ev.name}') issued in role '{role.name}' but producer is '{ev.prod.name}'",
                region=f"event:{ev.name}",
                hint="only the producing role may SetFlag an event",
            )
        _check_stage(ev, stage, "commit")
        self._emit(Commit("Commit", role, event=ev, stage=int(stage)))

    def wait(self, event, stage: int = 0):
        role = self._need_role("wait")
        ev = _event_of(event)
        if role != ev.cons:
            raise IRConstructionError(
                f"wait('{ev.name}') issued in role '{role.name}' but consumer is '{ev.cons.name}'",
                region=f"event:{ev.name}",
                hint="only the consuming role may WaitFlag an event",
            )
        _check_stage(ev, stage, "wait")
        self._emit(Wait("Wait", role, event=ev, stage=int(stage)))

    def hard_barrier(self, scope: str = "block"):
        role = self._need_role("hard_barrier")
        if scope not in ("block",):
            raise IRConstructionError(f"unsupported barrier scope {scope!r}", region="hard_barrier")
        self._emit(HardBarrier("HardBarrier", role, scope=scope))

    # ------------------------------------------------------------- plumbing

    def _emit(self, op: Op) -> Op:
        op.id = len(self.prog.ops)
        op.tags = self._current_loop_tags()
        self.prog.ops.append(op)
        return op


def _event_of(x):
    if isinstance(x, EventRef):
        return x.event
    from .ir import Event
    if isinstance(x, Event):
        return x
    raise IRConstructionError(f"expected an event reference, got {type(x).__name__}", region="event")


def _check_stage(ev, stage: int, what: str):
    if ev.pipeline is not None:
        if not (0 <= stage < ev.pipeline.stages):
            raise IRConstructionError(
                f"{what}('{ev.name}', stage={stage}): pipeline '{ev.pipeline.name}' has {ev.pipeline.stages} stage(s)",
                region=f"event:{ev.name}",
                hint=f"use stage in [0, {ev.pipeline.stages})",
            )
    elif stage != 0:
        raise IRConstructionError(
            f"{what}('{ev.name}', stage={stage}): event has no pipeline; only stage 0 exists",
            region=f"event:{ev.name}",
            hint="attach a pipeline to the event for staged handoffs",
        )


class EventRef:
    """Thin wrapper so commit()/wait() accept the event object."""

    def __init__(self, ctx: KernelContext, event):
        self._ctx = ctx
        self.event = event

    @property
    def name(self):
        return self.event.name


class _Loop:
    """A traced, unrolled loop."""

    def __init__(self, ctx: KernelContext, var: str, count: int, kind: str):
        self.ctx = ctx
        self.var = var
        self.count = count
        self.kind = kind

    def __iter__(self):
        ctx = self.ctx
        ctx._emit(LoopMarker("Loop", ctx._role or Role("", "SCALAR"),
                             var=self.var, count=self.count,
                             is_begin=True, depth=ctx._loop_depth))
        ctx._loop_depth += 1
        for i in range(self.count):
            ctx._loop_stack.append((self.var, i))
            try:
                yield i
            finally:
                ctx._loop_stack.pop()
        ctx._loop_depth -= 1
        ctx._emit(LoopMarker("Loop", ctx._role or Role("", "SCALAR"),
                             var=self.var, count=self.count,
                             is_begin=False, depth=ctx._loop_depth))


# --------------------------------------------------------------------------
# build entry points


def build(fn, *, name: str, block_dim: int, target: str = "ascend910b",
          core_id: int = 0, provenance: dict | None = None) -> Program:
    """Trace `fn(m)` into a Program for one core."""
    get_arch(target)  # validates target name early (paper: exact match)
    if block_dim < 1:
        raise IRConstructionError(f"block_dim must be >= 1, got {block_dim}", region="block_dim")
    prog = Program(name=name, arch_name=target, params=[], block_dim=int(block_dim))
    prog.provenance = dict(provenance or {})
    ctx = KernelContext(prog, core_id=core_id, core_count=int(block_dim))
    fn(ctx)
    return prog


def build_all_cores(fn, *, name: str, block_dim: int, target: str = "ascend910b",
                    provenance: dict | None = None) -> list:
    """Trace one Program per core (core_id() differs; ops are concrete)."""
    return [
        build(fn, name=name, block_dim=block_dim, target=target,
              core_id=c, provenance=provenance)
        for c in range(block_dim)
    ]


class schedule:
    """Decorator for LLM/hand-authored kernels (paper Fig. 3 style).

        @asc.schedule(name="gemm", block_dim=4)
        def kern(m): ...

        prog = asc.build(kern)           # core 0
        progs = asc.build_all_cores(kern)
    """

    def __init__(self, name: str, block_dim: int = 1, target: str = "ascend910b"):
        self.name = name
        self.block_dim = block_dim
        self.target = target
        self.fn = None

    def __call__(self, fn):
        self.fn = fn
        fn.build = lambda core_id=0, provenance=None: build(
            fn, name=self.name, block_dim=self.block_dim, target=self.target,
            core_id=core_id, provenance=provenance)
        fn.build_all_cores = lambda provenance=None: build_all_cores(
            fn, name=self.name, block_dim=self.block_dim, target=self.target,
            provenance=provenance)
        return fn
