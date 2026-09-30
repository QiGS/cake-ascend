"""Lowering: Cake-IR (Ascend flavor) -> AscendC source.

The paper's division of labor: the schedule states *what* happens; lowering
derives *how* — tiered buffer positions (UB/VECCALC vs L1/A1/A2 vs L0C/C1C2),
bufferNum depths, HardEvent ids, flag pairing, loop re-rolling from
LoopMarkers — all computed from declarations rather than written by the
agent (paper 2.2).

Emitted code targets the real AscendC surface (kernel_operator.h):
- GM_ADDR entry parameters + __gm__ typed pointer casts (operator style)
- using namespace AscendC; bfloat16_t/half_t/float_t/int32_t
- TPipe + TBuf<TPosition::VECCALC> (UB) / TCubeTBuf<TPosition::A1|A2>
  (cube operand staging) / TCubeTBuf<TPosition::C1C2> (L0C accumulator)
- SetFlag/WaitFlag with the real HardEvent enum members (cube queue = M)
  and EVENT_ID<n> per pipeline stage
- matmul::Matmul + SetTensorA/B/C + Iterate; CopyTensor for the L0C->UB
  epilogue; real elementwise vector primitives (Add/Sub/Mul/Min/Max/Muls/
  Adds/Exp/Abs/Cast)
Composite epilogue ops (reduce/argmin/transpose/broadcast) have no direct
primitive and are emitted with an explicit NOTE marker — the gemm and
vec_add paths are primitive-clean.

Compiling still requires a CANN toolchain + Ascend hardware, which this
machine lacks; the simulator remains the local execution authority.
"""
from __future__ import annotations

from . import dtypes as dt
from .arch import Arch
from .ir import (
    Commit,
    Gm2L1,
    Gm2Ub,
    HardBarrier,
    L0c2Ub,
    LoopMarker,
    Matmul,
    Program,
    Ub2Gm,
    VArgmin,
    VBcast,
    VBinary,
    VCast,
    VReduce,
    VTranspose,
    VUnary,
    Wait,
    shape_numel,
)

_ASCENDC_TYPE = {dt.BF16: "bfloat16_t", dt.FP16: "half_t", dt.FP32: "float_t",
                 dt.INT32: "int32_t"}

# Real HardEvent enum members (kernel_operator.h); the cube queue is "M".
# Same-queue pairs have no member: the queue is in-order, so no flag is
# needed and lowering emits a comment instead.
_HARD_EVENT = {
    ("MTE2", "MTE1"): "HardEvent::MTE2_MTE1",
    ("MTE1", "MTE2"): "HardEvent::MTE1_MTE2",
    ("MTE1", "V"): "HardEvent::MTE1_V",
    ("V", "MTE1"): "HardEvent::V_MTE1",
    ("MTE1", "CUBE"): "HardEvent::MTE1_M",
    ("CUBE", "MTE1"): "HardEvent::M_MTE1",
    ("MTE2", "V"): "HardEvent::MTE2_V",
    ("V", "MTE2"): "HardEvent::V_MTE2",
    ("MTE2", "CUBE"): "HardEvent::MTE2_M",
    ("CUBE", "MTE2"): "HardEvent::M_MTE2",
    ("MTE2", "MTE3"): "HardEvent::MTE2_MTE3",
    ("MTE3", "MTE2"): "HardEvent::MTE3_MTE2",
    ("MTE3", "V"): "HardEvent::MTE3_V",
    ("V", "MTE3"): "HardEvent::V_MTE3",
    ("MTE3", "CUBE"): "HardEvent::MTE3_M",
    ("CUBE", "MTE3"): "HardEvent::M_MTE3",
    ("CUBE", "V"): "HardEvent::M_V",
    ("V", "CUBE"): "HardEvent::V_M",
}

# real elementwise primitives (dst, src0, src1, count) / (dst, src, count)
_V_BINARY_FN = {"add": "Add", "sub": "Sub", "mul": "Mul", "min": "Min", "max": "Max"}
_V_UNARY_FN = {"abs": "Abs", "exp": "Exp", "sqrt": "Sqrt"}


class AscendCCodeGen:
    def __init__(self, programs, arch: Arch):
        self.programs = programs            # per-core programs
        self.arch = arch
        self.lines = []
        self.indent = 0
        self._event_ids = {}
        self._buf_ids = {}
        self._tmp = 0

    # ------------------------------------------------------------- helpers

    def w(self, text=""):
        self.lines.append("    " * self.indent + text if text else "")

    def comment(self, text):
        self.w(f"// {text}")

    def _event_id(self, event, stage):
        key = (event.name, stage)
        if key not in self._event_ids:
            pair = (event.prod.kind, event.cons.kind)
            he = _HARD_EVENT.get(pair)
            self._event_ids[key] = (f"EVT_{event.name.upper()}_{stage}", he, pair)
        return self._event_ids[key]

    def _local(self, view, stage, prefix="buf"):
        key = (view.name, stage)
        if key not in self._buf_ids:
            self._buf_ids[key] = f"{prefix}_{view.name}_{stage}"
        return self._buf_ids[key]

    def _tensor(self, view, stage):
        return f"{self._local(view, stage)}.GetTensor<{_ASCENDC_TYPE[view.dtype]}>()"

    # --------------------------------------------------------------- emit

    def generate(self) -> str:
        p0 = self.programs[0]
        self._header(p0)
        self.w(f'extern "C" __global__ __aicore__ void {p0.name}(')
        self.indent += 2
        for i, p in enumerate(p0.params):
            sep = "," if i < len(p0.params) - 1 else ""
            self.w(f"GM_ADDR {p.name}_gm{sep}  /* {p.shape} {p.dtype} */")
        self.indent -= 2
        self.w(") {")
        self.indent += 1
        for p in p0.params:
            t = _ASCENDC_TYPE[p.dtype]
            self.w(f"__gm__ {t}* __restrict__ {p.name} = "
                   f"(__gm__ {t}* __restrict__){p.name}_gm;")
        if len(self.programs) == 1:
            self._body(self.programs[0], core_guard=None)
        else:
            for core, prog in enumerate(self.programs):
                self._body(prog, core_guard=core)
        self.indent -= 1
        self.w("}")
        return "\n".join(self.lines) + "\n"

    def _header(self, prog: Program):
        self.comment("Generated by CAKE-Ascend lowering - inspectable by design.")
        self.comment(f"kernel: {prog.name}   target: {prog.arch_name}   "
                     f"block_dim: {prog.block_dim}")
        self.comment("Lowering derives: tiered buffer positions, bufferNum depths,")
        self.comment("HardEvent ids + EVENT_IDs, flag pairing and loop structure")
        self.comment("from the IR declarations.")
        self.w('#include "kernel_operator.h"')
        self.w('#include "lib/matmul_intf.h"  // matmul class; path varies by CANN version')
        self.w()
        self.w("using namespace AscendC;")
        self.w()

    # --------------------------------------------------------------- body

    def _body(self, prog: Program, core_guard):
        if core_guard is not None:
            self.w(f"if (GetBlockIdx() == {core_guard}) {{")
            self.indent += 1
        self._emit_pipeline(prog)
        self._emit_ops(prog)
        if core_guard is not None:
            self.indent -= 1
            self.w("}")

    def _emit_pipeline(self, prog: Program):
        self.comment(f"---- core program: roles "
                     f"{', '.join(f'{r.name}:{r.kind}' for r in prog.roles)} ----")
        # derived event declarations
        seen = set()
        for e in prog.events:
            for s in range(e.pipeline.stages if e.pipeline else 1):
                eid, he, pair = self._event_id(e, s)
                if eid in seen:
                    continue
                seen.add(eid)
                if he is not None:
                    self.w(f"constexpr auto {eid} = {he};")
                else:
                    self.comment(f"event '{e.name}' stage {s}: {pair[0]}->{pair[1]} "
                                 f"is same-queue; the queue is in-order, no flag needed")
        if seen:
            self.w()
        # derived buffer init: one TPipe per program; one buffer per staged slot.
        # Tier placement follows the pool the view lives in: L1 views are the
        # cube operand path (A1/A2), UB views feed the vector units (VECCALC);
        # the accumulator is L0C (C1C2).
        mm_a, mm_b = _matmul_operand_views(prog)
        self.w("TPipe pipe;")
        for view in prog.ub_views:
            bytes_ = view.slot_bytes
            bufnum = view.stages
            if view.pool.tier == "l1":
                pos = "A1" if view.name in mm_a else \
                    "A2" if view.name in mm_b else "A1"
                buf_cls = "TCubeTBuf"
            else:
                pos = "VECCALC"
                buf_cls = "TBuf"
            self.comment(f"view '{view.name}': offset {view.offset} B, "
                         f"{view.shape} x {view.dtype}, bufferNum={bufnum}, "
                         f"tier={view.pool.tier}, position={pos}")
            for s in range(view.stages):
                local = self._local(view, s)
                self.w(f"{buf_cls}<TPosition::{pos}> {local};")
                self.w(f"pipe.InitBuffer({local}, {bytes_});")
        for r in prog.l0c_regions:
            self.w(f"TCubeTBuf<TPosition::C1C2> l0c_{r.name}; "
                   f"/* {r.shape} fp32 accumulator, L0C */")
        self.w()

    def _emit_ops(self, prog: Program):
        # re-roll loops from markers; core_id-dependent constants already baked
        for op in prog.ops:
            if isinstance(op, LoopMarker):
                if op.is_begin:
                    var = op.var.replace("@", "_")
                    self.w(f"for (int {var} = 0; {var} < {op.count}; ++{var}) {{")
                    self.indent += 1
                else:
                    self.indent -= 1
                    self.w("}")
                continue
            if isinstance(op, (Gm2Ub, Gm2L1)):
                self._emit_gm_load(op)
            elif isinstance(op, Ub2Gm):
                self._emit_ub2gm(op)
            elif isinstance(op, L0c2Ub):
                self._emit_l0c2ub(op)
            elif isinstance(op, Matmul):
                self._emit_matmul(op)
            elif isinstance(op, VBinary):
                self._emit_vbinary(op)
            elif isinstance(op, VUnary):
                self._emit_vunary(op)
            elif isinstance(op, VCast):
                self._emit_vcast(op)
            elif isinstance(op, (VReduce, VArgmin, VTranspose, VBcast)):
                self._emit_composite(op)
            elif isinstance(op, Commit):
                self._emit_flag(op, is_commit=True)
            elif isinstance(op, Wait):
                self._emit_flag(op, is_commit=False)
            elif isinstance(op, HardBarrier):
                self.w("SyncAll();  /* hard barrier: block scope */")

    # -------------------------------------------------------- op emitters

    @staticmethod
    def _linear_offset(gm_off, param_shape) -> int:
        """Concrete row-major element offset: sum(off[d] * stride[d]).

        Both operands are literal ints at trace time, so the emitted address
        is exact and process-independent (deterministic by construction).
        """
        strides = [1] * len(param_shape)
        for d in range(len(param_shape) - 2, -1, -1):
            strides[d] = strides[d + 1] * param_shape[d + 1]
        return sum(off * st for off, st in zip(gm_off, strides))

    def _emit_gm_load(self, op):
        """DataCopy GM -> UB (vector tier) / GM -> L1 (cube operand tier)."""
        off = ", ".join(str(o) for o in op.gm_off)
        n = shape_numel(op.dst.shape)
        lin = self._linear_offset(op.gm_off, op.src.shape)
        tier = op.dst.pool.tier.upper()
        self.comment(f"gm2{tier.lower()} {op.src.name}[{off}] -> "
                     f"{op.dst.name}[{op.stage}] ({op.dst.shape})")
        self.w(f"DataCopy({self._tensor(op.dst, op.stage)}, "
               f"{op.src.name} + {lin}, {n});")

    def _emit_ub2gm(self, op: Ub2Gm):
        off = ", ".join(str(o) for o in op.gm_off)
        n = shape_numel(op.src.shape)
        lin = self._linear_offset(op.gm_off, op.dst.shape)
        self.comment(f"ub2gm {op.src.name}[{op.stage}] -> {op.dst.name}[{off}]")
        self.w(f"DataCopy({op.dst.name} + {lin}, "
               f"{self._tensor(op.src, op.stage)}, {n});")

    def _emit_l0c2ub(self, op: L0c2Ub):
        n = shape_numel(op.src.shape)
        self.comment(f"l0c2ub {op.src.name} -> {op.dst.name}[{op.stage}] "
                     f"({op.dst.shape})")
        self.w(f"CopyTensor({self._tensor(op.dst, op.stage)}, "
               f"l0c_{op.src.name}.GetTensor<float_t>(), {n});")

    def _emit_matmul(self, op: Matmul):
        m, k = op.a.shape
        _, n = op.b.shape
        at, bt = _ASCENDC_TYPE[op.a.dtype], _ASCENDC_TYPE[op.b.dtype]
        self.comment(f"matmul {op.a.name}[{op.a_stage}]x{op.b.name}[{op.b_stage}] "
                     f"-> {op.acc.name} ({m}x{k}x{n}, clear={op.clear})")
        self.w("{")
        self.indent += 1
        self.w("matmul::Matmul<{at}, {bt}, float_t, matmul::MatmulFormat::ND> mm;"
               .format(at=at, bt=bt))
        self.w(f"mm.SetTensorA({self._local(op.a, op.a_stage)}.GetTensor<{at}>());")
        self.w(f"mm.SetTensorB({self._local(op.b, op.b_stage)}.GetTensor<{bt}>());")
        self.w(f"mm.SetTensorC(l0c_{op.acc.name}.GetTensor<float_t>());")
        self.w("mm.Iterate();")
        self.indent -= 1
        self.w("}")

    def _emit_vbinary(self, op: VBinary):
        dst = self._tensor(op.dst, op.dst_stage)
        x = self._tensor(op.x, op.x_stage)
        n = shape_numel(op.dst.shape)
        if op.y is not None:
            y = self._tensor(op.y, op.y_stage)
            fn = _V_BINARY_FN[op.op]
            self.comment(f"{op.op}: {op.x.name} op {op.y.name} -> {op.dst.name}")
            self.w(f"{fn}({dst}, {x}, {y}, {n});")
        else:
            self.comment(f"{op.op}: {op.x.name} * {op.y_scalar} -> {op.dst.name}")
            self.w(f"Muls({dst}, {x}, {op.y_scalar}, {n});")

    def _emit_vunary(self, op: VUnary):
        dst = self._tensor(op.dst, op.dst_stage)
        x = self._tensor(op.x, op.x_stage)
        n = shape_numel(op.dst.shape)
        if op.op == "neg":
            self.comment(f"neg: {op.x.name} -> {op.dst.name}")
            self.w(f"Muls({dst}, {x}, -1, {n});")
        elif op.op == "copy":
            self.comment(f"copy: {op.x.name} -> {op.dst.name}")
            self.w(f"Adds({dst}, {x}, 0, {n});")
        else:
            fn = _V_UNARY_FN[op.op]
            self.w(f"{fn}({dst}, {x}, {n});  /* {op.op} */")

    def _emit_vcast(self, op: VCast):
        n = shape_numel(op.dst.shape)
        self.comment(f"cast {op.x.dtype}->{op.dst.dtype}: "
                     f"{op.x.name} -> {op.dst.name}")
        self.w(f"Cast({self._tensor(op.dst, op.dst_stage)}, "
               f"{self._tensor(op.x, op.x_stage)}, {n}, RoundMode::CAST_NONE);")

    def _emit_composite(self, op):
        """Composite epilogue ops without a single real primitive."""
        kind = op.kind
        detail = {
            "VReduce": f"reduce {op.op} axis=-1: {op.x.name} -> {op.dst.name}",
            "VArgmin": f"argmin axis=-1: {op.x.name} -> {op.dst.name}",
            "VTranspose": f"transpose: {op.x.name} -> {op.dst.name}",
            "VBcast": f"bcast {op.mode}: {op.x.name} -> {op.dst.name}",
        }[kind]
        self.comment(f"NOTE(composite): {detail}")
        self.comment("no direct AscendC primitive: lower via vector primitives "
                     "or a custom epilogue")

    def _emit_flag(self, op, is_commit: bool):
        eid, he, pair = self._event_id(op.event, op.stage)
        verb = "SetFlag" if is_commit else "WaitFlag"
        what = "commit" if is_commit else "wait"
        if he is None:
            self.comment(f"{what} '{op.event.name}' stage {op.stage}: same-queue "
                         f"({pair[0]}->{pair[1]}), ordering already guaranteed")
            return
        self.w(f"{verb}<{he}>(EVENT_ID{op.stage});  /* {what} '{op.event.name}' "
               f"stage {op.stage} */")


def _matmul_operand_views(prog: Program):
    """Views consumed as matmul A/B operands: (a-view names, b-view names).

    The IR keeps operands in L1-tier views; lowering places the A operand at
    A1 and the B operand at A2 (L1 -> L0A/L0B staging managed by the matmul
    module).
    """
    a_names, b_names = set(), set()
    for op in prog.ops:
        if isinstance(op, Matmul):
            a_names.add(op.a.name)
            b_names.add(op.b.name)
    return a_names, b_names


def generate_ascendc(programs, arch: Arch) -> str:
    if not programs:
        raise ValueError("no programs to lower")
    return AscendCCodeGen(programs, arch).generate()
