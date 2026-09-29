"""Calibrated analytic cost model (paper: "Performance modeling" + Table 1).

- `analytic_cycles` is the *predictor*: pure rate model (bandwidth/FLOPs/
  lanes) plus learned per-class overheads from Calibration. Used to rank
  and filter candidates before any simulation time.
- `measured_cycles` is the simulator's ground truth: same rates plus
  intrinsic per-op overheads (copy startup, flag latency, pipe drain).
  The systematic gap between the two is exactly what compiler evolution
  closes by calibrating (paper: "a systematic misprediction becomes a
  calibration target").

Per paper B.5, if the target has no calibration evidence the model says so
instead of silently inheriting anchors from another SKU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import dtypes as dt
from .arch import Arch
from .ir import (
    Commit,
    Gm2Ub,
    HardBarrier,
    L0c2Ub,
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

COPY = "copy"
MATMUL = "matmul"
VECTOR = "vector"
SYNC = "sync"

INTRINSIC_OVERHEAD = {COPY: 64.0, MATMUL: 24.0, VECTOR: 8.0, SYNC: 4.0}
_BARRIER_CYCLES = 32.0


def cost_class(op) -> str:
    if isinstance(op, (Gm2Ub, Ub2Gm, L0c2Ub)):
        return COPY
    if isinstance(op, Matmul):
        return MATMUL
    if isinstance(op, (VBinary, VUnary, VReduce, VArgmin, VTranspose, VBcast, VCast)):
        return VECTOR
    return SYNC


def _bytes_view(view, stage_count=1):
    return shape_numel(view.shape) * dt.bytes_of(view.dtype)


def work_cycles(op, arch: Arch) -> float:
    """Idealized work (no per-op overheads)."""
    if isinstance(op, Gm2Ub):
        return _bytes_view(op.dst) / arch.mte2_bytes_per_cycle
    if isinstance(op, Ub2Gm):
        return _bytes_view(op.src) / arch.mte3_bytes_per_cycle
    if isinstance(op, L0c2Ub):
        return _bytes_view(op.dst) / (arch.vec_lanes_per_cycle * 4)
    if isinstance(op, Matmul):
        m, k = op.a.shape
        _, n = op.b.shape
        return (m * k * n) / arch.cube_macs_per_cycle
    if isinstance(op, (VBinary, VUnary, VCast)):
        return shape_numel(op.dst.shape) / arch.vec_lanes_per_cycle
    if isinstance(op, (VReduce, VArgmin)):
        return shape_numel(op.x.shape) / arch.vec_lanes_per_cycle
    if isinstance(op, (VTranspose, VBcast)):
        return shape_numel(op.dst.shape) / arch.vec_lanes_per_cycle
    if isinstance(op, HardBarrier):
        return _BARRIER_CYCLES
    return 2.0  # commit / wait flag ops


def measured_cycles(op, arch: Arch) -> float:
    """Simulator ground truth: work + intrinsic per-op overhead."""
    return work_cycles(op, arch) + INTRINSIC_OVERHEAD[cost_class(op)]


def analytic_cycles(op, arch: Arch, cal: "Calibration") -> float:
    """Predictor: work + calibration multiplier + learned overhead."""
    cls = cost_class(op)
    mult, over = cal.for_arch(arch.name, cls)
    return work_cycles(op, arch) * mult + over


# --------------------------------------------------------------------------
# calibration


@dataclass
class Calibration:
    """Learned per-class multipliers/overheads, keyed by arch name."""

    per_arch: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def for_arch(self, arch_name: str, cls: str):
        a = self.per_arch.get(arch_name, {})
        return (a.get("multipliers", {}).get(cls, 1.0),
                a.get("overheads", {}).get(cls, 0.0))

    def update(self, arch_name: str, multipliers=None, overheads=None, note=""):
        a = self.per_arch.setdefault(arch_name, {"multipliers": {}, "overheads": {}})
        a["multipliers"].update(multipliers or {})
        a["overheads"].update(overheads or {})
        self.history.append({"arch": arch_name, "note": note,
                             "multipliers": dict(multipliers or {}),
                             "overheads": dict(overheads or {})})

    def has_arch(self, arch_name: str) -> bool:
        return arch_name in self.per_arch

    def to_json(self) -> dict:
        return {"per_arch": self.per_arch, "history": self.history}

    @classmethod
    def from_json(cls, data) -> "Calibration":
        data = data or {}
        return cls(per_arch=data.get("per_arch", {}), history=data.get("history", []))


# --------------------------------------------------------------------------
# reports


@dataclass
class CostReport:
    program_name: str
    arch_name: str
    span_cycles: float
    launch_cycles: float
    busy: dict                    # cost class -> cycles
    bottleneck: str
    attribution: dict             # class -> fraction of span
    coverage: str                 # "calibrated" | "default-anchors"
    hints: list = field(default_factory=list)

    @property
    def span_us(self) -> float:
        return self.span_cycles  # caller converts via arch seconds_per_cycle

    def format(self) -> str:
        lines = [
            f"cost: {self.program_name} @ {self.arch_name} "
            f"~{self.span_cycles:.0f} cyc (launch {self.launch_cycles:.0f}) "
            f"[{self.coverage}] bottleneck={self.bottleneck}",
        ]
        for cls, cyc in sorted(self.busy.items(), key=lambda kv: -kv[1]):
            frac = self.attribution.get(cls, 0.0)
            lines.append(f"  {cls:<8} {cyc:9.0f} cyc  ({frac * 100:4.1f}% of span)")
        for h in self.hints:
            lines.append(f"  hint: {h}")
        return "\n".join(lines)


def _bottleneck(span, busy):
    if span <= 0:
        return "unknown", {}
    attr = {cls: cyc / span for cls, cyc in busy.items()}
    agg = {COPY: 0.0, MATMUL: 0.0, VECTOR: 0.0, SYNC: 0.0}
    for cls, frac in attr.items():
        agg[cls] = agg.get(cls, 0.0) + frac
    top = max(agg, key=lambda c: agg[c])
    if agg[top] < 0.25:
        return "launch_overhead", agg
    return {
        COPY: "memory_bound",
        MATMUL: "cube_bound",
        VECTOR: "vector_bound",
        SYNC: "sync_bound",
    }[top], agg


def _hints(program, bottleneck, agg):
    hints = []
    if bottleneck == "memory_bound":
        hints.append("memory bound: raise arithmetic intensity (larger tiles) or add "
                     "pipeline stages to overlap MTE with compute")
    if bottleneck == "cube_bound":
        hints.append("cube bound: larger K per matmul step or split-K across cores "
                     "increases tensor-core utilization")
    if bottleneck == "vector_bound":
        hints.append("vector bound: fuse vector ops, widen tiles, or move work into "
                     "the cube path (GEMM formulation)")
    if bottleneck == "sync_bound":
        hints.append("sync bound: coarsen handoffs (fewer stages, larger tiles per event) "
                     "to amortize flag latency")
    if any(p.stages == 1 for p in program.pipelines) and program.events:
        hints.append("a pipeline is running with stages=1: no producer/consumer overlap")
    return hints


# --------------------------------------------------------------------------
# prediction


def predict(program: Program, arch: Arch, cal: Calibration | None = None) -> CostReport:
    """Analytic timing of one core program via the deterministic scheduler
    (timing-only, no numerics, analytic durations)."""
    from .interpreter import CoreInterpreter  # lazy: interpreter imports this module

    cal = cal or Calibration()
    interp = CoreInterpreter(
        program, arch, gm={}, gm_shapes={}, execute_values=False,
        duration_fn=lambda op: analytic_cycles(op, arch, cal))
    interp.run()
    span = max(interp.role_end.values()) if interp.role_end else 0.0
    launch = arch.launch_overhead_us * 1e-6 / arch.seconds_per_cycle()
    busy = {}
    for op, _s, _e in interp.exec_log:
        c = cost_class(op)
        busy[c] = busy.get(c, 0.0) + analytic_cycles(op, arch, cal)
    bn, agg = _bottleneck(span, busy)
    coverage = "calibrated" if (arch.calibrated and cal.has_arch(arch.name)) else \
        "default-anchors"
    return CostReport(
        program_name=program.name, arch_name=arch.name, span_cycles=span + launch,
        launch_cycles=launch, busy=busy, bottleneck=bn, attribution=agg,
        coverage=coverage, hints=_hints(program, bn, agg))


def predict_cores(programs, arch: Arch, cal: Calibration | None = None) -> CostReport:
    """Predict a multi-core candidate: span = slowest core (data-parallel launch)."""
    if not programs:
        raise ValueError("no programs")
    reports = [predict(p, arch, cal) for p in programs]
    span = max(r.span_cycles for r in reports)
    busy = {}
    for r in reports:
        for cls, cyc in r.busy.items():
            busy[cls] = busy.get(cls, 0.0) + cyc
    bn, agg = _bottleneck(span, busy)
    r0 = reports[0]
    return CostReport(
        program_name=r0.program_name, arch_name=r0.arch_name, span_cycles=span,
        launch_cycles=r0.launch_cycles, busy=busy, bottleneck=bn, attribution=agg,
        coverage=r0.coverage, hints=_hints(programs[0], bn, agg))


def misprediction(predicted: float, measured: float | None) -> float | None:
    """Signed relative error; None when unmeasurable."""
    if measured is None or measured == 0:
        return None
    return (predicted - measured) / measured
