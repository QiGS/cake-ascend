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
    Gm2L1,
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
    if isinstance(op, (Gm2Ub, Gm2L1, Ub2Gm, L0c2Ub)):
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
    if isinstance(op, (Gm2Ub, Gm2L1)):
        # GM->UB and GM->L1 both stream on the MTE2 path; the matmul op's
        # cost additionally covers the L1->L0A/L0B staging managed by the
        # cube module (implicitly, as in the AscendC Matmul API)
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
    busy: dict                    # cost class -> summed op cycles (roles may overlap)
    bottleneck: str
    attribution: dict             # class -> fraction of span active (interval union, <= 1)
    coverage: str                 # "learned-calibration" | "default-anchors" | "hardware-anchored+learned"
    hints: list = field(default_factory=list)

    def format(self) -> str:
        lines = [
            f"cost: {self.program_name} @ {self.arch_name} "
            f"~{self.span_cycles:.0f} cyc (launch {self.launch_cycles:.0f}) "
            f"[{self.coverage}] bottleneck={self.bottleneck}",
        ]
        for cls, cyc in sorted(self.busy.items(), key=lambda kv: -kv[1]):
            frac = self.attribution.get(cls, 0.0)
            lines.append(f"  {cls:<8} {cyc:9.0f} cyc busy ({frac * 100:4.1f}% of span active)")
        for h in self.hints:
            lines.append(f"  hint: {h}")
        return "\n".join(lines)


def _utilization(exec_log, span) -> dict:
    """Per-class fraction of the span during which that class is executing.

    Computed as a union of execution intervals, so concurrent roles of the
    same class do not double-count: every fraction is <= 1. Classes may sum
    above 1 across categories (they overlap by design in a pipelined
    schedule) — that is utilization, not exclusive time share.
    """
    per_class = {}
    for op, s, e in exec_log:
        per_class.setdefault(cost_class(op), []).append((s, e))
    out = {}
    for cls, ivs in per_class.items():
        ivs.sort()
        total, cs, ce = 0.0, None, None
        for s, e in ivs:
            if cs is None:
                cs, ce = s, e
            elif s <= ce:
                ce = max(ce, e)
            else:
                total += ce - cs
                cs, ce = s, e
        if cs is not None:
            total += ce - cs
        out[cls] = total / span if span > 0 else 0.0
    return out


def _bottleneck(agg):
    if not agg or max(agg.values()) < 0.25:
        return "launch_overhead", agg
    top = max(agg, key=lambda c: agg[c])
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


def _coverage(arch: Arch, cal: Calibration) -> str:
    """Honest coverage label (paper B.5: decline to inherit anchors)."""
    if cal.has_arch(arch.name):
        return "hardware-anchored+learned" if arch.calibrated else "learned-calibration"
    return "hardware-anchored" if arch.calibrated else "default-anchors"


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
    agg = _utilization(interp.exec_log, span)
    bn, _ = _bottleneck(agg)
    return CostReport(
        program_name=program.name, arch_name=arch.name, span_cycles=span + launch,
        launch_cycles=launch, busy=busy, bottleneck=bn, attribution=agg,
        coverage=_coverage(arch, cal), hints=_hints(program, bn, agg))


def predict_cores(programs, arch: Arch, cal: Calibration | None = None) -> CostReport:
    """Predict a multi-core candidate: span = slowest core (data-parallel launch).

    Attribution uses the critical (slowest) core's own utilization profile so
    that per-class fractions of span stay <= 1.0; summing across cores would
    double-count concurrent execution.
    """
    if not programs:
        raise ValueError("no programs")
    reports = [predict(p, arch, cal) for p in programs]
    critical = max(reports, key=lambda r: r.span_cycles)
    bn, _ = _bottleneck(critical.attribution)
    return CostReport(
        program_name=critical.program_name, arch_name=critical.arch_name,
        span_cycles=critical.span_cycles, launch_cycles=critical.launch_cycles,
        busy=critical.busy, bottleneck=bn, attribution=critical.attribution,
        coverage=critical.coverage, hints=_hints(programs[0], bn, critical.attribution))


def misprediction(predicted: float, measured: float | None) -> float | None:
    """Signed relative error; None when unmeasurable."""
    if measured is None or measured == 0:
        return None
    return (predicted - measured) / measured
