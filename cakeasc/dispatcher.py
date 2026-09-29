"""Generalization stage (paper Sec. 6): from a tuned shape to a library.

Separate objectives from the inner loop: this stage is scored on
dispatcher-inclusive performance over a *declared* shape domain, after
strong per-shape seeds exist. Routes are chosen by the workload's guard
(divisibility predicates); the naive schedule is the explicit fallback.
Guards may only partition the declared domain — no new evaluation shapes
are introduced (preventing evaluation leakage), and every route is
numerically validated before it can claim a shape.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import builder as asc
from .costmodel import predict_cores
from .interpreter import run_simulation
from .verifier import verify_cores
from .diagnostics import GATE


@dataclass
class Route:
    shape: dict
    note: str = ""
    params_desc: str = ""
    measured_cycles: float | None = None
    naive_cycles: float | None = None
    ok: bool = False
    detail: str = ""
    is_fallback: bool = False

    @property
    def speedup(self) -> float | None:
        if self.ok and self.measured_cycles and self.naive_cycles:
            return self.naive_cycles / self.measured_cycles
        return None


@dataclass
class PortfolioReport:
    workload: str
    routes: list = field(default_factory=list)
    fallbacks: int = 0
    failures: int = 0

    @property
    def gspan(self) -> float | None:
        sps = [r.speedup for r in self.routes if r.speedup]
        if not sps:
            return None
        return math.exp(sum(math.log(s) for s in sps) / len(sps))

    def format(self, seconds_per_cycle) -> str:
        lines = [f"dispatcher portfolio: {self.workload} "
                 f"({len(self.routes)} shapes, {self.fallbacks} fallback, "
                 f"{self.failures} route rejections)"]
        for r in self.routes:
            us = r.measured_cycles * seconds_per_cycle * 1e6 if r.measured_cycles else None
            sp = f"{r.speedup:.3f}x" if r.speedup else "-"
            us_txt = f"{us:>8.2f}us" if us is not None else "        -"
            lines.append(
                f"  {r.shape}: {r.params_desc:<46} "
                f"{'' if r.ok else 'INVALID '}"
                f"{us_txt}  {sp:>8}  "
                f"{'(fallback)' if r.is_fallback else ''} {r.detail}")
        if self.gspan:
            lines.append(f"  Gspan (geomean speedup vs naive, dispatcher-inclusive): "
                         f"{self.gspan:.3f}x")
        return "\n".join(lines)


def build_portfolio(evolution, domain=None, log=None) -> PortfolioReport:
    wl = evolution.wl
    arch = evolution.arch
    domain = domain or wl.domain
    report = PortfolioReport(workload=wl.name)

    # correct candidates from the archive, best-first (paper: per-shape seeds)
    correct = [c for c in evolution.archive if c.stage == "correct" and c.params]
    correct.sort(key=lambda c: c.measured_cycles)

    for shape in domain:
        naive = _measure(evolution, shape, wl.default_params(), "naive")
        route = Route(shape=shape, naive_cycles=naive and naive[0])
        chosen = None
        for cand in correct:
            if not wl.domain_guard(shape, cand.params):
                continue
            measured = _measure(evolution, shape, cand.params, cand.note)
            if measured is None:
                report.failures += 1
                continue
            span, ok, detail = measured
            if not ok:
                report.failures += 1
                continue              # route rejected; try next seed
            chosen = (cand, span, detail)
            break
        if chosen is None:
            span, ok, detail = naive
            route.is_fallback = True
            report.fallbacks += 1
            route.params_desc = "naive(fallback)"
            route.measured_cycles = span
            route.ok = bool(ok)
            route.detail = detail or ""
        else:
            cand, span, detail = chosen
            route.params_desc = _params_desc(cand.params)
            route.note = cand.note
            route.measured_cycles = span
            route.ok = True
            route.detail = detail or ""
        report.routes.append(route)
    if log:
        log(report.format(arch.seconds_per_cycle()))
    return report


def _params_desc(params) -> str:
    d = params.__dict__
    return ",".join(f"{k}={v}" for k, v in sorted(d.items()))[:44]


def _measure(evolution, shape, params, note):
    """Build + verify + simulate one (shape, params); None on gate/build failure."""
    wl = evolution.wl
    arch = evolution.arch
    try:
        fn = wl.kernel_fn(shape, params)
        progs = asc.build_all_cores(fn, name=f"{wl.name}_portfolio",
                                    block_dim=wl.block_dim(shape, params),
                                    provenance={"params": params.__dict__, "note": note})
    except Exception:
        return None
    findings = verify_cores(progs, arch, evolution.registry)
    if any(f.severity == GATE for f in findings):
        return None
    inputs = evolution._contract(shape)[0]
    sim = run_simulation(progs, arch, inputs=inputs,
                         input_shapes=wl.param_shapes(shape))
    if not sim.ok:
        return None
    expected = evolution._contract(shape)[1]
    ok, detail = wl.compare(shape, expected, sim.outputs, inputs)
    return sim.span_cycles, ok, detail
