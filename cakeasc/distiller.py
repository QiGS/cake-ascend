"""Distiller: recurring evidence -> verifier rules / cost calibration.

Two paths, mirroring the paper (Fig. 4):
1. dynamic failure recurrence  -> install the matching static rule template
   (an opaque runtime crash becomes a verifier rule; a repeated illegal
   lowering pattern becomes a static check);
2. systematic cost misprediction -> learn per-class overheads for the
   analytic cost model (a calibration target).

Everything is corpus-test-gated before activation (paper: "a new verifier
rule without corpus validation can reject valid kernels"), and human
merge-gated: proposals are printed as diffs and only persisted with
`--apply` (paper: "compiler evolution is still human-guided at merge
gates").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .costmodel import Calibration, COPY, MATMUL, VECTOR, SYNC
from .evidence import EvidenceStore

# dynamic/finding family -> rule template
FAMILY_TO_TEMPLATE = {
    "gm_oob": "gm_bounds",
    "SAFETY.gm_out_of_bounds": "gm_bounds",
    "ub_overflow": "ub_capacity",
    "SAFETY.ub_overflow": "ub_capacity",
    "slot_race": "slot_back_pressure",
    "SAFETY.slot_race": "slot_back_pressure",
    "matmul_unaligned": "matmul_alignment",
    "HARDWARE.matmul_unaligned": "matmul_alignment",
    "copy_unaligned": "data_copy_alignment",
    "HARDWARE.copy_unaligned": "data_copy_alignment",
    "uninit_acc": "matmul_first_clear",
    "DATA.matmul_uninit_acc": "matmul_first_clear",
}

# repair tactics (distilled knowledge fed back to the proposer)
TACTICS = {
    "gm_oob": "tile dims must divide the problem shape; shrink or re-partition",
    "ub_overflow": "halve the largest tile dimension or drop a pipeline stage",
    "slot_race": "keep back-pressure events for every reused slot (wait consumer-done "
                 "before rewriting a stage)",
    "matmul_unaligned": "round M/N/K tile dims to multiples of 16",
    "copy_unaligned": "round the contiguous tile dim to a 32-byte multiple "
                      "(16 x bf16 / 8 x fp32 elements)",
    "uninit_acc": "first matmul of each accumulation chain needs clear=True",
    "dead_write": "drop loads whose results are never consumed",
}


@dataclass
class Proposal:
    kind: str                       # "rule" | "calibration"
    name: str
    detail: str
    gate_result: str = ""           # corpus gate outcome
    accepted: bool = False

    def format(self) -> str:
        status = "APPLIED" if self.accepted else "PROPOSED (dry-run; use --apply)"
        return (f"[{self.kind}] {self.name} - {self.detail}\n"
                f"    corpus gate: {self.gate_result}\n"
                f"    status: {status}")


@dataclass
class DistillOutcome:
    proposals: list = field(default_factory=list)
    installed_rules: list = field(default_factory=list)
    calibration_updated: bool = False


def distill(evidence: EvidenceStore, registry, calibration: Calibration,
            corpus_gate, arch, apply: bool = False) -> DistillOutcome:
    outcome = DistillOutcome()

    for family in evidence.pending_distillations():
        if family == "cost_mispredict":
            outcome.proposals.append(_distill_calibration(
                evidence, calibration, arch, apply))
            evidence.mark_distilled(family)
            if outcome.proposals[-1] and outcome.proposals[-1].accepted:
                outcome.calibration_updated = True
            continue
        template = FAMILY_TO_TEMPLATE.get(family)
        if template is None:
            continue
        if registry.has(template):
            evidence.mark_distilled(family)
            continue
        occurrences = len(evidence.by_family[family])
        ok, detail = corpus_gate.test_rule_template(template)
        prop = Proposal(
            kind="rule", name=f"install {template}",
            detail=(f"family '{family}' recurred {occurrences}x "
                    f"(threshold {evidence.distill_threshold})"),
            gate_result=detail)
        if ok and apply:
            rule = registry.install(
                template,
                provenance={"family": family, "occurrences": occurrences,
                            "evidence": [i.detail for i in
                                         evidence.by_family[family][:3]]})
            outcome.installed_rules.append(rule)
            prop.accepted = True
        elif ok and not apply:
            prop.accepted = False
        else:
            prop.gate_result = f"REJECTED — {detail}"
        evidence.mark_distilled(family)
        outcome.proposals.append(prop)

    return outcome


def _distill_calibration(evidence: EvidenceStore, calibration: Calibration,
                         arch, apply: bool) -> Proposal | None:
    samples = [s for s in evidence.cost_samples
               if s.measured and s.measured > 0 and s.predicted > 0]
    if not samples:
        return None
    overheads = _fit_overheads(samples)

    # validation: MAE over samples must improve
    def mae(ovh):
        errs = []
        for s in samples:
            pred = s.predicted + sum(ovh.get(c, 0.0) * n
                                     for c, n in s.class_counts.items())
            errs.append(abs(pred - s.measured) / s.measured)
        return sum(errs) / len(errs)

    base = mae({})
    after = mae(overheads)
    if after >= base:
        return Proposal(kind="calibration", name="cost-model overheads",
                        detail=f"rejected: MAE {after:.3f} not better than {base:.3f}",
                        gate_result="validation failed", accepted=False)
    prop = Proposal(
        kind="calibration", name=f"cost-model overheads @ {arch.name}",
        detail=(f"per-op overheads cyc: " +
                ", ".join(f"{k}={v:.1f}" for k, v in overheads.items()) +
                f"; median rel mispredict {base:.0%} -> {after:.0%}"),
        gate_result=f"validation passed (MAE {base:.3f} -> {after:.3f})")
    if apply:
        calibration.update(arch.name, overheads=overheads,
                           note="distilled from recurring cost misprediction")
        prop.accepted = True
    return prop


_OVERHEAD_CLAMP = {COPY: 512.0, MATMUL: 256.0, VECTOR: 64.0, SYNC: 16.0}


def _fit_overheads(samples, classes=(COPY, MATMUL, VECTOR, SYNC)):
    """Least-squares fit of per-class fixed overheads:
    minimize || measured - predicted - N o ||^2 over samples, o in [0, clamp]."""
    n = len(classes)
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for s in samples:
        r = s.measured - s.predicted
        counts = [s.class_counts.get(c, 0) for c in classes]
        if not any(counts) or r < 0:
            continue          # negative residual: overheads hidden under overlap
        for i in range(n):
            if counts[i]:
                b[i] += counts[i] * r
                for j in range(n):
                    A[i][j] += counts[i] * counts[j]
    sol = _solve_linear(A, b)
    return {c: max(0.0, min(_OVERHEAD_CLAMP.get(c, 64.0), sol[i]))
            for i, c in enumerate(classes)}


def _solve_linear(A, b):
    """Gaussian elimination with partial pivoting; zero row -> zero unknown."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col] / pv
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    out = [0.0] * n
    for i in range(n):
        if abs(M[i][i]) > 1e-12:
            out[i] = M[i][n] / M[i][i]
    return out
