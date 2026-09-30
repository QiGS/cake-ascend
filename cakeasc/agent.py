"""The kernel-agent loop (paper Sec. 4).

Four stages per iteration:
  1. generate structurally distinct candidates (proposer)
  2. filter: IR construction checks, verifier hard gates, cost-model
     ranking — before spending simulation time
  3. evaluate survivors: simulator + external oracle + measured span
  4. route evidence: gate findings -> candidates; recurring dynamic
     failures -> verifier rules; mispredictions -> cost calibration;
     construction gaps -> IR-evolution proposals (human merge gate)

The workload contract (shapes, oracle, tolerance, hardware) is fixed
authority; every result is retained in the archive for auditability.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

from . import builder as asc
from . import workloads as wl
from .arch import Arch
from .costmodel import Calibration, CostReport, cost_class, predict_cores
from .diagnostics import GATE, Finding
from .distiller import distill
from .evidence import CostSample, EvidenceStore
from .heuristics import HeuristicProposer, LLMProposer
from .interpreter import run_simulation
from .ir import IRConstructionError
from .rules import Registry
from .verifier import verify_cores

STAGES = ("constructed", "gate_rejected", "construction_failed",
          "sim_rejected", "incorrect", "correct")


@dataclass
class CandidateEval:
    cid: str
    iteration: int
    note: str = ""
    params: object = None
    block_dim: int = 1
    sig: str = ""
    stage: str = "constructed"
    findings: list = field(default_factory=list)
    predicted_cycles: float | None = None
    measured_cycles: float | None = None
    correct_note: str = ""
    programs: list = field(default_factory=list)


@dataclass
class EvolutionOpts:
    seed: int = 7
    propose_n: int = 6
    eval_topk: int = 4
    distill_threshold: int = 3
    apply_evolution: bool = False
    use_llm: bool = False
    verbose: bool = True


class Evolution:
    def __init__(self, workload, arch: Arch, registry: Registry,
                 calibration: Calibration | None = None,
                 evidence: EvidenceStore | None = None,
                 opts: EvolutionOpts | None = None):
        self.wl = workload
        self.arch = arch
        self.registry = registry
        self.cal = calibration or Calibration()
        self.evidence = evidence or EvidenceStore(
            distill_threshold=(opts.distill_threshold if opts else 3))
        self.opts = opts or EvolutionOpts()
        self.rng = random.Random(self.opts.seed)
        self.proposer = (LLMProposer(workload, rng=self.rng) if self.opts.use_llm
                         else HeuristicProposer(workload, self.rng))
        self.archive: list[CandidateEval] = []
        self.best: CandidateEval | None = None
        self.baseline_span: float | None = None
        self.seen_sigs: set[str] = set()
        self.iteration = 0
        self.rule_events: list[str] = []
        self._inputs_cache = {}
        self._expected_cache = {}

    # ------------------------------------------------------------- helpers

    def _contract(self, shape):
        key = tuple(sorted(shape.items()))
        if key not in self._inputs_cache:
            inputs = self.wl.make_inputs(shape, seed=self.opts.seed)
            self._inputs_cache[key] = inputs
        if key not in self._expected_cache:
            self._expected_cache[key] = self.wl.oracle(shape, self._inputs_cache[key])
        return self._inputs_cache[key], self._expected_cache[key]

    def _class_counts(self, programs) -> dict:
        counts = {}
        for prog in programs:
            core = {}
            for op in prog.ops:
                c = cost_class(op)
                core[c] = core.get(c, 0) + 1
            for c, n in core.items():
                counts[c] = max(counts.get(c, 0), n)
        return counts

    def _log(self, text):
        if self.opts.verbose:
            print(text)

    # ---------------------------------------------------------- evaluation

    def _evaluate(self, cand: CandidateEval, shape) -> CandidateEval:
        inputs, expected = self._contract(shape)
        t0 = time.time()
        sim = run_simulation(cand.programs, self.arch, inputs=inputs,
                             input_shapes=self.wl.param_shapes(shape))
        cand.findings.extend(sim.findings)
        if not sim.ok:
            cand.stage = "sim_rejected"
            self.evidence.record_findings(sim.findings, self.iteration, cand.cid)
            return cand
        cand.measured_cycles = sim.span_cycles
        ok, note = self.wl.compare(shape, expected, sim.outputs, inputs)
        if not ok:
            cand.stage = "incorrect"
            cand.correct_note = note
            self.evidence.record("numerical_mismatch", "NUMERICAL.mismatch",
                                 detail=note, iteration=self.iteration,
                                 candidate=cand.cid)
            return cand
        cand.stage = "correct"
        cand.correct_note = note
        # cost misprediction sample (paper: calibration target evidence)
        if cand.predicted_cycles and sim.span_cycles:
            self.evidence.record_cost(CostSample(
                iteration=self.iteration, candidate=cand.cid,
                predicted=cand.predicted_cycles, measured=sim.span_cycles,
                class_counts=getattr(cand, "_class_counts", {})))
        cand._eval_secs = time.time() - t0
        return cand

    # -------------------------------------------------------------- search

    def _parents(self):
        rows = []
        for c in self.archive:
            if c.stage == "correct":
                rows.append((c.params, c.measured_cycles, True))
        if not rows and self.baseline_span is not None:
            rows.append((self.wl.default_params(), self.baseline_span, True))
        return rows

    def _findings_by_parent(self):
        out = {}
        for c in self.archive:
            if c.stage in ("gate_rejected", "sim_rejected") and c.params is not None:
                fams = [f.meta.get("family") or f.code for f in c.findings]
                if fams:
                    out.setdefault(id(c.params), []).extend(fams)
        return out

    def run(self, iterations: int, shape: dict | None = None) -> dict:
        shape = shape or self.wl.shape
        self._log(f"=== kernel evolution: {self.wl.name} {shape} @ {self.arch.name} "
                  f"(iters={iterations}, proposer={'llm' if self.opts.use_llm else 'heuristic'}) ===")

        # baseline: naive-but-correct seed (paper: tuned-baseline normalization)
        base = self._run_candidate(("params", self.wl.default_params(), "naive-baseline"),
                                   shape)
        self.baseline_span = base.measured_cycles
        self._log(f"baseline (naive, stages=1): {self._fmt_span(base)}")

        for it in range(1, iterations + 1):
            self.iteration = it
            specs = self.proposer.propose(
                shape, self._parents(), self.opts.propose_n,
                findings_by_parent=self._findings_by_parent(),
                evidence_summary=self._evidence_summary(),
                arch=self.arch)
            evaluated, gate_rej, dup, failed = [], 0, 0, 0
            ranked = []
            for spec in specs:
                cand = self._build_spec(spec, shape)
                if cand is None:
                    failed += 1
                    continue
                if cand.sig in self.seen_sigs:
                    dup += 1
                    continue
                # stage 2: verifier gates
                findings = verify_cores(cand.programs, self.arch, self.registry)
                cand.findings = findings
                gates = [f for f in findings if f.severity == GATE]
                if gates:
                    cand.stage = "gate_rejected"
                    self.archive.append(cand)
                    self.seen_sigs.add(cand.sig)
                    self.evidence.record_findings(gates, it, cand.cid)
                    gate_rej += 1
                    continue
                # stage 2b: cost ranking (cheap analysis before sim time)
                rep = predict_cores(cand.programs, self.arch, self.cal)
                cand.predicted_cycles = rep.span_cycles
                cand._cost_report = rep
                cand._class_counts = self._class_counts(cand.programs)
                ranked.append((rep.span_cycles, cand))
            ranked.sort(key=lambda x: x[0])
            for _pred, cand in ranked[: self.opts.eval_topk]:
                self.seen_sigs.add(cand.sig)
                self.archive.append(cand)
                self._evaluate(cand, shape)          # stage 3: oracle + measurement
                evaluated.append(cand)
                if cand.stage == "correct":
                    if self.best is None or cand.measured_cycles < self.best.measured_cycles:
                        self.best = cand

            # stage 4: route evidence -> compiler evolution
            outcome = distill(self.evidence, self.registry, self.cal,
                              _Gate(), self.arch, apply=self.opts.apply_evolution)
            for prop in outcome.proposals:
                self.rule_events.append(prop.format())
                self._log("compiler evolution:\n" + prop.format())

            best_txt = (f"{self._fmt_span(self.best)} "
                        f"({self.baseline_span / self.best.measured_cycles:.3f}x vs naive)"
                        if self.best else "none yet")
            fam_txt = ", ".join(f"{f}x{n}" for f, n, _ in
                                self.evidence.families_summary()[:4]) or "-"
            self._log(f"it {it:02d} | proposed {len(specs)} (dup {dup}, build-fail {failed}) "
                      f"| gate-rej {gate_rej} | eval {len(evaluated)} "
                      f"(correct {sum(1 for c in evaluated if c.stage == 'correct')}) "
                      f"| best {best_txt} | evidence: {fam_txt}")
        return self.summary()

    # ------------------------------------------------------------- building

    def _build_spec(self, spec, shape) -> CandidateEval | None:
        kind, payload, note = spec
        it = self.iteration
        cid = f"c{it:02d}-{len(self.archive):03d}"
        cand = CandidateEval(cid=cid, iteration=it, note=note)
        try:
            if kind == "params":
                cand.params = payload
                cand.block_dim = self.wl.block_dim(shape, payload)
                fn = self.wl.kernel_fn(shape, payload)
                cand.programs = asc.build_all_cores(
                    fn, name=f"{self.wl.name}_{cid}", block_dim=cand.block_dim,
                    provenance={"params": payload.__dict__, "note": note})
            else:  # LLM/hand-authored source
                # SECURITY: exec() of remote-LLM output is arbitrary code
                # execution by design (same trust model as applying an LLM
                # patch to any repo). Only point the proposer at endpoints
                # you control; set CAKEASC_DISALLOW_EXEC=1 to hard-disable.
                if os.environ.get("CAKEASC_DISALLOW_EXEC"):
                    raise IRConstructionError(
                        "execution of authored schedule sources is disabled "
                        "(CAKEASC_DISALLOW_EXEC is set)",
                        region="source",
                        hint="run with the heuristic proposer only, or unset "
                             "CAKEASC_DISALLOW_EXEC in a trusted environment")
                ns = {}
                exec(payload, ns)  # noqa: S102 — trusted-source policy above
                kern = ns.get("kern")
                if kern is None:
                    raise IRConstructionError("source defines no `def kern(m)`",
                                              region="source")
                cand.block_dim = 1
                cand.programs = asc.build_all_cores(
                    kern, name=f"{self.wl.name}_{cid}", block_dim=1,
                    provenance={"source": payload, "note": note})
        except IRConstructionError as e:
            self.evidence.record("ir_construction", "IR.construction_error",
                                 region=e.region, detail=e.message,
                                 iteration=it, candidate=cid)
            cand.stage = "construction_failed"
            cand.findings = [_construction_finding(e)]
            self.archive.append(cand)
            return None
        cand.sig = cand.programs[0].structural_signature()
        return cand

    def _run_candidate(self, spec, shape) -> CandidateEval:
        self.iteration = 0
        cand = self._build_spec(spec, shape)
        cand.cid = "baseline"
        inputs, expected = self._contract(shape)
        rep = predict_cores(cand.programs, self.arch, self.cal)
        cand.predicted_cycles = rep.span_cycles
        cand._class_counts = self._class_counts(cand.programs)
        sim = run_simulation(cand.programs, self.arch, inputs=inputs,
                             input_shapes=self.wl.param_shapes(shape))
        cand.findings.extend(sim.findings)
        cand.measured_cycles = sim.span_cycles
        if sim.ok:
            ok, note = self.wl.compare(shape, expected, sim.outputs, inputs)
            cand.stage = "correct" if ok else "incorrect"
            cand.correct_note = note
        else:
            cand.stage = "sim_rejected"
        self.archive.append(cand)
        self.seen_sigs.add(cand.sig)
        return cand

    # -------------------------------------------------------------- summary

    def _fmt_span(self, cand) -> str:
        if cand is None or cand.measured_cycles is None:
            return "?"
        us = cand.measured_cycles * self.arch.seconds_per_cycle() * 1e6
        return f"{us:.2f}us ({cand.measured_cycles:.0f} cyc)"

    def _evidence_summary(self) -> str:
        rows = self.evidence.families_summary()
        if not rows:
            return ""
        top = ", ".join(f"{f}x{n} ({st})" for f, n, st in rows[:6])
        return f"distilled knowledge / open evidence: {top}"

    def summary(self) -> dict:
        correct = [c for c in self.archive if c.stage == "correct"]
        correct.sort(key=lambda c: c.measured_cycles)
        speedup = (self.baseline_span / self.best.measured_cycles
                   if (self.best and self.baseline_span) else None)
        return {
            "workload": self.wl.name,
            "iterations": self.iteration,
            "evaluated": len(self.archive),
            "correct_count": len(correct),
            "baseline_cycles": self.baseline_span,
            "best_cycles": self.best.measured_cycles if self.best else None,
            "best_params": (self.best.params.__dict__ if self.best and self.best.params
                            else None),
            "best_note": self.best.note if self.best else None,
            "speedup_vs_naive": speedup,
            "distilled_rules": [r.rule_id for r in self.registry.rules
                                if r.rule_id.startswith("DISTILLED.")],
            "rule_events": self.rule_events,
            "evidence": self.evidence.families_summary(),
        }


def _construction_finding(e: IRConstructionError):
    return Finding(category="ir_construction", severity=GATE,
                   code="IR.construction_error", message=e.message,
                   region=e.region, hint=e.hint)


class _Gate:
    """Lazy corpus gate adapter (avoids an import cycle at module load)."""

    def __init__(self):
        self._gate = None

    def test_rule_template(self, template):
        if self._gate is None:
            from .corpus import CorpusGate
            self._gate = CorpusGate()
        return self._gate.test_rule_template(template)
