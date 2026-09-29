import unittest

from cakeasc.agent import Evolution, EvolutionOpts
from cakeasc.arch import get_arch
from cakeasc.costmodel import Calibration
from cakeasc.distiller import _fit_overheads
from cakeasc.evidence import CostSample, EvidenceStore
from cakeasc.heuristics import HeuristicProposer, LLMProposer
from cakeasc import workloads as wl
from cakeasc.rules import Registry


def synth_samples():
    """measured = predicted + 64*ncopy + 4*nsync (+noise-free)."""
    out = []
    layouts = [({"copy": 10, "sync": 100}, 0), ({"copy": 40, "sync": 50}, 0),
               ({"copy": 5, "sync": 200}, 0), ({"copy": 80, "sync": 10}, 0)]
    for counts, _ in layouts:
        pred = 1000.0
        meas = pred + 64 * counts["copy"] + 4 * counts["sync"]
        out.append(CostSample(iteration=0, candidate="x", predicted=pred,
                              measured=meas, class_counts=counts))
    return out


class TestDistiller(unittest.TestCase):
    def test_overhead_fit_recovers_values(self):
        ovh = _fit_overheads(synth_samples())
        self.assertAlmostEqual(ovh["copy"], 64.0, delta=1.0)
        self.assertAlmostEqual(ovh["sync"], 4.0, delta=1.0)
        self.assertEqual(ovh["matmul"], 0.0)
        self.assertEqual(ovh["vector"], 0.0)

    def test_rule_distillation_from_recurring_evidence(self):
        ev = EvidenceStore(distill_threshold=3)
        for _ in range(3):
            ev.record("gm_oob", "SAFETY.gm_out_of_bounds", detail="x")
        self.assertIn("gm_oob", ev.pending_distillations())
        reg = Registry.baseline()
        from cakeasc.corpus import CorpusGate
        from cakeasc.distiller import distill
        outcome = distill(ev, reg, Calibration(), CorpusGate(),
                          get_arch("ascend910b"), apply=True)
        self.assertTrue(reg.has("gm_bounds"))
        self.assertEqual(len(outcome.installed_rules), 1)
        self.assertIn("gm_oob", ev.distilled)
        # once distilled, the family is not re-proposed
        ev.record("gm_oob", "SAFETY.gm_out_of_bounds", detail="again")
        self.assertNotIn("gm_oob", ev.pending_distillations())

    def test_dry_run_does_not_install(self):
        ev = EvidenceStore(distill_threshold=3)
        for _ in range(3):
            ev.record("ub_overflow", "SAFETY.ub_overflow", detail="x")
        reg = Registry.baseline()
        from cakeasc.corpus import CorpusGate
        from cakeasc.distiller import distill
        outcome = distill(ev, reg, Calibration(), CorpusGate(),
                          get_arch("ascend910b"), apply=False)
        self.assertFalse(reg.has("ub_capacity"))
        self.assertEqual(outcome.installed_rules, [])
        self.assertTrue(outcome.proposals)

    def test_corpus_gate_rejects_unknown_template(self):
        from cakeasc.corpus import CorpusGate
        ok, detail = CorpusGate().test_rule_template("nope")
        self.assertFalse(ok)


class TestProposers(unittest.TestCase):
    def test_heuristic_proposes_distinct(self):
        w = wl.get("vec_add")
        prop = HeuristicProposer(w, rng=__import__("random").Random(1))
        base = w.default_params()
        specs = prop.propose(w.shape, [(base, 1000.0, True)], n=6)
        self.assertGreaterEqual(len(specs), 1)
        kinds = {s[0] for s in specs}
        self.assertEqual(kinds, {"params"})
        keys = set()
        for kind, params, note in specs:
            keys.add(tuple(sorted(params.__dict__.items())))
        self.assertEqual(len(keys), len(specs))

    def test_llm_falls_back_offline(self):
        w = wl.get("vec_add")
        prop = LLMProposer(w)  # no env configured
        specs = prop.propose(w.shape, [], 3)
        self.assertTrue(specs)
        self.assertEqual(specs[0][0], "params")


class TestEvolutionLoop(unittest.TestCase):
    def test_small_run_improves(self):
        w = wl.get("vec_add")
        opts = EvolutionOpts(seed=3, propose_n=4, eval_topk=3, verbose=False)
        evo = Evolution(w, get_arch("ascend910b"), Registry.baseline(), opts=opts)
        summary = evo.run(3)
        self.assertIsNotNone(evo.baseline_span)
        self.assertIsNotNone(evo.best)
        self.assertLess(evo.best.measured_cycles, evo.baseline_span)
        self.assertGreater(summary["speedup_vs_naive"], 1.0)
        self.assertTrue(any(c.stage == "correct" for c in evo.archive))
        # contract stability: oracle recomputed per shape, not per candidate
        self.assertEqual(len(evo._expected_cache), 1)


if __name__ == "__main__":
    unittest.main()
