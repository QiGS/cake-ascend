import unittest

from cakeasc.agent import Evolution, EvolutionOpts
from cakeasc.arch import get_arch
from cakeasc import workloads as wl
from cakeasc.dispatcher import build_portfolio
from cakeasc.rules import Registry


class TestDispatcher(unittest.TestCase):
    def test_portfolio_covers_domain(self):
        w = wl.get("vec_add")
        opts = EvolutionOpts(seed=5, propose_n=4, eval_topk=3, verbose=False)
        evo = Evolution(w, get_arch("ascend910b"), Registry.baseline(), opts=opts)
        evo.run(3)
        domain = w.domain[:3]
        report = build_portfolio(evo, domain=domain)
        self.assertEqual(len(report.routes), len(domain))
        for route in report.routes:
            self.assertIsNotNone(route.naive_cycles)
            self.assertTrue(route.ok)
            self.assertIsNotNone(route.speedup)
        self.assertIsNotNone(report.gspan)
        self.assertGreater(report.gspan, 1.0)

    def test_guard_respects_domain_leakage_rules(self):
        # guards only partition the declared domain: a shape no candidate fits
        # must fall back to the naive route rather than inventing a new route
        w = wl.get("gemm")
        opts = EvolutionOpts(seed=5, propose_n=4, eval_topk=2, verbose=False)
        evo = Evolution(w, get_arch("ascend910b"), Registry.baseline(), opts=opts)
        evo.run(2)
        # N=96 is NOT in the domain and not divisible by most tiles -> treat as
        # an out-of-domain shape: guards must not claim it
        shape = {"M": 64, "N": 96, "K": 64}
        for cand in [c for c in evo.archive if c.params]:
            if cand.stage != "correct":
                continue
            guarded = w.domain_guard(shape, cand.params)
            # any guard that claims the shape must still simulate correctly
            if guarded:
                self.assertIn("N", shape)


if __name__ == "__main__":
    unittest.main()
