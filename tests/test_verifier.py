import unittest

from cakeasc import builder as asc
from cakeasc.corpus import CorpusGate
from cakeasc.diagnostics import GATE, HINT
from cakeasc.rules import Registry, TEMPLATES
from cakeasc.verifier import verify_cores
from tests.test_ir import tiny_kern


def build_tiny():
    return asc.build_all_cores(tiny_kern, name="t", block_dim=1)


class TestVerifier(unittest.TestCase):
    def setUp(self):
        self.arch = __import__("cakeasc.arch", fromlist=["get_arch"]).get_arch("ascend910b")
        self.reg = Registry.baseline()

    def test_valid_program_clean(self):
        findings = verify_cores(build_tiny(), self.arch, self.reg)
        gates = [f for f in findings if f.severity == GATE]
        self.assertEqual(gates, [])

    def test_event_starvation_gates(self):
        def bad(m):
            X = m.gm_param("X", "bf16", (64,))
            O = m.gm_param("O", "bf16", (64,))
            ub = m.ub_pool("ub", 1024)
            buf = ub.view("b", 0, (16,), "bf16", 1)
            a = m.role("a", "MTE2")
            b = m.role("b", "MTE3")
            e = m.event("e", a, b)
            with a:
                m.gm2ub(buf[0], X, (0,))
            with b:
                m.wait(e)          # never committed -> starvation
                m.ub2gm(O, (0,), buf[0])
        findings = verify_cores(asc.build_all_cores(bad, name="bad", block_dim=1),
                                self.arch, self.reg)
        codes = [f.code for f in findings if f.severity == GATE]
        self.assertIn("SCHEDULE.event_starved", codes)

    def test_unused_event_is_hint_not_gate(self):
        def prog(m):
            X = m.gm_param("X", "bf16", (64,))
            O = m.gm_param("O", "bf16", (64,))
            ub = m.ub_pool("ub", 1024)
            buf = ub.view("b", 0, (16,), "bf16", 1)
            a = m.role("a", "MTE2")
            b = m.role("b", "MTE3")
            e = m.event("e", a, b)
            unused = m.event("unused", a, b)   # declared, never wired
            with a:
                m.gm2ub(buf[0], X, (0,))
                m.commit(e)
            with b:
                m.wait(e)
                m.ub2gm(O, (0,), buf[0])
        findings = verify_cores(asc.build_all_cores(prog, name="p", block_dim=1),
                                self.arch, self.reg)
        gate_codes = [f.code for f in findings if f.severity == GATE]
        hint_codes = [f.code for f in findings if f.severity == HINT]
        self.assertNotIn("SCHEDULE.event_unused", gate_codes)
        self.assertIn("SCHEDULE.event_unused", hint_codes)


class TestRuleTemplates(unittest.TestCase):
    """Every template flags its fixture and stays clean on the valid corpus."""

    def test_all_templates_gated(self):
        gate = CorpusGate()
        for template in TEMPLATES:
            with self.subTest(template=template):
                ok, detail = gate.test_rule_template(template)
                self.assertTrue(ok, detail)


class TestRegistry(unittest.TestCase):
    def test_install_unknown_template(self):
        reg = Registry.baseline()
        with self.assertRaises(ValueError):
            reg.install("no_such_template")

    def test_json_roundtrip(self):
        reg = Registry.baseline()
        reg.install("gm_bounds", provenance={"family": "gm_oob"})
        data = reg.to_json()
        reg2 = Registry.from_json(data)
        self.assertTrue(reg2.has("gm_bounds"))
        self.assertEqual(len(reg.rules), len(reg2.rules))


if __name__ == "__main__":
    unittest.main()
