import unittest

from cakeasc import builder as asc
from cakeasc import dtypes as dt
from cakeasc import workloads as wl
from cakeasc.arch import get_arch
from cakeasc.interpreter import run_simulation
from cakeasc.diagnostics import DYN


def run(w, shape, params, seed=11):
    progs = asc.build_all_cores(w.kernel_fn(shape, params), name="t",
                                block_dim=w.block_dim(shape, params))
    inputs = w.make_inputs(shape, seed)
    sim = run_simulation(progs, get_arch("ascend910b"), inputs=inputs,
                         input_shapes=w.param_shapes(shape))
    expected = w.oracle(shape, inputs)
    return sim, expected, inputs


class TestNumerics(unittest.TestCase):
    def test_vec_add(self):
        w = wl.get("vec_add")
        sim, exp, inputs = run(w, w.shape, w.default_params())
        self.assertTrue(sim.ok, [f.format() for f in sim.findings])
        ok, note = w.compare(w.shape, exp, sim.outputs, inputs)
        self.assertTrue(ok, note)

    def test_gemm(self):
        w = wl.get("gemm")
        sim, exp, inputs = run(w, w.shape, w.default_params())
        self.assertTrue(sim.ok, [f.format() for f in sim.findings])
        ok, note = w.compare(w.shape, exp, sim.outputs, inputs)
        self.assertTrue(ok, note)

    def test_kmeans_exact_indices(self):
        w = wl.get("kmeans_assign")
        sim, exp, inputs = run(w, w.shape, w.default_params())
        self.assertTrue(sim.ok, [f.format() for f in sim.findings])
        ok, note = w.compare(w.shape, exp, sim.outputs, inputs)
        self.assertTrue(ok, note)

    def test_bf16_storage_quantization(self):
        self.assertEqual(dt.quantize(1.0, "bf16"), 1.0)
        # bf16 keeps 7 mantissa bits; the midpoint 1+2^-8 rounds to even (1.0)
        self.assertEqual(dt.quantize(1.0 + 2 ** -8, "bf16"), 1.0)
        # above the midpoint rounds up to the next representable value
        self.assertEqual(dt.quantize(1.0 + 2 ** -8 + 2 ** -12, "bf16"), 1.0 + 2 ** -7)
        self.assertEqual(dt.quantize(1.0 + 2 ** -7, "bf16"), 1.0 + 2 ** -7)
        # halfway between 1+2^-7 and 1+2^-6 rounds to even mantissa (1+2^-6)
        self.assertEqual(dt.quantize(1.0 + 2 ** -7 + 2 ** -8, "bf16"), 1.0 + 2 ** -6)


class TestDynamicFindings(unittest.TestCase):
    def setUp(self):
        self.arch = get_arch("ascend910b")

    def _run(self, kern, shapes, inputs):
        progs = asc.build_all_cores(kern, name="t", block_dim=1)
        return run_simulation(progs, self.arch, inputs=inputs, input_shapes=shapes)

    def test_gm_oob_localized(self):
        def kern(m):
            X = m.gm_param("X", "bf16", (64,))
            O = m.gm_param("O", "bf16", (64,))
            ub = m.ub_pool("ub", 1024)
            buf = ub.view("b", 0, (32,), "bf16", 1)
            a = m.role("a", "MTE2")
            b = m.role("b", "MTE3")
            e = m.event("e", a, b)
            with a:
                m.gm2ub(buf[0], X, (48,))     # 48+32 > 64
                m.commit(e)
            with b:
                m.wait(e)
                m.ub2gm(O, (0,), buf[0])
        sim = self._run(kern, {"X": (64,), "O": (64,)},
                        {"X": [0.0] * 64, "O": [0.0] * 64})
        self.assertFalse(sim.ok)
        f = next(f for f in sim.findings if f.code == "SAFETY.gm_out_of_bounds")
        self.assertIn("op#", f.region)
        self.assertIn("repair", f.format())

    def test_missing_back_pressure_race(self):
        def kern(m):
            X = m.gm_param("X", "bf16", (4 * 16,))
            O1 = m.gm_param("O1", "bf16", (64,))
            ub = m.ub_pool("ub", 4096)
            buf = ub.view("b", 0, (16,), "bf16", 2)
            a = m.role("a", "MTE2")
            b = m.role("b", "MTE3")
            pipe = m.pipeline("p", 2)
            rdy = m.event("rdy", a, b, pipe)
            with a:
                for t in m.tile_loop("t", 4):
                    m.gm2ub(buf[t % 2], X, (t * 16,))
                    m.commit(rdy, stage=t % 2)
            with b:
                # deliberately slow consumer (6 ops/tile vs producer 2) and no
                # back-pressure events: the producer overwrites slot contents
                # before the consumer reads them -> version-skip race
                for t in m.tile_loop("t", 4):
                    s = t % 2
                    m.wait(rdy, stage=s)
                    m.ub2gm(O1, (t * 16,), buf[s])
                    m.ub2gm(O1, (t * 16,), buf[s])
                    m.ub2gm(O1, (t * 16,), buf[s])
                    m.ub2gm(O1, (t * 16,), buf[s])
                    m.ub2gm(O1, (t * 16,), buf[s])
        sim = self._run(kern, {"X": (64,), "O1": (64,)},
                        {"X": [float(i) for i in range(64)],
                         "O1": [0.0] * 64})
        self.assertFalse(sim.ok)
        self.assertIn("SAFETY.slot_race", [f.code for f in sim.findings])

    def test_deadlock(self):
        def kern(m):
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
                m.wait(e)      # producer never commits
                m.ub2gm(O, (0,), buf[0])
        sim = self._run(kern, {"X": (64,), "O": (64,)},
                        {"X": [0.0] * 64, "O": [0.0] * 64})
        self.assertFalse(sim.ok)
        self.assertIn("SCHEDULE.deadlock", [f.code for f in sim.findings])
        self.assertIsNone(sim.span_cycles)

    def test_uninitialized_accumulator(self):
        def kern(m):
            A = m.gm_param("A", "bf16", (32, 32))
            B = m.gm_param("B", "bf16", (32, 32))
            C = m.gm_param("C", "fp32", (32, 32))
            ub = m.ub_pool("ub", 64 * 1024)
            ba = ub.view("a", 0, (32, 32), "bf16", 1)
            bb = ub.view("b", 4096, (32, 32), "bf16", 1)
            bc = ub.view("c", 8192, (32, 32), "fp32", 1)
            acc = m.l0c("acc", (32, 32))
            ld = m.role("ld", "MTE2")
            cu = m.role("cu", "CUBE")
            st = m.role("st", "MTE3")
            rdy = m.event("rdy", ld, cu)
            crdy = m.event("crdy", cu, st)
            with ld:
                m.gm2ub(ba[0], A, (0, 0))
                m.gm2ub(bb[0], B, (0, 0))
                m.commit(rdy)
            with cu:
                m.wait(rdy)
                m.matmul(acc, ba[0], bb[0], clear=False)  # first use, not cleared
                m.l0c2ub(bc[0], acc)
                m.commit(crdy)
            with st:
                m.wait(crdy)
                m.ub2gm(C, (0, 0), bc[0])
        sim = self._run(kern, {"A": (32, 32), "B": (32, 32), "C": (32, 32)},
                        {"A": [0.5] * 1024, "B": [0.5] * 1024, "C": [0.0] * 1024})
        self.assertIn("DATA.matmul_uninit_acc", [f.code for f in sim.findings])

    def test_cross_core_gm_write_overlap(self):
        def kern(m):
            X = m.gm_param("X", "bf16", (64,))
            O = m.gm_param("O", "bf16", (64,))
            ub = m.ub_pool("ub", 1024)
            buf = ub.view("b", 0, (16,), "bf16", 1)
            a = m.role("a", "MTE2")
            b = m.role("b", "MTE3")
            e = m.event("e", a, b)
            # both cores write the SAME GM range (bad partition)
            with a:
                m.gm2ub(buf[0], X, (m.core_id() * 16,))
                m.commit(e)
            with b:
                m.wait(e)
                m.ub2gm(O, (16,), buf[0])
        progs = asc.build_all_cores(kern, name="t", block_dim=2)
        sim = run_simulation(progs, self.arch,
                             inputs={"X": [0.0] * 64, "O": [0.0] * 64},
                             input_shapes={"X": (64,), "O": (64,)})
        self.assertFalse(sim.ok)
        self.assertIn("SAFETY.gm_write_overlap", [f.code for f in sim.findings])

    def test_determinism(self):
        w = wl.get("gemm")
        p = w.default_params()
        s1, e1, i1 = run(w, w.shape, p, seed=3)
        s2, e2, i2 = run(w, w.shape, p, seed=3)
        self.assertEqual(s1.outputs["C"], s2.outputs["C"])
        self.assertEqual(s1.span_cycles, s2.span_cycles)


if __name__ == "__main__":
    unittest.main()
