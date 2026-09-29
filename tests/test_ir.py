import unittest

from cakeasc import builder as asc
from cakeasc.arch import get_arch
from cakeasc.ir import IRConstructionError, LoopMarker, Matmul


def tiny_kern(m):
    X = m.gm_param("X", "bf16", (64, 64))
    O = m.gm_param("O", "bf16", (64, 64))
    ub = m.ub_pool("ub", 64 * 1024)
    buf = ub.view("b", 0, (32, 32), "bf16", 2)
    ld = m.role("ld", "MTE2")
    st = m.role("st", "MTE3")
    pipe = m.pipeline("p", 2)
    rdy = m.event("rdy", ld, st, pipe)
    free = m.event("free", st, ld, pipe)
    with ld:
        for t in m.tile_loop("t", 3):
            s = t % 2
            if t >= 2:
                m.wait(free, stage=s)
            m.gm2ub(buf[s], X, (t * 32, 0))
            m.commit(rdy, stage=s)
    with st:
        for t in m.tile_loop("t", 3):
            s = t % 2
            m.wait(rdy, stage=s)
            m.ub2gm(O, (t * 32, 0), buf[s])
            m.commit(free, stage=s)


class TestBuilder(unittest.TestCase):
    def test_trace_unrolls_and_records(self):
        prog = asc.build(tiny_kern, name="t", block_dim=1)
        matmuls = [o for o in prog.ops if isinstance(o, Matmul)]
        self.assertEqual(matmuls, [])
        loads = [o for o in prog.ops if o.kind == "Gm2Ub"]
        self.assertEqual(len(loads), 3)
        markers = [o for o in prog.ops if isinstance(o, LoopMarker)]
        self.assertEqual(len(markers), 4)  # 2 begins + 2 ends (per role)
        tags = [o.tags for o in loads]
        self.assertEqual([t.get("t") for t in tags], [0, 1, 2])

    def test_slot_bounds_checked(self):
        def bad(m):
            X = m.gm_param("X", "bf16", (64,))
            ub = m.ub_pool("ub", 1024)
            buf = ub.view("b", 0, (16,), "bf16", 1)
            r = m.role("r", "MTE2")
            with r:
                m.gm2ub(buf[1], X, (0,))  # stage index out of range
        with self.assertRaises(IRConstructionError):
            asc.build(bad, name="bad", block_dim=1)

    def test_role_event_mismatch(self):
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
                m.commit(e)
            with b:
                m.wait(e)
                m.ub2gm(O, (0,), buf[0])
                m.commit(e)  # consumer committing a producer event
        with self.assertRaises(IRConstructionError):
            asc.build(bad, name="bad", block_dim=1)

    def test_structural_signature(self):
        p1 = asc.build(tiny_kern, name="t", block_dim=1)
        p2 = asc.build(tiny_kern, name="t", block_dim=1)
        self.assertEqual(p1.structural_signature(), p2.structural_signature())

        def other(m):
            X = m.gm_param("X", "bf16", (64, 64))
            ub = m.ub_pool("ub", 64 * 1024)
            buf = ub.view("b", 0, (32, 32), "bf16", 1)
            r = m.role("r", "MTE2")
            with r:
                m.gm2ub(buf[0], X, (0, 0))
        p3 = asc.build(other, name="o", block_dim=1)
        self.assertNotEqual(p1.structural_signature(), p3.structural_signature())


class TestArch(unittest.TestCase):
    def test_unknown_target_rejected(self):
        with self.assertRaises(ValueError):
            get_arch("ascend310p-nope")

    def test_910b_defaults(self):
        arch = get_arch("ascend910b")
        self.assertGreater(arch.ub_bytes, 200 * 1024)
        self.assertEqual(arch.cube_align, 16)


if __name__ == "__main__":
    unittest.main()
