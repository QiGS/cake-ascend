import unittest

from cakeasc import builder as asc
from cakeasc import workloads as wl
from cakeasc.arch import get_arch
from cakeasc.codegen import generate_ascendc
from tests.test_ir import tiny_kern


def gen():
    progs = asc.build_all_cores(tiny_kern, name="t", block_dim=1)
    return generate_ascendc(progs, get_arch("ascend910b"))


class TestCodegen(unittest.TestCase):
    def test_expected_constructs(self):
        src = gen()
        for needle in ("__global__ __aicore__ void", "DataCopy", "SetFlag<",
                       "WaitFlag<", "InitBuffer", "for (int t = 0"):
            self.assertIn(needle, src)

    def test_multi_core_guard(self):
        w = wl.get("vec_add")
        params = type(w.default_params())(tile=1024, stages=1, block_dim=2,
                                          prec="bf16")
        progs = asc.build_all_cores(w.kernel_fn(w.shape, params), name="va",
                                    block_dim=2)
        src = generate_ascendc(progs, get_arch("ascend910b"))
        self.assertIn("GetBlockIdx() == 0", src)
        self.assertIn("GetBlockIdx() == 1", src)

    def test_deterministic_output(self):
        self.assertEqual(gen(), gen())


if __name__ == "__main__":
    unittest.main()
