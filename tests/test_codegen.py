import os
import subprocess
import sys
import unittest

from cakeasc import builder as asc
from cakeasc import workloads as wl
from cakeasc.arch import get_arch
from cakeasc.codegen import generate_ascendc
from tests.test_ir import tiny_kern

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gen():
    progs = asc.build_all_cores(tiny_kern, name="t", block_dim=1)
    return generate_ascendc(progs, get_arch("ascend910b"))


# cross-process snapshot: catches PYTHONHASHSEED-dependent output
_SNAPSHOT_SCRIPT = (
    "import sys; sys.path.insert(0, r'{root}'); "
    "from cakeasc.codegen import generate_ascendc; "
    "from cakeasc import builder as asc; "
    "from cakeasc.arch import get_arch; "
    "from tests.test_ir import tiny_kern; "
    "progs = asc.build_all_cores(tiny_kern, name='t', block_dim=1); "
    "sys.stdout.write(generate_ascendc(progs, get_arch('ascend910b')))"
).format(root=_REPO_ROOT)


def _gen_in_subprocess():
    out = subprocess.run([sys.executable, "-c", _SNAPSHOT_SCRIPT],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise AssertionError(f"snapshot subprocess failed: {out.stderr[-400:]}")
    return out.stdout


class TestCodegen(unittest.TestCase):
    def test_expected_constructs(self):
        src = gen()
        for needle in ("__global__ __aicore__", "DataCopy", "SetFlag<HardEvent::",
                       "WaitFlag<HardEvent::", "pipe.InitBuffer", "for (int t = 0"):
            self.assertIn(needle, src)

    def test_real_ascendc_surface(self):
        """The real-API contracts: header, types, GM entry style, namespace,
        HardEvent enum members."""
        src = gen()
        self.assertIn('#include "kernel_operator.h"', src)
        self.assertIn("using namespace AscendC;", src)
        self.assertIn("GM_ADDR X_gm", src)                      # entry style
        self.assertIn("__gm__ bfloat16_t* __restrict__ X =", src)  # typed cast
        self.assertIn("bfloat16_t", src)                        # real type names
        self.assertRegex(src, r"SetFlag<HardEvent::[A-Z0-9_]+>\(EVENT_ID\d\)")

    def test_gemm_path_primitive_clean(self):
        """gemm lowers to real primitives only (no composite NOTE markers)."""
        w = wl.get("gemm")
        p = type(w.default_params())(**{**w.default_params().__dict__,
                                        "stages": 2, "block_dim": 1})
        progs = asc.build_all_cores(w.kernel_fn(w.shape, p), name="g",
                                    block_dim=1)
        src = generate_ascendc(progs, get_arch("ascend910b"))
        self.assertNotIn("NOTE(composite)", src)
        self.assertIn("matmul::Matmul<bfloat16_t, bfloat16_t, float_t", src)
        self.assertIn("TCubeTBuf<TPosition::A1> buf_A_0", src)
        self.assertIn("TCubeTBuf<TPosition::A2> buf_B_0", src)
        self.assertIn("TCubeTBuf<TPosition::C1C2> l0c_acc", src)  # L0C via TCubeTBuf
        self.assertIn("SetFlag<HardEvent::MTE2_M>", src)        # real cube sync

    def test_addresses_are_real_linear_offsets(self):
        # P0 regression: emitted GM addresses must be concrete linear element
        # offsets (param + <int>), never synthesized/undefined symbols
        src = gen()
        self.assertNotIn("_offset_", src)
        self.assertRegex(src, r"X \+ \d+")

    def test_multi_core_guard(self):
        w = wl.get("vec_add")
        params = type(w.default_params())(tile=1024, stages=1, block_dim=2,
                                          prec="bf16")
        progs = asc.build_all_cores(w.kernel_fn(w.shape, params), name="va",
                                    block_dim=2)
        src = generate_ascendc(progs, get_arch("ascend910b"))
        self.assertIn("GetBlockIdx() == 0", src)
        self.assertIn("GetBlockIdx() == 1", src)

    def test_vec_add_primitive_clean(self):
        w = wl.get("vec_add")
        progs = asc.build_all_cores(
            w.kernel_fn(w.shape, w.default_params()), name="va", block_dim=1)
        src = generate_ascendc(progs, get_arch("ascend910b"))
        self.assertNotIn("NOTE(composite)", src)
        self.assertIn("Add(", src)                              # real elementwise

    def test_deterministic_output_across_processes(self):
        # same-process equality cannot catch salted hash() use; compare two
        # independent interpreter processes instead
        first = _gen_in_subprocess()
        second = _gen_in_subprocess()
        self.assertEqual(first, second)
        self.assertEqual(first, gen())


if __name__ == "__main__":
    unittest.main()
