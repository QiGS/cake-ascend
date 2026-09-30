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
        for needle in ("__global__ __aicore__ void", "DataCopy", "SetFlag<",
                       "WaitFlag<", "InitBuffer", "for (int t = 0"):
            self.assertIn(needle, src)

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

    def test_deterministic_output_across_processes(self):
        # same-process equality cannot catch salted hash() use; compare two
        # independent interpreter processes instead
        first = _gen_in_subprocess()
        second = _gen_in_subprocess()
        self.assertEqual(first, second)
        self.assertEqual(first, gen())


if __name__ == "__main__":
    unittest.main()
