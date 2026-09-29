import unittest

from cakeasc import builder as asc
from cakeasc import workloads as wl
from cakeasc.arch import get_arch
from cakeasc.costmodel import Calibration, predict_cores


def build(w, shape, params, name="t"):
    return asc.build_all_cores(w.kernel_fn(shape, params), name=name,
                               block_dim=w.block_dim(shape, params))


class TestCostModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch = get_arch("ascend910b")
        cls.w = wl.get("gemm")
        cls.shape = cls.w.shape

    def test_pipelining_predicted_faster(self):
        naive = self.w.default_params()
        tuned = type(naive)(**{**naive.__dict__, "stages": 2, "bm": 64,
                               "bn": 64, "bk": 64, "block_dim": 1})
        p1 = predict_cores(build(self.w, self.shape, naive), self.arch, Calibration())
        p2 = predict_cores(build(self.w, self.shape, tuned), self.arch, Calibration())
        self.assertLess(p2.span_cycles, p1.span_cycles,
                        "double buffering should reduce predicted span")

    def test_bottleneck_labels(self):
        vw = wl.get("vec_add")
        params = vw.default_params()
        rep = predict_cores(build(vw, vw.shape, params), self.arch, Calibration())
        self.assertEqual(rep.bottleneck, "memory_bound")

    def test_calibration_json_roundtrip(self):
        cal = Calibration()
        cal.update("ascend910b", multipliers={"copy": 1.1}, overheads={"sync": 4.0},
                   note="test")
        cal2 = Calibration.from_json(cal.to_json())
        self.assertEqual(cal2.for_arch("ascend910b", "sync"), (1.0, 4.0))
        self.assertEqual(cal2.for_arch("ascend910b", "copy"), (1.1, 0.0))

    def test_uncalibrated_coverage_flag(self):
        rep = predict_cores(build(self.w, self.shape, self.w.default_params()),
                            self.arch, Calibration())
        self.assertEqual(rep.coverage, "default-anchors")
        cal = Calibration()
        cal.update("ascend910b", overheads={"sync": 4.0})
        rep2 = predict_cores(build(self.w, self.shape, self.w.default_params()),
                             self.arch, cal)
        self.assertEqual(rep2.coverage, "calibrated")

    def test_report_has_hints(self):
        vw = wl.get("vec_add")
        rep = predict_cores(build(vw, vw.shape, vw.default_params()),
                            self.arch, Calibration())
        self.assertTrue(any("memory bound" in h for h in rep.hints))


if __name__ == "__main__":
    unittest.main()
