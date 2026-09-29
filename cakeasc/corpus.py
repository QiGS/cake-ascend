"""Corpus test gate (paper P6: "evaluate IR changes against the kernel-matrix
tests"; "a new verifier rule without corpus validation can reject valid
kernels").

A distilled rule is activated only if:
  1. it rejects its own invalid fixture (the rule is effective), and
  2. it produces no gate findings on the valid corpus (no false positives).

The valid corpus is every workload's naive + a mid-tuned parameter set,
built and simulated once per gate run — small by design.
"""
from __future__ import annotations

from . import builder as asc
from . import workloads as wl
from .arch import Arch, get_arch
from .diagnostics import GATE
from .ir import IRConstructionError


class CorpusGate:
    def __init__(self, arch: Arch | None = None):
        self.arch = arch or get_arch("ascend910b")
        self._valid_cache = None

    # ------------------------------------------------------------ corpus

    def valid_corpus(self):
        """(name, programs) pairs known-correct at their contract shapes."""
        if self._valid_cache is not None:
            return self._valid_cache
        out = []
        for name in wl.all_names():
            w = wl.get(name)
            base = w.default_params()
            mid = _bump_stages(base)
            for tag, params in (("naive", base), ("mid", mid)):
                try:
                    progs = _build(w, w.shape, params, f"{name}_{tag}")
                except IRConstructionError:
                    continue
                out.append((f"{name}:{tag}", progs))
        self._valid_cache = out
        return out

    # ----------------------------------------------------------- fixtures

    def invalid_fixture(self, template: str):
        """A program that the template's rule must reject, with the reason."""
        fn = _FIXTURES.get(template)
        if fn is None:
            return None
        return fn()

    # --------------------------------------------------------------- gate

    def test_rule_template(self, template: str) -> tuple[bool, str]:
        from .rules import TEMPLATES
        if template not in TEMPLATES:
            return False, f"unknown template {template}"
        rule = TEMPLATES[template](rule_id=f"GATECHECK.{template}")
        arch = self.arch

        fixture = self.invalid_fixture(template)
        if fixture is None:
            return False, "no invalid fixture defined for template"
        fprogs = fixture[1]
        hits = [f for p in fprogs for f in rule.check(p, arch) if f.severity == GATE]
        if not hits:
            return False, f"rule ineffective: did not flag its fixture ({fixture[0]})"

        false_pos = []
        for name, progs in self.valid_corpus():
            for p in progs:
                bad = [f for f in rule.check(p, arch) if f.severity == GATE]
                if bad:
                    false_pos.append(f"{name}: {bad[0].code}")
                    break
            if false_pos:
                break
        if false_pos:
            return False, f"false positive on valid corpus: {false_pos[0]}"
        return True, (f"flagged fixture '{fixture[0]}' with {hits[0].code}; "
                      f"no false positives on {len(self.valid_corpus())} corpus programs")


def _bump_stages(params):
    d = dict(params.__dict__)
    if "stages" in d and d["stages"] == 1:
        d["stages"] = 2
    return type(params)(**d)


def _build(w, shape, params, name):
    fn = w.kernel_fn(shape, params)
    return asc.build_all_cores(fn, name=name, block_dim=w.block_dim(shape, params))


# --------------------------------------------------------------------------
# invalid fixtures: minimal programs each violating exactly one contract


def _fx_gm_bounds():
    def kern(m):
        X = m.gm_param("X", "bf16", (64, 64))
        O = m.gm_param("O", "bf16", (64, 64))
        ub = m.ub_pool("ub", 232 * 1024)
        buf = ub.view("b", 0, (32, 32), "bf16", 1)
        r = m.role("r", "MTE2")
        st = m.role("st", "MTE3")
        rdy = m.event("rdy", r, st)
        with r:
            m.gm2ub(buf[0], X, (48, 0))       # 48+32 > 64 -> OOB
            m.commit(rdy)
        with st:
            m.wait(rdy)
            m.ub2gm(O, (0, 0), buf[0])
    return "gm_oob", asc.build_all_cores(kern, name="fx_gm_oob", block_dim=1)


def _fx_ub_capacity():
    def kern(m):
        X = m.gm_param("X", "bf16", (64, 64))
        O = m.gm_param("O", "bf16", (64, 64))
        ub = m.ub_pool("ub", 232 * 1024 + 1024)   # exceeds device UB
        buf = ub.view("b", 0, (32, 32), "bf16", 1)
        r = m.role("r", "MTE2")
        st = m.role("st", "MTE3")
        rdy = m.event("rdy", r, st)
        with r:
            m.gm2ub(buf[0], X, (0, 0))
            m.commit(rdy)
        with st:
            m.wait(rdy)
            m.ub2gm(O, (0, 0), buf[0])
    return "ub_pool_oversized", asc.build_all_cores(kern, name="fx_ub", block_dim=1)


def _fx_matmul_alignment():
    def kern(m):
        A = m.gm_param("A", "bf16", (48, 48))
        B = m.gm_param("B", "bf16", (48, 48))
        C = m.gm_param("C", "fp32", (48, 48))
        ub = m.ub_pool("ub", 64 * 1024)
        ba = ub.view("a", 0, (40, 48), "bf16", 1)   # 40 % 16 != 0
        bb = ub.view("b", 8192, (48, 48), "bf16", 1)
        bc = ub.view("c", 16384, (40, 48), "fp32", 1)
        acc = m.l0c("acc", (40, 48))
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
            m.matmul(acc, ba[0], bb[0], clear=True)
            m.l0c2ub(bc[0], acc)
            m.commit(crdy)
        with st:
            m.wait(crdy)
            m.ub2gm(C, (0, 0), bc[0])
    return "matmul_unaligned", asc.build_all_cores(kern, name="fx_mma_align", block_dim=1)


def _fx_copy_alignment():
    def kern(m):
        X = m.gm_param("X", "bf16", (64, 64))
        O = m.gm_param("O", "bf16", (64, 64))
        ub = m.ub_pool("ub", 64 * 1024)
        buf = ub.view("b", 0, (32, 24), "bf16", 1)   # 24*2B = 48B not 32B-aligned
        r = m.role("r", "MTE2")
        st = m.role("st", "MTE3")
        rdy = m.event("rdy", r, st)
        with r:
            m.gm2ub(buf[0], X, (0, 0))
            m.commit(rdy)
        with st:
            m.wait(rdy)
            m.ub2gm(O, (0, 0), buf[0])
    return "copy_unaligned", asc.build_all_cores(kern, name="fx_copy_align", block_dim=1)


def _fx_view_overlap():
    def kern(m):
        X = m.gm_param("X", "bf16", (64, 64))
        O = m.gm_param("O", "bf16", (64, 64))
        ub = m.ub_pool("ub", 64 * 1024)
        b1 = ub.view("b1", 0, (32, 32), "bf16", 1)
        b2 = ub.view("b2", 512, (32, 32), "bf16", 1)   # overlaps b1 (2048B)
        r = m.role("r", "MTE2")
        st = m.role("st", "MTE3")
        rdy = m.event("rdy", r, st)
        with r:
            m.gm2ub(b1[0], X, (0, 0))
            m.commit(rdy)
        with st:
            m.wait(rdy)
            m.ub2gm(O, (0, 0), b2[0])
    return "view_overlap", asc.build_all_cores(kern, name="fx_overlap", block_dim=1)


def _fx_back_pressure():
    # producer rewrites a slot (trip 3 > stages 2) without any back event
    def kern(m):
        X = m.gm_param("X", "bf16", (3 * 32, 32))
        O = m.gm_param("O", "bf16", (3 * 32, 32))
        ub = m.ub_pool("ub", 64 * 1024)
        buf = ub.view("b", 0, (32, 32), "bf16", 2)
        ld = m.role("ld", "MTE2")
        st = m.role("st", "MTE3")
        pipe = m.pipeline("p", 2)
        rdy = m.event("rdy", ld, st, pipe)
        with ld:
            for t in m.tile_loop("t", 3):
                m.gm2ub(buf[t % 2], X, (t * 32, 0))       # no wait before rewrite
                m.commit(rdy, stage=t % 2)
        with st:
            for t in m.tile_loop("t", 3):
                m.wait(rdy, stage=t % 2)
                m.ub2gm(O, (t * 32, 0), buf[t % 2])
    return "slot_overwrite_hazard", asc.build_all_cores(kern, name="fx_bp", block_dim=1)


def _fx_first_clear():
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
            m.matmul(acc, ba[0], bb[0], clear=False)   # first use, not cleared
            m.l0c2ub(bc[0], acc)
            m.commit(crdy)
        with st:
            m.wait(crdy)
            m.ub2gm(C, (0, 0), bc[0])
    return "first_use_not_cleared", asc.build_all_cores(kern, name="fx_clear", block_dim=1)


def _fx_l0c_capacity():
    def kern(m):
        A = m.gm_param("A", "bf16", (256, 256))
        B = m.gm_param("B", "bf16", (256, 256))
        C = m.gm_param("C", "fp32", (256, 256))
        ub = m.ub_pool("ub", 232 * 1024)
        ba = ub.view("a", 0, (128, 128), "bf16", 1)
        bb = ub.view("b", 32768, (128, 128), "bf16", 1)
        acc = m.l0c("acc", (512, 512))                  # 1MB > l0c capacity
        ld = m.role("ld", "MTE2")
        cu = m.role("cu", "CUBE")
        rdy = m.event("rdy", ld, cu)
        with ld:
            m.gm2ub(ba[0], A, (0, 0))
            m.commit(rdy)
        with cu:
            m.wait(rdy)
    return "l0c_oversized", asc.build_all_cores(kern, name="fx_l0c", block_dim=1)


_FIXTURES = {
    "gm_bounds": _fx_gm_bounds,
    "ub_capacity": _fx_ub_capacity,
    "matmul_alignment": _fx_matmul_alignment,
    "data_copy_alignment": _fx_copy_alignment,
    "view_overlap": _fx_view_overlap,
    "slot_back_pressure": _fx_back_pressure,
    "matmul_first_clear": _fx_first_clear,
    "l0c_capacity": _fx_l0c_capacity,
}
