"""Proposers: candidate generation for stage 1 of the agent loop.

Two interchangeable sources behind the same interface (paper: "generate
structurally distinct Cake IR candidates"); both return a list of
("params", params, note) or ("source", python_source, note) specs:

- HeuristicProposer: deterministic mutation search over a workload's
  parameter space, biased by distilled repair tactics and archive fitness.
- LLMProposer: prompts an OpenAI-compatible model with the IR guide, the
  workload contract, parent candidates and routed evidence; parses
  schedule functions from the reply. Falls back to heuristics offline.
"""
from __future__ import annotations

import random

from .distiller import TACTICS
from .llm import IR_GUIDE, LLMClient, LLMError, arch_card, extract_code_blocks
from .workloads import mutate_params, params_dict


class HeuristicProposer:
    def __init__(self, workload, rng: random.Random | None = None):
        self.wl = workload
        self.rng = rng or random.Random(0)
        self._spec = None

    @property
    def spec(self):
        if self._spec is None:
            mod = __import__(f"cakeasc.workloads.{self.wl.name}", fromlist=["SPEC"])
            self._spec = mod.SPEC
        return self._spec

    def propose(self, shape, parents, n, findings_by_parent=None,
                evidence_summary="", arch=None) -> list:
        """parents: [(params, measured_span|None, correct)]; returns candidate specs."""
        out = []
        seen = set()
        findings_by_parent = findings_by_parent or {}
        for _ in range(n * 4):
            parent, _span, _correct = self._pick_parent(parents)
            families = findings_by_parent.get(id(parent), []) or []
            specs = self._mutate(parent, families)
            for params, note in specs:
                key = tuple(sorted(params_dict(params).items()))
                if key in seen:
                    continue
                seen.add(key)
                out.append(("params", params, note))
                if len(out) >= n:
                    return out
        return out

    def _pick_parent(self, parents):
        if not parents:
            return self.wl.default_params(), None, True
        scored = []
        for params, span, correct in parents:
            weight = 1.0 / (1.0 + span) if (correct and span) else 0.05
            scored.append((weight, params, span, correct))
        total = sum(s for s, *_ in scored)
        r = self.rng.random() * total
        acc = 0.0
        for s, params, span, correct in scored:
            acc += s
            if acc >= r:
                return params, span, correct
        return scored[-1][1], scored[-1][2], scored[-1][3]

    def _mutate(self, parent, families):
        # repair-first: distilled tactics bias the mutation space
        params, notes = parent, []
        for fam in families:
            tactic = {
                "gm_oob": lambda p: _shrink_to_fit(p, self.wl, self.spec),
                "matmul_unaligned": lambda p: _round_fields(p, mult=16),
                "copy_unaligned": _round_inner,
                "ub_overflow": lambda p: _halve_biggest(p, self.spec),
            }.get(fam)
            if tactic:
                params = tactic(parent)
                notes.append(f"repair({fam})")
                break
        mutants = mutate_params(params, self.spec, self.rng, 3)
        note = "explore" if not notes else "; ".join(notes)
        specs = [(params, note)] if notes else []
        specs += [(p, "explore") for p in mutants]
        return specs


def _shrink_to_fit(params, wl, spec):
    d = params_dict(params)
    for f in spec:
        for v in sorted(spec[f]):
            if v >= d[f]:
                continue
            d2 = dict(d)
            d2[f] = v
            p = type(params)(**d2)
            if wl.domain_guard(wl.shape, p):
                return p
    return params


def _round_fields(params, mult):
    d = params_dict(params)
    for f, v in d.items():
        if isinstance(v, int) and f not in ("stages", "block_dim"):
            d[f] = max(mult, (v // mult) * mult)
    return type(params)(**d)


def _round_inner(params):
    d = params_dict(params)
    for f in ("bn", "bk", "tile", "db", "tb"):
        if f in d and isinstance(d[f], int):
            d[f] = max(8, (d[f] // 8) * 8)
    return type(params)(**d)


def _halve_biggest(params, spec):
    d = params_dict(params)
    int_fields = [(f, v) for f, v in d.items()
                  if isinstance(v, int) and f not in ("stages", "block_dim") and v > 8]
    if not int_fields:
        if d.get("stages", 1) > 1:
            d["stages"] -= 1
        return type(params)(**d)
    f, v = max(int_fields, key=lambda kv: kv[1])
    smaller = [c for c in spec.get(f, []) if c < v]
    d[f] = max(smaller) if smaller else max(4, v // 2)
    return type(params)(**d)


class LLMProposer:
    def __init__(self, workload, client: LLMClient | None = None,
                 rng: random.Random | None = None):
        self.wl = workload
        self.client = client or LLMClient()
        self.rng = rng or random.Random(0)
        self.fallback = HeuristicProposer(workload, self.rng)
        self.calls = 0

    def propose(self, shape, parents, n, findings_by_parent=None,
                evidence_summary="", arch=None) -> list:
        if not self.client.available:
            return self.fallback.propose(shape, parents, n, findings_by_parent)
        prompt = self._prompt(shape, parents, findings_by_parent or {},
                              evidence_summary, n, arch)
        messages = [
            {"role": "system", "content": (
                "You are a kernel engineer writing CAKE-Ascend schedules. "
                "Reply with Python code blocks only.\n\n" + IR_GUIDE)},
            {"role": "user", "content": prompt},
        ]
        try:
            self.calls += 1
            text = self.client.chat(messages)
            sources = [b for b in extract_code_blocks(text) if "def kern" in b]
        except LLMError:
            sources = []
        if not sources:
            return self.fallback.propose(shape, parents, n, findings_by_parent)
        return [("source", src, "llm") for src in sources[:n]]

    def _prompt(self, shape, parents, findings, evidence_summary, n, arch):
        w = self.wl
        ios = ", ".join(f"{nm} {dt}{tuple(shp)} ({kind})" for nm, dt, shp, kind in w.io)
        lines = [f"Workload contract (stable authority — do not change semantics):",
                 f"  {w.description}", f"  shape: {shape}", f"  params: {ios}",
                 f"  tolerance: {w.tolerance}"]
        if arch is not None:
            lines.append("  " + arch_card(arch))
        lines.append("")
        lines.append("Best candidates so far:")
        if parents:
            for params, span, correct in parents[:3]:
                lines.append(f"- params={params_dict(params)} "
                             f"measured_span_cycles={span} correct={correct}")
        else:
            lines.append("- (none yet; the naive baseline is stages=1, fully synchronized)")
        lines.append("")
        lines.append("Recent failure evidence routed back:")
        if findings:
            for pid, fams in list(findings.items())[:3]:
                lines.append(f"- candidate {pid}: {', '.join(map(str, fams))}")
        else:
            lines.append("- none")
        if evidence_summary:
            lines.append(evidence_summary)
        lines.append("")
        lines.append(
            f"Write {n} structurally distinct candidate kernels as `def kern(m):` "
            "functions.\nVary tile sizes, pipeline stages, role structure, and the "
            "compute path.\nPrefer candidates that fix the failure evidence above. "
            "Each code block = one kernel.")
        return "\n".join(lines)
