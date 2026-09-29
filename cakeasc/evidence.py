"""Evidence store + recurrence detection (paper Sec. 3.2 / Fig. 4).

Every evaluation routes its evidence here: gate findings, dynamic findings,
correctness mismatches, cost mispredictions, IR gaps. Recurring or
high-cost failure modes become distillation targets:

  an opaque runtime crash   -> a verifier rule
  a repeated illegal pattern -> a static check
  a systematic misprediction -> a cost-model calibration target
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class EvidenceItem:
    family: str            # gm_oob | ub_overflow | slot_race | ... | cost_mispredict | ir_gap
    code: str
    region: str = ""
    detail: str = ""
    iteration: int = -1
    candidate: str = ""


@dataclass
class CostSample:
    iteration: int
    candidate: str
    predicted: float                 # analytic span (cycles, no learned overheads)
    measured: float | None           # simulator span (cycles)
    class_counts: dict = field(default_factory=dict)   # cost class -> op count


class EvidenceStore:
    def __init__(self, distill_threshold: int = 3, mispredict_rel: float = 0.15):
        self.items: list[EvidenceItem] = []
        self.by_family: dict[str, list[EvidenceItem]] = defaultdict(list)
        self.cost_samples: list[CostSample] = []
        self.distill_threshold = distill_threshold
        self.mispredict_rel = mispredict_rel
        self.distilled: set[str] = set()      # families already distilled

    # ------------------------------------------------------------- record

    def record(self, family: str, code: str, region: str = "", detail: str = "",
               iteration: int = -1, candidate: str = "") -> EvidenceItem:
        item = EvidenceItem(family=family, code=code, region=region, detail=detail,
                            iteration=iteration, candidate=candidate)
        self.items.append(item)
        self.by_family[family].append(item)
        return item

    def record_findings(self, findings, iteration: int, candidate: str = ""):
        for f in findings:
            family = f.meta.get("family") or f.code
            self.record(family, f.code, region=f.region, detail=f.message,
                        iteration=iteration, candidate=candidate)

    def record_cost(self, sample: CostSample):
        self.cost_samples.append(sample)

    # ---------------------------------------------------------- recurrence

    def recurrence(self, family: str) -> int:
        return len(self.by_family.get(family, []))

    def pending_distillations(self) -> list[str]:
        """Families at/over threshold not yet distilled."""
        out = []
        for family, items in self.by_family.items():
            if family in self.distilled:
                continue
            if len(items) >= self.distill_threshold:
                out.append(family)
        if "cost_mispredict" not in self.distilled:
            if len(self.cost_samples) >= self.distill_threshold and \
                    self._mispredict_recurring():
                out.append("cost_mispredict")
        return out

    def _mispredict_recurring(self) -> bool:
        rels = [abs(s.predicted - s.measured) / s.measured
                for s in self.cost_samples
                if s.measured and s.measured > 0]
        if len(rels) < self.distill_threshold:
            return False
        rels.sort()
        return rels[len(rels) // 2] > self.mispredict_rel

    def mark_distilled(self, family: str):
        self.distilled.add(family)

    def families_summary(self) -> list[tuple[str, int, str]]:
        rows = [(f, len(v), "distilled" if f in self.distilled else "open")
                for f, v in sorted(self.by_family.items())]
        if self.cost_samples:
            rows.append(("cost_mispredict", len(self.cost_samples),
                         "distilled" if "cost_mispredict" in self.distilled else "open"))
        return rows
