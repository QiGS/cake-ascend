"""Localized diagnostics — the harness's externally visible contract.

Paper Table 1 categories, mapped 1:1:

  Program safety        (pre-compile gate)  sync/ordering/memory hazards
  Hardware conformance  (pre-compile gate)  resource/instruction/arch contracts
  Data consistency      (pre-compile gate)  dataflow + producer/consumer compat
  Schedule semantics    (pre-compile gate)  structural invariants of the schedule
  Numerical validation  (execution gate)    compiled output vs external oracle
  Performance analysis  (report)            cost estimate + bottleneck class
  Optimization guidance (hint)              non-blocking suggestions

A Finding always identifies the affected region (op id + loop tags +
resource name) and the violated contract class, and carries a repair hint —
"cheap analysis filters candidates before they consume GPU time".
"""
from __future__ import annotations

from dataclasses import dataclass, field

GATE = "gate"      # blocking pre-compile check
DYN = "dyn"        # execution-gate finding (from the simulator)
REPORT = "report"  # performance analysis, non-blocking
HINT = "hint"      # optimization guidance, non-blocking

CATEGORIES = (
    "program_safety",
    "hardware_conformance",
    "data_consistency",
    "schedule_semantics",
    "numerical_validation",
    "performance_analysis",
    "ir_construction",
)


@dataclass
class Finding:
    category: str
    severity: str                 # GATE | DYN | REPORT | HINT
    code: str                     # stable code, e.g. "HARDWARE.ub_capacity"
    message: str
    region: str = ""              # op#/resource/loop context localization
    hint: str = ""                # actionable repair target
    rule_id: str = ""             # which rule produced it ("" for built-ins)
    meta: dict = field(default_factory=dict)

    def format(self) -> str:
        head = f"[{self.severity}:{self.category}] {self.code}"
        loc = f" @ {self.region}" if self.region else ""
        body = f": {self.message}"
        hint = f"\n    repair: {self.hint}" if self.hint else ""
        return f"{head}{loc}{body}{hint}"

    def signature(self) -> str:
        """Stable signature for evidence recurrence counting."""
        return self.code


class GateRejected(Exception):
    """Raised by the harness when a candidate fails a blocking pre-compile gate.

    Carries the findings so the agent loop can route them as evidence."""

    def __init__(self, findings):
        self.findings = list(findings)
        text = "\n".join(f.format() for f in self.findings)
        super().__init__(f"candidate rejected by pre-compile gates:\n{text}")
