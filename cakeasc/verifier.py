"""Verifier: run the rule registry over a candidate before any GPU time.

Returns localized findings; blocking gate failures reject the candidate
cheaply (paper Sec. 4: "filter them with IR construction checks, verifier
hard gates, and cost-model ranking before spending GPU time").
"""
from __future__ import annotations

from .arch import Arch
from .diagnostics import GATE, Finding, GateRejected
from .ir import Program


def verify(program: Program, arch: Arch, registry) -> list:
    """Run all active rules; returns findings (gates first)."""
    findings = registry.check_all(program, arch)
    findings.sort(key=lambda f: (f.severity != GATE, f.category, f.code))
    return findings


def verify_or_raise(program: Program, arch: Arch, registry) -> list:
    findings = verify(program, arch)
    gates = [f for f in findings if f.severity == GATE]
    if gates:
        raise GateRejected(gates)
    return findings


def verify_cores(programs, arch: Arch, registry) -> list:
    """Verify every per-core program; findings tagged with the core."""
    out = []
    for core, prog in enumerate(programs):
        for f in verify(prog, arch, registry):
            out.append(f)
            f.meta.setdefault("core", core)
    return out


def gates_of(findings) -> list:
    return [f for f in findings if f.severity == GATE]
