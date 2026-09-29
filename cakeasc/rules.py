"""Evolvable verifier rule registry + built-in rule templates.

Faithful to the paper's compiler-evolution story: the harness starts with
only construction-time typing (builder) plus a minimal always-on set, and
*acquires* static rules when recurring dynamic failures are distilled into
them (see distiller.py). Rule templates live here; the registry decides
which are active. Every installed rule is corpus-test-gated before it can
reject candidates (paper: "a new verifier rule without corpus validation
can reject valid kernels").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import dtypes as dt
from .arch import Arch
from .diagnostics import Finding, GATE, HINT
from .ir import (
    Gm2Ub,
    L0c2Ub,
    LoopMarker,
    Matmul,
    Program,
    Ub2Gm,
    VArgmin,
    VBcast,
    VBinary,
    VCast,
    VReduce,
    VTranspose,
    VUnary,
    Wait,
    Commit,
    shape_numel,
)

# --------------------------------------------------------------------------
# rule protocol


class Rule:
    template: str = "?"
    category: str = "?"
    severity: str = GATE
    description: str = ""

    def __init__(self, rule_id: str, params: dict | None = None,
                 provenance: dict | None = None):
        self.rule_id = rule_id
        self.params = dict(params or {})
        self.provenance = dict(provenance or {})

    def check(self, program: Program, arch: Arch):
        raise NotImplementedError

    def finding(self, code: str, message: str, region: str, hint: str = "",
                severity: str | None = None, **meta) -> Finding:
        return Finding(category=self.category,
                       severity=severity or self.severity,
                       code=code, message=message, region=region,
                       hint=hint, rule_id=self.rule_id, meta=meta)

    def to_json(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "template": self.template,
            "params": self.params,
            "provenance": self.provenance,
        }


# --------------------------------------------------------------------------
# always-on baseline rules


class DtypeSupport(Rule):
    template = "dtype_support"
    category = "hardware_conformance"
    description = "views and params use dtypes the target supports"

    def check(self, program: Program, arch: Arch):
        out = []
        for v in program.ub_views:
            if v.dtype not in arch.supports:
                out.append(self.finding(
                    "HARDWARE.dtype_unsupported",
                    f"view '{v.name}' uses dtype {v.dtype} unsupported on {arch.name}",
                    region=f"view:{v.name}",
                    hint=f"use one of {sorted(arch.supports)}"))
        return out


class PipelineLimit(Rule):
    template = "pipeline_limit"
    category = "hardware_conformance"
    description = "pipeline depth within queue/bufferNum limit"

    def check(self, program: Program, arch: Arch):
        out = []
        for p in program.pipelines:
            if p.stages > arch.max_buffer_num:
                out.append(self.finding(
                    "HARDWARE.pipeline_too_deep",
                    f"pipeline '{p.name}' has {p.stages} stages; target supports {arch.max_buffer_num}",
                    region=f"pipe:{p.name}",
                    hint=f"reduce stages to <= {arch.max_buffer_num}"))
        return out


class EventBalance(Rule):
    template = "event_balance"
    category = "schedule_semantics"
    description = "every wait has a matching commit (starvation gates; over-signaling hints)"

    def check(self, program: Program, arch: Arch):
        out = []
        commits, waits = {}, {}
        for o in program.ops:
            if isinstance(o, Commit):
                commits[(o.event.name, o.stage)] = commits.get((o.event.name, o.stage), 0) + 1
            elif isinstance(o, Wait):
                waits[(o.event.name, o.stage)] = waits.get((o.event.name, o.stage), 0) + 1
        for key in sorted(set(commits) | set(waits)):
            c, w = commits.get(key, 0), waits.get(key, 0)
            if w > c:
                out.append(self.finding(
                    "SCHEDULE.event_starved",
                    f"event '{key[0]}' stage {key[1]}: {w} wait(s) but only {c} commit(s) — "
                    f"the schedule deadlocks",
                    region=f"event:{key[0]}",
                    hint="every WaitFlag needs exactly one matching SetFlag on the same "
                         "(event, stage)"))
        for e in program.events:
            if not any(k[0] == e.name for k in commits):
                out.append(self.finding(
                    "SCHEDULE.event_unused", f"event '{e.name}' is declared but never committed",
                    region=f"event:{e.name}",
                    hint="remove it or wire it into the choreography",
                    severity=HINT))       # wasteful, not incorrect (e.g. idle core)
        return out


# --------------------------------------------------------------------------
# distiller-installable rule templates


class UbCapacity(Rule):
    template = "ub_capacity"
    category = "hardware_conformance"
    description = "UB pool fits on-chip; views inside pool; 32B-aligned offsets"

    def check(self, program: Program, arch: Arch):
        out = []
        for pool in program.ub_pools:
            if pool.size > arch.ub_bytes:
                out.append(self.finding(
                    "HARDWARE.ub_pool_oversized",
                    f"UB pool '{pool.name}' is {pool.size} B; target has {arch.ub_bytes} B",
                    region=f"ub:{pool.name}",
                    hint="shrink tiles or reduce pipeline stages"))
        for v in program.ub_views:
            if v.offset % arch.alignment_bytes != 0:
                out.append(self.finding(
                    "HARDWARE.ub_offset_unaligned",
                    f"view '{v.name}' offset {v.offset} B is not {arch.alignment_bytes}B-aligned",
                    region=f"view:{v.name}",
                    hint=f"round offsets up to multiples of {arch.alignment_bytes}"))
            if v.offset + v.total_bytes > v.pool.size:
                out.append(self.finding(
                    "HARDWARE.ub_view_out_of_pool",
                    f"view '{v.name}' needs {v.offset}+{v.total_bytes} B but pool '{v.pool.name}' holds {v.pool.size} B",
                    region=f"view:{v.name}",
                    hint="raise pool size within UB capacity or shrink the tile"))
        return out


class L0cCapacity(Rule):
    template = "l0c_capacity"
    category = "hardware_conformance"
    description = "L0C region fits and is cube-aligned"

    def check(self, program: Program, arch: Arch):
        out = []
        for r in program.l0c_regions:
            n = shape_numel(r.shape) * dt.bytes_of(r.dtype)
            if n > arch.l0c_bytes:
                out.append(self.finding(
                    "HARDWARE.l0c_oversized",
                    f"l0c '{r.name}' needs {n} B; target has {arch.l0c_bytes} B",
                    region=f"l0c:{r.name}", hint="shrink the M/N tile"))
            for d in r.shape[:2]:
                if d % arch.cube_align != 0:
                    out.append(self.finding(
                        "HARDWARE.l0c_unaligned",
                        f"l0c '{r.name}' shape {r.shape}: dims must be multiples of {arch.cube_align}",
                        region=f"l0c:{r.name}",
                        hint=f"pad the tile to a multiple of {arch.cube_align}"))
        return out


class MatmulAlignment(Rule):
    template = "matmul_alignment"
    category = "hardware_conformance"
    description = "cube operands M/N/K aligned"

    def check(self, program: Program, arch: Arch):
        out = []
        seen = set()
        for o in program.ops:
            if not isinstance(o, Matmul):
                continue
            m, k = o.a.shape
            k2, n = o.b.shape
            key = (o.a.name, o.b.name, o.acc.name)
            if key in seen:
                continue
            seen.add(key)
            bad = [f"{lbl}={val}" for lbl, val in (("M", m), ("K", k), ("N", n))
                   if val % arch.cube_align != 0]
            if bad:
                out.append(self.finding(
                    "HARDWARE.matmul_unaligned",
                    f"matmul {o.a.name}({m},{k})x{o.b.name}({k2},{n}): {', '.join(bad)} not multiple of {arch.cube_align}",
                    region=o.locate(),
                    hint=f"choose tile dims divisible by {arch.cube_align} (bf16 32B lanes)"))
        return out


class DataCopyAlignment(Rule):
    template = "data_copy_alignment"
    category = "hardware_conformance"
    description = "DataCopy inner dimension and offsets 32B-aligned"

    def check(self, program: Program, arch: Arch):
        out = []
        for o in program.ops:
            if isinstance(o, Gm2Ub):
                shape, off, dtype, pname = o.dst.shape, o.gm_off, o.dst.dtype, o.src.name
            elif isinstance(o, Ub2Gm):
                shape, off, dtype, pname = o.src.shape, o.gm_off, o.src.dtype, o.dst.name
            else:
                continue
            inner_bytes = shape[-1] * dt.bytes_of(dtype)
            off_bytes = off[-1] * dt.bytes_of(dtype)
            if inner_bytes % arch.alignment_bytes != 0:
                out.append(self.finding(
                    "HARDWARE.copy_inner_unaligned",
                    f"copy {pname}: inner dim {shape[-1]} x {dtype} = {inner_bytes} B not {arch.alignment_bytes}B-aligned",
                    region=o.locate(),
                    hint=f"make the contiguous dimension a multiple of {arch.alignment_bytes // dt.bytes_of(dtype)} elements"))
            elif off_bytes % arch.alignment_bytes != 0:
                out.append(self.finding(
                    "HARDWARE.copy_offset_unaligned",
                    f"copy {pname}: offset {off[-1]} x {dtype} = {off_bytes} B not {arch.alignment_bytes}B-aligned",
                    region=o.locate(),
                    hint=f"advance offsets in multiples of {arch.alignment_bytes // dt.bytes_of(dtype)} elements"))
        return out


class GmBounds(Rule):
    template = "gm_bounds"
    category = "program_safety"
    description = "GM accesses in bounds (exact: programs are unrolled)"

    def check(self, program: Program, arch: Arch):
        out = []
        for o in program.ops:
            if isinstance(o, Gm2Ub):
                shape, off, param = o.dst.shape, o.gm_off, o.src
            elif isinstance(o, Ub2Gm):
                shape, off, param = o.src.shape, o.gm_off, o.dst
            else:
                continue
            for d, (o_, s_, p_) in enumerate(zip(off, shape, param.shape)):
                if o_ < 0 or o_ + s_ > p_:
                    out.append(self.finding(
                        "SAFETY.gm_out_of_bounds",
                        f"{param.name}: dim {d} access [{o_}, {o_ + s_}) exceeds extent {p_}",
                        region=o.locate(),
                        hint="clamp the tile partition or shrink the tile"))
                    break
        return out


class ViewOverlap(Rule):
    template = "view_overlap"
    category = "data_consistency"
    description = "views in a pool do not overlap"

    def check(self, program: Program, arch: Arch):
        out = []
        for pool in program.ub_pools:
            views = [v for v in program.ub_views if v.pool is pool]
            for i in range(len(views)):
                for j in range(i + 1, len(views)):
                    a, b = views[i].byte_range(), views[j].byte_range()
                    if a[0] < b[1] and b[0] < a[1]:
                        out.append(self.finding(
                            "DATA.view_overlap",
                            f"views '{views[i].name}' {a} and '{views[j].name}' {b} overlap in pool '{pool.name}'",
                            region=f"view:{views[j].name}",
                            hint="give each view a disjoint offset range"))
        return out


class SlotBackPressure(Rule):
    template = "slot_back_pressure"
    category = "program_safety"
    description = "slot rewrites are gated by consumer back-pressure events"

    def check(self, program: Program, arch: Arch):
        out = []
        writes, reads = _slot_accesses(program)
        for (vname, stage), w_ops in writes.items():
            if len(w_ops) < 2:
                continue
            writer = w_ops[0].role
            reader_roles = {r.role for r in reads.get((vname, stage), [])}
            reader_roles.discard(writer)
            if not reader_roles:
                continue        # same-role consumers are program-order safe
            for k in range(1, len(w_ops)):
                w_prev, w_cur = w_ops[k - 1], w_ops[k]
                if _writer_waited_between(program, w_prev, w_cur, reader_roles):
                    continue
                out.append(self.finding(
                    "SAFETY.slot_overwrite_hazard",
                    f"view '{vname}' slot {stage} rewritten at {w_cur.locate()} without "
                    f"waiting for consumer ({', '.join(sorted(r.name for r in reader_roles))}) "
                    f"to release the previous tile",
                    region=w_cur.locate(),
                    hint=f"declare event(prod=consumer, cons={writer.name}) and wait() it "
                         f"before rewriting slot {stage}"))
                break
        return out


class MatmulFirstClear(Rule):
    template = "matmul_first_clear"
    category = "data_consistency"
    description = "first accumulation into an l0c region initializes it"

    def check(self, program: Program, arch: Arch):
        out = []
        first = {}
        for o in program.ops:
            if isinstance(o, Matmul) and o.acc.name not in first:
                first[o.acc.name] = o
        for name, o in first.items():
            if not o.clear:
                out.append(self.finding(
                    "DATA.matmul_first_use_not_cleared",
                    f"first matmul into l0c '{name}' at {o.locate()} has clear=False; accumulator is uninitialized",
                    region=o.locate(),
                    hint="set clear=True on the first K-step of each accumulation chain"))
        return out


# --------------------------------------------------------------------------
# helpers


def _slot_accesses(program: Program):
    """Map (view.name, stage) -> ([write ops], [read ops]) in program order."""
    writes, reads = {}, {}

    def rec_write(op, view, stage):
        writes.setdefault((view.name, stage), []).append(op)

    def rec_read(op, view, stage):
        reads.setdefault((view.name, stage), []).append(op)

    for o in program.ops:
        if isinstance(o, (LoopMarker, Commit, Wait)):
            continue
        if isinstance(o, Gm2Ub):
            rec_write(o, o.dst, o.stage)
        elif isinstance(o, Ub2Gm):
            rec_read(o, o.src, o.stage)
        elif isinstance(o, L0c2Ub):
            rec_write(o, o.dst, o.stage)
        elif isinstance(o, Matmul):
            rec_read(o, o.a, o.a_stage)
            rec_read(o, o.b, o.b_stage)
        elif isinstance(o, (VBinary,)):
            rec_write(o, o.dst, o.dst_stage)
            rec_read(o, o.x, o.x_stage)
            if o.y is not None:
                rec_read(o, o.y, o.y_stage)
        elif isinstance(o, (VUnary, VReduce, VArgmin, VTranspose, VCast)):
            rec_write(o, o.dst, o.dst_stage)
            rec_read(o, o.x, o.x_stage)
        elif isinstance(o, VBcast):
            rec_write(o, o.dst, o.dst_stage)
            rec_read(o, o.x, o.x_stage)
    return writes, reads


def _writer_waited_between(program: Program, w_prev, w_cur, reader_roles) -> bool:
    """True if the writer waits, between two of its own writes, on an event
    produced by one of the slot's reader roles (back-pressure)."""
    for o in program.ops:
        if o.id <= w_prev.id or o.id >= w_cur.id:
            continue
        if isinstance(o, Wait) and o.role == w_cur.role:
            if o.event.prod in reader_roles:
                return True
    return False


TEMPLATES = {
    r.template: r
    for r in (UbCapacity, L0cCapacity, MatmulAlignment, DataCopyAlignment,
              GmBounds, ViewOverlap, SlotBackPressure, MatmulFirstClear)
}

ALWAYS_ON = (DtypeSupport, PipelineLimit, EventBalance)

ALL_TEMPLATE_CLASSES = dict(TEMPLATES)
for _r in ALWAYS_ON:
    ALL_TEMPLATE_CLASSES[_r.template] = _r


# --------------------------------------------------------------------------
# registry


@dataclass
class Registry:
    """Ordered set of active rules; itself a target of evolution."""

    rules: list = field(default_factory=list)

    @classmethod
    def baseline(cls) -> "Registry":
        reg = cls()
        for i, r in enumerate(ALWAYS_ON):
            reg.rules.append(r(rule_id=f"BUILTIN.{r.template}.0", provenance={"origin": "baseline"}))
        return reg

    def install(self, template: str, params: dict | None = None,
                provenance: dict | None = None) -> Rule:
        if template not in TEMPLATES:
            raise ValueError(f"unknown rule template {template!r}")
        n = sum(1 for r in self.rules if r.template == template)
        rule = TEMPLATES[template](
            rule_id=f"DISTILLED.{template}.{n}", params=params, provenance=provenance)
        self.rules.append(rule)
        return rule

    def has(self, template: str) -> bool:
        return any(r.template == template for r in self.rules)

    def check_all(self, program: Program, arch: Arch):
        findings = []
        for rule in self.rules:
            findings.extend(rule.check(program, arch))
        return findings

    def to_json(self) -> list:
        return [r.to_json() for r in self.rules]

    @classmethod
    def from_json(cls, data: list) -> "Registry":
        reg = cls()
        for item in data or []:
            tpl, rid = item.get("template"), item.get("rule_id", "?")
            if tpl in ALL_TEMPLATE_CLASSES:
                reg.rules.append(ALL_TEMPLATE_CLASSES[tpl](
                    rule_id=rid, params=item.get("params"),
                    provenance=item.get("provenance")))
        return reg
