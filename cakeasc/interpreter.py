"""Deterministic dataflow simulator for Ascend-schedule programs.

This is the local stand-in for "GPU runtime + Compute Sanitizer + CUPTI"
(paper Fig. 1): it executes per-core programs across roles with event
choreography, produces outputs (numerics), raises *localized dynamic
findings* (races, OOB, overflow, deadlocks, uninitialized reads — the
"opaque runtime crash" evidence that the distiller turns into static
rules), and measures a deterministic span in cycles.

Execution model:
- Roles are instruction streams issued in program order; ops execute when
  their WaitFlags are satisfied (k-th commit of (event,stage) satisfies
  the k-th wait — FIFO handoffs, like SetFlag/WaitFlag queues).
- The scheduler is deterministic: round-robin over roles in declaration
  order, one op per turn. Since loops unrolled at trace time, control
  flow is value-independent, so timing is deterministic too.
- Numerics: storage quantization on every buffer write (see dtypes.py);
  fp32 accumulation emulated in fp64 with one rounding per tile step.
- Measured timing includes intrinsic per-op overheads (copy startup, flag
  latency) that the *analytic* cost model omits — the systematic gap that
  compiler-evolution calibration learns to close.

Race detection (the interesting part): every UB slot carries a monotone
content version. Reads record the version they consumed. Because a role's
stream executes in order, same-role hazards cannot occur; cross-role
hazards are detected post-hoc: for each (slot, reader-role), the sequence
of consumed versions written by *other* roles must be 0,1,2,... — a skip
means a producer overwrote a generation before its consumer read it
(missing back-pressure event), and the finding localizes both ops.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from operator import mul

from . import dtypes as dt
from .arch import Arch
from .diagnostics import DYN, Finding
from .ir import (
    Commit,
    Gm2L1,
    Gm2Ub,
    HardBarrier,
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
    shape_numel,
)
from .costmodel import measured_cycles

# dynamic finding codes (prefix families map to rule templates in distiller)
DYN_CODES = {
    "gm_oob": "SAFETY.gm_out_of_bounds",
    "ub_overflow": "SAFETY.ub_overflow",
    "slot_race": "SAFETY.slot_race",
    "deadlock": "SCHEDULE.deadlock",
    "matmul_unaligned": "HARDWARE.matmul_unaligned",
    "copy_unaligned": "HARDWARE.copy_unaligned",
    "uninit_acc": "DATA.matmul_uninit_acc",
    "uninit_read": "DATA.ub_uninit_read",
    "dead_write": "SAFETY.slot_dead_write",
}


class SimulatorAborted(Exception):
    """Fatal dynamic failure (deadlock) — carries findings."""

    def __init__(self, findings):
        self.findings = list(findings)
        super().__init__("; ".join(f.format() for f in self.findings))


@dataclass
class SimResult:
    ok: bool
    findings: list
    outputs: dict                    # param name -> flat list (final GM)
    span_cycles: float | None
    per_role_busy: dict = field(default_factory=dict)
    exec_count: int = 0


# --------------------------------------------------------------------------
# flat tensor helpers


def _strides(shape):
    s = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        s[i] = s[i + 1] * shape[i + 1]
    return s


def _extract(flat, tensor_shape, off, view_shape):
    """Row-major slice of `flat` at element offset `off` with `view_shape`."""
    tstr = _strides(tensor_shape)
    vstr = _strides(view_shape)
    n = shape_numel(view_shape)
    out = [0.0] * n
    if len(view_shape) == 2:
        r, c = view_shape
        base = off[0] * tstr[0] + off[1] * tstr[1] if len(off) == 2 else off[0]
        for i in range(r):
            row_base = base + i * tstr[0]
            src = row_base
            out[i * c:(i + 1) * c] = flat[src:src + c]
        return out
    for idx in range(n):
        pos = 0
        rem = idx
        for d in range(len(view_shape)):
            coord = rem // vstr[d]
            rem -= coord * vstr[d]
            pos += (off[d] + coord) * tstr[d]
        out[idx] = flat[pos]
    return out


def _deposit(flat, tensor_shape, off, view_shape, values):
    tstr = _strides(tensor_shape)
    if len(view_shape) == 2 and len(off) == 2:
        r, c = view_shape
        base = off[0] * tstr[0] + off[1] * tstr[1]
        for i in range(r):
            dst = base + i * tstr[0]
            flat[dst:dst + c] = values[i * c:(i + 1) * c]
        return
    vstr = _strides(view_shape)
    for idx, v in enumerate(values):
        pos = 0
        rem = idx
        for d in range(len(view_shape)):
            coord = rem // vstr[d]
            rem -= coord * vstr[d]
            pos += (off[d] + coord) * tstr[d]
        flat[pos] = v


def _in_bounds(tensor_shape, off, view_shape):
    return all(0 <= o and o + s <= p for o, s, p in zip(off, view_shape, tensor_shape))


# --------------------------------------------------------------------------
# core interpreter


class CoreInterpreter:
    def __init__(self, program: Program, arch: Arch, gm: dict, gm_shapes: dict,
                 execute_values: bool = True, duration_fn=None):
        self.prog = program
        self.arch = arch
        self.gm = gm
        self.gm_shapes = gm_shapes
        self.execute_values = execute_values
        self.duration_fn = duration_fn or (lambda op: measured_cycles(op, arch))
        self.findings = []
        self.aborted = False
        self.exec_seq = 0

        # role instruction streams (markers retained, skipped at execution)
        self.streams = {}
        self.role_order = []
        for op in program.ops:
            rname = op.role.name if op.role else ""
            if not rname:
                continue
            if rname not in self.streams:
                self.streams[rname] = []
                self.role_order.append(rname)
            self.streams[rname].append(op)

        # slot state (UB views)
        self.slot_data = {}        # (view.name, stage) -> list | None
        self.slot_writer = {}      # last write op (timing dep anchor)
        self.slot_version = {}
        self.slot_writes = {}      # (view.name, stage) -> [(version, op, seq, role_name)]
        self.slot_reads = {}       # (view.name, stage) -> [(version, op, seq, role_name)]
        # l0c state
        self.l0c_data = {}
        self.l0c_init = {}
        self.l0c_writer = {}
        self.l0c_consumed = {}     # l0c2ub executed since last chain start
        self.cross_role_l0c_reader = False

        # event state: (name, stage) -> [n_commit, n_wait, [commit ops]]
        self.evt = {}

        # timing
        self.role_end = {r: 0.0 for r in self.role_order}
        self.op_end = {}
        self.exec_log = []
        self.executed = set()
        self.busy = {}

        # cross-core GM write accounting: (param, start, end) element ranges
        self.gm_writes = []

        self._prepare()

    # ------------------------------------------------------------ prepare

    def _prepare(self):
        arch = self.arch
        prog = self.prog
        for pool in prog.ub_pools:
            capacity = arch.l1_bytes if pool.tier == "l1" else arch.ub_bytes
            tier = pool.tier.upper()
            if pool.size > capacity:
                self._dyn("ub_overflow",
                          f"{tier} pool '{pool.name}' is {pool.size} B; "
                          f"device {tier} is {capacity} B",
                          region=f"{pool.tier}:{pool.name}", fatal=True,
                          hint="shrink tiles or stages to fit on-chip memory")
        for v in prog.ub_views:
            if v.offset + v.total_bytes > v.pool.size:
                self._dyn("ub_overflow",
                          f"view '{v.name}' exceeds pool '{v.pool.name}' "
                          f"({v.offset}+{v.total_bytes} > {v.pool.size})",
                          region=f"view:{v.name}", fatal=True,
                          hint="give the view a disjoint in-pool range")
            for s in range(v.stages):
                key = (v.name, s)
                self.slot_data[key] = None
                self.slot_writer[key] = None
                self.slot_version[key] = -1
                self.slot_writes[key] = []
                self.slot_reads[key] = []
        for r in prog.l0c_regions:
            self.l0c_data[r.name] = None
            self.l0c_init[r.name] = False
            self.l0c_writer[r.name] = None
            self.l0c_consumed[r.name] = False
        for e in prog.events:
            for s in range(e.pipeline.stages if e.pipeline else 1):
                self.evt[(e.name, s)] = [0, 0, []]
        cube_roles = {r.name for r in prog.roles if r.kind == "CUBE"}
        for op in prog.ops:
            if isinstance(op, L0c2Ub) and op.role.name not in cube_roles:
                self.cross_role_l0c_reader = True

    def _dyn(self, family: str, message: str, region: str = "", hint: str = "",
             fatal: bool = True):
        self.findings.append(Finding(
            category=_dyn_category(family), severity=DYN, code=DYN_CODES[family],
            message=message, region=region, hint=hint,
            meta={"fatal": fatal, "family": family}))

    # -------------------------------------------------------------- run

    def run(self) -> float | None:
        pcs = {r: 0 for r in self.role_order}
        streams = self.streams
        while True:
            progressed = False
            for rname in self.role_order:
                stream = streams[rname]
                while pcs[rname] < len(stream) and isinstance(stream[pcs[rname]], LoopMarker):
                    pcs[rname] += 1
                    progressed = True
                if pcs[rname] >= len(stream):
                    continue
                op = stream[pcs[rname]]
                if isinstance(op, Wait):
                    st = self.evt[(op.event.name, op.stage)]
                    if st[0] <= st[1]:      # no committed ticket yet
                        continue
                self._execute(op)
                pcs[rname] += 1
                progressed = True
            if not progressed:
                break
        blocked = []
        for rname in self.role_order:
            stream = streams[rname]
            i = pcs[rname]
            while i < len(stream) and isinstance(stream[i], LoopMarker):
                i += 1
            if i < len(stream):
                blocked.append(stream[i])
        if blocked and not self.aborted:
            self._deadlock(blocked)
            return None
        self._analyze_slot_versions()
        return max(self.role_end.values()) if self.role_end else 0.0

    def _deadlock(self, blocked):
        parts = []
        for op in blocked:
            if isinstance(op, Wait):
                st = self.evt[(op.event.name, op.stage)]
                parts.append(f"{op.locate()} waits on '{op.event.name}' stage {op.stage} "
                             f"({st[0]} committed / {st[1]} waited)")
            else:
                parts.append(op.locate())
        self._dyn("deadlock",
                  f"schedule deadlocks; blocked: {'; '.join(parts)}",
                  region=blocked[0].locate() if blocked else "",
                  hint="unmatched SetFlag/WaitFlag choreography — every wait needs a "
                       "committed producer ticket on the same (event, stage)")
        self.aborted = True

    # ---------------------------------------------------------- execute

    def _execute(self, op):
        start = self.role_end.get(op.role.name, 0.0)
        deps = []

        if isinstance(op, Wait):
            st = self.evt[(op.event.name, op.stage)]
            commit_op = st[2][st[1]] if st[1] < len(st[2]) else None
            if commit_op is not None:
                deps.append(self._end_of(commit_op))
            st[1] += 1
        elif isinstance(op, Commit):
            st = self.evt[(op.event.name, op.stage)]
            st[0] += 1
            st[2].append(op)
        elif isinstance(op, HardBarrier):
            start = max(self.role_end.values())
        else:
            for (view, stage) in _reads_of(op):
                w = self.slot_writer.get((view.name, stage))
                if w is not None:
                    deps.append(self._end_of(w))
            for r in _l0c_reads_of(op):
                w = self.l0c_writer.get(r.name)
                if w is not None:
                    deps.append(self._end_of(w))

        if deps:
            start = max(start, max(deps))
        dur = self.duration_fn(op)
        end = start + dur
        self.role_end[op.role.name] = end
        self.op_end[op.id] = end
        self.exec_log.append((op, start, end))
        self.executed.add(op.id)
        self.exec_seq += 1
        self.busy[op.kind] = self.busy.get(op.kind, 0.0) + dur

        if self.execute_values and not isinstance(op, (Commit, Wait, HardBarrier, LoopMarker)):
            self._values(op)

    def _end_of(self, op) -> float:
        return self.op_end.get(op.id, self.role_end.get(op.role.name, 0.0))

    # ---------------------------------------------------- race accounting

    def _record_read(self, view, stage, op):
        key = (view.name, stage)
        self.slot_reads[key].append((self.slot_version[key], op, self.exec_seq,
                                     op.role.name))

    def _record_write(self, view, stage, op):
        key = (view.name, stage)
        self.slot_version[key] += 1
        self.slot_writes[key].append((self.slot_version[key], op, self.exec_seq,
                                      op.role.name))

    def _analyze_slot_versions(self):
        """Cross-role version-skip analysis (see module docstring)."""
        for key, writes in self.slot_writes.items():
            reads = self.slot_reads[key]
            reader_roles = {rn for (_v, _o, _s, rn) in reads}
            for rr in reader_roles:
                cross_versions = [v for (v, _o, _s, wn) in writes if wn != rr]
                if not cross_versions:
                    continue
                raw = [(v, o) for (v, o, _s, rn) in reads if rn == rr]
                # collapse consecutive re-reads of the same generation (legal)
                seq = []
                for v, o in raw:
                    if not seq or seq[-1][0] != v:
                        seq.append((v, o))
                for k, (v, op) in enumerate(seq):
                    if v > k:
                        # reader was owed cross-role generation k but content was
                        # already at v: generations k..v-1 were overwritten first
                        overwrite = next((w for (wv, w, _s, wn) in writes
                                          if wv == v and wn != rr), None)
                        self._dyn(
                            "slot_race",
                            f"slot '{key[0]}[{key[1]}]': reader '{rr}' at {op.locate()} "
                            f"consumed generation {v} but was owed {k}; "
                            f"producer overwrote it at "
                            f"{overwrite.locate() if overwrite else '?'}",
                            region=op.locate(),
                            hint=(f"add/keep a back-pressure event (prod='{rr}', "
                                  f"cons='{overwrite.role.name if overwrite else '?'}') "
                                  f"and wait() it before overwriting slot {key[1]}"))
                        break
                # dead cross-role writes: generations never read by a role that
                # reads later generations (non-fatal: waste, not corruption)
                read_set = {v for (v, _o) in seq}
                for wv, w, _s, wn in writes:
                    if wn != rr and wv not in read_set and any(v > wv for v in read_set):
                        self._dyn("dead_write",
                                  f"slot '{key[0]}[{key[1]}]': generation {wv} written at "
                                  f"{w.locate()} was never read (overwritten before use)",
                                  region=w.locate(), fatal=False,
                                  hint="dead load — remove it or fix the consumer loop")
                        break

    # ---------------------------------------------------------- numerics

    def _values(self, op):
        try:
            if isinstance(op, (Gm2Ub, Gm2L1)):
                self._do_gm2ub(op)
            elif isinstance(op, Ub2Gm):
                self._do_ub2gm(op)
            elif isinstance(op, L0c2Ub):
                self._do_l0c2ub(op)
            elif isinstance(op, Matmul):
                self._do_matmul(op)
            elif isinstance(op, VBinary):
                self._do_vbinary(op)
            elif isinstance(op, VUnary):
                self._do_vunary(op)
            elif isinstance(op, VReduce):
                self._do_vreduce(op)
            elif isinstance(op, VArgmin):
                self._do_vargmin(op)
            elif isinstance(op, VTranspose):
                self._do_vtranspose(op)
            elif isinstance(op, VBcast):
                self._do_vbcast(op)
            elif isinstance(op, VCast):
                self._do_vcast(op)
        except SimulatorAborted:
            raise
        except Exception as e:  # pragma: no cover — defensive localization
            self._dyn("deadlock", f"internal simulator error at {op.locate()}: {e}",
                      region=op.locate())

    def _slot(self, view, stage, reader_op):
        key = (view.name, stage)
        if self.slot_data[key] is None:
            self._dyn("uninit_read",
                      f"view '{view.name}' slot {stage} read at {reader_op.locate()} "
                      f"before any producer wrote it",
                      region=reader_op.locate(),
                      hint="missing gm2ub into this slot, or a wait on the wrong (event, stage)")
            self._record_read(view, stage, reader_op)
            return [0.0] * shape_numel(view.shape)
        self._record_read(view, stage, reader_op)
        return self.slot_data[key]

    def _write_slot(self, view, stage, values, op):
        key = (view.name, stage)
        vals = dt.quantize_list(values, view.dtype)
        self.slot_data[key] = vals
        self.slot_writer[key] = op
        self._record_write(view, stage, op)

    def _do_gm2ub(self, op):
        """Shared numerics for GM->UB and GM->L1 staging copies."""
        src_shape = self.gm_shapes[op.src.name]
        if len(op.gm_off) != len(src_shape):
            return
        if not _in_bounds(src_shape, op.gm_off, op.dst.shape):
            bad = [f"dim {d}: [{o}, {o + s}) vs {p}"
                   for d, (o, s, p) in enumerate(zip(op.gm_off, op.dst.shape, src_shape))
                   if o < 0 or o + s > p]
            self._dyn("gm_oob",
                      f"{op.src.name} access out of bounds ({'; '.join(bad)})",
                      region=op.locate(),
                      hint="shrink the tile or fix the per-core partition arithmetic")
            return
        if self._copy_unaligned(op.dst.shape, op.gm_off, op.dst.dtype):
            self._dyn("copy_unaligned",
                      f"DataCopy {op.src.name}->{op.dst.name} violates "
                      f"{self.arch.alignment_bytes}B alignment",
                      region=op.locate(),
                      hint=f"make the contiguous dim and offsets multiples of "
                           f"{self.arch.alignment_bytes // dt.bytes_of(op.dst.dtype)} elements")
        flat = self.gm[op.src.name]
        vals = _extract(flat, src_shape, op.gm_off, op.dst.shape)
        self._write_slot(op.dst, op.stage, vals, op)

    def _do_ub2gm(self, op: Ub2Gm):
        dst_shape = self.gm_shapes[op.dst.name]
        if len(op.gm_off) != len(dst_shape) or \
                not _in_bounds(dst_shape, op.gm_off, op.src.shape):
            self._dyn("gm_oob",
                      f"{op.dst.name} store out of bounds at {op.locate()}",
                      region=op.locate(),
                      hint="shrink the tile or fix the per-core partition arithmetic")
            return
        # record written element ranges for cross-core aliasing analysis
        strides = _strides(dst_shape)
        shape, off = op.src.shape, op.gm_off
        if len(shape) == 2:
            base = off[0] * strides[0] + off[1] * strides[1]
            for i in range(shape[0]):
                start = base + i * strides[0]
                self.gm_writes.append((op.dst.name, start, start + shape[1]))
        else:
            start = off[0] * strides[0]
            self.gm_writes.append((op.dst.name, start, start + shape_numel(shape)))
        if self._copy_unaligned(op.src.shape, op.gm_off, op.src.dtype):
            self._dyn("copy_unaligned",
                      f"DataCopy {op.src.name}->{op.dst.name} violates "
                      f"{self.arch.alignment_bytes}B alignment",
                      region=op.locate(),
                      hint="align the contiguous dimension and GM offset")
        vals = self._slot(op.src, op.stage, op)
        vals = dt.quantize_list(vals, op.dst.dtype)
        _deposit(self.gm[op.dst.name], dst_shape, op.gm_off, op.src.shape, vals)

    def _copy_unaligned(self, shape, off, dtype) -> bool:
        al = self.arch.alignment_bytes
        inner = shape[-1] * dt.bytes_of(dtype)
        return inner % al != 0 or (off[-1] * dt.bytes_of(dtype)) % al != 0

    def _do_l0c2ub(self, op: L0c2Ub):
        if not self.l0c_init[op.src.name]:
            self._dyn("uninit_read",
                      f"l0c '{op.src.name}' copied out at {op.locate()} before any matmul "
                      f"wrote it", region=op.locate(),
                      hint="matmul into the accumulator before l0c2ub")
            vals = [0.0] * shape_numel(op.src.shape)
        else:
            vals = self.l0c_data[op.src.name]
        self.l0c_writer[op.src.name] = op
        self.l0c_consumed[op.src.name] = True
        self._write_slot(op.dst, op.stage, vals, op)

    def _do_matmul(self, op: Matmul):
        a = self._slot(op.a, op.a_stage, op)
        b = self._slot(op.b, op.b_stage, op)
        m, k = op.a.shape
        k2, n = op.b.shape
        if m % self.arch.cube_align or n % self.arch.cube_align or k % self.arch.cube_align:
            self._dyn("matmul_unaligned",
                      f"matmul ({m},{k})x({k2},{n}) dims not multiple of "
                      f"{self.arch.cube_align} at {op.locate()}",
                      region=op.locate(),
                      hint=f"tile M/N/K in multiples of {self.arch.cube_align}")
        if not op.clear and not self.l0c_init[op.acc.name]:
            self._dyn("uninit_acc",
                      f"matmul at {op.locate()} accumulates into '{op.acc.name}' "
                      f"without clear=True first",
                      region=op.locate(),
                      hint="set clear=True on the first K-step of each accumulation chain")
        if op.clear and self.l0c_init[op.acc.name] and not self.l0c_consumed[op.acc.name] \
                and self.cross_role_l0c_reader:
            self._dyn("slot_race",
                      f"l0c '{op.acc.name}' re-initialized at {op.locate()} before the "
                      f"cross-role epilogue consumed the previous result",
                      region=op.locate(),
                      hint="wait on an epilogue-done event before starting the next "
                           "accumulation chain")
        bt_cols = [b[j::n] for j in range(n)]
        acc = self.l0c_data[op.acc.name] if (not op.clear and self.l0c_init[op.acc.name]) \
            else [0.0] * (m * n)
        for i in range(m):
            arow = a[i * k:(i + 1) * k]
            base = i * n
            for j in range(n):
                acc[base + j] += sum(map(mul, arow, bt_cols[j]))
        self.l0c_data[op.acc.name] = dt.quantize_list(acc, dt.FP32)
        self.l0c_init[op.acc.name] = True
        self.l0c_writer[op.acc.name] = op
        if op.clear:
            self.l0c_consumed[op.acc.name] = False

    def _do_vbinary(self, op: VBinary):
        x = self._slot(op.x, op.x_stage, op)
        if op.y is not None:
            y = self._slot(op.y, op.y_stage, op)
            fn = _BINARY_FNS[op.op]
            vals = [fn(a, b) for a, b in zip(x, y)]
        else:
            k = op.y_scalar
            fn = _BINARY_FNS[op.op]
            vals = [fn(a, k) for a in x]
        self._write_slot(op.dst, op.dst_stage, vals, op)

    def _do_vunary(self, op: VUnary):
        x = self._slot(op.x, op.x_stage, op)
        fn = _UNARY_FNS[op.op]
        self._write_slot(op.dst, op.dst_stage, [fn(a) for a in x], op)

    def _do_vreduce(self, op: VReduce):
        x = self._slot(op.x, op.x_stage, op)
        r, c = op.x.shape
        fn = _REDUCE_FNS[op.op]
        vals = [fn(x[i * c:(i + 1) * c]) for i in range(r)]
        self._write_slot(op.dst, op.dst_stage, vals, op)

    def _do_vargmin(self, op: VArgmin):
        x = self._slot(op.x, op.x_stage, op)
        r, c = op.x.shape
        vals = []
        for i in range(r):
            row = x[i * c:(i + 1) * c]
            best = 0
            for j in range(1, c):
                if row[j] < row[best]:
                    best = j
            vals.append(float(best))
        self._write_slot(op.dst, op.dst_stage, vals, op)

    def _do_vtranspose(self, op: VTranspose):
        x = self._slot(op.x, op.x_stage, op)
        r, c = op.x.shape
        # (r,c) row-major -> (c,r) row-major
        vals = [x[j * c + i] for i in range(c) for j in range(r)]
        self._write_slot(op.dst, op.dst_stage, vals, op)

    def _do_vbcast(self, op: VBcast):
        x = self._slot(op.x, op.x_stage, op)
        r, c = op.dst.shape
        if op.mode == "row":
            vals = [x[i] for i in range(r) for _ in range(c)]
        else:
            vals = [x[j] for _ in range(r) for j in range(c)]
        self._write_slot(op.dst, op.dst_stage, vals, op)

    def _do_vcast(self, op: VCast):
        x = self._slot(op.x, op.x_stage, op)
        self._write_slot(op.dst, op.dst_stage, x, op)


def _safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


def _safe_sqrt(x):
    return math.sqrt(x) if x >= 0 else float("nan")


_BINARY_FNS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "min": min,
    "max": max,
}
_UNARY_FNS = {
    "neg": lambda a: -a,
    "abs": abs,
    "sqrt": _safe_sqrt,
    "exp": _safe_exp,
    "copy": lambda a: a,
}
_REDUCE_FNS = {
    "sum": sum,
    "max": max,
    "min": min,
}


def _dyn_category(family):
    if family in ("gm_oob", "ub_overflow", "slot_race", "dead_write"):
        return "program_safety"
    if family == "deadlock":
        return "schedule_semantics"
    if family in ("matmul_unaligned", "copy_unaligned"):
        return "hardware_conformance"
    return "data_consistency"


# --------------------------------------------------------------------------
# access classification


def _reads_of(op):
    out = []
    if isinstance(op, Ub2Gm):
        out.append((op.src, op.stage))
    elif isinstance(op, Matmul):
        out.append((op.a, op.a_stage))
        out.append((op.b, op.b_stage))
    elif isinstance(op, VBinary):
        out.append((op.x, op.x_stage))
        if op.y is not None:
            out.append((op.y, op.y_stage))
    elif isinstance(op, (VUnary, VReduce, VArgmin, VTranspose, VCast, VBcast)):
        out.append((op.x, op.x_stage))
    return out


def _writes_of(op):
    out = []
    if isinstance(op, (Gm2Ub, Gm2L1)):
        out.append((op.dst, op.stage))
    elif isinstance(op, L0c2Ub):
        out.append((op.dst, op.stage))
    elif isinstance(op, (VBinary, VUnary, VReduce, VArgmin, VTranspose, VCast, VBcast)):
        out.append((op.dst, op.dst_stage))
    return out


def _l0c_reads_of(op):
    return [op.src] if isinstance(op, L0c2Ub) else []


# --------------------------------------------------------------------------
# top-level: run all cores


def run_simulation(programs, arch: Arch, inputs: dict, input_shapes: dict,
                   execute_values: bool = True) -> SimResult:
    """Execute per-core programs over shared GM; returns outputs + findings + span."""
    gm = {name: list(vals) for name, vals in inputs.items()}
    findings = []
    span = 0.0
    span_dead = False
    busy_total = {}
    exec_count = 0
    fatal = False
    per_core_writes = []            # (core_idx, [(param, start, end), ...])
    for core, prog in enumerate(programs):
        interp = CoreInterpreter(prog, arch, gm, input_shapes, execute_values=execute_values)
        try:
            core_span = interp.run()
        except SimulatorAborted as e:
            findings.extend(e.findings)
            fatal = True
            span_dead = True
            per_core_writes.append((core, list(interp.gm_writes)))
            continue
        findings.extend(interp.findings)
        if any(f.meta.get("fatal", True) for f in interp.findings):
            fatal = True
        if core_span is None:
            span_dead = True
        else:
            span = max(span, core_span)
        per_core_writes.append((core, list(interp.gm_writes)))
        for k, v in interp.busy.items():
            busy_total[k] = busy_total.get(k, 0.0) + v
        exec_count += len(interp.exec_log)
    findings.extend(_cross_core_overlaps(per_core_writes))
    if any(f.meta.get("fatal", True) for f in findings):
        fatal = True
    if not programs:
        return SimResult(ok=not fatal, findings=findings, outputs=gm,
                         span_cycles=None, per_role_busy=busy_total)
    launch = arch.launch_overhead_us * 1e-6 / arch.seconds_per_cycle()
    return SimResult(ok=not fatal, findings=findings, outputs=gm,
                     span_cycles=None if span_dead else span + launch,
                     per_role_busy=busy_total, exec_count=exec_count)


def _cross_core_overlaps(per_core_writes) -> list:
    """Detect two cores writing overlapping GM ranges (write-write aliasing).

    Per-core execution is sequential, so same-core ordering is well defined;
    across cores an overlap is a genuine race (last writer wins silently).
    """
    out = []
    reported = set()
    for i in range(len(per_core_writes)):
        for j in range(i + 1, len(per_core_writes)):
            ci, wi = per_core_writes[i]
            cj, wj = per_core_writes[j]
            for (pi, si, ei) in wi:
                for (pj, sj, ej) in wj:
                    if pi != pj or si >= ej or sj >= ei:
                        continue
                    key = (pi, max(si, sj), min(ei, ej))
                    if key in reported:
                        continue
                    reported.add(key)
                    out.append(Finding(
                        category="program_safety", severity=DYN,
                        code="SAFETY.gm_write_overlap",
                        message=(f"cores {ci} and {cj} both write {pj} elements "
                                 f"[{max(si, sj)}, {min(ei, ej)}) — cross-core "
                                 f"write overlap (last writer wins)"),
                        region=f"param:{pj}",
                        hint="partition the output across cores disjointly "
                             "(per-core tile ranges must not intersect)",
                        meta={"fatal": True, "family": "gm_write_overlap"}))
    return out
