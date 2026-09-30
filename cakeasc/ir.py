"""Cake-IR for Ascend: typed op vocabulary + declarative resource model.

Mapping from the paper's NVIDIA concepts to the Ascend execution model:

  paper (CUDA)          ->  here (AscendC concepts)
  ---------------------------------------------------------------
  warp specialization   ->  role specialization over execution agents
                            (MTE2 load / CUBE / V vector / MTE3 store)
  SMEM pool + views     ->  UB (Unified Buffer) pool + staged views
  TMEM accumulator      ->  L0C accumulator region
  mbarrier + phases     ->  SetFlag/WaitFlag events (per stage, FIFO)
  pipeline stages       ->  TPipe bufferNum / staged views
  TMA bulk copies       ->  DataCopy (MTE2 GM->UB / MTE3 UB->GM)

Programs are *traced*: the schedule body is ordinary Python executed against
a builder context (paper Fig. 3 style); loops unroll at trace time so the
resulting op list is fully concrete and statically analyzable (paper P4/P5).
Loop structure is retained via LoopMarker ops so codegen can re-roll loops.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from . import dtypes as dt
from .arch import Arch, get_arch

# --------------------------------------------------------------------------
# errors


class IRConstructionError(Exception):
    """Raised when a schedule violates typing rules during construction.

    Carries a Finding so the agent loop can route it as structured evidence
    (paper: "reject ill-typed programs during construction", P4).
    """

    def __init__(self, message: str, *, region: str = "", hint: str = ""):
        super().__init__(message)
        self.message = message
        self.region = region
        self.hint = hint


# --------------------------------------------------------------------------
# shapes


def shape_numel(shape) -> int:
    return dt.numel(shape)


def check_same(shape_a, shape_b, what: str):
    if tuple(shape_a) != tuple(shape_b):
        raise IRConstructionError(
            f"{what}: shape mismatch {tuple(shape_a)} vs {tuple(shape_b)}",
            region=what,
            hint="make producer and consumer tile shapes agree (insert v_transpose/v_cast or adjust views)",
        )


def check_matmul(a, b, acc):
    am, ak = a
    bk, bn = b
    cm, cn = acc
    if ak != bk or am != cm or bn != cn:
        raise IRConstructionError(
            f"matmul shape mismatch: ({am},{ak})x({bk},{bn}) -> ({cm},{cn})",
            region="matmul",
            hint="matmul operands must be (M,K)x(K,N) accumulating into (M,N)",
        )


# --------------------------------------------------------------------------
# resources


@dataclass
class GmParam:
    name: str
    dtype: str
    shape: tuple
    desc: str = ""


@dataclass
class UbPool:
    """A tiered on-chip memory pool.

    tier="ub": Unified Buffer — feeds the vector units (VECCALC position).
    tier="l1": L1 / local buffer — stages cube operands (A1/A2 positions,
               drained by the matmul module into L0A/L0B). The cube operand
               path on Ascend is GM -> L1 -> L0A/L0B -> (accumulate) L0C;
               cube operands must NOT live in UB.
    """

    name: str
    size: int
    tier: str = "ub"

    def view(self, name: str, offset: int, shape, dtype: str, stages: int = 1) -> UbView:
        """Declare a staged view into this pool (`pool.view(...)` registers it
        on the program via the builder-attached `_ctx`)."""
        ctx = getattr(self, "_ctx", None)
        if ctx is None:
            raise IRConstructionError(
                f"pool '{self.name}' was not created by a builder context",
                region=f"ub:{self.name}")
        return ctx._declare_view(self, name, offset, shape, dtype, stages)


@dataclass
class Slot:
    """A (view, stage) pair — `buf[stage]` addressing."""
    view: "UbView"
    stage: int


@dataclass
class UbView:
    name: str
    pool: UbPool
    offset: int          # bytes into pool, per stage slot base
    shape: tuple
    dtype: str
    stages: int = 1

    def __getitem__(self, stage: int) -> Slot:
        if not (0 <= stage < self.stages):
            raise IRConstructionError(
                f"view '{self.name}' has {self.stages} stage(s); slot index {stage} out of range",
                region=f"view:{self.name}",
                hint=f"use a stage index in [0, {self.stages})",
            )
        return Slot(self, stage)

    @property
    def slot_bytes(self) -> int:
        return shape_numel(self.shape) * dt.bytes_of(self.dtype)

    @property
    def total_bytes(self) -> int:
        return self.slot_bytes * self.stages

    def byte_range(self):
        return (self.offset, self.offset + self.total_bytes)


@dataclass
class L0cRegion:
    name: str
    shape: tuple
    dtype: str = dt.FP32


@dataclass
class Role:
    name: str
    kind: str            # MTE2 | MTE3 | CUBE | V | SCALAR

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Role) and self.name == other.name

    # `with m.role(...) as r:` support — the builder attaches `_ctx`.
    def __enter__(self):
        if getattr(self, "_ctx", None) is None:
            raise IRConstructionError(
                f"role '{self.name}' was not created by a builder context",
                region=f"role:{self.name}",
            )
        self._ctx._role = self
        return self

    def __exit__(self, *exc):
        if getattr(self, "_ctx", None) is not None:
            self._ctx._role = None
        return False


VALID_ROLE_KINDS = ("MTE2", "MTE3", "CUBE", "V", "SCALAR")


@dataclass
class Pipeline:
    name: str
    stages: int


@dataclass
class Event:
    name: str
    prod: Role
    cons: Role
    pipeline: Pipeline | None = None


# --------------------------------------------------------------------------
# ops


@dataclass
class Op:
    kind: str
    role: Role
    id: int = -1
    tags: dict = field(default_factory=dict)   # loop vars at trace time

    def locate(self) -> str:
        ctx = " ".join(f"{k}={v}" for k, v in self.tags.items())
        base = f"op#{self.id}[{self.kind}]@{self.role.name}"
        return f"{base}({ctx})" if ctx else base


@dataclass
class LoopMarker(Op):
    var: str = ""
    count: int = 0
    is_begin: bool = True
    depth: int = 0


@dataclass
class Gm2Ub(Op):
    dst: UbView = None
    stage: int = 0
    src: GmParam = None
    gm_off: tuple = ()


@dataclass
class Gm2L1(Op):
    """DataCopy GM -> L1, feeding the cube operand path (A1/A2 staging)."""
    dst: UbView = None
    stage: int = 0
    src: GmParam = None
    gm_off: tuple = ()


@dataclass
class Ub2Gm(Op):
    dst: GmParam = None
    gm_off: tuple = ()
    src: UbView = None
    stage: int = 0


@dataclass
class L0c2Ub(Op):
    dst: UbView = None
    stage: int = 0
    src: L0cRegion = None


@dataclass
class Matmul(Op):
    acc: L0cRegion = None
    a: UbView = None
    a_stage: int = 0
    b: UbView = None
    b_stage: int = 0
    clear: bool = False


@dataclass
class VBinary(Op):
    op: str = ""          # add | sub | mul | min | max
    dst: UbView = None
    dst_stage: int = 0
    x: UbView = None
    x_stage: int = 0
    y: UbView = None
    y_stage: int = 0
    y_scalar: float = None


@dataclass
class VUnary(Op):
    op: str = ""          # neg | abs | sqrt | exp | copy
    dst: UbView = None
    dst_stage: int = 0
    x: UbView = None
    x_stage: int = 0


@dataclass
class VReduce(Op):
    op: str = ""          # sum | max | min
    dst: UbView = None
    dst_stage: int = 0
    x: UbView = None
    x_stage: int = 0


@dataclass
class VArgmin(Op):
    dst: UbView = None
    dst_stage: int = 0
    x: UbView = None
    x_stage: int = 0


@dataclass
class VTranspose(Op):
    dst: UbView = None
    dst_stage: int = 0
    x: UbView = None
    x_stage: int = 0


@dataclass
class VBcast(Op):
    mode: str = ""        # row | col
    dst: UbView = None
    dst_stage: int = 0
    x: UbView = None
    x_stage: int = 0


@dataclass
class VCast(Op):
    dst: UbView = None
    dst_stage: int = 0
    x: UbView = None
    x_stage: int = 0


@dataclass
class Commit(Op):
    event: Event = None
    stage: int = 0


@dataclass
class Wait(Op):
    event: Event = None
    stage: int = 0


@dataclass
class HardBarrier(Op):
    scope: str = "block"


MEMORY_OPS = ("Gm2Ub", "Gm2L1", "Ub2Gm", "L0c2Ub")
COMPUTE_OPS = ("Matmul", "VBinary", "VUnary", "VReduce", "VArgmin",
               "VTranspose", "VBcast", "VCast")
SYNC_OPS = ("Commit", "Wait", "HardBarrier")
ROLE_KIND_OPS = {
    # GM->UB and GM->L1 are both issued on the MTE2 queue
    "MTE2": ("Gm2Ub", "Gm2L1"),
    "MTE3": ("Ub2Gm",),
    # l0c->ub epilogue copy is issued from the cube pipeline (FIX-unit analog)
    "CUBE": ("Matmul", "L0c2Ub"),
    "V": ("L0c2Ub", "VBinary", "VUnary", "VReduce", "VArgmin",
          "VTranspose", "VBcast", "VCast"),
    "SCALAR": (),
}

V_BINARY_OPS = ("add", "sub", "mul", "min", "max")
V_UNARY_OPS = ("neg", "abs", "sqrt", "exp", "copy")
V_REDUCE_OPS = ("sum", "max", "min")


# --------------------------------------------------------------------------
# program


@dataclass
class Program:
    name: str
    arch_name: str
    params: list                      # [GmParam]
    block_dim: int = 1
    ub_pools: list = field(default_factory=list)
    ub_views: list = field(default_factory=list)
    l0c_regions: list = field(default_factory=list)
    roles: list = field(default_factory=list)
    pipelines: list = field(default_factory=list)
    events: list = field(default_factory=list)
    ops: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)  # parent/mutations/notes

    @property
    def arch(self) -> Arch:
        return get_arch(self.arch_name)

    def views_by_name(self) -> dict:
        return {v.name: v for v in self.ub_views}

    def events_by_name(self) -> dict:
        return {e.name: e for e in self.events}

    def roles_by_name(self) -> dict:
        return {r.name: r for r in self.roles}

    def role_ops(self, role_name: str) -> list:
        return [o for o in self.ops if getattr(o, "role", None)
                and o.role.name == role_name
                and not isinstance(o, LoopMarker)]

    def structural_signature(self) -> str:
        """Compact hash capturing structure (op kinds+shapes+stages+block_dim).

        Used for candidate diversity (paper: "generate structurally distinct
        candidates") and archive dedup.
        """
        parts = [f"bd{self.block_dim}"]
        for v in self.ub_views:
            parts.append(f"v:{v.name}:{v.shape}x{v.dtype}x{v.stages}@{v.offset}")
        for r in self.l0c_regions:
            parts.append(f"l:{r.name}:{r.shape}")
        for o in self.ops:
            if isinstance(o, LoopMarker):
                continue
            shape = getattr(o, "shape", None) or _op_shape_sig(o)
            parts.append(f"{o.kind}:{shape}")
        blob = "|".join(str(p) for p in parts)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _op_shape_sig(o: Op):
    out = []
    for attr in ("dst", "src", "a", "b", "acc", "x", "y"):
        v = getattr(o, attr, None)
        if isinstance(v, (UbView, L0cRegion)):
            out.append(f"{v.name}{v.shape}")
        elif isinstance(v, GmParam):
            out.append(f"{v.name}{v.shape}")
    if getattr(o, "y_scalar", None) is not None:
        out.append("k")
    return ",".join(out)
