"""Ascend device model (cost anchors + resource contracts).

Following the paper (B.5): the compiler requires an exact target match and
declines to predict performance where target-specific calibration is absent.
All throughput numbers below are *modeling defaults* from public figures,
expressed per-core; they are calibratable (see costmodel.Calibration) and
should be re-anchored on real hardware before trusting absolute numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Arch:
    name: str
    # ---- on-chip memory (bytes, per AI Core) ----
    ub_bytes: int            # Unified Buffer: feeds the vector units
    l1_bytes: int            # L1 / local buffer: stages cube operands (A1/A2)
    l0c_bytes: int           # Cube accumulator memory per cube core
    # ---- throughput (per AI Core, per cycle) ----
    cube_macs_per_cycle: int    # bf16/fp16 MAC throughput of one cube unit
    vec_lanes_per_cycle: int    # vector lanes per cycle (fp32 element op)
    mte2_bytes_per_cycle: int   # GM -> UB streaming bandwidth share
    mte3_bytes_per_cycle: int   # UB -> GM streaming bandwidth share
    # ---- machine ----
    clock_mhz: int
    num_aic_per_core_group: int
    num_aiv_per_core_group: int
    launch_overhead_us: float = 0.5
    alignment_bytes: int = 32          # DataCopy alignment contract
    max_buffer_num: int = 4            # modeled queue/bufferNum depth limit
    cube_align: int = 16               # matmul M/N/K multiple (bf16/fp16)
    # True only when cost anchors were *measured on real hardware*; modeling
    # defaults from public figures do not count (see coverage labels in
    # costmodel: the model never claims hardware anchoring it does not have).
    calibrated: bool = False
    supports: frozenset = frozenset({BF16 := "bf16", "fp16", "fp32", "int32"})
    notes: tuple = field(default=())

    @property
    def cube_flops_per_cycle(self) -> int:
        return 2 * self.cube_macs_per_cycle

    def seconds_per_cycle(self) -> float:
        return 1.0 / (self.clock_mhz * 1e6)


ASCEND_910B = Arch(
    name="ascend910b",
    ub_bytes=232 * 1024,
    l1_bytes=512 * 1024,
    l0c_bytes=256 * 1024,
    cube_macs_per_cycle=4096,
    vec_lanes_per_cycle=256,
    mte2_bytes_per_cycle=32,
    mte3_bytes_per_cycle=32,
    clock_mhz=1800,
    num_aic_per_core_group=1,
    num_aiv_per_core_group=7,
    calibrated=False,   # modeling defaults, NOT hardware-measured anchors
    notes=(
        "Modeling defaults from public 910B-class figures; per-core shares "
        "of HBM bandwidth are approximate. Not hardware-calibrated: re-anchor "
        "on a real device (costmodel calibration) before trusting absolute "
        "numbers.",
    ),
)

ARCHES = {a.name: a for a in (ASCEND_910B,)}


def get_arch(name: str) -> Arch:
    try:
        return ARCHES[name]
    except KeyError:
        known = ", ".join(sorted(ARCHES))
        raise ValueError(f"unknown target {name!r} (known: {known})") from None
