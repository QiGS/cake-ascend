"""Dtype registry and storage-quantization numerics for CAKE-Ascend.

Numerics model (documented honesty):
- Buffer values are held as Python floats (fp64) between operations.
- On every *write* into a buffer (UB slot, L0C, GM) the value is quantized
  to the buffer's storage dtype (bf16/fp16: round-to-nearest-even; fp32:
  single-precision rounding; int32: nearest integer).
- Reads return the stored (already quantized) values verbatim.
This mirrors "storage quantization" on real hardware closely enough for
tolerance-based validation while staying pure-stdlib.
"""
from __future__ import annotations

import struct

BF16 = "bf16"
FP16 = "fp16"
FP32 = "fp32"
INT32 = "int32"
FP64 = "fp64"  # oracle-only

_DTYPES = (BF16, FP16, FP32, INT32, FP64)

_BITS = {BF16: 16, FP16: 16, FP32: 32, INT32: 32, FP64: 64}


def is_dtype(name: str) -> bool:
    return name in _DTYPES


def is_float(name: str) -> bool:
    return name in (BF16, FP16, FP32, FP64)


def bits(name: str) -> int:
    return _BITS[name]


def bytes_of(name: str) -> int:
    return _BITS[name] // 8


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _quantize_bf16(x: float) -> float:
    """Round-to-nearest-even bfloat16 truncation of an fp32 value."""
    if x != x or x in (float("inf"), float("-inf")):
        return x
    # Clamp to finite fp32 range first.
    f32 = struct.unpack("<f", struct.pack("<f", x))[0]
    if f32 in (float("inf"), float("-inf")):
        return f32
    u = struct.unpack("<I", struct.pack("<f", f32))[0]
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def _quantize_fp16(x: float) -> float:
    try:
        return struct.unpack("<e", struct.pack("<e", x))[0]
    except OverflowError:
        return float("inf") if x > 0 else float("-inf")


def _quantize_fp32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def quantize(x: float, dtype: str) -> float:
    """Quantize a Python float to a storage dtype value (returned as float)."""
    if dtype == BF16:
        return _quantize_bf16(x)
    if dtype == FP16:
        return _quantize_fp16(x)
    if dtype == FP32:
        return _quantize_fp32(x)
    if dtype == INT32:
        return float(round(x))
    if dtype == FP64:
        return x
    raise ValueError(f"unknown dtype {dtype!r}")


def quantize_list(values, dtype: str) -> list:
    if dtype == FP64:
        return list(values)
    q = quantize
    return [q(v, dtype) for v in values]
