"""Workload registry + the workload contract (paper Sec. 4).

"The workload contract fixes the shapes, oracle, tolerances, hardware, and
permitted references" — everything here is stable authority for a run;
evolution only touches candidates, never the contract.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, fields


@dataclass
class Workload:
    name: str
    description: str
    shape: dict                         # contract shape (single-shape evolution)
    domain: list                        # shape domain (generalization stage)
    io: list                            # [(name, dtype, shape_tuple, kind)] kind=in|out
    tolerance: tuple                    # (rtol, atol)
    make_inputs: object = None          # (shape, seed) -> {name: [floats]} (quantized)
    oracle: object = None               # (shape, inputs_fp) -> {name: [floats]}
    compare: object = None              # (shape, expected, got) -> (ok, note)
    params_class: object = None
    default_params: object = None       # () -> naive-but-correct baseline params
    kernel_fn: object = None            # (shape, params) -> kern(m) traceable
    block_dim: object = None            # (shape, params) -> int
    domain_guard: object = None         # (shape, params) -> bool (dispatcher)
    label: str = ""

    def param_shapes(self, shape) -> dict:
        out = {}
        for name, _dt, shp, kind in self.io:
            out[name] = shape_dims(shp, shape)
        return out

    def summary(self) -> str:
        ios = ", ".join(f"{n}:{dt}{shape_dims(s, self.shape)}{'*' if k == 'out' else ''}"
                        for n, dt, s, k in self.io)
        return f"{self.name}({ios}) tol={self.tolerance}"


def shape_dims(shp, shape) -> tuple:
    return tuple(int(shape[d]) if isinstance(d, str) else int(d) for d in shp)


def default_compare(tolerance):
    rtol, atol = tolerance

    def compare(shape, expected, got, inputs=None):
        worst = 0.0
        for name, vals in expected.items():
            if name.startswith("_"):
                continue
            g = got.get(name)
            if g is None:
                return False, f"missing output '{name}'"
            if len(g) != len(vals):
                return False, f"output '{name}' length {len(g)} != {len(vals)}"
            for a, b in zip(vals, g):
                err = abs(a - b)
                scale = atol + rtol * abs(a)
                if err > scale:
                    return False, (f"output '{name}' mismatch: expected {a:.6g}, "
                                   f"got {b:.6g} (err {err:.3g} > {scale:.3g})")
                if scale > 0:
                    worst = max(worst, err / scale)
        return True, f"max rel-scaled err {worst:.3g}"
    return compare


def gen_uniform(rng: random.Random, n: int, lo=-1.0, hi=1.0):
    return [rng.uniform(lo, hi) for _ in range(n)]


def argmin_rows(vals, r, c):
    out = []
    for i in range(r):
        row = vals[i * c:(i + 1) * c]
        best = 0
        for j in range(1, c):
            if row[j] < row[best]:
                best = j
        out.append(float(best))
    return out


def row_sum(vals, r, c):
    return [math.fsum(vals[i * c:(i + 1) * c]) for i in range(r)]


_REGISTRY = {}


def register(wl: Workload):
    _REGISTRY[wl.name] = wl
    return wl


def get(name: str) -> Workload:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown workload {name!r}; known: {sorted(_REGISTRY)}") from None


def all_names():
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------
# generic params mutation support


def params_dict(params) -> dict:
    return {f.name: getattr(params, f.name) for f in fields(params)}


def params_replace(params, **kw):
    d = params_dict(params)
    d.update(kw)
    return type(params)(**d)


def mutate_params(params, spec: dict, rng: random.Random, n: int):
    """Propose `n` neighbor param sets from a per-field choice spec.

    Each mutation changes 1-2 fields; callers dedupe by program signature.
    """
    keys = [k for k in spec if spec[k]]
    out = []
    for _ in range(n * 3):
        d = params_dict(params)
        k = rng.choice(keys)
        choices = [v for v in spec[k] if v != d[k]]
        if not choices:
            continue
        d[k] = rng.choice(choices)
        if rng.random() < 0.35 and len(keys) > 1:
            k2 = rng.choice(keys)
            ch2 = [v for v in spec[k2] if v != d[k2]]
            if ch2:
                d[k2] = rng.choice(ch2)
        out.append(type(params)(**d))
        if len(out) >= n:
            break
    return out


from . import vec_add  # noqa: E402
from . import gemm  # noqa: E402
from . import kmeans_assign  # noqa: E402
