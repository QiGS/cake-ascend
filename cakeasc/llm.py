"""Pluggable LLM client (OpenAI-compatible) + the IR language guide used
in prompts.

Paper context: the agent model is held fixed while the *environment*
evolves — so the LLM is just one proposer behind the same four-stage loop;
a deterministic heuristic proposer stands in when no endpoint is
configured (env: CAKEASC_LLM_BASE_URL / CAKEASC_LLM_API_KEY /
CAKEASC_LLM_MODEL / CAKEASC_LLM_TIMEOUT).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, base_url=None, api_key=None, model=None, timeout=None):
        self.base_url = (base_url or os.environ.get("CAKEASC_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("CAKEASC_LLM_API_KEY", "")
        self.model = model or os.environ.get("CAKEASC_LLM_MODEL", "gpt-5.6-sol")
        try:
            self.timeout = float(timeout or os.environ.get("CAKEASC_LLM_TIMEOUT", "90"))
        except ValueError:
            self.timeout = 90.0

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    def chat(self, messages, temperature=0.4, max_tokens=4096) -> str:
        if not self.available:
            raise LLMError("LLM not configured (set CAKEASC_LLM_BASE_URL/API_KEY/MODEL)")
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"LLM request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM returned invalid JSON: {e}") from e
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected LLM response shape: {e}") from e


# --------------------------------------------------------------------------
# IR guide (shared by prompts and human-facing docs)


IR_GUIDE = """\
CAKE-Ascend schedule IR — a typed, hardware-explicit representation.
A kernel is a Python function `def kern(m)` traced by the builder; loops
unroll at trace time (counts must be Python ints). Everything must be
concrete: shapes, offsets, and trip counts are literals.

Resources (declare once, use by name):
  m.gm_param(name, dtype, shape)            # GM tensor param (bf16|fp16|fp32|int32)
  ub = m.ub_pool(name, size_bytes)          # Unified Buffer pool (<= 232448 B)
  v  = ub.view(name, offset, shape, dtype, stages)   # staged tile view into pool
  acc = m.l0c(name, shape)                  # fp32 cube accumulator (L0C)
  r  = m.role(name, kind)                   # kind: MTE2|CUBE|V|MTE3|SCALAR
  pipe = m.pipeline(name, stages)           # 1..4
  e  = m.event(name, prod=role, cons=role, pipeline=pipe)  # SetFlag/WaitFlag pair

Control:
  with m.role(...):                         # ops issue from that role only
  for i in m.tile_loop("i", count):         # traced loop (unrolls)
  m.core_id() / m.core_count()              # per-core specialization
  m.num_tiles(total, tile)                  # ceil-div helper

Ops (inside role blocks only):
  m.gm2ub(view[stage], param, gm_off)       # DataCopy GM->UB (MTE2); dtype must match
  m.ub2gm(param, gm_off, view[stage])       # DataCopy UB->GM (MTE3); dtype must match
  m.matmul(acc, a[stage], b[stage], clear)  # CUBE: (M,K)x(K,N)->acc(M,N) fp32;
                                            # clear=True on first K-step of a chain
  m.l0c2ub(view[stage], acc)                # accumulator -> UB
  m.v_binary(op, dst, x, y|scalar)          # add|sub|mul|min|max (elementwise)
  m.v_unary(op, dst, x)                     # neg|abs|sqrt|exp|copy
  m.v_reduce(op, dst1d, x2d)                # sum|max|min along last axis -> (R,)
  m.v_argmin(dst_int32_1d, x2d)             # argmin along last axis -> (R,)
  m.v_transpose(dst, x)                     # (R,C) -> (C,R)
  m.v_bcast("row"|"col", dst2d, x1d)        # (R,)->(R,C) rows / (C,)->(R,C) cols
  m.v_cast(dst, x)                          # dtype conversion

Synchronization (the choreography the verifier checks):
  m.commit(event, stage=s)                  # only inside event.prod role
  m.wait(event, stage=s)                    # only inside event.cons role
  k-th commit of (event, stage) satisfies the k-th wait (FIFO, like
  SetFlag/WaitFlag). Events without a pipeline use stage 0 only.

Hard contracts (the harness enforces these — violations are localized
findings, not opaque crashes):
  * UB pool fits on-chip; views disjoint and 32B-offset aligned.
  * DataCopy: contiguous dim and GM offsets 32B-aligned; dtypes match
    exactly (no implicit conversion — use v_cast).
  * matmul M/N/K multiples of 16; operands bf16/fp16; acc fp32.
  * A slot (view, stage) rewritten for tile t+1 needs the consumer's
    back-pressure event waited by the producer before the rewrite
    (WAR hazard otherwise). Every reused slot needs BOTH directions:
    prod->cons (data ready) and cons->prod (slot free).
  * first matmul into an l0c region after a fresh accumulation chain
    must pass clear=True.
  * GM accesses must be in bounds; per-core partitions must tile exactly.
Write the schedule as ONE Python function. Keep all sizes literal ints.
"""


def arch_card(arch) -> str:
    return (f"target: {arch.name}  UB={arch.ub_bytes}B  L0C={arch.l0c_bytes}B  "
            f"cube={arch.cube_macs_per_cycle} MAC/cyc  vector={arch.vec_lanes_per_cycle} "
            f"lanes/cyc  MTE2/MTE3={arch.mte2_bytes_per_cycle}/{arch.mte3_bytes_per_cycle} "
            f"B/cyc  clock={arch.clock_mhz}MHz  align={arch.alignment_bytes}B  "
            f"max_buffer_num={arch.max_buffer_num}  cube_align={arch.cube_align}")


def extract_code_blocks(text: str) -> list:
    """Extract ```python fenced blocks (or bare `def kern(m)` fallback)."""
    blocks = []
    buf, in_block = [], False
    for line in text.splitlines():
        if line.strip().startswith("```") and not in_block:
            in_block = True
            buf = []
            continue
        if line.strip() == "```" and in_block:
            blocks.append("\n".join(buf))
            in_block = False
            continue
        if in_block:
            buf.append(line)
    if not blocks and "def kern" in text:
        start = text.index("def kern")
        blocks.append(text[start:])
    return blocks
