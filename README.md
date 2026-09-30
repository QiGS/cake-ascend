# CAKE-Ascend

**Compiler–Agent Co-Design for Ascend Kernel Evolution** — a from-scratch
implementation of the CAKE paper's architecture
([arXiv:2608.12629](https://arxiv.org/abs/2608.12629), *"CAKE: Compiler–Agent
Co-Design for Frontier Kernel Evolution"*), re-targeted from NVIDIA CUDA to the
**Huawei Ascend** execution model.

Kernel agents and compiler stacks usually evolve separately; the gap between
them is where expert kernels are lost. CAKE closes the gap by co-designing
both: agents author a **typed, hardware-explicit schedule IR** and get
**localized correctness/performance diagnostics** back, and the **harness
itself is a target of evolution** — recurring failures become verifier rules,
cost-model calibrations, and repair tactics rather than one-off workarounds.

```
        ┌────────────────────── Kernel Evolution (inner loop) ──────────────────────┐
        │  1. generate structurally distinct candidates (proposer: heuristic | LLM)  │
        │  2. filter: IR construction checks -> verifier gates -> cost-model rank    │
        │  3. evaluate survivors: simulator + external oracle + measured span        │
        │  4. route evidence: candidates | verifier rules | calibration | IR gaps    │
        └───────────────────────────────┬───────────────────────────────────────────┘
                                        │ recurring evidence
        ┌─────────────────── Compiler-Harness Evolution (outer loop) ───────────────┐
        │  dynamic failure recurrence  -> static rule (corpus-gated)                │
        │  systematic misprediction    -> cost-model calibration (validated fit)    │
        │  repair knowledge            -> tactics fed back to the proposer           │
        │  human merge gate: proposals print as diffs; --apply persists them         │
        └───────────────────────────────────────────────────────────────────────────┘
```

## Install

Pure Python 3.10+ standard library — no third-party dependencies.

```
python -m cakeasc demo kmeans_assign --iters 12
```

## What maps where (paper -> this repo)

| Paper (NVIDIA)            | Here (Ascend)                                              |
|---------------------------|------------------------------------------------------------|
| Cake IR schedule language | `cakeasc/ir.py`, `builder.py` — traced Python DSL          |
| warp specialization       | role specialization: `MTE2 / CUBE / V / MTE3` agents      |
| SMEM pool + staged views  | UB (Unified Buffer) pools + `ub.view(..., stages=)`        |
| TMEM accumulator          | L0C accumulator region (`m.l0c`)                           |
| mbarrier + phase bits     | SetFlag/WaitFlag events, FIFO per (event, stage)          |
| pipeline (bufferNum)      | `m.pipeline(stages=)` staged slots                         |
| TMA bulk copy             | `DataCopy` MTE2 (GM->UB) / MTE3 (UB->GM)                   |
| pre-compile gates         | `rules.py` + `verifier.py` (localized findings + hints)    |
| numerical validation      | `interpreter.py` dataflow simulator + workload oracles     |
| CUPTI span                | deterministic timing trace (cycles) from the simulator     |
| cost model + calibration  | `costmodel.py` (analytic predictor vs measured, learnable) |
| compiler evolution        | `evidence.py`, `distiller.py`, `corpus.py` (test-gated)    |
| generalization / dispatch | `dispatcher.py` (guards, fallback, Gspan over a domain)    |

## The IR in one look

```python
@asc.schedule(name="gemm", block_dim=4)
def kern(m):
    A = m.gm_param("A", "bf16", (M, K))
    B = m.gm_param("B", "bf16", (K, N))
    C = m.gm_param("C", "fp32", (M, N))
    ub = m.ub_pool("ub", 232 * 1024)
    bufA = ub.view("A", 0, (BM, BK), "bf16", stages=2)      # staged double buffer
    bufB = ub.view("B", off, (BK, BN), "bf16", stages=2)
    acc  = m.l0c("acc", (BM, BN))                            # fp32 L0C accumulator
    ld, cu, st = m.role("ld", "MTE2"), m.role("cu", "CUBE"), m.role("st", "MTE3")
    pipe = m.pipeline("main", stages=2)
    a_rdy  = m.event("a_rdy",  ld, cu, pipe)   # data ready (SetFlag/WaitFlag)
    a_free = m.event("a_free", cu, ld, pipe)   # back-pressure (slot free)
    with ld:
        for t in m.tile_loop("t", TILES):      # unrolls at trace time
            s = t % 2
            if t >= 2: m.wait(a_free, stage=s)
            m.gm2ub(bufA[s], A, (row0, col0)); m.commit(a_rdy, stage=s)
    ...
```

Schedules are ordinary Python traced against the builder: loops unroll, so
every op is concrete and statically analyzable (paper P4/P5), while lowering
derives the mechanical parts (UB offsets, bufferNum, HardEvent ids, loop
structure) from the declarations.

## CLI

```
python -m cakeasc demo <workload> [--iters N] [--llm] [--apply] [--no-generalize]
python -m cakeasc verify  examples/vec_add_schedule.py     # gates + cost report
python -m cakeasc codegen examples/vec_add_schedule.py -o kernel.cpp
python -m cakeasc corpus                                    # rule-template gate suite
python -m cakeasc guide                                     # IR reference (for humans/LLMs)
python -m cakeasc state                                     # evolved rules + calibration
```

Workloads: `vec_add` (memory-bound warm-up), `gemm` (cube-bound tiling/stages),
`kmeans_assign` (the paper's workload: argmin-k ||x-c||² via GEMM formulation
with a four-role choreography).

### Demo of the co-evolution arc

`python -m cakeasc demo gemm --iters 12 --apply` shows the whole paper story:

1. tile mutations that don't divide M produce **dynamic** `gm_oob` findings
   (33 recurrences) -> the distiller installs the static `gm_bounds` rule,
   corpus-gated; later iterations reject such candidates **pre-compile**;
2. a racy candidate family recurs 3x -> `slot_back_pressure` rule installed;
3. the analytic cost model systematically under-predicts (median 39% off)
   -> least-squares calibration drops misprediction to ~3%;
4. the kernel itself evolves ~5.7x over the naive-but-correct baseline
   (same protocol both sides), and the dispatcher portfolio generalizes it
   across 12 domain shapes with an explicit fallback and zero leakage.

## Repository layout

```
cakeasc/
  dtypes.py        storage quantization (bf16/fp16 RNE emulation)
  arch.py          Ascend device model (calibratable cost anchors)
  ir.py            typed op vocabulary + declarative resources + Program
  builder.py       tracing builder (`m.*` API), construction-time typing
  diagnostics.py   Finding contract (paper Table 1 categories)
  rules.py         evolvable rule registry + 8 rule templates
  verifier.py      pre-compile gates over per-core programs
  costmodel.py     analytic predictor, bottleneck attribution, calibration
  interpreter.py   dataflow simulator: numerics, localized dynamic findings,
                   deterministic timing (ground truth for calibration)
  codegen.py       AscendC lowering (inspectable; derived metadata)
  agent.py         four-stage evolution loop + archive + evidence routing
  heuristics.py    heuristic mutation proposer (repair-first tactics)
  llm.py           OpenAI-compatible LLM proposer + IR guide
  evidence.py      evidence store + recurrence detection
  distiller.py     rule distillation + cost calibration (merge-gated)
  corpus.py        corpus test gate (valid corpus + per-rule fixtures)
  dispatcher.py    generalization: guards, fallback, Gspan
  workloads/       vec_add, gemm, kmeans_assign (contract + oracle + seed)
examples/          hand-authored schedule + generated AscendC sample
tests/             38 tests (also part of the corpus gate's valid set)
```

## Honest deviations from the paper

- **No Ascend hardware locally**: execution authority is the deterministic
  simulator (numerics + timing), which plays the roles of runtime, Compute
  Sanitizer and CUPTI. `codegen.py` emits real, inspectable AscendC source
  with concrete linear GM addresses (deterministic by construction), a
  single `TPipe` with per-slot `TBuf`s, cube operands staged at `A1/A2`
  (L1->L0A/L0B) rather than UB, `SetFlag/WaitFlag` choreography, and the
  Matmul API — but it has never been compiled by a CANN toolchain; treat it
  as a faithful structural lowering, not a proven build.
- **Security note on the LLM proposer**: schedule sources authored by a
  remote model are `exec`-uted locally to be traced into IR — arbitrary
  code execution by design, the same trust model as applying an LLM-written
  patch to any repo. Only configure endpoints you control
  (`CAKEASC_LLM_BASE_URL`); set `CAKEASC_DISALLOW_EXEC=1` to hard-disable
  executing authored sources.
- **Arch numbers are modeling defaults** from public 910B-class figures
  (per-core bandwidth shares, cube MACs, launch overhead). They are
  calibratable by design; re-anchor on hardware before trusting absolute
  numbers (paper B.5: decline to predict where uncalibrated).
- **Rule distillation instantiates templates**, it does not synthesize
  arbitrary checker code: recurring dynamic failure families map to
  parametrized static templates, which keeps evolution safe by construction.
  The static analyses are intentionally incomplete (paper Appendix C) — e.g.
  the back-pressure rule approximates the choreography contract, and dynamic
  version-skip race analysis catches what static checks miss.
- **LLM proposer** is pluggable (`CAKEASC_LLM_BASE_URL` / `_API_KEY` /
  `_MODEL`) but the offline demo uses the deterministic heuristic proposer,
  which exercises the identical four-stage loop and harness.
- **Single-shape evolution then portfolio generalization** follows paper Sec. 6
  (separate objectives, guards partition a declared domain, explicit fallback):
  the domain is split deterministically into tuning and held-out shards —
  route selection happens on tuning shards only, held-out shards inherit
  routes through guard predicates and are used purely for validation
  (anti-leakage), and Gspan is reported over the full domain.

## Extending

- New workload: add `cakeasc/workloads/<name>.py` with a `Workload(...)`:
  contract shape, domain, IO spec, oracle + compare, tunable params, a
  parameterized seed kernel (the choreography template), and a domain guard.
- New rule template: subclass `Rule` in `rules.py`, add an invalid fixture in
  `corpus.py`; it becomes distiller-installable automatically.
- Real hardware: replace `interpreter.run_simulation` with an AscendCL runner
  (compile the generated AscendC, launch, check outputs, CUPTI-like timing)
  and feed measured spans back — the calibration loop is already waiting for
  exactly that evidence.
