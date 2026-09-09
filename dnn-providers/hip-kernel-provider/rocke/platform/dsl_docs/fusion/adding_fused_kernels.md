# Adding Fused Kernels

Source: `helpers/fuse.py`, `helpers/epilogues.py`, `helpers/atoms.py`,
`helpers/schedule.py`, `helpers/pipeline.py`, the `helpers/fusion_*` modules,
`instances/common/gemm_wsp3.py`, `core/lower_llvm.py` + `platform/cpp/`,
`tools/check_byte_identity.py`. Companion to
[`fusion/overview.md`](./overview.md), which documents the graph-fusion
*subsystem*; this doc is the *decision guide* — when a fusion is worth doing,
when it is not, and how to land one in rocKE.

Compliance note ([`platform/AGENTS.md`](../../AGENTS.md) §Compliance): this doc
records methodology and levers only. No measured performance numbers, product or
code names, or internal links belong here or in any artifact derived from it.

## Why fuse

Fusion combines what would be two or more kernels/stages into one so the data
between them never leaves the chip. Concretely it buys:

- **Eliminating round-trips of intermediates to HBM.** The producer's output,
  which the consumer immediately consumes, is never written to and re-read from
  global memory. This is usually the dominant win.
- **Keeping producer output resident in registers/LDS.** The consumer reads the
  intermediate at register/LDS latency instead of global-memory latency.
- **Cutting launch and sync overhead.** One dispatch (and one set of barriers)
  instead of several. This matters most when each stage is short.
- **Exposing cross-stage scheduling.** With both stages in one kernel the
  scheduler can overlap the producer's tail with the consumer's head
  (ping-pong / interwave), which separate launches cannot.

### Can fusion pay at all?

Fusion pays only if the stage is limited by something fusion removes. Diagnose
the bound *before* writing anything:

| Bound | Symptom | Does fusion help? |
|---|---|---|
| Bandwidth-bound | The intermediate's bytes moved dominate the stage; arithmetic per byte is low | **Yes** — removing the HBM write+read of the intermediate is the whole win |
| Launch/sync-bound | Many short kernels; per-dispatch and barrier overhead dominate wall time | **Yes** — collapsing launches recovers it |
| Compute-bound | MMA/ALU already saturated; the intermediate is small or reused | **Little** — nothing to remove, and added register/LDS pressure can *lose* occupancy |

Establish the bound with a roofline argument (bytes moved vs. flops) backed by a
trace and resource inspection, not intuition. See
[`optimization/optimization_runbook.md`](../optimization/optimization_runbook.md)
(Step 0 lever sweep) and the ISA/occupancy and trace utilities it links. A
candidate that is already compute-bound is usually not a fusion candidate.

## When not to fuse

Fusion is the wrong tool — or a net loss — when:

- **Divergent tile shapes or grid geometry.** If the stages want different tiling
  or a different grid (e.g. a matmul tiling vs. a full-tensor reduction),
  forcing them into one launch starves one of the two. Keep them separate.
- **Register/LDS pressure that costs occupancy.** Holding the intermediate
  resident consumes the resource that lets many waves run concurrently. Past the
  budget, fewer waves run and the fused kernel is slower than the pair. The
  legalizer (`helpers/fusion_legalize.py`, `FusionLegalizer.legalize`) rejects
  regions that exceed the LDS budget after staging for exactly this reason.
- **A global sync or full-tensor reduction on the boundary.** If the consumer
  needs the whole producer output (a barrier or a reduction across the grid),
  the intermediate cannot stay resident, it must be materialized to a workspace
  (`helpers/fusion_memory.py`), and the residency win evaporates.
- **Stages that need independent tuning.** Different atoms, schedule policies, or
  autotune spaces per stage mean fusing locks in one joint configuration and
  removes a tuning knob. Keep them separable if each stage tunes differently.
- **One-off shapes.** If the fused variant serves a single narrow shape, the
  instance-space, golden, and test cost (below) usually outweighs the win. Ship
  the composed unfused path instead.

The legalizer also rejects unsupported dtype combinations, alignment /
vector-width violations in the epilogue, and atomics in a non-atomic region.
Treat a legalizer rejection as a signal the boundary is wrong, not as an obstacle
to force past.

## How to fuse in rocKE

Pick the lightest form that captures the win. They are ordered cheapest to
heaviest.

### 1. Epilogue fusion

Fold per-element work (bias, activation, scale, clamp, cast, residual) into the
producer's store. This is the most common and most reusable form.

- Compute epilogues are `EpilogueOp` subclasses chained in a `FusedEpilogue`
  (`helpers/fuse.py`); attach to a GEMM spec via its `_fused_epilogue` attribute.
- The store path itself is `DirectEpilogue` / `CShuffleEpilogue`
  (`helpers/epilogues.py`), which move accumulators to global memory; the fused
  ops are applied post-accumulate before the store.
- Adding a new activation is one `EpilogueOp.emit` and (for graph capture) a
  `_PATTERN_TABLE` entry — not a new kernel. See
  [`fusion/overview.md`](./overview.md) and
  [`development/extending.md`](../development/extending.md) §4.

### 2. Producer-consumer fusion

Warp-specialize within one kernel: some warps produce (load / MMA) into LDS while
others consume, so the intermediate lives in LDS across the pipeline. Suited to
stages that share a grid and tile-compatible shapes.

- The realized pattern is the warp-specialized GEMM pipeline
  (`instances/common/gemm_wsp3.py`, `wsp3`), with `Lds{Producer,Consumer}Layout`
  buffers and `CK_WSP3_*` env flags.
- Composed from `MfmaAtom` / `WmmaAtom` (`helpers/atoms.py`), a `SchedulePolicy`
  (`helpers/schedule.py`), and a `SoftwarePipeline` (`helpers/pipeline.py`) — the
  same building blocks any instance uses.

### 3. Whole-pipeline fusion

When stages cannot share on-chip state but the launch/sync overhead is the cost,
fuse at the pipeline level rather than the kernel level.

- Graph-level: the driver in `helpers/fuse.py` plus `fusion_ir.py`,
  `fusion_scheduler.py` (picks region boundaries), `fusion_lowering.py`, and
  `fusion_memory.py` (workspace for intermediates that escape a region).
  Entry points `compile_fn` / `explain_fn`; use `explain_fn` first to see what
  the planner matched.
- Launch-level: capture a multi-kernel pipeline into one replayable HIP graph
  (`PipelineLauncher`) to remove per-dispatch overhead when kernels stay
  distinct but always run together.

### Mechanics every fused change must clear

- **Mirror the emission in both engines.** The #1 invariant
  ([`development/engine_parity.md`](../development/engine_parity.md),
  [`development/invariants.md`](../development/invariants.md)): the Python engine
  (`core/lower_llvm.py`) and the C++ engine (`platform/cpp/`) must emit
  byte-identical LLVM-IR. Any op / atom / epilogue / fusion / attribute change
  lands in both engines in the same change. Exception: kernels under
  `library/kernels/` have no C++ mirror by design, there the Python lowering is
  the ground truth, but they still gate the golden.
- **Re-run the gate.** `tools/check_byte_identity.py` GREEN for every family at
  every LLVM flavor (`llvm20` and `llvm22`). If the emission is meant to change,
  re-bless the golden IR hash in the same change, never separately.
- **Keep the unfused path as the correctness reference.** Byte-identity and the
  golden are blind to a wrong-but-stable kernel, they pin stability, not
  correctness. Correctness is only established against an independent numpy/torch
  reference on a real device. Exercise fused vs. unfused vs. reference through
  `run_fusion_validation_matrix` (`helpers/fusion_validation.py`) and the
  differential numeric lanes. Do not delete the unfused implementation, it is the
  oracle the fused path is checked against.

## Worked example: fusing a selection stage (indexer + top-k)

A concrete candidate from preliminary design work: a content-based selection
front-end for sparse attention. A cheap indexer scores every KV position for a
query (a light per-head dot product with a ReLU and a per-head weighted sum, kept
in a low-precision format), and a top-k keeps only the highest-scoring
positions. The selected subset then feeds a sparse-attention (SDPA) consumer that
attends over just those positions. Walking it through this doc's framework:

**Why it fuses.** The indexer emits one score per KV position: an intermediate
that grows with context length, and the top-k immediately collapses it to a
fixed, much smaller index set. The producer is compute-light, so a standalone
indexer would spend most of its time writing that score array to HBM only for the
top-k to read it all back and discard it. Fusing indexer→top-k keeps the scores
resident and emits only the surviving indices: the textbook bandwidth-bound,
large-throwaway-intermediate case from *Why fuse*.

**Bound.** Bandwidth-bound on the score array, not compute-bound, so fusion can
pay. Confirm on the target with a trace before committing.

**Form and boundary.** Fuse the selection stage (indexer + top-k) as one unit
and keep the sparse-attention consumer a separate kernel. Two entries from *When
not to fuse* decide this seam:

- **Dtype boundary.** The indexer runs in a low-precision format while the
  consumer runs in a wider one; welding the low-precision scorer into the
  attention consumer is the fragile-dtype-boundary case. Keep the seam between
  them.
- **Independent reuse/skip point.** A downstream feature may reuse a
  previously-computed index set and skip selection entirely for some layers.
  Drawing the fused boundary around indexer+top-k makes that a clean unit to skip
  — the *natural boundary* rule.

**Costs this incurs.** Because the score array never materializes, the top-k must
run as a streaming top-k over scores as they are produced, not over a finished
array — the "forces a streaming algorithm" cost from *When not to fuse*, and a
real piece of new work rather than a free consequence of fusing. In the prefill
case the per-query top-k is also ragged (each query row selects a different
subset), so the gather stays in the consumer, not in the fused selection stage.

**Reuse.** The fused unit is a generic shape: a content-based scorer feeding a
hard top-k. Express the scorer as a compile-time parameter rather than hardcoding
this indexer's formula, and the same fused skeleton serves other content-based
selection variants while this case plugs in its own scorer. That reuse boundary is
content-based hard-top-k selection, it does not extend to static-sparsity schemes
(no scorer to fuse) or soft-selection schemes (which need the full score array,
breaking the residency premise).

**Status.** Design-stage and unverified: no measured numbers, and the fused path
is gated against the unfused selection plus an independent reference as the
correctness oracle before any performance claim.

## Maintainability and reusability

Every fused variant multiplies the instance space, the goldens, build time, and
the test matrix. A bespoke fused monolith pays that cost once per variant and
shares nothing.

- **Compose, don't monolith.** Prefer expressing a fusion as existing atoms and
  epilogues over a hand-written combined kernel. A new activation is a new
  `EpilogueOp`; a new captured pattern is a `_PATTERN_TABLE` entry; a new tiling
  is a schedule/atom choice. Each reuses the lowering, the gate, and the tests.
- **Parameterize the variable stage.** When one half of a fusion is
  case-specific (e.g. the scoring function of a selection stage) and the other is
  generic (e.g. the reduction or the consumer it feeds), express the specific
  half as a compile-time parameter/functor rather than hardcoding it. The
  intermediate still stays resident, and the generic skeleton is reused across
  variants instead of copied.
- **Fuse at natural boundaries.** Draw the fused unit at a stable input→output
  contract and, where a downstream feature may reuse or skip a stage, at that
  skip point, so the boundary is a reuse seam, not an arbitrary cut.

## Checklist

1. **Measure the unfused baseline first** and identify the bound
   (bandwidth / launch / compute) with a roofline argument and a trace.
2. **Confirm fusion can pay** — the stage is bandwidth- or launch-bound, not
   already compute-bound.
3. **Choose the lightest form** that captures the win (epilogue < producer-consumer
   < whole-pipeline).
4. **Compose over bespoke** — add an `EpilogueOp` / pattern entry / schedule choice
   rather than a monolithic kernel; parameterize the case-specific stage.
5. **Mirror both engines** and re-run `tools/check_byte_identity.py` GREEN at
   `llvm20` and `llvm22`; re-bless the golden in the same change if emission
   changed.
6. **Verify correctness before claiming a win** — fused vs. unfused vs.
   numpy/torch reference via `run_fusion_validation_matrix` / the numeric lanes,
   within tolerance.
7. **Record the measurement** per the compliance policy — methodology and levers
   in-repo, measured numbers to the protected location only.
8. **Keep the unfused path** as the reference oracle.
