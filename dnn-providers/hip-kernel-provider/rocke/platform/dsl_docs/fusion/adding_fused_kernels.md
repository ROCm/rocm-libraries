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
  budget, fewer waves run and the fused kernel is slower than the pair. Nothing
  checks this for you — make the occupancy argument from the resource counts
  before committing (see the ISA/occupancy utilities in the runbook).
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
- **A fragile dtype boundary.** If the producer runs in a materially narrower
  format than the consumer, welding them keeps the narrow format live across the
  seam and couples two independent precision decisions. Keep the seam.
- **Fusion that forces a harder algorithm.** If keeping the intermediate resident
  means the consumer must be rewritten to stream (e.g. a streaming top-k instead
  of one over a finished array), that rewrite is real new work, not a free
  consequence of fusing — price it in.

The legalizer also rejects unsupported dtypes and layouts, op kinds outside
`supported_ops`, side-effecting ops, and shape/rank violations (matmul K mismatch,
mixed-dtype matmul, incompatible broadcast ranks). Treat a legalizer rejection as
a signal the boundary is wrong, not as an obstacle to force past.

## Fusion and adjacent techniques in rocKE

Three techniques, ordered cheapest to heaviest. Only the first is fusion in the
strict sense — merging distinct operations into one kernel so the intermediate
never leaves the chip. The other two are adjacent techniques that attack the same
launch/bandwidth overheads without necessarily merging ops. Pick the lightest one
that captures the win.

### 1. Epilogue fusion

Fold per-element work (bias, activation, scale, clamp, cast, residual) into the
producer's store. This is the most common and most reusable form.

- Compute epilogues are `EpilogueOp` subclasses chained in a `FusedEpilogue`
  (`helpers/fuse.py`); attach to a GEMM spec via its `_fused_epilogue` attribute.
- The store path itself is `DirectEpilogue` / `CShuffleEpilogue`
  (`helpers/epilogues.py`), which move accumulators to global memory; the fused
  ops are applied post-accumulate before the store.
- Adding a new activation is an `EpilogueOp` subclass implementing `apply_element`
  and `tag` (plus `declare_params` when it needs kernel arguments) and, for graph
  capture, a `_PATTERN_TABLE` entry — not a new kernel. See
  [`fusion/overview.md`](./overview.md) and
  [`development/extending.md`](../development/extending.md) §4.

### 2. Warp specialization (producer-consumer)

Split the warps of one kernel: some produce (load / MMA) into LDS while others
consume, so the intermediate lives in LDS across the pipeline. In its shipped form
this is an intra-kernel scheduling/pipelining technique on a single op, not fusion;
it *becomes* fusion only when the producer and consumer are genuinely distinct
operations (e.g. a scorer feeding a selector). Suited to stages that share a grid
and tile-compatible shapes.

- The realized pattern is the warp-specialized GEMM pipeline
  (`instances/common/gemm_wsp3.py`, `wsp3`), where a subset of warps does the
  global→LDS load and the rest consume, tuned via `CK_WSP3_*` env flags.
- Note the current `wsp3` emits directly with `IRBuilder` and reuses the GEMM
  helpers `_resolve_mma_op`, `_emit_smem_load`, `_emit_mma`, and
  `_emit_epilogue_default` from `gemm_universal`; it does not instantiate the
  generic `MfmaAtom` / `SchedulePolicy` / `SoftwarePipeline` classes.

### 3. Pipeline orchestration (and graph-level fusion)

When stages cannot share on-chip state but the launch/sync overhead is the cost,
work at the pipeline level rather than the kernel level. The launch-level form
below is orchestration, not fusion — the kernels stay distinct; only the
graph-level form is true fusion, and it is not wired in yet.

- Graph-level: `compile_fn` / `explain_fn` in `helpers/fuse.py` currently match
  `_PATTERN_TABLE` directly and support a single GEMM-plus-epilogue kernel; use
  `explain_fn` first to see what matched. The `fusion_ir.py`, `fusion_scheduler.py`,
  `fusion_lowering.py`, and `fusion_memory.py` modules are multi-region planning
  scaffolding (region boundaries, workspace for escaping intermediates) but are not
  wired into those entry points yet.
- Launch-level: chain the stages on a single stream (`PipelineLauncher`) so they
  run in FIFO order without host-side synchronization between them, when kernels
  stay distinct but always run together. Note this does not eliminate per-dispatch
  overhead — each stage is still its own dispatch.

### Mechanics every fused change must clear

- **Mirror the emission in both engines.** The #1 invariant
  ([`development/engine_parity.md`](../development/engine_parity.md),
  [`development/invariants.md`](../development/invariants.md)): the Python engine
  (`core/lower_llvm.py`) and the C++ engine (`platform/cpp/`) must emit
  byte-identical LLVM-IR. Any op / atom / epilogue / fusion / attribute change
  lands in both engines in the same change. Nuance: some `library/kernels/` paths
  (e.g. `attention_dense`) have no C++ builder mirror and take Python lowering as
  ground truth, but that is per-kernel and settled at port time — in general the
  serialized IR is still lowered by the C++ backend, so a new IR op needs matching
  C++ lowerer support or an explicit `BackendCoverageGap`. The golden still gates
  either way.
- **Re-run the gate.** `tools/check_byte_identity.py` GREEN for every family at
  every LLVM flavor (`llvm20`, `llvm22`, and `llvm23`). If the emission is meant to
  change, re-bless the golden IR hash in the same change, never separately.
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
and keep the sparse-attention consumer a separate kernel. Two rules decide this
seam:

- **A fragile dtype boundary** (from *When not to fuse*). The indexer runs in a
  low-precision format while the consumer runs in a wider one; welding the
  low-precision scorer into the attention consumer keeps the narrow format live
  across the seam. Keep the seam between them.
- **Fuse at natural boundaries** (from *Maintainability and reusability*). A
  downstream feature may reuse a previously-computed index set and skip selection
  entirely for some layers; drawing the fused boundary around indexer+top-k makes
  that a clean unit to skip.

**Costs this incurs.** Because the score array never materializes, the top-k must
run as a streaming top-k over scores as they are produced, not over a finished
array — the *fusion that forces a harder algorithm* case from *When not to fuse*,
and real new work rather than a free consequence of fusing. In the prefill
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
3. **Choose the lightest form** that captures the win (epilogue fusion < warp
   specialization < pipeline orchestration).
4. **Compose over bespoke** — add an `EpilogueOp` / pattern entry / schedule choice
   rather than a monolithic kernel; parameterize the case-specific stage.
5. **Mirror both engines** and re-run `tools/check_byte_identity.py` GREEN at
   `llvm20`, `llvm22`, and `llvm23`; re-bless the golden in the same change if
   emission changed.
6. **Verify correctness before claiming a win** — run fused and unfused paths
   against a numpy/torch reference and explicitly assert every reported error
   against tolerance. Note `run_fusion_validation_matrix` records `max_abs` but does
   not itself enforce `atol` / `rtol`, so the assertion is on you.
7. **Record the measurement** per the compliance policy — methodology and levers
   in-repo, measured numbers to the protected location only.
8. **Keep the unfused path** as the reference oracle.
