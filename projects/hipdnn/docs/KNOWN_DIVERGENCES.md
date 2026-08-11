# Known divergences: cuDNN frontend compatibility shim

This document describes intentional behavioral differences between NVIDIA cuDNN
frontend v9 and hipDNN's cuDNN-shaped compatibility shim. The shim is source
compatibility for supported v9 graph-API translation units, not ABI compatibility
and not a full cuDNN backend implementation.

## Scope

The shim targets the cuDNN frontend v9 graph API surface exposed by
`<hipdnn_compatibility/cudnn/cudnn_frontend.h>`. The v0.x / v8 builder and backend
descriptor API surface is out of scope.

## Heuristic modes

The shim accepts every cuDNN frontend `HeurMode_t` value, including `A`, `B`,
`FALLBACK`, and `OPENSOURCE`. hipDNN does not currently expose matching cuDNN
heuristic modes, so non-`FALLBACK` modes are accepted but are not honored as
cuDNN heuristics. The shim logs a warning and forwards selection to hipDNN's
fallback/default engine-selection path.

Impact: plan choice and performance may differ from cuDNN for the same requested
heuristic mode.

## Numerical-note filters

hipDNN does not currently expose per-plan numerical-note metadata. The shim
therefore triages numerical-note filters by whether dropping the filter can change
correctness.

| Note | `select_numeric_notes` | `deselect_numeric_notes` |
|---|---|---|
| `NOT_SET` | no-op | no-op |
| `NONDETERMINISTIC` | warn and ignore | error |
| `REDUCED_PRECISION_REDUCTION` | warn and ignore | error |
| `TENSOR_CORE` | warn and ignore | warn and ignore |
| `DOWN_CONVERT_INPUTS` | warn and ignore | warn and ignore |
| `FFT` | warn and ignore | warn and ignore |
| `WINOGRAD` | warn and ignore | warn and ignore |
| `WINOGRAD_TILE_4x4` | warn and ignore | warn and ignore |
| `WINOGRAD_TILE_6x6` | warn and ignore | warn and ignore |
| `WINOGRAD_TILE_13x13` | warn and ignore | warn and ignore |
| `STRICT_NAN_PROP` | warn and ignore | warn and ignore |

`deselect_numeric_notes({NONDETERMINISTIC})` is treated as a request for a
deterministic plan. `deselect_numeric_notes({REDUCED_PRECISION_REDUCTION})` is
treated as a request for full-precision reduction. The shim cannot guarantee
either without backend metadata, so it records an error rather than silently
returning an unfiltered plan.

## Behavior-note filters

hipDNN exposes some per-engine behavior notes, but not cuDNN's CUDA-specific
behavior notes. The shim stores known hipDNN behavior-note filters and applies
them after native plan creation by querying applicable engine behavior metadata.

cuDNN-only behavior notes (below) are never emitted by hipDNN engines. The shim
triages them by whether ignoring the filter can violate the caller's request:

- `deselect_behavior_notes({<cuDNN-only note>})` asks to *exclude* engines that
  report the note. No hipDNN engine reports it, so nothing is excluded — the
  filter is a safe no-op. The shim logs a warning and continues.
- `select_behavior_notes({<cuDNN-only note>})` asks to *require* the note. No
  hipDNN engine can satisfy it, so the request is unsatisfiable. The shim records
  `GRAPH_NOT_SUPPORTED` rather than returning a plan that silently ignores the
  requirement.

The cuDNN-only behavior notes handled this way:

- `REQUIRES_FILTER_INT8x32_REORDER`
- `REQUIRES_BIAS_INT8x32_REORDER`
- `SUPPORTS_CUDA_GRAPH_NATIVE_API`
- `CUBLASLT_DEPENDENCY`

For behavior notes hipDNN *does* expose, both filters apply after native plan
creation. If known behavior-note filters remove every applicable hipDNN engine,
plan creation returns `GRAPH_NOT_SUPPORTED`.

When a filter bars only some engines, the shim narrows the candidate set the way
cuDNN does: `build_plans(HEURISTICS_CHOICE)` retargets onto the top-ranked
surviving plan rather than failing because the top-ranked one was barred.

## Engine IDs

cuDNN frontend presents integer engine IDs as dense graph-local indices bounded
by `get_engine_count()`.

hipDNN native engine IDs are stable hashes of engine names. The shim therefore
maintains a graph-local mapping:

```text
cuDNN-shaped engine index -> native hipDNN engine ID
```

`get_engine_count()` refreshes this map from hipDNN's ranked applicable engine
list. `get_knobs_for_engine`, `create_execution_plan`, and
`deselect_engines(vector<int64_t>)` translate cuDNN-shaped dense indices through
that map before calling native hipDNN APIs.

Impact: ordering follows hipDNN's ranked applicable engine list, not cuDNN's
backend ordering.

## Knobs

cuDNN frontend exposes knobs as a fixed `KnobType_t` enum plus integer
`minValue`, `maxValue`, and `stride` metadata. hipDNN native knobs use provider-
defined string IDs, variant-valued settings, descriptions, and richer constraints.
The shim uses explicit conversion rather than aliasing these incompatible models.

hipDNN knob IDs are namespaced (for example `global.workspace_size_limit`) while
cuDNN's `KnobType_t` is flat. The shim matches on the final dot-separated segment,
so both `tile_size` and `provider.tile_size` project onto the same cuDNN knob.
Matching is case-insensitive, and `workspace_size_limit` is the one bare name that
differs from its cuDNN counterpart (`workspace`).

When returning knobs from `get_knobs_for_engine`, native hipDNN knobs are projected
to cuDNN-shaped knobs only when all of the following are true:

- the native string knob ID maps to a known cuDNN `KnobType_t`
- the native knob value type is `int64`
- the native constraint can be represented as `minValue` / `maxValue` / `stride`

Native knobs that cannot be represented this way are omitted with a warning.

When creating a plan, each cuDNN `KnobType_t` key is resolved against the knobs the
target engine actually exposes, so a choice lands on the provider's own knob ID (for
example `miopen.tile_size`) rather than on a guessed bare name. A knob choice is
never silently dropped: `INVALID_VALUE` is returned for a key the engine does not
expose, for a key that more than one of the engine's knobs maps onto, and for a value
the native knob cannot hold (a string-valued knob, or a magnitude past 2^53 for a
float-valued knob). Integer choices are widened to `double` for float-valued knobs.

## Workspace and shared-memory caps

`deselect_workspace_greater_than` is forwarded to hipDNN and reapplied after plan
creation because native plan creation resets native filters.

`deselect_shared_mem_greater_than(0)` is a no-op. Any non-zero value records
`GRAPH_NOT_SUPPORTED`; hipDNN does not currently expose per-plan shared-memory
usage metadata for the shim to filter on.

## Plan and engine introspection

The shim forwards plan-name, plan-workspace, per-plan execution, knob query,
autotune, and current-plan behavior-note APIs where native hipDNN exposes an
equivalent.

`get_behavior_notes_for_plan_at_index` is best-effort. Native hipDNN exposes plan
name by index but not engine ID by index, so the shim resolves registered engine
names back to IDs. Unknown engine names fail rather than fabricating behavior
notes.

`warmup` is implemented as `execute_plan_at_index(..., 0)`.

## CUDA graph capture

`populate_cuda_graph` and `update_cuda_graph` are compile-time-present runtime
error stubs. They return `GRAPH_NOT_SUPPORTED` because there is no in-scope
HIP-graph capture analogue in the shim.

## Execute overloads

The UID-map and tensor-attribute-map execute overloads forward to native hipDNN.
The flat pointer-array `execute(cudnnHandle_t, void**, int, void*)` overload is
present for source compatibility but returns `INVALID_VALUE`; the shim cannot
safely reconstruct the required tensor UID mapping from a flat pointer array.

## Logging

`cudnnCreate()` performs a best-effort bridge from cuDNN frontend logging
environment variables to hipDNN logging configuration before creating the hipDNN
handle:

- enabled cuDNN logging maps to `HIPDNN_LOG_LEVEL=info`
- file-path `CUDNN_FRONTEND_LOG_FILE` maps to `HIPDNN_LOG_FILE`
- `stdout` / `stderr` targets enable hipDNN logging but do not set `HIPDNN_LOG_FILE`
- disabled or no-target cases map to `HIPDNN_LOG_LEVEL=off`

The bridge is effective only if it runs before hipDNN backend logging initializes.
The `CUDNN_FE_LOG*` macros forward to hipDNN frontend logging macros; they do not
own a separate cuDNN frontend stream logger.
