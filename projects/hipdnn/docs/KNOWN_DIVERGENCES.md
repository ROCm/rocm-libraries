# Known divergences: cuDNN frontend compatibility shim

This document describes intentional behavioral differences between NVIDIA cuDNN
frontend v9 and hipDNN's cuDNN-shaped compatibility shim. The shim is source
compatibility for supported v9 graph-API translation units, not ABI compatibility
and not a full cuDNN backend implementation.

NVIDIA, CUDA, and cuDNN are trademarks and/or registered trademarks of NVIDIA
Corporation. hipDNN is not affiliated with or endorsed by NVIDIA.

## Scope

The shim targets the cuDNN frontend v9 graph API surface exposed by
`<hipdnn_compatibility/cudnn/cudnn_frontend.h>`. The v0.x / v8 builder and backend
descriptor API surface is out of scope.

## hipDNN extensions

The shim exposes one method with no cuDNN frontend equivalent:

`Graph::is_supported_ext(cudnnHandle_t, std::vector<HeurMode_t> = {FALLBACK})` is a
lightweight applicability probe forwarding to the native hipDNN method of the same
name. It returns `OK` when at least one engine can run the graph and
`GRAPH_NOT_SUPPORTED` when none can. Like the native method it auto-builds the
operation graph when needed; it neither creates nor invalidates execution plans, so
it may be called before or after `create_execution_plans`. A node-less graph has
nothing to probe and is reported supported.

Because it has no cuDNN counterpart, source that must also compile against real
cuDNN cannot use it.

## Heuristic modes

The shim accepts every cuDNN frontend `HeurMode_t` value, including `A`, `B`,
`FALLBACK`, and `OPENSOURCE`. hipDNN does not currently expose matching cuDNN
heuristic modes, so non-`FALLBACK` modes are accepted but are not honored as
cuDNN heuristics. The shim logs a warning and forwards selection to hipDNN's
fallback/default engine-selection path.

Impact: plan choice and performance may differ from cuDNN for the same requested
heuristic mode.

## Note filters

hipDNN exposes no per-plan numerical-note metadata and reports only its own
behavior notes. The shim therefore triages each note by what ignoring the
request could do. The tables below are the canonical triage; the shim's
`note_triage.h` switches and their table test follow them.

| Action | Effect |
|---|---|
| `no-op` | accepted silently |
| `warn` | logged as a warning on every call, otherwise ignored |
| `map` | honored by filtering applicable engines; logged at info |
| `error` | records `GRAPH_NOT_SUPPORTED`, returned by the next `validate()`, plan creation, support check, or plan build |

### Numerical notes

| Note | `select_numeric_notes` | `deselect_numeric_notes` | Why |
|---|---|---|---|
| `NOT_SET` | no-op | no-op | placeholder |
| `TENSOR_CORE` | warn | warn | performance only |
| `DOWN_CONVERT_INPUTS` | warn | error | excluding it is a precision guarantee the shim cannot check |
| `REDUCED_PRECISION_REDUCTION` | warn | error | excluding it is a precision guarantee the shim cannot check |
| `FFT` | warn | warn | algorithm class |
| `NONDETERMINISTIC` | warn | map | deselect keeps only engines that claim determinism (see [Determinism](#determinism)) |
| `WINOGRAD`, `WINOGRAD_TILE_4x4`, `WINOGRAD_TILE_6x6`, `WINOGRAD_TILE_13x13` | warn | warn | algorithm class |
| `STRICT_NAN_PROP` | error | warn | selecting it is a NaN-propagation guarantee the shim cannot check; excluding it only relaxes |
| out-of-range value | error | error | unclassifiable |

Deselecting an out-of-range value is an error here but a no-op upstream:
without numerical metadata the shim cannot show the exclusion is vacuous.

### Behavior notes

Notes hipDNN engines report (`isKnownBehaviorNote`) map for both select and
deselect: the shim filters applicable engines by their reported behavior notes.
Any other value, including cuDNN's CUDA-specific notes, errors on select (no
hipDNN engine can satisfy it) and warns on deselect (it excludes nothing).

| Note | `select_behavior_notes` | `deselect_behavior_notes` | Why |
|---|---|---|---|
| `NOT_SET` | no-op | no-op | placeholder |
| `RUNTIME_COMPILATION` | map | map | hipDNN note |
| `REQUIRES_FILTER_INT8x32_REORDER` | error | warn | cuDNN only |
| `REQUIRES_BIAS_INT8x32_REORDER` | error | warn | cuDNN only |
| `SUPPORTS_CUDA_GRAPH_NATIVE_API` | error | warn | cuDNN only; `populate_cuda_graph` is unsupported, and this is not stream capture (`SUPPORTS_GRAPH_CAPTURE`) |
| `CUBLASLT_DEPENDENCY` | error | warn | cuDNN only; not equivalent to `EXTERNAL_LIBRARY_DEPENDENCY` |
| `REQUIRES_LAYOUT_TRANSFORM` | map | map | hipDNN note |
| `SUPPORTS_GRAPH_CAPTURE` | map | map | hipDNN note |
| `EXTERNAL_LIBRARY_DEPENDENCY` | map | map | hipDNN note |
| `SUPPORTS_EXECUTION_PLAN_SERIALIZATION` | map | map | hipDNN note |
| out-of-range value | error | warn | reported by no hipDNN engine |

### Determinism

`deselect_numeric_notes({NONDETERMINISTIC})` keeps only engines that positively
claim deterministic results. hipDNN engines declare no numerical notes, so a
missing claim proves nothing. Until engine metadata carries the claim, the shim
keeps its own allowlist, today `MIOPEN_ENGINE_DETERMINISTIC`.

## Engine filtering

`deselect_engines`, the behavior-note filters, and the determinism filter are
applied to the applicable engines after native plan creation. A filter set after
plans were created takes effect at the next support check or plan build,
including for plans already created.

When the filters bar only some engines, the shim narrows the candidate set the
way cuDNN does: `build_plans(HEURISTICS_CHOICE)` retargets onto the top-ranked
surviving plan rather than failing because the top-ranked one was barred.

If no created plan survives the filters, the next plan creation
(`create_execution_plans`, `create_execution_plan`), support check
(`check_support()`), or plan build (`build_plans()`, `build_plan_at_index()`)
returns `GRAPH_NOT_SUPPORTED`; upstream returns
`GRAPH_EXECUTION_PLAN_CREATION_FAILED` from `check_support()`. This includes
`deselect_engines` alone barring every applicable engine.

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
name by index but not engine ID by index, so the shim resolves the plan name back
to an ID: built-in engine names by hash, and every other engine (including plugin
engines) through the `0x`-prefixed hexadecimal ID native reports for it.

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
