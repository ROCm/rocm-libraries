# RFC 0025: Per-engine numerical notes

- Contributors: TBD
- **Status**: Implemented
- **Status notes**: Backend `HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE`, `EngineDetails.numerical_notes`, `Graph::get_numerical_notes_for_engine()`, the Python binding, the UED `numerical_notes` vocabulary, the MIOpen determinism claims, note filtering on the native and Python `Graph` with cuDNN's filter semantics for every engine filter, and real-note filtering in the cuDNN v9 and v8 compatibility layers ship together. No in-tree provider reports a precision note.
- **Last verified**: 2026-11-13 against `develop` (7d8674e8542)
- **Follows**: [RFC 0010: Per-engine behavior notes](./0010_BehaviorNotes.md)

## Table of Contents

1. [Summary](#1-summary)
2. [Design](#2-design)
3. [Deltas from RFC 0010](#3-deltas-from-rfc-0010)
4. [cuDNN parity](#4-cudnn-parity)
5. [Provider claims](#5-provider-claims)
6. [Compatibility layers](#6-compatibility-layers)
7. [Out of scope](#7-out-of-scope)

## 1. Summary

A numerical note is metadata an engine reports about the numerics of its results, such as `NONDETERMINISTIC`. hipDNN reports numerical notes the way [RFC 0010](./0010_BehaviorNotes.md) reports behavior notes: a plugin writes them into `EngineDetails` for the finalized graph, the backend serves them through `HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE`, and the frontend returns them from `Graph::get_numerical_notes_for_engine()`. The native and Python `Graph` and the cuDNN compatibility layers read the same attribute to filter engines.

Where this RFC is silent, RFC 0010 applies with "numerical" for "behavior". This RFC records the differences.

## 2. Design

- **Enum.** `hipdnnBackendNumericalNote_t` is an `int32_t` with cuDNN's ten values: `TENSOR_CORE=0`, `DOWN_CONVERT_INPUTS=1`, `REDUCED_PRECISION_REDUCTION=2`, `FFT=3`, `NONDETERMINISTIC=4`, `WINOGRAD=5`, `WINOGRAD_TILE_4x4=6`, `WINOGRAD_TILE_6x6=7`, `WINOGRAD_TILE_13x13=8`, `STRICT_NAN_PROP=9`, `TYPE_COUNT=10`. It is defined in `backend/include/HipdnnBackendNumericalNote.h` and duplicated, under the same include guard, in `plugin_sdk/include/hipdnn_plugin_sdk/NumericalNote.h`; plugins include the latter explicitly.
- **Metadata.** `engine_details.fbs` carries `numerical_notes: [int32]`. `EngineDetailsWrapper::numericalNotes()` returns `std::nullopt` when the field is absent. Plugins write the field with `CreateEngineDetails` or `CreateEngineDetailsDirect`; `EngineDetailsT::Pack` drops an empty vector and cannot express an explicit empty list.
- **Backend.** `EngineDescriptor` loads and caches the notes with the other engine details and serves `HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE` as `HIPDNN_TYPE_NUMERICAL_NOTE`, get-only. A negative value fails with `BAD_PARAM`; an unknown non-negative value is passed through. Set returns `NOT_SUPPORTED`.
- **Frontend.** `Graph::get_numerical_notes_for_engine(int64_t, std::vector<NumericalNote>&) const` follows the contract of RFC 0010 §5. Values the frontend does not know are dropped. `fromHipdnnNumericalNote` and `isKnownNumericalNote` in `Types.hpp` convert between the backend and frontend numberings.
- **Python.** `hipdnn_frontend.NumericalNote`, also spelled `numerical_note` as in cuDNN's Python frontend, and `Graph.get_numerical_notes_for_engine(engine_id)`, which raises `RuntimeError` on error. `Graph.select_numeric_notes`, `deselect_numeric_notes`, `select_behavior_notes` and `deselect_behavior_notes` return the graph.
- **Filtering.** `Graph::select_numeric_notes`, `deselect_numeric_notes`, `select_behavior_notes` and `deselect_behavior_notes` append to four lists and return the graph; an empty argument is a no-op. An engine passes when it reports every selected note and none of the deselected ones, in each family, and it must pass both families. A plan whose engine fails is filtered, like one removed by `deselect_engines` or `deselect_workspace_greater_than` ([RFC 0013](./0013_Autotune.md)): `check_support()` answers "is this supported with these filters?", and `build_plans`, `build_plan_at_index` and `autotune` skip it. An engine's notes are read only when a note filter is set, only for the engines a call evaluates, and once per filter state; a failed read returns the getter's error from the evaluating call. Filters persist until the `Graph` is destroyed, and they apply to plans created later, including plans from `add_engine_*()`.
- **Scope of a note.** Notes describe an engine for the finalized graph passed to `GetEngineDetails(engineId, opGraph)`. There are no plan-, configuration- or knob-level notes. The plugin API version does not change with the note field.

## 3. Deltas from RFC 0010

| Topic | Behavior notes | Numerical notes |
|---|---|---|
| Absent `EngineDetails` field | Empty list | Not guaranteed reproducible: the backend reports `{NONDETERMINISTIC}` |
| Explicit empty list | Same as absent | The provider's claim that no numerical note applies, that is, that the engine is deterministic; a test must prove it |
| Enum | hipDNN values 0-4 | cuDNN's ten values |
| UED vocabulary | `runtime_compilation`, `supports_graph_capture` | The ten cuDNN note names, lowercase; an unknown name or a repeat is a load error |
| Native and Python select and deselect | `select_behavior_notes`, `deselect_behavior_notes` | `select_numeric_notes`, `deselect_numeric_notes` |

`NONDETERMINISTIC` means the results are not guaranteed bitwise reproducible. Every engine defaults to it, so an absent field is accurate for an engine nobody has proven otherwise, and it needs no plugin API bump. A provider claims determinism with an explicit empty list only when a test proves bitwise-reproducible results across repeated runs, as `IntegrationGpuDeterministic` does for `MIOPEN_ENGINE_DETERMINISTIC`. The default matches cuDNN's meaning of the note and is not a parity divergence.

A UED may declare any of the ten cuDNN numerical notes by lowercase name. The UED loader and the hip-kernel packer map each name to its `HIPDNN_NUMERICAL_NOTE_*` value and reject an unknown or repeated name. The native and Python filters act on every note an engine reports; the v9 layer acts only on `NONDETERMINISTIC`. A UED without a `numerical_notes` key leaves the field absent; an authored list, `[]` or one without `nondeterministic`, claims determinism, so an engine not proven deterministic lists `nondeterministic` beside its other notes. See [RFC 0020](./0020_UniversalEngineDescriptor.md) §9.

## 4. cuDNN parity

| hipDNN | cuDNN | Status |
|---|---|---|
| `hipdnnBackendNumericalNote_t` values | `cudnnBackendNumericalNote_t` | Parity |
| `HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE`, read-only, zero or more `HIPDNN_TYPE_NUMERICAL_NOTE` | `CUDNN_ATTR_ENGINE_NUMERICAL_NOTE` | Parity of meaning: an engine that declares nothing is reported `{NONDETERMINISTIC}`, not guaranteed reproducible. A buffer smaller than the note count fails with `BAD_PARAM`; cuDNN truncates to the requested count. The element type is `int32_t`, so an unknown value from a newer plugin passes through; cuDNN's is a C enum |
| `hipdnn_frontend.NumericalNote` | `cudnn.numerical_note` | Match: the same ten members under both `NumericalNote` and cuDNN's `numerical_note` |
| v9 `select_numeric_notes` and `deselect_numeric_notes` for `NONDETERMINISTIC` | Same functions | Parity of the filter |
| v8 `hasNumericalNote`, `ExecutionPlan::getNumericNotes`, `ExecutionPlan::getAllNumericNotes` | Same functions | Parity: same read path; an engine without a proven claim reports `{NONDETERMINISTIC}` |
| Native and Python frontends: select and deselect of numerical notes | Python `Graph.select_numeric_notes` and `deselect_numeric_notes` | Parity of the match rule: an engine passes when it reports every selected note and none of the deselected ones (cuDNN accumulates the same rule). The differences are listed below |
| `Graph::get_numerical_notes_for_engine` | None: cuDNN's frontend has no per-engine numerical-note getter in its graph API | hipDNN extension |
| UED `numerical_notes` | None: cuDNN has no descriptor files | A closed vocabulary of the ten cuDNN note names. A descriptor that names an unknown tag fails to load |
| Compatibility-layer plan JSON | The layer's own format, not cuDNN's | Format `version` `1` with `engine.numericalNotes`; a reader without that key rejects the plan as an unknown engine key, so plans saved with notes do not load in builds without notes |

### Differences from cuDNN filtering

Every engine filter (`deselect_engines`, the note filters, `deselect_workspace_greater_than`) persists until changed. `check_support` answers "is this supported with these filters?", and `build_plans` builds the best engine that passes them, keeping its fallback and logging. The differences:

- A failed note read fails the evaluating call, and only when a note filter is set. cuDNN fails `create_execution_plans` with `HEURISTIC_QUERY_FAILED` on every call.
- Note filters set before plans exist apply to the plans created later; cuDNN ignores them.
- `add_engine_*()` autotune candidates are filtered. Compiled-plan `autotune` applies filters set after `build_plans(ALL)`; cuDNN times every built plan.
- A finalized plan over the workspace limit despite its engine config is rejected after it finalizes. cuDNN 9.2+ trusts the config value. The plugin contract makes the engine-config workspace an upper bound on the workspace of any plan finalized from it, so the rejection is a safety net.
- Python filter methods return the graph; cuDNN returns `None`.
- `build_plan_at_index` on a plan whose engine-config workspace cannot be read returns `GRAPH_EXECUTION_PLAN_CREATION_FAILED`; cuDNN returns the read error. `check_support` and the build walks skip such a plan, as cuDNN does.
- A `deselect_engines` or note filter cannot be removed from a `Graph`: lists only grow, and creating plans does not reset them. cuDNN's `deselect_engines` replaces its list, so `deselect_engines({})` clears it. Use a new `Graph` to drop a filter.
- A repeated `build_plans(HEURISTICS_CHOICE)` after a build returns the backend's already-finalized error and leaves the active plan unchanged; cuDNN returns OK.
- `check_support()` marks no plan. cuDNN's `check_support` marks the plans it removes.
- A plan an earlier evaluation barred stays barred when a filter is relaxed or cleared, until the plans are recreated (`create_execution_plans()`, `create_execution_plan_ext()`, rebuilding the operation graph, deserialization or plan-spec `autotune()`).
- A deserialized plan is never filtered: `check_support()` treats it as passing and `build_plans(ALL)` skips it. `build_plan_at_index()` on it returns `INVALID_VALUE`, because it has no engine config.
- `build_plans(ALL)` with no filter set and no plan built returns `HIPDNN_BACKEND_ERROR`; with a filter set it returns `GRAPH_EXECUTION_PLAN_CREATION_FAILED`. Both apply only when no plan is active after the call: once a plan is built, later filter changes do not undo it, so `build_plans(ALL)` returns OK even when a new filter removes the active plan, which stays active and executable (cuDNN `plans.h`). `build_plan_at_index` on a filtered index, the active one included, still returns the no-survivor code and leaves the active plan untouched.
- `deselect_engines` matches exact engine IDs; cuDNN matches name substrings.
- A negative workspace limit means no limit; in cuDNN it bars every config.
- `hipdnn.Graph.check_support` and `build_plans` return `Error`; `pygraph` raises, as cuDNN does.
- Behavior-note members are hipDNN's in C++ and Python.

Compatibility-layer divergences from cuDNN are in [KNOWN_DIVERGENCES.md](../KNOWN_DIVERGENCES.md#numerical-notes).

## 5. Provider claims

| Engine | Numerical notes |
|---|---|
| `MIOPEN_ENGINE_DETERMINISTIC` | Explicit `[]` |
| `MIOPEN_ENGINE` | Explicit `[NONDETERMINISTIC]` |
| Every other engine | No tested claim: field absent, reported `{NONDETERMINISTIC}` |

`MIOPEN_ENGINE_DETERMINISTIC` claims determinism for every graph it accepts; the claim is about the engine ID, not per-solver evidence. No in-tree provider reports a precision note. Precision claims, such as MIOpen's `FP16_ALT_IMPL` on gfx90a and hipBLASLt compute types, are future work.

## 6. Compatibility layers

- **v9 graph API.** `select_numeric_notes` and `deselect_numeric_notes` of `NONDETERMINISTIC` map to filtering engines by `get_numerical_notes_for_engine()`, and a failed query propagates as an error. If no engine survives, the shim returns `GRAPH_NOT_SUPPORTED`. The shim filters on `NONDETERMINISTIC` only; every other numerical-note value follows the disposition table in [KNOWN_DIVERGENCES](../KNOWN_DIVERGENCES.md). The shim's engine filters reach the native `Graph` as `deselect_engines`, and `build_plans` calls the native `build_plans`, which builds the next plan the filters leave. The workspace limit applies to that walk, to `check_support` and to `build_plan_at_index`.
- **v8 builders.** `hasNumericalNote`, `ExecutionPlan::getNumericNotes` and `getAllNumericNotes` read `HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE`. Only `hasNumericalNote` answers false when the query fails; a failed query fails `ExecutionPlanBuilder::build()`, as upstream, except that `NOT_SUPPORTED` (the status of an older same-major backend that lacks the attribute) reads as `{NONDETERMINISTIC}`. The native `Graph::get_numerical_notes_for_engine`, and so the v9 filters and the Python binding, read `NOT_SUPPORTED` the same way: an older same-major backend yields `{NONDETERMINISTIC}`, and every other failure is an error. The plan JSON always writes `engine.numericalNotes` next to `behaviorNotes`, as the compatibility layer's own `cudnnBackendNumericalNote_t` values (mapped to hipDNN's by name, not NVIDIA's numbering), range-checked on load. JSON without the field loads as `{NONDETERMINISTIC}`. The JSON `version` is `1`; a reader without `numericalNotes` rejects it (see the plan-JSON row of §4).

See [RFC 0024](./0024_CuDNN_V8_Compatibility.md) and [RFC 0012](./0012_CuDNN_Shim.md).

## 7. Out of scope

- Precision notes from any in-tree provider.
- A plugin API version bump. The engine-config workspace query MUST return at least the workspace of any plan finalized from that config, and benchmarking-config coverage is also a MUST; both change documentation only. The ABI is unchanged, the in-tree plugins already comply, and the frontend's safety net rejects a plan that finalizes over the limit, so a non-compliant plugin degrades gracefully.
