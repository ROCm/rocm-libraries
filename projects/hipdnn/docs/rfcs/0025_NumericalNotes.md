# RFC 0025: Per-engine numerical notes

- Contributors: TBD
- **Status**: Implemented
- **Status notes**: Backend `HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE`, `EngineDetails.numerical_notes`, `Graph::get_numerical_notes_for_engine()`, the Python binding, the UED `numerical_notes` vocabulary, the MIOpen determinism claims, and real-note filtering in the cuDNN v9 and v8 compatibility layers ship together. No in-tree provider reports a precision note.
- **Last verified**: 2026-11-10 against `develop` (1c63e26254a)
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

A numerical note is metadata an engine reports about the numerics of its results, such as `NONDETERMINISTIC`. hipDNN reports numerical notes the way [RFC 0010](./0010_BehaviorNotes.md) reports behavior notes: a plugin writes them into `EngineDetails` for the finalized graph, the backend serves them through `HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE`, and the frontend returns them from `Graph::get_numerical_notes_for_engine()`. The cuDNN compatibility layers read the same attribute to filter engines.

Where this RFC is silent, RFC 0010 applies with "numerical" for "behavior". This RFC records the differences.

## 2. Design

- **Enum.** `hipdnnBackendNumericalNote_t` is an `int32_t` with cuDNN's ten values: `TENSOR_CORE=0`, `DOWN_CONVERT_INPUTS=1`, `REDUCED_PRECISION_REDUCTION=2`, `FFT=3`, `NONDETERMINISTIC=4`, `WINOGRAD=5`, `WINOGRAD_TILE_4x4=6`, `WINOGRAD_TILE_6x6=7`, `WINOGRAD_TILE_13x13=8`, `STRICT_NAN_PROP=9`, `TYPE_COUNT=10`. It is defined in `backend/include/HipdnnBackendNumericalNote.h` and duplicated, under the same include guard, in `plugin_sdk/include/hipdnn_plugin_sdk/NumericalNote.h`; plugins include the latter explicitly.
- **Metadata.** `engine_details.fbs` carries `numerical_notes: [int32]`. `EngineDetailsWrapper::numericalNotes()` returns `std::nullopt` when the field is absent. Plugins write the field with `CreateEngineDetails` or `CreateEngineDetailsDirect`; `EngineDetailsT::Pack` drops an empty vector and cannot express an explicit empty list.
- **Backend.** `EngineDescriptor` loads and caches the notes with the other engine details and serves `HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE` as `HIPDNN_TYPE_NUMERICAL_NOTE`, get-only. A negative value fails with `BAD_PARAM`; an unknown non-negative value is passed through. Set returns `NOT_SUPPORTED`.
- **Frontend.** `Graph::get_numerical_notes_for_engine(int64_t, std::vector<NumericalNote>&) const` follows the contract of RFC 0010 §5. Values the frontend does not know are dropped. `fromHipdnnNumericalNote` and `isKnownNumericalNote` in `Types.hpp` convert between the backend and frontend numberings.
- **Python.** `hipdnn_frontend.NumericalNote`, also spelled `numerical_note` as in cuDNN's Python frontend, and `Graph.get_numerical_notes_for_engine(engine_id)`, which raises `RuntimeError` on error.
- **Scope of a note.** Notes describe an engine for the finalized graph passed to `GetEngineDetails(engineId, opGraph)`. There are no plan-, configuration- or knob-level notes. The plugin API version does not change with the note field.

## 3. Deltas from RFC 0010

| Topic | Behavior notes | Numerical notes |
|---|---|---|
| Absent `EngineDetails` field | Empty list | Not guaranteed reproducible: the backend reports `{NONDETERMINISTIC}` |
| Explicit empty list | Same as absent | The provider's claim that no numerical note applies, that is, that the engine is deterministic; a test must prove it |
| Enum | hipDNN values 0-4 | cuDNN's ten values |
| UED vocabulary | `runtime_compilation`, `supports_graph_capture` | The ten cuDNN note names, lowercase; an unknown name or a repeat is a load error |
| Native select and deselect | Not provided | Not provided |

`NONDETERMINISTIC` means the results are not guaranteed bitwise reproducible. Every engine defaults to it, so an absent field is accurate for an engine nobody has proven otherwise, and it needs no plugin API bump. A provider claims determinism with an explicit empty list only when a test proves bitwise-reproducible results across repeated runs, as `IntegrationGpuDeterministic` does for `MIOPEN_ENGINE_DETERMINISTIC`. The default matches cuDNN's meaning of the note and is not a parity divergence.

A UED may declare any of the ten cuDNN numerical notes by lowercase name. The UED loader and the hip-kernel packer map each name to its `HIPDNN_NUMERICAL_NOTE_*` value and reject an unknown or repeated name. hipDNN acts only on `NONDETERMINISTIC` today. A UED without a `numerical_notes` key leaves the field absent; an authored list, `[]` or one without `nondeterministic`, claims determinism, so an engine not proven deterministic lists `nondeterministic` beside its other notes. See [RFC 0020](./0020_UniversalEngineDescriptor.md) §9.

## 4. cuDNN parity

| hipDNN | cuDNN | Status |
|---|---|---|
| `hipdnnBackendNumericalNote_t` values | `cudnnBackendNumericalNote_t` | Parity |
| `HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE`, read-only, zero or more `HIPDNN_TYPE_NUMERICAL_NOTE` | `CUDNN_ATTR_ENGINE_NUMERICAL_NOTE` | Parity of meaning: an engine that declares nothing is reported `{NONDETERMINISTIC}`, not guaranteed reproducible. A buffer smaller than the note count fails with `BAD_PARAM`; cuDNN truncates to the requested count. The element type is `int32_t`, so an unknown value from a newer plugin passes through; cuDNN's is a C enum |
| `hipdnn_frontend.NumericalNote` | `cudnn.numerical_note` | Match: the same ten members under both `NumericalNote` and cuDNN's `numerical_note` |
| v9 `select_numeric_notes` and `deselect_numeric_notes` for `NONDETERMINISTIC` | Same functions | Parity of the filter |
| v8 `hasNumericalNote`, `ExecutionPlan::getNumericNotes`, `ExecutionPlan::getAllNumericNotes` | Same functions | Parity: same read path; an engine without a proven claim reports `{NONDETERMINISTIC}` |
| Native and Python frontends: select and deselect of numerical notes | Python `Graph.select_numeric_notes` and `deselect_numeric_notes` | Not provided (§7); only the v9 compatibility layer filters by note |
| `Graph::get_numerical_notes_for_engine` | None: cuDNN's frontend has no per-engine numerical-note getter in its graph API | hipDNN extension |
| UED `numerical_notes` | None: cuDNN has no descriptor files | A closed vocabulary of the ten cuDNN note names. A descriptor that names an unknown tag fails to load |
| Compatibility-layer plan JSON | The layer's own format, not cuDNN's | Format `version` `1` with `engine.numericalNotes`; a reader without that key rejects the plan as an unknown engine key, so plans saved with notes do not load in builds without notes |

Compatibility-layer divergences from cuDNN are in [KNOWN_DIVERGENCES.md](../KNOWN_DIVERGENCES.md#numerical-notes).

## 5. Provider claims

| Engine | Numerical notes |
|---|---|
| `MIOPEN_ENGINE_DETERMINISTIC` | Explicit `[]` |
| `MIOPEN_ENGINE` | Explicit `[NONDETERMINISTIC]` |
| Every other engine | No tested claim: field absent, reported `{NONDETERMINISTIC}` |

`MIOPEN_ENGINE_DETERMINISTIC` claims determinism for every graph it accepts; the claim is about the engine ID, not per-solver evidence. No in-tree provider reports a precision note. Precision claims, such as MIOpen's `FP16_ALT_IMPL` on gfx90a and hipBLASLt compute types, are future work.

## 6. Compatibility layers

- **v9 graph API.** `select_numeric_notes` and `deselect_numeric_notes` of `NONDETERMINISTIC` map to filtering engines by `get_numerical_notes_for_engine()`, and a failed query propagates as an error. If no engine survives, the shim returns `GRAPH_NOT_SUPPORTED`. The shim filters on `NONDETERMINISTIC` only; every other numerical-note value follows the disposition table in [KNOWN_DIVERGENCES](../KNOWN_DIVERGENCES.md).
- **v8 builders.** `hasNumericalNote`, `ExecutionPlan::getNumericNotes` and `getAllNumericNotes` read `HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE`. Only `hasNumericalNote` answers false when the query fails; a failed query fails `ExecutionPlanBuilder::build()`, as upstream, except that `NOT_SUPPORTED` (the status of an older same-major backend that lacks the attribute) reads as `{NONDETERMINISTIC}`. The native `Graph::get_numerical_notes_for_engine`, and so the v9 filters and the Python binding, read `NOT_SUPPORTED` the same way: an older same-major backend yields `{NONDETERMINISTIC}`, and every other failure is an error. The plan JSON always writes `engine.numericalNotes` next to `behaviorNotes`, as the compatibility layer's own `cudnnBackendNumericalNote_t` values (mapped to hipDNN's by name, not NVIDIA's numbering), range-checked on load. JSON without the field loads as `{NONDETERMINISTIC}`. The JSON `version` is `1`; a reader without `numericalNotes` rejects it (see the plan-JSON row of §4).

See [RFC 0024](./0024_CuDNN_V8_Compatibility.md) and [RFC 0012](./0012_CuDNN_Shim.md).

## 7. Out of scope

- Native `Graph` select and deselect by numerical note.
- Precision notes from any in-tree provider.
- A plugin API version bump.
