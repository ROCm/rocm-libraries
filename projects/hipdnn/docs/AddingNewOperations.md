# Adding a New Operation to hipDNN

A contributor walkthrough for landing a new op across the hipDNN stack. The code generator handles the bulk of the implementation — schema wiring, descriptor boilerplate, frontend plumbing, and tests — leaving only review and a few judgment-heavy pieces for you. Without agentic AI, the same steps apply but the final integration details and polishing will need to be completed by hand.

## Table of Contents

**Quick Start**
- [Two Decisions Before You Run the Agent](#two-decisions-before-you-run-the-agent)
- [Run `/hipdnn-codegen`](#run-hipdnn-codegen)
- [Review Implementation](#review-implementation)
- [PR Checklist](#pr-checklist)

**Supplementary Reference** *(read when you need it, not by default)*
- [cuDNN Parity Rules](#cudnn-parity-rules)
- [File Map (PR Diff Template)](#file-map-pr-diff-template)
- [Layer-by-Layer Reference](#layer-by-layer-reference)
- [Testing Requirements](#testing-requirements)
- [Troubleshooting](#troubleshooting)
- [Reference Links](#reference-links)

---

# Quick Start

Adding a new op touches FlatBuffers schemas, backend descriptors, frontend nodes, JSON utilities, and tests at four levels. Most of it is mechanical.

> [!IMPORTANT]
> Preferred workflow: use an agentic AI that can run the [`hipdnn-codegen`](../tools/DescriptorGenerator/.claude/skills/hipdnn-codegen/SKILL.md) skill. The skill drives the full workflow — schema construction, code generation, file placement, build, and tests — with only a handful of confirmation prompts from you.

```
Decisions (you) → /hipdnn-codegen (agent) → Review implementation (you) → PR
```

**Without agentic AI:** either follow the [`hipdnn-codegen` skill file](../tools/DescriptorGenerator/.claude/skills/hipdnn-codegen/SKILL.md) as the manual procedure, or use a chat-only AI as a reading assistant while you carry out the referenced repo edits yourself. Pair it with the [Layer-by-Layer Reference](#layer-by-layer-reference) and [Testing Requirements](#testing-requirements) below.

## Two Decisions Before You Run the Agent

1. **FBS schema** — decide whether to write the schema yourself or let the agent construct it. If you have a schema or a written description of one, point the agent at it. Otherwise, leave it to the agent: it derives the schema from the cudnn-frontend source and asks you to confirm before continuing.
2. **Op semantics** — anything that diverges from or extends cuDNN: discrete mode enum members (e.g., reduction op type, pointwise kind), extra inputs, different output-shape rules, hipDNN-only behavior. The agent uses mode members to populate the `enum_def` block; deviations become comments in the generated code.

The agent handles cuDNN naming parity (`_EXT` decisions) automatically and will prompt you if it cannot determine the cuDNN equivalent. See [cuDNN Parity Rules](#cudnn-parity-rules) for details.

## Run `/hipdnn-codegen`

```
/hipdnn-codegen <op-name>
```

| Scenario | Mode |
|---|---|
| Brand new operation (nothing exists yet) | `--mode full` |
| Adding backend only (frontend exists or will land later) | `--mode backend` |
| Adding frontend only (backend descriptor already exists) | `--mode frontend` |

The agent will:

1. Construct (or accept) the FBS schema and **ask you to confirm it** before continuing — this is the mandatory user gate.
2. Wire the schema into `flatbuffers_sdk/schemas/graph.fbs` and `flatbuffers_sdk/CMakeLists.txt`, then rebuild the FlatBuffers SDK target.
3. Pick the next free descriptor type and attribute name range, derive the YAML config, run the generator.
4. Place generated source files; insert fragment snippets into shared files (enum headers, factory switches, string utilities, CMake lists).
5. Build the project and run unit tests.
6. May ask clarifying questions afterward on isolated pieces of logic (e.g., output-shape strategy, custom validation rules).

If you've already written the FBS schema or have a written description of one, point the agent at it instead.

## Review Implementation

After the agent finishes, go through the output before opening a PR:

- **Schema** — re-read the generated FBS schema. Confirm field names, types, and any shared includes look right. The agent asks for confirmation before continuing, but a second read after generation catches anything missed in the preview.
- **Backend enums vs cuDNN** — open [`docs/PortingGuide.md`](./PortingGuide.md) and the cudnn-frontend node header side-by-side with the generated `HipdnnBackend*.h` entries. Verify `_EXT` is applied where expected and absent where the name matches cuDNN exactly. Pay particular attention to the compute-data-type attribute: `MATH_PREC` vs `COMP_TYPE` follows cuDNN — check what the cuDNN equivalent (or the closest op in the same family) uses if unsure.
- **Node logic** — read `<Op>Node.hpp` alongside the corresponding cudnn-frontend node class (`include/cudnn_frontend/node/<op>.h`). Compare `infer_properties_node()` and `pre_validate_node()` against the cuDNN equivalent: fill in stubs, add any validation rules or shape logic that cuDNN enforces but the agent did not emit, and verify that optional tensors and fields still have appropriate validation (e.g., conditional dim checks, presence guards).
- **Test cases** — skim the generated integration tests. Add operation-specific tests beyond the round-trips where needed: multi-input variants, multi-op graph chains, edge cases (zero-sized dims, fused activations).

**Optional (if applicable):**

- **End-user sample** under `samples/<op>/` — add if an engine is available for the op. Wrap `graph->build(handle)` with `HIPDNN_FE_CHECK_SKIPPABLE` so the test is gracefully skipped when no engine supports the configuration. See `samples/convolution/ConvFprop.cpp` as a template.
- **Python bindings** — the one piece the agent does not generate automatically. Add a `.def(...)` line in `python/src/graph_bindings.cpp` and register the attributes class in `python/src/attributes_bindings.cpp` (check that file for the current curated set). You can also explicitly ask the agent to do this before it finishes.

## PR Checklist

Copy-paste this into your PR description.

```markdown
## Adding-a-New-Operation Checklist

### Review
- [ ] FBS schema field names, types, and shared includes look correct
- [ ] cuDNN equivalent identified (per docs/PortingGuide.md); attribute set and semantics aligned
- [ ] `_EXT` suffix correct: present on hipDNN-specific names, absent where the name matches cuDNN exactly (numeric value parity is NOT required)
- [ ] `infer_properties_node()` and `pre_validate_node()` reviewed against the cudnn-frontend node class; stubs filled in, validation rules aligned, and optional tensors/fields have appropriate presence guards
- [ ] Integration tests cover the op's meaningful variants; no important edge cases missing
- [ ] (Optional) Python bindings updated if op is on the Python surface

### Implementation
- [ ] FBS schema added in `flatbuffers_sdk/schemas/` (NOT `data_sdk/`)
- [ ] Schema added to the `SCHEMAS` list in `flatbuffers_sdk/CMakeLists.txt`
- [ ] `graph.fbs` `NodeAttributes` union updated and include added
- [ ] `hipdnn_flatbuffers_sdk` target rebuilt; generated headers committed
- [ ] Backend descriptor type enum value assigned
- [ ] Backend attribute name range assigned
- [ ] (If new mode enum) `HipdnnBackendAttributeType.h` type tag added; `Hipdnn<Op>Mode.h` written; `hipdnn_backend.h` updated; `DataTypeConversion.{hpp,cpp}` and `DescriptorAttributeUtils.{hpp,cpp}` updated
- [ ] Backend descriptor + DescriptorFactory + NodeFactory + BackendEnumStringUtils.hpp updated
- [ ] Frontend `Types.hpp` enum + `toBackend*` + `fromHipdnn*` converters present (if new mode enum)
- [ ] Frontend attributes + packer + unpacker + OperationUnpacker switch + node + Graph API method present
- [ ] All `CMakeLists.txt` updated (backend src, backend tests, frontend tests, tests/frontend)
- [ ] All test layers from the matrix have at least one test (see Testing Requirements)
- [ ] Lowering + lifting round-trip + tensor sharing + auto-UID + per-scalar tests present
- [ ] `TestBackendEnumStringUtils.cpp` updated with new enum entries
- [ ] `ninja unit-check` and `ninja integration-check` pass
- [ ] `ninja format` + `ninja check_format` pass
- [ ] ASAN build (`cmake -DBUILD_ADDRESS_SANITIZER=ON ..` + `ninja check`) clean
- [ ] No clang-tidy errors, no compiler warnings
- [ ] Test coverage ≥80% (no regression)
```

---

# Supplementary Reference

The sections below are for when you need them — debugging an agent failure, reviewing a PR, or doing a layer manually.

## cuDNN Parity Rules

hipDNN aims for **naming parity** with cuDNN where the operation has a cuDNN equivalent. The `_EXT` suffix encodes "this name has no cuDNN equivalent (or differs from cuDNN)".

> [!IMPORTANT]
> Parity is **nominal, not numeric**. Enum *names* mirror cuDNN; enum *values* do not. hipDNN values pack densely from 1 (e.g., `HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR = 31`); attribute names use clean 100/200/300 ranges. The agent picks the next free value when generating an enum entry.

### The `_EXT` Rule

A name **matches cuDNN** when everything after the `HIPDNN_` / `CUDNN_` prefix is identical. If a match exists, omit `_EXT`; if there is no cuDNN equivalent or the shape differs, add it.

| Case | Example | Suffix |
|---|---|---|
| Exact cuDNN equivalent | `HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR` ↔ `CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR` | **No** `_EXT` |
| hipDNN-specific (no cuDNN equivalent) | `HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT` — no cuDNN operation-type enum | `_EXT` |
| hipDNN name differs from cuDNN equivalent | `HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT` — cuDNN uses `CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR`, not a batchnorm-specific name | `_EXT` |

The decision applies to four name surfaces: the descriptor type name, each per-tensor / per-data-field attribute name, the compute-data-type attribute name, and the operation-type enum name. Reviewers verify by inspecting the resulting `HipdnnBackend*.h` enums.

### Sources of Truth

- [`docs/PortingGuide.md`](./PortingGuide.md) — cuDNN ↔ hipDNN API mapping
- [`docs/OperationSupport.md`](./OperationSupport.md) — which cuDNN ops are currently supported
- The open-source [`cudnn-frontend`](https://github.com/NVIDIA/cudnn-frontend) headers — for the canonical `CUDNN_*` constant identifiers as referenced by the wrapper

### Recording Deviations

A `_EXT` suffix is itself the marker for "not in cuDNN". Add a short comment in the descriptor or schema for any non-obvious semantic difference.

---

## File Map (PR Diff Template)

The complete surface area for a single op, using **Matmul** as the canonical example. Use as the diff template when reviewing your own PR.

| Layer | File |
|---|---|
| FBS schema | `flatbuffers_sdk/schemas/matmul_attributes.fbs` |
| FBS union entry | `flatbuffers_sdk/schemas/graph.fbs` (`NodeAttributes` union) |
| FBS CMake list | `flatbuffers_sdk/CMakeLists.txt` (`SCHEMAS` variable) |
| Backend descriptor type enum | `backend/include/HipdnnBackendDescriptorType.h` |
| Backend attribute name enum | `backend/include/HipdnnBackendAttributeName.h` |
| Backend attribute type enum | `backend/include/HipdnnBackendAttributeType.h` |
| Backend operation type enum | `backend/include/HipdnnOperationType.h` |
| Backend descriptor class | `backend/src/descriptors/MatmulOperationDescriptor.{hpp,cpp}` |
| Backend descriptor factory | `backend/src/descriptors/DescriptorFactory.cpp` |
| Backend node factory (lifting) | `backend/src/descriptors/NodeFactory.cpp` |
| Backend enum string utils | `backend/src/BackendEnumStringUtils.hpp` (header-only) |
| Backend type conversion | `backend/src/descriptors/DataTypeConversion.{hpp,cpp}` |
| Backend attribute set/get helpers | `backend/src/descriptors/DescriptorAttributeUtils.{hpp,cpp}` |
| Backend CMake | `backend/src/CMakeLists.txt` |
| Frontend Types (enum + converters) | `frontend/include/hipdnn_frontend/Types.hpp` |
| Frontend attributes class | `frontend/include/hipdnn_frontend/attributes/MatmulAttributes.hpp` |
| Frontend packer (lowering) | `frontend/include/hipdnn_frontend/detail/MatmulPacker.hpp` |
| Frontend unpacker (lifting) | `frontend/include/hipdnn_frontend/detail/MatmulUnpacker.hpp` |
| Frontend operation unpacker switch | `frontend/include/hipdnn_frontend/detail/OperationUnpacker.hpp` |
| Frontend node class | `frontend/include/hipdnn_frontend/node/MatmulNode.hpp` |
| Frontend Graph API | `frontend/include/hipdnn_frontend/Graph.hpp` (`Graph::matmul(...)`) |
| JSON utility | `flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/utilities/json/MatmulAttributes.hpp` |
| Python bindings (optional) | `python/src/graph_bindings.cpp`, `python/src/attributes_bindings.cpp` |
| Backend descriptor unit test | `backend/tests/descriptors/TestMatmulOperationDescriptor.cpp` |
| Backend fromNode test | `backend/tests/descriptors/TestMatmulOperationFromNode.cpp` |
| Backend graph descriptor test | `backend/tests/descriptors/TestGraphDescriptorMatmul.cpp` |
| Backend enum string test | `backend/tests/TestBackendEnumStringUtils.cpp` |
| Frontend attributes test | `frontend/tests/TestMatmulAttributes.cpp` |
| Frontend node test | `frontend/tests/TestMatmulNode.cpp` |
| Integration lowering test | `tests/frontend/IntegrationMatmulDescriptorLowering.cpp` |
| Integration lifting test | `tests/frontend/IntegrationMatmulDescriptorLifting.cpp` |
| Test SDK constants | `test_sdk/include/hipdnn_test_sdk/constants/MatmulConstants.hpp` |
| YAML config | `tools/DescriptorGenerator/configs/matmul.yaml` |

---

## Layer-by-Layer Reference

For when an agent step fails or you need to do a layer by hand. Subsections are ordered by compile dependency.

### Backend Layer

| File | Purpose |
|---|---|
| `backend/include/HipdnnBackendDescriptorType.h` | Descriptor type enum entry. Pick the next free value. |
| `backend/include/HipdnnBackendAttributeName.h` | Attribute name range. Pick the next free 100-block. |
| `backend/include/HipdnnBackendAttributeType.h` | Type tag entry — required for new mode enums. |
| `backend/include/HipdnnOperationType.h` | Operation type enum entry — required for lifting. |
| `backend/include/Hipdnn<Op>Mode.h` | New C-API mode enum header (only if the op introduces a new mode enum). |
| `backend/include/hipdnn_backend.h` | `#include` for the new mode enum header. |
| `backend/src/descriptors/<Op>OperationDescriptor.{hpp,cpp}` | The descriptor class itself. |
| `backend/src/descriptors/DescriptorFactory.cpp` | Forward factory (lowering). |
| `backend/src/descriptors/NodeFactory.cpp` | Lifting factory. |
| `backend/src/BackendEnumStringUtils.hpp` | Enum-to-string utilities (header-only — there is no `.cpp`). |
| `backend/src/descriptors/DataTypeConversion.{hpp,cpp}` | `toSdk` / `fromSdk` converters for new mode enums. |
| `backend/src/descriptors/DescriptorAttributeUtils.{hpp,cpp}` | `set` / `get` helpers for new mode enums. |
| `backend/src/CMakeLists.txt`, `backend/tests/CMakeLists.txt` | Build glue. |

### Frontend Layer (Compile-Order)

| File | Purpose |
|---|---|
| `frontend/include/hipdnn_frontend/Types.hpp` | Frontend enum class + `toBackend<Op>Mode` + `fromHipdnn<Op>Mode` converters. **Must exist before the packer/unpacker compile.** |
| `frontend/include/hipdnn_frontend/attributes/<Op>Attributes.hpp` | Attribute class. |
| `frontend/include/hipdnn_frontend/detail/<Op>Packer.hpp` | Lowering serializer (frontend → backend). |
| `frontend/include/hipdnn_frontend/detail/<Op>Unpacker.hpp` | Lifting deserializer (backend → frontend). |
| `frontend/include/hipdnn_frontend/detail/OperationUnpacker.hpp` | Switch entry calling the unpacker. |
| `frontend/include/hipdnn_frontend/node/<Op>Node.hpp` | Node class. |
| `frontend/include/hipdnn_frontend/Graph.hpp` | `Graph::<op>(...)` API method + include. |
| `frontend/CMakeLists.txt`, `frontend/tests/CMakeLists.txt` | Build glue. |

### JSON Utilities and Python Bindings

See [Review Implementation](#review-implementation), especially the optional sample and Python binding follow-ups.

---

## Testing Requirements

Each layer has a required test. Every checkbox in the [PR Checklist](#pr-checklist) maps to a row here.

| Layer | Required test | Source | Matmul reference |
|---|---|---|---|
| Backend descriptor unit | construction, finalize, get/setAttribute | generator | `backend/tests/descriptors/TestMatmulOperationDescriptor.cpp` |
| `fromNode` | FlatBuffer → backend descriptor lifting | generator | `backend/tests/descriptors/TestMatmulOperationFromNode.cpp` |
| Backend graph descriptor | full graph including the node, descriptor mode | generator | `backend/tests/descriptors/TestGraphDescriptorMatmul.cpp` |
| Frontend attribute | getters/setters/validation | generator | `frontend/tests/TestMatmulAttributes.cpp` |
| Frontend node | construction, `pre_validate_node` | generator | `frontend/tests/TestMatmulNode.cpp` |
| Frontend graph | end-to-end frontend graph | generator | `frontend/tests/TestGraph.cpp` (generator emits a per-op file for new ops) |
| Lowering integration | frontend node → backend descriptor round-trip | generator | `tests/frontend/IntegrationMatmulDescriptorLowering.cpp` |
| Lifting integration | backend descriptor → frontend node round-trip; tensor sharing; auto-assigned UIDs; per-scalar preservation | generator | `tests/frontend/IntegrationMatmulDescriptorLifting.cpp` |
| Constants header | shared per-op test constants | generator | `test_sdk/include/hipdnn_test_sdk/constants/MatmulConstants.hpp` |
| Backend enum string | one `EXPECT_STREQ` per new attribute and descriptor type | hand-add | `backend/tests/TestBackendEnumStringUtils.cpp` |
| Sample (optional but expected) | end-user-style usage | hand-author | `samples/<op>/` |

> [!IMPORTANT]
> Reuse the shared helpers from [`test_sdk/include/hipdnn_test_sdk/utilities/`](../test_sdk/include/hipdnn_test_sdk/utilities/): `IntegrationTestFixture`, `LoweringTestHelpers`, `LiftingTestHelpers`. Do not roll your own fixtures.

> [!IMPORTANT]
> Run the AddressSanitizer build before submitting:
> ```bash
> cmake -DBUILD_ADDRESS_SANITIZER=ON ..
> ninja check
> ```
> See [Building.md](./Building.md) for the full toolchain invocation.

> [!IMPORTANT]
> Test coverage must remain ≥80% per [`CONTRIBUTING.md`](../CONTRIBUTING.md).

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Linker error: missing `fromHipdnn<Op>Mode` | The mode enum was added to the backend but not to `Types.hpp`. Insert the converter from `mode_frontend_plumbing_<field>.txt`. |
| Backend reports "unknown attribute name" | `_EXT` suffix mismatch between `HipdnnBackendAttributeName.h` and the YAML. Verify both sides use the same suffix. |
| Compile error about a duplicate enum value | Enum value collision (someone landed on the same value while you were rebasing). Re-pick the next free value and re-run insertion. |
| Generator output not built | The CMake fragment was not inserted. Confirm the snippet from `fragments/*_cmake_*.txt` was applied to the right `CMakeLists.txt`. |
| Lifting test fails on tensor sharing | The unpacker is not using `getOrCreate` semantics for shared tensors. Check the unpacker against `MatmulUnpacker.hpp`. |
| Unpacker references a converter that exists in backend headers but not in frontend `Types.hpp` | The mode-frontend-plumbing fragment was not fully applied. Insert the inverse converter from the fragment. |
| Segfaults during graph execution plan build | Plugins require PIC/PIE. Set `set(CMAKE_POSITION_INDEPENDENT_CODE ON)` in your consumer CMake. See [HowTo.md troubleshooting](./HowTo.md#troubleshooting). |

---

## Reference Links

- [Descriptor Code Generator README](../tools/DescriptorGenerator/README.md) — needed only if you want to inspect the generator's YAML format or run it directly
- [`CONTRIBUTING.md`](../CONTRIBUTING.md)
- [Design Guide](./Design.md)
- [cuDNN Porting Guide](./PortingGuide.md)
- [Operation Support Matrix](./OperationSupport.md)
- [Building Guide](./Building.md)
- [Testing Guide](./Testing.md)
- [HowTo Guide](./HowTo.md)
