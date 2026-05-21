---
name: hipdnn-codegen
description: Generate hipDNN operation boilerplate from a YAML config. Use when the user wants to add a new operation type to hipDNN, or generate descriptor/packer/unpacker code.
argument-hint: "<schema-path-or-op-name> [mode: backend|frontend|full]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, AskUserQuestion
---

# hipDNN Code Generator Skill

Generate all boilerplate code needed to add a new operation to hipDNN from a YAML config.

## Philosophy

**You MUST run the generator.** The code generator exists to produce correct, pattern-matched boilerplate. Do not skip it and write code by hand — even if you think you understand the patterns well enough. The generator's output is the starting point for every file and fragment. If Bash access is unavailable or denied, stop and ask the user to grant it rather than proceeding without the generator.

After running the generator, the agent owns the integration. Generated output is scaffolding, not a finished product. Before placing any generated file or inserting any fragment, check if the target already exists in hipDNN. If it does, compare the generated version with the existing code. Use whichever is more correct, or merge them if each covers different parts. If a fragment's insertion point has changed, adapt the fragment to the current code structure.

Generated code is not always perfect. Enum value numbering may differ between the backend C-API, SDK, and frontend — use `frontend_value` and `sdk_name` overrides in `enum_def` to handle mismatches. Test patterns may need adjustment for unusual field types. Fragment insertion points are guidelines; always read the target file. But these are adjustments to generator output, not reasons to bypass the generator entirely.

The goal is a clean, building, tested integration. If generated code needs tweaks to compile, make them. If a fragment conflicts with existing code, resolve it. Judgment and adaptation are applied to generated output, not as a substitute for it.

## Arguments

- `$ARGUMENTS` can contain:
  - **Schema or operation**: Path to `.fbs` schema file, path to existing YAML config, or operation name (e.g., `convolution_fwd`)
  - **Mode** (one of):
    - `backend` (default) — Descriptor, Packer, Unpacker, backend tests, lifting fragments, and lifting integration test (the previous `lift-only` mode is folded in)
    - `frontend` — Node, Attributes, Graph method + frontend tests
    - `full` — Everything (backend + frontend)

## Directory Locations

Determine the hipDNN project root by finding the nearest parent directory containing `tools/DescriptorGenerator/`. Set paths relative to that:

```
HIPDNN_SRC=<path to projects/hipdnn>
CODEGEN=$HIPDNN_SRC/tools/DescriptorGenerator
VENV=$CODEGEN/.venv
```

Hint: if the current working directory is inside a rocm-libraries worktree, the hipDNN root is at `<worktree>/projects/hipdnn/`. If invoked from a standalone hipDNN checkout, it is the repo root.

## Execution Steps

### 1. Parse Arguments and Set Up

Parse `$ARGUMENTS` for:
- Schema path, config path, or operation name
- Mode (default: `backend`)

Locate the hipDNN project root. Verify the codegen venv exists:
```bash
cd $CODEGEN
if [ ! -d .venv ]; then
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi
```

### 2. Determine Input Type

- If argument is a `.fbs` file path: proceed to Step 3 (create YAML config from schema)
- If argument is a `.yaml` file path: skip to Step 5 (run generator)
- If argument is an operation name and `configs/<name>.yaml` exists: skip to Step 5
- If none of the above: ask the user to provide a schema or config path

### 3. Create YAML Config from FBS Schema

Read the FBS schema file. Map fields to YAML config following these rules:

| FBS Field Pattern | YAML Section | YAML `type` |
|---|---|---|
| `*_tensor_uid: long` | `tensor_fields` | (tensors are UIDs) |
| `field: [long]` | `data_fields` | `vector_int64` |
| `field: SomeEnum` | `data_fields` | `mode` |
| `field: float` | `data_fields` | `scalar_float` |
| `field: long` (non-UID) | `data_fields` | `scalar_int64` |
| `field: bool` | `data_fields` | `bool` |
| `field: [long]` (array of UIDs) | `tensor_array_fields` | (tensor arrays) |

Derive all config fields from the schema and existing codebase — do NOT ask the user for these. Use existing configs and backend code to determine:

- **Operation name**: Derive from the FBS table name (e.g., `ConvolutionFwdAttributes` → `ConvolutionFwd`)
- **Descriptor type enum**: Search `$HIPDNN_SRC/backend/include/HipdnnBackendDescriptorType.h` for existing enum entries
- **Operation attribute prefix**: Search `$HIPDNN_SRC/backend/include/HipdnnBackendAttributeName.h` for existing `HIPDNN_ATTR_OPERATION_*` entries
- **Shared attributes**: Compare attribute names against existing operations in the codebase
- **Compute data type**: Check if the FBS schema has compute precision fields; check existing operations for patterns
- **Test tensor UIDs**: Read existing configs in `$CODEGEN/configs/` to find used UID ranges, pick the next available
- **Frontend fields** (graph method name, NodeType, union type): Derive from existing frontend code or operation name conventions

If a field truly cannot be determined, use sensible defaults derived from the operation name. Only ask the user as a last resort for genuinely ambiguous decisions.

Write the config to `$CODEGEN/configs/<operation>.yaml`.

Use `$CODEGEN/configs/convolution_fwd.yaml` as the reference template for all config fields.

#### 3a. Populate `enum_def` for New Enum Types

For each `mode` data field, check if the backend enum infrastructure already exists:
```bash
grep -l "HIPDNN_TYPE_" $HIPDNN_SRC/backend/include/HipdnnBackendAttributeType.h | head -1
grep "<EnumShortType>" $HIPDNN_SRC/backend/include/Hipdnn*.h
```

If the enum type is **new** (no existing backend header), populate the `enum_def` block by reading the FBS enum values:

```yaml
    enum_def:
      backend_header: "Hipdnn<Foo>Mode.h"     # Output C-API header filename
      backend_prefix: "HIPDNN_<FOO>_"          # Prefix for C-API enum constants
      values:
        - { name: "VALUE_A", value: 0 }        # name = backend C-API suffix
        - { name: "VALUE_B", value: 1 }
        # Optional fields per value:
        #   sentinel: true       — marks UNSET/NOT_SET values excluded from backend C-API enum
        #   sdk_name: "ALT"      — override SDK enum name if different (e.g., "MAX_OP" for "MAX")
        #   frontend_name: "X"   — override frontend enum name (e.g., "TOP_LEFT" for "TOP_LEFT_EXT")
        #   description: "text"  — Doxygen comment for the enum constant (rendered as ///< text)
        #   frontend_value: N    — frontend enum numeric value when different from backend (omit if same)
```

**Rules for populating values:**
- Read the FBS enum definition for the list of values
- `name` is the backend C-API suffix: `backend_prefix + name` = full C-API constant (e.g., `HIPDNN_POINTWISE_` + `ABS` = `HIPDNN_POINTWISE_ABS`)
- `value` is the backend C-API numeric value (may differ from FBS value)
- Mark FBS sentinel values (UNSET, NOT_SET) with `sentinel: true` — these are excluded from the backend C-API enum but included in the frontend enum as `NOT_SET = 0`
- Use `sdk_name` when the SDK enum name differs from the backend name (e.g., FBS `MAX_OP` but backend `MAX`)
- Use `frontend_name` when the frontend enum member name differs from the backend suffix (e.g., backend `TOP_LEFT_EXT` but frontend `TOP_LEFT`)
- Add `description` to each value for Doxygen comments in generated code. Read the existing frontend `Types.hpp` enum class comments for the correct descriptions.
- `frontend_value` is optional (null/omitted = use backend value). Only set when the frontend enum has different numeric values than the backend C-API (e.g., ConvolutionMode has `CROSS_CORRELATION=1, CONVOLUTION=2` in the frontend but `CONVOLUTION=0, CROSS_CORRELATION=1` in the backend). Check the existing frontend enum class in `Types.hpp` to determine if overrides are needed.
- Set `shared: false` on the data field so the generator produces the mode enum plumbing

**`shared` and `enum_def` coexistence:** The `enum_def` block can be present on both `shared: true` and `shared: false` fields:
- When `shared: false`: the generator produces all mode enum plumbing from `enum_def` (C-API header, backend/frontend plumbing fragments).
- When `shared: true`: the `enum_def` serves as documentation and reference only — the generator skips plumbing generation because another operation already defined this enum. The benefit is that `enum_def` documents the enum values inline in the config and enables the agent to verify the enum infrastructure without searching the codebase.

If the enum type **already exists** in the backend, set `shared: true` on the data field. You may optionally include `enum_def` for documentation purposes — the generator will skip plumbing generation regardless.

### 4. Handle Mode Enum Fields

If the YAML config has `mode` data fields, determine whether the enum is new or existing:

**4a. If `enum_def` is present on the data field (new enum):**

The generator automatically produces all mode enum plumbing:
- `backend/include/<header>.h` — Complete C-API enum header (new file)
- `fragments/mode_backend_plumbing_<field>.txt` — Combined backend fragment with sections for:
  - `HipdnnBackendAttributeType.h` (type tag entry)
  - `DataTypeConversion.hpp/.cpp` (toSdk/fromSdk converters)
  - `DescriptorAttributeUtils.hpp/.cpp` (set/get helpers)
  - `BackendEnumStringUtils.hpp` (string case)
  - `hipdnn_backend.h` (#include directive)
- `fragments/mode_frontend_plumbing_<field>.txt` — Combined frontend fragment with sections for:
  - `Types.hpp` (frontend enum class, toBackend, fromHipdnn converters)

Insert each section from the fragment files into the corresponding target file, just like other fragments. The type tag `PLACEHOLDER_VALUE` must be replaced with the next available value in `HipdnnBackendAttributeType.h`.

**4b. If `enum_def` is NOT present (existing enum):**

Check whether the required infrastructure exists manually:
```bash
grep -r "HIPDNN_TYPE_" $HIPDNN_SRC/backend/include/HipdnnBackendAttributeType.h
```

If missing, create the plumbing by hand following existing patterns (ConvMode, PointwiseMode).

**4c. Place inverse converter in Types.hpp (REQUIRED for backend and full modes):**

The unpacker calls the inverse converter (e.g., `fromHipdnnPointwiseMode`). It MUST exist in `Types.hpp` or the code will not compile. Check if it already exists:
```bash
grep "fromHipdnn" $HIPDNN_SRC/frontend/include/hipdnn_frontend/Types.hpp
```

If it does NOT exist:
- If `enum_def` is present: the converter is in the generated `fragments/mode_frontend_plumbing_<field>.txt` under the "fromHipdnn converter" section. Insert it into `Types.hpp`.
- If `enum_def` is absent: generate the converter by reading the existing forward converter and inverting the mapping.

**Also check** `toBackend<Foo>Mode` — if it's missing too, insert it from the same fragment. The unpacker and tests may need it.

### 5. Run the Generator (MANDATORY)

**This step is non-negotiable.** You must run the generator to produce the output files and fragments. Do not write boilerplate code by hand — even if you have read the templates and understand the patterns. The generator ensures consistency, handles edge cases in the templates, and produces tested output.

If Bash access is denied, ask the user to approve it. Do not proceed to Step 6 without generator output.

```bash
cd $CODEGEN
OUTPUT_DIR=/tmp/hipdnn-codegen-output
rm -rf $OUTPUT_DIR
$VENV/bin/python generate.py \
    --config configs/<operation>.yaml \
    --output-dir $OUTPUT_DIR \
    --mode $MODE
```

If the generator fails, show the error and stop.

List all generated files:
```bash
find $OUTPUT_DIR -type f | sort
```

Read each generated file before placing it — understand what the generator produced so you can compare it with existing code in the next step.

### 6. Place Generated Files

Before copying each generated file, check if the target already exists. If it does, read the existing file and compare. For new operations, the generated files are typically placed directly. For operations that already have partial implementations, merge generated code with existing code -- keeping what works and adding what's missing.

Copy each generated file to its target location in the project tree.

**Backend files** (for `backend` or `full` mode):

| Generated Path | Target |
|----------------|--------|
| `backend/src/descriptors/<Op>OperationDescriptor.hpp` | `$HIPDNN_SRC/backend/src/descriptors/` |
| `backend/src/descriptors/<Op>OperationDescriptor.cpp` | `$HIPDNN_SRC/backend/src/descriptors/` |
| `frontend/include/hipdnn_frontend/detail/<Op>Packer.hpp` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/detail/` |
| `frontend/include/hipdnn_frontend/detail/<Op>Unpacker.hpp` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/detail/` |
| `backend/tests/descriptors/Test<Op>OperationDescriptor.cpp` | `$HIPDNN_SRC/backend/tests/descriptors/` |
| `backend/tests/descriptors/TestGraphDescriptor<Op>.cpp` | `$HIPDNN_SRC/backend/tests/descriptors/` |
| `backend/tests/descriptors/Test<Op>OperationFromNode.cpp` | `$HIPDNN_SRC/backend/tests/descriptors/` |
| `tests/frontend/Integration<Op>DescriptorLowering.cpp` | `$HIPDNN_SRC/tests/frontend/` |
| `test_sdk/include/hipdnn_test_sdk/constants/<Op>Constants.hpp` | `$HIPDNN_SRC/test_sdk/include/hipdnn_test_sdk/constants/` |

**Note**: The constants file is only generated when `constants_include` is NOT set in the YAML config. When `constants_include` IS set, the generated test files reference the existing header instead.

**Frontend files** (for `frontend` or `full` mode):

| Generated Path | Target |
|----------------|--------|
| `frontend/include/hipdnn_frontend/attributes/<Op>Attributes.hpp` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/attributes/` |
| `frontend/include/hipdnn_frontend/node/<Op>Node.hpp` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/node/` |
| `frontend/tests/Test<Op>Attributes.cpp` | `$HIPDNN_SRC/frontend/tests/` |
| `frontend/tests/Test<Op>Node.cpp` | `$HIPDNN_SRC/frontend/tests/` |
| `frontend/tests/TestGraph<Op>.cpp` | `$HIPDNN_SRC/frontend/tests/` |

### 7. Insert Fragment Snippets

Read each fragment file from the output and insert it into the correct shared file.

**CRITICAL**: Read each target file FIRST to find the correct insertion point. Never blindly append.

**Backend fragments** (for `backend` or `full` mode):

| Fragment | Target File | Insertion Point |
|----------|-------------|----------------|
| `fragments/attribute_enum_block.txt` | `$HIPDNN_SRC/backend/include/HipdnnBackendAttributeName.h` | After the last operation attribute enum range. Assign next available range (read existing ranges to find the next one). Replace `PLACEHOLDER_VALUE` with actual values. |
| `fragments/descriptor_type_enum.txt` | `$HIPDNN_SRC/backend/include/HipdnnBackendDescriptorType.h` | Before the closing brace of `hipdnnBackendDescriptorType_t` enum. |
| `fragments/string_utils_block.txt` | `$HIPDNN_SRC/backend/src/BackendEnumStringUtils.hpp` | In the appropriate switch statements (descriptor type name + attribute name). |
| `fragments/factory_case.txt` | `$HIPDNN_SRC/backend/src/descriptors/DescriptorFactory.cpp` | Add `#include` at top, add `case` in `create()` switch. |
| `fragments/cmake_entries.txt` | Multiple CMakeLists.txt files | Add source to `backend/src/CMakeLists.txt`, tests to `backend/tests/CMakeLists.txt` and `tests/frontend/CMakeLists.txt`. |

**Lifting fragments** (for `backend` or `full` mode):

| Fragment | Target File | Insertion Point |
|----------|-------------|----------------|
| `fragments/node_factory_case.txt` | `$HIPDNN_SRC/backend/src/descriptors/NodeFactory.cpp` | In the `createFromNode()` switch. |
| `fragments/operation_unpacker_case.txt` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/detail/OperationUnpacker.hpp` | In the `createNodeForType()` switch. |
| `fragments/operation_type_enum.txt` | `$HIPDNN_SRC/backend/include/HipdnnOperationType.h` | Before the closing brace of the enum. |
| `fragments/node_unpack_override.txt` | Frontend node header | Add method to the node class. |

**Mode enum fragments** (for `backend` or `full` mode when `enum_def` is present on a data field):

| Fragment | Target Files | Insertion |
|----------|-------------|-----------|
| `fragments/mode_backend_plumbing_<field>.txt` | Multiple backend files | Read the fragment — it has clearly labeled sections for each target file. Insert each section into the corresponding file. Replace `PLACEHOLDER_VALUE` in the type tag section. |
| `fragments/mode_frontend_plumbing_<field>.txt` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/Types.hpp` | Read the fragment — it has sections for the enum class, toBackend, and fromHipdnn. Insert each near existing similar code. |

The generated `backend/include/<header>.h` file is a complete file — copy it directly to `$HIPDNN_SRC/backend/include/`.

**Frontend fragments** (for `frontend` or `full` mode):

| Fragment | Target File | Insertion Point |
|----------|-------------|----------------|
| `fragments/graph_method.txt` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/Graph.hpp` | After the last operation method (find the pattern `_sub_nodes.emplace_back`). |
| `fragments/graph_includes.txt` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/Graph.hpp` | In the includes section at the top, grouped with other operation includes. |
| `fragments/frontend_cmake_entries.txt` | `$HIPDNN_SRC/frontend/tests/CMakeLists.txt` | Add test files to the test target source list. |

### 8. Add Enum Test Coverage

After inserting enum fragments, add test entries to `$HIPDNN_SRC/backend/tests/TestBackendEnumStringUtils.cpp`:

- One `EXPECT_STREQ` for the descriptor type name
- One `EXPECT_STREQ` per new attribute enum value
- Place entries near existing similar tests

### 9. Apply Descriptor Lifting Additions (backend and full modes)

If `fragments/descriptor_lifting_additions.txt` exists:
- Read it for the exact changes needed to the existing descriptor `.hpp` and `.cpp`
- Apply each change (add `#include <unordered_map>`, `fromNode()` declaration, `_name` member to `.hpp`; add operation name/type handling and `fromNode()` impl to `.cpp`)

### 10. Wire Frontend Node (if node class exists or was generated)

For `backend` mode when a frontend node already exists, or for `full` mode:
- Add `#include` for packer header to the node class
- Add `create_operation()` override calling the packer
- Add `#include` for unpacker header
- Add `unpack_from_descriptor()` override calling the unpacker

### 11. Ask About Operation-Specific Logic

For `frontend` or `full` mode, these are the ONLY questions to ask the user:

**infer_properties_node()**:
- "How should output dimensions be inferred?"
  - Option 1: Output matches input dimensions (e.g., batchnorm, pointwise unary)
  - Option 2: Output is broadcast of inputs (e.g., pointwise binary)
  - Option 3: Custom formula (ask user to describe, generate TODO stub with the description)
  - Option 4: Leave as stub (default)

**pre_validate_node()**:
- The generated code already checks for null required tensors and non-empty dimensions
- "Are there additional validation rules?"
  - e.g., "input channels must match weight channels"
  - e.g., "stride and dilation must be > 0"
  - Default: leave with just the standard null/dim checks

Do NOT ask the user about any other fields or decisions — derive everything else from the schema, existing code, and conventions.

### 12. Review Generated Integration Tests

The generator now produces complete integration tests for both lowering and lifting:

**Lowering** (`Integration<Op>DescriptorLowering.cpp`):
- `<Op>LoweringRoundTrip` — Full round-trip with explicit UIDs, per-tensor dims/strides from the constants header, mode and vector field verification
- Per-optional-scalar preservation tests

**Lifting** (`Integration<Op>DescriptorLifting.cpp`):
- `Basic<Op>RoundTrip` — Full lifting round-trip with field-by-field validation
- `<Op>TensorSharingPreserved` — Pointer equality verification
- `<Op>LiftWithoutFinalization` — Backend binary serialization path
- `AutoAssignedUidsPreservedInLiftingRoundTrip` — Auto-assigned UID distinctness and round-trip
- Per-optional-scalar preservation tests

All tests use `K_TENSOR_*` constants from the shared constants header — no inline literals.

Review the generated tests and consider adding operation-specific tests for multi-input variants (e.g., pointwise ternary) or multi-operation graphs (e.g., conv+bias+relu chains) as needed.

If the frontend node class or graph method does not exist yet (e.g., backend-only mode), skip this step but note it as pending.

### 13. Build and Test (MANDATORY)

**You MUST build and test before reporting success.** A build verifies that all fragments were inserted, all converters exist, all includes resolve, and all types match. Skipping the build means you cannot know if the integration is correct.

Before building, do a quick self-check:
- Every function called in the unpacker exists (grep for it in the target file)
- Every `#include` in new files points to a real header
- Every switch case references a valid enum constant
- CMake lists include all new source/test files

Build using the ROCm Clang toolchain:
```bash
cd $HIPDNN_SRC
mkdir -p build && cd build
cmake .. -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=<repo-root>/cmake/toolchains/rocm-clang.cmake
ninja 2>&1 | tail -100
```

If the build fails, **read the errors and fix them**. Common issues:
- Missing converter function in `Types.hpp` → insert from `mode_frontend_plumbing_<field>.txt`
- Missing `#include` → add it
- Wrong attribute name (missing `_EXT` suffix) → check `HipdnnBackendAttributeName.h`
- Type mismatch → check the generated code against existing patterns

After the build succeeds, run unit tests:
```bash
ninja unit-check 2>&1 | tail -50
```

If tests fail, diagnose and fix. Do not report success with failing tests.

### 14. Report Results

Summarize what was generated and placed:
- List all files created/modified
- Note any stubs that still need implementation (custom infer_properties, custom validation)
- Note any fragment insertions that need manual verification (enum value ranges, CMake)
- Confirm build passed and tests passed
- If anything failed and could not be fixed, explain what and why

## Error Handling

- If the FBS schema has field types not supported by the generator, report them and ask the user how to handle
- If enum value ranges conflict with existing values, report and ask the user for the correct range
- If the generator script fails, show the full error output
- If build fails after placement, show the last 100 lines and help diagnose
- If a target file for fragment insertion cannot be found, report and skip that insertion

## Notes

- The generated integration tests (lowering and lifting) are complete with round-trip, tensor sharing, auto-UIDs, and per-scalar tests. Review and add operation-specific tests as needed.
- When `descriptor_lifting_additions.txt` is emitted (any backend run), apply its changes to the existing descriptor `.hpp`/`.cpp` in-place.
- `mode` type is REQUIRED for all enum fields in new operations. Never use the legacy `enum` type.
- Read `$CODEGEN/CLAUDE.md` for the full detailed post-generation workflow if you need additional context on any step.
- Always use `convolution_fwd.yaml` as the reference config when creating new configs.
- Do NOT ask the user questions about config fields, enum names, UIDs, or other derivable information. The only user-facing questions should be about `infer_properties` strategy and custom validation rules.
- Generated mode enum plumbing (`enum_def`) supports `frontend_value` for cases where frontend and backend enum values differ. Always verify against existing `Types.hpp` when populating `enum_def`.
- The generator auto-produces a shared constants header (`<Op>Constants.hpp`) from YAML `test_data` when `constants_include` is not set. All test templates reference it. When `constants_include` IS set, the existing header is used and no constants file is generated.
- The generated code is a starting point. Review each file and fragment against the current state of hipDNN before placing. Prefer existing code when it's correct; merge when each covers different parts; use generated code when the target doesn't exist yet.
