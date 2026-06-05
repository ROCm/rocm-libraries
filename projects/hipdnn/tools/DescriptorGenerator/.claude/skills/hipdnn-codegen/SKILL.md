---
name: hipdnn-codegen
description: Generate hipDNN operation boilerplate from a YAML config. Use when the user wants to add a new operation type to hipDNN, or generate descriptor/packer/unpacker code.
argument-hint: "<schema-path-or-op-name> [mode: backend|frontend|full]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, WebFetch, AskUserQuestion
---

# hipDNN Code Generator Skill

Generate all boilerplate code needed to add a new operation to hipDNN from a YAML config.

This skill is **near-autonomous**. The agent first tries to resolve everything itself by deriving from the schema, the existing codebase, and `cudnn-frontend` on GitHub. It **may** prompt the user as a fallback on three classes of decision where derivation has failed:
1. **cuDNN naming uncertainty** — when the cuDNN equivalent descriptor / attribute / frontend-node name cannot be confidently identified after consulting `cudnn-frontend` (drives the `_EXT` decision; see Step 3a-i).
2. **`infer_properties_node()` strategy** — output-shape inference rule, when the op's shape behavior isn't obvious from analogous ops.
3. **`pre_validate_node()` rules** — additional validation beyond default null/dim checks, when domain-specific constraints aren't derivable.

In a clean, well-known op (cuDNN-aligned with a clear analog), the agent may not prompt at all on the three classes above.

There is one **mandatory** prompt whenever the agent had to construct the FBS schema (from a user description, or from `cudnn-frontend`): **schema confirmation** (Step 2a-ii). Schema mistakes propagate through every layer, so the agent must show the schema and get explicit user sign-off before continuing. This prompt is skipped only when the user supplied a complete FBS file.

Everything else is mechanical and is derived from the FBS schema, existing codebase state, and the conventions in this skill.

For human-facing context, including the cuDNN parity rules, layer-by-layer reference, testing matrix, and PR checklist, see [`projects/hipdnn/docs/AddingNewOperations.md`](../../../../../docs/AddingNewOperations.md). For a complete worked example, see [`configs/convolution_fwd.yaml`](../../../configs/convolution_fwd.yaml).

## Philosophy

**You MUST run the generator.** The code generator exists to produce correct, pattern-matched boilerplate. Do not skip it and write code by hand — even if you think you understand the patterns well enough. The generator's output is the starting point for every file and fragment. If Bash access is unavailable or denied, stop and ask the user to grant it rather than proceeding without the generator.

After running the generator, the agent owns the integration. Generated output is scaffolding, not a finished product. Before placing any generated file or inserting any fragment, check if the target already exists in hipDNN. If it does, compare the generated version with the existing code. Use whichever is more correct, or merge them if each covers different parts. If a fragment's insertion point has changed, adapt the fragment to the current code structure.

Generated code is not always perfect. Enum value numbering may differ between the backend C-API, SDK, and frontend — use `frontend_value` and `sdk_name` overrides in `enum_def` to handle mismatches. Test patterns may need adjustment for unusual field types. Fragment insertion points are guidelines; always read the target file. But these are adjustments to generator output, not reasons to bypass the generator entirely.

The goal is a clean, building, tested integration. If generated code needs tweaks to compile, make them. If a fragment conflicts with existing code, resolve it. Judgment and adaptation are applied to generated output, not as a substitute for it.

## When to Use Which Mode

| Scenario | Mode |
|---|---|
| Brand new operation (nothing exists yet) | `full` |
| Adding backend only (frontend exists or will land later) | `backend` |
| Adding frontend only (backend descriptor already exists) | `frontend` |

## Arguments

- `$ARGUMENTS` can contain:
  - **Schema or operation**: Path to `.fbs` schema file, path to existing YAML config, or operation name (e.g., `convolution_fwd`)
  - **Mode** (one of):
    - `backend` (default) - Descriptor, Packer, Unpacker + backend tests
    - `frontend` - Node, Attributes, Graph method + frontend tests
    - `full` - Everything (backend + frontend)

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
- If argument is an operation name and **no FBS schema exists** for it: proceed to Step 2a (author the FBS schema)
- If the argument is unrecognized or ambiguous: ask the user to clarify

### 2a. Author the FBS Schema (if it does not exist)

Schemas live ONLY in `$HIPDNN_SRC/flatbuffers_sdk/schemas/` (they were removed from `data_sdk/`). Before deriving the YAML, you may need to write the FBS schema yourself.

#### 2a-i. Source the schema content

Pick the path that matches what the user provided:

- **User provided an FBS file** → use it directly, skip to 2a-ii.
- **User provided a description** (text listing inputs, outputs, attributes, modes) → translate it directly into the table, skip to 2a-ii.
- **User gave only an op name** → derive the schema from `cudnn-frontend` on GitHub:
  1. Run the directory listing + node-file fetch from Step 3a-i to locate the matching cuDNN node header (e.g., `include/cudnn_frontend/node/<op>.h`).
  2. From that header, find the **attributes class** it references (e.g., `Conv_fprop_attributes`). Fetch its source — usually under `include/cudnn_frontend/graph_properties.h` or alongside the node header.
  3. Walk the attributes class:
     - Each tensor input/output method (typically `set_<name>(...)`) becomes a `<name>_tensor_uid: long` field. Note which are required vs optional.
     - Each scalar/vector attribute (typically `set_<attr>(...)`) becomes a typed field with the matching cuDNN name.
     - Mode-style enums become a `<field>: <Enum>` field; capture the enum members from the cuDNN definition.
  4. Use the same naming as cuDNN (lowercase snake_case for fields, matching cuDNN attribute identifiers). Apply `_EXT` per the rule in Step 3a only to fields with no cuDNN equivalent.
  5. Write the draft schema to `$HIPDNN_SRC/flatbuffers_sdk/schemas/<op_name>_attributes.fbs`.

#### 2a-ii. Confirm the schema with the user

When this prompt is required:

| Schema source | Confirmation required? |
|---|---|
| User supplied a complete FBS file | No (they already authored it) |
| User supplied a description and the agent translated it | **Yes** |
| Agent derived from `cudnn-frontend` | **Yes** |

When required, show the schema and the parsed structure, and get explicit sign-off via `AskUserQuestion`:

> "Here is the FBS schema I will use for `<op_name>` (sourced from `<description / cudnn-frontend URL>`):
>
> ```flatbuffers
> <full schema contents>
> ```
>
> Tensor inputs: `<list>`. Tensor outputs: `<list>`. Optional fields: `<list>`. Mode enums: `<list>`.
>
> Confirm before I proceed: is this schema correct? (Yes / Edit-and-resend / No-rewrite)"

Do NOT proceed to 2a-iii until the user confirms. If the user requests edits, apply them and ask again. Schema mistakes propagate to every layer (FBS table, generated headers, descriptor, packer, unpacker, attributes class, node, tests) — fixing them late is expensive.

#### 2a-iii. Wire the schema into the build

Once confirmed:

1. **Update `graph.fbs`:**
   - Add `include "<op_name>_attributes.fbs";`
   - Add `<Op>Attributes` to the `NodeAttributes` union
2. **Update the `SCHEMAS` list** in `$HIPDNN_SRC/flatbuffers_sdk/CMakeLists.txt` so the new schema is compiled into the SDK.
3. **Rebuild the FlatBuffers SDK target** so the generated headers are present before the code generator runs:
   ```bash
   cd $HIPDNN_SRC/build  # or your active build dir
   ninja hipdnn_flatbuffers_sdk
   ```
   (If no build dir exists yet, see [`docs/Building.md`](../../../../../docs/Building.md) for first-time setup.)

After 2a-iii, proceed to Step 3.

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

#### 3a. cuDNN Naming Parity Check (`_EXT` Decision Rule)

Parity with cuDNN is **nominal** (names match) but **not numeric** (values do not align — pick the next free value as Step 3 already does).

For each name being constructed in the YAML — `enum_name`, `attr_suffix` (per tensor/data field), `compute_data_type_attr`, `operation_type_enum` — apply the rule:

| Name has a real cuDNN equivalent? | Suffix |
|---|---|
| Yes, exact match (e.g., `CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR`) | **No** `_EXT` |
| No equivalent, or hipDNN aggregates/splits differently from cuDNN | `_EXT` |

Sources of truth, in order of preference:

1. **`cudnn-frontend` source on GitHub** — fetch directly to identify class names, attribute names, and the constant identifiers it references. See sub-step 3a-i below. This is the only external cuDNN source the agent should consult.
2. [`docs/PortingGuide.md`](../../../../../docs/PortingGuide.md) — cuDNN ↔ hipDNN API mapping (may be incomplete or out of date).

Reference examples:
- [`configs/reduction.yaml`](../../../configs/reduction.yaml) — every name matches cuDNN, **no** `_EXT`.
- [`configs/batchnorm.yaml`](../../../configs/batchnorm.yaml) — cuDNN uses generic `NORM_FORWARD`/`NORM_BACKWARD` descriptors; hipDNN uses explicit `BATCHNORM`-named descriptors — names differ, all `_EXT`.
- [`configs/convolution_fwd.yaml`](../../../configs/convolution_fwd.yaml) — descriptor and attributes match cuDNN (no `_EXT`); `operation_type_enum` is hipDNN-only (`_EXT`).

**`MATH_PREC` vs `COMP_TYPE` for `compute_data_type_attr`** — this is not a free choice; it follows cuDNN. Check the cudnn-frontend node header for the op (or the closest op in the same family) before setting this name. Examples: `CUDNN_ATTR_POINTWISE_MATH_PREC` → `HIPDNN_ATTR_POINTWISE_MATH_PREC` (no `_EXT`); `CUDNN_ATTR_REDUCTION_COMP_TYPE` → `HIPDNN_ATTR_REDUCTION_COMP_TYPE` (no `_EXT`). For ops with no cuDNN equivalent, use `COMP_TYPE_EXT`.

##### 3a-i. Web-check `cudnn-frontend` before prompting

`cudnn-frontend` is open-source at <https://github.com/NVIDIA/cudnn-frontend>. Use it to discover the cuDNN node class name, its attribute getters, and the `CUDNN_ATTR_*` / `CUDNN_BACKEND_OPERATION_*` constants the wrapper references. **Node file naming is hard to guess** (e.g., `conv_fprop.h` not `convolution_forward.h`), so always list the directory first, then fetch the matching file.

1. **List the node directory** to discover the candidate filename:
   ```bash
   gh api repos/NVIDIA/cudnn-frontend/contents/include/cudnn_frontend/node \
       --jq '.[].name' 2>/dev/null
   ```
   Or via WebFetch as a fallback: `https://github.com/NVIDIA/cudnn-frontend/tree/main/include/cudnn_frontend/node`.

2. **Pick the file that matches your op semantically.** Examples of name divergence to watch for: `conv_fprop.h` for forward conv, `dbn_weight.h` for batchnorm-backward weight grad, `sdpa.h` for scaled-dot-product-attention. If multiple candidates look plausible, fetch each header in parallel.

3. **Fetch the matched header** to read the cuDNN class, attribute getters, and the backend constants it wraps:
   ```bash
   gh api repos/NVIDIA/cudnn-frontend/contents/include/cudnn_frontend/node/<file>.h \
       --jq '.content' | base64 -d
   ```
   Or via WebFetch on the raw URL: `https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/main/include/cudnn_frontend/node/<file>.h`.

4. **Extract the parity facts** you need:
   - The cuDNN node class name (e.g., `Conv_fprop_attributes`) — use to confirm the YAML's `compatibility_typedef`.
   - Which `CUDNN_BACKEND_OPERATION_<X>_DESCRIPTOR` it constructs — use to set `enum_name` and decide `_EXT`.
   - Each `CUDNN_ATTR_OPERATION_<X>_<FIELD>` it sets via `setAttribute` — use per-field `attr_suffix` and `_EXT` decisions.

5. **If the directory listing has no matching file**, treat the op as hipDNN-specific: apply `_EXT` to all four name fields. No need to prompt.

6. **If web access fails or the file is ambiguous**, fall back to prompting the user via `AskUserQuestion` with one or more of:
   - *"I don't know the cuDNN equivalent for this operation. What is the full cuDNN backend descriptor constant name (e.g., `CUDNN_BACKEND_OPERATION_<X>_DESCRIPTOR`), or 'none' if hipDNN-specific?"*
   - *"I cannot confidently match attribute `<name>` to a cuDNN constant. What is the full `CUDNN_ATTR_<X>` name, or 'none' if hipDNN-specific?"*
   - *"I'm not sure what to name the frontend node class / Graph API method (e.g., `Graph::reduction` vs `Graph::reduce`). Confirm the preferred name."*

7. **Surface the source in the Step 14 summary.** For every name decision driven by the web-check, record the URL fetched and the line/snippet referenced (e.g., "`enum_name=...REDUCTION_DESCRIPTOR` per cudnn-frontend `node/reduction.h` L42"). This lets the human spot a misread.

On "none" answers (or no matching cudnn-frontend file), apply `_EXT` to the corresponding YAML name. On confirmed cuDNN equivalents, omit `_EXT`.

#### 3b. Populate `enum_def` for New Enum Types

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

**4c. Place inverse converter in Types.hpp (REQUIRED for `backend` and `full` modes):**

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
| `fragments/descriptor_lifting_additions.txt` | Existing `<Op>OperationDescriptor.{hpp,cpp}` | Apply per Step 9 — adds `<unordered_map>` include, `fromNode()` declaration, `_name` member, operation name/type handling, and `fromNode()` implementation. |

**Mode enum fragments** (for any mode when `enum_def` is present):

| Fragment | Target Files | Insertion |
|----------|-------------|-----------|
| `fragments/mode_backend_plumbing_<field>.txt` → `HipdnnBackendAttributeType.h` section | `$HIPDNN_SRC/backend/include/HipdnnBackendAttributeType.h` | Type tag entry. **Replace `PLACEHOLDER_VALUE`** with the next available value in `HipdnnBackendAttributeType.h`. |
| `fragments/mode_backend_plumbing_<field>.txt` → other sections | `DataTypeConversion.{hpp,cpp}`, `DescriptorAttributeUtils.{hpp,cpp}`, `BackendEnumStringUtils.hpp`, `hipdnn_backend.h` | Insert each section into the corresponding file. The fragment has clearly labeled sections. |
| `fragments/mode_frontend_plumbing_<field>.txt` | `$HIPDNN_SRC/frontend/include/hipdnn_frontend/Types.hpp` | Sections for the enum class, `toBackend<Foo>Mode`, and `fromHipdnn<Foo>Mode`. Insert each near existing similar code. |

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

### 9. Apply Descriptor Lifting Additions (`backend` and `full` modes)

If `fragments/descriptor_lifting_additions.txt` exists:
- Read it for the exact changes needed to the existing descriptor `.hpp` and `.cpp`
- Apply each change (add `#include <unordered_map>`, `fromNode()` declaration, `_name` member to `.hpp`; add operation name/type handling and `fromNode()` impl to `.cpp`)

If the op uses a mode enum, verify the inverse `fromHipdnn<Foo>Mode` converter is present in `Types.hpp` — the unpacker calls it:
```bash
grep "fromHipdnn<Foo>Mode" $HIPDNN_SRC/frontend/include/hipdnn_frontend/Types.hpp
```
If absent, insert it from `mode_frontend_plumbing_<field>.txt`.

### 10. Wire Frontend Node (if node class exists or was generated)

For `backend` mode when a frontend node already exists, or for `full` mode:
- Add `#include` for packer header to the node class
- Add `create_operation()` override calling the packer
- Add `#include` for unpacker header
- Add `unpack_from_descriptor()` override calling the unpacker

### 11. Ask About Operation-Specific Logic

There are two classes of question the agent is expected to ask the user. Everything else is derived from the schema, existing code, and conventions.

**Class A: cuDNN naming uncertainty (fallback only — see Step 3a-i).** First try the cudnn-frontend GitHub web-check. Only ask the user when the directory listing is ambiguous, the matched file disagrees with itself, or web access is unavailable.

- "I don't know the cuDNN equivalent for this operation. What is the full cuDNN backend descriptor constant name (e.g., `CUDNN_BACKEND_OPERATION_<X>_DESCRIPTOR`), or 'none' if hipDNN-specific?"
- "I cannot confidently match attribute `<name>` to a cuDNN constant. What is the full `CUDNN_ATTR_<X>` name, or 'none' if hipDNN-specific?"
- "I'm not sure what to name the frontend node class / Graph API method (e.g., `Graph::reduction` vs `Graph::reduce`). Confirm the preferred name."

**Class B: operation-specific logic (only for `frontend` or `full` mode).**

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
  - **Default:** leave with just the standard null/dim checks

Do NOT ask the user about any other fields or decisions — derive everything else from the schema, existing code, and conventions.

### 12. Verify Generated Integration Tests

The generator produces complete integration tests for both lowering and lifting. Verify each item below is present in the placed files; if any is missing, the generator output was incomplete and you must regenerate or hand-add it.

**Lowering** (`Integration<Op>DescriptorLowering.cpp`):
- [ ] `<Op>LoweringRoundTrip` — full round-trip with explicit UIDs, per-tensor dims/strides from the constants header, mode and vector field verification
- [ ] Per-optional-scalar preservation tests

**Lifting** (`Integration<Op>DescriptorLifting.cpp`):
- [ ] `Basic<Op>RoundTrip` — full lifting round-trip with field-by-field validation
- [ ] `<Op>TensorSharingPreserved` — pointer equality verification
- [ ] `<Op>LiftWithoutFinalization` — backend binary serialization path
- [ ] `AutoAssignedUidsPreservedInLiftingRoundTrip` — auto-assigned UID distinctness and round-trip
- [ ] Per-optional-scalar preservation tests

All tests must use `K_TENSOR_*` constants from the shared constants header — no inline literals.

Both integration test files must include and use the shared utilities from `test_sdk/include/hipdnn_test_sdk/utilities/`:
- `IntegrationTestFixture.hpp` — base fixture class; test classes inherit from `hipdnn_tests::IntegrationTestFixture`
- `LoweringTestHelpers.hpp` — `lowerAndDeserialize`, `TestableGraphLowering`, `buildTensorMap` (lowering file)
- `LiftingTestHelpers.hpp` — `lowerAndLift`, `TestableGraphLifting` (lifting file)

The generator templates produce these includes automatically. If they are absent from a placed file, the template was not used or the output was truncated — regenerate rather than hand-adding helpers ad hoc.

After verifying, consider adding operation-specific tests for multi-input variants (e.g., pointwise ternary) or multi-operation graphs (e.g., conv+bias+relu chains) as needed.

If the frontend node class or graph method does not exist yet (e.g., backend-only mode), skip this step but note it as pending.

### 13. Build and Test (MANDATORY)

**You MUST build and test before reporting success.** A build verifies that all fragments were inserted, all converters exist, all includes resolve, and all types match. Skipping the build means you cannot know if the integration is correct.

Before building, do a quick self-check:
- Every function called in the unpacker exists (grep for it in the target file)
- Every `#include` in new files points to a real header
- Every switch case references a valid enum constant
- CMake lists include all new source/test files

Build the project. The exact `cmake` invocation depends on whether you're in a standalone hipDNN checkout or a `rocm-libraries` superbuild — see [`docs/Building.md`](../../../../../docs/Building.md) for the canonical commands. In an existing build directory:

```bash
cd $HIPDNN_SRC/build  # or your active build dir
ninja 2>&1 | tail -100
```

If the build fails, **read the errors and fix them**. Common issues:
- **Missing converter function in `Types.hpp`** → insert from `mode_frontend_plumbing_<field>.txt`. Includes both `toBackend<Foo>Mode` and `fromHipdnn<Foo>Mode` — the unpacker calls the inverse.
- **Missing `#include`** → add it.
- **Wrong attribute name (missing or extra `_EXT` suffix)** → check `HipdnnBackendAttributeName.h` against the YAML's `attr_suffix` per field. Re-derive per the cuDNN parity rule in Step 3a.
- **Test file not built** → the CMake fragment for tests was not inserted. Confirm `backend/tests/CMakeLists.txt`, `frontend/tests/CMakeLists.txt`, and `tests/frontend/CMakeLists.txt` each list the new files.
- **Unpacker calls a converter that exists in backend headers but not in frontend `Types.hpp`** → the `mode_frontend_plumbing` fragment was only partially applied. Insert the inverse converter from the fragment.
- **Enum value collision** → re-pick the next free value from `HipdnnBackendDescriptorType.h` or `HipdnnBackendAttributeName.h` and re-run insertion.
- **Type mismatch** → check the generated code against existing patterns.

After the build succeeds, run unit tests:
```bash
ninja unit-check 2>&1 | tail -50
```

If tests fail, diagnose and fix. Do not report success with failing tests.

### 14. Report Results

Summarize what was generated and placed:
- List all files created/modified
- **Surface the cuDNN parity decisions made and their source** — for each YAML name field (`enum_name`, `attr_suffix`, `compute_data_type_attr`, `operation_type_enum`), state either the matching cuDNN constant or "no equivalent — `_EXT` applied". When the source was the cudnn-frontend web-check, include the GitHub URL fetched and the snippet/line referenced (per Step 3a-i). This lets review verify Step 3a was applied correctly and spot any misread.
- Note any stubs that still need implementation (custom infer_properties, custom validation)
- Note any fragment insertions that need manual verification (enum value ranges, CMake)
- Confirm build passed and tests passed
- **List the hand-authored pieces still required** (out of scope for this skill):
  - JSON serialization helper at `flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/utilities/json/<Op>Attributes.hpp` (template: `ConvolutionFwdAttributes.hpp`)
  - Python bindings (only if the op is on the curated Python surface — see [`AddingNewOperations.md`](../../../../../docs/AddingNewOperations.md#json-utilities-and-python-bindings))
  - End-user sample under `samples/<op>/`
- If anything failed and could not be fixed, explain what and why

## Error Handling

- If the FBS schema has field types not supported by the generator, report them and ask the user how to handle
- If enum value ranges conflict with existing values, report and ask the user for the correct range
- If the generator script fails, show the full error output
- If build fails after placement, show the last 100 lines and help diagnose
- If a target file for fragment insertion cannot be found, report and skip that insertion

## Notes

- The generated integration tests (lowering and lifting) are complete with round-trip, tensor sharing, auto-UIDs, and per-scalar tests. Review and add operation-specific tests as needed.
- For adding lifting to an existing op, run `--mode backend`; the `descriptor_lifting_additions.txt` fragment in the output contains the in-place changes needed for the existing descriptor files.
- `mode` type is REQUIRED for all enum fields in new operations. Never use the legacy `enum` type.
- Read `$CODEGEN/CLAUDE.md` for the full detailed post-generation workflow if you need additional context on any step.
- Always use `convolution_fwd.yaml` as the reference config when creating new configs.
- Do NOT ask the user questions about config fields, enum names, UIDs, or other derivable information. The only user-facing questions should be about `infer_properties` strategy and custom validation rules.
- Generated mode enum plumbing (`enum_def`) supports `frontend_value` for cases where frontend and backend enum values differ. Always verify against existing `Types.hpp` when populating `enum_def`.
- The generator auto-produces a shared constants header (`<Op>Constants.hpp`) from YAML `test_data` when `constants_include` is not set. All test templates reference it. When `constants_include` IS set, the existing header is used and no constants file is generated.
- The generated code is a starting point. Review each file and fragment against the current state of hipDNN before placing. Prefer existing code when it's correct; merge when each covers different parts; use generated code when the target doesn't exist yet.
- For human-facing context — cuDNN parity rules, layer-by-layer reference, testing matrix, and PR checklist — see [`projects/hipdnn/docs/AddingNewOperations.md`](../../../../../docs/AddingNewOperations.md).
