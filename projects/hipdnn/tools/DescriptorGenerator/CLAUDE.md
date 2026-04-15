# Post-Generation Integration Guide

## Purpose and Philosophy

This tool bulk-generates the boilerplate code needed to push a new operation type all the way through hipDNN — from FBS schema to backend descriptor, packer/unpacker, frontend attributes/node, tests, and all the plumbing fragments (enum entries, factory cases, CMake entries, mode enum infrastructure). It replaces hours of copy-paste-adapt work with a single command.

**Always run the generator.** The generator must be executed to produce output files and fragments. Never bypass it by writing boilerplate code from scratch — even if you understand the templates and patterns. The generator ensures consistency across all artifacts, handles template edge cases, and produces tested output. Writing code by hand introduces drift from the established patterns and defeats the purpose of the tool.

The tool generates; the agent integrates. Generated code is a starting point, not a final product. Before placing any generated file, review the current state of hipDNN. Files, enums, converters, or plumbing may already exist — partially or fully. When a target file already exists, compare the generated version with the existing code and decide whether to use, keep, or merge. Fragment snippets show what to insert, but the exact insertion point, surrounding whitespace, and adjacent code must be verified by reading the target file. Small adjustments — fixing naming mismatches, adjusting value mappings, adding missing includes — are expected as part of a clean integration. The agent owns correctness end-to-end: if a generated file does not compile or a fragment conflicts with existing code, the agent fixes it. These adjustments are applied to generator output, not as a substitute for running the generator.

Existing code takes precedence. When generated code conflicts with hand-written code that is already correct and tested, prefer the existing code. The generator's value is in creating the 80% of boilerplate that does not exist yet, not in overwriting code that works.

---

## End-to-End Workflow

Adding a new operation follows this sequence:

1. **Create the FBS schema** — Define the operation's FlatBuffer schema (e.g., `matmul_attributes.fbs`) in `data_sdk/schemas/` and run `flatc` to generate the C++ header
2. **Create or update the YAML config** — Write `configs/<operation>.yaml` referencing the schema fields
3. **Run the generator** — `python generate.py --config configs/<op>.yaml --output-dir /tmp/output`
4. **Add enums** — Insert new enum values into the backend headers (see steps below)
5. **Add enum test coverage** — Add `EXPECT_STREQ` entries to `TestBackendEnumStringUtils.cpp`
6. **Place generated files** — Copy generated source and test files into the project tree
7. **Add lifting support** — For operations that need lifting (graph → descriptor → frontend), run `--lift-only` to generate unpacker, fromNode test, and fragment templates. Apply changes from `fragments/descriptor_lifting_additions.txt` to existing descriptor files.
8. **Update CMake** — Add new source and test files to the build
9. **Review and build** — Compile, run tests, review generated code
10. **Extract test constants** — Replace inline test literals with named constants (see Step 10 below)
11. **Review integration test** — The generated integration test includes a lowering round-trip test and per-scalar preservation tests; add hand-written tests for auto-UIDs, multi-input variants, and multi-operation graphs as needed (see Step 11 below)

---

## Step 4: Add Enums

### 4a. Attribute Enum — `backend/include/HipdnnBackendAttributeName.h`

Insert the content from `fragments/attribute_enum_block.txt`.

- Find the last operation attribute enum range (e.g., ConvFwd uses 1400-1405)
- Assign the next available range to the new operation
- Replace `PLACEHOLDER_VALUE` with the actual enum values
- Also add shared attribute names if the operation introduces new ones

### 4b. Descriptor Type Enum — `backend/include/HipdnnBackendDescriptorType.h`

Insert the content from `fragments/descriptor_type_enum.txt`.

- Add the new descriptor type to the `hipdnnBackendDescriptorType_t` enum

### 4c. String Utilities — `backend/src/BackendEnumStringUtils.hpp`

Insert the content from `fragments/string_utils_block.txt`.

- Add switch cases to `hipdnnGetBackendDescriptorTypeName()`
- Add switch cases to `hipdnnGetBackendAttributeNameString()`
- The fragment contains both sets of cases

### 4d. Descriptor Factory — `backend/src/descriptors/DescriptorFactory.cpp`

Insert the content from `fragments/factory_case.txt`.

- Add the `#include` for the new descriptor header at the top
- Add a `case` entry in the `DescriptorFactory::create()` switch

---

## Step 5: Add Enum Test Coverage

After adding enums and string utility cases, add corresponding test coverage in `backend/tests/TestBackendEnumStringUtils.cpp`:

### Descriptor Type Name Test

Add to the `GetBackendDescriptorTypeName` test:

```cpp
EXPECT_STREQ(
    hipdnnGetBackendDescriptorTypeName(HIPDNN_BACKEND_OPERATION_<OP>_DESCRIPTOR),
    "HIPDNN_BACKEND_OPERATION_<OP>_DESCRIPTOR");
```

### Attribute Name Test

Add to the `GetBackendAttributeName` test — one `EXPECT_STREQ` for each new attribute enum:

```cpp
// Operation-specific attributes
EXPECT_STREQ(hipdnnGetAttributeNameString(HIPDNN_ATTR_OPERATION_<OP>_X),
             "HIPDNN_ATTR_OPERATION_<OP>_X");
EXPECT_STREQ(hipdnnGetAttributeNameString(HIPDNN_ATTR_OPERATION_<OP>_Y),
             "HIPDNN_ATTR_OPERATION_<OP>_Y");
// ... one entry per attribute

// Shared attributes (if introducing new ones)
EXPECT_STREQ(hipdnnGetAttributeNameString(HIPDNN_ATTR_<SHARED>_COMP_TYPE),
             "HIPDNN_ATTR_<SHARED>_COMP_TYPE");
```

Every enum value added to `BackendEnumStringUtils.hpp` must have a corresponding `EXPECT_STREQ` in this test file.

---

## Step 6: Place Generated Files

Copy the complete generated files to their target locations:

| Generated File | Target Location |
|----------------|-----------------|
| `backend/src/descriptors/<Op>OperationDescriptor.hpp` | `projects/hipdnn/backend/src/descriptors/` |
| `backend/src/descriptors/<Op>OperationDescriptor.cpp` | `projects/hipdnn/backend/src/descriptors/` |
| `frontend/include/hipdnn_frontend/detail/<Op>Packer.hpp` | `projects/hipdnn/frontend/include/hipdnn_frontend/detail/` |
| `backend/tests/descriptors/Test<Op>OperationDescriptor.cpp` | `projects/hipdnn/backend/tests/descriptors/` |
| `backend/tests/descriptors/TestGraphDescriptor<Op>.cpp` | `projects/hipdnn/backend/tests/descriptors/` |
| `tests/frontend/Integration<Op>DescriptorLowering.cpp` | `projects/hipdnn/tests/frontend/` |
| `tests/frontend/Integration<Op>DescriptorLifting.cpp` | `projects/hipdnn/tests/frontend/` |

### 6a. Wire `create_operation` in the Frontend Node

The generated packer file provides a `create<Op>Operation()` function, but the frontend node class must call it. If the node class already exists (e.g., `ConvolutionWgradNode.hpp` in `frontend/include/hipdnn_frontend/node/`), add:

1. An `#include` for the generated packer header:
```cpp
#include "hipdnn_frontend/detail/<Op>Packer.hpp"
```

2. A `create_operation` override that calls the packer:
```cpp
Error create_operation(
    std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
{
    return detail::create<Op>Operation(get_attributes(), tensorDescs, operations);
}
```

Use `ConvolutionFpropNode.hpp` as the reference for this pattern. The `create_operation` method is what connects the frontend graph builder to the backend descriptor API via the generated packer.

**If the frontend node class does NOT exist yet**, skip this step and note in your summary that the packer is ready but the node needs `create_operation` wired up.

---

## Step 7: Add Lifting Support

For operations that need lifting (reconstructing frontend graph attributes from backend C API descriptors), use the `--lift-only` flag:

```bash
.venv/bin/python generate.py --config configs/<op>.yaml --output-dir /tmp/lift-output --lift-only
```

This generates lifting-related files including the unpacker, fromNode tests, lifting integration tests, and fragment templates:

| Generated File | Purpose |
|----------------|---------|
| `frontend/include/hipdnn_frontend/detail/<Op>Unpacker.hpp` | Frontend unpacker (inverse of packer) |
| `backend/tests/descriptors/Test<Op>OperationFromNode.cpp` | fromNode() round-trip tests |
| `tests/frontend/Integration<Op>DescriptorLifting.cpp` | Lifting round-trip integration tests |
| `fragments/node_factory_case.txt` | NodeFactory switch case for this operation |
| `fragments/operation_unpacker_case.txt` | OperationUnpacker switch case |
| `fragments/operation_type_enum.txt` | hipdnnOperationType_ext_t enum entry |
| `fragments/node_unpack_override.txt` | Node class unpack_from_descriptor override |
| `fragments/descriptor_lifting_additions.txt` | Manual changes for existing descriptor files |

### 7a. Apply Descriptor Additions

The `descriptor_lifting_additions.txt` file contains the exact changes to make to the existing `<Op>OperationDescriptor.hpp/.cpp`:

- **HPP**: Add `#include <unordered_map>`, `fromNode()` declaration, `_name` member
- **CPP**: Add `HIPDNN_ATTR_OPERATION_NAME_EXT` handling in setAttribute/getAttribute, `HIPDNN_ATTR_OPERATION_TYPE_EXT` handling in getAttribute, `_name` in buildNode/toString, and the full `fromNode()` implementation

### 7b. Place Lifting Files

| Generated File | Target Location |
|----------------|-----------------|
| `<Op>Unpacker.hpp` | `projects/hipdnn/frontend/include/hipdnn_frontend/detail/` |
| `Test<Op>OperationFromNode.cpp` | `projects/hipdnn/backend/tests/descriptors/` |
| `Integration<Op>DescriptorLifting.cpp` | `projects/hipdnn/tests/frontend/` |

### 7c. Wire Lifting Fragments

Insert the content from each fragment into the corresponding shared file:

| Fragment | Target File | What to Add |
|----------|-------------|-------------|
| `node_factory_case.txt` | `backend/src/descriptors/NodeFactory.cpp` | Case in `createFromNode()` switch |
| `operation_unpacker_case.txt` | `frontend/src/OperationUnpacker.cpp` | Case in the unpacker dispatch |
| `operation_unpacker_test.txt` | `frontend/tests/TestOperationUnpacker.cpp` | Uncomment existing test or insert generated test in the `createNodeForType` tests section. Also uncomment the corresponding `#include` for the node header. |
| `operation_type_enum.txt` | `backend/include/HipdnnOperationType.h` | Enum entry for this operation |
| `node_unpack_override.txt` | Frontend node header (e.g., `ConvolutionFpropNode.hpp`) | `unpack_from_descriptor` override |

### 7d. Wire `unpack_from_descriptor` in the Frontend Node

The generated `node_unpack_override.txt` provides the `unpack_from_descriptor()` override. Add it to the frontend node class along with the unpacker include:

```cpp
#include "hipdnn_frontend/detail/<Op>Unpacker.hpp"

Error unpack_from_descriptor(
    hipdnn_backend_descriptor_t const opDesc) override
{
    return detail::unpack<Op>(get_attributes(), opDesc);
}
```

### 7e. Update CMake

Add the new test file to `backend/tests/CMakeLists.txt`. The cmake_entries.txt fragment (from full generation) includes the fromNode test entry.

### 7f. Update Packer for Name Support

The lift-only path adds `_name` and `HIPDNN_ATTR_OPERATION_NAME_EXT` handling to the descriptor, but the **existing packer** must also be updated to pack the name. Apply the code from `fragments/packer_name_addition.txt` — add the name-setting block before the `finalizeDescriptor()` call in the packer.

Without this change, frontend→backend→frontend round-trips will silently lose the operation name.

### 7g. Graph Descriptor Name Tests (Auto-Generated)

The graph descriptor test template (`test_graph_ops.cpp.j2`) now auto-generates operation name tests:

- The `createFinalized<Op>Op` helper includes a `const std::string& name = ""` parameter
- `OperationNamePreservedInSerialization` test — verifies name appears in deserialized FlatBuffer
- `OperationNameRoundTripThroughLifting` test — verifies name survives full serialize, deserializeGraph, re-serialize cycle

These tests are generated unconditionally (not gated on `data_fields`), so operations with no data fields still get name test coverage. No manual work is needed for this step.

### 7h. Deepen fromNode Test Coverage

The generated `Test<Op>OperationFromNode.cpp` provides basic scaffolding but requires manual deepening. The generated tests only verify tensor UIDs — they do NOT verify full tensor reconstruction. Add:

1. **`verifyTensorDescriptor` helper** on the test fixture that verifies UID, data_type, dims, AND strides through the `getAttribute` API (query count first with `requestedElementCount=0`, then retrieve). This proves the entire tensor object rebuilds correctly, not just its identity.

2. **`GetAttributeWorksAfterFromNode`** — exhaustive test that calls `getAttribute` for EVERY attribute on the descriptor after `fromNode()`, verifying full tensor values (UID + data_type + dims + strides) for all tensors via the helper. Also verify compute data type, operation type, and operation name.

3. **`SetsTensorReferencesWithFullValues`** — verify tensor data_type, dims, and strides through the direct accessor path (`desc->getXDesc()->getData().data_type`, `.dims`, `.strides`), not just UIDs.

4. **Finalize validation tests** — if the descriptor has cross-field validation in `finalize()` (e.g., both-or-none constraints), add tests that exercise those constraints through `fromNode()`.

5. **Pointer identity tests** — verify that `desc->getXDesc() == _tensorMap[expectedUid]` (same `shared_ptr` instance, not a copy). This proves `fromNode()` reuses tensors from the map rather than creating duplicates.

**Critical rule**: Never write tests that only verify UIDs. Every test that touches a tensor must verify at minimum UID + data_type. Round-trip and getAttribute tests must verify the full quartet: UID, data_type, dims, strides. Surface-level UID-only checks give false confidence — they pass even when tensor data is corrupted or missing.

---

## Step 8: Update CMake

Insert the content from `fragments/cmake_entries.txt`.

- Add the new `.cpp` source file to `backend/src/CMakeLists.txt`
- Add the new test `.cpp` files to the appropriate test CMakeLists files:
  - `backend/tests/CMakeLists.txt` for descriptor unit tests
  - `tests/CMakeLists.txt` for integration tests

---

## Step 9: Review and Build

```bash
cd projects/hipdnn/build
ninja                 # Builds without errors
ninja unit-check      # All unit tests pass
ninja check           # All tests pass (if GPU available)
```

Review the generated code for correctness, paying attention to:
- Enum values are in the correct range and don't conflict
- String utility switch cases match the enum names exactly
- Test coverage covers all new enums
- Factory case uses the correct descriptor type and class

### Test Quality Checklist

Before considering tests complete, verify:

- [ ] **Full tensor verification**: All tests that access tensors verify UID + data_type + dims + strides (not just UID)
- [ ] **getAttribute path**: `GetAttributeWorksAfterFromNode` tests every attribute via the backend API, drilling into packed tensor descriptors to verify full values
- [ ] **Direct accessor path**: `SetsTensorReferencesWithFullValues` verifies tensor values through `getData()` for all tensors
- [ ] **Pointer identity**: `TensorReferencesMatchTensorMap` confirms `shared_ptr` identity (not just value equality)
- [ ] **Round-trip**: `BuildNodeRoundTrip` verifies all fields including optional tensors and tensor arrays
- [ ] **Name E2E**: Name round-trips through both `fromNode`→`getAttribute` and graph serialize→deserialize→fromNode→re-serialize
- [ ] **Validation constraints**: Cross-field invariants in `finalize()` are tested via `fromNode()` (e.g., both-or-none, all-or-none)
- [ ] **Error paths**: Missing required tensors and missing-but-referenced optional tensors all tested

---

## Step 10: Place Generated Constants File

The generator automatically produces a shared constants header (`<Op>Constants.hpp`) from the YAML config's `test_data` section. This file contains `K_TENSOR_<NAME>_UID`, `K_TENSOR_<NAME>_DIMS`, `K_TENSOR_<NAME>_STRIDES` constants for each tensor, plus constants for vector data fields (e.g., `K_CONV_PADDING`). All generated test files reference this header.

**When `constants_include` is NOT set** in the YAML (the common case for new operations), the generator produces the constants file at:
```
test_sdk/include/hipdnn_test_sdk/constants/<Op>Constants.hpp
```
Place this file alongside the other generated files.

**When `constants_include` IS set** (for operations with pre-existing hand-written constants headers), the generator skips constants file generation and the test files reference the existing header.

### Post-placement review

After placing the generated constants file:
1. Verify the values match the YAML config's `test_data` section
2. If your operation shares constants with related operations (e.g., ConvBwd shares tensor shapes with ConvFwd), consider using the existing shared header by setting `constants_include` in the YAML instead
3. Constants follow the naming convention `K_TENSOR_<NAME>_UID/DIMS/STRIDES` for tensors and `K_<CATEGORY>_<NAME>` for data fields (e.g., `K_CONV_PADDING`)

---

## Step 11: Review and Extend the Integration Test

The generated integration test (`Integration<Op>DescriptorLowering.cpp`) includes:

- A `buildAndDeserialize` helper method on the fixture that creates tensors, calls the graph method, validates, lowers via `build_operation_graph_via_descriptors`, serializes via `hipdnnBackendGetSerializedBinaryGraph_ext`, and deserializes into a `GraphT`
- `<Op>LoweringRoundTrip`: verifies required tensor UIDs, mode field, and required vector fields in the deserialized FlatBuffer
- Per-optional-scalar tests (`<ScalarName>PreservedInRoundTrip`): one test per optional scalar field

### Additional Hand-Written Tests

The generated tests cover the basic single-input scenario and include an `AutoAssignedUidsPreservedInLiftingRoundTrip` test. Consider adding these as needed:

**Multi-input/ternary variants** — For operations like pointwise that have optional additional inputs (in_1, in_2), add tests exercising binary and ternary graph method overloads.

**Multi-operation graphs** — If the operation commonly appears in chains (e.g., conv+bias+relu), test that multi-node graphs serialize correctly.

### Integration Test Dependencies

The integration test requires the frontend node type to exist. Specifically:
- The frontend attributes class (e.g., `ConvFpropAttributes`) in `frontend/include/hipdnn_frontend/attributes/`
- The frontend node class (e.g., `ConvolutionFpropNode`) in `frontend/include/hipdnn_frontend/nodes/`
- The graph method (e.g., `graph->conv_fprop()`) in `frontend/include/hipdnn_frontend/Graph.hpp`

If these don't exist yet, the integration test cannot be compiled. In that case:
- Still place the generated file in `tests/frontend/`
- Do NOT add it to `tests/frontend/CMakeLists.txt` until the frontend types exist
- Note in your summary that the integration test is pending frontend implementation

---

## Creating a YAML Config from an FBS Schema

If a YAML config does not already exist for your operation, create one from the FBS schema. The YAML maps schema fields to hipDNN backend API concepts.

### Mapping FBS Fields to YAML

Given an FBS schema like:

```fbs
table ConvolutionWrwAttributes {
    x_tensor_uid: long;       // → tensor_fields entry
    dy_tensor_uid: long;      // → tensor_fields entry
    dw_tensor_uid: long;      // → tensor_fields entry
    pre_padding: [long];      // → data_fields entry (vector_int64)
    post_padding: [long];     // → data_fields entry (vector_int64)
    stride: [long];           // → data_fields entry (vector_int64)
    dilation: [long];         // → data_fields entry (vector_int64)
    conv_mode: ConvMode;      // → data_fields entry (enum)
}
```

Apply these rules:

| FBS Field Pattern | YAML Section | YAML `type` |
|---|---|---|
| `*_tensor_uid: long` | `tensor_fields` | (implicit — tensors are always UIDs) |
| `field: [long]` | `data_fields` | `vector_int64` |
| `field: SomeEnum` | `data_fields` | `mode` (preferred; see "Enum Field Types" section) |
| `field: float` | `data_fields` | `scalar_float` |
| `field: long` (non-UID) | `data_fields` | `scalar_int64` |
| `field: bool` | `data_fields` | `bool` |
| `field: [long]` (array of UIDs) | `tensor_array_fields` | (for peer_stats etc.) |

### Required YAML Fields

```yaml
operation:
  # Identity — derived from the FBS table name
  name: "ConvolutionWrw"                          # PascalCase, used in class/file names
  class_name: "ConvolutionWrwOperationDescriptor"  # = name + "OperationDescriptor"
  fbs_table: "ConvolutionWrwAttributes"            # Must match the FBS table name exactly
  fbs_generated_header: "convolution_wrw_attributes_generated.h"  # The flatc-generated header

  # Backend enum names — follow existing naming conventions
  descriptor_type:
    enum_name: "HIPDNN_BACKEND_OPERATION_CONVOLUTION_WRW_DESCRIPTOR"
  operation_attr_prefix: "HIPDNN_ATTR_OPERATION_CONVOLUTION_WRW"  # Tensor attrs = prefix + "_" + suffix

  # Frontend mapping — look at the existing node/attributes classes
  frontend:
    packer_function: "createConvWgradOperation"    # Function name in the generated packer
    node_class: "ConvolutionWgradNode"              # The frontend node class (if it exists)
    attributes_class: "ConvWgradAttributes"         # The frontend attributes class (if it exists)
    # Lifting support (optional, required for --lift-only mode)
    unpacker_function: "unpackConvFprop"             # Function name in the generated unpacker
    unpacker_include: ""                              # Override derived include file name (optional)

  # Shared attributes — if this operation reuses attributes from another operation
  # (e.g., all conv ops share HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS), use the SAME
  # attr_name values. Do NOT create new per-operation copies.
  has_compute_data_type: true
  compute_data_type_attr: "HIPDNN_ATTR_CONVOLUTION_COMP_TYPE"  # Shared across conv ops
  operation_type_enum: "HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT"  # For HIPDNN_ATTR_OPERATION_TYPE_EXT

  # Test data — UIDs should be distinct across operations to avoid confusion
  test_data:
    tensor_uids: { x: 20, dy: 21, dw: 22 }         # Unique UIDs for test tensors
    tensor_configs:                                    # Realistic dims/strides for each tensor
      x: { dims: [1, 3, 32, 32], strides: [3072, 1024, 32, 1] }
      dy: { dims: [1, 64, 32, 32], strides: [65536, 1024, 32, 1] }
      dw: { dims: [64, 3, 3, 3], strides: [27, 9, 3, 1] }
    field_values:                                      # Test values for data fields
      pre_padding: [1, 1]
      post_padding: [1, 1]
      stride: [1, 1]
      dilation: [1, 1]
```

### Additional Data Field Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `shared` | bool | `false` | If `true`, the attribute enum already exists (defined by another operation). Fragment templates skip shared fields to avoid duplicate enum entries. Core templates still include them for setAttribute/getAttribute. Note: `enum_def` can be present alongside `shared: true` -- see below. |
| `test_enum_value` | string | `""` | **Required for enum fields.** The enum constant to use in generated tests (e.g., `CROSS_CORRELATION` for ConvMode, `ADD` for PointwiseMode). |
| `test_label` | string | `""` | Label used in generated test case names (e.g., `"Convolution"`). |
| `test_constant_name` | string | `""` | Named constant reference for test values (e.g., `"K_CONV_PADDING"`). |
| `build_node_check` | bool | `true` | Whether this field is verified in the buildNode round-trip test. |
| `default_value` | string | `""` | Default value expression for the field (e.g., `"ConvolutionMode::CROSS_CORRELATION"`). |
| `frontend_inverse_converter` | string | `""` | Conversion function from backend C-API value back to frontend enum (used in unpacker). Only needed for `mode` fields. Example: `toFrontendConvMode` |

#### `shared` and `enum_def` Coexistence

The `enum_def` block can be present on both `shared: true` and `shared: false` fields:

- **When `shared: false`**: the generator produces all mode enum plumbing from `enum_def` (backend C-API header, backend plumbing fragment, frontend plumbing fragment).
- **When `shared: true`**: the `enum_def` serves as documentation and reference only -- the generator skips plumbing generation because another operation already defined this enum. The practical benefit is that `enum_def` documents the enum values inline in the config and enables the skill agent to verify the enum infrastructure without searching the codebase.

For example, `convolution_bwd.yaml` has `shared: true` AND `enum_def` on its conv_mode field because the ConvMode enum was already created by convolution_fwd. The `enum_def` is present purely to document the enum values for reference.

### Mode Field Properties

Mode fields (data fields with `type: mode`) require additional properties to wire up the enum plumbing. All mode fields must include:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `cpp_enum` | string | Yes | Fully-qualified SDK enum type (e.g., `hipdnn_data_sdk::data_objects::ConvMode`) |
| `frontend_type` | string | No | Frontend enum name when different from SDK short name (e.g., `ConvolutionMode` for SDK `ConvMode`) |
| `backend_type_name` | string | Yes | Backend type tag (e.g., `HIPDNN_TYPE_CONVOLUTION_MODE`) |
| `backend_setter` | string | Yes | Setter helper name (e.g., `setConvMode`) |
| `backend_getter` | string | Yes | Getter helper name (e.g., `getConvMode`) |
| `backend_converter` | string | Yes | Frontend-to-backend converter (e.g., `toBackendConvMode`) |
| `frontend_inverse_converter` | string | Yes (for lifting) | Backend-to-frontend converter (e.g., `fromHipdnnConvMode`) |
| `test_c_type` | string | Yes | C-API typedef name (e.g., `hipdnnConvolutionMode_t`) |
| `test_backend_value` | string | Yes | C-API constant for test default (e.g., `HIPDNN_CROSS_CORRELATION`) |
| `test_default_value` | string | No | Alternative C-API constant for default-value tests |
| `test_alt_enum_value` | string | No | Alternative SDK enum value for round-trip tests (e.g., `CONVOLUTION`) |
| `default_value` | string | No | Frontend default value expression (e.g., `ConvolutionMode::CROSS_CORRELATION`) |

### Operation-Level Shared Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `compute_data_type_shared` | bool | `false` | If `true`, the compute data type attribute enum already exists. Fragment templates omit it. Use for operations that share compute type attributes with another operation (e.g., ConvBwd/ConvWrw share `HIPDNN_ATTR_CONVOLUTION_COMP_TYPE` from ConvFwd). |

### Enhanced `tensor_array_fields`

Tensor array fields support additional properties for test generation:

```yaml
tensor_array_fields:
  - name: "peer_stats"
    fbs_field: "peer_stats_tensor_uid"
    attr_name: "HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS"
    frontend_getter: "get_peer_stats()"
    required: false         # Whether the field must be set before finalize
    test_uids: [100, 101]   # UIDs for test tensor descriptors
    test_label: "PeerStats"  # Label used in test case names
```

### Tips

- **Use `convolution_fwd.yaml` as the reference** -- it's the most complete and validated config
- **Shared attributes**: Convolution ops all share `HIPDNN_ATTR_CONVOLUTION_*` attributes. Matmul, pointwise, and batchnorm each have their own attribute namespaces. Use `shared: true` on data fields and `compute_data_type_shared: true` at operation level for operations that reuse another operation's attribute enums.
- **Frontend naming**: The packer function, node class, and attributes class names must match the existing frontend code. Check `frontend/include/hipdnn_frontend/` for the actual class names
- **Enum fields**: Set `cpp_enum` to the fully-qualified FBS enum type (e.g., `hipdnn_data_sdk::data_objects::ConvMode`). Set `required: false` if the FBS has a default value. Always set `test_enum_value` to a valid enum constant.

---

## Enum Field Types: `mode` vs `enum` (Legacy)

### `mode` Type — REQUIRED for All Enum Fields in New Operations

Every enum field in a new operation MUST use the `mode` type. The `mode` pattern provides:
- A dedicated backend C-API enum (e.g., `hipdnnConvolutionMode_t`)
- A dedicated type tag (e.g., `HIPDNN_TYPE_CONVOLUTION_MODE`)
- Bidirectional conversion functions in `DataTypeConversion.hpp/.cpp`
- Shared setter/getter helpers in `DescriptorAttributeUtils`
- A frontend converter function

All enums on develop already follow this pattern:
- ConvMode — `HIPDNN_TYPE_CONVOLUTION_MODE` (= 24)
- PointwiseMode — `HIPDNN_TYPE_POINTWISE_MODE` (= 25)
- DiagonalAlignment — `HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT` (= 26)
- AttentionImplementation — `HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT` (= 27)
- NormFwdPhase — `HIPDNN_TYPE_NORM_FWD_PHASE`

### `enum` Type — LEGACY, Do NOT Use for New Operations

The `enum` type uses `HIPDNN_TYPE_INT64` with raw `static_cast`. It exists only for backward compatibility with older configs. Do not use it in new operation YAML configs.

### Adding a New Mode Enum Type

When an operation introduces an enum not already in the backend, the `enum_def` block in the YAML config automates all plumbing. When `enum_def` is present on a `mode` field, the generator produces:

- **Backend C-API header** — `backend/include/<header>.h` (complete file with the C enum typedef)
- **Backend plumbing fragment** — `fragments/mode_backend_plumbing_<field>.txt` containing 7 labeled sections: type tag entry, SDK-to-backend converter, backend-to-SDK converter, attribute utils setter, attribute utils getter, string utils case, and backend include directive
- **Frontend plumbing fragment** — `fragments/mode_frontend_plumbing_<field>.txt` containing 5 labeled sections: frontend-to-backend converter, backend-to-frontend converter, frontend enum string converter, frontend include directive, and attribute conversion case

#### `enum_def` Block Properties

| Property | Type | Description |
|----------|------|-------------|
| `backend_header` | string | Output filename for the C-API enum header (e.g., `HipdnnPointwiseMode.h`) |
| `backend_prefix` | string | Prefix for backend enum constants (e.g., `HIPDNN_POINTWISE_`) |
| `values` | list | Enum values (see below) |

#### `enum_def` Values Entry Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `name` | string | (required) | Backend C-API suffix. Combined with `backend_prefix` to form the full constant name |
| `value` | int | (required) | Backend C-API numeric value |
| `sentinel` | bool | `false` | If true, excluded from backend C-API enum. Appears as `NOT_SET = 0` in frontend |
| `description` | string | `""` | Doxygen description rendered as `///< text` on enum members |
| `sdk_name` | string | `""` | SDK enum constant name when it differs from `name` (e.g., `MAX_OP` for backend `MAX`) |
| `frontend_name` | string | `""` | Frontend enum member name when it differs from `name` (e.g., `TOP_LEFT` for backend `TOP_LEFT_EXT`) |
| `frontend_value` | int or null | `null` | Frontend numeric value when it differs from `value`. Omit when frontend matches backend (the common case). Example: ConvMode backend `CONVOLUTION=0` but frontend `CONVOLUTION=2` |

When `enum_def` is absent, the enum already exists in the codebase and the manual steps apply: add the backend C-API enum header, type tag in `HipdnnBackendAttributeType.h`, SDK conversions in `DataTypeConversion.hpp/.cpp`, shared helpers in `DescriptorAttributeUtils.hpp/.cpp`, string utility case in `BackendEnumStringUtils.hpp`, and frontend converter in `Types.hpp`.

The `mode` type is REQUIRED for all enum fields — both when using `enum_def` and when referencing pre-existing enums.

---

## Post-Generation Refactoring: TEST_P

The code generator produces `TEST_F`-based tests as a baseline. After generation, refactor tests into parameterized `TEST_P` suites where doing so reduces duplication. The templates do not generate TEST_P directly; this is a manual post-generation step.

### When to Refactor to TEST_P

1. **Compute data type variations** — Instead of separate test cases per type (e.g., `BuildNodeWithHalfComputeType`, `BuildNodeWithBfloat16ComputeType`), parameterize over `{backendType, sdkType}` pairs.

2. **Tensor set/get round-trips** — When multiple tensor fields follow the same pattern, parameterize over a struct with `{attr_name, member_ptr, test_uid}`.

3. **Enum value round-trips** — Test each valid enum value via TEST_P instead of one-per-test-case.

4. **Error cases by attribute** — When multiple attributes share the same error pattern (e.g., wrong type, null pointer), parameterize over the attribute name.

### Example: Compute Type Round-Trip

```cpp
struct ComputeTypeParam
{
    hipdnnDataType_t backendType;
    DataType sdkType;
};

class TestComputeTypeRoundTrip : public Test<Op>OperationDescriptor,
                                  public ::testing::WithParamInterface<ComputeTypeParam>
{};

TEST_P(TestComputeTypeRoundTrip, BuildNodePreservesComputeType)
{
    auto [backendType, sdkType] = GetParam();
    // ... set up, finalize, buildNode, verify sdkType ...
}

INSTANTIATE_TEST_SUITE_P(ComputeTypes, TestComputeTypeRoundTrip,
    ::testing::Values(
        ComputeTypeParam{HIPDNN_DATA_FLOAT, DataType::FLOAT},
        ComputeTypeParam{HIPDNN_DATA_HALF, DataType::HALF},
        ComputeTypeParam{HIPDNN_DATA_BFLOAT16, DataType::BFLOAT16}
    ));
```

---

## Required Utilities and Patterns

Generated code and post-generation edits MUST use existing utilities rather than reimplementing equivalent logic.

### Test Utilities — Use, Do Not Reimplement

| Utility | Header | Purpose |
|---------|--------|---------|
| `createDescriptor<T>()` | `DescriptorTestUtils.hpp` | Create typed backend descriptors |
| `createFinalizedTensor()` | `TensorDescriptorTestUtils.hpp` | Create finalized tensor descriptors |
| `ASSERT_THROW_HIPDNN_STATUS()` | `TestMacros.hpp` | Assert specific `hipdnnStatus_t` errors |
| `toVec()` | `test_sdk/utilities/ToVec.hpp` | Convert `std::array` constants to `std::vector` |
| Constants (e.g., `K_TENSOR_X_UID`) | `test_sdk/constants/` | Shared test values |

### Descriptor Attribute Utilities — Use Shared Helpers

| Helper | Purpose |
|--------|---------|
| `setInt64Vector()` / `getInt64Vector()` | Vector fields |
| `setDataType()` / `getDataType()` | Compute data type |
| `setScalar<T>()` / `getScalar<T>()` | Scalar fields |
| Mode-specific (e.g., `setConvMode()` / `getConvMode()`) | Mode enum fields |
| `checkSetArgs()` / `checkGetArgs()` | Type validation |

### Reference Patterns in Graph Tests

| Pattern | Location | Purpose |
|---------|----------|---------|
| `findTensorByUid()` + `verifyTensor()` | `TestGraphDescriptorOps.cpp:64-90` | Locate and validate tensors in deserialized graphs |
| `verify<Op>Node()` helpers | `TestGraphDescriptorOps.cpp:93-116` | Validate operation node attributes |
| Bundle pattern (`ConvOpBundle`) | `TestGraphDescriptorOps.cpp:118-137` | Composite test setup for multi-descriptor operations |
| `UnPackGraph()` | Graph test utils | FlatBuffer deserialization (use instead of `->UnPack()`) |
| `std::unique_ptr<HipdnnBackendDescriptor>` | getAttribute results | Safe ownership wrapping |

### Anti-Patterns to Avoid

- **Raw owning pointers** from `UnPack()` or `getAttribute()` — always wrap in `unique_ptr`
- **Reimplementing existing helpers** — check `DescriptorAttributeUtils` and test utility headers first
- **Using `HIPDNN_TYPE_INT64` for enum fields** — use proper `mode` types with dedicated type tags
- **Inline test values when shared constants exist** — use `test_sdk/constants/` headers
- **Duplicating test logic** that could be parameterized with TEST_P

---

## Notes

- The generated descriptor `.hpp` and `.cpp` files are complete and ready to use as-is
- The packer `.hpp` file is complete and ready to use as-is
- The unit test and graph test files are complete and ready to compile
- The integration test now generates lowering round-trip and per-scalar tests; add operation-specific tests as needed
- Fragment files contain comments indicating where to insert each snippet
- Enum values (PLACEHOLDER_VALUE) must be replaced with actual numeric values following the existing numbering scheme
- The unpacker `.hpp` file is complete and ready to use as-is
- The fromNode test file is complete and ready to compile
- Fragment files for lifting (NodeFactory, OperationUnpacker, operation type enum, node unpack override) contain comments indicating where to insert each snippet
- When `enum_def` is present on a mode field, the generator produces a backend C-API header, a backend plumbing fragment, and a frontend plumbing fragment. Fragment sections are labeled with their target file for easy insertion.
