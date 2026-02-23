# Post-Generation Integration Guide

## End-to-End Workflow

Adding a new operation follows this sequence:

1. **Create the FBS schema** — Define the operation's FlatBuffer schema (e.g., `matmul_attributes.fbs`) in `data_sdk/schemas/` and run `flatc` to generate the C++ header
2. **Create or update the YAML config** — Write `configs/<operation>.yaml` referencing the schema fields
3. **Run the generator** — `python generate.py --config configs/<op>.yaml --output-dir /tmp/output`
4. **Add enums** — Insert new enum values into the backend headers (see steps below)
5. **Add enum test coverage** — Add `EXPECT_STREQ` entries to `TestBackendEnumStringUtils.cpp`
6. **Place generated files** — Copy generated source and test files into the project tree
7. **Update CMake** — Add new source and test files to the build
8. **Review and build** — Compile, run tests, review generated code
9. **Extract test constants** — Replace inline test literals with named constants (see Step 9 below)
10. **Implement integration test** — The generated integration test is a stub; implement full E2E round-trip tests (see Step 10 below)

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

## Step 7: Update CMake

Insert the content from `fragments/cmake_entries.txt`.

- Add the new `.cpp` source file to `backend/src/CMakeLists.txt`
- Add the new test `.cpp` files to the appropriate test CMakeLists files:
  - `backend/tests/CMakeLists.txt` for descriptor unit tests
  - `tests/CMakeLists.txt` for integration tests

---

## Step 8: Review and Build

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

---

## Step 9: Extract Test Constants

The generator inlines literal test values from the YAML config (e.g., `{1, 1}` for padding, `1` for tensor UIDs). After placing the generated files, review the test code and replace inline literals with named constants.

### When to Use Shared Constants (test_sdk)

If the test values are shared across multiple operations or test files, define them in a constants header in the test SDK:

**Location:** `test_sdk/include/hipdnn_test_sdk/constants/<Op>Constants.hpp`

**Example:** `ConvFpropConstants.hpp`
```cpp
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

constexpr int64_t K_TENSOR_X_UID = 1;
constexpr std::array<int64_t, 4> K_TENSOR_X_DIMS = {1, 3, 32, 32};
constexpr std::array<int64_t, 4> K_TENSOR_X_STRIDES = {3072, 1024, 32, 1};

constexpr std::array<int64_t, 2> K_CONV_PADDING = {1, 1};
constexpr std::array<int64_t, 2> K_CONV_STRIDE = {1, 1};
constexpr std::array<int64_t, 2> K_CONV_DILATION = {1, 1};

} // namespace hipdnn_tests::constants
```

Then in the test files, replace inline values:
```cpp
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// Before (generated):
_xDesc = createFinalizedTensor(1, {1, 3, 32, 32}, {3072, 1024, 32, 1});
std::vector<int64_t> prePadding = {1, 1};

// After (with constants):
_xDesc = createFinalizedTensor(K_TENSOR_X_UID, toVec(K_TENSOR_X_DIMS), toVec(K_TENSOR_X_STRIDES));
auto prePadding = toVec(K_CONV_PADDING);
```

The `toVec()` utility converts `std::array` constants to `std::vector` and is located at `test_sdk/include/hipdnn_test_sdk/utilities/ToVec.hpp`.

### When to Keep Values Inline

If the test values are only used in a single test file and are not meaningful beyond that file, it's fine to leave them as inline literals or define them as local constants at the top of the test file.

### Guidelines

- **Convolution ops** (ConvFwd, ConvBwd, ConvWrw) share tensor dimensions and convolution parameters — use shared constants in `test_sdk/constants/`
- **Operation-specific values** (e.g., pointwise mode, batchnorm epsilon) that only appear in one test file — keep inline or define as file-local constants
- **Tensor UIDs** should use named constants when shared across test files (unit tests, graph tests, integration tests for the same operation)
- Constants follow the naming convention `K_<CATEGORY>_<NAME>` (e.g., `K_TENSOR_X_UID`, `K_CONV_PADDING`)

---

## Step 10: Implement the Integration Test

**IMPORTANT**: The generated integration test (`Integration<Op>DescriptorLowering.cpp`) is a **stub** — it has the fixture and setup, but no actual test cases. You MUST implement the full E2E round-trip tests before the work is considered complete. Integration tests should use named constants (see Step 9).

Use `tests/frontend/IntegrationConvFpropDescriptorLowering.cpp` as the reference. Each integration test should:

1. Build a frontend graph using the frontend API (e.g., `graph->conv_fprop(x, w, attrs)`)
2. Call `graph->validate()` and `graph->build_operation_graph_via_descriptors(_handle)` to lower to backend
3. Retrieve the serialized graph via `hipdnnBackendGetSerializedGraph_ext()`
4. Deserialize the FlatBuffer into a `GraphT`
5. Verify all tensor attributes (UIDs, dims, strides, data type, name)
6. Verify the node's operation attributes (tensor UID references, padding, stride, dilation, mode, etc.)

### Required Test Cases

At minimum, implement these two test cases (matching the ConvFprop reference):

**`<Op>GraphRoundTrip`** — Full round-trip with explicit UIDs:
- Create tensors with explicit UIDs and specific dims/strides
- Set all operation parameters (padding, stride, etc.) with non-default values
- Lower to backend, deserialize, and verify every field matches

**`AutoAssignedUidsPreservedInRoundTrip`** — Round-trip with auto-assigned UIDs:
- Create tensors without setting UIDs (let the frontend auto-assign)
- Lower to backend, deserialize
- Verify all tensor UIDs are unique and the node references them correctly

### Integration Test Dependencies

The integration test requires the frontend node type to exist. Specifically:
- The frontend attributes class (e.g., `ConvFpropAttributes`) in `frontend/include/hipdnn_frontend/attributes/`
- The frontend node class (e.g., `ConvolutionFpropNode`) in `frontend/include/hipdnn_frontend/nodes/`
- The graph method (e.g., `graph->conv_fprop()`) in `frontend/include/hipdnn_frontend/Graph.hpp`

If these don't exist yet, the integration test cannot be compiled. In that case:
- Still place the stub file in `tests/frontend/`
- Do NOT add it to `tests/frontend/CMakeLists.txt` until the frontend types exist
- Note in your summary that the integration test is pending frontend implementation

### Example: ConvFprop Integration Test Structure

```cpp
TEST_F(IntegrationConvFpropDescriptorLowering, ConvFpropGraphRoundTrip)
{
    // 1. Build frontend graph
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestConvGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_TENSOR_X_DIMS)).set_stride(toVec(K_TENSOR_X_STRIDES));
    // ... set up w, convAttrs, call graph->conv_fprop(x, w, convAttrs)

    // 2. Lower to backend
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // 3. Retrieve and deserialize
    auto rawDesc = graph->get_raw_graph_descriptor();
    size_t serializedSize = 0;
    hipdnnBackendGetSerializedGraph_ext(rawDesc, &serializedSize, nullptr);
    std::vector<uint8_t> serializedData(serializedSize);
    hipdnnBackendGetSerializedGraph_ext(rawDesc, &serializedSize, serializedData.data());
    auto graphT = GetGraph(serializedData.data())->UnPack();

    // 4. Verify tensors and node attributes
    ASSERT_EQ(graphT->tensors.size(), 3u);
    ASSERT_EQ(graphT->nodes.size(), 1u);
    // ... verify each tensor's uid, dims, strides, data_type
    // ... verify node's operation attributes (tensor UIDs, padding, stride, etc.)
}
```

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
| `field: SomeEnum` | `data_fields` | `enum` (set `cpp_enum` to fully-qualified enum type) |
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

  # Shared attributes — if this operation reuses attributes from another operation
  # (e.g., all conv ops share HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS), use the SAME
  # attr_name values. Do NOT create new per-operation copies.
  has_compute_data_type: true
  compute_data_type_attr: "HIPDNN_ATTR_CONVOLUTION_COMP_TYPE"  # Shared across conv ops

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
| `shared` | bool | `false` | If `true`, the attribute enum already exists (defined by another operation). Fragment templates skip shared fields to avoid duplicate enum entries. Core templates still include them for setAttribute/getAttribute. |
| `test_enum_value` | string | `""` | **Required for enum fields.** The enum constant to use in generated tests (e.g., `CROSS_CORRELATION` for ConvMode, `ADD` for PointwiseMode). |

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

## Notes

- The generated descriptor `.hpp` and `.cpp` files are complete and ready to use as-is
- The packer `.hpp` file is complete and ready to use as-is
- The unit test and graph test files are complete and ready to compile
- **The integration test is a stub** — it must be implemented following the pattern above
- Fragment files contain comments indicating where to insert each snippet
- Enum values (PLACEHOLDER_VALUE) must be replaced with actual numeric values following the existing numbering scheme
