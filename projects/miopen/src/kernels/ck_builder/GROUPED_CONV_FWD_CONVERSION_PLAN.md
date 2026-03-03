# CK Builder Grouped Conv Fwd Instance Conversion Plan

## Context

The MIOpen CK Builder system modernizes kernel instance definitions from ComposableKernel's legacy tuple-of-template-specializations format to constexpr functions returning `std::array` of structs. This enables better compile-time validation using C++20 concepts and improves maintainability.

This plan converts 24 grouped convolution forward kernel instance files from CK to the new MIOpen format.

## Documentation References

- [CK Builder MIOpen Integration README](src/kernels/ck_builder/README.md) - Overview of the migration approach and CK Builder concepts
- [CK Builder Kernel Instance Data README](src/kernels/ck_builder/include/miopen/ck_builder/instance_data/README.md) - Documentation for the instance data structs

### Helper Function Source Files

These files contain the `constexpr` helper functions that map CK template parameters to CK Builder struct fields:

- [xdl.hpp](src/kernels/ck_builder/include/miopen/ck_builder/instance_data/xdl.hpp) - `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle` helper
- [xdl_v3.hpp](src/kernels/ck_builder/include/miopen/ck_builder/instance_data/xdl_v3.hpp) - `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3` helper
- [xdl_large_tensor.hpp](src/kernels/ck_builder/include/miopen/ck_builder/instance_data/xdl_large_tensor.hpp) - `DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor` helper
- [wmma.hpp](src/kernels/ck_builder/include/miopen/ck_builder/instance_data/wmma.hpp) - `DeviceGroupedConvFwdMultipleD_Wmma_CShuffle` helper
- [dl.hpp](src/kernels/ck_builder/include/miopen/ck_builder/instance_data/dl.hpp) - `DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK` helper
- [wmma_v3.hpp](src/kernels/ck_builder/include/miopen/ck_builder/instance_data/wmma_v3.hpp) - `DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3` **stub (unsupported)**
- [wmma_v3_large_tensor.hpp](src/kernels/ck_builder/include/miopen/ck_builder/instance_data/wmma_v3_large_tensor.hpp) - `DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor` **stub (unsupported)**

### Static Tests

Static compile-time validation tests are located in:
- `src/kernels/ck_builder/include/miopen/ck_builder/static_tests/`

## Source and Target Locations

**Source:** `composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/`

**Target:** `src/kernels/ck_builder/include/miopen/ck_builder/factories/grouped_conv_fwd/`

**Helper Functions:** `src/kernels/ck_builder/include/miopen/ck_builder/instance_data/`

## Helper Function Mapping

| Source Template Class | Helper Function | Returns |
| --------------------- | --------------- | ------- |
| `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle` | `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle` (xdl.hpp) | `XdlInstance` |
| `DeviceGroupedConvFwdMultipleD_Wmma_CShuffle` | `DeviceGroupedConvFwdMultipleD_Wmma_CShuffle` (wmma.hpp) | `WmmaInstance` |
| `DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK` | `DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK` (dl.hpp) | `DlInstance` |
| `DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor` | `DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor` (xdl_large_tensor.hpp) | `XdlLargeTensorInstance` |
| `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3` | `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3` (xdl_v3.hpp) | `XdlV3Instance` |
| `DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3` | stub (wmma_v3.hpp) | **Unsupported** |
| `DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor` | stub (wmma_v3_large_tensor.hpp) | **Unsupported** |

> **Note on "_V3" naming vs `pipeline_version` parameter:** The `_V3` suffix in template class names (e.g., `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3`) refers to a different kernel implementation variant and is **not related** to the `pipeline_version` parameter (`ckb::PipelineVersion::V1`, `V2`, etc.). The `_V3` templates have a different parameter signature and algorithm structure compared to non-V3 templates. When converting:
>
> - Use the correct helper function based on the template class name, not the pipeline version value
> - The `pipeline_version` parameter value must be copied exactly from the CK source file being converted—do not assume or invent values

## Files to Convert (24 Total)

### XDL Files (12 files)

1. device_grouped_conv_fwd_xdl_instance.hpp
2. device_grouped_conv_fwd_xdl_bilinear_instance.hpp
3. device_grouped_conv_fwd_xdl_binary_outelementop_instance.hpp
4. device_grouped_conv_fwd_xdl_comp_instance.hpp
5. device_grouped_conv_fwd_xdl_dynamic_op_instance.hpp
6. device_grouped_conv_fwd_xdl_large_tensor_instance.hpp
7. device_grouped_conv_fwd_xdl_mem_instance.hpp
8. device_grouped_conv_fwd_xdl_merged_groups_instance.hpp
9. device_grouped_conv_fwd_xdl_outelementop_instance.hpp
10. device_grouped_conv_fwd_xdl_scale_instance.hpp
11. device_grouped_conv_fwd_xdl_scaleadd_ab_instance.hpp
12. device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_instance.hpp

### WMMA Files (11 files)

1. device_grouped_conv_fwd_wmma_instance.hpp
2. device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp
3. device_grouped_conv_fwd_wmma_cshufflev3_bilinear_instance.hpp
4. device_grouped_conv_fwd_wmma_cshufflev3_binary_outelementop_instance.hpp
5. device_grouped_conv_fwd_wmma_cshufflev3_dynamic_op_instance.hpp
6. device_grouped_conv_fwd_wmma_cshufflev3_large_tensor_instance.hpp
7. device_grouped_conv_fwd_wmma_cshufflev3_outelementop_instance.hpp
8. device_grouped_conv_fwd_wmma_cshufflev3_scale_instance.hpp
9. device_grouped_conv_fwd_wmma_cshufflev3_scaleadd_ab_instance.hpp
10. device_grouped_conv_fwd_wmma_cshufflev3_scaleadd_scaleadd_relu_instance.hpp
11. device_grouped_conv_fwd_wmma_cshufflev3_wave_transfer_instance.hpp

### DL Files (1 file)

1. device_grouped_conv_fwd_dl_instance.hpp

## Conversion Pattern

### Original Format (CK)

```cpp
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_bf16_instances = std::tuple<
    // clang-format off
    DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,DsLayout,ELayout,BF16,BF16,F32,BF16,DsDataTypes,BF16,PassThrough,PassThrough,OutElementOp,ConvSpec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,S<4,16,1>,S<1,0,2>,S<1,0,2>,2,1,8,1,S<4,16,1>,S<1,0,2>,S<1,0,2>,2,1,8,1,1,1,S<1,16,1,4>,1>,
    DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,DsLayout,ELayout,BF16,BF16,F32,BF16,DsDataTypes,BF16,PassThrough,PassThrough,OutElementOp,ConvSpec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,S<4,16,1>,S<1,0,2>,S<1,0,2>,2,8,8,1,S<4,16,1>,S<1,0,2>,S<1,0,2>,2,8,8,1,1,1,S<1,16,1,4>,1>
    // clang-format on
>;
```

### Converted Format (CK Builder) - ONE LINE PER INSTANCE

**IMPORTANT:** All template parameters from the original `using` statement must become function parameters in the converted `constexpr` function. If a template parameter has a default value, propagate that default to the function parameter. This includes:

- `NDimSpatial` → `spatial_dim`
- `ALayout` → `input_layout`
- `BLayout` → `weight_layout`
- `DsLayout` → `ds_layouts` (array)
- `ELayout` → `output_layout`
- `ConvSpec` → `conv_spec`
- `DsDataTypes = Tuple<>` → `ds_types = {}` (empty array default)
- `OutElementOp = PassThrough` → `output_op = PassThrough` (default value preserved)

Use `using` statements or `constexpr` constants to reduce boilerplate in each function call:

```cpp
// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <miopen/ck_builder/instance_data/xdl.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_fwd {

using namespace instance;

// Type aliases to match original CK naming
using BF16 = ckb::DataType::BF16;
using F16  = ckb::DataType::F16;
using F32  = ckb::DataType::F32;

// Layout aliases
using GNHWC = ckb::TensorLayout::GNHWC;
using GKYXC = ckb::TensorLayout::GKYXC;
using GNHWK = ckb::TensorLayout::GNHWK;
// ... more layouts as needed

// Operation aliases
constexpr auto PassThrough = ckb::ElementwiseOperation::PASS_THROUGH;
constexpr auto Scale       = ckb::ElementwiseOperation::SCALE;

// Specialization aliases
constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;
constexpr auto ConvFwd1x1P0   = ckb::ConvSpecialization::FILTER_1X1_PAD_0;
constexpr auto GemmMNKPadding = ckb::GemmSpecialization::MNK_PADDING;

// All template parameters become function parameters (with defaults where applicable)
template <std::size_t NumDTensor = 0>
constexpr auto device_grouped_conv_fwd_xdl_bf16_instances(
    std::size_t spatial_dim,
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    ckb::ConvSpecialization conv_spec,
    const std::array<ckb::DataType, NumDTensor>& ds_types = {},
    ckb::ElementwiseOperation output_op = ckb::ElementwiseOperation::PASS_THROUGH)
{
    return std::array{
        // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,BF16,BF16),
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,BF16,BF16)
        // clang-format on
    };
}

} // namespace grouped_conv_fwd
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
```

**Key formatting requirements:**

1. **All template parameters must become function parameters** - the function signature must include parameters for all template arguments from the original `using` statement
2. Each helper function call must be on a single line with `// clang-format off/on` guards
3. Use `using` statements for data types and layouts to match original CK naming
4. Use `constexpr` constants for operations and specializations
5. Use inline array initializers for sequences (e.g., `{4,16,1}` instead of named constants)

**Default parameter values:**

The helper functions preserve the same default values as the original CK templates for trailing parameters. When converting, if the CK source omits trailing template parameters (relying on defaults), the converted helper function call can also omit those parameters. For example:

- `loop_scheduler` defaults to `ckb::PipelineScheduler::DEFAULT`
- `num_conv_groups_to_merge` defaults to `1`
- `pipeline_version` (V3 only) defaults vary by helper

If a converted call appears to have fewer parameters than the helper function signature, check whether the "missing" parameters have default values that match the intended behavior. Do not add explicit values for parameters where the default is correct—this keeps the converted code concise and matches the original CK source's intent.

**Type conversions:**

When converting from CK template parameters to helper function arguments, note these type differences:

| CK Source | Helper Function | Conversion |
|-----------|-----------------|------------|
| `S<4, 16, 1>` | `std::array<std::size_t, 3>` | `{4, 16, 1}` |
| `1` (for bool params like `ABlockLdsAddExtraM`) | `bool` | `true` |
| `0` (for bool params) | `bool` | `false` |
| `ck::half_t` / `F16` | `ckb::DataType` | `ckb::DataType::F16` (or alias `F16`) |
| `ck::bhalf_t` / `BF16` | `ckb::DataType` | `ckb::DataType::BF16` (or alias `BF16`) |
| `PassThrough` | `ckb::ElementwiseOperation` | `ckb::ElementwiseOperation::PASS_THROUGH` |

The boolean conversion is particularly important: CK templates use integer `1`/`0` for parameters like `ABlockLdsAddExtraM` and `BBlockLdsAddExtraN`, but the helper functions expect `bool` (`true`/`false`). Always convert these explicitly.

**NumDTensor template parameter:**

The helper functions are templated on `NumDTensor`, which specifies the number of D-tensors (auxiliary tensors). This value **must be explicitly specified** as a template argument—the compiler cannot deduce it from brace-init-lists.

To determine `NumDTensor` from the CK source:
- Count the types in `DsDataTypes = Tuple<...>` (e.g., `Tuple<>` → 0, `Tuple<F32>` → 1, `Tuple<F32, F32>` → 2)
- The `DsLayout` template parameter will have the same count

Examples:
```cpp
// NumDTensor = 0 (empty Tuple<>)
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<0>(..., {}, ..., {}, ...)

// NumDTensor = 1 (Tuple<F32> for DsDataTypes)
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<1>(..., {GNHWK}, ..., {F32}, ...)

// NumDTensor = 2
DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<2>(..., {GNHWK, GNHWK}, ..., {F32, F32}, ...)
```

The `ds_layouts` and `ds_types` arrays must both have exactly `NumDTensor` elements.

## Type Aliases

Define these at the top of each file to reduce boilerplate:

### Data Types

```cpp
using BF16 = ckb::DataType::BF16;
using F16  = ckb::DataType::F16;
using F32  = ckb::DataType::F32;
using TF32 = ckb::DataType::TF32;
using F8   = ckb::DataType::F8;
using BF8  = ckb::DataType::BF8;
```

### Layouts

```cpp
using GNHWC = ckb::TensorLayout::GNHWC;
using GKYXC = ckb::TensorLayout::GKYXC;
using GNHWK = ckb::TensorLayout::GNHWK;
// ... add others as needed per file
```

### Operations

```cpp
constexpr auto PassThrough = ckb::ElementwiseOperation::PASS_THROUGH;
constexpr auto Scale       = ckb::ElementwiseOperation::SCALE;
constexpr auto AddClamp    = ckb::ElementwiseOperation::ADD_CLAMP;
// ... add others as needed per file
```

### Specializations

```cpp
constexpr auto ConvFwdDefault  = ckb::ConvSpecialization::DEFAULT;
constexpr auto ConvFwd1x1P0    = ckb::ConvSpecialization::FILTER_1X1_PAD_0;
constexpr auto ConvFwd1x1S1P0  = ckb::ConvSpecialization::FILTER_1X1_STRIDE_1_PAD_0;
constexpr auto GemmMNKPadding  = ckb::GemmSpecialization::MNK_PADDING;
```

## Implementation Steps

1. **Create target directory**

   ```bash
   mkdir -p src/kernels/ck_builder/include/miopen/ck_builder/factories/grouped_conv_fwd
   ```

2. **For each source file, convert in TWO STAGES:**

   Due to the large size of these files (often containing hundreds of kernel instances), each file should be converted in two passes. This allows for incremental progress and easier review. The files will not compile until the conversion is complete. These two passes should be done as separate steps or you will run into token output limits and fail. Each function in Stage 2 should also be done as a separate step.

   ### Stage 1: Create file structure with stub functions

   - Read the source file
   - Collect all unique data types, layouts, operations, and specializations used
   - Create the target file with:
     - SPDX license and copyright header
     - Required `#include` statements
     - Namespace declarations
     - Type aliases (`using` statements) and `constexpr` constants at file top
     - Preserve `#ifdef CK_ENABLE_FP8` / `#ifdef CK_ENABLE_BF8` guards
   - For each `using` statement (template alias) in the source file:
     - Create a corresponding `constexpr` function with the same name
     - Stub the function body to return an empty array of the correct instance type (e.g., `return std::array<XdlInstance, 0>{};`)
     - This is valid C++ and allows compilation verification during Stage 1

   **Example Stage 1 output:**

   ```cpp
   // SPDX-License-Identifier: MIT
   // Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

   #pragma once

   #include <miopen/ck_builder/instance_data/xdl.hpp>
   #include <array>

   namespace miopen {
   namespace conv {
   namespace ck_builder {
   namespace factories {
   namespace grouped_conv_fwd {

   using namespace instance;

   // Type aliases
   using BF16 = ckb::DataType::BF16;
   using F16  = ckb::DataType::F16;
   using F32  = ckb::DataType::F32;

   // Layout aliases
   using GNHWC = ckb::TensorLayout::GNHWC;
   using GKYXC = ckb::TensorLayout::GKYXC;
   using GNHWK = ckb::TensorLayout::GNHWK;

   // Operation aliases
   constexpr auto PassThrough = ckb::ElementwiseOperation::PASS_THROUGH;

   // Specialization aliases
   constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;
   constexpr auto GemmMNKPadding = ckb::GemmSpecialization::MNK_PADDING;

   // All original template parameters become function parameters (with defaults where applicable)
   template <std::size_t NumDTensor = 0>
   constexpr auto device_grouped_conv_fwd_xdl_bf16_instances(
       std::size_t spatial_dim,
       ckb::TensorLayout input_layout,
       ckb::TensorLayout weight_layout,
       const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
       ckb::TensorLayout output_layout,
       ckb::ConvSpecialization conv_spec,
       const std::array<ckb::DataType, NumDTensor>& ds_types = {},
       ckb::ElementwiseOperation output_op = ckb::ElementwiseOperation::PASS_THROUGH)
   {
       // TODO: Stage 2 - Convert template instantiations
       return std::array<XdlInstance, 0>{};
   }

   template <std::size_t NumDTensor = 0>
   constexpr auto device_grouped_conv_fwd_xdl_bf16_generic_instances(
       std::size_t spatial_dim,
       ckb::TensorLayout input_layout,
       ckb::TensorLayout weight_layout,
       const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
       ckb::TensorLayout output_layout,
       ckb::ConvSpecialization conv_spec,
       const std::array<ckb::DataType, NumDTensor>& ds_types = {},
       ckb::ElementwiseOperation output_op = ckb::ElementwiseOperation::PASS_THROUGH)
   {
       // TODO: Stage 2 - Convert template instantiations
       return std::array<XdlInstance, 0>{};
   }

   // ... more stub functions

   } // namespace grouped_conv_fwd
   } // namespace factories
   } // namespace ck_builder
   } // namespace conv
   } // namespace miopen
   ```

   ### Stage 2: Fill in the converted template instantiations

   - For each stubbed function, replace `return std::array{ };` with the actual converted instances
   - Use inline array initializers for sequences (e.g., `{4,16,1}`)
   - Map all template parameters to function parameters using the short aliases
   - Keep each instance call on ONE LINE with clang-format guards
   - Identify helper function to use (xdl, wmma, dl, xdl_v3, xdl_large_tensor)

   **Example Stage 2 update (for one function):**

   ```cpp
   // All original template parameters become function parameters (with defaults where applicable)
   template <std::size_t NumDTensor = 0>
   constexpr auto device_grouped_conv_fwd_xdl_bf16_instances(
       std::size_t spatial_dim,
       ckb::TensorLayout input_layout,
       ckb::TensorLayout weight_layout,
       const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
       ckb::TensorLayout output_layout,
       ckb::ConvSpecialization conv_spec,
       const std::array<ckb::DataType, NumDTensor>& ds_types = {},
       ckb::ElementwiseOperation output_op = ckb::ElementwiseOperation::PASS_THROUGH)
   {
       return std::array{
           // clang-format off
           DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,64,32,8,8,32,32,2,2,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,{4,16,1},{1,0,2},{1,0,2},2,1,8,true,1,1,{1,16,1,4},1,BF16,BF16),
           DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NumDTensor>(spatial_dim,input_layout,weight_layout,ds_layouts,output_layout,BF16,BF16,F32,BF16,ds_types,BF16,PassThrough,PassThrough,output_op,conv_spec,GemmMNKPadding,1,64,64,32,32,8,8,32,32,2,1,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,{4,16,1},{1,0,2},{1,0,2},2,8,8,true,1,1,{1,16,1,4},1,BF16,BF16)
           // clang-format on
       };
   }
   ```

3. **Handle special cases:**
   - `#ifdef CK_ENABLE_FP8` / `#ifdef CK_ENABLE_BF8` guards preserved
   - NumDTensor template parameter for D-tensor arrays
   - V3 variants need pipeline_version and direct_load parameters

## Verification

1. All converted files compile without errors
2. Function names match original `using` statement names exactly
3. Parameter counts match helper function signatures
4. All aliases are correctly defined and used
5. One-line format with clang-format guards is maintained

### Agent Verification

After each file is converted, an agent should verify the conversion by comparing the original CK source file against the converted MIOpen ck_builder file. The verification agent must:

1. Read this conversion plan document in full
2. Read all referenced documentation (README files, helper function headers)
3. Read the original CK source file being verified
4. Read the converted MIOpen ck_builder file
5. Verify:
   - All template aliases were converted to constexpr functions
   - Instance counts match between original and converted
   - Data type mappings are correct (BF16, F16, F32, TF32, int8_t→I8, int32_t→I32, etc.)
   - All numeric parameters were copied correctly
   - S<a,b,c> sequences were converted to {a,b,c} initializer lists
   - Boolean values (1 → true) are correctly converted
   - Any conditional compilation guards (#ifdef) are correctly placed inside functions with #else returning empty arrays
6. Report any discrepancies found

## Conversion Checklist

### XDL Files

| File | Audited | Templates Used | Converted | Agent Verified | Tested |
|------|---------|----------------|-----------|----------------|--------|
| device_grouped_conv_fwd_xdl_instance.hpp | [x] | Xdl | [x] | [x] | [ ] |
| device_grouped_conv_fwd_xdl_bilinear_instance.hpp | [x] | Xdl | [x] | [x] | [ ] |
| device_grouped_conv_fwd_xdl_binary_outelementop_instance.hpp | [x] | Xdl | [x] | [x] | [ ] |
| device_grouped_conv_fwd_xdl_comp_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_xdl_dynamic_op_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_xdl_large_tensor_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_xdl_mem_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_xdl_merged_groups_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_xdl_outelementop_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_xdl_scale_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_xdl_scaleadd_ab_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_xdl_scaleadd_scaleadd_relu_instance.hpp | [ ] | | [ ] | [ ] | [ ] |

### WMMA Files

| File | Audited | Templates Used | Converted | Agent Verified | Tested |
|------|---------|----------------|-----------|----------------|--------|
| device_grouped_conv_fwd_wmma_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_wmma_cshufflev3_bilinear_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_wmma_cshufflev3_binary_outelementop_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_wmma_cshufflev3_dynamic_op_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_wmma_cshufflev3_large_tensor_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_wmma_cshufflev3_outelementop_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_wmma_cshufflev3_scale_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_wmma_cshufflev3_scaleadd_ab_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_wmma_cshufflev3_scaleadd_scaleadd_relu_instance.hpp | [ ] | | [ ] | [ ] | [ ] |
| device_grouped_conv_fwd_wmma_cshufflev3_wave_transfer_instance.hpp | [ ] | | [ ] | [ ] | [ ] |

### DL Files

| File | Audited | Templates Used | Converted | Agent Verified | Tested |
|------|---------|----------------|-----------|----------------|--------|
| device_grouped_conv_fwd_dl_instance.hpp | [ ] | | [ ] | [ ] | [ ] |

### Checklist Legend

- **Audited**: Source file reviewed to identify all template classes used
- **Templates Used**: List template classes found (e.g., `Xdl`, `XdlV3`, `Wmma`, `WmmaV3`, `XdlLargeTensor`, `Dl`)
- **Converted**: Factory file fully converted (both Stage 1 and Stage 2 complete)
- **Agent Verified**: Conversion verified by an agent following the Agent Verification process (see above)
- **Tested**: Static tests added and compilation verified

### Stage Tracking (Optional Detail)

For files being actively worked on, you may track intermediate progress:

- **Stage 1**: File structure with stub functions (`return std::array<InstanceType, 0>{};`) created and compiles successfully
- **Stage 2**: All template instantiations converted to helper function calls

### Per-File Conversion Steps

For each file, follow the two-stage approach:

#### Stage 1: Stub Creation

1. [ ] Read source file and count all template instantiations
2. [ ] Identify all template classes used (may be multiple per file)
3. [ ] List unique data types, layouts, operations, specializations
4. [ ] Identify all template parameters from each `using` statement (e.g., `NDimSpatial`, `ALayout`, `BLayout`, `DsLayout`, `ELayout`, `ConvSpec`, `DsDataTypes`, `OutElementOp`) and note any default values (e.g., `DsDataTypes = Tuple<>`, `OutElementOp = PassThrough`)
5. [ ] Create factory file with appropriate includes and namespace structure
6. [ ] Define type/layout/operation aliases at file top
7. [ ] Create separate factory function for each template alias used
8. [ ] **IMPORTANT:** All template parameters must become function parameters, preserving default values where applicable (see examples above)
9. [ ] Stub each function body with `return std::array<InstanceType, 0>{};` (e.g., `std::array<XdlInstance, 0>{}`)
10. [ ] Compile the header to verify valid C++ syntax and correct includes

#### Stage 2: Template Conversion

1. [ ] For each stubbed function, convert the template instantiations
2. [ ] Replace `return std::array{ };` with actual converted instances
3. [ ] Use the function parameters (e.g., `input_layout`, `output_op`) in helper function calls instead of hardcoded values
4. [ ] Verify parameter count matches helper signature
5. [ ] Keep each instance call on ONE LINE with clang-format guards

#### Final Verification

1. [ ] Add static test for new factory
2. [ ] Compile and verify no errors
