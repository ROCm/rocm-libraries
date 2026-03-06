# CK to MIOpen CK Builder: .cpp File Conversion Plan

## Overview

This plan converts 96 `.cpp` files from ComposableKernel's grouped_conv2d_fwd tensor operation instances to use MIOpen's CK Builder framework. These files call `add_device_operation_instances` with tuples of template specializations that have been converted to constexpr factory functions (as documented in `GROUPED_CONV_FWD_CONVERSION_PLAN.md`).

## Background

### Original Pattern (ComposableKernel .cpp files)

Each `.cpp` file contains one or more functions that populate a vector with kernel instances:

```cpp
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    // Each call passes a tuple template alias (device_grouped_conv_fwd_xdl_f16_instances)
    // with template parameters that specify the kernel configuration
    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_f16_instances<2,
                                                                             NHWGC,
                                                                             GKYXC,
                                                                             Empty_Tuple,
                                                                             NHWGK,
                                                                             ConvFwdDefault>{});

    add_device_operation_instances(instances,
                                   device_grouped_conv_fwd_xdl_f16_instances<2,
                                                                             NHWGC,
                                                                             GKYXC,
                                                                             Empty_Tuple,
                                                                             NHWGK,
                                                                             ConvFwd1x1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
```

### Converted Pattern (MIOpen .cpp files)

The converted files use the new `add_device_operation_instances` function that takes a constexpr array:

```cpp
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD>>& instances)
{
    namespace ckb = ck_tile::builder;
    using namespace factories::grouped_conv_fwd;

    // Layout aliases
    constexpr auto NHWGC = ckb::TensorLayout::NHWGC;
    constexpr auto GKYXC = ckb::TensorLayout::GKYXC;
    constexpr auto NHWGK = ckb::TensorLayout::NHWGK;

    // Specialization aliases
    constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;
    constexpr auto ConvFwd1x1P0   = ckb::ConvSpecialization::FILTER_1X1_PAD_0;

    // Each call passes the result of a constexpr factory function as a template parameter
    // The factory function returns std::array of kernel descriptors
    add_device_operation_instances<
        device_grouped_conv_fwd_xdl_f16_instances<0>(
            2,              // spatial_dim
            NHWGC,          // input_layout
            GKYXC,          // weight_layout
            {},             // ds_layouts (empty for NumDTensor=0)
            NHWGK,          // output_layout
            ConvFwdDefault) // conv_spec
    >(instances);

    add_device_operation_instances<
        device_grouped_conv_fwd_xdl_f16_instances<0>(
            2,
            NHWGC,
            GKYXC,
            {},
            NHWGK,
            ConvFwd1x1P0)
    >(instances);
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
```

## Source and Target Locations

**Source Directory:** `/home/AMD/dahawkin/work/ck_study/rocm-libraries/projects/composablekernel/library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/`

**Target Directory:** `/home/AMD/dahawkin/work/ck_study/rocm-libraries/projects/miopen/src/kernels/ck_builder/factories/grouped_conv2d_fwd/`

**Directory Structure:** The subdirectory structure must be preserved during conversion:
- `dl/` → `dl/`
- `wmma/` → `wmma/`
- `wmma/large_tensor/` → `wmma/large_tensor/`
- `xdl/` → `xdl/`
- `xdl/comp/` → `xdl/comp/`
- `xdl/large_tensor/` → `xdl/large_tensor/`
- `xdl/mem/` → `xdl/mem/`
- `xdl/merged_groups/` → `xdl/merged_groups/`

## Factory Function Mapping

The tuple template aliases in the original .cpp files map to constexpr factory functions that have already been converted (see `GROUPED_CONV_FWD_CONVERSION_PLAN.md`):

| Original Include | New Include |
|------------------|-------------|
| `device_grouped_conv_fwd_xdl_instance.hpp` | `device_grouped_conv_fwd_xdl_instance.hpp` |
| `device_grouped_conv_fwd_wmma_instance.hpp` | `device_grouped_conv_fwd_wmma_instance.hpp` |
| `device_grouped_conv_fwd_dl_instance.hpp` | `device_grouped_conv_fwd_dl_instance.hpp` |
| `device_grouped_conv_fwd_xdl_large_tensor_instance.hpp` | `device_grouped_conv_fwd_xdl_large_tensor_instance.hpp` |
| `device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp` | `device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp` |
| And others... | Corresponding factory files |

The factory functions in these headers have the same names as the tuple template aliases but are now `constexpr` functions that return `std::array` of kernel descriptors.

## Template Parameter Conversion Examples

These examples show the most common conversion patterns. See "Template Signatures by Factory Type" above for complete details on parameter differences between XDL, WMMA, and DL.

### XDL/WMMA Conversion Example

**Original CK Pattern:**
```cpp
add_device_operation_instances(instances,
                               device_grouped_conv_fwd_xdl_f16_instances<
                                   2,              // NDimSpatial
                                   NHWGC,          // ALayout
                                   GKYXC,          // BLayout
                                   Empty_Tuple,    // DsLayout
                                   NHWGK,          // ELayout
                                   ConvFwdDefault, // ConvSpec
                                   Empty_Tuple,    // DsDataTypes (often omitted)
                                   PassThrough     // OutElementOp (often omitted)
                               >{});
```

**Converted MIOpen Pattern:**
```cpp
// Layout and specialization aliases defined earlier in function
add_device_operation_instances<
    device_grouped_conv_fwd_xdl_f16_instances<0>(  // NumDTensor template param
        2,              // spatial_dim
        NHWGC,          // input_layout (using alias)
        GKYXC,          // weight_layout (using alias)
        {},             // ds_layouts (empty array for NumDTensor=0)
        NHWGK,          // output_layout (using alias)
        ConvFwdDefault) // conv_spec (using alias)
        // ds_types and output_op omitted (use defaults)
>(instances);
```

### DL Conversion Example (NO spatial_dim parameter!)

**Original CK Pattern:**
```cpp
add_device_operation_instances(instances,
                               device_grouped_conv2d_fwd_dl_f16_instances<
                                   GNHWC,          // InLayout (NO NDimSpatial!)
                                   GKYXC,          // WeiLayout
                                   Empty_Tuple,    // DsLayout
                                   GNHWK,          // OutLayout
                                   Empty_Tuple,    // DsDatatype
                                   PassThrough,    // CDEElementOp
                                   ConvFwdDefault  // ConvSpec
                               >{});
```

**Converted MIOpen Pattern:**
```cpp
// Layout and specialization aliases defined earlier in function
add_device_operation_instances<
    device_grouped_conv2d_fwd_dl_f16_instances<0>(  // NumDTensor template param
        GNHWC,          // input_layout (NO spatial_dim for DL!)
        GKYXC,          // weight_layout
        {},             // ds_layouts (empty array)
        GNHWK,          // output_layout
        {},             // ds_types (empty array)
        PassThrough,    // cde_element_op
        ConvFwdDefault) // conv_spec
>(instances);
```

### Parameter Mapping by Factory Type

#### XDL and WMMA Factory Parameter Mapping

| CK Template Parameter | Factory Function Parameter | Notes |
|-----------------------|----------------------------|-------|
| `NDimSpatial` (e.g., `2`) | `spatial_dim` | First function argument for XDL/WMMA |
| `ALayout` (e.g., `NHWGC`) | `input_layout` | Use constexpr alias (e.g., `NHWGC`) |
| `BLayout` (e.g., `GKYXC`) | `weight_layout` | Use constexpr alias (e.g., `GKYXC`) |
| `DsLayout` (e.g., `Empty_Tuple`) | `ds_layouts` | Becomes `{}` for empty, or array like `{NHWGK}` |
| `ELayout` (e.g., `NHWGK`) | `output_layout` | Use constexpr alias (e.g., `NHWGK`) |
| `ConvSpec` (e.g., `ConvFwdDefault`) | `conv_spec` | Use constexpr alias (e.g., `ConvFwdDefault`) |
| `DsDataTypes` | `ds_types` | Optional parameter, defaults to `{}` |
| `OutElementOp` | `output_op` | Optional parameter, defaults to `PASS_THROUGH` |

#### DL Factory Parameter Mapping (7 parameters - NO NDimSpatial)

| CK Template Parameter | Factory Function Parameter | Notes |
|-----------------------|----------------------------|-------|
| `InLayout` | `input_layout` | Use constexpr alias - NO spatial_dim parameter! |
| `WeiLayout` | `weight_layout` | Use constexpr alias |
| `DsLayout` | `ds_layouts` | Becomes `{}` for empty, or array like `{GNHWK}` |
| `OutLayout` | `output_layout` | Use constexpr alias |
| `DsDatatype` | `ds_types` | Becomes `{}` for empty, or array like `{F16}` |
| `CDEElementOp` | `cde_element_op` | Different from XDL's separate ops |
| `ConvSpec` | `conv_spec` | Use constexpr alias |

### NumDTensor Template Parameter

The `NumDTensor` template parameter must be determined from the `DsLayout` and `DsDataTypes`:

**For XDL and WMMA:**
- `Empty_Tuple` in DsLayout → `NumDTensor = 0`, `ds_layouts = {}`, `ds_types = {}`
- `Tuple<Layout1>` in DsLayout → `NumDTensor = 1`, `ds_layouts = {NHWGK}`, `ds_types = {F32}`
- `Tuple<Layout1, Layout2>` in DsLayout → `NumDTensor = 2`, `ds_layouts = {NHWGK, NHWGK}`, `ds_types = {F32, F32}`

**For DL:**
- Same logic applies, but both `DsLayout` AND `DsDatatype` template parameters must be converted to arrays
- Example: `Empty_Tuple, Empty_Tuple` → `NumDTensor = 0`, `ds_layouts = {}`, `ds_types = {}`
- Example: `Tuple<GNHWK>, Tuple<F16>` → `NumDTensor = 1`, `ds_layouts = {GNHWK}`, `ds_types = {F16}`

### Layout and Specialization Aliases

To improve readability and match the original CK style, define `constexpr auto` aliases at the start of each function for the layouts and specializations used in that file:

```cpp
// Layout aliases (define only those used in the file)
constexpr auto GNHWC = ckb::TensorLayout::GNHWC;
constexpr auto NHWGC = ckb::TensorLayout::NHWGC;
constexpr auto GKYXC = ckb::TensorLayout::GKYXC;
constexpr auto GNHWK = ckb::TensorLayout::GNHWK;
constexpr auto NHWGK = ckb::TensorLayout::NHWGK;
constexpr auto NGCHW = ckb::TensorLayout::NGCHW;
constexpr auto GKCYX = ckb::TensorLayout::GKCYX;
constexpr auto NGKHW = ckb::TensorLayout::NGKHW;

// Specialization aliases (define only those used in the file)
constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;
constexpr auto ConvFwd1x1P0   = ckb::ConvSpecialization::FILTER_1X1_PAD_0;
constexpr auto ConvFwd1x1S1P0 = ckb::ConvSpecialization::FILTER_1X1_STRIDE_1_PAD_0;
constexpr auto ConvFwdOddC    = ckb::ConvSpecialization::ODD_C;
```

**Important:** Only define the aliases that are actually used in the file to avoid namespace pollution.

## Function Signature Conversion

### Original CK Signature

```cpp
void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
```

### Converted MIOpen Signature

```cpp
void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD>>& instances)
```

**Key Change:** The template parameters are removed from `DeviceGroupedConvFwdMultipleABD` in the MIOpen version because the builder system handles type information internally.

## Conversion Steps for Each File

### 1. Update Copyright and Includes

**Original:**
```cpp
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp"
```

**Converted:**
```cpp
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/kernel_instantiation.hpp>
#include <miopen/ck_builder/factories/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp>
```

### 2. Update Namespace

**Original:**
```cpp
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**Converted:**
```cpp
namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
```

### 3. Simplify Function Signature

Remove all template parameters from `DeviceGroupedConvFwdMultipleABD` in the function signature.

### 4. Add Namespace Aliases and Constant Definitions (at function start)

```cpp
namespace ckb = ck_tile::builder;
using namespace factories::grouped_conv_fwd;
// NOTE: The above 'using namespace' statement imports data type aliases (F16, BF16, F32, etc.)
// from the factory header, so you don't need to redefine them here.

// Layout aliases (only define those used in this file)
constexpr auto NHWGC = ckb::TensorLayout::NHWGC;
constexpr auto GKYXC = ckb::TensorLayout::GKYXC;
constexpr auto NHWGK = ckb::TensorLayout::NHWGK;

// Specialization aliases (only define those used in this file)
constexpr auto ConvFwdDefault = ckb::ConvSpecialization::DEFAULT;
constexpr auto ConvFwd1x1P0   = ckb::ConvSpecialization::FILTER_1X1_PAD_0;
constexpr auto ConvFwd1x1S1P0 = ckb::ConvSpecialization::FILTER_1X1_STRIDE_1_PAD_0;
constexpr auto ConvFwdOddC    = ckb::ConvSpecialization::ODD_C;
```

### 5. Convert Each add_device_operation_instances Call

For each call in the original function:

1. **Identify the factory type** (XDL, WMMA, or DL) from the file name and include
2. **Identify the tuple template alias name** (e.g., `device_grouped_conv_fwd_xdl_f16_instances`)
3. **Count and identify all template parameters** (varies by factory type: XDL/WMMA=8, DL=7)
4. **Determine `NumDTensor`** from `DsLayout`/`DsDataTypes` (count tuple elements)
5. **Convert template parameters to function arguments** using the appropriate mapping table above
6. **Call the new `add_device_operation_instances<...>(instances)`** with constexpr factory function

**Example Conversion:**

Original:
```cpp
add_device_operation_instances(instances,
                               device_grouped_conv_fwd_xdl_f16_instances<2,
                                                                         NHWGC,
                                                                         GKYXC,
                                                                         Empty_Tuple,
                                                                         NHWGK,
                                                                         ConvFwdDefault>{});
```

Converted:
```cpp
add_device_operation_instances<
    device_grouped_conv_fwd_xdl_f16_instances<0>(
        2,
        NHWGC,
        GKYXC,
        {},
        NHWGK,
        ConvFwdDefault)
>(instances);
```

Note: The layout and specialization aliases (NHWGC, GKYXC, etc.) are defined earlier in the function body.

### 6. Handle Optional Parameters

Some factory functions have optional parameters with defaults:
- `ds_types` - defaults to `{}`
- `output_op` - defaults to `ckb::ElementwiseOperation::PASS_THROUGH`

Only specify these if they differ from the default.

## File Organization

### Total Files to Convert: 96

#### DL Files (4 files)
```
dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
```

#### WMMA Files (28 files)
```
wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part1.cpp
wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part2.cpp
wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part3.cpp
wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part4.cpp
wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part1.cpp
wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part2.cpp
wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part3.cpp
wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part4.cpp
wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_wave_transfer_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_wave_transfer_nhwgc_gkyxc_nhwgk_f16_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1p0_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1s1p0_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_oddc_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1p0_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1s1p0_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_oddc_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_1x1p0_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_1x1s1p0_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_oddc_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1p0_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1s1p0_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_instance.cpp
wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_oddc_instance.cpp
wmma/large_tensor/device_grouped_conv2d_fwd_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
wmma/large_tensor/device_grouped_conv2d_fwd_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_f16_instance.cpp
```

#### XDL Files (64 files)
```
xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_comp_2x_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_comp_part2_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_2x_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_part2_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_comp_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_2x_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_part2_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_2x_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_part2_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_comp_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_comp_instance.cpp
xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_comp_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_bf16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_16x16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_16x16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_16x16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_bf16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f32_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_int8_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_16x16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_16x16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_16x16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_16x16_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_instance.cpp
xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f16_instance.cpp
xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_instance.cpp
xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_int8_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_mem_inter_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_mem_intra_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_inter_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_intra_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_inter_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_intra_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_direct_load_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_inter_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_intra_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_direct_load_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_mem_inter_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_mem_intra_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_mem_inter_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_mem_intra_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_mem_inter_instance.cpp
xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_mem_intra_instance.cpp
xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_bf16_instance.cpp
xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f16_instance.cpp
xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f32_instance.cpp
xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f16_instance.cpp
xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_instance.cpp
xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_int8_instance.cpp
```

## Conversion Process

### Prerequisites

Before starting conversions, ensure all factory header files (`.hpp`) have been converted as documented in `GROUPED_CONV_FWD_CONVERSION_PLAN.md`. The .cpp files depend on these factory functions.

### Recommended Approach

Convert files in batches by subdirectory:

1. **DL files (4 files)** - Simplest, good starting point
2. **WMMA base files (18 files)** - Moderate complexity
3. **WMMA V3 files (8 files)** - Note: V3 templates are unsupported, these will need special handling
4. **WMMA large tensor files (2 files)** - Note: V3 large tensor is unsupported
5. **XDL base files (19 files)** - Most common pattern
6. **XDL comp files (15 files)** - Compression-specific
7. **XDL large tensor files (5 files)** - Large tensor support
8. **XDL mem files (16 files)** - Memory scheduling variants
9. **XDL merged groups files (8 files)** - Merged group optimization

### Per-File Checklist

For each `.cpp` file:

- [ ] **Read the source file** from ComposableKernel
- [ ] **Create target subdirectory** if it doesn't exist (e.g., `mkdir -p factories/grouped_conv2d_fwd/dl/`)
- [ ] **Identify the factory type** (XDL, WMMA, or DL) from the include path and file name
- [ ] **Identify the subdirectory** (dl, wmma, wmma/large_tensor, xdl, xdl/comp, xdl/large_tensor, xdl/mem, xdl/merged_groups)
- [ ] **Identify the factory function(s)** being called (tuple template aliases)
- [ ] **Identify all `add_device_operation_instances` calls**
- [ ] **For each call:**
  - [ ] Identify template parameter count (8 for XDL/WMMA, 7 for DL)
  - [ ] Extract all template parameters
  - [ ] Determine `NumDTensor` from `DsLayout` (count tuple elements)
  - [ ] Identify which layouts are used (for constexpr aliases)
  - [ ] Identify which conv specs are used (for constexpr aliases)
  - [ ] Check if `DsDataTypes` or `OutElementOp`/`CDEElementOp` differ from defaults
- [ ] **Create target file with:**
  - [ ] Updated copyright and includes (use factory header, not CK header)
  - [ ] Updated namespace (`miopen::conv::ck_builder::instance`)
  - [ ] Simplified function signature (no template params on DeviceGroupedConvFwdMultipleABD)
  - [ ] Namespace aliases (`namespace ckb = ck_tile::builder;`)
  - [ ] Using directive (`using namespace factories::grouped_conv_fwd;`)
  - [ ] NOTE comment that data type aliases are imported (no need to redefine)
  - [ ] Layout aliases (only those used in the file)
  - [ ] Specialization aliases (only those used in the file)
  - [ ] Converted `add_device_operation_instances<...>` calls using correct parameter mapping for factory type
- [ ] **Add file to `src/kernels/ck_builder/CMakeLists.txt`:**
  - [ ] Determine the correct source list variable (see mapping above)
  - [ ] Add relative path from `src/kernels/ck_builder/` in alphabetical order within section
  - [ ] Verify CMakeLists.txt syntax
- [ ] **Verify compilation:**
  - [ ] File compiles without errors
  - [ ] No warnings generated

## Special Cases

### 1. Files with Multiple Calls

Some files call `add_device_operation_instances` multiple times with different conv specializations. Each call must be converted separately.

Example:
```cpp
// Original
add_device_operation_instances(instances,
                               device_grouped_conv_fwd_xdl_f16_instances<2, NHWGC, GKYXC, Empty_Tuple, NHWGK, ConvFwdDefault>{});
add_device_operation_instances(instances,
                               device_grouped_conv_fwd_xdl_f16_instances<2, NHWGC, GKYXC, Empty_Tuple, NHWGK, ConvFwd1x1P0>{});

// Converted (with aliases defined earlier in the function)
add_device_operation_instances<
    device_grouped_conv_fwd_xdl_f16_instances<0>(
        2, NHWGC, GKYXC, {}, NHWGK, ConvFwdDefault)
>(instances);
add_device_operation_instances<
    device_grouped_conv_fwd_xdl_f16_instances<0>(
        2, NHWGC, GKYXC, {}, NHWGK, ConvFwd1x1P0)
>(instances);
```

### 2. Files with Part Suffixes

Some files have `_part1`, `_part2`, etc. suffixes because they were split for compilation performance. The factory functions have matching suffixes (e.g., `device_grouped_conv_fwd_wmma_cshufflev3_f16_instances_part1`).

### 3. WMMA V3 and V3 Large Tensor Files

These files use unsupported template classes (`DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3`, `DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor`). The factory headers have stub implementations that return empty arrays. These conversions should still be performed, but the resulting .cpp files will not generate any kernel instances.

### 4. Files with Non-Default OutputElementOp

If the original file uses an `OutElementOp` other than `PassThrough`, this must be passed as the `output_op` parameter to the factory function. Check the function signature in the original file.

### 5. Files with DsDataTypes

If the original file has non-empty `DsDataTypes` (e.g., `Tuple<F32>` instead of `Empty_Tuple`), this must be converted to the `ds_types` parameter. The `NumDTensor` template parameter must match the size.

Example:
```cpp
// Original: DsDataTypes = Tuple<F32>
add_device_operation_instances(instances,
                               some_factory<2, NHWGC, GKYXC, Tuple<NHWGK>, NHWGK, ConvFwdDefault>{});

// Converted: NumDTensor=1, ds_types={F32}
// (with layout/specialization aliases and F32 data type alias defined earlier)
add_device_operation_instances<
    some_factory<1>(
        2,
        NHWGC,
        GKYXC,
        {NHWGK},
        NHWGK,
        ConvFwdDefault,
        {F32})  // ds_types parameter
>(instances);
```

## CMakeLists.txt Updates

**CRITICAL:** Each converted `.cpp` file must be added to the main CMakeLists.txt file at `src/kernels/ck_builder/CMakeLists.txt` to ensure it gets compiled.

### CMakeLists.txt Location

All .cpp files are registered in a **single CMakeLists.txt**:
```
src/kernels/ck_builder/CMakeLists.txt
```

The .cpp files themselves are organized in subdirectories:
```
src/kernels/ck_builder/factories/grouped_conv2d_fwd/
├── dl/
│   └── *.cpp files
├── wmma/
│   ├── *.cpp files
│   └── large_tensor/
│       └── *.cpp files
└── xdl/
    ├── *.cpp files
    ├── comp/
    │   └── *.cpp files
    ├── large_tensor/
    │   └── *.cpp files
    ├── mem/
    │   └── *.cpp files
    └── merged_groups/
        └── *.cpp files
```

### CMakeLists.txt Structure Setup

**PREREQUISITE:** Before converting files, the CMakeLists.txt must be updated to include source list sections. The current CMakeLists.txt only contains static tests. You need to add source list variables.

**Option 1: Add source lists as files are converted**
- Start with empty source list variables
- Add each file as you convert it

**Option 2: Create all source list sections upfront**
- Create all necessary source list variables before conversion
- Add files to appropriate sections during conversion

### Recommended CMakeLists.txt Structure

Add the following sections to `src/kernels/ck_builder/CMakeLists.txt` (before or after the existing `add_library(ck_builder ...)` block):

```cmake
# Grouped Conv2D Forward - DL (4 files)
set(GROUPED_CONV2D_FWD_DL_SOURCES)

# Grouped Conv2D Forward - WMMA (16 base files)
set(GROUPED_CONV2D_FWD_WMMA_SOURCES)

# Grouped Conv2D Forward - WMMA V3 (10 files + 2 large tensor = 12 files)
set(GROUPED_CONV2D_FWD_WMMA_V3_SOURCES)

# Grouped Conv2D Forward - XDL Base (19 files)
set(GROUPED_CONV2D_FWD_XDL_SOURCES)

# Grouped Conv2D Forward - XDL Comp (15 files)
set(GROUPED_CONV2D_FWD_XDL_COMP_SOURCES)

# Grouped Conv2D Forward - XDL Large Tensor (5 files)
set(GROUPED_CONV2D_FWD_XDL_LARGE_TENSOR_SOURCES)

# Grouped Conv2D Forward - XDL Mem (16 files)
set(GROUPED_CONV2D_FWD_XDL_MEM_SOURCES)

# Grouped Conv2D Forward - XDL Merged Groups (8 files)
set(GROUPED_CONV2D_FWD_XDL_MERGED_GROUPS_SOURCES)
```

Then modify the `add_library(ck_builder ...)` to include these sources:

```cmake
add_library(ck_builder STATIC
    # Existing static tests
    static_tests/grouped_conv_2d_fwd_xdl_f32.cpp
    static_tests/grouped_conv_2d_fwd_xdl_v3_f32.cpp
    static_tests/grouped_conv_2d_fwd_xdl_large_tensor_f32.cpp
    static_tests/grouped_conv_2d_fwd_wmma_f32.cpp
    static_tests/grouped_conv_2d_fwd_dl_f32.cpp

    # Grouped conv2d fwd sources
    ${GROUPED_CONV2D_FWD_DL_SOURCES}
    ${GROUPED_CONV2D_FWD_WMMA_SOURCES}
    ${GROUPED_CONV2D_FWD_WMMA_V3_SOURCES}
    ${GROUPED_CONV2D_FWD_XDL_SOURCES}
    ${GROUPED_CONV2D_FWD_XDL_COMP_SOURCES}
    ${GROUPED_CONV2D_FWD_XDL_LARGE_TENSOR_SOURCES}
    ${GROUPED_CONV2D_FWD_XDL_MEM_SOURCES}
    ${GROUPED_CONV2D_FWD_XDL_MERGED_GROUPS_SOURCES}
)
```

### Adding Files to CMakeLists.txt

For each converted `.cpp` file:

1. **Open** `src/kernels/ck_builder/CMakeLists.txt`
2. **Determine the appropriate source list variable** based on subdirectory:
   - `dl/*.cpp` → `GROUPED_CONV2D_FWD_DL_SOURCES`
   - `wmma/*.cpp` (not V3) → `GROUPED_CONV2D_FWD_WMMA_SOURCES`
   - `wmma/*cshufflev3*.cpp` or `wmma/large_tensor/*.cpp` → `GROUPED_CONV2D_FWD_WMMA_V3_SOURCES`
   - `xdl/*.cpp` (base directory) → `GROUPED_CONV2D_FWD_XDL_SOURCES`
   - `xdl/comp/*.cpp` → `GROUPED_CONV2D_FWD_XDL_COMP_SOURCES`
   - `xdl/large_tensor/*.cpp` → `GROUPED_CONV2D_FWD_XDL_LARGE_TENSOR_SOURCES`
   - `xdl/mem/*.cpp` → `GROUPED_CONV2D_FWD_XDL_MEM_SOURCES`
   - `xdl/merged_groups/*.cpp` → `GROUPED_CONV2D_FWD_XDL_MERGED_GROUPS_SOURCES`
3. **Add the relative path** from `src/kernels/ck_builder/` to the appropriate set() command
4. **Maintain alphabetical ordering** within each section
5. **Format correctly** - one file per line, indented

Example after adding some files:
```cmake
# Grouped Conv2D Forward - DL (4 files)
set(GROUPED_CONV2D_FWD_DL_SOURCES
    factories/grouped_conv2d_fwd/dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
    factories/grouped_conv2d_fwd/dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
    factories/grouped_conv2d_fwd/dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
    factories/grouped_conv2d_fwd/dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
)
```

### Verification Checklist (Per File)

After adding to CMakeLists.txt:
- [ ] File is listed in `src/kernels/ck_builder/CMakeLists.txt`
- [ ] Path is relative to `src/kernels/ck_builder/` directory
- [ ] File is in the correct section for its subdirectory
- [ ] Filename matches exactly (including any `_part1`, `_part2` suffixes)
- [ ] Files are in alphabetical order within the section
- [ ] CMakeLists.txt compiles without errors

## Verification and Testing

### Automated Verification by Agent

**IMPORTANT:** Verification should be performed by a dedicated agent to ensure thorough checking. When launching the verification agent:

```
Launch a new agent with the following instructions:

"Your task is to verify the conversion of grouped_conv2d_fwd .cpp files from ComposableKernel to MIOpen CK Builder.

Before starting verification, you MUST read:
1. GROUPED_CONV_FWD_CONVERSION_PLAN.md - Original header file conversion plan
2. CPP_CONVERSION_PLAN.md - This .cpp file conversion plan
3. src/kernels/ck_builder/README.md - CK Builder overview
4. src/kernels/ck_builder/include/miopen/ck_builder/instance_data/README.md - Instance data documentation
5. All relevant helper function headers in instance_data/ directory
6. The original source files being verified
7. The converted target files being verified

For each converted .cpp file, verify:
1. All includes are correct and point to MIOpen headers
2. Namespace is updated from ck::tensor_operation::device::instance to miopen::conv::ck_builder::instance
3. Function signature has simplified DeviceGroupedConvFwdMultipleABD (no template params)
4. Layout and specialization aliases are defined at function start
5. All add_device_operation_instances calls use the new template parameter syntax
6. Template parameters are correctly converted to function arguments
7. NumDTensor is correctly determined from DsLayout/DsDataTypes
8. Layout names use the constexpr auto aliases (not fully qualified)
9. Specialization names use the constexpr auto aliases (not fully qualified)
10. The number of add_device_operation_instances calls matches the original
11. Each converted call references the correct factory function
12. File is added to src/kernels/ck_builder/CMakeLists.txt in the correct section and alphabetical order
13. File compiles without errors

After verification, provide a summary report listing:
- Files verified successfully
- Files with issues (with detailed descriptions)
- Compilation status
- Any missing CMakeLists.txt entries"
```

### Manual Verification (Per File)

After conversion, each file should:

1. **Structural Checks:**
   - [ ] Correct copyright header (MIT license, AMD copyright)
   - [ ] Includes updated to MIOpen headers
   - [ ] Namespace changed to miopen::conv::ck_builder::instance
   - [ ] Function signature simplified (no template params on DeviceGroupedConvFwdMultipleABD)
   - [ ] Layout and specialization aliases defined

2. **Conversion Accuracy:**
   - [ ] Same number of `add_device_operation_instances` calls as original
   - [ ] Correct factory functions referenced
   - [ ] Template parameters correctly mapped to function arguments
   - [ ] NumDTensor correctly determined and specified
   - [ ] All conv specializations preserved from original

3. **Code Quality:**
   - [ ] Uses short aliases for layouts/specializations (not fully qualified)
   - [ ] Proper formatting and indentation
   - [ ] Comments preserved where relevant

4. **Build Integration:**
   - [ ] File added to `src/kernels/ck_builder/CMakeLists.txt` in correct section
   - [ ] Relative path is correct
   - [ ] File compiles without errors
   - [ ] No warnings generated

### Testing

Once all files are converted and verified:

1. **Compilation Test:**
   ```bash
   cd build
   cmake ..
   make ck_builder_grouped_conv2d_fwd
   ```

2. **Static Analysis:**
   - Run static tests to verify kernel instantiation
   - Check that constexpr factory functions are evaluated at compile time

3. **Kernel Count Verification:**
   - Compare total kernel instance count against original CK library
   - Ensure no instances were lost or duplicated during conversion

4. **Regression Testing:**
   - If integration tests exist, run them to ensure functional equivalence
   - Compare kernel selection behavior with original CK implementation

## Dependencies

This conversion depends on:

1. **Factory header files (`.hpp`)** being fully converted (see `GROUPED_CONV_FWD_CONVERSION_PLAN.md`)
2. **The new `add_device_operation_instances` function** in `kernel_instantiation.hpp`
3. **All helper functions** in `instance_data/*.hpp` (xdl.hpp, wmma.hpp, dl.hpp, xdl_v3.hpp, xdl_large_tensor.hpp, wmma_v3.hpp, wmma_v3_large_tensor.hpp)
4. **CMakeLists.txt** at `src/kernels/ck_builder/CMakeLists.txt` with appropriate source list sections
5. **DeviceGroupedConvFwdMultipleABD** base class definition (simplified interface without template parameters)

### Prerequisite Verification

Before starting .cpp conversions, verify:

- [ ] All 24 factory header files from `GROUPED_CONV_FWD_CONVERSION_PLAN.md` are converted
- [ ] `kernel_instantiation.hpp` contains the new `add_device_operation_instances<auto arr>` function
- [ ] All helper functions in `instance_data/` directory are available
- [ ] Directory structure exists: `src/kernels/ck_builder/factories/grouped_conv2d_fwd/` with all subdirectories
- [ ] `src/kernels/ck_builder/CMakeLists.txt` exists and has sections for grouped_conv2d_fwd sources

## Progress Tracking

Use the following phase-based checklist to track conversion progress. Complete each phase before moving to the next for better organization.

### Phase 1: DL Files (4 files) - SIMPLEST, START HERE

Good for learning the conversion process. DL has 7-parameter templates (no NDimSpatial).

- [x] dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
- [x] dl/device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
- [x] dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
- [x] dl/device_grouped_conv2d_fwd_dl_nhwgc_gkyxc_nhwgk_f32_instance.cpp

**Phase 1 Complete:** ☑ All 4 DL files converted and compiling

---

### Phase 2: WMMA Base Files (16 files) - MODERATE COMPLEXITY

Standard WMMA files with 8-parameter templates. Good practice before tackling XDL.

- [x] wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1p0_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1s1p0_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_oddc_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1p0_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1s1p0_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_oddc_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_1x1p0_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_1x1s1p0_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_oddc_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1p0_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1s1p0_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_oddc_instance.cpp

**Phase 2 Complete:** ☑ All 16 WMMA base files converted and compiling

---

### Phase 3: WMMA V3 Files (12 files) - UNSUPPORTED BUT CONVERT ANYWAY

These use unsupported V3 templates and will generate empty arrays. Convert for completeness and future support.

- [x] wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part1.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part2.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part3.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part4.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part1.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part2.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part3.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part4.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_wave_transfer_nhwgc_gkyxc_nhwgk_f16_instance.cpp
- [x] wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_wave_transfer_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
- [x] wmma/large_tensor/device_grouped_conv2d_fwd_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_f16_instance.cpp
- [x] wmma/large_tensor/device_grouped_conv2d_fwd_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instance.cpp

**Phase 3 Complete:** ☑ All 12 WMMA V3 files converted and compiling (will generate no kernel instances)

---

### Phase 4: XDL Base Files (20 files) - MOST COMMON PATTERN

Standard XDL files. These follow the same pattern as Phase 2 but with XDL-specific factories.

- [x] xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_bf16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f32_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_16x16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_16x16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_16x16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_16x16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_16x16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_16x16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_16x16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_bf16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f16_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f32_instance.cpp
- [x] xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_int8_instance.cpp

**Phase 4 Complete:** ☑ All 20 XDL base files converted and compiling

---

### Phase 5: XDL Specialized Files (44 files) - LARGEST BATCH

Specialized XDL variants (comp, large_tensor, mem, merged_groups). Can be parallelized across subdirectories.

#### XDL Comp (15 files)
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_2x_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_part2_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_2x_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_comp_part2_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_comp_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_comp_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_comp_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_comp_2x_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_comp_part2_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_2x_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_part2_instance.cpp
- [x] xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_comp_instance.cpp

#### XDL Large Tensor (5 files)
- [x] xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
- [x] xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f16_instance.cpp
- [x] xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_instance.cpp
- [x] xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
- [x] xdl/large_tensor/device_grouped_conv2d_fwd_xdl_large_tensor_nhwgc_gkyxc_nhwgk_int8_instance.cpp

#### XDL Mem (16 files)
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_inter_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_intra_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_direct_load_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_mem_inter_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_mem_intra_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_direct_load_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_mem_inter_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_mem_intra_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_mem_inter_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_mem_intra_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_mem_inter_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_mem_intra_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_inter_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_mem_intra_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_inter_instance.cpp
- [x] xdl/mem/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_mem_intra_instance.cpp

#### XDL Merged Groups (8 files)
- [x] xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
- [x] xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f16_instance.cpp
- [x] xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_instance.cpp
- [x] xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_tf32_instance.cpp
- [x] xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_nhwgc_gkyxc_nhwgk_int8_instance.cpp
- [x] xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_bf16_instance.cpp
- [x] xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f16_instance.cpp
- [x] xdl/merged_groups/device_grouped_conv2d_fwd_xdl_merged_groups_ngchw_gkcyx_ngkhw_f32_instance.cpp

**Phase 5 Complete:** ☑ All 44 XDL specialized files converted and compiling

---

### Overall Progress Summary

- **Phase 1 (DL):** ☑ 4/4 files
- **Phase 2 (WMMA Base):** ☑ 16/16 files
- **Phase 3 (WMMA V3):** ☑ 12/12 files
- **Phase 4 (XDL Base):** ☑ 20/20 files
- **Phase 5 (XDL Specialized):** ☑ 44/44 files

**TOTAL:** ☑ 96/96 files converted and compiling

## Summary

This plan outlines the conversion of 96 `.cpp` files from ComposableKernel to MIOpen's CK Builder framework. Each file converts calls from the old tuple-based `add_device_operation_instances` to the new constexpr array-based version. The conversion is mechanical but requires careful attention to:

1. Template parameter → function argument mapping
2. Layout and specialization enum conversions
3. NumDTensor determination from DsLayout
4. Proper handling of optional parameters
5. Namespace and include updates

The files should be converted in subdirectory batches, with DL files serving as the simplest starting point.
