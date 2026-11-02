// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_utils.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//   include/cudnn_frontend/knobs.h
//     Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_utils.h
 * @brief cuDNN FE enums, knob types, and utility functions for the shim.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <hipdnn_compatibility/cudnn/cudnn_runtime_version.h>
#include <hipdnn_frontend/Types.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{

// cuDNN FE enums, aliased 1:1 to the hipDNN enums of the same name.
using hipdnn_frontend::AttentionImplementation_t;
using hipdnn_frontend::BehaviorNote_t;
using hipdnn_frontend::BuildPlanPolicy_t;
using hipdnn_frontend::ConvolutionMode_t;
using hipdnn_frontend::DataType_t;
using hipdnn_frontend::DiagonalAlignment_t;
using hipdnn_frontend::HeurMode_t;
using hipdnn_frontend::MoeGroupedMatmulMode_t;
using hipdnn_frontend::NormFwdPhase_t;
using hipdnn_frontend::NumericalNote_t;
using hipdnn_frontend::PaddingMode_t;
using hipdnn_frontend::PointwiseMode_t;
using hipdnn_frontend::ReductionMode_t;
using hipdnn_frontend::ResampleMode_t;

/// @brief cuDNN FE knob type; mapped to hipDNN's string knob IDs by name.
// NOLINTNEXTLINE(readability-identifier-naming)
enum class KnobType_t
{
    NOT_SET,
    SWIZZLE,
    TILE_SIZE,
    EDGE,
    MULTIPLY,
    SPLIT_K_BUF,
    TILEK,
    STAGES,
    REDUCTION_MODE,
    SPLIT_K_SLC,
    IDX_MODE,
    SPECFILT,
    KERNEL_CFG,
    WORKSPACE,
    TILE_CGA_M,
    TILE_CGA_N,
    BLOCK_SIZE,
    OCCUPANCY,
    ARRAY_SIZE_PER_THREAD,
    SPLIT_COLS,
    TILE_ROWS,
    TILE_COLS,
    LOAD_SIZE,
    CTA_COUNT,
    STREAM_K,
    SPLIT_P_SLC,
    TILE_M,
    TILE_N,
    WARP_SPEC_CFG
};

/// @brief cuDNN FE knob: type plus int64 min/max/stride.
class Knob
{
public:
    KnobType_t type = KnobType_t::NOT_SET;
    int64_t maxValue = 0;
    int64_t minValue = 0;
    int64_t stride = 0;

    Knob() = default;
    Knob(KnobType_t knobType, int64_t max, int64_t min, int64_t str)
        : type(knobType)
        , maxValue(max)
        , minValue(min)
        , stride(str)
    {
    }
};

/// @brief Empty placeholder: `set_kernel_cache()` accepts it; nothing is honored.
struct KernelCache
{
};

/// @brief Empty placeholder: `set_device_properties()` accepts it; nothing is honored.
struct DeviceProperties
{
};

// Shim-owned: hipDNN publishes no counterpart for these enums.
// NOLINTBEGIN(readability-identifier-naming)
/// @brief cuDNN FE tensor reordering mode.
enum class TensorReordering_t
{
    NONE,
    INT8x32,
    F16x16,
    F8_128x4
};

/// @brief cuDNN FE reshape mode.
enum class ReshapeMode_t
{
    NOT_SET,
    VIEW_ONLY,
    LOGICAL
};
// NOLINTEND(readability-identifier-naming)

// NormMode_t, RngDistribution_t, and DescriptorType_t are not provided yet.

} // namespace hipdnn_frontend::compatibility::cudnn_frontend

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
{

// NOLINTBEGIN(readability-identifier-naming): mirrors cuDNN FE's snake_case spelling.

// Returns CUDNN_VERSION; cuDNN FE queries the loaded libcudnn instead.
inline size_t get_backend_version()
{
    return CUDNN_VERSION;
}

inline size_t get_element_size_in_bits(DataType_t datatype)
{
    switch(datatype)
    {
    case DataType_t::INT8x32:
        return 256;
    case DataType_t::COMPLEX_FP64:
        return 128;
    case DataType_t::DOUBLE:
    case DataType_t::INT64:
    case DataType_t::COMPLEX_FP32:
        return 64;
    case DataType_t::FLOAT:
    case DataType_t::INT32:
    case DataType_t::INT8x4:
    case DataType_t::UINT8x4:
        return 32;
    case DataType_t::HALF:
    case DataType_t::BFLOAT16:
        return 16;
    case DataType_t::INT8:
    case DataType_t::UINT8:
    case DataType_t::FP8_E4M3:
    case DataType_t::FP8_E5M2:
    case DataType_t::FAST_FLOAT_FOR_FP8:
    case DataType_t::FP8_E8M0:
    case DataType_t::FP8_E4M3_FNUZ:
    case DataType_t::FP8_E5M2_FNUZ:
        return 8;
    case DataType_t::FP6_E2M3:
    case DataType_t::FP6_E3M2:
        return 6;
    case DataType_t::FP4_E2M1:
    case DataType_t::INT4:
        return 4;
    case DataType_t::BOOLEAN:
        return 1;
    case DataType_t::NOT_SET:
    default:
        return 0;
    }
}

// NOLINTEND(readability-identifier-naming)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail

// Declared so this header is self-contained; populated by the cudnn_frontend/* headers.
namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
