// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend (include/cudnn_frontend_utils.h),
// used under the MIT license.

/**
 * @file cudnn_frontend_utils.h
 * @brief FE-namespace enums and free helpers for the hipDNN cuDNN-compatibility shim.
 *
 * The v9 graph API signatures use cuDNN's FE-namespace enums (`DataType_t`,
 * `PointwiseMode_t`, …), not the C-API ones from `<cudnn.h>`. hipDNN already
 * publishes these as cuDNN-named `_t` typedefs (see `Types.hpp`), so this header
 * just aliases them into `<shim_ns>` with `using` declarations — zero overhead,
 * no numeric cast between enum families.
 *
 * It also carries the `cudnn_frontend::detail` free functions upstream splits
 * across `cudnn_frontend_utils.h` and `cudnn_frontend_shim.h`.
 *
 * @note Internal-to-shim; pulled in by the umbrella `cudnn_frontend.h`.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <hipdnn_compatibility/cudnn/cudnn_runtime_version.h>
#include <hipdnn_frontend/Types.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{

// FE-namespace enums hipDNN publishes 1:1, aliased so e.g.
// `cudnn_frontend::DataType_t` *is* `hipdnn_frontend::DataType_t`.
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

// cuDNN FE exposes knobs as a fixed enum plus a simple int64 range record. hipDNN
// native knobs are open-ended string IDs with richer typed constraints, so the
// shim intentionally owns a cuDNN-shaped type here and maps it explicitly when
// forwarding to hipDNN.
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

// Deliberately-hollow placeholders: they exist only so the cuDNN-spelled setter
// signatures (set_kernel_cache / set_device_properties) compile against hipified
// consumer source. They carry none of cuDNN's real API — a consumer that calls
// methods on them will not compile. Grow real members only when the shim can
// honor the corresponding feature.
struct KernelCache
{
};

struct DeviceProperties
{
};

// Shim-owned FE enums. hipDNN publishes no counterpart, and cuDNN's numeric
// values are not part of its public contract, so these are declared here rather
// than aliased. They exist so consumer source that names them compiles.
// NOLINTBEGIN(readability-identifier-naming)
enum class TensorReordering_t
{
    NONE,
    INT8x32,
    F16x16,
    F8_128x4
};

enum class ReshapeMode_t
{
    NOT_SET,
    VIEW_ONLY,
    LOGICAL
};
// NOLINTEND(readability-identifier-naming)

// Remaining cuDNN FE-namespace enums (NormMode_t, RngDistribution_t,
// DescriptorType_t) are not aliased yet: hipDNN does not publish them and their
// nodes are out of scope. They are aliased when their node lands.

} // namespace hipdnn_frontend::compatibility::cudnn_frontend

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
{

// NOLINTBEGIN(readability-identifier-naming): these mirror cuDNN FE's
// snake_case spelling for source compatibility.

// cuDNN FE resolves this from the loaded libcudnn; the shim has a single
// compile-time claim, so runtime and compiled version are the same number.
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

// The `graph` sub-namespace is populated by the `cudnn_frontend/*` headers the
// umbrella pulls in; declared empty here so this header is self-contained when
// included on its own.
namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
