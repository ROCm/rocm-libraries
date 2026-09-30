// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <initializer_list>
#include <optional>
#include <string>

namespace asm_sdpa_engine
{
namespace plan_utils
{

// =============================================================================
// Tensor dtype classification
// =============================================================================
//
// True when every tensor dtype in `types` equals `expected`. Used by the
// forward and backward plan builders to recognise the single-dtype tensor sets
// the CSV schema keys on (e.g. all-BF16 or all-FP16).
inline bool
    allDataTypesEqual(hipdnn_flatbuffers_sdk::data_objects::DataType expected,
                      std::initializer_list<hipdnn_flatbuffers_sdk::data_objects::DataType> types)
{
    return std::all_of(
        types.begin(), types.end(), [expected](auto type) { return type == expected; });
}

// =============================================================================
// Mask classification
// =============================================================================
//
// Shared by SdpaFwdPlanBuilder and SdpaBwdPlanBuilder. The CSV `mask` column
// stores these ordinals directly, so the integer values are part of the
// dispatch contract and must not be reordered.
// Mask types for AITER ASM (.co) kernel dispatch.
// Ordinals match AITER's asm_mask_type() return values (mha_bwd.cu / mha_fwd.cu)
// and the CSV `mask` column — do not reorder.
// Note: SLIDING_WINDOW (3) is distinct from AITER's mask_enum::window_generic (also
// ordinal 3), which maps to -1 (unsupported) and falls back to CK kernels.
enum class MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL = 1,
    BOTTOM_RIGHT_CAUSAL = 2,
    SLIDING_WINDOW = 3
};

// Classify the mask requested by an SDPA (forward or backward) attribute set.
//
// Two sources can describe the mask: the modern left_bound / right_bound /
// diagonal_alignment trio, and the deprecated causal_mask /
// causal_mask_bottom_right booleans. When a deprecated boolean is set it takes
// precedence and the modern trio is ignored; otherwise the trio is
// authoritative. The two deprecated booleans are mutually exclusive, so setting
// both throws HipdnnPluginException(INVALID_VALUE).
//
// Guaranteeing the two parameter sets agree belongs in the hipDNN frontend; this
// helper only resolves which source wins for dispatch.
//
// Absence-awareness: the generated flatbuffer accessors expose the causal_mask*
// fields as plain bool defaulting to false, with no has_*() accessor.
// "Explicitly false" and "unset" are therefore indistinguishable; a false bool
// is treated as "not requested". left_bound / right_bound are
// flatbuffers::Optional, but an unset bound is treated as unbounded (-1) to
// match the canonical convention used across the SDPA path, so a partially
// specified trio (e.g. only right_bound = 0) still derives a mask rather than
// silently falling back to NO_MASK.
template <typename SdpaAttrsT>
MaskType getMaskType(const SdpaAttrsT& attrs)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const bool causalDeprecated = attrs.causal_mask();
    const bool bottomRightDeprecated = attrs.causal_mask_bottom_right();

    // The two deprecated booleans are mutually exclusive.
    if(causalDeprecated && bottomRightDeprecated)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "SDPA: causal_mask and causal_mask_bottom_right are mutually exclusive "
            "but both are set");
    }

    // Deprecated booleans take precedence: when either is set, defer to it and
    // ignore the modern bounds trio.
    if(causalDeprecated)
    {
        return MaskType::TOP_LEFT_CAUSAL;
    }
    if(bottomRightDeprecated)
    {
        return MaskType::BOTTOM_RIGHT_CAUSAL;
    }

    // No deprecated boolean set: the modern bounds trio is authoritative. An
    // unset bound means unbounded, represented here as -1, so a partially
    // specified trio still resolves to the mask it describes.
    const int64_t left = attrs.left_bound().has_value() ? attrs.left_bound().value() : -1;
    const int64_t right = attrs.right_bound().has_value() ? attrs.right_bound().value() : -1;
    if(left == -1 && right == -1) // both unbounded
    {
        return MaskType::NO_MASK;
    }
    if(left == -1 && right == 0) // causal: attend up to the diagonal
    {
        return attrs.diagonal_alignment() == DiagonalAlignment::BOTTOM_RIGHT
                   ? MaskType::BOTTOM_RIGHT_CAUSAL
                   : MaskType::TOP_LEFT_CAUSAL;
    }
    return MaskType::SLIDING_WINDOW; // anything else is a sliding window
}

// =============================================================================
// Sliding-window mask coordinate transformation
// =============================================================================
//
// Converts raw window sizes (left_bound, right_bound from the hipDNN graph)
// into the precomputed mask coordinates (mask_y, mask_x) that the AITER ASM
// DQDKDV kernel expects in its argument struct.
//
// AITER reference: ck_tile_shim.h::compute_mask_coordinates()
// Called only for mask type 3 (SLIDING_WINDOW); mask types 0-2 bake mask
// behavior into the kernel binary and ignore mask_x/mask_y.
//
// Negative window sizes (including -1) are treated as unbounded: replaced with
// seqLen-1 on the corresponding axis, matching AITER's semantics.
inline std::pair<int32_t, int32_t> computeMaskCoordinates(
    int32_t leftSize, int32_t rightSize, int32_t seqLenQ, int32_t seqLenK, bool isTopLeft)
{
    const int32_t leftDefault = isTopLeft ? seqLenQ - 1 : seqLenK - 1;
    const int32_t rightDefault = isTopLeft ? seqLenK - 1 : seqLenQ - 1;
    leftSize = leftSize < 0 ? leftDefault : leftSize;
    rightSize = rightSize < 0 ? rightDefault : rightSize;
    const int32_t xOff = isTopLeft ? 0 : seqLenK - seqLenQ;
    const int32_t yOff = isTopLeft ? 0 : seqLenQ - seqLenK;
    return {1 + leftSize + yOff, 1 + rightSize + xOff}; // {mask_y, mask_x}
}

// =============================================================================
// Byte-stride overflow primitive
// =============================================================================
//
// Returns true when `elements * elementBytes` fits in a uint32_t (the kernarg
// stride field width) and `elements` is non-negative; logs the offending field
// and returns false otherwise. The per-tensor wrappers that enumerate the
// concrete stride fields live in the fwd / bwd plan builders, since the tensor
// sets differ between passes.
inline bool byteStrideFitsU32(const char* name, int64_t elements, int64_t elementBytes)
{
    constexpr auto K_U32_MAX_AS_I64 = static_cast<int64_t>(UINT32_MAX);
    if(elements >= 0 && elements * elementBytes <= K_U32_MAX_AS_I64)
    {
        return true;
    }
    HIPDNN_PLUGIN_LOG_INFO("SDPA: byte stride overflows uint32_t (field="
                           << name << ", elements=" << elements << ", elementBytes=" << elementBytes
                           << ", scaled=" << elements * elementBytes << ", max=" << K_U32_MAX_AS_I64
                           << ")");
    return false;
}

// =============================================================================
// HIP device string query with error handling
// =============================================================================
//
// Query the HIP device string for the stream, logging `logPrefix` on failure.
// Returns std::nullopt when the HIP runtime throws.
inline std::optional<std::string> tryGetDeviceString(hipStream_t stream, const char* logPrefix)
{
    try
    {
        return hip_kernel_provider_common::getDeviceString(stream);
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR(logPrefix << e.what());
        return std::nullopt;
    }
}

} // namespace plan_utils
} // namespace asm_sdpa_engine
