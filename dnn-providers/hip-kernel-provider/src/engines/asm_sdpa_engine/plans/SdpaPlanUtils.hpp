// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <initializer_list>
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
enum class MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL = 1,
    BOTTOM_RIGHT_CAUSAL = 2,
    WINDOW_GENERIC = 3
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
    return MaskType::WINDOW_GENERIC; // anything else is a sliding window
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
// Per-launch post-check
// =============================================================================
//
// Surfaces async launch faults that hipModuleLaunchKernel itself returned
// success for. Without this, a memory-access fault inside the ASM kernel would
// propagate silently to the next stage. Throws
// HipdnnPluginException(INTERNAL_ERROR) on async error so the caller's API
// status reflects the actual root cause.
inline void throwOnLaunchPostError(const char* stage)
{
    const hipError_t err = hipPeekAtLastError();
    if(err != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string(stage) + ": post-launch error: " + hipGetErrorString(err));
    }
}

} // namespace plan_utils
} // namespace asm_sdpa_engine
