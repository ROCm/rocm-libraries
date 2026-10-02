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
#include <utility>

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

// The mask an SDPA (forward or backward) attribute set asks for: its kind plus
// the band it describes in the left_bound / right_bound / alignment convention
// (-1 = unbounded). `left`, `right` and `topLeft` are what a SLIDING_WINDOW
// kernel needs; for the other kinds they restate the mask.
struct ResolvedMask
{
    MaskType type = MaskType::NO_MASK;
    int64_t left = -1;
    int64_t right = -1;
    bool topLeft = true;
};

// The kind of mask a band is, in the left / right / alignment convention
// (-1 = unbounded).
inline MaskType classifyBand(int64_t left, int64_t right, bool topLeft)
{
    if(left == -1 && right == -1)
    {
        return MaskType::NO_MASK;
    }
    if(left == -1 && right == 0) // causal: attend up to the diagonal
    {
        return topLeft ? MaskType::TOP_LEFT_CAUSAL : MaskType::BOTTOM_RIGHT_CAUSAL;
    }
    return MaskType::SLIDING_WINDOW; // anything else is a sliding window
}

// Resolve the mask requested by an SDPA (forward or backward) attribute set.
//
// Two sources can describe the mask: the modern left_bound / right_bound /
// diagonal_alignment trio, and the deprecated causal_mask /
// causal_mask_bottom_right booleans. A deprecated boolean fixes the diagonal
// (right bound 0) and its alignment, overriding diagonal_alignment, but it keeps
// a real left_bound: causal_mask plus left_bound is a causal sliding window, as
// cuDNN reads set_causal_mask(true) next to a window. This matches the CPU and
// GPU SDPA references (extractDiagonalBandParams). Without a deprecated boolean
// the trio is authoritative.
//
// left_bound counts like flash-attn's window_size_left: with the causal diagonal,
// left_bound L keeps L + 1 keys per row, the diagonal included. cuDNN's
// set_sliding_window_length(L) keeps L, so the two spellings are not the same
// window (issue #12982). The engines and both references all use the L + 1 count.
//
// Invalid combinations throw HipdnnPluginException(INVALID_VALUE): both
// deprecated booleans at once; a bound below -1 (the references reject those
// too); and a deprecated boolean next to a positive right_bound, which the
// boolean would otherwise silently override with 0 (cuDNN's Python binding and
// the gfx950 dense pack reject that combination as well). An explicit
// right_bound of -1 or 0 next to a boolean is accepted.
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
ResolvedMask resolveMask(const SdpaAttrsT& attrs)
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

    const int64_t leftBound = attrs.left_bound().has_value() ? attrs.left_bound().value() : -1;
    const int64_t rightBound = attrs.right_bound().has_value() ? attrs.right_bound().value() : -1;
    if(leftBound < -1 || rightBound < -1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "SDPA: left_bound and right_bound must be >= -1 (left_bound="
                + std::to_string(leftBound) + ", right_bound=" + std::to_string(rightBound) + ")");
    }

    if((causalDeprecated || bottomRightDeprecated) && rightBound > 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "SDPA: a causal mask fixes right_bound at 0, but right_bound="
                + std::to_string(rightBound) + " is set");
    }

    ResolvedMask mask;
    mask.left = leftBound;
    if(causalDeprecated || bottomRightDeprecated)
    {
        mask.right = 0;
        mask.topLeft = causalDeprecated;
    }
    else
    {
        mask.right = rightBound;
        mask.topLeft = attrs.diagonal_alignment() != DiagonalAlignment::BOTTOM_RIGHT;
    }
    mask.type = classifyBand(mask.left, mask.right, mask.topLeft);
    return mask;
}

// The kind of mask resolveMask() derives from the attributes alone; see there for
// the precedence rules. An engine choosing a kernel wants resolveMaskFor(), which
// also accounts for the sequence lengths.
template <typename SdpaAttrsT>
MaskType getMaskType(const SdpaAttrsT& attrs)
{
    return resolveMask(attrs).type;
}

// Narrow a SLIDING_WINDOW mask's bounds to the kernel's int32 window fields.
//
// A bound that reaches the widest offset the band can span for this alignment
// and these sequence lengths masks nothing on its side, so it becomes -1. That
// is the same mask: computeMaskCoordinates() replaces -1 with exactly that
// widest offset (seqLen - 1 on the matching axis). Every bound that survives is
// below a sequence length, so the int32 cast is exact; a raw int64 bound such
// as 4294967298 would otherwise wrap to 2 and shrink the window.
//
// Expects bounds >= -1 (resolveMask() rejects the rest) and sequence lengths
// that fit int32 (the kernel argument fields are int32 as well).
inline std::pair<int32_t, int32_t>
    kernelWindowBounds(const ResolvedMask& mask, int64_t seqLenQ, int64_t seqLenKv)
{
    const int64_t leftSpan = mask.topLeft ? seqLenQ - 1 : seqLenKv - 1;
    const int64_t rightSpan = mask.topLeft ? seqLenKv - 1 : seqLenQ - 1;
    const auto narrow = [](int64_t bound, int64_t span) {
        return (bound < 0 || bound >= span) ? int32_t{-1} : static_cast<int32_t>(bound);
    };
    return {narrow(mask.left, leftSpan), narrow(mask.right, rightSpan)};
}

// The mask a kernel has to apply to this graph at these sequence lengths:
// resolveMask() with both bounds narrowed by kernelWindowBounds() and the kind
// classified again on the narrowed bounds. A window that covers the whole
// sequence is the plain mask it equals, so for example causal_mask with
// left_bound 128 at Sq = Skv <= 129 is TOP_LEFT_CAUSAL and runs on a causal
// kernel instead of needing a sliding-window one. `left` and `right` fit the
// kernels' int32 window fields.
template <typename SdpaAttrsT>
ResolvedMask resolveMaskFor(const SdpaAttrsT& attrs, int64_t seqLenQ, int64_t seqLenKv)
{
    ResolvedMask mask = resolveMask(attrs);
    const auto [left, right] = kernelWindowBounds(mask, seqLenQ, seqLenKv);
    mask.left = left;
    mask.right = right;
    mask.type = classifyBand(left, right, mask.topLeft);
    return mask;
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
