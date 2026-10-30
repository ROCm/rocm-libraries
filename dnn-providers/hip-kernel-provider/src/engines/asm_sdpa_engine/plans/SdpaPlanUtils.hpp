// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
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

// The diagonal band an SDPA (forward or backward) attribute set requests, with
// the deprecated causal_mask / causal_mask_bottom_right booleans merged in.
//
// The bounds are in the kernels' FlashAttention convention (window_size_left /
// window_size_right), not the graph's: leftBound is the number of keys kept
// before the diagonal, so it is the graph's cuDNN-semantics left_bound minus 1
// (see resolveDiagonalBand()). rightBound means the same in both conventions.
// -1 is unbounded.
struct DiagonalBand
{
    int64_t leftBound;
    int64_t rightBound;
    hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment alignment;
};

// Replace each bound that already covers the whole sequence span with -1
// (unbounded).
//
// A bound that reaches the widest offset the band can span for this alignment
// and these sequence lengths masks nothing on its side, so it becomes -1. That
// is the same mask: computeMaskCoordinates() replaces -1 with exactly that
// widest offset (seqLen - 1 on the matching axis). A covering window therefore
// classifies as plain causal or no-mask. Bounds that survive are returned
// unchanged as int64, so a kernel field filled from one must first confirm it
// fits int32. Every survivor is strictly below a sequence length, so the
// dimFitsI32() gate the plan builders apply to seqLenQ / seqLenKv is what makes
// that narrowing safe.
//
// A sequence length of 0 or 1 gives a span <= 0, so a bound of 0 also becomes
// -1 (e.g. top-left causal with seqLenKv == 1 is unmasked).
//
// Expects bounds >= -1 (resolveDiagonalBand() rejects the rest).
inline DiagonalBand makeCoveringBoundsUnbound(DiagonalBand band, int64_t seqLenQ, int64_t seqLenKv)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    const bool bottomRight = band.alignment == DiagonalAlignment::BOTTOM_RIGHT;
    const int64_t leftSpan = bottomRight ? seqLenKv - 1 : seqLenQ - 1;
    const int64_t rightSpan = bottomRight ? seqLenQ - 1 : seqLenKv - 1;
    const auto unboundIfCovering
        = [](int64_t bound, int64_t span) { return (bound < 0 || bound >= span) ? -1 : bound; };
    band.leftBound = unboundIfCovering(band.leftBound, leftSpan);
    band.rightBound = unboundIfCovering(band.rightBound, rightSpan);
    return band;
}

// Extract the requested diagonal band, merging the deprecated booleans the way
// cuDNN's setters compose: a boolean acts as if its setter ran first (right
// bound 0, plus BOTTOM_RIGHT alignment for causal_mask_bottom_right) and the
// stored bounds and alignment are applied on top. An unset bound, or one equal
// to -1, does not override the causal right bound. causal_mask keeps the stored
// alignment; causal_mask_bottom_right always resolves to BOTTOM_RIGHT, because
// TOP_LEFT is the schema default and indistinguishable from an unset alignment.
//
// The graph's left_bound follows cuDNN: left_bound = L keeps the L keys ending at
// the diagonal, diagonal included. The returned band holds the FlashAttention
// window_size_left the kernels take, L - 1 (see DiagonalBand).
//
// Setting both booleans, a bound below -1, or left_bound = 0 (which keeps no key at or
// before the diagonal, and which L - 1 would turn into "unbounded") throws
// HipdnnPluginException(INVALID_VALUE).
template <typename SdpaAttrsT>
DiagonalBand resolveDiagonalBand(const SdpaAttrsT& attrs)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const bool causalDeprecated = attrs.causal_mask();
    const bool bottomRightDeprecated = attrs.causal_mask_bottom_right();

    if(causalDeprecated && bottomRightDeprecated)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "SDPA: causal_mask and causal_mask_bottom_right are mutually exclusive "
            "but both are set");
    }

    const int64_t left = attrs.left_bound().has_value() ? attrs.left_bound().value() : -1;
    const int64_t right = attrs.right_bound().has_value() ? attrs.right_bound().value() : -1;

    if(left < -1 || right < -1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "SDPA: left_bound and right_bound must be >= -1 (got left_bound=" + std::to_string(left)
                + ", right_bound=" + std::to_string(right) + ")");
    }
    if(left == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "SDPA: left_bound must be -1 (unbounded) or >= 1; left_bound = L keeps the L keys "
            "ending at the diagonal, so 0 keeps no key at or before it");
    }

    const int64_t windowLeft = left > 0 ? left - 1 : -1;

    if(!causalDeprecated && !bottomRightDeprecated)
    {
        return {windowLeft, right, attrs.diagonal_alignment()};
    }

    return {windowLeft,
            right >= 0 ? right : 0,
            bottomRightDeprecated ? DiagonalAlignment::BOTTOM_RIGHT : attrs.diagonal_alignment()};
}

// Classify a diagonal band (from resolveDiagonalBand(), typically passed through
// makeCoveringBoundsUnbound() first).
inline MaskType getMaskType(int64_t leftBound,
                            int64_t rightBound,
                            hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment alignment)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;

    if(leftBound == -1 && rightBound == -1) // both unbounded
    {
        return MaskType::NO_MASK;
    }
    if(leftBound == -1 && rightBound == 0) // causal: attend up to the diagonal
    {
        return alignment == DiagonalAlignment::BOTTOM_RIGHT ? MaskType::BOTTOM_RIGHT_CAUSAL
                                                            : MaskType::TOP_LEFT_CAUSAL;
    }
    return MaskType::SLIDING_WINDOW; // anything else is a sliding window
}

// =============================================================================
// Sliding-window mask coordinate transformation
// =============================================================================
//
// Converts FlashAttention-convention window sizes (a DiagonalBand's bounds, not
// the graph's cuDNN-semantics left_bound) into the precomputed mask coordinates
// (mask_y, mask_x) that the AITER ASM
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
// Kernarg range primitives
// =============================================================================
//
// Dimensions and strides reach the kernels through 32-bit fields: the kernarg
// structs store them as uint32_t, the launch grid takes unsigned int, and the
// kernel registries are keyed on int. A value that does not fit its field
// truncates silently and the kernel then runs on a different shape than the
// graph asked for, so the plan builders check every one of them in
// isApplicable, ahead of the first narrowing cast. Each predicate logs the
// offending field and returns false (rather than returning early) so a caller
// can report every violation of a tensor in one pass.

// For dimensions whose only 32-bit narrowing is the uint32_t kernarg field or
// the unsigned grid dimension that carries them: batch and the head counts.
inline bool dimFitsU32(const char* name, int64_t dim)
{
    constexpr auto K_U32_MAX_AS_I64 = static_cast<int64_t>(UINT32_MAX);
    if(dim >= 0 && dim <= K_U32_MAX_AS_I64)
    {
        return true;
    }
    HIPDNN_PLUGIN_LOG_INFO("SDPA: dimension overflows its uint32_t kernarg field (field="
                           << name << ", value=" << dim << ", max=" << K_U32_MAX_AS_I64 << ")");
    return false;
}

// For dimensions that pass through a signed 32-bit value on the way to their
// kernarg field: the head dims are cast to int for the kernel-registry lookup,
// and the sequence lengths become int32 mask coordinates on the backward
// window path (computeMaskCoordinates) and feed `(extent + ts - 1) / ts` grid
// math evaluated in unsigned int, which is exact only while extent stays below
// 2^31.
inline bool dimFitsI32(const char* name, int64_t dim)
{
    constexpr auto K_I32_MAX_AS_I64 = static_cast<int64_t>(INT32_MAX);
    if(dim >= 0 && dim <= K_I32_MAX_AS_I64)
    {
        return true;
    }
    HIPDNN_PLUGIN_LOG_INFO("SDPA: dimension overflows its int32 narrowing (field="
                           << name << ", value=" << dim << ", max=" << K_I32_MAX_AS_I64 << ")");
    return false;
}

// Returns true when `elements * elementBytes` fits in a uint32_t (the kernarg
// stride field width) and `elements` is non-negative; logs the offending field
// and returns false otherwise. The comparison divides instead of multiplying,
// so the product it rules out is never formed. `elementBytes` folds in any
// constant factor the kernel applies on top of the element size (the backward
// Ts field multiplies the K sequence stride by the tile size as well), and a
// non-positive `elementBytes` is rejected rather than divided by. The
// per-tensor wrappers that enumerate the concrete stride fields live in the fwd
// / bwd plan builders, since the tensor sets differ between passes.
inline bool byteStrideFitsU32(const char* name, int64_t elements, int64_t elementBytes)
{
    constexpr auto K_U32_MAX_AS_I64 = static_cast<int64_t>(UINT32_MAX);
    if(elements >= 0 && elementBytes > 0 && elements <= K_U32_MAX_AS_I64 / elementBytes)
    {
        return true;
    }
    HIPDNN_PLUGIN_LOG_INFO("SDPA: byte stride overflows uint32_t (field="
                           << name << ", elements=" << elements << ", elementBytes=" << elementBytes
                           << ", max elements="
                           << (elementBytes > 0 ? K_U32_MAX_AS_I64 / elementBytes : 0) << ")");
    return false;
}

// =============================================================================
// Declared device-pointer alignment
// =============================================================================
//
// The backend rejects, at execute, any device pointer whose address is not a multiple of the
// tensor's declared alignment (TensorAttributes::alignment(), bytes, default 16). A declared
// alignment A therefore only guarantees ptr % A == 0, so it covers a need of N bytes exactly when
// N divides A: declared 16 covers 16, 8, 4, 2 and 1; declared 8 covers 8 but not 16; declared 24
// covers 8 but not 16. A non-positive declared value guarantees nothing.
//
// Per-tensor NEEDS (bytes). The kernels here are vendored AITER ASM code objects with no source in
// this tree and no documented alignment contract, so each need is the byte width of the row
// load/store the kernel is expected to issue (elements per lane x element size), not a
// disassembly result. Each is a single named constant so a gfx942/gfx950 exact-placement probe can
// confirm or relax it in one place.
//
// Forward (SdpaFwdPlanBuilder):
//   Q, O      16  16-byte (8 x bf16 / 16 x fp8) row loads of Q and row stores of O
//   K, V      16  16-byte (8 x bf16) row loads
//   K, V fp8   8  8-byte (8 x fp8) row loads; the fp8 kernels read K/V rows half as wide
//   LSE       16  UNCONFIRMED: written per Q row as fp32; assumed to be vector-stored
//   descales   4  fp32 scalar (SMEM s_load_dword reads need dword alignment) [INFERENCE]
// Backward (SdpaBwdPlanBuilder):
//   Q, K, V, O, dO, dQ, dK, dV  16  16-byte row loads/stores (odo, dqdkdv and dq_convert stages)
//   LSE (stats)                 16  UNCONFIRMED: fp32 per-row stats assumed vector-loaded
// The dq_acc and D buffers of the backward pass live in the plan workspace (64-byte aligned
// sub-allocations, SdpaKernelUtils.hpp), not in graph tensors, so they carry no declared alignment.
inline constexpr int64_t K_ROW_ALIGNMENT_BYTES = 16;
inline constexpr int64_t K_FP8_KV_ROW_ALIGNMENT_BYTES = 8;
inline constexpr int64_t K_STATS_ALIGNMENT_BYTES = 16;
inline constexpr int64_t K_DESCALE_ALIGNMENT_BYTES = 4;

// True when `tensor`'s declared alignment guarantees `needBytes`-byte alignment.
inline bool
    declaredAlignmentCovers(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor,
                            int64_t needBytes)
{
    const int64_t declared = tensor.alignment();
    return needBytes > 0 && declared > 0 && declared % needBytes == 0;
}

struct TensorAlignmentNeed
{
    const char* name;
    // nullptr marks an absent optional tensor; it is skipped.
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor;
    int64_t needBytes;
};

// The decline reason for the first listed tensor whose declared alignment does not cover its
// need, or nullopt when every present tensor is covered.
inline std::optional<std::string>
    findUnderAlignedTensor(std::initializer_list<TensorAlignmentNeed> needs)
{
    for(const auto& need : needs)
    {
        if(need.tensor != nullptr && !declaredAlignmentCovers(*need.tensor, need.needBytes))
        {
            return std::string{need.name} + " tensor declares alignment "
                   + std::to_string(need.tensor->alignment()) + " bytes, but the kernel needs "
                   + std::to_string(need.needBytes) + "-byte aligned pointers";
        }
    }
    return std::nullopt;
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

// The attention scale the kernels apply when the graph has no scale tensor:
// attn_scale_value, or 1.0 (no scaling, cuDNN's default) when it is unset.
template <typename SdpaAttrsT>
float attnScaleOrDefault(const SdpaAttrsT& attrs)
{
    return attrs.attn_scale_value().value_or(1.0f);
}

} // namespace plan_utils
} // namespace asm_sdpa_engine
