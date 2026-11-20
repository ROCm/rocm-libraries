// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/**
 * @file AttentionDenseStrides.hpp
 * @brief The launch-time strides of the dense-attention kernels: the twelve stride kernel
 *        arguments, and the host predicate that decides which Q, K, V and O views a launch
 *        may pass them for.
 *
 * Every dense kernel of both catalogs takes twelve trailing i32 arguments, a (batch, head,
 * token) element-stride triple for each of Q, K, V and O. This header restates the
 * rocKE Python that owns them, `rocke/library/kernels/common/_attention_dense_strides.py`
 * (`dense_strides_admit`, `canonical_dense_strides`, `packed_launch_strides`,
 * `dense_slab_bytes`, `dense_kv_reach_rows`), check for check and in the same order, with the
 * same decline reasons. Both are pinned to one vectors file
 * (`rocke/platform/tests/golden/attention_dense_stride_vectors.json`), which the Python test
 * and this engine's TestAttentionDenseStrides both run.
 *
 * Header-only and dependency-free, like AttentionDensePersistent.hpp: pure functions of
 * integers and strings, compiled with or without any engine gate (the FlyDSL helpers compile
 * only under HIPKERNELPROVIDER_ENABLE_FLYDSL, and this must not). Its arithmetic is its own:
 * every product and sum below goes through an overflow-checked step that reports "past the
 * limit" instead of wrapping, and none rejects a zero factor (a canonical stride may be 0),
 * which is why pack_features::checkedMultiply is not used.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The number of stride kernel arguments every dense kernel takes last
/// (attention_dense::DENSE_STRIDE_NAMES).
inline constexpr std::size_t ATTENTION_DENSE_STRIDE_COUNT = 12;

/// Element strides of Q, K, V and O, in attention_dense::DENSE_STRIDE_NAMES order.
using AttentionDenseStrides = std::array<int64_t, ATTENTION_DENSE_STRIDE_COUNT>;

namespace attention_dense
{

/// The twelve stride kernel arguments, in kernarg order: for Q, K, V and O in turn, the
/// batch, head and token stride in elements, each an i32. Every kernel of both catalogs takes
/// them last, after every other argument; the head-dim stride is 1 and is not an argument.
/// This one list names the kernel arguments (kernelSignature), orders the launch
/// (AttentionDenseKernelArguments) and names the strided contract's saved values. The rocKE
/// builders declare the same names in the same order
/// (`kernels/common/_attention_dense_strides.py`, DENSE_STRIDE_NAMES).
inline constexpr std::array<std::string_view, ATTENTION_DENSE_STRIDE_COUNT> DENSE_STRIDE_NAMES{
    "stride_q_batch",
    "stride_q_head",
    "stride_q_token",
    "stride_k_batch",
    "stride_k_head",
    "stride_k_token",
    "stride_v_batch",
    "stride_v_head",
    "stride_v_token",
    "stride_o_batch",
    "stride_o_head",
    "stride_o_token"};

/// Bytes per element of every dense tensor (fp16 and bf16).
inline constexpr int64_t DENSE_ELEM_BYTES = 2;
/// Every stride of an axis with extent > 1 is a multiple of this many elements (16 bytes).
/// One conservative rule for every body: the kernels tolerate less on some operands.
inline constexpr int64_t DENSE_STRIDE_MULTIPLE = 8;
/// Every operand's base address is a multiple of this many bytes (hipDNN's declared
/// alignment default).
inline constexpr int64_t DENSE_POINTER_ALIGN = 16;
/// K/V loads may reach this many tiles past the padded length: an empty KV segment still
/// loads tiles n_upper and n_upper + 1.
inline constexpr int64_t DENSE_SLAB_SLACK_TILES = 2;
/// The kernel's 32-bit index limit: strides, K/V byte offsets and Q/O element offsets stay
/// below it.
inline constexpr int64_t DENSE_INT32_LIMIT = int64_t{1} << 31;

/// Why denseStridesAdmit declined, in the order its checks run.
enum class DenseStrideReason
{
    /// Not a rank-4 tensor of the problem's (B, H, S, D); virtual; or carrying a ragged offset.
    OPERAND,
    /// The head-dim stride is not 1.
    HEAD_DIM_STRIDE,
    /// A stride of an axis of extent > 1 is not positive, not a multiple of 8 elements, or a
    /// token stride is below the head size.
    STRIDE_ALIGNMENT,
    /// The operand's view overlaps itself.
    OVERLAP,
    /// The base address is not a multiple of 16 bytes.
    POINTER_ALIGNMENT,
    /// A stride is 2^31 or more, a slab passes the kernel's 32-bit offsets, or a slab base
    /// passes 64 bits.
    SIZE_LIMIT
};

/// The reason tokens, spelled as the Python and the vectors file spell them.
inline constexpr std::array<std::string_view, 6> DENSE_STRIDE_REASONS{
    "operand", "head_dim_stride", "stride_alignment", "overlap", "pointer_alignment", "size_limit"};

constexpr std::string_view denseStrideReasonToken(DenseStrideReason reason)
{
    return DENSE_STRIDE_REASONS.at(static_cast<std::size_t>(reason));
}

/// The facts of one launch the predicate needs: the problem's extents and the tile the
/// candidate kernel is built with (block_n sets how far a K/V load may reach past the data).
struct DenseStrideProblem
{
    int64_t batch = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t headSize = 0;
    int64_t blockN = 0;
};

/// One dense tensor as the predicate sees it. `dims` is (B, H, S, D) and `strides`
/// (batch, head, token, head-dim) in elements, as the tensor declares them; any length, so
/// a tensor of another rank is declined, not mis-read. `align` is the byte alignment its base
/// address is declared to have.
struct DenseOperand
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    int64_t align = DENSE_POINTER_ALIGN;
    bool isVirtual = false;
    bool isRagged = false;
};

/// The answer of denseStridesAdmit: admitted with the canonical strides to launch with, or
/// declined with the first failing check's reason and a sentence naming the operand.
struct DenseStrideVerdict
{
    bool ok = false;
    DenseStrideReason reason = DenseStrideReason::OPERAND;
    std::string detail;
    AttentionDenseStrides strides{};
};

namespace dense_strides_detail
{

/// @p left * @p right of two non-negative values, or nullopt past int64_t.
inline std::optional<int64_t> checkedMul(int64_t left, int64_t right)
{
    if(left == 0 || right == 0)
    {
        return int64_t{0};
    }
    if(left > std::numeric_limits<int64_t>::max() / right)
    {
        return std::nullopt;
    }
    return left * right;
}

/// @p left + @p right of two non-negative values, or nullopt past int64_t.
inline std::optional<int64_t> checkedAdd(int64_t left, int64_t right)
{
    if(left > std::numeric_limits<int64_t>::max() - right)
    {
        return std::nullopt;
    }
    return left + right;
}

inline DenseStrideVerdict decline(DenseStrideReason reason, std::string detail)
{
    DenseStrideVerdict verdict;
    verdict.reason = reason;
    verdict.detail = std::move(detail);
    return verdict;
}

} // namespace dense_strides_detail

/// The stride index of tensor @p tensor (0 = Q, 1 = K, 2 = V, 3 = O) and axis @p axis
/// (0 = batch, 1 = head, 2 = token) in a DENSE_STRIDE_NAMES-ordered array.
constexpr std::size_t denseStrideIndex(std::size_t tensor, std::size_t axis)
{
    return tensor * 3 + axis;
}

/// The (B, H, S, D) extents of operand @p tensor (0 = Q, 1 = K, 2 = V, 3 = O) of @p problem.
inline std::array<int64_t, 4> denseOperandDims(const DenseStrideProblem& problem,
                                               std::size_t tensor)
{
    const bool queryShaped = tensor == 0 || tensor == 3;
    return {problem.batch,
            queryShaped ? problem.numQueryHeads : problem.numKvHeads,
            queryShaped ? problem.seqLenQ : problem.seqLenKv,
            problem.headSize};
}

/**
 * @brief @p raw with every extent-1 axis canonicalized: batch and head to 0, token to the
 *        head size.
 *
 * An extent-1 axis is never stepped, so its stride carries no information and may be
 * anything; the kernel is launched with these values whatever the graph declared.
 */
inline AttentionDenseStrides canonicalDenseStrides(const DenseStrideProblem& problem,
                                                   const AttentionDenseStrides& raw)
{
    AttentionDenseStrides canonical{};
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        const auto dims = denseOperandDims(problem, tensor);
        for(std::size_t axis = 0; axis < 3; ++axis)
        {
            const auto index = denseStrideIndex(tensor, axis);
            if(dims.at(axis) == 1)
            {
                canonical.at(index) = axis == 2 ? problem.headSize : int64_t{0};
            }
            else
            {
                canonical.at(index) = raw.at(index);
            }
        }
    }
    return canonical;
}

/**
 * @brief The twelve values a launch passes for packed BSHD tensors of @p problem, in
 *        canonical form (rocKE's `packed_launch_strides`), or nullopt when one passes int64_t.
 *
 * For a tensor of H heads, S tokens and head size D stored token-major with the head fastest:
 * batch stride S*H*D, head stride D, token stride H*D. Q and O take the query heads and
 * seqlen_q, K and V the key/value heads and seqlen_kv. The values are not limited to i32.
 */
inline std::optional<AttentionDenseStrides> packedLaunchStrides(const DenseStrideProblem& problem)
{
    using dense_strides_detail::checkedMul;
    AttentionDenseStrides packed{};
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        const auto dims = denseOperandDims(problem, tensor);
        const auto tokenStride = checkedMul(dims.at(1), problem.headSize);
        const auto batchStride
            = tokenStride.has_value() ? checkedMul(dims.at(2), *tokenStride) : std::nullopt;
        if(!tokenStride.has_value() || !batchStride.has_value())
        {
            return std::nullopt;
        }
        packed.at(denseStrideIndex(tensor, 0)) = *batchStride;
        packed.at(denseStrideIndex(tensor, 1)) = problem.headSize;
        packed.at(denseStrideIndex(tensor, 2)) = *tokenStride;
    }
    return canonicalDenseStrides(problem, packed);
}

/**
 * @brief The extent of the first @p rows rows of one (batch, head) slab, in units of
 *        @p elemBytes: `rows > 0 ? elemBytes * ((rows - 1) * tokenStride + headSize) : 0`
 *        (rocKE's `dense_slab_bytes`), or nullopt when it passes int64_t.
 *
 * With elemBytes 1 it counts elements. One function for the host predicate and the kernel's
 * buffer-resource bound, so the gate and `num_records` cannot diverge.
 */
inline std::optional<int64_t>
    denseSlabBytes(int64_t rows, int64_t tokenStride, int64_t headSize, int64_t elemBytes)
{
    using dense_strides_detail::checkedAdd;
    using dense_strides_detail::checkedMul;
    if(rows <= 0)
    {
        return int64_t{0};
    }
    const auto reach = checkedMul(rows - 1, tokenStride);
    const auto last = reach.has_value() ? checkedAdd(*reach, headSize) : std::nullopt;
    return last.has_value() ? checkedMul(*last, elemBytes) : std::nullopt;
}

/// The rows of a K/V slab a load may address (rocKE's `dense_kv_reach_rows`): the length
/// padded up to a tile of @p blockN rows, plus DENSE_SLAB_SLACK_TILES more tiles. Nullopt
/// past int64_t or for a non-positive @p blockN.
inline std::optional<int64_t> denseKvReachRows(int64_t seqLenKv, int64_t blockN)
{
    using dense_strides_detail::checkedAdd;
    using dense_strides_detail::checkedMul;
    if(seqLenKv < 0 || blockN <= 0)
    {
        return std::nullopt;
    }
    const int64_t remainder = seqLenKv % blockN;
    const auto padded = remainder == 0 ? std::optional<int64_t>(seqLenKv)
                                       : checkedAdd(seqLenKv, blockN - remainder);
    const auto slack = checkedMul(DENSE_SLAB_SLACK_TILES, blockN);
    return padded.has_value() && slack.has_value() ? checkedAdd(*padded, *slack) : std::nullopt;
}

/// Whether every offset a K or V load may form, up to the furthest element of
/// denseKvReachRows, starts below 2^31 bytes (the buffer resource's i32 byte offset).
inline bool denseKvSlabFits(int64_t seqLenKv, int64_t tokenStride, int64_t headSize, int64_t blockN)
{
    const auto rows = denseKvReachRows(seqLenKv, blockN);
    if(!rows.has_value())
    {
        return false;
    }
    const auto reach = denseSlabBytes(*rows, tokenStride, headSize, DENSE_ELEM_BYTES);
    return reach.has_value() && *reach - DENSE_ELEM_BYTES < DENSE_INT32_LIMIT;
}

/// Whether a Q or O slab (a 64-bit base, then a 32-bit element index) holds fewer than 2^31
/// elements: `(seqLenQ - 1) * tokenStride + headSize < 2^31`.
inline bool denseQoSlabFits(int64_t seqLenQ, int64_t tokenStride, int64_t headSize)
{
    const auto elements = denseSlabBytes(seqLenQ, tokenStride, headSize, 1);
    return elements.has_value() && *elements < DENSE_INT32_LIMIT;
}

namespace dense_strides_detail
{

/// No two index tuples of the view share an element: walking the axes of extent > 1 by
/// increasing stride, each stride clears the span of the axes before it. A span past int64_t
/// is past every stride, so a further axis then overlaps.
inline bool overlapFree(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    std::vector<std::pair<int64_t, int64_t>> axes;
    for(std::size_t axis = 0; axis < dims.size(); ++axis)
    {
        if(dims.at(axis) > 1)
        {
            axes.emplace_back(strides.at(axis), dims.at(axis));
        }
    }
    std::sort(axes.begin(), axes.end());
    std::optional<int64_t> span = int64_t{1};
    for(const auto& [stride, extent] : axes)
    {
        if(!span.has_value() || stride < *span)
        {
            return false;
        }
        const auto step = checkedMul(extent - 1, stride);
        span = step.has_value() ? checkedAdd(*span, *step) : std::nullopt;
    }
    return true;
}

} // namespace dense_strides_detail

/**
 * @brief Admits the four dense operands (Q, K, V, O, in that order) for @p problem, or
 *        declines with the first failing check. rocKE's `dense_strides_admit`.
 *
 * The checks run in the order of DenseStrideReason; inside a check the operands go Q, K, V,
 * O:
 *  - OPERAND: rank 4, dims equal to the problem's (B, H, S, D), not virtual, no ragged offset;
 *  - HEAD_DIM_STRIDE: the head-dim stride is exactly 1;
 *  - STRIDE_ALIGNMENT: on every axis of extent > 1 the stride is positive and a multiple of
 *    DENSE_STRIDE_MULTIPLE elements, and the token stride is at least the head size (zero and
 *    negative strides are declined);
 *  - OVERLAP: no operand overlaps itself;
 *  - POINTER_ALIGNMENT: the base is a multiple of DENSE_POINTER_ALIGN bytes;
 *  - SIZE_LIMIT: every canonical stride is below 2^31, the slab offsets fit (denseKvSlabFits,
 *    denseQoSlabFits), and the byte offset of the last slab's base plus its S rows,
 *    `2 * ((B - 1) * sB + (H - 1) * sH) + 2 * ((S - 1) * sS + D)`, is below 2^63.
 *
 * Extent-1 axes are exempt and canonicalized (canonicalDenseStrides); an admitted verdict
 * carries the canonical values. The whole-tensor 2^31 preflight of the spec is a separate
 * check the engine keeps beside this one. Every product here is overflow-checked, and one
 * that passes int64_t is past the limit it is compared with.
 */
inline DenseStrideVerdict denseStridesAdmit(const DenseStrideProblem& problem,
                                            const std::array<DenseOperand, 4>& operands)
{
    using dense_strides_detail::checkedAdd;
    using dense_strides_detail::checkedMul;
    using dense_strides_detail::decline;
    constexpr std::array<std::string_view, 4> ROLES{"q", "k", "v", "o"};
    constexpr std::array<std::string_view, 3> AXES{"batch", "head", "token"};
    const int64_t headSize = problem.headSize;

    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        const auto& operand = operands.at(tensor);
        const std::string role(ROLES.at(tensor));
        if(operand.dims.size() != 4 || operand.strides.size() != 4)
        {
            return decline(DenseStrideReason::OPERAND,
                           role + " has rank " + std::to_string(operand.dims.size()) + ", not 4");
        }
        if(operand.isVirtual)
        {
            return decline(DenseStrideReason::OPERAND, role + " is a virtual tensor");
        }
        if(operand.isRagged)
        {
            return decline(DenseStrideReason::OPERAND, role + " carries a ragged offset");
        }
        if(std::any_of(
               operand.dims.begin(), operand.dims.end(), [](int64_t dim) { return dim <= 0; }))
        {
            return decline(DenseStrideReason::OPERAND, role + " has a non-positive extent");
        }
        const auto want = denseOperandDims(problem, tensor);
        if(!std::equal(want.begin(), want.end(), operand.dims.begin()))
        {
            return decline(DenseStrideReason::OPERAND,
                           role + " dims (B, H, S, D) differ from the problem's");
        }
    }
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        const auto headDimStride = operands.at(tensor).strides.at(3);
        if(headDimStride != 1)
        {
            return decline(DenseStrideReason::HEAD_DIM_STRIDE,
                           std::string(ROLES.at(tensor)) + " head-dim stride is "
                               + std::to_string(headDimStride) + ", not 1");
        }
    }
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        const auto& operand = operands.at(tensor);
        const std::string role(ROLES.at(tensor));
        for(std::size_t axis = 0; axis < 3; ++axis)
        {
            const int64_t extent = operand.dims.at(axis);
            const int64_t stride = operand.strides.at(axis);
            if(extent == 1)
            {
                continue;
            }
            const std::string where
                = role + " " + std::string(AXES.at(axis)) + " stride " + std::to_string(stride);
            if(stride <= 0)
            {
                return decline(DenseStrideReason::STRIDE_ALIGNMENT,
                               where + " is not positive (extent " + std::to_string(extent) + ")");
            }
            if(stride % DENSE_STRIDE_MULTIPLE != 0)
            {
                return decline(DenseStrideReason::STRIDE_ALIGNMENT,
                               where + " is not a multiple of "
                                   + std::to_string(DENSE_STRIDE_MULTIPLE) + " elements");
            }
            if(axis == 2 && stride < headSize)
            {
                return decline(DenseStrideReason::STRIDE_ALIGNMENT,
                               where + " is below the head size " + std::to_string(headSize));
            }
        }
    }
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        const auto& operand = operands.at(tensor);
        if(!dense_strides_detail::overlapFree(operand.dims, operand.strides))
        {
            return decline(DenseStrideReason::OVERLAP,
                           std::string(ROLES.at(tensor)) + " overlaps itself");
        }
    }
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        const auto align = operands.at(tensor).align;
        if(align % DENSE_POINTER_ALIGN != 0)
        {
            return decline(DenseStrideReason::POINTER_ALIGNMENT,
                           std::string(ROLES.at(tensor)) + " base is " + std::to_string(align)
                               + "-byte aligned, needs " + std::to_string(DENSE_POINTER_ALIGN));
        }
    }

    AttentionDenseStrides raw{};
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        for(std::size_t axis = 0; axis < 3; ++axis)
        {
            raw.at(denseStrideIndex(tensor, axis)) = operands.at(tensor).strides.at(axis);
        }
    }
    const auto canonical = canonicalDenseStrides(problem, raw);
    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        const std::string role(ROLES.at(tensor));
        for(std::size_t axis = 0; axis < 3; ++axis)
        {
            const int64_t stride = canonical.at(denseStrideIndex(tensor, axis));
            if(stride >= DENSE_INT32_LIMIT)
            {
                return decline(DenseStrideReason::SIZE_LIMIT,
                               role + " " + std::string(AXES.at(axis)) + " stride "
                                   + std::to_string(stride) + " is not below 2^31");
            }
        }
        const int64_t batchStride = canonical.at(denseStrideIndex(tensor, 0));
        const int64_t headStride = canonical.at(denseStrideIndex(tensor, 1));
        const int64_t tokenStride = canonical.at(denseStrideIndex(tensor, 2));
        const bool isKeyValue = tensor == 1 || tensor == 2;
        const bool fits
            = isKeyValue ? denseKvSlabFits(problem.seqLenKv, tokenStride, headSize, problem.blockN)
                         : denseQoSlabFits(problem.seqLenQ, tokenStride, headSize);
        if(!fits)
        {
            return decline(DenseStrideReason::SIZE_LIMIT,
                           role + " slab with token stride " + std::to_string(tokenStride)
                               + " passes the 32-bit offset limit");
        }
        const auto dims = denseOperandDims(problem, tensor);
        const auto batchOffset = checkedMul(dims.at(0) - 1, batchStride);
        const auto headOffset = checkedMul(dims.at(1) - 1, headStride);
        const auto elementOffset = batchOffset.has_value() && headOffset.has_value()
                                       ? checkedAdd(*batchOffset, *headOffset)
                                       : std::nullopt;
        const auto baseBytes = elementOffset.has_value()
                                   ? checkedMul(DENSE_ELEM_BYTES, *elementOffset)
                                   : std::nullopt;
        const auto slabBytes = denseSlabBytes(dims.at(2), tokenStride, headSize, DENSE_ELEM_BYTES);
        const auto lastByte = baseBytes.has_value() && slabBytes.has_value()
                                  ? checkedAdd(*baseBytes, *slabBytes)
                                  : std::nullopt;
        // checkedAdd stops at INT64_MAX = 2^63 - 1, so a sum that is not below 2^63 has no value.
        if(!lastByte.has_value())
        {
            return decline(DenseStrideReason::SIZE_LIMIT,
                           role + " slab base offset passes 64 bits");
        }
    }
    DenseStrideVerdict verdict;
    verdict.ok = true;
    verdict.strides = canonical;
    return verdict;
}

} // namespace attention_dense
} // namespace hip_kernel_provider::kernel_ingestor_engine
