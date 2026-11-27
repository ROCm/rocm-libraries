// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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
 * and this engine's TestAttentionDenseStrides both run. It also holds the checks the predicate
 * cannot make, since it sees each operand alone and no address: at launch, the output must not
 * share bytes with an input (denseOperandSpanBytes and denseOutputOverlapsInput, the twins of the
 * same module's `_byte_span` and `_refuse_output_overlap`), and the LSE (stats) output must not
 * share bytes with Q, K, V, O or the additive bias (denseStatsOverlapsOperand, the twin of
 * `_lse_store._check_lse_buffer`). The vectors file has no rows for these checks, so each side's
 * own tests pin them; a change to one is made on both sides.
 *
 * Header-only and dependency-free, like AttentionDensePersistent.hpp: pure functions of
 * integers and strings, compiled with or without any engine gate (the FlyDSL helpers compile
 * only under HIPKERNELPROVIDER_ENABLE_FLYDSL, and this must not). Every product and sum below
 * goes through the compiler's overflow builtins, as sdpa_graph::checkedProduct does, and reports
 * "past the limit" instead of wrapping; none rejects a zero factor (a canonical stride may be
 * 0), which is why pack_features::checkedMultiply is not used. Nor is sdpa_graph::checkedProduct
 * itself: its header pulls in the flatbuffers data objects, and this one must not.
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

/// Bytes per element of every dense tensor (fp16 and bf16). The Python twin sizes K and V with
/// `dense_kv_elem_bytes`, which is 1 for the gfx950 fp8 K/V-cache spec. That spec has no twin
/// here: the graph match admits only BFLOAT16 and HALF operands of one data type, so no fp8 K/V
/// reaches this predicate. Serving one needs an element-size parameter and a vectors-file row.
inline constexpr int64_t DENSE_ELEM_BYTES = 2;
/// Bytes per element of the LSE (stats) output and of the additive bias: the graph match admits
/// only FLOAT for both.
inline constexpr int64_t DENSE_FP32_BYTES = 4;
/// Every stride of an axis with extent > 1 is a multiple of this many elements (16 bytes of
/// 2-byte elements). One conservative rule for every body: the kernels tolerate less on some
/// operands.
inline constexpr int64_t DENSE_STRIDE_MULTIPLE = 8;
/// Every operand's base address is a multiple of this many bytes (hipDNN's declared
/// alignment default).
inline constexpr int64_t DENSE_POINTER_ALIGN = 16;
/// K/V loads may reach this many tiles past the padded length: an empty KV segment still
/// loads tiles n_upper and n_upper + 1. This widens only the 2^31 check (denseKvSlabFits), not
/// a memory bound: the kernel bounds each K/V slab's buffer resource at the last valid row, so
/// those loads return 0 and read nothing past the view.
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

/// @p left * @p right, or nullopt past int64_t. Every caller passes non-negative values.
inline std::optional<int64_t> checkedMul(int64_t left, int64_t right)
{
    int64_t product = 0;
    return __builtin_mul_overflow(left, right, &product) ? std::nullopt
                                                         : std::optional<int64_t>(product);
}

/// @p left + @p right, or nullopt past int64_t. Every caller passes non-negative values.
inline std::optional<int64_t> checkedAdd(int64_t left, int64_t right)
{
    int64_t sum = 0;
    return __builtin_add_overflow(left, right, &sum) ? std::nullopt : std::optional<int64_t>(sum);
}

/// @p values as the Python twin prints a tuple: `(2, 8, 512, 128)`.
template <typename Values>
std::string tupleText(const Values& values)
{
    std::string text = "(";
    for(const int64_t value : values)
    {
        text += (text.size() > 1 ? ", " : "") + std::to_string(value);
    }
    return text + ")";
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
 * seqlen_q, K and V the key/value heads and seqlen_kv. The values are not limited to i32: the
 * engine's packedI32LaunchStrides is the form bounded to kernel arguments.
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
 * With elemBytes 1 it counts elements. One formula for the host predicate and the kernel's
 * buffer-resource bound, so the gate and `num_records` cannot diverge, but the two pass
 * different row counts: the kernel passes the valid rows (`seqlen_kv`), so `num_records` ends
 * at the last valid element and never passes the view; the host passes denseKvReachRows (the
 * padded length plus the slack) to bound the furthest offset a load can form. Not the unified
 * 3-D engine's gfx942KvHeadSpan: that kernel has its own element size and limits, and its
 * formula gives a slab of no rows no special answer.
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

/**
 * @brief The bytes operand @p tensor (0 = Q, 1 = K, 2 = V, 3 = O) of @p problem spans from its
 *        base address, one past its last element:
 *        `2 * ((B - 1) * sB + (H - 1) * sH + (S - 1) * sS + D)`, or nullopt past int64_t.
 *
 * rocKE's `_byte_span` (kernels/common/_attention_dense_strides.py) gives the same end,
 * `elem_bytes * (1 + sum((n - 1) * s))` over the view's four axes: the head-dim axis adds
 * `D - 1` at stride 1, and the 1 makes it `D`. Both start at the base address: @p strides are
 * canonical (canonicalDenseStrides), so none is negative and the base is the lowest address the
 * view reaches. Every extent here is at least 1; Python's twin answers None for an empty view.
 * The predicate holds this below 2^63 (SIZE_LIMIT), and the launch compares the output's span
 * with the inputs' (denseOutputOverlapsInput).
 */
inline std::optional<int64_t> denseOperandSpanBytes(const DenseStrideProblem& problem,
                                                    const AttentionDenseStrides& strides,
                                                    std::size_t tensor)
{
    using dense_strides_detail::checkedAdd;
    using dense_strides_detail::checkedMul;
    const auto dims = denseOperandDims(problem, tensor);
    const auto batchOffset = checkedMul(dims.at(0) - 1, strides.at(denseStrideIndex(tensor, 0)));
    const auto headOffset = checkedMul(dims.at(1) - 1, strides.at(denseStrideIndex(tensor, 1)));
    const auto elementOffset = batchOffset.has_value() && headOffset.has_value()
                                   ? checkedAdd(*batchOffset, *headOffset)
                                   : std::nullopt;
    const auto baseBytes
        = elementOffset.has_value() ? checkedMul(DENSE_ELEM_BYTES, *elementOffset) : std::nullopt;
    const auto slabBytes = denseSlabBytes(
        dims.at(2), strides.at(denseStrideIndex(tensor, 2)), problem.headSize, DENSE_ELEM_BYTES);
    return baseBytes.has_value() && slabBytes.has_value() ? checkedAdd(*baseBytes, *slabBytes)
                                                          : std::nullopt;
}

/// Whether the byte ranges [@p first, @p first + @p firstBytes) and [@p second, @p second +
/// @p secondBytes) share a byte. The test conv_aot::bufferRangesDoNotOverlap makes for each pair:
/// by the difference of the ordered addresses, so a range that ends past the address space does
/// not wrap. An empty range shares none.
constexpr bool denseByteRangesIntersect(std::uintptr_t first,
                                        int64_t firstBytes,
                                        std::uintptr_t second,
                                        int64_t secondBytes)
{
    if(firstBytes <= 0 || secondBytes <= 0)
    {
        return false;
    }
    return first <= second ? second - first < static_cast<std::uintptr_t>(firstBytes)
                           : first - second < static_cast<std::uintptr_t>(secondBytes);
}

/**
 * @brief The first input (0 = Q, 1 = K, 2 = V) whose bytes the output's view shares, for
 *        operands based at @p bases (Q, K, V, O) and addressed with @p strides over
 *        @p problem; nullopt when O shares none. A zero base (no buffer) is skipped.
 *
 * The kernels declare o_ptr noalias and write O while other workgroups still read Q, K and V,
 * so an output that is a view of an input (O cut out of a fused QKV buffer, or O passed as Q)
 * gives a wrong result that can change from run to run, with no fault. The inputs may share
 * storage with each other: the kernels only read them. Spans are compared, not elements, so an
 * output interleaved with an input without sharing an element is reported too (the
 * conservative answer). A span past int64_t, which the predicate never admits, counts as
 * shared.
 *
 * The twin of rocKE's `_refuse_output_overlap` (kernels/common/_attention_dense_strides.py),
 * which `admit_dense_launch_strides` asks before a Python runner launches: the same scope (O
 * against Q, K and V), the same span test, the same skip of an operand with no address. A change
 * to the operands checked, or to the span (denseOperandSpanBytes, `_byte_span`), is made on both
 * sides. The LSE output is held to the same rule against every other buffer, the bias included,
 * by denseStatsOverlapsOperand (Python's `_lse_store._check_lse_buffer`).
 */
inline std::optional<std::size_t>
    denseOutputOverlapsInput(const DenseStrideProblem& problem,
                             const AttentionDenseStrides& strides,
                             const std::array<std::uintptr_t, 4>& bases)
{
    constexpr std::size_t OUTPUT = 3;
    if(bases.at(OUTPUT) == 0)
    {
        return std::nullopt;
    }
    const auto outputBytes = denseOperandSpanBytes(problem, strides, OUTPUT);
    for(std::size_t input = 0; input < OUTPUT; ++input)
    {
        if(bases.at(input) == 0)
        {
            continue;
        }
        const auto inputBytes = denseOperandSpanBytes(problem, strides, input);
        if(!outputBytes.has_value() || !inputBytes.has_value()
           || denseByteRangesIntersect(
               bases.at(OUTPUT), *outputBytes, bases.at(input), *inputBytes))
        {
            return input;
        }
    }
    return std::nullopt;
}

/**
 * @brief The bytes the LSE (stats) output of @p problem spans from its base address:
 *        `4 * B * Hq * Sq`, or nullopt past int64_t.
 *
 * The engine serves only the packed FP32 [B, Hq, Sq, 1] stats tensor (the graph match's
 * isServableStatsTensor), and the kernel writes row (b, h, s) at element (b * Hq + h) * Sq + s,
 * so the span is the whole tensor. rocKE's `_check_lse_buffer` takes the same end for the
 * caller's contiguous buffer, `numel * element_size`.
 */
inline std::optional<int64_t> denseStatsSpanBytes(const DenseStrideProblem& problem)
{
    using dense_strides_detail::checkedMul;
    const auto rows = checkedMul(problem.batch, problem.numQueryHeads);
    const auto elements = rows.has_value() ? checkedMul(*rows, problem.seqLenQ) : std::nullopt;
    return elements.has_value() ? checkedMul(DENSE_FP32_BYTES, *elements) : std::nullopt;
}

/**
 * @brief The bytes the additive bias of @p problem spans from its base address, one past the
 *        last element the kernel reads:
 *        `4 * ((B - 1) * sB + (Hq - 1) * sH + (Sq - 1) * sS + Skv)`, or nullopt past int64_t.
 *
 * The kernel reads the bias as FP32 [B|1, Hq|1, Sq|1, Skv] with a unit-stride key axis, and
 * steps the leading axes by @p biasStrides (batch, query head, query row, in elements, none
 * negative). A broadcast axis has stride 0, so the problem's full extents give the exact end for
 * it too. rocKE's `_check_lse_buffer` takes the same end for the bias tensor:
 * `1 + sum((n - 1) * s)` elements over its own extents and strides.
 */
inline std::optional<int64_t> denseBiasSpanBytes(const DenseStrideProblem& problem,
                                                 const std::array<int64_t, 3>& biasStrides)
{
    using dense_strides_detail::checkedAdd;
    using dense_strides_detail::checkedMul;
    const std::array<int64_t, 3> extents{problem.batch, problem.numQueryHeads, problem.seqLenQ};
    std::optional<int64_t> elements = problem.seqLenKv;
    for(std::size_t axis = 0; axis < extents.size() && elements.has_value(); ++axis)
    {
        const auto reach = checkedMul(extents.at(axis) - 1, biasStrides.at(axis));
        elements = reach.has_value() ? checkedAdd(*elements, *reach) : std::nullopt;
    }
    return elements.has_value() ? checkedMul(DENSE_FP32_BYTES, *elements) : std::nullopt;
}

/**
 * @brief The first buffer whose bytes the LSE (stats) output at @p lseBase shares: 0 = Q,
 *        1 = K, 2 = V, 3 = O (based at @p bases and addressed with @p strides over @p problem),
 *        4 = the additive bias at @p biasBase, read with @p biasStrides; nullopt when it shares
 *        none.
 *
 * The kernel writes each row's LSE while other workgroups still read Q, K, V and the bias and
 * write O, so a stats buffer that shares their bytes gives a wrong result with no fault, as an
 * output inside an input does (denseOutputOverlapsInput). The spans compared are
 * denseStatsSpanBytes, denseOperandSpanBytes and denseBiasSpanBytes, not elements, and a span past
 * int64_t counts as shared. A zero base names no buffer and is skipped: a zero @p lseBase shares
 * nothing, and the bias is compared only when @p biasBase is nonzero and @p biasStrides holds its
 * strides (a launch whose kernel reads no bias passes nullopt).
 *
 * The twin of rocKE's `_lse_store._check_lse_buffer`, which `validate_dense_lse` asks before a
 * Python runner launches a kernel that writes LSE: the LSE against q, k, v, out and the optional
 * tensors, the bias among them. The dense engines serve none of the other optional tensors
 * (sinks, block tables, varlen offsets). A change to the buffers compared or to a span is made on
 * both sides.
 */
inline std::optional<std::size_t>
    denseStatsOverlapsOperand(const DenseStrideProblem& problem,
                              const AttentionDenseStrides& strides,
                              const std::array<std::uintptr_t, 4>& bases,
                              std::uintptr_t lseBase,
                              std::uintptr_t biasBase,
                              const std::optional<std::array<int64_t, 3>>& biasStrides)
{
    constexpr std::size_t BIAS = 4;
    if(lseBase == 0)
    {
        return std::nullopt;
    }
    const auto lseBytes = denseStatsSpanBytes(problem);
    const auto shares
        = [&lseBytes, lseBase](std::uintptr_t base, const std::optional<int64_t>& bytes) {
              return !lseBytes.has_value() || !bytes.has_value()
                     || denseByteRangesIntersect(lseBase, *lseBytes, base, *bytes);
          };
    for(std::size_t operand = 0; operand < bases.size(); ++operand)
    {
        if(bases.at(operand) != 0
           && shares(bases.at(operand), denseOperandSpanBytes(problem, strides, operand)))
        {
            return operand;
        }
    }
    if(biasBase != 0 && biasStrides.has_value()
       && shares(biasBase, denseBiasSpanBytes(problem, *biasStrides)))
    {
        return BIAS;
    }
    return std::nullopt;
}

namespace dense_strides_detail
{

/// No two index tuples of the view share an element: walking the axes of extent > 1 by
/// increasing stride, each stride clears the span of the axes before it. A span past int64_t
/// is past every stride, so a further axis then overlaps. Not sdpa_graph::elementsDoNotOverlap:
/// that one also declines when the last axis's span passes int64_t, and this one is pinned to
/// rocKE's predicate by the shared vectors file (such a tensor is declined here by SIZE_LIMIT).
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
 *    denseQoSlabFits), and the operand's span (denseOperandSpanBytes: the byte offset of the
 *    last slab's base plus its S rows) is below 2^63.
 *
 * Extent-1 axes are exempt and canonicalized (canonicalDenseStrides); an admitted verdict
 * carries the canonical values. The whole-tensor 2^31 preflight of the spec is a separate
 * check the engine keeps beside this one. Every product here is overflow-checked, and one
 * that passes int64_t is past the limit it is compared with.
 */
inline DenseStrideVerdict denseStridesAdmit(const DenseStrideProblem& problem,
                                            const std::array<DenseOperand, 4>& operands)
{
    using dense_strides_detail::decline;
    using dense_strides_detail::tupleText;
    constexpr std::array<std::string_view, 4> ROLES{"q", "k", "v", "o"};
    constexpr std::array<std::string_view, 3> AXES{"batch", "head", "token"};
    const int64_t headSize = problem.headSize;

    for(std::size_t tensor = 0; tensor < 4; ++tensor)
    {
        const auto& operand = operands.at(tensor);
        const std::string role(ROLES.at(tensor));
        if(operand.dims.size() != 4)
        {
            return decline(DenseStrideReason::OPERAND,
                           role + " has rank " + std::to_string(operand.dims.size()) + ", not 4");
        }
        if(operand.strides.size() != 4)
        {
            return decline(DenseStrideReason::OPERAND,
                           role + " has " + std::to_string(operand.strides.size())
                               + " strides for its 4 dims, not 4");
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
                           role + " dims (B, H, S, D) " + tupleText(operand.dims)
                               + " differ from the problem's " + tupleText(want));
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
                           std::string(ROLES.at(tensor)) + " dims " + tupleText(operand.dims)
                               + " with strides " + tupleText(operand.strides)
                               + " overlap themselves");
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
        // The checked sums stop at INT64_MAX = 2^63 - 1, so a span that is not below 2^63 has
        // no value.
        if(!denseOperandSpanBytes(problem, canonical, tensor).has_value())
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
