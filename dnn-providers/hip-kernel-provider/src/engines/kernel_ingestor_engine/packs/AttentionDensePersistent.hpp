// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file AttentionDensePersistent.hpp
 * @brief The launch grids of the dense-attention kernel past the default one, restated from
 *        rocKE's shared work-item decodes: the one-CTA-per-block grid at its other block
 *        orders, and the persistent grid with its fast-division kernargs.
 *
 * The Python is `rocke/library/kernels/common/attention_dense_decode.py`
 * (`NONPERSIST_DECODES`, `PERSIST_DECODES`, `fastdiv_kernargs`) and
 * `rocke/platform/python/rocke/helpers/magic_division.py` (`calculate_magic_numbers`, whose
 * C twin is `rocke/platform/cpp/helpers/transforms.cpp`). Both arch bodies call the same
 * decodes, so this header is arch-neutral; only the gfx942 catalog ships these grids.
 *
 * A decode maps a CTA's work index to the (query block, query head, batch) it computes, and
 * each is a bijection onto the work space only where its legality holds. hq_minor_swz at a
 * head count that is not a multiple of the XCD count drops and repeats work items with a
 * success status, so the matcher, prepare() and restore all refuse such a launch here. A
 * persistent grid other than exactly num_persistent CTAs skips or repeats work items too,
 * so its CTA count comes only from the candidate's baked value.
 *
 * Header-only and dependency-free: pure functions of strings and integers, testable with
 * no HIP context.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::attention_dense
{

/// The XCDs the hardware round-robins workgroups over (`chiplet_num_xcds`, at its default in
/// every shipped spec). hq_minor_swz gives each one a band of Hq / this many heads.
inline constexpr int64_t CHIPLET_NUM_XCDS = 8;

/// The persistent decode's fast division is exact for dividends below 2^30
/// (`FASTDIV_DIVIDEND_LIMIT`): the mul-hi plus dividend add is emitted nsw.
inline constexpr int64_t FASTDIV_DIVIDEND_LIMIT = int64_t{1} << 30;

/// The block order of the one-CTA-per-block grid, or the persistent grid. Spelled as the
/// saved value `grid_order` and, off the persistent grid, as the KMD `nonpersist_decode`.
enum class AttentionDenseGridOrder
{
    QB_MINOR,
    BT_HKV_MINOR,
    HQ_MINOR_SWZ,
    PERSISTENT
};

/// The persistent grid's work-item decodes that take the shape at runtime. `auto` resolves
/// from the build shape and gqa_pair / gqa_pair_2phase bake it, so neither is one.
enum class AttentionDensePersistDecode
{
    QB_MAJOR,
    HKV_MAJOR,
    BT_HKV_MINOR,
    HQ_MINOR_SWZ
};

/// The extents a persistent decode divides by in its grid-stride loop.
enum class AttentionDenseFastDivDivisor
{
    BATCH,
    NUM_QUERY_HEADS,
    GQA,
    NQB
};

/// The grid a candidate is built for. `decode` and `numPersistent` are read only on the
/// persistent grid.
struct AttentionDenseGrid
{
    AttentionDenseGridOrder order = AttentionDenseGridOrder::QB_MINOR;
    AttentionDensePersistDecode decode = AttentionDensePersistDecode::QB_MAJOR;
    int64_t numPersistent = 0;

    bool persistent() const
    {
        return order == AttentionDenseGridOrder::PERSISTENT;
    }
};

/// The launch shape a grid is sized and judged against.
struct AttentionDenseGridShape
{
    int64_t batch = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t blockM = 0;
    bool causal = false;
    int64_t slidingWindow = 0;
};

inline std::string_view gridOrderName(AttentionDenseGridOrder order)
{
    switch(order)
    {
    case AttentionDenseGridOrder::QB_MINOR:
        return "qb_minor";
    case AttentionDenseGridOrder::BT_HKV_MINOR:
        return "bt_hkv_minor";
    case AttentionDenseGridOrder::HQ_MINOR_SWZ:
        return "hq_minor_swz";
    case AttentionDenseGridOrder::PERSISTENT:
        return "persistent";
    default:
        break;
    }
    return "";
}

inline std::string_view persistDecodeName(AttentionDensePersistDecode decode)
{
    switch(decode)
    {
    case AttentionDensePersistDecode::QB_MAJOR:
        return "qb_major";
    case AttentionDensePersistDecode::HKV_MAJOR:
        return "hkv_major";
    case AttentionDensePersistDecode::BT_HKV_MINOR:
        return "bt_hkv_minor";
    case AttentionDensePersistDecode::HQ_MINOR_SWZ:
        return "hq_minor_swz";
    default:
        break;
    }
    return "";
}

/// The kernarg name prefix of a divisor: `<name>_magic`, `<name>_shift`.
inline std::string_view divisorName(AttentionDenseFastDivDivisor divisor)
{
    switch(divisor)
    {
    case AttentionDenseFastDivDivisor::BATCH:
        return "batch";
    case AttentionDenseFastDivDivisor::NUM_QUERY_HEADS:
        return "num_query_heads";
    case AttentionDenseFastDivDivisor::GQA:
        return "gqa";
    case AttentionDenseFastDivDivisor::NQB:
        return "nqb";
    default:
        break;
    }
    return "";
}

/// A `grid_order`: a non-persistent block order, or `persistent`.
inline std::optional<AttentionDenseGridOrder> parseGridOrder(std::string_view name)
{
    for(const auto order : {AttentionDenseGridOrder::QB_MINOR,
                            AttentionDenseGridOrder::BT_HKV_MINOR,
                            AttentionDenseGridOrder::HQ_MINOR_SWZ,
                            AttentionDenseGridOrder::PERSISTENT})
    {
        if(name == gridOrderName(order))
        {
            return order;
        }
    }
    return std::nullopt;
}

/// A `nonpersist_decode` (`NONPERSIST_DECODES`); `auto` is refused, as it resolves from the
/// build shape.
inline std::optional<AttentionDenseGridOrder> parseNonpersistDecode(std::string_view name)
{
    const auto order = parseGridOrder(name);
    if(!order.has_value() || *order == AttentionDenseGridOrder::PERSISTENT)
    {
        return std::nullopt;
    }
    return order;
}

/// A `persist_decode` with fast-division divisors; `auto`, gqa_pair and gqa_pair_2phase
/// are refused.
inline std::optional<AttentionDensePersistDecode> parsePersistDecode(std::string_view name)
{
    for(const auto decode : {AttentionDensePersistDecode::QB_MAJOR,
                             AttentionDensePersistDecode::HKV_MAJOR,
                             AttentionDensePersistDecode::BT_HKV_MINOR,
                             AttentionDensePersistDecode::HQ_MINOR_SWZ})
    {
        if(name == persistDecodeName(decode))
        {
            return decode;
        }
    }
    return std::nullopt;
}

/// `PersistDecode.fastdiv_divisors`, in kernarg order.
inline std::vector<AttentionDenseFastDivDivisor> fastDivDivisors(AttentionDensePersistDecode decode)
{
    using D = AttentionDenseFastDivDivisor;
    switch(decode)
    {
    case AttentionDensePersistDecode::QB_MAJOR:
        return {D::BATCH, D::NUM_QUERY_HEADS, D::GQA};
    case AttentionDensePersistDecode::HKV_MAJOR:
        return {D::BATCH, D::GQA, D::NQB};
    case AttentionDensePersistDecode::BT_HKV_MINOR:
        return {D::BATCH, D::NUM_QUERY_HEADS};
    case AttentionDensePersistDecode::HQ_MINOR_SWZ:
        return {D::NQB, D::NUM_QUERY_HEADS, D::GQA};
    default:
        break;
    }
    return {};
}

/// The magic multiplier (its uint32 bit pattern as an i32) and shift of one divisor.
struct AttentionDenseFastDivPair
{
    int32_t magic = 0;
    int32_t shift = 0;

    friend bool operator==(const AttentionDenseFastDivPair& a, const AttentionDenseFastDivPair& b)
    {
        return a.magic == b.magic && a.shift == b.shift;
    }
};

/**
 * @brief `calculate_magic_numbers(divisor)`, with the magic as the two's-complement i32 the
 *        kernel's unsigned mul-hi reads.
 *
 *     shift = smallest s with 2^s >= divisor
 *     magic = ((2^shift - divisor) << 32) // divisor + 1
 *
 * @p divisor must be positive and below 2^31 (callers pass i32 kernel arguments).
 */
inline AttentionDenseFastDivPair fastDivPair(int64_t divisor)
{
    uint32_t shift = 0;
    while((uint64_t{1} << shift) < static_cast<uint64_t>(divisor))
    {
        ++shift;
    }
    const uint64_t magic = (((uint64_t{1} << shift) - static_cast<uint64_t>(divisor)) << 32U)
                               / static_cast<uint64_t>(divisor)
                           + 1U;
    return {static_cast<int32_t>(static_cast<uint32_t>(magic)), static_cast<int32_t>(shift)};
}

/// Query blocks per head, `_spec_nqb`: ceil(seqlen_q / block_m).
inline int64_t queryBlocks(const AttentionDenseGridShape& shape)
{
    return (shape.seqLenQ + shape.blockM - 1) / shape.blockM;
}

/// `balance_causal`: the decodes balance causal work unless a window shorter than the keys
/// gives every query block about the same work.
inline bool balancesCausal(const AttentionDenseGridShape& shape)
{
    return shape.causal && (shape.slidingWindow <= 0 || shape.slidingWindow >= shape.seqLenKv);
}

/**
 * @brief The largest runtime fast-division dividend of a persistent decode:
 *        `fastdiv_dividend_bound`, the work count nqb * Hq * B, and for bt_hkv_minor also
 *        gqa * Hq.
 */
inline int64_t fastDivDividendBound(AttentionDensePersistDecode decode,
                                    const AttentionDenseGridShape& shape)
{
    const int64_t work = queryBlocks(shape) * shape.numQueryHeads * shape.batch;
    if(decode == AttentionDensePersistDecode::BT_HKV_MINOR)
    {
        return std::max(work, (shape.numQueryHeads / shape.numKvHeads) * shape.numQueryHeads);
    }
    return work;
}

/**
 * @brief Why @p grid cannot launch over @p shape, or nullopt when it can.
 *
 * Restates the decode's `check` that `Gfx942AttentionDenseSpec` construction runs at the
 * shape, and the dividend bound `fastdiv_kernargs` raises on:
 *   - hq_minor_swz (either grid): Hq a multiple of CHIPLET_NUM_XCDS; on the persistent
 *     grid also num_persistent, and an even query-block count under balance_causal;
 *   - persistent: num_persistent positive and the dividend bound below 2^30.
 *
 * Callers must have validated the extents as positive and Hq as a multiple of Hkv.
 */
inline std::optional<std::string> gridIllegality(const AttentionDenseGrid& grid,
                                                 const AttentionDenseGridShape& shape)
{
    const auto swzHeads = [&shape]() -> std::optional<std::string> {
        if(shape.numQueryHeads % CHIPLET_NUM_XCDS != 0)
        {
            return "hq_minor_swz needs num_query_heads (" + std::to_string(shape.numQueryHeads)
                   + ") divisible by chiplet_num_xcds (" + std::to_string(CHIPLET_NUM_XCDS) + ")";
        }
        return std::nullopt;
    };
    if(!grid.persistent())
    {
        return grid.order == AttentionDenseGridOrder::HQ_MINOR_SWZ ? swzHeads() : std::nullopt;
    }
    if(grid.numPersistent <= 0)
    {
        return "num_persistent must be positive, got " + std::to_string(grid.numPersistent);
    }
    if(grid.decode == AttentionDensePersistDecode::HQ_MINOR_SWZ)
    {
        if(shape.numQueryHeads % CHIPLET_NUM_XCDS != 0
           || grid.numPersistent % CHIPLET_NUM_XCDS != 0)
        {
            return "hq_minor_swz needs num_query_heads (" + std::to_string(shape.numQueryHeads)
                   + ") and num_persistent (" + std::to_string(grid.numPersistent)
                   + ") divisible by " + std::to_string(CHIPLET_NUM_XCDS);
        }
        if(balancesCausal(shape) && queryBlocks(shape) % 2 != 0)
        {
            return "hq_minor_swz needs an even number of query blocks under causal masking "
                   "without a sliding window, got "
                   + std::to_string(queryBlocks(shape));
        }
    }
    const int64_t bound = fastDivDividendBound(grid.decode, shape);
    if(bound >= FASTDIV_DIVIDEND_LIMIT)
    {
        return "persistent decode " + std::string(persistDecodeName(grid.decode))
               + " dividend bound " + std::to_string(bound)
               + " (the work-item count nqb*Hq*B, or the decode's own) must be below 2^30 for "
                 "the work decode's fast division";
    }
    return std::nullopt;
}

/**
 * @brief The launch grid, `attention_dense_grid`: (num_persistent, 1, 1) on the persistent
 *        grid, else `NONPERSIST_DECODES[order].grid`:
 *
 *     qb_minor      (nqb, Hq, B)
 *     bt_hkv_minor  (B, Hq, nqb)
 *     hq_minor_swz  (8 * nqb, Hq / 8, B)
 *
 * Callers must have checked gridIllegality.
 */
inline std::array<int64_t, 3> gridDims(const AttentionDenseGrid& grid,
                                       const AttentionDenseGridShape& shape)
{
    const int64_t nqb = queryBlocks(shape);
    switch(grid.order)
    {
    case AttentionDenseGridOrder::QB_MINOR:
        return {nqb, shape.numQueryHeads, shape.batch};
    case AttentionDenseGridOrder::BT_HKV_MINOR:
        return {shape.batch, shape.numQueryHeads, nqb};
    case AttentionDenseGridOrder::HQ_MINOR_SWZ:
        return {CHIPLET_NUM_XCDS * nqb, shape.numQueryHeads / CHIPLET_NUM_XCDS, shape.batch};
    case AttentionDenseGridOrder::PERSISTENT:
        return {grid.numPersistent, 1, 1};
    default:
        break;
    }
    return {0, 0, 0};
}

/**
 * @brief The fast-division kernarg pairs a persistent launch passes after the shape, in
 *        kernarg order (`fastdiv_kernargs`); empty off the persistent grid.
 *
 * The divisors' values are batch, Hq, gqa = Hq / Hkv and nqb = seqlen_q / block_m (exact:
 * the matcher requires seqlen_q % block_m == 0). Callers must have checked gridIllegality.
 */
inline std::vector<AttentionDenseFastDivPair> fastDivKernargs(const AttentionDenseGrid& grid,
                                                              const AttentionDenseGridShape& shape)
{
    std::vector<AttentionDenseFastDivPair> pairs;
    if(!grid.persistent())
    {
        return pairs;
    }
    for(const auto divisor : fastDivDivisors(grid.decode))
    {
        int64_t value = 0;
        switch(divisor)
        {
        case AttentionDenseFastDivDivisor::BATCH:
            value = shape.batch;
            break;
        case AttentionDenseFastDivDivisor::NUM_QUERY_HEADS:
            value = shape.numQueryHeads;
            break;
        case AttentionDenseFastDivDivisor::GQA:
            value = shape.numQueryHeads / shape.numKvHeads;
            break;
        case AttentionDenseFastDivDivisor::NQB:
            value = shape.seqLenQ / shape.blockM;
            break;
        default:
            break;
        }
        pairs.push_back(fastDivPair(value));
    }
    return pairs;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::attention_dense
