// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

/// @file TensorFillers.hpp
/// @brief Producing the contents a problem declares (proposed RFC 0019.13 §4 addition).
///
/// Most tensors can hold anything: the kernel does the same work either way, and a benchmark
/// that fills them with zeros measures the same thing as one that fills them with data.
///
/// Some cannot. An MoE grouped matmul's `first_token_offset` says how many tokens each expert
/// receives, and that decides the size of every grouped GEMM. Fill it with zeros and every
/// expert gets nothing; fill it uniformly and every expert gets an equal share. Those are
/// different problems with byte-identical graphs, and §12.2 lists the difference -- routing
/// skew, capacity factor -- as parameters of the space.
///
/// So this file exists to make a declared fill reproducible: same problem, same bytes, on any
/// machine. Everything is produced host-side into a byte buffer rather than on device, because
/// a fill has to be checkable, and a routing that quietly assigned no tokens to any expert
/// would otherwise look exactly like a fast kernel.
namespace hipdnn_bench
{

/// Element widths this filler can write. Kept narrow deliberately: an index or offset tensor
/// is integral, and the fills that matter are all integral. A float tensor takes ZEROS or
/// UNIFORM, which need no per-dtype semantics beyond width.
enum class FillElement
{
    INT32,
    INT64,
    FLOAT32,
    BYTES ///< Anything else: only ZEROS and UNIFORM are meaningful.
};

/// A tokens-to-experts assignment, which is what both routing fills are views onto.
///
/// Kept as one computation rather than two because the offsets and the per-token indices must
/// agree: an offset table saying expert 3 owns tokens [40, 60) alongside an index table that
/// never names expert 3 describes no coherent routing, and the kernel would read one and the
/// corpus would describe the other.
struct Routing
{
    /// `numExperts + 1` entries: expert *e* owns tokens `[offsets[e], offsets[e+1])`.
    std::vector<int64_t> offsets;

    /// `numTokens` entries: the expert each token is assigned to.
    std::vector<int64_t> assignment;
};

/// @brief Distributes @p numTokens over @p numExperts.
///
/// @param imbalanced When false, every expert receives an equal share (the remainder going to
///                   the first few, so the offsets still sum exactly). When true, the
///                   distribution is deliberately skewed -- the first expert receives close to
///                   half the tokens and the rest divide what remains.
///
/// Skew is the parameter that matters and the reason this is not simply `numTokens /
/// numExperts`: a balanced routing makes every grouped GEMM the same size, which is the case
/// an implementation is most likely to be tuned for and least likely to be tested on. A corpus
/// of only balanced routings would train a heuristic that has never seen the shape that hurts.
inline Routing computeRouting(int64_t numTokens, int64_t numExperts, bool imbalanced)
{
    Routing routing;
    if(numExperts <= 0 || numTokens < 0)
    {
        return routing;
    }

    std::vector<int64_t> counts(static_cast<size_t>(numExperts), 0);
    if(imbalanced)
    {
        // Half to the first expert, the rest shared. Deterministic rather than random so the
        // same problem point yields the same routing on every machine.
        const int64_t head = std::max<int64_t>(1, numTokens / 2);
        counts[0] = std::min(numTokens, head);
        int64_t remaining = numTokens - counts[0];
        for(size_t e = 1; e < counts.size() && remaining > 0; ++e)
        {
            const auto share = remaining / static_cast<int64_t>(counts.size() - e);
            counts[e] = std::max<int64_t>(share, 0);
            remaining -= counts[e];
        }
        counts.back() += remaining;
    }
    else
    {
        const auto share = numTokens / numExperts;
        auto remainder = numTokens % numExperts;
        for(auto& count : counts)
        {
            count = share + (remainder-- > 0 ? 1 : 0);
        }
    }

    routing.offsets.reserve(counts.size() + 1);
    int64_t running = 0;
    routing.offsets.push_back(0);
    for(size_t e = 0; e < counts.size(); ++e)
    {
        running += counts[e];
        routing.offsets.push_back(running);
        for(int64_t t = 0; t < counts[e]; ++t)
        {
            routing.assignment.push_back(static_cast<int64_t>(e));
        }
    }
    return routing;
}

namespace detail
{

inline size_t elementWidth(FillElement element)
{
    switch(element)
    {
    case FillElement::INT32:
    case FillElement::FLOAT32: return 4;
    case FillElement::INT64: return 8;
    case FillElement::BYTES:
    default: return 1;
    }
}

/// Writes @p value into @p bytes at @p index, in the element's width.
inline void writeElement(std::vector<uint8_t>& bytes,
                         size_t index,
                         FillElement element,
                         int64_t value)
{
    const auto width = elementWidth(element);
    const auto offset = index * width;
    if(offset + width > bytes.size())
    {
        return;
    }

    if(element == FillElement::INT32)
    {
        const auto narrowed = static_cast<int32_t>(value);
        std::memcpy(bytes.data() + offset, &narrowed, sizeof(narrowed));
    }
    else if(element == FillElement::INT64)
    {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }
    else if(element == FillElement::FLOAT32)
    {
        const auto asFloat = static_cast<float>(value);
        std::memcpy(bytes.data() + offset, &asFloat, sizeof(asFloat));
    }
    else
    {
        bytes[offset] = static_cast<uint8_t>(value);
    }
}

} // namespace detail

/// @brief Zero-filled buffer of @p bytes.
inline std::vector<uint8_t> fillZeros(size_t bytes)
{
    return std::vector<uint8_t>(bytes, 0);
}

/// @brief 0, 1, 2, ... as @p element, for index tensors that must be in range.
inline std::vector<uint8_t> fillSequence(size_t bytes, FillElement element, int64_t modulo = 0)
{
    std::vector<uint8_t> buffer(bytes, 0);
    const auto width = detail::elementWidth(element);
    const auto count = width == 0 ? 0 : bytes / width;
    for(size_t i = 0; i < count; ++i)
    {
        const auto value = modulo > 0 ? static_cast<int64_t>(i) % modulo : static_cast<int64_t>(i);
        detail::writeElement(buffer, i, element, value);
    }
    return buffer;
}

/// @brief Uniform values in `[0, bound)`, reproducible from @p seed.
inline std::vector<uint8_t>
    fillUniform(size_t bytes, FillElement element, int64_t bound, uint64_t seed)
{
    std::vector<uint8_t> buffer(bytes, 0);
    const auto width = detail::elementWidth(element);
    const auto count = width == 0 ? 0 : bytes / width;

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> distribution(0, std::max<int64_t>(1, bound) - 1);
    for(size_t i = 0; i < count; ++i)
    {
        detail::writeElement(buffer, i, element, distribution(rng));
    }
    return buffer;
}

/// @brief The offset table of @p routing, as @p element.
inline std::vector<uint8_t>
    fillRoutingOffsets(size_t bytes, FillElement element, const Routing& routing)
{
    std::vector<uint8_t> buffer(bytes, 0);
    for(size_t i = 0; i < routing.offsets.size(); ++i)
    {
        detail::writeElement(buffer, i, element, routing.offsets[i]);
    }
    return buffer;
}

/// @brief The per-token expert index of @p routing, as @p element.
inline std::vector<uint8_t>
    fillExpertAssignment(size_t bytes, FillElement element, const Routing& routing)
{
    std::vector<uint8_t> buffer(bytes, 0);
    for(size_t i = 0; i < routing.assignment.size(); ++i)
    {
        detail::writeElement(buffer, i, element, routing.assignment[i]);
    }
    return buffer;
}

} // namespace hipdnn_bench
