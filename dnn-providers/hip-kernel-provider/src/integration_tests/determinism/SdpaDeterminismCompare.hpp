// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file SdpaDeterminismCompare.hpp
 * @brief The host-only half of the SDPA determinism suite: the poison pattern written before a
 *        launch, the check that a launch wrote every logical element of an output with a usable
 *        value, and the byte comparison of a launch against launch 0. No frontend or HIP types.
 *
 * Only logical elements are checked or compared. A tensor's storage can hold elements no
 * result owns (the gap between packed sequences, the tail after the last one, the positions
 * of a padded batch past its length, the padding of a row stride); those are poisoned like the
 * rest and are neither "unwritten" nor a difference. A tensor's logical elements are given as
 * LiveRuns, runs of storage elements, built by logicalRuns() and packedRuns().
 */
namespace hip_kernel_provider::sdpa_determinism
{

/// Every output and the workspace are filled with this byte before each launch. It makes a
/// NaN in FLOAT, BFLOAT16 and HALF, so an element a launch never wrote is also not a number.
inline constexpr uint8_t POISON_BYTE = 0xFF;

/// The element encodings the suite fills, writes or reads.
enum class ElementType
{
    FLOAT32,
    BFLOAT16,
    HALF,
    FP8_E4M3_FNUZ,
};

constexpr size_t elementBytes(ElementType type)
{
    switch(type)
    {
    case ElementType::FLOAT32:
        return 4;
    case ElementType::BFLOAT16:
    case ElementType::HALF:
        return 2;
    case ElementType::FP8_E4M3_FNUZ:
        return 1;
    default:
        throw std::invalid_argument("elementBytes: unhandled ElementType value");
    }
}

/// A half-open run [begin, end) of storage elements.
struct ElementRun
{
    size_t begin = 0;
    size_t end = 0;
};

/// The logical elements of one tensor, as sorted, disjoint runs of storage elements.
using LiveRuns = std::vector<ElementRun>;

/// Fills @p bytes bytes at @p data with POISON_BYTE.
inline void poison(void* data, size_t bytes)
{
    std::memset(data, POISON_BYTE, bytes);
}

/// The runs of the storage elements a tensor of @p dims and @p strides addresses: every
/// logical element, or only those @p isLive accepts (it receives the logical index). Storage
/// elements no logical element addresses (a padded row stride) are outside every run.
inline LiveRuns logicalRuns(const std::vector<int64_t>& dims,
                            const std::vector<int64_t>& strides,
                            const std::function<bool(const std::vector<int64_t>&)>& isLive = {})
{
    if(dims.size() != strides.size())
    {
        throw std::invalid_argument("logicalRuns: dims and strides differ in rank");
    }
    size_t total = 1;
    for(const auto extent : dims)
    {
        if(extent <= 0)
        {
            return {};
        }
        total *= static_cast<size_t>(extent);
    }

    std::vector<size_t> offsets;
    offsets.reserve(total);
    std::vector<int64_t> index(dims.size(), 0);
    for(size_t n = 0; n < total; ++n)
    {
        if(!isLive || isLive(index))
        {
            int64_t offset = 0;
            for(size_t axis = 0; axis < dims.size(); ++axis)
            {
                offset += index[axis] * strides[axis];
            }
            offsets.push_back(static_cast<size_t>(offset));
        }
        for(size_t axis = dims.size(); axis-- > 0;)
        {
            if(++index[axis] < dims[axis])
            {
                break;
            }
            index[axis] = 0;
        }
    }

    std::sort(offsets.begin(), offsets.end());
    LiveRuns runs;
    for(const auto offset : offsets)
    {
        if(!runs.empty() && offset < runs.back().end)
        {
            continue;
        }
        if(!runs.empty() && offset == runs.back().end)
        {
            ++runs.back().end;
            continue;
        }
        runs.push_back({offset, offset + 1});
    }
    return runs;
}

/// The runs of a packed THD tensor stored token-major with @p tokenElements elements per
/// token: sequence b owns the tokens [origins[b], origins[b] + lengths[b]). A gap between
/// origins and a tail after the last sequence are outside every run.
inline LiveRuns packedRuns(const std::vector<int32_t>& origins,
                           const std::vector<int32_t>& lengths,
                           size_t tokenElements)
{
    if(origins.size() != lengths.size())
    {
        throw std::invalid_argument("packedRuns: origins and lengths differ in size");
    }
    LiveRuns sequences;
    for(size_t b = 0; b < origins.size(); ++b)
    {
        if(origins[b] < 0 || lengths[b] < 0)
        {
            throw std::invalid_argument("packedRuns: negative origin or length");
        }
        if(lengths[b] > 0)
        {
            sequences.push_back({static_cast<size_t>(origins[b]) * tokenElements,
                                 static_cast<size_t>(origins[b] + lengths[b]) * tokenElements});
        }
    }
    std::sort(sequences.begin(), sequences.end(), [](const ElementRun& a, const ElementRun& b) {
        return a.begin < b.begin;
    });
    LiveRuns runs;
    for(const auto& run : sequences)
    {
        if(!runs.empty() && run.begin <= runs.back().end)
        {
            runs.back().end = std::max(runs.back().end, run.end);
        }
        else
        {
            runs.push_back(run);
        }
    }
    return runs;
}

/// What a check found at one element of one output.
enum class FindingKind
{
    UNWRITTEN,
    NOT_A_NUMBER,
    INFINITE,
    DIFFERS,
};

struct Finding
{
    FindingKind kind = FindingKind::DIFFERS;
    std::string tensor;
    /// The storage element index of the first finding.
    size_t element = 0;
    /// The launch the finding was made in; set by the caller that knows it.
    size_t launch = 0;

    std::string describe() const
    {
        const char* what = "differs from launch 0";
        switch(kind)
        {
        case FindingKind::UNWRITTEN:
            what = "was left at the poison pattern (unwritten)";
            break;
        case FindingKind::NOT_A_NUMBER:
            what = "is not a number";
            break;
        case FindingKind::INFINITE:
            what = "is infinite";
            break;
        case FindingKind::DIFFERS:
            break;
        default:
            throw std::invalid_argument("Finding::describe: unhandled FindingKind value");
        }
        return tensor + "[" + std::to_string(element) + "] " + what + " in launch "
               + std::to_string(launch);
    }
};

namespace detail
{

template <typename Bits>
Bits loadBits(const uint8_t* element)
{
    Bits bits;
    std::memcpy(&bits, element, sizeof(bits));
    return bits;
}

/// Whether the element at @p element is a NaN of @p type.
inline bool isNotANumber(ElementType type, const uint8_t* element)
{
    switch(type)
    {
    case ElementType::FLOAT32:
    {
        const auto bits = loadBits<uint32_t>(element);
        return (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0;
    }
    case ElementType::BFLOAT16:
    {
        const auto bits = loadBits<uint16_t>(element);
        return (bits & 0x7F80U) == 0x7F80U && (bits & 0x007FU) != 0;
    }
    case ElementType::HALF:
    {
        const auto bits = loadBits<uint16_t>(element);
        return (bits & 0x7C00U) == 0x7C00U && (bits & 0x03FFU) != 0;
    }
    case ElementType::FP8_E4M3_FNUZ:
        return *element == 0x80;
    default:
        throw std::invalid_argument("isNotANumber: unhandled ElementType value");
    }
}

/// Whether the element at @p element is an infinity of @p type (FP8 E4M3 FNUZ has none).
inline bool isInfinity(ElementType type, const uint8_t* element)
{
    switch(type)
    {
    case ElementType::FLOAT32:
        return (loadBits<uint32_t>(element) & 0x7FFFFFFFU) == 0x7F800000U;
    case ElementType::BFLOAT16:
        return (loadBits<uint16_t>(element) & 0x7FFFU) == 0x7F80U;
    case ElementType::HALF:
        return (loadBits<uint16_t>(element) & 0x7FFFU) == 0x7C00U;
    case ElementType::FP8_E4M3_FNUZ:
        return false;
    default:
        throw std::invalid_argument("isInfinity: unhandled ElementType value");
    }
}

/// Whether every byte of the @p size-byte element is the poison byte.
inline bool isPoison(const uint8_t* element, size_t size)
{
    return std::all_of(element, element + size, [](uint8_t byte) { return byte == POISON_BYTE; });
}

} // namespace detail

/// The first logical element of @p data (storage elements of @p type) that the launch left at
/// the poison pattern, holds a NaN, or holds an infinity unless @p allowInfinity. An element
/// whose every byte is the poison byte is reported as unwritten, before the NaN check, since
/// the poison pattern is a NaN in every floating encoding.
inline std::optional<Finding> findInvalidElement(const std::string& tensor,
                                                 ElementType type,
                                                 const void* data,
                                                 const LiveRuns& live,
                                                 bool allowInfinity)
{
    const auto* bytes = static_cast<const uint8_t*>(data);
    const size_t size = elementBytes(type);
    for(const auto& run : live)
    {
        for(size_t element = run.begin; element < run.end; ++element)
        {
            const uint8_t* at = bytes + element * size;
            if(detail::isPoison(at, size))
            {
                return Finding{FindingKind::UNWRITTEN, tensor, element, 0};
            }
            if(detail::isNotANumber(type, at))
            {
                return Finding{FindingKind::NOT_A_NUMBER, tensor, element, 0};
            }
            if(!allowInfinity && detail::isInfinity(type, at))
            {
                return Finding{FindingKind::INFINITE, tensor, element, 0};
            }
        }
    }
    return std::nullopt;
}

/// The first logical element whose @p elementSize bytes differ between @p reference and
/// @p current.
inline std::optional<Finding> findFirstDifference(const std::string& tensor,
                                                  size_t elementSize,
                                                  const void* reference,
                                                  const void* current,
                                                  const LiveRuns& live)
{
    const auto* expected = static_cast<const uint8_t*>(reference);
    const auto* actual = static_cast<const uint8_t*>(current);
    for(const auto& run : live)
    {
        for(size_t element = run.begin; element < run.end; ++element)
        {
            if(std::memcmp(
                   expected + element * elementSize, actual + element * elementSize, elementSize)
               != 0)
            {
                return Finding{FindingKind::DIFFERS, tensor, element, 0};
            }
        }
    }
    return std::nullopt;
}

/// How many logical elements differ between @p reference and @p current.
inline size_t countDifferences(size_t elementSize,
                               const void* reference,
                               const void* current,
                               const LiveRuns& live)
{
    const auto* expected = static_cast<const uint8_t*>(reference);
    const auto* actual = static_cast<const uint8_t*>(current);
    size_t differing = 0;
    for(const auto& run : live)
    {
        for(size_t element = run.begin; element < run.end; ++element)
        {
            differing += std::memcmp(expected + element * elementSize,
                                     actual + element * elementSize,
                                     elementSize)
                                 != 0
                             ? 1
                             : 0;
        }
    }
    return differing;
}

} // namespace hip_kernel_provider::sdpa_determinism
