// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

/// @file VariantPackBuilder.hpp
/// @brief Sizing the buffers a deserialized graph needs (RFC 0019.13 §5.3).
///
/// A problem arrives as a serialized graph, so the benchmark knows nothing about the
/// operation it is about to run. It has tensors, and it must allocate for them. That is the
/// whole job, and the two ways to get it wrong both produce a number rather than a crash:
///
///  - **Sizing by element count.** `count * sizeof(element)` is the footprint of a *packed*
///    tensor. A tensor with padded or aligned leading dimensions occupies more, and the
///    kernel will write past the end of a buffer sized this way.
///  - **Sizing sub-byte types by a byte-valued element size.** FP4 and INT4 are half a byte,
///    so any per-element size rounded to 1 over-allocates harmlessly, but rounded to 0
///    allocates nothing. The arithmetic here is in bits for that reason.
///
/// Neither shows up as a failure in a corpus run. They show up as a time.
namespace hipdnn_bench
{

/// Bits one element of @p dataType occupies, or 0 for a type this tool cannot size.
///
/// Bits rather than bytes because FP4_E2M1 and INT4 are four. Returning 0 rather than
/// guessing is deliberate: a problem naming a type not listed here is refused with its name,
/// which is a fixable message, where a guessed width is a buffer overrun.
inline int64_t elementBits(hipdnn_frontend::DataType dataType)
{
    using hipdnn_frontend::DataType;
    switch(dataType)
    {
    case DataType::DOUBLE:
    case DataType::INT64:
    case DataType::COMPLEX_FP32: return 64;

    case DataType::FLOAT:
    case DataType::INT32:
    // Four packed 8-bit values addressed as one element.
    case DataType::INT8x4:
    case DataType::UINT8x4:
    case DataType::FAST_FLOAT_FOR_FP8: return 32;

    case DataType::HALF:
    case DataType::BFLOAT16: return 16;

    case DataType::INT8:
    case DataType::UINT8:
    case DataType::BOOLEAN:
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E8M0:
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ: return 8;

    case DataType::FP6_E2M3:
    case DataType::FP6_E3M2: return 6;

    case DataType::FP4_E2M1:
    case DataType::INT4: return 4;

    case DataType::COMPLEX_FP64: return 128;
    case DataType::INT8x32: return 256;

    case DataType::NOT_SET:
    default: return 0;
    }
}

/// @brief Elements spanned by @p dims with @p strides -- the extent that must be addressable,
///        which is not the element count unless the tensor is packed.
///
/// `sum_i((dim_i - 1) * stride_i) + 1`. A sum rather than a maximum over dimensions: the
/// furthest addressable element is the one at the last index of *every* dimension at once, so
/// taking the largest single term would under-count and size the buffer short.
///
/// Returns 0 for a malformed tensor (rank mismatch, non-positive extent) so a caller refuses
/// rather than allocating something arbitrary.
inline int64_t elementSpan(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    if(dims.empty() || dims.size() != strides.size())
    {
        return 0;
    }

    int64_t furthest = 0;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(dims[i] <= 0 || strides[i] < 0)
        {
            return 0;
        }
        furthest += (dims[i] - 1) * strides[i];
    }
    return furthest + 1;
}

/// @brief Bytes to allocate for a tensor of @p dims / @p strides / @p dataType.
///
/// Rounds up, so a sub-byte tensor with an odd span still gets a whole byte. Returns nullopt
/// for an unsizeable tensor rather than a zero a caller might allocate.
inline std::optional<int64_t> tensorBytes(const std::vector<int64_t>& dims,
                                          const std::vector<int64_t>& strides,
                                          hipdnn_frontend::DataType dataType)
{
    const auto bits = elementBits(dataType);
    const auto span = elementSpan(dims, strides);
    if(bits == 0 || span == 0)
    {
        return std::nullopt;
    }

    // Overflow is a real possibility: the corpus search proposes across orders of magnitude,
    // and a span near int64_t's range multiplied by 256 bits leaves it.
    if(span > (std::numeric_limits<int64_t>::max() / bits))
    {
        return std::nullopt;
    }
    return ((span * bits) + 7) / 8;
}

/// One tensor the benchmark must provide memory for.
struct TensorRequirement
{
    int64_t uid = 0;
    std::string name;
    int64_t bytes = 0;
    hipdnn_frontend::DataType dataType = hipdnn_frontend::DataType::NOT_SET;
};

/// What a graph needs before it can be executed.
struct VariantPackPlan
{
    std::vector<TensorRequirement> tensors;

    /// Non-empty when some tensor could not be sized; the plan is then unusable. Named
    /// rather than counted, because "one tensor could not be sized" is not actionable and
    /// "tensor W is FP6_E2M3, which this tool cannot size" is.
    std::string error;
};

/// @brief Every non-virtual tensor of @p graph, with the bytes each needs.
///
/// Virtual tensors are skipped: they are intermediates the engine materialises itself, and
/// allocating for them would both waste memory and hand the plan a pointer for something the
/// variant pack must not contain.
inline VariantPackPlan planVariantPack(const hipdnn_frontend::graph::Graph& graph)
{
    VariantPackPlan plan;

    for(const auto& [uid, tensor] : graph.getTensorsByUid())
    {
        if(tensor == nullptr || tensor->get_is_virtual())
        {
            continue;
        }

        const auto bytes
            = tensorBytes(tensor->get_dim(), tensor->get_stride(), tensor->get_data_type());
        if(!bytes.has_value())
        {
            plan.error = "cannot size tensor '" + tensor->get_name() + "' (uid "
                         + std::to_string(uid) + ")";
            return plan;
        }

        TensorRequirement requirement;
        requirement.uid = uid;
        requirement.name = tensor->get_name();
        requirement.bytes = *bytes;
        requirement.dataType = tensor->get_data_type();
        plan.tensors.push_back(std::move(requirement));
    }

    if(plan.tensors.empty())
    {
        plan.error = "the graph declares no non-virtual tensors";
    }

    // Sorted by uid so a plan is reproducible: getTensorsByUid returns an unordered_map, and
    // allocation order would otherwise vary between runs of the same problem.
    std::sort(plan.tensors.begin(),
              plan.tensors.end(),
              [](const TensorRequirement& a, const TensorRequirement& b) { return a.uid < b.uid; });
    return plan;
}

} // namespace hipdnn_bench
