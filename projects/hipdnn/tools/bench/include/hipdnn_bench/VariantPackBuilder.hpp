// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

/// @file VariantPackBuilder.hpp
/// @brief Sizing the buffers a deserialized graph needs (RFC 0019.13 §5.3).
///
/// Sizes come from strides, not element counts (padded tensors are larger than packed), and
/// are computed in bits so sub-byte types (FP4, INT4) are not rounded to zero.
namespace hipdnn_bench
{

/// Bits one element of @p dataType occupies, or 0 for a type this tool cannot size (refused
/// by name rather than guessed).
inline int64_t elementBits(hipdnn_frontend::DataType dataType)
{
    using hipdnn_frontend::DataType;
    switch(dataType)
    {
    case DataType::DOUBLE:
    case DataType::INT64:
    case DataType::COMPLEX_FP32:
        return 64;

    case DataType::FLOAT:
    case DataType::INT32:
    // Four packed 8-bit values addressed as one element.
    case DataType::INT8x4:
    case DataType::UINT8x4:
    case DataType::FAST_FLOAT_FOR_FP8:
        return 32;

    case DataType::HALF:
    case DataType::BFLOAT16:
        return 16;

    case DataType::INT8:
    case DataType::UINT8:
    case DataType::BOOLEAN:
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E8M0:
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ:
        return 8;

    case DataType::FP6_E2M3:
    case DataType::FP6_E3M2:
        return 6;

    case DataType::FP4_E2M1:
    case DataType::INT4:
        return 4;

    case DataType::COMPLEX_FP64:
        return 128;
    case DataType::INT8x32:
        return 256;

    case DataType::NOT_SET:
    default:
        return 0;
    }
}

/// @brief Elements spanned by @p dims with @p strides: `sum_i((dim_i - 1) * stride_i) + 1`,
///        which equals the element count only for packed tensors.
///
/// Returns 0 for a malformed tensor (rank mismatch, non-positive extent, negative stride) or
/// one whose span overflows int64_t; overflow is checked before each product is formed.
inline int64_t elementSpan(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    if(dims.empty() || dims.size() != strides.size())
    {
        return 0;
    }

    constexpr int64_t LIMIT = std::numeric_limits<int64_t>::max();
    int64_t furthest = 0;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(dims[i] <= 0 || strides[i] < 0)
        {
            return 0;
        }
        const int64_t lastIndex = dims[i] - 1;
        // lastIndex * stride + furthest <= LIMIT, tested without forming either side.
        if(lastIndex != 0 && strides[i] > (LIMIT - furthest) / lastIndex)
        {
            return 0;
        }
        furthest += lastIndex * strides[i];
    }
    return furthest == LIMIT ? 0 : furthest + 1;
}

/// @brief Bytes to allocate for a tensor of @p dims / @p strides / @p dataType.
///
/// Rounds up to whole bytes. Returns nullopt for an unsizeable tensor or one whose byte count
/// overflows int64_t.
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

    // ceil(span * bits / 8), split so no intermediate exceeds the result; `span * bits + 7`
    // can overflow for 4/6-bit spans near int64_t's range.
    const int64_t whole = span / 8;
    const int64_t tail = (((span % 8) * bits) + 7) / 8; // at most 256 bits: never overflows
    if(whole > (std::numeric_limits<int64_t>::max() - tail) / bits)
    {
        return std::nullopt;
    }
    return (whole * bits) + tail;
}

/// Where the benchmark must put a tensor's memory.
enum class TensorStorage
{
    DEVICE, ///< Device memory: every tensor a kernel reads or writes.
    HOST ///< Host memory: a runtime user-supplied pass-by-value scalar (RFC 0016).
};

/// One tensor the benchmark must provide memory for.
struct TensorRequirement
{
    int64_t uid = 0;
    std::string name;
    int64_t bytes = 0;
    hipdnn_frontend::DataType dataType = hipdnn_frontend::DataType::NOT_SET;

    /// Some node of the graph writes it. Produced tensors are cross-checked; the rest get the
    /// input fill.
    bool produced = false;

    TensorStorage storage = TensorStorage::DEVICE;
};

/// What a graph needs before it can be executed.
struct VariantPackPlan
{
    std::vector<TensorRequirement> tensors;

    /// Non-empty, naming the offending tensor, when some tensor could not be sized; the plan
    /// is then unusable.
    std::string error;
};

/// @brief Uids of every tensor some node of @p graph writes, virtual ones included.
inline std::unordered_set<int64_t> producedUids(const hipdnn_frontend::graph::Graph& graph)
{
    std::unordered_set<int64_t> produced;
    const std::function<void(const hipdnn_frontend::graph::INode&)> collect
        = [&produced](const hipdnn_frontend::graph::INode& node) {
              for(const auto& tensor : node.getNodeOutputTensorAttributes())
              {
                  if(tensor != nullptr && tensor->has_uid())
                  {
                      produced.insert(tensor->get_uid());
                  }
              }
          };
    graph.visit(collect);
    return produced;
}

/// @brief Every tensor of @p graph the variant pack must carry, with the bytes each needs,
///        whether the graph writes it, and where its memory lives.
///
/// Skips virtual tensors and scalars with a baked value (RFC 0016; they have no slot).
/// Runtime user-supplied scalars get host memory because the provider reads them on the CPU.
inline VariantPackPlan planVariantPack(const hipdnn_frontend::graph::Graph& graph)
{
    VariantPackPlan plan;
    const auto produced = producedUids(graph);

    for(const auto& [uid, tensor] : graph.getTensorsByUid())
    {
        if(tensor == nullptr || tensor->get_is_virtual() || tensor->get_has_compile_time_constant()
           || tensor->get_pass_by_value().has_value())
        {
            continue;
        }

        const auto bytes
            = tensorBytes(tensor->get_dim(), tensor->get_stride(), tensor->get_data_type());
        if(!bytes.has_value())
        {
            plan.error = "cannot size tensor '" + tensor->get_name() + "' (uid "
                         + std::to_string(uid) + "): "
                         + (elementBits(tensor->get_data_type()) == 0
                                ? "its data type has no known element width"
                                : "its dims and strides do not describe an allocation int64_t "
                                  "can represent");
            return plan;
        }

        TensorRequirement requirement;
        requirement.uid = uid;
        requirement.name = tensor->get_name();
        requirement.bytes = *bytes;
        requirement.dataType = tensor->get_data_type();
        requirement.produced = produced.count(uid) != 0;
        requirement.storage = !requirement.produced && tensor->get_is_runtime_pass_by_value()
                                  ? TensorStorage::HOST
                                  : TensorStorage::DEVICE;
        plan.tensors.push_back(std::move(requirement));
    }

    if(plan.tensors.empty())
    {
        plan.error = "the graph declares no non-virtual tensors";
    }

    // Sorted by uid so allocation order is reproducible across runs.
    std::sort(plan.tensors.begin(),
              plan.tensors.end(),
              [](const TensorRequirement& a, const TensorRequirement& b) { return a.uid < b.uid; });
    return plan;
}

} // namespace hipdnn_bench
