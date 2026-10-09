// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// Builds a tensor field token such as `input_a.dtype`.
inline std::string tensorField(std::string_view root, std::string_view field)
{
    return std::string(root) + "." + std::string(field);
}

/// Builds an indexed tensor field token such as `input_a.dims[0]`.
inline std::string tensorElement(std::string_view root, std::string_view field, size_t index)
{
    return tensorField(root, field) + "[" + std::to_string(index) + "]";
}

/// Validates @p tensor and publishes its fields under @p root. Publish each root
/// once. If this returns false, discard the whole binding.
inline bool publishTensor(BoundTokens& bound,
                          std::string_view root,
                          const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    const auto isLetter
        = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; };
    if(root.empty() || !isLetter(root.front()))
    {
        return false;
    }
    for(const auto c : root)
    {
        if(!isLetter(c) && (c < '0' || c > '9'))
        {
            return false;
        }
    }
    if(tensor == nullptr || tensor->dims() == nullptr || tensor->strides() == nullptr
       || tensor->dims()->empty() || tensor->dims()->size() != tensor->strides()->size())
    {
        return false;
    }

    int64_t elementCount = 1;
    for(const auto dim : *tensor->dims())
    {
        if(dim <= 0 || elementCount > std::numeric_limits<int64_t>::max() / dim)
        {
            return false;
        }
        elementCount *= dim;
    }

    const auto* dtype = data_objects::EnumNameDataType(tensor->data_type());
    if(dtype == nullptr || *dtype == '\0')
    {
        return false;
    }
    const bool stored = tensor->value_type() != data_objects::TensorValue::NONE;
    const bool runtime = tensor->is_runtime_pass_by_value();
    if((stored != (tensor->value() != nullptr))
       || ((stored || runtime) && (elementCount != 1 || tensor->virtual_())))
    {
        return false;
    }

    std::optional<float> scalar;
    if(stored)
    {
        try
        {
            // Validate runtime defaults, but do not publish them as constants.
            scalar = hipdnn_flatbuffers_sdk::utilities::extractFiniteFloatFromTensorValue(
                tensor, "binding tensor");
        }
        catch(const std::runtime_error&)
        {
            return false;
        }
    }

    const std::vector<int64_t> dims(tensor->dims()->begin(), tensor->dims()->end());
    const std::vector<int64_t> strides(tensor->strides()->begin(), tensor->strides()->end());
    const bool packed = !tensor->ragged_offset_tensor_uid().has_value()
                        && hipdnn_data_sdk::utilities::isTensorPacked(dims, strides);
    auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(strides);

    bound.emplace(std::string(root), tensor->uid());
    bound.emplace(tensorField(root, "uid"), tensor->uid());
    bound.emplace(tensorField(root, "rank"), static_cast<int64_t>(dims.size()));
    bound.emplace(tensorField(root, "dtype"), std::string(dtype));
    bound.emplace(tensorField(root, "stride_order"), std::move(strideOrder));
    bound.emplace(tensorField(root, "packed"), packed);
    bound.emplace(tensorField(root, "virtual"), tensor->virtual_());
    bound.emplace(tensorField(root, "is_runtime_pass_by_value"), runtime);
    for(size_t i = 0; i < dims.size(); ++i)
    {
        bound.emplace(tensorElement(root, "dims", i), dims[i]);
        bound.emplace(tensorElement(root, "strides", i), strides[i]);
    }
    if(scalar.has_value() && !runtime)
    {
        bound.emplace(tensorField(root, "value_f32"), static_cast<double>(*scalar));
    }
    return true;
}

/// Publish facts for the whole graph, not just the matched nodes.
inline bool publishGraph(BoundTokens& bound,
                         const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
{
    bound.emplace("graph.node_count", static_cast<int64_t>(graph.nodeCount()));
    bound.emplace("graph.is_override_shape_enabled", graph.getGraph().is_override_shape_enabled());
    return true;
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
