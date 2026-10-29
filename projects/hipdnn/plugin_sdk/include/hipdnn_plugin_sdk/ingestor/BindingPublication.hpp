// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
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
#include <hipdnn_plugin_sdk/ingestor/DescriptorJsonRules.hpp>
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

/// The fields publishTensor() publishes under a root, besides the root itself and the
/// indexed fields below. Kept beside publishTensor() so the two change together; the
/// load-time reference check (CompiledCriteria.hpp) reads these, not the publisher.
inline constexpr std::array<std::string_view, 8> PUBLISHED_TENSOR_FIELDS
    = {"uid",
       "rank",
       "dtype",
       "stride_order",
       "packed",
       "virtual",
       "is_runtime_pass_by_value",
       "value_f32"};

/// The per-axis fields publishTensor() publishes as `<field>[i]`, one token per axis.
inline constexpr std::array<std::string_view, 2> PUBLISHED_INDEXED_TENSOR_FIELDS
    = {"dims", "strides"};

/// The fields publishGraph() publishes under the reserved `graph` root.
inline constexpr std::array<std::string_view, 2> PUBLISHED_GRAPH_FIELDS
    = {"node_count", "is_override_shape_enabled"};

namespace detail
{

/// Splits `name[i]` into its name and index. The index must be written the way
/// tensorElement() writes one: decimal digits, no sign, and no leading zero, so a token
/// that can never be published is refused rather than read as null forever.
inline std::optional<std::pair<std::string_view, size_t>> splitIndexedField(std::string_view field)
{
    const auto open = field.find('[');
    if(open == std::string_view::npos || open == 0 || field.back() != ']')
    {
        return std::nullopt;
    }
    const auto digits = field.substr(open + 1, field.size() - open - 2);
    // 18 digits always fit size_t on a 64-bit host and no tensor has that many axes.
    if(digits.empty() || digits.size() > 18 || (digits.size() > 1 && digits.front() == '0'))
    {
        return std::nullopt;
    }
    size_t index = 0;
    for(const char digit : digits)
    {
        if(digit < '0' || digit > '9')
        {
            return std::nullopt;
        }
        index = (index * 10) + static_cast<size_t>(digit - '0');
    }
    return std::pair{field.substr(0, open), index};
}

/// True when publishTensor() can publish @p field under a root: one of
/// PUBLISHED_TENSOR_FIELDS, or `dims[i]` / `strides[i]`. Whether axis `i` exists depends
/// on the graph's rank, so any index is accepted here and an absent one reads as null.
inline bool isPublishedTensorField(std::string_view field)
{
    const auto contains = [](const auto& names, std::string_view name) {
        return std::find(names.begin(), names.end(), name) != names.end();
    };
    if(contains(PUBLISHED_TENSOR_FIELDS, field))
    {
        return true;
    }
    const auto indexed = splitIndexedField(field);
    return indexed.has_value() && contains(PUBLISHED_INDEXED_TENSOR_FIELDS, indexed->first);
}

} // namespace detail

/// Validates @p tensor and publishes its fields under @p root. Publish each root
/// once. If this returns false, discard the whole binding.
///
/// Publishes:
///  - `<root>` and `<root>.uid`: the tensor uid.
///  - `<root>.rank`, and `<root>.dims[i]` and `<root>.strides[i]` for each axis.
///  - `<root>.dtype`: the DataType enum name.
///  - `<root>.stride_order`: axis ranks from extractStrideOrder() in ShapeUtilities.hpp.
///    Ties follow that function, so four unit strides give [3, 2, 1, 0].
///  - `<root>.packed`: the tensor fills a dense, non-overlapping block with positive
///    strides, ignoring strides on size-one axes (isTensorPacked()). A ragged tensor is
///    never packed.
///  - `<root>.virtual` and `<root>.is_runtime_pass_by_value`.
///  - `<root>.value_f32`: only for a stored scalar that is not passed by value at run
///    time. The value is rounded to a finite binary32 float and stored as a double. A
///    runtime default is validated the same way but not published.
///
/// Returns false when @p root is not an identifier; the dims or strides are missing,
/// empty, or of different lengths; a dim is not positive or the element count overflows
/// int64_t; the dtype has no enum name; the stored value disagrees with the value type;
/// a stored or runtime-passed scalar is virtual or has more than one element; or a stored
/// value is not finite in binary32.
inline bool publishTensor(BoundTokens& bound,
                          std::string_view root,
                          const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    if(!detail::isIdentifier(root))
    {
        return false;
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

/// Publishes facts for the whole graph, not just the matched nodes: the
/// PUBLISHED_GRAPH_FIELDS under the `graph` root.
inline void publishGraph(BoundTokens& bound,
                         const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
{
    bound.emplace("graph.node_count", static_cast<int64_t>(graph.nodeCount()));
    bound.emplace("graph.is_override_shape_enabled", graph.getGraph().is_override_shape_enabled());
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
