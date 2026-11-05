// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "harness/BundleMetadata.hpp"
#include "harness/bundle/OutputComparison.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace detail
{

// A host view of a dense tensor that reports a ragged iteration layout: positions whose
// index on `axis` is at least the length of their batch are out of block. Addressing
// stays dense, so every position reads its own element. The reported origins and
// sequence stride come from the dims, so two views with the same dims, axis and lengths
// report the same layout whatever their strides.
// NOLINTBEGIN(portability-template-virtual-member-function)
template <class T>
class LiveRowView : public hipdnn_data_sdk::utilities::ShallowTensor<T>
{
public:
    LiveRowView(void* memory,
                const std::vector<int64_t>& dims,
                const std::vector<int64_t>& strides,
                int axis,
                std::vector<int64_t> lengths)
        : hipdnn_data_sdk::utilities::ShallowTensor<T>(memory, dims, strides)
    {
        int64_t batchPitch = 1;
        for(size_t i = 1; i < dims.size(); ++i)
        {
            batchPitch *= dims[i];
        }
        int64_t rowPitch = 1;
        for(auto i = static_cast<size_t>(axis) + 1; i < dims.size(); ++i)
        {
            rowPitch *= dims[i];
        }
        _layout.origins.reserve(lengths.size());
        for(size_t b = 0; b < lengths.size(); ++b)
        {
            _layout.origins.push_back(static_cast<int64_t>(b) * batchPitch);
        }
        _layout.lengths = std::move(lengths);
        _layout.seqAxis = axis;
        _layout.seqStride = rowPitch;
    }

    std::optional<hipdnn_data_sdk::utilities::RaggedIterationInfo>
        raggedIterationInfo() const override
    {
        return _layout;
    }

    bool isPacked() const override
    {
        return false;
    }

private:
    hipdnn_data_sdk::utilities::RaggedIterationInfo _layout{};
};
// NOLINTEND(portability-template-virtual-member-function)

// Returns an empty string when `rows` fits a tensor of `dims`, otherwise the problem.
inline std::string liveRowsProblem(const std::vector<int64_t>& dims, const LiveRows& rows)
{
    const auto rank = static_cast<int64_t>(dims.size());
    if(rows.axis < 1 || rows.axis >= rank)
    {
        return "axis " + std::to_string(rows.axis) + " is not in the range 1 to "
               + std::to_string(rank - 1);
    }
    if(static_cast<int64_t>(rows.lengths.size()) != dims[0])
    {
        return std::to_string(rows.lengths.size()) + " lengths for " + std::to_string(dims[0])
               + " batches (dims[0])";
    }
    const int64_t extent = dims[static_cast<size_t>(rows.axis)];
    for(size_t b = 0; b < rows.lengths.size(); ++b)
    {
        const int64_t length = rows.lengths[b];
        if(length < 0 || length > extent)
        {
            std::string problem = "length ";
            problem.append(std::to_string(length))
                .append(" of batch ")
                .append(std::to_string(b))
                .append(" is not in the range 0 to ")
                .append(std::to_string(extent));
            return problem;
        }
    }
    return {};
}

// A LiveRowView of the host data of `tensor`, whose element type is `dataType`. The
// caller keeps `tensor` alive while the view lives. Throws std::invalid_argument when
// `rows` does not fit the tensor or `dataType` is not a floating-point type.
inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    makeLiveRowView(hipdnn_data_sdk::utilities::ITensor& tensor,
                    hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                    const LiveRows& rows)
{
    const auto problem = liveRowsProblem(tensor.dims(), rows);
    if(!problem.empty())
    {
        throw std::invalid_argument("live rows do not fit the tensor: " + problem);
    }
    const auto view = [&](auto tag) -> std::unique_ptr<hipdnn_data_sdk::utilities::ITensor> {
        using T = typename decltype(tag)::type;
        return std::make_unique<LiveRowView<T>>(
            tensor.rawHostData(), tensor.dims(), tensor.strides(), rows.axis, rows.lengths);
    };
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using hipdnn_test_sdk::detail::TypeTag;
    namespace types = hipdnn_data_sdk::types;
    switch(dataType)
    {
    case DataType::FLOAT:
        return view(TypeTag<float>{});
    case DataType::HALF:
        return view(TypeTag<types::half>{});
    case DataType::BFLOAT16:
        return view(TypeTag<types::bfloat16>{});
    case DataType::DOUBLE:
        return view(TypeTag<double>{});
    case DataType::FP8_E4M3:
        return view(TypeTag<types::fp8_e4m3>{});
    case DataType::FP8_E5M2:
        return view(TypeTag<types::fp8_e5m2>{});
    case DataType::FP8_E4M3_FNUZ:
        return view(TypeTag<types::fp8_e4m3_fnuz>{});
    case DataType::FP8_E5M2_FNUZ:
        return view(TypeTag<types::fp8_e5m2_fnuz>{});
    default:
        throw std::invalid_argument(
            std::string("live rows need a floating-point tensor, not one of data type ")
            + hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType));
    }
}

inline std::vector<int64_t> readLengths(hipdnn_data_sdk::utilities::ITensor& tensor)
{
    std::vector<int64_t> lengths;
    lengths.reserve(tensor.elementCount());
    const std::size_t elementSize = tensor.elementSize();
    for(void* value : tensor)
    {
        if(elementSize == sizeof(int32_t))
        {
            lengths.push_back(*static_cast<const int32_t*>(value));
        }
        else if(elementSize == sizeof(int64_t))
        {
            lengths.push_back(*static_cast<const int64_t*>(value));
        }
        else
        {
            throw std::runtime_error("an element is " + std::to_string(elementSize)
                                     + " bytes, not 4 or 8");
        }
    }
    return lengths;
}

} // namespace detail

// The tensor of one uid in the current run, or null when the run has none.
using RunTensorLookup = std::function<hipdnn_data_sdk::utilities::ITensor*(int64_t uid)>;

// Resolves the metadata `live_rows` of a bundle against the tensors of one run. A
// `lengths` entry reads the lengths tensor. A `like` entry takes the live lengths and
// the sequence axis of the ragged tensor. Throws std::runtime_error naming the output
// when a tensor is missing or not ragged, when a `like` tensor's sequence axis does not
// fit the output, or when a length is outside 0 to the output's extent on the axis.
inline LiveRowMap
    resolveLiveRows(const std::vector<hipdnn_integration_tests::OutputLiveRows>& entries,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& graph,
                    const RunTensorLookup& tensorFor)
{
    using hipdnn_integration_tests::bundle_metadata_detail::describeLiveRows;

    LiveRowMap map;
    const auto& attributes = graph.getTensorMap();
    for(const auto& entry : entries)
    {
        const auto fail = [&](const std::string& problem) {
            std::string message = describeLiveRows(entry.uid);
            message.append(": ").append(problem);
            return message;
        };
        const auto it = attributes.find(entry.uid);
        if(it == attributes.end() || it->second == nullptr || it->second->dims() == nullptr)
        {
            throw std::runtime_error(fail("the output is not a tensor of the graph"));
        }
        const std::vector<int64_t> dims(it->second->dims()->begin(), it->second->dims()->end());

        LiveRows rows;
        if(entry.likeUid.has_value())
        {
            const std::string likeName = "like tensor " + std::to_string(*entry.likeUid);
            const auto* like = tensorFor(*entry.likeUid);
            if(like == nullptr)
            {
                throw std::runtime_error(fail(likeName + " has no data in this run"));
            }
            auto layout = like->raggedIterationInfo();
            if(!layout.has_value())
            {
                throw std::runtime_error(fail(likeName + " is not ragged in this run"));
            }
            const int axis = layout->seqAxis;
            if(axis < 1 || static_cast<size_t>(axis) >= dims.size()
               || static_cast<size_t>(axis) >= like->dims().size())
            {
                throw std::runtime_error(fail(likeName + " has the sequence axis "
                                              + std::to_string(axis)
                                              + ", which the output does not have"));
            }
            if(like->dims()[static_cast<size_t>(axis)] != dims[static_cast<size_t>(axis)])
            {
                throw std::runtime_error(fail(likeName
                                              + " and the output have different extents on axis "
                                              + std::to_string(axis)));
            }
            rows.axis = axis;
            rows.lengths = std::move(layout->lengths);
        }
        else
        {
            const int64_t lengthsUid = entry.lengthsUid.value_or(0);
            const std::string lengthsName = "lengths tensor " + std::to_string(lengthsUid);
            auto* lengths = tensorFor(lengthsUid);
            if(lengths == nullptr)
            {
                throw std::runtime_error(fail(lengthsName + " has no data in this run"));
            }
            try
            {
                rows.lengths = detail::readLengths(*lengths);
            }
            catch(const std::exception& e)
            {
                throw std::runtime_error(fail(lengthsName + ": " + e.what()));
            }
            const int64_t axis = entry.axis.value_or(0);
            if(axis < 1 || axis >= static_cast<int64_t>(dims.size()))
            {
                throw std::runtime_error(
                    fail("axis " + std::to_string(axis) + " is not in the range 1 to "
                         + std::to_string(static_cast<int64_t>(dims.size()) - 1)));
            }
            rows.axis = static_cast<int>(axis);
        }

        const auto problem = detail::liveRowsProblem(dims, rows);
        if(!problem.empty())
        {
            throw std::runtime_error(fail(problem));
        }
        map.emplace(entry.uid, std::move(rows));
    }
    return map;
}

} // namespace hipdnn_integration_tests::bundle
