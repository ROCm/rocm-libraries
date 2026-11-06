// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesConv.hpp>

// The dynamic tolerance for convolution-forward outputs stored as half or bfloat16.
//
// The fixed per-op default (TestTolerances.hpp conv::getToleranceFwd, 0.1 for both atol and
// rtol) is about 10% of a typical output at every size: far looser than the output's own
// rounding at small reductions and unrelated to the error at large ones. This computes the
// tolerance from the graph and the input tensors instead; the model and its calibration are
// described with calculateConvFpropOutputTolerance() and in docs/tolerances.md.

namespace hipdnn_integration_tests::tolerance
{

/// Finds an input tensor by UID; nullptr when the UID is not a graph input held here.
using InputTensorLookup = std::function<hipdnn_data_sdk::utilities::ITensor*(int64_t uid)>;

namespace detail
{

template <typename T>
void accumulateSquares(hipdnn_data_sdk::utilities::ITensor& tensor,
                       double& sumOfSquares,
                       double& maxAbs,
                       uint64_t& count)
{
    for(auto it = tensor.begin(); it != tensor.end(); ++it)
    {
        const auto value = static_cast<double>(*static_cast<const T*>(*it));
        sumOfSquares += value * value;
        maxAbs = std::max(maxAbs, std::abs(value));
        ++count;
    }
}

struct Moments
{
    double meanSquare = 0.0;
    double maxAbs = 0.0;
};

/// Mean of v^2 and max |v| over the logical elements of a tensor, read on the host.
/// nullopt for element types the model does not cover.
inline std::optional<Moments> hostMoments(hipdnn_data_sdk::utilities::ITensor& tensor,
                                          hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    double sumOfSquares = 0.0;
    double maxAbs = 0.0;
    uint64_t count = 0;
    switch(dataType)
    {
    case DataType::FLOAT:
        accumulateSquares<float>(tensor, sumOfSquares, maxAbs, count);
        break;
    case DataType::HALF:
        accumulateSquares<hipdnn_data_sdk::types::half>(tensor, sumOfSquares, maxAbs, count);
        break;
    case DataType::BFLOAT16:
        accumulateSquares<hipdnn_data_sdk::types::bfloat16>(tensor, sumOfSquares, maxAbs, count);
        break;
    case DataType::DOUBLE:
        accumulateSquares<double>(tensor, sumOfSquares, maxAbs, count);
        break;
    default:
        return std::nullopt;
    }
    if(count == 0)
    {
        return std::nullopt;
    }
    return Moments{sumOfSquares / static_cast<double>(count), maxAbs};
}

} // namespace detail

/// The error scale of a graph with exactly one ConvolutionFwd node that accumulates in float,
/// whose x and w are graph inputs. Any multiply in the graph whose second operand is a graph
/// input contributes that operand's max |v| as an output scale; other pointwise operations
/// change an error by at most their slope, which is close to 1 for every supported activation.
///
/// nullopt when the graph is outside the model: no or several convolutions, a non-float
/// accumulator, or inputs that are not held here.
inline std::optional<hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale>
    convFwdErrorScale(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
                      const InputTensorLookup& inputFor)
{
    namespace data = hipdnn_flatbuffers_sdk::data_objects;

    const data::ConvolutionFwdAttributes* conv = nullptr;
    for(uint32_t i = 0; i < wrapper.nodeCount(); ++i)
    {
        const auto& node = wrapper.getNode(i);
        if(node.attributes_type() == data::NodeAttributes::ConvolutionFwdAttributes)
        {
            if(conv != nullptr || node.compute_data_type() != data::DataType::FLOAT)
            {
                return std::nullopt;
            }
            conv = node.attributes_as_ConvolutionFwdAttributes();
        }
    }
    if(conv == nullptr)
    {
        return std::nullopt;
    }

    const auto& attrs = wrapper.getTensorMap();
    auto* x = inputFor(conv->x_tensor_uid());
    auto* w = inputFor(conv->w_tensor_uid());
    if(x == nullptr || w == nullptr || w->dims().size() < 2)
    {
        return std::nullopt;
    }
    const auto xMoments = detail::hostMoments(*x, attrs.at(conv->x_tensor_uid())->data_type());
    const auto wMoments = detail::hostMoments(*w, attrs.at(conv->w_tensor_uid())->data_type());
    if(!xMoments || !wMoments)
    {
        return std::nullopt;
    }

    hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale scale;
    scale.accumulationLength = 1;
    for(size_t d = 1; d < w->dims().size(); ++d)
    {
        scale.accumulationLength *= static_cast<uint64_t>(w->dims()[d]);
    }
    scale.inputMeanSquare = xMoments->meanSquare;
    scale.weightMeanSquare = wMoments->meanSquare;

    for(uint32_t i = 0; i < wrapper.nodeCount(); ++i)
    {
        const auto& node = wrapper.getNode(i);
        if(node.attributes_type() != data::NodeAttributes::PointwiseAttributes)
        {
            continue;
        }
        const auto* pointwise = node.attributes_as_PointwiseAttributes();
        if(pointwise->operation() != data::PointwiseMode::MUL
           || !pointwise->in_1_tensor_uid().has_value())
        {
            continue;
        }
        const int64_t operandUid = *pointwise->in_1_tensor_uid();
        auto* operand = inputFor(operandUid);
        if(operand == nullptr)
        {
            continue;
        }
        const auto moments = detail::hostMoments(*operand, attrs.at(operandUid)->data_type());
        if(!moments)
        {
            return std::nullopt;
        }
        scale.outputScale *= moments->maxAbs;
    }
    return scale;
}

/// The dynamic tolerance for one output tensor, or nullopt when its type is not modeled
/// (only half and bfloat16 outputs are).
inline std::optional<hipdnn_test_sdk::utilities::conv::AbsRelTolerance>
    convFwdOutputTolerance(const hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale& scale,
                           hipdnn_flatbuffers_sdk::data_objects::DataType outputType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    namespace conv = hipdnn_test_sdk::utilities::conv;
    switch(outputType)
    {
    case DataType::HALF:
        return conv::calculateConvFpropOutputTolerance<hipdnn_data_sdk::types::half>(scale);
    case DataType::BFLOAT16:
        return conv::calculateConvFpropOutputTolerance<hipdnn_data_sdk::types::bfloat16>(scale);
    default:
        return std::nullopt;
    }
}

} // namespace hipdnn_integration_tests::tolerance
