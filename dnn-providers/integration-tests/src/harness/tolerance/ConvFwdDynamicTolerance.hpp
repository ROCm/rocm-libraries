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

#include "harness/input-init/FillRecipe.hpp"

// The dynamic tolerance for convolution-forward outputs stored as half or bfloat16.
//
// The fixed per-op default (TestTolerances.hpp conv::getToleranceFwd, 0.1 for both atol and
// rtol) is about 10% of a typical output at every size: far looser than the output's own
// rounding at small reductions and unrelated to the error at large ones. This computes the
// tolerance from the graph and the input statistics instead; the model and its calibration
// are described with calculateConvFpropOutputTolerance() and in docs/tolerances.md.

namespace hipdnn_integration_tests::tolerance
{

/// The two statistics of an input the tolerance needs.
struct InputMoments
{
    double meanSquare = 0.0;
    double maxAbs = 0.0;
};

/// Moments of one graph input by UID; nullopt when the UID is not an input held here or its
/// element type is not modeled.
using InputMomentsLookup = std::function<std::optional<InputMoments>(int64_t uid)>;

/// Moments of a tensor filled from a recipe, without reading it: E[x^2] of a uniform draw on
/// [lo, hi] is (lo^2 + lo*hi + hi^2) / 3. The fill rounds each draw to the element type, which
/// changes the mean square by far less than the tolerance's margin.
inline InputMoments recipeMoments(const FillRecipe& recipe)
{
    if(recipe.kind == FillRecipe::Kind::FIXED)
    {
        const auto v = static_cast<double>(recipe.value);
        return {v * v, std::abs(v)};
    }
    const auto lo = static_cast<double>(recipe.lo);
    const auto hi = static_cast<double>(recipe.hi);
    return {(lo * lo + lo * hi + hi * hi) / 3.0, std::max(std::abs(lo), std::abs(hi))};
}

namespace detail
{

template <typename T>
void accumulateSquares(hipdnn_data_sdk::utilities::ITensor& tensor,
                       double& sumOfSquares,
                       double& maxAbs,
                       uint64_t& count)
{
    const auto add = [&](double v) {
        sumOfSquares += v * v;
        maxAbs = std::max(maxAbs, std::abs(v));
        ++count;
    };
    if(tensor.isPacked() && !tensor.raggedIterationInfo().has_value())
    {
        // Every element exactly once, in some order: a linear walk of the buffer suffices
        // and is far cheaper than the index-tracking iterator.
        const auto* data = static_cast<const T*>(tensor.rawHostData());
        const auto n = tensor.elementCount();
        for(size_t i = 0; i < n; ++i)
        {
            add(static_cast<double>(data[i]));
        }
        return;
    }
    for(auto it = tensor.begin(); it != tensor.end(); ++it)
    {
        add(static_cast<double>(*static_cast<const T*>(*it)));
    }
}

} // namespace detail

/// Moments of a tensor read on the host: for inputs that came from data (golden blobs) rather
/// than from a fill recipe. nullopt for element types the model does not cover.
inline std::optional<InputMoments>
    hostMoments(hipdnn_data_sdk::utilities::ITensor& tensor,
                hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    double sumOfSquares = 0.0;
    double maxAbs = 0.0;
    uint64_t count = 0;
    switch(dataType)
    {
    case DataType::FLOAT:
        detail::accumulateSquares<float>(tensor, sumOfSquares, maxAbs, count);
        break;
    case DataType::HALF:
        detail::accumulateSquares<hipdnn_data_sdk::types::half>(
            tensor, sumOfSquares, maxAbs, count);
        break;
    case DataType::BFLOAT16:
        detail::accumulateSquares<hipdnn_data_sdk::types::bfloat16>(
            tensor, sumOfSquares, maxAbs, count);
        break;
    case DataType::DOUBLE:
        detail::accumulateSquares<double>(tensor, sumOfSquares, maxAbs, count);
        break;
    default:
        return std::nullopt;
    }
    if(count == 0)
    {
        return std::nullopt;
    }
    return InputMoments{sumOfSquares / static_cast<double>(count), maxAbs};
}

/// The error scale of a graph with exactly one ConvolutionFwd node that accumulates in float,
/// whose x and w are graph inputs. Any multiply in the graph whose second operand is a graph
/// input contributes that operand's max |v| as an output scale; other pointwise operations
/// change an error by at most their slope, which is close to 1 for every supported activation.
///
/// nullopt when the graph is outside the model: no or several convolutions, a non-float
/// accumulator, or inputs whose moments are not available.
inline std::optional<hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale>
    convFwdErrorScale(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
                      const InputMomentsLookup& momentsFor)
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
    const auto wIt = attrs.find(conv->w_tensor_uid());
    if(wIt == attrs.end() || wIt->second->dims() == nullptr || wIt->second->dims()->size() < 2)
    {
        return std::nullopt;
    }
    const auto xMoments = momentsFor(conv->x_tensor_uid());
    const auto wMoments = momentsFor(conv->w_tensor_uid());
    if(!xMoments || !wMoments)
    {
        return std::nullopt;
    }

    hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale scale;
    scale.accumulationLength = 1;
    const auto* wDims = wIt->second->dims();
    for(flatbuffers::uoffset_t d = 1; d < wDims->size(); ++d)
    {
        scale.accumulationLength *= static_cast<uint64_t>(wDims->Get(d));
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
        if(attrs.find(operandUid) == attrs.end() || attrs.at(operandUid)->virtual_())
        {
            continue;
        }
        const auto moments = momentsFor(operandUid);
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
