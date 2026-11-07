// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesConv.hpp>

#include "harness/input-init/FillRecipe.hpp"

// The dynamic tolerance for convolution-forward outputs stored as half or bfloat16.
//
// The fixed per-op default (TestTolerances.hpp conv::getToleranceFwd, 0.1 for both atol and
// rtol) is about 10% of a typical output from a reduction length of about 1,000 on: far looser
// than the output's own rounding at small reductions and unrelated to the error at large ones.
// This computes the tolerance from the graph and the input statistics instead; the model and its
// calibration are in docs/tolerances.md.

namespace hipdnn_integration_tests::tolerance
{

/// The two statistics of an input the tolerance needs.
struct InputMoments
{
    double meanSquare = 0.0;
    double maxAbs = 0.0;
};

/// Moments of one graph input by UID; nullopt when the UID is not an input held here or its
/// element type is not modeled. Asked only for non-virtual graph inputs that carry no baked
/// value: a baked value is read from the graph, not from here.
using InputMomentsLookup = std::function<std::optional<InputMoments>(int64_t uid)>;

/// Moments of a tensor filled from a recipe, without reading it: E[x^2] of a uniform draw on
/// [lo, hi] is (lo^2 + lo*hi + hi^2) / 3. The fill rounds each draw to the element type, which
/// changes the mean square by far less than the tolerance's margin.
inline InputMoments recipeMoments(const FillRecipe& recipe)
{
    const auto maxAbs = static_cast<double>(recipe.maxMagnitude());
    if(recipe.kind == FillRecipe::Kind::FIXED)
    {
        const auto v = static_cast<double>(recipe.value);
        return {v * v, maxAbs};
    }
    const auto lo = static_cast<double>(recipe.lo);
    const auto hi = static_cast<double>(recipe.hi);
    return {(lo * lo + lo * hi + hi * hi) / 3.0, maxAbs};
}

/// Moments of a tensor whose value is baked into the graph (a constant scalar, the way the
/// bundle generator emits a constant scale): the engine and the reference both read that value,
/// whatever the fill recipe says. nullopt when the tensor bakes no value, or one that is not a
/// finite number of the tensor's own type.
inline std::optional<InputMoments>
    bakedMoments(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs)
{
    if(attrs.value_type() == hipdnn_flatbuffers_sdk::data_objects::TensorValue::NONE)
    {
        return std::nullopt;
    }
    double v = 0.0;
    try
    {
        v = hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(&attrs, "baked value");
    }
    catch(const std::exception&)
    {
        return std::nullopt;
    }
    if(!std::isfinite(v))
    {
        return std::nullopt;
    }
    return InputMoments{v * v, std::abs(v)};
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
/// than from a fill recipe. nullopt for element types the model does not cover, an empty tensor,
/// and data holding a NaN or an infinity (no finite statistic describes it).
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
    if(count == 0 || !std::isfinite(sumOfSquares) || !std::isfinite(maxAbs))
    {
        return std::nullopt;
    }
    return InputMoments{sumOfSquares / static_cast<double>(count), maxAbs};
}

/// The error scale of a graph with exactly one ConvolutionFwd node that accumulates in float,
/// whose x and w are graph inputs (not virtual, not produced by another node). Every multiply
/// whose one operand comes from the convolution and the other is a graph input contributes that
/// operand's max |v| as an output scale; the scales of several multiplies multiply. The other
/// pointwise modes are not examined: each is assumed to change an error by at most its slope,
/// about 1, and so leaves it alone. That holds for the modes the shipped bundles use (ADD, MUL,
/// RELU_FWD); a mode with another gain (DIV, RECIPROCAL) is not detected. Nodes that are neither
/// convolution nor pointwise are ignored. The scale is one value for the whole graph: it does not
/// say which output a multiply sits on.
///
/// A graph input's moments are those of its baked value when it bakes one, else the lookup's. A
/// moment that is not finite makes the inputs unavailable, whichever source it came from.
///
/// nullopt when the graph is outside the model: no or several convolutions, a non-float
/// accumulator, an x or w that is not a graph input, a multiply that does not pair the
/// convolution's result with one graph input, or inputs whose moments are not available.
inline std::optional<hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale>
    convFwdErrorScale(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
                      const InputMomentsLookup& momentsFor)
{
    namespace data = hipdnn_flatbuffers_sdk::data_objects;

    const data::ConvolutionFwdAttributes* conv = nullptr;
    // Every tensor some node writes, as opposed to a graph input.
    std::set<int64_t> produced;
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
            produced.insert(conv->y_tensor_uid());
        }
        else if(node.attributes_type() == data::NodeAttributes::PointwiseAttributes)
        {
            produced.insert(node.attributes_as_PointwiseAttributes()->out_0_tensor_uid());
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
    const auto* wDims = wIt->second->dims();
    hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale scale;
    scale.accumulationLength = hipdnn_test_sdk::utilities::conv::convFpropAccumulationLength(
        std::vector<int64_t>(wDims->begin(), wDims->end()));
    if(scale.accumulationLength == 0)
    {
        return std::nullopt;
    }

    const auto graphInputMoments = [&](int64_t uid) -> std::optional<InputMoments> {
        const auto it = attrs.find(uid);
        if(it == attrs.end() || it->second->virtual_() || produced.count(uid) != 0)
        {
            return std::nullopt;
        }
        if(it->second->value_type() != data::TensorValue::NONE)
        {
            return bakedMoments(*it->second);
        }
        // A recipe's moments are taken as given, so one that is not finite (a fill of an
        // infinity) is stopped here, as hostMoments() and bakedMoments() stop theirs.
        const auto moments = momentsFor(uid);
        if(moments && !(std::isfinite(moments->meanSquare) && std::isfinite(moments->maxAbs)))
        {
            return std::nullopt;
        }
        return moments;
    };

    const auto xMoments = graphInputMoments(conv->x_tensor_uid());
    const auto wMoments = graphInputMoments(conv->w_tensor_uid());
    if(!xMoments || !wMoments)
    {
        return std::nullopt;
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
        if(pointwise->operation() != data::PointwiseMode::MUL)
        {
            continue;
        }
        // One operand carries the convolution's result (possibly after other pointwise
        // nodes), the other is the scale; which of in_0 and in_1 is which does not matter.
        if(!pointwise->in_1_tensor_uid().has_value())
        {
            return std::nullopt;
        }
        const int64_t first = pointwise->in_0_tensor_uid();
        const int64_t second = *pointwise->in_1_tensor_uid();
        const bool firstProduced = produced.count(first) != 0;
        if(firstProduced == (produced.count(second) != 0))
        {
            return std::nullopt;
        }
        const auto moments = graphInputMoments(firstProduced ? second : first);
        if(!moments)
        {
            return std::nullopt;
        }
        scale.outputScale *= moments->maxAbs;
    }
    return scale;
}

/// Whether convFwdOutputTolerance() models an output stored as this type: half and bfloat16.
/// A caller asks before it pays for the statistics of the graph's inputs.
inline bool hasDynamicToleranceFor(hipdnn_flatbuffers_sdk::data_objects::DataType outputType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    return outputType == DataType::HALF || outputType == DataType::BFLOAT16;
}

/// The dynamic tolerance for one output tensor, or nullopt when its type is not modeled
/// (see hasDynamicToleranceFor()).
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
