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
#include <type_traits>
#include <variant>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesConv.hpp>

#include "harness/input-init/FillRecipe.hpp"

// The dynamic tolerance for convolution outputs (forward, backward data and backward weights)
// stored as float, half or bfloat16.
//
// The fixed per-op defaults (TestTolerances.hpp conv::getToleranceFwd/Bwd/Wrw, one value for
// both atol and rtol) do not scale with the problem: for half and bfloat16 they are far looser
// than the output's own rounding, and for float they sit below the summation error of long
// reductions. This computes the tolerance from the graph and the input statistics instead; the
// model and its calibration are in docs/tolerances.md.
//
// It applies only to a graph whose convolution and pointwise nodes compute in float and whose
// intermediates (the tensors one node writes and another reads, virtual or not) are float: the
// model has one rounding, of the final result to the output type, and the roundings of
// intermediates stored in half or bfloat16 are not modelled (ALMIOPEN-3239). Such a graph keeps
// the per-op default.

namespace hipdnn_integration_tests::tolerance
{

/// The statistics of an input the tolerance needs.
struct InputMoments
{
    double meanSquare = 0.0;
    double maxAbs = 0.0;
    double mean = 0.0;
};

/// Moments of one graph input by UID; nullopt when the UID is not an input held here or its
/// element type is not modeled. Asked only for non-virtual graph inputs that carry no baked
/// value: a baked value is read from the graph, not from here.
using InputMomentsLookup = std::function<std::optional<InputMoments>(int64_t uid)>;

/// Moments of a tensor filled from a recipe, without reading it: a uniform draw on [lo, hi] has
/// E[x] = (lo + hi) / 2 and E[x^2] = (lo^2 + lo*hi + hi^2) / 3, and a value list the moments of
/// the list (the list repeats over the elements). The fill rounds each draw to the element type,
/// which changes the moments by far less than the tolerance's margin.
inline InputMoments recipeMoments(const FillRecipe& recipe)
{
    const auto maxAbs = static_cast<double>(recipe.maxMagnitude());
    if(recipe.kind == FillRecipe::Kind::FIXED)
    {
        const auto v = static_cast<double>(recipe.value);
        return {v * v, maxAbs, v};
    }
    if(recipe.kind == FillRecipe::Kind::VALUES && recipe.list != nullptr)
    {
        return std::visit(
            [maxAbs](const auto& numbers) {
                double sum = 0.0;
                double sumOfSquares = 0.0;
                for(const auto number : numbers)
                {
                    const auto v = static_cast<double>(number);
                    sum += v;
                    sumOfSquares += v * v;
                }
                const auto n = static_cast<double>(numbers.size());
                return InputMoments{sumOfSquares / n, maxAbs, sum / n};
            },
            *recipe.list);
    }
    const auto lo = static_cast<double>(recipe.lo);
    const auto hi = static_cast<double>(recipe.hi);
    return {(lo * lo + lo * hi + hi * hi) / 3.0, maxAbs, (lo + hi) / 2.0};
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
    return InputMoments{v * v, std::abs(v), v};
}

namespace detail
{

struct ElementSums
{
    double sum = 0.0;
    double sumOfSquares = 0.0;
    double maxAbs = 0.0;
    uint64_t count = 0;
};

template <typename T>
void accumulateElements(hipdnn_data_sdk::utilities::ITensor& tensor, ElementSums& sums)
{
    const auto add = [&](double v) {
        sums.sum += v;
        sums.sumOfSquares += v * v;
        sums.maxAbs = std::max(sums.maxAbs, std::abs(v));
        ++sums.count;
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
    detail::ElementSums sums;
    switch(dataType)
    {
    case DataType::FLOAT:
        detail::accumulateElements<float>(tensor, sums);
        break;
    case DataType::HALF:
        detail::accumulateElements<hipdnn_data_sdk::types::half>(tensor, sums);
        break;
    case DataType::BFLOAT16:
        detail::accumulateElements<hipdnn_data_sdk::types::bfloat16>(tensor, sums);
        break;
    case DataType::DOUBLE:
        detail::accumulateElements<double>(tensor, sums);
        break;
    default:
        return std::nullopt;
    }
    if(sums.count == 0 || !std::isfinite(sums.sumOfSquares) || !std::isfinite(sums.maxAbs))
    {
        return std::nullopt;
    }
    const auto n = static_cast<double>(sums.count);
    return InputMoments{sums.sumOfSquares / n, sums.maxAbs, sums.sum / n};
}

/// The error scale of the one convolution of a graph, by direction: forward, backward data
/// (dgrad) or backward weights (wrw).
using ConvErrorScale = std::variant<hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale,
                                    hipdnn_test_sdk::utilities::conv::ConvDgradErrorScale,
                                    hipdnn_test_sdk::utilities::conv::ConvWrwErrorScale>;

namespace detail
{

/// The convolution node of a graph, whichever its direction. `first` and `second` are the two
/// tensors whose products it sums: x and w forward, dy and w for dgrad, x and dy for wrw.
struct ConvNode
{
    hipdnn_flatbuffers_sdk::data_objects::NodeAttributes type
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::NONE;
    int64_t first = 0;
    int64_t second = 0;
    int64_t result = 0;
    const flatbuffers::Vector<int64_t>* stride = nullptr;
    const flatbuffers::Vector<int64_t>* dilation = nullptr;
};

inline std::optional<ConvNode> convNodeOf(const hipdnn_flatbuffers_sdk::data_objects::Node& node)
{
    namespace data = hipdnn_flatbuffers_sdk::data_objects;
    switch(node.attributes_type())
    {
    case data::NodeAttributes::ConvolutionFwdAttributes:
    {
        const auto* a = node.attributes_as_ConvolutionFwdAttributes();
        return ConvNode{node.attributes_type(),
                        a->x_tensor_uid(),
                        a->w_tensor_uid(),
                        a->y_tensor_uid(),
                        a->stride(),
                        a->dilation()};
    }
    case data::NodeAttributes::ConvolutionBwdAttributes:
    {
        const auto* a = node.attributes_as_ConvolutionBwdAttributes();
        return ConvNode{node.attributes_type(),
                        a->dy_tensor_uid(),
                        a->w_tensor_uid(),
                        a->dx_tensor_uid(),
                        a->stride(),
                        a->dilation()};
    }
    case data::NodeAttributes::ConvolutionWrwAttributes:
    {
        const auto* a = node.attributes_as_ConvolutionWrwAttributes();
        return ConvNode{node.attributes_type(),
                        a->x_tensor_uid(),
                        a->dy_tensor_uid(),
                        a->dw_tensor_uid(),
                        a->stride(),
                        a->dilation()};
    }
    default:
        return std::nullopt;
    }
}

/// The dims of a tensor of the graph with at least two of them, none negative; nullopt otherwise.
inline std::optional<std::vector<int64_t>>
    convDims(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper, int64_t uid)
{
    const auto& attrs = wrapper.getTensorMap();
    const auto it = attrs.find(uid);
    if(it == attrs.end() || it->second->dims() == nullptr || it->second->dims()->size() < 2)
    {
        return std::nullopt;
    }
    std::vector<int64_t> dims(it->second->dims()->begin(), it->second->dims()->end());
    if(std::any_of(dims.begin(), dims.end(), [](int64_t extent) { return extent < 0; }))
    {
        return std::nullopt;
    }
    return dims;
}

/// K of the convolution: the products summed per output element. nullopt when the tensors or
/// the parameters do not describe a convolution the count is defined for, or K is 0.
inline std::optional<uint64_t> convAccumulationLength(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper, const ConvNode& conv)
{
    namespace data = hipdnn_flatbuffers_sdk::data_objects;
    namespace sdk = hipdnn_test_sdk::utilities::conv;
    uint64_t k = 0;
    if(conv.type == data::NodeAttributes::ConvolutionFwdAttributes)
    {
        const auto wDims = convDims(wrapper, conv.second);
        if(!wDims)
        {
            return std::nullopt;
        }
        k = sdk::convFpropAccumulationLength(*wDims);
    }
    else if(conv.type == data::NodeAttributes::ConvolutionBwdAttributes)
    {
        const auto wDims = convDims(wrapper, conv.second);
        const auto dxDims = convDims(wrapper, conv.result);
        if(!wDims || !dxDims || conv.stride == nullptr || conv.dilation == nullptr)
        {
            return std::nullopt;
        }
        const std::vector<int64_t> stride(conv.stride->begin(), conv.stride->end());
        const std::vector<int64_t> dilation(conv.dilation->begin(), conv.dilation->end());
        const int64_t channelsPerGroup = (*wDims)[1];
        const auto atLeastOne = [](int64_t v) { return v >= 1; };
        if(channelsPerGroup < 1 || (*dxDims)[1] % channelsPerGroup != 0
           || stride.size() + 2 != wDims->size() || dilation.size() + 2 != wDims->size()
           || !std::all_of(stride.begin(), stride.end(), atLeastOne)
           || !std::all_of(dilation.begin(), dilation.end(), atLeastOne))
        {
            return std::nullopt;
        }
        const int64_t groups = (*dxDims)[1] / channelsPerGroup;
        if(groups < 1 || (*wDims)[0] % groups != 0)
        {
            return std::nullopt;
        }
        k = sdk::convDgradAccumulationLength(*wDims, groups, stride, dilation);
    }
    else
    {
        const auto dyDims = convDims(wrapper, conv.second);
        if(!dyDims)
        {
            return std::nullopt;
        }
        k = sdk::convWrwAccumulationLength(*dyDims);
    }
    if(k == 0)
    {
        return std::nullopt;
    }
    return k;
}

} // namespace detail

/// The error scale of a graph with exactly one convolution node (ConvolutionFwd, ConvolutionBwd
/// or ConvolutionWrw) that accumulates in float, whose two operands (x and w forward, dy and w
/// for dgrad, x and dy for wrw) are graph inputs (not virtual, not produced by another node).
/// Every multiply whose one operand comes from the convolution and the other is a graph input
/// contributes that operand's max |v| as an output scale; the scales of several multiplies
/// multiply. The other pointwise modes are not examined: each is assumed to change an error by at
/// most its slope, about 1, and so leaves it alone. The activations of the conv test-plan bundles
/// (relu with its clips, identity, sigmoid, tanh, softplus, elu, swish, both gelus, at the
/// bundles' parameters, and the backward modes of the same functions behind a dgrad) have a slope
/// of at most about 1.13; a mode with another gain (DIV, RECIPROCAL) is not detected. Nodes that
/// are neither convolution nor pointwise are ignored. The scale is one value for the whole graph:
/// it does not say which output a multiply sits on.
///
/// The tolerance is sized by the convolution's output, not by the graph's. For half, behind an
/// activation that bounds its output (sigmoid, tanh, a clipped relu) the atol can exceed the
/// range of that output, and then no output can fail it: in the conv-fwd test-plan bundles the
/// full-tier fp16 sigmoid case L19_1d_i1024_n1c64k64_f1024_fp16_cl_sigmoid (K = 65,536) has atol
/// 1.0014 against [0, 1], and 43 of the 268 half bounded-activation cases have an atol above the
/// fixed 0.1. bfloat16 and float are not affected.
///
/// A graph input's moments are those of its baked value when it bakes one, else the lookup's. A
/// moment that is not finite makes the inputs unavailable, whichever source it came from.
///
/// nullopt when the graph is outside the model: no or several convolutions, a non-float
/// accumulator, a pointwise node computing in a type other than float, an intermediate (a tensor
/// one node writes and another reads, virtual or not: the convolution's result and the pointwise
/// results) stored as a type other than float, an operand that is not a graph input, a multiply
/// that does not pair the convolution's result with one graph input, dims or parameters for which
/// K is not defined or is 0, or inputs whose moments are not available. An unset type counts as
/// not float. The model has one rounding, of the final result to the output type; the roundings
/// of intermediates stored in half or bfloat16 are not modelled (ALMIOPEN-3239).
inline std::optional<ConvErrorScale>
    convErrorScale(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
                   const InputMomentsLookup& momentsFor)
{
    namespace data = hipdnn_flatbuffers_sdk::data_objects;
    namespace sdk = hipdnn_test_sdk::utilities::conv;

    std::optional<detail::ConvNode> conv;
    // Every tensor some node writes, as opposed to a graph input.
    std::set<int64_t> produced;
    // Every tensor some node reads.
    std::set<int64_t> consumed;
    for(uint32_t i = 0; i < wrapper.nodeCount(); ++i)
    {
        const auto& node = wrapper.getNode(i);
        if(const auto found = detail::convNodeOf(node))
        {
            if(conv.has_value() || node.compute_data_type() != data::DataType::FLOAT)
            {
                return std::nullopt;
            }
            conv = found;
            produced.insert(conv->result);
            consumed.insert(conv->first);
            consumed.insert(conv->second);
        }
        else if(node.attributes_type() == data::NodeAttributes::PointwiseAttributes)
        {
            if(node.compute_data_type() != data::DataType::FLOAT)
            {
                return std::nullopt;
            }
            const auto* pointwise = node.attributes_as_PointwiseAttributes();
            produced.insert(pointwise->out_0_tensor_uid());
            consumed.insert(pointwise->in_0_tensor_uid());
            for(const auto operand : {pointwise->in_1_tensor_uid(), pointwise->in_2_tensor_uid()})
            {
                if(operand.has_value())
                {
                    consumed.insert(*operand);
                }
            }
        }
    }
    if(!conv.has_value())
    {
        return std::nullopt;
    }

    const auto& attrs = wrapper.getTensorMap();
    // An intermediate, one node writes and another reads, is rounded to its own type, virtual or
    // not. The graph's outputs, which no node reads, are rounded once and are the model's.
    for(const auto uid : produced)
    {
        if(consumed.count(uid) == 0)
        {
            continue;
        }
        const auto it = attrs.find(uid);
        if(it == attrs.end() || it->second->data_type() != data::DataType::FLOAT)
        {
            return std::nullopt;
        }
    }
    const auto accumulationLength = detail::convAccumulationLength(wrapper, *conv);
    if(!accumulationLength)
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
        if(moments
           && !(std::isfinite(moments->meanSquare) && std::isfinite(moments->maxAbs)
                && std::isfinite(moments->mean)))
        {
            return std::nullopt;
        }
        return moments;
    };

    const auto firstMoments = graphInputMoments(conv->first);
    const auto secondMoments = graphInputMoments(conv->second);
    if(!firstMoments || !secondMoments)
    {
        return std::nullopt;
    }

    double outputScale = 1.0;
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
        outputScale *= moments->maxAbs;
    }

    switch(conv->type)
    {
    case data::NodeAttributes::ConvolutionFwdAttributes:
        return sdk::ConvFpropErrorScale{*accumulationLength,
                                        firstMoments->meanSquare,
                                        secondMoments->meanSquare,
                                        outputScale,
                                        firstMoments->mean,
                                        secondMoments->mean};
    case data::NodeAttributes::ConvolutionBwdAttributes:
        return sdk::ConvDgradErrorScale{*accumulationLength,
                                        firstMoments->meanSquare,
                                        secondMoments->meanSquare,
                                        outputScale,
                                        firstMoments->mean,
                                        secondMoments->mean};
    default:
        return sdk::ConvWrwErrorScale{*accumulationLength,
                                      firstMoments->meanSquare,
                                      secondMoments->meanSquare,
                                      outputScale,
                                      firstMoments->mean,
                                      secondMoments->mean};
    }
}

/// Whether convOutputTolerance() models an output stored as this type: float, half and
/// bfloat16. A caller asks before it pays for the statistics of the graph's inputs.
inline bool hasDynamicToleranceFor(hipdnn_flatbuffers_sdk::data_objects::DataType outputType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    return outputType == DataType::FLOAT || outputType == DataType::HALF
           || outputType == DataType::BFLOAT16;
}

namespace detail
{

template <typename OutputType>
hipdnn_test_sdk::utilities::conv::AbsRelTolerance convOutputToleranceAs(const ConvErrorScale& scale)
{
    namespace sdk = hipdnn_test_sdk::utilities::conv;
    return std::visit(
        [](const auto& s) -> sdk::AbsRelTolerance {
            using Scale = std::decay_t<decltype(s)>;
            if constexpr(std::is_same_v<Scale, sdk::ConvFpropErrorScale>)
            {
                return sdk::calculateConvFpropOutputTolerance<OutputType>(s);
            }
            else if constexpr(std::is_same_v<Scale, sdk::ConvDgradErrorScale>)
            {
                return sdk::calculateConvDgradOutputTolerance<OutputType>(s);
            }
            else
            {
                return sdk::calculateConvWrwOutputTolerance<OutputType>(s);
            }
        },
        scale);
}

} // namespace detail

/// The dynamic tolerance for one output tensor, or nullopt when its type is not modeled
/// (see hasDynamicToleranceFor()).
inline std::optional<hipdnn_test_sdk::utilities::conv::AbsRelTolerance>
    convOutputTolerance(const ConvErrorScale& scale,
                        hipdnn_flatbuffers_sdk::data_objects::DataType outputType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(outputType)
    {
    case DataType::FLOAT:
        return detail::convOutputToleranceAs<float>(scale);
    case DataType::HALF:
        return detail::convOutputToleranceAs<hipdnn_data_sdk::types::half>(scale);
    case DataType::BFLOAT16:
        return detail::convOutputToleranceAs<hipdnn_data_sdk::types::bfloat16>(scale);
    default:
        return std::nullopt;
    }
}

} // namespace hipdnn_integration_tests::tolerance
