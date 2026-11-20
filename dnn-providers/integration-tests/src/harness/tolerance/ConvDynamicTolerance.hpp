// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
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
// stored as half, bfloat16 or float.
//
// The fixed per-op defaults (TestTolerances.hpp conv::getToleranceFwd/Bwd/Wrw, one value for both
// atol and rtol) do not scale with the problem: for half and bfloat16 they are far looser than the
// output's own rounding at small reductions and unrelated to the error at large ones, and for
// float far tighter than the order of a float summation needs from a reduction length of a few
// hundred on. This computes the tolerance from the graph and the input statistics instead. The
// model and its calibration are in docs/tolerances.md.
//
// It applies to a graph whose convolution accumulates in float and whose pointwise nodes and
// intermediates (the tensors one node writes and another reads, virtual or not) are half,
// bfloat16, float or double: the roundings to half or bfloat16 between the convolution and the
// output are rounding points that widen atol (ALMIOPEN-3239). A graph with a type the model does
// not know keeps the per-op default. A float forward output is graded with the summation-order
// coefficient the engine's TOML sets, or the default of the SDK (ALMIOPEN-3249); backward data
// and backward weights use their calibrated constants.

namespace hipdnn_integration_tests::tolerance
{

/// The statistics of an input the tolerance needs.
struct InputMoments
{
    double meanSquare = 0.0;
    double maxAbs = 0.0;
    double mean = 0.0;
    /// Every element holds one value: a FIXED fill recipe, or a value baked into the graph.
    bool constant = false;
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
        return {v * v, maxAbs, v, true};
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
    return InputMoments{v * v, std::abs(v), v, true};
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

namespace detail
{

/// Machine epsilon of an element type the model can round to; nullopt for any other, an unset
/// type included.
inline std::optional<double> epsilonOf(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(type)
    {
    case DataType::FLOAT:
        return static_cast<double>(std::numeric_limits<float>::epsilon());
    case DataType::DOUBLE:
        return std::numeric_limits<double>::epsilon();
    case DataType::HALF:
        return static_cast<double>(std::numeric_limits<hipdnn_data_sdk::types::half>::epsilon());
    case DataType::BFLOAT16:
        return static_cast<double>(
            std::numeric_limits<hipdnn_data_sdk::types::bfloat16>::epsilon());
    default:
        return std::nullopt;
    }
}

/// An element type with fewer bits than float: a value stored or computed in it is rounded.
inline bool isNarrowerThanFloat(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    return type == DataType::HALF || type == DataType::BFLOAT16;
}

/// The tensors a pointwise node reads, in_0 first; unset operands stay empty.
inline std::array<std::optional<int64_t>, 3>
    operandsOf(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwise)
{
    std::array<std::optional<int64_t>, 3> operands;
    operands[0] = pointwise.in_0_tensor_uid();
    operands[1] = pointwise.in_1_tensor_uid();
    operands[2] = pointwise.in_2_tensor_uid();
    return operands;
}

/// Whether the model takes the pointwise node to change an error in an operand by at most its
/// slope, about 1: ADD, SUBTRACT, BINARY_SELECT (which passes one operand on), IDENTITY, the
/// activations the conv test-plan bundles use (RELU_FWD, SIGMOID_FWD, TANH_FWD, SOFTPLUS_FWD,
/// ELU_FWD, SWISH_FWD, GELU_FWD, GELU_APPROX_TANH_FWD), their backward modes behind a dgrad
/// (which multiply the gradient by the activation's derivative at the forward input: at most 1
/// for relu, sigmoid, tanh and softplus, about 1.1 for swish and both gelus) and MUL, whose scale
/// the walk reads. Every other mode (DIV, RECIPROCAL, SQRT, ...) has no slope this model knows,
/// and so has a parameter that steepens one of the above: a leaky relu with a slope above 1 and
/// an elu with an alpha above 1, forward or backward.
inline bool
    hasModeledSlope(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwise)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    switch(pointwise.operation())
    {
    case PointwiseMode::ADD:
    case PointwiseMode::SUB:
    case PointwiseMode::MUL:
    case PointwiseMode::BINARY_SELECT:
    case PointwiseMode::IDENTITY:
    case PointwiseMode::SIGMOID_FWD:
    case PointwiseMode::TANH_FWD:
    case PointwiseMode::SOFTPLUS_FWD:
    case PointwiseMode::SWISH_FWD:
    case PointwiseMode::GELU_FWD:
    case PointwiseMode::GELU_APPROX_TANH_FWD:
    case PointwiseMode::SIGMOID_BWD:
    case PointwiseMode::TANH_BWD:
    case PointwiseMode::SOFTPLUS_BWD:
    case PointwiseMode::SWISH_BWD:
    case PointwiseMode::GELU_BWD:
    case PointwiseMode::GELU_APPROX_TANH_BWD:
        return true;
    case PointwiseMode::RELU_FWD:
    case PointwiseMode::RELU_BWD:
        return !pointwise.relu_lower_clip_slope().has_value()
               || std::abs(*pointwise.relu_lower_clip_slope()) <= 1.0f;
    case PointwiseMode::ELU_FWD:
    case PointwiseMode::ELU_BWD:
        return !pointwise.elu_alpha().has_value() || std::abs(*pointwise.elu_alpha()) <= 1.0f;
    default:
        return false;
    }
}

/// Whether the mode is the backward form of an activation: it multiplies a gradient (in_0) by the
/// activation's derivative at the forward input (in_1). The model prices it only behind a
/// ConvolutionBwd, whose result is that gradient.
inline bool isBackwardActivation(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    switch(mode)
    {
    case PointwiseMode::RELU_BWD:
    case PointwiseMode::SIGMOID_BWD:
    case PointwiseMode::TANH_BWD:
    case PointwiseMode::SOFTPLUS_BWD:
    case PointwiseMode::ELU_BWD:
    case PointwiseMode::SWISH_BWD:
    case PointwiseMode::GELU_BWD:
    case PointwiseMode::GELU_APPROX_TANH_BWD:
        return true;
    default:
        return false;
    }
}

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

/// The error scale of the one convolution of a graph, by direction: forward, backward data
/// (dgrad) or backward weights (wrw).
using ConvErrorScale = std::variant<hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale,
                                    hipdnn_test_sdk::utilities::conv::ConvDgradErrorScale,
                                    hipdnn_test_sdk::utilities::conv::ConvWrwErrorScale>;

/// The error scale of one output of a graph with exactly one convolution node (ConvolutionFwd,
/// ConvolutionBwd or ConvolutionWrw) that accumulates in float, whose two operands (x and w
/// forward, dy and w for dgrad, x and dy for wrw) are graph inputs (not virtual, not produced by
/// another node), and that has nothing else but pointwise nodes. For dgrad, dx's channels must be
/// a multiple of the weight's, the group count that gives must divide the weight's output
/// channels, and stride and dilation need one entry of at least 1 per filter dimension.
/// `outputUid` is the output that is graded: the convolution's result or the result of a
/// pointwise node. Another output of the graph has its own scale, and what lies on its path only
/// is not part of this one.
///
/// outputScale is the factor by which the output carries the convolution's result. A multiply
/// whose one operand carries it and the other is a graph input multiplies the factor by that
/// input's max |v|; an add, a subtract and every other mode with several operands add the factors
/// (two paths from the convolution carry the same error, so they add: ADD(MUL(y, 0.5), y) is
/// 1.5 y); a mode with one operand leaves the factor as it is. A multiply on the branch of
/// another output does not enter it. The modes that leave an error alone are the ones
/// detail::hasModeledSlope() lists, each assumed to change an error by at most its slope, about
/// 1: the activations of the conv test-plan bundles (relu with its clips, identity, sigmoid,
/// tanh, softplus, elu, swish, both gelus, at the bundles' parameters, and their backward modes
/// behind a dgrad) have a slope of at most about 1.13. Any other mode, and a node that is neither
/// the convolution nor pointwise, leave the graph outside the model.
///
/// The roundings below float. The graph is walked from the convolution to its outputs, and every
/// place where a value is rounded to half or bfloat16 on the way to the graded output becomes a
/// RoundingPoint:
///  - a tensor one node writes and another reads, stored as half or bfloat16 (virtual or not),
///    other than the graded output itself, whose rounding to its own type is the model's rtol;
///  - an operand of a pointwise node whose compute type is half or bfloat16, when the operand
///    is stored as another type (the node converts it, and rounds);
///  - the result of such a node, when it is stored as another type than the node computes in,
///    whether or not a node reads it: a graph output that is one of these results is rounded
///    twice, and the first rounding is a point.
/// Any other result that no node reads is a graph output: its one rounding to its own type is the
/// model's rtol, not a point. A node that does not reach the graded output has no point here.
///
/// The last two kinds, a node's own conversions and result in its compute type, were first written
/// to cover a CPU reference that ignored the node's compute type (it added in double and rounded
/// once). The reference honours it since #15139, and re-measured against it the terms are still
/// needed: a one-step difference in the convolution's sum can flip the conversion of its result to
/// a half or bfloat16 compute type, and a later add can cancel the rest. The points at stored
/// tensors stay for the same reason.
///
/// The magnitude at a point is a root mean square. A value is modeled as a factor (the one above)
/// times the convolution's result plus a remainder that does not depend on it: a multiply
/// multiplies both by the scale's max |v|; an add, a subtract and every other mode with several
/// operands add the factors and the remainders' mean squares (independent inputs add in
/// quadrature); a mode with one operand leaves the value as it is. The slope from a point to the
/// graded output is the sum, over the paths between the two, of the product of the scales' max |v|
/// of the multiplies on the path (1 through every other mode). A tensor that two nodes read whose
/// results meet again, as ADD(MUL(y, s), y) does, so gets max|s| + 1 and not max|s|. The slope of
/// a multiply's scale operand is the magnitude of the convolution's result side.
///
/// A graph input's moments are those of its baked value when it bakes one, else the lookup's. A
/// moment that is not finite makes the inputs unavailable, whichever source it came from. The
/// moments of an input other than the convolution's operands and a multiply's scale are asked for
/// only when a rounding point needs the magnitude of a value that depends on it.
///
/// nullopt when the graph is outside the model: no or several convolutions, a convolution that
/// does not accumulate in float, a node that is neither the convolution nor pointwise, a pointwise
/// mode without a modeled slope, a backward activation in a graph whose convolution is not
/// backward data, a pointwise node computing in a type other than half, bfloat16,
/// float or double, an intermediate stored as another type than these (an unset type counts as
/// none of them), an operand of the convolution that is not a graph input, dims or parameters for
/// which K is not defined or is 0, a multiply that does not pair the convolution's result with one
/// graph input, an output that no node writes or that does not depend on the convolution, a
/// nonfinite magnitude, or inputs whose moments are not available. Also when both operands hold
/// one value (a FIXED fill recipe, or a value baked into the graph): every step of a sequential
/// sum then rounds the same way and the error grows as K, not as sqrt(K), which the model's form
/// does not describe; the per-op default grades that graph.
inline std::optional<ConvErrorScale>
    convErrorScale(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
                   const InputMomentsLookup& momentsFor,
                   int64_t outputUid)
{
    namespace data = hipdnn_flatbuffers_sdk::data_objects;
    namespace sdk = hipdnn_test_sdk::utilities::conv;
    using hipdnn_test_sdk::utilities::conv::RoundingPoint;

    std::optional<detail::ConvNode> conv;
    // The pointwise nodes, in the order the graph lists them.
    struct PointwiseNode
    {
        const data::PointwiseAttributes* attrs;
        data::DataType compute;
    };
    std::vector<PointwiseNode> pointwise;
    // Every tensor some node writes, as opposed to a graph input.
    std::set<int64_t> produced;
    // Every tensor some node reads.
    std::set<int64_t> consumed;
    bool backwardActivation = false;
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
            if(!detail::epsilonOf(node.compute_data_type()).has_value())
            {
                return std::nullopt;
            }
            const auto* attrs = node.attributes_as_PointwiseAttributes();
            if(!detail::hasModeledSlope(*attrs))
            {
                return std::nullopt;
            }
            backwardActivation
                = backwardActivation || detail::isBackwardActivation(attrs->operation());
            pointwise.push_back({attrs, node.compute_data_type()});
            produced.insert(attrs->out_0_tensor_uid());
            consumed.insert(attrs->in_0_tensor_uid());
            for(const auto operand : {attrs->in_1_tensor_uid(), attrs->in_2_tensor_uid()})
            {
                if(operand.has_value())
                {
                    consumed.insert(*operand);
                }
            }
        }
        else
        {
            return std::nullopt;
        }
    }
    if(!conv.has_value()
       || (backwardActivation && conv->type != data::NodeAttributes::ConvolutionBwdAttributes))
    {
        return std::nullopt;
    }

    const auto& attrs = wrapper.getTensorMap();
    const auto typeOf = [&](int64_t uid) -> std::optional<data::DataType> {
        const auto it = attrs.find(uid);
        if(it == attrs.end())
        {
            return std::nullopt;
        }
        return it->second->data_type();
    };
    // An intermediate, one node writes and another reads, is rounded to its own type, virtual or
    // not, and the model has to know that type. The graph's outputs, which no node reads, are
    // rounded once and are the model's.
    for(const auto uid : produced)
    {
        if(consumed.count(uid) == 0)
        {
            continue;
        }
        const auto type = typeOf(uid);
        if(!type.has_value() || !detail::epsilonOf(*type).has_value())
        {
            return std::nullopt;
        }
    }
    const auto accumulationLength = detail::convAccumulationLength(wrapper, *conv);
    if(!accumulationLength)
    {
        return std::nullopt;
    }
    // Worked in the fields of the forward scale; the variant of the direction is built from it at
    // the end.
    sdk::ConvFpropErrorScale scale;
    scale.accumulationLength = *accumulationLength;

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

    const auto xMoments = graphInputMoments(conv->first);
    const auto wMoments = graphInputMoments(conv->second);
    if(!xMoments || !wMoments)
    {
        return std::nullopt;
    }
    if(xMoments->constant && wMoments->constant)
    {
        return std::nullopt;
    }
    scale.inputMeanSquare = xMoments->meanSquare;
    scale.weightMeanSquare = wMoments->meanSquare;
    scale.inputMean = xMoments->mean;
    scale.weightMean = wMoments->mean;

    // The pointwise node that writes each tensor.
    std::map<int64_t, size_t> writer;
    for(size_t n = 0; n < pointwise.size(); ++n)
    {
        writer[pointwise[n].attrs->out_0_tensor_uid()] = n;
    }

    // What the walk learns of each pointwise node, by its index in `pointwise`.
    struct NodeInfo
    {
        // A multiply: the tensor that carries the convolution's result (possibly after other
        // pointwise nodes), the largest magnitude of the scale, and the operand slot of the scale.
        bool isScale = false;
        int64_t carried = 0;
        double scaleMax = 1.0;
        size_t scaleSlot = 0;
        // How much of an error in each operand reaches the node's result: the scale's largest
        // magnitude for the carried operand of a multiply, 1 otherwise. The scale operand of a
        // multiply is the exception: an error in the scale is multiplied by the size of
        // `carried`, which is worked out when a rounding point needs it.
        std::array<double, 3> gain = {1.0, 1.0, 1.0};
    };
    std::vector<NodeInfo> info(pointwise.size());
    // The pointwise nodes in dependency order.
    std::vector<size_t> walk;
    std::set<int64_t> written = {conv->result};
    std::vector<bool> walked(pointwise.size(), false);
    for(bool progressed = true; progressed;)
    {
        progressed = false;
        for(size_t n = 0; n < pointwise.size(); ++n)
        {
            if(walked[n])
            {
                continue;
            }
            const auto* pw = pointwise[n].attrs;
            const auto operands = detail::operandsOf(*pw);
            const bool ready = std::all_of(operands.begin(), operands.end(), [&](const auto& uid) {
                return !uid.has_value() || produced.count(*uid) == 0 || written.count(*uid) != 0;
            });
            if(!ready)
            {
                continue;
            }
            walked[n] = true;
            progressed = true;

            if(pw->operation() == data::PointwiseMode::MUL)
            {
                // One operand carries the convolution's result (possibly after other pointwise
                // nodes), the other is the scale; which of in_0 and in_1 is which does not matter.
                if(!pw->in_1_tensor_uid().has_value())
                {
                    return std::nullopt;
                }
                const int64_t first = pw->in_0_tensor_uid();
                const int64_t second = *pw->in_1_tensor_uid();
                const bool firstProduced = produced.count(first) != 0;
                if(firstProduced == (produced.count(second) != 0))
                {
                    return std::nullopt;
                }
                const int64_t scaleUid = firstProduced ? second : first;
                const auto moments = graphInputMoments(scaleUid);
                if(!moments)
                {
                    return std::nullopt;
                }
                auto& node = info[n];
                node.isScale = true;
                node.carried = firstProduced ? first : second;
                node.scaleMax = moments->maxAbs;
                node.scaleSlot = firstProduced ? 1 : 0;
                node.gain[firstProduced ? 0 : 1] = moments->maxAbs;
            }
            written.insert(pw->out_0_tensor_uid());
            walk.push_back(n);
        }
    }
    if(walk.size() != pointwise.size())
    {
        return std::nullopt;
    }

    // The factor by which each tensor carries the convolution's result, and which tensors depend
    // on it at all (a scale of 0 gives the factor 0 and still depends). No input is looked up.
    std::map<int64_t, double> factor = {{conv->result, 1.0}};
    std::set<int64_t> dependent = {conv->result};
    const auto factorOf = [&](int64_t uid) {
        const auto it = factor.find(uid);
        return it == factor.end() ? 0.0 : it->second;
    };
    for(const auto n : walk)
    {
        const auto& node = info[n];
        double result = 0.0;
        bool depends = false;
        const auto* pw = pointwise[n].attrs;
        if(node.isScale)
        {
            result = factorOf(node.carried) * node.scaleMax;
            depends = dependent.count(node.carried) != 0;
        }
        else if(detail::isBackwardActivation(pw->operation()))
        {
            // The gradient (in_0) carries the convolution's result and the forward input (in_1)
            // does not; the other way round the result is not a slope of the convolution's.
            const auto forwardInput = pw->in_1_tensor_uid();
            if(dependent.count(pw->in_0_tensor_uid()) == 0
               || (forwardInput.has_value() && dependent.count(*forwardInput) != 0))
            {
                return std::nullopt;
            }
            result = factorOf(pw->in_0_tensor_uid());
            depends = true;
        }
        else
        {
            for(const auto& operand : detail::operandsOf(*pw))
            {
                if(operand.has_value())
                {
                    result += factorOf(*operand);
                    depends = depends || dependent.count(*operand) != 0;
                }
            }
        }
        const int64_t out = pointwise[n].attrs->out_0_tensor_uid();
        factor[out] = result;
        if(depends)
        {
            dependent.insert(out);
        }
    }
    if(dependent.count(outputUid) == 0)
    {
        return std::nullopt;
    }
    scale.outputScale = factorOf(outputUid);
    if(!std::isfinite(scale.outputScale))
    {
        return std::nullopt;
    }

    // The root mean square of the convolution's result (see the test SDK's model): with means,
    // the partial sums do not cancel and the result grows as K * mean * mean.
    const auto products = static_cast<double>(scale.accumulationLength);
    const double meanProduct = scale.inputMean * scale.weightMean;
    const double sigma = std::sqrt(products * scale.inputMeanSquare * scale.weightMeanSquare
                                   + products * (products - 1.0) * meanProduct * meanProduct);
    // The typical magnitude of a tensor, from the statistics of the inputs it depends on; none
    // when one of them has none. A value is `factor` times the convolution's result plus a
    // remainder of mean square `independent`: a multiply multiplies the remainder by the square
    // of the scale's max |v|; a backward activation keeps its gradient's (it scales the gradient
    // by a derivative of at most about 1, and the forward input adds nothing); every other node
    // adds the remainders of its operands (independent inputs add in quadrature). Worked out when
    // a point asks for it, together with the remainders it depends on, so that no input is looked
    // up for nothing. The walk above is complete, so the graph has no cycle and this ends.
    std::map<int64_t, std::optional<double>> independent;
    const auto independentSquareOf = [&](int64_t target) -> std::optional<double> {
        std::vector<int64_t> pending = {target};
        while(!pending.empty())
        {
            const int64_t uid = pending.back();
            if(independent.count(uid) != 0)
            {
                pending.pop_back();
                continue;
            }
            if(uid == conv->result)
            {
                independent[uid] = 0.0;
                continue;
            }
            if(produced.count(uid) == 0)
            {
                const auto moments = graphInputMoments(uid);
                independent[uid]
                    = moments ? std::optional<double>(moments->meanSquare) : std::nullopt;
                continue;
            }
            const size_t n = writer.at(uid);
            const auto& node = info[n];
            std::vector<int64_t> reads;
            if(node.isScale)
            {
                reads.push_back(node.carried);
            }
            else if(detail::isBackwardActivation(pointwise[n].attrs->operation()))
            {
                reads.push_back(pointwise[n].attrs->in_0_tensor_uid());
            }
            else
            {
                for(const auto& operand : detail::operandsOf(*pointwise[n].attrs))
                {
                    if(operand.has_value())
                    {
                        reads.push_back(*operand);
                    }
                }
            }
            const size_t waiting = pending.size();
            for(const auto read : reads)
            {
                if(independent.count(read) == 0)
                {
                    pending.push_back(read);
                }
            }
            if(pending.size() != waiting)
            {
                continue;
            }
            std::optional<double> result = 0.0;
            for(const auto read : reads)
            {
                const auto& operand = independent.at(read);
                if(!operand.has_value())
                {
                    result.reset();
                    break;
                }
                *result += *operand;
            }
            if(result.has_value() && node.isScale)
            {
                *result *= node.scaleMax * node.scaleMax;
            }
            independent[uid] = result;
        }
        return independent.at(target);
    };
    const auto rootMeanSquare = [&](int64_t uid) -> std::optional<double> {
        const auto remainder = independentSquareOf(uid);
        if(!remainder.has_value())
        {
            return std::nullopt;
        }
        const double carried = factorOf(uid) == 0.0 ? 0.0 : factorOf(uid) * sigma;
        return std::sqrt(carried * carried + *remainder);
    };

    // How much of an error in operand k of a node reaches its result.
    const auto gainOf = [&](size_t n, size_t k) -> std::optional<double> {
        const auto& node = info[n];
        if(node.isScale && k == node.scaleSlot)
        {
            return rootMeanSquare(node.carried);
        }
        return node.gain[k];
    };

    // The slope from each tensor to the graded output: 1 at the output itself, and for any other
    // tensor the sum, over the nodes that read it (a node that reads it twice counts twice), of
    // the slope of the node's result times the node's gain. The slopes of paths that meet again
    // add. A tensor with no path to the output has none.
    std::map<int64_t, double> slope = {{outputUid, 1.0}};
    for(auto it = walk.rbegin(); it != walk.rend(); ++it)
    {
        const auto* pw = pointwise[*it].attrs;
        const auto result = slope.find(pw->out_0_tensor_uid());
        if(result == slope.end())
        {
            continue;
        }
        const double fromResult = result->second;
        const auto operands = detail::operandsOf(*pw);
        for(size_t k = 0; k < operands.size(); ++k)
        {
            if(operands[k].has_value() && produced.count(*operands[k]) != 0)
            {
                slope[*operands[k]] += fromResult * info[*it].gain[k];
            }
        }
    }
    const auto slopeOf = [&](int64_t uid) {
        const auto it = slope.find(uid);
        return it == slope.end() ? 0.0 : it->second;
    };

    const auto addPoint = [&](data::DataType type, int64_t uid, double s) -> bool {
        const auto eps = detail::epsilonOf(type);
        const auto magnitude = rootMeanSquare(uid);
        if(!eps.has_value() || !magnitude.has_value())
        {
            return false;
        }
        if(!std::isfinite(*magnitude) || !std::isfinite(s))
        {
            return false;
        }
        scale.roundingPoints.push_back(RoundingPoint{*eps, *magnitude, s});
        return true;
    };

    for(const auto n : walk)
    {
        const auto& node = pointwise[n];
        if(!detail::isNarrowerThanFloat(node.compute))
        {
            continue;
        }
        const auto* pw = node.attrs;
        const double out = slopeOf(pw->out_0_tensor_uid());
        if(out == 0.0)
        {
            // The node does not reach the graded output: its roundings are another output's.
            continue;
        }
        const auto operands = detail::operandsOf(*pw);
        for(size_t k = 0; k < operands.size(); ++k)
        {
            if(!operands[k].has_value())
            {
                continue;
            }
            const auto type = typeOf(*operands[k]);
            if(!type.has_value() || !detail::epsilonOf(*type).has_value())
            {
                return std::nullopt;
            }
            if(*type == node.compute)
            {
                continue;
            }
            // The node converts this operand to its compute type, and rounds it.
            double operandSlope = 0.0;
            if(produced.count(*operands[k]) != 0)
            {
                operandSlope = slopeOf(*operands[k]);
            }
            else
            {
                const auto gain = gainOf(n, k);
                if(!gain.has_value())
                {
                    return std::nullopt;
                }
                operandSlope = out * *gain;
            }
            if(!addPoint(node.compute, *operands[k], operandSlope))
            {
                return std::nullopt;
            }
        }
        // The result is rounded to the compute type; storing it as another type rounds it again,
        // which the storage rule below counts for an intermediate.
        const auto outType = typeOf(pw->out_0_tensor_uid());
        if(outType.has_value() && detail::epsilonOf(*outType).has_value()
           && *outType != node.compute && !addPoint(node.compute, pw->out_0_tensor_uid(), out))
        {
            return std::nullopt;
        }
    }
    for(const auto uid : produced)
    {
        if(consumed.count(uid) == 0 || uid == outputUid)
        {
            continue;
        }
        const double toOutput = slopeOf(uid);
        if(toOutput == 0.0)
        {
            continue;
        }
        const auto type = *typeOf(uid);
        if(detail::isNarrowerThanFloat(type) && !addPoint(type, uid, toOutput))
        {
            return std::nullopt;
        }
    }
    switch(conv->type)
    {
    case data::NodeAttributes::ConvolutionFwdAttributes:
        return scale;
    case data::NodeAttributes::ConvolutionBwdAttributes:
    {
        sdk::ConvDgradErrorScale dgrad;
        dgrad.accumulationLength = scale.accumulationLength;
        dgrad.dyMeanSquare = scale.inputMeanSquare;
        dgrad.weightMeanSquare = scale.weightMeanSquare;
        dgrad.outputScale = scale.outputScale;
        dgrad.roundingPoints = std::move(scale.roundingPoints);
        dgrad.dyMean = scale.inputMean;
        dgrad.weightMean = scale.weightMean;
        return dgrad;
    }
    default:
    {
        sdk::ConvWrwErrorScale wrw;
        wrw.accumulationLength = scale.accumulationLength;
        wrw.inputMeanSquare = scale.inputMeanSquare;
        wrw.dyMeanSquare = scale.weightMeanSquare;
        wrw.outputScale = scale.outputScale;
        wrw.roundingPoints = std::move(scale.roundingPoints);
        wrw.inputMean = scale.inputMean;
        wrw.dyMean = scale.weightMean;
        return wrw;
    }
    }
}

/// Whether convOutputTolerance() models an output stored as this type: half, bfloat16 and
/// float. A caller asks before it pays for the statistics of the graph's inputs.
inline bool hasDynamicToleranceFor(hipdnn_flatbuffers_sdk::data_objects::DataType outputType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    return outputType == DataType::HALF || outputType == DataType::BFLOAT16
           || outputType == DataType::FLOAT;
}

namespace detail
{

template <typename OutputType>
hipdnn_test_sdk::utilities::conv::AbsRelTolerance convOutputToleranceAs(
    const ConvErrorScale& scale,
    const hipdnn_test_sdk::utilities::conv::ConvFpropCoefficients& coefficients)
{
    namespace sdk = hipdnn_test_sdk::utilities::conv;
    return std::visit(
        [&](const auto& s) -> sdk::AbsRelTolerance {
            using Scale = std::decay_t<decltype(s)>;
            if constexpr(std::is_same_v<Scale, sdk::ConvFpropErrorScale>)
            {
                return sdk::calculateConvFpropOutputTolerance<OutputType>(s, coefficients);
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
/// (see hasDynamicToleranceFor()). `coefficients` are the ones the engine sets for forward
/// convolutions ([conv_fwd_tolerance] in its TOML); an unset one keeps the default of the output
/// type. Backward data and backward weights use their calibrated constants.
inline std::optional<hipdnn_test_sdk::utilities::conv::AbsRelTolerance>
    convOutputTolerance(const ConvErrorScale& scale,
                        hipdnn_flatbuffers_sdk::data_objects::DataType outputType,
                        const hipdnn_test_sdk::utilities::conv::ConvFpropCoefficients& coefficients
                        = {})
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(outputType)
    {
    case DataType::HALF:
        return detail::convOutputToleranceAs<hipdnn_data_sdk::types::half>(scale, coefficients);
    case DataType::BFLOAT16:
        return detail::convOutputToleranceAs<hipdnn_data_sdk::types::bfloat16>(scale, coefficients);
    case DataType::FLOAT:
        return detail::convOutputToleranceAs<float>(scale, coefficients);
    default:
        return std::nullopt;
    }
}

} // namespace hipdnn_integration_tests::tolerance
