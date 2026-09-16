// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "core/Utils.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file ConvBiasNative.cpp
 * @brief The Hackweek:ConvBias engine's native half: matching, scoring, dispatch, and the
 *        one function that registers them.
 *
 * This engine serves a **two-node fusion**, which is what separates it from every other
 * pack in this directory: a ConvolutionFwd whose y is virtual, immediately consumed by a
 * Pointwise(ADD). Because that producer-consumer edge is elementwise over the same
 * iteration space, the whole graph is one launch with no workspace -- the convolution sum
 * lives in a register for the length of one thread and the add is applied in that same
 * thread. The matcher therefore checks node *topology* and a UID edge, not just a node
 * type: it is not enough that the graph holds a conv and a pointwise, the pointwise must
 * consume the conv's own y and that y must be virtual.
 *
 * The kernel (kernels/ConvBiasFusedFwd.cpp) is layout-agnostic and shape-agnostic: it
 * takes all sixteen tensor strides and every extent, group count, padding, stride and
 * dilation as runtime arguments. Layout is not a field in the hipDNN schema -- it exists
 * only as the stride vector -- so an NCHW and an NHWC graph of the same dims differ here
 * and nowhere else. What the kernel is NOT agnostic about is rank (it unravels exactly
 * four coordinates) and element type (HKP_CONV_BIAS_TYPE changes the declared types of
 * four of its parameters), so the matcher refuses anything else rather than letting it in
 * and miscomputing.
 *
 * Two conventions below are taken from the hipDNN CPU reference rather than from the
 * operation's popular name, because both are places where libraries disagree, and getting
 * either wrong produces numbers rather than an error:
 *
 *   * `post_padding` never enters the input-coordinate mapping. It only ever widened the
 *     output extent, and the output extent arrives from the graph's own tensors. It is
 *     still *checked* here -- see the output-extent identity below -- so an inconsistent
 *     post_padding is refused rather than silently ignored.
 *   * Groups are carried by no schema field. They are implied by the channel
 *     relationship groups = x.dims[1] / w.dims[1], with w's first dimension indexed by
 *     the GLOBAL output channel.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "Hackweek.conv_bias.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "Hackweek.conv_bias.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "Hackweek.conv_bias.score";
constexpr std::string_view DISPATCH_SYMBOL = "Hackweek.conv_bias.dispatch";

// KMD fields this pack varies along, and the tokens matching binds for dispatch.
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view X_TOKEN = "conv_bias.x.uid";
constexpr std::string_view W_TOKEN = "conv_bias.w.uid";
constexpr std::string_view BIAS_TOKEN = "conv_bias.bias.uid";
constexpr std::string_view OUT_TOKEN = "conv_bias.out.uid";

/// The kernel unravels a flat index into exactly four coordinates and names each tensor's
/// four strides as separate parameters, so rank 4 is not a simplification here.
constexpr uint32_t SUPPORTED_RANK = 4;
/// Two spatial axes, so pre_padding, post_padding, stride and dilation are each length 2.
constexpr size_t SUPPORTED_SPATIAL_RANK = 2;

/// This engine's graph is exactly two nodes, in this topological order.
constexpr uint32_t SUPPORTED_NODE_COUNT = 2;
constexpr uint32_t CONV_NODE_INDEX = 0;
constexpr uint32_t POINTWISE_NODE_INDEX = 1;

/// The one storage type this pack ships a candidate for, required on all five tensors and
/// required equal across them.
///
/// The source compiles for HALF and BFLOAT16 too -- all three tags were built through
/// hipRTC for gfx1151 -- but neither was ever run against a reference. The step those
/// widths would actually exercise is rounding the float accumulator back through the
/// element type before the add, and that step is the identity for FLOAT, so it is exactly
/// what went unexercised. Widening this means widening the matcher, the shipped
/// candidates and the numerics proof together; it is not a constant to relax on its own.
constexpr data_objects::DataType SUPPORTED_DATA_TYPE = data_objects::DataType::FLOAT;
/// sizeof(float). Every access the kernel makes is a scalar load or store at an offset
/// computed from the graph's own strides -- no vectorised access, no shared memory -- so
/// this is the whole alignment requirement, and a graph's usual alignment=16 is a stronger
/// guarantee than the kernel consumes.
constexpr int64_t SUPPORTED_ELEMENT_BYTES = 4;

/// The accumulator is float unconditionally, so a node asking for any other compute
/// precision would be served at a precision it did not ask for and then compared against a
/// reference that honoured the request.
constexpr data_objects::DataType SUPPORTED_COMPUTE_DATA_TYPE = data_objects::DataType::FLOAT;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The four device tensors a matched conv+bias graph binds. The conv's y is deliberately
/// absent: it is virtual, so it has no buffer and never reaches the kernel.
struct ConvBiasBinding
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t bias = 0;
    int64_t out = 0;

    bool operator==(const ConvBiasBinding& other) const
    {
        return x == other.x && w == other.w && bias == other.bias && out == other.out;
    }
};

/// The 33 scalars the kernel takes after its four pointers, in its own declared order.
struct ConvBiasGeometry
{
    int32_t outN = 0;
    int32_t outC = 0;
    int32_t outP = 0;
    int32_t outQ = 0;
    int32_t convK = 0;
    int32_t xC = 0;
    int32_t wC = 0;
    int32_t xH = 0;
    int32_t xW = 0;
    int32_t filtR = 0;
    int32_t filtS = 0;
    int32_t padH = 0;
    int32_t padW = 0;
    int32_t strideH = 0;
    int32_t strideW = 0;
    int32_t dilationH = 0;
    int32_t dilationW = 0;
    std::array<int64_t, SUPPORTED_RANK> xStrides = {0, 0, 0, 0};
    std::array<int64_t, SUPPORTED_RANK> wStrides = {0, 0, 0, 0};
    /// Already zeroed on every axis the bias broadcasts along -- see zeroedBiasStrides().
    std::array<int64_t, SUPPORTED_RANK> biasStrides = {0, 0, 0, 0};
    std::array<int64_t, SUPPORTED_RANK> outStrides = {0, 0, 0, 0};
    /// outN * outC * outP * outQ, in int64_t. The iteration space, and the grid divisor.
    int64_t total = 0;
};

/// Everything one admitted graph resolves to: which buffers, and what to launch over.
struct ConvBiasPlan
{
    ConvBiasBinding binding;
    ConvBiasGeometry geometry;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// True when @p value is representable in the int32_t parameter the kernel declares for
/// it. Not pedantry: the extents reach the kernel as int32_t, and a dim above 2^31 would
/// be truncated into a smaller positive number, producing a short iteration space and a
/// silently partial output rather than any error.
bool fitsInt32(int64_t value)
{
    return value >= std::numeric_limits<int32_t>::min()
           && value <= std::numeric_limits<int32_t>::max();
}

/// One spatial attribute vector -- pre_padding, post_padding, stride or dilation -- read
/// as its two values.
///
/// An absent vector reads as @p fallback, which is the frontend attribute default (0 for
/// the paddings, 1 for stride and dilation) and not a guess. A vector that is *present*
/// must be exactly length 2 and every element at least @p minimum; anything else is a
/// refusal rather than a clamp, because a 3-D convolution's length-3 vector is a graph
/// this kernel cannot serve, not one to truncate.
std::optional<std::array<int64_t, SUPPORTED_SPATIAL_RANK>>
    spatialParameter(const flatbuffers::Vector<int64_t>* values, int64_t fallback, int64_t minimum)
{
    if(values == nullptr)
    {
        return std::array<int64_t, SUPPORTED_SPATIAL_RANK>{fallback, fallback};
    }
    if(values->size() != SUPPORTED_SPATIAL_RANK)
    {
        return std::nullopt;
    }

    std::array<int64_t, SUPPORTED_SPATIAL_RANK> result = {0, 0};
    for(flatbuffers::uoffset_t axis = 0; axis < SUPPORTED_SPATIAL_RANK; ++axis)
    {
        const auto value = values->Get(axis);
        if(value < minimum || !fitsInt32(value))
        {
            return std::nullopt;
        }
        result[axis] = value;
    }
    return result;
}

/// True when @p tensor is rank 4 with a usable stride per dim, every extent int32-sized.
///
/// Strides are otherwise unconstrained -- that is the point of this kernel. The one
/// refusal is a stride below 1 on an axis of extent > 1. For the output that is the
/// documented hazard: it is written once per logical coordinate, so two coordinates
/// sharing an address is a race. For the inputs a zero stride would merely re-read one
/// element, which the kernel would survive -- but no such graph was ever run, so it is
/// declined rather than claimed. An extent-1 axis may carry any stride, because its
/// coordinate is always 0 and the term drops out.
bool hasSupportedShape(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || strides->size() != dims->size()
       || dims->size() != SUPPORTED_RANK)
    {
        return false;
    }

    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        const auto extent = dims->Get(axis);
        if(extent < 1 || !fitsInt32(extent))
        {
            return false;
        }
        if(extent > 1 && strides->Get(axis) < 1)
        {
            return false;
        }
    }
    return true;
}

/// True when @p tensor is real device data this kernel can address: a supported shape, the
/// one supported element type, not virtual, not a pass-by-value scalar, carrying no inline
/// constant and no ragged offset, and aligned at least to its element.
bool isSupportedOperand(const data_objects::TensorAttributes& tensor)
{
    if(!hasSupportedShape(tensor))
    {
        return false;
    }
    if(tensor.data_type() != SUPPORTED_DATA_TYPE)
    {
        return false;
    }

    // Virtual means "no caller buffer"; every one of these four is addressed through a
    // device pointer, so a virtual one has nothing to address.
    if(tensor.virtual_())
    {
        return false;
    }

    // A rank-4 tensor is also the shape a pass-by-value scalar can take; that variant-pack
    // slot holds a host pointer, not a device one.
    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor))
    {
        return false;
    }

    // An inline constant is a value the kernel does not read. Admitting one would silently
    // ignore it and use whatever the caller's buffer happened to hold.
    if(tensor.value_type() != data_objects::TensorValue::NONE)
    {
        return false;
    }

    // Ragged offsets rebase each batch; this kernel's flat index-times-stride addressing
    // would be wrong for them, and both reference executors decline them too.
    if(tensor.ragged_offset_tensor_uid().has_value())
    {
        return false;
    }

    return tensor.alignment() >= SUPPORTED_ELEMENT_BYTES;
}

/// The conv's y: not a device operand, so it is checked as a *shape* only -- but it must
/// be virtual, which is the fact that makes this fusion an elision rather than a dropped
/// output. A non-virtual y is a graph whose conv result someone else can observe, and
/// eliding it would silently leave that buffer unwritten.
bool isSupportedVirtualIntermediate(const data_objects::TensorAttributes& tensor)
{
    return hasSupportedShape(tensor) && tensor.data_type() == SUPPORTED_DATA_TYPE
           && tensor.virtual_() && tensor.value_type() == data_objects::TensorValue::NONE
           && !tensor.ragged_offset_tensor_uid().has_value()
           && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor);
}

std::string dataTypeName(data_objects::DataType dataType)
{
    return data_objects::EnumNameDataType(dataType);
}

/// The two nodes this engine's matchers read, or nullopt if the graph is not a
/// ConvolutionFwd followed by a Pointwise.
struct ConvBiasNodes
{
    const data_objects::ConvolutionFwdAttributes* conv = nullptr;
    const data_objects::PointwiseAttributes* pointwise = nullptr;
};

std::optional<ConvBiasNodes> convBiasNodes(const MatchContext& context)
{
    if(context.graph.nodeCount() != SUPPORTED_NODE_COUNT)
    {
        return std::nullopt;
    }

    const auto& convNode = context.graph.getNodeWrapper(CONV_NODE_INDEX);
    const auto& pointwiseNode = context.graph.getNodeWrapper(POINTWISE_NODE_INDEX);
    if(convNode.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes
       || pointwiseNode.attributesType() != data_objects::NodeAttributes::PointwiseAttributes)
    {
        return std::nullopt;
    }

    return ConvBiasNodes{&convNode.attributesAs<data_objects::ConvolutionFwdAttributes>(),
                         &pointwiseNode.attributesAs<data_objects::PointwiseAttributes>()};
}

/// True when every Pointwise field this engine does not implement is absent.
///
/// None of these is consulted by an ADD, so each is individually harmless -- which is
/// exactly why they are checked. A graph that sets one is a graph whose author expected
/// something this kernel does not do, and admitting it would return a plain sum while
/// silently discarding the request. in_2 is the third operand of BINARY_SELECT, already
/// excluded by operation == ADD; requiring its absence costs nothing and closes the case
/// where both are set.
bool pointwiseExtrasAbsent(const data_objects::PointwiseAttributes& pointwise)
{
    return !pointwise.in_2_tensor_uid().has_value() && !pointwise.axis_tensor_uid().has_value()
           && !pointwise.relu_lower_clip().has_value() && !pointwise.relu_upper_clip().has_value()
           && !pointwise.relu_lower_clip_slope().has_value() && !pointwise.swish_beta().has_value()
           && !pointwise.elu_alpha().has_value() && !pointwise.softplus_beta().has_value();
}

/// The bias operand's strides with every broadcast axis zeroed.
///
/// The kernel indexes bias at the OUTPUT's coordinates and holds no bias extents and no
/// broadcast test of its own -- kernels/ConvBiasFusedFwd.cpp:76-77 states that as its
/// contract with the caller. So an axis where the bias has extent 1 against an output
/// extent greater than 1 must arrive as stride 0, or every thread on that axis reads past
/// the operand. Zeroing on extent 1 unconditionally is correct in both sub-cases: where
/// the output extent is also 1 the coordinate is always 0, so the product is 0 either way.
///
/// This is the computed, conditional kind of value the seam assigns to the handler rather
/// than to a descriptor: the descriptor's substituter does literal replacement and could
/// not derive it.
std::array<int64_t, SUPPORTED_RANK>
    zeroedBiasStrides(const data_objects::TensorAttributes& bias)
{
    std::array<int64_t, SUPPORTED_RANK> strides = {0, 0, 0, 0};
    for(flatbuffers::uoffset_t axis = 0; axis < SUPPORTED_RANK; ++axis)
    {
        strides[axis] = bias.dims()->Get(axis) == 1 ? int64_t{0} : bias.strides()->Get(axis);
    }
    return strides;
}

/**
 * @brief The whole applicability decision and the launch geometry it implies, in one
 *        function.
 *
 * Deliberately one function used by BOTH the graph matcher and prepare(): the matcher
 * admits a graph exactly when this returns a value, and the handler launches exactly the
 * geometry this computed. Splitting them is how a matcher and a handler come to disagree
 * about which shapes are legal, and that disagreement is not diagnosed anywhere -- the
 * handler would simply launch over a shape nobody checked.
 */
std::optional<ConvBiasPlan> describeConvBias(const MatchContext& context)
{
    const auto nodes = convBiasNodes(context);
    if(!nodes.has_value())
    {
        return std::nullopt;
    }
    const auto& conv = *nodes->conv;
    const auto& pointwise = *nodes->pointwise;

    // Graph-level gates. The accumulator is float unconditionally, so both nodes must ask
    // for float compute. intermediate_data_type is the type the virtual y would have been
    // stored as, and the kernel rounds its accumulator through exactly that type before
    // the add -- so it must be the element type, not merely something.
    const auto& graph = context.graph.getGraph();
    if(graph.is_override_shape_enabled())
    {
        // The extents come from the tensor dims recorded in the graph, so a shape
        // overridden after lowering would never reach the kernel.
        return std::nullopt;
    }
    // intermediate_data_type is the type the virtual y would have been stored as, and the
    // kernel rounds its accumulator through exactly that type before the add -- so it must
    // not disagree with the element type. It is checked as "not a contradiction" rather
    // than "equal", because the schema calls this field a DEFAULT: the frontend stamps it
    // onto each tensor left unset and the per-element types are what get compared
    // (graph.fbs, the comment above compute_data_type). The authoritative statement of
    // what the intermediate is stored as is the y tensor's own data_type, and that is
    // required to be the element type below, which subsumes this. Requiring the default
    // itself would refuse a graph whose five tensors are all explicitly FLOAT merely
    // because nobody stamped the graph-level field.
    if(graph.intermediate_data_type() != data_objects::DataType::UNSET
       && graph.intermediate_data_type() != SUPPORTED_DATA_TYPE)
    {
        return std::nullopt;
    }
    if(context.graph.getNode(CONV_NODE_INDEX).compute_data_type() != SUPPORTED_COMPUTE_DATA_TYPE
       || context.graph.getNode(POINTWISE_NODE_INDEX).compute_data_type()
              != SUPPORTED_COMPUTE_DATA_TYPE)
    {
        return std::nullopt;
    }

    // The two implemented enumerators. CONVOLUTION is refused rather than left as an
    // untested branch: the hipDNN CPU reference rejects it too, so a flipped-filter path
    // could not have been proved against any available oracle. UNSET is the FlatBuffers
    // default for an absent scalar and is caught by the same comparison.
    if(conv.conv_mode() != data_objects::ConvMode::CROSS_CORRELATION)
    {
        return std::nullopt;
    }
    if(pointwise.operation() != data_objects::PointwiseMode::ADD)
    {
        return std::nullopt;
    }
    if(!pointwiseExtrasAbsent(pointwise))
    {
        return std::nullopt;
    }

    // The UID edge that makes this a fusion: the pointwise must consume the conv's own y.
    //
    // Bound by IDENTITY, not by position. ADD is commutative and the frontend is free to
    // emit the operands either way round, so whichever of in_0/in_1 is the conv's y is the
    // fused operand and the other is the bias. Matching on position alone would refuse
    // half the legal graphs, or worse, bind the bias pointer as the convolution result.
    const auto yUid = conv.y_tensor_uid();
    if(!pointwise.in_1_tensor_uid().has_value())
    {
        // A unary pointwise: there is no second operand to add.
        return std::nullopt;
    }
    const auto in0Uid = pointwise.in_0_tensor_uid();
    const auto in1Uid = pointwise.in_1_tensor_uid().value();

    int64_t biasUid = 0;
    if(in0Uid == yUid && in1Uid != yUid)
    {
        biasUid = in1Uid;
    }
    else if(in1Uid == yUid && in0Uid != yUid)
    {
        biasUid = in0Uid;
    }
    else
    {
        // Neither operand is the conv's y (the two nodes are unrelated and this is not a
        // fusion at all), or both are (the conv result added to itself, which would need
        // the same buffer twice and was never run).
        return std::nullopt;
    }

    const auto xUid = conv.x_tensor_uid();
    const auto wUid = conv.w_tensor_uid();
    const auto outUid = pointwise.out_0_tensor_uid();

    // out is the only pointer written through, so it is the only one whose aliasing is
    // undefined behaviour under the kernel's four __restrict__ qualifiers: two const
    // inputs naming one tensor is a read/read overlap, which restrict permits. out naming
    // an input is not, and the frontend does not forbid it. Refusing it here is the
    // deliberate half of that trade -- an in-place graph is declined rather than
    // miscompiled, and every graph that is not in-place keeps the alias information.
    // Nothing proved the in-place numerics either way.
    if(outUid == xUid || outUid == wUid || outUid == biasUid)
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, xUid);
    const auto* w = findTensor(context, wUid);
    const auto* y = findTensor(context, yUid);
    const auto* bias = findTensor(context, biasUid);
    const auto* out = findTensor(context, outUid);
    if(x == nullptr || w == nullptr || y == nullptr || bias == nullptr || out == nullptr)
    {
        return std::nullopt;
    }

    if(!isSupportedOperand(*x) || !isSupportedOperand(*w) || !isSupportedOperand(*bias)
       || !isSupportedOperand(*out) || !isSupportedVirtualIntermediate(*y))
    {
        return std::nullopt;
    }

    const auto spatialPrePadding
        = spatialParameter(conv.pre_padding(), /*fallback=*/0, /*minimum=*/0);
    const auto spatialPostPadding
        = spatialParameter(conv.post_padding(), /*fallback=*/0, /*minimum=*/0);
    const auto spatialStride = spatialParameter(conv.stride(), /*fallback=*/1, /*minimum=*/1);
    const auto spatialDilation = spatialParameter(conv.dilation(), /*fallback=*/1, /*minimum=*/1);
    if(!spatialPrePadding.has_value() || !spatialPostPadding.has_value()
       || !spatialStride.has_value() || !spatialDilation.has_value())
    {
        return std::nullopt;
    }

    const auto* xDims = x->dims();
    const auto* wDims = w->dims();
    const auto* yDims = y->dims();
    const auto* biasDims = bias->dims();
    const auto* outDims = out->dims();

    const auto xC = xDims->Get(1);
    const auto convK = wDims->Get(0);
    const auto wC = wDims->Get(1);

    // Groups are carried by no schema field: this quotient IS the group count, and w's
    // first dimension is indexed by the global output channel with x's channel offset by
    // the group. Both divisibility facts are the kernel's own preconditions -- it computes
    // groups = xC / wC and kPerGroup = convK / groups with integer division and no check.
    if(wC < 1 || xC % wC != 0)
    {
        return std::nullopt;
    }
    const auto groups = xC / wC;
    if(groups < 1 || convK % groups != 0)
    {
        return std::nullopt;
    }

    // The conv's own output shape, checked against the graph's stated y.
    if(yDims->Get(0) != xDims->Get(0) || yDims->Get(1) != convK)
    {
        return std::nullopt;
    }

    // The output extent identity. This is where post_padding is actually consumed: it does
    // not enter the input-coordinate mapping, but the extent it implies must be the one
    // the graph states, so an inconsistent post_padding is refused rather than ignored.
    for(flatbuffers::uoffset_t spatial = 0; spatial < SUPPORTED_SPATIAL_RANK; ++spatial)
    {
        const auto axis = static_cast<flatbuffers::uoffset_t>(2 + spatial);
        const auto window
            = (wDims->Get(axis) - 1) * (*spatialDilation)[spatial] + 1;
        const auto numerator = xDims->Get(axis) + (*spatialPrePadding)[spatial]
                               + (*spatialPostPadding)[spatial] - window;
        if(numerator < 0)
        {
            return std::nullopt;
        }
        if(yDims->Get(axis) != numerator / (*spatialStride)[spatial] + 1)
        {
            return std::nullopt;
        }
        // The conv's spatial extents are the added tensor's, so the fusion is elementwise
        // over one iteration space.
        if(yDims->Get(axis) != outDims->Get(axis))
        {
            return std::nullopt;
        }
    }

    // Either the convolution's channels align with the added tensor's, or there is exactly
    // one filter broadcast across them -- the convK == 1 case the kernel implements with
    // kOut = 0. No other relationship is admitted, because no other one is computed.
    if(yDims->Get(1) != outDims->Get(1) && yDims->Get(1) != 1)
    {
        return std::nullopt;
    }
    if(yDims->Get(0) != outDims->Get(0))
    {
        return std::nullopt;
    }

    // The bias is per-axis broadcast compatible with the output, numpy-style: each of its
    // four dims equals the output's or is 1. Anything else would be read out of bounds,
    // since the kernel addresses it at the output's coordinates.
    for(flatbuffers::uoffset_t axis = 0; axis < SUPPORTED_RANK; ++axis)
    {
        if(biasDims->Get(axis) != outDims->Get(axis) && biasDims->Get(axis) != 1)
        {
            return std::nullopt;
        }
    }

    ConvBiasGeometry geometry;
    geometry.outN = static_cast<int32_t>(outDims->Get(0));
    geometry.outC = static_cast<int32_t>(outDims->Get(1));
    geometry.outP = static_cast<int32_t>(outDims->Get(2));
    geometry.outQ = static_cast<int32_t>(outDims->Get(3));
    geometry.convK = static_cast<int32_t>(convK);
    geometry.xC = static_cast<int32_t>(xC);
    geometry.wC = static_cast<int32_t>(wC);
    geometry.xH = static_cast<int32_t>(xDims->Get(2));
    geometry.xW = static_cast<int32_t>(xDims->Get(3));
    geometry.filtR = static_cast<int32_t>(wDims->Get(2));
    geometry.filtS = static_cast<int32_t>(wDims->Get(3));
    geometry.padH = static_cast<int32_t>((*spatialPrePadding)[0]);
    geometry.padW = static_cast<int32_t>((*spatialPrePadding)[1]);
    geometry.strideH = static_cast<int32_t>((*spatialStride)[0]);
    geometry.strideW = static_cast<int32_t>((*spatialStride)[1]);
    geometry.dilationH = static_cast<int32_t>((*spatialDilation)[0]);
    geometry.dilationW = static_cast<int32_t>((*spatialDilation)[1]);

    for(flatbuffers::uoffset_t axis = 0; axis < SUPPORTED_RANK; ++axis)
    {
        geometry.xStrides[axis] = x->strides()->Get(axis);
        geometry.wStrides[axis] = w->strides()->Get(axis);
        geometry.outStrides[axis] = out->strides()->Get(axis);
    }
    geometry.biasStrides = zeroedBiasStrides(*bias);

    // int64_t: the matcher admits shapes whose element count exceeds 2^31, and a 32-bit
    // product here would wrap both the grid size and the comparison the kernel's own
    // bounds guard makes against it.
    geometry.total = static_cast<int64_t>(geometry.outN) * geometry.outC
                     * static_cast<int64_t>(geometry.outP) * geometry.outQ;

    return ConvBiasPlan{ConvBiasBinding{xUid, wUid, biasUid, outUid}, geometry};
}

/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * @warning Returning std::nullopt empties this engine's WHOLE catalog. That is correct
 *          here because this engine has exactly one pack: there is no other pack whose
 *          shapes a refusal could wrongly suppress.
 */
std::optional<BoundTokens> convBiasGraphMatches(const MatchContext& context)
{
    const auto plan = describeConvBias(context);
    if(!plan.has_value())
    {
        return std::nullopt;
    }

    // Binds operand uids for the dispatch handler to read back rather than re-deriving
    // them. The conv's y is not among them: it is virtual, so it has no device buffer.
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = plan->binding.x;
    bound[std::string(W_TOKEN)] = plan->binding.w;
    bound[std::string(BIAS_TOKEN)] = plan->binding.bias;
    bound[std::string(OUT_TOKEN)] = plan->binding.out;
    return bound;
}

/**
 * @brief Kernel-scoped applicability: is this candidate's baked dtype the graph's?
 *
 * HKP_CONV_BIAS_TYPE changes the declared types of four of the kernel's parameters, so a
 * candidate compiled for the wrong width would read and write every operand at the wrong
 * stride and return numbers rather than failing.
 *
 * block_size is deliberately NOT compared here. Every block size is correct for every
 * admitted shape, because the kernel guards its own bounds -- so it is a ranking axis, not
 * an applicability one, and gating on it here would make the scorer's job unreachable.
 */
bool convBiasKernelMatches(const MatchContext& context,
                           const BoundTokens& bound,
                           const KernelDefinition& kernel)
{
    const auto xUid = tryGetBoundInt(bound, X_TOKEN);
    if(!xUid.has_value())
    {
        return false;
    }
    const auto* x = findTensor(context, *xUid);
    if(x == nullptr)
    {
        return false;
    }

    return kernel.getStringMetadata(std::string(DTYPE_FIELD)) == dataTypeName(x->data_type());
}

/**
 * @brief Ranks the block-size variants by how little of the launched grid is idle.
 *
 * The free axis is block size, and the kernel guards its own bounds, so every candidate is
 * correct for every admitted shape and the ranking is purely about the rounded-up final
 * block. occupancy = total / (gridDim.x * blockDim.x) in (0, 1]: at a 1-element output a
 * 64-thread block wastes 63 lanes and a 1024-thread block wastes 1023, and this says so.
 *
 * The `+ blockSize * 1e-9` term is a tie-break, not a preference: it only separates
 * candidates whose occupancy is equal, where the larger block launches fewer blocks. It is
 * scaled small enough that it can never outweigh an occupancy difference, since the
 * smallest non-zero such difference over admitted shapes is far above 1024 * 1e-9.
 */
double convBiasScore(const MatchContext& context,
                     const BoundTokens& bound,
                     const KernelDefinition& kernel)
{
    const auto blockSize = kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD));
    const auto outUid = tryGetBoundInt(bound, OUT_TOKEN);
    if(!outUid.has_value() || blockSize <= 0)
    {
        return 0.0;
    }
    const auto* out = findTensor(context, *outUid);
    if(out == nullptr || out->dims() == nullptr)
    {
        return 0.0;
    }

    int64_t total = 1;
    for(const auto dim : *out->dims())
    {
        total *= dim;
    }

    const int64_t blocks = (total + blockSize - 1) / blockSize;
    const double launched = static_cast<double>(blocks) * static_cast<double>(blockSize);
    return static_cast<double>(total) / launched + static_cast<double>(blockSize) * 1e-9;
}

/**
 * @brief Re-reads the operand bindings a match established.
 *
 * @throws HipdnnPluginException if a token is missing or does not hold a tensor uid.
 */
ConvBiasBinding convBiasBinding(const BoundTokens& bound)
{
    // Every token was written by this engine's graph match, which admitted this graph; a
    // missing one means the catalog was built by an engine other than ours.
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_bias dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };

    return {read(X_TOKEN), read(W_TOKEN), read(BIAS_TOKEN), read(OUT_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The compiled kernel plus the geometry its launch needs, owning nothing that points back
/// into the MatchContext it was built from.
class PreparedConvBias : public PreparedDispatch
{
public:
    PreparedConvBias(std::unique_ptr<compilation::ICompiledProgram> program,
                     std::unique_ptr<compilation::IRunnableKernel> kernel,
                     ConvBiasBinding binding,
                     ConvBiasGeometry geometry)
        : _program(std::move(program))
        , _kernel(std::move(kernel))
        , _binding(binding)
        , _geometry(geometry)
    {
    }

    const compilation::IRunnableKernel& kernel() const
    {
        return *_kernel;
    }

    const ConvBiasBinding& binding() const
    {
        return _binding;
    }

    const ConvBiasGeometry& geometry() const
    {
        return _geometry;
    }

private:
    // The runnable kernel is a view into its program's module, so the program must outlive
    // it; both are held here for the plan's lifetime.
    std::unique_ptr<compilation::ICompiledProgram> _program;
    std::unique_ptr<compilation::IRunnableKernel> _kernel;
    ConvBiasBinding _binding;
    ConvBiasGeometry _geometry;
};

/// The tag ConvBiasFusedFwd.cpp's two-level macro paste maps to a device type, from this
/// candidate's dtype metadata.
///
/// Only FLOAT is accepted, matching the one dtype the matcher admits and the one this pack
/// ships a candidate for. The source's paste also defines HALF and BFLOAT16; adding either
/// here alone would produce a candidate no graph could ever select, because the matcher
/// would refuse the graph first. Widening is a three-place change -- this function, the
/// matcher's SUPPORTED_DATA_TYPE, and the shipped candidate list -- plus the numerics to
/// justify it.
std::string dtypeTagFor(const KernelDefinition& kernel)
{
    const auto& dtype = kernel.getStringMetadata(std::string(DTYPE_FIELD));
    if(dtype == "FLOAT")
    {
        return dtype;
    }

    // Unreachable via matching, which admits only the dtype this pack declares.
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "kernel '" + toString(kernel.kernelId)
                                                       + "' declares unsupported dtype '" + dtype
                                                       + "'");
}

/**
 * @brief The native dispatch behind this pack's UDD: sizes and launches the fused
 *        conv+bias kernel. Everything graph- and kernel-derived resolves once at
 *        prepare(); launch() only resolves buffers and launches, so nothing mutates once
 *        prepared and concurrent execution is safe.
 */
class ConvBiasDispatchHandler : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    /// @param kpackLoader Likewise. Unused by the descriptors this pack ships, which are
    ///        all embedded_source, but buildIngestorKernelCode takes it for the kinds it
    ///        also serves -- which is what routing through it buys.
    ConvBiasDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                            const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// No scratch. The conv's y is virtual, so it is never materialised: the sum lives in a
    /// register for the length of one thread and the add is applied in that same thread.
    /// One launch, no shared memory, no temporary buffer.
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        // Re-derives the same decision the matcher made, so the geometry launched is
        // exactly the geometry that was admitted.
        const auto plan = describeConvBias(context);
        if(!plan.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_bias dispatch was given a graph its own matcher does not admit");
        }

        // And cross-checks it against what matching actually bound. These can only differ
        // if the catalog was built by a different engine, which would otherwise surface as
        // a wrong answer: the uids decide which device buffer each pointer resolves to.
        const auto binding = convBiasBinding(bound);
        if(!(binding == plan->binding))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_bias dispatch bound operands that disagree with its own match");
        }

        const auto blockSize
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));
        if(blockSize == 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "kernel '" + toString(kernel.kernelId) + "' declares a zero block_size");
        }

        const auto& xTensor = *findTensor(context, binding.x);
        compilation::KernelCompileOptions options(&xTensor, context.deviceProperties.gcnArchName);
        // The two macros ConvBiasFusedFwd.cpp guards with #error. Both handler-supplied
        // rather than descriptor-bound: they are read out of this candidate's metadata and
        // validated before becoming flags, and the descriptor substituter does literal
        // replacement only. HKP_CONV_BIAS_BLOCK_SIZE in particular is bound into
        // __launch_bounds__, so it must equal the blockDim.x set below or the launch bound
        // and the actual block disagree -- which is why both come from the one blockSize
        // read above rather than from two places.
        options.add("HKP_CONV_BIAS_TYPE", dtypeTagFor(kernel));
        options.add("HKP_CONV_BIAS_BLOCK_SIZE", blockSize);

        // buildIngestorKernelCode rather than _kernelCompiler.compile directly: it is the
        // one place source loading and path containment are handled for every
        // kernel_source.kind, so a later hiprtc_file descriptor under this pack loads
        // instead of throwing at plan-build time.
        auto code
            = buildIngestorKernelCode(_kernelCompiler, _kpackLoader, context, kernel, options);

        const auto gridSize = static_cast<unsigned int>(
            (plan->geometry.total + static_cast<int64_t>(blockSize) - 1)
            / static_cast<int64_t>(blockSize));

        // Rounded up on purpose: the kernel returns early for index >= total, so a
        // partially populated final block is correct and an exact-multiple grid would leave
        // the tail unwritten. The guard is the kernel's precisely because this grid is
        // computed from the shape.
        code.kernel->setBlockSize(blockSize, 1, 1);
        code.kernel->setGridSize(gridSize, 1, 1);

        return std::make_unique<PreparedConvBias>(
            std::move(code.program), std::move(code.kernel), binding, plan->geometry);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedConvBias = dynamic_cast<const PreparedConvBias&>(prepared);
        const auto& binding = preparedConvBias.binding();
        const auto& geometry = preparedConvBias.geometry();

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto bias
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.bias, deviceBuffers, numDeviceBuffers);
        const auto out
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.out, deviceBuffers, numDeviceBuffers);

        // 37 arguments, in the order kernels/ConvBiasFusedFwd.cpp declares them: four
        // pointers, then 17 int32_t extents and conv parameters, then 16 int64_t strides.
        //
        // Neither arity nor order is diagnosed anywhere downstream -- hipRTC compiles the
        // kernel, getKernel() resolves it, and hipModuleLaunchKernel reads one pointer per
        // parameter the KERNEL declared. A short list reads whatever is next in memory and
        // two same-typed arguments swapped is a wrong answer with no diagnostic. The types
        // below are int32_t/int64_t in the geometry struct for the same reason: the
        // variadic pushes exactly what it is given, so a plain `int` where the kernel
        // declares int64_t would misalign every argument after it.
        preparedConvBias.kernel().launch(handle.getStream(),
                                         x.ptr,
                                         w.ptr,
                                         bias.ptr,
                                         out.ptr,
                                         geometry.outN,
                                         geometry.outC,
                                         geometry.outP,
                                         geometry.outQ,
                                         geometry.convK,
                                         geometry.xC,
                                         geometry.wC,
                                         geometry.xH,
                                         geometry.xW,
                                         geometry.filtR,
                                         geometry.filtS,
                                         geometry.padH,
                                         geometry.padW,
                                         geometry.strideH,
                                         geometry.strideW,
                                         geometry.dilationH,
                                         geometry.dilationW,
                                         geometry.xStrides[0],
                                         geometry.xStrides[1],
                                         geometry.xStrides[2],
                                         geometry.xStrides[3],
                                         geometry.wStrides[0],
                                         geometry.wStrides[1],
                                         geometry.wStrides[2],
                                         geometry.wStrides[3],
                                         geometry.biasStrides[0],
                                         geometry.biasStrides[1],
                                         geometry.biasStrides[2],
                                         geometry.biasStrides[3],
                                         geometry.outStrides[0],
                                         geometry.outStrides[1],
                                         geometry.outStrides[2],
                                         geometry.outStrides[3]);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

static compilation::KpackModuleCache& convBiasKpackModuleCache()
{
    // Process-lifetime, as the pointwise pack's is: a loaded module outlives the plan that
    // loaded it, and the cache is what makes that one module rather than one per dispatch.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetConvBiasModuleCache()
{
    convBiasKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer
/// to it, but a provider's Container is created and destroyed per handle, so it (and the
/// compiler and loader it holds) must outlive every Container.
const ConvBiasDispatchHandler& convBiasDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convBiasKpackModuleCache());
    static const ConvBiasDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvBiasSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &convBiasGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &convBiasKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &convBiasScore);
    scope.add(std::string(DISPATCH_SYMBOL), &convBiasDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
