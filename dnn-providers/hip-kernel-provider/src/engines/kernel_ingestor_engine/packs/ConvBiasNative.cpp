// Copyright Â© Advanced Micro Devices, Inc., or its affiliates.
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
#include <vector>

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
 * takes all twenty tensor strides and every extent, group count, padding, stride and
 * dilation as runtime arguments. Layout is not a field in the hipDNN schema -- it exists
 * only as the stride vector -- so an NCHW and an NHWC graph of the same dims differ here
 * and nowhere else.
 *
 * RANK is served the same way, by canonicalisation rather than by branching. The kernel is
 * written once in a five-axis (N, C, D, H, W) form with three spatial axes, and this
 * matcher expresses every admitted rank in it: a graph's rank - 2 spatial axes are
 * RIGHT-aligned into the canonical three and the leading slots left over are made
 * degenerate -- extent 1, filter 1, pad 0, stride 1, dilation 1, tensor stride 0. That is
 * an identity, not an approximation: the only coordinate such a slot can take is 0, its
 * loop runs once, and every address it contributes is a multiple of zero. So ranks 3, 4
 * and 5 reach one kernel body through one set of checks, and there is no per-rank code
 * path in either file to get out of step with the other.
 *
 * What the kernel is still NOT agnostic about is element type -- HKP_CONV_BIAS_TYPE
 * changes the declared types of four of its parameters -- and rank beyond five, which
 * would need a fourth spatial loop it does not have. The matcher refuses both rather than
 * letting them in and miscomputing.
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

/// The kernel is written in one canonical five-axis form, (N, C, D, H, W), and every
/// admitted rank is expressed in it. Nothing below is a rank-specific code path: a rank-3
/// or rank-4 graph is the same kernel with its unused leading spatial slots made
/// degenerate, which canonicalSpatial() below constructs.
constexpr uint32_t CANONICAL_RANK = 5;
/// Three canonical spatial axes, so the kernel's padding, stride and dilation arguments
/// are each a triple regardless of what the graph carried.
constexpr size_t CANONICAL_SPATIAL_RANK = 3;

/// The graph ranks this engine admits, inclusive. Two batch-and-channel axes plus one to
/// three spatial ones. Rank 2 is a graph with no spatial axis at all -- a convolution
/// whose window has nowhere to slide -- and the reference does not produce one; rank 6 and
/// above needs a fourth spatial loop this kernel does not have.
constexpr uint32_t MIN_SUPPORTED_RANK = 3;
constexpr uint32_t MAX_SUPPORTED_RANK = CANONICAL_RANK;

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

/// One value per canonical spatial axis, in (D, H, W) order.
using CanonicalSpatial = std::array<int64_t, CANONICAL_SPATIAL_RANK>;
/// One value per canonical tensor axis, in (N, C, D, H, W) order.
using CanonicalAxes = std::array<int64_t, CANONICAL_RANK>;

/// The 43 scalars the kernel takes after its four pointers, in its own declared order.
///
/// Everything here is already canonical: a rank-3 or rank-4 graph has had its degenerate
/// leading spatial slots filled in, so nothing downstream of describeConvBias() needs to
/// know what rank the graph was.
struct ConvBiasGeometry
{
    int32_t outN = 0;
    int32_t outC = 0;
    /// (outD, outP, outQ) -- the output's canonical spatial extents.
    std::array<int32_t, CANONICAL_SPATIAL_RANK> outSpatial = {1, 1, 1};
    int32_t convK = 0;
    int32_t xC = 0;
    int32_t wC = 0;
    /// (xD, xH, xW).
    std::array<int32_t, CANONICAL_SPATIAL_RANK> xSpatial = {1, 1, 1};
    /// (filtT, filtR, filtS).
    std::array<int32_t, CANONICAL_SPATIAL_RANK> filter = {1, 1, 1};
    /// pre_padding, stride and dilation, each canonicalised. A degenerate slot carries the
    /// identity -- pad 0, stride 1, dilation 1 -- which is what makes its only coordinate,
    /// 0, map to input coordinate 0.
    std::array<int32_t, CANONICAL_SPATIAL_RANK> padding = {0, 0, 0};
    std::array<int32_t, CANONICAL_SPATIAL_RANK> stride = {1, 1, 1};
    std::array<int32_t, CANONICAL_SPATIAL_RANK> dilation = {1, 1, 1};
    /// Canonical strides. A degenerate spatial slot carries 0, so its always-zero
    /// coordinate contributes nothing to any address.
    CanonicalAxes xStrides = {0, 0, 0, 0, 0};
    CanonicalAxes wStrides = {0, 0, 0, 0, 0};
    /// Already zeroed on every axis the bias broadcasts along -- see zeroedBiasStrides().
    CanonicalAxes biasStrides = {0, 0, 0, 0, 0};
    CanonicalAxes outStrides = {0, 0, 0, 0, 0};
    /// outN * outC * outD * outP * outQ, in int64_t. The iteration space, and the grid
    /// divisor.
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

/// The canonical slot a graph's spatial axis occupies.
///
/// The graph's @p spatialRank real spatial axes are RIGHT-aligned into the canonical three:
/// a rank-4 graph's (H, W) become canonical (H, W) and leave D degenerate, and a rank-3
/// graph's single axis becomes canonical W. Right-aligning rather than left-aligning is
/// what keeps the kernel's fastest-varying iteration axis the same axis at every rank, so
/// there is one unravel order there rather than three.
size_t canonicalSpatialSlot(size_t spatialRank, size_t spatialAxis)
{
    return CANONICAL_SPATIAL_RANK - spatialRank + spatialAxis;
}

/// One spatial attribute vector -- pre_padding, post_padding, stride or dilation -- read
/// into canonical (D, H, W) slots.
///
/// An absent vector reads as @p fallback throughout, which is the frontend attribute
/// default (0 for the paddings, 1 for stride and dilation) and not a guess. A vector that
/// is *present* must be exactly @p spatialRank long -- the graph's own rank minus its two
/// batch/channel axes -- and every element at least @p minimum. A length that disagrees
/// with the tensors' rank is a refusal rather than a truncation or a pad: a length-2
/// stride on a rank-5 graph is a graph whose author meant something, and guessing which
/// two of the three axes they meant is not this matcher's job.
///
/// Slots the graph does not occupy receive @p identity, which is the value that makes the
/// kernel's degenerate loop a no-op: 0 for padding, 1 for stride and dilation. Note this
/// is the same number as @p fallback for all four callers, but it is a different fact --
/// one is "the frontend omitted this", the other is "this axis does not exist" -- so it is
/// spelled separately rather than reused.
std::optional<CanonicalSpatial> spatialParameter(const flatbuffers::Vector<int64_t>* values,
                                                 size_t spatialRank,
                                                 int64_t fallback,
                                                 int64_t minimum,
                                                 int64_t identity)
{
    CanonicalSpatial result = {identity, identity, identity};

    if(values == nullptr)
    {
        for(size_t axis = 0; axis < spatialRank; ++axis)
        {
            result[canonicalSpatialSlot(spatialRank, axis)] = fallback;
        }
        return result;
    }
    if(values->size() != spatialRank)
    {
        return std::nullopt;
    }

    for(size_t axis = 0; axis < spatialRank; ++axis)
    {
        const auto value = values->Get(static_cast<flatbuffers::uoffset_t>(axis));
        if(value < minimum || !fitsInt32(value))
        {
            return std::nullopt;
        }
        result[canonicalSpatialSlot(spatialRank, axis)] = value;
    }
    return result;
}

/// A tensor's dims in canonical (N, C, D, H, W) slots, degenerate spatial slots reading 1.
///
/// Extent 1 rather than 0 on an absent axis: it is an axis of exactly one coordinate, so
/// the kernel's loop over it runs once and its bounds test passes, which is what makes the
/// degenerate slot an identity rather than an empty iteration space.
CanonicalAxes canonicalDims(const data_objects::TensorAttributes& tensor, size_t rank)
{
    const size_t spatialRank = rank - 2;
    CanonicalAxes dims = {1, 1, 1, 1, 1};
    dims[0] = tensor.dims()->Get(0);
    dims[1] = tensor.dims()->Get(1);
    for(size_t axis = 0; axis < spatialRank; ++axis)
    {
        dims[2 + canonicalSpatialSlot(spatialRank, axis)]
            = tensor.dims()->Get(static_cast<flatbuffers::uoffset_t>(2 + axis));
    }
    return dims;
}

/// A tensor's strides in canonical (N, C, D, H, W) slots, degenerate spatial slots
/// reading 0 -- which is what makes their always-zero coordinate contribute nothing to the
/// address the kernel computes.
CanonicalAxes canonicalStrides(const data_objects::TensorAttributes& tensor, size_t rank)
{
    const size_t spatialRank = rank - 2;
    CanonicalAxes strides = {0, 0, 0, 0, 0};
    strides[0] = tensor.strides()->Get(0);
    strides[1] = tensor.strides()->Get(1);
    for(size_t axis = 0; axis < spatialRank; ++axis)
    {
        strides[2 + canonicalSpatialSlot(spatialRank, axis)]
            = tensor.strides()->Get(static_cast<flatbuffers::uoffset_t>(2 + axis));
    }
    return strides;
}

/// True when @p tensor is rank @p rank with a usable stride per dim, every extent
/// int32-sized.
///
/// Strides are otherwise unconstrained -- that is the point of this kernel. The one
/// refusal is a stride below 1 on an axis of extent > 1. For the output that is the
/// documented hazard: it is written once per logical coordinate, so two coordinates
/// sharing an address is a race. For the inputs a zero stride would merely re-read one
/// element, which the kernel would survive -- but no such graph was ever run, so it is
/// declined rather than claimed. An extent-1 axis may carry any stride, because its
/// coordinate is always 0 and the term drops out.
bool hasSupportedShape(const data_objects::TensorAttributes& tensor, size_t rank)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || strides->size() != dims->size()
       || dims->size() != rank)
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
bool isSupportedOperand(const data_objects::TensorAttributes& tensor, size_t rank)
{
    if(!hasSupportedShape(tensor, rank))
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

    // An admitted rank is also a shape a pass-by-value scalar can take; that variant-pack
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
bool isSupportedVirtualIntermediate(const data_objects::TensorAttributes& tensor, size_t rank)
{
    return hasSupportedShape(tensor, rank) && tensor.data_type() == SUPPORTED_DATA_TYPE
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

/// The bias operand's canonical strides with every broadcast axis zeroed.
///
/// The kernel indexes bias at the OUTPUT's coordinates and holds no bias extents and no
/// broadcast test of its own -- see the contract stated in kernels/ConvBiasFusedFwd.cpp's
/// entry-point doc comment. So an axis where the bias has extent 1 against an output
/// extent greater than 1 must arrive as stride 0, or every thread on that axis reads past
/// the operand. Zeroing on extent 1 unconditionally is correct in both sub-cases: where
/// the output extent is also 1 the coordinate is always 0, so the product is 0 either way.
///
/// It is also what makes a degenerate spatial slot free here rather than a special case:
/// canonicalDims() reports extent 1 for a slot the graph does not have, so this zeroes it
/// by the same rule that zeroes a genuine broadcast, and canonicalStrides() would have
/// given 0 anyway.
///
/// This is the computed, conditional kind of value the seam assigns to the handler rather
/// than to a descriptor: the descriptor's substituter does literal replacement and could
/// not derive it.
CanonicalAxes zeroedBiasStrides(const data_objects::TensorAttributes& bias, size_t rank)
{
    const auto dims = canonicalDims(bias, rank);
    const auto strides = canonicalStrides(bias, rank);

    CanonicalAxes zeroed = {0, 0, 0, 0, 0};
    for(size_t axis = 0; axis < CANONICAL_RANK; ++axis)
    {
        zeroed[axis] = dims[axis] == 1 ? int64_t{0} : strides[axis];
    }
    return zeroed;
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

    // The graph's rank, taken from the sole graph output and then REQUIRED of the other
    // four by the isSupportedOperand calls below, which each compare against it. Reading it
    // off one tensor and enforcing it on the rest is the point: a graph mixing a rank-4 x
    // with a rank-5 w has no canonical form at all, and admitting it would canonicalise
    // each tensor against a different axis alignment and address one of them wrongly.
    const auto* outDimsVector = out->dims();
    if(outDimsVector == nullptr || outDimsVector->size() < MIN_SUPPORTED_RANK
       || outDimsVector->size() > MAX_SUPPORTED_RANK)
    {
        return std::nullopt;
    }
    const size_t rank = outDimsVector->size();
    const size_t spatialRank = rank - 2;

    if(!isSupportedOperand(*x, rank) || !isSupportedOperand(*w, rank)
       || !isSupportedOperand(*bias, rank) || !isSupportedOperand(*out, rank)
       || !isSupportedVirtualIntermediate(*y, rank))
    {
        return std::nullopt;
    }

    // Canonicalised straight out of the accessor: each is length 3 in (D, H, W) order with
    // the graph's own axes right-aligned into it, and the slots this rank does not have
    // carrying the identity. A vector whose length disagrees with `spatialRank` is refused
    // here rather than padded -- that is the check that keeps a rank-5 graph carrying a
    // length-2 stride out, instead of silently convolving it with an invented third value.
    const auto spatialPrePadding = spatialParameter(
        conv.pre_padding(), spatialRank, /*fallback=*/0, /*minimum=*/0, /*identity=*/0);
    const auto spatialPostPadding = spatialParameter(
        conv.post_padding(), spatialRank, /*fallback=*/0, /*minimum=*/0, /*identity=*/0);
    const auto spatialStride = spatialParameter(
        conv.stride(), spatialRank, /*fallback=*/1, /*minimum=*/1, /*identity=*/1);
    const auto spatialDilation = spatialParameter(
        conv.dilation(), spatialRank, /*fallback=*/1, /*minimum=*/1, /*identity=*/1);
    if(!spatialPrePadding.has_value() || !spatialPostPadding.has_value()
       || !spatialStride.has_value() || !spatialDilation.has_value())
    {
        return std::nullopt;
    }

    // From here down everything is canonical five-axis, so none of the checks below is
    // written once per rank. A degenerate spatial slot reads extent 1 on every tensor, and
    // the output-extent identity it has to satisfy is (1 + 0 + 0 - ((1-1)*1+1))/1 + 1 == 1,
    // which holds -- so the loop over the three canonical spatial axes proves the real ones
    // and passes trivially over the rest.
    const auto xDims = canonicalDims(*x, rank);
    const auto wDims = canonicalDims(*w, rank);
    const auto yDims = canonicalDims(*y, rank);
    const auto biasDims = canonicalDims(*bias, rank);
    const auto outDims = canonicalDims(*out, rank);

    const auto xC = xDims[1];
    const auto convK = wDims[0];
    const auto wC = wDims[1];

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
    if(yDims[0] != xDims[0] || yDims[1] != convK)
    {
        return std::nullopt;
    }

    // The output extent identity. This is where post_padding is actually consumed: it does
    // not enter the input-coordinate mapping, but the extent it implies must be the one
    // the graph states, so an inconsistent post_padding is refused rather than ignored.
    for(size_t spatial = 0; spatial < CANONICAL_SPATIAL_RANK; ++spatial)
    {
        const size_t axis = 2 + spatial;
        const auto window = (wDims[axis] - 1) * (*spatialDilation)[spatial] + 1;
        const auto numerator
            = xDims[axis] + (*spatialPrePadding)[spatial] + (*spatialPostPadding)[spatial] - window;
        if(numerator < 0)
        {
            return std::nullopt;
        }
        if(yDims[axis] != numerator / (*spatialStride)[spatial] + 1)
        {
            return std::nullopt;
        }
        // The conv's spatial extents are the added tensor's, so the fusion is elementwise
        // over one iteration space.
        if(yDims[axis] != outDims[axis])
        {
            return std::nullopt;
        }
    }

    // Either the convolution's channels align with the added tensor's, or there is exactly
    // one filter broadcast across them -- the convK == 1 case the kernel implements with
    // kOut = 0. No other relationship is admitted, because no other one is computed.
    if(yDims[1] != outDims[1] && yDims[1] != 1)
    {
        return std::nullopt;
    }
    if(yDims[0] != outDims[0])
    {
        return std::nullopt;
    }

    // The bias is per-axis broadcast compatible with the output, numpy-style: each of its
    // dims equals the output's or is 1. Anything else would be read out of bounds, since
    // the kernel addresses it at the output's coordinates. A degenerate spatial slot is
    // extent 1 on both sides, so it satisfies this by either arm.
    for(size_t axis = 0; axis < CANONICAL_RANK; ++axis)
    {
        if(biasDims[axis] != outDims[axis] && biasDims[axis] != 1)
        {
            return std::nullopt;
        }
    }

    ConvBiasGeometry geometry;
    geometry.outN = static_cast<int32_t>(outDims[0]);
    geometry.outC = static_cast<int32_t>(outDims[1]);
    geometry.convK = static_cast<int32_t>(convK);
    geometry.xC = static_cast<int32_t>(xC);
    geometry.wC = static_cast<int32_t>(wC);
    for(size_t spatial = 0; spatial < CANONICAL_SPATIAL_RANK; ++spatial)
    {
        const size_t axis = 2 + spatial;
        geometry.outSpatial[spatial] = static_cast<int32_t>(outDims[axis]);
        geometry.xSpatial[spatial] = static_cast<int32_t>(xDims[axis]);
        geometry.filter[spatial] = static_cast<int32_t>(wDims[axis]);
        geometry.padding[spatial] = static_cast<int32_t>((*spatialPrePadding)[spatial]);
        geometry.stride[spatial] = static_cast<int32_t>((*spatialStride)[spatial]);
        geometry.dilation[spatial] = static_cast<int32_t>((*spatialDilation)[spatial]);
    }

    geometry.xStrides = canonicalStrides(*x, rank);
    geometry.wStrides = canonicalStrides(*w, rank);
    geometry.outStrides = canonicalStrides(*out, rank);
    geometry.biasStrides = zeroedBiasStrides(*bias, rank);

    // int64_t: the matcher admits shapes whose element count exceeds 2^31, and a 32-bit
    // product here would wrap both the grid size and the comparison the kernel's own
    // bounds guard makes against it.
    geometry.total = static_cast<int64_t>(geometry.outN) * geometry.outC
                     * static_cast<int64_t>(geometry.outSpatial[0]) * geometry.outSpatial[1]
                     * static_cast<int64_t>(geometry.outSpatial[2]);

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

/// The tensor KernelCompileOptions derives its macros from -- a rank-4, packed-NCHW
/// stand-in rather than the graph's own x.
///
/// KernelCompileOptions' constructor emits HIP_PLUGIN_USE_FP32/FP16/BFP16 and
/// HIP_PLUGIN_LAYOUT_NHWC from whatever tensor it is handed, and it derives the layout one
/// through core::utils::isChannelLastLayout(), which is a total function on neither axis it
/// reads: it is defined for rank 4 and rank 5 only, and at each of those for exactly two
/// stride orders. Every other shape THROWS. That throw is what made this engine's rank-3
/// admission unservable -- the matcher accepted the graph, and then every one of the three
/// candidate kernels failed to build a plan with "Tensor must be 4D or 5D for layout
/// detection. Got 3D tensor.", which reaches the caller as a plan-finalize INTERNAL_ERROR.
/// The same throw also closed rank 4 and rank 5 to every stride order but two, against a
/// matcher that documents and enforces no layout requirement at all.
///
/// The fix is to stop asking, because this kernel never wanted the answer.
/// ConvBiasFusedFwd.cpp reads none of those eight macros: its element type arrives as
/// HKP_CONV_BIAS_TYPE, resolved from the candidate's own dtype metadata by dtypeTagFor()
/// below, and its layout arrives as twenty runtime stride arguments taken from the graph.
/// So passing the graph's x here was never a use of the derived value, only a coupling to
/// the deriving function's domain. This stand-in carries the one dtype the matcher admits,
/// which is exactly the dtype the graph's tensors were required to have, and an
/// unambiguously NCHW stride order -- so the macros come out well-formed, consistent with
/// the graph, and unread.
///
/// Static rather than rebuilt per prepare(): it depends on nothing but SUPPORTED_DATA_TYPE.
/// The buffer must outlive every KernelCompileOptions built from it, and prepare() is
/// called on any thread, so it is a function-local static initialised once.
const data_objects::TensorAttributes* layoutProbeStandIn()
{
    static const flatbuffers::DetachedBuffer s_buffer = [] {
        flatbuffers::FlatBufferBuilder builder;
        // Extents above 1 and strictly descending strides so the stride order is a strict
        // ordering rather than a tie the probe would have to break: this is NCHW and
        // nothing else.
        const std::vector<int64_t> dims = {2, 2, 2, 2};
        const std::vector<int64_t> strides = {8, 4, 2, 1};
        builder.Finish(data_objects::CreateTensorAttributesDirect(
            builder, /*uid=*/0, /*name=*/nullptr, SUPPORTED_DATA_TYPE, &strides, &dims));
        return builder.Release();
    }();
    return flatbuffers::GetRoot<data_objects::TensorAttributes>(s_buffer.data());
}

/// The symbol a graph of two spatial axes launches: the entry point whose 37-argument
/// parameter list the authoring contract froze. Serves rank 3 and rank 4.
constexpr const char* THREE_SPATIAL_ENTRY_POINT = "ConvBiasFusedFwd3d";

/// True when the canonical depth slot carries the identity everywhere it can appear, so
/// dropping its ten arguments changes nothing.
///
/// Canonicalisation right-aligns a graph's real spatial axes, so a rank-3 or rank-4 graph
/// arrives with slot 0 degenerate and a rank-5 graph does not. This asks the geometry rather
/// than remembering the rank, deliberately: the ten depth arguments are exactly what the
/// 37-argument entry point cannot carry, so the question that decides which symbol to launch
/// is "is every one of them the identity", not "what was the rank". The two agree for every
/// admitted graph, and where they could differ -- a rank-5 graph whose depth axis is
/// genuinely trivial -- this predicate is the safe one: it drops only what provably
/// contributes nothing.
///
/// prepare() and launch() both call this on the same stored geometry, so the symbol resolved
/// and the argument list pushed cannot disagree.
bool hasDegenerateDepthSlot(const ConvBiasGeometry& geometry)
{
    return geometry.outSpatial[0] == 1 && geometry.xSpatial[0] == 1 && geometry.filter[0] == 1
           && geometry.padding[0] == 0 && geometry.stride[0] == 1 && geometry.dilation[0] == 1
           && geometry.xStrides[2] == 0 && geometry.wStrides[2] == 0
           && geometry.biasStrides[2] == 0 && geometry.outStrides[2] == 0;
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

        // Not the graph's x: see layoutProbeStandIn() for why this kernel must not be asked
        // to have a detectable layout, and why a rank-3 graph could not build a plan while
        // it was.
        compilation::KernelCompileOptions options(layoutProbeStandIn(),
                                                  context.deviceProperties.gcnArchName);
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

        // The descriptor's entry_point is ConvBiasFusedFwd, the two-spatial-axis symbol, and
        // that is what buildIngestorKernelCode() has already resolved. A rank-5 graph needs
        // the other symbol in the same compiled program: its depth axis is ten arguments
        // ConvBiasFusedFwd's frozen parameter list has nowhere to put, so it is a different
        // entry point rather than a wider one. Re-resolving from `code.program` is why this
        // costs no second compile and no second descriptor.
        if(!hasDegenerateDepthSlot(plan->geometry))
        {
            code.kernel = code.program->getKernel(THREE_SPATIAL_ENTRY_POINT);
        }

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

        // Neither arity nor order is diagnosed anywhere downstream -- hipRTC compiles the
        // kernel, getKernel() resolves it, and hipModuleLaunchKernel reads one pointer per
        // parameter the KERNEL declared. A short list reads whatever is next in memory and
        // two same-typed arguments swapped is a wrong answer with no diagnostic. The types
        // below are int32_t/int64_t in the geometry struct for the same reason: the
        // variadic pushes exactly what it is given, so a plain `int` where the kernel
        // declares int64_t would misalign every argument after it.
        //
        // Which of the two lists to push is decided by the same predicate prepare() used to
        // decide which symbol to resolve, called on the same stored geometry. That is what
        // keeps the symbol and the arguments in step; they are two halves of one ABI and
        // nothing downstream would report them disagreeing.
        if(hasDegenerateDepthSlot(geometry))
        {
            // 37 arguments, in the order kernels/ConvBiasFusedFwd.cpp declares them for
            // ConvBiasFusedFwd: four pointers, then 17 int32_t extents and conv parameters,
            // then 16 int64_t strides. This is the parameter list the authoring contract
            // froze, and it is transcribed here unchanged.
            //
            // The depth slot is dropped, not passed as 1: it is index 0 of every canonical
            // spatial array and index 2 of every canonical stride array, and the predicate
            // above has just established that each is the identity. The entry point supplies
            // those same identity values itself.
            preparedConvBias.kernel().launch(handle.getStream(),
                                             x.ptr,
                                             w.ptr,
                                             bias.ptr,
                                             out.ptr,
                                             geometry.outN,
                                             geometry.outC,
                                             geometry.outSpatial[1],
                                             geometry.outSpatial[2],
                                             geometry.convK,
                                             geometry.xC,
                                             geometry.wC,
                                             geometry.xSpatial[1],
                                             geometry.xSpatial[2],
                                             geometry.filter[1],
                                             geometry.filter[2],
                                             geometry.padding[1],
                                             geometry.padding[2],
                                             geometry.stride[1],
                                             geometry.stride[2],
                                             geometry.dilation[1],
                                             geometry.dilation[2],
                                             geometry.xStrides[0],
                                             geometry.xStrides[1],
                                             geometry.xStrides[3],
                                             geometry.xStrides[4],
                                             geometry.wStrides[0],
                                             geometry.wStrides[1],
                                             geometry.wStrides[3],
                                             geometry.wStrides[4],
                                             geometry.biasStrides[0],
                                             geometry.biasStrides[1],
                                             geometry.biasStrides[3],
                                             geometry.biasStrides[4],
                                             geometry.outStrides[0],
                                             geometry.outStrides[1],
                                             geometry.outStrides[3],
                                             geometry.outStrides[4]);
            return;
        }

        // 47 arguments, in the order kernels/ConvBiasFusedFwd.cpp declares them for
        // ConvBiasFusedFwd3d: four pointers, then 23 int32_t extents and conv parameters,
        // then 20 int64_t strides. Reached only by a graph whose depth axis is real.
        preparedConvBias.kernel().launch(handle.getStream(),
                                         x.ptr,
                                         w.ptr,
                                         bias.ptr,
                                         out.ptr,
                                         geometry.outN,
                                         geometry.outC,
                                         geometry.outSpatial[0],
                                         geometry.outSpatial[1],
                                         geometry.outSpatial[2],
                                         geometry.convK,
                                         geometry.xC,
                                         geometry.wC,
                                         geometry.xSpatial[0],
                                         geometry.xSpatial[1],
                                         geometry.xSpatial[2],
                                         geometry.filter[0],
                                         geometry.filter[1],
                                         geometry.filter[2],
                                         geometry.padding[0],
                                         geometry.padding[1],
                                         geometry.padding[2],
                                         geometry.stride[0],
                                         geometry.stride[1],
                                         geometry.stride[2],
                                         geometry.dilation[0],
                                         geometry.dilation[1],
                                         geometry.dilation[2],
                                         geometry.xStrides[0],
                                         geometry.xStrides[1],
                                         geometry.xStrides[2],
                                         geometry.xStrides[3],
                                         geometry.xStrides[4],
                                         geometry.wStrides[0],
                                         geometry.wStrides[1],
                                         geometry.wStrides[2],
                                         geometry.wStrides[3],
                                         geometry.wStrides[4],
                                         geometry.biasStrides[0],
                                         geometry.biasStrides[1],
                                         geometry.biasStrides[2],
                                         geometry.biasStrides[3],
                                         geometry.biasStrides[4],
                                         geometry.outStrides[0],
                                         geometry.outStrides[1],
                                         geometry.outStrides[2],
                                         geometry.outStrides[3],
                                         geometry.outStrides[4]);
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
