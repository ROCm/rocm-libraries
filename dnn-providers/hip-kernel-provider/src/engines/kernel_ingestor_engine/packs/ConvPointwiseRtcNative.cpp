// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstddef>
#include <cstdint>
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
 * @file ConvPointwiseRtcNative.cpp
 * @brief The hipkernel:ConvPointwiseRtc engine's native half: matching, scoring,
 *        dispatch, and the one function that registers them.
 *
 * This engine serves a TWO-node graph, which is what separates it from every other pack
 * in this directory: a rank-4 `ConvolutionFwdAttributes` whose output tensor is
 * `virtual`, immediately consumed by a unary `PointwiseAttributes`. Because the
 * convolution's output has no caller buffer at all, the kernel
 * (kernels/ConvFwdPointwiseFused.cpp) keeps the accumulator in a register, applies the
 * activation to it, and stores only the Pointwise output -- one launch, no workspace, and
 * one round trip through the intermediate's declared element type elided. That elision is
 * why the matcher below insists the virtual tensor is declared FLOAT: FLOAT is the type
 * the accumulator already has, so eliding the store changes nothing, and at any other
 * declared type it would.
 *
 * The kernel addresses every operand through that operand's own strides, so layout is not
 * a code path here: NCHW, NHWC and a row pitch wider than the width extent differ only in
 * the twelve stride arguments the handler passes. What it is NOT agnostic about is rank --
 * it indexes n/c/h/w explicitly -- so the matcher refuses anything but rank 4.
 *
 * One pack, one fusion, so a single graph matcher both admits the topology and validates
 * it: the ConvNative/BatchnormInference shape rather than the Pointwise one, which splits
 * an applicability matcher across three packs.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.conv_pointwise_rtc.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_pointwise_rtc.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.conv_pointwise_rtc.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_pointwise_rtc.dispatch";

// KMD fields this pack varies along, and the tokens matching binds for dispatch.
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view IO_DTYPE_FIELD = "io_dtype";
constexpr std::string_view ACTIVATION_FIELD = "activation";
constexpr std::string_view X_TOKEN = "conv_pointwise_rtc.x.uid";
constexpr std::string_view W_TOKEN = "conv_pointwise_rtc.w.uid";
constexpr std::string_view Y_TOKEN = "conv_pointwise_rtc.y.uid";

/// x is N,C,H,W; w is (G*K_g),C_g,R,S; both the virtual intermediate and the output are
/// N,K,P,Q. The kernel unravels exactly four coordinates per operand, so rank 4 is a
/// requirement and not a simplification.
constexpr uint32_t SUPPORTED_RANK = 4;
/// Two spatial axes, so stride/dilation/pre_padding/post_padding are each length 2.
constexpr uint32_t SUPPORTED_SPATIAL_RANK = 2;

/// The accumulator is `float` unconditionally (ConvFwdPointwiseFused.cpp:192), so a node
/// asking for any other compute precision would be served at a precision it did not ask
/// for and compared against a reference that honoured the request.
constexpr data_objects::DataType SUPPORTED_COMPUTE_DATA_TYPE = data_objects::DataType::FLOAT;

/// The one storage element type a shipped candidate is compiled for.
///
/// `HKP_IO_DTYPE` also accepts HKP_DTYPE_HALF and HKP_DTYPE_BFLOAT16 and both compile, but
/// neither was ever executed against a reference, and both carry an unresolved semantic
/// question this fusion cannot dodge: for an all-16-bit graph the virtual convolution
/// output would be declared 16-bit, the reference materializes it at that type and this
/// kernel does not. The elided rounding commutes with RELU/ABS/NEG but not with a clipped
/// ReLU at arbitrary clips. Until one of those is measured, no candidate ships for it and
/// this constant is the single place that says so.
constexpr data_objects::DataType SUPPORTED_IO_DATA_TYPE = data_objects::DataType::FLOAT;

/// The unary PointwiseModes `HKP_ACTIVATION` has a specialization for AND an oracle ran.
///
/// HKP_ACT_IDENTITY is deliberately absent even though the kernel compiles it: hipDNN's
/// CPU reference declines IDENTITY (it is in getUnaryModesBitset but not in
/// getImplementedUnaryModesBitset), so the authoring round recorded it as declined rather
/// than passed and it is an unmeasured bucket. The integration suite's GPU reference does
/// implement it, so this is a gap a later round can close with a measurement, not a
/// property of the kernel.
constexpr std::array<data_objects::PointwiseMode, 3> SUPPORTED_POINTWISE_MODES = {
    data_objects::PointwiseMode::RELU_FWD,
    data_objects::PointwiseMode::ABS,
    data_objects::PointwiseMode::NEG,
};

/// gfx90a's limit on the grid's y dimension, observed via rocminfo ("Grid Max Size per
/// Dimension: y 65535") during the authoring round. The pack declares `arch: [gfx90a]`, so
/// this is that architecture's number and not a portable constant. Exceeding it makes
/// hipModuleLaunchKernel return hipErrorInvalidConfiguration -- loud rather than silent --
/// but a graph outside the envelope is still refused at match time rather than failed at
/// dispatch, because a decline is attributable and a dispatch failure is a crash report.
constexpr int64_t MAX_GRID_Y = 65535;
/// The same device's limit on the grid's x dimension. Checked per CANDIDATE rather than
/// per graph, because the x extent is ceil(spatial / blockSize) and therefore depends on
/// which block size is baked -- a graph too wide for a 64-thread block may still be served
/// by the 1024-thread one, and putting this in graph_match would deny it both.
constexpr int64_t MAX_GRID_X = 2147483647;

/// The three clip defaults `hipdnn_test_sdk::pointwise::ReluForward`'s constructor
/// supplies, and therefore the values the reference computes with when the graph leaves
/// the attributes null. The kernel takes them as runtime arguments -- they are floats, and
/// KernelCompileOptions renders only bool/int/string, so they could not be `-D` values on
/// this path even if that were wanted.
constexpr float DEFAULT_RELU_LOWER_CLIP = 0.0f;
constexpr float DEFAULT_RELU_UPPER_CLIP = FLT_MAX;
constexpr float DEFAULT_RELU_LOWER_CLIP_SLOPE = 0.0f;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched fusion binds. Three, not four: the convolution's own output
/// is virtual and never reaches a device buffer, so there is nothing for launch() to
/// resolve it to.
struct ConvPointwiseBinding
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t y = 0;
};

/// The two nodes this engine reads, already checked for type and order.
struct FusedNodes
{
    const data_objects::ConvolutionFwdAttributes* conv = nullptr;
    const data_objects::PointwiseAttributes* pointwise = nullptr;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// The fusion's two nodes, or an empty pair when @p context is not exactly a
/// ConvolutionFwd followed by a Pointwise.
///
/// Node ORDER is checked here, and the uid edge between them is checked in the matcher
/// below. Neither on its own is enough: a Pointwise-then-Convolution graph has the same
/// node-type multiset, and two adjacent nodes of the right types that do not share a
/// tensor are two operations rather than one fusion.
FusedNodes fusedNodes(const MatchContext& context)
{
    if(context.graph.nodeCount() != 2)
    {
        return {};
    }

    const auto& convNode = context.graph.getNodeWrapper(0);
    const auto& pointwiseNode = context.graph.getNodeWrapper(1);
    if(convNode.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes
       || pointwiseNode.attributesType() != data_objects::NodeAttributes::PointwiseAttributes)
    {
        return {};
    }

    return {&convNode.attributesAs<data_objects::ConvolutionFwdAttributes>(),
            &pointwiseNode.attributesAs<data_objects::PointwiseAttributes>()};
}

/// True when @p values is present, exactly @p SUPPORTED_SPATIAL_RANK long, and every
/// element is at least @p minimum.
///
/// A null or wrongly sized vector is a refusal rather than a vacuous pass: an empty vector
/// satisfies "every element is at least N" for a convolution whose spatial rank this pack
/// cannot serve at all.
bool isSpatialVector(const flatbuffers::Vector<int64_t>* values, int64_t minimum)
{
    if(values == nullptr || values->size() != SUPPORTED_SPATIAL_RANK)
    {
        return false;
    }
    for(const auto value : *values)
    {
        if(value < minimum)
        {
            return false;
        }
    }
    return true;
}

/// True when @p tensor is rank-4 real device data stored as FLOAT with a stride per dim
/// the kernel can multiply a coordinate by.
///
/// Strides are otherwise unconstrained -- that is the point of this kernel, and it is why
/// there is no packed-layout check here as ConvNative has. A stride of 0 is the one
/// refusal, and only on an axis of extent > 1: there it is a broadcast, which on x or w
/// would read one element repeatedly and on y would have every thread on that axis race to
/// write one address. An extent-1 axis may carry any stride, because its coordinate is
/// always 0 and the term drops out of the address.
bool isSupportedOperand(const data_objects::TensorAttributes& tensor)
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
        if(dims->Get(axis) < 1)
        {
            return false;
        }
        if(dims->Get(axis) > 1 && strides->Get(axis) < 1)
        {
            return false;
        }
    }

    if(tensor.virtual_())
    {
        return false;
    }

    // Covers both `is_runtime_pass_by_value` and a baked `value`: a rank-4 tensor is also
    // the shape a pass-by-value scalar can take, and that variant-pack slot holds a host
    // pointer rather than a device one.
    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor))
    {
        return false;
    }

    // The kernel bases every operand at offset 0 and addresses it by strides, so an
    // offset table is not something it can read. Both reference executors decline ragged
    // tensors too (CHECK_NO_RAGGED_TENSORS), so such a graph would be unmeasurable as
    // well as wrong.
    if(tensor.ragged_offset_tensor_uid().has_value())
    {
        return false;
    }

    return tensor.data_type() == SUPPORTED_IO_DATA_TYPE;
}

/// True when @p tensor is the fused edge: rank-4 FLOAT, virtual, and carrying none of the
/// indirection the kernel cannot honour.
///
/// Deliberately NOT isSupportedOperand with the virtual test inverted. This tensor is
/// never allocated and never addressed -- the accumulator goes straight into the
/// activation -- so its STRIDES are read by nothing and constraining them would decline
/// graphs for a field that cannot affect the answer. Its dims and its declared dtype are
/// consumed: the dims are where P and Q come from, and FLOAT is the precondition that
/// makes eliding the intermediate store a no-op rather than a dropped rounding step.
bool isSupportedVirtualIntermediate(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    if(dims == nullptr || dims->size() != SUPPORTED_RANK)
    {
        return false;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        if(dims->Get(axis) < 1)
        {
            return false;
        }
    }

    if(!tensor.virtual_())
    {
        return false;
    }
    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor))
    {
        return false;
    }
    if(tensor.ragged_offset_tensor_uid().has_value())
    {
        return false;
    }

    return tensor.data_type() == SUPPORTED_IO_DATA_TYPE;
}

bool isSupportedPointwiseMode(data_objects::PointwiseMode mode)
{
    return std::find(SUPPORTED_POINTWISE_MODES.begin(), SUPPORTED_POINTWISE_MODES.end(), mode)
           != SUPPORTED_POINTWISE_MODES.end();
}

std::string dataTypeName(data_objects::DataType dataType)
{
    return data_objects::EnumNameDataType(dataType);
}

std::string pointwiseModeName(data_objects::PointwiseMode mode)
{
    return data_objects::EnumNamePointwiseMode(mode);
}

/// The output extent hipDNN's own validateConvolutionParams asserts, for one spatial axis,
/// or nullopt when the padded input is too small to hold the dilated filter at all.
///
/// The kernel takes P and Q from the graph's y dims rather than recomputing them, which is
/// exactly why this has to be checked: post_padding appears nowhere in the fprop index map
/// -- positions it creates fall outside the input and are skipped by the same bound test
/// that skips pre-padding -- and appears ONLY here. Without this equality a graph whose y
/// dims disagree with its padding is accepted and computed over the wrong extent. That is
/// what "post_padding is inert" means for this pack: inert under an enforced condition,
/// never ignored.
std::optional<int64_t> expectedOutputExtent(int64_t inputExtent,
                                            int64_t prePadding,
                                            int64_t postPadding,
                                            int64_t filterExtent,
                                            int64_t dilation,
                                            int64_t stride)
{
    const int64_t dilatedFilter = (dilation * (filterExtent - 1)) + 1;
    const int64_t padded = inputExtent + prePadding + postPadding;
    if(padded < dilatedFilter)
    {
        // Truncating division of a negative numerator would hand back 1 rather than a
        // diagnosis, and 1 is a plausible-looking extent.
        return std::nullopt;
    }
    return ((padded - dilatedFilter) / stride) + 1;
}

/// x's output-plane size (P*Q) and the y grid extent, for the candidate matcher and the
/// scorer. nullopt whenever this engine would not have matched the graph in the first
/// place, so neither has to re-derive the topology.
struct LaunchExtents
{
    int64_t spatialCount = 0;
    int64_t planeCount = 0;
};

std::optional<LaunchExtents> launchExtents(const MatchContext& context)
{
    const auto nodes = fusedNodes(context);
    if(nodes.conv == nullptr)
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, nodes.conv->x_tensor_uid());
    const auto* w = findTensor(context, nodes.conv->w_tensor_uid());
    const auto* y = findTensor(context, nodes.pointwise->out_0_tensor_uid());
    if(x == nullptr || w == nullptr || y == nullptr || x->dims() == nullptr
       || w->dims() == nullptr || y->dims() == nullptr || x->dims()->size() != SUPPORTED_RANK
       || w->dims()->size() != SUPPORTED_RANK || y->dims()->size() != SUPPORTED_RANK)
    {
        return std::nullopt;
    }

    const auto inChannelsPerGroup = w->dims()->Get(1);
    if(inChannelsPerGroup < 1 || x->dims()->Get(1) % inChannelsPerGroup != 0)
    {
        return std::nullopt;
    }

    LaunchExtents extents;
    // Not groupCount * batchCount * outChannelsPerGroup spelled out: their product IS
    // batchCount * (G * K_g), and G * K_g is w's leading extent by construction.
    extents.planeCount = x->dims()->Get(0) * w->dims()->Get(0);
    extents.spatialCount = y->dims()->Get(2) * y->dims()->Get(3);
    return extents;
}

/**
 * @brief Graph-scoped applicability: is this the one fused shape this engine's kernel can
 *        launch?
 *
 * Every field of both attribute tables is answered for here, because a field nobody
 * mentions is a field this engine claims by accident. The ones that are neither consumed
 * nor compared are inert under a condition this function enforces, and each says which.
 *
 * @warning Returning nullopt empties this engine's WHOLE catalog. That is the intended
 *          behaviour for a single-pack engine: there is no other pack whose applicability
 *          this could wrongly suppress.
 */
std::optional<BoundTokens> convPointwiseRtcGraphMatches(const MatchContext& context)
{
    const auto nodes = fusedNodes(context);
    if(nodes.conv == nullptr)
    {
        return std::nullopt;
    }
    const auto& conv = *nodes.conv;
    const auto& pointwise = *nodes.pointwise;

    // The extents come from the tensors' own dims, so a graph whose shapes may be
    // replaced at execute time is one whose dims this pack cannot trust.
    if(context.graph.getGraph().is_override_shape_enabled())
    {
        return std::nullopt;
    }

    // Per NODE, not the graph-level default: graph.fbs states the graph-level
    // compute/intermediate/io types are defaults the frontend stamps onto each node and
    // tensor left unset, and it is the stamped per-node value the reference's signature
    // key compares. Checking the default instead would pass a graph that overrode it.
    if(context.graph.getNode(0).compute_data_type() != SUPPORTED_COMPUTE_DATA_TYPE
       || context.graph.getNode(1).compute_data_type() != SUPPORTED_COMPUTE_DATA_TYPE)
    {
        return std::nullopt;
    }

    // An explicit equality, never a default. CONVOLUTION is a flipped filter -- a
    // different operation this kernel computes wrongly with no diagnostic -- and UNSET is
    // the zero enumerator FlatBuffers hands back for an absent field, which must not fall
    // through to the one mode that happens to be implemented. Neither can lean on the
    // oracle: ConvolutionFwdPlanBuilder::isApplicable never tests conv_mode, so the CPU
    // reference would execute a CONVOLUTION graph as cross-correlation and agree with a
    // wrong kernel.
    if(conv.conv_mode() != data_objects::ConvMode::CROSS_CORRELATION)
    {
        return std::nullopt;
    }

    const auto* stride = conv.stride();
    const auto* dilation = conv.dilation();
    const auto* prePadding = conv.pre_padding();
    const auto* postPadding = conv.post_padding();
    if(!isSpatialVector(stride, 1) || !isSpatialVector(dilation, 1)
       || !isSpatialVector(prePadding, 0) || !isSpatialVector(postPadding, 0))
    {
        return std::nullopt;
    }

    // A membership test against the three modes with a compiled specialization AND a
    // measurement, so UNSET and every unimplemented enumerator are refused by the same
    // check rather than by an else-branch nobody wrote.
    if(!isSupportedPointwiseMode(pointwise.operation()))
    {
        return std::nullopt;
    }

    // The uid edge that makes two nodes one fusion. Checked rather than assumed from the
    // node order, which is checked separately in fusedNodes().
    if(pointwise.in_0_tensor_uid() != conv.y_tensor_uid())
    {
        return std::nullopt;
    }

    // A second or third operand is a binary or ternary pointwise: a different fusion with
    // a different argument list and, when an operand broadcasts, a different index map.
    // axis_tensor_uid is not a uid despite the name -- it is a plain axis index used only
    // by GEN_INDEX, which is not an admitted mode -- and is refused on the same grounds.
    if(pointwise.in_1_tensor_uid().has_value() || pointwise.in_2_tensor_uid().has_value()
       || pointwise.axis_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    // Parameters of modes this pack does not admit. A non-null value here is not merely
    // unused: it also sends the CPU reference down its parameterized branch, which throws
    // for every mode except RELU_FWD and SWISH_FWD.
    if(pointwise.swish_beta().has_value() || pointwise.elu_alpha().has_value()
       || pointwise.softplus_beta().has_value())
    {
        return std::nullopt;
    }

    // The three clips belong to RELU_FWD alone. The kernel would ignore them for ABS and
    // NEG, which sounds harmless and is not: the reference throws on that combination, so
    // an accepted graph would have no oracle at all.
    const bool hasClip = pointwise.relu_lower_clip().has_value()
                         || pointwise.relu_upper_clip().has_value()
                         || pointwise.relu_lower_clip_slope().has_value();
    if(hasClip && pointwise.operation() != data_objects::PointwiseMode::RELU_FWD)
    {
        return std::nullopt;
    }

    const auto xUid = conv.x_tensor_uid();
    const auto wUid = conv.w_tensor_uid();
    const auto intermediateUid = conv.y_tensor_uid();
    const auto yUid = pointwise.out_0_tensor_uid();

    // y is the only pointer written through and the kernel keeps `__restrict__` on it, so
    // y naming an input is undefined behaviour rather than a slow path. Two const inputs
    // naming one tensor is a read/read overlap, which restrict permits and this admits.
    // The intermediate is compared too: it is virtual, and a virtual tensor sharing a uid
    // with a real one is a graph neither side can mean.
    if(yUid == xUid || yUid == wUid || yUid == intermediateUid || intermediateUid == xUid
       || intermediateUid == wUid || xUid == wUid)
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, xUid);
    const auto* w = findTensor(context, wUid);
    const auto* intermediate = findTensor(context, intermediateUid);
    const auto* y = findTensor(context, yUid);
    if(x == nullptr || w == nullptr || intermediate == nullptr || y == nullptr)
    {
        return std::nullopt;
    }

    if(!isSupportedOperand(*x) || !isSupportedOperand(*w) || !isSupportedOperand(*y)
       || !isSupportedVirtualIntermediate(*intermediate))
    {
        return std::nullopt;
    }

    const auto* xDims = x->dims();
    const auto* wDims = w->dims();
    const auto* yDims = y->dims();
    const auto* intermediateDims = intermediate->dims();

    // The activation is elementwise in place of a materialized intermediate, so the
    // Pointwise output must be exactly the shape the convolution would have written.
    for(flatbuffers::uoffset_t axis = 0; axis < SUPPORTED_RANK; ++axis)
    {
        if(yDims->Get(axis) != intermediateDims->Get(axis))
        {
            return std::nullopt;
        }
    }

    // Groups are encoded purely as the ratio of the two channel extents; there is no
    // group field in the schema. Both quotients must be exact, or the kernel's
    // (group, batch, out-channel) unravel addresses channels that do not exist.
    const auto batchCount = xDims->Get(0);
    const auto inChannels = xDims->Get(1);
    const auto inChannelsPerGroup = wDims->Get(1);
    const auto outChannels = wDims->Get(0);
    if(inChannelsPerGroup < 1 || inChannels % inChannelsPerGroup != 0)
    {
        return std::nullopt;
    }
    const auto groupCount = inChannels / inChannelsPerGroup;
    if(groupCount < 1 || outChannels % groupCount != 0)
    {
        return std::nullopt;
    }

    if(yDims->Get(0) != batchCount || yDims->Get(1) != outChannels)
    {
        return std::nullopt;
    }

    for(flatbuffers::uoffset_t axis = 0; axis < SUPPORTED_SPATIAL_RANK; ++axis)
    {
        const auto expected = expectedOutputExtent(xDims->Get(axis + 2),
                                                   prePadding->Get(axis),
                                                   postPadding->Get(axis),
                                                   wDims->Get(axis + 2),
                                                   dilation->Get(axis),
                                                   stride->Get(axis));
        if(!expected.has_value() || *expected != yDims->Get(axis + 2))
        {
            return std::nullopt;
        }
    }

    // The grid's y extent, exactly: batchCount * (groupCount * outChannelsPerGroup), and
    // the second factor is w's leading extent. Candidate-independent -- no block size
    // changes it -- so it belongs here rather than in the kernel matcher.
    if(batchCount * outChannels > MAX_GRID_Y)
    {
        return std::nullopt;
    }

    // Binds the operand uids for the dispatch handler to read back rather than
    // re-deriving them. The intermediate is not bound: it is virtual, so there is no
    // device buffer for launch() to resolve it to.
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = xUid;
    bound[std::string(W_TOKEN)] = wUid;
    bound[std::string(Y_TOKEN)] = yUid;
    return bound;
}

/**
 * @brief Kernel-scoped applicability: are this candidate's two baked axes the graph's, and
 *        does its block size keep the launch inside the device's grid?
 *
 * Two metadata comparisons rather than one, because `HKP_IO_DTYPE` and `HKP_ACTIVATION`
 * are independent specializations of one source and each is compared against a different
 * part of the graph. `graph_match` can only say "one of the three admitted activations";
 * deciding WHICH is per-candidate by construction, and without this comparison a relu
 * graph could reach the ABS binary and return wrong numbers rather than failing.
 *
 * The grid bound is here for the reason RUNBOOK §Matching gives for tile-dependent
 * divisibility: it is a property of (graph, candidate) rather than of the graph, so
 * checking it here lets a larger-block candidate serve a plane a smaller-block one cannot.
 */
bool convPointwiseRtcKernelMatches(const MatchContext& context,
                                   const BoundTokens& /*bound*/,
                                   const KernelDefinition& kernel)
{
    const auto nodes = fusedNodes(context);
    if(nodes.conv == nullptr)
    {
        return false;
    }

    const auto* x = findTensor(context, nodes.conv->x_tensor_uid());
    if(x == nullptr)
    {
        return false;
    }

    if(kernel.getStringMetadata(std::string(IO_DTYPE_FIELD)) != dataTypeName(x->data_type()))
    {
        return false;
    }
    if(kernel.getStringMetadata(std::string(ACTIVATION_FIELD))
       != pointwiseModeName(nodes.pointwise->operation()))
    {
        return false;
    }

    const auto extents = launchExtents(context);
    const auto blockSize = kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD));
    if(!extents.has_value() || blockSize < 1)
    {
        return false;
    }
    return ((extents->spatialCount + blockSize - 1) / blockSize) <= MAX_GRID_X;
}

/**
 * @brief Ranks the block-size variants by how much of the launched grid is idle.
 *
 * The free axis is block size: the kernel declares no block-size macro and reads
 * `blockDim.x`, and it guards its own tail, so every candidate is correct for every
 * admitted shape and the ranking is purely about the rounded-up final block in x. Nothing
 * in this round was timed, so this deliberately ranks a COMPUTED property of the launch
 * rather than standing in for a performance model nobody measured.
 *
 * `wasted` is an integer count of threads that return immediately, and it is scaled by
 * 2048 so that one wasted thread outweighs the entire tie-break term -- block size is at
 * most 1024. The tie-break therefore decides exactly the candidates whose waste is equal,
 * where the larger block launches fewer blocks, and can never reorder two candidates whose
 * waste differs. BatchnormInferenceNative.cpp:433 reaches for the same ordering through a
 * floating occupancy ratio plus `blockSize * 1e-9` and has to argue that no admitted shape
 * produces a smaller non-zero difference; both quantities here are integers, so the bound
 * is arithmetic rather than an assertion about the corpus.
 */
double convPointwiseRtcScore(const MatchContext& context,
                             const BoundTokens& /*bound*/,
                             const KernelDefinition& kernel)
{
    const auto extents = launchExtents(context);
    const auto blockSize = kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD));
    if(!extents.has_value() || blockSize < 1)
    {
        return 0.0;
    }

    const int64_t blocks = (extents->spatialCount + blockSize - 1) / blockSize;
    const int64_t wasted = (blocks * blockSize) - extents->spatialCount;
    return -(static_cast<double>(wasted) * 2048.0) + static_cast<double>(blockSize);
}

/**
 * @brief Re-reads the operand bindings a match established.
 *
 * @throws HipdnnPluginException if the graph is not one this matcher accepts.
 */
ConvPointwiseBinding convPointwiseRtcBinding(const BoundTokens& bound)
{
    // Every token was written by this engine's graph match, which admitted the graph; a
    // missing one means the catalog was built by an engine other than ours.
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_pointwise_rtc dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };

    return {read(X_TOKEN), read(W_TOKEN), read(Y_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The 31 scalars the kernel takes after its three pointers, in its own declaration order.
///
/// Every one is `int64_t` or `float` explicitly rather than `auto`-deduced, because
/// IRunnableKernel::launch takes the ADDRESS of each argument and hipModuleLaunchKernel
/// reads one kernarg per parameter the kernel declared: an `int` here would hand the
/// device four bytes where it expects eight, and nothing in the toolchain would say so.
struct ConvPointwiseGeometry
{
    int64_t batchCount = 0;
    int64_t groupCount = 0;
    int64_t inChannelsPerGroup = 0;
    int64_t outChannelsPerGroup = 0;
    int64_t inHeight = 0;
    int64_t inWidth = 0;
    int64_t filterHeight = 0;
    int64_t filterWidth = 0;
    int64_t outHeight = 0;
    int64_t outWidth = 0;
    int64_t strideHeight = 0;
    int64_t strideWidth = 0;
    int64_t dilationHeight = 0;
    int64_t dilationWidth = 0;
    int64_t prePadHeight = 0;
    int64_t prePadWidth = 0;
    int64_t xStrides[SUPPORTED_RANK] = {0, 0, 0, 0};
    int64_t wStrides[SUPPORTED_RANK] = {0, 0, 0, 0};
    int64_t yStrides[SUPPORTED_RANK] = {0, 0, 0, 0};
    float reluLowerClip = DEFAULT_RELU_LOWER_CLIP;
    float reluUpperClip = DEFAULT_RELU_UPPER_CLIP;
    float reluLowerClipSlope = DEFAULT_RELU_LOWER_CLIP_SLOPE;
};

/// The compiled kernel plus the geometry its launch needs, owning nothing that points back
/// into the MatchContext it was built from.
class PreparedConvPointwiseRtc : public PreparedDispatch
{
public:
    PreparedConvPointwiseRtc(std::unique_ptr<compilation::ICompiledProgram> program,
                             std::unique_ptr<compilation::IRunnableKernel> kernel,
                             ConvPointwiseBinding binding,
                             ConvPointwiseGeometry geometry)
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

    const ConvPointwiseBinding& binding() const
    {
        return _binding;
    }

    const ConvPointwiseGeometry& geometry() const
    {
        return _geometry;
    }

private:
    // The runnable kernel is a view into its program's module, so the program must outlive
    // it; both are held here for the plan's lifetime.
    std::unique_ptr<compilation::ICompiledProgram> _program;
    std::unique_ptr<compilation::IRunnableKernel> _kernel;
    ConvPointwiseBinding _binding;
    ConvPointwiseGeometry _geometry;
};

/// The `HKP_IO_DTYPE` tag for a candidate's io dtype metadata.
///
/// A validated lookup rather than `"HKP_DTYPE_" + dtype`. A blind paste of an unexpected
/// metadata string would produce an undeclared type name, which the kernel's own header
/// turns into a compile error -- that is by design and it is still the wrong diagnosis to
/// ship: a named plugin exception says which kernel and which field, and a hipRTC log says
/// that `HKP_TYPE_HKP_DTYPE_FLOAT32` does not name a type.
std::string ioDtypeTagFor(const KernelDefinition& kernel)
{
    const auto& dtype = kernel.getStringMetadata(std::string(IO_DTYPE_FIELD));
    if(dtype == "FLOAT")
    {
        return "HKP_DTYPE_FLOAT";
    }

    // Unreachable via matching, which admits only the dtype this pack declares.
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "kernel '" + toString(kernel.kernelId)
                                                       + "' declares unsupported io_dtype '"
                                                       + dtype + "'");
}

/// The `HKP_ACTIVATION` tag for a candidate's activation metadata, validated for the same
/// reason ioDtypeTagFor is.
std::string activationTagFor(const KernelDefinition& kernel)
{
    const auto& activation = kernel.getStringMetadata(std::string(ACTIVATION_FIELD));
    if(activation == "RELU_FWD")
    {
        return "HKP_ACT_RELU_FWD";
    }
    if(activation == "ABS")
    {
        return "HKP_ACT_ABS";
    }
    if(activation == "NEG")
    {
        return "HKP_ACT_NEG";
    }

    // Unreachable via matching, which admits only the three modes this pack declares.
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "kernel '" + toString(kernel.kernelId)
                                                       + "' declares unsupported activation '"
                                                       + activation + "'");
}

const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    if(it == tensors.end() || it->second == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched conv_pointwise_rtc graph has no tensor for uid " + std::to_string(uid));
    }
    return *it->second;
}

FusedNodes requireFusedNodes(const MatchContext& context)
{
    const auto nodes = fusedNodes(context);
    if(nodes.conv == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched conv_pointwise_rtc graph is not a ConvolutionFwd followed by a Pointwise");
    }
    return nodes;
}

/**
 * @brief The native dispatch behind this pack's UDD: specializes, sizes and launches the
 *        fused kernel. Everything graph- and kernel-derived resolves once at prepare();
 *        launch() only resolves buffers and launches, so nothing mutates once prepared and
 *        concurrent execution is safe.
 */
class ConvPointwiseRtcDispatchHandler
    : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    /// @param kpackLoader Likewise. Unused by the descriptors this pack ships, which are
    ///        all embedded_source, but buildIngestorKernelCode takes it for the kinds it
    ///        also serves.
    ConvPointwiseRtcDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                    const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// No scratch. The convolution's output tensor is virtual, so the accumulator never
    /// leaves a register and there is no intermediate to stage; every thread writes its
    /// own output element exactly once. Zero here is the kernel's fact, not a default.
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
        // Reads the operand uids the graph match bound rather than re-deriving them.
        const auto binding = convPointwiseRtcBinding(bound);
        const auto nodes = requireFusedNodes(context);

        const auto& xTensor = requireTensor(context, binding.x);
        const auto& wTensor = requireTensor(context, binding.w);
        const auto& yTensor = requireTensor(context, binding.y);

        // Rank 4, dims and strides present and equally sized, the two channel quotients
        // exact and the output extents consistent with the padding -- all validated by the
        // graph matcher, which is why this reads rather than re-checks.
        const auto* xDims = xTensor.dims();
        const auto* wDims = wTensor.dims();
        const auto* yDims = yTensor.dims();

        ConvPointwiseGeometry geometry;
        geometry.batchCount = xDims->Get(0);
        geometry.inChannelsPerGroup = wDims->Get(1);
        geometry.groupCount = xDims->Get(1) / geometry.inChannelsPerGroup;
        geometry.outChannelsPerGroup = wDims->Get(0) / geometry.groupCount;
        geometry.inHeight = xDims->Get(2);
        geometry.inWidth = xDims->Get(3);
        geometry.filterHeight = wDims->Get(2);
        geometry.filterWidth = wDims->Get(3);
        // From the graph's y dims, never recomputed from the padding: the matcher has
        // already proved the two agree, and the kernel's contract is that these are the
        // extents it iterates.
        geometry.outHeight = yDims->Get(2);
        geometry.outWidth = yDims->Get(3);
        geometry.strideHeight = nodes.conv->stride()->Get(0);
        geometry.strideWidth = nodes.conv->stride()->Get(1);
        geometry.dilationHeight = nodes.conv->dilation()->Get(0);
        geometry.dilationWidth = nodes.conv->dilation()->Get(1);
        geometry.prePadHeight = nodes.conv->pre_padding()->Get(0);
        geometry.prePadWidth = nodes.conv->pre_padding()->Get(1);
        // post_padding is read nowhere below, and that is the enforced-inert disposition
        // the matcher's output-extent equality pays for -- not an omission.
        for(flatbuffers::uoffset_t axis = 0; axis < SUPPORTED_RANK; ++axis)
        {
            geometry.xStrides[axis] = xTensor.strides()->Get(axis);
            geometry.wStrides[axis] = wTensor.strides()->Get(axis);
            // The POINTWISE output's strides. The convolution's own output tensor is
            // virtual and is never written, so its strides address nothing.
            geometry.yStrides[axis] = yTensor.strides()->Get(axis);
        }

        // The defaults ReluForward's constructor supplies, which is also the path the
        // reference takes when the attributes are absent -- PointwisePlan::execute only
        // enters its parameterized branch when one of them has a value. Supplied for every
        // activation, not only RELU_FWD: the kernel's argument list is fixed-arity across
        // the specialization axis (hkpActivation_HKP_ACT_ABS and _NEG take and discard
        // them), so there is no conditional ABI to replay here.
        geometry.reluLowerClip
            = nodes.pointwise->relu_lower_clip().value_or(DEFAULT_RELU_LOWER_CLIP);
        geometry.reluUpperClip
            = nodes.pointwise->relu_upper_clip().value_or(DEFAULT_RELU_UPPER_CLIP);
        geometry.reluLowerClipSlope
            = nodes.pointwise->relu_lower_clip_slope().value_or(DEFAULT_RELU_LOWER_CLIP_SLOPE);

        const auto blockSize
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));

        compilation::KernelCompileOptions options(&xTensor, context.deviceProperties.gcnArchName);
        // The two macros ConvFwdPointwiseFused.cpp guards with #error, both handler-
        // supplied rather than descriptor-bound: each is a VOCABULARY TRANSLATION of this
        // candidate's metadata (FLOAT -> HKP_DTYPE_FLOAT, RELU_FWD -> HKP_ACT_RELU_FWD)
        // and the descriptor substituter does literal replacement and nothing else.
        // Growing the substituter to carry this would be the wrong side of that seam.
        options.add("HKP_IO_DTYPE", ioDtypeTagFor(kernel));
        options.add("HKP_ACTIVATION", activationTagFor(kernel));

        // Routed rather than compiled directly. ConvNative.cpp calls
        // _kernelCompiler.compile(source.sourceFile, options) itself, which serves
        // embedded_source alone -- which is why a hiprtc_file descriptor under that pack
        // throws at plan build however correct the descriptor is. This pack's descriptors
        // are all embedded_source today and the routing still belongs here.
        auto code
            = buildIngestorKernelCode(_kernelCompiler, _kpackLoader, context, kernel, options);

        // int64_t throughout: the matcher admits output planes beyond 2^31, and a 32-bit
        // product would wrap both the grid extent and the comparison the kernel's own
        // bounds guard makes against it.
        const int64_t spatialCount = geometry.outHeight * geometry.outWidth;
        const int64_t planeCount
            = geometry.groupCount * geometry.batchCount * geometry.outChannelsPerGroup;
        const auto gridX = static_cast<unsigned int>(
            (spatialCount + static_cast<int64_t>(blockSize) - 1) / static_cast<int64_t>(blockSize));
        const auto gridY = static_cast<unsigned int>(planeCount);

        // x is rounded up on purpose: the kernel returns early for
        // spatialIndex >= outHeight*outWidth, so a partially populated final block is
        // correct and an exact-multiple grid would leave the tail unwritten. y is exact.
        code.kernel->setBlockSize(blockSize, 1, 1);
        code.kernel->setGridSize(gridX, gridY, 1);

        return std::make_unique<PreparedConvPointwiseRtc>(
            std::move(code.program), std::move(code.kernel), binding, geometry);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedFusion = dynamic_cast<const PreparedConvPointwiseRtc&>(prepared);
        const auto& binding = preparedFusion.binding();
        const auto& geometry = preparedFusion.geometry();

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        // 34 arguments, in the order kernels/ConvFwdPointwiseFused.cpp declares them.
        // Wrong arity or wrong order is diagnosed NOWHERE: hipRTC compiles the kernel,
        // getKernel() resolves it, and hipModuleLaunchKernel reads one kernarg per
        // parameter the kernel declared, so a short list reads whatever is next in memory
        // and two swapped same-typed scalars are a wrong answer with no error. The
        // twelve strides are the live hazard -- x's, w's and y's are interleaved nowhere
        // and grouped per operand here for exactly that reason.
        preparedFusion.kernel().launch(handle.getStream(),
                                       x.ptr,
                                       w.ptr,
                                       y.ptr,
                                       geometry.batchCount,
                                       geometry.groupCount,
                                       geometry.inChannelsPerGroup,
                                       geometry.outChannelsPerGroup,
                                       geometry.inHeight,
                                       geometry.inWidth,
                                       geometry.filterHeight,
                                       geometry.filterWidth,
                                       geometry.outHeight,
                                       geometry.outWidth,
                                       geometry.strideHeight,
                                       geometry.strideWidth,
                                       geometry.dilationHeight,
                                       geometry.dilationWidth,
                                       geometry.prePadHeight,
                                       geometry.prePadWidth,
                                       geometry.xStrides[0],
                                       geometry.xStrides[1],
                                       geometry.xStrides[2],
                                       geometry.xStrides[3],
                                       geometry.wStrides[0],
                                       geometry.wStrides[1],
                                       geometry.wStrides[2],
                                       geometry.wStrides[3],
                                       geometry.yStrides[0],
                                       geometry.yStrides[1],
                                       geometry.yStrides[2],
                                       geometry.yStrides[3],
                                       geometry.reluLowerClip,
                                       geometry.reluUpperClip,
                                       geometry.reluLowerClipSlope);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& convPointwiseRtcKpackModuleCache()
{
    // Process-lifetime, as the pointwise and batchnorm packs' are: a loaded module
    // outlives the plan that loaded it, and the cache is what makes that one module rather
    // than one per dispatch.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetConvPointwiseRtcModuleCache()
{
    convPointwiseRtcKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer
/// to it, but a provider's Container is created and destroyed per handle, so it (and the
/// compiler and loader it holds) must outlive every Container.
const ConvPointwiseRtcDispatchHandler& convPointwiseRtcDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(convPointwiseRtcKpackModuleCache());
    static const ConvPointwiseRtcDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerConvPointwiseRtcSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &convPointwiseRtcGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &convPointwiseRtcKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &convPointwiseRtcScore);
    scope.add(std::string(DISPATCH_SYMBOL), &convPointwiseRtcDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
