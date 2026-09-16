// Copyright Â© Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestConvBiasMatchers.cpp
 * @brief What Hackweek:ConvBias's matchers accept and refuse.
 *
 * The acceptance cases are the shape envelope the kernel was proved over (the graph as
 * given, all-extents-one, a non-block-multiple element count, non-packed and channel-last
 * strides, grouped, strided/dilated/padded, a broadcast bias, swapped ADD operands); the
 * refusals are every precondition the kernel has that the frontend does not enforce. Both
 * directions are load-bearing: an accept-only suite passes for a matcher stuck on true, a
 * refusal-only suite for one stuck on false.
 *
 * RANK is the axis with the most cases on both sides, because it is the axis where the
 * matcher and the kernel most easily come apart. The kernel is one canonical five-axis
 * body and the matcher expresses ranks 3, 4 and 5 in it, so the accepts prove each rank
 * carries its OWN conv parameters, layout and broadcast -- not merely that a rank is let
 * through -- and the refusals pin the two boundaries (rank 2 below, rank 6 above) plus the
 * two ways a graph can be internally inconsistent about its rank: operands that disagree
 * with each other, and a spatial attribute vector that disagrees with the operands.
 *
 * This engine is the only two-node one in the provider, so several refusals here have no
 * counterpart in the sibling suites: a graph whose pointwise does not consume the
 * convolution's y is not a fusion at all, and a graph whose y is not virtual is one whose
 * convolution result someone else can observe.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::MatchContext;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// The fourth engine: two nodes, one pack, so `operationMatcher` is empty -- its graph
/// match admits the topology and validates it in one pass.
///
/// Declared here rather than in PointwiseTestGraphs.hpp because the three token slots
/// PackSymbols carries do not fit a four-operand fusion; the fourth is named below.
constexpr PackSymbols CONV_BIAS{"Hackweek:ConvBias",
                                "Hackweek.conv_bias.graph_match",
                                "",
                                "Hackweek.conv_bias.kernel_match",
                                "Hackweek.conv_bias.score",
                                "Hackweek.conv_bias.dispatch",
                                "conv_bias.x.uid",
                                "conv_bias.w.uid",
                                "conv_bias.out.uid"};

/// The operand token PackSymbols has no slot for. The convolution's y is deliberately
/// absent from the token set: it is virtual, so it has no device buffer.
constexpr std::string_view CB_BIAS_TOKEN = "conv_bias.bias.uid";

/// Tensor uids buildConvBiasGraph() uses, matching the proved graph's own numbering.
constexpr int64_t CB_Y_UID = 0;
constexpr int64_t CB_X_UID = 1;
constexpr int64_t CB_W_UID = 2;
constexpr int64_t CB_BIAS_UID = 3;
constexpr int64_t CB_OUT_UID = 4;

/// Knobs for one built graph. A struct rather than twenty defaulted parameters: this
/// matcher gates more independent facts than any sibling, and a call site reading
/// `{.convMode = CONVOLUTION}` says which refusal it is exercising.
struct ConvBiasGraphSpec
{
    data_objects::DataType dataType = data_objects::DataType::FLOAT;
    data_objects::ConvMode convMode = data_objects::ConvMode::CROSS_CORRELATION;
    data_objects::PointwiseMode operation = data_objects::PointwiseMode::ADD;

    std::vector<int64_t> xDims = {1, 3, 8, 8};
    std::vector<int64_t> wDims = {1, 3, 1, 1};
    /// Defaults to y's own shape, which is derived from the conv parameters.
    std::optional<std::vector<int64_t>> outDims = std::nullopt;
    /// Defaults to outDims -- a non-broadcast bias.
    std::optional<std::vector<int64_t>> biasDims = std::nullopt;
    /// Defaults to the shape the conv parameters produce.
    std::optional<std::vector<int64_t>> yDims = std::nullopt;

    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    std::vector<int64_t> prePadding = {0, 0};
    std::vector<int64_t> postPadding = {0, 0};

    std::optional<std::vector<int64_t>> xStrides = std::nullopt;
    std::optional<std::vector<int64_t>> outStrides = std::nullopt;

    /// ADD is commutative, so the frontend may emit the convolution's y as either operand.
    bool swapPointwiseOperands = false;
    /// The fusion's defining property. Cleared for the "conv result is a real output"
    /// refusal.
    bool yVirtual = true;
    /// Points the pointwise's second operand at a uid that is not the conv's y, so the two
    /// nodes are unrelated and this is not a fusion.
    bool detachPointwiseFromConv = false;
    /// Emits out's uid as the conv's x, for the in-place refusal.
    bool aliasOutWithX = false;
    /// A real third operand, which only BINARY_SELECT has.
    bool includeThirdOperand = false;
    /// Set when a mode that consults it would -- here, always a refusal.
    std::optional<float> reluLowerClip = std::nullopt;

    std::optional<data_objects::DataType> nodeComputeDataType = std::nullopt;
    std::optional<data_objects::DataType> biasDataType = std::nullopt;
    bool isOverrideShapeEnabled = false;
    bool outPassByValue = false;
};

/// The spatial extent the convolution parameters produce, per the reference's own mapping.
///
/// A non-positive stride produces no extent. Callers that mean to build such a graph state
/// yDims themselves and never reach this; the guard is here so that a case which forgets
/// refuses through the matcher rather than faulting the whole binary on a divide.
int64_t convolvedExtent(const ConvBiasGraphSpec& spec, size_t spatial)
{
    const auto axis = 2 + spatial;
    const auto stride = spec.stride.at(spatial);
    if(stride <= 0)
    {
        return spec.xDims.at(axis);
    }
    const auto window = (spec.wDims.at(axis) - 1) * spec.dilation.at(spatial) + 1;
    return (spec.xDims.at(axis) + spec.prePadding.at(spatial) + spec.postPadding.at(spatial)
            - window)
               / stride
           + 1;
}

/// Builds the two-node fused graph: ConvolutionFwd -> Pointwise(ADD) over a virtual y.
flatbuffers::FlatBufferBuilder buildConvBiasGraph(const ConvBiasGraphSpec& spec = {})
{
    // Derived at whatever rank xDims carries, so a rank-3 or rank-5 case states only its
    // tensors and its conv parameters and gets the y the reference's mapping produces --
    // the same way a rank-4 case does. A case that means to state a y the parameters do
    // NOT produce (the extent-identity refusal) overrides yDims instead.
    //
    // Derived only when it is actually needed, NOT through value_or: several refusal cases
    // are deliberately self-inconsistent -- a rank-5 x against a rank-4 w, or a length-2
    // stride against three spatial axes -- and deriving a y from them would index past the
    // end of a vector the case shortened on purpose. Those cases all state yDims
    // themselves, so the derivation must not run for them. Eagerly computing it turns a
    // matcher refusal into a std::out_of_range from the builder, which passes for a
    // failure while proving nothing about the matcher.
    std::vector<int64_t> yDims;
    if(spec.yDims.has_value())
    {
        yDims = *spec.yDims;
    }
    else
    {
        yDims = {spec.xDims.at(0), spec.wDims.at(0)};
        for(size_t spatial = 0; spatial + 2 < spec.xDims.size(); ++spatial)
        {
            yDims.push_back(convolvedExtent(spec, spatial));
        }
    }
    // The added tensor's channels default to the convolution's, which is the aligned case;
    // a caller overriding outDims exercises the convK == 1 broadcast instead.
    const auto outDims = spec.outDims.value_or(yDims);
    const auto biasDims = spec.biasDims.value_or(outDims);

    const auto xStrides = spec.xStrides.value_or(packedRowMajorStrides(spec.xDims));
    const auto wStrides = packedRowMajorStrides(spec.wDims);
    const auto yStrides = packedRowMajorStrides(yDims);
    const auto biasStrides = packedRowMajorStrides(biasDims);
    const auto outStrides = spec.outStrides.value_or(packedRowMajorStrides(outDims));

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;

    const auto convXUid = spec.aliasOutWithX ? CB_OUT_UID : CB_X_UID;
    if(!spec.aliasOutWithX)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, CB_X_UID, nullptr, spec.dataType, &xStrides, &spec.xDims));
    }
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, CB_W_UID, nullptr, spec.dataType, &wStrides, &spec.wDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, CB_Y_UID, nullptr, spec.dataType, &yStrides, &yDims, spec.yVirtual));
    tensors.push_back(
        data_objects::CreateTensorAttributesDirect(builder,
                                                   CB_BIAS_UID,
                                                   nullptr,
                                                   spec.biasDataType.value_or(spec.dataType),
                                                   &biasStrides,
                                                   &biasDims));
    tensors.push_back(
        data_objects::CreateTensorAttributesDirect(builder,
                                                   CB_OUT_UID,
                                                   nullptr,
                                                   spec.dataType,
                                                   &outStrides,
                                                   &outDims,
                                                   false,
                                                   data_objects::TensorValue::NONE,
                                                   0,
                                                   spec.outPassByValue));
    // A real third operand needs a tensor of its own; reuse the bias shape.
    constexpr int64_t THIRD_OPERAND_UID = 7;
    if(spec.includeThirdOperand)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, THIRD_OPERAND_UID, nullptr, spec.dataType, &biasStrides, &biasDims));
    }

    auto convAttributes
        = data_objects::CreateConvolutionFwdAttributesDirect(builder,
                                                             convXUid,
                                                             CB_W_UID,
                                                             CB_Y_UID,
                                                             &spec.prePadding,
                                                             &spec.postPadding,
                                                             &spec.stride,
                                                             &spec.dilation,
                                                             spec.convMode);

    // The fused operand is the conv's y unless this case deliberately detaches it.
    const auto fusedUid = spec.detachPointwiseFromConv ? CB_BIAS_UID : CB_Y_UID;
    data_objects::PointwiseAttributesBuilder pointwiseBuilder(builder);
    pointwiseBuilder.add_operation(spec.operation);
    if(spec.swapPointwiseOperands)
    {
        pointwiseBuilder.add_in_0_tensor_uid(CB_BIAS_UID);
        pointwiseBuilder.add_in_1_tensor_uid(fusedUid);
    }
    else
    {
        pointwiseBuilder.add_in_0_tensor_uid(fusedUid);
        pointwiseBuilder.add_in_1_tensor_uid(CB_BIAS_UID);
    }
    if(spec.includeThirdOperand)
    {
        pointwiseBuilder.add_in_2_tensor_uid(THIRD_OPERAND_UID);
    }
    if(spec.reluLowerClip.has_value())
    {
        pointwiseBuilder.add_relu_lower_clip(*spec.reluLowerClip);
    }
    pointwiseBuilder.add_out_0_tensor_uid(CB_OUT_UID);
    auto pointwiseAttributes = pointwiseBuilder.Finish();

    const auto computeDataType
        = spec.nodeComputeDataType.value_or(data_objects::DataType::FLOAT);
    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder,
                                       "ConvolutionFprop_0",
                                       computeDataType,
                                       data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                       convAttributes.Union()));
    nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                   "Pointwise_1",
                                                   computeDataType,
                                                   data_objects::NodeAttributes::PointwiseAttributes,
                                                   pointwiseAttributes.Union()));

    auto name = builder.CreateString("conv_bias_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_compute_data_type(data_objects::DataType::FLOAT);
    graphBuilder.add_intermediate_data_type(data_objects::DataType::FLOAT);
    graphBuilder.add_io_data_type(data_objects::DataType::FLOAT);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    graphBuilder.add_is_override_shape_enabled(spec.isOverrideShapeEnabled);
    builder.Finish(graphBuilder.Finish());

    return builder;
}

bool matches(const MatchContext& context)
{
    return matchesGraph(CONV_BIAS, context).has_value();
}

/// A KernelDefinition for a conv_bias candidate: one block size, one dtype, matching the
/// KMD this pack's config declares.
hipdnn_plugin_sdk::ingestor::KernelDefinition makeConvBiasKernel(int64_t blockSize,
                                                                 const std::string& dtype = "FLOAT")
{
    return makeKernel(blockSize, dtype, "ConvBiasFusedFwd");
}

/// One graph and a readable name for a failing run.
struct GraphCase
{
    std::string name;
    ConvBiasGraphSpec spec;
};

// ---------------------------------------------------------------------------
// Graph-scoped matcher: the shapes the kernel was proved over
// ---------------------------------------------------------------------------

class TestConvBiasGraphMatcherAcceptance : public ::testing::TestWithParam<GraphCase>
{
};

TEST_P(TestConvBiasGraphMatcherAcceptance, Accepts)
{
    const GraphFixture fixture(buildConvBiasGraph(GetParam().spec));

    EXPECT_TRUE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestConvBiasGraphMatcherAcceptance,
    ::testing::ValuesIn(std::vector<GraphCase>{
        {"TheProvedGraphShape", ConvBiasGraphSpec{}},
        {"AllExtentsOne",
         // Validated by the authoring harness at an exact match, max_abs_err 0.
         ConvBiasGraphSpec{.xDims = {1, 1, 1, 1}, .wDims = {1, 1, 1, 1}}},
        {"ElementCountNotABlockMultiple",
         // 140 output elements under a 256-thread block: the final block is partially
         // populated and the kernel's own bounds guard is what makes that correct.
         ConvBiasGraphSpec{.xDims = {1, 1, 7, 20}, .wDims = {1, 1, 1, 1}}},
        {"BroadcastConvOutputAcrossChannels",
         // convK == 1 against a three-channel added tensor: one filter's result added to
         // every channel of the bias. The kOut = 0 branch in the kernel.
         ConvBiasGraphSpec{.xDims = {1, 3, 8, 8},
                           .wDims = {1, 3, 1, 1},
                           .outDims = std::vector<int64_t>{1, 3, 8, 8}}},
        {"ChannelLastStrides",
         // Layout is not a schema field; NHWC is the same graph with channel-last strides,
         // and the kernel addresses through them rather than assuming an order.
         ConvBiasGraphSpec{.xDims = {1, 3, 4, 4},
                           .wDims = {3, 3, 1, 1},
                           .xStrides = std::vector<int64_t>{48, 1, 12, 3},
                           .outStrides = std::vector<int64_t>{48, 1, 12, 3}}},
        {"NonContiguousX",
         // x is a view into a larger buffer while out is packed: the two stride vectors are
         // read independently, which is the case a shared-stride assumption would break.
         ConvBiasGraphSpec{.xDims = {1, 3, 4, 4},
                           .wDims = {3, 3, 1, 1},
                           .xStrides = std::vector<int64_t>{192, 64, 8, 1}}},
        {"StridedDilatedAndPadded",
         // The three conv parameters that the shipped ConvFwd scaffold refuses outright.
         ConvBiasGraphSpec{.xDims = {1, 2, 9, 9},
                           .wDims = {2, 2, 3, 3},
                           .stride = {2, 2},
                           .dilation = {2, 2},
                           .prePadding = {2, 2}}},
        {"GroupedConvolution",
         // groups = xC / wC = 4 / 2, carried by no schema field.
         ConvBiasGraphSpec{.xDims = {1, 4, 5, 5}, .wDims = {4, 2, 1, 1}}},
        {"BroadcastBias",
         // A [1, C, 1, 1] bias against a full output: the handler zeroes the broadcast
         // axes' strides, which is the kernel's stated precondition.
         ConvBiasGraphSpec{.xDims = {1, 3, 6, 6},
                           .wDims = {3, 3, 1, 1},
                           .biasDims = std::vector<int64_t>{1, 3, 1, 1}}},
        {"SwappedAddOperands",
         // ADD is commutative and the frontend may emit either order, so the conv's y is
         // bound by identity rather than by position.
         ConvBiasGraphSpec{.swapPointwiseOperands = true}},

        // --- Rank. The kernel is one canonical five-axis body; these are the ranks the
        // matcher canonicalises into it. Each has a bundle case behind it.
        {"Rank3",
         // One spatial axis, so every conv vector is length 1. Canonically it occupies the
         // W slot and leaves D and H degenerate -- extent 1, filter 1, stride 0 -- which is
         // why the same kernel serves it.
         ConvBiasGraphSpec{.xDims = {1, 2, 8},
                           .wDims = {2, 2, 3},
                           .stride = {1},
                           .dilation = {1},
                           .prePadding = {0},
                           .postPadding = {0}}},
        {"Rank3StridedDilatedAndPadded",
         // The conv parameters at the new rank, not just the trivial ones: refusing to
         // prove these separately is how a rank widening quietly serves only stride 1.
         // (8 + 2 - ((3-1)*2+1)) / 2 + 1 = 4.
         ConvBiasGraphSpec{.xDims = {1, 2, 9},
                           .wDims = {2, 2, 3},
                           .stride = {2},
                           .dilation = {2},
                           .prePadding = {2},
                           .postPadding = {0}}},
        {"Rank3ChannelLastStrides",
         // NWC: layout is the stride vector at rank 3 exactly as it is at rank 4.
         ConvBiasGraphSpec{.xDims = {1, 2, 8},
                           .wDims = {2, 2, 1},
                           .stride = {1},
                           .dilation = {1},
                           .prePadding = {0},
                           .postPadding = {0},
                           .xStrides = std::vector<int64_t>{16, 1, 2},
                           .outStrides = std::vector<int64_t>{16, 1, 2}}},
        {"Rank3BroadcastBias",
         // A [1, C, 1] bias: the broadcast-axis zeroing has to happen in canonical space,
         // or the zeroed axis and the addressed axis are different axes.
         ConvBiasGraphSpec{.xDims = {1, 2, 8},
                           .wDims = {2, 2, 1},
                           .biasDims = std::vector<int64_t>{1, 2, 1},
                           .stride = {1},
                           .dilation = {1},
                           .prePadding = {0},
                           .postPadding = {0}}},
        {"Rank5",
         // Three spatial axes, which is the canonical form itself: no slot is degenerate
         // and all three of the kernel's spatial loops carry a real extent.
         ConvBiasGraphSpec{.xDims = {1, 2, 4, 4, 4},
                           .wDims = {2, 2, 2, 2, 2},
                           .stride = {1, 1, 1},
                           .dilation = {1, 1, 1},
                           .prePadding = {0, 0, 0},
                           .postPadding = {0, 0, 0}}},
        {"Rank5StridedAndPadded",
         // (5 + 1 - 3) / 2 + 1 = 2 on each of the three axes.
         ConvBiasGraphSpec{.xDims = {1, 2, 5, 5, 5},
                           .wDims = {2, 2, 3, 3, 3},
                           .stride = {2, 2, 2},
                           .dilation = {1, 1, 1},
                           .prePadding = {1, 1, 1},
                           .postPadding = {0, 0, 0}}},
        {"Rank5ChannelLastStrides",
         // NDHWC.
         ConvBiasGraphSpec{.xDims = {1, 2, 4, 4, 4},
                           .wDims = {2, 2, 1, 1, 1},
                           .stride = {1, 1, 1},
                           .dilation = {1, 1, 1},
                           .prePadding = {0, 0, 0},
                           .postPadding = {0, 0, 0},
                           .xStrides = std::vector<int64_t>{128, 1, 32, 8, 2},
                           .outStrides = std::vector<int64_t>{128, 1, 32, 8, 2}}},
        {"Rank5GroupedConvolution",
         // groups = 4 / 2 at the rank where the canonical channel slot is furthest from
         // the axis the graph wrote it on.
         ConvBiasGraphSpec{.xDims = {1, 4, 3, 3, 3},
                           .wDims = {4, 2, 1, 1, 1},
                           .stride = {1, 1, 1},
                           .dilation = {1, 1, 1},
                           .prePadding = {0, 0, 0},
                           .postPadding = {0, 0, 0}}},
        {"Rank5BroadcastBias",
         ConvBiasGraphSpec{.xDims = {1, 2, 3, 3, 3},
                           .wDims = {2, 2, 1, 1, 1},
                           .biasDims = std::vector<int64_t>{1, 2, 1, 1, 1},
                           .stride = {1, 1, 1},
                           .dilation = {1, 1, 1},
                           .prePadding = {0, 0, 0},
                           .postPadding = {0, 0, 0}}},
    }),
    [](const ::testing::TestParamInfo<GraphCase>& info) { return info.param.name; });

TEST(TestConvBiasBinding, BindsFourOperandUidsAndNotTheVirtualY)
{
    const GraphFixture fixture(buildConvBiasGraph());

    const auto bound = matchesGraph(CONV_BIAS, fixture.context());
    ASSERT_TRUE(bound.has_value());

    using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
    EXPECT_EQ(tryGetBoundInt(*bound, CONV_BIAS.inputAToken), CB_X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, CONV_BIAS.inputBToken), CB_W_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, CB_BIAS_TOKEN), CB_BIAS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, CONV_BIAS.outputToken), CB_OUT_UID);
    // The convolution's y is virtual, so it has no device buffer and must not be bound as
    // one. A token for it would resolve to no buffer at launch.
    EXPECT_FALSE(tryGetBoundInt(*bound, "conv_bias.y.uid").has_value());
}

TEST(TestConvBiasBinding, BindsTheBiasByIdentityWhenTheAddOperandsAreSwapped)
{
    const GraphFixture fixture(buildConvBiasGraph({.swapPointwiseOperands = true}));

    const auto bound = matchesGraph(CONV_BIAS, fixture.context());
    ASSERT_TRUE(bound.has_value());

    // The point of the case: in_0 is now the bias and in_1 is the conv's y, and the bias
    // token must still name the bias. Binding by position would have swapped them, which
    // is a wrong answer with no diagnostic rather than a failure to match.
    using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
    EXPECT_EQ(tryGetBoundInt(*bound, CB_BIAS_TOKEN), CB_BIAS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, CONV_BIAS.outputToken), CB_OUT_UID);
}

// ---------------------------------------------------------------------------
// Graph-scoped matcher: refusals
// ---------------------------------------------------------------------------

class TestConvBiasGraphMatcherRefusal : public ::testing::TestWithParam<GraphCase>
{
};

TEST_P(TestConvBiasGraphMatcherRefusal, Refuses)
{
    const GraphFixture fixture(buildConvBiasGraph(GetParam().spec));

    EXPECT_FALSE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestConvBiasGraphMatcherRefusal,
    ::testing::ValuesIn(std::vector<GraphCase>{
        {"ConvolutionMode",
         // Not implemented rather than implemented-and-untested: the hipDNN CPU reference
         // rejects CONVOLUTION too, so a flipped-filter path has no oracle.
         ConvBiasGraphSpec{.convMode = data_objects::ConvMode::CONVOLUTION}},
        {"UnsetConvMode",
         // The FlatBuffers default for an absent scalar, delivered as a valid enumerator
         // with no defined kernel meaning.
         ConvBiasGraphSpec{.convMode = data_objects::ConvMode::UNSET}},
        {"PointwiseMul", ConvBiasGraphSpec{.operation = data_objects::PointwiseMode::MUL}},
        {"PointwiseSub", ConvBiasGraphSpec{.operation = data_objects::PointwiseMode::SUB}},
        {"UnsetPointwiseMode",
         ConvBiasGraphSpec{.operation = data_objects::PointwiseMode::UNSET}},
        {"NonVirtualConvOutput",
         // The conv's y is a real output someone else can observe, so eliding it would
         // leave that buffer unwritten. This is the fusion's defining precondition.
         ConvBiasGraphSpec{.yVirtual = false}},
        {"PointwiseDoesNotConsumeTheConvOutput",
         // Two unrelated nodes that happen to be a conv and a pointwise. Not a fusion, and
         // matching on node types alone would have admitted it.
         ConvBiasGraphSpec{.detachPointwiseFromConv = true}},
        {"InPlaceGraph",
         // out aliasing an input is undefined behaviour against the kernel's four
         // `__restrict__` pointers, and no run ever exercised it.
         ConvBiasGraphSpec{.aliasOutWithX = true}},
        {"TernaryPointwise",
         // in_2 is BINARY_SELECT's third operand; requiring its absence closes the case
         // where operation is ADD and it is set anyway.
         ConvBiasGraphSpec{.includeThirdOperand = true}},
        {"ReluClipSet",
         // Consulted only by RELU_* modes, so an ADD carrying one is a graph whose author
         // expected something this kernel does not do.
         ConvBiasGraphSpec{.reluLowerClip = 0.0F}},
        {"HalfDataType",
         // Compiles, but no numerics were ever run for it, so it is refused rather than
         // claimed.
         ConvBiasGraphSpec{.dataType = data_objects::DataType::HALF}},
        {"Bfloat16DataType", ConvBiasGraphSpec{.dataType = data_objects::DataType::BFLOAT16}},
        {"DoubleDataType", ConvBiasGraphSpec{.dataType = data_objects::DataType::DOUBLE}},
        {"MixedOperandDataTypes",
         // One HKP_CONV_BIAS_TYPE covers all five tensors, so a graph storing the bias at a
         // different width has no candidate that can serve it.
         ConvBiasGraphSpec{.biasDataType = data_objects::DataType::HALF}},
        {"Rank2Tensors",
         // Two axes is N and C with no spatial axis at all -- a convolution whose window
         // has nowhere to slide. Below the canonical form's floor, and the reference does
         // not produce one.
         ConvBiasGraphSpec{.xDims = {1, 3},
                           .wDims = {1, 3},
                           .stride = {},
                           .dilation = {},
                           .prePadding = {},
                           .postPadding = {}}},
        {"Rank6Tensors",
         // Four spatial axes. The kernel has three spatial loops, so this is the first rank
         // the canonical form cannot express -- refused rather than silently folded.
         ConvBiasGraphSpec{.xDims = {1, 2, 3, 3, 3, 3},
                           .wDims = {2, 2, 1, 1, 1, 1},
                           .stride = {1, 1, 1, 1},
                           .dilation = {1, 1, 1, 1},
                           .prePadding = {0, 0, 0, 0},
                           .postPadding = {0, 0, 0, 0}}},
        {"MixedRankOperands",
         // A rank-5 x against a rank-4 everything-else. There is no single canonical
         // alignment for this graph, and canonicalising each tensor against its own rank
         // would address x on axes the other four do not have.
         ConvBiasGraphSpec{.xDims = {1, 2, 1, 4, 4},
                           .wDims = {2, 2, 1, 1},
                           .outDims = std::vector<int64_t>{1, 2, 4, 4},
                           .yDims = std::vector<int64_t>{1, 2, 4, 4}}},
        {"SpatialVectorShorterThanTheTensorRank",
         // A rank-5 graph carrying a length-2 stride. Padding it to three would be
         // inventing the value for an axis its author did say something about.
         ConvBiasGraphSpec{.xDims = {1, 2, 4, 4, 4},
                           .wDims = {2, 2, 1, 1, 1},
                           .yDims = std::vector<int64_t>{1, 2, 4, 4, 4},
                           .stride = {1, 1},
                           .dilation = {1, 1, 1},
                           .prePadding = {0, 0, 0},
                           .postPadding = {0, 0, 0}}},
        {"ChannelsNotDivisibleByFilterChannels",
         // xC / wC is the group count, and an inexact quotient is not one.
         ConvBiasGraphSpec{.xDims = {1, 5, 4, 4}, .wDims = {2, 2, 1, 1}}},
        {"OutputChannelsNotDivisibleByGroups",
         // groups = 4 / 2 = 2, and convK = 3 does not divide evenly among them.
         ConvBiasGraphSpec{.xDims = {1, 4, 4, 4}, .wDims = {3, 2, 1, 1}}},
        {"StatedOutputExtentDisagreesWithTheConvParameters",
         // The identity that makes post_padding consumed rather than ignored: the stated y
         // must be the extent these parameters actually produce.
         ConvBiasGraphSpec{.xDims = {1, 1, 8, 8},
                           .wDims = {1, 1, 3, 3},
                           .yDims = std::vector<int64_t>{1, 1, 8, 8}}},
        {"BiasNotBroadcastCompatibleWithTheOutput",
         // Read at the output's coordinates, so an extent that is neither equal nor 1 is
         // read past its end.
         ConvBiasGraphSpec{.xDims = {1, 3, 6, 6},
                           .wDims = {3, 3, 1, 1},
                           .biasDims = std::vector<int64_t>{1, 2, 6, 6}}},
        {"ConvChannelsNeitherAlignedNorBroadcast",
         // y.dims[1] is 2 against a 3-channel output: neither the aligned case nor the
         // convK == 1 broadcast the kernel implements.
         ConvBiasGraphSpec{.xDims = {1, 3, 4, 4},
                           .wDims = {2, 3, 1, 1},
                           .outDims = std::vector<int64_t>{1, 3, 4, 4}}},
        {"ZeroStrideOnTheOutput",
         // Written once per logical coordinate, so two coordinates sharing an address is a
         // race rather than a broadcast.
         ConvBiasGraphSpec{.xDims = {1, 3, 4, 4},
                           .wDims = {3, 3, 1, 1},
                           .outStrides = std::vector<int64_t>{48, 0, 4, 1}}},
        {"SpatialVectorLongerThanTheTensorRank",
         // A length-3 stride on a rank-4 graph. The tensors say two spatial axes and the
         // attribute says three; truncating would pick two of the author's three values
         // and convolve with those.
         ConvBiasGraphSpec{.stride = {1, 1, 1}}},
        {"ZeroStride",
         // yDims are stated as a stride of 1 would produce them, so the stated-extent
         // identity holds and the zero stride is the only thing left to refuse. Derived
         // from the parameters instead, the extent would disagree too and this case would
         // still pass with the matcher's stride check deleted.
         ConvBiasGraphSpec{.yDims = std::vector<int64_t>{1, 1, 8, 8}, .stride = {0, 0}}},
        {"NegativePadding",
         ConvBiasGraphSpec{.prePadding = {-1, -1}}},
        {"NarrowerComputePrecision",
         // The accumulator is float unconditionally, so a node asking for half compute
         // would be served at a precision it did not ask for.
         ConvBiasGraphSpec{.nodeComputeDataType = data_objects::DataType::HALF}},
        {"OverrideShapesEnabled",
         // The extents come from the dims recorded in the graph, so a shape overridden
         // after lowering would never reach the kernel.
         ConvBiasGraphSpec{.isOverrideShapeEnabled = true}},
        {"PassByValueOutput",
         // That variant-pack slot holds a host pointer, not a device one.
         ConvBiasGraphSpec{.outPassByValue = true}},
    }),
    [](const ::testing::TestParamInfo<GraphCase>& info) { return info.param.name; });

// Refusals driven by the graph's node structure rather than by a convolution attribute.
// Their own suite because gtest rejects a suite that mixes TEST with the fixture the
// parameterized refusals above are built on, and the whole suite fails when it does.
TEST(TestConvBiasGraphStructureRefusal, RefusesASingleNodeConvolution)
{
    // The sibling engine's graph: a conv on its own, with no pointwise to fuse into. A
    // node-type check that forgot to count nodes would admit it.
    const GraphFixture fixture(buildConvFwdGraph());

    EXPECT_FALSE(matches(fixture.context()));
}

TEST(TestConvBiasGraphStructureRefusal, RefusesASingleNodePointwiseAdd)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_FALSE(matches(fixture.context()));
}

TEST(TestConvBiasGraphStructureRefusal, RefusesTwoPointwiseNodes)
{
    // Two nodes, and the second does consume the first's virtual output -- but the first is
    // not a convolution. Node count alone is not the gate either.
    const GraphFixture fixture(buildTwoNodePointwiseGraph());

    EXPECT_FALSE(matches(fixture.context()));
}

// ---------------------------------------------------------------------------
// Kernel-scoped matcher
// ---------------------------------------------------------------------------

TEST(TestConvBiasKernelMatcher, AcceptsTheCandidateWhoseDtypeIsTheGraphs)
{
    const GraphFixture fixture(buildConvBiasGraph());
    const auto bound = matchesGraph(CONV_BIAS, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_TRUE(
        matchesKernel(CONV_BIAS, fixture.context(), makeConvBiasKernel(256, "FLOAT"), *bound));
}

TEST(TestConvBiasKernelMatcher, RefusesACandidateBakedForAnotherDtype)
{
    const GraphFixture fixture(buildConvBiasGraph());
    const auto bound = matchesGraph(CONV_BIAS, fixture.context());
    ASSERT_TRUE(bound.has_value());

    // HKP_CONV_BIAS_TYPE changes the declared types of four of the kernel's parameters, so
    // a HALF binary reached by a FLOAT graph returns numbers rather than failing.
    EXPECT_FALSE(
        matchesKernel(CONV_BIAS, fixture.context(), makeConvBiasKernel(256, "HALF"), *bound));
}

TEST(TestConvBiasKernelMatcher, AdmitsEveryBlockSizeForOneGraph)
{
    const GraphFixture fixture(buildConvBiasGraph());
    const auto bound = matchesGraph(CONV_BIAS, fixture.context());
    ASSERT_TRUE(bound.has_value());

    // Block size is a RANKING axis, not an applicability one: the kernel guards its own
    // bounds, so every one of these is correct for this graph and the scorer is what
    // separates them. A kernel matcher that gated on block size would leave the scorer
    // nothing to choose between, and the score test below would pass anyway.
    for(const auto blockSize : {64, 256, 1024})
    {
        EXPECT_TRUE(matchesKernel(
            CONV_BIAS, fixture.context(), makeConvBiasKernel(blockSize, "FLOAT"), *bound))
            << "block size " << blockSize;
    }
}

// ---------------------------------------------------------------------------
// Score
// ---------------------------------------------------------------------------

TEST(TestConvBiasScore, VariesWithTheDescriptorItIsGiven)
{
    // One output element: a 64-thread block wastes 63 lanes and a 1024-thread block wastes
    // 1023, so occupancy separates them. A constant score makes selection order arbitrary,
    // and nothing else catches that until a performance pass on real hardware.
    const GraphFixture fixture(
        buildConvBiasGraph({.xDims = {1, 1, 1, 1}, .wDims = {1, 1, 1, 1}}));
    const auto bound = matchesGraph(CONV_BIAS, fixture.context());
    ASSERT_TRUE(bound.has_value());

    const auto small
        = scoreKernel(CONV_BIAS, fixture.context(), makeConvBiasKernel(64, "FLOAT"), *bound);
    const auto large
        = scoreKernel(CONV_BIAS, fixture.context(), makeConvBiasKernel(1024, "FLOAT"), *bound);

    EXPECT_NE(small, large);
}

TEST(TestConvBiasScore, PrefersTheSmallBlockOnATinyGraph)
{
    const GraphFixture fixture(
        buildConvBiasGraph({.xDims = {1, 1, 1, 1}, .wDims = {1, 1, 1, 1}}));
    const auto bound = matchesGraph(CONV_BIAS, fixture.context());
    ASSERT_TRUE(bound.has_value());

    // The direction, not just the difference: higher wins, and at one element the 64-thread
    // block is the one that wastes least.
    EXPECT_GT(scoreKernel(CONV_BIAS, fixture.context(), makeConvBiasKernel(64, "FLOAT"), *bound),
              scoreKernel(CONV_BIAS, fixture.context(), makeConvBiasKernel(1024, "FLOAT"), *bound));
}

TEST(TestConvBiasScore, PrefersTheLargeBlockWhenBothDivideTheElementCount)
{
    // 4096 elements: every candidate reaches occupancy 1, so the tie-break decides, and it
    // prefers the candidate launching fewer blocks.
    const GraphFixture fixture(
        buildConvBiasGraph({.xDims = {1, 1, 64, 64}, .wDims = {1, 1, 1, 1}}));
    const auto bound = matchesGraph(CONV_BIAS, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_GT(scoreKernel(CONV_BIAS, fixture.context(), makeConvBiasKernel(1024, "FLOAT"), *bound),
              scoreKernel(CONV_BIAS, fixture.context(), makeConvBiasKernel(64, "FLOAT"), *bound));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
