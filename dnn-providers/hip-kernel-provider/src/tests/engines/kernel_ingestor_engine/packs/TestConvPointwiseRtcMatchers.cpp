// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestConvPointwiseRtcMatchers.cpp
 * @brief Matcher acceptance/refusal shapes for hipkernel:ConvPointwiseRtc.
 *
 * The engine serves a two-node fusion, so its matcher has two classes of obligation no
 * single-node pack has: the node ORDER and the uid EDGE between the two nodes. Both are
 * exercised below, because each passes on its own for a graph the other rejects.
 *
 * Every refusal here is one the kernel would otherwise compute a WRONG NUMBER for rather
 * than fail on -- a flipped filter, a y extent that disagrees with the padding, an
 * activation with no compiled specialization. That is the reason they are unit tests and
 * not left to the device suite: the device suite compares against a reference, and for
 * conv_mode the reference makes the same mistake (ConvolutionFwdPlanBuilder::isApplicable
 * never tests conv_mode, so the CPU reference executes a CONVOLUTION graph as
 * cross-correlation and agrees with a wrong kernel).
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
// Only the two types this file NAMES. The provider builds its tests under clang-tidy with
// misc-unused-using-decls as an ERROR, so a declaration for a type the cases reach only
// through `auto` fails the build.
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// This pack's symbol contract, as its descriptors and native file spell it. Declared
/// here rather than in PointwiseTestGraphs.hpp because only this file needs it, and the
/// shared header is read by three other packs' suites.
constexpr PackSymbols CONV_POINTWISE_RTC{"hipkernel:ConvPointwiseRtc",
                                         "hipkernel.conv_pointwise_rtc.graph_match",
                                         "",
                                         "hipkernel.conv_pointwise_rtc.kernel_match",
                                         "hipkernel.conv_pointwise_rtc.score",
                                         "hipkernel.conv_pointwise_rtc.dispatch",
                                         "conv_pointwise_rtc.x.uid",
                                         "conv_pointwise_rtc.w.uid",
                                         "conv_pointwise_rtc.y.uid"};

/// Tensor uids the builder below uses. The intermediate is the convolution's own output
/// and is virtual; it is the fused edge and gets no buffer.
constexpr int64_t X_UID = 3;
constexpr int64_t W_UID = 2;
constexpr int64_t INTERMEDIATE_UID = 0;
constexpr int64_t Y_UID = 1;

/// A fixed device, constructed BY VALUE rather than queried from the host. An arch-gated
/// matcher test that instead calls hipGetDeviceProperties() is vacuous everywhere except
/// whatever arch happens to be running CI (TestAsmSdpaForwardMatchers.cpp:27-33) -- it
/// would silently stop testing the one thing an arch-gated matcher exists to gate the
/// moment CI's hardware changes.
DeviceProperties fixedDeviceProperties()
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx90a";
    properties.warpSize = 64;
    return properties;
}

/// Knobs the builder exposes: exactly the fields this pack's matcher gates, so a refusal
/// case can move one of them and leave everything else valid.
struct FusionSpec
{
    data_objects::PointwiseMode operation = data_objects::PointwiseMode::RELU_FWD;
    data_objects::ConvMode convMode = data_objects::ConvMode::CROSS_CORRELATION;
    data_objects::DataType dataType = data_objects::DataType::FLOAT;
    /// The virtual intermediate's own declared type, independent of the rest: eliding its
    /// store is a no-op only at FLOAT.
    std::optional<data_objects::DataType> intermediateDataType;
    std::vector<int64_t> xDims = {1, 16, 16, 8};
    std::vector<int64_t> wDims = {1, 16, 3, 3};
    std::vector<int64_t> yDims = {1, 1, 16, 8};
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    std::vector<int64_t> prePadding = {1, 1};
    std::vector<int64_t> postPadding = {1, 1};
    std::optional<std::vector<int64_t>> xStrides;
    std::optional<float> reluLowerClip;
    std::optional<float> reluUpperClip;
    std::optional<float> reluLowerClipSlope;
    /// Adds a second pointwise operand, making the epilogue binary.
    bool binaryPointwise = false;
    /// Makes the convolution's output tensor real, so nothing is fused away.
    bool intermediateIsReal = false;
    /// Breaks the uid edge: the pointwise consumes x rather than the convolution's output.
    bool breakFusedEdge = false;
    /// Emits the Pointwise node first, so the node-type multiset is unchanged but the
    /// order is not.
    bool swapNodeOrder = false;
    /// Drops the Pointwise node entirely, leaving a bare single-node convolution.
    bool convolutionOnly = false;
    data_objects::DataType nodeComputeDataType = data_objects::DataType::FLOAT;
    bool overrideShapeEnabled = false;
};

/// Builds the ConvolutionFwd -> Pointwise fusion this engine serves, parameterized on
/// everything its matcher gates. The defaults are the shape the authoring round proved:
/// x [1,16,16,8], w [1,16,3,3], 3x3 same-padding, relu with no clips, FLOAT throughout.
flatbuffers::FlatBufferBuilder buildFusionGraph(const FusionSpec& spec = {})
{
    flatbuffers::FlatBufferBuilder builder;

    const auto xStrides = spec.xStrides.value_or(packedRowMajorStrides(spec.xDims));
    const auto wStrides = packedRowMajorStrides(spec.wDims);
    const auto yStrides = packedRowMajorStrides(spec.yDims);
    const auto intermediateDataType = spec.intermediateDataType.value_or(spec.dataType);

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 INTERMEDIATE_UID,
                                                                 "ConvolutionFprop_0::Y",
                                                                 intermediateDataType,
                                                                 &yStrides,
                                                                 &spec.yDims,
                                                                 !spec.intermediateIsReal));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, Y_UID, "Pointwise_1::OUT_0", spec.dataType, &yStrides, &spec.yDims, false));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, W_UID, "w", spec.dataType, &wStrides, &spec.wDims, false));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, X_UID, "x", spec.dataType, &xStrides, &spec.xDims, false));

    auto convAttributes = data_objects::CreateConvolutionFwdAttributesDirect(builder,
                                                                             X_UID,
                                                                             W_UID,
                                                                             INTERMEDIATE_UID,
                                                                             &spec.prePadding,
                                                                             &spec.postPadding,
                                                                             &spec.stride,
                                                                             &spec.dilation,
                                                                             spec.convMode);

    data_objects::PointwiseAttributesBuilder pointwiseBuilder(builder);
    pointwiseBuilder.add_operation(spec.operation);
    pointwiseBuilder.add_in_0_tensor_uid(spec.breakFusedEdge ? X_UID : INTERMEDIATE_UID);
    if(spec.binaryPointwise)
    {
        pointwiseBuilder.add_in_1_tensor_uid(X_UID);
    }
    if(spec.reluLowerClip.has_value())
    {
        pointwiseBuilder.add_relu_lower_clip(*spec.reluLowerClip);
    }
    if(spec.reluUpperClip.has_value())
    {
        pointwiseBuilder.add_relu_upper_clip(*spec.reluUpperClip);
    }
    if(spec.reluLowerClipSlope.has_value())
    {
        pointwiseBuilder.add_relu_lower_clip_slope(*spec.reluLowerClipSlope);
    }
    pointwiseBuilder.add_out_0_tensor_uid(Y_UID);
    auto pointwiseAttributes = pointwiseBuilder.Finish();

    const auto convNode = data_objects::CreateNodeDirect(
        builder,
        "ConvolutionFprop_0",
        spec.nodeComputeDataType,
        data_objects::NodeAttributes::ConvolutionFwdAttributes,
        convAttributes.Union());
    const auto pointwiseNode
        = data_objects::CreateNodeDirect(builder,
                                         "Pointwise_1",
                                         spec.nodeComputeDataType,
                                         data_objects::NodeAttributes::PointwiseAttributes,
                                         pointwiseAttributes.Union());

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    if(spec.convolutionOnly)
    {
        nodes.push_back(convNode);
    }
    else if(spec.swapNodeOrder)
    {
        nodes.push_back(pointwiseNode);
        nodes.push_back(convNode);
    }
    else
    {
        nodes.push_back(convNode);
        nodes.push_back(pointwiseNode);
    }

    auto name = builder.CreateString("conv_fwd_pointwise_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_compute_data_type(data_objects::DataType::FLOAT);
    graphBuilder.add_intermediate_data_type(data_objects::DataType::FLOAT);
    graphBuilder.add_io_data_type(spec.dataType);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    graphBuilder.add_is_override_shape_enabled(spec.overrideShapeEnabled);
    builder.Finish(graphBuilder.Finish());

    return builder;
}

/// A candidate of this pack, whose KMD carries three fields.
KernelDefinition makeFusionKernel(int64_t blockSize,
                                  const std::string& activation = "RELU_FWD",
                                  const std::string& ioDtype = "FLOAT")
{
    auto kernel = makeKernel(blockSize, ioDtype, "ConvFwdPointwiseFused");
    kernel.source.entryPoint = "hkpConvFwdPointwiseFused";
    kernel.metadata = {{"block_size", blockSize}, {"io_dtype", ioDtype}, {"activation", activation}};
    return kernel;
}

bool serves(const FusionSpec& spec)
{
    const GraphFixture fixture(buildFusionGraph(spec), fixedDeviceProperties());
    return matchesGraph(CONV_POINTWISE_RTC, fixture.context()).has_value();
}

// ---------------------------------------------------------------------------
// Acceptance
// ---------------------------------------------------------------------------

TEST(ConvPointwiseRtcMatchers, AcceptsAGraphThisEngineServes)
{
    const GraphFixture fixture(buildFusionGraph(), fixedDeviceProperties());
    const auto bound = matchesGraph(CONV_POINTWISE_RTC, fixture.context());

    ASSERT_TRUE(bound.has_value());
    // The bound tokens ARE the launch contract: prepare() reads them back rather than
    // re-deriving the uids, so a matcher that admits the graph and binds the wrong tensor
    // is a wrong answer with no diagnostic.
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, CONV_POINTWISE_RTC.inputAToken),
              std::optional<int64_t>{X_UID});
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, CONV_POINTWISE_RTC.inputBToken),
              std::optional<int64_t>{W_UID});
    // The POINTWISE output, not the convolution's virtual one. Binding the intermediate
    // here would make launch() resolve a uid that has no device buffer.
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, CONV_POINTWISE_RTC.outputToken),
              std::optional<int64_t>{Y_UID});
}

TEST(ConvPointwiseRtcMatchers, AcceptsTheThreeMeasuredActivations)
{
    for(const auto operation : {data_objects::PointwiseMode::RELU_FWD,
                                data_objects::PointwiseMode::ABS,
                                data_objects::PointwiseMode::NEG})
    {
        FusionSpec spec;
        spec.operation = operation;
        EXPECT_TRUE(serves(spec)) << data_objects::EnumNamePointwiseMode(operation);
    }
}

/// Layout is not a schema field for this kernel -- it is the dims/strides relationship,
/// and every access goes through a stride argument. An NHWC x and a row pitch wider than
/// the width extent are therefore acceptances, not refusals, and this is the test that
/// stops a later "tidy-up" from adding ConvNative's packed-strides check here.
TEST(ConvPointwiseRtcMatchers, AcceptsNonPackedAndChannelLastInputStrides)
{
    FusionSpec nhwc;
    // [N, C, H, W] extents with channel-fastest strides.
    nhwc.xStrides = std::vector<int64_t>{2048, 1, 128, 16};
    EXPECT_TRUE(serves(nhwc));

    FusionSpec gapped;
    // A row pitch of 10 for a width of 8: elementSpace exceeds elementCount.
    gapped.xStrides = std::vector<int64_t>{2560, 160, 10, 1};
    EXPECT_TRUE(serves(gapped));
}

TEST(ConvPointwiseRtcMatchers, AcceptsGroupedStridedDilatedAndAsymmetricallyPaddedConvolutions)
{
    FusionSpec grouped;
    // G = 2: x has 16 channels, w has 8 per group and 4 output channels overall.
    grouped.xDims = {1, 16, 16, 8};
    grouped.wDims = {4, 8, 3, 3};
    grouped.yDims = {1, 4, 16, 8};
    EXPECT_TRUE(serves(grouped));

    FusionSpec strided;
    strided.stride = {2, 3};
    strided.dilation = {2, 1};
    strided.prePadding = {2, 1};
    strided.postPadding = {1, 2};
    // ((16 + 2 + 1 - ((2 * 2) + 1)) / 2) + 1 = 8, ((8 + 1 + 2 - 3) / 3) + 1 = 3.
    strided.yDims = {1, 1, 8, 3};
    EXPECT_TRUE(serves(strided));
}

// ---------------------------------------------------------------------------
// Refusal
// ---------------------------------------------------------------------------

TEST(ConvPointwiseRtcMatchers, DeclinesAGraphOutsideItsApplicability)
{
    // A flipped filter. This is the refusal with the least margin for error in the whole
    // matcher: the kernel computes CONVOLUTION as cross-correlation and the CPU reference
    // does the same, so nothing downstream can catch an over-accepting gate here.
    FusionSpec flipped;
    flipped.convMode = data_objects::ConvMode::CONVOLUTION;
    EXPECT_FALSE(serves(flipped));

    // UNSET is the zero enumerator FlatBuffers hands back for an absent conv_mode, and
    // must not fall through to the one mode that happens to be implemented.
    FusionSpec unsetMode;
    unsetMode.convMode = data_objects::ConvMode::UNSET;
    EXPECT_FALSE(serves(unsetMode));
}

TEST(ConvPointwiseRtcMatchers, DeclinesAnUnmeasuredOrAbsentActivation)
{
    // Compiled by the kernel but declined by hipDNN's CPU reference, so it is an
    // unmeasured bucket rather than a pass. An unmeasured enumerator is not a claimed one.
    FusionSpec identity;
    identity.operation = data_objects::PointwiseMode::IDENTITY;
    EXPECT_FALSE(serves(identity));

    // The zero enumerator for an absent operation.
    FusionSpec unset;
    unset.operation = data_objects::PointwiseMode::UNSET;
    EXPECT_FALSE(serves(unset));

    FusionSpec sigmoid;
    sigmoid.operation = data_objects::PointwiseMode::SIGMOID_FWD;
    EXPECT_FALSE(serves(sigmoid));
}

/// The whole reason this fusion is one launch: the convolution's output has no caller
/// buffer, so the accumulator never leaves a register. A real intermediate would need a
/// store this kernel does not perform, and the graph's output would be left unwritten.
TEST(ConvPointwiseRtcMatchers, DeclinesAFusionWhoseIntermediateIsMaterialized)
{
    FusionSpec real;
    real.intermediateIsReal = true;
    EXPECT_FALSE(serves(real));
}

/// Eliding the intermediate store is a no-op only when the intermediate is declared at
/// the accumulator's own type. At any other declared type the reference performs a
/// rounding this fusion skips, and the two disagree by construction.
TEST(ConvPointwiseRtcMatchers, DeclinesANonFloatIntermediate)
{
    FusionSpec halfIntermediate;
    halfIntermediate.intermediateDataType = data_objects::DataType::HALF;
    EXPECT_FALSE(serves(halfIntermediate));
}

/// Two nodes of the right types are not a fusion. Each of these passes a check the other
/// fails, which is why the matcher makes both and why both are tested.
TEST(ConvPointwiseRtcMatchers, DeclinesTopologiesThatOnlyLookLikeTheFusion)
{
    FusionSpec disconnected;
    disconnected.breakFusedEdge = true;
    EXPECT_FALSE(serves(disconnected)) << "pointwise consuming x rather than the conv output";

    FusionSpec swapped;
    swapped.swapNodeOrder = true;
    EXPECT_FALSE(serves(swapped)) << "same node-type multiset, wrong order";

    FusionSpec single;
    single.convolutionOnly = true;
    EXPECT_FALSE(serves(single)) << "a bare ConvolutionFwd is hipkernel:ConvFwd's graph";
}

TEST(ConvPointwiseRtcMatchers, DeclinesABinaryPointwiseEpilogue)
{
    FusionSpec binary;
    binary.binaryPointwise = true;
    EXPECT_FALSE(serves(binary));
}

/// post_padding enters no index map; it enters only the output-extent formula. Without
/// this equality a graph whose y dims disagree with its padding is accepted and computed
/// over the wrong extent, which is a wrong answer rather than an out-of-bounds write.
TEST(ConvPointwiseRtcMatchers, DeclinesOutputExtentsThatDisagreeWithThePadding)
{
    FusionSpec shrunk;
    // Everything else is the accepted default; only post_padding moves, and the y dims
    // that were correct for {1,1} are not correct for {0,0}.
    shrunk.postPadding = {0, 0};
    EXPECT_FALSE(serves(shrunk));

    FusionSpec tooSmall;
    tooSmall.xDims = {1, 16, 2, 2};
    tooSmall.prePadding = {0, 0};
    tooSmall.postPadding = {0, 0};
    // The 3x3 filter does not fit in a 2x2 unpadded input at all; the extent formula has
    // no non-negative answer and truncating division must not manufacture one.
    tooSmall.yDims = {1, 1, 1, 1};
    EXPECT_FALSE(serves(tooSmall));
}

TEST(ConvPointwiseRtcMatchers, DeclinesChannelCountsThatDoNotDivide)
{
    FusionSpec ragged;
    // 16 input channels against 5 filter channels: no integer group count.
    ragged.wDims = {1, 5, 3, 3};
    EXPECT_FALSE(serves(ragged));

    FusionSpec unevenOutput;
    // G = 2 from 16/8, but 3 output channels do not split evenly across two groups.
    unevenOutput.wDims = {3, 8, 3, 3};
    unevenOutput.yDims = {1, 3, 16, 8};
    EXPECT_FALSE(serves(unevenOutput));
}

/// A clip belongs to RELU_FWD alone. The kernel would simply ignore it for ABS, which
/// sounds harmless and is not: the reference throws on that combination, so an accepted
/// graph would have no oracle at all.
TEST(ConvPointwiseRtcMatchers, DeclinesReluClipsOnANonReluActivation)
{
    FusionSpec clippedAbs;
    clippedAbs.operation = data_objects::PointwiseMode::ABS;
    clippedAbs.reluUpperClip = 6.0f;
    EXPECT_FALSE(serves(clippedAbs));

    // ...and admits the same clips on RELU_FWD, so the refusal above is about the pairing
    // rather than about clips being unsupported.
    FusionSpec clippedRelu;
    clippedRelu.reluLowerClip = 0.0f;
    clippedRelu.reluUpperClip = 6.0f;
    clippedRelu.reluLowerClipSlope = 0.1f;
    EXPECT_TRUE(serves(clippedRelu));
}

TEST(ConvPointwiseRtcMatchers, DeclinesRanksAndComputeTypesOutsideTheEnvelope)
{
    FusionSpec rank3;
    rank3.xDims = {16, 16, 8};
    rank3.wDims = {1, 16, 3};
    rank3.yDims = {1, 16, 8};
    rank3.stride = {1};
    rank3.dilation = {1};
    rank3.prePadding = {1};
    rank3.postPadding = {1};
    EXPECT_FALSE(serves(rank3));

    FusionSpec halfCompute;
    halfCompute.nodeComputeDataType = data_objects::DataType::HALF;
    EXPECT_FALSE(serves(halfCompute));

    FusionSpec overridable;
    overridable.overrideShapeEnabled = true;
    EXPECT_FALSE(serves(overridable));
}

// ---------------------------------------------------------------------------
// Candidate matching and scoring
// ---------------------------------------------------------------------------

/// `graph_match` can only say "one of the three admitted activations". Deciding WHICH is
/// per-candidate, because the activation is baked by HKP_ACTIVATION -- so without this
/// comparison a relu graph reaches the ABS binary and returns wrong numbers.
TEST(ConvPointwiseRtcMatchers, CandidateMatchingSeparatesTheBakedActivationAndDtype)
{
    const GraphFixture fixture(buildFusionGraph(), fixedDeviceProperties());
    const auto context = fixture.context();

    EXPECT_TRUE(matchesKernel(CONV_POINTWISE_RTC, context, makeFusionKernel(256, "RELU_FWD")));
    EXPECT_FALSE(matchesKernel(CONV_POINTWISE_RTC, context, makeFusionKernel(256, "ABS")));
    EXPECT_FALSE(matchesKernel(CONV_POINTWISE_RTC, context, makeFusionKernel(256, "NEG")));
    EXPECT_FALSE(
        matchesKernel(CONV_POINTWISE_RTC, context, makeFusionKernel(256, "RELU_FWD", "HALF")));

    FusionSpec absSpec;
    absSpec.operation = data_objects::PointwiseMode::ABS;
    const GraphFixture absFixture(buildFusionGraph(absSpec), fixedDeviceProperties());
    EXPECT_TRUE(
        matchesKernel(CONV_POINTWISE_RTC, absFixture.context(), makeFusionKernel(256, "ABS")));
    EXPECT_FALSE(
        matchesKernel(CONV_POINTWISE_RTC, absFixture.context(), makeFusionKernel(256, "RELU_FWD")));
}

TEST(ConvPointwiseRtcMatchers, ScoreReadsAtLeastOneDescriptorField)
{
    // A 16x8 output plane is 128 positions: a 64-thread block wastes nothing and a
    // 256-thread block wastes 128 lanes, so the two must not score alike.
    const GraphFixture fixture(buildFusionGraph(), fixedDeviceProperties());
    const auto context = fixture.context();

    const auto small = scoreKernel(CONV_POINTWISE_RTC, context, makeFusionKernel(64));
    const auto large = scoreKernel(CONV_POINTWISE_RTC, context, makeFusionKernel(256));
    EXPECT_NE(small, large);
    EXPECT_GT(small, large);
}

/// The tie-break, and the property that makes it safe: one wasted thread is scaled by
/// 2048 and the block size is at most 1024, so waste strictly dominates and the block-size
/// term decides only candidates whose waste is equal.
TEST(ConvPointwiseRtcMatchers, ScoreBreaksTiesTowardTheLargerBlock)
{
    FusionSpec exactPlane;
    // A 16x16 output plane is 256 positions: blocks of 64 and 256 both waste nothing.
    exactPlane.yDims = {1, 1, 16, 16};
    exactPlane.xDims = {1, 16, 16, 16};
    const GraphFixture fixture(buildFusionGraph(exactPlane), fixedDeviceProperties());
    const auto context = fixture.context();

    const auto block64 = scoreKernel(CONV_POINTWISE_RTC, context, makeFusionKernel(64));
    const auto block256 = scoreKernel(CONV_POINTWISE_RTC, context, makeFusionKernel(256));
    const auto block1024 = scoreKernel(CONV_POINTWISE_RTC, context, makeFusionKernel(1024));

    EXPECT_GT(block256, block64) << "equal waste, so the larger block wins";
    EXPECT_GT(block256, block1024) << "1024 wastes 768 lanes here and must not win on size";
}

TEST(ConvPointwiseRtcMatchersPlaceholder, FixtureConstructsADeviceByValue)
{
    const auto properties = fixedDeviceProperties();
    EXPECT_FALSE(properties.gcnArchName.empty());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
