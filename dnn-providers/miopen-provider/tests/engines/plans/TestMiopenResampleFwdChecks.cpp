// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Applicability rules of resample_fwd_applicability::resolve(), one test per rule, each matching
// its decline message, plus the boundaries and transformations the provider computes itself. None
// of these tests needs a GPU: resolve() only reads the graph.

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "engines/plans/MiopenResampleFwdChecks.hpp"

using namespace miopen_plugin;

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::PaddingMode;
using hipdnn_flatbuffers_sdk::data_objects::ResampleMode;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_test_sdk::utilities::createResampleFwdGraph;
using hipdnn_test_sdk::utilities::ResampleFwdGraphSpec;
using resample_fwd_applicability::ResampleFwdConfig;

namespace
{

constexpr int64_t INT_MAX_VALUE = std::numeric_limits<int>::max();

// 2x3x6x6 -> 2x3x3x3 with a 3x3 window, stride 2 and padding 1, so the padding rules have
// something to bite on.
ResampleFwdGraphSpec defaultSpec(ResampleMode mode = ResampleMode::MAXPOOL,
                                 std::optional<DataType> indexType = std::nullopt)
{
    return ResampleFwdGraphSpec::packed(
        mode, {2, 3, 6, 6}, {3, 3}, {2, 2}, {1, 1}, DataType::FLOAT, indexType);
}

ResampleFwdConfig resolveSpec(const ResampleFwdGraphSpec& spec)
{
    auto fbb = createResampleFwdGraph(spec);
    const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());
    return resample_fwd_applicability::resolve(graph);
}

void expectAccepted(const ResampleFwdGraphSpec& spec)
{
    auto fbb = createResampleFwdGraph(spec);
    const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());
    ASSERT_TRUE(graph.isValid());

    EXPECT_TRUE(resample_fwd_applicability::isSupported(graph));
    try
    {
        resample_fwd_applicability::resolve(graph);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        ADD_FAILURE() << "resolve() declined: " << e.what();
    }
}

// Declined with BAD_PARAM, for the reason the message fragment names. The fragment guards against
// a decline test passing because some other rule happened to fire first.
void expectDeclined(const flatbuffers::FlatBufferBuilder& fbb, const std::string& reasonFragment)
{
    const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());
    ASSERT_TRUE(graph.isValid());

    EXPECT_FALSE(resample_fwd_applicability::isSupported(graph));
    try
    {
        resample_fwd_applicability::resolve(graph);
        ADD_FAILURE() << "resolve() accepted a graph it should decline (" << reasonFragment << ")";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
        EXPECT_THAT(e.what(), ::testing::HasSubstr(reasonFragment));
    }
}

void expectDeclined(const ResampleFwdGraphSpec& spec, const std::string& reasonFragment)
{
    expectDeclined(createResampleFwdGraph(spec), reasonFragment);
}

// Max pooling with a 1x1 window and stride, so Y matches X and only the X spatial size matters.
ResampleFwdGraphSpec identityWindowSpec(const std::vector<int64_t>& xDims, DataType indexType)
{
    const std::vector<int64_t> ones(xDims.size() - 2, 1);
    const std::vector<int64_t> zeros(xDims.size() - 2, 0);
    return ResampleFwdGraphSpec::packed(
        ResampleMode::MAXPOOL, xDims, ones, ones, zeros, DataType::FLOAT, indexType);
}

} // namespace

// ============================================================================
// Graph-level rules.
// ============================================================================

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForMultiNodeGraph)
{
    auto spec = defaultSpec();
    spec.nodeCount = 2;
    expectDeclined(spec, "single-node");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForOtherNodeType)
{
    expectDeclined(hipdnn_test_sdk::utilities::createValidResampleBwdGraph(),
                   "unsupported node attributes");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForNonFloatComputeType)
{
    auto spec = defaultSpec();
    spec.computeDataType = DataType::HALF;
    expectDeclined(spec, "compute_data_type");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForModeNotSet)
{
    auto spec = defaultSpec();
    spec.mode = ResampleMode::NOT_SET;
    expectDeclined(spec, "unsupported resample mode");
}

// ============================================================================
// X and Y tensor rules. Every per-tensor rule is checked on X; Y goes through the same check.
// ============================================================================

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForYAliasingX)
{
    auto spec = defaultSpec();
    spec.y.uid = spec.x.uid;
    expectDeclined(spec, "in-place");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForUnresolvableX)
{
    auto spec = defaultSpec();
    spec.x.inGraph = false;
    expectDeclined(spec, "X tensor (uid 1) does not resolve");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForUnresolvableY)
{
    auto spec = defaultSpec();
    spec.y.inGraph = false;
    expectDeclined(spec, "Y tensor (uid 2) does not resolve");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForVirtualX)
{
    auto spec = defaultSpec();
    spec.x.isVirtual = true;
    expectDeclined(spec, "X tensor must not be virtual");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForPassByValueX)
{
    auto spec = defaultSpec();
    spec.x.isRuntimePassByValue = true;
    expectDeclined(spec, "X tensor must not be pass-by-value");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForRaggedX)
{
    auto spec = defaultSpec();
    spec.x.raggedOffsetTensorUid = 7;
    expectDeclined(spec, "X tensor must not be ragged");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForMissingXStrides)
{
    auto spec = defaultSpec();
    spec.x.strides = std::nullopt;
    expectDeclined(spec, "X tensor must have one stride per dimension");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForZeroDimInX)
{
    auto spec = defaultSpec();
    spec.x.dims[0] = 0;
    expectDeclined(spec, "X tensor has a non-positive dimension");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForXElementCountAboveInt32Max)
{
    // 46341^2 > INT32_MAX; the stride collapses Y to a single element.
    expectDeclined(ResampleFwdGraphSpec::packed(
                       ResampleMode::MAXPOOL, {1, 1, 46341, 46341}, {1, 1}, {46341, 46341}, {0, 0}),
                   "X tensor element count exceeds INT32_MAX");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForThreeDimensionalTensors)
{
    expectDeclined(ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL, {2, 3, 6}, {2}, {2}, {0}),
                   "X must be 4D or 5D");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForRankMismatch)
{
    auto spec = defaultSpec();
    spec.y.dims = {2, 3, 3, 3, 1};
    spec.y.strides = std::vector<int64_t>{27, 9, 3, 1, 1};
    expectDeclined(spec, "X and Y must have the same rank");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForUnsupportedDataType)
{
    auto spec = defaultSpec();
    spec.x.dataType = DataType::DOUBLE;
    spec.y.dataType = DataType::DOUBLE;
    expectDeclined(spec, "X data type must be FLOAT, HALF or BFLOAT16");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForMismatchedYDataType)
{
    auto spec = defaultSpec();
    spec.y.dataType = DataType::HALF;
    expectDeclined(spec, "Y data type must match X");
}

// ============================================================================
// Pooling parameter rules.
// ============================================================================

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForWindowLengthMismatch)
{
    auto spec = defaultSpec();
    spec.window = std::vector<int64_t>{3, 3, 3};
    expectDeclined(spec, "window must have one entry per spatial dimension");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForZeroStride)
{
    auto spec = defaultSpec();
    spec.stride = std::vector<int64_t>{0, 2};
    expectDeclined(spec, "stride values must be in [1, INT_MAX]");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForWindowAboveIntMax)
{
    auto spec = defaultSpec();
    spec.window = std::vector<int64_t>{INT_MAX_VALUE + 1, 3};
    expectDeclined(spec, "window values must be in [1, INT_MAX]");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForAsymmetricPadding)
{
    auto spec = defaultSpec();
    spec.postPadding = std::vector<int64_t>{0, 0};
    expectDeclined(spec, "asymmetric padding");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForNegativePrePadding)
{
    auto spec = defaultSpec();
    spec.setPadding({-1, -1});
    expectDeclined(spec, "pre_padding values must be in [0, INT_MAX]");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForNegativePostPadding)
{
    auto spec = defaultSpec();
    spec.postPadding = std::vector<int64_t>{-1, -1};
    expectDeclined(spec, "post_padding values must be in [0, INT_MAX]");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForPaddingEqualToWindow)
{
    expectDeclined(
        ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL, {2, 3, 6, 6}, {2, 2}, {2, 2}, {0, 2}),
        "padding must be smaller than the window");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForPaddingOneBelowWindow)
{
    expectAccepted(
        ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL, {1, 1, 4, 4}, {3, 3}, {1, 1}, {2, 2}));
}

// MIOpen pads max pooling with -inf and average pooling with zeros. Any other padding_mode is only
// equivalent when no position is padded.
TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForMismatchedPaddingModeWithPadding)
{
    auto spec = defaultSpec(ResampleMode::MAXPOOL);
    spec.paddingMode = PaddingMode::ZERO_PAD;
    expectDeclined(spec, "requires all padding to be zero");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForMismatchedPaddingModeWithoutPadding)
{
    auto spec = ResampleFwdGraphSpec::packed(
        ResampleMode::AVGPOOL_INCLUDE_PADDING, {2, 3, 6, 6}, {3, 3}, {2, 2}, {0, 0});
    spec.paddingMode = PaddingMode::NEG_INF_PAD;
    expectAccepted(spec);
}

// PADDING_NOT_SET is the frontend's default, so graphs built without a padding mode take this
// rule; MIOpen still pads max pooling with -inf and average pooling with zeros.
TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForPaddingModeNotSetWithPadding)
{
    for(const auto mode : {ResampleMode::MAXPOOL,
                           ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                           ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        SCOPED_TRACE(hipdnn_flatbuffers_sdk::data_objects::EnumNameResampleMode(mode));
        auto spec = defaultSpec(mode);
        spec.paddingMode = PaddingMode::PADDING_NOT_SET;
        expectDeclined(spec, "requires all padding to be zero");
    }
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForAveragePoolingWithNegInfPaddingAndPadding)
{
    for(const auto mode :
        {ResampleMode::AVGPOOL_EXCLUDE_PADDING, ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        SCOPED_TRACE(hipdnn_flatbuffers_sdk::data_objects::EnumNameResampleMode(mode));
        auto spec = defaultSpec(mode);
        spec.paddingMode = PaddingMode::NEG_INF_PAD;
        expectDeclined(spec, "requires all padding to be zero");
    }
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForPaddingModeNotSetWithoutPadding)
{
    auto spec
        = ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL, {2, 3, 6, 6}, {3, 3}, {2, 2}, {0, 0});
    spec.paddingMode = PaddingMode::PADDING_NOT_SET;
    expectAccepted(spec);
}

// ============================================================================
// Shape rules.
// ============================================================================

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForBatchMismatch)
{
    auto spec = defaultSpec();
    spec.y.dims = {1, 3, 3, 3};
    spec.y.strides = std::vector<int64_t>{27, 9, 3, 1};
    expectDeclined(spec, "same batch and channel dimensions");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForChannelMismatch)
{
    auto spec = defaultSpec();
    spec.y.dims = {2, 4, 3, 3};
    spec.y.strides = std::vector<int64_t>{36, 9, 3, 1};
    expectDeclined(spec, "same batch and channel dimensions");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForWindowLargerThanPaddedInput)
{
    // 2 + 2 * 1 < 5 in both spatial dimensions.
    auto spec
        = ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL, {1, 1, 4, 4}, {5, 5}, {1, 1}, {1, 1});
    spec.x.dims = {1, 1, 2, 2};
    spec.x.strides = std::vector<int64_t>{4, 4, 2, 1};
    spec.y.dims = {1, 1, 1, 1};
    spec.y.strides = std::vector<int64_t>{1, 1, 1, 1};
    expectDeclined(spec, "window must fit inside the padded input");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForWindowCoveringPaddedInputExactly)
{
    expectAccepted(ResampleFwdGraphSpec::packed(
        ResampleMode::AVGPOOL_INCLUDE_PADDING, {1, 1, 2, 2}, {4, 4}, {1, 1}, {1, 1}));
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForOutputOfOtherThanFloorModeSize)
{
    auto spec = defaultSpec();
    spec.y.dims = {2, 3, 4, 3};
    spec.y.strides = std::vector<int64_t>{36, 12, 3, 1};
    expectDeclined(spec, "Y spatial dimension 0 is 4, expected 3");
}

// ============================================================================
// Layout rules.
// ============================================================================

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForChannelsFirstXWithChannelsLastY)
{
    auto spec = defaultSpec();
    spec.y.strides = std::vector<int64_t>{27, 1, 9, 3};
    expectDeclined(spec, "both be packed channels-first");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForArbitraryStridesOnLengthOneAxes)
{
    auto spec
        = ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL, {1, 1, 6, 6}, {3, 3}, {2, 2}, {1, 1});
    spec.x.strides = std::vector<int64_t>{7, 999, 6, 1};
    spec.y.strides = std::vector<int64_t>{0, 1, 3, 1};
    expectAccepted(spec);
}

// ============================================================================
// Index rules.
// ============================================================================

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForMaxPoolGenerateIndexWithoutIndexTensor)
{
    auto spec = defaultSpec(ResampleMode::MAXPOOL);
    spec.generateIndex = true;
    expectDeclined(spec, "generate_index is set but the node has no index tensor");
}

// Average pooling produces no index: the flag without an index tensor asks for nothing and is
// ignored. Graph::resample() builds this graph for average pooling.
TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForAveragePoolingGenerateIndexWithoutIndexTensor)
{
    auto spec = defaultSpec(ResampleMode::AVGPOOL_EXCLUDE_PADDING);
    spec.generateIndex = true;
    expectAccepted(spec);
    EXPECT_FALSE(resolveSpec(spec).indexUid.has_value());
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForIndexTensorWithoutGenerateIndex)
{
    auto spec = defaultSpec(ResampleMode::MAXPOOL, DataType::INT32);
    spec.generateIndex = std::nullopt;
    expectDeclined(spec, "generate_index is not set");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForIndexWithAveragePooling)
{
    expectDeclined(defaultSpec(ResampleMode::AVGPOOL_EXCLUDE_PADDING, DataType::INT32),
                   "only max pooling produces indices");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForIndexWithChannelsLastTensors)
{
    auto spec = defaultSpec(ResampleMode::MAXPOOL, DataType::INT32);
    spec.makeChannelsLast();
    expectDeclined(spec, "requires channels-first");
}

// Workaround for ALMIOPEN-3295 (see Workarounds.hpp): MIOpen's BFLOAT16 max pooling can write an
// index into the padding, so an index with padding is declined. Remove this test with the
// workaround; the two accept tests below stay.
TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForBfloat16IndexWithPadding)
{
    expectDeclined(ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL,
                                                {2, 3, 6, 6},
                                                {3, 3},
                                                {2, 2},
                                                {1, 1},
                                                DataType::BFLOAT16,
                                                DataType::INT32),
                   "on BFLOAT16 tensors requires zero padding");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForBfloat16IndexWithoutPadding)
{
    expectAccepted(ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL,
                                                {2, 3, 6, 6},
                                                {3, 3},
                                                {2, 2},
                                                {0, 0},
                                                DataType::BFLOAT16,
                                                DataType::INT32));
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForBfloat16PaddingWithoutIndex)
{
    expectAccepted(ResampleFwdGraphSpec::packed(
        ResampleMode::MAXPOOL, {2, 3, 6, 6}, {3, 3}, {2, 2}, {1, 1}, DataType::BFLOAT16));
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForIndexAliasingX)
{
    auto spec = defaultSpec(ResampleMode::MAXPOOL, DataType::INT32);
    spec.index->uid = spec.x.uid;
    expectDeclined(spec, "index tensor must be distinct from X and Y");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForUnresolvableIndex)
{
    auto spec = defaultSpec(ResampleMode::MAXPOOL, DataType::INT32);
    spec.index->inGraph = false;
    expectDeclined(spec, "index tensor (uid 3) does not resolve");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForUnsupportedIndexDataType)
{
    expectDeclined(defaultSpec(ResampleMode::MAXPOOL, DataType::INT64),
                   "index tensor data type must be INT8 or INT32");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForIndexDimsMismatch)
{
    auto spec = defaultSpec(ResampleMode::MAXPOOL, DataType::INT32);
    spec.index->dims = {2, 3, 3, 4};
    spec.index->strides = std::vector<int64_t>{36, 12, 4, 1};
    expectDeclined(spec, "index tensor dimensions must equal Y dimensions");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForChannelsLastIndex)
{
    auto spec = defaultSpec(ResampleMode::MAXPOOL, DataType::INT32);
    spec.index->strides = std::vector<int64_t>{27, 1, 9, 3};
    expectDeclined(spec, "index tensor must be packed channels-first");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForInt8IndexAtSpatialSize128)
{
    // Indices run up to 8 * 16 - 1 = 127, the largest INT8 value. N and C do not count.
    expectAccepted(identityWindowSpec({1, 2, 8, 16}, DataType::INT8));
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForInt8IndexAtSpatialSize129)
{
    expectDeclined(identityWindowSpec({1, 2, 3, 43}, DataType::INT8),
                   "X spatial size 129 exceeds the range of");
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForInt32IndexAtSpatialSize129)
{
    expectAccepted(identityWindowSpec({1, 2, 3, 43}, DataType::INT32));
}

// ============================================================================
// resolve() output.
// ============================================================================

TEST(TestMiopenResampleFwdChecks, ResolveMapsResampleModesToMiopenModes)
{
    EXPECT_EQ(resolveSpec(defaultSpec(ResampleMode::MAXPOOL)).mode, miopenPoolingMax);
    EXPECT_EQ(resolveSpec(defaultSpec(ResampleMode::AVGPOOL_EXCLUDE_PADDING)).mode,
              miopenPoolingAverage);
    EXPECT_EQ(resolveSpec(defaultSpec(ResampleMode::AVGPOOL_INCLUDE_PADDING)).mode,
              miopenPoolingAverageInclusive);
}

TEST(TestMiopenResampleFwdChecks, ResolveReportsPoolingParameters)
{
    // Y: H = (7 + 2 - 3) / 2 + 1 = 4, W = (5 + 0 - 2) / 1 + 1 = 4.
    const auto config = resolveSpec(ResampleFwdGraphSpec::packed(
        ResampleMode::AVGPOOL_EXCLUDE_PADDING, {1, 2, 7, 5}, {3, 2}, {2, 1}, {1, 0}));
    EXPECT_EQ(config.window, (std::vector<int>{3, 2}));
    EXPECT_EQ(config.padding, (std::vector<int>{1, 0}));
    EXPECT_EQ(config.stride, (std::vector<int>{2, 1}));
}

TEST(TestMiopenResampleFwdChecks, ResolveReportsDataTypeUidsAndDims)
{
    auto spec = ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL,
                                             {2, 3, 6, 6},
                                             {3, 3},
                                             {2, 2},
                                             {0, 0},
                                             DataType::BFLOAT16,
                                             DataType::INT32);
    spec.x.uid = 10;
    spec.y.uid = 20;
    spec.index->uid = 30;

    const auto config = resolveSpec(spec);
    EXPECT_EQ(config.dataType, DataType::BFLOAT16);
    EXPECT_EQ(config.xUid, 10);
    EXPECT_EQ(config.yUid, 20);
    EXPECT_EQ(config.indexUid, std::optional<int64_t>(30));
    EXPECT_EQ(config.xDims, (std::vector<int64_t>{2, 3, 6, 6}));
    EXPECT_EQ(config.yDims, (std::vector<int64_t>{2, 3, 2, 2}));
}

TEST(TestMiopenResampleFwdChecks, ResolveReportsIndexTypes)
{
    const auto int32Config = resolveSpec(defaultSpec(ResampleMode::MAXPOOL, DataType::INT32));
    EXPECT_EQ(int32Config.indexUid, std::optional<int64_t>(ResampleFwdGraphSpec::INDEX_UID));
    EXPECT_EQ(int32Config.indexType, miopenIndexUint32);

    const auto int8Config = resolveSpec(defaultSpec(ResampleMode::MAXPOOL, DataType::INT8));
    EXPECT_EQ(int8Config.indexType, miopenIndexUint8);

    EXPECT_FALSE(resolveSpec(defaultSpec(ResampleMode::MAXPOOL)).indexUid.has_value());
}

TEST(TestMiopenResampleFwdChecks, ResolveReportsChannelsLastStrides)
{
    auto spec = defaultSpec();
    spec.makeChannelsLast();
    const auto config = resolveSpec(spec);
    EXPECT_EQ(config.xStrides, (std::vector<int64_t>{108, 1, 18, 3}));
    EXPECT_EQ(config.yStrides, (std::vector<int64_t>{27, 1, 9, 3}));
}

TEST(TestMiopenResampleFwdChecks, ResolveReplacesStridesOfLengthOneAxes)
{
    auto spec
        = ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL, {1, 1, 6, 6}, {3, 3}, {2, 2}, {1, 1});
    spec.x.strides = std::vector<int64_t>{7, 999, 6, 1};
    spec.y.strides = std::vector<int64_t>{0, 1, 3, 1};

    const auto config = resolveSpec(spec);
    EXPECT_EQ(config.xStrides, (std::vector<int64_t>{36, 36, 6, 1}));
    EXPECT_EQ(config.yStrides, (std::vector<int64_t>{9, 9, 3, 1}));
}

TEST(TestMiopenResampleFwdChecks, ResolvePrefersChannelsFirstWhenBothLayoutsMatch)
{
    // With C == 1, NHWC strides address like NCHW for both tensors.
    auto spec = ResampleFwdGraphSpec::packed(
        ResampleMode::AVGPOOL_EXCLUDE_PADDING, {2, 1, 4, 4}, {2, 2}, {2, 2}, {0, 0});
    spec.makeChannelsLast();

    const auto config = resolveSpec(spec);
    EXPECT_EQ(config.xStrides, (std::vector<int64_t>{16, 16, 4, 1}));
    EXPECT_EQ(config.yStrides, (std::vector<int64_t>{4, 4, 2, 1}));
}

TEST(TestMiopenResampleFwdChecks, ResolveFollowsChannelsLastInputForAmbiguousOutput)
{
    // 2x4x3x3 -> 2x4x1x1 with channels-last X. Y has only length-1 spatial axes, so its
    // channels-first-style strides match both layouts; X settles it as channels-last.
    auto spec
        = ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL, {2, 4, 3, 3}, {3, 3}, {1, 1}, {0, 0});
    spec.makeChannelsLast();
    spec.y.strides = std::vector<int64_t>{4, 1, 1, 1};

    const auto config = resolveSpec(spec);
    EXPECT_EQ(config.xStrides, (std::vector<int64_t>{36, 1, 12, 4}));
    EXPECT_EQ(config.yStrides, (std::vector<int64_t>{4, 1, 4, 4}));
}

// ============================================================================
// Stride normalization: a spatial dimension with a single output never steps its window, so
// resolve() replaces its stride by the smallest one that still yields one output,
// x + 2 * pad - window + 1. Dimensions with more outputs keep their stride.
// ============================================================================

TEST(TestMiopenResampleFwdChecks, ResolveNormalizesStrideOfGlobalPooling)
{
    const auto config = resolveSpec(ResampleFwdGraphSpec::packed(
        ResampleMode::MAXPOOL, {1, 2, 64, 64}, {64, 64}, {64, 64}, {0, 0}));
    EXPECT_EQ(config.yDims, (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(config.stride, (std::vector<int>{1, 1}));
}

TEST(TestMiopenResampleFwdChecks, ResolveNormalizesStrideAgainstPaddedInput)
{
    // 5 + 2 * 1 - 3 + 1 = 5; the padding itself is left alone.
    const auto config = resolveSpec(ResampleFwdGraphSpec::packed(
        ResampleMode::MAXPOOL, {1, 1, 5, 5}, {3, 3}, {10, 10}, {1, 1}));
    EXPECT_EQ(config.stride, (std::vector<int>{5, 5}));
    EXPECT_EQ(config.padding, (std::vector<int>{1, 1}));
}

TEST(TestMiopenResampleFwdChecks, ResolveNormalizesOnlySingleOutputDims)
{
    // H: one output, 7 - 3 + 1 = 5. W: (12 - 3) / 4 + 1 = 3 outputs, stride kept.
    const auto config = resolveSpec(ResampleFwdGraphSpec::packed(
        ResampleMode::MAXPOOL, {1, 1, 7, 12}, {3, 3}, {100, 4}, {0, 0}));
    EXPECT_EQ(config.yDims, (std::vector<int64_t>{1, 1, 1, 3}));
    EXPECT_EQ(config.stride, (std::vector<int>{5, 4}));
}

// ============================================================================
// Channels-last tile limit. MIOpen pools NHWC/NDHWC only through its tiled Forward2d/ForwardNd
// kernels, which are applicable while their per-lane tile fits 1024 * 8191 / 64 = 131056 bytes:
//   2D: roundUp16(esize * ((16 - 1) * stride_h + window_h) * window_w) + 64
//   3D: roundUp16(esize * prod((4 - 1) * stride + window))
// with the strides after normalization. Channels-first graphs fall back to the naive kernel.
// ============================================================================

namespace
{

constexpr const char* TILE_LIMIT_REASON = "too large for MIOpen's channels-last";

ResampleFwdGraphSpec tileSpec(const std::vector<int64_t>& xDims,
                              const std::vector<int64_t>& window,
                              const std::vector<int64_t>& stride,
                              bool channelsLast,
                              DataType dataType,
                              const std::optional<std::vector<int64_t>>& padding = std::nullopt)
{
    auto spec
        = ResampleFwdGraphSpec::packed(ResampleMode::MAXPOOL,
                                       xDims,
                                       window,
                                       stride,
                                       padding.value_or(std::vector<int64_t>(window.size(), 0)),
                                       dataType);
    if(channelsLast)
    {
        spec.makeChannelsLast();
    }
    return spec;
}

} // namespace

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForPlanarFloatTileAtLimit)
{
    // 4 * (15 * 545 + 12) * 4 = 130992, + 64 = 131056. H keeps two outputs so its stride is not
    // normalized.
    expectAccepted(tileSpec({1, 2, 557, 4}, {12, 4}, {545, 1}, true, DataType::FLOAT));
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForPlanarFloatTileOverLimit)
{
    // One window row more: 4 * 8188 * 4 = 131008, + 64 = 131072.
    expectDeclined(tileSpec({1, 2, 558, 4}, {13, 4}, {545, 1}, true, DataType::FLOAT),
                   TILE_LIMIT_REASON);
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForPlanarFloatTileOverLimitInChannelsFirst)
{
    expectAccepted(tileSpec({1, 2, 558, 4}, {13, 4}, {545, 1}, false, DataType::FLOAT));
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForPlanar16BitTileAtLimit)
{
    // 2 * (15 * 545 + 12) * 8 = 130992, + 64 = 131056.
    expectAccepted(tileSpec({1, 2, 557, 8}, {12, 8}, {545, 1}, true, DataType::HALF));
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForFloatTileThatFitsAs16Bit)
{
    // The 16-bit at-limit shape needs twice the bytes in fp32.
    expectDeclined(tileSpec({1, 2, 557, 8}, {12, 8}, {545, 1}, true, DataType::FLOAT),
                   TILE_LIMIT_REASON);
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForVolumetricFloatTileUnderLimit)
{
    // Window 7, stride 8: 4 * 31^3 = 119164.
    expectAccepted(tileSpec({1, 2, 16, 16, 16}, {7, 7, 7}, {8, 8, 8}, true, DataType::FLOAT));
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForVolumetricFloatTileOverLimit)
{
    // Window 8, stride 8: 4 * 32^3 = 131072.
    expectDeclined(tileSpec({1, 2, 16, 16, 16}, {8, 8, 8}, {8, 8, 8}, true, DataType::FLOAT),
                   TILE_LIMIT_REASON);
}

TEST(TestMiopenResampleFwdChecks, IsSupportedTrueForChannelsLastGlobalPoolingThanksToNormalization)
{
    // Unnormalized, stride 64 would need 4 * (15 * 64 + 64) * 64 = 262144 bytes; normalized to 1,
    // 4 * (15 + 64) * 64 = 20224.
    expectAccepted(tileSpec({1, 2, 64, 64}, {64, 64}, {64, 64}, true, DataType::FLOAT));
}

TEST(TestMiopenResampleFwdChecks, IsSupportedFalseForChannelsLastIntMaxWindowWithoutOverflow)
{
    // Padding INT_MAX - 1 lets an INT_MAX window and stride produce two outputs, so nothing is
    // normalized and the tile estimate would overflow without the early stop.
    const auto big = INT_MAX_VALUE;
    expectDeclined(tileSpec({1, 2, 2, 2},
                            {big, 1},
                            {big, 1},
                            true,
                            DataType::FLOAT,
                            std::vector<int64_t>{big - 1, 0}),
                   TILE_LIMIT_REASON);
}
