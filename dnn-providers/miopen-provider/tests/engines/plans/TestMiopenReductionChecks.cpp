// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

#include "engines/plans/MiopenReductionChecks.hpp"

using namespace miopen_plugin;

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::EnumNameReductionMode;
using hipdnn_flatbuffers_sdk::data_objects::EnumValuesReductionMode;
using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
using hipdnn_test_sdk::utilities::createReductionGraph;
using hipdnn_test_sdk::utilities::MockGraph;
using hipdnn_test_sdk::utilities::ReductionGraphSpec;

namespace
{

constexpr int64_t INT32_LIMIT = std::numeric_limits<int32_t>::max();

// The largest element count of a packed FLOAT tensor whose byte span still fits in INT32_MAX.
constexpr int64_t MAX_FLOAT_SPAN_ELEMENTS = INT32_LIMIT / 4;

// The largest input element count whose single-output reduction keeps MIOpen's padded reduce
// length, ceil(n / 256) * 512, within INT32_MAX.
constexpr int64_t MAX_SINGLE_OUTPUT_ELEMENTS = INT32_LIMIT / 512 * 256;

// The reason checkSupported() declines `graph` with, or nullopt if it accepts the graph.
std::optional<std::string> declineReason(const IGraph& graph)
{
    try
    {
        reduction_applicability::checkSupported(graph);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM) << e.getMessage();
        return e.getMessage();
    }
    return std::nullopt;
}

::testing::AssertionResult isAccepted(const ReductionGraphSpec& spec)
{
    const auto builder = createReductionGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const auto reason = declineReason(graph);
    if(!reason.has_value())
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << "declined with \"" << *reason << "\"";
}

// Succeeds only if checkSupported() declines the graph with `reason`. Matching the reason rather
// than only the verdict pins each test to its rule: several rules overlap (an in-place graph also
// has no reduced axis, a dim above INT32_MAX also overflows the byte span), so a bare verdict
// check could keep passing after the rule under test was deleted.
::testing::AssertionResult isDeclinedWith(const IGraph& graph, const std::string& reason)
{
    const auto actual = declineReason(graph);
    if(!actual.has_value())
    {
        return ::testing::AssertionFailure() << "accepted; expected a decline with: " << reason;
    }
    if(actual->find(reason) == std::string::npos)
    {
        return ::testing::AssertionFailure()
               << "declined with \"" << *actual << "\"; expected \"" << reason << "\"";
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult isDeclinedWith(const flatbuffers::FlatBufferBuilder& builder,
                                          const std::string& reason)
{
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    return isDeclinedWith(graph, reason);
}

::testing::AssertionResult isDeclinedWith(const ReductionGraphSpec& spec, const std::string& reason)
{
    return isDeclinedWith(createReductionGraph(spec), reason);
}

// tensorSpanBytes() of a standalone tensor with the given layout.
int64_t spanBytesOf(const std::vector<int64_t>& dims,
                    const std::vector<int64_t>& strides,
                    DataType dataType = DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "t", dataType, &strides, &dims));
    return reduction_applicability::tensorSpanBytes(
        *flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer()));
}

} // namespace

// ============================================================================
// isSupported: checkSupported() as a predicate
// ============================================================================

// The builder's IsApplicable test covers both answers but needs a GPU; this runs everywhere.
TEST(TestMiopenReductionChecks, IsSupportedReturnsFalseForDeclinedGraph)
{
    ReductionGraphSpec spec;
    spec.mode = ReductionMode::MUL_NO_ZEROS;
    const auto builder = createReductionGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(reduction_applicability::isSupported(graph));
}

// ============================================================================
// Byte span
// ============================================================================

TEST(TestMiopenReductionChecks, AcceptsLargestFloatSpan)
{
    EXPECT_TRUE(isAccepted(ReductionGraphSpec::packed(
        ReductionMode::ADD, {MAX_FLOAT_SPAN_ELEMENTS}, {1}, DataType::FLOAT)));
}

TEST(TestMiopenReductionChecks, DeclinesFloatSpanAboveInt32Max)
{
    EXPECT_TRUE(
        isDeclinedWith(ReductionGraphSpec::packed(
                           ReductionMode::ADD, {MAX_FLOAT_SPAN_ELEMENTS + 1}, {1}, DataType::FLOAT),
                       "input tensor spans more than INT32_MAX bytes"));
}

TEST(TestMiopenReductionChecks, TensorSpanBytesForPackedTensor)
{
    EXPECT_EQ(spanBytesOf({2, 3, 4, 5}, {60, 20, 5, 1}, DataType::FLOAT), 120 * 4);
    EXPECT_EQ(spanBytesOf({2, 3, 4, 5}, {60, 20, 5, 1}, DataType::HALF), 120 * 2);
}

TEST(TestMiopenReductionChecks, TensorSpanBytesCountsPaddingUpToLastElement)
{
    // Last offset 1 * 120 + 2 * 40 + 3 * 8 + 4 * 1 = 228; trailing padding is not counted.
    EXPECT_EQ(spanBytesOf({2, 3, 4, 5}, {120, 40, 8, 1}), (228 + 1) * 4);
}

TEST(TestMiopenReductionChecks, TensorSpanBytesCapsEachAxisAtInt32Max)
{
    // Uncapped, (3 - 1) * INT32_MAX elements; capped, the term is INT32_MAX.
    EXPECT_EQ(spanBytesOf({3}, {INT32_LIMIT}), (INT32_LIMIT + 1) * 4);

    // Six axes of 2^40 by 2^40 would overflow int64; each term saturates instead.
    constexpr int64_t HUGE_EXTENT = int64_t{1} << 40;
    const std::vector<int64_t> huge(6, HUGE_EXTENT);
    EXPECT_EQ(spanBytesOf(huge, huge), (6 * INT32_LIMIT + 1) * 4);
}

// ============================================================================
// Input element count and single-output padded reduce length
// ============================================================================

namespace
{

// x reads `xDims` through stride-1 axes, so the byte span stays tiny however many elements x
// has, and is reduced over every axis but the first. Reading an element more than once is
// harmless; only the output must not overlap.
ReductionGraphSpec selfOverlappingInputSpec(const std::vector<int64_t>& xDims)
{
    std::vector<int64_t> yDims(xDims.size(), 1);
    yDims.front() = xDims.front();
    auto spec = ReductionGraphSpec::packed(ReductionMode::ADD, xDims, yDims);
    spec.x.strides = std::vector<int64_t>(xDims.size(), 1);
    spec.y.strides = std::vector<int64_t>(yDims.size(), 1);
    return spec;
}

} // namespace

TEST(TestMiopenReductionChecks, AcceptsLargestShapeableInputElementCount)
{
    // INT32_MAX is prime, so an input of exactly INT32_MAX elements has an axis of that length
    // and spans more than INT32_MAX bytes. This is the next count down: 2^31 - 2. It is also the
    // rank-6 accept, and its 331 outputs keep it clear of the single-output limit.
    EXPECT_TRUE(isAccepted(selfOverlappingInputSpec({331, 151, 31, 11, 7, 18})));
}

TEST(TestMiopenReductionChecks, DeclinesInputElementCountThatOverflowsInt64)
{
    // 2^96 elements: an unsaturated product would wrap to 0 and pass the limit.
    EXPECT_TRUE(isDeclinedWith(selfOverlappingInputSpec(std::vector<int64_t>(6, 65536)),
                               "input tensor has more than INT32_MAX elements"));
}

TEST(TestMiopenReductionChecks, AcceptsSingleOutputAtPaddedReduceLengthLimit)
{
    EXPECT_TRUE(isAccepted(ReductionGraphSpec::packed(
        ReductionMode::ADD, {MAX_SINGLE_OUTPUT_ELEMENTS}, {1}, DataType::HALF)));
}

TEST(TestMiopenReductionChecks, DeclinesSingleOutputAbovePaddedReduceLengthLimit)
{
    EXPECT_TRUE(isDeclinedWith(
        ReductionGraphSpec::packed(
            ReductionMode::ADD, {MAX_SINGLE_OUTPUT_ELEMENTS + 1}, {1}, DataType::HALF),
        "a single-output reduction over this many elements exceeds MIOpen's int32 padded reduce "
        "length"));
}

// ============================================================================
// Output overlap
// ============================================================================

TEST(TestMiopenReductionChecks, AcceptsPaddedOutputStrides)
{
    // C padded from 3 to 4 elements: gaps between output elements are fine.
    ReductionGraphSpec spec;
    spec.y.strides = std::vector<int64_t>{4, 1, 1, 1};

    EXPECT_TRUE(isAccepted(spec));
}

TEST(TestMiopenReductionChecks, AcceptsArbitraryStridesOnLengthOneOutputAxes)
{
    // Length-1 axes address one element whatever their stride: axis 2 repeats axis 1's stride
    // and axis 3's lands inside axis 1's reach, which would overlap were the axes longer.
    ReductionGraphSpec spec;
    spec.y.strides = std::vector<int64_t>{3, 1, 1, 2};

    EXPECT_TRUE(isAccepted(spec));
}

namespace
{

// y [2, 2, 2, 1] with strides {1, 2, outerStride, 1}. Any two of its first three axes address
// distinct elements; with an outer stride of 3, output (0, 0, 1) also lands on (1, 1, 0).
ReductionGraphSpec threeAxisOutputSpec(int64_t outerStride)
{
    auto spec = ReductionGraphSpec::packed(ReductionMode::ADD, {2, 2, 2, 4}, {2, 2, 2, 1});
    spec.y.strides = std::vector<int64_t>{1, 2, outerStride, 1};
    return spec;
}

} // namespace

TEST(TestMiopenReductionChecks, AcceptsOutputStridePastCombinedReachOfInnerAxes)
{
    EXPECT_TRUE(isAccepted(threeAxisOutputSpec(4)));
}

TEST(TestMiopenReductionChecks, DeclinesOutputOverlappingOnlyThroughCombinedInnerAxes)
{
    EXPECT_TRUE(isDeclinedWith(threeAxisOutputSpec(3),
                               "output tensor strides make distinct elements overlap"));
}

// ============================================================================
// Declined: graph and node level
// ============================================================================

TEST(TestMiopenReductionChecks, DeclinesMultiNodeGraph)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));
    // The node-count guard must reject the graph before anything else is inspected: if it were
    // deleted or weakened, hasOnlySupportedAttributes() would be reached next, and this
    // expectation would fail the test instead of silently passing via gmock's default action.
    EXPECT_CALL(mockGraph, hasOnlySupportedAttributes(::testing::_)).Times(0);

    EXPECT_TRUE(isDeclinedWith(mockGraph, "applicable only for single-node graphs"));
}

TEST(TestMiopenReductionChecks, DeclinesNonReductionNode)
{
    EXPECT_TRUE(isDeclinedWith(hipdnn_test_sdk::utilities::createPointwiseGraph(
                                   hipdnn_test_sdk::utilities::PointwiseGraphSpec::unary()),
                               "graph contains unsupported node attributes"));
}

TEST(TestMiopenReductionChecks, DeclinesOverrideShapeEnabled)
{
    ReductionGraphSpec spec;
    spec.overrideShapeEnabled = true;

    EXPECT_TRUE(isDeclinedWith(spec, "does not support override shapes"));
}

TEST(TestMiopenReductionChecks, DeclinesNonFloatComputeType)
{
    ReductionGraphSpec spec;
    spec.computeDataType = DataType::HALF;

    EXPECT_TRUE(isDeclinedWith(spec, "only supports nodes with an fp32 compute_data_type"));
}

TEST(TestMiopenReductionChecks, DeclinesDeterministicReduction)
{
    ReductionGraphSpec spec;
    spec.isDeterministic = true;

    EXPECT_TRUE(isDeclinedWith(spec, "deterministic reduction is not supported"));
}

TEST(TestMiopenReductionChecks, DeclinesInPlaceReduction)
{
    // x and y resolve to the same tensor, so the no-reduced-axis rule would decline this graph
    // too; only the reason tells the two rules apart.
    ReductionGraphSpec spec;
    spec.y.uid = spec.x.uid;

    EXPECT_TRUE(isDeclinedWith(spec, "does not support in-place execution"));
}

TEST(TestMiopenReductionChecks, DeclinesUnresolvableInputUid)
{
    ReductionGraphSpec spec;
    spec.x.inGraph = false;

    EXPECT_TRUE(isDeclinedWith(spec, "in/out tensor uids do not resolve"));
}

TEST(TestMiopenReductionChecks, DeclinesUnresolvableOutputUid)
{
    ReductionGraphSpec spec;
    spec.y.inGraph = false;

    EXPECT_TRUE(isDeclinedWith(spec, "in/out tensor uids do not resolve"));
}

TEST(TestMiopenReductionChecks, DeclinesUnservedMode)
{
    // NOT_SET takes the same branch; the mapping test covers which modes have no MIOpen op.
    ReductionGraphSpec spec;
    spec.mode = ReductionMode::MUL_NO_ZEROS;

    EXPECT_TRUE(isDeclinedWith(spec, "unsupported reduction mode: MUL_NO_ZEROS"));
}

// ============================================================================
// Declined: per-tensor rules
// ============================================================================

// x and y go through the same checks, labelled "input" and "output". Each test breaks one rule
// on x of the valid default spec; DeclinesVirtualOutputTensor shows that y is checked too.

TEST(TestMiopenReductionChecks, DeclinesVirtualInputTensor)
{
    ReductionGraphSpec spec;
    spec.x.isVirtual = true;

    EXPECT_TRUE(isDeclinedWith(spec, "input tensor must be non-virtual"));
}

TEST(TestMiopenReductionChecks, DeclinesVirtualOutputTensor)
{
    ReductionGraphSpec spec;
    spec.y.isVirtual = true;

    EXPECT_TRUE(isDeclinedWith(spec, "output tensor must be non-virtual"));
}

TEST(TestMiopenReductionChecks, DeclinesPassByValueTensor)
{
    ReductionGraphSpec spec;
    spec.x.isRuntimePassByValue = true;

    EXPECT_TRUE(isDeclinedWith(spec, "input tensor must not be pass-by-value"));
}

TEST(TestMiopenReductionChecks, DeclinesRaggedTensor)
{
    ReductionGraphSpec spec;
    spec.x.raggedOffsetTensorUid = 7;

    EXPECT_TRUE(isDeclinedWith(spec, "input tensor must not be ragged"));
}

TEST(TestMiopenReductionChecks, DeclinesBfloat16Tensor)
{
    ReductionGraphSpec spec;
    spec.dataType = DataType::BFLOAT16;
    spec.yDataType = DataType::FLOAT;

    EXPECT_TRUE(isDeclinedWith(spec, "input tensor data type BFLOAT16 is not supported"));
}

TEST(TestMiopenReductionChecks, DeclinesNullStrides)
{
    ReductionGraphSpec spec;
    spec.x.strides = std::nullopt;

    EXPECT_TRUE(isDeclinedWith(spec, "input tensor dims or strides are null"));
}

TEST(TestMiopenReductionChecks, DeclinesDimsStridesSizeMismatch)
{
    ReductionGraphSpec spec;
    spec.x.strides = std::vector<int64_t>{60, 20, 5};

    EXPECT_TRUE(isDeclinedWith(spec, "input tensor dims and strides size mismatch"));
}

TEST(TestMiopenReductionChecks, DeclinesRankZero)
{
    ReductionGraphSpec spec;
    spec.x.dims.clear();
    spec.x.strides = std::vector<int64_t>{};

    EXPECT_TRUE(isDeclinedWith(spec, "input tensor rank must be between 1 and 6, got 0"));
}

TEST(TestMiopenReductionChecks, DeclinesRankSeven)
{
    // Three leading length-1 axes take the rank-4 tensor to rank 7 without changing its span.
    ReductionGraphSpec spec;
    spec.x.dims = {1, 1, 1, 2, 3, 4, 5};
    spec.x.strides = std::vector<int64_t>{60, 60, 60, 60, 20, 5, 1};

    EXPECT_TRUE(isDeclinedWith(spec, "input tensor rank must be between 1 and 6, got 7"));
}

TEST(TestMiopenReductionChecks, DeclinesNonPositiveDim)
{
    ReductionGraphSpec spec;
    spec.x.dims[1] = 0;
    EXPECT_TRUE(isDeclinedWith(spec, "input tensor has a non-positive dim or stride at axis 1"));

    spec.x.dims[1] = -1;
    EXPECT_TRUE(isDeclinedWith(spec, "input tensor has a non-positive dim or stride at axis 1"));
}

TEST(TestMiopenReductionChecks, DeclinesNonPositiveStride)
{
    ReductionGraphSpec spec;
    spec.x.strides = std::vector<int64_t>{60, 0, 5, 1};
    EXPECT_TRUE(isDeclinedWith(spec, "input tensor has a non-positive dim or stride at axis 1"));

    spec.x.strides = std::vector<int64_t>{60, -1, 5, 1};
    EXPECT_TRUE(isDeclinedWith(spec, "input tensor has a non-positive dim or stride at axis 1"));
}

TEST(TestMiopenReductionChecks, DeclinesDimAboveInt32Max)
{
    ReductionGraphSpec spec;
    spec.x.dims[1] = INT32_LIMIT + 1;

    EXPECT_TRUE(
        isDeclinedWith(spec, "input tensor has a dim or stride exceeding INT32_MAX at axis 1"));
}

TEST(TestMiopenReductionChecks, AcceptsStrideAtInt32MaxOnLengthOneAxis)
{
    // A length-1 axis contributes nothing to the byte span, so only the per-axis limit applies.
    auto spec = ReductionGraphSpec::packed(ReductionMode::ADD, {1, 8}, {1, 1});
    spec.x.strides = std::vector<int64_t>{INT32_LIMIT, 1};

    EXPECT_TRUE(isAccepted(spec));
}

TEST(TestMiopenReductionChecks, DeclinesStrideAboveInt32MaxOnLengthOneAxis)
{
    auto spec = ReductionGraphSpec::packed(ReductionMode::ADD, {1, 8}, {1, 1});
    spec.x.strides = std::vector<int64_t>{INT32_LIMIT + 1, 1};

    EXPECT_TRUE(
        isDeclinedWith(spec, "input tensor has a dim or stride exceeding INT32_MAX at axis 0"));
}

// ============================================================================
// Declined: rules relating x to y
// ============================================================================

TEST(TestMiopenReductionChecks, DeclinesMixedDataTypes)
{
    ReductionGraphSpec spec;
    spec.yDataType = DataType::HALF;

    EXPECT_TRUE(isDeclinedWith(spec, "input and output tensors must have the same data type"));
}

TEST(TestMiopenReductionChecks, DeclinesRankMismatch)
{
    EXPECT_TRUE(
        isDeclinedWith(ReductionGraphSpec::packed(ReductionMode::ADD, {2, 3, 4, 5}, {2, 3, 1}),
                       "input and output tensors must have the same rank"));
}

TEST(TestMiopenReductionChecks, DeclinesOutputDimNeitherOneNorInputDim)
{
    EXPECT_TRUE(
        isDeclinedWith(ReductionGraphSpec::packed(ReductionMode::ADD, {2, 3, 4, 5}, {2, 3, 2, 1}),
                       "output dim at axis 2 must be 1 or equal to the input dim"));
}

TEST(TestMiopenReductionChecks, DeclinesWhenOnlyLengthOneAxesAreReduced)
{
    // MIOpen counts axis 0 as reduced because the output dim is 1, but it reduces one element.
    EXPECT_TRUE(isDeclinedWith(ReductionGraphSpec::packed(ReductionMode::ADD, {1, 8}, {1, 8}),
                               "no axis is reduced"));
}

// ============================================================================
// toMiopenReduceTensorOp
// ============================================================================

TEST(TestMiopenReductionChecks, ToMiopenReduceTensorOpMapsEveryReductionMode)
{
    const std::map<ReductionMode, std::optional<miopenReduceTensorOp_t>> expected
        = {{ReductionMode::NOT_SET, std::nullopt},
           {ReductionMode::ADD, MIOPEN_REDUCE_TENSOR_ADD},
           {ReductionMode::MUL, MIOPEN_REDUCE_TENSOR_MUL},
           {ReductionMode::MIN_OP, MIOPEN_REDUCE_TENSOR_MIN},
           {ReductionMode::MAX_OP, MIOPEN_REDUCE_TENSOR_MAX},
           {ReductionMode::AMAX, MIOPEN_REDUCE_TENSOR_AMAX},
           {ReductionMode::AVG, MIOPEN_REDUCE_TENSOR_AVG},
           {ReductionMode::NORM1, MIOPEN_REDUCE_TENSOR_NORM1},
           {ReductionMode::NORM2, MIOPEN_REDUCE_TENSOR_NORM2},
           {ReductionMode::MUL_NO_ZEROS, std::nullopt}};

    // Iterating the schema's enumerators, not the table, makes a newly added mode fail here.
    for(const auto mode : EnumValuesReductionMode())
    {
        const auto it = expected.find(mode);
        ASSERT_NE(it, expected.end()) << "no expectation for " << EnumNameReductionMode(mode);
        EXPECT_EQ(reduction_applicability::toMiopenReduceTensorOp(mode), it->second)
            << EnumNameReductionMode(mode);
    }

    // A value outside the schema takes the default branch.
    EXPECT_EQ(reduction_applicability::toMiopenReduceTensorOp(static_cast<ReductionMode>(42)),
              std::nullopt);
}
