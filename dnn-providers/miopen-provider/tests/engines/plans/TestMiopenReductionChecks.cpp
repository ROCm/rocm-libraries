// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

#include "common/ReductionCommon.hpp"
#include "engines/plans/MiopenReductionChecks.hpp"

using namespace miopen_plugin;
using namespace reduction_common;

using hipdnn_flatbuffers_sdk::data_objects::EnumNameReductionMode;
using hipdnn_flatbuffers_sdk::data_objects::EnumValuesReductionMode;
using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
using hipdnn_test_sdk::utilities::MockGraph;

namespace
{

constexpr int64_t INT32_LIMIT = std::numeric_limits<int32_t>::max();

// The largest element counts of a packed tensor whose byte span still fits in INT32_MAX.
constexpr int64_t MAX_FLOAT_SPAN_ELEMENTS = INT32_LIMIT / 4;
constexpr int64_t MAX_HALF_SPAN_ELEMENTS = INT32_LIMIT / 2;

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

struct DataTypeCase
{
    DataType dataType;
    const char* name;
};

std::ostream& operator<<(std::ostream& os, const DataTypeCase& dataTypeCase)
{
    return os << dataTypeCase.name;
}

const std::vector<DataTypeCase>& getServedDataTypeCases()
{
    static const std::vector<DataTypeCase> s_cases
        = {{DataType::FLOAT, "Float"}, {DataType::HALF, "Half"}};
    return s_cases;
}

} // namespace

// ============================================================================
// isSupported: checkSupported() as a predicate
// ============================================================================

TEST(TestMiopenReductionChecks, IsSupportedReturnsTrueForValidGraph)
{
    const auto builder = createReductionGraph(ReductionGraphSpec{});
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(reduction_applicability::isSupported(graph));
}

TEST(TestMiopenReductionChecks, IsSupportedReturnsFalseForDeclinedGraph)
{
    ReductionGraphSpec spec;
    spec.mode = ReductionMode::MUL_NO_ZEROS;
    const auto builder = createReductionGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(reduction_applicability::isSupported(graph));
}

// ============================================================================
// Accepted graphs
// ============================================================================

class TestMiopenReductionChecksModes
    : public ::testing::TestWithParam<std::tuple<ReductionModeCase, DataTypeCase>>
{
};

TEST_P(TestMiopenReductionChecksModes, AcceptsValidGraph)
{
    ReductionGraphSpec spec;
    spec.mode = std::get<0>(GetParam()).mode;
    spec.dataType = std::get<1>(GetParam()).dataType;

    EXPECT_TRUE(isAccepted(spec));
}

INSTANTIATE_TEST_SUITE_P(
    AllCases,
    TestMiopenReductionChecksModes,
    ::testing::Combine(::testing::ValuesIn(getServedReductionModeCases()),
                       ::testing::ValuesIn(getServedDataTypeCases())),
    [](const ::testing::TestParamInfo<std::tuple<ReductionModeCase, DataTypeCase>>& info) {
        return std::string(std::get<0>(info.param).name) + std::get<1>(info.param).name;
    });

class TestMiopenReductionChecksRanks : public ::testing::TestWithParam<size_t>
{
};

TEST_P(TestMiopenReductionChecksRanks, AcceptsInnermostAxisReduction)
{
    const std::vector<int64_t> allDims{2, 3, 4, 5, 6, 7};
    const std::vector<int64_t> xDims(allDims.begin(),
                                     allDims.begin() + static_cast<std::ptrdiff_t>(GetParam()));
    auto yDims = xDims;
    yDims.back() = 1;

    EXPECT_TRUE(isAccepted(ReductionGraphSpec::packed(ReductionMode::ADD, xDims, yDims)));
}

// Rank 6 is MIOpen's maximum; rank 7 is declined in TestMiopenReductionChecksTensors.
INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestMiopenReductionChecksRanks,
                         ::testing::Range<size_t>(1, 7),
                         [](const ::testing::TestParamInfo<size_t>& info) {
                             return "Rank" + std::to_string(info.param);
                         });

TEST(TestMiopenReductionChecks, AcceptsNhwcStrides)
{
    ReductionGraphSpec spec;
    spec.x.strides = nhwcStrides(spec.x.dims);
    spec.y.strides = nhwcStrides(spec.y.dims);

    EXPECT_TRUE(isAccepted(spec));
}

TEST(TestMiopenReductionChecks, AcceptsNonAdjacentMultiAxisReduction)
{
    EXPECT_TRUE(
        isAccepted(ReductionGraphSpec::packed(ReductionMode::ADD, {2, 3, 4, 5}, {1, 3, 1, 5})));
}

TEST(TestMiopenReductionChecks, AcceptsFullReduction)
{
    EXPECT_TRUE(
        isAccepted(ReductionGraphSpec::packed(ReductionMode::ADD, {2, 3, 4, 5}, {1, 1, 1, 1})));
}

TEST(TestMiopenReductionChecks, AcceptsPaddedOutputStrides)
{
    // C padded from 3 to 4 elements.
    ReductionGraphSpec spec;
    spec.y.strides = std::vector<int64_t>{4, 1, 1, 1};

    EXPECT_TRUE(isAccepted(spec));
}

TEST(TestMiopenReductionChecks, AcceptsPaddedInputStrides)
{
    // Rows of W = 5 padded to 8 elements, and each H x W plane padded from 32 to 40.
    ReductionGraphSpec spec;
    spec.x.strides = std::vector<int64_t>{120, 40, 8, 1};

    EXPECT_TRUE(isAccepted(spec));
}

TEST(TestMiopenReductionChecks, AcceptsUnreducedLengthOneAxisBesideReducedAxis)
{
    // Axis 1 is 1 on both sides; only axis 3 is really reduced.
    EXPECT_TRUE(
        isAccepted(ReductionGraphSpec::packed(ReductionMode::ADD, {2, 1, 4, 5}, {2, 1, 4, 1})));
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

TEST(TestMiopenReductionChecks, AcceptsHalfWithElementCountDeclinedAsFloat)
{
    // The span limit is in bytes, so the element count declined as FLOAT above fits as HALF.
    EXPECT_TRUE(isAccepted(ReductionGraphSpec::packed(
        ReductionMode::ADD, {MAX_FLOAT_SPAN_ELEMENTS + 1}, {1}, DataType::HALF)));
}

namespace
{

// A HALF x whose last element sits `lastOffset` elements from its first, reduced over its inner
// axis into two outputs, so neither the element-count nor the single-output rule applies.
ReductionGraphSpec halfSpanSpec(int64_t lastOffset)
{
    constexpr int64_t INNER = MAX_HALF_SPAN_ELEMENTS / 2;
    auto spec = ReductionGraphSpec::packed(ReductionMode::ADD, {2, INNER}, {2, 1}, DataType::HALF);
    spec.x.strides = std::vector<int64_t>{lastOffset - (INNER - 1), 1};
    return spec;
}

} // namespace

TEST(TestMiopenReductionChecks, AcceptsLargestHalfSpan)
{
    EXPECT_TRUE(isAccepted(halfSpanSpec(MAX_HALF_SPAN_ELEMENTS - 1)));
}

TEST(TestMiopenReductionChecks, DeclinesHalfSpanAboveInt32Max)
{
    EXPECT_TRUE(isDeclinedWith(halfSpanSpec(MAX_HALF_SPAN_ELEMENTS),
                               "input tensor spans more than INT32_MAX bytes"));
}

TEST(TestMiopenReductionChecks, DeclinesDimAtInt32MaxOnlyBySpanRule)
{
    // A dim of exactly INT32_MAX passes the per-axis limit; it is the byte span that declines it.
    EXPECT_TRUE(isDeclinedWith(
        ReductionGraphSpec::packed(ReductionMode::ADD, {INT32_LIMIT}, {1}, DataType::HALF),
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

TEST(TestMiopenReductionChecks, TensorSpanBytesIgnoresStrideOfLengthOneAxis)
{
    EXPECT_EQ(spanBytesOf({1, 8}, {INT32_LIMIT, 1}), 8 * 4);
}

TEST(TestMiopenReductionChecks, TensorSpanBytesForSelfOverlappingStrides)
{
    EXPECT_EQ(spanBytesOf({65536, 65536}, {1, 1}), (2 * 65535 + 1) * 4);
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

TEST(TestMiopenReductionChecks, AcceptsSelfOverlappingInputStrides)
{
    // Reading an element more than once is harmless; only the output must not overlap.
    auto spec = ReductionGraphSpec::packed(ReductionMode::ADD, {4, 8}, {4, 1});
    spec.x.strides = std::vector<int64_t>{1, 1};

    EXPECT_TRUE(isAccepted(spec));
}

namespace
{

// x reads `xDims` through stride-1 axes, so the byte span stays tiny however many elements x
// has, and is reduced over every axis but the first.
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
    // and spans more than INT32_MAX bytes. This is the next count down: 2^31 - 2.
    EXPECT_TRUE(isAccepted(selfOverlappingInputSpec({331, 151, 31, 11, 7, 18})));
}

TEST(TestMiopenReductionChecks, DeclinesInputElementCountAboveInt32Max)
{
    // 2^31 = INT32_MAX + 1 elements in a 384 KiB span.
    EXPECT_TRUE(isDeclinedWith(selfOverlappingInputSpec({65536, 32768}),
                               "input tensor has more than INT32_MAX elements"));
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

TEST(TestMiopenReductionChecks, DeclinesRankTwoSingleOutputAbovePaddedReduceLengthLimit)
{
    const std::vector<int64_t> xDims{2, (MAX_SINGLE_OUTPUT_ELEMENTS + 2) / 2};
    EXPECT_TRUE(isDeclinedWith(
        ReductionGraphSpec::packed(ReductionMode::ADD, xDims, {1, 1}, DataType::HALF),
        "a single-output reduction over this many elements exceeds MIOpen's int32 "
        "padded reduce length"));
}

TEST(TestMiopenReductionChecks, AcceptsTwoOutputsAbovePaddedReduceLengthLimit)
{
    // The same input as above, kept in two outputs: the padded-length limit does not apply.
    const std::vector<int64_t> xDims{2, (MAX_SINGLE_OUTPUT_ELEMENTS + 2) / 2};
    EXPECT_TRUE(
        isAccepted(ReductionGraphSpec::packed(ReductionMode::ADD, xDims, {2, 1}, DataType::HALF)));
}

// ============================================================================
// Output overlap
// ============================================================================

namespace
{

// x [4, 3, 8] reduced over axis 1 into y [4, 1, 8] with y's outer stride set to `outerStride`.
// A packed y has 8; anything less lands output (1, 0, 0) on or before (0, 0, 7).
ReductionGraphSpec outputOuterStrideSpec(int64_t outerStride)
{
    auto spec = ReductionGraphSpec::packed(ReductionMode::ADD, {4, 3, 8}, {4, 1, 8});
    spec.y.strides = std::vector<int64_t>{outerStride, 8, 1};
    return spec;
}

} // namespace

TEST(TestMiopenReductionChecks, AcceptsOutputStrideJustPastInnerAxis)
{
    EXPECT_TRUE(isAccepted(outputOuterStrideSpec(8)));
}

TEST(TestMiopenReductionChecks, DeclinesOutputStrideReachingIntoInnerAxis)
{
    EXPECT_TRUE(isDeclinedWith(outputOuterStrideSpec(7),
                               "output tensor strides make distinct elements overlap"));
}

TEST(TestMiopenReductionChecks, DeclinesOutputWithTwoUnitStrideAxes)
{
    auto spec = ReductionGraphSpec::packed(ReductionMode::ADD, {8, 4, 8}, {8, 1, 8});
    spec.y.strides = std::vector<int64_t>{1, 1, 1};

    EXPECT_TRUE(isDeclinedWith(spec, "output tensor strides make distinct elements overlap"));
}

TEST(TestMiopenReductionChecks, AcceptsOutputWithGapsBetweenElements)
{
    auto spec = ReductionGraphSpec::packed(ReductionMode::ADD, {2, 8, 4, 4}, {2, 8, 1, 1});
    spec.y.strides = std::vector<int64_t>{16, 2, 1, 1};

    EXPECT_TRUE(isAccepted(spec));
}

TEST(TestMiopenReductionChecks, AcceptsNhwcOutputReducedOverChannels)
{
    // y [2, 1, 4, 5] channels-last has strides {20, 1, 5, 1}: C and W share stride 1.
    auto spec = ReductionGraphSpec::packed(ReductionMode::ADD, {2, 3, 4, 5}, {2, 1, 4, 5});
    spec.x.strides = nhwcStrides(spec.x.dims);
    spec.y.strides = nhwcStrides(spec.y.dims);

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

class TestMiopenReductionChecksUnservedModes : public ::testing::TestWithParam<ReductionModeCase>
{
};

TEST_P(TestMiopenReductionChecksUnservedModes, DeclinesUnservedMode)
{
    ReductionGraphSpec spec;
    spec.mode = GetParam().mode;

    EXPECT_TRUE(isDeclinedWith(spec,
                               std::string("unsupported reduction mode: ")
                                   + EnumNameReductionMode(GetParam().mode)));
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestMiopenReductionChecksUnservedModes,
                         ::testing::Values(ReductionModeCase{ReductionMode::NOT_SET, "NotSet"},
                                           ReductionModeCase{ReductionMode::MUL_NO_ZEROS,
                                                             "MulNoZeros"}),
                         [](const ::testing::TestParamInfo<ReductionModeCase>& info) {
                             return std::string(info.param.name);
                         });

// ============================================================================
// Declined: rules applied to each of x and y
// ============================================================================

namespace
{

struct TensorRoleCase
{
    bool isOutput;
    // The label the gate prefixes its tensor messages with.
    const char* label;
    const char* name;
};

std::ostream& operator<<(std::ostream& os, const TensorRoleCase& roleCase)
{
    return os << roleCase.name;
}

} // namespace

// Each case starts from the valid default spec and breaks one rule on either x or y.
class TestMiopenReductionChecksTensors : public ::testing::TestWithParam<TensorRoleCase>
{
protected:
    ReductionTensorSpec& tensorUnderTest()
    {
        return GetParam().isOutput ? _spec.y : _spec.x;
    }

    std::vector<int64_t>& stridesUnderTest()
    {
        return tensorUnderTest().strides.value();
    }

    // Changes the data type of the tensor under test only.
    void setDataTypeUnderTest(DataType dataType)
    {
        if(GetParam().isOutput)
        {
            _spec.yDataType = dataType;
        }
        else
        {
            _spec.dataType = dataType;
            _spec.yDataType = DataType::FLOAT;
        }
    }

    static std::string reason(const std::string& rule)
    {
        return std::string(GetParam().label) + " tensor " + rule;
    }

    ReductionGraphSpec _spec;
};

TEST_P(TestMiopenReductionChecksTensors, DeclinesVirtualTensor)
{
    tensorUnderTest().isVirtual = true;

    EXPECT_TRUE(isDeclinedWith(_spec, reason("must be non-virtual")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesPassByValueTensor)
{
    tensorUnderTest().isRuntimePassByValue = true;

    EXPECT_TRUE(isDeclinedWith(_spec, reason("must not be pass-by-value")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesRaggedTensor)
{
    tensorUnderTest().raggedOffsetTensorUid = 7;

    EXPECT_TRUE(isDeclinedWith(_spec, reason("must not be ragged")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesBfloat16Tensor)
{
    setDataTypeUnderTest(DataType::BFLOAT16);

    EXPECT_TRUE(isDeclinedWith(_spec, reason("data type BFLOAT16 is not supported")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesDoubleTensor)
{
    setDataTypeUnderTest(DataType::DOUBLE);

    EXPECT_TRUE(isDeclinedWith(_spec, reason("data type DOUBLE is not supported")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesInt8Tensor)
{
    setDataTypeUnderTest(DataType::INT8);

    EXPECT_TRUE(isDeclinedWith(_spec, reason("data type INT8 is not supported")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesNullStrides)
{
    tensorUnderTest().strides = std::nullopt;

    EXPECT_TRUE(isDeclinedWith(_spec, reason("dims or strides are null")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesDimsStridesSizeMismatch)
{
    stridesUnderTest().pop_back();

    EXPECT_TRUE(isDeclinedWith(_spec, reason("dims and strides size mismatch")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesRankZero)
{
    tensorUnderTest().dims.clear();
    stridesUnderTest().clear();

    EXPECT_TRUE(isDeclinedWith(_spec, reason("rank must be between 1 and 6, got 0")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesRankSeven)
{
    // Three leading length-1 axes take the rank-4 tensor to rank 7 without changing its span.
    auto& tensor = tensorUnderTest();
    auto& strides = stridesUnderTest();
    const auto outermostStride = strides.front();
    tensor.dims.insert(tensor.dims.begin(), 3, 1);
    strides.insert(strides.begin(), 3, outermostStride);

    EXPECT_TRUE(isDeclinedWith(_spec, reason("rank must be between 1 and 6, got 7")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesZeroDim)
{
    tensorUnderTest().dims[1] = 0;

    EXPECT_TRUE(isDeclinedWith(_spec, reason("has a non-positive dim or stride at axis 1")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesNegativeDim)
{
    tensorUnderTest().dims[1] = -1;

    EXPECT_TRUE(isDeclinedWith(_spec, reason("has a non-positive dim or stride at axis 1")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesZeroStride)
{
    stridesUnderTest()[1] = 0;

    EXPECT_TRUE(isDeclinedWith(_spec, reason("has a non-positive dim or stride at axis 1")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesNegativeStride)
{
    stridesUnderTest()[1] = -1;

    EXPECT_TRUE(isDeclinedWith(_spec, reason("has a non-positive dim or stride at axis 1")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesDimAboveInt32Max)
{
    tensorUnderTest().dims[1] = INT32_LIMIT + 1;

    EXPECT_TRUE(isDeclinedWith(_spec, reason("has a dim or stride exceeding INT32_MAX at axis 1")));
}

TEST_P(TestMiopenReductionChecksTensors, AcceptsStrideAtInt32MaxOnLengthOneAxis)
{
    // A length-1 axis contributes nothing to the byte span, so only the per-axis limit applies.
    _spec = ReductionGraphSpec::packed(ReductionMode::ADD, {1, 8}, {1, 1});
    stridesUnderTest()[0] = INT32_LIMIT;

    EXPECT_TRUE(isAccepted(_spec));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesStrideAboveInt32MaxOnLengthOneAxis)
{
    _spec = ReductionGraphSpec::packed(ReductionMode::ADD, {1, 8}, {1, 1});
    stridesUnderTest()[0] = INT32_LIMIT + 1;

    EXPECT_TRUE(isDeclinedWith(_spec, reason("has a dim or stride exceeding INT32_MAX at axis 0")));
}

TEST_P(TestMiopenReductionChecksTensors, DeclinesStridedSpanAboveInt32Max)
{
    // Few elements, but the outermost stride spreads them over more than INT32_MAX bytes.
    stridesUnderTest()[0] = MAX_FLOAT_SPAN_ELEMENTS + 1;

    EXPECT_TRUE(isDeclinedWith(_spec, reason("spans more than INT32_MAX bytes")));
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestMiopenReductionChecksTensors,
                         ::testing::Values(TensorRoleCase{false, "input", "Input"},
                                           TensorRoleCase{true, "output", "Output"}),
                         [](const ::testing::TestParamInfo<TensorRoleCase>& info) {
                             return std::string(info.param.name);
                         });

// ============================================================================
// Declined: rules relating x to y
// ============================================================================

TEST(TestMiopenReductionChecks, DeclinesMixedDataTypes)
{
    ReductionGraphSpec floatToHalf;
    floatToHalf.yDataType = DataType::HALF;
    EXPECT_TRUE(
        isDeclinedWith(floatToHalf, "input and output tensors must have the same data type"));

    ReductionGraphSpec halfToFloat;
    halfToFloat.dataType = DataType::HALF;
    halfToFloat.yDataType = DataType::FLOAT;
    EXPECT_TRUE(
        isDeclinedWith(halfToFloat, "input and output tensors must have the same data type"));
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

TEST(TestMiopenReductionChecks, DeclinesOutputDimAboveInputDim)
{
    EXPECT_TRUE(
        isDeclinedWith(ReductionGraphSpec::packed(ReductionMode::ADD, {2, 3, 4, 5}, {2, 3, 8, 1}),
                       "output dim at axis 2 must be 1 or equal to the input dim"));
}

TEST(TestMiopenReductionChecks, DeclinesWhenNoAxisIsReduced)
{
    EXPECT_TRUE(
        isDeclinedWith(ReductionGraphSpec::packed(ReductionMode::ADD, {2, 3, 4, 5}, {2, 3, 4, 5}),
                       "no axis is reduced"));
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
}

TEST(TestMiopenReductionChecks, ToMiopenReduceTensorOpReturnsNulloptForUnknownMode)
{
    EXPECT_EQ(reduction_applicability::toMiopenReduceTensorOp(static_cast<ReductionMode>(42)),
              std::nullopt);
}
