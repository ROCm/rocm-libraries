// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestPointwiseAddMatchers.cpp
 * @brief The pack's two matcher shapes: what each accepts, and what each refuses.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::MatchContext;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

bool matches(const MatchContext& context)
{
    return matchesGraph(POINTWISE_ADD, context).has_value();
}

/// The bound value at @p token as a string, or nullopt when the token is absent or holds
/// something else. The mirror of tryGetBoundInt, which the SDK ships but has no string
/// twin.
std::optional<std::string> boundString(const BoundTokens& bound, std::string_view token)
{
    const auto entry = bound.find(std::string(token));
    if(entry == bound.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<std::string>(&entry->second);
    return value == nullptr ? std::nullopt : std::make_optional(*value);
}

using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

// Graph-scoped matcher: acceptances

TEST(TestPointwiseAddGraphMatcher, AcceptsASingleElementFloatAdd)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_TRUE(matches(fixture.context()));
}

TEST(TestPointwiseAddGraphMatcher, AcceptsAHalfPrecisionAdd)
{
    // Graph-level gate is dtype-agnostic; the kernel-scoped matcher pins dtype.
    const GraphFixture fixture(
        buildPointwiseGraph(data_objects::PointwiseMode::ADD, data_objects::DataType::HALF));

    EXPECT_TRUE(matches(fixture.context()));
}

TEST(TestPointwiseAddGraphMatcher, AcceptsTheUpperSupportedRank)
{
    const GraphFixture fixture(buildPointwiseGraph(
        data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1, 1, 1, 1, 1}));

    EXPECT_TRUE(matches(fixture.context()));
}

// Graph-scoped matcher: refusals

/// Builder is a plain function pointer, not std::function: FlatBufferBuilder is move-only.
struct GraphMatcherRefusalCase
{
    std::string name;
    flatbuffers::FlatBufferBuilder (*buildGraph)();
};

class TestPointwiseAddGraphMatcherRefusal : public ::testing::TestWithParam<GraphMatcherRefusalCase>
{
};

TEST_P(TestPointwiseAddGraphMatcherRefusal, Refuses)
{
    const GraphFixture fixture(GetParam().buildGraph());

    EXPECT_FALSE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestPointwiseAddGraphMatcherRefusal,
    ::testing::ValuesIn(std::vector<GraphMatcherRefusalCase>{
        {"MultiElementTensors",
         // The kernel writes only element 0; a larger tensor leaves the rest unwritten.
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1, 1, 2, 2});
         }},
        {"ATensorWithNoStrides",
         // The layout classifier dereferences strides(); applicability runs before validation.
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/std::nullopt,
                                        /*includeThirdOperand=*/false,
                                        /*danglingInputBUid=*/std::nullopt,
                                        /*inputAVirtual=*/false,
                                        /*inputAIsRuntimePassByValue=*/false,
                                        /*outputVirtual=*/false,
                                        /*omitStrides=*/true);
         }},
        {"DimsWhoseProductIsOneButAreNotAllOne",
         // {-1,-1,1,1} multiplies to 1; the kernel indexes element 0 only.
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {-1, -1, 1, 1});
         }},
        {"ARankTheDispatchPathCannotServe",
         []() {
             return buildPointwiseGraph(
                 data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1});
         }},
        {"AStrideOrderTheDispatchPathCannotClassify",
         // Layout derives from stride order; only NCHW/NHWC classify.
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::vector<int64_t>{8, 2, 4, 1});
         }},
        {"AUnaryPointwise",
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/false);
         }},
        {"AMultiNodeGraph", []() { return buildTwoNodePointwiseGraph(); }},
        {"CrossOperandDtypeMismatch",
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/data_objects::DataType::HALF);
         }},
        {"AThirdOperand",
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/std::nullopt,
                                        /*includeThirdOperand=*/true);
         }},
        {"ADanglingTensorUid",
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/std::nullopt,
                                        /*includeThirdOperand=*/false,
                                        /*danglingInputBUid=*/DEFAULT_DANGLING_UID);
         }},
        {"AVirtualOperand",
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/std::nullopt,
                                        /*includeThirdOperand=*/false,
                                        /*danglingInputBUid=*/std::nullopt,
                                        /*inputAVirtual=*/true);
         }},
        {"ARuntimePassByValueOperand",
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/std::nullopt,
                                        /*includeThirdOperand=*/false,
                                        /*danglingInputBUid=*/std::nullopt,
                                        /*inputAVirtual=*/false,
                                        /*inputAIsRuntimePassByValue=*/true);
         }},
        {"AVirtualOutput",
         // Checks every operand, not just input A: a virtual output has no buffer to
         // resolve at launch.
         []() {
             return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                        data_objects::DataType::FLOAT,
                                        {1, 1, 1, 1},
                                        std::nullopt,
                                        /*binary=*/true,
                                        /*explicitStrides=*/std::nullopt,
                                        /*inputBDataType=*/std::nullopt,
                                        /*includeThirdOperand=*/false,
                                        /*danglingInputBUid=*/std::nullopt,
                                        /*inputAVirtual=*/false,
                                        /*inputAIsRuntimePassByValue=*/false,
                                        /*outputVirtual=*/true);
         }},
    }),
    [](const ::testing::TestParamInfo<GraphMatcherRefusalCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Graph-scoped operation matchers: the one fact separating this engine's packs
// ---------------------------------------------------------------------------

/// The engine's graph match deliberately admits any operation, so these are what stop a
/// multiplication reaching an add kernel. Asserted for both packs against both graphs,
/// because "each accepts its own" and "each refuses the other's" are separate claims and
/// a criterion that returned true unconditionally would satisfy only the first.
TEST(TestPointwiseOperationMatchers, EachPackAdmitsOnlyItsOwnOperation)
{
    const GraphFixture add(buildPointwiseGraph(data_objects::PointwiseMode::ADD));
    const GraphFixture mul(buildPointwiseGraph(data_objects::PointwiseMode::MUL));

    const BoundTokens bound;
    EXPECT_TRUE(matchesOperation(POINTWISE_ADD, add.context(), bound));
    EXPECT_FALSE(matchesOperation(POINTWISE_ADD, mul.context(), bound));

    EXPECT_TRUE(matchesOperation(POINTWISE_MUL, mul.context(), bound));
    EXPECT_FALSE(matchesOperation(POINTWISE_MUL, add.context(), bound));
}

/// The shared half of the split, stated as its own claim: the expensive checks do not
/// re-run per pack, so they must not encode an operation.
TEST(TestPointwiseGraphMatcher, AdmitsEveryOperationItsPacksBetweenThemServe)
{
    const GraphFixture add(buildPointwiseGraph(data_objects::PointwiseMode::ADD));
    const GraphFixture mul(buildPointwiseGraph(data_objects::PointwiseMode::MUL));

    EXPECT_TRUE(matchesGraph(POINTWISE_ADD, add.context()).has_value());
    EXPECT_TRUE(matchesGraph(POINTWISE_ADD, mul.context()).has_value());
}

// ---------------------------------------------------------------------------
// Kernel-scoped matcher

TEST(TestPointwiseAddKernelMatcher, AcceptsAKernelWhoseDtypeMatchesTheGraph)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_TRUE(matchesKernel(POINTWISE_ADD, fixture.context(), makeKernel(64, "FLOAT")));
}

TEST(TestPointwiseAddKernelMatcher, RefusesAKernelBakedForAnotherDtype)
{
    // An f16 kernel handed f32 operands does not fail; it returns wrong numbers.
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_FALSE(matchesKernel(POINTWISE_ADD, fixture.context(), makeKernel(64, "HALF")));
}

TEST(TestPointwiseAddKernelMatcher, AcceptsAHalfKernelForAHalfGraph)
{
    const GraphFixture fixture(
        buildPointwiseGraph(data_objects::PointwiseMode::ADD, data_objects::DataType::HALF));

    EXPECT_TRUE(matchesKernel(POINTWISE_ADD, fixture.context(), makeKernel(64, "HALF")));
}

TEST(TestPointwiseAddKernelMatcher, IgnoresBlockSizeWhichTheGraphDoesNotConstrain)
{
    // block_size ranks kernels but never gates applicability.
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_TRUE(matchesKernel(POINTWISE_ADD, fixture.context(), makeKernel(64, "FLOAT")));
    EXPECT_TRUE(matchesKernel(POINTWISE_ADD, fixture.context(), makeKernel(256, "FLOAT")));
}

// Score and binding

TEST(TestPointwiseAddScore, PrefersTheLargerBlockSize)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_GT(scoreKernel(POINTWISE_ADD, fixture.context(), makeKernel(256, "FLOAT")),
              scoreKernel(POINTWISE_ADD, fixture.context(), makeKernel(64, "FLOAT")));
}

TEST(TestPointwiseAddBinding, TheGraphMatchBindsTheOperandUidsItResolved)
{
    const GraphFixture fixture(buildPointwiseGraph());

    const auto bound = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value());

    // Asserted by token name: the contract a descriptor's dispatch formulas reference.
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, POINTWISE_ADD.inputAToken),
              INPUT_A_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, POINTWISE_ADD.inputBToken),
              INPUT_B_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, POINTWISE_ADD.outputToken),
              OUTPUT_UID);
}

TEST(TestPointwiseAddBinding, ARejectedGraphBindsNothingToDispatchFrom)
{
    const GraphFixture fixture(buildTwoNodePointwiseGraph());

    // A refused graph yields no token map at all, so a later pack has nothing stale to
    // read.
    EXPECT_FALSE(matchesGraph(POINTWISE_ADD, fixture.context()).has_value());
}

// ---------------------------------------------------------------------------
// Problem binding: the dims, dtypes and cost fields a UHD ranks on
// ---------------------------------------------------------------------------
//
// Token names are written out as literals rather than composed from the pack's
// constants: the string IS the contract a UHD's features_signature references, so a
// rename must fail here rather than quietly rename both sides at once.

TEST(TestPointwiseAddBinding, BindsEveryOperandDimPositionallyAndNoneItDoesNotHave)
{
    const GraphFixture fixture(buildPointwiseGraph());

    const auto bound = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value());

    for(const auto* root : {"pointwise.input_a", "pointwise.input_b", "pointwise.output"})
    {
        const std::string prefix(root);
        EXPECT_EQ(tryGetBoundInt(*bound, prefix + ".dims[0]"), 1);
        EXPECT_EQ(tryGetBoundInt(*bound, prefix + ".dims[1]"), 1);
        EXPECT_EQ(tryGetBoundInt(*bound, prefix + ".dims[2]"), 1);
        EXPECT_EQ(tryGetBoundInt(*bound, prefix + ".dims[3]"), 1);
        // The rank-4 graph has no fifth axis; publishing one would be inventing a dim.
        EXPECT_FALSE(tryGetBoundInt(*bound, prefix + ".dims[4]").has_value());
    }
}

TEST(TestPointwiseAddBinding, BindsTheFifthDimOfARankFiveGraph)
{
    const GraphFixture fixture(buildPointwiseGraph(
        data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1, 1, 1, 1, 1}));

    const auto bound = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value());

    // Dims are published from the tensor's own rank, not from a fixed count.
    EXPECT_EQ(tryGetBoundInt(*bound, "pointwise.input_a.dims[4]"), 1);
}

TEST(TestPointwiseAddBinding, BindsDtypeAsTheRuntimeSpellingNotTheFlatbufferEnumName)
{
    const GraphFixture floatFixture(buildPointwiseGraph());
    const GraphFixture doubleFixture(
        buildPointwiseGraph(data_objects::PointwiseMode::ADD, data_objects::DataType::DOUBLE));

    const auto floatBound = matchesGraph(POINTWISE_ADD, floatFixture.context());
    const auto doubleBound = matchesGraph(POINTWISE_ADD, doubleFixture.context());
    ASSERT_TRUE(floatBound.has_value());
    // This matcher gates no dtype -- only the kernel-scoped one does -- so the binding
    // must spell dtypes past the two the shipped kernels are compiled for.
    ASSERT_TRUE(doubleBound.has_value());

    // to_string(DataType)'s spelling, which is the vocabulary CategoricalEncoding.hpp
    // encodes. EnumNameDataType would answer "FLOAT"/"DOUBLE" and `float32`/`float64` are
    // the plausible near-misses; the encoder knows none of the four, so a wrong spelling
    // here costs the feature rather than warning.
    EXPECT_EQ(boundString(*floatBound, "pointwise.input_a.dtype"), "fp32");
    EXPECT_EQ(boundString(*floatBound, "pointwise.input_b.dtype"), "fp32");
    EXPECT_EQ(boundString(*floatBound, "pointwise.output.dtype"), "fp32");
    EXPECT_EQ(boundString(*doubleBound, "pointwise.input_a.dtype"), "fp64");

    // A string, never a pre-encoded number: the integer code space belongs to the feature
    // extractor, and freezing it inside the matcher would let the two drift apart.
    EXPECT_FALSE(tryGetBoundInt(*floatBound, "pointwise.input_a.dtype").has_value());
}

TEST(TestPointwiseAddBinding, BindsFlopsAndBytesForTheOneElementTheKernelTouches)
{
    const GraphFixture fixture(buildPointwiseGraph());

    const auto bound = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value());

    // By hand: the output is 1x1x1x1, so one element, and a binary add is one flop per
    // output element -- 1 flop. Bytes reads both inputs and writes the output, one fp32
    // element each: 3 * 4 = 12.
    EXPECT_EQ(tryGetBoundInt(*bound, "pointwise.flops"), 1);
    EXPECT_EQ(tryGetBoundInt(*bound, "pointwise.bytes"), 12);
}

TEST(TestPointwiseAddBinding, ByteCountFollowsTheOperandDtypeWidth)
{
    const GraphFixture fixture(
        buildPointwiseGraph(data_objects::PointwiseMode::ADD, data_objects::DataType::DOUBLE));

    const auto bound = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value());

    // The same three elements at 8 bytes each = 24: the width is read off each tensor,
    // not assumed to be the fp32 the shipped kernels happen to use.
    EXPECT_EQ(tryGetBoundInt(*bound, "pointwise.bytes"), 24);
    // flops is a pure shape count and must not move with precision.
    EXPECT_EQ(tryGetBoundInt(*bound, "pointwise.flops"), 1);
}

/// RFC 0019 §13.6: a cost field with no exact form must be an upper bound or absent,
/// never silently wrong. A sub-byte dtype has no per-element byte width at all -- its
/// footprint is a property of the packing -- so `bytes` is omitted rather than rounded up
/// to one byte, which would overstate an fp4 tensor by 2x.
TEST(TestPointwiseAddBinding, OmitsBytesForADtypeWithNoStatableElementWidth)
{
    const GraphFixture fixture(
        buildPointwiseGraph(data_objects::PointwiseMode::ADD, data_objects::DataType::FP4_E2M1));

    const auto bound = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_FALSE(tryGetBoundInt(*bound, "pointwise.bytes").has_value());
    // The dtype itself is still knowable, and flops does not depend on element width.
    EXPECT_EQ(boundString(*bound, "pointwise.input_a.dtype"), "fp4_e2m1");
    EXPECT_EQ(tryGetBoundInt(*bound, "pointwise.flops"), 1);
}

/// The other half of the same rule: an unrecognized dtype has no runtime spelling, and
/// binding to_string's "unknown" fallthrough would hand the encoder a category it
/// deliberately does not carry (CategoricalEncoding.hpp). Absent instead.
TEST(TestPointwiseAddBinding, OmitsDtypeAndBytesForAnUnsetDtype)
{
    const GraphFixture fixture(
        buildPointwiseGraph(data_objects::PointwiseMode::ADD, data_objects::DataType::UNSET));

    const auto bound = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_FALSE(boundString(*bound, "pointwise.input_a.dtype").has_value());
    EXPECT_FALSE(tryGetBoundInt(*bound, "pointwise.bytes").has_value());
    // Shape is unaffected by an unknown element type.
    EXPECT_EQ(tryGetBoundInt(*bound, "pointwise.input_a.dims[0]"), 1);
    EXPECT_EQ(tryGetBoundInt(*bound, "pointwise.flops"), 1);
}

/// RFC 0019 §13.6 requires `bytes` to sum each operand's OWN dtype width, and this pack
/// cannot demonstrate the difference: its kernel is compiled for a single element type,
/// so the matcher refuses any graph whose operands disagree (see the
/// CrossOperandDtypeMismatch refusal above) and no mixed-dtype graph ever reaches the
/// binding. The implementation reads each operand's dtype separately anyway, so a pack
/// that later admits mixed precision inherits the right sum. This test pins the reason
/// the stronger assertion is absent, so its absence is not read as an oversight.
TEST(TestPointwiseAddBinding, AMixedPrecisionGraphIsRefusedSoPerOperandWidthCannotBeObservedHere)
{
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                                   data_objects::DataType::FLOAT,
                                                   {1, 1, 1, 1},
                                                   std::nullopt,
                                                   /*binary=*/true,
                                                   /*explicitStrides=*/std::nullopt,
                                                   /*inputBDataType=*/data_objects::DataType::HALF));

    EXPECT_FALSE(matchesGraph(POINTWISE_ADD, fixture.context()).has_value());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
