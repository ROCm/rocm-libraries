// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <limits>
#include <optional>
#include <string>
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

TEST(TestPointwiseAddGraphMatcher, RetainsSingletonStrideAndDtypeAdmission)
{
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                                   data_objects::DataType::INT32,
                                                   {1, 1, 1, 1},
                                                   std::nullopt,
                                                   true,
                                                   std::vector<int64_t>{0, 0, 0, 0}));
    EXPECT_TRUE(matches(fixture.context()));
    EXPECT_FALSE(matchesKernel(POINTWISE_ADD, fixture.context(), makeKernel(64, "FLOAT")));
}

TEST(TestPointwiseAddGraphMatcher, RejectsScalarAndVirtualStateOnEveryOperand)
{
    // The baseline must match, or the refusals below prove nothing.
    EXPECT_TRUE(matches(GraphFixture(buildPointwiseGraph()).context()));

    for(size_t operand = 0; operand < 3; ++operand)
    {
        for(int state = 0; state < 4; ++state)
        {
            const GraphFixture fixture(transformGraph(buildPointwiseGraph(), [=](auto& graph) {
                auto& tensor = *graph.tensors[operand];
                tensor.virtual_ = state == 0;
                tensor.is_runtime_pass_by_value = state == 1 || state == 3;
                if(state >= 2)
                {
                    tensor.value.Set(data_objects::Float32Value(2.0f));
                }
            }));
            EXPECT_FALSE(matches(fixture.context())) << operand << ":" << state;
        }
    }
}

TEST(TestPointwiseAddBinding, PublishesActualNodeScalarsWithoutTensorEdgesOrGraphDefaults)
{
    const GraphFixture fixture(transformGraph(buildPointwiseGraph(), [](auto& graph) {
        graph.compute_data_type = data_objects::DataType::HALF;
        graph.is_override_shape_enabled = true;
        graph.nodes[0]->compute_data_type = data_objects::DataType::DOUBLE;
        auto* attributes = graph.nodes[0]->attributes.AsPointwiseAttributes();
        attributes->relu_lower_clip = 0.0f;
        attributes->relu_upper_clip = 6.0f;
        attributes->relu_lower_clip_slope = 0.25f;
        attributes->axis_tensor_uid = 0;
        attributes->swish_beta = 1.0f;
        attributes->elu_alpha = 2.0f;
        attributes->softplus_beta = 3.0f;
    }));
    const auto bound = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(std::get<std::string>(bound->at("pointwise.operation")), "ADD");
    EXPECT_EQ(std::get<std::string>(bound->at("pointwise.compute_data_type")), "DOUBLE");
    EXPECT_EQ(std::get<double>(bound->at("pointwise.relu_lower_clip")), 0.0);
    EXPECT_EQ(std::get<double>(bound->at("pointwise.relu_upper_clip")), 6.0);
    EXPECT_EQ(std::get<double>(bound->at("pointwise.relu_lower_clip_slope")), 0.25);
    EXPECT_EQ(std::get<double>(bound->at("pointwise.swish_beta")), 1.0);
    EXPECT_EQ(std::get<double>(bound->at("pointwise.elu_alpha")), 2.0);
    EXPECT_EQ(std::get<double>(bound->at("pointwise.softplus_beta")), 3.0);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, "pointwise.axis_tensor_uid"), 0);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, "graph.node_count"), 1);
    EXPECT_TRUE(std::get<bool>(bound->at("graph.is_override_shape_enabled")));
    EXPECT_EQ(bound->count("pointwise.in_0_tensor_uid"), 0U);
    EXPECT_EQ(bound->count("pointwise.out_0_tensor_uid"), 0U);
}

TEST(TestPointwiseAddBinding, RejectsInvalidOptionalNodeScalarRatherThanOmittingIt)
{
    const GraphFixture fixture(transformGraph(buildPointwiseGraph(), [](auto& graph) {
        graph.nodes[0]->attributes.AsPointwiseAttributes()->swish_beta
            = std::numeric_limits<float>::quiet_NaN();
    }));
    EXPECT_FALSE(matches(fixture.context()));
}

TEST(TestPointwiseAddBinding, ResidentFactsRetainUidZeroAndSubtractionOrderWithoutRawGraphReads)
{
    const auto bound = [] {
        const GraphFixture fixture(
            transformGraph(buildPointwiseGraph(data_objects::PointwiseMode::SUB), [](auto& graph) {
                graph.tensors[0]->uid = 0;
                auto* attributes = graph.nodes[0]->attributes.AsPointwiseAttributes();
                attributes->in_0_tensor_uid = INPUT_B_UID;
                attributes->in_1_tensor_uid = 0;
            }));
        return matchesGraph(POINTWISE_SUB, fixture.context());
    }();
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, "input_a"), INPUT_B_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, "input_a.uid"), INPUT_B_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, "input_b"), 0);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, "input_b.uid"), 0);
    EXPECT_EQ(std::get<std::vector<int64_t>>(bound->at("input_b.stride_order")),
              (std::vector<int64_t>{3, 2, 1, 0}));

    // Matchers must use the saved binding, not the different graph supplied here.
    const GraphFixture other(
        buildPointwiseGraph(data_objects::PointwiseMode::MUL, data_objects::DataType::HALF));
    EXPECT_TRUE(matchesOperation(POINTWISE_SUB, other.context(), *bound));
    EXPECT_FALSE(matchesOperation(POINTWISE_MUL, other.context(), *bound));
    EXPECT_TRUE(kernelMatcher(POINTWISE_SUB)(other.context(), *bound, makeKernel(64, "FLOAT")));
    EXPECT_FALSE(kernelMatcher(POINTWISE_SUB)(other.context(), *bound, makeKernel(64, "HALF")));
    EXPECT_EQ(bound->count("pointwise.swish_beta"), 0U);
    for(const auto* root : {"input_a", "input_b", "output"})
    {
        EXPECT_EQ(bound->count(std::string(root) + ".value_f32"), 0U);
    }
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

    const auto addBound = matchesGraph(POINTWISE_ADD, add.context());
    const auto mulBound = matchesGraph(POINTWISE_MUL, mul.context());
    ASSERT_TRUE(addBound.has_value());
    ASSERT_TRUE(mulBound.has_value());
    EXPECT_TRUE(matchesOperation(POINTWISE_ADD, add.context(), *addBound));
    EXPECT_FALSE(matchesOperation(POINTWISE_ADD, mul.context(), *mulBound));

    EXPECT_TRUE(matchesOperation(POINTWISE_MUL, mul.context(), *mulBound));
    EXPECT_FALSE(matchesOperation(POINTWISE_MUL, add.context(), *addBound));
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
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, "input_a.uid"), INPUT_A_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, "input_b.uid"), INPUT_B_UID);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, "output.uid"), OUTPUT_UID);
    for(const auto* root : {"input_a", "input_b", "output"})
    {
        EXPECT_EQ(bound->count(std::string(root) + ".value_f32"), 0U);
    }
}

TEST(TestPointwiseAddBinding, ARejectedGraphBindsNothingToDispatchFrom)
{
    const GraphFixture fixture(buildTwoNodePointwiseGraph());

    // A refused graph yields no token map at all, so a later pack has nothing stale to
    // read.
    EXPECT_FALSE(matchesGraph(POINTWISE_ADD, fixture.context()).has_value());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
