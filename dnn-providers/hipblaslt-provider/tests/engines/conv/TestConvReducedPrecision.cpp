// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <HipblasltPlugin.hpp>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "common/ConvRepackFormulation.hpp"
#include "engines/conv/ConvGemmTestGraphs.hpp"
#include "engines/conv/ConvReducedPrecision.hpp"

/**
 * @file TestConvReducedPrecision.cpp
 * @brief The reduced-precision-reduction convolution engines' host-side rules: the switch that
 *        registers them, the one form they serve (reducedPrecisionGemmCalls), and one decline per
 *        gate of that form.
 */
namespace
{

using namespace hipblaslt_plugin;
using namespace hipblaslt_plugin::test;
using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;
using DataType = conv_data_objects::DataType;

/// The form's shape: a 1x1 fp16 backward weights of 4 channels-first images, which no single GEMM
/// reads exactly.
ConvGemmShape rprShape()
{
    ConvGemmShape shape;
    shape.n = 4;
    shape.dataType = DataType::HALF;
    return shape;
}

std::optional<ConvGemmPlanCalls> rprCalls(const ConvGemmShape& shape)
{
    const ConvGemmGraph graph(shape, ConvDirection::BACKWARD_WEIGHTS);
    const auto problem = parseConvProblem(graph.graph(), ConvDirection::BACKWARD_WEIGHTS);
    if(!problem.has_value())
    {
        return std::nullopt;
    }
    return reducedPrecisionGemmCalls(*problem);
}

std::vector<int64_t> engineIds()
{
    uint32_t count = 0;
    EXPECT_EQ(hipdnnEnginePluginGetAllEngineIdsImpl(nullptr, 0, &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    std::vector<int64_t> ids(count, 0);
    EXPECT_EQ(hipdnnEnginePluginGetAllEngineIdsImpl(ids.data(), count, &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    return ids;
}

bool hasEngine(const std::vector<int64_t>& ids, int64_t id)
{
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

TEST(TestConvReducedPrecision, SwitchReadsOnlyAffirmativeValues)
{
    ScopedEnvironmentVariableSetter env(CONV_RPR_ENV_NAME);
    EXPECT_FALSE(convReducedPrecisionEnginesEnabled());
    for(const auto* value : {"1", "true", "ON", " yes "})
    {
        env.setValue(value);
        EXPECT_TRUE(convReducedPrecisionEnginesEnabled()) << value;
    }
    for(const auto* value : {"0", "false", "off", "2", "enable"})
    {
        env.setValue(value);
        EXPECT_FALSE(convReducedPrecisionEnginesEnabled()) << value;
    }
}

TEST(TestConvReducedPrecision, NoEngineRegisteredWithoutTheSwitch)
{
    using namespace hipdnn_data_sdk::utilities;
    ScopedEnvironmentVariableSetter env(CONV_RPR_ENV_NAME);
    const auto unset = engineIds();
    EXPECT_FALSE(hasEngine(unset, ROCBLAS_CONV_RPR_ENGINE_ID));
    EXPECT_FALSE(hasEngine(unset, HIPBLASLT_CONV_RPR_ENGINE_ID));
#ifdef HIPBLASLTPROVIDER_ENABLE_CONV
    env.setValue("1");
    const auto set = engineIds();
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    EXPECT_TRUE(hasEngine(set, ROCBLAS_CONV_RPR_ENGINE_ID));
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
    EXPECT_TRUE(hasEngine(set, HIPBLASLT_CONV_RPR_ENGINE_ID));
#endif
#endif
}

TEST(TestConvReducedPrecision, OneByOneChannelsFirstHalfWeightsIsOneAccumulatingGemmPerImage)
{
    auto shape = rprShape();
    shape.groups = 2;
    const auto plan = rprCalls(shape);
    ASSERT_TRUE(plan.has_value());
    EXPECT_FALSE(plan->hasRepack());
    EXPECT_EQ(plan->workspaceBytes, 0u);
    ASSERT_EQ(plan->stages.size(), 1u);
    const auto& calls = plan->stages[0].calls;
    ASSERT_EQ(calls.size(), static_cast<size_t>(shape.n));
    for(size_t image = 0; image < calls.size(); ++image)
    {
        const auto& call = calls[image];
        // Into w itself, in fp16: each image after the first rounds the running sum again.
        EXPECT_EQ(call.d.uid, CONV_W_UID);
        EXPECT_EQ(call.d.dataType, DataType::HALF);
        EXPECT_EQ(call.accumulate, image > 0);
        EXPECT_EQ(call.batchCount, shape.groups);
        // A reduction over one image's pixels.
        EXPECT_EQ(call.k, 5 * 7);
    }
    // Consecutive images read consecutive images of x and dy.
    EXPECT_NE(calls[0].a.offset, calls[1].a.offset);
    EXPECT_NE(calls[0].b.offset, calls[1].b.offset);
}

TEST(TestConvReducedPrecision, ThreeDAndUnnestedImagesAreServed)
{
    auto threeD = rprShape();
    threeD.input = {3, 4, 5};
    threeD.filter = {1, 1, 1};
    threeD.prePadding = threeD.postPadding = {0, 0, 0};
    threeD.stride = threeD.dilation = {1, 1, 1};
    EXPECT_TRUE(rprCalls(threeD).has_value());
    auto padded = rprShape();
    padded.imagePadding = 3;
    EXPECT_TRUE(rprCalls(padded).has_value());
}

TEST(TestConvReducedPrecision, ServesHalfComputeAsFloatOnHalf)
{
    for(const auto compute : {DataType::UNSET, DataType::FLOAT, DataType::HALF})
    {
        auto shape = rprShape();
        shape.computeType = compute;
        EXPECT_TRUE(rprCalls(shape).has_value()) << conv_data_objects::EnumNameDataType(compute);
    }
    auto bf16Compute = rprShape();
    bf16Compute.computeType = DataType::BFLOAT16;
    EXPECT_FALSE(rprCalls(bf16Compute).has_value());
}

struct Decline
{
    std::string name;
    ConvGemmShape shape;
    ConvDirection direction = ConvDirection::BACKWARD_WEIGHTS;
};

std::vector<Decline> declines()
{
    std::vector<Decline> cases;
    const auto add = [&cases](const std::string& name, ConvGemmShape shape) {
        cases.push_back({name, std::move(shape)});
    };
    auto bf16 = rprShape();
    bf16.dataType = DataType::BFLOAT16;
    add("Bfloat16", bf16);
    auto fp32 = rprShape();
    fp32.dataType = DataType::FLOAT;
    add("Float", fp32);
    auto oneImage = rprShape();
    oneImage.n = 1;
    add("OneImageIsOneExactGemm", oneImage);
    auto last = rprShape();
    last.xChannelsLast = last.wChannelsLast = last.yChannelsLast = true;
    add("ChannelsLastIsOneExactGemm", last);
    auto filter = rprShape();
    filter.filter = {3, 3};
    add("NotOneByOne", filter);
    auto strided = rprShape();
    strided.input = {6, 8};
    strided.stride = {2, 2};
    add("Strided", strided);
    auto paddedSpatial = rprShape();
    paddedSpatial.prePadding = paddedSpatial.postPadding = {1, 1};
    add("Padded", paddedSpatial);
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        cases.push_back({direction == ConvDirection::FORWARD ? "Forward" : "BackwardData",
                         rprShape(),
                         direction});
    }
    return cases;
}

class TestConvReducedPrecisionDeclines : public ::testing::TestWithParam<Decline>
{
};

TEST_P(TestConvReducedPrecisionDeclines, OneGate)
{
    const auto& decline = GetParam();
    const ConvGemmGraph graph(decline.shape, decline.direction);
    const auto problem = parseConvProblem(graph.graph(), decline.direction);
    ASSERT_TRUE(problem.has_value()) << "the test graph is not a well-formed convolution";
    EXPECT_FALSE(reducedPrecisionGemmCalls(*problem).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestConvReducedPrecisionDeclines,
                         ::testing::ValuesIn(declines()),
                         [](const auto& info) { return info.param.name; });

} // namespace
