// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleQuantize.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleQuantizePlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestBlockScaleQuantizePlan, ExecutePlan)
{
    auto builder = createValidBlockScaleQuantizeGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();

    // Create two tensor bundles with same data for plan vs direct comparison
    const unsigned int seed = getGlobalTestSeed();
    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    // Fill input tensor with random values
    planBundle.getTensor(1).fillTensorWithRandomValues(0.0f, 1.0f, seed);
    directBundle.getTensor(1).fillTensorWithRandomValues(0.0f, 1.0f, seed);

    const auto* nodeAttributes = node.attributes_as_BlockScaleQuantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    // Execute via plan
    BlockScaleQuantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                    nodeAttributes->block_size(),
                                    nodeAttributes->axis());

    // Direct execution for reference
    auto directXTensor
        = createShallowTensor<float>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directYTensor
        = createShallowTensor<float>(params.yTensor, directBundle.getTensor(2).rawHostData());
    auto directScaleTensor
        = createShallowTensor<float>(params.scaleTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleQuantize::quantize(
        *directXTensor, *directYTensor, *directScaleTensor, params.blockSize, params.axis);

    // Plan execution
    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleQuantizePlan<float, float, float, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(2), planBundle.getTensor(2)));
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

TEST(TestBlockScaleQuantizePlan, ExecutePlanWithoutAxis)
{
    auto builder = createValidBlockScaleQuantizeGraph({2, 64, 32, 32},
                                                      {65536, 1024, 32, 1},
                                                      {2, 64, 32, 1},
                                                      {2048, 32, 1, 1},
                                                      32,
                                                      DataType::FLOAT,
                                                      DataType::FLOAT,
                                                      DataType::FLOAT,
                                                      DataType::FLOAT,
                                                      std::nullopt);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();

    // Create two tensor bundles with same data for plan vs direct comparison
    const unsigned int seed = getGlobalTestSeed();
    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    // Fill input tensor with random values
    planBundle.getTensor(1).fillTensorWithRandomValues(0.0f, 1.0f, seed);
    directBundle.getTensor(1).fillTensorWithRandomValues(0.0f, 1.0f, seed);

    const auto* nodeAttributes = node.attributes_as_BlockScaleQuantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    // Execute via plan
    BlockScaleQuantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                    nodeAttributes->block_size(),
                                    nodeAttributes->axis());

    // Direct execution for reference
    auto directXTensor
        = createShallowTensor<float>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directYTensor
        = createShallowTensor<float>(params.yTensor, directBundle.getTensor(2).rawHostData());
    auto directScaleTensor
        = createShallowTensor<float>(params.scaleTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleQuantize::quantize(
        *directXTensor, *directYTensor, *directScaleTensor, params.blockSize, params.axis);

    // Plan execution
    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleQuantizePlan<float, float, float, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(2), planBundle.getTensor(2)));
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

TEST(TestBlockScaleQuantizePlanBuilder, PlanConstruction)
{
    auto builder = createValidBlockScaleQuantizeGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const BlockScaleQuantizePlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<BlockScaleQuantizePlan<float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBlockScaleQuantizePlanBuilder, IsApplicable)
{
    auto builder = createValidBlockScaleQuantizeGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const BlockScaleQuantizePlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const BlockScaleQuantizePlanBuilder<DataType::FLOAT,
                                        DataType::HALF,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(
        badTypesPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // MX combinations: FLOAT input with E8M0 scale and FP8_E4M3 output
    auto mxBuilder = createValidBlockScaleQuantizeGraph({2, 64, 32, 32},
                                                        {65536, 1024, 32, 1},
                                                        {2, 2, 32, 32},
                                                        {2048, 1024, 32, 1},
                                                        32,
                                                        DataType::HALF,
                                                        DataType::FP8_E4M3,
                                                        DataType::FP8_E8M0,
                                                        DataType::FLOAT);
    const GraphWrapper mxGraphWrapper(mxBuilder.GetBufferPointer(), mxBuilder.GetSize());

    const BlockScaleQuantizePlanBuilder<DataType::HALF,
                                        DataType::FP8_E4M3,
                                        DataType::FP8_E8M0,
                                        DataType::FLOAT>
        floatE4M3Builder;
    EXPECT_TRUE(
        floatE4M3Builder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));

    const BlockScaleQuantizePlanBuilder<DataType::HALF,
                                        DataType::FP8_E5M2,
                                        DataType::FP8_E8M0,
                                        DataType::FLOAT>
        floatE5M2Builder;
    EXPECT_FALSE(
        floatE5M2Builder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));
}

// ============================================================================
// MX plan typed tests: all narrow types with fp8_e8m0 scale
// ============================================================================

template <DataType InputDT, DataType OutputDT, DataType ScaleDT>
struct MxPlanConfig
{
    static constexpr auto INPUT_DATA_TYPE = InputDT;
    static constexpr auto OUTPUT_DATA_TYPE = OutputDT;
    static constexpr auto SCALE_DATA_TYPE = ScaleDT;
    using InputType = DataTypeToNative<InputDT>;
    using OutputType = DataTypeToNative<OutputDT>;
    using ScaleType = DataTypeToNative<ScaleDT>;
};

using MxPlanTypes
    = ::testing::Types<MxPlanConfig<DataType::FLOAT, DataType::FP8_E4M3, DataType::FP8_E8M0>,
                       MxPlanConfig<DataType::HALF, DataType::FP8_E4M3, DataType::FP8_E8M0>,
                       MxPlanConfig<DataType::FLOAT, DataType::FP8_E5M2, DataType::FP8_E8M0>,
                       MxPlanConfig<DataType::HALF, DataType::FP8_E5M2, DataType::FP8_E8M0>,
                       MxPlanConfig<DataType::FLOAT, DataType::FP4_E2M1, DataType::FP8_E8M0>,
                       MxPlanConfig<DataType::HALF, DataType::FP4_E2M1, DataType::FP8_E8M0>,
                       MxPlanConfig<DataType::FLOAT, DataType::FP6_E2M3, DataType::FP8_E8M0>,
                       MxPlanConfig<DataType::HALF, DataType::FP6_E2M3, DataType::FP8_E8M0>,
                       MxPlanConfig<DataType::FLOAT, DataType::FP6_E3M2, DataType::FP8_E8M0>,
                       MxPlanConfig<DataType::HALF, DataType::FP6_E3M2, DataType::FP8_E8M0>>;

template <class T>
class BlockScaleQuantizeMxPlan : public ::testing::Test
{
};

TYPED_TEST_SUITE(BlockScaleQuantizeMxPlan, MxPlanTypes, );

TYPED_TEST(BlockScaleQuantizeMxPlan, ExecutePlan)
{
    using namespace hipdnn_data_sdk::types;
    using Config = TypeParam;
    using InputType = typename Config::InputType;
    using OutputType = typename Config::OutputType;
    using ScaleType = typename Config::ScaleType;

    auto builder = createValidBlockScaleQuantizeGraph({2, 64, 32, 32},
                                                      {65536, 1024, 32, 1},
                                                      {2, 2, 32, 32},
                                                      {2048, 1024, 32, 1},
                                                      32,
                                                      Config::INPUT_DATA_TYPE,
                                                      Config::OUTPUT_DATA_TYPE,
                                                      Config::SCALE_DATA_TYPE);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();
    const auto* nodeAttributes = node.attributes_as_BlockScaleQuantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    const unsigned int seed = getGlobalTestSeed();
    planBundle.getTensor(1).fillTensorWithRandomValues(1.0f, 2.0f, seed);
    directBundle.getTensor(1).fillTensorWithRandomValues(1.0f, 2.0f, seed);

    BlockScaleQuantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                    nodeAttributes->block_size(),
                                    nodeAttributes->axis());

    // Direct reference execution
    auto directXTensor
        = createShallowTensor<InputType>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directYTensor
        = createShallowTensor<OutputType>(params.yTensor, directBundle.getTensor(2).rawHostData());
    auto directScaleTensor = createShallowTensor<ScaleType>(
        params.scaleTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleQuantize::quantize(
        *directXTensor, *directYTensor, *directScaleTensor, params.blockSize, params.axis);

    // Plan execution
    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleQuantizePlan<InputType, OutputType, ScaleType, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<OutputType> cpuRefOutputValidation(tolerance, tolerance);
    const CpuFpReferenceValidation<ScaleType> cpuRefScaleValidation(tolerance, tolerance);

    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(2), planBundle.getTensor(2)));
    EXPECT_TRUE(cpuRefScaleValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

// ============================================================================
// Non-float IO plan typed tests with float scale
// ============================================================================

using NonFloatIOPlanTypes
    = ::testing::Types<MxPlanConfig<DataType::HALF, DataType::HALF, DataType::FLOAT>,
                       MxPlanConfig<DataType::HALF, DataType::BFLOAT16, DataType::FLOAT>,
                       MxPlanConfig<DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT>,
                       MxPlanConfig<DataType::BFLOAT16, DataType::HALF, DataType::FLOAT>>;

template <class T>
class BlockScaleQuantizeNonFloatIOPlan : public ::testing::Test
{
};

TYPED_TEST_SUITE(BlockScaleQuantizeNonFloatIOPlan, NonFloatIOPlanTypes, );

TYPED_TEST(BlockScaleQuantizeNonFloatIOPlan, ExecutePlan)
{
    using namespace hipdnn_data_sdk::types;
    using Config = TypeParam;
    using InputType = typename Config::InputType;
    using OutputType = typename Config::OutputType;
    using ScaleType = typename Config::ScaleType;

    auto builder = createValidBlockScaleQuantizeGraph({2, 64, 32, 32},
                                                      {65536, 1024, 32, 1},
                                                      {2, 2, 32, 32},
                                                      {2048, 1024, 32, 1},
                                                      32,
                                                      Config::INPUT_DATA_TYPE,
                                                      Config::OUTPUT_DATA_TYPE,
                                                      Config::SCALE_DATA_TYPE);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();
    const auto* nodeAttributes = node.attributes_as_BlockScaleQuantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    const unsigned int seed = getGlobalTestSeed();
    planBundle.getTensor(1).fillTensorWithRandomValues(1.0f, 2.0f, seed);
    directBundle.getTensor(1).fillTensorWithRandomValues(1.0f, 2.0f, seed);

    BlockScaleQuantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                    nodeAttributes->block_size(),
                                    nodeAttributes->axis());

    // Direct reference execution
    auto directXTensor
        = createShallowTensor<InputType>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directYTensor
        = createShallowTensor<OutputType>(params.yTensor, directBundle.getTensor(2).rawHostData());
    auto directScaleTensor = createShallowTensor<ScaleType>(
        params.scaleTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleQuantize::quantize(
        *directXTensor, *directYTensor, *directScaleTensor, params.blockSize, params.axis);

    // Plan execution
    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleQuantizePlan<InputType, OutputType, ScaleType, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<OutputType> cpuRefOutputValidation(tolerance, tolerance);
    const CpuFpReferenceValidation<ScaleType> cpuRefScaleValidation(tolerance, tolerance);

    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(2), planBundle.getTensor(2)));
    EXPECT_TRUE(cpuRefScaleValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

// ============================================================================
// MX IsApplicable typed tests: narrow types with fp8_e8m0 scale
// ============================================================================

template <DataType CorrectDT, DataType WrongDT>
struct MxIsApplicableConfig
{
    static constexpr auto CORRECT_DATA_TYPE = CorrectDT;
    static constexpr auto WRONG_DATA_TYPE = WrongDT;
};

using MxIsApplicableTypes
    = ::testing::Types<MxIsApplicableConfig<DataType::FP4_E2M1, DataType::FP6_E2M3>,
                       MxIsApplicableConfig<DataType::FP6_E2M3, DataType::FP6_E3M2>,
                       MxIsApplicableConfig<DataType::FP6_E3M2, DataType::FP4_E2M1>>;

template <class T>
class BlockScaleQuantizeMxIsApplicable : public ::testing::Test
{
};

TYPED_TEST_SUITE(BlockScaleQuantizeMxIsApplicable, MxIsApplicableTypes, );

TYPED_TEST(BlockScaleQuantizeMxIsApplicable, MatchingTypeIsApplicable)
{
    using Config = TypeParam;
    auto mxBuilder = createValidBlockScaleQuantizeGraph({2, 64, 32, 32},
                                                        {65536, 1024, 32, 1},
                                                        {2, 2, 32, 32},
                                                        {2048, 1024, 32, 1},
                                                        32,
                                                        DataType::FLOAT,
                                                        Config::CORRECT_DATA_TYPE,
                                                        DataType::FP8_E8M0,
                                                        DataType::FLOAT);
    const GraphWrapper mxGraphWrapper(mxBuilder.GetBufferPointer(), mxBuilder.GetSize());

    const BlockScaleQuantizePlanBuilder<DataType::FLOAT,
                                        Config::CORRECT_DATA_TYPE,
                                        DataType::FP8_E8M0,
                                        DataType::FLOAT>
        matchingBuilder;
    EXPECT_TRUE(
        matchingBuilder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));

    const BlockScaleQuantizePlanBuilder<DataType::FLOAT,
                                        Config::WRONG_DATA_TYPE,
                                        DataType::FP8_E8M0,
                                        DataType::FLOAT>
        wrongTypeBuilder;
    EXPECT_FALSE(
        wrongTypeBuilder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));
}
