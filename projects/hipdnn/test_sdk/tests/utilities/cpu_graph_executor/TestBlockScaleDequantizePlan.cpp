// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleDequantize.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleDequantizePlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestBlockScaleDequantizePlan, ExecutePlan)
{
    auto builder = createValidBlockScaleDequantizeGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();

    // Create two tensor bundles with same data for plan vs direct comparison
    const unsigned int seed = getGlobalTestSeed();
    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    // Fill x and scale with same random data
    planBundle.getTensor(1).fillTensorWithRandomValues(0.0f, 1.0f, seed);
    planBundle.getTensor(2).fillTensorWithRandomValues(0.1f, 2.0f, seed);
    directBundle.getTensor(1).fillTensorWithRandomValues(0.0f, 1.0f, seed);
    directBundle.getTensor(2).fillTensorWithRandomValues(0.1f, 2.0f, seed);

    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    // Execute via plan
    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    // Direct execution for reference
    auto directXTensor
        = createShallowTensor<float>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor
        = createShallowTensor<float>(params.scaleTensor, directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<float>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(*directXTensor,
                                                   *directScaleTensor,
                                                   *directYTensor,
                                                   blockSize,
                                                   nodeAttributes->is_negative_scale());

    // Plan execution
    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<float, float, float, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

TEST(TestBlockScaleDequantizePlan, ExecutePlanNegativeScale)
{
    auto builder = createValidBlockScaleDequantizeGraph(
        {65536, 2048, 64, 1}, {2, 32, 32, 64}, DataType::FLOAT, DataType::FLOAT, true);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();

    const unsigned int seed = getGlobalTestSeed();
    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    planBundle.getTensor(1).fillTensorWithRandomValues(0.0f, 1.0f, seed);
    planBundle.getTensor(2).fillTensorWithRandomValues(0.1f, 2.0f, seed);
    directBundle.getTensor(1).fillTensorWithRandomValues(0.0f, 1.0f, seed);
    directBundle.getTensor(2).fillTensorWithRandomValues(0.1f, 2.0f, seed);

    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    ASSERT_TRUE(nodeAttributes->is_negative_scale());

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    auto directXTensor
        = createShallowTensor<float>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor
        = createShallowTensor<float>(params.scaleTensor, directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<float>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(
        *directXTensor, *directScaleTensor, *directYTensor, blockSize, true);

    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<float, float, float, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

TEST(TestBlockScaleDequantizePlanBuilder, PlanConstruction)
{
    auto builder = createValidBlockScaleDequantizeGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const BlockScaleDequantizePlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<BlockScaleDequantizePlan<float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBlockScaleDequantizePlanBuilder, IsApplicable)
{
    auto builder = createValidBlockScaleDequantizeGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const BlockScaleDequantizePlanBuilder<DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const BlockScaleDequantizePlanBuilder<DataType::HALF,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(
        badTypesPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // MX combinations: FP8 input with E8M0 scale
    auto mxBuilder = createValidBlockScaleDequantizeMxGraph(
        DataType::FP8_E4M3, DataType::FP8_E8M0, DataType::FLOAT);
    const GraphWrapper mxGraphWrapper(mxBuilder.GetBufferPointer(), mxBuilder.GetSize());

    const BlockScaleDequantizePlanBuilder<DataType::FP8_E4M3,
                                          DataType::FP8_E8M0,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        e4m3FloatBuilder;
    EXPECT_TRUE(
        e4m3FloatBuilder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));

    const BlockScaleDequantizePlanBuilder<DataType::FP8_E5M2,
                                          DataType::FP8_E8M0,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        e5m2FloatBuilder;
    EXPECT_FALSE(
        e5m2FloatBuilder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));
}

// ============================================================================
// MX plan typed tests: all narrow types with fp8_e8m0 scale
// ============================================================================

template <DataType XDT, DataType ScaleDT, DataType OutputDT>
struct MxPlanConfig
{
    static constexpr auto X_DATA_TYPE = XDT;
    static constexpr auto SCALE_DATA_TYPE = ScaleDT;
    static constexpr auto OUTPUT_DATA_TYPE = OutputDT;
    using XType = DataTypeToNative<XDT>;
    using ScaleType = DataTypeToNative<ScaleDT>;
    using OutputType = DataTypeToNative<OutputDT>;
};

using MxPlanTypes
    = ::testing::Types<MxPlanConfig<DataType::FP8_E4M3, DataType::FP8_E8M0, DataType::FLOAT>,
                       MxPlanConfig<DataType::FP8_E4M3, DataType::FP8_E8M0, DataType::HALF>,
                       MxPlanConfig<DataType::FP8_E5M2, DataType::FP8_E8M0, DataType::FLOAT>,
                       MxPlanConfig<DataType::FP8_E5M2, DataType::FP8_E8M0, DataType::HALF>,
                       MxPlanConfig<DataType::FP4_E2M1, DataType::FP8_E8M0, DataType::FLOAT>,
                       MxPlanConfig<DataType::FP4_E2M1, DataType::FP8_E8M0, DataType::HALF>,
                       MxPlanConfig<DataType::FP6_E2M3, DataType::FP8_E8M0, DataType::FLOAT>,
                       MxPlanConfig<DataType::FP6_E2M3, DataType::FP8_E8M0, DataType::HALF>,
                       MxPlanConfig<DataType::FP6_E3M2, DataType::FP8_E8M0, DataType::FLOAT>,
                       MxPlanConfig<DataType::FP6_E3M2, DataType::FP8_E8M0, DataType::HALF>>;

template <class T>
class BlockScaleDequantizeMxPlan : public ::testing::Test
{
};

TYPED_TEST_SUITE(BlockScaleDequantizeMxPlan, MxPlanTypes, );

TYPED_TEST(BlockScaleDequantizeMxPlan, ExecutePlan)
{
    using namespace hipdnn_data_sdk::types;
    using Config = TypeParam;
    using XType = typename Config::XType;
    using ScaleType = typename Config::ScaleType;
    using OutputType = typename Config::OutputType;

    auto builder = createValidBlockScaleDequantizeMxGraph(
        Config::X_DATA_TYPE, Config::SCALE_DATA_TYPE, Config::OUTPUT_DATA_TYPE);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();
    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    // Set x values via raw host data
    auto* planXData = static_cast<XType*>(planBundle.getTensor(1).rawHostData());
    planXData[0] = XType(1.0f);
    planXData[1] = XType(1.0f);
    planXData[2] = XType(2.0f);
    planXData[3] = XType(2.0f);

    auto* directXData = static_cast<XType*>(directBundle.getTensor(1).rawHostData());
    directXData[0] = XType(1.0f);
    directXData[1] = XType(1.0f);
    directXData[2] = XType(2.0f);
    directXData[3] = XType(2.0f);

    // Set scale values (fp8_e8m0): bits=127 => 1.0, bits=128 => 2.0
    auto* planScaleData = static_cast<ScaleType*>(planBundle.getTensor(2).rawHostData());
    planScaleData[0] = fp8_e8m0::from_bits(127);
    planScaleData[1] = fp8_e8m0::from_bits(128);

    auto* directScaleData = static_cast<ScaleType*>(directBundle.getTensor(2).rawHostData());
    directScaleData[0] = fp8_e8m0::from_bits(127);
    directScaleData[1] = fp8_e8m0::from_bits(128);

    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    // Direct reference execution
    auto directXTensor
        = createShallowTensor<XType>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor = createShallowTensor<ScaleType>(
        params.scaleTensor, directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<OutputType>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(
        *directXTensor, *directScaleTensor, *directYTensor, blockSize, false);

    // Plan execution
    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<XType, ScaleType, OutputType, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<OutputType> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

// ============================================================================
// Non-float input plan typed tests: half/bfloat16 input with float scale
// ============================================================================

using NonFloatInputPlanTypes
    = ::testing::Types<MxPlanConfig<DataType::HALF, DataType::FLOAT, DataType::FLOAT>,
                       MxPlanConfig<DataType::HALF, DataType::FLOAT, DataType::HALF>,
                       MxPlanConfig<DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT>,
                       MxPlanConfig<DataType::BFLOAT16, DataType::FLOAT, DataType::BFLOAT16>>;

template <class T>
class BlockScaleDequantizeNonFloatInputPlan : public ::testing::Test
{
};

TYPED_TEST_SUITE(BlockScaleDequantizeNonFloatInputPlan, NonFloatInputPlanTypes, );

TYPED_TEST(BlockScaleDequantizeNonFloatInputPlan, ExecutePlan)
{
    using namespace hipdnn_data_sdk::types;
    using Config = TypeParam;
    using XType = typename Config::XType;
    using ScaleType = typename Config::ScaleType;
    using OutputType = typename Config::OutputType;

    auto builder = createValidBlockScaleDequantizeMxGraph(
        Config::X_DATA_TYPE, Config::SCALE_DATA_TYPE, Config::OUTPUT_DATA_TYPE);
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();
    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    // Set x values
    auto* planXData = static_cast<XType*>(planBundle.getTensor(1).rawHostData());
    planXData[0] = XType(1.0f);
    planXData[1] = XType(1.0f);
    planXData[2] = XType(2.0f);
    planXData[3] = XType(2.0f);

    auto* directXData = static_cast<XType*>(directBundle.getTensor(1).rawHostData());
    directXData[0] = XType(1.0f);
    directXData[1] = XType(1.0f);
    directXData[2] = XType(2.0f);
    directXData[3] = XType(2.0f);

    // Set float scale values: scale[0] = 1.5, scale[1] = 2.0
    auto* planScaleData = static_cast<ScaleType*>(planBundle.getTensor(2).rawHostData());
    planScaleData[0] = ScaleType(1.5f);
    planScaleData[1] = ScaleType(2.0f);

    auto* directScaleData = static_cast<ScaleType*>(directBundle.getTensor(2).rawHostData());
    directScaleData[0] = ScaleType(1.5f);
    directScaleData[1] = ScaleType(2.0f);

    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    auto directXTensor
        = createShallowTensor<XType>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor = createShallowTensor<ScaleType>(
        params.scaleTensor, directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<OutputType>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(
        *directXTensor, *directScaleTensor, *directYTensor, blockSize, false);

    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<XType, ScaleType, OutputType, float> plan(std::move(params));
    plan.execute(variantPack);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<OutputType> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
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
class BlockScaleDequantizeMxIsApplicable : public ::testing::Test
{
};

TYPED_TEST_SUITE(BlockScaleDequantizeMxIsApplicable, MxIsApplicableTypes, );

TYPED_TEST(BlockScaleDequantizeMxIsApplicable, MatchingTypeIsApplicable)
{
    using Config = TypeParam;

    auto mxBuilder = createValidBlockScaleDequantizeMxGraph(
        Config::CORRECT_DATA_TYPE, DataType::FP8_E8M0, DataType::FLOAT);
    const GraphWrapper mxGraphWrapper(mxBuilder.GetBufferPointer(), mxBuilder.GetSize());

    const BlockScaleDequantizePlanBuilder<Config::CORRECT_DATA_TYPE,
                                          DataType::FP8_E8M0,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        matchingBuilder;
    EXPECT_TRUE(
        matchingBuilder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));

    const BlockScaleDequantizePlanBuilder<Config::WRONG_DATA_TYPE,
                                          DataType::FP8_E8M0,
                                          DataType::FLOAT,
                                          DataType::FLOAT>
        wrongTypeBuilder;
    EXPECT_FALSE(
        wrongTypeBuilder.isApplicable(mxGraphWrapper.getNode(0), mxGraphWrapper.getTensorMap()));
}
