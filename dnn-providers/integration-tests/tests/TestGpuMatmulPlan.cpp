// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <unordered_map>

#include "MatmulGraphTestUtils.hpp"

#include "harness/gpu-graph-executor/detail/GpuMatmulPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuMatmulSignatureKey.hpp"
#include "harness/gpu-graph-executor/detail/GpuPlanBuilderRegistry.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_test_sdk::utilities;

TEST(TestGpuMatmulPlanBuilder, PlanConstruction)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {4, 8, 2, 2};
    const auto strides = generateStrides(dims);

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;

    auto builtPlan = planBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<GpuMatmulPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);

    // Matmul builder should not be able to build a batchnorm fwd graph
    auto batchnormGraphBuilder = createValidBatchnormFwdTrainingGraph();
    auto batchnormGraphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        batchnormGraphBuilder.GetBufferPointer(), batchnormGraphBuilder.GetSize());
    EXPECT_THROW(planBuilder.buildNodePlan(batchnormGraphWrapper, batchnormGraphWrapper.getNode(0)),
                 std::runtime_error);
}

TEST(TestGpuMatmulPlanBuilder, IsApplicable)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {4, 8, 2, 2};
    const auto strides = generateStrides(dims);

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Missing tensor should return false
    auto tensorMapCopy = graphWrapper.getTensorMap();
    tensorMapCopy.erase(A_UID);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrapper.getNode(0), tensorMapCopy));
}

TEST(TestGpuMatmulPlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto graphBuilder = createValidBatchnormFwdTrainingGraph();
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;

    EXPECT_FALSE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestGpuMatmulPlanBuilder, IsApplicableReturnsFalseForWrongOutputDataType)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {4, 8, 2, 2};
    const auto strides = generateStrides(dims);

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuMatmulPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT>
        halfPlanBuilder;
    EXPECT_FALSE(
        halfPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestGpuMatmulPlanBuilder, IsApplicableReturnsFalseForWrongComputeDataType)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {4, 8, 2, 2};
    const auto strides = generateStrides(dims);

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF>
        halfComputePlanBuilder;
    EXPECT_FALSE(
        halfComputePlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestGpuMatmulPlanBuilder, IsApplicableReturnsFalseForRankMismatch)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {4, 8, 2, 2};
    const auto strides = generateStrides(dims);
    const std::vector<int64_t> wrongDims = {8, 2, 2};
    const auto wrongStrides = generateStrides(wrongDims);
    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    auto wrongAGraphBuilder = createMatmulGraph(A_UID,
                                                B_UID,
                                                C_UID,
                                                wrongDims,
                                                wrongStrides,
                                                dims,
                                                strides,
                                                dims,
                                                strides,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT);
    auto wrongAGraphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        wrongAGraphBuilder.GetBufferPointer(), wrongAGraphBuilder.GetSize());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(wrongAGraphWrapper.getNode(0),
                                               wrongAGraphWrapper.getTensorMap()));

    auto wrongBGraphBuilder = createMatmulGraph(A_UID,
                                                B_UID,
                                                C_UID,
                                                dims,
                                                strides,
                                                wrongDims,
                                                wrongStrides,
                                                dims,
                                                strides,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT);
    auto wrongBGraphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        wrongBGraphBuilder.GetBufferPointer(), wrongBGraphBuilder.GetSize());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(wrongBGraphWrapper.getNode(0),
                                               wrongBGraphWrapper.getTensorMap()));

    auto wrongCGraphBuilder = createMatmulGraph(A_UID,
                                                B_UID,
                                                C_UID,
                                                dims,
                                                strides,
                                                dims,
                                                strides,
                                                wrongDims,
                                                wrongStrides,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT);
    auto wrongCGraphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        wrongCGraphBuilder.GetBufferPointer(), wrongCGraphBuilder.GetSize());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(wrongCGraphWrapper.getNode(0),
                                               wrongCGraphWrapper.getTensorMap()));
}

TEST(TestGpuMatmulPlanBuilder, IsApplicableReturnsFalseForRankTooSmall)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {2};
    const auto strides = generateStrides(dims);

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestGpuMatmulPlanBuilder, IsApplicableReturnsFalseForRankTooLarge)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {2, 3, 4, 8, 2, 2};
    const auto strides = generateStrides(dims);

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestGpuMatmulPlanBuilder, IsApplicableReturnsFalseForBatchIncompatible)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {4, 8, 2, 2};
    const auto strides = generateStrides(dims);
    const std::vector<int64_t> wrongDims = {5, 8, 2, 2};
    const auto wrongStrides = generateStrides(wrongDims);
    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          wrongDims,
                                          wrongStrides,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestGpuMatmulPlanBuilder, IsApplicableReturnsFalseForKMismatch)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {4, 8, 2, 2};
    const auto strides = generateStrides(dims);
    const std::vector<int64_t> wrongDims = {5, 8, 2, 3};
    const auto wrongStrides = generateStrides(wrongDims);
    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          wrongDims,
                                          wrongStrides,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestGpuMatmulPlanBuilder, IsApplicableReturnsFalseForMMismatch)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {4, 8, 2, 2};
    const auto strides = generateStrides(dims);
    const std::vector<int64_t> wrongDims = {5, 8, 3, 2};
    const auto wrongStrides = generateStrides(wrongDims);
    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          wrongDims,
                                          wrongStrides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestGpuMatmulPlanBuilder, IsApplicableReturnsFalseForNMismatch)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {4, 8, 2, 2};
    const auto strides = generateStrides(dims);
    const std::vector<int64_t> wrongDims = {5, 8, 2, 3};
    const auto wrongStrides = generateStrides(wrongDims);
    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          wrongDims,
                                          wrongStrides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestGpuMatmulPlanBuilder, IsApplicableReturnsFalseForPassByValueTensors)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {4, 8, 2, 2};
    const auto strides = generateStrides(dims);

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          true);
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

// ====================================================
// Templated helper for plan execution vs CPU reference
// ====================================================

namespace
{

template <typename AType, typename BType, typename CType, typename ComputeType>
void runPlanExecuteVsCpuRef(const std::vector<int64_t>& aDims,
                            const std::vector<int64_t>& bDims,
                            const std::vector<int64_t>& cDims,
                            const std::vector<int64_t>& aStrides,
                            const std::vector<int64_t>& bStrides,
                            const std::vector<int64_t>& cStrides,
                            float tolerance)
{
    constexpr int64_t A_UID = 1;
    constexpr int64_t B_UID = 2;
    constexpr int64_t C_UID = 3;

    auto aDataType = nativeTypeToDataType<AType>();
    auto bDataType = nativeTypeToDataType<BType>();
    auto cDataType = nativeTypeToDataType<CType>();
    auto computeDataType = nativeTypeToDataType<ComputeType>();

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          aDims,
                                          aStrides,
                                          bDims,
                                          bStrides,
                                          cDims,
                                          cStrides,
                                          aDataType,
                                          bDataType,
                                          cDataType,
                                          computeDataType);
    const GraphWrapper graphWrapper(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_MatmulAttributes();
    const auto& tensorMap = graphWrapper.getTensorMap();

    GpuMatmulParams params(*tensorMap.at(nodeAttributes->a_tensor_uid()),
                           *tensorMap.at(nodeAttributes->b_tensor_uid()),
                           *tensorMap.at(nodeAttributes->c_tensor_uid()));
    GpuMatmulPlan<AType, BType, CType, ComputeType> gpuPlan(std::move(params));

    Tensor<AType> aTensor(aDims, aStrides);
    Tensor<BType> bTensor(bDims, bStrides);

    constexpr unsigned int SEED = 42;
    aTensor.fillWithRandomValues(static_cast<AType>(-1.0), static_cast<AType>(1.0), SEED);
    bTensor.fillWithRandomValues(static_cast<BType>(-1.0), static_cast<BType>(1.0), SEED + 1);

    Tensor<CType> gpuC(cDims, cStrides);
    Tensor<CType> cpuC(cDims, cStrides);

    std::unordered_map<int64_t, void*> gpuVariantPack;
    gpuVariantPack[A_UID] = aTensor.rawDeviceData();
    gpuVariantPack[B_UID] = bTensor.rawDeviceData();
    gpuVariantPack[C_UID] = gpuC.rawDeviceData();

    gpuPlan.execute(gpuVariantPack);
    gpuC.markDeviceModified();

    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[A_UID] = aTensor.rawHostData();
    cpuVariantPack[B_UID] = bTensor.rawHostData();
    cpuVariantPack[C_UID] = cpuC.rawHostData();

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), cpuVariantPack);
    cpuC.markHostModified();

    // Despite a comment claiming otherwise, `const T* hostData() const` cannot automatically migrate memory from the device to the host and requires a call to the non-const `T* hostData()`. Unfortunately, `iterateAlongDimensions` only provides const indices, so we need to manually call the non-const hostData to migrate the data from device to host and make it available for comparison via the const hostData
    gpuC.memory().hostData();

    iterateAlongDimensions(gpuC.dims(), [&](const std::vector<int64_t>& indices) {
        EXPECT_NEAR(static_cast<float>(gpuC.getHostValue(indices)),
                    static_cast<float>(cpuC.getHostValue(indices)),
                    tolerance)
            << "Mismatch in C at indices " << vecToString(indices);
    });
}

// ====================
// Plan execution tests
// ====================

TEST(TestGpuMatmulPlanPureFp32, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float>({2, 4, 8, 6},
                                                       {2, 4, 6, 9},
                                                       {2, 4, 8, 9},
                                                       generateStrides({2, 4, 8, 6}),
                                                       generateStrides({2, 4, 6, 9}),
                                                       generateStrides({2, 4, 8, 9}),
                                                       matmul::getTolerance<float>());
}

TEST(TestGpuMatmulPlanPureFp16, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, half, float>({2, 4, 8, 6},
                                                    {2, 4, 6, 9},
                                                    {2, 4, 8, 9},
                                                    generateStrides({2, 4, 8, 6}),
                                                    generateStrides({2, 4, 6, 9}),
                                                    generateStrides({2, 4, 8, 9}),
                                                    matmul::getTolerance<half>());
}

TEST(TestGpuMatmulPlanPureBfp16, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>({2, 4, 8, 6},
                                                                {2, 4, 6, 9},
                                                                {2, 4, 8, 9},
                                                                generateStrides({2, 4, 8, 6}),
                                                                generateStrides({2, 4, 6, 9}),
                                                                generateStrides({2, 4, 8, 9}),
                                                                matmul::getTolerance<bfloat16>());
}

TEST(TestGpuMatmulPlanUpcastFp16, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, float, float>({2, 4, 8, 6},
                                                     {2, 4, 6, 9},
                                                     {2, 4, 8, 9},
                                                     generateStrides({2, 4, 8, 6}),
                                                     generateStrides({2, 4, 6, 9}),
                                                     generateStrides({2, 4, 8, 9}),
                                                     matmul::getTolerance<float>());
}

TEST(TestGpuMatmulPlanUpcastBfp16, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, float, float>({2, 4, 8, 6},
                                                             {2, 4, 6, 9},
                                                             {2, 4, 8, 9},
                                                             generateStrides({2, 4, 8, 6}),
                                                             generateStrides({2, 4, 6, 9}),
                                                             generateStrides({2, 4, 8, 9}),
                                                             matmul::getTolerance<float>());
}

TEST(TestGpuMatmulPlanPureFp32, ExecutePlanBroadcast)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float>({3, 4, 8, 6},
                                                       {9, 2, 6, 9},
                                                       {9, 4, 8, 9},
                                                       generateStrides({3, 4, 8, 6}),
                                                       generateStrides({9, 2, 6, 9}),
                                                       generateStrides({9, 4, 8, 9}),
                                                       matmul::getTolerance<float>());
}

TEST(TestGpuMatmulPlanPureFp16, ExecutePlanBroadcast)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, half, float>({3, 4, 8, 6},
                                                    {9, 2, 6, 9},
                                                    {9, 4, 8, 9},
                                                    generateStrides({3, 4, 8, 6}),
                                                    generateStrides({9, 2, 6, 9}),
                                                    generateStrides({9, 4, 8, 9}),
                                                    matmul::getTolerance<half>());
}

TEST(TestGpuMatmulPlanPureBfp16, ExecutePlanBroadcast)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>({3, 4, 8, 6},
                                                                {9, 2, 6, 9},
                                                                {9, 4, 8, 9},
                                                                generateStrides({3, 4, 8, 6}),
                                                                generateStrides({9, 2, 6, 9}),
                                                                generateStrides({9, 4, 8, 9}),
                                                                matmul::getTolerance<bfloat16>());
}

TEST(TestGpuMatmulPlanUpcastFp16, ExecutePlanBroadcast)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, float, float>({3, 4, 8, 6},
                                                     {9, 2, 6, 9},
                                                     {9, 4, 8, 9},
                                                     generateStrides({3, 4, 8, 6}),
                                                     generateStrides({9, 2, 6, 9}),
                                                     generateStrides({9, 4, 8, 9}),
                                                     matmul::getTolerance<float>());
}

TEST(TestGpuMatmulPlanUpcastBfp16, ExecutePlanBroadcast)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, float, float>({3, 4, 8, 6},
                                                             {9, 2, 6, 9},
                                                             {9, 4, 8, 9},
                                                             generateStrides({3, 4, 8, 6}),
                                                             generateStrides({9, 2, 6, 9}),
                                                             generateStrides({9, 4, 8, 9}),
                                                             matmul::getTolerance<float>());
}

TEST(TestGpuMatmulPlanPureFp32, ExecutePlanUnpacked)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float>({3, 4, 8, 6},
                                                       {9, 2, 6, 9},
                                                       {9, 4, 8, 9},
                                                       {10, 1, 100, 1000},
                                                       {1, 9, 18, 256},
                                                       {750, 150, 1, 15},
                                                       matmul::getTolerance<float>());
}

TEST(TestGpuMatmulPlanPureFp16, ExecutePlanUnpacked)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, half, float>({3, 4, 8, 6},
                                                    {9, 2, 6, 9},
                                                    {9, 4, 8, 9},
                                                    {10, 1, 100, 1000},
                                                    {1, 9, 18, 256},
                                                    {750, 150, 1, 15},
                                                    matmul::getTolerance<half>());
}

TEST(TestGpuMatmulPlanPureBfp16, ExecutePlanUnpacked)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>({3, 4, 8, 6},
                                                                {9, 2, 6, 9},
                                                                {9, 4, 8, 9},
                                                                {10, 1, 100, 1000},
                                                                {1, 9, 18, 256},
                                                                {750, 150, 1, 15},
                                                                matmul::getTolerance<bfloat16>());
}

TEST(TestGpuMatmulPlanUpcastFp16, ExecutePlanUnpacked)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, float, float>({3, 4, 8, 6},
                                                     {9, 2, 6, 9},
                                                     {9, 4, 8, 9},
                                                     {10, 1, 100, 1000},
                                                     {1, 9, 18, 256},
                                                     {750, 150, 1, 15},
                                                     matmul::getTolerance<float>());
}

TEST(TestGpuMatmulPlanUpcastBfp16, ExecutePlanUnpacked)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, float, float>({3, 4, 8, 6},
                                                             {9, 2, 6, 9},
                                                             {9, 4, 8, 9},
                                                             {10, 1, 100, 1000},
                                                             {1, 9, 18, 256},
                                                             {750, 150, 1, 15},
                                                             matmul::getTolerance<float>());
}

// ============================================================================
// Rejection test — unregistered signature
// ============================================================================

TEST(TestMatmulFwdPlanBuilder, UnregisteredSignatureThrows)
{
    GpuPlanBuilderRegistry registry;

    const GpuMatmulSignatureKey unregisteredKey{
        DataType::INT8, DataType::INT8, DataType::INT8, DataType::FLOAT};

    EXPECT_THROW(registry.getPlanBuilder(unregisteredKey), std::runtime_error);
}

} // namespace
