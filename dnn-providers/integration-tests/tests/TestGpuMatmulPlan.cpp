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

    // Half builder should not be applicable for a float graph
    const GpuMatmulPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT>
        halfPlanBuilder;
    EXPECT_FALSE(
        halfPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Half compute builder should not be applicable for a graph with a float compute type
    const GpuMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF>
        halfComputePlanBuilder;
    EXPECT_FALSE(
        halfComputePlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Missing tensor should return false
    auto tensorMapCopy = graphWrapper.getTensorMap();
    tensorMapCopy.erase(A_UID);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrapper.getNode(0), tensorMapCopy));

    // Matmul builder should not be applicable for a batchnorm fwd graph
    auto batchnormGraphBuilder = createValidBatchnormFwdTrainingGraph();
    auto batchnormGraphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        batchnormGraphBuilder.GetBufferPointer(), batchnormGraphBuilder.GetSize());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(batchnormGraphWrapper.getNode(0),
                                               batchnormGraphWrapper.getTensorMap()));
}

// ====================================================
// Templated helper for plan execution vs CPU reference
// ====================================================

namespace
{

template <typename AType, typename BType, typename CType, typename ComputeType>
void runPlanExecuteVsCpuRef(const std::vector<int64_t>& dims, float tolerance)
{
    constexpr int64_t A_UID = 1;
    constexpr int64_t B_UID = 2;
    constexpr int64_t C_UID = 3;

    const auto strides = generateStrides(dims);

    auto aDataType = nativeTypeToDataType<AType>();
    auto bDataType = nativeTypeToDataType<BType>();
    auto cDataType = nativeTypeToDataType<CType>();
    auto computeDataType = nativeTypeToDataType<ComputeType>();

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
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

    Tensor<AType> aTensor(dims);
    Tensor<BType> bTensor(dims);

    constexpr unsigned int SEED = 42;
    aTensor.fillWithRandomValues(static_cast<AType>(-1.0), static_cast<AType>(1.0), SEED);
    bTensor.fillWithRandomValues(static_cast<BType>(-1.0), static_cast<BType>(1.0), SEED + 1);

    Tensor<CType> gpuC(dims);
    Tensor<CType> cpuC(dims);

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

    const auto* gpuCData = static_cast<const CType*>(gpuC.rawHostData());
    const auto* cpuCData = static_cast<const CType*>(cpuC.rawHostData());
    for(size_t i = 0; i < cpuC.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuCData[i]), static_cast<float>(cpuCData[i]), tolerance)
            << "Mismatch in C at index " << i;
    }
}

// ====================
// Plan execution tests
// ====================

TEST(TestGpuMatmulPlanFp32, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float>({2, 4, 8, 8}, matmul::getTolerance<float>());
}

TEST(TestGpuMatmulPlanFp16, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, half, float>({2, 4, 8, 8}, matmul::getTolerance<half>());
}

TEST(TestGpuMatmulPlanBfp16, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>({2, 4, 8, 8},
                                                                matmul::getTolerance<bfloat16>());
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
