// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FloatOperandTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaDensifyTestGraphs.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "SdpaFwdGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/gpu-graph-executor/detail/GpuPlanBuilderRegistry.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaFwdPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaRaggedFwdPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaRaggedFwdSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

using hipdnn_test_sdk::utilities::createRaggedSdpaFwdGraph;
using hipdnn_test_sdk::utilities::FloatOperandSpec;
using hipdnn_test_sdk::utilities::RaggedSdpaFwdGraphOptions;

namespace
{

constexpr int64_t Q_UID = 10;
constexpr int64_t K_UID = 11;
constexpr int64_t V_UID = 12;
constexpr int64_t O_UID = 13;
constexpr int64_t RAGGED_OFFSET_Q_UID = 20;
constexpr int64_t RAGGED_OFFSET_KV_UID = 21;

// One ragged batch. Only dtypes and ragged_offset matter here, not the shape.
const std::vector<int64_t> DIMS = hipdnn_test_sdk::utilities::raggedDims(1, 8, 2, 16);

flatbuffers::FlatBufferBuilder makeRaggedGraph(DataType dataType)
{
    return createRaggedSdpaFwdGraph(Q_UID,
                                    K_UID,
                                    V_UID,
                                    O_UID,
                                    RAGGED_OFFSET_Q_UID,
                                    RAGGED_OFFSET_KV_UID,
                                    /*batch=*/1,
                                    DIMS,
                                    DIMS,
                                    DIMS,
                                    DIMS,
                                    dataType);
}

// The ragged bf16 graph with Q's ragged_offset removed: K, V and O stay ragged.
flatbuffers::FlatBufferBuilder makeGraphWithDenseQ()
{
    auto ragged = makeRaggedGraph(DataType::BFLOAT16);
    auto graph = UnPackGraph(ragged.GetBufferPointer());
    for(auto& tensor : graph->tensors)
    {
        if(tensor->uid == Q_UID)
        {
            tensor->ragged_offset_tensor_uid = flatbuffers::nullopt;
        }
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, graph.get()));
    return builder;
}

// The executor's registry (the one GpuReferenceGraphExecutor looks the node up in) holds the
// native ragged builder for these Q/K/V/O types under the node's ragged key, and that builder
// takes the node. isApplicable alone cannot show this: it falls back to the dense rewrite.
template <DataType QType, DataType KType, DataType VType, DataType OType>
void expectRoutedToNativeRaggedPlan(const void* graphBuffer, size_t graphSize)
{
    auto graphWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(graphBuffer, graphSize);
    const auto& node = graphWrap.getNode(0);
    const auto& tensorMap = graphWrap.getTensorMap();
    GpuPlanBuilderRegistry registry;
    const IGpuGraphNodePlanBuilder* builder = nullptr;
    ASSERT_NO_THROW(builder
                    = &registry.getPlanBuilder(GpuSdpaRaggedFwdSignatureKey(node, tensorMap)));
    EXPECT_NE(
        (dynamic_cast<const GpuSdpaRaggedFwdPlanBuilder<QType, KType, VType, OType>*>(builder)),
        nullptr);
    EXPECT_TRUE(builder->isApplicable(node, tensorMap));
}

} // namespace

TEST(TestGpuSdpaRaggedFwdSignatureKey, EqualityOperator)
{
    const GpuSdpaRaggedFwdSignatureKey key1{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const GpuSdpaRaggedFwdSignatureKey key2{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    EXPECT_TRUE(key1 == key2);

    // Differing output type makes the keys unequal.
    const GpuSdpaRaggedFwdSignatureKey key3{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT};
    EXPECT_FALSE(key1 == key3);
}

TEST(TestGpuSdpaRaggedFwdSignatureKey, HashFunction)
{
    const GpuSdpaRaggedFwdSignatureKey key1{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const GpuSdpaRaggedFwdSignatureKey key2{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    // The same dtype placed in different fields must hash differently.
    const GpuSdpaRaggedFwdSignatureKey key3{
        DataType::HALF, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const GpuSdpaRaggedFwdSignatureKey key4{
        DataType::BFLOAT16, DataType::HALF, DataType::BFLOAT16, DataType::BFLOAT16};
    EXPECT_NE(key3.hashSelf(), key4.hashSelf());
}

TEST(TestGpuSdpaRaggedFwdSignatureKey, CreateFromNodeAndTensorMap)
{
    auto graphBuilder = makeRaggedGraph(DataType::BFLOAT16);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuSdpaRaggedFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    const GpuSdpaRaggedFwdSignatureKey expectedKey{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};

    EXPECT_TRUE(keyFromNode == expectedKey);
}

// Ragged and dense nodes are both applicable, each through its own plan (checked below).
TEST(TestGpuSdpaRaggedFwdSignatureKey, ExecutorRoutesRaggedBf16NodeToRaggedPlan)
{
    using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;

    auto raggedBuilder = makeRaggedGraph(DataType::BFLOAT16);
    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(raggedBuilder.GetBufferPointer(), raggedBuilder.GetSize()));
    expectRoutedToNativeRaggedPlan<DataType::BFLOAT16,
                                   DataType::BFLOAT16,
                                   DataType::BFLOAT16,
                                   DataType::BFLOAT16>(raggedBuilder.GetBufferPointer(),
                                                       raggedBuilder.GetSize());

    // Same shape/dtype but no ragged_offset: a dense node, handled by the dense plan.
    auto denseBuilder = createSdpaFwdGraph(
        Q_UID, K_UID, V_UID, O_UID, DIMS, DIMS, DIMS, DIMS, DataType::BFLOAT16);
    EXPECT_TRUE(executor.isApplicable(denseBuilder.GetBufferPointer(), denseBuilder.GetSize()));
}

// The dense plan must not take a ragged node, or it would read packed data as dense.
TEST(TestGpuSdpaRaggedFwdSignatureKey, DensePlanRejectsRaggedNode)
{
    auto graphBuilder = makeRaggedGraph(DataType::BFLOAT16);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuSdpaFwdPlanBuilder<DataType::BFLOAT16,
                                DataType::BFLOAT16,
                                DataType::BFLOAT16,
                                DataType::BFLOAT16>
        denseBuilder;
    EXPECT_FALSE(denseBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

// Dense Q with ragged K/V/O fits neither plan. It must not be run by the dense one.
TEST(TestGpuSdpaRaggedFwdSignatureKey, ExecutorRejectsPartlyRaggedNode)
{
    using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;

    auto graphBuilder = makeGraphWithDenseQ();
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
}

// fp8 Q/K/V with bf16 O and scalar descales keys and routes to the ragged plan too.
TEST(TestGpuSdpaRaggedFwdSignatureKey, Fp8NodeKeyAndRouting)
{
    using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;

    constexpr int64_t DESCALE_Q_UID = 30;
    constexpr int64_t DESCALE_K_UID = 31;
    constexpr int64_t DESCALE_V_UID = 32;

    RaggedSdpaFwdGraphOptions options;
    options.descaleQ = FloatOperandSpec{DESCALE_Q_UID};
    options.descaleK = FloatOperandSpec{DESCALE_K_UID};
    options.descaleV = FloatOperandSpec{DESCALE_V_UID};
    options.oDataType = DataType::BFLOAT16;
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 /*batch=*/1,
                                                 DIMS,
                                                 DIMS,
                                                 DIMS,
                                                 DIMS,
                                                 DataType::FP8_E4M3,
                                                 options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuSdpaRaggedFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());
    const GpuSdpaRaggedFwdSignatureKey expectedKey{
        DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::BFLOAT16};
    EXPECT_TRUE(keyFromNode == expectedKey);

    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
    expectRoutedToNativeRaggedPlan<DataType::FP8_E4M3,
                                   DataType::FP8_E4M3,
                                   DataType::FP8_E4M3,
                                   DataType::BFLOAT16>(graphBuilder.GetBufferPointer(),
                                                       graphBuilder.GetSize());
}

// The executor end to end on a bf16 ragged graph the native ragged plan takes (see
// ExecutorRoutesRaggedBf16NodeToRaggedPlan), against the same graph with an all-zero bias,
// which every ragged plan declines and so runs through the dense rewrite.
TEST(TestGpuSdpaRaggedFwdSignatureKey, ExecutorRaggedBf16PlanEqualsZeroBiasRewrite)
{
    SKIP_IF_NO_DEVICES();
    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_data_sdk::utilities::Tensor;
    using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;
    namespace rdt = hipdnn_test_sdk::utilities::ragged_densify_test;

    const auto bf16Tensor = [](const std::vector<float>& values) {
        Tensor<bfloat16> t({static_cast<int64_t>(values.size())});
        auto* host = t.memory().hostData();
        for(size_t i = 0; i < values.size(); ++i)
        {
            host[i] = static_cast<bfloat16>(values[i]);
        }
        return t;
    };
    const auto intTensor = [](const std::vector<int32_t>& values) {
        Tensor<int32_t> t({static_cast<int64_t>(values.size())});
        auto* host = t.memory().hostData();
        for(size_t i = 0; i < values.size(); ++i)
        {
            host[i] = values[i];
        }
        return t;
    };
    constexpr int64_t TOKENS_Q = 10;
    constexpr int64_t TOKENS_KV = 13;
    auto q = bf16Tensor(rdt::packedValues(TOKENS_Q, rdt::H * rdt::D, 61));
    auto k = bf16Tensor(rdt::packedValues(TOKENS_KV, rdt::H * rdt::D, 62));
    auto v = bf16Tensor(rdt::packedValues(TOKENS_KV, rdt::H * rdt::D, 63));
    auto qOffsets = intTensor(rdt::Q_TOKENS);
    auto kvOffsets = intTensor(rdt::KV_TOKENS);
    Tensor<float> zeroBias({rdt::SQ * rdt::SKV});
    zeroBias.fillWithValue(0.0F);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto run = [&](bool bias) {
        auto graph = rdt::raggedGraph(false, bias, /*ragged=*/true, DataType::BFLOAT16);
        Tensor<bfloat16> o({TOKENS_Q * rdt::H * rdt::D});
        Tensor<float> lse({TOKENS_Q * rdt::H});
        o.fillWithValue(static_cast<bfloat16>(nan));
        lse.fillWithValue(nan);
        std::unordered_map<int64_t, void*> pack
            = {{rdt::Q, q.memory().deviceData()},
               {rdt::K, k.memory().deviceData()},
               {rdt::V, v.memory().deviceData()},
               {rdt::O, o.memory().deviceData()},
               {rdt::STATS, lse.memory().deviceData()},
               {rdt::Q_OFFSETS, qOffsets.memory().deviceData()},
               {rdt::KV_OFFSETS, kvOffsets.memory().deviceData()}};
        if(bias)
        {
            pack[rdt::BIAS] = zeroBias.memory().deviceData();
        }
        if(!bias)
        {
            expectRoutedToNativeRaggedPlan<DataType::BFLOAT16,
                                           DataType::BFLOAT16,
                                           DataType::BFLOAT16,
                                           DataType::BFLOAT16>(graph.data(), graph.size());
        }
        GpuReferenceGraphExecutor executor;
        EXPECT_TRUE(executor.isApplicable(graph.data(), graph.size()));
        executor.execute(graph.data(), graph.size(), pack);
        o.memory().markDeviceModified();
        lse.memory().markDeviceModified();
        std::vector<float> oHost(static_cast<size_t>(TOKENS_Q * rdt::H * rdt::D));
        for(size_t i = 0; i < oHost.size(); ++i)
        {
            oHost[i] = static_cast<float>(o.memory().hostData()[i]);
        }
        const std::vector<float> lseHost(lse.memory().hostData(),
                                         lse.memory().hostData() + TOKENS_Q * rdt::H);
        return std::make_pair(oHost, lseHost);
    };
    const auto native = run(false);
    const auto rewritten = run(true);

    // Both arms compute in float and round O to bf16, so they may differ by one bf16 step.
    for(size_t i = 0; i < native.first.size(); ++i)
    {
        ASSERT_FALSE(std::isnan(native.first[i])) << "native plan left O element " << i;
        EXPECT_NEAR(native.first[i], rewritten.first[i], 0.0078125F) << "O element " << i;
    }
    for(size_t i = 0; i < native.second.size(); ++i)
    {
        ASSERT_FALSE(std::isnan(native.second[i])) << "native plan left LSE element " << i;
        // A dead row's LSE is -inf on both sides (SEM-010); infinities compare exactly.
        if(std::isinf(native.second[i]))
        {
            EXPECT_EQ(native.second[i], rewritten.second[i]) << "LSE element " << i;
        }
        else
        {
            EXPECT_NEAR(native.second[i], rewritten.second[i], 1e-5F) << "LSE element " << i;
        }
    }
}
