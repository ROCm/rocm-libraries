// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Verifies that the CPU reference plan builders reject any graph containing a
// ragged tensor. The check lives in every plan builder's isApplicable (see
// CHECK_NO_RAGGED_TENSORS in PlanUtils.hpp); here we exercise it end-to-end
// through CpuReferenceGraphExecutor::isApplicable, which dispatches to the
// correct builder for each node.

#include <cstdint>
#include <memory>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include "BlockScaleDequantizeGraphUtils.hpp"
#include "ConvolutionGraphUtils.hpp"
#include "ConvolutionTensorBundles.hpp"
#include "LayernormGraphUtils.hpp"
#include "MatmulGraphUtils.hpp"
#include "MatmulTensorBundles.hpp"
#include "MoeGroupedMatmulBwdGraphUtils.hpp"
#include "MoeGroupedMatmulBwdTensorBundles.hpp"
#include "MoeGroupedMatmulGraphUtils.hpp"
#include "MoeGroupedMatmulTensorBundles.hpp"
#include "PointwiseGraphUtils.hpp"
#include "RMSNormGraphUtils.hpp"
#include "ReductionGraphUtils.hpp"
#include "ReductionTensorBundles.hpp"
#ifdef HIPDNN_ENABLE_SDPA
#include "SdpaGraphUtils.hpp"
#include "SdpaTensorBundles.hpp"
#endif

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

namespace
{

// Re-serializes a graph with its first non-virtual tensor marked ragged. The
// applicability check keys on ragged_offset_tensor_uid being set, so we point
// it at another existing tensor uid to keep the value plausible.
flatbuffers::DetachedBuffer makeGraphWithRaggedTensor(const std::vector<uint8_t>& serializedGraph)
{
    const Graph* graph = GetGraph(serializedGraph.data());
    auto graphT = std::unique_ptr<GraphT>(graph->UnPack());

    TensorAttributesT* target = nullptr;
    for(auto& tensor : graphT->tensors)
    {
        if(!tensor->virtual_)
        {
            target = tensor.get();
            break;
        }
    }
    EXPECT_NE(target, nullptr) << "Test graph has no non-virtual tensor to mark ragged";
    if(target != nullptr)
    {
        target->ragged_offset_tensor_uid = graphT->tensors.back()->uid;
    }

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

// Validates and serializes a frontend graph.
std::vector<uint8_t> serialize(const std::shared_ptr<hipdnn_frontend::graph::Graph>& graph)
{
    auto result = graph->validate();
    EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

    auto [serializedGraph, serErr] = graph->to_binary();
    EXPECT_TRUE(serErr.is_good()) << serErr.get_message();
    return serializedGraph;
}

// Asserts that an otherwise-applicable graph becomes not-applicable once it
// contains a ragged tensor. The baseline assertion guards against the negative
// result coming from an unrelated mismatch.
void expectRaggedTensorRejected(const std::vector<uint8_t>& serializedGraph)
{
    CpuReferenceGraphExecutor executor;

    ASSERT_TRUE(
        executor.isApplicable(const_cast<uint8_t*>(serializedGraph.data()), serializedGraph.size()))
        << "Baseline (non-ragged) graph should be applicable";

    auto raggedGraph = makeGraphWithRaggedTensor(serializedGraph);
    EXPECT_FALSE(executor.isApplicable(raggedGraph.data(), raggedGraph.size()))
        << "Graph containing a ragged tensor must be rejected";
}

} // namespace

TEST(TestCpuReferenceRaggedRejection, ConvolutionFwd)
{
    ConvolutionFwdTensorBundle<float> bundle(
        {1, 1, 2, 2}, {1, 1, 1, 1}, {1, 1, 2, 2}, 1, TensorLayout::NCHW);
    auto [graph, variantPack] = buildConvolutionFwdGraph(bundle, DataType::FLOAT, DataType::FLOAT);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, ConvolutionBwd)
{
    ConvolutionBwdTensorBundle<float> bundle(
        {1, 1, 2, 2}, {1, 1, 1, 1}, {1, 1, 2, 2}, 1, TensorLayout::NCHW);
    auto [graph, variantPack] = buildConvolutionBwdGraph(bundle, DataType::FLOAT, DataType::FLOAT);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, ConvolutionWrw)
{
    ConvolutionWrwTensorBundle<float> bundle(
        {1, 1, 2, 2}, {1, 1, 1, 1}, {1, 1, 2, 2}, 1, TensorLayout::NCHW);
    auto [graph, variantPack] = buildConvolutionWrwGraph(bundle, DataType::FLOAT, DataType::FLOAT);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, BatchnormFwdInference)
{
    auto graph = buildBatchnormFwdInferenceGraph(DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 {1, 3, 14, 14},
                                                 TensorLayout::NCHW,
                                                 true);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, BatchnormBwd)
{
    BatchnormBwdTensorBundle<float, float, float> bundle({1, 3, 14, 14}, 1, TensorLayout::NCHW);
    auto [graph, variantPack] = buildBatchnormBwdGraph(
        bundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, BatchnormTrain)
{
    BatchnormTrainTensorBundle<float, float, float> bundle(
        {1, 3, 14, 14}, 1, TensorLayout::NCHW, false);
    auto [graph, variantPack] = buildBatchnormTrainGraph(
        bundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, false);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, Matmul)
{
    MatmulTensorBundle<float> bundle({2, 5, 3}, {2, 3, 4}, {2, 5, 4}, false, false, 1);
    auto [graph, variantPack] = buildMatmulGraph(bundle, DataType::FLOAT, DataType::FLOAT);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, MoeGroupedMatmul)
{
    MoeGroupedMatmulTensorBundle<float> bundle(2, 3, 4, 6, 6, MoeGroupedMatmulMode::NONE, 0);
    auto [graph, variantPack]
        = buildMoeGroupedMatmulGraph(bundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, MoeGroupedMatmulBwd)
{
    MoeGroupedMatmulBwdTensorBundle<float> bundle(2, 3, 4, 8);
    auto [graph, variantPack]
        = buildMoeGroupedMatmulBwdGraph(bundle, DataType::FLOAT, DataType::FLOAT);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, Pointwise)
{
    auto [graph, bundle, variantPack]
        = buildPointwiseBinaryGraph({1, 3, 2, 2},
                                    {1, 3, 2, 2},
                                    {1, 3, 2, 2},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ADD,
                                    1,
                                    TensorLayout::NCHW);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, Layernorm)
{
    auto graph = buildLayernormFpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          {1, 3, 14, 14},
                                          2, // normalize over last 2 dims
                                          TensorLayout::NCHW);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, RMSNormFwd)
{
    auto graph = buildRMSNormFwdGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, {1, 3, 14, 14}, TensorLayout::NCHW);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, RMSNormBwd)
{
    auto graph = buildRMSNormBwdGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, {1, 3, 14, 14}, TensorLayout::NCHW);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, Reduction)
{
    ReductionTensorBundle<float> bundle({2, 3, 4, 4}, {2, 3, 1, 1}, 1);
    auto [graph, variantPack] = buildReductionGraph(bundle, DataType::FLOAT, DataType::FLOAT);
    expectRaggedTensorRejected(serialize(graph));
}

TEST(TestCpuReferenceRaggedRejection, BlockScaleDequantize)
{
    BlockScaleDequantizeTensorBundle<float, float> bundle({2, 32, 32, 64}, {2, 32, 32, 2});
    auto [graph, variantPack] = buildBlockScaleDequantizeGraph(bundle,
                                                               DataType::FLOAT,
                                                               DataType::FLOAT,
                                                               DataType::FLOAT,
                                                               DataType::FLOAT,
                                                               std::vector<int32_t>{32});
    expectRaggedTensorRejected(serialize(graph));
}

#ifdef HIPDNN_ENABLE_SDPA
TEST(TestCpuReferenceRaggedRejection, SdpaFwd)
{
    SdpaFwdTensorBundle<float> bundle({1, 2, 4, 8}, {1, 2, 4, 8}, {1, 2, 4, 8});
    auto [graph, variantPack] = buildSdpaFwdGraph(bundle, DataType::FLOAT);
    expectRaggedTensorRejected(serialize(graph));
}
#endif
