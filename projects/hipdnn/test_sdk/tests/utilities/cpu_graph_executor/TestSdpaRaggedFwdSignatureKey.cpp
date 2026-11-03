// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanRegistrySignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaRaggedFwdSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;

namespace
{

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t RAGGED_OFFSET_Q_UID = 10;
constexpr int64_t RAGGED_OFFSET_KV_UID = 11;

// One ragged batch. The key reads only dtypes, so the shape does not matter.
const std::vector<int64_t> DIMS = raggedDims(1, 4, 2, 8);

flatbuffers::FlatBufferBuilder makeRaggedGraph(DataType dataType,
                                               const RaggedSdpaFwdGraphOptions& options = {})
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
                                    dataType,
                                    options);
}

// The ragged graph with a different dtype for each of Q, K, V and O.
flatbuffers::DetachedBuffer
    makeGraphWithDistinctDtypes(DataType q, DataType k, DataType v, DataType o)
{
    const auto ragged = makeRaggedGraph(DataType::BFLOAT16);
    auto graphT = std::unique_ptr<GraphT>(GetGraph(ragged.GetBufferPointer())->UnPack());
    const std::unordered_map<int64_t, DataType> dataTypes
        = {{Q_UID, q}, {K_UID, k}, {V_UID, v}, {O_UID, o}};
    for(auto& tensor : graphT->tensors)
    {
        const auto it = dataTypes.find(tensor->uid);
        if(it != dataTypes.end())
        {
            tensor->data_type = it->second;
        }
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

SdpaRaggedFwdSignatureKey keyFromGraph(const void* graph, size_t size)
{
    const GraphWrapper graphWrapper(graph, size);
    return {graphWrapper.getNode(0), graphWrapper.getTensorMap()};
}

} // namespace

TEST(TestSdpaRaggedFwdSignatureKey, EqualityOperator)
{
    const SdpaRaggedFwdSignatureKey key1{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const SdpaRaggedFwdSignatureKey key2{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    EXPECT_TRUE(key1 == key2);

    const SdpaRaggedFwdSignatureKey fp8Key{
        DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::BFLOAT16};
    EXPECT_FALSE(key1 == fp8Key);

    // Swapping the K and V dtypes must give a different key.
    const SdpaRaggedFwdSignatureKey distinct{
        DataType::BFLOAT16, DataType::HALF, DataType::FLOAT, DataType::FP8_E4M3};
    const SdpaRaggedFwdSignatureKey swappedKv{
        DataType::BFLOAT16, DataType::FLOAT, DataType::HALF, DataType::FP8_E4M3};
    EXPECT_FALSE(distinct == swappedKv);
}

TEST(TestSdpaRaggedFwdSignatureKey, HashFunction)
{
    const SdpaRaggedFwdSignatureKey key1{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const SdpaRaggedFwdSignatureKey key2{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    // The same dtype in each of the four fields must hash four different ways.
    const SdpaRaggedFwdSignatureKey inQ{
        DataType::HALF, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const SdpaRaggedFwdSignatureKey inK{
        DataType::BFLOAT16, DataType::HALF, DataType::BFLOAT16, DataType::BFLOAT16};
    const SdpaRaggedFwdSignatureKey inV{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::HALF, DataType::BFLOAT16};
    const SdpaRaggedFwdSignatureKey inO{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::HALF};
    const std::vector<size_t> hashes
        = {inQ.hashSelf(), inK.hashSelf(), inV.hashSelf(), inO.hashSelf()};
    for(size_t i = 0; i < hashes.size(); ++i)
    {
        for(size_t j = i + 1; j < hashes.size(); ++j)
        {
            EXPECT_NE(hashes[i], hashes[j]) << "fields " << i << " and " << j;
        }
    }
    EXPECT_FALSE(inK == inV);
}

TEST(TestSdpaRaggedFwdSignatureKey, CreateFromBf16RaggedNode)
{
    const auto graph = makeRaggedGraph(DataType::BFLOAT16);
    const SdpaRaggedFwdSignatureKey expectedKey{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    EXPECT_TRUE(keyFromGraph(graph.GetBufferPointer(), graph.GetSize()) == expectedKey);
}

TEST(TestSdpaRaggedFwdSignatureKey, CreateFromFp8RaggedNodeWithBf16Output)
{
    RaggedSdpaFwdGraphOptions options;
    options.oDataType = DataType::BFLOAT16;
    const auto graph = makeRaggedGraph(DataType::FP8_E4M3, options);
    const SdpaRaggedFwdSignatureKey expectedKey{
        DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::BFLOAT16};
    EXPECT_TRUE(keyFromGraph(graph.GetBufferPointer(), graph.GetSize()) == expectedKey);
}

// Each field must come from its own tensor, so a node with four distinct dtypes keeps them in
// order.
TEST(TestSdpaRaggedFwdSignatureKey, CreateFromNodeReadsEachFieldFromItsTensor)
{
    const auto graph = makeGraphWithDistinctDtypes(
        DataType::BFLOAT16, DataType::HALF, DataType::FLOAT, DataType::FP8_E4M3);
    const auto key = keyFromGraph(graph.data(), graph.size());
    EXPECT_EQ(key.qDataType, DataType::BFLOAT16);
    EXPECT_EQ(key.kDataType, DataType::HALF);
    EXPECT_EQ(key.vDataType, DataType::FLOAT);
    EXPECT_EQ(key.oDataType, DataType::FP8_E4M3);
}

// Dense and ragged keys with the same dtypes must stay apart in the plan registry, or a ragged
// node could be handed to the dense plan.
TEST(TestSdpaRaggedFwdSignatureKey, DiffersFromDenseKeyWithSameDtypes)
{
    const PlanRegistrySignatureKey dense = SdpaFwdSignatureKey{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const PlanRegistrySignatureKey ragged = SdpaRaggedFwdSignatureKey{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};

    const PlanRegistrySignatureKeyEqual eq;
    EXPECT_FALSE(eq(dense, ragged));
    EXPECT_FALSE(eq(ragged, dense));
    EXPECT_TRUE(eq(ragged, ragged));

    std::unordered_map<PlanRegistrySignatureKey,
                       int,
                       PlanRegistrySignatureKeyHash,
                       PlanRegistrySignatureKeyEqual>
        registry;
    registry[dense] = 1;
    registry[ragged] = 2;
    EXPECT_EQ(registry.size(), 2u);
    EXPECT_EQ(registry.at(dense), 1);
    EXPECT_EQ(registry.at(ragged), 2);
}

TEST(TestSdpaRaggedFwdSignatureKey, StreamOperator)
{
    const SdpaRaggedFwdSignatureKey key{
        DataType::BFLOAT16, DataType::HALF, DataType::FLOAT, DataType::FP8_E4M3};
    std::ostringstream os;
    os << key;
    EXPECT_EQ(os.str(), "SdpaRaggedFwd(q=BFLOAT16, k=HALF, v=FLOAT, o=FP8_E4M3)");
}
