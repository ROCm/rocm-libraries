// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/SdpaGraphMutators.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_test_sdk::utilities;

namespace
{

const TensorAttributesT& findTensor(const GraphT& graph, const std::string& name)
{
    for(const auto& tensor : graph.tensors)
    {
        if(tensor->name == name)
        {
            return *tensor;
        }
    }
    throw std::runtime_error("No tensor named " + name);
}

const TensorAttributesT& findTensor(const GraphT& graph, int64_t uid)
{
    for(const auto& tensor : graph.tensors)
    {
        if(tensor->uid == uid)
        {
            return *tensor;
        }
    }
    throw std::runtime_error("No tensor with uid " + std::to_string(uid));
}

void expectSeqstartTensor(const TensorAttributesT& offset, int64_t batch)
{
    EXPECT_EQ(offset.data_type, DataType::INT32);
    EXPECT_EQ(offset.dims, (std::vector<int64_t>{batch + 1, 1, 1, 1}));
    EXPECT_EQ(offset.strides, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_FALSE(offset.virtual_);
}

flatbuffers::FlatBufferBuilder withoutTensor(flatbuffers::FlatBufferBuilder&& builder,
                                             const std::string& name)
{
    auto graph = UnPackGraph(builder.GetBufferPointer());
    auto& tensors = graph->tensors;
    tensors.erase(std::remove_if(tensors.begin(),
                                 tensors.end(),
                                 [&](const auto& tensor) { return tensor->name == name; }),
                  tensors.end());
    flatbuffers::FlatBufferBuilder result;
    result.Finish(CreateGraph(result, graph.get()));
    return result;
}

} // namespace

TEST(TestSdpaGraphMutators, RaggedQkvoShareQoAndKvOffsets)
{
    // Batch 3 distinguishes the [B+1] offset extent from the default graph's other dims.
    const std::vector<int64_t> dims = {3, 8, 16, 64};
    const std::vector<int64_t> strides = {8192, 1024, 64, 1};
    auto builder = sdpa::withRaggedTensors(
        createValidSdpaFwdGraph(dims, strides, dims, strides, dims, strides, dims, strides),
        {"q", "k", "v", "o"});
    const auto graph = UnPackGraph(builder.GetBufferPointer());

    const auto& q = findTensor(*graph, "q");
    const auto& k = findTensor(*graph, "k");
    const auto& v = findTensor(*graph, "v");
    const auto& o = findTensor(*graph, "o");

    ASSERT_TRUE(q.ragged_offset_tensor_uid.has_value());
    ASSERT_TRUE(k.ragged_offset_tensor_uid.has_value());
    EXPECT_EQ(o.ragged_offset_tensor_uid, q.ragged_offset_tensor_uid);
    EXPECT_EQ(v.ragged_offset_tensor_uid, k.ragged_offset_tensor_uid);
    EXPECT_NE(*q.ragged_offset_tensor_uid, *k.ragged_offset_tensor_uid);

    for(const auto* tensor : {&q, &k, &v, &o})
    {
        EXPECT_EQ(tensor->ragged_offset_multiplier, strides[2]) << tensor->name;
    }

    const auto& qoOffset = findTensor(*graph, *q.ragged_offset_tensor_uid);
    const auto& kvOffset = findTensor(*graph, *k.ragged_offset_tensor_uid);
    EXPECT_EQ(qoOffset.name, "qo_ragged_offset");
    EXPECT_EQ(kvOffset.name, "kv_ragged_offset");
    expectSeqstartTensor(qoOffset, dims[0]);
    expectSeqstartTensor(kvOffset, dims[0]);

    // Two offsets appended with fresh uids; the original four tensors keep theirs.
    ASSERT_EQ(graph->tensors.size(), 6u);
    EXPECT_GT(*q.ragged_offset_tensor_uid, 4);
    EXPECT_GT(*k.ragged_offset_tensor_uid, 4);
}

TEST(TestSdpaGraphMutators, RaggedMarksOnlyNamedTensors)
{
    auto builder = sdpa::withRaggedTensors(createValidSdpaFwdGraph(), {"q"});
    const auto graph = UnPackGraph(builder.GetBufferPointer());

    EXPECT_TRUE(findTensor(*graph, "q").ragged_offset_tensor_uid.has_value());
    // O would share Q's offset but is not named, so it stays dense.
    for(const auto* name : {"k", "v", "o"})
    {
        const auto& tensor = findTensor(*graph, name);
        EXPECT_FALSE(tensor.ragged_offset_tensor_uid.has_value()) << name;
        EXPECT_EQ(tensor.ragged_offset_multiplier, 1) << name;
    }
}

TEST(TestSdpaGraphMutators, RaggedOAndVAloneUseSharedOffsetNames)
{
    auto builder = sdpa::withRaggedTensors(createValidSdpaFwdGraph(), {"o", "v"});
    const auto graph = UnPackGraph(builder.GetBufferPointer());

    const auto& o = findTensor(*graph, "o");
    const auto& v = findTensor(*graph, "v");
    ASSERT_TRUE(o.ragged_offset_tensor_uid.has_value());
    ASSERT_TRUE(v.ragged_offset_tensor_uid.has_value());
    EXPECT_EQ(findTensor(*graph, *o.ragged_offset_tensor_uid).name, "qo_ragged_offset");
    EXPECT_EQ(findTensor(*graph, *v.ragged_offset_tensor_uid).name, "kv_ragged_offset");
    EXPECT_FALSE(findTensor(*graph, "q").ragged_offset_tensor_uid.has_value());
    EXPECT_FALSE(findTensor(*graph, "k").ragged_offset_tensor_uid.has_value());
}

TEST(TestSdpaGraphMutators, RaggedNonQkvoTensorGetsOwnOffset)
{
    auto builder = sdpa::withRaggedTensors(createValidSdpaFwdGraph({2, 8, 16, 64},
                                                                   {8192, 1024, 64, 1},
                                                                   {2, 8, 16, 64},
                                                                   {8192, 1024, 64, 1},
                                                                   {2, 8, 16, 64},
                                                                   {8192, 1024, 64, 1},
                                                                   {2, 8, 16, 64},
                                                                   {8192, 1024, 64, 1},
                                                                   DataType::HALF,
                                                                   /*withAttnMask=*/false,
                                                                   /*withScale=*/true),
                                           {"scale"});
    const auto graph = UnPackGraph(builder.GetBufferPointer());

    const auto& scale = findTensor(*graph, "scale");
    ASSERT_TRUE(scale.ragged_offset_tensor_uid.has_value());
    const auto& offset = findTensor(*graph, *scale.ragged_offset_tensor_uid);
    EXPECT_EQ(offset.name, "scale_ragged_offset");
    expectSeqstartTensor(offset, 2);
    // A rank-1 tensor has no sequence stride, so its multiplier keeps the default.
    EXPECT_EQ(scale.ragged_offset_multiplier, 1);

    // No Q/K/V/O name was requested, so no qo/kv offsets are appended.
    EXPECT_EQ(graph->tensors.size(), 6u); // q, k, v, o, scale, scale_ragged_offset
    EXPECT_FALSE(findTensor(*graph, "q").ragged_offset_tensor_uid.has_value());
}

TEST(TestSdpaGraphMutators, RaggedReadsBatchFromBackwardGraphQ)
{
    const std::vector<int64_t> qDims = {5, 8, 16, 64};
    const std::vector<int64_t> strides = {8192, 1024, 64, 1};
    auto builder = sdpa::withRaggedTensors(createValidSdpaBwdGraph(qDims, strides), {"q"});
    const auto graph = UnPackGraph(builder.GetBufferPointer());

    const auto& q = findTensor(*graph, "q");
    ASSERT_TRUE(q.ragged_offset_tensor_uid.has_value());
    expectSeqstartTensor(findTensor(*graph, *q.ragged_offset_tensor_uid), qDims[0]);
}

TEST(TestSdpaGraphMutators, RaggedThrowsWithoutSdpaQTensor)
{
    EXPECT_THROW(sdpa::withRaggedTensors(createEmptyValidGraph(), {"q"}), std::runtime_error);
    EXPECT_THROW(sdpa::withRaggedTensors(createValidMatmulGraph(), {"q"}), std::runtime_error);
    EXPECT_THROW(sdpa::withRaggedTensors(withoutTensor(createValidSdpaFwdGraph(), "q"), {"k"}),
                 std::runtime_error);
}

TEST(TestSdpaGraphMutators, SeqLenTensorsAttachToForwardNode)
{
    auto builder = sdpa::withSeqLenKv(sdpa::withSeqLenQ(createValidSdpaFwdGraph()));
    const auto graph = UnPackGraph(builder.GetBufferPointer());

    ASSERT_FALSE(graph->nodes.empty());
    const auto* attrs = graph->nodes.front()->attributes.AsSdpaAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_TRUE(attrs->seq_len_q_tensor_uid.has_value());
    ASSERT_TRUE(attrs->seq_len_kv_tensor_uid.has_value());
    EXPECT_NE(*attrs->seq_len_q_tensor_uid, *attrs->seq_len_kv_tensor_uid);

    for(const auto& [uid, name] : {std::pair{*attrs->seq_len_q_tensor_uid, "seq_len_q"},
                                   std::pair{*attrs->seq_len_kv_tensor_uid, "seq_len_kv"}})
    {
        const auto& tensor = findTensor(*graph, uid);
        EXPECT_EQ(tensor.name, name);
        EXPECT_EQ(tensor.data_type, DataType::INT32);
        EXPECT_EQ(tensor.dims, std::vector<int64_t>{1});
        EXPECT_EQ(tensor.strides, std::vector<int64_t>{1});
    }
}

TEST(TestSdpaGraphMutators, SeqLenThrowsWithoutForwardSdpaNode)
{
    EXPECT_THROW(sdpa::withSeqLenQ(createEmptyValidGraph()), std::runtime_error);
    EXPECT_THROW(sdpa::withSeqLenKv(createValidSdpaBwdGraph()), std::runtime_error);
}
