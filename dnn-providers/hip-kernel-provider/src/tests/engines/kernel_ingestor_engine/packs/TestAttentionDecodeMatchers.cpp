// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include <functional>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <limits>
namespace
{
namespace d = hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine;

d::GraphT graph()
{
    d::GraphT g;
    for(int64_t uid = 1; uid <= 5; ++uid)
    {
        auto t = std::make_unique<d::TensorAttributesT>();
        t->uid = uid;
        t->data_type = uid == 5 ? d::DataType::INT32 : d::DataType::BFLOAT16;
        const bool cache = uid == 2 || uid == 3;
        t->dims = uid == 5 ? std::vector<int64_t>{1, 1, 1, 1}
                           : std::vector<int64_t>{1, cache ? 2 : 8, cache ? 128 : 1, 128};
        t->strides = uid == 5
                         ? std::vector<int64_t>{1, 1, 1, 1}
                         : std::vector<int64_t>{cache ? 32768 : 1024, 128, cache ? 256 : 1024, 1};
        g.tensors.push_back(std::move(t));
    }
    d::SdpaAttributesT a;
    a.q_tensor_uid = 1;
    a.k_tensor_uid = 2;
    a.v_tensor_uid = 3;
    a.o_tensor_uid = 4;
    a.seq_len_kv_tensor_uid = 5;
    a.attn_scale_value = 0.08838835F;
    auto n = std::make_unique<d::NodeT>();
    n->compute_data_type = d::DataType::FLOAT;
    n->attributes.Set(d::SdpaAttributesT{a});
    g.nodes.push_back(std::move(n));
    return g;
}

bool accepts(const std::function<void(d::GraphT&, d::SdpaAttributesT&)>& change = {})
{
    registerNativeIngestorSymbols();
    auto g = graph();
    if(change)
    {
        change(g, *g.nodes[0]->attributes.AsSdpaAttributes());
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(d::Graph::Pack(builder, &g));
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper const wrapper(
        builder.GetBufferPointer(), builder.GetSize());
    DeviceProperties props;
    props.gcnArchName = "gfx942";
    props.warpSize = 64;
    MatchContext const ctx{wrapper, 0, props};
    return GraphMatchRegistry::resolve("hipkernel.attention_decode.graph_match")(ctx).has_value();
}
TEST(AttentionDecodeMatchers, AcceptsSingleTokenAndBottomRight)
{
    EXPECT_TRUE(accepts());
    EXPECT_TRUE(accepts([](auto&, auto& a) {
        a.right_bound = 0;
        a.diagonal_alignment = d::DiagonalAlignment::BOTTOM_RIGHT;
    }));
    EXPECT_TRUE(accepts([](auto&, auto& a) {
        a.causal_mask_bottom_right = true;
        a.padding_mask = true;
    }));
}
TEST(AttentionDecodeMatchers, DeclinesWrongMasksAndAuxiliaryOutputs)
{
    EXPECT_FALSE(accepts([](auto&, auto& a) { a.causal_mask = true; }));
    EXPECT_FALSE(accepts([](auto&, auto& a) { a.right_bound = 0; }));
    EXPECT_FALSE(accepts([](auto&, auto& a) { a.left_bound = 64; }));
    EXPECT_FALSE(accepts([](auto&, auto& a) { a.stats_tensor_uid = 99; }));
    EXPECT_FALSE(accepts([](auto&, auto& a) { a.generate_stats = true; }));
    EXPECT_FALSE(accepts([](auto&, auto& a) { a.attn_mask_tensor_uid = 99; }));
    EXPECT_FALSE(accepts([](auto&, auto& a) { a.page_table_k_tensor_uid = 99; }));
    EXPECT_FALSE(accepts([](auto&, auto& a) { a.seq_len_q_tensor_uid = 99; }));
}
TEST(AttentionDecodeMatchers, DeclinesWrongShapeLayoutAndLengthDescriptor)
{
    EXPECT_FALSE(
        accepts([](auto& g, auto&) { g.nodes[0]->compute_data_type = d::DataType::HALF; }));
    EXPECT_FALSE(accepts([](auto& g, auto&) { g.tensors[0]->dims[2] = 2; }));
    EXPECT_FALSE(accepts([](auto& g, auto&) { g.tensors[1]->strides[1] = 16384; }));
    EXPECT_FALSE(accepts([](auto& g, auto&) { g.tensors[2]->dims[2] = 64; }));
    EXPECT_FALSE(accepts([](auto& g, auto&) { g.tensors[3]->data_type = d::DataType::HALF; }));
    EXPECT_FALSE(accepts([](auto& g, auto&) { g.tensors[1]->dims[2] = 65; }));
    EXPECT_FALSE(accepts([](auto& g, auto&) { g.tensors[1]->dims = {1}; }));
    EXPECT_FALSE(accepts([](auto& g, auto&) { g.tensors[4]->data_type = d::DataType::FLOAT; }));
    EXPECT_FALSE(accepts([](auto& g, auto&) { g.tensors[4]->dims[0] = 2; }));
    EXPECT_FALSE(accepts([](auto&, auto& a) { a.seq_len_kv_tensor_uid = flatbuffers::nullopt; }));
}
TEST(AttentionDecodeMatchers, ChecksCapacityEndpoints)
{
    for(const int64_t capacity : {64, 65536, 0, -64, 63, 65537, 65600})
    {
        SCOPED_TRACE(capacity);
        EXPECT_EQ(accepts([&](auto& g, auto&) {
                      for(const size_t index : {1U, 2U})
                      {
                          g.tensors[index]->dims[2] = capacity;
                          g.tensors[index]->strides[0] = capacity * 256;
                      }
                  }),
                  capacity == 64 || capacity == 65536);
    }
}
TEST(AttentionDecodeMatchers, RequiresFiniteExplicitScale)
{
    EXPECT_FALSE(accepts([](auto&, auto& a) { a.attn_scale_value = flatbuffers::nullopt; }));
    for(const float scale : {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()})
    {
        EXPECT_FALSE(accepts([&](auto&, auto& a) { a.attn_scale_value = scale; }));
    }
    for(const float scale : {0.0F, -0.5F, 1.0F})
    {
        EXPECT_TRUE(accepts([&](auto&, auto& a) { a.attn_scale_value = scale; }));
    }
}
TEST(AttentionDecodeMatchers, RequiresNonvirtualWellFormedDeviceOperands)
{
    for(size_t index = 0; index < 5; ++index)
    {
        SCOPED_TRACE(index);
        EXPECT_FALSE(accepts([&](auto& g, auto&) { g.tensors[index]->virtual_ = true; }));
        EXPECT_FALSE(accepts([&](auto& g, auto&) { g.tensors[index]->strides.clear(); }));
        EXPECT_FALSE(accepts([&](auto& g, auto&) { g.tensors[index]->dims.clear(); }));
    }
    EXPECT_FALSE(accepts([](auto& g, auto&) { g.tensors.pop_back(); }));
}
TEST(AttentionDecodeMatchers, IgnoresOnlyUnitExtentStrides)
{
    EXPECT_TRUE(accepts([](auto& g, auto&) {
        for(auto& tensor : g.tensors)
        {
            for(size_t axis = 0; axis < tensor->dims.size(); ++axis)
            {
                if(tensor->dims[axis] == 1)
                {
                    tensor->strides[axis] = 7;
                }
            }
        }
    }));
    EXPECT_FALSE(accepts([](auto& g, auto&) { g.tensors[0]->strides[1] = 7; }));
    EXPECT_FALSE(accepts([](auto& g, auto&) { g.tensors[1]->strides[2] = 7; }));
}
} // namespace
#endif
