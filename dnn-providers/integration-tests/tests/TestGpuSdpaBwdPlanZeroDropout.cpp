// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Dropout probability 0 on an SDPA backward node (isSdpaZeroProbabilityDropout, shared with the
// CPU plans): the GPU backward reference treats it as no dropout, so dQ, dK and dV are
// bit-identical to the same graph without dropout fields, and equal to the CPU executor.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::Tensor;

namespace
{

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t DO_UID = 5;
constexpr int64_t STATS_UID = 6;
constexpr int64_t DQ_UID = 7;
constexpr int64_t DK_UID = 8;
constexpr int64_t DV_UID = 9;
constexpr int64_t SEED_UID = 23;
constexpr int64_t OFFSET_UID = 24;

const std::vector<int64_t> Q_DIMS = {2, 2, 6, 16};
const std::vector<int64_t> KV_DIMS = {2, 2, 8, 16};
const std::vector<int64_t> STATS_DIMS = {2, 2, 6, 1};

std::vector<uint8_t> makeGraph(bool zeroDropout, bool seedAndOffset)
{
    SdpaBackwardAttributesT attrs;
    attrs.q_tensor_uid = Q_UID;
    attrs.k_tensor_uid = K_UID;
    attrs.v_tensor_uid = V_UID;
    attrs.o_tensor_uid = O_UID;
    attrs.do_tensor_uid = DO_UID;
    attrs.stats_tensor_uid = STATS_UID;
    attrs.dq_tensor_uid = DQ_UID;
    attrs.dk_tensor_uid = DK_UID;
    attrs.dv_tensor_uid = DV_UID;
    attrs.attn_scale_value = 0.25F;
    attrs.right_bound = 0;
    attrs.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
    if(zeroDropout)
    {
        attrs.dropout_probability = 0.0F;
        if(seedAndOffset)
        {
            attrs.seed_tensor_uid = SEED_UID;
            attrs.offset_tensor_uid = OFFSET_UID;
        }
    }

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    std::vector<std::vector<int64_t>> strides;
    strides.reserve(11);
    const auto add = [&](int64_t uid, DataType type, const std::vector<int64_t>& dims) {
        strides.push_back(hipdnn_data_sdk::utilities::generateStrides(dims));
        tensors.push_back(
            CreateTensorAttributesDirect(builder, uid, "T", type, &strides.back(), &dims));
    };
    for(const auto uid : {Q_UID, O_UID, DO_UID, DQ_UID})
    {
        add(uid, DataType::FLOAT, Q_DIMS);
    }
    for(const auto uid : {K_UID, V_UID, DK_UID, DV_UID})
    {
        add(uid, DataType::FLOAT, KV_DIMS);
    }
    add(STATS_UID, DataType::FLOAT, STATS_DIMS);
    const std::vector<int64_t> scalar = {1};
    if(zeroDropout && seedAndOffset)
    {
        add(SEED_UID, DataType::INT64, scalar);
        add(OFFSET_UID, DataType::INT64, scalar);
    }
    auto sdpaAttrs = CreateSdpaBackwardAttributes(builder, &attrs);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_bwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaBackwardAttributes,
                                     sdpaAttrs.Union()));
    builder.Finish(CreateGraphDirect(builder,
                                     "SdpaBwdZeroDropout",
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     &tensors,
                                     &nodes));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

struct Gradients
{
    std::vector<float> dq;
    std::vector<float> dk;
    std::vector<float> dv;
};

std::vector<float> toVector(Tensor<float>& t)
{
    const auto* data = t.memory().hostData();
    return {data, data + t.elementCount()};
}

// One run of `graph` on fixed inputs, on the GPU executor (device buffers) or the CPU one.
Gradients run(std::vector<uint8_t> graph, bool gpu)
{
    Tensor<float> q(Q_DIMS);
    Tensor<float> k(KV_DIMS);
    Tensor<float> v(KV_DIMS);
    Tensor<float> o(Q_DIMS);
    Tensor<float> dO(Q_DIMS);
    Tensor<float> stats(STATS_DIMS);
    Tensor<float> dq(Q_DIMS);
    Tensor<float> dk(KV_DIMS);
    Tensor<float> dv(KV_DIMS);
    q.fillWithRandomValues(-1.0F, 1.0F, 61);
    k.fillWithRandomValues(-1.0F, 1.0F, 62);
    v.fillWithRandomValues(-1.0F, 1.0F, 63);
    o.fillWithRandomValues(-1.0F, 1.0F, 64);
    dO.fillWithRandomValues(-1.0F, 1.0F, 65);
    stats.fillWithRandomValues(0.0F, 1.0F, 66);
    Tensor<int64_t> seed({1});
    Tensor<int64_t> offset({1});
    seed.fillWithValue(99);
    offset.fillWithValue(3);
    const auto data
        = [gpu](auto& t) -> void* { return gpu ? t.memory().deviceData() : t.memory().hostData(); };
    const std::unordered_map<int64_t, void*> pack = {{Q_UID, data(q)},
                                                     {K_UID, data(k)},
                                                     {V_UID, data(v)},
                                                     {O_UID, data(o)},
                                                     {DO_UID, data(dO)},
                                                     {STATS_UID, data(stats)},
                                                     {DQ_UID, data(dq)},
                                                     {DK_UID, data(dk)},
                                                     {DV_UID, data(dv)},
                                                     {SEED_UID, data(seed)},
                                                     {OFFSET_UID, data(offset)}};
    if(gpu)
    {
        hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;
        EXPECT_TRUE(executor.isApplicable(graph.data(), graph.size()));
        executor.execute(graph.data(), graph.size(), pack);
        dq.memory().markDeviceModified();
        dk.memory().markDeviceModified();
        dv.memory().markDeviceModified();
    }
    else
    {
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor executor;
        EXPECT_TRUE(executor.isApplicable(graph.data(), graph.size()));
        executor.execute(graph.data(), graph.size(), pack);
    }
    return {toVector(dq), toVector(dk), toVector(dv)};
}

bool bitsEqual(const std::vector<float>& a, const std::vector<float>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void expectNear(const std::vector<float>& expected,
                const std::vector<float>& actual,
                const char* name)
{
    ASSERT_EQ(expected.size(), actual.size());
    for(size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(expected[i], actual[i], 2e-4F) << name << "[" << i << "]";
    }
}

} // namespace

TEST(TestGpuSdpaBwdPlanZeroDropout, EqualsNoDropoutBitForBitAndTheCpuExecutor)
{
    SKIP_IF_NO_DEVICES();
    const auto noDropout = run(makeGraph(false, false), true);
    for(const bool seedAndOffset : {true, false})
    {
        const auto zero = run(makeGraph(true, seedAndOffset), true);
        EXPECT_TRUE(bitsEqual(zero.dq, noDropout.dq)) << "seed and offset " << seedAndOffset;
        EXPECT_TRUE(bitsEqual(zero.dk, noDropout.dk)) << "seed and offset " << seedAndOffset;
        EXPECT_TRUE(bitsEqual(zero.dv, noDropout.dv)) << "seed and offset " << seedAndOffset;
    }

    const auto cpu = run(makeGraph(true, true), false);
    expectNear(noDropout.dq, cpu.dq, "dQ");
    expectNear(noDropout.dk, cpu.dk, "dK");
    expectNear(noDropout.dv, cpu.dv, "dV");
}

// The GPU executor's gate is the shared one: a probability above 0 without an explicit mask is
// declined.
TEST(TestGpuSdpaBwdPlanZeroDropout, DeclinesAProbabilityAboveZeroWithoutAMask)
{
    auto graph = makeGraph(true, true);
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graph.data(), graph.size()));

    auto unpacked = UnPackGraph(graph.data());
    auto* sdpa = unpacked->nodes.front()->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(sdpa, nullptr);
    sdpa->dropout_probability = 0.1F;
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, unpacked.get()));
    EXPECT_FALSE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}
