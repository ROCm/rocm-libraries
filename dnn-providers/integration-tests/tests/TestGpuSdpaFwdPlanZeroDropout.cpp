// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Dropout probability 0 without an rng_dump (isSdpaZeroProbabilityDropout, shared with the CPU
// plans): the GPU forward reference treats it as no dropout, bit-identical to the same graph
// without dropout fields, and equal to the CPU executor.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/SdpaFwdTestGraphs.hpp>
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
constexpr int64_t LSE_UID = 5;
constexpr int64_t SEED_UID = 23;
constexpr int64_t OFFSET_UID = 24;

const std::vector<int64_t> DIMS = {2, 2, 6, 16};
const std::vector<int64_t> LSE_DIMS = {2, 2, 6, 1};

std::vector<uint8_t> makeGraph(bool zeroDropout, bool seedAndOffset)
{
    SdpaAttributesT attrs;
    attrs.q_tensor_uid = Q_UID;
    attrs.k_tensor_uid = K_UID;
    attrs.v_tensor_uid = V_UID;
    attrs.o_tensor_uid = O_UID;
    attrs.stats_tensor_uid = LSE_UID;
    attrs.generate_stats = true;
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
    strides.reserve(8);
    const auto add = [&](int64_t uid, DataType type, const std::vector<int64_t>& dims) {
        strides.push_back(hipdnn_data_sdk::utilities::generateStrides(dims));
        tensors.push_back(
            CreateTensorAttributesDirect(builder, uid, "T", type, &strides.back(), &dims));
    };
    for(const auto uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        add(uid, DataType::FLOAT, DIMS);
    }
    add(LSE_UID, DataType::FLOAT, LSE_DIMS);
    const std::vector<int64_t> scalar = {1};
    if(zeroDropout && seedAndOffset)
    {
        add(SEED_UID, DataType::INT64, scalar);
        add(OFFSET_UID, DataType::INT64, scalar);
    }
    auto sdpaAttrs = CreateSdpaAttributes(builder, &attrs);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttrs.Union()));
    builder.Finish(CreateGraphDirect(builder,
                                     "SdpaFwdZeroDropout",
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     &tensors,
                                     &nodes));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

struct Outputs
{
    std::vector<float> o;
    std::vector<float> lse;
};

// One run of `graph` on fixed inputs, on the GPU executor (device buffers) or the CPU one.
Outputs run(std::vector<uint8_t> graph, bool gpu)
{
    Tensor<float> q(DIMS);
    Tensor<float> k(DIMS);
    Tensor<float> v(DIMS);
    Tensor<float> o(DIMS);
    Tensor<float> lse(LSE_DIMS);
    q.fillWithRandomValues(-1.0F, 1.0F, 51);
    k.fillWithRandomValues(-1.0F, 1.0F, 52);
    v.fillWithRandomValues(-1.0F, 1.0F, 53);
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
                                                     {LSE_UID, data(lse)},
                                                     {SEED_UID, data(seed)},
                                                     {OFFSET_UID, data(offset)}};
    if(gpu)
    {
        hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;
        EXPECT_TRUE(executor.isApplicable(graph.data(), graph.size()));
        executor.execute(graph.data(), graph.size(), pack);
        o.memory().markDeviceModified();
        lse.memory().markDeviceModified();
    }
    else
    {
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor executor;
        EXPECT_TRUE(executor.isApplicable(graph.data(), graph.size()));
        executor.execute(graph.data(), graph.size(), pack);
    }
    const auto* oData = o.memory().hostData();
    const auto* lseData = lse.memory().hostData();
    return {{oData, oData + o.elementCount()}, {lseData, lseData + lse.elementCount()}};
}

bool bitsEqual(const std::vector<float>& a, const std::vector<float>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

} // namespace

TEST(TestGpuSdpaFwdPlanZeroDropout, EqualsNoDropoutBitForBitAndTheCpuExecutor)
{
    SKIP_IF_NO_DEVICES();
    const auto noDropout = run(makeGraph(false, false), true);
    for(const bool seedAndOffset : {true, false})
    {
        const auto zero = run(makeGraph(true, seedAndOffset), true);
        EXPECT_TRUE(bitsEqual(zero.o, noDropout.o)) << "seed and offset " << seedAndOffset;
        EXPECT_TRUE(bitsEqual(zero.lse, noDropout.lse)) << "seed and offset " << seedAndOffset;
    }

    const auto cpu = run(makeGraph(true, true), false);
    ASSERT_EQ(cpu.o.size(), noDropout.o.size());
    for(size_t i = 0; i < cpu.o.size(); ++i)
    {
        EXPECT_NEAR(noDropout.o[i], cpu.o[i], 2e-5F) << "O[" << i << "]";
    }
    for(size_t i = 0; i < cpu.lse.size(); ++i)
    {
        EXPECT_NEAR(noDropout.lse[i], cpu.lse[i], 2e-5F) << "LSE[" << i << "]";
    }
}

// The GPU executor's gate is the shared one: a probability above 0 still needs the rng_dump.
TEST(TestGpuSdpaFwdPlanZeroDropout, DeclinesAProbabilityAboveZeroWithoutADump)
{
    auto graph = makeGraph(true, true);
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graph.data(), graph.size()));

    auto unpacked = UnPackGraph(graph.data());
    auto* sdpa = unpacked->nodes.front()->attributes.AsSdpaAttributes();
    ASSERT_NE(sdpa, nullptr);
    sdpa->dropout_probability = 0.1F;
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, unpacked.get()));
    EXPECT_FALSE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}

namespace
{

namespace fp8 = hipdnn_test_sdk::utilities::sdpa_graph_test;

// O (bf16 bits) and the two amax outputs of one GPU run of an fp8 graph with the full cuDNN
// sdpa_fp8 scaling, on fixed inputs.
struct Fp8Outputs
{
    std::vector<uint16_t> o;
    float amaxS = 0.0F;
    float amaxO = 0.0F;
};

Fp8Outputs runFp8OnGpu(const fp8::SdpaGraphSpec& spec)
{
    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_data_sdk::types::fp8_e4m3;
    auto graph = fp8::makeSdpaGraph(spec);
    const std::vector<int64_t> dims = {1, 2, 4, 8};
    Tensor<fp8_e4m3> q(dims);
    Tensor<fp8_e4m3> k(dims);
    Tensor<fp8_e4m3> v(dims);
    Tensor<bfloat16> o(dims);
    q.fillWithRandomValues(fp8_e4m3(-1.0F), fp8_e4m3(1.0F), 31);
    k.fillWithRandomValues(fp8_e4m3(-1.0F), fp8_e4m3(1.0F), 32);
    v.fillWithRandomValues(fp8_e4m3(-1.0F), fp8_e4m3(1.0F), 33);
    // descale q, k, v, s; scale s, o; amax s, o.
    const std::vector<std::pair<int64_t, float>> scalarValues = {{fp8::DESCALE_Q, 0.5F},
                                                                 {fp8::DESCALE_K, 2.0F},
                                                                 {fp8::DESCALE_V, 1.5F},
                                                                 {fp8::DESCALE_S, 0.25F},
                                                                 {fp8::SCALE_S, 4.0F},
                                                                 {fp8::SCALE_O, 3.0F},
                                                                 {fp8::AMAX_S, 0.0F},
                                                                 {fp8::AMAX_O, 0.0F}};
    std::deque<Tensor<float>> scalars;
    Tensor<int64_t> seed({1});
    Tensor<int64_t> offset({1});
    seed.fillWithValue(1234);
    offset.fillWithValue(7);
    std::unordered_map<int64_t, void*> pack = {{fp8::Q, q.memory().deviceData()},
                                               {fp8::K, k.memory().deviceData()},
                                               {fp8::V, v.memory().deviceData()},
                                               {fp8::O, o.memory().deviceData()},
                                               {fp8::SEED, seed.memory().deviceData()},
                                               {fp8::OFFSET, offset.memory().deviceData()}};
    for(const auto& [uid, value] : scalarValues)
    {
        scalars.emplace_back(std::vector<int64_t>{1});
        scalars.back().fillWithValue(value);
        pack[uid] = scalars.back().memory().deviceData();
    }
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
    executor.execute(graph.GetBufferPointer(), graph.GetSize(), pack);
    o.memory().markDeviceModified();
    scalars[6].memory().markDeviceModified();
    scalars[7].memory().markDeviceModified();
    Fp8Outputs out;
    out.o.resize(o.elementCount());
    std::memcpy(out.o.data(), o.memory().hostData(), out.o.size() * sizeof(uint16_t));
    out.amaxS = *scalars[6].memory().hostData();
    out.amaxO = *scalars[7].memory().hostData();
    return out;
}

bool fp8ApplicableOnGpu(const fp8::SdpaGraphSpec& spec)
{
    auto graph = fp8::makeSdpaGraph(spec);
    return hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor().isApplicable(
        graph.GetBufferPointer(), graph.GetSize());
}

} // namespace

// The shared gate: probability 0 without an rng_dump combines with cuDNN's sdpa_fp8 scaling;
// with an rng_dump, or beside an explicit mask, dropout still declines there.
TEST(TestGpuSdpaFwdPlanZeroDropout, CombinesWithFp8Scaling)
{
    fp8::SdpaGraphSpec spec;
    spec.withFullScaling().dropoutProbability = 0.0F;
    EXPECT_TRUE(fp8ApplicableOnGpu(spec));
    spec.rngDump = true;
    EXPECT_FALSE(fp8ApplicableOnGpu(spec));
    spec.rngDump = false;
    spec.dropout = true;
    EXPECT_FALSE(fp8ApplicableOnGpu(spec));
}

// O and both amax outputs are bit-identical to the scaled fp8 graph without dropout.
TEST(TestGpuSdpaFwdPlanZeroDropout, Fp8ScalingEqualsNoDropoutBitForBit)
{
    SKIP_IF_NO_DEVICES();
    fp8::SdpaGraphSpec spec;
    spec.withFullScaling();
    const auto noDropout = runFp8OnGpu(spec);
    spec.dropoutProbability = 0.0F;
    const auto zero = runFp8OnGpu(spec);
    EXPECT_EQ(zero.o, noDropout.o);
    EXPECT_EQ(zero.amaxS, noDropout.amaxS);
    EXPECT_EQ(zero.amaxO, noDropout.amaxO);
    EXPECT_GT(noDropout.amaxS, 0.0F);
    EXPECT_GT(noDropout.amaxO, 0.0F);
}
