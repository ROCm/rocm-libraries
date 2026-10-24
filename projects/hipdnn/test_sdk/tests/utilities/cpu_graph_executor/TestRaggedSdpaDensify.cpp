// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaDensifyTestGraphs.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaDensify.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_test_sdk::utilities::CpuFpReferenceSdpa;
using hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor;
using hipdnn_test_sdk::utilities::SdpaBwdOptions;
using hipdnn_test_sdk::utilities::SdpaFwdOptions;
using namespace hipdnn_test_sdk::utilities::ragged_densify_test;

TEST(TestRaggedSdpaDensify, RewritesRaggedPrimariesToDensePadding)
{
    const auto buffer = raggedGraph(false, false);
    const auto dense
        = hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(buffer.data(), buffer.size());
    ASSERT_TRUE(dense.has_value());
    EXPECT_EQ(dense->primaries.size(), 5u); // Q, K, V, O and the packed LSE
    ASSERT_TRUE(dense->addedSeqLenQUid.has_value());
    ASSERT_TRUE(dense->addedSeqLenKvUid.has_value());

    const auto graph = UnPackGraph(dense->graph.data());
    for(const auto& t : graph->tensors)
    {
        EXPECT_FALSE(t->ragged_offset_tensor_uid.has_value()) << t->uid;
        EXPECT_NE(t->uid, static_cast<int64_t>(Q_OFFSETS));
        EXPECT_NE(t->uid, static_cast<int64_t>(KV_OFFSETS));
        if(t->uid == static_cast<int64_t>(Q))
        {
            EXPECT_EQ(t->strides, denseStrides(SQ));
        }
        if(t->uid == *dense->addedSeqLenQUid)
        {
            EXPECT_EQ(t->data_type, DataType::INT32);
            EXPECT_EQ(t->dims, (std::vector<int64_t>{B, 1, 1, 1}));
        }
    }
    const auto* attrs = graph->nodes.front()->attributes.AsSdpaAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_TRUE(attrs->padding_mask);
    EXPECT_EQ(attrs->seq_len_q_tensor_uid.value(), *dense->addedSeqLenQUid);

    const auto plain = raggedGraph(false, false, /*ragged=*/false);
    EXPECT_FALSE(
        hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(plain.data(), plain.size()).has_value());
}

TEST(TestRaggedSdpaDensify, PackAndUnpackRoundTrip)
{
    const auto buffer = raggedGraph(false, false);
    const auto dense
        = hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(buffer.data(), buffer.size());
    ASSERT_TRUE(dense.has_value());
    const auto& q = dense->primaries.front();
    const auto offsets = hipdnn_test_sdk::detail::raggedElementOffsets(q, Q_TOKENS.data(), B);
    EXPECT_EQ(offsets, (std::vector<int64_t>{0, 4 * H * D, 10 * H * D}));
    const auto lengths = hipdnn_test_sdk::detail::raggedLengths(q, offsets);
    EXPECT_EQ(lengths, Q_LENS);

    auto packed = packedValues(10, H * D, 3);
    std::vector<uint8_t> denseBytes(hipdnn_test_sdk::detail::raggedDenseBytes(q), 0);
    hipdnn_test_sdk::detail::copyRaggedTokens(
        q, offsets, lengths, reinterpret_cast<uint8_t*>(packed.data()), denseBytes.data(), true);
    const std::vector<float> expected = unpack(packed, Q_TOKENS, SQ, D);
    std::vector<float> got(expected.size());
    std::memcpy(got.data(), denseBytes.data(), denseBytes.size());
    EXPECT_EQ(got, expected);

    std::vector<float> back(packed.size(), -1.0F);
    hipdnn_test_sdk::detail::copyRaggedTokens(
        q, offsets, lengths, reinterpret_cast<uint8_t*>(back.data()), denseBytes.data(), false);
    EXPECT_EQ(back, packed);
}

// An all-zero bias changes nothing numerically but the native ragged plan declines it, so the
// same data runs once through the native ragged reference and once through the dense rewrite.
TEST(TestRaggedSdpaDensify, ZeroBiasThroughTheRewriteEqualsTheNativeRaggedPlan)
{
    auto q = packedValues(10, H * D, 11);
    auto k = packedValues(13, H * D, 12);
    auto v = packedValues(13, H * D, 13);
    std::vector<float> zeroBias(static_cast<size_t>(SQ * SKV), 0.0F);
    auto qOffsets = Q_TOKENS;
    auto kvOffsets = KV_TOKENS;

    const auto run = [&](bool bias) {
        const auto graph = raggedGraph(false, bias);
        std::vector<float> o(10 * H * D, -7.0F);
        std::vector<float> lse(10 * H, -7.0F);
        std::unordered_map<int64_t, void*> pack = {{Q, q.data()},
                                                   {K, k.data()},
                                                   {V, v.data()},
                                                   {O, o.data()},
                                                   {STATS, lse.data()},
                                                   {Q_OFFSETS, qOffsets.data()},
                                                   {KV_OFFSETS, kvOffsets.data()}};
        if(bias)
        {
            pack[BIAS] = zeroBias.data();
        }
        CpuReferenceGraphExecutor executor;
        EXPECT_TRUE(executor.isApplicable(const_cast<uint8_t*>(graph.data()), graph.size()));
        executor.execute(const_cast<uint8_t*>(graph.data()), graph.size(), pack);
        return std::make_pair(o, lse);
    };
    const auto native = run(false);
    const auto rewritten = run(true);
    for(size_t i = 0; i < native.first.size(); ++i)
    {
        EXPECT_NEAR(native.first[i], rewritten.first[i], 1e-6F) << "O element " << i;
    }
    for(size_t i = 0; i < native.second.size(); ++i)
    {
        EXPECT_NEAR(native.second[i], rewritten.second[i], 1e-5F) << "LSE element " << i;
    }
}

// A ragged backward graph (no native plan) through the executor equals the dense backward
// reference on the unpacked data with the same per-batch lengths.
TEST(TestRaggedSdpaDensify, RaggedBackwardEqualsDensePaddedBackward)
{
    auto q = packedValues(10, H * D, 21);
    auto k = packedValues(13, H * D, 22);
    auto v = packedValues(13, H * D, 23);
    auto dO = packedValues(10, H * D, 24);
    auto qOffsets = Q_TOKENS;
    auto kvOffsets = KV_TOKENS;

    // Dense forward on the unpacked data gives O and the LSE the backward consumes.
    const auto toTensor = [](const std::vector<float>& values, int64_t s, int64_t width) {
        Tensor<float> t({B, H, s, width});
        std::memcpy(t.memory().hostData(), values.data(), values.size() * sizeof(float));
        return t;
    };
    auto qd = toTensor(unpack(q, Q_TOKENS, SQ, D), SQ, D);
    auto kd = toTensor(unpack(k, KV_TOKENS, SKV, D), SKV, D);
    auto vd = toTensor(unpack(v, KV_TOKENS, SKV, D), SKV, D);
    auto dOd = toTensor(unpack(dO, Q_TOKENS, SQ, D), SQ, D);
    Tensor<int32_t> seqQ({B, 1, 1, 1});
    Tensor<int32_t> seqKv({B, 1, 1, 1});
    for(int64_t b = 0; b < B; ++b)
    {
        seqQ.setHostValue(static_cast<int32_t>(Q_LENS[static_cast<size_t>(b)]), b, 0, 0, 0);
        seqKv.setHostValue(static_cast<int32_t>(KV_LENS[static_cast<size_t>(b)]), b, 0, 0, 0);
    }
    Tensor<float> od({B, H, SQ, D});
    Tensor<float> lse({B, H, SQ, 1});
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward(qd,
                                kd,
                                vd,
                                od,
                                SdpaFwdOptions<float>{}
                                    .setScale(0.5F)
                                    .setMask(noMask)
                                    .setBand(-1, 0, false)
                                    .setSeqLens(&seqQ, &seqKv)
                                    .setLse(&lse));
    Tensor<float> dQd({B, H, SQ, D});
    Tensor<float> dKd({B, H, SKV, D});
    Tensor<float> dVd({B, H, SKV, D});
    CpuFpReferenceSdpa::backward(qd,
                                 kd,
                                 vd,
                                 od,
                                 dOd,
                                 dQd,
                                 dKd,
                                 dVd,
                                 SdpaBwdOptions<float>{}
                                     .setScale(0.5F)
                                     .setLse(&lse)
                                     .setMask(noMask)
                                     .setBand(-1, 0, false)
                                     .setSeqLens(&seqQ, &seqKv));

    // Pack O and the LSE the way the ragged graph expects them.
    std::vector<float> o(10 * H * D);
    std::vector<float> lsePacked(10 * H);
    for(int64_t b = 0; b < B; ++b)
    {
        for(int64_t tok = 0; tok < Q_LENS[static_cast<size_t>(b)]; ++tok)
        {
            for(int64_t h = 0; h < H; ++h)
            {
                const auto row = (Q_TOKENS[static_cast<size_t>(b)] + tok) * H + h;
                lsePacked[static_cast<size_t>(row)] = lse.getHostValue(b, h, tok, 0);
                for(int64_t d = 0; d < D; ++d)
                {
                    o[static_cast<size_t>(row * D + d)] = od.getHostValue(b, h, tok, d);
                }
            }
        }
    }

    const auto graph = raggedGraph(true, false);
    std::vector<float> dQ(10 * H * D, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> dK(13 * H * D, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> dV(13 * H * D, std::numeric_limits<float>::quiet_NaN());
    const std::unordered_map<int64_t, void*> pack = {{Q, q.data()},
                                                     {K, k.data()},
                                                     {V, v.data()},
                                                     {O, o.data()},
                                                     {DO, dO.data()},
                                                     {STATS, lsePacked.data()},
                                                     {DQ, dQ.data()},
                                                     {DK, dK.data()},
                                                     {DV, dV.data()},
                                                     {Q_OFFSETS, qOffsets.data()},
                                                     {KV_OFFSETS, kvOffsets.data()}};
    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(const_cast<uint8_t*>(graph.data()), graph.size()));
    executor.execute(const_cast<uint8_t*>(graph.data()), graph.size(), pack);

    const auto expectNear =
        [](const std::vector<float>& got, const Tensor<float>& want, const char* name) {
            ASSERT_EQ(got.size(), want.elementCount()) << name;
            for(size_t i = 0; i < got.size(); ++i)
            {
                EXPECT_NEAR(got[i], want.memory().hostData()[i], 1e-5F) << name << " element " << i;
            }
        };
    expectNear(unpack(dQ, Q_TOKENS, SQ, D), dQd, "dQ");
    expectNear(unpack(dK, KV_TOKENS, SKV, D), dKd, "dK");
    expectNear(unpack(dV, KV_TOKENS, SKV, D), dVd, "dV");
}

// vLLM's shape: ragged Q over a paged K/V cache (with and without a sink) runs through the
// ragged-as-dense rewrite and equals the dense reference on the unpacked Q, the gathered pages and
// the same per-batch lengths. The unreferenced cache block holds NaN, so reading it would show.
TEST(TestRaggedSdpaDensify, RaggedQOverPagedKvEqualsGatheredDense)
{
    for(const bool sink : {false, true})
    {
        SCOPED_TRACE(sink ? "with sink" : "no sink");
        auto q = packedValues(10, H * D, 31);
        Tensor<float> kCache({NUM_BLOCKS, H, BLOCK, D});
        Tensor<float> vCache({NUM_BLOCKS, H, BLOCK, D});
        kCache.fillWithRandomValues(-1.0F, 1.0F, 32);
        vCache.fillWithRandomValues(-1.0F, 1.0F, 33);
        for(int64_t h = 0; h < H; ++h)
        {
            for(int64_t s = 0; s < BLOCK; ++s)
            {
                for(int64_t d = 0; d < D; ++d)
                {
                    kCache.setHostValue(std::numeric_limits<float>::quiet_NaN(), 2, h, s, d);
                    vCache.setHostValue(std::numeric_limits<float>::quiet_NaN(), 2, h, s, d);
                }
            }
        }
        Tensor<int32_t> table({B, 1, TABLE, 1});
        Tensor<int32_t> seqQ({B, 1, 1, 1});
        Tensor<int32_t> seqKv({B, 1, 1, 1});
        for(int64_t b = 0; b < B; ++b)
        {
            for(int64_t i = 0; i < TABLE; ++i)
            {
                table.setHostValue(PAGE_TABLE[static_cast<size_t>(b * TABLE + i)], b, 0, i, 0);
            }
            seqQ.setHostValue(static_cast<int32_t>(Q_LENS[static_cast<size_t>(b)]), b, 0, 0, 0);
            seqKv.setHostValue(static_cast<int32_t>(KV_LENS[static_cast<size_t>(b)]), b, 0, 0, 0);
        }
        Tensor<float> sinkTensor({1, H, 1, 1});
        sinkTensor.fillWithRandomValues(-1.0F, 1.0F, 34);

        const auto graph = raggedQPagedKvGraph(sink);
        std::vector<float> o(10 * H * D, -7.0F);
        auto qOffsets = Q_TOKENS;
        std::unordered_map<int64_t, void*> pack = {{Q, q.data()},
                                                   {K, kCache.memory().hostData()},
                                                   {V, vCache.memory().hostData()},
                                                   {O, o.data()},
                                                   {Q_OFFSETS, qOffsets.data()},
                                                   {SEQ_Q, seqQ.memory().hostData()},
                                                   {SEQ_KV, seqKv.memory().hostData()},
                                                   {PAGE_TABLE_K, table.memory().hostData()},
                                                   {PAGE_TABLE_V, table.memory().hostData()}};
        if(sink)
        {
            pack[SINK] = sinkTensor.memory().hostData();
        }
        CpuReferenceGraphExecutor executor;
        ASSERT_TRUE(executor.isApplicable(const_cast<uint8_t*>(graph.data()), graph.size()));
        executor.execute(const_cast<uint8_t*>(graph.data()), graph.size(), pack);

        // Expected: gather the pages, run the dense reference, pack O like Q.
        Tensor<float> qd({B, H, SQ, D});
        const auto unpacked = unpack(q, Q_TOKENS, SQ, D);
        std::memcpy(qd.memory().hostData(), unpacked.data(), unpacked.size() * sizeof(float));
        auto kd = CpuFpReferenceSdpa::gatherPagedKv(kCache, table, seqKv, SKV);
        auto vd = CpuFpReferenceSdpa::gatherPagedKv(vCache, table, seqKv, SKV);
        Tensor<float> od({B, H, SQ, D});
        CpuFpReferenceSdpa::forward(qd,
                                    kd,
                                    vd,
                                    od,
                                    SdpaFwdOptions<float>{}
                                        .setScale(0.5F)
                                        .setBand(-1, 0, false)
                                        .setSeqLens(&seqQ, &seqKv)
                                        .setSink(sink ? &sinkTensor : nullptr));
        for(int64_t b = 0; b < B; ++b)
        {
            const auto first = Q_TOKENS[static_cast<size_t>(b)];
            for(int64_t tok = 0; tok < Q_LENS[static_cast<size_t>(b)]; ++tok)
            {
                for(int64_t h = 0; h < H; ++h)
                {
                    for(int64_t d = 0; d < D; ++d)
                    {
                        const auto got = o[static_cast<size_t>(((first + tok) * H + h) * D + d)];
                        ASSERT_FALSE(std::isnan(got)) << "b=" << b << " tok=" << tok;
                        EXPECT_NEAR(got, od.getHostValue(b, h, tok, d), 1e-6F)
                            << "b=" << b << " tok=" << tok << " h=" << h << " d=" << d;
                    }
                }
            }
        }
    }
}
