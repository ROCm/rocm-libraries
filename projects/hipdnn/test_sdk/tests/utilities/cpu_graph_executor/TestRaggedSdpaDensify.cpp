// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/RaggedLayoutTestSupport.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaDensifyTestGraphs.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaRaggedFwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaDensify.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_test_sdk::utilities::CpuFpReferenceSdpa;
using hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor;
using hipdnn_test_sdk::utilities::SdpaBwdOptions;
using hipdnn_test_sdk::utilities::SdpaFwdOptions;
using namespace hipdnn_test_sdk::utilities::ragged_densify_test;
using hipdnn_test_sdk::utilities::ragged_layout_test::stridedAuxBuffer;

TEST(TestRaggedSdpaDensify, RewritesRaggedPrimariesToDensePadding)
{
    const auto buffer = raggedGraph(false, false);
    const auto dense
        = hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(buffer.data(), buffer.size());
    ASSERT_TRUE(dense.has_value());
    EXPECT_EQ(dense->primaries.size(), 5u); // Q, K, V, O and the ragged LSE
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
    const auto lengths = hipdnn_test_sdk::detail::raggedLengths(q, offsets, nullptr);
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

namespace
{

// Runs the unbiased ragged forward graph and the same graph with an all-zero bias on the same
// data. A zero bias changes nothing numerically, but every ragged plan declines a bias, so the
// biased arm always runs through the dense rewrite. Returns the unbiased and biased (O, LSE).
template <typename T>
std::pair<std::pair<std::vector<T>, std::vector<float>>,
          std::pair<std::vector<T>, std::vector<float>>>
    runUnbiasedAndZeroBias(DataType dataType)
{
    const auto toT = [](const std::vector<float>& values) {
        std::vector<T> out;
        out.reserve(values.size());
        for(const float x : values)
        {
            out.push_back(static_cast<T>(x));
        }
        return out;
    };
    auto q = toT(packedValues(10, H * D, 11));
    auto k = toT(packedValues(13, H * D, 12));
    auto v = toT(packedValues(13, H * D, 13));
    std::vector<float> zeroBias(static_cast<size_t>(SQ * SKV), 0.0F);
    auto qOffsets = Q_TOKENS;
    auto kvOffsets = KV_TOKENS;

    const auto run = [&](bool bias) {
        const auto graph = raggedGraph(false, bias, /*ragged=*/true, dataType);
        std::vector<T> o(10 * H * D, static_cast<T>(-7.0F));
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
    return {run(false), run(true)};
}

// Compares the two arms of runUnbiasedAndZeroBias: O within oTolerance, LSE within 1e-5.
template <typename T>
void expectSameOutputs(const std::pair<std::vector<T>, std::vector<float>>& native,
                       const std::pair<std::vector<T>, std::vector<float>>& rewritten,
                       float oTolerance)
{
    for(size_t i = 0; i < native.first.size(); ++i)
    {
        EXPECT_NEAR(
            static_cast<float>(native.first[i]), static_cast<float>(rewritten.first[i]), oTolerance)
            << "O element " << i;
    }
    for(size_t i = 0; i < native.second.size(); ++i)
    {
        // A dead row's LSE is -inf on both sides (SEM-010); EXPECT_NEAR(-inf, -inf) fails on
        // googletest versions that compute |a - b| = NaN, so infinities compare exactly.
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

} // namespace

// The native ragged registry holds all-BF16 and FP8 E4M3 -> BF16 only, so with BF16 data the
// unbiased arm runs SdpaRaggedFwdPlan and the zero-bias arm runs the dense rewrite.
TEST(TestRaggedSdpaDensify, ZeroBiasThroughTheRewriteEqualsTheNativeRaggedPlan)
{
    using hipdnn_data_sdk::types::bfloat16;

    // The unbiased graph is the native plan's; the biased one is not (it takes the rewrite).
    const auto builders = hipdnn_test_sdk::detail::SdpaRaggedFwdSignatureKey::getPlanBuilders();
    const auto builderIt = builders.find(hipdnn_test_sdk::detail::SdpaRaggedFwdSignatureKey(
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16));
    ASSERT_NE(builderIt, builders.end());
    for(const bool bias : {false, true})
    {
        const auto graph = raggedGraph(false, bias, /*ragged=*/true, DataType::BFLOAT16);
        const GraphWrapper wrapper(graph.data(), graph.size());
        EXPECT_EQ(builderIt->second->isApplicable(wrapper.getNode(0), wrapper.getTensorMap()),
                  !bias)
            << (bias ? "zero-bias graph" : "unbiased graph");
    }

    // Both arms compute in float and round O to bf16, so they may differ by one bf16 step.
    const auto [native, rewritten] = runUnbiasedAndZeroBias<bfloat16>(DataType::BFLOAT16);
    expectSameOutputs(native, rewritten, 0.0078125F);
}

// FLOAT has no native ragged plan: both arms run through the dense rewrite, with and without
// the bias input.
TEST(TestRaggedSdpaDensify, ZeroBiasThroughTheRewriteEqualsTheUnbiasedRewrite)
{
    const auto [unbiased, zeroBias] = runUnbiasedAndZeroBias<float>(DataType::FLOAT);
    expectSameOutputs(unbiased, zeroBias, 1e-6F);
}

// Philox dropout over packed (THD) operands, validated through the RNG dump (SEM-015): the
// ragged graph runs through the dense rewrite, which keeps the dense [B, H, SQ, SKV] dump, and
// equals the dense reference on the unpacked data with the same lengths and the dump (scaled
// by 1 / (1 - p)) as its mask. The mask drops something, so the undropped graph differs.
TEST(TestRaggedSdpaDensify, PhiloxDropoutThroughTheRewriteAppliesTheDenseDump)
{
    constexpr float P = 0.25F;
    auto q = packedValues(10, H * D, 31);
    auto k = packedValues(13, H * D, 32);
    auto v = packedValues(13, H * D, 33);
    auto qOffsets = Q_TOKENS;
    auto kvOffsets = KV_TOKENS;
    int64_t seed = 1234;
    int64_t offset = 8;
    // A keep pattern no draw rule makes, at every element (rows and keys past a sequence's
    // lengths included, as the engines' dumps hold them).
    std::vector<float> dump(static_cast<size_t>(B * H * SQ * SKV));
    for(size_t i = 0; i < dump.size(); ++i)
    {
        dump[i] = (i * 7U + i / 5U) % 4U == 0U ? 0.0F : 1.0F;
    }

    const auto run = [&](const std::vector<uint8_t>& graph) {
        std::vector<float> o(10 * H * D, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> lse(10 * H, std::numeric_limits<float>::quiet_NaN());
        const std::unordered_map<int64_t, void*> pack = {{Q, q.data()},
                                                         {K, k.data()},
                                                         {V, v.data()},
                                                         {O, o.data()},
                                                         {STATS, lse.data()},
                                                         {Q_OFFSETS, qOffsets.data()},
                                                         {KV_OFFSETS, kvOffsets.data()},
                                                         {SEED, &seed},
                                                         {OFFSET, &offset},
                                                         {RNG_DUMP, dump.data()}};
        CpuReferenceGraphExecutor executor;
        EXPECT_TRUE(executor.isApplicable(const_cast<uint8_t*>(graph.data()), graph.size()));
        executor.execute(const_cast<uint8_t*>(graph.data()), graph.size(), pack);
        return o;
    };
    const auto dropped = run(withPhiloxDropout(raggedGraph(false, false), P));
    const auto undropped = run(raggedGraph(false, false));

    const auto toTensor = [](const std::vector<float>& values, int64_t s, int64_t width) {
        Tensor<float> t({B, H, s, width});
        std::memcpy(t.memory().hostData(), values.data(), values.size() * sizeof(float));
        return t;
    };
    auto qd = toTensor(unpack(q, Q_TOKENS, SQ, D), SQ, D);
    auto kd = toTensor(unpack(k, KV_TOKENS, SKV, D), SKV, D);
    auto vd = toTensor(unpack(v, KV_TOKENS, SKV, D), SKV, D);
    Tensor<float> mask({B, H, SQ, SKV});
    std::memcpy(mask.memory().hostData(), dump.data(), dump.size() * sizeof(float));
    Tensor<int32_t> seqQ({B, 1, 1, 1});
    Tensor<int32_t> seqKv({B, 1, 1, 1});
    for(int64_t b = 0; b < B; ++b)
    {
        seqQ.setHostValue(static_cast<int32_t>(Q_LENS[static_cast<size_t>(b)]), b, 0, 0, 0);
        seqKv.setHostValue(static_cast<int32_t>(KV_LENS[static_cast<size_t>(b)]), b, 0, 0, 0);
    }
    Tensor<float> od({B, H, SQ, D});
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
                                    .setDropout(&mask, 1.0F / (1.0F - P)));

    const auto got = unpack(dropped, Q_TOKENS, SQ, D);
    const auto plain = unpack(undropped, Q_TOKENS, SQ, D);
    double differs = 0.0;
    for(int64_t b = 0; b < B; ++b)
    {
        for(int64_t h = 0; h < H; ++h)
        {
            for(int64_t s = 0; s < Q_LENS[static_cast<size_t>(b)]; ++s)
            {
                for(int64_t d = 0; d < D; ++d)
                {
                    const auto i = static_cast<size_t>(((b * H + h) * SQ + s) * D + d);
                    EXPECT_NEAR(got[i], od.getHostValue(b, h, s, d), 1e-5F)
                        << "b=" << b << " h=" << h << " s=" << s << " d=" << d;
                    differs = std::max(differs, static_cast<double>(std::abs(got[i] - plain[i])));
                }
            }
        }
    }
    EXPECT_GT(differs, 1e-3) << "the dump dropped nothing";
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

// A ragged_offset on a paged K/V container would count tokens through its pages, which neither
// reference defines. The rewrite declines it, so the executor reports the graph not applicable
// instead of accepting it and throwing at execute. Ragged Q over the same paged K/V still rewrites.
TEST(TestRaggedSdpaDensify, RaggedOffsetOnPagedKvIsDeclined)
{
    const auto raggedKv = raggedQPagedKvGraph(false, true);
    EXPECT_FALSE(hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(raggedKv.data(), raggedKv.size())
                     .has_value());
    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(const_cast<uint8_t*>(raggedKv.data()), raggedKv.size()));

    const auto denseKv = raggedQPagedKvGraph(false);
    EXPECT_TRUE(hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(denseKv.data(), denseKv.size())
                    .has_value());
}

namespace
{

// Offset tables of the edited graphs (token units, multiplier = the tensor's seq stride unless
// stated). Not in the Uid enum.
constexpr int64_t STATS_OFFSETS = 100;
constexpr int64_t MAX_UID = 101;
constexpr int64_t SUM_EXP_UID = 102;
constexpr int64_t DQ_OFFSETS = 103;
constexpr int64_t DKV_OFFSETS = 104;

void addInt32(GraphT& graph, int64_t uid, int64_t count)
{
    graph.tensors.push_back(tensor(uid, DataType::INT32, {count, 1, 1, 1}, {1, 1, 1, 1}));
}

void addSeqLens(GraphT& graph)
{
    addInt32(graph, SEQ_Q, B);
    addInt32(graph, SEQ_KV, B);
}

const std::vector<int32_t> Q_LENS32 = {4, 6};
const std::vector<int32_t> KV_LENS32 = {8, 5};

// Token index of (b, tok) in a packed layout whose batch starts are `tokens`.
size_t tokenRow(const std::vector<int32_t>& tokens, int64_t b, int64_t tok)
{
    return static_cast<size_t>(tokens[static_cast<size_t>(b)] + tok);
}

} // namespace

// Ragged Max and Sum_exp are served through the rewrite with seq_len_q lengths. Stats, Max and
// Sum_exp sit at non-monotonic origins (batch 0 after batch 1); Max + log(Sum_exp) equals Stats,
// and Stats equals the run with monotonic origins and no Max / Sum_exp. Bottom-right causal
// masking leaves query 0 of batch 1 (6 queries, 5 keys) with no visible key: that row's Stats and
// Max are -inf and its Sum_exp is 0, checked exactly because a tolerance check on -inf is NaN.
TEST(TestRaggedSdpaDensify, RaggedMaxAndSumExpAtNonMonotonicOriginsEqualStats)
{
    const std::vector<int32_t> statsTokens = {6, 0, 0};
    const auto base = raggedGraph(false, false);
    const auto graph = editGraph(base, [&](GraphT& g) {
        addSeqLens(g);
        addInt32(g, STATS_OFFSETS, B + 1);
        for(const int64_t uid : {static_cast<int64_t>(STATS), MAX_UID, SUM_EXP_UID})
        {
            if(uid != STATS)
            {
                g.tensors.push_back(
                    tensor(uid, DataType::FLOAT, {B, H, SQ, 1}, {SQ * H, 1, H, 1}, STATS_OFFSETS));
            }
        }
        tensorOf(g, STATS).ragged_offset_tensor_uid = STATS_OFFSETS;
        auto* attrs = g.nodes.front()->attributes.AsSdpaAttributes();
        attrs->max_tensor_uid = MAX_UID;
        attrs->sum_exp_tensor_uid = SUM_EXP_UID;
        attrs->seq_len_q_tensor_uid = SEQ_Q;
        attrs->seq_len_kv_tensor_uid = SEQ_KV;
    });

    auto q = packedValues(10, H * D, 41);
    auto k = packedValues(13, H * D, 42);
    auto v = packedValues(13, H * D, 43);
    auto qOffsets = Q_TOKENS;
    auto kvOffsets = KV_TOKENS;
    auto seqQ = Q_LENS32;
    auto seqKv = KV_LENS32;
    auto statsOffsets = statsTokens;
    std::vector<float> o(10 * H * D, -7.0F);
    std::vector<float> stats(10 * H, -7.0F);
    std::vector<float> maxRows(10 * H, -7.0F);
    std::vector<float> sumExp(10 * H, -7.0F);
    const std::unordered_map<int64_t, void*> pack = {{Q, q.data()},
                                                     {K, k.data()},
                                                     {V, v.data()},
                                                     {O, o.data()},
                                                     {STATS, stats.data()},
                                                     {MAX_UID, maxRows.data()},
                                                     {SUM_EXP_UID, sumExp.data()},
                                                     {Q_OFFSETS, qOffsets.data()},
                                                     {KV_OFFSETS, kvOffsets.data()},
                                                     {STATS_OFFSETS, statsOffsets.data()},
                                                     {SEQ_Q, seqQ.data()},
                                                     {SEQ_KV, seqKv.data()}};
    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(const_cast<uint8_t*>(graph.data()), graph.size()));
    executor.execute(const_cast<uint8_t*>(graph.data()), graph.size(), pack);

    // Reference: monotonic Stats, no Max / Sum_exp.
    const auto baseline = raggedGraph(false, false);
    std::vector<float> oBase(10 * H * D, -7.0F);
    std::vector<float> statsBase(10 * H, -7.0F);
    const std::unordered_map<int64_t, void*> basePack = {{Q, q.data()},
                                                         {K, k.data()},
                                                         {V, v.data()},
                                                         {O, oBase.data()},
                                                         {STATS, statsBase.data()},
                                                         {Q_OFFSETS, qOffsets.data()},
                                                         {KV_OFFSETS, kvOffsets.data()}};
    executor.execute(const_cast<uint8_t*>(baseline.data()), baseline.size(), basePack);

    for(int64_t b = 0; b < B; ++b)
    {
        for(int64_t tok = 0; tok < Q_LENS[static_cast<size_t>(b)]; ++tok)
        {
            for(int64_t h = 0; h < H; ++h)
            {
                const auto at = tokenRow(statsTokens, b, tok) * H + static_cast<size_t>(h);
                const auto ref = tokenRow(Q_TOKENS, b, tok) * H + static_cast<size_t>(h);
                if(std::isinf(statsBase[ref]))
                {
                    EXPECT_EQ(statsBase[ref], -std::numeric_limits<float>::infinity())
                        << "b=" << b << " tok=" << tok << " h=" << h;
                    EXPECT_EQ(stats[at], statsBase[ref])
                        << "b=" << b << " tok=" << tok << " h=" << h;
                    EXPECT_EQ(maxRows[at], statsBase[ref])
                        << "b=" << b << " tok=" << tok << " h=" << h;
                    EXPECT_EQ(sumExp[at], 0.0F) << "b=" << b << " tok=" << tok << " h=" << h;
                    continue;
                }
                EXPECT_NEAR(maxRows[at] + std::log(sumExp[at]), stats[at], 1e-5F)
                    << "b=" << b << " tok=" << tok << " h=" << h;
                EXPECT_NEAR(stats[at], statsBase[ref], 1e-5F)
                    << "b=" << b << " tok=" << tok << " h=" << h;
            }
        }
    }
}

namespace
{

struct BackwardData
{
    std::vector<float> q = packedValues(10, H* D, 51);
    std::vector<float> k = packedValues(13, H* D, 52);
    std::vector<float> v = packedValues(13, H* D, 53);
    std::vector<float> o = packedValues(10, H* D, 54);
    std::vector<float> dO = packedValues(10, H* D, 55);
    // Stats [T, H]: lse[(token) * H + h], arbitrary but fixed per (token, h).
    std::vector<float> lseTH = packedValues(10, H, 56);
};

struct Gradients
{
    std::vector<float> dQ;
    std::vector<float> dK;
    std::vector<float> dV;
};

// Runs a ragged backward graph through the executor. Gradient buffers are NaN-filled so a gap
// that is written (or a row that is not) shows.
Gradients runBackward(const std::vector<uint8_t>& graph,
                      BackwardData& data,
                      std::vector<float>& lse,
                      std::unordered_map<int64_t, std::vector<int32_t>>& tables)
{
    Gradients g{std::vector<float>(10 * H * D, std::numeric_limits<float>::quiet_NaN()),
                std::vector<float>(13 * H * D, std::numeric_limits<float>::quiet_NaN()),
                std::vector<float>(13 * H * D, std::numeric_limits<float>::quiet_NaN())};
    std::unordered_map<int64_t, void*> pack = {{Q, data.q.data()},
                                               {K, data.k.data()},
                                               {V, data.v.data()},
                                               {O, data.o.data()},
                                               {DO, data.dO.data()},
                                               {STATS, lse.data()},
                                               {DQ, g.dQ.data()},
                                               {DK, g.dK.data()},
                                               {DV, g.dV.data()}};
    for(auto& [uid, values] : tables)
    {
        pack[uid] = values.data();
    }
    CpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(const_cast<uint8_t*>(graph.data()), graph.size()));
    executor.execute(const_cast<uint8_t*>(graph.data()), graph.size(), pack);
    return g;
}

void expectSame(const std::vector<float>& a, const std::vector<float>& b, const char* name)
{
    ASSERT_EQ(a.size(), b.size()) << name;
    for(size_t i = 0; i < a.size(); ++i)
    {
        EXPECT_NEAR(a[i], b[i], 1e-6F) << name << " element " << i;
    }
}

} // namespace

// Each tensor has its own origins. Q starts at token 0 (monotonic), while dQ sits at {6, 0} and
// dK/dV at {5, 0} (batch 0 after batch 1), with lengths from seq_len. The gradients, read at their
// own origins, equal those of the all-monotonic run.
TEST(TestRaggedSdpaDensify, BackwardGradientsAtIndependentOriginsEqualMonotonicRun)
{
    BackwardData data;
    std::unordered_map<int64_t, std::vector<int32_t>> baseTables
        = {{Q_OFFSETS, Q_TOKENS}, {KV_OFFSETS, KV_TOKENS}};
    auto lse = data.lseTH;
    const auto expected = runBackward(raggedGraph(true, false), data, lse, baseTables);

    const std::vector<int32_t> dqTokens = {6, 0, 0};
    const std::vector<int32_t> dkvTokens = {5, 0, 0};
    const auto graph = editGraph(raggedGraph(true, false), [&](GraphT& g) {
        addSeqLens(g);
        addInt32(g, DQ_OFFSETS, B + 1);
        addInt32(g, DKV_OFFSETS, B + 1);
        tensorOf(g, DQ).ragged_offset_tensor_uid = DQ_OFFSETS;
        tensorOf(g, DK).ragged_offset_tensor_uid = DKV_OFFSETS;
        tensorOf(g, DV).ragged_offset_tensor_uid = DKV_OFFSETS;
        auto* attrs = g.nodes.front()->attributes.AsSdpaBackwardAttributes();
        attrs->seq_len_q_tensor_uid = SEQ_Q;
        attrs->seq_len_kv_tensor_uid = SEQ_KV;
    });
    std::unordered_map<int64_t, std::vector<int32_t>> tables = {{Q_OFFSETS, Q_TOKENS},
                                                                {KV_OFFSETS, KV_TOKENS},
                                                                {DQ_OFFSETS, dqTokens},
                                                                {DKV_OFFSETS, dkvTokens},
                                                                {SEQ_Q, Q_LENS32},
                                                                {SEQ_KV, KV_LENS32}};
    const auto got = runBackward(graph, data, lse, tables);

    expectSame(unpack(got.dQ, dqTokens, Q_LENS, SQ, D), unpack(expected.dQ, Q_TOKENS, SQ, D), "dQ");
    expectSame(
        unpack(got.dK, dkvTokens, KV_LENS, SKV, D), unpack(expected.dK, KV_TOKENS, SKV, D), "dK");
    expectSame(
        unpack(got.dV, dkvTokens, KV_LENS, SKV, D), unpack(expected.dV, KV_TOKENS, SKV, D), "dV");
}

// The same logical Stats as [T, H] (L7b, the token-major default) and as [H, T] (L7a: head
// stride T = 10, sequence stride 1, element-unit offsets) give equal gradients.
TEST(TestRaggedSdpaDensify, BackwardStatsHeadMajorEqualsTokenMajor)
{
    BackwardData data;
    std::unordered_map<int64_t, std::vector<int32_t>> baseTables
        = {{Q_OFFSETS, Q_TOKENS}, {KV_OFFSETS, KV_TOKENS}};
    auto lseTH = data.lseTH;
    const auto expected = runBackward(raggedGraph(true, false), data, lseTH, baseTables);

    std::vector<float> lseHT(10 * H);
    for(int64_t t = 0; t < 10; ++t)
    {
        for(int64_t h = 0; h < H; ++h)
        {
            lseHT[static_cast<size_t>(h * 10 + t)] = data.lseTH[static_cast<size_t>(t * H + h)];
        }
    }
    const auto graph = editGraph(raggedGraph(true, false), [&](GraphT& g) {
        auto& stats = tensorOf(g, STATS);
        stats.strides = {10 * H, 10, 1, 1};
        stats.ragged_offset_multiplier = 1;
    });
    // Element-unit offsets of the [H, T] Stats: one element per token.
    std::unordered_map<int64_t, std::vector<int32_t>> tables
        = {{Q_OFFSETS, Q_TOKENS}, {KV_OFFSETS, KV_TOKENS}};
    // Q's table is shared with Stats (ragged_offset uid), so Stats needs its own table of element
    // offsets, which equals the token offsets here.
    const auto split = editGraph(graph, [&](GraphT& g) {
        addInt32(g, STATS_OFFSETS, B + 1);
        tensorOf(g, STATS).ragged_offset_tensor_uid = STATS_OFFSETS;
    });
    tables[STATS_OFFSETS] = Q_TOKENS;
    const auto got = runBackward(split, data, lseHT, tables);

    expectSame(unpack(got.dQ, Q_TOKENS, SQ, D), unpack(expected.dQ, Q_TOKENS, SQ, D), "dQ");
    expectSame(unpack(got.dK, KV_TOKENS, SKV, D), unpack(expected.dK, KV_TOKENS, SKV, D), "dK");
    expectSame(unpack(got.dV, KV_TOKENS, SKV, D), unpack(expected.dV, KV_TOKENS, SKV, D), "dV");
}

// The layout table round-trips through the pack/unpack copies: every live element reaches the
// dense form and returns to its address; elements the layout does not address stay untouched.
TEST(TestRaggedSdpaDensify, LayoutTableRoundTrips)
{
    using hipdnn_test_sdk::utilities::ragged_layout_test::raggedLayoutCase;
    for(const char* id : {"L2", "L4", "L5", "L6", "L7a", "L8", "L9"})
    {
        SCOPED_TRACE(id);
        const auto& layout = raggedLayoutCase(id);
        hipdnn_test_sdk::detail::RaggedSdpaPrimary primary;
        primary.dims = layout.dims;
        primary.packedStrides = layout.strides;
        primary.elementSize = sizeof(float);
        std::vector<int64_t> offsets;
        offsets.reserve(layout.offsets.size());
        for(size_t i = 0; i < layout.offsets.size(); ++i)
        {
            offsets.push_back(static_cast<int64_t>(layout.offsets[i]) * layout.multiplier);
        }
        const auto lengths = hipdnn_test_sdk::detail::raggedLengths(
            primary, offsets, layout.seqLen.empty() ? nullptr : layout.seqLen.data());
        EXPECT_EQ(lengths, layout.lengths);

        std::vector<float> packed(layout.space, -1.0F);
        for(int64_t b = 0; b < layout.batch(); ++b)
        {
            for(int64_t h = 0; h < layout.dims[1]; ++h)
            {
                for(int64_t s = 0; s < lengths[static_cast<size_t>(b)]; ++s)
                {
                    for(int64_t d = 0; d < layout.dims[3]; ++d)
                    {
                        packed[static_cast<size_t>(layout.address(b, h, s, d))]
                            = static_cast<float>(1 + b * 1000 + h * 100 + s * 10 + d);
                    }
                }
            }
        }
        EXPECT_EQ(hipdnn_test_sdk::detail::raggedPackedBytes(primary, offsets, lengths),
                  layout.space * sizeof(float));

        std::vector<uint8_t> dense(hipdnn_test_sdk::detail::raggedDenseBytes(primary), 0);
        hipdnn_test_sdk::detail::copyRaggedTokens(primary,
                                                  offsets,
                                                  lengths,
                                                  reinterpret_cast<uint8_t*>(packed.data()),
                                                  dense.data(),
                                                  true);
        std::vector<float> back(packed.size(), -1.0F);
        hipdnn_test_sdk::detail::copyRaggedTokens(primary,
                                                  offsets,
                                                  lengths,
                                                  reinterpret_cast<uint8_t*>(back.data()),
                                                  dense.data(),
                                                  false);
        EXPECT_EQ(back, packed);
    }
}

// Aliasing origins and out-of-range lengths are rejected by name, not copied.
TEST(TestRaggedSdpaDensify, RaggedLengthsRejectsInvalidLayouts)
{
    using hipdnn_test_sdk::utilities::ragged_layout_test::raggedLayoutCase;
    for(const char* id : {"L10", "L11"})
    {
        const auto& layout = raggedLayoutCase(id);
        hipdnn_test_sdk::detail::RaggedSdpaPrimary primary;
        primary.dims = layout.dims;
        primary.packedStrides = layout.strides;
        std::vector<int64_t> offsets;
        offsets.reserve(layout.offsets.size());
        for(const auto o : layout.offsets)
        {
            offsets.push_back(static_cast<int64_t>(o) * layout.multiplier);
        }
        EXPECT_THROW(hipdnn_test_sdk::detail::raggedLengths(
                         primary, offsets, layout.seqLen.empty() ? nullptr : layout.seqLen.data()),
                     std::invalid_argument)
            << id;
    }
    // seq_len beyond S_max.
    const auto& l9 = raggedLayoutCase("L9");
    hipdnn_test_sdk::detail::RaggedSdpaPrimary primary;
    primary.dims = l9.dims;
    primary.packedStrides = l9.strides;
    const std::vector<int32_t> tooLong = {2, 5};
    EXPECT_THROW(hipdnn_test_sdk::detail::raggedLengths(primary, {10, 0, 0}, tooLong.data()),
                 std::invalid_argument);
}

namespace
{

// Forward graph with seq_len_q/kv (lengths shorter than the offset blocks) whose token offsets
// and seq_len tensors have batch stride `auxStride`; a bias sends it through the dense rewrite,
// without one the native ragged plan runs it. Returns O and Stats.
std::pair<std::vector<float>, std::vector<float>> runWithAuxStride(bool bias, int64_t auxStride)
{
    const auto graph = editGraph(raggedGraph(false, bias), [&](GraphT& g) {
        addSeqLens(g);
        for(const int64_t uid : {Q_OFFSETS, KV_OFFSETS, SEQ_Q, SEQ_KV})
        {
            tensorOf(g, uid).strides = {auxStride, 1, 1, 1};
        }
        auto* attrs = g.nodes.front()->attributes.AsSdpaAttributes();
        attrs->seq_len_q_tensor_uid = SEQ_Q;
        attrs->seq_len_kv_tensor_uid = SEQ_KV;
        attrs->padding_mask = true;
    });
    auto q = packedValues(10, H * D, 61);
    auto k = packedValues(13, H * D, 62);
    auto v = packedValues(13, H * D, 63);
    std::vector<float> zeroBias(static_cast<size_t>(SQ * SKV), 0.0F);
    auto qOffsets = stridedAuxBuffer(Q_TOKENS, auxStride);
    auto kvOffsets = stridedAuxBuffer(KV_TOKENS, auxStride);
    auto seqQ = stridedAuxBuffer(Q_LENS32, auxStride);
    auto seqKv = stridedAuxBuffer(KV_LENS32, auxStride);
    std::vector<float> o(10 * H * D, -7.0F);
    std::vector<float> stats(10 * H, -7.0F);
    std::unordered_map<int64_t, void*> pack = {{Q, q.data()},
                                               {K, k.data()},
                                               {V, v.data()},
                                               {O, o.data()},
                                               {STATS, stats.data()},
                                               {Q_OFFSETS, qOffsets.data()},
                                               {KV_OFFSETS, kvOffsets.data()},
                                               {SEQ_Q, seqQ.data()},
                                               {SEQ_KV, seqKv.data()}};
    if(bias)
    {
        pack[BIAS] = zeroBias.data();
    }
    CpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(const_cast<uint8_t*>(graph.data()), graph.size()));
    executor.execute(const_cast<uint8_t*>(graph.data()), graph.size(), pack);
    return {o, stats};
}

void expectStridedEqualsPacked(bool bias)
{
    const auto packed = runWithAuxStride(bias, 1);
    const auto strided = runWithAuxStride(bias, 2);
    ASSERT_EQ(packed.first.size(), strided.first.size());
    for(size_t i = 0; i < packed.first.size(); ++i)
    {
        EXPECT_EQ(packed.first[i], strided.first[i]) << "O element " << i;
    }
    for(size_t i = 0; i < packed.second.size(); ++i)
    {
        EXPECT_EQ(packed.second[i], strided.second[i]) << "Stats element " << i;
    }
}

} // namespace

// Token offsets and seq_len tensors with batch stride 2 (odd slots poisoned) give the packed
// run's results through the dense rewrite.
TEST(TestRaggedSdpaDensify, StridedAuxEqualsPacked)
{
    expectStridedEqualsPacked(true);
}

// The same through the native ragged plan.
TEST(TestRaggedSdpaDensify, StridedAuxEqualsPackedNativePlan)
{
    expectStridedEqualsPacked(false);
}

namespace
{

bool densifies(const std::vector<uint8_t>& graph)
{
    return hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(graph.data(), graph.size()).has_value();
}

} // namespace

// An offset table or seq_len tensor whose values lie off the batch axis would be read at
// strides past its buffer; the graph is declined instead.
TEST(TestRaggedSdpaDensify, OffsetTableOffTheBatchAxisIsDeclined)
{
    const auto graph = editGraph(raggedGraph(false, true), [](GraphT& g) {
        auto& offsets = tensorOf(g, Q_OFFSETS);
        offsets.dims = {1, B + 1, 1, 1};
        offsets.strides = {B + 1, 1, 1, 1};
    });
    EXPECT_FALSE(densifies(graph));
    EXPECT_TRUE(densifies(raggedGraph(false, true)));
}

// Every offset table is read with Q's batch, so a ragged K/V with a smaller batch would be
// read past its own table; the graph is declined instead.
TEST(TestRaggedSdpaDensify, RaggedRoleWithAnotherBatchIsDeclined)
{
    // K/V get `batch` and a KV offset table sized for it, so only the batch differs from Q's.
    for(const int64_t batch : {B - 1, B + 1})
    {
        const auto graph = editGraph(raggedGraph(false, true), [&](GraphT& g) {
            for(const int64_t uid : {static_cast<int64_t>(K), static_cast<int64_t>(V)})
            {
                tensorOf(g, uid).dims[0] = batch;
            }
            tensorOf(g, KV_OFFSETS).dims = {batch + 1, 1, 1, 1};
        });
        EXPECT_FALSE(densifies(graph)) << "K/V batch " << batch;
    }
}

TEST(TestRaggedSdpaDensify, SeqLenOffTheBatchAxisIsDeclined)
{
    // Reshapes one seq_len tensor (SEQ_Q or SEQ_KV) onto `dims`; the other stays [B, 1, 1, 1].
    const auto make
        = [](int64_t reshaped, std::vector<int64_t> dims, std::vector<int64_t> strides) {
              return editGraph(raggedGraph(false, true), [&](GraphT& g) {
                  addSeqLens(g);
                  tensorOf(g, reshaped).dims = dims;
                  tensorOf(g, reshaped).strides = strides;
                  auto* attrs = g.nodes.front()->attributes.AsSdpaAttributes();
                  attrs->seq_len_q_tensor_uid = SEQ_Q;
                  attrs->seq_len_kv_tensor_uid = SEQ_KV;
              });
          };
    for(const int64_t uid : {static_cast<int64_t>(SEQ_Q), static_cast<int64_t>(SEQ_KV)})
    {
        EXPECT_FALSE(densifies(make(uid, {1, B, 1, 1}, {B, 1, 1, 1}))) << "uid " << uid;
        EXPECT_TRUE(densifies(make(uid, {B, 1, 1, 1}, {1, 1, 1, 1}))) << "uid " << uid;
    }
}

// The rewrite reads seq_len values as INT32, so any other element type is declined, whichever
// of seq_len_q and seq_len_kv carries it.
TEST(TestRaggedSdpaDensify, NonInt32SeqLenIsDeclined)
{
    const auto make = [](int64_t typed, DataType type) {
        return editGraph(raggedGraph(false, true), [&](GraphT& g) {
            addSeqLens(g);
            tensorOf(g, typed).data_type = type;
            auto* attrs = g.nodes.front()->attributes.AsSdpaAttributes();
            attrs->seq_len_q_tensor_uid = SEQ_Q;
            attrs->seq_len_kv_tensor_uid = SEQ_KV;
        });
    };
    for(const int64_t uid : {static_cast<int64_t>(SEQ_Q), static_cast<int64_t>(SEQ_KV)})
    {
        EXPECT_FALSE(densifies(make(uid, DataType::INT64))) << "uid " << uid;
        EXPECT_TRUE(densifies(make(uid, DataType::INT32))) << "uid " << uid;
    }
}

// A non-ragged Q or K of rank below 4 is declined, not read past its dims. The ragged side is
// the other of Q and K; seq_len tensors cover the dense one.
TEST(TestRaggedSdpaDensify, NonRaggedQOrKBelowRankFourIsDeclined)
{
    const auto make = [](bool denseQ, bool lowRank) {
        return editGraph(raggedGraph(false, false), [&](GraphT& g) {
            addSeqLens(g);
            auto* attrs = g.nodes.front()->attributes.AsSdpaAttributes();
            attrs->seq_len_q_tensor_uid = SEQ_Q;
            attrs->seq_len_kv_tensor_uid = SEQ_KV;
            const std::vector<int64_t> group
                = denseQ ? std::vector<int64_t>{Q, O, STATS} : std::vector<int64_t>{K, V};
            for(const int64_t uid : group)
            {
                auto& t = tensorOf(g, uid);
                t.ragged_offset_tensor_uid.reset();
                t.ragged_offset_multiplier = 1;
                if(uid == STATS)
                {
                    t.strides = {H * SQ, SQ, 1, 1};
                }
                else
                {
                    t.strides = (uid == K || uid == V) ? denseStrides(SKV) : denseStrides(SQ);
                }
            }
            if(lowRank)
            {
                auto& t = tensorOf(g, denseQ ? static_cast<int64_t>(Q) : static_cast<int64_t>(K));
                t.dims = {B, denseQ ? SQ : SKV};
                t.strides = {t.dims[1], 1};
            }
        });
    };
    for(const bool denseQ : {true, false})
    {
        EXPECT_TRUE(densifies(make(denseQ, false))) << "dense Q " << denseQ;
        EXPECT_FALSE(densifies(make(denseQ, true))) << "dense Q " << denseQ;
    }
}

// An INT64 offset table densifies to the same graph and lengths as an INT32 one.
TEST(TestRaggedSdpaDensify, Int64OffsetTablesMatchInt32)
{
    const auto int64Graph = editGraph(raggedGraph(false, true), [](GraphT& g) {
        tensorOf(g, Q_OFFSETS).data_type = DataType::INT64;
        tensorOf(g, KV_OFFSETS).data_type = DataType::INT64;
    });
    const auto int32Graph = raggedGraph(false, true);
    const auto dense32
        = hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(int32Graph.data(), int32Graph.size());
    const auto dense64
        = hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(int64Graph.data(), int64Graph.size());
    ASSERT_TRUE(dense32.has_value());
    ASSERT_TRUE(dense64.has_value());
    EXPECT_EQ(dense64->graph, dense32->graph);
    ASSERT_EQ(dense64->primaries.size(), dense32->primaries.size());
    for(size_t i = 0; i < dense64->primaries.size(); ++i)
    {
        EXPECT_EQ(dense32->primaries[i].offsetElementSize, sizeof(int32_t));
        EXPECT_EQ(dense64->primaries[i].offsetElementSize, sizeof(int64_t));
    }

    // The same table values, read at either width, give the same element offsets.
    const std::vector<int64_t> tokens64(Q_TOKENS.begin(), Q_TOKENS.end());
    const auto& primary32 = dense32->primaries.front();
    const auto& primary64 = dense64->primaries.front();
    EXPECT_EQ(hipdnn_test_sdk::detail::raggedElementOffsets(primary64, tokens64.data(), B),
              hipdnn_test_sdk::detail::raggedElementOffsets(primary32, Q_TOKENS.data(), B));

    // A host copy sized by raggedOffsetBytes holds a strided table of either width: values at
    // every stride-th slot, a sentinel in the gaps, so a read at the wrong slot or width differs.
    for(const int64_t stride : {int64_t{1}, int64_t{3}})
    {
        auto strided32 = primary32;
        auto strided64 = primary64;
        strided32.offsetStride = stride;
        strided64.offsetStride = stride;
        const auto slots = static_cast<size_t>(B * stride + 1);
        ASSERT_EQ(hipdnn_test_sdk::detail::raggedOffsetBytes(strided32, B), slots * 4);
        ASSERT_EQ(hipdnn_test_sdk::detail::raggedOffsetBytes(strided64, B), slots * 8);

        std::vector<int32_t> table32(slots, -1);
        std::vector<int64_t> table64(slots, -1);
        for(int64_t b = 0; b <= B; ++b)
        {
            table32[static_cast<size_t>(b * stride)] = Q_TOKENS[static_cast<size_t>(b)];
            table64[static_cast<size_t>(b * stride)] = Q_TOKENS[static_cast<size_t>(b)];
        }
        EXPECT_EQ(hipdnn_test_sdk::detail::raggedElementOffsets(strided64, table64.data(), B),
                  hipdnn_test_sdk::detail::raggedElementOffsets(strided32, table32.data(), B))
            << "stride " << stride;
    }
}

// A primary whose data type has no native mapping is declined, not thrown on.
TEST(TestRaggedSdpaDensify, UnmappedPrimaryDataTypeIsDeclined)
{
    const auto make = [](DataType type) {
        return editGraph(raggedGraph(false, true),
                         [&](GraphT& g) { tensorOf(g, Q).data_type = type; });
    };
    EXPECT_TRUE(densifies(make(DataType::FLOAT)));
    EXPECT_FALSE(densifies(make(DataType::INT8)));
}

// A ragged_offset tensor another attribute still names stays in the dense graph.
TEST(TestRaggedSdpaDensify, OffsetTableNamedByAnotherAttributeIsKept)
{
    const auto graph = editGraph(raggedGraph(false, true), [](GraphT& g) {
        g.nodes.front()->attributes.AsSdpaAttributes()->scale_tensor_uid = Q_OFFSETS;
    });
    const auto dense = hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(graph.data(), graph.size());
    ASSERT_TRUE(dense.has_value());
    const auto rewritten = UnPackGraph(dense->graph.data());
    const auto has = [&](int64_t uid) {
        return std::any_of(rewritten->tensors.begin(),
                           rewritten->tensors.end(),
                           [&](const auto& t) { return t->uid == uid; });
    };
    EXPECT_TRUE(has(Q_OFFSETS));
    EXPECT_FALSE(has(KV_OFFSETS));
}

// The same for a table named only by an attribute of the backward node.
TEST(TestRaggedSdpaDensify, OffsetTableNamedByBackwardAttributeIsKept)
{
    const auto graph = editGraph(raggedGraph(true, false), [](GraphT& g) {
        g.nodes.front()->attributes.AsSdpaBackwardAttributes()->dbias_tensor_uid = Q_OFFSETS;
    });
    const auto dense = hipdnn_test_sdk::detail::densifyRaggedSdpaGraph(graph.data(), graph.size());
    ASSERT_TRUE(dense.has_value());
    const auto rewritten = UnPackGraph(dense->graph.data());
    const auto has = [&](int64_t uid) {
        return std::any_of(rewritten->tensors.begin(),
                           rewritten->tensors.end(),
                           [&](const auto& t) { return t->uid == uid; });
    };
    EXPECT_TRUE(has(Q_OFFSETS));
    EXPECT_FALSE(has(KV_OFFSETS));
}
