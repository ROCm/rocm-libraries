// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Dropout probability 0 without an rng_dump (isSdpaZeroProbabilityDropout): the references treat
// it as no dropout, a hipDNN reference choice (SEM-015). Probability 0 drops no element, so the
// result must be bit-identical to the same graph with no dropout fields.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/SdpaFwdTestGraphs.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor;

namespace
{

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t LSE_UID = 5;
constexpr int64_t SEQ_LEN_Q_UID = 6;
constexpr int64_t SEQ_LEN_KV_UID = 7;
constexpr int64_t MASK_UID = 20;
constexpr int64_t SCALE_UID = 21;
constexpr int64_t RNG_DUMP_UID = 22;
constexpr int64_t SEED_UID = 23;
constexpr int64_t OFFSET_UID = 24;

// FLOAT Q/K/V/O [B=2, H=2, S=5, D=8] and an LSE [2, 2, 5, 1].
const std::vector<int64_t> DIMS = {2, 2, 5, 8};
const std::vector<int64_t> LSE_DIMS = {2, 2, 5, 1};
const std::vector<int64_t> SCORE_DIMS = {2, 2, 5, 5};

struct ExtraTensor
{
    int64_t uid;
    DataType dataType;
    std::vector<int64_t> dims;
};

const ExtraTensor SEED{SEED_UID, DataType::INT64, {1}};
const ExtraTensor OFFSET{OFFSET_UID, DataType::INT64, {1}};

flatbuffers::FlatBufferBuilder makeGraph(SdpaAttributesT attrs,
                                         const std::vector<ExtraTensor>& extras)
{
    attrs.q_tensor_uid = Q_UID;
    attrs.k_tensor_uid = K_UID;
    attrs.v_tensor_uid = V_UID;
    attrs.o_tensor_uid = O_UID;
    attrs.stats_tensor_uid = LSE_UID;
    attrs.generate_stats = true;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    std::vector<std::vector<int64_t>> strides;
    strides.reserve(extras.size() + 5);
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
    for(const auto& extra : extras)
    {
        add(extra.uid, extra.dataType, extra.dims);
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
    return builder;
}

bool applicable(const SdpaAttributesT& attrs, const std::vector<ExtraTensor>& extras)
{
    auto graph = makeGraph(attrs, extras);
    return CpuReferenceGraphExecutor().isApplicable(graph.GetBufferPointer(), graph.GetSize());
}

SdpaAttributesT zeroProbability()
{
    SdpaAttributesT attrs;
    attrs.dropout_probability = 0.0F;
    attrs.seed_tensor_uid = SEED_UID;
    attrs.offset_tensor_uid = OFFSET_UID;
    return attrs;
}

// The graph's outputs after one run on fixed inputs.
struct Outputs
{
    std::vector<float> o;
    std::vector<float> lse;
};

Outputs run(const SdpaAttributesT& attrs, const std::vector<ExtraTensor>& extras)
{
    auto graph = makeGraph(attrs, extras);
    Tensor<float> q(DIMS);
    Tensor<float> k(DIMS);
    Tensor<float> v(DIMS);
    Tensor<float> o(DIMS);
    Tensor<float> lse(LSE_DIMS);
    q.fillWithRandomValues(-1.0F, 1.0F, 91);
    k.fillWithRandomValues(-1.0F, 1.0F, 92);
    v.fillWithRandomValues(-1.0F, 1.0F, 93);
    Tensor<int64_t> seed({1});
    Tensor<int64_t> offset({1});
    seed.fillWithValue(1234);
    offset.fillWithValue(7);
    Tensor<int32_t> seqLenQ({2, 1, 1, 1});
    Tensor<int32_t> seqLenKv({2, 1, 1, 1});
    seqLenQ.memory().hostData()[0] = 3;
    seqLenQ.memory().hostData()[1] = 5;
    seqLenKv.memory().hostData()[0] = 4;
    seqLenKv.memory().hostData()[1] = 2;
    Tensor<float> mask(SCORE_DIMS);
    mask.fillWithValue(1.0F);
    Tensor<float> scale({1});
    scale.fillWithValue(1.0F);

    std::unordered_map<int64_t, void*> pack = {{Q_UID, q.memory().hostData()},
                                               {K_UID, k.memory().hostData()},
                                               {V_UID, v.memory().hostData()},
                                               {O_UID, o.memory().hostData()},
                                               {LSE_UID, lse.memory().hostData()}};
    const std::unordered_map<int64_t, void*> extraData
        = {{SEED_UID, seed.memory().hostData()},
           {OFFSET_UID, offset.memory().hostData()},
           {SEQ_LEN_Q_UID, seqLenQ.memory().hostData()},
           {SEQ_LEN_KV_UID, seqLenKv.memory().hostData()},
           {MASK_UID, mask.memory().hostData()},
           {SCALE_UID, scale.memory().hostData()}};
    for(const auto& extra : extras)
    {
        pack[extra.uid] = extraData.at(extra.uid);
    }
    CpuReferenceGraphExecutor().execute(graph.GetBufferPointer(), graph.GetSize(), pack);
    const auto* oData = o.memory().hostData();
    const auto* lseData = lse.memory().hostData();
    return {{oData, oData + o.elementCount()}, {lseData, lseData + lse.elementCount()}};
}

bool bitsEqual(const std::vector<float>& a, const std::vector<float>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

} // namespace

// Probability 0 without an rng_dump is accepted with or without a seed and an offset; each
// neighbouring form keeps its own rule.
TEST(TestSdpaFwdPlanZeroDropout, Applicability)
{
    EXPECT_TRUE(applicable(zeroProbability(), {SEED, OFFSET}));

    auto noSeedOrOffset = zeroProbability();
    noSeedOrOffset.seed_tensor_uid.reset();
    noSeedOrOffset.offset_tensor_uid.reset();
    EXPECT_TRUE(applicable(noSeedOrOffset, {}));

    auto seedOnly = zeroProbability();
    seedOnly.offset_tensor_uid.reset();
    EXPECT_TRUE(applicable(seedOnly, {SEED}));

    auto negativeZero = zeroProbability();
    negativeZero.dropout_probability = -0.0F;
    EXPECT_TRUE(applicable(negativeZero, {SEED, OFFSET}));
}

// Each case changes one field of the accepted probability-0 graph.
TEST(TestSdpaFwdPlanZeroDropout, DeclinesEveryOtherForm)
{
    // A seed or offset the graph names but does not hold.
    EXPECT_FALSE(applicable(zeroProbability(), {OFFSET}));
    EXPECT_FALSE(applicable(zeroProbability(), {SEED}));

    // A probability above 0 still needs the rng_dump.
    auto dropping = zeroProbability();
    dropping.dropout_probability = 0.1F;
    EXPECT_FALSE(applicable(dropping, {SEED, OFFSET}));
    dropping.dropout_probability = std::numeric_limits<float>::denorm_min();
    EXPECT_FALSE(applicable(dropping, {SEED, OFFSET}));
    dropping.dropout_probability = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(applicable(dropping, {SEED, OFFSET}));

    // Probability 0 beside cuDNN's explicit mask and scale is a graph half in each form.
    auto withMask = zeroProbability();
    withMask.dropout_mask_tensor_uid = MASK_UID;
    withMask.dropout_scale_tensor_uid = SCALE_UID;
    EXPECT_FALSE(applicable(withMask,
                            {SEED,
                             OFFSET,
                             {MASK_UID, DataType::FLOAT, SCORE_DIMS},
                             {SCALE_UID, DataType::FLOAT, {1}}}));
    auto withMaskOnly = zeroProbability();
    withMaskOnly.dropout_mask_tensor_uid = MASK_UID;
    EXPECT_FALSE(applicable(withMaskOnly, {SEED, OFFSET, {MASK_UID, DataType::FLOAT, SCORE_DIMS}}));
    auto withScale = zeroProbability();
    withScale.dropout_scale_tensor_uid = SCALE_UID;
    EXPECT_FALSE(applicable(withScale, {SEED, OFFSET, {SCALE_UID, DataType::FLOAT, {1}}}));
}

// With an rng_dump, probability 0 is the Philox-with-dump form, unchanged (the dump is the mask).
TEST(TestSdpaFwdPlanZeroDropout, WithAnRngDumpKeepsTheDumpForm)
{
    auto withDump = zeroProbability();
    withDump.rng_dump_tensor_uid = RNG_DUMP_UID;
    EXPECT_TRUE(applicable(withDump, {SEED, OFFSET, {RNG_DUMP_UID, DataType::FLOAT, SCORE_DIMS}}));
    EXPECT_FALSE(applicable(withDump, {SEED, OFFSET, {RNG_DUMP_UID, DataType::HALF, SCORE_DIMS}}));
}

// O and LSE are bit-identical to the graph without dropout, dense and padded with a band.
TEST(TestSdpaFwdPlanZeroDropout, EqualsNoDropoutBitForBit)
{
    const auto noDropout = run(SdpaAttributesT{}, {});
    const auto zero = run(zeroProbability(), {SEED, OFFSET});
    EXPECT_TRUE(bitsEqual(zero.o, noDropout.o));
    EXPECT_TRUE(bitsEqual(zero.lse, noDropout.lse));

    auto noSeedOrOffset = zeroProbability();
    noSeedOrOffset.seed_tensor_uid.reset();
    noSeedOrOffset.offset_tensor_uid.reset();
    const auto bare = run(noSeedOrOffset, {});
    EXPECT_TRUE(bitsEqual(bare.o, noDropout.o));
    EXPECT_TRUE(bitsEqual(bare.lse, noDropout.lse));

    SdpaAttributesT padded;
    padded.seq_len_q_tensor_uid = SEQ_LEN_Q_UID;
    padded.seq_len_kv_tensor_uid = SEQ_LEN_KV_UID;
    padded.padding_mask = true;
    padded.right_bound = 0;
    padded.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
    const std::vector<ExtraTensor> lengths = {{SEQ_LEN_Q_UID, DataType::INT32, {2, 1, 1, 1}},
                                              {SEQ_LEN_KV_UID, DataType::INT32, {2, 1, 1, 1}}};
    auto paddedZero = padded;
    paddedZero.dropout_probability = 0.0F;
    paddedZero.seed_tensor_uid = SEED_UID;
    paddedZero.offset_tensor_uid = OFFSET_UID;
    auto paddedZeroExtras = lengths;
    paddedZeroExtras.push_back(SEED);
    paddedZeroExtras.push_back(OFFSET);
    const auto paddedNoDropout = run(padded, lengths);
    const auto paddedZeroOut = run(paddedZero, paddedZeroExtras);
    EXPECT_TRUE(bitsEqual(paddedZeroOut.o, paddedNoDropout.o));
    EXPECT_TRUE(bitsEqual(paddedZeroOut.lse, paddedNoDropout.lse));
    // The padded case differs from the dense one, so the comparison above is not vacuous.
    EXPECT_FALSE(bitsEqual(paddedNoDropout.o, noDropout.o));
}

namespace
{

namespace fp8 = hipdnn_test_sdk::utilities::sdpa_graph_test;

bool fp8Applicable(const fp8::SdpaGraphSpec& spec)
{
    auto graph = fp8::makeSdpaGraph(spec);
    return CpuReferenceGraphExecutor().isApplicable(graph.GetBufferPointer(), graph.GetSize());
}

// O (bf16 bits) and the two amax outputs of one run of an fp8 graph with the full cuDNN sdpa_fp8
// scaling, on fixed inputs.
struct Fp8Outputs
{
    std::vector<uint16_t> o;
    float amaxS = 0.0F;
    float amaxO = 0.0F;
};

Fp8Outputs runFp8(const fp8::SdpaGraphSpec& spec)
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
    float descaleQ = 0.5F;
    float descaleK = 2.0F;
    float descaleV = 1.5F;
    float descaleS = 0.25F;
    float scaleS = 4.0F;
    float scaleO = 3.0F;
    float amaxS = 0.0F;
    float amaxO = 0.0F;
    int64_t seed = 1234;
    int64_t offset = 7;
    const std::unordered_map<int64_t, void*> pack = {{fp8::Q, q.memory().hostData()},
                                                     {fp8::K, k.memory().hostData()},
                                                     {fp8::V, v.memory().hostData()},
                                                     {fp8::O, o.memory().hostData()},
                                                     {fp8::DESCALE_Q, &descaleQ},
                                                     {fp8::DESCALE_K, &descaleK},
                                                     {fp8::DESCALE_V, &descaleV},
                                                     {fp8::DESCALE_S, &descaleS},
                                                     {fp8::SCALE_S, &scaleS},
                                                     {fp8::SCALE_O, &scaleO},
                                                     {fp8::AMAX_S, &amaxS},
                                                     {fp8::AMAX_O, &amaxO},
                                                     {fp8::SEED, &seed},
                                                     {fp8::OFFSET, &offset}};
    CpuReferenceGraphExecutor().execute(graph.GetBufferPointer(), graph.GetSize(), pack);
    Fp8Outputs out;
    const auto* oData = o.memory().hostData();
    out.o.resize(o.elementCount());
    std::memcpy(out.o.data(), oData, out.o.size() * sizeof(uint16_t));
    out.amaxS = amaxS;
    out.amaxO = amaxO;
    return out;
}

} // namespace

// cuDNN's sdpa_fp8 softmax / output scaling (SEM-024) does not combine with dropout, but
// probability 0 without an rng_dump is no dropout, so it combines like the graph without dropout.
// Every other dropout form still declines beside the scaling; each one is accepted without it.
TEST(TestSdpaFwdPlanZeroDropout, CombinesWithFp8Scaling)
{
    fp8::SdpaGraphSpec spec;
    spec.withFullScaling().dropoutProbability = 0.0F;
    EXPECT_TRUE(fp8Applicable(spec));

    // Probability 0 with an rng_dump is the Philox-with-dump form: dropout.
    spec.rngDump = true;
    EXPECT_FALSE(fp8Applicable(spec));
    // A probability above 0 with an rng_dump.
    spec.dropoutProbability = 0.25F;
    EXPECT_FALSE(fp8Applicable(spec));
    // cuDNN's explicit mask beside probability 0.
    spec = {};
    spec.withFullScaling().dropoutProbability = 0.0F;
    spec.dropout = true;
    EXPECT_FALSE(fp8Applicable(spec));

    // Without the scaling, the same dropout forms are accepted, so the declines above are the
    // scaling rule's.
    fp8::SdpaGraphSpec plain;
    plain.dropoutProbability = 0.0F;
    plain.rngDump = true;
    EXPECT_TRUE(fp8Applicable(plain));
    plain.dropoutProbability = 0.25F;
    EXPECT_TRUE(fp8Applicable(plain));
    plain.rngDump = false;
    plain.dropoutProbability.reset();
    plain.dropout = true;
    EXPECT_TRUE(fp8Applicable(plain));
}

// O and both amax outputs are bit-identical to the scaled fp8 graph without dropout.
TEST(TestSdpaFwdPlanZeroDropout, Fp8ScalingEqualsNoDropoutBitForBit)
{
    fp8::SdpaGraphSpec spec;
    spec.withFullScaling();
    const auto noDropout = runFp8(spec);
    spec.dropoutProbability = 0.0F;
    const auto zero = runFp8(spec);
    EXPECT_EQ(zero.o, noDropout.o);
    EXPECT_EQ(zero.amaxS, noDropout.amaxS);
    EXPECT_EQ(zero.amaxO, noDropout.amaxO);
    // The outputs are not trivially zero.
    EXPECT_GT(noDropout.amaxS, 0.0F);
    EXPECT_GT(noDropout.amaxO, 0.0F);
}
