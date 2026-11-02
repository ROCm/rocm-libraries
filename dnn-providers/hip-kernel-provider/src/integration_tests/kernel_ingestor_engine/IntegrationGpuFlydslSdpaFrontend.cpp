// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_frontend/Types.hpp>

#include "FlydslSdpaFrontendGraphs.hpp"

/**
 * @file IntegrationGpuFlydslSdpaFrontend.cpp
 * @brief The FlyDSL SDPA engines end to end through the frontend Graph API (the frontend-built
 *        lane, FlydslSdpaFrontendGraphs.hpp).
 *
 * The attention-sink contract (SEM-022): a model's 16-bit sink as it is stored, a bf16 sink with
 * a bf16 query (vLLM's model-dtype sink) and a bf16 sink with an fp16 query (FlashAttention 4's
 * form), with rows no key reaches so the LSE is the sink itself.
 *
 * The cross-family parity cases: each graph a FlyDSL engine serves on every GPU family, built
 * through the frontend and checked against the CPU reference on the same graph (dropout through
 * the engine's RNG dump, the reference's explicit mask), head dims off 8 among them.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

struct FrontendSinkCase
{
    const char* name;
    FrontendSdpaSpec spec;
};

std::vector<FrontendSinkCase> frontendSinkCases()
{
    std::vector<FrontendSinkCase> cases;
    // Bottom-right with Sq > Skv: the first Sq - Skv rows of each batch reach no key, so their
    // LSE is the sink and O is 0.
    FrontendSdpaSpec spec;
    spec.seqQ = 96;
    spec.seqKv = 64;
    spec.sinkDataType = hipdnn_frontend::DataType::BFLOAT16;
    spec.name = "bf16_query_bf16_sink";
    cases.push_back({"Bf16QueryBf16Sink", spec});
    spec.dataType = hipdnn_frontend::DataType::HALF;
    spec.name = "f16_query_bf16_sink";
    cases.push_back({"F16QueryBf16Sink", spec});
    return cases;
}

using IntegrationGpuFlydslSdpaFrontend = IntegrationGpuFlydslFrontendLane<FrontendSinkCase>;

using FrontendParityCase = FrontendSinkCase;

/// Dropout under padding (G1): per-batch lengths with and without padding_mask, a full batch
/// and a shorter one, bottom-right causal, GQA, the seed and offset on the device or by value.
std::vector<FrontendParityCase> frontendPaddedDropoutCases()
{
    std::vector<FrontendParityCase> cases;
    FrontendSdpaSpec spec;
    spec.seqQ = 128;
    spec.seqKv = 128;
    spec.seqLensQ = {128, 77};
    spec.seqLensKv = {128, 50};
    spec.dropout = FrontendDropout{};
    spec.name = "dropout_under_padding";
    cases.push_back({"DropoutUnderPadding", spec});
    spec.paddingMask = false;
    spec.dataType = hipdnn_frontend::DataType::HALF;
    spec.dropout->onDevice = false;
    spec.name = "dropout_under_lengths_without_padding_mask";
    cases.push_back({"DropoutUnderLengthsWithoutThePaddingMask", spec});
    spec.paddingMask = true;
    spec.alignment = hipdnn_frontend::DiagonalAlignment::TOP_LEFT;
    spec.seqLensKv = {100, 0};
    spec.name = "dropout_under_padding_top_left_keyless";
    cases.push_back({"DropoutUnderPaddingTopLeftWithAKeylessBatch", spec});
    return cases;
}

/// A left bound with no right one (G2), at both corners, with more queries than keys, padded.
std::vector<FrontendParityCase> frontendLeftOnlyWindowCases()
{
    std::vector<FrontendParityCase> cases;
    FrontendSdpaSpec spec;
    spec.seqQ = 160;
    spec.seqKv = 96;
    spec.leftBound = 40;
    spec.rightBound.reset();
    spec.alignment = hipdnn_frontend::DiagonalAlignment::TOP_LEFT;
    spec.name = "left_only_window_top_left";
    cases.push_back({"LeftOnlyWindowTopLeft", spec});
    spec.alignment = hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT;
    spec.name = "left_only_window_bottom_right";
    cases.push_back({"LeftOnlyWindowBottomRight", spec});
    spec.seqLensQ = {160, 61};
    spec.seqLensKv = {96, 70};
    spec.dataType = hipdnn_frontend::DataType::HALF;
    spec.name = "left_only_window_bottom_right_padded";
    cases.push_back({"LeftOnlyWindowBottomRightPadded", spec});
    return cases;
}

/// V with a head count of its own (G3): fewer and more V heads than K heads, causal, and
/// padded; head_dim 96 so CDNA's bucket objects serve it.
std::vector<FrontendParityCase> frontendValueHeadsCases()
{
    std::vector<FrontendParityCase> cases;
    FrontendSdpaSpec spec;
    spec.queryHeads = 8;
    spec.kvHeads = 2;
    spec.valueHeads = 4;
    spec.headSize = 96;
    spec.seqQ = 100;
    spec.seqKv = 140;
    spec.name = "value_heads_more_than_key_heads";
    cases.push_back({"ValueHeadsMoreThanKeyHeads", spec});
    spec.kvHeads = 4;
    spec.valueHeads = 2;
    spec.dataType = hipdnn_frontend::DataType::HALF;
    spec.name = "value_heads_fewer_than_key_heads";
    cases.push_back({"ValueHeadsFewerThanKeyHeads", spec});
    spec.seqLensQ = {100, 37};
    spec.seqLensKv = {140, 66};
    spec.name = "value_heads_of_their_own_padded";
    cases.push_back({"ValueHeadsOfTheirOwnPadded", spec});
    return cases;
}

/// The parity extras (E1-E6): dropout at head_dim 96 and with a bias broadcast along the keys;
/// a head past 256, a Q/K-V pair and un-absorbed MLA with a bias; MLA with a negative scale.
std::vector<FrontendParityCase> frontendParityExtrasCases()
{
    using hipdnn_frontend::DataType;
    std::vector<FrontendParityCase> cases;
    FrontendSdpaSpec base;
    base.seqQ = 64;
    base.seqKv = 96;
    auto d96 = base;
    d96.headSize = 96;
    d96.dropout = FrontendDropout{};
    d96.name = "dropout_at_head_dim_96";
    cases.push_back({"DropoutAtHeadDim96", d96});
    auto keyBroadcast = base;
    keyBroadcast.dropout = FrontendDropout{};
    keyBroadcast.biasDataType = DataType::BFLOAT16;
    keyBroadcast.biasDims = {base.batch, base.queryHeads, base.seqQ, 1};
    keyBroadcast.name = "dropout_with_a_key_broadcast_bias";
    cases.push_back({"DropoutWithAKeyBroadcastBias", keyBroadcast});
    // Four query rows of a four-head group: one 16-row tile, which gfx11's wide decode bias
    // objects take.
    auto wide = base;
    wide.seqQ = 4;
    wide.headSize = 384;
    wide.biasDataType = DataType::BFLOAT16;
    wide.name = "head_dim_384_with_a_bias";
    cases.push_back({"HeadDim384WithABias", wide});
    auto pair = base;
    pair.headSize = 128;
    pair.valueHeadSize = 64;
    pair.biasDataType = DataType::BFLOAT16;
    pair.name = "query_key_128_value_64_with_a_bias";
    cases.push_back({"QueryKey128Value64WithABias", pair});
    auto mla = base;
    mla.headSize = 192;
    mla.valueHeadSize = 128;
    mla.biasDataType = DataType::BFLOAT16;
    mla.name = "query_key_192_value_128_with_a_bias";
    cases.push_back({"QueryKey192Value128WithABias", mla});
    mla.biasDataType.reset();
    mla.scale = -0.1F;
    mla.name = "query_key_192_value_128_with_a_negative_scale";
    cases.push_back({"QueryKey192Value128WithANegativeScale", mla});
    return cases;
}

/// head_dim 256 on gfx950's dual-wave kernel (vendored-kernel modification 43: 4 waves of 32
/// rows), pinned there: several KV tiles with ragged tails, GQA, no mask and a bottom-right
/// diagonal off zero with the LSE. Other GPUs serve them on their first-ranked FlyDSL engine.
std::vector<FrontendParityCase> frontendDualwaveHeadDim256Cases()
{
    std::vector<FrontendParityCase> cases;
    FrontendSdpaSpec spec;
    spec.headSize = 256;
    spec.seqQ = 640;
    spec.seqKv = 700;
    spec.rightBound.reset();
    spec.engine = "hipkernel:flydsl_sdpa_cdna_dualwave";
    spec.engineArch = "gfx950";
    spec.name = "dualwave_head_dim_256";
    cases.push_back({"NoMask", spec});
    spec.rightBound = 0;
    spec.seqQ = 300;
    spec.dataType = hipdnn_frontend::DataType::HALF;
    spec.name = "dualwave_head_dim_256_bottom_right";
    cases.push_back({"BottomRightCausal", spec});
    return cases;
}

/// Head dims off 8 (FlydslSdpaHeadPad.hpp): AOTriton's prime head dims at their own width, so
/// every row pitch is odd, on each family through padded copies -- plain, with a bias in the
/// query's dtype, dropping, padded, packed and in f32.
std::vector<FrontendParityCase> frontendOddHeadDimCases()
{
    using hipdnn_frontend::DataType;
    std::vector<FrontendParityCase> cases;
    FrontendSdpaSpec base;
    base.seqQ = 67;
    base.seqKv = 143;
    auto d7 = base;
    d7.headSize = 7;
    d7.name = "head_dim_7";
    cases.push_back({"HeadDim7", d7});
    auto d53 = base;
    d53.headSize = 53;
    d53.biasDataType = DataType::BFLOAT16;
    d53.name = "head_dim_53_with_a_bias";
    cases.push_back({"HeadDim53WithABias", d53});
    auto d113 = base;
    d113.headSize = 113;
    d113.dataType = DataType::HALF;
    d113.dropout = FrontendDropout{};
    d113.name = "head_dim_113_dropping";
    cases.push_back({"HeadDim113Dropping", d113});
    auto d241 = base;
    d241.headSize = 241;
    d241.seqLensQ = {67, 19};
    d241.seqLensKv = {143, 40};
    d241.name = "head_dim_241_padded";
    cases.push_back({"HeadDim241Padded", d241});
    auto d23 = base;
    d23.headSize = 23;
    d23.packedLensQ = {67, 19};
    d23.packedLensKv = {143, 40};
    d23.name = "head_dim_23_packed";
    cases.push_back({"HeadDim23Packed", d23});
    auto f32 = base;
    f32.headSize = 37;
    f32.dataType = DataType::FLOAT;
    f32.name = "head_dim_37_f32";
    cases.push_back({"HeadDim37F32", f32});
    auto pair = base;
    pair.headSize = 64;
    pair.valueHeadSize = 37;
    pair.name = "query_key_64_value_37";
    cases.push_back({"QueryKey64Value37", pair});
    return cases;
}

using IntegrationGpuFlydslSdpaFrontendParity = IntegrationGpuFlydslFrontendLane<FrontendParityCase>;

} // namespace

/// A 16-bit sink as the model stores it, through Graph::validate, the backend and a FlyDSL
/// engine, against hipDNN's CPU reference on the same graph (O and the LSE).
TEST_P(IntegrationGpuFlydslSdpaFrontend, ServesAModelDtypeSinkAsTheCpuReferenceDoes)
{
    const auto& testCase = GetParam();
    auto sdpa = buildFrontendSdpaGraph(testCase.spec);
    validateServeAndVerify(sdpa, testCase.spec.dataType);
}

/// A graph every FlyDSL family serves, through Graph::validate, the backend and a FlyDSL engine,
/// against hipDNN's CPU reference on the same graph (O, the LSE, and the RNG dump's 0/1 values
/// and keep rate when the graph drops).
TEST_P(IntegrationGpuFlydslSdpaFrontendParity, ServesTheGraphAsTheCpuReferenceDoes)
{
    const auto& testCase = GetParam();
    auto sdpa = buildFrontendSdpaGraph(testCase.spec);
    validateServeAndVerify(sdpa, testCase.spec.dataType);
}

INSTANTIATE_TEST_SUITE_P(PaddedDropout,
                         IntegrationGpuFlydslSdpaFrontendParity,
                         ::testing::ValuesIn(frontendPaddedDropoutCases()),
                         [](const ::testing::TestParamInfo<FrontendParityCase>& info) {
                             return std::string(info.param.name);
                         });

INSTANTIATE_TEST_SUITE_P(LeftOnlyWindow,
                         IntegrationGpuFlydslSdpaFrontendParity,
                         ::testing::ValuesIn(frontendLeftOnlyWindowCases()),
                         [](const ::testing::TestParamInfo<FrontendParityCase>& info) {
                             return std::string(info.param.name);
                         });

INSTANTIATE_TEST_SUITE_P(ValueHeads,
                         IntegrationGpuFlydslSdpaFrontendParity,
                         ::testing::ValuesIn(frontendValueHeadsCases()),
                         [](const ::testing::TestParamInfo<FrontendParityCase>& info) {
                             return std::string(info.param.name);
                         });

INSTANTIATE_TEST_SUITE_P(ParityExtras,
                         IntegrationGpuFlydslSdpaFrontendParity,
                         ::testing::ValuesIn(frontendParityExtrasCases()),
                         [](const ::testing::TestParamInfo<FrontendParityCase>& info) {
                             return std::string(info.param.name);
                         });

INSTANTIATE_TEST_SUITE_P(DualwaveHeadDim256,
                         IntegrationGpuFlydslSdpaFrontendParity,
                         ::testing::ValuesIn(frontendDualwaveHeadDim256Cases()),
                         [](const ::testing::TestParamInfo<FrontendParityCase>& info) {
                             return std::string(info.param.name);
                         });

INSTANTIATE_TEST_SUITE_P(OddHeadDims,
                         IntegrationGpuFlydslSdpaFrontendParity,
                         ::testing::ValuesIn(frontendOddHeadDimCases()),
                         [](const ::testing::TestParamInfo<FrontendParityCase>& info) {
                             return std::string(info.param.name);
                         });

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuFlydslSdpaFrontend,
                         ::testing::ValuesIn(frontendSinkCases()),
                         [](const ::testing::TestParamInfo<FrontendSinkCase>& info) {
                             return std::string(info.param.name);
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENABLE_SDPA
