// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_test_sdk/utilities/SdpaVllmFormTestGraphs.hpp>

#include "FlydslSdpaFrontendGraphs.hpp"

/**
 * @file IntegrationGpuFlydslSdpaFrontendFeatures.cpp
 * @brief One frontend-built case per feature the FlyDSL SDPA engines serve (the frontend-built
 *        lane, FlydslSdpaFrontendGraphs.hpp): each graph goes through Graph::validate, the
 *        backend and the first-ranked FlyDSL engine, and is compared with hipDNN's CPU reference
 *        graph executor on the same graph.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

struct FrontendFeatureCase
{
    const char* name;
    FrontendSdpaSpec spec;
    /// The architecture prefix of the only family that serves the graph (e.g. "gfx9" for a
    /// CDNA-only feature); every FlyDSL family when null.
    const char* onlyOn = nullptr;
};

using IntegrationGpuFlydslSdpaFrontendFeatures
    = IntegrationGpuFlydslFrontendLane<FrontendFeatureCase>;

std::string caseName(const ::testing::TestParamInfo<FrontendFeatureCase>& info)
{
    return info.param.name;
}

/// Dropout probability 0 with no RNG dump: every family runs its plain objects (the undropped
/// result), and the references treat it as no dropout (SEM-015). The seed and offset are on the
/// device or passed by value.
std::vector<FrontendFeatureCase> zeroProbabilityDropoutCases()
{
    std::vector<FrontendFeatureCase> cases;
    FrontendSdpaSpec spec;
    spec.seqQ = 80;
    spec.seqKv = 112;
    spec.dropout = FrontendDropout{};
    spec.dropout->probability = 0.0F;
    spec.dropout->withRngDump = false;
    spec.name = "dropout_probability_zero";
    cases.push_back({"DropoutProbabilityZero", spec});
    spec.dropout->onDevice = false;
    spec.dataType = hipdnn_frontend::DataType::HALF;
    spec.seqLensQ = {80, 33};
    spec.seqLensKv = {112, 47};
    spec.name = "dropout_probability_zero_by_value_padded";
    cases.push_back({"DropoutProbabilityZeroByValuePadded", spec});
    return cases;
}

/// vLLM's windows through their mapping to hipDNN's bounds (SEM-027): a decoder sliding window
/// W stored as (W - 1, 0) is left_bound W, right_bound 0; an encoder's (W - 1, W - 1) is left W,
/// right W - 1; FA's non-causal (w, w) is left w + 1, right w; all bottom-right. Padded and
/// packed (vLLM's query_start_loc), the way vLLM batches requests.
std::vector<FrontendFeatureCase> faWindowCases()
{
    std::vector<FrontendFeatureCase> cases;
    FrontendSdpaSpec decoder;
    decoder.seqQ = 96;
    decoder.seqKv = 160;
    decoder.leftBound = 64;
    decoder.rightBound = 0;
    decoder.seqLensQ = {96, 40};
    decoder.seqLensKv = {160, 75};
    decoder.name = "decoder_sliding_window_padded";
    cases.push_back({"DecoderSlidingWindowPadded", decoder});
    auto packedDecoder = decoder;
    packedDecoder.seqLensQ.clear();
    packedDecoder.seqLensKv.clear();
    packedDecoder.packedLensQ = {96, 40};
    packedDecoder.packedLensKv = {160, 75};
    packedDecoder.dataType = hipdnn_frontend::DataType::HALF;
    packedDecoder.name = "decoder_sliding_window_packed";
    cases.push_back({"DecoderSlidingWindowPacked", packedDecoder});

    FrontendSdpaSpec encoder;
    encoder.seqQ = 128;
    encoder.seqKv = 128;
    encoder.leftBound = 32;
    encoder.rightBound = 31;
    encoder.packedLensQ = {128, 70};
    encoder.packedLensKv = {128, 70};
    encoder.name = "encoder_window_packed";
    cases.push_back({"EncoderWindowPacked", encoder});

    FrontendSdpaSpec nonCausal;
    nonCausal.seqQ = 64;
    nonCausal.seqKv = 96;
    nonCausal.leftBound = 17;
    nonCausal.rightBound = 16;
    nonCausal.dataType = hipdnn_frontend::DataType::HALF;
    nonCausal.name = "non_causal_fa_window";
    cases.push_back({"NonCausalFaWindow", nonCausal});
    return cases;
}

/// Keys no live row sees hold NaN or infinity, in the engine's and the reference's inputs: past
/// each batch's key length (padded, and in each packed slot past its live length), and in K
/// before a decoder window; finite garbage in K and V before the window. None may reach a result.
/// NaN or infinity in V inside the key range but outside a row's window is not here: a kernel
/// multiplies a tile's zero probabilities by its V, so such a V element reaches the rows of its
/// tile (COVERAGE §5.7, "NaN propagation is the hardware's"); the references never read it.
std::vector<FrontendFeatureCase> unseenKeyCases()
{
    std::vector<FrontendFeatureCase> cases;
    FrontendSdpaSpec padded;
    padded.seqQ = 64;
    padded.seqKv = 160;
    padded.seqLensQ = {64, 21};
    padded.seqLensKv = {100, 37};
    padded.unseenKeyValue = std::numeric_limits<float>::quiet_NaN();
    padded.name = "nan_past_the_key_length_padded";
    cases.push_back({"NanPastTheKeyLengthPadded", padded});
    FrontendSdpaSpec slots;
    slots.seqQ = 64;
    slots.seqKv = 128;
    slots.packedLensQ = {64, 30};
    slots.packedLensKv = {128, 90};
    slots.seqLensQ = {50, 30};
    slots.seqLensKv = {101, 33};
    slots.unseenKeyValue = std::numeric_limits<float>::infinity();
    slots.dataType = hipdnn_frontend::DataType::HALF;
    slots.name = "infinity_past_the_key_length_in_packed_slots";
    cases.push_back({"InfinityPastTheKeyLengthInPackedSlots", slots});
    FrontendSdpaSpec window;
    window.seqQ = 16;
    window.seqKv = 256;
    window.leftBound = 32;
    window.rightBound = 0;
    window.unseenKeyValue = std::numeric_limits<float>::quiet_NaN();
    window.unseenInV = false;
    window.name = "nan_in_k_before_a_decoder_window";
    cases.push_back({"NanInKBeforeADecoderWindow", window});
    auto garbage = window;
    garbage.unseenKeyValue = 30000.0F;
    garbage.unseenInV = true;
    garbage.name = "garbage_before_a_decoder_window";
    cases.push_back({"GarbageBeforeADecoderWindow", garbage});
    return cases;
}

/// One case per served feature the other suites do not build through the frontend.
std::vector<FrontendFeatureCase> backfillCases()
{
    using hipdnn_frontend::DataType;
    std::vector<FrontendFeatureCase> cases;
    FrontendSdpaSpec base;
    base.seqQ = 64;
    base.seqKv = 96;

    auto floatSink = base;
    floatSink.sinkDataType = DataType::FLOAT;
    floatSink.name = "float_sink";
    cases.push_back({"FloatSink", floatSink});

    auto packedSink = base;
    packedSink.packedLensQ = {64, 23};
    packedSink.packedLensKv = {96, 40};
    packedSink.sinkDataType = DataType::FLOAT;
    packedSink.name = "packed_with_a_sink";
    cases.push_back({"PackedWithASink", packedSink});

    auto p3 = base;
    p3.packedLensQ = {64, 23};
    p3.packedLensKv = {96, 40};
    p3.seqLensQ = {50, 23};
    p3.seqLensKv = {81, 12};
    p3.dataType = DataType::HALF;
    p3.name = "packed_with_lengths";
    cases.push_back({"PackedWithLengths", p3});

    // PyTorch's varlen seqused_k without seq_len_q (SEM-028): the query rows are the packed
    // sequences' own, the keys each slot's first seq_len_kv.
    auto sequsedK = p3;
    sequsedK.seqLensQ.clear();
    sequsedK.dataType = DataType::BFLOAT16;
    sequsedK.name = "packed_with_seq_len_kv_alone";
    cases.push_back({"PackedWithSeqLenKvAlone", sequsedK});

    auto zeroScale = base;
    zeroScale.scale = 0.0F;
    zeroScale.name = "zero_scale";
    cases.push_back({"ZeroScale", zeroScale});

    auto f32Bias = base;
    f32Bias.biasDataType = DataType::FLOAT;
    f32Bias.name = "f32_bias_with_a_16_bit_query";
    cases.push_back({"F32BiasWithA16BitQuery", f32Bias});

    auto biasAndSink = base;
    biasAndSink.biasDataType = DataType::BFLOAT16;
    biasAndSink.sinkDataType = DataType::FLOAT;
    biasAndSink.name = "bias_and_a_sink";
    cases.push_back({"BiasAndASink", biasAndSink});

    auto topLeft = base;
    topLeft.seqQ = 96;
    topLeft.seqKv = 64;
    topLeft.alignment = hipdnn_frontend::DiagonalAlignment::TOP_LEFT;
    topLeft.name = "top_left_causal_more_queries_than_keys";
    cases.push_back({"TopLeftCausalMoreQueriesThanKeys", topLeft});

    auto mla = base;
    mla.headSize = 192;
    mla.valueHeadSize = 128;
    mla.packedLensQ = {64, 29};
    mla.packedLensKv = {96, 51};
    mla.name = "mla_192_128_packed_with_the_lse";
    cases.push_back({"Mla192By128PackedWithTheLse", mla});

    // Exact f32: the CDNA generic objects (vendored-kernel modification 35) and the RDNA
    // prefill objects (prefill kernel modification 34).
    auto f32 = base;
    f32.dataType = DataType::FLOAT;
    f32.name = "f32_io";
    cases.push_back({"F32Io", f32});
    auto f32Masked = f32;
    f32Masked.biasDataType = DataType::FLOAT;
    f32Masked.name = "f32_with_a_bias";
    cases.push_back({"F32WithABias", f32Masked});
    auto f32Padded = f32;
    f32Padded.seqLensQ = {50, 23};
    f32Padded.seqLensKv = {81, 12};
    f32Padded.name = "f32_with_lengths";
    cases.push_back({"F32WithLengths", f32Padded});
    return cases;
}

using IntegrationGpuFlydslSdpaFrontendDeclines
    = IntegrationGpuFlydslFrontendLane<FrontendFeatureCase>;

} // namespace

/// vLLM's paged call (ragged Q through query_start_loc, paged K/V, seq_len_kv only; SEM-026)
/// passes the frontend, and every FlyDSL engine declines it: paging is not served yet. The same
/// problem dense and padded is served, so the decline is the paging's.
TEST_F(IntegrationGpuFlydslSdpaFrontendDeclines, DeclinesPagedKvUnderARaggedQuery)
{
    using namespace hipdnn_test_sdk::utilities::sdpa_vllm_test;
    VllmProblem problem;
    problem.dataType = hipdnn_frontend::DataType::BFLOAT16;
    problem.qHeads = 8;
    problem.kvHeads = 2;
    problem.headDim = 64;
    problem.window.right = 0;
    problem.withStats = false;

    // The first FlyDSL SDPA engine the backend ranks for the problem's graph, built through the
    // frontend and validated; 0 if none.
    std::string ranking;
    const auto firstFlydslEngineFor = [this](const VllmProblem& p, std::string& ranked) {
        auto graph = makeVllmGraph(p);
        const auto result = graph->build_operation_graph(this->_handle);
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        return firstRankedFlydslEngine(*graph, ranked);
    };

    auto dense = problem;
    dense.raggedQuery = false;
    dense.paged = false;
    EXPECT_NE(firstFlydslEngineFor(dense, ranking), 0)
        << "the dense padded control is served (" << ranking << ")";

    EXPECT_EQ(firstFlydslEngineFor(problem, ranking), 0) << ranking;
    auto withSeqLenQ = problem;
    withSeqLenQ.explicitSeqLenQ = true;
    EXPECT_EQ(firstFlydslEngineFor(withSeqLenQ, ranking), 0) << ranking;
    auto paddedQuery = problem;
    paddedQuery.raggedQuery = false;
    EXPECT_EQ(firstFlydslEngineFor(paddedQuery, ranking), 0) << ranking;
}

TEST_P(IntegrationGpuFlydslSdpaFrontendFeatures, ServesTheFeatureAsTheCpuReferenceDoes)
{
    const auto& testCase = GetParam();
    if(testCase.onlyOn != nullptr && _arch.rfind(testCase.onlyOn, 0) != 0)
    {
        GTEST_SKIP() << "only " << testCase.onlyOn << " serves " << testCase.name;
    }
    auto sdpa = buildFrontendSdpaGraph(testCase.spec);
    validateServeAndVerify(sdpa, testCase.spec.dataType);
}

INSTANTIATE_TEST_SUITE_P(FaWindows,
                         IntegrationGpuFlydslSdpaFrontendFeatures,
                         ::testing::ValuesIn(faWindowCases()),
                         caseName);

INSTANTIATE_TEST_SUITE_P(UnseenKeys,
                         IntegrationGpuFlydslSdpaFrontendFeatures,
                         ::testing::ValuesIn(unseenKeyCases()),
                         caseName);

INSTANTIATE_TEST_SUITE_P(Backfill,
                         IntegrationGpuFlydslSdpaFrontendFeatures,
                         ::testing::ValuesIn(backfillCases()),
                         caseName);

INSTANTIATE_TEST_SUITE_P(ZeroProbabilityDropout,
                         IntegrationGpuFlydslSdpaFrontendFeatures,
                         ::testing::ValuesIn(zeroProbabilityDropoutCases()),
                         caseName);

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENABLE_SDPA
