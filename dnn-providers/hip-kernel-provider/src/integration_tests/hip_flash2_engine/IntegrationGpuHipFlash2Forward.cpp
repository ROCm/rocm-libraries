// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Integration tests for HipFlash2Engine — FP16 Flash-Attention 2 SDPA.
//
// These tests verify correctness of the HipFlash2FwdPlan kernel against
// a CPU FP32 reference across the shapes validated on MI300X, MI325X,
// and MI355X hardware.
//
// Run with: hip_kernel_provider_integration_tests --gtest_filter="*HipFlash2*"

#include "IntegrationGraphVerificationHarness.hpp"

#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>

#include <cmath>
#include <vector>

namespace hip_flash2_engine
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

struct Flash2TestConfig
{
    int64_t batch;
    int64_t numQHeads;
    int64_t numKVHeads; // <numQHeads for GQA
    int64_t seqQ;
    int64_t seqKV;
    int64_t headDim;
    bool causal;
    std::string description;
};

static std::vector<Flash2TestConfig> getTestConfigs()
{
    return {
        // MHA causal — shapes validated on MI300X/MI325X/MI355X
        {1, 32, 32, 512, 512, 128, true, "MHA_seq512_D128_causal"},
        {1, 32, 32, 1024, 1024, 128, true, "MHA_seq1024_D128_causal"},
        {1, 32, 32, 2048, 2048, 128, true, "MHA_seq2048_D128_causal"},
        {1, 32, 32, 4096, 4096, 128, true, "MHA_seq4096_D128_causal"},

        // MHA non-causal
        {1, 32, 32, 2048, 2048, 128, false, "MHA_seq2048_D128_noncausal"},
        {1, 32, 32, 4096, 4096, 128, false, "MHA_seq4096_D128_noncausal"},

        // D=64
        {1, 32, 32, 2048, 2048, 64, true, "MHA_seq2048_D64_causal"},
        {1, 32, 32, 4096, 4096, 64, true, "MHA_seq4096_D64_causal"},

        // GQA (grouped query attention) — 4 query heads per KV head
        {1, 32, 8, 2048, 2048, 128, true, "GQA4_seq2048_D128_causal"},
        {1, 32, 8, 4096, 4096, 128, true, "GQA4_seq4096_D128_causal"},

        // Batch > 1
        {4, 32, 32, 1024, 1024, 128, true, "MHA_batch4_seq1024_D128_causal"},
    };
}

static std::shared_ptr<Graph> buildSdpaGraph(const Flash2TestConfig& cfg)
{
    const std::vector<int64_t> qDims = {cfg.batch, cfg.seqQ, cfg.numQHeads, cfg.headDim};
    const std::vector<int64_t> kvDims = {cfg.batch, cfg.seqKV, cfg.numKVHeads, cfg.headDim};
    const std::vector<int64_t> oDims = {cfg.batch, cfg.seqQ, cfg.numQHeads, cfg.headDim};

    auto graph = std::make_shared<Graph>();
    graph->set_io_data_type(DataType::HALF)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_dim(qDims)
        .set_stride({qDims[1] * qDims[2] * qDims[3], qDims[2] * qDims[3], qDims[3], 1})
        .set_data_type(DataType::HALF)
        .set_uid(1);

    auto k = std::make_shared<TensorAttributes>();
    k->set_dim(kvDims)
        .set_stride({kvDims[1] * kvDims[2] * kvDims[3], kvDims[2] * kvDims[3], kvDims[3], 1})
        .set_data_type(DataType::HALF)
        .set_uid(2);

    auto v = std::make_shared<TensorAttributes>();
    v->set_dim(kvDims)
        .set_stride({kvDims[1] * kvDims[2] * kvDims[3], kvDims[2] * kvDims[3], kvDims[3], 1})
        .set_data_type(DataType::HALF)
        .set_uid(3);

    auto o = std::make_shared<TensorAttributes>();
    o->set_dim(oDims)
        .set_stride({oDims[1] * oDims[2] * oDims[3], oDims[2] * oDims[3], oDims[3], 1})
        .set_data_type(DataType::HALF)
        .set_uid(4)
        .set_output(true);

    const float smScale = 1.0f / std::sqrt(static_cast<float>(cfg.headDim));

    auto sdpaAttrs = graph->sdpa(q, k, v, {smScale, cfg.causal, false, {}}, o);
    (void)sdpaAttrs;

    return graph;
}

class HipFlash2ForwardTest : public ::testing::TestWithParam<Flash2TestConfig>
{
};

TEST_P(HipFlash2ForwardTest, VerifyCorrectness)
{
    const auto& cfg = GetParam();

    auto graph = buildSdpaGraph(cfg);
    ASSERT_NE(graph, nullptr);

    // Build and verify via the integration harness
    // The harness checks that HipFlash2Engine is selected and produces
    // output matching the CPU FP32 reference within tolerance
    IntegrationGraphVerificationHarness harness;
    harness.setMaxAbsoluteError(5e-3f); // FP16 tolerance
    harness.setMaxRelativeError(1e-2f);

    EXPECT_TRUE(harness.buildAndVerify(*graph))
        << "HipFlash2Engine correctness check failed for: " << cfg.description;
}

INSTANTIATE_TEST_SUITE_P(HipFlash2,
                         HipFlash2ForwardTest,
                         ::testing::ValuesIn(getTestConfigs()),
                         [](const ::testing::TestParamInfo<Flash2TestConfig>& info) {
                             return info.param.description;
                         });

} // namespace hip_flash2_engine
