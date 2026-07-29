// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hip_kernel_provider::test_utilities;

namespace
{

struct Flash2TestConfig
{
    std::string name;
    int batch;
    int num_heads_q;
    int num_heads_kv;
    int seq_q;
    int seq_kv;
    int head_dim;
    bool causal;
    float scale;
    std::string expected_arch; // empty = run on any gfx942
};

/**
 * @brief GTest fixture for HipFlash2Engine FP16 SDPA integration tests.
 *
 * Derives from IntegrationGraphVerificationHarness which handles plugin loading,
 * hipDNN handle creation, stream setup, and CPU reference comparison.
 */
class IntegrationGpuHipFlash2Forward
    : public IntegrationGraphVerificationHarness<__half, Flash2TestConfig>
{
protected:
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first, -1.0f, 1.0f, seed);
        }
    }

    void runFlash2Test(float tolerance)
    {
        const Flash2TestConfig& cfg = this->GetParam();

        // Skip on non-gfx942 devices
        const auto deviceArch = hip_kernel_provider_common::getDeviceString(this->stream());
        if(deviceArch != "gfx942")
        {
            GTEST_SKIP() << "Skipping: requires gfx942, current device is " << deviceArch;
        }

        // Build the graph using the frontend API
        auto graph = std::make_shared<Graph>();
        graph->set_io_data_type(DataType_t::HALF)
            .set_compute_data_type(DataType_t::FLOAT)
            .set_intermediate_data_type(DataType_t::FLOAT);

        auto Q = graph->tensor(TensorAttributes()
                                   .set_name("Q")
                                   .set_dim({cfg.batch, cfg.num_heads_q, cfg.seq_q, cfg.head_dim})
                                   .set_data_type(DataType_t::HALF));
        auto K = graph->tensor(TensorAttributes()
                                   .set_name("K")
                                   .set_dim({cfg.batch, cfg.num_heads_kv, cfg.seq_kv, cfg.head_dim})
                                   .set_data_type(DataType_t::HALF));
        auto V = graph->tensor(TensorAttributes()
                                   .set_name("V")
                                   .set_dim({cfg.batch, cfg.num_heads_kv, cfg.seq_kv, cfg.head_dim})
                                   .set_data_type(DataType_t::HALF));

        auto sdpa_opts = SdpaFwdAttributes()
                             .set_causal_mask(cfg.causal)
                             .set_attn_scale(cfg.scale)
                             .set_generate_stats(false);

        auto [O, /*stats=*/] = graph->sdpa(Q, K, V, sdpa_opts);
        O->set_name("O").set_output(true).set_data_type(DataType_t::HALF);

        auto validationResult = graph->validate();
        ASSERT_TRUE(validationResult.is_good())
            << "Graph validation failed for " << cfg.name << ": " << validationResult.get_message();

        // Register O for comparison against CPU reference
        this->registerValidator(O, tolerance);
        this->verifyGraph(*graph, 42U);
    }
};

std::vector<Flash2TestConfig> getFlash2TestConfigs()
{
    return {
        // MHA causal
        {"mha_d128_causal_b1_sq1024", 1, 32, 32, 1024, 1024, 128, true, 1.f / 11.314f},
        {"mha_d128_causal_b2_sq512", 2, 32, 32, 512, 512, 128, true, 1.f / 11.314f},
        {"mha_d64_causal_b1_sq2048", 1, 32, 32, 2048, 2048, 64, true, 1.f / 8.f},
        // MHA non-causal
        {"mha_d128_noncausal_b1_sq512", 1, 32, 32, 512, 512, 128, false, 1.f / 11.314f},
        // GQA causal (4:1 ratio)
        {"gqa4_d128_causal_b1_sq1024", 1, 32, 8, 1024, 1024, 128, true, 1.f / 11.314f},
        {"gqa4_d128_causal_b2_sq512", 2, 32, 8, 512, 512, 128, true, 1.f / 11.314f},
        // Cross-attention (seq_q != seq_kv)
        {"mha_d128_cross_b1", 1, 32, 32, 512, 1024, 128, false, 1.f / 11.314f},
    };
}

} // namespace

TEST_P(IntegrationGpuHipFlash2Forward, Correctness)
{
    runFlash2Test(1e-2f);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuHipFlash2Forward,
                         testing::ValuesIn(getFlash2TestConfigs()),
                         [](const testing::TestParamInfo<Flash2TestConfig>& info) {
                             return info.param.name;
                         });
