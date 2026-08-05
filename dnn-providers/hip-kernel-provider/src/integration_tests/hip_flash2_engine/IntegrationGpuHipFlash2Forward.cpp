// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
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

// B1: use hipdnn_data_sdk::types::half (not __half which is undeclared here)
using half_t = hipdnn_data_sdk::types::half;

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
    std::string expected_arch; // B1: all struct fields initialized
};

class IntegrationGpuHipFlash2Forward
    : public IntegrationGraphVerificationHarness<half_t, Flash2TestConfig>
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

        const auto deviceArch = hip_kernel_provider_common::getDeviceString(this->stream());
        if(deviceArch != "gfx942")
        {
            GTEST_SKIP() << "Skipping: requires gfx942, current device is " << deviceArch;
        }

        auto graph = std::make_shared<Graph>();
        graph->set_io_data_type(DataType_t::HALF)
            .set_compute_data_type(DataType_t::FLOAT)
            .set_intermediate_data_type(DataType_t::FLOAT);

        const std::vector<int64_t> qDims{cfg.batch, cfg.num_heads_q, cfg.seq_q, cfg.head_dim};
        const std::vector<int64_t> kvDims{cfg.batch, cfg.num_heads_kv, cfg.seq_kv, cfg.head_dim};
        auto Q = graph->tensor(TensorAttributes()
                                   .set_name("Q")
                                   .set_dim(qDims)
                                   .set_stride(generateStrides(qDims))
                                   .set_data_type(DataType_t::HALF));
        auto K = graph->tensor(TensorAttributes()
                                   .set_name("K")
                                   .set_dim(kvDims)
                                   .set_stride(generateStrides(kvDims))
                                   .set_data_type(DataType_t::HALF));
        auto V = graph->tensor(TensorAttributes()
                                   .set_name("V")
                                   .set_dim(kvDims)
                                   .set_stride(generateStrides(kvDims))
                                   .set_data_type(DataType_t::HALF));

        // B1: use SdpaAttributes (not SdpaFwdAttributes which does not exist)
        SdpaAttributes sdpa_attrs;
        sdpa_attrs.set_causal_mask(cfg.causal).set_attn_scale(cfg.scale).set_generate_stats(false);

        // B1: auto [O, stats] not auto [O, /*stats=*/]
        auto [O, stats] = graph->sdpa(Q, K, V, sdpa_attrs);
        O->set_name("O").set_output(true).set_data_type(DataType_t::HALF);

        auto validationResult = graph->validate();
        ASSERT_TRUE(validationResult.is_good())
            << "Graph validation failed for " << cfg.name << ": " << validationResult.get_message();

        this->registerValidator(O, tolerance);
        this->verifyGraph(*graph, 42U);
    }
};

std::vector<Flash2TestConfig> getFlash2TestConfigs()
{
    // K3: all seq_q must be multiples of 64
    return {
        {"mha_d128_causal_b1_sq512", 1, 32, 32, 512, 512, 128, true, 1.f / 11.314f, ""},
        {"mha_d128_causal_b2_sq512", 2, 32, 32, 512, 512, 128, true, 1.f / 11.314f, ""},
        {"mha_d64_causal_b1_sq512", 1, 32, 32, 512, 512, 64, true, 1.f / 8.f, ""},
        {"mha_d128_noncausal_b1_sq512", 1, 32, 32, 512, 512, 128, false, 1.f / 11.314f, ""},
        {"gqa4_d128_causal_b1_sq512", 1, 32, 8, 512, 512, 128, true, 1.f / 11.314f, ""},
        {"gqa4_d128_causal_b2_sq512", 2, 32, 8, 512, 512, 128, true, 1.f / 11.314f, ""},
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
