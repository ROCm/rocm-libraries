// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hip_kernel_provider_common/SdpaConfigEnumerations.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "AsmSdpaConfigHelpers.hpp"
#include "asm_fmha_v3_fwd_configs.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hip_kernel_provider::test_utilities;
using namespace asm_sdpa_engine;

namespace
{

/**
 * @brief Test fixture that takes a GraphTestCase as parameter.
 */
template <typename DataType>
class IntegrationSdpaFwd : public IntegrationGraphVerificationHarness<DataType, GraphTestCase>
{
protected:
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first, _minVal, _maxVal, seed);
        }
    }

    void runGraphTest(float tolerance)
    {
        auto deviceString = hip_kernel_provider_common::getDeviceString(this->stream());
        const GraphTestCase& testCase = this->GetParam();

        // Skip if device is not supported
        if(testCase.arch != deviceString)
        {
            GTEST_SKIP() << "Skipped: Test case requires " << testCase.arch
                         << " but current device architecture is " << deviceString;
        }

        auto graph = buildSdpaFwdGraph(testCase);

        auto validationResult = graph->validate();
        ASSERT_TRUE(validationResult.is_good())
            << "Graph validation failed for config: " << testCase.name << " - "
            << validationResult.get_message();

        // Register output tensor validator
        graph->visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(!tensorAttr->get_is_virtual())
                {
                    this->registerValidator(tensorAttr, tolerance);
                }
            }
        });

        this->verifyGraph(*graph, 0);
    }

    float _minVal = -1.0;
    float _maxVal = 1.0;
};

using IntegrationGpuSdpaFwdBf16 = IntegrationSdpaFwd<bfloat16>;

/**
 * @brief Test fixture for shape-sweep tests parameterized on SdpaFwdTestCase.
 *
 * Builds a forward SDPA graph from explicit tensor dimensions, enabling
 * shape sweeps, GQA, and asymmetric sequence-length testing.
 */
template <typename DataType>
class IntegrationSdpaFwdShapeSweep
    : public IntegrationGraphVerificationHarness<DataType, SdpaFwdTestCase>
{
protected:
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first, _minVal, _maxVal, seed);
        }
    }

    void runGraphTest(float tolerance)
    {
        // MI300/MI308 coverage: Both MI300 and MI308 report "gfx942" via getDeviceString()
        // but load different .co files (MI300/ vs MI308/ subdirectory in getKernelCoPath()).
        // Integration tests exercise whichever device the CI agent has. Full coverage
        // requires CI agents with both MI300 and MI308 hardware.
        auto deviceString = hip_kernel_provider_common::getDeviceString(this->stream());
        const SdpaFwdTestCase& testCase = this->GetParam();

        if(testCase.arch != deviceString)
        {
            GTEST_SKIP() << "Skipped: Test case requires " << testCase.arch
                         << " but current device architecture is " << deviceString;
        }

        const auto dataType = getDataTypeEnumFromType<DataType>();

        Graph graph;
        graph.set_io_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

        auto q = std::make_shared<TensorAttributes>();
        q->set_dim(testCase.qDims)
            .set_stride(generateStrides(testCase.qDims))
            .set_data_type(dataType);

        auto k = std::make_shared<TensorAttributes>();
        k->set_dim(testCase.kDims)
            .set_stride(generateStrides(testCase.kDims))
            .set_data_type(dataType);

        auto v = std::make_shared<TensorAttributes>();
        v->set_dim(testCase.vDims)
            .set_stride(generateStrides(testCase.vDims))
            .set_data_type(dataType);

        SdpaAttributes attributes;
        attributes.set_name("SdpaFwdShapeSweepTest");

        // Configure mask: rightBound >= 0 indicates a causal/sliding-window mask.
        // leftBound is always forwarded (the -1 sentinel means "unbounded left").
        if(testCase.rightBound >= 0)
        {
            attributes.set_diagonal_band_left_bound(testCase.leftBound);
            attributes.set_diagonal_band_right_bound(testCase.rightBound);
            if(testCase.topLeftAlignment)
            {
                attributes.set_diagonal_alignment(DiagonalAlignment::TOP_LEFT);
                attributes.set_causal_mask(true);
            }
            else
            {
                attributes.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
                attributes.set_causal_mask_bottom_right(true);
            }
        }

        auto [o, stats] = graph.sdpa(q, k, v, attributes);

        o->set_output(true).set_data_type(dataType);

        auto validationResult = graph.validate();
        ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();

        graph.visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(!tensorAttr->get_is_virtual())
                {
                    this->registerValidator(tensorAttr, tolerance);
                }
            }
        });

        this->verifyGraph(graph, 0);
    }

    float _minVal = -1.0;
    float _maxVal = 1.0;
};

std::vector<SdpaFwdTestCase> getSdpaFwdShapeSweepTestCases()
{
    std::vector<SdpaFwdTestCase> cases;

    // --- gfx942 NO_MASK cases ---
    // Minimal batch/heads
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 256, 128}, std::vector<int64_t>{1, 1, 256, 128}, "gfx942");
    // Multi-batch
    cases.emplace_back(
        std::vector<int64_t>{4, 2, 128, 128}, std::vector<int64_t>{4, 2, 128, 128}, "gfx942");
    // Long Q, short KV
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 512, 128}, std::vector<int64_t>{1, 1, 128, 128}, "gfx942");
    // Short Q, long KV
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 128, 128}, std::vector<int64_t>{1, 1, 512, 128}, "gfx942");
    // HD192 kernel
    cases.emplace_back(
        std::vector<int64_t>{1, 2, 256, 192}, std::vector<int64_t>{1, 2, 128, 128}, "gfx942");

    // --- gfx942 BOTTOM_RIGHT_CAUSAL cases ---
    // Causal basic
    cases.emplace_back(std::vector<int64_t>{1, 1, 256, 128},
                       std::vector<int64_t>{1, 1, 256, 128},
                       "gfx942",
                       -1,
                       0,
                       false);
    // Causal multi-batch
    cases.emplace_back(std::vector<int64_t>{2, 4, 256, 128},
                       std::vector<int64_t>{2, 4, 256, 128},
                       "gfx942",
                       -1,
                       0,
                       false);
    // Causal HD192
    cases.emplace_back(std::vector<int64_t>{1, 1, 128, 192},
                       std::vector<int64_t>{1, 1, 128, 128},
                       "gfx942",
                       -1,
                       0,
                       false);

    // --- gfx950 NO_MASK cases ---
    // Minimal batch/heads
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 256, 128}, std::vector<int64_t>{1, 1, 256, 128}, "gfx950");
    // Multi-batch
    cases.emplace_back(
        std::vector<int64_t>{4, 2, 128, 128}, std::vector<int64_t>{4, 2, 128, 128}, "gfx950");
    // Long Q, short KV
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 512, 128}, std::vector<int64_t>{1, 1, 128, 128}, "gfx950");
    // Short Q, long KV
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 128, 128}, std::vector<int64_t>{1, 1, 512, 128}, "gfx950");
    // HD192 kernel
    cases.emplace_back(
        std::vector<int64_t>{1, 2, 256, 192}, std::vector<int64_t>{1, 2, 128, 128}, "gfx950");

    return cases;
}

std::vector<SdpaFwdTestCase> getSdpaFwdGqaTestCases()
{
    std::vector<SdpaFwdTestCase> cases;

    // --- gfx942 GQA NO_MASK ---
    // GQA ratio=4
    cases.emplace_back(
        std::vector<int64_t>{1, 4, 256, 128}, std::vector<int64_t>{1, 1, 256, 128}, "gfx942");
    // GQA ratio=2
    cases.emplace_back(
        std::vector<int64_t>{1, 4, 256, 128}, std::vector<int64_t>{1, 2, 256, 128}, "gfx942");
    // GQA ratio=8
    cases.emplace_back(
        std::vector<int64_t>{1, 8, 128, 128}, std::vector<int64_t>{1, 1, 128, 128}, "gfx942");
    // GQA + asymmetric seq
    cases.emplace_back(
        std::vector<int64_t>{2, 4, 256, 128}, std::vector<int64_t>{2, 1, 128, 128}, "gfx942");
    // GQA + HD192
    cases.emplace_back(
        std::vector<int64_t>{1, 4, 256, 192}, std::vector<int64_t>{1, 1, 256, 128}, "gfx942");

    // --- gfx942 GQA BOTTOM_RIGHT_CAUSAL ---
    cases.emplace_back(std::vector<int64_t>{1, 4, 256, 128},
                       std::vector<int64_t>{1, 1, 256, 128},
                       "gfx942",
                       -1,
                       0,
                       false);

    // --- gfx950 GQA NO_MASK ---
    // GQA ratio=4
    cases.emplace_back(
        std::vector<int64_t>{1, 4, 256, 128}, std::vector<int64_t>{1, 1, 256, 128}, "gfx950");
    // GQA ratio=2
    cases.emplace_back(
        std::vector<int64_t>{1, 4, 256, 128}, std::vector<int64_t>{1, 2, 256, 128}, "gfx950");
    // GQA ratio=8
    cases.emplace_back(
        std::vector<int64_t>{1, 8, 128, 128}, std::vector<int64_t>{1, 1, 128, 128}, "gfx950");
    // GQA + asymmetric seq
    cases.emplace_back(
        std::vector<int64_t>{2, 4, 256, 128}, std::vector<int64_t>{2, 1, 128, 128}, "gfx950");
    // GQA + HD192
    cases.emplace_back(
        std::vector<int64_t>{1, 4, 256, 192}, std::vector<int64_t>{1, 1, 256, 128}, "gfx950");

    return cases;
}

std::vector<SdpaFwdTestCase> getSdpaFwdAsymSeqTestCases()
{
    std::vector<SdpaFwdTestCase> cases;

    // --- gfx942 NO_MASK ---
    // Q > KV
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 256, 128}, std::vector<int64_t>{1, 1, 128, 128}, "gfx942");
    // KV > Q
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 128, 128}, std::vector<int64_t>{1, 1, 256, 128}, "gfx942");
    // Large asymmetry
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 512, 128}, std::vector<int64_t>{1, 1, 64, 128}, "gfx942");
    // Asym + HD192
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 256, 192}, std::vector<int64_t>{1, 1, 512, 128}, "gfx942");

    // --- gfx942 BOTTOM_RIGHT_CAUSAL ---
    // Asym + causal
    cases.emplace_back(std::vector<int64_t>{1, 1, 256, 128},
                       std::vector<int64_t>{1, 1, 128, 128},
                       "gfx942",
                       -1,
                       0,
                       false);
    // Reversed asym + causal
    cases.emplace_back(std::vector<int64_t>{1, 1, 128, 128},
                       std::vector<int64_t>{1, 1, 256, 128},
                       "gfx942",
                       -1,
                       0,
                       false);

    // --- gfx950 NO_MASK ---
    // Q > KV
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 256, 128}, std::vector<int64_t>{1, 1, 128, 128}, "gfx950");
    // KV > Q
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 128, 128}, std::vector<int64_t>{1, 1, 256, 128}, "gfx950");
    // Large asymmetry
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 512, 128}, std::vector<int64_t>{1, 1, 64, 128}, "gfx950");
    // Asym + HD192
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 256, 192}, std::vector<int64_t>{1, 1, 512, 128}, "gfx950");

    return cases;
}

using IntegrationGpuSdpaFwdShapeSweepBf16 = IntegrationSdpaFwdShapeSweep<bfloat16>;

} // namespace

TEST_P(IntegrationGpuSdpaFwdBf16, Correctness)
{
    auto tolerance = 1e-2f;
    runGraphTest(tolerance);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaFwdBf16,
                         testing::ValuesIn(getCompatibleGraphTestCases(cfg_fmha_fwd)),
                         GraphTestCase::getName);

TEST_P(IntegrationGpuSdpaFwdShapeSweepBf16, Correctness)
{
    auto tolerance = 1e-2f;
    runGraphTest(tolerance);
}

INSTANTIATE_TEST_SUITE_P(ShapeSweep,
                         IntegrationGpuSdpaFwdShapeSweepBf16,
                         testing::ValuesIn(getSdpaFwdShapeSweepTestCases()),
                         SdpaFwdTestCase::getName);

INSTANTIATE_TEST_SUITE_P(Gqa,
                         IntegrationGpuSdpaFwdShapeSweepBf16,
                         testing::ValuesIn(getSdpaFwdGqaTestCases()),
                         SdpaFwdTestCase::getName);

INSTANTIATE_TEST_SUITE_P(AsymmetricSeq,
                         IntegrationGpuSdpaFwdShapeSweepBf16,
                         testing::ValuesIn(getSdpaFwdAsymSeqTestCases()),
                         SdpaFwdTestCase::getName);
