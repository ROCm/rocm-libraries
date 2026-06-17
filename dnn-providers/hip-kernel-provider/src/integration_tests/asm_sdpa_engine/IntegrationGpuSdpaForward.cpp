// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hip_kernel_provider_common/SdpaConfigEnumerations.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
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

        auto validationResult = testCase.graph->validate();
        ASSERT_TRUE(validationResult.is_good())
            << "Graph validation failed for config: " << testCase.name << " - "
            << validationResult.get_message();

        // Register output tensor validator
        testCase.graph->visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(!tensorAttr->get_is_virtual())
                {
                    this->registerValidator(tensorAttr, tolerance);
                }
            }
        });

        this->verifyGraph(*testCase.graph, 0);
    }

    float _minVal = -1.0;
    float _maxVal = 1.0;
};

using IntegrationGpuSdpaFwdBf16 = IntegrationSdpaFwd<bfloat16>;

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
