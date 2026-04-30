// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_test_sdk/utilities/FrontendGraphFactory.hpp>

#include "../tests/common/TestWorkarounds.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_test_sdk::utilities;

namespace
{

/// Custom test name generator for gtest parameterized tests
struct OperationTypeNameGenerator
{
    std::string operator()(const ::testing::TestParamInfo<OperationType>& info) const
    {
        return operationTypeToString(info.param);
    }
};

/// Test fixture for benchmarking knob integration tests
/// Inherits from IntegrationGraphVerificationHarness for SetUp/TearDown
class IntegrationGpuBenchmarkingKnob
    : public miopen_plugin::test_utilities::IntegrationGraphVerificationHarness<float,
                                                                                OperationType>
{
};

/// Single parameterized test that runs for all operations
TEST_P(IntegrationGpuBenchmarkingKnob, ExecutesSuccessfully)
{
    if(GetParam() == OperationType::CONV_FWD_BIAS_ACTIV)
    {
        SKIP_IF_WORKAROUND_ISSUE_5409();
    }

    auto graph = FrontendGraphFactory::create(GetParam());

    std::vector<KnobSetting> knobSettings;
    knobSettings.emplace_back(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1LL);

    executeGraphWithKnobs(graph, knobSettings);
}

/// Instantiate test suite for all operation types
INSTANTIATE_TEST_SUITE_P(BenchmarkingSmoke,
                         IntegrationGpuBenchmarkingKnob,
                         ::testing::Values(OperationType::CONV_FORWARD,
                                           OperationType::CONV_BACKWARD_DATA,
                                           OperationType::CONV_BACKWARD_WEIGHTS,
                                           OperationType::CONV_FWD_BIAS_ACTIV,
                                           OperationType::BATCHNORM_TRAINING,
                                           OperationType::BATCHNORM_INFERENCE,
                                           OperationType::BATCHNORM_BACKWARD),
                         OperationTypeNameGenerator());

} // namespace
