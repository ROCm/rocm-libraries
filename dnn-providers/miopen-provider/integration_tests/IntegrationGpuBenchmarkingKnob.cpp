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

/// CBA-only fixture: overrides the engine-config hook to skip when the device
/// has no applicable CBA fusion solution (gfx90a + RDNA archs without CK
/// fusion kernels).
class IntegrationGpuBenchmarkingKnobCba : public IntegrationGpuBenchmarkingKnob
{
protected:
    std::optional<std::string>
        shouldSkipOnEngineConfigResult(const hipdnn_frontend::Error& result) override
    {
        if(IS_WORKAROUND_ISSUE_6979(result))
        {
            return WORKAROUND_ISSUE_6979_SKIP_MSG;
        }
        return std::nullopt;
    }
};

/// Single parameterized test that runs for all operations
TEST_P(IntegrationGpuBenchmarkingKnob, ExecutesSuccessfully)
{
    // rocBLAS/Tensile heap-buffer-overflow on gfx90a; CK ASAN stall on gfx942
    if(GetParam() == OperationType::CONV_FORWARD || GetParam() == OperationType::CONV_BACKWARD_DATA
       || GetParam() == OperationType::CONV_BACKWARD_WEIGHTS)
    {
        SKIP_IF_ASAN();
    }
    auto graph = FrontendGraphFactory::create(GetParam());

    std::vector<KnobSetting> knobSettings;
    knobSettings.emplace_back(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1LL);

    executeGraphWithKnobs(graph, knobSettings);
}

TEST_P(IntegrationGpuBenchmarkingKnobCba, ExecutesSuccessfully)
{
    // rocBLAS/Tensile heap-buffer-overflow on gfx90a; CK ASAN stall on gfx942
    SKIP_IF_ASAN();
    auto graph = FrontendGraphFactory::create(GetParam());

    std::vector<KnobSetting> knobSettings;
    knobSettings.emplace_back(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, 1LL);

    executeGraphWithKnobs(graph, knobSettings);
}

/// Instantiate test suite for all non-CBA operation types
INSTANTIATE_TEST_SUITE_P(BenchmarkingSmoke,
                         IntegrationGpuBenchmarkingKnob,
                         ::testing::Values(OperationType::CONV_FORWARD,
                                           OperationType::CONV_BACKWARD_DATA,
                                           OperationType::CONV_BACKWARD_WEIGHTS,
                                           OperationType::BATCHNORM_TRAINING,
                                           OperationType::BATCHNORM_INFERENCE,
                                           OperationType::BATCHNORM_BACKWARD),
                         OperationTypeNameGenerator());

/// CBA gets its own instantiation so the fixture override applies.
INSTANTIATE_TEST_SUITE_P(BenchmarkingSmoke,
                         IntegrationGpuBenchmarkingKnobCba,
                         ::testing::Values(OperationType::CONV_FWD_BIAS_ACTIV),
                         OperationTypeNameGenerator());

} // namespace
