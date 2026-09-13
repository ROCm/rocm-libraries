// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Clear HIPDNN_FORCE_BENCHMARKING for the whole process. It is a global override of
    // the global.benchmarking knob, so a value inherited from the runner would silently
    // change what buildPlan() produces and what the knob-only cases assert. Clearing it
    // once here means no individual case can be missed the way a per-fixture guard can.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter forceBenchmarkingGuard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME);

    // Initialize test logging infrastructure to forward logs to std::cerr based
    // on the current environment HIPDNN_LOG_LEVEL value when this function is called.
    // NOTE: Logs are not routed to the backend as this is an SDK unit test harness.
    hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();

    auto result = RUN_ALL_TESTS();
    return result;
}
