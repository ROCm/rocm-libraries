/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Clear HIPDNN_FORCE_BENCHMARKING for the whole process. It is a global override of
    // the global.benchmarking knob, so a value inherited from the runner would silently
    // flip what the knob-only cases assert. The override's own cases set it explicitly,
    // so this only establishes the baseline they start from.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter forceBenchmarkingGuard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME);

    // Initialize test logging infrastructure to forward logs to std::cerr based
    // on the current environment HIPDNN_LOG_LEVEL value when this function is called.
    // NOTE: Logs are not routed to the backend by the recordingCallback returned here
    // which is the desired behaviour because this is a plugin unit test harness.
    auto recordingCallback = hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();

    // Initialize plugin logger with test recording callback so that plugin logs
    // logs are first routed to the log recorder for capture and use by the unit tests.
    hipdnn_plugin_sdk::logging::initializeCallbackLogging("miopen_plugin_tests", recordingCallback);

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    return RUN_ALL_TESTS();
}
