// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Mirror the miopen-provider unit-test main: route plugin logs
    // through the test-sdk recorder so they appear on stderr based on
    // HIPDNN_LOG_LEVEL but are not pushed to a real backend.
    auto recordingCallback = hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();
    hipdnn_plugin_sdk::logging::initializeCallbackLogging("ck_dsl_provider_unit_tests",
                                                          recordingCallback);

    return RUN_ALL_TESTS();
}
