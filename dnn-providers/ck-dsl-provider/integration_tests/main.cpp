// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    auto recordingCallback = hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();
    hipdnn_plugin_sdk::logging::initializeCallbackLogging("ck_dsl_provider_integration_tests",
                                                          recordingCallback);

    return RUN_ALL_TESTS();
}
