/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>

#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize test logging infrastructure to forward logs to std::cerr based
    // on the current environment HIPDNN_LOG_LEVEL value when this function is called.
    // NOTE: Logs are not routed to the backend by the recordingCallback returned here
    // which is the desired behaviour because this is a frontend unit test harness.
    auto recordingCallback = hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();

    // Initialize frontend logging with test recording callback so that frontend
    // logs are routed to the log recorder for capture and use by the unit tests.
    hipdnn_frontend::initializeFrontendLogging(recordingCallback);

    auto result = RUN_ALL_TESTS();
    return result;
}
