/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>

#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize test log recording infrastructure to always forward logs to hipdnnLoggingCallback_ext().
    // NOTE: The frontend logger must be initialized with recordingCallback to ensure the logs
    // are recorded and forwarded to the hipdnnLoggingCallback_ext() function.
    auto recordingCallback = hipdnn_test_sdk::utilities::initializeChainedTestLogRecordingShared(
        hipdnnLoggingCallback_ext);

    // Initialize frontend logging with test recording callback returned above.
    // This must be called to override the default callback that would otherwise
    // be used by HIPDNN_FE_LOG_* macros with lazy-initialization.
    hipdnn_frontend::initializeFrontendLogging(recordingCallback);

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    auto result = RUN_ALL_TESTS();
    return result;
}
