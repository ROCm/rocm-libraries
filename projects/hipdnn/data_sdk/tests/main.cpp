// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize test logging infrastructure to forward logs to std::cerr based
    // on the current environment HIPDNN_LOG_LEVEL value when this function is called.
    // NOTE: Logs are not routed to the backend as this is an SDK unit test harness.
    hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();

    auto result = RUN_ALL_TESTS();
    return result;
}
