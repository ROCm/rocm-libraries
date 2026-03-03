// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_test_sdk/utilities/LoggingUtils.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Register callback to output logs during tests.
    // Log level is automatically initialized from HIPDNN_LOG_LEVEL env var on first use.
    hipdnn_data_sdk::logging::registerLoggingCallback(
        hipdnn_test_sdk::utilities::testLoggingCallback);

    auto result = RUN_ALL_TESTS();
    return result;
}
