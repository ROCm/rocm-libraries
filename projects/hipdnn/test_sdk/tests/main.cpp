// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/test_utilities/HipErrorHandler.hpp>
#include <hipdnn_sdk/test_utilities/LoggingUtils.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    hipdnn::logging::initializeCallbackLogging(COMPONENT_NAME,
                                               hipdnn_sdk::test_utilities::testLoggingCallback);

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_sdk::test_utilities::HipErrorHandler);

    return RUN_ALL_TESTS();
}
