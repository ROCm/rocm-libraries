/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_test_sdk/utilities/LoggingUtils.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    hipdnn::logging::initializeCallbackLogging(COMPONENT_NAME,
                                               hipdnn_test_sdk::utilities::testLoggingCallback);

    return RUN_ALL_TESTS();
}
