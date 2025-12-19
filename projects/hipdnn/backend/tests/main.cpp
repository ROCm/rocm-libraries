/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include "logging/Logging.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>

int main(int argc, char** argv)
{
    hipdnn_backend::logging::initialize();

    ::testing::InitGoogleTest(&argc, argv);

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    return RUN_ALL_TESTS();
}
