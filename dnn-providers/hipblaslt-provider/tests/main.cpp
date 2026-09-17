/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#ifndef HIPDNN_TEST_SKIP_HIP_ERROR_HANDLER
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#endif

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

#ifndef HIPDNN_TEST_SKIP_HIP_ERROR_HANDLER
    // Register HipErrorHandler to check and clear HIP errors after each test.
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);
#endif

    return RUN_ALL_TESTS();
}
