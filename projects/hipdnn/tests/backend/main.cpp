// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>

// Custom main() to register HipErrorHandler event listener.
// Cannot use GTest::gtest_main because event listeners must be
// registered before RUN_ALL_TESTS(), which requires custom main().
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    auto result = RUN_ALL_TESTS();
    return result;
}
