// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/TestSdk.hpp>

TEST(TestSdk, Example)
{
    hipdnn::test_sdk::hello();
    EXPECT_TRUE(true);
}
