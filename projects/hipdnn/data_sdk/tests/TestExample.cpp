// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/DataSdk.hpp>

TEST(TestDataSdk, Example)
{
    hipdnn_data_sdk::hello();
    EXPECT_TRUE(true);
}
