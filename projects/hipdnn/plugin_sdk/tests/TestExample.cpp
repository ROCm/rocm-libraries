// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/PluginSdk.hpp>

TEST(PluginSdk, Example)
{
    hipdnn::plugin_sdk::hello();
    EXPECT_TRUE(true);
}
