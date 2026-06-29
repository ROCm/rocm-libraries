// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "RocKEContext.hpp"

TEST(TestRocKEContext, DefaultConstructsEmptyContext)
{
    rocke_client::RocKEContext context;

    EXPECT_FALSE(context.hasValidPlan());
    EXPECT_NE(static_cast<HipdnnEnginePluginExecutionContext*>(&context), nullptr);
}
