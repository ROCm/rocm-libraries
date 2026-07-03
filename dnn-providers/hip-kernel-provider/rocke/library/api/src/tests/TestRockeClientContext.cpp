// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "RockeClientContext.hpp"

TEST(TestRockeClientContext, DefaultConstructsEmptyContext)
{
    rocke_client::RockeClientContext context;

    EXPECT_FALSE(context.hasValidPlan());
}
