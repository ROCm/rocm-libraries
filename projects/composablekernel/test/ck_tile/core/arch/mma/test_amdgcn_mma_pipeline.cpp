// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_pipeline.hpp"

namespace {
using namespace ck_tile::core::arch::mma;
}

TEST(MmaPipelineOptionFlagsTests, ConversionTests)
{
    MmaPipelineOptionFlags flags_0{};
    MmaPipelineOptionFlags flags_1{MmaPipelineOptionFlag::C_TRANSPOSE};
    MmaPipelineOptionFlags flags_2{MmaPipelineOptionFlag::COMPRESS_A};
    MmaPipelineOptionFlags flags_3{0b11};

    EXPECT_EQ(flags_0, 0);
    EXPECT_TRUE(flags_0.testFlag(MmaPipelineOptionFlag::NONE));

    EXPECT_EQ(flags_1, 1);
    EXPECT_TRUE(flags_1.testFlag(MmaPipelineOptionFlag::C_TRANSPOSE));

    EXPECT_EQ(flags_2, 2);
    EXPECT_TRUE(flags_2.testFlag(MmaPipelineOptionFlag::COMPRESS_A));

    EXPECT_EQ(flags_3, 3);
    EXPECT_TRUE(flags_3.testFlag(MmaPipelineOptionFlag::COMPRESS_A));
    EXPECT_TRUE(flags_3.testFlag(MmaPipelineOptionFlag::C_TRANSPOSE));
}

TEST(MmaPipelineOptionFlagsTests, OperatorsTests)
{
    MmaPipelineOptionFlags flags{};

    EXPECT_TRUE(flags.testFlag(MmaPipelineOptionFlag::NONE));

    flags |= MmaPipelineOptionFlag::C_TRANSPOSE;

    EXPECT_FALSE(flags.testFlag(MmaPipelineOptionFlag::NONE));
    EXPECT_TRUE(flags.testFlag(MmaPipelineOptionFlag::C_TRANSPOSE));

    flags |= MmaPipelineOptionFlag::COMPRESS_A;

    EXPECT_FALSE(flags.testFlag(MmaPipelineOptionFlag::NONE));
    EXPECT_TRUE(flags.testFlag(MmaPipelineOptionFlag::C_TRANSPOSE));
    EXPECT_TRUE(flags.testFlag(MmaPipelineOptionFlag::COMPRESS_A));

    flags &= MmaPipelineOptionFlag::COMPRESS_A;

    EXPECT_FALSE(flags.testFlag(MmaPipelineOptionFlag::NONE));
    EXPECT_FALSE(flags.testFlag(MmaPipelineOptionFlag::C_TRANSPOSE));
    EXPECT_TRUE(flags.testFlag(MmaPipelineOptionFlag::COMPRESS_A));

    EXPECT_FALSE((~flags).testFlag(MmaPipelineOptionFlag::NONE));
    EXPECT_TRUE((~flags).testFlag(MmaPipelineOptionFlag::C_TRANSPOSE));
    EXPECT_FALSE((~flags).testFlag(MmaPipelineOptionFlag::COMPRESS_A));
}
