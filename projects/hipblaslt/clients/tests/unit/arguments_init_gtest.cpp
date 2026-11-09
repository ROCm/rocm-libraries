// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit test of Arguments::init(). See README.md in this directory
// for the HostUnit/smoke_ naming.

#include "hipblaslt_arguments.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>

// init() sets every field, so its result does not depend on what the memory
// held before; batch offsets, bias_stride and the trailing members of
// per-problem arrays such as M used to keep their previous contents.
TEST(HostUnitArguments, smoke_InitSetsEveryField)
{
    auto zeroed  = std::make_unique<Arguments>();
    auto garbage = std::make_unique<Arguments>();
    std::memset(static_cast<void*>(garbage.get()), 0xA5, sizeof(Arguments));
    zeroed->init();
    garbage->init();
    EXPECT_EQ(garbage->batch_offset_a, 0);
    EXPECT_EQ(garbage->bias_stride, 0);
    EXPECT_EQ(garbage->M[1], 0);
    EXPECT_EQ(std::memcmp(zeroed.get(), garbage.get(), sizeof(Arguments)), 0);
}
