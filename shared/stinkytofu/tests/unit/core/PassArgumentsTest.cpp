// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "stinkytofu/support/PassArguments.hpp"

using namespace stinkytofu;

TEST(PassArgumentsTest, SplitPassArgsTrimsEachArgument) {
    EXPECT_EQ(splitPassArgs(" apply , classes=vs ,\tallocator=greedy+greedy-compact "),
              (std::vector<std::string>{"apply", "classes=vs", "allocator=greedy+greedy-compact"}));
}

TEST(PassArgumentsTest, SplitPassArgsDropsEmptyArguments) {
    EXPECT_EQ(splitPassArgs("apply,, ,report,"), (std::vector<std::string>{"apply", "report"}));
    EXPECT_TRUE(splitPassArgs("").empty());
    EXPECT_TRUE(splitPassArgs(" , ").empty());
}
