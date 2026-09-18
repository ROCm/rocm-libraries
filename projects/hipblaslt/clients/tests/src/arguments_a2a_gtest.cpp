// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for the fused-A2A fields on Arguments.

#include "hipblaslt_arguments.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

TEST(arguments_a2a_smoke, defaults_are_the_single_rank_case)
{
    Arguments arg;
    arg.init();

    EXPECT_EQ(arg.a2a_world, 1);
    EXPECT_EQ(arg.a2a_extent, 0);
}

TEST(arguments_a2a_smoke, fields_are_part_of_the_argument_list)
{
    Arguments arg;
    arg.init();
    arg.a2a_extent = 10240;

    std::ostringstream oss;
    oss << arg;
    const std::string yaml = oss.str();

    EXPECT_NE(yaml.find("a2a_world: "), std::string::npos) << yaml;
    EXPECT_NE(yaml.find("a2a_extent: 10240"), std::string::npos) << yaml;
}
