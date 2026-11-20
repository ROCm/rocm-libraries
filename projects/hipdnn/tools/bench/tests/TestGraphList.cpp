// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_bench/GraphList.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

TEST(TestGraphList, PathsKeepTheirOrderAndSpacesAndLoseOnlyAWindowsLineEnd)
{
    std::istringstream list("b/second.fb\r\n\n   \na dir/first graph.fb\nc.json");
    EXPECT_EQ(hipdnn_bench::readGraphList(list),
              (std::vector<std::string>{"b/second.fb", "a dir/first graph.fb", "c.json"}));
}

TEST(TestGraphList, AListNamingNoGraphIsRefused)
{
    std::istringstream list("\n \r\n");
    EXPECT_THROW(static_cast<void>(hipdnn_bench::readGraphList(list)), std::invalid_argument);
}
