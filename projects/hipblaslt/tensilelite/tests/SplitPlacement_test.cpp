// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "SplitPlacement.hpp"

#include <gtest/gtest.h>

#include <string>

using namespace TensileLite::Client;

TEST(SplitPlacement, ValidForMSplit)
{
    for(auto p : {"RRRR", "RSRR", "RRRS", "RSRS"})
        EXPECT_EQ(splitPlacementError(p, false), "") << p;
}

TEST(SplitPlacement, ValidForNSplit)
{
    for(auto p : {"RRRR", "SRRR", "RRRS", "SRRS"})
        EXPECT_EQ(splitPlacementError(p, true), "") << p;
}

TEST(SplitPlacement, SharedSplitOperandRejected)
{
    for(auto p : {"SRRR", "SSRR", "SRRS"})
        EXPECT_NE(splitPlacementError(p, false), "") << p;
    for(auto p : {"RSRR", "RSRS", "SSRR"})
        EXPECT_NE(splitPlacementError(p, true), "") << p;
}

TEST(SplitPlacement, SharedCRejected)
{
    for(bool splitN : {false, true})
        for(auto p : {"RRSR", "RRSS", "SSSS"})
            EXPECT_NE(splitPlacementError(p, splitN), "") << p;
}

TEST(SplitPlacement, BadFormatRejected)
{
    for(bool splitN : {false, true})
        for(auto p : {"", "RRR", "RRRRR", "RRRX", "rrrr", "RR R"})
            EXPECT_NE(splitPlacementError(p, splitN).find("4 letters"), std::string::npos) << p;
}

TEST(SplitPlacement, FormatOnlyIgnoresTheSharingRules)
{
    // Without a split only the format is checked: every R/S combination is accepted.
    for(auto p : {"RRRR", "SRRR", "RSRR", "RRSR", "SSSS"})
        EXPECT_EQ(splitPlacementFormatError(p), "") << p;
    for(auto p : {"", "RRR", "RRRRR", "RRRX", "rrrr", "RR R"})
        EXPECT_NE(splitPlacementFormatError(p).find("4 letters"), std::string::npos) << p;
}

TEST(SplitPlacement, MessageNamesTheOperand)
{
    auto const m = splitPlacementError("SRRR", false);
    EXPECT_NE(m.find("A cannot be shared in an M split"), std::string::npos) << m;
    EXPECT_NE(m.find("only B"), std::string::npos) << m;
    auto const n = splitPlacementError("RSRR", true);
    EXPECT_NE(n.find("B cannot be shared in an N split"), std::string::npos) << n;
    EXPECT_NE(n.find("only A"), std::string::npos) << n;
    auto const c = splitPlacementError("RRSR", false);
    EXPECT_NE(c.find("C cannot be shared in an M split"), std::string::npos) << c;
    EXPECT_NE(c.find("each device has its own C slice"), std::string::npos) << c;
}
