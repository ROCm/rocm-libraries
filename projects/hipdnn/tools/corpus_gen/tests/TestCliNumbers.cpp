// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCliNumbers.cpp
 * @brief A numeric command-line value is the whole of its text, or it is refused.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/CliNumbers.hpp>

namespace hipdnn_corpus_gen
{

TEST(TestCliNumbers, AnIntegerIsTheWholeText)
{
    EXPECT_EQ(parseInteger("200"), 200);
    EXPECT_EQ(parseInteger("-3"), -3);
    // strtoll would read these as 0, 5 and 0: a different request than the one typed.
    EXPECT_FALSE(parseInteger("abc"));
    EXPECT_FALSE(parseInteger("5x"));
    EXPECT_FALSE(parseInteger(""));
    EXPECT_FALSE(parseInteger("1.5"));
    EXPECT_FALSE(parseInteger("99999999999999999999"));
}

TEST(TestCliNumbers, ARealIsTheWholeTextAndFinite)
{
    EXPECT_DOUBLE_EQ(*parseReal("0.05"), 0.05);
    EXPECT_DOUBLE_EQ(*parseReal("2"), 2.0);
    EXPECT_FALSE(parseReal("x"));
    EXPECT_FALSE(parseReal("0.5x"));
    EXPECT_FALSE(parseReal("nan"));
    EXPECT_FALSE(parseReal("inf"));
}

TEST(TestCliNumbers, AnEngineIdIsDecimalOrHex)
{
    EXPECT_EQ(parseEngineId("31"), 31);
    EXPECT_EQ(parseEngineId("0x1F"), 31);
    EXPECT_EQ(parseEngineId("-677973494996465522"), -677973494996465522);
    EXPECT_FALSE(parseEngineId("ten"));
    EXPECT_FALSE(parseEngineId("0x"));
}

} // namespace hipdnn_corpus_gen
