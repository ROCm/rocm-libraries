// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCsvOutput.cpp
 * @brief Covers quoting for the free-text skip_reason column (RFC 0019.13 §7.4).
 */

#include <gtest/gtest.h>

#include <hipdnn_bench/CsvOutput.hpp>

namespace hipdnn_bench
{

TEST(TestCsvOutput, AnOrdinaryFieldIsLeftAlone)
{
    EXPECT_EQ(csvField(""), "");
    EXPECT_EQ(csvField("config_not_applicable: engine declined"),
              "config_not_applicable: engine declined");
    EXPECT_EQ(csvField("0.123456"), "0.123456");
}

TEST(TestCsvOutput, ACommaIsQuotedRatherThanSplittingTheRow)
{
    EXPECT_EQ(csvField("not_applicable: M below minimum, K above maximum"),
              "\"not_applicable: M below minimum, K above maximum\"");
}

TEST(TestCsvOutput, AnEmbeddedQuoteIsDoubled)
{
    // RFC 4180.
    EXPECT_EQ(csvField("tile \"m\" too large"), "\"tile \"\"m\"\" too large\"");
}

TEST(TestCsvOutput, ANewlineIsQuoted)
{
    EXPECT_EQ(csvField("declined:\nno kernel"), "\"declined:\nno kernel\"");
}

TEST(TestCsvOutput, AQuotedFieldRoundTripsBackToItsInput)
{
    const auto parse = [](const std::string& field) {
        if(field.size() < 2 || field.front() != '"')
        {
            return field;
        }
        std::string out;
        for(size_t i = 1; i + 1 < field.size(); ++i)
        {
            if(field[i] == '"' && field[i + 1] == '"')
            {
                ++i;
            }
            out += field[i];
        }
        return out;
    };

    for(const auto& original : {std::string("plain"),
                                std::string("a,b"),
                                std::string("say \"hi\""),
                                std::string("line\nbreak"),
                                std::string("\"leading quote")})
    {
        EXPECT_EQ(parse(csvField(original)), original) << "did not round-trip: " << original;
    }
}

} // namespace hipdnn_bench
