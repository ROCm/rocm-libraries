// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCsvOutput.cpp
 * @brief Covers quoting for the one free-text column a harvest emits.
 *
 * skip_reason (RFC 0019.13 §7.4) is written by this tool and read by the trainer. The failure
 * being guarded is not a parse error: an unquoted comma shifts every column to its right, so
 * the timings land under the wrong headers and the model trains on transposed features.
 */

#include <gtest/gtest.h>

#include <hipdnn_bench/CsvOutput.hpp>

namespace hipdnn_bench
{

TEST(TestCsvOutput, AnOrdinaryFieldIsLeftAlone)
{
    // Quoting everything would be safe and would also change every existing column, so the
    // common case has to pass through untouched.
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
    // RFC 4180. A single quote inside an unquoted field, or an unescaped one inside a quoted
    // field, terminates it early and silently truncates the reason.
    EXPECT_EQ(csvField("tile \"m\" too large"), "\"tile \"\"m\"\" too large\"");
}

TEST(TestCsvOutput, ANewlineIsQuoted)
{
    // An engine message carrying a newline would otherwise end the CSV row mid-field and turn
    // the remainder into a short row that a reader may accept.
    EXPECT_EQ(csvField("declined:\nno kernel"), "\"declined:\nno kernel\"");
}

TEST(TestCsvOutput, AQuotedFieldRoundTripsBackToItsInput)
{
    // The property that matters: whatever reason text the engine produces must be recoverable
    // by a conforming reader, since auditing reads exactly this column to answer which
    // variants were never eligible and why.
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
