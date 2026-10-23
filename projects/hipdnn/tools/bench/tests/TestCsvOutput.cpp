// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCsvOutput.cpp
 * @brief Covers quoting for the bench CSV's free-text columns: the problem id, query values,
 *        knob names and values, the engine name, and skip_reason (RFC 0019.13 §7.4); and the
 *        columns uhd_gen's dataset reads.
 */

#include <gtest/gtest.h>

#include <hipdnn_bench/CsvOutput.hpp>
#include <hipdnn_bench/DeviceIdentity.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

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

TEST(TestCsvOutput, ACarriageReturnIsQuoted)
{
    // A CRLF-terminated line inside a reason would otherwise end the row early.
    EXPECT_EQ(csvField("declined\r\nno kernel"), "\"declined\r\nno kernel\"");
    EXPECT_EQ(csvField("declined\r"), "\"declined\r\"");
}

TEST(TestCsvOutput, AWindowsPathWithACommaIsOneField)
{
    // The problem column defaults to the graph's path.
    EXPECT_EQ(csvField(R"(C:\graphs\conv,fp16.json)"), R"("C:\graphs\conv,fp16.json")");
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

/// The header is the one uhd_gen's dataset tests convert (fixtures/hipdnn_bench.csv), so a
/// column bench drops or renames fails here before the dataset refuses real output.
TEST(TestCsvOutput, TheHeaderIsTheOneTheDatasetFixtureConverts)
{
    hipDeviceProp_t properties{};
    std::snprintf(
        properties.gcnArchName, sizeof(properties.gcnArchName), "%s", "gfx942:sramecc+:xnack-");
    const auto columns = deviceColumns(properties);
    std::vector<std::string> deviceNames;
    deviceNames.reserve(columns.size());
    for(const auto& column : columns)
    {
        deviceNames.push_back(column.first);
    }

    // Staged with the test data at the same offset from the test binary in the build and
    // install trees (tests/CMakeLists.txt).
    const auto samplePath = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory()
                            / HIPDNN_TEST_DATA_RELDIR / HIPDNN_BENCH_CSV_SAMPLE_SUBDIR
                            / "hipdnn_bench.csv";
    std::ifstream sample(samplePath);
    std::string expected;
    ASSERT_TRUE(std::getline(sample, expected)) << samplePath;
    EXPECT_EQ(csvHeader(deviceNames, {"M"}, {"tile_m"}), expected);
}

TEST(TestCsvOutput, ASweepNamesItsProblemByTheFeaturesTheRuntimeExtracted)
{
    const auto columns = problemColumns(nlohmann::json{{"graph.flops", 2.5e12},
                                                       {"sdpa.head_dim", 128},
                                                       {"graph.nodes[0].causal", true},
                                                       {"sdpa.layout", "bshd"}});
    // In the page's (sorted) order, each under its published name.
    const std::vector<std::pair<std::string, std::string>> expected{
        {"graph.flops", "2500000000000.0"},
        {"graph.nodes[0].causal", "True"},
        {"sdpa.head_dim", "128"},
        {"sdpa.layout", "bshd"}};
    EXPECT_EQ(columns, expected);
    EXPECT_TRUE(problemColumns(nlohmann::json::object()).empty());
}

TEST(TestCsvOutput, AnUntimedCandidateHasEmptyTimings)
{
    EXPECT_EQ(csvTiming(true, 0.5F), "0.5");
    EXPECT_EQ(csvTiming(false, 0.0F), "");
    EXPECT_EQ(csvTiming(false, 0.5F), "");
}

namespace
{

std::vector<std::string> splitRow(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    for(const char c : line)
    {
        if(c == ',')
        {
            fields.push_back(field);
            field.clear();
            continue;
        }
        field += c;
    }
    fields.push_back(field);
    return fields;
}

} // namespace

TEST(TestCsvOutput, EveryValueOfARowLandsUnderItsOwnHeaderColumn)
{
    // Distinct values throughout, so a swapped column (minTimeMs written as avgTimeMs, the
    // training label) shows.
    CandidateOutcome outcome;
    outcome.engine = "E";
    outcome.rank = 3;
    outcome.succeeded = true;
    outcome.iterationsRun = 30;
    outcome.verdict = "True";
    outcome.validation = "agreed";
    outcome.minTimeMs = 1.25F;
    outcome.avgTimeMs = 2.5F;
    outcome.robustTimeMs = 2.25F;
    outcome.stddevMs = 0.125F;
    outcome.converged = true;
    outcome.workspaceBytes = 4096;

    const auto header = splitRow(csvHeader({"device.cu_count"}, {"M"}, {"tile_m"}));
    const auto row = splitRow(csvRow("p", "1234", {"304"}, {"1024"}, {"128"}, outcome));
    ASSERT_EQ(row.size(), header.size());
    std::map<std::string, std::string> cell;
    for(size_t i = 0; i < header.size(); ++i)
    {
        cell[header[i]] = row[i];
    }
    const std::map<std::string, std::string> expected{{"problem", "p"},
                                                      {"benchmark", "1234"},
                                                      {"device.cu_count", "304"},
                                                      {"q.M", "1024"},
                                                      {"kernel.tile_m", "128"},
                                                      {"engine", "E"},
                                                      {"rank", "3"},
                                                      {"succeeded", "1"},
                                                      {"is_valid", "True"},
                                                      {"numerically_valid", "True"},
                                                      {"validation", "agreed"},
                                                      {"skip_reason", ""},
                                                      {"minTimeMs", "1.25"},
                                                      {"avgTimeMs", "2.5"},
                                                      {"robustMeanMs", "2.25"},
                                                      {"stddevMs", "0.125"},
                                                      {"iters", "30"},
                                                      {"converged", "1"},
                                                      {"workspace_bytes", "4096"}};
    EXPECT_EQ(cell, expected);
}

TEST(TestCsvOutput, AnUntimedRowIsInvalidWithItsReasonAndNoTimings)
{
    CandidateOutcome outcome;
    outcome.succeeded = true; // success without an iteration is not a measurement
    outcome.avgTimeMs = 2.5F;
    const auto header = splitRow(csvHeader({}, {}, {}));
    const auto row = splitRow(csvRow("p", "1", {}, {}, {}, outcome));
    ASSERT_EQ(row.size(), header.size());
    std::map<std::string, std::string> cell;
    for(size_t i = 0; i < header.size(); ++i)
    {
        cell[header[i]] = row[i];
    }
    EXPECT_EQ(cell["is_valid"], "False");
    EXPECT_EQ(cell["avgTimeMs"], "");
    EXPECT_EQ(cell["skip_reason"].rfind("not_timed", 0), 0U) << cell["skip_reason"];
}

} // namespace hipdnn_bench
