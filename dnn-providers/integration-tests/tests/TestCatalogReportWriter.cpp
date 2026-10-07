// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The JSON Lines report: appended, one line per record, never truncated.

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "harness/catalog-sweep/CatalogReportWriter.hpp"

using namespace hipdnn_integration_tests::catalog_sweep;
using hipdnn_test_sdk::utilities::claimScratchDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

std::string slurp(const std::filesystem::path& path)
{
    const std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

std::vector<nlohmann::json> lines(const std::filesystem::path& path)
{
    std::vector<nlohmann::json> records;
    std::ifstream in(path, std::ios::binary);
    std::string line;
    while(std::getline(in, line))
    {
        records.push_back(nlohmann::json::parse(line));
    }
    return records;
}

} // namespace

TEST(TestCatalogReportWriter, EachAppendIsOneLineAndTheEarlierLineSurvives)
{
    const auto scratch = claimScratchDirectory("catalog_report");
    const auto path = scratch.path() / "report.jsonl";

    CatalogReportWriter writer(path);
    writer.append({{"graph", "a"}, {"status", "PASS"}});
    writer.append({{"graph", "b"}, {"status", "FAIL_BUILD"}});

    const auto records = lines(path);
    ASSERT_EQ(records.size(), 2U);
    EXPECT_EQ(records[0].at("graph"), "a");
    EXPECT_EQ(records[1].at("status"), "FAIL_BUILD");
}

// Each record is on disk when append() returns: a sweep killed mid-run keeps what it had
// finished. Read through a second handle while the writer is still open.
TEST(TestCatalogReportWriter, ARecordIsReadableBeforeTheWriterIsClosed)
{
    const auto scratch = claimScratchDirectory("catalog_report");
    const auto path = scratch.path() / "report.jsonl";

    CatalogReportWriter writer(path);
    writer.append({{"graph", "a"}});

    EXPECT_EQ(lines(path).size(), 1U);
}

TEST(TestCatalogReportWriter, ANewWriterOnAnExistingReportAppendsAfterItsContent)
{
    const auto scratch = claimScratchDirectory("catalog_report");
    const auto path = scratch.path() / "report.jsonl";
    {
        CatalogReportWriter first(path);
        first.append({{"graph", "shard0"}});
    }
    {
        CatalogReportWriter second(path);
        second.append({{"graph", "shard1"}});
    }

    const auto records = lines(path);
    ASSERT_EQ(records.size(), 2U);
    EXPECT_EQ(records[0].at("graph"), "shard0");
    EXPECT_EQ(records[1].at("graph"), "shard1");
}

TEST(TestCatalogReportWriter, PreexistingTextIsPreservedByteForByte)
{
    const auto scratch = claimScratchDirectory("catalog_report");
    const auto path = scratch.path() / "report.jsonl";
    const std::string earlier = "{\"graph\":\"from_an_earlier_run\"}\n";
    std::ofstream(path, std::ios::binary) << earlier;

    CatalogReportWriter writer(path);
    writer.append({{"graph", "new"}});

    const auto text = slurp(path);
    EXPECT_EQ(text.substr(0, earlier.size()), earlier);
    EXPECT_GT(text.size(), earlier.size());
}

TEST(TestCatalogReportWriter, RecordsAreCompactLinesWithoutCarriageReturns)
{
    const auto scratch = claimScratchDirectory("catalog_report");
    const auto path = scratch.path() / "report.jsonl";

    CatalogReportWriter writer(path);
    writer.append({{"message", "two\nlines"}, {"graph", "a"}});

    const auto text = slurp(path);
    EXPECT_EQ(text.find('\r'), std::string::npos) << "CRLF line ends break line-oriented readers";
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 1)
        << "a newline inside a value must be escaped";
}

TEST(TestCatalogReportWriter, AMessageThatIsNotUtf8IsStillRecorded)
{
    const auto scratch = claimScratchDirectory("catalog_report");
    const auto path = scratch.path() / "report.jsonl";

    CatalogReportWriter writer(path);
    EXPECT_NO_THROW(writer.append({{"message", std::string("bad \xff byte")}, {"graph", "a"}}));
    EXPECT_EQ(lines(path).size(), 1U);
}

TEST(TestCatalogReportWriter, ANonexistentDirectoryRefusesWithThePathNamed)
{
    const auto scratch = claimScratchDirectory("catalog_report");
    const auto path = scratch.path() / "no_such_dir" / "report.jsonl";

    try
    {
        const CatalogReportWriter writer(path);
        FAIL() << "opened a report in a directory that does not exist";
    }
    catch(const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("report.jsonl"), std::string::npos) << e.what();
    }
}

// NOLINTEND(readability-identifier-naming)
