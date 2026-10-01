// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/EngineCoverage.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <string>

namespace hipdnn_corpus_gen
{

TEST(TestEngineCoverage, TheShippedTableParses)
{
    // What it lists is data, not contract; that CorpusGen can read it at all is the contract.
    std::ifstream file(HIPDNN_CORPUS_GEN_OPERATIONS_DIR "/engines.json");
    ASSERT_TRUE(file.good()) << "engines.json is not beside the declarations";
    EngineCoverageTable table;
    std::string error;
    EXPECT_TRUE(parseEngineCoverage(nlohmann::json::parse(file), table, error)) << error;
}

TEST(TestEngineCoverage, EachEntryIsRecordedUnderItsEngineNameWithItsKindAndReason)
{
    EngineCoverageTable table;
    std::string error;
    ASSERT_TRUE(parseEngineCoverage(nlohmann::json::parse(R"({"engines": {
                                        "baked": {"coverage": "pack", "reason": "every field baked"},
                                        "open": {"coverage": "search", "reason": "runtime shapes"}
                                    }})"),
                                    table,
                                    error))
        << error;

    ASSERT_EQ(table.size(), 2U);

    const auto baked = table.find("baked");
    ASSERT_NE(baked, table.end());
    EXPECT_EQ(baked->second.coverage, EngineCoverage::PACK);
    EXPECT_EQ(baked->second.reason, "every field baked");

    const auto open = table.find("open");
    ASSERT_NE(open, table.end());
    EXPECT_EQ(open->second.coverage, EngineCoverage::SEARCH);
    EXPECT_EQ(open->second.reason, "runtime shapes");
}

TEST(TestEngineCoverage, AnEntryWithNoCoverageIsSearched)
{
    EngineCoverageTable table;
    std::string error;
    ASSERT_TRUE(parseEngineCoverage(
        nlohmann::json::parse(R"({"engines": {"x": {"reason": "r"}}})"), table, error))
        << error;
    ASSERT_EQ(table.count("x"), 1U);
    EXPECT_EQ(table.at("x").coverage, EngineCoverage::SEARCH);
}

TEST(TestEngineCoverage, APackClaimWithNoReasonIsRefused)
{
    // An entry stops a search; one nobody can check is what the table exists to avoid.
    EngineCoverageTable table;
    std::string error;
    EXPECT_FALSE(parseEngineCoverage(
        nlohmann::json::parse(R"({"engines": {"x": {"coverage": "pack"}}})"), table, error));
    EXPECT_NE(error.find("no reason"), std::string::npos) << error;
}

TEST(TestEngineCoverage, AnUnknownCoverageKindIsRefusedNotDefaulted)
{
    EngineCoverageTable table;
    std::string error;
    EXPECT_FALSE(parseEngineCoverage(
        nlohmann::json::parse(R"({"engines": {"x": {"coverage": "packs", "reason": "r"}}})"),
        table,
        error));
    EXPECT_NE(error.find("packs"), std::string::npos) << error;
}

TEST(TestEngineCoverage, ADocumentWithNoEnginesObjectIsRefused)
{
    // A misspelled top-level key must not read as an empty table, which would silently turn
    // every pack-only engine back into a searched one.
    EngineCoverageTable table;
    std::string error;
    EXPECT_FALSE(parseEngineCoverage(nlohmann::json::parse(R"({"engine": {}})"), table, error));
    EXPECT_NE(error.find("engines"), std::string::npos) << error;
}

TEST(TestEngineCoverage, AnUnlistedEngineIsAbsentFromTheTable)
{
    // CorpusGen keeps a default-constructed entry, which searches, for an engine the table
    // does not name.
    EngineCoverageTable table;
    std::string error;
    ASSERT_TRUE(parseEngineCoverage(
        nlohmann::json::parse(R"({"engines": {"baked": {"coverage": "pack", "reason": "r"}}})"),
        table,
        error))
        << error;
    EXPECT_EQ(table.count("ASM_SDPA_ENGINE"), 0U);
    EXPECT_EQ(EngineCoverageEntry{}.coverage, EngineCoverage::SEARCH);
}

} // namespace hipdnn_corpus_gen
