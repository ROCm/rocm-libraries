// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/EngineCoverage.hpp>

#include <nlohmann/json.hpp>

#include "OperationsDir.hpp"
#include <fstream>
#include <string>

namespace hipdnn_corpus_gen
{

TEST(TestEngineCoverage, TheShippedTableParses)
{
    // The contents are data; only that CorpusGen can parse it is asserted.
    std::ifstream file(hipdnn_corpus_gen::test::operationsDir() + "/engines.json");
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
    // An entry disables search, so it must say why.
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
    // A misspelled top-level key must not read as an empty table.
    EngineCoverageTable table;
    std::string error;
    EXPECT_FALSE(parseEngineCoverage(nlohmann::json::parse(R"({"engine": {}})"), table, error));
    EXPECT_NE(error.find("engines"), std::string::npos) << error;
}

TEST(TestEngineCoverage, AnUnlistedEngineIsAbsentFromTheTable)
{
    // CorpusGen then uses a default entry, which searches.
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
