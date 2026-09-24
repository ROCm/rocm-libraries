// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/EngineCoverage.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <string>

namespace hipdnn_corpus_gen
{

TEST(TestEngineCoverage, TheShippedTableParsesAndRecordsTheDenseAttentionEngineAsPack)
{
    std::ifstream file(HIPDNN_CORPUS_GEN_OPERATIONS_DIR "/engines.json");
    ASSERT_TRUE(file.good()) << "engines.json is not beside the declarations";
    EngineCoverageTable table;
    std::string error;
    ASSERT_TRUE(parseEngineCoverage(nlohmann::json::parse(file), table, error)) << error;

    const auto dense = table.find("hipkernel:Gfx942AttentionDense");
    ASSERT_NE(dense, table.end());
    EXPECT_EQ(dense->second.coverage, EngineCoverage::PACK);
    EXPECT_FALSE(dense->second.reason.empty());
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

TEST(TestEngineCoverage, AnUnlistedEngineIsSearched)
{
    EngineCoverageTable table;
    std::string error;
    ASSERT_TRUE(parseEngineCoverage(nlohmann::json::parse(R"({"engines": {}})"), table, error));
    EXPECT_EQ(table.count("ASM_SDPA_ENGINE"), 0U);
    EXPECT_EQ(EngineCoverageEntry{}.coverage, EngineCoverage::SEARCH);
}

} // namespace hipdnn_corpus_gen
