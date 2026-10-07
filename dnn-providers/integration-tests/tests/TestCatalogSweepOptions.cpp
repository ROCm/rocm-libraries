// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The command-line rules of --sweep-knob-catalog: what a shard owns, what a filter
// selects, and the environments the sweep refuses to start in.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "harness/catalog-sweep/CatalogSweepOptions.hpp"

using namespace hipdnn_integration_tests::catalog_sweep;
using hipdnn_test_sdk::utilities::claimScratchDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

EnvLookup envOf(std::map<std::string, std::string> values)
{
    return [values = std::move(values)](const std::string& name) -> std::optional<std::string> {
        const auto it = values.find(name);
        if(it == values.end())
        {
            return std::nullopt;
        }
        return it->second;
    };
}

std::vector<std::string> sampleGraphs()
{
    std::vector<std::string> names;
    names.reserve(200);
    for(int i = 0; i < 200; ++i)
    {
        names.push_back("quick_Pointwise_nchw_case" + std::to_string(i) + ".Small");
    }
    return names;
}

} // namespace

// ---- --catalog-shard ---------------------------------------------------------------

TEST(TestCatalogSweepShard, ParsesAZeroBasedIndexAndCount)
{
    const auto shard = parseShard("2/5");
    EXPECT_EQ(shard.index, 2U);
    EXPECT_EQ(shard.count, 5U);
}

TEST(TestCatalogSweepShard, RejectsMalformedAndOutOfRangeSpecs)
{
    for(const char* bad :
        {"", "3", "/4", "1/", "a/b", "-1/4", "1/-4", "1/0", "4/4", "5/4", "1/4/2", " 1/4", "1.5/4"})
    {
        EXPECT_THROW(parseShard(bad), std::invalid_argument) << "'" << bad << "' was accepted";
    }
}

TEST(TestCatalogSweepShard, EveryGraphBelongsToExactlyOneShard)
{
    constexpr std::size_t K_COUNT = 7;
    for(const auto& graph : sampleGraphs())
    {
        std::size_t owners = 0;
        for(std::size_t index = 0; index < K_COUNT; ++index)
        {
            owners += shardOwns({index, K_COUNT}, graph) ? 1U : 0U;
        }
        EXPECT_EQ(owners, 1U) << graph;
    }
}

TEST(TestCatalogSweepShard, ASingleShardOwnsEverything)
{
    for(const auto& graph : sampleGraphs())
    {
        EXPECT_TRUE(shardOwns({0, 1}, graph)) << graph;
    }
}

// Shards run on different hosts and at different times must agree on who owns a graph,
// so the assignment cannot depend on anything but the name. These are FNV-1a 64 values:
// a change here silently moves graphs between shards of a sweep already in flight.
TEST(TestCatalogSweepShard, OwnershipIsAFixedFunctionOfTheGraphName)
{
    // fnv1a("a") = 0xaf63dc4c8601ec8c
    EXPECT_TRUE(shardOwns({0xaf63dc4c8601ec8cULL % 4, 4}, "a"));
    // fnv1a("") = 0xcbf29ce484222325 (the offset basis)
    EXPECT_TRUE(shardOwns({0xcbf29ce484222325ULL % 4, 4}, ""));
}

TEST(TestCatalogSweepShard, SplitsManyGraphsAcrossEveryShard)
{
    constexpr std::size_t K_COUNT = 4;
    std::set<std::size_t> populated;
    for(const auto& graph : sampleGraphs())
    {
        for(std::size_t index = 0; index < K_COUNT; ++index)
        {
            if(shardOwns({index, K_COUNT}, graph))
            {
                populated.insert(index);
            }
        }
    }
    EXPECT_EQ(populated.size(), K_COUNT) << "a shard received no graphs out of 200";
}

// ---- --catalog-filter --------------------------------------------------------------

TEST(TestCatalogSweepFilter, AnUnfilteredRunMatchesEverything)
{
    const CatalogFilter filter;
    EXPECT_TRUE(filter.isUnfiltered());
    EXPECT_TRUE(filter.matches("any.graph", "ANY_ENGINE"));
    EXPECT_TRUE(filter.matchesGraph("any.graph"));
}

TEST(TestCatalogSweepFilter, GlobMatchesTheWholeNameOnly)
{
    const auto filter = CatalogFilter::parse("engine=ROCKE_*");
    EXPECT_TRUE(filter.matches("g", "ROCKE_POINTWISE"));
    EXPECT_FALSE(filter.matches("g", "MY_ROCKE_POINTWISE")) << "a glob is anchored at the start";
    EXPECT_FALSE(filter.matches("g", "ROCKE")) << "'ROCKE_*' needs the underscore";

    const auto exact = CatalogFilter::parse("engine=MIOPEN_ENGINE");
    EXPECT_TRUE(exact.matches("g", "MIOPEN_ENGINE"));
    EXPECT_FALSE(exact.matches("g", "MIOPEN_ENGINE2"));
}

TEST(TestCatalogSweepFilter, QuestionMarkIsExactlyOneCharacter)
{
    const auto filter = CatalogFilter::parse("engine=E?");
    EXPECT_TRUE(filter.matches("g", "E1"));
    EXPECT_FALSE(filter.matches("g", "E"));
    EXPECT_FALSE(filter.matches("g", "E12"));
}

TEST(TestCatalogSweepFilter, SubstringMatchesAnywhere)
{
    const auto filter = CatalogFilter::parse("graph~Pointwise");
    EXPECT_TRUE(filter.matches("quick_Pointwise_nchw.Small", "E"));
    EXPECT_FALSE(filter.matches("quick_Matmul_nchw.Small", "E"));
}

TEST(TestCatalogSweepFilter, NegationInvertsTheClause)
{
    const auto notGlob = CatalogFilter::parse("engine!=MIOPEN*");
    EXPECT_FALSE(notGlob.matches("g", "MIOPEN_ENGINE"));
    EXPECT_TRUE(notGlob.matches("g", "ROCKE"));

    const auto notSubstring = CatalogFilter::parse("graph!~Matmul");
    EXPECT_FALSE(notSubstring.matches("quick_Matmul.Small", "E"));
    EXPECT_TRUE(notSubstring.matches("quick_Pointwise.Small", "E"));
}

TEST(TestCatalogSweepFilter, AlternativesAreOrAndClausesAreAnd)
{
    const auto filter = CatalogFilter::parse("engine=A|B , graph~Point");
    EXPECT_TRUE(filter.matches("Pointwise.X", "A"));
    EXPECT_TRUE(filter.matches("Pointwise.X", "B"));
    EXPECT_FALSE(filter.matches("Pointwise.X", "C")) << "no alternative matched";
    EXPECT_FALSE(filter.matches("Matmul.X", "A")) << "the graph clause failed";
}

TEST(TestCatalogSweepFilter, MatchesGraphIgnoresEngineClauses)
{
    const auto filter = CatalogFilter::parse("graph~Point,engine=ROCKE");
    EXPECT_TRUE(filter.matchesGraph("Pointwise.X"));
    EXPECT_FALSE(filter.matchesGraph("Matmul.X"));
}

TEST(TestCatalogSweepFilter, RejectsMalformedExpressionsNamingTheProblem)
{
    for(const char* bad : {"",
                           "  ",
                           "engine",
                           "=x",
                           "engine=",
                           "engine=a|",
                           "engine=a,",
                           ",engine=a",
                           "device=gfx1151",
                           "engine?x",
                           "engine!x",
                           "graph=a,,engine=b"})
    {
        EXPECT_THROW(CatalogFilter::parse(bad), std::invalid_argument)
            << "'" << bad << "' was accepted";
    }

    try
    {
        CatalogFilter::parse("device=gfx1151");
        FAIL() << "unknown field accepted";
    }
    catch(const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("device"), std::string::npos) << e.what();
    }
}

// ---- flag resolution ---------------------------------------------------------------

TEST(TestCatalogSweepOptions, NoSweepFlagsMeansNoSweep)
{
    const auto resolution = resolveCatalogSweepOptions({});
    EXPECT_FALSE(resolution.options.has_value());
    EXPECT_FALSE(resolution.error.has_value());
}

TEST(TestCatalogSweepOptions, ASweepWithoutAReportIsRefused)
{
    CatalogSweepRequest request;
    request.sweep = true;
    const auto resolution = resolveCatalogSweepOptions(request);
    ASSERT_TRUE(resolution.error.has_value());
    EXPECT_NE(resolution.error->find("--catalog-report"), std::string::npos) << *resolution.error;
    EXPECT_FALSE(resolution.options.has_value());
}

TEST(TestCatalogSweepOptions, CatalogFlagsWithoutASweepAreRefusedNotIgnored)
{
    for(int which = 0; which < 3; ++which)
    {
        CatalogSweepRequest request;
        if(which == 0)
        {
            request.reportPath = "1/2";
        }
        else if(which == 1)
        {
            request.filter = "1/2";
        }
        else
        {
            request.shard = "1/2";
        }
        const auto resolution = resolveCatalogSweepOptions(request);
        ASSERT_TRUE(resolution.error.has_value()) << which;
        EXPECT_NE(resolution.error->find("--sweep-knob-catalog"), std::string::npos);
    }
}

TEST(TestCatalogSweepOptions, ABadFilterOrShardSurfacesAsAnErrorNotAnException)
{
    CatalogSweepRequest request;
    request.sweep = true;
    request.reportPath = "report.jsonl";

    request.shard = "4/4";
    EXPECT_TRUE(resolveCatalogSweepOptions(request).error.has_value());

    request.shard = "0/4";
    request.filter = "nonsense";
    EXPECT_TRUE(resolveCatalogSweepOptions(request).error.has_value());
}

TEST(TestCatalogSweepOptions, ValidFlagsResolveToOptions)
{
    CatalogSweepRequest request;
    request.sweep = true;
    request.reportPath = "report.jsonl";
    request.filter = "engine=ROCKE*";
    request.shard = "1/3";

    const auto resolution = resolveCatalogSweepOptions(request);
    ASSERT_TRUE(resolution.options.has_value()) << resolution.error.value_or("");
    EXPECT_EQ(resolution.options->reportPath, std::filesystem::path("report.jsonl"));
    EXPECT_EQ(resolution.options->shard.index, 1U);
    EXPECT_EQ(resolution.options->shard.count, 3U);
    EXPECT_TRUE(resolution.options->filter.matches("g", "ROCKE_X"));
    EXPECT_FALSE(resolution.options->filter.matches("g", "MIOPEN"));
}

// ---- refusal to start ----------------------------------------------------------------

TEST(TestCatalogSweepEnvironment, ForceBenchmarkingSetToAnythingRefuses)
{
    const auto fresh = claimScratchDirectory("catalog_env");
    for(const char* value : {"1", "0", "true", ""})
    {
        const auto refusal = checkSweepEnvironment(envOf(
            {{"HIPDNN_FORCE_BENCHMARKING", value}, {"HIPDNN_CACHE_DIR", fresh.path().string()}}));
        ASSERT_TRUE(refusal.has_value()) << "value '" << value << "' did not refuse";
        EXPECT_NE(refusal->find("HIPDNN_FORCE_BENCHMARKING"), std::string::npos) << *refusal;
    }
}

TEST(TestCatalogSweepEnvironment, ACacheDirectoryIsRequired)
{
    for(const auto& env : {envOf({}), envOf({{"HIPDNN_CACHE_DIR", ""}})})
    {
        const auto refusal = checkSweepEnvironment(env);
        ASSERT_TRUE(refusal.has_value());
        EXPECT_NE(refusal->find("HIPDNN_CACHE_DIR"), std::string::npos) << *refusal;
    }
}

TEST(TestCatalogSweepEnvironment, ADirectoryThatDoesNotExistYetIsFresh)
{
    const auto scratch = claimScratchDirectory("catalog_env");
    const auto absent = scratch.path() / "never_created";
    EXPECT_FALSE(checkSweepEnvironment(envOf({{"HIPDNN_CACHE_DIR", absent.string()}})).has_value());
}

TEST(TestCatalogSweepEnvironment, AnEmptyDirectoryIsFresh)
{
    const auto scratch = claimScratchDirectory("catalog_env");
    EXPECT_FALSE(
        checkSweepEnvironment(envOf({{"HIPDNN_CACHE_DIR", scratch.path().string()}})).has_value());
}

TEST(TestCatalogSweepEnvironment, ADirectoryWithContentIsNotFresh)
{
    const auto scratch = claimScratchDirectory("catalog_env");
    std::ofstream(scratch.path() / "ingestor-winners.jsonl") << "{}\n";

    const auto refusal
        = checkSweepEnvironment(envOf({{"HIPDNN_CACHE_DIR", scratch.path().string()}}));
    ASSERT_TRUE(refusal.has_value());
    EXPECT_NE(refusal->find("not empty"), std::string::npos) << *refusal;
}

TEST(TestCatalogSweepEnvironment, APathThatIsAFileIsNotAFreshDirectory)
{
    const auto scratch = claimScratchDirectory("catalog_env");
    const auto file = scratch.path() / "cache";
    std::ofstream(file) << "x";

    const auto refusal = checkSweepEnvironment(envOf({{"HIPDNN_CACHE_DIR", file.string()}}));
    ASSERT_TRUE(refusal.has_value());
    EXPECT_NE(refusal->find("not a directory"), std::string::npos) << *refusal;
}

// NOLINTEND(readability-identifier-naming)
