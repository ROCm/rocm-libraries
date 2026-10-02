// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/PoolAssembly.hpp>

#include <gtest/gtest.h>

#include <map>
#include <set>
#include <string>
#include <vector>

/// @file TestPoolAssembly.cpp
/// @brief What a corpus is made of, and in what proportion.
///
/// Cases carried over from the retired Python assembler's suite.

using namespace hipdnn_corpus_gen;

namespace
{

/// A pool entry distinguished only by the fields deduplication and spreading see.
PoolEntry entryAt(const std::string& source, int64_t index, const std::string& regime)
{
    PoolEntry entry;
    entry.point["batch"] = index;
    entry.point["dtype"] = std::string("bf16");
    entry.source = source;
    entry.origin = source + ":" + std::to_string(index);
    entry.regime = regime;
    return entry;
}

std::map<std::string, int64_t> regimeCounts(const std::vector<PoolEntry>& entries)
{
    std::map<std::string, int64_t> counts;
    for(const auto& entry : entries)
    {
        ++counts[entry.regime];
    }
    return counts;
}

} // namespace

TEST(TestPoolAssembly, ASourceThatCannotFillItsShareHandsTheRestBack)
{
    // Otherwise the shortfall is invisible.
    const auto allocation
        = allocate(100, {{"model", 5}, {"kernel", 400}, {"sweep", 400}}, defaultShares());

    int64_t total = 0;
    for(const auto& entry : allocation)
    {
        total += entry.second;
    }
    EXPECT_EQ(total, 100);
    EXPECT_EQ(allocation.at("model"), 5);
}

TEST(TestPoolAssembly, EverySourceIsRepresentedEvenInASmallCorpus)
{
    // Not just packed geometries in a small corpus.
    const auto allocation
        = allocate(30, {{"model", 50}, {"kernel", 500}, {"sweep", 500}}, defaultShares());
    for(const auto& entry : allocation)
    {
        EXPECT_GT(entry.second, 0) << entry.first << " got nothing";
    }
}

TEST(TestPoolAssembly, AZeroShareExcludesItsSourceRatherThanDeferringIt)
{
    // Kernel-only means only the pack's geometries; the corpus is capped by the pack rather
    // than refilled from disabled sources.
    const auto allocation = allocate(5000,
                                     {{"model", 500}, {"kernel", 974}, {"sweep", 9000}},
                                     {{"model", 0.0}, {"kernel", 1.0}, {"sweep", 0.0}});

    EXPECT_EQ(allocation.at("model"), 0);
    EXPECT_EQ(allocation.at("kernel"), 974);
    EXPECT_EQ(allocation.at("sweep"), 0);
}

TEST(TestPoolAssembly, ATruncatedPoolKeepsItsRegimeMixRatherThanAnAlphabeticalPrefix)
{
    // Model shapes arrive grouped by file, so a prefix would be alphabetical. The cut keeps the
    // pool's proportions rather than one row per regime.
    std::vector<PoolEntry> pool;
    pool.reserve(100);
    for(int64_t index = 0; index < 80; ++index)
    {
        pool.push_back(entryAt("model", index, "prefill_short_mha"));
    }
    for(int64_t index = 80; index < 100; ++index)
    {
        pool.push_back(entryAt("model", index, "decode_long_gqa"));
    }

    std::map<std::string, int64_t> allocation;
    const auto selected = select(
        {{"model", pool}}, 20, {{"model", 1.0}, {"kernel", 0.0}, {"sweep", 0.0}}, allocation);

    EXPECT_EQ(allocation.at("model"), 20);
    const std::map<std::string, int64_t> expected{{"prefill_short_mha", 16},
                                                  {"decode_long_gqa", 4}};
    EXPECT_EQ(regimeCounts(selected), expected)
        << "a 4:1 pool truncated to a fifth is still 4:1, not 20 rows of whichever file "
           "sorted first";

    std::set<std::string> distinct;
    for(const auto& entry : selected)
    {
        distinct.insert(detail::describe(entry.point));
    }
    EXPECT_EQ(distinct.size(), 20u);
}

TEST(TestPoolAssembly, ASpreadPoolKeepsEachRegimesInternalOrder)
{
    // Order within a regime is its only ranking, so it must survive the spread.
    std::vector<PoolEntry> pool;
    pool.reserve(6);
    for(int64_t index = 0; index < 6; ++index)
    {
        pool.push_back(entryAt("model", index, index % 2 == 0 ? "even" : "odd"));
    }

    const auto spread = detail::spread(pool);
    std::vector<int64_t> evens;
    for(const auto& entry : spread)
    {
        if(entry.regime == "even")
        {
            evens.push_back(std::get<int64_t>(entry.point.at("batch")));
        }
    }
    const std::vector<int64_t> expected{0, 2, 4};
    EXPECT_EQ(evens, expected);
}

TEST(TestPoolAssembly, TheSameProblemFromTwoSourcesIsMeasuredOnce)
{
    // Otherwise one graph is benchmarked under two names. The earlier source wins.
    SourcePools pools;
    pools["model"] = {entryAt("model", 1, "r"), entryAt("model", 2, "r")};
    pools["kernel"] = {entryAt("kernel", 2, "r"), entryAt("kernel", 3, "r")};
    pools["sweep"] = {entryAt("sweep", 1, "r"), entryAt("sweep", 4, "r")};

    std::map<std::string, int64_t> dropped;
    const auto unique = deduplicate(pools, defaultShares(), dropped);

    EXPECT_EQ(dropped.at("model"), 0);
    EXPECT_EQ(dropped.at("kernel"), 1);
    EXPECT_EQ(dropped.at("sweep"), 1);
    EXPECT_EQ(unique.at("model").size(), 2u);
    EXPECT_EQ(unique.at("kernel").size(), 1u);
    EXPECT_EQ(unique.at("kernel").front().source, "kernel");
    EXPECT_EQ(unique.at("sweep").size(), 1u);
}

TEST(TestPoolAssembly, ADisabledSourceCannotTakeAPointFromAnEnabledOne)
{
    // The model pool names one of the pack's two geometries; with share 0 it must not keep it.
    const std::map<std::string, double> kernelOnly{{"model", 0.0}, {"kernel", 1.0}, {"sweep", 0.0}};
    SourcePools pools;
    pools["model"] = {entryAt("model", 2, "r")};
    pools["kernel"] = {entryAt("kernel", 2, "r"), entryAt("kernel", 3, "r")};
    pools["sweep"] = {entryAt("sweep", 3, "r")};

    std::map<std::string, int64_t> dropped;
    const auto unique = deduplicate(pools, kernelOnly, dropped);
    std::map<std::string, int64_t> allocation;
    const auto selected = select(unique, 2, kernelOnly, allocation);

    ASSERT_EQ(selected.size(), 2U);
    std::set<int64_t> batches;
    for(const auto& entry : selected)
    {
        EXPECT_EQ(entry.source, "kernel");
        batches.insert(std::get<int64_t>(entry.point.at("batch")));
    }
    EXPECT_EQ(batches, (std::set<int64_t>{2, 3}));
    EXPECT_EQ(dropped.at("kernel"), 0) << "nothing enabled duplicates a kernel geometry";
}

TEST(TestPoolAssembly, ASourceWithNoPoolIsNotAnError)
{
    // An operation with no kernel catalog simply has no kernel pool.
    std::map<std::string, int64_t> allocation;
    const auto selected = select({{"sweep", {entryAt("sweep", 1, "r"), entryAt("sweep", 2, "r")}}},
                                 2,
                                 defaultShares(),
                                 allocation);

    EXPECT_EQ(allocation.at("model"), 0);
    EXPECT_EQ(allocation.at("kernel"), 0);
    EXPECT_EQ(selected.size(), 2u);
}

TEST(TestPoolAssembly, ACutSweepPoolKeepsEveryCategoricalCombination)
{
    // A search arrives one combination at a time, so a prefix would drop whole dtypes.
    std::vector<PoolEntry> pool;
    for(int64_t index = 0; index < 60; ++index)
    {
        auto entry = entryAt("sweep", index, "");
        entry.stratum = index < 30 ? "dtype=fp32,|" : "dtype=bf16,|";
        pool.push_back(entry);
    }

    std::map<std::string, int64_t> allocation;
    const auto selected = select({{"sweep", pool}}, 20, defaultShares(), allocation);

    std::map<std::string, int64_t> perStratum;
    for(const auto& entry : selected)
    {
        ++perStratum[entry.stratum];
    }
    const std::map<std::string, int64_t> expected{{"dtype=bf16,|", 10}, {"dtype=fp32,|", 10}};
    EXPECT_EQ(perStratum, expected);
}

TEST(TestPoolAssembly, ARegimeQuotaIsFilledBeforeTheProportionalCut)
{
    // A proportional cut of 20 would take 2 of the rare regime; the quota of 6 is honoured first.
    std::vector<PoolEntry> pool;
    pool.reserve(100);
    for(int64_t index = 0; index < 100; ++index)
    {
        pool.push_back(entryAt("sweep", index, index < 90 ? "common" : "rare"));
    }

    std::map<std::string, int64_t> allocation;
    std::map<std::string, RegimeQuotaOutcome> outcome;
    const auto selected
        = select({{"sweep", pool}}, 20, defaultShares(), allocation, {{"rare", 6}}, outcome);

    const std::map<std::string, int64_t> expected{{"common", 14}, {"rare", 6}};
    EXPECT_EQ(regimeCounts(selected), expected);
    EXPECT_EQ(outcome.at("rare").asked, 6);
    EXPECT_EQ(outcome.at("rare").taken, 6);
    EXPECT_EQ(allocation.at("sweep"), 20);
}

TEST(TestPoolAssembly, ACountBelowTheQuotasDoesNotTrimThem)
{
    std::vector<PoolEntry> pool;
    pool.reserve(40);
    for(int64_t index = 0; index < 40; ++index)
    {
        pool.push_back(entryAt("sweep", index, index % 2 == 0 ? "a" : "b"));
    }

    std::map<std::string, int64_t> allocation;
    std::map<std::string, RegimeQuotaOutcome> outcome;
    const auto selected
        = select({{"sweep", pool}}, 5, defaultShares(), allocation, {{"a", 8}, {"b", 7}}, outcome);

    const std::map<std::string, int64_t> expected{{"a", 8}, {"b", 7}};
    EXPECT_EQ(regimeCounts(selected), expected);
}

TEST(TestPoolAssembly, AQuotaThePoolsCannotFillSaysHowShortItIs)
{
    // Reported, never padded from another regime.
    std::map<std::string, int64_t> allocation;
    std::map<std::string, RegimeQuotaOutcome> outcome;
    const auto selected
        = select({{"sweep", {entryAt("sweep", 1, "rare"), entryAt("sweep", 2, "x")}}},
                 0,
                 defaultShares(),
                 allocation,
                 {{"rare", 5}},
                 outcome);

    EXPECT_EQ(outcome.at("rare").taken, 1);
    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected.front().regime, "rare");
}

TEST(TestPoolAssembly, AQuotaTakesARecordedShapeBeforeASample)
{
    // A recorded model shape takes precedence, as elsewhere.
    SourcePools pools;
    pools["model"] = {entryAt("model", 1, "rare")};
    pools["sweep"] = {entryAt("sweep", 2, "rare"), entryAt("sweep", 3, "rare")};

    std::map<std::string, int64_t> allocation;
    std::map<std::string, RegimeQuotaOutcome> outcome;
    const auto selected = select(pools, 0, defaultShares(), allocation, {{"rare", 2}}, outcome);

    ASSERT_EQ(selected.size(), 2u);
    EXPECT_EQ(selected[0].source, "model");
    EXPECT_EQ(selected[1].source, "sweep");
    EXPECT_EQ(allocation.at("model"), 1);
    EXPECT_EQ(allocation.at("sweep"), 1);
}
