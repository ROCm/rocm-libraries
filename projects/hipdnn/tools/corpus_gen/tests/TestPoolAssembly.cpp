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
/// These are the cases the retired Python assembler's own suite pinned, carried over so the
/// behaviour survives the port rather than being re-derived with it deleted. Each
/// one names a corpus that was actually built wrong.

using namespace hipdnn_corpus_gen;

namespace
{

/// A pool entry distinguished only by the fields under test. The point is what deduplication
/// and spreading see, so it is what varies.
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
    // Otherwise a corpus asked for 100 problems returns however many the smallest source had,
    // and the shortfall is invisible.
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
    // A corpus of 30 that is 30 packed geometries has none of the problems the heuristic exists
    // to get right in it.
    const auto allocation
        = allocate(30, {{"model", 50}, {"kernel", 500}, {"sweep", 500}}, defaultShares());
    for(const auto& entry : allocation)
    {
        EXPECT_GT(entry.second, 0) << entry.first << " got nothing";
    }
}

TEST(TestPoolAssembly, AZeroShareExcludesItsSourceRatherThanDeferringIt)
{
    // An engine whose kernels are compiled per exact shape wants ONLY its pack's geometries;
    // every other problem is a decline it pays a measurement for. Redistribution used to refill
    // the sources the caller had just switched off: asking for kernel-only against a
    // 974-geometry pack returned 974 kernel problems and 4026 from `model` and `sweep`. The
    // corpus is capped by the pack, and that is the honest answer.
    const auto allocation = allocate(5000,
                                     {{"model", 500}, {"kernel", 974}, {"sweep", 9000}},
                                     {{"model", 0.0}, {"kernel", 1.0}, {"sweep", 0.0}});

    EXPECT_EQ(allocation.at("model"), 0);
    EXPECT_EQ(allocation.at("kernel"), 974);
    EXPECT_EQ(allocation.at("sweep"), 0);
}

TEST(TestPoolAssembly, ATruncatedPoolKeepsItsRegimeMixRatherThanAnAlphabeticalPrefix)
{
    // The model pool arrives grouped: shape files are read in name order, so each file's rows
    // are contiguous. Taking the front of that is taking the alphabet -- against a published
    // shape directory it dropped 95 of 200 shapes, and any regime written down in a late-named
    // file left the corpus entirely.
    //
    // Proportional, not one row of each: the pool's own mix is what real models run, so a fifth
    // of the pool should look like the pool.
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
    // The spread reorders across regimes and must not reorder within one: a pool's own order
    // inside a regime is the only ranking it carries, and shuffling it would silently pick
    // different members whenever the allocation changed.
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
    // Two entries differing only in provenance are one problem measured twice: the same graph
    // benchmarked under two names, which inflates a corpus and biases whichever regime it lands
    // in. The earlier source wins, so the manifest records the provenance worth auditing.
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
    // Kernel-only, against a pack of two geometries, with a model pool that happens to name
    // one of them. The model pool won that geometry in deduplication and was then given no
    // share, so the corpus came back with one problem of the two the pack holds.
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
    // An operation whose declaration carries no kernel catalog simply has no kernel pool. That
    // is the whole mechanism by which coverage is "whatever has a declaration", so it must be a
    // quiet zero rather than a missing key that throws.
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
    // A search arrives combination by combination -- every fp32 problem, then every bf16 one.
    // Taking a prefix of that dropped bf16 from a pooling corpus entirely, though the engine
    // served it. Spread over the stratum, a cut keeps each combination's share.
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
    // Ninety of one population and ten of another: a proportional cut of twenty takes two of
    // the rare one, which is the lopsided corpus a quota exists to correct. Owed six, it gets
    // six, and the rest of the count is cut from what is left, as before.
    std::vector<PoolEntry> pool;
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
    // The quotas are what the caller measured it needs; a count set with them in mind must not
    // quietly undo them.
    std::vector<PoolEntry> pool;
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
    // Reported, never padded from another regime: a short quota is a finding about what the
    // engine serves, and filling it with something else would hide it.
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
    // Same precedence as everything else: of two problems in the owed regime, the recorded
    // model shape is the one worth measuring.
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
