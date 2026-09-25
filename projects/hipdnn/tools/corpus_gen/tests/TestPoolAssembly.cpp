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
    for(int64_t index = 0; index < 80; ++index)
    {
        pool.push_back(entryAt("model", index, "prefill_short_mha"));
    }
    for(int64_t index = 80; index < 100; ++index)
    {
        pool.push_back(entryAt("model", index, "decode_long_gqa"));
    }

    std::map<std::string, int64_t> allocation;
    const auto selected = select({{"model", pool}},
                                 20,
                                 {{"model", 1.0}, {"kernel", 0.0}, {"sweep", 0.0}},
                                 allocation);

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
    const auto unique = deduplicate(pools, dropped);

    EXPECT_EQ(dropped.at("model"), 0);
    EXPECT_EQ(dropped.at("kernel"), 1);
    EXPECT_EQ(dropped.at("sweep"), 1);
    EXPECT_EQ(unique.at("model").size(), 2u);
    EXPECT_EQ(unique.at("kernel").size(), 1u);
    EXPECT_EQ(unique.at("kernel").front().source, "kernel");
    EXPECT_EQ(unique.at("sweep").size(), 1u);
}

TEST(TestPoolAssembly, ASourceWithNoPoolIsNotAnError)
{
    // An operation whose declaration carries no kernel catalog simply has no kernel pool. That
    // is the whole mechanism by which coverage is "whatever has a declaration", so it must be a
    // quiet zero rather than a missing key that throws.
    std::map<std::string, int64_t> allocation;
    const auto selected
        = select({{"sweep", {entryAt("sweep", 1, "r"), entryAt("sweep", 2, "r")}}},
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
