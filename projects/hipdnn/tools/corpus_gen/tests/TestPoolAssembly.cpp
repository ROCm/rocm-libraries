// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/OperationMetadata.hpp>
#include <hipdnn_corpus_gen/PoolAssembly.hpp>

#include <gtest/gtest.h>

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

/// @file TestPoolAssembly.cpp
/// @brief What a corpus is made of, and in what proportion.

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

/// What one call of `select` returned and reported.
struct Selection
{
    std::vector<PoolEntry> entries;
    std::map<std::string, int64_t> allocation;
    std::map<std::string, RegimeQuotaOutcome> outcome;
};

Selection selectWith(const SourcePools& pools,
                     int64_t count,
                     const std::map<std::string, int64_t>& quotas,
                     const std::set<std::string>& reserved = {},
                     const std::map<std::string, double>& shares = defaultShares())
{
    Selection result;
    result.entries
        = select(pools, count, shares, result.allocation, quotas, result.outcome, reserved);
    return result;
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

TEST(TestPoolAssembly, TwoPointsThatAreOneGraphAreOneProblem)
{
    // Points differing only in a parameter the graph does not carry build the same graph;
    // keyed by graph, the corpus holds it once.
    SourcePools pools;
    pools["sweep"] = {entryAt("sweep", 1, "r"), entryAt("sweep", 2, "r"), entryAt("sweep", 3, "r")};
    const auto sameGraph = [](const PoolEntry& entry) {
        return std::get<int64_t>(entry.point.at("batch")) == 3 ? std::string("other")
                                                               : std::string("one graph");
    };

    std::map<std::string, int64_t> dropped;
    const auto unique = deduplicate(pools, defaultShares(), dropped, sameGraph);

    ASSERT_EQ(unique.at("sweep").size(), 2U);
    EXPECT_EQ(std::get<int64_t>(unique.at("sweep").front().point.at("batch")), 1);
    EXPECT_EQ(dropped.at("sweep"), 1);
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

TEST(TestPoolAssembly, ACutKeepsTheProportionOfStrataThatWeightedCombinationsGave)
{
    // The search hands the cut pools whose combinations are sized 2:1:1 by their declared
    // weights; the cut must pass that proportion on, not equalise the strata.
    std::vector<PoolEntry> pool;
    int64_t index = 0;
    for(const auto& [stratum, size] : std::vector<std::pair<std::string, int64_t>>{
            {"layout=a,|", 120}, {"layout=b,|", 60}, {"layout=c,|", 60}})
    {
        for(int64_t i = 0; i < size; ++i)
        {
            auto entry = entryAt("sweep", index++, "");
            entry.stratum = stratum;
            pool.push_back(entry);
        }
    }

    std::map<std::string, int64_t> allocation;
    const auto selected = select({{"sweep", pool}}, 60, defaultShares(), allocation);

    std::map<std::string, int64_t> perStratum;
    for(const auto& entry : selected)
    {
        ++perStratum[entry.stratum];
    }
    const std::map<std::string, int64_t> expected{
        {"layout=a,|", 30}, {"layout=b,|", 15}, {"layout=c,|", 15}};
    EXPECT_EQ(perStratum, expected);
}

namespace
{

/// @p count entries of one group and weight, in the strata `<group>|r0`, `<group>|r1`.
void addGroup(std::vector<PoolEntry>& pool, const std::string& group, double weight, int64_t count)
{
    for(int64_t i = 0; i < count; ++i)
    {
        auto entry = entryAt("sweep", static_cast<int64_t>(pool.size()), "");
        entry.group = group;
        entry.weight = weight;
        entry.stratum = group + "|r" + std::to_string(i % 2);
        pool.push_back(entry);
    }
}

std::map<std::string, int64_t> groupCounts(const std::vector<PoolEntry>& entries)
{
    std::map<std::string, int64_t> counts;
    for(const auto& entry : entries)
    {
        ++counts[entry.group];
    }
    return counts;
}

} // namespace

TEST(TestPoolAssembly, ACutHoldsEachWeightedGroupAtItsWeightNotAtWhatThePoolHoldsOfIt)
{
    // The pool holds five times as much of group a as of b, and a cut of 40 takes 20 of each
    // when they weigh the same, and 30 and 10 when a weighs three times b, whatever the pool
    // holds.
    std::vector<PoolEntry> pool;
    addGroup(pool, "a", 1.0, 200);
    addGroup(pool, "b", 1.0, 40);
    std::map<std::string, int64_t> allocation;
    auto selected = select({{"sweep", pool}}, 40, defaultShares(), allocation);
    EXPECT_EQ(groupCounts(selected), (std::map<std::string, int64_t>{{"a", 20}, {"b", 20}}));

    for(auto& entry : pool)
    {
        entry.weight = entry.group == "a" ? 3.0 : 1.0;
    }
    selected = select({{"sweep", pool}}, 40, defaultShares(), allocation);
    EXPECT_EQ(groupCounts(selected), (std::map<std::string, int64_t>{{"a", 30}, {"b", 10}}));
}

TEST(TestPoolAssembly, AWeightedGroupThatRunsOutHandsItsShareToTheOthers)
{
    std::vector<PoolEntry> pool;
    addGroup(pool, "a", 1.0, 200);
    addGroup(pool, "b", 1.0, 6);
    std::map<std::string, int64_t> allocation;
    const auto selected = select({{"sweep", pool}}, 40, defaultShares(), allocation);
    EXPECT_EQ(groupCounts(selected), (std::map<std::string, int64_t>{{"a", 34}, {"b", 6}}));
}

TEST(TestPoolAssembly, ACutTooShortForEveryWeightedGroupToHaveAMemberStillSpreadsThemByWeight)
{
    // 20 groups weigh 2.5 and 20 weigh 1; a cut of 15 expects about 4 members from the light
    // ones. Rounding each group to nearest would give every one of the 15 to the heavy groups,
    // the lighter combinations never getting a turn.
    std::vector<PoolEntry> pool;
    for(int64_t group = 0; group < 40; ++group)
    {
        addGroup(pool, "g" + std::to_string(group), group < 20 ? 2.5 : 1.0, 10);
    }
    std::map<std::string, int64_t> allocation;
    const auto selected = select({{"sweep", pool}}, 15, defaultShares(), allocation);

    int64_t light = 0;
    for(const auto& entry : selected)
    {
        light += entry.weight < 2.0 ? 1 : 0;
    }
    EXPECT_GE(light, 2);
    EXPECT_LE(light, 7);
}

TEST(TestPoolAssembly, WithinAWeightedGroupTheStrataKeepTheirMembersMix)
{
    std::vector<PoolEntry> pool;
    addGroup(pool, "a", 1.0, 40);
    for(int64_t i = 0; i < 10; ++i)
    {
        // A third stratum holding a quarter of group a.
        auto entry = entryAt("sweep", 100 + i, "");
        entry.group = "a";
        entry.weight = 1.0;
        entry.stratum = "a|rare";
        pool.push_back(entry);
    }
    std::map<std::string, int64_t> allocation;
    const auto selected = select({{"sweep", pool}}, 20, defaultShares(), allocation);

    std::map<std::string, int64_t> perStratum;
    for(const auto& entry : selected)
    {
        ++perStratum[entry.stratum];
    }
    EXPECT_EQ(perStratum,
              (std::map<std::string, int64_t>{{"a|r0", 8}, {"a|r1", 8}, {"a|rare", 4}}));
}

TEST(TestPoolAssembly, TheGroupsOfAGeneratedPoolWeighTheirCombinationsAtTheDeclaredShares)
{
    // `layout` is weighted 3:1 and `form` is not; form z is declared only with layout a, so a
    // plain product of weights would leave it a smaller share than x and y.
    const auto load = parseOperationMetadata(nlohmann::json::parse(R"({
      "schema_version": "1.0",
      "operation": "toy",
      "parameters": {
        "M":      { "type": "int64" },
        "layout": { "type": "enum", "values": ["a", "b"], "weights": {"a": 3, "b": 1} },
        "form":   { "type": "enum", "values": ["x", "y", "z"] }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "createValidMatmulGraph", "source": "x.hpp",
                         "arguments": [] },
      "constraints": [ { "or": [ { "==": ["$q.form", "x"] }, { "==": ["$q.form", "y"] },
                                 { "==": ["$q.layout", "a"] } ] } ]
    })"));
    ASSERT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());

    std::vector<PoolEntry> pool;
    for(const auto* layout : {"a", "b"})
    {
        for(const auto* form : {"x", "y", "z"})
        {
            if(std::string(layout) == "b" && std::string(form) == "z")
            {
                continue;
            }
            auto entry = entryAt("sweep", static_cast<int64_t>(pool.size()), "");
            entry.point["layout"] = std::string(layout);
            entry.point["form"] = std::string(form);
            pool.push_back(entry);
        }
    }
    const auto weighing = weighGroups(pool, *load.metadata);
    EXPECT_TRUE(weighing.weighed);
    EXPECT_TRUE(weighing.uneven.empty());

    std::map<std::string, double> perValue;
    double total = 0.0;
    for(const auto& entry : pool)
    {
        EXPECT_FALSE(entry.group.empty());
        perValue["layout=" + std::get<std::string>(entry.point.at("layout"))] += entry.weight;
        perValue["form=" + std::get<std::string>(entry.point.at("form"))] += entry.weight;
        total += entry.weight;
    }
    EXPECT_NEAR(perValue.at("layout=a") / total, 0.75, 1e-6);
    EXPECT_NEAR(perValue.at("layout=b") / total, 0.25, 1e-6);
    for(const auto* form : {"x", "y", "z"})
    {
        EXPECT_NEAR(perValue.at(std::string("form=") + form) / total, 1.0 / 3.0, 1e-6) << form;
    }
}

namespace
{

/// A toy declaration with `fusion` and `mode` tied by a constraint (no fusion only with relu),
/// `layout` weighted 3:1 when @p weighted, and the pool of every combination it admits.
std::vector<PoolEntry> tiedPool(bool weighted, std::optional<OperationMetadata>& metadata)
{
    const std::string layout = weighted ? R"({ "type": "enum", "values": ["a", "b"],
                                               "weights": {"a": 3, "b": 1} })"
                                        : R"({ "type": "enum", "values": ["a", "b"] })";
    const auto load = parseOperationMetadata(nlohmann::json::parse(R"({
      "schema_version": "1.0",
      "operation": "toy",
      "parameters": {
        "M":      { "type": "int64" },
        "layout": )" + layout + R"(,
        "fusion": { "type": "enum", "values": ["none", "add"] },
        "mode":   { "type": "enum", "values": ["relu", "gelu"] }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "createValidMatmulGraph", "source": "x.hpp",
                         "arguments": [] },
      "constraints": [ { "or": [ { "==": ["$q.fusion", "add"] },
                                 { "==": ["$q.mode", "relu"] } ] } ]
    })"));
    EXPECT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());
    metadata = load.metadata;

    std::vector<PoolEntry> pool;
    for(const auto* layoutValue : {"a", "b"})
    {
        for(const auto* fusion : {"none", "add"})
        {
            for(const auto* mode : {"relu", "gelu"})
            {
                if(std::string(fusion) == "none" && std::string(mode) == "gelu")
                {
                    continue;
                }
                auto entry = entryAt("sweep", static_cast<int64_t>(pool.size()), "");
                entry.point["layout"] = std::string(layoutValue);
                entry.point["fusion"] = std::string(fusion);
                entry.point["mode"] = std::string(mode);
                pool.push_back(entry);
            }
        }
    }
    return pool;
}

} // namespace

TEST(TestPoolAssembly, AParameterWhoseEqualSharesWouldStarveACombinationIsLeftUneven)
{
    // Equal fusion and mode shares put all of `none` on relu, so relu could take no `add` row:
    // (add, relu) would weigh about 0 and drop out of a cut. One of the two is left as the
    // combinations carry it; the other still takes equal shares, and layout its weights.
    std::optional<OperationMetadata> metadata;
    auto pool = tiedPool(/*weighted=*/true, metadata);
    ASSERT_TRUE(metadata.has_value());

    const auto weighing = weighGroups(pool, *metadata);
    EXPECT_TRUE(weighing.weighed);
    ASSERT_EQ(weighing.uneven.size(), 1U);
    const std::string kept = weighing.uneven.front() == "fusion" ? "mode" : "fusion";
    EXPECT_TRUE(weighing.uneven.front() == "fusion" || weighing.uneven.front() == "mode");

    double total = 0.0;
    double layoutA = 0.0;
    std::map<std::string, double> keptShares;
    for(const auto& entry : pool)
    {
        total += entry.weight;
        layoutA += std::get<std::string>(entry.point.at("layout")) == "a" ? entry.weight : 0.0;
        keptShares[std::get<std::string>(entry.point.at(kept))] += entry.weight;
    }
    for(const auto& entry : pool)
    {
        EXPECT_GT(entry.weight, 0.25 * total / static_cast<double>(pool.size())) << entry.group;
    }
    EXPECT_NEAR(layoutA / total, 0.75, 1e-6);
    for(const auto& [value, share] : keptShares)
    {
        EXPECT_NEAR(share / total, 0.5, 1e-6) << kept << "=" << value;
    }
}

TEST(TestPoolAssembly, ADeclarationWithNoWeightsLeavesItsPoolCutByWhatItHolds)
{
    std::optional<OperationMetadata> metadata;
    auto pool = tiedPool(/*weighted=*/false, metadata);
    ASSERT_TRUE(metadata.has_value());

    EXPECT_FALSE(weighGroups(pool, *metadata).weighed);

    for(const auto& entry : pool)
    {
        EXPECT_TRUE(entry.group.empty());
        EXPECT_EQ(entry.weight, 0.0);
    }
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

    const auto result = selectWith({{"sweep", pool}}, 20, {{"rare", 6}});

    const std::map<std::string, int64_t> expected{{"common", 14}, {"rare", 6}};
    EXPECT_EQ(regimeCounts(result.entries), expected);
    EXPECT_EQ(result.outcome.at("rare").asked, 6);
    EXPECT_EQ(result.outcome.at("rare").taken, 6);
    EXPECT_EQ(result.allocation.at("sweep"), 20);
}

TEST(TestPoolAssembly, ACountBelowTheQuotasDoesNotTrimThem)
{
    std::vector<PoolEntry> pool;
    pool.reserve(40);
    for(int64_t index = 0; index < 40; ++index)
    {
        pool.push_back(entryAt("sweep", index, index % 2 == 0 ? "a" : "b"));
    }

    const auto result = selectWith({{"sweep", pool}}, 5, {{"a", 8}, {"b", 7}});

    const std::map<std::string, int64_t> expected{{"a", 8}, {"b", 7}};
    EXPECT_EQ(regimeCounts(result.entries), expected);
}

TEST(TestPoolAssembly, AQuotaThePoolsCannotFillSaysHowShortItIs)
{
    // Reported, never padded from another regime.
    const auto result = selectWith(
        {{"sweep", {entryAt("sweep", 1, "rare"), entryAt("sweep", 2, "x")}}}, 0, {{"rare", 5}});

    EXPECT_EQ(result.outcome.at("rare").taken, 1);
    ASSERT_EQ(result.entries.size(), 1u);
    EXPECT_EQ(result.entries.front().regime, "rare");
}

TEST(TestPoolAssembly, OneEntryFillsBothItsRegimesQuotaAndItsFormsQuota)
{
    // The pick for whichever quota is served first also counts for the other, so the corpus
    // holds one graph, not one per quota. The regime names sort before and after the form key,
    // so both orders in which the quotas are served are covered.
    for(const char* regime : {"alpha", "zulu"})
    {
        auto carrier = entryAt("sweep", 1, regime);
        carrier.forms = {"form:padded"};
        const auto sameRegime = entryAt("sweep", 2, regime);
        auto sameForm = entryAt("sweep", 3, "other");
        sameForm.forms = {"form:padded"};

        const auto result = selectWith(
            {{"sweep", {carrier, sameRegime, sameForm}}}, 0, {{regime, 1}, {"form:padded", 1}});

        ASSERT_EQ(result.entries.size(), 1u) << regime;
        EXPECT_EQ(result.entries.front().origin, carrier.origin) << regime;
        EXPECT_EQ(result.outcome.at(regime).taken, 1) << regime;
        EXPECT_EQ(result.outcome.at("form:padded").taken, 1) << regime;
    }
}

TEST(TestPoolAssembly, AFormQuotaNoEntryCarriesIsReportedShortRatherThanFailing)
{
    auto padded = entryAt("sweep", 1, "r");
    padded.forms = {"form:padded"};

    const auto result = selectWith(
        {{"sweep", {padded, entryAt("sweep", 2, "r")}}}, 0, {{"form:padded", 1}, {"form:sink", 1}});

    ASSERT_EQ(result.entries.size(), 1u);
    EXPECT_EQ(result.outcome.at("form:padded").taken, 1);
    EXPECT_EQ(result.outcome.at("form:sink").asked, 1);
    EXPECT_EQ(result.outcome.at("form:sink").taken, 0);
}

TEST(TestPoolAssembly, AQuotaTakesARecordedShapeBeforeASample)
{
    // A recorded model shape takes precedence, as elsewhere.
    SourcePools pools;
    pools["model"] = {entryAt("model", 1, "rare")};
    pools["sweep"] = {entryAt("sweep", 2, "rare"), entryAt("sweep", 3, "rare")};

    const auto result = selectWith(pools, 0, {{"rare", 2}});

    ASSERT_EQ(result.entries.size(), 2u);
    EXPECT_EQ(result.entries[0].source, "model");
    EXPECT_EQ(result.entries[1].source, "sweep");
    EXPECT_EQ(result.allocation.at("model"), 1);
    EXPECT_EQ(result.allocation.at("sweep"), 1);
}

TEST(TestPoolAssembly, ReservedPointsAreTakenFirstAndCountAgainstTheTotal)
{
    // Reserved points (the served edges) are taken whatever the cut, count against the total,
    // and count towards their regime's quota.
    std::vector<PoolEntry> pool;
    pool.reserve(100);
    for(int64_t index = 0; index < 100; ++index)
    {
        pool.push_back(entryAt("sweep", index, index < 90 ? "common" : "rare"));
    }
    const std::set<std::string> reserved{detail::describe(pool[89].point),
                                         detail::describe(pool[99].point)};

    const auto result = selectWith({{"sweep", pool}}, 10, {{"rare", 3}}, reserved);

    EXPECT_EQ(result.entries.size(), 10U);
    std::set<std::string> taken;
    for(const auto& entry : result.entries)
    {
        taken.insert(detail::describe(entry.point));
    }
    for(const auto& key : reserved)
    {
        EXPECT_EQ(taken.count(key), 1U) << key;
    }
    EXPECT_EQ(result.outcome.at("rare").taken, 3) << "the reserved rare point is one of the three";
    EXPECT_EQ(regimeCounts(result.entries),
              (std::map<std::string, int64_t>{{"common", 7}, {"rare", 3}}));
    EXPECT_EQ(result.allocation.at("sweep"), 10);
}

TEST(TestPoolAssembly, AQuotaIsFilledFromASourceWithNoShare)
{
    // A focused search and the served edges add to the sweep pool; with `--sweep-share 0` the
    // sweep draws nothing of its own, but what a quota or an edge asked for is still owed.
    const std::map<std::string, double> noSweep{{"model", 0.15}, {"kernel", 0.6}, {"sweep", 0.0}};
    auto focused = entryAt("sweep", 1, "rare");
    focused.owed = true;
    auto edge = entryAt("sweep", 2, "common");
    edge.owed = true;
    SourcePools pools;
    pools["sweep"] = {focused, edge, entryAt("sweep", 3, "rare")};

    std::map<std::string, int64_t> dropped;
    const auto unique = deduplicate(pools, noSweep, dropped);
    const auto result
        = selectWith(unique, 0, {{"rare", 1}}, {detail::describe(edge.point)}, noSweep);

    EXPECT_EQ(result.outcome.at("rare").taken, 1);
    std::set<int64_t> batches;
    for(const auto& entry : result.entries)
    {
        batches.insert(std::get<int64_t>(entry.point.at("batch")));
    }
    EXPECT_EQ(batches, (std::set<int64_t>{1, 2})) << "the unowed sample stays out";
    EXPECT_EQ(result.allocation.at("sweep"), 2);
}

TEST(TestPoolAssembly, AnOwedPointStillLosesToAnEarlierSourceHoldingIt)
{
    // One problem is measured once, whichever pool names it first.
    const std::map<std::string, double> noSweep{{"model", 0.15}, {"kernel", 0.6}, {"sweep", 0.0}};
    auto owed = entryAt("sweep", 2, "r");
    owed.owed = true;
    SourcePools pools;
    pools["kernel"] = {entryAt("kernel", 2, "r")};
    pools["sweep"] = {owed};

    std::map<std::string, int64_t> dropped;
    const auto unique = deduplicate(pools, noSweep, dropped);

    EXPECT_EQ(unique.at("kernel").size(), 1U);
    EXPECT_TRUE(unique.at("sweep").empty());
    EXPECT_EQ(dropped.at("sweep"), 1);
}
