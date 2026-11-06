// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestTreeDataAdapterScoring.cpp
 * @brief TreeDataAdapter's rank-compare walk against a direct walk, and early-exit
 *        (staged) catalog scoring.
 */

#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/TreeDataAdapter.hpp>

#include <hipdnn_test_sdk/utilities/GbdtModelTestBuilder.hpp>
#include <hipdnn_test_sdk/utilities/UhdFeatureRows.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

using hipdnn_plugin_sdk::uhd::TreeDataAdapter;
using hipdnn_test_sdk::utilities::scoreRows;

namespace
{

using Builder = hipdnn_test_sdk::utilities::GbdtModelTestBuilder;

constexpr const char* HASH = "sha256:scoring";
constexpr int32_t FEATURES = 5;
constexpr double NAN_VALUE = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();

/// A random split/leaf tree over `table`: random features, threshold ids, both comparisons
/// and both missing-value directions. Splits are numbered breadth-first from the root.
Builder::SplitLeafSpec randomTree(std::mt19937& rng, size_t tableSize, int depth)
{
    std::uniform_int_distribution<int32_t> feature(0, FEATURES - 1);
    std::uniform_int_distribution<int32_t> threshold(0, static_cast<int32_t>(tableSize) - 1);
    std::uniform_real_distribution<double> leaf(-3.0, 3.0);
    std::bernoulli_distribution coin(0.5);
    std::bernoulli_distribution stop(0.2);

    Builder::SplitLeafSpec tree;
    tree.defaultLeft = std::vector<uint8_t>{};
    tree.decisionLte = std::vector<uint8_t>{};
    // Splits still to give children, with their depth, breadth-first.
    std::vector<std::pair<size_t, int>> splitsToExpand;
    const auto newSplit = [&]() {
        tree.features.push_back(feature(rng));
        tree.thresholdIds.push_back(threshold(rng));
        tree.lefts.push_back(0);
        tree.rights.push_back(0);
        tree.defaultLeft->push_back(coin(rng) ? 1 : 0);
        tree.decisionLte->push_back(coin(rng) ? 1 : 0);
        return static_cast<int32_t>(tree.features.size() - 1);
    };
    const auto newLeaf = [&]() {
        tree.leaves.push_back(leaf(rng));
        return ~static_cast<int32_t>(tree.leaves.size() - 1);
    };
    const int32_t root = newSplit();
    splitsToExpand.emplace_back(static_cast<size_t>(root), 0);
    for(size_t at = 0; at < splitsToExpand.size(); ++at)
    {
        const auto [split, level] = splitsToExpand[at];
        for(int side = 0; side < 2; ++side)
        {
            int32_t child = 0;
            if(level + 1 >= depth || stop(rng))
            {
                child = newLeaf();
            }
            else
            {
                child = newSplit();
                splitsToExpand.emplace_back(static_cast<size_t>(child), level + 1);
            }
            (side == 0 ? tree.lefts : tree.rights)[split] = child;
        }
    }
    return tree;
}

/// The direct walk the adapter must reproduce: compare values, NaN or a short row take the
/// default direction, and leaves are summed in tree order before the base score is added.
double directScore(const std::vector<Builder::SplitLeafSpec>& forest,
                   const std::vector<double>& table,
                   double baseScore,
                   const std::vector<double>& row,
                   size_t begin = 0,
                   size_t end = std::numeric_limits<size_t>::max(),
                   bool addBase = true)
{
    double sum = 0.0;
    for(size_t t = begin; t < std::min(end, forest.size()); ++t)
    {
        const auto& tree = forest[t];
        int32_t node = 0;
        while(node >= 0)
        {
            const auto i = static_cast<size_t>(node);
            const auto feature = static_cast<size_t>(tree.features[i]);
            const double threshold = table[static_cast<size_t>(tree.thresholdIds[i])];
            bool goLeft = (*tree.defaultLeft)[i] != 0;
            if(feature < row.size() && !std::isnan(row[feature]))
            {
                goLeft = (*tree.decisionLte)[i] != 0 ? row[feature] <= threshold
                                                     : row[feature] < threshold;
            }
            node = goLeft ? tree.lefts[i] : tree.rights[i];
        }
        const int32_t leaf = ~node;
        sum += tree.leaves[static_cast<size_t>(leaf)];
    }
    return addBase ? baseScore + sum : sum;
}

/// A sorted table with near neighbours, a signed zero and wide magnitudes.
std::vector<double> table()
{
    std::vector<double> values = {-1e300, -7.5, -1.0, -0.0, 1e-300, 0.25, 0.5, 1.0, 3.0, 1e18};
    return values;
}

/// Rows on, just beside and between every threshold, plus NaN, +-infinity and +-zero.
std::vector<std::vector<double>> probeRows(uint32_t seed, int count, const std::vector<double>& t)
{
    std::vector<double> probes = {NAN_VALUE, INF, -INF, 0.0, -0.0};
    for(const double v : t)
    {
        probes.push_back(v);
        probes.push_back(std::nextafter(v, -INF));
        probes.push_back(std::nextafter(v, INF));
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, probes.size() - 1);
    std::vector<std::vector<double>> rows(static_cast<size_t>(count));
    for(auto& row : rows)
    {
        for(int f = 0; f < FEATURES; ++f)
        {
            row.push_back(probes[pick(rng)]);
        }
    }
    return rows;
}

std::vector<Builder::SplitLeafSpec> randomForest(uint32_t seed, int trees, int depth)
{
    std::mt19937 rng(seed);
    std::vector<Builder::SplitLeafSpec> forest;
    forest.reserve(static_cast<size_t>(trees));
    for(int i = 0; i < trees; ++i)
    {
        forest.push_back(randomTree(rng, table().size(), depth));
    }
    return forest;
}

Builder modelOf(const std::vector<Builder::SplitLeafSpec>& forest, double baseScore)
{
    Builder builder;
    builder.setNumFeatures(FEATURES).setFeaturesHash(HASH).setBaseScore(baseScore);
    builder.setThresholdTable(table());
    for(const auto& tree : forest)
    {
        builder.addSplitLeafTree(tree);
    }
    return builder;
}

std::unique_ptr<TreeDataAdapter> load(const std::vector<uint8_t>& buffer,
                                      const std::string& objective = "max")
{
    return TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), HASH, "", objective);
}

/// A one-split tree on feature 0 at threshold id 0 (value 0.0 in `stumpTable`).
Builder::SplitLeafSpec stump(double below, double above)
{
    Builder::SplitLeafSpec tree;
    tree.features = {0};
    tree.thresholdIds = {0};
    tree.lefts = {-1};
    tree.rights = {-2};
    tree.leaves = {below, above};
    return tree;
}

/// Three stumps; the first misorders rows 1 and 2 against the whole model.
Builder stumps()
{
    Builder builder;
    builder.setNumFeatures(1).setFeaturesHash(HASH).setBaseScore(100.0);
    builder.setThresholdTable({0.0, 1.0, 2.0});
    const Builder::SplitLeafSpec first = stump(0.0, 1.0); // row value <= 0 -> 0, else 1
    Builder::SplitLeafSpec second;
    second.features = {0};
    second.thresholdIds = {1};
    second.lefts = {-1};
    second.rights = {-2};
    second.leaves = {0.0, 5.0}; // > 1 -> +5
    Builder::SplitLeafSpec third;
    third.features = {0};
    third.thresholdIds = {2};
    third.lefts = {-1};
    third.rights = {-2};
    third.leaves = {0.0, 10.0}; // > 2 -> +10
    builder.addSplitLeafTree(first).addSplitLeafTree(second).addSplitLeafTree(third);
    return builder;
}

const std::vector<std::vector<double>> STUMP_ROWS = {{0.0}, {0.5}, {1.5}, {2.5}};
// Full scores 100, 101, 106, 116; after the first tree 0, 1, 1, 1.

} // namespace

/// Rank comparison reproduces a direct value walk bit for bit, for both comparisons, both
/// missing directions, values exactly on, just beside and between thresholds, NaN, +-inf and
/// signed zeros, through score() and scoreBatch().
TEST(TestTreeDataAdapterScoring, RankWalkScoresExactlyAsADirectWalk)
{
    const auto forest = randomForest(17, 60, 7);
    const auto adapter = load(modelOf(forest, 0.375).build());
    ASSERT_NE(adapter, nullptr);
    const auto rows = probeRows(23, 400, table());
    const auto batch = scoreRows(*adapter, rows);
    ASSERT_EQ(batch.raw.size(), rows.size());
    EXPECT_TRUE(batch.stage.empty());
    for(size_t i = 0; i < rows.size(); ++i)
    {
        const double expected = directScore(forest, table(), 0.375, rows[i]);
        EXPECT_EQ(adapter->score(rows[i]), expected) << "row " << i;
        EXPECT_EQ(batch.raw[i], expected) << "row " << i;
    }
}

/// Rows sharing values with the row before reuse its positions; the reuse must not leak a
/// position across a different value.
TEST(TestTreeDataAdapterScoring, BatchesWithRepeatedValuesScoreAsSingleRows)
{
    const auto forest = randomForest(5, 30, 6);
    const auto adapter = load(modelOf(forest, -1.0).build());
    ASSERT_NE(adapter, nullptr);
    auto rows = probeRows(9, 200, table());
    // Runs of rows that differ in one feature at a time, as a catalog's rows do.
    for(size_t i = 1; i < rows.size(); i += 2)
    {
        rows[i] = rows[i - 1];
        rows[i][i % FEATURES] = rows[(i + 7) % rows.size()][i % FEATURES];
    }
    const auto batch = scoreRows(*adapter, rows);
    for(size_t i = 0; i < rows.size(); ++i)
    {
        EXPECT_EQ(batch.raw[i], directScore(forest, table(), -1.0, rows[i])) << "row " << i;
    }
}

/// Features past a short row's end take the default direction, as a NaN does.
TEST(TestTreeDataAdapterScoring, ShortRowsReadMissingFeaturesAsMissing)
{
    const auto forest = randomForest(31, 30, 6);
    const auto adapter = load(modelOf(forest, 0.0).build());
    ASSERT_NE(adapter, nullptr);
    for(auto row : probeRows(3, 100, table()))
    {
        row.resize(2);
        EXPECT_EQ(adapter->score(row), directScore(forest, table(), 0.0, row));
    }
}

/// A schedule that never prunes (keep at least the catalog) scores every row fully.
TEST(TestTreeDataAdapterScoring, ScheduleKeepingTheCatalogScoresEveryRowFully)
{
    const auto forest = randomForest(41, 20, 5);
    auto builder = modelOf(forest, 0.5);
    builder.setEarlyExit({{3, 200}, {9, 100}});
    const auto adapter = load(builder.build());
    ASSERT_NE(adapter, nullptr);
    ASSERT_EQ(adapter->earlyExit().size(), 2u);
    const auto rows = probeRows(43, 50, table());
    const auto batch = scoreRows(*adapter, rows);
    ASSERT_EQ(batch.stage.size(), rows.size());
    EXPECT_EQ(batch.stages, 2);
    for(size_t i = 0; i < rows.size(); ++i)
    {
        EXPECT_TRUE(batch.fullyScored(i));
        EXPECT_EQ(batch.raw[i], directScore(forest, table(), 0.5, rows[i]));
    }
}

/// After one tree, rows 1-3 tie at 1 and row 0 trails; keep 2 keeps the earliest two of the
/// tied rows. Survivors carry full scores; the pruned rows their one-tree partial sums.
TEST(TestTreeDataAdapterScoring, PrunesOnThePrefixScoreAndFinishesSurvivors)
{
    auto builder = stumps();
    builder.setEarlyExit({{1, 2}});
    const auto adapter = load(builder.build(), "max");
    ASSERT_NE(adapter, nullptr);
    const auto batch = scoreRows(*adapter, STUMP_ROWS);
    ASSERT_EQ(batch.stage.size(), 4u);
    EXPECT_EQ(batch.stages, 1);
    EXPECT_EQ(batch.stage, (std::vector<uint8_t>{0, 1, 1, 0}));
    EXPECT_EQ(batch.raw[1], 101.0);
    EXPECT_EQ(batch.raw[2], 106.0);
    // Pruned: partial sums over the first tree, without the base score.
    EXPECT_EQ(batch.raw[0], 0.0);
    EXPECT_EQ(batch.raw[3], 1.0);
}

/// Under `min` the lowest prefix scores survive.
TEST(TestTreeDataAdapterScoring, PruningFollowsTheObjective)
{
    auto builder = stumps();
    builder.setEarlyExit({{1, 1}});
    const auto adapter = load(builder.build(), "min");
    ASSERT_NE(adapter, nullptr);
    const auto batch = scoreRows(*adapter, STUMP_ROWS);
    EXPECT_EQ(batch.stage, (std::vector<uint8_t>{1, 0, 0, 0}));
    EXPECT_EQ(batch.raw[0], 100.0);
}

/// A second stage records how far each pruned row got.
TEST(TestTreeDataAdapterScoring, RowsRecordTheStageTheyWerePrunedAt)
{
    auto builder = stumps();
    builder.setEarlyExit({{1, 3}, {2, 1}});
    const auto adapter = load(builder.build(), "max");
    ASSERT_NE(adapter, nullptr);
    const auto batch = scoreRows(*adapter, STUMP_ROWS);
    // Stage 0 drops row 0 (prefix 0). After two trees rows 1, 2, 3 have 1, 6, 6; keep 1
    // keeps row 2 (the earlier of the tie).
    EXPECT_EQ(batch.stage, (std::vector<uint8_t>{0, 1, 2, 1}));
    EXPECT_EQ(batch.raw[2], 106.0);
    EXPECT_EQ(batch.raw[1], 1.0);
    EXPECT_EQ(batch.raw[3], 6.0);
}

/// On a random forest: survivors carry the direct full score; a row pruned at stage s
/// carries its direct partial sum over the trees stage s summed; and at every stage each row
/// that went on scored at least as well, on that stage's prefix, as each row pruned there.
TEST(TestTreeDataAdapterScoring, StagedScoresMatchDirectPrefixSums)
{
    const auto forest = randomForest(77, 40, 6);
    const std::vector<std::pair<uint32_t, uint32_t>> schedule = {{5, 12}, {15, 4}};
    auto builder = modelOf(forest, 0.25);
    builder.setEarlyExit(schedule);
    const auto adapter = load(builder.build());
    ASSERT_NE(adapter, nullptr);
    const auto rows = probeRows(79, 40, table());
    const auto batch = scoreRows(*adapter, rows);
    ASSERT_EQ(batch.stage.size(), rows.size());
    size_t survivors = 0;
    for(size_t i = 0; i < rows.size(); ++i)
    {
        if(batch.fullyScored(i))
        {
            ++survivors;
            EXPECT_EQ(batch.raw[i], directScore(forest, table(), 0.25, rows[i]));
            continue;
        }
        const auto trees = schedule[batch.stage[i]].first;
        EXPECT_EQ(batch.raw[i], directScore(forest, table(), 0.25, rows[i], 0, trees, false));
    }
    EXPECT_EQ(survivors, 4u);
    for(size_t s = 0; s < schedule.size(); ++s)
    {
        const auto trees = schedule[s].first;
        for(size_t pruned = 0; pruned < rows.size(); ++pruned)
        {
            if(batch.fullyScored(pruned) || batch.stage[pruned] != s)
            {
                continue;
            }
            const double prunedPrefix
                = directScore(forest, table(), 0.25, rows[pruned], 0, trees, false);
            for(size_t kept = 0; kept < rows.size(); ++kept)
            {
                if(batch.fullyScored(kept) || batch.stage[kept] > s)
                {
                    EXPECT_GE(directScore(forest, table(), 0.25, rows[kept], 0, trees, false),
                              prunedPrefix)
                        << "stage " << s << " kept row " << kept << " pruned row " << pruned;
                }
            }
        }
    }
}

/// Malformed schedules are refused, each for its reason.
TEST(TestTreeDataAdapterScoring, MalformedSchedulesAreRefused)
{
    const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> malformed = {
        {{0, 2}}, // no trees before pruning
        {{3, 2}}, // as many trees as the model has
        {{1, 0}}, // keeps nothing
        {{2, 2}, {1, 1}}, // trees do not increase
        {{1, 2}, {2, 2}}, // keep does not decrease
    };
    for(const auto& schedule : malformed)
    {
        auto builder = stumps();
        builder.setEarlyExit(schedule);
        EXPECT_EQ(load(builder.build()), nullptr) << schedule.front().first;
    }
}

/// A grouped model's layer 1 ranks groups, so its schedule is ignored, not refused.
TEST(TestTreeDataAdapterScoring, GroupedModelsIgnoreTheSchedule)
{
    const auto makeTree = [](double left, double right) {
        Builder::TreeSpec tree;
        tree.featureIndices = {0, 0, 0};
        tree.thresholds = {0.5, 0.0, 0.0};
        tree.leftChildren = {1, -1, -1};
        tree.rightChildren = {2, -1, -1};
        tree.leafValues = {0.0, left, right};
        tree.defaultLeft = {0, 0, 0};
        return tree;
    };
    Builder builder;
    builder.setNumFeatures(2).setFeaturesHash(HASH).setGroupByFeatureIndex(0);
    builder.addTree(makeTree(1.0, 9.0)).addTree(makeTree(0.0, 0.0));
    builder.addGroup(0.0, {makeTree(100.0, 100.0)}).addGroup(1.0, {makeTree(200.0, 200.0)});
    const auto plain = load(builder.build());
    builder.setEarlyExit({{1, 1}});
    const auto scheduled = load(builder.build());
    ASSERT_NE(plain, nullptr);
    ASSERT_NE(scheduled, nullptr);
    EXPECT_TRUE(scheduled->earlyExit().empty());
    const std::vector<std::vector<double>> rows = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};
    EXPECT_EQ(scoreRows(*scheduled, rows).raw, scoreRows(*plain, rows).raw);
    EXPECT_TRUE(scoreRows(*scheduled, rows).stage.empty());
}
