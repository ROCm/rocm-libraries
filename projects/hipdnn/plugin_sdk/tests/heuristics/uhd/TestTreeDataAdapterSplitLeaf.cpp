// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestTreeDataAdapterSplitLeaf.cpp
 * @brief TreeDataAdapter reading the split/leaf tree encoding (gbdt_model.fbs).
 *
 * The encoding stores splits and leaves apart, each array in the type the model's union tag
 * names. The same model written both ways must score identically; a tag this runtime does not
 * know, and every malformed relationship between the arrays, must be refused at load.
 */

#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/TreeDataAdapter.hpp>

#include <hipdnn_test_sdk/utilities/GbdtModelTestBuilder.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

using hipdnn_plugin_sdk::uhd::TreeDataAdapter;

namespace
{

using Builder = hipdnn_test_sdk::utilities::GbdtModelTestBuilder;
namespace fb = hipdnn_flatbuffers_sdk::data_objects;

constexpr const char* HASH = "sha256:split_leaf";
constexpr int32_t FEATURES = 6;

/// A random node-array tree, node 0 its root, its splits drawing thresholds from a small pool
/// (so the table deduplicates) and routing missing values both ways.
Builder::TreeSpec randomTree(std::mt19937& rng, int depth)
{
    Builder::TreeSpec tree;
    std::uniform_int_distribution<int32_t> feature(0, FEATURES - 1);
    std::uniform_int_distribution<int> pool(0, 15);
    std::uniform_real_distribution<double> leaf(-2.0, 2.0);
    std::bernoulli_distribution coin(0.5);
    std::bernoulli_distribution stop(0.25);

    // Grow breadth-first so node 0 stays the root.
    struct Pending
    {
        size_t node;
        int depth;
    };
    std::vector<Pending> queue{{0, 0}};
    tree.featureIndices.push_back(0);
    tree.thresholds.push_back(0.0);
    tree.leftChildren.push_back(-1);
    tree.rightChildren.push_back(-1);
    tree.leafValues.push_back(0.0);
    tree.defaultLeft.push_back(0);
    for(size_t at = 0; at < queue.size(); ++at)
    {
        const auto [node, level] = queue[at];
        if(level >= depth || (level > 0 && stop(rng)))
        {
            tree.leafValues[node] = leaf(rng);
            continue;
        }
        const auto left = static_cast<int32_t>(tree.leftChildren.size());
        for(int child = 0; child < 2; ++child)
        {
            tree.featureIndices.push_back(0);
            tree.thresholds.push_back(0.0);
            tree.leftChildren.push_back(-1);
            tree.rightChildren.push_back(-1);
            tree.leafValues.push_back(0.0);
            tree.defaultLeft.push_back(0);
            queue.push_back({tree.leftChildren.size() - 1, level + 1});
        }
        tree.featureIndices[node] = feature(rng);
        tree.thresholds[node] = 0.125 * pool(rng);
        tree.leftChildren[node] = left;
        tree.rightChildren[node] = left + 1;
        tree.defaultLeft[node] = coin(rng) ? 1 : 0;
    }
    return tree;
}

std::vector<Builder::TreeSpec> randomForest(uint32_t seed, int trees, int depth)
{
    std::mt19937 rng(seed);
    std::vector<Builder::TreeSpec> forest;
    forest.reserve(static_cast<size_t>(trees));
    for(int i = 0; i < trees; ++i)
    {
        forest.push_back(randomTree(rng, depth));
    }
    return forest;
}

/// Rows on, between and beyond the thresholds, with missing values mixed in.
std::vector<std::vector<double>> probeRows(uint32_t seed, int count)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> grid(-1, 17);
    std::bernoulli_distribution missing(0.15);
    std::vector<std::vector<double>> rows(static_cast<size_t>(count));
    for(auto& row : rows)
    {
        for(int f = 0; f < FEATURES; ++f)
        {
            row.push_back(missing(rng) ? std::numeric_limits<double>::quiet_NaN()
                                       : 0.0625 * grid(rng));
        }
    }
    return rows;
}

Builder modelOf(const std::vector<Builder::TreeSpec>& forest, double baseScore = 0.25)
{
    Builder builder;
    builder.setNumFeatures(FEATURES).setFeaturesHash(HASH).setBaseScore(baseScore);
    for(const auto& tree : forest)
    {
        builder.addTree(tree);
    }
    return builder;
}

std::unique_ptr<TreeDataAdapter> load(const std::vector<uint8_t>& buffer)
{
    return TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), HASH);
}

} // namespace

/// One model, written both ways, scores bit for bit the same: float64 leaves and a float64
/// threshold table hold exactly what the node arrays hold.
TEST(TestTreeDataAdapterSplitLeaf, ScoresExactlyAsTheNodeArrayEncodingDoes)
{
    const auto forest = randomForest(7, 40, 6);
    const auto nodeArrays = load(modelOf(forest).build());
    const auto splitLeaf = load(modelOf(forest).encodeAsSplitLeaf().build());
    ASSERT_NE(nodeArrays, nullptr);
    ASSERT_NE(splitLeaf, nullptr);
    EXPECT_EQ(splitLeaf->treeCount(), nodeArrays->treeCount());
    for(const auto& row : probeRows(11, 500))
    {
        EXPECT_EQ(splitLeaf->score(row), nodeArrays->score(row));
    }
}

/// Every integer width the union names reads back the same model.
class TestTreeDataAdapterSplitLeafWidths : public ::testing::TestWithParam<fb::IntegerArray>
{
};

TEST_P(TestTreeDataAdapterSplitLeafWidths, EveryIntegerTypeReadsTheSameModel)
{
    const auto forest = randomForest(3, 12, 4);
    const auto nodeArrays = load(modelOf(forest).build());
    const auto encoded
        = load(modelOf(forest).encodeAsSplitLeaf(fb::RealArray::Float64Array, GetParam()).build());
    ASSERT_NE(encoded, nullptr) << fb::EnumNameIntegerArray(GetParam());
    for(const auto& row : probeRows(5, 200))
    {
        EXPECT_EQ(encoded->score(row), nodeArrays->score(row));
    }
}

// Children are negative for leaves, so only the signed types (and the narrowest-fit choice)
// hold every array of a tree.
INSTANTIATE_TEST_SUITE_P(Widths,
                         TestTreeDataAdapterSplitLeafWidths,
                         ::testing::Values(fb::IntegerArray::NONE,
                                           fb::IntegerArray::Int8Array,
                                           fb::IntegerArray::Int16Array,
                                           fb::IntegerArray::Int32Array),
                         [](const auto& info) {
                             return std::string(fb::EnumNameIntegerArray(info.param));
                         });

/// The unsigned types hold features and threshold ids, never children. A table past 32767
/// entries tells a UInt16 id from one misread as signed.
TEST(TestTreeDataAdapterSplitLeaf, UnsignedFeaturesAndThresholdIdsReadAtEveryWidth)
{
    std::vector<double> table(40000);
    for(size_t i = 0; i < table.size(); ++i)
    {
        table[i] = 0.5 + static_cast<double>(i);
    }
    for(const auto type : {fb::IntegerArray::UInt8Array, fb::IntegerArray::UInt16Array})
    {
        const bool wide = type == fb::IntegerArray::UInt16Array;
        const auto top = wide ? 39999 : 255;
        Builder::SplitLeafSpec tree;
        tree.features = {0, 1};
        tree.featuresType = type;
        tree.thresholdIds = {0, top};
        tree.thresholdIdsType = type;
        tree.lefts = {1, ~0};
        tree.rights = {~2, ~1};
        tree.leaves = {1.0, 2.0, 3.0};

        Builder builder;
        builder.setNumFeatures(2).setFeaturesHash(HASH).setThresholdTable(table).addSplitLeafTree(
            tree);
        const auto adapter = load(builder.build());
        ASSERT_NE(adapter, nullptr) << fb::EnumNameIntegerArray(type);
        // f0 <= 0.5 ? (f1 <= table[top] ? leaf 0 : leaf 1) : leaf 2
        EXPECT_EQ(adapter->score({0.0, top + 0.5}), 1.0) << fb::EnumNameIntegerArray(type);
        EXPECT_EQ(adapter->score({0.0, top + 1.0}), 2.0) << fb::EnumNameIntegerArray(type);
        EXPECT_EQ(adapter->score({1.0, 0.0}), 3.0) << fb::EnumNameIntegerArray(type);
    }
}

/// float32 leaves move each leaf by at most 2^-24 of itself, and nothing else.
TEST(TestTreeDataAdapterSplitLeaf, Float32LeavesStayWithinTheirRounding)
{
    const auto forest = randomForest(13, 40, 6);
    const auto exact = load(modelOf(forest).build());
    const auto narrow
        = load(modelOf(forest).encodeAsSplitLeaf(fb::RealArray::Float32Array).build());
    ASSERT_NE(narrow, nullptr);
    double leafMagnitude = 0.0;
    for(const auto& tree : forest)
    {
        double largest = 0.0;
        for(const double value : tree.leafValues)
        {
            largest = std::max(largest, std::abs(value));
        }
        leafMagnitude += largest;
    }
    for(const auto& row : probeRows(17, 300))
    {
        EXPECT_NEAR(narrow->score(row), exact->score(row), leafMagnitude * std::ldexp(1.0, -24));
    }
}

/// A threshold table stored as float32 is read as the same values when each fits exactly.
TEST(TestTreeDataAdapterSplitLeaf, AFloat32ThresholdTableIsReadAsItsValues)
{
    const auto forest = randomForest(19, 10, 4); // every threshold is a multiple of 1/8
    std::vector<double> table;
    for(int i = 0; i <= 15; ++i)
    {
        table.push_back(0.125 * i);
    }
    const auto exact = load(modelOf(forest).build());
    const auto narrow = load(modelOf(forest)
                                 .setThresholdTable(table, fb::RealArray::Float32Array)
                                 .encodeAsSplitLeaf()
                                 .build());
    ASSERT_NE(narrow, nullptr);
    for(const auto& row : probeRows(23, 200))
    {
        EXPECT_EQ(narrow->score(row), exact->score(row));
    }
}

/// Without a per-split vector, every split routes a missing value `default_left_fill`; a
/// `split_decision_lte` false compares with `<`.
TEST(TestTreeDataAdapterSplitLeaf, RoutingFlagsDefaultAsTheSchemaSays)
{
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    Builder::SplitLeafSpec tree;
    tree.features = {0};
    tree.thresholdIds = {0};
    tree.lefts = {~0};
    tree.rights = {~1};
    tree.leaves = {10.0, 20.0};

    tree.defaultLeftFill = true;
    auto adapter = load(Builder()
                            .setNumFeatures(1)
                            .setFeaturesHash(HASH)
                            .setThresholdTable({1.0})
                            .addSplitLeafTree(tree)
                            .build());
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->score({nan}), 10.0);
    EXPECT_EQ(adapter->score({1.0}), 10.0) << "absent decision_lte is <=";

    tree.defaultLeftFill = false;
    tree.decisionLte = std::vector<uint8_t>{0};
    adapter = load(Builder()
                       .setNumFeatures(1)
                       .setFeaturesHash(HASH)
                       .setThresholdTable({1.0})
                       .addSplitLeafTree(tree)
                       .build());
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->score({nan}), 20.0);
    EXPECT_EQ(adapter->score({1.0}), 20.0) << "decision_lte false is <";
    EXPECT_EQ(adapter->score({0.5}), 10.0);
}

/// A tree with no splits is its one leaf, and needs no table.
TEST(TestTreeDataAdapterSplitLeaf, ASingleLeafTreeIsItsLeaf)
{
    Builder::SplitLeafSpec tree;
    tree.leaves = {4.5};
    const auto adapter
        = load(Builder().setNumFeatures(1).setFeaturesHash(HASH).addSplitLeafTree(tree).build());
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->score({0.0}), 4.5);
}

/// The two encodings may share one model: each tree says which it uses.
TEST(TestTreeDataAdapterSplitLeaf, OneModelMayMixEncodings)
{
    const auto forest = randomForest(29, 2, 3);
    const auto both = load(modelOf(forest).build());

    std::vector<double> table;
    for(int i = 0; i <= 15; ++i)
    {
        table.push_back(0.125 * i);
    }
    const auto mixed = load(modelOf({forest[0]})
                                .setThresholdTable(table)
                                .addSplitLeafTree(Builder::toSplitLeaf(forest[1], table))
                                .build());
    ASSERT_NE(mixed, nullptr);
    for(const auto& row : probeRows(31, 100))
    {
        EXPECT_EQ(mixed->score(row), both->score(row));
    }
}

/// A grouped model's layer-2 ensembles read the same either way.
TEST(TestTreeDataAdapterSplitLeaf, GroupedModelsRankTheSameEitherWay)
{
    const auto layer1 = randomForest(37, 5, 3);
    const auto groupA = randomForest(41, 4, 3);
    const auto groupB = randomForest(43, 4, 3);
    const auto make = [&](bool splitLeaf) {
        auto builder = modelOf(layer1, 0.0);
        builder.setGroupByFeatureIndex(5).addGroup(0.0, groupA).addGroup(1.0, groupB);
        if(splitLeaf)
        {
            builder.encodeAsSplitLeaf();
        }
        return load(builder.build());
    };
    const auto nodeArrays = make(false);
    const auto splitLeaf = make(true);
    ASSERT_NE(splitLeaf, nullptr);
    auto rows = probeRows(47, 60);
    for(size_t i = 0; i < rows.size(); ++i)
    {
        rows[i][5] = static_cast<double>(i % 2);
    }
    EXPECT_EQ(splitLeaf->scoreBatch(rows), nodeArrays->scoreBatch(rows));
}

namespace
{

/// A valid two-split tree to break one relationship at a time.
Builder::SplitLeafSpec twoSplits()
{
    Builder::SplitLeafSpec tree;
    tree.features = {0, 1};
    tree.thresholdIds = {0, 1};
    tree.lefts = {1, ~0};
    tree.rights = {~2, ~1};
    tree.leaves = {1.0, 2.0, 3.0};
    return tree;
}

/// The load's refusal for @p tree, or "" when it loads; asserts the refusal was an ERROR.
std::string refusal(const Builder::SplitLeafSpec& tree,
                    const std::vector<double>& table = {0.5, 1.5},
                    std::optional<fb::RealArray> tableTag = std::nullopt)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    Builder builder;
    builder.setNumFeatures(2).setFeaturesHash(HASH).setThresholdTable(table).addSplitLeafTree(tree);
    if(tableTag)
    {
        builder.overrideThresholdTableTag(*tableTag);
    }
    const auto buffer = builder.build();
    if(TreeDataAdapter::loadFromBuffer(buffer.data(), buffer.size(), HASH) != nullptr)
    {
        return "";
    }
    EXPECT_GE(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1U);
    return recorder.getRecordedLogsAsString();
}

} // namespace

TEST(TestTreeDataAdapterSplitLeaf, TheValidBaselineLoads)
{
    EXPECT_EQ(refusal(twoSplits()), "");
}

/// A writer may store an array in a type a newer schema adds; this runtime names it and
/// refuses the model rather than reading its bytes as another type.
TEST(TestTreeDataAdapterSplitLeaf, AnArrayTypeThisRuntimeDoesNotKnowIsRefusedByName)
{
    const auto unknownInteger = static_cast<fb::IntegerArray>(9);
    const auto unknownReal = static_cast<fb::RealArray>(7);

    auto tree = twoSplits();
    tree.featuresType = unknownInteger;
    EXPECT_NE(
        refusal(tree).find(
            "split_features is stored as integer array type 9, which this runtime cannot read"),
        std::string::npos);

    tree = twoSplits();
    tree.childrenType = unknownInteger;
    EXPECT_NE(refusal(tree).find("split children are stored in a type this runtime cannot read"),
              std::string::npos);

    tree = twoSplits();
    tree.leavesType = unknownReal;
    EXPECT_NE(refusal(tree).find("leaves are stored in a type this runtime cannot read"),
              std::string::npos);

    EXPECT_NE(refusal(twoSplits(), {0.5, 1.5}, unknownReal)
                  .find("threshold_table is stored as real array type 7, which this runtime "
                        "cannot read"),
              std::string::npos);
}

/// Each relationship the verifier cannot see, broken one at a time.
TEST(TestTreeDataAdapterSplitLeaf, MalformedTreesAreRefusedWithTheirReason)
{
    const auto expect = [](const Builder::SplitLeafSpec& tree,
                           const std::string& reason,
                           const std::vector<double>& table = {0.5, 1.5}) {
        const auto logged = refusal(tree, table);
        EXPECT_NE(logged.find(reason), std::string::npos)
            << "expected '" << reason << "', logged:\n"
            << logged;
    };

    auto tree = twoSplits();
    tree.lefts = {2, ~0}; // split 2 of 2
    expect(tree, "child index outside the tree");

    tree = twoSplits();
    tree.rights = {~3, ~1}; // leaf 3 of 3
    expect(tree, "child index outside the tree");

    tree = twoSplits();
    tree.lefts = {0, ~0}; // split 0 is its own child
    expect(tree, "cycle in child indices");

    tree = twoSplits();
    tree.thresholdIds = {0, 2};
    expect(tree, "split threshold index outside the threshold table");
    expect(twoSplits(), "split threshold index outside the threshold table", {});

    expect(twoSplits(),
           "split threshold is not finite",
           {0.5, std::numeric_limits<double>::infinity()});

    tree = twoSplits();
    tree.features = {0, 2}; // two features declared
    expect(tree, "split feature outside the declared feature count");

    tree = twoSplits();
    tree.features = {0, -1};
    expect(tree, "split feature outside the declared feature count");

    tree = twoSplits();
    tree.leaves = {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    expect(tree, "leaf prediction is not finite");

    tree = twoSplits();
    tree.features = {0};
    expect(tree, "split arrays do not hold one entry per split");

    tree = twoSplits();
    tree.thresholdIds = {0};
    expect(tree, "split arrays do not hold one entry per split");

    tree = twoSplits();
    tree.lefts = {1};
    expect(tree, "split arrays do not hold one entry per split");

    tree = twoSplits();
    tree.rights = {~2};
    expect(tree, "split arrays do not hold one entry per split");

    tree = twoSplits();
    tree.decisionLte = std::vector<uint8_t>{1};
    expect(tree, "split arrays do not hold one entry per split");

    tree = twoSplits();
    tree.defaultLeft = std::vector<uint8_t>{1};
    expect(tree, "split arrays do not hold one entry per split");

    tree = twoSplits();
    tree.leaves = {1.0};
    expect(tree, "a single-leaf tree has split arrays");

    tree = twoSplits();
    tree.leaves = {};
    expect(tree, "missing nodes or a required node array");
}

/// As for node arrays: every `trees` entry may point at one tree table, so the nodes they
/// claim are bounded by the bytes the buffer could back, not by the entries.
TEST(TestTreeDataAdapterSplitLeaf, RepeatedTreesCannotClaimMoreNodesThanTheBufferHolds)
{
    const auto buildRepeated = [](size_t repeats) {
        flatbuffers::FlatBufferBuilder fbb;
        const std::vector<uint8_t> features{0};
        const std::vector<uint8_t> ids{0};
        const std::vector<int8_t> lefts{~0};
        const std::vector<int8_t> rights{~1};
        const std::vector<float> leaves{1.0F, 2.0F};
        const auto featureArray = fb::CreateUInt8ArrayDirect(fbb, &features).Union();
        const auto idArray = fb::CreateUInt8ArrayDirect(fbb, &ids).Union();
        const auto leftArray = fb::CreateInt8ArrayDirect(fbb, &lefts).Union();
        const auto rightArray = fb::CreateInt8ArrayDirect(fbb, &rights).Union();
        const auto leafArray = fb::CreateFloat32ArrayDirect(fbb, &leaves).Union();
        fb::GbdtTreeBuilder treeBuilder(fbb);
        treeBuilder.add_split_features_type(fb::IntegerArray::UInt8Array);
        treeBuilder.add_split_features(featureArray);
        treeBuilder.add_split_thresholds_type(fb::IntegerArray::UInt8Array);
        treeBuilder.add_split_thresholds(idArray);
        treeBuilder.add_split_left_type(fb::IntegerArray::Int8Array);
        treeBuilder.add_split_left(leftArray);
        treeBuilder.add_split_right_type(fb::IntegerArray::Int8Array);
        treeBuilder.add_split_right(rightArray);
        treeBuilder.add_leaves_type(fb::RealArray::Float32Array);
        treeBuilder.add_leaves(leafArray);
        const auto tree = treeBuilder.Finish();
        const std::vector<flatbuffers::Offset<fb::GbdtTree>> trees(repeats, tree);
        const std::vector<double> table{0.5};
        const auto treesVector = fbb.CreateVector(trees);
        const auto tableArray = fb::CreateFloat64ArrayDirect(fbb, &table).Union();
        fb::GbdtModelBuilder model(fbb);
        model.add_trees(treesVector);
        model.add_num_features(1);
        model.add_threshold_table_type(fb::RealArray::Float64Array);
        model.add_threshold_table(tableArray);
        fbb.Finish(model.Finish(), fb::GbdtModelIdentifier());
        return std::vector<uint8_t>(fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize());
    };

    const auto shared = buildRepeated(4);
    const auto sharedAdapter = TreeDataAdapter::loadFromBuffer(shared.data(), shared.size(), "");
    ASSERT_NE(sharedAdapter, nullptr);
    EXPECT_DOUBLE_EQ(sharedAdapter->score({0.0}), 4.0);

    // One int8 child and two float32 leaves: 9 bytes a tree claims.
    constexpr size_t REPEATS = 4000;
    const auto inflated = buildRepeated(REPEATS);
    ASSERT_GT(9 * REPEATS, inflated.size());
    EXPECT_EQ(TreeDataAdapter::loadFromBuffer(inflated.data(), inflated.size(), ""), nullptr);
}
