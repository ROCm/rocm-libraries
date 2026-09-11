// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestKnobEnumeration.cpp
 * @brief Covers reading the configuration space off an engine's knobs.
 *
 * The configuration half of §5.6's joint space is declared rather than searched, so the risk
 * here is not missing a region — it is covering less than the caller believes. A knob that was
 * sampled and reported as swept produces a model with a blind spot that reads as noise, so
 * every case below is as much about what gets reported as about what gets enumerated.
 */

#include <gtest/gtest.h>

#include <hipdnn_bench/KnobEnumeration.hpp>

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <string>
#include <vector>

namespace hipdnn_bench
{
namespace
{

/// Knob's constructor is private; tryCreate is the factory. Failures are fatal here rather
/// than skipped, since a test that silently built no knob would assert over an empty set.
hipdnn_frontend::Knob intKnob(const std::string& id,
                              int64_t min,
                              int64_t max,
                              int64_t step = 1,
                              std::unordered_set<int64_t> explicitValues = {})
{
    auto [error, knob] = hipdnn_frontend::Knob::tryCreate(
        id,
        "test knob",
        min,
        /*deprecated=*/false,
        std::make_shared<hipdnn_frontend::IntConstraint>(
            min, max, step, std::move(explicitValues)));
    EXPECT_TRUE(error.is_good()) << error.get_message();
    return knob;
}

/// A knob with no value set to enumerate: the EmptyConstraint tryCreate defaults to.
hipdnn_frontend::Knob unconstrainedKnob(const std::string& id)
{
    auto [error, knob]
        = hipdnn_frontend::Knob::tryCreate(id, "no constraint", int64_t{1}, false);
    EXPECT_TRUE(error.is_good()) << error.get_message();
    return knob;
}

} // namespace

TEST(TestKnobEnumeration, ReadsAnExplicitValueList)
{
    // An engine that names its values has said the set is small and meaningful; each one is a
    // kernel variant, so the list is taken whole and sorted for reproducibility.
    const auto values = enumerateKnob(intKnob("block_size", 32, 256, 1, {64, 32, 256, 128}));

    EXPECT_EQ(values.values, (std::vector<int64_t>{32, 64, 128, 256}));
    EXPECT_FALSE(values.truncated);
}

TEST(TestKnobEnumeration, WalksARangeByItsStep)
{
    const auto values = enumerateKnob(intKnob("split_k", 1, 9, 2));
    EXPECT_EQ(values.values, (std::vector<int64_t>{1, 3, 5, 7, 9}));
    EXPECT_FALSE(values.truncated);
}

TEST(TestKnobEnumeration, SaysSoWhenARangeWasOnlySampled)
{
    // The failure this guards: a caller believing it swept a knob it truncated will read the
    // model's blind spot as noise rather than as a gap it chose.
    const auto values = enumerateKnob(intKnob("wide", 1, 1000000, 1), /*limit=*/8);

    EXPECT_EQ(values.values.size(), 8U);
    EXPECT_TRUE(values.truncated);
}

TEST(TestKnobEnumeration, DeclinesAKnobItCannotEnumerate)
{
    // No constraint means no declared value set. Empty is "not swept" and the caller keeps the
    // engine default, which is not the same as a knob with a single value.
    const auto values = enumerateKnob(unconstrainedKnob("mystery"));

    EXPECT_TRUE(values.values.empty());
    EXPECT_FALSE(values.truncated);
}

TEST(TestKnobEnumeration, ProducesOneConfigurationWhenNothingIsEnumerable)
{
    // An engine with no enumerable knobs still has exactly one way to run, and a corpus needs
    // that row. Returning nothing here would drop the engine from the corpus entirely.
    const auto space = enumerateConfigurations({});

    ASSERT_EQ(space.configurations.size(), 1U);
    EXPECT_TRUE(space.configurations[0].empty());
}

TEST(TestKnobEnumeration, TakesTheProductOfSeveralKnobs)
{
    const auto space = enumerateConfigurations(
        {intKnob("a", 1, 2), intKnob("b", 10, 30, 10)});

    EXPECT_EQ(space.configurations.size(), 2U * 3U);
    EXPECT_TRUE(space.notFullyCovered.empty());
    for(const auto& configuration : space.configurations)
    {
        EXPECT_EQ(configuration.size(), 2U);
    }
}

TEST(TestKnobEnumeration, NamesEveryKnobItDidNotFullyCover)
{
    // Three distinct reasons a knob may be short, all of which must be visible: no constraint,
    // a range cut to the limit, and a product that would have exploded.
    const auto space = enumerateConfigurations(
        {intKnob("small", 1, 2),
         unconstrainedKnob("bare"),
         intKnob("huge", 1, 100000, 1)},
        /*maxConfigurations=*/4,
        /*valuesPerKnob=*/32);

    ASSERT_EQ(space.notFullyCovered.size(), 3U);
    EXPECT_NE(space.notFullyCovered[0].find("bare"), std::string::npos);
    EXPECT_NE(space.notFullyCovered[1].find("huge"), std::string::npos);
    EXPECT_NE(space.notFullyCovered[2].find("huge"), std::string::npos);

    // Capped, and the knob that would have blown the cap is simply absent from every
    // configuration rather than silently sampled.
    EXPECT_LE(space.configurations.size(), 4U);
}

} // namespace hipdnn_bench
