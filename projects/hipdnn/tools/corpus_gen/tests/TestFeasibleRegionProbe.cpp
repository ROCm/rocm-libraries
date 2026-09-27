// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestFeasibleRegionProbe.cpp
 * @brief Covers the probe against regions whose shape is already known.
 *
 * The point of an injected oracle is that the search can be tested without an engine. Each
 * case here is a region the test defines, so the expected answer is arithmetic rather than
 * something observed and then asserted.
 *
 * Two cases carry the weight. One shows the search detecting a region it cannot characterise
 * and saying so; the other shows it failing to detect one and reporting honestly anyway,
 * because the holes align with its own stride. Both matter, and the second more: a search
 * that returns a confident bound for a region it cannot characterise looks exactly like one
 * that got it right, until the model underperforms.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/FeasibleRegionProbe.hpp>

#include <cstdint>
#include <vector>

namespace hipdnn_corpus_gen
{
namespace
{

/// Counts calls, so a test can assert the search is logarithmic rather than a walk.
class CountingOracle
{
public:
    explicit CountingOracle(MembershipOracle oracle)
        : _oracle(std::move(oracle))
    {
    }

    bool operator()(int64_t value)
    {
        ++_calls;
        return _oracle(value);
    }

    int64_t calls() const
    {
        return _calls;
    }

private:
    MembershipOracle _oracle;
    int64_t _calls = 0;
};

} // namespace

TEST(TestFeasibleRegionProbe, FindsAnUpperBoundExactly)
{
    // Everything up to 3000 accepted. The bound is a value the test chose, so "correct" here
    // is exact rather than approximate.
    const auto region = [](int64_t value) { return value <= 3000; };

    const auto bound = probeUpperBound(region, 16, 1 << 20);

    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(bound->value, 3000);
    EXPECT_FALSE(bound->acceptedBeyondBound);
}

TEST(TestFeasibleRegionProbe, FindsALowerBoundExactly)
{
    const auto region = [](int64_t value) { return value >= 64; };

    const auto bound = probeLowerBound(region, 4096);

    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(bound->value, 64);
    EXPECT_FALSE(bound->acceptedBeyondBound);
}

TEST(TestFeasibleRegionProbe, CostsAHandfulOfProbesRatherThanAWalk)
{
    // The reason a probe is affordable at all: a graph build and a predicate call per probe,
    // logarithmic in the range. A linear search over a range this size would be millions.
    const auto region = [](int64_t value) { return value <= 500000; };
    CountingOracle counting{region};

    const auto bound = probeUpperBound([&counting](int64_t v) { return counting(v); }, 1, 1 << 24);

    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(bound->value, 500000);
    // Doubling to find a rejection, bisecting to land on it, then the verification samples.
    // Generous, and still three orders of magnitude below the range.
    EXPECT_LT(counting.calls(), 80);
    EXPECT_EQ(bound->probes, counting.calls());
}

TEST(TestFeasibleRegionProbe, AnAlignmentRuleMatchingTheSearchStrideIsInvisible)
{
    // The blind spot, asserted rather than left to be discovered. The region admits only
    // multiples of 512 and the search doubles from 512, so every value it asks about is a
    // multiple of 512 and every answer is yes. It accepts to the ceiling and truthfully
    // reports no contradiction -- there was none in anything it saw.
    //
    // This is why acceptedBeyondBound is evidence of holes and never evidence of their
    // absence, and why a caller must keep putting each candidate to the predicate rather than
    // trusting the interval.
    const auto region = [](int64_t value) { return value % 512 == 0; };

    const auto bound = probeUpperBound(region, 512, 1 << 20);

    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(bound->value, 1 << 20);
    EXPECT_FALSE(bound->acceptedBeyondBound);

    // The same region caught when the start is not aligned with its stride: 500 is rejected,
    // so the search has no foothold and says so rather than guessing.
    EXPECT_FALSE(probeUpperBound(region, 500, 1 << 20).has_value());
}

TEST(TestFeasibleRegionProbe, RandomInteriorSamplingSeesWhatTheStrideSearchCannot)
{
    // The complement to the case above. Both bound searches move in powers of two and are
    // blind to this region; uniform draws have no stride to collide with, so they land off a
    // multiple of 512 almost every time and the interval is exposed as porous.
    const auto region = [](int64_t value) { return value % 512 == 0; };

    const auto density = estimateInteriorDensity(region, 512, 1 << 20, 200, /*seed=*/1234);

    EXPECT_EQ(density.sampled, 200);
    // 1/512 of the interval qualifies, so 200 draws expect well under one hit.
    EXPECT_LT(density.acceptedFraction(), 0.05);
}

TEST(TestFeasibleRegionProbe, ASolidIntervalSamplesSolid)
{
    // The control. Without it, a sampler that rejected everything would pass the case above
    // while telling us nothing.
    const auto region = [](int64_t value) { return value >= 1 && value <= 100000; };

    const auto density = estimateInteriorDensity(region, 1, 100000, 200, /*seed=*/1234);

    EXPECT_EQ(density.accepted, 200);
    EXPECT_DOUBLE_EQ(density.acceptedFraction(), 1.0);
}

TEST(TestFeasibleRegionProbe, InteriorSamplingIsReproducibleFromItsSeed)
{
    // Randomised against the region's structure, not across runs: §5.8 needs the corpus this
    // feeds to be reproducible, and a differing seed must actually change the draw or the
    // parameter is decoration.
    // Compares the values drawn rather than how many were accepted. Two seeds produce
    // different sequences with certainty; their accepted *counts* agree by chance about one
    // run in nine, which would be a rare, confusing failure rather than a signal.
    const auto sequenceFor = [](uint64_t seed) {
        std::vector<int64_t> asked;
        const auto recording = [&asked](int64_t value) {
            asked.push_back(value);
            return value % 7 == 0;
        };
        estimateInteriorDensity(recording, 1, 1000000, 100, seed);
        return asked;
    };

    EXPECT_EQ(sequenceFor(99), sequenceFor(99));
    EXPECT_NE(sequenceFor(99), sequenceFor(100));
}

TEST(TestFeasibleRegionProbe, ReportsHolesForAnExcludedBand)
{
    // A band carved out of the middle: accepted below 1000 and above 8000, rejected between.
    // The upward search stops at the near edge of the gap, but the region continues past it.
    const auto region = [](int64_t value) { return value <= 1000 || value >= 8000; };

    const auto bound = probeUpperBound(region, 16, 1 << 16);

    ASSERT_TRUE(bound.has_value());
    EXPECT_TRUE(bound->acceptedBeyondBound)
        << "an excluded band was reported as the end of the region";
}

TEST(TestFeasibleRegionProbe, AnUnboundedParameterStopsAtTheCeiling)
{
    // Nothing is rejected. The region may well continue, but nothing above the ceiling can be
    // benchmarked, so the ceiling is the only answer with any consequence for a corpus.
    const auto region = [](int64_t) { return true; };

    const auto bound = probeUpperBound(region, 32, 4096);

    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(bound->value, 4096);
    EXPECT_FALSE(bound->acceptedBeyondBound);
}

TEST(TestFeasibleRegionProbe, DeclinesWhenTheStartingValueIsRejected)
{
    // No foothold. Returning a bound here would mean inventing one, and a caller that passed
    // a rejected value has a bug worth surfacing rather than papering over -- its
    // footprint-first draw produced something the engine will not take.
    const auto region = [](int64_t value) { return value >= 1024; };

    EXPECT_FALSE(probeUpperBound(region, 16, 1 << 20).has_value());
}

TEST(TestFeasibleRegionProbe, IsDeterministicAcrossRuns)
{
    // §5.8 requires a corpus to be reproducible from its seed. A bound that moved between
    // runs would move the corpus built from it, so the verification pass samples on a fixed
    // schedule rather than at random.
    const auto region = [](int64_t value) { return value % 512 == 0; };

    const auto first = probeUpperBound(region, 512, 1 << 20);
    const auto second = probeUpperBound(region, 512, 1 << 20);

    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(first->value, second->value);
    EXPECT_EQ(first->probes, second->probes);
    EXPECT_EQ(first->acceptedBeyondBound, second->acceptedBeyondBound);
}

TEST(TestFeasibleRegionProbe, VerificationCanBeTurnedOffAndThenClaimsNothing)
{
    // Zero verification is faster and gives up the only evidence the bound means what it
    // appears to. The flag then reads false because nothing looked, which is indistinguishable
    // from looking and finding nothing -- precisely why §5.10 requires the probe count beside
    // the bound rather than the flag alone.
    const auto region = [](int64_t value) { return value % 512 == 0; };

    const auto unchecked = probeUpperBound(region, 512, 1 << 20, /*verification=*/0);

    ASSERT_TRUE(unchecked.has_value());
    EXPECT_FALSE(unchecked->acceptedBeyondBound);
}

} // namespace hipdnn_corpus_gen
