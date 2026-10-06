// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests for the online tuner's bookkeeping: what registering a
// problem does, how exploration ends, and what the lock-free resolution table
// answers. The tuner carries no Tensile types and reaches no HIP call on any
// path these take, so like the store tests they need neither a device nor the
// hooks hipblaslt-test uses.

#include "OnlineTuner.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <vector>

#ifdef WIN32
static int setenv(const char* name, const char* value, int)
{
    return _putenv_s(name, value);
}
#endif

using rocblaslt::OnlineTuner;

namespace
{
    // What main() puts in the environment before the singleton reads it.
    constexpr int kTopK    = 3;
    constexpr int kRepeats = 2;

    // Nothing here records an event, so no candidate ever collects a sample and
    // exploration can only end by running out its visit budget. Bounded well
    // above that budget, so a tuner that never gives up fails an assertion
    // instead of hanging the suite.
    constexpr int kCallCeiling = 4096;

    // The resolution table is direct-mapped over the key and its size is the
    // tuner's own business, so this only has to be a multiple of whatever that
    // size is for the colliding-key test to mean anything. Getting it wrong
    // costs the test its point, not its result.
    constexpr size_t kResolutionSlots = 4096;

    OnlineTuner& tuner()
    {
        return OnlineTuner::getInstance();
    }

    // Offers the same ranking until the problem resolves, and reports how many
    // offers that took, or -1 if it never did.
    int exploreToResolution(size_t key, const std::vector<int>& ranked)
    {
        for(int calls = 1; calls <= kCallCeiling; ++calls)
        {
            tuner().selectCandidate(key, ranked);
            if(tuner().resolution(key))
                return calls;
        }

        return -1;
    }

    TEST(OnlineTuner, ConfigurationComesFromTheEnvironment)
    {
        EXPECT_TRUE(tuner().enabled());
        EXPECT_EQ(tuner().topK(), kTopK);
        EXPECT_EQ(tuner().repeats(), kRepeats);
    }

    TEST(OnlineTuner, AProblemNobodyHasOfferedHasNoResolution)
    {
        EXPECT_EQ(tuner().resolution(0x1001), nullptr);
    }

    // An empty ranking is not a problem to register: there is nothing to
    // reorder and nothing to learn.
    TEST(OnlineTuner, AnEmptyRankingRegistersNothing)
    {
        constexpr size_t key = 0x1002;

        EXPECT_EQ(tuner().selectCandidate(key, {}), -1);
        EXPECT_EQ(tuner().resolution(key), nullptr);
    }

    // One candidate is nothing to choose between, so the problem is born
    // resolved with no winner and every later call takes the read-only path.
    TEST(OnlineTuner, ASingleCandidateResolvesOnSightWithNoWinner)
    {
        constexpr size_t key = 0x1003;

        EXPECT_EQ(tuner().selectCandidate(key, {11}), -1);

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr);
        EXPECT_EQ(resolved->winner(), -1);
    }

    // Exploration promotes the first candidate that still owes samples and,
    // with nothing ever measuring one, gives up at the visit budget rather than
    // holding the write lock for the life of the process.
    TEST(OnlineTuner, ExplorationEndsAtTheVisitBudget)
    {
        constexpr size_t       key = 0x1004;
        const std::vector<int> ranked{21, 22, 23, 24};

        // Only the top K are candidates, and the first of them is promoted.
        EXPECT_EQ(tuner().selectCandidate(key, ranked), 0);
        EXPECT_EQ(tuner().resolution(key), nullptr);

        const int calls = exploreToResolution(key, ranked);
        ASSERT_GT(calls, 0) << "exploration never ended";
        EXPECT_GT(calls, kTopK * kRepeats);

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr);
        EXPECT_EQ(resolved->winner(), -1);

        // Resolved, so the caller's ordering is left alone and no launch is
        // offered an event pair.
        EXPECT_EQ(tuner().selectCandidate(key, ranked), -1);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, 21, start, stop));
        EXPECT_EQ(start, nullptr);
        EXPECT_EQ(stop, nullptr);

        // Nothing is outstanding on a resolved problem, so harvesting is a
        // lookup and no more.
        tuner().harvestPending(key);
        EXPECT_EQ(tuner().resolution(key), resolved);
    }

    // The slot a key lands in is shared, so the table has to answer on the key
    // itself and not on the slot.
    TEST(OnlineTuner, AKeyInAResolvedKeysSlotIsNotMistakenForIt)
    {
        constexpr size_t key       = 0x1005;
        constexpr size_t colliding = key + kResolutionSlots;

        ASSERT_EQ(tuner().selectCandidate(key, {31}), -1);
        ASSERT_NE(tuner().resolution(key), nullptr);

        EXPECT_EQ(tuner().resolution(colliding), nullptr);
    }

    // The winner object is taken from the first caller to offer one, and the
    // position is only a hint the caller writes back.
    TEST(OnlineTuner, AResolutionStartsWithNoWinnerObjectAndNoPosition)
    {
        constexpr size_t key = 0x1006;

        ASSERT_EQ(tuner().selectCandidate(key, {41}), -1);
        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr);

        EXPECT_EQ(resolved->pinned(), nullptr);
        EXPECT_EQ(resolved->position(), -1);

        const std::shared_ptr<TensileLite::ContractionSolution> nothing;
        tuner().pinWinner(*resolved, nothing, key, 0);
        EXPECT_EQ(resolved->pinned(), nullptr) << "a caller with nothing to offer pinned something";

        resolved->setPosition(2);
        EXPECT_EQ(resolved->position(), 2);
    }

    // The selection hook registers a problem and the measurement hook only
    // measures one it registered, so a key the two disagree on is refused
    // rather than measured blind.
    TEST(OnlineTuner, BeginMeasurementRefusesAProblemSelectionNeverRegistered)
    {
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(0x1007, 51, start, stop));
        EXPECT_EQ(start, nullptr);
        EXPECT_EQ(stop, nullptr);
    }
} // namespace

int main(int argc, char** argv)
{
    // The tuner reads its configuration once, when the singleton is first
    // touched, so this has to happen before any test runs.
    setenv("HIPBLASLT_ORIGAMI_ONLINE_TUNE_TOP_K", "3", 1);
    setenv("HIPBLASLT_ORIGAMI_ONLINE_TUNE_REPEATS", "2", 1);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
