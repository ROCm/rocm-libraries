// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the online tuner's bookkeeping: registration, how exploration ends, the
// lock-free resolution table and the event pool.
//
// They run against a private copy of the tuner: OnlineTuner.cpp is compiled into the
// test binary with the onlineTunerConfig() defined at the bottom of this file. A shared
// libhipblaslt exports neither, so the library's tuner is a separate object; a static
// build runs these cases in hipblaslt-test-online-tuner instead. See
// clients/tests/src/CMakeLists.txt.
//
// The cases at the end of the file need a device and skip without one. The one case
// that needs two devices is in the OnlineTuner_multi_gpu suite.

#include "OnlineTuner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <future>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using rocblaslt::OnlineTuner;

/**
 * Declared by OnlineTuner.hpp and defined only here. Read-only access to state
 * the public surface does not expose.
 */
struct OnlineTunerTestAccess
{
    // Heap the tuner still holds for a problem's exploration.
    static size_t exploringBytes(const OnlineTuner& tuner, size_t problemKey)
    {
        const auto found = tuner.m_problems.find(problemKey);
        if(found == tuner.m_problems.end())
            return 0;

        const OnlineTuner::ProblemState& state = found->second;

        size_t bytes = state.m_samples.capacity() * sizeof(std::vector<float>)
                       + state.m_pending.capacity() * sizeof(OnlineTuner::PendingMeasurement);

        for(const std::vector<float>& samples : state.m_samples)
            bytes += samples.capacity() * sizeof(float);

        return bytes;
    }

    // Pairs waiting in a device's pool, newest last.
    static std::vector<std::pair<hipEvent_t, hipEvent_t>> pooledPairs(const OnlineTuner& tuner,
                                                                      int                device)
    {
        std::vector<std::pair<hipEvent_t, hipEvent_t>> pairs;

        const auto found = tuner.m_pairs.find(device);
        if(found != tuner.m_pairs.end())
            for(const OnlineTuner::EventPair& pair : found->second)
                pairs.emplace_back(pair.m_start, pair.m_stop);

        return pairs;
    }

    // Samples a problem has taken, over all its candidates.
    static size_t samples(const OnlineTuner& tuner, size_t problemKey)
    {
        const auto found = tuner.m_problems.find(problemKey);
        if(found == tuner.m_problems.end())
            return 0;

        size_t count = 0;
        for(const std::vector<float>& candidate : found->second.m_samples)
            count += candidate.size();
        return count;
    }

    // Pairs created on a device and not yet retired, pooled or outstanding.
    static int livePairs(const OnlineTuner& tuner, int device)
    {
        const auto found = tuner.m_livePairs.find(device);
        return found == tuner.m_livePairs.end() ? 0 : found->second;
    }

    // The problems a reclaim pass would walk.
    static std::vector<size_t> holders(const OnlineTuner& tuner)
    {
        return tuner.m_holders;
    }
};

namespace
{
    // What this test's onlineTunerConfig() hands the singleton.
    constexpr int kTopK    = 3;
    constexpr int kRepeats = 2;

    // Small, so a case reaches the cap with a handful of shapes.
    constexpr int kLivePairCap = 16;

    // Host-only cases record no event, so exploration can only end at the visit
    // budget. Bounded well above it, so a tuner that never gives up fails instead of
    // hanging.
    constexpr int kCallCeiling = 4096;

    // visitBudget() bounds the visits the registrant may spend; abandonBudget()
    // bounds how long a problem nobody is exploring stays unresolved.
    constexpr int kVisitBudget   = kTopK * kRepeats * OnlineTuner::c_visitBudgetFactor;
    constexpr int kAbandonBudget = kVisitBudget * OnlineTuner::c_abandonBudgetFactor;

    // Well past the registrant's budget and well short of the abandonment ceiling.
    constexpr int kEarlyEnough = 512;

    // The resolution table is direct-mapped, so keys this far apart share a slot.
    constexpr size_t kResolutionSlots = OnlineTuner::c_resolutionSlots;

    OnlineTuner& tuner()
    {
        return OnlineTuner::getInstance();
    }

    // What the livePairCapReached callback was told, in order: (device, cap).
    std::vector<std::pair<int, int>>& capReports()
    {
        static std::vector<std::pair<int, int>> reports;
        return reports;
    }

    void recordCapReached(int device, int cap)
    {
        capReports().emplace_back(device, cap);
    }

    /**
     * Stands in for the problem identity the library passes alongside the
     * resolution key. Varying it under a fixed key makes a merged sibling: a
     * problem the key covers that ranks or runs differently.
     */
    size_t identityOf(size_t key, int variant = 0)
    {
        return key * 1000003u + static_cast<size_t>(variant);
    }

    /**
     * A solution pointer the tuner can carry but nothing dereferences; the
     * solution library is not linked here.
     */
    std::shared_ptr<TensileLite::ContractionSolution> placeholderSolution()
    {
        static char storage = 0;
        return std::shared_ptr<TensileLite::ContractionSolution>(
            std::shared_ptr<void>(), reinterpret_cast<TensileLite::ContractionSolution*>(&storage));
    }

    // Offers the same ranking for the same problem until it resolves. Returns the
    // number of offers that took, or -1 if it never resolved.
    int exploreToResolution(size_t key, size_t problem, const std::vector<int>& ranked)
    {
        for(int calls = 1; calls <= kCallCeiling; ++calls)
        {
            tuner().selectCandidate(key, problem, ranked);
            if(tuner().resolution(key))
                return calls;
        }

        return -1;
    }

    TEST(OnlineTuner, ConfigurationComesFromTheLink)
    {
        EXPECT_TRUE(tuner().enabled());
        EXPECT_EQ(tuner().topK(), kTopK);
        EXPECT_EQ(tuner().repeats(), kRepeats);
        EXPECT_EQ(tuner().livePairCap(), kLivePairCap);

        EXPECT_EQ(tuner().coldCalls(), 0);
    }

    TEST(OnlineTuner, AProblemNobodyHasOfferedHasNoResolution)
    {
        EXPECT_EQ(tuner().resolution(0x1001), nullptr);
    }

    // An empty ranking has nothing to reorder, so nothing is registered.
    TEST(OnlineTuner, AnEmptyRankingRegistersNothing)
    {
        constexpr size_t key = 0x1002;

        EXPECT_EQ(tuner().selectCandidate(key, identityOf(key), {}), -1);
        EXPECT_EQ(tuner().resolution(key), nullptr);
    }

    // One candidate is nothing to choose between: the problem is born resolved
    // with no winner.
    TEST(OnlineTuner, ASingleCandidateResolvesOnSightWithNoWinner)
    {
        constexpr size_t key = 0x1003;

        EXPECT_EQ(tuner().selectCandidate(key, identityOf(key), {11}), -1);

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr);
        EXPECT_EQ(resolved->winner(), -1);

        EXPECT_FLOAT_EQ(resolved->winnerTimeUs(), 0.0f);
    }

    // With nothing ever measured, exploration gives up at the visit budget rather
    // than holding the write lock for the life of the process.
    TEST(OnlineTuner, ExplorationEndsAtTheVisitBudget)
    {
        constexpr size_t       key = 0x1004;
        const std::vector<int> ranked{21, 22, 23, 24};

        EXPECT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0);
        EXPECT_EQ(tuner().resolution(key), nullptr);

        const int calls = exploreToResolution(key, identityOf(key), ranked);
        ASSERT_GT(calls, 0) << "exploration never ended";
        EXPECT_GT(calls, kTopK * kRepeats);

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr);
        EXPECT_EQ(resolved->winner(), -1);

        EXPECT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), -1);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, 21, nullptr, start, stop));
        EXPECT_EQ(start, nullptr);
        EXPECT_EQ(stop, nullptr);

        tuner().harvestPending(key);
        EXPECT_EQ(tuner().resolution(key), resolved);
    }

    // Slots are shared, so the table must answer on the key, not the slot.
    TEST(OnlineTuner, AKeyInAResolvedKeysSlotIsNotMistakenForIt)
    {
        constexpr size_t key       = 0x1005;
        constexpr size_t colliding = key + kResolutionSlots;

        ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), {31}), -1);
        ASSERT_NE(tuner().resolution(key), nullptr);

        EXPECT_EQ(tuner().resolution(colliding), nullptr);
    }

    // The winner object is taken from the first caller to offer one, and the
    // position is only a hint the caller writes back.
    TEST(OnlineTuner, AResolutionStartsWithNoWinnerObjectAndNoPosition)
    {
        constexpr size_t key = 0x1006;

        ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), {41}), -1);
        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr);

        EXPECT_EQ(resolved->pinned(), nullptr);
        EXPECT_EQ(resolved->position(), -1);

        const std::shared_ptr<TensileLite::ContractionSolution> nothing;
        EXPECT_EQ(tuner().pinWinner(
                      *resolved, nothing, identityOf(key), 0, 0, TensileLite::ProblemOverride{}),
                  nullptr)
            << "a caller with nothing to offer pinned something";
        EXPECT_EQ(resolved->pinned(), nullptr) << "a caller with nothing to offer pinned something";

        resolved->setPosition(2);
        EXPECT_EQ(resolved->position(), 2);
    }

    // A record is installed once, and pinWinner says which caller installed it, so
    // only that caller writes a row.
    TEST(OnlineTuner, OnlyOneCallerInstallsTheRecord)
    {
        constexpr size_t key = 0x1008;

        ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), {61}), -1);
        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr);
        ASSERT_EQ(resolved->pinned(), nullptr);

        TensileLite::ProblemOverride measured;
        measured.m = 4096;
        TensileLite::ProblemOverride sibling;
        sibling.m = 8192;

        const auto solution = placeholderSolution();

        const OnlineTuner::PinnedWinner* installed
            = tuner().pinWinner(*resolved, solution, identityOf(key), 256, 1024, measured);
        ASSERT_NE(installed, nullptr);
        EXPECT_EQ(installed, resolved->pinned());
        EXPECT_EQ(installed->requiredWorkspace, 256u);
        EXPECT_EQ(installed->searchWorkspace, 1024u);
        EXPECT_EQ(installed->tuningKey.m, 4096u);

        EXPECT_EQ(tuner().pinWinner(*resolved, solution, identityOf(key), 0, 0, sibling), nullptr)
            << "a second caller was told it had installed the record";
        EXPECT_EQ(resolved->pinned(), installed);
        EXPECT_EQ(resolved->pinned()->tuningKey.m, 4096u)
            << "a later caller restated the key the row is written under";
    }

    // A key the selection hook never registered is refused, not measured blind.
    TEST(OnlineTuner, BeginMeasurementRefusesAProblemSelectionNeverRegistered)
    {
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(0x1007, 51, nullptr, start, stop));
        EXPECT_EQ(start, nullptr);
        EXPECT_EQ(stop, nullptr);
    }

    // ----------------------------------------------------------------------
    // Merged siblings: two problems under one resolution key.
    //
    // The key merges problems that differ in, for example, leading dimension or
    // epilogue, which both move a kernel's runtime. Their launches are not
    // comparable samples, so only the registering problem is offered candidates,
    // timed, and recorded.
    // ----------------------------------------------------------------------

    // A sibling is left its own ordering and spends none of the visit budget,
    // which it could otherwise exhaust.
    TEST(OnlineTuner, AMergedSiblingIsOfferedNoCandidate)
    {
        constexpr size_t key = 0x1009;

        // Same m, n, k and types; different leading dimension.
        const size_t registering = identityOf(key, 0);
        const size_t merged      = identityOf(key, 1);

        const std::vector<int> ranked{71, 72, 73};

        EXPECT_EQ(tuner().selectCandidate(key, registering, ranked), 0);

        // Well past the visit budget.
        for(int visit = 0; visit < 512; ++visit)
            ASSERT_EQ(tuner().selectCandidate(key, merged, ranked), -1)
                << "a merged sibling was handed a probe candidate on visit " << visit;

        EXPECT_EQ(tuner().resolution(key), nullptr)
            << "a merged sibling's visits ran the exploration out of its budget";

        // Still owed its first candidate: nothing was credited.
        EXPECT_EQ(tuner().selectCandidate(key, registering, ranked), 0);
    }

    // A sibling's own pick is often a candidate this exploration still owes, so
    // matching a launch on the solution index alone would credit one problem with
    // the other's launch. Needs a device: without one no launch is accepted and
    // this would pass for the wrong reason.
    TEST(OnlineTuner, AMergedSiblingsOwnLaunchIsNotSampled)
    {
        constexpr size_t key = 0x100a;

        const size_t registering = identityOf(key, 0);
        const size_t merged      = identityOf(key, 1);

        const std::vector<int> ranked{81, 82, 83};

        ASSERT_EQ(tuner().selectCandidate(key, registering, ranked), 0);
        ASSERT_EQ(tuner().selectCandidate(key, merged, ranked), -1);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[0], nullptr, start, stop))
            << "a merged sibling's launch was timed as a sample of this problem";
        EXPECT_EQ(start, nullptr);
        EXPECT_EQ(stop, nullptr);

        // Had the sibling's launch been credited, the rotation would have moved on.
        EXPECT_EQ(tuner().selectCandidate(key, registering, ranked), 0)
            << "a merged sibling's launch advanced this problem's rotation";
    }

    // The offer names one candidate and buys one sample of it.
    TEST(OnlineTuner, OnlyTheCandidateThisVisitWasOfferedIsSampled)
    {
        constexpr size_t key = 0x100b;

        const std::vector<int> ranked{84, 85, 86};

        ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;

        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[1], nullptr, start, stop))
            << "a candidate the rotation did not ask for was timed";
        EXPECT_EQ(start, nullptr);
        EXPECT_EQ(stop, nullptr);
    }

    // The record's key belongs to the problem that was measured. The sibling
    // arrives first here, and its ranking holds the winner, so only whose key ends
    // up on the record tells the two apart.
    TEST(OnlineTuner, OnlyTheMeasuredProblemInstallsTheRecord)
    {
        constexpr size_t key = 0x100c;

        const size_t measured = identityOf(key, 0);
        const size_t merged   = identityOf(key, 1);

        ASSERT_GT(exploreToResolution(key, measured, {94, 95, 96}), 0) << "exploration never ended";

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr);
        ASSERT_EQ(resolved->pinned(), nullptr);

        // One resolution key, two file keys.
        TensileLite::ProblemOverride measuredKey;
        measuredKey.m          = 4096;
        measuredKey.colStrideA = 4096;
        TensileLite::ProblemOverride mergedKey;
        mergedKey.m          = 4096;
        mergedKey.colStrideA = 8192;

        const auto solution = placeholderSolution();

        EXPECT_EQ(tuner().pinWinner(*resolved, solution, merged, 0, 0, mergedKey), nullptr)
            << "a problem whose launches were never sampled installed the record";
        EXPECT_EQ(resolved->pinned(), nullptr)
            << "a problem whose launches were never sampled installed the record";

        const OnlineTuner::PinnedWinner* installed
            = tuner().pinWinner(*resolved, solution, measured, 256, 1024, measuredKey);
        ASSERT_NE(installed, nullptr) << "the measured problem was refused its own record";
        EXPECT_EQ(installed, resolved->pinned());
        EXPECT_EQ(installed->problem, measured);
        EXPECT_EQ(installed->tuningKey.colStrideA, 4096u)
            << "the row would be written under a merged sibling's key";
    }

    // ----------------------------------------------------------------------
    // HIP graph capture.
    //
    // A graph replays its kernel choice without coming back through selection, so a
    // probe captured during exploration would never be corrected, and is never
    // timed. A resolved problem's measured winner is what belongs in a graph.
    // ----------------------------------------------------------------------

    // A capture is offered nothing during exploration and spends none of its budget.
    TEST(OnlineTuner, CapturesAreOfferedNoCandidateAndSpendNoVisitBudget)
    {
        constexpr size_t key = 0x100d;

        const size_t           problem = identityOf(key);
        const std::vector<int> ranked{101, 102, 103};

        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked), 0);

        // Well past the visit budget.
        for(int visit = 0; visit < 512; ++visit)
            ASSERT_EQ(tuner().selectCandidate(key, problem, ranked, /*capturing=*/true), -1)
                << "a probe candidate was promoted into a captured graph on visit " << visit;

        EXPECT_EQ(tuner().resolution(key), nullptr)
            << "captured visits ran the exploration out of its budget";

        // Still owed its first candidate.
        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked), 0)
            << "a captured visit advanced the rotation";
    }

    // ----------------------------------------------------------------------
    // Heuristic queries.
    //
    // The caller keeps the returned algo and dispatches it itself, so a probe
    // promoted into it would be kept, and its launch cannot be paired with the
    // visit that produced it. A query is treated like a capture: served a measured
    // winner if there is one, otherwise its own ranking, with nothing registered,
    // counted or offered.
    // ----------------------------------------------------------------------

    // However often the same query comes back.
    TEST(OnlineTuner, AHeuristicQueryIsNeverOfferedAProbeHoweverOftenItComesBack)
    {
        constexpr size_t key = 0x1013;

        const size_t           problem = identityOf(key);
        const std::vector<int> ranked{151, 152, 153};

        for(int visit = 0; visit < 512; ++visit)
            ASSERT_EQ(tuner().selectCandidate(key, problem, ranked, false, /*retained=*/true), -1)
                << "a probe was promoted into an answer the caller keeps, on query " << visit;

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[0], nullptr, start, stop))
            << "a launch of a heuristic answer was timed";

        EXPECT_EQ(tuner().resolution(key), nullptr)
            << "heuristic queries ran an exploration out of its budget";

        // The problem's own dispatches still register and are offered the first probe.
        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked), 0)
            << "heuristic queries left the problem unable to explore";
    }

    // A query and a dispatch of one problem carry different identities under the
    // same key (the query ranks with a dummy beta and the preference's workspace).
    // A query registers nothing, so a later dispatch is the registrant rather than a
    // sibling of the query.
    TEST(OnlineTuner, ADispatchIsNotASiblingOfAQueryThatOnlyAskedFirst)
    {
        constexpr size_t key = 0x1020;

        const size_t           asQueried    = identityOf(key, 0);
        const size_t           asDispatched = identityOf(key, 1);
        const std::vector<int> ranked{271, 272, 273};

        // First sight, and a visit that comes back, both from the query.
        ASSERT_EQ(tuner().selectCandidate(key, asQueried, ranked, false, /*retained=*/true), -1);
        tuner().selectCandidate(key, asQueried, ranked, false, /*retained=*/true);

        EXPECT_EQ(tuner().selectCandidate(key, asDispatched, ranked), 0)
            << "the problem's own dispatch was treated as a merged sibling of the query";

        const int calls = exploreToResolution(key, asDispatched, ranked);
        ASSERT_GT(calls, 0) << "exploration never ended";
        EXPECT_LE(calls, kVisitBudget + 1)
            << "the dispatch did not drive the exploration, something else ended it";
    }

    // ----------------------------------------------------------------------
    // Problems the library stops being asked about.
    //
    // A problem whose registrant stops dispatching would otherwise never resolve,
    // and every dispatch under its key would keep paying for the locked lookups.
    // ----------------------------------------------------------------------

    TEST(OnlineTuner, AProblemSelectionStopsVisitingGivesUpAndPublishes)
    {
        constexpr size_t key = 0x1016;

        const size_t           problem = identityOf(key);
        const std::vector<int> ranked{181, 182, 183};

        // A probe whose launch never comes: the application moved on.
        ASSERT_EQ(tuner().selectCandidate(key, problem, ranked), 0);
        ASSERT_EQ(tuner().resolution(key), nullptr);

        // Dispatches under the key without a selection visit of the registrant.
        int dispatches = 0;
        for(; dispatches < kCallCeiling && !tuner().resolution(key); ++dispatches)
            tuner().harvestPending(key);

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr) << "a problem nothing can measure never gave up";
        EXPECT_EQ(resolved->winner(), -1) << "nothing was measured, so there is no winner";

        // Bounded by abandonBudget(): the counter also advances on siblings'
        // dispatches, so a tighter ceiling would end healthy explorations early.
        EXPECT_GT(dispatches, kTopK * kRepeats);
        EXPECT_LE(dispatches, kAbandonBudget + 2) << "the budget did not bound it";

        // Published, so both entry points answer from the table.
        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked), -1);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[0], nullptr, start, stop));
    }

    // Every call of an ordinary exploration also reaches the measurement hook; a
    // selection visit resets the run, so those calls do not count against it.
    TEST(OnlineTuner, DispatchesBetweenSelectionVisitsDoNotShortenAnExploration)
    {
        constexpr size_t selectionOnlyKey = 0x1017;
        constexpr size_t dispatchingKey   = 0x1018;

        const std::vector<int> ranked{191, 192, 193};

        const int selectionOnly
            = exploreToResolution(selectionOnlyKey, identityOf(selectionOnlyKey), ranked);
        ASSERT_GT(selectionOnly, 0) << "exploration never ended";

        int withDispatches = 0;
        for(; withDispatches < kCallCeiling && !tuner().resolution(dispatchingKey);)
        {
            ++withDispatches;
            tuner().selectCandidate(dispatchingKey, identityOf(dispatchingKey), ranked);
            tuner().harvestPending(dispatchingKey);
        }

        ASSERT_NE(tuner().resolution(dispatchingKey), nullptr) << "exploration never ended";
        EXPECT_EQ(withDispatches, selectionOnly)
            << "the dispatches of an ordinary exploration shortened its own budget";
    }

    // A merged sibling neither ends an exploration early nor keeps one alive.
    // Its dispatches spend none of the registrant's visit budget, but they do
    // advance the abandonment counter, so a registrant that stops visiting is still
    // given up on. Driven from this thread in a fixed order so the interleaving is
    // deterministic; the concurrent case follows.
    TEST(OnlineTuner, AMergedSiblingEndsAnExplorationLateAndNeverEarly)
    {
        constexpr size_t key = 0x1019;

        const size_t registering = identityOf(key, 0);
        const size_t merged      = identityOf(key, 1);

        const std::vector<int> ranked{201, 202, 203};

        ASSERT_EQ(tuner().selectCandidate(key, registering, ranked), 0);

        // The window between the two ceilings.
        ASSERT_GT(kEarlyEnough, kVisitBudget);
        ASSERT_LT(kEarlyEnough, kAbandonBudget);

        for(int visit = 0; visit < kEarlyEnough; ++visit)
        {
            ASSERT_EQ(tuner().selectCandidate(key, merged, ranked), -1)
                << "a merged sibling was handed a probe candidate on visit " << visit;
            tuner().harvestPending(key);
        }

        EXPECT_EQ(tuner().resolution(key), nullptr)
            << "a merged sibling's dispatches ran the exploration out of its budget";

        // Still owed its first candidate.
        EXPECT_EQ(tuner().selectCandidate(key, registering, ranked), 0);

        // The registrant stops visiting while the sibling keeps dispatching.
        int siblingVisits = 0;
        for(; siblingVisits < kCallCeiling && !tuner().resolution(key); ++siblingVisits)
        {
            tuner().selectCandidate(key, merged, ranked);
            tuner().harvestPending(key);
        }

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr)
            << "a merged sibling held an abandoned exploration open for ever";
        EXPECT_EQ(resolved->winner(), -1) << "nothing was measured, so there is no winner";

        // And it was the abandonment ceiling that ended it, not the visit budget.
        EXPECT_GT(siblingVisits, kVisitBudget) << "the registrant's budget ended it";
    }

    // The same rule under real threads: a shape no caller is exploring must end
    // under every interleaving of sibling traffic.
    TEST(OnlineTuner, ConcurrentMergedSiblingsCannotKeepADeadExplorationAlive)
    {
        constexpr size_t key = 0x101C;

        const size_t registering = identityOf(key, 0);

        const std::vector<int> ranked{241, 242, 243};

        // Registered and offered a candidate, then never dispatched.
        ASSERT_EQ(tuner().selectCandidate(key, registering, ranked), 0);

        constexpr int kSiblings = 4;

        std::atomic<bool> resolved{false};
        std::atomic<int>  visits{0};

        std::vector<std::thread> siblings;
        siblings.reserve(kSiblings);

        for(int id = 0; id < kSiblings; ++id)
        {
            siblings.emplace_back([&, id] {
                const size_t merged = identityOf(key, id + 1);

                for(int visit = 0; visit < kCallCeiling && !resolved.load(); ++visit)
                {
                    // Never a probe, whoever else is inside the tuner.
                    EXPECT_EQ(tuner().selectCandidate(key, merged, ranked), -1)
                        << "a merged sibling was handed a probe candidate";
                    tuner().harvestPending(key);
                    visits.fetch_add(1);

                    if(tuner().resolution(key))
                        resolved.store(true);
                }
            });
        }

        for(std::thread& sibling : siblings)
            sibling.join();

        ASSERT_TRUE(resolved.load())
            << "concurrent siblings held an abandoned exploration open for " << visits.load()
            << " dispatches";

        const OnlineTuner::Resolution* answer = tuner().resolution(key);
        ASSERT_NE(answer, nullptr);
        EXPECT_EQ(answer->winner(), -1) << "nothing was measured, so there is no winner";
    }

    // A resolved problem releases its samples and in-flight record; the problem
    // map never erases, so holding them would add up.
    TEST(OnlineTuner, AResolvedProblemGivesBackItsExplorationStorage)
    {
        constexpr size_t key = 0x101A;

        const std::vector<int> ranked{211, 212, 213};

        ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0);
        ASSERT_GT(OnlineTunerTestAccess::exploringBytes(tuner(), key), 0u)
            << "an exploring problem was holding nothing to measure into";

        ASSERT_GT(exploreToResolution(key, identityOf(key), ranked), 0) << "never resolved";

        EXPECT_EQ(OnlineTunerTestAccess::exploringBytes(tuner(), key), 0u)
            << "a resolved problem is still holding its exploration storage";

        // Born resolved, with nothing to compare.
        constexpr size_t single = 0x101B;

        ASSERT_EQ(tuner().selectCandidate(single, identityOf(single), {221}), -1);
        ASSERT_NE(tuner().resolution(single), nullptr);

        EXPECT_EQ(OnlineTunerTestAccess::exploringBytes(tuner(), single), 0u)
            << "a problem born resolved is holding exploration storage it never used";
    }

    // Last of the host-only cases, because filling the table evicts every key
    // above it. A problem resolved after as many problems as the table has slots
    // must still publish: the table is direct-mapped, so slots are shared, not
    // spent.
    TEST(OnlineTuner, AProblemResolvedAfterTheTableIsFullStillPublishes)
    {
        // Aligned so the loop covers each slot exactly once.
        constexpr size_t base = 0x40000;
        static_assert(base % kResolutionSlots == 0, "the fill would not cover the table");

        for(size_t slot = 0; slot < kResolutionSlots; ++slot)
        {
            const size_t key = base + slot;
            ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), {301}), -1) << slot;
        }

        ASSERT_NE(tuner().resolution(base), nullptr) << "the table never filled";

        constexpr size_t late = base + kResolutionSlots + 7;

        ASSERT_EQ(tuner().selectCandidate(late, identityOf(late), {302}), -1);

        EXPECT_NE(tuner().resolution(late), nullptr)
            << "a problem that resolved past the slot count was refused a published entry";

        // The remaining boundary: the late key took its colliding slot, and the
        // evicted key falls back to the locked path rather than another's winner.
        EXPECT_EQ(tuner().resolution(base + 7), nullptr) << "a collision answered on the slot";
    }

    // ----------------------------------------------------------------------
    // Cases that need a device: a candidate's issue count only moves when a launch
    // is offered an event pair, and the pool is itself under test.
    // ----------------------------------------------------------------------

    int deviceCount()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess ? count : 0;
    }

    // A stream and a scrap buffer to bracket between a measurement's two events:
    // back-to-back records can read exactly 0.0 ms, which the sign test rejects.
    class Device
    {
    public:
        explicit Device(int id)
            : m_id(id)
        {
            EXPECT_EQ(hipSetDevice(id), hipSuccess);
            EXPECT_EQ(hipStreamCreate(&m_stream), hipSuccess);
            EXPECT_EQ(hipMalloc(&m_scratch, c_scratchBytes), hipSuccess);
        }

        ~Device()
        {
            static_cast<void>(hipSetDevice(m_id));
            static_cast<void>(hipFree(m_scratch));
            static_cast<void>(hipStreamDestroy(m_stream));
        }

        Device(const Device&)            = delete;
        Device& operator=(const Device&) = delete;

        int id() const
        {
            return m_id;
        }
        hipStream_t stream() const
        {
            return m_stream;
        }

        // What the launch does for the library: record both ends around some work,
        // return, and leave the pair for the next harvest.
        void bracket(size_t key, hipEvent_t start, hipEvent_t stop) const
        {
            EXPECT_EQ(hipEventRecord(start, m_stream), hipSuccess);
            EXPECT_EQ(hipMemsetAsync(m_scratch, 0, c_scratchBytes, m_stream), hipSuccess);
            EXPECT_EQ(hipEventRecord(stop, m_stream), hipSuccess);
            tuner().endMeasurement(key, start);
            EXPECT_EQ(hipStreamSynchronize(m_stream), hipSuccess);
        }

    private:
        static constexpr size_t c_scratchBytes = 4u << 20;

        int         m_id      = 0;
        hipStream_t m_stream  = nullptr;
        void*       m_scratch = nullptr;
    };

    // One whole visit: take the offer, time the launch, read it back. Returns the
    // position offered.
    int sampleOneVisit(size_t                  key,
                       size_t                  problem,
                       const std::vector<int>& ranked,
                       const Device&           device)
    {
        const int position = tuner().selectCandidate(key, problem, ranked);
        if(position < 0)
            return position;

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_TRUE(tuner().beginMeasurement(key, ranked[position], device.stream(), start, stop))
            << "the tuner refused the launch it had just asked for";

        if(start && stop)
        {
            device.bracket(key, start, stop);
            tuner().harvestPending(key);
        }

        return position;
    }

    // One dispatch as the library makes it: harvest, select, time the launch. The
    // pair stays outstanding until the key's next dispatch. Returns whether the
    // launch was timed.
    bool dispatchOnce(size_t                  key,
                      size_t                  problem,
                      const std::vector<int>& ranked,
                      const Device&           device)
    {
        tuner().harvestPending(key);

        const int position = tuner().selectCandidate(key, problem, ranked);
        if(position < 0)
            return false;

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        if(!tuner().beginMeasurement(key, ranked[position], device.stream(), start, stop))
            return false;

        device.bracket(key, start, stop);
        return true;
    }

    // A stream held on a host-visible value until release(), so what is recorded
    // on it stays in flight for as long as a case needs.
    class HeldStream
    {
    public:
        static bool supported(int id)
        {
            int canWait = 0;
            return hipDeviceGetAttribute(&canWait, hipDeviceAttributeCanUseStreamWaitValue, id)
                       == hipSuccess
                   && canWait != 0;
        }

        explicit HeldStream(int id)
        {
            EXPECT_EQ(hipSetDevice(id), hipSuccess);
            EXPECT_EQ(hipStreamCreate(&m_stream), hipSuccess);
            EXPECT_EQ(hipHostMalloc(reinterpret_cast<void**>(&m_go), sizeof(*m_go), 0), hipSuccess);
            if(m_go)
            {
                *m_go = 0;
                EXPECT_EQ(hipStreamWaitValue32(m_stream, m_go, 1, hipStreamWaitValueEq),
                          hipSuccess);
            }
        }

        ~HeldStream()
        {
            release();
            static_cast<void>(hipStreamDestroy(m_stream));
            static_cast<void>(hipHostFree(m_go));
        }

        HeldStream(const HeldStream&)            = delete;
        HeldStream& operator=(const HeldStream&) = delete;

        // A launch's two records, behind the wait, and its return.
        void bracket(size_t key, hipEvent_t start, hipEvent_t stop) const
        {
            EXPECT_EQ(hipEventRecord(start, m_stream), hipSuccess);
            EXPECT_EQ(hipEventRecord(stop, m_stream), hipSuccess);
            tuner().endMeasurement(key, start);
        }

        void release()
        {
            if(m_go && !m_released)
                __atomic_store_n(m_go, 1u, __ATOMIC_SEQ_CST);
            m_released = true;
            EXPECT_EQ(hipStreamSynchronize(m_stream), hipSuccess);
        }

    private:
        hipStream_t m_stream   = nullptr;
        uint32_t*   m_go       = nullptr;
        bool        m_released = false;
    };

    // A capture that arrives part way through exploration, when the owed probe is
    // not the caller's own first choice, is refused it.
    TEST(OnlineTuner, ACaptureMidExplorationIsRefusedTheProbeAndLeavesItOwed)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);

        constexpr size_t key = 0x100e;

        const size_t           problem = identityOf(key);
        const std::vector<int> ranked{111, 112, 113};

        // nextCandidate() takes the least-issued candidate, so one visit short of a
        // full pass leaves the last candidate owed.
        for(int candidate = 0; candidate < kTopK - 1; ++candidate)
            ASSERT_EQ(sampleOneVisit(key, problem, ranked, device), candidate)
                << "the rotation did not advance as expected";

        ASSERT_EQ(tuner().resolution(key), nullptr) << "exploration ended early";

        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked, /*capturing=*/true), -1)
            << "probe candidate " << (kTopK - 1) << " was promoted into a captured graph";

        // The refused probe is still the one the next real dispatch is asked for.
        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked), kTopK - 1)
            << "a captured visit consumed the candidate it was refused";
    }

    // A resolved problem's measured winner is what a graph should get.
    TEST(OnlineTuner, ACaptureIsStillServedTheMeasuredWinner)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);

        constexpr size_t key = 0x100f;

        const size_t           problem = identityOf(key);
        const std::vector<int> ranked{121, 122, 123};

        for(int visit = 0; visit < kTopK * kRepeats; ++visit)
            ASSERT_GE(sampleOneVisit(key, problem, ranked, device), 0) << "visit " << visit;

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr) << "exploration never ended";
        ASSERT_GE(resolved->winner(), 0) << "nothing was measured, so there is no winner to serve";

        const int expected = static_cast<int>(
            std::find(ranked.begin(), ranked.end(), resolved->winner()) - ranked.begin());
        ASSERT_LT(expected, static_cast<int>(ranked.size()));

        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked, /*capturing=*/true), expected)
            << "a capture was denied the measured winner";
    }

    // Likewise for a heuristic query, once dispatches have measured a winner.
    TEST(OnlineTuner, AHeuristicQueryIsStillServedTheMeasuredWinner)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);

        constexpr size_t key = 0x1015;

        const size_t           problem = identityOf(key);
        const std::vector<int> ranked{171, 172, 173};

        for(int visit = 0; visit < kTopK * kRepeats; ++visit)
            ASSERT_GE(sampleOneVisit(key, problem, ranked, device), 0) << "visit " << visit;

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr) << "exploration never ended";
        ASSERT_GE(resolved->winner(), 0) << "nothing was measured, so there is no winner to serve";

        const int expected = static_cast<int>(
            std::find(ranked.begin(), ranked.end(), resolved->winner()) - ranked.begin());
        ASSERT_LT(expected, static_cast<int>(ranked.size()));

        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked, false, /*retained=*/true), expected)
            << "a heuristic query was denied the measured winner";
    }

    /**
     * A thread that runs one piece of work at a time, on demand.
     *
     * The offer is held on the thread that selection answered, so the cases
     * below vary which thread acts while run() fixes the order of steps. A
     * std::thread per step would not do: the offer must survive between
     * steps on the same thread.
     */
    class Worker
    {
    public:
        Worker()
            : m_thread([this] { loop(); })
        {
        }

        ~Worker()
        {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_stop = true;
            }
            m_wake.notify_one();
            m_thread.join();
        }

        Worker(const Worker&)            = delete;
        Worker& operator=(const Worker&) = delete;

        // Returns once the work is done.
        void run(std::function<void()> work)
        {
            std::packaged_task<void()> task(std::move(work));
            std::future<void>          done = task.get_future();

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_task = &task;
            }

            m_wake.notify_one();
            done.get();
        }

    private:
        void loop()
        {
            for(;;)
            {
                std::packaged_task<void()>* task = nullptr;

                {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_wake.wait(lock, [this] { return m_task != nullptr || m_stop; });

                    if(m_task == nullptr)
                        return;

                    task   = m_task;
                    m_task = nullptr;
                }

                (*task)();
            }
        }

        std::mutex                  m_mutex;
        std::condition_variable     m_wake;
        std::packaged_task<void()>* m_task = nullptr;
        bool                        m_stop = false;
        std::thread                 m_thread;
    };

    // A merged sibling on another thread cannot claim the registrant's offer, even
    // when its own pick is the offered solution, and the registrant can still claim
    // it afterwards.
    TEST(OnlineTuner, AnotherThreadCannotClaimTheOfferThisOneWasMade)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);

        constexpr size_t key = 0x101D;

        const size_t registering = identityOf(key, 0);
        const size_t merged      = identityOf(key, 1);

        const std::vector<int> ranked{251, 252, 253};

        Worker registrant;
        Worker sibling;

        registrant.run([&] {
            ASSERT_EQ(hipSetDevice(device.id()), hipSuccess);
            ASSERT_EQ(tuner().selectCandidate(key, registering, ranked), 0)
                << "the registrant was not offered its first candidate";
        });

        // The sibling's own first choice is the solution the registrant was offered.
        hipEvent_t stolenStart = nullptr;
        hipEvent_t stolenStop  = nullptr;

        sibling.run([&] {
            ASSERT_EQ(hipSetDevice(device.id()), hipSuccess);
            ASSERT_EQ(tuner().selectCandidate(key, merged, ranked), -1);

            EXPECT_FALSE(
                tuner().beginMeasurement(key, ranked[0], device.stream(), stolenStart, stolenStop))
                << "a merged sibling on another thread took the sample the "
                   "registrant's offer was made for";
        });

        // The offer is still there for the launch it was made to.
        registrant.run([&] {
            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;

            ASSERT_TRUE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop))
                << "the registrant was refused the launch its own offer named";

            device.bracket(key, start, stop);
            tuner().harvestPending(key);
        });

        // The sample landed, so the rotation moved on.
        EXPECT_EQ(tuner().selectCandidate(key, registering, ranked), 1)
            << "the registrant's sample was not credited";
    }

    // A sibling's visit between every one of the registrant's, from another
    // thread, must not cancel the registrant's offers and starve it of a winner.
    TEST(OnlineTuner, SiblingVisitsDoNotStarveTheRegistrantOfItsWinner)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);

        constexpr size_t key = 0x101E;

        const size_t registering = identityOf(key, 0);
        const size_t merged      = identityOf(key, 1);

        const std::vector<int> ranked{261, 262, 263};

        Worker sibling;

        ASSERT_EQ(hipSetDevice(device.id()), hipSuccess);

        int visits = 0;
        for(; visits < kCallCeiling && !tuner().resolution(key); ++visits)
        {
            const int position = tuner().selectCandidate(key, registering, ranked);

            sibling.run([&] {
                ASSERT_EQ(hipSetDevice(device.id()), hipSuccess);
                ASSERT_EQ(tuner().selectCandidate(key, merged, ranked), -1);
            });

            if(position < 0)
                continue;

            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;

            ASSERT_TRUE(
                tuner().beginMeasurement(key, ranked[position], device.stream(), start, stop))
                << "a sibling's visit cancelled the offer the registrant held, on visit " << visits;

            device.bracket(key, start, stop);
            tuner().harvestPending(key);
        }

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr) << "exploration never ended";
        EXPECT_GE(resolved->winner(), 0)
            << "sibling traffic starved a measurable shape into resolving with no winner";
        EXPECT_GT(resolved->winnerTimeUs(), 0.0f) << "a winner with no measured time";
    }

    // A thread holding an offer from before reloadForTest() cannot claim it
    // against the key's new registration, even for the same candidate.
    TEST(OnlineTuner, AnOfferFromAnEarlierRegistrationOfTheKeyIsNotClaimed)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);

        constexpr size_t       key     = 0x1027A;
        const size_t           problem = identityOf(key);
        const std::vector<int> ranked{241, 242, 243};

        Worker stale;
        stale.run([&] {
            ASSERT_EQ(hipSetDevice(device.id()), hipSuccess);
            ASSERT_EQ(tuner().selectCandidate(key, problem, ranked), 0);
        });

        tuner().reloadForTest();

        // Registered again from this thread, which is offered the same candidate.
        ASSERT_EQ(tuner().selectCandidate(key, problem, ranked), 0);

        stale.run([&] {
            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            const bool taken
                = tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop);
            EXPECT_FALSE(taken) << "an offer made before the key was registered again was claimed";
            if(taken)
                tuner().endMeasurement(key, start);
        });

        // The current registration's own offer is still good.
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        ASSERT_TRUE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop));
        device.bracket(key, start, stop);
        tuner().harvestPending(key);
        EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), key), 1u);

        tuner().reloadForTest();
    }

    // ----------------------------------------------------------------------
    // The harvest's sign test.
    //
    // A pair queries as complete whether or not a launch recorded it, so a sample
    // is kept only when the elapsed time is positive. Pairs are pooled inverted and
    // only after a sample, so every way of missing a record reads non-positive. A
    // rejected pair is retired, not pooled: pooling it would invert it twice.
    // ----------------------------------------------------------------------

    // A sample is taken and its pair goes back to the pool start-for-stop.
    TEST(OnlineTuner, AnAcceptedPairIsPooledInverted)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);

        constexpr size_t       key = 0x1022;
        const std::vector<int> ranked{291, 292, 293};

        ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        ASSERT_TRUE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop));
        device.bracket(key, start, stop);
        tuner().harvestPending(key);

        const auto pooled = OnlineTunerTestAccess::pooledPairs(tuner(), device.id());
        ASSERT_FALSE(pooled.empty()) << "a pair that produced a sample was not pooled";
        EXPECT_EQ(pooled.back().first, stop) << "the pair was pooled the way round it was measured";
        EXPECT_EQ(pooled.back().second, start)
            << "the pair was pooled the way round it was measured";

        EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), key), 1u) << "the sample was not kept";
    }

    // Each way a launch can fail to record its pair is dropped, credits nothing,
    // and retires the pair.
    TEST(OnlineTuner, EveryUnrecordedPairIsDroppedAndRetired)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id = device.id();

        // Nothing outstanding from earlier cases.
        tuner().reloadForTest();

        const std::vector<int> ranked{301, 302, 303};

        // One measurement of key, recorded as record() says, then harvested. Returns
        // whether the pair handed out came from the pool.
        auto measure = [&](size_t key, const std::function<void(hipEvent_t, hipEvent_t)>& record) {
            const size_t pooledBefore = OnlineTunerTestAccess::pooledPairs(tuner(), id).size();

            EXPECT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0) << key;

            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            EXPECT_TRUE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop))
                << key;

            record(start, stop);
            tuner().endMeasurement(key, start);
            EXPECT_EQ(hipStreamSynchronize(device.stream()), hipSuccess);

            const int liveBefore = OnlineTunerTestAccess::livePairs(tuner(), id);
            tuner().harvestPending(key);

            // The harvest runs inside the caller's matmul, so it must leave no error in
            // the thread's HIP state.
            EXPECT_EQ(hipGetLastError(), hipSuccess)
                << "key " << key << ": the harvest left an error in the caller's HIP state";

            EXPECT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), liveBefore - 1)
                << "key " << key << ": a rejected pair was not retired";

            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), key), 0u)
                << "key " << key << ": a pair no launch recorded was taken as a sample";

            return pooledBefore > 0;
        };

        auto recordNothing   = [](hipEvent_t, hipEvent_t) { };
        auto recordStartOnly = [&](hipEvent_t start, hipEvent_t) {
            EXPECT_EQ(hipEventRecord(start, device.stream()), hipSuccess);
        };

        // A pooled pair first, so put one there with a real sample.
        {
            constexpr size_t warm = 0x1023;
            ASSERT_EQ(tuner().selectCandidate(warm, identityOf(warm), ranked), 0);

            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            ASSERT_TRUE(tuner().beginMeasurement(warm, ranked[0], device.stream(), start, stop));
            device.bracket(warm, start, stop);
            tuner().harvestPending(warm);
        }

        // Take all but one pooled pair, so the next pair handed out is known.
        size_t drain = 0x1100;
        while(OnlineTunerTestAccess::pooledPairs(tuner(), id).size() > 1)
        {
            ASSERT_LT(drain, 0x1100 + static_cast<size_t>(kLivePairCap))
                << "the pool held more pairs than the cap, or none was taken";
            ASSERT_EQ(tuner().selectCandidate(drain, identityOf(drain), ranked), 0);
            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            ASSERT_TRUE(tuner().beginMeasurement(drain, ranked[0], device.stream(), start, stop));
            ++drain;
        }

        // A reused pooled pair with nothing recorded reads negative.
        EXPECT_TRUE(measure(0x1024, recordNothing)) << "the pooled case did not get a pooled pair";

        // A pooled pair with only its start recorded reads an older stop against a
        // newer start.
        {
            constexpr size_t warm = 0x1025;
            ASSERT_EQ(tuner().selectCandidate(warm, identityOf(warm), ranked), 0);

            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            ASSERT_TRUE(tuner().beginMeasurement(warm, ranked[0], device.stream(), start, stop));
            device.bracket(warm, start, stop);
            tuner().harvestPending(warm);
        }
        EXPECT_TRUE(measure(0x1026, recordStartOnly))
            << "the pooled case did not get a pooled pair";

        // Fresh pairs from here on, holding only the inverted records made at
        // creation.
        ASSERT_TRUE(OnlineTunerTestAccess::pooledPairs(tuner(), id).empty());
        EXPECT_FALSE(measure(0x1027, recordNothing)) << "the fresh case got a pooled pair";
        EXPECT_FALSE(measure(0x1028, recordStartOnly)) << "the fresh case got a pooled pair";

        // None of the rejected pairs went back to the pool.
        EXPECT_TRUE(OnlineTunerTestAccess::pooledPairs(tuner(), id).empty())
            << "a rejected pair was pooled again";

        tuner().reloadForTest();
    }

    // A device never holds more than livePairCap() pairs. Launches held in flight
    // keep the device at the cap; past it the launch runs unmeasured, and nothing
    // in flight is read back early.
    TEST(OnlineTuner, LiveEventPairsOnADeviceAreCapped)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";
        if(!HeldStream::supported(0))
            GTEST_SKIP() << "the device cannot hold a stream on a host value";

        const Device device(0);
        const int    id = device.id();

        // Retire what earlier cases left outstanding, so only pooled pairs are live.
        tuner().reloadForTest();

        const int cap = tuner().livePairCap();
        ASSERT_GT(cap, 0);

        constexpr size_t       base = 0x70000;
        const std::vector<int> ranked{281, 282, 283};

        HeldStream held(id);

        int measured = 0;
        for(int shape = 0; shape < cap + 8; ++shape)
        {
            const size_t key = base + static_cast<size_t>(shape);
            ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0) << shape;

            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            if(!tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop))
            {
                EXPECT_EQ(start, nullptr) << "a declined launch was handed an event";
                EXPECT_EQ(stop, nullptr) << "a declined launch was handed an event";
                continue;
            }

            ++measured;
            held.bracket(key, start, stop);
        }

        EXPECT_EQ(measured, cap) << "launches in flight held more live event pairs than the cap";
        EXPECT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), cap);

        held.release();

        for(int shape = 0; shape < cap; ++shape)
        {
            const size_t key = base + static_cast<size_t>(shape);
            tuner().harvestPending(key);
            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), key), 1u)
                << "shape " << shape << ": a launch still in flight was read back early";
        }

        // Those harvests pooled the pairs, so a new shape is measured again.
        const size_t late = base + static_cast<size_t>(cap) + 8;
        ASSERT_EQ(tuner().selectCandidate(late, identityOf(late), ranked), 0);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        ASSERT_TRUE(tuner().beginMeasurement(late, ranked[0], device.stream(), start, stop))
            << "a pair handed back to the pool could not be measured with";
        device.bracket(late, start, stop);

        // Leave nothing outstanding for the cases after this one.
        tuner().reloadForTest();
    }

    // One-shot shapes hold the whole cap with finished launches. A shape that keeps
    // coming back is still explored to a winner, because at the cap finished pairs
    // are read back and reused, and each one-shot shape keeps its sample.
    TEST(OnlineTuner, AtTheCapTheFinishedPairsOfOneShotShapesAreReclaimed)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id = device.id();

        tuner().reloadForTest();

        const int cap = tuner().livePairCap();

        constexpr size_t       base = 0x71000;
        const std::vector<int> ranked{271, 272, 273};

        for(int shape = 0; shape < cap; ++shape)
        {
            const size_t key = base + static_cast<size_t>(shape);
            ASSERT_TRUE(dispatchOnce(key, identityOf(key), ranked, device)) << shape;
        }

        ASSERT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), cap);
        ASSERT_TRUE(OnlineTunerTestAccess::pooledPairs(tuner(), id).empty());

        constexpr size_t repeated = 0x72000;

        int visits = 0;
        while(!tuner().resolution(repeated) && visits < kCallCeiling)
        {
            EXPECT_TRUE(dispatchOnce(repeated, identityOf(repeated), ranked, device) || visits > 0)
                << "the first launch of a new shape at the cap was not measured";
            ++visits;
        }

        const OnlineTuner::Resolution* resolved = tuner().resolution(repeated);
        ASSERT_NE(resolved, nullptr);
        EXPECT_GE(resolved->winner(), 0) << "a shape that kept coming back was never measured";
        EXPECT_EQ(visits, kTopK * kRepeats + 1) << "exploration took longer than an uncapped one";

        for(int shape = 0; shape < cap; ++shape)
            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), base + static_cast<size_t>(shape)),
                      1u)
                << "shape " << shape << ": the sample its launch bought was not kept";

        EXPECT_LE(OnlineTunerTestAccess::livePairs(tuner(), id), cap);

        tuner().reloadForTest();
    }

    // A reclaim pass walks only problems that may hold a pair: resolved shapes and
    // shapes whose pairs it took back are dropped from the list.
    TEST(OnlineTuner, AReclaimPassWalksOnlyTheProblemsHoldingPairs)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id = device.id();

        tuner().reloadForTest();

        const int              cap = tuner().livePairCap();
        const std::vector<int> ranked{201, 202, 203};

        constexpr size_t resolvedBase = 0x7B000;
        constexpr int    resolvedKeys = 64;
        for(int shape = 0; shape < resolvedKeys; ++shape)
        {
            const size_t key = resolvedBase + static_cast<size_t>(shape);
            for(int visit = 0; visit < kCallCeiling && !tuner().resolution(key); ++visit)
                sampleOneVisit(key, identityOf(key), ranked, device);
            ASSERT_NE(tuner().resolution(key), nullptr) << shape;
        }

        // The device at the cap, every pair held by a finished one-shot launch.
        size_t key = 0x7C000;
        for(int shape = 0; !OnlineTunerTestAccess::pooledPairs(tuner(), id).empty()
                           || OnlineTunerTestAccess::livePairs(tuner(), id) < cap;
            ++shape, ++key)
        {
            ASSERT_LT(shape, 2 * cap) << "the device never reached the cap";
            ASSERT_TRUE(dispatchOnce(key, identityOf(key), ranked, device));
        }

        // One more shape: its pair comes from a reclaim pass.
        ASSERT_TRUE(dispatchOnce(key, identityOf(key), ranked, device))
            << "the reclaim pass found nothing to take back";

        const std::vector<size_t> walked = OnlineTunerTestAccess::holders(tuner());
        EXPECT_LE(walked.size(), static_cast<size_t>(cap));
        for(size_t holder : walked)
            EXPECT_TRUE(holder < resolvedBase || holder >= resolvedBase + resolvedKeys)
                << "resolved key " << holder << " is still walked";

        tuner().reloadForTest();
    }

    // More shapes in rotation than the cap has pairs for all explore at once and
    // resolve with a winner, because at the cap the others' finished pairs are
    // reused.
    TEST(OnlineTuner, MoreShapesInRotationThanTheCapAllResolveWithWinners)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id = device.id();

        tuner().reloadForTest();

        const int cap = tuner().livePairCap();

        // More shapes than waves of the cap's size could reach within the budget.
        const int shapes = cap * OnlineTuner::c_visitBudgetFactor + cap;

        constexpr size_t       base = 0x73000;
        const std::vector<int> ranked{261, 262, 263};

        int rounds = 0;
        for(int unresolved = shapes; unresolved > 0 && rounds < kCallCeiling; ++rounds)
        {
            unresolved = 0;
            for(int shape = 0; shape < shapes; ++shape)
            {
                const size_t key = base + static_cast<size_t>(shape);
                if(tuner().resolution(key))
                    continue;

                dispatchOnce(key, identityOf(key), ranked, device);
                if(!tuner().resolution(key))
                    ++unresolved;
            }
        }

        int starved = 0;
        for(int shape = 0; shape < shapes; ++shape)
        {
            const OnlineTuner::Resolution* resolved
                = tuner().resolution(base + static_cast<size_t>(shape));
            ASSERT_NE(resolved, nullptr) << shape;
            if(resolved->winner() < 0)
                ++starved;
        }

        EXPECT_EQ(starved, 0) << "of " << shapes << " shapes in rotation, " << starved
                              << " resolved with no winner";
        EXPECT_EQ(rounds, kTopK * kRepeats + 1) << "the shapes did not all explore at once";
        EXPECT_LE(OnlineTunerTestAccess::livePairs(tuner(), id), cap);

        tuner().reloadForTest();
    }

    // beginMeasurement() hands a pair out under the lock and the launch records it
    // after the lock is released. A reclaim must not touch such a pair: taking it
    // would hand one pair to two launches, and retiring it would destroy a handle
    // about to be recorded.
    TEST(OnlineTuner, APairHandedOutAndNotYetRecordedIsNeverReclaimed)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id = device.id();

        tuner().reloadForTest();

        const int cap = tuner().livePairCap();

        const std::vector<int> ranked{251, 252, 253};

        // At the cap with every pair pooled, so what is handed out below is a pooled,
        // inverted pair.
        std::vector<size_t> filled;
        for(size_t key = 0x74000; !OnlineTunerTestAccess::pooledPairs(tuner(), id).empty()
                                  || OnlineTunerTestAccess::livePairs(tuner(), id) < cap;
            ++key)
        {
            ASSERT_TRUE(dispatchOnce(key, identityOf(key), ranked, device));
            filled.push_back(key);
        }
        for(size_t key : filled)
            tuner().harvestPending(key);
        ASSERT_EQ(OnlineTunerTestAccess::pooledPairs(tuner(), id).size(), static_cast<size_t>(cap));

        // All of them handed out for launches that have not recorded yet.
        struct Taken
        {
            size_t     key;
            hipEvent_t start;
            hipEvent_t stop;
        };
        std::vector<Taken> taken;
        for(int shape = 0; shape < cap; ++shape)
        {
            const size_t key = 0x75000 + static_cast<size_t>(shape);
            ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0);

            Taken pair{key, nullptr, nullptr};
            ASSERT_TRUE(
                tuner().beginMeasurement(key, ranked[0], device.stream(), pair.start, pair.stop));
            taken.push_back(pair);
        }

        // At the cap with nothing pooled, so the reclaim runs and must take nothing.
        constexpr size_t late = 0x76000;
        ASSERT_EQ(tuner().selectCandidate(late, identityOf(late), ranked), 0);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(late, ranked[0], device.stream(), start, stop))
            << "a pair handed out for a launch not yet recorded was taken back";
        EXPECT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), cap)
            << "a pair handed out for a launch not yet recorded was retired";

        // Each launch records its pair and its own harvest finds the sample.
        for(const Taken& pair : taken)
        {
            device.bracket(pair.key, pair.start, pair.stop);
            tuner().harvestPending(pair.key);
            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), pair.key), 1u)
                << "key " << pair.key << " lost the launch it had a pair for";
        }

        tuner().reloadForTest();
    }

    // Two threads dispatching one shape: one holds a pair it has not recorded yet
    // while the other harvests the shape. The pair reads like an unrecorded one,
    // but must not be retired before its launch returns, since recording a
    // destroyed event is a use after free. Twice, so at least one pair is pooled.
    TEST(OnlineTuner, AnotherDispatchOfTheShapeLeavesAPairItsLaunchHasNotRecorded)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id = device.id();

        tuner().reloadForTest();

        constexpr size_t       key     = 0x79000;
        const size_t           problem = identityOf(key);
        const std::vector<int> ranked{221, 222, 223};

        Worker other;

        for(size_t round = 1; round <= 2; ++round)
        {
            SCOPED_TRACE("round " + std::to_string(round));

            const int position = tuner().selectCandidate(key, problem, ranked);
            ASSERT_GE(position, 0);

            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            ASSERT_TRUE(
                tuner().beginMeasurement(key, ranked[position], device.stream(), start, stop));

            // Whatever the pair holds is complete, so a harvest that read it would get an
            // answer rather than "outstanding".
            ASSERT_EQ(hipStreamSynchronize(device.stream()), hipSuccess);

            const int live = OnlineTunerTestAccess::livePairs(tuner(), id);
            other.run([&] {
                ASSERT_EQ(hipSetDevice(id), hipSuccess);
                tuner().harvestPending(key);
            });

            // Asserted before the record, which on a destroyed pair would fault.
            ASSERT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), live)
                << "another dispatch retired a pair whose launch had not recorded it";

            device.bracket(key, start, stop);
            other.run([&] { tuner().harvestPending(key); });

            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), key), round)
                << "the launch's sample was lost";
        }

        tuner().reloadForTest();
    }

    // Several threads dispatching one shape at a time through many shapes. Nothing
    // is bracketed, so most samples are rejected and retired, putting retires next
    // to records. Returns 0, 3 if a record failed, or 4 if the shapes did not all
    // resolve within the step bound.
    int dispatchOneShapeFromManyThreads()
    {
        tuner().reloadForTest();

        constexpr int    kThreads = 8;
        constexpr int    kSteps   = 20000;
        constexpr size_t kFirst   = 0x7A000;
        constexpr size_t kLast    = kFirst + 200;

        const std::vector<int> ranked{211, 212, 213};

        std::atomic<size_t>      current{kFirst};
        std::atomic<int>         failures{0};
        std::vector<std::thread> threads;

        for(int thread = 0; thread < kThreads; ++thread)
            threads.emplace_back([&] {
                hipStream_t stream = nullptr;
                if(hipSetDevice(0) != hipSuccess || hipStreamCreate(&stream) != hipSuccess)
                {
                    ++failures;
                    return;
                }

                for(int step = 0; step < kSteps; ++step)
                {
                    const size_t key = current.load();
                    if(key >= kLast)
                        break;

                    tuner().harvestPending(key);

                    const int position = tuner().selectCandidate(key, identityOf(key), ranked);

                    hipEvent_t start = nullptr;
                    hipEvent_t stop  = nullptr;
                    if(position >= 0
                       && tuner().beginMeasurement(key, ranked[position], stream, start, stop))
                    {
                        std::this_thread::yield();
                        if(hipEventRecord(start, stream) != hipSuccess
                           || hipEventRecord(stop, stream) != hipSuccess)
                            ++failures;
                        tuner().endMeasurement(key, start);
                    }

                    size_t expected = key;
                    if(tuner().resolution(key))
                        current.compare_exchange_strong(expected, key + 1);
                }

                if(hipStreamSynchronize(stream) != hipSuccess)
                    ++failures;
                static_cast<void>(hipStreamDestroy(stream));
            });

        for(std::thread& thread : threads)
            thread.join();

        return failures.load() != 0 ? 3 : current.load() < kLast ? 4 : 0;
    }

    // The same race left to the scheduler, in a child process because losing it
    // faults rather than failing an assertion.
    TEST(OnlineTuner, ThreadsDispatchingOneShapeNeverRecordARetiredPair)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        GTEST_FLAG_SET(death_test_style, "threadsafe");
        EXPECT_EXIT(std::exit(dispatchOneShapeFromManyThreads()), ::testing::ExitedWithCode(0), "");
    }

    // The reclaim runs inside the caller's matmul, so reading fresh pairs whose
    // launches returned without recording them must leave no error in the
    // thread's HIP state. Such pairs are not taken; their own harvests retire them.
    TEST(OnlineTuner, AReclaimLeavesNoErrorInTheCallersHipState)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id = device.id();

        tuner().reloadForTest();

        const int              cap = tuner().livePairCap();
        const std::vector<int> ranked{231, 232, 233};

        // Empty the pool: a pooled pair whose launch records nothing reads negative
        // and is retired by its own harvest.
        size_t key = 0x78000;
        for(int taken = 0; !OnlineTunerTestAccess::pooledPairs(tuner(), id).empty(); ++taken)
        {
            ASSERT_LT(taken, cap) << "the pool held more pairs than the cap, or none was taken";
            ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0);
            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            ASSERT_TRUE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop));
            tuner().endMeasurement(key, start);
            tuner().harvestPending(key);
            ++key;
        }
        ASSERT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), 0);
        ASSERT_EQ(hipStreamSynchronize(device.stream()), hipSuccess);

        // The whole cap in fresh pairs whose launches returned without recording.
        std::vector<size_t> keys;
        for(int shape = 0; shape < cap; ++shape, ++key)
        {
            ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0);
            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            ASSERT_TRUE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop));
            tuner().endMeasurement(key, start);
            keys.push_back(key);
        }
        ASSERT_EQ(hipStreamSynchronize(device.stream()), hipSuccess);
        ASSERT_EQ(hipGetLastError(), hipSuccess);

        // A new shape at the cap: the reclaim reads all of them.
        ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0);
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop))
            << "a pair no launch recorded was taken back as a sample";
        EXPECT_EQ(hipGetLastError(), hipSuccess)
            << "the reclaim left an error in the caller's HIP state";
        EXPECT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), cap)
            << "the reclaim retired a pair";

        for(size_t owner : keys)
        {
            tuner().harvestPending(owner);
            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), owner), 0u)
                << "key " << owner << ": a pair no launch recorded was taken as a sample";
        }
        EXPECT_EQ(hipGetLastError(), hipSuccess);
        EXPECT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), 0)
            << "a pair no launch recorded was not retired by its own harvest";

        tuner().reloadForTest();
    }

    // A launch untimed because every pair on its device is in flight is reported
    // once per device. Reaching the cap and reclaiming is not reported.
    TEST(OnlineTuner, ALaunchUntimedAtTheCapIsReportedOncePerDevice)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";
        if(!HeldStream::supported(0))
            GTEST_SKIP() << "the device cannot hold a stream on a host value";

        const Device device(0);
        const int    id = device.id();

        tuner().reloadForTest();
        capReports().clear();

        const int              cap = tuner().livePairCap();
        const std::vector<int> ranked{241, 242, 243};

        // At the cap with finished launches: reclaimed, nothing to report.
        size_t key = 0x77000;
        for(int shape = 0; shape <= cap; ++shape, ++key)
            ASSERT_TRUE(dispatchOnce(key, identityOf(key), ranked, device)) << shape;
        EXPECT_TRUE(capReports().empty()) << "reaching the cap and reclaiming was reported";

        // Every pair in flight, twice over: one report, for this device.
        for(int hold = 0; hold < 2; ++hold)
        {
            HeldStream held(id);

            int untimed = 0;
            for(int shape = 0; shape < cap + 3; ++shape, ++key)
            {
                ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0);

                hipEvent_t start = nullptr;
                hipEvent_t stop  = nullptr;
                if(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop))
                    held.bracket(key, start, stop);
                else
                    ++untimed;
            }
            EXPECT_GE(untimed, 3) << "the device never ran out of pairs";

            ASSERT_EQ(capReports().size(), 1u) << "hold " << hold;
            EXPECT_EQ(capReports()[0], std::make_pair(id, cap));
        }

        tuner().reloadForTest();
    }

    // The pool is keyed by device: hipEventRecord rejects another device's event
    // with hipErrorInvalidResourceHandle, which the launch would then carry.
    TEST(OnlineTuner_multi_gpu, APooledEventPairIsNotHandedToAnotherDevice)
    {
        if(deviceCount() < 2)
            GTEST_SKIP() << "needs two HIP devices";

        const Device zero(0);
        const Device one(1);

        constexpr size_t keyZero = 0x1010;
        constexpr size_t keyOne  = 0x1011;

        const std::vector<int> ranked{131, 132, 133};

        // Device 0 pools one pair, inverted.
        ASSERT_EQ(hipSetDevice(zero.id()), hipSuccess);
        hipEvent_t startZero = nullptr;
        hipEvent_t stopZero  = nullptr;
        ASSERT_EQ(tuner().selectCandidate(keyZero, identityOf(keyZero), ranked), 0);
        ASSERT_TRUE(
            tuner().beginMeasurement(keyZero, ranked[0], zero.stream(), startZero, stopZero));
        zero.bracket(keyZero, startZero, stopZero);
        tuner().harvestPending(keyZero);

        // Device 1 now explores a shape of its own.
        ASSERT_EQ(hipSetDevice(one.id()), hipSuccess);
        hipEvent_t startOne = nullptr;
        hipEvent_t stopOne  = nullptr;
        ASSERT_EQ(tuner().selectCandidate(keyOne, identityOf(keyOne), ranked), 0);
        ASSERT_TRUE(tuner().beginMeasurement(keyOne, ranked[0], one.stream(), startOne, stopOne));

        EXPECT_NE(startOne, startZero) << "device 1 was handed device 0's pair";
        EXPECT_NE(startOne, stopZero) << "device 1 was handed device 0's pair";
        EXPECT_NE(stopOne, startZero) << "device 1 was handed device 0's pair";
        EXPECT_NE(stopOne, stopZero) << "device 1 was handed device 0's pair";

        // Whatever device 1 was given must be recordable there.
        EXPECT_EQ(hipEventRecord(startOne, one.stream()), hipSuccess)
            << "the pair handed to device 1 cannot be recorded there";
        EXPECT_EQ(hipEventRecord(stopOne, one.stream()), hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(one.stream()), hipSuccess);

        ASSERT_EQ(hipSetDevice(zero.id()), hipSuccess);
    }

} // namespace

// The configuration this file's private copy of the tuner is built from, so the
// tests pin it to known values whatever the environment says.
rocblaslt::OnlineTunerConfig rocblaslt::onlineTunerConfig()
{
    OnlineTunerConfig config;
    config.enabled     = true;
    config.topK        = kTopK;
    config.repeats     = kRepeats;
    config.livePairCap = kLivePairCap;

    config.livePairCapReached = recordCapReached;
    return config;
}
