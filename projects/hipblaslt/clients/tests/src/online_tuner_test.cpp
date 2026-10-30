// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the online tuner's bookkeeping: what registering a problem does,
// how exploration ends, and what the lock-free resolution table answers. The
// tuner carries no Tensile types, so like the store tests these need neither
// the solution library nor the hooks the rest of hipblaslt-test uses.
//
// They run inside hipblaslt-test, against a private copy of the tuner:
// OnlineTuner.cpp is compiled into the test binary alongside this file, and
// the onlineTunerConfig() at the bottom of this file is what that copy is
// built from. The library's own tuner is a different object. libhipblaslt
// builds with hidden visibility and exports neither the class nor
// onlineTunerConfig(), so the two copies never meet, and nothing here changes
// what the library's tuner does in the cases that drive it through the API.
// See clients/tests/src/CMakeLists.txt.
//
// Most need no device either, because they are decided before any HIP call.
// The ones gathered at the end of the file do: advancing the rotation means
// issuing a candidate, issuing one means acquiring an event pair, and the
// event pool is itself under test. Those skip where there is no device rather
// than passing for the wrong reason. The one case that needs two devices is
// in the OnlineTuner_multi_gpu suite, so CI runs it in its multi_gpu tier
// rather than skipping it on every single-GPU runner.

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
 * Declared by OnlineTuner.hpp and defined only here.
 *
 * What a resolved problem has stopped holding is not something it answers, and
 * every other case in this file goes through the public surface. This is the
 * exception, it only reads, and it exists because the alternative way to see a
 * few hundred bytes a problem gave back is to measure the heap.
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
    // What this test's own onlineTunerConfig() hands the singleton. The library
    // maps these off the tuning mode and TuningPolicy, neither of which is
    // linked here, so supplying them is also what keeps this test host-only.
    constexpr int kTopK    = 3;
    constexpr int kRepeats = 2;

    // Far below the library's default, so a case reaches the cap with a
    // handful of shapes rather than thousands.
    constexpr int kLivePairCap = 16;

    // The host-only cases below record no event, so no candidate collects a
    // sample and exploration can only end by running out its visit budget.
    // Bounded well above that budget, so a tuner that never gives up fails an
    // assertion instead of hanging the suite.
    constexpr int kCallCeiling = 4096;

    // The tuner's two ceilings, in the terms it computes them in.
    // visitBudget() bounds the visits the REGISTRANT may spend and
    // abandonBudget() bounds how long a problem nobody is exploring stays
    // unresolved.
    constexpr int kVisitBudget   = kTopK * kRepeats * OnlineTuner::c_visitBudgetFactor;
    constexpr int kAbandonBudget = kVisitBudget * OnlineTuner::c_abandonBudgetFactor;

    // A run of sibling dispatches long enough to be well past the registrant's
    // budget and well short of the abandonment ceiling.
    constexpr int kEarlyEnough = 512;

    // The resolution table is direct-mapped over the key, so two keys this far
    // apart share a slot.
    constexpr size_t kResolutionSlots = OnlineTuner::c_resolutionSlots;

    OnlineTuner& tuner()
    {
        return OnlineTuner::getInstance();
    }

    // What the tuner has told this file's livePairCapReached callback, in
    // order: the device, and the cap it was at.
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
     * Stands in for std::hash<ContractionProblemGemm>, the problem identity the
     * library hands the tuner alongside the resolution key.
     *
     * The two are deliberately different granularities: a resolution key covers
     * every problem the prediction model ranks the same way, so problems that
     * differ only in a leading dimension, an epilogue, a stride or a scaling
     * mode share one. Every call below that varies this while holding the key
     * fixed is one of those merged siblings, and that is the case these tests
     * exist for.
     */
    size_t identityOf(size_t key, int variant = 0)
    {
        return key * 1000003u + static_cast<size_t>(variant);
    }

    /**
     * A solution pointer the tuner can carry but nothing can dereference.
     *
     * ContractionSolution is forward-declared in the tuner's header and the
     * solution library is not linked here, which is the whole point: the tuner
     * only ever copies this pointer and hands it back. A non-owning alias over
     * a byte of this translation unit is a solution as far as any of that is
     * concerned, and owns nothing to destroy.
     */
    std::shared_ptr<TensileLite::ContractionSolution> placeholderSolution()
    {
        static char storage = 0;
        return std::shared_ptr<TensileLite::ContractionSolution>(
            std::shared_ptr<void>(), reinterpret_cast<TensileLite::ContractionSolution*>(&storage));
    }

    // Offers the same ranking on behalf of the same problem until it resolves,
    // and reports how many offers that took, or -1 if it never did.
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

        // No cold visits, so every test below registers its problem on the
        // first offer.
        EXPECT_EQ(tuner().coldCalls(), 0);
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

        EXPECT_EQ(tuner().selectCandidate(key, identityOf(key), {}), -1);
        EXPECT_EQ(tuner().resolution(key), nullptr);
    }

    // One candidate is nothing to choose between, so the problem is born
    // resolved with no winner and every later call takes the read-only path.
    TEST(OnlineTuner, ASingleCandidateResolvesOnSightWithNoWinner)
    {
        constexpr size_t key = 0x1003;

        EXPECT_EQ(tuner().selectCandidate(key, identityOf(key), {11}), -1);

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr);
        EXPECT_EQ(resolved->winner(), -1);

        // Nothing was measured, so there is no time to report and no row worth
        // writing.
        EXPECT_FLOAT_EQ(resolved->winnerTimeUs(), 0.0f);
    }

    // Exploration promotes the first candidate that still owes samples and,
    // with nothing ever measuring one, gives up at the visit budget rather than
    // holding the write lock for the life of the process.
    TEST(OnlineTuner, ExplorationEndsAtTheVisitBudget)
    {
        constexpr size_t       key = 0x1004;
        const std::vector<int> ranked{21, 22, 23, 24};

        // Only the top K are candidates, and the first of them is promoted.
        EXPECT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0);
        EXPECT_EQ(tuner().resolution(key), nullptr);

        const int calls = exploreToResolution(key, identityOf(key), ranked);
        ASSERT_GT(calls, 0) << "exploration never ended";
        EXPECT_GT(calls, kTopK * kRepeats);

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr);
        EXPECT_EQ(resolved->winner(), -1);

        // Resolved, so the caller's ordering is left alone and no launch is
        // offered an event pair.
        EXPECT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), -1);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, 21, nullptr, start, stop));
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

    // A record is installed once, and pinWinner says which caller installed it.
    // Whoever writes the row has to be that caller and no other: the key on the
    // record is the one problem the winner was measured for, while the callers
    // that lose the race carry keys nothing measured.
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

    // The selection hook registers a problem and the measurement hook only
    // measures one it registered, so a key the two disagree on is refused
    // rather than measured blind.
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
    // The resolution key covers the fields the prediction model ranks on, so
    // problems differing only in a leading dimension or an epilogue share one.
    // Both move a kernel's runtime, so their launches are not interchangeable
    // measurements, and resolve() compares a candidate's samples against the
    // other candidates' as though they were. The three tests below are the
    // three places that has to be stopped: who is offered a candidate, whose
    // launch is timed, and whose key a record is built from.
    // ----------------------------------------------------------------------

    // Exploration belongs to the problem that registered it. A sibling visiting
    // the same key is left its own ordering -- and spends none of the visit
    // budget, which it could otherwise run out, ending exploration early on
    // however many samples had been taken by then.
    TEST(OnlineTuner, AMergedSiblingIsOfferedNoCandidate)
    {
        constexpr size_t key = 0x1009;

        // Same m, n, k and types; different leading dimension.
        const size_t registering = identityOf(key, 0);
        const size_t merged      = identityOf(key, 1);

        const std::vector<int> ranked{71, 72, 73};

        EXPECT_EQ(tuner().selectCandidate(key, registering, ranked), 0);

        // Comfortably past the visit budget, which is candidates * repeats * a
        // slack factor and so cannot be more than a few dozen here.
        for(int visit = 0; visit < 512; ++visit)
            ASSERT_EQ(tuner().selectCandidate(key, merged, ranked), -1)
                << "a merged sibling was handed a probe candidate on visit " << visit;

        EXPECT_EQ(tuner().resolution(key), nullptr)
            << "a merged sibling's visits ran the exploration out of its budget";

        // Still owed its first candidate, which is what nothing having been
        // credited to it looks like from here.
        EXPECT_EQ(tuner().selectCandidate(key, registering, ranked), 0);
    }

    // Refusing the sibling a candidate is not enough on its own. It still
    // dispatches its own ranking's pick, and when two merged problems rank
    // alike that pick is a candidate of this exploration's that still owes
    // samples -- so matching a launch on the solution index alone would take
    // the sample anyway and credit one problem with the other's launch.
    //
    // The refusal itself is decided before any HIP call, so it holds with or
    // without a device. What it separates it from does not: a launch is only
    // accepted if an event pair can be created, so on a machine with no HIP
    // device this passes for the wrong reason. Judge it where one exists.
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

        // Candidate 0 still owes every one of its samples, so the next visit of
        // the registering problem is offered it again. Had the sibling's launch
        // been credited to it, the rotation would have moved on.
        EXPECT_EQ(tuner().selectCandidate(key, registering, ranked), 0)
            << "a merged sibling's launch advanced this problem's rotation";
    }

    // A launch of a candidate this visit was not asked for is not a sample of
    // it either, whoever issues it: the offer names one candidate and buys one
    // sample of it.
    TEST(OnlineTuner, OnlyTheCandidateThisVisitWasOfferedIsSampled)
    {
        constexpr size_t key = 0x100b;

        const std::vector<int> ranked{84, 85, 86};

        ASSERT_EQ(tuner().selectCandidate(key, identityOf(key), ranked), 0);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;

        // Still owes all its samples, and is still not what was asked for.
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[1], nullptr, start, stop))
            << "a candidate the rotation did not ask for was timed";
        EXPECT_EQ(start, nullptr);
        EXPECT_EQ(stop, nullptr);
    }

    // The key a row is written under belongs to the problem that was measured.
    //
    // OnlyOneCallerInstallsTheRecord above proves something narrower -- that
    // one caller installs and the rest are told they did not -- which the
    // sequential case cannot tell apart from this one, because there the
    // installer and the measured problem are the same caller. Here the merged
    // sibling arrives FIRST and its ranking does hold the winner, so nothing
    // about its own list refuses it and installer-only writes is satisfied
    // either way. What separates the two is whose key ends up on the record.
    TEST(OnlineTuner, OnlyTheMeasuredProblemInstallsTheRecord)
    {
        constexpr size_t key = 0x100c;

        const size_t measured = identityOf(key, 0);
        const size_t merged   = identityOf(key, 1);

        ASSERT_GT(exploreToResolution(key, measured, {94, 95, 96}), 0) << "exploration never ended";

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr);
        ASSERT_EQ(resolved->pinned(), nullptr);

        // One resolution key, two file keys, differing in exactly the field the
        // key merges and the row separates.
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
    // A capture is not a dispatch. It writes a kernel choice down once and the
    // caller replays it for the life of the graph, never coming back through
    // selection -- so a probe promoted during capture is a knowingly
    // non-optimal kernel with no path to correction. It also buys nothing: the
    // measurement hook refuses a capturing stream, so the probe is never timed.
    //
    // Serving a RESOLVED problem is the opposite case and is kept: a measured
    // winner is precisely what belongs in a graph. The two tests below are
    // those two halves.
    // ----------------------------------------------------------------------

    // A capture is offered nothing while exploration is still running, and
    // costs the exploration nothing by being refused -- the same shape of rule
    // as the merged sibling above, and for the same two reasons.
    TEST(OnlineTuner, CapturesAreOfferedNoCandidateAndSpendNoVisitBudget)
    {
        constexpr size_t key = 0x100d;

        const size_t           problem = identityOf(key);
        const std::vector<int> ranked{101, 102, 103};

        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked), 0);

        // Comfortably past the visit budget, which is candidates * repeats * a
        // slack factor and so cannot be more than a few dozen here.
        for(int visit = 0; visit < 512; ++visit)
            ASSERT_EQ(tuner().selectCandidate(key, problem, ranked, /*capturing=*/true), -1)
                << "a probe candidate was promoted into a captured graph on visit " << visit;

        EXPECT_EQ(tuner().resolution(key), nullptr)
            << "captured visits ran the exploration out of its budget";

        // Still owed its first candidate, which is what the captures having
        // cost it nothing looks like from here.
        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked), 0)
            << "a captured visit advanced the rotation";
    }

    // ----------------------------------------------------------------------
    // Heuristic queries.
    //
    // hipblasLtMatmulAlgoGetHeuristic hands the caller an algo and the caller
    // dispatches it itself, for as long as it likes. A probe promoted into
    // that answer is kept for as long as the caller keeps the algo, which for
    // the ordinary framework pattern -- query once, dispatch forever -- is the
    // life of the process. And it is never measured: the library cannot pair
    // the launch of a caller-supplied algo with the visit that handed it out,
    // so that launch goes past both measurement hooks.
    //
    // So a query is treated as a capture is: served a measured winner once
    // there is one, and otherwise its own ranking, with nothing registered,
    // counted or offered.
    // ----------------------------------------------------------------------

    // However often it comes back. A query that asked again used to be taken
    // as an ordinary visit and offered the rotation's next probe -- a probe
    // its own launch could never be measured for, and that a merged sibling's
    // caller-supplied launch on the same thread could claim instead.
    TEST(OnlineTuner, AHeuristicQueryIsNeverOfferedAProbeHoweverOftenItComesBack)
    {
        constexpr size_t key = 0x1013;

        const size_t           problem = identityOf(key);
        const std::vector<int> ranked{151, 152, 153};

        for(int visit = 0; visit < 512; ++visit)
            ASSERT_EQ(tuner().selectCandidate(key, problem, ranked, false, /*retained=*/true), -1)
                << "a probe was promoted into an answer the caller keeps, on query " << visit;

        // Nothing to claim, so the launch of what it was handed is not timed.
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[0], nullptr, start, stop))
            << "a launch of a heuristic answer was timed";

        EXPECT_EQ(tuner().resolution(key), nullptr)
            << "heuristic queries ran an exploration out of its budget";

        // And the problem's own dispatches are not held back by any of it:
        // the first one registers and is offered the first probe.
        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked), 0)
            << "heuristic queries left the problem unable to explore";
    }

    // A heuristic query and a dispatch of the same problem are not the same
    // problem to the solution library. The query ranks under a dummy beta and
    // the preference's workspace ceiling, the dispatch under its own beta and
    // allocation, so the identity each brings differs while the resolution key
    // is the same. Whoever registers owns the exploration, and a query that
    // registered first used to make every such dispatch a merged sibling of it:
    // offered nothing, sampled by nothing, and holding the exclusive lock on
    // every visit until the abandonment ceiling gave up with no winner. A
    // query registers nothing now, so the dispatch is the registrant.
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
    // Exploration advances at the selection hook, and a caller holding an algo
    // it fetched once never reaches it. Such a problem would otherwise sit one
    // visit short of its budget for ever: never resolving, never publishing,
    // and so costing every dispatch of that shape the two locked lookups a
    // published resolution answers without a lock.
    // ----------------------------------------------------------------------

    TEST(OnlineTuner, AProblemSelectionStopsVisitingGivesUpAndPublishes)
    {
        constexpr size_t key = 0x1016;

        const size_t           problem = identityOf(key);
        const std::vector<int> ranked{181, 182, 183};

        // One selection visit, offered a probe whose launch never comes: the
        // application has moved on from the shape.
        ASSERT_EQ(tuner().selectCandidate(key, problem, ranked), 0);
        ASSERT_EQ(tuner().resolution(key), nullptr);

        // Dispatches under the key that reach the measurement hook without a
        // selection visit of the registrant -- a merged sibling's, say.
        int dispatches = 0;
        for(; dispatches < kCallCeiling && !tuner().resolution(key); ++dispatches)
            tuner().harvestPending(key);

        const OnlineTuner::Resolution* resolved = tuner().resolution(key);
        ASSERT_NE(resolved, nullptr) << "a problem nothing can measure never gave up";
        EXPECT_EQ(resolved->winner(), -1) << "nothing was measured, so there is no winner";

        // Bounded by abandonBudget(), not by the registrant's own visit budget:
        // this counter is advanced by every dispatch under the key, including a
        // merged sibling's, so a ceiling tight enough to bound an exploration's
        // progress would end a healthy one early. See the merged-sibling case.
        EXPECT_GT(dispatches, kTopK * kRepeats);
        EXPECT_LE(dispatches, kAbandonBudget + 2) << "the budget did not bound it";

        // And the point of publishing: both entry points answer from the table
        // now, without taking anything.
        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked), -1);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[0], nullptr, start, stop));
    }

    // The other side of the same counter. Every call of an ordinary
    // exploration reaches the measurement hook too, so counting those against
    // the budget would halve how long a problem has to finish -- which is the
    // margin GPU-bound shapes already need all of. A selection visit is what
    // resets the run.
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

    // A merged sibling spends none of the exploration's budget at the
    // selection hook, and its dispatches must spend none of it at the
    // measurement hook either. A sibling dispatching often enough would
    // otherwise end the exploration early on however many samples had been
    // taken by then -- which is the whole reason the sibling is exempt in the
    // first place.
    //
    // It must not do the opposite either, and the version of this case that
    // asserted only the first half is what kept an earlier defect in place: a
    // sibling's selection visit reset the give-up counter, so a registrant
    // that had stopped visiting could never be noticed, the problem stayed
    // unresolved, and every one of those sibling dispatches took the tuner's
    // process-wide mutex EXCLUSIVELY for the life of the process. Asserting
    // only "still exploring at 512" was satisfied by never ending at all.
    //
    // So the two halves are two ceilings, and both are asserted. The
    // registrant's own visit budget is small and a sibling cannot touch it;
    // the abandonment ceiling is much larger and is the one a sibling's
    // dispatches do advance, because what it measures is how long a shape
    // NOBODY is exploring may stay unresolved.
    //
    // Deterministic by construction rather than by timing: the interleaving
    // under test is "a sibling's visit between the registrant's", and driving
    // both from this thread in a fixed order is that interleaving with the
    // scheduler taken out. The concurrent case follows.
    TEST(OnlineTuner, AMergedSiblingEndsAnExplorationLateAndNeverEarly)
    {
        constexpr size_t key = 0x1019;

        const size_t registering = identityOf(key, 0);
        const size_t merged      = identityOf(key, 1);

        const std::vector<int> ranked{201, 202, 203};

        ASSERT_EQ(tuner().selectCandidate(key, registering, ranked), 0);

        // Comfortably past the registrant's visit budget and comfortably short
        // of the abandonment ceiling, which is the window the first half of
        // the rule lives in.
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

        // Still owed its first candidate: the sibling spent none of the
        // registrant's visits either.
        EXPECT_EQ(tuner().selectCandidate(key, registering, ranked), 0);

        // The other half. The registrant now stops visiting -- a shape the
        // application has moved on from -- while the sibling keeps
        // dispatching. Nothing can finish this
        // exploration, so it has to be given up on.
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

        // And it was the large ceiling that ended it, not the small one. If
        // these two numbers could be met by one budget there would be nothing
        // to get wrong here.
        EXPECT_GT(siblingVisits, kVisitBudget) << "the registrant's budget ended it";
    }

    // The same rule with the scheduler putting the visits in order instead of
    // this thread. The property is not about any one interleaving: a shape no
    // caller is exploring must end under EVERY interleaving of sibling
    // traffic, so this asserts termination with real threads and a bound.
    TEST(OnlineTuner, ConcurrentMergedSiblingsCannotKeepADeadExplorationAlive)
    {
        constexpr size_t key = 0x101C;

        const size_t registering = identityOf(key, 0);

        const std::vector<int> ranked{241, 242, 243};

        // The registrant registers, is offered its first candidate, and never
        // comes back -- the dispatch never arrives, so no sample ever will.
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

    // The samples and the in-flight record are what exploring costs, and a
    // problem map that never erases is what makes holding them afterwards add
    // up.
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

        // The other way a problem resolves: born with nothing to compare, so
        // the arrays are dead before it has answered once.
        constexpr size_t single = 0x101B;

        ASSERT_EQ(tuner().selectCandidate(single, identityOf(single), {221}), -1);
        ASSERT_NE(tuner().resolution(single), nullptr);

        EXPECT_EQ(OnlineTunerTestAccess::exploringBytes(tuner(), single), 0u)
            << "a problem born resolved is holding exploration storage it never used";
    }

    // Last of the host-only cases, because filling the table evicts every key
    // above it.
    //
    // Publishing used to stop once as many problems had resolved as the table
    // has slots. The table is direct-mapped, so a slot is shared and not spent:
    // the effect was that the problem after that, and every one after it, kept
    // a null resolution() for the life of the process and so took the full
    // locked path on every dispatch -- however free the slot it would have
    // landed in.
    TEST(OnlineTuner, AProblemResolvedAfterTheTableIsFullStillPublishes)
    {
        // Clear of every other key in the file, and aligned so the loop below
        // covers each slot exactly once.
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

        // And the boundary that does remain, which is the point of publishing
        // past the count rather than a cost of it: the late key took the slot
        // of the one it collides with, and that one is left to the locked path
        // rather than answered with somebody else's winner.
        EXPECT_EQ(tuner().resolution(base + 7), nullptr) << "a collision answered on the slot";
    }

    // ----------------------------------------------------------------------
    // Cases that need a device.
    //
    // Everything above is decided before any HIP call. These are not: a
    // candidate's issue count only moves when a launch is offered an event
    // pair, so the one case the self-limiting capture above is not -- a
    // capture arriving part way through an exploration, where the candidate
    // owed is no longer the caller's own rank 0 -- cannot be built without
    // one. Nor can anything about the pool itself.
    // ----------------------------------------------------------------------

    int deviceCount()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess ? count : 0;
    }

    // A stream and a scrap buffer to put between a measurement's two events.
    // Two back-to-back records can read exactly 0.0 ms, and the harvest's sign
    // test rejects that, so a sample needs something real bracketed.
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

        // What the kernel launch does for the library: record both ends around
        // something, return, and leave the pair readable by the next harvest.
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

    // One whole visit of a problem: take the offer, time the launch it asks
    // for, read it back. Returns the position offered.
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

    // One dispatch the way the library makes it: harvest this key's finished
    // launches first, then select, then time the launch selection asked for.
    // Unlike sampleOneVisit() the pair is left outstanding until the key's
    // next dispatch, which is what holds pairs between visits and what a shape
    // dispatched once never gives back. Returns whether the launch was timed.
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

    // A stream the GPU stops on until the host says go, so what is recorded on
    // it stays in flight for as long as a case needs: the stream waits on a
    // host-visible value that release() writes.
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

    // The case the capture test above cannot reach, and the only damaging one:
    // exploration has advanced past its first candidate, and a capture lands
    // on one the caller's own ranking does not put first. Freezing that into a
    // graph is a kernel the heuristic rejected and the tuner has not finished
    // judging, kept for the graph's lifetime.
    TEST(OnlineTuner, ACaptureMidExplorationIsRefusedTheProbeAndLeavesItOwed)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);

        constexpr size_t key = 0x100e;

        const size_t           problem = identityOf(key);
        const std::vector<int> ranked{111, 112, 113};

        // Run the rotation off candidate 0, so what is owed next is a probe
        // sitting at a position the caller did not rank first. nextCandidate()
        // takes the least-issued candidate, so the offers go 0, 1, 2, 0, 1, 2
        // and one visit short of a full pass leaves the last one owed.
        for(int candidate = 0; candidate < kTopK - 1; ++candidate)
            ASSERT_EQ(sampleOneVisit(key, problem, ranked, device), candidate)
                << "the rotation did not advance as expected";

        ASSERT_EQ(tuner().resolution(key), nullptr) << "exploration ended early";

        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked, /*capturing=*/true), -1)
            << "probe candidate " << (kTopK - 1) << " was promoted into a captured graph";

        // Nothing was credited to the capture, so the probe it was refused is
        // still the one the next real dispatch is asked for.
        EXPECT_EQ(tuner().selectCandidate(key, problem, ranked), kTopK - 1)
            << "a captured visit consumed the candidate it was refused";
    }

    // The other half of the rule. A resolved problem has a measured winner and
    // a graph is where that answer belongs: it is the kernel cache mode would
    // replay, picked the same way. Refusing a capture outright would hand it
    // the heuristic's rank 1 instead and throw the measurement away.
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

    // The same half of the rule for a heuristic query: once the problem's own
    // dispatches have measured a winner, a caller that asks once and keeps the
    // answer is handed it, which is the kernel cache mode would replay, and
    // keeps that.
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
     * The offer that pairs a selection visit with the launch it asked for is
     * held on the thread selection answered, so WHICH thread acts is the
     * variable the two cases below are about and WHEN it acts must not be.
     * Ordering the run() calls from the test body makes the interleaving total
     * and fixed: no sleeps, no retries, no spinning on a flag, and the same
     * sequence on every machine. A plain std::thread per step would not do, as
     * the offer has to survive from one of this thread's steps to the next.
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

        // Runs work on this worker and does not return until it is done, so
        // the caller's next statement is strictly after it.
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

    // The sample a sibling could steal.
    //
    // The offer used to be one slot on the problem, matched at the measurement
    // hook on the solution index alone, so any launch of that index under the
    // same resolution key claimed it -- and the key deliberately merges
    // problems that differ in epilogue, bias, strides or scaling, whose own
    // rank 0 is very often a candidate of this exploration's. The sample taken
    // was then a measurement of a different problem's kernel filed under this
    // problem's candidate: the pooled-samples defect, reached under
    // concurrency.
    //
    // Both halves are asserted, because the old slot got both wrong at once:
    // the thief must be refused, AND the launch the offer was made to must
    // still be able to claim it afterwards.
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

        // The sibling is refused a candidate of its own, as it always was, and
        // then dispatches its own rank 0 -- which is the solution index the
        // registrant was just offered.
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

        // And the offer is still there to be claimed by the launch it was made
        // to, which is the half a thief also destroys.
        registrant.run([&] {
            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;

            ASSERT_TRUE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop))
                << "the registrant was refused the launch its own offer named";

            device.bracket(key, start, stop);
            tuner().harvestPending(key);
        });

        // The sample landed, so the exploration moved on to the next candidate
        // rather than still owing the first one.
        EXPECT_EQ(tuner().selectCandidate(key, registering, ranked), 1)
            << "the registrant's sample was not credited";
    }

    // The same defect's other two halves, read
    // forwards instead of backwards: the slot was dropped at the top of every
    // selection visit, above the merged-sibling guard, so a sibling CANCELLED
    // an offer the registrant had not consumed yet. The registrant's launch
    // then went unsampled while its visit had already been spent, and enough
    // sibling traffic starved the exploration into resolving with no winner at
    // all -- for a shape that measures perfectly well.
    //
    // So this drives the whole exploration with a sibling's visit between
    // every one of the registrant's, from another thread, and asks for the one
    // thing starvation takes away: a winner.
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
            // Between the registrant's visit and its launch, every time.
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

    // An offer names one registration of a problem, not just its key. A
    // thread still holding an offer made before reloadForTest() must not be
    // able to claim it against the candidate list the same key registers
    // afterwards, even where the candidate it names is the same.
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

        // The same key and ranking registered again, from this thread, which
        // is offered the same candidate the stale thread holds.
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
    // A pair queries as complete whether or not a launch recorded it, so the
    // elapsed time is the only evidence that one did, and the harvest keeps a
    // sample only when it is positive. Every way of missing a record has to
    // read otherwise, and that holds because of how pairs are pooled: only a
    // pair that produced a sample goes back, and it goes back inverted. The
    // cases below are the ways of missing one, each of which must be dropped
    // without crediting the candidate, and must retire the pair rather than
    // pool it, since a rejected pair pooled again would be inverted twice.
    // ----------------------------------------------------------------------

    // The pairing the rest depends on, on one device: a sample is taken, and
    // its pair goes back to the pool start-for-stop.
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

    // Each way a launch can fail to record its pair, read back by the harvest.
    // Every one is dropped, credits nothing, and retires the pair.
    TEST(OnlineTuner, EveryUnrecordedPairIsDroppedAndRetired)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id = device.id();

        // Nothing outstanding from earlier cases, so the counts below move
        // only for what this case does.
        tuner().reloadForTest();

        const std::vector<int> ranked{301, 302, 303};

        // One measurement of key on this thread, recorded as record() says,
        // then harvested. Returns whether the pair it was handed came from
        // the pool.
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

            // The harvest runs inside the caller's matmul, so reading back a
            // pair no launch recorded must leave nothing in this thread's HIP
            // last-error state for the caller to take for its own.
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

        // Every pooled pair but one is taken by an outstanding measurement, so
        // what is handed out next is known.
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

        // A pooled pair reused with nothing recorded reads the negative of
        // the duration it last measured.
        EXPECT_TRUE(measure(0x1024, recordNothing)) << "the pooled case did not get a pooled pair";

        // A pooled pair whose start was recorded and whose launch then
        // failed reads its older stop against the newer start.
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

        // The pool is empty now, so the next pairs are fresh: one never
        // recorded by its launch at all, and one with only its start recorded.
        // Each still holds the inverted records it was created with.
        ASSERT_TRUE(OnlineTunerTestAccess::pooledPairs(tuner(), id).empty());
        EXPECT_FALSE(measure(0x1027, recordNothing)) << "the fresh case got a pooled pair";
        EXPECT_FALSE(measure(0x1028, recordStartOnly)) << "the fresh case got a pooled pair";

        // None of the rejected pairs went back to the pool.
        EXPECT_TRUE(OnlineTunerTestAccess::pooledPairs(tuner(), id).empty())
            << "a rejected pair was pooled again";

        tuner().reloadForTest();
    }

    // The bound the cap exists for: however many shapes are measured, a device
    // holds no more than livePairCap() pairs. A pair can only come back once
    // the launch it timed is finished, so the way to hold a device at the cap
    // is launches the GPU has not reached. Past it no new measurement starts;
    // the launch runs unmeasured and nothing about it changes. And nothing
    // still in flight is read back early: each launch's own harvest finds its
    // sample once the GPU gets there.
    TEST(OnlineTuner, LiveEventPairsOnADeviceAreCapped)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";
        if(!HeldStream::supported(0))
            GTEST_SKIP() << "the device cannot hold a stream on a host value";

        const Device device(0);
        const int    id = device.id();

        // Hands back what earlier cases left outstanding, so what is live now
        // is only what the pool holds, and every pooled pair is one a new
        // measurement can take without creating another.
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

    // Shapes dispatched once and never again hold the whole cap, with every
    // launch they timed long finished. A shape that keeps coming back must
    // still be explored to a winner: at the cap the finished launches are read
    // back and their pairs reused, rather than the cap ending tuning on the
    // device for the rest of the process. Each one-shot shape keeps the sample
    // its one launch bought.
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

    // A reclaim pass walks the problems that may hold a pair, not every problem
    // the tuner has seen: the shapes already resolved are dropped from what it
    // walks, and so are the ones whose pairs it took back.
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

    // More shapes in rotation than the cap has pairs for. Each holds its pair
    // only between its own visits, and at the cap the pairs of the others'
    // finished launches are reused, so every shape is explored at once and
    // resolves with a winner. Without that the shapes went through in waves
    // of the cap's size, each waiting shape spending its visit budget on
    // declined launches, and every shape past the waves that budget allows
    // resolved with no winner at all.
    TEST(OnlineTuner, MoreShapesInRotationThanTheCapAllResolveWithWinners)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id = device.id();

        tuner().reloadForTest();

        const int cap = tuner().livePairCap();

        // Past the last shape the waves could reach.
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

    // The window a reclaim must never reach into. beginMeasurement() hands a
    // pair out under the lock and the launch records it after the lock is
    // released, so on another thread a pair can be outstanding and not yet
    // recorded at the moment a dispatch at the cap looks for pairs to take
    // back. Such a pair must be left exactly where it is: taking it would hand
    // one pair to two launches, and retiring it would destroy a handle a launch
    // is about to record.
    TEST(OnlineTuner, APairHandedOutAndNotYetRecordedIsNeverReclaimed)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id = device.id();

        tuner().reloadForTest();

        const int cap = tuner().livePairCap();

        const std::vector<int> ranked{251, 252, 253};

        // Bring the device to the cap and pool every pair, so what is handed
        // out below is what a busy device hands out: a pooled pair, inverted.
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

        // Every one of them handed out for a launch that has not recorded yet.
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

        // A new shape at the cap with nothing pooled, so the reclaim runs, and
        // it must find nothing it may take.
        constexpr size_t late = 0x76000;
        ASSERT_EQ(tuner().selectCandidate(late, identityOf(late), ranked), 0);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(late, ranked[0], device.stream(), start, stop))
            << "a pair handed out for a launch not yet recorded was taken back";
        EXPECT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), cap)
            << "a pair handed out for a launch not yet recorded was retired";

        // Each launch now records its pair, and its own harvest finds the
        // sample: nothing was taken, pooled or destroyed underneath it.
        for(const Taken& pair : taken)
        {
            device.bracket(pair.key, pair.start, pair.stop);
            tuner().harvestPending(pair.key);
            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), pair.key), 1u)
                << "key " << pair.key << " lost the launch it had a pair for";
        }

        tuner().reloadForTest();
    }

    // Two dispatches of one shape on two threads. One has been handed a pair
    // and has not recorded it yet when the other harvests the shape. Such a
    // pair reads exactly like one whose launch failed to record it, and those
    // are retired -- but this one is about to be recorded, and recording a
    // destroyed event is a use after free inside the caller's matmul. It must
    // be left alone until its launch returns, and then be read back as usual.
    // Twice, so the pair handed out is a pooled one at least once.
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

            // Whatever the pair already holds is complete, so a harvest that
            // reads it gets an answer rather than "outstanding".
            ASSERT_EQ(hipStreamSynchronize(device.stream()), hipSuccess);

            const int live = OnlineTunerTestAccess::livePairs(tuner(), id);
            other.run([&] {
                ASSERT_EQ(hipSetDevice(id), hipSuccess);
                tuner().harvestPending(key);
            });

            // Asserted before the record below, which on a destroyed pair
            // would fault rather than fail.
            ASSERT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), live)
                << "another dispatch retired a pair whose launch had not recorded it";

            device.bracket(key, start, stop);
            other.run([&] { tuner().harvestPending(key); });

            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), key), round)
                << "the launch's sample was lost";
        }

        tuner().reloadForTest();
    }

    // Several threads dispatching one shape at a time, as a server's workers
    // do, through many shapes. Launches record nothing between their two
    // events, so most samples are rejected and their pairs retired, which is
    // what puts a retire next to a record. Returns 0, or 3 if a record failed,
    // or 4 if the shapes were not all resolved within the step bound.
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

    // The same race as above, left to the scheduler. In a child process,
    // because losing it records a destroyed event, which faults rather than
    // failing an assertion.
    TEST(OnlineTuner, ThreadsDispatchingOneShapeNeverRecordARetiredPair)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        GTEST_FLAG_SET(death_test_style, "threadsafe");
        EXPECT_EXIT(std::exit(dispatchOneShapeFromManyThreads()), ::testing::ExitedWithCode(0), "");
    }

    // The reclaim reads other shapes' pairs on the caller's thread, inside the
    // caller's matmul, so it must leave that thread's HIP state as it found it
    // -- including when what it reads is a fresh pair whose launch returned
    // without recording it, which holds only the records it was created with.
    // Such a pair is never taken, and its own harvest later retires it.
    TEST(OnlineTuner, AReclaimLeavesNoErrorInTheCallersHipState)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id = device.id();

        tuner().reloadForTest();

        const int              cap = tuner().livePairCap();
        const std::vector<int> ranked{231, 232, 233};

        // Empty the pool by retiring what it holds: a pooled pair whose launch
        // returns without recording it reads negative, and its own harvest
        // retires it.
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

        // The whole cap in fresh pairs, each handed to a launch that returned
        // without recording it.
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

        // A new shape at the cap: the reclaim reads every one of them.
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

    // A launch that goes untimed because its device is at the cap with every
    // pair in flight is reported, once per device: otherwise the only trace of
    // it is declined= on a winner line, which reads like a shape that was
    // measured. Reaching the cap and reclaiming is not reported, and a device
    // that has reported once stays quiet when it happens again.
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

    // The event pool is on a process-wide singleton and the device is per
    // caller. Keyed on nothing, one vector hands device 1 the pairs device 0
    // recycled -- and hipEventRecord rejects a pair belonging to another
    // device with hipErrorInvalidResourceHandle, measured on gfx950, which the
    // production launch then has to carry.
    TEST(OnlineTuner_multi_gpu, APooledEventPairIsNotHandedToAnotherDevice)
    {
        if(deviceCount() < 2)
            GTEST_SKIP() << "needs two HIP devices";

        const Device zero(0);
        const Device one(1);

        constexpr size_t keyZero = 0x1010;
        constexpr size_t keyOne  = 0x1011;

        const std::vector<int> ranked{131, 132, 133};

        // Device 0 takes one sample, and the harvest puts its pair in the
        // pool. recycleEvents() stores it inverted, so an unkeyed pool would
        // hand this pair's stop back as the next caller's start.
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

        // The assertion that does not depend on how the pool is organised:
        // whatever device 1 was given has to be recordable on device 1.
        EXPECT_EQ(hipEventRecord(startOne, one.stream()), hipSuccess)
            << "the pair handed to device 1 cannot be recorded there";
        EXPECT_EQ(hipEventRecord(stopOne, one.stream()), hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(one.stream()), hipSuccess);

        ASSERT_EQ(hipSetDevice(zero.id()), hipSuccess);
    }

} // namespace

// The configuration the tuner is built with, which inside hipBLASLt comes from
// the tuning mode and TuningPolicy. Supplied here instead, for the test
// binary's private copy of the tuner, so these tests pin the bookkeeping to
// known values whatever the environment says. The library's copy is built from
// its own definition and never sees this one.
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
