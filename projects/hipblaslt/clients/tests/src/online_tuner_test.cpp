// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the online tuner's bookkeeping: what registering a problem does,
// how exploration ends, and what the lock-free resolution table answers. The
// tuner carries no Tensile types, so like the store tests these need neither
// the solution library nor the hooks hipblaslt-test uses.
//
// Most need no device either, because they are decided before any HIP call.
// The ones gathered at the end of the file do: advancing the rotation means
// issuing a candidate, issuing one means acquiring an event pair, and the
// event pool is itself under test. Those skip where there is no device rather
// than passing for the wrong reason.

#include "OnlineTuner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <vector>

using rocblaslt::OnlineTuner;

namespace
{
    // What this test's own onlineTunerConfig() hands the singleton. The library
    // maps these off the tuning mode and TuningPolicy, neither of which is
    // linked here, so supplying them is also what keeps this test host-only.
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
        EXPECT_EQ(installed->m_requiredWorkspace, 256u);
        EXPECT_EQ(installed->m_searchWorkspace, 1024u);
        EXPECT_EQ(installed->m_tuningKey.m, 4096u);

        EXPECT_EQ(tuner().pinWinner(*resolved, solution, identityOf(key), 0, 0, sibling), nullptr)
            << "a second caller was told it had installed the record";
        EXPECT_EQ(resolved->pinned(), installed);
        EXPECT_EQ(resolved->pinned()->m_tuningKey.m, 4096u)
            << "a later caller restated the key the row is written under";
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
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[0], start, stop))
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
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[1], start, stop))
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
        EXPECT_EQ(installed->m_problem, measured);
        EXPECT_EQ(installed->m_tuningKey.colStrideA, 4096u)
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
        // something, and leave the pair readable by the next harvest.
        void bracket(hipEvent_t start, hipEvent_t stop) const
        {
            EXPECT_EQ(hipEventRecord(start, m_stream), hipSuccess);
            EXPECT_EQ(hipMemsetAsync(m_scratch, 0, c_scratchBytes, m_stream), hipSuccess);
            EXPECT_EQ(hipEventRecord(stop, m_stream), hipSuccess);
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
        EXPECT_TRUE(tuner().beginMeasurement(key, ranked[position], start, stop))
            << "the tuner refused the launch it had just asked for";

        if(start && stop)
        {
            device.bracket(start, stop);
            tuner().harvestPending(key);
        }

        return position;
    }

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

} // namespace

// The configuration the tuner is built with, which inside hipBLASLt comes from
// the tuning mode and TuningPolicy. Supplied here instead, so these tests pin
// the bookkeeping to known values and the tuner stays linkable without the
// library, a device or an environment.
rocblaslt::OnlineTunerConfig rocblaslt::onlineTunerConfig()
{
    OnlineTunerConfig config;
    config.enabled = true;
    config.topK    = kTopK;
    config.repeats = kRepeats;
    return config;
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
