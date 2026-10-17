// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests for the adaptive sm_count_target estimator and helpers in
// rocblaslt_adaptive_sm_count.hpp, reached by relative #include, and for the
// benchmark timing guard built on them. Launch scenarios drive
// Estimator::startLaunch/finishLaunch over a hand-filled probe slot array.

#include "../../../library/src/amd_detail/rocblaslt/src/include/rocblaslt_adaptive_sm_count.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{
    using namespace rocblaslt::adaptive_sm;

    constexpr uint32_t kCus       = 256;
    constexpr uint32_t kContended = 192; // CUs left by a cotenant

    struct Options
    {
        bool               canProbe = true;
        bool               recordOk = true; // the probe's event record succeeds
        unsigned long long stream   = 1;
    };

    struct Result
    {
        LaunchPlan plan;
        bool       finishChanged = false; // finishLaunch's return
        bool       changed       = false; // hint() differs from plan.oldHint
    };

    // One estimator, with a probe slot array the tests fill by hand.
    class HostUnitAdaptiveSmCount : public ::testing::Test
    {
    protected:
        Config                cfg;
        Estimator             est;
        std::vector<uint32_t> slots                   = std::vector<uint32_t>(c_probeSlots, 0);
        uint32_t              epoch                   = 0; // last probe epoch
        uint32_t              completed               = 0; // probes up to this epoch are done
        uint32_t              ringEpochs[c_probeRing] = {};
        // State of the probe at a ring position.
        std::function<ProbeState(uint32_t)> probeState = [this](uint32_t i) {
            return static_cast<int32_t>(completed - ringEpochs[i]) >= 0 ? ProbeState::Done
                                                                        : ProbeState::Pending;
        };

        // Probe `e` marked the first `n` CUs.
        void markCus(uint32_t e, uint32_t n)
        {
            std::fill_n(slots.begin(), n, e);
        }

        // One tracked launch; a planned probe takes the next epoch.
        Result launch(Options o = {})
        {
            Result r;
            r.plan = est.startLaunch(
                cfg,
                o.canProbe,
                [&](unsigned long long* id) {
                    *id = o.stream;
                    return true;
                },
                slots.data(),
                kCus,
                probeState);
            if(r.plan.probe)
                epoch = nextEpoch(epoch);
            r.finishChanged = est.finishLaunch(r.plan, r.plan.probe, epoch, [&](uint32_t i) {
                if(o.recordOk)
                    ringEpochs[i] = epoch;
                return o.recordOk;
            });
            r.changed       = est.hint() != r.plan.oldHint;
            return r;
        }

        // `n` launches, each recorded probe completing on `cus` CUs before the
        // next launch. Returns how many changed the hint.
        int runOn(uint32_t cus, uint32_t n, Options o = {})
        {
            int changes = 0;
            for(uint32_t i = 0; i < n; ++i)
            {
                const Result r = launch(o);
                changes += r.changed;
                if(r.plan.probe && o.recordOk)
                {
                    markCus(epoch, cus);
                    completed = epoch;
                }
            }
            return changes;
        }

        // `n` launches whose probes never complete.
        void runPending(uint32_t n, Options o = {})
        {
            for(uint32_t i = 0; i < n; ++i)
                launch(o);
        }

        // Launches until the next cadence point, that one included.
        int runToCadence(uint32_t cus, Options o = {})
        {
            return runOn(cus, cfg.probeEvery - est.sinceCadence(), o);
        }

        // Settles the hint at `cus` from a fresh estimator.
        void settleAt(uint32_t cus)
        {
            runOn(cus, cfg.probeWindow * cfg.probeEvery + 1);
            ASSERT_EQ(est.hint(), cus);
            ASSERT_EQ(est.missed(), 0u);
        }
    };

    TEST_F(HostUnitAdaptiveSmCount, smoke_Candidate)
    {
        struct Case
        {
            const char*           label;
            std::vector<uint32_t> counts;
            uint32_t              hint;
            uint32_t              expected;
        };
        const std::vector<Case> cases = {
            {"full GPU, no hint", {256, 256}, 0, 0},
            {"full GPU, from a hint", {256, 256}, 128, 0},
            {"below, agreeing: exact", {186, 186}, 0, 186},
            {"below, disagreeing: the smallest", {186, 200}, 0, 186},
            {"below a hint: the smallest", {100, 120}, 128, 100},
            {"above a hint: the smallest", {200, 230}, 128, 200},
            {"above a hint, one at nCu: the smallest", {200, 256}, 128, 200},
            {"both sides of a hint: holds", {100, 200}, 128, 128},
            {"one at the hint: holds", {128, 200}, 128, 128},
            {"full and below, no hint: holds", {256, 128}, 0, 0},
        };
        for(const Case& c : cases)
        {
            SCOPED_TRACE(c.label);
            EXPECT_EQ(candidate(c.counts.data(), uint32_t(c.counts.size()), c.hint, kCus),
                      c.expected);
        }
    }

    // Readings fed one per launch, each from the previous launch's probe.
    TEST_F(HostUnitAdaptiveSmCount, smoke_ReadingsToHint)
    {
        struct Case
        {
            const char*           label;
            uint32_t              window;
            std::vector<uint32_t> readings;
            uint32_t              expected;
        };
        const std::vector<Case> cases = {
            {"one short of the window: none", 2, {192}, 0},
            {"a full window: the smallest", 2, {192, 200}, 192},
            {"up from a hint: the window's smallest", 2, {192, 192, 200, 230}, 200},
            {"a zero reading is ignored", 2, {192, 0, 192}, 192},
            {"window 1: down", 1, {186, 150}, 150},
            {"window 1: full", 1, {186, 256}, 0},
            {"window 3: a mixed window holds", 3, {192, 192, 192, 100, 200}, 192},
        };
        for(const Case& c : cases)
        {
            SCOPED_TRACE(c.label);
            est             = {};
            cfg             = {};
            cfg.probeWindow = c.window;
            cfg.probeEvery  = 1;
            for(uint32_t count : c.readings)
                runOn(count, 1);
            launch();
            EXPECT_EQ(est.hint(), c.expected);
            EXPECT_EQ(est.probesRead(), c.readings.size());
        }
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_ProbeFallsDueEveryProbeEveryLaunches)
    {
        const uint32_t        rounds = 3;
        std::vector<uint32_t> probes, expected;
        for(uint32_t l = 0; l < rounds * cfg.probeEvery; ++l)
        {
            if(launch().plan.probe)
            {
                probes.push_back(l);
                completed = epoch;
            }
            if((l + 1) % cfg.probeEvery == 0)
                expected.push_back(l);
        }
        EXPECT_EQ(probes, expected);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_DueProbeGoesToTheNextLaunchThatCanProbe)
    {
        runOn(kCus, cfg.probeEvery - 1);
        const Result cadence = launch({.canProbe = false});
        EXPECT_TRUE(cadence.plan.cadence);
        EXPECT_FALSE(cadence.plan.probe);
        EXPECT_TRUE(est.due());
        EXPECT_EQ(est.missed(), 1u);
        EXPECT_TRUE(launch().plan.probe);
        EXPECT_EQ(est.missed(), 0u);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_ContentionSetsTheHintAfterAFullWindow)
    {
        EXPECT_EQ(runOn(kContended, cfg.probeWindow * cfg.probeEvery), 0);
        EXPECT_EQ(est.hint(), 0u);
        EXPECT_TRUE(launch().changed);
        EXPECT_EQ(est.hint(), kContended);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_FullReadingsClearTheHint)
    {
        settleAt(kContended);
        EXPECT_EQ(runOn(kCus, cfg.probeWindow * cfg.probeEvery + 1), 1);
        EXPECT_EQ(est.hint(), 0u);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_MissedProbesResetTheHint)
    {
        settleAt(kContended);
        for(uint32_t m = 1; m < c_resetMisses; ++m)
            runToCadence(kContended, {.canProbe = false});
        EXPECT_EQ(est.missed(), c_resetMisses - 1);
        runOn(kContended, cfg.probeEvery - 1, {.canProbe = false});
        EXPECT_EQ(est.hint(), kContended);
        const Result last = launch({.canProbe = false});
        EXPECT_TRUE(last.finishChanged);
        EXPECT_EQ(est.hint(), 0u);
        EXPECT_EQ(est.readingCount(), 0u);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_ProbeBetweenMissesRestartsTheCount)
    {
        settleAt(kContended);
        runToCadence(kContended, {.canProbe = false});
        ASSERT_EQ(est.missed(), 1u);
        runOn(kContended, 1);
        EXPECT_EQ(est.missed(), 0u);
        runToCadence(kContended, {.canProbe = false});
        EXPECT_EQ(est.missed(), 1u);
        EXPECT_EQ(est.hint(), kContended);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_FailedRecordsAddUpToAMiss)
    {
        settleAt(kContended);
        runToCadence(kContended, {.recordOk = false});
        for(uint32_t f = 1; f < c_eventFailuresPerMiss; ++f)
        {
            EXPECT_EQ(est.missed(), 0u) << "after " << f << " failed records";
            launch({.recordOk = false});
        }
        EXPECT_EQ(est.missed(), 1u);
        EXPECT_EQ(est.outstanding(), 0u);
        EXPECT_TRUE(est.due());
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_FailedRecordsResetTheHint)
    {
        settleAt(kContended);
        runToCadence(kContended, {.recordOk = false});
        runOn(kContended, c_resetMisses * c_eventFailuresPerMiss - 2, {.recordOk = false});
        EXPECT_EQ(est.hint(), kContended);
        EXPECT_TRUE(launch({.recordOk = false}).finishChanged);
        EXPECT_EQ(est.hint(), 0u);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_RecordedProbeClearsTheFailureCount)
    {
        settleAt(kContended);
        for(int round = 0; round < 2; ++round)
        {
            SCOPED_TRACE(round);
            runToCadence(kContended, {.recordOk = false});
            runOn(kContended, c_eventFailuresPerMiss - 2, {.recordOk = false});
            EXPECT_TRUE(launch().plan.probe);
            EXPECT_EQ(est.missed(), 0u);
            markCus(epoch, kContended);
            completed = epoch;
        }
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_FullRingDefersTheDueProbe)
    {
        settleAt(kContended);
        runPending(c_probeRing * cfg.probeEvery - est.sinceCadence());
        ASSERT_EQ(est.outstanding(), c_probeRing);
        const uint32_t probesBefore = epoch;
        runPending((c_resetMisses + 1) * cfg.probeEvery);
        EXPECT_EQ(epoch, probesBefore);
        EXPECT_EQ(est.missed(), 0u);
        EXPECT_TRUE(est.due());
        EXPECT_EQ(est.hint(), kContended);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_FullRingDefersOverALaunchThatCannotProbe)
    {
        runPending(c_probeRing * cfg.probeEvery);
        ASSERT_EQ(est.outstanding(), c_probeRing);
        runPending((c_resetMisses + 1) * cfg.probeEvery, {.canProbe = false});
        EXPECT_EQ(est.missed(), 0u);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_DeferredProbeTakenOnceTheRingDrains)
    {
        runPending((c_probeRing + 1) * cfg.probeEvery);
        ASSERT_EQ(est.outstanding(), c_probeRing);
        ASSERT_TRUE(est.due());
        markCus(epoch, kContended);
        completed      = epoch;
        const Result r = launch();
        EXPECT_FALSE(r.plan.cadence);
        EXPECT_TRUE(r.plan.read);
        EXPECT_TRUE(r.plan.probe);
        EXPECT_EQ(est.outstanding(), 1u);
    }

    // A reading counts the CUs its probe marked, not older probes' marks.
    TEST_F(HostUnitAdaptiveSmCount, smoke_ReadingCountsOnlyItsProbesMarks)
    {
        runOn(kCus, cfg.probeEvery + 1);
        ASSERT_EQ(est.lastCount(), kCus);
        runToCadence(kContended);
        ASSERT_EQ(est.outstanding(), 1u);
        EXPECT_TRUE(launch().plan.read);
        EXPECT_EQ(est.lastCount(), kContended);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_FailedProbeIsDroppedWithoutAReading)
    {
        runPending(c_probeRing * cfg.probeEvery);
        ASSERT_EQ(est.outstanding(), c_probeRing);
        markCus(epoch, kContended);
        completed  = epoch;
        probeState = [](uint32_t) { return ProbeState::Failed; };
        EXPECT_FALSE(launch().plan.read);
        EXPECT_EQ(est.probesRead(), 0u);
        EXPECT_EQ(est.outstanding(), c_probeRing - 1);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_ForceProbeDiscardsReadings)
    {
        for(uint32_t every : {Config{}.probeEvery, 1u})
        {
            SCOPED_TRACE("probeEvery " + std::to_string(every));
            est              = {};
            cfg              = {};
            cfg.force        = kContended;
            cfg.forceProbe   = true;
            cfg.probeEvery   = every;
            const int rounds = 4;
            EXPECT_EQ(runOn(128, rounds * every + 1), 0);
            EXPECT_EQ(est.probesRead(), uint32_t(rounds));
            EXPECT_EQ(est.lastCount(), 128u);
            EXPECT_EQ(est.readingsTaken(), 0u);
            EXPECT_EQ(est.hint(), 0u);
        }
    }

    // A stream slot reused by another stream resets at the next cadence point.
    TEST_F(HostUnitAdaptiveSmCount, smoke_StreamChangeResetsTheEstimate)
    {
        settleAt(kContended);
        runOn(kContended, cfg.probeEvery - 1 - est.sinceCadence(), {.stream = 2});
        EXPECT_EQ(est.hint(), kContended);
        const Result r = launch({.stream = 2});
        EXPECT_TRUE(r.plan.cadence);
        EXPECT_EQ(est.hint(), 0u);
        EXPECT_EQ(est.readingCount(), 0u);
        EXPECT_EQ(est.probesRead(), 0u);
        // The new stream's first probe.
        EXPECT_TRUE(r.plan.probe);
        EXPECT_EQ(est.outstanding(), 1u);
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_CountFromAcrossEpochWrap)
    {
        std::vector<uint32_t> slots(c_probeSlots, 0);
        slots[0] = 0xfffffffeu;
        slots[1] = 0xffffffffu;
        slots[2] = 1;
        slots[3] = 2;
        EXPECT_EQ(countFrom(slots.data(), 0xffffffffu, kCus), 3u);
        EXPECT_EQ(countFrom(slots.data(), 1, kCus), 2u);
        EXPECT_EQ(countFrom(slots.data(), 1, 1), 1u);
        EXPECT_EQ(nextEpoch(0xffffffffu), 1u);
        EXPECT_EQ(nextEpoch(1), 2u);
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_EpochClearsAtTheInterval)
    {
        ProbeEpoch p;
        EXPECT_FALSE(p.advance(0));
        EXPECT_EQ(p.epoch, 1u);
        p.epoch = c_epochClearInterval - 1;
        EXPECT_FALSE(p.advance(0));
        EXPECT_EQ(p.epoch, c_epochClearInterval);
        EXPECT_TRUE(p.advance(0));
        EXPECT_EQ(p.cleared, c_epochClearInterval);
        EXPECT_EQ(p.epoch, c_epochClearInterval + 1);
        EXPECT_FALSE(p.advance(0));
    }

    // The guard matters only with more than one probe outstanding
    // (c_probeRing > 1); with one, a new probe implies none unread.
    TEST(HostUnitAdaptiveSmCountPure, smoke_EpochClearWaitsForUnreadProbes)
    {
        ProbeEpoch p;
        p.epoch = c_epochClearInterval;
        EXPECT_FALSE(p.advance(1));
        EXPECT_EQ(p.cleared, 0u);
        EXPECT_EQ(p.epoch, c_epochClearInterval + 1);
        EXPECT_TRUE(p.advance(0));
        EXPECT_EQ(p.cleared, c_epochClearInterval + 1);
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_EpochClearAcrossTheWrap)
    {
        ProbeEpoch p;
        p.cleared = 0u - c_epochClearInterval;
        p.epoch   = 0xffffffffu;
        EXPECT_FALSE(p.advance(0));
        EXPECT_EQ(p.epoch, 1u);
        // The skipped 0 counts toward the interval.
        EXPECT_TRUE(p.advance(0));
        EXPECT_EQ(p.cleared, 1u);
        EXPECT_EQ(p.epoch, 2u);
    }

    // A CU marked by probe 5 and held since then: once the epoch is 2^31 + 1
    // past it, the stale mark reads as newer. The clear before then drops it.
    TEST(HostUnitAdaptiveSmCountPure, smoke_ClearedSlotsAreNotCounted)
    {
        constexpr uint32_t    stale = 5, held = 10, marked = 20;
        constexpr uint32_t    readsStaleAsNewer = stale + (1u << 31) + 1;
        std::vector<uint32_t> slots(c_probeSlots, 0);
        std::fill_n(slots.begin(), held, stale);
        EXPECT_EQ(countFrom(slots.data(), readsStaleAsNewer, kCus), held);

        ProbeEpoch p;
        p.epoch = c_epochClearInterval;
        ASSERT_TRUE(p.advance(0));
        clearSlots(slots.data());
        EXPECT_EQ(std::count(slots.begin(), slots.end(), 0u), ptrdiff_t(c_probeSlots));
        std::fill_n(slots.begin() + held, marked, p.epoch);
        EXPECT_EQ(countFrom(slots.data(), p.epoch, kCus), marked);
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_TracksLaunch)
    {
        Config forced;
        forced.force = 192;
        Config forcedProbe;
        forcedProbe.force      = 192;
        forcedProbe.forceProbe = true;
        struct Case
        {
            const char* label;
            Config      cfg;
            bool        capturing;
            bool        expected;
        };
        const std::vector<Case> cases = {
            {"adaptive", Config{}, false, true},
            {"adaptive, capturing", Config{}, true, false},
            {"forced", forced, false, false},
            {"forced with probes", forcedProbe, false, true},
            {"forced with probes, capturing", forcedProbe, true, false},
        };
        for(const Case& c : cases)
        {
            SCOPED_TRACE(c.label);
            EXPECT_EQ(tracksLaunch(c.cfg, c.capturing), c.expected);
        }
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_LaunchHint)
    {
        Config forced;
        forced.force = 192;
        struct Case
        {
            const char* label;
            Config      cfg;
            bool        capturing;
            uint32_t    published;
            uint32_t    expected;
        };
        const std::vector<Case> cases = {
            {"published", Config{}, false, 96, 96},
            {"capturing: none", Config{}, true, 96, 0},
            {"forced", forced, false, 96, 192},
            {"forced, capturing", forced, true, 96, 192},
            {"no slot: none", Config{}, false, 0, 0},
            {"no slot, forced", forced, false, 0, 192},
        };
        for(const Case& c : cases)
        {
            SCOPED_TRACE(c.label);
            EXPECT_EQ(launchHint(c.cfg, c.capturing, c.published), c.expected);
        }
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_AppliesOnlyWithoutExplicitHint)
    {
        // The second argument is StreamK AUTO or ON.
        EXPECT_TRUE(applies(true, true, 0, 0));
        EXPECT_FALSE(applies(false, true, 0, 0)) << "mode off";
        EXPECT_FALSE(applies(true, false, 0, 0)) << "StreamK OFF";
        EXPECT_FALSE(applies(true, true, 128, 0)) << "desc hint";
        EXPECT_FALSE(applies(true, true, 0, 128)) << "handle hint";
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_LaunchProbed)
    {
        struct Case
        {
            const char* label;
            bool        hasAddr, kernelProbes, grouped;
            uint32_t    x, y, z;
            bool        expected;
        };
        const uint32_t          minGrid = kCus;
        const std::vector<Case> cases   = {
              {"no address", false, true, false, minGrid, 1, 1, false},
              {"kernel without probe", true, false, false, minGrid, 1, 1, false},
              {"grouped GEMM", true, true, true, minGrid, 1, 1, false},
              {"at minGrid", true, true, false, minGrid, 1, 1, true},
              {"above minGrid", true, true, false, 4 * minGrid, 1, 1, true},
              {"1-D below minGrid", true, true, false, minGrid - 1, 1, 1, false},
              {"2-D below minGrid", true, true, false, minGrid / 2 - 1, 2, 1, false},
              {"2-D at minGrid", true, true, false, minGrid / 2, 2, 1, true},
              {"3-D at minGrid", true, true, false, minGrid / 4, 2, 2, true},
        };
        for(const Case& c : cases)
        {
            SCOPED_TRACE(c.label);
            EXPECT_EQ(launchProbed(c.hasAddr, c.kernelProbes, c.grouped, c.x, c.y, c.z, minGrid),
                      c.expected);
        }
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_TinyProblem)
    {
        Config cfg;
        cfg.tinyMflops = 100;
        EXPECT_TRUE(tinyProblem(128, 128, 1024, 1, cfg)) << "33.6 MFLOP";
        EXPECT_FALSE(tinyProblem(256, 256, 1024, 1, cfg)) << "134 MFLOP";
        EXPECT_FALSE(tinyProblem(128, 128, 1024, 4, cfg)) << "batch counts";
        EXPECT_TRUE(tinyProblem(128, 128, 1024, 0, cfg)) << "batch 0 counts as 1";
        cfg.tinyMflops = 0;
        EXPECT_FALSE(tinyProblem(1, 1, 1, 1, cfg)) << "threshold 0";
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_AlgoTagRoundTrip)
    {
        const int32_t index   = 0x1234;
        uint8_t       data[8] = {};
        std::memcpy(data, &index, sizeof(index));
        uint32_t hint = 1;
        EXPECT_FALSE(algoTag(data, &hint));
        tagAlgo(data, 192);
        ASSERT_TRUE(algoTag(data, &hint));
        EXPECT_EQ(hint, 192u);
        int32_t kept = 0;
        std::memcpy(&kept, data, sizeof(kept));
        EXPECT_EQ(kept, index);
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_AlgoTagVersionMismatchIsUntagged)
    {
        uint8_t data[8] = {};
        tagAlgo(data, 192);
        data[5]       = c_algoTagVersion + 1;
        uint32_t hint = 0;
        EXPECT_FALSE(algoTag(data, &hint));
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_PickReselection)
    {
        {
            SCOPED_TRACE("first fitting candidate");
            const auto c = pickReselection(7, {3, 5, 9}, {false, true, true}, true);
            EXPECT_EQ(c.index, 5);
            EXPECT_FALSE(c.keepQueryHint);
        }
        {
            SCOPED_TRACE("none fits, original does");
            const auto c = pickReselection(7, {3, 5}, {false, false}, true);
            EXPECT_EQ(c.index, 7);
            EXPECT_FALSE(c.keepQueryHint);
        }
        {
            SCOPED_TRACE("nothing fits: original under the query hint");
            const auto c = pickReselection(7, {3, 5}, {false, false}, false);
            EXPECT_EQ(c.index, 7);
            EXPECT_TRUE(c.keepQueryHint);
        }
        {
            SCOPED_TRACE("no candidates");
            const auto c = pickReselection(7, {}, {}, false);
            EXPECT_EQ(c.index, 7);
            EXPECT_TRUE(c.keepQueryHint);
        }
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_SmallMemoEvictsRoundRobin)
    {
        SmallMemo<std::pair<int, int>, 2> memo;
        auto key = [](int k) { return [k](auto const& e) { return e.first == k; }; };
        EXPECT_EQ(memo.find(key(1)), nullptr);
        EXPECT_EQ(memo.insert({1, 10})->second, 10);
        memo.insert({2, 20});
        ASSERT_NE(memo.find(key(1)), nullptr);
        EXPECT_EQ(memo.find(key(2))->second, 20);
        memo.insert({3, 30});
        EXPECT_EQ(memo.find(key(1)), nullptr) << "oldest evicted first";
        EXPECT_EQ(memo.find(key(3))->second, 30);
        memo.insert({4, 40});
        EXPECT_EQ(memo.find(key(2)), nullptr);
        EXPECT_EQ(memo.size(), 2u);
        memo.clear();
        EXPECT_EQ(memo.size(), 0u);
        EXPECT_EQ(memo.find(key(4)), nullptr);
    }

    using Env = std::vector<std::pair<const char*, const char*>>;

    Config parseEnv(const Env& env, std::vector<std::string>* warnings)
    {
        return Config::fromEnv(
            [&](const char* name) -> const char* {
                for(auto const& [k, v] : env)
                    if(std::string(k) == name)
                        return v;
                return nullptr;
            },
            warnings);
    }

    TEST(HostUnitAdaptiveSmCountConfig, smoke_DefaultsWhenUnset)
    {
        std::vector<std::string> w;
        const Config             c = parseEnv({}, &w);
        EXPECT_EQ(c.describe(), Config{}.describe());
        EXPECT_FALSE(c.enabled);
        EXPECT_FALSE(c.hintVaries());
        EXPECT_TRUE(w.empty());
    }

    TEST(HostUnitAdaptiveSmCountConfig, smoke_ParsesValues)
    {
        std::vector<std::string> w;
        const Config             c = parseEnv(
            {{kEnvEnable, " On "}, {kEnvProbeEvery, "1"}, {kEnvProbeWindow, "4"}, {kEnvLog, ""}},
            &w);
        EXPECT_TRUE(c.enabled);
        EXPECT_TRUE(c.hintVaries());
        EXPECT_EQ(c.probeEvery, 1u);
        EXPECT_EQ(c.probeWindow, 4u);
        EXPECT_EQ(c.log, Config{}.log) << "empty keeps the default";
        EXPECT_TRUE(w.empty());
    }

    TEST(HostUnitAdaptiveSmCountConfig, smoke_ForceEnablesWithAFixedHint)
    {
        std::vector<std::string> w;
        const Config             c = parseEnv({{kEnvForce, "192"}, {kEnvForceProbe, "1"}}, &w);
        EXPECT_TRUE(c.enabled);
        EXPECT_FALSE(c.hintVaries());
        EXPECT_EQ(c.force, 192u);
        EXPECT_TRUE(c.forceProbe);
        EXPECT_TRUE(w.empty());
        EXPECT_NE(c.describe().find("force=192 force_probe=1"), std::string::npos);
    }

    // Each invalid value keeps every default and warns once, naming its variable.
    TEST(HostUnitAdaptiveSmCountConfig, smoke_InvalidValueWarnsAndKeepsTheDefault)
    {
        const Env cases = {
            {kEnvEnable, "yes"},
            {kEnvForce, "x"},
            {kEnvForce, "65536"},
            {kEnvProbeEvery, "0"},
            {kEnvProbeEvery, "-1"},
            {kEnvProbeEvery, "99999999999999999999"},
            {kEnvProbeWindow, "0"},
            {kEnvProbeWindow, "17"},
            {kEnvTinyMflops, "12abc"},
            {kEnvLog, "-1"},
            {kEnvForceProbe, "1"}, // without FORCE
        };
        for(auto const& [name, value] : cases)
        {
            SCOPED_TRACE(std::string(name) + "=" + value);
            std::vector<std::string> w;
            const Config             c = parseEnv({{name, value}}, &w);
            EXPECT_EQ(c.describe(), Config{}.describe());
            ASSERT_EQ(w.size(), 1u);
            EXPECT_EQ(w[0].rfind(std::string(name) + "=", 0), 0u) << w[0];
        }
    }

    // hipblaslt_timing_allowed_with_adaptive_sm_count reads the process
    // environment on each call; the variables are restored afterwards.
    class HostUnitAdaptiveSmCountTiming : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            for(const char* name : kVariables)
            {
                const char* value = std::getenv(name);
                m_saved.emplace_back(name,
                                     value ? std::optional<std::string>(value) : std::nullopt);
                set(name, nullptr);
            }
        }

        void TearDown() override
        {
            for(const auto& [name, value] : m_saved)
                set(name, value ? value->c_str() : nullptr);
        }

        static void set(const char* name, const char* value)
        {
#if defined(_WIN32)
            _putenv_s(name, value ? value : "");
#else
            if(value)
                setenv(name, value, 1);
            else
                unsetenv(name);
#endif
        }

        static constexpr const char* kVariables[] = {kEnvEnable, kEnvForce};

        std::vector<std::pair<const char*, std::optional<std::string>>> m_saved;
    };

    TEST_F(HostUnitAdaptiveSmCountTiming, smoke_RefusedOnlyWhenTheHintVaries)
    {
        struct Case
        {
            const char* label;
            const char* enable;
            const char* force;
            bool        allowed;
        };
        const std::vector<Case> cases = {
            {"off", nullptr, nullptr, true},
            {"adaptive", "1", nullptr, false},
            {"adaptive, forced", "1", "128", true},
        };
        for(const Case& c : cases)
        {
            SCOPED_TRACE(c.label);
            set(kEnvEnable, c.enable);
            set(kEnvForce, c.force);
            EXPECT_EQ(hipblaslt_timing_allowed_with_adaptive_sm_count(), c.allowed);
        }
    }
} // namespace
