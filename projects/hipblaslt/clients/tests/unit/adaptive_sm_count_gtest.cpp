// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests for the adaptive sm_count_target estimator and helpers in
// rocblaslt_adaptive_sm_count.hpp, reached by relative #include, and for the
// benchmark timing guard built on them. Launch scenarios drive
// Estimator::beforeLaunch/afterProbe over a hand-filled probe slot array.

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
        bool canProbe = true;
        bool recordOk = true; // the probe's event record succeeds
    };

    struct Result
    {
        bool probe   = false; // given an epoch
        bool tick    = false; // a tick
        bool read    = false; // a completed probe was read
        bool changed = false; // hint() changed
    };

    // One estimator, with a probe slot array the tests fill by hand.
    class HostUnitAdaptiveSmCount : public ::testing::Test
    {
    protected:
        Config                cfg;
        std::vector<uint32_t> slots     = std::vector<uint32_t>(c_probeSlots, 0);
        Estimator             est       = Estimator(cfg, slots.data(), kCus);
        uint32_t              epoch     = 0; // last probe epoch
        uint32_t              completed = 0; // probes up to this epoch are done
        uint32_t              recorded  = 0; // epoch of the recorded probe
        // State of the recorded probe.
        std::function<ProbeState()> probeState = [this] {
            return static_cast<int32_t>(completed - recorded) >= 0 ? ProbeState::Done
                                                                   : ProbeState::Pending;
        };

        // A fresh estimator under the current cfg, over cleared slots.
        void reset()
        {
            clearSlots(slots.data());
            est       = Estimator(cfg, slots.data(), kCus);
            epoch     = 0;
            completed = 0;
            recorded  = 0;
        }

        // Probe `e` marked the first `n` CUs.
        void markCus(uint32_t e, uint32_t n)
        {
            std::fill_n(slots.begin(), n, e);
        }

        // One tracked launch, called as the library does.
        Result launch(Options o = {})
        {
            const uint32_t   oldHint = est.hint();
            const uint32_t   read    = est.probesRead();
            const ProbeState state   = est.pending() ? probeState() : ProbeState::Pending;
            const uint32_t   e       = est.beforeLaunch(o.canProbe, state);
            Result           r;
            r.probe = e != 0;
            r.tick  = est.sinceTick() == 0;
            r.read  = est.probesRead() != read;
            if(r.probe)
            {
                epoch = e;
                if(o.recordOk)
                    recorded = e;
                est.afterProbe(o.recordOk);
            }
            r.changed = est.hint() != oldHint;
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
                if(r.probe && o.recordOk)
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

        // Launches until the next tick, that one included.
        int runToTick(uint32_t cus, Options o = {})
        {
            return runOn(cus, cfg.probeEvery - est.sinceTick(), o);
        }

        // Runs to the next tick, whose probe completes on `cus` CUs, and
        // reads it with one more launch.
        Result readAt(uint32_t cus)
        {
            runToTick(cus);
            return launch();
        }

        // Settles the hint at `cus` from a fresh estimator.
        void settleAt(uint32_t cus)
        {
            runOn(cus, cfg.probeWindow * cfg.probeEvery + 1);
            ASSERT_EQ(est.hint(), cus);
            ASSERT_EQ(est.missed(), 0u);
        }
    };

    TEST(HostUnitAdaptiveSmCountPure, smoke_Candidate)
    {
        struct Case
        {
            const char* label;
            uint32_t    lo, hi, hint, expected;
        };
        const std::vector<Case> cases = {
            {"full GPU, no hint", 256, 256, 0, 0},
            {"full GPU, from a hint", 256, 256, 128, 0},
            {"below, agreeing: exact", 186, 186, 0, 186},
            {"below, disagreeing: the smallest", 186, 200, 0, 186},
            {"below a hint: the smallest", 100, 120, 128, 100},
            {"above a hint: the smallest", 200, 230, 128, 200},
            {"above a hint, one at nCu: the smallest", 200, 256, 128, 200},
            {"both sides of a hint: holds", 100, 200, 128, 128},
            {"one at the hint: holds", 128, 200, 128, 128},
            {"full and below, no hint: holds", 128, 256, 0, 0},
        };
        for(const Case& c : cases)
        {
            SCOPED_TRACE(c.label);
            EXPECT_EQ(candidate(c.lo, c.hi, c.hint, kCus), c.expected);
        }
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_ReadingWindowKeepsTheLastN)
    {
        ReadingWindow w;
        EXPECT_FALSE(w.add(150, 3));
        EXPECT_FALSE(w.add(100, 3));
        EXPECT_TRUE(w.add(200, 3));
        EXPECT_EQ(w.lo(), 100u);
        EXPECT_EQ(w.hi(), 200u);
        EXPECT_TRUE(w.add(180, 3)); // over 150
        EXPECT_TRUE(w.add(190, 3)); // over 100
        EXPECT_EQ(w.size, 3u);
        EXPECT_EQ(w.lo(), 180u);
        EXPECT_EQ(w.hi(), 200u);
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
            {"window 3: one full reading clears a hint", 3, {192, 192, 192, 256}, 0},
            {"window 3: then a hint needs a full window", 3, {192, 192, 192, 256, 192, 192}, 0},
            {"window 3: a full window again", 3, {192, 192, 192, 256, 192, 192, 192}, 192},
            {"window 3: up below nCu needs the window", 3, {192, 192, 192, 230, 230}, 192},
            {"window 3: up below nCu, a full window", 3, {192, 192, 192, 230, 230, 230}, 230},
        };
        for(const Case& c : cases)
        {
            SCOPED_TRACE(c.label);
            cfg             = {};
            cfg.probeWindow = c.window;
            cfg.probeEvery  = 1;
            reset();
            for(uint32_t count : c.readings)
                runOn(count, 1);
            launch();
            EXPECT_EQ(est.hint(), c.expected);
            EXPECT_EQ(est.probesRead(), c.readings.size());
        }
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_ProbeFallsDueAtOnceThenEveryProbeEveryLaunches)
    {
        const uint32_t        rounds = 3;
        std::vector<uint32_t> probes, expected;
        for(uint32_t l = 0; l < rounds * cfg.probeEvery; ++l)
        {
            if(launch().probe)
            {
                probes.push_back(l);
                completed = epoch;
            }
            if(l == 0 || (l + 1) % cfg.probeEvery == 0)
                expected.push_back(l);
        }
        EXPECT_EQ(probes, expected);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_DueProbeGoesToTheNextLaunchThatCanProbe)
    {
        runOn(kCus, cfg.probeEvery - 1);
        const Result r = launch({.canProbe = false});
        EXPECT_TRUE(r.tick);
        EXPECT_FALSE(r.probe);
        EXPECT_TRUE(est.due());
        EXPECT_EQ(est.missed(), 0u);
        EXPECT_TRUE(launch().probe);
    }

    // The window fills with the reading of the probe at its last tick.
    TEST_F(HostUnitAdaptiveSmCount, smoke_ContentionSetsTheHintAfterAFullWindow)
    {
        EXPECT_EQ(runOn(kContended, (cfg.probeWindow - 1) * cfg.probeEvery), 0);
        EXPECT_EQ(est.hint(), 0u);
        EXPECT_TRUE(launch().changed);
        EXPECT_EQ(est.hint(), kContended);
    }

    // One reading of nCu drops a hint at once; a new hint then needs a full
    // window.
    TEST_F(HostUnitAdaptiveSmCount, smoke_FullReadingClearsTheHintAtOnce)
    {
        settleAt(kContended);
        EXPECT_TRUE(readAt(kCus).changed);
        EXPECT_EQ(est.hint(), 0u);
        EXPECT_EQ(est.window().size, 1u);
        for(uint32_t i = 1; i < cfg.probeWindow; ++i)
            EXPECT_FALSE(readAt(kContended).changed);
        EXPECT_TRUE(readAt(kContended).changed);
        EXPECT_EQ(est.hint(), kContended);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_HigherReadingBelowNCuNeedsTheWindow)
    {
        settleAt(kContended);
        const uint32_t higher = kContended + 32;
        for(uint32_t i = 1; i < cfg.probeWindow; ++i)
            EXPECT_FALSE(readAt(higher).changed);
        EXPECT_TRUE(readAt(higher).changed);
        EXPECT_EQ(est.hint(), higher);
    }

    // Only launches that cannot probe: the probe falls due at a tick, and
    // each later tick that finds it still due misses.
    TEST_F(HostUnitAdaptiveSmCount, smoke_MissedProbesResetTheHint)
    {
        settleAt(kContended);
        runToTick(kContended, {.canProbe = false});
        ASSERT_TRUE(est.due());
        EXPECT_EQ(est.missed(), 0u);
        for(uint32_t m = 1; m < c_resetMisses; ++m)
        {
            runToTick(kContended, {.canProbe = false});
            EXPECT_EQ(est.missed(), m);
        }
        runOn(kContended, cfg.probeEvery - 1, {.canProbe = false});
        EXPECT_EQ(est.hint(), kContended);
        const Result last = launch({.canProbe = false});
        EXPECT_TRUE(last.tick);
        EXPECT_TRUE(last.changed);
        EXPECT_EQ(est.hint(), 0u);
        EXPECT_EQ(est.window().size, 0u);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_ProbeBetweenMissesRestartsTheCount)
    {
        settleAt(kContended);
        runToTick(kContended, {.canProbe = false});
        runToTick(kContended, {.canProbe = false});
        ASSERT_EQ(est.missed(), 1u);
        runOn(kContended, 1);
        EXPECT_EQ(est.missed(), 0u);
        runToTick(kContended, {.canProbe = false});
        runToTick(kContended, {.canProbe = false});
        EXPECT_EQ(est.missed(), 1u);
        EXPECT_EQ(est.hint(), kContended);
    }

    // probeEvery - 1 launches that cannot probe between ones that can, at
    // any phase: each due probe is taken before the next tick.
    TEST_F(HostUnitAdaptiveSmCount, smoke_TinyLaunchesBetweenProbesDoNotMiss)
    {
        for(uint32_t phase = 0; phase < cfg.probeEvery; ++phase)
        {
            SCOPED_TRACE("phase " + std::to_string(phase));
            reset();
            settleAt(kContended);
            runOn(kContended, phase);
            for(uint32_t i = 0; i < 4 * cfg.probeEvery; ++i)
            {
                runOn(kContended, cfg.probeEvery - 1, {.canProbe = false});
                ASSERT_EQ(est.missed(), 0u);
                runOn(kContended, 1);
            }
            EXPECT_EQ(est.hint(), kContended);
        }
    }

    // 2 * probeEvery - 1 launches that cannot probe between ones that can, at
    // any phase: at most one miss before a probe is taken.
    TEST_F(HostUnitAdaptiveSmCount, smoke_LongerTinyRunsBetweenProbesKeepTheHint)
    {
        for(uint32_t phase = 0; phase < cfg.probeEvery; ++phase)
        {
            SCOPED_TRACE("phase " + std::to_string(phase));
            reset();
            settleAt(kContended);
            runOn(kContended, phase);
            for(uint32_t i = 0; i < 4 * cfg.probeEvery; ++i)
            {
                runOn(kContended, 2 * cfg.probeEvery - 1, {.canProbe = false});
                ASSERT_LT(est.missed(), c_resetMisses);
                runOn(kContended, 1);
            }
            EXPECT_EQ(est.hint(), kContended);
        }
    }

    // A probe whose event record fails stays due, so the next tick misses.
    TEST_F(HostUnitAdaptiveSmCount, smoke_FailedRecordsAreMisses)
    {
        settleAt(kContended);
        runToTick(kContended, {.recordOk = false});
        EXPECT_EQ(est.missed(), 0u);
        EXPECT_FALSE(est.pending());
        EXPECT_TRUE(est.due());
        for(uint32_t m = 1; m < c_resetMisses; ++m)
        {
            runToTick(kContended, {.recordOk = false});
            EXPECT_EQ(est.missed(), m);
            EXPECT_EQ(est.hint(), kContended);
        }
        EXPECT_EQ(runToTick(kContended, {.recordOk = false}), 1);
        EXPECT_EQ(est.hint(), 0u);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_PendingProbeDefersTheDueProbe)
    {
        settleAt(kContended);
        runPending(cfg.probeEvery - est.sinceTick());
        ASSERT_TRUE(est.pending());
        const uint32_t probesBefore = epoch;
        runPending((c_resetMisses + 1) * cfg.probeEvery);
        EXPECT_EQ(epoch, probesBefore);
        EXPECT_EQ(est.missed(), 0u);
        EXPECT_TRUE(est.due());
        EXPECT_EQ(est.hint(), kContended);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_PendingProbeDefersOverALaunchThatCannotProbe)
    {
        runPending(cfg.probeEvery);
        ASSERT_TRUE(est.pending());
        runPending((c_resetMisses + 1) * cfg.probeEvery, {.canProbe = false});
        EXPECT_EQ(est.missed(), 0u);
    }

    TEST_F(HostUnitAdaptiveSmCount, smoke_DeferredProbeTakenOnceThePendingOneIsRead)
    {
        runPending(2 * cfg.probeEvery);
        ASSERT_TRUE(est.pending());
        ASSERT_TRUE(est.due());
        markCus(epoch, kContended);
        completed      = epoch;
        const Result r = launch();
        EXPECT_FALSE(r.tick);
        EXPECT_TRUE(r.read);
        EXPECT_TRUE(r.probe);
        EXPECT_TRUE(est.pending());
    }

    // A reading counts the CUs its probe marked, not older probes' marks.
    TEST_F(HostUnitAdaptiveSmCount, smoke_ReadingCountsOnlyItsProbesMarks)
    {
        runOn(kCus, cfg.probeEvery + 1);
        ASSERT_EQ(est.lastCount(), kCus);
        runToTick(kContended);
        ASSERT_TRUE(est.pending());
        EXPECT_TRUE(launch().read);
        EXPECT_EQ(est.lastCount(), kContended);
    }

    // A failed probe gives no reading and falls due again: the launch that
    // finds it failed takes a new one.
    TEST_F(HostUnitAdaptiveSmCount, smoke_FailedProbeIsDroppedWithoutAReading)
    {
        runPending(1);
        ASSERT_TRUE(est.pending());
        const uint32_t failed = epoch;
        markCus(epoch, kContended);
        completed      = epoch;
        probeState     = [] { return ProbeState::Failed; };
        const Result r = launch();
        EXPECT_FALSE(r.read);
        EXPECT_EQ(est.probesRead(), 0u);
        EXPECT_TRUE(r.probe);
        EXPECT_NE(epoch, failed);
        EXPECT_TRUE(est.pending());
    }

    // A failed probe that no launch takes again by the next tick is a miss.
    TEST_F(HostUnitAdaptiveSmCount, smoke_FailedProbeNotRetakenIsAMiss)
    {
        runPending(1);
        ASSERT_TRUE(est.pending());
        probeState = [] { return ProbeState::Failed; };
        launch({.canProbe = false});
        EXPECT_FALSE(est.pending());
        EXPECT_TRUE(est.due());
        EXPECT_EQ(est.missed(), 0u);
        runPending(cfg.probeEvery - est.sinceTick(), {.canProbe = false});
        EXPECT_EQ(est.missed(), 1u);
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
    }

    bool allZero(const std::vector<uint32_t>& slots)
    {
        return std::count(slots.begin(), slots.end(), 0u) == ptrdiff_t(slots.size());
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_EpochClearsAtTheInterval)
    {
        std::vector<uint32_t> slots(c_probeSlots, 7);
        ProbeEpoch            p;
        EXPECT_EQ(p.advance(slots.data()), 1u);
        p.epoch = c_epochClearInterval - 1;
        EXPECT_EQ(p.advance(slots.data()), c_epochClearInterval);
        EXPECT_FALSE(allZero(slots));
        EXPECT_EQ(p.advance(slots.data()), c_epochClearInterval + 1);
        EXPECT_TRUE(allZero(slots));
        EXPECT_EQ(p.cleared, c_epochClearInterval);
        slots.assign(c_probeSlots, 7);
        p.advance(slots.data());
        EXPECT_FALSE(allZero(slots));
    }

    TEST(HostUnitAdaptiveSmCountPure, smoke_EpochClearAcrossTheWrap)
    {
        std::vector<uint32_t> slots(c_probeSlots, 7);
        ProbeEpoch            p;
        p.cleared = 0u - c_epochClearInterval;
        p.epoch   = 0xffffffffu;
        EXPECT_EQ(p.advance(slots.data()), 1u) << "skips 0";
        EXPECT_FALSE(allZero(slots));
        // The skipped 0 counts toward the interval.
        EXPECT_EQ(p.advance(slots.data()), 2u);
        EXPECT_TRUE(allZero(slots));
        EXPECT_EQ(p.cleared, 1u);
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
        p.epoch          = c_epochClearInterval;
        const uint32_t e = p.advance(slots.data());
        EXPECT_TRUE(allZero(slots));
        std::fill_n(slots.begin() + held, marked, e);
        EXPECT_EQ(countFrom(slots.data(), e, kCus), marked);
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

    using Env = std::vector<std::pair<const char*, const char*>>;

    // The environment parseEnv() reads from.
    const Env* g_env = nullptr;

    const char* fakeGetenv(const char* name)
    {
        for(auto const& [k, v] : *g_env)
            if(std::string(k) == name)
                return v;
        return nullptr;
    }

    Config parseEnv(const Env& env, std::vector<std::string>* warnings)
    {
        g_env = &env;
        return Config::fromEnv(fakeGetenv, warnings);
    }

    TEST(HostUnitAdaptiveSmCountConfig, smoke_DefaultsWhenUnset)
    {
        std::vector<std::string> w;
        const Config             c = parseEnv({}, &w);
        EXPECT_EQ(c.describe(), Config{}.describe());
        EXPECT_FALSE(c.enabled);
        EXPECT_FALSE(c.hintVaries());
        EXPECT_TRUE(c.tracksLaunches());
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
        EXPECT_TRUE(c.tracksLaunches());
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
        EXPECT_TRUE(c.tracksLaunches());
        EXPECT_TRUE(w.empty());
        EXPECT_NE(c.describe().find("force=192 force_probe=1"), std::string::npos);
    }

    TEST(HostUnitAdaptiveSmCountConfig, smoke_ForceWithoutProbeTracksNoLaunches)
    {
        std::vector<std::string> w;
        const Config             c = parseEnv({{kEnvForce, "192"}}, &w);
        EXPECT_TRUE(c.enabled);
        EXPECT_FALSE(c.forceProbe);
        EXPECT_FALSE(c.tracksLaunches());
        EXPECT_TRUE(w.empty());
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
            {kEnvProbeEvery, "4294967296"},
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
