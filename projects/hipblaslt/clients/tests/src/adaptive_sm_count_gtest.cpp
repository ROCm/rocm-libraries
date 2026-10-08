// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the adaptive sm_count_target estimator over a simulated probe slot
// array (host-only), plus heuristic-query and launch checks on the GPU. The
// estimator header is included by relative path (see arch_revision_gtest.cpp).

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include "../../../library/src/amd_detail/rocblaslt/src/include/rocblaslt_adaptive_sm_count.hpp"

#include <cstdlib>
#include <deque>
#include <utility>
#include <vector>

namespace
{
    using namespace rocblaslt::adaptive_sm;

    constexpr uint32_t c_nCu = 256;

    // gfx950-like slot index: 8 XCCs x 32 CUs.
    uint32_t cuSlot(uint32_t cu)
    {
        const uint32_t xcc = cu / 32, local = cu % 32;
        const uint32_t se = local / 8, sh = (local / 4) % 2, c = local % 4;
        return xcc << 8 | se << 5 | sh << 4 | c;
    }

    // Mirrors _rocblaslt_handle::adaptiveSmCountForLaunch: a read every
    // cfg.period launches, which may schedule a later launch as a probe launch.
    // The GPU runs `lag` launches behind the host. A regular launch probes
    // when it runs at hint 0 with a grid covering N_CU (`canProbe`, and not
    // `blind`), or when its re-selected grid still does (`probesUnderHint`).
    // Scheduled probe launches and every cfg.probeStride-th launch probe when
    // their kernel can (`canProbe` and `probeLaunchesWrite`): padded with
    // probe-only workgroups, or at hint 0. Tiny launches (`tiny`, by launch
    // index) take no epoch; a long run of them drops the hint.
    struct Sim
    {
        std::vector<uint32_t> slots                   = std::vector<uint32_t>(c_probeSlots, 0);
        uint32_t              hostEpoch               = 0;
        uint32_t              lag                     = 0;
        uint32_t              batch                   = 1; // GPU completes launches in groups
        bool                  probesUnderHint         = false;
        bool                  probeLaunchesWrite      = true;
        bool                  blind                    = false; // grid below N_CU even at hint 0
        bool (*canProbe)(uint32_t epoch)              = nullptr;
        bool (*tiny)(uint32_t launch)                  = nullptr;
        uint32_t                              launches = 0;
        uint32_t                              tinyRun  = 0;
        uint32_t                              hint     = 0; // published
        uint32_t                              probeAt = 0;
        std::vector<uint32_t>                 probeEpochs;
        Config                                cfg;
        Estimator                             est;
        int                                   changes = 0;
        int                                   probes  = 0;
        std::deque<std::pair<uint32_t, bool>> queue;

        // GPU completes launch `e` on CUs [0, cus).
        void gpuRun(uint32_t e, uint32_t cus)
        {
            for(uint32_t cu = 0; cu < cus; ++cu)
                slots[cuSlot(cu)] = e;
        }

        // Host issues `n` launches; the GPU executes on `cus` CUs (0: stalled).
        void run(uint32_t n, uint32_t cus)
        {
            for(uint32_t i = 0; i < n; ++i)
            {
                if(tiny && tiny(launches++))
                {
                    if(tinyRunDecays(++tinyRun, hint, cfg))
                    {
                        est.reset();
                        probeAt = 0;
                        hint    = 0;
                        ++changes;
                    }
                    continue;
                }
                tinyRun = 0;
                if(++hostEpoch == 0)
                    ++hostEpoch;
                const bool claimed = probeDueAt(hostEpoch, probeAt);
                if(claimed)
                {
                    probeAt = 0;
                    ++probes;
                    probeEpochs.push_back(hostEpoch);
                }
                const LaunchPlan plan = planLaunch(hostEpoch, claimed, cfg);
                if(plan.read)
                {
                    changes += est.read(slots.data(), hostEpoch, c_nCu, cfg);
                    if(const uint32_t next = est.probeDue(hostEpoch, cfg))
                        probeAt = next;
                    hint = est.hint();
                }
                const bool fits   = !canProbe || canProbe(hostEpoch);
                const bool writes = (plan.pad && probeLaunchesWrite && fits)
                                    || (est.hint() == 0 && fits && !blind);
                queue.emplace_back(hostEpoch, writes || probesUnderHint);
                if(queue.size() < lag + batch)
                    continue;
                while(cus && queue.size() > lag)
                {
                    if(queue.front().second)
                        gpuRun(queue.front().first, cus);
                    queue.pop_front();
                }
            }
        }

        // Launches until the hint equals `hint`; returns how many it took.
        uint32_t until(uint32_t hint, uint32_t cus, uint32_t limit)
        {
            uint32_t n = 0;
            while(est.hint() != hint && n < limit)
            {
                run(1, cus);
                ++n;
            }
            return n;
        }
    };

    TEST(adaptive_sm_count, quantise)
    {
        EXPECT_EQ(quantise(256, 256, 8), 0u);
        EXPECT_EQ(quantise(248, 256, 8), 0u);
        EXPECT_EQ(quantise(247, 256, 8), 0u); // rounds to 256
        EXPECT_EQ(quantise(239, 256, 8), 224u);
        EXPECT_EQ(quantise(192, 256, 8), 192u);
        EXPECT_EQ(quantise(111, 256, 8), 96u);
        EXPECT_EQ(quantise(112, 256, 8), 128u);
        EXPECT_EQ(quantise(47, 256, 8), 32u);
        EXPECT_EQ(quantise(48, 256, 8), 64u);
        EXPECT_EQ(quantise(5, 256, 8), 32u);
        EXPECT_EQ(quantise(0, 256, 8), 32u);
    }

    TEST(adaptive_sm_count, confirmations)
    {
        const Config c;
        EXPECT_EQ(confirmations(0, 192, c_nCu, c), c.confirm);
        EXPECT_EQ(confirmations(0, 224, c_nCu, c), c.confirm);
        EXPECT_EQ(confirmations(0, 128, c_nCu, c), c.confirmDown);
        EXPECT_EQ(confirmations(192, 128, c_nCu, c), c.confirm);
        EXPECT_EQ(confirmations(192, 64, c_nCu, c), c.confirmDown);
        EXPECT_EQ(confirmations(96, 64, c_nCu, c), c.confirmDown);
        EXPECT_EQ(confirmations(64, 0, c_nCu, c), c.confirmUp);
        EXPECT_EQ(confirmations(32, 192, c_nCu, c), c.confirm);
        EXPECT_EQ(confirmations(192, 0, c_nCu, c), c.confirmUp);
        EXPECT_EQ(confirmations(0, 0, c_nCu, c), c.confirm);
    }

    TEST(adaptive_sm_count, applies_only_without_explicit_hint)
    {
        EXPECT_TRUE(applies(true, true, 0, 0));
        EXPECT_FALSE(applies(false, true, 0, 0));
        EXPECT_FALSE(applies(true, false, 0, 0));
        EXPECT_FALSE(applies(true, true, 128, 0));
        EXPECT_FALSE(applies(true, true, 0, 128));
        EXPECT_FALSE(applies(true, true, 64, 128));
    }

    TEST(adaptive_sm_count, algo_tag_round_trip)
    {
        uint8_t  data[8] = {0x34, 0x12, 0, 0, 0, 0, 0, 0};
        uint32_t hint    = 1;
        EXPECT_FALSE(algoTag(data, &hint));
        tagAlgo(data, 192);
        ASSERT_TRUE(algoTag(data, &hint));
        EXPECT_EQ(hint, 192u);
        EXPECT_EQ(*(int*)data, 0x1234);
    }

    TEST(adaptive_sm_count, pick_reselection)
    {
        // First fitting candidate wins.
        auto c = pickReselection(7, {3, 5, 9}, {false, true, true}, true);
        EXPECT_EQ(c.index, 5);
        EXPECT_FALSE(c.keepQueryHint);
        // Nothing fits: the original, under the launch hint if it fits that.
        c = pickReselection(7, {3, 5}, {false, false}, true);
        EXPECT_EQ(c.index, 7);
        EXPECT_FALSE(c.keepQueryHint);
        // Nor does the original: it runs under the query hint.
        c = pickReselection(7, {3, 5}, {false, false}, false);
        EXPECT_EQ(c.index, 7);
        EXPECT_TRUE(c.keepQueryHint);
        c = pickReselection(7, {}, {}, false);
        EXPECT_EQ(c.index, 7);
        EXPECT_TRUE(c.keepQueryHint);
    }

    TEST(adaptive_sm_count, probe_schedule)
    {
        Config         cfg;
        ProbeSchedule  ps;
        const uint32_t P = cfg.probePeriod;
        // A recent sample: no probe.
        EXPECT_EQ(ps.due(1000, 1000 - P + 1, true, 900, cfg), 0u);
        // P launches without a sample, with or without a hint: the next
        // launch probes.
        EXPECT_EQ(ps.due(1000, 1000 - P, true, 900, cfg), 1001u);
        EXPECT_TRUE(ps.pending());
        // Outstanding until the anchor reaches its epoch...
        EXPECT_EQ(ps.due(1500, 0, true, 1000, cfg), 0u);
        EXPECT_EQ(ps.due(1500, 1300, true, 1001, cfg), 1501u);
        EXPECT_EQ(ps.lost(), 0u);
        // ...or the timeout passes, stretched by twice the host's lead at the
        // last sample (1300 - 1001). A lost probe moves the next one a
        // launch later.
        const uint32_t t = 1501 + cfg.probeTimeout() + 2 * 299;
        EXPECT_EQ(ps.due(t - 1, 0, true, 1200, cfg), 0u);
        EXPECT_EQ(ps.due(t, 0, true, 1200, cfg), t + 2);
        EXPECT_EQ(ps.lost(), 1u);
        // Wrap-safe.
        ps.reset();
        EXPECT_EQ(ps.due(0xffffffffu, 0xffffffffu - P, false, 0, cfg), 1u);
        EXPECT_EQ(ps.due(20, 0, true, 0xfffffff0u, cfg), 0u);
        EXPECT_EQ(ps.due(30, 30 - P, true, 12, cfg), 31u);
    }

    TEST(adaptive_sm_count, small_memo)
    {
        SmallMemo<std::pair<int, int>, 2> memo;
        auto key = [](int k) { return [k](auto const& e) { return e.first == k; }; };
        EXPECT_EQ(memo.find(key(1)), nullptr);
        EXPECT_EQ(memo.insert({1, 10})->second, 10);
        memo.insert({2, 20});
        ASSERT_NE(memo.find(key(1)), nullptr);
        EXPECT_EQ(memo.find(key(2))->second, 20);
        // Full: evicts round-robin, oldest slot first.
        memo.insert({3, 30});
        EXPECT_EQ(memo.find(key(1)), nullptr);
        EXPECT_EQ(memo.find(key(3))->second, 30);
        memo.insert({4, 40});
        EXPECT_EQ(memo.find(key(2)), nullptr);
        EXPECT_EQ(memo.size(), 2u);
    }

    TEST(adaptive_sm_count, full_gpu_publishes_nothing)
    {
        Sim s;
        s.lag = 3;
        s.run(200, c_nCu);
        EXPECT_EQ(s.est.hint(), 0u);
        EXPECT_EQ(s.changes, 0);
        EXPECT_EQ(s.probes, 0);
        EXPECT_EQ(s.est.knownSlots(), c_nCu);
    }

    TEST(adaptive_sm_count, contention_is_detected_and_confirmed)
    {
        Sim s;
        s.run(64, c_nCu); // learn every CU slot
        s.run(8, 192);
        EXPECT_EQ(s.est.hint(), 0u); // one sample is not enough
        s.run(8, 192);
        EXPECT_EQ(s.est.hint(), 192u);
        EXPECT_EQ(s.est.lastCount(), 192u);
    }

    TEST(adaptive_sm_count, large_downward_move_needs_more_samples)
    {
        Sim s;
        s.run(64, c_nCu);
        s.run(8 * (s.cfg.confirmDown - 1), 128);
        EXPECT_EQ(s.est.hint(), 0u);
        s.run(8, 128);
        EXPECT_EQ(s.est.hint(), 128u);
    }

    TEST(adaptive_sm_count, transient_dip_is_not_published)
    {
        Sim s;
        s.run(64, c_nCu);
        s.run(8 * (s.cfg.confirmDown - 1), 40);
        s.run(64, c_nCu);
        EXPECT_EQ(s.est.hint(), 0u);
        EXPECT_EQ(s.changes, 0);
    }

    TEST(adaptive_sm_count, batched_completion_onset)
    {
        // Anchors advance 20 at a time, more than the read period: at hint 0
        // every launch still probes, so the window stays `window` launches.
        Sim s;
        s.batch = 20;
        s.lag   = 20;
        s.run(400, c_nCu);
        const uint32_t n = s.until(128, 128, 1000);
        EXPECT_LE(n, (s.cfg.confirmDown + 2) * s.batch);
        EXPECT_EQ(s.est.lastCount(), 128u);
    }

    TEST(adaptive_sm_count, host_lag_does_not_read_as_contention)
    {
        // Host 600 launches ahead: the window anchors on E_max, not hostEpoch.
        Sim s;
        s.lag = 600;
        s.run(2000, c_nCu);
        EXPECT_EQ(s.est.hint(), 0u);
        s.run(2000, 128);
        EXPECT_EQ(s.est.hint(), 128u);
    }

    TEST(adaptive_sm_count, probe_launches_keep_the_hint)
    {
        // Under the hint the re-selected grid is below N_CU and never probes;
        // periodic probe launches keep the hint confirmed.
        Sim s;
        s.cfg.probeStride = 0; // scheduled probe launches only
        s.lag = 600;
        s.run(1000, c_nCu);
        ASSERT_LT(s.until(192, 192, 2000), 2000u);
        const int changes = s.changes;
        s.run(20000, 192);
        EXPECT_EQ(s.est.hint(), 192u);
        EXPECT_EQ(s.changes, changes);
        EXPECT_EQ(s.est.lastCount(), 192u);
        // About one probe launch per lag + probePeriod launches.
        EXPECT_GT(s.probes, 20000 / int(2 * (s.lag + s.cfg.probePeriod)));
        EXPECT_LT(s.probes, 20000 / int(s.lag + s.cfg.probePeriod) + 2);
    }

    TEST(adaptive_sm_count, probe_launch_recovers_when_contention_ends)
    {
        Sim s;
        s.cfg.probeStride = 0; // scheduled probe launches only
        s.lag = 600;
        s.run(1000, c_nCu);
        ASSERT_LT(s.until(128, 128, 3000), 3000u);
        s.run(3000, 128);
        ASSERT_EQ(s.est.hint(), 128u);
        // confirmUp probe launches, each lag + period + read apart.
        const uint32_t n = s.until(0, c_nCu, 10000);
        EXPECT_LE(n, (s.cfg.confirmUp + 1) * (s.lag + s.cfg.probePeriod + s.cfg.period));
        EXPECT_EQ(s.est.hint(), 0u);
    }

    TEST(adaptive_sm_count, deeper_contention_under_a_hint_is_seen)
    {
        // Slots the cotenant took keep their last probe epoch; the window of
        // distinct probe epochs drops them after `window` probe launches.
        Sim s;
        s.cfg.probeStride = 0; // scheduled probe launches only
        s.lag = 100;
        s.run(500, c_nCu);
        ASSERT_LT(s.until(192, 192, 2000), 2000u);
        s.run(1000, 192);
        const uint32_t limit
            = (s.cfg.window + s.cfg.confirm + 2) * (s.lag + s.cfg.probePeriod + s.cfg.period);
        EXPECT_LT(s.until(128, 128, limit), limit);
        EXPECT_EQ(s.est.lastCount(), 128u);
    }

    TEST(adaptive_sm_count, stalled_gpu_holds_hint_with_one_probe_per_timeout)
    {
        Sim s;
        s.cfg.probeStride = 0; // scheduled probe launches only
        s.run(64, c_nCu);
        s.run(16, 192);
        ASSERT_EQ(s.est.hint(), 192u);
        const int probes = s.probes;
        // GPU stalls: nothing advances. The hint holds; the first probe launch
        // stays outstanding until its timeout.
        s.run(s.cfg.probePeriod + s.cfg.period, 0);
        ASSERT_EQ(s.probes, probes + 1);
        uint32_t n = 0;
        while(s.probes == probes + 1 && n < 10000)
        {
            s.run(1, 0);
            ++n;
        }
        EXPECT_GE(n, s.cfg.probeTimeout() - s.cfg.period);
        EXPECT_LT(n, 10000u);
        EXPECT_EQ(s.est.hint(), 192u);
    }

    TEST(adaptive_sm_count, lost_probes_fall_back_to_hint_zero)
    {
        // Probe launches never show up as the anchor: after probeLosses of
        // them, timeouts not stretching, the hint drops to 0 and the
        // full-grid launches re-measure.
        Sim s;
        s.cfg.probeStride = 0; // scheduled probe launches only
        s.lag = 100;
        s.run(500, c_nCu);
        ASSERT_LT(s.until(192, 192, 2000), 2000u);
        s.probeLaunchesWrite = false;
        s.probeEpochs.clear();
        const uint32_t gap = s.cfg.probeTimeout() + 2 * (s.lag + 2 * s.cfg.period) + s.cfg.period;
        const uint32_t n   = s.until(0, 192, 20000);
        EXPECT_LE(n, (s.cfg.probeLosses + 1) * gap);
        EXPECT_EQ(s.probeEpochs.size(), s.cfg.probeLosses);
        for(size_t i = 1; i < s.probeEpochs.size(); ++i)
            EXPECT_LE(s.probeEpochs[i] - s.probeEpochs[i - 1], gap);
        // Contention is still there: the hint comes back.
        EXPECT_LT(s.until(192, 192, 2000), 2000u);
    }

    TEST(adaptive_sm_count, periodic_workload_probes_on_another_gemm)
    {
        // Four GEMMs per iteration; the one on epochs divisible by 8 (every
        // read) cannot probe at hint 0. Probes must land elsewhere and see the
        // CUs come back.
        Sim s;
        s.cfg.probeStride = 0; // scheduled probe launches only
        s.lag      = 40;
        s.canProbe = [](uint32_t e) { return e % 8 != 0; };
        s.run(400, c_nCu);
        ASSERT_LT(s.until(128, 128, 2000), 2000u);
        s.run(3000, 128);
        ASSERT_EQ(s.est.hint(), 128u);
        const uint32_t n = s.until(0, c_nCu, 10000);
        EXPECT_LE(n, 3 * (s.lag + s.cfg.probePeriod + 2 * s.cfg.period));
        EXPECT_EQ(s.est.hint(), 0u);
    }

    TEST(adaptive_sm_count, blind_shape_probes_with_padding)
    {
        // The grid is below N_CU even at hint 0: only padded launches write,
        // every probeStride launches, so samples keep the GPU's pace however
        // far the host runs ahead.
        Sim s;
        s.lag   = 600;
        s.blind = true;
        s.run(5000, c_nCu);
        EXPECT_EQ(s.est.hint(), 0u);
        EXPECT_EQ(s.changes, 0);
        EXPECT_EQ(s.est.lastCount(), c_nCu);
        const uint32_t step = s.cfg.probeStride;
        const uint32_t onset
            = s.lag + (s.cfg.window + s.cfg.confirmDown + 1) * step + 2 * s.cfg.period;
        EXPECT_LT(s.until(128, 128, onset), onset);
        EXPECT_EQ(s.est.lastCount(), 128u);
        const uint32_t recovery = s.lag + (s.cfg.confirmUp + 2) * step + 2 * s.cfg.period;
        EXPECT_LT(s.until(0, c_nCu, recovery), recovery);
    }

    TEST(adaptive_sm_count, blind_shape_scheduled_probes)
    {
        // Without padded strides only scheduled probe launches write, about
        // one per lag + probePeriod launches, at hint 0 too.
        Sim s;
        s.cfg.probeStride = 0;
        s.lag             = 100;
        s.blind           = true;
        s.run(5000, c_nCu);
        EXPECT_EQ(s.est.hint(), 0u);
        EXPECT_EQ(s.changes, 0);
        EXPECT_GT(s.probes, 5000 / int(2 * (s.lag + s.cfg.probePeriod)));
        EXPECT_LT(s.probes, 5000 / int(s.lag + s.cfg.probePeriod) + 2);
        // One sample per probe interval. Onset waits for the window to drop
        // the lost CUs' epochs and for the downward confirmations; recovery
        // needs only the confirmations.
        const uint32_t interval = s.lag + s.cfg.probePeriod + s.cfg.period;
        const uint32_t onset    = (s.cfg.window + s.cfg.confirmDown + 1) * interval;
        EXPECT_LT(s.until(128, 128, onset), onset);
        EXPECT_EQ(s.est.lastCount(), 128u);
        const uint32_t recovery = (s.cfg.confirmUp + 2) * interval;
        EXPECT_LT(s.until(0, c_nCu, recovery), recovery);
    }

    TEST(adaptive_sm_count, padded_strides_keep_the_hint)
    {
        // Under the hint the grid is below N_CU; padded strides keep the hint
        // confirmed without scheduled probe launches.
        Sim s;
        s.lag = 600;
        s.run(1000, c_nCu);
        ASSERT_LT(s.until(192, 192, 2000), 2000u);
        const int changes = s.changes;
        const int probes  = s.probes;
        s.run(20000, 192);
        EXPECT_EQ(s.est.hint(), 192u);
        EXPECT_EQ(s.changes, changes);
        EXPECT_EQ(s.est.lastCount(), 192u);
        EXPECT_LE(s.probes - probes, 2);
        // Recovery at the padded strides' pace.
        const uint32_t recovery
            = s.lag + (s.cfg.confirmUp + 2) * s.cfg.probeStride + 2 * s.cfg.period;
        EXPECT_LT(s.until(0, c_nCu, recovery), recovery);
    }

    TEST(adaptive_sm_count, tiny_problems)
    {
        Config cfg;
        cfg.tinyMflops = 100;
        EXPECT_TRUE(tinyProblem(128, 128, 1024, 1, cfg)); // 33.6 MFLOP
        EXPECT_FALSE(tinyProblem(256, 256, 1024, 1, cfg)); // 134 MFLOP
        EXPECT_FALSE(tinyProblem(128, 128, 1024, 4, cfg));
        EXPECT_TRUE(tinyProblem(128, 128, 1024, 0, cfg));
        cfg.tinyMflops = 0;
        EXPECT_FALSE(tinyProblem(1, 1, 1, 1, cfg));
    }

    TEST(adaptive_sm_count, launch_plan)
    {
        Config cfg;
        // Stride launches pad, reads every period; FORCE pads but never reads.
        auto p = planLaunch(cfg.probeStride, false, cfg);
        EXPECT_TRUE(p.pad);
        EXPECT_FALSE(p.probeLaunch);
        EXPECT_TRUE(p.read);
        p = planLaunch(cfg.probeStride + 1, false, cfg);
        EXPECT_FALSE(p.pad);
        EXPECT_FALSE(p.read);
        p = planLaunch(cfg.probeStride + 1, true, cfg);
        EXPECT_TRUE(p.pad && p.probeLaunch);
        cfg.force = 128;
        p         = planLaunch(cfg.probeStride, false, cfg);
        EXPECT_TRUE(p.pad);
        EXPECT_FALSE(p.read);
        cfg.probeStride = 0;
        EXPECT_FALSE(planLaunch(64, false, cfg).pad);
        // The scheduled probe is due at or past its epoch, wrap-safe.
        EXPECT_FALSE(probeDueAt(100, 0));
        EXPECT_FALSE(probeDueAt(99, 100));
        EXPECT_TRUE(probeDueAt(100, 100));
        EXPECT_TRUE(probeDueAt(3, 0xfffffffeu));
        // Hint-0 fallback only when the hint-0 pick can probe.
        EXPECT_TRUE(probeFallbackToHintZero(true, 128, false, true));
        EXPECT_FALSE(probeFallbackToHintZero(true, 128, false, false));
        EXPECT_FALSE(probeFallbackToHintZero(true, 128, true, true));
        EXPECT_FALSE(probeFallbackToHintZero(true, 0, false, true));
        EXPECT_FALSE(probeFallbackToHintZero(false, 128, false, true));
    }

    TEST(adaptive_sm_count, tiny_launches_take_no_epoch_or_probe)
    {
        // Every other launch is tiny: the scheduled probe and the stride land
        // on non-tiny launches, whose epochs stay consecutive.
        Sim s;
        s.cfg.probeStride = 0;
        s.lag             = 50;
        s.tiny            = [](uint32_t l) { return l % 2 == 1; };
        s.run(400, c_nCu);
        ASSERT_LT(s.until(192, 192, 4000), 4000u);
        EXPECT_EQ(s.hostEpoch, s.launches / 2 + s.launches % 2);
        const uint32_t at = s.probeAt;
        s.run(1, 192); // a tiny launch
        EXPECT_EQ(s.probeAt, at);
        EXPECT_EQ(s.est.hint(), 192u);
    }

    TEST(adaptive_sm_count, all_tiny_stream_drops_a_stale_hint)
    {
        Sim s;
        s.lag = 50;
        s.run(400, c_nCu);
        ASSERT_LT(s.until(128, 128, 2000), 2000u);
        // From here on only tiny launches: nothing re-measures, so the hint
        // drops after tinyStale() of them.
        s.tiny               = [](uint32_t) { return true; };
        const uint32_t epoch = s.hostEpoch;
        s.run(s.cfg.tinyStale() - 1, c_nCu);
        EXPECT_EQ(s.hint, 128u);
        s.run(1, c_nCu);
        EXPECT_EQ(s.hint, 0u);
        EXPECT_EQ(s.est.hint(), 0u);
        EXPECT_EQ(s.hostEpoch, epoch);
        // Non-tiny launches measure again.
        s.tiny = nullptr;
        EXPECT_LT(s.until(128, 128, 2000), 2000u);
    }

    TEST(adaptive_sm_count, masked_stream_backs_off_discovery_scans)
    {
        // A CU-masked stream never writes the other slots: the discovery scan
        // finds nothing and backs off to the periodic full scan.
        Sim s;
        s.probesUnderHint       = true;
        const uint32_t launches = 64 * s.cfg.period * s.cfg.fullScanEvery;
        s.run(launches, 128);
        EXPECT_EQ(s.est.hint(), 128u);
        const uint32_t reads = launches / s.cfg.period;
        EXPECT_LE(s.est.fullScans(), 2 * reads / s.cfg.fullScanEvery + 16);
    }

    TEST(adaptive_sm_count, slots_seen_late_are_found_by_full_scan)
    {
        Sim s;
        s.probesUnderHint = true;
        // Nothing written yet: every read scans until a slot shows up.
        s.run(24, 0);
        s.run(s.cfg.period, 64);
        EXPECT_EQ(s.est.knownSlots(), 64u);
        // Only 64 CUs so far, all in the window: the next read scans again.
        s.run(s.cfg.period, c_nCu);
        EXPECT_EQ(s.est.knownSlots(), c_nCu);
        EXPECT_EQ(s.est.hint(), 0u);
        // After a long shortage the discovery scan has backed off, but the
        // periodic full scan still finds returning CUs.
        Sim t;
        t.probesUnderHint = true;
        t.run(16 * t.cfg.period * t.cfg.fullScanEvery, 64);
        ASSERT_EQ(t.est.hint(), 64u);
        t.run(t.cfg.period * (t.cfg.fullScanEvery + 2 * t.cfg.confirmUp), c_nCu);
        EXPECT_EQ(t.est.knownSlots(), c_nCu);
        EXPECT_EQ(t.est.hint(), 0u);
    }

    TEST(adaptive_sm_count, future_epochs_clamp_to_age_zero)
    {
        Sim s;
        s.hostEpoch = 64;
        s.gpuRun(s.hostEpoch + 100, c_nCu);
        EXPECT_FALSE(s.est.read(s.slots.data(), s.hostEpoch, c_nCu, s.cfg));
        EXPECT_EQ(s.est.lastCount(), c_nCu);
        EXPECT_EQ(s.est.hint(), 0u);
    }

    TEST(adaptive_sm_count, stream_change_resets_the_estimate)
    {
        Sim s;
        EXPECT_FALSE(s.est.bindStream(1));
        s.run(64, c_nCu);
        s.run(16, 192);
        ASSERT_EQ(s.est.hint(), 192u);
        EXPECT_FALSE(s.est.bindStream(1));
        EXPECT_EQ(s.est.hint(), 192u);
        EXPECT_TRUE(s.est.bindStream(2));
        EXPECT_EQ(s.est.hint(), 0u);
        EXPECT_EQ(s.est.knownSlots(), c_nCu);
    }

    TEST(adaptive_sm_count, epoch_wrap)
    {
        Sim s;
        s.hostEpoch = 0xfffffe00u;
        s.run(256, c_nCu);
        s.run(512, 192); // crosses 2^32 with pre-wrap epochs in the slots
        EXPECT_EQ(s.est.hint(), 192u);
        EXPECT_EQ(s.est.lastCount(), 192u);
    }

    TEST(adaptive_sm_count, ancient_slots_are_ignored)
    {
        // Slots last written over 2^31 epochs ago read as "ahead" of the host
        // epoch; they must neither anchor the window nor count.
        Sim s;
        s.hostEpoch = 0x80000100u;
        s.gpuRun(0x50, c_nCu);
        s.run(64, 192);
        EXPECT_EQ(s.est.hint(), 192u);
        EXPECT_EQ(s.est.lastCount(), 192u);
        EXPECT_LT(s.hostEpoch - s.est.anchor(), 2 * s.cfg.probePeriod);
    }

    // bf16 TN 512x8192x8192 with AUTO Stream-K scheduling; skips when the
    // library has no solution.
    struct Bf16Gemm
    {
        static constexpr int64_t m = 512, n = 8192, k = 8192;
        hipblasLtHandle_t        handle = nullptr;
        hipblasLtMatmulDesc_t    desc   = nullptr;
        hipblasLtMatrixLayout_t  la = nullptr, lb = nullptr, lc = nullptr;
        float                    alpha = 1.f, beta = 0.f;

        Bf16Gemm()
        {
            EXPECT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            const hipblasOperation_t opT = HIPBLAS_OP_T;
            EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                          desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)),
                      HIPBLAS_STATUS_SUCCESS);
            const int32_t sk = HIPBLASLT_STREAMK_TILE_SCHEDULING_AUTO;
            EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                          desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &sk, sizeof(sk)),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, k, m, k),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, k, n, k),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16BF, m, n, m),
                      HIPBLAS_STATUS_SUCCESS);
        }

        ~Bf16Gemm()
        {
            hipblasLtMatrixLayoutDestroy(la);
            hipblasLtMatrixLayoutDestroy(lb);
            hipblasLtMatrixLayoutDestroy(lc);
            hipblasLtMatmulDescDestroy(desc);
            hipblasLtDestroy(handle);
        }

        int query(hipblasLtMatmulHeuristicResult_t* r)
        {
            hipblasLtMatmulPreference_t pref;
            EXPECT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            const uint64_t ws = uint64_t(128) << 20;
            EXPECT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)),
                      HIPBLAS_STATUS_SUCCESS);
            int count = 0;
            hipblasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, lc, pref, 1, r, &count);
            hipblasLtMatmulPreferenceDestroy(pref);
            return count;
        }
    };

    bool envSet(const char* name)
    {
        const char* v = std::getenv(name);
        return v && std::atoi(v) > 0;
    }

    // With adaptive mode off the heuristic result is untagged and reports the
    // plain workspace size the algo check returns.
    TEST(adaptive_sm_count_query, disabled_query_is_unchanged)
    {
        if(envSet("HIPBLASLT_ADAPTIVE_SM_COUNT") || envSet("HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE"))
            GTEST_SKIP() << "adaptive mode is on";
        Bf16Gemm                         g;
        hipblasLtMatmulHeuristicResult_t r{};
        if(g.query(&r) == 0)
            GTEST_SKIP() << "no bf16 TN solution in this library";
        uint32_t tag = 0;
        EXPECT_FALSE(algoTag(r.algo.data, &tag));
        size_t ws = 0;
        ASSERT_EQ(hipblaslt_ext::matmulIsAlgoSupported(
                      g.handle, g.desc, &g.alpha, g.la, g.lb, &g.beta, g.lc, g.lc, r.algo, ws),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(r.workspaceSize, ws);
    }

    // A tagged result launched under a forced hint with exactly the reported
    // workspaceSize must run (re-selected or re-gridded) without a workspace
    // error. Needs HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE (read once per process).
    TEST(adaptive_sm_count_query, forced_hint_fits_reported_workspace)
    {
        if(!envSet("HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE"))
            GTEST_SKIP() << "set HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE";
        Bf16Gemm                         g;
        hipblasLtMatmulHeuristicResult_t r{};
        if(g.query(&r) == 0)
            GTEST_SKIP() << "no bf16 TN solution in this library";
        uint32_t tag = 0;
        ASSERT_TRUE(algoTag(r.algo.data, &tag));

        void *a = nullptr, *b = nullptr, *d = nullptr, *ws = nullptr;
        ASSERT_EQ(hipMalloc(&a, g.m * g.k * 2), hipSuccess);
        ASSERT_EQ(hipMalloc(&b, g.k * g.n * 2), hipSuccess);
        ASSERT_EQ(hipMalloc(&d, g.m * g.n * 2), hipSuccess);
        if(r.workspaceSize)
            ASSERT_EQ(hipMalloc(&ws, r.workspaceSize), hipSuccess);
        hipStream_t stream;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        for(int i = 0; i < 32; ++i)
            EXPECT_EQ(hipblasLtMatmul(g.handle,
                                      g.desc,
                                      &g.alpha,
                                      a,
                                      g.la,
                                      b,
                                      g.lb,
                                      &g.beta,
                                      d,
                                      g.lc,
                                      d,
                                      g.lc,
                                      &r.algo,
                                      ws,
                                      r.workspaceSize,
                                      stream),
                      HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        hipStreamDestroy(stream);
        for(void* p : {a, b, d, ws})
            static_cast<void>(hipFree(p));
    }
} // namespace
