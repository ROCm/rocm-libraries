// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Adaptive sm_count_target through the C API: heuristic-query tagging and
// workspace, launches under a forced hint, probe readings, graph capture, the
// stream lock and launch-time re-selection. The mode is read from the
// environment once per process, so the cases for a mode the environment does
// not select skip; run with HIPBLASLT_ADAPTIVE_SM_COUNT=1 or
// HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE=<cus> (and
// HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE_PROBE=1) to cover them. The cotenant cases
// are Linux only; the exact one is manual (HIPBLASLT_TEST_COTENANT_PROBE=1,
// idle GPU). Host-only cases are in
// clients/tests/unit/adaptive_sm_count_gtest.cpp.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include "../../../library/src/amd_detail/rocblaslt/src/include/rocblaslt_adaptive_sm_controller.hpp"
#include "streamk_test_util.hpp"
#ifndef _WIN32
#include "cotenant.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" bool hipblaslt_adaptive_sm_state_for_test(void*       handle,
                                                     hipStream_t stream,
                                                     void*       state,
                                                     size_t      size);
extern "C" bool hipblaslt_adaptive_sm_launch_lock_for_test(
    void* handle, hipStream_t stream, bool tiny, bool* probe, bool* heldBetween, bool* heldAfter);
extern "C" size_t hipblaslt_adaptive_reselections_for_test(void* matmulDesc);
extern "C" int    hipblaslt_tuning_last_launch_for_test();

namespace rocblaslt::adaptive_sm
{
    inline void PrintTo(const StreamStateForTest& s, std::ostream* os)
    {
        *os << "{published=" << s.published << " epoch=" << s.epoch << " pending=" << s.pending
            << " missed=" << s.missed << " sinceTick=" << s.sinceTick << " due=" << s.due
            << " probesRead=" << s.probesRead << " lastCount=" << s.lastCount << "}";
    }
} // namespace rocblaslt::adaptive_sm

namespace
{
    using namespace rocblaslt::adaptive_sm;
    using namespace streamk_test;

    // BF16 NN, large enough to probe (not tiny) and to rank Stream-K solutions.
    constexpr int64_t kM = 512, kN = 8192, kK = 8192;

    class AdaptiveSmCount_pre_checkin : public ::testing::Test
    {
    protected:
        int64_t      m = kM, n = kN, k = kK;
        Resources    r;
        hipStream_t  stream = nullptr;
        const Config cfg    = Config::fromEnv();
        float        alpha = 1.0f, beta = 0.0f;

        void SetUp() override
        {
            if(!gpuAvailable())
                GTEST_SKIP() << "No GPU available";
            createProblem(r, m, n, k);
            if(IsSkipped() || HasFatalFailure())
                return;
            setStreamK(HIPBLASLT_STREAMK_TILE_SCHEDULING_AUTO);
            ASSERT_EQ(hipMalloc(&r.dA, m * k * sizeof(uint16_t)), hipSuccess);
            ASSERT_EQ(hipMalloc(&r.dB, k * n * sizeof(uint16_t)), hipSuccess);
            r.dD.assign(1, nullptr);
            ASSERT_EQ(hipMalloc(&r.dD[0], m * n * sizeof(uint16_t)), hipSuccess);
            r.dWs.assign(1, nullptr);
            ASSERT_EQ(hipMalloc(&r.dWs[0], kWsBudgetBytes), hipSuccess);
            r.streams.assign(1, nullptr);
            ASSERT_EQ(hipStreamCreate(&r.streams[0]), hipSuccess);
            stream = r.streams[0];
        }

        void TearDown() override
        {
            if(stream)
                static_cast<void>(hipStreamSynchronize(stream));
        }

        void setStreamK(int32_t sk)
        {
            ASSERT_EQ(
                hipblasLtMatmulDescSetAttribute(
                    r.desc, HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT, &sk, sizeof(sk)),
                HIPBLAS_STATUS_SUCCESS);
        }

        // Up to `count` heuristic results under workspace limit `limit`.
        std::vector<hipblasLtMatmulHeuristicResult_t> query(int      count = 1,
                                                            uint64_t limit = kWsBudgetBytes)
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> out(count);
            hipblasLtMatmulPreference_t                   pref = nullptr;
            if(hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS)
                return {};
            int returned = 0;
            if(hipblasLtMatmulPreferenceSetAttribute(
                   pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit))
                   != HIPBLAS_STATUS_SUCCESS
               || hipblasLtMatmulAlgoGetHeuristic(r.handle,
                                                  r.desc,
                                                  r.layA,
                                                  r.layB,
                                                  r.layD,
                                                  r.layD,
                                                  pref,
                                                  count,
                                                  out.data(),
                                                  &returned)
                      != HIPBLAS_STATUS_SUCCESS)
                returned = 0;
            hipblasLtMatmulPreferenceDestroy(pref);
            out.resize(returned);
            return out;
        }

        // Workspace `algo` needs under the desc's current settings.
        size_t required(hipblasLtMatmulAlgo_t algo)
        {
            size_t bytes = 0;
            EXPECT_EQ(
                hipblaslt_ext::matmulIsAlgoSupported(
                    r.handle, r.desc, &alpha, r.layA, r.layB, &beta, r.layD, r.layD, algo, bytes),
                HIPBLAS_STATUS_SUCCESS);
            return bytes;
        }

        hipblasStatus_t run(const hipblasLtMatmulAlgo_t& algo, size_t workspace)
        {
            return hipblasLtMatmul(r.handle,
                                   r.desc,
                                   &alpha,
                                   r.dA,
                                   r.layA,
                                   r.dB,
                                   r.layB,
                                   &beta,
                                   r.dD[0],
                                   r.layD,
                                   r.dD[0],
                                   r.layD,
                                   &algo,
                                   r.dWs[0],
                                   workspace,
                                   stream);
        }

        StreamStateForTest state()
        {
            StreamStateForTest st{};
            EXPECT_TRUE(hipblaslt_adaptive_sm_state_for_test(r.handle, stream, &st, sizeof(st)));
            return st;
        }

        size_t reselections()
        {
            return hipblaslt_adaptive_reselections_for_test(r.desc);
        }

        // Under FORCE the query is ranked and tagged for the forced hint and
        // every launch takes it, so nothing re-selects. Probes run only with
        // FORCE_PROBE, and their readings never move the hint.
        void checkForcedHint(int32_t sk)
        {
            if(!cfg.force)
                GTEST_SKIP() << "set HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE";
            setStreamK(sk);
            const auto results = query();
            if(results.empty())
                GTEST_SKIP() << "No solution for this problem in the loaded library";
            const auto& result = results[0];
            uint32_t    tag    = 0;
            ASSERT_TRUE(algoTag(result.algo.data, &tag));
            EXPECT_EQ(tag, cfg.force);

            for(uint32_t i = 0; i < cfg.probeEvery + 1; ++i)
                ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(reselections(), 0u);
            const StreamStateForTest st = state();
            EXPECT_EQ(st.published, 0u) << "nothing is published under FORCE";
            if(cfg.forceProbe)
                EXPECT_GT(st.epoch, 0u);
            else
                EXPECT_EQ(st.epoch, 0u);
        }

        // The top result retagged as if ranked for `hint`; skips without
        // FORCE or a solution.
        void retaggedResult(uint32_t hint, hipblasLtMatmulHeuristicResult_t* out)
        {
            if(!cfg.force)
                GTEST_SKIP() << "set HIPBLASLT_ADAPTIVE_SM_COUNT_FORCE";
            const auto results = query();
            if(results.empty())
                GTEST_SKIP() << "No solution for this problem in the loaded library";
            *out         = results[0];
            uint32_t tag = 0;
            ASSERT_TRUE(algoTag(out->algo.data, &tag));
            ASSERT_NE(hint, cfg.force);
            tagAlgo(out->algo.data, hint);
        }
    };

    // Adaptive mode reports min(limit, max(plain, c_adaptiveWorkspace)) for the
    // tagged top result; with it off the result is untagged and reports the
    // plain size.
    TEST_F(AdaptiveSmCount_pre_checkin, QueryReportsTheAdaptiveWorkspace)
    {
        for(uint64_t limit : {kWsBudgetBytes / 4, kWsBudgetBytes * 2})
        {
            SCOPED_TRACE("limit " + std::to_string(limit));
            const auto results = query(1, limit);
            if(results.empty())
                GTEST_SKIP() << "No solution for this problem in the loaded library";
            const auto& result = results[0];
            uint32_t    tag    = 0;
            EXPECT_EQ(algoTag(result.algo.data, &tag), cfg.enabled);
            const size_t plain = required(result.algo);
            ASSERT_LE(plain, limit);
            const size_t expected
                = cfg.enabled ? std::min<size_t>(limit, std::max(plain, c_adaptiveWorkspace))
                              : plain;
            EXPECT_EQ(result.workspaceSize, expected);
        }
    }

    // StreamK OFF never takes the adaptive hint.
    TEST_F(AdaptiveSmCount_pre_checkin, StreamKOffIsUntagged)
    {
        if(!cfg.enabled)
            GTEST_SKIP() << "adaptive mode is off";
        setStreamK(HIPBLASLT_STREAMK_TILE_SCHEDULING_OFF);
        const auto results = query();
        if(results.empty())
            GTEST_SKIP() << "No solution for this problem in the loaded library";
        uint32_t tag = 0;
        EXPECT_FALSE(algoTag(results[0].algo.data, &tag));
    }

    TEST_F(AdaptiveSmCount_pre_checkin, ForcedHintForQueryAndLaunchAuto)
    {
        checkForcedHint(HIPBLASLT_STREAMK_TILE_SCHEDULING_AUTO);
    }

    TEST_F(AdaptiveSmCount_pre_checkin, ForcedHintForQueryAndLaunchOn)
    {
        checkForcedHint(HIPBLASLT_STREAMK_TILE_SCHEDULING_ON);
    }

    // A probe recorded on the stream completes and is read by the next launch.
    TEST_F(AdaptiveSmCount_pre_checkin, ProbeProducesAReading)
    {
        if(!cfg.force || !cfg.forceProbe)
            GTEST_SKIP() << "needs FORCE and FORCE_PROBE, so the solution and grid stay fixed";
        const auto results = query();
        if(results.empty())
            GTEST_SKIP() << "No solution for this problem in the loaded library";
        const auto& result = results[0];
        for(uint32_t i = 0; i < cfg.probeEvery; ++i)
            ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const StreamStateForTest before = state();
        if(!before.pending)
            GTEST_SKIP() << "no probe recorded: the launched solution does not probe";

        ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
        const StreamStateForTest after = state();
        EXPECT_GT(after.probesRead, before.probesRead);
        int device = 0, cus = 0;
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        ASSERT_EQ(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device),
                  hipSuccess);
        EXPECT_GT(after.lastCount, 0u);
        EXPECT_LE(after.lastCount, uint32_t(cus));
        EXPECT_EQ(after.published, 0u) << "the reading is discarded";
    }

#ifndef _WIN32
    // The cotenant holds this fraction of the CUs.
    constexpr int kHeldCuDivisor = 4;
    // One cotenant workgroup per CU, reserving all of its LDS, so no GEMM
    // or probe-only workgroup fits beside it.
    constexpr int kWholeCuOccupancy = 1;
    // Opt-in for the manual exact-count test.
    constexpr const char* kEnvCotenantProbe = "HIPBLASLT_TEST_COTENANT_PROBE";
    // Readings per phase.
    constexpr int kReadings = 3;

    // Probes beside a cotenant holding `held` whole CUs. Runs under FORCE with
    // FORCE_PROBE, so the solution and grid stay fixed; with FORCE at most the
    // free CU count, the probe-only padding workgroups reach the free CUs the
    // StreamK grid does not.
    class AdaptiveSmCountCotenant_pre_checkin : public AdaptiveSmCount_pre_checkin
    {
    protected:
        hipblasLtMatmulHeuristicResult_t result{};
        int                              cus  = 0;
        int                              held = 0;

        // Skips when the mode, library or device does not fit.
        void prepare()
        {
            if(!cfg.force || !cfg.forceProbe)
                GTEST_SKIP() << "needs FORCE and FORCE_PROBE, so the solution and grid stay fixed";
            const auto results = query();
            if(results.empty())
                GTEST_SKIP() << "No solution for this problem in the loaded library";
            result = results[0];

            int device = 0, ldsPerCu = 0, optin = 0;
            ASSERT_EQ(hipGetDevice(&device), hipSuccess);
            ASSERT_EQ(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device),
                      hipSuccess);
            ASSERT_EQ(hipDeviceGetAttribute(
                          &ldsPerCu, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device),
                      hipSuccess);
            ASSERT_EQ(
                hipDeviceGetAttribute(&optin, hipDeviceAttributeSharedMemPerBlockOptin, device),
                hipSuccess);
            try
            {
                hipblaslt_cotenant::lds_reservation(ldsPerCu, optin, kWholeCuOccupancy);
            }
            catch(const std::runtime_error& e)
            {
                GTEST_SKIP() << "a workgroup cannot reserve a whole CU's LDS: " << e.what();
            }
            held = cus / kHeldCuDivisor;
            if(held == 0)
                GTEST_SKIP() << "fewer than " << kHeldCuDivisor << " CUs";
            if(cfg.force > uint32_t(cus - held))
                GTEST_SKIP() << "FORCE must not exceed the " << cus - held
                             << " CUs the cotenant leaves free";
        }

        // Launches until a probe is pending, waits for it, then reads it
        // with one more launch. Launch failures are fatal; `*probed` is false
        // when no launch probes.
        void nextReading(bool* probed, uint32_t* count)
        {
            *probed             = false;
            const uint32_t read = state().probesRead;
            for(uint32_t i = 0; i <= cfg.probeEvery && !state().pending; ++i)
                ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
            if(!state().pending)
                return;
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
            const StreamStateForTest st = state();
            ASSERT_EQ(st.probesRead, read + 1);
            *probed = true;
            *count  = st.lastCount;
        }

        // Starts the cotenant and drops any probe launched before it was
        // resident. Skips if other work keeps its CUs from becoming free.
        void startCotenant(std::optional<hipblaslt_cotenant::Scoped>* cotenant)
        {
            try
            {
                cotenant->emplace(held, kWholeCuOccupancy, stream);
            }
            catch(const std::runtime_error& e)
            {
                GTEST_SKIP() << "cotenant did not become resident: " << e.what();
            }
            if(state().pending)
                ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
        }

        // Takes kReadings readings beside the cotenant, checking each with
        // `check`.
        template <typename Check>
        void readBesideCotenant(Check check)
        {
            // Destroyed on every exit, so the cotenant stops on failure too.
            std::optional<hipblaslt_cotenant::Scoped> cotenant;
            ASSERT_NO_FATAL_FAILURE(startCotenant(&cotenant));
            if(IsSkipped())
                return;
            for(int i = 0; i < kReadings; ++i)
            {
                SCOPED_TRACE("reading " + std::to_string(i));
                bool     probed = false;
                uint32_t count  = 0;
                ASSERT_NO_FATAL_FAILURE(nextReading(&probed, &count));
                if(!probed && i == 0)
                    GTEST_SKIP() << "no probe reading: the launched solution does not probe";
                ASSERT_TRUE(probed) << "no probe reading under the cotenant";
                check(count);
            }
        }
    };

    // Other work can only take CUs, so the reading never exceeds what the
    // cotenant leaves free.
    TEST_F(AdaptiveSmCountCotenant_pre_checkin, ProbeCountsAtMostTheCusACotenantLeaves)
    {
        ASSERT_NO_FATAL_FAILURE(prepare());
        if(IsSkipped())
            return;
        readBesideCotenant([&](uint32_t count) {
            EXPECT_GT(count, 0u);
            EXPECT_LE(count, uint32_t(cus - held));
        });
    }

    // Manual: other processes that take CUs change the count, so this runs
    // only when HIPBLASLT_TEST_COTENANT_PROBE=1, on an otherwise idle GPU.
    TEST_F(AdaptiveSmCountCotenant_pre_checkin, ProbeCountsExactlyTheCusACotenantLeaves)
    {
        const char* optIn = std::getenv(kEnvCotenantProbe);
        if(!optIn || std::string(optIn) != "1")
            GTEST_SKIP() << "manual: set " << kEnvCotenantProbe
                         << "=1 on an otherwise idle GPU; other processes that take CUs "
                            "change the count";
        ASSERT_NO_FATAL_FAILURE(prepare());
        if(IsSkipped())
            return;
        bool     probed = false;
        uint32_t idle   = 0;
        ASSERT_NO_FATAL_FAILURE(nextReading(&probed, &idle));
        if(!probed)
            GTEST_SKIP() << "no probe reading: the launched solution does not probe";
        if(idle != uint32_t(cus))
            GTEST_SKIP() << "idle reading " << idle << " of " << cus
                         << " CUs: other work occupies CUs";
        readBesideCotenant([&](uint32_t count) { EXPECT_EQ(count, idle - held); });
    }
#endif // _WIN32

    // A launch on a capturing stream neither takes a probe nor counts toward
    // the next tick: the stream's state after the capture is the state before.
    TEST_F(AdaptiveSmCount_pre_checkin, CaptureLeavesTheStateAlone)
    {
        if(!cfg.enabled || !cfg.tracksLaunches())
            GTEST_SKIP() << "launches leave the state alone in this mode";
        if(cfg.probeEvery < 2)
            GTEST_SKIP() << "needs a launch between ticks";
        const auto results = query();
        if(results.empty())
            GTEST_SKIP() << "No solution for this problem in the loaded library";
        const auto& result = results[0];

        // A probe falls due, completes and is read, leaving the stream one
        // launch past a tick.
        for(uint32_t i = 0; i < cfg.probeEvery; ++i)
            ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const bool probed = state().pending;
        ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        const StreamStateForTest before = state();
        ASSERT_NE(before.sinceTick, 0u);
        if(probed)
            ASSERT_GT(before.probesRead, 0u);

        ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        for(uint32_t i = 0; i < 3 * cfg.probeEvery; ++i)
            EXPECT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
        hipGraph_t graph = nullptr;
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
        hipGraphExec_t   exec = nullptr;
        const hipError_t inst = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        if(inst == hipSuccess)
        {
            EXPECT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
            EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
            static_cast<void>(hipGraphExecDestroy(exec));
        }
        static_cast<void>(hipGraphDestroy(graph));
        ASSERT_EQ(inst, hipSuccess);

        EXPECT_EQ(state(), before);

        // The next live launch carries on from there.
        ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(state().sinceTick, (before.sinceTick + 1) % cfg.probeEvery);
    }

    // A probe launch holds its stream's adaptive lock from its first phase to
    // its second, so another thread cannot enqueue a later probe in between.
    TEST_F(AdaptiveSmCount_pre_checkin, ProbeLaunchHoldsTheStreamLockBetweenItsPhases)
    {
        if(!cfg.enabled || !cfg.tracksLaunches())
            GTEST_SKIP() << "launches probe only in adaptive mode or under FORCE_PROBE";
        bool probe = false, heldBetween = false, heldAfter = false;
        for(uint32_t i = 0; i < cfg.probeEvery && !probe; ++i)
            ASSERT_TRUE(hipblaslt_adaptive_sm_launch_lock_for_test(
                r.handle, stream, /*tiny=*/false, &probe, &heldBetween, &heldAfter));
        ASSERT_TRUE(probe);
        EXPECT_TRUE(heldBetween);
        EXPECT_FALSE(heldAfter);
    }

    // A launch that does not probe holds the lock only inside its first phase.
    TEST_F(AdaptiveSmCount_pre_checkin, NonProbeLaunchReleasesTheStreamLock)
    {
        if(!cfg.enabled || !cfg.tracksLaunches())
            GTEST_SKIP() << "launches leave the state alone in this mode";
        for(uint32_t i = 0; i < cfg.probeEvery; ++i)
        {
            SCOPED_TRACE("launch " + std::to_string(i));
            bool probe = true, heldBetween = true, heldAfter = true;
            ASSERT_TRUE(hipblaslt_adaptive_sm_launch_lock_for_test(
                r.handle, stream, /*tiny=*/true, &probe, &heldBetween, &heldAfter));
            EXPECT_FALSE(probe);
            EXPECT_FALSE(heldBetween);
            EXPECT_FALSE(heldAfter);
        }
    }

    // Under FORCE a result tagged with another hint is re-selected at launch,
    // once: the repeat launch reuses the memoised choice.
    TEST_F(AdaptiveSmCount_pre_checkin, LaunchReselectsOnceForAnotherHint)
    {
        hipblasLtMatmulHeuristicResult_t result{};
        retaggedResult(cfg.force ^ 1, &result);
        if(IsSkipped() || HasFatalFailure())
            return;
        EXPECT_EQ(reselections(), 0u);
        ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(reselections(), 1u);
        ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(reselections(), 1u);
    }

    // A launch on a capturing stream runs a result tagged with another hint
    // as given; the next live launch re-selects it.
    TEST_F(AdaptiveSmCount_pre_checkin, CapturedLaunchDoesNotReselect)
    {
        hipblasLtMatmulHeuristicResult_t result{};
        retaggedResult(cfg.force ^ 1, &result);
        if(IsSkipped() || HasFatalFailure())
            return;
        ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        const hipblasStatus_t captured = run(result.algo, result.workspaceSize);
        hipGraph_t            graph    = nullptr;
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
        hipGraphExec_t   exec = nullptr;
        const hipError_t inst = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        if(inst == hipSuccess)
        {
            EXPECT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
            EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
            static_cast<void>(hipGraphExecDestroy(exec));
        }
        static_cast<void>(hipGraphDestroy(graph));
        ASSERT_EQ(captured, HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(inst, hipSuccess);
        EXPECT_EQ(reselections(), 0u);

        ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(reselections(), 1u);
    }

    // Under uniform summation order a result tagged with another hint is
    // re-selected too, and the launch passes the order's launch check.
    TEST_F(AdaptiveSmCount_pre_checkin, UniformSummationOrderReselects)
    {
        const int32_t uso = 1;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      r.desc, HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT, &uso, sizeof(uso)),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatmulHeuristicResult_t result{};
        retaggedResult(cfg.force ^ 1, &result);
        if(IsSkipped() || HasFatalFailure())
            return;
        ASSERT_EQ(run(result.algo, result.workspaceSize), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(reselections(), 1u);
    }

    // A long K, so that the top result uses a workspace when it has one.
    class AdaptiveSmCountLongK_pre_checkin : public AdaptiveSmCount_pre_checkin
    {
    protected:
        AdaptiveSmCountLongK_pre_checkin()
        {
            m = 256;
            n = 256;
            k = 16384;
        }
    };

    // The pick is checked against the caller's workspace, not the query's: a
    // StreamK solution needs less workspace when given less, so even without
    // one the launch re-selects to it and runs.
    TEST_F(AdaptiveSmCountLongK_pre_checkin, ReselectsWithoutWorkspace)
    {
        hipblasLtMatmulHeuristicResult_t result{};
        retaggedResult(cfg.force ^ 1, &result);
        if(IsSkipped() || HasFatalFailure())
            return;
        static_cast<void>(hipblaslt_tuning_last_launch_for_test());
        ASSERT_EQ(run(result.algo, 0), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(reselections(), 1u);
        // Under FORCE the launch ranks as the query did, so its pick is the
        // query's top result.
        EXPECT_EQ(hipblaslt_tuning_last_launch_for_test(),
                  hipblaslt_ext::getIndexFromAlgo(result.algo));
    }
} // namespace
