/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// Regression test for the multi-stream Stream-K deadlock (ROCM-29670).
//
// Stream-K's remainder path synchronises workgroups through a flag region. That
// region used to be shared by every stream on a handle, so two Stream-K kernels
// running concurrently could clear a flag the other was still spinning on and
// the waiting workgroup would spin forever. Each stream now gets its own block
// of handle->StreamKFlags.
//
// The test fans the same GEMM out over many streams and joins them back, which
// raises the concurrency enough to hit the hazard: against a library without
// the fix, gfx950 stops draining on the first iteration.
//
// Progress is polled with hipStreamQuery against a stall budget rather than a
// blocking wait, so a deadlock fails the case with a message instead of hanging
// the job. The budget is re-armed every iteration and only starts once the
// iteration is fully enqueued, so it measures the queue failing to drain, not a
// slow device or slow host-side work such as a first code object load. Once it
// does expire the process exits without unwinding, because the queue is wedged
// and hipFree and hipStreamDestroy would block too.
//
// The case skips unless the library's own launch summary shows the GEMM taking
// the flag-synchronised remainder path, since a pass would be meaningless
// otherwise.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include "streamk_test_util.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Re-reads TENSILE_DB inside libhipblaslt.so; see streamk5_test.cpp.
extern "C" void hipblaslt_debug_reload();

namespace
{
    using namespace streamk_test;

    // Shape the customer workload deadlocked on.
    constexpr int64_t kM = 1024;
    constexpr int64_t kN = 7168;
    constexpr int64_t kK = 4096;

    // The hazard needs a grid that does not divide the tile count. The CU
    // budget nudges the heuristic toward one for this shape, but it is only a
    // hint, so whether a device actually gets there is read back from the
    // launch. The API attribute is used because the TENSILE_PERSISTENT_*
    // variables are cached on first read.
    constexpr int32_t kSmCountTarget = 96;

    constexpr int kStreams    = 32;
    constexpr int kIterations = 500;

    // How long the queue may go without draining a single iteration, counted
    // from when that iteration is fully enqueued, before the run is called
    // deadlocked. A slow device takes many of these in a row and is never
    // killed for being slow.
    constexpr int kStallSeconds = 60;

    // Wall-clock budget for the concurrent-handle case. Unlike the deadlock
    // test, that case does not re-arm a per-iteration stall timer.
    constexpr int kDeadlineSeconds = 120;

    constexpr char kLaunchSummaryBit[]    = "0x200000";
    constexpr char kLaunchSummaryMarker[] = "TensileLite::StreamK LAUNCH SUMMARY";

    void setEnv(const char* name, const char* value)
    {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }

    void unsetEnv(const char* name)
    {
#ifdef _WIN32
        _putenv_s(name, "");
#else
        unsetenv(name);
#endif
    }

    // Turns on the Stream-K launch summary and captures std::cerr for its
    // lifetime, restoring both on the way out.
    class LaunchSummaryCapture
    {
    public:
        LaunchSummaryCapture()
            : m_hadPrior(std::getenv("TENSILE_DB") != nullptr)
            , m_prior(m_hadPrior ? std::getenv("TENSILE_DB") : "")
            , m_savedCerr(std::cerr.rdbuf(m_captured.rdbuf()))
        {
            setEnv("TENSILE_DB", kLaunchSummaryBit);
            hipblaslt_debug_reload();
        }

        ~LaunchSummaryCapture()
        {
            if(m_hadPrior)
                setEnv("TENSILE_DB", m_prior.c_str());
            else
                unsetEnv("TENSILE_DB");
            hipblaslt_debug_reload();
            std::cerr.rdbuf(m_savedCerr);
        }

        LaunchSummaryCapture(const LaunchSummaryCapture&)            = delete;
        LaunchSummaryCapture& operator=(const LaunchSummaryCapture&) = delete;

        // The first summary printed so far, or empty if there is none.
        std::string summary() const
        {
            const std::string out = m_captured.str();
            const auto        at  = out.find(kLaunchSummaryMarker);
            return at == std::string::npos ? std::string() : out.substr(at);
        }

    private:
        bool               m_hadPrior;
        std::string        m_prior;
        std::ostringstream m_captured;
        std::streambuf*    m_savedCerr;
    };

    // Value of a "key = value" line in a launch summary, or empty.
    std::string summaryField(const std::string& summary, const std::string& key)
    {
        std::istringstream in(summary);
        for(std::string line; std::getline(in, line);)
        {
            const auto eq = line.find(" = ");
            if(eq == std::string::npos)
                continue;
            const auto first = line.find_first_not_of(' ');
            const auto last  = line.find_last_not_of(' ', eq);
            if(first <= last && line.compare(first, last - first + 1, key) == 0)
                return line.substr(eq + 3);
        }
        return {};
    }

    TEST(StreamKMultiStream, ConcurrentStreamsDoNotDeadlock)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        Resources r;
        createProblem(r, kM, kN, kK);
        if(::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure())
            return;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(r.desc,
                                                  HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                                  &kSmCountTarget,
                                                  sizeof(kSmCountTarget)),
                  HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulHeuristicResult_t heuristic{};
        int                              returned = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(r.handle,
                                                  r.desc,
                                                  r.layA,
                                                  r.layB,
                                                  r.layD,
                                                  r.layD,
                                                  r.pref,
                                                  1,
                                                  &heuristic,
                                                  &returned),
                  HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            GTEST_SKIP() << "No solution for " << kM << "x" << kN << "x" << kK << " on this device";

        // A and B are read-only, so every stream can share them. Only D and
        // the workspace have to be private, and the workspace is sized from
        // what the chosen solution actually asks for rather than the budget,
        // which keeps 32 streams inside a gigabyte.
        const size_t bytesA  = static_cast<size_t>(kM * kK) * sizeof(uint16_t);
        const size_t bytesB  = static_cast<size_t>(kK * kN) * sizeof(uint16_t);
        const size_t bytesD  = static_cast<size_t>(kM * kN) * sizeof(uint16_t);
        const size_t bytesWs = heuristic.workspaceSize;

        if(hipMalloc(&r.dA, bytesA) != hipSuccess || hipMalloc(&r.dB, bytesB) != hipSuccess)
            GTEST_SKIP() << "Insufficient device memory for the shared operands";
        static_cast<void>(hipMemset(r.dA, 0, bytesA));
        static_cast<void>(hipMemset(r.dB, 0, bytesB));

        for(int i = 0; i < kStreams; ++i)
        {
            void* d  = nullptr;
            void* ws = nullptr;
            if(hipMalloc(&d, bytesD) != hipSuccess
               || (bytesWs > 0 && hipMalloc(&ws, bytesWs) != hipSuccess))
            {
                static_cast<void>(hipFree(d));
                static_cast<void>(hipFree(ws));
                GTEST_SKIP() << "Insufficient device memory for " << kStreams << " streams";
            }
            r.dD.push_back(d);
            r.dWs.push_back(ws);

            hipStream_t s = nullptr;
            if(hipStreamCreate(&s) != hipSuccess)
                GTEST_SKIP() << "Could not create " << kStreams << " streams";
            r.streams.push_back(s);
        }

        const float alpha = 1.0f, beta = 0.0f;

        // One GEMM on its own first, to read what the library actually
        // launches. Alone on one stream the hazard cannot occur, so it is
        // waited on without a budget; it is also the first launch of this
        // solution, whose code object load can take minutes on some platforms.
        std::string summary;
        {
            LaunchSummaryCapture capture;
            ASSERT_EQ(hipblasLtMatmul(r.handle,
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
                                      &heuristic.algo,
                                      r.dWs[0],
                                      bytesWs,
                                      r.streams[0]),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipStreamSynchronize(r.streams[0]), hipSuccess);
            summary = capture.summary();
        }
        if(summary.empty())
            GTEST_SKIP()
                << "The chosen solution is not Stream-K, so the remainder path is not reached";

        // The flags are only read by a static Stream-K launch with tree
        // reduction whose grid does not divide the tile count, which is what
        // leaves partial tiles to fix up. SM_COUNT_TARGET is only a hint, so
        // the heuristic is free to pick a grid that divides evenly.
        const bool streamK = summary.find("TileProcessingStrategy=StreamK") != std::string::npos;
        const bool staticGrid
            = summary.find("EffectiveWorkAssignment=StaticGrid") != std::string::npos;
        const std::string reduction = summaryField(summary, "reduction");
        const std::string dpOnly    = summaryField(summary, "dpOnly");
        const std::string gridText  = summaryField(summary, "final");
        const std::string tilesText = summaryField(summary, "tiles");
        const size_t      grid      = std::strtoull(gridText.c_str(), nullptr, 10);
        const size_t      tiles     = std::strtoull(tilesText.c_str(), nullptr, 10);
        if(!streamK || !staticGrid || reduction != "tree" || dpOnly != "no" || grid == 0
           || tiles % grid == 0)
            GTEST_SKIP() << "The Stream-K launch does not reach the remainder path (Stream-K: "
                         << (streamK ? "yes" : "no")
                         << ", static grid: " << (staticGrid ? "yes" : "no")
                         << ", reduction: " << reduction << ", dpOnly: " << dpOnly
                         << ", grid: " << gridText << ", tiles: " << tilesText
                         << "). Setting TENSILE_PERSISTENT_FIXED_GRID before the process "
                            "starts, to a grid that does not divide the tile count, forces it.";

        hipStream_t main = nullptr;
        ASSERT_EQ(hipStreamCreate(&main), hipSuccess);
        r.streams.push_back(main);

        for(int iter = 0; iter < kIterations; ++iter)
        {
            // Fan out from one stream and join back, so the side streams run
            // their Stream-K kernels against each other rather than in turn.
            hipEvent_t fork = nullptr;
            ASSERT_EQ(hipEventCreateWithFlags(&fork, hipEventDisableTiming), hipSuccess);
            ASSERT_EQ(hipEventRecord(fork, main), hipSuccess);

            std::vector<hipEvent_t> joins;
            joins.reserve(kStreams);
            for(int i = 0; i < kStreams; ++i)
            {
                ASSERT_EQ(hipStreamWaitEvent(r.streams[i], fork, 0), hipSuccess);
                ASSERT_EQ(hipblasLtMatmul(r.handle,
                                          r.desc,
                                          &alpha,
                                          r.dA,
                                          r.layA,
                                          r.dB,
                                          r.layB,
                                          &beta,
                                          r.dD[i],
                                          r.layD,
                                          r.dD[i],
                                          r.layD,
                                          &heuristic.algo,
                                          r.dWs[i],
                                          bytesWs,
                                          r.streams[i]),
                          HIPBLAS_STATUS_SUCCESS);

                hipEvent_t join = nullptr;
                ASSERT_EQ(hipEventCreateWithFlags(&join, hipEventDisableTiming), hipSuccess);
                ASSERT_EQ(hipEventRecord(join, r.streams[i]), hipSuccess);
                joins.push_back(join);
            }
            for(auto e : joins)
                ASSERT_EQ(hipStreamWaitEvent(main, e, 0), hipSuccess);

            const auto stallAt
                = std::chrono::steady_clock::now() + std::chrono::seconds(kStallSeconds);

            while(hipStreamQuery(main) == hipErrorNotReady)
            {
                if(std::chrono::steady_clock::now() > stallAt)
                {
                    std::fprintf(stderr,
                                 "\n[  FAILED  ] StreamKMultiStream.ConcurrentStreamsDoNotDeadlock\n"
                                 "  Iteration %d of %d has not drained in %d s across %d streams.\n"
                                 "  The Stream-K flag region may be shared across streams;\n"
                                 "  see ROCM-29670. Exiting without cleanup because the queue\n"
                                 "  is wedged and hipFree would block as well.\n\n",
                                 iter,
                                 kIterations,
                                 kStallSeconds,
                                 kStreams);
                    std::fflush(stderr);
                    ADD_FAILURE() << kStreams << " concurrent streams stopped draining at iteration "
                                  << iter;
                    std::_Exit(1);
                }
            }

            static_cast<void>(hipEventDestroy(fork));
            for(auto e : joins)
                static_cast<void>(hipEventDestroy(e));
        }

        EXPECT_EQ(hipStreamSynchronize(main), hipSuccess);
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    // A handle owns a fixed number of Stream-K flag blocks, one per distinct
    // stream, claimed on that stream's first matmul and held until the handle
    // is destroyed. Past that the library reports an error.
    //
    // Refusing is the deliberate choice: handing back an already-claimed block
    // would put two streams back on shared flags, which is the deadlock this
    // separation exists to prevent, and the caller would be told everything
    // succeeded while a workgroup spins forever. This test pins that contract
    // down so it cannot quietly regress to sharing.
    //
    // Raising c_syncSkStreamSlots is a fine thing to do; it just has to be
    // matched here.
    constexpr int kStreamCapacity = 64;

    TEST(StreamKMultiStream, StreamsBeyondCapacityAreRejected)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        // A block is claimed on any matmul, not only a Stream-K one, so this
        // can stay small and fast.
        constexpr int64_t m = 1024, n = 1024, k = 1024;

        Resources r;
        createProblem(r, m, n, k);
        if(::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure())
            return;

        hipblasLtMatmulHeuristicResult_t heuristic{};
        int                              returned = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(r.handle,
                                                  r.desc,
                                                  r.layA,
                                                  r.layB,
                                                  r.layD,
                                                  r.layD,
                                                  r.pref,
                                                  1,
                                                  &heuristic,
                                                  &returned),
                  HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            GTEST_SKIP() << "No solution for " << m << "x" << n << "x" << k << " on this device";

        const size_t bytesA  = static_cast<size_t>(m * k) * sizeof(uint16_t);
        const size_t bytesB  = static_cast<size_t>(k * n) * sizeof(uint16_t);
        const size_t bytesD  = static_cast<size_t>(m * n) * sizeof(uint16_t);
        const size_t bytesWs = heuristic.workspaceSize;

        void* ws = nullptr;
        if(hipMalloc(&r.dA, bytesA) != hipSuccess || hipMalloc(&r.dB, bytesB) != hipSuccess
           || (bytesWs > 0 && hipMalloc(&ws, bytesWs) != hipSuccess))
            GTEST_SKIP() << "Insufficient device memory";
        r.dWs.push_back(ws);
        static_cast<void>(hipMemset(r.dA, 0, bytesA));
        static_cast<void>(hipMemset(r.dB, 0, bytesB));

        // One extra stream to step past the capacity, and one D per stream so
        // concurrent launches are not racing each other's output.
        const int total = kStreamCapacity + 1;
        for(int i = 0; i < total; ++i)
        {
            void* d = nullptr;
            if(hipMalloc(&d, bytesD) != hipSuccess)
                GTEST_SKIP() << "Insufficient device memory for " << total << " outputs";
            r.dD.push_back(d);

            hipStream_t s = nullptr;
            if(hipStreamCreate(&s) != hipSuccess)
                GTEST_SKIP() << "Could not create " << total << " streams";
            r.streams.push_back(s);
        }

        const float alpha = 1.0f, beta = 0.0f;
        for(int i = 0; i < total; ++i)
        {
            const hipblasStatus_t status = hipblasLtMatmul(r.handle,
                                                           r.desc,
                                                           &alpha,
                                                           r.dA,
                                                           r.layA,
                                                           r.dB,
                                                           r.layB,
                                                           &beta,
                                                           r.dD[i],
                                                           r.layD,
                                                           r.dD[i],
                                                           r.layD,
                                                           &heuristic.algo,
                                                           r.dWs[0],
                                                           bytesWs,
                                                           r.streams[i]);
            if(i < kStreamCapacity)
                EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS)
                    << "stream " << i << " is within the " << kStreamCapacity
                    << "-block capacity and should have been served";
            else
                EXPECT_EQ(status, HIPBLAS_STATUS_INTERNAL_ERROR)
                    << "stream " << i << " is past the " << kStreamCapacity
                    << "-block capacity, so it must be refused rather than handed a block "
                       "another stream already owns";
        }

        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    // Four threads, each with its own handle and non-blocking stream, run
    // heuristic selection and matmul together. Shapes differ per thread and
    // rotate per iteration. A and B are all ones, so every element of D equals K.
    constexpr int kConcurrentThreads    = 4;
    constexpr int kConcurrentIterations = 40;
    constexpr int kConcurrentAlgos      = 4;

    // K stays in {256, 512, 768, 1024}: sums of ones up to there are exact in bf16.
    struct ConcurrentShape
    {
        int64_t m, n, k;
    };
    constexpr ConcurrentShape kConcurrentShapes[] = {
        {512, 512, 256}, {1024, 256, 768}, {256, 1024, 512}, {768, 768, 1024}};
    constexpr int     kConcurrentShapeCount = 4;
    constexpr int64_t kConcurrentMaxElems   = 1 << 20;

    constexpr uint16_t kBf16One = 0x3F80;

    uint16_t bf16FromExactInt(int64_t v)
    {
        const float f   = static_cast<float>(v);
        uint32_t    raw = 0;
        static_assert(sizeof(raw) == sizeof(f), "float is 32 bits");
        std::memcpy(&raw, &f, sizeof(raw));
        return static_cast<uint16_t>(raw >> 16);
    }

    struct ConcurrentWorker
    {
        int         id        = 0;
        bool        noSolution = false;
        int         completed = 0;
        std::string error; // first failure; gtest assertions stay on the main thread
    };

    void runConcurrentWorker(ConcurrentWorker&   w,
                             std::atomic<int>&   ready,
                             std::atomic<bool>&  go,
                             std::atomic<bool>&  abortAll,
                             const std::chrono::steady_clock::time_point deadline)
    {
        auto fail = [&](const std::string& what) {
            if(w.error.empty())
                w.error = what;
            abortAll = true;
        };
        auto hipOk = [&](hipError_t e, const char* what) {
            if(e == hipSuccess)
                return true;
            fail(std::string(what) + ": " + hipGetErrorString(e));
            return false;
        };
        auto blasOk = [&](hipblasStatus_t s, const char* what) {
            if(s == HIPBLAS_STATUS_SUCCESS)
                return true;
            fail(std::string(what) + ": hipblasStatus " + std::to_string(static_cast<int>(s)));
            return false;
        };

        // The gate is released even on a setup failure, or the other threads
        // would wait on this one forever.
        bool arrived = false;
        auto arrive  = [&] {
            if(!arrived)
            {
                arrived = true;
                ++ready;
            }
        };

        hipblasLtHandle_t           handle = nullptr;
        hipblasLtMatmulPreference_t pref   = nullptr;
        hipStream_t                 stream = nullptr;
        void *                      dA = nullptr, *dB = nullptr, *dD = nullptr, *dWs = nullptr;
        size_t                      wsBytes = 0;

        auto cleanup = [&] {
            if(stream)
                static_cast<void>(hipStreamSynchronize(stream));
            static_cast<void>(hipFree(dA));
            static_cast<void>(hipFree(dB));
            static_cast<void>(hipFree(dD));
            static_cast<void>(hipFree(dWs));
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            if(stream)
                static_cast<void>(hipStreamDestroy(stream));
            if(handle)
                hipblasLtDestroy(handle);
        };

        const size_t bytes = static_cast<size_t>(kConcurrentMaxElems) * sizeof(uint16_t);
        bool         setup = hipOk(hipSetDevice(0), "hipSetDevice")
                     && blasOk(hipblasLtCreate(&handle), "hipblasLtCreate")
                     && hipOk(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                              "hipStreamCreateWithFlags")
                     && blasOk(hipblasLtMatmulPreferenceCreate(&pref), "PreferenceCreate")
                     && hipOk(hipMalloc(&dA, bytes), "hipMalloc A")
                     && hipOk(hipMalloc(&dB, bytes), "hipMalloc B")
                     && hipOk(hipMalloc(&dD, bytes), "hipMalloc D");
        if(setup)
        {
            const uint64_t wsBudget = kWsBudgetBytes;
            hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBudget, sizeof(wsBudget));

            std::vector<uint16_t> ones(static_cast<size_t>(kConcurrentMaxElems), kBf16One);
            setup = hipOk(hipMemcpy(dA, ones.data(), bytes, hipMemcpyHostToDevice), "memcpy A")
                    && hipOk(hipMemcpy(dB, ones.data(), bytes, hipMemcpyHostToDevice), "memcpy B");
        }

        arrive();
        while(!go)
            std::this_thread::yield();
        if(!setup || abortAll)
        {
            cleanup();
            return;
        }

        const float alpha = 1.0f, beta = 0.0f;
        for(int iter = 0; iter < kConcurrentIterations && !abortAll; ++iter)
        {
            const ConcurrentShape sh = kConcurrentShapes[(w.id + iter) % kConcurrentShapeCount];

            hipblasLtMatmulDesc_t   desc = nullptr;
            hipblasLtMatrixLayout_t layA = nullptr, layB = nullptr, layD = nullptr;

            bool ok = blasOk(hipblasLtMatrixLayoutCreate(&layA, HIP_R_16BF, sh.m, sh.k, sh.m),
                             "LayoutCreate A")
                      && blasOk(hipblasLtMatrixLayoutCreate(&layB, HIP_R_16BF, sh.k, sh.n, sh.k),
                                "LayoutCreate B")
                      && blasOk(hipblasLtMatrixLayoutCreate(&layD, HIP_R_16BF, sh.m, sh.n, sh.m),
                                "LayoutCreate D")
                      && blasOk(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                                "DescCreate");

            hipblasLtMatmulHeuristicResult_t heuristic[kConcurrentAlgos]{};
            int                              returned = 0;
            if(ok)
            {
                const hipblasOperation_t opN = HIPBLAS_OP_N;
                hipblasLtMatmulDescSetAttribute(
                    desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
                hipblasLtMatmulDescSetAttribute(
                    desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
                ok = blasOk(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                            desc,
                                                            layA,
                                                            layB,
                                                            layD,
                                                            layD,
                                                            pref,
                                                            kConcurrentAlgos,
                                                            heuristic,
                                                            &returned),
                            "AlgoGetHeuristic");
            }
            if(ok && returned == 0)
            {
                w.noSolution = true;
                ok           = false;
            }

            if(ok && heuristic[0].workspaceSize > wsBytes)
            {
                static_cast<void>(hipFree(dWs));
                dWs     = nullptr;
                wsBytes = heuristic[0].workspaceSize;
                ok      = hipOk(hipMalloc(&dWs, wsBytes), "hipMalloc workspace");
            }

            if(ok)
                ok = hipOk(hipMemsetAsync(dD, 0, bytes, stream), "memset D")
                     && blasOk(hipblasLtMatmul(handle,
                                               desc,
                                               &alpha,
                                               dA,
                                               layA,
                                               dB,
                                               layB,
                                               &beta,
                                               dD,
                                               layD,
                                               dD,
                                               layD,
                                               &heuristic[0].algo,
                                               dWs,
                                               heuristic[0].workspaceSize,
                                               stream),
                               "hipblasLtMatmul");

            std::vector<uint16_t> out;
            if(ok)
            {
                out.assign(static_cast<size_t>(sh.m * sh.n), 0);
                ok = hipOk(hipMemcpyAsync(out.data(),
                                          dD,
                                          out.size() * sizeof(uint16_t),
                                          hipMemcpyDeviceToHost,
                                          stream),
                           "memcpy D");
            }
            if(ok)
            {
                // Poll against the deadline, as above: a wedged queue must fail
                // the test with a message rather than hang the job.
                hipError_t q = hipStreamQuery(stream);
                while(q == hipErrorNotReady)
                {
                    if(std::chrono::steady_clock::now() > deadline)
                    {
                        std::fprintf(stderr,
                                     "\n[  FAILED  ] StreamKMultiStream.smoke_ConcurrentHandlesMatmul\n"
                                     "  thread %d made no progress at iteration %d of %d.\n"
                                     "  Exiting without cleanup because the queue is wedged.\n\n",
                                     w.id,
                                     iter,
                                     kConcurrentIterations);
                        std::fflush(stderr);
                        std::_Exit(1);
                    }
                    std::this_thread::yield();
                    q = hipStreamQuery(stream);
                }
                ok = hipOk(q, "hipStreamQuery");
            }
            if(ok)
            {
                const uint16_t want = bf16FromExactInt(sh.k);
                for(size_t i = 0; i < out.size(); ++i)
                    if(out[i] != want)
                    {
                        char bits[64];
                        std::snprintf(bits, sizeof(bits), "got bf16 0x%04x, want 0x%04x", out[i], want);
                        fail("wrong result at iteration " + std::to_string(iter) + ", shape "
                             + std::to_string(sh.m) + "x" + std::to_string(sh.n) + "x"
                             + std::to_string(sh.k) + ", element " + std::to_string(i) + ": "
                             + bits);
                        ok = false;
                        break;
                    }
            }
            if(ok)
                ++w.completed;

            if(layA)
                hipblasLtMatrixLayoutDestroy(layA);
            if(layB)
                hipblasLtMatrixLayoutDestroy(layB);
            if(layD)
                hipblasLtMatrixLayoutDestroy(layD);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(!ok)
                break;
        }
        cleanup();
    }

    TEST(StreamKMultiStream, smoke_ConcurrentHandlesMatmul)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        std::atomic<int>  ready{0};
        std::atomic<bool> go{false};
        std::atomic<bool> abortAll{false};
        const auto        deadline
            = std::chrono::steady_clock::now() + std::chrono::seconds(kDeadlineSeconds);

        std::vector<ConcurrentWorker> workers(kConcurrentThreads);
        std::vector<std::thread>      threads;
        for(int i = 0; i < kConcurrentThreads; ++i)
        {
            workers[i].id = i;
            threads.emplace_back(runConcurrentWorker,
                                 std::ref(workers[i]),
                                 std::ref(ready),
                                 std::ref(go),
                                 std::ref(abortAll),
                                 deadline);
        }
        while(ready < kConcurrentThreads)
            std::this_thread::yield();
        go = true;
        for(auto& t : threads)
            t.join();

        bool anySkip = false;
        for(const auto& w : workers)
        {
            anySkip = anySkip || w.noSolution;
            EXPECT_TRUE(w.error.empty()) << "thread " << w.id << ": " << w.error;
        }
        if(anySkip)
            GTEST_SKIP() << "No solution for one of the test shapes on this device";
        for(const auto& w : workers)
            EXPECT_EQ(w.completed, kConcurrentIterations) << "thread " << w.id;
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }
} // namespace
