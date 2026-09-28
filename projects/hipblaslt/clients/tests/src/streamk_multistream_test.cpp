/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// Regression test for the multi-stream Stream-K deadlock (ROCM-29670).
//
// Stream-K's remainder path synchronises workgroups through a region of
// handle->Synchronizer. That region used to be shared by every stream on a
// handle, so two Stream-K kernels running concurrently could clear a flag the
// other was still spinning on and the waiting workgroup would spin forever.
// The fix moves those flags to handle->StreamKFlags and gives each stream its
// own block.
//
// The test fans the same GEMM out over many streams and joins them back, which
// is what raises the concurrency enough to hit the hazard. Against an
// unpatched library on gfx950 the queue stops draining on the first iteration.
//
// Two things are deliberate here:
//
//   * Progress is checked with hipStreamQuery against a deadline rather than
//     hipStreamSynchronize. Once the deadlock happens the queue never drains,
//     so a blocking wait would turn a failure into a CI timeout with no
//     message. Polling lets the test say what went wrong.
//
//   * On timeout the process exits immediately instead of unwinding. The GPU
//     queue is wedged at that point, so hipFree and hipStreamDestroy would
//     block too and the failure would never be printed.
//
// The test skips rather than fails when Stream-K is not what the heuristic
// picks for this device, since the hazard is unreachable then.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
    // Shape the customer workload deadlocked on, and the one the heuristic
    // picks a Stream-K solution for on gfx950.
    constexpr int64_t kM = 1024;
    constexpr int64_t kN = 7168;
    constexpr int64_t kK = 4096;

    // The hazard only exists on the Stream-K remainder path, reached when the
    // tile count does not divide evenly by the grid. This has to be forced:
    // left alone the heuristic sizes both grid and tiles at 256 for this
    // shape, and a test on the even path passes against a broken library.
    //
    // getSKGrid folds smCountTarget into its CU budget, so a 96-CU target
    // gives grid 96 against 180 tiles, leaving a remainder of 84. Checked on
    // gfx950 with TENSILE_DB=0xFFFF; 64 and 128 both come back evenly divided
    // and are not usable here. This is the supported API spelling of
    // TENSILE_STREAMK_MAX_CUS: those environment variables are cached in
    // function-local statics on first read, so a test binary cannot set them
    // reliably once anything else has touched the library.
    constexpr int32_t kSmCountTarget = 96;

    constexpr int kStreams    = 32;
    constexpr int kIterations = 500;

    // Only the ceiling offered to the heuristic. What actually gets allocated
    // per stream is the workspace the chosen solution reports needing.
    constexpr size_t kWsBudgetBytes = 128ull << 20;

    // A healthy run of this size takes a couple of seconds. The margin is for
    // a loaded CI machine, not for the kernel.
    constexpr int kDeadlineSeconds = 120;

    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    // Frees whatever was allocated. Only safe while the queue still drains.
    struct Resources
    {
        hipblasLtHandle_t           handle = nullptr;
        hipblasLtMatmulDesc_t       desc   = nullptr;
        hipblasLtMatrixLayout_t     layA = nullptr, layB = nullptr, layD = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
        void *                      dA = nullptr, *dB = nullptr;
        std::vector<void*>          dD, dWs;
        std::vector<hipStream_t>    streams;

        ~Resources()
        {
            for(auto s : streams)
                if(s)
                    static_cast<void>(hipStreamDestroy(s));
            for(auto p : dD)
                static_cast<void>(hipFree(p));
            for(auto p : dWs)
                static_cast<void>(hipFree(p));
            static_cast<void>(hipFree(dA));
            static_cast<void>(hipFree(dB));
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            if(layA)
                hipblasLtMatrixLayoutDestroy(layA);
            if(layB)
                hipblasLtMatrixLayoutDestroy(layB);
            if(layD)
                hipblasLtMatrixLayoutDestroy(layD);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(handle)
                hipblasLtDestroy(handle);
        }
    };

    TEST(StreamKMultiStream, ConcurrentStreamsDoNotDeadlock)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        Resources r;
        ASSERT_EQ(hipblasLtCreate(&r.handle), HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.layA, HIP_R_16BF, kM, kK, kM),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.layB, HIP_R_16BF, kK, kN, kK),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.layD, HIP_R_16BF, kM, kN, kM),
                  HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatmulDescCreate(&r.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        const hipblasOperation_t opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(r.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        hipblasLtMatmulDescSetAttribute(r.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(r.desc,
                                                  HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                                  &kSmCountTarget,
                                                  sizeof(kSmCountTarget)),
                  HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&r.pref), HIPBLAS_STATUS_SUCCESS);
        const uint64_t wsBudget = kWsBudgetBytes;
        hipblasLtMatmulPreferenceSetAttribute(
            r.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBudget, sizeof(wsBudget));

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
        // what the chosen solution actually asks for rather than the budget
        // above, which keeps 32 streams inside a gigabyte.
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
        hipStream_t main = nullptr;
        ASSERT_EQ(hipStreamCreate(&main), hipSuccess);
        r.streams.push_back(main);

        const auto deadline
            = std::chrono::steady_clock::now() + std::chrono::seconds(kDeadlineSeconds);

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

            // Poll rather than block: a deadlocked queue never drains, and a
            // blocking wait here would hang the job instead of failing it.
            while(hipStreamQuery(main) == hipErrorNotReady)
            {
                if(std::chrono::steady_clock::now() > deadline)
                {
                    std::fprintf(stderr,
                                 "\n[  FAILED  ] StreamKMultiStream.ConcurrentStreamsDoNotDeadlock\n"
                                 "  %d streams stopped making progress at iteration %d of %d.\n"
                                 "  The Stream-K flag region is being shared across streams;\n"
                                 "  see ROCM-29670. Exiting without cleanup because the queue\n"
                                 "  is wedged and hipFree would block as well.\n\n",
                                 kStreams,
                                 iter,
                                 kIterations);
                    std::fflush(stderr);
                    ADD_FAILURE() << kStreams << " concurrent streams deadlocked at iteration "
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
        ASSERT_EQ(hipblasLtCreate(&r.handle), HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.layA, HIP_R_16BF, m, k, m),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.layB, HIP_R_16BF, k, n, k),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.layD, HIP_R_16BF, m, n, m),
                  HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatmulDescCreate(&r.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        const hipblasOperation_t opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(r.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        hipblasLtMatmulDescSetAttribute(r.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&r.pref), HIPBLAS_STATUS_SUCCESS);
        const uint64_t wsBudget = kWsBudgetBytes;
        hipblasLtMatmulPreferenceSetAttribute(
            r.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBudget, sizeof(wsBudget));

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
} // namespace
