// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test for the object API binding its Stream-K flags to the wrong
// stream.
//
// hipblaslt_ext::GemmInstance::initialize() takes a stream that defaults to 0,
// and makeArgument() bakes that stream's flag region into the kernel
// arguments. The kernel actually runs on the stream given to run(), so the
// common initialize(algo, ws) + run(streamX) pattern would point every object
// at one region however many streams they run on -- exactly the aliasing the
// per-stream regions exist to prevent.
//
// The test builds that pattern and then makes the aliasing matter: every object
// runs concurrently on a stream of its own, so if they are all still on the
// initialize() stream's region they clear each other's flags and the queue
// stops draining. Nothing short of the run() stream's pointer reaching the
// kernel gets this to the end -- rebinding the inputs without re-solving leaves
// the old pointer baked into the arguments and deadlocks just the same.
//
// Watchdog and exit-without-unwinding as in streamk_multistream_test.cpp: a
// deadlocked queue never drains, so a blocking wait would hang the job.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include "streamk_test_util.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

namespace
{
    // The shape and CU target the heuristic answers with a Stream-K solution on
    // the remainder path; see streamk_multistream_test.cpp. Only that path reads
    // the flags, so only it can deadlock on them.
    constexpr int64_t kM = 1024;
    constexpr int64_t kN = 7168;
    constexpr int64_t kK = 4096;
    constexpr int32_t kSmCountTarget = 96;

    // Concurrent run() streams. Each needs its own D and workspace.
    constexpr int kExtStreams = 8;

    constexpr int kIterations = 200;

    constexpr size_t kWsBudgetBytes = 128ull << 20;

    constexpr int kDeadlineSeconds = 120;

    TEST(StreamKExtStream, RunStreamOwnsTheFlagRegion)
    {
        int devices = 0;
        if(hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
            GTEST_SKIP() << "No GPU available";

        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatrixLayout_t layA = nullptr, layB = nullptr, layD = nullptr;
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layA, HIP_R_16BF, kM, kK, kM),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layB, HIP_R_16BF, kK, kN, kK),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layD, HIP_R_16BF, kM, kN, kM),
                  HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulDesc_t desc = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        const hipblasOperation_t opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc,
                                                  HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                                  &kSmCountTarget,
                                                  sizeof(kSmCountTarget)),
                  HIPBLAS_STATUS_SUCCESS);

        const size_t bytesA = static_cast<size_t>(kM * kK) * sizeof(uint16_t);
        const size_t bytesB = static_cast<size_t>(kK * kN) * sizeof(uint16_t);
        const size_t bytesD = static_cast<size_t>(kM * kN) * sizeof(uint16_t);

        void *dA = nullptr, *dB = nullptr;
        ASSERT_EQ(hipMalloc(&dA, bytesA), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, bytesB), hipSuccess);
        static_cast<void>(hipMemset(dA, 0, bytesA));
        static_cast<void>(hipMemset(dB, 0, bytesB));

        std::vector<void*> dD(kExtStreams, nullptr);
        for(auto& d : dD)
            ASSERT_EQ(hipMalloc(&d, bytesD), hipSuccess);

        const float alpha = 1.0f, beta = 0.0f;

        auto makeGemm = [&](void* d) {
            auto gemm = std::make_unique<hipblaslt_ext::Gemm>(handle,
                                                              HIPBLAS_OP_N,
                                                              HIPBLAS_OP_N,
                                                              HIP_R_16BF,
                                                              HIP_R_16BF,
                                                              HIP_R_16BF,
                                                              HIP_R_16BF,
                                                              HIPBLAS_COMPUTE_32F);
            gemm->setMaxWorkspaceBytes(kWsBudgetBytes);
            if(gemm->setProblem(desc, &alpha, dA, layA, dB, layB, &beta, d, layD, d, layD)
               != HIPBLAS_STATUS_SUCCESS)
                gemm.reset();
            return gemm;
        };

        std::vector<hipblasLtMatmulHeuristicResult_t> algos;
        {
            auto probe = makeGemm(dD[0]);
            ASSERT_NE(probe, nullptr);
            hipblaslt_ext::GemmPreference pref;
            pref.setMaxWorkspaceBytes(kWsBudgetBytes);
            static_cast<void>(probe->algoGetHeuristic(1, pref, algos));
        }
        if(algos.empty())
            GTEST_SKIP() << "No solution for " << kM << "x" << kN << "x" << kK;

        const std::string picked = streamk_test::solutionName(handle, algos[0].algo);
        if(!streamk_test::isStreamKSolutionName(picked))
            GTEST_SKIP() << "The heuristic did not pick a Stream-K solution for this device: "
                         << picked;

        const size_t       bytesWs = algos[0].workspaceSize;
        std::vector<void*> dWs(kExtStreams, nullptr);
        if(bytesWs > 0)
            for(auto& w : dWs)
                ASSERT_EQ(hipMalloc(&w, bytesWs), hipSuccess);

        // The pattern under test: no stream at initialize(), a different one at
        // every run(). The streams must not implicitly synchronise with the
        // initialize() stream, which is the legacy null one, or the launches
        // would be serialised and there would be nothing to deadlock.
        std::vector<hipStream_t>                          extStreams;
        std::vector<std::unique_ptr<hipblaslt_ext::Gemm>> gemms;
        for(int i = 0; i < kExtStreams; ++i)
        {
            hipStream_t s = nullptr;
            ASSERT_EQ(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), hipSuccess);
            extStreams.push_back(s);

            auto gemm = makeGemm(dD[i]);
            ASSERT_NE(gemm, nullptr);
            ASSERT_EQ(gemm->initialize(algos[0].algo, dWs[i]), HIPBLAS_STATUS_SUCCESS);
            gemms.push_back(std::move(gemm));
        }

        const auto deadline
            = std::chrono::steady_clock::now() + std::chrono::seconds(kDeadlineSeconds);

        for(int iter = 0; iter < kIterations; ++iter)
        {
            for(int i = 0; i < kExtStreams; ++i)
                ASSERT_EQ(gemms[i]->run(extStreams[i]), HIPBLAS_STATUS_SUCCESS)
                    << "run() refused on stream " << i << " at iteration " << iter;

            bool draining = true;
            while(draining)
            {
                draining = false;
                for(auto s : extStreams)
                    if(hipStreamQuery(s) == hipErrorNotReady)
                        draining = true;
                if(draining && std::chrono::steady_clock::now() > deadline)
                {
                    std::fprintf(stderr,
                                 "\n[  FAILED  ] StreamKExtStream.RunStreamOwnsTheFlagRegion\n"
                                 "  %d objects initialized on one stream and run on %d others\n"
                                 "  stopped making progress at iteration %d of %d: they are\n"
                                 "  sharing the initialize() stream's Stream-K flag region.\n"
                                 "  Exiting without cleanup because the queue is wedged and\n"
                                 "  hipFree would block as well.\n\n",
                                 kExtStreams,
                                 kExtStreams,
                                 iter,
                                 kIterations);
                    std::fflush(stderr);
                    ADD_FAILURE() << kExtStreams
                                  << " objects run on distinct streams deadlocked at iteration "
                                  << iter;
                    std::_Exit(1);
                }
                if(draining)
                    // Yield between polls: on a real deadlock this loop runs
                    // for the whole deadline, and a bare spin would hold a core.
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(hipGetLastError(), hipSuccess);

        gemms.clear();
        for(auto s : extStreams)
            static_cast<void>(hipStreamDestroy(s));
        for(auto w : dWs)
            static_cast<void>(hipFree(w));
        for(auto d : dD)
            static_cast<void>(hipFree(d));
        static_cast<void>(hipFree(dA));
        static_cast<void>(hipFree(dB));
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtMatrixLayoutDestroy(layA);
        hipblasLtMatrixLayoutDestroy(layB);
        hipblasLtMatrixLayoutDestroy(layD);
        hipblasLtDestroy(handle);
    }
}
