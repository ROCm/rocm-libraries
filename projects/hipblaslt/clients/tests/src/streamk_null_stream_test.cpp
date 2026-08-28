// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test for the legacy null stream losing its Stream-K flag block.
//
// streamKFlagsForStream() keys a block on the raw hipStream_t. For the legacy
// null stream that key is nullptr, which is also the value the claim loop uses
// to mean "block free", so the CAS succeeds without marking the block owned.
// The null stream then holds nothing: it is handed the lowest free block, and
// the next stream to claim one is handed that same block and marks it. The two
// are now on one flag region, which is the cross-stream deadlock the per-stream
// blocks exist to prevent.
//
// So the test drives exactly that pairing: a Stream-K matmul on the null stream
// and one on a stream that has never claimed before, in flight together, once
// per iteration with a fresh side stream each time. Correctly keyed, the null
// stream owns a block from its first matmul and the side streams take the
// others, so no two launches ever meet on one region.
//
// The watchdog and the exit-without-unwinding are the same as in
// streamk_multistream_test.cpp and for the same reasons: a deadlocked queue
// never drains, so a blocking wait would hang the job instead of failing it.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
    // Shape and CU target the heuristic answers with a Stream-K solution on the
    // remainder path; see streamk_multistream_test.cpp for how they were picked.
    // Only that path reads the flags, so only it can deadlock on them.
    constexpr int64_t kM = 1024;
    constexpr int64_t kN = 7168;
    constexpr int64_t kK = 4096;
    constexpr int32_t kSmCountTarget = 96;

    // One fresh side stream per iteration, so every iteration is the pairing
    // described above rather than only the first. Beyond the 64-block capacity
    // the side streams fall back to the shared region, which is safe here
    // because only one of them is ever in flight.
    constexpr int kIterations = 128;

    constexpr size_t kWsBudgetBytes = 128ull << 20;

    constexpr int kDeadlineSeconds = 120;

    TEST(StreamKNullStream, NullStreamKeepsItsFlagBlock)
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

        hipblasLtMatmulPreference_t pref = nullptr;
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        const uint64_t wsBudget = kWsBudgetBytes;
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBudget, sizeof(wsBudget));

        hipblasLtMatmulHeuristicResult_t heuristic{};
        int                              returned = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, layA, layB, layD, layD, pref, 1, &heuristic, &returned),
                  HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            GTEST_SKIP() << "No solution for " << kM << "x" << kN << "x" << kK;

        const size_t bytesA  = static_cast<size_t>(kM * kK) * sizeof(uint16_t);
        const size_t bytesB  = static_cast<size_t>(kK * kN) * sizeof(uint16_t);
        const size_t bytesD  = static_cast<size_t>(kM * kN) * sizeof(uint16_t);
        const size_t bytesWs = heuristic.workspaceSize;

        // A and B are read-only and shared; the two concurrent launches need
        // their own D and workspace.
        void *dA = nullptr, *dB = nullptr;
        void *dDNull = nullptr, *dDSide = nullptr;
        void *dWsNull = nullptr, *dWsSide = nullptr;
        ASSERT_EQ(hipMalloc(&dA, bytesA), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, bytesB), hipSuccess);
        ASSERT_EQ(hipMalloc(&dDNull, bytesD), hipSuccess);
        ASSERT_EQ(hipMalloc(&dDSide, bytesD), hipSuccess);
        if(bytesWs > 0)
        {
            ASSERT_EQ(hipMalloc(&dWsNull, bytesWs), hipSuccess);
            ASSERT_EQ(hipMalloc(&dWsSide, bytesWs), hipSuccess);
        }
        static_cast<void>(hipMemset(dA, 0, bytesA));
        static_cast<void>(hipMemset(dB, 0, bytesB));

        const float alpha = 1.0f, beta = 0.0f;
        auto        matmul = [&](hipStream_t s, void* d, void* ws) {
            return hipblasLtMatmul(handle,
                                   desc,
                                   &alpha,
                                   dA,
                                   layA,
                                   dB,
                                   layB,
                                   &beta,
                                   d,
                                   layD,
                                   d,
                                   layD,
                                   &heuristic.algo,
                                   ws,
                                   bytesWs,
                                   s);
        };

        // Side streams must not implicitly synchronise with the legacy null
        // stream, or the two launches would never be in flight together and
        // there would be nothing to deadlock.
        std::vector<hipStream_t> sideStreams;
        sideStreams.reserve(kIterations);

        const auto deadline
            = std::chrono::steady_clock::now() + std::chrono::seconds(kDeadlineSeconds);

        for(int iter = 0; iter < kIterations; ++iter)
        {
            hipStream_t side = nullptr;
            ASSERT_EQ(hipStreamCreateWithFlags(&side, hipStreamNonBlocking), hipSuccess);
            sideStreams.push_back(side);

            // The null stream asks first, so under the defective key it is the
            // one holding the block `side` is about to be given.
            ASSERT_EQ(matmul(nullptr, dDNull, dWsNull), HIPBLAS_STATUS_SUCCESS)
                << "the null stream was refused at iteration " << iter;
            ASSERT_EQ(matmul(side, dDSide, dWsSide), HIPBLAS_STATUS_SUCCESS)
                << "the side stream was refused at iteration " << iter;

            while(hipStreamQuery(nullptr) == hipErrorNotReady
                  || hipStreamQuery(side) == hipErrorNotReady)
            {
                if(std::chrono::steady_clock::now() > deadline)
                {
                    std::fprintf(stderr,
                                 "\n[  FAILED  ] StreamKNullStream.NullStreamKeepsItsFlagBlock\n"
                                 "  The null stream and a fresh side stream stopped making\n"
                                 "  progress at iteration %d of %d: they are sharing one\n"
                                 "  Stream-K flag region, so the null stream's claim reserved\n"
                                 "  nothing. Exiting without cleanup because the queue is\n"
                                 "  wedged and hipFree would block as well.\n\n",
                                 iter,
                                 kIterations);
                    std::fflush(stderr);
                    ADD_FAILURE() << "the null stream aliased a side stream's flag region at "
                                     "iteration "
                                  << iter;
                    std::_Exit(1);
                }
            }
        }

        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

        for(auto s : sideStreams)
            static_cast<void>(hipStreamDestroy(s));
        static_cast<void>(hipFree(dA));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dDNull));
        static_cast<void>(hipFree(dDSide));
        static_cast<void>(hipFree(dWsNull));
        static_cast<void>(hipFree(dWsSide));
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtMatrixLayoutDestroy(layA);
        hipblasLtMatrixLayoutDestroy(layB);
        hipblasLtMatrixLayoutDestroy(layD);
        hipblasLtDestroy(handle);
    }
}
