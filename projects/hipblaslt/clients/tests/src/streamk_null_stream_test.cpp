// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test for the legacy null stream losing its Stream-K flag block.
//
// streamKFlagsForStream() keys a block on the raw hipStream_t. For the legacy
// null stream that key is nullptr, which is also the value the claim loop uses
// to mean "block free", so the CAS succeeds without marking the block owned.
// The null stream therefore reserves nothing and later gets displaced by other
// streams -- which both aliases it onto another stream's flags (the deadlock
// this separation exists to prevent) and eventually locks it out entirely.
//
// The aliasing itself is not visible from outside the library, but the
// capacity accounting is, and it has the same root cause: run the null stream
// first, fill the table with other streams, then use the null stream again.
//
// Sequential and tiny by design -- no concurrency, no Stream-K shape needed,
// since a block is claimed for every matmul before the solution is known.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <vector>

namespace
{
    constexpr int64_t kM = 128, kN = 128, kK = 128;

    // _rocblaslt_handle::c_syncSkStreamSlots. Enough extra streams to fill the
    // table if the null stream's own claim did not reserve anything.
    constexpr int kCapacity = 64;

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

        hipblasLtMatmulPreference_t pref = nullptr;
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        const uint64_t wsBudget = 32ull << 20;
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBudget, sizeof(wsBudget));

        hipblasLtMatmulHeuristicResult_t heuristic{};
        int                              returned = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, layA, layB, layD, layD, pref, 1, &heuristic, &returned),
                  HIPBLAS_STATUS_SUCCESS);
        if(returned == 0)
            GTEST_SKIP() << "No solution for " << kM << "x" << kN << "x" << kK;

        void *dA = nullptr, *dB = nullptr, *dD = nullptr, *dWs = nullptr;
        ASSERT_EQ(hipMalloc(&dA, static_cast<size_t>(kM * kK) * sizeof(uint16_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, static_cast<size_t>(kK * kN) * sizeof(uint16_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD, static_cast<size_t>(kM * kN) * sizeof(uint16_t)), hipSuccess);
        if(heuristic.workspaceSize > 0)
            ASSERT_EQ(hipMalloc(&dWs, heuristic.workspaceSize), hipSuccess);

        const float alpha = 1.0f, beta = 0.0f;
        auto        matmul = [&](hipStream_t s) {
            const hipblasStatus_t st = hipblasLtMatmul(handle,
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
                                                       &heuristic.algo,
                                                       dWs,
                                                       heuristic.workspaceSize,
                                                       s);
            // One matmul in flight at a time: only the claim bookkeeping is
            // under test, so the shared D and workspace must not be raced on.
            static_cast<void>(hipStreamSynchronize(s));
            return st;
        };

        // The null stream goes first, so it holds a block from here on.
        ASSERT_EQ(matmul(nullptr), HIPBLAS_STATUS_SUCCESS)
            << "the null stream could not run at all";

        // Fill the table. The last of these is expected to be refused once the
        // null stream's block is correctly accounted for, so failures here are
        // not themselves the defect -- only their count is diagnostic.
        std::vector<hipStream_t> streams;
        int                      admitted = 0;
        for(int i = 0; i < kCapacity; ++i)
        {
            hipStream_t s = nullptr;
            if(hipStreamCreate(&s) != hipSuccess)
                break;
            streams.push_back(s);
            if(matmul(s) == HIPBLAS_STATUS_SUCCESS)
                ++admitted;
        }

        const hipblasStatus_t afterOthers = matmul(nullptr);
        const size_t          created     = streams.size();

        for(auto s : streams)
            static_cast<void>(hipStreamDestroy(s));
        static_cast<void>(hipFree(dA));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dD));
        static_cast<void>(hipFree(dWs));
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtMatrixLayoutDestroy(layA);
        hipblasLtMatrixLayoutDestroy(layB);
        hipblasLtMatrixLayoutDestroy(layD);
        hipblasLtDestroy(handle);

        // The count below only means anything if the table was really filled.
        ASSERT_EQ(created, static_cast<size_t>(kCapacity)) << "could not create enough streams";

        // A stream that has already been served must keep working.
        EXPECT_EQ(afterOthers, HIPBLAS_STATUS_SUCCESS)
            << "the null stream ran first but was displaced by later streams";

        // Corroborating signal: the null stream occupies one of the blocks, so
        // only kCapacity - 1 of the other streams can fit.
        EXPECT_LT(admitted, kCapacity) << "the null stream's claim reserved no capacity: all "
                                       << kCapacity << " later streams were admitted alongside it";
    }
}
