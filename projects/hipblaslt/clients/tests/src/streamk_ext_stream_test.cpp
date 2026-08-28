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
// Which region an object holds is not visible from outside the library, but
// how many regions the handle has handed out is: hipblasLtMatmul() claims one
// per stream unconditionally, so it can be used to count what is left. Objects
// run on N distinct streams have to consume at least N regions.
//
// Sequential by design -- no concurrency, since only the accounting is under
// test.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <memory>
#include <vector>

namespace
{
    // The shape the heuristic picks a Stream-K solution for on gfx950; see
    // streamk_multistream_test.cpp. Only Stream-K solutions bind a region
    // through the object API, so the test skips when this is not one.
    constexpr int64_t kM = 1024;
    constexpr int64_t kN = 7168;
    constexpr int64_t kK = 4096;

    // Distinct run() streams. Small: every one of them costs a region, and the
    // probe below needs room to tell how many were taken.
    constexpr int kExtStreams = 4;

    // _rocblaslt_handle::c_syncSkStreamSlots.
    constexpr int kCapacity = 64;

    constexpr size_t kWsBudgetBytes = 128ull << 20;

    // Probes with a shape small enough to be free, laid over the buffers the
    // Stream-K problem already allocated.
    constexpr int64_t kProbeMNK = 128;

    TEST(StreamKExtStream, RunStreamOwnsTheFlagRegion)
    {
        int devices = 0;
        if(hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
            GTEST_SKIP() << "No GPU available";

        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        void *dA = nullptr, *dB = nullptr, *dD = nullptr, *dWs = nullptr;
        ASSERT_EQ(hipMalloc(&dA, static_cast<size_t>(kM * kK) * sizeof(uint16_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, static_cast<size_t>(kK * kN) * sizeof(uint16_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD, static_cast<size_t>(kM * kN) * sizeof(uint16_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dWs, kWsBudgetBytes), hipSuccess);

        const float alpha = 1.0f, beta = 0.0f;

        auto makeGemm = [&]() {
            hipblaslt_ext::GemmEpilogue epilogue;
            hipblaslt_ext::GemmInputs   inputs;

            auto gemm = std::make_unique<hipblaslt_ext::Gemm>(handle,
                                                              HIPBLAS_OP_N,
                                                              HIPBLAS_OP_N,
                                                              HIP_R_16BF,
                                                              HIP_R_16BF,
                                                              HIP_R_16BF,
                                                              HIP_R_16BF,
                                                              HIPBLAS_COMPUTE_32F);
            inputs.setA(dA);
            inputs.setB(dB);
            inputs.setC(dD);
            inputs.setD(dD);
            inputs.setAlpha(&alpha);
            inputs.setBeta(&beta);
            gemm->setMaxWorkspaceBytes(kWsBudgetBytes);
            if(gemm->setProblem(kM, kN, kK, 1, epilogue, inputs) != HIPBLAS_STATUS_SUCCESS)
                gemm.reset();
            return gemm;
        };

        std::vector<hipblasLtMatmulHeuristicResult_t> algos;
        {
            auto probe = makeGemm();
            ASSERT_NE(probe, nullptr);
            hipblaslt_ext::GemmPreference pref;
            pref.setMaxWorkspaceBytes(kWsBudgetBytes);
            static_cast<void>(probe->algoGetHeuristic(1, pref, algos));
        }
        if(algos.empty())
            GTEST_SKIP() << "No solution for " << kM << "x" << kN << "x" << kK;

        // The pattern under test: no stream at initialize(), a different one at
        // every run().
        std::vector<hipStream_t>                          extStreams;
        std::vector<std::unique_ptr<hipblaslt_ext::Gemm>> gemms;
        for(int i = 0; i < kExtStreams; ++i)
        {
            hipStream_t s = nullptr;
            ASSERT_EQ(hipStreamCreate(&s), hipSuccess);
            extStreams.push_back(s);

            auto gemm = makeGemm();
            ASSERT_NE(gemm, nullptr);
            ASSERT_EQ(gemm->initialize(algos[0].algo, dWs), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(gemm->run(s), HIPBLAS_STATUS_SUCCESS) << "run() refused on stream " << i;
            // One kernel in flight at a time: the shared D and workspace must
            // not be raced on, and only the accounting is under test.
            ASSERT_EQ(hipStreamSynchronize(s), hipSuccess);
            gemms.push_back(std::move(gemm));
        }

        // Count what is left. hipblasLtMatmul() claims a region for every
        // stream it sees, whatever solution it ends up running.
        hipblasLtMatrixLayout_t layA = nullptr, layB = nullptr, layD = nullptr;
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layA, HIP_R_16BF, kProbeMNK, kProbeMNK, kProbeMNK),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layB, HIP_R_16BF, kProbeMNK, kProbeMNK, kProbeMNK),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layD, HIP_R_16BF, kProbeMNK, kProbeMNK, kProbeMNK),
                  HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulDesc_t desc = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        const hipblasOperation_t opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        hipblasLtMatmulPreference_t pref = nullptr;
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        const uint64_t wsBudget = kWsBudgetBytes;
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBudget, sizeof(wsBudget));

        hipblasLtMatmulHeuristicResult_t probeHeuristic{};
        int                              returned = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, layA, layB, layD, layD, pref, 1, &probeHeuristic, &returned),
                  HIPBLAS_STATUS_SUCCESS);

        std::vector<hipStream_t> probeStreams;
        int                      admitted = 0;
        if(returned > 0)
        {
            for(int i = 0; i < kCapacity; ++i)
            {
                hipStream_t s = nullptr;
                if(hipStreamCreate(&s) != hipSuccess)
                    break;
                probeStreams.push_back(s);
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
                                                           &probeHeuristic.algo,
                                                           dWs,
                                                           probeHeuristic.workspaceSize,
                                                           s);
                static_cast<void>(hipStreamSynchronize(s));
                if(st != HIPBLAS_STATUS_SUCCESS)
                    break;
                ++admitted;
            }
        }
        const size_t probesCreated = probeStreams.size();

        gemms.clear();
        for(auto s : probeStreams)
            static_cast<void>(hipStreamDestroy(s));
        for(auto s : extStreams)
            static_cast<void>(hipStreamDestroy(s));
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtMatrixLayoutDestroy(layA);
        hipblasLtMatrixLayoutDestroy(layB);
        hipblasLtMatrixLayoutDestroy(layD);
        static_cast<void>(hipFree(dA));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dD));
        static_cast<void>(hipFree(dWs));
        hipblasLtDestroy(handle);

        ASSERT_GT(returned, 0) << "no solution for the " << kProbeMNK << "-cubed probe";
        ASSERT_GT(probesCreated, 0u) << "could not create any probe stream";

        // Regions the object API took: the initialize() stream plus one per
        // run() stream.
        const int consumed = kCapacity - admitted;
        if(consumed == 0)
            GTEST_SKIP() << "the heuristic did not pick a Stream-K solution for " << kM << "x" << kN
                         << "x" << kK << ", so the object API bound no flag region";

        EXPECT_GE(consumed, kExtStreams)
            << "the object API took " << consumed << " flag regions while running on "
            << kExtStreams << " distinct streams: it bound the initialize() stream, not the run()"
            << " one";
    }
}
