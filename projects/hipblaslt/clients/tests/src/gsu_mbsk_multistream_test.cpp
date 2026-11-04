/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// Regression test for the multi-stream MBSK synchronizer race (ROCM-8845).
//
// MultipleBufferSingleKernel (the solution name contains GSUAMBSK) keeps its
// flags in the handle synchronizer. GSU <= 2 makes the last GSU workgroup
// spin until the counter is 1, then clear it. A second kernel that uses the
// same counter atomic-decrements it from 1 back to 0, and the waiter never
// sees 1. GSU > 2 only uses atomic_dec to pick a reducer, so it does not hang.
//
// Each stream now owns its own flag block. This test runs one handle from two
// threads and two streams, locking only around hipblasLtMatmul, and passes
// when 500 overlapping pairs drain. Sharing the flag again wedges the queue.
//
// gfx942 only. Only a solution whose name contains _GSU2_GSUAMBSK_ is
// selected, so the busy-wait path is the one that runs. The f16-in / f32-out
// kernel (Cijk_Ailk_Bljk_HSS_..._GSU2_GSUAMBSK_) is the one that used to hang.
// If none of the candidate shapes supports such a solution, the test skips.
//
// Progress is polled. On a stall the process exits immediately, because the
// queue is wedged and hipFree would block too.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
    // Tried in order. The first shape that a fixed-GSU=2 MBSK solution
    // accepts is the one that runs. Several are listed because equality
    // solutions only match their tuned sizes.
    struct Shape
    {
        int64_t m, n, k;
    };
    // 256x64x8192 is one the f16->f32 GSU=2 kernel accepts, and it is small
    // enough that two launches stay resident together.
    constexpr Shape kShapes[] = {
        {256, 64, 8192},
        {256, 128, 8192},
        {128, 128, 4096},
        {512, 64, 4096},
        {64, 64, 4096},
    };

    constexpr int kThreads      = 2;
    constexpr int kIterations   = 500;
    constexpr int kStallSeconds = 60;
    // Busy-wait path. Must match tensilelite/Components/GSU.py gsuThreshold.
    constexpr int kRacingGsu = 2;

    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    struct Gate
    {
        std::mutex              mutex;
        std::condition_variable cv;
        int                     arrived    = 0;
        int                     generation = 0;

        void arrive(int parties)
        {
            std::unique_lock<std::mutex> lock(mutex);
            const int                    generationAtEntry = generation;
            if(++arrived == parties)
            {
                arrived = 0;
                ++generation;
                cv.notify_all();
            }
            else
            {
                cv.wait(lock, [&] { return generation != generationAtEntry; });
            }
        }
    };

    struct Resources
    {
        hipblasLtHandle_t       handle = nullptr;
        hipblasLtMatmulDesc_t   desc   = nullptr;
        hipblasLtMatrixLayout_t layA = nullptr, layB = nullptr, layD = nullptr;
        void *                  dA = nullptr, *dB = nullptr;
        std::vector<void*>      dD, dWs;
        std::vector<hipStream_t> streams;

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

    // -1 means GlobalSplitU is computed at launch (the GSUn1 token). A positive
    // value is a fixed split. 0 means the token was not understood.
    int parseFixedGsu(const std::string& name)
    {
        const auto marker = name.find("_GSUAMBSK_");
        if(marker == std::string::npos)
            return 0;
        if(marker == 0)
            return -1;
        const auto token = name.rfind("_GSU", marker - 1);
        if(token == std::string::npos)
            return -1;
        if(name.compare(token, 5, "_GSUn") == 0)
            return -1;
        int gsu = 0;
        if(std::sscanf(name.c_str() + token, "_GSU%d", &gsu) != 1 || gsu <= 0)
            return 0;
        return gsu;
    }

    bool drain(const std::vector<hipStream_t>& streams, int iter)
    {
        const auto deadline
            = std::chrono::steady_clock::now() + std::chrono::seconds(kStallSeconds);
        for(;;)
        {
            bool pending = false;
            for(int i = 0; i < kThreads; ++i)
            {
                const hipError_t query = hipStreamQuery(streams[i]);
                if(query == hipErrorNotReady)
                    pending = true;
                else if(query != hipSuccess)
                    return true;
            }
            if(!pending)
                return true;
            if(std::chrono::steady_clock::now() > deadline)
            {
                std::fprintf(stderr,
                             "\n[  FAILED  ] GsuMbskMultiStream.OneHandleTwoStreamsDoNotHang\n"
                             "  %d threads stopped making progress after queueing %d iterations.\n"
                             "  GSUAMBSK kernels on one handle are sharing Synchronizer;\n"
                             "  see ROCM-8845. Exiting without cleanup because the queue\n"
                             "  is wedged and hipFree would block as well.\n\n",
                             kThreads,
                             iter);
                std::fflush(stderr);
                ADD_FAILURE() << kThreads << " overlapping streams wedged after " << iter
                              << " iterations";
                std::_Exit(1);
            }
        }
    }

    TEST(GsuMbskMultiStream, OneHandleTwoStreamsDoNotHang)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        hipDeviceProp_t prop{};
        ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
        if(std::string(prop.gcnArchName).find("gfx942") == std::string::npos)
            GTEST_SKIP() << "GSUAMBSK synchronizer repro is scoped to gfx942, this device is "
                         << prop.gcnArchName;

        const int cus = prop.multiProcessorCount;

        Resources r;
        ASSERT_EQ(hipblasLtCreate(&r.handle), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescCreate(&r.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        const hipblasOperation_t opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(r.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        hipblasLtMatmulDescSetAttribute(r.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        std::vector<hipblasLtMatmulHeuristicResult_t> algos;
        const hipblasStatus_t                         algoStatus
            = hipblaslt_ext::getAllAlgos(r.handle,
                                         hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                         HIPBLAS_OP_N,
                                         HIPBLAS_OP_N,
                                         HIP_R_16F,
                                         HIP_R_16F,
                                         HIP_R_32F,
                                         HIP_R_32F,
                                         HIPBLAS_COMPUTE_32F,
                                         algos);
        if(algoStatus != HIPBLAS_STATUS_SUCCESS || algos.empty())
            GTEST_SKIP() << "No f16-in f32-out NN algorithms loaded for this device";

        std::vector<hipblasLtMatmulHeuristicResult_t*> gsu2;
        gsu2.reserve(64);
        for(auto& candidate : algos)
        {
            const std::string name
                = hipblaslt_ext::getSolutionNameFromAlgo(r.handle, candidate.algo);
            if(name.find("_GSU2_GSUAMBSK_") == std::string::npos)
                continue;
            if(name.find("ISA942") == std::string::npos)
                continue;
            if(parseFixedGsu(name) != kRacingGsu)
                continue;
            gsu2.push_back(&candidate);
        }
        if(gsu2.empty())
            GTEST_SKIP() << "No gfx942 solution named _GSU2_GSUAMBSK_ is loaded";

        const float                      alpha = 1.0f, beta = 0.0f;
        hipblasLtMatmulHeuristicResult_t chosen{};
        std::string                      chosenName;
        size_t                           chosenWs = 0;
        Shape                            chosenShape{};
        bool                             found = false;

        for(const Shape& shape : kShapes)
        {
            if(r.layA)
                hipblasLtMatrixLayoutDestroy(r.layA);
            if(r.layB)
                hipblasLtMatrixLayoutDestroy(r.layB);
            if(r.layD)
                hipblasLtMatrixLayoutDestroy(r.layD);
            r.layA = r.layB = r.layD = nullptr;
            if(hipblasLtMatrixLayoutCreate(&r.layA, HIP_R_16F, shape.m, shape.k, shape.m)
                   != HIPBLAS_STATUS_SUCCESS
               || hipblasLtMatrixLayoutCreate(&r.layB, HIP_R_16F, shape.k, shape.n, shape.k)
                      != HIPBLAS_STATUS_SUCCESS
               || hipblasLtMatrixLayoutCreate(&r.layD, HIP_R_32F, shape.m, shape.n, shape.m)
                      != HIPBLAS_STATUS_SUCCESS)
                continue;

            for(auto* candidate : gsu2)
            {
                size_t workspace = 0;
                if(hipblaslt_ext::matmulIsAlgoSupported(r.handle,
                                                        r.desc,
                                                        &alpha,
                                                        r.layA,
                                                        r.layB,
                                                        &beta,
                                                        r.layD,
                                                        r.layD,
                                                        candidate->algo,
                                                        workspace)
                   != HIPBLAS_STATUS_SUCCESS)
                    continue;

                chosen      = *candidate;
                chosenName  = hipblaslt_ext::getSolutionNameFromAlgo(r.handle, chosen.algo);
                chosenWs    = workspace;
                chosenShape = shape;
                found       = true;
                break;
            }
            if(found)
                break;
        }

        if(!found)
            GTEST_SKIP() << "No loaded _GSU2_GSUAMBSK_ solution supports the candidate shapes";

        std::fprintf(stderr,
                     "[GsuMbskMultiStream] gfx942 CUs=%d shape=%ldx%ldx%ld GSU=%d "
                     "workspace=%zu index=%d solution=%s\n",
                     cus,
                     static_cast<long>(chosenShape.m),
                     static_cast<long>(chosenShape.n),
                     static_cast<long>(chosenShape.k),
                     kRacingGsu,
                     chosenWs,
                     hipblaslt_ext::getIndexFromAlgo(chosen.algo),
                     chosenName.c_str());

        const size_t bytesA = static_cast<size_t>(chosenShape.m * chosenShape.k) * sizeof(uint16_t);
        const size_t bytesB = static_cast<size_t>(chosenShape.k * chosenShape.n) * sizeof(uint16_t);
        const size_t bytesD = static_cast<size_t>(chosenShape.m * chosenShape.n) * sizeof(float);
        if(hipMalloc(&r.dA, bytesA) != hipSuccess || hipMalloc(&r.dB, bytesB) != hipSuccess)
            GTEST_SKIP() << "Insufficient device memory for the shared operands";
        static_cast<void>(hipMemset(r.dA, 0, bytesA));
        static_cast<void>(hipMemset(r.dB, 0, bytesB));

        for(int i = 0; i < kThreads; ++i)
        {
            void* d  = nullptr;
            void* ws = nullptr;
            if(hipMalloc(&d, bytesD) != hipSuccess
               || (chosenWs > 0 && hipMalloc(&ws, chosenWs) != hipSuccess))
            {
                static_cast<void>(hipFree(d));
                static_cast<void>(hipFree(ws));
                GTEST_SKIP() << "Insufficient device memory for " << kThreads << " streams";
            }
            r.dD.push_back(d);
            r.dWs.push_back(ws);

            hipStream_t stream = nullptr;
            if(hipStreamCreate(&stream) != hipSuccess)
                GTEST_SKIP() << "Could not create " << kThreads << " streams";
            r.streams.push_back(stream);
        }

        std::mutex            apiLock;
        std::atomic<bool>     launchFailed{false};
        hipblasLtMatmulAlgo_t algo = chosen.algo;

        // No per-iteration barrier. A barrier lets the GPU finish the pair
        // before the next one is queued, and a single pair of this kernel
        // still completes its handshake. The race shows up when both streams
        // keep a queue of in-flight GSU=2 kernels.
        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for(int idx = 0; idx < kThreads; ++idx)
        {
            threads.emplace_back([&, idx] {
                for(int iter = 0; iter < kIterations; ++iter)
                {
                    std::lock_guard<std::mutex> lock(apiLock);
                    const hipblasStatus_t       status = hipblasLtMatmul(r.handle,
                                                                   r.desc,
                                                                   &alpha,
                                                                   r.dA,
                                                                   r.layA,
                                                                   r.dB,
                                                                   r.layB,
                                                                   &beta,
                                                                   r.dD[idx],
                                                                   r.layD,
                                                                   r.dD[idx],
                                                                   r.layD,
                                                                   &algo,
                                                                   r.dWs[idx],
                                                                   chosenWs,
                                                                   r.streams[idx]);
                    if(status != HIPBLAS_STATUS_SUCCESS)
                        launchFailed.store(true);
                }
            });
        }
        for(auto& thread : threads)
            thread.join();
        if(!launchFailed.load())
            drain(r.streams, kIterations);

        EXPECT_FALSE(launchFailed.load())
            << "hipblasLtMatmul failed while launching the GSUAMBSK solution";
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }
} // namespace
