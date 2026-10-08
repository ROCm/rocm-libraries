// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// End-to-end test of the user kernel registration C API. Checks that:
//
//   1. A registered kernel can be run by index immediately.
//   2. Registering does not change heuristic selection.
//   3. Mapping an exact shape does change it, even when the selection for that
//      shape is already cached.
//
// The heuristic is queried before registering so that (3) runs against a warm
// cache. The payload is a shipped bf16 TN shard.

#include <gtest/gtest.h>

#include "user_kernel_test_common.hpp"

namespace
{
    using namespace user_kernel_test;

    struct DeviceBuffer
    {
        void* ptr = nullptr;

        explicit DeviceBuffer(size_t bytes)
        {
            if(hipMalloc(&ptr, bytes) != hipSuccess)
                ptr = nullptr;
        }
        ~DeviceBuffer()
        {
            if(ptr)
                static_cast<void>(hipFree(ptr));
        }
        DeviceBuffer(DeviceBuffer const&)            = delete;
        DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    };

    float bf16ToFloat(uint16_t bits)
    {
        uint32_t widened = static_cast<uint32_t>(bits) << 16;
        float    value;
        std::memcpy(&value, &widened, sizeof(value));
        return value;
    }
} // namespace

// Run in a child process by UserKernelRegister.EndToEnd.
TEST(UserKernelRegister, DISABLED_Worker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);

    std::vector<uint16_t> hA(K * M, BF16_ONE), hB(K * N, BF16_ONE);
    DeviceBuffer          dA(hA.size() * 2), dB(hB.size() * 2);
    DeviceBuffer          dC(M * N * 2), dD(M * N * 2), dWs(MAX_WORKSPACE);
    ASSERT_NE(dA.ptr, nullptr);
    ASSERT_NE(dB.ptr, nullptr);
    ASSERT_NE(dC.ptr, nullptr);
    ASSERT_NE(dD.ptr, nullptr);
    ASSERT_NE(dWs.ptr, nullptr);
    ASSERT_EQ(hipMemcpy(dA.ptr, hA.data(), hA.size() * 2, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dB.ptr, hB.data(), hB.size() * 2, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(dC.ptr, 0, M * N * 2), hipSuccess);
    ASSERT_EQ(hipMemset(dD.ptr, 0, M * N * 2), hipSuccess);

    // ---- 1. warm the selection cache BEFORE registering ---------------------
    hipblasLtMatmulHeuristicResult_t shipped{};
    ASSERT_EQ(topResult(handle, p, shipped), HIPBLAS_STATUS_SUCCESS);
    const int shippedTop = algoIndex(shipped.algo);
    EXPECT_GE(shippedTop, 0) << "heuristic returns a shipped kernel before registration";

    int isUser = -1;
    ASSERT_EQ(hipblasLtMatmulAlgoIsUserKernel(&shipped.algo, &isUser), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(isUser, 0) << "a shipped kernel is not reported as a user kernel";

    size_t registered = 123, selectable = 123;
    ASSERT_EQ(hipblasLtUserKernelGetCounts(handle, &registered, &selectable),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(registered, 0u) << "nothing registered at start";
    ASSERT_EQ(selectable, 0u) << "nothing selectable at start";

    // ---- 2. register --------------------------------------------------------
    // Not every shard has a kernel that runs on this GPU; try until one does.
    int                              chosen   = -1;
    size_t                           chosenWs = 0;
    hipblasLtMatmulHeuristicResult_t chosenResult{};
    size_t                           totalRegistered = 0;

    for(auto const& stem : candidates)
    {
        const std::string dat = stem + ".dat", co = stem + ".co";

        std::vector<int> indices(4096);
        int              numIndices = 0;
        ASSERT_EQ(hipblasLtUserKernelRegister(handle,
                                              dat.c_str(),
                                              co.c_str(),
                                              indices.data(),
                                              static_cast<int>(indices.size()),
                                              &numIndices),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(numIndices, 0) << "the payload produced at least one kernel";
        indices.resize(std::min<size_t>(numIndices, indices.size()));
        totalRegistered += static_cast<size_t>(numIndices);

        for(int idx : indices)
            ASSERT_GE(idx, USER_INDEX_BASE)
                << "every assigned index is in the user range [3 * 2^29, 2^31)";

        ASSERT_EQ(hipblasLtUserKernelGetCounts(handle, &registered, &selectable),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(registered, totalRegistered) << "registered count matches";
        EXPECT_EQ(selectable, 0u) << "registering alone makes nothing selectable";

        // ---- 3. registering must not change selection -----------------------
        hipblasLtMatmulHeuristicResult_t afterRegister{};
        ASSERT_EQ(topResult(handle, p, afterRegister), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(algoIndex(afterRegister.algo), shippedTop)
            << "registration should not change selection";

        // ---- 4. a registered kernel is executable by index right away -------
        chosen = findUsableKernel(handle, p, indices, chosenResult, chosenWs);
        if(chosen >= 0)
        {
            std::printf("registered %d kernels from %s\n",
                        numIndices,
                        std::filesystem::path(stem).filename().string().c_str());
            break;
        }
    }
    ASSERT_GE(chosen, 0) << "a registered kernel resolves by index and supports the problem";

    ASSERT_EQ(hipblasLtMatmulAlgoIsUserKernel(&chosenResult.algo, &isUser), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(isUser, 1) << "the registered kernel is reported as a user kernel";
    EXPECT_EQ(algoIndex(chosenResult.algo), chosen) << "the algo round-trips its assigned index";

    {
        float alpha = 1.0f, beta = 0.0f;
        ASSERT_EQ(hipblasLtMatmul(handle,
                                  p.desc,
                                  &alpha,
                                  dA.ptr,
                                  p.la,
                                  dB.ptr,
                                  p.lb,
                                  &beta,
                                  dC.ptr,
                                  p.lc,
                                  dD.ptr,
                                  p.ld,
                                  &chosenResult.algo,
                                  dWs.ptr,
                                  chosenWs,
                                  nullptr),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<uint16_t> hD(M * N, 0);
        ASSERT_EQ(hipMemcpy(hD.data(), dD.ptr, hD.size() * 2, hipMemcpyDeviceToHost), hipSuccess);

        bool exact = std::all_of(hD.begin(), hD.end(), [](uint16_t bits) {
            return bf16ToFloat(bits) == static_cast<float>(K);
        });
        EXPECT_TRUE(exact) << "unexpected result from the registered kernel: D[0] = "
                           << bf16ToFloat(hD[0]) << " expected " << K;
    }

    // ---- 5. mapping the shape DOES change selection, on a warm cache --------
    ASSERT_EQ(hipblasLtUserKernelSetExactMatch(handle, chosen, M, N, 1, K), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtUserKernelGetCounts(handle, &registered, &selectable),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(selectable, 1u) << "one kernel is now selectable";

    hipblasLtMatmulHeuristicResult_t afterMap{};
    ASSERT_EQ(topResult(handle, p, afterMap), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(algoIndex(afterMap.algo), chosen)
        << "selection should return the mapped kernel despite the cached result";

    // ---- 6. an unmapped shape is untouched ----------------------------------
    Problem neighbour;
    ASSERT_EQ(buildProblem(neighbour, 32), HIPBLAS_STATUS_SUCCESS);
    hipblasLtMatmulHeuristicResult_t other{};
    ASSERT_EQ(topResult(handle, neighbour, other), HIPBLAS_STATUS_SUCCESS)
        << "a neighbouring shape still resolves";
    ASSERT_EQ(hipblasLtMatmulAlgoIsUserKernel(&other.algo, &isUser), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(isUser, 0) << "an unmapped shape (M=32) should still get a shipped kernel";
}

TEST(UserKernelRegister, EndToEnd)
{
    if(candidatePayloads().empty())
        GTEST_SKIP() << "no payload shard in this build's library";

    EXPECT_EQ(runWorker("UserKernelRegister.DISABLED_Worker"), 0)
        << "the registration worker failed; its output is above";
}
