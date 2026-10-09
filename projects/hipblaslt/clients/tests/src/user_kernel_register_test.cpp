// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// End-to-end tests of the user kernel registration C API, using shipped bf16 TN
// shards as payloads. A build with the option ON always produces such shards,
// so a missing payload is a failure, not a skip.

#include <gtest/gtest.h>

#include "user_kernel_test_common.hpp"

namespace
{
    using namespace user_kernel_test;

    // Two devices of one architecture, so a payload for one serves both.
    bool haveTwoMatchingDevices()
    {
        int count = 0;
        if(hipGetDeviceCount(&count) != hipSuccess || count < 2)
            return false;
        hipDeviceProp_t a, b;
        if(hipGetDeviceProperties(&a, 0) != hipSuccess
           || hipGetDeviceProperties(&b, 1) != hipSuccess)
            return false;
        return std::strcmp(a.gcnArchName, b.gcnArchName) == 0;
    }

    void expectNothingRegistered(hipblasLtHandle_t handle)
    {
        size_t registered = 123, selectable = 123;
        ASSERT_EQ(hipblasLtUserKernelGetCounts(handle, &registered, &selectable),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(registered, 0u) << "nothing registered at start";
        ASSERT_EQ(selectable, 0u) << "nothing selectable at start";
    }
} // namespace

// Checks that:
//
//   1. A registered kernel can be run by index immediately.
//   2. Registering does not change heuristic selection.
//   3. Mapping an exact shape puts the kernel first, ahead of the shipped
//      library's choices in their original order, even though that shape was
//      already queried.
//   4. Other shapes are unaffected.
TEST(UserKernelRegister, DISABLED_Worker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    expectNothingRegistered(handle);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);

    const std::vector<int> shipped = topIndices(handle, p, 8);
    ASSERT_FALSE(shipped.empty());
    for(int idx : shipped)
        ASSERT_LT(idx, USER_INDEX_BASE) << "only shipped kernels before registration";

    int  isUser       = -1;
    auto shippedFirst = topResults(handle, p, 1);
    ASSERT_EQ(shippedFirst.size(), 1u);
    ASSERT_EQ(hipblasLtMatmulAlgoIsUserKernel(&shippedFirst[0].algo, &isUser),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(isUser, 0) << "a shipped kernel is not reported as a user kernel";

    Registered r = registerUsable(handle, p, candidates);
    ASSERT_GE(r.index, 0) << "a registered kernel resolves by index and supports the problem";
    std::printf("using a kernel from %s\n",
                std::filesystem::path(r.stem).filename().string().c_str());

    size_t registered = 0, selectable = 0;
    ASSERT_EQ(hipblasLtUserKernelGetCounts(handle, &registered, &selectable),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_GT(registered, 0u);
    EXPECT_EQ(selectable, 0u) << "registering alone makes nothing selectable";
    EXPECT_GE(r.index, USER_INDEX_BASE);

    EXPECT_EQ(topIndices(handle, p, 8), shipped) << "registration should not change selection";

    ASSERT_EQ(hipblasLtMatmulAlgoIsUserKernel(&r.result.algo, &isUser), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(isUser, 1) << "the registered kernel is reported as a user kernel";
    EXPECT_EQ(algoIndex(r.result.algo), r.index) << "the algo round-trips its assigned index";
    runAndCheck(handle, p, r.result.algo, r.workspace);

    ASSERT_EQ(hipblasLtUserKernelSetExactMatch(handle, r.index, M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtUserKernelGetCounts(handle, &registered, &selectable),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(selectable, 1u);

    const std::vector<int> mapped = topIndices(handle, p, 8);
    ASSERT_GE(mapped.size(), std::min<size_t>(2, shipped.size() + 1));
    EXPECT_EQ(mapped[0], r.index) << "the mapped kernel should come first";
    EXPECT_EQ(std::count(mapped.begin(), mapped.end(), r.index), 1)
        << "the mapped kernel should appear once";
    EXPECT_TRUE(std::equal(mapped.begin() + 1, mapped.end(), shipped.begin()))
        << "the shipped choices should follow in their original order";

    auto best = topResults(handle, p, 1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(algoIndex(best[0].algo), r.index);
    runAndCheck(handle, p, best[0].algo, best[0].workspaceSize);

    Problem neighbour;
    ASSERT_EQ(buildProblem(neighbour, 32), HIPBLAS_STATUS_SUCCESS);
    const int other = topIndex(handle, neighbour);
    EXPECT_GE(other, 0) << "a neighbouring shape still resolves";
    EXPECT_LT(other, USER_INDEX_BASE) << "an unmapped shape (M=32) should get a shipped kernel";

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(UserKernelRegister, EndToEnd)
{
    EXPECT_EQ(runWorker("UserKernelRegister.DISABLED_Worker"), 0)
        << "the registration worker failed; its output is above";
}

// Kernels from two payloads, mapped to two shapes, each selected for its own.
TEST(UserKernelRegister, DISABLED_TwoShapesWorker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    expectNothingRegistered(handle);

    Problem small, large;
    ASSERT_EQ(buildProblem(small, 16), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(buildProblem(large, 64), HIPBLAS_STATUS_SUCCESS);
    ASSERT_GE(topIndex(handle, small), 0);
    ASSERT_GE(topIndex(handle, large), 0);

    Registered first = registerUsable(handle, small, candidates);
    ASSERT_GE(first.index, 0);
    Registered second = registerUsable(handle, large, candidates, first.stem);
    ASSERT_GE(second.index, 0);
    std::printf("payloads: %s and %s\n",
                std::filesystem::path(first.stem).filename().string().c_str(),
                std::filesystem::path(second.stem).filename().string().c_str());

    ASSERT_EQ(hipblasLtUserKernelSetExactMatch(handle, first.index, small.m, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtUserKernelSetExactMatch(handle, second.index, large.m, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);

    size_t registered = 0, selectable = 0;
    ASSERT_EQ(hipblasLtUserKernelGetCounts(handle, &registered, &selectable),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(selectable, 2u);

    for(auto const* entry : {&small, &large})
    {
        const int expected = entry == &small ? first.index : second.index;
        auto      best     = topResults(handle, *entry, 1);
        ASSERT_EQ(best.size(), 1u);
        EXPECT_EQ(algoIndex(best[0].algo), expected) << "M=" << entry->m;
        runAndCheck(handle, *entry, best[0].algo, best[0].workspaceSize);
    }

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(UserKernelRegister, TwoShapes)
{
    EXPECT_EQ(runWorker("UserKernelRegister.DISABLED_TwoShapesWorker"), 0)
        << "the worker failed; its output is above";
}

// Registered and mapped through a handle on device 0, then selected and run
// through a handle on device 1, which has never loaded the code object.
TEST(UserKernelRegister, DISABLED_TwoDevicesWorker)
{
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t first;
    ASSERT_EQ(hipblasLtCreate(&first), HIPBLAS_STATUS_SUCCESS);
    expectNothingRegistered(first);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    Registered r = registerUsable(first, p, candidates);
    ASSERT_GE(r.index, 0);
    ASSERT_EQ(hipblasLtUserKernelSetExactMatch(first, r.index, M, N, 1, K), HIPBLAS_STATUS_SUCCESS);

    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    hipblasLtHandle_t second;
    ASSERT_EQ(hipblasLtCreate(&second), HIPBLAS_STATUS_SUCCESS);

    auto best = topResults(second, p, 1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(algoIndex(best[0].algo), r.index) << "selection is shared by every device";
    runAndCheck(second, p, best[0].algo, best[0].workspaceSize);

    EXPECT_EQ(hipblasLtDestroy(second), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    EXPECT_EQ(hipblasLtDestroy(first), HIPBLAS_STATUS_SUCCESS);
}

TEST(UserKernelRegister, TwoDevices)
{
    if(!haveTwoMatchingDevices())
        GTEST_SKIP() << "needs two devices of the same architecture";

    EXPECT_EQ(runWorker("UserKernelRegister.DISABLED_TwoDevicesWorker"), 0)
        << "the worker failed; its output is above";
}

// None of these register anything, so they run in this process.
TEST(UserKernelRegister, RejectsInvalidArguments)
{
    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    const std::string missing    = "/nonexistent/user_kernel_payload";
    int               numIndices = -1;
    EXPECT_EQ(
        hipblasLtUserKernelRegister(
            handle, (missing + ".dat").c_str(), (missing + ".co").c_str(), nullptr, 0, &numIndices),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(numIndices, 0);

    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";
    EXPECT_EQ(hipblasLtUserKernelRegister(handle,
                                          (candidates[0] + ".dat").c_str(),
                                          (missing + ".co").c_str(),
                                          nullptr,
                                          0,
                                          &numIndices),
              HIPBLAS_STATUS_INVALID_VALUE)
        << "a missing code object is refused before anything is registered";

    EXPECT_EQ(hipblasLtUserKernelSetExactMatch(handle, 0, M, N, 1, K), HIPBLAS_STATUS_INVALID_VALUE)
        << "a shipped index cannot be mapped";
    EXPECT_EQ(hipblasLtUserKernelSetExactMatch(handle, INT32_MAX, M, N, 1, K),
              HIPBLAS_STATUS_INVALID_VALUE)
        << "an unregistered user index cannot be mapped";

    int isUser = 0;
    EXPECT_EQ(hipblasLtMatmulAlgoIsUserKernel(nullptr, &isUser), HIPBLAS_STATUS_INVALID_VALUE);

    size_t registered = 123;
    ASSERT_EQ(hipblasLtUserKernelGetCounts(handle, &registered, nullptr), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(registered, 0u) << "a refused registration leaves nothing behind";

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}
