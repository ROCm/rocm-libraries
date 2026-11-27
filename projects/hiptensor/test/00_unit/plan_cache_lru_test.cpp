/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include "plan_cache.hpp"
#include "util.hpp"

namespace
{
    using Queue     = hiptensor::UpdatablePriorityQueue<std::size_t, int>;
    using Items     = std::vector<std::pair<std::size_t, int>>;
    using HashId    = hiptensor::PlanCache::HashId;
    using Uid       = hiptensor::PlanCache::Uid;
    using TimePoint = hiptensor::PlanCache::time_point;

    // Runs a cleanup function when the scope exits, including through a failed
    // ASSERT_* or GTEST_SKIP, which return early.
    template <typename F>
    struct ScopeExit
    {
        F mFn;
        ~ScopeExit()
        {
            mFn();
        }
    };
    template <typename F>
    ScopeExit(F) -> ScopeExit<F>;

    Items drain(Queue& queue)
    {
        Items items;
        while(!queue.empty())
        {
            items.push_back(queue.top());
            queue.pop();
        }
        return items;
    }

    // A file in the test temp directory, unique to this process and test, removed on
    // destruction. ctest registers this binary once per test category, so several copies can
    // run at the same time and must not share files.
    struct TempFile
    {
        std::string mPath;

        explicit TempFile(char const* tag)
        {
            std::ostringstream name;
            name << ::testing::TempDir() << "hiptensor_"
                 << ::testing::UnitTest::GetInstance()->current_test_info()->name() << "_" << tag
                 << "_" << std::hex << std::random_device{}() << std::random_device{}() << ".bin";
            mPath = name.str();
        }
        ~TempFile()
        {
            std::remove(mPath.c_str());
        }
        TempFile(TempFile const&)            = delete;
        TempFile& operator=(TempFile const&) = delete;

        char const* c_str() const
        {
            return mPath.c_str();
        }
    };

    struct CacheLine
    {
        HashId hash;
        Uid    uid;
    };

    // Writes a plan cache file in the layout PlanCache::readFile expects. The time of each
    // line is its index in seconds, so the first line is the least recently used.
    void writeCacheFile(TempFile const& file, uint32_t maxCachelines, std::vector<CacheLine> lines)
    {
        std::ofstream out(file.mPath, std::ios::out | std::ios::binary);
        out.write(reinterpret_cast<char const*>(&maxCachelines), sizeof(maxCachelines));
        std::size_t size = lines.size();
        out.write(reinterpret_cast<char const*>(&size), sizeof(size));
        for(auto const& line : lines)
        {
            out.write(reinterpret_cast<char const*>(&line.hash), sizeof(line.hash));
            out.write(reinterpret_cast<char const*>(&line.uid), sizeof(line.uid));
        }
        out.write(reinterpret_cast<char const*>(&size), sizeof(size));
        for(std::size_t i = 0; i < lines.size(); i++)
        {
            TimePoint time{std::chrono::seconds(i + 1)};
            out.write(reinterpret_cast<char const*>(&lines[i].hash), sizeof(lines[i].hash));
            out.write(reinterpret_cast<char const*>(&time), sizeof(time));
        }
    }

    // Reads the cache lines of a plan cache file written by PlanCache::writeFile.
    std::vector<CacheLine> readCacheFile(TempFile const& file)
    {
        std::ifstream in(file.mPath, std::ios::in | std::ios::binary);
        uint32_t      maxCachelines = 0;
        std::size_t   size          = 0;
        in.read(reinterpret_cast<char*>(&maxCachelines), sizeof(maxCachelines));
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        std::vector<CacheLine> lines(size);
        for(auto& line : lines)
        {
            in.read(reinterpret_cast<char*>(&line.hash), sizeof(line.hash));
            in.read(reinterpret_cast<char*>(&line.uid), sizeof(line.uid));
        }
        EXPECT_TRUE(in.good()) << "truncated plan cache file " << file.mPath;
        return lines;
    }

    std::set<HashId> hashesIn(TempFile const& file)
    {
        std::set<HashId> hashes;
        for(auto const& line : readCacheFile(file))
        {
            hashes.insert(line.hash);
        }
        return hashes;
    }

    // Runs one 4x5 fp32 permutation, which adds its problem to the handle's plan cache.
    // Call it through ASSERT_NO_FATAL_FAILURE.
    void runPermute(hiptensorHandle_t handle)
    {
        constexpr int64_t M = 4, N = 5;
        int64_t           lensA[2] = {M, N};
        int64_t           lensB[2] = {N, M};
        int32_t           modeA[2] = {0, 1};
        int32_t           modeB[2] = {1, 0};

        hiptensorTensorDescriptor_t    descA{}, descB{};
        hiptensorOperationDescriptor_t opDesc{};
        hiptensorPlanPreference_t      pref{};
        hiptensorPlan_t                plan{};
        float *                        dA = nullptr, *dB = nullptr;
        ScopeExit                      cleanup{[&] {
            (void)hipFree(dA);
            (void)hipFree(dB);
            if(plan)
                hiptensorDestroyPlan(plan);
            if(pref)
                hiptensorDestroyPlanPreference(pref);
            if(opDesc)
                hiptensorDestroyOperationDescriptor(opDesc);
            if(descA)
                hiptensorDestroyTensorDescriptor(descA);
            if(descB)
                hiptensorDestroyTensorDescriptor(descB);
        }};

        ASSERT_EQ(hiptensorCreateTensorDescriptor(
                      handle, &descA, 2, lensA, nullptr, HIPTENSOR_R_32F, 128),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreateTensorDescriptor(
                      handle, &descB, 2, lensB, nullptr, HIPTENSOR_R_32F, 128),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreatePermutation(handle,
                                             &opDesc,
                                             descA,
                                             modeA,
                                             HIPTENSOR_OP_IDENTITY,
                                             descB,
                                             modeB,
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreatePlanPreference(
                      handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
                  HIPTENSOR_STATUS_SUCCESS);

        uint64_t wsSize = 0;
        ASSERT_EQ(hiptensorEstimateWorkspaceSize(
                      handle, opDesc, pref, HIPTENSOR_WORKSPACE_DEFAULT, &wsSize),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreatePlan(handle, &plan, opDesc, pref, wsSize),
                  HIPTENSOR_STATUS_SUCCESS);

        ASSERT_EQ(hipMalloc(&dA, sizeof(float) * M * N), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, sizeof(float) * M * N), hipSuccess);

        float alpha = 1.0f;
        ASSERT_EQ(hiptensorPermute(handle, plan, &alpha, dA, dB, nullptr),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    // The cache line the permutation in runPermute adds, taken from a fresh handle.
    // Call it through ASSERT_NO_FATAL_FAILURE.
    void permuteCacheLine(CacheLine* line)
    {
        TempFile          file("permute");
        hiptensorHandle_t handle = nullptr;
        ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
        ScopeExit destroyHandle{[&] { hiptensorDestroy(handle); }};

        ASSERT_NO_FATAL_FAILURE(runPermute(handle));
        ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(handle, file.c_str()),
                  HIPTENSOR_STATUS_SUCCESS);

        auto lines = readCacheFile(file);
        ASSERT_EQ(lines.size(), 1u);
        *line = lines[0];
    }

} // namespace

// The queue owns its elements, so copying it would share them; it must be move-only.
TEST(UpdatablePriorityQueueTest, IsMoveOnly)
{
    static_assert(!std::is_copy_constructible_v<Queue>);
    static_assert(!std::is_copy_assignable_v<Queue>);
    static_assert(std::is_move_constructible_v<Queue>);
    static_assert(std::is_move_assignable_v<Queue>);
}

TEST(UpdatablePriorityQueueTest, SortedMatchesPopOrderWithoutChangingQueue)
{
    Queue queue;
    queue.push(1, 30);
    queue.push(2, 10);
    queue.push(3, 20);
    queue.push(4, 40);
    EXPECT_TRUE(queue.updateItem(4, 5));

    auto sorted = queue.sorted();
    EXPECT_EQ(sorted, (Items{{4, 5}, {2, 10}, {3, 20}, {1, 30}}));
    EXPECT_EQ(queue.size(), 4u);
    EXPECT_EQ(drain(queue), sorted);
}

// Moving keeps the elements, and the moved-to queue can still update them by key.
TEST(UpdatablePriorityQueueTest, MoveKeepsElements)
{
    Queue original;
    original.push(1, 30);
    original.push(2, 10);

    Queue moved(std::move(original));
    EXPECT_TRUE(moved.updateItem(1, 5));
    EXPECT_EQ(drain(moved), (Items{{1, 5}, {2, 10}}));
}

class PlanCacheFileTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Without a plan cache, the read and write functions succeed without doing anything.
        if(hiptensor::checkEnvironmentVariableEnabled("HIPTENSOR_DISABLE_PLAN_CACHE"))
        {
            GTEST_SKIP() << "plan cache disabled by HIPTENSOR_DISABLE_PLAN_CACHE";
        }
    }
};

// Writing the plan cache used to copy its LRU queue without copying the elements, so
// destroying the handle afterwards freed every cache line's element a second time.
TEST_F(PlanCacheFileTest, WriteThenDestroyHandle)
{
    TempFile input("in"), output("out");
    writeCacheFile(input, 8, {{0x1111, 1}, {0x2222, 2}, {0x3333, 3}});

    hiptensorHandle_t handle   = nullptr;
    uint32_t          numLines = 0;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    ScopeExit destroyHandle{[&] {
        if(handle)
            hiptensorDestroy(handle);
    }};
    ASSERT_EQ(hiptensorHandleReadPlanCacheFromFile(handle, input.c_str(), &numLines),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(numLines, 3u);

    // Write twice: each write reads the whole queue.
    EXPECT_EQ(hiptensorHandleWritePlanCacheToFile(handle, output.c_str()),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(hiptensorHandleWritePlanCacheToFile(handle, output.c_str()),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(hiptensorDestroy(handle), HIPTENSOR_STATUS_SUCCESS);
    handle = nullptr;

    EXPECT_EQ(hashesIn(output), (std::set<HashId>{0x1111, 0x2222, 0x3333}));
}

// A full plan cache evicts its least recently used line to make room for a new one.
TEST_F(PlanCacheFileTest, FullCacheEvictsLeastRecentlyUsedForNewLine)
{
    CacheLine permute{};
    ASSERT_NO_FATAL_FAILURE(permuteCacheLine(&permute));

    TempFile input("in"), output("out");
    writeCacheFile(input, 2, {{0x1111, 1}, {0x2222, 2}});

    hiptensorHandle_t handle   = nullptr;
    uint32_t          numLines = 0;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    ScopeExit destroyHandle{[&] { hiptensorDestroy(handle); }};
    ASSERT_EQ(hiptensorHandleReadPlanCacheFromFile(handle, input.c_str(), &numLines),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(numLines, 2u);

    ASSERT_NO_FATAL_FAILURE(runPermute(handle));
    ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(handle, output.c_str()),
              HIPTENSOR_STATUS_SUCCESS);

    // 0x1111 is the oldest line.
    EXPECT_EQ(hashesIn(output), (std::set<HashId>{0x2222, permute.hash}));
}

// Adding a line the full cache already holds updates it in place and evicts nothing.
TEST_F(PlanCacheFileTest, FullCacheUpdatesExistingLineInPlace)
{
    CacheLine permute{};
    ASSERT_NO_FATAL_FAILURE(permuteCacheLine(&permute));

    TempFile input("in"), output("out");
    writeCacheFile(input, 2, {{0x1111, 1}, permute});

    hiptensorHandle_t handle   = nullptr;
    uint32_t          numLines = 0;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    ScopeExit destroyHandle{[&] { hiptensorDestroy(handle); }};
    ASSERT_EQ(hiptensorHandleReadPlanCacheFromFile(handle, input.c_str(), &numLines),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(numLines, 2u);

    ASSERT_NO_FATAL_FAILURE(runPermute(handle));
    ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(handle, output.c_str()),
              HIPTENSOR_STATUS_SUCCESS);

    EXPECT_EQ(hashesIn(output), (std::set<HashId>{0x1111, permute.hash}));
}

namespace
{
    // Runs the trinary contraction E[m,n] = A[m,k] * B[k,q] * C[q,n] + D[m,n] once on
    // handle, which adds its kernel to the handle's plan cache. A to D are packed and
    // stay the same; E has 4 x 5 elements and the given strides.
    // With created, it stores the status of creating the contraction there and returns
    // without running when the device has no FP32 contraction (ARCH_MISMATCH), so the
    // caller can skip. Without created, creating the contraction must succeed.
    // Call it through ASSERT_NO_FATAL_FAILURE.
    void runTrinary(hiptensorHandle_t  handle,
                    int64_t const      stridesE[2],
                    hiptensorStatus_t* created = nullptr)
    {
        constexpr int64_t M = 4, N = 5, K = 3, Q = 2;
        int64_t const     lensA[]{M, K}, lensB[]{K, Q}, lensC[]{Q, N}, lensD[]{M, N};
        int32_t const modeA[]{'m', 'k'}, modeB[]{'k', 'q'}, modeC[]{'q', 'n'}, modeD[]{'m', 'n'};
        // E sits on positions 0 to (M - 1) * strideM + (N - 1) * strideN.
        std::size_t const elementsE = (M - 1) * stridesE[0] + (N - 1) * stridesE[1] + 1;

        hiptensorTensorDescriptor_t descA = nullptr, descB = nullptr, descC = nullptr,
                                    descD = nullptr, descE = nullptr;
        hiptensorOperationDescriptor_t opDesc = nullptr;
        hiptensorPlanPreference_t      pref   = nullptr;
        hiptensorPlan_t                plan   = nullptr;
        void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dE = nullptr,
             *workspace = nullptr;
        ScopeExit cleanup{[&] {
            for(auto ptr : {dA, dB, dC, dD, dE, workspace})
                (void)hipFree(ptr);
            if(plan)
                hiptensorDestroyPlan(plan);
            if(pref)
                hiptensorDestroyPlanPreference(pref);
            if(opDesc)
                hiptensorDestroyOperationDescriptor(opDesc);
            for(auto desc : {descA, descB, descC, descD, descE})
                if(desc)
                    hiptensorDestroyTensorDescriptor(desc);
        }};

        for(auto [desc, lens] : {std::pair{&descA, lensA},
                                 std::pair{&descB, lensB},
                                 std::pair{&descC, lensC},
                                 std::pair{&descD, lensD}})
        {
            ASSERT_EQ(hiptensorCreateTensorDescriptor(
                          handle, desc, 2, lens, nullptr, HIPTENSOR_R_32F, 128),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        ASSERT_EQ(hiptensorCreateTensorDescriptor(
                      handle, &descE, 2, lensD, stridesE, HIPTENSOR_R_32F, 128),
                  HIPTENSOR_STATUS_SUCCESS);
        auto const status = hiptensorCreateContractionTrinary(handle,
                                                              &opDesc,
                                                              descA,
                                                              modeA,
                                                              HIPTENSOR_OP_IDENTITY,
                                                              descB,
                                                              modeB,
                                                              HIPTENSOR_OP_IDENTITY,
                                                              descC,
                                                              modeC,
                                                              HIPTENSOR_OP_IDENTITY,
                                                              descD,
                                                              modeD,
                                                              HIPTENSOR_OP_IDENTITY,
                                                              descE,
                                                              modeD,
                                                              HIPTENSOR_COMPUTE_DESC_32F);
        if(created)
        {
            *created = status;
            if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
            {
                return;
            }
        }
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreatePlanPreference(
                      handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
                  HIPTENSOR_STATUS_SUCCESS);

        uint64_t wsSize = 0;
        ASSERT_EQ(hiptensorEstimateWorkspaceSize(
                      handle, opDesc, pref, HIPTENSOR_WORKSPACE_DEFAULT, &wsSize),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreatePlan(handle, &plan, opDesc, pref, wsSize),
                  HIPTENSOR_STATUS_SUCCESS);

        for(auto [ptr, elements] : {std::pair{&dA, M * K},
                                    std::pair{&dB, K * Q},
                                    std::pair{&dC, Q * N},
                                    std::pair{&dD, M * N},
                                    std::pair{&dE, static_cast<int64_t>(elementsE)}})
        {
            ASSERT_EQ(hipMalloc(ptr, sizeof(float) * elements), hipSuccess);
            ASSERT_EQ(hipMemset(*ptr, 0, sizeof(float) * elements), hipSuccess);
        }
        if(wsSize > 0)
        {
            ASSERT_EQ(hipMalloc(&workspace, wsSize), hipSuccess);
        }

        float alpha = 1.0f, beta = 1.0f;
        ASSERT_EQ(hiptensorContractTrinary(
                      handle, plan, &alpha, dA, dB, dC, &beta, dD, dE, workspace, wsSize, nullptr),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    constexpr int64_t kPackedE[]{1, 4};
    // Each column of 4 elements starts 7 elements after the previous one.
    constexpr int64_t kPaddedE[]{1, 7};

    // The number of lines in handle's plan cache.
    // Call it through ASSERT_NO_FATAL_FAILURE.
    void countCacheLines(hiptensorHandle_t handle, std::size_t* count)
    {
        TempFile file("trinary");
        ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(handle, file.c_str()),
                  HIPTENSOR_STATUS_SUCCESS);
        *count = readCacheFile(file).size();
    }
} // namespace

// The control: running one trinary contraction twice adds one cache line.
TEST_F(PlanCacheFileTest, SameTrinaryContractionOneCacheLine)
{
    hiptensorHandle_t handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    ScopeExit destroyHandle{[&] { hiptensorDestroy(handle); }};

    hiptensorStatus_t created{};
    ASSERT_NO_FATAL_FAILURE(runTrinary(handle, kPackedE, &created));
    if(created == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "no FP32 contraction on this device";
    }
    ASSERT_NO_FATAL_FAILURE(runTrinary(handle, kPackedE));

    std::size_t count = 0;
    ASSERT_NO_FATAL_FAILURE(countCacheLines(handle, &count));
    EXPECT_EQ(count, 1u);
}

// A cached kernel is reused for every problem with the same hash, so two trinary
// contractions that differ only in E's strides must get a cache line each: the kernel
// chosen for one layout of E may not support the other.
TEST_F(PlanCacheFileTest, TrinaryContractionsDifferingInEStridesTwoCacheLines)
{
    hiptensorHandle_t handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    ScopeExit destroyHandle{[&] { hiptensorDestroy(handle); }};

    hiptensorStatus_t created{};
    ASSERT_NO_FATAL_FAILURE(runTrinary(handle, kPackedE, &created));
    if(created == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "no FP32 contraction on this device";
    }
    ASSERT_NO_FATAL_FAILURE(runTrinary(handle, kPaddedE));

    std::size_t count = 0;
    ASSERT_NO_FATAL_FAILURE(countCacheLines(handle, &count));
    EXPECT_EQ(count, 2u);
}
