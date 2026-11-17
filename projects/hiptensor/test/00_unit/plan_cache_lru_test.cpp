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
