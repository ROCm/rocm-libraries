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
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include "plan_cache.hpp"

namespace
{
    using Queue     = hiptensor::UpdatablePriorityQueue<std::size_t, int>;
    using HashId    = hiptensor::PlanCache::HashId;
    using Uid       = hiptensor::PlanCache::Uid;
    using TimePoint = hiptensor::PlanCache::time_point;

    std::vector<std::pair<std::size_t, int>> drain(Queue& queue)
    {
        std::vector<std::pair<std::size_t, int>> items;
        while(!queue.empty())
        {
            items.push_back(queue.top());
            queue.pop();
        }
        return items;
    }

    // Writes a plan cache file in the layout PlanCache::readFile expects. The time of each
    // line is its index in seconds, so the first line is the least recently used.
    void writeCacheFile(std::string const& path, uint32_t maxCachelines, std::vector<HashId> hashes)
    {
        std::ofstream file(path, std::ios::out | std::ios::binary);
        file.write(reinterpret_cast<char const*>(&maxCachelines), sizeof(maxCachelines));
        std::size_t size = hashes.size();
        file.write(reinterpret_cast<char const*>(&size), sizeof(size));
        for(auto hash : hashes)
        {
            Uid uid = 42;
            file.write(reinterpret_cast<char const*>(&hash), sizeof(hash));
            file.write(reinterpret_cast<char const*>(&uid), sizeof(uid));
        }
        file.write(reinterpret_cast<char const*>(&size), sizeof(size));
        for(std::size_t i = 0; i < hashes.size(); i++)
        {
            TimePoint time{std::chrono::seconds(i + 1)};
            file.write(reinterpret_cast<char const*>(&hashes[i]), sizeof(hashes[i]));
            file.write(reinterpret_cast<char const*>(&time), sizeof(time));
        }
    }

    // Number of cache lines a plan cache file holds, read through a fresh handle.
    uint32_t cachelinesInFile(std::string const& path)
    {
        hiptensorHandle_t handle   = nullptr;
        uint32_t          numLines = 0;
        EXPECT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(handle, path.c_str(), &numLines),
                  HIPTENSOR_STATUS_SUCCESS);
        hiptensorDestroy(handle);
        return numLines;
    }

    // Runs one 4x5 fp32 permutation, which adds its problem to the handle's plan cache.
    void runPermute(hiptensorHandle_t handle)
    {
        constexpr int64_t M = 4, N = 5;
        int64_t           lensA[2] = {M, N};
        int64_t           lensB[2] = {N, M};
        int32_t           modeA[2] = {0, 1};
        int32_t           modeB[2] = {1, 0};

        hiptensorTensorDescriptor_t descA{}, descB{};
        ASSERT_EQ(hiptensorCreateTensorDescriptor(
                      handle, &descA, 2, lensA, nullptr, HIPTENSOR_R_32F, 128),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreateTensorDescriptor(
                      handle, &descB, 2, lensB, nullptr, HIPTENSOR_R_32F, 128),
                  HIPTENSOR_STATUS_SUCCESS);

        hiptensorOperationDescriptor_t opDesc{};
        ASSERT_EQ(hiptensorCreatePermutation(handle,
                                             &opDesc,
                                             descA,
                                             modeA,
                                             HIPTENSOR_OP_IDENTITY,
                                             descB,
                                             modeB,
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);

        hiptensorPlanPreference_t pref{};
        ASSERT_EQ(hiptensorCreatePlanPreference(
                      handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
                  HIPTENSOR_STATUS_SUCCESS);

        uint64_t wsSize = 0;
        ASSERT_EQ(hiptensorEstimateWorkspaceSize(
                      handle, opDesc, pref, HIPTENSOR_WORKSPACE_DEFAULT, &wsSize),
                  HIPTENSOR_STATUS_SUCCESS);

        hiptensorPlan_t plan{};
        ASSERT_EQ(hiptensorCreatePlan(handle, &plan, opDesc, pref, wsSize),
                  HIPTENSOR_STATUS_SUCCESS);

        float *dA = nullptr, *dB = nullptr;
        ASSERT_EQ(hipMalloc(&dA, sizeof(float) * M * N), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, sizeof(float) * M * N), hipSuccess);

        float alpha = 1.0f;
        EXPECT_EQ(hiptensorPermute(handle, plan, &alpha, dA, dB, nullptr),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

        (void)hipFree(dA);
        (void)hipFree(dB);
        hiptensorDestroyPlan(plan);
        hiptensorDestroyPlanPreference(pref);
        hiptensorDestroyOperationDescriptor(opDesc);
        hiptensorDestroyTensorDescriptor(descA);
        hiptensorDestroyTensorDescriptor(descB);
    }

} // namespace

// A copy must own its elements: popping the copy must leave the original intact, and
// destroying both must not free any element twice.
TEST(UpdatablePriorityQueueTest, CopyIsIndependent)
{
    Queue original;
    original.push(1, 30);
    original.push(2, 10);
    original.push(3, 20);

    {
        Queue copy(original);
        EXPECT_EQ(drain(copy),
                  (std::vector<std::pair<std::size_t, int>>{{2, 10}, {3, 20}, {1, 30}}));
    }

    EXPECT_EQ(original.size(), 3u);
    EXPECT_TRUE(original.updateItem(1, 5));
    EXPECT_EQ(drain(original),
              (std::vector<std::pair<std::size_t, int>>{{1, 5}, {2, 10}, {3, 20}}));
}

TEST(UpdatablePriorityQueueTest, AssignmentIsIndependent)
{
    Queue original;
    original.push(1, 30);
    original.push(2, 10);

    Queue assigned;
    assigned.push(9, 1);
    assigned = original;
    EXPECT_EQ(drain(assigned), (std::vector<std::pair<std::size_t, int>>{{2, 10}, {1, 30}}));

    EXPECT_EQ(original.size(), 2u);
    EXPECT_EQ(drain(original), (std::vector<std::pair<std::size_t, int>>{{2, 10}, {1, 30}}));
}

// Writing the plan cache copies its LRU queue. Before the copy was deep, destroying the handle
// afterwards freed every cache line's queue element a second time.
TEST(PlanCacheFileTest, WriteThenDestroyHandle)
{
    auto const input  = ::testing::TempDir() + "hiptensor_plan_cache_lru_in.bin";
    auto const output = ::testing::TempDir() + "hiptensor_plan_cache_lru_out.bin";
    writeCacheFile(input, 8, {0x1111, 0x2222, 0x3333});

    hiptensorHandle_t handle   = nullptr;
    uint32_t          numLines = 0;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorHandleReadPlanCacheFromFile(handle, input.c_str(), &numLines),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(numLines, 3u);

    // Write twice: each write copies the queue.
    ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(handle, output.c_str()),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(handle, output.c_str()),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(hiptensorDestroy(handle), HIPTENSOR_STATUS_SUCCESS);

    EXPECT_EQ(cachelinesInFile(output), 3u);
}

// A full plan cache must evict its least recently used line to make room for a new one.
TEST(PlanCacheFileTest, FullCacheEvictsForNewLine)
{
    auto const input  = ::testing::TempDir() + "hiptensor_plan_cache_full_in.bin";
    auto const output = ::testing::TempDir() + "hiptensor_plan_cache_full_out.bin";
    writeCacheFile(input, 2, {0x1111, 0x2222});

    hiptensorHandle_t handle   = nullptr;
    uint32_t          numLines = 0;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorHandleReadPlanCacheFromFile(handle, input.c_str(), &numLines),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(numLines, 2u);

    // The permutation's problem is new to the full cache.
    runPermute(handle);
    ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(handle, output.c_str()),
              HIPTENSOR_STATUS_SUCCESS);
    hiptensorDestroy(handle);

    EXPECT_EQ(cachelinesInFile(output), 2u);
}
