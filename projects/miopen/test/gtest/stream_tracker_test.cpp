// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"

#include <miopen/handle.hpp>
#include <miopen/stream_tracker.hpp>

#include <hip/hip_runtime.h>

namespace {

class GPU_StreamTracker_NONE : public ::testing::Test
{
protected:
    miopen::Handle& handle         = get_handle();
    miopen::StreamTracker& tracker = handle.GetStreamTracker();
};

} // namespace

TEST_F(GPU_StreamTracker_NONE, AcquireRelease)
{
    auto slot = tracker.acquire(handle);
    ASSERT_GT(slot.pool_id, 0);
    ASSERT_NE(slot.stream, nullptr);

    int saved_id = slot.pool_id;
    tracker.release(slot);

    auto slot2 = tracker.acquire(handle);
    EXPECT_EQ(slot2.pool_id, saved_id);
    tracker.release(slot2);
}

TEST_F(GPU_StreamTracker_NONE, AcquireGrowsPool)
{
    auto slot1 = tracker.acquire(handle);
    auto slot2 = tracker.acquire(handle);
    EXPECT_NE(slot1.pool_id, slot2.pool_id);

    tracker.release(slot2);
    tracker.release(slot1);
}

TEST_F(GPU_StreamTracker_NONE, AbandonAndReclaim)
{
    auto slot = tracker.acquire(handle);

    auto* dev_ptr = static_cast<char*>(nullptr);
    ASSERT_EQ(hipMalloc(&dev_ptr, 64), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(dev_ptr, 0, 64, slot.stream), hipSuccess);

    int abandoned_id = slot.pool_id;
    tracker.abandon(slot);

    ASSERT_EQ(hipStreamSynchronize(slot.stream), hipSuccess);

    auto reclaimed = tracker.acquire(handle);
    EXPECT_EQ(reclaimed.pool_id, abandoned_id);
    tracker.release(reclaimed);

    ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
}

TEST_F(GPU_StreamTracker_NONE, AbandonStillDraining)
{
    auto slot = tracker.acquire(handle);

    auto* dev_ptr     = static_cast<char*>(nullptr);
    size_t large_size = 256 * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dev_ptr, large_size), hipSuccess);
    for(int i = 0; i < 64; ++i)
        (void)hipMemsetAsync(dev_ptr, 0, large_size, slot.stream);

    int abandoned_id = slot.pool_id;
    tracker.abandon(slot);

    auto next = tracker.acquire(handle);
    EXPECT_NE(next.pool_id, abandoned_id);
    tracker.release(next);

    ASSERT_EQ(hipStreamSynchronize(slot.stream), hipSuccess);

    auto reclaimed = tracker.acquire(handle);
    EXPECT_EQ(reclaimed.pool_id, abandoned_id);
    tracker.release(reclaimed);

    ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
}

TEST_F(GPU_StreamTracker_NONE, CascadeAbandonReclaim)
{
    constexpr int kCount = 4;
    std::vector<miopen::StreamTracker::Slot> slots;
    std::vector<int> abandoned_ids;

    for(int i = 0; i < kCount; ++i)
    {
        auto slot = tracker.acquire(handle);
        abandoned_ids.push_back(slot.pool_id);
        tracker.abandon(slot);
    }

    for(int i = 0; i < kCount; ++i)
    {
        slots.emplace_back(tracker.acquire(handle));
    }

    // Wait for everything to drain
    for(auto& s : slots)
        tracker.release(s);

    // Now acquire kCount — all should come from available (no new pool growth)
    auto before = tracker.acquire(handle);
    int max_id  = before.pool_id;
    tracker.release(before);

    for(int i = 0; i < kCount; ++i)
    {
        auto s = tracker.acquire(handle);
        EXPECT_LE(s.pool_id, max_id);
        tracker.release(s);
    }
}
