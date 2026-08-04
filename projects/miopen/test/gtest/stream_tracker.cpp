// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"

#include <miopen/handle.hpp>
#include <miopen/stream_tracker.hpp>

#include <hip/hip_runtime.h>

#include <condition_variable>
#include <mutex>

namespace {

struct StreamGate
{
    std::mutex mtx;
    std::condition_variable cv;
    bool released = false;

    static void callback(void* arg)
    {
        auto* self = static_cast<StreamGate*>(arg);
        std::unique_lock<std::mutex> lk(self->mtx);
        self->cv.wait(lk, [self] { return self->released; });
    }

    void open()
    {
        {
            std::lock_guard<std::mutex> lk(mtx);
            released = true;
        }
        cv.notify_one();
    }
};

class GPU_StreamTracker_FP32 : public ::testing::Test
{
protected:
    miopen::Handle& handle = get_handle();
    miopen::StreamTracker tracker;
};

} // namespace

TEST_F(GPU_StreamTracker_FP32, AcquireRelease)
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

TEST_F(GPU_StreamTracker_FP32, AcquireGrowsPool)
{
    auto slot1 = tracker.acquire(handle);
    auto slot2 = tracker.acquire(handle);
    EXPECT_NE(slot1.pool_id, slot2.pool_id);

    tracker.release(slot2);
    tracker.release(slot1);
}

TEST_F(GPU_StreamTracker_FP32, AbandonAndReclaim)
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

TEST_F(GPU_StreamTracker_FP32, AbandonStillDraining)
{
    auto slot = tracker.acquire(handle);

    StreamGate gate;
    ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, &gate), hipSuccess);

    int abandoned_id = slot.pool_id;
    tracker.abandon(slot);

    auto next = tracker.acquire(handle);
    EXPECT_NE(next.pool_id, abandoned_id);

    gate.open();
    ASSERT_EQ(hipStreamSynchronize(slot.stream), hipSuccess);

    // Don't release `next` yet — keep available_ empty so acquire scans draining
    auto reclaimed = tracker.acquire(handle);
    EXPECT_EQ(reclaimed.pool_id, abandoned_id);
    tracker.release(reclaimed);
    tracker.release(next);
}

TEST_F(GPU_StreamTracker_FP32, CascadeAbandonReclaim)
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

TEST_F(GPU_StreamTracker_FP32, ScratchAllocateAndReuse)
{
    auto s1 = handle.GetScratchBuffer(1024);
    ASSERT_NE(s1, nullptr);
    EXPECT_GE(s1->size, 1024u);

    // Same or smaller request while s1 is alive → same allocation returned
    auto s2 = handle.GetScratchBuffer(s1->size);
    EXPECT_EQ(s1, s2);

    auto s3 = handle.GetScratchBuffer(1);
    EXPECT_EQ(s1, s3);
}

TEST_F(GPU_StreamTracker_FP32, ScratchFreedWhenCallersRelease)
{
    // Core memory-pressure fix: Handle holds only weak_ptr, so scratch is freed
    // as soon as all callers (TryNaiveWithTimeout stack frame + draining slots) drop refs.
    auto scratch = handle.GetScratchBuffer(1024);
    ASSERT_NE(scratch, nullptr);
    EXPECT_EQ(scratch.use_count(), 1); // only local; Handle has weak_ptr

    std::weak_ptr<ScratchAllocation> weak = scratch;
    scratch.reset();             // simulate caller (find phase) completing
    EXPECT_TRUE(weak.expired()); // no strong refs remain → allocation freed

    // Next call must allocate fresh (weak_ptr expired)
    auto s2 = handle.GetScratchBuffer(1024);
    ASSERT_NE(s2, nullptr);
    EXPECT_EQ(s2.use_count(), 1);
}

TEST_F(GPU_StreamTracker_FP32, ScratchGrows)
{
    auto s1 = handle.GetScratchBuffer(1);
    ASSERT_NE(s1, nullptr);
    auto* raw1 = s1->buffer.get();

    auto s2 = handle.GetScratchBuffer(s1->size + 1);
    ASSERT_NE(s2, nullptr);
    EXPECT_NE(s2->buffer.get(), raw1);
    EXPECT_GE(s2->size, s1->size + 1);
}

TEST_F(GPU_StreamTracker_FP32, ScratchReturnsNullOnOversize)
{
    auto s = handle.GetScratchBuffer(handle.GetGlobalMemorySize());
    EXPECT_EQ(s, nullptr);
}

TEST_F(GPU_StreamTracker_FP32, ScratchReturnsNullOnZero)
{
    auto s = handle.GetScratchBuffer(0);
    EXPECT_EQ(s, nullptr);
}

TEST_F(GPU_StreamTracker_FP32, ScratchSurvivesAbandon)
{
    auto prev     = handle.GetScratchBuffer(1);
    const auto sz = (prev ? prev->size : 0) + 65536;
    prev.reset();

    auto scratch = handle.GetScratchBuffer(sz);
    ASSERT_NE(scratch, nullptr);
    // Handle holds weak_ptr only; local is the sole strong ref
    EXPECT_EQ(scratch.use_count(), 1);

    auto slot    = tracker.acquire(handle);
    slot.scratch = scratch;
    EXPECT_EQ(scratch.use_count(), 2); // local + slot

    tracker.abandon(std::move(slot));
    EXPECT_EQ(scratch.use_count(), 2); // local + draining slot

    // No work on stream → hipStreamQuery succeeds → reclaim resets scratch
    auto reclaimed = tracker.acquire(handle);
    EXPECT_EQ(scratch.use_count(), 1); // draining slot scratch reset; only local remains
    tracker.release(reclaimed);
}
