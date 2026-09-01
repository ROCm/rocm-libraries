// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"

#include <miopen/handle.hpp>
#include <miopen/stream_tracker.hpp>

#include <hip/hip_runtime.h>

#include <condition_variable>
#include <mutex>
#include <set>

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
    ASSERT_NE(slot.stream, nullptr);

    auto saved_stream = slot.stream;
    tracker.release(slot);

    auto slot2 = tracker.acquire(handle);
    EXPECT_EQ(slot2.stream, saved_stream);
    tracker.release(slot2);
}

TEST_F(GPU_StreamTracker_FP32, AcquireIsNotAHandlePoolStream)
{
    // The Handle's stream-pool indices are claimed by hardcoded number elsewhere
    // (MHA, RNN), so a tracker slot must never alias one of them.
    constexpr int kPoolSize = 2;
    handle.ReserveExtraStreamsInPool(kPoolSize);

    auto slot = tracker.acquire(handle);
    ASSERT_NE(slot.stream, nullptr);

    for(int id = 0; id <= kPoolSize; ++id)
    {
        handle.SetStreamFromPool(id);
        EXPECT_NE(slot.stream, handle.GetStream()) << "aliases stream-pool id " << id;
    }
    handle.SetStreamFromPool(0);

    tracker.release(slot);
}

TEST_F(GPU_StreamTracker_FP32, AcquireGrowsPool)
{
    auto slot1 = tracker.acquire(handle);
    auto slot2 = tracker.acquire(handle);
    EXPECT_NE(slot1.stream, slot2.stream);

    tracker.release(slot2);
    tracker.release(slot1);
}

TEST_F(GPU_StreamTracker_FP32, AbandonAndReclaim)
{
    auto slot = tracker.acquire(handle);

    auto* dev_ptr = static_cast<char*>(nullptr);
    ASSERT_EQ(hipMalloc(&dev_ptr, 64), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(dev_ptr, 0, 64, slot.stream), hipSuccess);

    auto abandoned_stream = slot.stream;
    tracker.abandon(slot);

    ASSERT_EQ(hipStreamSynchronize(abandoned_stream), hipSuccess);

    auto reclaimed = tracker.acquire(handle);
    EXPECT_EQ(reclaimed.stream, abandoned_stream);
    tracker.release(reclaimed);

    ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
}

TEST_F(GPU_StreamTracker_FP32, AbandonStillDraining)
{
    auto slot = tracker.acquire(handle);

    StreamGate gate;
    ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, &gate), hipSuccess);

    auto abandoned_stream = slot.stream;
    tracker.abandon(slot);

    auto next = tracker.acquire(handle);
    EXPECT_NE(next.stream, abandoned_stream);

    gate.open();
    ASSERT_EQ(hipStreamSynchronize(abandoned_stream), hipSuccess);

    // Don't release `next` yet — keep available_ empty so acquire scans draining
    auto reclaimed = tracker.acquire(handle);
    EXPECT_EQ(reclaimed.stream, abandoned_stream);
    tracker.release(reclaimed);
    tracker.release(next);
}

TEST_F(GPU_StreamTracker_FP32, CascadeAbandonReclaim)
{
    constexpr int kCount = 4;
    std::vector<miopen::StreamTracker::Slot> slots;
    std::set<hipStream_t> seen;

    for(int i = 0; i < kCount; ++i)
    {
        auto slot = tracker.acquire(handle);
        seen.insert(slot.stream);
        tracker.abandon(slot);
    }

    for(int i = 0; i < kCount; ++i)
    {
        auto slot = tracker.acquire(handle);
        seen.insert(slot.stream);
        slots.emplace_back(std::move(slot));
    }

    for(auto& s : slots)
        tracker.release(s);

    // Every stream is idle and reclaimed, so acquiring kCount more must not
    // create any stream that hasn't been handed out before.
    for(int i = 0; i < kCount; ++i)
    {
        auto s = tracker.acquire(handle);
        EXPECT_EQ(seen.count(s.stream), 1u);
        tracker.release(s);
    }
}

TEST_F(GPU_StreamTracker_FP32, SweepReclaimsIdleStream)
{
    auto slot                   = tracker.acquire(handle);
    const auto abandoned_stream = slot.stream;
    tracker.abandon(std::move(slot));

    tracker.sweep();

    auto reclaimed = tracker.acquire(handle);
    EXPECT_EQ(reclaimed.stream, abandoned_stream);
    tracker.release(reclaimed);
}

TEST_F(GPU_StreamTracker_FP32, SweepLeavesBusyStreamDraining)
{
    auto slot = tracker.acquire(handle);

    StreamGate gate;
    ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, &gate), hipSuccess);

    auto busy_stream = slot.stream;
    tracker.abandon(std::move(slot));

    tracker.sweep();

    // Still gated, so the slot must not have been reclaimed
    auto next = tracker.acquire(handle);
    EXPECT_NE(next.stream, busy_stream);
    tracker.release(next);

    gate.open();
    ASSERT_EQ(hipStreamSynchronize(busy_stream), hipSuccess);

    tracker.sweep();

    // available_ is LIFO, so the just-swept slot is on top
    auto reclaimed = tracker.acquire(handle);
    EXPECT_EQ(reclaimed.stream, busy_stream);
    tracker.release(reclaimed);
}

TEST_F(GPU_StreamTracker_FP32, SweepReleasesScratch)
{
    auto prev     = handle.GetScratchBuffer(1);
    const auto sz = (prev ? prev->size : 0) + 65536;
    prev.reset();

    auto scratch = handle.GetScratchBuffer(sz);
    ASSERT_NE(scratch, nullptr);
    ASSERT_EQ(scratch.use_count(), 1);

    auto slot    = tracker.acquire(handle);
    slot.scratch = scratch;
    tracker.abandon(std::move(slot));
    ASSERT_EQ(scratch.use_count(), 2); // local + draining slot

    // No work on the stream, so sweep reclaims and drops the slot's reference
    tracker.sweep();
    EXPECT_EQ(scratch.use_count(), 1);
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
    // Fresh Handle so no other test can hold a ref to this scratch allocation.
    miopen::Handle fresh_handle{};

    auto scratch = fresh_handle.GetScratchBuffer(1024);
    ASSERT_NE(scratch, nullptr);
    EXPECT_GE(scratch->size, 1024u);

    std::weak_ptr<miopen::ScratchAllocation> weak = scratch;
    scratch.reset();
    EXPECT_TRUE(weak.expired());
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
