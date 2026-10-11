// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"

#include <miopen/handle.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/stream_tracker.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <thread>

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

TEST_F(GPU_StreamTracker_FP32, SweepDoesNotInvalidateAnUnrelatedGraphCapture)
{
    // Regression for issue #12121. Reclaiming polls abandoned streams with
    // hipStreamQuery. Under the default global capture mode HIP rejects that
    // call from any thread while any capture is in flight, and the rejection
    // invalidates the capture - so a stream abandoned by one find used to break
    // a graph capture belonging to an entirely unrelated caller.
    auto slot = tracker.acquire(handle);

    StreamGate gate;
    ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, &gate), hipSuccess);
    const auto busy_stream = slot.stream;
    tracker.abandon(std::move(slot));

    // Allocate before the capture opens: hipMalloc is itself illegal during one.
    void* dev_ptr = nullptr;
    ASSERT_EQ(hipMalloc(&dev_ptr, 64), hipSuccess);

    hipStream_t capture_stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&capture_stream, hipStreamNonBlocking), hipSuccess);

    ASSERT_EQ(hipStreamBeginCapture(capture_stream, hipStreamCaptureModeGlobal), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(dev_ptr, 0, 64, capture_stream), hipSuccess);

    // The call under test, standing in for the reclaim that used to ride along
    // on every kernel launch.
    tracker.sweep();

    hipGraph_t graph      = nullptr;
    const auto end_status = hipStreamEndCapture(capture_stream, &graph);

    gate.open();
    ASSERT_EQ(hipStreamSynchronize(busy_stream), hipSuccess);

    EXPECT_EQ(end_status, hipSuccess) << "sweep() invalidated an unrelated capture";
    EXPECT_NE(graph, nullptr);

    if(graph != nullptr)
        (void)hipGraphDestroy(graph);
    (void)hipStreamDestroy(capture_stream);
    ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
}

TEST_F(GPU_StreamTracker_FP32, AcquireWaitsOutABacklogRatherThanGrowOrDecline)
{
    // At MaxDraining(), acquire() waits for a stream to retire and then hands
    // back that one. It must not decline: the caller would skip a solver
    // evaluation, and a skipped naive solver can lose a find it would have won -
    // which reaches the user find-db and outlives the backlog that caused it.
    const auto cap = miopen::StreamTracker::MaxDraining();

    std::vector<std::unique_ptr<StreamGate>> gates;
    std::vector<hipStream_t> abandoned;

    for(std::size_t i = 0; i < cap; ++i)
    {
        auto slot = tracker.acquire(handle);
        ASSERT_NE(slot.stream, nullptr) << "no stream before reaching the cap, at " << i;
        gates.push_back(std::make_unique<StreamGate>());
        ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, gates.back().get()),
                  hipSuccess);
        abandoned.push_back(slot.stream);
        tracker.abandon(std::move(slot));
    }

    // acquire() below blocks until one retires, so the gates open elsewhere.
    std::thread opener([&gates] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        for(auto& g : gates)
            g->open();
    });

    const auto before = std::chrono::steady_clock::now();
    auto slot         = tracker.acquire(handle);
    const std::chrono::duration<double, std::milli> waited =
        std::chrono::steady_clock::now() - before;
    opener.join();

    ASSERT_NE(slot.stream, nullptr) << "declined instead of waiting";
    EXPECT_NE(std::find(abandoned.begin(), abandoned.end(), slot.stream), abandoned.end())
        << "grew the pool past the backlog cap instead of waiting";
    EXPECT_GT(waited.count(), 50.0) << "returned without waiting for the backlog";
    tracker.release(slot);
}

TEST_F(GPU_StreamTracker_FP32, ReclaimTracksTheDoneEventWhenOneIsSupplied)
{
    // A slot carrying a completion event is reclaimed on that event rather than
    // on a stream poll, and not before it signals.
    auto slot = tracker.acquire(handle);

    StreamGate gate;
    ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, &gate), hipSuccess);

    auto done = miopen::make_hip_event();
    ASSERT_EQ(hipEventRecord(done.get(), slot.stream), hipSuccess);

    const auto gated_stream = slot.stream;
    slot.done               = miopen::StreamTracker::EventPtr{std::move(done)};
    tracker.abandon(std::move(slot));

    // Event is queued behind the gate, so nothing may be reclaimed yet.
    tracker.sweep();
    auto other = tracker.acquire(handle);
    EXPECT_NE(other.stream, gated_stream);
    tracker.release(other);

    gate.open();
    ASSERT_EQ(hipStreamSynchronize(gated_stream), hipSuccess);

    tracker.sweep();
    auto reclaimed = tracker.acquire(handle);
    EXPECT_EQ(reclaimed.stream, gated_stream);
    EXPECT_EQ(reclaimed.done, nullptr) << "reclaim left the spent event on the slot";
    tracker.release(reclaimed);
}

TEST_F(GPU_StreamTracker_FP32, PinnedScratchGatesTheLaunchPathAndClearsWithTheBuffer)
{
    auto prev     = handle.GetScratchBuffer(1);
    const auto sz = (prev ? prev->size : 0) + 65536;
    prev.reset();

    auto scratch = handle.GetScratchBuffer(sz);
    ASSERT_NE(scratch, nullptr);
    EXPECT_FALSE(tracker.HasPinnedScratch());

    auto slot    = tracker.acquire(handle);
    slot.scratch = scratch;
    tracker.abandon(std::move(slot));
    EXPECT_TRUE(tracker.HasPinnedScratch()) << "launch path would skip a pinned buffer";

    tracker.sweep();

    // Buffer handed back, so the launch-path test goes quiet again even though
    // the slot itself lives on in the pool.
    EXPECT_EQ(scratch.use_count(), 1);
    EXPECT_FALSE(tracker.HasPinnedScratch()) << "launch path still paying after the release";
}

TEST_F(GPU_StreamTracker_FP32, ReclaimIsSkippedWhileTheGivenStreamIsCapturing)
{
    // The launch-path entry point reclaims only outside a capture: the release
    // ends in a deallocator call, which has no business running mid-recording.
    auto scratch = handle.GetScratchBuffer(65536);
    ASSERT_NE(scratch, nullptr);

    auto slot    = tracker.acquire(handle);
    slot.scratch = scratch;
    tracker.abandon(std::move(slot)); // stream is idle, so a sweep would reclaim

    void* dev_ptr = nullptr;
    ASSERT_EQ(hipMalloc(&dev_ptr, 64), hipSuccess);
    hipStream_t capture_stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&capture_stream, hipStreamNonBlocking), hipSuccess);

    ASSERT_EQ(hipStreamBeginCapture(capture_stream, hipStreamCaptureModeGlobal), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(dev_ptr, 0, 64, capture_stream), hipSuccess);

    tracker.SweepUnlessCapturing(capture_stream);
    EXPECT_TRUE(tracker.HasPinnedScratch()) << "released a buffer mid-capture";

    hipGraph_t graph = nullptr;
    EXPECT_EQ(hipStreamEndCapture(capture_stream, &graph), hipSuccess);

    // Same call once the capture is closed does reclaim.
    tracker.SweepUnlessCapturing(handle.GetStream());
    EXPECT_FALSE(tracker.HasPinnedScratch()) << "failed to reclaim outside a capture";

    if(graph != nullptr)
        (void)hipGraphDestroy(graph);
    (void)hipStreamDestroy(capture_stream);
    ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
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
