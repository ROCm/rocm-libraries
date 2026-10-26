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
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

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

// Fixed rather than DefaultMaxDraining(): several tests abandon a gated stream
// and then acquire on the same thread before opening the gate, which needs room
// for a second stream. A cap of 1, from the caller's GPU_MAX_HW_QUEUES, would
// block that acquire() on a gate only this thread can open.
constexpr std::size_t kTestMaxDraining = 2;

class GPU_StreamTracker_FP32 : public ::testing::Test
{
protected:
    miopen::Handle& handle = get_handle();
    miopen::StreamTracker tracker{kTestMaxDraining};
};

/// Abandoned scratch is released by the tracker's worker thread, microseconds
/// after its kernel retires. The bound is generous so a loaded machine is not a
/// failure; a release that never comes still is.
bool ReleasedWithin(const std::weak_ptr<miopen::ScratchAllocation>& scratch,
                    std::chrono::milliseconds limit = std::chrono::seconds(10))
{
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while(!scratch.expired())
    {
        if(std::chrono::steady_clock::now() > deadline)
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

/// A scratch buffer no other test holds, so its only owner is the slot it is
/// given to and its release is observable. Call it before gating any stream:
/// dropping the probe buffer frees it with hipFree, which waits for every
/// stream on the device, a gated one included.
std::shared_ptr<miopen::ScratchAllocation> FreshScratch(miopen::Handle& handle)
{
    auto prev     = handle.GetScratchBuffer(1);
    const auto sz = (prev ? prev->size : 0) + 65536;
    prev.reset();
    return handle.GetScratchBuffer(sz);
}

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

TEST_F(GPU_StreamTracker_FP32, AbandonedScratchIsReleasedWhenItsKernelRetires)
{
    // The release must not wait for a later MIOpen call: a caller that finds,
    // captures, and from then on only replays the graph makes none (issue
    // #12121). Nothing below touches the tracker or the handle after abandon().
    auto slot    = tracker.acquire(handle);
    slot.scratch = FreshScratch(handle);
    ASSERT_NE(slot.scratch, nullptr);
    const std::weak_ptr<miopen::ScratchAllocation> scratch = slot.scratch;
    StreamGate gate;
    ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, &gate), hipSuccess);
    tracker.abandon(std::move(slot));

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(scratch.expired()) << "released while its kernel was still running";

    gate.open();
    EXPECT_TRUE(ReleasedWithin(scratch)) << "scratch not released after its kernel retired";
}

TEST_F(GPU_StreamTracker_FP32, ReleaseDuringAnUnrelatedGlobalCaptureLeavesItValid)
{
    // Regression for issue #12121. The release runs on the tracker's worker
    // whenever the kernel retires, so it can land inside anyone's capture. Under
    // the default global mode HIP refuses the deallocator's hipFree from any
    // thread while any capture is open, and the refusal invalidates that capture.
    auto slot    = tracker.acquire(handle);
    slot.scratch = FreshScratch(handle);
    ASSERT_NE(slot.scratch, nullptr);
    const std::weak_ptr<miopen::ScratchAllocation> scratch = slot.scratch;
    StreamGate gate;
    ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, &gate), hipSuccess);
    tracker.abandon(std::move(slot));

    // Allocate before the capture opens: hipMalloc is itself illegal during one.
    void* dev_ptr = nullptr;
    ASSERT_EQ(hipMalloc(&dev_ptr, 64), hipSuccess);
    hipStream_t capture_stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&capture_stream, hipStreamNonBlocking), hipSuccess);

    ASSERT_EQ(hipStreamBeginCapture(capture_stream, hipStreamCaptureModeGlobal), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(dev_ptr, 0, 64, capture_stream), hipSuccess);

    gate.open();
    const bool released = ReleasedWithin(scratch);

    hipGraph_t graph      = nullptr;
    const auto end_status = hipStreamEndCapture(capture_stream, &graph);

    EXPECT_TRUE(released) << "scratch not released during the capture";
    EXPECT_EQ(end_status, hipSuccess) << "the release invalidated an unrelated capture";
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
    const auto cap = tracker.MaxDraining();

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
    // A slot is not reclaimed before its completion event signals, and comes
    // back with the spent event cleared.
    auto slot = tracker.acquire(handle);

    StreamGate gate;
    ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, &gate), hipSuccess);

    auto done = miopen::make_hip_event();
    ASSERT_EQ(hipEventRecord(done.get(), slot.stream), hipSuccess);

    const auto gated_stream = slot.stream;
    slot.done               = miopen::StreamTracker::EventPtr{std::move(done)};
    tracker.abandon(std::move(slot));

    // Event is queued behind the gate, so nothing may be reclaimed yet.
    auto other = tracker.acquire(handle);
    EXPECT_NE(other.stream, gated_stream);
    tracker.release(other);

    gate.open();
    ASSERT_EQ(hipStreamSynchronize(gated_stream), hipSuccess);

    auto reclaimed = tracker.acquire(handle);
    EXPECT_EQ(reclaimed.stream, gated_stream);
    EXPECT_EQ(reclaimed.done, nullptr) << "reclaim left the spent event on the slot";
    tracker.release(reclaimed);
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

TEST(CPU_StreamTracker_NONE, ZeroCapIsRaisedToOne)
{
    // A zero cap would leave acquire() waiting on an empty backlog forever.
    EXPECT_EQ(miopen::StreamTracker{0}.MaxDraining(), 1u);
}
