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
#include <deque>
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
    bool entered  = false;
    bool released = false;

    static void callback(void* arg)
    {
        auto* self = static_cast<StreamGate*>(arg);
        std::unique_lock<std::mutex> lk(self->mtx);
        self->entered = true;
        self->cv.notify_all();
        self->cv.wait(lk, [self] { return self->released; });
    }

    /// Returns once the stream has reached the gate, so everything queued
    /// before it on that stream - host callbacks included - has finished.
    void WaitEntered()
    {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [this] { return entered; });
    }

    void open()
    {
        {
            std::lock_guard<std::mutex> lk(mtx);
            released = true;
        }
        cv.notify_all();
    }
};

/// Opens every gate when it goes out of scope. Declare it after any local that
/// frees device memory on destruction: hipFree waits for every stream, a gated
/// one included, so such a local must not go while a gate is still shut - as it
/// would if a failed assertion ended the test early.
struct OpenGatesOnExit
{
    std::deque<StreamGate>& gates;

    ~OpenGatesOnExit()
    {
        for(auto& gate : gates)
            gate.open();
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
    /// A gate the fixture opens at teardown if the test has not: a test that
    /// stops early on a failed assertion must report it, not leave the tracker's
    /// destructor waiting on a stream that never drains. Gates are declared
    /// ahead of the tracker, so each outlives the stream holding it.
    StreamGate& NewGate() { return gates_.emplace_back(); }

    void TearDown() override
    {
        for(auto& gate : gates_)
            gate.open();
    }

    std::deque<StreamGate> gates_;
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

constexpr auto kSlowFree = std::chrono::milliseconds(500);

void* PlainAlloc(void*, std::size_t size)
{
    void* ptr = nullptr;
    return hipMalloc(&ptr, size) == hipSuccess ? ptr : nullptr;
}

/// Stands in for any deallocator that blocks, such as the default one, whose
/// hipFree waits for every stream on the device.
void SlowFree(void*, void* ptr)
{
    std::this_thread::sleep_for(kSlowFree);
    (void)hipFree(ptr);
}

/// A SlowFree that reports when it starts freeing one particular buffer, so a
/// test can act while that free is in progress.
struct WatchedFree
{
    std::mutex m;
    std::condition_variable cv;
    void* target = nullptr;
    bool started = false;

    static void Free(void* context, void* ptr)
    {
        auto* self = static_cast<WatchedFree*>(context);
        {
            std::lock_guard<std::mutex> lk(self->m);
            if(ptr == self->target)
                self->started = true;
        }
        self->cv.notify_all();
        SlowFree(nullptr, ptr);
    }

    bool WaitStarted(std::chrono::seconds limit)
    {
        std::unique_lock<std::mutex> lk(m);
        return cv.wait_for(lk, limit, [this] { return started; });
    }
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

    auto& gate = NewGate();
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
    auto& gate                                             = NewGate();
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
    auto& gate                                             = NewGate();
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

    std::vector<StreamGate*> gates;
    std::vector<hipStream_t> abandoned;

    for(std::size_t i = 0; i < cap; ++i)
    {
        auto slot = tracker.acquire(handle);
        ASSERT_NE(slot.stream, nullptr) << "no stream before reaching the cap, at " << i;
        gates.push_back(&NewGate());
        ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, gates.back()), hipSuccess);
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

    auto& gate = NewGate();
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

TEST_F(GPU_StreamTracker_FP32, ALateCallbackDoesNotRetireALaterReuseOfItsStream)
{
    // acquire() can reclaim a slot on its done event before the worker has
    // handled that slot's callback. If the stream is then reused and abandoned
    // again, the stale callback must not retire the new abandonment - that would
    // free its scratch while its kernel still runs.
    auto scratch_b = FreshScratch(handle);
    ASSERT_NE(scratch_b, nullptr);
    const std::weak_ptr<miopen::ScratchAllocation> b_scratch = scratch_b;
    const OpenGatesOnExit open_gates{gates_}; // before scratch_b is freed, on any exit

    auto a            = tracker.acquire(handle);
    const auto stream = a.stream;
    auto done         = miopen::make_hip_event();
    ASSERT_EQ(hipEventRecord(done.get(), stream), hipSuccess); // a's work is finished...
    auto& hold_a_callback = NewGate();                         // ...but its callback waits here
    ASSERT_EQ(hipLaunchHostFunc(stream, StreamGate::callback, &hold_a_callback), hipSuccess);
    a.done = miopen::StreamTracker::EventPtr{std::move(done)};
    tracker.abandon(std::move(a));

    auto b = tracker.acquire(handle);
    ASSERT_EQ(b.stream, stream) << "acquire() did not reclaim on the done event";
    auto& b_running = NewGate();
    ASSERT_EQ(hipLaunchHostFunc(stream, StreamGate::callback, &b_running), hipSuccess);
    b.scratch = std::move(scratch_b);
    tracker.abandon(std::move(b));

    hold_a_callback.open();
    b_running.WaitEntered(); // a's callback has returned, by stream order
    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // and the worker has seen it
    EXPECT_FALSE(b_scratch.expired())
        << "a late callback retired a later abandonment of the same stream";

    b_running.open();
    EXPECT_TRUE(ReleasedWithin(b_scratch)) << "the later abandonment was never retired";
}

TEST_F(GPU_StreamTracker_FP32, DestroyingTheTrackerWaitsForAbandonedWorkAndReleasesIt)
{
    // Teardown must outlast the abandoned kernel - it runs from code objects the
    // Handle unloads next - let every callback run, and stop the worker with all
    // the scratch released.
    auto owned = FreshScratch(handle);
    ASSERT_NE(owned, nullptr);
    const std::weak_ptr<miopen::ScratchAllocation> scratch = owned;

    StreamGate gate;
    std::thread opener;
    std::chrono::steady_clock::time_point teardown;
    {
        miopen::StreamTracker local{kTestMaxDraining};
        auto slot = local.acquire(handle);
        ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, &gate), hipSuccess);
        slot.scratch = std::move(owned);
        local.abandon(std::move(slot));
        teardown = std::chrono::steady_clock::now();
        opener   = std::thread([&gate] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            gate.open();
        });
    }
    const std::chrono::duration<double, std::milli> waited =
        std::chrono::steady_clock::now() - teardown;
    opener.join();

    EXPECT_GE(waited.count(), 95.0) << "torn down before its abandoned work drained";
    EXPECT_TRUE(scratch.expired()) << "teardown left the abandoned scratch behind";
}

TEST_F(GPU_StreamTracker_FP32, ConcurrentFindsOnOneTrackerNeitherDeadlockNorLeak)
{
    // Threads running find on one handle drive acquire/abandon/release against
    // the runtime's callback thread and the worker, with acquire() and the worker
    // racing to reclaim the same slots and acquire() waiting on a full backlog.
    constexpr int kThreads    = 4;
    constexpr int kIterations = 1000;

    auto shared = FreshScratch(handle);
    ASSERT_NE(shared, nullptr);
    const std::weak_ptr<miopen::ScratchAllocation> scratch = shared;
    {
        miopen::StreamTracker local{kTestMaxDraining};
        std::vector<std::thread> finds;
        for(int t = 0; t < kThreads; ++t)
        {
            finds.emplace_back([&] {
                for(int i = 0; i < kIterations; ++i)
                {
                    auto slot    = local.acquire(handle);
                    slot.scratch = shared;
                    if(i % 2 == 0)
                    {
                        auto done = miopen::make_hip_event();
                        (void)hipEventRecord(done.get(), slot.stream);
                        slot.done = miopen::StreamTracker::EventPtr{std::move(done)};
                    }
                    local.abandon(std::move(slot));
                    local.release(local.acquire(handle));
                }
            });
        }
        for(auto& find : finds)
            find.join();
    }
    shared.reset();
    EXPECT_TRUE(scratch.expired()) << "a slot kept its scratch past teardown";
}

TEST_F(GPU_StreamTracker_FP32, AcquireLeavesTheDeallocatorToTheWorker)
{
    // acquire() runs inside find. It may reclaim a slot itself, on the slot's done
    // event, but freeing the scratch is the worker's job: a deallocator can block
    // - the default one's hipFree waits for every stream on the device, other
    // abandoned kernels included - and find must not wait on it.
    miopen::Handle slow_handle{};
    slow_handle.SetAllocator(PlainAlloc, SlowFree, nullptr);
    auto& slow_tracker = slow_handle.GetStreamTracker();

    auto slot         = slow_tracker.acquire(slow_handle);
    const auto stream = slot.stream;
    slot.scratch      = slow_handle.GetScratchBuffer(65536);
    ASSERT_NE(slot.scratch, nullptr);
    const std::weak_ptr<miopen::ScratchAllocation> scratch = slot.scratch;
    auto done                                              = miopen::make_hip_event();
    ASSERT_EQ(hipEventRecord(done.get(), stream), hipSuccess); // its work is finished...
    auto& hold_callback = NewGate(); // ...but its callback waits here, so acquire() reclaims it
    ASSERT_EQ(hipLaunchHostFunc(stream, StreamGate::callback, &hold_callback), hipSuccess);
    slot.done = miopen::StreamTracker::EventPtr{std::move(done)};
    slow_tracker.abandon(std::move(slot));

    // Opened from elsewhere: a deallocator run inside acquire() would wait on
    // this very gate - hipFree waits for every stream - and this thread could
    // not open it.
    std::thread opener([&hold_callback] {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        hold_callback.open();
    });

    const auto before = std::chrono::steady_clock::now();
    auto next         = slow_tracker.acquire(slow_handle);
    const std::chrono::duration<double, std::milli> waited =
        std::chrono::steady_clock::now() - before;
    opener.join();

    EXPECT_EQ(next.stream, stream) << "acquire() did not reclaim on the done event";
    EXPECT_LT(waited.count(), kSlowFree.count() / 2.0) << "acquire() ran the deallocator itself";

    EXPECT_TRUE(ReleasedWithin(scratch)) << "the worker never released the scratch";
    slow_tracker.release(std::move(next));
}

TEST_F(GPU_StreamTracker_FP32, AcquireAtTheCapReclaimsFinishedSlotsWhoseCallbacksAreHeld)
{
    // SweepLocked is the backstop for a callback that is late or never comes:
    // with the backlog full, acquire() must take back a slot whose done event
    // says its kernel has finished, without waiting for that slot's callback.
    const auto cap = tracker.MaxDraining();
    std::vector<StreamGate*> kernels;
    std::vector<StreamGate*> callbacks;
    std::vector<hipStream_t> abandoned;
    for(std::size_t i = 0; i < cap; ++i)
    {
        auto slot = tracker.acquire(handle);
        kernels.push_back(&NewGate());
        callbacks.push_back(&NewGate());
        // Stands in for the kernel.
        ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, kernels.back()), hipSuccess);
        auto done = miopen::make_hip_event();
        ASSERT_EQ(hipEventRecord(done.get(), slot.stream), hipSuccess);
        // Holds back the tracker's callback, which abandon() queues after it.
        ASSERT_EQ(hipLaunchHostFunc(slot.stream, StreamGate::callback, callbacks.back()),
                  hipSuccess);
        slot.done = miopen::StreamTracker::EventPtr{std::move(done)};
        abandoned.push_back(slot.stream);
        tracker.abandon(std::move(slot));
    }

    // Finishes the kernels at 100 ms, and releases the callbacks once acquire()
    // has returned - or after 2 s, so a regression fails rather than hangs.
    std::mutex m;
    std::condition_variable cv;
    bool acquired = false;
    std::thread opener([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        for(auto& k : kernels)
            k->open();
        std::unique_lock<std::mutex> lk(m);
        cv.wait_for(lk, std::chrono::seconds(2), [&] { return acquired; });
        for(auto& c : callbacks)
            c->open();
    });

    const auto before = std::chrono::steady_clock::now();
    auto slot         = tracker.acquire(handle);
    const std::chrono::duration<double, std::milli> waited =
        std::chrono::steady_clock::now() - before;
    {
        std::lock_guard<std::mutex> lk(m);
        acquired = true;
    }
    cv.notify_one();
    opener.join();

    EXPECT_LT(waited.count(), 1000.0)
        << "acquire() waited for the held callbacks instead of reading the done events";
    EXPECT_NE(std::find(abandoned.begin(), abandoned.end(), slot.stream), abandoned.end())
        << "grew the pool instead of reclaiming a finished slot";
    tracker.release(slot);
}

TEST_F(GPU_StreamTracker_FP32, SharedScratchSurvivesUntilItsLastHolderLetsGo)
{
    // Evaluations share one scratch buffer to keep the footprint down. Retiring
    // an abandoned holder must only drop its reference: the buffer has to stay
    // allocated while another evaluation may still be writing it.
    auto shared = FreshScratch(handle);
    ASSERT_NE(shared, nullptr);
    const std::weak_ptr<miopen::ScratchAllocation> scratch = shared;

    auto abandoned    = tracker.acquire(handle);
    const auto stream = abandoned.stream;
    auto& gate        = NewGate();
    ASSERT_EQ(hipLaunchHostFunc(stream, StreamGate::callback, &gate), hipSuccess);
    abandoned.scratch = shared;
    tracker.abandon(std::move(abandoned));

    // A later evaluation, still running, on the same buffer.
    auto running    = tracker.acquire(handle);
    running.scratch = shared;
    shared.reset();

    gate.open();
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess); // its callback has run
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while(scratch.use_count() > 1 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_EQ(scratch.use_count(), 1) << "the abandoned holder was never retired";

    const auto survivor = scratch.lock();
    ASSERT_NE(survivor, nullptr) << "retiring one holder freed a buffer another still holds";
    EXPECT_NE(survivor->buffer.get(), nullptr)
        << "retiring one holder freed a buffer another still holds";

    tracker.release(std::move(running));
}

TEST_F(GPU_StreamTracker_FP32, ReleaseReturnsTheSlotToThePoolClean)
{
    // A released slot sits idle in the pool, so it must not pin its buffer; and
    // a done event left on it would read as complete if the stream were later
    // abandoned without a fresh one.
    auto slot         = tracker.acquire(handle);
    const auto stream = slot.stream;
    slot.scratch      = FreshScratch(handle);
    ASSERT_NE(slot.scratch, nullptr);
    const std::weak_ptr<miopen::ScratchAllocation> scratch = slot.scratch;
    auto done                                              = miopen::make_hip_event();
    ASSERT_EQ(hipEventRecord(done.get(), stream), hipSuccess);
    slot.done = miopen::StreamTracker::EventPtr{std::move(done)};

    tracker.release(std::move(slot));
    EXPECT_TRUE(scratch.expired()) << "a released slot kept its scratch";

    auto again = tracker.acquire(handle);
    ASSERT_EQ(again.stream, stream);
    EXPECT_EQ(again.done, nullptr) << "a released slot kept its done event";
    tracker.release(std::move(again));
}

TEST_F(GPU_StreamTracker_FP32, AReusedSlotIsNotReclaimedOnItsPreviousDoneEvent)
{
    // The worker retires a slot and it goes back into the pool. If the old,
    // completed done event stayed on it, the stream's next abandonment would be
    // reclaimed at once while its work still ran.
    auto first        = tracker.acquire(handle);
    const auto stream = first.stream;
    first.scratch     = FreshScratch(handle);
    ASSERT_NE(first.scratch, nullptr);
    const std::weak_ptr<miopen::ScratchAllocation> first_scratch = first.scratch;
    auto done                                                    = miopen::make_hip_event();
    ASSERT_EQ(hipEventRecord(done.get(), stream), hipSuccess);
    first.done = miopen::StreamTracker::EventPtr{std::move(done)};
    tracker.abandon(std::move(first));
    // Its work is already finished, so its callback runs at once. The scratch
    // goes after the slot is back in the pool: this is the worker's retirement.
    ASSERT_TRUE(ReleasedWithin(first_scratch));

    auto second = tracker.acquire(handle);
    ASSERT_EQ(second.stream, stream);
    EXPECT_EQ(second.done, nullptr) << "the worker left the spent done event on the slot";

    auto& gate = NewGate();
    ASSERT_EQ(hipLaunchHostFunc(stream, StreamGate::callback, &gate), hipSuccess);
    tracker.abandon(std::move(second)); // still running, and no fresh done event

    auto third = tracker.acquire(handle);
    EXPECT_NE(third.stream, stream) << "reclaimed a running stream on its previous done event";
    tracker.release(std::move(third));

    gate.open();
}

TEST_F(GPU_StreamTracker_FP32, TheWorkerFreesOutsideTheTrackerLock)
{
    // The worker runs the deallocator, which can block - the default one's
    // hipFree waits for every stream on the device. Holding the tracker lock
    // through it would stall every acquire(), and so find, for as long.
    WatchedFree watch; // outlives the handle, whose teardown may free through it
    miopen::Handle slow_handle{};
    slow_handle.SetAllocator(PlainAlloc, WatchedFree::Free, &watch);
    auto& slow_tracker = slow_handle.GetStreamTracker();

    auto slot    = slow_tracker.acquire(slow_handle);
    slot.scratch = slow_handle.GetScratchBuffer(65536);
    ASSERT_NE(slot.scratch, nullptr);
    {
        std::lock_guard<std::mutex> lk(watch.m);
        watch.target = slot.scratch->buffer.get();
    }
    slow_tracker.abandon(std::move(slot)); // idle, so the worker frees it at once
    ASSERT_TRUE(watch.WaitStarted(std::chrono::seconds(10))) << "the worker never freed it";

    const auto before = std::chrono::steady_clock::now();
    auto next         = slow_tracker.acquire(slow_handle);
    const std::chrono::duration<double, std::milli> waited =
        std::chrono::steady_clock::now() - before;
    slow_tracker.release(std::move(next));

    EXPECT_LT(waited.count(), kSlowFree.count() / 2.0)
        << "acquire() waited for the worker's deallocation";
}

TEST(CPU_StreamTracker_NONE, ZeroCapIsRaisedToOne)
{
    // A zero cap would leave acquire() waiting on an empty backlog forever.
    EXPECT_EQ(miopen::StreamTracker{0}.MaxDraining(), 1u);
}

TEST(CPU_StreamTracker_NONE, BacklogCapFollowsGpuMaxHwQueues)
{
    using miopen::StreamTracker;
    // The hardware rings less the handle's stream and the slot being handed out,
    // and at least 1.
    EXPECT_EQ(StreamTracker::MaxDrainingFor(nullptr), 2u) << "unset is ROCclr's default of 4";
    EXPECT_EQ(StreamTracker::MaxDrainingFor("4"), 2u);
    EXPECT_EQ(StreamTracker::MaxDrainingFor("8"), 6u);
    EXPECT_EQ(StreamTracker::MaxDrainingFor("3"), 1u);
    EXPECT_EQ(StreamTracker::MaxDrainingFor("1"), 1u);

    // A value ROCclr would not take as a queue count falls back to its default -
    // never to a cap so large the backlog is effectively unbounded.
    for(const char* bad : {"0", "-1", "abc", "", "99999999999999999999999"})
        EXPECT_EQ(StreamTracker::MaxDrainingFor(bad), 2u) << "GPU_MAX_HW_QUEUES=\"" << bad << '"';
}
