// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_STREAM_TRACKER_HPP_
#define GUARD_MIOPEN_STREAM_TRACKER_HPP_

#include <miopen/config.hpp>
#include <miopen/allocator.hpp>

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>
#include <hip/hip_runtime_api.h>

namespace miopen {

struct Handle;

struct ScratchAllocation
{
    Allocator::ManageDataPtr buffer;
    std::size_t size = 0;
};

/// Pool of streams dedicated to speculative kernel evaluation. The streams are
/// created and owned here rather than taken from the Handle's index-addressed
/// stream pool, whose ids MHA and RNN claim by hardcoded number: a slot handed
/// out here must not be one another solver is also writing to.
///
/// An abandoned slot is retired by the runtime, not by a later MIOpen call:
/// abandon() queues a host callback behind the slot's work, and when the
/// stream reaches it a worker thread owned by this tracker returns the slot to
/// the pool and drops its scratch. Release therefore follows the kernel by
/// microseconds, whatever the caller does next - including nothing but graph
/// replay (issue #12121).
struct MIOPEN_INTERNALS_EXPORT StreamTracker
{
    using StreamPtr = std::shared_ptr<std::remove_pointer_t<hipStream_t>>;
    using EventPtr  = std::shared_ptr<std::remove_pointer_t<hipEvent_t>>;

    struct Slot
    {
        hipStream_t stream = nullptr;
        std::shared_ptr<ScratchAllocation> scratch;

        /// Completion of the evaluation's last kernel, recorded on `stream`.
        /// Set this before abandoning a slot: acquire() tests it to reclaim a
        /// slot without waiting for the worker, and it is the only way to see a
        /// finished slot whose callback is late or lost - polling `stream` reads
        /// busy until that callback has returned. Optional, but never hand over
        /// an event that was not recorded: HIP reports it as already complete.
        EventPtr done;

        /// Set by abandon(): names this abandonment to its completion callback,
        /// so a callback arriving late cannot retire a later reuse of the
        /// same stream.
        std::uint64_t ticket = 0;
    };

    /// Backlog cap from DefaultMaxDraining().
    StreamTracker() : StreamTracker(DefaultMaxDraining()) {}

    /// Backlog cap of `max_draining`, raised to 1 if zero: a cap of zero would
    /// leave acquire() waiting on an empty backlog.
    explicit StreamTracker(std::size_t max_draining)
        : max_draining_(max_draining > 0 ? max_draining : 1)
    {
    }

    /// Blocks until every owned stream has drained, then stops the worker.
    /// Kernels left running by a timed-out evaluation execute from code objects
    /// the Handle unloads during teardown, so they must finish before this
    /// tracker's owner goes away.
    ~StreamTracker();

    StreamTracker(const StreamTracker&)            = delete;
    StreamTracker& operator=(const StreamTracker&) = delete;

    /// Reclaims what has already retired, then hands out a slot, always a
    /// usable one.
    ///
    /// Waits if no slot is free and the backlog is already at MaxDraining(),
    /// rather than declining. Declining would mean skipping a solver evaluation,
    /// and a skipped naive solver can lose a find it would have won - the outcome
    /// MIOPEN_NAIVE_TIMEOUT exists to avoid. Find results reach the user find-db,
    /// so a transient backlog would persist as a worse recorded solver choice. A
    /// stall is recoverable; that is not.
    ///
    /// Throws instead of waiting once the runtime reports a device error, since
    /// HIP never clears one and the backlog could never drain; see SweepLocked.
    Slot acquire(const Handle& handle);

    /// Hands back a slot whose work has finished; abandon() is for work that may
    /// not have. The scratch is dropped on the calling thread, outside the
    /// tracker's lock, and the done event cleared, so the slot returns to the
    /// pool clean.
    void release(Slot slot);

    /// Hands the slot back while its work may still be running, and queues its
    /// retirement behind that work. Set slot.done first where one is available;
    /// see Slot::done.
    void abandon(Slot slot);

    /// How many abandoned streams may be outstanding before acquire() blocks.
    std::size_t MaxDraining() const { return max_draining_; }

    /// The cap a default-constructed tracker uses: MaxDrainingFor() applied to
    /// the process's GPU_MAX_HW_QUEUES, read once.
    static std::size_t DefaultMaxDraining();

    /// The backlog cap for a GPU_MAX_HW_QUEUES value, null meaning unset.
    /// GPU_MAX_HW_QUEUES is the variable ROCclr uses to decide how many hardware
    /// rings streams get before they start sharing. A slot sharing a ring with a
    /// runaway kernel from an earlier abandonment times that contention rather
    /// than the solver, and that number is what reaches the find-db, so the
    /// backlog is bounded by the runtime's own budget: the rings less the
    /// handle's stream and the slot being handed out, and at least 1. An unset,
    /// non-positive, or unparseable value counts as ROCclr's default of 4.
    static std::size_t MaxDrainingFor(const char* gpu_max_hw_queues);

private:
    struct RetireContext;

    /// hipLaunchHostFunc callback queued by abandon(). Runs on the runtime's
    /// callback thread, where no HIP call is allowed - a deallocator's hipFree
    /// there would wait on the very stream running the callback - so it only
    /// hands the ticket to the worker.
    static void OnSlotRetired(void* context);

    /// Worker body: retires the slots named by callbacks and frees all retired
    /// scratch, its own and SweepLocked's, outside mutex_.
    void RetireLoop();

    /// Callers already holding mutex_. Retires only slots whose query succeeds,
    /// and hands their scratch to the worker rather than freeing it. Stops at and
    /// returns the first answer that is neither success nor "not ready" - the
    /// runtime's sticky device error - and returns hipSuccess otherwise.
    ///
    /// The backstop for a callback that is late or never comes. The worker
    /// normally gets there first, but a query does not depend on the callback, so
    /// this is what guarantees a finished slot is reclaimed whatever happens to
    /// it; without it a lost callback would leave acquire() waiting at the cap
    /// forever. That is also why slots carry tickets: a slot reclaimed here may
    /// still have its callback to come. Only slots with a done event are
    /// covered - a stream query reads busy until the callback queued on that
    /// stream has returned - which is why every abandonment in find sets one.
    hipError_t SweepLocked();

    /// Guards the slot containers and next_ticket_. acquire()/release()/abandon()
    /// run during find while the worker retires slots, so two threads can reach
    /// these at once.
    std::mutex mutex_;

    /// Signalled by the worker whenever it returns a slot to available_, for an
    /// acquire() waiting on a full backlog.
    std::condition_variable retired_cv_;

    const std::size_t max_draining_;
    std::uint64_t next_ticket_ = 0;

    /// Declared before the slots so it is destroyed last: the slots below borrow
    /// these streams, and the destructor synchronizes on them before they go away.
    std::vector<StreamPtr> owned_streams_;
    std::vector<Slot> available_;
    std::vector<Slot> draining_;

    /// Hand-off to the worker, from the runtime's callback thread and from
    /// SweepLocked. Kept apart from mutex_ so the callback thread never waits
    /// behind a find holding the slot containers. Taken after mutex_ where both
    /// are held, never before it.
    std::mutex retire_mutex_;
    std::condition_variable retire_cv_;
    std::vector<std::uint64_t> retired_tickets_;
    std::vector<std::shared_ptr<ScratchAllocation>> deferred_scratch_;
    bool stop_worker_ = false;

    /// Started by the first abandon(), joined by the destructor.
    std::thread worker_;
};

} // namespace miopen

#endif // GUARD_MIOPEN_STREAM_TRACKER_HPP_
