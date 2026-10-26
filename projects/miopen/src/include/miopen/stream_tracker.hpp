// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_STREAM_TRACKER_HPP_
#define GUARD_MIOPEN_STREAM_TRACKER_HPP_

#include <miopen/config.hpp>
#include <miopen/allocator.hpp>

#include <atomic>
#include <memory>
#include <mutex>
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
struct MIOPEN_INTERNALS_EXPORT StreamTracker
{
    using StreamPtr = std::shared_ptr<std::remove_pointer_t<hipStream_t>>;
    using EventPtr  = std::shared_ptr<std::remove_pointer_t<hipEvent_t>>;

    struct Slot
    {
        hipStream_t stream = nullptr;
        std::shared_ptr<ScratchAllocation> scratch;

        /// Completion of the evaluation's last kernel, recorded on `stream`.
        /// Set this before abandoning a slot: it is the precise point at which
        /// the stream is reusable, and cheaper to test than the stream itself.
        /// Optional - a slot abandoned without one falls back to polling
        /// `stream`, which is correct but coarser.
        EventPtr done;
    };

    /// Backlog cap from DefaultMaxDraining().
    StreamTracker() : StreamTracker(DefaultMaxDraining()) {}

    /// Backlog cap of `max_draining`, raised to 1 if zero: a cap of zero would
    /// leave acquire() waiting on an empty backlog.
    explicit StreamTracker(std::size_t max_draining)
        : max_draining_(max_draining > 0 ? max_draining : 1)
    {
    }

    /// Blocks until every abandoned stream has drained. Kernels left running by
    /// a timed-out evaluation execute from code objects the Handle unloads during
    /// teardown, so they must finish before this tracker's owner goes away.
    ~StreamTracker();

    StreamTracker(const StreamTracker&)            = delete;
    StreamTracker& operator=(const StreamTracker&) = delete;

    /// Reclaims what it can, then hands out a slot, always a usable one.
    ///
    /// Polls if no slot is free and the backlog is already at MaxDraining(),
    /// waiting for an abandoned stream to retire rather than declining. Declining
    /// would mean skipping a solver evaluation, and a skipped naive solver can
    /// lose a find it would have won - the outcome MIOPEN_NAIVE_TIMEOUT exists to
    /// avoid. Find results reach the user find-db, so a transient backlog would
    /// persist as a worse recorded solver choice. A stall is recoverable; that is
    /// not.
    ///
    /// Throws instead of waiting once the runtime reports a device error, since
    /// HIP never clears one and the backlog could never drain; see SweepLocked.
    Slot acquire(const Handle& handle);

    /// Reclaims abandoned slots whose stream has gone idle, dropping the scratch
    /// references they hold. Non-blocking: a slot whose stream is still busy is
    /// left in place for a later sweep, as is one whose query returns an error:
    /// only success proves the kernel stopped writing the scratch.
    ///
    /// Unconditional, so call it only where a HIP graph capture cannot be in
    /// flight - find and the allocation path qualify by construction. Anywhere a
    /// capture is possible, go through SweepUnlessCapturing instead: reclaiming
    /// both polls streams and ends in a deallocator call, and neither belongs in
    /// the middle of someone's recording (issue #12121).
    void sweep();

    /// True while an abandoned slot still awaits reclaim with a scratch buffer.
    ///
    /// The kernel launch path tests this before doing anything else, so a build
    /// with nothing abandoned - which is nearly all of them - pays one uncontended
    /// atomic load per launch and no lock. It goes false again as soon as the
    /// buffers are released, even though the slots themselves live on, because an
    /// idle stream sitting in the pool costs nothing worth hurrying over.
    bool HasPinnedScratch() const { return pinned_scratch_.load(std::memory_order_relaxed) != 0; }

    /// Reclaims, unless `stream` is mid-capture.
    ///
    /// Reclaiming ends in a deallocator call, which is not something to run while
    /// a caller is recording a graph - under PyTorch that lands in a caching
    /// allocator doing its own capture-time bookkeeping. Skipping leaves the
    /// buffer pinned until the next launch that is not being captured.
    ///
    /// So the release guarantee is "at the next uncaptured launch, allocation, or
    /// find on the handle", not "as soon as the kernel retires". Graph replay does
    /// not go through Handle::Run, so a workload that finds, captures, and from
    /// then on only replays keeps the scratch until the handle is destroyed.
    void SweepUnlessCapturing(hipStream_t stream);

    void release(Slot slot);

    /// Set slot.done first where one is available; see Slot::done.
    void abandon(Slot slot);

    /// How many abandoned streams may be outstanding before acquire() blocks.
    std::size_t MaxDraining() const { return max_draining_; }

    /// The cap a default-constructed tracker uses, derived from GPU_MAX_HW_QUEUES,
    /// the same variable ROCclr uses to decide how many hardware rings streams
    /// get before they start sharing. A slot sharing a ring with a runaway kernel
    /// from an earlier abandonment times that contention rather than the solver,
    /// and that number is what reaches the find-db, so the backlog is bounded by
    /// the runtime's own budget.
    static std::size_t DefaultMaxDraining();

private:
    /// Callers already holding mutex_. Retires only slots whose query succeeds.
    /// Returns the first answer that is neither success nor "not ready" - the
    /// runtime's sticky device error - or hipSuccess.
    hipError_t SweepLocked();

    /// Guards every container below. acquire()/release()/abandon() run during
    /// find while sweep() also runs from the allocation and kernel launch paths,
    /// so two threads sharing a handle can reach these concurrently.
    std::mutex mutex_;

    const std::size_t max_draining_;

    /// Abandoned slots still holding a buffer. Readable without the lock so the
    /// launch path can opt out cheaply; mutated only under it.
    std::atomic<std::size_t> pinned_scratch_{0};

    /// Declared before the slots so it is destroyed last: the slots below borrow
    /// these streams, and the destructor synchronizes on them before they go away.
    std::vector<StreamPtr> owned_streams_;
    std::vector<Slot> available_;
    std::vector<Slot> draining_;
};

} // namespace miopen

#endif // GUARD_MIOPEN_STREAM_TRACKER_HPP_
