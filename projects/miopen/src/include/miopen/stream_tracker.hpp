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

    StreamTracker() = default;

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
    Slot acquire(const Handle& handle);

    /// Reclaims abandoned slots whose stream has gone idle, dropping the scratch
    /// references they hold. Non-blocking: a slot whose stream is still busy is
    /// left in place for a later sweep.
    ///
    /// Unconditional, so call it only where a HIP graph capture cannot be in
    /// flight - find and the allocation path qualify by construction. Anywhere a
    /// capture is possible, go through SweepUnlessCapturing instead: reclaiming
    /// both polls streams and ends in a deallocator call, and neither belongs in
    /// the middle of someone's recording (issue #12121).
    void sweep();

    /// True while an abandoned slot is still holding a scratch buffer alive.
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
    void SweepUnlessCapturing(hipStream_t stream);

    void release(Slot slot);

    /// Set slot.done first where one is available; see Slot::done.
    void abandon(Slot slot);

    /// How many abandoned streams may be outstanding before acquire() blocks.
    ///
    /// Derived from GPU_MAX_HW_QUEUES, the same variable ROCclr uses to decide
    /// how many hardware rings streams get before they start sharing. A slot
    /// sharing a ring with a runaway kernel from an earlier abandonment times
    /// that contention rather than the solver, and that number is what reaches
    /// the find-db, so the backlog is bounded by the runtime's own budget.
    /// MIOPEN_NAIVE_MAX_DRAINING overrides it outright.
    static std::size_t MaxDraining();

private:
    /// Callers already holding mutex_.
    void SweepLocked();

    /// Guards every container below. acquire()/release()/abandon() run during
    /// find while sweep() also runs from the allocation and kernel launch paths,
    /// so two threads sharing a handle can reach these concurrently.
    std::mutex mutex_;

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
