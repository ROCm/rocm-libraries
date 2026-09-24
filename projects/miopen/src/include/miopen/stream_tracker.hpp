// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_STREAM_TRACKER_HPP_
#define GUARD_MIOPEN_STREAM_TRACKER_HPP_

#include <miopen/config.hpp>
#include <miopen/allocator.hpp>

#include <memory>
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

    struct Slot
    {
        hipStream_t stream = nullptr;
        std::shared_ptr<ScratchAllocation> scratch;
    };

    StreamTracker() = default;

    /// Blocks until every abandoned stream has drained. Kernels left running by
    /// a timed-out evaluation execute from code objects the Handle unloads during
    /// teardown, so they must finish before this tracker's owner goes away.
    ~StreamTracker();

    StreamTracker(const StreamTracker&)            = delete;
    StreamTracker& operator=(const StreamTracker&) = delete;

    Slot acquire(const Handle& handle);

    /// Reclaims abandoned slots whose stream has gone idle, dropping the scratch
    /// references they hold. Non-blocking: a slot whose stream is still busy is
    /// left in place for a later sweep.
    void sweep();

    void release(Slot slot)
    {
        slot.scratch.reset();
        available_.push_back(std::move(slot));
    }

    void abandon(Slot slot) { draining_.push_back(std::move(slot)); }

private:
    /// Declared first so it is destroyed last: the slots below borrow these
    /// streams, and the destructor synchronizes on them before they go away.
    std::vector<StreamPtr> owned_streams_;
    std::vector<Slot> available_;
    std::vector<Slot> draining_;
};

} // namespace miopen

#endif // GUARD_MIOPEN_STREAM_TRACKER_HPP_
