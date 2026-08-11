// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_STREAM_TRACKER_HPP_
#define GUARD_MIOPEN_STREAM_TRACKER_HPP_

#include <miopen/config.hpp>
#include <miopen/allocator.hpp>

#include <memory>
#include <vector>
#include <hip/hip_runtime_api.h>

namespace miopen {

struct Handle;

struct ScratchAllocation
{
    Allocator::ManageDataPtr buffer;
    std::size_t size = 0;
};

struct MIOPEN_INTERNALS_EXPORT StreamTracker
{
    struct Slot
    {
        int pool_id;
        hipStream_t stream;
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
    std::vector<Slot> available_;
    std::vector<Slot> draining_;
    int next_id_ = 1;
};

} // namespace miopen

#endif // GUARD_MIOPEN_STREAM_TRACKER_HPP_
