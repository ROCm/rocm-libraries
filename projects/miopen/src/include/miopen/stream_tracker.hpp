// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_STREAM_TRACKER_HPP_
#define GUARD_MIOPEN_STREAM_TRACKER_HPP_

#include <miopen/config.h>
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

struct MIOPEN_EXPORT StreamTracker
{
    struct Slot
    {
        int pool_id;
        hipStream_t stream;
        std::shared_ptr<ScratchAllocation> scratch;
    };

    Slot acquire(const Handle& handle);

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
