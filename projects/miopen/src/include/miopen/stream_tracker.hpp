// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_STREAM_TRACKER_HPP_
#define GUARD_MIOPEN_STREAM_TRACKER_HPP_

#include <miopen/config.h>

#include <vector>
#include <hip/hip_runtime_api.h>

namespace miopen {

struct Handle;

struct MIOPEN_EXPORT StreamTracker
{
    struct Slot
    {
        int pool_id;
        hipStream_t stream;
    };

    Slot acquire(const Handle& handle);
    void release(Slot slot) { available_.push_back(slot); }
    void abandon(Slot slot) { draining_.push_back(slot); }

private:
    std::vector<Slot> available_;
    std::vector<Slot> draining_;
    int next_id_ = 1;
};

} // namespace miopen

#endif // GUARD_MIOPEN_STREAM_TRACKER_HPP_
