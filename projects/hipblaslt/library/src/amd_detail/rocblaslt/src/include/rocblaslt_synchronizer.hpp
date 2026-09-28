// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../../include/rocblaslt-types.h"

namespace rocblaslt
{
    template <typename Solution>
    bool readsStreamKFlags(const Solution& solution)
    {
        // Amax uses Synchronizer for its counter, so retain its GSU region.
        return solution.sizeMapping.streamK > 0 && solution.sizeMapping.streamKAtomic == 0
               && !solution.problemType.outputAmaxD;
    }

    template <typename Inputs>
    void bindSynchronizer(Inputs& inputs, void* streamKRegion, void* gsuRegion)
    {
        // Inputs survive initialize() calls. Always overwrite a previous binding.
        inputs.Synchronizer = streamKRegion != nullptr ? streamKRegion : gsuRegion;
    }

    // Handle is a template so tests can supply storage without allocating GPU
    // memory. Production callers use _rocblaslt_handle and TensileLite inputs.
    template <typename Handle, typename Inputs>
    rocblaslt_status bindSynchronizerForStream(
        Handle& handle, bool readsFlags, hipStream_t stream, size_t index, Inputs& inputs)
    {
        void* region = nullptr;
        if(readsFlags)
        {
            auto status = handle.streamKFlagsForStream(stream, index, &region);
            if(status != rocblaslt_status_success)
                return status;
        }
        bindSynchronizer(inputs, region, handle.gsuFlagsForProblem(index));
        return rocblaslt_status_success;
    }

    template <typename Handle, typename Inputs>
    rocblaslt_status bindGroupedSynchronizers(Handle&     handle,
                                              bool        readsFlags,
                                              hipStream_t stream,
                                              Inputs&     inputs)
    {
        // All problems in the launch may be active together: never wrap a
        // problem onto another problem's flags. Reject before changing bindings.
        if(readsFlags && inputs.size() > Handle::c_syncSkSlotsPerStream)
            return rocblaslt_status_invalid_value;
        for(size_t i = 0; i < inputs.size(); ++i)
        {
            auto status = bindSynchronizerForStream(handle, readsFlags, stream, i, inputs[i]);
            if(status != rocblaslt_status_success)
                return status;
        }
        return rocblaslt_status_success;
    }
}
