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

    // Stream-K needs one flag per workgroup; GSU reduction needs a much larger
    // region. Reuse the existing input pointer: adding a ContractionInputs field
    // would change a layout also exported by other libraries embedding TensileLite.
    // Direct dispatch already has both regions. Inputs may outlive the solution
    // that selected them, so the caller must assign the result on every solve.
    template <typename Solution>
    void* synchronizerForSolution(const Solution& solution, void* streamKRegion, void* gsuRegion)
    {
        if(readsStreamKFlags(solution) && streamKRegion != nullptr)
            return streamKRegion;
        return gsuRegion;
    }

    // The object API learns its stream at initialize(). Bind one input or a
    // contiguous group before solve() embeds these pointers in kernel arguments.
    // Handle is a template so tests can provide storage without GPU allocation.
    template <typename Handle, typename Solution, typename Inputs>
    rocblaslt_status bindSynchronizers(Handle&         handle,
                                       const Solution& solution,
                                       hipStream_t     stream,
                                       Inputs*         inputs,
                                       size_t          count = 1)
    {
        const bool readsFlags = readsStreamKFlags(solution);
        // Reject before changing bindings: problems in one launch must never
        // wrap onto another problem's flags.
        if(readsFlags && count > Handle::c_syncSkSlotsPerStream)
            return rocblaslt_status_invalid_value;

        for(size_t i = 0; i < count; ++i)
        {
            void* region = nullptr;
            if(readsFlags)
            {
                auto status = handle.streamKFlagsForStream(stream, i, &region);
                if(status != rocblaslt_status_success)
                    return status;
            }
            // Always replace the old binding, including when Stream-K storage
            // is unavailable or the newly selected solution does not read flags.
            if(region == nullptr)
                region = handle.gsuFlagsForProblem(i);
            inputs[i].Synchronizer = region;
        }
        return rocblaslt_status_success;
    }
}
