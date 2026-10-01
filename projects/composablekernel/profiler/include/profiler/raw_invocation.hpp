// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/host_utility/hip_check_error.hpp"
#include "ck/stream_config.hpp"

namespace ck {
namespace profiler {

inline constexpr int kRawInvocationRepeats = 50;

// The events bracket one complete, untimed invocation (including any clears,
// packing, or conversion that the invoker enqueues). The workspace and inputs
// must already have been prepared by the caller. No cache flushing or correction
// is applied; each repeat has its own start/stop interval on the same stream.
template <typename Invoker, typename Argument>
float measure_raw_invocation(Invoker& invoker,
                             Argument* argument,
                             hipStream_t stream = nullptr,
                             int repeats        = kRawInvocationRepeats)
{
    const StreamConfig config{stream, false, 0, 0, 1, false};

    // Compile and warm all of the invocation's kernels before measuring. Drain
    // the stream so that the first interval does not include preceding work.
    invoker.Run(argument, config);
    hip_check_error(hipStreamSynchronize(stream));

    struct Events
    {
        hipEvent_t start;
        hipEvent_t stop;

        Events()
        {
            hip_check_error(hipEventCreate(&start));
            try
            {
                hip_check_error(hipEventCreate(&stop));
            }
            catch(...)
            {
                static_cast<void>(hipEventDestroy(start));
                throw;
            }
        }

        ~Events()
        {
            static_cast<void>(hipEventDestroy(stop));
            static_cast<void>(hipEventDestroy(start));
        }
    } events;

    float total_ms = 0;
    for(int i = 0; i < repeats; ++i)
    {
        hip_check_error(hipEventRecord(events.start, stream));
        invoker.Run(argument, config);
        hip_check_error(hipEventRecord(events.stop, stream));
        hip_check_error(hipEventSynchronize(events.stop));

        float elapsed_ms = 0;
        hip_check_error(hipEventElapsedTime(&elapsed_ms, events.start, events.stop));
        total_ms += elapsed_ms;
    }
    return total_ms / repeats;
}

} // namespace profiler
} // namespace ck
