// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/experimental/thread-trace/dispatch.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace rocRoller
{
    namespace profiler
    {
        struct InstructionProfile
        {
            // Note: many of these latency numbers are architecture-dependent
            // Refer to comments for `rocprofiler_thread_trace_decoder_inst_t` for details

            // Total latency in cycles (includes stall)
            uint64_t totalLatency{0};

            // Total latency including gaps since previous instruction
            uint64_t totalLatencyWithPrecedingNone{0};

            uint64_t    hitcount{0}; // Number of times instruction was executed
            std::string instruction; // Disassembled instruction text

            uint64_t    meanLatency() const;
            uint64_t    meanLatencyWithPrecedingNone() const;
            std::string toString() const;
        };

        std::string toString(std::vector<InstructionProfile> const& profiles);

        using InstructionLatencyMap
            = std::map<rocprofiler_thread_trace_decoder_pc_t, InstructionProfile>;

        /**
         * @brief Call a function that dispatches a single kernel and collects data
         *
         * When the agent is enabled, this function will repeatedly call the dispatch function
         * until instruction profiling data is successfully collected. The dispatch function
         * must only dispatch a single kernel per invocation.
         *
         * @param dispatch Function that dispatches a single kernel
         * @return Vector of InstructionProfile from the dispatch. Returns empty vector if
         *         agent is disabled.
         */
        std::vector<InstructionProfile> loopUntilDispatchData(std::function<void()> dispatch);

        /**
         * @brief Reset the internal state of the profiler agent
         *
         * For unexpected errors causing the agent to be in a bad state.
         */
        void reset();

    } // namespace profiler
} // namespace rocRoller
