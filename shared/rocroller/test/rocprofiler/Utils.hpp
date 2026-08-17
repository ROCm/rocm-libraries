// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Agent.hpp"
#include <rocRoller/CodeGen/Instruction.hpp>

#include <string>
#include <tuple>
#include <vector>

namespace rocRoller
{
    /**
     * @brief Number of runs for profiling tests
     *
     * Should be odd, as median is used for statistical analysis
     */
    constexpr int NUM_RUNS = 5;

    /**
     * @brief Formats a comparison between model predictions and profiler measurements
     *
     * @param filteredInstructions The list of instructions to compare
     * @param latencies The median latencies for each instruction
     * @return Formatted string comparing model predictions with profiler measurements
     */
    std::string
        formatLatencyComparison(const std::vector<Instruction>& filteredInstructions,
                                const std::vector<std::tuple<std::string, size_t>>& latencies);

    /**
     * @brief Filters instructions to exclude comments and verifies alignment with profiler data
     *
     * @param instructions Raw instructions from the kernel
     * @param latencies Profiler latencies to verify against
     * @return Vector of filtered instructions (non-comment only)
     */
    std::vector<Instruction>
        filterAndVerifyInstructions(const std::vector<Instruction>&                  instructions,
                                    const std::vector<profiler::InstructionProfile>& latencies);

} // namespace rocRoller
