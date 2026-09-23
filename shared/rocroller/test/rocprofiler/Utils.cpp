// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Utils.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sstream>

namespace rocRoller
{
    std::string
        formatLatencyComparison(const std::vector<Instruction>& filteredInstructions,
                                const std::vector<std::tuple<std::string, size_t>>& latencies)
    {
        std::stringstream infoMessage;
        for(size_t i = 0; i < std::min(filteredInstructions.size(), latencies.size()); ++i)
        {
            const auto& inst                 = filteredInstructions[i];
            const auto& [inst_name, latency] = latencies[i];

            int const modelLatency = inst.totalCycles() * 4;
            int const delta        = static_cast<int>(latency) - modelLatency;

            infoMessage << fmt::format("{} {}, model {}, profiler {}, delta {}\n",
                                       delta != 0 ? "*" : " ",
                                       inst_name,
                                       modelLatency,
                                       latency,
                                       delta);
        }
        return infoMessage.str();
    }

    std::vector<Instruction>
        filterAndVerifyInstructions(const std::vector<Instruction>&                  instructions,
                                    const std::vector<profiler::InstructionProfile>& latencies)
    {
        std::vector<Instruction> filteredInstructions;
        for(const auto& inst : instructions)
        {
            if(not inst.toString(LogLevel::Terse).empty())
                filteredInstructions.push_back(inst);
        }

        std::stringstream deltas;
        for(size_t i = 0; i < std::max(filteredInstructions.size(), latencies.size()); ++i)
        {
            const auto& inst
                = i < filteredInstructions.size() ? filteredInstructions[i] : Instruction();
            const auto& profile
                = i < latencies.size() ? latencies[i] : profiler::InstructionProfile();

            deltas << fmt::format(
                "{}: filtered {}, profiler {}\n", i, inst.getOpCode(), profile.instruction);
        }
        INFO(deltas.str());

        REQUIRE(filteredInstructions.size() == latencies.size());

        return filteredInstructions;
    }

} // namespace rocRoller
