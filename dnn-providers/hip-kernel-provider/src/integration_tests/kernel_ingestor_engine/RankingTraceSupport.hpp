// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

/**
 * @file RankingTraceSupport.hpp
 * @brief Reads which ranker ordered an engine's catalog from the plugin's `uhd trace` lines,
 *        and the check a cold case applies to it. No GPU dependency, so a CPU unit test drives
 *        both with synthetic log lines (TestRankingTraceSupport.cpp).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration::detail
{

/// The `decided_by` of the trained ranker's `uhd trace` line for @p engineName ("model", or
/// "declared_order" when it could not rank and fell back), or nullopt when no trained ranker
/// traced for that engine (the native score decided, or there was one candidate).
inline std::optional<std::string>
    rankingDecidedBy(const std::string& engineName,
                     const std::vector<hipdnn_test_sdk::utilities::RecordedLog>& logs)
{
    const std::string traced = "uhd trace: engine '" + engineName + "'";
    const std::string field = " decided_by=";
    for(const auto& log : logs)
    {
        if(log.message.find(traced) == std::string::npos)
        {
            continue;
        }
        const auto at = log.message.find(field);
        if(at != std::string::npos)
        {
            const auto begin = at + field.size();
            return log.message.substr(begin, log.message.find(' ', begin) - begin);
        }
    }
    return std::nullopt;
}

/// A cold ranking that fell back to declared kernel order is a failure, unless the bound
/// ranker was left out of the pack because its `model.bin` was never pulled
/// (@p rankerLeftOutAsUnpulled), the one configuration in which that fallback is expected.
inline void expectRankerDecided(const std::optional<std::string>& decidedBy,
                                bool rankerLeftOutAsUnpulled,
                                const std::string& capturedLogs)
{
    if(decidedBy == std::optional<std::string>("declared_order") && !rankerLeftOutAsUnpulled)
    {
        ADD_FAILURE() << "the bound ranker could not rank this graph and fell back to declared "
                         "kernel order. Captured logs:\n"
                      << capturedLogs;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration::detail
