// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file engine_filters.h
 * @brief Per-engine predicates and selection diagnostics shared by the cuDNN shim APIs.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
{

inline bool behaviorNotesMatch(const std::vector<BehaviorNote_t>& selected,
                               const std::vector<BehaviorNote_t>& deselected,
                               const std::vector<BehaviorNote_t>& engineNotes)
{
    const auto hasNote = [&engineNotes](BehaviorNote_t note) {
        return std::find(engineNotes.begin(), engineNotes.end(), note) != engineNotes.end();
    };
    return std::all_of(selected.begin(), selected.end(), hasNote)
           && std::none_of(deselected.begin(), deselected.end(), hasNote);
}

// Positive claims only: hipDNN engines declare no numerical notes, so an engine
// not named here must be treated as possibly non-deterministic.
inline bool isDeterminismClaimed(int64_t engineId)
{
    return engineId == hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID;
}

// cuDNN heuristic modes the shim accepts but cannot honor: hipDNN only has
// fallback selection. The v9 and v8 APIs name different subsets of these.
enum class UnhonoredHeuristicMode : std::uint8_t
{
    A,
    B,
    INSTANT,
    OPENSOURCE,
};

inline const char* unhonoredHeuristicModeName(UnhonoredHeuristicMode mode)
{
    switch(mode)
    {
    case UnhonoredHeuristicMode::A:
        return "A";
    case UnhonoredHeuristicMode::B:
        return "B";
    case UnhonoredHeuristicMode::INSTANT:
        return "INSTANT";
    case UnhonoredHeuristicMode::OPENSOURCE:
    default:
        return "OPENSOURCE";
    }
}

// One bit per UnhonoredHeuristicMode already warned about in this process.
// Exposed so tests can observe the first warning regardless of test order.
inline std::atomic<std::uint32_t>& warnedHeuristicModes()
{
    static std::atomic<std::uint32_t> s_warned{0};
    return s_warned;
}

// Warns on the first use of each mode per process.
inline void warnUnhonoredHeuristicMode(UnhonoredHeuristicMode mode)
{
    const std::uint32_t bit = std::uint32_t{1} << static_cast<std::uint32_t>(mode);
    if((warnedHeuristicModes().fetch_or(bit, std::memory_order_relaxed) & bit) != 0)
    {
        return;
    }
    HIPDNN_FE_LOG_WARN("[cudnn_frontend] cuDNN heuristic mode "
                       << unhonoredHeuristicModeName(mode)
                       << " is accepted but not honored; hipDNN uses fallback selection. Plan "
                          "choice may differ from cuDNN.");
}

// A and B select hipDNN's ModeA/ModeB prediction policies; OPENSOURCE has no hipDNN
// counterpart and runs fallback selection.
inline void warnUnhonoredHeuristicMode(HeurMode_t mode)
{
    if(mode == HeurMode_t::OPENSOURCE)
    {
        warnUnhonoredHeuristicMode(UnhonoredHeuristicMode::OPENSOURCE);
    }
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
