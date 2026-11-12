// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file engine_filters.h
 * @brief Per-engine predicates and selection diagnostics shared by the cuDNN compatibility layer APIs.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
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

// Set once the OPENSOURCE warning has been logged in this process. Exposed so
// tests can observe the first warning regardless of test order.
inline std::atomic<bool>& openSourceModeWarned()
{
    static std::atomic<bool> s_warned{false};
    return s_warned;
}

// OPENSOURCE has no hipDNN counterpart: both APIs accept it and run fallback
// selection. Every other mode is honored.
inline void warnUnhonoredHeuristicMode(HeurMode_t mode)
{
    if(mode != HeurMode_t::OPENSOURCE
       || openSourceModeWarned().exchange(true, std::memory_order_relaxed))
    {
        return;
    }
    HIPDNN_FE_LOG_WARN("[cudnn_frontend] cuDNN heuristic mode OPENSOURCE is accepted but not "
                       "honored; hipDNN uses fallback selection. Plan choice may differ from "
                       "cuDNN.");
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
