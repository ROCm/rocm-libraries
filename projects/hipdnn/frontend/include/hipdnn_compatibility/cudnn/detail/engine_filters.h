// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file engine_filters.h
 * @brief Per-engine predicates behind the cuDNN shim's note filters.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

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

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
