// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

namespace hipdnn_data_sdk::utilities
{

/**
 * @brief Sorts engine IDs with MIOpen-specific ordering requirements.
 *
 * Ordering rationale:
 * - MIOPEN_ENGINE first: Default engine with full operation support
 * - Other engines middle: Stable order preserved for predictability
 * - MIOPEN_ENGINE_DETERMINISTIC last: Limited to conv operations only,
 *   deprioritized due to performance trade-offs and reduced operation support
 *
 * This is a header-only implementation shared between backend and heuristic plugins.
 *
 * @param engineIds Vector of engine IDs to sort (modified in-place)
 */
inline void sortEngineIds(std::vector<int64_t>& engineIds)
{
    // Sort engine IDs: MIOPEN_ENGINE first, MIOPEN_ENGINE_DETERMINISTIC last, others in middle
    // Using index-based sorting with std::sort to achieve stable sort behavior

    std::vector<size_t> indices(engineIds.size());
    std::iota(indices.begin(), indices.end(), 0);

    auto getPriority = [](int64_t engineId) -> int {
        if(engineId == hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID)
        {
            return 0;
        }
        if(engineId == hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID)
        {
            return 2;
        }
        return 1; // Other engines
    };

    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        const int priA = getPriority(engineIds[i]);
        const int priB = getPriority(engineIds[j]);
        return (priA != priB) ? (priA < priB) : (i < j);
    });

    // Reorder engineIds based on sorted indices
    std::vector<int64_t> sorted;
    sorted.reserve(engineIds.size());
    for(const size_t idx : indices)
    {
        sorted.push_back(engineIds[idx]);
    }
    engineIds = std::move(sorted);
}

} // namespace hipdnn_data_sdk::utilities
