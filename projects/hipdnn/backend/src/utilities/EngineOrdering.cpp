// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <numeric>

#include "hipdnn_data_sdk/utilities/EngineNames.hpp"
#include "utilities/EngineOrdering.hpp"

namespace hipdnn_backend
{
namespace utilities
{

void sortEngineIds(std::vector<int64_t>& engineIds)
{
    // Sort engine IDs: MIOPEN_ENGINE first, MIOPEN_ENGINE_DETERMINISTIC last, others in middle
    // Using index-based sorting with std::sort to achieve stable sort behavior

    std::vector<size_t> indices(engineIds.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&engineIds](size_t i, size_t j) {
        int64_t a = engineIds[i];
        int64_t b = engineIds[j];

        bool aIsMiopen = (a == hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID);
        bool bIsMiopen = (b == hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID);
        bool aIsMiopenDet = (a == hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID);
        bool bIsMiopenDet = (b == hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID);

        // MIOPEN_ENGINE always comes before everything
        // Logic to check for dupes isnt really need in the backend but its here just in case
        // that changes in the future.
        if(aIsMiopen != bIsMiopen)
        {
            return aIsMiopen;
        }
        if(aIsMiopen)
        {
            return i < j; // Preserve original order among duplicates
        }

        // MIOPEN_ENGINE_DETERMINISTIC always comes after everything
        if(aIsMiopenDet != bIsMiopenDet)
        {
            return !aIsMiopenDet;
        }

        // For other engines, preserve original order (using index as tie-breaker for stability)
        return i < j;
    });

    // Reorder engineIds based on sorted indices
    std::vector<int64_t> sorted;
    sorted.reserve(engineIds.size());
    for(size_t idx : indices)
    {
        sorted.push_back(engineIds[idx]);
    }
    engineIds = std::move(sorted);
}

} // namespace utilities
} // namespace hipdnn_backend
