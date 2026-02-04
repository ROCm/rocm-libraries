// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "utilities/EngineOrdering.hpp"
#include "hipdnn_data_sdk/utilities/EngineNames.hpp"
#include <algorithm>

namespace hipdnn_backend
{
namespace utilities
{

void sortEngineIds(std::vector<int64_t>& engineIds)
{
    // Sort engine IDs: MIOPEN_ENGINE first, MIOPEN_ENGINE_DETERMINISTIC last, others in middle
    std::stable_sort(engineIds.begin(), engineIds.end(), [](int64_t a, int64_t b) {
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
            return false; // They are the same engine so not strictly LT.
        }

        // MIOPEN_ENGINE_DETERMINISTIC always comes after everything
        if(aIsMiopenDet != bIsMiopenDet)
        {
            return !aIsMiopenDet;
        }

        // For other engines, preserve original order (stable_sort)
        return false;
    });
}

} // namespace utilities
} // namespace hipdnn_backend
