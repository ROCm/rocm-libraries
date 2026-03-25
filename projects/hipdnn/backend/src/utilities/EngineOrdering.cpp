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

} // namespace utilities
} // namespace hipdnn_backend
