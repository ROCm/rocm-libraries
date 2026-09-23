// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "utilities/EngineOrdering.hpp"

#include <hipdnn_data_sdk/utilities/EngineOrdering.hpp>

namespace hipdnn_backend
{
namespace utilities
{

void sortEngineIds(std::vector<int64_t>& engineIds)
{
    // Delegate to data_sdk implementation (shared with heuristic plugins)
    hipdnn_data_sdk::utilities::sortEngineIds(engineIds);
}

} // namespace utilities
} // namespace hipdnn_backend
