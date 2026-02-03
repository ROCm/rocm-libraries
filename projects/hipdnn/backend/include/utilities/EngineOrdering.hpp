// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <vector>

namespace hipdnn_backend
{
namespace utilities
{

/// @brief Sorts engine IDs with MIOpen-specific ordering requirements.
/// Ordering: MIOPEN_ENGINE first, other engines in middle (stable order), MIOPEN_ENGINE_DETERMINISTIC last
/// @param engineIds Vector of engine IDs to sort (modified in-place)
void sortEngineIds(std::vector<int64_t>& engineIds);

} // namespace utilities
} // namespace hipdnn_backend
