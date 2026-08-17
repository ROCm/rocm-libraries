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
///
/// Ordering rationale:
/// - MIOPEN_ENGINE first: Default engine with full operation support
/// - Other engines middle: Stable order preserved for predictability
/// - MIOPEN_ENGINE_DETERMINISTIC last: Limited to conv operations only,
///   deprioritized due to performance trade-offs and reduced operation support
///
/// @param engineIds Vector of engine IDs to sort (modified in-place)
void sortEngineIds(std::vector<int64_t>& engineIds);

} // namespace utilities
} // namespace hipdnn_backend
