/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Moves each GIR fence inside its admissible window to the slot with the deepest counter
/// residual, so the wait the next pass anchors on it drains as little as possible.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createGirFencePlacementPass();

}  // namespace stinkytofu
