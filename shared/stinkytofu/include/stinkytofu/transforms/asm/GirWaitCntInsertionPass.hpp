/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Inserts dscnt/tensorcnt waits for frame hazards reconstructed on the scheduled ST CFG.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createGirWaitCntInsertionPass();

}  // namespace stinkytofu
