// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Removes NOP instructions from selected basic blocks. By default both s_nop and v_nop
/// are removed; pass vNopOnly=true to remove only v_nop.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyRemoveNopPass(bool vNopOnly = false);

}  // namespace stinkytofu
