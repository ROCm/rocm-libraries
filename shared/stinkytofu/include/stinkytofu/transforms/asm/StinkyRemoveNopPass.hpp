// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Removes NOP instructions (s_nop and v_nop) from selected basic blocks.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyRemoveNopPass();

}  // namespace stinkytofu
