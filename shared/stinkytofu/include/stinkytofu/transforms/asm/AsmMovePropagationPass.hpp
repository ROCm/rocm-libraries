// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Move propagation for asm-level mov instructions.
///
/// This pass performs conservative, basic-block-local propagation for simple
/// register moves (e.g. v_mov_b32 / s_mov_b32):
///   dst = mov src
/// Later uses of dst are rewritten to src while the mapping remains valid.
///
/// After propagation, a mov is erased only when it is guaranteed dead inside
/// the same block (its destination is redefined before any remaining use).
STINKYTOFU_EXPORT std::unique_ptr<Pass> createAsmMovePropagationPass();

}  // namespace stinkytofu
