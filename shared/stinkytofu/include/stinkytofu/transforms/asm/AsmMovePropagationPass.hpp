// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Creates a basic-block-local move propagation pass.
///
/// "Move propagation" in this pass means:
/// - detect an eligible mov (`v_mov_b32` / `s_mov_b32`) with one src and one dst
/// - record a mapping `dst -> src`
/// - rewrite later uses of `dst` to use `src` in the same basic block
///
/// Example:
/// ```
/// Before:
///   v0 = v_mov_b32 v1
///   v2 = v_add_f32 v0, v3
///   v0 = v_sub_f32 v4, v5
///
/// After:
///   v2 = v_add_f32 v1, v3
///   v0 = v_sub_f32 v4, v5
/// ```
///
/// Rules:
/// - Scope is one basic block only (no cross-block propagation)
/// - A mapping is dropped when an overlapping register is redefined
/// - A mov is erased only if it is:
///   - identity (`mov x, x`), or
///   - provably dead in the block (dst is redefined before any later use)
STINKYTOFU_EXPORT std::unique_ptr<Pass> createAsmMovePropagationPass();

}  // namespace stinkytofu
