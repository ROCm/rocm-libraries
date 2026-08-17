/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Creates a redundant mov elimination pass that removes duplicate mov-type
/// instructions within basic blocks.
///
/// This pass finds mov-type instructions (v_mov_b32, s_mov_b32, etc.) that are
/// identical to previous instructions in the same basic block and removes them.
///
/// Key features:
/// - Identifies mov instructions with identical opcodes and operands
/// - Removes redundant mov instructions
/// - Works within basic blocks only (no CFG analysis)
/// - Easy to extend to other instruction types
///
/// Example:
/// ```
/// Before Redundant Mov Elimination:
///   v_mov_b32 v0, 0x2222       // First assignment
///   v_add_f32 v1, v0, 2
///   v_mov_b32 v0, 0x2222       // Duplicate! Same instruction
///   v_add_f32 v2, v0, 3
///
/// After Redundant Mov Elimination:
///   v_mov_b32 v0, 0x2222       // Original kept
///   v_add_f32 v1, v0, 2
///   // v_mov_b32 removed (redundant)
///   v_add_f32 v2, v0, 3
/// ```
///
/// Safety:
/// - Only processes eligible instruction types (configurable)
/// - Only operates within basic blocks
/// - Preserves all control flow and side-effect instructions
///
/// Usage:
/// ```cpp
/// PassManager pm;
/// pm.addPass(createRedundantMovEliminationPass());
/// pm.run();
/// ```
///
/// Note: This pass is designed to be simple and conservative. It can be extended
/// to handle more instruction types by modifying isEligibleForRedundantMovElimination().
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRedundantMovEliminationPass();

}  // namespace stinkytofu
