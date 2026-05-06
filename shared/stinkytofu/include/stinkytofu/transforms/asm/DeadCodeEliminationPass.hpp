/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

/// Creates a dead code elimination (DCE) pass that removes dead store instructions.
///
/// This pass uses a simple forward-scan pattern to eliminate dead stores:
/// 1. Find an instruction that defines a register (e.g., v0)
/// 2. Scan forward to see if v0 is redefined before being used
/// 3. If redefined with no uses in between, remove the first definition
///
/// This pattern is safe across blocks and function calls because:
/// - If a register is used in another block, we see it at the branch/jump
/// - If a register is used across function boundaries, we see it at the call site
/// - We only remove when there's a provable overwrite with no intervening uses
///
/// Example:
/// ```
/// Before DCE:
///   v_mul_f32 v0, v1, v2     // Dead store: v0 overwritten below
///   v_add_f32 v3, v4, v5
///   v_mov_b32 v0, v6         // Redefines v0
///   global_store_dword v0    // Uses the second v0
///
/// After DCE:
///   v_add_f32 v3, v4, v5
///   v_mov_b32 v0, v6
///   global_store_dword v0
/// ```
///
/// Key features:
/// - Simple forward scan - no complex def-use analysis needed
/// - Safe across basic blocks (scans entire function)
/// - Iteratively removes dead stores until fixpoint
/// - Preserves side-effecting instructions
///
/// Usage:
/// ```cpp
/// PassManager pm;
/// pm.addPass(createPeepholeOptimizationPass());  // May create dead stores
/// pm.addPass(createDeadCodeEliminationPass());   // Clean up
/// pm.run();
/// ```
STINKYTOFU_EXPORT std::unique_ptr<Pass> createDeadCodeEliminationPass();

}  // namespace stinkytofu
