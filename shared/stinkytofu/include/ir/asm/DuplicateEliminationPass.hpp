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

namespace stinkytofu
{
    class Pass;

    /// Creates a duplicate elimination pass (also known as Common Subexpression
    /// Elimination or CSE) that identifies and eliminates redundant computations.
    ///
    /// This pass finds instructions that compute the same value as a previous
    /// instruction and replaces uses of the duplicate with the original result.
    /// This is a form of local CSE within basic blocks.
    ///
    /// Key features:
    /// - Identifies instructions with identical opcodes and operands
    /// - Replaces duplicate computations with register copies or direct reuse
    /// - Works within basic blocks (local CSE)
    /// - Preserves program semantics through careful dependency analysis
    ///
    /// Example:
    /// ```
    /// Before Duplicate Elimination:
    ///   v_mul_f32 v0, v1, v2       // First computation
    ///   v_add_f32 v3, v4, v5
    ///   v_mul_f32 v6, v1, v2       // Duplicate! Same operands
    ///   v_sub_f32 v7, v6, v0
    ///
    /// After Duplicate Elimination:
    ///   v_mul_f32 v0, v1, v2       // Original kept
    ///   v_add_f32 v3, v4, v5
    ///   // v_mul_f32 removed, v6 replaced with v0
    ///   v_sub_f32 v7, v0, v0
    /// ```
    ///
    /// Safety:
    /// - Only eliminates pure instructions (no side effects)
    /// - Ensures source operands haven't been modified between duplicate instructions
    /// - Preserves instruction ordering for correctness
    /// - Works well with Dead Code Elimination to remove unused duplicates
    ///
    /// Usage:
    /// ```cpp
    /// PassManager pm;
    /// pm.addPass(createDuplicateEliminationPass());
    /// pm.addPass(createDeadCodeEliminationPass());  // Clean up unused originals
    /// pm.run();
    /// ```
    ///
    /// Note: This pass performs LOCAL CSE within basic blocks. For global CSE
    /// across basic blocks, a more sophisticated analysis is needed.
    std::unique_ptr<Pass> createDuplicateEliminationPass();

} // namespace stinkytofu
