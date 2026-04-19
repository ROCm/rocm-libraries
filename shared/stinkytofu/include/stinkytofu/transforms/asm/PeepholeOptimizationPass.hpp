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

namespace stinkytofu {
class Pass;

/// Creates a peephole optimization pass that applies instruction fusion patterns
/// defined in PeepholePatterns.pattern.
///
/// This pass performs local peephole optimizations such as:
/// - Add+FMA fusion (e.g., v_add_f32 + v_fma_f32 -> v_fma_f32)
/// - Mul+Add fusion (e.g., v_mul_f32 + v_add_f32 -> v_fma_f32)
/// - Constant folding and propagation
///
/// The patterns are declared in PeepholePatterns.pattern and automatically
/// generated into C++ matcher code by the TableGen tool.
///
/// Example usage:
/// ```cpp
/// PassManager pm;
/// pm.addPass(createPeepholeOptimizationPass());
/// pm.run();
/// ```
std::unique_ptr<Pass> createPeepholeOptimizationPass();

}  // namespace stinkytofu
