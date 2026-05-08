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

/// High-level IR peephole optimization pass
///
/// Applies architecture-independent optimizations on high-level IR
/// before lowering to assembly IR. Includes:
///   - Constant folding (e.g., mul(mul(x, c1), c2) -> mul(x, c1*c2))
///   - Algebraic simplifications (e.g., mul(x, 1) -> mov(x))
///   - Instruction fusion (e.g., add+mul -> fma)
///   - Dead move elimination
///
/// This pass operates on LogicalInstruction objects in IRList before they
/// are lowered to StinkyInstruction (assembly IR).
///
/// Now uses the unified Pass infrastructure:
///   - Operates on Function -> BasicBlock -> IRList
///   - Works with raw LogicalInstruction* pointers
///   - Integrates with PassManager
///
/// Usage:
/// ```cpp
/// PassManager pm;
/// pm.addPass(createLogicalPeepholePass());
/// pm.run();
/// ```
STINKYTOFU_EXPORT std::unique_ptr<Pass> createLogicalPeepholePass();

}  // namespace stinkytofu
