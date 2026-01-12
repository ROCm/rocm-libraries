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

#include <cstddef> // for size_t

namespace stinkytofu
{
    // Forward declaration
    class IRModule;

    /// High-level IR peephole optimization pass
    ///
    /// Applies architecture-independent optimizations on high-level IR
    /// before lowering to assembly IR. Includes:
    ///   - Constant folding (e.g., mul(mul(x, c1), c2) -> mul(x, c1*c2))
    ///   - Algebraic simplifications (e.g., mul(x, 1) -> mov(x))
    ///   - Instruction fusion (e.g., add+mul -> fma)
    ///   - Dead move elimination
    ///
    /// This pass operates on IRInstruction objects before they are lowered
    /// to StinkyInstruction (assembly IR).
    ///
    /// NOTE: This is a module-level pass, not part of IRInstPassManager.
    ///       It should be called directly on an IRModule.
    class HighLevelPeepholePass
    {
    public:
        HighLevelPeepholePass() = default;

        const char* name() const
        {
            return "HighLevelPeepholePass";
        }

        /// Run the pass on the given IRModule
        /// @param module The IRModule to optimize
        /// @return true if any changes were made, false otherwise
        bool run(IRModule& module);

        /// Get statistics about optimizations applied
        size_t getOptimizationCount() const
        {
            return optimizationCount;
        }

    private:
        size_t optimizationCount = 0;

        // TODO: When full pattern matching is implemented, these will be used
        // bool applyPatterns(IRModule& module, IRInstruction* inst);
    };

} // namespace stinkytofu
