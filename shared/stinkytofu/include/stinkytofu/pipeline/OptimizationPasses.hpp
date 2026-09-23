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

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/pipeline/OptLevel.hpp"
#include "stinkytofu/transforms/asm/DeadCodeEliminationPass.hpp"
#include "stinkytofu/transforms/asm/PeepholeOptimizationPass.hpp"
#include "stinkytofu/transforms/asm/RedundantMovEliminationPass.hpp"

namespace stinkytofu {
/// Add optimization passes (Peephole, RedundantMovElim, DCE) to PassManager.
/// Pass selection depends on OptLevel.
/// Shared across backends.
inline void addPeepholeOptPasses(PassManager& pm, OptLevel optLevel) {
    if (optLevel >= OptLevel::O1) pm.addPass(createPeepholeOptimizationPass());

    if (optLevel >= OptLevel::O3) pm.addPass(createRedundantMovEliminationPass());

    if (optLevel >= OptLevel::O2) pm.addPass(createDeadCodeEliminationPass());
}

}  // namespace stinkytofu
