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

namespace stinkytofu {
class Pass;

/**
 * @brief Converts simple IR instructions to assembly instructions (1:1 mapping)
 *
 * This pass handles non-composite IR instructions that map directly to
 * a single assembly instruction. For example:
 * - VAddF32 (IR) -> v_add_f32 (assembly)
 * - VMulF32 (IR) -> v_mul_f32 (assembly)
 * - SBarrier (IR) -> s_barrier (assembly)
 *
 * Composite instructions (like VAddPKF32, VMovB64) should be expanded
 * by CompositeInstructionLoweringPass BEFORE this pass runs.
 *
 * The pass uses auto-generated IR mnemonic mappings to find the correct
 * assembly mnemonic for the target architecture.
 *
 * Now uses unified Pass infrastructure:
 * - Operates on Function -> BasicBlock -> IRList
 * - Replaces LogicalInstruction* with StinkyInstruction* in-place
 * - Integrates with PassManager
 *
 * Usage:
 * ```cpp
 * PassManager pm;
 * pm.addPass(createCompositeInstructionLoweringPass());
 * pm.addPass(createToStinkyAsmPass());
 * pm.run();
 * ```
 */
std::unique_ptr<Pass> createToStinkyAsmPass();

}  // namespace stinkytofu
