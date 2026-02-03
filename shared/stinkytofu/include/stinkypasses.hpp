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

namespace rocisa
{
    struct Module;
}

namespace stinkytofu
{
    // NOTE: This header assumes stinkytofu.hpp is already included.
    // It defines factory functions for creating pass instances.
    // Pass base classes (IRInstPass, Pass, etc.) are in stinkytofu.hpp.

    // High-level IR pass factories (now using unified Pass infrastructure)
    std::unique_ptr<Pass> createCompositeInstructionLoweringPass();
    std::unique_ptr<Pass> createToStinkyAsmPass();

    struct WaitCntConfig;

    std::unique_ptr<Pass> createStinkyDAGSchedulerPass();
    std::unique_ptr<Pass> createStinkyConservativeWaitCntPass();
    std::unique_ptr<Pass> createStinkyMinimalWaitCntPass();
    std::unique_ptr<Pass> createStinkyUnrollWaitCntPass();
    std::unique_ptr<Pass> createStinkyCustomWaitCntPass(const WaitCntConfig& config);
    std::unique_ptr<Pass> createScheduleLastLRsPass();
    std::unique_ptr<Pass> createScheduleFirstLRsPass();

    // CFGBuilderPass splits a single flat BasicBlock into multiple BasicBlocks
    // based on label boundaries and builds CFG edges between them.
    //
    // This pass is optional and only needed when the input IR contains
    // multiple labels and branches (rare case for tensilelite).
    //
    // Input:  Function with one BasicBlock containing labels and branches
    // Output: Function with multiple BasicBlocks connected via CFG edges
    std::unique_ptr<Pass> createCFGBuilderPass();

    // PeepholeOptimizationPass applies local instruction fusion patterns
    // defined in PeepholePatterns.pattern to optimize instruction sequences.
    std::unique_ptr<Pass> createPeepholeOptimizationPass();

    // DeadCodeEliminationPass removes instructions whose results are never used.
    // This pass is particularly useful after peephole optimization which may
    // create dead intermediate values.
    std::unique_ptr<Pass> createDeadCodeEliminationPass();

    // RedundantMovEliminationPass removes duplicate mov-type instructions within
    // basic blocks. Simple and conservative, designed to handle v_mov_b32,
    // s_mov_b32, and similar instructions.
    std::unique_ptr<Pass> createRedundantMovEliminationPass();

    // DelayAluInsertionPass inserts s_delay_alu instructions for RDNA3 (gfx11xx)
    // architectures to handle ALU instruction dependencies. Only runs on gfx11xx.
    std::unique_ptr<Pass> createDelayAluInsertionPass();
}
