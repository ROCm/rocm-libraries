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
class BasicBlock;
class Function;
class Pass;
struct DominanceInfo;

/// Builds def-use chains for all instructions in the given Function.
///
/// Two phases:
///   1. insertPhiInstructions() — places PHI pseudo-instructions at CFG join
///      points using Cytron et al.'s dominance-frontier algorithm (semi-pruned).
///   2. Chain construction — a single RPO traversal with dominator-inherited
///      reaching definitions links every instruction's sources and users.
///
/// sources layout:
///   - Regular instructions: one entry per DWORD source register that has a
///     reaching definition (skips undefined / pseudo registers).
///   - PHI instructions: sources[j] = reaching def from predecessor j
///     (nullptr if undefined on that edge).
///
/// The chains are stored directly in StinkyInstruction objects:
///   * inst->getSources()  — definitions consumed by this instruction
///   * inst->getUsers()        — instructions that consume this instruction
///
/// Dead PHIs (those with no users) are removed after chain construction.
///
/// Usage:
///   buildUseDefChain(function, true);   // rebuild (clears old PHIs/chains)
///   buildUseDefChain(function, false);  // first time (no existing state)
///   for(auto* def : inst->getSources()) { /* use operand def */ }
///   for(auto* user : inst->getUsers()) { /* use user */ }
///
/// @param clearExisting  When true, existing PHI instructions and def-use
///                        chains are removed before rebuilding.  Pass false
///                        on a fresh function that has no PHIs/chains yet.
///
/// Note: Assumes non-SSA form (physical registers). Requires CFG to be built.
void buildUseDefChain(Function& func, bool clearExisting);

/// Overload that accepts pre-computed dominance info to avoid
/// redundant computeDominanceInfo() calls.
void buildUseDefChain(Function& func, const DominanceInfo& domInfo, bool clearExisting);

/// Creates a Pass that builds the def-use chain for a Function.
/// Use this to run buildUseDefChain as part of a pass pipeline.
///
/// @param clearExisting  Forwarded to buildUseDefChain(). When true,
///                        existing PHIs and chains are removed first.
std::unique_ptr<Pass> createBuildUseDefChainPass(bool clearExisting = true);

}  // namespace stinkytofu
