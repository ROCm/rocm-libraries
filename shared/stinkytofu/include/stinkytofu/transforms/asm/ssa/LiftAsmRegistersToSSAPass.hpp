/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include <cstddef>
#include <memory>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/ir/asm/ssa/SSAOperandUnits.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {
class Function;
class Pass;
struct DominanceInfo;

struct LiftAsmRegistersToSSAOptions {
    /// Register classes to lift. Anything outside this set stays physical: its
    /// operands keep their immediate payload and SSA destruction never rewrites
    /// them, so those registers come out exactly as the producer wrote them.
    ///
    /// Narrow it to allocate one class at a time. Lifting SGPRs alone, for
    /// instance, leaves every VGPR untouched, which also keeps the VGPR
    /// high-water mark and its kernel metadata unchanged.
    ///
    /// A class outside isLiftableRegClass() is still an error, not a silent skip:
    /// this selects among the classes lifting can model, it does not extend them.
    RegClassSet classes = RegClassSet::all();

    /// Verify attached SSA before handing the function back.
    bool verify = true;

    /// Treat a read with no reaching definition as a function live-in.
    ///
    /// Physical input does not say which registers are genuine kernel inputs,
    /// so this conservative default preserves the meaning of the original
    /// program. Set false to require that every read is defined, which is the
    /// strict mode used once entry metadata is available.
    bool allowInferredLiveIns = true;
};

struct LiftAttachedSSAResult {
    size_t valueCount = 0;
    size_t blockArgumentCount = 0;
};

/// Attach SSA directly onto \p function from its physical register operands.
///
/// Physical registers are treated as mutable variables: every reaching
/// definition of a register unit becomes its own SSA value, and each value
/// keeps a PhysicalBinding for legacy replay.
///
/// Values that merge at a control-flow join become block arguments, placed at
/// iterated dominance frontiers and pruned by liveness so no dead argument is
/// created. Reducible and irreducible CFGs are both supported.
///
/// Current scope is deliberately narrow. Operands must be full-DWORD VGPRs or
/// SGPRs, and every block must be reachable from the entry. Run
/// StinkyUnreachableBlockElimPass after CFG construction to drop dead blocks.
/// Literals, special registers such as EXEC or SCC, and pseudo registers
/// become immediate SSA operands rather than values. Anything else -
/// accumulator classes, unresolved template virtual registers, True16 halves,
/// calls, or leftover analysis PHIs - is reported as an error instead of
/// being silently mishandled.
///
/// The function must already be free of def-use analysis state; a leftover
/// `GFX::PHI` is an error rather than something to clean up. Run
/// RemoveDefUseAnalysisPass first.
///
/// On success, the function's SSAArena, instruction AttachedSSA payloads, and
/// BasicBlock arguments are rebuilt. On failure, attached SSA is left empty.
Expected<LiftAttachedSSAResult> liftAsmRegistersToAttachedSSA(
    Function& function, const LiftAsmRegistersToSSAOptions& options = {});

/// As above, reusing dominance information the caller already computed.
Expected<LiftAttachedSSAResult> liftAsmRegistersToAttachedSSA(
    Function& function, const DominanceInfo& dominance,
    const LiftAsmRegistersToSSAOptions& options = {});

/// True when any function in \p functions contains a call site.
///
/// A kernel must be recoloured as a whole or not at all. Caller and callee agree
/// on registers only through the convention the producer used, and nothing
/// records that agreement yet, so recolouring one side would silently break it.
/// A pipeline enabling allocation therefore preflights the whole kernel with
/// this and keeps the legacy path for all of it when the answer is true, rather
/// than deciding function by function.
bool kernelHasCallSites(const std::vector<const Function*>& functions);

/// Creates a pass that lifts a function's physical registers to attached SSA
/// on the IR.
///
/// Running this pass applies LiftAsmRegistersToSSAOptions and emits located
/// missed-optimization remarks for unsupported input.
///
/// The pass is function-wide: PHI placement and renaming need every block, so
/// it refuses to run at all when basic-block filtering excludes any block.
///
/// The function must already be free of def-use analysis state; run
/// RemoveDefUseAnalysisPass first.
///
/// Unsupported input is a missed-optimization remark, not a hard error.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createLiftAsmRegistersToSSAPass(
    const LiftAsmRegistersToSSAOptions& options = {});

}  // namespace stinkytofu
