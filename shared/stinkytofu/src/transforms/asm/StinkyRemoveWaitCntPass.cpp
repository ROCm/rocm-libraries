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

// ----------------------------------------------------------------------------
// StinkyRemoveWaitCntPass
//
// Precondition pass that strips stale wait-counter instructions so that
// StinkyWaitCntInsertionPass can run later in the pipeline against a clean
// slate and own every emitted wait. The gfx1250 backend invokes this pass
// (with the default removeTensorWaitCnt = true) right after the CFG builder;
// see docs/user/stinky-waitcnt-insertion-pass.md, section
// "Companion: StinkyRemoveWaitCntPass".
//
// Removal is driven by two *disjoint* instruction flag bits:
//
//   - IF_WaitCnt        Always removed via isWaitCnt(). Covers the standard
//                       wait-counter opcodes: s_wait_dscnt, s_wait_loadcnt,
//                       s_wait_storecnt, s_wait_asynccnt, s_wait_kmcnt,
//                       s_wait_xcnt, s_wait_loadcnt_dscnt,
//                       s_wait_storecnt_dscnt, s_waitcnt.
//   - IF_WaitTensorCnt  Removed via isTensorWaitCnt() iff removeTensorWaitCnt
//                       is true (the default). The only opcode carrying this
//                       flag is s_wait_tensorcnt.
//
// Because the two flag bits never coexist on the same opcode, the per-
// instruction predicate is the simple OR:
//   isWaitCnt(inst) || (removeTensorWaitCnt && isTensorWaitCnt(inst))
// ----------------------------------------------------------------------------

#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace {
using namespace stinkytofu;

/// True iff `stinkyInst` is `s_wait_tensorcnt`, the only opcode carrying
/// `IF_WaitTensorCnt`. This flag is disjoint from `IF_WaitCnt`, so
/// `isWaitCnt()` does not match `s_wait_tensorcnt` and a dedicated check is
/// required when tensor-wait removal is enabled.
bool isTensorWaitCnt(StinkyInstruction* stinkyInst) {
    return stinkyInst != nullptr && stinkyInst->is(InstFlag::IF_WaitTensorCnt);
}

/// Erase every wait-counter instruction in `bb` that matches the disjoint
/// flag-bit predicate described in the file-level comment.
///
/// @param bb                   Basic block to mutate in place.
/// @param removeTensorWaitCnt  When true (the default policy), also strip
///                             `s_wait_tensorcnt` so the downstream insertion
///                             pass starts from a fully clean slate. When
///                             false, leave tensor waits in place so a
///                             subsequent insertion pass can reuse them.
void removeWaitCntsInBlock(BasicBlock& bb, bool removeTensorWaitCnt) {
    for (auto it = bb.begin(); it != bb.end();) {
        auto* stinkyInst = dyn_cast<StinkyInstruction>(it.getNodePtr());

        if (stinkyInst &&
            (isWaitCnt(*stinkyInst) || (removeTensorWaitCnt && isTensorWaitCnt(stinkyInst)))) {
            it = bb.eraseIR(it);
        } else {
            ++it;
        }
    }
}

class StinkyRemoveWaitCntPass : public StinkyInstPass {
   public:
    StinkyRemoveWaitCntPass(bool removeTensorWaitCnt) : removeTensorWaitCnt(removeTensorWaitCnt) {}

    static char ID;

    const char* getName() const override {
        return "StinkyRemoveWaitCntPass";
    }

    PassID getPassID() const override {
        return &StinkyRemoveWaitCntPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        for (BasicBlock& bb : func) {
            if (passCtx.shouldProcessBasicBlock(bb)) {
                removeWaitCntsInBlock(bb, removeTensorWaitCnt);
            }
        }
        return preserveCFGAnalyses();
    }

   private:
    bool removeTensorWaitCnt;
};

char StinkyRemoveWaitCntPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyRemoveWaitCntPass(bool removeTensorWaitCnt) {
    return std::make_unique<StinkyRemoveWaitCntPass>(removeTensorWaitCnt);
}
}  // namespace stinkytofu
