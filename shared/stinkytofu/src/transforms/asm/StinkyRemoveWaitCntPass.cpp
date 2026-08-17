// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
//   - IF_WaitCnt        Removed via isWaitCnt(), except s_wait_xcnt follows a
//                       temporary SIA4/SIA0 split. TODO: remove this special
//                       case once a dedicated hazard pass handles xcnt
//                       placement. Covers the standard wait-counter opcodes:
//                       s_wait_dscnt, s_wait_loadcnt,
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

bool isXcntWaitCnt(StinkyInstruction* stinkyInst) {
    return stinkyInst != nullptr && stinkyInst->getUnifiedOpcode() == GFX::s_wait_xcnt;
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
/// @param removeXcntWaitCnt    When true, remove s_wait_xcnt for SIA4. TODO:
///                             remove this temporary split once a dedicated
///                             hazard pass handles xcnt placement. Until then,
///                             non-SIA4 paths preserve hand-authored xcnt waits.
void removeWaitCntsInBlock(BasicBlock& bb, bool removeTensorWaitCnt, bool removeXcntWaitCnt) {
    for (auto it = bb.begin(); it != bb.end();) {
        auto* stinkyInst = dyn_cast<StinkyInstruction>(it.getNodePtr());

        if (stinkyInst &&
            ((isWaitCnt(*stinkyInst) && (removeXcntWaitCnt || !isXcntWaitCnt(stinkyInst))) ||
             (removeTensorWaitCnt && isTensorWaitCnt(stinkyInst)))) {
            it = bb.eraseIR(it);
        } else {
            ++it;
        }
    }
}

class StinkyRemoveWaitCntPass : public StinkyInstPass {
   public:
    StinkyRemoveWaitCntPass(bool removeTensorWaitCnt, bool removeXcntWaitCnt)
        : removeTensorWaitCnt(removeTensorWaitCnt), removeXcntWaitCnt(removeXcntWaitCnt) {}

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
                removeWaitCntsInBlock(bb, removeTensorWaitCnt, removeXcntWaitCnt);
            }
        }
        return preserveCFGAnalyses();
    }

   private:
    bool removeTensorWaitCnt;
    bool removeXcntWaitCnt;
};

char StinkyRemoveWaitCntPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyRemoveWaitCntPass(bool removeTensorWaitCnt,
                                                    bool removeXcntWaitCnt) {
    return std::make_unique<StinkyRemoveWaitCntPass>(removeTensorWaitCnt, removeXcntWaitCnt);
}
}  // namespace stinkytofu
