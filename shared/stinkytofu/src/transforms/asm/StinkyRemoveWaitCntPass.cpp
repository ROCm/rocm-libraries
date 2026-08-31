// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ----------------------------------------------------------------------------
// StinkyRemoveWaitCntPass
//
// Precondition pass that strips stale wait-counter instructions so that
// StinkyWaitCntInsertionPass can run later in the pipeline against a clean
// slate and own every emitted wait. The gfx1250 backend invokes this pass right
// after the CFG builder; see docs/user/stinky-waitcnt-insertion-pass.md,
// section "The reconstruction contract".
//
// Removal spans two *disjoint* instruction flag bits: IF_WaitCnt (s_wait_dscnt,
// s_wait_loadcnt, s_wait_storecnt, s_wait_asynccnt, s_wait_kmcnt, s_wait_xcnt,
// s_wait_loadcnt_dscnt, s_wait_storecnt_dscnt, s_waitcnt) and IF_WaitTensorCnt
// (s_wait_tensorcnt alone).
//
// "Clean slate" is bounded by a legality rule: a wait may only be removed if
// some pass regenerates it, otherwise the hazard it guarded goes unguarded.
// waitcnt::waitReconstruction() -- which lives with the dataflow that does the
// regenerating -- is the single source of truth for that, so this pass cannot
// drop a wait the compiler has no way to put back. Legality is decided first and
// is not configurable; RemoveWaitCntOptions then applies policy to whatever
// remains legal.
// ----------------------------------------------------------------------------

#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"

namespace {
using namespace stinkytofu;

/// Legality: removing a wait no pass regenerates drops the hazard it guarded, so
/// it is never legal however the pass is configured.
bool isLegalToRemove(const StinkyInstruction& inst) {
    return waitcnt::waitReconstruction(inst) != waitcnt::WaitReconstruction::None;
}

/// Policy: which of the legal candidates the caller actually wants gone. See
/// RemoveWaitCntOptions for why each opcode is exempt.
bool isRemovalWanted(const StinkyInstruction& inst, const RemoveWaitCntOptions& options) {
    switch (inst.getUnifiedOpcode()) {
        case GFX::s_wait_tensorcnt:
            return options.removeTensor;
        case GFX::s_wait_xcnt:
            return options.removeXcnt;
        case GFX::s_wait_kmcnt:
            return options.removeKmcnt;
        default:
            return true;
    }
}

bool shouldRemove(const StinkyInstruction& inst, const RemoveWaitCntOptions& options) {
    // IF_WaitTensorCnt is disjoint from IF_WaitCnt, so isWaitCnt() alone would
    // miss s_wait_tensorcnt.
    if (!isWaitCnt(inst) && !inst.is(InstFlag::IF_WaitTensorCnt)) return false;
    return isLegalToRemove(inst) && isRemovalWanted(inst, options);
}

void removeWaitCntsInBlock(BasicBlock& bb, const RemoveWaitCntOptions& options) {
    for (auto it = bb.begin(); it != bb.end();) {
        auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
        if (inst != nullptr && shouldRemove(*inst, options)) {
            it = bb.eraseIR(it);
        } else {
            ++it;
        }
    }
}

class StinkyRemoveWaitCntPass : public StinkyInstPass {
   public:
    explicit StinkyRemoveWaitCntPass(RemoveWaitCntOptions options) : options(options) {}

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
                removeWaitCntsInBlock(bb, options);
            }
        }
        return preserveCFGAnalyses();
    }

   private:
    RemoveWaitCntOptions options;
};

char StinkyRemoveWaitCntPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createStinkyRemoveWaitCntPass(RemoveWaitCntOptions options) {
    return std::make_unique<StinkyRemoveWaitCntPass>(options);
}
}  // namespace stinkytofu
