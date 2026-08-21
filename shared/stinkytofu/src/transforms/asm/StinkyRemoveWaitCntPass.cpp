// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ----------------------------------------------------------------------------
// StinkyRemoveWaitCntPass
//
// Precondition pass that strips stale wait-counter instructions so that
// StinkyWaitCntInsertionPass can run later in the pipeline against a clean
// slate and own every emitted wait. The gfx1250 backend invokes this pass right
// after the CFG builder; see docs/user/stinky-waitcnt-insertion-pass.md,
// section "Companion: StinkyRemoveWaitCntPass".
//
// Removal spans two *disjoint* instruction flag bits: IF_WaitCnt (s_wait_dscnt,
// s_wait_loadcnt, s_wait_storecnt, s_wait_asynccnt, s_wait_kmcnt, s_wait_xcnt,
// s_wait_loadcnt_dscnt, s_wait_storecnt_dscnt, s_waitcnt) and IF_WaitTensorCnt
// (s_wait_tensorcnt alone).
//
// "Clean slate" is bounded by one rule: a wait may only be stripped if some
// pass regenerates it. waitcnt::waitReconstruction() -- which lives with the
// dataflow that does the regenerating -- is the single source of truth, and
// shouldRemove() gates on it, so this pass cannot drop a guard the compiler has
// no way to put back. See RemoveWaitCntOptions for the policy exemptions layered
// on top.
// ----------------------------------------------------------------------------

#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"

namespace {
using namespace stinkytofu;

/// Two gates, in order: a wait must be rebuildable at all, and then policy may
/// still keep it. The first gate is what makes the pass safe by construction --
/// there is no path here that strips a wait nothing regenerates.
bool shouldRemove(const StinkyInstruction& inst, const RemoveWaitCntOptions& options) {
    // IF_WaitTensorCnt is disjoint from IF_WaitCnt, so isWaitCnt() alone would
    // miss s_wait_tensorcnt.
    if (!isWaitCnt(inst) && !inst.is(InstFlag::IF_WaitTensorCnt)) return false;

    switch (waitcnt::waitReconstruction(inst)) {
        case waitcnt::WaitReconstruction::None:
            return false;
        case waitcnt::WaitReconstruction::HazardPass:
            return options.removeXcnt;
        case waitcnt::WaitReconstruction::WaitCntInsertion:
            break;
    }

    // Rebuildable, but kept anyway on request. Unlike the gate above, these are
    // policy choices; see RemoveWaitCntOptions for each rationale.
    switch (inst.getUnifiedOpcode()) {
        case GFX::s_wait_tensorcnt:
            return options.removeTensor;
        case GFX::s_wait_kmcnt:
            return options.removeKmcnt;
        default:
            return true;
    }
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
