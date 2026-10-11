// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/InsertHazardGapNopPass.hpp"

#include <algorithm>
#include <iostream>
#include <unordered_map>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

#define DEBUG_TYPE "InsertHazardGapNopPass"

namespace {
using namespace stinkytofu;

// s_nop N gives N+1 wait states; N is a 4-bit field.
constexpr int kMaxNopWaitStates = 16;

class InsertHazardGapNopPassImpl : public Pass {
   public:
    static char ID;

    const char* getName() const override {
        return "InsertHazardGapNopPass";
    }

    PassID getPassID() const override {
        return &InsertHazardGapNopPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        const auto& triple = passCtx.getGemmTileConfig().arch;
        const GfxArchID arch = getGfxArchID(triple[0], triple[1], triple[2]);
        unsigned inserted = 0;
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            for (const HazardRule& rule : kCdna5HazardRules) {
                if (!rule.enforced) continue;
                inserted += padBlock(bb, rule, arch);
            }
        }
        PASS_DEBUG(std::cerr << "[InsertHazardGapNop] @" << func.getName() << ": " << inserted
                             << " s_nop inserted\n");
        return preserveCFGAnalyses();
    }

   private:
    static unsigned padBlock(BasicBlock& bb, const HazardRule& rule, GfxArchID arch) {
        // Per register: the cycle at which its latest writer finished, if that
        // writer is a producer of this rule. A non-producer write ends the hazard.
        std::unordered_map<unsigned, int> producerEnd;
        int now = 0;
        unsigned inserted = 0;
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
            if (!inst) continue;

            if (rule.isConsumer(*inst)) {
                int latest = -1;
                for (const StinkyRegister& r : inst->getSrcRegs()) {
                    if (r.dataType != StinkyRegister::Type::Register || r.reg.type != rule.regType)
                        continue;
                    for (unsigned i = 0; i < r.reg.num; ++i) {
                        auto p = producerEnd.find(r.reg.idx + i);
                        if (p != producerEnd.end()) latest = std::max(latest, p->second);
                    }
                }
                if (latest >= 0 && now - latest < rule.distance) {
                    int missing = rule.distance - (now - latest);
                    while (missing > 0) {
                        const int states = std::min(missing, kMaxNopWaitStates);
                        AsmIRBuilder builder(bb, arch);
                        StinkyInstruction* nop =
                            builder.create(getMCIDByUOp(GFX::s_nop, arch), inst);
                        nop->addSrcReg(StinkyRegister(states - 1));
                        now += states;
                        missing -= states;
                        ++inserted;
                    }
                    PASS_DEBUG(std::cerr << "[InsertHazardGapNop] " << rule.name << " bb=\""
                                         << bb.getLabel() << "\": padded before "
                                         << inst->getHwInstDesc()->mnemonic << "\n");
                }
            }

            now += hazardGapCycles(*inst);
            const bool isProducer = rule.isProducer(*inst);
            for (const StinkyRegister& r : inst->getDestRegs()) {
                if (r.dataType != StinkyRegister::Type::Register || r.reg.type != rule.regType)
                    continue;
                for (unsigned i = 0; i < r.reg.num; ++i) {
                    if (isProducer)
                        producerEnd[r.reg.idx + i] = now;
                    else
                        producerEnd.erase(r.reg.idx + i);
                }
            }
        }
        return inserted;
    }
};

char InsertHazardGapNopPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createInsertHazardGapNopPass() {
    return std::make_unique<InsertHazardGapNopPassImpl>();
}
}  // namespace stinkytofu
