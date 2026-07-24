// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "stinkytofu/transforms/asm/AsmMovePropagationPass.hpp"

#include <unordered_map>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace {
using namespace stinkytofu;

bool isSupportedMov(const StinkyInstruction& inst) {
    auto uop = inst.getUnifiedOpcode();
    return uop == GFX::v_mov_b32 || uop == GFX::s_mov_b32;
}

bool isEligibleMov(const StinkyInstruction& inst) {
    if (!isSupportedMov(inst)) return false;
    if (inst.getDestRegs().size() != 1 || inst.getSrcRegs().size() != 1) return false;

    const StinkyRegister& dst = inst.getDestReg(0);
    const StinkyRegister& src = inst.getSrcReg(0);
    if (!dst.isRegister() || !src.isRegister()) return false;
    if (dst.reg.num != 1 || src.reg.num != 1) return false;
    if (isPseudoReg(dst) || isPseudoReg(src)) return false;

    return true;
}

bool hasSrcOverlap(const StinkyInstruction& inst, const StinkyRegister& reg) {
    for (const StinkyRegister& src : inst.getSrcRegs()) {
        if (src.isOverlap(reg)) return true;
    }
    return false;
}

bool hasDestOverlap(const StinkyInstruction& inst, const StinkyRegister& reg) {
    for (const StinkyRegister& dst : inst.getDestRegs()) {
        if (dst.isOverlap(reg)) return true;
    }
    return false;
}

class AsmMovePropagationPassImpl : public Pass {
   public:
    static constexpr const char* PassName = "AsmMovePropagationPass";
    static char ID;

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return PassName;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            runOnBasicBlock(bb);
        }
        return preserveCFGAnalyses();
    }

   private:
    // Algorithm (currently only basic-block-local):
    // Phase A - propagation:
    //   Build a per-basic-block map incrementally:
    //     - start with an empty map
    //     - when an eligible mov is seen, record/update {dst -> src}
    //       (src is the rewritten source at that point)
    //     - erase entries when current defs overlap either mapped dst or src (break the chain)
    //   Then walk instructions in order using that evolving map.
    //   For each instruction:
    //     1) rewrite each register source via the current map
    //     2) invalidate map entries touched by current defs (key/value overlap)
    //     3) if instruction is an eligible mov, add/update {dst -> src}
    //
    // Phase B - mov cleanup:
    //   Re-scan mov instructions and erase only when safe:
    //     - dst is redefined before any later use in the same block.
    //     - identity mov (mov x, x)
    //   Otherwise keep the mov conservatively (it may be live-out).
    void runOnBasicBlock(BasicBlock& bb) {
        std::vector<StinkyInstruction*> instructions;
        for (IRBase& node : bb) {
            if (node.getType() == IRBase::IRType::StinkyTofu) {
                instructions.push_back(cast<StinkyInstruction>(&node));
            }
        }

        std::unordered_map<StinkyRegister, StinkyRegister> moveMap;

        auto resolveMappedSrc = [&moveMap](const StinkyRegister& reg) {
            StinkyRegister resolved = reg;
            // moveMap is invalidated on defs, so chains should not form cycles.
            while (true) {
                auto it = moveMap.find(resolved);
                if (it == moveMap.end()) break;
                const StinkyRegister& next = it->second;
                if (next == resolved) break;
                resolved = next;
            }
            return resolved;
        };

        auto invalidateByDef = [&moveMap](const StinkyRegister& defReg) {
            if (!defReg.isRegister()) return;
            for (auto it = moveMap.begin(); it != moveMap.end();) {
                if (it->first.isOverlap(defReg) || it->second.isOverlap(defReg)) {
                    it = moveMap.erase(it);
                } else {
                    ++it;
                }
            }
        };

        // 1) rewrite current instruction sources using mappings from earlier instructions
        // 2) invalidate mappings killed by current instruction defs
        // 3) if current instruction is an eligible mov, add its new mapping
        for (StinkyInstruction* inst : instructions) {
            for (size_t i = 0; i < inst->getNumSrcRegs(); ++i) {
                const StinkyRegister& oldSrc = inst->getSrcReg(i);
                if (!oldSrc.isRegister()) continue;

                StinkyRegister newSrc = resolveMappedSrc(oldSrc);
                if (newSrc != oldSrc) {
                    inst->setSrcReg(i, newSrc);
                }
            }

            for (const StinkyRegister& dst : inst->getDestRegs()) {
                invalidateByDef(dst);
            }

            if (!isEligibleMov(*inst)) continue;
            const StinkyRegister& dst = inst->getDestReg(0);
            const StinkyRegister& src = inst->getSrcReg(0);
            if (dst != src) moveMap[dst] = src;
        }

        std::vector<StinkyInstruction*> toErase;
        for (size_t i = 0; i < instructions.size(); ++i) {
            StinkyInstruction* inst = instructions[i];
            if (!isEligibleMov(*inst)) continue;

            const StinkyRegister& dst = inst->getDestReg(0);
            const StinkyRegister& src = inst->getSrcReg(0);

            // Identity move has no semantic effect.
            if (dst == src) {
                toErase.push_back(inst);
                continue;
            }

            bool redefined = false;
            for (size_t j = i + 1; j < instructions.size(); ++j) {
                StinkyInstruction* later = instructions[j];
                if (hasSrcOverlap(*later, dst)) {
                    break;
                }
                if (hasDestOverlap(*later, dst)) {
                    redefined = true;
                    break;
                }
            }

            // Erase mov only when dst is redefined before any later use in this BB.
            // Otherwise keep it conservatively (it may still be live-out).
            if (redefined) {
                toErase.push_back(inst);
            }
        }

        for (StinkyInstruction* inst : toErase) {
            inst->erase();
        }
    }
};

char AsmMovePropagationPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createAsmMovePropagationPass() {
    return std::make_unique<AsmMovePropagationPassImpl>();
}
}  // namespace stinkytofu
