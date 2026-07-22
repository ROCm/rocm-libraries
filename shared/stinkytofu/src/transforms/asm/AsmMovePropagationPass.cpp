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
#include "stinkytofu/transforms/asm/AsmMovePropagationPass.hpp"

#include <unordered_map>
#include <unordered_set>
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
    if (mustPreserveInstruction(inst)) return false;
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
    int runOnBasicBlock(BasicBlock& bb) {
        std::vector<StinkyInstruction*> instructions;
        for (IRBase& node : bb) {
            if (node.getType() == IRBase::IRType::StinkyTofu) {
                instructions.push_back(cast<StinkyInstruction>(&node));
            }
        }

        int numChanged = 0;
        std::unordered_map<StinkyRegister, StinkyRegister> moveMap;

        auto resolveMappedSrc = [&moveMap](const StinkyRegister& reg) {
            if (!reg.isRegister()) return reg;

            StinkyRegister resolved = reg;
            for (int depth = 0; depth < 8; ++depth) {
                auto it = moveMap.find(resolved);
                if (it == moveMap.end()) break;
                if (it->second == resolved) break;
                resolved = it->second;
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

        for (StinkyInstruction* inst : instructions) {
            for (size_t i = 0; i < inst->getNumSrcRegs(); ++i) {
                const StinkyRegister& oldSrc = inst->getSrcReg(i);
                if (!oldSrc.isRegister()) continue;

                StinkyRegister newSrc = resolveMappedSrc(oldSrc);
                if (newSrc != oldSrc) {
                    inst->setSrcReg(i, newSrc);
                    numChanged++;
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

        std::unordered_set<StinkyInstruction*> toErase;
        for (size_t i = 0; i < instructions.size(); ++i) {
            StinkyInstruction* inst = instructions[i];
            if (!isEligibleMov(*inst)) continue;

            const StinkyRegister& dst = inst->getDestReg(0);
            const StinkyRegister& src = inst->getSrcReg(0);

            // Identity move has no semantic effect.
            if (dst == src) {
                toErase.insert(inst);
                continue;
            }

            bool usedBeforeRedef = false;
            bool redefined = false;
            for (size_t j = i + 1; j < instructions.size(); ++j) {
                StinkyInstruction* later = instructions[j];
                if (hasSrcOverlap(*later, dst)) {
                    usedBeforeRedef = true;
                    break;
                }
                if (hasDestOverlap(*later, dst)) {
                    redefined = true;
                    break;
                }
            }

            if (redefined && !usedBeforeRedef) {
                toErase.insert(inst);
            }
        }

        for (StinkyInstruction* inst : toErase) {
            inst->erase();
            numChanged++;
        }

        return numChanged;
    }
};

char AsmMovePropagationPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createAsmMovePropagationPass() {
    return std::make_unique<AsmMovePropagationPassImpl>();
}
}  // namespace stinkytofu
