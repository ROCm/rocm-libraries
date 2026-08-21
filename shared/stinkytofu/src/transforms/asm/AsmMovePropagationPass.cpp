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

bool hasRegisterSourceModifier(const StinkyRegister& reg) {
    return reg.isRegister() && (reg.reg.isMinus || reg.reg.isAbs);
}

bool hasVop3SourceModifier(const StinkyInstruction& inst, size_t srcIdx) {
    const VOP3Modifiers* vop3 = inst.getModifier<VOP3Modifiers>();
    if (!vop3 || srcIdx > 2) return false;
    switch (srcIdx) {  // NOLINT(bugprone-switch-missing-default-case)
        case 0:
            return vop3->neg_src0 || vop3->abs_src0;
        case 1:
            return vop3->neg_src1 || vop3->abs_src1;
        case 2:
            return vop3->neg_src2 || vop3->abs_src2;
    }
    return false;
}

bool isSafeMovDstReg(const StinkyRegister& reg) {
    if (!reg.isRegister()) return false;
    switch (reg.reg.type) {
        case RegType::SCC:
        case RegType::VCC:
        case RegType::VCC_LO:
        case RegType::VCC_HI:
        case RegType::EXEC:
        case RegType::EXEC_LO:
        case RegType::EXEC_HI:
            return false;
        default:
            return true;
    }
}

bool isEligibleMov(const StinkyInstruction& inst) {
    if (!isSupportedMov(inst)) return false;
    if (inst.getDestRegs().size() != 1 || inst.getSrcRegs().size() != 1) return false;

    const StinkyRegister& dst = inst.getDestReg(0);
    const StinkyRegister& src = inst.getSrcReg(0);
    if (!dst.isRegister() || !src.isRegister()) return false;
    // Keep mov with source modifiers untouched.
    if (hasRegisterSourceModifier(dst) || hasRegisterSourceModifier(src)) return false;
    if (dst.reg.num != 1 || src.reg.num != 1) return false;
    if (isPseudoReg(dst) || isPseudoReg(src)) return false;
    // Never optimize mov that writes exec/vcc/scc, because they affect lane-mask/condition state
    // and are implicitly consumed by later instructions.
    if (!isSafeMovDstReg(dst)) return false;

    return true;
}

enum class RegClass { Vgpr, Sgpr, Other };

RegClass classifyReg(const StinkyRegister& reg) {
    if (!reg.isRegister()) return RegClass::Other;

    switch (reg.reg.type) {
        case RegType::V:
            return RegClass::Vgpr;
        case RegType::S:
        case RegType::SCC:
        case RegType::VCC:
        case RegType::VCC_LO:
        case RegType::VCC_HI:
        case RegType::EXEC:
        case RegType::EXEC_LO:
        case RegType::EXEC_HI:
            return RegClass::Sgpr;
        default:
            return RegClass::Other;
    }
}

bool isSamePropagatableClass(const StinkyRegister& dst, const StinkyRegister& src) {
    const RegClass dstClass = classifyReg(dst);
    const RegClass srcClass = classifyReg(src);
    return dstClass != RegClass::Other && dstClass == srcClass;
}

struct RegLaneKey {
    RegType type;
    uint32_t idx;

    bool operator==(const RegLaneKey& other) const noexcept {
        return type == other.type && idx == other.idx;
    }
};

struct RegLaneKeyHash {
    size_t operator()(const RegLaneKey& key) const noexcept {
        const size_t typeHash = std::hash<int>{}(static_cast<int>(key.type));
        const size_t idxHash = std::hash<uint32_t>{}(key.idx);
        return typeHash ^ (idxHash << 1);
    }
};

enum class NextEvent { None, Use, Def };

void markRegisterLanes(const StinkyRegister& reg, NextEvent event,
                       std::unordered_map<RegLaneKey, NextEvent, RegLaneKeyHash>& nextEvents) {
    if (!reg.isRegister()) return;
    for (uint32_t lane = 0; lane < reg.reg.num; ++lane) {
        nextEvents[{reg.reg.type, reg.reg.idx + lane}] = event;
    }
}

NextEvent getNextEvent(
    const StinkyRegister& reg,
    const std::unordered_map<RegLaneKey, NextEvent, RegLaneKeyHash>& nextEvents) {
    auto it = nextEvents.find({reg.reg.type, reg.reg.idx});
    if (it == nextEvents.end()) return NextEvent::None;
    return it->second;
}

struct MovePropStats {
    uint64_t inputVmov = 0;
    uint64_t inputSmov = 0;
    uint64_t erasedVmov = 0;
    uint64_t erasedSmov = 0;
    uint64_t erasedIdentityVmov = 0;
    uint64_t erasedIdentitySmov = 0;
    uint64_t erasedRedefinedVmov = 0;
    uint64_t erasedRedefinedSmov = 0;

    void add(const MovePropStats& other) {
        inputVmov += other.inputVmov;
        inputSmov += other.inputSmov;
        erasedVmov += other.erasedVmov;
        erasedSmov += other.erasedSmov;
        erasedIdentityVmov += other.erasedIdentityVmov;
        erasedIdentitySmov += other.erasedIdentitySmov;
        erasedRedefinedVmov += other.erasedRedefinedVmov;
        erasedRedefinedSmov += other.erasedRedefinedSmov;
    }
};

void countInputMovStat(const StinkyInstruction& inst, MovePropStats& stats) {
    if (inst.getUnifiedOpcode() == GFX::v_mov_b32) {
        stats.inputVmov++;
    } else if (inst.getUnifiedOpcode() == GFX::s_mov_b32) {
        stats.inputSmov++;
    }
}

void countErasedMovStat(const StinkyInstruction& inst, MovePropStats& stats, bool identity) {
    const bool isVmov = inst.getUnifiedOpcode() == GFX::v_mov_b32;
    if (isVmov) {
        stats.erasedVmov++;
        if (identity)
            stats.erasedIdentityVmov++;
        else
            stats.erasedRedefinedVmov++;
    } else {
        stats.erasedSmov++;
        if (identity)
            stats.erasedIdentitySmov++;
        else
            stats.erasedRedefinedSmov++;
    }
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
        MovePropStats totalStats;
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            totalStats.add(runOnBasicBlock(bb));
        }
        func.setMetaData("AsmMovePropagationPass.inputVmov", totalStats.inputVmov);
        func.setMetaData("AsmMovePropagationPass.inputSmov", totalStats.inputSmov);
        func.setMetaData("AsmMovePropagationPass.erasedVmov", totalStats.erasedVmov);
        func.setMetaData("AsmMovePropagationPass.erasedSmov", totalStats.erasedSmov);
        func.setMetaData("AsmMovePropagationPass.erasedIdentityVmov",
                         totalStats.erasedIdentityVmov);
        func.setMetaData("AsmMovePropagationPass.erasedIdentitySmov",
                         totalStats.erasedIdentitySmov);
        func.setMetaData("AsmMovePropagationPass.erasedRedefinedVmov",
                         totalStats.erasedRedefinedVmov);
        func.setMetaData("AsmMovePropagationPass.erasedRedefinedSmov",
                         totalStats.erasedRedefinedSmov);
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
    MovePropStats runOnBasicBlock(BasicBlock& bb) {
        MovePropStats stats;
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

        // Phase A - forward propagation loop.
        // 1) rewrite current instruction sources using mappings from earlier instructions
        // 2) invalidate mappings killed by current instruction defs
        // 3) if current instruction is an eligible mov, add its new mapping
        for (StinkyInstruction* inst : instructions) {
            for (size_t i = 0; i < inst->getNumSrcRegs(); ++i) {
                const StinkyRegister& oldSrc = inst->getSrcReg(i);
                if (!oldSrc.isRegister()) continue;
                // Skip source operands that carry modifiers
                // (inline reg modifiers or VOP3 source modifiers).
                if (hasRegisterSourceModifier(oldSrc) || hasVop3SourceModifier(*inst, i)) continue;

                StinkyRegister newSrc = resolveMappedSrc(oldSrc);
                if (hasRegisterSourceModifier(newSrc)) continue;
                if (newSrc != oldSrc && isSamePropagatableClass(oldSrc, newSrc)) {
                    inst->setSrcReg(i, newSrc);
                }
            }

            for (const StinkyRegister& dst : inst->getDestRegs()) {
                invalidateByDef(dst);
            }

            if (!isEligibleMov(*inst)) continue;
            countInputMovStat(*inst, stats);
            const StinkyRegister& dst = inst->getDestReg(0);
            const StinkyRegister& src = inst->getSrcReg(0);
            if (dst != src && isSamePropagatableClass(dst, src)) moveMap[dst] = src;
        }

        // Phase B - mov cleanup loop (O(N) backward next-event scan).
        // For each lane, track the first event after current instruction:
        //   - Use means mov must be kept.
        //   - Def means mov can be erased.
        //   - None means keep conservatively (potential live-out).
        std::vector<StinkyInstruction*> toErase;
        std::unordered_map<RegLaneKey, NextEvent, RegLaneKeyHash> nextEvents;
        for (size_t i = instructions.size(); i-- > 0;) {
            StinkyInstruction* inst = instructions[i];
            if (isEligibleMov(*inst)) {
                const StinkyRegister& dst = inst->getDestReg(0);
                const StinkyRegister& src = inst->getSrcReg(0);

                // Identity move has no semantic effect.
                if (dst == src) {
                    countErasedMovStat(*inst, stats, /*identity=*/true);
                    toErase.push_back(inst);
                } else if (getNextEvent(dst, nextEvents) == NextEvent::Def) {
                    // Erase mov only when dst is redefined before any later use in this BB.
                    // Otherwise keep it conservatively (it may still be live-out).
                    countErasedMovStat(*inst, stats, /*identity=*/false);
                    toErase.push_back(inst);
                }
            }

            // For same-instruction read/write, keep "use-before-def" semantics by
            // applying defs first, then sources overwrite as Use.
            for (const StinkyRegister& defReg : inst->getDestRegs()) {
                markRegisterLanes(defReg, NextEvent::Def, nextEvents);
            }
            for (const StinkyRegister& useReg : inst->getSrcRegs()) {
                markRegisterLanes(useReg, NextEvent::Use, nextEvents);
            }
        }

        for (StinkyInstruction* inst : toErase) {
            inst->erase();
        }
        return stats;
    }
};

char AsmMovePropagationPassImpl::ID = 0;

}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createAsmMovePropagationPass() {
    return std::make_unique<AsmMovePropagationPassImpl>();
}
}  // namespace stinkytofu
