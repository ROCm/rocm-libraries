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

bool hasVop3pSourceModifier(const StinkyInstruction& inst, size_t srcIdx) {
    const VOP3PModifiers* vop3p = inst.getModifier<VOP3PModifiers>();
    if (!vop3p) return false;

    auto hasNonZeroAt = [srcIdx](const std::vector<int>& values) {
        return srcIdx < values.size() && values[srcIdx] != 0;
    };
    return hasNonZeroAt(vop3p->op_sel) || hasNonZeroAt(vop3p->op_sel_hi) ||
           hasNonZeroAt(vop3p->byte_sel);
}

bool hasSdwaSourceModifier(const StinkyInstruction& inst, size_t srcIdx) {
    const SDWAModifiers* sdwa = inst.getModifier<SDWAModifiers>();
    if (!sdwa) return false;

    if (srcIdx == 0) return sdwa->src0_sel != SDWAModifiers::SelectBit::SEL_NONE;
    if (srcIdx == 1) return sdwa->src1_sel != SDWAModifiers::SelectBit::SEL_NONE;
    return false;
}

bool hasTrue16SourceModifier(const StinkyInstruction& inst, size_t srcIdx) {
    const True16Modifiers* true16 = inst.getModifier<True16Modifiers>();
    if (!true16) return false;
    if (srcIdx >= true16->getSrcCount()) return false;
    return true16->getSrc(srcIdx) != HighBitSel::NONE;
}

bool hasMfmaSourceModifier(const StinkyInstruction& inst, size_t srcIdx) {
    const MFMAModifiers* mfma = inst.getModifier<MFMAModifiers>();
    if (!mfma || srcIdx > 2 || mfma->negBits.numSrcs == 0) return false;
    return mfma->negBits.negLo[srcIdx] != 0 || mfma->negBits.negHi[srcIdx] != 0;
}

bool hasInstructionSourceModifier(const StinkyInstruction& inst, size_t srcIdx) {
    return hasVop3SourceModifier(inst, srcIdx) || hasVop3pSourceModifier(inst, srcIdx) ||
           hasSdwaSourceModifier(inst, srcIdx) || hasTrue16SourceModifier(inst, srcIdx) ||
           hasMfmaSourceModifier(inst, srcIdx);
}

bool isSpecialControlReg(const StinkyRegister& reg) {
    if (!reg.isRegister()) return false;
    switch (reg.reg.type) {
        case RegType::SCC:
        case RegType::VCC:
        case RegType::VCC_LO:
        case RegType::VCC_HI:
        case RegType::EXEC:
        case RegType::EXEC_LO:
        case RegType::EXEC_HI:
            return true;
        default:
            return false;
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
    // Never optimize mov when either endpoint is exec/vcc/scc-style control state.
    // - special src: prevents propagating snapshots (e.g. "save exec") into later uses.
    // - special dst: prevents deleting/mutating explicit control-state writes.
    if (isSpecialControlReg(src) || isSpecialControlReg(dst)) return false;

    return true;
}

enum class RegClass { Vgpr, Sgpr, Other };

RegClass classifyReg(const StinkyRegister& reg) {
    if (!reg.isRegister()) return RegClass::Other;
    if (isSpecialControlReg(reg)) return RegClass::Other;

    switch (reg.reg.type) {
        case RegType::V:
            return RegClass::Vgpr;
        case RegType::S:
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

struct MoveMapKey {
    RegType type;
    uint32_t idx;
    uint16_t num;
    int16_t offset;

    bool operator==(const MoveMapKey& other) const noexcept {
        return type == other.type && idx == other.idx && num == other.num && offset == other.offset;
    }
};

struct MoveMapKeyHash {
    size_t operator()(const MoveMapKey& key) const noexcept {
        const size_t typeHash = std::hash<int>{}(static_cast<int>(key.type));
        const size_t idxHash = std::hash<uint32_t>{}(key.idx);
        const size_t numHash = std::hash<uint16_t>{}(key.num);
        const size_t offsetHash = std::hash<int16_t>{}(key.offset);
        return typeHash ^ (idxHash << 1) ^ (numHash << 2) ^ (offsetHash << 3);
    }
};

MoveMapKey toMoveMapKey(const StinkyRegister& reg) {
    return {reg.reg.type, reg.reg.idx, reg.reg.num, reg.reg.offset};
}

bool hasSameRegisterIdentity(const StinkyRegister& lhs, const StinkyRegister& rhs) {
    if (!lhs.isRegister() || !rhs.isRegister()) return lhs == rhs;
    return toMoveMapKey(lhs) == toMoveMapKey(rhs);
}

bool overlapsWithKey(const MoveMapKey& key, const StinkyRegister& reg) {
    if (!reg.isRegister() || key.type != reg.reg.type) return false;
    const uint32_t keyBegin = key.idx;
    const uint32_t keyEnd = key.idx + key.num;
    const uint32_t regBegin = reg.reg.idx;
    const uint32_t regEnd = reg.reg.idx + reg.reg.num;
    return !(keyEnd <= regBegin || regEnd <= keyBegin);
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

        std::unordered_map<MoveMapKey, StinkyRegister, MoveMapKeyHash> moveMap;

        auto resolveMappedSrc = [&moveMap](const StinkyRegister& reg) {
            StinkyRegister resolved = reg;
            // moveMap is invalidated on defs, so chains should not form cycles.
            while (true) {
                if (!resolved.isRegister()) break;
                auto it = moveMap.find(toMoveMapKey(resolved));
                if (it == moveMap.end()) break;
                resolved = it->second;
            }
            return resolved;
        };

        auto invalidateByDef = [&moveMap](const StinkyRegister& defReg) {
            if (!defReg.isRegister()) return;
            for (auto it = moveMap.begin(); it != moveMap.end();) {
                if (overlapsWithKey(it->first, defReg) || it->second.isOverlap(defReg)) {
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
            // Treat call as a hard barrier: do not propagate mappings across callee boundary.
            if (isCall(*inst)) {
                moveMap.clear();
                continue;
            }

            for (size_t i = 0; i < inst->getNumSrcRegs(); ++i) {
                const StinkyRegister& oldSrc = inst->getSrcReg(i);
                if (!oldSrc.isRegister()) continue;
                // Skip source operands that carry modifiers
                // (inline reg modifiers or VOP3 source modifiers).
                if (hasRegisterSourceModifier(oldSrc) || hasInstructionSourceModifier(*inst, i))
                    continue;

                StinkyRegister newSrc = resolveMappedSrc(oldSrc);
                if (hasRegisterSourceModifier(newSrc)) continue;
                if (!hasSameRegisterIdentity(newSrc, oldSrc) &&
                    isSamePropagatableClass(oldSrc, newSrc)) {
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
            if (!hasSameRegisterIdentity(dst, src) && isSamePropagatableClass(dst, src)) {
                moveMap[toMoveMapKey(dst)] = src;
            }
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
            // Calls are liveness barriers for this local cleanup:
            // never infer mov-deadness across a call boundary.
            if (isCall(*inst)) {
                nextEvents.clear();
                continue;
            }

            if (isEligibleMov(*inst)) {
                const StinkyRegister& dst = inst->getDestReg(0);
                const StinkyRegister& src = inst->getSrcReg(0);

                // Identity move has no semantic effect.
                if (hasSameRegisterIdentity(dst, src)) {
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
