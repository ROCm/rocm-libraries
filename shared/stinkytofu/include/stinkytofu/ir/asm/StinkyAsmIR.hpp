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
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/core/IRBuilder.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
#define GET_ISAINFO_UNIFIED_OPCODES
#include "hardware/gfxIsa.inc"

// Represents a single assembly instruction.
struct STINKYTOFU_EXPORT StinkyInstruction : public IRBase {
    friend class IRBase;
    friend class AsmIRBuilder;
    friend class DefUseChainUpdater;
    friend class DefUseChainBuilder;

    const HwInstDesc* hwInstDesc;
    int issueCycles;
    int latencyCycles;

   private:
    // Def-use chain:
    // instructions that USE the value DEFINED by this instruction
    // (consumers of this instruction's output).
    std::vector<StinkyInstruction*> users;

    // instructions that DEFINE the values USED by this instruction
    // (producers of this instruction's operands).
    std::vector<StinkyInstruction*> sources;

    // Modifiers are extra bits/fields in the instruction encoding that
    // tweak how the operation is performed, without needing a completely
    // different opcode.
    std::vector<std::unique_ptr<Modifier>> modifiers;

    // Implicit special registers (SCC, VCC, EXEC) are added via HW flags
    // (IF_ImplicitReadSCC, IF_ImplicitWriteSCC, etc.) during lowering; see
    // addRegistersToInstruction() in ToStinkyTofuUtils.cpp.
    std::vector<StinkyRegister> destRegs;
    std::vector<StinkyRegister> srcRegs;

    StinkyInstruction(const HwInstDesc* mcid)
        : IRBase(IRType::StinkyTofu),
          hwInstDesc(mcid),
          issueCycles(mcid->issue),
          latencyCycles(mcid->latency) {}

    ~StinkyInstruction() override = default;

   public:
    void addSrcReg(const StinkyRegister& srcReg) {
        srcRegs.push_back(srcReg);
    }

    void addDestReg(const StinkyRegister& destReg) {
        destRegs.push_back(destReg);
    }

    /// Get destination registers (read-only)
    const std::vector<StinkyRegister>& getDestRegs() const {
        return destRegs;
    }

    /// Get source registers (read-only)
    const std::vector<StinkyRegister>& getSrcRegs() const {
        return srcRegs;
    }

    /// Get number of destination registers
    size_t getNumDestRegs() const {
        return destRegs.size();
    }

    /// Get number of source registers
    size_t getNumSrcRegs() const {
        return srcRegs.size();
    }

    /// Get instructions that use the value defined by this instruction (def-use chain)
    const std::vector<StinkyInstruction*>& getUsers() const {
        return users;
    }

    /// Get instructions that define the operands used by this instruction (def-use chain)
    const std::vector<StinkyInstruction*>& getSources() const {
        return sources;
    }

    /// Get destination register by index
    const StinkyRegister& getDestReg(size_t idx) const {
        return destRegs.at(idx);
    }

    /// Get source register by index
    const StinkyRegister& getSrcReg(size_t idx) const {
        return srcRegs.at(idx);
    }

    uint16_t getISAOpcode() const {
        return hwInstDesc->isaOpcode;
    }

    uint16_t getUnifiedOpcode() const {
        return hwInstDesc->unifiedOpcode;
    }

    const HwInstDesc* getHwInstDesc() const {
        return hwInstDesc;
    }

    void updateHwInstDesc(const HwInstDesc* newDesc) {
        if (newDesc) {
            hwInstDesc = newDesc;
            issueCycles = newDesc->issue;
            latencyCycles = newDesc->latency;
        }
    }

    bool is(InstFlag flag) const {
        return hwInstDesc->has(flag);
    }

    template <class ModType>
    void addModifier(const ModType& mod) {
        static_assert(std::is_base_of_v<Modifier, ModType>, "ModType must derive from Modifier");
        modifiers.push_back(std::make_unique<ModType>(mod));
    }

    template <class ModType>
    const ModType* getModifier() const {
        for (const std::unique_ptr<Modifier>& mod : modifiers)
            if (mod->getType() == ModType::Type) return static_cast<const ModType*>(mod.get());
        return nullptr;
    }

    template <class ModType>
    ModType* getModifier() {
        for (std::unique_ptr<Modifier>& mod : modifiers)
            if (mod->getType() == ModType::Type) return static_cast<ModType*>(mod.get());
        return nullptr;
    }

    const std::vector<std::unique_ptr<Modifier>>& getModifiers() const {
        return modifiers;
    }

    void dump(std::ostream& out) const override;

    void dump() const;

    /// Set all source registers.
    void setSrcRegs(const std::vector<StinkyRegister>& newSrcRegs) {
        srcRegs = newSrcRegs;
    }

    /// Set all destination registers.
    void setDestRegs(const std::vector<StinkyRegister>& newDestRegs) {
        destRegs = newDestRegs;
    }

    /// Set a specific source register by index.
    void setSrcReg(size_t idx, const StinkyRegister& srcReg) {
        srcRegs.at(idx) = srcReg;
    }

    /// Set a specific destination register by index.
    void setDestReg(size_t idx, const StinkyRegister& destReg) {
        destRegs.at(idx) = destReg;
    }

    /// Resize source registers array (for pattern rewriting).
    void resizeSrcRegs(size_t size) {
        srcRegs.resize(size);
    }

    /// Resize destination registers array (for pattern rewriting).
    void resizeDestRegs(size_t size) {
        destRegs.resize(size);
    }

    /**
     * @brief Clone this instruction (deep copy)
     *
     * Creates a new instruction with the same descriptor, registers, and modifiers.
     *
     * Notes:
     * - Modifiers are deep copied using copy constructors (works for POD modifiers)
     * - users/sources are NOT copied (dependency tracking should be rebuilt)
     *
     * This is needed because StinkyInstruction inherits from IntrusiveListNode
     * which has a deleted copy constructor.
     *
     * @return New instruction (caller owns the pointer)
     */
    StinkyInstruction* clone() const override {
        StinkyInstruction* cloned = IRBase::createIR<StinkyInstruction>(hwInstDesc);

        // Copy register lists
        cloned->destRegs = destRegs;
        cloned->srcRegs = srcRegs;

        // Copy issue/latency cycles
        cloned->issueCycles = issueCycles;
        cloned->latencyCycles = latencyCycles;

        // Deep copy modifiers via virtual clone() (TypedModifier implements it per type).
        for (const auto& mod : modifiers) {
            cloned->modifiers.push_back(mod->clone());
        }

        // Note: users/sources are intentionally NOT copied
        // These are dependency tracking and should be rebuilt if needed

        return cloned;
    }

    static bool classof(const IRBase* ir) {
        return ir->getType() == IRType::StinkyTofu;
    }
};

/// Builder for assembly-level IR.
/// In passes, create StinkyInstruction only through this builder.
class STINKYTOFU_EXPORT AsmIRBuilder : public IRBuilder {
   public:
    const GfxArchID arch;

    StinkyInstruction* create(const HwInstDesc* mcid, IRBase* insertBefore = nullptr) {
        assert(mcid != nullptr &&
               "Cannot create instruction with null descriptor - instruction not supported "
               "on this architecture");

        if (insertBefore == nullptr) return createIR<StinkyInstruction>(mcid);

        return createIR<StinkyInstruction>(insertBefore, mcid);
    }

    /// Creates a LABEL instruction. TODO: remove when basic-block labels are supported.
    StinkyInstruction* createLabel(const std::string& label, uint16_t alignment = 1);

    /// PHI instruction properties:
    /// - Must appear at the beginning of a basic block
    /// - Defines a single register (DWORD only)
    /// - One incoming value per predecessor
    ///   * For a basic block B:
    ///     . If B has N predecessors, each PHI in B must have exactly N incoming
    ///       pairs, and each predecessor appears exactly once
    /// - Operand is nullptr if undefined on that edge
    ///
    /// PHI instruction format:
    /// def = PHI(pred0_def, pred1_def, ..., predN_def)

    /// Creates a scheduling fence pseudo-instruction (emits no assembly).
    /// Acts as a hard region boundary in the DAG scheduler — nothing can be
    /// reordered across it.
    StinkyInstruction* createFence() {
        static const HwInstDesc fenceMCID{
            GFX::FENCE, GFX::FENCE, 0, 0, 0, "FENCE", makeFlagSet({InstFlag::IF_HasSideEffect})};
        return create(&fenceMCID);
    }

    /// Creates and inserts a PHI instruction at the beginning of the block.
    /// The PHI defines one DWORD register and has one placeholder srcReg per
    /// predecessor. sources and users are NOT initialized — the caller
    /// (PhiPlacement / BuildDefUseChain) is responsible for linking chains.
    ///
    /// If \p insertPt is non-null, the PHI is inserted before \p insertPt
    /// instead of before bb->begin(). This allows callers to build PHIs in
    /// a stable order by passing a fixed anchor point.
    StinkyInstruction* createPhi(RegType type, unsigned regIdx, IRBase* insertPt = nullptr);

    AsmIRBuilder(BasicBlock& bb, GfxArchID arch) : IRBuilder(bb), arch(arch) {}
};

STINKYTOFU_EXPORT const HwInstDesc* getMCIDByUOp(GFX unifiedOpcode, GfxArchID arch);
STINKYTOFU_EXPORT const HwInstDesc* getMCIDByIsaOp(IsaOpcode isaOpcode, GfxArchID arch);
STINKYTOFU_EXPORT uint16_t getMnemonicToIsaOpcode(const std::string& mnemonic, GfxArchID arch);

// Processing StinkyTofu IR.
class StinkyInstPass : public Pass {};

inline StinkyInstruction& getStinkyInst(IRList::iterator it) {
    return *cast<StinkyInstruction>(it.getNodePtr());
}

inline StinkyInstruction& getStinkyInst(IRList::reverse_iterator it) {
    return *cast<StinkyInstruction>(it.getNodePtr());
}

//----------------------------------------------------------------------
// StinkyInstruction utilities
//----------------------------------------------------------------------
uint32_t getBytesPerGlobalLoad(const StinkyInstruction& inst);

inline bool isMUBUFLoad(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_MUBUFLoad);
}

inline bool isMUBUFStore(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_MUBUFStore);
}

inline bool isMUBUFAtomic(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_MUBUFAtomic);
}

inline bool isFLATLoad(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_FLATLoad);
}

inline bool isFLATStore(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_FLATStore);
}

inline bool isFLATAtomic(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_FLATAtomic);
}

inline bool isGLOBALLoad(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_GLOBALLoad);
}

inline bool isGLOBALStore(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_GLOBALStore);
}

inline bool isSMemLoad(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_SMemLoad);
}

inline bool isSMemStore(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_SMemStore);
}

inline bool isBufferMemLoad(const StinkyInstruction& inst) {
    return isMUBUFLoad(inst) || isFLATLoad(inst) || isGLOBALLoad(inst);
}

inline bool isBufferMemStore(const StinkyInstruction& inst) {
    return isMUBUFStore(inst) || isFLATStore(inst) || isGLOBALStore(inst);
}

/// Check if instruction is a scheduling fence pseudo-instruction.
/// Fences emit no assembly but carry MemTokenData ordering constraints.
inline bool isFence(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::FENCE;
}

/// Check if instruction is a pseudo instruction (LABEL, PHI, or FENCE) that should be
/// skipped for def-use chain processing of "real" instructions.
inline bool isPseudoInst(const StinkyInstruction* inst) {
    return inst->getUnifiedOpcode() == GFX::LABEL || inst->getUnifiedOpcode() == GFX::PHI ||
           inst->getUnifiedOpcode() == GFX::FENCE;
}

inline bool isGlobalMemLoad(const StinkyInstruction& inst) {
    return isMUBUFLoad(inst) || isFLATLoad(inst) || isGLOBALLoad(inst) || isSMemLoad(inst);
}

inline bool isGlobalMemAtomic(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_SMemAtomic) || isMUBUFAtomic(inst) || isFLATAtomic(inst);
}

inline bool isGlobalMemStore(const StinkyInstruction& inst) {
    return isSMemStore(inst) || isFLATStore(inst) || isMUBUFStore(inst) || isGLOBALStore(inst);
}

inline bool isTensorLoad(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_TENSORLoadToLds);
}

inline bool isDSRead(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_DSRead);
}

inline bool isDSWrite(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_DSStore);
}

inline bool isDSAtomic(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_DSAtomic);
}

inline bool isBarrier(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_Barrier);
}

inline bool isBarrierSignal(const StinkyInstruction& inst) {
    return inst.getHwInstDesc()->unifiedOpcode == GFX::s_barrier_signal;
}

inline bool isBarrierWait(const StinkyInstruction& inst) {
    return inst.getHwInstDesc()->unifiedOpcode == GFX::s_barrier_wait;
}

/// Returns true if a split barrier (signal/wait) uses all-wave mode (-1).
/// Non-(-1) values indicate workgroup-sync mode where signal and wait may be
/// in different basic blocks (e.g. if wave==0: signal; all waves: wait).
inline bool isSplitBarrierAllWave(const StinkyInstruction& inst) {
    assert((isBarrierSignal(inst) || isBarrierWait(inst)) &&
           "Expected s_barrier_signal or s_barrier_wait");
    if (inst.getSrcRegs().empty()) return false;
    const StinkyRegister& src = inst.getSrcRegs()[0];
    return src.dataType == StinkyRegister::Type::LiteralInt && src.getLiteralInt() == -1;
}

inline bool isBranch(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_Branch);
}

inline bool isConditionalBranch(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_ConditionalBranch);
}

inline bool isUnconditionalBranch(const StinkyInstruction& inst) {
    return isBranch(inst) && !isConditionalBranch(inst);
}

inline bool isIndirectBranch(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_IndirectBranch);
}

// Structural call predicate. Only s_swappc_b64 is a call mnemonic in the tree.
inline bool isCall(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::s_swappc_b64;
}

// Label names of basic-block targets for \p given branch instruction.
//
// At most one target is returned today. Switch / multi-way branch semantics
// (several labels from one terminator) are not modeled.
//
// Resolution (first match wins):
//   - Not a branch → {}
//   - LabelData{label} → {label} (rocisa converter or LongBranchLoweringPass)
//   - IF_IndirectBranch without LabelData → {}
//   - First src is LiteralString → {that string} (raw .s s_branch / s_cbranch_*)
//   - Otherwise → {}
inline std::vector<std::string> getBranchTargets(const StinkyInstruction& inst) {
    if (!isBranch(inst)) return {};

    if (const auto* label = inst.getModifier<LabelData>()) {
        return {label->label};
    }

    if (isIndirectBranch(inst)) return {};

    if (inst.getSrcRegs().empty()) return {};
    const StinkyRegister& targetReg = inst.getSrcRegs()[0];
    if (targetReg.dataType != StinkyRegister::Type::LiteralString) return {};
    return {targetReg.getLiteralString()};
}

// Single-target shim. Returns the first label from getBranchTargets(), or "" if
// the instruction has no statically-known branch target label.
inline std::string getBranchTarget(const StinkyInstruction& inst) {
    auto targets = getBranchTargets(inst);
    return targets.empty() ? std::string{} : targets.front();
}

inline bool isWaitCnt(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_WaitCnt);
}

inline bool isMFMA(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_MFMA);
}

inline bool isSMFMA(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_SMFMA);
}

inline bool isWMMA(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_WMMA);
}

inline bool isSWMMA(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_SWMMA);
}

inline bool isMXWMMA(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_MXWMMA);
}

inline bool isMatrixInstruction(const StinkyInstruction& inst) {
    return isMFMA(inst) || isSMFMA(inst) || isWMMA(inst) || isSWMMA(inst) || isMXWMMA(inst);
}

inline bool isHasSideEffect(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_HasSideEffect);
}

inline bool isLabel(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::LABEL;
}

/// Determines if an instruction must be preserved and cannot be eliminated.
///
/// This is a comprehensive check that covers all instructions with observable effects,
/// including memory operations, control flow, barriers, and instructions explicitly
/// marked with side effects.
///
/// This function is distinct from isHasSideEffect() which only checks the
/// IF_HasSideEffect flag. This function performs a broader classification suitable
/// for optimization passes like dead code elimination.
///
/// @param inst The instruction to check
/// @return true if the instruction must be preserved, false if it can be eliminated
inline bool mustPreserveInstruction(const StinkyInstruction& inst) {
    // Memory operations (loads/stores)
    if (isGlobalMemLoad(inst) || isGlobalMemStore(inst)) return true;

    if (isDSRead(inst) || isDSWrite(inst)) return true;

    if (isGlobalMemAtomic(inst)) return true;

    if (isTensorLoad(inst)) return true;

    // Control flow
    if (isBranch(inst)) return true;

    // Barriers and synchronization
    if (isBarrier(inst)) return true;

    // Instructions explicitly marked with side effects
    if (isHasSideEffect(inst)) return true;

    return false;
}

/// Returns true if the instruction has LDS pseudo-register operands,
/// indicating MemTokenData has been assigned and ordering is enforced
/// by the DAG via def-use edges.
inline bool hasLdsPseudoRegs(const StinkyInstruction& inst) {
    for (const StinkyRegister& r : inst.getSrcRegs())
        if (r.isRegister() && r.reg.type == RegType::LDS) return true;
    for (const StinkyRegister& r : inst.getDestRegs())
        if (r.isRegister() && r.reg.type == RegType::LDS) return true;
    return false;
}

/// Returns true if the instruction forces the DAG scheduler to cut a new
/// region.  This covers true side effects (stores, branches, waits) and
/// memory ops that lack MemTokenData (LDS pseudo-registers), where the
/// scheduler has no dependency edges to prove reordering is safe.
inline bool hasSideEffect(const StinkyInstruction& inst) {
    if (!inst.getHwInstDesc()) return false;
    if (isGlobalMemStore(inst) || isBranch(inst) || isWaitCnt(inst) || isHasSideEffect(inst))
        return true;
    if ((isBarrier(inst) || isTensorLoad(inst) || isDSRead(inst) || isDSWrite(inst)) &&
        !hasLdsPseudoRegs(inst))
        return true;
    return false;
}

/// Check if instruction is a vector ALU instruction (v_*)
/// Excludes transcendental instructions which are classified separately
inline bool isVectorALU(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_VALU) && !inst.is(InstFlag::IF_Transcendental);
}

/// Check if instruction is a transcendental instruction
/// Includes: v_s_*, v_exp_*, v_log_*, v_rcp_*, v_rsq_*, v_sqrt_*
inline bool isTranscendental(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_Transcendental);
}

/// Check if instruction is a scalar ALU instruction (s_*)
/// Excludes: control flow, memory operations, waitcnt, barrier, delay_alu
inline bool isScalarALU(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_SALU);
}

/// Check if instruction is an XDL WMMA/SWMMAC instruction.
/// Excludes FP32-input WMMA (v_wmma_f32_16x16x4_f32).
inline bool isXDLWMMA(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_WMMA_XDL);
}

/// Check if instruction is a 64-bit transcendental.
/// Includes v_rcp_f64, v_rsq_f64, v_sqrt_f64.
inline bool isTrans64(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_Trans64);
}

/// Check if instruction is a double-precision MACC VALU.
/// Includes v_add_f64, v_fma_f64, f64 compares (v_cmp_lt_f64), v_cvt_u32_f64.
/// Excludes f64 transcendentals (v_rcp_f64) — those are Trans64/TRANS.
inline bool isDPMACC(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_DPMACC);
}

}  // namespace stinkytofu
