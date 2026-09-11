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
#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"
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
    // VALU co-issue window (bit i = VALU may co-issue at cycle i after issue).
    // Resolved from hwInstDesc, then possibly overridden by matrix data format.
    uint16_t coIssueWindow;

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
    std::optional<AttachedSSA> attachedSSA_;

    StinkyInstruction(const HwInstDesc* mcid)
        : IRBase(IRType::StinkyTofu),
          hwInstDesc(mcid),
          issueCycles(mcid->issue),
          latencyCycles(mcid->latency),
          coIssueWindow(mcid->coIssueWindow) {}

    ~StinkyInstruction() override {
        clearAttachedSSA();
    }

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
            coIssueWindow = newDesc->coIssueWindow;
            resolveMatrixFmtOverrides();
        }
    }

    // Apply matrix-data-format-keyed hardware overrides (issue/latency cycles
    // and co-issue window) once the MatrixFmtModifiers is known. These are
    // hardware properties of the instruction, so they are baked onto the
    // instruction's own fields and every pass reads the corrected values.
    void resolveMatrixFmtOverrides();

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

    bool hasAttachedSSA() const {
        return attachedSSA_.has_value();
    }
    void attachSSA(AttachedSSA ssa);
    void clearAttachedSSA();

    size_t getNumSSAResults() const;
    StinkySSAValue* getSSAResult(size_t i) const;
    size_t getNumSSAOperands() const;
    StinkyOpOperand* getSSAOperand(size_t i);
    const StinkyOpOperand* getSSAOperand(size_t i) const;
    StinkySSAValue* getSSAOperandValue(size_t i) const;
    void setSSAOperandValue(size_t i, StinkySSAValue* v);

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

        // Copy issue/latency cycles and co-issue window
        cloned->issueCycles = issueCycles;
        cloned->latencyCycles = latencyCycles;
        cloned->coIssueWindow = coIssueWindow;

        // Deep copy modifiers via virtual clone() (TypedModifier implements it per type).
        for (const auto& mod : modifiers) {
            cloned->modifiers.push_back(mod->clone());
        }

        // Note: users/sources and attached SSA are intentionally NOT copied.
        // Def-use chains and SSA use-lists must be rebuilt or reattached.

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
            GFX::FENCE, GFX::FENCE, 0, 0, 0, 0, "FENCE", makeFlagSet({InstFlag::IF_HasSideEffect})};
        return create(&fenceMCID);
    }

    /// Creates a pseudo marker for the ASM placement of a function body.
    /// It records where the named function should appear in the final linear ASM stream.
    /// e.g.
    ///   st.func @entry() {
    ///     ^label_ASM_End:
    ///       FUNCTION_ASM_PLACEMENT_MARKER "label_Activation_Relu_VW1"
    ///   }
    ///
    ///   st.func @label_Activation_Relu_VW1() {
    ///     ...
    ///   }
    ///
    ///   This means that the function body will be placed at the ^label_ASM_End position in the
    ///   final linear ASM stream.
    StinkyInstruction* createFunctionAsmPlacementMarker(const std::string& functionName) {
        static const HwInstDesc functionAsmPlacementMarkerMCID{
            GFX::FUNCTION_ASM_PLACEMENT_MARKER,
            GFX::FUNCTION_ASM_PLACEMENT_MARKER,
            0,
            0,
            0,
            0,
            "FUNCTION_ASM_PLACEMENT_MARKER",
            makeFlagSet({InstFlag::IF_HasSideEffect})};
        StinkyInstruction* inst = create(&functionAsmPlacementMarkerMCID);
        inst->addModifier<LabelData>(LabelData{functionName});
        return inst;
    }

    /// Opaque pseudo-instruction that groups a narrow-exec-write..full-mask-reset span
    /// so the DAG scheduler treats it as one atomic node. Own descriptor carries no
    /// IF_HasSideEffect; hasSideEffect() below still inherits it from children.
    StinkyInstruction* createExecMaskGroup(IRBase* insertBefore) {
        static const HwInstDesc execGroupMCID{GFX::EXEC_GROUP, GFX::EXEC_GROUP, 0, 0, 0, 0,
                                              "EXEC_GROUP",    makeFlagSet({})};
        return create(&execGroupMCID, insertBefore);
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

inline bool isGlobalStoreAsyncFromLds(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_GLOBALStoreAsyncFromLds);
}

inline bool isGLOBALAtomic(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_GLOBALAtomic);
}

inline bool isGlobalPrefetch(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_GLOBALPrefetch);
}

inline bool isGLOBAL(const StinkyInstruction& inst) {
    return isGLOBALLoad(inst) || isGLOBALStore(inst);
}

inline bool isGLOBALOrAtomic(const StinkyInstruction& inst) {
    return isGLOBAL(inst) || isGLOBALAtomic(inst);
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

inline bool isVmemTexStore(const StinkyInstruction& inst) {
    return isMUBUFStore(inst) || isFLATStore(inst) || isGLOBALStore(inst) ||
           isGlobalStoreAsyncFromLds(inst);
}

/// Check if instruction is a scheduling fence pseudo-instruction.
/// Fences emit no assembly but carry MemTokenData ordering constraints.
inline bool isFence(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::FENCE;
}

/// Check if instruction is a function ASM placement marker.
inline bool isFunctionAsmPlacementMarker(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::FUNCTION_ASM_PLACEMENT_MARKER;
}

inline bool isExecMaskGroup(const StinkyInstruction& inst) {
    return inst.getUnifiedOpcode() == GFX::EXEC_GROUP;
}

/// Check if instruction is a pseudo instruction (LABEL, PHI, FENCE, or
/// FUNCTION_ASM_PLACEMENT_MARKER) that should be skipped for def-use chain
/// processing of "real" instructions.
inline bool isPseudoInst(const StinkyInstruction* inst) {
    return inst->getUnifiedOpcode() == GFX::LABEL || inst->getUnifiedOpcode() == GFX::PHI ||
           inst->getUnifiedOpcode() == GFX::FENCE ||
           inst->getUnifiedOpcode() == GFX::FUNCTION_ASM_PLACEMENT_MARKER;
}

inline bool isGlobalMemLoad(const StinkyInstruction& inst) {
    return isMUBUFLoad(inst) || isFLATLoad(inst) || isGLOBALLoad(inst) || isSMemLoad(inst);
}

inline bool isGlobalMemAtomic(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_SMemAtomic) || isMUBUFAtomic(inst) || isFLATAtomic(inst) ||
           isGLOBALAtomic(inst);
}

inline bool isGlobalMemStore(const StinkyInstruction& inst) {
    return isSMemStore(inst) || isFLATStore(inst) || isMUBUFStore(inst) || isGLOBALStore(inst) ||
           isGlobalStoreAsyncFromLds(inst);
}

/// A destination register is implicit (not printed) when it was added solely
/// for dependency tracking. Shared between the assembly emitter (decides
/// whether to print `th:TH_ATOMIC_RETURN`) and the waitcnt dataflow (decides
/// whether an atomic's destination is a trackable value) so both agree on
/// exactly the same "does this atomic return a value" answer.
inline bool isImplicitDest(const StinkyRegister& reg, const StinkyInstruction& inst) {
    if (reg.dataType != StinkyRegister::Type::Register) return false;

    RegType t = reg.reg.type;

    if (t == RegType::SCC) {
        assert(inst.is(InstFlag::IF_ImplicitWriteSCC) &&
               "SCC should always be an implicit dest or src");
        return true;
    }

    if ((t == RegType::EXEC || t == RegType::EXEC_LO || t == RegType::EXEC_HI) &&
        inst.is(InstFlag::IF_ImplicitWriteEXEC)) {
        return true;
    }

    return false;
}

/// True iff `inst` is a returning MUBUF/FLAT/GLOBAL atomic -- i.e. one whose
/// destination is a real (non-pseudo, non-implicit) register that a later
/// instruction can consume, as opposed to a fire-and-forget atomic with no
/// usable result. Scalar-memory atomics (IF_SMemAtomic) signal their return
/// via `glc`, not `th:`, and are excluded here; they are also not currently
/// reachable through any StinkyTofu-enabled architecture.
///
/// NOTE: this class of instruction is deliberately NOT modeled through
/// MemTokenData/pseudo-register dependency edges the way LDS ops are --
/// its destination is a plain, real register, so ordinary SSA def-use
/// already connects a returning atomic to its consumers. `classifyMemOp`
/// (see WaitDataflow.cpp) still needs to bucket it into a counter so the
/// waitcnt dataflow can compute -- and safely regenerate -- the wait that
/// guards that register.
inline bool isReturningAtomic(const StinkyInstruction& inst) {
    if (!isMUBUFAtomic(inst) && !isFLATAtomic(inst) && !isGLOBALAtomic(inst)) return false;

    for (const StinkyRegister& d : inst.getDestRegs()) {
        if (!isPseudoReg(d) && !isImplicitDest(d, inst)) return true;
    }
    return false;
}

inline bool isTensorLoad(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_TENSORLoadToLds);
}

// Async memory ops tracked by ASYNCcnt (s_wait_asynccnt). Shared FIFO counter
// across the whole async family; extend this predicate as async loads /
// cluster-async / ds_atomic_async_barrier_arrive are added.
inline bool isAsyncMemOp(const StinkyInstruction& inst) {
    return isGlobalStoreAsyncFromLds(inst);
}

// Vector memory outside the DS and FLAT classes.
inline bool isVmemTex(const StinkyInstruction& inst) {
    return isMUBUFLoad(inst) || isMUBUFStore(inst) || isMUBUFAtomic(inst) ||
           isGLOBALOrAtomic(inst) || isAsyncMemOp(inst) || isGlobalPrefetch(inst) ||
           isTensorLoad(inst);
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

/// True when the instruction ends the current Function: kernel exit (`s_endpgm`)
/// or a register-target `s_setpc_b64` return. Annotated `s_setpc_b64` is a
/// branch to a known label, not a function return.
inline bool isEndOfFunction(const StinkyInstruction& inst) {
    if (inst.getUnifiedOpcode() == GFX::s_endpgm) return true;
    return inst.getUnifiedOpcode() == GFX::s_setpc_b64 && inst.getModifier<LabelData>() == nullptr;
}

inline bool isIndirectBranch(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_IndirectBranch);
}

/// True for call-like control transfers (IF_Call, e.g. s_swappc_b64), not branches.
inline bool isCall(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_Call);
}

/// Any intra-function control effect that is not ordinary dataflow: branches or calls.
inline bool isControlTransfer(const StinkyInstruction& inst) {
    return isBranch(inst) || isCall(inst);
}

/// Possible callee entry labels for a call site (`s_swappc_b64` with optional
/// `CallTargetData` from the rocisa producer). Empty when unknown or omitted.
/// This is for call-graph / scheduling analysis only; it is not a CFG successor list.
inline std::vector<std::string> getCallTargets(const StinkyInstruction& inst) {
    if (!isCall(inst)) return {};
    if (const auto* meta = inst.getModifier<CallTargetData>()) {
        return meta->callees;
    }
    return {};
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
//   - Calls (`IF_Call`, e.g. `s_swappc_b64`) are not branches; use getCallTargets().
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

// ---------------------------------------------------------------------------
// Packed wait immediate (s_wait_loadcnt_dscnt / s_wait_storecnt_dscnt)
//
// Both opcodes carry two counters in one SIMM16:
//   DS  = SIMM16[5:0]   -- wait for dscnt <= N
//   MEM = SIMM16[13:8]  -- wait for loadcnt <= N or storecnt <= N, whichever
//                          the opcode selects
// Bits [7:6] and [15:14] are reserved. Every counter named here is 6 bits
// wide, so the field max doubles as the field mask.
// ---------------------------------------------------------------------------

/// Largest value a 6-bit dependency-counter field can hold.
constexpr int kWaitCntMax = 63;

/// SIMM16 bits that belong to a counter field; the rest are reserved.
constexpr int kPackedWaitCntMask = (kWaitCntMax << 8) | kWaitCntMax;

inline bool isValidPackedWaitCnt(int simm16) {
    return (simm16 & ~kPackedWaitCntMask) == 0;
}

/// Saturate a counter to what its field can hold. Upstream producers tolerate
/// over-counting (TensileLite's dscnt is a static estimate, and rocisa likewise
/// clamps when printing), so this is not an error: a smaller count is always
/// the stricter wait. Only a negative value is a bug -- it means the "-1 means
/// absent" sentinel escaped the caller's guard.
inline int clampToWaitCntField(int count) {
    assert(count >= 0 && "s_wait immediate: negative counter reached encoding");
    return std::min(count, kWaitCntMax);
}

/// Build the SIMM16 for a packed wait. Saturating each half is what keeps an
/// over-large count from bleeding into the neighbouring field.
inline int packMemDsWaitCnt(int memCount, int dsCount) {
    return (clampToWaitCntField(memCount) << 8) | clampToWaitCntField(dsCount);
}

inline int unpackMemWaitCnt(int simm16) {
    assert(isValidPackedWaitCnt(simm16) &&
           "packed s_wait immediate: reserved bits SIMM16[15:14]/[7:6] are set");
    return (simm16 >> 8) & kWaitCntMax;
}

inline int unpackDsWaitCnt(int simm16) {
    assert(isValidPackedWaitCnt(simm16) &&
           "packed s_wait immediate: reserved bits SIMM16[15:14]/[7:6] are set");
    return simm16 & kWaitCntMax;
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

    if (isCall(inst)) return true;

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
    if ((isGlobalMemStore(inst) && !isGlobalStoreAsyncFromLds(inst)) || isBranch(inst) ||
        isCall(inst) || isWaitCnt(inst) || isHasSideEffect(inst))
        return true;
    if ((isBarrier(inst) || isTensorLoad(inst) || isDSRead(inst) || isDSWrite(inst) ||
         isGlobalStoreAsyncFromLds(inst)) &&
        !hasLdsPseudoRegs(inst))
        return true;
    if (isExecMaskGroup(inst)) {
        if (const auto* groupData = inst.getModifier<ExecGroupData>()) {
            for (const StinkyInstruction* child : groupData->children)
                if (hasSideEffect(*child)) return true;
        }
    }
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

/// Check if instruction is a Tensor-LUT op (v_perm_pk16*).
inline bool isTensorLUT(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_TensorLUT);
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
