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

#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"  // For StinkyRegister definition
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/logical/LogicalInstructionData.hpp"
#include "stinkytofu/ir/logical/LogicalOpcode.hpp"

namespace stinkytofu {

/**
 * @brief Base class for all high-level IR instructions (architecture-independent)
 *
 * This is the LLVM-style high-level IR. Instructions at this level are:
 * - Architecture-independent (no arch-specific mnemonics)
 * - Logical operations (may be composite)
 * - Will be lowered to StinkyAsmIR (StinkyInstruction) by passes
 *
 * Design philosophy:
 * - Simple enum-based design (no inheritance hierarchy)
 * - Holds operands, modifiers, and metadata
 * - Fast opcode-based pattern matching (no virtual dispatch or string comparison)
 *
 * New enum-based design (Phase 2 refactoring):
 * - Single class with logical::Opcode field (no subclasses)
 * - Factory functions for type-safe construction: VAddF32(dest, src0, src1)
 * - Backward compatible with existing code
 */
class LogicalInstruction : public IRBase {
   private:
    friend class IRBase;

    logical::Opcode opcode_;  ///< Opcode identifying the instruction type
    void* specialData_;       ///< Special data for MFMA/Label/IntrinsicCall (owned)
    InstFlagSet flags_;       ///< Instruction flags (matches Assembly IR flags)

   protected:
    /// Protected so subclasses (IntrinsicCall, GenericIRInstruction) can delegate; use createIR for
    /// creation.
    explicit LogicalInstruction(logical::Opcode opcode)
        : IRBase(IRType::LogicalIR), opcode_(opcode), specialData_(nullptr) {}

    LogicalInstruction()
        : IRBase(IRType::LogicalIR), opcode_(logical::UNKNOWN), specialData_(nullptr) {}

    ~LogicalInstruction() override {
        freeSpecialData();
    }

   public:
    /// When true, this instruction is owned externally (e.g. by Python); safeErase() will only
    /// remove, not delete.
    bool ownedExternally = false;

    std::vector<StinkyRegister> dests;  ///< Destination registers
    std::vector<StinkyRegister> srcs;   ///< Source registers
    std::string comment;                ///< Optional comment

    // Instruction modifiers (optional, only used by specific instructions)
    std::optional<DPPModifiers> dpp;    ///< Data parallel processing modifier
    std::optional<SDWAModifiers> sdwa;  ///< Sub-dword addressing modifier
    std::optional<DSModifiers> ds;      ///< LDS/GDS modifier

    /// LLVM-style casting support
    static bool classof(const IRBase* ir) {
        return ir->getType() == IRType::LogicalIR;
    }

    /**
     * @brief Get the logical name of this instruction (for lowering)
     * @return Logical name like "VAddF32", "DSLoadB64"
     */
    virtual const char* getLogicalName() const {
        return logical::getOpcodeName(opcode_);
    }

    /**
     * @brief Get the opcode of this instruction (for pattern matching)
     * @return Logical IR opcode enum value
     *
     * Enables fast integer comparison in pattern matching:
     *   if (inst->getOpcode() == logical::VAddF32) { ... }
     *   switch (inst->getOpcode()) { case logical::VMaxF32: ... }
     */
    logical::Opcode getOpcode() const {
        return opcode_;
    }

    /**
     * @brief Check if instruction has a specific flag set
     * @param flag The flag to check (e.g., IF_DSRead, IF_MFMA, IF_Commutative)
     * @return true if the flag is set
     *
     * Enables fast flag-based checks (same as Assembly IR):
     *   if (inst->is(IF_DSRead)) { ... }
     *   if (inst->is(IF_Commutative)) { ... }
     */
    bool is(InstFlag flag) const {
        return flags_.test(flag);
    }

    /**
     * @brief Get the instruction flags
     * @return Reference to the flag set
     */
    const InstFlagSet& getFlags() const {
        return flags_;
    }

    /**
     * @brief Set instruction flags (for tablegen-generated factory functions)
     */
    void setFlags(const InstFlagSet& flags) {
        flags_ = flags;
    }

    /**
     * @brief Remove this logical instruction from its parent block; delete only if not
     * ownedExternally. When ownedExternally is true (e.g. Python-owned), only remove() is called so
     * the external owner retains the object. Otherwise behaves like erase().
     */
    void safeErase() {
        if (ownedExternally)
            remove();
        else
            erase();
    }

    /**
     * @brief Check if this is a composite instruction that needs expansion
     * @return true if composite (expands to multiple instructions)
     */
    bool isComposite() const;  // Auto-generated implementation

    /**
     * @brief Check if this instruction is commutative (src0 and src1 can be swapped)
     * @return true if commutative (e.g., add, mul, min, max)
     *
     * Commutative instructions allow pattern matching to work regardless of
     * operand order. For example, a pattern for `a + 1.0` will also match `1.0 + a`.
     */
    bool isCommutative() const {
        return is(IF_Commutative);
    }

    /**
     * @brief Accessors for AMDGPU instruction modifiers
     *
     * These accessors provide access to instruction-specific modifiers.
     * These are instruction semantics (how the instruction operates),
     * not assembly scheduling hints.
     *
     * Examples:
     * - DPP: Data parallel processing (row shuffle, broadcast)
     * - SDWA: Sub-dword operand selection
     * - DS: LDS/GDS offsets and flags
     */
    std::optional<DPPModifiers> getDPP() const {
        return dpp;
    }

    std::optional<SDWAModifiers> getSDWA() const {
        return sdwa;
    }

    std::optional<DSModifiers> getDS() const {
        return ds;
    }

    /**
     * @brief Dump instruction to output stream
     */
    void dump(std::ostream& out) const override {
        // Special handling for Label instructions
        if (opcode_ == logical::Label) {
            auto* labelData = asLabel();
            if (labelData) {
                out << labelData->labelName << ":";
                return;
            }
        }

        // Regular instructions
        out << getLogicalName() << " (IR)";
        if (!comment.empty()) out << "  // " << comment;
    }

    // ====================================================================
    // Special Data Accessors (for pure enum design)
    // ====================================================================
    //
    // Pure enum design with specialData_ for metadata-rich instructions:
    //
    // Regular instructions (e.g., VAddF32):
    //   - Only use opcode_ and dests/srcs vectors
    //   - specialData_ is nullptr
    //
    // Special instructions (e.g., MFMA, Label):
    //   - Use opcode_ to identify instruction
    //   - Store registers in dests/srcs vectors (like regular instructions)
    //   - Store metadata in specialData_ (dimensions, types, flags, etc.)
    //
    // Usage example:
    //   auto* inst = MFMA("bf16", "f32", 16, 16, 4, 1, false, acc, a, b);
    //
    //   // Later in a pass:
    //   if (inst->getOpcode() == logical::MFMA) {
    //       auto* mfmaData = inst->asMFMA();
    //       int m = mfmaData->m;
    //       std::string type = mfmaData->instType;
    //
    //       // Registers are still in standard locations:
    //       StinkyRegister dest = inst->dests[0];
    //       StinkyRegister src0 = inst->srcs[0];
    //   }
    //
    // ====================================================================

    /**
     * @brief Set special data (takes ownership)
     */
    void setSpecialData(void* data) {
        freeSpecialData();  // Free any existing data
        specialData_ = data;
    }

    /**
     * @brief Get MFMA data (returns nullptr if not MFMA)
     */
    MFMAData* asMFMA() {
        return (opcode_ == logical::MFMA) ? static_cast<MFMAData*>(specialData_) : nullptr;
    }

    const MFMAData* asMFMA() const {
        return (opcode_ == logical::MFMA) ? static_cast<const MFMAData*>(specialData_) : nullptr;
    }

    /**
     * @brief Get MXMFMA data (returns nullptr if not MXMFMA)
     */
    MXMFMAData* asMXMFMA() {
        return (opcode_ == logical::MXMFMA) ? static_cast<MXMFMAData*>(specialData_) : nullptr;
    }

    const MXMFMAData* asMXMFMA() const {
        return (opcode_ == logical::MXMFMA) ? static_cast<const MXMFMAData*>(specialData_)
                                            : nullptr;
    }

    /**
     * @brief Get SMFMA data (returns nullptr if not SMFMA)
     */
    SMFMAData* asSMFMA() {
        return (opcode_ == logical::SMFMA) ? static_cast<SMFMAData*>(specialData_) : nullptr;
    }

    const SMFMAData* asSMFMA() const {
        return (opcode_ == logical::SMFMA) ? static_cast<const SMFMAData*>(specialData_) : nullptr;
    }

    /**
     * @brief Get Label data (returns nullptr if not Label)
     */
    LogicalLabelData* asLabel() {
        return (opcode_ == logical::Label) ? static_cast<LogicalLabelData*>(specialData_) : nullptr;
    }

    const LogicalLabelData* asLabel() const {
        return (opcode_ == logical::Label) ? static_cast<const LogicalLabelData*>(specialData_)
                                           : nullptr;
    }

    /**
     * @brief Get IntrinsicCall data (returns nullptr if not IntrinsicCall)
     */
    IntrinsicCallData* asIntrinsicCall() {
        return (opcode_ == logical::IntrinsicCall) ? static_cast<IntrinsicCallData*>(specialData_)
                                                   : nullptr;
    }

    const IntrinsicCallData* asIntrinsicCall() const {
        return (opcode_ == logical::IntrinsicCall)
                   ? static_cast<const IntrinsicCallData*>(specialData_)
                   : nullptr;
    }

   private:
    /**
     * @brief Free special data based on opcode
     */
    void freeSpecialData() {
        if (!specialData_) return;

        switch (opcode_) {
            case logical::MFMA:
                delete static_cast<MFMAData*>(specialData_);
                break;
            case logical::MXMFMA:
                delete static_cast<MXMFMAData*>(specialData_);
                break;
            case logical::SMFMA:
                delete static_cast<SMFMAData*>(specialData_);
                break;
            case logical::Label:
                delete static_cast<LogicalLabelData*>(specialData_);
                break;
            case logical::IntrinsicCall:
                delete static_cast<IntrinsicCallData*>(specialData_);
                break;
            default:
                // No special data for regular instructions
                break;
        }

        specialData_ = nullptr;
    }
};

inline std::shared_ptr<LogicalInstruction> makeLogicalInstructionShared(LogicalInstruction* p) {
    p->ownedExternally = true;
    return std::shared_ptr<LogicalInstruction>(p, [](LogicalInstruction* x) {
        // Don't use safeErase() here; we want to delete the instruction because shared_ptr is the
        // only owner.
        if (x) x->erase();
    });
}

}  // namespace stinkytofu

// ========================================================================
// High-Level IR Instruction Definitions (Auto-generated by TableGen)
// ========================================================================
//
// All IR instruction classes are now generated from a single source of truth
// in tools/tablegen/GenLogicalIR.cpp. This ensures consistency across:
// - IR class definitions
// - Builder method signatures
// - Mnemonic mappings for ToStinkyAsmPass
//
// To add a new instruction:
// 1. Update getIRInstructions() in tools/tablegen/GenLogicalIR.cpp
// 2. Rebuild (TableGen runs automatically)
// 3. Done! IR class, builder, and mnemonics are all generated
//
// Note: The generated file has its own namespace stinkytofu block
#include "stinkytofu/ir/logical/LogicalInstructions_generated.hpp"
