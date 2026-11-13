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

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>  // TODO: don't use iostream.
#include <ostream>

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {
//----------------------------------------------------------------------
// StinkyRegister implementation
//----------------------------------------------------------------------
void StinkyRegister::dump(std::ostream& out, const std::string& prefix) const {
    out << prefix;
    AsmPrinter printer(out, AsmPrinterOptions());
    printer.print(*this);
}

void StinkyRegister::dump() const {
    dump(std::cerr);
}

//----------------------------------------------------------------------
// StinkyInstruction implementation
//----------------------------------------------------------------------
void StinkyInstruction::dump() const {
    dump(std::cerr);
}

void StinkyInstruction::dump(std::ostream& out) const {
    AsmPrinter printer(out);
    printer.print(*this);
}

void StinkyInstruction::attachSSA(AttachedSSA ssa) {
    clearAttachedSSA();
    attachedSSA_ = std::move(ssa);
    for (size_t i = 0; i < attachedSSA_->results.size(); ++i) {
        StinkySSAValue* value = attachedSSA_->results[i];
        if (value != nullptr) value->bindDef(this, static_cast<uint16_t>(i));
    }
    for (size_t i = 0; i < attachedSSA_->operands.size(); ++i) {
        if (attachedSSA_->operands[i])
            attachedSSA_->operands[i]->bindOwner(this, static_cast<uint16_t>(i));
    }
}

void StinkyInstruction::clearAttachedSSA() {
    if (!attachedSSA_) return;
    for (StinkySSAValue* value : attachedSSA_->results) {
        if (value != nullptr && value->defOp() == this) value->unbindDef();
    }
    attachedSSA_.reset();
}

size_t StinkyInstruction::getNumSSAResults() const {
    return attachedSSA_ ? attachedSSA_->results.size() : 0;
}

StinkySSAValue* StinkyInstruction::getSSAResult(size_t i) const {
    assert(attachedSSA_ && "getSSAResult requires attached SSA");
    return attachedSSA_->results.at(i);
}

size_t StinkyInstruction::getNumSSAOperands() const {
    return attachedSSA_ ? attachedSSA_->operands.size() : 0;
}

StinkyOpOperand* StinkyInstruction::getSSAOperand(size_t i) {
    assert(attachedSSA_ && "getSSAOperand requires attached SSA");
    return attachedSSA_->operands.at(i).get();
}

const StinkyOpOperand* StinkyInstruction::getSSAOperand(size_t i) const {
    assert(attachedSSA_ && "getSSAOperand requires attached SSA");
    return attachedSSA_->operands.at(i).get();
}

StinkySSAValue* StinkyInstruction::getSSAOperandValue(size_t i) const {
    const StinkyOpOperand* operand = getSSAOperand(i);
    return operand != nullptr ? operand->value() : nullptr;
}

void StinkyInstruction::setSSAOperandValue(size_t i, StinkySSAValue* v) {
    StinkyOpOperand* operand = getSSAOperand(i);
    assert(operand != nullptr);
    operand->setValue(v);
}

void StinkyInstruction::resolveMatrixFmtOverrides() {
    if (!hwInstDesc) return;
    const bool hasCost = !hwInstDesc->matrixFmtCostOverrides.empty();
    const bool hasCoIssue = !hwInstDesc->matrixFmtCoIssueOverrides.empty();
    if (!hasCost && !hasCoIssue) return;

    const auto* fmt = getModifier<MatrixFmtModifiers>();
    if (!fmt) return;
    const auto a = static_cast<uint8_t>(fmt->fmtA);
    const auto b = static_cast<uint8_t>(fmt->fmtB);

    for (const auto& ov : hwInstDesc->matrixFmtCostOverrides) {
        if (ov.fmtA == a && ov.fmtB == b) {
            issueCycles = ov.issue;
            latencyCycles = ov.latency;
            break;
        }
    }
    for (const auto& ov : hwInstDesc->matrixFmtCoIssueOverrides) {
        if (ov.fmtA == a && ov.fmtB == b) {
            coIssueWindow = ov.coIssueWindow;
            break;
        }
    }
}

//----------------------------------------------------------------------
// AsmIRBuilder implementation
//----------------------------------------------------------------------
StinkyInstruction* AsmIRBuilder::createLabel(const std::string& label, uint16_t alignment) {
    static const HwInstDesc labelMCID{
        GFX::LABEL, GFX::LABEL, 0, 0, 0, 0, "LABEL", makeFlagSet({InstFlag::IF_HasSideEffect})};

    StinkyInstruction* labelInst = create(&labelMCID);
    labelInst->addModifier<LabelData>(LabelData{label, alignment});
    return labelInst;
}

StinkyInstruction* AsmIRBuilder::createPhi(RegType type, unsigned regIdx, IRBase* insertPt) {
    static const HwInstDesc phiMCID{
        GFX::PHI, GFX::PHI, 0, 0, 0, 0, "PHI", makeFlagSet({InstFlag::IF_HasSideEffect})};

    const size_t numPreds = bb->getPredecessors().size();

    if (insertPt == nullptr) insertPt = bb->empty() ? nullptr : &*bb->begin();

    if (insertPt != nullptr) {
        auto it = IRList::iterator(insertPt);
        assert(
            (it == bb->begin() ||
             cast<StinkyInstruction>(std::prev(it).getNodePtr())->getUnifiedOpcode() == GFX::PHI) &&
            "insertPt must be at block begin or immediately after a PHI");
    }

    StinkyInstruction* phi = insertPt ? create(&phiMCID, insertPt) : create(&phiMCID);
    phi->addDestReg(StinkyRegister(type, regIdx, 1));

    for (size_t i = 0; i < numPreds; ++i) phi->addSrcReg(StinkyRegister(0));

    return phi;
}

static const HwInstDesc* getMCIDTable(GfxArchID arch) {
    const auto* archInfo =
        ArchHelper::getInstance().getArchInfo(arch);  // Ensure ArchHelper is initialized
    return archInfo->getMCIDTable();  // Call the virtual method to ensure proper initialization
}

const HwInstDesc* getMCIDByUOp(GFX unifiedOpcode, GfxArchID arch) {
    IsaOpcode isaOpcode = GFX::INVALID;

    const auto* archInfo =
        ArchHelper::getInstance().getArchInfo(arch);  // Ensure ArchHelper is initialized
    isaOpcode = archInfo->getIsaOpcode(unifiedOpcode);

    if (isaOpcode == GFX::INVALID) {
        // Return nullptr if instruction not supported on this architecture
        // Callers use this to probe architecture support
        return nullptr;
    }
    return &getMCIDTable(arch)[isaOpcode];
}

const HwInstDesc* getMCIDByIsaOp(IsaOpcode isaOpcode, GfxArchID arch) {
    return &getMCIDTable(arch)[isaOpcode];
}

uint16_t getMnemonicToIsaOpcode(const std::string& mnemonic, GfxArchID arch) {
    auto get = [&](const std::unordered_map<std::string, uint16_t>& map,
                   const std::string& mnemonic) -> uint16_t {
        auto it = map.find(mnemonic);
        if (it == map.end()) {
            // Keep this check in release builds too: returning it->second on a
            // past-the-end iterator is UB and previously caused a segfault when
            // an unmapped mnemonic reached the emitter.
            std::cerr << "Error: No ISA opcode found for mnemonic " << mnemonic << " in arch "
                      << getArchName(arch) << "\n";
            return GFX::INVALID;
        }
        return it->second;
    };

    const auto* archInfo =
        ArchHelper::getInstance().getArchInfo(arch);  // Ensure ArchHelper is initialized
    return get(archInfo->getMnemonicToIsaOpcodeMap(), mnemonic);
}

//----------------------------------------------------------------------
// StinkyInstruction utilities
//----------------------------------------------------------------------

uint32_t getBytesPerGlobalLoad(const StinkyInstruction& inst) {
    auto op = inst.getUnifiedOpcode();

    // FIXME: Should we add metadata for the buffer load inst in tablegen to
    //        query the bytes per load?
    switch (op) {
        case GFX::buffer_load_u8:
            return 1;
        case GFX::buffer_load_d16_u8:
        case GFX::buffer_load_d16_hi_u8:
        case GFX::buffer_load_u16:
        case GFX::buffer_load_d16_hi_b16:
            return 2;
        case GFX::buffer_load_b32:
            return 4;
        case GFX::buffer_load_b64:
            return 8;
        case GFX::buffer_load_b128:
            return 16;

        default:
            assert(false && "Calling getBytesPerGlobalLoad but input is not GlobalRead");
            return 0;
    }
}
}  // namespace stinkytofu
