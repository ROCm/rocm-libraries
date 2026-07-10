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

#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"

#include <iomanip>
#include <sstream>

#include "ModifierSerializer.hpp"
#include "stinkytofu/hardware/HwRegHelpers.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
//----------------------------------------------------------------------
// AsmPrinter implementation
//----------------------------------------------------------------------
void AsmPrinter::print(const StinkyRegister& reg) {
    printRegister(reg);
}

void AsmPrinter::printRegister(const StinkyRegister& reg) {
    switch (reg.dataType) {
        case StinkyRegister::Type::Register: {
            std::string prefix = regTypeToString(reg.reg.type);
            if (reg.reg.type == RegType::AGPR) prefix = "acc";

            // Don't use symbolic name for now
            // if(reg.hasSymbolicName())
            // {
            //     if(reg.reg.num == 1)
            //         os << prefix << "[" << reg.getSymbolicName() << "]";
            //     else
            //         os << prefix << "[" << reg.getSymbolicName() << ":" << reg.getSymbolicName()
            //            << "+" << (reg.reg.num - 1) << "]";
            // }
            if (reg.reg.num == 1)
                os << prefix << reg.reg.idx;
            else
                os << prefix << "[" << reg.reg.idx << ":" << (reg.reg.idx + reg.reg.num - 1) << "]";
            break;
        }
        case StinkyRegister::Type::LiteralInt:
            os << reg.getLiteralInt();
            break;
        case StinkyRegister::Type::LiteralDouble:
            os << std::fixed << std::setprecision(6) << reg.getLiteralDouble();
            break;
        case StinkyRegister::Type::LiteralString:
            os << reg.getLiteralString();
            break;
        case StinkyRegister::Type::HwReg:
            HwReg::printOperand(os, reg);
            break;
        case StinkyRegister::Type::Invalid:
            os << "<invalid>";
            break;
    }
}

void AsmPrinter::print(const StinkyInstruction& inst) {
    printInstruction(inst);
}

void AsmPrinter::print(const AsmDirective& directive) {
    printDirective(directive);
}

void AsmPrinter::print(const Function& function) {
    os << "st.func @" << function.getName() << "() {\n";
    size_t index = 0;
    for (const BasicBlock& bb : function) {
        printBlock(bb, index);
        ++index;
    }
    os << "}\n";
}

void AsmPrinter::printBlock(const BasicBlock& bb, size_t blockIndex) {
    std::string blockId =
        bb.getLabel().empty() ? ("bb" + std::to_string(blockIndex)) : bb.getLabel();
    os << "^" << blockId << ":\n";
    for (const IRBase& ir : bb) printIR(ir);
    printSuccessorsLine(bb);
}

void AsmPrinter::printIR(const IRBase& ir) {
    switch (ir.getType()) {
        case IRBase::IRType::StinkyTofu: {
            if (const StinkyInstruction* inst = dyn_cast<StinkyInstruction>(&ir))
                printInstruction(*inst);
            else {
                // other StinkyTofu: indent and dump
                os << std::string(static_cast<size_t>(options.indent), ' ');
                ir.dump(os);
                os << "\n";
            }
            break;
        }
        case IRBase::IRType::StinkyAsmDirective:
            if (const AsmDirective* directive = dyn_cast<AsmDirective>(&ir))
                printDirective(*directive);
            else {
                os << std::string(static_cast<size_t>(options.indent), ' ');
                ir.dump(os);
                os << "\n";
            }
            break;
        case IRBase::IRType::LogicalIR:
            os << std::string(static_cast<size_t>(options.indent), ' ');
            ir.dump(os);
            os << "\n";
            break;
    }
}

void AsmPrinter::printInstruction(const StinkyInstruction& inst) {
    // labels are block boundaries; do not print LABEL as instruction
    if (inst.getUnifiedOpcode() == GFX::LABEL) return;

    os << std::string(static_cast<size_t>(options.indent), ' ');

    if (!inst.getDestRegs().empty()) {
        for (size_t i = 0; i < inst.getDestRegs().size(); ++i) {
            if (i > 0) {
                os << ", ";
            }
            printRegister(inst.getDestRegs()[i]);
        }
        os << " = ";
    }

    const std::string irNamespace = "st";
    os << "\"" << irNamespace << "." << inst.getHwInstDesc()->mnemonic << "\"";

    os << "(";
    for (size_t i = 0; i < inst.getSrcRegs().size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        printRegister(inst.getSrcRegs()[i]);
    }
    os << ")";

    // Attributes: issueCycles, latencyCycles, then mod.X = { ... } for each modifier
    os << " { issueCycles = " << inst.issueCycles << ", latencyCycles = " << inst.latencyCycles;
    for (const auto& mod : inst.getModifiers()) {
        printModifierAsDict(*mod);
    }
    os << " }\n";
}

bool AsmPrinter::printModifierAsDict(const Modifier& mod) {
    return ModifierSerializer::serialize(mod, os);
}

void AsmPrinter::printDirective(const AsmDirective& directive) {
    os << std::string(static_cast<size_t>(options.indent), ' ');
    os << "\"st.asm_directive\"(\"" << directive.name << "\"";
    if (!directive.symbol.empty()) os << ", \"" << directive.symbol << "\"";
    os << ")\n";
}

void AsmPrinter::printSuccessorsLine(const BasicBlock& bb) {
    const auto& succs = bb.getSuccessors();
    if (succs.empty()) return;
    os << std::string(static_cast<size_t>(options.indent), ' ');
    os << "Successors: ";
    for (size_t i = 0; i < succs.size(); ++i) {
        if (i > 0) os << ", ";
        os << "^" << succs[i]->getLabel();
    }
    os << "\n";
}

}  // namespace stinkytofu
