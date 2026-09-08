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

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <vector>

#include "ModifierSerializer.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/HwRegHelpers.hpp"
#include "stinkytofu/ir/asm/ssa/SSAOperandUnits.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

/// Register classes \p inst's attached SSA was built for, which decides how many
/// slots each operand consumes.
///
/// An instruction printed outside a function has no arena to ask, so it falls
/// back to every liftable class — the answer that predates lift scoping. Printing
/// is diagnostic, so a wrong guess there misreads a dump rather than the program.
const RegClassSet& ssaClassesOf(const StinkyInstruction& inst) {
    static const RegClassSet kAllClasses = RegClassSet::all();
    const BasicBlock* block = inst.getParentBlock();
    const Function* function = block == nullptr ? nullptr : block->getParentFunc();
    return function == nullptr ? kAllClasses : function->ssaArena().liftedClasses();
}

}  // namespace

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
            if (reg.reg.isMinus) os << "-";
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
    printInstruction(inst, 0);
}

void AsmPrinter::print(const AsmDirective& directive) {
    printDirective(directive, 0);
}

void AsmPrinter::print(const Function& function) {
    printFunction(function, 0);
}

void AsmPrinter::print(const StinkyAsmModule& module) {
    os << "st.module @" << module.getName() << " {\n";
    const auto functions = module.getFunctions();
    for (size_t i = 0; i < functions.size(); ++i) {
        if (i > 0) os << "\n";
        printFunction(*functions[i], options.indent);
    }
    os << "}\n";
}

void AsmPrinter::printFunction(const Function& function, int baseIndent) {
    os << std::string(static_cast<size_t>(baseIndent), ' ');
    os << "st.func @" << function.getName() << "() {\n";
    size_t index = 0;
    for (const BasicBlock& bb : function) {
        printBlock(bb, index, baseIndent + options.indent);
        ++index;
    }
    os << std::string(static_cast<size_t>(baseIndent), ' ') << "}\n";
}

void AsmPrinter::printBlock(const BasicBlock& bb, size_t blockIndex) {
    printBlock(bb, blockIndex, 0);
}

void AsmPrinter::printBlock(const BasicBlock& bb, size_t blockIndex, int baseIndent) {
    std::string blockId =
        bb.getLabel().empty() ? ("bb" + std::to_string(blockIndex)) : bb.getLabel();
    os << std::string(static_cast<size_t>(baseIndent), ' ');
    os << "^" << blockId;
    if (options.ssaForm) printBlockArgumentList(bb);
    os << ":\n";
    if (options.ssaForm) printBlockArgumentSources(bb, baseIndent);
    for (const IRBase& ir : bb) printIR(ir, baseIndent);
    printSuccessorsLine(bb, baseIndent);
}

void AsmPrinter::printSSAValue(const StinkySSAValue* value) {
    if (value == nullptr) {
        os << "%<null>";
        return;
    }
    os << "%" << value->valueId() << ":" << regTypeToString(value->type().regType);
}

void AsmPrinter::printBlockArgumentList(const BasicBlock& bb) {
    if (bb.ssaArguments().empty()) return;
    os << "(";
    for (size_t i = 0; i < bb.ssaArguments().size(); ++i) {
        if (i > 0) os << ", ";
        printSSAValue(bb.ssaArguments()[i].value);
    }
    os << ")";
}

void AsmPrinter::printBlockArgumentSources(const BasicBlock& bb, int baseIndent) {
    // Lift appends incoming as the dominator walk reaches each predecessor, so
    // the stored order is deterministic but tracks that walk rather than the CFG.
    // Printing in predecessor order lines the edges up with the block's
    // predecessor list and keeps a dump stable if the walk ever changes.
    const std::vector<BasicBlock*>& preds = bb.getPredecessors();

    for (const SSABlockArgument& arg : bb.ssaArguments()) {
        // A live-in has no incoming edge; the header alone already names it.
        if (arg.incoming.empty()) continue;

        std::vector<const SSABlockIncoming*> ordered;
        ordered.reserve(arg.incoming.size());
        for (const SSABlockIncoming& incoming : arg.incoming) ordered.push_back(&incoming);
        std::stable_sort(ordered.begin(), ordered.end(),
                         [&preds](const SSABlockIncoming* lhs, const SSABlockIncoming* rhs) {
                             const auto position = [&preds](const BasicBlock* block) {
                                 const auto it = std::find(preds.begin(), preds.end(), block);
                                 return static_cast<size_t>(std::distance(preds.begin(), it));
                             };
                             return position(lhs->predecessor) < position(rhs->predecessor);
                         });

        os << std::string(static_cast<size_t>(baseIndent + options.indent), ' ');
        printSSAValue(arg.value);
        os << " = phi(";
        for (size_t i = 0; i < ordered.size(); ++i) {
            if (i > 0) os << ", ";
            const SSABlockIncoming& incoming = *ordered[i];
            os << "^"
               << (incoming.predecessor == nullptr ? "<null>" : incoming.predecessor->getLabel())
               << ": ";
            printSSAValue(incoming.use == nullptr ? nullptr : incoming.use->value());
        }
        os << ")\n";
    }
}

bool AsmPrinter::printsSSA(const StinkyInstruction& inst) const {
    return options.ssaForm && inst.hasAttachedSSA();
}

void AsmPrinter::printSSAOperandGroup(const StinkyInstruction& inst, const StinkyRegister& reg,
                                      bool isDestination, size_t& cursor) {
    const size_t units = liftedSSAUnits(reg, ssaClassesOf(inst));
    const size_t available = isDestination ? inst.getNumSSAResults() : inst.getNumSSAOperands();

    if (units == 0) {
        // Not lifted: the SSA slot holds the same payload the physical operand
        // spells, so print the physical form. Destinations consume no slot.
        if (!isDestination && cursor < available) ++cursor;
        printRegister(reg);
        return;
    }
    if (cursor + units > available) {
        // Attached SSA does not describe this operand; stay inspectable rather
        // than reading past the end.
        printRegister(reg);
        return;
    }

    // Brackets keep a multi-DWORD range distinguishable from separate operands.
    if (units > 1) os << "[";
    for (size_t unit = 0; unit < units; ++unit) {
        if (unit > 0) os << ", ";
        printSSAValue(isDestination ? inst.getSSAResult(cursor++)
                                    : inst.getSSAOperandValue(cursor++));
    }
    if (units > 1) os << "]";
}

void AsmPrinter::printIR(const IRBase& ir) {
    printIR(ir, 0);
}

void AsmPrinter::printIR(const IRBase& ir, int baseIndent) {
    switch (ir.getType()) {
        case IRBase::IRType::StinkyTofu: {
            if (const StinkyInstruction* inst = dyn_cast<StinkyInstruction>(&ir))
                printInstruction(*inst, baseIndent);
            else {
                // other StinkyTofu: indent and dump
                os << std::string(static_cast<size_t>(baseIndent + options.indent), ' ');
                ir.dump(os);
                os << "\n";
            }
            break;
        }
        case IRBase::IRType::StinkyAsmDirective:
            if (const AsmDirective* directive = dyn_cast<AsmDirective>(&ir))
                printDirective(*directive, baseIndent);
            else {
                os << std::string(static_cast<size_t>(baseIndent + options.indent), ' ');
                ir.dump(os);
                os << "\n";
            }
            break;
        case IRBase::IRType::LogicalIR:
            os << std::string(static_cast<size_t>(baseIndent + options.indent), ' ');
            ir.dump(os);
            os << "\n";
            break;
    }
}

void AsmPrinter::printInstruction(const StinkyInstruction& inst, int baseIndent) {
    // labels are block boundaries; do not print LABEL as instruction
    if (inst.getUnifiedOpcode() == GFX::LABEL) return;

    os << std::string(static_cast<size_t>(baseIndent + options.indent), ' ');

    const bool ssa = printsSSA(inst);

    if (!inst.getDestRegs().empty()) {
        size_t resultCursor = 0;
        for (size_t i = 0; i < inst.getDestRegs().size(); ++i) {
            if (i > 0) {
                os << ", ";
            }
            if (ssa)
                printSSAOperandGroup(inst, inst.getDestRegs()[i], /*isDestination=*/true,
                                     resultCursor);
            else
                printRegister(inst.getDestRegs()[i]);
        }
        os << " = ";
    }

    const std::string irNamespace = "st";
    os << "\"" << irNamespace << "." << inst.getHwInstDesc()->mnemonic << "\"";

    os << "(";
    size_t operandCursor = 0;
    for (size_t i = 0; i < inst.getSrcRegs().size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        if (ssa)
            printSSAOperandGroup(inst, inst.getSrcRegs()[i], /*isDestination=*/false,
                                 operandCursor);
        else
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

void AsmPrinter::printDirective(const AsmDirective& directive, int baseIndent) {
    os << std::string(static_cast<size_t>(baseIndent + options.indent), ' ');
    os << "\"st.asm_directive\"(\"" << directive.name << "\"";
    if (!directive.symbol.empty()) os << ", \"" << directive.symbol << "\"";
    os << ")\n";
}

void AsmPrinter::printSuccessorsLine(const BasicBlock& bb, int baseIndent) {
    const auto& succs = bb.getSuccessors();
    if (succs.empty()) return;
    os << std::string(static_cast<size_t>(baseIndent + options.indent), ' ');
    os << "Successors: ";
    for (size_t i = 0; i < succs.size(); ++i) {
        if (i > 0) os << ", ";
        os << "^" << succs[i]->getLabel();
    }
    os << "\n";
}

}  // namespace stinkytofu
