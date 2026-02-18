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
#include <iostream>

namespace
{
    void printBasicBlock(stinkytofu::AsmPrinter& printer, const stinkytofu::BasicBlock& bb)
    {
        for(const stinkytofu::IRBase& ir : bb)
        {
            if(const stinkytofu::StinkyInstruction* inst
               = dyn_cast<stinkytofu::StinkyInstruction>(&ir))
            {
                printer.getStream() << std::setw(printer.getOptions().indent) << "";
                printer.print(*inst);
            }
        }
    }
} // namespace

namespace stinkytofu
{
    //----------------------------------------------------------------------
    // RegisterPrinter implementation
    //----------------------------------------------------------------------
    void RegisterPrinter::print(const StinkyRegister& reg)
    {
        switch(reg.dataType)
        {
        case StinkyRegister::Type::Register:
        {
            // Print register with optional prefix for AGPR (shown as "acc")
            std::string prefix = regTypeToString(reg.reg.type);
            if(reg.reg.type == RegType::AGPR)
                prefix = "acc";

            // Single register without brackets (e.g., v12, BARRIER0)
            // Range with brackets (e.g., v[10:13])
            if(reg.hasSymbolicName())
            {
                if(reg.reg.num == 1)
                {
                    os << prefix << "[" << reg.getSymbolicName() << "]";
                }
                else
                {
                    os << prefix << "[" << reg.getSymbolicName() << ":" << reg.getSymbolicName()
                       << "+" << (reg.reg.num - 1) << "]";
                }
            }
            else if(reg.reg.num == 1)
            {
                os << prefix << reg.reg.idx;
            }
            else
            {
                os << prefix << "[" << reg.reg.idx << ":" << (reg.reg.idx + reg.reg.num - 1) << "]";
            }
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

        case StinkyRegister::Type::Invalid:
            os << "<invalid>";
            break;
        }
    }

    //----------------------------------------------------------------------
    // AsmPrinter implementation
    //----------------------------------------------------------------------
    void AsmPrinter::print(const StinkyInstruction& inst)
    {
        // Check if this is a label
        if(inst.getUnifiedOpcode() == GFX::LABEL)
        {
            if(const LabelData* labelMod = inst.getModifier<LabelData>())
            {
                os << labelMod->label << ":\n";
                return;
            }
        }

        // MLIR-style format:
        // destRegs = "operation.mnemonic"(srcRegs) { attributes }

        // Print destination registers
        if(!inst.getDestRegs().empty())
        {
            for(size_t i = 0; i < inst.getDestRegs().size(); ++i)
            {
                if(i > 0)
                    os << ", ";
                regPrinter.print(inst.getDestRegs()[i]);
            }
            os << " = ";
        }

        // Print operation name with namespace
        // TODO: get namespace from IR context
        const std::string irNamespace = "st";
        os << "\"" << irNamespace << "." << inst.getHwInstDesc()->mnemonic << "\"";

        // Print source registers as operands
        os << "(";
        if(!inst.getSrcRegs().empty())
        {
            // Check if instruction has VOP3 modifiers
            const VOP3Modifiers* vop3Mod = inst.getModifier<VOP3Modifiers>();

            for(size_t i = 0; i < inst.getSrcRegs().size(); ++i)
            {
                if(i > 0)
                    os << ", ";

                bool needsNeg = false;
                bool needsAbs = false;

                // Check VOP3 modifiers for this source operand
                if(vop3Mod)
                {
                    switch(i)
                    {
                    case 0:
                        needsNeg = vop3Mod->neg_src0;
                        needsAbs = vop3Mod->abs_src0;
                        break;
                    case 1:
                        needsNeg = vop3Mod->neg_src1;
                        needsAbs = vop3Mod->abs_src1;
                        break;
                    case 2:
                        needsNeg = vop3Mod->neg_src2;
                        needsAbs = vop3Mod->abs_src2;
                        break;
                    }
                }

                // Print modifiers
                if(needsNeg)
                    os << "-";
                if(needsAbs)
                    os << "abs(";

                regPrinter.print(inst.getSrcRegs()[i]);

                if(needsAbs)
                    os << ")";
            }
        }
        os << ")";
        if(auto waitCntData = inst.getModifier<SWaitCntData>())
        {
            os << " (" << *waitCntData << ")";
        }

        // Print attributes
        os << " { ";
        // Always print issue and latency cycles
        os << "issueCycles = " << inst.issueCycles;
        os << ", latencyCycles = " << inst.latencyCycles;

        os << " }";
        os << "\n";
    }

    void AsmPrinter::print(const Function& function)
    {
        for(const BasicBlock& bb : function)
            printBasicBlock(*this, bb);
    }

} // namespace stinkytofu
