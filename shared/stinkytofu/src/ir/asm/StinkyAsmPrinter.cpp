/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include "ir/asm/StinkyAsmPrinter.hpp"
#include <iomanip>
#include <iostream>

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
            std::string prefix = reg.regType;
            if(reg.regType == "AGPR")
                prefix = "acc";

            // Single register without brackets (e.g., v12, BARRIER0)
            // Range with brackets (e.g., v[10:13])
            if(reg.regNum == 1)
            {
                os << prefix << reg.regIdx;
            }
            else
            {
                os << prefix << "[" << reg.regIdx << ":" << (reg.regIdx + reg.regNum - 1) << "]";
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
        if(!inst.destRegs.empty())
        {
            for(size_t i = 0; i < inst.destRegs.size(); ++i)
            {
                if(i > 0)
                    os << ", ";
                regPrinter.print(inst.destRegs[i]);
            }
            os << " = ";
        }

        // Print operation name with namespace
        // TODO: get namespace from IR context
        const std::string irNamespace = "st";
        os << "\"" << irNamespace << "." << inst.getHwInstDesc()->mnemonic << "\"";

        // Print source registers as operands
        os << "(";
        if(!inst.srcRegs.empty())
        {
            for(size_t i = 0; i < inst.srcRegs.size(); ++i)
            {
                if(i > 0)
                    os << ", ";
                regPrinter.print(inst.srcRegs[i]);
            }
        }
        os << ")";

        // Print attributes
        os << " { ";
        // Always print issue and latency cycles
        os << "issueCycles = " << inst.issueCycles;
        os << ", latencyCycles = " << inst.latencyCycles;

        os << " }";
        os << "\n";
    }

    void AsmPrinter::print(const IRList& irlist)
    {
        for(const IRBase& ir : irlist)
        {
            if(const StinkyInstruction* inst = dyn_cast<StinkyInstruction>(&ir))
            {
                os << std::setw(getOptions().indent) << "";
                print(*inst);
            }
        }
    }

} // namespace stinkytofu
