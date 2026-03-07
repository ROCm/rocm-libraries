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

#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#include <iostream>
#include <sstream>

namespace stinkytofu
{
    char StinkyIRVerifierPass::ID = 0;

    void StinkyIRVerifierPass::run(Function& func, PassContext&)
    {
        std::string error = validateStinkyIR(func, config_);
        if(!error.empty())
        {
            if(config_.abortOnError)
            {
                // Temporarily disable unreachable error due to rocisa
                // STINKY_UNREACHABLE(error.c_str());
            }
            std::cerr << "[StinkyIRVerifier] " << error;
        }
    }

    // ===========================================================================
    // Register width validation (ASM-only; uses HwInstDesc from hardware defs)
    // ===========================================================================

    static RegType fieldTypeToRegType(FieldType ft)
    {
        switch(ft)
        {
        case FieldType::vgpr:
        case FieldType::src_vgpr:
        case FieldType::src_vgpr_or_inline:
            return RegType::V;
        case FieldType::sreg:
        case FieldType::sreg_m0:
        case FieldType::sgpr:
        case FieldType::sdst:
        case FieldType::ssrc:
            return RegType::S;
        default:
            return RegType::UNKNOWN;
        }
    }

    static std::string checkRegisterWidths(const StinkyInstruction* inst,
                                           const AsmVerifierConfig& config)
    {
        const HwInstDesc* hwDesc = inst->getHwInstDesc();
        if(!hwDesc || !hwDesc->mnemonic)
            return "";

        if(hwDesc->operandFields.empty())
            return "";

        std::stringstream errors;

        unsigned destIdx = 0;
        unsigned srcIdx  = 0;
        for(const auto& field : hwDesc->operandFields)
        {
            bool        isDest       = field.isDest;
            unsigned    operandIndex = isDest ? destIdx++ : srcIdx++;
            const auto& regs = isDest ? inst->getDestRegs() : inst->getSrcRegs();

            if(field.fieldSizeBits == 0)
                continue;

            unsigned expectedWidth = field.fieldSizeBits / 32;
            if(expectedWidth <= 1)
                continue;

            if(operandIndex >= regs.size())
            {
                errors << "Instruction '" << hwDesc->mnemonic << "' missing operand "
                       << (isDest ? "dest[" : "src[") << operandIndex << "]\n";
                continue;
            }

            const StinkyRegister& reg = regs[operandIndex];
            if(reg.dataType != StinkyRegister::Type::Register)
                continue;

            if(reg.reg.num != expectedWidth)
            {
                errors << "Instruction '" << hwDesc->mnemonic << "' operand "
                       << (isDest ? "dest[" : "src[") << operandIndex << "] "
                       << "has register width " << reg.reg.num << ", expected " << expectedWidth
                       << "\n";
            }

            RegType expectedType = fieldTypeToRegType(field.fieldType);
            if(expectedType != RegType::UNKNOWN && reg.reg.type != expectedType)
            {
                errors << "Instruction '" << hwDesc->mnemonic << "' operand "
                       << (isDest ? "dest[" : "src[") << operandIndex << "] "
                       << "has register type '" << regTypeToString(reg.reg.type) << "', expected '"
                       << regTypeToString(expectedType) << "'\n";
            }
        }

        return errors.str();
    }

    // ===========================================================================
    // StinkyTofu Assembly IR validation (ASM pipeline only)
    // ===========================================================================

    std::string validateStinkyIR(Function& func, const AsmVerifierConfig& config)
    {
        if(config.verbose)
        {
            std::cout << "[StinkyIRVerifier] Verifying StinkyTofu Assembly IR...\n";
        }

        if(func.empty())
            return "Function is empty (no basic blocks)";

        if(!func.getEntryBlock())
            return "Function has no entry basic block";

        size_t            logicalCount  = 0;
        size_t            stinkyCount   = 0;
        size_t            totalBlocks   = 0;
        size_t            invalidHwDesc = 0;
        std::stringstream widthErrors;

        for(BasicBlock& bb : func)
        {
            totalBlocks++;

            for(IRBase& ir : bb)
            {
                if(ir.getType() == IRBase::IRType::LogicalIR)
                {
                    logicalCount++;
                }
                else if(ir.getType() == IRBase::IRType::StinkyTofu)
                {
                    stinkyCount++;

                    auto*             stinkyInst = static_cast<StinkyInstruction*>(&ir);
                    const HwInstDesc* hwDesc     = stinkyInst->getHwInstDesc();
                    if(!hwDesc || !hwDesc->mnemonic || hwDesc->mnemonic[0] == '\0')
                    {
                        invalidHwDesc++;
                    }
                    else if(config.checkRegisterWidths)
                    {
                        std::string widthError = checkRegisterWidths(stinkyInst, config);
                        if(!widthError.empty())
                            widthErrors << widthError;
                    }
                }
            }
        }

        if(logicalCount > 0)
        {
            std::stringstream ss;
            ss << "StinkyTofu Assembly IR contains " << logicalCount << " Logical instructions. "
               << "This suggests IR is not fully lowered or mixed.";
            return ss.str();
        }

        if(stinkyCount == 0)
            return "Function contains no StinkyTofu instructions (empty IR)";

        if(invalidHwDesc > 0)
        {
            std::stringstream ss;
            ss << "Found " << invalidHwDesc << " StinkyTofu instruction(s) with invalid or missing "
               << "hardware instruction descriptors";
            return ss.str();
        }

        std::string widthErrorStr = widthErrors.str();
        if(!widthErrorStr.empty())
        {
            std::stringstream ss;
            ss << "Register width validation failed:\n" << widthErrorStr;
            return ss.str();
        }

        if(config.verbose)
        {
            std::cout << "[StinkyIRVerifier] OK: " << totalBlocks << " blocks, " << stinkyCount
                      << " stinky instructions\n";
        }

        return "";
    }
} // namespace stinkytofu
