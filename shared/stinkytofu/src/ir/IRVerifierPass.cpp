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

#include "ir/IRVerifierPass.hpp"
#include "ErrorHandling.hpp"
#include "ir/asm/StinkyAsmIR.hpp"

#include <iostream>
#include <sstream>

namespace stinkytofu
{
    // ===========================================================================
    // Pass ID definitions
    // ===========================================================================

    char LogicalIRVerifierPass::ID = 0;
    char StinkyIRVerifierPass::ID  = 0;

    // ===========================================================================
    // Pass Implementations (thin wrappers)
    // ===========================================================================

    void LogicalIRVerifierPass::run(Function& func, PassContext& ctx)
    {
        std::string error = validateLogicalIR(func, config_);
        if(!error.empty())
        {
            STINKY_UNREACHABLE(error.c_str());
        }
    }

    void StinkyIRVerifierPass::run(Function& func, PassContext& ctx)
    {
        std::string error = validateStinkyIR(func, config_);
        if(!error.empty())
        {
            STINKY_UNREACHABLE(error.c_str());
        }
    }

    // ===========================================================================
    // Register Width Validation (using metadata from HwInstDesc)
    // ===========================================================================

    /// Check if register widths and types match expected values from hardware metadata
    ///
    /// Requirements are defined per-architecture in hardware files
    /// (e.g., Gfx1250.cpp defines that tensor_load_to_lds requires src0=4 SGPRs, src1=8 SGPRs)
    static std::string checkRegisterWidths(const StinkyInstruction* inst,
                                           const IRVerifierConfig&  config)
    {
        const HwInstDesc* hwDesc = inst->getHwInstDesc();
        if(!hwDesc || !hwDesc->mnemonic)
            return ""; // Already caught by earlier checks

        // Check if this instruction has operand requirements in hardware metadata
        if(hwDesc->operandWidths.empty())
        {
            // No requirements - validation passes
            return "";
        }

        std::stringstream errors;

        // Validate each operand requirement from hardware metadata
        for(const auto& req : hwDesc->operandWidths)
        {
            const auto& regs = req.isDest ? inst->getDestRegs() : inst->getSrcRegs();

            if(req.operandIndex >= regs.size())
            {
                errors << "Instruction '" << hwDesc->mnemonic << "' missing operand "
                       << (req.isDest ? "dest[" : "src[") << (int)req.operandIndex << "]\n";
                continue;
            }

            const StinkyRegister& reg = regs[req.operandIndex];
            if(reg.dataType != StinkyRegister::Type::Register)
            {
                // Not a register (literal, etc.) - skip validation
                continue;
            }

            // Check register width
            if(reg.reg.num != req.width)
            {
                errors << "Instruction '" << hwDesc->mnemonic << "' operand "
                       << (req.isDest ? "dest[" : "src[") << (int)req.operandIndex << "] "
                       << "has register width " << reg.reg.num << ", expected " << (int)req.width
                       << "\n";
            }

            // Check register type (if specified)
            if(req.expectedType != RegType::UNKNOWN && reg.reg.type != req.expectedType)
            {
                errors << "Instruction '" << hwDesc->mnemonic << "' operand "
                       << (req.isDest ? "dest[" : "src[") << (int)req.operandIndex << "] "
                       << "has register type '" << regTypeToString(reg.reg.type) << "', expected '"
                       << regTypeToString(req.expectedType) << "'\n";
            }
        }

        return errors.str();
    }

    // ===========================================================================
    // Standalone Validation Functions
    // ===========================================================================

    std::string validateLogicalIR(Function& func, const IRVerifierConfig& config)
    {
        if(config.verbose)
        {
            std::cout << "[LogicalIRVerifier] Verifying Logical IR...\n";
        }

        // Basic structure checks
        if(func.empty())
        {
            return "Function is empty (no basic blocks)";
        }

        if(!func.getEntryBlock())
        {
            return "Function has no entry basic block";
        }

        // Count instructions of each type
        size_t logicalCount = 0;
        size_t stinkyCount  = 0;
        size_t totalBlocks  = 0;

        for(BasicBlock& bb : func)
        {
            totalBlocks++;

            for(IRBase& ir : bb.getIR())
            {
                if(ir.getType() == IRBase::IRType::LogicalIR)
                {
                    logicalCount++;
                }
                else if(ir.getType() == IRBase::IRType::StinkyTofu)
                {
                    stinkyCount++;
                }
            }
        }

        // Logical IR should not have StinkyTofu instructions
        if(stinkyCount > 0)
        {
            std::stringstream ss;
            ss << "Logical IR contains " << stinkyCount << " StinkyTofu (assembly) instructions. "
               << "This suggests IR is partially lowered or mixed.";
            return ss.str();
        }

        if(logicalCount == 0)
        {
            return "Function contains no Logical instructions (empty IR)";
        }

        if(config.verbose)
        {
            std::cout << "[LogicalIRVerifier] OK: " << totalBlocks << " blocks, " << logicalCount
                      << " logical instructions\n";
        }

        return ""; // Valid
    }

    std::string validateStinkyIR(Function& func, const IRVerifierConfig& config)
    {
        if(config.verbose)
        {
            std::cout << "[StinkyIRVerifier] Verifying StinkyTofu Assembly IR...\n";
        }

        // Basic structure checks
        if(func.empty())
        {
            return "Function is empty (no basic blocks)";
        }

        if(!func.getEntryBlock())
        {
            return "Function has no entry basic block";
        }

        // Count instructions of each type
        size_t            logicalCount  = 0;
        size_t            stinkyCount   = 0;
        size_t            totalBlocks   = 0;
        size_t            invalidHwDesc = 0;
        std::stringstream widthErrors;

        for(BasicBlock& bb : func)
        {
            totalBlocks++;

            for(IRBase& ir : bb.getIR())
            {
                if(ir.getType() == IRBase::IRType::LogicalIR)
                {
                    logicalCount++;
                }
                else if(ir.getType() == IRBase::IRType::StinkyTofu)
                {
                    stinkyCount++;

                    // Check StinkyInstruction has valid hardware descriptor
                    auto*             stinkyInst = static_cast<StinkyInstruction*>(&ir);
                    const HwInstDesc* hwDesc     = stinkyInst->getHwInstDesc();
                    if(!hwDesc || !hwDesc->mnemonic || hwDesc->mnemonic[0] == '\0')
                    {
                        invalidHwDesc++;
                    }
                    else if(config.checkRegisterWidths)
                    {
                        // Check register widths for instructions that have requirements
                        std::string widthError = checkRegisterWidths(stinkyInst, config);
                        if(!widthError.empty())
                        {
                            widthErrors << widthError;
                        }
                    }
                }
            }
        }

        // StinkyTofu IR should not have Logical instructions
        if(logicalCount > 0)
        {
            std::stringstream ss;
            ss << "StinkyTofu Assembly IR contains " << logicalCount << " Logical instructions. "
               << "This suggests IR is not fully lowered or mixed.";
            return ss.str();
        }

        if(stinkyCount == 0)
        {
            return "Function contains no StinkyTofu instructions (empty IR)";
        }

        if(invalidHwDesc > 0)
        {
            std::stringstream ss;
            ss << "Found " << invalidHwDesc << " StinkyTofu instruction(s) with invalid or missing "
               << "hardware instruction descriptors";
            return ss.str();
        }

        // Check for register width errors
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

        return ""; // Valid
    }

} // namespace stinkytofu
