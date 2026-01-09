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

#include "ir/passes/IntrinsicExpansionPass.hpp"
#include "ir/IntrinsicCall.hpp"
#include <iostream>
#include <sstream>

namespace stinkytofu
{
    IntrinsicExpansionPass::IntrinsicExpansionPass(std::shared_ptr<IntrinsicLibrary> library)
        : library_(library)
    {
    }

    bool IntrinsicExpansionPass::run(IRModule* module)
    {
        bool changed = false;

        // Iterate through all instructions in the module
        std::vector<IRInstruction*> toExpand;

        for(auto* inst : module->instructions)
        {
            // Check if this is an IntrinsicCall
            if(strcmp(inst->getLogicalName(), "IntrinsicCall") == 0)
            {
                toExpand.push_back(inst);
            }
        }

        // Expand each IntrinsicCall
        for(auto* call : toExpand)
        {
            auto expanded = expandIntrinsic(call);
            if(!expanded.empty())
            {
                // Find position of IntrinsicCall in module
                auto it = std::find(module->instructions.begin(), module->instructions.end(), call);
                if(it != module->instructions.end())
                {
                    // Replace IntrinsicCall with expanded instructions
                    auto pos = std::distance(module->instructions.begin(), it);
                    module->instructions.erase(it);
                    module->instructions.insert(
                        module->instructions.begin() + pos, expanded.begin(), expanded.end());

                    // Delete the IntrinsicCall
                    delete call;
                    changed = true;
                }
            }
        }

        return changed;
    }

    std::vector<IRInstruction*> IntrinsicExpansionPass::expandIntrinsic(IRInstruction* inst)
    {
        // Cast to IntrinsicCall
        IntrinsicCall* call = dynamic_cast<IntrinsicCall*>(inst);
        if(!call)
        {
            return {};
        }

        const std::string& funcName = call->getFunctionName();

        // Look up intrinsic definition
        const Pattern* pattern = library_->lookup(funcName);
        if(!pattern)
        {
            std::cerr << "Error: Unknown intrinsic: " << funcName << "\n";
            return {};
        }

        // Build register map from intrinsic argument names to actual registers
        std::unordered_map<std::string, StinkyRegister> regMap;

        if(call->dests.size() != pattern->arguments.size())
        {
            std::cerr << "Error: Intrinsic " << funcName << " expects " << pattern->arguments.size()
                      << " arguments, got " << call->dests.size() << "\n";
            return {};
        }

        for(size_t i = 0; i < pattern->arguments.size(); ++i)
        {
            regMap[pattern->arguments[i].name] = call->dests[i];
        }

        // Create expanded instructions
        std::vector<IRInstruction*> expanded;

        for(const auto& instDef : pattern->body)
        {
            IRInstruction* newInst = createInstruction(instDef, regMap);
            if(newInst)
            {
                expanded.push_back(newInst);
            }
            else
            {
                std::cerr << "Error: Failed to create instruction: " << instDef.operation << "\n";
                // Clean up and return empty
                for(auto* i : expanded)
                {
                    delete i;
                }
                return {};
            }
        }

        return expanded;
    }

    IRInstruction* IntrinsicExpansionPass::createInstruction(
        const IntrinsicInstruction&                            inst,
        const std::unordered_map<std::string, StinkyRegister>& regMap)
    {
        // Map destination register
        StinkyRegister destReg;
        if(!parseOperand(inst.destReg, regMap, destReg))
        {
            std::cerr << "Error: Destination must be a register: " << inst.destReg << "\n";
            return nullptr;
        }

        // Map source registers and immediates
        std::vector<StinkyRegister> srcRegs;

        for(const auto& operand : inst.operands)
        {
            StinkyRegister srcReg;
            if(parseOperand(operand, regMap, srcReg))
            {
                // It's a register
                srcRegs.push_back(srcReg);
            }
            else
            {
                // It's an immediate - encode it as a special register
                // For now, we'll use a simple encoding
                // TODO: Handle immediates properly with IRBuilder extensions
                int64_t immValue = 0;
                try
                {
                    immValue = std::stoll(operand);
                }
                catch(...)
                {
                    // Try parsing as float
                    try
                    {
                        double fValue = std::stod(operand);
                        // Convert to bits for integer storage
                        immValue = *reinterpret_cast<int64_t*>(&fValue);
                    }
                    catch(...)
                    {
                        std::cerr << "Error: Invalid immediate: " << operand << "\n";
                        return nullptr;
                    }
                }

                // Create a special immediate register (idx = immediate value)
                srcRegs.push_back(StinkyRegister(RegType::S, static_cast<int>(immValue), 1));
            }
        }

        // Create IR instruction
        // For now, create a generic instruction
        // TODO: In the future, we'll create specific typed IR instructions
        // based on the operation name (e.g., VCmpGtF32, VSelectF32, etc.)

        IRInstruction* result = new IRInstruction(IRType::StinkyTofu);
        result->dests.push_back(destReg);
        result->srcs = srcRegs;

        return result;
    }

    bool IntrinsicExpansionPass::parseOperand(
        const std::string&                                     operand,
        const std::unordered_map<std::string, StinkyRegister>& regMap,
        StinkyRegister&                                        outReg)
    {
        // Check if operand is a register name from the intrinsic definition
        auto it = regMap.find(operand);
        if(it != regMap.end())
        {
            outReg = it->second;
            return true;
        }

        // Not a register, treat as immediate
        return false;
    }

} // namespace stinkytofu
