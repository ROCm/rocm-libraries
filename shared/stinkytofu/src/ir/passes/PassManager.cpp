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

#include "ir/passes/PassManager.hpp"
#include "ErrorHandling.hpp"
#include "ir/IRModule.hpp"
#include "ir/StinkyInstructions.hpp"

namespace stinkytofu
{
    IRInstPassManager::IRInstPassManager(GfxArchID arch)
        : arch(arch)
    {
    }

    std::unique_ptr<IRList> IRInstPassManager::run(IRModule* module)
    {
        auto result = std::make_unique<IRList>();

        if(!module)
        {
            return result;
        }

        // Start with the input IR instructions (convert shared_ptr to raw pointers for pass processing)
        std::vector<IRInstruction*> currentIR;
        for(const auto& inst : module->getInstructions())
        {
            currentIR.push_back(inst.get());
        }

        // Run all IR transformation passes
        for(auto& pass : passes)
        {
            if(pass->producesAsm())
            {
                // This is the final lowering pass (IRInstruction -> StinkyInstruction)
                // We know it's IRInstToAsmPass because producesAsm() == true
                auto* toAsmPass = static_cast<IRInstToAsmPass*>(pass.get());

                // Lower each IR instruction to assembly
                for(auto* irInst : currentIR)
                {
                    auto asmInsts = toAsmPass->lower(irInst, arch);
                    for(auto* asmInst : asmInsts)
                    {
                        result->push_back(asmInst);
                    }
                }

                return result;
            }
            else
            {
                // IRInstruction transformation pass (IRInst -> IRInst)
                // We know it's IRInstTransformPass because producesAsm() == false
                auto* transformPass = static_cast<IRInstTransformPass*>(pass.get());

                // Transform each IR instruction
                std::vector<IRInstruction*> nextIR;
                for(auto* irInst : currentIR)
                {
                    auto transformed = transformPass->transform(irInst, arch);
                    nextIR.insert(nextIR.end(), transformed.begin(), transformed.end());
                }

                currentIR = std::move(nextIR);
            }
        }

        // No IRInstToAsmPass in pipeline - this is an error
        STINKY_UNREACHABLE(
            "IRInstPassManager pipeline must end with an IRInstToAsmPass (e.g., ToStinkyAsmPass)");
        return result;
    }

} // namespace stinkytofu
