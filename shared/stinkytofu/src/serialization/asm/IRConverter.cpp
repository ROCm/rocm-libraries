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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include <iostream>

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"

namespace stinkytofu
{
    StinkyIRConverter::StinkyIRConverter()
        : arch({9, 4, 2})
    {
    }

    StinkyIRConverter::StinkyIRConverter(const std::array<int, 3>& targetArch)
        : arch(targetArch)
    {
    }

    StinkyErrorCode StinkyIRConverter::populateFunctionFromString(const std::string& irText,
                                                                  Function&          func,
                                                                  PassContext&       passCtx,
                                                                  GfxArchID          arch)
    {
        // Parse the raw instruction string
        auto parsedInstructions = parseSourceString(irText);

        // Create an entry BasicBlock to hold all instructions
        BasicBlock* entryBB = func.createBasicBlock("entry");

        // Create the IR builder
        AsmIRBuilder irBuilder(*entryBB, arch);

        // Convert parsed instructions to StinkyInstruction objects
        for(const auto& inst : parsedInstructions)
        {
            // Check if it's a label
            if(inst->isLabel)
            {
                irBuilder.createLabel(inst->opcodeStr);
                continue;
            }

            // Get the opcode and hardware instruction descriptor
            auto              opcode     = getMnemonicToIsaOpcode(inst->opcodeStr, arch);
            const HwInstDesc* hwInstDesc = getMCIDByIsaOp(static_cast<IsaOpcode>(opcode), arch);

            if(hwInstDesc == nullptr)
            {
                std::cerr << "Warning: No hardware instruction descriptor found for opcode "
                          << opcode << " in arch gfx" << static_cast<int>(arch) << "\n";
            }
            else
            {
                StinkyInstruction* stinkyInst = irBuilder.create(hwInstDesc);

                // Move destination and source registers
                stinkyInst->setDestRegs(inst->destRegs);
                stinkyInst->setSrcRegs(inst->srcRegs);

                // Overwrite cycles when valid (> 0), otherwise use default from HwInstDesc
                if(inst->issueCycles > 0)
                {
                    stinkyInst->issueCycles = inst->issueCycles;
                }

                if(inst->latencyCycles > 0)
                {
                    stinkyInst->latencyCycles = inst->latencyCycles;
                }
            }
        }

        return StinkyErrorCode::SUCCESS;
    }

    Function* StinkyIRConverter::convertToFunction(const std::string& rawInstructions)
    {
        function = std::make_unique<Function>("kernel");
        passCtx  = std::make_unique<PassContext>();

        // Set up kernel configuration
        GemmTileConfig config;
        config.arch     = arch;
        config.TileA0   = 0;
        config.TileB0   = 0;
        config.TileM0   = 0;
        config.NumGRA   = 0;
        config.NumGRB   = 0;
        config.NumGRM   = 0;
        config.NumWaves = 0;
        passCtx->setGemmTileConfig(config);

        GfxArchID       archID = getGfxArchID(arch[0], arch[1], arch[2]);
        StinkyErrorCode result
            = populateFunctionFromString(rawInstructions, *function, *passCtx, archID);
        if(result != StinkyErrorCode::SUCCESS)
        {
            function.reset();
            passCtx.reset();
            return nullptr;
        }

        return function.get();
    }

    PassContext* StinkyIRConverter::getPassContext()
    {
        return passCtx.get();
    }

    void StinkyIRConverter::cleanup()
    {
        function.reset();
        passCtx.reset();
    }

    StinkyIRConverter::~StinkyIRConverter()
    {
        cleanup();
    }
} // namespace stinkytofu
