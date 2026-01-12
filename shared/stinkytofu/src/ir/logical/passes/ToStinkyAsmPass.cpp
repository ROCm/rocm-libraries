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

#include "ir/logical/passes/ToStinkyAsmPass.hpp"
#include "ErrorHandling.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/logical/LogicalInstructions.hpp"
#include "isa/ArchHelper.hpp"
#include "stinkytofu.hpp"
#include "support/Casting.hpp"
#include <string>
#include <vector>

namespace
{
    using namespace stinkytofu;

    // Helper to create assembly instruction from IR
    StinkyInstruction* createAsmFromIR(LogicalInstruction* irInst, GfxArchID arch)
    {
        const char* logicalName = irInst->getLogicalName();

        // Get the architecture-specific mnemonic
        uint16_t    isaOpcode = 0;
        std::string mnemonic;

        // Auto-generated mappings from TableGen
        if(false)
        {
            // Placeholder for generated code
        }
#include "ir/IRMnemonics_generated.inc"
        else
        {
            STINKY_UNREACHABLE(
                ("ToStinkyAsmPass: Unknown IR instruction: " + std::string(logicalName)).c_str());
            return nullptr;
        }

        // Get the ISA opcode for this mnemonic on the target architecture
        isaOpcode              = getMnemonicToIsaOpcode(mnemonic, arch);
        const HwInstDesc* desc = getMCIDByIsaOp(isaOpcode, arch);

        if(!desc)
        {
            STINKY_UNREACHABLE(
                ("ToStinkyAsmPass: Instruction not supported on architecture: " + mnemonic)
                    .c_str());
            return nullptr;
        }

        // Create the assembly instruction
        StinkyInstruction* asmInst = new StinkyInstruction(desc);

        // Copy operands from IR to assembly
        if(!irInst->dests.empty())
        {
            asmInst->setDestRegs(irInst->dests);
        }
        if(!irInst->srcs.empty())
        {
            asmInst->setSrcRegs(irInst->srcs);
        }

        // Copy comment
        if(!irInst->comment.empty())
        {
            asmInst->addModifier(CommentData(irInst->comment));
        }

        // TODO: Copy DPP, SDWA, DS modifiers when needed

        return asmInst;
    }

    /// Implementation of ToStinkyAsmPass using unified Pass infrastructure
    class ToStinkyAsmPassImpl : public Pass
    {
    public:
        static constexpr const char* PassName = "ToStinkyAsmPass";
        static char                  ID;

        PassID getPassID() const override
        {
            return &ID;
        }

        const char* getName() const override
        {
            return PassName;
        }

        void run(Function& func, PassContext& passCtx) override
        {
            GfxArchID arch = getGfxArchID(passCtx.getGemmTileConfig().arch[0],
                                          passCtx.getGemmTileConfig().arch[1],
                                          passCtx.getGemmTileConfig().arch[2]);

            // Process all basic blocks
            for(BasicBlock& bb : func)
            {
                // Skip filtered basic blocks
                if(!passCtx.shouldProcessBasicBlock(bb))
                    continue;

                lowerToAsm(bb, arch);
            }
        }

    private:
        void lowerToAsm(BasicBlock& bb, GfxArchID arch)
        {
            IRList& irlist = bb.getIR();

            // Use iterators to allow insertion/removal during traversal
            auto it = irlist.begin();
            while(it != irlist.end())
            {
                IRBase* irNode = &(*it);

                if(irNode->getType() == IRBase::IRType::LogicalIR)
                {
                    LogicalInstruction* logicalInst = cast<LogicalInstruction>(irNode);

                    // Lower to assembly
                    StinkyInstruction* asmInst = createAsmFromIR(logicalInst, arch);

                    if(asmInst)
                    {
                        // Insert assembly instruction before the logical instruction
                        irlist.insert(it, static_cast<IRBase*>(asmInst));

                        // Remove the logical instruction from IRList
                        auto toRemove = it;
                        ++it; // Move to next before removing
                        irlist.remove(&(*toRemove));

                        // Only delete if NOT externally owned (e.g., by shared_ptr)
                        if(!logicalInst->isExternallyOwned())
                        {
                            delete logicalInst;
                        }
                        continue;
                    }
                }

                ++it;
            }
        }
    };

    char ToStinkyAsmPassImpl::ID = 0;

} // anonymous namespace

namespace stinkytofu
{
    std::unique_ptr<Pass> createToStinkyAsmPass()
    {
        return std::make_unique<ToStinkyAsmPassImpl>();
    }

} // namespace stinkytofu
