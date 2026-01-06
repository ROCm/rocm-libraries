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
#include "ir/asm/DelayAluInsertionPass.hpp"
#include "ErrorHandling.hpp"
#include "StinkyBuilder.hpp"
#include "ir/asm/DefUseChain.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "isa/ArchHelper.hpp"
#include "support/Casting.hpp"

#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

namespace
{
    using namespace stinkytofu;

    /// ALU instruction types for s_delay_alu encoding
    enum class DelayAluType
    {
        VALU, // Vector ALU (v_*)
        SALU, // Scalar ALU (s_*)
        TRANS, // Transcendental (v_s_*, v_exp_*, v_log_*, v_rcp_*, v_rsq_*, v_sqrt_*)
        OTHER // Non-ALU or doesn't need delays
    };

    /// Maximum lookback distance for each ALU type
    static const std::unordered_map<DelayAluType, unsigned> ALU_DEP_MAX = {
        {DelayAluType::VALU, 4}, // VALU can look back max 4 instructions
        {DelayAluType::SALU, 1}, // SALU can look back max 1 instruction
        {DelayAluType::TRANS, 3}, // TRANS can look back max 3 instructions
        {DelayAluType::OTHER, 0}, // OTHER doesn't participate
    };

    /// Maximum skip distance between s_delay_alu instructions
    static const unsigned SKIP_MAX = 5;

    /// Classifies an instruction by its ALU type for delay encoding
    DelayAluType classifyDelayAluType(const StinkyInstruction* inst)
    {
        if(!inst || !inst->getHwInstDesc() || !inst->getHwInstDesc()->mnemonic)
            return DelayAluType::OTHER;

        std::string opcode = inst->getHwInstDesc()->mnemonic;

        // Transcendental instructions: v_s_* or v_exp/log/rcp/rsq/sqrt
        if(opcode.compare(0, 4, "v_s_") == 0)
        {
            return DelayAluType::TRANS;
        }

        if(opcode.compare(0, 6, "v_exp_") == 0 || opcode.compare(0, 6, "v_log_") == 0
           || opcode.compare(0, 6, "v_rcp_") == 0 || opcode.compare(0, 6, "v_rsq_") == 0
           || opcode.compare(0, 7, "v_sqrt_") == 0)
        {
            return DelayAluType::TRANS;
        }

        // Vector ALU: v_*
        if(opcode.compare(0, 2, "v_") == 0)
        {
            return DelayAluType::VALU;
        }

        // Scalar ALU: s_* but excluding control flow, memory ops, waitcnt, barrier, delay_alu
        if(opcode.compare(0, 2, "s_") == 0)
        {
            // Exclude instructions that don't need delay
            if(opcode == "s_waitcnt" || opcode == "s_barrier" || opcode == "s_delay_alu"
               || opcode == "s_endpgm" || opcode == "s_nop" || opcode == "s_branch"
               || opcode.compare(0, 10, "s_cbranch_") == 0 || opcode.compare(0, 6, "s_load") == 0
               || opcode.compare(0, 7, "s_store") == 0 || opcode.compare(0, 8, "s_buffer") == 0)
            {
                return DelayAluType::OTHER;
            }

            return DelayAluType::SALU;
        }

        return DelayAluType::OTHER;
    }

    /// Delay information for an instruction
    struct DelayInfo
    {
        DelayAluType type;
        unsigned     typeCount; // Count of this ALU type seen so far
        unsigned     totalCount; // Total instruction count
    };

    /// Information needed to insert s_delay_alu
    struct PendingDelayAlu
    {
        DelayAluType type; // Type of the dependency (producer type)
        unsigned     dist; // Distance from dependent instruction (instid0 count)
    };

    /// Implementation of the DelayAluInsertionPass
    class DelayAluInsertionPassImpl : public Pass
    {
    public:
        static char ID;

        const char* getName() const override
        {
            return "DelayAluInsertionPass";
        }

        PassID getPassID() const override
        {
            return &DelayAluInsertionPassImpl::ID;
        }

        void run(Function& func, PassContext& passCtx) override
        {
            // Only run for architectures with HasSchedMode (RDNA4 gfx12xx)
            // s_delay_alu is RDNA4-specific (HasSchedMode = checkInList(isaVersion[0], {12}))
            GfxArchID arch = getGfxArchID(passCtx.getGemmTileConfig().arch[0],
                                          passCtx.getGemmTileConfig().arch[1],
                                          passCtx.getGemmTileConfig().arch[2]);

            // Temporary: Use Gfx1250 (CDNA5) for testing since it's modified from RDNA4
            // and has similar scheduling requirements
            // TODO: When proper RDNA4 (gfx12xx) support is added, check for gfx12xx variants
            bool hasSchedMode = (arch == GfxArchID::Gfx1250); // gfx1250 is modified from RDNA4

            if(!hasSchedMode)
            {
                // Architecture doesn't have HasSchedMode - this is a configuration error
                STINKY_UNREACHABLE("Expert Schedule mode not supported for this architecture!");
            }

            int totalInserted = 0;

            // Process each BasicBlock
            for(BasicBlock& bb : func)
            {
                if(!passCtx.shouldProcessBasicBlock(bb))
                    continue;

                int inserted = processBasicBlock(bb, func, passCtx);
                totalInserted += inserted;
            }

            // TODO: Add verbose support when PassContext::isVerbose() is available
            // if(passCtx.isVerbose())
            // {
            //     std::cout << "DelayAluInsertion: Inserted " << totalInserted
            //               << " s_delay_alu instructions" << std::endl;
            // }
        }

    private:
        int processBasicBlock(BasicBlock& bb, Function& func, PassContext& passCtx)
        {
            GfxArchID arch = getGfxArchID(passCtx.getGemmTileConfig().arch[0],
                                          passCtx.getGemmTileConfig().arch[1],
                                          passCtx.getGemmTileConfig().arch[2]);

            // Build use-def chains (stores in inst->sources and inst->users)
            buildUseDefChain(bb);

            // Collect all instructions in this basic block
            std::vector<StinkyInstruction*> instructions;
            for(IRBase& irNode : bb.getIR())
            {
                if(irNode.getType() == IRBase::IRType::StinkyTofu)
                {
                    instructions.push_back(cast<StinkyInstruction>(&irNode));
                }
            }

            if(instructions.size() < 2)
                return 0; // Need at least 2 instructions for dependencies

            // Build position and delay info map for all instructions
            std::unordered_map<StinkyInstruction*, DelayInfo> instDelayInfo;

            // Track ALU type counts (forward scan to build delay info)
            std::unordered_map<DelayAluType, unsigned> typeCount;
            typeCount[DelayAluType::VALU]  = 0;
            typeCount[DelayAluType::SALU]  = 0;
            typeCount[DelayAluType::TRANS] = 0;

            for(size_t i = 0; i < instructions.size(); ++i)
            {
                StinkyInstruction* inst     = instructions[i];
                DelayAluType       instType = classifyDelayAluType(inst);

                // Record delay info for this instruction
                instDelayInfo[inst] = {instType, typeCount[instType], static_cast<unsigned>(i)};

                // Increment type count if this is an ALU instruction
                if(instType != DelayAluType::OTHER)
                {
                    typeCount[instType]++;
                }
            }

            // Map of instruction -> s_delay_alu to insert before it
            std::unordered_map<StinkyInstruction*, PendingDelayAlu> delaysToInsert;

            // Analyze dependencies using use-def chains
            for(StinkyInstruction* inst : instructions)
            {
                // Check each source instruction (dependencies are already computed!)
                for(StinkyInstruction* srcInst : inst->sources)
                {
                    auto srcInfoIt = instDelayInfo.find(srcInst);
                    if(srcInfoIt == instDelayInfo.end())
                        continue;

                    DelayAluType defType     = srcInfoIt->second.type;
                    unsigned     maxLookback = ALU_DEP_MAX.at(defType);

                    if(maxLookback == 0)
                        continue; // Producer doesn't need delay

                    auto instInfoIt = instDelayInfo.find(inst);
                    if(instInfoIt == instDelayInfo.end())
                        continue;

                    // Calculate distance in terms of producer's type
                    unsigned dist = instInfoIt->second.typeCount - srcInfoIt->second.typeCount;

                    // Only insert delay if within valid range
                    if(dist > 0 && dist <= maxLookback)
                    {
                        // Insert s_delay_alu before this instruction
                        delaysToInsert[inst] = {defType, dist};
                        break; // Only handle first dependency for now
                    }
                }
            }

            // Create IRBuilder for this basic block
            auto irBuilder = passCtx.getIRBuilder<StinkyInstIRBuilder>(bb.getIR(), arch);

            // Insert s_delay_alu instructions (traverse forwards to get iterators)
            int     insertCount = 0;
            IRList& irlist      = bb.getIR();
            for(auto it = irlist.begin(); it != irlist.end(); ++it)
            {
                if(it->getType() != IRBase::IRType::StinkyTofu)
                    continue;

                StinkyInstruction* inst    = cast<StinkyInstruction>(&*it);
                auto               delayIt = delaysToInsert.find(inst);
                if(delayIt == delaysToInsert.end())
                    continue;

                // Create s_delay_alu instruction before this instruction
                if(!createDelayAluInst(delayIt->second, irBuilder, it, arch))
                    continue;

                insertCount++;
            }

            return insertCount;
        }

        bool createDelayAluInst(const PendingDelayAlu& delay,
                                StinkyInstIRBuilder&   irBuilder,
                                IRList::iterator       insertPoint,
                                GfxArchID              arch)
        {
            // Architecture check already done in run() - we only reach here for RDNA3
            // TODO: Use proper GFX::s_delay_alu enum when available
            uint16_t          isaOpcode = getMnemonicToIsaOpcode("s_delay_alu", arch);
            const HwInstDesc* desc      = getMCIDByIsaOp(static_cast<IsaOpcode>(isaOpcode), arch);

            StinkyInstruction* inst = irBuilder.createStinkyInstBefore(insertPoint, desc);

            // Build operand string based on delay type and distance
            std::string operand;
            switch(delay.type)
            {
            case DelayAluType::VALU:
                operand = "instid0(VALU_DEP_" + std::to_string(delay.dist) + ")";
                break;
            case DelayAluType::SALU:
                operand = "instid0(SALU_CYCLE_1)"; // SALU always uses CYCLE_1
                break;
            case DelayAluType::TRANS:
                operand = "instid0(TRANS32_DEP_" + std::to_string(delay.dist) + ")";
                break;
            default:
                operand = "instid0(NO_DEP)";
                break;
            }

            // Add operand as a literal string (s_delay_alu has special operand syntax)
            StinkyRegister symbolicOperand;
            symbolicOperand.dataType     = StinkyRegister::Type::LiteralString;
            symbolicOperand.literalValue = operand;
            inst->srcRegs.push_back(symbolicOperand);

            return true;
        }
    };

    char DelayAluInsertionPassImpl::ID = 0;

} // anonymous namespace

namespace stinkytofu
{
    std::unique_ptr<Pass> createDelayAluInsertionPass()
    {
        return std::make_unique<DelayAluInsertionPassImpl>();
    }

} // namespace stinkytofu
