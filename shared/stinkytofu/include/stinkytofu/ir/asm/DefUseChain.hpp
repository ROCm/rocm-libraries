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
#pragma once

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"

#include <unordered_map>

namespace stinkytofu
{
    /// Builds a use-def chain for instructions in the given IRList.
    ///
    /// This links each instruction's sources to their most recent definitions
    /// and each instruction's users to the instructions that use its results.
    ///
    /// The use-def chain is stored directly in the StinkyInstruction objects:
    ///   * inst->sources: Instructions that define the registers used by this instruction
    ///   * inst->users:   Instructions that use the results of this instruction
    ///
    /// Time Complexity: O(n) where n is the number of instructions
    ///
    /// Usage:
    ///   buildUseDefChain(basicBlock.getIR());
    ///   for(auto* src : inst->sources) { /* use source */ }
    ///   for(auto* user : inst->users) { /* use user */ }
    ///
    /// Note: This function assumes instructions are in top-down order.
    ///       It handles multiple consecutive registers (e.g., regIdx 0,1,2,3).
    inline void buildUseDefChain(IRList& insts)
    {
        struct RegisterKey
        {
            RegType  type;
            unsigned regIdx;

            bool operator==(const RegisterKey& o) const noexcept
            {
                return regIdx == o.regIdx && type == o.type;
            }
        };

        struct RegisterKeyHash
        {
            size_t operator()(const RegisterKey& k) const noexcept
            {
                size_t h1 = std::hash<int>{}(static_cast<int>(k.type));
                size_t h2 = std::hash<unsigned>{}(k.regIdx);
                return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
            }
        };

        std::unordered_map<RegisterKey, StinkyInstruction*, RegisterKeyHash> lastDef;

        // Clear existing chains
        for(IRBase& ir : insts)
        {
            if(ir.getType() == IRBase::IRType::StinkyTofu)
            {
                auto* inst = cast<StinkyInstruction>(&ir);
                inst->sources.clear();
                inst->users.clear();
            }
        }

        // Build use-def chains for each instruction in top-down order
        for(IRBase& ir : insts)
        {
            if(ir.getType() != IRBase::IRType::StinkyTofu)
                continue;

            auto* inst = cast<StinkyInstruction>(&ir);

            // Link sources: for each source register, find its most recent definition
            for(const auto& srcReg : inst->getSrcRegs())
            {
                if(!srcReg.isRegister())
                    continue;

                // Handle consecutive registers (e.g., v[0:3] means v0, v1, v2, v3)
                for(unsigned i = 0; i < srcReg.reg.num; ++i)
                {
                    RegisterKey key{srcReg.reg.type, srcReg.reg.idx + i};

                    if(lastDef.count(key))
                    {
                        StinkyInstruction* defInst = lastDef[key];

                        // Add bidirectional link
                        inst->sources.push_back(defInst);
                        defInst->users.push_back(inst);
                    }
                }
            }

            // Update lastDef: mark this instruction as the defining instruction for its destinations
            for(const auto& destReg : inst->getDestRegs())
            {
                if(!destReg.isRegister())
                    continue;

                // Handle consecutive registers
                for(unsigned i = 0; i < destReg.reg.num; ++i)
                {
                    RegisterKey key{destReg.reg.type, destReg.reg.idx + i};
                    lastDef[key] = inst;
                }
            }
        }
    }

    /// Builds use-def chain for a single BasicBlock
    inline void buildUseDefChain(BasicBlock& bb)
    {
        buildUseDefChain(bb.getIR());
    }

    /// Builds use-def chain for all BasicBlocks in a Function
    inline void buildUseDefChain(Function& func)
    {
        for(BasicBlock& bb : func)
        {
            buildUseDefChain(bb.getIR());
        }
    }

} // namespace stinkytofu
