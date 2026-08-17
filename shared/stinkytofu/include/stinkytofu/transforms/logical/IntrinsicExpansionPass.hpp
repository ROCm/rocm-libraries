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

#include <memory>
#include <unordered_map>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/PatternParser.hpp"

namespace stinkytofu {
class LogicalInstruction;

/**
 * @brief Pass that expands IntrinsicCall instructions using IntrinsicRegistry
 *
 * This pass:
 *   1. Finds all IntrinsicCall instructions in the function
 *   2. Looks up the intrinsic definition from IntrinsicRegistry
 *   3. Expands the call into concrete LogicalInstructions
 *   4. Replaces the IntrinsicCall with the expanded instructions
 *
 * Example:
 *   Before:
 *     IntrinsicCall("ReluF32", [v0, v1, v2])  // dest, src, temp
 *
 *   After (expanded from intrinsics.st.bc):
 *     v2 = VCmpGtF32(v1, 0.0)
 *     v0 = VCndMaskB32(0.0, v1, v2)
 *
 * This pass must run BEFORE ToStinkyAsmPass, as it operates on LogicalIR.
 */
class IntrinsicExpansionPass : public Pass {
   public:
    IntrinsicExpansionPass();
    ~IntrinsicExpansionPass() override;

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override;

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return "IntrinsicExpansion";
    }

   private:
    static char ID;

    /**
     * @brief Expand all IntrinsicCall instructions in a BasicBlock
     *
     * @param bb BasicBlock to process
     */
    void expandIntrinsicsInBlock(BasicBlock& bb);

    /**
     * @brief Expand a single IntrinsicCall instruction
     *
     * @param call IntrinsicCall instruction to expand
     * @return Vector of expanded LogicalInstructions (raw pointers for IRList)
     */
    std::vector<LogicalInstruction*> expandIntrinsic(LogicalInstruction* call);

    /**
     * @brief Create a LogicalInstruction from intrinsic body instruction
     *
     * @param instDef Intrinsic instruction definition
     * @param regMap Map from intrinsic argument names to actual registers
     * @return Created LogicalInstruction, or nullptr on error
     */
    LogicalInstruction* createInstructionFromIntrinsic(
        const IntrinsicInstruction& instDef,
        const std::unordered_map<std::string, StinkyRegister>& regMap);

    /**
     * @brief Resolve operand string to register
     *
     * @param operand Operand string (register name or immediate value)
     * @param regMap Map from intrinsic argument names to actual registers
     * @param outReg Output register if operand is a register
     * @return true if operand is a register, false if it's an immediate
     */
    bool resolveOperand(const std::string& operand,
                        const std::unordered_map<std::string, StinkyRegister>& regMap,
                        StinkyRegister& outReg);
};

/**
 * @brief Factory function to create IntrinsicExpansionPass
 * @return Unique pointer to IntrinsicExpansionPass
 */
STINKYTOFU_EXPORT std::unique_ptr<Pass> createIntrinsicExpansionPass();

}  // namespace stinkytofu
