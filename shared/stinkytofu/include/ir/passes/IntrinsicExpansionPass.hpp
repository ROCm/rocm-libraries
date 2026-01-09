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

#include "ir/IRModule.hpp"
#include "ir/IntrinsicLibrary.hpp"
#include "ir/asm/StinkyAsmIR.hpp" // For StinkyRegister
#include <memory>
#include <unordered_map>

namespace stinkytofu
{
    /**
     * @brief Pass that expands IntrinsicCall instructions using IntrinsicLibrary
     *
     * This pass:
     *   1. Finds all IntrinsicCall instructions in the module
     *   2. Looks up the intrinsic definition from IntrinsicLibrary
     *   3. Expands the call into concrete high-level IR instructions
     *   4. Replaces the IntrinsicCall with the expanded instructions
     *
     * Example:
     *   Before:
     *     IntrinsicCall("ReluF32", [v0, v1, v2])  // dest, src, temp
     *
     *   After:
     *     v2 = v_cmp_gt_f32(v1, 0.0)
     *     v0 = v_select_f32(v2, v1, 0.0)
     */
    class IntrinsicExpansionPass
    {
    public:
        /**
         * @brief Construct expansion pass with intrinsic library
         *
         * @param library Shared pointer to loaded IntrinsicLibrary
         */
        explicit IntrinsicExpansionPass(std::shared_ptr<IntrinsicLibrary> library);

        /**
         * @brief Run the expansion pass on an IR module
         *
         * @param module IR module to transform
         * @return true if any expansions were performed, false otherwise
         */
        bool run(IRModule* module);

    private:
        std::shared_ptr<IntrinsicLibrary> library_;

        /**
         * @brief Expand a single IntrinsicCall instruction
         *
         * @param call IntrinsicCall instruction to expand
         * @return Vector of expanded IR instructions, or empty on error
         */
        std::vector<IRInstruction*> expandIntrinsic(IRInstruction* call);

        /**
         * @brief Create an IR instruction from intrinsic body instruction
         *
         * @param inst Intrinsic instruction definition
         * @param regMap Map from intrinsic argument names to actual registers
         * @return Created IRInstruction, or nullptr on error
         */
        IRInstruction*
            createInstruction(const IntrinsicInstruction&                            inst,
                              const std::unordered_map<std::string, StinkyRegister>& regMap);

        /**
         * @brief Parse operand string and resolve to register or immediate
         *
         * @param operand Operand string (register name or immediate value)
         * @param regMap Map from intrinsic argument names to actual registers
         * @param outReg Output register if operand is a register
         * @return true if operand is a register, false if it's an immediate
         */
        bool parseOperand(const std::string&                                     operand,
                          const std::unordered_map<std::string, StinkyRegister>& regMap,
                          StinkyRegister&                                        outReg);
    };

} // namespace stinkytofu
