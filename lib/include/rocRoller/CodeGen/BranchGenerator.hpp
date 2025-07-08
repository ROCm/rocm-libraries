/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    /**
     * @brief Generator for generating conditional and unconditional branches.
     */
    class BranchGenerator
    {
    public:
        BranchGenerator(ContextPtr);

        ~BranchGenerator();

        /**
         * @brief Generate an unconditional branch.
         *
         * @param destLabel A register value with regType() == Register::Type::Label
         * @param comment="" An optional comment to include in the generated instruction
         * @return Generator<Instruction>
         */
        Generator<Instruction> branch(Register::ValuePtr destLabel, std::string comment = "");

        /**
         * @brief Generate an conditional branch.
         *
         * If condition is SCC or VCC, the corressponding branch instruction is generated.
         * If condition is a scalar register it is copied into VCC and a VCC branch is generated.
         *
         * @param destLabel A register value with regType() == Register::Type::Label
         * @param condition The register to conditionally branch on.
         * @param zero=true If true branch zero is used, if false branch nonzero is used
         * @param comment="" An optional comment to include in the generated instruction
         * @return Generator<Instruction>
         */
        Generator<Instruction> branchConditional(Register::ValuePtr destLabel,
                                                 Register::ValuePtr condition,
                                                 bool               zero    = true,
                                                 std::string        comment = "");

        /**
         * @brief Generates a branch on zero instruction.
         *
         * Calls branchConditional with zero=true.
         *
         * @param destLabel A register value with regType() == Register::Type::Label
         * @param condition The register to conditionally branch on.
         * @param comment="" An optional comment to include in the generated instruction
         * @return Generator<Instruction>
         * @sa branchConditional
         */
        Generator<Instruction> branchIfZero(Register::ValuePtr destLabel,
                                            Register::ValuePtr condition,
                                            std::string        comment = "");

        /**
         * @brief Generates a branch on nonzero instruction.
         *
         * Calls branchConditional with zero=false.
         *
         * @param destLabel A register value with regType() == Register::Type::Label
         * @param condition The register to conditionally branch on.
         * @param comment="" An optional comment to include in the generated instruction
         * @return Generator<Instruction>
         * @sa branchConditional
         */
        Generator<Instruction> branchIfNonZero(Register::ValuePtr destLabel,
                                               Register::ValuePtr condition,
                                               std::string        comment = "");

        /**
         * @brief Creates a result register for a condition expression
         *
         * @param expr Conditional expression
         * @return Result
         */
        Register::ValuePtr resultRegister(Expression::ExpressionPtr expr);

    private:
        std::weak_ptr<Context> m_context;
    };
}

#include <rocRoller/CodeGen/BranchGenerator_impl.hpp>
