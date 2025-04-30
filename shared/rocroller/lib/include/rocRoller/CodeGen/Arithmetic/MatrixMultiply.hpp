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

#include <memory>
#include <tuple>

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply_fwd.hpp>

namespace rocRoller
{
    namespace InstructionGenerators
    {
        struct MatrixMultiply
        {
            /**
             * Context, accumulation type, input type.
             */
            using Argument = ContextPtr;
            using Base     = MatrixMultiply;

            static const std::string Basename;

            /**
             * Performs matrix multiplication: DEST = LHS * R1HS + R2HS
             * using MFMA instructions.
             *
             * LHS and R1HS are stored in registers.  R2HS is the accumulator and can be the same as DEST.
             *
             * LHS is M x K with B batches.  RHS is K x N with B batches.
             */
            virtual Generator<Instruction> mul(Register::ValuePtr dest,
                                               Register::ValuePtr lhs,
                                               Register::ValuePtr r1hs,
                                               Register::ValuePtr r2hs,
                                               int                M,
                                               int                N,
                                               int                K,
                                               int                B)
                = 0;
        };

        struct MatrixMultiplyGenerator : public MatrixMultiply
        {
            using Base = MatrixMultiply;

            MatrixMultiplyGenerator(ContextPtr context)
                : m_context(context){};

            static const std::string Name;

            static bool Match(Argument const& arg)
            {
                return true;
            }

            static MatrixMultiplyPtr Build(Argument const& arg)
            {
                auto context = arg;
                return std::make_shared<MatrixMultiplyGenerator>(context);
            }

            virtual Generator<Instruction> mul(Register::ValuePtr dest,
                                               Register::ValuePtr lhs,
                                               Register::ValuePtr r1hs,
                                               Register::ValuePtr r2hs,
                                               int                M,
                                               int                N,
                                               int                K,
                                               int                B) override;

        protected:
            ContextPtr m_context;
        };

    }
}
