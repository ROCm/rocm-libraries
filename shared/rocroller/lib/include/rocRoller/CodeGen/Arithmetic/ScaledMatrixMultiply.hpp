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

#include "ScaledMatrixMultiply_fwd.hpp"

#include <memory>
#include <tuple>

#include "ArithmeticGenerator.hpp"

namespace rocRoller
{
    namespace InstructionGenerators
    {
        struct ScaledMatrixMultiply
        {
            /**
             * Context, accumulation type, input type.
             */
            using Argument = std::tuple<ContextPtr, DataType, DataType>;
            using Base     = ScaledMatrixMultiply;

            static const std::string Basename;

            /**
             * Performs matrix multiplication: dest = matA * matB + matC
             * using matrix instructions with scaleA and scaleB.
             *
             * matA and matB are stored in registers.  matC is the accumulator and can be the same as dest.
             *
             * matA is M x K with B batches.  matB is K x N with B batches.
             */
            virtual Generator<Instruction> mul(Register::ValuePtr  dest,
                                               Register::ValuePtr  matA,
                                               Register::ValuePtr  matB,
                                               Register::ValuePtr  matC,
                                               Register::ValuePtr  scaleA,
                                               Register::ValuePtr  scaleB,
                                               int                 M,
                                               int                 N,
                                               int                 K,
                                               std::optional<uint> maybeScaleBlockSize)
                = 0;
        };

        struct ScaledMatrixMultiplyGenerator : public ScaledMatrixMultiply
        {
            static bool constexpr isValidInputType(auto const vtype)
            {
                return (vtype == DataType::FP8x4 || vtype == DataType::BF8x4
                        || vtype == DataType::FP6x16 || vtype == DataType::BF6x16
                        || vtype == DataType::FP4x8);
            }

            static bool constexpr isValidOutputType(auto const atype)
            {
                return atype == DataType::Float;
            }

            using Base = ScaledMatrixMultiply;

            ScaledMatrixMultiplyGenerator(ContextPtr context)
                : m_context(context){};

            static const std::string Name;

            static bool Match(Argument const& arg)
            {
                auto atype = std::get<1>(arg);
                auto vtype = std::get<2>(arg);
                return isValidOutputType(atype) && isValidInputType(vtype);
            }

            static ScaledMatrixMultiplyPtr Build(Argument const& arg)
            {
                if(not Match(arg))
                    return nullptr;

                auto context = std::get<0>(arg);
                return std::make_shared<ScaledMatrixMultiplyGenerator>(context);
            }

            virtual Generator<Instruction> mul(Register::ValuePtr  dest,
                                               Register::ValuePtr  matA,
                                               Register::ValuePtr  matB,
                                               Register::ValuePtr  matC,
                                               Register::ValuePtr  scaleA,
                                               Register::ValuePtr  scaleB,
                                               int                 M,
                                               int                 N,
                                               int                 K,
                                               std::optional<uint> maybeScaleBlockSize) override;

        protected:
            ContextPtr m_context;
        };

    }
}
