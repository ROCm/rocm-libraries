/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022 Advanced Micro Devices, Inc.
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

#include "MatrixMultiply_fwd.hpp"

#include <memory>
#include <tuple>

#include "../../Context.hpp"
#include "../../DataTypes/DataTypes.hpp"
#include "../../Expression.hpp"
#include "../../InstructionValues/Register_fwd.hpp"
#include "../../Utilities/Generator.hpp"

namespace rocRoller
{
    namespace InstructionGenerators
    {
        struct MatrixMultiply
        {
            /**
             * Context, accumulation type, input type.
             */
            using Argument = std::tuple<std::shared_ptr<Context>, DataType, DataType>;
            using Base     = MatrixMultiply;

            static const std::string Name;

            /**
             * Initialises DEST = 0.
             */
            virtual Generator<Instruction> zero(std::shared_ptr<Register::Value> dest) = 0;

            /**
             * Performs matrix multiplication: DEST = LHS * RHS + DEST
             * using MFMA instructions.
             *
             * LHS and RHS are stored in registers.  DEST is accumulated.
             *
             * LHS is M x K with B batches.  RHS is K x N with B batches.
             */
            virtual Generator<Instruction> mul(std::shared_ptr<Register::Value> dest,
                                               std::shared_ptr<Register::Value> lhs,
                                               std::shared_ptr<Register::Value> rhs,
                                               int                              M,
                                               int                              N,
                                               int                              K,
                                               int                              B)
                = 0;
        };

        template <DataType ACC, DataType INPUT>
        struct MatrixMultiplyGenerator : public MatrixMultiply
        {
            using Base = MatrixMultiply;

            MatrixMultiplyGenerator<ACC, INPUT>(std::shared_ptr<Context> context)
                : m_context(context){};

            static const std::string Name;
            static const std::string Basename;

            virtual ~MatrixMultiplyGenerator<ACC, INPUT>() = default;

            static bool Match(Argument const& arg)
            {
                auto atype = std::get<1>(arg);
                auto vtype = std::get<2>(arg);
                return atype == ACC && vtype == INPUT;
            }

            static std::shared_ptr<MatrixMultiply> Build(Argument const& arg)
            {
                auto context = std::get<0>(arg);
                return std::make_shared<MatrixMultiplyGenerator<ACC, INPUT>>(context);
            }

            virtual Generator<Instruction> zero(std::shared_ptr<Register::Value> dest) override;

            virtual Generator<Instruction> mul(std::shared_ptr<Register::Value> dest,
                                               std::shared_ptr<Register::Value> lhs,
                                               std::shared_ptr<Register::Value> rhs,
                                               int                              M,
                                               int                              N,
                                               int                              K,
                                               int                              B) override;

        protected:
            std::shared_ptr<Context> m_context;
        };

    }
}
