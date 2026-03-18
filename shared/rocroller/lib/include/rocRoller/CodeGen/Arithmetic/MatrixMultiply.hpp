// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
             * LHS is mi.m x mi.k with mi.b batches.  RHS is mi.k x mi.n with mi.b batches.
             */
            virtual Generator<Instruction> mul(Register::ValuePtr  dest,
                                               Register::ValuePtr  lhs,
                                               Register::ValuePtr  r1hs,
                                               Register::ValuePtr  r2hs,
                                               MatrixMultiplySizes mi)
                = 0;
        };

        struct MatrixMultiplyGenerator : public MatrixMultiply
        {
            using Base = MatrixMultiply;

            MatrixMultiplyGenerator(ContextPtr context)
                : m_context(context){};

            inline static const std::string Name = "MatrixMultiplyGenerator";

            static bool Match(Argument const& arg)
            {
                return true;
            }

            static MatrixMultiplyPtr Build(Argument const& arg)
            {
                auto context = arg;
                return std::make_shared<MatrixMultiplyGenerator>(context);
            }

            virtual Generator<Instruction> mul(Register::ValuePtr  dest,
                                               Register::ValuePtr  lhs,
                                               Register::ValuePtr  r1hs,
                                               Register::ValuePtr  r2hs,
                                               MatrixMultiplySizes mi) override;

        protected:
            ContextPtr m_context;
        };

    }
}
