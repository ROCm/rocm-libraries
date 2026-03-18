// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply_fwd.hpp>
#include <rocRoller/CodeGen/Arithmetic/ScaledMatrixMultiply_fwd.hpp>

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
             * Performs matrix multiplication: dest = scale(scaleA, matA) * scale(scaleB, matB) + matC
             * using matrix instructions with scaleA and scaleB.
             *
             * matA and matB are stored in registers.  matC is the accumulator and can be the same as dest.
             *
             * matA is mi.m x mi.k with mi.b batches.  matB is mi.k x mi.n with mi.b batches.
             */
            virtual Generator<Instruction> mul(Register::ValuePtr  dest,
                                               Register::ValuePtr  matA,
                                               Register::ValuePtr  matB,
                                               Register::ValuePtr  matC,
                                               Register::ValuePtr  scaleA,
                                               Register::ValuePtr  scaleB,
                                               MatrixMultiplySizes mi,
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

            inline static const std::string Name = "ScaledMatrixMultiplyGenerator";

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
                                               MatrixMultiplySizes mi,
                                               std::optional<uint> maybeScaleBlockSize) override;

        protected:
            ContextPtr m_context;
        };

    }
}
