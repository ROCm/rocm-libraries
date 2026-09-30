// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>

namespace rocRoller
{
    namespace Arithmetic
    {
        /**
         * @brief Represent a single Register::Value as two Register::Values each the size of a single DWord
        */
        void get2LiteralDwords(Register::ValuePtr& lsd,
                               Register::ValuePtr& msd,
                               Register::ValuePtr  input);

        /**
         * @brief Get the modifier string for MFMA's input matrix types
        */
        std::string getModifier(DataType dataType);

        /**
         * @brief Get the modifier string for MI's input matrix scale types
         */
        std::string getScaleTypeModifier(DataType dtype);

        /**
         * Returns opsel modifiers to index byte `lhsByte` for a lhs operand and `rhsByte` for a rhs operand.
         *
         * This means:
         *
         * lhsByte and rhsByte must be in {0, 1, 2, 3}
         *
         * Returns
         * "op_sel[a0, b0]", "op_sel_hi[a1, b1]"
         *
         * where
         * a0 = bit 0 of lhsByte
         * a1 = bit 1 of lhsByte
         * b0 = bit 0 of rhsByte
         * b1 = bit 1 of rhsByte
         *
         */
        std::tuple<std::string, std::string> getOpselModifiers2xByte(uint lhsByte, uint rhsByte);
    }
}
