/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
