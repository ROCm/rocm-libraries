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

#include "ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu
{
    namespace test
    {

        /**
     * @brief Add a vector of instructions to an IRList
     *
     * Helper function for tests using StinkyTofu builder, which returns
     * std::vector<StinkyInstruction*> to support composite instructions.
     *
     * @param irList The IRList to append instructions to
     * @param insts Vector of instructions from StinkyTofu builder
     * @return The first instruction, or nullptr if vector is empty
     *
     * Usage:
     * @code
     *   StinkyTofu builder({9, 4, 2});
     *   IRList& irList = bb->getIR();
     *
     *   // Add single instruction
     *   auto* add = addToIRList(irList, builder.VAddF32(vgpr(0), vgpr(1), vgpr(2)));
     *
     *   // Add composite instruction (might be multiple instructions)
     *   auto* pk = addToIRList(irList, builder.VAddPKF32(...));
     * @endcode
     */
        inline StinkyInstruction* addToIRList(IRList&                           irList,
                                              std::vector<StinkyInstruction*>&& insts)
        {
            if(insts.empty())
                return nullptr;

            for(auto* inst : insts)
            {
                irList.push_back(inst);
            }
            return insts[0];
        }

        /**
     * @brief Convenience function to create a VGPR register
     */
        inline StinkyRegister vgpr(int idx, int count = 1)
        {
            return StinkyRegister("v", idx, count);
        }

        /**
     * @brief Convenience function to create an SGPR register
     */
        inline StinkyRegister sgpr(int idx, int count = 1)
        {
            return StinkyRegister("s", idx, count);
        }

        /**
     * @brief Convenience function to create an AGPR register
     */
        inline StinkyRegister agpr(int idx, int count = 1)
        {
            return StinkyRegister("a", idx, count);
        }

        /**
     * @brief Convenience function to create a literal constant register
     */
        inline StinkyRegister literal(double value)
        {
            StinkyRegister reg;
            reg.dataType      = StinkyRegister::Type::LiteralDouble;
            reg.literalDouble = value;
            return reg;
        }

    } // namespace test
} // namespace stinkytofu
