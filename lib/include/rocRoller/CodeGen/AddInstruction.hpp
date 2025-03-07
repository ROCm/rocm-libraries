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

#include <rocRoller/CodeGen/Instruction.hpp>

namespace rocRoller
{
    Instruction ScalarAddInt32(ContextPtr         ctx,
                               Register::ValuePtr dest,
                               Register::ValuePtr lhs,
                               Register::ValuePtr rhs,
                               std::string        comment = "");

    Instruction ScalarAddUInt32(ContextPtr         ctx,
                                Register::ValuePtr dest,
                                Register::ValuePtr lhs,
                                Register::ValuePtr rhs,
                                std::string        comment = "");

    Instruction ScalarAddUInt32CarryInOut(ContextPtr         ctx,
                                          Register::ValuePtr dest,
                                          Register::ValuePtr lhs,
                                          Register::ValuePtr rhs,
                                          std::string        comment = "");

    Instruction VectorAddUInt32(ContextPtr         ctx,
                                Register::ValuePtr dest,
                                Register::ValuePtr lhs,
                                Register::ValuePtr rhs,
                                std::string        comment = "");

    Instruction VectorAddUInt32CarryOut(ContextPtr         ctx,
                                        Register::ValuePtr dest,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        std::string        comment = "");

    Instruction VectorAddUInt32CarryOut(ContextPtr         ctx,
                                        Register::ValuePtr dest,
                                        Register::ValuePtr carry,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        std::string        comment = "");

    Instruction VectorAddUInt32CarryInOut(ContextPtr         ctx,
                                          Register::ValuePtr dest,
                                          Register::ValuePtr carryOut,
                                          Register::ValuePtr carryIn,
                                          Register::ValuePtr lhs,
                                          Register::ValuePtr rhs,
                                          std::string        comment = "");

    Instruction VectorAddUInt32CarryInOut(ContextPtr         ctx,
                                          Register::ValuePtr dest,
                                          Register::ValuePtr lhs,
                                          Register::ValuePtr rhs,
                                          std::string        comment = "");

    Instruction VectorAdd3UInt32(ContextPtr         ctx,
                                 Register::ValuePtr dest,
                                 Register::ValuePtr lhs,
                                 Register::ValuePtr rhs1,
                                 Register::ValuePtr rhs2,
                                 std::string        comment = "");
}
