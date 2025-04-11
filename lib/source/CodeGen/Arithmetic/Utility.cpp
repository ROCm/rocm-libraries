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

#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>

namespace rocRoller
{
    namespace Arithmetic
    {
        void get2LiteralDwords(Register::ValuePtr& lsd,
                               Register::ValuePtr& msd,
                               Register::ValuePtr  input)
        {
            assert(input->regType() == Register::Type::Literal);
            uint64_t value = std::visit(
                [](auto v) {
                    using T = std::decay_t<decltype(v)>;
                    AssertFatal((std::is_integral_v<T>));
                    if constexpr(std::is_pointer_v<T>)
                    {
                        return reinterpret_cast<uint64_t>(v);
                    }
                    else
                    {
                        return static_cast<uint64_t>(v);
                    }
                },
                input->getLiteralValue());

            lsd = Register::Value::Literal(static_cast<uint32_t>(value));
            msd = Register::Value::Literal(static_cast<uint32_t>(value >> 32));
        }

        std::string getModifier(DataType dtype)
        {
            switch(dtype)
            {
            case DataType::FP8x4:
                return "0b000";
            case DataType::BF8x4:
                return "0b001";
            case DataType::FP6x16:
                return "0b010";
            case DataType::BF6x16:
                return "0b011";
            case DataType::FP4x8:
                return "0b100";
            default:
                Throw<FatalError>("Unable to determine MI modifier: unhandled data type.",
                                  ShowValue(dtype));
            }
        }
    }
}
