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

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/Conditional.hpp>

namespace rocRoller
{
    RegisterComponent(ConditionalGenerator);

    template <>
    std::shared_ptr<TernaryArithmeticGenerator<Expression::Conditional>>
        GetGenerator<Expression::Conditional>(Register::ValuePtr dst,
                                              Register::ValuePtr lhs,
                                              Register::ValuePtr r1hs,
                                              Register::ValuePtr r2hs,
                                              Expression::Conditional const&)
    {
        return Component::Get<TernaryArithmeticGenerator<Expression::Conditional>>(
            getContextFromValues(dst, lhs, r1hs, r2hs),
            dst->regType(),
            dst->variableType().dataType);
    }

    Generator<Instruction> ConditionalGenerator::generate(Register::ValuePtr dest,
                                                          Register::ValuePtr cond,
                                                          Register::ValuePtr r1hs,
                                                          Register::ValuePtr r2hs,
                                                          Expression::Conditional const&)
    {
        AssertFatal(cond != nullptr);
        AssertFatal(r1hs != nullptr);
        AssertFatal(r2hs != nullptr);
        AssertFatal(dest->valueCount() == 1,
                    "Non-1 value count not supported",
                    ShowValue(dest->valueCount()),
                    ShowValue(dest->toString()));

        if(dest->regType() == Register::Type::Scalar)
        {
            Register::ValuePtr left, right;

            // Swap sides depending if we use SCC or !SCC
            if(!cond->isSCC())
            {
                co_yield(
                    Instruction::Lock(Scheduling::Dependency::SCC, "Start of Conditional(SCC)"));
                co_yield generateOp<Expression::Equal>(
                    m_context->getSCC(), cond, Register::Value::Literal(0));
                left  = std::move(r2hs);
                right = std::move(r1hs);
            }
            else
            {
                left  = std::move(r1hs);
                right = std::move(r2hs);
            }

            auto const elementSize = dest->variableType().getElementSize();
            if(elementSize == 8)
            {
                co_yield_(Instruction("s_cselect_b64", {dest}, {left, right}, {}, ""));
            }
            else if(elementSize == 4)
            {
                co_yield_(Instruction("s_cselect_b32", {dest}, {left, right}, {}, ""));
            }
            else
            {
                AssertFatal(false, "Unsupported scalar size ", ShowValue(elementSize));
            }

            if(!cond->isSCC())
            {
                co_yield(Instruction::Unlock("End of Conditional(SCC)"));
            }
        }
        else
        {
            AssertFatal(cond->isVCC() || cond->regType() == Register::Type::Scalar,
                        ShowValue(cond->regType()));

            auto const elementSize = dest->variableType().getElementSize();
            if(elementSize == 8)
            {
                co_yield_(Instruction("v_cndmask_b32",
                                      {dest->subset({0})},
                                      {r2hs->subset({0}), r1hs->subset({0}), cond},
                                      {},
                                      ""));
                co_yield_(Instruction("v_cndmask_b32",
                                      {dest->subset({1})},
                                      {r2hs->subset({1}), r1hs->subset({1}), cond},
                                      {},
                                      ""));
            }
            else if(elementSize == 4)
            {
                co_yield_(Instruction("v_cndmask_b32", {dest}, {r2hs, r1hs, cond}, {}, ""));
            }
            else
            {
                AssertFatal(false, "Unsupported size ", ShowValue(elementSize));
            }
        }
    }
}
