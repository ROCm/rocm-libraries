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

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

namespace rocRoller
{
    namespace Expression
    {
        FastArithmetic::FastArithmetic(ContextPtr context)
            : m_context(context)
        {
        }

        std::vector<ExpressionTransformType> FastArithmetic::getTransforms() const
        {
            std::vector<ExpressionTransformType> transforms
                = {splitBitfieldCombine,
                   lowerBitfieldCombine,
                   convertPropagation,
                   [this](auto e) { return fastDivision(e, m_context); },
                   simplify,
                   lowerExponential,
                   fastMultiplication,
                   lowerUnsignedArithmeticShiftR,
                   fuseAssociative,
                   combineShifts,
                   fuseTernary,
                   [this](auto e) { return launchTimeSubExpressions(e, m_context); },
                   convertPropagation,
                   simplify};

            return transforms;
        }

        ExpressionPtr FastArithmetic::applyTransforms(
            ExpressionPtr expr, const std::vector<ExpressionTransformType>& transforms) const
        {
            for(const auto& transform : transforms)
                expr = transform(expr);
            return expr;
        }

        ExpressionPtr FastArithmetic::operator()(ExpressionPtr x) const
        {
            if(!x)
            {
                return x;
            }
            ExpressionPtr orig = x;

            auto transforms = getTransforms();
            x               = applyTransforms(x, transforms);

            if(!identical(orig, x))
            {
                // auto comment = Instruction::Comment(
                //     concatenate("FastArithmetic:", ShowValue(orig), ShowValue(x)));
                // m_context->schedule(comment);

                auto origResultType = resultType(orig);
                AssertFatal(origResultType.varType == resultType(x).varType,
                            ShowValue(origResultType),
                            ShowValue(resultType(x)),
                            ShowValue(orig),
                            ShowValue(x));
            }

            return x;
        }
    }
}
