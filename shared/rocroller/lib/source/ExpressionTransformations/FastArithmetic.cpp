// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
            std::vector<ExpressionTransformType> transforms = {
                splitBitfieldCombine,
                lowerBitfieldCombine,
                [this](auto e) { return fastDivision(e, m_context); },
                convertPropagation, // go after fastDivision as it might change data types of denominator and numerator
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
