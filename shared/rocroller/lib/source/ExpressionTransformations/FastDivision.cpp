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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Logging.hpp>

#include <bit>

#define cast_to_unsigned(N) static_cast<typename std::make_unsigned<T>::type>(N)

namespace rocRoller
{
    namespace Expression
    {
        /**
         * Fast Division
         *
         * Attempt to replace division operations found within an expression with faster
         * operations.
         */

        std::tuple<ExpressionPtr, ExpressionPtr, ExpressionPtr>
            getMagicMultipleShiftAndSign(ExpressionPtr denominator, ContextPtr context)
        {
            auto multiple = launchTimeSubExpressions(magicMultiple(denominator), context);

            auto resultType = resultVariableType(denominator);
            auto typeInfo   = DataTypeInfo::Get(resultType);

            if(!typeInfo.isSigned)
            {
                auto shifts = launchTimeSubExpressions(magicShifts(denominator), context);

                return {multiple, shifts, nullptr};
            }

            auto bitfield = launchTimeSubExpressions(magicShiftAndSign(denominator), context);

            auto mask   = typeInfo.elementBits - 1;
            auto shifts = bitfield & literal(mask);

            // Sign is in the most significant bit of the least significant byte
            // (i.e. bit 7). We want every bit to match bit 7.
            // 1. Shift it left into the most significant bit.
            // 2. Arithmetic shift right by the width of the type to sign
            // extend into every bit.
            int  startingBit = 7;
            int  endingBit   = typeInfo.elementBits - 1;
            auto sign        = (convert(resultType, bitfield) << literal((endingBit - startingBit)))
                        >> literal(endingBit);

            return {multiple, shifts, sign};
        }

        void enableDivideBy(ExpressionPtr expr, ContextPtr context)
        {
            expr = FastArithmetic(context)(expr);

            auto resultType = resultVariableType(expr);
            bool isSigned   = DataTypeInfo::Get(resultType).isSigned;

            AssertFatal(resultType == DataType::Int32 || resultType == DataType::Int64
                            || resultType == DataType::UInt32,
                        ShowValue(resultType),
                        ShowValue(expr));

            auto exprTimes = evaluationTimes(expr);

            if(exprTimes[EvaluationTime::Translate])
            {
                Log::warn(
                    "Not adding arguments for division by {} due to translate-time evaluation.",
                    toString(expr));
                return;
            }

            AssertFatal(exprTimes[EvaluationTime::KernelLaunch], ShowValue(exprTimes));

            auto const& [magicExpr, numShiftsExpr, signExpr]
                = getMagicMultipleShiftAndSign(expr, context);

            auto magicTimes = evaluationTimes(magicExpr);
            auto shiftTimes = evaluationTimes(numShiftsExpr);

            EvaluationTimes theTimes = magicTimes & shiftTimes;

            AssertFatal(theTimes[EvaluationTime::KernelExecute],
                        ShowValue(magicTimes),
                        ShowValue(shiftTimes),
                        ShowValue(magicExpr),
                        ShowValue(numShiftsExpr));

            if(isSigned)
            {
                AssertFatal(signExpr != nullptr);
                auto signTimes = evaluationTimes(signExpr);

                AssertFatal(signTimes[EvaluationTime::KernelExecute],
                            ShowValue(signTimes),
                            ShowValue(signExpr));
            }
        }

        ExpressionPtr magicNumberDivision(ExpressionPtr numerator,
                                          ExpressionPtr denominator,
                                          ContextPtr    context)
        {
            auto numeratorType   = resultVariableType(numerator);
            auto denominatorType = resultVariableType(denominator);

            if(!(denominatorType == DataType::Int32 || denominatorType == DataType::Int64
                 || denominatorType == DataType::UInt32))
            {
                // Unhandled case
                return nullptr;
            }

            AssertFatal(
                numeratorType.getElementSize() == denominatorType.getElementSize(),
                "Can't mix 32-bit and 64-bit types in fast division, use a Convert expression.",
                ShowValue(numeratorType),
                ShowValue(denominatorType),
                ShowValue(numerator),
                ShowValue(denominator));

            bool isSigned = DataTypeInfo::Get(denominatorType).isSigned;

            auto k = context->kernel();

            auto const& [magicExpr, numShiftsExpr, signExpr]
                = getMagicMultipleShiftAndSign(denominator, context);

            {
                EvaluationTimes evalTimes
                    = evaluationTimes(magicExpr) & evaluationTimes(numShiftsExpr);

                if(!evalTimes[EvaluationTime::KernelExecute]
                   && !evalTimes[EvaluationTime::Translate])
                {
                    Log::debug("Returning nullptr from magicNumberDivision expr / shift "
                               "({})\nmagic: {}\n shift: {}",
                               toString(evalTimes),
                               toString(magicExpr),
                               toString(numShiftsExpr));
                    return nullptr;
                }
            }

            ExpressionPtr result;

            auto one = literal(1, denominatorType);

            if(!isSigned)
            {
                auto q = multiplyHigh(numerator, magicExpr);
                setComment(q, "Magic q (unsigned)");

                auto t = (arithmeticShiftR(numerator - q, one)) + q;
                setComment(t, "Magic t (unsigned)");
                result = arithmeticShiftR(t, numShiftsExpr);
                setComment(result, "Magic result (unsigned)");
            }
            else
            {
                {
                    EvaluationTimes evalTimes = evaluationTimes(signExpr);

                    if(!evalTimes[EvaluationTime::KernelExecute]
                       && !evalTimes[EvaluationTime::Translate])
                    {
                        Log::debug("Returning nullptr from magicNumberDivision sign ({})",
                                   toString(evalTimes));
                        return nullptr;
                    }
                }

                // Create expression that performs division using the new arguments

                auto numBytes = denominatorType.getElementSize();

                // We are doing signed division; and in particular the
                // magic-multiple might be negative.  Make sure the
                // numerator is signed so that the arithmetic
                // generators for the expressions below (q etc) use
                // signed instructions.
                if(not DataTypeInfo::Get(numeratorType).isSigned)
                {
                    auto signedNumeratorType
                        = getIntegerType(true, DataTypeInfo::Get(numeratorType).elementBytes);
                    numerator = std::make_shared<Expression>(
                        Convert{{.arg{numerator}}, signedNumeratorType});
                }

                auto q = multiplyHigh(numerator, magicExpr) + numerator;
                setComment(q, "Magic q (signed)");
                auto signOfQ = arithmeticShiftR(q, literal(numBytes * 8 - 1, denominatorType));
                setComment(signOfQ, "Magic signOfQ");
                auto magicIsPow2 = conditional(magicExpr == literal(0, denominatorType),
                                               literal(-1, denominatorType),
                                               literal(0, denominatorType));
                setComment(magicIsPow2, "Magic isPow2");

                auto handleSignOfLHS = q + (signOfQ & ((one << numShiftsExpr) + magicIsPow2));
                setComment(handleSignOfLHS, "Magic handleSignOfLHS");

                auto shiftedQ = arithmeticShiftR(handleSignOfLHS, numShiftsExpr);
                setComment(shiftedQ, "Magic shiftedQ");

                result = (shiftedQ ^ signExpr) - signExpr;
                setComment(result, "Magic result (signed)");
            }

            result = launchTimeSubExpressions(simplify(result), context);

            {
                auto evalTimes = evaluationTimes(result);
                AssertFatal(evalTimes[EvaluationTime::KernelExecute], toString(result), evalTimes);
            }

            return result;
        }

        template <typename T>
        ExpressionPtr powerOfTwoDivision(ExpressionPtr lhs, T rhs)
        {
            throw std::runtime_error("Power Of 2 Division not supported for this type");
        }

        // Power Of Two division for unsigned integers
        template <>
        ExpressionPtr powerOfTwoDivision(ExpressionPtr lhs, unsigned int rhs)
        {
            uint shiftAmount = std::countr_zero(rhs);
            auto new_rhs     = literal(shiftAmount);
            return arithmeticShiftR(lhs, new_rhs);
        }

        // Power of Two division for signed integers
        template <>
        ExpressionPtr powerOfTwoDivision(ExpressionPtr lhs, int rhs)
        {
            int          shiftAmount        = std::countr_zero(static_cast<unsigned int>(rhs));
            unsigned int signBits           = sizeof(int) * 8 - 1;
            unsigned int reverseShiftAmount = sizeof(int) * 8 - shiftAmount;

            auto shiftAmountExpr        = literal(shiftAmount);
            auto signBitsExpr           = literal(signBits);
            auto reverseShiftAmountExpr = literal(reverseShiftAmount);

            return (lhs + logicalShiftR(lhs >> signBitsExpr, reverseShiftAmountExpr))
                   >> shiftAmountExpr;
        }

        template <typename T>
        ExpressionPtr powerOfTwoModulo(ExpressionPtr lhs, T rhs)
        {
            throw std::runtime_error("Power Of 2 Modulo not supported for this type");
        }

        // Power of Two Modulo for unsigned integers
        template <>
        ExpressionPtr powerOfTwoModulo(ExpressionPtr lhs, unsigned int rhs)
        {
            unsigned int mask    = rhs - 1u;
            auto         new_rhs = literal(mask);
            return lhs & new_rhs;
        }

        // Power of Two Modulo for signed integers
        template <>
        ExpressionPtr powerOfTwoModulo(ExpressionPtr lhs, int rhs)
        {
            int          shiftAmount        = std::countr_zero(static_cast<unsigned int>(rhs));
            unsigned int signBits           = sizeof(int) * 8 - 1;
            unsigned int reverseShiftAmount = sizeof(int) * 8 - shiftAmount;
            int          mask               = ~(rhs - 1);

            auto maskExpr               = literal(mask);
            auto signBitsExpr           = literal(signBits);
            auto reverseShiftAmountExpr = literal(reverseShiftAmount);

            return lhs
                   - ((lhs + logicalShiftR(lhs >> signBitsExpr, reverseShiftAmountExpr))
                      & maskExpr);
        }

        struct DivisionByConstant
        {
            // Fast Modulo for when the divisor is a constant integer
            template <typename T>
            std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, ExpressionPtr>
                operator()(T rhs)
            {
                if(rhs == 0)
                {
                    throw std::runtime_error("Attempting to divide by 0 in expression");
                }
                else if(rhs == 1)
                {
                    return m_lhs;
                }
                else if(rhs == -1)
                {
                    return std::make_shared<Expression>(Multiply({m_lhs, literal(rhs)}));
                }
                // Power of 2 Division
                else if(std::has_single_bit(cast_to_unsigned(rhs)))
                {
                    return powerOfTwoDivision<T>(m_lhs, cast_to_unsigned(rhs));
                }
                else
                {
                    auto rhsExpr = literal(rhs);
                    return magicNumberDivision(m_lhs, rhsExpr, m_context);
                }
            }

            // If the divisor is not an integer, use the original Division operation
            template <typename T>
            std::enable_if_t<!std::is_integral_v<T> || std::is_same_v<T, bool>, ExpressionPtr>
                operator()(T rhs)
            {
                return m_lhs / literal(rhs);
            }

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }

            DivisionByConstant(ContextPtr ctx)
                : m_context(ctx)
            {
            }

        private:
            ContextPtr    m_context;
            ExpressionPtr m_lhs;
        };

        struct ModuloByConstant
        {
            // Fast Modulo for when the divisor is a constant integer
            template <typename T>
            std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, ExpressionPtr>
                operator()(T rhs)
            {
                if(rhs == 0)
                {
                    Throw<FatalError>("Attempting to perform modulo by 0 in expression");
                }
                else if(rhs == 1 || rhs == -1)
                {
                    return literal<T>(0);
                }
                // Power of 2 Modulo
                else if(std::has_single_bit(cast_to_unsigned(rhs)))
                {
                    return powerOfTwoModulo(m_lhs, rhs);
                }
                else
                {
                    auto rhsExpr = literal(rhs);
                    auto div     = magicNumberDivision(m_lhs, rhsExpr, m_context);
                    return m_lhs - (div * rhsExpr);
                }
            }

            // If the divisor is not an integer, use the original Modulo operation
            template <typename T>
            std::enable_if_t<!std::is_integral_v<T> || std::is_same_v<T, bool>, ExpressionPtr>
                operator()(T rhs)
            {
                return m_lhs % literal(rhs);
            }

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }

            ModuloByConstant(ContextPtr ctx)
                : m_context(ctx)
            {
            }

        private:
            ContextPtr    m_context;
            ExpressionPtr m_lhs;
        };

        struct FastDivisionExpressionVisitor
        {
            FastDivisionExpressionVisitor(ContextPtr cxt)
                : m_context(cxt)
            {
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.rhs)
                {
                    cpy.rhs = call(expr.rhs);
                }
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                if(expr.matA)
                {
                    cpy.matA = call(expr.matA);
                }
                if(expr.matB)
                {
                    cpy.matB = call(expr.matB);
                }
                if(expr.matC)
                {
                    cpy.matC = call(expr.matC);
                }
                if(expr.scaleA)
                {
                    cpy.scaleA = call(expr.scaleA);
                }
                if(expr.scaleB)
                {
                    cpy.scaleB = call(expr.scaleB);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.r1hs)
                {
                    cpy.r1hs = call(expr.r1hs);
                }
                if(expr.r2hs)
                {
                    cpy.r2hs = call(expr.r2hs);
                }
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(Divide const& expr) const
            {
                auto lhs          = call(expr.lhs);
                auto rhs          = call(expr.rhs);
                auto rhsEvalTimes = evaluationTimes(rhs);

                std::string extraComment;

                // Obtain a CommandArgumentValue from rhs. If there is one,
                // attempt to replace the division with faster operations.
                if(rhsEvalTimes[EvaluationTime::Translate])
                {
                    auto rhsVal     = evaluate(rhs);
                    auto divByConst = DivisionByConstant(m_context);
                    auto rv         = divByConst.call(lhs, rhsVal);
                    copyComment(rv, expr);
                    return rv;
                }

                auto rhsType = resultVariableType(rhs);
                if(rhsEvalTimes[EvaluationTime::KernelLaunch])
                {
                    auto div = magicNumberDivision(lhs, rhs, m_context);
                    if(div)
                    {
                        copyComment(div, expr);
                        return div;
                    }

                    extraComment = " (magicNumberDivision returned nullptr)";
                }
                return std::make_shared<Expression>(Divide{lhs, rhs, expr.comment + extraComment});
            }

            ExpressionPtr operator()(Modulo const& expr) const
            {

                auto        lhs          = call(expr.lhs);
                auto        rhs          = call(expr.rhs);
                auto        rhsEvalTimes = evaluationTimes(rhs);
                std::string extraComment;

                // Obtain a CommandArgumentValue from rhs. If there is one,
                // attempt to replace the modulo with faster operations.
                if(rhsEvalTimes[EvaluationTime::Translate])
                {
                    auto rhsVal     = evaluate(rhs);
                    auto modByConst = ModuloByConstant(m_context);
                    auto rv         = modByConst.call(lhs, rhsVal);
                    copyComment(rv, expr);
                    return rv;
                }

                auto rhsType = resultVariableType(rhs);

                if(rhsEvalTimes[EvaluationTime::KernelLaunch])
                {
                    auto div = magicNumberDivision(lhs, rhs, m_context);
                    if(div)
                    {
                        auto rv = lhs - (div * rhs);
                        copyComment(rv, expr);
                        return rv;
                    }

                    extraComment = " (modulo: magicNumberDivision returned nullptr)";
                }

                return std::make_shared<Expression>(Modulo{lhs, rhs, expr.comment + extraComment});
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                auto rv = std::visit(*this, *expr);

                Log::trace("visitor:\n    {}\n    {}", toString(expr), toString(rv));
                return rv;
            }

        private:
            ContextPtr m_context;
        };

        /**
         * Attempts to use fastDivision for all of the divisions within an Expression.
         */
        ExpressionPtr fastDivision(ExpressionPtr expr, ContextPtr cxt)
        {
            auto visitor = FastDivisionExpressionVisitor(cxt);
            auto rv      = visitor.call(expr);

            Log::trace("fastDivision:\n    {}\n    {}", toString(expr), toString(rv));

            return rv;
        }

    }
}
