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

#include <llvm/Config/llvm-config.h>
#include <llvm/Support/DivisionByConstantInfo.h>

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

        void magicNumbersUnsigned(unsigned int  divisor,
                                  u_int64_t&    magicNumber,
                                  unsigned int& numPreShifts,
                                  unsigned int& numPostShifts,
                                  bool&         isAdd)
        {
            llvm::UnsignedDivisionByConstantInfo magicu
                = llvm::UnsignedDivisionByConstantInfo::get(llvm::APInt(32, divisor));

            magicNumber   = magicu.Magic.getZExtValue();
            numPreShifts  = magicu.PreShift;
            numPostShifts = magicu.PostShift;
            isAdd         = magicu.IsAdd;
        }

        void magicNumbersSigned(int           divisor,
                                long int&     magicNumber,
                                unsigned int& numShifts,
                                bool&         isNegative)
        {
            llvm::SignedDivisionByConstantInfo magics
                = llvm::SignedDivisionByConstantInfo::get(llvm::APInt(32, divisor, true));

            magicNumber = magics.Magic.getSExtValue();
            numShifts   = magics.ShiftAmount;
            isNegative  = magicNumber < 0L;
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

            auto magicExpr     = launchTimeSubExpressions(magicMultiple(expr), context);
            auto numShiftsExpr = launchTimeSubExpressions(magicShifts(expr), context);

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
                auto signExpr  = launchTimeSubExpressions(magicSign(expr), context);
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

            auto magicExpr     = launchTimeSubExpressions(magicMultiple(denominator), context);
            auto numShiftsExpr = launchTimeSubExpressions(magicShifts(denominator), context);

            {
                EvaluationTimes evalTimes
                    = evaluationTimes(magicExpr) & evaluationTimes(numShiftsExpr);

                if(!evalTimes[EvaluationTime::KernelExecute])
                {
                    Log::debug("Returning nullptr from magicNumberDivision ({})",
                               toString(evalTimes));
                    return nullptr;
                }
            }

            ExpressionPtr result;

            auto one = literal(1, denominatorType);

            if(!isSigned)
            {
                auto q = multiplyHigh(numerator, magicExpr);
                auto t = (arithmeticShiftR(numerator - q, one)) + q;
                result = arithmeticShiftR(t, numShiftsExpr);
            }
            else
            {
                auto signExpr = launchTimeSubExpressions(magicSign(denominator), context);

                {
                    EvaluationTimes evalTimes = evaluationTimes(signExpr);

                    if(!evalTimes[EvaluationTime::KernelExecute])
                    {
                        Log::debug("Returning nullptr from magicNumberDivision ({})",
                                   toString(evalTimes));
                        return nullptr;
                    }
                }

                // Create expression that performs division using the new arguments

                auto numBytes = denominatorType.getElementSize();

                auto q           = multiplyHigh(numerator, magicExpr) + numerator;
                auto signOfQ     = arithmeticShiftR(q, literal(numBytes * 8 - 1, denominatorType));
                auto magicIsPow2 = conditional(magicExpr == literal(0, denominatorType),
                                               literal(-1, denominatorType),
                                               literal(0, denominatorType));

                auto handleSignOfLHS = q + (signOfQ & ((one << numShiftsExpr) + magicIsPow2));

                auto shiftedQ = arithmeticShiftR(handleSignOfLHS, numShiftsExpr);

                result = (shiftedQ ^ signExpr) - signExpr;
            }

            result = launchTimeSubExpressions(result, context);

            {
                auto evalTimes = evaluationTimes(result);
                AssertFatal(evalTimes[EvaluationTime::KernelExecute], toString(result), evalTimes);
            }

            return result;
        }

        template <typename T>
        ExpressionPtr magicNumberDivisionByConstant(ExpressionPtr lhs, T rhs)
        {
            throw std::runtime_error("Magic Number Dvision not supported for this type");
        }

        // Magic number division for unsigned integers
        template <>
        ExpressionPtr magicNumberDivisionByConstant(ExpressionPtr lhs, unsigned int rhs)
        {
            u_int64_t    magicNumber;
            unsigned int numPreShifts, numPostShifts;
            bool         isAdd;

            magicNumbersUnsigned(rhs, magicNumber, numPreShifts, numPostShifts, isAdd);

            auto magicNumberExpr = literal(static_cast<unsigned int>(magicNumber));
            auto lhsPreShifted   = lhs;
            if(numPreShifts > 0)
            {
                ExpressionPtr numPreShiftsExpr = literal(numPreShifts);
                lhsPreShifted                  = logicalShiftR(lhs, numPreShiftsExpr);
            }

            auto          magicMultiple     = multiplyHigh(lhsPreShifted, magicNumberExpr);
            ExpressionPtr numPostShiftsExpr = literal(numPostShifts);

            if(isAdd)
            {
                ExpressionPtr one = literal(1u);
                return logicalShiftR(logicalShiftR(lhs - magicMultiple, one) + magicMultiple,
                                     numPostShiftsExpr);
            }
            else
            {
                return logicalShiftR(magicMultiple, numPostShiftsExpr);
            }
        }

        // Magic number division for signed integers
        template <>
        ExpressionPtr magicNumberDivisionByConstant(ExpressionPtr lhs, int rhs)
        {
            int64_t      magicNumber;
            unsigned int numShifts;
            bool         isNegative;

            magicNumbersSigned(rhs, magicNumber, numShifts, isNegative);

            auto magicNumberExpr = literal(static_cast<int>(magicNumber));
            auto magicMultiple   = multiplyHigh(lhs, magicNumberExpr);

            if(rhs > 0 && isNegative)
            {
                magicMultiple = magicMultiple + lhs;
            }
            else if(rhs < 0 && magicNumber > 0L)
            {
                magicMultiple = magicMultiple - lhs;
            }

            ExpressionPtr numShiftsExpr = literal(numShifts);
            ExpressionPtr signBitsExpr  = literal<uint32_t>(sizeof(int) * 8 - 1);
            ExpressionPtr shifted       = (magicMultiple >> numShiftsExpr);

            return shifted + logicalShiftR(shifted, signBitsExpr);
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
                    return magicNumberDivisionByConstant<T>(m_lhs, rhs);
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

        private:
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
                    auto div     = magicNumberDivisionByConstant(m_lhs, rhs);
                    auto rhsExpr = literal(rhs);
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

        private:
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
                    auto divByConst = DivisionByConstant();
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
                    auto modByConst = ModuloByConstant();
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
