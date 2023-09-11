#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Utilities/Error.hpp>

#include <bit>

#include "llvm/Config/llvm-config.h"
#if LLVM_VERSION_MAJOR >= 14
#include "llvm/Support/DivisionByConstantInfo.h"
#else
#include "llvm/ADT/APInt.h"
#endif

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
                                  unsigned int& numShifts,
                                  bool&         isAdd)
        {
#if LLVM_VERSION_MAJOR >= 14
            UnsignedDivisonByConstantInfo magicu
                = UnsignedDivisonByConstantInfo::get(llvm::APInt(32, divisor));

            magicNumber = magicu.Magic.getLimitedValue();
            numShifts   = magicu.ShiftAmount;
            isAdd       = magicu.IsAdd;
#else
            auto magicu = llvm::APInt(32, divisor).magicu();

            magicNumber = magicu.m.getLimitedValue();
            numShifts   = magicu.s;
            isAdd       = magicu.a;
#endif
        }

        void magicNumbersSigned(int           divisor,
                                long int&     magicNumber,
                                unsigned int& numShifts,
                                bool&         isNegative)
        {
#if LLVM_VERSION_MAJOR >= 14
            SignedDivisonByConstantInfo magics
                = SignedDivisonByConstantInfo::get(llvm::APInt(32, divisor, true));

            magicNumber = magics.Magic.getLimitedValue();
            numShifts   = magics.ShiftAmount;
#else
            auto magics = llvm::APInt(32, divisor, true).magic();
            magicNumber = (long int)magics.m.getLimitedValue();
            numShifts   = magics.s;
            isNegative  = magics.m.isNegative();
#endif
        }

        ExpressionPtr magicNumberDivision(ExpressionPtr lhs, ExpressionPtr rhs, ContextPtr context)
        {
            auto rhsType = resultVariableType(rhs);
            auto lhsType = resultVariableType(lhs);

            if(!(rhsType == DataType::Int32 || rhsType == DataType::Int64))
            {
                // Unhandled case
                return nullptr;
            }

            auto numerator   = lhs;
            auto denominator = rhs;
            auto dataType    = DataType::Int32;

            if(DataTypeInfo::Get(rhsType).elementSize > 4
               || DataTypeInfo::Get(lhsType).elementSize > 4)
            {
                numerator   = convert(DataType::Int64, lhs);
                denominator = convert(DataType::Int64, rhs);
                dataType    = DataType::Int64;
            }

            auto k = context->kernel();

            // Create unique names for the new arguments
            auto magicNumStr   = concatenate("magic_num_", k->arguments().size());
            auto magicShiftStr = concatenate("magic_shifts_", k->arguments().size());
            auto magicSignStr  = concatenate("magic_sign_", k->arguments().size());

            // Add the new arguments to the AssemblyKernel
            k->addArgument(
                {magicNumStr, dataType, DataDirection::ReadOnly, magicMultiple(denominator)});
            k->addArgument({magicShiftStr,
                            DataType::Int32,
                            DataDirection::ReadOnly,
                            magicShifts(denominator)});
            k->addArgument(
                {magicSignStr, dataType, DataDirection::ReadOnly, magicSign(denominator)});

            // Create expressions of the new arguments
            auto magicExpr = std::make_shared<Expression>(
                std::make_shared<AssemblyKernelArgument>(k->findArgument(magicNumStr)));
            auto numShiftsExpr = std::make_shared<Expression>(
                std::make_shared<AssemblyKernelArgument>(k->findArgument(magicShiftStr)));
            auto signExpr = std::make_shared<Expression>(
                std::make_shared<AssemblyKernelArgument>(k->findArgument(magicSignStr)));

            // Create expression that performs division using the new arguments

            auto numBytes = DataTypeInfo::Get(dataType).elementSize;

            auto q       = multiplyHigh(numerator, magicExpr) + numerator;
            auto signOfQ = arithmeticShiftR(q, literal(numBytes * 8 - 1));

            // auto magicIsPow2 = -(magicExpr == literal(0, dataType));

            auto magicIsPow2 = logicalShiftR((-magicExpr) | magicExpr, literal(numBytes * 8 - 1))
                               - literal(1, dataType); // 0 if != 0, -1 if equal to 0

            auto handleSignOfLHS
                = q + (signOfQ & ((literal(1, dataType) << numShiftsExpr) + magicIsPow2));

            auto shiftedQ = arithmeticShiftR(handleSignOfLHS, numShiftsExpr);

            auto result = (shiftedQ ^ signExpr) - signExpr;

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
            unsigned int numShifts;
            bool         isAdd;

            magicNumbersUnsigned(rhs, magicNumber, numShifts, isAdd);

            auto magicNumberExpr = literal(static_cast<unsigned int>(magicNumber));
            auto magicMultiple   = multiplyHigh(lhs, magicNumberExpr);

            if(isAdd)
            {
                ExpressionPtr one           = literal(1u);
                ExpressionPtr numShiftsExpr = literal(numShifts - 1u);
                return logicalShiftR(logicalShiftR(lhs - magicMultiple, one) + magicMultiple,
                                     numShiftsExpr);
            }
            else
            {

                ExpressionPtr numShiftsExpr = literal(numShifts);
                return logicalShiftR(magicMultiple, numShiftsExpr);
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
            else if(rhs < 0 && !isNegative)
            {
                magicMultiple = magicMultiple - lhs;
            }

            ExpressionPtr numShiftsExpr = literal(numShifts);
            ExpressionPtr signBitsExpr  = literal<int32_t>(sizeof(int) * 8 - 1);

            return (magicMultiple >> numShiftsExpr) + logicalShiftR(magicMultiple, signBitsExpr);
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
                    return std::make_shared<Expression>(Multiply({m_lhs, literal(-1)}));
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
                    return literal(0);
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

                // Obtain a CommandArgumentValue from rhs. If there is one,
                // attempt to replace the division with faster operations.
                if(rhsEvalTimes[EvaluationTime::Translate])
                {
                    auto rhsVal     = evaluate(rhs);
                    auto divByConst = DivisionByConstant();
                    return divByConst.call(lhs, rhsVal);
                }

                auto rhsType = resultVariableType(rhs);
                if(rhsEvalTimes[EvaluationTime::KernelLaunch]
                   && (rhsType == DataType::Int32 || rhsType == DataType::Int64))
                {
                    auto div = magicNumberDivision(lhs, rhs, m_context);
                    if(div)
                        return div;
                }
                return std::make_shared<Expression>(Divide({lhs, rhs}));
            }

            ExpressionPtr operator()(Modulo const& expr) const
            {

                auto lhs          = call(expr.lhs);
                auto rhs          = call(expr.rhs);
                auto rhsEvalTimes = evaluationTimes(rhs);

                // Obtain a CommandArgumentValue from rhs. If there is one,
                // attempt to replace the modulo with faster operations.
                if(rhsEvalTimes[EvaluationTime::Translate])
                {
                    auto rhsVal     = evaluate(rhs);
                    auto modByConst = ModuloByConstant();
                    return modByConst.call(lhs, rhsVal);
                }

                auto rhsType = resultVariableType(rhs);

                if(rhsEvalTimes[EvaluationTime::KernelLaunch]
                   && (rhsType == DataType::Int32 || rhsType == DataType::Int64))
                {
                    auto div = magicNumberDivision(lhs, rhs, m_context);
                    if(div)
                        return lhs - (div * rhs);
                }

                return std::make_shared<Expression>(Modulo({lhs, rhs}));
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

                return std::visit(*this, *expr);
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
            return visitor.call(expr);
        }

    }
}
