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
            UnsignedDivisonByConstantInfo magics
                = UnsignedDivisonByConstantInfo::get(llvm::APInt(32, divisor));

            magicNumber = magics.Magic.getLimitedValue();
            numShifts   = magics.ShiftAmount;
            isAdd       = magics.IsAdd;
#else
            auto magics = llvm::APInt(32, divisor).magicu();

            magicNumber = magics.m.getLimitedValue();
            numShifts   = magics.s;
            isAdd       = magics.a;
#endif
        }

        void magicNumbersSigned(int           divisor,
                                long int&     magicNumber,
                                unsigned int& numShifts,
                                bool&         isNegative)
        {
#if LLVM_VERSION_MAJOR >= 14
            UnsignedDivisonByConstantInfo magics
                = SignedDivisonByConstantInfo::get(llvm::APInt(32, divisor));

            magicNumber = magics.Magic.getLimitedValue();
            numShifts   = magics.ShiftAmount;
#else
            auto magics = llvm::APInt(32, divisor).magic();
            magicNumber = (long int)magics.m.getLimitedValue();
            numShifts   = magics.s;
            isNegative  = magics.m.isNegative();
#endif
        }

        ExpressionPtr magicNumberDivision(ExpressionPtr            lhs,
                                          ExpressionPtr            rhs,
                                          std::shared_ptr<Context> context)
        {
            auto k = context->kernel();

            // Create unique names for the new arguments
            auto magicNumStr   = concatenate("magic_num_", k->arguments().size());
            auto magicShiftStr = concatenate("magic_shifts_", k->arguments().size());
            auto magicSignStr  = concatenate("magic_sign_", k->arguments().size());

            // Add the new arguments to the AssemblyKernel
            k->addArgument(
                {magicNumStr, DataType::Int32, DataDirection::ReadOnly, magicMultiple(rhs)});
            k->addArgument(
                {magicShiftStr, DataType::Int32, DataDirection::ReadOnly, magicShifts(rhs)});
            k->addArgument(
                {magicSignStr, DataType::Int32, DataDirection::ReadOnly, magicSign(rhs)});

            // Create expressions of the new arguments
            auto magicExpr = std::make_shared<Expression>(
                std::make_shared<AssemblyKernelArgument>(k->findArgument(magicNumStr)));
            auto numShiftsExpr = std::make_shared<Expression>(
                std::make_shared<AssemblyKernelArgument>(k->findArgument(magicShiftStr)));
            auto signExpr = std::make_shared<Expression>(
                std::make_shared<AssemblyKernelArgument>(k->findArgument(magicSignStr)));

            // Create expression that performs division using the new arguments
            auto q       = multiplyHigh(lhs, magicExpr) + lhs;
            auto signOfQ = q >> literal(31);
            auto handleSignOfLHS
                = q + (signOfQ & ((literal(1) << numShiftsExpr) - (magicExpr == literal(0))));
            auto shiftedQ = handleSignOfLHS >> numShiftsExpr;
            auto result   = (shiftedQ ^ signExpr) - signExpr;
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
            auto magicNumberExpr = literal(magicNumber);
            auto magicMultiple   = multiplyHigh(lhs, magicNumberExpr);

            if(isAdd)
            {
                ExpressionPtr one           = literal(1);
                ExpressionPtr numShiftsExpr = std::make_shared<Expression>(numShifts - 1);
                return shiftR(shiftR(lhs - magicMultiple, one) + magicMultiple, numShiftsExpr);
            }
            else
            {
                ExpressionPtr numShiftsExpr = std::make_shared<Expression>(numShifts);
                return shiftR(magicMultiple, numShiftsExpr);
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
            auto magicNumberExpr = literal(magicNumber);
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

            return (magicMultiple >> numShiftsExpr) + shiftR(magicMultiple, signBitsExpr);
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
            return shiftR(lhs, new_rhs);
        }

        // Power of Two division for signed integers
        template <>
        ExpressionPtr powerOfTwoDivision(ExpressionPtr lhs, int rhs)
        {
            int shiftAmount        = std::countr_zero(static_cast<unsigned int>(rhs));
            int signBits           = sizeof(int) * 8 - 1;
            int reverseShiftAmount = sizeof(int) * 8 - shiftAmount;

            auto shiftAmountExpr        = literal(shiftAmount);
            auto signBitsExpr           = literal(signBits);
            auto reverseShiftAmountExpr = literal(reverseShiftAmount);

            return (lhs + shiftR(lhs >> signBitsExpr, reverseShiftAmountExpr)) >> shiftAmountExpr;
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
            return std::make_shared<Expression>(BitwiseAnd({lhs, new_rhs}));
        }

        // Power of Two Modulo for signed integers
        template <>
        ExpressionPtr powerOfTwoModulo(ExpressionPtr lhs, int rhs)
        {
            int shiftAmount        = std::countr_zero(static_cast<unsigned int>(rhs));
            int signBits           = sizeof(int) * 8 - 1;
            int reverseShiftAmount = sizeof(int) * 8 - shiftAmount;
            int mask               = ~(rhs - 1);

            auto maskExpr               = literal(mask);
            auto signBitsExpr           = literal(signBits);
            auto reverseShiftAmountExpr = literal(reverseShiftAmount);

            return lhs - ((lhs + shiftR(lhs >> signBitsExpr, reverseShiftAmountExpr)) & maskExpr);
        }

        struct DivisionByConstant
        {
            ExpressionPtr m_lhs;

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
                    return powerOfTwoDivision<T>(m_lhs, rhs);
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

            ExpressionPtr operator()(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }
        };

        struct ModuloByConstant
        {
            ExpressionPtr m_lhs;

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
                // Power of 2 Division
                else if(std::has_single_bit(cast_to_unsigned(rhs)))
                {
                    return powerOfTwoModulo(m_lhs, rhs);
                }
                else
                {
                    auto div      = magicNumberDivisionByConstant(m_lhs, rhs);
                    auto rhs_expr = literal(rhs);
                    return m_lhs - (div * rhs_expr);
                }
            }

            // If the divisor is not an integer, use the original Modulo operation
            template <typename T>
            std::enable_if_t<!std::is_integral_v<T> || std::is_same_v<T, bool>, ExpressionPtr>
                operator()(T rhs)
            {
                return m_lhs % literal(rhs);
            }

            ExpressionPtr operator()(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                m_lhs = lhs;
                return visit(*this, rhs);
            }
        };

        struct FastDivisionExpressionVisitor
        {
            FastDivisionExpressionVisitor(std::shared_ptr<Context> cxt)
                : m_context(cxt)
            {
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto result = std::make_shared<Expression>(
                    Expr({(*this)(expr.lhs), (*this)(expr.r1hs), (*this)(expr.r2hs)}));

                return result;
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto result
                    = std::make_shared<Expression>(Expr({(*this)(expr.lhs), (*this)(expr.rhs)}));

                return result;
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto result = std::make_shared<Expression>(Expr({(*this)(expr.arg)}));

                return result;
            }

            ExpressionPtr operator()(Divide const& expr) const
            {
                auto lhs          = (*this)(expr.lhs);
                auto rhs          = (*this)(expr.rhs);
                auto rhsEvalTimes = evaluationTimes(rhs);

                // Obtain a CommandArgumentValue from rhs. If there is one,
                // attempt to replace the division with faster operations.
                if(rhsEvalTimes[EvaluationTime::Translate])
                {
                    auto rhs_val = evaluate(rhs);
                    return DivisionByConstant()(lhs, rhs_val);
                }

                auto rhs_type = resultVariableType(rhs);

                if(rhsEvalTimes[EvaluationTime::KernelLaunch] && rhs_type == DataType::Int32)
                {
                    return magicNumberDivision(lhs, rhs, m_context);
                }

                return std::make_shared<Expression>(Divide({lhs, rhs}));
            }

            ExpressionPtr operator()(Modulo const& expr) const
            {

                auto lhs          = (*this)(expr.lhs);
                auto rhs          = (*this)(expr.rhs);
                auto rhsEvalTimes = evaluationTimes(rhs);

                // Obtain a CommandArgumentValue from rhs. If there is one,
                // attempt to replace the modulo with faster operations.
                if(rhsEvalTimes[EvaluationTime::Translate])
                {
                    auto rhs_val = evaluate(rhs);
                    return ModuloByConstant()(lhs, rhs_val);
                }

                auto rhs_type = resultVariableType(rhs);

                if(rhsEvalTimes[EvaluationTime::KernelLaunch] && rhs_type == DataType::Int32)
                {
                    auto div = magicNumberDivision(lhs, rhs, m_context);
                    return lhs - (div * rhs);
                }

                return std::make_shared<Expression>(Modulo({lhs, rhs}));
            }

            ExpressionPtr operator()(MatrixMultiply const& expr) const
            {
                return std::make_shared<Expression>(MatrixMultiply(expr));
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr operator()(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }

        private:
            std::shared_ptr<Context> m_context;
        };

        /**
         * Attempts to use fastDivision for all of the divisions within an Expression.
         */
        ExpressionPtr fastDivision(ExpressionPtr expr, std::shared_ptr<Context> cxt)
        {
            auto visitor = FastDivisionExpressionVisitor(cxt);
            return visitor(expr);
        }

    }
}
