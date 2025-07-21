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

#include <variant>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Utilities/Timer.hpp>

/**
 * NOTE TO EDITORS OF THIS FILE:
 *
 * DO NOT call toString() or ShowValue() on an expression from this file.
 * Expression::toString() now includes the expression result type, so calling
 * toString from resultType will result in unbounded recursion.
 */

namespace rocRoller
{
    namespace Expression
    {
        /*
         * result type
         */

        class ExpressionResultTypeVisitor
        {
            std::weak_ptr<Context> m_context;

        public:
            template <typename T>
            requires(CBinary<T>&& CArithmetic<T>) ResultType operator()(T const& expr)
            {
                auto lhsVal = call(expr.lhs);
                auto rhsVal = call(expr.rhs);

                auto regType = Register::PromoteType(lhsVal.regType, rhsVal.regType);

                VariableType varType;

                // A shift's type is the same as the shifted value.
                if constexpr(CShift<T>)
                {
                    varType = lhsVal.varType;
                }
                else if(std::same_as<T, Subtract> && lhsVal.varType.isPointer()
                        && lhsVal.varType == rhsVal.varType)
                {
                    varType = DataType::Int64;
                }
                else
                {
                    varType = VariableType::Promote(lhsVal.varType, rhsVal.varType);
                }

                return {regType, varType};
            }

            template <typename T>
            requires(CTernary<T>&& CArithmetic<T>) ResultType operator()(T const& expr)
            {
                auto lhsVal  = call(expr.lhs);
                auto r1hsVal = call(expr.r1hs);
                auto r2hsVal = call(expr.r2hs);

                auto regType = Register::PromoteType(lhsVal.regType, r1hsVal.regType);
                regType      = Register::PromoteType(regType, r2hsVal.regType);

                auto varType = VariableType::Promote(lhsVal.varType, r1hsVal.varType);
                varType      = VariableType::Promote(varType, r2hsVal.varType);

                return {regType, varType};
            }

            ResultType operator()(AddShiftL const& expr)
            {
                auto lhsVal  = call(expr.lhs);
                auto r1hsVal = call(expr.r1hs);

                auto regType = Register::PromoteType(lhsVal.regType, r1hsVal.regType);
                auto varType = VariableType::Promote(lhsVal.varType, r1hsVal.varType);

                return {regType, varType};
            }

            ResultType operator()(ShiftLAdd const& expr)
            {
                auto lhsVal  = call(expr.lhs);
                auto r2hsVal = call(expr.r2hs);

                auto regType = Register::PromoteType(lhsVal.regType, r2hsVal.regType);
                auto varType = VariableType::Promote(lhsVal.varType, r2hsVal.varType);

                return {regType, varType};
            }

            ResultType operator()(ScaledMatrixMultiply const& expr)
            {
                auto matAVal = call(expr.matA);
                auto matBVal = call(expr.matB);
                auto matCVal = call(expr.matC);

                auto regType = Register::PromoteType(matAVal.regType, matBVal.regType);
                regType      = Register::PromoteType(regType, matCVal.regType);

                auto varType = VariableType::Promote(matAVal.varType, matBVal.varType);
                varType      = VariableType::Promote(varType, matCVal.varType);

                return {regType, varType};
            }

            template <typename T>
            requires(CUnary<T>&& CArithmetic<T>) ResultType operator()(T const& expr)
            {
                auto argVal = call(expr.arg);

                if constexpr(std::same_as<T, MagicShifts>)
                    return {argVal.regType, DataType::Int32};
                else if constexpr(std::same_as<T, MagicShiftAndSign>)
                    return {argVal.regType, DataType::UInt32};

                if constexpr(std::same_as<T, ToScalar>)
                    return {Register::Type::Scalar, argVal.varType};

                return argVal;
            }

            ResultType operator()(Convert const& expr)
            {
                auto argVal = call(expr.arg);
                return {argVal.regType, expr.destinationType};
            }

            template <DataType DATATYPE>
            ResultType operator()(SRConvert<DATATYPE> const& expr)
            {
                // SR conversion currently only supports FP8 and BF8
                static_assert(DATATYPE == DataType::FP8 || DATATYPE == DataType::BF8);
                auto argVal = call(expr.lhs);
                return {argVal.regType, DATATYPE};
            }

            ResultType operator()(BitFieldExtract const& expr)
            {
                auto argVal = call(expr.arg);
                return {argVal.regType, expr.outputDataType};
            }

            template <typename T>
            requires(CBinary<T>&& CComparison<T>) ResultType operator()(T const& expr)
            {
                auto lhsVal = call(expr.lhs);
                auto rhsVal = call(expr.rhs);

                // Can't compare between two different types on the GPU.
                AssertFatal(lhsVal.regType == Register::Type::Literal
                                || rhsVal.regType == Register::Type::Literal
                                || lhsVal.varType == rhsVal.varType,
                            ShowValue(lhsVal.varType),
                            ShowValue(rhsVal.varType),
                            ShowValue(lhsVal),
                            ShowValue(rhsVal));

                auto inputRegType = Register::PromoteType(lhsVal.regType, rhsVal.regType);

                // Two pointers of the same type can be compared. Otherwise the types have to
                // be compatible according to the promotion rules.
                if(lhsVal.varType != rhsVal.varType || !lhsVal.varType.isPointer())
                {
                    auto varType = VariableType::Promote(lhsVal.varType, rhsVal.varType);
                    if(varType == DataType::None)
                        return {inputRegType, DataType::None};
                }

                switch(inputRegType)
                {
                case Register::Type::Literal:
                    return {Register::Type::Literal, DataType::Bool};
                case Register::Type::Scalar:
                    return {Register::Type::Scalar, DataType::Bool};
                case Register::Type::Vector:
                    if(auto context = m_context.lock(); context)
                    {
                        if(context->kernel()->wavefront_size() == 32)
                            return {Register::Type::Scalar, DataType::Bool32};
                        return {Register::Type::Scalar, DataType::Bool64};
                    }
                    // If you are reading this, it probably means that this visitor
                    // was called on an expression with registers that didn't have
                    // a context.
                    // Throw<FatalError>("Need context to determine wavefront size", ShowValue(name(expr)));
                    return {Register::Type::Scalar, DataType::None};
                default:
                    break;
                }
                Throw<FatalError>("Invalid register types for comparison: ",
                                  ShowValue(lhsVal.regType),
                                  ShowValue(rhsVal.regType));
            }

            template <typename T>
            requires(CBinary<T>&& CLogical<T>) ResultType operator()(T const& expr)
            {
                auto lhsVal = call(expr.lhs);
                auto rhsVal = call(expr.rhs);
                return logical(lhsVal, rhsVal);
            }

            ResultType logical(ResultType lhsVal, ResultType rhsVal)
            {
                if(lhsVal.varType == DataType::Bool
                   && (rhsVal.varType == DataType::Bool32 || rhsVal.varType == DataType::Bool64))
                {
                    std::swap(lhsVal, rhsVal);
                }

                // Can't compare between two different types on the GPU.
                AssertFatal(
                    lhsVal.regType == Register::Type::Literal
                        || rhsVal.regType == Register::Type::Literal
                        || lhsVal.varType == rhsVal.varType
                        || (lhsVal.varType == DataType::Bool32 && rhsVal.varType == DataType::Bool)
                        || (lhsVal.varType == DataType::Bool64 && rhsVal.varType == DataType::Bool),
                    ShowValue(lhsVal.varType),
                    ShowValue(rhsVal.varType));

                auto inputRegType = Register::PromoteType(lhsVal.regType, rhsVal.regType);
                auto inputVarType = VariableType::Promote(lhsVal.varType, rhsVal.varType);

                switch(inputRegType)
                {
                case Register::Type::Scalar:
                    if(inputVarType == DataType::Bool //
                       || inputVarType == DataType::Bool32 //
                       || inputVarType == DataType::Bool64)
                    {
                        return {Register::Type::Scalar, inputVarType};
                    }
                case Register::Type::Literal:
                {
                    if(inputVarType == DataType::Bool)
                        return {inputRegType, inputVarType};
                }
                default:
                    break;
                }
                Throw<FatalError>("Invalid register types for logical: ",
                                  ShowValue(lhsVal),
                                  ShowValue(rhsVal),
                                  ShowValue(inputRegType),
                                  ShowValue(inputVarType));
            }

            template <typename T>
            requires(CUnary<T>&& CLogical<T>) ResultType operator()(T const& expr)
            {
                auto val = call(expr.arg);
                switch(val.regType)
                {
                case Register::Type::Scalar:
                {
                    if(val.varType == DataType::Bool || val.varType == DataType::Bool32
                       || val.varType == DataType::Bool64 || val.varType == DataType::Raw32)
                        return val;
                }
                case Register::Type::Literal:
                {
                    if(val.varType == DataType::Bool)
                        return val;
                }
                default:
                    Throw<FatalError>("Invalid register/variable type for unary logical: ",
                                      ShowValue(val));
                }
            }

            ResultType operator()(Conditional const& expr)
            {
                auto lhsVal  = call(expr.lhs);
                auto r1hsVal = call(expr.r1hs);
                auto r2hsVal = call(expr.r2hs);

                AssertFatal(r2hsVal.varType == r1hsVal.varType,
                            ShowValue(r1hsVal.varType),
                            ShowValue(r2hsVal.varType));
                auto varType = r2hsVal.varType;

                if(lhsVal.varType == DataType::Bool32 || lhsVal.varType == DataType::Bool64
                   || lhsVal.regType == Register::Type::Vector
                   || r1hsVal.regType == Register::Type::Vector
                   || r2hsVal.regType == Register::Type::Vector)
                {
                    return {Register::Type::Vector, varType};
                }
                return {Register::Type::Scalar, varType};
            }

            ResultType operator()(CommandArgumentPtr const& expr)
            {
                if(expr == nullptr)
                    return {Register::Type::Count, DataType::Count};

                return {Register::Type::Literal, expr->variableType()};
            }

            ResultType operator()(AssemblyKernelArgumentPtr const& expr)
            {
                if(expr == nullptr)
                    return {Register::Type::Count, DataType::Count};

                return {Register::Type::Scalar, expr->variableType};
            }

            ResultType operator()(CommandArgumentValue const& expr)
            {
                return {Register::Type::Literal, variableType(expr)};
            }

            ResultType operator()(Register::ValuePtr const& expr)
            {
                if(expr == nullptr)
                    return {Register::Type::Count, DataType::Count};

                m_context = expr->context();
                return {expr->regType(), expr->variableType()};
            }

            ResultType operator()(DataFlowTag const& expr)
            {
                return {expr.regType, expr.varType};
            }

            ResultType operator()(PositionalArgument const& expr)
            {
                Throw<FatalError>("Can not get result type of PositionalArgument.");
            }

            ResultType operator()(WaveTilePtr const& expr)
            {
                return call(expr->vgpr);
            }

            ResultType call(Expression const& expr)
            {
                return std::visit(*this, expr);
            }

            ResultType call(ExpressionPtr const& expr)
            {
                if(expr == nullptr)
                    return {Register::Type::Count, DataType::Count};

                return call(*expr);
            }
        };

        VariableType resultVariableType(Expression const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr).varType;
        }

        VariableType resultVariableType(ExpressionPtr const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr).varType;
        }

        Register::Type resultRegisterType(Expression const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr).regType;
        }

        Register::Type resultRegisterType(ExpressionPtr const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr).regType;
        }

        ResultType resultType(ExpressionPtr const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr);
        }

        ResultType resultType(Expression const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr);
        }

    }
}
