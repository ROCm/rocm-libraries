// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
        // This function determines the value count of an expression
        // based on the value counts of its operands.
        // Note that the value count of each operand in an expression must either be 1
        // or equal to all other non-1 value counts in the expression.
        // For example:
        //     lhs.valueCount = 1, rhs.valueCount = 1: Valid, return 1
        //     lhs.valueCount = 1, rhs.valueCount = 16: Valid, return 16
        //     lhs.valueCount = 8, rhs.valueCount = 1: Valid, return 8
        //     lhs.valueCount = 4, rhs.valueCount = 4: Valid, return 4
        //     lhs.valueCount = 4, rhs.valueCount = 2: INVALID
        inline size_t broadcastValueCount(std::vector<ResultType> operands)
        {
            size_t valueCount = 1;
            for(auto operand : operands)
            {
                if(operand.valueCount != 1)
                {
                    if(valueCount == 1)
                        return valueCount = operand.valueCount;
                    else
                        AssertFatal(valueCount == operand.valueCount,
                                    "Each operand's value count in an expression must either "
                                    "be 1 or equal to all other non-1 value counts\n",
                                    ShowValue(valueCount),
                                    ShowValue(operand.valueCount));
                }
            }

            return valueCount;
        }

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
                // A bitfieldCombine's type is the same as the destination type.
                else if constexpr(std::same_as<T, BitfieldCombine>)
                {
                    varType = rhsVal.varType;
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

                auto valueCount = broadcastValueCount({lhsVal, rhsVal});

                return {regType, varType, valueCount};
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

                size_t valueCount = broadcastValueCount({lhsVal, r1hsVal, r2hsVal});

                return {regType, varType, valueCount};
            }

            ResultType operator()(AddShiftL const& expr)
            {
                auto lhsVal  = call(expr.lhs);
                auto r1hsVal = call(expr.r1hs);
                auto r2hsVal = call(expr.r2hs);

                auto regType    = Register::PromoteType(lhsVal.regType, r1hsVal.regType);
                auto varType    = VariableType::Promote(lhsVal.varType, r1hsVal.varType);
                auto valueCount = broadcastValueCount({lhsVal, r1hsVal, r2hsVal});

                return {regType, varType, valueCount};
            }

            ResultType operator()(ShiftLAdd const& expr)
            {
                auto lhsVal  = call(expr.lhs);
                auto r1hsVal = call(expr.r1hs);
                auto r2hsVal = call(expr.r2hs);

                auto regType    = Register::PromoteType(lhsVal.regType, r2hsVal.regType);
                auto varType    = VariableType::Promote(lhsVal.varType, r2hsVal.varType);
                auto valueCount = broadcastValueCount({lhsVal, r1hsVal, r2hsVal});

                return {regType, varType, valueCount};
            }

            ResultType operator()(MatrixMultiply const& expr)
            {
                auto matAVal = call(expr.lhs);
                auto matBVal = call(expr.r1hs);
                auto matCVal = call(expr.r2hs);

                auto regType = Register::PromoteType(matAVal.regType, matBVal.regType);
                regType      = Register::PromoteType(regType, matCVal.regType);

                auto varType = VariableType::Promote(matAVal.varType, matBVal.varType);
                varType      = VariableType::Promote(varType, matCVal.varType);

                return {regType, varType, matCVal.valueCount};
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

                return {regType, varType, matCVal.valueCount};
            }

            template <typename T>
            requires(CUnary<T>&& CArithmetic<T>) ResultType operator()(T const& expr)
            {
                auto argVal = call(expr.arg);

                if constexpr(std::same_as<T, MagicShifts>)
                    return {argVal.regType, DataType::Int32, argVal.valueCount};
                else if constexpr(std::same_as<T, MagicShiftAndSign>)
                    return {argVal.regType, DataType::UInt32, argVal.valueCount};

                if constexpr(std::same_as<T, ToScalar>)
                    return {Register::Type::Scalar, argVal.varType, argVal.valueCount};

                return argVal;
            }

            ResultType operator()(Convert const& expr)
            {
                auto argVal = call(expr.arg);
                return {argVal.regType, expr.destinationType, argVal.valueCount};
            }

            ResultType operator()(Reinterpret const& expr)
            {
                auto argVal = call(expr.arg);
                return {argVal.regType, expr.destinationType, argVal.valueCount};
            }

            template <DataType DATATYPE>
            ResultType operator()(SRConvert<DATATYPE> const& expr)
            {
                // SR conversion currently only supports FP8 and BF8
                static_assert(DATATYPE == DataType::FP8 || DATATYPE == DataType::BF8);
                auto argVal = call(expr.lhs);
                return {argVal.regType, DATATYPE, argVal.valueCount};
            }

            ResultType operator()(BitFieldExtract const& expr)
            {
                auto argVal = call(expr.arg);
                return {argVal.regType, expr.outputDataType, argVal.valueCount};
            }

            template <typename T>
            requires(CBinary<T>&& CComparison<T>) ResultType operator()(T const& expr)
            {
                auto lhsVal = call(expr.lhs);
                auto rhsVal = call(expr.rhs);

                size_t valueCount = broadcastValueCount({lhsVal, rhsVal});

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
                        return {inputRegType, DataType::None, valueCount};
                }

                switch(inputRegType)
                {
                case Register::Type::Literal:
                    return {Register::Type::Literal, DataType::Bool, valueCount};
                case Register::Type::Scalar:
                    return {Register::Type::Scalar, DataType::Bool, valueCount};
                case Register::Type::Vector:
                    if(auto context = m_context.lock(); context)
                    {
                        if(context->kernel()->wavefront_size() == 32)
                            return {Register::Type::Scalar, DataType::Bool32, valueCount};
                        return {Register::Type::Scalar, DataType::Bool64, valueCount};
                    }
                    // If you are reading this, it probably means that this visitor
                    // was called on an expression with registers that didn't have
                    // a context.
                    // Throw<FatalError>("Need context to determine wavefront size", ShowValue(name(expr)));
                    return {Register::Type::Scalar, DataType::None, valueCount};
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
                size_t valueCount = broadcastValueCount({lhsVal, rhsVal});

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
                        return {Register::Type::Scalar, inputVarType, valueCount};
                    }
                case Register::Type::Literal:
                {
                    if(inputVarType == DataType::Bool)
                        return {inputRegType, inputVarType, valueCount};
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

                size_t valueCount = broadcastValueCount({lhsVal, r1hsVal, r2hsVal});

                if(lhsVal.varType == DataType::Bool32 || lhsVal.varType == DataType::Bool64
                   || lhsVal.regType == Register::Type::Vector
                   || r1hsVal.regType == Register::Type::Vector
                   || r2hsVal.regType == Register::Type::Vector)
                {
                    return {Register::Type::Vector, varType, valueCount};
                }
                return {Register::Type::Scalar, varType, valueCount};
            }

            ResultType operator()(Concatenate const& expr)
            {
                auto         registerType = Register::Type::Literal;
                VariableType variableType = expr.destinationType;

                auto expectedNumRegister   = DataTypeInfo::Get(expr.destinationType).registerCount;
                unsigned actualNumRegister = 0;

                size_t valueCount = 1;

                for(auto const& operand : expr.operands)
                {
                    auto&& [operandRegisterType, operandVariableType, operandValueCount]
                        = call(operand);
                    switch(operandRegisterType)
                    {
                    case Register::Type::Literal:
                    case Register::Type::Scalar:
                    case Register::Type::Vector:
                        break;
                    default:
                        Throw<FatalError>(
                            "Invalid register type for concatenate expression operands",
                            ShowValue(operand));
                    }

                    registerType = Register::PromoteType(registerType, operandRegisterType);
                    actualNumRegister
                        = actualNumRegister + DataTypeInfo::Get(operandVariableType).registerCount;

                    AssertFatal(operandValueCount == 1,
                                "All operands to Concatenate must have value count 1",
                                ShowValue(operandValueCount));
                }

                AssertFatal(expectedNumRegister == actualNumRegister,
                            ShowValue(expr.destinationType),
                            ShowValue(expectedNumRegister),
                            ShowValue(actualNumRegister));

                return {registerType, variableType, valueCount};
            }

            ResultType operator()(CommandArgumentPtr const& expr)
            {
                if(expr == nullptr)
                    return {Register::Type::Count, DataType::Count, 0};

                auto packing = expr->variableType().dataType == DataType::None
                                   ? 1
                                   : DataTypeInfo::Get(expr->variableType()).packing;
                return {Register::Type::Literal, expr->variableType(), packing};
            }

            ResultType operator()(AssemblyKernelArgumentPtr const& expr)
            {
                if(expr == nullptr)
                    return {Register::Type::Count, DataType::Count, 0};

                auto packing = expr->getVariableType().dataType == DataType::None
                                   ? 1
                                   : DataTypeInfo::Get(expr->getVariableType()).packing;
                return {Register::Type::Scalar, expr->getVariableType(), packing};
            }

            ResultType operator()(CommandArgumentValue const& expr)
            {
                auto packing = variableType(expr).dataType == DataType::None
                                   ? 1
                                   : DataTypeInfo::Get(variableType(expr)).packing;
                return {Register::Type::Literal, variableType(expr), packing};
            }

            ResultType operator()(Register::ValuePtr const& expr)
            {
                if(expr == nullptr)
                    return {Register::Type::Count, DataType::Count, 0};

                m_context    = expr->context();
                auto packing = expr->variableType().dataType == DataType::None
                                   ? 1
                                   : DataTypeInfo::Get(expr->variableType()).packing;
                return {expr->regType(), expr->variableType(), expr->valueCount() * packing};
            }

            ResultType operator()(DataFlowTag const& expr)
            {
                auto packing = expr.varType.dataType == DataType::None
                                   ? 1
                                   : DataTypeInfo::Get(expr.varType).packing;
                return {expr.regType, expr.varType, packing};
            }

            ResultType operator()(PositionalArgument const& expr)
            {
                auto packing = expr.varType.dataType == DataType::None
                                   ? 1
                                   : DataTypeInfo::Get(expr.varType).packing;
                return {expr.regType, expr.varType, packing};
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
                    return {Register::Type::Count, DataType::Count, 0};

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

        size_t resultValueCount(Expression const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr).valueCount;
        }

        size_t resultValueCount(ExpressionPtr const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr).valueCount;
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
