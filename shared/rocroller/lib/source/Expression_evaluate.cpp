// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Expression_evaluate_detail.hpp>

#include <rocRoller/AssemblyKernelArgument.hpp>

namespace rocRoller
{
    namespace Expression
    {
        /**
         * Visitor for an Expression.  Walks the expression tree,
         * calling the OperationEvaluatorVisitor to perform actual
         * operations.
         */
        struct EvaluateVisitor
        {
            RuntimeArguments args;

            template <CTernary TernaryExp>
            CommandArgumentValue operator()(TernaryExp const& expr)
            {
                auto arg1 = call(expr.lhs);
                auto arg2 = call(expr.r1hs);
                auto arg3 = call(expr.r2hs);
                return EvaluateDetail::evaluateOp(expr, arg1, arg2, arg3);
            }

            template <CBinary BinaryExp>
            CommandArgumentValue operator()(BinaryExp const& expr)
            {
                // TODO: Short-circuit logic
                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);
                return EvaluateDetail::evaluateOp(expr, lhs, rhs);
            }

            template <CUnary UnaryExp>
            CommandArgumentValue operator()(UnaryExp const& expr)
            {
                auto arg = call(expr.arg);
                return EvaluateDetail::evaluateOp(expr, arg);
            }

            CommandArgumentValue operator()(Concatenate const& expr)
            {
                throw std::runtime_error("N-ary operation present in runtime expression");
            }

            CommandArgumentValue operator()(BitfieldCombine const& expr)
            {
                BitfieldCombine cpy = expr;
                cpy.lhs             = std::make_shared<Expression>(call(expr.lhs));
                cpy.rhs             = std::make_shared<Expression>(call(expr.rhs));

                return evaluate(lowerBitfieldCombine(std::make_shared<Expression>(cpy)));
            }

            CommandArgumentValue operator()(MatrixMultiply const& expr)
            {
                throw std::runtime_error("Matrix multiply present in runtime expression.");
            }

            CommandArgumentValue operator()(ScaledMatrixMultiply const& expr)
            {
                throw std::runtime_error("Scaled Matrix multiply present in runtime expression.");
            }

            CommandArgumentValue operator()(Register::ValuePtr const& expr)
            {
                if(expr->regType() == Register::Type::Literal)
                    return expr->getLiteralValue();

                Throw<FatalError>("Register present in runtime expression", ShowValue(expr));
            }

            CommandArgumentValue operator()(ToScalar const& expr)
            {
                return call(expr.arg);
            }

            CommandArgumentValue operator()(CommandArgumentPtr const& expr)
            {
                return expr->getValue(args);
            }

            CommandArgumentValue operator()(CommandArgumentValue const& expr)
            {
                return expr;
            }

            CommandArgumentValue operator()(AssemblyKernelArgumentPtr const& expr)
            {
                return call(expr->getExpression());
            }

            CommandArgumentValue operator()(DataFlowTag const& expr)
            {
                Throw<FatalError>("Data flow tag present in runtime expression", ShowValue(expr));
            }

            CommandArgumentValue operator()(PositionalArgument const& expr)
            {
                Throw<FatalError>("Positional argument present in runtime expression",
                                  ShowValue(expr));
            }

            CommandArgumentValue operator()(WaveTilePtr const& expr)
            {
                Throw<FatalError>("Wave tile present in runtime expression", ShowValue(expr));
            }

            CommandArgumentValue call(Expression const& expr)
            {
                auto rv = std::visit(*this, expr);

                auto exprType = resultVariableType(expr);
                auto result   = resultType(rv).varType;
                AssertFatal(
                    exprType == result, ShowValue(expr), ShowValue(exprType), ShowValue(result));

                return rv;
            }

            CommandArgumentValue call(ExpressionPtr const& expr)
            {
                AssertFatal(expr != nullptr, "Found nullptr in expression");
                return call(*expr);
            }

            CommandArgumentValue call(ExpressionPtr const&    expr,
                                      RuntimeArguments const& runtimeArgs)
            {
                args = runtimeArgs;
                return call(expr);
            }

            CommandArgumentValue call(Expression const& expr, RuntimeArguments const& runtimeArgs)
            {
                args = runtimeArgs;
                return call(expr);
            }
        };

        CommandArgumentValue evaluate(ExpressionPtr const& expr, RuntimeArguments const& args)
        {
            return EvaluateVisitor().call(expr, args);
        }

        CommandArgumentValue evaluate(Expression const& expr, RuntimeArguments const& args)
        {
            return EvaluateVisitor().call(expr, args);
        }

        CommandArgumentValue evaluate(ExpressionPtr const& expr)
        {
            return EvaluateVisitor().call(expr);
        }

        CommandArgumentValue evaluate(Expression const& expr)
        {
            return EvaluateVisitor().call(expr);
        }

        bool canEvaluateTo(CommandArgumentValue val, ExpressionPtr const& expr)
        {
            if(evaluationTimes(expr)[EvaluationTime::Translate])
            {
                return evaluate(expr) == val;
            }
            return false;
        }

        std::optional<CommandArgumentValue> tryEvaluate(ExpressionPtr const& expr)
        {
            return expr ? tryEvaluate(*expr) : std::nullopt;
        }

        std::optional<CommandArgumentValue> tryEvaluate(Expression const& expr)
        {
            if(evaluationTimes(expr)[EvaluationTime::Translate])
                return evaluate(expr);
            return std::nullopt;
        }
    }
}
