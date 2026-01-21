// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Expression.hpp>
#include <rocRoller/Expression_evaluate_detail.hpp>

namespace rocRoller::Expression::EvaluateDetail
{
    template <CBinary BinaryExpr>
    struct BinaryEvaluatorVisitor
    {
        using TheEvaluator = OperationEvaluatorVisitor<BinaryExpr>;

        template <CCommandArgumentValue LHS, CCommandArgumentValue RHS>
        requires CCanEvaluateBinary<TheEvaluator, LHS, RHS>
            CommandArgumentValue operator()(LHS const& lhs, RHS const& rhs) const
        {
            auto evaluator = static_cast<TheEvaluator const*>(this);
            return evaluator->evaluate(lhs, rhs);
        }

        template <CCommandArgumentValue LHS, CCommandArgumentValue RHS>
        requires(!CCanEvaluateBinary<TheEvaluator, LHS, RHS>) CommandArgumentValue
            operator()(LHS const& lhs, RHS const& rhs) const
        {
            Throw<FatalError>("Type mismatch for expression: ",
                              friendlyTypeName<BinaryExpr>(),
                              ". Argument ",
                              friendlyTypeName<LHS>(),
                              " incompatible with ",
                              friendlyTypeName<RHS>(),
                              ").");
        }

        CommandArgumentValue call(CommandArgumentValue const& lhs,
                                  CommandArgumentValue const& rhs) const
        {
            return std::visit(*this, lhs, rhs);
        }
    };

    /**
     * For example, CAN_OPERATE_CONCEPT(Add, +) would define CCanAdd<LHS, RHS>
     * which is satisifed if (lhs + rhs) is a valid expression.
     */
#define CAN_OPERATE_CONCEPT(name, op)               \
    template <typename LHS, typename RHS>           \
    concept CCan##name = requires(LHS lhs, RHS rhs) \
    {                                               \
        lhs op rhs;                                 \
    }

    /**
     * Declares a BinaryEvaluatorVisitor that can be defined in C++ by a single
     * binary expression, such as Multiply which can be defined by (lhs * rhs).
     *
     * E.g. BINARY_EVALUATOR_VISITOR(Multiply, *) would declare for Multiply a
     * visitor that will apply the '*' operator to pairs of types where that is
     * valid, and throw an exception for pairs where that is not valid.
     */
#define BINARY_EVALUATOR_VISITOR(name, op)                                          \
    template <>                                                                     \
    struct OperationEvaluatorVisitor<name> : public BinaryEvaluatorVisitor<name>    \
    {                                                                               \
        template <CCommandArgumentValue LHS, CCommandArgumentValue RHS>             \
        requires CCan##name<LHS, RHS> constexpr auto evaluate(LHS const& lhs,       \
                                                              RHS const& rhs) const \
        {                                                                           \
            return lhs op rhs;                                                      \
        }                                                                           \
    }

#define SIMPLE_BINARY_OP(name, op) \
    CAN_OPERATE_CONCEPT(name, op); \
    BINARY_EVALUATOR_VISITOR(name, op)

    template <CBinary T>
    __attribute__((noinline)) CommandArgumentValue
        evaluateOp(T const& op, CommandArgumentValue const& lhs, CommandArgumentValue const& rhs)
    {
        BinaryEvaluatorVisitor<T> visitor;
        return visitor.call(lhs, rhs);
    }

#define INSTANTIATE_BINARY_OP(Op)                 \
    template CommandArgumentValue evaluateOp<Op>( \
        Op const& op, CommandArgumentValue const& lhs, CommandArgumentValue const& rhs)

} // namespace rocRoller
