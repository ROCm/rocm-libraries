// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Expression.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>

namespace rocRoller::Expression::EvaluateDetail
{
    template <typename T>
    concept CIntegral = std::integral<T> && !std::same_as<bool, T>;

    template <typename T>
    concept isFP32 = std::same_as<float, T>;

    /**
     * Visitor for a specific Operation expression.  Performs that
     * specific Operation (Add, subtract, etc).  Does not walk the
     * expression tree.
     */
    template <typename T, DataType DESTTYPE = DataType::None>
    struct OperationEvaluatorVisitor
    {
    };

    /**
     * Visitor for an Expression.  Walks the expression tree,
     * calling the OperationEvaluatorVisitor to perform actual
     * operations.
     */
    struct EvaluateVisitor;

    /**
     * Is satisfied if the unary operation associated with TheEvaluator can be
     * applied to ARG.
     * e.g. TheEvaluator == OperationEvaluatorVisitor<Not>, Arg -> bool
     *
     *
     * @tparam TheEvaluator Specialization of OperationEvaluatorVisitor class
     *                      for a specific operation
     * @tparam ARG
     */
    template <typename TheEvaluator, typename ARG>
    concept CCanEvaluateUnary = requires(TheEvaluator ev, ARG arg)
    {
        requires CCommandArgumentValue<ARG>;

        {
            ev.evaluate(arg)
            } -> CCommandArgumentValue;
    };

    /**
     * Is satisfied if the binary operation associated with TheEvaluator can be
     * applied to LHS and RHS.
     * e.g. TheEvaluator == OperationEvaluatorVisitor<Add>, LHS -> int, RHS -> int
     *
     * Note that this depends on evaluate() not being defined for invalid pairs of
     * types, thus the use of e.g. `CCanAdd` below.
     *
     * @tparam TheEvaluator Specialization of OperationEvaluatorVisitor class
     *                      for a specific operation
     * @tparam LHS
     * @tparam RHS
     */
    template <typename TheEvaluator, typename LHS, typename RHS>
    concept CCanEvaluateBinary = requires(TheEvaluator ev, LHS lhs, RHS rhs)
    {
        requires CCommandArgumentValue<LHS>;
        requires CCommandArgumentValue<RHS>;

        {
            ev.evaluate(lhs, rhs)
            } -> CCommandArgumentValue;
    };

    template <typename TheEvaluator, typename LHS, typename R1HS, typename R2HS>
    concept CCanEvaluateTernary = requires(TheEvaluator ev, LHS lhs, R1HS r1hs, R2HS r2hs)
    {
        requires CCommandArgumentValue<LHS>;
        requires CCommandArgumentValue<R1HS>;
        requires CCommandArgumentValue<R2HS>;

        {
            ev.evaluate(lhs, r1hs, r2hs)
            } -> CCommandArgumentValue;
    };

    template <CUnary T>
    CommandArgumentValue evaluateOp(T const& op, CommandArgumentValue const& arg);

    template <CBinary T>
    CommandArgumentValue
        evaluateOp(T const& op, CommandArgumentValue const& lhs, CommandArgumentValue const& rhs);

    template <CTernary T>
    CommandArgumentValue evaluateOp(T const&                    op,
                                    CommandArgumentValue const& lhs,
                                    CommandArgumentValue const& r1hs,
                                    CommandArgumentValue const& r2hs);

    template <CCommandArgumentValue T>
    void assertNonNullPointer(T const& val)
    {
        if constexpr(std::is_pointer<T>::value)
        {
            AssertFatal(val, "Can't offset from nullptr!");
        }
    }

    CommandArgumentValue reinterpretTruncateValue(CommandArgumentValue const& value,
                                                  DataType                    targetDataType,
                                                  std::endian                 endianness);
}

#include <rocRoller/Expression_impl.hpp>
