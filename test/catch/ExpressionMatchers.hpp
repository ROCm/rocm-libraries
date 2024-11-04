
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/Expression.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace ExpressionMatchers
{
    struct EquivalentExpressionMatcher : Catch::Matchers::MatcherGenericBase
    {

        EquivalentExpressionMatcher(rocRoller::Expression::ExpressionPtr       expr,
                                    rocRoller::Expression::AlgebraicProperties props)
            : m_expression(expr)
            , m_properties(props)
        {
        }

        bool match(rocRoller::Expression::ExpressionPtr result) const
        {
            return equivalent(result, m_expression, m_properties);
        }

        std::string describe() const override
        {
            return rocRoller::concatenate(m_expression, ", properties: ", m_properties);
        }

    private:
        rocRoller::Expression::ExpressionPtr       m_expression;
        rocRoller::Expression::AlgebraicProperties m_properties;
    };

    struct IdenticalExpressionMatcher : Catch::Matchers::MatcherGenericBase
    {
        IdenticalExpressionMatcher(rocRoller::Expression::ExpressionPtr expr)
            : m_expression(expr)
        {
        }

        bool match(rocRoller::Expression::ExpressionPtr result) const
        {
            return identical(result, m_expression);
        }

        std::string describe() const override
        {
            return rocRoller::concatenate("\n", m_expression);
        }

    private:
        rocRoller::Expression::ExpressionPtr m_expression;
    };

    template <rocRoller::Expression::CExpression T>
    struct ExpressionContainsTypeMatcher : Catch::Matchers::MatcherGenericBase
    {
        ExpressionContainsTypeMatcher() {}

        bool match(rocRoller::Expression::ExpressionPtr result) const
        {
            return contains<T>(result);
        }

        std::string describe() const override
        {
            return "Contains " + rocRoller::Expression::name(T{});
        }
    };
}

/**
 * Checks that the expression executes and returns hipSuccess (== 0).
 * Gets the Hip error on failure.
 */
inline auto EquivalentTo(rocRoller::Expression::ExpressionPtr       expr,
                         rocRoller::Expression::AlgebraicProperties props
                         = rocRoller::Expression::AlgebraicProperties::All())
{
    return ExpressionMatchers::EquivalentExpressionMatcher(expr, props);
}

/**
 * Checks that the expression executes and returns hipSuccess (== 0).
 * Gets the Hip error on failure.
 */
inline auto IdenticalTo(rocRoller::Expression::ExpressionPtr expr)
{
    return ExpressionMatchers::IdenticalExpressionMatcher(expr);
}

template <rocRoller::Expression::CExpression T>
inline auto Contains()
{
    return ExpressionMatchers::ExpressionContainsTypeMatcher<T>();
}
