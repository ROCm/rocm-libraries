// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace ExpressionMatchers
{
    struct EquivalentExpressionMatcher : Catch::Matchers::MatcherGenericBase
    {

        EquivalentExpressionMatcher(rocRoller::Expression::ExpressionPtr        expr,
                                    rocRoller::Expression::AlgebraicProperties  props,
                                    rocRoller::Expression::ExpressionTransducer transformation
                                    = nullptr)
            : m_reference(expr)
            , m_properties(props)
            , m_transformation(transformation)
        {
        }

        bool match(rocRoller::Expression::ExpressionPtr result) const
        {
            if(equivalent(result, m_reference, m_properties))
                return true;

            if(m_transformation)
            {
                auto transformedReference = m_transformation(m_reference);
                auto transformedResult    = m_transformation(result);

                if(equivalent(result, transformedReference, m_properties))
                    return true;

                if(equivalent(transformedResult, m_reference, m_properties))
                    return true;

                if(equivalent(transformedResult, transformedReference, m_properties))
                    return true;
            }

            return false;
        }

        std::string describe() const override
        {
            return rocRoller::concatenate(m_reference, ", properties: ", m_properties);
        }

    private:
        rocRoller::Expression::ExpressionPtr        m_reference;
        rocRoller::Expression::AlgebraicProperties  m_properties;
        rocRoller::Expression::ExpressionTransducer m_transformation;
    };

    struct IdenticalExpressionMatcher : Catch::Matchers::MatcherGenericBase
    {
        IdenticalExpressionMatcher(rocRoller::Expression::ExpressionPtr expr)
            : m_reference(expr)
        {
        }

        bool match(rocRoller::Expression::ExpressionPtr result) const
        {
            return identical(result, m_reference);
        }

        std::string describe() const override
        {
            return rocRoller::concatenate("\n", m_reference);
        }

    private:
        rocRoller::Expression::ExpressionPtr m_reference;
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
 * Checks that the expression is equivalent to expr, when considering the given
 * properties.
 */
inline auto EquivalentTo(rocRoller::Expression::ExpressionPtr       expr,
                         rocRoller::Expression::AlgebraicProperties props
                         = rocRoller::Expression::AlgebraicProperties::All())
{
    return ExpressionMatchers::EquivalentExpressionMatcher(expr, props, nullptr);
}

/**
 * Checks that the expression is equivalent to expr, when considering the given
 * properties and expression transformation
 */
inline auto SimplifiesTo(rocRoller::Expression::ExpressionPtr       expr,
                         rocRoller::Expression::AlgebraicProperties props
                         = rocRoller::Expression::AlgebraicProperties::All(),
                         rocRoller::Expression::ExpressionTransducer xform
                         = rocRoller::Expression::simplify)
{
    return ExpressionMatchers::EquivalentExpressionMatcher(expr, props, xform);
}

/**
 * Checks that the expression is identical to expr.
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
