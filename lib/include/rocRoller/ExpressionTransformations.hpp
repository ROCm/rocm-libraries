#pragma once

#include <Context_fwd.hpp>
#include <Expression_fwd.hpp>

namespace rocRoller
{
    namespace Expression
    {
        ExpressionPtr identity(ExpressionPtr expr);

        ExpressionPtr launchTimeSubExpressions(ExpressionPtr expr, ContextPtr context);

        /**
         * @brief Attempt to replace division operations found within an expression with faster operations.
         *
         * @param expr Input expression
         * @param context
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr fastDivision(ExpressionPtr expr, ContextPtr context);

        /**
         * @brief Attempt to replace multiplication operations found within an expression with faster operations.
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr fastMultiplication(ExpressionPtr expr);

        /**
         * @brief Simplify expressions
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr simplify(ExpressionPtr expr);

        /**
         * @brief Fuse binary expressions into ternaries.
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr fuseTernary(ExpressionPtr expr);

        /**
         * @brief Fuse binary expressions if one combination is able to be condensed by association
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr fuseAssociative(ExpressionPtr expr);

        /**
         * Helper (lambda/transducer) for applying all fast arithmetic transformations.
         *
         * Usage:
         *
         *   FastArithmetic transformer(context);
         *   auto fast_expr = transformer(expr);
         *
         * Can also be passed as an ExpressionTransducer.
         */
        struct FastArithmetic
        {
            FastArithmetic() = delete;
            FastArithmetic(ContextPtr);

            ExpressionPtr operator()(ExpressionPtr) const;

        private:
            ContextPtr m_context;
        };
    }
}
