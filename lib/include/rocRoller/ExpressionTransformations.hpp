#pragma once

#include <Context_fwd.hpp>
#include <Expression_fwd.hpp>

namespace rocRoller
{
    namespace Expression
    {
        ExpressionPtr launchTimeSubExpressions(ExpressionPtr expr, ContextPtr context);

        /**
         * Attempt to replace division operations found within an expression with faster
         * operations.
         */
        ExpressionPtr fastDivision(ExpressionPtr expr, std::shared_ptr<Context> context);

        /**
         * Attempt to replace multiplication operations found within an expression with faster
         * operations.
         */
        ExpressionPtr fastMultiplication(ExpressionPtr expr);

        /**
         * Simplify expressions.
         */
        ExpressionPtr simplify(ExpressionPtr expr);

        /**
         * Fuse expressions.
         */
        ExpressionPtr fuse(ExpressionPtr expr);

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
            FastArithmetic(std::shared_ptr<Context>);

            ExpressionPtr operator()(ExpressionPtr);

        private:
            std::shared_ptr<Context> m_context;
        };
    }
}
