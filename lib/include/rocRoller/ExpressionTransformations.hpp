#pragma once

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Expression_fwd.hpp>

namespace rocRoller
{
    namespace Expression
    {
        ExpressionPtr identity(ExpressionPtr expr);

        /**
         * Transform sub-expressions of `expr` into new kernel arguments
         *
         * Return value should be Translate time or KernelExecute time evaluable.
         */
        ExpressionPtr launchTimeSubExpressions(ExpressionPtr expr, ContextPtr context);

        /**
         * Restore any command arguments that have been cleaned (transformed from command
         * arguments into kernel arguments.)
         *
         * Return value should be Translate time or KernelLaunch time evaluable.
         */
        ExpressionPtr restoreCommandArguments(ExpressionPtr expr);

        /**
         * @brief Attempt to replace division operations found within an expression with faster operations.
         *
         * @param expr Input expression
         * @param context
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr fastDivision(ExpressionPtr expr, ContextPtr context);

        /**
         * Ensures that the kernel arguments will include the magic constants required to divide/modulo
         * by `expr`.
         * Requires `expr` to have a type of either Int32 or Int64, and to be evaluable at kernel launch time.
         */
        void enableDivideBy(ExpressionPtr expr, ContextPtr context);

        /**
         * @brief Attempt to replace multiplication operations found within an expression with faster operations.
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr fastMultiplication(ExpressionPtr expr);

        /**
         * Attempt to combine multiple shifts:
         * - Opposite shifts by same amount: mask off bits that would be zeroed out.
         */
        ExpressionPtr combineShifts(ExpressionPtr expr);

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
         * Resolve all DataFlowTags in the given expression.
         * Each DataFlowTag is transformed into either an expression or a register value.
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr dataFlowTagPropagation(ExpressionPtr expr, ContextPtr context);

        /**
         * @brief Attempt to compute e^x operations by using exp2(x * log2(e)).
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr lowerExponential(ExpressionPtr expr);

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
            explicit FastArithmetic(ContextPtr);

            ExpressionPtr operator()(ExpressionPtr) const;
            ExpressionPtr call(ExpressionPtr) const;

        private:
            ContextPtr m_context;
        };

        /**
         * @brief Transform RandomNumber expression into a set of expressions that
         *        implement the PRNG algorithm when PRNG instruction is unavailable.
         *
         * @param expr Input expression
         * @param context
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr lowerPRNG(ExpressionPtr exp, ContextPtr context);

        /**
         * @brief Resolve all ValuePtr expressions that are bitfields into
         * BitFieldExtract expressions.
         *
         * @param expr Input expression
         * @return ExpressionPtr Transformed expression
         */
        ExpressionPtr lowerBitfieldValues(ExpressionPtr expr);
    }
}
