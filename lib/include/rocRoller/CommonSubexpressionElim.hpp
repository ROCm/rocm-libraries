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

#pragma once

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <set>
#include <vector>

namespace rocRoller
{
    namespace Expression
    {
        struct ExpressionNode
        {
            /**
             * The destination register for the expression.
             * Holds placeholder registers for temporary values.
             * The root node register can be set to nullptr before a dest is determined.
             *
             * Set to nullptr when done using to release the reference.
             */
            Register::ValuePtr reg;

            /**
             * The granular expression.
             *
             * Set to nullptr when done using to release register references.
             */
            ExpressionPtr expr;

            /**
             * A set of dependencies that this expression relies on.
             * These are the indices of nodes in the parent ExpressionTree.
             *
             * NOTE: std::unordered_set doesn't work with std::includes
             */
            std::set<int> deps;

            /**
             * Metric that tracks the number of subexpressions that have been eliminated.
             * This count is for this node and its dependencies.
             *
             */
            int consolidationCount = 0;
        };

        /**
         * The expression as a tree object.
         *
         * The root node will always be the back element of the collection.
         *
         */
        using ExpressionTree = std::vector<ExpressionNode>;

        /**
         * @brief Using Common Subexpression Elimination (CSE), transform the expression into a tree
         * and eliminate identical nodes.
         *
         * @param expr The expression to be consolidated
         * @param context
         * @return ExpressionTree
         */
        ExpressionTree consolidateSubExpressions(ExpressionPtr expr, ContextPtr context);

        /**
         * @brief Get the number of consolidations performed by Common Subexpression Elimination
         *
         * @param tree Tree to fetch count from
         * @return Count of subexpressions consolidatated from original expression
         */
        int getConsolidationCount(ExpressionTree const& tree);

        /**
         * @brief Rebuilds an Expression from an ExpressionTree
         *
         * @param tree The tree to rebuild
         * @return ExpressionPtr
         */
        ExpressionPtr rebuildExpression(ExpressionTree const& tree);
    }
}
