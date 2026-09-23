// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
             * The root node register can be set to nullptr before a dest is
             * determined.
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

            Register::Type regType() const;

            /**
             * A set of dependencies that this expression relies on.
             * These are the indices of nodes in the parent ExpressionTree.
             *
             * NOTE: std::unordered_set doesn't work with std::includes
             */
            std::set<int> deps;

            /**
             * Metric that tracks the number of subexpressions that have been
             * eliminated. This count is for this node and its dependencies.
             *
             */
            int consolidationCount = 0;

            /**
             * What is the maximum distance of this node from the root node?
             *
             * This is used as a heuristic to avoid allocating temporary
             * registers a long time before they are used.
             */
            int distanceFromRoot = 0;

            /**
             * This is used as a heuristic to guide which nodes to generate first
             * to avoid allocating temporary registers a long time before they are used.
             */
            int priorityOrder = 0;
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

        /**
         * Calculates the distanceFromRoot values in every node of `tree`.
         *
         * Assumes (and asserts) that it is sorted in topological order.
         */
        void updateDistances(ExpressionTree& tree);

        /**
         * Calculates the priorityOrder values in every node of `tree` using a
         * Sethi-Ullman inspired algorithm.
         *
         * This computes weights that approximate register pressure and then
         * determines a traversal order where heavier subtrees are evaluated first.
         * This helps minimize register pressure during code generation.
         *
         * Assumes (and asserts) that it is sorted in topological order.
         */
        void updatePriorityOrder(ExpressionTree& tree);

        /**
         * Returns a string containing some interesting statistical information
         * about `tree`.
         *
         * Assumes (and asserts) that it is sorted in topological order.
         */
        std::string statistics(ExpressionTree tree);

        /**
         * Returns a DOT/graphviz representation of `tree`.
         */
        std::string toDOT(ExpressionTree const& tree);
    }
}
