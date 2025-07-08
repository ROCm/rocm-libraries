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
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer_fwd.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {
        /**
         * Coordinate/index transformer.
         *
         * Workgroup and Workitem (work coordinates) are implicitly
         * set if a context is passed to the constructor.
         */
        class Transformer
        {
        public:
            Transformer() = delete;
            Transformer(CoordinateGraph const* graph);
            Transformer(CoordinateGraph const* graph, Expression::ExpressionTransducer transducer);

            /**
             * Set expression transducer.
             */
            void setTransducer(Expression::ExpressionTransducer);

            /**
             * Set the index expression for the dimension.
             */
            void setCoordinate(int, Expression::ExpressionPtr);

            /**
             * Get the index expression for the dimension.
             */
            Expression::ExpressionPtr getCoordinate(int) const;

            /**
             * Determine if dimension has a value already.
             */
            bool hasCoordinate(int) const;

            /**
             * Remove the index expression for the dimension.
             */
            void removeCoordinate(int);

            /**
             * Forward (bottom-up) coordinate transform.
             *
             * Given current location (see set()), traverse the graph
             * until the `dsts` dimensions are reached.
             */
            std::vector<Expression::ExpressionPtr> forward(std::vector<int> const&) const;

            /**
             * Reverse (top-down) coordinate transform.
             */
            std::vector<Expression::ExpressionPtr> reverse(std::vector<int> const&) const;

            /**
             * Forward incremental coordinate transform.
             */
            std::vector<Expression::ExpressionPtr>
                forwardStride(int, Expression::ExpressionPtr, std::vector<int> const&) const;

            /**
             * Reverse incremental coordinate transform.
             */
            std::vector<Expression::ExpressionPtr>
                reverseStride(int, Expression::ExpressionPtr, std::vector<int> const&) const;

            /**
             * True if we can reach the target.
             */
            bool hasPath(std::vector<int> const&, bool forward) const;

            /**
             * Implicitly set indexes for all Workgroup and Workitem dimensions in the graph.
             */
            void fillExecutionCoordinates(ContextPtr context);

        private:
            Expression::ExpressionPtr transduce(Expression::ExpressionPtr exp) const;
            std::vector<Expression::ExpressionPtr>
                transduce(std::vector<Expression::ExpressionPtr> exps) const;

            template <typename Visitor>
            std::vector<Expression::ExpressionPtr>
                stride(std::vector<int> const&, bool forward, Visitor& visitor) const;

            std::map<int, Expression::ExpressionPtr> m_indexes;

            CoordinateGraph const*           m_graph;
            Expression::ExpressionTransducer m_transducer;
        };
    }
}
