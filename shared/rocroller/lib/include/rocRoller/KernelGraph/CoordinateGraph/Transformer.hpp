// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
             * Set the Transformer's coordinate graph.
             */
            void setCoordinateGraph(CoordinateGraph const* graph)
            {
                AssertFatal(graph != nullptr);
                m_graph = graph;
            }

            auto const& getIndexes() const
            {
                return m_indexes;
            }

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
             *
             * Uses Workgroups and Workitems from the context.
             */
            void fillExecutionCoordinates(ContextPtr context);

            /**
             * Implicitly set indexes for all Workgroup and Workitem dimensions in the graph
             * using the provided expression arrays.
             *
             * Does not update the RegisterTagManager; use fillExecutionCoordinates(ContextPtr)
             * when a live context is available.
             */
            void fillExecutionCoordinates(
                std::array<Expression::ExpressionPtr, 3> const& kernelWorkgroupIndexes,
                std::array<Expression::ExpressionPtr, 3> const& kernelWorkitemIndexes);

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
