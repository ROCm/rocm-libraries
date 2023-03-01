
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
            Transformer(std::shared_ptr<CoordinateGraph>,
                        std::shared_ptr<Context>         = nullptr,
                        Expression::ExpressionTransducer = nullptr);

            /**
             * Set/get expression transducer.
             */
            void setTransducer(Expression::ExpressionTransducer);

            Expression::ExpressionTransducer getTransducer() const;

            /**
             * Set the index expression for the dimension.
             */
            void setCoordinate(int, Expression::ExpressionPtr);

            Expression::ExpressionPtr getCoordinate(int) const;

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
             * Implicitly set indexes for all Workgroup and Workitem dimensions in the graph.
             */
            void fillExecutionCoordinates();

        private:
            template <typename Visitor>
            std::vector<Expression::ExpressionPtr>
                stride(std::vector<int> const&, bool forward, Visitor& visitor) const;

            std::map<int, Expression::ExpressionPtr> m_indexes;
            std::shared_ptr<CoordinateGraph>         m_graph;
            std::shared_ptr<Context>                 m_context;
            Expression::ExpressionTransducer         m_transducer;
        };
    }
}
