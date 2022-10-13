
#pragma once

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/KernelGraph/CoordinateTransform/HyperGraph.hpp>
#include <rocRoller/KernelGraph/CoordinateTransform/Transformer_fwd.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateTransform
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
            Transformer(std::shared_ptr<HyperGraph>,
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
            void setCoordinate(Dimension const&, Expression::ExpressionPtr);

            /**
             * Remove the index expression for the dimension.
             */
            void removeCoordinate(Dimension const&);

            /**
             * Forward (bottom-up) coordinate transform.
             *
             * Given current location (see set()), traverse the graph
             * until the `dsts` dimensions are reached.
             */
            std::vector<Expression::ExpressionPtr> forward(std::vector<Dimension> const&) const;

            /**
             * Reverse (top-down) coordinate transform.
             */
            std::vector<Expression::ExpressionPtr> reverse(std::vector<Dimension> const&) const;

            /**
             * Forward incremental coordinate transform.
             */
            std::vector<Expression::ExpressionPtr> forwardStride(
                Dimension const&, Expression::ExpressionPtr, std::vector<Dimension> const&) const;

            /**
             * Reverse incremental coordinate transform.
             */
            std::vector<Expression::ExpressionPtr> reverseStride(
                Dimension const&, Expression::ExpressionPtr, std::vector<Dimension> const&) const;

            /**
             * Implicitly set indexes for all Workgroup and Workitem dimensions in the graph.
             */
            void fillExecutionCoordinates();

        private:
            template <typename Visitor>
            std::vector<Expression::ExpressionPtr>
                stride(std::vector<Dimension> const&, bool forward, Visitor& visitor) const;

            std::map<TagType, Dimension>                 m_dimensions;
            std::map<TagType, Expression::ExpressionPtr> m_indexes;
            std::shared_ptr<HyperGraph>                  m_graph;
            std::shared_ptr<Context>                     m_context;
            Expression::ExpressionTransducer             m_transducer;
        };
    }
}
