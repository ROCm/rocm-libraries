#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>

#include "CoordinateEdge.hpp"
#include "Dimension.hpp"

namespace rocRoller
{
    /**
     * Coordinate transform (index calculations) routines.
     *
     * Coordinate transforms are represented as graphs.  A coordinate
     * transform graph encodes:
     *
     * - data flow
     * - data locality
     * - how indexes are computed/transformed
     *
     * Nodes in the graph represent abstract "dimensions".  These can
     * represent, for example: tensors provided by the user, linear
     * arrays stored in LDS, or loop indexes.
     *
     * Edges in the graph represent how coordinates are transformed.
     *
     * Throughout this documentation we use the following notation to
     * describe coordinate transforms: consider a "flatten" transform
     * with input dimensions
     *
     *   I = Dimension(size=n_i, stride=s_i)
     *   J = Dimension(size=n_j, stride=s_j)
     *
     * and output dimension
     *
     *   O = Dimension().
     *
     * The forward coordinate transform is denoted
     *
     *   Flatten(I, J; O)(i, j) = i * n_j + j
     *
     * and the inverse coordinate transform is denoted
     *
     *   Flatten'(O; I, J)(o) = { o / n_j, o % n_j }.
     *
     * That is,
     *
     *   Flatten(input dimensions; output dimensions)(input indexes)
     *
     * and inverse
     *
     *   Flatten'(output dimensions; input dimensions)(output indexes)
     *
     *
     * Nodes may have negative tag values, to indicate unallocated tags.
     * When the node is added to a graph (control or coordinate), it will
     * be replaced with a positive value that is otherwise unused in that
     * graph.
     *
     */
    namespace KernelGraph::CoordinateGraph
    {
        /**
         * Coordinate-transform HyperGraph.
         *
         * Nodes in the graph represent single dimensions (or
         * coordinates).
         *
         * Hyper-edges describe how to transform coordinates and/or
         * apply operations.
         */
        class CoordinateGraph : public Graph::Hypergraph<Dimension, Edge>
        {
        public:
            using Base = Graph::Hypergraph<Dimension, Edge>;

            CoordinateGraph() = default;

            std::vector<Expression::ExpressionPtr>
                forward(std::vector<Expression::ExpressionPtr> sdims,
                        std::vector<int> const&                srcs,
                        std::vector<int> const&                dsts,
                        Expression::ExpressionTransducer       transducer);

            std::vector<Expression::ExpressionPtr>
                reverse(std::vector<Expression::ExpressionPtr> sdims,
                        std::vector<int> const&                srcs,
                        std::vector<int> const&                dsts,
                        Expression::ExpressionTransducer       transducer);

            EdgeType getEdgeType(int index);

            template <Graph::Direction Dir, typename Visitor>
            std::vector<Expression::ExpressionPtr>
                traverse(std::vector<Expression::ExpressionPtr> sdims,
                         std::vector<int> const&                srcs,
                         std::vector<int> const&                dsts,
                         Visitor&                               visitor,
                         Expression::ExpressionTransducer       transducer);

            /**
             * @brief Get a node/edge from the coordinate graph.
             *
             * If the element specified by tag cannot be converted to
             * T, the return value is empty.
             *
             * @param tag Graph tag/index.
             */
            template <typename T>
            requires(std::constructible_from<CoordinateGraph::Element, T>) std::optional<T> get(
                int tag)
            const;
        };

        std::string name(CoordinateGraph::Element const& el);
    }
}

#include "CoordinateGraph_impl.hpp"
