#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/TagType.hpp>

#include "Dimension.hpp"
#include "Edge.hpp"
#include "Utilities/Utils.hpp"

namespace rocRoller
{
    /**
     * Coordinate transform (index calculations) routines.
     *
     * Coordinate transforms are represented as graphs.  A coordinate
     * transform graph encodes:
     *
     * - algorithm decomposition onto hardware
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
    namespace KernelGraph::CoordinateTransform
    {
        /*
         * CoordinateGraph
         */

        using NodeMapType = std::map<TagType, Dimension>;
        using DimsKeyType = std::vector<TagType>;

        struct EdgeKeyType
        {
            DimsKeyType stags;
            DimsKeyType dtags;
            EdgeType    type = EdgeType::None;

            bool operator<(const EdgeKeyType& b) const
            {
                if(type != b.type)
                    return type < b.type;
                if(stags != b.stags)
                    return stags < b.stags;
                if(dtags != b.dtags)
                    return dtags < b.dtags;
                return false;
            }

            bool operator==(const EdgeKeyType& b) const
            {
                return type == b.type && stags == b.stags && dtags == b.dtags;
            }
        };

        using EdgeMapType = std::map<EdgeKeyType, Edge>;

        /**
         * Coordinate-transform HyperGraph.
         *
         * Nodes in the graph represent single dimensions (or
         * coordinates).
         *
         * Hyper-edges describe how to transform coordinates and/or
         * apply operations.
         */
        class HyperGraph
        {
        public:
            HyperGraph() {}
            HyperGraph(int nextTag)
                : m_nextTag(nextTag)
            {
            }

            /**
             * Add dimensions (nodes) to the graph.
             *
             * Not usually necessary as adding edges implicitly adds nodes.
             */
            TagType              addDimension(Dimension& dim);
            std::vector<TagType> addDimensions(std::vector<Dimension>& dims);

            /**
             * Assigns a new, unused tag value to `dim`.
             */
            TagType allocateTag(Dimension& dim);

            /**
             * Returns a new, unused tag.
             */
            int allocateTag();

            /**
             * Returns a tag value that is greater than all tag values currently in this graph.
             */
            int nextTag() const;

            /**
             * Set/reset dimension.
             */
            void resetDimension(Dimension dim);

            /**
             * Return single dimension given tag.
             */
            Dimension getDimension(TagType tag) const;

            template <typename T>
            T getDimension(T dim) const;

            /**
             * Return vector of dimensions given tags.
             */
            std::vector<Dimension> getDimensions(std::vector<TagType> tags) const;

            std::vector<Dimension> getOutputs(TagType tag, EdgeType edgeType) const;

            /**
             * Return vector of all dimensions.
             */
            std::vector<Dimension> getDimensions() const;

            /**
             * Return vector of (linear) dimensions given dataflow tags.
             */
            std::vector<Dimension> getLinearDimensions(std::unordered_set<int> ndtags) const;

            /**
             * Add a hyper-edge to the graph; connects nodes in `srcs` to nodes in `dsts`.
             */
            void addEdge(std::vector<Dimension>  srcs,
                         std::vector<Dimension>  dsts,
                         CoordinateTransformEdge edge);

            void addEdge(std::vector<Dimension> srcs, std::vector<Dimension> dsts, DataFlow edge);

            /**
             * Return all edge keys.
             */
            std::vector<EdgeKeyType> getEdges() const;

            /**
             * Return edge.
             */
            Edge getEdge(EdgeKeyType e) const;

            /**
             * Remove a hyper-edge from the graph.
             */
            void removeEdge(EdgeKeyType ekey);

            /**
             * Forward (bottom-up) coordinate transform.
             *
             * Given the expressions in `indexes` corresponding to the
             * dimensions in `srcs`, traverse the graph until the
             * `dsts` dimensions are reached.  While traversing the
             * graph, expressions to compute the coordinate transforms
             * from `srcs` to `dsts` are computed according to the
             * edges in the graph.
             */
            std::vector<Expression::ExpressionPtr>
                forward(std::vector<Expression::ExpressionPtr> const& indexes,
                        std::vector<Dimension> const&                 srcs,
                        std::vector<Dimension> const&                 dsts,
                        Expression::ExpressionTransducer              transducer = nullptr) const;

            /**
             * Reverse (top-down) coordinate transform.
             *
             * Same as the forward transform, except the `indexes`
             * correspond to the dimensions in `dsts`, can we compute
             * the coordinate transforms for the dimensions in `srcs`.
             */
            std::vector<Expression::ExpressionPtr>
                reverse(std::vector<Expression::ExpressionPtr> const& indexes,
                        std::vector<Dimension> const&                 srcs,
                        std::vector<Dimension> const&                 dsts,
                        Expression::ExpressionTransducer              transducer = nullptr) const;

            template <typename Visitor>
            std::vector<Expression::ExpressionPtr>
                traverse(std::vector<Expression::ExpressionPtr> const& indexes,
                         std::vector<Dimension> const&                 srcs,
                         std::vector<Dimension> const&                 dsts,
                         bool                                          forward,
                         Visitor&                                      visitor,
                         Expression::ExpressionTransducer              transducer = nullptr) const;
            /**
             * Return string representation of graph.
             *
             * By default, the graph is traversed in topological order
             * when printing.
             *
             * When `topologicalOrder` is false, the graph is printed
             * as it is stored; this can be useful for debugging.
             */
            std::string toString(bool topologicalOrder = true) const;

            /**
             * Return DOT representation of graph.
             *
             * By default, the graph is traversed in topological order.
             *
             * When `topologicalOrder` is false, the graph is printed
             * as it is stored; this can be useful for debugging.
             */
            std::string toDOT(bool topologicalOrder = true) const;

            /**
             * Return "bottom" dimensions.
             *
             * Usually these would be the user dimensions that are read from.
             */
            std::vector<Dimension> bottom(EdgeType type = EdgeType::Any) const;

            /**
             * Return "top" dimensions.
             *
             * Usually these would be the user dimensions that are written to.
             */
            std::vector<Dimension> top(EdgeType type = EdgeType::Any) const;

            /**
             * Return ordered edges.
             *
             * Starting from the nodes/dimensions in `bottom()`
             * dimensions, walk the graph and return edges in
             * topological order until the `top()` dimensions are
             * reached.
             */
            std::vector<EdgeKeyType> topographicalSort(EdgeType type) const;

            /**
             * Return ordered edges along path from `srcs` to `dsts`.
             *
             * Starting from the tags in `srcs`, walk the graph and
             * return edges in topological order until the `dsts` tags
             * are reached.
             */
            std::vector<EdgeKeyType> path(std::vector<TagType> const& srcs,
                                          std::vector<TagType> const& dsts,
                                          EdgeType                    type,
                                          bool                        forward = true) const;

        private:
            NodeMapType m_nodes;
            EdgeMapType m_edges;

            int m_nextTag = -1;
        };

        /*
         * Helpers
         */

        template <typename T>
        inline T HyperGraph::getDimension(T dim) const
        {
            AssertRecoverable(m_nodes.count(getTag(dim)) > 0, "No dimension: " + dim.toString());
            return std::get<T>(getDimension(getTag(dim)));
        }
    }
}

#include "HyperGraph_impl.hpp"
