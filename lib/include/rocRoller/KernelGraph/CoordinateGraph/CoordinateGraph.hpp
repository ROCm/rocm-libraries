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

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateEdge.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>

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
     * Given the geometry of a tensor (sizes, strides), the graph can
     * be used to determine index values of given working coordinates.
     *
     * The graph answers the question:
     * What is the relationship between the dimensions of:
     *  - Input and output tensors
     *  - GPU execution constructs such as workgroups, workitems, wavefronts
     *  - Software constructs such as loops & unrolls
     *  - Temporary storage such as registers (V/SGPRs), LDS, scratch space
     *
     * It maps problem memory space to the hardware aspects, like
     * workgroups, workitems, VGPRs, LDS, and software constraints,
     * like loop dimensions/unrolls.
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
                        std::vector<int> const&                dsts) const;

            std::vector<Expression::ExpressionPtr>
                reverse(std::vector<Expression::ExpressionPtr> sdims,
                        std::vector<int> const&                srcs,
                        std::vector<int> const&                dsts) const;

            EdgeType getEdgeType(int index) const;

            template <Graph::Direction Dir, typename Visitor>
            std::vector<Expression::ExpressionPtr>
                traverse(std::vector<Expression::ExpressionPtr> sdims,
                         std::vector<int> const&                srcs,
                         std::vector<int> const&                dsts,
                         Visitor&                               visitor) const;

            template <Graph::Direction Dir>
            bool hasPath(std::vector<int> const& srcs, std::vector<int> const& dsts) const;

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

#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph_impl.hpp>
