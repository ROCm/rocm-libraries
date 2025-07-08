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

#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index_container.hpp>

#include <functional>
#include <map>
#include <ranges>
#include <unordered_set>
#include <variant>
#include <vector>

#include <rocRoller/Graph/Hypergraph_fwd.hpp>
#include <rocRoller/Serialization/Base_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace Graph
    {
        enum class ElementType : int
        {
            Node = 0,
            Edge,
            Count
        };

        std::string toString(ElementType e);

        /**
         * @brief Returns the complementary type to `t`.
         */
        ElementType getConnectingType(ElementType t);

        enum class Direction : int
        {
            Upstream = 0,
            Downstream,
            Count
        };

        std::string         toString(Direction d);
        constexpr Direction opposite(Direction d);

        enum class GraphModification : int
        {
            DeleteElement = 0,
            AddElement,
            SetElement,
            Count
        };

        std::string toString(GraphModification m);

        namespace mi = boost::multi_index;

        struct HypergraphIncident
        {
            int src;
            int dst;
            int edgeOrder;
        };

        template <typename Node, typename Edge, bool Hyper = true>
        class Hypergraph
        {
        public:
            using Element = std::variant<Node, Edge>;

            inline static constexpr bool IsHyper = Hyper;

            static std::string ElementName(Element const& el);

            using Incident = HypergraphIncident;

            /**
             *
             */
            struct Location
            {
                int              index;
                std::vector<int> incoming;
                std::vector<int> outgoing;

                Element element;

                constexpr inline bool operator==(Location const& rhs) const;
            };

            bool exists(int index) const;

            /**
             * @brief Returns whether `index` points to a node or an edge.
             */
            ElementType getElementType(int index) const;

            template <typename T = Node>
            T getNode(int index) const;

            template <typename T = Edge>
            T getEdge(int index) const;

            /**
             * @brief Returns whether `e` is a node or an edge.
             */
            ElementType getElementType(Element const& e) const;

            virtual void clearCache(GraphModification modification);

            template <typename T>
            int addElement(T&& element);

            /**
             * Adds each element to the graph and returns a tuple of int.
             */
            template <typename T, typename... Ts>
            auto addElements(T&& element, Ts&&... rest);

            /**
             * @brief Set (overwrite) existing element.
             *
             * Asserts that the index exists already.
             */
            template <typename T>
            void setElement(int index, T&& element);

            template <typename T>
            int addElement(T&&                        element,
                           std::initializer_list<int> inputs,
                           std::initializer_list<int> outputs);

            template <typename T, CForwardRangeOf<int> T_Inputs, CForwardRangeOf<int> T_Outputs>
            int addElement(T&& element, T_Inputs const& inputs, T_Outputs const& outputs);

            template <typename T, CForwardRangeOf<int> T_Inputs, CForwardRangeOf<int> T_Outputs>
            void addElement(int              index,
                            T&&              element,
                            T_Inputs const&  inputs,
                            T_Outputs const& outputs);

            void deleteElement(int index);

            template <CForwardRangeOf<int>        T_Inputs,
                      CForwardRangeOf<int>        T_Outputs,
                      std::predicate<Edge const&> T_Predicate>
            void deleteElement(T_Inputs const&  inputs,
                               T_Outputs const& outputs,
                               T_Predicate      edgePredicate);

            template <typename T, CForwardRangeOf<int> T_Inputs, CForwardRangeOf<int> T_Outputs>
            requires(std::constructible_from<Edge, T>) void deleteElement(T_Inputs const&  inputs,
                                                                          T_Outputs const& outputs);

            size_t getIncidenceSize() const;
            size_t getElementCount() const;

            Element const& getElement(int index) const;

            /**
             * @brief Returns a Location info object detailing connections to the element `index`.
             */
            Location getLocation(int index) const;

            /**
             * @brief Yields element indices without any incoming connections.
             */
            Generator<int> roots() const;

            /**
             * @brief Yields element indices without any outgoing connections.
             */
            Generator<int> leaves() const;

            /**
            * @brief Yields element indices that are the child nodes of a given element
            */
            Generator<int> childNodes(int parent) const;

            /**
            * @brief Yields element indices that are the parent nodes of a given element
            */
            Generator<int> parentNodes(int child) const;

            Generator<int> allElements() const;

            /**
             * @brief Yields node indices connected in the specified direction to start, in depth-first order
             */
            Generator<int> depthFirstVisit(int start, Direction dir = Direction::Downstream) const;

            /**
             * @brief Yields node indices connected in the specified direction to start, that satisfy the node selector.
             */
            template <std::predicate<int> Predicate>
            Generator<int> findNodes(int       start,
                                     Predicate nodeSelector,
                                     Direction dir = Direction::Downstream) const;

            /**
             * @brief Yields node indices connected in the specified direction to start, that satisfy the node selector.
             */
            template <CForwardRangeOf<int> Range, std::predicate<int> Predicate>
            Generator<int> findNodes(Range const& starts,
                                     Predicate    nodeSelector,
                                     Direction    dir = Direction::Downstream) const;

            /**
             * @brief Yields node indices that satisfy the node selector.
             */
            template <std::predicate<int> Predicate>
            Generator<int> findElements(Predicate nodeSelector) const;

            /**
             * @brief Yields node indices connected in the specified direction to starts, in depth-first order
             */
            template <CForwardRangeOf<int> Range>
            Generator<int> depthFirstVisit(Range const& starts,
                                           Direction    dir = Direction::Downstream) const;

            /**
             * @brief Yields node indices connected in the specified direction to starts, in depth-first order.
             *
             * Will only visit through edges if the edgePredicate returns true.
             */
            template <CForwardRangeOf<int> Range, std::predicate<int> Predicate>
            Generator<int> depthFirstVisit(Range const& starts,
                                           Predicate    edgePredicate,
                                           Direction    dir = Direction::Downstream) const;

            template <std::predicate<int> Predicate>
            Generator<int> depthFirstVisit(int start, Predicate edgePredicate, Direction dir) const;

            /**
             * @brief Yields node indices connected in the specified direction to start, in depth-first order.
             *
             * Will not yield any nodes in `visitedNodes`, and will insert nodes `visitedNodes` to track already
             * visited nodes.
             */
            template <Direction Dir>
            Generator<int> depthFirstVisit(int start, std::unordered_set<int>& visitedNodes) const;

            template <Direction Dir, std::predicate<int> Predicate>
            Generator<int> depthFirstVisit(int                      start,
                                           Predicate                edgePredicate,
                                           std::unordered_set<int>& visitedNodes) const;

            /**
             * @brief Yields node indices connected downstream of start, in breadth-first order.
             */
            Generator<int> breadthFirstVisit(int       start,
                                             Direction dir = Direction::Downstream) const;
            Generator<int> breadthFirstVisitDownstream(int start) const;
            Generator<int> breadthFirstVisitUpstream(int start) const;

            /**
            * @brief Yields element indices (both nodes and edges) that form the paths
            * from the starts to the ends
            */
            template <Direction            Dir,
                      CForwardRangeOf<int> RangeStart,
                      CForwardRangeOf<int> RangeEnd,
                      std::predicate<int>  Predicate>
            Generator<int> path(RangeStart const&    starts,
                                RangeEnd const&      ends,
                                Predicate            edgeSelector,
                                std::map<int, bool>& visitedElements) const;

            template <Direction            Dir,
                      CForwardRangeOf<int> RangeStart,
                      CForwardRangeOf<int> RangeEnd,
                      std::predicate<int>  Predicate>
            Generator<int>
                path(RangeStart const& starts, RangeEnd const& ends, Predicate edgeSelector) const;

            template <Direction Dir, CForwardRangeOf<int> RangeStart, CForwardRangeOf<int> RangeEnd>
            Generator<int> path(RangeStart const& starts, RangeEnd const& ends) const;

            template <Direction Dir>
            Generator<int> getNeighbours(int const element) const;

            Generator<int> getNeighbours(int const element, Direction Dir) const;

            /**
             * @brief Return edges in topological order.
             *
             * Traversing edges in topological order preserves edge
             * dependencies.
             */
            Generator<int> topologicalSort() const;

            /**
             * @brief Return edges in reverse topological order.
             *
             * Traversing edges in reverse topological order can preserves edge
             * dependencies.
             */
            Generator<int> reverseTopologicalSort() const;

            std::string toDOT(std::string const& prefix = "", bool standalone = true) const;

            static bool identity(Edge const&)
            {
                return true;
            }

            template <std::predicate<Edge const&> Predicate>
            std::string toDOT(Predicate edgePredicate = identity) const;

            template <typename T>
            requires(std::constructible_from<Node, T> || std::constructible_from<Edge, T>)
                Generator<int> getElements()
            const;

            /**
             * @brief Yields indices of all Nodes of class T.
             */
            template <typename T = Node>
            requires(std::constructible_from<Node, T>) Generator<int> getNodes()
            const;

            /**
             * Return all Edges of class T.
             */
            template <typename T = Edge>
            requires(std::constructible_from<Edge, T>) Generator<int> getEdges()
            const;

            /**
             * @brief Yields indices of nodes immediately connected to `dst` through Edges of type T, in direction Dir.
             */
            template <typename T, Direction Dir>
            requires(std::constructible_from<Edge, T>) Generator<int> getConnectedNodeIndices(
                int const dst)
            const;

            /**
             * @brief Yields indices of nodes immediately connected to `dst` through Edges that satisfy the edgePredicate, in direction Dir.
             */
            template <Direction Dir, std::predicate<Edge const&> Predicate>
            Generator<int> getConnectedNodeIndices(int const dst, Predicate edgePredicate) const;

            /**
             * @brief Yields indices of nodes that immediately preceed `dst` where the Edges are of type T.
             */
            template <typename T>
            requires(std::constructible_from<Edge, T>) Generator<int> getInputNodeIndices(
                int const dst)
            const;

            /**
             * @brief Yields indices of nodes that immediately preceed `dst` where the Edges satisfy the edgePredicate.
             */
            template <std::predicate<Edge const&> Predicate>
            Generator<int> getInputNodeIndices(int const dst, Predicate edgePredicate) const;

            Generator<std::tuple<int, Edge>> getInputNodesAndEdges(int dst);

            /**
             * @brief Yields indices of nodes that immediately follow `src` where the Edges are of type T.
             */
            template <typename T>
            requires(std::constructible_from<Edge, T>) Generator<int> getOutputNodeIndices(
                int const src)
            const;

            /**
             * @brief Yields indices of nodes that immediately follow `src` where the Edges satisfy the edgePredicate.
             */
            template <std::predicate<Edge const&> Predicate>
            Generator<int> getOutputNodeIndices(int const src, Predicate edgePredicate) const;

            /**
             * @brief Return all downstream nodes that are connected to `candidates` via the specified edge type.
             * The set of original candidates is included in the returned set.
             *
             * Note that this function recursively follows edges.
             *
             * @param candidates Set of node ids
             * @return std::set<int> Set of node ids expanded
             */
            template <typename T>
            requires(std::constructible_from<Edge, T>) std::set<int> followEdges(
                std::set<int> const& candidates);

            // clang-format off
        private:
            // clang-format on
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;
            int m_nextIndex = 1;

            mutable std::map<int, Location> m_locationCache;

            // TODO: May need to replace with multi_index for in-place rewriting.
            std::map<int, Element> m_elements;

            struct BySrc
            {
            };
            struct ByDst
            {
            };
            struct BySrcDst
            {
            };

            using Incidence = mi::multi_index_container<
                Incident,
                mi::indexed_by<
                    mi::ordered_non_unique<mi::tag<BySrc>,
                                           mi::key<&Incident::src, &Incident::edgeOrder>>,
                    mi::ordered_non_unique<mi::tag<ByDst>,
                                           mi::key<&Incident::dst, &Incident::edgeOrder>>,
                    // This prevents parallel incidents.
                    mi::ordered_unique<mi::tag<BySrcDst>,
                                       mi::key<&Incident::src, &Incident::dst>>>>;

            Incidence m_incidence;

            template <Direction Dir>
            bool edgeSatisfied(int const edge, std::map<int, bool> const& visitedElements) const;
        };

        template <typename Node, typename Edge, bool Hyper>
        std::ostream& operator<<(std::ostream& stream, Hypergraph<Node, Edge, Hyper> const& graph);

        template <typename Cls>
        std::string variantToString(Cls const& el)
        {
            return std::visit([](auto const& v) { return toString(v); }, el);
        }

        /**
         * Yields nodes connected to start:
         *
         * - In direction `Dir`
         * - Connected to `start` by edges satisfying `edgePredicate` and nodes satisfying `nodePredicate`
         * - The nodes that are yielded must satisfy destNodePredicate.
         *
         * @param graph A graph
         * @param start The starting node
         * @param nodePredicate Only traverse nodes that satisfy this predicate.
         * @param edgePredicate Only traverse edges that satisfy this predicate.
         * @param destNodePredicate Only yield nodes that satisfy this predicate.
         */
        template <Graph::Direction Dir, typename Node, typename Edge, bool Hyper>
        Generator<int> reachableNodes(Graph::Hypergraph<Node, Edge, Hyper> const& graph,
                                      int                                         start,
                                      auto                                        nodePredicate,
                                      auto                                        edgePredicate,
                                      auto destNodePredicate);

        template <typename T>
        concept CCalmGraph = !T::IsHyper;
    }
}

#include <rocRoller/Graph/Hypergraph_impl.hpp>
