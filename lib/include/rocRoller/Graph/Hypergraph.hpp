#pragma once

#include "Hypergraph_fwd.hpp"

#include <boost/multi_index_container.hpp>

#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>

#include <functional>
#include <map>
#include <ranges>
#include <unordered_set>
#include <variant>
#include <vector>

#include "../Utilities/Generator.hpp"
#include "../Utilities/Utils.hpp"

#include "../Serialization/Base_fwd.hpp"

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

        Direction opposite(Direction);

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

            template <typename T>
            T getNode(int index) const;

            /**
             * @brief Returns whether `e` is a node or an edge.
             */
            ElementType getElementType(Element const& e) const;

            virtual void clearCache();

            template <typename T>
            int addElement(T&& element);

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
            Generator<int> breadthFirstVisit(int start) const;

            /**
            * @brief Yields node indices that form the paths from the starts to the ends
            */
            template <Direction            Dir,
                      std::predicate<int>  Predicate,
                      CForwardRangeOf<int> RangeStart,
                      CForwardRangeOf<int> RangeEnd>
            Generator<int> path(RangeStart const&    starts,
                                RangeEnd const&      ends,
                                std::map<int, bool>& visitedElements,
                                Predicate            edgeSelector) const;

            template <Direction Dir, CForwardRangeOf<int> RangeStart, CForwardRangeOf<int> RangeEnd>
            Generator<int> path(RangeStart const&    starts,
                                RangeEnd const&      ends,
                                std::map<int, bool>& visitedElements) const;

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

    }
}

#include "Hypergraph_impl.hpp"
