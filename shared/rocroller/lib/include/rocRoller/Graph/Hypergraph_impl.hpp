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

#include <algorithm>
#include <compare>
#include <map>
#include <ranges>
#include <unordered_set>
#include <variant>
#include <vector>

#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/Utilities/Comparison.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    namespace Graph
    {
        inline std::string toString(ElementType e)
        {
            switch(e)
            {
            case ElementType::Node:
                return "Node";
            case ElementType::Edge:
                return "Edge";
            case ElementType::Count:
                break;
            }
            throw std::runtime_error("Invalid ElementType");
        }

        inline std::ostream& operator<<(std::ostream& stream, ElementType const& e)
        {
            return stream << toString(e);
        }

        inline std::string toString(Direction d)
        {
            switch(d)
            {
            case Direction::Upstream:
                return "Upstream";
            case Direction::Downstream:
                return "Downstream";
            case Direction::Count:
                break;
            }
            throw std::runtime_error("Invalid Direction");
        }

        constexpr inline Direction opposite(Direction d)
        {
            return d == Direction::Downstream ? Direction::Upstream : Direction::Downstream;
        }

        inline std::string toString(GraphModification g)
        {
            switch(g)
            {
            case GraphModification::DeleteElement:
                return "DeleteElement";
            case GraphModification::AddElement:
                return "AddElement";
            case GraphModification::SetElement:
                return "SetElement";
            case GraphModification::Count:
                return "Count";
            }

            Throw<FatalError>("Invalid GraphModification ", static_cast<int>(g));
        }

        template <typename Node, typename Edge, bool Hyper>
        std::string Hypergraph<Node, Edge, Hyper>::ElementName(
            typename Hypergraph<Node, Edge, Hyper>::Element const& el)
        {
            return std::visit(rocRoller::overloaded{[](Node const&) { return "Node"; },
                                                    [](Edge const&) { return "Edge"; }},
                              el);
        }

        template <typename Node, typename Edge, bool Hyper>
        constexpr inline bool
            Hypergraph<Node, Edge, Hyper>::Location::operator==(Location const& rhs) const
        {
            // clang-format off
            return LexicographicCompare(
                        index, rhs.index,
                        incoming, rhs.incoming,
                        outgoing, rhs.outgoing) == 0;
            // clang-format on
        }

        template <typename Node, typename Edge, bool Hyper>
        bool Hypergraph<Node, Edge, Hyper>::exists(int index) const
        {
            return m_elements.count(index) == 1;
        }

        template <typename Node, typename Edge, bool Hyper>
        ElementType Hypergraph<Node, Edge, Hyper>::getElementType(int index) const
        {
            return getElementType(m_elements.at(index));
        }

        template <typename Node, typename Edge, bool Hyper>
        ElementType Hypergraph<Node, Edge, Hyper>::getElementType(Element const& e) const
        {
            if(holds_alternative<Node>(e))
                return ElementType::Node;
            return ElementType::Edge;
        }

        inline ElementType getConnectingType(ElementType t)
        {
            return t == ElementType::Node ? ElementType::Edge : ElementType::Node;
        }

        template <typename Node, typename Edge, bool Hyper>
        void Hypergraph<Node, Edge, Hyper>::clearCache(GraphModification)
        {
            m_locationCache.clear();
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        int Hypergraph<Node, Edge, Hyper>::addElement(T&& element)
        {
            int index = m_nextIndex++;
            AssertFatal(m_elements.find(index) == m_elements.end());
            m_elements.emplace(index, std::forward<T>(element));
            clearCache(GraphModification::AddElement);
            return index;
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T, typename... Ts>
        auto Hypergraph<Node, Edge, Hyper>::addElements(T&& element, Ts&&... rest)
        {
            auto myValue = addElement(std::forward<T>(element));
            return std::tuple{myValue, addElement(std::forward<Ts>(rest))...};
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        void Hypergraph<Node, Edge, Hyper>::setElement(int index, T&& element)
        {
            AssertFatal(m_elements.find(index) != m_elements.end());

            m_elements[index] = std::forward<T>(element);
            clearCache(GraphModification::SetElement);
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        int Hypergraph<Node, Edge, Hyper>::addElement(T&&                        element,
                                                      std::initializer_list<int> inputs,
                                                      std::initializer_list<int> outputs)
        {
            return addElement<T, std::initializer_list<int>, std::initializer_list<int>>(
                std::forward<T>(element), inputs, outputs);
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T, CForwardRangeOf<int> T_Inputs, CForwardRangeOf<int> T_Outputs>
        int Hypergraph<Node, Edge, Hyper>::addElement(T&&              element,
                                                      T_Inputs const&  inputs,
                                                      T_Outputs const& outputs)
        {
            int index = m_nextIndex;
            m_nextIndex++;
            addElement(index, std::forward<T>(element), inputs, outputs);

            return index;
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T, CForwardRangeOf<int> T_Inputs, CForwardRangeOf<int> T_Outputs>
        void Hypergraph<Node, Edge, Hyper>::addElement(int              index,
                                                       T&&              element,
                                                       T_Inputs const&  inputs,
                                                       T_Outputs const& outputs)
        {
            AssertFatal(m_elements.find(index) == m_elements.end());

            auto elementType    = getElementType(element);
            auto connectingType = getConnectingType(elementType);

            for(int cIdx : inputs)
            {
                AssertFatal(getElementType(cIdx) == connectingType);
            }
            for(int cIdx : outputs)
            {
                AssertFatal(getElementType(cIdx) == connectingType);
            }

            clearCache(GraphModification::AddElement);

            m_elements.emplace(index, std::forward<T>(element));

            if(elementType == ElementType::Edge)
            {
                int incidentOrder = 0;
                for(int src : inputs)
                {
                    m_incidence.insert({src, index, incidentOrder++});
                }
                AssertFatal(Hyper || incidentOrder <= 1);
                incidentOrder = 0;
                for(int dst : outputs)
                {
                    m_incidence.insert({index, dst, incidentOrder++});
                }
                AssertFatal(Hyper || incidentOrder <= 1);
            }
            else
            {
                auto const& bySrc = m_incidence.template get<BySrc>();
                for(int src : inputs)
                {
                    int incidentOrder = 0;

                    auto lastSrcIter = bySrc.lower_bound(std::make_tuple(src + 1, 0));
                    if(lastSrcIter != bySrc.begin())
                        lastSrcIter--;
                    if(lastSrcIter != bySrc.end() && lastSrcIter->src == src)
                        incidentOrder = lastSrcIter->edgeOrder + 1;

                    AssertFatal(Hyper || incidentOrder == 0);
                    m_incidence.insert({src, index, incidentOrder});
                }

                auto const& byDst = m_incidence.template get<ByDst>();
                for(int dst : outputs)
                {
                    int incidentOrder = 0;

                    auto lastDstIter = byDst.lower_bound(std::make_tuple(dst + 1, 0));
                    if(lastDstIter != byDst.begin())
                        lastDstIter--;
                    if(lastDstIter != byDst.end() && lastDstIter->dst == dst)
                        incidentOrder = lastDstIter->edgeOrder + 1;

                    AssertFatal(Hyper || incidentOrder == 0);
                    m_incidence.insert({index, dst, incidentOrder});
                }
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        void Hypergraph<Node, Edge, Hyper>::deleteElement(int index)
        {
            auto elem = getElement(index);

            clearCache(GraphModification::DeleteElement);

            auto& src = m_incidence.template get<BySrc>();

            for(auto iter = src.lower_bound(std::make_tuple(index, 0));
                iter != src.end() && iter->src == index;)
            {
                iter = src.erase(iter);
            }

            auto& dst = m_incidence.template get<ByDst>();

            for(auto iter = dst.lower_bound(std::make_tuple(index, 0));
                iter != dst.end() && iter->dst == index;)
            {
                iter = dst.erase(iter);
            }

            m_elements.erase(index);
        }

        // delete edge between the inputs and outputs with exact match
        // deletes the first match found (duplicates not deleted)
        template <typename Node, typename Edge, bool Hyper>
        template <CForwardRangeOf<int>        T_Inputs,
                  CForwardRangeOf<int>        T_Outputs,
                  std::predicate<Edge const&> T_Predicate>
        void Hypergraph<Node, Edge, Hyper>::deleteElement(T_Inputs const&  inputs,
                                                          T_Outputs const& outputs,
                                                          T_Predicate      edgePredicate)
        {
            AssertFatal(!inputs.empty() && !outputs.empty());

            clearCache(GraphModification::DeleteElement);

            for(int cIdx : inputs)
            {
                AssertFatal(getElementType(cIdx) == ElementType::Node, "Requires node handles");
            }
            for(int cIdx : outputs)
            {
                AssertFatal(getElementType(cIdx) == ElementType::Node, "Requires node handles");
            }

            auto outgoing_edges
                = getNeighbours<Graph::Direction::Downstream>(inputs[0]).template to<std::vector>();
            auto match = false;
            for(auto e : outgoing_edges)
            {
                auto elem = getElement(e);
                if(!edgePredicate(std::get<Edge>(elem)))
                    continue;

                match = true;

                auto srcs = getNeighbours<Graph::Direction::Upstream>(e)
                                .template to<std::unordered_set>();
                if(srcs.size() != inputs.size())
                {
                    match = false;
                    continue;
                }
                for(auto src : inputs)
                {
                    if(srcs.find(src) == srcs.end())
                    {
                        match = false;
                        break;
                    }
                }

                if(match)
                {
                    auto dsts = getNeighbours<Graph::Direction::Downstream>(e)
                                    .template to<std::unordered_set>();
                    if(dsts.size() != outputs.size())
                    {
                        match = false;
                        continue;
                    }
                    for(auto dst : outputs)
                    {
                        if(dsts.find(dst) == dsts.end())
                        {
                            match = false;
                            break;
                        }
                    }
                }

                if(match)
                {
                    deleteElement(e);
                    return;
                }
            }
            AssertFatal(match, "edge to delete : match not found");
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T, CForwardRangeOf<int> T_Inputs, CForwardRangeOf<int> T_Outputs>
        requires(std::constructible_from<Edge, T>) void Hypergraph<Node, Edge, Hyper>::
            deleteElement(T_Inputs const& inputs, T_Outputs const& outputs)
        {
            return deleteElement(
                inputs, outputs, [](Edge const& edge) { return std::holds_alternative<T>(edge); });
        }

        template <typename Node, typename Edge, bool Hyper>
        size_t Hypergraph<Node, Edge, Hyper>::getIncidenceSize() const
        {
            return m_incidence.size();
        }

        template <typename Node, typename Edge, bool Hyper>
        size_t Hypergraph<Node, Edge, Hyper>::getElementCount() const
        {
            return m_elements.size();
        }

        template <typename Node, typename Edge, bool Hyper>
        auto Hypergraph<Node, Edge, Hyper>::getElement(int index) const -> Element const&
        {
            AssertFatal(index >= 0 && index < m_nextIndex, "Element not found ", ShowValue(index));
            return m_elements.at(index);
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        T Hypergraph<Node, Edge, Hyper>::getNode(int index) const
        {
            static_assert(std::constructible_from<Node, T>);
            auto const& node = std::get<Node>(getElement(index));
            if constexpr(std::same_as<Node, T>)
            {
                return node;
            }
            else
            {
                return std::get<T>(node);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        T Hypergraph<Node, Edge, Hyper>::getEdge(int index) const
        {
            static_assert(std::constructible_from<Edge, T>);
            auto const& edge = std::get<Edge>(getElement(index));
            if constexpr(std::same_as<Edge, T>)
            {
                return edge;
            }
            else
            {
                return std::get<T>(edge);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        auto Hypergraph<Node, Edge, Hyper>::getLocation(int index) const -> Location
        {
            {
                auto iter = m_locationCache.find(index);
                if(iter != m_locationCache.end())
                    return iter->second;
            }

            Location rv;
            rv.index   = index;
            rv.element = m_elements.at(index);

            {
                // Incoming: Find the src for incidents whose dst is our node.
                auto const& incomingLookup = m_incidence.template get<ByDst>();

                auto incomingBegin = incomingLookup.lower_bound(std::make_tuple(index, 0));
                auto incomingEnd   = incomingLookup.lower_bound(std::make_tuple(index + 1, 0));

                std::transform(incomingBegin,
                               incomingEnd,
                               std::back_inserter(rv.incoming),
                               [](auto const& inc) { return inc.src; });
            }

            {
                // Outgoing: Find the dst for incidents whose src is our node.
                auto const& outgoingLookup = m_incidence.template get<BySrc>();

                auto outgoingBegin = outgoingLookup.lower_bound(std::make_tuple(index, 0));
                auto outgoingEnd   = outgoingLookup.lower_bound(std::make_tuple(index + 1, 0));

                std::transform(outgoingBegin,
                               outgoingEnd,
                               std::back_inserter(rv.outgoing),
                               [](auto const& inc) { return inc.dst; });
            }

            m_locationCache[index] = rv;

            return rv;
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::roots() const
        {
            auto const& lookup = m_incidence.template get<ByDst>();

            for(auto const& pair : m_elements)
            {
                int  index = pair.first;
                auto iter  = lookup.lower_bound(std::make_tuple(index, 0));
                if(iter == lookup.end() || iter->dst != index)
                    co_yield index;
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::leaves() const
        {
            auto const& lookup = m_incidence.template get<BySrc>();

            for(auto const& pair : m_elements)
            {
                int  index = pair.first;
                auto iter  = lookup.lower_bound(std::make_tuple(index, 0));
                if(iter == lookup.end() || iter->src != index)
                    co_yield index;
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::childNodes(int parent) const
        {
            if(getElementType(parent) == ElementType::Node)
            {
                std::set<int> visited;
                for(auto const& edgeIndex : getNeighbours<Direction::Downstream>(parent))
                {
                    auto neighbours = getNeighbours<Direction::Downstream>(edgeIndex);
                    for(auto const& neighbour : neighbours)
                    {
                        if(!visited.contains(neighbour))
                        {
                            visited.insert(neighbour);
                            co_yield neighbour;
                        }
                    }
                }
            }
            else
            {
                co_yield getNeighbours<Direction::Downstream>(parent);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::parentNodes(int child) const
        {
            if(getElementType(child) == ElementType::Node)
            {
                std::set<int> visited;
                for(auto const& edgeIndex : getNeighbours<Direction::Upstream>(child))
                {
                    auto neighbours = getNeighbours<Direction::Upstream>(edgeIndex);
                    for(auto const& neighbour : neighbours)
                    {
                        if(!visited.contains(neighbour))
                        {
                            visited.insert(neighbour);
                            co_yield neighbour;
                        }
                    }
                }
            }
            else
            {
                co_yield getNeighbours<Direction::Upstream>(child);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <CForwardRangeOf<int> Range>
        Generator<int> Hypergraph<Node, Edge, Hyper>::depthFirstVisit(Range const& starts,
                                                                      Direction    dir) const
        {
            std::unordered_set<int> visitedNodes;
            if(dir == Direction::Downstream)
            {
                for(int index : starts)
                    co_yield depthFirstVisit<Direction::Downstream>(index, visitedNodes);
            }
            else
            {
                for(int index : starts)
                    co_yield depthFirstVisit<Direction::Upstream>(index, visitedNodes);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <CForwardRangeOf<int> Range, std::predicate<int> Predicate>
        Generator<int> Hypergraph<Node, Edge, Hyper>::depthFirstVisit(Range const& starts,
                                                                      Predicate    edgePredicate,
                                                                      Direction    dir) const
        {
            std::unordered_set<int> visitedNodes;
            if(dir == Direction::Downstream)
            {
                for(int index : starts)
                    co_yield depthFirstVisit<Direction::Downstream>(
                        index, edgePredicate, visitedNodes);
            }
            else
            {
                for(int index : starts)
                    co_yield depthFirstVisit<Direction::Upstream>(
                        index, edgePredicate, visitedNodes);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <std::predicate<int> Predicate>
        Generator<int> Hypergraph<Node, Edge, Hyper>::depthFirstVisit(int       start,
                                                                      Predicate edgePredicate,
                                                                      Direction dir) const
        {
            std::unordered_set<int> visitedNodes;
            if(dir == Direction::Downstream)
            {
                co_yield depthFirstVisit<Direction::Downstream>(start, edgePredicate, visitedNodes);
            }
            else
            {
                co_yield depthFirstVisit<Direction::Upstream>(start, edgePredicate, visitedNodes);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::depthFirstVisit(int       start,
                                                                      Direction dir) const
        {
            std::initializer_list<int> starts{start};
            co_yield depthFirstVisit(starts, dir);
        }

        template <typename Node, typename Edge, bool Hyper>
        template <std::predicate<int> Predicate>
        Generator<int> Hypergraph<Node, Edge, Hyper>::findNodes(int       start,
                                                                Predicate nodeSelector,
                                                                Direction dir) const
        {
            co_yield filter(nodeSelector, depthFirstVisit(start, dir));
        }

        template <typename Node, typename Edge, bool Hyper>
        template <CForwardRangeOf<int> Range, std::predicate<int> Predicate>
        Generator<int> Hypergraph<Node, Edge, Hyper>::findNodes(Range const& starts,
                                                                Predicate    nodeSelector,
                                                                Direction    dir) const
        {
            co_yield filter(nodeSelector, depthFirstVisit(starts, dir));
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::allElements() const
        {
            for(auto const& pair : m_elements)
                co_yield pair.first;
        }

        template <typename Node, typename Edge, bool Hyper>
        template <std::predicate<int> Predicate>
        Generator<int> Hypergraph<Node, Edge, Hyper>::findElements(Predicate nodeSelector) const
        {
            co_yield filter(nodeSelector, allElements());
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::breadthFirstVisit(int       start,
                                                                        Direction dir) const
        {
            if(dir == Direction::Downstream)
                co_yield breadthFirstVisitDownstream(start);
            else
                co_yield breadthFirstVisitUpstream(start);
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::breadthFirstVisitDownstream(int start) const
        {
            std::unordered_set<int> visitedNodes;

            visitedNodes.insert(start);

            co_yield start;

            auto chain = [](auto& iters, auto end) -> Generator<int> {
                for(int i = 0; i < iters.size(); i++)
                {
                    int src = iters[i].second;
                    for(auto iter = iters[i].first; iter != end && iter->src == src; iter++)
                    {
                        co_yield iter->dst;
                    }
                }
            };

            auto const& lookup = m_incidence.template get<BySrc>();

            auto startIter = lookup.lower_bound(std::make_tuple(start, 0));
            auto iters     = std::vector{std::make_pair(startIter, start)};

            for(int node : chain(iters, lookup.end()))
            {
                if(visitedNodes.count(node))
                    continue;

                visitedNodes.insert(node);
                co_yield node;

                iters.emplace_back(lookup.lower_bound(std::make_tuple(node, 0)), node);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::breadthFirstVisitUpstream(int start) const
        {
            std::unordered_set<int> visitedNodes;

            visitedNodes.insert(start);

            co_yield start;

            auto chain = [](auto& iters, auto end) -> Generator<int> {
                for(int i = 0; i < iters.size(); i++)
                {
                    int dst = iters[i].second;
                    for(auto iter = iters[i].first; iter != end && iter->dst == dst; iter++)
                    {
                        co_yield iter->src;
                    }
                }
            };

            auto const& lookup = m_incidence.template get<ByDst>();

            auto startIter = lookup.lower_bound(std::make_tuple(start, 0));
            auto iters     = std::vector{std::make_pair(startIter, start)};

            for(int node : chain(iters, lookup.end()))
            {
                if(visitedNodes.count(node))
                    continue;

                visitedNodes.insert(node);
                co_yield node;

                iters.emplace_back(lookup.lower_bound(std::make_tuple(node, 0)), node);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <Direction Dir>
        Generator<int> Hypergraph<Node, Edge, Hyper>::depthFirstVisit(
            int start, std::unordered_set<int>& visitedNodes) const
        {
            if(visitedNodes.count(start))
                co_return;

            visitedNodes.insert(start);

            co_yield start;

            for(auto const element : getNeighbours<Dir>(start))
            {
                co_yield depthFirstVisit<Dir>(element, visitedNodes);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <Direction Dir, std::predicate<int> Predicate>
        Generator<int> Hypergraph<Node, Edge, Hyper>::depthFirstVisit(
            int start, Predicate edgePredicate, std::unordered_set<int>& visitedElements) const
        {
            if(visitedElements.count(start))
                co_return;

            visitedElements.insert(start);

            co_yield start;

            for(auto const tag : getNeighbours<Dir>(start))
            {
                visitedElements.insert(tag);
                if(edgePredicate(tag))
                {
                    for(auto const child : getNeighbours<Dir>(tag))
                    {
                        co_yield depthFirstVisit<Dir>(child, edgePredicate, visitedElements);
                    }
                }
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <Direction Dir, CForwardRangeOf<int> RangeStart, CForwardRangeOf<int> RangeEnd>
        Generator<int> Hypergraph<Node, Edge, Hyper>::path(RangeStart const& starts,
                                                           RangeEnd const&   ends) const
        {
            auto truePred = [](int) { return true; };
            co_yield path<Dir>(starts, ends, truePred);
        }

        template <typename Node, typename Edge, bool Hyper>
        template <Direction            Dir,
                  CForwardRangeOf<int> RangeStart,
                  CForwardRangeOf<int> RangeEnd,
                  std::predicate<int>  Predicate>
        Generator<int> Hypergraph<Node, Edge, Hyper>::path(RangeStart const& starts,
                                                           RangeEnd const&   ends,
                                                           Predicate         edgeSelector) const
        {
            std::map<int, bool> visitedElements;
            co_yield path<Dir>(starts, ends, edgeSelector, visitedElements);
        }

        template <typename Node, typename Edge, bool Hyper>
        template <Direction            Dir,
                  CForwardRangeOf<int> RangeStart,
                  CForwardRangeOf<int> RangeEnd,
                  std::predicate<int>  Predicate>
        Generator<int>
            Hypergraph<Node, Edge, Hyper>::path(RangeStart const&    starts,
                                                RangeEnd const&      ends,
                                                Predicate            edgeSelector,
                                                std::map<int, bool>& visitedElements) const
        {
            constexpr Direction reverseDir = opposite(Dir);

            for(auto const end : ends)
            {
                if(visitedElements.contains(end))
                {
                    continue;
                }

                if(std::count(starts.begin(), starts.end(), end) > 0)
                {
                    visitedElements[end] = true;
                    co_yield end;
                    continue;
                }

                visitedElements[end] = false;

                std::vector<int> results;
                for(auto const nextElement : getNeighbours<reverseDir>(end))
                {
                    if(getElementType(nextElement) == ElementType::Edge
                       && !edgeSelector(nextElement))
                    {
                        continue;
                    }

                    std::vector<int> branchResults
                        = path<Dir>(
                              starts, std::vector<int>{nextElement}, edgeSelector, visitedElements)
                              .template to<std::vector>();
                    results.insert(results.end(), branchResults.begin(), branchResults.end());

                    bool satisfied = (getElementType(end) != ElementType::Edge
                                      || edgeSatisfied<reverseDir>(end, visitedElements));

                    visitedElements[end] = visitedElements[end] || visitedElements[nextElement];
                    visitedElements[end] = visitedElements[end] && satisfied;
                }

                if(visitedElements.at(end))
                {
                    for(int const result : results)
                    {
                        co_yield result;
                    }
                    co_yield end;
                }
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <Direction Dir>
        bool Hypergraph<Node, Edge, Hyper>::edgeSatisfied(
            int const edge, std::map<int, bool> const& visitedElements) const
        {
            for(auto const element : getNeighbours<Dir>(edge))
            {
                auto iter = visitedElements.find(element);
                if(iter == visitedElements.end() || !iter->second)
                    return false;
            }

            return true;
        }

        template <typename Node, typename Edge, bool Hyper>
        template <Direction Dir>
        Generator<int> Hypergraph<Node, Edge, Hyper>::getNeighbours(int const element) const
        {
            if constexpr(Dir == Direction::Downstream)
            {
                auto const& lookup = m_incidence.template get<BySrc>();

                for(auto iter = lookup.lower_bound(std::make_tuple(element, 0));
                    iter != lookup.end() && iter->src == element;
                    iter++)
                {
                    co_yield iter->dst;
                }
            }
            else
            {
                auto const& lookup = m_incidence.template get<ByDst>();

                for(auto iter = lookup.lower_bound(std::make_tuple(element, 0));
                    iter != lookup.end() && iter->dst == element;
                    iter++)
                {
                    co_yield iter->src;
                }
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        Generator<int> Hypergraph<Node, Edge, Hyper>::getNeighbours(int const element,
                                                                    Direction Dir) const
        {
            if(Dir == Direction::Downstream)
            {
                co_yield getNeighbours<Direction::Downstream>(element);
            }
            else
            {
                co_yield getNeighbours<Direction::Upstream>(element);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        inline Generator<int> Hypergraph<Node, Edge, Hyper>::topologicalSort() const
        {
            auto start = roots().template to<std::vector>();
            auto end   = leaves().template to<std::vector>();
            co_yield path<Graph::Direction::Downstream>(start, end);
        }

        template <typename Node, typename Edge, bool Hyper>
        inline Generator<int> Hypergraph<Node, Edge, Hyper>::reverseTopologicalSort() const
        {
            auto start = roots().template to<std::vector>();
            auto end   = leaves().template to<std::vector>();
            co_yield path<Graph::Direction::Upstream>(end, start);
        }

        template <typename Node, typename Edge, bool Hyper>
        std::string Hypergraph<Node, Edge, Hyper>::toDOT(std::string const& prefix,
                                                         bool               standalone) const
        {
            std::ostringstream msg;

            if(standalone)
                msg << "digraph {" << std::endl;

            for(auto const& pair : m_elements)
            {
                msg << '"' << prefix << pair.first << '"' << "[label=\"";
                if(getElementType(pair.second) == ElementType::Node)
                {
                    auto x = std::get<Node>(pair.second);
                    msg << toString(x) << "(" << pair.first << ")\"";
                }
                else
                {
                    auto x = std::get<Edge>(pair.second);
                    msg << toString(x) << "(" << pair.first << ")\",shape=box";
                }
                msg << "];" << std::endl;
            }

            auto const& container = m_incidence.template get<BySrc>();
            for(auto const& incident : container)
            {
                msg << '"' << prefix << incident.src << "\" -> \"" << prefix << incident.dst << '"'
                    << std::endl;
            }

            // Enforce left-to-right ordering for elements connected to an edge.
            for(auto const& pair : m_elements)
            {
                if(getElementType(pair.second) != ElementType::Edge)
                    continue;

                auto const& loc = getLocation(pair.first);

                if(loc.incoming.size() > 1)
                {
                    msg << "{\nrank=same\n";
                    bool first = true;
                    for(int idx : loc.incoming)
                    {
                        if(!first)
                            msg << "->";
                        msg << '"' << prefix << idx << '"';
                        first = false;
                    }
                    msg << "[style=invis]\nrankdir=LR\n}\n";
                }

                if(loc.outgoing.size() > 1)
                {
                    msg << "{\nrank=same\n";
                    bool first = true;
                    for(int idx : loc.outgoing)
                    {
                        if(!first)
                            msg << "->";
                        msg << '"' << prefix << idx << '"';
                        first = false;
                    }
                    msg << "[style=invis]\nrankdir=LR\n}\n";
                }
            }

            if(standalone)
                msg << "}" << std::endl;

            return msg.str();
        }

        template <typename Node, typename Edge, bool Hyper>
        template <std::predicate<Edge const&> Predicate>
        std::string Hypergraph<Node, Edge, Hyper>::toDOT(Predicate edgePredicate) const
        {
            std::ostringstream msg;

            std::string const prefix = "";

            msg << "digraph {" << std::endl;

            for(auto const& pair : m_elements)
            {
                if(getElementType(pair.second) == ElementType::Node)
                {
                    auto x = std::get<Node>(pair.second);
                    msg << '"' << prefix << pair.first << '"' << "[label=\"";
                    msg << toString(x) << "(" << pair.first << ")\"";
                    msg << "];" << std::endl;
                }
                else
                {
                    auto x = std::get<Edge>(pair.second);
                    if(edgePredicate(x))
                    {
                        msg << '"' << prefix << pair.first << '"' << "[label=\"";
                        msg << toString(x) << "(" << pair.first << ")\",shape=box";
                        msg << "];" << std::endl;
                    }
                }
            }

            for(auto const& pair : m_elements)
            {
                if(getElementType(pair.second) == ElementType::Edge)
                {
                    auto x = std::get<Edge>(pair.second);
                    if(edgePredicate(x))
                    {
                        for(auto y : getNeighbours<Direction::Upstream>(pair.first))
                        {
                            msg << '"' << prefix << y << "\" -> \"" << prefix << pair.first << '"'
                                << std::endl;
                        }
                        for(auto y : getNeighbours<Direction::Downstream>(pair.first))
                        {
                            msg << '"' << prefix << pair.first << "\" -> \"" << prefix << y << '"'
                                << std::endl;
                        }
                    }
                }
            }

            msg << "}" << std::endl;

            return msg.str();
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        requires(std::constructible_from<Node, T> || std::constructible_from<Edge, T>)
            Generator<int> Hypergraph<Node, Edge, Hyper>::getElements()
        const
        {
            for(auto const& elem : m_elements)
            {
                if constexpr(std::same_as<T, Node> || std::same_as<T, Edge>)
                {
                    if(std::holds_alternative<T>(elem.second))
                        co_yield elem.first;
                }
                else if constexpr(std::constructible_from<Node, T>)
                {
                    if(std::holds_alternative<Node>(elem.second))
                    {
                        auto const& node = std::get<Node>(elem.second);
                        if(std::holds_alternative<T>(node))
                            co_yield elem.first;
                    }
                }
                else if constexpr(std::constructible_from<Edge, T>)
                {
                    if(std::holds_alternative<Edge>(elem.second))
                    {
                        auto const& edge = std::get<Edge>(elem.second);
                        if(std::holds_alternative<T>(edge))
                            co_yield elem.first;
                    }
                }
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        requires(std::constructible_from<Node, T>)
            Generator<int> Hypergraph<Node, Edge, Hyper>::getNodes()
        const
        {
            co_yield getElements<T>();
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        requires(std::constructible_from<Edge, T>)
            Generator<int> Hypergraph<Node, Edge, Hyper>::getEdges()
        const
        {
            co_yield getElements<T>();
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T, Direction Dir>
        requires(std::constructible_from<Edge, T>)
            Generator<int> Hypergraph<Node, Edge, Hyper>::getConnectedNodeIndices(int const dst)
        const
        {
            if constexpr(std::same_as<Edge, T>)
            {
                auto truePredicate = [](auto const&) { return true; };
                co_yield getConnectedNodeIndices<Dir>(dst, truePredicate);
            }
            else
            {
                auto edgePredicate
                    = [](Edge const& edge) { return std::holds_alternative<T>(edge); };
                co_yield getConnectedNodeIndices<Dir>(dst, edgePredicate);
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <Direction Dir, std::predicate<Edge const&> Predicate>
        Generator<int>
            Hypergraph<Node, Edge, Hyper>::getConnectedNodeIndices(int const dst,
                                                                   Predicate edgePredicate) const
        {
            AssertFatal(getElementType(dst) == ElementType::Node, "Require a node handle");

            for(int const elem : getNeighbours<Dir>(dst))
            {
                if(edgePredicate(std::get<Edge>(getElement(elem))))
                {
                    co_yield getNeighbours<Dir>(elem);
                }
            }
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        requires(std::constructible_from<Edge, T>)
            Generator<int> Hypergraph<Node, Edge, Hyper>::getInputNodeIndices(int const dst)
        const
        {
            co_yield getConnectedNodeIndices<T, Direction::Upstream>(dst);
        }

        template <typename Node, typename Edge, bool Hyper>
        template <std::predicate<Edge const&> Predicate>
        Generator<int>
            Hypergraph<Node, Edge, Hyper>::getInputNodeIndices(int const dst,
                                                               Predicate edgePredicate) const
        {
            co_yield getConnectedNodeIndices<Direction::Upstream>(dst, edgePredicate);
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        requires(std::constructible_from<Edge, T>)
            Generator<int> Hypergraph<Node, Edge, Hyper>::getOutputNodeIndices(int const src)
        const
        {
            co_yield getConnectedNodeIndices<T, Direction::Downstream>(src);
        }

        template <typename Node, typename Edge, bool Hyper>
        template <std::predicate<Edge const&> Predicate>
        Generator<int>
            Hypergraph<Node, Edge, Hyper>::getOutputNodeIndices(int const src,
                                                                Predicate edgePredicate) const
        {
            co_yield getConnectedNodeIndices<Direction::Downstream>(src, edgePredicate);
        }

        template <typename Node, typename Edge, bool Hyper>
        template <typename T>
        requires(std::constructible_from<Edge, T>)
            std::set<int> Hypergraph<Node, Edge, Hyper>::followEdges(
                std::set<int> const& candidates)
        {
            // Nodes to be analyzed
            std::set<int> currentNodes = candidates;

            // Full set of connected nodes to be returned
            std::set<int> connectedNodes = candidates;

            auto numCandidates = connectedNodes.size();

            do
            {
                // Nodes which are found by this sweep
                std::set<int> foundNodes;

                numCandidates = connectedNodes.size();

                for(auto tag : currentNodes)
                {
                    auto outTags = getOutputNodeIndices<T>(tag);
                    foundNodes.insert(outTags.begin(), outTags.end());
                }

                connectedNodes.insert(foundNodes.begin(), foundNodes.end());
                currentNodes = std::move(foundNodes);
            } while(numCandidates != connectedNodes.size());

            return connectedNodes;
        }

        template <typename Node, typename Edge, bool Hyper>
        inline std::ostream& operator<<(std::ostream&                        stream,
                                        Hypergraph<Node, Edge, Hyper> const& graph)
        {
            return stream << graph.toDOT();
        }

        template <Graph::Direction Dir, typename Node, typename Edge, bool Hyper>
        Generator<int> reachableNodes(Graph::Hypergraph<Node, Edge, Hyper> const& graph,
                                      int                                         start,
                                      auto                                        nodePredicate,
                                      auto                                        edgePredicate,
                                      auto                                        destNodePredicate)
        {
            for(int nextNode : graph.template getConnectedNodeIndices<Dir>(start, edgePredicate))
            {
                auto const& node = graph.getNode(nextNode);
                if(destNodePredicate(node))
                    co_yield nextNode;

                if(nodePredicate(node))
                    co_yield reachableNodes<Dir>(
                        graph, nextNode, nodePredicate, edgePredicate, destNodePredicate);
            }
        }

    }
}
