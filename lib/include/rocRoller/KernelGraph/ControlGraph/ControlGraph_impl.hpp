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

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>

namespace rocRoller::KernelGraph::ControlGraph
{
    inline std::string toString(NodeOrdering n)
    {
        switch(n)
        {
        case NodeOrdering::LeftFirst:
            return "LeftFirst";
        case NodeOrdering::LeftInBodyOfRight:
            return "LeftInBodyOfRight";
        case NodeOrdering::Undefined:
            return "Undefined";
        case NodeOrdering::RightInBodyOfLeft:
            return "RightInBodyOfLeft";
        case NodeOrdering::RightFirst:
            return "RightFirst";

        case NodeOrdering::Count:
        default:
            break;
        }
        Throw<FatalError>("Invalid NodeOrdering");
    }

    inline std::ostream& operator<<(std::ostream& stream, NodeOrdering n)
    {
        return stream << toString(n);
    }

    inline std::string toString(CacheStatus c)
    {
        switch(c)
        {
        case CacheStatus::Invalid:
            return "Invalid";
        case CacheStatus::Partial:
            return "Partial";
        case CacheStatus::Valid:
            return "Valid";
        default:
            break;
        }
        Throw<FatalError>("Invalid CacheStatus");
    }

    inline std::ostream& operator<<(std::ostream& stream, CacheStatus c)
    {
        return stream << toString(c);
    }

    static_assert(CCountedEnum<NodeOrdering>);

    inline std::string abbrev(NodeOrdering n)
    {
        switch(n)
        {
        case NodeOrdering::LeftFirst:
            return " LF";
        case NodeOrdering::LeftInBodyOfRight:
            return "LIB";
        case NodeOrdering::Undefined:
            return "und";
        case NodeOrdering::RightInBodyOfLeft:
            return "RIB";
        case NodeOrdering::RightFirst:
            return " RF";

        case NodeOrdering::Count:
        default:
            break;
        }
        Throw<FatalError>("Invalid NodeOrdering");
    }

    inline NodeOrdering opposite(NodeOrdering order)
    {
        switch(order)
        {
        case NodeOrdering::LeftFirst:
            return NodeOrdering::RightFirst;
        case NodeOrdering::LeftInBodyOfRight:
            return NodeOrdering::RightInBodyOfLeft;
        case NodeOrdering::Undefined:
            return NodeOrdering::Undefined;
        case NodeOrdering::RightInBodyOfLeft:
            return NodeOrdering::LeftInBodyOfRight;
        case NodeOrdering::RightFirst:
            return NodeOrdering::LeftFirst;

        case NodeOrdering::Count:
        default:
            break;
        }
        Throw<FatalError>("Invalid NodeOrdering");
    }

    inline void ControlGraph::checkOrderCache() const
    {
        if(m_orderCache.empty())
            populateOrderCache();
    }

    inline NodeOrdering ControlGraph::lookupOrderCache(int nodeA, int nodeB) const
    {
        if(nodeA > nodeB)
            return opposite(lookupOrderCache(nodeB, nodeA));

        auto iter = m_orderCache.find({nodeA, nodeB});
        if(iter == m_orderCache.end())
            return NodeOrdering::Undefined;

        return iter->second;
    }

    inline NodeOrdering ControlGraph::compareNodes(int nodeA, int nodeB) const
    {
        AssertFatal(nodeA != nodeB, ShowValue(nodeA));
        AssertFatal(getElementType(nodeA) == Graph::ElementType::Node
                        && getElementType(nodeB) == Graph::ElementType::Node,
                    ShowValue(getElementType(nodeA)),
                    ShowValue(getElementType(nodeB)));

        auto const order = lookupOrderCache(nodeA, nodeB);
        if(order != NodeOrdering::Undefined || m_cacheStatus == CacheStatus::Valid)
            return order;

        populateOrderCache();
        return lookupOrderCache(nodeA, nodeB);
    }

    inline Generator<int> ControlGraph::nodesAfter(int node) const
    {
        populateOrderCache();

        for(auto const& pair : m_orderCache)
        {
            if(std::get<0>(pair.first) == node && pair.second == NodeOrdering::LeftFirst)
                co_yield std::get<1>(pair.first);
            else if(std::get<1>(pair.first) == node && pair.second == NodeOrdering::RightFirst)
                co_yield std::get<0>(pair.first);
        }
    }

    inline Generator<int> ControlGraph::nodesBefore(int node) const
    {
        populateOrderCache();

        for(auto const& pair : m_orderCache)
        {
            if(std::get<0>(pair.first) == node && pair.second == NodeOrdering::RightFirst)
                co_yield std::get<1>(pair.first);
            else if(std::get<1>(pair.first) == node && pair.second == NodeOrdering::LeftFirst)
                co_yield std::get<0>(pair.first);
        }
    }

    inline Generator<int> ControlGraph::nodesInBody(int node) const
    {
        populateOrderCache();

        for(auto const& pair : m_orderCache)
        {
            if(std::get<0>(pair.first) == node && pair.second == NodeOrdering::RightInBodyOfLeft)
                co_yield std::get<1>(pair.first);
            else if(std::get<1>(pair.first) == node
                    && pair.second == NodeOrdering::LeftInBodyOfRight)
                co_yield std::get<0>(pair.first);
        }
    }

    inline Generator<int> ControlGraph::nodesContaining(int node) const
    {
        populateOrderCache();

        for(auto const& pair : m_orderCache)
        {
            if(std::get<1>(pair.first) == node && pair.second == NodeOrdering::RightInBodyOfLeft)
                co_yield std::get<0>(pair.first);
            else if(std::get<0>(pair.first) == node
                    && pair.second == NodeOrdering::LeftInBodyOfRight)
                co_yield std::get<1>(pair.first);
        }
    }

    template <typename T>
    requires(std::constructible_from<Operation, T>) inline std::set<
        std::pair<int, int>> ControlGraph::ambiguousNodes() const
    {
        std::set<std::pair<int, int>> badNodes;

        auto memNodes = getNodes<T>().template to<std::set>();
        for(auto iter = memNodes.begin(); iter != memNodes.end(); iter++)
        {
            std::set otherNodes(std::next(iter), memNodes.end());
            for(auto node : otherNodes)
            {
                if(compareNodes(*iter, node) == NodeOrdering::Undefined)
                {
                    badNodes.insert(std::make_pair(*iter, node));
                }
            }
        }
        return badNodes;
    }

    template <typename T>
    requires(std::constructible_from<ControlGraph::Element,
                                     T>) inline std::optional<T> ControlGraph::get(int tag) const
    {
        auto x = getElement(tag);
        if constexpr(CIsAnyOf<T, Operation, ControlEdge>)
        {
            if(std::holds_alternative<T>(x))
            {
                return std::get<T>(x);
            }
        }
        else
        {
            if constexpr(std::constructible_from<ControlEdge, T>)
            {
                if(std::holds_alternative<ControlEdge>(x))
                {
                    if(std::holds_alternative<T>(std::get<ControlEdge>(x)))
                    {
                        return std::get<T>(std::get<ControlEdge>(x));
                    }
                }
            }
            if constexpr(std::constructible_from<Operation, T>)
            {
                if(std::holds_alternative<Operation>(x))
                {
                    if(std::holds_alternative<T>(std::get<Operation>(x)))
                    {
                        return std::get<T>(std::get<Operation>(x));
                    }
                }
            }
        }
        return {};
    }

    template <CControlEdge Edge, std::convertible_to<int>... Nodes>
    void ControlGraph::chain(int a, int b, Nodes... remaining)
    {
        addElement(Edge(), {a}, {b});

        if constexpr(sizeof...(remaining) > 0)
            chain<Edge>(b, remaining...);
    }

    template <typename T>
    requires(std::constructible_from<Operation, T>) inline bool isOperation(auto const& x)
    {
        if(std::holds_alternative<Operation>(x))
        {
            if(std::holds_alternative<T>(std::get<Operation>(x)))
                return true;
        }

        return false;
    }

    template <typename T>
    requires(std::constructible_from<ControlEdge, T>) inline bool isEdge(auto const& x)
    {
        if(std::holds_alternative<ControlEdge>(x))
        {
            if(std::holds_alternative<T>(std::get<ControlEdge>(x)))
                return true;
        }

        return false;
    }

    inline std::string name(ControlGraph::Element const& el)
    {
        return ControlGraph::ElementName(el);
    }

    template <CForwardRangeOf<int> Range>
    inline void ControlGraph::orderMemoryNodes(Range const& aControlStack,
                                               Range const& bControlStack,
                                               bool         ordered)
    {
        int src  = -1;
        int dest = -1;
        for(int i = 0; (i < aControlStack.size()) && (i < bControlStack.size()); ++i)
        {
            if(aControlStack.at(i) != bControlStack.at(i))
            {
                auto setCoordA = get<SetCoordinate>(aControlStack.at(i));
                auto setCoordB = get<SetCoordinate>(bControlStack.at(i));
                if(ordered)
                {
                    src  = aControlStack.at(i);
                    dest = bControlStack.at(i);
                }
                else if(setCoordA && setCoordB
                        && evaluationTimes(setCoordA->value)[Expression::EvaluationTime::Translate]
                        && evaluationTimes(setCoordB->value)[Expression::EvaluationTime::Translate])
                {
                    src  = aControlStack.at(i);
                    dest = bControlStack.at(i);
                    if(getUnsignedInt(evaluate(setCoordB->value))
                       < getUnsignedInt(evaluate(setCoordA->value)))
                    {
                        src  = bControlStack.at(i);
                        dest = aControlStack.at(i);
                    }
                }
                else
                {
                    src  = std::min(aControlStack.at(i), bControlStack.at(i));
                    dest = std::max(aControlStack.at(i), bControlStack.at(i));
                }
                break;
            }
        }
        if(src == -1 || dest == -1)
        {
            int aIndex = aControlStack.size() - 1;
            int bIndex = bControlStack.size() - 1;
            if(aControlStack.size() > bControlStack.size())
            {
                aIndex = bIndex;
            }
            else
            {
                bIndex = aIndex;
            }
            src  = std::min(aControlStack.at(aIndex), bControlStack.at(bIndex));
            dest = std::max(aControlStack.at(aIndex), bControlStack.at(bIndex));
        }

        addElement(Sequence(), {src}, {dest});
    }
}
