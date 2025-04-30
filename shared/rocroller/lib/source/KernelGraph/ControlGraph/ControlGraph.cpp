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

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include <cmath>
#include <iomanip>

namespace rocRoller::KernelGraph::ControlGraph
{
    std::unordered_map<std::tuple<int, int>, NodeOrdering> const&
        ControlGraph::nodeOrderTable() const
    {
        populateOrderCache();
        return m_orderCache;
    }

    std::string ControlGraph::nodeOrderTableString(std::set<int> const& nodes) const
    {
        populateOrderCache();

        if(nodes.empty())
        {
            return "Empty order cache.\n";
        }

        std::ostringstream msg;

        int width = std::ceil(std::log10(static_cast<float>(*nodes.rbegin())));
        width     = std::max(width, 3);

        msg << std::setw(width) << " "
            << "\\";
        for(int n : nodes)
            msg << " " << std::setw(width) << n;

        for(int i : nodes)
        {
            msg << std::endl << std::setw(width) << i << "|";
            for(int j : nodes)
            {
                if(i == j)
                {
                    msg << " ";
                    auto oldFill = msg.fill('-');
                    msg << std::setw(width) << "-";
                    msg.fill(oldFill);
                }
                else
                {
                    msg << " " << std::setw(width) << abbrev(lookupOrder(CacheOnly, i, j));
                }
            }

            msg << " | " << std::setw(width) << i;
        }

        msg << std::endl
            << std::setw(width) << " "
            << "|";
        for(int n : nodes)
            msg << " " << std::setw(width) << n;

        msg << std::endl;

        return msg.str();
    }

    std::string ControlGraph::nodeOrderTableString() const
    {
        populateOrderCache();

        TIMER(t, "nodeOrderTable");

        std::set<int> nodes;

        for(auto const& pair : m_orderCache)
        {
            nodes.insert(std::get<0>(pair.first));
            nodes.insert(std::get<1>(pair.first));
        }

        return nodeOrderTableString(nodes);
    }

    void ControlGraph::clearCache(Graph::GraphModification modification)
    {
        Hypergraph<Operation, ControlEdge, false>::clearCache(modification);

        if(modification == Graph::GraphModification::AddElement
           && m_cacheStatus != CacheStatus::Invalid)
        {
            // If adding a new element and order is non-empty (partial or valid)
            m_cacheStatus = CacheStatus::Partial;
        }
        else
        {
            m_orderCache.clear();
            m_cacheStatus = CacheStatus::Invalid;
        }

        m_descendentCache.clear();
    }

    void ControlGraph::populateOrderCache() const
    {
        TIMER(t, "populateOrderCache");

        auto r = roots().to<std::set>();
        populateOrderCache(r);
        m_cacheStatus = CacheStatus::Valid;
    }

    template <CForwardRangeOf<int> Range>
    std::set<int> ControlGraph::populateOrderCache(Range const& startingNodes) const
    {
        std::set<int> rv;

        auto it = startingNodes.begin();
        if(it == startingNodes.end())
            return std::move(rv);

        rv = populateOrderCache(*it);

        for(it++; it != startingNodes.end(); it++)
        {
            auto nodes = populateOrderCache(*it);
            rv.insert(nodes.begin(), nodes.end());
        }

        return std::move(rv);
    }

    std::set<int> ControlGraph::populateOrderCache(int startingNode) const
    {
        auto ccEntry = m_descendentCache.find(startingNode);
        if(ccEntry != m_descendentCache.end())
            return ccEntry->second;

        auto addDescendents = [this](Generator<int> nodes) {
            auto theNodes = nodes.to<std::set>();

            auto descendents = populateOrderCache(theNodes);
            theNodes.insert(descendents.begin(), descendents.end());

            return theNodes;
        };

        auto initNodes     = addDescendents(getOutputNodeIndices<Initialize>(startingNode));
        auto bodyNodes     = addDescendents(getOutputNodeIndices<Body>(startingNode));
        auto elseNodes     = addDescendents(getOutputNodeIndices<Else>(startingNode));
        auto incNodes      = addDescendents(getOutputNodeIndices<ForLoopIncrement>(startingNode));
        auto sequenceNodes = addDescendents(getOutputNodeIndices<Sequence>(startingNode));

        // {init, body, else, inc} nodes are in the body of the current node
        writeOrderCache({startingNode}, initNodes, NodeOrdering::RightInBodyOfLeft);
        writeOrderCache({startingNode}, bodyNodes, NodeOrdering::RightInBodyOfLeft);
        writeOrderCache({startingNode}, elseNodes, NodeOrdering::RightInBodyOfLeft);
        writeOrderCache({startingNode}, incNodes, NodeOrdering::RightInBodyOfLeft);

        // Sequence connected nodes are after the current node
        writeOrderCache({startingNode}, sequenceNodes, NodeOrdering::LeftFirst);

        // {body, else, inc, sequence} are after init nodes
        writeOrderCache(initNodes, bodyNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, elseNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, incNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, sequenceNodes, NodeOrdering::LeftFirst);

        // {else, inc, sequence} are after body nodes
        writeOrderCache(bodyNodes, elseNodes, NodeOrdering::LeftFirst);
        writeOrderCache(bodyNodes, incNodes, NodeOrdering::LeftFirst);
        writeOrderCache(bodyNodes, sequenceNodes, NodeOrdering::LeftFirst);

        // {inc, sequence} are after else nodes
        writeOrderCache(elseNodes, incNodes, NodeOrdering::LeftFirst);
        writeOrderCache(elseNodes, sequenceNodes, NodeOrdering::LeftFirst);

        // sequence are after inc nodes.
        writeOrderCache(incNodes, sequenceNodes, NodeOrdering::LeftFirst);

        auto allNodes = std::move(sequenceNodes);
        allNodes.insert(bodyNodes.begin(), bodyNodes.end());
        allNodes.insert(elseNodes.begin(), elseNodes.end());
        allNodes.insert(incNodes.begin(), incNodes.end());
        allNodes.insert(initNodes.begin(), initNodes.end());

        m_descendentCache[startingNode] = allNodes;

        return std::move(allNodes);
    }

    template <CForwardRangeOf<int> ARange, CForwardRangeOf<int> BRange>
    void ControlGraph::writeOrderCache(ARange const& nodesA,
                                       BRange const& nodesB,
                                       NodeOrdering  order) const
    {
        for(int nodeA : nodesA)
            for(int nodeB : nodesB)
                writeOrderCache(nodeA, nodeB, order);
    }

    void ControlGraph::writeOrderCache(int nodeA, int nodeB, NodeOrdering order) const
    {
        if(nodeA > nodeB)
        {
            writeOrderCache(nodeB, nodeA, opposite(order));
        }
        else
        {
            auto it = m_orderCache.find({nodeA, nodeB});
            if(it != m_orderCache.end())
            {
                AssertFatal(it->second == order,
                            "Different kinds of orderings!",
                            ShowValue(nodeA),
                            ShowValue(nodeB),
                            ShowValue(it->second),
                            ShowValue(order));
            }

            m_orderCache[{nodeA, nodeB}] = order;
        }
    }

    NodeOrdering ControlGraph::lookupOrder(IgnoreCachePolicy const, int nodeA, int nodeB) const
    {
        TIMER(t, "ControlGraph::lookupOrder");

        using GD = Graph::Direction;
        std::unordered_set<int> visited_nodes;

        // Corresponding variant index of edge type:
        //   Sequence(0), Initialize(1), ForLoopIncrement(2), Body(3), Else(4)
        //
        // And order of edge type
        //   Initialize -> Body -> Else -> ForLoopIncrement -> Sequence.

        // Decide order of A and B when A is parent of B
        auto const getOrderIfParent = [&](int edge) {
            auto const index = getEdge(edge).index();
            AssertFatal(index <= 4, "Invalid edge");
            return index == 0 ? NodeOrdering::LeftFirst : NodeOrdering::RightInBodyOfLeft;
        };

        // Decide order of A and B when A and B are descendants of a node
        auto const getOrderOfDescendants = [&](int edgeATypeIndex, int edgeBTypeIndex) {
            AssertFatal(edgeATypeIndex != edgeBTypeIndex, "edgeA and edgeB should not be the same");
            AssertFatal(edgeATypeIndex <= 4 && edgeBTypeIndex <= 4, "Invalid edge");

            switch(edgeATypeIndex)
            {
            case 0: // edgeA is Sequence
                return opposite(NodeOrdering::LeftFirst);
            case 1: // edgeA is Initialize
                return NodeOrdering::LeftFirst;
            case 2: // edgeA is ForLoopIncrement
                return edgeBTypeIndex == 0 ? NodeOrdering::LeftFirst
                                           : opposite(NodeOrdering::LeftFirst);
            case 3: // edgeA is Body
                return edgeBTypeIndex == 1 ? opposite(NodeOrdering::LeftFirst)
                                           : NodeOrdering::LeftFirst;
            case 4: // edgeA is Else
                return edgeBTypeIndex == 1 || edgeBTypeIndex == 3
                           ? opposite(NodeOrdering::LeftFirst)
                           : NodeOrdering::LeftFirst;
            default:
                return NodeOrdering::Undefined;
            }
        };

        // {key, value} = {node index, edge type (index of variant)}
        std::unordered_map<int, int> A_ancestors;
        std::vector<int>             stk{nodeA};

        // Traverse upstream from A to collect all ancestors of A
        while(!stk.empty())
        {
            auto node = stk.back();
            stk.pop_back();

            for(auto edge : getNeighbours<GD::Upstream>(node))
            {
                for(auto parent : getNeighbours<GD::Upstream>(edge))
                {
                    if(parent == nodeB)
                        return opposite(getOrderIfParent(edge));

                    A_ancestors.insert({parent, getEdge(edge).index()});

                    if(!visited_nodes.contains(parent))
                    {
                        visited_nodes.insert(parent);
                        stk.push_back(parent);
                    }
                }
            }
        }

        AssertFatal(stk.empty());
        stk.push_back(nodeB);
        visited_nodes.clear();

        // Traverse upstream from B to find common ancestors to decide order
        while(!stk.empty())
        {
            auto node = stk.back();
            stk.pop_back();

            for(auto edge : getNeighbours<GD::Upstream>(node))
            {
                for(auto parent : getNeighbours<GD::Upstream>(edge))
                {
                    if(parent == nodeA)
                        return getOrderIfParent(edge);

                    if(A_ancestors.contains(parent))
                    {
                        // If this is a common ancestor, compare the types of both edges to
                        // know the order
                        auto const edgeBTypeIndex = getEdge(edge).index();
                        if(A_ancestors.at(parent) != edgeBTypeIndex)
                        {
                            return getOrderOfDescendants(A_ancestors.at(parent), edgeBTypeIndex);
                        }
                    }

                    if(!visited_nodes.contains(parent))
                    {
                        visited_nodes.insert(parent);
                        stk.push_back(parent);
                    }
                }
            }
        }

        return NodeOrdering::Undefined;
    }

    static void validateNodes(ControlGraph const& control, int nodeA, int nodeB)
    {
        AssertFatal(nodeA != nodeB, ShowValue(nodeA));
        AssertFatal(control.getElementType(nodeA) == Graph::ElementType::Node
                        && control.getElementType(nodeB) == Graph::ElementType::Node,
                    ShowValue(control.getElementType(nodeA)),
                    ShowValue(control.getElementType(nodeB)));
    }

    NodeOrdering ControlGraph::compareNodes(CacheOnlyPolicy const, int nodeA, int nodeB) const
    {
        AssertFatal(m_cacheStatus == CacheStatus::Valid);

        validateNodes(*this, nodeA, nodeB);
        return lookupOrder(CacheOnly, nodeA, nodeB);
    }

    NodeOrdering ControlGraph::compareNodes(UpdateCachePolicy const, int nodeA, int nodeB) const
    {
        if(m_cacheStatus != CacheStatus::Valid)
            populateOrderCache();

        return compareNodes(CacheOnly, nodeA, nodeB);
    }

    NodeOrdering
        ControlGraph::compareNodes(UseCacheIfAvailablePolicy const, int nodeA, int nodeB) const
    {
        validateNodes(*this, nodeA, nodeB);

        return m_cacheStatus == CacheStatus::Valid ? compareNodes(CacheOnly, nodeA, nodeB)
                                                   : compareNodes(IgnoreCache, nodeA, nodeB);
    }

    NodeOrdering ControlGraph::compareNodes(IgnoreCachePolicy const, int nodeA, int nodeB) const
    {
        validateNodes(*this, nodeA, nodeB);
        return lookupOrder(IgnoreCache, nodeA, nodeB);
    }
}
