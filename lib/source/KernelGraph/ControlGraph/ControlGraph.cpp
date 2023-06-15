
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include <cmath>

namespace rocRoller::KernelGraph::ControlGraph
{

    std::string ControlGraph::nodeOrderTable() const
    {
        populateOrderCache();

        TIMER(t, "nodeOrderTable");

        std::ostringstream msg;

        int prevR = -1;

        std::set<int> nodes;

        for(auto const& pair : m_orderCache)
        {
            nodes.insert(pair.first.first);
            nodes.insert(pair.first.second);
        }

        if(nodes.empty())
            msg << "Empty order cache." << std::endl;

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
                    msg << " " << std::setw(width) << abbrev(lookupOrderCache(i, j));
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

    void ControlGraph::clearCache()
    {
        Hypergraph<Operation, ControlEdge, false>::clearCache();

        m_orderCache.clear();
        m_descendentCache.clear();
    }

    void ControlGraph::populateOrderCache() const
    {
        TIMER(t, "populateOrderCache");

        auto r = roots().to<std::set>();
        populateOrderCache(r);
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
        auto incNodes      = addDescendents(getOutputNodeIndices<ForLoopIncrement>(startingNode));
        auto sequenceNodes = addDescendents(getOutputNodeIndices<Sequence>(startingNode));

        // {init, body, inc} nodes are in the body of the current node
        writeOrderCache({startingNode}, initNodes, NodeOrdering::RightInBodyOfLeft);
        writeOrderCache({startingNode}, bodyNodes, NodeOrdering::RightInBodyOfLeft);
        writeOrderCache({startingNode}, incNodes, NodeOrdering::RightInBodyOfLeft);

        // Sequence connected nodes are after the current node
        writeOrderCache({startingNode}, sequenceNodes, NodeOrdering::LeftFirst);

        // {body, inc, sequence} are after init nodes
        writeOrderCache(initNodes, bodyNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, incNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, sequenceNodes, NodeOrdering::LeftFirst);

        // {inc, sequence} are after body nodes
        writeOrderCache(bodyNodes, incNodes, NodeOrdering::LeftFirst);
        writeOrderCache(bodyNodes, sequenceNodes, NodeOrdering::LeftFirst);

        // sequence are after inc nodes.
        writeOrderCache(incNodes, sequenceNodes, NodeOrdering::LeftFirst);

        auto allNodes = std::move(sequenceNodes);
        allNodes.insert(bodyNodes.begin(), bodyNodes.end());
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

}
