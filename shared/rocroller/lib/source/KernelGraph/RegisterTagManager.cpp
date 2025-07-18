/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>

namespace rocRoller
{
    std::string toString(RegisterExpressionAttributes const& t)
    {
        return concatenate("{\n",
                           ShowValue(t.dataType),
                           ShowValue(t.unitStride),
                           ShowValue(t.elementBlockSize),
                           ShowValue(t.elementBlockStride),
                           ShowValue(t.trLoadPairStride),
                           "}");
    }

    void RegisterTagManager::initialize(KernelGraph::KernelGraph const& kgraph)
    {
        namespace CT = KernelGraph::CoordinateGraph;

        auto isAlias = [&kgraph](int idx) {
            auto edge = kgraph.coordinates.getEdge(idx);
            if(!std::holds_alternative<CT::DataFlowEdge>(edge))
                return false;

            return std::holds_alternative<CT::Alias>(std::get<CT::DataFlowEdge>(edge));
        };

        for(auto edge : kgraph.coordinates.getEdges().filter(isAlias))
        {
            auto loc = kgraph.coordinates.getLocation(edge);
            AssertFatal(loc.incoming.size() == 1, ShowValue(loc.incoming.size()));
            AssertFatal(loc.outgoing.size() == 1, ShowValue(loc.outgoing.size()));

            auto src = loc.incoming[0];
            auto dst = loc.outgoing[0];

            addAlias(src, dst);
        }

        auto isIndexPredicate = [&kgraph](int idx) {
            auto edge = kgraph.coordinates.getEdge(idx);
            if(!std::holds_alternative<CT::DataFlowEdge>(edge))
                return false;

            return std::holds_alternative<CT::Index>(std::get<CT::DataFlowEdge>(edge));
        };

        for(auto edge : kgraph.coordinates.getEdges().filter(isIndexPredicate))
        {
            auto loc = kgraph.coordinates.getLocation(edge);
            AssertFatal(loc.incoming.size() == 1, ShowValue(loc.incoming.size()));
            AssertFatal(loc.outgoing.size() == 1, ShowValue(loc.outgoing.size()));

            auto src = loc.incoming[0];
            auto dst = loc.outgoing[0];

            auto indexEdge
                = std::get<CT::Index>(std::get<CT::DataFlowEdge>(kgraph.coordinates.getEdge(edge)));
            auto index = indexEdge.index;
            AssertFatal(index != -1, "index value for Index edge is not set");
            addIndex(src, dst, index);
        }

        auto isSegmentPredicate = [&kgraph](int idx) {
            auto edge = kgraph.coordinates.getEdge(idx);
            if(!std::holds_alternative<CT::DataFlowEdge>(edge))
                return false;

            return std::holds_alternative<CT::Segment>(std::get<CT::DataFlowEdge>(edge));
        };

        for(auto edge : kgraph.coordinates.getEdges().filter(isSegmentPredicate))
        {
            auto loc = kgraph.coordinates.getLocation(edge);
            AssertFatal(loc.incoming.size() == 1, ShowValue(loc.incoming.size()));
            AssertFatal(loc.outgoing.size() == 1, ShowValue(loc.outgoing.size()));

            auto src = loc.incoming[0];
            auto dst = loc.outgoing[0];

            auto segmentEdge = std::get<CT::Segment>(
                std::get<CT::DataFlowEdge>(kgraph.coordinates.getEdge(edge)));
            auto index = segmentEdge.index;
            AssertFatal(index != -1, "index value for Segment edge is not set");
            addSegment(src, dst, index);
        }
    }

    void RegisterTagManager::addAlias(int src, int dst)
    {
        AssertFatal(src > 0, ShowValue(src));
        AssertFatal(dst > 0, ShowValue(dst));

        AssertFatal(!m_aliases.contains(src));
        if(m_aliases.contains(dst))
        {
            AssertFatal(
                m_aliases.at(dst) == ALIAS_DEST, ShowValue(dst), ShowValue(m_aliases.at(dst)));
        }

        m_aliases[src] = dst;
        m_aliases[dst] = ALIAS_DEST;
    }

    void RegisterTagManager::addIndex(int src, int dst, int index)
    {
        AssertFatal(src > 0, ShowValue(src));
        AssertFatal(dst > 0, ShowValue(dst));

        AssertFatal(!m_indexes.contains(src));
        m_indexes[src] = {dst, index};
    }

    void RegisterTagManager::addSegment(int src, int dst, int index)
    {
        AssertFatal(src > 0, ShowValue(src));
        AssertFatal(dst > 0, ShowValue(dst));

        AssertFatal(!m_segments.contains(src));
        m_segments[src] = {dst, index};
    }
}
