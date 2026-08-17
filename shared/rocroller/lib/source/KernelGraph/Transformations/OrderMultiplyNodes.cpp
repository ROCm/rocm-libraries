// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/Transforms/OrderMultiplyNodes.hpp>
#include <rocRoller/KernelGraph/Transforms/OrderMultiplyNodes_detail.hpp>

#include <rocRoller/KernelGraph/NodeSchedulingUtils.hpp>

#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <rocRoller/Utilities/Logging.hpp>

namespace rocRoller::KernelGraph
{
    namespace OrderMultiplyNodesDetail
    {
        BestNodeOrder::BestNodeOrder(KernelGraph const& graph)
            : m_graph(graph)
            , m_tracer(graph)
        {
        }

        std::optional<bool> BestNodeOrder::existingOrder(int a, int b) const
        {
            if(a == b)
                return std::nullopt;

            auto existingOrder = m_graph.control.compareNodes(rocRoller::UpdateCache, a, b);

            if(existingOrder == ControlGraph::NodeOrdering::LeftFirst)
                return true;

            if(existingOrder == ControlGraph::NodeOrdering::RightFirst)
                return false;

            AssertFatal(existingOrder == ControlGraph::NodeOrdering::Undefined,
                        "These nodes should not contain each other",
                        ShowValue(a),
                        ShowValue(b));

            return std::nullopt;
        }

        std::vector<int> const& BestNodeOrder::aTagReplacements(int node) const
        {
            {
                auto iter = m_aTagReplacements.find(node);
                if(iter != m_aTagReplacements.end())
                    return iter->second;
            }

            auto getLoopOp = [&](int op) -> std::optional<int> {
                auto stack = controlStack(op, m_graph);

                for(auto parent : std::views::reverse(stack))
                {
                    if(m_graph.control.get<ControlGraph::ForLoopOp>(parent))
                        return parent;
                }

                return std::nullopt;
            };

            auto nodeLoop = getLoopOp(node);

            auto isLeft = [](ControlToCoordinateMapper::Connection const& c) {
                auto arg = getNaryArgument(c);
                return arg == NaryArgument::LHS || arg == NaryArgument::LHS_SCALE;
            };

            auto          connections_ = m_graph.mapper.getConnections(node);
            auto          connections  = connections_ | std::views::filter(isLeft);
            std::set<int> tags;

            for(auto const& connection : connections)
            {
                Log::debug("Op {}, Connection {}", node, toString(connection));
                tags.insert(connection.coordinate);
            }

            std::map<int, int> tagNodes;
            for(auto const& tag : tags)
            {
                auto tagTrace = m_tracer.coordinatesReadWrite(tag);

                for(auto const& rw : tagTrace)
                {
                    if(rw.rw == ControlFlowRWTracer::WRITE
                       && m_graph.control.get<ControlGraph::LoadLDSTile>(rw.control)
                       && getLoopOp(rw.control) == nodeLoop)
                    {
                        auto iter = tagNodes.find(tag);
                        if(iter == tagNodes.end()
                           || m_graph.control.compareNodes(UpdateCache, iter->second, rw.control)
                                  == ControlGraph::NodeOrdering::RightFirst)
                        {
                            tagNodes[tag] = rw.control;
                        }
                    }
                }
            }

            std::vector<int> rv;
            rv.reserve(tagNodes.size());

            for(auto const& [_, op] : tagNodes)
                rv.push_back(op);

            std::ranges::sort(rv, TopologicalCompare(m_graph));

            m_aTagReplacements[node] = std::move(rv);

            if(Log::getLogger()->should_log(LogLevel::Debug))
            {
                auto showEntry = [](std::pair<int, std::vector<int>> const& entry) {
                    auto const& [tag, deps] = entry;
                    return fmt::format("Multiply {}: ({})", tag, fmt::join(deps, ", "));
                };

                auto iter = m_aTagReplacements.find(node);
                Log::debug(showEntry(*iter));
            }

            return m_aTagReplacements[node];
        }

        std::optional<bool> BestNodeOrder::orderByATagReplacements(int a, int b) const
        {
            auto const& as = aTagReplacements(a);
            auto const& bs = aTagReplacements(b);

            auto aIter = as.begin(), bIter = bs.begin();
            for(; aIter != as.end() && bIter != bs.end(); ++aIter, ++bIter)
            {
                if(auto order = existingOrder(*aIter, *bIter))
                    return *order;
            }

            if(aIter != as.end())
                return true;
            if(bIter != bs.end())
                return false;

            return std::nullopt;
        }

        std::optional<int> BestNodeOrder::downstreamMemoryNode(int node) const
        {
            auto iter = m_downstreamMemoryNodes.find(node);
            if(iter != m_downstreamMemoryNodes.end())
                return iter->second;

            auto isMemoryNode = [&](int idx) -> bool {
                auto node = m_graph.control.get<ControlGraph::Operation>(idx);
                if(!node.has_value())
                    return false;

                auto _isMemoryNode = []<typename T>(T const& theNode) {
                    using namespace ControlGraph;
                    return CIsAnyOf<T,
                                    LoadLDSTile,
                                    LoadLinear,
                                    LoadVGPR,
                                    LoadSGPR,
                                    LoadTiled,
                                    StoreLDSTile,
                                    LoadTileDirect2LDS,
                                    LoadTiledTDMToLDS,
                                    StoreLinear,
                                    StoreTiled,
                                    StoreVGPR,
                                    StoreSGPR>;
                };

                return std::visit(_isMemoryNode, *node);
            };

            auto downstreamMemoryNode
                = m_graph.control.breadthFirstVisit(node, Graph::Direction::Downstream)
                      .filter(isMemoryNode)
                      .take(1)
                      .only();

            m_downstreamMemoryNodes[node] = downstreamMemoryNode;
            return downstreamMemoryNode;
        }

        std::optional<bool> BestNodeOrder::orderByDownstreamMemoryNodes(int a, int b) const
        {
            auto downstreamMemoryNodeA = downstreamMemoryNode(a);
            auto downstreamMemoryNodeB = downstreamMemoryNode(b);

            if(downstreamMemoryNodeA.has_value() && downstreamMemoryNodeB.has_value())
            {
                return existingOrder(*downstreamMemoryNodeA, *downstreamMemoryNodeB);
            }

            if(downstreamMemoryNodeA.has_value())
                return true;

            if(downstreamMemoryNodeB.has_value())
                return false;

            return std::nullopt;
        }

        std::vector<int> const& BestNodeOrder::reversedTagDependencies(int node) const
        {
            auto iter = m_reversedTagDependencies.find(node);
            if(iter != m_reversedTagDependencies.end())
                return iter->second;

            auto          allRecords = m_tracer.coordinatesReadWrite();
            std::set<int> coordinatesReadByNode;
            for(auto const& rec : allRecords)
            {
                if(rec.control == node
                   && (rec.rw == ControlFlowRWTracer::READ
                       || rec.rw == ControlFlowRWTracer::READWRITE))
                    coordinatesReadByNode.insert(rec.coordinate);
            }

            std::vector<int> nodesThatWriteThoseCoordinatesBeforeTheNode;

            for(auto const& rec : allRecords)
            {
                if(rec.rw != ControlFlowRWTracer::READ && rec.control != node
                   && coordinatesReadByNode.contains(rec.coordinate)
                   && m_graph.control.compareNodes(UpdateCache, rec.control, node)
                          == ControlGraph::NodeOrdering::LeftFirst)
                {
                    nodesThatWriteThoseCoordinatesBeforeTheNode.push_back(rec.control);
                }
            }

            AssertFatal(!nodesThatWriteThoseCoordinatesBeforeTheNode.empty());

            auto reverseTopologicalCompare = [&](int a, int b) {
                return a != b
                       && m_graph.control.compareNodes(UpdateCache, a, b)
                              == ControlGraph::NodeOrdering::RightFirst;
            };

            std::sort(nodesThatWriteThoseCoordinatesBeforeTheNode.begin(),
                      nodesThatWriteThoseCoordinatesBeforeTheNode.end(),
                      reverseTopologicalCompare);

            m_reversedTagDependencies[node]
                = std::move(nodesThatWriteThoseCoordinatesBeforeTheNode);
            return m_reversedTagDependencies[node];
        }

        std::optional<bool> BestNodeOrder::orderByLastTagDependencies(int a, int b) const
        {
            auto const& as = reversedTagDependencies(a);
            auto const& bs = reversedTagDependencies(b);

            auto aIter = as.begin(), bIter = bs.begin();
            for(; aIter != as.end() && bIter != bs.end(); ++aIter, ++bIter)
            {
                if(auto order = existingOrder(*aIter, *bIter))
                    return *order;
            }

            if(aIter != as.end())
                return false;
            if(bIter != bs.end())
                return true;

            return std::nullopt;
        }

        bool BestNodeOrder::operator()(int a, int b) const
        {
            if(auto order = orderByATagReplacements(a, b))
                return *order;

            if(auto order = orderByDownstreamMemoryNodes(a, b))
                return *order;

            if(auto order = orderByLastTagDependencies(a, b))
                return *order;

            return a < b;
        }

        void BestNodeOrder::populateCache(auto range) const
        {
            for(auto x : range)
            {
                aTagReplacements(x);
                downstreamMemoryNode(x);
                reversedTagDependencies(x);
            }
        }

        void BestNodeOrder::logTagData() const
        {
            auto showEntry = [](std::pair<int, std::vector<int>> const& entry) {
                auto const& [tag, deps] = entry;
                return fmt::format("Multiply {}: ({})\n", tag, fmt::join(deps, ", "));
            };

            auto entries = m_aTagReplacements | std::views::transform(showEntry);

            Log::debug("Tag info: {}", fmt::join(entries, ""));
        }

    }

    KernelGraph OrderMultiplyNodes::apply(KernelGraph const& original)
    {
        auto rv = original;

        {
            auto groupedMultiplyNodes = NodeScheduling::getGroupedNodes<ControlGraph::Multiply>(rv);
            for(auto& [parent, nodes] : groupedMultiplyNodes)
            {
                {
                    OrderMultiplyNodesDetail::BestNodeOrder comp(rv);
                    // Pre-populate cache because some STL algorithms take the comparator by value.
                    comp.populateCache(nodes);
                    NodeScheduling::orderNodes(rv, nodes, comp);
                    comp.logTagData();
                }

                for(size_t idx = 0; idx + 1 < nodes.size(); idx++)
                {
                    rv.control.chain<ControlGraph::Sequence>(nodes[idx], nodes[idx + 1]);
                }
            }
        }

        return rv;
    }

    ConstraintStatus NoUnorderedMultiplyNodes(const KernelGraph& k)
    {
        ConstraintStatus retval;

        auto groupedMultiplyNodes = NodeScheduling::getGroupedNodes<ControlGraph::Multiply>(k);

        std::set<int> ambiguousNodes;

        for(auto& [parent, nodes] : groupedMultiplyNodes)
        {
            for(size_t idx = 0; idx + 1 < nodes.size(); idx++)
            {
                if(k.control.compareNodes(UpdateCache, nodes[idx], nodes[idx + 1])
                   == ControlGraph::NodeOrdering::Undefined)
                {
                    ambiguousNodes.insert(nodes[idx]);
                    ambiguousNodes.insert(nodes[idx + 1]);
                }
            }
        }

        if(!ambiguousNodes.empty())
        {
            std::ostringstream msg;

            msg << "\\(";
            streamJoin(msg, ambiguousNodes, "|");
            msg << "\\)";

            retval.combine(false,
                           "Unordered multiply nodes found: " + ShowValue(ambiguousNodes)
                               + " Handy regex search string: " + msg.str());
        }

        return retval;
    }
}
