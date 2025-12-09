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

#include <rocRoller/KernelGraph/Transforms/ClusterParallelChains.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <rocRoller/Graph/GraphUtilities.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>

namespace rocRoller
{
    /**
      *         ForLoop
      *
      *      v body
      *   Multiply
      *      v
      *   Multiply
      *      v
      *   Multiply
      *      v
      *   Multiply
      *      v
      *
      *
      *
      */

    namespace KernelGraph
    {
        using vec  = std::vector<int>;
        using vec2 = std::vector<vec>;
        using vec3 = std::vector<vec2>;

        vec2 makeChains(KernelGraph const& graph, std::vector<int> nodes)
        {
            std::ranges::sort(nodes, TopologicalCompare(graph));
            Log::debug("makeChains({})", ShowValue(nodes));

            auto isBarrier = [&](int idx) -> bool {
                return graph.control.get<ControlGraph::Barrier>(idx).has_value();
            };

            for(int i = 0; i + 1 < nodes.size(); i++)
            {
                auto order = graph.control.compareNodes(UpdateCache, nodes[i], nodes[i + 1]);
                AssertFatal(order == ControlGraph::NodeOrdering::LeftFirst,
                            ShowValue(order),
                            ShowValue(nodes[i]),
                            ShowValue(nodes[i + 1]));
            }

            vec2 rv;

            if(nodes.empty())
                return rv;

            std::vector<int> currentChain;

            currentChain.push_back(nodes[0]);

            for(int i = 1; i < nodes.size(); i++)
            {
                // The current chain can continue if
                // - There is a direct sequence edge between the last element and the current one, or
                // - There is a Barrier node directly between them.

                if(!graph.control.findEdge(currentChain.back(), nodes[i]))
                {
                    // auto barrier
                    //     = graph.control
                    //           .getOutputNodeIndices<ControlGraph::Sequence>(currentChain.back())
                    //           .filter(isBarrier)
                    //           .only();

                    // if(barrier
                    //    && !graph.control.getOutputNodeIndices<ControlGraph::Sequence>(*barrier)
                    //            .filter([&nodes, &i](int x) { return x == nodes[i]; })
                    //            .empty())
                    // {
                    // }
                    // else
                    {
                        rv.push_back(std::move(currentChain));
                        currentChain.clear();
                    }
                }

                currentChain.push_back(nodes[i]);
            }

            rv.push_back(std::move(currentChain));

            return rv;
        }

        vec filterLastCoordinateReads(KernelGraph const& graph, vec const& chain)
        {
            ControlFlowRWTracer tracer(graph);

            std::unordered_map<int, int> lastUses;

            for(auto op : chain)
            {
                auto records = tracer.opReadWrite(op);
                for(auto rec : records)
                {
                    lastUses[rec.coordinate] = op;
                }
            }

            auto lastNodes = [&]() {
                auto value = [](auto const& x) { return x.second; };
                auto tmp   = lastUses | std::views::transform(value);
                return std::unordered_set(tmp.begin(), tmp.end());
            }();

            auto isLast = [&](int x) { return lastNodes.contains(x); };

            auto rv = chain | std::views::filter(isLast);
            return std::vector(rv.begin(), rv.end());
        }

        vec2 findMultiplyChains(KernelGraph const& graph)
        {
            auto isMultiply = [&](int idx) -> bool {
                return graph.control.get<ControlGraph::Multiply>(idx).has_value();
            };

            auto multiplies = graph.control.getNodes().filter(isMultiply).to<std::vector>();

            auto rv = makeChains(graph, std::move(multiplies));

            for(auto & chain: rv)
                chain = filterLastCoordinateReads(graph, chain);

            return rv;
        }

        void getImmediateBodyParents(KernelGraph const& graph, std::vector<int>& nodes)
        {
            auto isSetCoordinate = [&](int idx) -> bool {
                return graph.control.get<ControlGraph::SetCoordinate>(idx).has_value();
            };

            for(auto& node : nodes)
            {
                while(auto bodyParent = graph.control.getInputNodeIndices<ControlGraph::Body>(node)
                                            .filter(isSetCoordinate)
                                            .only())
                {
                    node = *bodyParent;
                }
            }
        }

        vec2 findLoadLDSChains(KernelGraph const& graph)
        {
            auto isLoadLDSTile = [&](int idx) -> bool {
                return graph.control.get<ControlGraph::LoadLDSTile>(idx).has_value();
            };

            auto nodes = graph.control.getNodes().filter(isLoadLDSTile).to<std::vector>();
            getImmediateBodyParents(graph, nodes);

            return makeChains(graph, std::move(nodes));
        }

        vec2 findLoadStoreChains(KernelGraph const& graph)
        {
            auto isLoadOrStoreTile = [&](int idx) -> bool {
                auto visitor = [](auto const& op) -> bool {
                    using T = std::decay_t<decltype(op)>;

                    return CIsAnyOf<T,
                                    ControlGraph::Barrier,
                                    ControlGraph::LoadTileDirect2LDS,
                                    ControlGraph::LoadTiled,
                                    ControlGraph::StoreLDSTile,
                                    ControlGraph::LoadLDSTile>;
                };

                auto node = graph.control.get<ControlGraph::Operation>(idx).value();
                return std::visit(visitor, node);
            };

            auto nodes = graph.control.getNodes().filter(isLoadOrStoreTile).to<std::vector>();
            getImmediateBodyParents(graph, nodes);

            return makeChains(graph, std::move(nodes));
        }

        void logChainTagTable(KernelGraph const& graph, vec chain)
        {
            ControlFlowRWTracer tracer(graph);

            std::vector<int>                              coords;
            std::map<int, DataType>                       coordDataTypes;
            std::map<int, ControlFlowRWTracer::ReadWrite> useTypes;
            std::map<int, int>                            lastUses;

            {
                std::set<int> coordSet;
                for(auto& op : chain)
                {
                    while(auto child
                          = graph.control.getOutputNodeIndices<ControlGraph::Body>(op).only())
                    {
                        op = *child;
                    }
                }

                std::unordered_set<int> nodes(chain.begin(), chain.end());
                for(auto const& rec : tracer.coordinatesReadWrite())
                {
                    if(nodes.contains(rec.control))
                        coordSet.insert(rec.coordinate);
                }

                for(auto op : chain)
                {
                    auto records = tracer.opReadWrite(op);
                    for(auto rec : records)
                    {
                        lastUses[rec.coordinate] = op;
                        auto iter                = useTypes.find(rec.coordinate);
                        if(iter == useTypes.end())
                            useTypes[rec.coordinate] = rec.rw;
                        else
                            iter->second = combine(iter->second, rec.rw);
                    }
                }

                for(auto const& rec : tracer.coordinatesReadWrite())
                {
                    if(rec.rw == ControlFlowRWTracer::WRITE && coordSet.contains(rec.coordinate)
                       && !coordDataTypes.contains(rec.coordinate))
                    {
                        auto node  = graph.control.getNode(rec.control);
                        auto dtype = getDataType(node);
                        if(dtype != DataType::None)
                            coordDataTypes[rec.coordinate] = dtype;
                    }
                }

                for(auto coord : coordSet)
                    if(!coordDataTypes.contains(coord))
                        coordDataTypes[coord] = DataType::None;

                auto coordOrder = [&](int a, int b) {
                    auto typeA = useTypes[a], typeB = useTypes[b];

                    if(typeA != typeB)
                        return typeA < typeB;

                    return TopologicalCompare(graph)(lastUses[a], lastUses[b]);
                };

                coords = std::vector(coordSet.begin(), coordSet.end());
                std::ranges::sort(coords, coordOrder);

                // for(auto it = coords.begin(); it != coords.end(); )
                // {
                //     if(coordDataTypes.contains(*it))
                //         ++it;
                //     else
                //         it = coords.erase(it);
                // }
            }

            auto value     = [](auto const& x) { return x.second; };
            auto lastNodes = [&]() {
                auto tmp = lastUses | std::views::transform(value);
                return std::unordered_set(tmp.begin(), tmp.end());
            }();

            auto isLast = [&](int x) { return lastNodes.contains(x); };
            auto lasts  = [&]() {
                auto tmp = chain | std::views::filter(isLast);
                return std::vector(tmp.begin(), tmp.end());
            }();

            auto formatElement = [](auto el) -> std::string { return fmt::format("{:^6}", el); };

            auto getDtype = [&](int coord) { return TypeAbbrev(coordDataTypes[coord]); };

            auto msg = fmt::format("|{}|{}",
                                   formatElement(""),
                                   fmt::join(coords | std::views::transform(formatElement), "|"));

            std::string line(msg.size(), '=');

            msg += fmt::format("\n{}\n", line);

            msg += fmt::format("|{}|{}\n{}\n",
                               formatElement(""),
                               fmt::join(coords | std::views::transform(getDtype)
                                             | std::views::transform(formatElement),
                                         "|"),
                               line);

            for(auto op : chain)
            {
                auto records    = tracer.opReadWrite(op);
                auto lookupNode = [&](int coord) -> std::string {
                    for(auto const& rec : records)
                    {
                        if(rec.coordinate == coord)
                        {
                            switch(rec.rw)
                            {
                            case ControlFlowRWTracer::READ:
                                return "V";
                            case ControlFlowRWTracer::WRITE:
                                return "^";
                            case ControlFlowRWTracer::READWRITE:
                                return "X";
                            default:
                                break;
                            }
                        }
                    }
                    return " ";
                };

                msg += fmt::format("|{}|{}\n",
                                   formatElement(op),
                                   fmt::join(coords | std::views::transform(lookupNode)
                                                 | std::views::transform(formatElement),
                                             "|"));
            }

            msg += fmt::format("Lasts: ({})({})\n", lasts.size(), fmt::join(lasts, ", "));

            Log::debug("\n{}", msg);
        }

        vec2 findLoadTiledChains(KernelGraph const& graph)
        {
            auto isLoadTiled = [&](int idx) -> bool {
                return graph.control.get<ControlGraph::LoadTiled>(idx).has_value();
            };

            auto nodes = graph.control.getNodes().filter(isLoadTiled).to<std::vector>();
            getImmediateBodyParents(graph, nodes);

            return makeChains(graph, std::move(nodes));
        }

        vec2 findD2LDSChains(KernelGraph const& graph)
        {
            auto isD2LDS = [&](int idx) -> bool {
                return graph.control.get<ControlGraph::LoadTileDirect2LDS>(idx).has_value();
            };

            auto nodes = graph.control.getNodes().filter(isD2LDS).to<std::vector>();
            getImmediateBodyParents(graph, nodes);

            return makeChains(graph, std::move(nodes));
        }

        std::string showChains(vec2 const& chains)
        {
            std::ostringstream msg;

            for(auto const& chain : chains)
            {
                msg << " - {";
                streamJoin(msg, chain, ", ");
                msg << "} (" << chain.size() << ")" << std::endl;
            }

            return msg.str();
        }

        std::string showGroups(vec3 const& groups)
        {
            std::string rv;

            for(auto const& group : groups)
            {
                rv += fmt::format("-=-=-=-= Group: =-=-=-=-\n{}\n", showChains(group));
            }

            rv += fmt::format("{} groups\n", groups.size());

            return rv;
        }

        vec3 identifyParallelChains(KernelGraph const& graph, vec3 groups)
        {
            vec3 rv;
            if(groups.empty())
                return rv;

            // rv.push_back(std::move(groups.back()));
            // groups.pop_back();

            while(!groups.empty())
            {
                auto myGroup = std::move(groups.back());
                groups.pop_back();

                bool hasJoined = false;

                for(auto& myChain : myGroup)
                {
                    AssertFatal(!myChain.empty());
                    auto myNode  = myChain.front();
                    auto myStack = controlStack(myNode, graph);
                    myStack.pop_back();
                    AssertFatal(!myStack.empty());

                    for(auto& aGroup : rv)
                    {
                        bool canJoin = true;
                        for(auto const& aChain : aGroup)
                        {
                            AssertFatal(!aChain.empty());

                            auto aNode = aChain.front();

                            auto aStack = controlStack(aNode, graph);
                            aStack.pop_back();
                            AssertFatal(!aStack.empty());

                            if(myStack.back() != aStack.back())
                            {
                                Log::debug("{} can't join {} due to different parents ({}/{})",
                                           myNode,
                                           aNode,
                                           myStack.back(),
                                           aStack.back());
                                canJoin = false;
                                break;
                            }

                            auto order = graph.control.compareNodes(UpdateCache, myNode, aNode);
                            if(order != ControlGraph::NodeOrdering::Undefined)
                            {
                                Log::debug("{} can't join {} due to defined order ({})",
                                           myNode,
                                           aNode,
                                           toString(order));
                                canJoin = false;
                                break;
                            }
                        }
                        if(canJoin)
                        {
                            aGroup.push_back(std::move(myChain));
                            hasJoined = true;
                            break;
                        }
                    }

                    if(!hasJoined)
                    {
                        rv.push_back({std::move(myChain)});
                    }
                }
            }

            for(auto iter = rv.begin(); iter != rv.end();)
            {
                if(iter->size() < 2)
                    iter = rv.erase(iter);
                else
                    ++iter;
            }

            for(auto const& group : rv)
            {
                for(auto const& chain : group)
                {
                    logChainTagTable(graph, chain);
                }
            }

            return rv;
        }

        vec3 identifyParallelMultiplyAndLDSChains(KernelGraph const& graph)
        {
            auto multiplyChains = findMultiplyChains(graph);
            auto ldsChains      = findLoadLDSChains(graph);
            // auto loadChains     = findLoadTiledChains(graph);

            Log::debug("Multiply chains: \n{}", showChains(multiplyChains));
            Log::debug("LDS chains: \n{}", showChains(ldsChains));
            // Log::debug("LoadTiled chains: \n{}", showChains(loadChains));

            return identifyParallelChains(graph, {std::move(multiplyChains), std::move(ldsChains)});
        }

        vec3 identifyParallelMultiplyD2LDSAndLDSChains(KernelGraph const& graph)
        {
            auto multiplyChains = findMultiplyChains(graph);
            auto ldsChains      = findLoadLDSChains(graph);
            auto d2Chains = findD2LDSChains(graph);
            // auto loadChains     = findLoadTiledChains(graph);

            Log::debug("Multiply chains: \n{}", showChains(multiplyChains));
            Log::debug("LDS chains: \n{}", showChains(ldsChains));
            Log::debug("D2LDS chains: \n{}", showChains(d2Chains));
            // Log::debug("LoadTiled chains: \n{}", showChains(loadChains));

            return identifyParallelChains(graph, {std::move(multiplyChains), std::move(ldsChains), std::move(d2Chains)});
        }

        vec3 identifyParallelMultiplyAndLoadStoreChains(KernelGraph const& graph)
        {
            auto multiplyChains  = findMultiplyChains(graph);
            auto loadStoreChains = findLoadStoreChains(graph);
            // auto ldsChains      = findLoadLDSChains(graph);

            Log::debug("Multiply chains: \n{}", showChains(multiplyChains));
            Log::debug("D2LDS chains: \n{}", showChains(loadStoreChains));
            // Log::debug("LDS chains: \n{}", showChains(ldsChains));
            // Log::debug("LoadTiled chains: \n{}", showChains(loadChains));

            return identifyParallelChains(graph,
                                          {std::move(multiplyChains), std::move(loadStoreChains)});
        }

        vec3 identifyParallelMultiplyAndD2LDSChains(KernelGraph const& graph)
        {
            auto multiplyChains = findMultiplyChains(graph);
            auto d2ldsChains    = findD2LDSChains(graph);
            // auto ldsChains      = findLoadLDSChains(graph);

            Log::debug("Multiply chains: \n{}", showChains(multiplyChains));
            Log::debug("D2LDS chains: \n{}", showChains(d2ldsChains));
            // Log::debug("LDS chains: \n{}", showChains(ldsChains));
            // Log::debug("LoadTiled chains: \n{}", showChains(loadChains));

            return identifyParallelChains(graph,
                                          {std::move(multiplyChains), std::move(d2ldsChains)});
        }

        /**
          *
          * groups[x]:    each x is a group of nodes that should be clustered
          * groups[x][y]: each y is a different chain of nodes that should be clustered with the other ys.
          * groups[x][y][z]: each z is a node within the chain.
          *
          */

        void clusterParallelChainsLTR(KernelGraph& graph, vec3 const& groups, int factor)
        {

            for(auto const& group : groups)
            {
                AssertFatal(group.size() > 1, ShowValue(group.size()));
                auto sizes = group | std::views::transform([](auto const& c) { return c.size(); });
                auto groupGCD = std::reduce(
                    sizes.begin(), sizes.end(), *sizes.begin(), std::gcd<size_t, size_t>);

                auto tmp = sizes | std::views::transform([factor, groupGCD](size_t size) {
                               return (size * factor) / groupGCD;
                           });

                std::vector clusterSizes(tmp.begin(), tmp.end());

                std::vector<size_t> idxs(group.size(), 0);

                bool anyLeft = false;

                std::vector<int> upstreamNodes;
                // std::optional<int> nop;

                do
                {
                    anyLeft = false;

                    if(!upstreamNodes.empty())
                    {
                        AssertFatal(upstreamNodes.size() == group.size(),
                                    ShowValue(upstreamNodes.size()),
                                    ShowValue(group.size()));
                        for(size_t clusterIdx = 0; clusterIdx < group.size(); clusterIdx++)
                        {
                            for(size_t clusterIdx2 = clusterIdx + 1; clusterIdx2 < group.size();
                                clusterIdx++)
                            {
                                // for(auto upstreamNode : upstreamNodes)
                                for(size_t clusterIdx2 = clusterIdx + 1; clusterIdx2 < group.size();
                                    clusterIdx++)
                                    if(idxs[clusterIdx2] < group[clusterIdx2].size())
                                    {
                                        auto upstreamNode = upstreamNodes[clusterIdx];
                                        if(upstreamNode > 0)
                                        {
                                            graph.control.chain<ControlGraph::Sequence>(
                                                upstreamNode,
                                                group[clusterIdx2][idxs[clusterIdx2]]);
                                        }

                                        anyLeft = true;
                                    }
                            }
                        }
                        upstreamNodes.clear();

                        if(!anyLeft)
                            break;
                    }

                    anyLeft = false;

                    // nop = graph.control.addElement(ControlGraph::NOP());

                    for(size_t clusterIdx = 0; clusterIdx < group.size(); clusterIdx++)
                    {
                        auto endIdx  = idxs[clusterIdx] + clusterSizes[clusterIdx];
                        auto lastIdx = endIdx - 1;

                        if(lastIdx < group[clusterIdx].size())
                        {
                            upstreamNodes.push_back(group[clusterIdx][lastIdx]);

                            anyLeft = true;
                        }
                        else
                        {
                            upstreamNodes.push_back(-1);
                        }

                        idxs[clusterIdx] = endIdx;
                    }

                } while(anyLeft);
            }
        }

        vec3 fixupGroups(vec3 groups)
        {
            for(auto& group : groups)
            {
                vec fixed(group[1].begin() + 2, group[1].end());
                group[1] = fixed;
            }

            return groups;
        }

        void distributeParallelChainsUpward(KernelGraph& graph, vec3 groups, int factor)
        {
            for(auto const& group : groups)
            {
                AssertFatal(group.size() > 1, ShowValue(group.size()));

                auto theSizes
                    = group | std::views::transform([](auto const& c) { return c.size(); });
                std::vector<int> sizes(theSizes.begin(), theSizes.end());
                auto             maxSize = std::ranges::max(sizes);

                std::vector<float> expectedNodes(group.size(), 0.f);
                std::vector<int>   seenNodes(group.size(), 0);

                auto done = [&]() {
                    for(int idx = 0; idx < group.size(); idx++)
                    {
                        if(seenNodes[idx] < sizes[idx])
                            return false;
                    }
                    return true;
                };

                for(int incr = 0; !done(); ++incr)
                {
                    for(int idx = group.size() - 1; idx >= 0; --idx)
                    {
                        expectedNodes[idx] = static_cast<float>(incr * sizes[idx]) / maxSize;
                        auto floor         = static_cast<int>(expectedNodes[idx]);
                        if(floor > seenNodes[idx])
                        {
                            if(idx + 1 < group.size() && seenNodes.back() < sizes.back()
                               && seenNodes[idx] < sizes[idx])
                            {
                                Log::debug("back[{}] -> group[{}][{}]",
                                           seenNodes.back(),
                                           idx,
                                           seenNodes.at(idx));
                                graph.control.chain<ControlGraph::Sequence>(
                                    group.back().at(seenNodes.back()),
                                    group[idx].at(seenNodes[idx]));
                            }

                            seenNodes[idx] = floor;
                        }
                    }

                    Log::debug("exp: ({}), seen: ({}) / ({})",
                               fmt::join(expectedNodes, ", "),
                               fmt::join(seenNodes, ", "),
                               fmt::join(sizes, ", "));
                }
            }
        }

        void clusterParallelChainsUpward(KernelGraph& graph, vec3 groups, int factor)
        {
            // groups = fixupGroups(groups);

            for(auto const& group : groups)
            {
                AssertFatal(group.size() > 1, ShowValue(group.size()));
                auto sizes = group | std::views::transform([](auto const& c) { return c.size(); });
                auto groupGCD = std::reduce(
                    sizes.begin(), sizes.end(), *sizes.begin(), std::gcd<size_t, size_t>);

                auto tmp = sizes | std::views::transform([factor, groupGCD](size_t size) {
                               return (size * factor) / groupGCD;
                           });

                std::vector clusterSizes(tmp.begin(), tmp.end());
                Log::debug(
                    "Cluster: ({}) -> ({})", fmt::join(sizes, ", "), fmt::join(clusterSizes, ", "));

                std::vector<int> firstIdxs(group.size(), 0);
                std::vector<int> lastIdxs(group.size(), 0);
                std::vector<int> nextIdxs(group.size(), 0);

                bool any = false;
                do
                {
                    any = false;

                    for(int clusterIdx = 0; clusterIdx < group.size(); clusterIdx++)
                    {
                        nextIdxs[clusterIdx] = firstIdxs[clusterIdx] + clusterSizes[clusterIdx];
                        lastIdxs[clusterIdx] = nextIdxs[clusterIdx] - 1;
                    }

                    for(int clusterIdx = 0; clusterIdx + 1 < group.size(); clusterIdx++)
                    {
                        if(lastIdxs[clusterIdx + 1] < group[clusterIdx + 1].size()
                           && nextIdxs[clusterIdx] < group[clusterIdx].size())
                        {
                            any         = true;
                            auto before = group[clusterIdx + 1][lastIdxs[clusterIdx + 1]];
                            auto after  = group[clusterIdx][nextIdxs[clusterIdx]];

                            graph.control.chain<ControlGraph::Sequence>(before, after);
                        }
                    }

                    std::swap(firstIdxs, nextIdxs);

                } while(any);
            }
        }

        void clusterParallelChainsNop(KernelGraph& graph,
                                      vec3 const&  groups,
                                      int          factor,
                                      vec          slip = {})
        {
            for(auto const& group : groups)
            {
                AssertFatal(group.size() > 1, ShowValue(group.size()));
                auto sizes = group | std::views::transform([](auto const& c) { return c.size(); });
                auto groupGCD = std::reduce(
                    sizes.begin(), sizes.end(), *sizes.begin(), std::gcd<size_t, size_t>);

                auto tmp = sizes | std::views::transform([factor, groupGCD](size_t size) {
                               return (size * factor) / groupGCD;
                           });

                std::vector clusterSizes(tmp.begin(), tmp.end());

                std::vector<int> idxs(group.size(), 0);
                if(!slip.empty())
                    idxs = slip;

                // bool first = true;
                bool anyLeft = false;

                std::optional<int> nop;

                do
                {
                    anyLeft = false;

                    if(nop)
                    {
                        for(int clusterIdx = 0; clusterIdx < group.size(); clusterIdx++)
                        {
                            if(idxs[clusterIdx] < group[clusterIdx].size())
                            {
                                graph.control.chain<ControlGraph::Sequence>(
                                    *nop, group[clusterIdx][idxs[clusterIdx]]);

                                anyLeft = true;
                            }
                        }

                        if(!anyLeft)
                            break;
                    }

                    anyLeft = false;

                    nop = graph.control.addElement(ControlGraph::NOP());

                    for(int clusterIdx = 0; clusterIdx < group.size(); clusterIdx++)
                    {
                        auto endIdx  = idxs[clusterIdx] + clusterSizes[clusterIdx];
                        auto lastIdx = endIdx - 1;

                        if(lastIdx < group[clusterIdx].size())
                        {
                            graph.control.chain<ControlGraph::Sequence>(group[clusterIdx][lastIdx],
                                                                        *nop);

                            anyLeft = true;
                        }

                        idxs[clusterIdx] = endIdx;
                    }
                    // first = false;

                } while(anyLeft);
            }
        }

        KernelGraph ClusterParallelChains::apply(KernelGraph const& original)
        {
            auto rv = original;

            if(false)
            {
                auto groups = identifyParallelMultiplyAndLoadStoreChains(rv);
                Log::debug(showGroups(groups));

                // clusterParallelChainsNop(rv, groups, 1, {0, 8});
                distributeParallelChainsUpward(rv, groups, 1);
            }
            else if(false)
            {
                {
                    auto groups = identifyParallelMultiplyAndD2LDSChains(rv);
                    Log::debug(showGroups(groups));

                    clusterParallelChainsNop(rv, groups, 1);
                }

                removeRedundantSequenceEdges(rv);

                {
                    auto groups = identifyParallelMultiplyAndLDSChains(rv);
                    Log::debug(showGroups(groups));

                    // clusterParallelChainsNop(rv, groups, 1, {0, 8});
                    clusterParallelChainsUpward(rv, groups, 1);
                }
            }
            else
            {
                {
                    auto groups = identifyParallelMultiplyAndLDSChains(rv);
                    Log::debug(showGroups(groups));

                    // clusterParallelChainsNop(rv, groups, 1, {0, 8});
                    distributeParallelChainsUpward(rv, groups, 1);
                }
            }

            return rv;
        }
    }
}
