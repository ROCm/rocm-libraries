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

#define debug critical

namespace rocRoller
{
    namespace KernelGraph
    {
        using vec  = std::vector<int>;
        using vec2 = std::vector<vec>;
        using vec3 = std::vector<vec2>;

        vec2 makeChains(KernelGraph const& graph, std::vector<int> nodes)
        {
            std::ranges::sort(nodes, TopologicalCompare(graph));

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
                if(!graph.control.findEdge(currentChain.back(), nodes[i]))
                {
                    rv.push_back(std::move(currentChain));
                    currentChain.clear();
                }

                currentChain.push_back(nodes[i]);
            }

            rv.push_back(std::move(currentChain));

            return rv;
        }

        vec2 findMultiplyChains(KernelGraph const& graph)
        {
            auto isMultiply = [&](int idx) -> bool {
                return graph.control.get<ControlGraph::Multiply>(idx).has_value();
            };

            auto multiplies = graph.control.getNodes().filter(isMultiply).to<std::vector>();

            return makeChains(graph, std::move(multiplies));
        }

        vec2 findLoadLDSChains(KernelGraph const& graph)
        {
            auto isLoadLDSTile = [&](int idx) -> bool {
                return graph.control.get<ControlGraph::LoadLDSTile>(idx).has_value();
            };

            auto nodes = graph.control.getNodes().filter(isLoadLDSTile).to<std::vector>();

            for(auto& node : nodes)
            {
                while(auto bodyParent
                      = graph.control.getInputNodeIndices<ControlGraph::Body>(node).only())
                {
                    node = *bodyParent;
                }
            }

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
                                Log::critical("{} can't join {} due to different parents ({}/{})",
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
                                Log::critical("{} can't join {} due to defined order ({})",
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

            return rv;
        }

        vec3 identifyParallelMultiplyAndLDSChains(KernelGraph const& graph)
        {
            auto multiplyChains = findMultiplyChains(graph);
            auto ldsChains      = findLoadLDSChains(graph);

            Log::critical("Multiply chains: \n{}", showChains(multiplyChains));
            Log::critical("LDS chains: \n{}", showChains(ldsChains));

            return identifyParallelChains(graph, {std::move(multiplyChains), std::move(ldsChains)});
        }

        void clusterParallelChains(KernelGraph& graph, vec3 const& groups, int factor)
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

                std::optional<int> nop;

                do
                {
                    anyLeft = false;

                    if(nop)
                    {
                        for(size_t clusterIdx = 0; clusterIdx < group.size(); clusterIdx++)
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

                    for(size_t clusterIdx = 0; clusterIdx < group.size(); clusterIdx++)
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

                } while(anyLeft);
            }
        }

        KernelGraph ClusterParallelChains::apply(KernelGraph const& original)
        {
            auto rv = original;

            Log::critical("lksjflskdj");

            auto groups = identifyParallelMultiplyAndLDSChains(rv);
            Log::critical(showGroups(groups));

            clusterParallelChains(rv, groups, 4);

            return rv;
        }
    }
}
