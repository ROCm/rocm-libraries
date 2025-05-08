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

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddDirect2LDS.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
namespace rocRoller
{
    namespace KernelGraph
    {
        bool isChain(KernelGraph const& kgraph, int op1, int op2)
        {
            for(auto tag : kgraph.control.depthFirstVisit(op1, Graph::Direction::Downstream))
            {
                if(tag == op2)
                {
                    return true;
                }
            }
            return false;
        }

        bool isLeaf(KernelGraph const& kgraph, int tag)
        {
            auto children
                = kgraph.control.getNeighbours<Graph::Direction::Downstream>(tag).to<std::vector>();
            return children.empty();
        }

        std::vector<std::pair<int, int>> searchCandidates(KernelGraph const&   kgraph,
                                                          CommandParametersPtr params,
                                                          ContextPtr           context)
        {
            using namespace ControlGraph;
            using namespace CoordinateGraph;

            auto loadTiledNodes    = kgraph.control.getNodes<LoadTiled>().to<std::vector>();
            auto storeLDSTileNodes = kgraph.control.getNodes<StoreLDSTile>().to<std::vector>();
            std::vector<std::pair<int, int>> result;
            for(auto loadGlobal : loadTiledNodes)
            {
                auto internalMacroTile = kgraph.mapper.get<MacroTile>(loadGlobal);
                auto macTile           = kgraph.coordinates.getNode<MacroTile>(internalMacroTile);

                auto maybeDirect2LDS = kgraph.mapper.get<LDS>(loadGlobal);
                if(maybeDirect2LDS == -1)
                    continue;

                for(auto storeLDS : storeLDSTileNodes)
                {

                    bool sameMacroTile
                        = (kgraph.mapper.get<MacroTile>(storeLDS) == internalMacroTile);
                    auto LDSTileTag = kgraph.mapper.get<LDS>(storeLDS);
                    auto LDSTile    = kgraph.coordinates.getNode<LDS>(LDSTileTag);

                    if(!LDSTile.isDirect2LDS)
                        continue;

                    if(sameMacroTile)
                        result.push_back({loadGlobal, storeLDS});
                }
            }
            return result;
        }

        void mergeOperations(KernelGraph& kgraph, int globalOp, int ldsOp)
        {
            using namespace ControlGraph;
            auto variableType = getVariableType(kgraph, globalOp);

            if(isChain(kgraph, globalOp, ldsOp))
            {
                Log::debug("  Merge LoadTiled {} and StoreLDSTile {} chain.", globalOp, ldsOp);

                // create LoadTileDirect2LDS operation
                auto direct2lds = kgraph.control.addElement(LoadTileDirect2LDS(variableType));
                moveConnections(kgraph, globalOp, direct2lds);
                moveConnections(kgraph, ldsOp, direct2lds);

                // replace operations
                replaceWith(kgraph, globalOp, kgraph.control.addElement(NOP()), false);
                replaceWith(kgraph, ldsOp, direct2lds, false);

                purgeNodes(kgraph, {globalOp});
                purgeNodes(kgraph, {ldsOp});
            }
            else
            {
                Log::debug("  Merge LoadTiled {} and StoreLDSTile {} leaf.", globalOp, ldsOp);

                // the LoadTiled and StoreLDSTile have to be the leaf node
                AssertFatal(isLeaf(kgraph, globalOp) && isLeaf(kgraph, ldsOp));

                // create LoadTileDirect2LDS operation
                auto direct2lds = kgraph.control.addElement(LoadTileDirect2LDS(variableType));
                moveConnections(kgraph, globalOp, direct2lds);
                moveConnections(kgraph, ldsOp, direct2lds);

                // find the barrier before StoreLDSTile operation
                int ldsBarrier      = -1;
                int computeIndexTag = -1;
                for(auto parent : kgraph.control.depthFirstVisit(ldsOp, Graph::Direction::Upstream))
                {
                    bool containing = ldsBarrier != -1
                                      && (kgraph.control.get<Body>(ldsBarrier)
                                          || kgraph.control.get<Sequence>(ldsBarrier));
                    ldsBarrier = parent;
                    if(kgraph.control.get<ComputeIndex>(parent))
                    {
                        computeIndexTag = parent;
                    }

                    auto maybeBarrier = kgraph.control.get<Barrier>(parent);
                    if(maybeBarrier && containing)
                    {
                        break;
                    }
                }
                AssertFatal(computeIndexTag != -1 && ldsBarrier != -1);

                // add LoadTileDirect2LDS operation to the graph
                auto barrier = kgraph.control.addElement(Barrier());
                reconnect<Graph::Direction::Upstream>(kgraph, -1, computeIndexTag);
                kgraph.control.addElement(Sequence(), {globalOp}, {computeIndexTag});
                replaceWith(kgraph, globalOp, kgraph.control.addElement(NOP()), false);
                replaceWith(kgraph, ldsBarrier, kgraph.control.addElement(NOP()), false);
                replaceWith(kgraph, ldsOp, barrier, false);
                kgraph.control.addElement(Sequence(), {barrier}, {direct2lds});
                purgeNodes(kgraph, {globalOp});
                purgeNodes(kgraph, {ldsOp});
                purgeNodes(kgraph, {ldsBarrier});
            }
        }
        /** This transformation does:
         *
         *    1. Search the pairs of LoadTiled and StoreLDSTile operations that connects to the same internal MacroTile
         *
         *    2. Merge each pair
         */
        KernelGraph AddDirect2LDS::apply(KernelGraph const& original)
        {
            auto kgraph = original;

            Log::debug("  AddDirect2LDS control graph transform.");
            auto candidates = searchCandidates(kgraph, m_params, m_context);

            if(candidates.size() > 0)
            {
                AssertFatal(
                    m_context->targetArchitecture().HasCapability(GPUCapability::HasDirectToLds),
                    "Not have DirectToLds capability");
            }

            for(auto loadAndStore : candidates)
            {
                mergeOperations(kgraph, loadAndStore.first, loadAndStore.second);
            }

            return kgraph;
        }
    }
}
