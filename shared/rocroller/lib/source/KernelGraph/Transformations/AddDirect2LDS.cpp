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
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
namespace rocRoller
{
    namespace KernelGraph
    {
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
                std::vector<int> storeLDSTag;

                auto internalMacroTile = kgraph.mapper.get<MacroTile>(loadGlobal);
                auto macTile           = kgraph.coordinates.getNode<MacroTile>(internalMacroTile);

                auto load = kgraph.control.get<LoadTiled>(loadGlobal).value();
                if(!load.isDirect2LDS)
                    continue;

                for(auto storeLDS : storeLDSTileNodes)
                {
                    auto LDSTileTag = kgraph.mapper.get<LDS>(storeLDS);
                    auto LDSTile    = kgraph.coordinates.getNode<LDS>(LDSTileTag);

                    auto sameMacroTile
                        = (kgraph.mapper.get<MacroTile>(storeLDS) == internalMacroTile);

                    if(!LDSTile.isDirect2LDS)
                        continue;

                    if(sameMacroTile)
                        storeLDSTag.push_back(storeLDS);
                }

                if(storeLDSTag.size() == 0)
                    continue;
                else if(storeLDSTag.size() == 1)
                    result.push_back({loadGlobal, storeLDSTag[0]});
                else if(storeLDSTag.size() == 2)
                {
                    for(const auto& storeLDS : storeLDSTag)
                    {
                        auto maybeForLoop1 = findContainingOperation<ForLoopOp>(loadGlobal, kgraph);
                        auto maybeForLoop2 = findContainingOperation<ForLoopOp>(storeLDS, kgraph);
                        auto bothInSameForLoop
                            = maybeForLoop1 && maybeForLoop2 && (*maybeForLoop1 == *maybeForLoop2);
                        auto bothBeforeForLoop = !maybeForLoop1 && !maybeForLoop2;
                        if(bothInSameForLoop || bothBeforeForLoop)
                            result.push_back({loadGlobal, storeLDS});
                    }
                }
                else
                {
                    Log::debug("  AddDirect2LDS: More than 2 ComputeIndex operation required for "
                               "StoreLDSTile.");
                }
            }
            return result;
        }

        void replaceLoadTiled(KernelGraph&                     kgraph,
                              std::vector<std::pair<int, int>> direct2LDSInfo,
                              std::vector<int>&                used)
        {
            using namespace ControlGraph;

            for(const auto& info : direct2LDSInfo)
            {
                auto globalOp = info.first;
                auto ldsOp    = info.second;

                auto variableType = getVariableType(kgraph, globalOp);

                auto codegen = getCodeGeneratorCoordinates(kgraph, ldsOp);

                // create LoadTileDirect2LDS operation
                auto direct2lds = kgraph.control.addElement(LoadTileDirect2LDS(variableType));
                moveConnections(kgraph, globalOp, direct2lds, 0);
                moveConnections(kgraph, ldsOp, direct2lds, codegen.size());

                // replace LoadTiled with Direct2LDS
                replaceWith(kgraph, globalOp, direct2lds, false);
                Log::debug(
                    "  Replace LoadTiled {} with LoadTileDirect2LDS {}.", globalOp, direct2lds);

                used.push_back(globalOp);
            }
        }

        void replaceStoreLDS(KernelGraph&                     kgraph,
                             std::vector<std::pair<int, int>> direct2LDSInfo,
                             std::vector<int>&                used)
        {
            using namespace ControlGraph;
            for(const auto& info : direct2LDSInfo)
            {
                auto ldsOp = info.second;

                auto it = std::find(used.cbegin(), used.cend(), ldsOp);
                if(it != used.cend())
                    continue;

                replaceWith(kgraph, ldsOp, kgraph.control.addElement(NOP()), false);
                Log::debug("  Replace StoreLDSTile {} with NOP.", ldsOp);

                used.push_back(ldsOp);
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
            TIMER(t, "KernelGraph::AddDirect2LDS");

            auto kgraph = original;

            Log::debug("  AddDirect2LDS control graph transform.");
            auto candidates = searchCandidates(kgraph, m_params, m_context);

            if(candidates.size() > 0)
            {

                AssertFatal(
                    m_context->targetArchitecture().HasCapability(GPUCapability::HasDirectToLds),
                    "Not have DirectToLds capability");

                std::vector<int> used;
                replaceLoadTiled(kgraph, candidates, used);
                replaceStoreLDS(kgraph, candidates, used);

                purgeNodes(kgraph, used);
            }
            else
            {
                Log::debug("No candidates for AddDirect2LDS.");
            }

            return kgraph;
        }
    }
}
