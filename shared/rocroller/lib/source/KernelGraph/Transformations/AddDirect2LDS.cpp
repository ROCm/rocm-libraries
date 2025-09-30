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
        namespace AddDirect2LDSDetail
        {
            std::vector<std::pair<int, int>> searchCandidates(KernelGraph const& kgraph)
            {
                using namespace ControlGraph;
                using namespace CoordinateGraph;

                std::vector<std::pair<int, int>> result;

                auto isDirect2LDSLoadTiled = [&kgraph](int tag) {
                    bool rv = false;
                    if(kgraph.control.get<LoadTiled>(tag))
                    {
                        auto macroTile
                            = kgraph.coordinates.get<MacroTile>(kgraph.mapper.get<MacroTile>(tag));
                        rv = macroTile && macroTile->memoryType == MemoryType::WAVE_Direct2LDS;
                    }
                    return rv;
                };

                for(auto loadGlobal : kgraph.control.findElements(isDirect2LDSLoadTiled))
                {
                    const auto storeLDSTags{
                        getAssociatedOps<LoadTiled, StoreLDSTile>(kgraph, loadGlobal)};

                    if(storeLDSTags.size() == 1)
                    {
                        result.push_back({loadGlobal, storeLDSTags[0]});
                    }
                    else
                    {
                        AssertFatal(
                            storeLDSTags.size() <= 2,
                            "AddDirect2LDS: More than 2 ComputeIndex operation required for "
                            "StoreLDSTile.");
                        for(const auto& storeLDS : storeLDSTags)
                        {
                            auto maybeForLoopOfLoad
                                = findContainingOperation<ForLoopOp>(loadGlobal, kgraph);
                            auto maybeForLoopOfStore
                                = findContainingOperation<ForLoopOp>(storeLDS, kgraph);

                            const auto isLoadInLoop  = maybeForLoopOfLoad.has_value();
                            const auto isLoadInStore = maybeForLoopOfStore.has_value();

                            const auto bothInSameLoop
                                = isLoadInLoop && isLoadInStore
                                  && maybeForLoopOfLoad.value() == maybeForLoopOfStore.value();

                            const auto bothNotInLoop = not isLoadInLoop && not isLoadInStore;

                            if(bothInSameLoop || bothNotInLoop)
                            {
                                result.push_back({loadGlobal, storeLDS});
                            }
                        }
                    }
                }
                return result;
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
            using namespace ControlGraph;
            using namespace CoordinateGraph;
            using namespace AddDirect2LDSDetail;

            Log::debug("  AddDirect2LDS control graph transform.");

            auto candidates = searchCandidates(original);
            if(std::ranges::empty(candidates))
            {
                Log::debug("No candidates for AddDirect2LDS.");
                return original;
            }

            const auto& arch           = m_context->targetArchitecture();
            const auto  hasDirectToLDS = arch.HasCapability(GPUCapability::HasDirectToLds);
            AssertFatal(
                hasDirectToLDS,
                fmt::format("Target {} does not support DirectToLDS but candidates were found!",
                            toString(arch.target()),
                            ShowValue(candidates.size())));

            auto kgraph{original};

            std::unordered_set<int> nodesToPurge;
            for(auto [loadTiledTag, storeLDSTileTag] : candidates)
            {
                Log::debug(
                    "  Found LoadTiled {} and StoreLDSTile {}.", loadTiledTag, storeLDSTileTag);

                // create LoadTileDirect2LDS operation
                auto variableType = getVariableType(kgraph, loadTiledTag);
                auto direct2lds   = kgraph.control.addElement(LoadTileDirect2LDS(variableType));

                { // Is this necessary?
                    const auto macroTileTag = kgraph.mapper.get<MacroTile>(loadTiledTag);
                    MacroTile  macroTile{*kgraph.coordinates.get<MacroTile>(macroTileTag)};
                    macroTile.memoryType = MemoryType::VGPR;
                    kgraph.coordinates.setElement(macroTileTag, macroTile);
                }

                replaceLoadTiledWithGlobalToLDSOp(
                    kgraph, loadTiledTag, storeLDSTileTag, direct2lds);
                nodesToPurge.insert(loadTiledTag);

                if(nodesToPurge.count(storeLDSTileTag) == 0)
                {
                    replaceWith(kgraph, storeLDSTileTag, kgraph.control.addElement(NOP()), false);
                    Log::debug("  Replaced StoreLDSTile {} with NOP.", storeLDSTileTag);
                    nodesToPurge.insert(storeLDSTileTag);
                }
            }

            for(auto node : nodesToPurge)
            {
                purgeNodes(kgraph, {node});
            }

            return kgraph;
        }
    }
}
