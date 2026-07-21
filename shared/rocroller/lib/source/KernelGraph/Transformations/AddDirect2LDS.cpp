// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddDirect2LDS.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils_impl.hpp>

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

                return getLoadTiledStoreLDSTilePairs(kgraph, isDirect2LDSLoadTiled);
            }

            void replaceLoadTiled(KernelGraph& kgraph,
                                  int          loadTiledTag,
                                  int          storeLDSTileTag,
                                  int          loadTileDirect2LDSTag)
            {
                using namespace ControlGraph;

                const auto element = kgraph.control.getElement(loadTileDirect2LDSTag);
                AssertFatal(std::holds_alternative<Operation>(element),
                            concatenate("Expected Operation but got Edge",
                                        ShowValue(loadTileDirect2LDSTag)));

                const auto op = std::get<Operation>(element);
                AssertFatal(std::holds_alternative<LoadTileDirect2LDS>(op),
                            fmt::format("Expected LoadTileDirect2LDS but got {}", toString(op)));

                auto codegen = getCodeGeneratorCoordinates(kgraph, storeLDSTileTag);

                moveConnections(kgraph, loadTiledTag, loadTileDirect2LDSTag, 0);
                moveConnections(kgraph, storeLDSTileTag, loadTileDirect2LDSTag, codegen.size());

                replaceWith(kgraph, loadTiledTag, loadTileDirect2LDSTag, false);
                Log::debug("  Replaced LoadTiled({}) with LoadTileDirect2LDS({}).",
                           loadTiledTag,
                           loadTileDirect2LDSTag);
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
                    "  Found LoadTiled({}) and StoreLDSTile({}).", loadTiledTag, storeLDSTileTag);

                // create LoadTileDirect2LDS operation
                auto variableType = getVariableType(kgraph, loadTiledTag);
                auto direct2lds   = kgraph.control.addElement(LoadTileDirect2LDS(variableType));

                replaceLoadTiled(kgraph, loadTiledTag, storeLDSTileTag, direct2lds);
                nodesToPurge.insert(loadTiledTag);

                if(nodesToPurge.count(storeLDSTileTag) == 0)
                {
                    replaceWith(kgraph, storeLDSTileTag, kgraph.control.addElement(NOP()), false);
                    Log::debug("  Replaced StoreLDSTile({}) with NOP.", storeLDSTileTag);
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
