// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddTDMToLDS.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace AddTDMToLDSDetail
        {
            std::vector<std::pair<int, int>> findCandidates(KernelGraph const& kgraph,
                                                            ContextPtr         context)
            {
                using namespace ControlGraph;
                using namespace CoordinateGraph;

                auto isTDMToLDSLoadTiled = [&kgraph](int tag) {
                    bool rv = false;
                    if(kgraph.control.get<LoadTiled>(tag))
                    {
                        auto macroTile
                            = kgraph.coordinates.get<MacroTile>(kgraph.mapper.get<MacroTile>(tag));
                        rv = macroTile && macroTile->memoryType == MemoryType::WAVE_TDMToLDS;
                    }
                    return rv;
                };

                return getLoadTiledStoreLDSTilePairs(kgraph, isTDMToLDSLoadTiled);
            }

            void replaceLoadTiled(KernelGraph& kgraph,
                                  int          loadTiledTag,
                                  int          storeLDSTileTag,
                                  int          loadTiledTDMToLDSTag)
            {
                using namespace ControlGraph;

                auto codegen = getCodeGeneratorCoordinates(kgraph, storeLDSTileTag);

                moveConnections(kgraph, loadTiledTag, loadTiledTDMToLDSTag, 0);
                moveConnections(kgraph, storeLDSTileTag, loadTiledTDMToLDSTag, codegen.size());

                replaceWith(kgraph, loadTiledTag, loadTiledTDMToLDSTag, false);
                Log::debug("  AddTDMToLDS: Replace LoadTiled({}) with LoadTiledTDMToLDS({}).",
                           loadTiledTag,
                           loadTiledTDMToLDSTag);
            }
        }

        /** @brief This transformation replaces LoadTiled that load MacroTiles
         *  of WAVE_TDMToLDS with LoadTiledTDMToLDS and turns the respective
         *  StoreLDSTile operations store said MacroTiles in LDS into NOPs.
         */
        KernelGraph AddTDMToLDS::apply(KernelGraph const& original)
        {
            using namespace ControlGraph;
            using namespace AddTDMToLDSDetail;

            Log::debug("KernelGraph::Transforms::AddTDMToLDS::apply()");

            const auto& arch   = m_context->targetArchitecture();
            const auto  hasTDM = arch.HasCapability(GPUCapability::HasTDM);
            if(!hasTDM)
            {
                Log::debug(
                    fmt::format("  AddTDMToLDS: Target {} does not support TDM. Nothing to do.\n",
                                toString(arch.target())));
                return original;
            }

            const auto candidates = findCandidates(original, m_context);
            if(std::ranges::empty(candidates))
            {
                Log::debug("  AddTDMToLDS: No candidates found. Nothing to do.\n");
                return original;
            }

            auto kgraph{original};

            std::unordered_set<int> nodesToPurge;
            for(auto [loadTiledTag, storeLDSTileTag] : candidates)
            {
                Log::debug(fmt::format("  AddTDMToLDS: found LoadTiled({}) and StoreLDSTile({})\n",
                                       loadTiledTag,
                                       storeLDSTileTag));

                auto variableType = getVariableType(kgraph, loadTiledTag);
                auto loadTiledTDMToLDSTag
                    = kgraph.control.addElement(LoadTiledTDMToLDS(variableType));

                replaceLoadTiled(kgraph, loadTiledTag, storeLDSTileTag, loadTiledTDMToLDSTag);
                nodesToPurge.insert(loadTiledTag);

                if(nodesToPurge.count(storeLDSTileTag) == 0)
                {
                    replaceWith(kgraph, storeLDSTileTag, kgraph.control.addElement(NOP()), false);
                    Log::debug("  AddTDMToLDS: Replace StoreLDSTile({}) with NOP.",
                               storeLDSTileTag);
                    nodesToPurge.insert(storeLDSTileTag);
                }
            }

            for(auto node : nodesToPurge)
            {
                purgeNodes(kgraph, {node});
            }

            return kgraph;
        }

        ConstraintStatus NoTDMToLDSTile(const KernelGraph& kgraph)
        {
            TIMER(t, "Constraint::NoTDMToLDSTile");
            using namespace ControlGraph;
            using namespace CoordinateGraph;

            ConstraintStatus retval;

            auto isLoadTiledOrStoreLDSTileOfTDMToLDSTile = [&kgraph](int tag) {
                bool rv = false;
                if(kgraph.control.get<LoadTiled>(tag))
                {
                    auto macroTile
                        = kgraph.coordinates.get<MacroTile>(kgraph.mapper.get<MacroTile>(tag));
                    rv = macroTile && macroTile->memoryType == MemoryType::WAVE_TDMToLDS;
                }
                else if(kgraph.control.get<StoreLDSTile>(tag))
                {
                    auto macroTile
                        = kgraph.coordinates.get<MacroTile>(kgraph.mapper.get<MacroTile>(tag));
                    rv = macroTile && macroTile->memoryType == MemoryType::WAVE_TDMToLDS;
                }
                return rv;
            };

            // Post-check: ensure all LDS LoadTiled and StoreLDSTiled have been replaced
            const auto unexpectedOps
                = filter(isLoadTiledOrStoreLDSTileOfTDMToLDSTile, kgraph.control.getNodes())
                      .to<std::vector>();
            retval.combine(std::ranges::empty(unexpectedOps),
                           concatenate("Found LoadTiled or StoreLDSTiled of WAVE_TDMToLDS Tile: ",
                                       unexpectedOps));

            return retval;
        }

        std::vector<GraphConstraint> AddTDMToLDS::postConstraints() const
        {
            return {NoTDMToLDSTile};
        }
    }
}
