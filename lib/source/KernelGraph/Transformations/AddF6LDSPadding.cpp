#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/KernelGraph/Transforms/AddF6LDSPadding.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using GD = Graph::Direction;

        bool isMatrixA(MacroTile const& macTile)
        {
            return macTile.layoutType == LayoutType::MATRIX_A;
        }

        bool isMatrixB(MacroTile const& macTile)
        {
            return macTile.layoutType == LayoutType::MATRIX_B;
        }

        MacroTile paddedMacroTile(MacroTile& macroTile)
        {
            uint const elementBits     = 6;
            uint const packing         = 16;
            auto       padElementBytes = extraLDSBytesPerElementBlock(elementBits);

            auto fastMovingDim = isMatrixA(macroTile) ? 0 : 1;
            auto padElements   = macroTile.sizes[fastMovingDim] / packing;

            std::vector<uint> padBytesA = {padElements * padElementBytes, 0};
            std::vector<uint> padBytesB = {0, padElements * padElementBytes};

            return MacroTile(macroTile, isMatrixA(macroTile) ? padBytesA : padBytesB);
        }

        auto findCandidates(KernelGraph const& graph)
        {
            auto start = *graph.control.roots().begin();

            auto isLoadLDSTileOfTransposedTile = [&](int tag) {
                auto elem = graph.control.getElement(tag);
                if(!std::holds_alternative<Operation>(elem))
                    return false;
                auto op = std::get<Operation>(elem);
                if(!std::holds_alternative<LoadLDSTile>(op))
                    return false;

                auto load                  = std::get<LoadLDSTile>(op);
                auto [macTileTag, macTile] = graph.getDimension<MacroTile>(tag);
                auto elementBits           = DataTypeInfo::Get(load.varType).elementBits;

                return elementBits == 6 && load.isTransposedTile
                       && (isMatrixA(macTile) || isMatrixB(macTile));
            };

            auto candidates
                = graph.control.findNodes(start, isLoadLDSTileOfTransposedTile, GD::Downstream);

            // remove `start` element from candidates.
            auto allButStart = [start](auto c) { return c != start; };
            return filter(allButStart, std::move(candidates)).to<std::vector>();
        }

        KernelGraph AddF6LDSPadding::apply(KernelGraph const& graph)
        {
            TIMER(t, "KernelGraph::AddF6LDSPadding");
            auto logger = rocRoller::Log::getLogger();
            logger->debug("KernelGraph::AddF6LDSPaddingVisitor");

            auto candidates = findCandidates(graph);

            // Return unchanged graph if no LoadLDSTile of transposed tile found.
            if(std::ranges::empty(candidates))
            {
                return graph;
            }

            auto kgraph{graph};

            std::unordered_set<int> visitedCoordinates;
            std::unordered_set<int> visitedOps;
            for(auto tag : candidates)
            {
                if(visitedOps.contains(tag))
                    continue;
                visitedOps.insert(tag);

                auto connections = graph.mapper.getConnections(tag);
                for(auto conn : connections)
                {
                    auto coordTag = conn.coordinate;

                    if(visitedCoordinates.contains(coordTag))
                        continue;
                    visitedCoordinates.insert(coordTag);

                    std::visit(rocRoller::overloaded{[&kgraph, tag = conn.coordinate](User user) {
                                                         auto newUser{user};
                                                         newUser.needsPadding = true;
                                                         kgraph.coordinates.setElement(tag,
                                                                                       newUser);
                                                     },
                                                     [&](auto coord) {}},
                               std::get<Dimension>(graph.coordinates.getElement(conn.coordinate)));

                    auto maybeLDS = graph.coordinates.get<LDS>(coordTag);
                    if(maybeLDS)
                    {
                        auto ldsTag = coordTag;
                        auto lds{*maybeLDS};
                        auto newLDS(lds);
                        newLDS.holdsTransposedTile = true;
                        kgraph.coordinates.setElement(ldsTag, newLDS);

                        for(auto conn : graph.mapper.getCoordinateConnections(ldsTag))
                        {
                            auto coordTag   = conn.coordinate;
                            auto controlTag = conn.control;

                            auto storeLDSTile = graph.control.get<StoreLDSTile>(controlTag);
                            if(storeLDSTile)
                            {
                                auto storeLDSTileTag{controlTag};
                                auto [macroTileTag, macroTile]
                                    = graph.getDimension<MacroTile>(storeLDSTileTag);

                                kgraph.coordinates.setElement(macroTileTag,
                                                              paddedMacroTile(macroTile));

                                for(auto conn : graph.mapper.getConnections(storeLDSTileTag))
                                {
                                    auto maybeMacroTile
                                        = graph.coordinates.get<MacroTile>(conn.coordinate);
                                    if(maybeMacroTile)
                                    {
                                        kgraph.coordinates.setElement(
                                            conn.coordinate, paddedMacroTile(*maybeMacroTile));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return kgraph;
        }
    }
}
