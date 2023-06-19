
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/ConnectWorkgroups.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordinateGraph;
        using GD = rocRoller::Graph::Direction;

        KernelGraph ConnectWorkgroups::apply(KernelGraph const& original)
        {
            auto logger = rocRoller::Log::getLogger();
            auto kgraph = original;

            auto tileNumTags = kgraph.coordinates.getNodes<MacroTileNumber>().to<std::vector>();
            for(auto const& tileNumTag : tileNumTags)
            {
                if(empty(kgraph.coordinates.getNeighbours<GD::Downstream>(tileNumTag)))
                {
                    // MacroTileNumber is dangling, connect it to a Workgroup
                    auto tileNum      = *kgraph.coordinates.get<MacroTileNumber>(tileNumTag);
                    auto workgroupTag = kgraph.coordinates.addElement(Workgroup(tileNum.dim));
                    logger->debug(
                        "KernelGraph::ConnectWorkgroups: Adding PassThrough from {} to {}",
                        tileNumTag,
                        workgroupTag);
                    kgraph.coordinates.addElement(PassThrough(), {tileNumTag}, {workgroupTag});
                }
                if(empty(kgraph.coordinates.getNeighbours<GD::Upstream>(tileNumTag)))
                {
                    // MacroTileNumber is dangling, connect it to a Workgroup
                    auto tileNum      = *kgraph.coordinates.get<MacroTileNumber>(tileNumTag);
                    auto workgroupTag = kgraph.coordinates.addElement(Workgroup(tileNum.dim));
                    logger->debug(
                        "KernelGraph::ConnectWorkgroups: Adding PassThrough from {} to {}",
                        workgroupTag,
                        tileNumTag);
                    kgraph.coordinates.addElement(PassThrough(), {workgroupTag}, {tileNumTag});
                }
            }

            return kgraph;
        }
    }
}
