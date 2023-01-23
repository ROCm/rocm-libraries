
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        // Add LDSLoad and LDSStore nodes following a LoadTiled node within
        // the control graph, if the tile has a memory type of LDS.
        void addLDSOps(KernelGraph& graph, std::shared_ptr<Context> context, int tag)
        {
            auto user_tag = graph.mapper.get<User>(tag);
            auto tile_tag = graph.mapper.get<MacroTile>(tag);
            auto tile     = graph.coordinates.getNode<MacroTile>(tile_tag);
            auto load     = graph.control.getNode<LoadTiled>(tag);

            if(tile.memoryType == MemoryType::LDS)
            {
                auto lds = graph.coordinates.addElement(LDS());

                auto storeLDS = graph.control.addElement(StoreLDSTile(load.vtype.dataType));
                auto barrier  = graph.control.addElement(Barrier());
                auto loadLDS  = graph.control.addElement(LoadLDSTile(load.vtype));

                graph.mapper.connect<LDS>(storeLDS, lds);
                graph.mapper.connect<MacroTile>(storeLDS, tile_tag);

                graph.mapper.connect<LDS>(loadLDS, lds);
                graph.mapper.connect<MacroTile>(loadLDS, tile_tag);

                auto workgroupSizes = context->kernel()->workgroupSize();
                loadMacroTileFromLDS(graph, loadLDS, lds, tile_tag, workgroupSizes);
                storeMacroTileIntoLDS(graph, storeLDS, lds, tile_tag, workgroupSizes);

                graph.coordinates.addElement(DataFlow(), {user_tag}, {tile_tag});

                // Find all edges leaving from load. Those should be changed to leave from loadLDS.
                auto outgoing_edges = graph.control.getNeighbours<Graph::Direction::Downstream>(tag)
                                          .to<std::vector>();
                for(auto e : outgoing_edges)
                {
                    auto elem = graph.control.getElement(e);
                    auto dst  = graph.control.getNeighbours<Graph::Direction::Downstream>(e)
                                   .to<std::vector>();
                    graph.control.deleteElement(e);
                    graph.control.addElement(e, elem, std::vector<int>{loadLDS}, dst);
                }

                graph.control.addElement(Sequence(), {tag}, {storeLDS});
                graph.control.addElement(Sequence(), {storeLDS}, {barrier});
                graph.control.addElement(Sequence(), {barrier}, {loadLDS});
            }
        }

        void addLoadWaveLDSOps(KernelGraph&                       graph,
                               std::shared_ptr<CommandParameters> params,
                               std::shared_ptr<Context>           context,
                               int                                tile,
                               int                                load,
                               int                                forK,
                               int                                K,
                               int                                waveMult)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::addLoadWaveLDSOps: LoadTiled({})", load);

            auto macrotile = graph.coordinates.getNode<MacroTile>(tile);

            if(macrotile.memoryType == MemoryType::WAVE_LDS)
            {
                // change loadTiled to LoadLDSTile under Multiply
                // and update its macrotile's memory type
                auto vtype = graph.control.getNode<LoadTiled>(load).vtype;
                graph.control.setElement(load, LoadLDSTile(vtype));
                macrotile.memoryType = MemoryType::WAVE;
                graph.coordinates.setElement(tile, macrotile);

                auto user = graph.coordinates.getInputNodeIndices(tile, CT::isEdge<DataFlow>)
                                .to<std::vector>();
                AssertFatal(user.size() == 1);
                graph.coordinates.deleteElement(user, std::vector<int>{tile}, CT::isEdge<DataFlow>);
                auto sdims = graph.coordinates.getOutputNodeIndices(user[0], CT::isEdge<Split>)
                                 .to<std::vector>();

                auto lds = graph.coordinates.addElement(LDS());

                // remove workgroups, macrotile numbers and tile edges from sdims
                loadWaveMacroTileFromLDS(graph, macrotile, load, sdims, K, lds);

                // add new loadTiled node to load a macrotile into VGPRs from global memory under the forK loop
                auto load_macrotile_from_global = graph.control.addElement(LoadTiled(vtype));
                graph.control.addElement(Body(), {forK}, {load_macrotile_from_global});
                graph.mapper.connect<User>(load_macrotile_from_global, user[0]);

                // create an internal macrotile to be loaded by one workgroup
                auto workgroupSizes = context->kernel()->workgroupSize();
                auto numWorkitems   = product(workgroupSizes);
                auto numElements    = product(macrotile.sizes);
                auto numVGPRs       = static_cast<int>(numElements / numWorkitems);
                // TODO : load the two Halfs into 1 VGPR
                //if(vtype == DataType::Half)
                //num_vgprs = num_vgprs / 2;
                auto t_m          = numVGPRs;
                auto t_n          = 1;
                auto internalTile = graph.coordinates.addElement(
                    MacroTile(macrotile.sizes, MemoryType::VGPR, {t_m, t_n}));
                auto internalTileDim       = graph.coordinates.getNode<MacroTile>(internalTile);
                internalTileDim.layoutType = macrotile.layoutType;
                graph.coordinates.setElement(internalTile, internalTileDim);
                graph.mapper.connect<MacroTile>(load_macrotile_from_global, internalTile);

                // user --DataFlow--> internalTile
                graph.coordinates.addElement(DataFlow(), {user[0]}, {internalTile});

                // lower tile LoadTiled : load macrotile from global memory
                loadMacroTileForLDS(graph,
                                    load_macrotile_from_global,
                                    user[0],
                                    internalTile,
                                    sdims,
                                    K,
                                    workgroupSizes);

                // add store from VGPRs to LDS following this new loadTiled under the forK loop
                auto store_macrotile_into_LDS
                    = graph.control.addElement(StoreLDSTile(vtype.dataType));
                auto barrier = graph.control.addElement(Barrier());
                graph.control.addElement(
                    Sequence(), {load_macrotile_from_global}, {store_macrotile_into_LDS});
                graph.control.addElement(Sequence(), {store_macrotile_into_LDS}, {barrier});
                graph.control.addElement(Sequence(), {barrier}, {waveMult});
                graph.mapper.connect<MacroTile>(store_macrotile_into_LDS, internalTile);

                // lower tile StoreLDSTile : store macrotile into LDS
                storeMacroTileIntoLDS(
                    graph, store_macrotile_into_LDS, lds, internalTile, workgroupSizes);

                // LDS --DataFlow--> macrotile
                graph.coordinates.addElement(DataFlow(), {lds}, {tile});

                graph.mapper.connect<LDS>(load, lds);
                graph.mapper.connect<LDS>(load_macrotile_from_global, lds);
                graph.mapper.connect<LDS>(store_macrotile_into_LDS, lds);
            }
        }

        void addStoreWaveLDSOps(KernelGraph&                       graph,
                                std::shared_ptr<CommandParameters> params,
                                std::shared_ptr<Context>           context,
                                int                                tile,
                                int                                store,
                                int                                upperLoop)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::addStoreWaveLDSOps: StoreTiled({})", store);

            auto macrotile = graph.coordinates.getNode<MacroTile>(tile);

            if(macrotile.memoryType == MemoryType::WAVE_LDS)
            {
                // change StoreTiled to StoreLDSTile
                // and update its macrotile's memory type
                auto dtype = graph.control.getNode<StoreTiled>(store).dataType;
                graph.control.setElement(store, StoreLDSTile(dtype));
                macrotile.memoryType = MemoryType::WAVE;
                graph.coordinates.setElement(tile, macrotile);

                auto user = graph.coordinates.getOutputNodeIndices(tile, CT::isEdge<DataFlow>)
                                .to<std::vector>();
                AssertFatal(user.size() == 1);
                graph.coordinates.deleteElement(std::vector<int>{tile}, user, CT::isEdge<DataFlow>);
                auto sdims = graph.coordinates.getInputNodeIndices(user[0], CT::isEdge<Join>)
                                 .to<std::vector>();
                AssertFatal(sdims.size() > 1);

                auto lds = graph.coordinates.addElement(LDS());

                // remove workgroups, macrotile numbers and tile edges from sdims
                storeWaveMacroTileIntoLDS(graph, macrotile, store, sdims, lds);

                // macrotile --DataFlow--> LDS
                graph.coordinates.addElement(DataFlow(), {tile}, {lds});
                graph.mapper.connect<LDS>(store, lds);

                auto barrier = graph.control.addElement(Barrier());
                graph.control.addElement(Sequence(), {store}, {barrier});

                // add new loadLDSTile node to load a macrotile into VGPRs from LDS
                auto load_macrotile_from_LDS
                    = graph.control.addElement(LoadLDSTile(VariableType(dtype)));
                graph.mapper.connect<LDS>(load_macrotile_from_LDS, lds);
                graph.control.addElement(Sequence(), {upperLoop}, {load_macrotile_from_LDS});

                // create an internal macrotile to be loaded by one workgroup
                auto workgroupSizes = context->kernel()->workgroupSize();
                auto numWorkitems   = product(workgroupSizes);
                auto numElements    = product(macrotile.sizes);
                auto numVGPRs       = static_cast<int>(numElements / numWorkitems);
                // TODO : load the two Halfs into 1 VGPR
                //if(vtype == DataType::Half)
                //num_vgprs = num_vgprs / 2;
                auto t_m          = numVGPRs;
                auto t_n          = 1;
                auto internalTile = graph.coordinates.addElement(
                    MacroTile(macrotile.sizes, MemoryType::VGPR, {t_m, t_n}));
                auto internalTileDim       = graph.coordinates.getNode<MacroTile>(internalTile);
                internalTileDim.layoutType = macrotile.layoutType;
                graph.coordinates.setElement(internalTile, internalTileDim);
                graph.mapper.connect<MacroTile>(load_macrotile_from_LDS, internalTile);

                // lower tile LoadLDSTile : load macrotile from LDS
                loadMacroTileFromLDS(
                    graph, load_macrotile_from_LDS, lds, internalTile, workgroupSizes);

                // add store from VGPRs to global following this new loadLDSTile
                auto store_macrotile_into_global = graph.control.addElement(StoreTiled(dtype));
                graph.control.addElement(
                    Sequence(), {load_macrotile_from_LDS}, {store_macrotile_into_global});
                graph.mapper.connect<MacroTile>(store_macrotile_into_global, internalTile);
                graph.mapper.connect<User>(store_macrotile_into_global, user[0]);
                // internalTile --DataFlow--> user
                graph.coordinates.addElement(DataFlow(), {internalTile}, {user[0]});

                storeMacroTileForLDS(graph,
                                     store_macrotile_into_global,
                                     user[0],
                                     internalTile,
                                     sdims,
                                     workgroupSizes);
            }
        }
    }
}
