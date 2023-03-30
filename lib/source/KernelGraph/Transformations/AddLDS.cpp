
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
        using namespace Register;

        struct AddLDSVisitor
        {
            struct ldsLoadInfo
            {
                int              loadTileFromGlobal;
                int              storeTileIntoLDS;
                std::vector<int> lds; //LDS coordinate tags for each unroll.
                int              internalTile;
                int              unrollCoord;
            };

            struct ldsStoreInfo
            {
                int                             internalTile;
                std::vector<DeferredConnection> loadConnections;
                std::vector<DeferredConnection> storeConnections;
                std::optional<int>              storeOperation;
            };

            AddLDSVisitor(std::shared_ptr<Context> context)
                : m_context(context)
            {
            }

            void addLoadThroughLDS(KernelGraph& graph, int tag);

            void addWaveLoadThroughLDSToControlGraph(KernelGraph& graph, int forLoop);
            void addWaveLoadThroughLDSToCoordinateGraph(KernelGraph& graph, int load);

            void addStoreThroughLDSToControlGraph(KernelGraph& graph, int store, int forLoop);
            void addStoreThroughLDSToCoordinateGraph(KernelGraph& graph, int store);

        private:
            void addLoadWaveLDSOps(KernelGraph&     graph,
                                   std::vector<int> loads,
                                   int              forK,
                                   std::vector<int> loopBodies,
                                   int              unrollIndex);

            std::tuple<int, bool> getOrigMacTile(KernelGraph& graph, std::vector<int> loads);

            std::map<int, ldsLoadInfo>  m_load;
            std::map<int, ldsStoreInfo> m_store;

            ContextPtr m_context;
        };

        KernelGraph addLDS(KernelGraph const& original, std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::addLDS");
            rocRoller::Log::getLogger()->debug("KernelGraph::addLDS()");

            auto k       = original;
            auto visitor = AddLDSVisitor(context);

            // TODO Query graphs to figure appropriate forLoop for LDS
            // store (see "TODO Query graphs .." below).  Then get rid
            // of this.
            // BEGIN get rid of this
            int kernel = *k.control.getNodes<Kernel>().begin();
            int topForLoop;
            for(auto tag : k.control.depthFirstVisit(kernel, Graph::Direction::Downstream))
            {
                auto maybeForLoop = k.control.get<ForLoopOp>(tag);
                if(maybeForLoop)
                {
                    topForLoop = tag;
                    break;
                }
            }
            // END get rid of this

            // Add LDS operations
            for(auto const& load : k.control.getNodes<LoadTiled>().to<std::vector>())
            {
                auto macroTileTag      = k.mapper.get<MacroTile>(load);
                auto macroTile         = k.coordinates.getNode<MacroTile>(macroTileTag);
                auto macroTileLocation = k.coordinates.getLocation(macroTileTag);
                // Only modify the coordinate graph for a load whose
                // associated MacroTile is not a duplicate.
                if(macroTile.memoryType == MemoryType::WAVE_LDS
                   && !macroTileLocation.incoming.empty())
                {
                    visitor.addWaveLoadThroughLDSToCoordinateGraph(k, load);
                }
                if(macroTile.memoryType == MemoryType::LDS)
                {
                    visitor.addLoadThroughLDS(k, load);
                }
            }

            for(auto const& forLoop : k.control.getNodes<ForLoopOp>())
            {
                visitor.addWaveLoadThroughLDSToControlGraph(k, forLoop);
            }

            // Add LDS coordinates and transforms
            for(auto const& store : k.control.getNodes<StoreTiled>().to<std::vector>())
            {
                auto [macroTileTag, macroTile] = k.getDimension<MacroTile>(store);
                if(macroTile.memoryType == MemoryType::WAVE_LDS)
                {
                    auto location = k.coordinates.getLocation(macroTileTag);
                    // Only modify the coordinate graph for a store
                    // whose associated MacroTile is not a duplicate.
                    if(!location.incoming.empty())
                    {
                        visitor.addStoreThroughLDSToCoordinateGraph(k, store);
                    }
                }
            }

            for(auto const& store : k.control.getNodes<StoreTiled>().to<std::vector>())
            {
                // TODO Query graphs to figure appropriate forLoop
                // Should probably do that logic inside
                // addStoreThroughLDSToControlGraph
                visitor.addStoreThroughLDSToControlGraph(k, store, topForLoop);
            }

            return k;
        }

        void AddLDSVisitor::addLoadThroughLDS(KernelGraph& graph, int tag)
        {
            auto userTag = graph.mapper.get<User>(tag);
            auto tileTag = graph.mapper.get<MacroTile>(tag);
            auto tile    = graph.coordinates.getNode<MacroTile>(tileTag);
            auto load    = graph.control.getNode<LoadTiled>(tag);

            if(tile.memoryType == MemoryType::LDS)
            {
                // change loadTiled to LoadLDSTile
                graph.control.setElement(tag, LoadLDSTile(load.vtype));

                graph.coordinates.deleteElement(
                    std::vector<int>{userTag}, std::vector<int>{tileTag}, CT::isEdge<DataFlow>);
                auto sdims = graph.coordinates.getOutputNodeIndices(userTag, CT::isEdge<Split>)
                                 .to<std::vector>();

                auto lds = graph.coordinates.addElement(LDS());

                // remove workgroups, macrotile numbers and tile edges from sdims
                updateLoadLDSMacroTile(graph, tile, tag, sdims, -1, lds, true);

                // add new loadTiled node to load a macrotile into VGPRs from global memory
                auto loadMacroTileFromGlobal = graph.control.addElement(LoadTiled(load.vtype));
                graph.mapper.connect<User>(loadMacroTileFromGlobal, userTag);

                // Find all incoming edges into tag. Those should be changed to come into loadMacroTileFromGlobal.
                auto incomingEdges = graph.control.getNeighbours<Graph::Direction::Upstream>(tag)
                                         .to<std::vector>();
                for(auto e : incomingEdges)
                {
                    auto elem = graph.control.getElement(e);
                    auto src  = graph.control.getNeighbours<Graph::Direction::Upstream>(e)
                                   .to<std::vector>();
                    graph.control.deleteElement(e);
                    graph.control.addElement(
                        e, elem, src, std::vector<int>{loadMacroTileFromGlobal});
                }

                // create an internal macrotile to be loaded by one workgroup
                auto workgroupSizes = m_context->kernel()->workgroupSize();
                auto internalTile   = graph.coordinates.addElement(
                    MacroTile(tile.sizes, MemoryType::VGPR, tile.subTileSizes));
                auto internalTileDim = graph.coordinates.getNode<MacroTile>(internalTile);
                graph.coordinates.setElement(internalTile, internalTileDim);
                graph.mapper.connect<MacroTile>(loadMacroTileFromGlobal, internalTile);

                // user --DataFlow--> internalTile
                graph.coordinates.addElement(DataFlow(), {userTag}, {internalTile});

                // lower tile LoadTiled : load macrotile from global memory
                loadMacroTileForLDS(graph,
                                    loadMacroTileFromGlobal,
                                    userTag,
                                    internalTile,
                                    sdims,
                                    -1,
                                    workgroupSizes,
                                    -1,
                                    true);

                // add store from VGPRs to LDS following this new loadTiled
                auto storeMacroTileIntoLDSNode
                    = graph.control.addElement(StoreLDSTile(load.vtype.dataType));
                auto barrier = graph.control.addElement(Barrier());
                graph.control.addElement(
                    Sequence(), {loadMacroTileFromGlobal}, {storeMacroTileIntoLDSNode});
                graph.control.addElement(Sequence(), {storeMacroTileIntoLDSNode}, {barrier});
                graph.control.addElement(Sequence(), {barrier}, {tag});
                graph.mapper.connect<MacroTile>(storeMacroTileIntoLDSNode, internalTile);

                // lower tile StoreLDSTile : store macrotile into LDS
                storeMacroTileIntoLDS(
                    graph, storeMacroTileIntoLDSNode, lds, internalTile, workgroupSizes, true);

                // LDS --DataFlow--> macrotile
                graph.coordinates.addElement(DataFlow(), {lds}, {tileTag});

                graph.mapper.connect<LDS>(tag, lds);
                graph.mapper.connect<LDS>(loadMacroTileFromGlobal, lds);
                graph.mapper.connect<LDS>(storeMacroTileIntoLDSNode, lds);
            }
        }

        std::tuple<int, bool> AddLDSVisitor::getOrigMacTile(KernelGraph&     graph,
                                                            std::vector<int> loads)
        {
            // Find all of the loads macro tiles and see if they are in the
            // info map. If not, use passthrough edge to find macrotile.
            int  origMacroTile = -1;
            bool origInLoads   = false;
            for(auto const& load : loads)
            {
                auto macroTile = graph.mapper.get<MacroTile>(load);
                if(m_load.count(macroTile) == 1)
                {
                    origInLoads   = true;
                    origMacroTile = macroTile;
                }
                else
                {
                    auto orig
                        = graph.coordinates.getOutputNodeIndices(macroTile, CT::isEdge<PassThrough>)
                              .to<std::vector>();
                    if(orig.size() == 1)
                    {
                        origMacroTile = orig[0];
                    }
                }
            }
            return std::make_tuple(origMacroTile, origInLoads);
        }

        void AddLDSVisitor::addLoadWaveLDSOps(KernelGraph&     graph,
                                              std::vector<int> loads,
                                              int              forK,
                                              std::vector<int> loopBodies,
                                              int              unrollIndex)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::addLoadWaveLDSOps: ForLoop({}) Unroll({})", forK, unrollIndex);

            int user = graph.mapper.get<User>(loads[0]);

            // Ensure that all loads have the same User value
            for(auto const& load : loads)
            {
                AssertFatal(graph.mapper.get<User>(load) == user,
                            "All loads should have same User value");
            }

            auto macrotile
                = graph.coordinates.getNode<MacroTile>(graph.mapper.get<MacroTile>(loads[0]));

            if(macrotile.memoryType == MemoryType::WAVE_LDS)
            {
                // Find all of the loads macro tiles and see if they are in the
                // info map. If not, use passthrough edge to find macrotile.
                int  origMacroTile                   = -1;
                bool origInLoads                     = false;
                std::tie(origMacroTile, origInLoads) = getOrigMacTile(graph, loads);
                AssertFatal(m_load.count(origMacroTile) == 1);
                auto localInfo = m_load[origMacroTile];

                auto vtype           = graph.control.getNode<LoadTiled>(loads[0]).vtype;
                macrotile.memoryType = MemoryType::WAVE;

                // change loadTiled to LoadLDSTile under Multiply
                for(auto const& load : loads)
                {
                    graph.control.setElement(load, LoadLDSTile(vtype));
                    auto loadTile = graph.mapper.get<MacroTile>(load);
                    graph.coordinates.setElement(loadTile, macrotile);
                    auto loadTileLocation = graph.coordinates.getLocation(loadTile);
                    graph.mapper.connect<LDS>(load, localInfo.lds[unrollIndex]);
                    for(auto const& c : graph.mapper.getConnections(load))
                    {
                        if(!graph.coordinates.exists(c.coordinate))
                        {
                            graph.mapper.disconnect(load, c.coordinate, c.connection);
                        }
                    }
                }

                // If not using data from the info map, create new nodes,
                // otherwise use the nodes previously created in addLoadThroughLDSToCoordinates.
                int loadTileFromGlobal;
                int internalTile;
                int storeTileIntoLDS;
                if(!origInLoads)
                {
                    internalTile = graph.coordinates.addElement(
                        graph.coordinates.getNode<MacroTile>(localInfo.internalTile));
                    graph.coordinates.addElement(
                        PassThrough(), {internalTile}, {localInfo.internalTile});

                    loadTileFromGlobal = graph.control.addElement(LoadTiled(vtype));
                    for(auto const& c : graph.mapper.getConnections(localInfo.loadTileFromGlobal))
                    {
                        graph.mapper.connect(loadTileFromGlobal, c.coordinate, c.connection);
                    }

                    storeTileIntoLDS = graph.control.addElement(StoreLDSTile(vtype.dataType));
                    for(auto const& c : graph.mapper.getConnections(localInfo.storeTileIntoLDS))
                    {
                        graph.mapper.connect(storeTileIntoLDS, c.coordinate, c.connection);
                    }
                }
                else
                {
                    loadTileFromGlobal = localInfo.loadTileFromGlobal;
                    internalTile       = localInfo.internalTile;
                    storeTileIntoLDS   = localInfo.storeTileIntoLDS;
                }

                graph.mapper.connect<MacroTile>(loadTileFromGlobal, internalTile);
                graph.mapper.connect<User>(loadTileFromGlobal, user);

                graph.mapper.connect<MacroTile>(storeTileIntoLDS, internalTile);
                graph.mapper.connect<LDS>(storeTileIntoLDS, localInfo.lds[unrollIndex]);

                if(localInfo.unrollCoord >= 0)
                {
                    auto setCoordForLoad
                        = graph.control.addElement(SetCoordinate(literal(unrollIndex)));
                    graph.mapper.connect<Unroll>(setCoordForLoad, localInfo.unrollCoord);
                    auto setCoordForStore
                        = graph.control.addElement(SetCoordinate(literal(unrollIndex)));
                    graph.mapper.connect<Unroll>(setCoordForStore, localInfo.unrollCoord);

                    graph.control.addElement(Body(), {forK}, {setCoordForLoad});

                    graph.control.addElement(Body(), {setCoordForLoad}, {loadTileFromGlobal});
                    graph.control.addElement(Body(), {setCoordForStore}, {storeTileIntoLDS});
                    loadTileFromGlobal = setCoordForLoad;
                    storeTileIntoLDS   = setCoordForStore;
                }
                else
                {
                    graph.control.addElement(Body(), {forK}, {loadTileFromGlobal});
                }

                // iteration barrier (right before StoreLDSTile) to ensure that no worker could write into
                // the same portion of LDS while another worker is reading from it in a previous iteration.
                auto iterationBarrier = graph.control.addElement(Barrier());
                graph.control.addElement(Sequence(), {loadTileFromGlobal}, {iterationBarrier});
                graph.control.addElement(Sequence(), {iterationBarrier}, {storeTileIntoLDS});

                auto barrier = graph.control.addElement(Barrier());
                graph.control.addElement(Sequence(), {storeTileIntoLDS}, {barrier});
                for(auto const& loopBody : loopBodies)
                {
                    graph.control.addElement(Sequence(), {barrier}, {loopBody});
                }
            }
        }

        void AddLDSVisitor::addWaveLoadThroughLDSToControlGraph(KernelGraph& graph, int forLoop)
        {
            auto bodies = graph.control.getOutputNodeIndices<Body>(forLoop).to<std::set>();

            // Find any Multiply commands directly under the For Loop.
            // Keep track of their loads and the entire body of the loop
            // that contains the Multiply.

            std::vector<int> loadAs;
            std::vector<int> loadBs;
            std::vector<int> bodiesWithMultiply;
            int              unrollCoord = -1;
            for(auto const& body : bodies)
            {
                // Find all multiplies within the body
                auto allMultiplies
                    = graph.control
                          .findNodes(
                              body,
                              [&](int tag) -> bool {
                                  return isOperation<Multiply>(graph.control.getElement(tag));
                              },
                              Graph::Direction::Downstream)
                          .to<std::vector>();

                if(allMultiplies.empty())
                    continue;

                // Make sure that the inner most loop that the Multiply
                // resides in is the original for loop.
                auto firstMultiply = allMultiplies[0];
                auto allForLoops   = graph.control.findNodes(
                    firstMultiply,
                    [&](int tag) -> bool {
                        return isOperation<ForLoopOp>(graph.control.getElement(tag));
                    },
                    Graph::Direction::Upstream);

                if(*allForLoops.begin() != forLoop)
                    continue;

                bodiesWithMultiply.push_back(body);

                auto loads
                    = graph.control
                          .findNodes(
                              body,
                              [&](int tag) -> bool {
                                  return isOperation<LoadTiled>(graph.control.getElement(tag));
                              },
                              Graph::Direction::Downstream)
                          .to<std::vector>();

                for(auto const& load : loads)
                {
                    auto macroTileTag = graph.mapper.get<MacroTile>(load);
                    auto macroTile    = graph.coordinates.getNode<MacroTile>(macroTileTag);
                    if(macroTile.layoutType == LayoutType::MATRIX_A)
                    {
                        loadAs.push_back(load);
                    }
                    else if(macroTile.layoutType == LayoutType::MATRIX_B)
                    {
                        loadBs.push_back(load);
                    }
                    else
                    {
                        Throw<FatalError>("Unsupported layout type for matrix multiply loads");
                    }
                }

                int origMacroTile                    = -1;
                std::tie(origMacroTile, std::ignore) = getOrigMacTile(graph, loads);
                if(m_load.count(origMacroTile) == 1)
                {
                    unrollCoord = m_load[origMacroTile].unrollCoord;
                }
            }

            if(bodiesWithMultiply.empty())
                return;

            // Call addLoadWaveLDSOps on A and B
            if(unrollCoord > 0)
            {
                auto unrollK        = getUnsignedInt(evaluate(
                    getSize(std::get<Dimension>(graph.coordinates.getElement(unrollCoord)))));
                int  bodiesInUnroll = bodiesWithMultiply.size() / unrollK;
                int  loadsInUnroll  = loadAs.size() / unrollK;
                for(int unroll = 0; unroll < unrollK; unroll++)
                {
                    std::vector<int> unrollBodies;
                    std::vector<int> unrollLoadAs;
                    std::vector<int> unrollLoadBs;
                    for(int i = 0; i < bodiesInUnroll; i++)
                    {
                        unrollBodies.push_back(bodiesWithMultiply[unroll * bodiesInUnroll + i]);
                    }
                    for(int i = 0; i < loadsInUnroll; i++)
                    {
                        unrollLoadAs.push_back(loadAs[unroll * loadsInUnroll + i]);
                        unrollLoadBs.push_back(loadBs[unroll * loadsInUnroll + i]);
                    }
                    addLoadWaveLDSOps(graph, unrollLoadAs, forLoop, unrollBodies, unroll);
                    addLoadWaveLDSOps(graph, unrollLoadBs, forLoop, unrollBodies, unroll);
                }
            }
            else
            {
                addLoadWaveLDSOps(graph, loadAs, forLoop, bodiesWithMultiply, 0);
                addLoadWaveLDSOps(graph, loadBs, forLoop, bodiesWithMultiply, 0);
            }
        }

        void AddLDSVisitor::addWaveLoadThroughLDSToCoordinateGraph(KernelGraph& graph, int load)
        {
            ldsLoadInfo localInfo;
            auto        loadTile  = graph.mapper.get<MacroTile>(load);
            auto        macrotile = graph.coordinates.getNode<MacroTile>(loadTile);

            if(macrotile.memoryType == MemoryType::WAVE_LDS)
            {
                auto vtype = graph.control.getNode<LoadTiled>(load).vtype;
                int  user  = graph.mapper.get<User>(load);
                auto sdims = graph.coordinates.getOutputNodeIndices(user, CT::isEdge<Split>)
                                 .to<std::vector>();

                localInfo.lds.push_back(graph.coordinates.addElement(LDS()));

                // Find K dimension
                auto allForLoops = graph.control.findNodes(
                    load,
                    [&](int tag) -> bool {
                        return isOperation<ForLoopOp>(graph.control.getElement(tag));
                    },
                    Graph::Direction::Upstream);
                auto forK        = *allForLoops.begin();
                auto connections = graph.mapper.getConnections(forK);
                AssertFatal(connections.size() == 1);
                auto loop_incr_tag = connections[0].coordinate;
                auto loopDims = graph.coordinates.getOutputNodeIndices<DataFlowEdge>(loop_incr_tag)
                                    .to<std::vector>();
                AssertFatal(loopDims.size() == 1);
                auto K = loopDims[0];

                auto useSwappedAccess
                    = m_context->kernelOptions().transposeMemoryAccess[macrotile.layoutType];

                // remove workgroups, macrotile numbers and tile edges from sdims
                updateLoadLDSMacroTile(
                    graph, macrotile, load, sdims, K, localInfo.lds[0], useSwappedAccess);

                // create an internal macrotile to be loaded by one workgroup
                auto workgroupSizes         = m_context->kernel()->workgroupSize();
                auto numWorkitems           = product(workgroupSizes);
                auto numElements            = product(macrotile.sizes);
                auto numElementsPerWorkitem = static_cast<int>(numElements / numWorkitems);
                auto thrTileM               = numElementsPerWorkitem;
                auto thrTileN               = 1;

                // load multiple smaller-precision(< 32-bit) elements into 1 VGPR
                auto packFactor = bytesPerRegister / DataTypeInfo::Get(vtype).elementSize;
                bool packed     = false;
                if(m_context->kernelOptions().packMultipleElementsInto1VGPR && packFactor > 1
                   && thrTileM % packFactor == 0)
                {
                    thrTileM = thrTileM / packFactor;
                    thrTileN = packFactor;

                    packed = true;
                }

                // enable the use of longer word instructions if possible
                if(m_context->kernelOptions().enableLongDwordInstructions
                   && (packed || packFactor <= 1))
                {
                    auto maxWidth = std::min(m_context->kernelOptions().loadGlobalWidth,
                                             m_context->kernelOptions().storeLocalWidth);

                    auto numDwordsPerElement
                        = std::max(1LU, DataTypeInfo::Get(vtype).elementSize / bytesPerRegister);

                    updateThreadTileForLongDwords(
                        thrTileM, thrTileN, maxWidth, numDwordsPerElement);
                }

                if(!useSwappedAccess)
                    std::swap(thrTileM, thrTileN);

                localInfo.internalTile = graph.coordinates.addElement(
                    MacroTile(macrotile.sizes, MemoryType::VGPR, {thrTileM, thrTileN}));
                auto internalTileDim = graph.coordinates.getNode<MacroTile>(localInfo.internalTile);
                internalTileDim.layoutType = macrotile.layoutType;
                graph.coordinates.setElement(localInfo.internalTile, internalTileDim);

                // user --DataFlow--> internalTile
                graph.coordinates.addElement(DataFlow(), {user}, {localInfo.internalTile});
                localInfo.loadTileFromGlobal = graph.control.addElement(LoadTiled(vtype));
                graph.mapper.connect<User>(localInfo.loadTileFromGlobal, user);

                // Find the unroll coordinate tag from the K for loop.
                localInfo.unrollCoord = -1;
                auto incomingEdges = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(K)
                                         .to<std::vector>();
                auto splitEdges = filterCoordinates<Split>(incomingEdges, graph);
                for(auto& edgeTag : splitEdges)
                {
                    auto upstreamNodes
                        = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(edgeTag)
                              .to<std::vector>();
                    if(upstreamNodes.size() == 0)
                    {
                        auto outgoingNodes
                            = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(edgeTag)
                                  .to<std::vector>();
                        auto unrollCoordinates = filterCoordinates<Unroll>(outgoingNodes, graph);
                        AssertFatal(unrollCoordinates.size() == 1);
                        localInfo.unrollCoord = *unrollCoordinates.begin();
                        graph.coordinates.deleteElement(edgeTag);
                    }
                }

                if(localInfo.unrollCoord >= 0)
                {
                    auto unrollK = getUnsignedInt(evaluate(getSize(
                        std::get<Dimension>(graph.coordinates.getElement(localInfo.unrollCoord)))));
                    for(int i = 1; i < unrollK; i++)
                    {
                        localInfo.lds.push_back(graph.coordinates.addElement(LDS()));
                        graph.coordinates.addElement(
                            PassThrough(), {localInfo.lds[0]}, {localInfo.lds[i]});
                    }
                }

                // lower tile LoadTiled : load macrotile from global memory
                loadMacroTileForLDS(graph,
                                    localInfo.loadTileFromGlobal,
                                    user,
                                    localInfo.internalTile,
                                    sdims,
                                    K,
                                    workgroupSizes,
                                    localInfo.unrollCoord,
                                    useSwappedAccess);

                localInfo.storeTileIntoLDS = graph.control.addElement(StoreLDSTile(vtype.dataType));
                // lower tile StoreLDSTile : store macrotile into LDS
                storeMacroTileIntoLDS(graph,
                                      localInfo.storeTileIntoLDS,
                                      localInfo.lds[0],
                                      localInfo.internalTile,
                                      workgroupSizes,
                                      useSwappedAccess);
                graph.coordinates.deleteElement(
                    std::vector<int>{user}, std::vector<int>{loadTile}, CT::isEdge<DataFlow>);
                graph.coordinates.addElement(DataFlow(), {localInfo.lds[0]}, {loadTile});
                m_load[loadTile] = localInfo;
            }
        }

        void AddLDSVisitor::addStoreThroughLDSToControlGraph(KernelGraph& graph,
                                                             int          storeTag,
                                                             int          forLoop)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::AddLDSVisitor::addStoreThroughLDSToControlGraph({})", storeTag);

            auto [macroTileTag, macroTile] = graph.getDimension<MacroTile>(storeTag);
            auto [userTag, user]           = graph.getDimension<User>(storeTag);
            if(macroTile.memoryType != MemoryType::WAVE_LDS)
                return;

            int ldsTag = -1;
            for(auto tag :
                graph.coordinates.depthFirstVisit(macroTileTag, Graph::Direction::Downstream))
            {
                if(graph.coordinates.get<LDS>(tag))
                    ldsTag = tag;
            }
            AssertFatal(ldsTag != -1);
            // change StoreTiled to StoreLDSTile
            auto info = m_store[ldsTag];
            // and update its macrotile's memory type
            // Change StoreTiled to StoreLDSTile
            auto dtype = graph.control.getNode<StoreTiled>(storeTag).dataType;

            graph.control.setElement(storeTag, StoreLDSTile(dtype));
            graph.mapper.disconnect<User>(storeTag, userTag);

            // Update its macroTile's memory type
            macroTile.memoryType = MemoryType::WAVE;
            graph.coordinates.setElement(macroTileTag, macroTile);

            auto storeDBarrierRW = graph.control.addElement(Barrier());
            // Find all incoming edges into StoreLDSTile.
            // Those should be changed to come into Barrier to avoid RW hazard.
            auto incoming = graph.control.getNeighbours<Graph::Direction::Upstream>(storeTag)
                                .to<std::vector>();
            for(auto e : incoming)
            {
                auto elem = graph.control.getElement(e);
                auto src
                    = graph.control.getNeighbours<Graph::Direction::Upstream>(e).to<std::vector>();
                graph.control.deleteElement(e);
                graph.control.addElement(e, elem, src, std::vector<int>{storeDBarrierRW});
            }
            graph.control.addElement(Sequence(), {storeDBarrierRW}, {storeTag});

            graph.mapper.connect<LDS>(storeTag, ldsTag);

            auto barrier = graph.control.addElement(Barrier());
            graph.control.addElement(Sequence(), {storeTag}, {barrier});

            // At this point we have added operations that store VGPRs
            // into LDS.
            //
            // If we have added the operations that store from LDS to
            // global for this LDS allocation already, we're done.
            if(m_store[ldsTag].storeOperation)
            {
                return;
            }

            auto loadMacroTileFromLDSNode
                = graph.control.addElement(LoadLDSTile(VariableType(dtype)));
            graph.control.addElement(Sequence(), {forLoop}, {loadMacroTileFromLDSNode});

            graph.mapper.connect<LDS>(loadMacroTileFromLDSNode, ldsTag);
            graph.mapper.connect<MacroTile>(loadMacroTileFromLDSNode, info.internalTile);
            for(auto dc : info.loadConnections)
                graph.mapper.connect(loadMacroTileFromLDSNode, dc.coordinate, dc.connectionSpec);

            auto storeMacroTileIntoGlobal = graph.control.addElement(StoreTiled(dtype));
            graph.control.addElement(
                Sequence(), {loadMacroTileFromLDSNode}, {storeMacroTileIntoGlobal});

            graph.coordinates.addElement(DataFlow(), {info.internalTile}, {userTag});
            // add new loadLDSTile node to load a macrotile into VGPRs from LDS
            graph.mapper.connect<User>(storeMacroTileIntoGlobal, userTag);
            graph.mapper.connect<MacroTile>(storeMacroTileIntoGlobal, info.internalTile);
            for(auto dc : info.storeConnections)
                graph.mapper.connect(storeMacroTileIntoGlobal, dc.coordinate, dc.connectionSpec);

            m_store[ldsTag].storeOperation = storeMacroTileIntoGlobal;
        }

        void AddLDSVisitor::addStoreThroughLDSToCoordinateGraph(KernelGraph& graph, int store)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::AddLDSVisitor::addStoreThroughLDSToCoordinateGraph");
            // create an internal macrotile to be loaded by one workgroup
            auto [macroTileTag, macroTile] = graph.getDimension<MacroTile>(store);
            if(macroTile.memoryType != MemoryType::WAVE_LDS)
                return;

            int user
                = *only(graph.coordinates.getOutputNodeIndices(macroTileTag, CT::isEdge<DataFlow>));

            graph.coordinates.deleteElement(
                std::vector<int>{macroTileTag}, std::vector<int>{user}, CT::isEdge<DataFlow>);
            auto sdims
                = graph.coordinates.getInputNodeIndices(user, CT::isEdge<Join>).to<std::vector>();
            AssertFatal(sdims.size() > 1);

            auto lds   = graph.coordinates.addElement(LDS());
            auto dtype = graph.control.getNode<StoreTiled>(store).dataType;

            // remove workgroups, macrotile numbers and tile edges from sdims
            updateStoreLDSMacroTile(graph, macroTile, store, sdims, lds);

            // macrotile --DataFlow--> LDS
            graph.coordinates.addElement(DataFlow(), {macroTileTag}, {lds});
            graph.mapper.connect<LDS>(store, lds);

            // create an internal macrotile to be loaded by one workgroup
            auto workgroupSizes         = m_context->kernel()->workgroupSize();
            auto numWorkitems           = product(workgroupSizes);
            auto numElements            = product(macroTile.sizes);
            auto numElementsPerWorkitem = static_cast<int>(numElements / numWorkitems);
            auto thrTileM               = numElementsPerWorkitem;
            auto thrTileN               = 1;

            // load multiple smaller-precision(< 32-bit) elements into 1 VGPR
            auto packFactor = bytesPerRegister / DataTypeInfo::Get(dtype).elementSize;
            bool packed     = false;
            if(m_context->kernelOptions().packMultipleElementsInto1VGPR && packFactor > 1
               && thrTileM % packFactor == 0)
            {
                thrTileM = thrTileM / packFactor;
                thrTileN = packFactor;

                packed = true;
            }

            // enable the use of longer word instructions if possible
            if(m_context->kernelOptions().enableLongDwordInstructions
               && (packed || packFactor <= 1))
            {
                auto maxWidth = std::min(m_context->kernelOptions().storeGlobalWidth,
                                         m_context->kernelOptions().loadLocalWidth);

                auto numDwordsPerElement
                    = std::max(1LU, DataTypeInfo::Get(dtype).elementSize / bytesPerRegister);

                updateThreadTileForLongDwords(thrTileM, thrTileN, maxWidth, numDwordsPerElement);
            }

            auto internalTile = MacroTile(macroTile.sizes, MemoryType::VGPR, {thrTileM, thrTileN});
            internalTile.layoutType = macroTile.layoutType;

            auto internalTileTag = graph.coordinates.addElement(internalTile);
            graph.coordinates.addElement(DataFlow(), {internalTileTag}, {user});

            ldsStoreInfo info;
            info.internalTile = internalTileTag;
            info.loadConnections
                = loadMacroTileFromLDS(graph, lds, internalTileTag, workgroupSizes);
            info.storeConnections
                = storeMacroTileForLDS(graph, user, internalTileTag, sdims, workgroupSizes);
            m_store[lds] = info;
        }

    }
}
