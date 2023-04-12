/**
 * AddLDS -- add load/store through LDS to the graph.
 *
 * Load/store operations inside loops, and tagged with MemoryType LDS
 * or WAVE_LDS, are transformed.
 *
 * An entire tile is loaded once per loop iteration (which may be
 * unrolled) into LDS.  Subsequent loads in the loop read from LDS.
 *
 * A unique LDS allocation is required for each: User, ForLoop, and
 * Unroll.  This is encapsulated by the LDSSpec struct (see below).
 *
 * Transformations are done using a "stage and commit" approach:
 *
 * During staging, we search for all Load/Store operations tagged for
 * LDS and:
 *
 * 1. Compute the LDS specifier.
 *
 * 2. Record all transformations that need to happen.  These include:
 *
 *    a. Creating a unique LDS storage node.
 *
 *    b. Updating the coordinate-transform graph to compute LDS buffer
 *       indexes.
 *
 *    c. Adding load-from-global and store-into-lds operations.
 *
 *    d. Transforming existing load operations into load-from-lds
 *       operations.
 *
 *    e. Adding appropriate synchronisation (barrier) operations.
 *
 * After staging, we know how the graph must change ahead-of-time; and
 * we commit our changes.
 *
 * Note that coordinate transforms for LDS index calculations are more
 * generic than the LDS allocations for load/store operations.  For
 * example, the coordinate transform might contain ForLoop and Unroll
 * dimensions, but which coordinate transform to use doesn't depend on
 * their values.  This is directly related to the PassThrough edges
 * that we add between LDS nodes: the transform is the same, but the
 * storage location is different.  In other words, the transform is
 * less specific than the storage node.
 */

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

        /*
         * Helpers
         */

        /**
         * @brief Return Unroll coordinate beside (as part of a Split
         * edge) the ForLoop coordinate.
         */
        std::optional<int> findUnrollNeighbour(KernelGraph const& kgraph, int forLoopCoord)
        {
            if(forLoopCoord < 0)
                return {};

            std::optional<int> rv;

            auto forNeighbours
                = kgraph.coordinates.getNeighbours<Graph::Direction::Upstream>(forLoopCoord)
                      .to<std::vector>();
            for(auto forNeighbour : forNeighbours)
            {
                auto split = kgraph.coordinates.get<Split>(forNeighbour);
                if(split)
                {
                    auto splitNeighbours
                        = kgraph.coordinates
                              .getNeighbours<Graph::Direction::Downstream>(forNeighbour)
                              .to<std::vector>();
                    for(auto splitNeighbour : splitNeighbours)
                    {
                        auto unroll = kgraph.coordinates.get<Unroll>(splitNeighbour);
                        if(unroll)
                        {
                            AssertFatal(!rv || rv == splitNeighbour,
                                        "More than one Unroll neighbour found.");
                            rv = splitNeighbour;
                        }
                    }
                }
            }

            return rv;
        }

        /**
         * @brief LDS specification.
         *
         * This uniquely identifies an LDS allocation.
         */
        struct LDSSpec
        {
            int userCoord;
            int forLoopCoord;
            int unrollCoord;
            int unrollCoordValue;
            int operation;

            VariableType variableType;
            MemoryType   memoryType;

            /**
             * @brief Return a specifier unique to each coordinate
             * transform required to compute indexes for this LDS
             * allocation.
             *
             * Recall that which coordinate transform to use is less
             * specific than which LDS allocation to use.  The
             * coordinate transform depends on the User, ForLoop, and
             * Unroll etc coordinates; but not on the ForLoopOp and/or
             * Unroll coordinate value.
             */
            LDSSpec forCoordinateTransform() const
            {
                return {userCoord, forLoopCoord, unrollCoord, -1, -1, variableType, memoryType};
            }
        };

        bool operator<(LDSSpec const& a, LDSSpec const& b)
        {
            int aDataType   = static_cast<int>(a.variableType.dataType);
            int bDataType   = static_cast<int>(b.variableType.dataType);
            int aMemoryType = static_cast<int>(a.memoryType);
            int bMemoryType = static_cast<int>(b.memoryType);
            return std::tie(a.userCoord,
                            a.forLoopCoord,
                            a.unrollCoord,
                            a.unrollCoordValue,
                            a.operation,
                            aDataType,
                            aMemoryType)
                   < std::tie(b.userCoord,
                              b.forLoopCoord,
                              b.unrollCoord,
                              b.unrollCoordValue,
                              b.operation,
                              bDataType,
                              bMemoryType);
        }

        bool operator==(LDSSpec const& a, LDSSpec const& b)
        {
            int aDataType   = static_cast<int>(a.variableType.dataType);
            int bDataType   = static_cast<int>(b.variableType.dataType);
            int aMemoryType = static_cast<int>(a.memoryType);
            int bMemoryType = static_cast<int>(b.memoryType);
            return std::tie(a.userCoord,
                            a.forLoopCoord,
                            a.unrollCoord,
                            a.unrollCoordValue,
                            a.operation,
                            aDataType,
                            aMemoryType)
                   == std::tie(b.userCoord,
                               b.forLoopCoord,
                               b.unrollCoord,
                               b.unrollCoordValue,
                               b.operation,
                               bDataType,
                               bMemoryType);
        }

        /**
         * @brief Return LDS specifier for the load operation.
         *
         * This inspects the graph and figures out the User, ForLoop,
         * Unroll etc coordinates that determine which LDS buffer the
         * load operation will populate.
         *
         * When determining the specific ForLoop and Unroll
         * coordinates, only the containing for loop is considered.
         *
         * The containing loop's tag is also used to: determine where
         * LDS is populated, and differentiate the LDS allocations.
         * This means jammed loops will use unique LDS allocations.
         *
         * If there is no containing ForLoop, the location of the
         * original load is used to: determine where LDS is populated,
         * and differentiate the LDS allocations.
         */
        LDSSpec getLDSSpec(KernelGraph const& k, int loadTag)
        {
            auto [userTag, user]           = k.getDimension<User>(loadTag);
            auto [macroTileTag, macroTile] = k.getDimension<MacroTile>(loadTag);

            auto [target, direction] = getOperationTarget(loadTag, k);
            auto [required, path]    = findRequiredCoordinates(target, direction, k);
            auto forLoopCoordinates  = filterCoordinates<ForLoop>(required, k);

            auto maybeForLoop = findContainingOperation<ForLoopOp>(loadTag, k);
            int  forLoopCoord = -1;
            int  operation;
            if(maybeForLoop)
            {
                operation = *maybeForLoop;
                auto f    = getForLoop(*maybeForLoop, k);
                if(forLoopCoordinates.contains(f))
                {
                    forLoopCoord = f;
                }
            }
            else
            {
                operation = loadTag;
            }

            auto maybeUnroll      = findUnrollNeighbour(k, forLoopCoord);
            int  unrollCoord      = -1;
            int  unrollCoordValue = -1;
            if(maybeUnroll)
            {
                unrollCoord = *maybeUnroll;

                auto setCoord
                    = k.control.get<SetCoordinate>(getSetCoordinateForDim(k, unrollCoord, loadTag));
                AssertFatal(evaluationTimes(
                                setCoord->value)[rocRoller::Expression::EvaluationTime::Translate],
                            "Unroll value should be a literal");

                unrollCoordValue = getUnsignedInt(evaluate(setCoord->value));
            }

            auto vtype = k.control.getNode<LoadTiled>(loadTag).vtype;

            return {userTag,
                    forLoopCoord,
                    unrollCoord,
                    unrollCoordValue,
                    operation,
                    vtype,
                    macroTile.memoryType};
        }

        /**
         * @brief Container for info related to loading from Global
         * into LDS.
         */
        struct ldsLoadInfo
        {
            int lds; // LDS allocation (coordinate)
            int internalTile; // Internal/intermediate VGPR MacroTile
            int loadTileFromGlobal; // LoadTiled operation
            int storeTileIntoLDS; // StoreLDStile operation
            int loadChain; // LoadTiled operation
            int storeChain; // StoreLDStile operation
        };

        struct ldsLoadConnections
        {
            int lds;
            int internalTile;

            std::vector<DeferredConnection> loadConnections;
            std::vector<DeferredConnection> storeConnections;
        };

        /**
         * @brief Container for info related to storing from LDS to
         * Global.
         */
        struct ldsStoreInfo
        {
            int                             internalTile;
            std::vector<DeferredConnection> loadConnections;
            std::vector<DeferredConnection> storeConnections;
            std::optional<int>              storeOperation;
        };

        /**
         * Add LDS transformer.
         */
        struct AddLDSVisitor
        {
            AddLDSVisitor(std::shared_ptr<Context> context)
                : m_context(context)
            {
            }

            ldsLoadConnections addVGPRLoadCoordinates(KernelGraph& graph, int loadTag) const;
            ldsLoadConnections addWAVELoadCoordinates(KernelGraph& graph, int loadTag) const;

            void addLoadOperations(KernelGraph& graph);

            void addStoreThroughLDSToControlGraph(KernelGraph& graph, int store);
            void addStoreThroughLDSToCoordinateGraph(KernelGraph& graph, int store);

            void stageLoad(KernelGraph const&, int loadTag);
            void commit(KernelGraph&);

        private:
            std::map<int, ldsStoreInfo> m_store;

            std::set<LDSSpec>              m_loadSpecs;
            std::map<LDSSpec, ldsLoadInfo> m_loadInfo;
            std::map<int, LDSSpec>         m_tileSpecs;
            std::map<int, LDSSpec>         m_stagedLoads;
            std::map<LDSSpec, int>         m_stagedCoordinates;

            ContextPtr m_context;
        };

        KernelGraph addLDS(KernelGraph const& original, std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::addLDS");
            rocRoller::Log::getLogger()->debug("KernelGraph::addLDS()");

            auto k       = original;
            auto visitor = AddLDSVisitor(context);

            // Add LDS operations
            for(auto const& loadTag : k.control.getNodes<LoadTiled>())
            {
                visitor.stageLoad(k, loadTag);
            }

            visitor.commit(k);

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
                visitor.addStoreThroughLDSToControlGraph(k, store);
            }

            return k;
        }

        /*
         * Stage everything for a load.  Does not modify the graph.
         */
        void AddLDSVisitor::stageLoad(KernelGraph const& k, int loadTag)
        {
            auto [userTag, user] = k.getDimension<User>(loadTag);
            auto [tileTag, tile] = k.getDimension<MacroTile>(loadTag);

            if(!(tile.memoryType == MemoryType::WAVE_LDS || tile.memoryType == MemoryType::LDS))
                return;

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::AddLDS()::stageLoad({}): User {}, MacroTile {}",
                loadTag,
                userTag,
                tileTag);

            auto spec = getLDSSpec(k, loadTag);

            //
            // Stage: create unique LDS allocation
            //
            m_loadSpecs.insert(spec);

            //
            // Stage: create coordinate transform
            //
            auto maybeParentTile
                = only(k.coordinates.getOutputNodeIndices(tileTag, CT::isEdge<PassThrough>));
            if(!maybeParentTile)
            {
                m_stagedCoordinates[spec.forCoordinateTransform()] = loadTag;
            }

            //
            // Stage: convert LoadTile to LoadLDSTile
            //
            m_stagedLoads[loadTag] = spec;
        }

        /*
         * Commit everything.  Modifies the graph.
         */
        void AddLDSVisitor::commit(KernelGraph& k)
        {
            rocRoller::Log::getLogger()->debug("KernelGraph::AddLDS()::commit()");

            AssertFatal(m_loadSpecs.size() >= m_stagedCoordinates.size());

            //
            // Commit: Create LDS nodes, internal tiles, and load/store operations.
            //
            for(auto spec : m_loadSpecs)
            {
                auto userTag            = spec.userCoord;
                auto ldsTag             = k.coordinates.addElement(LDS());
                auto internalTileTag    = k.coordinates.addElement(MacroTile());
                auto loadTileFromGlobal = k.control.addElement(LoadTiled(spec.variableType));
                auto storeTileIntoLDS
                    = k.control.addElement(StoreLDSTile(spec.variableType.dataType));

                k.mapper.connect<MacroTile>(loadTileFromGlobal, internalTileTag);
                k.mapper.connect<User>(loadTileFromGlobal, userTag);

                k.mapper.connect<MacroTile>(storeTileIntoLDS, internalTileTag);
                k.mapper.connect<LDS>(storeTileIntoLDS, ldsTag);

                auto loadChain  = loadTileFromGlobal;
                auto storeChain = storeTileIntoLDS;

                if(spec.unrollCoord >= 0)
                {
                    auto setCoordForLoad
                        = k.control.addElement(SetCoordinate(literal(spec.unrollCoordValue)));
                    k.mapper.connect<Unroll>(setCoordForLoad, spec.unrollCoord);
                    auto setCoordForStore
                        = k.control.addElement(SetCoordinate(literal(spec.unrollCoordValue)));
                    k.mapper.connect<Unroll>(setCoordForStore, spec.unrollCoord);

                    k.control.addElement(Body(), {setCoordForLoad}, {loadTileFromGlobal});
                    k.control.addElement(Body(), {setCoordForStore}, {storeTileIntoLDS});
                    loadChain  = setCoordForLoad;
                    storeChain = setCoordForStore;
                }

                m_loadInfo[spec] = {ldsTag,
                                    internalTileTag,
                                    loadTileFromGlobal,
                                    storeTileIntoLDS,
                                    loadChain,
                                    storeChain};
            }

            for(auto [loadTag, loadSpec] : m_stagedLoads)
            {
                auto macroTileTag         = k.mapper.get<MacroTile>(loadTag);
                m_tileSpecs[macroTileTag] = getLDSSpec(k, loadTag);
            }

            // At this point: LDS nodes, internal tiles, and
            // load/store operations have been added, but they aren't
            // connected (through the mapper nor in the graphs).

            //
            // Commit: Update coordinate graph
            //
            std::map<LDSSpec, ldsLoadConnections> loadBySpec;

            for(auto [specCT, loadTag] : m_stagedCoordinates)
            {
                switch(specCT.memoryType)
                {
                case MemoryType::WAVE_LDS:
                    // Add/update coordinate transforms:
                    // 1. User to internal VGPR MacroTile
                    // 2. Internal VGPR MacroTile to LDS
                    // 3. LDS to WaveTile
                    loadBySpec[specCT] = addWAVELoadCoordinates(k, loadTag);
                    break;
                case MemoryType::LDS:
                    loadBySpec[specCT] = addVGPRLoadCoordinates(k, loadTag);
                    break;
                default:
                    break;
                }
            }

            //
            // Commit: Apply deferred connections, attach storage nodes via PassThrough
            //
            for(auto spec : m_loadSpecs)
            {
                auto lInfo = m_loadInfo.at(spec);
                auto oInfo = loadBySpec.at(spec.forCoordinateTransform());

                for(auto dc : oInfo.loadConnections)
                {
                    k.mapper.connect(lInfo.loadTileFromGlobal, dc.coordinate, dc.connectionSpec);
                }

                for(auto dc : oInfo.storeConnections)
                {
                    k.mapper.connect(lInfo.storeTileIntoLDS, dc.coordinate, dc.connectionSpec);
                }

                if(lInfo.internalTile != oInfo.internalTile)
                {
                    k.coordinates.setElement(lInfo.internalTile,
                                             k.coordinates.getNode<MacroTile>(oInfo.internalTile));
                    k.coordinates.addElement(
                        PassThrough(), {lInfo.internalTile}, {oInfo.internalTile});
                }

                if(lInfo.lds != oInfo.lds)
                {
                    k.coordinates.addElement(PassThrough(), {oInfo.lds}, {lInfo.lds});
                }
            }

            // At this point: LDS nodes, internal tiles, and
            // load/store operations have been added.  Operations
            // are connected to their coordinate nodes, but they
            // haven't been inserted into the control grpah yet.

            //
            // Commit: Connect load/store operations in the control graph.
            //
            addLoadOperations(k);

            //
            // Commit: Change all LoadTiled operations to LoadLDSTile
            //
            for(auto [loadTag, loadSpec] : m_stagedLoads)
            {
                auto [macroTileTag, macroTile] = k.getDimension<MacroTile>(loadTag);

                auto vtype = k.control.getNode<LoadTiled>(loadTag).vtype;
                if(macroTile.memoryType == MemoryType::WAVE_LDS)
                    macroTile.memoryType = MemoryType::WAVE;
                else
                    macroTile.memoryType = MemoryType::VGPR;

                k.control.setElement(loadTag, LoadLDSTile(vtype));
                k.coordinates.setElement(k.mapper.get<MacroTile>(loadTag), macroTile);
                k.mapper.connect<LDS>(loadTag, m_loadInfo.at(loadSpec).lds);
                for(auto const& c : k.mapper.getConnections(loadTag))
                {
                    if(!k.coordinates.exists(c.coordinate))
                    {
                        k.mapper.disconnect(loadTag, c.coordinate, c.connection);
                    }
                }
            }
        }

        //
        // Rework this to stage+commit workflow
        //
        ldsLoadConnections AddLDSVisitor::addVGPRLoadCoordinates(KernelGraph& graph,
                                                                 int          loadTag) const
        {
            auto userTag = graph.mapper.get<User>(loadTag);
            auto tileTag = graph.mapper.get<MacroTile>(loadTag);
            auto tile    = graph.coordinates.getNode<MacroTile>(tileTag);
            auto load    = graph.control.getNode<LoadTiled>(loadTag);

            AssertFatal(tile.memoryType == MemoryType::LDS);

            graph.coordinates.deleteElement(
                std::vector<int>{userTag}, std::vector<int>{tileTag}, CT::isEdge<DataFlow>);
            auto sdims = graph.coordinates.getOutputNodeIndices(userTag, CT::isEdge<Split>)
                             .to<std::vector>();

            auto ldsTag = m_loadInfo.at(m_tileSpecs.at(tileTag)).lds;

            // remove workgroups, macrotile numbers and tile edges from sdims
            updateLoadLDSMacroTile(graph, tile, loadTag, sdims, -1, ldsTag, true);

            // create an internal macrotile to be loaded by one workgroup
            auto workgroupSizes  = m_context->kernel()->workgroupSize();
            auto internalTileTag = m_loadInfo.at(m_tileSpecs.at(tileTag)).internalTile;
            auto internalTile    = MacroTile(tile.sizes, MemoryType::VGPR, tile.subTileSizes);
            graph.coordinates.setElement(internalTileTag, internalTile);

            // user --DataFlow--> internalTile
            graph.coordinates.addElement(DataFlow(), {userTag}, {internalTileTag});

            // lower tile LoadTiled : load macrotile from global memory
            auto loadConnections = loadMacroTileForLDS(
                graph, userTag, internalTileTag, sdims, -1, workgroupSizes, -1, true);

            // lower tile StoreLDSTile : store macrotile into LDS
            auto storeConnections
                = storeMacroTileIntoLDS(graph, ldsTag, internalTileTag, workgroupSizes, true);

            // LDS --DataFlow--> macrotile
            graph.coordinates.addElement(DataFlow(), {ldsTag}, {tileTag});

            return {ldsTag, internalTileTag, loadConnections, storeConnections};
        }

        void addLoadOperationsNoPrefetch(KernelGraph&         graph,
                                         int                  loadTileFromGlobalChain,
                                         int                  storeTileIntoLDSChain,
                                         int                  operation,
                                         std::set<int> const& dependencies)
        {
            // Iteration barrier (right before StoreLDSTile) to ensure
            // that no worker could write into the same portion of LDS
            // while another worker is reading from it in a previous
            // iteration.
            auto iterationBarrier = graph.control.addElement(Barrier());
            graph.control.addElement(Sequence(), {loadTileFromGlobalChain}, {iterationBarrier});
            graph.control.addElement(Sequence(), {iterationBarrier}, {storeTileIntoLDSChain});

            auto barrier = graph.control.addElement(Barrier());
            graph.control.addElement(Sequence(), {storeTileIntoLDSChain}, {barrier});

            auto maybeForLoop = graph.control.get<ForLoopOp>(operation);
            if(maybeForLoop)
            {
                for(auto const& depdency : dependencies)
                {
                    auto edge
                        = *only(graph.control.getNeighbours<Graph::Direction::Upstream>(depdency));
                    graph.control.deleteElement(edge);
                    graph.control.addElement(Sequence(), {barrier}, {depdency});
                }

                graph.control.addElement(Body(), {operation}, {loadTileFromGlobalChain});
            }
            else
            {
                insertBefore(graph, operation, loadTileFromGlobalChain, barrier);
            }
        }

        /*
         * Splice load-from-global and store-to-lds operations into
         * the control graph.
         *
         * Adds synchronisation as necessary.
         */
        void AddLDSVisitor::addLoadOperations(KernelGraph& graph)
        {
            for(auto spec : m_loadSpecs)
            {
                std::vector<int> loads;
                for(auto [loadTag, loadSpec] : m_stagedLoads)
                {
                    if(loadSpec == spec)
                        loads.push_back(loadTag);
                }

                auto dependencies = getTopSetCoordinates(graph, loads);
                auto info         = m_loadInfo.at(spec);
                addLoadOperationsNoPrefetch(
                    graph, info.loadChain, info.storeChain, spec.operation, dependencies);
            }
        }

        /*
         * Update the coordinate transform graph to compute LDS indexes.
         *
         * DO NOT MODIFY THE CONTROL GRAPH.
         *
         * Passing a loadTag here might seem non-intuitive.  It is
         * used to look up mappings etc.
         */
        ldsLoadConnections AddLDSVisitor::addWAVELoadCoordinates(KernelGraph& graph,
                                                                 int          loadTag) const
        {
            auto [macroTileTag, macroTile] = graph.getDimension<MacroTile>(loadTag);

            AssertFatal(macroTile.memoryType == MemoryType::WAVE_LDS);

            auto vtype = graph.control.getNode<LoadTiled>(loadTag).vtype;
            int  user  = graph.mapper.get<User>(loadTag);
            auto sdims
                = graph.coordinates.getOutputNodeIndices(user, CT::isEdge<Split>).to<std::vector>();

            auto maybeForLoop = findContainingOperation<ForLoopOp>(loadTag, graph);
            AssertFatal(maybeForLoop, "Unable to find containing ForLoop");
            auto forLoopCoord = getForLoop(*maybeForLoop, graph);

            auto maybeUnroll = findUnrollNeighbour(graph, forLoopCoord);
            int  unrollCoord = maybeUnroll.value_or(-1);

            auto ldsTag = m_loadInfo.at(m_tileSpecs.at(macroTileTag)).lds;

            auto useSwappedAccess
                = m_context->kernelOptions().transposeMemoryAccess[macroTile.layoutType];

            // Remove Workgroups, MacroTileNumbers, and Tile edges from sdims
            updateLoadLDSMacroTile(
                graph, macroTile, loadTag, sdims, forLoopCoord, ldsTag, useSwappedAccess);

            // Create an internal MacroTile to be loaded by one workgroup
            auto workgroupSizes         = m_context->kernel()->workgroupSize();
            auto numWorkitems           = product(workgroupSizes);
            auto numElements            = product(macroTile.sizes);
            auto numElementsPerWorkitem = static_cast<int>(numElements / numWorkitems);
            auto thrTileM               = numElementsPerWorkitem;
            auto thrTileN               = 1;

            // Load multiple smaller-precision(< 32-bit) elements into 1 VGPR
            auto packFactor = bytesPerRegister / DataTypeInfo::Get(vtype).elementSize;
            bool packed     = false;
            if(m_context->kernelOptions().packMultipleElementsInto1VGPR && packFactor > 1
               && thrTileM % packFactor == 0)
            {
                thrTileM = thrTileM / packFactor;
                thrTileN = packFactor;

                packed = true;
            }

            // Enable the use of longer word instructions if possible
            if(m_context->kernelOptions().enableLongDwordInstructions
               && (packed || packFactor <= 1))
            {
                auto maxWidth = std::min(m_context->kernelOptions().loadGlobalWidth,
                                         m_context->kernelOptions().storeLocalWidth);

                auto numDwordsPerElement
                    = std::max(1LU, DataTypeInfo::Get(vtype).elementSize / bytesPerRegister);

                updateThreadTileForLongDwords(thrTileM, thrTileN, maxWidth, numDwordsPerElement);
            }

            if(!useSwappedAccess)
                std::swap(thrTileM, thrTileN);

            auto internalTileTag = m_loadInfo.at(m_tileSpecs.at(macroTileTag)).internalTile;
            auto internalTile = MacroTile(macroTile.sizes, MemoryType::VGPR, {thrTileM, thrTileN});
            internalTile.layoutType = macroTile.layoutType;
            graph.coordinates.setElement(internalTileTag, internalTile);

            // DataFlow
            graph.coordinates.addElement(DataFlow(), {user}, {internalTileTag});

            // lower tile LoadTiled : load macrotile from global memory
            auto loadConnections = loadMacroTileForLDS(graph,
                                                       user,
                                                       internalTileTag,
                                                       sdims,
                                                       forLoopCoord,
                                                       workgroupSizes,
                                                       unrollCoord,
                                                       useSwappedAccess);

            loadConnections.push_back(DC<User>(user));

            // lower tile StoreLDSTile : store macrotile into LDS
            auto storeConnections = storeMacroTileIntoLDS(
                graph, ldsTag, internalTileTag, workgroupSizes, useSwappedAccess);
            graph.coordinates.deleteElement(
                std::vector<int>{user}, std::vector<int>{macroTileTag}, CT::isEdge<DataFlow>);
            graph.coordinates.addElement(DataFlow(), {ldsTag}, {macroTileTag});

            return {ldsTag, internalTileTag, loadConnections, storeConnections};
        }

        //
        // Rework this to stage-and-commit
        //
        void AddLDSVisitor::addStoreThroughLDSToControlGraph(KernelGraph& graph, int storeTag)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::AddLDSVisitor::addStoreThroughLDSToControlGraph({})", storeTag);

            auto [macroTileTag, macroTile] = graph.getDimension<MacroTile>(storeTag);
            auto [userTag, user]           = graph.getDimension<User>(storeTag);
            if(macroTile.memoryType != MemoryType::WAVE_LDS)
                return;

            // TODO Query graphs to figure appropriate forLoop for LDS
            // store
            // BEGIN fix this
            int kernel = *graph.control.getNodes<Kernel>().begin();
            int forLoop;
            for(auto tag : graph.control.depthFirstVisit(kernel, Graph::Direction::Downstream))
            {
                auto maybeForLoop = graph.control.get<ForLoopOp>(tag);
                if(maybeForLoop)
                {
                    forLoop = tag;
                    break;
                }
            }
            // END fix this

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

        //
        // Rework this to stage-and-commit
        //
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
