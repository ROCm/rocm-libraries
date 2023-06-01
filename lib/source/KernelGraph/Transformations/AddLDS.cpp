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
 *
 *
 * Prefetching; currently only a single pre-fetch is supported.
 *
 * If prefetching is requested, we look for prefetch candidates (see
 * findPrefetch).  For each ForLoop prefetch candidate we:
 *
 * 1. Colour the body of ForLoop according to the value of its Unroll
 *    coordinate.
 *
 * 2. Cut any edges between different colours and detach each colour
 *    island from the ForLoop.
 *
 * 3. Add a pre-loop global-prefetch segment:
 *
 *    a. For each colour, issue a global read.
 *
 *    b. Issue store-into-LDS for the first colour.
 *
 *    c. Issue a barrier.
 *
 * 4. Construct prefetch segments.
 */

#include "KernelGraph/ControlGraph/ControlGraph.hpp"
#include "KernelGraph/ControlGraph/Operation.hpp"
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDS.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;
        namespace CF = rocRoller::KernelGraph::ControlGraph;

        using GD = rocRoller::Graph::Direction;
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
                = kgraph.coordinates.getNeighbours<GD::Upstream>(forLoopCoord).to<std::vector>();
            for(auto forNeighbour : forNeighbours)
            {
                auto split = kgraph.coordinates.get<Split>(forNeighbour);
                if(split)
                {
                    auto splitNeighbours
                        = kgraph.coordinates.getNeighbours<GD::Downstream>(forNeighbour)
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
            int lds; // LDS allocation coordinate
            int user; // User coordinate
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

            void stagePrefetch(KernelGraph const& graph);

            void addLoadOperationsPrefetch(KernelGraph& graph, int forLoop, int numUnroll);

            void stageLoad(KernelGraph const&, int loadTag);
            void commit(KernelGraph&);

        private:
            std::map<int, ldsStoreInfo> m_store;

            std::set<LDSSpec>              m_loadSpecs;
            std::map<LDSSpec, ldsLoadInfo> m_loadInfo;
            std::map<int, LDSSpec>         m_tileSpecs;
            std::map<int, LDSSpec>         m_stagedLoads;
            std::map<LDSSpec, int>         m_stagedCoordinates;

            // Prefetch related
            std::map<int, int>                                    m_scopes;
            std::map<int, int>                                    m_prefetchLoops;
            std::map<int, std::map<int, std::unordered_set<int>>> m_prefetchUnrollBodyStarts;
            std::map<int, std::map<int, std::unordered_set<int>>> m_prefetchUnrollBodyEnds;
            std::map<int, std::map<int, std::set<std::tuple<int, int, int, int>>>>
                                                   m_prefetchFromLDSChains;
            std::map<int, std::unordered_set<int>> m_prefetchDelete;

            ContextPtr m_context;
        };

        /**
         * @brief Find loads that can be prefetched.
         *
         * To find prefetch candidates:
         *
         * 1. Look for LoadTiled operations that have ForLoop
         *    dimensions in their associated coordinate transform.
         *
         * 2. Find their containing ForLoop operation and make sure
         *    the loops associated coordinate is contained in the set
         *    above.
         *
         * 3. Make sure there is a neighbouring Unroll coordinate
         *    beside the ForLoop coordinate.
         *
         * 4. Make sure the size of the Unroll coorindate is
         *    consistent with the requested number of prefetches.
         */
        std::map<int, int> findPrefetch(KernelGraph const& kgraph)
        {
            std::map<int, int> rv;

            auto candidates = kgraph.control.getNodes<LoadTiled>();
            for(auto const& candidate : candidates)
            {
                auto [user, direction] = getOperationTarget(candidate, kgraph);
                auto maybeUser         = kgraph.coordinates.get<User>(user);
                if(!maybeUser)
                    continue;
                auto [required, path]   = findRequiredCoordinates(user, direction, kgraph);
                auto forLoopCoordinates = filterCoordinates<ForLoop>(required, kgraph);
                auto unrollCoordinates  = filterCoordinates<Unroll>(required, kgraph);

                auto maybeForLoop = findContainingOperation<ForLoopOp>(candidate, kgraph);

                if(maybeForLoop)
                {
                    // TODO: Only do the K-Loop for now
                    auto fl = kgraph.control.get<ForLoopOp>(*maybeForLoop);
                    if(fl->name != rocRoller::KLOOP)
                        continue;

                    auto forLoopCoord     = getForLoop(*maybeForLoop, kgraph);
                    auto maybeUnrollCoord = findUnrollNeighbour(kgraph, forLoopCoord);
                    if(forLoopCoordinates.contains(forLoopCoord) && maybeUnrollCoord)
                    {
                        Dimension unroll     = *kgraph.coordinates.get<Unroll>(*maybeUnrollCoord);
                        auto      unrollSize = getUnsignedInt(evaluate(getSize(unroll)));

                        rocRoller::Log::getLogger()->debug(
                            "KernelGraph::AddLDS(): ForLoop {} is a prefetch candidate: {} {}",
                            *maybeForLoop,
                            *maybeUnrollCoord,
                            unrollSize);

                        rv[*maybeForLoop] = unrollSize;
                    }
                }
            }

            return rv;
        }

        KernelGraph AddLDS::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::addLDS");
            rocRoller::Log::getLogger()->debug("KernelGraph::addLDS()");

            auto k       = original;
            auto visitor = AddLDSVisitor(m_context);

            // Add LDS operations
            for(auto const& loadTag : k.control.getNodes<LoadTiled>())
            {
                visitor.stageLoad(k, loadTag);
            }

            if(m_context->kernelOptions().prefetch)
            {
                AssertFatal(m_context->kernelOptions().unrollK > 1,
                            "KLoop must be unrolled when prefetching.");
                visitor.stagePrefetch(k);
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
                                    userTag,
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
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::AddLDS()::commit(): LoadLDSTile {}: lds {}",
                    loadTag,
                    m_loadInfo.at(loadSpec).lds);
                for(auto const& c : k.mapper.getConnections(loadTag))
                {
                    if(!k.coordinates.exists(c.coordinate))
                    {
                        k.mapper.disconnect(loadTag, c.coordinate, c.connection);
                    }
                }
            }

            //
            // Commit: Connect load/store operations in the control graph.
            //
            addLoadOperations(k);
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
                for(auto const& dependency : dependencies)
                {
                    auto edge = *only(graph.control.getNeighbours<GD::Upstream>(dependency));
                    graph.control.deleteElement(edge);
                    graph.control.addElement(Sequence(), {barrier}, {dependency});
                }

                graph.control.addElement(Body(), {operation}, {loadTileFromGlobalChain});
            }
            else
            {
                insertBefore(graph, operation, loadTileFromGlobalChain, barrier);
            }
        }

        int duplicateChain(KernelGraph& graph, std::vector<int> const& startNodes)
        {
            GraphReindexer reindexer;
            return duplicateControlNodes(
                graph, reindexer, startNodes, [](int x) { return true; })[0];
        }

        void
            AddLDSVisitor::addLoadOperationsPrefetch(KernelGraph& graph, int forLoop, int numUnroll)
        {
            auto logger = rocRoller::Log::getLogger();
            logger->debug("KernelGraph::AddLDS()::addLoadOperationsPrefetch({})", forLoop);

            AssertFatal(isOperation<ForLoopOp>(graph.control.getElement(forLoop)));

            auto forLoopCoord = getForLoop(forLoop, graph);
            auto unrollCoord  = *findUnrollNeighbour(graph, forLoopCoord);

            //
            // Delete connecting edges
            //
            for(auto tag : m_prefetchDelete[forLoop])
            {
                if(graph.control.exists(tag))
                    graph.control.deleteElement(tag);
            }

            // At this point, each of the unrolled loop bodies are
            // detached and isolated from the rest of the graph.

            std::map<int, std::vector<ldsLoadInfo>> globalLoadsByUnroll;
            for(auto spec : m_loadSpecs)
            {
                if(spec.operation == forLoop)
                {
                    globalLoadsByUnroll[spec.unrollCoordValue].push_back(m_loadInfo[spec]);
                }
            }

            //
            // Add Scope above the ForLoop
            //
            if(!m_scopes.contains(forLoop))
            {
                m_scopes[forLoop]
                    = replaceWith(graph, forLoop, graph.control.addElement(Scope()), false);
            }
            auto scope = m_scopes[forLoop];

            //
            // Prefetch before ForLoop
            //
            auto preBarrier = graph.control.addElement(Barrier());
            auto preNOP     = graph.control.addElement(NOP());
            graph.control.addElement(Sequence(), {preNOP}, {forLoop});

            std::vector<int> preChain;

            int numInFlight = m_context->kernelOptions().prefetchInFlight;

            // Loads first
            for(int u = 0; u < numInFlight; ++u)
            {
                for(auto load : globalLoadsByUnroll[u])
                {
                    logger->debug(
                        "  prefetch: pre-loop global load: unroll {} user {}", 0, load.user);
                    auto loadChain = duplicateChain(graph, {load.loadChain});
                    preChain.push_back(loadChain);
                }
            }

            // StoreLDS next
            for(auto load : globalLoadsByUnroll[0])
            {
                logger->debug("  prefetch: pre-loop commit lds: unroll {} user {} lds {}",
                              0,
                              load.user,
                              load.lds);
                auto storeChain = duplicateChain(graph, {load.storeChain});
                preChain.push_back(storeChain);
            }

            graph.control.addElement(Body(), {scope}, {preChain[0]});
            for(uint i = 1; i < preChain.size(); ++i)
            {
                graph.control.addElement(Sequence(), {preChain[i - 1]}, {preChain[i]});
            }
            graph.control.addElement(Sequence(), {preChain.back()}, {preBarrier});

            auto addPrefetchChains = [&](int u, int pre, int post, bool duplicate) {
                std::map<int, std::vector<int>> prefetchChain;
                for(auto [user, _ignore1, _ignore2, chain] : m_prefetchFromLDSChains[forLoop][u])
                {
                    int dchain = duplicate ? duplicateChain(graph, {chain}) : chain;
                    prefetchChain[user].push_back(dchain);
                }

                for(auto& [user, chain] : prefetchChain)
                {
                    graph.control.addElement(Sequence(), {pre}, {chain[0]});
                    for(uint i = 1; i < chain.size(); ++i)
                    {
                        graph.control.addElement(Sequence(), {chain[i - 1]}, {chain[i]});
                    }
                    graph.control.addElement(Sequence(), {chain.back()}, {post});
                }
            };

            addPrefetchChains(0, preBarrier, preNOP, true);
            graph.control.addElement(Sequence(), {preBarrier}, {preNOP});

            //
            // ForLoop body
            //

            // Update SetCoordinates for LoadTile operations
            //
            // The pre-loop LoadTiles were duplicated above, so their
            // SetCoordinates are intact.
            for(uint u = 0; u < numUnroll; ++u)
            {
                auto prefetchCoordExpr = literal(numUnroll - u);
                if(numInFlight > 1)
                    prefetchCoordExpr = literal(numUnroll + u);

                for(auto load : globalLoadsByUnroll[u])
                {
                    logger->debug("  prefetch: in-loop global load {} user {} expr {}",
                                  u,
                                  load.user,
                                  toString(prefetchCoordExpr));

                    auto setPrefetchCoord = SetCoordinate(prefetchCoordExpr);

                    auto maybeSetCoordinate = graph.control.get<SetCoordinate>(load.loadChain);
                    auto loadUnrollCoord    = graph.mapper.get<Unroll>(load.loadChain);
                    if(maybeSetCoordinate && loadUnrollCoord == unrollCoord)
                    {
                        graph.control.setElement(load.loadChain, setPrefetchCoord);
                    }
                    else
                    {
                        Throw<FatalError>("Mismatched SetCoordinate node above LoadTile.");
                    }
                }
            }

            // Build Unroll segment boundaries
            std::vector<int> segmentBoundaries = {forLoop};
            for(uint u = 0; u < numUnroll; ++u)
                segmentBoundaries.push_back(graph.control.addElement(NOP()));

            auto separateMemOps = !m_context->kernelOptions().prefetchMixMemOps;

            for(uint u = 0; u < numUnroll; ++u)
            {
                logger->debug("  prefetch: in-loop: segment {}", u);

                // Connect the segment to the preceeding segment boundary
                for(auto tag : m_prefetchUnrollBodyStarts[forLoop][u])
                {
                    if(u == 0)
                        graph.control.addElement(Body(), {segmentBoundaries[u]}, {tag});
                    else
                        graph.control.addElement(Sequence(), {segmentBoundaries[u]}, {tag});
                }

                auto globalPrefetchU = (u + numInFlight) % numUnroll;
                auto ldsPrefetchU    = (u + 1) % numUnroll;
                auto barrier         = graph.control.addElement(Barrier());

                auto nop = separateMemOps ? graph.control.addElement(NOP()) : -1;

                // Issue global loads
                auto globalLoads = globalLoadsByUnroll[globalPrefetchU];
                if(separateMemOps)
                {
                    graph.control.addElement(Sequence(), {nop}, {globalLoads[0].loadChain});
                }
                else if(u == 0)
                {
                    graph.control.addElement(
                        Body(), {segmentBoundaries[0]}, {globalLoads[0].loadChain});
                }
                else
                {
                    graph.control.addElement(
                        Sequence(), {segmentBoundaries[u]}, {globalLoads[0].loadChain});
                }

                logger->debug("  prefetch: in-loop: global load {} user {} lds {}",
                              globalPrefetchU,
                              globalLoads[0].user,
                              globalLoads[0].lds);

                for(int i = 1; i < globalLoads.size(); i++)
                {
                    graph.control.addElement(
                        Sequence(), {globalLoads[i - 1].loadChain}, {globalLoads[i].loadChain});

                    logger->debug("  prefetch: in-loop: global load {} user {} lds {}",
                                  globalPrefetchU,
                                  globalLoads[i].user,
                                  globalLoads[i].lds);
                }

                if(separateMemOps)
                {
                    graph.control.addElement(Sequence(),
                                             {globalLoads[globalLoads.size() - 1].loadChain},
                                             {segmentBoundaries[u + 1]});
                }

                // Commit in-flight to LDS
                auto globalStores = globalLoadsByUnroll[ldsPrefetchU];
                if(globalPrefetchU == ldsPrefetchU)
                {
                    graph.control.addElement(Sequence(),
                                             {globalLoads[globalLoads.size() - 1].loadChain},
                                             {globalStores[0].storeChain});
                }
                else if(separateMemOps)
                {
                    graph.control.addElement(Sequence(), {nop}, {globalStores[0].storeChain});
                }
                else if(u == 0)
                {
                    graph.control.addElement(
                        Body(), {segmentBoundaries[u]}, {globalStores[0].storeChain});
                }
                else
                {
                    graph.control.addElement(
                        Sequence(), {segmentBoundaries[u]}, {globalStores[0].storeChain});
                }

                logger->debug("  prefetch: in-loop: commit lds {} user {} lds {}",
                              ldsPrefetchU,
                              globalStores[0].user,
                              globalStores[0].lds);

                for(int i = 1; i < globalStores.size(); i++)
                {
                    graph.control.addElement(
                        Sequence(), {globalStores[i - 1].storeChain}, {globalStores[i].storeChain});

                    logger->debug("  prefetch: in-loop: commit lds {} user {} lds {}",
                                  ldsPrefetchU,
                                  globalStores[i].user,
                                  globalStores[i].lds);
                }

                graph.control.addElement(
                    Sequence(), {globalStores[globalStores.size() - 1].storeChain}, {barrier});

                // Prefetch from LDS
                if(m_prefetchFromLDSChains[forLoop].contains(ldsPrefetchU))
                {
                    addPrefetchChains(ldsPrefetchU, barrier, segmentBoundaries[u + 1], false);
                }
                else
                {
                    graph.control.addElement(Sequence(), {barrier}, {segmentBoundaries[u + 1]});
                }

                // Connect the segment to the proceeding segment boundary
                if(separateMemOps)
                {
                    for(auto tag : m_prefetchUnrollBodyEnds[forLoop][u])
                    {
                        graph.control.addElement(Sequence(), {tag}, {nop});
                    }
                }
                else
                {
                    for(auto tag : m_prefetchUnrollBodyEnds[forLoop][u])
                    {
                        graph.control.addElement(Sequence(), {tag}, {segmentBoundaries[u + 1]});
                    }
                }
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
            if(!m_prefetchLoops.empty())
            {
                for(auto [forLoop, numUnroll] : m_prefetchLoops)
                {
                    addLoadOperationsPrefetch(graph, forLoop, numUnroll);
                }

                return;
            }

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
                // TODO: Can we just insert load/store chains below containing for loop?
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
            for(auto tag : graph.control.depthFirstVisit(kernel, GD::Downstream))
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
            for(auto tag : graph.coordinates.depthFirstVisit(macroTileTag, GD::Downstream))
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
            auto incoming = graph.control.getNeighbours<GD::Upstream>(storeTag).to<std::vector>();
            for(auto e : incoming)
            {
                auto elem = graph.control.getElement(e);
                auto src  = graph.control.getNeighbours<GD::Upstream>(e).to<std::vector>();
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

        void AddLDSVisitor::stagePrefetch(KernelGraph const& k)
        {
            auto logger = rocRoller::Log::getLogger();

            m_prefetchLoops = findPrefetch(k);

            // Map: Operation (in loop body) to Unroll coordinate value
            std::map<int, int> operationUnroll;

            //
            // Find unroll bodies
            //
            // Need to build `operationUnroll` mapping.
            //
            // TODO: This implementation is fragile.
            //
            // First pass: Starting from each body edge out of the
            // ForLoop, we do a depth first search and look for
            // SetCoordinate nodes that set the appropriate Unroll
            // coordinate.  We then associate the unroll value from
            // the SetCoordinate node to the starting body edge.
            //
            // Second pass, go through the body edges again, and
            // propagate the Unroll value down.
            //
            // The fragile part here is: when two unroll bodies are
            // connected (via some kind of Sequence edge; usually from
            // a loop-carried-dependency) the order in which they are
            // visited during each pass matters...
            //
            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                std::map<int, int> bodyTopToCoordValue;

                auto forLoopCoord     = getForLoop(forLoop, k);
                auto maybeUnrollCoord = findUnrollNeighbour(k, forLoopCoord);
                AssertFatal(maybeUnrollCoord, "Prefetch with no unroll coordinate.");
                auto unrollCoord = *maybeUnrollCoord;

                auto bodies = k.control.getOutputNodeIndices<Body>(forLoop).to<std::set>();
                for(auto bodyTop : bodies)
                {
                    for(auto bodyElem : k.control.depthFirstVisit(bodyTop, GD::Downstream))
                    {
                        auto maybeSetCoordinate = k.control.get<SetCoordinate>(bodyElem);
                        if(!maybeSetCoordinate)
                            continue;

                        auto coordinate = k.mapper.get<Unroll>(bodyElem);
                        if(coordinate != unrollCoord)
                            continue;

                        AssertFatal(
                            evaluationTimes(
                                maybeSetCoordinate
                                    ->value)[rocRoller::Expression::EvaluationTime::Translate],
                            "Unroll value should be a literal");

                        auto unrollCoordValue = getUnsignedInt(evaluate(maybeSetCoordinate->value));

                        bodyTopToCoordValue[bodyTop] = unrollCoordValue;
                    }
                }

                for(auto bodyTop : bodies)
                {
                    for(auto bodyElem : k.control.depthFirstVisit(bodyTop, GD::Downstream))
                    {
                        operationUnroll[bodyElem] = bodyTopToCoordValue[bodyTop];
                    }
                }
            }

            //
            // Find top of unroll bodies
            //
            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                auto forLoopEdges = k.control.getNeighbours<GD::Downstream>(forLoop).to<std::set>();
                for(auto edge : forLoopEdges)
                {
                    if(!k.control.get<Body>(edge))
                        continue;
                    auto node = *only(k.control.getNeighbours<GD::Downstream>(edge));

                    m_prefetchUnrollBodyStarts[forLoop][operationUnroll[node]].insert(node);
                    m_prefetchDelete[forLoop].insert(edge);
                }
            }

            //
            // Find separator edges and mark for deletion
            //
            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                auto bodies = k.control.getOutputNodeIndices<Body>(forLoop).to<std::set>();
                for(auto bodyTop : bodies)
                {
                    for(auto bodyElem :
                        k.control.depthFirstVisit(bodyTop, GD::Downstream).to<std::unordered_set>())
                    {
                        if(!operationUnroll.contains(bodyElem))
                            continue;

                        auto bodyElemEdges
                            = k.control.getNeighbours<GD::Downstream>(bodyElem).to<std::set>();
                        for(auto edge : bodyElemEdges)
                        {
                            if(!k.control.get<Sequence>(edge))
                                continue;

                            auto otherElem = *only(k.control.getNeighbours<GD::Downstream>(edge));

                            if(operationUnroll.contains(otherElem)
                               && operationUnroll[otherElem] != operationUnroll[bodyElem])
                            {
                                m_prefetchDelete[forLoop].insert(edge);
                            }
                        }
                    }
                }
            }

            //
            // Within each segment, carve out loads
            //
            int splitLDSPrefetchFactor = m_context->kernelOptions().prefetchLDSFactor;

            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                auto forLoopCoord     = getForLoop(forLoop, k);
                auto maybeUnrollCoord = findUnrollNeighbour(k, forLoopCoord);
                AssertFatal(maybeUnrollCoord, "Prefetch with no unroll coordinate.");
                auto prefetchUnrollCoord = *maybeUnrollCoord;

                for(auto u = 0; u < numUnroll; ++u)
                {
                    logger->debug("KernelGraph::AddLDS()::stagePrefetch: segment {}", u);

                    auto starts = m_prefetchUnrollBodyStarts[forLoop][u];

                    for(auto start : starts)
                    {
                        for(auto elem : k.control.depthFirstVisit(start, GD::Downstream))
                        {
                            auto maybeSetCoordinate = k.control.get<SetCoordinate>(elem);
                            if(!maybeSetCoordinate)
                                continue;

                            auto coordinate = k.mapper.get<Unroll>(elem);

                            AssertFatal(
                                evaluationTimes(
                                    maybeSetCoordinate
                                        ->value)[rocRoller::Expression::EvaluationTime::Translate],
                                "Unroll value should be a literal");

                            auto unrollCoordValue
                                = getUnsignedInt(evaluate(maybeSetCoordinate->value));

                            AssertFatal(coordinate != prefetchUnrollCoord);
                            logger->debug("  SetCoordinate {} Unroll {} value {}",
                                          elem,
                                          coordinate,
                                          unrollCoordValue);

                            if(splitLDSPrefetchFactor == 0)
                                break;

                            if(splitLDSPrefetchFactor > 0)
                            {
                                auto [_ignore, unrollCoord] = k.getDimension<Unroll>(elem);
                                auto unrollCoordSize = getUnsignedInt(evaluate(unrollCoord.size));
                                if(unrollCoordValue >= unrollCoordSize / splitLDSPrefetchFactor)
                                    break;
                            }

                            // Find Load operation underneath
                            auto loadTag = *only(k.control.findNodes(
                                elem,
                                [&](int x) {
                                    return isOperation<LoadTiled>(k.control.getElement(x));
                                },
                                GD::Downstream));

                            auto userTag = k.mapper.get<User>(loadTag);
                            auto tileTag = k.mapper.get<MacroTile>(loadTag);

                            m_prefetchFromLDSChains[forLoop][u].insert(
                                {0, tileTag, unrollCoordValue, elem});

                            m_prefetchUnrollBodyStarts[forLoop][operationUnroll[elem]].erase(elem);

                            for(auto inEdge : k.control.getNeighbours<GD::Upstream>(elem))
                            {
                                m_prefetchDelete[forLoop].insert(inEdge);
                            }

                            for(auto outEdge : k.control.getNeighbours<GD::Downstream>(elem))
                            {
                                if(!k.control.get<Sequence>(outEdge))
                                    continue;

                                auto outNode
                                    = *only(k.control.getNeighbours<GD::Downstream>(outEdge));

                                m_prefetchUnrollBodyStarts[forLoop][operationUnroll[elem]].insert(
                                    outNode);

                                m_prefetchDelete[forLoop].insert(outEdge);
                            }

                            break;
                        }
                    }
                }
            }

            //
            // Find tail end of each unrolled loop body
            //
            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                for(auto u = 0; u < numUnroll; ++u)
                {
                    auto starts = m_prefetchUnrollBodyStarts[forLoop][u];

                    int  fLoop                   = forLoop;
                    auto onlyFollowSequenceEdges = [&](int x) -> bool {
                        auto isSequence = CF::isEdge<Sequence>(k.control.getElement(x));
                        auto willDelete = m_prefetchDelete[fLoop].contains(x);
                        return isSequence && !willDelete;
                    };

                    for(auto op :
                        k.control.depthFirstVisit(starts, onlyFollowSequenceEdges, GD::Downstream))
                    {
                        auto outgoing
                            = k.control.getNeighbours(op, GD::Downstream).to<std::unordered_set>();
                        for(auto tag : m_prefetchDelete[forLoop])
                        {
                            outgoing.erase(tag);
                        }
                        if(outgoing.empty())
                        {
                            m_prefetchUnrollBodyEnds[forLoop][u].insert(op);
                        }
                    }
                }
            }
        }
    }
}
