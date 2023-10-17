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
 * Transformations are done using a "stage and commit" approach.
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
#include "KernelGraph/ControlToCoordinateMapper.hpp"
#include "KernelGraph/CoordinateGraph/Dimension.hpp"
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
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
        using LD = rocRoller::KernelGraph::Connections::LDSLoadStore;
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
         * @brief Return LDS specifier for the load/store operation.
         *
         * This inspects the graph and figures out the User, ForLoop,
         * Unroll etc coordinates that determine which LDS buffer the
         * operation will populate.
         *
         * When determining the specific ForLoop and Unroll
         * coordinates, only the containing for loop is considered.
         *
         * The containing loop's tag is also used to: determine where
         * LDS is populated, and differentiate the LDS allocations.
         * This means jammed loops will use unique LDS allocations.
         *
         * If there is no containing ForLoop, the location of the
         * original operation is used to: determine where LDS is
         * populated, and differentiate the LDS allocations.
         */
        LDSSpec getLDSSpec(KernelGraph const& k, int opTag)
        {
            auto [userTag, user]           = k.getDimension<User>(opTag);
            auto [macroTileTag, macroTile] = k.getDimension<MacroTile>(opTag);

            auto [target, direction] = getOperationTarget(opTag, k);
            auto [required, path]    = findRequiredCoordinates(target, direction, k);
            auto forLoopCoordinates  = filterCoordinates<ForLoop>(required, k);

            auto maybeForLoop = findContainingOperation<ForLoopOp>(opTag, k);
            int  forLoopCoord = -1;
            int  operation;
            if(maybeForLoop)
            {
                operation = *maybeForLoop;
                auto f    = getForLoopCoords(*maybeForLoop, k).first;
                if(forLoopCoordinates.contains(f))
                {
                    forLoopCoord = f;
                }
            }
            else
            {
                operation = opTag;
            }

            auto maybeUnroll      = findUnrollNeighbour(k, forLoopCoord);
            int  unrollCoord      = -1;
            int  unrollCoordValue = -1;
            if(maybeUnroll)
            {
                unrollCoord = *maybeUnroll;

                auto setCoord
                    = k.control.get<SetCoordinate>(getSetCoordinateForDim(k, unrollCoord, opTag));
                AssertFatal(evaluationTimes(
                                setCoord->value)[rocRoller::Expression::EvaluationTime::Translate],
                            "Unroll value should be a literal");

                unrollCoordValue = getUnsignedInt(evaluate(setCoord->value));
            }

            return {userTag,
                    forLoopCoord,
                    unrollCoord,
                    unrollCoordValue,
                    operation,
                    getVariableType(k, opTag),
                    macroTile.memoryType};
        }

        /**
         * @brief Container for info related to loading from Global
         * into LDS.
         */
        struct ldsOperationInfo
        {
            bool load;
            int  user; // User coordinate
            int  globalOperation; // LoadTiled/StoreTiled operation
            int  ldsOperation; // StoreLDSTile/LoadLDSTile operation
            int  globalChain; // LoadTiled/StoreTiled operation
            int  ldsChain; // StoreLDStile/LoadLDSTile operation
        };

        /**
         * Add LDS transformer.
         */
        struct AddLDSVisitor
        {
            AddLDSVisitor(ContextPtr context)
                : m_context(context)
            {
            }

            void addLoadOperations(KernelGraph& graph);
            void addStoreOperations(KernelGraph& graph);
            void addStoreOperations(KernelGraph& graph, int storetag, LDSSpec const& spec);
            void addLoadOperationsPrefetch(KernelGraph& graph, int forLoop, int numUnroll);

            void stageLoad(KernelGraph const&, int loadTag);
            void stageStore(KernelGraph const&, int storeTag);
            void stagePrefetch(KernelGraph const& graph);
            void commit(KernelGraph&);

        private:
            std::set<LDSSpec>      m_loadSpecs;
            std::map<int, LDSSpec> m_stagedLoads;

            std::set<LDSSpec>      m_storeSpecs;
            std::map<int, LDSSpec> m_stagedStores;

            std::map<int, LDSSpec>              m_stagedOps;
            std::map<LDSSpec, ldsOperationInfo> m_info;

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
         * 3. Make sure there is a neighboring Unroll coordinate
         *    beside the ForLoop coordinate.
         *
         * 4. Make sure the size of the Unroll coordinate is
         *    consistent with the requested number of prefetches.
         */
        std::map<int, int> findPrefetch(KernelGraph const& kgraph)
        {
            std::map<int, int> rv;

            auto candidates = kgraph.control.getNodes<LoadTiled>();
            for(auto const& candidate : candidates)
            {
                auto [user, direction] = getOperationTarget(candidate, kgraph);
                if(!kgraph.coordinates.get<User>(user).has_value())
                    continue;
                auto [required, path]   = findRequiredCoordinates(user, direction, kgraph);
                auto forLoopCoordinates = filterCoordinates<ForLoop>(required, kgraph);
                auto unrollCoordinates  = filterCoordinates<Unroll>(required, kgraph);

                auto maybeForLoop = findContainingOperation<ForLoopOp>(candidate, kgraph);

                if(maybeForLoop)
                {
                    // TODO: Only do the K-Loop for now
                    auto fl = kgraph.control.get<ForLoopOp>(*maybeForLoop);
                    if(fl->loopName != rocRoller::KLOOP)
                        continue;

                    auto forLoopCoord     = getForLoopCoords(*maybeForLoop, kgraph).first;
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

            auto k = original;

            auto visitor = AddLDSVisitor(m_context);

            // Add LDS operations
            for(auto const& loadTag : k.control.getNodes<LoadTiled>())
            {
                visitor.stageLoad(k, loadTag);
            }

            for(auto const& storeTag : k.control.getNodes<StoreTiled>())
            {
                visitor.stageStore(k, storeTag);
            }

            if(m_context->kernelOptions().prefetch)
            {
                AssertFatal(m_context->kernelOptions().unrollK > 1,
                            "KLoop must be unrolled when prefetching.");
                visitor.stagePrefetch(k);
            }

            visitor.commit(k);

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
            // Stage: convert LoadTile to LoadLDSTile
            //
            m_stagedLoads[loadTag] = spec;
            m_stagedOps[loadTag]   = spec;
        }

        /*
         * Stage everything for a store.  Does not modify the graph.
         */
        void AddLDSVisitor::stageStore(KernelGraph const& k, int storeTag)
        {
            auto [userTag, user] = k.getDimension<User>(storeTag);
            auto [tileTag, tile] = k.getDimension<MacroTile>(storeTag);

            if(!(tile.memoryType == MemoryType::WAVE_LDS || tile.memoryType == MemoryType::LDS))
                return;

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::AddLDS()::stageStore({}): User {}, MacroTile {}",
                storeTag,
                userTag,
                tileTag);

            auto spec = getLDSSpec(k, storeTag);
            m_storeSpecs.insert(spec);
            m_stagedStores[storeTag] = spec;
            m_stagedOps[storeTag]    = spec;
        }

        /*
         * Commit everything.  Modifies the graph.
         */
        void AddLDSVisitor::commit(KernelGraph& k)
        {
            rocRoller::Log::getLogger()->debug("KernelGraph::AddLDS()::commit()");

            //
            // Commit: Create operations for LoadTiled.
            //
            for(auto spec : m_loadSpecs)
            {
                auto userTag            = spec.userCoord;
                auto loadTileFromGlobal = k.control.addElement(LoadTiled(spec.variableType));
                auto storeTileIntoLDS
                    = k.control.addElement(StoreLDSTile(spec.variableType.dataType));

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

                m_info[spec]
                    = {true, userTag, loadTileFromGlobal, storeTileIntoLDS, loadChain, storeChain};
            }

            //
            // Commit: Create operations for StoreTiled.
            //
            for(auto spec : m_storeSpecs)
            {
                auto userTag         = spec.userCoord;
                auto loadTileFromLDS = k.control.addElement(LoadLDSTile(spec.variableType));
                auto storeTileToGlobal
                    = k.control.addElement(StoreTiled(spec.variableType.dataType));

                auto loadChain  = loadTileFromLDS;
                auto storeChain = storeTileToGlobal;

                if(spec.unrollCoord >= 0)
                {
                    auto setCoordForLoad
                        = k.control.addElement(SetCoordinate(literal(spec.unrollCoordValue)));
                    k.mapper.connect<Unroll>(setCoordForLoad, spec.unrollCoord);
                    auto setCoordForStore
                        = k.control.addElement(SetCoordinate(literal(spec.unrollCoordValue)));
                    k.mapper.connect<Unroll>(setCoordForStore, spec.unrollCoord);

                    k.control.addElement(Body(), {setCoordForLoad}, {loadTileFromLDS});
                    k.control.addElement(Body(), {setCoordForStore}, {storeTileToGlobal});
                    loadChain  = setCoordForLoad;
                    storeChain = setCoordForStore;
                }

                m_info[spec]
                    = {false, userTag, storeTileToGlobal, loadTileFromLDS, storeChain, loadChain};
            }

            // At this point: load/store operations have been added,
            // but they aren't connected (through the mapper nor in
            // the graphs).

            //
            // Commit: Change all LoadTiled operations to LoadLDSTile
            //
            std::set<int> updatedTiles;
            for(auto [opTag, opSpec] : m_stagedOps)
            {
                auto globalOp = m_info.at(opSpec).globalOperation;
                auto ldsOp    = m_info.at(opSpec).ldsOperation;
                for(auto& c : k.mapper.getConnections(opTag))
                {
                    if(std::holds_alternative<Connections::LDSTypeAndSubDimension>(c.connection))
                    {
                        auto ldsConnection
                            = std::get<Connections::LDSTypeAndSubDimension>(c.connection);

                        auto newConnection = Connections::TypeAndSubDimension{
                            ldsConnection.id, ldsConnection.subdimension};
                        if(ldsConnection.direction == LD::LOAD_FROM_GLOBAL
                           || ldsConnection.direction == LD::STORE_INTO_GLOBAL)
                        {
                            // Operation was LoadTiled or StoreTiled, connections are for
                            // global traffic.
                            //
                            // Therefore use globalOp.

                            k.mapper.connect(globalOp, c.coordinate, newConnection);
                        }
                        else if((m_info[opSpec].load
                                 && ldsConnection.direction == LD::STORE_INTO_LDS)
                                || (!m_info[opSpec].load
                                    && ldsConnection.direction == LD::LOAD_FROM_LDS))
                        {
                            // Operation was LoadTiled, connections are for traffic into LDS; or
                            // Operation was StoreTiled, connections are for traffic out of LDS.
                            //
                            // Therefore use ldsOp.

                            k.mapper.connect(ldsOp, c.coordinate, newConnection);
                        }
                        else
                        {
                            // Otherwise, use original opTag.

                            k.mapper.connect(opTag, c.coordinate, newConnection);
                        }
                        k.mapper.disconnect(opTag, c.coordinate, c.connection);
                    }
                }

                // Update dimension and operation
                auto [macroTileTag, macroTile] = k.getDimension<MacroTile>(opTag);

                if(m_info[opSpec].load)
                {
                    k.control.setElement(opTag, LoadLDSTile(opSpec.variableType));
                }
                else
                {
                    k.control.setElement(opTag, StoreLDSTile(opSpec.variableType.dataType));
                }

                if(!updatedTiles.contains(macroTileTag))
                {
                    if(macroTile.memoryType == MemoryType::WAVE_LDS)
                        macroTile.memoryType = MemoryType::WAVE;
                    else
                        macroTile.memoryType = MemoryType::VGPR;
                    k.coordinates.setElement(macroTileTag, macroTile);

                    updatedTiles.insert(macroTileTag);
                }
            }

            //
            // Commit: Connect load/store operations in the control graph.
            //
            addLoadOperations(k);
            addStoreOperations(k);
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
                    for(auto body : filter(graph.control.isElemType<Body>(),
                                           graph.control.getNeighbours<GD::Upstream>(dependency))
                                        .to<std::vector>())
                    {
                        graph.control.deleteElement(body);
                        graph.control.addElement(Sequence(), {barrier}, {dependency});
                    }
                }
                graph.control.addElement(Body(), {operation}, {loadTileFromGlobalChain});

                auto existingLoads
                    = filter(graph.control.isElemType<LoadTiled>(),
                             graph.control.depthFirstVisit(operation, Graph::Direction::Downstream))
                          .to<std::set>();
                orderMemoryNodes(graph, existingLoads, {loadTileFromGlobalChain}, true);
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

            auto forLoopCoord = getForLoopCoords(forLoop, graph).first;
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

            std::map<int, std::vector<ldsOperationInfo>> globalLoadsByUnroll;
            for(auto spec : m_loadSpecs)
            {
                if(spec.operation == forLoop)
                {
                    globalLoadsByUnroll[spec.unrollCoordValue].push_back(m_info[spec]);
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
                        "  prefetch: pre-loop global load: unroll {} user {}", u, load.user);
                    auto loadChain = duplicateChain(graph, {load.globalChain});
                    preChain.push_back(loadChain);
                }
            }

            // StoreLDS next
            for(auto load : globalLoadsByUnroll[0])
            {
                logger->debug("  prefetch: pre-loop commit lds: unroll {} user {}", 0, load.user);
                auto storeChain = duplicateChain(graph, {load.ldsChain});
                preChain.push_back(storeChain);
            }

            graph.control.addElement(Body(), {scope}, {preChain[0]});
            for(uint i = 1; i < preChain.size(); ++i)
            {
                graph.control.addElement(Sequence(), {preChain[i - 1]}, {preChain[i]});
            }
            graph.control.addElement(Sequence(), {preChain.back()}, {preBarrier});

            auto addLDSPrefetchChains = [&](int u, int pre, int post, bool duplicate) {
                std::map<int, std::vector<int>> prefetchChain;
                for(auto [user, _ignore1, _ignore2, chain] : m_prefetchFromLDSChains[forLoop][u])
                {
                    int dchain = duplicate ? duplicateChain(graph, {chain}) : chain;
                    prefetchChain[user].push_back(dchain);
                }

                for(auto& [user, chain] : prefetchChain)
                {
                    logger->debug("  addPrefetchChains: connecting {} to {}", pre, chain[0]);
                    graph.control.addElement(Sequence(), {pre}, {chain[0]});
                    for(uint i = 1; i < chain.size(); ++i)
                    {
                        logger->debug(
                            "  addPrefetchChains: connecting {} to {}", chain[i - 1], chain[i]);
                        graph.control.addElement(Sequence(), {chain[i - 1]}, {chain[i]});
                    }
                    logger->debug("  addPrefetchChains: connecting {} to {}", chain.back(), post);
                    graph.control.addElement(Sequence(), {chain.back()}, {post});
                }
            };

            addLDSPrefetchChains(0, preBarrier, preNOP, true);
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
                auto prefetchGlobalU   = (u + numInFlight) % numUnroll;
                auto prefetchCoordExpr = literal(u + numInFlight);

                for(auto load : globalLoadsByUnroll[prefetchGlobalU])
                {
                    logger->debug("  prefetch: in-loop: set coordinate: load {} user {} expr {}",
                                  prefetchGlobalU,
                                  load.user,
                                  toString(prefetchCoordExpr));

                    auto setPrefetchCoord = SetCoordinate(prefetchCoordExpr);

                    auto maybeSetCoordinate = graph.control.get<SetCoordinate>(load.globalChain);
                    auto loadUnrollCoord    = graph.mapper.get<Unroll>(load.globalChain);
                    if(maybeSetCoordinate && loadUnrollCoord == unrollCoord)
                    {
                        graph.control.setElement(load.globalChain, setPrefetchCoord);
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

                // Connect the segment to the preceding segment boundary
                //
                // Note that the first boundary is the forLoop, and
                // the remaining boundaries are NOPs.  Therefore
                // segmentBoundaries[u] is the "preceding" boundary.
                for(auto tag : m_prefetchUnrollBodyStarts[forLoop][u])
                {
                    if(u == 0)
                        graph.control.addElement(Body(), {segmentBoundaries[u]}, {tag});
                    else
                    {
                        auto descOfSegmentStart
                            = graph.control
                                  .depthFirstVisit(
                                      tag, graph.control.isElemType<Sequence>(), GD::Downstream)
                                  .to<std::set>();

                        if(!descOfSegmentStart.contains(segmentBoundaries[u]))
                        {
                            graph.control.addElement(Sequence(), {segmentBoundaries[u]}, {tag});
                        }
                    }
                }

                auto globalPrefetchU = (u + numInFlight) % numUnroll;
                auto ldsPrefetchU    = (u + 1) % numUnroll;
                auto barrier         = graph.control.addElement(Barrier());

                auto nop = separateMemOps ? graph.control.addElement(NOP()) : -1;

                // Issue global loads
                auto globalLoads = globalLoadsByUnroll[globalPrefetchU];
                logger->debug("  Issue global loads: {}", globalLoads[0].globalChain);
                if(separateMemOps)
                {
                    graph.control.addElement(Sequence(), {nop}, {globalLoads[0].globalChain});
                }
                else if(u == 0)
                {
                    graph.control.addElement(
                        Body(), {segmentBoundaries[u]}, {globalLoads[0].globalChain});
                }
                else
                {
                    graph.control.addElement(
                        Sequence(), {segmentBoundaries[u]}, {globalLoads[0].globalChain});
                }

                logger->debug("  prefetch: in-loop: global load {} user {}",
                              globalPrefetchU,
                              globalLoads[0].user);

                for(int i = 1; i < globalLoads.size(); i++)
                {
                    graph.control.addElement(
                        Sequence(), {globalLoads[i - 1].globalChain}, {globalLoads[i].globalChain});

                    logger->debug("  prefetch: in-loop: global load {} user {}",
                                  globalPrefetchU,
                                  globalLoads[i].user);
                }

                if(separateMemOps)
                {
                    graph.control.addElement(Sequence(),
                                             {globalLoads[globalLoads.size() - 1].globalChain},
                                             {segmentBoundaries[u + 1]});
                }

                // Commit in-flight to LDS
                auto globalStores = globalLoadsByUnroll[ldsPrefetchU];
                if(separateMemOps)
                {
                    graph.control.addElement(Sequence(), {nop}, {globalStores[0].ldsChain});
                }
                else if(u == 0)
                {
                    graph.control.addElement(
                        Body(), {segmentBoundaries[u]}, {globalStores[0].ldsChain});
                }
                else
                {
                    graph.control.addElement(
                        Sequence(), {segmentBoundaries[u]}, {globalStores[0].ldsChain});
                }

                logger->debug("  prefetch: in-loop: commit lds {} user {}",
                              ldsPrefetchU,
                              globalStores[0].user);

                for(int i = 1; i < globalStores.size(); i++)
                {
                    graph.control.addElement(
                        Sequence(), {globalStores[i - 1].ldsChain}, {globalStores[i].ldsChain});

                    logger->debug("  prefetch: in-loop: connecting: {} to {}",
                                  globalStores[i - 1].ldsChain,
                                  globalStores[i].ldsChain);

                    logger->debug("  prefetch: in-loop: commit lds {} user {}",
                                  ldsPrefetchU,
                                  globalStores[i].user);
                }

                graph.control.addElement(
                    Sequence(), {globalStores[globalStores.size() - 1].ldsChain}, {barrier});

                logger->debug("  prefetch: in-loop: ordering {} to {}",
                              globalLoads[globalLoads.size() - 1].globalChain,
                              globalStores[0].ldsChain);
                graph.control.addElement(Sequence(),
                                         {globalLoads[globalLoads.size() - 1].globalChain},
                                         {globalStores[0].ldsChain});

                // Prefetch from LDS
                if(m_prefetchFromLDSChains[forLoop].contains(ldsPrefetchU))
                {
                    addLDSPrefetchChains(ldsPrefetchU, barrier, segmentBoundaries[u + 1], false);
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
                auto info         = m_info.at(spec);
                // TODO: Can we just insert load/store chains below containing for loop?
                addLoadOperationsNoPrefetch(
                    graph, info.globalChain, info.ldsChain, spec.operation, dependencies);
            }
        }

        void
            AddLDSVisitor::addStoreOperations(KernelGraph& graph, int storeTag, LDSSpec const& spec)
        {
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

            auto barrier = graph.control.addElement(Barrier());
            graph.control.addElement(Sequence(), {storeTag}, {barrier});

            auto forLoop                  = spec.operation;
            auto loadMacroTileFromLDSNode = m_info[spec].ldsChain;
            auto storeMacroTileIntoGlobal = m_info[spec].globalChain;

            graph.control.addElement(Sequence(), {forLoop}, {loadMacroTileFromLDSNode});

            graph.control.addElement(
                Sequence(), {loadMacroTileFromLDSNode}, {storeMacroTileIntoGlobal});
        }

        void AddLDSVisitor::addStoreOperations(KernelGraph& graph)
        {
            rocRoller::Log::getLogger()->debug("KernelGraph::AddLDSVisitor::addStoreOperations()");

            for(auto spec : m_storeSpecs)
            {
                for(auto [storeTag, storeSpec] : m_stagedStores)
                {
                    addStoreOperations(graph, storeTag, spec);
                }
            }
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
            // First pass: Starting from each body edge out of the
            // ForLoop, we do a depth first search along body edges
            // and look for SetCoordinate nodes that set the appropriate
            // Unroll coordinate.  We then associate the unroll value from
            // the SetCoordinate node to the starting body edge.
            //
            // Second pass, go through the body edges again, and
            // propagate the Unroll value down.
            //
            // Finally we need to find the unroll value for all of the multiplies.
            // We do this by populating macroTileToCoordVal with mappings from
            // macrotiles to unroll values by looking through all loadtiled nodes
            // and mapping it's macrotile to the previously assigned unroll.
            // Once we have a mapping for all the macrotiles, we assign the unroll
            // for the multiply to the unroll of its LHS macrotile.
            //
            // If there are ever any nodes other than SetCoordinate, LoadTiled, and
            // Multiply in the forloop, they will need to be handled.
            //
            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                auto bodies
                    = k.control.getOutputNodeIndices<Body>(forLoop).to<std::unordered_set>();

                auto forLoopCoord     = getForLoopCoords(forLoop, k).first;
                auto maybeUnrollCoord = findUnrollNeighbour(k, forLoopCoord);
                AssertFatal(maybeUnrollCoord, "Prefetch with no unroll coordinate.");
                auto unrollCoord = *maybeUnrollCoord;

                std::map<int, int> bodyTopToCoordValue;
                for(auto bodyTop : bodies)
                {
                    for(auto bodyElem :
                        filter(k.control.isElemType<SetCoordinate>(),
                               k.control.depthFirstVisit(
                                   bodyTop, k.control.isElemType<Body>(), GD::Downstream)))
                    {
                        auto setCoord   = *(k.control.get<SetCoordinate>(bodyElem));
                        auto coordinate = k.mapper.get<Unroll>(bodyElem);
                        if(coordinate != unrollCoord)
                            continue;

                        AssertFatal(
                            evaluationTimes(
                                setCoord.value)[rocRoller::Expression::EvaluationTime::Translate],
                            "Unroll value should be a literal");

                        auto unrollCoordValue = getUnsignedInt(evaluate(setCoord.value));

                        bodyTopToCoordValue[bodyTop] = unrollCoordValue;
                        break;
                    }
                    AssertFatal(bodyTopToCoordValue.count(bodyTop),
                                "SetCoordinate must belong to an unroll.");
                }

                for(auto bodyTop : bodies)
                {
                    for(auto bodyElem : k.control.depthFirstVisit(
                            bodyTop, k.control.isElemType<Body>(), GD::Downstream))
                    {
                        operationUnroll[bodyElem] = bodyTopToCoordValue[bodyTop];
                    }
                }

                std::map<int, int> macroTileToCoordVal;
                for(auto bodyTop : bodies)
                {
                    for(auto loadTag : filter(k.control.isElemType<LoadTiled>(),
                                              k.control.depthFirstVisit(bodyTop, GD::Downstream)))
                    {
                        auto [macroTileTag, macroTile] = k.getDimension<MacroTile>(loadTag);
                        if(macroTileToCoordVal.count(macroTileTag))
                        {
                            AssertFatal(macroTileToCoordVal.at(macroTileTag)
                                            == operationUnroll.at(loadTag),
                                        "All loads that belong to a given unroll must use the same "
                                        "macrotile.");
                        }
                        else
                        {
                            macroTileToCoordVal[macroTileTag] = operationUnroll.at(loadTag);
                        }
                    }
                }

                for(auto multiply : filter(k.control.isElemType<Multiply>(),
                                           k.control.depthFirstVisit(forLoop, GD::Downstream)))
                {
                    auto [macroTileTagLHS, macLHS] = k.getDimension<MacroTile>(
                        multiply, Connections::typeArgument<MacroTile>(NaryArgument::LHS));
                    auto [macroTileTagRHS, macRHS] = k.getDimension<MacroTile>(
                        multiply, Connections::typeArgument<MacroTile>(NaryArgument::RHS));

                    AssertFatal(macroTileToCoordVal.at(macroTileTagLHS)
                                    == macroTileToCoordVal.at(macroTileTagRHS),
                                "The LHS and RHS of a multiply must be part of the same unroll.");

                    operationUnroll[multiply] = macroTileToCoordVal.at(macroTileTagLHS);
                }
            }

            //
            // Find top of unroll bodies
            //
            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                for(auto edge : filter(k.control.isElemType<Body>(),
                                       k.control.getNeighbours<GD::Downstream>(forLoop)))
                {
                    auto node = *only(k.control.getNeighbours<GD::Downstream>(edge));

                    m_prefetchUnrollBodyStarts[forLoop][operationUnroll[node]].insert(node);
                    m_prefetchDelete[forLoop].insert(edge);
                }
            }

            //
            // Find separator edges and mark for deletion
            //
            auto inOperationUnroll = [&](int x) -> bool { return operationUnroll.contains(x); };
            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                for(auto bodyTop : k.control.getOutputNodeIndices<Body>(forLoop))
                {
                    for(auto bodyElem : filter(inOperationUnroll,
                                               k.control.depthFirstVisit(bodyTop, GD::Downstream)))
                    {
                        for(auto edge : filter(k.control.isElemType<Sequence>(),
                                               k.control.getNeighbours<GD::Downstream>(bodyElem)))
                        {
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
                auto forLoopCoord     = getForLoopCoords(forLoop, k).first;
                auto maybeUnrollCoord = findUnrollNeighbour(k, forLoopCoord);
                AssertFatal(maybeUnrollCoord, "Prefetch with no unroll coordinate.");
                auto prefetchUnrollCoord = *maybeUnrollCoord;

                for(auto u = 0; u < numUnroll; ++u)
                {
                    logger->debug("KernelGraph::AddLDS()::stagePrefetch: segment {}", u);

                    auto starts = m_prefetchUnrollBodyStarts[forLoop][u];

                    for(auto start : starts)
                    {
                        for(auto elem :
                            filter(k.control.isElemType<SetCoordinate>(),
                                   k.control.depthFirstVisit(
                                       start, k.control.isElemType<Body>(), GD::Downstream)))
                        {
                            auto setCoord   = *(k.control.get<SetCoordinate>(elem));
                            auto coordinate = k.mapper.get<Unroll>(elem);

                            AssertFatal(
                                evaluationTimes(
                                    setCoord
                                        .value)[rocRoller::Expression::EvaluationTime::Translate],
                                "Unroll value should be a literal");

                            auto unrollCoordValue = getUnsignedInt(evaluate(setCoord.value));

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
                                elem, k.control.isElemType<LoadTiled>(), GD::Downstream));

                            auto userTag = k.mapper.get<User>(loadTag);
                            auto tileTag = k.mapper.get<MacroTile>(loadTag);

                            m_prefetchFromLDSChains[forLoop][u].insert(
                                {0, tileTag, unrollCoordValue, elem});

                            m_prefetchUnrollBodyStarts[forLoop][operationUnroll[elem]].erase(elem);

                            for(auto inEdge : k.control.getNeighbours<GD::Upstream>(elem))
                            {
                                m_prefetchDelete[forLoop].insert(inEdge);
                            }

                            for(auto outEdge :
                                filter(k.control.isElemType<Sequence>(),
                                       k.control.getNeighbours<GD::Downstream>(elem)))
                            {
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

        ConstraintStatus AcceptablePrefetchNodes(const KernelGraph& k)
        {
            ConstraintStatus retval;

            for(auto [forLoop, numUnroll] : findPrefetch(k))
            {
                for(auto bodyTop : k.control.getOutputNodeIndices<Body>(forLoop))
                {
                    for(auto bodyElem : k.control.depthFirstVisit(bodyTop, GD::Downstream))
                    {
                        if(k.control.get<Body>(bodyElem) || k.control.get<Sequence>(bodyElem)
                           || k.control.get<SetCoordinate>(bodyElem)
                           || k.control.get<LoadTiled>(bodyElem)
                           || k.control.get<Multiply>(bodyElem))
                        {
                            continue;
                        }
                        retval.combine(false,
                                       concatenate("Found unsupported node '",
                                                   bodyElem,
                                                   "' in forloop '",
                                                   forLoop,
                                                   "'."));
                    }
                }
            }

            return retval;
        }
    }
}
