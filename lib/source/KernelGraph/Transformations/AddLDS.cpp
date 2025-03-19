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

/**
@class AddLDS
@brief Add load/store through LDS to the graph; and prefetching.

# Overview

Load/store operations inside loops, and tagged with MemoryType LDS
or WAVE_LDS, are transformed.

An entire tile is loaded once per loop iteration (which may be
unrolled) into LDS.  Subsequent loads in the loop read from LDS.

## Prefetching

Prefetching is automatically applied to tiles loaded through LDS
within an unrolled ForLoop.

Prefetching is controlled by:

1. `bool prefetch` - Enable the prefetch transformation.

2. `int prefetchInFlight` - How many loads are put in-flight.

3. `bool prefetchMixMemOps` - Should global loads be inter-mixed with
    math operations, or isolated?

4. `int prefetchLDSFactor` - Ratio of how many sub-tiles are
   prefetched from LDS before the next unroll segment.

### Single prefetch

Focusing on how the two sets of buffers are operated on, we can view
the control flow of loads vertically as follows

    Buf0      Buf1
    ============== for loop preamble
     PG        PG
     CL
    -------------- barrier
     PL
    ============== unrolled for loop begin (counter += 2)
    -=-=-=-=-=-=-= unroll u=0 segment
     CV
     OPR
     PG        CL
    -------------- barrier
               PL
    -=-=-=-=-=-=-= unroll u=1 segment
               CV
               OPR
     CL        PG
    -------------- barrier
     PL
    ============== for loop end

where

- `PG` denotes _prefetch global_: issuing global to vgpr loads
- `CL` denotes _commit to lds_: vgpr to lds stores
- `PL` denotes _prefetch lds_: issuing lds to vgpr loads
- `CV` denotes _commit to vgpr_: waiting on lds to vgpr loads
- `OPR` denotes _operating on vgpr_: doing math (hopefully many cycles)

The order of these must be: `PG CL PL CV OPR`.

Note that:

1. Within the for-loop, there is always a barrier between a `CL` and a
   `PL`.

2. Global prefetches are not stalled by barriers.

3. Within a unrolled segment; `CV` and `OPR` can be mixed.  The
   rocRoller scheduler will make sure, eg, appropriate wait-counts are
   inserted to enforce dependencies.

4. The `PL` may be a "partial" prefetch.  In this case, the remaining
   loads from LDS into VGPRs happends in the immediately following
   `CV`.

### Two prefetches, unrolled

With two-prefetches, we obtain

    Buf0      Buf1      Buf2
    ======================== for loop preamble
     PG        PG        PG
     CL
    ------------------------ barrier
     PL
    ======================== unrolled for loop begin (counter += 3)
    -=-=-=-=-=-=-=-=-=-=-=-= unroll u=0 segment
     CV
     OPR
     PG        CL
    ------------------------ barrier
               PL
    -=-=-=-=-=-=-=-=-=-=-=-= unroll u=1 segment
               CV
               OPR
               PG        CL
    ------------------------ barrier
                         PL
    -=-=-=-=-=-=-=-=-=-=-=-= unroll u=2 segment
                         CV
                         OPR
     CL                  PG
    ------------------------ barrier
     PL
    ======================== for loop end

In the single pre-fetch mode, we have $U=2$ unrolled segments.  In the
two-prefetch mode, we have $U=3$ unrolled segments.  In both cases,
the scheduling of operations in unrolled segment $u$ are:

    -=-=-=-=-=-=-=-=-=-=-=-=-=- unroll u segment
    CV  [u]
    OPR [u]
    PG  [u]     CL [(u+1) % U]
    --------------------------- barrier
                PL [(u+1) % U]

*/

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDS.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CF = rocRoller::KernelGraph::ControlGraph;
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        using GD = rocRoller::Graph::Direction;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;
        using namespace Register;

        /**
         * @brief For LDS load/stores, follow DataFlow edges from the
         * LDS node to find the associated User node.
         */
        int getLDSOperationTarget(KernelGraph const& k, int opTag)
        {
            auto [target, direction] = getOperationTarget(opTag, k);

            // TODO: Design a better way of binding storage to coordinates
            auto maybeLDS = k.coordinates.get<LDS>(target);
            if(maybeLDS)
            {
                // If target is LDS; it might be a duplicated LDS
                // node.  For the purposes of figuring out required
                // coordinates, use the parent LDS as the target
                // instead.
                auto maybeParentLDS
                    = only(k.coordinates.getOutputNodeIndices(target, CT::isEdge<Duplicate>));
                if(maybeParentLDS)
                    target = *maybeParentLDS;
            }

            auto isDataFlow
                = [&](int tag) -> bool { return k.coordinates.get<DataFlow>(tag).has_value(); };

            while(true)
            {
                auto edge
                    = only(filter(isDataFlow, k.coordinates.getNeighbours(target, direction)));
                if(!edge)
                    break;
                target = *only(k.coordinates.getNeighbours(*edge, direction));
            }

            return target;
        }

        template <Graph::Direction Dir>
        void insertInstead(rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph& coordinates,
                           int                                                       newTag,
                           int                                                       oldTag)
        {
            auto edgeTags = coordinates.getNeighbours<Dir>(oldTag).template to<std::vector>();
            for(auto edgeTag : edgeTags)
            {
                auto edge = coordinates.getElement(edgeTag);

                auto upDirTags = coordinates.getNeighbours<Graph::opposite(Dir)>(edgeTag)
                                     .template to<std::vector>();
                auto downDirTags
                    = coordinates.getNeighbours<Dir>(edgeTag).template to<std::vector>();

                std::replace(upDirTags.begin(), upDirTags.end(), oldTag, newTag);

                coordinates.deleteElement(edgeTag);
                if constexpr(Dir == Graph::Direction::Upstream)
                {
                    coordinates.addElement(edge, downDirTags, upDirTags);
                }
                else
                {
                    coordinates.addElement(edge, upDirTags, downDirTags);
                }
            }
        }

        int duplicateChain(KernelGraph& graph, std::vector<int> const& startNodes)
        {
            return duplicateControlNodes(graph, nullptr, startNodes, [](int x) { return true; })[0];
        }

        /**
         * @brief Container for info related to loading from Global
         * into LDS.
         */
        struct LDSOperationInfo
        {
            int user; // User coordinate
            int globalOperation; // LoadTiled/StoreTiled operation
            int ldsOperation; // StoreLDSTile/LoadLDSTile operation
            int globalChain; // LoadTiled/StoreTiled operation
            int ldsChain; // StoreLDStile/LoadLDSTile operation
        };

        /**
         * Add LDS transformer.
         *
         * Splits LoadTiled operations into:
         * - LoadTiled
         * - StoreLDSTile
         * - LoadLDSTile
         *
         * Similarly for StoreTiled operations.
         */
        struct AddLDSVisitor
        {
            AddLDSVisitor(CommandParametersPtr params, ContextPtr context)
                : m_params(params)
                , m_context(context)
            {
            }

            void stage(KernelGraph const&, int);
            void commit(KernelGraph&);

        private:
            std::set<int> m_operations;

            CommandParametersPtr m_params;
            ContextPtr           m_context;
        };

        /**
         * Add Barrier transformer.
         *
         * Adds Barrier operations for non-prefetched load/store
         * operations that go through LDS.
         */
        struct AddBarrierVisitor
        {
            AddBarrierVisitor(ContextPtr context)
                : m_context(context)
            {
            }

            void stage(KernelGraph const&, int);
            void commit(KernelGraph&);

        private:
            std::set<int> m_storeLDSTileOperations;

            ContextPtr m_context;
        };

        /**
         * Add prefetch transformer.
         */
        struct AddPrefetchVisitor
        {
            AddPrefetchVisitor(CommandParametersPtr params, ContextPtr context)
                : m_params(params)
                , m_context(context)
            {
            }

            void commitForLoop(KernelGraph& graph, int forLoop, int numUnroll);
            void orderMultiplies(KernelGraph& graph, int forLoop, int numUnroll);

            void stage(KernelGraph const& graph);
            void commit(KernelGraph&);

            std::unordered_set<int> storeLDSTileOperations() const;

        private:
            // Keys are: ForLoop tag, Unroll value/segment, LDS tag
            std::map<int, std::map<int, std::map<int, LDSOperationInfo>>> m_info;

            std::map<int, int>                                    m_scopes;
            std::map<int, int>                                    m_prefetchLoops;
            std::map<int, std::map<int, std::unordered_set<int>>> m_prefetchUnrollBodyStarts;
            std::map<int, std::map<int, std::unordered_set<int>>> m_prefetchUnrollBodyEnds;
            std::map<int, std::map<int, std::set<std::tuple<int, int, int>>>>
                                                           m_prefetchFromLDSChains;
            std::map<int, std::map<int, std::vector<int>>> m_loadFromLDSChains;
            std::map<int, std::unordered_set<int>>         m_prefetchDelete;

            std::unordered_set<int> m_storeLDSTileOperations;

            CommandParametersPtr m_params;
            ContextPtr           m_context;

            void trackStores(KernelGraph const& graph, int start);
        };

        /**
         * @brief Find loops (and loads in them) that can be prefetched.
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
                    if(rv.contains(*maybeForLoop))
                        continue;

                    // TODO: Only do the K-Loop for now
                    auto fl = kgraph.control.get<ForLoopOp>(*maybeForLoop);
                    if(fl->loopName != rocRoller::KLOOP)
                        continue;

                    auto forLoopCoord     = getForLoopCoords(*maybeForLoop, kgraph).first;
                    auto maybeUnrollCoord = findUnrollNeighbour(kgraph, forLoopCoord);
                    if(forLoopCoordinates.contains(forLoopCoord) && maybeUnrollCoord)
                    {
                        auto myUnroll = getUnrollValueForOp(kgraph, *maybeUnrollCoord, candidate);

                        if(myUnroll > 0)
                        {
                            Dimension unroll
                                = kgraph.coordinates.get<Unroll>(*maybeUnrollCoord).value();
                            auto unrollSize = getUnsignedInt(evaluate(getSize(unroll)));

                            Log::debug("KernelGraph::AddLDS(): ForLoop {} is a prefetch candidate: "
                                       "{} {} ({})",
                                       *maybeForLoop,
                                       *maybeUnrollCoord,
                                       unrollSize,
                                       candidate);

                            rv[*maybeForLoop] = unrollSize;
                        }
                    }
                }
            }

            return rv;
        }

        KernelGraph AddLDS::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::AddLDS");

            auto graph = original;

            auto visitor = AddLDSVisitor(m_params, m_context);
            for(auto const& loadTag : graph.control.getNodes<LoadTiled>())
                visitor.stage(graph, loadTag);
            for(auto const& storeTag : graph.control.getNodes<StoreTiled>())
                visitor.stage(graph, storeTag);

            visitor.commit(graph);

            return graph;
        }

        ConstraintStatus NoLDSTiles(const KernelGraph& graph)
        {
            ConstraintStatus retval;
            for(auto tag : graph.coordinates.getNodes<MacroTile>())
            {
                auto tile = *graph.coordinates.get<MacroTile>(tag);
                if(tile.memoryType == MemoryType::LDS || tile.memoryType == MemoryType::WAVE_LDS
                   || tile.memoryType == MemoryType::WAVE_Direct2LDS)
                {
                    retval.combine(false, concatenate("Tile has LDS memory type: ", tag));
                }
            }
            return retval;
        }

        std::vector<GraphConstraint> AddLDS::postConstraints() const
        {
            return {NoLDSTiles};
        }

        void AddPrefetchVisitor::trackStores(KernelGraph const& graph, int start)
        {
            for(auto tag : graph.control.depthFirstVisit(start, GD::Downstream))
            {
                auto maybeStoreLDSTile = graph.control.get<StoreLDSTile>(tag);
                if(maybeStoreLDSTile)
                    m_storeLDSTileOperations.insert(tag);
            }
        }

        std::unordered_set<int> AddPrefetchVisitor::storeLDSTileOperations() const
        {
            return m_storeLDSTileOperations;
        }

        KernelGraph AddPrefetch::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::AddPrefetch");

            auto graph = removeRedundantNOPs(removeRedundantBodyEdges(original));

            std::unordered_set<int> storeHasBarrierAlready;

            if(m_params->prefetch)
            {
                auto visitor = AddPrefetchVisitor(m_params, m_context);
                AssertFatal(m_params->unrollK > 1, "KLoop must be unrolled when prefetching.");
                visitor.stage(graph);
                visitor.commit(graph);

                storeHasBarrierAlready = visitor.storeLDSTileOperations();
            }

            auto barrierVisitor = AddBarrierVisitor(m_context);
            for(auto const& tag : graph.control.getNodes<StoreLDSTile>())
            {
                if(!storeHasBarrierAlready.contains(tag))
                    barrierVisitor.stage(graph, tag);
            }
            barrierVisitor.commit(graph);

            return graph;
        }

        void AddLDSVisitor::stage(KernelGraph const& k, int opTag)
        {
            auto [userTag, user] = k.getDimension<User>(opTag);
            auto [tileTag, tile] = k.getDimension<MacroTile>(opTag);

            if(!(tile.memoryType == MemoryType::WAVE_LDS || tile.memoryType == MemoryType::LDS
                 || tile.memoryType == MemoryType::WAVE_Direct2LDS))
                return;

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::AddLDS()::stage({}): User {}, MacroTile {}", opTag, userTag, tileTag);

            m_operations.insert(opTag);
        }

        void AddLDSVisitor::commit(KernelGraph& k)
        {
            //
            // Commit: Create operations and DataFlow
            //
            for(auto opTag : m_operations)
            {
                Log::debug("KernelGraph::AddLDS()::commit({})", opTag);

                auto isLoad       = k.control.get<LoadTiled>(opTag).has_value();
                auto tileTag      = k.mapper.get<MacroTile>(opTag);
                auto userTag      = k.mapper.get<User>(opTag);
                auto varType      = getVariableType(k, opTag);
                auto tile         = k.coordinates.getNode<MacroTile>(tileTag);
                auto isDirect2LDS = tile.memoryType == MemoryType::WAVE_Direct2LDS;

                // Create new coordinates
                auto              ldsTag      = k.coordinates.addElement(LDS(isDirect2LDS));
                std::vector<uint> jammedTiles = {1, 1};
                bool              splitStore  = false;

                if(!isLoad)
                {
                    // For StoreTiled operations, the waves are
                    // "serialized" in OrderEpilogueBlocks, so we can
                    // use a smaller internal tile.
                    jammedTiles = m_params->getWaveTilesPerWavefront();
                    splitStore  = m_params->getSplitStoreTileIntoWaveBlocks();
                }

                auto internalTag = createInternalTile(
                    k, varType, tileTag, jammedTiles, splitStore, m_params, m_context);

                // Connect coordinates with DataFlow edges
                insertInstead<Graph::Direction::Downstream>(k.coordinates, internalTag, tileTag);
                k.coordinates.addElement(DataFlow(), {internalTag}, {ldsTag});
                k.coordinates.addElement(DataFlow(), {ldsTag}, {tileTag});

                // Create new operations and update old operation
                bool isTransposedTile = false;
                if(isLoad)
                {
                    auto loadTile    = k.control.get<LoadTiled>(opTag).value();
                    isTransposedTile = loadTile.isTransposedTile;
                }
                auto loadLDSOp  = k.control.addElement(LoadLDSTile(varType, isTransposedTile));
                auto storeLDSOp = k.control.addElement(StoreLDSTile(varType.dataType));

                // Update tile
                if(isDirect2LDS)
                    tile.memoryType = MemoryType::WAVE;
                if(tile.memoryType == MemoryType::WAVE_LDS)
                    tile.memoryType = MemoryType::WAVE;
                if(tile.memoryType == MemoryType::LDS)
                    tile.memoryType = MemoryType::VGPR;
                k.coordinates.setElement(tileTag, tile);

                if(!isLoad)
                {
                    // For StoreTiled operations, the waves are
                    // "serialized" in OrderEpilogueBlocks, so we can
                    // use a smaller tile size.
                    tile.sizes[0] /= m_params->getWaveTilesPerWavefront()[0];
                    tile.sizes[1] /= m_params->getWaveTilesPerWavefront()[1];
                    k.coordinates.setElement(tileTag, tile);
                }

                // Update connections and connect operations
                k.control.addElement(Sequence(), {storeLDSOp}, {loadLDSOp});

                k.mapper.purge(opTag);
                k.mapper.connect<User>(opTag, userTag);
                k.mapper.connect<LDS>(storeLDSOp, ldsTag);
                k.mapper.connect<User>(loadLDSOp, userTag); // For F6 Padding
                k.mapper.connect<LDS>(loadLDSOp, ldsTag);

                if(isDirect2LDS)
                    k.mapper.connect<LDS>(opTag, ldsTag);

                if(isLoad)
                {
                    insertAfter(k, opTag, storeLDSOp, loadLDSOp);

                    k.mapper.connect<MacroTile>(opTag, internalTag);
                    k.mapper.connect<MacroTile>(storeLDSOp, internalTag);
                    k.mapper.connect<MacroTile>(loadLDSOp, tileTag);
                }
                else
                {
                    insertBefore(k, opTag, storeLDSOp, loadLDSOp);

                    k.mapper.connect<MacroTile>(storeLDSOp, tileTag);
                    k.mapper.connect<MacroTile>(loadLDSOp, internalTag);
                    k.mapper.connect<MacroTile>(opTag, internalTag);
                }
            }
        }

        void AddBarrierVisitor::stage(KernelGraph const& graph, int opTag)
        {
            auto maybeStoreLDSTile = graph.control.get<StoreLDSTile>(opTag);
            if(!maybeStoreLDSTile)
                return;
            m_storeLDSTileOperations.insert(opTag);
        }

        void AddBarrierVisitor::commit(KernelGraph& graph)
        {
            for(auto storeLDSTileTag : m_storeLDSTileOperations)
            {
                auto preBarrier  = graph.control.addElement(Barrier());
                auto postBarrier = graph.control.addElement(Barrier());

                insertBefore(graph, storeLDSTileTag, preBarrier, preBarrier);
                insertAfter(graph, storeLDSTileTag, postBarrier, postBarrier);
            }
        }

        void AddPrefetchVisitor::commit(KernelGraph& k)
        {
            Log::debug("KernelGraph::AddPrefetch()::commit()");

            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                commitForLoop(k, forLoop, numUnroll);
            }
        }

        void AddPrefetchVisitor::orderMultiplies(KernelGraph& graph, int forLoop, int u)
        {
            auto starts = m_prefetchUnrollBodyStarts[forLoop][u];

            std::map<int, int> loadMap;
            for(auto node : graph.control.depthFirstVisit(starts))
            {
                auto maybeLoadLDSTile = graph.control.get<LoadLDSTile>(node);
                if(!maybeLoadLDSTile)
                    continue;
                auto tileTag     = graph.mapper.get<MacroTile>(node);
                loadMap[tileTag] = getTopSetCoordinate(graph, node);
            }

            for(auto node : graph.control.depthFirstVisit(starts))
            {
                auto maybeMultiply = graph.control.get<Multiply>(node);
                if(!maybeMultiply)
                    continue;
                auto [macATag, macA] = graph.getDimension<MacroTile>(
                    node, Connections::typeArgument<MacroTile>(NaryArgument::LHS));
                auto [macBTag, macB] = graph.getDimension<MacroTile>(
                    node, Connections::typeArgument<MacroTile>(NaryArgument::RHS));

                if(loadMap.contains(macATag))
                {
                    graph.control.addElement(Sequence(), {loadMap[macATag]}, {node});
                    m_prefetchUnrollBodyStarts[forLoop][u].erase(node);
                }
                if(loadMap.contains(macBTag))
                {
                    graph.control.addElement(Sequence(), {loadMap[macBTag]}, {node});
                    m_prefetchUnrollBodyStarts[forLoop][u].erase(node);
                }
            }
        }

        void AddPrefetchVisitor::commitForLoop(KernelGraph& graph, int forLoop, int numUnroll)
        {
            auto logger = rocRoller::Log::getLogger();
            logger->debug("KernelGraph::AddPrefetch()::commitForLoop({})", forLoop);

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

            for(int u = 0; u < numUnroll; ++u)
                orderMultiplies(graph, forLoop, u);

            // At this point, each of the unrolled loop bodies are
            // detached and isolated from the rest of the graph.

            std::map<int, std::vector<LDSOperationInfo>> loadsByUnroll;
            for(int u = 0; u < numUnroll; u++)
            {
                for(auto [target, info] : m_info[forLoop][u])
                    loadsByUnroll[u].push_back(info);
            }

            AssertFatal(loadsByUnroll.size() == numUnroll);

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

            int numInFlight = m_params->prefetchInFlight;

            // Loads first
            for(int u = 0; u < numInFlight; ++u)
            {
                for(auto load : loadsByUnroll[u])
                {
                    logger->debug(
                        "  prefetch: pre-loop global load: unroll {} user {}", u, load.user);
                    auto loadChain = duplicateChain(graph, {load.globalChain});
                    preChain.push_back(loadChain);
                }
            }

            // StoreLDS next
            for(auto load : loadsByUnroll[0])
            {
                logger->debug("  prefetch: pre-loop commit lds: unroll {} user {}", 0, load.user);
                auto storeChain = duplicateChain(graph, {load.ldsChain});
                trackStores(graph, storeChain);
                preChain.push_back(storeChain);
            }

            graph.control.addElement(Body(), {scope}, {preChain[0]});
            for(uint i = 1; i < preChain.size(); ++i)
            {
                graph.control.addElement(Sequence(), {preChain[i - 1]}, {preChain[i]});
            }
            graph.control.addElement(Sequence(), {preChain.back()}, {preBarrier});

            auto addLDSPrefetchChains = [&](int u, int pre, int post, bool duplicate) -> int {
                std::vector<int> prefetchChain;
                for(auto [_ignore1, _ignore2, chain] : m_prefetchFromLDSChains[forLoop][u])
                {
                    int dchain = duplicate ? duplicateChain(graph, {chain}) : chain;
                    prefetchChain.push_back(dchain);
                }

                AssertFatal(!prefetchChain.empty());

                logger->debug("  prefetch: lds prefetch: ordering {} to {} (top)",
                              pre,
                              prefetchChain.front());
                graph.control.addElement(Sequence(), {pre}, {prefetchChain.front()});
                for(uint i = 1; i < prefetchChain.size(); ++i)
                {
                    logger->debug("  prefetch: lds prefetch: ordering {} to {} (chain)",
                                  prefetchChain[i - 1],
                                  prefetchChain[i]);
                    graph.control.addElement(
                        Sequence(), {prefetchChain[i - 1]}, {prefetchChain[i]});
                }
                logger->debug("  prefetch: lds prefetch: ordering {} to {} (bottom)",
                              prefetchChain.back(),
                              post);
                graph.control.addElement(Sequence(), {prefetchChain.back()}, {post});

                return prefetchChain.front();
            };

            if(!m_prefetchFromLDSChains[forLoop].empty())
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

                for(auto load : loadsByUnroll[prefetchGlobalU])
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

            auto separateMemOps = !m_params->prefetchMixMemOps;

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

                auto toOrder = filter(graph.control.isElemType<LoadLDSTile>(),
                                      graph.control.depthFirstVisit(segmentBoundaries[u],
                                                                    Graph::Direction::Downstream))
                                   .to<std::vector>();
                orderMemoryNodes(graph, toOrder, false);

                auto globalPrefetchU = (u + numInFlight) % numUnroll;
                auto ldsPrefetchU    = (u + 1) % numUnroll;
                auto barrier         = graph.control.addElement(Barrier());

                auto nop = separateMemOps ? graph.control.addElement(NOP()) : -1;

                // Issue global loads
                auto globalLoads = loadsByUnroll[globalPrefetchU];
                logger->debug("  prefetch: in-loop: issue global loads {}",
                              globalLoads[0].globalChain);
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
                auto globalStores = loadsByUnroll[ldsPrefetchU];
                trackStores(graph, globalStores[0].ldsChain);
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
                    trackStores(graph, globalStores[i].ldsChain);
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
                int firstPrefetchFromLDS = -1;
                if(m_prefetchFromLDSChains[forLoop].contains(ldsPrefetchU))
                {
                    firstPrefetchFromLDS = addLDSPrefetchChains(
                        ldsPrefetchU, barrier, segmentBoundaries[u + 1], false);
                }
                else
                {
                    graph.control.addElement(Sequence(), {barrier}, {segmentBoundaries[u + 1]});
                }

                // To ensure proper memory ordering, the last
                // load-from-lds chain of the current segment must
                // finish before the first prefetch-from-lds chain
                // starts (which prefetchs for the next segment; but
                // appears in the current segment)
                if(firstPrefetchFromLDS != -1)
                {
                    int lastLoadFromLDS = -1;
                    for(auto tagA : m_loadFromLDSChains[forLoop][u])
                    {
                        bool isLast = true;
                        for(auto tagB : m_loadFromLDSChains[forLoop][u])
                        {
                            if(tagA == tagB)
                                continue;
                            if(graph.control.compareNodes(tagA, tagB) == NodeOrdering::LeftFirst)
                            {
                                isLast = false;
                                break;
                            }
                        }
                        if(isLast)
                        {
                            lastLoadFromLDS = tagA;
                            break;
                        }
                    }
                    Log::debug("  prefetch: in-loop: lds ordering {} to {}",
                               lastLoadFromLDS,
                               firstPrefetchFromLDS);
                    graph.control.addElement(Sequence(), {lastLoadFromLDS}, {firstPrefetchFromLDS});

                    // The last load-from-lds for the current
                    // iteration must also finish before the barrier.
                    // If not, an eager wave may enter the next
                    // segment and over-write LDS.
                    //
                    // For prefetch > 2 this is not necessary.
                    if(numInFlight <= 2)
                        graph.control.addElement(Sequence(), {lastLoadFromLDS}, {barrier});
                }

                // Connect the segment to the proceeding segment boundary
                if(separateMemOps)
                {
                    for(auto tag : m_prefetchUnrollBodyEnds[forLoop][u])
                        graph.control.addElement(Sequence(), {tag}, {nop});
                }
                else
                {
                    for(auto tag : m_prefetchUnrollBodyEnds[forLoop][u])
                        graph.control.addElement(Sequence(), {tag}, {segmentBoundaries[u + 1]});
                }
            }
        }

        void AddPrefetchVisitor::stage(KernelGraph const& k)
        {
            m_prefetchLoops = findPrefetch(k);

            auto colouring = colourByUnrollValue(k);
            auto isBody    = k.control.isElemType<Body>();

            std::map<int, int> unrollCoordSizes;
            {
                for(auto unrollTag : k.coordinates.getNodes<Unroll>())
                {
                    auto unroll                 = *k.coordinates.get<Unroll>(unrollTag);
                    unrollCoordSizes[unrollTag] = getUnsignedInt(evaluate(unroll.size));
                }
            }

            // Map: Operation (in loop body) to Unroll coordinate value
            std::map<int, int> operationUnroll;

            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                auto forLoopCoord     = getForLoopCoords(forLoop, k).first;
                auto maybeUnrollCoord = findUnrollNeighbour(k, forLoopCoord);
                AssertFatal(maybeUnrollCoord, "Prefetch with no unroll coordinate.");
                auto unrollCoord = *maybeUnrollCoord;

                Log::debug("AddPrefetch::stage: Unroll coordinate is {}", unrollCoord);

                for(auto [opTag, opColours] : colouring.operationColour)
                {
                    if(opColours.contains(unrollCoord))
                        operationUnroll[opTag] = opColours[unrollCoord];
                }
            }

            auto alreadySeen = std::unordered_set<int>();

            //
            // Find global loads, LDS stores, and remaining (which
            // will be top of unroll bodies)
            //
            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                auto bodyEdges = filter(isBody, k.control.getNeighbours<GD::Downstream>(forLoop))
                                     .to<std::vector>();

                // Find global loads and detach them
                auto isLoadTiled = k.control.isElemType<LoadTiled>();
                for(auto loadTag : k.control.findNodes(bodyEdges, isLoadTiled, GD::Downstream))
                {
                    auto top = getTopSetCoordinate(k, loadTag);
                    for(auto edge : k.control.getNeighbours(top, GD::Upstream))
                        m_prefetchDelete[forLoop].insert(edge);
                    for(auto edge : k.control.getNeighbours(top, GD::Downstream))
                    {
                        if(!isBody(edge))
                            m_prefetchDelete[forLoop].insert(edge);
                    }

                    auto user   = k.mapper.get<User>(loadTag);
                    auto target = getLDSOperationTarget(k, loadTag);

                    AssertFatal(user == target);

                    m_info[forLoop][operationUnroll[loadTag]][target].user            = user;
                    m_info[forLoop][operationUnroll[loadTag]][target].globalOperation = loadTag;
                    m_info[forLoop][operationUnroll[loadTag]][target].globalChain     = top;

                    Log::debug(
                        "AddPrefetch::stage: Global load operation {} top {} target {} user {}",
                        loadTag,
                        top,
                        target,
                        user);

                    alreadySeen.insert(top);
                }

                // Find LDS stores and detach them
                auto isStoreLDSTile = k.control.isElemType<StoreLDSTile>();
                for(auto storeLDSTag :
                    k.control.findNodes(bodyEdges, isStoreLDSTile, GD::Downstream))
                {
                    auto top = getTopSetCoordinate(k, storeLDSTag);
                    for(auto edge : k.control.getNeighbours(top, GD::Upstream))
                        m_prefetchDelete[forLoop].insert(edge);
                    for(auto edge : k.control.getNeighbours(top, GD::Downstream))
                    {
                        if(!isBody(edge))
                            m_prefetchDelete[forLoop].insert(edge);
                    }

                    auto target = getLDSOperationTarget(k, storeLDSTag);

                    m_info[forLoop][operationUnroll[storeLDSTag]][target].ldsOperation
                        = storeLDSTag;
                    m_info[forLoop][operationUnroll[storeLDSTag]][target].ldsChain = top;

                    Log::debug("AddPrefetch::stage: LDS store operation {} top {} target {}",
                               storeLDSTag,
                               top,
                               target);

                    alreadySeen.insert(top);
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

            // Find things that haven't been seen yet (so aren't
            // global loads or LDS stores) and that aren't in a
            // cluster, and mark them as the body-starts
            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                auto headless = [&](int x) {
                    if(alreadySeen.contains(x))
                        return false;
                    for(auto incomingEdge : k.control.getNeighbours<GD::Upstream>(x))
                        if(!m_prefetchDelete[forLoop].contains(incomingEdge))
                            return false;
                    if(!operationUnroll.contains(x))
                        return false;
                    return true;
                };

                auto bodyEdges = filter(isBody, k.control.getNeighbours<GD::Downstream>(forLoop))
                                     .to<std::vector>();

                for(auto start : k.control.findNodes(bodyEdges, headless, GD::Downstream))
                    m_prefetchUnrollBodyStarts[forLoop][operationUnroll[start]].insert(start);
            }

            //
            // Within each segment, determine which LoadLDSTile
            // operations should be prefetched.
            //
            // That is, some LoadLDSTile operation from the next
            // segment can be moved into (prefetched) the current
            // segment.
            //
            int splitLDSPrefetchFactor = m_params->prefetchLDSFactor;
            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                auto forLoopCoord     = getForLoopCoords(forLoop, k).first;
                auto maybeUnrollCoord = findUnrollNeighbour(k, forLoopCoord);
                AssertFatal(maybeUnrollCoord, "Prefetch with no unroll coordinate.");
                auto prefetchUnrollCoord = *maybeUnrollCoord;

                auto bodyEdges = filter(isBody, k.control.getNeighbours<GD::Downstream>(forLoop))
                                     .to<std::vector>();
                auto isLoadLDSTile = k.control.isElemType<LoadLDSTile>();

                int prefetchLDSUnrollCoord = -1;
                {
                    // LDS prefetch colouring is based on the "small k", not jammed coordinates.
                    for(auto loadLDSTileTag :
                        k.control.findNodes(bodyEdges, isLoadLDSTile, GD::Downstream))
                    {
                        auto colour = colouring.operationColour[loadLDSTileTag];
                        for(auto kv : colour)
                        {
                            auto unrollCoord = kv.first;
                            if(unrollCoord == prefetchUnrollCoord)
                                continue;
                            auto incoming
                                = k.coordinates.getInputNodeIndices(unrollCoord, CT::isEdge<Split>)
                                      .to<std::vector>();
                            auto incomingJammed
                                = filterCoordinates<JammedWaveTileNumber>(incoming, k);
                            if(incomingJammed.empty())
                            {
                                prefetchLDSUnrollCoord = unrollCoord;
                            }
                        }
                    }
                }
                AssertFatal(prefetchLDSUnrollCoord != -1, "Can not find LDS prefetch coordinate.");

                for(auto loadLDSTileTag :
                    k.control.findNodes(bodyEdges, isLoadLDSTile, GD::Downstream))
                {
                    auto colour = colouring.operationColour[loadLDSTileTag];
                    auto u      = colour[prefetchUnrollCoord];

                    auto prefetchLDSUnrollValue = colour[prefetchLDSUnrollCoord];

                    auto loadLDSTileChain = getTopSetCoordinate(k, loadLDSTileTag);

                    if(splitLDSPrefetchFactor == 0)
                    {
                        // Keep load in same segment
                        m_loadFromLDSChains[forLoop][u].push_back(loadLDSTileChain);
                        break;
                    }

                    if(splitLDSPrefetchFactor > 0)
                    {
                        if(prefetchLDSUnrollValue
                           >= unrollCoordSizes[prefetchLDSUnrollCoord] / splitLDSPrefetchFactor)
                        {
                            // Keep load in same segment
                            m_loadFromLDSChains[forLoop][u].push_back(loadLDSTileChain);
                            continue;
                        }
                    }

                    // Move load to previous segment (it is prefetchable)
                    auto target = getLDSOperationTarget(k, loadLDSTileTag);
                    m_prefetchFromLDSChains[forLoop][u].insert(
                        {target, prefetchLDSUnrollValue, loadLDSTileChain});
                    m_prefetchUnrollBodyStarts[forLoop][u].erase(loadLDSTileChain);

                    for(auto inEdge : k.control.getNeighbours<GD::Upstream>(loadLDSTileChain))
                    {
                        m_prefetchDelete[forLoop].insert(inEdge);
                    }

                    for(auto outEdge :
                        filter(k.control.isElemType<Sequence>(),
                               k.control.getNeighbours<GD::Downstream>(loadLDSTileChain)))
                    {
                        if(m_prefetchDelete[forLoop].contains(outEdge))
                            continue;

                        auto outNode = *only(k.control.getNeighbours<GD::Downstream>(outEdge));

                        m_prefetchUnrollBodyStarts[forLoop][u].insert(outNode);
                        m_prefetchDelete[forLoop].insert(outEdge);
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
                           || k.control.get<LoadLDSTile>(bodyElem)
                           || k.control.get<StoreLDSTile>(bodyElem)
                           || k.control.get<Multiply>(bodyElem) || k.control.get<NOP>(bodyElem))
                        {
                            continue;
                        }

                        auto op = k.control.getNode(bodyElem);
                        retval.combine(false,
                                       concatenate("Found unsupported node ",
                                                   toString(op),
                                                   " (",
                                                   bodyElem,
                                                   ") in forloop ",
                                                   forLoop,
                                                   "."));
                    }
                }
            }

            return retval;
        }
    }
}
