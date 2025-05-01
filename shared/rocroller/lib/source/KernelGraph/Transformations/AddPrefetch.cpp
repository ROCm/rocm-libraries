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
@class AddPrefetch
@brief Add prefetching for load operations.

# Overview

Prefetching is automatically applied to tiles loaded through LDS
within an unrolled ForLoop.

Prefetching is controlled by:

1. `bool prefetch` - Enable the prefetch transformation.

2. `int prefetchInFlight` - How many loads are put in-flight.

3. `bool prefetchMixMemOps` - Should global loads be inter-mixed with
    math operations, or isolated?

4. `int prefetchLDSFactor` - Ratio of how many sub-tiles are
   prefetched from LDS before the next unroll segment.

## Single prefetch

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

## Two prefetches, unrolled

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
#include <rocRoller/KernelGraph/Transforms/AddPrefetch.hpp>
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
            void orderLoadsBeforeMultiplies(KernelGraph& graph, int forLoop, int numUnroll);

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

            std::map<int, std::map<int, std::vector<int>>> m_deferredToOrder;

            std::map<int, int> m_exchangeLoadMap;

            CommandParametersPtr m_params;
            ContextPtr           m_context;

            void trackStores(KernelGraph const& graph, int start);
        };

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

                            Log::debug(
                                "KernelGraph::AddPrefetch(): ForLoop {} is a prefetch candidate: "
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

            auto graph = original;
            removeRedundantBodyEdges(graph);
            removeRedundantNOPs(graph);

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

        void AddPrefetchVisitor::commit(KernelGraph& k)
        {
            Log::debug("KernelGraph::AddPrefetch()::commit()");

            for(auto [forLoop, numUnroll] : m_prefetchLoops)
            {
                commitForLoop(k, forLoop, numUnroll);
            }

            removeRedundantSequenceEdges(k);
            removeRedundantBodyEdges(k);
        }

        /**
        * @brief Order loads before Multiplies; and record direct
        * load operations within the segment that need to be ordered.
        *
        * We can't order direct loads just yet, as the graph might be
        * in an invalid state when orderLoadsBeforeMultiplies is
        * called.
        */
        void AddPrefetchVisitor::orderLoadsBeforeMultiplies(KernelGraph& graph, int forLoop, int u)
        {
            auto starts = m_prefetchUnrollBodyStarts[forLoop][u];

            auto isLoadPredicate = [&graph](int x) {
                return graph.control.get<LoadTiled>(x).has_value()
                       || graph.control.get<LoadLDSTile>(x).has_value();
            };

            auto isMultiplyPredicate = graph.control.isElemType<Multiply>();

            auto isLHSRHSPredicate = [](Connections::ConnectionSpec const& spec) -> bool {
                auto nary = std::visit(
                    rocRoller::overloaded{[](Connections::JustNaryArgument arg) {
                                              return std::optional<NaryArgument>{arg.argument};
                                          },
                                          [](Connections::TypeAndNaryArgument arg) {
                                              return std::optional<NaryArgument>{arg.argument};
                                          },
                                          [](auto x) { return std::optional<NaryArgument>{}; }},
                    spec);
                if(!nary)
                    return false;
                return *nary == NaryArgument::LHS || *nary == NaryArgument::RHS
                       || *nary == NaryArgument::LHS_SCALE || *nary == NaryArgument::RHS_SCALE;
            };

            std::map<int, int> loadMap = m_exchangeLoadMap;
            for(auto loadTag : graph.control.findNodes(starts, isLoadPredicate))
            {
                auto tileTag     = graph.mapper.get<MacroTile>(loadTag);
                loadMap[tileTag] = getTopSetCoordinate(graph, loadTag);
            }

            auto isExchangePredicate = graph.control.isElemType<Exchange>();

            for(auto exchangeTag : graph.control.findNodes(starts, isExchangePredicate))
            {
                for(auto conn : graph.mapper.getConnections(exchangeTag))
                {
                    auto coord = only(graph.coordinates.getOutputNodeIndices(conn.coordinate,
                                                                             CT::isEdge<CT::Index>))
                                     .value_or(conn.coordinate);
                    if(not loadMap.contains(coord))
                        continue;

                    Log::debug("Adding load-before-exchange Sequence edge from {} to {} for {}",
                               loadMap[coord],
                               exchangeTag,
                               toString(conn.connection));

                    graph.control.addElement(Sequence(), {loadMap[coord]}, {exchangeTag});
                    m_exchangeLoadMap[coord] = loadMap[coord];
                }
                m_prefetchUnrollBodyStarts[forLoop][u].erase(exchangeTag);
            }

            for(auto exchangeTag : graph.control.findNodes(starts, isExchangePredicate))
            {
                auto destTileTag = graph.mapper.get(exchangeTag, NaryArgument::DEST);
                auto tileTags
                    = graph.coordinates.getInputNodeIndices(destTileTag, CT::isEdge<Index>)
                          .to<std::vector>();
                AssertFatal(!tileTags.empty(), "swizzle indexed tiles not found");
                for(auto tileTag : tileTags)
                    loadMap[tileTag] = getTopSetCoordinate(graph, exchangeTag);
            }

            for(auto multiplyTag : graph.control.findNodes(starts, isMultiplyPredicate))
            {
                for(auto conn : graph.mapper.getConnections(multiplyTag))
                {
                    // Multiply arguments may have been loaded by LDS
                    // prefetching, or may be single-scales from a
                    // LoadSGPR operations.
                    //
                    // These arguments won't be in the loadMap, and we
                    // don't need to order them here.
                    if(not loadMap.contains(conn.coordinate))
                        continue;
                    if(not isLHSRHSPredicate(conn.connection))
                        continue;

                    Log::debug("Adding load-before-multiply Sequence edge from {} to {} for {}",
                               loadMap[conn.coordinate],
                               multiplyTag,
                               toString(conn.connection));

                    graph.control.addElement(Sequence(), {loadMap[conn.coordinate]}, {multiplyTag});
                }
                m_prefetchUnrollBodyStarts[forLoop][u].erase(multiplyTag);
            }

            auto loads
                = filter(graph.control.isElemType<LoadTiled>(),
                         graph.control.depthFirstVisit(m_prefetchUnrollBodyStarts[forLoop][u],
                                                       Graph::Direction::Downstream))
                      .to<std::vector>();
            for(auto x : loads)
            {
                m_deferredToOrder[forLoop][u].push_back(getTopSetCoordinate(graph, x));
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

            // Unrolled loop over prefetch segments
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
                            if(graph.control.compareNodes(rocRoller::UpdateCache, tagA, tagB)
                               == NodeOrdering::LeftFirst)
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

                    if(lastLoadFromLDS != -1)
                    {
                        Log::debug("  prefetch: in-loop: lds ordering {} to {}",
                                   lastLoadFromLDS,
                                   firstPrefetchFromLDS);
                        graph.control.addElement(
                            Sequence(), {lastLoadFromLDS}, {firstPrefetchFromLDS});

                        // The last load-from-lds for the current
                        // iteration must also finish before the barrier.
                        // If not, an eager wave may enter the next
                        // segment and over-write LDS.
                        //
                        // For prefetch > 2 this is not necessary.
                        if(numInFlight <= 2)
                            graph.control.addElement(Sequence(), {lastLoadFromLDS}, {barrier});
                    }
                }

                orderLoadsBeforeMultiplies(graph, forLoop, u);

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

            for(uint u = 0; u < numUnroll; ++u)
            {
                orderMemoryNodes(graph, m_deferredToOrder[forLoop][u], false);
            }
        }

        void AddPrefetchVisitor::stage(KernelGraph const& k)
        {
            m_prefetchLoops = findPrefetch(k);

            auto colouring       = colourByUnrollValue(k);
            auto isBodyPredicate = k.control.isElemType<Body>();

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
                auto bodyEdges
                    = filter(isBodyPredicate, k.control.getNeighbours<GD::Downstream>(forLoop))
                          .to<std::vector>();

                // Find LDS stores and mark them for detachment
                auto isStoreLDSTile = k.control.isElemType<StoreLDSTile>();
                for(auto storeLDSTag :
                    k.control.findNodes(bodyEdges, isStoreLDSTile, GD::Downstream))
                {
                    auto top = getTopSetCoordinate(k, storeLDSTag);
                    for(auto edge : k.control.getNeighbours(top, GD::Upstream))
                        m_prefetchDelete[forLoop].insert(edge);
                    for(auto edge : k.control.getNeighbours(top, GD::Downstream))
                    {
                        if(!isBodyPredicate(edge))
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

                // Find global loads and detach them
                auto isLoadTiled = k.control.isElemType<LoadTiled>();
                for(auto loadTag : k.control.findNodes(bodyEdges, isLoadTiled, GD::Downstream))
                {
                    // If there isn't an info entry yet, then there
                    // isn't a matching StoreLDSTile operation.  In
                    // this case, LDS isn't being used for this User
                    // coordinate; don't try pre-fetching it.
                    auto user = k.mapper.get<User>(loadTag);
                    if(!m_info[forLoop][operationUnroll[loadTag]].contains(user))
                    {
                        if(m_params->prefetchMixMemOps)
                        {
                            Throw<FatalError>(
                                "AddPrefetch: A direct load (not through LDS) was detected, "
                                "and memory-operation mixing is enabled.  The AddPrefetch pass "
                                "can not continue.  To remedy this: ensure that all loads have LDS "
                                "enabled OR disable memory operation mixing (prefetchMixMemOps).");

                            // The problem is...
                            //
                            // We add LoadTile operations above the
                            // ForLoop to prefetch the first set of
                            // tiles.  These are in-flight across the
                            // top of the loop boundary.
                            //
                            // Now consider the last segment.  If
                            // memory operations are allowed to be
                            // mixed AND a direct load appears before
                            // a multiply, then this direct load will
                            // force the mixed-in prefetch loads that
                            // are in-flight to complete.
                            //
                            // Then, at the bottom of the loop nothing
                            // will be in-flight.
                            //
                            // This is inconsistent with the top of
                            // the loop.
                            //
                        }
                        Log::debug("AddPrefetch::stage: Skipping global non-LDS load operation {}",
                                   loadTag);
                        continue;
                    }

                    auto top = getTopSetCoordinate(k, loadTag);
                    for(auto edge : k.control.getNeighbours(top, GD::Upstream))
                        m_prefetchDelete[forLoop].insert(edge);
                    for(auto edge : k.control.getNeighbours(top, GD::Downstream))
                    {
                        if(!isBodyPredicate(edge))
                            m_prefetchDelete[forLoop].insert(edge);
                    }

                    m_info[forLoop][operationUnroll[loadTag]][user].user            = user;
                    m_info[forLoop][operationUnroll[loadTag]][user].globalOperation = loadTag;
                    m_info[forLoop][operationUnroll[loadTag]][user].globalChain     = top;

                    Log::debug("AddPrefetch::stage: Global load operation {} top {} user {}",
                               loadTag,
                               top,
                               user);

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

                auto bodyEdges
                    = filter(isBodyPredicate, k.control.getNeighbours<GD::Downstream>(forLoop))
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

                auto bodyEdges
                    = filter(isBodyPredicate, k.control.getNeighbours<GD::Downstream>(forLoop))
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

                Log::debug("AddPrefetch::stage: LDS prefetch: factor {} Unroll "
                           "coordinate {} size {}",
                           splitLDSPrefetchFactor,
                           prefetchLDSUnrollCoord,
                           unrollCoordSizes[prefetchLDSUnrollCoord]);

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
                        continue;
                    }

                    if(splitLDSPrefetchFactor > 0)
                    {
                        if(prefetchLDSUnrollValue
                           >= unrollCoordSizes[prefetchLDSUnrollCoord] / splitLDSPrefetchFactor)
                        {
                            // Keep load in same segment
                            Log::debug(
                                "AddPrefetch::stage: LDS load operation {} keep (unroll value {})",
                                loadLDSTileTag,
                                prefetchLDSUnrollValue);
                            m_loadFromLDSChains[forLoop][u].push_back(loadLDSTileChain);
                            continue;
                        }
                    }

                    // Move load to previous segment (it is prefetchable)
                    Log::debug("AddPrefetch::stage: LDS load operation {} move (unroll value {})",
                               loadLDSTileTag,
                               prefetchLDSUnrollValue);

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
                           || k.control.get<Exchange>(bodyElem) || k.control.get<Multiply>(bodyElem)
                           || k.control.get<NOP>(bodyElem))
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
