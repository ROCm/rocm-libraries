// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/PrefetchScale.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelOptions_detail.hpp>

// TODO: Extract most of this to a detail header file and add tests

namespace rocRoller
{
    namespace KernelGraph
    {
        using GD = rocRoller::Graph::Direction;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        /**
         * Represents the transition point between loads with different K unroll values.
         */
        struct LoadBoundary
        {
            int prevLoad; /**< Last load before K iteration changes */
            int nextLoad; /**< First load after K iteration changes */
        };

        /**
         * Finds where the unroll K value changes between consecutive loads.
         *
         * @param graph Kernel graph containing the loads
         * @param colouring Maps operations to unroll values
         * @param loads Load operations to scan for the unroll K value change
         * @return LoadBoundary with the loads before and after the unroll K value change
         */
        LoadBoundary FindLoadBoundary(KernelGraph const&      graph,
                                      UnrollColouring const&  colouring,
                                      std::vector<int> const& loads)
        {
            std::optional<int> unrollKVal;
            std::optional<int> prevLoad;
            std::optional<int> nextLoad;

            for(auto const load : loads)
            {
                auto unrollMap  = colouring.operationColour.at(load);
                auto unrollKDim = graph.mapper.get<Unroll>(load, rocRoller::KLOOP_UNROLL);
                if(unrollKVal && *unrollKVal != unrollMap.at(unrollKDim))
                {
                    nextLoad = load;
                    break;
                }
                unrollKVal = unrollMap.at(unrollKDim);
                prevLoad   = load;
            }

            AssertFatal(prevLoad && nextLoad, "couldn't find a position to in-loop scale prefetch");

            return LoadBoundary{*prevLoad, *nextLoad};
        }

        /**
         * Separates loads by memory type: WAVE_SWIZZLE vs standard layouts.
         */
        struct SwizzlePartition
        {
            std::vector<int> swizzleLoads;
            std::vector<int> nonSwizzleLoads;
        };

        /**
         * Groups loads by memory type to apply appropriate prefetch strategies.
         *
         * @param graph Kernel graph containing the loads
         * @param loads Load operation tags to group
         * @return SwizzlePartition with loads separated by memory type
         */
        SwizzlePartition GroupLoadsBySwizzle(KernelGraph const&      graph,
                                             std::vector<int> const& loads)
        {
            SwizzlePartition result;
            for(auto const loadTag : loads)
            {
                auto macTileTag = graph.mapper.get<MacroTile>(loadTag);
                auto macTile    = graph.coordinates.getNode<MacroTile>(macTileTag);
                if(macTile.memoryType == MemoryType::WAVE_SWIZZLE)
                    result.swizzleLoads.push_back(loadTag);
                else
                    result.nonSwizzleLoads.push_back(loadTag);
            }

            return result;
        }

        /**
         * Inserts prefetched loads before the loop to enable data pre-loading.
         * Chains loads with sequence edges and positions them at the K iteration boundary.
         *
         * @param graph Kernel graph to modify
         * @param colouring Maps operations to unroll values
         * @param preLoopLoads Load operations to insert before the loop
         * @param loads All loads in scope, used to find the insertion boundary
         */
        void InsertPreLoopLoads(KernelGraph&            graph,
                                UnrollColouring const&  colouring,
                                std::vector<int> const& preLoopLoads,
                                std::vector<int>        loads)
        {
            auto preNOP = graph.control.addElement(NOP());
            auto prev   = preNOP;
            for(auto const next : preLoopLoads)
            {
                graph.control.addElement(Sequence(), {prev}, {next});
                prev = next;
            }

            auto boundary  = FindLoadBoundary(graph, colouring, loads);
            auto prevTopOp = getTopSetCoordinate(graph, boundary.prevLoad);
            auto nextTopOp = getTopSetCoordinate(graph, boundary.nextLoad);
            graph.control.addElement(Sequence(), {prevTopOp}, {preNOP});
            graph.control.addElement(Sequence(), {prev}, {nextTopOp});
        }

        /**
         * Inserts duplicated load chains inside the loop with incremented K coordinates.
         *
         * @param graph Kernel graph to modify
         * @param colouring Maps operations to unroll values
         * @param inLoopLoads (load chain, unroll dimension) pairs to insert
         * @param forLoop For-loop where loads will be inserted
         */
        void InsertInLoopLoads(KernelGraph&                            graph,
                               UnrollColouring const&                  colouring,
                               std::vector<std::pair<int, int>> const& inLoopLoads,
                               int                                     forLoop)
        {
            auto loopLoads = filter(graph.control.isElemType<LoadTiled>(),
                                    graph.control.depthFirstVisit(forLoop))
                                 .to<std::vector>();
            AssertFatal(!loopLoads.empty());

            std::sort(loopLoads.begin(), loopLoads.end(), TopologicalCompare(graph));

            auto boundary = FindLoadBoundary(graph, colouring, loopLoads);

            auto postNOP = graph.control.addElement(NOP());
            auto prev    = postNOP;
            for(auto const [loadChain, _ignore] : inLoopLoads)
            {
                graph.control.addElement(Sequence(), {prev}, {loadChain});
                prev = loadChain;
            }
            auto prevTopOp = getTopSetCoordinate(graph, boundary.prevLoad);
            auto nextTopOp = getTopSetCoordinate(graph, boundary.nextLoad);
            graph.control.addElement(Sequence(), {prevTopOp}, {postNOP});
            graph.control.addElement(Sequence(), {prev}, {nextTopOp});

            for(auto const [loadChain, unrollCoord] : inLoopLoads)
            {
                std::optional<int> maybeOperation = loadChain;
                while(maybeOperation)
                {
                    auto operationTag = maybeOperation.value();

                    if(isOperation<LoadTiled>(graph.control.getElement(operationTag)))
                        break;

                    auto maybeSetCoordinate = graph.control.get<SetCoordinate>(operationTag);
                    AssertFatal(maybeSetCoordinate);
                    auto unroll = graph.mapper.get<Unroll>(operationTag);
                    AssertFatal(unroll > 0,
                                "SetCoordinate is not connected to the Unroll dimension",
                                ShowValue(operationTag),
                                ShowValue(unroll));

                    if(unroll == unrollCoord)
                    {
                        int  unrollKSize = getUnrollSize(graph, unroll);
                        auto valueExpr   = maybeSetCoordinate.value().value;
                        AssertFatal(
                            evaluationTimes(valueExpr)[Expression::EvaluationTime::Translate],
                            "SetCoordinate::value should be a literal.");
                        auto value = getUnsignedInt(evaluate(valueExpr));
                        auto newOp = SetCoordinate(Expression::literal(value + unrollKSize));
                        graph.control.setElement(operationTag, newOp);
                        break;
                    }
                    maybeOperation = only(graph.control.getOutputNodeIndices<Body>(operationTag));
                }
            }
        }

        /**
         * Inserts copy operations before their associated exchanges inside the loop.
         * Sorts exchanges topologically and sequences copies before them to ensure data availability.
         *
         * @param graph Kernel graph to modify
         * @param copies Map from copy tags to their associated exchange tags
         */
        void InsertInLoopCopies(KernelGraph& graph, auto const& copies)
        {
            for(auto const copy : copies)
            {
                auto exchangeTags = copy.second;
                std::sort(exchangeTags.begin(), exchangeTags.end(), TopologicalCompare(graph));
                insertBefore(graph, exchangeTags[0], copy.first, copy.first);
                for(auto const exchangeTag : exchangeTags)
                    graph.control.addElement(Sequence(), {copy.first}, {exchangeTag});
            }
        }

        /**
         * Metadata for creating copy operations: data type and register requirements.
         * Cached and reused across multiple loads to avoid redundant computation.
         */
        struct CopyInfo
        {
            DataType dataType; /**< Type of values being copied */
            size_t   numVGPRs; /**< VGPRs required to hold the copy */
        };

        /**
         * Computes data type and VGPR count needed to copy a load's data.
         * Derives requirements from wave tile size, variable type, and packing factor.
         *
         * @param graph Kernel graph containing the load
         * @param context
         * @param loadTag Load operation to analyze
         * @return CopyInfo with data type and VGPR count
         */
        CopyInfo GetCopyInfo(KernelGraph const& graph, ContextPtr context, int loadTag)
        {
            auto waveTileTag = graph.mapper.get<WaveTile>(loadTag);
            auto waveTile    = graph.coordinates.get<WaveTile>(waveTileTag);
            auto elements    = waveTile.value().elements();

            auto varType = getVariableType(graph, loadTag);
            // TODO: This assumes that eventually (after
            // LoadPacked) the incoming data will be packed
            auto maybePacked = DataTypeInfo::Get(varType).packedVariableType();
            if(maybePacked)
                varType = *maybePacked;
            auto packFactor = DataTypeInfo::Get(varType).packing;

            uint wfs = context->kernel()->wavefront_size();

            return CopyInfo{varType.dataType, elements / wfs / packFactor};
        }

        /**
         * Creates an Assign operation that copies loaded data into new VGPRs.
         * Creates a DataFlow edge from source to destination macro tile.
         *
         * @param graph Kernel graph to modify
         * @param context
         * @param loadTag Load operation (used to compute copyInfo if not cached)
         * @param macTileTag Source macro tile with original data
         * @param copyInfo Optional cached copy information; computed if not provided
         * @return (copy operation tag, destination macro tile tag)
         */
        std::pair<int, int> CreateCopy(KernelGraph&             graph,
                                       ContextPtr               context,
                                       int                      loadTag,
                                       int                      macTileTag,
                                       std::optional<CopyInfo>& copyInfo)
        {
            if(!copyInfo)
                copyInfo = GetCopyInfo(graph, context, loadTag);

            auto copyExpr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{macTileTag, Register::Type::Vector, copyInfo->dataType});
            auto copyTag = graph.control.addElement(
                Assign{Register::Type::Vector, copyExpr, copyInfo->numVGPRs});
            auto destMacTileTag = graph.coordinates.addElement(MacroTile());
            // macTile is being copied into destMacTile through this assign node
            graph.coordinates.addElement(DataFlow(), {macTileTag}, {destMacTileTag});
            graph.mapper.connect(copyTag, destMacTileTag, NaryArgument::DEST);

            return std::make_pair(copyTag, destMacTileTag);
        }

        /**
         * Redirects exchanges to use the copied destination tile instead of the original source.
         * Rewires Index edges from source to destination and records exchanges for proper sequencing.
         *
         * @param graph Kernel graph to modify
         * @param loadTag Unused, kept for consistency
         * @param macTileTag Source macro tile originally used by exchanges
         * @param destMacTileTag Destination macro tile for exchanges to use
         * @param copyTag Copy operation used as key in inLoopCopies
         * @param inLoopCopies Output map of exchange tags associated with each copy
         */
        void UpdateExchangeMacroTiles(KernelGraph&                     graph,
                                      int                              loadTag,
                                      int                              macTileTag,
                                      int                              destMacTileTag,
                                      int                              copyTag,
                                      std::map<int, std::vector<int>>& inLoopCopies)
        {
            auto location = graph.coordinates.getLocation(macTileTag);
            for(auto const& input : location.incoming)
            {
                auto edge       = graph.coordinates.getElement(input);
                auto maybeIndex = graph.coordinates.get<Index>(input);
                if(!maybeIndex)
                    continue;
                auto exchangeTileTag
                    = only(graph.coordinates.getNeighbours<Graph::Direction::Upstream>(input));
                AssertFatal(exchangeTileTag,
                            "Expected a single upstream Exchange neighbour for: ",
                            ShowValue(input));
                graph.coordinates.deleteElement(input);
                graph.coordinates.addElement(edge, {*exchangeTileTag}, {destMacTileTag});

                std::optional<int> exchangeTag;
                for(auto const c : graph.mapper.getCoordinateConnections(*exchangeTileTag))
                {
                    auto maybeExchange = graph.control.get<Exchange>(c.control);
                    if(maybeExchange)
                    {
                        exchangeTag = c.control;
                        break;
                    }
                }
                AssertFatal(exchangeTag,
                            "Expected an Exchange connection for: ",
                            ShowValue(*exchangeTileTag));
                auto exchangeTopOp = getTopSetCoordinate(graph, *exchangeTag);
                inLoopCopies[copyTag].push_back(exchangeTopOp);
            }
        }

        /**
         * Implements prefetch-scale for WAVE_SWIZZLE loads using pre-loop copies.
         * Moves first iteration outside loop, creates copies, and duplicates load chains inside loop.
         * Used when macKUnrollSize * unrollKSize is divisible by factorK.
         *
         * @param graph Kernel graph to transform
         * @param context
         * @param loads All loads in scope
         * @param loopLoads Loads inside loop to transform
         * @param colouring Maps operations to unroll values
         */
        void PrefetchScaleLoadsImpl(KernelGraph&            graph,
                                    ContextPtr              context,
                                    std::vector<int> const& loads,
                                    std::vector<int> const& loopLoads,
                                    UnrollColouring const&  colouring)
        {
            std::vector<int>                 preLoopLoads;
            std::map<int, std::vector<int>>  inLoopCopies;
            std::vector<std::pair<int, int>> inLoopLoads;

            auto                    forLoop = -1;
            std::optional<CopyInfo> copyInfo;

            auto  partition    = GroupLoadsBySwizzle(graph, loopLoads);
            auto& swizzleLoads = partition.swizzleLoads;

            for(auto const loadTag : swizzleLoads)
            {
                auto macTileTag = graph.mapper.get<MacroTile>(loadTag);

                // the swizzle scale loads must be inside the loop K
                auto maybeForLoop = findContainingOperation<ForLoopOp>(loadTag, graph);
                AssertFatal(maybeForLoop);
                AssertFatal(forLoop == -1 || forLoop == *maybeForLoop,
                            ShowValue(forLoop),
                            ShowValue(*maybeForLoop));
                forLoop = *maybeForLoop;

                auto [copyTag, destMacTileTag]
                    = CreateCopy(graph, context, loadTag, macTileTag, copyInfo);

                auto topOp = getTopSetCoordinate(graph, loadTag);
                replaceWith(graph, topOp, graph.control.addElement(NOP()), false);

                preLoopLoads.push_back(topOp);

                auto inLoopLoad = duplicateChain(graph, {topOp});
                graph.control.addElement(Sequence(), {copyTag}, {inLoopLoad});
                auto unrollDim = graph.mapper.get<Unroll>(loadTag, rocRoller::KLOOP_UNROLL);
                inLoopLoads.push_back(std::make_pair(inLoopLoad, unrollDim));

                UpdateExchangeMacroTiles(
                    graph, loadTag, macTileTag, destMacTileTag, copyTag, inLoopCopies);
            }

            if(preLoopLoads.empty())
                return;

            AssertFatal(!loads.empty(),
                        "WAVE_SWIZZLE loads require non-empty loads for prefetch positioning");
            InsertPreLoopLoads(graph, colouring, preLoopLoads, loads);
            AssertFatal(forLoop != -1);
            InsertInLoopLoads(graph, colouring, inLoopLoads, forLoop);
            InsertInLoopCopies(graph, inLoopCopies);
        }

        /**
         * Finds all exchange operations connected to a load's macro tile.
         *
         * @param loadTag Load operation whose exchanges to find
         * @param graph Kernel graph containing the load and exchanges
         * @return Exchange operation tags associated with the load
         */
        std::vector<int> GetExchangesForLoad(int loadTag, KernelGraph const& graph)
        {
            auto isExchangePredicate = [&](int operation) -> bool {
                auto maybeExchange = graph.control.get<Exchange>(operation);
                return maybeExchange.has_value();
            };

            auto findConnections = [&](int coordinate) -> std::vector<int> {
                std::vector<int> exchanges;
                for(auto c : graph.mapper.getCoordinateConnections(coordinate))
                    if(isExchangePredicate(c.control))
                        exchanges.push_back(c.control);
                return exchanges;
            };

            auto tileTag = graph.mapper.get<MacroTile>(loadTag);

            if(auto exchanges = findConnections(tileTag); !exchanges.empty())
                return exchanges;

            std::vector<int> exchanges;
            for(auto edge : graph.coordinates.getNeighbours<GD::Upstream>(tileTag))
            {
                auto maybeIndex = graph.coordinates.get<Index>(edge);
                if(!maybeIndex)
                    continue;

                auto indexTileTag = *only(graph.coordinates.getNeighbours<GD::Upstream>(edge));

                if(auto temp = findConnections(indexTileTag); !temp.empty())
                    exchanges.insert(exchanges.end(), temp.begin(), temp.end());
            }

            return exchanges;
        }

        /**
         * Insertion positions for prefetch and exchange operations.
         * Indexed by sub-iteration, relative to top SetCoordinate of non-swizzle loads.
         */
        struct PrefetchPositions
        {
            std::vector<int> prefetchPosition; /**< Where to insert prefetch loads */
            std::vector<int> exchangePosition; /**< Where to insert exchanges and copies */
        };

        /**
         * Computes insertion positions for prefetch and exchange operations.
         * Analyzes non-swizzle loads to find unroll K value boundaries and tracks loop entry.
         *
         * @param graph Kernel graph containing the loads
         * @param colouring Maps operations to unroll values
         * @param nonSwizzleLoads Non-WAVE_SWIZZLE loads used as position anchors
         * @param kLoopLoadSet Loads that are inside the K loop (used to distinguish
         *                     pre-loop loads from K loop body loads)
         * @return PrefetchPositions indexed by sub-iteration
         */
        PrefetchPositions DeterminePrefetchPositions(KernelGraph const&             graph,
                                                     UnrollColouring const&         colouring,
                                                     std::vector<int> const&        nonSwizzleLoads,
                                                     std::unordered_set<int> const& kLoopLoadSet)
        {
            std::vector<int>   prefetchPosition;
            std::vector<int>   exchangePosition;
            std::optional<int> prevPosition;
            std::optional<int> unrollKVal;
            bool               isInsideLoop = false;

            for(auto const loadTag : nonSwizzleLoads)
            {
                auto unrollMap  = colouring.operationColour.at(loadTag);
                auto unrollKDim = graph.mapper.get<Unroll>(loadTag, rocRoller::KLOOP_UNROLL);
                auto subiter    = unrollMap.at(unrollKDim);

                // Detect if we've entered the K loop (not just any ForLoopOp)
                if(!isInsideLoop)
                {
                    if(kLoopLoadSet.contains(loadTag))
                        isInsideLoop = true;
                }

                auto topOp = getTopSetCoordinate(graph, loadTag);
                if(!unrollKVal || *unrollKVal != subiter)
                {
                    unrollKVal = subiter;
                    prefetchPosition.push_back(topOp);

                    // Only add to exchangePosition once we're inside the loop
                    if(isInsideLoop && prevPosition)
                        exchangePosition.push_back(*prevPosition);

                    prevPosition = topOp;
                }
            }

            if(prevPosition)
                exchangePosition.push_back(*prevPosition);

            AssertFatal(!nonSwizzleLoads.empty(),
                        "WAVE_SWIZZLE loads require non-swizzle loads for prefetch positioning");
            AssertFatal(!prefetchPosition.empty(),
                        "Failed to determine prefetch positions from non-swizzle loads");

            return PrefetchPositions{std::move(prefetchPosition), std::move(exchangePosition)};
        }

        /**
         * Inserts prefetched loads at computed positions to enable early execution.
         * Chains prefetch nodes with sequence edges and inserts before computed positions.
         *
         * @param graph Kernel graph to modify
         * @param nextIterPrefetch Map from sub-iteration to prefetch operation tags
         * @param prefetchPosition Control flow positions indexed by sub-iteration
         */
        void InsertPrefetchNodes(KernelGraph&                           graph,
                                 std::map<int, std::vector<int>> const& nextIterPrefetch,
                                 std::vector<int> const&                prefetchPosition)
        {
            for(auto const [subiter, prefetchNodes] : nextIterPrefetch)
            {
                auto preNOP = graph.control.addElement(NOP());
                auto prev   = preNOP;
                for(auto const next : prefetchNodes)
                {
                    graph.control.addElement(Sequence(), {prev}, {next});
                    prev = next;
                }

                auto pos = prefetchPosition[subiter];
                Log::debug("    Inserting prefetch for subiter={} at position {}", subiter, pos);
                insertBefore(graph, pos, preNOP, prev);
            }
        }

        /**
         * Inserts exchanges and copies at computed positions.
         *
         * @param graph Kernel graph to modify
         * @param exchangePrefetch Map from sub-iteration to exchange operation tags
         * @param copy Map from sub-iteration to copy operation tags (may be empty)
         * @param exchangePosition Control flow positions indexed by sub-iteration
         */
        void InsertExchangeNodes(KernelGraph&                           graph,
                                 std::map<int, std::vector<int>> const& exchangePrefetch,
                                 std::map<int, std::vector<int>> const& copy,
                                 std::vector<int> const&                exchangePosition)
        {
            for(auto const [subiter, exchangeNodes] : exchangePrefetch)
            {
                auto preNOP = graph.control.addElement(NOP());
                auto prev   = preNOP;
                if(!copy.empty())
                {
                    for(auto const c : copy.at(subiter))
                    {
                        graph.control.addElement(Sequence(), {prev}, {c});
                        prev = c;
                    }
                }
                for(auto const next : exchangeNodes)
                {
                    graph.control.addElement(Sequence(), {prev}, {next});
                    prev = next;
                }

                auto pos = exchangePosition[subiter];
                Log::debug("    Inserting exchange for subiter={} at position {}", subiter, pos);
                insertBefore(graph, pos, preNOP, prev);
            }
        }

        /**
         * Implements prefetch-scale for WAVE_SWIZZLE loads using per-unroll prefetching.
         * Replaces loads with NOPs and creates prefetched versions at computed insertion points.
         *
         * @param graph Kernel graph to transform
         * @param context
         * @param loads All loads in scope
         * @param loopLoads Loads inside loop
         * @param colouring Maps operations to unroll values
         */
        void PrefetchScaleLoadsPerUnrollImpl(KernelGraph&            graph,
                                             ContextPtr              context,
                                             std::vector<int> const& loads,
                                             std::vector<int> const& loopLoads,
                                             UnrollColouring const&  colouring)
        {
            std::optional<int>              numInFlight;
            std::map<int, std::vector<int>> nextIterPrefetch;
            std::map<int, std::vector<int>> exchangePrefetch;
            std::map<int, std::vector<int>> copy;
            std::optional<CopyInfo>         copyInfo;

            auto loopPartition = GroupLoadsBySwizzle(graph, loopLoads);
            auto allPartition  = GroupLoadsBySwizzle(graph, loads);

            auto& swizzleLoads    = loopPartition.swizzleLoads;
            auto& nonSwizzleLoads = allPartition.nonSwizzleLoads;

            if(swizzleLoads.empty())
                return;

            std::unordered_set<int> kLoopLoadSet(loopLoads.begin(), loopLoads.end());

            auto [prefetchPosition, exchangePosition]
                = DeterminePrefetchPositions(graph, colouring, nonSwizzleLoads, kLoopLoadSet);

            for(auto const loadTag : swizzleLoads)
            {
                auto macTileTag = graph.mapper.get<MacroTile>(loadTag);

                if(!numInFlight)
                    numInFlight = static_cast<int>(prefetchPosition.size());

                auto unrollMap  = colouring.operationColour.at(loadTag);
                auto unrollKDim = graph.mapper.get<Unroll>(loadTag, rocRoller::KLOOP_UNROLL);
                auto subiter    = unrollMap.at(unrollKDim);

                auto topOp = getTopSetCoordinate(graph, loadTag);
                replaceWith(graph, topOp, graph.control.addElement(NOP()), false);
                nextIterPrefetch[subiter].push_back(topOp);

                if(context->kernelOptions()->scaleSkipPermlane
                   != rocRoller::ScaleSkipPermlaneMode::None)
                {
                    auto [copyTag, destMacTileTag]
                        = CreateCopy(graph, context, loadTag, macTileTag, copyInfo);

                    auto unrollKSize = getUnrollSize(graph, unrollKDim);
                    auto location    = graph.coordinates.getLocation(macTileTag);
                    for(auto const& input : location.incoming)
                    {
                        auto edge       = graph.coordinates.getElement(input);
                        auto maybeIndex = graph.coordinates.get<Index>(input);
                        if(!maybeIndex)
                            continue;
                        auto exchangeTileTag = only(
                            graph.coordinates.getNeighbours<Graph::Direction::Upstream>(input));
                        AssertFatal(exchangeTileTag);
                        graph.coordinates.deleteElement(input);
                        graph.coordinates.addElement(edge, {*exchangeTileTag}, {destMacTileTag});

                        for(auto const c : graph.mapper.getCoordinateConnections(*exchangeTileTag))
                        {
                            auto maybeExchange = graph.control.get<Exchange>(c.control);
                            if(maybeExchange)
                            {
                                auto exchange = c.control;
                                replaceWith(
                                    graph, exchange, graph.control.addElement(NOP()), false);
                                exchangePrefetch[subiter].push_back(exchange);
                                copy[subiter].push_back(copyTag);
                                if(subiter == 0)
                                {
                                    auto exchangeDup = duplicateControlNode(graph, exchange);
                                    exchangePrefetch[unrollKSize].push_back(exchangeDup);
                                    auto copyDup = duplicateControlNode(graph, copyTag);
                                    copy[unrollKSize].push_back(copyDup);
                                }
                                break;
                            }
                        }
                    }
                }
                else
                {
                    auto exchanges   = GetExchangesForLoad(loadTag, graph);
                    auto unrollKSize = getUnrollSize(graph, unrollKDim);
                    for(auto const exchange : exchanges)
                    {
                        replaceWith(graph, exchange, graph.control.addElement(NOP()), false);
                        exchangePrefetch[subiter].push_back(exchange);

                        if(subiter == 0)
                        {
                            auto exchangeDup = duplicateControlNode(graph, exchange);
                            exchangePrefetch[unrollKSize].push_back(exchangeDup);
                        }
                    }
                }

                if(subiter < *numInFlight)
                {
                    auto               prefetchTopOp  = duplicateChain(graph, {topOp});
                    std::optional<int> maybeOperation = prefetchTopOp;
                    while(maybeOperation)
                    {
                        auto operationTag = *maybeOperation;

                        if(isOperation<LoadTiled>(graph.control.getElement(operationTag)))
                            break;

                        auto maybeSetCoordinate = graph.control.get<SetCoordinate>(operationTag);
                        AssertFatal(maybeSetCoordinate);
                        auto unroll = graph.mapper.get<Unroll>(operationTag);
                        AssertFatal(unroll > 0,
                                    "SetCoordinate is not connected to the Unroll dimension",
                                    ShowValue(operationTag),
                                    ShowValue(unroll));

                        if(unroll == unrollKDim)
                        {
                            int  unrollKSize = getUnrollSize(graph, unroll);
                            auto valueExpr   = maybeSetCoordinate.value().value;
                            AssertFatal(
                                evaluationTimes(valueExpr)[Expression::EvaluationTime::Translate],
                                "SetCoordinate::value should be a literal.");
                            auto value = getUnsignedInt(evaluate(valueExpr));
                            auto newOp = SetCoordinate(Expression::literal(value + unrollKSize));
                            graph.control.setElement(operationTag, newOp);
                            nextIterPrefetch[value + unrollKSize].push_back(prefetchTopOp);
                            break;
                        }
                        maybeOperation
                            = only(graph.control.getOutputNodeIndices<Body>(operationTag));
                    }
                }
            }

            InsertPrefetchNodes(graph, nextIterPrefetch, prefetchPosition);
            InsertExchangeNodes(graph, exchangePrefetch, copy, exchangePosition);
        }

        /**
         * Orchestrates prefetch-scale transformation for WAVE_SWIZZLE loads in the K loop.
         * Collects loads from enclosing scope and selects the appropriate prefetch strategy.
         *
         * @param graph Kernel graph to transform
         * @param context
         * @param forLoop K loop operation to transform
         */
        void PrefetchScaleLoads(KernelGraph& graph, ContextPtr context, int forLoop)
        {
            auto               stack = controlStack(forLoop, graph);
            std::optional<int> scope;
            for(auto iter = stack.rbegin(); iter != stack.rend(); ++iter)
            {
                if(graph.control.get<Scope>(*iter))
                {
                    scope = *iter;
                    break;
                }
            }
            AssertFatal(
                scope.has_value(), "Could not find enclosing Scope for forLoop {}", forLoop);

            auto bodyChildren = graph.control.getOutputNodeIndices<Body>(*scope).to<std::vector>();
            std::unordered_set<int> loadSet;

            auto dontWalkPastForLoop = [&](int edgeTag) -> bool {
                auto dsts = graph.control.getNeighbours<Graph::Direction::Downstream>(edgeTag);
                for(auto dst : dsts)
                {
                    if(graph.control.get<ForLoopOp>(dst))
                        return false;
                }
                return true;
            };

            for(auto node :
                filter(graph.control.isElemType<LoadTiled>(),
                       graph.control.depthFirstVisit(
                           bodyChildren, dontWalkPastForLoop, Graph::Direction::Downstream))
                    .to<std::vector>())
            {
                loadSet.insert(node);
            }

            // Only include loads that are inside the loop
            std::vector<int> loopLoads;
            for(auto bodyNode : graph.control.getOutputNodeIndices<Body>(forLoop))
            {
                auto bodyLoads = filter(graph.control.isElemType<LoadTiled>(),
                                        graph.control.depthFirstVisit(bodyNode))
                                     .to<std::vector>();
                for(auto load : bodyLoads)
                {
                    auto [it, inserted] = loadSet.insert(load);
                    if(inserted)
                        loopLoads.push_back(load);
                }
            }

            std::vector<int> loads(loadSet.begin(), loadSet.end());

            std::sort(loads.begin(), loads.end(), TopologicalCompare(graph));

            auto swizzleLoadIt = std::find_if(loopLoads.begin(), loopLoads.end(), [&](int loadTag) {
                auto macTileTag = graph.mapper.get<MacroTile>(loadTag);
                auto macTile    = graph.coordinates.getNode<MacroTile>(macTileTag);
                return macTile.memoryType == MemoryType::WAVE_SWIZZLE;
            });

            if(swizzleLoadIt == loopLoads.end())
                return; // No WAVE_SWIZZLE loads found in this loop

            auto loadTag    = *swizzleLoadIt;
            auto macTileTag = graph.mapper.get<MacroTile>(loadTag);
            auto macTile    = graph.coordinates.getNode<MacroTile>(macTileTag);

            AssertFatal(macTile.layoutType == LayoutType::MATRIX_A
                            || macTile.layoutType == LayoutType::MATRIX_B,
                        ShowValue(macTile.layoutType));

            auto unroll0 = graph.mapper.get<Unroll>(loadTag, rocRoller::XLOOP_UNROLL);
            auto unroll1 = graph.mapper.get<Unroll>(loadTag, rocRoller::YLOOP_UNROLL);
            auto unroll2 = graph.mapper.get<Unroll>(loadTag, rocRoller::KLOOP_UNROLL);

            unsigned int xyUnrollSize = 0, macKUnrollSize = 0;
            if(macTile.layoutType == LayoutType::MATRIX_A)
            {
                // A : M x K
                xyUnrollSize   = getUnrollSize(graph, unroll0);
                macKUnrollSize = getUnrollSize(graph, unroll1);
            }
            else
            {
                // B : K x N
                xyUnrollSize   = getUnrollSize(graph, unroll1);
                macKUnrollSize = getUnrollSize(graph, unroll0);
            }

            // if unroll2 is -1, this returns 1.
            auto unrollKSize = getUnrollSize(graph, unroll2);

            AssertFatal(xyUnrollSize != 0 && macKUnrollSize != 0 && unrollKSize != 0,
                        ShowValue(xyUnrollSize),
                        ShowValue(macKUnrollSize),
                        ShowValue(unrollKSize));

            auto waveM = macTile.subTileSizes.at(0);
            auto waveN = macTile.subTileSizes.at(1);
            auto waveK = macTile.subTileSizes.at(2);
            AssertFatal(
                waveM == waveN, "waveM is not equal to waveN", ShowValue(waveM), ShowValue(waveN));
            auto miM = macTile.miTileSizes.at(0);
            auto miN = macTile.miTileSizes.at(1);
            auto miK = macTile.miTileSizes.at(2);
            AssertFatal(miM == miN, "miM is not equal to miN", ShowValue(miM), ShowValue(miN));

            auto factorMN  = waveM / miM;
            auto factorK   = waveK / miK;
            auto colouring = colourByUnrollValue(graph);

            if(xyUnrollSize % factorMN == 0 && macKUnrollSize % factorK == 0)
            {
                Log::debug("  Using PrefetchScaleLoadsPerUnrollImpl (xy={}, macK={}, factors: "
                           "MN={}, K={})",
                           xyUnrollSize,
                           macKUnrollSize,
                           factorMN,
                           factorK);
                PrefetchScaleLoadsPerUnrollImpl(graph, context, loads, loopLoads, colouring);
            }
            else if(xyUnrollSize % factorMN == 0 && (macKUnrollSize * unrollKSize) % factorK == 0)
            {
                Log::debug("  Using PrefetchScaleLoadsImpl (xy={}, macK={}, unrollK={}, factors: "
                           "MN={}, K={})",
                           xyUnrollSize,
                           macKUnrollSize,
                           unrollKSize,
                           factorMN,
                           factorK);
                PrefetchScaleLoadsImpl(graph, context, loads, loopLoads, colouring);
            }
            else
            {
                Throw<FatalError>("Prefetch Scale not supported for the given swizzled tile.");
            }
        }

        KernelGraph PrefetchScale::apply(KernelGraph const& original)
        {
            auto newGraph = original;
            auto root     = *newGraph.control.roots().only();

            std::optional<int> maybeKLoop;
            for(auto const loop : filter(newGraph.control.isElemType<ForLoopOp>(),
                                         newGraph.control.depthFirstVisit(root)))
            {
                auto forloop = *newGraph.control.get<ForLoopOp>(loop);
                if(forloop.loopName == rocRoller::KLOOP)
                {
                    maybeKLoop = loop;
                    break;
                }
            }

            AssertFatal(maybeKLoop, "PrefetchScale requires a KLoop");
            auto kLoop = *maybeKLoop;
            PrefetchScaleLoads(newGraph, m_context, kLoop);

            return newGraph;
        }
    }
}
