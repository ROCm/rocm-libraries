/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/PrefetchScale.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;
        using GD     = rocRoller::Graph::Direction;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        void insertPreLoopLoads(KernelGraph&            graph,
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

            std::optional<int> unrollKVal;
            std::optional<int> prevLoad;
            std::optional<int> nextLoad;
            for(auto const load : loads)
            {
                auto unrollMap  = colouring.operationColour.at(load);
                auto unrollKDim = graph.mapper.get<Unroll>(load, 2);
                if(unrollKVal.has_value() && unrollKVal.value() != unrollMap.at(unrollKDim))
                {
                    nextLoad = load;
                    break;
                }
                unrollKVal = unrollMap.at(unrollKDim);
                prevLoad   = load;
            }

            AssertFatal(prevLoad.has_value() && nextLoad.has_value(),
                        "couldn't find a position to in-loop scale prefetch");

            auto prevTopOp = getTopSetCoordinate(graph, prevLoad.value());
            auto nextTopOp = getTopSetCoordinate(graph, nextLoad.value());
            graph.control.addElement(Sequence(), {prevTopOp}, {preNOP});
            graph.control.addElement(Sequence(), {prev}, {nextTopOp});
        }

        void insertInLoopLoads(KernelGraph&                            graph,
                               UnrollColouring const&                  colouring,
                               std::vector<std::pair<int, int>> const& inLoopLoads,
                               int                                     forLoop)
        {
            auto loopLoads = filter(graph.control.isElemType<LoadTiled>(),
                                    graph.control.depthFirstVisit(forLoop))
                                 .to<std::vector>();
            AssertFatal(!loopLoads.empty());

            std::sort(loopLoads.begin(),
                      loopLoads.end(),
                      TopologicalCompare(std::make_shared<KernelGraph>(graph)));

            std::optional<int> unrollKVal;
            std::optional<int> prevLoad;
            std::optional<int> nextLoad;
            for(auto const loopLoad : loopLoads)
            {
                auto unrollMap  = colouring.operationColour.at(loopLoad);
                auto unrollKDim = graph.mapper.get<Unroll>(loopLoad, 2);
                if(unrollKVal.has_value() && unrollKVal.value() != unrollMap.at(unrollKDim))
                {
                    nextLoad = loopLoad;
                    break;
                }
                unrollKVal = unrollMap.at(unrollKDim);
                prevLoad   = loopLoad;
            }

            AssertFatal(prevLoad.has_value() && nextLoad.has_value(),
                        "couldn't find a position to in-loop scale prefetch");

            auto postNOP = graph.control.addElement(NOP());
            auto prev    = postNOP;
            for(auto const [loadChain, _ignore] : inLoopLoads)
            {
                graph.control.addElement(Sequence(), {prev}, {loadChain});
                prev = loadChain;
            }
            auto prevTopOp = getTopSetCoordinate(graph, prevLoad.value());
            auto nextTopOp = getTopSetCoordinate(graph, nextLoad.value());
            graph.control.addElement(Sequence(), {prevTopOp}, {postNOP});
            graph.control.addElement(Sequence(), {prev}, {nextTopOp});

            // Update SetCoordinates
            for(auto const [loadChain, unrollCoord] : inLoopLoads)
            {
                std::optional<int> maybeOperation = loadChain;
                while(maybeOperation)
                {
                    auto operationTag = maybeOperation.value();

                    if(isOperation<LoadTiled>(graph.control.getElement(operationTag)))
                        break;

                    auto maybeSetCoordinate = graph.control.get<SetCoordinate>(operationTag);
                    AssertFatal(maybeSetCoordinate.has_value());
                    auto unroll = graph.mapper.get<Unroll>(operationTag);
                    AssertFatal(unroll > 0,
                                "SetCoordinate is not connected to the Unroll dimension");

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

        void insertInLoopCopies(KernelGraph& graph, auto const& copies)
        {
            for(auto const copy : copies)
            {
                auto exchangeTags = copy.second;
                std::sort(exchangeTags.begin(),
                          exchangeTags.end(),
                          TopologicalCompare(std::make_shared<KernelGraph>(graph)));
                insertBefore(graph, exchangeTags[0], copy.first, copy.first);
                for(auto const exchangeTag : exchangeTags)
                    graph.control.addElement(Sequence(), {copy.first}, {exchangeTag});
            }
        }

        void prefetchScaleLoads(KernelGraph& graph, ContextPtr context)
        {
            auto root = graph.control.roots().only().value();

            std::vector<int>                 preLoopLoads;
            std::map<int, std::vector<int>>  inLoopCopies;
            std::vector<std::pair<int, int>> inLoopLoads;

            auto forLoop = -1;

            // all loads that follow the root
            auto loads
                = filter(graph.control.isElemType<LoadTiled>(), graph.control.depthFirstVisit(root))
                      .to<std::vector>();

            auto colouring = colourByUnrollValue(graph);

            // sort the loads
            // this brings the first load in the sequence to the front
            std::sort(loads.begin(),
                      loads.end(),
                      TopologicalCompare(std::make_shared<KernelGraph>(graph)));

            for(auto const loadTag : loads)
            {
                auto macTileTag = graph.mapper.get<MacroTile>(loadTag);
                auto macTile    = graph.coordinates.getNode<MacroTile>(macTileTag);
                // identify swizzle scale loads
                if(macTile.memoryType == MemoryType::WAVE_SWIZZLE)
                {
                    // the swizzle scale loads must be inside the loop K
                    auto maybeForLoop = findContainingOperation<ForLoopOp>(loadTag, graph);
                    AssertFatal(maybeForLoop.has_value());
                    AssertFatal(forLoop == -1 || forLoop == maybeForLoop.value());
                    forLoop = maybeForLoop.value();

                    // Copy loaded data into a new set of VGPRs
                    DataType dataType;
                    size_t   numVGPRs;
                    {
                        auto waveTileTag = graph.mapper.get<WaveTile>(loadTag);
                        auto waveTile    = graph.coordinates.get<WaveTile>(waveTileTag);
                        auto elements    = waveTile.value().elements();

                        auto varType = getVariableType(graph, loadTag);
                        // TODO: This assumes that eventually (after
                        // LoadPacked) the incoming data will be
                        // packed
                        auto maybePacked = DataTypeInfo::Get(varType).packedVariableType();
                        if(maybePacked)
                            varType = *maybePacked;
                        auto packFactor = DataTypeInfo::Get(varType).packing;

                        uint wfs = context->kernel()->wavefront_size();

                        dataType = varType.dataType;
                        numVGPRs = elements / wfs / packFactor;
                    }

                    auto copyExpr = std::make_shared<Expression::Expression>(
                        Expression::DataFlowTag{macTileTag, Register::Type::Vector, dataType});
                    auto copyTag = graph.control.addElement(
                        Assign{Register::Type::Vector, copyExpr, numVGPRs});
                    auto destMacTileTag = graph.coordinates.addElement(MacroTile());
                    // macTile is being copied into destMacTile through this assign node
                    graph.coordinates.addElement(DataFlow(), {macTileTag}, {destMacTileTag});
                    graph.mapper.connect(copyTag, destMacTileTag, NaryArgument::DEST);

                    auto topOp = getTopSetCoordinate(graph, loadTag);
                    replaceWith(graph, topOp, graph.control.addElement(NOP()), false);

                    preLoopLoads.push_back(topOp);
                    auto inLoopLoad = duplicateChain(graph, {topOp});
                    graph.control.addElement(Sequence(), {copyTag}, {inLoopLoad});
                    inLoopLoads.push_back(
                        std::make_pair(inLoopLoad, graph.mapper.get<Unroll>(loadTag, 2)));

                    // update the indexes of the associated exchange macrotiles
                    auto location = graph.coordinates.getLocation(macTileTag);
                    for(auto const& input : location.incoming)
                    {
                        auto edge       = graph.coordinates.getElement(input);
                        auto maybeIndex = graph.coordinates.get<Index>(input);
                        if(!maybeIndex.has_value())
                            continue;
                        auto exchangeTileTag = only(
                            graph.coordinates.getNeighbours<Graph::Direction::Upstream>(input));
                        AssertFatal(exchangeTileTag.has_value());
                        graph.coordinates.deleteElement(input);
                        graph.coordinates.addElement(
                            edge, {exchangeTileTag.value()}, {destMacTileTag});

                        std::optional<int> exchangeTag;
                        for(auto const c :
                            graph.mapper.getCoordinateConnections(exchangeTileTag.value()))
                        {
                            auto maybeExchange = graph.control.get<Exchange>(c.control);
                            if(maybeExchange)
                            {
                                exchangeTag = c.control;
                                break;
                            }
                        }
                        AssertFatal(exchangeTag.has_value());
                        inLoopCopies[copyTag].push_back(
                            getTopSetCoordinate(graph, exchangeTag.value()));
                    }
                }
            }

            if(preLoopLoads.empty())
                return;

            AssertFatal(!loads.empty());
            insertPreLoopLoads(graph, colouring, preLoopLoads, loads);

            AssertFatal(forLoop != -1);
            insertInLoopLoads(graph, colouring, inLoopLoads, forLoop);

            insertInLoopCopies(graph, inLoopCopies);
        }

        KernelGraph PrefetchScale::apply(KernelGraph const& original)
        {
            auto newGraph = original;

            prefetchScaleLoads(newGraph, m_context);

            return newGraph;
        }
    }
}
