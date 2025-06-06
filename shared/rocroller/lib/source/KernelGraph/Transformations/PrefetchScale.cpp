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

        class topoComp
        {
        public:
            topoComp() = delete;
            topoComp(KernelGraphPtr graph)
                : m_graph(graph)
            {
                AssertFatal(graph, "no graph found!");
            };

            bool operator()(int a, int b)
            {
                return m_graph->control.compareNodes(rocRoller::UpdateCache, a, b)
                       == NodeOrdering::LeftFirst;
            }

        private:
            KernelGraphPtr m_graph;
        };

        void insertPreLoopLoads(KernelGraph&            graph,
                                std::vector<int> const& preLoopLoad,
                                int                     firstLoad)
        {
            auto preNOP = graph.control.addElement(NOP());
            auto prev   = preNOP;
            for(auto next : preLoopLoad)
            {
                graph.control.addElement(Sequence(), {prev}, {next});
                prev = next;
            }
            auto topOp = getTopSetCoordinate(graph, firstLoad);
            insertBefore(graph, topOp, preNOP, prev);
        }

        void insertInLoopLoads(KernelGraph&                            graph,
                               std::vector<std::pair<int, int>> const& inLoopLoad,
                               int                                     forLoop)
        {
            auto loopLoads = filter(graph.control.isElemType<LoadTiled>(),
                                    graph.control.depthFirstVisit(forLoop))
                                 .to<std::vector>();
            AssertFatal(!loopLoads.empty());

            std::sort(
                loopLoads.begin(), loopLoads.end(), topoComp(std::make_shared<KernelGraph>(graph)));

            auto postNOP = graph.control.addElement(NOP());
            auto prev    = postNOP;
            for(auto [loadChain, _ignore] : inLoopLoad)
            {
                graph.control.addElement(Sequence(), {prev}, {loadChain});
                prev = loadChain;
            }
            auto topOp = getTopSetCoordinate(graph, loopLoads[0]);
            insertBefore(graph, topOp, postNOP, prev);

            // Update SetCoordinates
            for(auto [loadChain, unrollCoord] : inLoopLoad)
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
            for(auto copy : copies)
            {
                auto exchangeTags = copy.second;
                std::sort(exchangeTags.begin(),
                          exchangeTags.end(),
                          topoComp(std::make_shared<KernelGraph>(graph)));
                insertBefore(graph, exchangeTags[0], copy.first, copy.first);
            }
        }

        void prefetchScaleLoads(KernelGraph& graph, ContextPtr context)
        {
            auto root = graph.control.roots().only().value();

            std::vector<int>                 preLoopLoad;
            std::map<int, std::vector<int>>  inLoopCopy;
            std::vector<std::pair<int, int>> inLoopLoad;

            auto forLoop = -1;

            // all loads that follow the root
            auto loads
                = filter(graph.control.isElemType<LoadTiled>(), graph.control.depthFirstVisit(root))
                      .to<std::vector>();

            // sort the loads
            // this brings the first load in the sequence to the front
            std::sort(loads.begin(), loads.end(), topoComp(std::make_shared<KernelGraph>(graph)));

            for(auto loadTag : loads)
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

                    preLoopLoad.push_back(topOp);
                    inLoopLoad.push_back(std::make_pair(duplicateChain(graph, {topOp}),
                                                        graph.mapper.get<Unroll>(loadTag, 2)));

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
                        for(auto c : graph.mapper.getCoordinateConnections(exchangeTileTag.value()))
                        {
                            auto maybeExchange = graph.control.get<Exchange>(c.control);
                            if(maybeExchange)
                            {
                                exchangeTag = c.control;
                                break;
                            }
                        }
                        AssertFatal(exchangeTag.has_value());
                        inLoopCopy[copyTag].push_back(
                            getTopSetCoordinate(graph, exchangeTag.value()));
                    }
                }
            }

            if(preLoopLoad.empty())
                return;

            AssertFatal(!loads.empty());
            insertPreLoopLoads(graph, preLoopLoad, loads[0]);

            AssertFatal(forLoop != -1);
            insertInLoopLoads(graph, inLoopLoad, forLoop);

            insertInLoopCopies(graph, inLoopCopy);
        }

        KernelGraph PrefetchScale::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::SwizzleScale");
            auto newGraph = original;

            prefetchScaleLoads(newGraph, m_context);

            return newGraph;
        }
    }
}
