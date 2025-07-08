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

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/Transforms/LoadPacked.hpp>
#include <rocRoller/KernelGraph/Transforms/LoadPacked_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    namespace LoadPackedDetail
    {

        std::map<int, Expression::ExpressionPtr> getStaticCoords(int                controlNode,
                                                                 KernelGraph const& graph)
        {
            std::map<int, Expression::ExpressionPtr> rv;

            // Map from coord to the node that we used to set that coord, so
            // that we can tell if we are above or below there.
            std::map<int, int> coordSetBy;

            auto setCoordNodes = getContainingSetCoordinates(graph, controlNode);

            for(auto setCoordNode : setCoordNodes)
            {
                auto setCoord
                    = graph.control.get<ControlGraph::SetCoordinate>(setCoordNode).value();

                auto connections = graph.mapper.getConnections(setCoordNode);
                AssertFatal(connections.size() == 1,
                            "Invalid SetCoordinate operation; coordinate missing.");
                auto coord = connections[0].coordinate;

                auto iter = coordSetBy.find(coord);
                if(iter == coordSetBy.end())
                {
                    AssertFatal(!rv.contains(coord));

                    rv[coord]         = setCoord.value;
                    coordSetBy[coord] = setCoordNode;
                }
                else
                {
                    auto ordering = graph.control.compareNodes(
                        rocRoller::UpdateCache, setCoordNode, iter->second);
                    if(ordering == ControlGraph::NodeOrdering::LeftInBodyOfRight)
                    {
                        Log::debug("{}: replacing {} with {}.",
                                   coord,
                                   toString(rv[coord]),
                                   toString(setCoord.value));
                        rv[coord]    = setCoord.value;
                        iter->second = setCoordNode;
                    }
                }
            }

            return rv;
        }

        std::map<int, Expression::ExpressionPtr>
            fillRegisterCoords(std::unordered_set<int> const& requiredCoords,
                               KernelGraph const&             graph,
                               ContextPtr                     context)
        {
            using namespace CoordinateGraph;
            std::map<int, Expression::ExpressionPtr> rv;

            auto isRegisterDim = [](auto dim) -> bool {
                using T = std::decay_t<decltype(dim)>;

                return CIsAnyOf<T, Wavefront, Workitem, Workgroup, ForLoop, MacroTileNumber>;
            };

            for(auto coord : requiredCoords)
            {
                if(std::visit(isRegisterDim, graph.coordinates.getNode(coord)))
                {
                    auto reg = Register::Value::Placeholder(
                        context, Register::Type::Vector, DataType::Int32, 1);

                    rv[coord] = reg->expression();
                }
            }

            return rv;
        }

        std::tuple<CoordinateGraph::Transformer, std::set<int>> getFakeTransformerForControlNode(
            int controlNode, KernelGraph const& graph, ContextPtr context)
        {
            auto const& [coords, paths] = findAllRequiredCoordinates(controlNode, graph);

            auto staticCoords = getStaticCoords(controlNode, graph);
            auto regCoords    = fillRegisterCoords(coords, graph, context);

            CoordinateGraph::Transformer xform(&graph.coordinates);

            for(auto const& [coord, expr] : staticCoords)
                xform.setCoordinate(coord, expr);

            for(auto const& [coord, expr] : regCoords)
            {
                xform.setCoordinate(coord, expr);
            }

            std::set<int> remainingCoords;
            for(auto coord : coords)
                if(!xform.hasCoordinate(coord))
                    remainingCoords.insert(coord);

            return {xform, remainingCoords};
        }

        int getFastestMovingCoord(std::set<int> const& coords, KernelGraph const& graph)
        {
            AssertFatal(!coords.empty());

            if(coords.size() == 1)
                return *coords.begin();

            std::optional<int> fastestCoord;

            std::optional<int> maxSubDim;
            for(auto coord : coords)
            {
                auto dim = graph.coordinates.getNode(coord);

                auto visitor = rocRoller::overloaded{
                    [&]<std::derived_from<CoordinateGraph::SubDimension> T>(T const& subdim) {
                        if(!maxSubDim || subdim.dim > maxSubDim)
                        {
                            fastestCoord = coord;
                            maxSubDim    = subdim.dim;
                        }
                    },
                    [&](CoordinateGraph::VGPR const& vgpr) {
                        fastestCoord = coord;
                        maxSubDim    = 1000;
                    },
                    [&](auto const& other) { Log::debug("Other: {}", toString(other)); }};

                std::visit(visitor, dim);
            }

            return fastestCoord.value();
        }
    }

    LoadPacked::LoadPacked(ContextPtr context)
        : m_context(context)
    {
    }

    KernelGraph LoadPacked::apply(KernelGraph const& original)
    {
        TIMER(t, "KernelGraph::LoadPacked");

        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace LoadPackedDetail;

        std::set<int> changedDims;
        auto          rv = original;

        auto pred = [&](int node) {
            auto visitor = [](auto op) -> bool {
                using T = std::decay_t<decltype(op)>;

                if constexpr(CIsAnyOf<T, LoadTiled, StoreTiled, LoadLDSTile, StoreLDSTile>)
                {
                    return DataTypeInfo::Get(op.varType).packedVariableType().has_value();
                }

                return false;
            };

            return rv.control.getElementType(node) == Graph::ElementType::Node
                   && std::visit(visitor, rv.control.getNode(node));
        };

        auto roots = rv.control.roots().to<std::vector>();

        for(auto node : rv.control.findNodes(roots, pred))
        {
            Log::debug("Load/store op: ({})", node);
            auto [target, direction] = getOperationTarget(node, rv);
            target                   = getTransformTarget(target, rv);

            auto op = rv.control.getNode(node);

            auto varType    = getVariableType(op);
            auto packedType = DataTypeInfo::Get(varType).packedVariableType().value();
            auto packedInfo = DataTypeInfo::Get(packedType);

            auto waveTileTag = rv.mapper.get<WaveTile>(node);
            if(waveTileTag > 0)
            {
                auto waveTile = rv.coordinates.get<WaveTile>(waveTileTag);

                auto elements = waveTile.value().elements();
                uint wfs      = m_context->kernel()->wavefront_size();

                AssertFatal(elements % wfs == 0);

                auto elementsPerThread = elements / wfs;
                if(elementsPerThread < packedInfo.packing)
                {
                    Log::debug(
                        "Skipping node {} due to insufficient elementsPerThread: {} (packing {})",
                        node,
                        elementsPerThread,
                        packedInfo.packing);
                    continue;
                }
            }

            if(rv.mapper.get<ElementNumber>(node, 0) > 0
               && rv.mapper.get<ElementNumber>(node, 1) > 0)
            {
                auto [elemXTag, elemX] = rv.getDimension<ElementNumber>(node, 0);
                auto [elemYTag, elemY] = rv.getDimension<ElementNumber>(node, 1);
                auto const m           = getUnsignedInt(evaluate(elemX.size));
                auto       n           = getUnsignedInt(evaluate(elemY.size));
                if(m * n < packedInfo.packing)
                {
                    Log::debug("Skipping node {}: {}x{} due to ElementNumber size.",
                               node,
                               ShowValue(m),
                               ShowValue(n));
                    continue;
                }
            }

            auto [xform, remainingCoords] = getFakeTransformerForControlNode(node, rv, m_context);

            Log::debug("Remaining coords: {}", concatenate(remainingCoords));

            if(remainingCoords.empty())
                continue;

            auto fastestCoord = getFastestMovingCoord(remainingCoords, rv);

            for(auto coord : remainingCoords)
            {
                xform.setCoordinate(coord, Expression::literal(0u));
            }

            Log::debug("Fastest coord: {}, Target: {}, looking {}",
                       fastestCoord,
                       target,
                       toString(direction));

            auto exprs = direction != Graph::Direction::Upstream
                             ? xform.reverseStride(fastestCoord, Expression::literal(1u), {target})
                             : xform.forwardStride(fastestCoord, Expression::literal(1u), {target});

            Log::debug("Stride: {}", toString(exprs[0]));

            if(evaluationTimes(exprs[0])[Expression::EvaluationTime::Translate])
            {
                Log::debug("Stride: {}", toString(evaluate(exprs[0])));

                if(getUnsignedInt(evaluate(exprs[0])) == 1)
                {
                    Log::debug("Setting {} varType from {} to {}.",
                               node,
                               toString(varType),
                               toString(packedType));
                    setVariableType(op, packedType);
                    rv.control.setElement(node, op);
                }
            }
        }

        return rv;
    }
}
