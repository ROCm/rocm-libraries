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

#include <algorithm>
#include <variant>

#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AssignIndexExpressions.hpp>
#include <rocRoller/KernelGraph/Transforms/AssignIndexExpressions_detail.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTile_details.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    using namespace CoordinateGraph;
    using namespace ControlGraph;
    namespace Expression = rocRoller::Expression;
    using namespace Expression;

    using GD = Graph::Direction;

    struct IndexChainSpec
    {
        int              target;
        std::vector<int> coords;
        int              location;
        Graph::Direction direction;
        int              forLoop                    = -1;
        bool             replaceWithScope           = true;
        bool             isStorePartOfGlobalToLDSOp = false;
    };

    bool operator<(const IndexChainSpec& a, const IndexChainSpec& b)
    {
        return std::tie(a.target,
                        a.coords,
                        a.location,
                        a.direction,
                        a.forLoop,
                        a.replaceWithScope,
                        a.isStorePartOfGlobalToLDSOp)
               < std::tie(b.target,
                          b.coords,
                          b.location,
                          b.direction,
                          b.forLoop,
                          b.replaceWithScope,
                          b.isStorePartOfGlobalToLDSOp);
    }

    /**
     * @brief Information needed to create Assign nodes for a single dimension.
     */
    struct ChainNodeInfo
    {
        int      nopTag     = -1; // Placeholder NOP node
        int      target     = -1;
        int      increment  = -1;
        int      baseOffset = -1; // The base offset coordinate (not the offset of this node)
        int      offset     = -1;
        int      stride     = -1;
        int      buffer     = -1;
        bool     forward    = false;
        DataType valueType  = DataType::Count;
        DataType offsetType = DataType::Count;
        DataType strideType = DataType::Count;
        bool     isStorePartOfGlobalToLDSOp = false;
    };

    struct IndexChain
    {
        int top, bottom;

        std::vector<ChainNodeInfo>      nodeInfos;
        std::vector<DeferredConnection> connections;

        int update = -1;
    };

    struct RequiredCoordinateInfo
    {
        int  coord, base, sdim;
        bool isUnroll;
        bool needsUpdate;
    };

    using BufferMap      = std::map<int, int>;
    using BaseAddressMap = std::map<int, int>;

    /**
     * @brief Return existing Buffer for load/stores from/to `dst`.
     *
     * Returns -1 if the operation doesn't need a buffer descriptor.
     *
     * If a Buffer edge doesn't already exist, we create a new
     * Workgroup coordinate and attach it with a Buffer edge to the
     * `dst`.
     */
    int getBuffer(KernelGraph& graph,
                  int          opTag,
                  int          dst,
                  BufferMap&   bufferMap,
                  bool         isStorePartOfDirect2LDSOp)
    {
        auto op                 = graph.control.getElement(opTag);
        const auto [_, macTile] = graph.getDimension<MacroTile>(opTag);
        if(not(isOperation<LoadTiled>(op) or isOperation<StoreTiled>(op)
               or isOperation<LoadTileDirect2LDS>(op))
           or isStorePartOfDirect2LDSOp or macTile.memoryType == MemoryType::WAVE_FROM_GLOBAL)
            return -1;

        if(!bufferMap.contains(dst))
        {
            auto wg        = graph.coordinates.addElement(Workgroup());
            bufferMap[dst] = graph.coordinates.addElement(Buffer(), {wg}, {dst});
        }

        return bufferMap[dst];
    }

    /**
     * @brief Return existing BaseAddress for load/stores from/to `dst`.
     *
     * Returns -1 if the operation doesn't need a baseAddress.
     *
     * If a BaseAddress edge doesn't already exist, we create a new
     * Workgroup coordinate and attach it with a BaseAddress edge to the
     * `dst`.
     */
    int getBaseAddress(KernelGraph& graph, int opTag, int dst, BaseAddressMap& baseAddressMap)
    {
        auto op                 = graph.control.getElement(opTag);
        const auto [_, macTile] = graph.getDimension<MacroTile>(opTag);
        if(not(isOperation<LoadTiled>(op) and macTile.memoryType == MemoryType::WAVE_FROM_GLOBAL))
            return -1;

        if(!baseAddressMap.contains(dst))
        {
            auto wg             = graph.coordinates.addElement(Workgroup());
            baseAddressMap[dst] = graph.coordinates.addElement(BaseAddress(), {wg}, {dst});
        }

        return baseAddressMap[dst];
    }

    /**
     * @brief True if ForLoopOp has a translate-time increment.
     */
    bool uniformForLoop(std::optional<int> maybeForLoop, KernelGraph const& kgraph)
    {
        if(!maybeForLoop)
            return false;

        auto [lhs, rhs] = getForLoopIncrement(kgraph, *maybeForLoop);
        return evaluationTimes(rhs)[EvaluationTime::Translate];
    }

    /**
     * @brief Create a placeholder NOP node and store info for later Assign creation.
     */
    ChainNodeInfo makeIndexPlaceholder(KernelGraph& graph,
                                       int          target,
                                       int          increment,
                                       int          base,
                                       int          offset,
                                       int          stride,
                                       int          buffer,
                                       int          baseAddress,
                                       bool         forward,
                                       DataType     valueType,
                                       DataType     offsetType,
                                       DataType     strideType,
                                       bool         isStorePartOfGlobalToLDSOp)
    {
        // Create a NOP placeholder that will be replaced with Assign nodes later
        auto nopTag = graph.control.addElement(NOP());

        rocRoller::Log::getLogger()->debug(
            "KernelGraph::makeIndexPlaceholder: nop {} {}/{} {}; {}/{}/{} {}/{}",
            nopTag,
            target,
            increment,
            forward,
            base,
            offset,
            stride,
            buffer,
            baseAddress);

        return ChainNodeInfo{nopTag,
                             target,
                             increment,
                             base,
                             offset,
                             stride,
                             buffer,
                             forward,
                             valueType,
                             offsetType,
                             strideType,
                             isStorePartOfGlobalToLDSOp};
    }

    /**
     * @brief Get coordinates in `path` attached to `coordinate` via a
     * CoordinateTransformEdge.
     */
    int getNeighbourNodeInPath(int                            coordinate,
                               Graph::Direction               direction,
                               std::unordered_set<int> const& path,
                               KernelGraph const&             graph)
    {
        auto neighbourNodes
            = (direction == Graph::Direction::Upstream)
                  ? graph.coordinates
                        .getOutputNodeIndices(coordinate,
                                              rocRoller::KernelGraph::CoordinateGraph::isEdge<
                                                  CoordinateTransformEdge>)
                        .to<std::unordered_set>()
                  : graph.coordinates
                        .getInputNodeIndices(coordinate,
                                             rocRoller::KernelGraph::CoordinateGraph::isEdge<
                                                 CoordinateTransformEdge>)
                        .to<std::unordered_set>();

        for(auto tag : neighbourNodes)
        {
            if(path.contains(tag))
                return tag;
        }

        return -1;
    }

    /**
     * @brief Get list of required coordinates, and how they relate to
     * each other.
     *
     * Builds a list of coordinates, slow-to-fast, that need
     * offset/strides for operation `op`.
     */
    std::vector<RequiredCoordinateInfo> getRequiredCoordinatesInfo(int                op,
                                                                   int                location,
                                                                   KernelGraph const& graph,
                                                                   bool isStorePartOfGlobalToLDSOp)
    {
        auto [target, direction] = getOperationTarget(op, graph, isStorePartOfGlobalToLDSOp);
        auto [required, path]    = findRequiredCoordinates(target, direction, graph);
        auto codegen = getCodeGeneratorCoordinates(graph, op, isStorePartOfGlobalToLDSOp);

        std::set<int>    isForLoop, isUnroll;
        std::vector<int> ordered;

        // If location is a ForLoop, its coordinate is the slowest.
        if(location != -1)
        {
            auto maybeForLoop = graph.control.get<ForLoopOp>(location);
            if(maybeForLoop)
            {
                auto forLoopCoord = graph.mapper.get<ForLoop>(location);
                forLoopCoord      = followIdentify(forLoopCoord, graph);

                auto coord = getNeighbourNodeInPath(forLoopCoord, direction, path, graph);
                if(coord != -1)
                {
                    ordered.push_back(coord);
                    isForLoop.insert(coord);
                }
            }
        }

        // Next, consider Unroll coordinates.
        auto unrolls = filterCoordinates<Unroll>(required, graph);

        for(auto unroll : unrolls)
        {
            // In StreamK, Unroll coordinates are connected via Identify edges.
            // followIdentify resolves these chains (or returns the original if none).
            auto unrollTarget = followIdentify(unroll, graph);

            // Find a neighbour of unrollTarget that's actually in the path
            auto coord = getNeighbourNodeInPath(unrollTarget, direction, path, graph);
            if(coord != -1 && !isForLoop.contains(coord))
            {
                auto it = std::find(codegen.cbegin(), codegen.cend(), coord);
                if(it == codegen.cend())
                {
                    // Check if this coordinate is already in ordered
                    if(std::find(ordered.begin(), ordered.end(), coord) == ordered.end())
                    {
                        ordered.push_back(coord);
                    }
                    isUnroll.insert(coord);
                }
            }
        }

        // Finally, the code-gen coordinates are the fastest moving.
        for(auto x : codegen)
            ordered.push_back(x);

        // Now build list... the slowest dimension doesn't have a
        // "base"; subsequent dimensions use the previous one as their
        // base.
        std::vector<RequiredCoordinateInfo> rv;

        int base = -1;
        for(auto coord : ordered)
        {
            // Compute the sub-dimension for code-gen coordinates.
            // TODO Slow to fast; lift this from Tensor directly
            int sdim = -1;
            {
                auto it = std::find(codegen.cbegin(), codegen.cend(), coord);
                if(it != codegen.cend())
                    sdim = std::distance(codegen.cbegin(), it);
            }

            if(isStorePartOfGlobalToLDSOp)
            {
                sdim += ordered.size();
            }

            if(!isUnroll.contains(coord))
            {
                auto needsUpdate = isForLoop.contains(coord) && uniformForLoop(location, graph);
                rv.push_back({coord, base, sdim, false, needsUpdate});
                base = coord;
            }
            else
            {
                rv.push_back({coord, -1, -1, true, false});
            }
        }

        return rv;
    }

    /**
     * @brief Return datatype that should be used for the offset when
     * generating `op`.
     */
    DataType getOffsetDataType(int op, KernelGraph const& graph, bool isStorePartOfGGlobalToLDSOp)
    {
        DataType rv = DataType::UInt64;
        auto     s  = graph.control.get<StoreTiled>(op);
        auto     l  = graph.control.get<LoadTiled>(op);
        auto     ll = graph.control.get<LoadLDSTile>(op);
        auto     sl = graph.control.get<StoreLDSTile>(op);

        auto isGlobalLoad = false;
        if(l)
        {
            auto [_, macTile] = graph.getDimension<MacroTile>(op);
            if(macTile.memoryType == MemoryType::WAVE_FROM_GLOBAL)
            {
                isGlobalLoad = true;
            }
        }

        if(s || (l and not isGlobalLoad) || ll || sl || isStorePartOfGGlobalToLDSOp)
        {
            rv = DataType::UInt32;
        }
        return rv;
    }

    void addUnrollStrideConnection(KernelGraph&                     kgraph,
                                   int                              candidate,
                                   bool                             isStorePartOfGlobalToLDSOp,
                                   const std::vector<int>&          strideCoords,
                                   std::vector<DeferredConnection>& connections)
    {
        auto [target, direction]
            = getOperationTarget(candidate, kgraph, isStorePartOfGlobalToLDSOp);
        auto [required, path] = findRequiredCoordinates(target, direction, kgraph);
        auto unrolls          = filterCoordinates<Unroll>(required, kgraph);

        for(auto const& unroll : unrolls)
        {
            auto proxy = followIdentify(unroll, kgraph);

            auto const subDimension = kgraph.mapper.getConnectionSubdimension(candidate, unroll);
            // Find the neighbour of the Unroll that:
            // 1. is in the load/store coordinate transform path
            // 2. has a Stride edge connected to it
            std::vector<int> neighbourNodes;
            if(direction == Graph::Direction::Downstream)
                neighbourNodes = kgraph.coordinates.parentNodes(proxy).to<std::vector>();
            else
                neighbourNodes = kgraph.coordinates.childNodes(proxy).to<std::vector>();

            for(auto neighbourNode : neighbourNodes)
            {
                if(path.contains(neighbourNode))
                {
                    auto neighbourEdges = kgraph.coordinates.getNeighbours(
                        neighbourNode, Graph::opposite(direction));
                    for(auto neighbourEdge : neighbourEdges)
                    {
                        auto maybeStride = kgraph.coordinates.get<Stride>(neighbourEdge);
                        if(maybeStride
                           && std::find(strideCoords.begin(), strideCoords.end(), neighbourEdge)
                                  != strideCoords.end())
                        {
                            auto maybeStrideTag = neighbourEdge;
                            auto newConnection  = makeConnection<Stride, Connections::UnrollStride>(
                                maybeStrideTag, subDimension);
                            connections.push_back(newConnection);
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Create an index assignment chain for `op`.
     */
    IndexChain createIndexChain(KernelGraph&          graph,
                                int                   op,
                                ExpressionPtr         step,
                                IndexChainSpec const& spec,
                                BufferMap&            bufferMap,
                                BaseAddressMap&       baseAddressMap)
    {
        rocRoller::Log::getLogger()->debug(
            "KernelGraph::AssignIndexExpressions(): op {} location {}", op, spec.location);

        auto dtype = getDataType(graph.control.getNode(op));

        auto [target, direction] = getOperationTarget(op, graph, spec.isStorePartOfGlobalToLDSOp);

        int                             update = -1;
        std::vector<int>                chain;
        std::vector<ChainNodeInfo>      nodeInfos;
        std::vector<DeferredConnection> connections;
        std::map<int, int>              offsetOfCoord;
        std::vector<int>                strideCoords;

        // Use spec.forLoop as the location if it exists (for hoisted chains),
        // otherwise use spec.location
        int locationForCoordInfo = (spec.forLoop > 0) ? spec.forLoop : spec.location;
        for(auto info : getRequiredCoordinatesInfo(
                op, locationForCoordInfo, graph, spec.isStorePartOfGlobalToLDSOp))
        {
            // Add coordinate nodes for Offset/Stride/Buffer
            int offset = -1, stride = -1, buffer = -1, baseAddress = -1;

            {
                auto inCoord  = target;
                auto outCoord = info.coord;
                if(direction == Graph::Direction::Upstream)
                {
                    std::swap(inCoord, outCoord);
                }

                if(!info.isUnroll)
                    offset = graph.coordinates.addElement(Offset(), {inCoord}, {outCoord});
                stride = graph.coordinates.addElement(Stride(), {inCoord}, {outCoord});
                if(info.base == -1 && offset != -1)
                {
                    const bool isDirect2LDS
                        = isOperation<LoadTileDirect2LDS>(graph.control.getElement(op));
                    const bool isStorePartOfDirect2LDSOp
                        = (isDirect2LDS && spec.isStorePartOfGlobalToLDSOp);
                    buffer = getBuffer(graph, op, target, bufferMap, isStorePartOfDirect2LDSOp);
                    baseAddress = getBaseAddress(graph, op, target, baseAddressMap);
                }
            }

            offsetOfCoord[info.coord] = offset;

            int base = (info.base == -1) ? -1 : offsetOfCoord.at(info.base);

            // For future: choose type based on buffer or non-buffer
            auto offsetDataType = getOffsetDataType(op, graph, spec.isStorePartOfGlobalToLDSOp);
            auto strideDataType = DataType::UInt64;

            if(info.isUnroll)
            {
                offsetDataType = DataType::Int64;
                strideDataType = DataType::Int64;
            }
            // Create placeholder NOP and store info for later Assign creation
            auto nodeInfo = makeIndexPlaceholder(graph,
                                                 target,
                                                 info.coord,
                                                 base,
                                                 offset,
                                                 stride,
                                                 buffer,
                                                 baseAddress,
                                                 direction == Graph::Direction::Upstream,
                                                 dtype,
                                                 offsetDataType,
                                                 strideDataType,
                                                 spec.isStorePartOfGlobalToLDSOp);
            chain.push_back(nodeInfo.nopTag);
            nodeInfos.push_back(nodeInfo);

            // Add connections for register allocate, and so tracer
            // can determine correct lifetimes
            if(offset != -1)
                connections.push_back(DC<Offset>(offset, info.sdim));
            if(stride != -1)
                connections.push_back(DC<Stride>(stride, info.sdim));
            if(buffer != -1)
                connections.push_back(DC<Buffer>(buffer));
            if(baseAddress != -1)
                connections.push_back(DC<BaseAddress>(baseAddress));
            if(base != -1)
                connections.push_back(
                    makeConnection<Offset, Connections::BaseOffset>(base, info.sdim));

            // save all stride coordinates for the memory operation
            // then select the unroll stride and add it to connection
            if(stride != -1)
                strideCoords.push_back(stride);

            if(info.needsUpdate)
            {
                auto offsetExpr = std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{offset, Register::Type::Vector, offsetDataType});
                auto strideExpr = std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{stride, Register::Type::Scalar, DataType::UInt64});

                if(step == nullptr)
                    update = graph.control.addElement(Assign{
                        Register::Type::Vector, convert(offsetDataType, offsetExpr + strideExpr)});
                else
                    update = graph.control.addElement(
                        Assign{Register::Type::Vector,
                               convert(offsetDataType, offsetExpr + step * strideExpr)});
                graph.mapper.connect(update, offset, NaryArgument::DEST);
            }
        }

        addUnrollStrideConnection(
            graph, op, spec.isStorePartOfGlobalToLDSOp, strideCoords, connections);

        for(int i = 1; i < chain.size(); ++i)
            graph.control.addElement(Sequence(), {chain[i - 1]}, {chain[i]});

        return {chain.front(), chain.back(), nodeInfos, connections, update};
    }

    namespace
    {
        /**
         * @brief Find the corresponding KLoopTail for a given KLoop.
         *
         * @param kgraph
         * @param kLoop The KLoop tag
         * @return The corresponding KLoopTail tag, or std::nullopt if none exists
         */
        std::optional<int> FindCorrespondingKLoopTail(KernelGraph const& kgraph, int kLoop)
        {
            // Strategy 1: Search downstream via Sequence edges (UnrollLoops case)
            for(auto node : kgraph.control.depthFirstVisit(kLoop, Graph::Direction::Downstream))
            {
                auto maybeForLoop = kgraph.control.get<ForLoopOp>(node);
                if(maybeForLoop && maybeForLoop->loopName == rocRoller::KLOOPTAIL)
                    return node;
            }

            // Strategy 2: Search for siblings under common parent Scope (AddPrefetch case)
            for(auto ancestor : kgraph.control.breadthFirstVisit(kLoop, Graph::Direction::Upstream))
            {
                if(!kgraph.control.get<Scope>(ancestor))
                    continue;

                // Search all descendants of this Scope for a KLoopTail
                for(auto descendant :
                    kgraph.control.depthFirstVisit(ancestor, Graph::Direction::Downstream))
                {
                    if(descendant == kLoop)
                        continue;

                    auto maybeForLoop = kgraph.control.get<ForLoopOp>(descendant);
                    if(maybeForLoop && maybeForLoop->loopName == rocRoller::KLOOPTAIL)
                        return descendant;
                }
            }

            return std::nullopt;
        }

        /**
         * @brief Find the closest common ancestor Scope between two nodes.
         *
         * @param kgraph
         * @param nodeA
         * @param nodeB
         * @return The tag of the common ancestor Scope, or std::nullopt if none exists
         */
        std::optional<int> FindCommonAncestorScope(KernelGraph const& kgraph, int nodeA, int nodeB)
        {
            // Collect all ancestors of nodeA
            auto ancestorsA = kgraph.control.breadthFirstVisit(nodeA, Graph::Direction::Upstream)
                                  .to<std::set>();

            // Traverse from nodeB and find first common ancestor that is a Scope
            for(auto node : kgraph.control.breadthFirstVisit(nodeB, Graph::Direction::Upstream))
            {
                if(ancestorsA.contains(node) && kgraph.control.get<Scope>(node))
                    return node;
            }

            return std::nullopt;
        }

    } // anonymous namespace

    namespace AssignIndexExpressionsDetail
    {
        using namespace CoordinateGraph;
        using namespace ControlGraph;

        std::pair<uint, uint>
            getElementBlockValues(KernelGraph const& graph, int target, const bool isTransposed)
        {
            namespace CT            = rocRoller::KernelGraph::CoordinateGraph;
            uint elementBlockNumber = 0;
            uint elementBlockIndex  = 0;

            using OpsAndTilesType
                = std::tuple<std::pair<int, Operation>, std::pair<int, MacroTile>, DataType>;
            std::vector<OpsAndTilesType> targetOpsAndTiles;

            for(auto conn : graph.mapper.getCoordinateConnections(target))
            {
                auto     opTag = conn.control;
                auto     op    = std::get<Operation>(graph.control.getElement(opTag));
                DataType dataType;
                if(std::visit(rocRoller::overloaded{[&](LoadTiled& load) {
                                                        dataType = load.varType.dataType;
                                                        return true;
                                                    },
                                                    [&](LoadLDSTile& load) {
                                                        dataType = load.varType.dataType;
                                                        return true;
                                                    },
                                                    [&](StoreTiled& store) {
                                                        dataType = store.varType.dataType;
                                                        return true;
                                                    },
                                                    [&](StoreLDSTile& store) {
                                                        dataType = store.varType.dataType;
                                                        return true;
                                                    },
                                                    [&](auto& other) { return false; }},
                              op))
                {
                    auto [macTileTag, macTile] = graph.getDimension<MacroTile>(opTag);

                    auto maybeParentTile = only(
                        graph.coordinates.getOutputNodeIndices(macTileTag, CT::isEdge<Duplicate>));

                    if(maybeParentTile)
                    {
                        macTileTag = *maybeParentTile;
                        macTile    = *graph.coordinates.get<MacroTile>(macTileTag);
                    }

                    targetOpsAndTiles.push_back({{opTag, op}, {macTileTag, macTile}, dataType});
                }
            }

            // If we get here and targetOpsAndTiles is empty, it is
            // because: we are using Direct2LDS to load scaling data
            // that will be swizzled (or is already pre-swizzled): no
            // remaining operations are directly connected to the LDS
            // target.
            if(targetOpsAndTiles.empty())
            {
                // Just look upstream of target
                auto [required, path]
                    = findRequiredCoordinates(target, Graph::Direction::Upstream, graph);
                for(auto coordTag : required)
                {
                    auto maybeElementNumber = graph.coordinates.get<ElementNumber>(coordTag);
                    if(maybeElementNumber)
                    {
                        if(maybeElementNumber->dim == 0)
                            elementBlockNumber = getUnsignedInt(evaluate(maybeElementNumber->size));
                        else if(maybeElementNumber->dim == 1)
                            elementBlockIndex = getUnsignedInt(evaluate(maybeElementNumber->size));
                    }
                }
                return {elementBlockNumber, elementBlockIndex};
            }

            auto [tagAndOp, tagAndTile, dataType] = [](auto opsAndTiles) -> OpsAndTilesType {
                for(OpsAndTilesType& elem : opsAndTiles)
                {
                    auto memType = std::get<1>(elem).second.memoryType;
                    if(memType == MemoryType::WAVE || memType == MemoryType::WAVE_SWIZZLE)
                    {
                        return elem;
                    }
                }
                return opsAndTiles[0];
            }(targetOpsAndTiles);

            auto [opTag, op]           = tagAndOp;
            auto [macTileTag, macTile] = tagAndTile;

            if(macTile.memoryType == MemoryType::VGPR
               || (macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR
                   && macTile.memoryType == MemoryType::WAVE_SPLIT))
            {
                auto [elementNumberXTag, elementNumberX]
                    = graph.getDimension<ElementNumber>(opTag, 0);
                AssertFatal(Expression::evaluationTimes(
                                elementNumberX.size)[Expression::EvaluationTime::Translate],
                            "Could not determine ElementNumberX size at translate-time.\n",
                            ShowValue(elementNumberX));

                auto [elementNumberYTag, elementNumberY]
                    = graph.getDimension<ElementNumber>(opTag, 1);
                AssertFatal(Expression::evaluationTimes(
                                elementNumberY.size)[Expression::EvaluationTime::Translate],
                            "Could not determine ElementNumber size at translate-time.\n",
                            ShowValue(elementNumberY));

                elementBlockNumber = getUnsignedInt(evaluate(elementNumberX.size));
                elementBlockIndex  = getUnsignedInt(evaluate(elementNumberY.size));
            }
            else if(macTile.memoryType == MemoryType::WAVE
                    || macTile.memoryType == MemoryType::WAVE_SWIZZLE)
            {
                auto [vgprBlockNumberTag, vgprBlockNumber]
                    = graph.getDimension<VGPRBlockNumber>(opTag, 0);
                AssertFatal(Expression::evaluationTimes(
                                vgprBlockNumber.size)[Expression::EvaluationTime::Translate],
                            "Could not determine VGPRBlockNumber size at translate-time.\n",
                            ShowValue(vgprBlockNumber));

                auto [vgprBlockIndexTag, vgprBlockIndex]
                    = graph.getDimension<VGPRBlockIndex>(opTag, 0);
                AssertFatal(Expression::evaluationTimes(
                                vgprBlockIndex.size)[Expression::EvaluationTime::Translate],
                            "Could not determine VGPRBlockIndex size at translate-time.\n",
                            ShowValue(vgprBlockIndex));

                elementBlockNumber = getUnsignedInt(evaluate(vgprBlockNumber.size));
                elementBlockIndex  = getUnsignedInt(evaluate(vgprBlockIndex.size));
                if(isScaleType(dataType))
                {
                    // Scales are another special case here. For Scales we need
                    // to get VGPR coordinate instead of VGPRBlockNumber/Index
                    // (see addLoadSwizzleTileCT).
                    auto [vgprTag, vgpr] = graph.getDimension<VGPR>(opTag, 0);
                    AssertFatal(Expression::evaluationTimes(
                                    vgpr.size)[Expression::EvaluationTime::Translate],
                                "Could not determine VGPR size at translate-time.\n",
                                ShowValue(vgpr));
                    // Multiplying by elementBlockNumber here forces the use
                    // of the widest load/store possible
                    elementBlockIndex = elementBlockNumber * getUnsignedInt(evaluate(vgpr.size));
                }

                if((!LowerTileDetails::isTileOfSubDwordTypeWithNonContiguousVGPRBlocks(
                        dataType,
                        {.m = macTile.subTileSizes[0],
                         .n = macTile.subTileSizes[1],
                         .k = macTile.subTileSizes[2]})
                    || isScaleType(dataType))
                   && !isTransposed)
                {
                    // For Scales and other kinds of tiles, VGPRBlockIndex holds
                    // number of VGPR per block and not elements per VGPRBlock.
                    elementBlockIndex *= packingFactorForDataType(dataType);
                }
            }
            else
            {
                Throw<FatalError>(
                    "Could not find ElementNumber or VGPRBlockNumber/Index coordinates.\n",
                    ShowValue(op),
                    ShowValue(macTile));
            }

            AssertFatal(elementBlockNumber > 0 && elementBlockIndex > 0,
                        "elemementBlockNumber & elementBlockIndex must be greater than zero. ",
                        ShowValue(elementBlockNumber),
                        ShowValue(elementBlockIndex));
            return {elementBlockNumber, elementBlockIndex};
        }

        int makeAssignBase(KernelGraph&              graph,
                           IndexComputeParams const& params,
                           const int                 target,
                           const int                 offset,
                           const bool                maybeLDS,
                           const bool                isTransposed,
                           const ContextPtr          context,
                           Transformer&              coords)
        {
            auto toBytes = [&](Expression::ExpressionPtr expr) -> Expression::ExpressionPtr {
                uint numBits = DataTypeInfo::Get(params.valueType).elementBits;

                // TODO: This would be a good place to add a GPU
                // assert.  If numBits is not a multiple of 8, assert
                // that (expr * numBits) is a multiple of 8.
                Log::debug("  toBytes: {}: numBits {}", toString(params.valueType), numBits);

                if(numBits % 8u == 0)
                    return expr * L(numBits / 8u);
                return (expr * L(numBits)) / L(8u);
            };

            auto offsetRegisterType = Register::Type::Vector;
            if(params.isStorePartOfGlobalToLDS)
                offsetRegisterType = Register::Type::Scalar;

            auto indexExpr
                = params.forward ? coords.forward({target})[0] : coords.reverse({target})[0];

            auto const& typeInfo = DataTypeInfo::Get(params.valueType);
            auto        numBits  = DataTypeInfo::Get(typeInfo.segmentVariableType).elementBits;

            auto const& arch = context->targetArchitecture();
            const auto  needsPadding
                = numBits == 6 && isTransposed
                  && arch.HasCapability(GPUCapability::DSReadTransposeB6PaddingBytes);

            Expression::ExpressionPtr paddingBytes{L(0u)};
            if(needsPadding && maybeLDS)
            {
                uint elementsPerTrLoad = bitsPerTransposeLoad(arch, numBits) / numBits;
                auto extraLdsBytes     = extraLDSBytesPerElementBlock(arch, numBits);
                paddingBytes           = indexExpr / L(elementsPerTrLoad) * L(extraLdsBytes);
            }

            auto expr = toBytes(indexExpr) + paddingBytes;

            if(params.isStorePartOfGlobalToLDS)
            {
                expr = std::make_shared<Expression::Expression>(Expression::ToScalar{expr});
            }

            auto assignNode         = Assign{offsetRegisterType, convert(params.offsetType, expr)};
            assignNode.variableType = params.offsetType;
            auto assignTag          = graph.control.addElement(assignNode);
            graph.mapper.connect(assignTag, offset, NaryArgument::DEST);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::makeAssignBase: assign {} expression {} to offset {}",
                assignTag,
                toString(assignNode.expression),
                offset);

            return assignTag;
        }

        int makeAssignStride(KernelGraph&              graph,
                             IndexComputeParams const& params,
                             const int                 target,
                             const int                 stride,
                             const int                 increment,
                             bool                      maybeLDS,
                             const bool                isTransposed,
                             const ContextPtr          context,
                             Transformer&              coords)
        {
            auto toBytes = [&](Expression::ExpressionPtr expr) -> Expression::ExpressionPtr {
                uint numBits = DataTypeInfo::Get(params.valueType).elementBits;

                // TODO: This would be a good place to add a GPU
                // assert.  If numBits is not a multiple of 8, assert
                // that (expr * numBits) is a multiple of 8.
                Log::debug("  toBytes: {}: numBits {}", toString(params.valueType), numBits);

                if(numBits % 8u == 0)
                    return expr * L(numBits / 8u);
                return (expr * L(numBits)) / L(8u);
            };

            auto indexExpr = params.forward ? coords.forwardStride(increment, L(1), {target})[0]
                                            : coords.reverseStride(increment, L(1), {target})[0];

            // We have to manually invoke m_fastArith here since it can't traverse into the
            // RegisterTagManager.
            // TODO: Revisit storing expressions in the RegisterTagManager.
            bool unitStride = false;
            if(Expression::evaluationTimes(indexExpr)[Expression::EvaluationTime::Translate])
            {
                if(getUnsignedInt(evaluate(indexExpr)) == 1u)
                    unitStride = true;
            }

            uint                      elementBlockSize = 0;
            Expression::ExpressionPtr elementBlockStride;
            Expression::ExpressionPtr trLoadPairStride;
            Expression::ExpressionPtr elementBlockStridePaddingBytes{L(0u)};
            Expression::ExpressionPtr trLoadPairStridePaddingBytes{L(0u)};
            Expression::ExpressionPtr indexExprPaddingBytes{L(0u)};

            auto const& typeInfo = DataTypeInfo::Get(params.valueType);
            auto        numBits  = DataTypeInfo::Get(typeInfo.segmentVariableType).elementBits;

            if(numBits == 16 || numBits == 8 || numBits == 6 || numBits == 4)
            {
                auto [elementBlockNumber, elementBlockIndex]
                    = getElementBlockValues(graph, target, isTransposed);

                elementBlockSize = elementBlockIndex;

                auto const& arch = context->targetArchitecture();
                if(isTransposed)
                {
                    // See addLoadWaveTileCTF8F6F4 in LowerTile.cpp
                    const auto wfs = arch.GetCapability(GPUCapability::DefaultWavefrontSize);
                    uint const numVBlocks
                        = wfs == 64 ? (numBits == 8 ? 2 : 1) : (numBits == 8 ? 4 : 2);
                    elementBlockSize = (elementBlockNumber / numVBlocks) * elementBlockSize;
                }
                AssertFatal(elementBlockSize > 0, "Invalid elementBlockSize: ", elementBlockSize);

                const auto needsPadding
                    = numBits == 6 && isTransposed
                      && arch.HasCapability(GPUCapability::DSReadTransposeB6PaddingBytes);

                // Padding is added after every 16 elements, thus for F6 datatypes that will
                // be transpose loaded from LDS elementBlockSize is set to 16 instead of 32.
                if(needsPadding)
                {
                    elementBlockSize = 16;
                }

                elementBlockStride
                    = params.forward
                          ? coords.forwardStride(increment, L(elementBlockSize), {target})[0]
                          : coords.reverseStride(increment, L(elementBlockSize), {target})[0];

                uint elementsPerTrLoad = elementBlockIndex;
                trLoadPairStride
                    = params.forward
                          ? coords.forwardStride(increment, L(elementsPerTrLoad), {target})[0]
                          : coords.reverseStride(increment, L(elementsPerTrLoad), {target})[0];

                if(needsPadding && maybeLDS)
                {
                    uint elementsPerTrLoad = bitsPerTransposeLoad(arch, numBits) / numBits;
                    auto extraLdsBytes     = extraLDSBytesPerElementBlock(arch, numBits);
                    elementBlockStridePaddingBytes
                        = elementBlockStride / L(elementsPerTrLoad) * L(extraLdsBytes);
                    trLoadPairStridePaddingBytes
                        = trLoadPairStride / L(elementsPerTrLoad) * L(extraLdsBytes);
                    indexExprPaddingBytes = indexExpr / L(elementsPerTrLoad) * L(extraLdsBytes);
                }
            }

            auto assignNode
                = Assign{Register::Type::Vector, toBytes(indexExpr) + indexExprPaddingBytes};
            assignNode.variableType = params.strideType;
            assignNode.strideExpressionAttributes
                = {params.strideType,
                   unitStride,
                   elementBlockSize,
                   toBytes(elementBlockStride) + elementBlockStridePaddingBytes,
                   toBytes(trLoadPairStride) + trLoadPairStridePaddingBytes};
            auto assignTag = graph.control.addElement(assignNode);
            graph.mapper.connect(assignTag, stride, NaryArgument::DEST);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::makeAssignStride: assign {} expression {} to stride {}",
                assignTag,
                toString(assignNode.expression),
                stride);
            return assignTag;
        }

        int makeBuffer(KernelGraph&              graph,
                       IndexComputeParams const& params,
                       const int                 target,
                       const int                 buffer,
                       const ContextPtr          context,
                       const CommandPtr          command)
        {
            // Check if target has a User coordinate
            auto user = graph.coordinates.get<User>(target);
            if(!user)
                return -1;

            AssertFatal(user->size, "Invalid User dimension: missing size.", ShowValue(target));

            auto toBytes = [&](Expression::ExpressionPtr expr) -> Expression::ExpressionPtr {
                uint numBits = DataTypeInfo::Get(params.valueType).elementBits;

                Log::debug("  toBytes: {}: numBits {}", toString(params.valueType), numBits);

                if(numBits % 8u == 0)
                    return expr * L(numBits / 8u);
                return (expr * L(numBits)) / L(8u);
            };

            auto bufferVarType = VariableType{DataType::None, PointerType::Buffer};
            auto bufferRegType = Register::Type::Scalar;

            // Create a buffer descriptor expression
            Expression::ExpressionPtr bufferExpr = L(rocRoller::Buffer{0, 0, 0, 0});
            Expression::ExpressionPtr basePointer
                = findArgumentByName(command, user->argumentName)->expression();

            if(user->offset)
                basePointer = basePointer + user->offset;

            bufferExpr = BufferDescriptor::SetBasePointer(bufferExpr, basePointer);
            bufferExpr = BufferDescriptor::SetOptions(bufferExpr,
                                                      BufferDescriptor::GetDefaultOptions(context));
            // TODO: Handle sizes larger than 32 bits
            bufferExpr = BufferDescriptor::SetSize(bufferExpr, toBytes(user->size));

            auto assignNode         = Assign{bufferRegType, bufferExpr};
            assignNode.variableType = bufferVarType;
            auto assignTag          = graph.control.addElement(assignNode);
            graph.mapper.connect(assignTag, buffer, NaryArgument::DEST);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::makeBuffer: assign {} expression {} to buffer {}",
                assignTag,
                toString(assignNode.expression),
                buffer);

            return assignTag;
        }

    } // namespace AssignIndexExpressionsDetail

    // Import detail namespace for internal use
    using namespace AssignIndexExpressionsDetail;

    /**
     * @brief Add index assignment operations.
     *
     * Adding index assignment operations to the control graph is done in
     * two phases: staging and committing.
     *
     * During the staging phase, we look at all load/store operations
     * in the control graph and "stage" the addition of index assignment
     * operations.  During the staging phase, we are able to detect
     * when two or more load/store operations would result in the same
     * chain of index assignments, and eliminate any redundancies.
     *
     * Usually index assignment operations come in sequential groups of
     * two or more operations, and hence we call them "index chains".
     *
     * During the commit stage, we add Assign operations to the
     * graphs, and add connections for load/store operations to the
     * newly created Base, Offset, and Stride elements of the
     * coordinate graph.
     *
     * For each candidate load/store operation:
     *
     * 1. The type of index chain is determined.
     *
     * 2. The required location of the index chain is determined.
     *
     * 3. The chain is staged.
     *
     * To determined where the chain should be placed:
     *
     * 1. Find all required coordinates by querying the Coordinate
     *    Transform graph.
     *
     * 2. If one-or-more Unroll dimension(s) are required:
     *
     *    a. Find SetCoordinate operations above the candidate and
     *       record the values of required Unroll dimensions.
     *
     *    b. Find the earliest matching set of SetCoordinate
     *       operations that are identical (ie, Unroll dimension and
     *       value) to the required Unroll dimensions.
     *
     *    c. The chain is added below the SetCoordinate operation from
     *       (b).
     *
     * 3. If a ForLoop dimension is required, find the containing
     *    ForLoop operation.  The chain is added above the ForLoop
     *    operation.
     *
     * 4. If both ForLoop and Unroll dimensions are required, the
     *    chain is added above the containing ForLoop.
     */
    struct AssignIndexer
    {
        AssignIndexer(ContextPtr context, CommandPtr command)
            : m_context(context)
            , m_command(command)
        {
        }

        void stageChain(KernelGraph const& graph,
                        int                target,
                        int                candidate,
                        int                location,
                        Graph::Direction   direction,
                        bool               isStorePartOfGlobalToLDSOp,
                        int                forLoop          = -1,
                        bool               replaceWithScope = true)
        {
            std::vector<int> specCoords;
            for(auto info :
                getRequiredCoordinatesInfo(candidate, location, graph, isStorePartOfGlobalToLDSOp))
            {
                specCoords.push_back(info.coord);
            }

            IndexChainSpec spec{target,
                                specCoords,
                                location,
                                direction,
                                forLoop,
                                replaceWithScope,
                                isStorePartOfGlobalToLDSOp};
            m_chains[spec].push_back(candidate);
        }

        void stage(KernelGraph const& kgraph, int candidate, bool isStorePartOfGlobalToLDSOp)
        {
            auto log = rocRoller::Log::getLogger();

            auto node = kgraph.control.getNode<Operation>(candidate);
            log->debug(
                "AssignIndexExpressions: processing candidate({}): {}", candidate, toString(node));

            auto [target, direction]
                = getOperationTarget(candidate, kgraph, isStorePartOfGlobalToLDSOp);
            auto [required, path]   = findRequiredCoordinates(target, direction, kgraph);
            auto forLoopCoordinates = filterCoordinates<ForLoop>(required, kgraph);
            auto unrollCoordinates  = filterCoordinates<Unroll>(required, kgraph);

            log->debug("  target: {}", target);
            for(auto r : required)
            {
                log->debug("  required: {}: {}", r, toString(kgraph.coordinates.getNode(r)));
            }

            auto maybeForLoop  = findContainingOperation<ForLoopOp>(candidate, kgraph);
            auto maybeScope    = findContainingOperation<Scope>(candidate, kgraph);
            auto hasForLoop    = !forLoopCoordinates.empty();
            auto hasUnroll     = !unrollCoordinates.empty();
            auto isUniformLoop = maybeForLoop && uniformForLoop(maybeForLoop, kgraph);

            // Check if this is a KLoop with a corresponding KLoopTail - if so, hoist to common ancestor
            if(maybeForLoop && hasForLoop && isUniformLoop)
            {
                auto maybeForLoopOp = kgraph.control.get<ForLoopOp>(*maybeForLoop);
                if(maybeForLoopOp && maybeForLoopOp->loopName == rocRoller::KLOOP)
                {
                    auto maybeKLoopTail = FindCorrespondingKLoopTail(kgraph, *maybeForLoop);
                    if(maybeKLoopTail)
                    {
                        auto maybeCommonAncestor
                            = FindCommonAncestorScope(kgraph, *maybeForLoop, *maybeKLoopTail);
                        if(maybeCommonAncestor)
                        {
                            log->debug(
                                "  staged as: KLoop with KLoopTail, hoisting to common ancestor {} "
                                "(KLoop={}, KLoopTail={})",
                                *maybeCommonAncestor,
                                *maybeForLoop,
                                *maybeKLoopTail);
                            // Stage the hoisted version at common ancestor; skip original KLoop location
                            stageChain(kgraph,
                                       target,
                                       candidate,
                                       *maybeCommonAncestor,
                                       GD::Upstream,
                                       isStorePartOfGlobalToLDSOp,
                                       *maybeForLoop, // Preserve forLoop for increment attachment
                                       true); // replaceWithScope shares scope at common ancestor
                            return;
                        }
                    }
                }
            }

            auto isReceiveTileLoop = false;
            if(maybeForLoop)
            {
                if(getForLoopName(kgraph, maybeForLoop.value()) == rocRoller::RECEIVE)
                    isReceiveTileLoop = true;
            }

            if(isReceiveTileLoop)
            {
                auto maybeTopOfLoop = findTopOfContainingOperation<ForLoopOp>(candidate, kgraph);
                log->debug("  staged as: isReceiveTileLoop, location {}, {}",
                           *maybeForLoop,
                           *maybeTopOfLoop);

                stageChain(kgraph,
                           target,
                           candidate,
                           *maybeTopOfLoop,
                           GD::Upstream,
                           isStorePartOfGlobalToLDSOp,
                           -1,
                           false);
                return;
            }

            if(hasForLoop && isUniformLoop)
            {
                log->debug("  staged as: hasForLoop and isUniformLoop, location {} forLoopOp {}",
                           *maybeForLoop,
                           *maybeForLoop);
                stageChain(kgraph,
                           target,
                           candidate,
                           *maybeForLoop,
                           GD::Upstream,
                           isStorePartOfGlobalToLDSOp,
                           *maybeForLoop);
                return;
            }

            // Prefetching
            // Find all children ForLoopOps. If any forLoopCoordinates are associated with the
            // children ForLoopOps, this is a prefetch.
            auto allChildForLoops
                = kgraph.control
                      .findNodes(
                          getTopSetCoordinate(kgraph, candidate),
                          [&](int tag) -> bool {
                              return isOperation<ForLoopOp>(kgraph.control.getElement(tag));
                          },
                          GD::Downstream)
                      .to<std::vector>();

            if(hasForLoop
               && std::any_of(allChildForLoops.begin(), allChildForLoops.end(), [&](auto tag) {
                      return forLoopCoordinates.count(kgraph.mapper.get<ForLoop>(tag)) > 0;
                  }))
            {
                log->debug("  staged as: hasForLoop and requiresDownstreamForLoop, location {} "
                           "forLoopOp {}",
                           *maybeForLoop,
                           *maybeForLoop);
                stageChain(kgraph,
                           target,
                           candidate,
                           *maybeScope,
                           GD::Upstream,
                           isStorePartOfGlobalToLDSOp,
                           -1);
                return;
            }

            if(maybeForLoop && !isUniformLoop && hasUnroll)
            {
                auto maybeTopOfLoop = findTopOfContainingOperation<ForLoopOp>(candidate, kgraph);
                log->debug("  staged as: hasForLoop and not isUniformLoop, location {}, {}",
                           *maybeForLoop,
                           *maybeTopOfLoop);
                stageChain(kgraph,
                           target,
                           candidate,
                           *maybeTopOfLoop,
                           GD::Upstream,
                           isStorePartOfGlobalToLDSOp,
                           -1,
                           false);
                return;
            }

            if(hasUnroll)
            {
                log->debug("  staged as: hasUnroll");

                auto kernel = *kgraph.control.roots().begin();
                stageChain(kgraph,
                           target,
                           candidate,
                           kernel,
                           GD::Downstream,
                           isStorePartOfGlobalToLDSOp,
                           -1);
                return;
            }

            if(isUniformLoop)
            {
                auto forLoop = *maybeForLoop;
                log->debug("  staged as: uniformForLoop, forLoopOp {}", forLoop);

                stageChain(kgraph,
                           target,
                           candidate,
                           forLoop,
                           GD::Upstream,
                           isStorePartOfGlobalToLDSOp,
                           forLoop);
                return;
            }

            log->debug("  staged as: immediate");
            stageChain(
                kgraph, target, candidate, candidate, GD::Upstream, isStorePartOfGlobalToLDSOp);
        }

        KernelGraph commit(KernelGraph const& original) const
        {
            auto               kgraph = original;
            std::map<int, int> scopes; // Maps location to actual scope node
            std::map<int, int>
                      serializationPoints; // Maps location to last chain bottom for serialization
            BufferMap bufferMap;
            BaseAddressMap baseAddressMap;

            // Build all chains and insert them into the graph
            for(auto const& [spec, candidates] : m_chains)
            {
                ExpressionPtr step = Expression::literal(1u);
                if(spec.forLoop > 0)
                {
                    auto [lhs, rhs] = getForLoopIncrement(kgraph, spec.forLoop);
                    step            = simplify(rhs);
                }

                // Use first candidate to compute indexes
                Log::debug("KernelGraph::AssignIndexExpressions()::commit({}) "
                           "isStorePartOfGlobalToLDSOp({}) "
                           "location={}",
                           candidates[0],
                           spec.isStorePartOfGlobalToLDSOp,
                           spec.location);

                auto chain = createIndexChain(kgraph, candidates[0], step, spec, bufferMap, baseAddressMap)

                if(spec.direction == GD::Downstream)
                {
                    // Add index assigns to an Initialize block below target
                    kgraph.control.addElement(Initialize(), {spec.location}, {chain.top});
                }
                else
                {
                    if(spec.replaceWithScope)
                    {
                        // Add index assigns in a Scope above target. Only the location
                        // is within the scope.
                        if(!scopes.contains(spec.location))
                        {
                            auto newScope = kgraph.control.addElement(Scope());
                            scopes[spec.location]
                                = replaceWith(kgraph, spec.location, newScope, false);
                            serializationPoints[spec.location] = scopes[spec.location];
                        }

                        auto scope = scopes[spec.location];
                        if(m_serializeAssigns)
                        {
                            auto insertionPoint = serializationPoints[spec.location];
                            auto isScope = kgraph.control.get<Scope>(insertionPoint).has_value();
                            kgraph.control.addElement(isScope ? ControlEdge(Body())
                                                              : ControlEdge(Sequence()),
                                                      {insertionPoint},
                                                      {chain.top});
                            kgraph.control.addElement(Sequence(), {chain.bottom}, {spec.location});
                            serializationPoints[spec.location] = chain.bottom;
                        }
                        else
                        {
                            kgraph.control.addElement(Body(), {scope}, {chain.top});
                            kgraph.control.addElement(Sequence(), {chain.bottom}, {spec.location});
                        }
                    }
                    else
                    {
                        // Add index assigns in a Scope above target. Everything underneath
                        // the location is within the scope.
                        if(!scopes.contains(spec.location))
                        {
                            scopes[spec.location] = kgraph.control.addElement(Scope());
                            insertWithBody(kgraph, spec.location, scopes[spec.location]);
                        }
                        insertBefore(kgraph, spec.location, chain.top, chain.bottom);
                    }
                }

                // If the chain has an update but no containing
                // ForLoopOp, it is from a pre-fetch
                if(chain.update > 0 && spec.forLoop < 0)
                {
                    kgraph.control.deleteElement(chain.update);
                    kgraph.mapper.purge(chain.update);
                    chain.update = -1;
                }

                // Attach increment to associate ForLoop
                if(chain.update > 0)
                {
                    kgraph.control.addElement(ForLoopIncrement(), {spec.forLoop}, {chain.update});
                }

                // Add deferred connections
                for(auto candidate : candidates)
                {
                    for(auto const& dc : chain.connections)
                    {
                        kgraph.mapper.connect(candidate, dc.coordinate, dc.connectionSpec);
                    }
                }

                // Now create Assign nodes for each placeholder in the chain
                for(auto const& nodeInfo : chain.nodeInfos)
                {
                    createAssignsForPlaceholder(kgraph, nodeInfo, m_context, m_command);
                }
            }

            return kgraph;
        }

    private:
        /**
         * @brief Create Assign nodes for a placeholder and replace the placeholder.
         */
        static void createAssignsForPlaceholder(KernelGraph&         kgraph,
                                                ChainNodeInfo const& nodeInfo,
                                                ContextPtr           context,
                                                CommandPtr           command)
        {
            int target = nodeInfo.target;

            // Determine if target is LDS
            auto maybeLDS = kgraph.coordinates.get<LDS>(target).has_value();
            if(maybeLDS)
            {
                // If target is LDS; it might be a duplicated LDS node.
                // For the purposes of computing indexes, use the parent LDS as the target instead.
                namespace CT = rocRoller::KernelGraph::CoordinateGraph;

                auto maybeParentLDS
                    = only(kgraph.coordinates.getOutputNodeIndices(target, CT::isEdge<Duplicate>));
                if(maybeParentLDS)
                    target = *maybeParentLDS;
            }
            maybeLDS = kgraph.coordinates.get<LDS>(target).has_value();

            // Determine if transposed
            auto isTransposed
                = kgraph.coordinates
                      .findNodes(target,
                                 [&](int tag) -> bool {
                                     auto maybeAdhoc = kgraph.coordinates.get<Adhoc>(tag);
                                     return maybeAdhoc
                                            && maybeAdhoc->name() == "Adhoc.transpose.simdsPerWave";
                                 })
                      .to<std::vector>()
                      .size()
                  == 1;

            // Build the params struct
            IndexComputeParams params{nodeInfo.forward,
                                      nodeInfo.isStorePartOfGlobalToLDSOp,
                                      nodeInfo.valueType,
                                      nodeInfo.offsetType,
                                      nodeInfo.strideType};

            // Build transformer at the placeholder location
            auto xform = kgraph.buildTransformer(nodeInfo.nopTag, rocRoller::IgnoreCache);

            // Set register coordinates
            auto const maybeForLoop = findContainingOperation<ForLoopOp>(nodeInfo.nopTag, kgraph);
            auto       direction
                = params.forward ? Graph::Direction::Upstream : Graph::Direction::Downstream;
            auto fullStop         = [&](int tag) { return tag == nodeInfo.increment; };
            auto [required, path] = findRequiredCoordinates(target, direction, fullStop, kgraph);

            auto isRegisterDim = [&maybeForLoop](auto dim) -> bool {
                using T = std::decay_t<decltype(dim)>;
                if(maybeForLoop)
                    return CIsAnyOf<T, Wavefront, Workitem, Workgroup, ForLoop>;
                else
                    return CIsAnyOf<T, Wavefront, Workitem, Workgroup>;
            };
            for(auto coord : required)
            {
                if(std::visit(isRegisterDim, kgraph.coordinates.getNode(coord)))
                {
                    auto registerType = Register::Type::Vector;
                    auto coordDF      = std::make_shared<Expression::Expression>(
                        Expression::DataFlowTag{coord, registerType, DataType::UInt32});
                    if(!xform.hasCoordinate(coord))
                        xform.setCoordinate(coord, coordDF);
                }
            }

            // Set remaining coordinates to 0
            for(auto coord : required)
                if((coord != nodeInfo.increment) && (!xform.hasCoordinate(coord)))
                    xform.setCoordinate(coord, L(0u));

            // Set the increment coordinate to zero if it doesn't already have a value
            bool initializeIncrement
                = !xform.hasPath({target}, direction == Graph::Direction::Upstream);
            if(initializeIncrement)
            {
                xform.setCoordinate(nodeInfo.increment, L(0u));
            }

            int assignStrideTag = -1, assignBaseTag = -1, assignBufferTag = -1;

            if(nodeInfo.baseOffset < 0 && nodeInfo.offset > 0)
            {
                assignBaseTag = makeAssignBase(kgraph,
                                               params,
                                               target,
                                               nodeInfo.offset,
                                               maybeLDS,
                                               isTransposed,
                                               context,
                                               xform);
            }

            if(nodeInfo.stride > 0)
            {
                assignStrideTag = makeAssignStride(kgraph,
                                                   params,
                                                   target,
                                                   nodeInfo.stride,
                                                   nodeInfo.increment,
                                                   maybeLDS,
                                                   isTransposed,
                                                   context,
                                                   xform);
            }

            if(nodeInfo.buffer > 0)
            {
                assignBufferTag
                    = makeBuffer(kgraph, params, target, nodeInfo.buffer, context, command);
            }

            // Insert Assign nodes after the NOP placeholder
            if(assignBufferTag != -1)
                insertAfter(kgraph, nodeInfo.nopTag, assignBufferTag, assignBufferTag);
            if(assignStrideTag != -1)
                insertAfter(kgraph, nodeInfo.nopTag, assignStrideTag, assignStrideTag);
            if(assignBaseTag != -1)
                insertAfter(kgraph, nodeInfo.nopTag, assignBaseTag, assignBaseTag);
        }

    private:
        std::map<IndexChainSpec, std::vector<int>> m_chains;

        bool       m_serializeAssigns = true;
        ContextPtr m_context;
        CommandPtr m_command;
    };

    KernelGraph AssignIndexExpressions::apply(KernelGraph const& original)
    {
        AssignIndexer indexer(m_context, m_command);

        for(auto candidate :
            findIndexAssignmentCandidates(original, *original.control.roots().begin()))
        {
            // Global to LDS ops have two sets of coordinates for the load and store parts
            indexer.stage(original, candidate, false);
            if(isGlobalToLDSOp(original, candidate))
                indexer.stage(original, candidate, /*isStorePartOfGlobalToLDSOp=*/true);
        }

        return indexer.commit(original);
    }
}
