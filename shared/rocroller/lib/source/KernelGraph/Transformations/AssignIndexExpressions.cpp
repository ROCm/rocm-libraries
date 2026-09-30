// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <variant>

#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/TensorDataMover.hpp>
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

        // When >= 0, this unroll coordinate's value is incorporated into
        // the base address (rather than appearing as a stride). This is
        // needed when a non-affine transform (e.g. LDS bank swizzle)
        // depends on the unroll position -- the base must be recomputed
        // per unroll value, producing separate chains.
        int inlineUnrollValue = -1; // literal value for this chain (-1 = unused)
        int inlineUnrollCoord = -1; // coordinate tag of the unroll dimension
    };

    bool operator<(const IndexChainSpec& a, const IndexChainSpec& b)
    {
        return std::tie(a.target,
                        a.coords,
                        a.location,
                        a.direction,
                        a.forLoop,
                        a.replaceWithScope,
                        a.isStorePartOfGlobalToLDSOp,
                        a.inlineUnrollValue,
                        a.inlineUnrollCoord)
               < std::tie(b.target,
                          b.coords,
                          b.location,
                          b.direction,
                          b.forLoop,
                          b.replaceWithScope,
                          b.isStorePartOfGlobalToLDSOp,
                          b.inlineUnrollValue,
                          b.inlineUnrollCoord);
    }

    /**
     * @brief Information needed to create Assign nodes for a single dimension.
     */
    struct ChainNodeInfo
    {
        int      nopTag      = -1; // Placeholder NOP node
        int      target      = -1;
        int      increment   = -1;
        int      baseOffset  = -1; // The base offset coordinate (not the offset of this node)
        int      offset      = -1;
        int      stride      = -1;
        int      buffer      = -1;
        int      baseAddress = -1;
        int      tdm         = -1;
        bool     forward     = false;
        DataType valueType   = DataType::Count;
        DataType offsetType  = DataType::Count;
        DataType strideType  = DataType::Count;
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
    using TDMMap         = std::map<int, int>;

    /**
     * @brief Describes where and how to place an index chain.
     */
    struct ChainPlacement
    {
        int              location; // Where to insert the chain
        Graph::Direction direction; // Upstream or Downstream of location
        int              forLoop          = -1; // ForLoop to attach increment to (-1 if none)
        bool             replaceWithScope = true; // Whether to wrap in a Scope
    };

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
     * @brief Return existing TDM for load/stores from/to `dst`.
     *
     * Returns -1 if the operation doesn't need a TDM descriptor.
     *
     * If a TDM edge doesn't already exist, we create a new
     * Workgroup coordinate and attach it with a TDM edge to the
     * `dst`.
     */
    int getTDM(KernelGraph& graph, int opTag, int dst, TDMMap& tdmMap, bool isStorePartOfTDMToLDSOp)
    {
        auto op = graph.control.getElement(opTag);
        if(not isOperation<LoadTiledTDMToLDS>(op) || isStorePartOfTDMToLDSOp)
            return -1;

        if(!tdmMap.contains(dst))
        {
            auto wg     = graph.coordinates.addElement(Workgroup());
            tdmMap[dst] = graph.coordinates.addElement(TDM(), {wg}, {dst});
        }

        return tdmMap[dst];
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
                                       int          tdm,
                                       bool         forward,
                                       DataType     valueType,
                                       DataType     offsetType,
                                       DataType     strideType,
                                       bool         isStorePartOfGlobalToLDSOp)
    {
        // Create a NOP placeholder that will be replaced with Assign nodes later
        auto nopTag = graph.control.addElement(NOP());

        rocRoller::Log::getLogger()->debug(
            "KernelGraph::makeIndexPlaceholder: nop {} {}/{} {}; {}/{}/{} {}/{} {}",
            nopTag,
            target,
            increment,
            forward,
            base,
            offset,
            stride,
            buffer,
            baseAddress,
            tdm);

        return ChainNodeInfo{nopTag,
                             target,
                             increment,
                             base,
                             offset,
                             stride,
                             buffer,
                             baseAddress,
                             tdm,
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
     * @brief Collect ForLoop coordinate if location is a ForLoop.
     */
    std::optional<int> getForLoopCoordinate(int                            location,
                                            Graph::Direction               direction,
                                            std::unordered_set<int> const& path,
                                            KernelGraph const&             graph)
    {
        if(location == -1)
            return std::nullopt;

        auto maybeForLoop = graph.control.get<ForLoopOp>(location);
        if(!maybeForLoop)
            return std::nullopt;

        auto forLoopCoord = followIdentify(graph.mapper.get<ForLoop>(location), graph);
        auto coord        = getNeighbourNodeInPath(forLoopCoord, direction, path, graph);

        return (coord != -1) ? std::make_optional(coord) : std::nullopt;
    }

    /**
     * @brief Collect Unroll coordinates that are in the transform path.
     */
    template <typename RequiredType, typename ForLoopSetType>
    std::vector<int> getUnrollCoordinates(RequiredType const&            required,
                                          ForLoopSetType const&          forLoopCoords,
                                          std::vector<int> const&        codegen,
                                          Graph::Direction               direction,
                                          std::unordered_set<int> const& path,
                                          KernelGraph const&             graph)
    {
        std::vector<int> result;
        auto             unrolls = filterCoordinates<Unroll>(required, graph);

        for(auto unroll : unrolls)
        {
            // In StreamK, Unroll coordinates are connected via Identify edges.
            // followIdentify resolves these chains (or returns the original if none).
            auto unrollTarget = followIdentify(unroll, graph);
            // Find a neighbour of unrollTarget that's actually in the path
            auto coord = getNeighbourNodeInPath(unrollTarget, direction, path, graph);

            // Skip if it's a ForLoop coord, already added, or a codegen coord
            if(coord == -1 || forLoopCoords.contains(coord))
                continue;
            if(std::find(codegen.begin(), codegen.end(), coord) != codegen.end())
                continue;
            if(std::find(result.begin(), result.end(), coord) != result.end())
                continue;

            result.push_back(coord);
        }

        return result;
    }

    /**
     * @brief Get list of required coordinates, and how they relate to each other.
     *
     * Builds a list of coordinates, slow-to-fast, that need offset/strides
     * for operation `op`. The ordering is:
     * 1. ForLoop coordinate (slowest, if present)
     * 2. Unroll coordinates
     * 3. Code-gen coordinates (fastest)
     */
    std::vector<RequiredCoordinateInfo> getRequiredCoordinatesInfo(int                op,
                                                                   int                location,
                                                                   KernelGraph const& graph,
                                                                   bool isStorePartOfGlobalToLDSOp)
    {
        auto [target, direction] = getOperationTarget(op, graph, isStorePartOfGlobalToLDSOp);
        auto [required, path]    = findRequiredCoordinates(target, direction, graph);
        auto codegen = getCodeGeneratorCoordinates(graph, op, isStorePartOfGlobalToLDSOp);

        // Build ordered list: ForLoop -> Unroll -> CodeGen (slow to fast)
        std::vector<int> ordered;
        std::set<int>    forLoopCoords;
        std::set<int>    unrollCoords;

        // 1. ForLoop coordinate (slowest)
        if(auto coord = getForLoopCoordinate(location, direction, path, graph))
        {
            ordered.push_back(*coord);
            forLoopCoords.insert(*coord);
        }

        // 2. Unroll coordinates
        for(auto coord :
            getUnrollCoordinates(required, forLoopCoords, codegen, direction, path, graph))
        {
            ordered.push_back(coord);
            unrollCoords.insert(coord);
        }

        // 3. Code-gen coordinates (fastest)
        for(auto coord : codegen)
            ordered.push_back(coord);

        // Build result with base pointers (each dimension references the previous)
        std::vector<RequiredCoordinateInfo> result;
        int                                 base = -1;

        for(auto coord : ordered)
        {
            // Compute sub-dimension index for code-gen coordinates
            // TODO: Slow to fast; lift this from Tensor directly
            int  sdim = -1;
            auto it   = std::find(codegen.begin(), codegen.end(), coord);
            if(it != codegen.end())
            {
                sdim = std::distance(codegen.begin(), it);
                if(isStorePartOfGlobalToLDSOp)
                    sdim += static_cast<int>(ordered.size());
            }

            if(unrollCoords.contains(coord))
            {
                // Unroll dimensions don't chain to previous dimensions
                result.push_back({coord, -1, -1, true, false});
            }
            else
            {
                bool needsUpdate = forLoopCoords.contains(coord) && uniformForLoop(location, graph);
                result.push_back({coord, base, sdim, false, needsUpdate});
                base = coord;
            }
        }

        return result;
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
                                   std::vector<DeferredConnection>& connections,
                                   int                              inlineUnrollCoord = -1)
    {
        auto [target, direction]
            = getOperationTarget(candidate, kgraph, isStorePartOfGlobalToLDSOp);
        auto [required, path] = findRequiredCoordinates(target, direction, kgraph);
        auto unrolls          = filterCoordinates<Unroll>(required, kgraph);

        for(auto const& unroll : unrolls)
        {
            // Skip the unroll coordinate whose value is incorporated into
            // the base address -- no stride needed for it.
            if(inlineUnrollCoord >= 0 && unroll == inlineUnrollCoord)
                continue;
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
     * @brief Create coordinate edges (Offset/Stride/Buffer) for a single dimension.
     */
    struct CoordinateEdges
    {
        int offset      = -1;
        int stride      = -1;
        int buffer      = -1;
        int baseAddress = -1;
        int tdm         = -1;
    };

    CoordinateEdges createCoordinateEdges(KernelGraph&     graph,
                                          int              op,
                                          int              target,
                                          int              coord,
                                          bool             isBase,
                                          bool             isUnroll,
                                          Graph::Direction direction,
                                          bool             isStorePartOfGlobalToLDSOp,
                                          BufferMap&       bufferMap,
                                          BaseAddressMap&  baseAddressMap,
                                          TDMMap&          tdmMap)
    {
        CoordinateEdges edges;

        auto inCoord  = target;
        auto outCoord = coord;
        if(direction == Graph::Direction::Upstream)
            std::swap(inCoord, outCoord);

        // Unroll dimensions don't need offset edges
        if(!isUnroll)
            edges.offset = graph.coordinates.addElement(Offset(), {inCoord}, {outCoord});

        edges.stride = graph.coordinates.addElement(Stride(), {inCoord}, {outCoord});

        // Only the base dimension (slowest moving) gets buffer/baseAddress/tdm
        if(isBase && edges.offset != -1)
        {
            bool isDirect2LDS = isOperation<LoadTileDirect2LDS>(graph.control.getElement(op));
            bool isStorePartOfDirect2LDSOp = (isDirect2LDS && isStorePartOfGlobalToLDSOp);
            bool isTDMToLDS = isOperation<LoadTiledTDMToLDS>(graph.control.getElement(op));
            bool isStorePartOfTDMToLDSOp = (isTDMToLDS && isStorePartOfGlobalToLDSOp);

            edges.buffer      = getBuffer(graph, op, target, bufferMap, isStorePartOfDirect2LDSOp);
            edges.baseAddress = getBaseAddress(graph, op, target, baseAddressMap);
            edges.tdm         = getTDM(graph, op, target, tdmMap, isStorePartOfGlobalToLDSOp);
            if(edges.tdm != -1)
            {
                edges.baseAddress = -1;
                edges.buffer      = -1;
            }
        }

        return edges;
    }

    /**
     * @brief Create the update (increment) operation for a ForLoop.
     */
    int createUpdateOperation(
        KernelGraph& graph, int offset, int stride, DataType offsetDataType, ExpressionPtr step)
    {
        auto offsetExpr = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{offset, Register::Type::Vector, offsetDataType});
        auto strideExpr = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{stride, Register::Type::Scalar, DataType::UInt64});

        auto incrementExpr
            = (step == nullptr) ? offsetExpr + strideExpr : offsetExpr + step * strideExpr;

        auto update = graph.control.addElement(
            Assign{Register::Type::Vector, convert(offsetDataType, incrementExpr)});
        graph.mapper.connect(update, offset, NaryArgument::DEST);

        return update;
    }

    /**
     * @brief Create an index assignment chain for `op`.
     */
    IndexChain createIndexChain(KernelGraph&          graph,
                                int                   op,
                                ExpressionPtr         step,
                                IndexChainSpec const& spec,
                                BufferMap&            bufferMap,
                                BaseAddressMap&       baseAddressMap,
                                TDMMap&               tdmMap)
    {
        Log::debug("KernelGraph::AssignIndexExpressions::createIndexChain(): op {} location {}",
                   op,
                   spec.location);

        auto dtype               = getDataType(graph.control.getNode(op));
        auto [target, direction] = getOperationTarget(op, graph, spec.isStorePartOfGlobalToLDSOp);

        int                             update = -1;
        std::vector<int>                chain;
        std::vector<ChainNodeInfo>      nodeInfos;
        std::vector<DeferredConnection> connections;
        std::map<int, int>              offsetOfCoord;
        std::vector<int>                strideCoords;

        int  locationForCoordInfo = (spec.forLoop > 0) ? spec.forLoop : spec.location;
        auto requiredCoords       = getRequiredCoordinatesInfo(
            op, locationForCoordInfo, graph, spec.isStorePartOfGlobalToLDSOp);

        for(auto const& info : requiredCoords)
        {
            // Create coordinate edges for this dimension
            // I have no base above me, therefore I'm the base
            bool isBase = (info.base == -1);
            auto edges  = createCoordinateEdges(graph,
                                               op,
                                               target,
                                               info.coord,
                                               isBase,
                                               info.isUnroll,
                                               direction,
                                               spec.isStorePartOfGlobalToLDSOp,
                                               bufferMap,
                                               baseAddressMap,
                                               tdmMap);

            offsetOfCoord[info.coord] = edges.offset;
            int base                  = isBase ? -1 : offsetOfCoord.at(info.base);

            // For future: choose type based on buffer or non-buffer
            // Determine data types
            auto offsetDataType = getOffsetDataType(op, graph, spec.isStorePartOfGlobalToLDSOp);
            auto strideDataType = DataType::UInt64;
            if(info.isUnroll)
            {
                offsetDataType = DataType::Int64;
                strideDataType = DataType::Int64;
            }

            // Create placeholder NOP for later Assign creation
            auto nodeInfo = makeIndexPlaceholder(graph,
                                                 target,
                                                 info.coord,
                                                 base,
                                                 edges.offset,
                                                 edges.stride,
                                                 edges.buffer,
                                                 edges.baseAddress,
                                                 edges.tdm,
                                                 direction == Graph::Direction::Upstream,
                                                 dtype,
                                                 offsetDataType,
                                                 strideDataType,
                                                 spec.isStorePartOfGlobalToLDSOp);
            chain.push_back(nodeInfo.nopTag);
            nodeInfos.push_back(nodeInfo);

            // Register connections for allocation and lifetime tracking
            if(edges.offset != -1)
                connections.push_back(DC<Offset>(edges.offset, info.sdim));
            if(edges.stride != -1)
                connections.push_back(DC<Stride>(edges.stride, info.sdim));
            if(edges.buffer != -1)
                connections.push_back(DC<Buffer>(edges.buffer));
            if(edges.baseAddress != -1)
                connections.push_back(DC<BaseAddress>(edges.baseAddress));
            if(edges.tdm != -1)
                connections.push_back(DC<TDM>(edges.tdm));
            if(base != -1)
                connections.push_back(
                    makeConnection<Offset, Connections::BaseOffset>(base, info.sdim));

            if(edges.stride != -1)
                strideCoords.push_back(edges.stride);

            // Create update operation if this dimension needs loop increment
            if(info.needsUpdate)
                update = createUpdateOperation(
                    graph, edges.offset, edges.stride, offsetDataType, step);
        }

        addUnrollStrideConnection(graph,
                                  op,
                                  spec.isStorePartOfGlobalToLDSOp,
                                  strideCoords,
                                  connections,
                                  spec.inlineUnrollCoord);

        // Link chain nodes with Sequence edges
        for(size_t i = 1; i < chain.size(); ++i)
            graph.control.addElement(Sequence(), {chain[i - 1]}, {chain[i]});

        return {chain.front(), chain.back(), nodeInfos, connections, update};
    }

    namespace AssignIndexExpressionsDetail
    {
        using namespace CoordinateGraph;
        using namespace ControlGraph;

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

        std::optional<int> FindCommonAncestorScope(KernelGraph const& kgraph, int nodeA, int nodeB)
        {
            auto ancestorsA = kgraph.control.breadthFirstVisit(nodeA, Graph::Direction::Upstream)
                                  .to<std::set>();

            for(auto node : kgraph.control.breadthFirstVisit(nodeB, Graph::Direction::Upstream))
            {
                if(ancestorsA.contains(node) && kgraph.control.get<Scope>(node))
                    return node;
            }

            return std::nullopt;
        }

        std::pair<uint, uint>
            GetElementBlockValues(KernelGraph const& graph, int target, const bool isTransposed)
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
                    if(memType == MemoryType::WAVE || memType == MemoryType::WAVE_SWIZZLE
                       || memType == MemoryType::WAVE_FROM_GLOBAL)
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
                    || macTile.memoryType == MemoryType::WAVE_SWIZZLE
                    || macTile.memoryType == MemoryType::WAVE_FROM_GLOBAL)
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

        std::pair<int, int> getUserSubDimensions(KernelGraph const& graph, int userTag)
        {

            auto isSubDimension
                = [](Dimension const& node) { return std::holds_alternative<SubDimension>(node); };
            auto truePred = [](auto const& node) { return true; };

            auto subDimTags
                = Graph::reachableNodes<Graph::Direction::Downstream>(
                      graph.coordinates, userTag, isSubDimension, isEdge<Split>, truePred)
                      .to<std::vector>();

            AssertFatal(subDimTags.size() == 2,
                        fmt::format("Expected User({}) to be split into only two "
                                    "SubDimension coordinates but found {}.",
                                    userTag,
                                    subDimTags.size()));

            return {subDimTags[0], subDimTags[1]};
        }

        int MakeAssignBase(KernelGraph&              graph,
                           IndexComputeParams const& params,
                           int                       target,
                           int                       offset,
                           int                       baseAddress,
                           int                       tdm,
                           bool                      isLDS,
                           bool                      isTransposed,
                           ContextPtr                context,
                           CommandPtr                command,
                           Transformer&              coords)
        {
            // Compute the base index expression
            auto indexExpr
                = params.forward ? coords.forward({target})[0] : coords.reverse({target})[0];

            // Compute LDS padding for transposed FP6 loads
            Expression::ExpressionPtr paddingBytes = L(0u);
            {
                auto const& typeInfo = DataTypeInfo::Get(params.valueType);
                auto        numBits  = DataTypeInfo::Get(typeInfo.segmentVariableType).elementBits;
                auto const& arch     = context->targetArchitecture();

                bool needsPadding
                    = (numBits == 6) && isTransposed && isLDS
                      && arch.HasCapability(GPUCapability::DSReadTransposeB6PaddingBytes);

                if(needsPadding)
                {
                    uint elementsPerTrLoad = bitsPerTransposeLoad(arch, numBits) / numBits;
                    auto extraLdsBytes     = extraLDSBytesPerElementBlock(arch, numBits);
                    paddingBytes           = indexExpr / L(elementsPerTrLoad) * L(extraLdsBytes);
                }
            }

            // Build the offset expression: bytes + padding
            auto expr = ToBytes(indexExpr, params.valueType) + paddingBytes;

            // For Direct2LDS stores, convert to scalar
            if(params.isStorePartOfGlobalToLDS)
                expr = std::make_shared<Expression::Expression>(Expression::ToScalar{expr});

            // Add base address offset for WAVE_FROM_GLOBAL or TDM operations
            if(baseAddress > 0 or tdm > 0)
            {
                const auto edgeTag  = (baseAddress > 0) ? baseAddress : tdm;
                const auto edgeName = (baseAddress > 0) ? "BaseAddress" : "TDM";
                namespace CG        = KernelGraph::CoordinateGraph;
                auto userTag
                    = only(graph.coordinates.getNeighbours<Graph::Direction::Downstream>(edgeTag))
                          .value();
                AssertFatal(
                    userTag > 0,
                    fmt::format("Could not find User connected to {}({})", edgeName, edgeTag));
                auto user = graph.coordinates.getNode<CG::User>(userTag);

                AssertFatal(command, "Expected command pointer but got nullptr");
                auto userAsCmdArg = findArgumentByName(command, user.argumentName);
                AssertFatal(userAsCmdArg,
                            "Argument for base address not found.",
                            ShowValue(user.argumentName));
                expr = expr + convert(params.offsetType, userAsCmdArg->expression());
            }

            // Create the Assign node
            auto offsetRegisterType
                = params.isStorePartOfGlobalToLDS ? Register::Type::Scalar : Register::Type::Vector;
            auto assignNode         = Assign{offsetRegisterType, convert(params.offsetType, expr)};
            assignNode.variableType = params.offsetType;
            auto assignTag          = graph.control.addElement(assignNode);
            graph.mapper.connect(assignTag, offset, NaryArgument::DEST);

            Log::debug("KernelGraph::makeAssignBase: assign {} expression {} to offset {}",
                       assignTag,
                       toString(assignNode.expression),
                       offset);

            return assignTag;
        }

        int MakeAssignStride(KernelGraph&              graph,
                             IndexComputeParams const& params,
                             int                       target,
                             int                       stride,
                             int                       increment,
                             bool                      isLDS,
                             bool                      isTransposed,
                             ContextPtr                context,
                             Transformer&              coords)
        {
            // Compute the stride expression for one element
            auto indexExpr = params.forward ? coords.forwardStride(increment, L(1), {target})[0]
                                            : coords.reverseStride(increment, L(1), {target})[0];

            // Check if this is a unit stride (stride of 1)
            bool unitStride = false;
            if(Expression::evaluationTimes(indexExpr)[Expression::EvaluationTime::Translate])
                unitStride = (getUnsignedInt(evaluate(indexExpr)) == 1u);

            // Get type info for padding calculations
            auto const& typeInfo = DataTypeInfo::Get(params.valueType);
            auto        numBits  = DataTypeInfo::Get(typeInfo.segmentVariableType).elementBits;
            auto const& arch     = context->targetArchitecture();

            // Initialize stride attributes for sub-dword types
            uint                      elementBlockSize               = 0;
            Expression::ExpressionPtr elementBlockStride             = L(0u);
            Expression::ExpressionPtr trLoadPairStride               = L(0u);
            Expression::ExpressionPtr elementBlockStridePaddingBytes = L(0u);
            Expression::ExpressionPtr trLoadPairStridePaddingBytes   = L(0u);
            Expression::ExpressionPtr indexExprPaddingBytes          = L(0u);

            // Sub-dword types (FP16, FP8, FP6, FP4) need special stride handling
            bool isSubDwordType = (numBits == 16 || numBits == 8 || numBits == 6 || numBits == 4);
            if(isSubDwordType)
            {
                auto [elementBlockNumber, elementBlockIndex]
                    = GetElementBlockValues(graph, target, isTransposed);
                elementBlockSize = elementBlockIndex;

                // Adjust block size for transposed loads
                if(isTransposed)
                {
                    // See addLoadWaveTileCTF8F6F4 in LowerTile.cpp
                    uint const wfs = arch.GetCapability(GPUCapability::DefaultWavefrontSize);
                    uint const numVBlocks
                        = (wfs == 64) ? (numBits == 8 ? 2 : 1) : (numBits == 8 ? 4 : 2);
                    elementBlockSize = (elementBlockNumber / numVBlocks) * elementBlockSize;
                }
                AssertFatal(elementBlockSize > 0, "Invalid elementBlockSize: ", elementBlockSize);

                // FP6 transposed loads need padding after every 16 elements
                bool needsPadding
                    = (numBits == 6) && isTransposed
                      && arch.HasCapability(GPUCapability::DSReadTransposeB6PaddingBytes);

                if(needsPadding)
                    elementBlockSize = 16;

                // Compute strides for element blocks
                elementBlockStride
                    = params.forward
                          ? coords.forwardStride(increment, L(elementBlockSize), {target})[0]
                          : coords.reverseStride(increment, L(elementBlockSize), {target})[0];

                trLoadPairStride
                    = params.forward
                          ? coords.forwardStride(increment, L(elementBlockIndex), {target})[0]
                          : coords.reverseStride(increment, L(elementBlockIndex), {target})[0];

                // Add padding bytes for LDS operations
                if(needsPadding && isLDS)
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

            // Create the Assign node with stride attributes
            auto assignNode         = Assign{Register::Type::Vector,
                                     ToBytes(indexExpr, params.valueType) + indexExprPaddingBytes};
            assignNode.variableType = params.strideType;
            assignNode.strideExpressionAttributes
                = {params.strideType,
                   unitStride,
                   elementBlockSize,
                   ToBytes(elementBlockStride, params.valueType) + elementBlockStridePaddingBytes,
                   ToBytes(trLoadPairStride, params.valueType) + trLoadPairStridePaddingBytes};

            auto assignTag = graph.control.addElement(assignNode);
            graph.mapper.connect(assignTag, stride, NaryArgument::DEST);

            Log::debug("KernelGraph::makeAssignStride: assign {} expression {} to stride {}",
                       assignTag,
                       toString(assignNode.expression),
                       stride);
            return assignTag;
        }

        int MakeBuffer(KernelGraph&              graph,
                       IndexComputeParams const& params,
                       int                       target,
                       int                       buffer,
                       ContextPtr                context,
                       CommandPtr                command)
        {
            // Check if target has a User coordinate
            auto user = graph.coordinates.get<User>(target);
            if(!user)
                return -1;

            AssertFatal(user->size, "Invalid User dimension: missing size.", ShowValue(target));
            auto bufferSize = ToBytes(user->size, params.valueType);
            Log::debug("KernelGraph::makeBuffer: using User.size for user {}", target);

            // Get the base pointer from command arguments
            auto arg = findArgumentByName(command, user->argumentName);
            AssertFatal(arg,
                        "Argument for buffer descriptor base pointer not found.",
                        ShowValue(user->argumentName));
            Expression::ExpressionPtr basePointer = arg->expression();
            if(user->offset)
                basePointer = basePointer + user->offset;

            // Build the buffer descriptor
            Expression::ExpressionPtr bufferExpr = L(rocRoller::Buffer{0, 0, 0, 0});
            bufferExpr = BufferDescriptor::SetBasePointer(bufferExpr, basePointer, context);
            bufferExpr = BufferDescriptor::SetOptions(bufferExpr,
                                                      BufferDescriptor::GetDefaultOptions(context));
            bufferExpr = BufferDescriptor::SetSize(bufferExpr, bufferSize, context);

            // Create the Assign node
            auto bufferVarType      = VariableType{DataType::None, PointerType::Buffer};
            auto assignNode         = Assign{Register::Type::Scalar, bufferExpr};
            assignNode.variableType = bufferVarType;
            auto assignTag          = graph.control.addElement(assignNode);
            graph.mapper.connect(assignTag, buffer, NaryArgument::DEST);

            Log::debug("KernelGraph::makeBuffer: assign {} expression {} to buffer {}",
                       assignTag,
                       toString(assignNode.expression),
                       buffer);

            return assignTag;
        }

        std::pair<int, int> GetInlineUnrollInfo(KernelGraph const& kgraph, int candidate)
        {
            if(!kgraph.control.get<LoadLDSTile>(candidate))
                return {-1, -1};

            auto macTileTag = kgraph.mapper.get<MacroTile>(candidate);
            auto macTile    = kgraph.coordinates.getNode<MacroTile>(macTileTag);
            auto dataType   = getDataType(kgraph.control.getNode(candidate));

            if(isScaleType(dataType))
                return {-1, -1};

            int kSubdim = -1;
            if(macTile.layoutType == LayoutType::MATRIX_A)
                kSubdim = 1;
            else if(macTile.layoutType == LayoutType::MATRIX_B)
                kSubdim = 0;
            else
                return {-1, -1};

            auto [target, direction]
                = getOperationTarget(candidate, kgraph, /*isStorePartOfGlobalToLDSOp=*/false);
            auto [required, path] = findRequiredCoordinates(target, direction, kgraph);
            auto unrolls          = filterCoordinates<Unroll>(required, kgraph);

            int kUnrollCoord = -1;
            for(auto const& unroll : unrolls)
            {
                auto sdim = kgraph.mapper.getConnectionSubdimension(candidate, unroll);
                if(sdim == kSubdim)
                {
                    kUnrollCoord = unroll;
                    break;
                }
            }

            if(kUnrollCoord < 0)
                return {-1, -1};

            // PairSwap and Rotate edges are used for LDS bank swizzling.
            // They are non-affine with respect to the K dimension: the
            // column permutation depends on the row index, so the unroll
            // value cannot be applied as a simple stride and must be
            // folded into the base address (done elsewhere).
            bool hasLDSSwizzle = false;
            for(auto nodeTag : path)
            {
                for(auto edgeTag :
                    kgraph.coordinates.getNeighbours(nodeTag, Graph::opposite(direction)))
                {
                    if(kgraph.coordinates.get<PairSwap>(edgeTag).has_value()
                       || kgraph.coordinates.get<Rotate>(edgeTag).has_value())
                    {
                        hasLDSSwizzle = true;
                        break;
                    }
                }
                if(hasLDSSwizzle)
                    break;
            }

            if(!hasLDSSwizzle)
                return {-1, -1};

            int current = candidate;
            while(true)
            {
                auto parents = kgraph.control.getInputNodeIndices<Body>(current).to<std::vector>();
                if(parents.empty())
                    break;

                AssertFatal(parents.size() == 1,
                            "GetInlineUnrollInfo: expected single Body parent",
                            ShowValue(current),
                            ShowValue(parents.size()));

                current            = parents[0];
                auto maybeSetCoord = kgraph.control.get<SetCoordinate>(current);
                if(!maybeSetCoord)
                    continue;

                auto coordTag = kgraph.mapper.get<Unroll>(current);
                if(coordTag != kUnrollCoord)
                    continue;

                auto valueExpr = maybeSetCoord->value;
                if(!evaluationTimes(valueExpr)[EvaluationTime::Translate])
                    break;

                auto value = static_cast<int>(getUnsignedInt(evaluate(valueExpr)));
                return {kUnrollCoord, value};
            }

            return {-1, -1};
        }

        int MakeTDM(KernelGraph&              graph,
                    IndexComputeParams const& params,
                    int                       target,
                    int                       tdm,
                    ContextPtr                context,
                    CommandPtr                command)
        {
            // Check if target has a User coordinate
            auto user = graph.coordinates.get<User>(target);
            if(!user)
                return -1;

            AssertFatal(user->size, "Invalid User dimension: missing size.", ShowValue(target));

            // Build the TDM descriptor
            Expression::ExpressionPtr tdmExpr = L(rocRoller::TDM{});
            tdmExpr                           = TDMDescriptor::SetDefaults(tdmExpr, context);

            auto [subDim0Tag, subDim1Tag] = getUserSubDimensions(graph, target);
            auto subDim0                  = graph.coordinates.getNode<SubDimension>(subDim0Tag);
            auto subDim1                  = graph.coordinates.getNode<SubDimension>(subDim1Tag);

            { // Detect if using swapped layout and ensure that subDim0 always is the fastest moving
                auto isLoadTiledTDMToLDS = [&](ControlToCoordinateMapper::Connection conn) {
                    return std::holds_alternative<LoadTiledTDMToLDS>(
                        graph.control.getNode(conn.control));
                };
                const auto loadTiledToTDMTag
                    = *filter(isLoadTiledTDMToLDS, graph.mapper.getCoordinateConnections(tdm))
                           .map([&](auto const& conn) -> int { return conn.control; })
                           .take(1)
                           .only();

                const auto [elementNumberTag, elementNumber]
                    = graph.getDimension<ElementNumber>(loadTiledToTDMTag, 0);

                if(isSwappedLayout(graph, elementNumberTag, elementNumber))
                {
                    std::swap(subDim0, subDim1);
                }
            }
            Log::debug(
                fmt::format("  SubDim0: {} SubDim1: {}", subDim0.toString(), subDim1.toString()));

            auto subDim0SizeAsUInt32Expr = Expression::convert(DataType::UInt32, subDim0.size);
            auto subDim1SizeAsUInt32Expr = Expression::convert(DataType::UInt32, subDim1.size);

            tdmExpr
                = TDMDescriptor::SetTensorDims(tdmExpr,
                                               ToBytes(subDim1SizeAsUInt32Expr, params.valueType),
                                               subDim0SizeAsUInt32Expr);
            tdmExpr = TDMDescriptor::SetTensorStrides(tdmExpr,
                                                      ToBytes(subDim0.stride, params.valueType),
                                                      ToBytes(subDim1.stride, params.valueType));

            // Create the Assign node
            auto assignNode         = Assign{Register::Type::Scalar, tdmExpr};
            assignNode.variableType = VariableType{DataType::None, PointerType::TDM};
            auto assignTag          = graph.control.addElement(assignNode);
            graph.mapper.connect(assignTag, tdm, NaryArgument::DEST);

            Log::debug("KernelGraph::makeTDM: assign {} expression {} to TDM {}",
                       assignTag,
                       toString(assignNode.expression),
                       tdm);

            return assignTag;
        }

    } // namespace AssignIndexExpressionsDetail

    // Import detail namespace for internal use
    using namespace AssignIndexExpressionsDetail;

    /**
     * @brief Orchestrates adding index assignment operations to the control graph.
     *
     * This class implements a two-phase approach:
     *
     * **Staging Phase** (`stage()`):
     * - Analyzes each load/store operation to determine where its index chain should be placed
     *   - Usually index assignment operations come in groups of two or more operations called "index chains"
     * - Groups operations that would use identical chains (deduplication)
     * - Placement is determined by ForLoop/Unroll coordinate requirements
     *
     * **Commit Phase** (`commit()`):
     * - Creates actual Assign nodes for offset/stride/buffer computations
     * - Inserts chains into the control graph at their determined locations
     * - Connects operations to shared coordinate edges
     *
     * **Placement Rules** (in priority order):
     * 1. KLoop with KLoopTail: Hoist to common ancestor scope
     * 2. Receive tile loop: Place at top of containing ForLoop
     * 3. Uniform ForLoop: Place above the ForLoop with increment attached
     * 4. Prefetch context: Place in containing scope
     * 5. Non-uniform loop with Unroll: Place at top of ForLoop
     * 6. Unroll only: Place in Initialize block
     * 7. Default: Immediate placement above operation
     */
    struct AssignIndexer
    {
        AssignIndexer(ContextPtr context, CommandPtr command)
            : m_context(context)
            , m_command(command)
        {
        }

        /**
         * @brief Stage an index chain for later creation.
         *
         * Chains with identical specs are grouped so they can share computation.
         */
        void stageChain(KernelGraph const& graph,
                        int                target,
                        int                candidate,
                        int                location,
                        Graph::Direction   direction,
                        bool               isStorePartOfGlobalToLDSOp,
                        int                forLoop           = -1,
                        bool               replaceWithScope  = true,
                        int                inlineUnrollValue = -1,
                        int                inlineUnrollCoord = -1)
        {
            // Build coordinate list for deduplication key
            std::vector<int> specCoords;
            for(auto const& info :
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
                                isStorePartOfGlobalToLDSOp,
                                inlineUnrollValue,
                                inlineUnrollCoord};
            m_chains[spec].push_back(candidate);
        }

        /**
         * @brief Determine placement for a KLoop with a corresponding KLoopTail.
         *
         * When both KLoop and KLoopTail exist, we hoist the chain to their
         * common ancestor so both loops can share the same index computation.
         */
        std::optional<ChainPlacement> tryPlaceKLoopWithTail(KernelGraph const& kgraph,
                                                            int                candidate,
                                                            std::optional<int> maybeForLoop,
                                                            bool               hasForLoop,
                                                            bool               isUniformLoop) const
        {
            if(!(maybeForLoop && hasForLoop && isUniformLoop))
                return std::nullopt;

            auto maybeForLoopOp = kgraph.control.get<ForLoopOp>(*maybeForLoop);
            if(!maybeForLoopOp || maybeForLoopOp->loopName != rocRoller::KLOOP)
                return std::nullopt;

            auto maybeKLoopTail = FindCorrespondingKLoopTail(kgraph, *maybeForLoop);
            if(!maybeKLoopTail)
                return std::nullopt;

            auto maybeCommonAncestor
                = FindCommonAncestorScope(kgraph, *maybeForLoop, *maybeKLoopTail);
            if(!maybeCommonAncestor)
                return std::nullopt;

            Log::debug("  staged as: KLoop with KLoopTail, hoisting to common ancestor {} "
                       "(KLoop={}, KLoopTail={})",
                       *maybeCommonAncestor,
                       *maybeForLoop,
                       *maybeKLoopTail);

            return ChainPlacement{*maybeCommonAncestor, GD::Upstream, *maybeForLoop, true};
        }

        /**
         * @brief Check if candidate is in a prefetch context.
         *
         * A prefetch occurs when the operation requires ForLoop coordinates
         * that belong to a downstream ForLoop (not the containing one).
         */
        bool isPrefetchContext(KernelGraph const&             kgraph,
                               int                            candidate,
                               std::unordered_set<int> const& forLoopCoordinates) const
        {
            auto allChildForLoops
                = kgraph.control
                      .findNodes(
                          getTopSetCoordinate(kgraph, candidate),
                          [&](int tag) -> bool {
                              return isOperation<ForLoopOp>(kgraph.control.getElement(tag));
                          },
                          GD::Downstream)
                      .to<std::vector>();

            return std::any_of(allChildForLoops.begin(), allChildForLoops.end(), [&](auto tag) {
                return forLoopCoordinates.count(kgraph.mapper.get<ForLoop>(tag)) > 0;
            });
        }

        void stage(KernelGraph const& kgraph, int candidate, bool isStorePartOfGlobalToLDSOp)
        {
            auto node = kgraph.control.getNode<Operation>(candidate);
            Log::debug("KernelGraph::AssignIndexExpressions::stage(): processing candidate({}): {}",
                       candidate,
                       toString(node));

            // Gather coordinate information
            auto [target, direction]
                = getOperationTarget(candidate, kgraph, isStorePartOfGlobalToLDSOp);
            auto [required, path]   = findRequiredCoordinates(target, direction, kgraph);
            auto forLoopCoordinates = filterCoordinates<ForLoop>(required, kgraph);
            auto unrollCoordinates  = filterCoordinates<Unroll>(required, kgraph);

            Log::debug("  target: {}", target);
            for(auto r : required)
                Log::debug("  required: {}: {}", r, toString(kgraph.coordinates.getNode(r)));

            // Check if an unroll coordinate must be incorporated into the base address
            auto [inlineUnrollCoord, inlineUnrollValue]
                = isStorePartOfGlobalToLDSOp ? std::pair{-1, -1}
                                             : GetInlineUnrollInfo(kgraph, candidate);

            // Gather loop context information
            auto maybeForLoop  = findContainingOperation<ForLoopOp>(candidate, kgraph);
            auto maybeScope    = findContainingOperation<Scope>(candidate, kgraph);
            bool hasForLoop    = !forLoopCoordinates.empty();
            bool hasUnroll     = !unrollCoordinates.empty();
            bool isUniformLoop = maybeForLoop && uniformForLoop(maybeForLoop, kgraph);

            // Try KLoop with KLoopTail placement (hoists to common ancestor)
            if(auto placement
               = tryPlaceKLoopWithTail(kgraph, candidate, maybeForLoop, hasForLoop, isUniformLoop))
            {
                stageChain(kgraph,
                           target,
                           candidate,
                           placement->location,
                           placement->direction,
                           isStorePartOfGlobalToLDSOp,
                           placement->forLoop,
                           placement->replaceWithScope,
                           inlineUnrollValue,
                           inlineUnrollCoord);
                return;
            }

            // Receive tile loop: place at top of containing ForLoop
            if(maybeForLoop && getForLoopName(kgraph, *maybeForLoop) == rocRoller::RECEIVE)
            {
                auto maybeTopOfLoop = findTopOfContainingOperation<ForLoopOp>(candidate, kgraph);
                AssertFatal(maybeTopOfLoop.has_value(),
                            "Expected to find top of ForLoop for ReceiveTileLoop placement");
                Log::debug("  staged as: ReceiveTileLoop, location {}", *maybeTopOfLoop);
                stageChain(kgraph,
                           target,
                           candidate,
                           *maybeTopOfLoop,
                           GD::Upstream,
                           isStorePartOfGlobalToLDSOp,
                           -1,
                           false,
                           inlineUnrollValue,
                           inlineUnrollCoord);
                return;
            }

            // Uniform ForLoop with ForLoop coordinates: place above the ForLoop
            if(hasForLoop && isUniformLoop)
            {
                Log::debug("  staged as: UniformForLoop, location {}", *maybeForLoop);
                stageChain(kgraph,
                           target,
                           candidate,
                           *maybeForLoop,
                           GD::Upstream,
                           isStorePartOfGlobalToLDSOp,
                           *maybeForLoop,
                           true,
                           inlineUnrollValue,
                           inlineUnrollCoord);
                return;
            }

            // Prefetch: place in containing scope (before the downstream ForLoop)
            if(hasForLoop && isPrefetchContext(kgraph, candidate, forLoopCoordinates))
            {
                AssertFatal(maybeScope.has_value(),
                            "Expected containing Scope for prefetch placement");
                Log::debug("  staged as: Prefetch, location {}", *maybeScope);
                stageChain(kgraph,
                           target,
                           candidate,
                           *maybeScope,
                           GD::Upstream,
                           isStorePartOfGlobalToLDSOp,
                           -1,
                           true,
                           inlineUnrollValue,
                           inlineUnrollCoord);
                return;
            }

            // Non-uniform loop with Unroll: place at top of ForLoop
            if(maybeForLoop && !isUniformLoop && hasUnroll)
            {
                auto maybeTopOfLoop = findTopOfContainingOperation<ForLoopOp>(candidate, kgraph);
                AssertFatal(
                    maybeTopOfLoop.has_value(),
                    "Expected to find top of ForLoop for NonUniformLoopWithUnroll placement");
                Log::debug("  staged as: NonUniformLoopWithUnroll, location {}", *maybeTopOfLoop);
                stageChain(kgraph,
                           target,
                           candidate,
                           *maybeTopOfLoop,
                           GD::Upstream,
                           isStorePartOfGlobalToLDSOp,
                           -1,
                           false,
                           inlineUnrollValue,
                           inlineUnrollCoord);
                return;
            }

            // Unroll only (no ForLoop): place in Initialize block below kernel root
            if(hasUnroll)
            {
                auto roots = kgraph.control.roots();
                AssertFatal(!roots.empty(), "Expected at least one root in control graph");
                auto kernel = *roots.begin();
                Log::debug("  staged as: UnrollOnly, location {}", kernel);
                stageChain(kgraph,
                           target,
                           candidate,
                           kernel,
                           GD::Downstream,
                           isStorePartOfGlobalToLDSOp,
                           -1,
                           true,
                           inlineUnrollValue,
                           inlineUnrollCoord);
                return;
            }

            // Uniform loop without ForLoop coordinates: place above the ForLoop
            if(isUniformLoop)
            {
                Log::debug("  staged as: UniformLoopNoCoord, location {}", *maybeForLoop);
                stageChain(kgraph,
                           target,
                           candidate,
                           *maybeForLoop,
                           GD::Upstream,
                           isStorePartOfGlobalToLDSOp,
                           *maybeForLoop,
                           true,
                           inlineUnrollValue,
                           inlineUnrollCoord);
                return;
            }

            // Default: immediate placement above the operation
            Log::debug("  staged as: Immediate, location {}", candidate);
            stageChain(
                kgraph, target, candidate, candidate, GD::Upstream, isStorePartOfGlobalToLDSOp);
        }

        /**
         * @brief Insert chain into graph for downstream placement.
         */
        static void insertChainDownstream(KernelGraph&          kgraph,
                                          IndexChainSpec const& spec,
                                          IndexChain const&     chain)
        {
            kgraph.control.addElement(Initialize(), {spec.location}, {chain.top});
        }

        /**
         * @brief Insert chain into graph for upstream placement with scope wrapping.
         */
        void insertChainUpstreamWithScope(KernelGraph&          kgraph,
                                          IndexChainSpec const& spec,
                                          IndexChain const&     chain,
                                          std::map<int, int>&   scopes,
                                          std::map<int, int>&   serializationPoints) const
        {
            // Create scope if needed
            if(!scopes.contains(spec.location))
            {
                auto newScope         = kgraph.control.addElement(Scope());
                scopes[spec.location] = replaceWith(kgraph, spec.location, newScope, false);
                serializationPoints[spec.location] = scopes[spec.location];
            }

            auto scope = scopes[spec.location];

            if(m_serializeAssigns)
            {
                // Chain after previous chain (or scope if first)
                auto insertionPoint = serializationPoints[spec.location];
                bool isScope        = kgraph.control.get<Scope>(insertionPoint).has_value();
                kgraph.control.addElement(isScope ? ControlEdge(Body()) : ControlEdge(Sequence()),
                                          {insertionPoint},
                                          {chain.top});
                kgraph.control.addElement(Sequence(), {chain.bottom}, {spec.location});
                serializationPoints[spec.location] = chain.bottom;
            }
            else
            {
                // All chains parallel under scope
                kgraph.control.addElement(Body(), {scope}, {chain.top});
                kgraph.control.addElement(Sequence(), {chain.bottom}, {spec.location});
            }
        }

        /**
         * @brief Insert chain into graph for upstream placement without scope replacement.
         */
        static void insertChainUpstreamNoReplace(KernelGraph&          kgraph,
                                                 IndexChainSpec const& spec,
                                                 IndexChain const&     chain,
                                                 std::map<int, int>&   scopes)
        {
            if(!scopes.contains(spec.location))
            {
                scopes[spec.location] = kgraph.control.addElement(Scope());
                insertWithBody(kgraph, spec.location, scopes[spec.location]);
            }
            insertBefore(kgraph, spec.location, chain.top, chain.bottom);
        }

        /**
         * @brief Commit all staged chains to the graph.
         *
         * Creates index chains for all staged operations and inserts them
         * into the control graph at their determined locations.
         */
        KernelGraph commit(KernelGraph const& original) const
        {
            auto kgraph = original;
            // Maps location to actual scope node
            std::map<int, int> scopes;
            // Maps location to last chain bottom for serialization
            std::map<int, int> serializationPoints;
            BufferMap          bufferMap;
            BaseAddressMap     baseAddressMap;
            TDMMap             tdmMap;

            // Build all chains and insert them into the graph
            for(auto const& [spec, candidates] : m_chains)
            {
                // Compute loop step for increment operations
                ExpressionPtr step = Expression::literal(1u);
                if(spec.forLoop > 0)
                {
                    auto [lhs, rhs] = getForLoopIncrement(kgraph, spec.forLoop);
                    step            = simplify(rhs);
                }

                Log::debug("KernelGraph::AssignIndexExpressions::commit(): candidate={} "
                           "isStorePartOfGlobalToLDSOp={} location={}",
                           candidates[0],
                           spec.isStorePartOfGlobalToLDSOp,
                           spec.location);

                // Create the index chain
                auto chain = createIndexChain(
                    kgraph, candidates[0], step, spec, bufferMap, baseAddressMap, tdmMap);

                // Insert chain into control graph
                if(spec.direction == GD::Downstream)
                {
                    // Add index assigns to an Initialize block below target
                    insertChainDownstream(kgraph, spec, chain);
                }
                else if(spec.replaceWithScope)
                {
                    // Add index assigns in a Scope above target. Only the location
                    // is within the scope.
                    insertChainUpstreamWithScope(kgraph, spec, chain, scopes, serializationPoints);
                }
                else
                {
                    // Add index assigns in a Scope above target. Everything underneath
                    // the location is within the scope.
                    insertChainUpstreamNoReplace(kgraph, spec, chain, scopes);
                }

                // Handle update operation for ForLoops
                if(chain.update > 0)
                {
                    if(spec.forLoop < 0)
                    {
                        // Prefetch case: remove orphan update
                        kgraph.control.deleteElement(chain.update);
                        kgraph.mapper.purge(chain.update);
                    }
                    else
                    {
                        // Attach increment to ForLoop
                        kgraph.control.addElement(
                            ForLoopIncrement(), {spec.forLoop}, {chain.update});
                    }
                }

                // Connect all candidates to the shared coordinate edges
                for(auto candidate : candidates)
                {
                    for(auto const& dc : chain.connections)
                        kgraph.mapper.connect(candidate, dc.coordinate, dc.connectionSpec);
                }

                // Create Assign nodes for each placeholder
                for(auto const& nodeInfo : chain.nodeInfos)
                    createAssignsForPlaceholder(kgraph,
                                                nodeInfo,
                                                m_context,
                                                m_command,
                                                spec.inlineUnrollValue,
                                                spec.inlineUnrollCoord);
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
                                                CommandPtr           command,
                                                int                  inlineUnrollValue = -1,
                                                int                  inlineUnrollCoord = -1)
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

            // Incorporate the unroll value into the base address so that
            // non-affine transforms (e.g. LDS bank swizzle) receive the
            // correct input. Must precede zeroing so the loop skips it.
            if(inlineUnrollCoord >= 0 && inlineUnrollValue >= 0
               && !xform.hasCoordinate(inlineUnrollCoord))
            {
                xform.setCoordinate(inlineUnrollCoord,
                                    L(static_cast<unsigned int>(inlineUnrollValue)));
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

            int assignStrideTag = -1, assignBaseTag = -1, assignBufferTag = -1, assignTDMTag = -1;

            if(nodeInfo.baseOffset < 0 && nodeInfo.offset > 0)
            {
                assignBaseTag = MakeAssignBase(kgraph,
                                               params,
                                               target,
                                               nodeInfo.offset,
                                               nodeInfo.baseAddress,
                                               nodeInfo.tdm,
                                               maybeLDS,
                                               isTransposed,
                                               context,
                                               command,
                                               xform);
            }

            if(nodeInfo.stride > 0)
            {
                assignStrideTag = MakeAssignStride(kgraph,
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
                    = MakeBuffer(kgraph, params, target, nodeInfo.buffer, context, command);
            }

            if(nodeInfo.tdm > 0)
            {
                assignTDMTag = MakeTDM(kgraph, params, target, nodeInfo.tdm, context, command);
            }

            // Insert Assign nodes after the NOP placeholder
            if(assignTDMTag != -1)
                insertAfter(kgraph, nodeInfo.nopTag, assignTDMTag, assignTDMTag);
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
