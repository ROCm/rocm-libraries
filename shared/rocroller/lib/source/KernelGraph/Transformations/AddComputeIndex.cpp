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

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddComputeIndex.hpp>
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

    struct ComputeIndexChainSpecification
    {
        int              target;
        std::vector<int> coords;
        int              location;
        Graph::Direction direction;
        int              forLoop                    = -1;
        bool             replaceWithScope           = true;
        bool             isStorePartOfGlobalToLDSOp = false;
    };

    bool operator<(const ComputeIndexChainSpecification& a, const ComputeIndexChainSpecification& b)
    {
        return std::tie(a.target, a.coords, a.location, a.direction)
               < std::tie(b.target, b.coords, b.location, b.direction);
    }

    struct ComputeIndexChain
    {
        int top, bottom;

        std::vector<DeferredConnection> connections;

        int update = -1;
    };

    struct RequiredCoordinateInfo
    {
        int  coord, base, sdim;
        bool isUnroll;
        bool needsUpdate;
    };

    using BufferMap = std::map<int, int>;

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
        auto op = graph.control.getElement(opTag);
        if(not(isOperation<LoadTiled>(op) || isOperation<StoreTiled>(op)
               || isOperation<LoadTileDirect2LDS>(op))
           || isStorePartOfDirect2LDSOp)
            return -1;

        if(!bufferMap.contains(dst))
        {
            auto wg        = graph.coordinates.addElement(Workgroup());
            bufferMap[dst] = graph.coordinates.addElement(Buffer(), {wg}, {dst});
        }

        return bufferMap[dst];
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
     * @brief Add a ComputeIndex node and add mapper connections.
     */
    int makeComputeIndex(KernelGraph& graph,
                         int          target,
                         int          increment,
                         int          base,
                         int          offset,
                         int          stride,
                         int          buffer,
                         bool         forward,
                         DataType     valueType,
                         DataType     offsetType,
                         DataType     strideType,
                         bool         isStorePartOfGlobalToLDSOp)
    {
        using CCI = Connections::ComputeIndex;
        using CCA = Connections::ComputeIndexArgument;

        auto ci = graph.control.addElement(
            ComputeIndex{forward, isStorePartOfGlobalToLDSOp, valueType, offsetType, strideType});

        if(base > 0)
            graph.mapper.connect(ci, base, CCI{CCA::BASE});
        if(buffer > 0)
            graph.mapper.connect(ci, buffer, CCI{CCA::BUFFER});
        if(increment > 0)
            graph.mapper.connect(ci, increment, CCI{CCA::INCREMENT});
        if(offset > 0)
            graph.mapper.connect(ci, offset, CCI{CCA::OFFSET});
        if(stride > 0)
            graph.mapper.connect(ci, stride, CCI{CCA::STRIDE});
        if(target > 0)
            graph.mapper.connect(ci, target, CCI{CCA::TARGET});

        rocRoller::Log::getLogger()->debug(
            "KernelGraph::makeComputeIndex: ci {} {}/{} {}; {}/{}/{}",
            ci,
            target,
            increment,
            forward,
            base,
            offset,
            stride);

        return ci;
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
                auto coord        = getNeighbourNodeInPath(forLoopCoord, direction, path, graph);

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
            std::vector<int> neighbourNodes;
            if(direction == Graph::Direction::Upstream)
                neighbourNodes = graph.coordinates.childNodes(unroll).to<std::vector>();
            else
                neighbourNodes = graph.coordinates.parentNodes(unroll).to<std::vector>();
            for(auto neighbourNode : neighbourNodes)
            {
                if(path.contains(neighbourNode) && !isForLoop.contains(neighbourNode))
                {
                    auto it = std::find(codegen.cbegin(), codegen.cend(), neighbourNode);
                    if(it == codegen.cend())
                    {
                        ordered.push_back(neighbourNode);
                        isUnroll.insert(neighbourNode);
                    }
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
        if(s || l || ll || sl || isStorePartOfGGlobalToLDSOp)
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
            {
                {
                    auto const subDimension
                        = kgraph.mapper.getConnectionSubdimension(candidate, unroll);
                    // Find the neighbour of the Unroll that:
                    // 1. is in the load/store coordinate transform path
                    // 2. has a Stride edge connected to it
                    std::vector<int> neighbourNodes;
                    if(direction == Graph::Direction::Downstream)
                        neighbourNodes = kgraph.coordinates.parentNodes(unroll).to<std::vector>();
                    else
                        neighbourNodes = kgraph.coordinates.childNodes(unroll).to<std::vector>();

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
                                   && std::find(
                                          strideCoords.begin(), strideCoords.end(), neighbourEdge)
                                          != strideCoords.end())
                                {
                                    auto maybeStrideTag = neighbourEdge;
                                    auto newConnection
                                        = makeConnection<Stride, Connections::UnrollStride>(
                                            maybeStrideTag, subDimension);
                                    connections.push_back(newConnection);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Add ComputeIndex nodes required for `op`.
     */
    ComputeIndexChain addComputeIndex(KernelGraph&                          graph,
                                      int                                   op,
                                      ExpressionPtr                         step,
                                      ComputeIndexChainSpecification const& spec,
                                      BufferMap&                            bufferMap)
    {
        rocRoller::Log::getLogger()->debug(
            "KernelGraph::AddComputeIndex()::genericComputeIndex(): op {} location {}",
            op,
            spec.location);

        auto dtype = getDataType(graph.control.getNode(op));

        auto [target, direction] = getOperationTarget(op, graph, spec.isStorePartOfGlobalToLDSOp);

        int                             update = -1;
        std::vector<int>                chain;
        std::vector<DeferredConnection> connections;
        std::map<int, int>              offsetOfCoord;
        bool                            hasUnroll = false;
        std::vector<int>                strideCoords;

        for(auto info :
            getRequiredCoordinatesInfo(op, spec.location, graph, spec.isStorePartOfGlobalToLDSOp))
        {
            if(info.isUnroll)
                hasUnroll = true;
            // Add ComputeIndex operation
            int offset = -1, stride = -1, buffer = -1;

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
            chain.push_back(makeComputeIndex(graph,
                                             target,
                                             info.coord,
                                             base,
                                             offset,
                                             stride,
                                             buffer,
                                             direction == Graph::Direction::Upstream,
                                             dtype,
                                             offsetDataType,
                                             strideDataType,
                                             spec.isStorePartOfGlobalToLDSOp));

            // Add connections for register allocate, and so tracer
            // can determine correct lifetimes
            if(offset != -1)
                connections.push_back(DC<Offset>(offset, info.sdim));
            if(stride != -1)
                connections.push_back(DC<Stride>(stride, info.sdim));
            if(buffer != -1)
                connections.push_back(DC<Buffer>(buffer));
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

        return {chain.front(), chain.back(), connections, update};
    }

    namespace
    {
        /**
         * @brief Find the corresponding KLoopTail for a given KLoop.
         *
         * KLoop and KLoopTail can be related in two ways:
         * 1. UnrollLoops.cpp: KLoopTail is downstream of KLoop via Sequence path
         * 2. AddPrefetch/commit: Both are children of the same ancestor Scope
         *
         * @param kgraph
         * @param kLoop The KLoop tag
         * @return The corresponding KLoopTail tag, or std::nullopt if none exists
         */
        std::optional<int> FindCorrespondingKLoopTail(KernelGraph const& kgraph, int kLoop)
        {
            // Strategy 1: Search downstream via Sequence edges (UnrollLoops case)
            // The KLoopTail is connected via: KLoop --Sequence--> ... --Sequence/Body--> KLoopTail
            for(auto node : kgraph.control.depthFirstVisit(kLoop, Graph::Direction::Downstream))
            {
                auto maybeForLoop = kgraph.control.get<ForLoopOp>(node);
                if(maybeForLoop && maybeForLoop->loopName == rocRoller::KLOOPTAIL)
                    return node;
            }

            // Strategy 2: Search for siblings under common parent Scope (AddPrefetch/commit case)
            // Find parent Scopes of kLoop
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
                    {
                        // Verify they share this common ancestor
                        auto kLoopAncestors
                            = kgraph.control.breadthFirstVisit(kLoop, Graph::Direction::Upstream)
                                  .to<std::set>();
                        if(kLoopAncestors.contains(ancestor))
                            return descendant;
                    }
                }
            }

            return std::nullopt;
        }

        /**
         * @brief Find the corresponding KLoop for a given KLoopTail.
         *
         * KLoop and KLoopTail can be related in two ways:
         * 1. UnrollLoops.cpp: KLoopTail is downstream of KLoop via Sequence path
         * 2. AddPrefetch/commit: Both are children of the same ancestor Scope
         *
         * @param kgraph
         * @param kLoopTail The KLoopTail tag
         * @return The corresponding KLoop tag, or std::nullopt if none exists
         */
        std::optional<int> FindCorrespondingKLoop(KernelGraph const& kgraph, int kLoopTail)
        {
            // Strategy 1: Search upstream via Sequence edges (UnrollLoops case)
            // The KLoop is connected via: KLoop --Sequence--> ... --Sequence/Body--> KLoopTail
            for(auto node : kgraph.control.breadthFirstVisit(kLoopTail, Graph::Direction::Upstream))
            {
                auto maybeForLoop = kgraph.control.get<ForLoopOp>(node);
                if(maybeForLoop && maybeForLoop->loopName == rocRoller::KLOOP)
                    return node;
            }

            // Strategy 2: Search for siblings under common parent Scope (AddPrefetch/commit case)
            // Find parent Scopes of kLoopTail
            for(auto ancestor :
                kgraph.control.breadthFirstVisit(kLoopTail, Graph::Direction::Upstream))
            {
                if(!kgraph.control.get<Scope>(ancestor))
                    continue;

                // Search all descendants of this Scope for a KLoop
                for(auto descendant :
                    kgraph.control.depthFirstVisit(ancestor, Graph::Direction::Downstream))
                {
                    if(descendant == kLoopTail)
                        continue;

                    auto maybeForLoop = kgraph.control.get<ForLoopOp>(descendant);
                    if(maybeForLoop && maybeForLoop->loopName == rocRoller::KLOOP)
                    {
                        // Verify they share this common ancestor
                        auto kLoopTailAncestors
                            = kgraph.control
                                  .breadthFirstVisit(kLoopTail, Graph::Direction::Upstream)
                                  .to<std::set>();
                        if(kLoopTailAncestors.contains(ancestor))
                            return descendant;
                    }
                }
            }

            return std::nullopt;
        }

        /**
         * @brief Find the closest common ancestor Scope between two nodes.
         *
         * Uses breadthFirstVisit to traverse from both nodes upstream to find their
         * first common ancestor that is a Scope node in the control graph.
         *
         * @param kgraph
         * @param nodeA
         * @param nodeB
         * @return The tag of the common ancestor Scope, or std::nullopt if none exists
         */
        std::optional<int> FindCommonAncestorScope(KernelGraph const& kgraph, int nodeA, int nodeB)
        {
            // Collect all ancestors of nodeA using breadthFirstVisit
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

        /**
         * @brief Find the immediate child of ancestor on the path to target.
         *
         * Uses breadthFirstVisit to traverse upstream from target to ancestor.
         *
         * @param kgraph
         * @param target The target node to start from
         * @param ancestor The ancestor node to reach
         * @return The child edge of ancestor on the path to target, or std::nullopt if no path found
         */
        std::optional<int> FindChildOnPath(KernelGraph const& kgraph, int target, int ancestor)
        {
            std::unordered_map<int, int> parentMap;
            parentMap[target] = -1;

            for(auto current : kgraph.control.breadthFirstVisit(target, Graph::Direction::Upstream))
            {
                if(current == ancestor)
                    return parentMap[ancestor];

                auto loc = kgraph.control.getLocation(current);
                for(auto edge : loc.incoming)
                {
                    auto parents = kgraph.control.getNeighbours<Graph::Direction::Upstream>(edge);
                    for(auto parent : parents)
                    {
                        if(!parentMap.contains(parent))
                        {
                            parentMap[parent] = current;
                        }
                    }
                }
            }

            return std::nullopt;
        }

        /**
         * @brief Find and delete the edge from parent to child.
         *
         * Asserts if the edge is not found.
         *
         * @param kgraph
         * @param parent
         * @param child
         */
        void FindAndDeleteEdge(KernelGraph& kgraph, int parent, int child)
        {
            auto parentLoc = kgraph.control.getLocation(parent);
            for(auto edge : parentLoc.outgoing)
            {
                auto children = kgraph.control.getNeighbours<Graph::Direction::Downstream>(edge);
                for(auto c : children)
                {
                    if(c == child)
                    {
                        kgraph.control.deleteElement(edge);
                        return;
                    }
                }
            }
            AssertFatal(false,
                        "Could not find edge from parent to child",
                        ShowValue(parent),
                        ShowValue(child));
        }

        /**
         * @brief Identify KLoop/KLoopTail pairs that can share compute index chains.
         *
         * Finds KLoop/KLoopTail pairs with matching target/coords and marks the KLoopTail
         * specs to be skipped during chain creation, since they will share the KLoop's chain.
         *
         * This function assumes:
         * - Each KLoop has at most one corresponding KLoopTail with matching target/coords
         * - All specs have valid location nodes that can be queried for ForLoopOp
         *
         * @param kgraph
         * @param chains Map of chain specifications to candidates
         * @return A pair containing:
         *         - Set of KLoopTail specs to skip during chain creation
         *         - Map from KLoop specs to their KLoopTail candidates for sharing
         */
        std::pair<std::set<ComputeIndexChainSpecification>,
                  std::map<ComputeIndexChainSpecification, std::vector<int>>>
            IdentifySharedKLoopKLoopTailChains(
                KernelGraph const&                                                kgraph,
                std::map<ComputeIndexChainSpecification, std::vector<int>> const& chains)
        {
            std::set<ComputeIndexChainSpecification>                   specsToSkip;
            std::map<ComputeIndexChainSpecification, std::vector<int>> kLoopToKLoopTailCandidates;

            // Find all KLoop specs and check if they have corresponding KLoopTail specs
            for(auto const& [spec, candidates] : chains)
            {
                auto maybeForLoop = kgraph.control.get<ForLoopOp>(spec.location);
                if(!maybeForLoop || maybeForLoop->loopName != rocRoller::KLOOP)
                    continue;

                // Use helper function to find the corresponding KLoopTail in the graph
                auto maybeKLoopTail = FindCorrespondingKLoopTail(kgraph, spec.location);
                if(!maybeKLoopTail)
                    continue;

                // Look for a spec with this KLoopTail location and matching target/coords
                for(auto const& [otherSpec, otherCandidates] : chains)
                {
                    if(otherSpec.location != *maybeKLoopTail)
                        continue;

                    auto maybeOtherForLoop = kgraph.control.get<ForLoopOp>(otherSpec.location);
                    if(!maybeOtherForLoop || maybeOtherForLoop->loopName != rocRoller::KLOOPTAIL)
                        continue;

                    // Check if they share the same target and coordinates
                    if(otherSpec.target == spec.target && otherSpec.coords == spec.coords)
                    {
                        Log::debug("Skipping KLoopTail chain creation (duplicate of KLoop): "
                                   "target={}, KLoop={}, KLoopTail={}",
                                   spec.target,
                                   spec.location,
                                   otherSpec.location);

                        specsToSkip.insert(otherSpec);
                        kLoopToKLoopTailCandidates[spec] = otherCandidates;
                        break; // Found the matching KLoopTail spec
                    }
                }
            }

            return {specsToSkip, kLoopToKLoopTailCandidates};
        }

        /**
         * @brief Reconnect KLoopTail subtrees to execute sequentially after their corresponding KLoops.
         *
         * Uses helper functions to identify corresponding pairs, supporting multiple
         * independent KLoop/KLoopTail pairs (e.g., in StreamK scenarios).
         *
         * This function assumes (all asserted):
         * - Every KLoopTail in the graph has a corresponding KLoop
         * - Every KLoop/KLoopTail pair has a common ancestor Scope
         * - A valid path exists from the common ancestor to the KLoopTail
         * - An edge exists from ancestor to the child on the path
         *
         * @param kgraph
         */
        void ReconnectKLoopTailsAfterKLoops(KernelGraph& kgraph)
        {
            // Find all KLoopTail ForLoopOps
            auto kernel = *kgraph.control.roots().begin();
            auto kLoopTails
                = kgraph.control
                      .findNodes(
                          kernel,
                          [&](int tag) -> bool {
                              auto maybeForLoop = kgraph.control.get<ForLoopOp>(tag);
                              return maybeForLoop && maybeForLoop->loopName == rocRoller::KLOOPTAIL;
                          },
                          GD::Downstream)
                      .to<std::vector>();

            // For each KLoopTail, find its corresponding KLoop and reconnect
            for(auto kLoopTail : kLoopTails)
            {
                // Use helper function to find the corresponding KLoop
                auto maybeKLoop = FindCorrespondingKLoop(kgraph, kLoopTail);
                AssertFatal(maybeKLoop,
                            "Could not find corresponding KLoop for KLoopTail",
                            ShowValue(kLoopTail));
                int kLoop = *maybeKLoop;

                // Find common ancestor Scope
                auto commonAncestor = FindCommonAncestorScope(kgraph, kLoop, kLoopTail);
                AssertFatal(commonAncestor,
                            "Could not find common ancestor Scope for KLoop and KLoopTail",
                            ShowValue(kLoop),
                            ShowValue(kLoopTail));

                // Find immediate child of ancestor on path to KLoopTail
                auto childOnPath = FindChildOnPath(kgraph, kLoopTail, *commonAncestor);
                AssertFatal(childOnPath,
                            "Could not find child on path from ancestor to KLoopTail",
                            ShowValue(kLoopTail),
                            ShowValue(*commonAncestor));

                // Disconnect the child from the common ancestor
                FindAndDeleteEdge(kgraph, *commonAncestor, *childOnPath);

                // Connect the child as a Sequence child of KLoop
                kgraph.control.addElement(Sequence(), {kLoop}, {*childOnPath});
                Log::debug("Reconnected KLoopTail {} to KLoop {} via common ancestor {}",
                           kLoopTail,
                           kLoop,
                           *commonAncestor);
            }
        }
    } // anonymous namespace

    /**
     * @brief Add ComputeIndex operations.
     *
     * Adding ComputeIndex operations to the control graph is done in
     * two phases: staging and committing.
     *
     * During the staging phase, we look at all load/store operations
     * in the control graph and "stage" the addition of ComputeIndex
     * operations.  During the staging phase, we are able to detect
     * when two or more load/store operations would result in the same
     * chain of ComputeIndex operations, and eliminate any
     * redundancies.
     *
     * Usually ComputeIndex operations come in sequential groups of
     * two or more operations, and hence we call them "compute index
     * chains".
     *
     * During the commit stage, we add ComputeIndex operations to the
     * graphs, and add connections for load/store operations to the
     * newly created Base, Offset, and Stride elements of the
     * coordinate graph.
     *
     * For each candidate load/store operation:
     *
     * 1. The type of ComputeIndex chain is determined.
     *
     * 2. The required location of the ComputeIndex chain is
     *    determined.
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
    struct AddComputeIndexer
    {
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

            ComputeIndexChainSpecification spec{target,
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
            log->debug("KernelGraph::addComputeIndex({}): {}", candidate, toString(node));

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
            std::map<int, int> scopes;
            BufferMap          bufferMap;

            removeRedundantSequenceEdges(kgraph);
            auto [specsToSkip, kLoopToKLoopTailCandidates]
                = IdentifySharedKLoopKLoopTailChains(kgraph, m_chains);

            for(auto const& [spec, candidates] : m_chains)
            {
                // Skip KLoopTail specs that will share KLoop's chain
                if(specsToSkip.contains(spec))
                    continue;

                AssertFatal(
                    !candidates.empty(),
                    "ComputeIndexChainSpecification must have at least one candidate operation");

                ExpressionPtr step = Expression::literal(1u);
                if(spec.forLoop > 0)
                {
                    auto [lhs, rhs] = getForLoopIncrement(kgraph, spec.forLoop);
                    step            = simplify(rhs);
                }

                // Check if this is KLoopTail and log why it's being created (not deduplicated)
                auto maybeForLoop = kgraph.control.get<ForLoopOp>(spec.location);
                if(maybeForLoop && maybeForLoop->loopName == rocRoller::KLOOPTAIL)
                {
                    Log::debug("Creating KLoopTail chain: target={}, location={} (unique coords "
                               "not shared with KLoop)",
                               spec.target,
                               spec.location);
                }

                // Use first candidate to compute indexes
                Log::debug("KernelGraph::AddComputeIndex()::commit({}) "
                           "isStorePartOfGlobalToLDSOp({}) location {}",
                           candidates[0],
                           spec.isStorePartOfGlobalToLDSOp,
                           spec.location);

                auto chain = addComputeIndex(kgraph, candidates[0], step, spec, bufferMap);

                if(spec.direction == GD::Downstream)
                {
                    // Add ComputeIndexes to an Initialize block below target
                    kgraph.control.addElement(Initialize(), {spec.location}, {chain.top});
                }
                else
                {
                    if(spec.replaceWithScope)
                    {
                        // Add ComputeIndexes in a Scope above target. Only the location
                        // is within the scope.
                        if(!scopes.contains(spec.location))
                        {
                            auto scopeNode = kgraph.control.addElement(Scope());
                            scopes[spec.location]
                                = replaceWith(kgraph, spec.location, scopeNode, false);
                        }

                        auto scope = scopes[spec.location];
                        if(m_serializeComputeIndex)
                        {
                            kgraph.control.addElement(kgraph.control.get<Scope>(scope).has_value()
                                                          ? ControlEdge(Body())
                                                          : ControlEdge(Sequence()),
                                                      {scope},
                                                      {chain.top});
                            kgraph.control.addElement(Sequence(), {chain.bottom}, {spec.location});
                            scopes[spec.location] = chain.bottom;
                        }
                        else
                        {
                            kgraph.control.addElement(Body(), {scope}, {chain.top});
                            kgraph.control.addElement(Sequence(), {chain.bottom}, {spec.location});
                        }
                    }
                    else
                    {
                        // Add ComputeIndexes in a Scope above target. Everything underneath
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
                    AssertFatal(spec.forLoop > 0,
                                "Chain has an update operation but no associated ForLoop",
                                ShowValue(chain.update),
                                ShowValue(spec.forLoop));
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

                // If this is a KLoop chain, also connect the KLoopTail candidates
                if(kLoopToKLoopTailCandidates.contains(spec))
                {
                    auto const& kLoopTailCandidates = kLoopToKLoopTailCandidates.at(spec);
                    for(auto candidate : kLoopTailCandidates)
                    {
                        for(auto const& dc : chain.connections)
                        {
                            kgraph.mapper.connect(candidate, dc.coordinate, dc.connectionSpec);
                        }
                    }
                }
            }

            ReconnectKLoopTailsAfterKLoops(kgraph);
            removeRedundantSequenceEdges(kgraph);

            return kgraph;
        }

    private:
        std::map<ComputeIndexChainSpecification, std::vector<int>> m_chains;

        bool m_serializeComputeIndex = true;
    };

    KernelGraph AddComputeIndex::apply(KernelGraph const& original)
    {
        AddComputeIndexer indexer;

        for(auto candidate :
            findComputeIndexCandidates(original, *original.control.roots().begin()))
        {
            // Global to LDS ops have two sets of coordinates for the load and store parts
            indexer.stage(original, candidate, false);
            if(isGlobalToLDSOp(original, candidate))
                indexer.stage(original, candidate, /*isStorePartOfGlobalToLDSOp=*/true);
        }

        return indexer.commit(original);
    }
}
