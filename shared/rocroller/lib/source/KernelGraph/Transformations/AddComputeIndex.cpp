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

        // Use spec.forLoop as the location if it exists (for hoisted chains),
        // otherwise use spec.location
        int locationForCoordInfo = (spec.forLoop > 0) ? spec.forLoop : spec.location;
        for(auto info : getRequiredCoordinatesInfo(
                op, locationForCoordInfo, graph, spec.isStorePartOfGlobalToLDSOp))
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

        /**
         * @brief Identify KLoop/KLoopTail pairs that can share compute index chains.
         *
         * @param kgraph
         * @param chains Map of chain specifications to candidates
         * @return Map from KLoop specs to their corresponding KLoopTail specs
         */
        std::map<ComputeIndexChainSpecification, ComputeIndexChainSpecification>
            IdentifySharedKLoopKLoopTailChains(
                KernelGraph const&                                                kgraph,
                std::map<ComputeIndexChainSpecification, std::vector<int>> const& chains)
        {
            std::map<ComputeIndexChainSpecification, ComputeIndexChainSpecification>
                kLoopToKLoopTailSpec;

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

                    // Check if they share the same target and coordinates
                    if(otherSpec.target == spec.target && otherSpec.coords == spec.coords)
                    {
                        Log::debug("Identified shared KLoop/KLoopTail chain: target={}, KLoop={}, "
                                   "KLoopTail={}",
                                   spec.target,
                                   spec.location,
                                   otherSpec.location);

                        // Both loops will compute their full chains including ForLoop offsets.
                        // The loop-invariant parts (buffer descriptors, strides) will be
                        // computed in the hoisted chain and reused.
                        kLoopToKLoopTailSpec[spec] = otherSpec;
                        break; // Found the matching KLoopTail spec
                    }
                }
            }

            return kLoopToKLoopTailSpec;
        }

        /**
         * @brief Add Deallocate nodes for arguments used exclusively by hoisted chains.
         *
         * When buffer descriptors are created in hoisted chains, their constituent kernel
         * arguments are copied into SGPRs (e.g., s_mov_b32 s52, s6). After this copy, the
         * original argument registers can be freed if they're not used by any other chains.
         * This function identifies such arguments and inserts explicit Deallocate nodes to
         * enable earlier register reuse, reducing SGPR pressure.
         *
         * @param kgraph
         * @param kLoopToKLoopTailSpec Map from KLoop specs to their KLoopTail counterparts
         * @param hoistedArgNames Set of argument names used by hoisted chains
         * @param argToChainBottom Map from argument names to the bottom control node of hoisted chains
         * @param chains All ComputeIndex chain specifications (to check for other usages)
         */
        void DeallocateHoistedArguments(
            KernelGraph& kgraph,
            std::map<ComputeIndexChainSpecification, ComputeIndexChainSpecification> const&
                                                                              kLoopToKLoopTailSpec,
            std::set<std::string> const&                                      hoistedArgNames,
            std::map<std::string, int> const&                                 argToChainBottom,
            std::map<ComputeIndexChainSpecification, std::vector<int>> const& chains)
        {
            // Build set of shared KLoop/KLoopTail specs
            std::set<ComputeIndexChainSpecification> sharedKLoopSpecs;
            for(auto const& [kLoopSpec, kLoopTailSpec] : kLoopToKLoopTailSpec)
            {
                sharedKLoopSpecs.insert(kLoopSpec);
                sharedKLoopSpecs.insert(kLoopTailSpec);
            }

            std::map<std::string, std::set<int>> argNameToBuffers;
            for(auto userNode : kgraph.coordinates.getNodes<User>())
            {
                auto user = kgraph.coordinates.get<User>(userNode);
                if(user && !user->argumentName.empty())
                {
                    auto userBuffers = kgraph.coordinates.parentNodes(userNode).to<std::set>();
                    argNameToBuffers[user->argumentName].insert(userBuffers.begin(),
                                                                userBuffers.end());
                }
            }

            // Helper: check if a spec uses any of the given buffers
            auto specUsesBuffers
                = [&](ComputeIndexChainSpecification const& spec, std::set<int> const& buffers) {
                      for(auto coord : spec.coords)
                      {
                          for(auto child : kgraph.coordinates.childNodes(coord))
                          {
                              if(buffers.contains(child))
                                  return true;
                          }
                      }
                      return false;
                  };

            Log::debug("Checking {} arguments for early deallocation", hoistedArgNames.size());

            for(auto const& argName : hoistedArgNames)
            {
                auto const& buffersUsingThisArg = argNameToBuffers[argName];

                // Check if any non-hoisted spec uses this argument
                auto nonHoistedSpecUsesArg
                    = std::any_of(chains.begin(), chains.end(), [&](auto const& entry) {
                          auto const& [spec, candidates] = entry;
                          if(sharedKLoopSpecs.contains(spec))
                              return false; // Skip hoisted specs

                          if(specUsesBuffers(spec, buffersUsingThisArg))
                          {
                              Log::debug("Argument '{}' used by non-hoisted spec (target={})",
                                         argName,
                                         spec.target);
                              return true;
                          }
                          return false;
                      });

                if(!nonHoistedSpecUsesArg)
                {
                    Log::debug("  Decision: SAFE to early deallocate '{}'", argName);
                    auto deallocate = kgraph.control.addElement(Deallocate{{argName}});
                    kgraph.control.addElement(
                        Sequence(), {argToChainBottom.at(argName)}, {deallocate});
                    Log::debug("  Added Deallocate node for '{}' after control node {}",
                               argName,
                               argToChainBottom.at(argName));
                }
                else
                {
                    Log::debug("  Decision: NOT safe to early deallocate '{}' - used elsewhere",
                               argName);
                }
            }
        }

        /**
         * @brief Hoist loop-invariant portions of ComputeIndex chains for KLoop/KLoopTail pairs.
         *
         * When KLoop and KLoopTail have similar ComputeIndex chains, they often share
         * loop-invariant parts (buffer descriptors, strides) but differ in loop-variant parts
         * (iteration-dependent offsets from ForLoop coordinates). This function:
         *
         * 1. Identifies loop-invariant coordinates (non-ForLoop) for each KLoop/KLoopTail pair
         * 2. Creates a "hoisted spec" containing only these loop-invariant coordinates
         * 3. Places the hoisted chain at the common ancestor Scope (above both loops)
         * 4. Tracks buffer arguments for potential early deallocation
         *
         * Both KLoop and KLoopTail still compute their full chains (including ForLoop offsets),
         * but they reuse the buffer descriptors from the hoisted chain, reducing SGPR pressure.
         *
         * @param kgraph The kernel graph to modify
         * @param kLoopToKLoopTailSpec Map from KLoop specs to their KLoopTail counterparts
         * @param chains All ComputeIndex chain specifications
         * @param bufferMap Buffer map for reusing buffer descriptors across chains
         */
        void HoistSharedKLoopKLoopTailChains(
            KernelGraph& kgraph,
            std::map<ComputeIndexChainSpecification, ComputeIndexChainSpecification> const&
                                                                              kLoopToKLoopTailSpec,
            std::map<ComputeIndexChainSpecification, std::vector<int>> const& chains,
            BufferMap&                                                        bufferMap)
        {
            // Create hoisted specs: loop-invariant coordinates at common ancestor
            std::map<ComputeIndexChainSpecification, ComputeIndexChainSpecification> hoistedSpecs;

            for(auto const& [kLoopSpec, kLoopTailSpec] : kLoopToKLoopTailSpec)
            {
                Log::debug("Analyzing KLoop={} <-> KLoopTail={} pair for hoisting",
                           kLoopSpec.location,
                           kLoopTailSpec.location);

                auto maybeCommonAncestor
                    = FindCommonAncestorScope(kgraph, kLoopSpec.location, kLoopTailSpec.location);
                if(!maybeCommonAncestor)
                {
                    Log::debug("  No common ancestor Scope found - skipping");
                    continue;
                }

                // Filter out ForLoop coordinates (loop-variant)
                auto forLoopCoords = filterCoordinates<ForLoop>(kLoopSpec.coords, kgraph);
                std::vector<int> loopInvariantCoords;
                for(auto coord : kLoopSpec.coords)
                {
                    if(!forLoopCoords.contains(coord))
                        loopInvariantCoords.push_back(coord);
                }

                Log::debug("  Total coords: {}, ForLoop coords (loop-variant): {}, "
                           "Loop-invariant coords: {}",
                           kLoopSpec.coords.size(),
                           forLoopCoords.size(),
                           loopInvariantCoords.size());

                if(!loopInvariantCoords.empty())
                {
                    Log::debug("  Creating hoisted spec with {} loop-invariant coordinates",
                               loopInvariantCoords.size());

                    ComputeIndexChainSpecification hoistedSpec = kLoopSpec;
                    hoistedSpec.coords                         = loopInvariantCoords;
                    hoistedSpec.location                       = *maybeCommonAncestor;
                    hoistedSpec.forLoop                        = -1;
                    hoistedSpec.replaceWithScope               = false;
                    hoistedSpecs[kLoopSpec]                    = hoistedSpec;
                }
                else
                {
                    Log::debug("  No loop-invariant coords to hoist - skipping");
                }
            }

            // Build all hoisted chains and track arguments for conditional deallocation
            std::set<std::string>      hoistedArgNames;
            std::map<std::string, int> argToChainBottom;

            Log::debug("Building {} hoisted chains", hoistedSpecs.size());

            for(auto const& [originalSpec, hoistedSpec] : hoistedSpecs)
            {
                Log::debug("Building hoisted chain: location={}, target={}, numCoords={}",
                           hoistedSpec.location,
                           hoistedSpec.target,
                           hoistedSpec.coords.size());

                auto const& candidates = chains.at(originalSpec);
                auto        chain      = addComputeIndex(
                    kgraph, candidates[0], Expression::literal(1u), hoistedSpec, bufferMap);

                // Hoisted chains are always placed at a Scope (common ancestor), never at kernel root
                AssertFatal(hoistedSpec.direction == GD::Upstream,
                            "Hoisted chains must be Upstream (inserted before their location)");
                insertBefore(kgraph, hoistedSpec.location, chain.top, chain.bottom);

                // Track buffer arguments and chain bottoms
                for(auto const& dc : chain.connections)
                {
                    for(auto userNode :
                        kgraph.coordinates.childNodes(dc.coordinate).to<std::vector>())
                    {
                        if(auto user = kgraph.coordinates.get<User>(userNode))
                        {
                            if(!user->argumentName.empty())
                            {
                                hoistedArgNames.insert(user->argumentName);
                                argToChainBottom[user->argumentName] = chain.bottom;
                            }
                        }
                    }
                }
            }

            Log::debug("Hoisting complete: {} unique arguments tracked", hoistedArgNames.size());

            // Deallocate arguments used only by hoisted chains
            DeallocateHoistedArguments(
                kgraph, kLoopToKLoopTailSpec, hoistedArgNames, argToChainBottom, chains);
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
            std::map<int, int> scopes;
            BufferMap          bufferMap;

            // Identify and hoist shared KLoop/KLoopTail chains
            auto kLoopToKLoopTailSpec = IdentifySharedKLoopKLoopTailChains(kgraph, m_chains);
            HoistSharedKLoopKLoopTailChains(kgraph, kLoopToKLoopTailSpec, m_chains, bufferMap);

            // Build chains for each loop (including ForLoop-dependent offsets)
            // Both KLoop and KLoopTail build full chains; bufferMap reuses hoisted buffers
            for(auto const& [spec, candidates] : m_chains)
            {
                ExpressionPtr step = Expression::literal(1u);
                if(spec.forLoop > 0)
                {
                    auto [lhs, rhs] = getForLoopIncrement(kgraph, spec.forLoop);
                    step            = simplify(rhs);
                }

                // Use first candidate to compute indexes
                Log::debug(
                    "KernelGraph::AddComputeIndex()::commit({}) isStorePartOfGlobalToLDSOp({}) "
                    "location={}",
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
                            scopes[spec.location] = replaceWith(
                                kgraph, spec.location, kgraph.control.addElement(Scope()), false);
                        }
                        auto scope = scopes[spec.location];
                        if(m_serializeComputeIndex)
                        {
                            auto isScope = kgraph.control.get<Scope>(scope).has_value();
                            kgraph.control.addElement(isScope ? ControlEdge(Body())
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
            }

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
