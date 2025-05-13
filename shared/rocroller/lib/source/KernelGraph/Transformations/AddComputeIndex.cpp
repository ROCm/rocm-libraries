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
#include <rocRoller/KernelGraph/Utils.hpp>

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
        int              forLoop          = -1;
        bool             replaceWithScope = true;
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
    int getBuffer(KernelGraph& graph, int opTag, int dst, BufferMap& bufferMap, bool isDirect2LDS)
    {
        auto op = graph.control.getElement(opTag);
        if(isOperation<LoadLDSTile>(op) || isOperation<StoreLDSTile>(op) || isDirect2LDS)
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
                         DataType     offsetType = DataType::UInt64,
                         DataType     strideType = DataType::UInt64)
    {
        using CCI = Connections::ComputeIndex;
        using CCA = Connections::ComputeIndexArgument;

        auto ci
            = graph.control.addElement(ComputeIndex(forward, valueType, offsetType, strideType));

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
                                                                   bool isDirect2LDS = false)
    {
        auto [target, direction] = getOperationTarget(op, graph, isDirect2LDS);
        auto [required, path]    = findRequiredCoordinates(target, direction, graph);
        auto codegen             = getCodeGeneratorCoordinates(graph, op, isDirect2LDS);

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

            if(isDirect2LDS)
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
    DataType getOffsetDataType(int op, KernelGraph const& graph, bool direct2LDS)

    {
        DataType rv = DataType::UInt64;
        auto     ll = graph.control.get<LoadLDSTile>(op);
        auto     sl = graph.control.get<StoreLDSTile>(op);
        if(ll || sl || direct2LDS)
        {
            rv = DataType::UInt32;
        }
        return rv;
    }

    /**
     * @brief Add ComputeIndex nodes required for `op`.
     */
    ComputeIndexChain addComputeIndex(KernelGraph&  graph,
                                      int           op,
                                      ExpressionPtr step,
                                      int           location,
                                      BufferMap&    bufferMap,
                                      bool          isDirect2LDS)
    {
        rocRoller::Log::getLogger()->debug(
            "KernelGraph::AddComputeIndex()::genericComputeIndex(): op {} location {}",
            op,
            location);

        auto dtype = getDataType(graph.control.getNode(op));

        auto [target, direction] = getOperationTarget(op, graph, isDirect2LDS);

        int                             update = -1;
        std::vector<int>                chain;
        std::vector<DeferredConnection> connections;
        std::map<int, int>              offsetOfCoord;

        for(auto info : getRequiredCoordinatesInfo(op, location, graph, isDirect2LDS))
        {
            // Add ComputeIndex operation
            int offset = -1, stride = -1, buffer = -1;
            if(direction == Graph::Direction::Downstream)
            {
                if(!info.isUnroll)
                    offset = graph.coordinates.addElement(Offset(), {target}, {info.coord});
                stride = graph.coordinates.addElement(Stride(), {target}, {info.coord});
                if(info.base == -1 && offset != -1)
                    buffer = getBuffer(graph, op, target, bufferMap, isDirect2LDS);
            }
            else
            {
                if(!info.isUnroll)
                    offset = graph.coordinates.addElement(Offset(), {info.coord}, {target});
                stride = graph.coordinates.addElement(Stride(), {info.coord}, {target});
                if(info.base == -1 && offset != -1)
                    buffer = getBuffer(graph, op, target, bufferMap, isDirect2LDS);
            }

            offsetOfCoord[info.coord] = offset;

            int base = (info.base == -1) ? -1 : offsetOfCoord.at(info.base);

            auto offsetDataType = getOffsetDataType(op, graph, isDirect2LDS);
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
                                             strideDataType));

            // Add connections for register allocate, and so tracer
            // can determine correct lifetimes
            if(offset != -1)
                connections.push_back(DC<Offset>(offset, info.sdim));
            if(stride != -1)
                connections.push_back(DC<Stride>(stride, info.sdim));
            if(buffer != -1)
                connections.push_back(DC<Buffer>(buffer));

            if(info.needsUpdate)
            {
                auto offsetExpr = std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{offset, Register::Type::Vector, DataType::UInt64});
                auto strideExpr = std::make_shared<Expression::Expression>(
                    Expression::DataFlowTag{stride, Register::Type::Scalar, DataType::UInt64});

                if(step == nullptr)
                    update = graph.control.addElement(
                        Assign{Register::Type::Vector, offsetExpr + strideExpr});
                else
                    update = graph.control.addElement(
                        Assign{Register::Type::Vector, offsetExpr + step * strideExpr});
                graph.mapper.connect(update, offset, NaryArgument::DEST);
            }
        }

        for(int i = 1; i < chain.size(); ++i)
            graph.control.addElement(Sequence(), {chain[i - 1]}, {chain[i]});

        return {chain.front(), chain.back(), connections, update};
    }

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
                        bool               isDirect2LDS     = false,
                        int                forLoop          = -1,
                        bool               replaceWithScope = true)
        {
            std::vector<int> specCoords;
            for(auto info : getRequiredCoordinatesInfo(candidate, location, graph, isDirect2LDS))
            {
                specCoords.push_back(info.coord);
            }

            ComputeIndexChainSpecification spec{
                target, specCoords, location, direction, forLoop, replaceWithScope};
            m_chains[spec].push_back(candidate);
        }

        void stage(KernelGraph const& kgraph, int candidate, bool isDirect2LDS)
        {
            auto log = rocRoller::Log::getLogger();

            auto node = kgraph.control.getNode<Operation>(candidate);
            log->debug("KernelGraph::addComputeIndex({}): {}", candidate, toString(node));

            auto [target, direction] = getOperationTarget(candidate, kgraph, isDirect2LDS);
            auto [required, path]    = findRequiredCoordinates(target, direction, kgraph);
            auto forLoopCoordinates  = filterCoordinates<ForLoop>(required, kgraph);
            auto unrollCoordinates   = filterCoordinates<Unroll>(required, kgraph);

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
                           isDirect2LDS,
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
                stageChain(kgraph, target, candidate, *maybeScope, GD::Upstream, isDirect2LDS, -1);
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
                           isDirect2LDS,
                           -1,
                           false);
                return;
            }

            if(hasUnroll)
            {
                log->debug("  staged as: hasUnroll");

                auto kernel = *kgraph.control.roots().begin();
                stageChain(kgraph, target, candidate, kernel, GD::Downstream, isDirect2LDS, -1);
                return;
            }

            if(isUniformLoop)
            {
                auto forLoop = *maybeForLoop;
                log->debug("  staged as: uniformForLoop, forLoopOp {}", forLoop);

                stageChain(kgraph, target, candidate, forLoop, GD::Upstream, isDirect2LDS, forLoop);
                return;
            }

            log->debug("  staged as: immediate");
            stageChain(kgraph, target, candidate, candidate, GD::Upstream, isDirect2LDS);
        }

        KernelGraph commit(KernelGraph const& original) const
        {
            auto               kgraph = original;
            std::map<int, int> scopes;
            BufferMap          bufferMap;

            for(auto const& [spec, candidates] : m_chains)
            {
                ExpressionPtr step = Expression::literal(1u);
                if(spec.forLoop > 0)
                {
                    auto [lhs, rhs] = getForLoopIncrement(kgraph, spec.forLoop);
                    step            = simplify(rhs);
                }

                auto isDirect2LDS
                    = (original.control.get<LoadTileDirect2LDS>(candidates[0]).has_value()
                       && original.coordinates.get<LDS>(spec.target).has_value());

                // Use first candidate to compute indexes
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::AddComputeIndex()::commit({}) isDirect2LDS({})",
                    candidates[0],
                    isDirect2LDS);

                auto chain = addComputeIndex(
                    kgraph, candidates[0], step, spec.location, bufferMap, isDirect2LDS);

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
                        kgraph.control.addElement(Body(), {scope}, {chain.top});
                        kgraph.control.addElement(Sequence(), {chain.bottom}, {spec.location});
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
    };

    KernelGraph AddComputeIndex::apply(KernelGraph const& original)
    {
        AddComputeIndexer indexer;

        for(auto candidate :
            findComputeIndexCandidates(original, *original.control.roots().begin()))
        {
            indexer.stage(original, candidate, false);
            auto isDirect2LDS = original.control.get<LoadTileDirect2LDS>(candidate).has_value();
            if(isDirect2LDS)
                indexer.stage(original, candidate, true);
        }

        return indexer.commit(original);
    }
}
