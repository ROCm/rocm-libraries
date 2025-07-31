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

#include <rocRoller/KernelGraph/Utils.hpp>

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>

#include <rocRoller/DataTypes/DataTypes.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {

        using GD = Graph::Direction;

        namespace CG = rocRoller::KernelGraph::ControlGraph;
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        /***********************************
         * Helpers
         */

        std::string toString(UnrollColouring const& colouring)
        {
            std::stringstream os;

            auto colourToString = [](auto colour) -> std::string {
                std::stringstream os;
                os << "[ ";
                for(auto [coord, value] : colour)
                {
                    os << "(" << coord << ", " << value << "), ";
                }
                os << "]";
                return os.str();
            };

            os << "Coordinate colours:" << std::endl;
            for(auto c : colouring.coordinateColour)
                os << "  " << c.first << ": " << colourToString(c.second) << std::endl;
            os << "Operation colours:" << std::endl;
            for(auto c : colouring.operationColour)
                os << "  " << c.first << ": " << colourToString(c.second) << std::endl;

            return os.str();
        }

        UnrollColouring colourByUnrollValue(KernelGraph const&             graph,
                                            int                            topOp,
                                            std::unordered_set<int> const& exclude)
        {
            UnrollColouring rv;

            if(topOp == -1)
                topOp = only(graph.control.roots()).value();

            auto bodies
                = graph.control.getOutputNodeIndices<CG::Body>(topOp).to<std::unordered_set>();

            //
            // First, look for SetCoordinate nodes and compute their colour.
            //
            std::map<int, std::pair<int, int>> setCoordinateColour;
            for(auto bodyTop : bodies)
            {
                for(auto setCoordTag :
                    filter(graph.control.isElemType<CG::SetCoordinate>(),
                           graph.control.depthFirstVisit(bodyTop, GD::Downstream)))
                {
                    auto setCoord   = graph.control.get<CG::SetCoordinate>(setCoordTag).value();
                    auto coordinate = graph.mapper.get<CT::Unroll>(setCoordTag);
                    if(coordinate == -1)
                        continue;

                    if(exclude.contains(coordinate))
                        continue;

                    if(!evaluationTimes(
                           setCoord.value)[rocRoller::Expression::EvaluationTime::Translate])
                        continue;

                    setCoordinateColour[setCoordTag]
                        = {coordinate, getUnsignedInt(evaluate(setCoord.value))};
                }
            }

            if(setCoordinateColour.empty())
                return rv;

            //
            // Next, colour SetCoordinate body operations, and any
            // coordinates that they are mapped to.
            //
            for(auto [setCoordinate, colouring] : setCoordinateColour)
            {
                auto [coord, value] = colouring;
                for(auto op : graph.control.depthFirstVisit(
                        setCoordinate, graph.control.isElemType<CG::Body>(), GD::Downstream))
                {
                    rv.operationColour[op][coord] = value;

                    Log::trace("colourByUnrollValue::SetCoordinate explicit operation {} unroll {} "
                               "colour {}",
                               op,
                               coord,
                               value);
                }
            }

            //
            // Now follow traces and propagate colour
            //
            auto trace = ControlFlowRWTracer(graph, topOp).coordinatesReadWrite();
            for(auto record : trace)
            {
                if(record.rw == ControlFlowRWTracer::ReadWrite::READ)
                {
                    if(!rv.coordinateColour.contains(record.coordinate))
                        continue;

                    for(auto [coord, value] : rv.coordinateColour[record.coordinate])
                    {
                        if(!rv.operationColour[record.control].contains(coord))
                        {
                            rv.operationColour[record.control][coord] = value;
                            Log::trace("colourByUnrollValue::operationColour READ operation {} "
                                       "coordinate {} "
                                       "unroll {} "
                                       "colour {}",
                                       record.control,
                                       record.coordinate,
                                       coord,
                                       value);
                        }
                    }
                }
                else
                {
                    // WRITE or READWRITE (in the case of READWRITE,
                    // the operation should have a colour already)
                    if(!rv.operationColour.contains(record.control))
                        continue;

                    for(auto [coord, value] : rv.operationColour[record.control])
                    {
                        rv.coordinateColour[record.coordinate][coord] = value;

                        Log::trace("colourByUnrollValue::coordinateColour WRITE/READWRITE "
                                   "operation {} coordinate {} unroll {} colour {}",
                                   record.control,
                                   record.coordinate,
                                   coord,
                                   value);
                    }
                }
            }

            //
            // Also, propagate colour up SetCoordinate-chains
            //
            for(auto [setCoordinate, _ignore] : setCoordinateColour)
            {
                // Go down Body edges
                int tag = setCoordinate;
                while(true)
                {
                    auto child = only(graph.control.getOutputNodeIndices<CG::Body>(tag));
                    if(!child)
                        break;
                    tag = child.value();
                }

                for(auto [coord, value] : rv.operationColour[tag])
                    rv.operationColour[setCoordinate][coord] = value;
            }

            //
            // Find Sequence separator edges
            //
            for(auto [bodyElem, _ignore] : rv.operationColour)
            {
                for(auto edge : filter(graph.control.isElemType<CG::Sequence>(),
                                       graph.control.getNeighbours<GD::Downstream>(bodyElem)))
                {
                    auto otherElem
                        = only(graph.control.getNeighbours<GD::Downstream>(edge)).value();

                    if(rv.operationColour.contains(otherElem)
                       && rv.operationColour[otherElem] != rv.operationColour[bodyElem])
                        rv.separators.insert(edge);
                }
            }

            return rv;
        }

        /**
         * Create a range-based for loop.
         */
        std::pair<int, int> rangeFor(KernelGraph&              graph,
                                     Expression::ExpressionPtr size,
                                     const std::string&        loopName,
                                     VariableType              vtype,
                                     int                       forLoopCoord)
        {
            auto sizeDataType
                = vtype == DataType::None ? Expression::resultVariableType(size) : vtype;

            auto unitStride = Expression::literal(1, sizeDataType);

            auto rangeK = graph.coordinates.addElement(CT::Linear(size, unitStride));

            int dimK = forLoopCoord;
            if(forLoopCoord <= 0)
                dimK = graph.coordinates.addElement(CT::ForLoop(size, unitStride));

            auto exprK = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{rangeK, Register::Type::Scalar, sizeDataType});

            auto forK  = graph.control.addElement(CG::ForLoopOp{exprK < size, loopName});
            auto initK = graph.control.addElement(
                CG::Assign{Register::Type::Scalar, Expression::literal(0, sizeDataType)});
            auto incrementK
                = graph.control.addElement(CG::Assign{Register::Type::Scalar, exprK + unitStride});

            graph.coordinates.addElement(CT::DataFlow(), {rangeK}, {dimK});
            graph.control.addElement(CG::Initialize(), {forK}, {initK});
            graph.control.addElement(CG::ForLoopIncrement(), {forK}, {incrementK});

            graph.mapper.connect(forK, rangeK, NaryArgument::DEST);
            graph.mapper.connect<CT::ForLoop>(forK, dimK);
            graph.mapper.connect(initK, rangeK, NaryArgument::DEST);
            graph.mapper.connect(incrementK, rangeK, NaryArgument::DEST);

            return {dimK, forK};
        }

        int cloneForLoop(KernelGraph& graph, int tag)
        {
            auto maybeForLoopOp = graph.control.get<CG::ForLoopOp>(tag);
            AssertFatal(maybeForLoopOp, "cloneForLoop is being called on a non-ForLoopOp");

            auto forLoopDim = graph.mapper.get<CT::ForLoop>(tag);

            auto forLoopSize = graph.coordinates.get<CT::ForLoop>(forLoopDim)->size;

            auto clone = rangeFor(
                graph, forLoopSize, maybeForLoopOp->loopName, DataType::None, forLoopDim);

            return clone.second;
        }

        std::pair<int, int> getForLoopCoords(int forLoopOp, KernelGraph const& kgraph)
        {
            auto range   = kgraph.mapper.get(forLoopOp, NaryArgument::DEST);
            auto forLoop = kgraph.mapper.get<CT::ForLoop>(forLoopOp);
            return {forLoop, range};
        }

        int getDEST(KernelGraph const& kgraph, int assign)
        {
            auto dst = only(kgraph.mapper.getConnections(assign));
            if(!dst)
                return -1;
            return dst->coordinate;
        }

        std::pair<Expression::ExpressionPtr, Expression::ExpressionPtr>
            getForLoopIncrement(KernelGraph const& graph, int forLoop)
        {
            // Find the ForLoopIcrement calculation

            // Grab all for loop increments from current for loop.
            // ForLoops coming from Compute Index may have more than one loop increment.
            // The forLoopIncrement that satifies all of the following conditions will be the
            // Increment that actually updates the iterator.
            auto loopIncrements = graph.control.getOutputNodeIndices<CG::ForLoopIncrement>(forLoop)
                                      .to<std::vector>();
            for(auto const& increment : loopIncrements)
            {
                auto loopIncrementOp = graph.control.getNode<CG::Assign>(increment);

                //Ensure that the forLoopIncrement has an add expression
                if(!(std::holds_alternative<Expression::Add>(*loopIncrementOp.expression)))
                    continue;
                auto addExpr = std::get<Expression::Add>(*loopIncrementOp.expression);

                auto dim_tag = graph.mapper.get(increment, NaryArgument::DEST);
                //Iterator should have a DataFlow expression as its LHS
                if(!(std::holds_alternative<Expression::DataFlowTag>(*addExpr.lhs)))
                    continue;
                //LHS should also be the loop iterator data flow tag.
                if(std::get<Expression::DataFlowTag>(*addExpr.lhs).tag != dim_tag)
                    continue;
                //If all else is true and the first connection of the forLoop is the dim_tag
                //Then we have the loopIncrement that we were searching for.
                if(graph.mapper.get(forLoop, NaryArgument::DEST) != dim_tag)
                    continue;
                return {addExpr.lhs, addExpr.rhs};
            }

            // There should be a loopIncrement that satisfies the above conditions
            // if not then throw an error.
            Throw<FatalError>("No forLoopIncrement for supplied forLoop.");
        }

        int replaceWith(KernelGraph& graph, int op, int newOp, bool includeBody)
        {
            auto& ctrl     = graph.control;
            auto  location = ctrl.getLocation(op);

            for(auto const& input : location.incoming)
            {
                auto parent    = *only(ctrl.getNeighbours<Graph::Direction::Upstream>(input));
                auto edge      = ctrl.getElement(input);
                auto maybeBody = ctrl.get<CG::Body>(input);

                if(maybeBody)
                {
                    ctrl.deleteElement(input);
                    auto updatedBodyParents
                        = ctrl.getInputNodeIndices<CG::Body>(newOp).to<std::unordered_set>();
                    if(!updatedBodyParents.contains(parent))
                    {
                        ctrl.addElement(edge, {parent}, {newOp});
                    }
                }
                else
                {
                    ctrl.deleteElement(input);
                    ctrl.addElement(edge, {parent}, {newOp});
                }
            }

            for(auto const& output : location.outgoing)
            {
                auto child = *only(ctrl.getNeighbours<Graph::Direction::Downstream>(output));
                auto edge  = ctrl.getElement(output);
                auto maybeSequence = ctrl.get<CG::Sequence>(output);
                auto maybeBody     = ctrl.get<CG::Body>(output);

                if(maybeSequence)
                {
                    ctrl.deleteElement(output);
                    ctrl.addElement(edge, {newOp}, {child});
                }
                else if(includeBody && maybeBody)
                {
                    ctrl.deleteElement(output);
                    auto updatedBodyChildren
                        = ctrl.getOutputNodeIndices<CG::Body>(newOp).to<std::unordered_set>();
                    if(!updatedBodyChildren.contains(child))
                        ctrl.addElement(edge, {newOp}, {child});
                }
            }

            return newOp;
        }

        void insertBefore(KernelGraph& graph, int op, int top, int bottom)
        {
            auto location = graph.control.getLocation(op);
            for(auto const& input : location.incoming)
            {
                auto edge = graph.control.getElement(input);
                int  parent
                    = *graph.control.getNeighbours<Graph::Direction::Upstream>(input).begin();
                graph.control.deleteElement(input);
                if(graph.control.getInputNodeIndices<CG::Body>(top).to<std::unordered_set>().count(
                       parent)
                   == 0)
                {
                    graph.control.addElement(edge, {parent}, {top});
                }
            }
            graph.control.addElement(CG::Sequence(), {bottom}, {op});
        }

        void insertAfter(KernelGraph& graph, int op, int top, int bottom)
        {
            auto location = graph.control.getLocation(op);
            for(auto const& output : location.outgoing)
            {
                auto maybeBody = graph.control.get<CG::Body>(output);
                if(maybeBody)
                    continue;
                auto edge = graph.control.getElement(output);
                int  child
                    = *graph.control.getNeighbours<Graph::Direction::Downstream>(output).begin();
                graph.control.deleteElement(output);
                graph.control.addElement(edge, {bottom}, {child});
            }
            graph.control.addElement(CG::Sequence(), {op}, {top});
        }

        void insertWithBody(KernelGraph& graph, int op, int newOp)
        {
            auto location = graph.control.getLocation(op);
            for(auto const& input : location.incoming)
            {
                auto edge = graph.control.getElement(input);
                int  parent
                    = *graph.control.getNeighbours<Graph::Direction::Upstream>(input).begin();
                graph.control.deleteElement(input);
                if(graph.control.getInputNodeIndices<CG::Body>(newOp)
                       .to<std::unordered_set>()
                       .count(parent)
                   == 0)
                {
                    graph.control.addElement(edge, {parent}, {newOp});
                }
            }
            graph.control.addElement(CG::Body(), {newOp}, {op});
        }

        bool needsComputeIndex(CG::Operation const& op)
        {
            if(std::holds_alternative<CG::StoreTiled>(op) //
               || std::holds_alternative<CG::StoreLDSTile>(op) //
               || std::holds_alternative<CG::LoadTiled>(op) //
               || std::holds_alternative<CG::LoadLDSTile>(op) //
               || std::holds_alternative<CG::LoadTileDirect2LDS>(op))
                return true;
            return false;
        }

        std::vector<int> findComputeIndexCandidates(KernelGraph const& kgraph, int start)
        {
            std::vector<int> rv;

            return kgraph.control
                .findNodes(
                    start,
                    [&](int tag) -> bool {
                        auto elem = kgraph.control.getElement(tag);
                        if(!std::holds_alternative<CG::Operation>(elem))
                            return false;
                        auto op = std::get<CG::Operation>(elem);
                        return needsComputeIndex(op);
                    },
                    GD::Downstream)
                .to<std::vector>();
        }

        void purgeFor(KernelGraph& kgraph, int loop)
        {
            // Get loop dimension and iterator first (before purging the operator)
            auto [forLoop, forIncr] = getForLoopCoords(loop, kgraph);

            // Purge the operator
            purgeNodeAndChildren(kgraph, loop);

            // If there is still a connection to the increment, a
            // similar loop exists elsewhere in the graph.
            if(!kgraph.mapper.getCoordinateConnections(forLoop).empty())
                return;

            //
            // Purge loop dimension and all incoming/outgoing edge
            //

            // Build list of candidate edges to purge
            std::vector<std::pair<int, Graph::Direction>> purgeCandidates;
            {
                auto location = kgraph.coordinates.getLocation(forLoop);
                for(auto tag : location.incoming)
                    purgeCandidates.push_back({tag, Graph::Direction::Upstream});
                for(auto tag : location.outgoing)
                    purgeCandidates.push_back({tag, Graph::Direction::Downstream});
            }

            // Purge edges if they are DataFlow or PassThrough edges
            for(auto [edgeTag, direction] : purgeCandidates)
            {
                auto maybePassThrough = kgraph.coordinates.get<CT::PassThrough>(edgeTag);
                auto maybeDataFlow    = kgraph.coordinates.get<CT::DataFlow>(edgeTag);
                if(maybePassThrough)
                {
                    // If it's a PassThrough edge, purge the coordinate
                    auto coordTag = only(kgraph.coordinates.getNeighbours(edgeTag, direction));
                    if(coordTag)
                    {
                        kgraph.coordinates.deleteElement(*coordTag);
                        kgraph.mapper.purgeMappingsTo(*coordTag);
                    }
                }
                if(maybePassThrough || maybeDataFlow)
                    kgraph.coordinates.deleteElement(edgeTag);
            }

            kgraph.coordinates.deleteElement(forIncr);
            kgraph.mapper.purgeMappingsTo(forIncr);

            kgraph.coordinates.deleteElement(forLoop);
            kgraph.mapper.purgeMappingsTo(forLoop);
        }

        void purgeNodeAndChildren(KernelGraph& kgraph, int node)
        {
            for(auto const& reap : kgraph.control.depthFirstVisit(node).to<std::vector>())
            {
                kgraph.control.deleteElement(reap);
                kgraph.mapper.purge(reap);
            }
            kgraph.mapper.purge(node);
        }

        bool isHardwareCoordinate(int tag, KernelGraph const& kgraph)
        {
            return kgraph.coordinates.get<CT::VGPR>(tag)
                   || kgraph.coordinates.get<CT::Workitem>(tag)
                   || kgraph.coordinates.get<CT::Workgroup>(tag);
        }

        bool isLoopishCoordinate(int tag, KernelGraph const& kgraph)
        {
            return kgraph.coordinates.get<CT::ForLoop>(tag)
                   || kgraph.coordinates.get<CT::Unroll>(tag);
        }

        bool isStorageCoordinate(int tag, KernelGraph const& kgraph)
        {
            return kgraph.coordinates.get<CT::LDS>(tag) || kgraph.coordinates.get<CT::User>(tag);
        }

        std::pair<int, Graph::Direction>
            getOperationTarget(int tag, KernelGraph const& kgraph, bool isDirect2LDS)
        {
            auto elem = kgraph.control.getElement(tag);
            if(isDirect2LDS)
            {
                return {kgraph.mapper.get<CT::LDS>(tag), GD::Upstream};
            }

            using result = std::pair<int, Graph::Direction>;

            return std::visit(
                rocRoller::overloaded{
                    [&](CIsAnyOf<CG::StoreTiled, CG::StoreVGPR, CG::StoreSGPR> auto const& op)
                        -> result {
                        return {kgraph.mapper.get<CT::User>(tag), GD::Upstream};
                    },
                    [&](CIsAnyOf<CG::LoadTiled,
                                 CG::LoadVGPR,
                                 CG::LoadSGPR,
                                 CG::LoadTileDirect2LDS> auto const& op) -> result {
                        return {kgraph.mapper.get<CT::User>(tag), GD::Downstream};
                    },
                    [&](CG::StoreLDSTile const& op) -> result {
                        return {kgraph.mapper.get<CT::LDS>(tag), GD::Upstream};
                    },
                    [&](CG::LoadLDSTile const& op) -> result {
                        return {kgraph.mapper.get<CT::LDS>(tag), GD::Downstream};
                    },
                    [&](CG::Assign const& op) -> result {
                        return {kgraph.mapper.getConnections(tag)[0].coordinate, GD::Downstream};
                    },
                    [&](auto const& op) -> result {
                        Throw<FatalError>(
                            "Operation is not a load, store, or assign: ", tag, " ", toString(op));
                        return {0, GD::Downstream};
                    }},
                std::get<CG::Operation>(elem));
        }

        int getTransformTarget(int storageTarget, KernelGraph const& kgraph)
        {
            auto isDuplicate = CT::isEdge<CT::Duplicate>;
            auto outbound    = kgraph.coordinates.getOutputNodeIndices(storageTarget, isDuplicate)
                                .to<std::vector>();

            if(outbound.empty())
                return storageTarget;

            AssertFatal(outbound.size() == 1,
                        "Only one outbound Duplicate edge is supported.",
                        ShowValue(outbound));

            return outbound[0];
        }

        std::pair<std::vector<int>, std::unordered_set<int>> findRequiredCoordinates(
            int target, Graph::Direction direction, KernelGraph const& kgraph)
        {
            return findRequiredCoordinates(
                target, direction, [](int tag) { return false; }, kgraph);
        }

        std::pair<std::vector<int>, std::unordered_set<int>>
            findRequiredCoordinates(int                      target,
                                    Graph::Direction         direction,
                                    std::function<bool(int)> fullStop,
                                    KernelGraph const&       kgraph)
        {
            // TODO: Design a better way of binding storage to coordinates
            auto maybeLDS = kgraph.coordinates.get<CT::LDS>(target);
            if(maybeLDS)
            {
                // If target is LDS; it might be a duplicated LDS
                // node.  For the purposes of figuring out required
                // coordinates, use the parent LDS as the target
                // instead.
                auto maybeParentLDS = only(
                    kgraph.coordinates.getOutputNodeIndices(target, CT::isEdge<CT::Duplicate>));
                if(maybeParentLDS)
                    target = *maybeParentLDS;
            }

            // First, construct the set of elements reachable from the
            // target.
            auto reachable
                = kgraph.coordinates.depthFirstVisit(target, direction).to<std::unordered_set>();

            // Next, from the target coordinate, walk the graph but
            // don't traverse all edges.  This will result in a list
            // of nodes that are used in the coordinate transform to
            // compute indexes for the target coordinate.
            //
            // The edge predicate is called on edges.  It looks in the
            // direction opposite the traversal direction and examines
            // reachable nodes.
            //
            // If the edge is not a coordinate-transform edge, then we
            // don't traverse it.
            //
            // If all reachable nodes are storage or loopish, then we
            // don't traverse the edge.
            //
            // If any of the reachable nodes is marked as "full stop",
            // then we don't traverse the edge.
            auto edgePredicate = [&](int tag) -> bool {
                auto element = kgraph.coordinates.getElement(tag);
                auto edge    = std::get<CT::Edge>(element);

                bool isCT = std::holds_alternative<CT::CoordinateTransformEdge>(edge);
                if(!isCT)
                    return false;

                bool allReachableNeighboursAreBad = true;
                for(auto neighbour : kgraph.coordinates.getNeighbours(tag, opposite(direction)))
                {
                    if(!reachable.contains(neighbour))
                        continue;

                    if(neighbour == target)
                        allReachableNeighboursAreBad = false;

                    if(!(isLoopishCoordinate(neighbour, kgraph)
                         || isStorageCoordinate(neighbour, kgraph)))
                        allReachableNeighboursAreBad = false;

                    if(fullStop(neighbour))
                        return false;
                }
                return !allReachableNeighboursAreBad;
            };

            auto candidates = kgraph.coordinates.depthFirstVisit(target, edgePredicate, direction)
                                  .to<std::vector>();

            // Internal nodes in the coordinate transform are computed
            // as part of the transform, so only leaf nodes and/or
            // hardware/loop coordinates are required.
            std::vector<int> required;
            std::copy_if(
                candidates.cbegin(), candidates.cend(), std::back_inserter(required), [&](int tag) {
                    bool isLeaf = true;
                    for(auto edgeTag : kgraph.coordinates.getNeighbours(tag, direction))
                    {
                        auto edge = kgraph.coordinates.getEdge(edgeTag);
                        if(std::holds_alternative<CT::CoordinateTransformEdge>(edge))
                        {
                            // If a connected node, in the opposing
                            // direction, is marked as "full stop",
                            // this edge doesn't count.
                            bool ignoreEdge = false;
                            for(auto neighbour :
                                kgraph.coordinates.getNeighbours(edgeTag, opposite(direction)))
                            {
                                if(fullStop(neighbour))
                                    ignoreEdge = true;
                            }
                            if(!ignoreEdge)
                                isLeaf = false;
                        }
                    }

                    bool isLeafy
                        = isHardwareCoordinate(tag, kgraph) || isLoopishCoordinate(tag, kgraph);

                    return isLeaf || isLeafy;
                });

            std::unordered_set<int> path;
            if(direction == Graph::Direction::Downstream)
            {
                path = kgraph.coordinates
                           .path<Graph::Direction::Upstream>(required, std::vector<int>{target})
                           .to<std::unordered_set>();
            }
            else
            {
                path = kgraph.coordinates
                           .path<Graph::Direction::Downstream>(required, std::vector<int>{target})
                           .to<std::unordered_set>();
            }

            return {required, path};
        }

        std::pair<std::unordered_set<int>, std::unordered_set<int>>
            findAllRequiredCoordinates(int tag, KernelGraph const& graph)
        {
            std::unordered_set<int> required;

            auto [target, direction] = getOperationTarget(tag, graph);
            Log::debug("{} target: {}", tag, target);
            auto [targetRequired, path] = findRequiredCoordinates(target, direction, graph);

            std::copy(targetRequired.cbegin(),
                      targetRequired.cend(),
                      std::inserter(required, required.end()));

            return {required, path};
        }

        std::unordered_set<int> includeEdgeNeighbours(
            rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph const& coordinates,
            Graph::Direction                                                direction,
            std::unordered_set<int> const&                                  path)
        {
            std::unordered_set<int> rv = path;

            for(auto elem : path)
            {
                if(coordinates.getElementType(elem) != Graph::ElementType::Edge)
                    continue;

                for(auto const node : coordinates.getNeighbours(elem, direction))
                    rv.insert(node);
            }
            return rv;
        }

        rocRoller::KernelGraph::CoordinateGraph::User newScratchCoordinate(
            Expression::ExpressionPtr size, VariableType varType, ContextPtr context)
        {
            auto currentOffset = context->getScratchAmount();
            auto newCoordinate = CT::User(size, currentOffset);
            // TODO Audit bytes/bits
            // Can we move size inside the CeilDivide?
            context->allocateScratch(
                size * Expression::literal(CeilDivide(DataTypeInfo::Get(varType).elementBits, 8u)));

            return newCoordinate;
        }

        std::optional<std::pair<int, Graph::Direction>>
            findStorageNeighbour(int tag, KernelGraph const& graph)
        {
            using rt = std::pair<int, Graph::Direction>;
            auto neighbourTag
                = only(graph.coordinates.getNeighbours(tag, Graph::Direction::Upstream));
            if(neighbourTag && isStorageCoordinate(*neighbourTag, graph))
            {
                return rt{*neighbourTag, Graph::Direction::Downstream};
            }
            neighbourTag = only(graph.coordinates.getNeighbours(tag, Graph::Direction::Downstream));
            if(neighbourTag && isStorageCoordinate(*neighbourTag, graph))
            {
                return rt{*neighbourTag, Graph::Direction::Upstream};
            }
            return {};
        }

        std::optional<int> findUnrollNeighbour(KernelGraph const& kgraph, int forLoopCoord)
        {
            if(forLoopCoord < 0)
                return {};

            std::optional<int> rv;

            auto forNeighbours
                = kgraph.coordinates.getNeighbours<GD::Upstream>(forLoopCoord).to<std::vector>();
            for(auto forNeighbour : forNeighbours)
            {
                auto split = kgraph.coordinates.get<CT::Split>(forNeighbour);
                if(split)
                {
                    auto splitNeighbours
                        = kgraph.coordinates.getNeighbours<GD::Downstream>(forNeighbour)
                              .to<std::vector>();
                    for(auto splitNeighbour : splitNeighbours)
                    {
                        auto unroll = kgraph.coordinates.get<CT::Unroll>(splitNeighbour);
                        if(unroll)
                        {
                            AssertFatal(!rv || rv == splitNeighbour,
                                        "More than one Unroll neighbour found.");
                            rv = splitNeighbour;
                        }
                    }
                }
            }

            return rv;
        }

        void duplicateMacroTile(KernelGraph& graph, int load)
        {
            TIMER(t, "duplicateMacroTile");
            auto original = graph.mapper.get<CT::MacroTile>(load);
            auto newMacroTile
                = graph.coordinates.addElement(graph.coordinates.getElement(original));
            graph.coordinates.addElement(CT::Duplicate(), {newMacroTile}, {original});
            graph.mapper.disconnect<CT::MacroTile>(load, original);
            graph.mapper.connect<CT::MacroTile>(load, newMacroTile);
        }

        int duplicateControlNode(KernelGraph& graph, int tag)
        {
            auto op = graph.control.addElement(graph.control.getElement(tag));
            for(auto const& c : graph.mapper.getConnections(tag))
            {
                graph.mapper.connect(op, c.coordinate, c.connection);
            }

            return op;
        }

        void deleteControlNode(KernelGraph& graph, int nodeIdx)
        {
            using namespace rocRoller::KernelGraph::ControlGraph;

            {
                auto incomingNodes
                    = graph.control.getInputNodeIndices<ControlEdge>(nodeIdx).to<std::vector>();
                for(auto inc : incomingNodes)
                    graph.control.deleteElement(graph.control.findEdge(inc, nodeIdx).value());
            }

            {
                auto outgoingNodes
                    = graph.control.getOutputNodeIndices<ControlEdge>(nodeIdx).to<std::vector>();
                for(auto out : outgoingNodes)
                    graph.control.deleteElement(graph.control.findEdge(nodeIdx, out).value());
            }

            graph.control.deleteElement(nodeIdx);
            graph.mapper.purge(nodeIdx);
        }

        void updateThreadTileForLongDwords(int& t_m,
                                           int& t_n,
                                           int  maxWidth,
                                           uint macTileFastMovingDimSize,
                                           int  numDwordsPerElement)
        {
            auto numDwordsPerWorkitem = t_m * numDwordsPerElement;

            std::vector<int> potentialFactors = {4, 3, 2, 1};

            auto start = potentialFactors.begin();
            auto end   = potentialFactors.end();
            auto factorPred
                = [numDwordsPerWorkitem, maxWidth, t_n, macTileFastMovingDimSize](int factor) {
                      const auto n = t_n * factor;
                      return factor <= maxWidth && numDwordsPerWorkitem % factor == 0
                             && (n <= macTileFastMovingDimSize);
                  };
            auto it = std::find_if(start, end, factorPred);

            if(it != potentialFactors.end())
            {
                auto dwordFactor = *it / numDwordsPerElement;
                AssertFatal(dwordFactor >= 1, "dword factor can't be less than 1");

                t_m = t_m / dwordFactor;
                t_n = t_n * dwordFactor;
            }
        }

        std::set<int> getContainingSetCoordinates(KernelGraph const& graph, int node)
        {
            std::set<int> result;
            int           tag = node;

            while(true)
            {
                auto parent = only(graph.control.getInputNodeIndices<CG::Body>(tag));
                if(!parent)
                    break;

                auto setCoord = graph.control.get<CG::SetCoordinate>(*parent);
                if(!setCoord)
                    break;

                tag = *parent;

                AssertFatal(graph.mapper.get<CT::Unroll>(tag) > 0,
                            "SetCoordinate needs Unroll dimension");

                result.insert(tag);
            }

            return result;
        }

        int getTopSetCoordinate(KernelGraph const& graph, int load)
        {
            TIMER(t, "getTopSetCoordinate");
            int tag = load;

            while(true)
            {
                auto parent = only(graph.control.getInputNodeIndices<CG::Body>(tag));
                if(!parent)
                    break;

                auto setCoord = graph.control.get<CG::SetCoordinate>(*parent);
                if(setCoord)
                    tag = *parent;
                else
                    break;

                AssertFatal(graph.mapper.get<CT::Unroll>(tag) > 0,
                            "SetCoordinate needs Unroll dimension");
            }
            return tag;
        }

        std::set<int> getTopSetCoordinates(KernelGraph& graph, std::vector<int> loads)
        {
            std::set<int> retval;
            for(auto& load : loads)
            {
                retval.insert(getTopSetCoordinate(graph, load));
            }
            return retval;
        }

        int getSetCoordinateForDim(KernelGraph const& graph, int dim, int load)
        {
            int tag = load;

            while(true)
            {
                auto parent = findContainingOperation<CG::SetCoordinate>(tag, graph);
                AssertFatal(
                    parent, "Could not find a containing SetCoordinate for ", ShowValue(tag));

                tag = *parent;

                auto unroll = graph.mapper.get<CT::Unroll>(tag);
                AssertFatal(unroll > 0, "SetCoordinate needs Unroll dimension");

                if(unroll == dim)
                    return tag;
            }
        }

        bool
            hasExistingSetCoordinate(KernelGraph const& graph, int op, int coordValue, int coordTag)
        {
            int tag = op;

            while(true)
            {
                auto parent = only(graph.control.getInputNodeIndices<CG::Body>(tag));
                if(!parent)
                    return false;

                tag           = parent.value();
                auto setCoord = graph.control.get<CG::SetCoordinate>(tag);
                if(!setCoord)
                    return false;

                auto valueExpr = setCoord.value().value;
                AssertFatal(evaluationTimes(valueExpr)[Expression::EvaluationTime::Translate],
                            "SetCoordinate::value should be a literal.");

                if(getUnsignedInt(evaluate(valueExpr)) == coordValue)
                {
                    for(auto const& dst : graph.mapper.getConnections(tag))
                    {
                        if(dst.coordinate == coordTag)
                            return true;
                    }
                }
            }
        }

        unsigned int getUnrollValueForOp(KernelGraph const& graph, int unrollDim, int op)
        {
            auto setCoordTag = getSetCoordinateForDim(graph, unrollDim, op);
            auto setCoord    = graph.control.get<CG::SetCoordinate>(setCoordTag);

            auto valueExpr = setCoord.value().value;

            AssertFatal(evaluationTimes(valueExpr)[Expression::EvaluationTime::Translate],
                        "Unroll value should be a literal.");

            return getUnsignedInt(evaluate(valueExpr));
        }

        VariableType getVariableType(KernelGraph const& graph, int opTag)
        {
            auto node = graph.control.getNode(opTag);
            return getVariableType(node);
        }

        void replaceMacroTile(KernelGraph&                   graph,
                              std::unordered_set<int> const& ops,
                              int                            oldMacTileTag,
                              int                            newMacTileTag)
        {
            for(auto const& opTag : ops)
            {
                auto element = graph.control.getElement(opTag);
                auto visitor = rocRoller::overloaded{
                    [&](CG::StoreTiled store) {
                        auto macroTile = graph.mapper.get<CT::MacroTile>(opTag);
                        if(macroTile == oldMacTileTag)
                        {
                            graph.mapper.disconnect<CT::MacroTile>(opTag, oldMacTileTag);
                            graph.mapper.connect<CT::MacroTile>(opTag, newMacTileTag);

                            // update the data flow in the coordinate graph
                            auto dstTag = graph.mapper.get<CT::User>(opTag);
                            auto df
                                = *only(graph.coordinates.getNeighbours<Graph::Direction::Upstream>(
                                    dstTag));
                            graph.coordinates.deleteElement(df);
                            graph.coordinates.addElement(CT::DataFlow(),
                                                         std::vector<int>{newMacTileTag},
                                                         std::vector<int>{dstTag});
                        }
                    },
                    [&](CG::Assign assign) {
                        GraphReindexer reindexer;
                        reindexer.coordinates.emplace(oldMacTileTag, newMacTileTag);
                        reindexExpressions(graph, opTag, reindexer);

                        // update the data flow in the coordinate graph
                        auto assignConnection = only(graph.mapper.getConnections(opTag));
                        AssertFatal(assignConnection,
                                    "There should be exactly one connection for an assignment");
                        auto             dstTag = assignConnection->coordinate;
                        std::vector<int> srcTags;
                        for(auto const& edgeTag :
                            graph.coordinates.getNeighbours<Graph::Direction::Upstream>(dstTag))
                        {
                            auto df = graph.coordinates.get<CT::DataFlow>(edgeTag);
                            if(!df)
                                continue;
                            auto srcs = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(
                                edgeTag);
                            for(auto const src : srcs)
                            {
                                if(src == oldMacTileTag)
                                    srcTags.push_back(newMacTileTag);
                                else
                                    srcTags.push_back(src);
                            }
                            graph.coordinates.deleteElement(edgeTag);
                        }
                        graph.coordinates.addElement(
                            CT::DataFlow(),
                            srcTags,
                            std::vector<int>{dstTag == oldMacTileTag ? newMacTileTag : dstTag});

                        if(dstTag == oldMacTileTag)
                        {
                            graph.mapper.disconnect(
                                opTag, assignConnection->coordinate, assignConnection->connection);
                            graph.mapper.connect(opTag, newMacTileTag, NaryArgument::DEST);
                        }
                    },
                    [&](auto op) { Throw<FatalError>("Not handled yet."); }};

                std::visit(visitor, std::get<CG::Operation>(element));
            }
        }

        void moveConnections(rocRoller::KernelGraph::KernelGraph& kgraph,
                             int                                  op,
                             int                                  newOp,
                             int                                  subdimStride)
        {
            for(auto& c : kgraph.mapper.getConnections(op))
            {
                auto curConnection = c.connection;
                auto maybeLDSTile  = kgraph.coordinates.get<CT::LDS>(c.coordinate);
                if(maybeLDSTile || subdimStride == 0)
                {
                    kgraph.mapper.connect(newOp, c.coordinate, c.connection);
                }
                else if(std::holds_alternative<Connections::TypeAndSubDimension>(c.connection))
                {
                    auto curConnection = std::get<Connections::TypeAndSubDimension>(c.connection);
                    auto subdim        = curConnection.subdimension;
                    auto newSubdim     = subdim + subdimStride;
                    auto newConnection
                        = Connections::TypeAndSubDimension{curConnection.id, newSubdim};
                    kgraph.mapper.connect(newOp, c.coordinate, newConnection);
                }
                else
                {
                    kgraph.mapper.connect(newOp, c.coordinate, c.connection);
                }
            }
        }

        Expression::ExpressionPtr tileCeilDivide(Expression::ExpressionPtr sdSize, int tileSize)
        {
            auto tileSizeExpr = std::has_single_bit(static_cast<uint>(tileSize))
                                    ? Expression::literal(static_cast<uint>(tileSize))
                                    : Expression::literal(static_cast<int64_t>(tileSize));

            auto one = Expression::literal(1u);

            return (sdSize + tileSizeExpr - one) / tileSizeExpr;
        }

        bool hasDeallocate(const KernelGraph& graph, int registerTag)
        {
            for(auto const& connection : graph.mapper.getCoordinateConnections(registerTag))
            {
                if(std::holds_alternative<CG::Deallocate>(
                       graph.control.getNode(connection.control)))
                {
                    return true;
                }
            }

            return false;
        }

        void orderMemoryNodes(KernelGraph&                         graph,
                              std::set<std::pair<int, int>> const& pairs,
                              bool                                 ordered)
        {
            std::map<int, std::deque<int>> traces;
            for(auto pair : pairs)
            {
                traces[pair.first]  = controlStack(pair.first, graph);
                traces[pair.second] = controlStack(pair.second, graph);
            }
            for(auto pair : pairs)
            {
                if(pair.first != pair.second
                   && graph.control.compareNodes(
                          rocRoller::UseCacheIfAvailable, pair.first, pair.second)
                          == CG::NodeOrdering::Undefined)
                {
                    graph.control.orderMemoryNodes(
                        traces.at(pair.first), traces.at(pair.second), ordered);
                }
            }
        }

        void orderMemoryNodes(KernelGraph&         graph,
                              std::set<int> const& srcs,
                              std::set<int> const& dests,
                              bool                 ordered)
        {
            std::set<std::pair<int, int>> pairs;
            for(auto src : srcs)
            {
                for(auto dest : dests)
                {
                    pairs.insert(std::make_pair(src, dest));
                }
            }
            orderMemoryNodes(graph, pairs, ordered);
        }

        void orderMemoryNodes(KernelGraph& graph, std::vector<int> const& nodes, bool ordered)
        {
            std::set<std::pair<int, int>> pairs;
            // Cartesian product; mimics the std::set version above
            for(int i = 0; i < nodes.size(); ++i)
                for(int j = 0; j < nodes.size(); ++j)
                    pairs.insert(std::make_pair(nodes[i], nodes[j]));
            orderMemoryNodes(graph, pairs, ordered);
        }

        int getLDSOperationTarget(KernelGraph const& k, int opTag)
        {
            auto [target, direction] = getOperationTarget(opTag, k);

            // TODO: Design a better way of binding storage to coordinates
            auto maybeLDS = k.coordinates.get<CT::LDS>(target);
            if(maybeLDS)
            {
                // If target is LDS; it might be a duplicated LDS
                // node.  For the purposes of figuring out required
                // coordinates, use the parent LDS as the target
                // instead.
                auto maybeParentLDS
                    = only(k.coordinates.getOutputNodeIndices(target, CT::isEdge<CT::Duplicate>));
                if(maybeParentLDS)
                    target = *maybeParentLDS;
            }

            auto isDataFlow
                = [&](int tag) -> bool { return k.coordinates.get<CT::DataFlow>(tag).has_value(); };

            while(true)
            {
                auto edge
                    = only(filter(isDataFlow, k.coordinates.getNeighbours(target, direction)));
                if(!edge)
                    break;
                target = *only(k.coordinates.getNeighbours(*edge, direction));
            }

            return target;
        }

        int duplicateChain(KernelGraph& graph, std::vector<int> const& startNodes)
        {
            return duplicateControlNodes(graph, nullptr, startNodes, [](int x) { return true; })[0];
        }

        unsigned int getUnrollSize(KernelGraph const& graph, int unroll)
        {
            if(unroll == -1)
                return 1u;
            AssertFatal(graph.coordinates.get<CT::Unroll>(unroll).has_value(),
                        "The argument is not an Unroll coordinate");

            CT::Dimension unrollDim = graph.coordinates.get<CT::Unroll>(unroll).value();
            return getUnsignedInt(evaluate(getSize(unrollDim)));
        }

        /**
        * @brief Get coordinates required by the code-generator.
        */
        std::vector<int>
            getCodeGeneratorCoordinates(KernelGraph const& graph, int tag, bool isDirect2LDS)
        {
            auto [tileTag, tile] = graph.getDimension<CT::MacroTile>(tag);
            if(isDirect2LDS)
            {
                return {graph.mapper.get<CT::ElementNumber>(tag, 2),
                        graph.mapper.get<CT::ElementNumber>(tag, 3)};
            }
            if(tile.memoryType == MemoryType::VGPR || tile.memoryType == MemoryType::WAVE_SPLIT)
            {
                return {graph.mapper.get<CT::ElementNumber>(tag, 0),
                        graph.mapper.get<CT::ElementNumber>(tag, 1)};
            }
            if(tile.layoutType == LayoutType::MATRIX_A)
            {
                return {graph.mapper.get<CT::WaveTileNumber>(tag, 1),
                        graph.mapper.get<CT::VGPR>(tag)};
            }
            if(tile.layoutType == LayoutType::MATRIX_B)
            {
                return {graph.mapper.get<CT::WaveTileNumber>(tag, 0),
                        graph.mapper.get<CT::VGPR>(tag)};
            }
            if(tile.layoutType == LayoutType::MATRIX_ACCUMULATOR)
            {
                return {graph.mapper.get<CT::VGPRBlockNumber>(tag),
                        graph.mapper.get<CT::VGPRBlockIndex>(tag)};
            }

            Throw<FatalError>("getCodeGeneratorCoordinates tile type not implemented yet.");
        }

        /**
         * @brief Helper for removeRedundantBodyEdgesBaselineMethod (early return).
         */
        bool static isRedundantBodyEdge(KernelGraph const& graph, int edge)
        {

            auto tail = *only(graph.control.getNeighbours<GD::Upstream>(edge));
            auto head = *only(graph.control.getNeighbours<GD::Downstream>(edge));

            auto onlyFollowDifferentBodyEdges = [&](int x) -> bool {
                auto isSame = x == edge;
                auto isBody = CG::isEdge<CG::Body>(graph.control.getElement(x));
                return !isSame && isBody;
            };

            {
                auto reachable
                    = !graph.control
                           .depthFirstVisit(tail, onlyFollowDifferentBodyEdges, GD::Downstream)
                           .filter([head](int x) { return x == head; })
                           .empty();

                if(reachable)
                    return true;
            }

            auto otherBodies = graph.control.getOutputNodeIndices<CG::Body>(tail).filter(
                [head](int x) { return x != head; });

            for(auto top : otherBodies)
            {
                auto reachable = !graph.control
                                      .depthFirstVisit(top,
                                                       graph.control.isElemType<CG::Sequence>(),
                                                       GD::Downstream)
                                      .filter([head](int x) { return x == head; })
                                      .empty();

                if(reachable)
                    return true;
            }
            return false;
        }

        /**
         * @brief Remove redundant body edges in control graph. This is a baseline method for
         *        verifying correctness.
         */
        void removeRedundantBodyEdgesBaselineMethod(KernelGraph& graph)
        {
            auto edges = graph.control.getEdges()
                             .filter(graph.control.isElemType<CG::Body>())
                             .to<std::vector>();
            for(auto edge : edges)
            {
                if(isRedundantBodyEdge(graph, edge))
                {
                    Log::debug("Deleting redundant Body edge: {}", edge);
                    graph.control.deleteElement(edge);
                }
            }
        }

        std::deque<int> controlStack(int control, ControlGraph::ControlGraph const& graph)
        {
            TIMER(t, "controlStack");
            std::deque<int> rv = {control};

            std::unordered_set<int> visitedNodes = {control};

            // Walk up the first edge we find until we have found a root (which
            // should be the only root). Any time we find an edge that is not a
            // Sequence edge, we have found a new body-parent. This should hold for
            // any valid (walkable) control graph.

            while(auto edge
                  = graph.getNeighbours<Graph::Direction::Upstream>(control).take(1).only())
            {
                auto node = graph.getNeighbours<Graph::Direction::Upstream>(*edge).take(1).only();
                AssertFatal(node.has_value(), "Edge does not connect two nodes!");
                AssertFatal(!visitedNodes.contains(*node), "Graph contains cycle!");
                visitedNodes.insert(*node);

                auto isContaining
                    = !std::holds_alternative<ControlGraph::Sequence>(graph.getEdge(*edge));
                if(isContaining)
                    rv.push_front(*node);

                control = *node;
            }

            return rv;
        }

        std::deque<int> controlStack(int control, KernelGraph const& graph)
        {
            return controlStack(control, graph.control);
        }

    }
}
