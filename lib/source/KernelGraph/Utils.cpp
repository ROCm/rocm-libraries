
#include <rocRoller/KernelGraph/Utils.hpp>

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;

        using GD = Graph::Direction;

        using namespace CoordinateGraph;
        using namespace ControlGraph;
        using namespace Expression;

        /***********************************
         * Helpers
         */

        /**
         * Create a range-based for loop.
         */
        std::pair<int, int> rangeFor(KernelGraph&              graph,
                                     Expression::ExpressionPtr size,
                                     const std::string&        loopName,
                                     VariableType              vtype)
        {
            auto unitStride = Expression::literal(1u);
            auto rangeK     = graph.coordinates.addElement(Linear(size, unitStride));
            auto dimK       = graph.coordinates.addElement(ForLoop(size, unitStride));
            auto sizeDataType
                = vtype == DataType::None ? Expression::resultVariableType(size) : vtype;
            auto exprK = std::make_shared<Expression::Expression>(
                DataFlowTag{rangeK, Register::Type::Scalar, sizeDataType});

            auto forK  = graph.control.addElement(ForLoopOp{exprK < size, loopName});
            auto initK = graph.control.addElement(
                Assign{Register::Type::Scalar, Expression::literal(0, sizeDataType)});
            auto incrementK
                = graph.control.addElement(Assign{Register::Type::Scalar, exprK + unitStride});

            graph.coordinates.addElement(DataFlow(), {rangeK}, {dimK});
            graph.control.addElement(Initialize(), {forK}, {initK});
            graph.control.addElement(ForLoopIncrement(), {forK}, {incrementK});

            graph.mapper.connect<Dimension>(forK, rangeK);
            graph.mapper.connect(initK, rangeK, NaryArgument::DEST);
            graph.mapper.connect(incrementK, rangeK, NaryArgument::DEST);

            return {dimK, forK};
        }

        int getForLoop(int forLoopOp, KernelGraph const& kgraph)
        {
            namespace CG = rocRoller::KernelGraph::CoordinateGraph;

            auto range = kgraph.mapper.getConnections(forLoopOp)[0].coordinate;
            auto forLoop
                = only(kgraph.coordinates.getOutputNodeIndices(range, CG::isEdge<CG::DataFlow>));
            return *forLoop;
        }

        std::pair<Expression::ExpressionPtr, Expression::ExpressionPtr>
            getForLoopIncrement(KernelGraph const& graph, int forLoop)
        {
            // Find the ForLoopIcrement calculation

            // Grab all for loop increments from current for loop.
            // ForLoops coming from Compute Index may have more than one loop increment.
            // The forLoopIncrement that satifies all of the following conditions will be the
            // Increment that actually updates the iterator.
            auto loopIncrements
                = graph.control.getOutputNodeIndices<ForLoopIncrement>(forLoop).to<std::vector>();
            for(auto const& increment : loopIncrements)
            {
                auto loopIncrementOp = graph.control.getNode<Assign>(increment);

                //Ensure that the forLoopIncrement has an add expression
                if(!(std::holds_alternative<Expression::Add>(*loopIncrementOp.expression)))
                    continue;
                auto addExpr = std::get<Expression::Add>(*loopIncrementOp.expression);

                auto connections = graph.mapper.getConnections(increment);
                //Iterator should have one connection, if it doesn't it's not connected to coordinate.
                if(connections.size() != 1)
                    continue;
                auto dim_tag = connections[0].coordinate;
                //Iterator should have a DataFlow expression as its LHS
                if(!(std::holds_alternative<Expression::DataFlowTag>(*addExpr.lhs)))
                    continue;
                //LHS should also be the loop iterator data flow tag.
                if(std::get<Expression::DataFlowTag>(*addExpr.lhs).tag != dim_tag)
                    continue;
                //If all else is true and the first connection of the forLoop is the dim_tag
                //Then we have the loopIncrement that we were searching for.
                if(graph.mapper.getConnections(forLoop)[0].coordinate != dim_tag)
                    continue;
                return {addExpr.lhs, addExpr.rhs};
            }
            // There should be a loopIncrement that satisfies the above conditions
            // if not then throw an error.
            throw FatalError("No forLoopIncrement for supplied forLoop.");
        }

        int replaceWith(KernelGraph& graph, int op, int newOp, bool includeBody)
        {
            auto location = graph.control.getLocation(op);
            for(auto const& input : location.incoming)
            {
                auto edge = graph.control.getElement(input);
                int  parent
                    = *graph.control.getNeighbours<Graph::Direction::Upstream>(input).begin();
                graph.control.deleteElement(input);
                if(graph.control.getInputNodeIndices<Body>(newOp).to<std::unordered_set>().count(
                       parent)
                   == 0)
                {
                    graph.control.addElement(edge, {parent}, {newOp});
                }
            }
            for(auto const& output : location.outgoing)
            {
                auto edge = graph.control.getElement(output);
                if(std::holds_alternative<ControlEdge>(edge))
                {
                    auto cedge = std::get<ControlEdge>(edge);
                    if(std::holds_alternative<Sequence>(cedge)
                       || (includeBody && std::holds_alternative<Body>(cedge)))
                    {
                        int child
                            = *graph.control.getNeighbours<Graph::Direction::Downstream>(output)
                                   .begin();
                        graph.control.deleteElement(output);
                        if(graph.control.getOutputNodeIndices<Body>(newOp)
                               .to<std::unordered_set>()
                               .count(child)
                           == 0)
                        {
                            graph.control.addElement(edge, {newOp}, {child});
                        }
                    }
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
                if(graph.control.getInputNodeIndices<Body>(top).to<std::unordered_set>().count(
                       parent)
                   == 0)
                {
                    graph.control.addElement(edge, {parent}, {top});
                }
            }
            graph.control.addElement(Sequence(), {bottom}, {op});
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
                if(graph.control.getInputNodeIndices<Body>(newOp).to<std::unordered_set>().count(
                       parent)
                   == 0)
                {
                    graph.control.addElement(edge, {parent}, {newOp});
                }
            }
            graph.control.addElement(Body(), {newOp}, {op});
        }

        bool needsComputeIndex(Operation const& op)
        {
            if(std::holds_alternative<StoreTiled>(op) || std::holds_alternative<StoreLDSTile>(op)
               || std::holds_alternative<LoadTiled>(op) || std::holds_alternative<LoadLDSTile>(op))
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
                        if(!std::holds_alternative<Operation>(elem))
                            return false;
                        auto op = std::get<Operation>(elem);
                        return needsComputeIndex(op);
                    },
                    GD::Downstream)
                .to<std::vector>();
        }

        void purgeFor(KernelGraph& kgraph, int loop)
        {
            // Purge loop dimension and iterator
            for(auto const& c : kgraph.mapper.getConnections(loop))
            {
                int iterator = c.coordinate;
                // TODO THIS IS A FRAGILE WAY OF DETECTING "NO MORE REFERENCES"
                if(kgraph.mapper.getCoordinateConnections(iterator).size() <= 3)
                {
                    auto dataflow = *only(
                        kgraph.coordinates.getNeighbours<Graph::Direction::Downstream>(iterator));
                    auto forLoop = *only(
                        kgraph.coordinates.getNeighbours<Graph::Direction::Downstream>(dataflow));
                    kgraph.coordinates.deleteElement(iterator);
                    kgraph.mapper.purgeMappingsTo(iterator);
                    kgraph.coordinates.deleteElement(dataflow);
                    kgraph.mapper.purgeMappingsTo(dataflow);
                    kgraph.coordinates.deleteElement(forLoop);
                    kgraph.mapper.purgeMappingsTo(forLoop);
                }
                // XXX THIS LEAVES SOME DANGLING COORDS; IS THIS STILL TRUE?
            }

            // Purge loop
            purgeNodeAndChildren(kgraph, loop);
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
            return kgraph.coordinates.get<VGPR>(tag) || kgraph.coordinates.get<Workitem>(tag)
                   || kgraph.coordinates.get<Workgroup>(tag);
        }

        bool isLoopishCoordinate(int tag, KernelGraph const& kgraph)
        {
            return kgraph.coordinates.get<ForLoop>(tag) || kgraph.coordinates.get<Unroll>(tag);
        }

        bool isStorageCoordinate(int tag, KernelGraph const& kgraph)
        {
            return kgraph.coordinates.get<LDS>(tag) || kgraph.coordinates.get<User>(tag);
        }

        std::pair<int, Graph::Direction> getOperationTarget(int tag, KernelGraph const& kgraph)
        {
            auto elem = kgraph.control.getElement(tag);
            return std::visit(
                rocRoller::overloaded{
                    [&](StoreTiled const& op) -> std::pair<int, Graph::Direction> {
                        return {kgraph.mapper.get<User>(tag), GD::Upstream};
                    },
                    [&](LoadTiled const& op) -> std::pair<int, Graph::Direction> {
                        return {kgraph.mapper.get<User>(tag), GD::Downstream};
                    },
                    [&](StoreLDSTile const& op) -> std::pair<int, Graph::Direction> {
                        return {kgraph.mapper.get<LDS>(tag), GD::Upstream};
                    },
                    [&](LoadLDSTile const& op) -> std::pair<int, Graph::Direction> {
                        return {kgraph.mapper.get<LDS>(tag), GD::Downstream};
                    },
                    [&](Assign const& op) -> std::pair<int, Graph::Direction> {
                        return {kgraph.mapper.getConnections(tag)[0].coordinate, GD::Downstream};
                    },
                    [&](auto const& op) -> std::pair<int, Graph::Direction> {
                        Throw<FatalError>(
                            "Operation is not a load, store, or assign: ", tag, " ", toString(op));
                        return {0, GD::Downstream};
                    }},
                std::get<Operation>(elem));
        }

        std::pair<std::vector<int>, std::unordered_set<int>> findRequiredCoordinates(
            int target, Graph::Direction direction, KernelGraph const& kgraph)
        {
            namespace CT = rocRoller::KernelGraph::CoordinateGraph;

            // TODO: Design a better way of binding storage to coordinates
            auto maybeLDS = kgraph.coordinates.get<LDS>(target);
            if(maybeLDS)
            {
                // If target is LDS; it might be a duplicated LDS
                // node.  For the purposes of figuring out required
                // coordinates, use the parent LDS as the target
                // instead.
                auto maybeParentLDS = only(
                    kgraph.coordinates.getOutputNodeIndices(target, CT::isEdge<PassThrough>));
                if(maybeParentLDS)
                    target = *maybeParentLDS;
            }

            auto dontWalkPastLoopOrStorageNodes = [&](int tag) -> bool {
                auto element = kgraph.coordinates.getElement(tag);
                auto edge    = std::get<CT::Edge>(element);

                bool isCT   = std::holds_alternative<CT::CoordinateTransformEdge>(edge);
                bool follow = true;
                for(auto neighbour : kgraph.coordinates.getNeighbours(tag, opposite(direction)))
                {
                    if(neighbour == target)
                        continue;
                    if(isLoopishCoordinate(neighbour, kgraph))
                        follow = false;
                    if(isStorageCoordinate(neighbour, kgraph))
                        follow = false;
                }
                return isCT && follow;
            };

            // From the target coordinate, walk the graph but stop at loop
            // or storage nodes.  This will result in a list of nodes that
            // are used in the coordinate transform to compute indexes for
            // the target coordinate.
            auto candidates
                = kgraph.coordinates
                      .depthFirstVisit(target, dontWalkPastLoopOrStorageNodes, direction)
                      .to<std::vector>();

            // Internal nodes in the coordinate transform are computed as
            // part of the transform, so just keep leaf nodes and/or
            // hardware/loop coordinates.
            std::vector<int> required;
            std::copy_if(
                candidates.cbegin(), candidates.cend(), std::back_inserter(required), [&](int tag) {
                    bool isLeaf = kgraph.coordinates.getNeighbours(tag, direction)
                                      .to<std::vector>()
                                      .empty();
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

            auto [target, direction]    = getOperationTarget(tag, graph);
            auto [targetRequired, path] = findRequiredCoordinates(target, direction, graph);

            std::copy(targetRequired.cbegin(),
                      targetRequired.cend(),
                      std::inserter(required, required.end()));

            auto elem      = graph.control.getElement(tag);
            auto ldsTarget = std::visit(
                rocRoller::overloaded{
                    [&](StoreTiled const& op) {
                        auto ldsSpec = Connections::LDSTypeAndSubDimension{
                            LDS().name(), 0, Connections::LDSLoadStore::STORE_INTO_LDS};
                        return graph.mapper.get(tag, ldsSpec);
                    },
                    [&](LoadTiled const& op) {
                        auto ldsSpec = Connections::LDSTypeAndSubDimension{
                            LDS().name(), 0, Connections::LDSLoadStore::LOAD_FROM_LDS};
                        return graph.mapper.get(tag, ldsSpec);
                    },
                    [&](auto const& op) { return -1; }},
                std::get<Operation>(elem));

            if(ldsTarget != -1)
            {
                auto [ldsRequired, ldsPath] = findRequiredCoordinates(ldsTarget, direction, graph);
                std::copy(ldsPath.cbegin(), ldsPath.cend(), std::inserter(path, path.end()));
                std::copy(
                    ldsRequired.cbegin(), ldsRequired.cend(), std::inserter(required, path.end()));
            }

            return {required, path};
        }

        CT::User newScratchCoordinate(ExpressionPtr size, VariableType varType, ContextPtr context)
        {
            auto currentOffset = context->getScratchAmount();
            auto newCoordinate = User(size, currentOffset);
            context->allocateScratch(size * literal(DataTypeInfo::Get(varType).elementSize));

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

        int duplicateControlNode(KernelGraph& graph, int tag)
        {
            auto op = graph.control.addElement(graph.control.getElement(tag));
            for(auto const& c : graph.mapper.getConnections(tag))
            {
                graph.mapper.connect(op, c.coordinate, c.connection);
            }

            return op;
        }

        void
            updateThreadTileForLongDwords(int& t_m, int& t_n, int maxWidth, int numDwordsPerElement)
        {
            auto numDwordsPerWorkitem = t_m * numDwordsPerElement;

            std::vector<int> potentialFactors = {4, 3, 2, 1};

            auto start      = potentialFactors.begin();
            auto end        = potentialFactors.end();
            auto factorPred = [numDwordsPerWorkitem, maxWidth](int factor) {
                return factor <= maxWidth && numDwordsPerWorkitem % factor == 0;
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

        int getTopSetCoordinate(KernelGraph const& graph, int load)
        {
            int tag = load;

            while(true)
            {
                auto parent = only(graph.control.getInputNodeIndices<Body>(tag));
                if(!parent)
                    break;

                auto setCoord = graph.control.get<SetCoordinate>(*parent);
                if(setCoord)
                    tag = *parent;
                else
                    break;

                AssertFatal(graph.mapper.get<Unroll>(tag) > 0,
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
                auto parent = only(graph.control.getInputNodeIndices<Body>(tag));
                AssertFatal(parent, "Dimension was not found in the parents.");

                auto setCoord = graph.control.get<SetCoordinate>(*parent);
                AssertFatal(setCoord, "Dimension was not found in the parents.");

                tag = *parent;

                auto unroll = graph.mapper.get<Unroll>(tag);
                AssertFatal(unroll > 0, "SetCoordinate needs Unroll dimension");

                if(unroll == dim)
                    return tag;
            }
        }

        std::vector<int> getLoadsForUnroll(KernelGraph&     graph,
                                           int              unrollCoord,
                                           std::vector<int> loads,
                                           int              unroll)
        {
            std::vector<int> retval;
            for(auto& load : loads)
            {
                int  tag      = getSetCoordinateForDim(graph, unrollCoord, load);
                auto setCoord = graph.control.get<SetCoordinate>(tag);
                AssertFatal(evaluationTimes(
                                setCoord->value)[rocRoller::Expression::EvaluationTime::Translate],
                            "Unroll value should be a literal");
                if(unroll == getUnsignedInt(evaluate(setCoord->value)))
                {
                    retval.push_back(load);
                }
            }
            return retval;
        }

        VariableType getVariableType(KernelGraph const& graph, int opTag)
        {
            auto l = graph.control.get<LoadTiled>(opTag);
            auto s = graph.control.get<StoreTiled>(opTag);
            if(l)
                return l->varType;
            if(s)
                return s->dataType;
            Throw<FatalError>("Invalid load/store operation.");
        }

        void orderMemoryNodes(KernelGraph&                         graph,
                              std::set<std::pair<int, int>> const& pairs,
                              bool                                 ordered)
        {
            LastRWTracer tracer(graph);

            std::map<int, std::deque<int>> traces;
            for(auto pair : pairs)
            {
                traces[pair.first]  = tracer.controlStack(pair.first);
                traces[pair.second] = tracer.controlStack(pair.second);
            }
            for(auto pair : pairs)
            {
                if(pair.first != pair.second
                   && graph.control.compareNodes(pair.first, pair.second)
                          == NodeOrdering::Undefined)
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
            for(int i = 1; i < nodes.size(); i++)
            {
                pairs.insert(std::make_pair(nodes[i - 1], nodes[i]));
            }
            orderMemoryNodes(graph, pairs, ordered);
        }
    }
}
