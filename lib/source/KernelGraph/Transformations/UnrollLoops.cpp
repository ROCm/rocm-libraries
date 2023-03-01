
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        using GD = Graph::Direction;

        using namespace ControlGraph;
        using namespace CoordinateGraph;

        enum RAW
        {
            INVALID,
            NAUGHT,
            WRITE,
            READ,
        };

        /**
         * @brief Create duplicates of all of the nodes downstream of the provided
         *        start nodes.
         *        Add the duplicates to the provided graph.
         *        Return the location of the new start nodes.
         *
         * @param graph
         * @param startNodes
         * @return std::vector<int>
         */
        std::vector<int> duplicateControlNodes(KernelGraph&                   graph,
                                               GraphReindexer&                reindexer,
                                               std::set<int> const&           startNodes,
                                               std::unordered_set<int> const& dontDuplicate)

        {
            std::vector<int> newStartNodes;

            // Create duplicates of all of the nodes downstream of the startNodes
            for(auto const& node :
                graph.control.depthFirstVisit(startNodes, Graph::Direction::Downstream))
            {
                // Only do this step if element is a node
                if(graph.control.getElementType(node) == Graph::ElementType::Node)
                {
                    auto op = graph.control.addElement(graph.control.getElement(node));
                    reindexer.control[node] = op;
                }
            }

            for(auto const& reindex : reindexer.control)
            {
                // Create all edges within new sub-graph
                auto location = graph.control.getLocation(reindex.first);
                for(auto const& output : location.outgoing)
                {
                    int child = *graph.control.getNeighbours<Graph::Direction::Downstream>(output)
                                     .begin();
                    graph.control.addElement(graph.control.getElement(output),
                                             {reindex.second},
                                             {reindexer.control[child]});
                }

                // Use the same coordinate graph mappings
                for(auto const& c : graph.mapper.getConnections(reindex.first))
                {
                    auto coord = c.coordinate;

                    if(dontDuplicate.contains(coord))
                    {
                        graph.mapper.connect(reindex.second, coord, c.connection);
                        reindexer.coordinates.insert_or_assign(coord, coord);
                        continue;
                    }

                    // If one of the mappings represents storage, duplicate it in the CoordinateGraph.
                    // This will allow the RegisterTagManager to recognize that the nodes are pointing
                    // to different data and to use different registers.
                    // Note: A PassThrough edge is added from any of the duplicate nodes to the original
                    // node.
                    auto mt = graph.coordinates.get<MacroTile>(c.coordinate);
                    if(mt)
                    {
                        if(reindexer.coordinates.count(c.coordinate) == 0)
                        {
                            auto dim = graph.coordinates.addElement(
                                graph.coordinates.getElement(c.coordinate));
                            reindexer.coordinates[c.coordinate] = dim;
                            auto duplicate
                                = graph.coordinates
                                      .getOutputNodeIndices(coord, CT::isEdge<PassThrough>)
                                      .to<std::vector>();
                            if(duplicate.empty())
                                graph.coordinates.addElement(PassThrough(), {dim}, {coord});
                            else
                                graph.coordinates.addElement(PassThrough(), {dim}, {duplicate[0]});
                        }
                        coord = reindexer.coordinates[c.coordinate];
                    }
                    graph.mapper.connect(reindex.second, coord, c.connection);
                }
            }

            // Change coordinate values in Expressions
            for(auto const& reindex : reindexer.control)
            {
                auto elem = graph.control.getElement(reindex.first);
                if(isOperation<Assign>(elem))
                {
                    auto new_assign = graph.control.getNode<Assign>(reindex.second);
                    ReindexExpressionVisitor visitor(reindexer);
                    new_assign.expression = visitor.call(new_assign.expression);
                    graph.control.setElement(reindex.second, new_assign);
                }
            }

            // Return the new start nodes
            for(auto const& startNode : startNodes)
            {
                newStartNodes.push_back(reindexer.control[startNode]);
            }

            return newStartNodes;
        }

        /**
         * @brief Gets the name of the current for loop.
         *
         * @param graph
         * @param start
         * @return std::string
         */
        std::string getForLoopName(KernelGraph& graph, int start)
        {
            // Find the number of forLoops downstream from start
            auto forLoop = graph.control.get<ForLoopOp>(start);
            return forLoop->name;
        }

        /**
         * @brief Determine how many times to unroll the loop.
         *
         * A value of 1 means do not unroll it.
         * Use getForLoopName to determine which forLoop we are attempting to unroll
         * Checks unrollX(Y) value, 0 is default unroll it all if we can.
         */
        unsigned int getUnrollAmount(KernelGraph& graph, int loopTag, KernelOptions& kernelOptions)
        {
            auto dimTag        = graph.mapper.get<Dimension>(loopTag);
            auto forLoopLength = getSize(std::get<Dimension>(graph.coordinates.getElement(dimTag)));

            auto unrollX = kernelOptions.unrollX;
            auto unrollY = kernelOptions.unrollY;
            auto unrollK = kernelOptions.unrollK;
            // Find the number of forLoops following this for loop.
            auto name = getForLoopName(graph, loopTag);
            if(name == rocRoller::XLOOP && unrollX > 0)
                return unrollX;
            else if(name == rocRoller::YLOOP && unrollY > 0)
                return unrollY;
            else if(name == rocRoller::KLOOP && unrollK > 0)
                return unrollK;

            // Use default behavior if the above isn't true
            // If loop length is a constant, unroll the loop by that amount
            if(Expression::evaluationTimes(forLoopLength)[Expression::EvaluationTime::Translate])
            {
                auto length = Expression::evaluate(forLoopLength);
                if(isInteger(length))
                    return getUnsignedInt(length);
            }

            return 1u;
        }

        std::optional<int> getForLoopCoordinate(KernelGraph const& kgraph, int forLoop)
        {
            auto iterator = kgraph.mapper.get<Dimension>(forLoop);
            return only(kgraph.coordinates.getOutputNodeIndices<DataFlowEdge>(iterator));
        }

        /**
         * @brief Container to describe a loop-carried dependency. 
         */
        struct LoopCarriedDependency
        {
            int coordinate; //< Coordinate that has a loop-carried dependency
            int control; //< Operation that needs to be made sequential within the unroll to satisfy loop-carried dependencies.
        };

        /**
         * @brief Find loop-carried dependencies.
         *  
         * If a coordinate:
         *
         * 1. is written to only once
         * 2. is read-after-write
         * 3. has the loop coordinate in it's coordinate transform
         *    for the write
         *
         * then we take a giant leap of faith and say that it
         * doesn't have loop-carried-dependencies.
         */
        std::optional<std::vector<LoopCarriedDependency>>
            findLoopCarriedDependencies(KernelGraph const& kgraph, int forLoop)
        {
            using RW = ControlFlowRWTracer::ReadWrite;

            rocRoller::Log::getLogger()->debug("KernelGraph::findLoopDependencies({})", forLoop);

            auto maybeTopForLoopCoord = getForLoopCoordinate(kgraph, forLoop);
            if(!maybeTopForLoopCoord)
                return {};
            auto topForLoopCoord = *maybeTopForLoopCoord;

            ControlFlowRWTracer tracer(kgraph, true);
            tracer.trace(forLoop);

            auto readwrite = tracer.coordinatesReadWrite();

            // Coordinates that are read/write in loop body
            std::unordered_set<int> coordinates;
            for(auto m : readwrite)
                coordinates.insert(m.coordinate);

            // Coordinates with loop-carried-dependencies; true by
            // default.
            std::unordered_set<int> loopCarriedDependencies;
            for(auto m : readwrite)
                loopCarriedDependencies.insert(m.coordinate);

            // Now we want to determine which coordinates don't have
            // loop carried dependencies...
            //
            // If a coordinate:
            //
            // 1. is written to only once
            // 2. is read-after-write
            // 3. has the loop coordinate in it's coordinate transform
            //    for the write
            //
            // then we take a giant leap of faith and say that it
            // doesn't have loop-carried-dependencies.
            for(auto coordinate : coordinates)
            {
                std::optional<int> writeOperation;

                // Is coordinate RAW?
                //
                // RAW finite state machine moves through:
                //   NAUGHT -> WRITE   upon write
                //   WRITE  -> READ    upon read
                //   READ   -> READ    upon read
                //          -> INVALID otherwise
                //
                // If it ends in READ, we have a RAW
                RAW raw = RAW::NAUGHT;
                for(auto x : readwrite)
                {
                    if(x.coordinate != coordinate)
                        continue;
                    // Note: "or raw == RAW::WRITE" is absent on purpose;
                    // only handle single write for now
                    if((raw == RAW::NAUGHT) && (x.rw == RW::WRITE || x.rw == RW::READWRITE))
                    {
                        raw            = RAW::WRITE;
                        writeOperation = x.control;
                        continue;
                    }
                    if((raw == RAW::WRITE || raw == RAW::READ) && (x.rw == RW::READ))
                    {
                        raw = RAW::READ;
                        continue;
                    }
                    raw = RAW::INVALID;
                }

                if(raw != RAW::READ)
                    continue;

                AssertFatal(writeOperation);

                // Does the coordinate transform for the write
                // operation contain the loop iteration?

                auto [target, direction] = getOperationTarget(*writeOperation, kgraph);
                auto required            = findRequiredCoordinates(target, direction, kgraph);
                if(required.size() == 0)
                {
                    // Local variable
                    loopCarriedDependencies.erase(coordinate);
                    continue;
                }
                auto forLoopCoords = filterCoordinates<ForLoop>(required, kgraph);
                for(auto forLoopCoord : forLoopCoords)
                    if(forLoopCoord == topForLoopCoord)
                        loopCarriedDependencies.erase(coordinate);
            }

            // For the loop-carried-depencies, find the write operation.
            std::vector<LoopCarriedDependency> rv;
            for(auto coordinate : loopCarriedDependencies)
            {
                for(auto x : readwrite)
                {
                    if(x.coordinate != coordinate)
                        continue;
                    if(x.rw == RW::WRITE || x.rw == RW::READWRITE)
                        rv.push_back({coordinate, x.control});
                }
            }

            return rv;
        }

        /**
         * @brief Make operations sequential.
         *
         * Make operations in `sequentialOperations` execute
         * sequentially.  This changes concurrent patterns similar to
         *
         *     Kernel/Loop/Scope
         *       |           |
         *      ...         ...
         *       |           |
         *     OperA       OperB
         *       |           |
         *      ...         ...
         *
         * into a sequential pattern
         *
         *     Kernel/Loop/Scope
         *       |           |
         *      ...         ...
         *       |           |
         *   SetCoord --->SetCoord ---> remaining
         *       |           |
         *     OperA       OperB
         *
         */
        void makeSequential(KernelGraph& graph, std::vector<int>& sequentialOperations)
        {
            std::vector<int> setCoordinateChain;
            for(auto op : sequentialOperations)
            {
                auto maybeSetCoordinate = findContainingOperation<SetCoordinate>(op, graph);
                AssertFatal(maybeSetCoordinate, "Can't find containing SetCoordinate");

                auto setCoordinate  = *maybeSetCoordinate;
                auto destCoordinate = graph.mapper.get<Unroll>(setCoordinate);
                auto incomingEdge   = *only(graph.control.getNeighbours<GD::Upstream>(op));

                graph.control.deleteElement(incomingEdge);

                auto setCoordinateNode = *graph.control.get<SetCoordinate>(setCoordinate);
                auto newSetCoordinate  = graph.control.addElement(setCoordinateNode);
                graph.mapper.connect<Unroll>(newSetCoordinate, destCoordinate);

                graph.control.addElement(Sequence(), {setCoordinate}, {newSetCoordinate});
                graph.control.addElement(Body(), {newSetCoordinate}, {op});

                setCoordinateChain.push_back(newSetCoordinate);
            }

            for(int i = 1; i < setCoordinateChain.size(); ++i)
                graph.control.addElement(
                    Sequence(), {setCoordinateChain[i - 1]}, {setCoordinateChain[i]});
        }

        struct UnrollLoopsVisitor : public BaseGraphVisitor
        {
            UnrollLoopsVisitor(std::shared_ptr<Context> context)
                : BaseGraphVisitor(context, Graph::Direction::Upstream, false)
            {
            }

            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        ForLoopOp const&   op) override
            {
                copyOperation(graph, original, reindexer, tag);
                auto newTag = reindexer.control.at(tag);
                auto bodies = graph.control.getOutputNodeIndices<Body>(newTag).to<std::set>();

                auto unrollAmount = getUnrollAmount(graph, newTag, m_context->kernelOptions());
                if(unrollAmount == 1)
                    return;

                auto maybeLoopCarriedDependencies = findLoopCarriedDependencies(original, tag);
                if(!maybeLoopCarriedDependencies)
                    return;

                auto loopCarriedDependencies = *maybeLoopCarriedDependencies;

                // TODO: Iron out storage node duplication
                //
                // If we aren't doing the KLoop: some assumptions made
                // during subsequent lowering stages (LDS) are no
                // longer satisifed.
                //
                // For now, to avoid breaking those assumptions, only
                // follow through with loop-carried analysis when we
                // are doing the K loop.
                //
                // To disable loop-carried analysis, just set the
                // dependencies to an empty list...
                //
                // The only test which tries to unroll the K-loop is
                // the BasicGEMMUnrollK test.
                if(getForLoopName(graph, newTag) != rocRoller::KLOOP || unrollAmount <= 1)
                {
                    loopCarriedDependencies = {};
                }

                std::unordered_set<int> dontDuplicate;
                for(auto const& dep : loopCarriedDependencies)
                {
                    dontDuplicate.insert(dep.coordinate);
                }

                // ---------------------------------
                // Add Unroll dimension to the coordinates graph

                // Use the same coordinate graph mappings
                auto loopIterator = original.mapper.getConnections(tag)[0].coordinate;

                // The loop iterator should have a dataflow link to the ForLoop dimension
                auto forLoopDimension
                    = graph.coordinates.getOutputNodeIndices<DataFlowEdge>(loopIterator)
                          .to<std::vector>()[0];

                // Find all incoming PassThrough edges to the ForLoop dimension and replace them with
                // a Split edge with an Unroll dimension.
                auto forLoopLocation = graph.coordinates.getLocation(forLoopDimension);
                auto unrollDimension = graph.coordinates.addElement(Unroll(unrollAmount));
                for(auto const& input : forLoopLocation.incoming)
                {
                    if(isEdge<PassThrough>(std::get<Edge>(graph.coordinates.getElement(input))))
                    {
                        int parent
                            = *graph.coordinates.getNeighbours<Graph::Direction::Upstream>(input)
                                   .begin();
                        graph.coordinates.addElement(
                            Split(), {parent}, {forLoopDimension, unrollDimension});
                        graph.coordinates.deleteElement(input);
                    }
                }

                // Find all outgoing PassThrough edges from the ForLoop dimension and replace them with
                // a Join edge with an Unroll dimension.
                for(auto const& output : forLoopLocation.outgoing)
                {
                    if(isEdge<PassThrough>(std::get<Edge>(graph.coordinates.getElement(output))))
                    {
                        int child
                            = *graph.coordinates.getNeighbours<Graph::Direction::Downstream>(output)
                                   .begin();
                        graph.coordinates.addElement(
                            Join(), {forLoopDimension, unrollDimension}, {child});
                        graph.coordinates.deleteElement(output);
                    }
                }

                // ------------------------------
                // Change the loop increment calculation
                // Multiply the increment amount by the unroll amount

                // Find the ForLoopIcrement calculation
                // TODO: Handle multiple ForLoopIncrement edges that might be in a different
                // format, such as ones coming from ComputeIndex.
                auto loopIncrement = graph.control.getOutputNodeIndices<ForLoopIncrement>(newTag)
                                         .to<std::vector>();
                AssertFatal(loopIncrement.size() == 1, "Should only have 1 loop increment edge");

                auto loopIncrementOp = graph.control.getNode<Assign>(loopIncrement[0]);

                auto [lhs, rhs] = getForLoopIncrement(graph, newTag);

                auto newAddExpr            = lhs + (rhs * Expression::literal(unrollAmount));
                loopIncrementOp.expression = newAddExpr;

                graph.control.setElement(loopIncrement[0], loopIncrementOp);

                // ------------------------------
                // Add a setCoordinate node in between the original ForLoopOp and the loop bodies

                // Delete edges between original ForLoopOp and original loop body
                for(auto const& child :
                    graph.control.getNeighbours<Graph::Direction::Downstream>(newTag))
                {
                    if(isEdge<Body>(graph.control.getElement(child)))
                    {
                        graph.control.deleteElement(child);
                    }
                }

                // Function for adding a SetCoordinate node inbetween the ForLoop
                // and a list of nodes.
                auto connectWithSetCoord = [&](auto const& toConnect, unsigned int coordValue) {
                    auto setCoord
                        = graph.control.addElement(SetCoordinate(Expression::literal(coordValue)));
                    graph.mapper.connect<Unroll>(setCoord, unrollDimension);
                    graph.control.addElement(Body(), {newTag}, {setCoord});
                    for(auto const& body : toConnect)
                    {
                        graph.control.addElement(Body(), {setCoord}, {body});
                    }
                };

                // Add setCoordinate nodes to original body
                connectWithSetCoord(bodies, 0u);

                // ------------------------------
                // Create duplicates of the loop body and add a setCoordinate node in between
                // the ForLoopOp and the new bodies

                std::map<int, std::vector<int>> sequentialOperations;
                for(unsigned int i = 1; i < unrollAmount; i++)
                {
                    GraphReindexer unrollReindexer;

                    auto newBodies
                        = duplicateControlNodes(graph, unrollReindexer, bodies, dontDuplicate);
                    connectWithSetCoord(newBodies, i);

                    for(auto const& dep : loopCarriedDependencies)
                    {
                        auto newOp = reindexer.control.at(dep.control);
                        auto dupOp = unrollReindexer.control.at(newOp);
                        sequentialOperations[newOp].push_back(dupOp);
                    }
                }

                for(auto [first, remaining] : sequentialOperations)
                {
                    std::vector<int> operations = {first};
                    std::copy(remaining.cbegin(), remaining.cend(), std::back_inserter(operations));
                    makeSequential(graph, operations);
                }
            }
        };

        KernelGraph unrollLoops(KernelGraph const& original, std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::unrollLoops");
            rocRoller::Log::getLogger()->debug("KernelGraph::unrollLoops()");

            auto visitor = UnrollLoopsVisitor(context);
            return rewrite(original, visitor);
        }
    }
}
