
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/UnrollLoops.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
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
            return forLoop->loopName;
        }

        /**
         * @brief Determine how many times to unroll the loop.
         *
         * A value of 1 means do not unroll it.
         * Use getForLoopName to determine which forLoop we are attempting to unroll
         */
        unsigned int
            getUnrollAmount(KernelGraph& graph, int loopTag, KernelOptions const& kernelOptions)
        {
            auto name = getForLoopName(graph, loopTag);

            // Only attempt to unroll some loops for now...
            auto onlyUnroll
                = std::set<std::string>{rocRoller::XLOOP, rocRoller::YLOOP, rocRoller::KLOOP};
            if(!onlyUnroll.contains(name))
                return 1u;

            auto dimTag        = graph.mapper.get(loopTag, NaryArgument::DEST);
            auto forLoopLength = getSize(std::get<Dimension>(graph.coordinates.getElement(dimTag)));

            auto unrollK = kernelOptions.unrollK;
            // Find the number of forLoops following this for loop.
            if(name == rocRoller::KLOOP && unrollK > 0)
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
         *
         * Returns a map with a coordinate node as a key and a list of
         * every control node that uses that coordinate node as its value.
         */
        std::map<int, std::vector<int>> findLoopCarriedDependencies(KernelGraph const& kgraph,
                                                                    int                forLoop)
        {
            using RW = ControlFlowRWTracer::ReadWrite;
            std::map<int, std::vector<int>> result;

            rocRoller::Log::getLogger()->debug("KernelGraph::findLoopDependencies({})", forLoop);

            auto topForLoopCoord = kgraph.mapper.get<ForLoop>(forLoop);

            ControlFlowRWTracer tracer(kgraph, forLoop);

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
                auto [required, path] = findAllRequiredCoordinates(*writeOperation, kgraph);
                if(required.empty())
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
            for(auto coordinate : loopCarriedDependencies)
            {
                for(auto x : readwrite)
                {
                    if(x.coordinate != coordinate)
                        continue;
                    if(x.rw == RW::WRITE || x.rw == RW::READWRITE)
                        result[coordinate].push_back(x.control);
                }
            }

            return result;
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
        void makeSequential(KernelGraph&                         graph,
                            const std::vector<std::vector<int>>& sequentialOperations)
        {
            for(int i = 0; i < sequentialOperations.size() - 1; i++)
            {
                graph.control.addElement(Sequence(),
                                         {sequentialOperations[i].back()},
                                         {sequentialOperations[i + 1].front()});
            }
        }

        struct UnrollLoopsVisitor
        {
            UnrollLoopsVisitor(ContextPtr context)
                : m_context(context)
            {
            }

            void unrollLoop(KernelGraph& graph, int tag)
            {
                if(m_unrolledLoopOps.count(tag) > 0)
                    return;
                m_unrolledLoopOps.insert(tag);

                auto bodies = graph.control.getOutputNodeIndices<Body>(tag).to<std::vector>();
                auto traverseBodies = graph.control.depthFirstVisit(bodies).to<std::vector>();
                for(const auto node : traverseBodies)
                {
                    if(graph.control.exists(node)
                       && isOperation<ForLoopOp>(graph.control.getElement(node)))
                    {
                        unrollLoop(graph, node);
                    }
                }

                auto unrollAmount = getUnrollAmount(graph, tag, m_context->kernelOptions());
                if(unrollAmount == 1)
                    return;

                Log::debug("  Unrolling loop {}, amount {}", tag, unrollAmount);

                auto loopCarriedDependencies = findLoopCarriedDependencies(graph, tag);

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
                std::unordered_set<int> dontDuplicate;

                if(getForLoopName(graph, tag) != rocRoller::KLOOP || unrollAmount <= 1)
                {
                    loopCarriedDependencies = {};
                    dontDuplicate = graph.coordinates.getNodes<LDS>().to<std::unordered_set>();
                }

                for(auto const& [coord, controls] : loopCarriedDependencies)
                {
                    dontDuplicate.insert(coord);
                }

                // ---------------------------------
                // Add Unroll dimension to the coordinates graph

                auto forLoopDimension = graph.mapper.get<ForLoop>(tag);
                AssertFatal(forLoopDimension >= 0,
                            "Unable to find ForLoop dimension for " + std::to_string(tag));

                int unrollDimension;

                if(m_unrolledLoopDimensions.count(forLoopDimension) > 0)
                {
                    unrollDimension = m_unrolledLoopDimensions[forLoopDimension];
                }
                else
                {
                    // Find all incoming PassThrough edges to the ForLoop dimension and replace them with
                    // a Split edge with an Unroll dimension.
                    auto forLoopLocation = graph.coordinates.getLocation(forLoopDimension);
                    unrollDimension      = graph.coordinates.addElement(Unroll(unrollAmount));
                    for(auto const& input : forLoopLocation.incoming)
                    {
                        if(isEdge<PassThrough>(std::get<Edge>(graph.coordinates.getElement(input))))
                        {
                            int parent
                                = *graph.coordinates.getNeighbours<GD::Upstream>(input).begin();
                            graph.coordinates.addElement(
                                Split(), {parent}, {forLoopDimension, unrollDimension});
                            graph.coordinates.deleteElement(input);
                        }
                    }

                    // Find all outgoing PassThrough edges from the ForLoop dimension and replace them with
                    // a Join edge with an Unroll dimension.
                    for(auto const& output : forLoopLocation.outgoing)
                    {
                        if(isEdge<PassThrough>(
                               std::get<Edge>(graph.coordinates.getElement(output))))
                        {
                            int child
                                = *graph.coordinates.getNeighbours<GD::Downstream>(output).begin();
                            graph.coordinates.addElement(
                                Join(), {forLoopDimension, unrollDimension}, {child});
                            graph.coordinates.deleteElement(output);
                        }
                    }

                    m_unrolledLoopDimensions[forLoopDimension] = unrollDimension;
                }

                // ------------------------------
                // Change the loop increment calculation
                // Multiply the increment amount by the unroll amount

                // Find the ForLoopIcrement calculation
                // TODO: Handle multiple ForLoopIncrement edges that might be in a different
                // format, such as ones coming from ComputeIndex.
                auto loopIncrement
                    = graph.control.getOutputNodeIndices<ForLoopIncrement>(tag).to<std::vector>();
                AssertFatal(loopIncrement.size() == 1, "Should only have 1 loop increment edge");

                auto loopIncrementOp = graph.control.getNode<Assign>(loopIncrement[0]);

                auto [lhs, rhs] = getForLoopIncrement(graph, tag);

                auto newAddExpr            = lhs + (rhs * Expression::literal(unrollAmount));
                loopIncrementOp.expression = newAddExpr;

                graph.control.setElement(loopIncrement[0], loopIncrementOp);

                // ------------------------------
                // Add a setCoordinate node in between the original ForLoopOp and the loop bodies

                // Delete edges between original ForLoopOp and original loop body
                for(auto const& child :
                    graph.control.getNeighbours<GD::Downstream>(tag).to<std::vector>())
                {
                    if(isEdge<Body>(graph.control.getElement(child)))
                    {
                        graph.control.deleteElement(child);
                    }
                }

                // Function for adding a SetCoordinate nodes around all load and
                // store operations. Only adds a SetCoordinate node if the coordinate is needed
                // by the load/store operation.
                auto connectWithSetCoord = [&](std::vector<int> const& toConnect,
                                               unsigned int coordValue) -> std::vector<int> {
                    std::vector<int> rv;
                    for(auto const& body : toConnect)
                    {
                        graph.control.addElement(Body(), {tag}, {body});
                        for(auto const& op : findComputeIndexCandidates(graph, body))
                        {
                            auto [required, path] = findAllRequiredCoordinates(op, graph);

                            if(path.count(unrollDimension) > 0)
                            {
                                auto setCoord = replaceWith(graph,
                                                            op,
                                                            graph.control.addElement(SetCoordinate(
                                                                Expression::literal(coordValue))),
                                                            false);
                                graph.mapper.connect<Unroll>(setCoord, unrollDimension);

                                graph.control.addElement(Body(), {setCoord}, {op});
                                rv.push_back(setCoord);
                            }
                            else
                            {
                                rv.push_back(op);
                            }
                        }
                    }
                    return rv;
                };

                std::vector<std::vector<int>> duplicatedBodies;

                // ------------------------------
                // Create duplicates of the loop body and populate the sequentialOperations
                // data structure. sequentialOperations is a map that uses a coordinate with
                // a loop carried dependency as a key and contains a vector of vectors
                // with every control node that uses that coordinate in each new body of the loop.
                std::map<int, std::vector<std::vector<int>>> sequentialOperations;
                for(unsigned int i = 0; i < unrollAmount; i++)
                {
                    if(m_unrollReindexers.count({forLoopDimension, i}) == 0)
                        m_unrollReindexers[{forLoopDimension, i}]
                            = std::make_shared<GraphReindexer>();

                    auto unrollReindexer = m_unrollReindexers[{forLoopDimension, i}];

                    auto dontDuplicatePredicate = [&](int x) { return dontDuplicate.contains(x); };

                    if(i == 0)
                        duplicatedBodies.push_back(bodies);
                    else
                        duplicatedBodies.push_back(duplicateControlNodes(
                            graph, unrollReindexer, bodies, dontDuplicatePredicate));

                    for(auto const& [coord, controls] : loopCarriedDependencies)
                    {
                        std::vector<int> dupControls;
                        for(auto const& control : controls)
                        {
                            if(i == 0)
                            {
                                dupControls.push_back(control);
                            }
                            else
                            {
                                auto dupOp = unrollReindexer->control.at(control);
                                dupControls.push_back(dupOp);
                            }
                        }
                        sequentialOperations[coord].emplace_back(std::move(dupControls));
                    }

                    unrollReindexer->control = {};
                }

                // Connect the duplicated bodies to the loop, adding SetCoordinate nodes
                // as needed.
                std::set<int> previousLoads;
                std::set<int> previousStores;
                for(int i = 0; i < unrollAmount; i++)
                {
                    duplicatedBodies[i] = connectWithSetCoord(duplicatedBodies[i], i);
                    auto currentLoads
                        = filter(graph.control.isElemType<LoadTiled>(),
                                 graph.control.depthFirstVisit(duplicatedBodies[i], GD::Downstream))
                              .to<std::set>();

                    auto currentStores
                        = filter(graph.control.isElemType<StoreTiled>(),
                                 graph.control.depthFirstVisit(duplicatedBodies[i], GD::Downstream))
                              .to<std::set>();
                    if(i > 0)
                    {
                        orderMemoryNodes(graph, previousLoads, currentLoads, true);
                        orderMemoryNodes(graph, previousStores, currentStores, true);
                    }
                    previousLoads  = currentLoads;
                    previousStores = currentStores;
                }

                // If there are any loop carried dependencies, add Sequence nodes
                // between the control nodes with dependencies.
                for(auto [coord, allControls] : sequentialOperations)
                {
                    makeSequential(graph, allControls);
                }
            }

            void commit(KernelGraph& kgraph)
            {
                for(const auto node :
                    kgraph.control.depthFirstVisit(*kgraph.control.roots().begin())
                        .to<std::vector>())
                {
                    if(kgraph.control.exists(node)
                       && isOperation<ForLoopOp>(kgraph.control.getElement(node)))
                    {
                        unrollLoop(kgraph, node);
                    }
                }
            }

            std::map<int, int>                                             m_unrolledLoopDimensions;
            std::map<std::pair<int, int>, std::shared_ptr<GraphReindexer>> m_unrollReindexers;
            std::unordered_set<int>                                        m_unrolledLoopOps;
            std::shared_ptr<Context>                                       m_context;
        };

        KernelGraph UnrollLoops::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::unrollLoops");
            auto newGraph = original;

            auto visitor = UnrollLoopsVisitor(m_context);
            visitor.commit(newGraph);
            return newGraph;
        }
    }
}
