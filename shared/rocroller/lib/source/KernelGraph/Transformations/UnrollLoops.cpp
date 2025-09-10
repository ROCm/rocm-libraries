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

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Transforms/UnrollLoops.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
namespace rocRoller
{
    namespace KernelGraph
    {
        using GD = Graph::Direction;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        std::string getForLoopName(KernelGraph& graph, int start)
        {
            auto forLoop = graph.control.get<ForLoopOp>(start);
            return forLoop->loopName;
        }
        unsigned int
            getUnrollAmount(KernelGraph& graph, int loopTag, CommandParametersPtr const& params)
        {
            auto name = getForLoopName(graph, loopTag);
            // Only attempt to unroll some loops for now...
            auto onlyUnroll
                = std::set<std::string>{rocRoller::XLOOP, rocRoller::YLOOP, rocRoller::KLOOP};
            if(!onlyUnroll.contains(name))
                return 1u;
            auto dimTag        = graph.mapper.get(loopTag, NaryArgument::DEST);
            auto forLoopLength = getSize(graph.coordinates.getNode(dimTag));
            auto unrollK       = params->unrollK;
            // Find the number of forLoops following this for loop.
            if(name == rocRoller::KLOOP && unrollK > 0)
                return unrollK;
            // Use default behavior if the above isn't true
            // If loop length is a constant, unroll the loop by that amount
            if(Expression::evaluationTimes(forLoopLength)[Expression::EvaluationTime::Translate])
            {
                auto length = Expression::evaluate(forLoopLength);
                if(isInteger(length))
                    return std::max(1u, getUnsignedInt(length));
            }
            return 1u;
        }
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
            enum class RAW
            {
                INVALID,
                NAUGHT,
                WRITE,
                READ,

                Count
            };
            using RW = ControlFlowRWTracer::ReadWrite;
            rocRoller::Log::getLogger()->debug("KernelGraph::findLoopDependencies({})", forLoop);
            auto                topForLoopCoord = kgraph.mapper.get<ForLoop>(forLoop);
            ControlFlowRWTracer tracer(kgraph, forLoop);
            auto                readwrite = tracer.coordinatesReadWrite();

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
            std::map<int, std::vector<int>> result;
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
                auto a    = getTopSetCoordinate(graph, sequentialOperations[i].back());
                auto b    = getTopSetCoordinate(graph, sequentialOperations[i + 1].front());
                auto edge = graph.control.addElement(Sequence(), {a}, {b});
                Log::debug("UnrollLoops::makeSequential:: Added Sequence edge {}", edge);
            }
        }

        static void buildControlStack(KernelGraph&             graph,
                                      int const                tag,
                                      std::unordered_set<int>& visited,
                                      std::vector<int>&        controlStack,
                                      bool const               bodyParent)
        {
            // cppcheck-suppress syntaxError
            auto const traverseEdge = [&]<typename EdgeType>() {
                for(auto parent : graph.control.getInputNodeIndices<EdgeType>(tag))
                {
                    if(visited.contains(parent))
                        continue;

                    visited.insert(parent);
                    buildControlStack(
                        graph, parent, visited, controlStack, !std::is_same_v<EdgeType, Sequence>);
                }
            };
            traverseEdge.template operator()<Body>();
            traverseEdge.template operator()<Else>();
            traverseEdge.template operator()<Sequence>();
            if(bodyParent)
                controlStack.push_back(tag);
        }

        static void orderCurrentAndPreviousNodes(KernelGraph&                        graph,
                                                 std::optional<std::pair<int, int>>& previousNodes,
                                                 std::set<int>&                      currentNodes)
        {
            if(currentNodes.empty())
                return;

            auto [firstNode, lastNode] = getFirstAndLastNodes(graph, currentNodes);

            if(previousNodes.has_value())
            {
                auto A = std::get<1>(previousNodes.value());
                auto B = firstNode;

                std::vector<int>        controlStackA;
                std::unordered_set<int> visited;
                buildControlStack(graph, A, visited, controlStackA, true);

                std::vector<int> controlStackB;
                visited.clear();
                buildControlStack(graph, B, visited, controlStackB, true);

                graph.control.orderMemoryNodes(controlStackA, controlStackB, true);
            }
            previousNodes = std::make_pair(firstNode, lastNode);
        }

        UnrollLoops::UnrollLoops(CommandParametersPtr params, ContextPtr context)
            : m_params(params)
            , m_context(context)
        {
        }

        void UnrollLoops::unrollLoop(KernelGraph& graph, int tag)
        {
            if(m_unrolledLoopOps.count(tag) > 0)
            {
                Log::debug("  Unrolled loop {} already, skipping.", tag);
                return;
            }

            m_unrolledLoopOps.insert(tag);

            auto bodies = graph.control.getOutputNodeIndices<Body>(tag).to<std::vector>();
            {
                // Unroll contained loops first.
                auto traverseBodies = graph.control.depthFirstVisit(bodies).to<std::vector>();
                for(const auto node : traverseBodies)
                {
                    if(graph.control.exists(node)
                       && isOperation<ForLoopOp>(graph.control.getElement(node)))
                    {
                        unrollLoop(graph, node);
                    }
                }
            }

            auto unrollAmount = getUnrollAmount(graph, tag, m_params);

            Log::debug("  Unrolling loop {}, amount {}", tag, unrollAmount);

            if(unrollAmount <= 1)
                return;

            auto forLoopDimension = graph.mapper.get<ForLoop>(tag);
            AssertFatal(forLoopDimension >= 0,
                        "Unable to find ForLoop dimension for " + std::to_string(tag));

            int unrollDimension = createUnrollDimension(graph, forLoopDimension, unrollAmount);

            {
                auto tailLoop
                    = createTailLoop(graph, tag, unrollAmount, unrollDimension, forLoopDimension);
                if(tailLoop)
                    m_unrolledLoopOps.insert(*tailLoop);
            }

            std::map<int, std::vector<int>> loopCarriedDependencies;

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
            // To disable loop-carried analysis, just leave the
            // dependencies as an empty list...

            std::unordered_set<int> dontDuplicate;
            if(getForLoopName(graph, tag) != rocRoller::KLOOP)
            {
                dontDuplicate = graph.coordinates.getNodes<LDS>().to<std::unordered_set>();
            }
            else
            {
                loopCarriedDependencies = findLoopCarriedDependencies(graph, tag);
                for(auto const& [coord, controls] : loopCarriedDependencies)
                {
                    // In the KLOOP, we do want to duplicate LDS (and paired tiles)
                    bool pairedWithLDS = false;
                    for(auto op : controls)
                        pairedWithLDS |= graph.mapper.get<LDS>(op) != -1;
                    if(pairedWithLDS)
                        continue;

                    dontDuplicate.insert(coord);
                }
            }

            // ------------------------------
            // Change the loop increment calculation
            // Multiply the increment amount by the unroll amount
            // Find the ForLoopIcrement calculation
            // TODO: Handle multiple ForLoopIncrement edges that might be in a different
            // format, such as ones coming from ComputeIndex.
            auto loopIncrement = graph.control.getOutputNodeIndices<ForLoopIncrement>(tag).only();
            AssertFatal(loopIncrement.has_value(), "Should only have 1 loop increment edge");
            auto loopIncrementOp       = graph.control.getNode<Assign>(loopIncrement.value());
            auto [lhs, rhs]            = getForLoopIncrement(graph, tag);
            auto newAddExpr            = lhs + (rhs * Expression::literal(unrollAmount));
            loopIncrementOp.expression = newAddExpr;
            graph.control.setElement(*loopIncrement, loopIncrementOp);

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
                                           unsigned int            coordValue) -> std::vector<int> {
                std::vector<int> rv;
                for(auto const& body : toConnect)
                {
                    graph.control.addElement(Body(), {tag}, {body});
                    for(auto const& op : findComputeIndexCandidates(graph, body))
                    {
                        auto pendingOp        = op;
                        auto [required, path] = findAllRequiredCoordinates(op, graph);
                        if(path.count(unrollDimension) > 0)
                        {
                            if(!hasExistingSetCoordinate(graph, op, coordValue, unrollDimension))
                            {
                                auto name = getForLoopName(graph, tag);
                                if(name == rocRoller::XLOOP)
                                    graph.mapper.connect<Unroll>(op, unrollDimension, 0);
                                else if(name == rocRoller::YLOOP)
                                    graph.mapper.connect<Unroll>(op, unrollDimension, 1);
                                else if(name == rocRoller::KLOOP)
                                    graph.mapper.connect<Unroll>(op, unrollDimension, 2);
                                auto setCoord = replaceWith(graph,
                                                            op,
                                                            graph.control.addElement(SetCoordinate(
                                                                Expression::literal(coordValue))),
                                                            false);
                                graph.mapper.connect<Unroll>(setCoord, unrollDimension);
                                graph.control.addElement(Body(), {setCoord}, {op});
                                pendingOp = setCoord;

                                Log::debug(
                                    "Added SetCoordinate {} for coordinate {} (value {}) above "
                                    "operation {}",
                                    setCoord,
                                    unrollDimension,
                                    coordValue,
                                    op);
                            }
                        }
                        rv.push_back(pendingOp);
                    }
                }
                return rv;
            };

            // ------------------------------
            // Create duplicates of the loop body and populate the sequentialOperations
            // data structure. sequentialOperations is a map that uses a coordinate with
            // a loop carried dependency as a key and contains a vector of vectors
            // with every control node that uses that coordinate in each new body of the loop.
            std::vector<std::vector<int>>                duplicatedBodies;
            std::map<int, std::vector<std::vector<int>>> sequentialOperations;
            for(unsigned int i = 0; i < unrollAmount; i++)
            {
                if(m_unrollReindexers.count({forLoopDimension, i}) == 0)
                    m_unrollReindexers[{forLoopDimension, i}] = std::make_shared<GraphReindexer>();
                auto unrollReindexer        = m_unrollReindexers[{forLoopDimension, i}];
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
            auto isLoadTiled    = graph.control.isElemType<LoadTiled>();
            auto isLoadLDSTile  = graph.control.isElemType<LoadLDSTile>();
            auto isStoreTiled   = graph.control.isElemType<StoreTiled>();
            auto isStoreLDSTile = graph.control.isElemType<StoreLDSTile>();

            auto getTops = [&](auto predicate, auto starts) {
                std::set<int> rv;
                for(auto op :
                    filter(predicate, graph.control.depthFirstVisit(starts, GD::Downstream)))
                {
                    rv.insert(getTopSetCoordinate(graph, op));
                }
                return rv;
            };

            Log::debug("  Ordering loop {}", tag);

            std::optional<std::pair<int, int>> previousLoads;
            std::optional<std::pair<int, int>> previousLDSLoads;
            std::optional<std::pair<int, int>> previousStores;
            std::optional<std::pair<int, int>> previousLDSStores;

            auto name = getForLoopName(graph, tag);
            for(int i = 0; i < unrollAmount; i++)
            {
                duplicatedBodies[i]   = connectWithSetCoord(duplicatedBodies[i], i);
                auto currentLoads     = getTops(isLoadTiled, duplicatedBodies[i]);
                auto currentLDSLoads  = getTops(isLoadLDSTile, duplicatedBodies[i]);
                auto currentStores    = getTops(isStoreTiled, duplicatedBodies[i]);
                auto currentLDSStores = getTops(isStoreLDSTile, duplicatedBodies[i]);

                orderCurrentAndPreviousNodes(graph, previousLoads, currentLoads);
                orderCurrentAndPreviousNodes(graph, previousLDSLoads, currentLDSLoads);
                orderCurrentAndPreviousNodes(graph, previousStores, currentStores);
                orderCurrentAndPreviousNodes(graph, previousLDSStores, currentLDSStores);

                currentLDSLoads
                    = filter(isLoadLDSTile,
                             graph.control.depthFirstVisit(duplicatedBodies[i], GD::Downstream))
                          .to<std::set>();
                for(auto ldsLoad : currentLDSLoads)
                {
                    if(name == rocRoller::KLOOP)
                        graph.mapper.connect<Unroll>(ldsLoad, unrollDimension, 2);
                }
            }

            // If there are any loop carried dependencies, add Sequence nodes
            // between the control nodes with dependencies.
            Log::debug("  Adding sequence edges between loop-carried-dependencies {}", tag);
            for(auto [coord, allControls] : sequentialOperations)
            {
                makeSequential(graph, allControls);
            }
        }

        // ---------------------------------
        // Add Unroll dimension to the coordinates graph and return it.
        int UnrollLoops::createUnrollDimension(KernelGraph& graph,
                                               int          forLoopDimension,
                                               int          unrollAmount)
        {
            if(m_unrolledLoopDimensions.count(forLoopDimension) > 0)
            {
                return m_unrolledLoopDimensions[forLoopDimension];
            }
            else
            {
                // Find all incoming PassThrough edges to the ForLoop dimension and replace
                // them with a Split edge with an Unroll dimension.
                auto forLoopLocation = graph.coordinates.getLocation(forLoopDimension);
                int  unrollDimension = graph.coordinates.addElement(Unroll(unrollAmount));
                for(auto const& input : forLoopLocation.incoming)
                {
                    if(isEdge<PassThrough>(graph.coordinates.getEdge(input)))
                    {
                        int parent = *graph.coordinates.getNeighbours<GD::Upstream>(input).begin();
                        graph.coordinates.addElement(
                            Split(), {parent}, {forLoopDimension, unrollDimension});
                        graph.coordinates.deleteElement(input);
                    }
                }
                // Find all outgoing PassThrough edges from the ForLoop dimension and replace
                // them with a Join edge with an Unroll dimension.
                for(auto const& output : forLoopLocation.outgoing)
                {
                    if(isEdge<PassThrough>(graph.coordinates.getEdge(output)))
                    {
                        int child
                            = *graph.coordinates.getNeighbours<GD::Downstream>(output).begin();
                        graph.coordinates.addElement(
                            Join(), {forLoopDimension, unrollDimension}, {child});
                        graph.coordinates.deleteElement(output);
                    }
                }
                m_unrolledLoopDimensions[forLoopDimension] = unrollDimension;
                return unrollDimension;
            }
        }

        /**
         * If needed/appropriate, create a tail loop. Will return the node ID of
         * the tail loop, if created, otherwise `nullopt`.
         *
         * Will not create a tail loop if:
         *  - The loop has a known trip count, which is divisible by unrollAmount
         *  - The loop is contained by another loop which is unrolled.
         *     - This is due to the FuseLoops transformation not yet being able to
         *       handle merging multiple tail loops together.
         *  - StreamK is enabled.
         *  - Tail loops are manually disabled.
         *
         * - Duplicate the for loop, making the duplicate after the original
         *   (sequence edge).
         * - Change the original loop's limit to round down to a multiple of
         *   unrollAmount ((x / unrollAmount) * unrollAmount).
         * - Change the new loop to start at that value.
         *
         *  -------------------------
         *  Before:
         *
         *  Other --seq--> SetCoord  --seq--> Other
         *                    |
         *                   Body
         *                    |
         *                    v
         *                 ForLoop
         *
         *  -------------------------
         *  After:
         *
         *                    /-------------------seq--------------|
         *                    |                                    v
         *  Other --seq--> SetCoord   --seq-->  SetCoord --seq--> Other
         *                    |                    |
         *                   Body                 Body
         *                    |                    |
         *                    v                    v
         *                 ForLoop              ForLoop
         *                 (original)           (tail)
         */
        std::optional<int> UnrollLoops::createTailLoop(KernelGraph& graph,
                                                       int          loop,
                                                       int          unrollAmount,
                                                       int          unrollDimension,
                                                       int          forLoopDimension)
        {
            if(!m_params->tailLoops)
            {
                Log::debug("Not adding tail loop for {} because tail loops are disabled.", loop);
                return std::nullopt;
            }
            if(m_params->streamK)
            {
                Log::debug("Not adding tail loop for {} because streamK is enabled.", loop);
                return std::nullopt;
            }
            if(getForLoopName(graph, loop) != rocRoller::KLOOP)
            {
                Log::debug("Not adding tail loop for {} because it is {}, not {}.",
                           loop,
                           getForLoopName(graph, loop),
                           KLOOP);
                return std::nullopt;
            }

            auto loopOp                   = graph.control.getNode<ForLoopOp>(loop);
            auto [loopVariable, loopSize] = split<Expression::LessThan>(loopOp.condition);
            auto isTranslateTime = evaluationTimes(loopSize)[Expression::EvaluationTime::Translate];

            if(isTranslateTime)
            {
                auto vis = rocRoller::overloaded{
                    [&](std::integral auto val) { return val % unrollAmount == 0; },
                    [](auto val) { return false; }};
                bool divisibleByUnroll = std::visit(vis, evaluate(loopSize));
                if(divisibleByUnroll)
                {
                    Log::debug(
                        "Not adding tail loop for {} because the size ({}) is divisible by {}",
                        loop,
                        toString(loopSize),
                        unrollAmount);
                    return std::nullopt;
                }
            }

            {
                auto containingForLoops = graph.control.nodesContaining(loop).filter(
                    graph.control.isElemType<ForLoopOp>());
                for(auto containingLoop : containingForLoops)
                {
                    if(getUnrollAmount(graph, containingLoop, m_params) > 1)
                    {
                        Log::debug("Not adding tail loop for {} because it is contained by {} "
                                   "which is also unrolled.",
                                   loop,
                                   containingLoop);
                        return std::nullopt;
                    }
                }
            }

            auto loopSizeType        = resultVariableType(loopSize);
            auto amount              = Expression::literal(unrollAmount, loopSizeType);
            auto loopSizeRoundedDown = (loopSize / amount) * amount;

            Log::debug("Adding tail loop for {}.  Size {} -> {}",
                       loop,
                       toString(loopSize),
                       toString(loopSizeRoundedDown));

            auto tailLoop = cloneForLoop(graph, loop);

            // Set original loop to end at a multiple of the unroll amount.
            {
                auto newCondition = loopVariable < loopSizeRoundedDown;
                copyComment(newCondition, loopOp.condition);
                loopOp.condition = newCondition;
                graph.control.setElement(loop, loopOp);
            }

            // Modify the init of the tail loop to start at the remainder.
            {
                auto init = graph.control.getOutputNodeIndices<Initialize>(tailLoop).only();
                AssertFatal(init.has_value(),
                            "There must be exactly one loop initialization node.");
                auto initOp = graph.control.getNode<Assign>(init.value());
                {
                    auto isZero
                        = rocRoller::overloaded{[](std::integral auto val) { return val == 0; },
                                                [](auto val) { return false; }};
                    AssertFatal(std::visit(isZero, evaluate(initOp.expression)),
                                "Init op to loop must equal 0.");
                }
                initOp.expression = loopSizeRoundedDown;
                graph.control.setElement(*init, initOp);
            }

            // Duplicate the body of the original for loop into the tail.
            {
                auto loopBodies = graph.control.getOutputNodeIndices<Body>(loop).to<std::vector>();
                auto newBodies
                    = duplicateControlNodes(graph, nullptr, loopBodies, [](int x) { return true; });
                for(auto node : newBodies)
                {
                    graph.control.chain<Body>(tailLoop, node);
                }
            }

            {
                // Set the value of the unroll dimension to 0 for the tail loop.
                // This SetCoord will contain the tail loop and so it's how we
                // ensure the tail loop happens immediately after the original loop
                // and before any other operations that were sequenced after the
                // original loop.
                auto setCoordK = graph.control.addElement(SetCoordinate(loopSizeRoundedDown));
                graph.mapper.connect<ForLoop>(setCoordK, forLoopDimension);
                auto setCoord = graph.control.addElement(SetCoordinate(Expression::literal(0u)));
                graph.mapper.connect<Unroll>(setCoord, unrollDimension);
                graph.control.chain<Body>(setCoordK, setCoord, tailLoop);
                // We are just adding more sequence edges and leaving the cleanup
                // for the Simplify graph transformation.
                for(auto node : graph.control.getOutputNodeIndices<Sequence>(loop))
                {
                    graph.control.chain<Sequence>(setCoord, node);
                }
                // The tail loop comes after the main loop.
                graph.control.chain<Sequence>(loop, setCoordK);
            }

            return tailLoop;
        }

        void UnrollLoops::commit(KernelGraph& kgraph)
        {
            for(const auto node :
                kgraph.control.depthFirstVisit(kgraph.control.roots().only().value())
                    .to<std::vector>())
            {
                if(kgraph.control.exists(node)
                   && isOperation<ForLoopOp>(kgraph.control.getElement(node)))
                {
                    unrollLoop(kgraph, node);
                }
            }
        }

        KernelGraph UnrollLoops::apply(KernelGraph const& original)
        {
            auto graph = original;
            commit(graph);
            return graph;
        }
    }
}
