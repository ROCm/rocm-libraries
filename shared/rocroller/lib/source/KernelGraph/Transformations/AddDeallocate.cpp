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

#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/Transforms/AddDeallocate.hpp>

#include <rocRoller/Context.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowArgumentTracer.hpp>
#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    using namespace CoordinateGraph;
    using namespace ControlGraph;

    std::set<int> getContainingForLoops(std::set<int> controls, KernelGraph const& graph)
    {
        std::set<int> rv;

        for(auto control : controls)
        {
            auto maybeForLoop = findContainingOperation<ForLoopOp>(control, graph);
            if(maybeForLoop)
                rv.insert(*maybeForLoop);
        }

        return rv;
    }

    void simplifyDependencies(KernelGraph const& graph, std::set<int>& deps)
    {
        TIMER(t, "AddDeallocate::simplifyDependencies");

        for(auto iterA = deps.begin(); iterA != deps.end();)
        {
            bool sameA = true;
            for(auto iterB = std::next(iterA); iterB != deps.end();)
            {
                if(iterA == iterB)
                    continue;

                auto rel = graph.control.compareNodes(UpdateCache, *iterA, *iterB);

                if(rel == NodeOrdering::LeftFirst)
                {
                    iterA = deps.erase(iterA);
                    sameA = false;
                    break;
                }
                else if(rel == NodeOrdering::RightFirst)
                {
                    iterB = deps.erase(iterB);
                }
                else if(rel == NodeOrdering::LeftInBodyOfRight
                        || rel == NodeOrdering::RightInBodyOfLeft)
                {
                    Throw<FatalError>("No body relationships should be here!");
                }
                else
                {
                    ++iterB;
                }
            }

            if(sameA)
                ++iterA;
        }
    }

    std::vector<int> addDataFlowTagDeallocates(KernelGraph& graph, LastRWTracer const& tracer)
    {
        auto locations = tracer.lastRWLocations();

        // Map of <incoming Sequence edges to add, tags to deallocate>
        std::map<std::set<int>, std::vector<int>> deallocateNodesToAdd;

        for(auto& [coordinate, controls] : locations)
        {
            // If there is a single entry in the hot-loop set when we
            // insert the Deallocate operation into the control graph,
            // the Deallocate will be added after the hot loop.
            std::set<int> hotLoop;

            // Add all containing loops of read/writes of an LDS
            // allocation to the hot-loop set.
            //
            // The effect is: if all read/writes of an LDS coordinate
            // happen in a single loop; we wait until after the loop
            // is done before deallocating.  This avoids LDS
            // allocation re-use within a hot-loop that may lead to
            // undefined behaviour.
            auto maybeLDS = graph.coordinates.get<LDS>(coordinate);
            if(maybeLDS)
            {
                hotLoop = getContainingForLoops(controls, graph);
            }

            // There is a single hot-loop, add the Deallocate
            // after it.
            if(hotLoop.size() == 1)
                controls = {*hotLoop.cbegin()};

            simplifyDependencies(graph, controls);

            deallocateNodesToAdd[controls].push_back(coordinate);
        }

        std::vector<int> deallocateNodes;
        deallocateNodes.reserve(deallocateNodesToAdd.size());

        auto const& logger = Log::getLogger();

        for(auto const& [controls, coords] : deallocateNodesToAdd)
        {
            {
                if(logger->should_log(LogLevel::Debug))
                {
                    std::ostringstream msg;
                    msg << "After {";
                    streamJoin(msg, controls, ", ");
                    msg << "}, Deallocate {";
                    streamJoin(msg, coords, ", ");
                    msg << "}";
                    Log::debug(msg.str());
                }

                // Create a Deallocate operation
                auto deallocate = graph.control.addElement(Deallocate());
                deallocateNodes.push_back(deallocate);

                int idx = 0;
                for(int coordinate : coords)
                    graph.mapper.connect<Dimension>(deallocate, coordinate, idx++);

                // Add sequence edges from each "last r/w" operation.
                //
                // There is either a single "last r/w" operation; or
                // are all of them are within the same body-parent.
                for(auto src : controls)
                    graph.control.addElement(Sequence(), {src}, {deallocate});
            }
        }

        return deallocateNodes;
    }

    std::vector<int>
        addArgumentDeallocates(KernelGraph& graph, LastRWTracer const& tracer, ContextPtr context)
    {

        ControlFlowArgumentTracer argTracer(graph, context->kernel());

        auto locations = tracer.lastArgLocations(argTracer);

        std::map<std::set<int>, std::vector<std::string>> deallocateNodesToAdd;

        for(auto& [arg, controls] : locations)
        {
            // Can't deallocate any args within a loop since they will need to
            // be around for the next iteration.
            auto hotLoop = getContainingForLoops(controls, graph);

            if(hotLoop.size() == 1)
                controls = {*hotLoop.cbegin()};

            simplifyDependencies(graph, controls);

            deallocateNodesToAdd[controls].push_back(arg);
        }

        std::vector<int> deallocateNodes;
        deallocateNodes.reserve(locations.size());

        auto const& logger = Log::getLogger();

        {
            TIMER(t, "addArgumentDeallocates::add nodes");
            for(auto const& [controls, args] : deallocateNodesToAdd)
            {
                if(logger->should_log(LogLevel::Debug))
                {
                    std::ostringstream msg;
                    msg << "After {";
                    streamJoin(msg, controls, ", ");
                    msg << "}, deallocate {";
                    streamJoin(msg, args, ", ");
                    msg << "}";
                    Log::debug(msg.str());
                }

                auto deallocate = graph.control.addElement(Deallocate{{args}});
                deallocateNodes.push_back(deallocate);

                // Add sequence edges from each "last r/w" operation.
                //
                // There is either a single "last r/w" operation; or
                // are all of them are within the same body-parent.
                for(auto src : controls)
                    graph.control.addElement(Sequence(), {src}, {deallocate});
            }
        }

        return deallocateNodes;
    }

    /**
     * Sequence Deallocate nodes before any other parallel nodes.  This will
     * ensure that if a tag is borrowed, it will be deallocated (returned)
     * before it is borrowed again.
     *
     * Before:
     * ```mermaid
     * graph LR
     *
     *  NodeA ---> NodeB
     *  NodeB ---> NodeC
     *  NodeA ---> Deallocate
     *  NodeB ---> Deallocate
     * ```
     *
     * If we don't simplify first, we will get:
     * ```mermaid
     * graph LR
     *
     *  NodeA ---> NodeB
     *  NodeB ---> NodeC
     *  Deallocate ---> NodeB
     *  Deallocate ---> NodeC
     *  NodeA ---> Deallocate
     *  NodeB ---> Deallocate
     * ```
     *
     * which contains a cycle.
     *
     * So we simplify:
     * ```mermaid
     * graph LR
     *
     *  NodeA ---> NodeB
     *  NodeB ---> NodeC
     *  NodeB ---> Deallocate
     * ```
     *
     * Then add new sequence edges:
     * ```mermaid
     * graph LR
     *
     *  NodeA ---> NodeB
     *  NodeB ---> NodeC
     *  NodeB ---> Deallocate
     *  Deallocate ---> NodeC
     * ```
     *
     * Then simplify again:
     * ```mermaid
     * graph LR
     *
     *  NodeA ---> NodeB
     *  NodeB ---> Deallocate
     *  Deallocate ---> NodeC
     * ```
     *
     */
    void sequenceDeallocatesBeforeOtherNodes(std::vector<int> const& deallocateNodes,
                                             KernelGraph&            graph)
    {
        removeRedundantSequenceEdges(graph);

        /**
         * Siblings of Deallocate nodes that are not Deallocate nodes must come
         * after the Deallocate node.
         */
        for(auto deallocate : deallocateNodes)
        {
            for(auto parent : graph.control.getInputNodeIndices<Sequence>(deallocate))
            {
                for(auto child : graph.control.getOutputNodeIndices<Sequence>(parent))
                {
                    if(!graph.control.get<Deallocate>(child))
                        graph.control.chain<Sequence>(deallocate, child);
                }
            }
        }

        removeRedundantSequenceEdges(graph);
    }

    void deleteControlNode(KernelGraph& graph, int nodeIdx)
    {
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

    template <CInputRangeOf<int> Range>
    void mergeDeallocateNodes(KernelGraph& graph, int dstIdx, Range& srcs)
    {
        auto dst = graph.control.getNode<Deallocate>(dstIdx);

        auto connectionIdx = graph.mapper.getConnections(dstIdx).size();

        for(int srcIdx : srcs)
        {
            auto src = graph.control.getNode<Deallocate>(srcIdx);

            dst.arguments.insert(dst.arguments.end(), src.arguments.begin(), src.arguments.end());

            for(auto const& c : graph.mapper.getConnections(srcIdx))
            {
                graph.mapper.connect<Dimension>(dstIdx, c.coordinate, connectionIdx);
                connectionIdx++;
            }

            deleteControlNode(graph, srcIdx);
        }
        graph.control.setElement(dstIdx, std::move(dst));
    }

    void mergeAdjacentDeallocates(KernelGraph& graph)
    {
        using Connection     = std::tuple<int, int>; // other node Idx, edge type
        using Connections    = std::set<Connection>;
        using ConnectedNodes = std::tuple<Connections, Connections>;

        std::map<ConnectedNodes, std::set<int>> groupedDeallocates;

        for(auto deallocate : graph.control.getNodes<Deallocate>())
        {
            auto incoming
                = graph.control.getInputNodeIndices<ControlEdge>(deallocate)
                      .map([&](int incomingIdx) {
                          int edgeType
                              = graph.control
                                    .getEdge(
                                        graph.control.findEdge(incomingIdx, deallocate).value())
                                    .index();
                          return std::make_tuple(incomingIdx, edgeType);
                      })
                      .to<std::set>();

            auto outgoing
                = graph.control.getOutputNodeIndices<ControlEdge>(deallocate)
                      .map([&](int outgoingIdx) {
                          int edgeType
                              = graph.control
                                    .getEdge(
                                        graph.control.findEdge(deallocate, outgoingIdx).value())
                                    .index();

                          return std::make_tuple(outgoingIdx, edgeType);
                      })
                      .to<std::set>();

            groupedDeallocates[{std::move(incoming), std::move(outgoing)}].insert(deallocate);
        }

        for(auto const& [key, deallocates] : groupedDeallocates)
        {
            if(deallocates.size() > 1)
            {
                auto dest = *deallocates.begin();
                auto srcs
                    = std::ranges::subrange(std::next(deallocates.begin()), deallocates.end());
                using T = std::decay_t<decltype(srcs)>;
                static_assert(CInputRangeOf<T, int>);

                mergeDeallocateNodes(graph, dest, srcs);
            }
        }
    }

    KernelGraph AddDeallocateDataFlow::apply(KernelGraph const& original)
    {
        TIMER(t, "KernelGraph::addDeallocateDataFlow");
        rocRoller::Log::getLogger()->debug("KernelGraph::addDeallocateDataFlow()");

        auto graph  = original;
        auto tracer = LastRWTracer(graph);

        auto deallocateNodes = addDataFlowTagDeallocates(graph, tracer);

        sequenceDeallocatesBeforeOtherNodes(deallocateNodes, graph);

        return graph;
    }

    KernelGraph AddDeallocateArguments::apply(KernelGraph const& original)
    {
        TIMER(t, "KernelGraph::addDeallocateArguments");
        rocRoller::Log::getLogger()->debug("KernelGraph::addDeallocate()");

        auto graph  = original;
        auto tracer = LastRWTracer(graph);

        auto deallocateNodes = addArgumentDeallocates(graph, tracer, m_context);

        sequenceDeallocatesBeforeOtherNodes(deallocateNodes, graph);

        return graph;
    }

    KernelGraph MergeAdjacentDeallocates::apply(KernelGraph const& original)
    {
        TIMER(t, "KernelGraph::mergeAdjacentDeallocates");
        rocRoller::Log::getLogger()->debug("KernelGraph::mergeAdjacentDeallocates()");

        auto graph = original;

        mergeAdjacentDeallocates(graph);

        return graph;
    }
}
