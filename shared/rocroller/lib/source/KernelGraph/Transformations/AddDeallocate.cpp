// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/Transforms/AddDeallocate.hpp>
#include <rocRoller/KernelGraph/Transforms/AddDeallocate_detail.hpp>

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

    namespace AddDeallocateDetail
    {
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
                        Throw<FatalError>("Unexpected body relationship between",
                                          ShowValue(*iterA),
                                          ShowValue(*iterB),
                                          ShowValue(rel));
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
                for(auto parent :
                    graph.control.getInputNodeIndices<Sequence>(deallocate).to<std::vector>())
                {
                    for(auto child :
                        graph.control.getOutputNodeIndices<Sequence>(parent).to<std::vector>())
                    {
                        if(!graph.control.get<Deallocate>(child)
                           && graph.control.compareNodes(UseCacheIfAvailable, deallocate, child)
                                  != rocRoller::KernelGraph::ControlGraph::NodeOrdering::RightFirst)
                            graph.control.chain<Sequence>(deallocate, child);
                    }
                }
            }

            removeRedundantSequenceEdges(graph);
        }

        void deleteUnusedArguments(AssemblyKernelPtr                kernel,
                                   ControlFlowArgumentTracer const& argTracer)
        {
            auto arguments = kernel->resetArguments();

            auto const& neverReferencedArguments = argTracer.neverReferencedArguments();

            if(!neverReferencedArguments.empty())
            {
                std::ostringstream msg;
                msg << "Deleting never-referenced arguments: ";
                streamJoin(msg, neverReferencedArguments, ", ");
                Log::debug(msg.str());
            }

            auto referencedArgs = arguments | std::views::filter([&](auto const& arg) {
                                      return !neverReferencedArguments.contains(arg.getName());
                                  });

            for(auto& arg : referencedArgs)
            {
                kernel->addArgument({std::move(arg.getName()),
                                     arg.getVariableType(),
                                     arg.getDataDirection(),
                                     arg.getExpression()});
            }
        }
    }

    using namespace AddDeallocateDetail;

    std::vector<int> addDataFlowTagDeallocates(KernelGraph& graph)
    {
        auto tracer    = LastRWTracer(graph);
        auto locations = tracer.lastRWLocations();

        // Map of <incoming Sequence edges to add, tags to deallocate>
        std::map<std::set<int>, std::vector<int>> deallocateNodesToAdd;

        // Stage
        for(auto& [coordinate, controls] : locations)
        {
            auto dependencies = controls;

            simplifyDependencies(graph, dependencies);

            deallocateNodesToAdd[dependencies].push_back(coordinate);
        }

        // Commit
        std::vector<int> deallocateNodes;
        deallocateNodes.reserve(deallocateNodesToAdd.size());

        auto const& logger = Log::getLogger();

        for(auto const& [controls, coords] : deallocateNodesToAdd)
        {
            {
                // Create a Deallocate operation
                auto deallocate = graph.control.addElement(Deallocate());
                deallocateNodes.push_back(deallocate);

                if(logger->should_log(LogLevel::Debug))
                {
                    std::ostringstream msg;
                    msg << "After {";
                    streamJoin(msg, controls, ", ");
                    msg << "}, Deallocate(" << deallocate << ") {";
                    streamJoin(msg, coords, ", ");
                    msg << "}";
                    Log::debug(msg.str());
                }

                int idx = 0;
                for(int coordinate : coords)
                    graph.mapper.connect<Dimension>(deallocate, coordinate, idx++);

                // Add sequence edges from each "last r/w" operation.
                //
                // There is either a single "last r/w" operation; or
                // are all of them are within the same body-parent.
                for(auto src : controls)
                    graph.control.addElement(Sequence(), {src}, {deallocate});

                // Add a barrier before deallocating LDS to avoid write-after-read races
                // when different LDS allocation alias to the same LDS offset/address.
                if(std::any_of(coords.begin(), coords.end(), [&](int coordinate) {
                       return graph.coordinates.get<LDS>(coordinate).has_value();
                   }))
                {
                    auto barrierTag = graph.control.addElement(Barrier());
                    insertBefore(graph, barrierTag, deallocate, deallocate);
                }
            }
        }

        return deallocateNodes;
    }

    std::vector<int> addArgumentDeallocates(KernelGraph&                     graph,
                                            LastRWTracer const&              tracer,
                                            ControlFlowArgumentTracer const& argTracer,
                                            ContextPtr                       context)
    {
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
        rocRoller::Log::getLogger()->debug("KernelGraph::addDeallocateDataFlow()");

        auto graph = original;

        auto deallocateNodes = addDataFlowTagDeallocates(graph);

        sequenceDeallocatesBeforeOtherNodes(deallocateNodes, graph);

        return graph;
    }

    KernelGraph AddDeallocateArguments::apply(KernelGraph const& original)
    {
        rocRoller::Log::getLogger()->debug("KernelGraph::addDeallocate()");

        auto graph = original;

        ControlFlowArgumentTracer argTracer(graph, m_context->kernel());
        deleteUnusedArguments(m_context->kernel(), argTracer);

        auto tracer          = LastRWTracer(graph);
        auto deallocateNodes = addArgumentDeallocates(graph, tracer, argTracer, m_context);

        sequenceDeallocatesBeforeOtherNodes(deallocateNodes, graph);

        return graph;
    }

    KernelGraph MergeAdjacentDeallocates::apply(KernelGraph const& original)
    {
        rocRoller::Log::getLogger()->debug("KernelGraph::mergeAdjacentDeallocates()");

        auto graph = original;

        mergeAdjacentDeallocates(graph);

        return graph;
    }
}
