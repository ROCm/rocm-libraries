// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iterator>

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/InlineIncrements.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;

        using GD = rocRoller::Graph::Direction;

        struct InlineIncrementStage
        {
            int           assign;
            int           increment;
            int           forLoop;
            std::set<int> lastRWOps;
        };

        struct InlineIncrementer
        {
            std::vector<InlineIncrementStage> inlineIncrement;

            void stageForLoop(KernelGraph const& graph, int forLoop)
            {
                auto tracer = LastRWTracer(graph, forLoop, true);

                auto rw = tracer.lastRWLocations();

                for(auto increment : graph.control.getNeighbours<GD::Downstream>(forLoop))
                {
                    auto maybeIncrement = graph.control.get<ForLoopIncrement>(increment);
                    if(!maybeIncrement)
                        continue;

                    for(auto assign : graph.control.getNeighbours<GD::Downstream>(increment))
                    {
                        auto maybeAssign = graph.control.get<Assign>(assign);
                        AssertFatal(maybeAssign, "Invalid ForLoopIncrement operation.");

                        auto dstTag = graph.mapper.get(assign, NaryArgument::DEST);

                        // Include operations that modify the
                        // increment.
                        auto ops = rw[dstTag];

                        // Include operations that modify any
                        // coordinates connected by a DataFlow edge.
                        if(graph.coordinates.getElementType(dstTag) == Graph::ElementType::Node)
                        {
                            for(auto df :
                                graph.coordinates.getOutputNodeIndices<DataFlowEdge>(dstTag))
                            {
                                std::copy(rw[df].cbegin(),
                                          rw[df].cend(),
                                          std::inserter(ops, ops.begin()));
                            }
                        }

                        inlineIncrement.push_back({assign, increment, forLoop, ops});
                    }
                }
            }

            void stage(KernelGraph const& graph)
            {
                for(auto forLoop : graph.control.getNodes<ForLoopOp>())
                {
                    stageForLoop(graph, forLoop);
                }
            }

            void commit(KernelGraph& graph)
            {
                // Commit staged changes
                for(auto const& s : inlineIncrement)
                {
                    graph.control.deleteElement(s.increment);
                    if(s.lastRWOps.empty())
                    {
                        graph.control.addElement(Body(), {s.forLoop}, {s.assign});
                    }
                    else
                    {
                        for(auto op : s.lastRWOps)
                        {
                            graph.control.addElement(Sequence(), {op}, {s.assign});
                        }
                    }
                }

                // Sanity checks
                for(auto const& s : inlineIncrement)
                {
                    auto forLoop = findContainingOperation<ForLoopOp>(s.assign, graph);
                    AssertFatal(forLoop.value_or(-1) == s.forLoop);
                }
            }
        };

        KernelGraph InlineIncrements::apply(KernelGraph const& original)
        {
            auto graph     = original;
            auto transform = InlineIncrementer();
            transform.stage(graph);
            transform.commit(graph);
            return graph;
        }
    }
}
