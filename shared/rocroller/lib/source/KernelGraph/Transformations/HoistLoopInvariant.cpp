// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/HoistLoopInvariant.hpp>
#include <rocRoller/KernelGraph/Transforms/HoistLoopInvariant_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

#include <algorithm>
#include <optional>
#include <set>
#include <vector>

namespace rocRoller::KernelGraph
{
    using namespace ControlGraph;
    using namespace CoordinateGraph;

    /**
     * @brief Visitor for extracting DataFlowTags from expressions
     * 
     * This visitor traverses an expression tree and collects all DataFlowTag
     * references found within it.
     */
    struct DataFlowTagExtractorVisitor
    {
        std::set<int> tags;

        void operator()(Expression::DataFlowTag const& expr)
        {
            tags.insert(expr.tag);
        }

        void operator()(Expression::ScaledMatrixMultiply const& expr)
        {
            call(expr.matA);
            call(expr.matB);
            call(expr.matC);
            call(expr.scaleA);
            call(expr.scaleB);
        }

        template <Expression::CNary Expr>
        void operator()(Expr const& expr)
        {
            for(auto const& operand : expr.operands)
            {
                call(operand);
            }
        }

        template <Expression::CTernary Expr>
        void operator()(Expr const& expr)
        {
            call(expr.lhs);
            call(expr.r1hs);
            call(expr.r2hs);
        }

        template <Expression::CBinary Expr>
        void operator()(Expr const& expr)
        {
            call(expr.lhs);
            call(expr.rhs);
        }

        template <Expression::CUnary Expr>
        void operator()(Expr const& expr)
        {
            call(expr.arg);
        }

        template <Expression::CValue Value>
        void operator()(Value const&)
        {
            // DataFlowTag already matched separately
        }

        void call(Expression::ExpressionPtr const& expr)
        {
            if(!expr)
                return;
            std::visit(*this, *expr);
        }

        void call(Expression::Expression const& expr)
        {
            std::visit(*this, expr);
        }
    };

    std::set<int> extractDataFlowTags(Expression::Expression const& expr)
    {
        DataFlowTagExtractorVisitor visitor;
        visitor.call(expr);
        return visitor.tags;
    }

    // If the control node is inside a loop, return the loop node, else std::nullopt
    std::optional<int> getParentLoop(KernelGraph const& graph, int control)
    {
        auto stack = controlStack(control, graph);
        for(auto it = stack.rbegin(); it != stack.rend(); ++it)
        {
            int node = *it;
            if(graph.control.get<ForLoopOp>(node).has_value()
               || graph.control.get<DoWhileOp>(node).has_value())
            {
                return node;
            }
        }
        return std::nullopt;
    }

    CoordinateToLoops buildCoordinateLoopMapping(KernelGraph const&         graph,
                                                 ControlFlowRWTracer const& tracer)
    {
        CoordinateToLoops result;

        const auto records = tracer.coordinatesReadWrite();

        for(const auto& record : records)
        {
            Log::debug("Processing record: coordinate {}, control node {}, rw {}",
                       record.coordinate,
                       record.control,
                       static_cast<int>(record.rw));

            if(record.rw == ControlFlowRWTracer::READ)
            {
                continue;
            }

            auto containingLoop = getParentLoop(graph, record.control);
            if(containingLoop.has_value())
            {
                result[record.coordinate][containingLoop.value()].insert(record.control);
            }
        }
        return result;
    }

    int hoistNodeBeforeLoop(KernelGraph& kgraph, int nodeToHoist, int loopNode)
    {
        int hoistedNode = duplicateControlNode(kgraph, nodeToHoist);
        Log::debug("HoistLoopInvariant: Hoisted node {} to new node {}", nodeToHoist, hoistedNode);
        insertBefore(kgraph, loopNode, hoistedNode, hoistedNode);
        kgraph.control.setElement(nodeToHoist, NOP{});
        kgraph.mapper.purge(nodeToHoist);
        return hoistedNode;
    }

    KernelGraph HoistLoopInvariant::apply(KernelGraph const& original)
    {
        auto graph = original;

        ControlFlowRWTracer tracer(graph);
        const auto          mapping = buildCoordinateLoopMapping(graph, tracer);

        for(const auto& [coordinate, loopGroups] : mapping)
        {
            for(const auto& [loopNode, controlNodes] : loopGroups)
            {
                AssertFatal(loopNode >= 0);

                // Skip if more than one write in the loop body
                // As currently does not handle multiple reaching definitions (e.g. branch within loop body)
                if(controlNodes.size() != 1
                   || countCoordinateWritesInLoop(graph, loopNode, coordinate, tracer) != 1)
                {
                    Log::debug(
                        "HoistLoopInvariant: skipping {} with {} writers in immediate loop body {}",
                        coordinate,
                        controlNodes.size(),
                        loopNode);
                    continue;
                }

                int controlNode = *controlNodes.begin();

                // Check for read-before-write pattern in the loop
                // This prevents incorrect hoisting when a read depends on the value from a previous iteration
                if(hasReadBeforeWriteInLoop(graph, loopNode, coordinate, controlNode, tracer))
                {
                    Log::debug("HoistLoopInvariant: skipping {} due to read-before-write pattern "
                               "in loop {}",
                               coordinate,
                               loopNode);
                    continue;
                }

                auto maybeAssign = graph.control.get<Assign>(controlNode);
                if(!maybeAssign.has_value())
                {
                    // Skip LoadLDSTile, StoreLDSTile, etc.
                    // These nodes are more purposefully placed and are unlikely to be beneficial to hoist
                    Log::debug("HoistLoopInvariant: skipping {}",
                               Graph::variantToString(graph.control.getElement(controlNode)));
                    continue;
                }

                auto assignNode = maybeAssign.value();

                auto usedTags = extractDataFlowTags(*assignNode.expression);

                bool allTagsLoopInvariant = true;
                for(auto tag : usedTags)
                {
                    if(countCoordinateWritesInLoop(graph, loopNode, tag, tracer) > 0)
                    {
                        Log::debug(
                            "HoistLoopInvariant:   DataFlowTag {} is written in loop {}, not "
                            "invariant",
                            tag,
                            loopNode);
                        allTagsLoopInvariant = false;
                        break;
                    }
                }

                if(!allTagsLoopInvariant)
                {
                    continue;
                }

                bool       coordinateUsedAfterLoop = false;
                const auto readingControlNodes     = [&]() -> Generator<int> {
                    for(const auto record : tracer.coordinatesReadWrite(coordinate))
                    {
                        if(record.rw == ControlFlowRWTracer::READ)
                        {
                            co_yield record.control;
                        }
                    }
                }()
                                                              .to<std::set>();

                // If coordinate is read after loop, hoisting with a zero-iteration loop changes behavior
                for(int nodeAfterLoop : graph.control.nodesAfter(loopNode))
                {
                    if(readingControlNodes.find(nodeAfterLoop) != readingControlNodes.end())
                    {
                        coordinateUsedAfterLoop = true;
                        break;
                    }
                }

                if(!coordinateUsedAfterLoop)
                {
                    Log::debug("HoistLoopInvariant, hoisting {}, {}, to, {}, {}",
                               controlNode,
                               Graph::variantToString(graph.control.getElement(controlNode)),
                               loopNode,
                               Graph::variantToString(graph.control.getElement(loopNode)));

                    hoistNodeBeforeLoop(graph, controlNode, loopNode);
                }
            }
        }
        return graph;
    }

    std::string HoistLoopInvariant::name() const
    {
        return "HoistLoopInvariant";
    }

    int countCoordinateWritesInLoop(KernelGraph const&         kgraph,
                                    int                        loopNode,
                                    int                        coordinate,
                                    ControlFlowRWTracer const& tracer)
    {
        auto records    = tracer.coordinatesReadWrite(coordinate);
        int  writeCount = 0;

        for(const auto& record : records)
        {
            if(record.rw == ControlFlowRWTracer::WRITE
               || record.rw == ControlFlowRWTracer::READWRITE)
            {
                auto stack = controlStack(record.control, kgraph);

                if(std::find(stack.begin(), stack.end(), loopNode) != stack.end())
                {
                    writeCount++;
                }
            }
        }

        return writeCount;
    }

    bool hasReadBeforeWriteInLoop(KernelGraph const&         graph,
                                  int                        loopNode,
                                  int                        coordinate,
                                  int                        writeControlNode,
                                  ControlFlowRWTracer const& tracer)
    {
        auto records = tracer.coordinatesReadWrite(coordinate);

        std::vector<int> readNodes;
        for(const auto& record : records)
        {
            if(record.rw == ControlFlowRWTracer::READ)
            {
                auto stack = controlStack(record.control, graph);
                if(std::find(stack.begin(), stack.end(), loopNode) != stack.end())
                {
                    readNodes.push_back(record.control);
                }
            }
        }

        // Check if any read node comes before the write node (within loop)
        for(int readNode : readNodes)
        {
            if(readNode == writeControlNode)
            {
                continue;
            }
            auto ordering = graph.control.compareNodes(
                rocRoller::UseCacheIfAvailable, readNode, writeControlNode);
            if(ordering == NodeOrdering::LeftFirst)
            {
                return true;
            }
        }

        return false;
    }
}
