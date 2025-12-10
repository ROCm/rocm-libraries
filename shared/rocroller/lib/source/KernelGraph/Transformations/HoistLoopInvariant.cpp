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

#include <rocRoller/Expression.hpp>
#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/HoistLoopInvariant.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

#include <algorithm>
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
     * references found within it, following the visitor pattern used elsewhere
     * in the codebase (e.g., DataFlowTagPropagation).
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

    /**
     * @brief Extract all DataFlowTags referenced in an expression
     */
    std::set<int> extractDataFlowTags(Expression::Expression const& expr)
    {
        DataFlowTagExtractorVisitor visitor;
        visitor.call(expr);
        return visitor.tags;
    }

    HoistLoopInvariant::CoordinateToLoops
        HoistLoopInvariant::buildCoordinateLoopMapping(KernelGraph const&         graph,
                                                       ControlFlowRWTracer const& tracer)
    {
        CoordinateToLoops result;

        const auto records = tracer.coordinatesReadWrite();

        for(const auto& record : records)
        {
            Log::info("Processing record: coordinate {}, control node {}, rw {}",
                      record.coordinate,
                      record.control,
                      static_cast<int>(record.rw));

            if(record.rw != ControlFlowRWTracer::WRITE
               && record.rw != ControlFlowRWTracer::READWRITE)
            {
                Log::info("  Skipping record due to rw type");
                continue;
            }

            auto stack          = controlStack(record.control, graph);
            int  containingLoop = -1;
            for(auto it = stack.rbegin(); it != stack.rend(); ++it)
            {
                int node = *it;
                Log::info("  Saw in control stack node {}",
                          Graph::variantToString(graph.control.getElement(node)));

                if(graph.control.get<ForLoopOp>(node).has_value()
                   || graph.control.get<DoWhileOp>(node).has_value()
                   || graph.control.get<Scope>(node).has_value())
                {
                    containingLoop = node;
                    break;
                }
            }

            result[record.coordinate][containingLoop].insert(record.control);
        }

        return result;
    }

    int HoistLoopInvariant::hoistNodeBeforeLoop(KernelGraph& kgraph, int nodeToHoist, int loopNode)
    {
        int hoistedNode = duplicateControlNode(kgraph, nodeToHoist);
        Log::info("HoistLoopInvariant: Hoisted node {} to new node {}", nodeToHoist, hoistedNode);
        insertBefore(kgraph, loopNode, hoistedNode, hoistedNode);
        // bypassAndDelete(kgraph, nodeToHoist);
        kgraph.control.setElement(nodeToHoist, NOP{});
        return hoistedNode;
    }

    KernelGraph HoistLoopInvariant::apply(KernelGraph const& original)
    {
        auto graph = original;

        ControlFlowRWTracer tracer(graph);
        const auto          mapping = buildCoordinateLoopMapping(graph, tracer);

        Log::info("HoistLoopInvariant: Coordinate to Loop Mapping:");
        for(const auto& [coordinate, loopGroups] : mapping)
        {
            Log::info("HoistLoopInvariant: Coordinate {}:", coordinate);
            for(const auto& [loop, controlNodes] : loopGroups)
            {
                Log::info("HoistLoopInvariant:   Loop {}: Control Nodes {}",
                          loop,
                          fmt::join(controlNodes, ", "));
            }
        }

        const size_t MAX_NODES_TO_HOIST = 9999; // TODO: remove
        size_t       hoistedCount       = 0;

        for(const auto& [coordinate, loopGroups] : mapping)
        {
            for(const auto& [loopNode, controlNodes] : loopGroups)
            {
                // -1 means top-level
                if(loopNode == -1)
                    continue;

                if(controlNodes.size() != 1)
                {
                    Log::info(
                        "HoistLoopInvariant: Coordinate {} has {} writers in loop {}, skipping",
                        coordinate,
                        controlNodes.size(),
                        loopNode);
                    continue;
                }

                int controlNode = *controlNodes.begin();

                auto maybeAssign = graph.control.get<Assign>(controlNode);
                if(!maybeAssign.has_value())
                {
                    Log::info("HoistLoopInvariant: Control node {} is not an Assign, skipping",
                              controlNode);
                    continue;
                }

                auto assignNode = maybeAssign.value();

                Log::info(
                    "HoistLoopInvariant: Analyzing Assign node {} for coordinate {} in loop {}",
                    controlNode,
                    coordinate,
                    loopNode);

                auto usedTags = extractDataFlowTags(*assignNode.expression);

                bool allTagsLoopInvariant = true;
                for(auto tag : usedTags)
                {
                    if(isCoordinateWrittenInLoop(graph, loopNode, tag, tracer))
                    {
                        Log::info("HoistLoopInvariant:   DataFlowTag {} is written in loop {}, not "
                                  "invariant",
                                  tag,
                                  loopNode);
                        allTagsLoopInvariant = false;
                        break;
                    }
                }

                Log::info("HoistLoopInvariant:   Used DataFlowTags: {}, all loop invariant: {}",
                          fmt::join(usedTags, ", "),
                          allTagsLoopInvariant);

                if(allTagsLoopInvariant)
                {
                    Log::info("HoistLoopInvariant: Hoisting Assign node {} out of loop {}",
                              controlNode,
                              loopNode);
                    hoistNodeBeforeLoop(graph, controlNode, loopNode);
                    hoistedCount++;

                    if(hoistedCount >= MAX_NODES_TO_HOIST)
                    {
                        Log::info("HoistLoopInvariant: Reached maximum hoisted node count of {}, "
                                  "stopping.",
                                  MAX_NODES_TO_HOIST);
                        return graph;
                    }
                }
            }
        }
        return graph;
    }

    std::string HoistLoopInvariant::name() const
    {
        return "HoistLoopInvariant";
    }

    std::optional<int> HoistLoopInvariant::findEnclosingLoop(KernelGraph const& kgraph,
                                                             int                controlNode)
    {
        for(int parentNode : kgraph.control.nodesContaining(controlNode))
        {
            // TODO: handle other loop types
            if(kgraph.control.get<ForLoopOp>(parentNode).has_value())
            {
                return parentNode;
            }
        }
        return std::nullopt;
    }

    bool HoistLoopInvariant::isCoordinateWrittenInLoop(KernelGraph const&         kgraph,
                                                       int                        loopNode,
                                                       int                        coordinate,
                                                       ControlFlowRWTracer const& tracer)
    {
        // Get all control nodes that write to this coordinate
        auto records = tracer.coordinatesReadWrite(coordinate);

        // Helper lambda to check if a node is a descendant of the loop
        std::function<bool(int, std::set<int>&)> isDescendantOfLoop;
        isDescendantOfLoop = [&](int node, std::set<int>& visited) -> bool {
            // Avoid infinite recursion
            if(visited.count(node) > 0)
                return false;
            visited.insert(node);

            // Check if this node is directly output of the loop via Initialize, Body, or ForLoopIncrement edges
            for(int initNode : kgraph.control.getOutputNodeIndices<Initialize>(loopNode))
            {
                if(initNode == node)
                    return true;
                if(isDescendantOfLoop(initNode, visited))
                    return true;
            }

            for(int bodyNode : kgraph.control.getOutputNodeIndices<Body>(loopNode))
            {
                if(bodyNode == node)
                    return true;
                if(isDescendantOfLoop(bodyNode, visited))
                    return true;
            }

            for(int incNode : kgraph.control.getOutputNodeIndices<ForLoopIncrement>(loopNode))
            {
                if(incNode == node)
                    return true;
                if(isDescendantOfLoop(incNode, visited))
                    return true;
            }

            return false;
        };

        // Check if any write operation is within the loop
        for(const auto& record : records)
        {
            if(record.rw == ControlFlowRWTracer::WRITE
               || record.rw == ControlFlowRWTracer::READWRITE)
            {
                std::set<int> visited;
                if(isDescendantOfLoop(record.control, visited))
                {
                    return true;
                }
            }
        }

        return false;
    }
}
