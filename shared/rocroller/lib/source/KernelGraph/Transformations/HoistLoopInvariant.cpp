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

    int HoistLoopInvariant::hoistNodeBeforeLoop(
        KernelGraph& kgraph, int nodeToHoist, int loopNode, int predecessorNode, int sequenceEdge)
    {
        // Duplicate the node we want to hoist
        int hoistedNode = duplicateControlNode(kgraph, nodeToHoist);

        // Insert the duplicated node before the loop
        // insertBefore will handle redirecting incoming edges and creating the sequence
        insertBefore(kgraph, loopNode, hoistedNode, hoistedNode);

        // Remove the original node from inside the loop while maintaining connectivity
        bypassAndDelete(kgraph, nodeToHoist);

        // Return the hoisted node as the new predecessor for subsequent hoisting
        return hoistedNode;
    }

    KernelGraph HoistLoopInvariant::apply(KernelGraph const& original)
    {

        {
            // std::fstream file("HoistLoopInvariant_before.dot");
            // file << original.toDOT(true);
        }

        {
            ControlFlowRWTracer tracer(original);
            const auto          result = tracer.coordinatesReadWrite();

            std::unordered_map<int, std::vector<int>> coordinateAccessMap;

            for(const auto& [control, coordinate, rw] : result)
            {
                if(rw == ControlFlowRWTracer::ReadWrite::WRITE)
                    coordinateAccessMap[coordinate].emplace_back(control);
            }

            std::vector<std::pair<int, int>> singleWriteCoordinates; // (coordinate, control)

            for(const auto& [coordinate, accessList] : coordinateAccessMap)
            {
                if(accessList.size() == 1)
                {
                    const auto control = accessList[0];
                    singleWriteCoordinates.emplace_back(coordinate, control);
                    Log::trace("Coordinate {} has single write-only access from control node {}",
                               coordinate,
                               control);
                }
            }

            for(const auto& [coordinate, control] : singleWriteCoordinates)
            {
                auto enclosingLoopOpt = findEnclosingLoop(original, control);

                if(!enclosingLoopOpt.has_value())
                {
                    Log::trace("No enclosing loop found for control node {}, skipping", control);
                    continue;
                }

                int loopNode = enclosingLoopOpt.value();

                auto assignNodeOpt = original.control.get<Assign>(control);
                if(!assignNodeOpt.has_value())
                {
                    Log::trace("Control node {} is not an Assign node, skipping", control);
                    auto element = original.control.getElement(control);
                    continue;
                }

                auto assignNode = assignNodeOpt.value();

                if(assignNode.expression == nullptr)
                {
                    Log::trace("Assign node {} has no expression, skipping", control);
                    continue;
                }

                auto usedTags = extractDataFlowTags(*assignNode.expression);

                bool allTagsLoopInvariant = true;
                for(auto tag : usedTags)
                {
                    const auto isWrittenInLoop
                        = isCoordinateWrittenInLoop(original, loopNode, tag, tracer);

                    if(isWrittenInLoop)
                    {
                        allTagsLoopInvariant = false;
                        break;
                    }
                }

                if(allTagsLoopInvariant && !usedTags.empty())
                {
                    Log::info("Hoisting Assign node {} before loop node {}, uses tags {}",
                              control,
                              loopNode,
                              usedTags);

                    auto loopPredecessors
                        = original.control.getInputNodeIndices<ControlEdge>(loopNode)
                              .to<std::vector>();

                    for(auto pred : loopPredecessors)
                    {
                        Log::info("pred {} {}",
                                  pred,
                                  Graph::variantToString(original.control.getElement(pred)));
                    }

                    AssertFatal(
                        loopPredecessors.size() == 1,
                        fmt::format("Got {} predecessors for loop node {} {}",
                                    loopPredecessors.size(),
                                    loopNode,
                                    Graph::variantToString(original.control.getElement(loopNode))));

                    int predecessorNode = loopPredecessors[0]; // Take the first predecessor

                    hoistNodeBeforeLoop(
                        const_cast<KernelGraph&>(original), control, loopNode, predecessorNode, -1);
                }
            }
        }

        {
            // std::fstream file("HoistLoopInvariant_after.dot");
            // file << original.toDOT(true);
        }

        return original;
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
