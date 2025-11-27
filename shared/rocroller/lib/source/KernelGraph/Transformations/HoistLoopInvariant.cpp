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
        // First, completely disconnect the node from the loop body
        // Get all edges connected to the node
        auto upstreamEdges = kgraph.control.getNeighbours<Graph::Direction::Upstream>(nodeToHoist);
        auto downstreamEdges
            = kgraph.control.getNeighbours<Graph::Direction::Downstream>(nodeToHoist);

        // Collect predecessors and successors before removing edges
        auto nodePredecessors
            = kgraph.control.getInputNodeIndices<Sequence>(nodeToHoist).to<std::vector>();
        auto nodeSuccessors
            = kgraph.control.getOutputNodeIndices<Sequence>(nodeToHoist).to<std::vector>();

        // Remove all upstream edges to the node
        for(auto edge : upstreamEdges)
        {
            kgraph.control.deleteElement(edge);
        }

        // Remove all downstream edges from the node
        for(auto edge : downstreamEdges)
        {
            kgraph.control.deleteElement(edge);
        }

        // Connect predecessors directly to successors to maintain control flow inside loop
        for(auto pred : nodePredecessors)
        {
            for(auto succ : nodeSuccessors)
            {
                // Only add the bypass edge if it doesn't already exist
                auto existingEdges
                    = kgraph.control.getNeighbours<Graph::Direction::Downstream>(pred);
                bool alreadyConnected = false;
                for(auto edge : existingEdges)
                {
                    auto targets = kgraph.control.getOutputNodeIndices<Sequence>(pred);
                    if(std::find(targets.begin(), targets.end(), succ) != targets.end())
                    {
                        alreadyConnected = true;
                        break;
                    }
                }
                if(!alreadyConnected)
                {
                    kgraph.control.addElement(Sequence{}, {pred}, {succ});
                }
            }
        }

        // Now insert the node before the loop
        // Remove the original sequence edge if it's valid
        if(sequenceEdge != -1)
        {
            kgraph.control.deleteElement(sequenceEdge);
        }

        // Add sequence from predecessor to hoisted node
        kgraph.control.addElement(Sequence{}, {predecessorNode}, {nodeToHoist});

        // Add sequence from hoisted node to loop
        kgraph.control.addElement(Sequence{}, {nodeToHoist}, {loopNode});

        // Return the hoisted node as the new predecessor for subsequent hoisting
        return nodeToHoist;
    }

    KernelGraph HoistLoopInvariant::apply(KernelGraph const& original)
    {
        {
            ControlFlowRWTracer tracer(original);
            const auto          result = tracer.coordinatesReadWrite();

            std::unordered_map<int, std::vector<std::pair<int, ControlFlowRWTracer::ReadWrite>>>
                coordinateAccessMap;

            for(const auto& [control, coordinate, rw] : result)
            {
                std::string rwStr;
                switch(rw)
                {
                    using enum ControlFlowRWTracer::ReadWrite;
                case READ:
                    rwStr = "Read";
                    break;
                case WRITE:
                    rwStr = "Write";
                    break;
                case READWRITE:
                    rwStr = "Read/Write";
                    break;
                default:
                    rwStr = "None";
                    break;
                }
                Log::trace("Control Node: {}, Coordinate Node: {}, Access Type: {}",
                           control,
                           coordinate,
                           rwStr);

                coordinateAccessMap[coordinate].emplace_back(control, rw);
            }

            std::vector<std::pair<int, int>> singleWriteCoordinates; // (coordinate, control)

            for(const auto& [coordinate, accessList] : coordinateAccessMap)
            {
                if(accessList.size() == 1)
                {
                    const auto& [control, rw] = accessList[0];
                    if(rw == ControlFlowRWTracer::ReadWrite::WRITE)
                    {
                        singleWriteCoordinates.emplace_back(coordinate, control);
                        Log::info("Coordinate {} has single write-only access from control node {}",
                                  coordinate,
                                  control);
                    }
                }
            }

            for(const auto& [coordinate, control] : singleWriteCoordinates)
            {
                auto enclosingLoopOpt = findEnclosingLoop(original, control);
                if(!enclosingLoopOpt.has_value())
                {
                    Log::info("No enclosing loop found for control node {}, skipping hoisting",
                              control);
                    continue;
                }

                int loopNode = enclosingLoopOpt.value();
                Log::info("Enclosing loop for control node {} is loop node {}", control, loopNode);
            }
        }

        {
            std::fstream file("HoistLoopInvariant_Before.dot", std::ios::out);
            file << original.toDOT(true);
        }

        AssertFatal(false);
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
            if(kgraph.control.get<ForLoopOp>(parentNode).has_value())
            {
                return parentNode;
            }
        }
        return std::nullopt;
    }
}
