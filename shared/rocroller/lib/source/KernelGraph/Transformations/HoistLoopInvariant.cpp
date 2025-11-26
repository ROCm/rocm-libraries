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

    /**
     * @brief Check if an Assign node depends on loop variables
     */
    bool isLoopInvariant(KernelGraph const& kgraph, int assignNode, int loopNode)
    {
        // Get the Assign operation
        auto assign = kgraph.control.get<Assign>(assignNode);
        if(!assign)
            return false;

        // Extract all DataFlowTags from the expression
        auto tags = extractDataFlowTags(*assign->expression);

        // Get the loop's index coordinate (destination)
        auto loopIndexCoord = kgraph.mapper.get(loopNode, NaryArgument::DEST);

        // Check if any of the tags in the expression match the loop index
        for(auto tag : tags)
        {
            if(tag == loopIndexCoord)
                return false; // Depends on loop variable, not invariant
        }

        // Also check if any tags come from operations inside the loop
        // that might be variant
        auto loopBody  = *only(kgraph.control.getOutputNodeIndices<Body>(loopNode));
        auto bodyNodes = kgraph.control.depthFirstVisit(loopBody, Graph::Direction::Downstream)
                             .to<std::vector>();

        for(auto tag : tags)
        {
            // Check if this DataFlowTag is produced by an operation inside the loop
            for(auto nodeTag : bodyNodes)
            {
                auto element = kgraph.control.getElement(nodeTag);
                if(std::holds_alternative<Operation>(element))
                {
                    // Check if this operation produces the DataFlowTag
                    // Get all connections from this control node
                    auto connections = kgraph.mapper.getConnections(nodeTag);
                    for(const auto& conn : connections)
                    {
                        if(conn.coordinate == tag && nodeTag != assignNode)
                        {
                            // This tag is produced by another operation in the loop
                            return false;
                        }
                    }
                }
            }
        }

        return true; // No dependencies on loop variables found
    }

    /**
     * @brief Find all Assign nodes that are loop-invariant within a given loop
     */
    std::vector<int> findLoopInvariantAssigns(KernelGraph const& kgraph, int loopNode)
    {
        std::vector<int> invariantAssigns;

        // Get the loop body
        auto bodyEdges = kgraph.control.getNeighbours<Graph::Direction::Downstream>(loopNode);
        int  loopBody  = -1;
        for(auto edge : bodyEdges)
        {
            auto edgeElem = kgraph.control.getElement(edge);
            if(std::holds_alternative<ControlEdge>(edgeElem))
            {
                auto controlEdge = std::get<ControlEdge>(edgeElem);
                if(std::holds_alternative<Body>(controlEdge))
                {
                    loopBody = *only(kgraph.control.getOutputNodeIndices<Body>(loopNode));
                    break;
                }
            }
        }

        if(loopBody == -1)
            return invariantAssigns;

        // Traverse the loop body to find Assign nodes
        auto bodyNodes = kgraph.control.depthFirstVisit(loopBody, Graph::Direction::Downstream)
                             .to<std::vector>();

        for(auto nodeTag : bodyNodes)
        {
            auto element = kgraph.control.getElement(nodeTag);
            if(std::holds_alternative<Operation>(element))
            {
                auto op = std::get<Operation>(element);
                if(std::holds_alternative<Assign>(op))
                {
                    if(isLoopInvariant(kgraph, nodeTag, loopNode))
                    {
                        invariantAssigns.push_back(nodeTag);
                    }
                }
            }
        }

        return invariantAssigns;
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
        auto kgraph = original;

        // Find all ForLoopOp nodes
        auto forLoops = kgraph.control.getNodes<ForLoopOp>().to<std::vector>();

        for(auto loopNode : forLoops)
        {
            // Find loop-invariant assigns in this loop
            auto invariantAssigns = findLoopInvariantAssigns(kgraph, loopNode);

            if(invariantAssigns.empty())
                continue;

            // Find the sequence edge that leads into the loop
            auto upstreamEdges = kgraph.control.getNeighbours<Graph::Direction::Upstream>(loopNode);
            if(upstreamEdges.empty())
                continue;

            int sequenceEdge    = -1;
            int predecessorNode = -1;
            for(auto edge : upstreamEdges)
            {
                auto edgeElem = kgraph.control.getElement(edge);
                if(std::holds_alternative<ControlEdge>(edgeElem))
                {
                    auto controlEdge = std::get<ControlEdge>(edgeElem);
                    if(std::holds_alternative<Sequence>(controlEdge))
                    {
                        sequenceEdge    = edge;
                        auto inputNodes = kgraph.control.getInputNodeIndices<Sequence>(loopNode)
                                              .to<std::vector>();
                        if(!inputNodes.empty())
                            predecessorNode = inputNodes[0];
                        break;
                    }
                }
            }

            if(sequenceEdge == -1 || predecessorNode == -1)
                continue;

            // Hoist each invariant assign using the helper function
            for(auto assignNode : invariantAssigns)
            {
                // Use the helper function to hoist the node
                predecessorNode = hoistNodeBeforeLoop(
                    kgraph, assignNode, loopNode, predecessorNode, sequenceEdge);

                // Mark sequence edge as invalid after first hoist (it gets deleted in the helper)
                sequenceEdge = -1;
            }
        }

        return kgraph;
    }

    std::string HoistLoopInvariant::name() const
    {
        return "HoistLoopInvariant";
    }
}
