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

#include <map>
#include <unordered_set>
#include <vector>

#include <rocRoller/Graph/GraphUtilities.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller::KernelGraph
{
    namespace CF = rocRoller::KernelGraph::ControlGraph;
    using GD     = rocRoller::Graph::Direction;

    using namespace ControlGraph;

    /**
     * @brief Remove redundant Sequence edges in the control graph.
     *
     * Consider a control graph similar to:
     *
     *        @
     *        |\ B
     *      A | \
     *        |  @
     *        | /
     *        |/ C
     *        @
     *
     * where @ are operations and all edges are Sequence edges.  The A
     * edge is redundant.
     */
    void removeRedundantSequenceEdges(KernelGraph& graph)
    {
        auto isSequence = graph.control.isElemType<Sequence>();
        for(auto edge : Graph::findRedundantEdges(graph.control, isSequence))
        {
            Log::debug("Deleting redundant Sequence edge: {}", edge);
            graph.control.deleteElement(edge);
        }
    }

    /*
     * Helper for removeRedundantBodyEdges (early return).
     */
    bool isRedundantBodyEdge(KernelGraph const& graph, int edge)
    {
        auto tail = *only(graph.control.getNeighbours<GD::Upstream>(edge));
        auto head = *only(graph.control.getNeighbours<GD::Downstream>(edge));

        auto onlyFollowDifferentBodyEdges = [&](int x) -> bool {
            auto isSame = x == edge;
            auto isBody = CF::isEdge<Body>(graph.control.getElement(x));
            return !isSame && isBody;
        };

        {
            auto reachable
                = !graph.control.depthFirstVisit(tail, onlyFollowDifferentBodyEdges, GD::Downstream)
                       .filter([head](int x) { return x == head; })
                       .empty();

            if(reachable)
                return true;
        }

        auto otherBodies = graph.control.getOutputNodeIndices<Body>(tail).filter(
            [head](int x) { return x != head; });

        for(auto top : otherBodies)
        {
            auto reachable
                = !graph.control
                       .depthFirstVisit(top, graph.control.isElemType<Sequence>(), GD::Downstream)
                       .filter([head](int x) { return x == head; })
                       .empty();

            if(reachable)
                return true;
        }
        return false;
    }

    /**
     * @brief Remove redundant Body edges in the control graph.
     *
     * Consider a control graph similar to:
     *
     *        @
     *        |\ B
     *      A | \
     *        |  @
     *        | /
     *        |/ C
     *        @
     *
     * where @ are operations; the A and B edges are Body edges, and
     * the C edge is a Sequence edge.  The A edge is redundant.
     *
     * If there were another Body edge parallel to A, it would also be
     * redundant and should be removed.
     */
    void removeRedundantBodyEdges(KernelGraph& graph)
    {
        auto edges
            = graph.control.getEdges().filter(graph.control.isElemType<Body>()).to<std::vector>();
        for(auto edge : edges)
        {
            if(isRedundantBodyEdge(graph, edge))
            {
                Log::debug("Deleting redundant Body edge: {}", edge);
                graph.control.deleteElement(edge);
            }
        }
    }

    /*
     * Helper for removeRedundantNOPs (early return).
     *
     * Returns true if it modified the graph; false otherwise.
     */
    bool removeNOPIfRedundant(KernelGraph& graph, int nop)
    {
        auto hasBody = !empty(graph.control.getOutputNodeIndices<Body>(nop));
        if(hasBody)
            return false;

        //
        // If the NOP has a single incoming Sequence edge, it's redundant.
        //
        auto singleIncomingSequence = only(graph.control.getInputNodeIndices<Sequence>(nop));
        if(singleIncomingSequence)
        {
            auto inNode = *singleIncomingSequence;

            auto incomingEdge = only(graph.control.getNeighbours<GD::Upstream>(nop));
            if(!incomingEdge)
                return false;

            auto outgoingEdges = graph.control.getNeighbours<GD::Downstream>(nop).to<std::vector>();

            for(auto outEdge : outgoingEdges)
            {
                auto outNode = *only(graph.control.getNeighbours<GD::Downstream>(outEdge));
                graph.control.addElement(Sequence(), {inNode}, {outNode});
            }

            Log::debug("Deleting single-incoming NOP: {}", nop);

            graph.control.deleteElement(*incomingEdge);
            for(auto outgoingEdge : outgoingEdges)
                graph.control.deleteElement(outgoingEdge);
            graph.control.deleteElement(nop);

            return true;
        }

        //
        // If the NOP has a single outgoing Sequence edge, it's redundant.
        //
        auto singleOutgoingSequence = only(graph.control.getOutputNodeIndices<Sequence>(nop));
        if(singleOutgoingSequence)
        {
            auto outNode = *singleOutgoingSequence;

            auto outgoingEdge = only(graph.control.getNeighbours<GD::Downstream>(nop));
            if(!outgoingEdge)
                return false;

            auto incomingEdges = graph.control.getNeighbours<GD::Upstream>(nop).to<std::vector>();

            for(auto inEdge : incomingEdges)
            {
                auto edgeType = graph.control.getElement(inEdge);
                auto inNode   = *only(graph.control.getNeighbours<GD::Upstream>(inEdge));
                graph.control.addElement(edgeType, {inNode}, {outNode});
            }

            Log::debug("Deleting single-outgoing NOP: {}", nop);

            graph.control.deleteElement(*outgoingEdge);
            for(auto incomingEdge : incomingEdges)
                graph.control.deleteElement(incomingEdge);
            graph.control.deleteElement(nop);

            return true;
        }

        //
        // Can we reach all output nodes from each input node?
        //
        auto inputs = graph.control.getInputNodeIndices(nop, [](ControlEdge) { return true; })
                          .to<std::vector>();
        auto outputs = graph.control.getOutputNodeIndices(nop, [](ControlEdge) { return true; })
                           .to<std::vector>();

        auto dontGoThroughNOP = [&](int x) -> bool {
            auto tail = *only(graph.control.getNeighbours<GD::Upstream>(x));
            auto head = *only(graph.control.getNeighbours<GD::Downstream>(x));
            return (nop != tail) && (nop != head);
        };

        for(auto input : inputs)
        {
            auto reachable = graph.control.depthFirstVisit(input, dontGoThroughNOP, GD::Downstream)
                                 .to<std::unordered_set>();
            for(auto output : outputs)
            {
                if(!reachable.contains(output))
                    return false;
            }
        }

        // If we get here, we can reach all outputs from each input
        // without going through the NOP.  Delete it!

        auto incoming = only(graph.control.getNeighbours<GD::Upstream>(nop));
        auto outgoing = only(graph.control.getNeighbours<GD::Downstream>(nop));
        if((!incoming) || (!outgoing))
            return false;

        Log::debug("Deleting redundant NOP: {}", nop);

        graph.control.deleteElement(*incoming);
        graph.control.deleteElement(*outgoing);
        graph.control.deleteElement(nop);

        return true;
    }

    /**
     * @brief Remove redundant NOP nodes.
     */
    void removeRedundantNOPs(KernelGraph& graph)
    {
        removeRedundantSequenceEdges(graph);
        bool graphChanged = true;
        while(graphChanged)
        {
            graphChanged = false;
            auto nops    = graph.control.getNodes<NOP>().to<std::vector>();
            for(auto nop : nops)
                graphChanged |= removeNOPIfRedundant(graph, nop);
            if(graphChanged)
                removeRedundantSequenceEdges(graph);
        }
    }

    // Remove removeRedundantNOPs is not fully tested.
    //
    // Note: if we remove enough NOPs, some edges might become
    // redundant.  Do fixed-point iterations, or is that too slow?

    KernelGraph Simplify::apply(KernelGraph const& original)
    {
        TIMER(t, "KernelGraph::Simplify");
        auto graph = original;
        removeRedundantSequenceEdges(graph);
        removeRedundantBodyEdges(graph);
        return graph;
    }

    std::string Simplify::name() const
    {
        return "Simplify";
    }
}
