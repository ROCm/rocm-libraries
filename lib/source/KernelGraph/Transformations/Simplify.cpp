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
    KernelGraph removeRedundantSequenceEdges(KernelGraph const& original)
    {
        auto graph  = original;
        auto logger = rocRoller::Log::getLogger();

        auto edges = filter(graph.control.isElemType<Sequence>(), graph.control.getEdges())
                         .to<std::vector>();
        for(auto edge : edges)
        {
            auto onlyFollowDifferentSequenceEdges = [&](int x) -> bool {
                auto isSame     = x == edge;
                auto isSequence = CF::isEdge<Sequence>(graph.control.getElement(x));
                return !isSame && isSequence;
            };

            auto tail = *only(graph.control.getNeighbours<GD::Upstream>(edge));
            auto head = *only(graph.control.getNeighbours<GD::Downstream>(edge));

            auto reachable
                = graph.control
                      .depthFirstVisit(tail, onlyFollowDifferentSequenceEdges, GD::Downstream)
                      .to<std::unordered_set>();

            if(reachable.contains(head))
            {
                logger->debug("Deleting redundant Sequence edge: {}", edge);
                graph.control.deleteElement(edge);
            }
        }
        return graph;
    }

    /*
     * Helper for removeRedundantBodyEdges (early return).
     */
    void removeBodyEdgeIfRedundant(KernelGraph& graph, int edge)
    {
        auto logger = rocRoller::Log::getLogger();

        auto tail = *only(graph.control.getNeighbours<GD::Upstream>(edge));
        auto head = *only(graph.control.getNeighbours<GD::Downstream>(edge));

        auto onlyFollowDifferentBodyEdges = [&](int x) -> bool {
            auto isSame = x == edge;
            auto isBody = CF::isEdge<Body>(graph.control.getElement(x));
            return !isSame && isBody;
        };

        auto reachable
            = graph.control.depthFirstVisit(tail, onlyFollowDifferentBodyEdges, GD::Downstream)
                  .to<std::unordered_set>();

        if(reachable.contains(head))
        {
            logger->debug("Deleting redundant Body edge: {}", edge);
            graph.control.deleteElement(edge);
            return;
        }

        auto otherBodies = graph.control.getOutputNodeIndices<Body>(tail).to<std::unordered_set>();
        otherBodies.erase(head);

        for(auto top : otherBodies)
        {
            auto reachable
                = graph.control
                      .depthFirstVisit(top, graph.control.isElemType<Sequence>(), GD::Downstream)
                      .to<std::unordered_set>();

            if(reachable.contains(head))
            {
                logger->debug("Deleting redundant Body edge: {}", edge);
                graph.control.deleteElement(edge);
                return;
            }
        }
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
    KernelGraph removeRedundantBodyEdges(KernelGraph const& original)
    {
        auto graph = original;

        auto edges
            = filter(graph.control.isElemType<Body>(), graph.control.getEdges()).to<std::vector>();
        for(auto edge : edges)
            removeBodyEdgeIfRedundant(graph, edge);

        return graph;
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

        // If we get here, you can get to all outputs from each input
        // without going through the NOP.  Delete it!

        auto incoming = only(graph.control.getNeighbours<GD::Upstream>(nop));
        auto outgoing = only(graph.control.getNeighbours<GD::Downstream>(nop));
        if((!incoming) || (!outgoing))
            return false;
        graph.control.deleteElement(*incoming);
        graph.control.deleteElement(*outgoing);
        graph.control.deleteElement(nop);

        return true;
    }

    /**
     * @brief Remove redundant NOP nodes.
     */
    KernelGraph removeRedundantNOPs(KernelGraph const& original)
    {
        auto graph        = removeRedundantSequenceEdges(original);
        bool graphChanged = true;
        while(graphChanged)
        {
            graphChanged = false;
            auto nops    = graph.control.getNodes<NOP>().to<std::vector>();
            for(auto nop : nops)
                graphChanged |= removeNOPIfRedundant(graph, nop);
            if(graphChanged)
                graph = removeRedundantSequenceEdges(graph);
        }
        return graph;
    }

    // Remove removeRedundantNOPs is not fully tested.
    //
    // Note: if we remove enough NOPs, some edges might become
    // redundant.  Do fixed-point iterations, or is that too slow?

    KernelGraph Simplify::apply(KernelGraph const& original)
    {
        TIMER(t, "KernelGraph::Simplify");
        auto graph = removeRedundantSequenceEdges(original);
        graph      = removeRedundantBodyEdges(graph);
        return graph;
    }

    std::string Simplify::name() const
    {
        return "Simplify";
    }
}
