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
#include <queue>
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
        TIMER(t, "removeRedundantSequenceEdges");

        std::unordered_map<int, std::unordered_set<int>> predecessors;
        std::unordered_map<int, std::vector<int>>        successors;

        for(auto const node : graph.control.getNodes())
        {
            predecessors[node];
            successors[node];
            for(auto const parent : graph.control.getInputNodeIndices<Sequence>(node))
            {
                if(predecessors.at(node).contains(parent))
                {
                    //
                    // Two nodes are connected by more than one (sequence) edge.
                    // Each pair of nodes should be connected by at most one
                    // (sequence) edge after the outer for-loop finishes.
                    //
                    graph.control.deleteElement<Sequence>(std::vector<int>{parent},
                                                          std::vector<int>{node});
                }
                else
                    predecessors.at(node).insert(parent);
            }
        }

        for(auto const& [k, pred] : predecessors)
            for(auto const& p : pred)
                successors.at(p).push_back(k);

        std::vector<int> roots;
        for(auto const& [k, pred] : predecessors)
        {
            if(pred.empty())
                roots.push_back(k);
        }

        //
        // Calculate depth of each node
        //
        std::unordered_map<int, int> depth;
        std::queue<int>              readyQueue;
        std::unordered_set<int>      visited;
        for(auto r : roots)
        {
            depth[r] = 0;
            readyQueue.push(r);
            visited.insert(r);
        }

        std::unordered_map<int, int> hit;
        while(not readyQueue.empty())
        {
            auto node = readyQueue.front();
            readyQueue.pop();

            for(auto successor : successors.at(node))
            {
                if(visited.contains(successor))
                    continue;

                depth[successor] = std::max(depth[successor], depth.at(node) + 1);
                hit[successor]++;
                if(hit.at(successor) == predecessors.at(successor).size())
                {
                    readyQueue.push(successor);
                    visited.insert(successor);
                }
            }
        }

        // TODO: sorting in unnecessary as we just want to know the maximum depth among successors
        for(auto& [node, children] : successors)
            std::sort(children.begin(), children.end(), [&](int a, int b) {
                return depth.at(a) < depth.at(b);
            });

        std::vector<int>             round = std::move(roots);
        std::vector<int>             next_round;
        std::unordered_map<int, int> seen;

        // TODO: might use a queue to replace the two buffers (round and next_round)
        while(!round.empty())
        {
            for(auto node : round)
            {
                //
                // Skip leaves as they have no successors
                //
                if(successors.at(node).empty())
                    continue;

                //
                // Check the node's successors to see if a successor can be reached (seen)
                // more than once. If true, the sequence edge between the node and that
                // successor is redundant and can be removed.
                //
                readyQueue = {}; // no clear method
                visited.clear();
                seen.clear();

                for(auto successor : successors.at(node))
                {
                    readyQueue.push(successor);
                    seen[successor]++;
                    visited.insert(successor);
                }

                //
                // Start traversal until reaching nodes that are below
                // successors
                //
                int const max_depth = depth.at(successors.at(node).back());
                while(not readyQueue.empty())
                {
                    auto const node = readyQueue.front();
                    readyQueue.pop();

                    for(auto successor : successors.at(node))
                    {
                        if(depth.at(successor) <= max_depth)
                        {
                            seen[successor]++;
                            if(not visited.contains(successor))
                            {
                                readyQueue.push(successor);
                                visited.insert(successor);
                            }
                        }
                    }
                }

                //
                // Sequence edges to the successors that have been seen
                // more than once are redundant.
                //
                for(auto const successor : successors.at(node))
                {
                    if(seen.at(successor) > 1)
                    {
                        graph.control.deleteElement<Sequence>(std::vector<int>{node},
                                                              std::vector<int>{successor});
                    }
                }

                //
                // Insert successors that are ready (all its parents are
                // visited) for next round
                //
                for(auto const successor : successors.at(node))
                {
                    auto const remaining = (--hit.at(successor));
                    if(remaining == 0)
                        next_round.push_back(successor);
                }
            }

            round.swap(next_round);
            next_round.clear();
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
    void removeRedundantBodyEdges(KernelGraph& graph)
    {
        TIMER(t, "removeRedundantBodyEdges");

        //
        // The idea is to use depth to identify redundant Body edges, where
        // "depth" means the longest path from root to the node.
        //
        // For example,
        //
        //       N0
        //        @
        //        |\ (Body)
        //  (Body)| \
        //        |  @ N1
        //        | /
        //        |/ (Sequence)
        //        @
        //       N2
        //
        // In this case, N0 happens to be the root node (i.e., depth(N0)=0),
        // and depth(N1) = 1, depth(N2) = 2. So edge N0-N2 is redundant.
        //
        // Notice that this approach assumes a Body node can only have
        // Sequence edge coming from another Body node with the same
        // parent. The approach will produce incorrect results if this
        // assumption does not hold.
        //

        std::unordered_map<int, std::unordered_set<int>> predecessors;
        std::unordered_map<int, std::unordered_set<int>> successors;

        std::unordered_map<int, std::unordered_set<int>> bodyPredecessors;

        for(auto const node : graph.control.getNodes())
        {
            bodyPredecessors[node];
            for(auto const parent : graph.control.getInputNodeIndices<Body>(node))
            {
                if(bodyPredecessors.at(node).contains(parent))
                {
                    //
                    // Two nodes are connected by more than one (Body) edge.
                    // Each pair of nodes should be connected by at most one
                    // (Body) edge after the outer for-loop finishes.
                    //
                    graph.control.deleteElement<Body>(std::vector<int>{parent},
                                                      std::vector<int>{node});
                }
                else
                    bodyPredecessors.at(node).insert(parent);
            }

            predecessors[node];
            successors[node];
            for(auto const parent : graph.control.getInputNodeIndices<Sequence>(node))
            {
                predecessors.at(node).insert(parent);
                successors[parent].insert(node);
            }
        }

        //
        // Include nodes via Body edges as successors (and predecessors)
        //
        for(auto const& [node, bPredecessors] : bodyPredecessors)
        {
            for(auto pred : bPredecessors)
            {
                successors.at(pred).insert(node);
                predecessors.at(node).insert(pred);
            }
        }

        std::unordered_map<int, int> depth;
        std::queue<int>              readyQueue;
        std::unordered_set<int>      visited;
        for(auto const& [node, pred] : predecessors)
        {
            if(pred.empty())
            {
                depth[node] = 0;
                readyQueue.push(node);
                visited.insert(node);
            }
        }

        //
        // Calculate the depth of each node
        //
        std::unordered_map<int, int> hit;
        while(not readyQueue.empty())
        {
            auto node = readyQueue.front();
            readyQueue.pop();

            for(auto successor : successors.at(node))
            {
                if(visited.contains(successor))
                    continue;

                depth[successor] = std::max(depth[successor], depth.at(node) + 1);
                hit[successor]++;
                if(hit.at(successor) == predecessors.at(successor).size())
                {
                    readyQueue.push(successor);
                    visited.insert(successor);
                }
            }
        }

        //
        // Check nodes via Body edges. If depth(node) != depth(predecessor) + 1, the Body
        // edge between the node and its predecessor is redundant and can be deleted.
        //
        for(auto const& [node, bPredecessors] : bodyPredecessors)
        {
            for(auto pred : bPredecessors)
            {
                AssertFatal(depth.at(pred) < depth.at(node),
                            "node's predecessor should have smaller depth");
                if(depth.at(pred) + 1 != depth.at(node))
                    graph.control.deleteElement<Body>(std::vector<int>{pred},
                                                      std::vector<int>{node});
            }
        }
    }

    /*
     * Helper for removeRedundantNOPs (early return).
     *
     * Remove NOPs that have multiple inputs/outputs (MIMO) if they are redundant
     *
     * Returns true if it modified the graph; false otherwise.
     */
    static bool removeMIMONOPIfRedundant(KernelGraph& graph, int nop)
    {
        auto hasBody = !empty(graph.control.getOutputNodeIndices<Body>(nop));
        if(hasBody)
            return false;

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

    /*
     * Helper for removeRedundantNOPs (early return).
     *
     * Remove NOPs that have single input/output (SISO) if they are redundant
     *
     * Returns true if it modified the graph; false otherwise.
     */
    static bool removeSISONOPIfRedundant(KernelGraph& graph, int nop)
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

        return false;
    }

    /**
     * @brief Remove redundant NOP nodes.
     */
    void removeRedundantNOPs(KernelGraph& graph)
    {
        removeRedundantSequenceEdges(graph);

        auto const removeNOPFcns = {removeSISONOPIfRedundant, removeMIMONOPIfRedundant};

        for(auto const& fcn : removeNOPFcns)
        {
            bool graphChanged = true;
            while(graphChanged)
            {
                graphChanged = false;
                auto nops    = graph.control.getNodes<NOP>().to<std::vector>();
                for(auto nop : nops)
                    graphChanged |= fcn(graph, nop);

                if(graphChanged)
                {
                    removeRedundantSequenceEdges(graph);
                }
            }
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
