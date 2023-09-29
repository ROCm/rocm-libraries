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
        auto graph  = original;
        auto logger = rocRoller::Log::getLogger();

        auto edges
            = filter(graph.control.isElemType<Body>(), graph.control.getEdges()).to<std::vector>();
        for(auto edge : edges)
        {
            removeBodyEdgeIfRedundant(graph, edge);
        }

        return graph;
    }

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
