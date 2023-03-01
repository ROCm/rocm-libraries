
#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    using namespace CoordinateGraph;
    using namespace ControlGraph;

    KernelGraph addDeallocate(KernelGraph const& original)
    {
        TIMER(t, "KernelGraph::addDeallocate");
        rocRoller::Log::getLogger()->debug("KernelGraph::addDeallocate()");

        auto graph  = original;
        auto tracer = ControlFlowRWTracer(graph);

        tracer.trace();
        for(auto kv : tracer.deallocateLocations())
        {
            auto coordinate = kv.first;
            auto control    = kv.second;
            auto deallocate = graph.control.addElement(Deallocate());

            std::vector<int> children;
            for(auto elem : graph.control.getNeighbours<Graph::Direction::Downstream>(control))
            {
                auto sequence = graph.control.get<Sequence>(elem);
                if(sequence)
                {
                    for(auto child :
                        graph.control.getNeighbours<Graph::Direction::Downstream>(elem))
                    {
                        children.push_back(child);
                    }
                    graph.control.deleteElement(elem);
                }
            }

            graph.control.addElement(Sequence(), {control}, {deallocate});

            for(auto child : children)
            {
                graph.control.addElement(Sequence(), {deallocate}, {child});
            }

            graph.mapper.connect<Dimension>(deallocate, coordinate);
        }

        return graph;
    }
}
