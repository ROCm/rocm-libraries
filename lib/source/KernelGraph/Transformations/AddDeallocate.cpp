
#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
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
        auto tracer = LastRWTracer(graph);

        tracer.trace();
        for(auto& [coordinate, controls] : tracer.lastRWLocations())
        {
            auto deallocate = graph.control.addElement(Deallocate());

            for(auto src : controls)
            {
                graph.control.addElement(Sequence(), {src}, {deallocate});
            }

            graph.mapper.connect<Dimension>(deallocate, coordinate);
        }

        return graph;
    }
}
