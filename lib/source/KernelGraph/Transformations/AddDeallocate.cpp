
#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddDeallocate.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    using namespace CoordinateGraph;
    using namespace ControlGraph;

    KernelGraph AddDeallocate::apply(KernelGraph const& original)
    {
        TIMER(t, "KernelGraph::addDeallocate");
        rocRoller::Log::getLogger()->debug("KernelGraph::addDeallocate()");

        auto graph  = original;
        auto tracer = LastRWTracer(graph);

        for(auto& [coordinate, controls] : tracer.lastRWLocations())
        {
            // If there is a single entry in the hot-loop set when we
            // insert the Deallocate operation into the control graph,
            // the Deallocate will be added after the hot loop.
            std::set<int> hotLoop;

            // Add all containing loops of read/writes of an LDS
            // allocation to the hot-loop set.
            //
            // The effect is: if all read/writes of an LDS coordinate
            // happen in a single loop; we wait until after the loop
            // is done before deallocating.  This avoids LDS
            // allocation re-use within a hot-loop that may lead to
            // undefined behaviour.
            auto maybeLDS = graph.coordinates.get<LDS>(coordinate);
            if(maybeLDS)
            {
                for(auto control : controls)
                {
                    auto maybeForLoop = findContainingOperation<ForLoopOp>(control, graph);
                    if(maybeForLoop)
                        hotLoop.insert(*maybeForLoop);
                }
            }

            // Create a Deallocate operation
            auto deallocate = graph.control.addElement(Deallocate());
            graph.mapper.connect<Dimension>(deallocate, coordinate);

            if(hotLoop.size() != 1)
            {
                // Add sequence edges from each "last r/w" operation.
                //
                // There is either a single "last r/w" operation; or
                // are all of them are within the same body-parent.
                for(auto src : controls)
                    graph.control.addElement(Sequence(), {src}, {deallocate});
            }
            else
            {
                // There is a single hot-loop, add the Deallocate
                // after it.
                graph.control.addElement(Sequence(), {*hotLoop.cbegin()}, {deallocate});
            }
        }

        return graph;
    }
}
