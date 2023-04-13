
#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    using namespace CoordinateGraph;
    using namespace ControlGraph;

    class AddDeallocateTracer : public ControlFlowRWTracer
    {
    public:
        AddDeallocateTracer(KernelGraph const& graph)
            : ControlFlowRWTracer(graph)
        {
        }

        /**
         * @brief Returns a mapping of all of the control nodes that can
         *        deallocate a value in the coordinate graph.
         *
         *        Returns a map where the key is the coordinate to deallocate
         *        and the value is a container with all of the control nodes
         *        that should deallocate the value.
         *
         * @return std::map<int, std::set<int>>
         */
        std::map<int, std::set<int>> deallocateLocations() const
        {
            // track operation(s) that concurrently touch coordinate
            std::map<int, std::set<int>> rv;

            // track event that touches coordinate last, at lowest
            // depth (closest to kernel)
            std::map<int, EventRecord> ev;

            // Pass 1 : populate the [coordinate, controls] map
            for(auto x : m_trace)
            {
                // have we seen this coordinate before?
                if(ev.count(x.coordinate) > 0)
                {
                    // if last recorded event is at the depth less than
                    // the new depth, deallocate at a new lower
                    // depth bodyParent location, keeping the lowest
                    // depth recorded event intact.
                    if(ev.at(x.coordinate).depth < x.depth)
                    {
                        auto n       = x.depth - ev.at(x.coordinate).depth;
                        auto control = x.control;
                        while(n > 0)
                        {
                            control = m_bodyParent.at(control);
                            n--;
                        }
                        rv[x.coordinate].clear();
                        rv[x.coordinate].insert(control);
                        continue;
                    }

                    // if last recorded event is at the depth more than
                    // the new depth, deallocate at the lower depth location.
                    if(ev.at(x.coordinate).depth > x.depth)
                    {
                        rv[x.coordinate].clear();
                    }
                }
                rv[x.coordinate].insert(x.control);
                ev.insert_or_assign(x.coordinate, x);
            }

            // Pass 2 : update the controls in the [coordinate, controls] map to handle
            // situations where the deallocate locations inside controls are at the
            // same depth but have different bodyParents.
            // Keep going up the bodyParent ladder until all deallocate locations inside
            // controls have a common bodyParent.
            for(auto& [coordinate, controls] : rv)
            {
                std::set<int> temp;
                while(temp.empty() && controls.size() > 1)
                {
                    for(auto const& src : controls)
                    {
                        auto parent = m_bodyParent.at(src);
                        if(temp.find(parent) == temp.end())
                        {
                            temp.insert(parent);
                        }
                    }
                    // if we haven't reached a common bodyParent yet.
                    if(temp.size() != 1)
                    {
                        controls = temp;
                        temp.clear();
                    }
                }
            }
            return rv;
        }
    };

    KernelGraph addDeallocate(KernelGraph const& original)
    {
        TIMER(t, "KernelGraph::addDeallocate");
        rocRoller::Log::getLogger()->debug("KernelGraph::addDeallocate()");

        auto graph  = original;
        auto tracer = AddDeallocateTracer(graph);

        tracer.trace();
        for(auto& [coordinate, controls] : tracer.deallocateLocations())
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
