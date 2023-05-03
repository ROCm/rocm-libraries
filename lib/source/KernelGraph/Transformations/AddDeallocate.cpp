
#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    using namespace CoordinateGraph;
    using namespace ControlGraph;

    namespace CF = rocRoller::KernelGraph::ControlGraph;
    using GD     = rocRoller::Graph::Direction;

    int common(std::deque<int> const& a, std::deque<int> const& b)
    {
        for(int i = 0; (i < a.size()) && (i < b.size()); ++i)
            if(a.at(i) != b.at(i))
                return i - 1;

        return std::min(a.size(), b.size()) - 1;
    }

    class AddDeallocateTracer : public ControlFlowRWTracer
    {
    public:
        AddDeallocateTracer(KernelGraph const& graph)
            : ControlFlowRWTracer(graph)
        {
        }

        std::deque<int> controlStack(int control) const
        {
            std::deque<int> rv = {control};
            while(m_bodyParent.contains(control))
            {
                control = m_bodyParent.at(control);
                rv.push_front(control);
            }
            return rv;
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
            std::map<int, std::set<int>> rv;

            std::map<int, std::deque<int>> stacks;

            for(auto x : m_trace)
            {
                if(stacks.contains(x.coordinate))
                {
                    auto last    = stacks.at(x.coordinate);
                    auto current = controlStack(x.control);

                    // Current stack looks like:
                    //   { kernel, body-parent, body-parent, ..., body-parent, operation }
                    // Last stack looks like:
                    //   { kernel, body-parent, ..., body-parent or operation }

                    // Find common body-parent
                    int c = common(last, current);
                    AssertFatal(c >= 0);

                    if(c == 0)
                    {
                        // Kernel is the only common parent.
                        stacks[x.coordinate] = {last.at(0)};

                        // Deallocate after Kernel.
                        rv[x.coordinate] = {stacks[x.coordinate].back()};
                    }
                    else if((c == last.size() - 1) && (c < current.size() - 1))
                    {
                        // Top of last is a body-parent, current is
                        // contained in last.
                        //
                        // Deallocate after last body-parent.
                        rv[x.coordinate] = {stacks[x.coordinate].back()};
                    }
                    else if((c == current.size() - 1) && (c < last.size() - 1))
                    {
                        Throw<FatalError>("Invalid operation stack during AddDeallocate.");
                    }
                    else
                    {
                        // Some shared, some not.

                        stacks[x.coordinate] = {};
                        std::copy(last.cbegin(),
                                  last.cbegin() + c + 1,
                                  std::back_inserter(stacks[x.coordinate]));

                        if((c < last.size() - 1) && (c < current.size() - 1))
                        {
                            // If one can be reached by following
                            // Sequence edges only, then we can safely
                            // deallocate after the later one.

                            auto onlyFollowSequenceEdges = [&](int x) -> bool {
                                auto isSequence
                                    = CF::isEdge<Sequence>(m_graph.control.getElement(x));
                                return isSequence;
                            };

                            auto reachable = m_graph.control
                                                 .depthFirstVisit(last.at(c + 1),
                                                                  onlyFollowSequenceEdges,
                                                                  GD::Downstream)
                                                 .to<std::set>();

                            if(reachable.contains(current.at(c + 1)))
                            {
                                for(auto it = last.cbegin() + c + 1; it != last.cend(); ++it)
                                {
                                    rv[x.coordinate].erase(*it);
                                }
                                rv[x.coordinate].insert(current.at(c + 1));
                                stacks[x.coordinate].push_back(current.at(c + 1));
                            }
                            else
                            {
                                rv[x.coordinate].insert(last.at(c + 1));
                                rv[x.coordinate].insert(current.at(c + 1));
                            }
                        }
                        else if(c < current.size() - 1)
                        {
                            // Not connected; deallocate after both.
                            rv[x.coordinate].insert(current.at(c + 1));
                        }
                        else if(c < last.size() - 1)
                        {
                            // Not connected; deallocate after both.
                            rv[x.coordinate].insert(last.at(c + 1));
                        }
                    }
                }
                else
                {
                    // First time we've seen this coordinate.
                    stacks[x.coordinate] = controlStack(x.control);
                    rv[x.coordinate]     = {stacks[x.coordinate].back()};
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
