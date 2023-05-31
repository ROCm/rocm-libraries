
#include <limits>
#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    int common(std::deque<int> const& a, std::deque<int> const& b)
    {
        for(int i = 0; (i < a.size()) && (i < b.size()); ++i)
            if(a.at(i) != b.at(i))
                return i - 1;

        return std::min(a.size(), b.size()) - 1;
    }

    std::deque<int> LastRWTracer::controlStack(int control) const
    {
        std::deque<int> rv = {control};
        while(m_bodyParent.contains(control))
        {
            control = m_bodyParent.at(control);
            rv.push_front(control);
        }
        return rv;
    }

    std::map<int, std::set<int>> LastRWTracer::lastRWLocations() const
    {
        // Precompute all stacks
        std::map<int, std::vector<std::deque<int>>> controlStacks;
        for(auto const& x : m_trace)
        {
            controlStacks[x.coordinate].push_back(controlStack(x.control));
        }

        std::map<int, std::set<int>> rv;
        for(auto const& [coordinate, stacks] : controlStacks)
        {
            if(stacks.size() == 1)
            {
                rv[coordinate].insert(stacks.back().back());
                continue;
            }

            // Find common body-parent of all operations.
            int c = std::numeric_limits<int>().max();
            for(int i = 1; i < stacks.size(); ++i)
            {
                c = std::min(c, common(stacks[i - 1], stacks[i]));
            }

            for(auto const& stack : stacks)
            {
                AssertFatal(c + 1 < stack.size(),
                            "LastRWTracer::lastRWLocations: Stacks are identical");
                rv[coordinate].insert(stack.at(c + 1));
            }
        }

        return rv;
    }
}
