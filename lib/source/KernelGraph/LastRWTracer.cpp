/*
 * When analysing when registers are modified, we construct a
 * "read/write" tree, where leaves are operations that can modify
 * registers and interior nodes are body-parents.  The root of the
 * tree is the kernel.  For example, the tree corresponding to the
 * pseudo-code:
 *
 *     v[0:3] = 0                 # Assign(2)
 *     for i in range(4):         # ForLoopOp(3)
 *         v[i] = i               # Assign(4)
 *
 * is:
 *
 *                           Kernel(1)
 *                            /    \
 *                   Assign(2)     ForLoopOp(3)
 *                                      \
 *                                     Assign(4)
 *
 * When we refer to a "control stack" in the LastRWTracer
 * implementation below, we mean a simple vector representing the path
 * from root of the tree to the leaf that modifies the register in
 * question.
 *
 * In the example above, there are two stacks for the `v` register:
 *
 * 1. The stack for Assign(2) is: [1, 2]
 * 2. The stack for Assign(4) is: [1, 3, 4]
 *
 * The common piece between those two stacks is simply [1], and hence
 * the return value of `common` is 0.
 *
 * The return value of `lastRWLocations` is then: { `v`: [2, 3] }.
 *
 * At the time of writing, we want to also track register usage inside
 * the condition expressions of ForLoopOps.  These are bit tricky,
 * because they aren't really leaves.
 */

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

            // Extra pass to handle conditions in ForLoopOps; bump c
            // so that ForLoopOps become leaf-like.
            for(auto const& stack : stacks)
            {
                if(c + 1 >= stack.size())
                    c = std::max(0, c - 1);
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
