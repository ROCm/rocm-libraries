#pragma once

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>

namespace rocRoller::KernelGraph
{
    class LastRWTracer : public ControlFlowRWTracer
    {
    public:
        LastRWTracer(KernelGraph const& graph, int start = -1, bool trackConnections = false)
            : ControlFlowRWTracer(graph, start, trackConnections)
        {
        }

        /**
         * @brief Return call-stack control operation.
         *
         * The return value is a deque of body-parents of the control
         * node.
         */
        std::deque<int> controlStack(int control) const;

        /**
         * @brief Return operations that read/write coordinate last.
         *
         * Returns a map where the keys are coordinate tags, and the
         * value is a set with all of the control nodes that touch the
         * coordinate last.
         *
         * @return std::map<int, std::set<int>>
         */
        std::map<int, std::set<int>> lastRWLocations() const;
    };

}
