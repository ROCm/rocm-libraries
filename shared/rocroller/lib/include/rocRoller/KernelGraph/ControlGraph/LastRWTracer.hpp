// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowArgumentTracer_fwd.hpp>

namespace rocRoller::KernelGraph
{
    class LastRWTracer : public ControlFlowRWTracer
    {
    public:
        LastRWTracer(KernelGraph const& graph, int start = -1, bool trackConnections = false)
            : ControlFlowRWTracer(graph, start, trackConnections)
        {
        }

        using ControlStack = std::deque<int>;

        /**
         * @brief Return operations that read/write coordinate last.
         *
         * Returns a map where the keys are coordinate tags, and the
         * value is a set with all of the control nodes that touch the
         * coordinate last.
         */
        std::unordered_map<int, std::set<int>> lastRWLocations() const;

        std::unordered_map<std::string, std::set<int>>
            lastArgLocations(ControlFlowArgumentTracer const& argTracer) const;

    private:
        template <typename Key>
        std::unordered_map<Key, std::set<int>> getLastLocationsFromControlStacks(
            std::unordered_map<Key, std::vector<ControlStack>> const& controlStacks) const;
    };

}
