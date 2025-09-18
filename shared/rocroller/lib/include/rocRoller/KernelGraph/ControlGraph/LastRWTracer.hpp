/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
