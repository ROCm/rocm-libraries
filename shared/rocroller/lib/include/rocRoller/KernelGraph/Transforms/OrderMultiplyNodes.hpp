/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        class OrderMultiplyNodes : public GraphTransform
        {
        public:
            OrderMultiplyNodes() = default;

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "OrderMultiplyNodes";
            }

            inline std::vector<GraphConstraint> preConstraints() const override
            {
                return {};
            }

        private:
            /**
             * Divides `nodes` into groups that are:
             *  - Within the same for loop as each other
             *  - All unordered relative to each other.
             *
             * There shouldn't be any body relationships between `nodes`.
             */
            std::vector<std::set<int>> getLocallyUnorderedGroups(KernelGraph const& graph,
                                                                 std::set<int>      nodes);

            /**
            * Attempts to sort `nodes` by the order of dependent downstream memory nodes.
            */
            std::vector<int> sortNodesByDownstreamMemoryNodes(KernelGraph const& graph,
                                                              std::set<int>&     nodes);

            std::vector<int> getReversedTagDependencies(KernelGraph const&         graph,
                                                        ControlFlowRWTracer const& tracer,
                                                        int                        node);

            std::vector<int> sortNodesByLastTagDependencies(KernelGraph const&   graph,
                                                            std::set<int> const& nodes);

            std::vector<std::vector<int>> findAndOrderGroups(KernelGraph const& graph);
        };
    }
}
