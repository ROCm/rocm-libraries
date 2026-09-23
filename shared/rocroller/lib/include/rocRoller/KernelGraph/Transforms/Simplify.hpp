// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Simplify a graph by removing redundant edges.
         */
        class Simplify : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override;
        };

        void removeRedundantSequenceEdges(KernelGraph& graph);
        void removeRedundantBodyEdges(KernelGraph& graph);
        void removeRedundantNOPs(KernelGraph& graph);

        void removeRedundantSequenceEdges(ControlGraph::ControlGraph& graph);
        void removeRedundantBodyEdges(ControlGraph::ControlGraph& graph);
    }
}
