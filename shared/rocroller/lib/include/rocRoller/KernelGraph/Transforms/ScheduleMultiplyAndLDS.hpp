// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * For certain kernels, each subIter should have three parallel chains of nodes:
         * LoadTileDirect2LDS, Multiply, LoadLDSTile.
         *
         * ScheduleMultiplyAndLDS will create Sequence edges between the Multiply and
         * LoadLDSTile chains in order to allow the AliasDataFlowTags transform to find more
         * aliases.
         *
         */
        class ScheduleMultiplyAndLDS : public GraphTransform
        {
        public:
            ScheduleMultiplyAndLDS() = default;

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "ScheduleMultiplyAndLDS";
            }

            inline std::vector<GraphConstraint> preConstraints() const override
            {
                return {};
            }
        };
    }
}
