// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Ensure there is a barrier between every LDS write and read operation.
         *
         * This transform adds barriers between:
         * - StoreLDSTile or LoadTileDirect2LDS (writes) and LoadLDSTile (reads)
         *
         * @ingroup Transformations
         */
        class AddLDSBarriers : public GraphTransform
        {
        public:
            AddLDSBarriers() = default;

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddLDSBarriers";
            }

            std::vector<GraphConstraint> postConstraints() const override;
        };
    }
}
