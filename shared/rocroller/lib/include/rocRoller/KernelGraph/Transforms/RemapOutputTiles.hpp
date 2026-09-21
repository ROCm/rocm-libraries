// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Perform workgroup mapping by applying
         * PiecewiseAffineJoin transformation on dangling
         * MacroTileNumbers
         *
         */
        class RemapOutputTiles : public GraphTransform
        {
        public:
            RemapOutputTiles(std::optional<int>        workgroupMappingDim,
                             Expression::ExpressionPtr workgroupMappingValue = nullptr);

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "RemapOutputTiles";
            }

        private:
            std::optional<int>        m_workgroupMappingDim;
            Expression::ExpressionPtr m_workgroupMappingValue;
        };
    }
}
