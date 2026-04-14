// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Swizzle the scale loads.
         */
        class SwizzleScale : public GraphTransform
        {
        public:
            SwizzleScale(CommandParametersPtr params, ContextPtr context)
                : m_params(params)
                , m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "SwizzleScale";
            }

        private:
            // TODO: remove m_params when enable prefetchLDSFactor > 0 for LDSScaleA/B
            CommandParametersPtr m_params;
            ContextPtr           m_context;
        };
    }
}
