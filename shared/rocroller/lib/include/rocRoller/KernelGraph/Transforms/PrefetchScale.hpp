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
         * @brief Prefetch the scale loads.
         */
        class PrefetchScale : public GraphTransform
        {
        public:
            PrefetchScale(CommandParametersPtr params, ContextPtr context)
                : m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "PrefetchScale";
            }

        private:
            ContextPtr m_context;
        };
    }
}
