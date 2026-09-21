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
         * @brief Rewrite KernelGraph to add padding to LDS buffers.
         * @ingroup Transformations
         */
        class AddLDSPadding : public GraphTransform
        {
        public:
            AddLDSPadding(ContextPtr context, CommandParametersPtr params)
                : m_context(context)
                , m_params(params)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddLDSPadding";
            }

        private:
            ContextPtr           m_context;
            CommandParametersPtr m_params;
        };
    }
}
