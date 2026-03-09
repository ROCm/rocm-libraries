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
         * @brief Rewrite KernelGraph to add LDS operations for
         * loading/storing data.
         *
         * Modifies the coordinate and control graphs to add LDS
         * information.
         *
         * @ingroup Transformations
         */
        class AddLDS : public GraphTransform
        {
        public:
            AddLDS(CommandParametersPtr params, ContextPtr context)
                : m_params(params)
                , m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddLDS";
            }

            std::vector<GraphConstraint> postConstraints() const override;

        private:
            CommandParametersPtr m_params;
            ContextPtr           m_context;
        };
    }
}
