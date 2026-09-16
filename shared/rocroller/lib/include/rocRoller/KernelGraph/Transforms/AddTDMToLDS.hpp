// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Merge the LoadTiled and StoreLDSTile operation to enable TDM to LDS
         *
         * @ingroup Transformations
         */
        class AddTDMToLDS : public GraphTransform
        {
        public:
            AddTDMToLDS(ContextPtr context)
                : m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;

            std::string name() const override
            {
                return "AddTDMToLDS";
            }

            std::vector<GraphConstraint> postConstraints() const override;

        private:
            ContextPtr m_context;
        };
    }
}
