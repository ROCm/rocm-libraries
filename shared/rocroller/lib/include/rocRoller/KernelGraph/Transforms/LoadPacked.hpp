// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        class LoadPacked : public GraphTransform
        {
        public:
            LoadPacked(ContextPtr context);

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "LoadPacked";
            }

            inline std::vector<GraphConstraint> preConstraints() const override
            {
                return {};
            }

        private:
            ContextPtr m_context;
        };
    }
}
