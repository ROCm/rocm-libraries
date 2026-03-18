// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief  Remap Workgroup to be more cache friendly
         * (consecutive workgroups land within the same XCC).
         *
         */
        class WorkgroupRemapXCC : public GraphTransform
        {

        public:
            WorkgroupRemapXCC(ContextPtr context, std::optional<int> workgroupRemapXCC)
                : m_context(context)
                , m_workgroupRemapXCC(workgroupRemapXCC)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "WorkgroupRemapXCC";
            }

        private:
            ContextPtr         m_context;
            std::optional<int> m_workgroupRemapXCC;
        };
    }
}
