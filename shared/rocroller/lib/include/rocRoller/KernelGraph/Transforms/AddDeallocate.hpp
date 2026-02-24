// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Rewrite KernelGraph to add Deallocate operations.
         *
         * The control graph is analyzed to determine register
         * lifetimes.  Deallocate operations are added when registers
         * are no longer needed.
         */
        class AddDeallocateDataFlow : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddDeallocateDataFlow";
            }

        private:
        };

        class AddDeallocateArguments : public GraphTransform
        {
        public:
            AddDeallocateArguments(ContextPtr context)
                : m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddDeallocateArguments";
            }

        private:
            ContextPtr m_context;
        };

        class MergeAdjacentDeallocates : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "MergeAdjacentDeallocates";
            }
        };
    }
}
