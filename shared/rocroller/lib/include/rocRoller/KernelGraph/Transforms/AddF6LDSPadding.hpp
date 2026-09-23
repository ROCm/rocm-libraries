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
         * @brief Update KernelGraph to add LDS padding if F6 datatypes will
         * be transpose loaded.
         *
         * When transpose loading F6 datatypes from LDS addresses must be
         * 128b aligned. This transform modifies KernelGraph in
         * such that MacroTile dimensions of F6 datatypes get padding in
         * fast-moving dimension. Padding information is kept in new
         * MacroTile/WaveTile field -- padBytesOfDim keeps track of byte
         * padding per dimension. Total padding bytes can be queried by
         * calling paddingBytes. Such padding causes extra LDS to be allocated.
         */
        class AddF6LDSPadding : public GraphTransform
        {
        public:
            AddF6LDSPadding(ContextPtr context)
                : m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddF6LDSPadding";
            }

        private:
            ContextPtr m_context;
        };
    }
}
