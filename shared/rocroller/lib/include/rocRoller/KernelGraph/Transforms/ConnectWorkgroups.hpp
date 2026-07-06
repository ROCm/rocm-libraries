// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Connect unbound (leaf) MacroTileNumber coordinates
         * to Workgroups.
         *
         * This transform searches for MacroTileNumber coordinates
         * that are leafs (don't have outgoing/incoming edges), and
         * attaches Workgroup coordinates to them.
         */
        class ConnectWorkgroups : public GraphTransform
        {

        public:
            ConnectWorkgroups(ContextPtr context)
                : m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "ConnectWorkgroups";
            }

        private:
            ContextPtr m_context;
        };
    }
}
