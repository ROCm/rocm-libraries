// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Remove all SetCoordinates from control graph
         *
         */
        class RemoveSetCoordinate : public GraphTransform
        {
        public:
            RemoveSetCoordinate() {}

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "RemoveSetCoordinate";
            }
        };
    }
}
