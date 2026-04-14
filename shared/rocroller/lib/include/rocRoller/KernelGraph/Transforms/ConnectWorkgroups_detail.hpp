// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/ConnectWorkgroups.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace ConnectWorkgroupsDetail
        {
            /**
             * @brief Connect dangling MacroTileNumber coordinate to
             * matching Workgroup coordinates.
             *
             */
            std::map<std::pair<int, rocRoller::Graph::Direction>, int>
                connectWorkgroups(KernelGraph& kgraph);
        }
    }
}
