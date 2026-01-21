// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/WorkgroupRemapXCC.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace WorkgroupRemapXCCDetail
        {

            /**
             * @brief Remap Workgroup to be more cache friendly
             * (consecutive workgroups land within the same XCC).
             *
             * Modifies the coordinate graph.
             *
             * Returns the newly added Workgroup dimension.
             */
            int remapWorkgroupXCC(rocRoller::KernelGraph::KernelGraph& graph,
                                  int                                  workgroupTag,
                                  uint                                 numXCC);
        }
    }
}
