// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/AddLDSPadding.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace AddLDSPaddingDetail
        {
            /**
             * @brief Information about LDS padding.
             *
             * Information about LDS padding that is being added to
             * the graph.
             *
             * For direct-to-LDS loads, the loadLaneWidth field is the
             * workgroup size (usually 256 lanes).
             */
            struct LDSPaddingInfo
            {
                int ldsTag; //< LDS coordinate.
                int upstreamEdge; //< Edge immediately upstream (usually a Flatten) of ldsTag.
                int downstreamEdge; //< Edge immediately downstream (usually a Tile) of ldsTag.
                std::array<int, 2> upstreamTags; //< Coordinates upstream of upstreamEdge.
                std::array<int, 2> downstreamTags; //< Coordinates downstream of downstreamEdge.
                DataType           dataType; //< DataType of the data in LDS.
                LayoutType         layoutType; //< LayoutType of the data in LDS.
                uint
                    loadInstructionByteWidth; //< Byte-width of the instructions used to load data destined for LDS.
                uint loadLaneWidth; //< Number of lanes that should be considered contiguous.
            };
        }
    }
}
