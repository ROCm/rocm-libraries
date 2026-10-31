// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

namespace TensileLite
{
    //! The GSU and StaggerU fields of the internalArgs word a launch of a solution passes for a problem, and the
    //! arguments that decide how a persistent launch splits the unroll iterations of each tile
    //! (ContractionSolution::launchSummationOrder). It includes nothing of Tensile, so the client's
    //! hardware-free f16-accumulation test compiles it.
    struct LaunchSummationOrder
    {
        enum class Persistent : uint32_t
        {
            None,
            DataParallel,
            StreamK, //!< two-tile StreamK with the fixup in the kernel
            StreamKParallel, //!< two-tile StreamK reduced by the reduction kernel
            StreamKDynamic //!< the work-queue path: DynamicWorkQueue, and the dynamic sub-mode of Hybrid
        };

        uint32_t gsu                 = 1;
        bool     gsuCoalesced        = false;
        uint32_t staggerU            = 0;
        uint32_t staggerUStrideShift = 0;
        uint32_t staggerUMapping     = 0;

        Persistent persistent        = Persistent::None;
        bool       fixupTree         = false; //!< StreamKFixupTreeReduction
        bool       perTileExtraIters = false; //!< the uniform-summation-order bit the kernel acts on
        size_t     tiles             = 0;
        size_t     itersPerTile      = 1;
        size_t     grid              = 0;
        size_t     skTiles           = 0; //!< StreamK: split tiles; dynamic: SKTiles
        size_t     skItersPerWG      = 0;
        size_t     extraIters        = 0;
        size_t     skSplit           = 0; //!< parallel: workgroups per tile; dynamic: SKSplit
        size_t     skItersPerWI      = 0;
        size_t     tiles0            = 0; //!< workgroups along M and N, which tileIndexToWorkGroup walks
        size_t     tiles1            = 0;
        int32_t    wgm               = 1; //!< the WGM field DefaultWGM reads
        bool       spaceFilling      = false; //!< the SpaceFillingAlgo walk replaces DefaultWGM
        uint32_t   spaceFillingAlgo  = 0; //!< SizeMapping::spaceFillingAlgo
        uint32_t   wgmWord           = 0; //!< the WGM word the SpaceFillingAlgo walk reads
    };
} // namespace TensileLite
