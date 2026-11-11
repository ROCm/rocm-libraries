// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// One entry from chooseSolutionIndexParameters, in that function's order.
// unroll is the prefetchInFlight parameter_selection.cpp generates the tile
// with. The non-temporal bits are kept for diagnostics; they only set cache
// bits on the loads, so one checked-in kernel serves either value.
struct RocRollerSelectedTile
{
    int  m                = 0;
    int  n                = 0;
    int  k                = 0;
    bool workgroupMapping = false;
    bool streamK          = false;
    bool nonTemporalA     = false;
    bool nonTemporalB     = false;
    int  unroll           = 0;
    // rocRoller fails to build this configuration, so it never runs it.
    bool unbuildable = false;
};

// What a checked-in rocRoller kernel declares in its custom.config: the
// macrotile, the unroll, whether it takes a workgroup-mapping kernarg, and
// whether it is a Stream-K kernel. Operand types, transposes, MX scale block
// and scale layout are the kernel's predicates, which the caller evaluates.
struct RocRollerKernelFields
{
    int  tileM            = 0;
    int  tileN            = 0;
    int  tileK            = 0;
    int  unroll           = 0;
    bool workgroupMapping = false;
    bool streamK          = false;
};

// Is this kernel the one rocRoller generates for this ranked tile? A kernel
// that declares no unroll is never a match.
inline bool rocRollerKernelMatchesTile(const RocRollerKernelFields& kernel,
                                       const RocRollerSelectedTile& tile)
{
    if(kernel.unroll <= 0 || kernel.unroll != tile.unroll)
        return false;
    if(kernel.tileM != tile.m || kernel.tileN != tile.n || kernel.tileK != tile.k)
        return false;
    if(kernel.streamK != tile.streamK)
        return false;
    // StreamK disables workgroup mapping even when the ranked tile requests it.
    return kernel.workgroupMapping == (tile.workgroupMapping && !tile.streamK);
}
