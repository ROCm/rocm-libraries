// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// One entry from chooseSolutionIndexParameters, in that function's order.
// streamK and the non-temporal bits are kept so a checked-in kernel is not
// substituted for a rocRoller kernel that was not captured.
struct RocRollerSelectedTile
{
    int  m                = 0;
    int  n                = 0;
    int  k                = 0;
    bool workgroupMapping = false;
    bool streamK          = false;
    bool nonTemporalA     = false;
    bool nonTemporalB     = false;
};

// What a checked-in rocRoller kernel's solution declares about it: the macro
// tile (customKernel.macrotile), StreamK (a RocRollerStreamK workspace) and
// workgroup mapping (an RR_WorkgroupMapping kernarg). Operand types,
// transposes, scale-block size and scale layout are the solution's predicates,
// which the caller evaluates. The unroll follows from the types and the tile,
// so it needs no match of its own.
struct RocRollerKernelTraits
{
    int  m                = 0;
    int  n                = 0;
    int  k                = 0;
    bool streamK          = false;
    bool workgroupMapping = false;
};

// Does a checked-in kernel with these traits implement exactly this ranked tile?
inline bool rocRollerKernelMatchesTile(const RocRollerKernelTraits& kernel,
                                       const RocRollerSelectedTile& tile)
{
    // No non-temporal variant is checked in.
    if(tile.nonTemporalA || tile.nonTemporalB)
        return false;
    if(kernel.m != tile.m || kernel.n != tile.n || kernel.k != tile.k)
        return false;
    if(kernel.streamK != tile.streamK)
        return false;
    // StreamK disables workgroup mapping even when the ranked tile requests it.
    return kernel.workgroupMapping == (tile.workgroupMapping && !tile.streamK);
}

// Where an accepting checked-in kernel stands when no ranked tile has one.
// Compared in this order, the first difference deciding:
//   1. the earliest position of the kernel's macro tile in the ranked list,
//      which is origami's predicted order; tiles the ranker does not list last;
//   2. the larger macro-tile area M x N;
//   3. fewer differences from the WGM and StreamK the ranker gives that tile;
//      for an unlisted tile, WGM exactly when K >= 4096 and no StreamK;
//   4. the lower solution index.
struct RocRollerFallbackRank
{
    size_t  tilePosition  = 0;
    int64_t area          = 0;
    int     mismatches    = 0;
    int     solutionIndex = 0;

    bool operator<(const RocRollerFallbackRank& other) const
    {
        if(tilePosition != other.tilePosition)
            return tilePosition < other.tilePosition;
        if(area != other.area)
            return area > other.area;
        if(mismatches != other.mismatches)
            return mismatches < other.mismatches;
        return solutionIndex < other.solutionIndex;
    }
};

inline RocRollerFallbackRank rocRollerFallbackRank(const RocRollerKernelTraits&              kernel,
                                                   const std::vector<RocRollerSelectedTile>& ranked,
                                                   int64_t problemK,
                                                   int     solutionIndex)
{
    RocRollerFallbackRank rank;
    rank.tilePosition = ranked.size();
    for(size_t i = 0; i < ranked.size(); ++i)
    {
        if(ranked[i].m == kernel.m && ranked[i].n == kernel.n && ranked[i].k == kernel.k)
        {
            rank.tilePosition = i;
            break;
        }
    }

    bool wantStreamK = false;
    bool wantWgm     = problemK >= 4096;
    if(rank.tilePosition < ranked.size())
    {
        wantStreamK = ranked[rank.tilePosition].streamK;
        wantWgm     = ranked[rank.tilePosition].workgroupMapping && !wantStreamK;
    }
    rank.area          = static_cast<int64_t>(kernel.m) * kernel.n;
    rank.mismatches    = (kernel.workgroupMapping != wantWgm) + (kernel.streamK != wantStreamK);
    rank.solutionIndex = solutionIndex;
    return rank;
}
