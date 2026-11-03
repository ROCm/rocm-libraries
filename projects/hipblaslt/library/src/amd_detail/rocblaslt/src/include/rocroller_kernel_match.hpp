// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
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
// tile (customKernel.macrotile) and workgroup mapping (an RR_WorkgroupMapping
// kernarg). No checked-in kernel is StreamK.
// Operand types, transposes, scale-block size and scale layout are the
// solution's predicates, which the caller evaluates. The unroll follows from
// the types and the tile, so it needs no match of its own.
struct RocRollerKernelTraits
{
    int  m                = 0;
    int  n                = 0;
    int  k                = 0;
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
    // No StreamK kernel is checked in.
    if(tile.streamK)
        return false;
    return kernel.workgroupMapping == tile.workgroupMapping;
}

// Output tiles of an M x N problem cut into tileM x tileN, per batch.
inline int64_t rocRollerOutputTiles(int64_t m, int64_t n, int tileM, int tileN)
{
    if(tileM <= 0 || tileN <= 0 || m <= 0 || n <= 0)
        return 0;
    return ((m + tileM - 1) / tileM) * ((n + tileN - 1) / tileN);
}

// Workgroup mapping only pays off on a large grid of wide-M tiles. Measured on
// gfx950 over the checked-in WGM and non-WGM twins: it helps from 512 output
// tiles with MT0 >= 128, costs time on smaller grids, and is erratic for narrow
// tiles. K >= 4096 is the long-standing bound below which it never helps.
inline bool rocRollerWantsWorkgroupMapping(int64_t m, int64_t n, int64_t k, int tileM, int tileN)
{
    return k >= 4096 && tileM >= 128 && rocRollerOutputTiles(m, n, tileM, tileN) >= 512;
}

// The CUs a launch may use: the caller's sm_count_target when it asks for fewer
// than the device has, resolved as the ranker resolves it.
inline int64_t rocRollerUsableCUs(int64_t requested, int64_t device)
{
    return requested > 0 && requested < device ? requested : device;
}

// Share of the CUs that the last wave of a one-workgroup-per-tile grid keeps
// busy: tiles / (ceil(tiles / CUs) * CUs).
inline double rocRollerLastWaveFill(
    const RocRollerKernelTraits& kernel, int64_t m, int64_t n, int64_t batch, int64_t cus)
{
    const int64_t tiles
        = rocRollerOutputTiles(m, n, kernel.m, kernel.n) * std::max<int64_t>(batch, 1);
    if(tiles <= 0 || cus <= 0)
        return 0.0;
    const int64_t waves = (tiles + cus - 1) / cus;
    return static_cast<double>(tiles) / static_cast<double>(waves * cus);
}

// A ranked list is kept in order unless its head leaves most CUs idle in the
// last wave: then the first of the leading few that fills at least half of
// them goes first. The window keeps the ranker's order in charge, which a full
// scan for fill would overrule.
constexpr size_t rocRollerRankedFillWindow = 5;
constexpr double rocRollerRankedMinFill    = 0.5;
// The fallback has no ranker order to defer to beyond tile position and area,
// so it scans every candidate, for a fuller last wave.
constexpr double rocRollerFallbackMinFill = 0.75;

// Position of the first of the leading `window` fills that reaches `minFill`;
// 0, keeping the head, when none does.
inline size_t rocRollerFirstFilling(const std::vector<double>& fills, size_t window, double minFill)
{
    const size_t end = std::min(window, fills.size());
    for(size_t i = 0; i < end; ++i)
    {
        if(fills[i] >= minFill)
            return i;
    }
    return 0;
}

// Where an accepting checked-in kernel stands when no ranked tile has one.
// Compared in this order, the first difference deciding:
//   1. the earliest position of the kernel's macro tile in the ranked list,
//      which is origami's predicted order; tiles the ranker does not list last;
//   2. the larger macro-tile area M x N;
//   3. fewer differences from the WGM and StreamK the ranker gives that tile;
//      for an unlisted tile, no StreamK and WGM exactly when
//      rocRollerWantsWorkgroupMapping, as the ranker would decide it;
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
                                                   int64_t problemM,
                                                   int64_t problemN,
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
    bool wantWgm = rocRollerWantsWorkgroupMapping(problemM, problemN, problemK, kernel.m, kernel.n);
    if(rank.tilePosition < ranked.size())
    {
        wantStreamK = ranked[rank.tilePosition].streamK;
        wantWgm     = ranked[rank.tilePosition].workgroupMapping && !wantStreamK;
    }
    rank.area          = static_cast<int64_t>(kernel.m) * kernel.n;
    rank.mismatches    = (kernel.workgroupMapping != wantWgm) + wantStreamK;
    rank.solutionIndex = solutionIndex;
    return rank;
}
