// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "RocRollerWgmKernargs.hpp"

namespace TensileLite
{
    // Hoisted kernargs for a rocRoller Stream-K custom kernel.
    //
    // rocRoller moves its whole work decomposition into kernargs: the host
    // evaluates the .expression trees serialised in the kernel's
    // .amdgpu_metadata and hands the kernel plain uint32 values. This mirrors
    // those trees. All arithmetic is uint32 and wraps, exactly as rocRoller's
    // launch-time evaluator does; only the two tile-count ceil-divides are
    // done in int64 before being truncated.
    //
    // MT_N is not always a power of two (192 occurs in the FP4 kernels), so
    // every division here is a real division, not a shift.
    struct RocRollerStreamKKernargs
    {
        uint32_t numTilesN;
        uint32_t numWGs;
        uint32_t numSKTiles;
        uint32_t numSKTilesPerWG;
        uint32_t numDPTiles;
        uint32_t numDPTilesPerWG;
        uint32_t magicMultipleIters;
        uint32_t magicShiftsIters;
        uint32_t magicMultipleNumTilesN;
        uint32_t magicShiftsNumTilesN;
    };

    // Whether a rocRoller Stream-K launch of numWGs workgroups runs data-parallel
    // tiles: true iff iters > 0 and tiles / numWGs >= 2, the regime in which
    // evaluateRocRollerStreamKKernargs gives numDPTilesPerWG != 0. With
    // numWGs <= tiles < 2 * numWGs the data-parallel loop body never runs.
    //
    // rocRoller's data-parallel tile loop steps by the unrolled K-loop counter,
    // so such a launch hangs or computes wrong results; every launch and
    // selection path refuses the regime through this one predicate. The tile
    // count is computed in uint64 and saturates, so a count the uint32 kernargs
    // would wrap is still recognised.
    inline bool
        rocRollerStreamKHasDataParallelTiles(uint64_t tiles, uint64_t iters, uint64_t numWGs)
    {
        if(numWGs == 0u || iters == 0u)
            return false;
        return tiles / numWGs >= 2u;
    }

    inline bool rocRollerStreamKHasDataParallelTiles(int64_t  m,
                                                     int64_t  n,
                                                     int64_t  k,
                                                     uint32_t tileM,
                                                     uint32_t tileN,
                                                     uint32_t tileK,
                                                     uint32_t numWGs)
    {
        if(m <= 0 || n <= 0 || k <= 0 || tileM == 0u || tileN == 0u || tileK == 0u)
            return false;
        uint64_t ntM   = (static_cast<uint64_t>(m) + tileM - 1u) / tileM;
        uint64_t ntN   = (static_cast<uint64_t>(n) + tileN - 1u) / tileN;
        uint64_t tiles = ntM > std::numeric_limits<uint64_t>::max() / ntN
                             ? std::numeric_limits<uint64_t>::max()
                             : ntM * ntN;
        // Truncating, as in evaluateRocRollerStreamKKernargs.
        uint64_t iters = static_cast<uint64_t>(k) / tileK;
        return rocRollerStreamKHasDataParallelTiles(tiles, iters, numWGs);
    }

    inline RocRollerStreamKKernargs evaluateRocRollerStreamKKernargs(int64_t  m,
                                                                     int64_t  n,
                                                                     int64_t  k,
                                                                     uint32_t tileM,
                                                                     uint32_t tileN,
                                                                     uint32_t tileK,
                                                                     uint32_t numWGs)
    {
        if(tileM == 0u || tileN == 0u || tileK == 0u)
            throw std::runtime_error("rocRoller Stream-K macrotile must be non-zero in M, N and K");
        if(numWGs == 0u)
            throw std::runtime_error("rocRoller Stream-K needs a non-zero workgroup count");

        // Tile counts: ceil-divide in int64, then truncate to uint32.
        uint32_t ntM = static_cast<uint32_t>((m + static_cast<int64_t>(tileM) - 1)
                                             / static_cast<int64_t>(tileM));
        uint32_t ntN = static_cast<uint32_t>((n + static_cast<int64_t>(tileN) - 1)
                                             / static_cast<int64_t>(tileN));

        uint32_t tiles = ntM * ntN;
        // Truncating, not ceil: a K remainder under one macrotile is not an
        // extra accumulation iteration in rocRoller's decomposition.
        uint32_t iters = static_cast<uint32_t>(k / static_cast<int64_t>(tileK));

        uint32_t numSKTiles      = 0;
        uint32_t numDPTiles      = 0;
        uint32_t numDPTilesPerWG = 0;
        if(tiles > numWGs - 1u)
        {
            numSKTiles           = ((tiles % numWGs) + numWGs) * iters;
            uint32_t fullRounds  = (tiles / numWGs) - 1u;
            numDPTiles           = (fullRounds * numWGs) * iters;
            numDPTilesPerWG      = fullRounds * iters;
        }
        else
        {
            numSKTiles = tiles * iters;
        }
        uint32_t numSKTilesPerWG = (numSKTiles + (numWGs - 1u)) / numWGs;

        RocRollerStreamKKernargs out{};
        out.numTilesN              = ntN;
        out.numWGs                 = numWGs;
        out.numSKTiles             = numSKTiles;
        out.numSKTilesPerWG        = numSKTilesPerWG;
        out.numDPTiles             = numDPTiles;
        out.numDPTilesPerWG        = numDPTilesPerWG;
        out.magicMultipleIters     = rocRollerMagicMultipleU32(iters);
        out.magicShiftsIters       = rocRollerMagicShiftsU32(iters);
        out.magicMultipleNumTilesN = rocRollerMagicMultipleU32(ntN);
        out.magicShiftsNumTilesN   = rocRollerMagicShiftsU32(ntN);
        return out;
    }
}
