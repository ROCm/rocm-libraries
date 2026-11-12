// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// The rocRoller-off walk over rocRoller's ranked tiles, and the checks that
// decide whether a checked-in kernel may serve a problem. Templated on the
// solution, problem and hardware types so the tests can drive them GPU-free.

#include "rocroller_kernel_match.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

// The kernel's own hardware and problem predicates accept the problem. These
// carry the operand types, transposes, MX scale block and scale layout.
template <typename Solution, typename Problem, typename Hardware>
bool rocRollerKernelAccepts(const Solution& solution,
                            const Problem&  problem,
                            const Hardware& hardware)
{
    if(solution.hardwarePredicate && !(*solution.hardwarePredicate)(hardware))
        return false;
    return !solution.problemPredicate || (*solution.problemPredicate)(problem);
}

template <typename Solution, typename Problem, typename Hardware>
bool rocRollerKernelFitsWorkspace(const Solution& solution,
                                  const Problem&  problem,
                                  const Hardware& hardware,
                                  size_t          maxWorkspaceBytes)
{
    return solution.requiredWorkspaceSize(problem, hardware) <= maxWorkspaceBytes;
}

// rocRoller's Stream-K loop cannot run data-parallel tiles, which its grid
// leaves once there are two or more tiles per workgroup. The launch throws at
// exactly these points (ContractionSolution::rocRollerStreamKHasDataParallelTiles).
template <typename Solution, typename Problem, typename Hardware>
bool rocRollerKernelLaunchHasDataParallelTiles(const Solution& solution,
                                               const Problem&  problem,
                                               const Hardware& hardware)
{
    return solution.rocRollerStreamKHasDataParallelTiles(problem, hardware);
}

enum class RocRollerWalkStop
{
    // Every ranked tile was visited.
    Exhausted,
    // The caller's algorithm count was reached.
    CountReached,
    // No checked-in kernel accepting the problem is the one rocRoller would
    // build for this tile.
    NoKernel,
    // More than one is, so which one rocRoller would run is unknown.
    Ambiguous,
    // The kernel exists but must not launch on this problem.
    KernelRefusesProblem,
};

struct RocRollerWalkResult
{
    // Indices into the kernel list, in rank order, each at most once.
    std::vector<size_t> picks;
    RocRollerWalkStop   stop = RocRollerWalkStop::Exhausted;
    // The ranked tile the walk stopped at. ranked.size() when exhausted.
    size_t stopTile = 0;
};

// Visits `ranked` in order, as getRocRollerBestSolutions does, and keeps the
// checked-in kernel for each tile rocRoller would build and run. `kernels` are
// the checked-in rocRoller kernels that accept the problem.
//
// A tile is passed over where rocRoller passes over it: the tile does not
// divide the problem, or the kernel needs more workspace than `fits` allows.
// A tile whose kernel was already kept is passed over too; it differs from
// the earlier one only in non-temporal hints. Any other tile without exactly
// one kernel ends the walk, so the result never holds a kernel rocRoller
// would not have picked at that rank.
//
// Where the two differ the walk is the more conservative: a tile rocRoller
// passes over because it cannot build the kernel, or because its own
// predicates reject the problem, ends the walk here. rocRoller also skips a
// tile that does not divide the problem only on a solution-cache miss, so a
// rocRoller process that already built that tile can still run it.
//
// refuses(i) is true when kernel i must not launch on the problem.
template <typename Fits, typename Refuses>
RocRollerWalkResult walkRocRollerTiles(const std::vector<RocRollerSelectedTile>& ranked,
                                       size_t                                    m,
                                       size_t                                    n,
                                       size_t                                    k,
                                       const std::vector<RocRollerKernelFields>& kernels,
                                       int                                       requestedAlgoCount,
                                       Fits                                      fits,
                                       Refuses                                   refuses)
{
    RocRollerWalkResult result;
    auto                stopAt = [&](RocRollerWalkStop why, size_t tile) {
        result.stop     = why;
        result.stopTile = tile;
        return result;
    };

    for(size_t t = 0; t < ranked.size(); ++t)
    {
        if(requestedAlgoCount >= 0
           && result.picks.size() >= static_cast<size_t>(requestedAlgoCount))
            return stopAt(RocRollerWalkStop::CountReached, t);

        const RocRollerSelectedTile& tile = ranked[t];
        if(tile.m > 0 && tile.n > 0 && tile.k > 0
           && (m % tile.m != 0 || n % tile.n != 0 || k % tile.k != 0))
            continue;

        size_t match   = kernels.size();
        int    matches = 0;
        for(size_t i = 0; i < kernels.size() && matches < 2; ++i)
        {
            if(rocRollerKernelMatchesTile(kernels[i], tile))
            {
                match = i;
                ++matches;
            }
        }
        if(matches == 0)
            return stopAt(RocRollerWalkStop::NoKernel, t);
        if(matches > 1)
            return stopAt(RocRollerWalkStop::Ambiguous, t);
        if(!fits(match))
            continue;
        if(refuses(match))
            return stopAt(RocRollerWalkStop::KernelRefusesProblem, t);
        if(std::find(result.picks.begin(), result.picks.end(), match) != result.picks.end())
            continue;
        result.picks.push_back(match);
    }
    return stopAt(RocRollerWalkStop::Exhausted, ranked.size());
}
