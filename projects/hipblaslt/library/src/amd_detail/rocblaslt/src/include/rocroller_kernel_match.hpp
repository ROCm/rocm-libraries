// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

// One entry from chooseSolutionIndexParameters, in that function's order.
// streamK and the non-temporal bits are kept so a checked-in kernel is not
// substituted for a rocRoller kernel that was not captured.
struct RocRollerSelectedTile
{
    int  m = 0;
    int  n = 0;
    int  k = 0;
    bool workgroupMapping = false;
    bool streamK          = false;
    bool nonTemporalA     = false;
    bool nonTemporalB     = false;
};

// Does a checked-in rocRoller kernel name describe exactly this ranked tile?
//
// `prefix` is the type/transpose prefix the caller already built, up to and
// including the trailing "_Float_"; everything this function inspects lives
// after it. Split out of matchesRocRollerSelectedKernel (tensile_host.cpp) so
// it can be unit-tested without a RocblasltContractionProblem or a GPU; that
// function builds the prefix and calls straight through to here.
inline bool rocRollerKernelNameMatchesTile(const std::string&           kernel,
                                           const std::string&           prefix,
                                           const RocRollerSelectedTile& tile)
{
    if(kernel.find(prefix) == std::string::npos)
        return false;

    // The prefix match stops at _Float_ and the tile match starts at WGT_, so the
    // scale tokens in between are never inspected. A pre-swizzled kernel carries
    // PSTA_/PTA_/PSTB_/PTB_ there and would otherwise match a non-pre-swizzled
    // problem on tile shape alone. None are checked in and useRocRoller() turns
    // the pre-swizzled pairs away, so reject them; if pre-swizzled rocRoller
    // kernels are ever added this has to become a positive match on the problem's
    // own scale format rather than a blanket reject.
    if(kernel.find("_PSTA_") != std::string::npos || kernel.find("_PTA_") != std::string::npos
       || kernel.find("_PSTB_") != std::string::npos || kernel.find("_PTB_") != std::string::npos)
        return false;

    // The leading '_' anchors the tile token: gemm.cpp always emits '_' before
    // WGT_, from either _Float_ or the last scale token.
    const std::string wgt = "_WGT_" + std::to_string(tile.m) + "x" + std::to_string(tile.n) + "x"
                            + std::to_string(tile.k) + "_UR_";
    if(kernel.find(wgt) == std::string::npos)
        return false;

    // gemm.cpp:142-147 emits "SK_" immediately before "WGT_". Without this check
    // "_WGT_16x16x128_UR_" is a substring of both the plain and the StreamK kernel
    // of the same tile, and either would match either tile.
    const bool hasSk = kernel.find("_SK" + wgt) != std::string::npos;
    if(hasSk != tile.streamK)
        return false;

    // parameter_selection.cpp:249-254 forces workgroupMappingDim = -1 for every
    // StreamK kernel, so an SK name never carries _WGM_ even when the ranked tile
    // has workgroupMapping == true.
    const bool hasWgm = kernel.find("_WGM_") != std::string::npos;
    return hasWgm == (tile.workgroupMapping && !tile.streamK);
}
