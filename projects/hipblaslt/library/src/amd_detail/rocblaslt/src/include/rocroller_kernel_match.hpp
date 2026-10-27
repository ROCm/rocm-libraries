// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

// One entry from chooseSolutionIndexParameters, in that function's order.
// streamK and the non-temporal bits are kept so a checked-in kernel is not
// substituted for a rocRoller kernel that was not captured.
//
// unroll is prefetchInFlight, the integer after _UR_. A scale-block extent of
// 0 means that operand has no block-scale size in the name. The A token is
// blockRow[x blockCol]; the B token is blockCol[x blockRow], matching
// gemm.cpp. The second extent is omitted from the name when it is 1.
struct RocRollerSelectedTile
{
    int  m = 0;
    int  n = 0;
    int  k = 0;
    bool workgroupMapping = false;
    bool streamK          = false;
    bool nonTemporalA     = false;
    bool nonTemporalB     = false;
    int  unroll           = 0;
    int  scaleABlockRow   = 0;
    int  scaleABlockCol   = 1;
    int  scaleBBlockCol   = 0;
    int  scaleBBlockRow   = 1;
};

// `token` occurs in `kernel` and the character after it is '_' or the end.
// Stops "_UR_2" from matching "_UR_20", and "_WGT_16x16x128" from matching a
// longer K extent.
inline bool rocRollerDelimitedToken(const std::string& kernel, const std::string& token)
{
    const auto pos = kernel.find(token);
    if(pos == std::string::npos)
        return false;
    const auto end = pos + token.size();
    return end == kernel.size() || kernel[end] == '_';
}

// The integer gemm.cpp writes after SA_<mode><type>_ or SB_<mode><type>_.
// primary is blockRowSize on A and blockColSize on B. secondary stays 1 when
// the name omits the x-extent. present is false when that operand has no
// block-scale size token. Returns false when the token is malformed.
inline bool rocRollerScaleBlockToken(const std::string& kernel,
                                    const char*        tag,
                                    int&               primary,
                                    int&               secondary,
                                    bool&              present)
{
    present   = false;
    primary   = 0;
    secondary = 1;
    const auto tagPos = kernel.find(tag);
    if(tagPos == std::string::npos)
        return true;

    const auto tagLen  = std::char_traits<char>::length(tag);
    const auto typeEnd = kernel.find('_', tagPos + tagLen);
    if(typeEnd == std::string::npos || typeEnd + 1 >= kernel.size() || kernel[typeEnd + 1] < '0'
       || kernel[typeEnd + 1] > '9')
        return true;

    present = true;
    auto i  = typeEnd + 1;
    while(i < kernel.size() && kernel[i] >= '0' && kernel[i] <= '9')
    {
        primary = primary * 10 + (kernel[i] - '0');
        ++i;
    }
    if(i < kernel.size() && kernel[i] == 'x')
    {
        ++i;
        secondary = 0;
        if(i >= kernel.size() || kernel[i] < '0' || kernel[i] > '9')
            return false;
        while(i < kernel.size() && kernel[i] >= '0' && kernel[i] <= '9')
        {
            secondary = secondary * 10 + (kernel[i] - '0');
            ++i;
        }
    }
    return i == kernel.size() || kernel[i] == '_';
}

inline bool rocRollerScaleBlockMatches(const std::string& kernel,
                                       const char*        tag,
                                       int                expectPrimary,
                                       int                expectSecondary)
{
    int  primary   = 0;
    int  secondary = 1;
    bool present   = false;
    if(!rocRollerScaleBlockToken(kernel, tag, primary, secondary, present))
        return false;
    if(expectPrimary == 0)
        return !present;
    return present && primary == expectPrimary && secondary == expectSecondary;
}

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

    // The prefix match stops at _Float_ and the tile match starts at WGT_. The
    // scale datatype between them is not compared; the block-size integers are,
    // below. A pre-swizzled kernel carries PSTA_/PTA_/PSTB_/PTB_ there and
    // would otherwise match a non-pre-swizzled problem on tile shape alone.
    // None are checked in, so reject them; if pre-swizzled rocRoller kernels
    // are ever added this has to become a positive match on the problem's own
    // scale format rather than a blanket reject. useRocRoller() is not the
    // guard here: it turns a pre-swizzled scale layout away only for FP4 x FP4
    // and E4M3 x E4M3, so every other pre-swizzled pair still reaches this
    // function.
    if(kernel.find("_PSTA_") != std::string::npos || kernel.find("_PTA_") != std::string::npos
       || kernel.find("_PSTB_") != std::string::npos || kernel.find("_PTB_") != std::string::npos)
        return false;

    // The leading '_' anchors the tile token: gemm.cpp always emits '_' before
    // WGT_, from either _Float_ or the last scale token. The unroll is part of
    // the token. "_UR_" alone matches every prefetch depth.
    const std::string wgt = "_WGT_" + std::to_string(tile.m) + "x" + std::to_string(tile.n) + "x"
                            + std::to_string(tile.k) + "_UR_" + std::to_string(tile.unroll);
    if(!rocRollerDelimitedToken(kernel, wgt))
        return false;

    // gemm.cpp emits "SK_" immediately before "WGT_". Without this check
    // "_WGT_16x16x128_UR_2" is a substring of both the plain and the StreamK
    // kernel of the same tile, and either would match either tile.
    const bool hasSk = kernel.find("_SK" + wgt) != std::string::npos;
    if(hasSk != tile.streamK)
        return false;

    // Block-scale size. Two kernels of one tile can differ here once more than
    // Block_32 is checked in, and the first hit in library->solutions would
    // otherwise win.
    if(!rocRollerScaleBlockMatches(kernel, "SA_", tile.scaleABlockRow, tile.scaleABlockCol)
       || !rocRollerScaleBlockMatches(kernel, "SB_", tile.scaleBBlockCol, tile.scaleBBlockRow))
        return false;

    // parameter_selection.cpp forces workgroupMappingDim = -1 for every
    // StreamK kernel, so an SK name never carries _WGM_ even when the ranked tile
    // has workgroupMapping == true.
    const bool hasWgm = kernel.find("_WGM_") != std::string::npos;
    return hasWgm == (tile.workgroupMapping && !tile.streamK);
}
