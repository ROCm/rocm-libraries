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
// `prefix` runs through the trailing "_Float_". The caller builds it.
inline bool rocRollerKernelNameMatchesTile(const std::string&           kernel,
                                           const std::string&           prefix,
                                           const RocRollerSelectedTile& tile)
{
    if(kernel.find(prefix) == std::string::npos)
        return false;

    // Captured kernels use natural scales. Preswizzled support would also need
    // to match these tokens against the problem's scale format.
    if(kernel.find("_PSTA_") != std::string::npos || kernel.find("_PTA_") != std::string::npos
       || kernel.find("_PSTB_") != std::string::npos || kernel.find("_PTB_") != std::string::npos)
        return false;

    // Include delimiters so a tile extent cannot match part of a larger number.
    const std::string wgt = "_WGT_" + std::to_string(tile.m) + "x" + std::to_string(tile.n) + "x"
                            + std::to_string(tile.k) + "_UR_";
    if(kernel.find(wgt) == std::string::npos)
        return false;

    // genKernelName places SK immediately before the workgroup tile token.
    const bool hasSk = kernel.find("_SK" + wgt) != std::string::npos;
    if(hasSk != tile.streamK)
        return false;

    // StreamK disables workgroup mapping even when the ranked tile requests it.
    const bool hasWgm = kernel.find("_WGM_") != std::string::npos;
    return hasWgm == (tile.workgroupMapping && !tile.streamK);
}
