// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaFwdParams.hpp"

#include <algorithm>
#include <cstdint>

namespace asm_sdpa_engine
{

struct SdpaFwdLaunchParams
{
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    uint32_t tuneOpt;
};

// Ports AITER's forward grid math. Causal masks halve the Q-tile grid via
// ceiling division; the kernel load-balances the full Q range onto the
// surviving workgroups. The hd192x128/gfx942 path swaps gridDimX/Y, uses
// blockDimX=256, and forces tuneOpt=0.
inline SdpaFwdLaunchParams computeFwdLaunchParams(const SdpaFwdParams& params)
{
    SdpaFwdLaunchParams lp{};

    if(params.tileSizeQo == 0U)
    {
        return lp; // zero guard — matches bwd KernelTiles::gridDim() pattern
    }

    const bool isHd192x128Gfx942
        = params.headDimQk == 192 && params.headDimV == 128 && params.archString == "gfx942";
    const bool masked = params.maskType != plan_utils::MaskType::NO_MASK;

    // tune_opt: default 5; downgrade to 3 when masked and either nhead is
    // not 8-aligned or seqLen exceeds 16K; override 0 for hd192x128/gfx942.
    uint32_t tuneOpt = 5;
    if(masked && ((params.numHeadsQ % 8 != 0) || (params.seqLenQ > 16384)))
    {
        tuneOpt = 3;
    }
    if(isHd192x128Gfx942)
    {
        tuneOpt = 0;
    }
    lp.tuneOpt = tuneOpt;

    // gridDimX = ceil(seqLenQ / tileSizeQo)
    unsigned int gridDimX = (params.seqLenQ + params.tileSizeQo - 1U) / params.tileSizeQo;

    // Halve the causal Q-tile grid; ceiling division keeps a single tile at 1.
    // hd192x128/gfx942 is excluded (its swap path never halves).
    // TODO: port AITER's group-mode gdz remap when group mode lands.
    if(masked && !isHd192x128Gfx942)
    {
        gridDimX = (gridDimX + 1U) / 2U;
    }

    unsigned int gridDimY = params.numHeadsQ;

    // hd192x128/gfx942: swap X/Y and use 256-wide blocks.
    if(isHd192x128Gfx942)
    {
        std::swap(gridDimX, gridDimY);
        lp.blockDimX = 256;
    }
    else
    {
        lp.blockDimX = 512;
    }

    lp.gridDimX = gridDimX;
    lp.gridDimY = gridDimY;
    lp.gridDimZ = params.batchSize;

    return lp;
}

} // namespace asm_sdpa_engine
