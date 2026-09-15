// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaFwdParams.hpp"

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

// Ports AITER's forward grid math (get_grid_dim, mha_fwd.cu). Batch mode sets
// gridDimX = ceil(Q-tiles / tgDiv) (causal masks set tgDiv=2, halving the grid),
// gridDimY = nhead, gridDimZ = batch. Two flat overrides follow AITER's order:
// hd192x128/gfx942 lays out gridDimX = nhead / gridDimY = full Q-tiles (no tgDiv)
// with 256-wide blocks and tuneOpt=0; group (ragged) mode then remaps to
// gridDimX = nhead / gridDimY = batch / gridDimZ = ceil(Q-tiles / tgDiv). hipDNN's
// group tensors are padded [B, S, H, D] with batch carried by gridDimY, so Q-tiles
// come from the per-batch seqLen S (not a packed total_q); per-batch ragged offsets
// bound the live tokens within S.
inline SdpaFwdLaunchParams computeFwdLaunchParams(const SdpaFwdParams& params)
{
    SdpaFwdLaunchParams lp{};

    if(params.tileSizeQo == 0U)
    {
        return lp; // zero guard — matches bwd KernelTiles::gridDim() pattern
    }

    const bool isGfx942 = params.archString == "gfx942";
    const bool isGroupMode = params.group.has_value();
    const bool isHd192x128 = params.headDimQk == 192 && params.headDimV == 128;
    const bool isHd192x128Gfx942 = isHd192x128 && isGfx942;
    const bool masked = params.maskType != plan_utils::MaskType::NO_MASK;

    // tune_opt: default 5; downgrade to 3 when masked and either nhead is
    // not 8-aligned or seqLen exceeds 16K; override 0 for hd192x128/gfx942.
    uint32_t tuneOpt = 5;
    if(masked && ((params.numHeadsQ % 8 != 0) || (params.seqLenQ > 16384U)))
    {
        tuneOpt = 3;
    }
    if(isHd192x128Gfx942)
    {
        tuneOpt = 0;
    }
    lp.tuneOpt = tuneOpt;

    // tgDiv=2 merges the causal head/tail Q-tiles onto shared workgroups; the
    // group hd192x128/gfx942 kernel keeps them separate (tgDiv=1).
    unsigned int tgDiv = masked ? 2U : 1U;
    if(isGfx942 && isGroupMode && isHd192x128)
    {
        tgDiv = 1U;
    }

    const unsigned int qTiles = (params.seqLenQ + params.tileSizeQo - 1U) / params.tileSizeQo;

    unsigned int gdx = (qTiles + tgDiv - 1U) / tgDiv;
    unsigned int gdy = params.numHeadsQ;
    unsigned int gdz = params.batchSize;

    if(isHd192x128Gfx942)
    {
        gdx = params.numHeadsQ;
        gdy = qTiles; // full tile count, no tgDiv
        gdz = params.batchSize;
    }
    if(isGroupMode)
    {
        gdx = params.numHeadsQ;
        gdy = params.batchSize;
        gdz = (qTiles + tgDiv - 1U) / tgDiv;
    }

    lp.gridDimX = gdx;
    lp.gridDimY = gdy;
    lp.gridDimZ = gdz;
    lp.blockDimX = isHd192x128Gfx942 ? 256U : 512U;

    return lp;
}

} // namespace asm_sdpa_engine
