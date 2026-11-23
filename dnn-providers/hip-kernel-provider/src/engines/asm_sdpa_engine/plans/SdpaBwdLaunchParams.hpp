// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaBwdParams.hpp"
#include "SdpaPlanUtils.hpp"

#include <cstdint>

namespace asm_sdpa_engine
{

/// The kernels of one SDPA backward plan, in launch order.
enum class SdpaBwdStage : uint8_t
{
    ODO,
    DQDKDV,
    POST
};

struct SdpaBwdLaunchParams
{
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
};

/// True when the DQDKDV kernel for dqdkdv CSV mask `maskOrdinal` runs on half
/// the K/V tiles, rounded up. Mirrors AITER v0.1.29 mha_bwd.cu fmha_v3_bwd()
/// (:713-716): only the causal kernels (masks 1 and 2) halve the grid; the
/// sliding-window kernels (masks 3 and 4) take the full grid.
inline bool halvesDqdkdvGrid(int32_t maskOrdinal)
{
    return maskOrdinal == static_cast<int32_t>(plan_utils::MaskType::TOP_LEFT_CAUSAL)
           || maskOrdinal == static_cast<int32_t>(plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL);
}

// Ports AITER v0.1.29 mha_bwd.cu fmha_v3_bwd() batch-mode launch math for the
// gfx942 and gfx950 kernels the engine serves. Every stage uses 256-wide
// blocks (:500, :700, :754), gridDimY = H_q and gridDimZ = B. gridDimX is one
// workgroup per tile of the stage's 'ts' column: ODO and post tile S_q
// (:501, :755); DQDKDV tiles S_kv (:701), halved for the causal masks
// (halvesDqdkdvGrid()). A stage whose tile size is unset gets gridDimX = 0
// (KernelTiles::gridDim()).
inline SdpaBwdLaunchParams computeBwdLaunchParams(const SdpaBwdParams& params, SdpaBwdStage stage)
{
    constexpr unsigned int K_BWD_BLOCK_DIM = 256;

    unsigned int gdx = 0;
    switch(stage)
    {
    case SdpaBwdStage::ODO:
        gdx = params.odoTiles.gridDim(params.seqLenQ);
        break;
    case SdpaBwdStage::DQDKDV:
        gdx = params.dqdkdvTiles.gridDim(params.seqLenKv, halvesDqdkdvGrid(params.maskOrdinal));
        break;
    case SdpaBwdStage::POST:
        gdx = params.postTiles.gridDim(params.seqLenQ);
        break;
    default:
        break; // gridDimX = 0, which the launch rejects
    }

    return {gdx, params.numHeadsQ, params.batchSize, K_BWD_BLOCK_DIM};
}

} // namespace asm_sdpa_engine
