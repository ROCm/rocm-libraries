// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename InDataType_,
          typename WeiDataType_,
          typename AccDataType_,
          typename OutDataType_,
          index_t BlockSize_,
          index_t TileH_,
          index_t TileW_,
          index_t FilterH_,
          index_t FilterW_,
          index_t StrideH_,
          index_t StrideW_,
          index_t DilationH_,
          index_t DilationW_,
          index_t PadH_,
          index_t PadW_,
          index_t NBatch_,
          index_t SubTileH_,
          index_t SubTileW_,
          index_t InVectorSize_,
          index_t OutVectorSize_>
struct DepthwiseConvFwdTraits
{
    using InDataType  = InDataType_;
    using WeiDataType = WeiDataType_;
    using AccDataType = AccDataType_;
    using OutDataType = OutDataType_;

    static constexpr index_t NDimSpatial = 2;

    static constexpr index_t BlockSize = BlockSize_;
    // TODO: hardcoded wave64; wave32 support requires adjusting TilePerWave/ThreadPerTile
    // derivation
    static constexpr index_t WaveSize = 64;

    static constexpr index_t TileOutH = TileH_;
    static constexpr index_t TileOutW = TileW_;
    static constexpr index_t TileInH  = TileOutH * StrideH_;
    static constexpr index_t TileInW  = TileOutW * StrideW_;

    static constexpr index_t FilterH = FilterH_;
    static constexpr index_t FilterW = FilterW_;

    static constexpr index_t StrideH   = StrideH_;
    static constexpr index_t StrideW   = StrideW_;
    static constexpr index_t DilationH = DilationH_;
    static constexpr index_t DilationW = DilationW_;
    static constexpr index_t PadH      = PadH_;
    static constexpr index_t PadW      = PadW_;

    static constexpr index_t LdsTileH = TileInH + 2 * PadH;
    static constexpr index_t LdsTileW = TileInW + 2 * PadW;

    static constexpr index_t NBatch = NBatch_;

    static constexpr index_t SubTileH = SubTileH_;
    static constexpr index_t SubTileW = SubTileW_;

    static constexpr index_t InVectorSize  = InVectorSize_;
    static constexpr index_t OutVectorSize = OutVectorSize_;
    // Hardcoded to 2: enables v_dot2 (fp16x2) on FP16 and even/odd weight packing for
    // 2-column-per-step processing in RunConvolution when StrideW=1
    static constexpr index_t WeiVectorSize = 2;

    static constexpr index_t HRepeats      = integer_divide_ceil(TileOutH, SubTileH);
    static constexpr index_t WRepeats      = integer_divide_ceil(TileOutW, SubTileW);
    static constexpr index_t TotalSubTiles = HRepeats * WRepeats;
    static constexpr index_t TilePerWave   = WaveSize / TotalSubTiles;
    static constexpr index_t ThreadPerTile = WaveSize / TilePerWave;

    // LdsStride must satisfy: LdsStride - LdsTileW >= PadW (padding vector overflow guard)
    static constexpr index_t LdsStrideBase = integer_least_multiple(LdsTileW, InVectorSize);
    static constexpr index_t LdsStrideMin  = LdsTileW + PadW;
    static constexpr index_t LdsStride     = (LdsStrideBase >= LdsStrideMin)
                                                 ? LdsStrideBase
                                                 : integer_least_multiple(LdsStrideMin, InVectorSize);

    static constexpr index_t LdsTileSize  = LdsTileH * LdsStride;
    static constexpr index_t LdsInputSize = LdsTileSize * TilePerWave * sizeof(InDataType);
    static constexpr index_t LdsSize      = LdsInputSize;

    using InVector  = ext_vector_t<InDataType, InVectorSize>;
    using OutVector = ext_vector_t<OutDataType, OutVectorSize>;
    using WeiVector = ext_vector_t<WeiDataType, WeiVectorSize>;

    // Capped at 4 for LDS access: 4 * sizeof(fp32) = 16 bytes = ds_read_b128 max width.
    // Conservative for FP16 (could be 8), but keeps the code uniform across data types.
    static constexpr index_t InVectorSizeInternal  = (InVectorSize < 4) ? InVectorSize : 4;
    static constexpr index_t OutVectorSizeInternal = (OutVectorSize < 4) ? OutVectorSize : 4;

    using InVectorInternal  = ext_vector_t<InDataType, InVectorSizeInternal>;
    using OutVectorInternal = ext_vector_t<OutDataType, OutVectorSizeInternal>;
    using AccVectorInternal = ext_vector_t<AccDataType, OutVectorSizeInternal>;

    static_assert(std::is_same_v<InDataType, fp16_t> || std::is_same_v<InDataType, float>,
                  "Only fp16 and float are supported currently");
    static_assert(BlockSize == 64 || BlockSize == 128 || BlockSize == 256,
                  "BlockSize must be 64, 128, or 256");
    static_assert(TotalSubTiles <= WaveSize, "TotalSubTiles must not exceed WaveSize");
    static_assert(DilationH == 1 && DilationW == 1, "Only dilation=1 is supported currently");
    static_assert(FilterH == FilterW, "Only square filters are supported currently");
    static_assert(FilterH % 2 == 1, "Only odd filter sizes are supported (3, 5, 7, 9)");
    static_assert((InVectorSize & (InVectorSize - 1)) == 0 &&
                      (OutVectorSize & (OutVectorSize - 1)) == 0,
                  "InVectorSize and OutVectorSize must be powers of 2");
    static_assert(SubTileH <= TileOutH && SubTileW <= TileOutW,
                  "SubTile dimensions must not exceed Tile output dimensions");
};

// TODO: split DepthwiseConvFwdTraits into Shape (Tile/SubTile/NBatch) +
//       FilterParams (Filter/Stride/Dilation/Pad) + Traits (DataType/VectorSize),
//       following the TileGemmShape/TileGemmTraits pattern to reduce template parameters.

} // namespace ck_tile
