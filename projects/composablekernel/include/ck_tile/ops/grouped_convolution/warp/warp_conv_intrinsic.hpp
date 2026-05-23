// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

namespace wcnn_mods {
// Helper functions to translate convolution parameters to MOD flags
// __builtin_amdgcn_convolve* merged 3 MOD flags into single aux_data
// aux_data bit 0  ~ 5:  MOD0
// aux_data bit 6  ~ 17: MOD1    VOP5M Filter 1x1
//          bit 6  ~ 13: MOD1    VOP6M Filter 3x3
// aux_data bit 18 ~ 21: MOD2    VOP5M Filter 1x1
//          bit 18 ~ 25: MOD2    VOP6M Filter 3x3

constexpr index_t WcnnAuxData_Mod0 = 0;
constexpr index_t WcnnAuxData_Mod1 = 6;
constexpr index_t WcnnAuxData_Mod2 = 18;

// Filters
template <index_t FilterSizeY, index_t FilterSizeX>
constexpr index_t GetFilterSizeMod()
{
    // MOD0[5:3]
    static_assert(FilterSizeY == FilterSizeX, "Only square filter supported!");
    return (FilterSizeY == 3) ? (1 << (3 + WcnnAuxData_Mod0)) : 0;
}

template <bool StartLane16>
constexpr index_t GetStartLaneMod()
{
    // MOD2[2]
    return StartLane16 ? (1 << (2 + WcnnAuxData_Mod1)) : 0;
}

template <bool Signed>
constexpr index_t GetWeightSignedMod()
{
    // MOD2[0]
    return Signed ? (1 << WcnnAuxData_Mod2) : 0;
}

template <index_t DilationX, index_t DilationY>
constexpr index_t GetDilationMod()
{
    // MOD1[1:0]
    return ((DilationX == 2) ? (1 << WcnnAuxData_Mod1) : 0) |
           ((DilationY == 2) ? (2 << WcnnAuxData_Mod1) : 0);
}

// Accumulator
template <bool IsBias>
constexpr index_t GetAccumIsBiasMod()
{
    // MOD2[1]
    return IsBias ? (1 << (1 + WcnnAuxData_Mod2)) : 0;
}

template <bool Aco>
constexpr index_t GetAccumChannelOrderMod()
{
    // MOD1[3]
    return Aco ? (1 << (3 + WcnnAuxData_Mod1)) : 0;
}

template <bool IntScale>
constexpr index_t GetIntScaleMod()
{
    // MOD2[3]
    return IntScale ? (1 << (3 + WcnnAuxData_Mod2)) : 0;
}

// Tensor
template <bool Signed>
constexpr index_t GetTensorSignedMod()
{
    // MOD2[2]
    return Signed ? (1 << (2 + WcnnAuxData_Mod2)) : 0;
}

template <index_t Iters>
constexpr index_t GetItersMod()
{
    // MOD1[7:6], 1x1 only
    static_assert(Iters <= 4, "Filter 1x1 only support Iters 0~3!!");
    return Iters ? ((Iters - 1) << (6 + WcnnAuxData_Mod1)) : 0;
}

template <typename DataType>
constexpr bool GetAcoFlag()
{
    // ACO uses different channel order for accumulators, which requires different MOD settings
    if constexpr(is_any_of<DataType, fp16_t, bf16_t>::value)
        return true;
    else
        return false;
}
} // namespace wcnn_mods

// Base class: provides GetMods() for all specializations
template <bool AcoFlag,
          index_t FilterSizeY,
          index_t FilterSizeX,
          index_t DilationY,
          index_t DilationX,
          index_t NumIter>
struct WarpConvIntrinsicBase
{
    template <bool TensorSigned = false, bool WeightSigned = false, bool HighLane = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetMods()
    {
        return wcnn_mods::GetFilterSizeMod<FilterSizeY, FilterSizeX>() |
               wcnn_mods::GetAccumIsBiasMod<false>() |
               wcnn_mods::GetAccumChannelOrderMod<AcoFlag>() |
               wcnn_mods::GetWeightSignedMod<WeightSigned>() |
               wcnn_mods::GetTensorSignedMod<TensorSigned>() | wcnn_mods::GetItersMod<NumIter>() |
               wcnn_mods::GetDilationMod<DilationX, DilationY>() |
               wcnn_mods::GetStartLaneMod<HighLane>();
    }
};

// Primary template — static_assert on unsupported combinations
template <typename ImgDataType,
          typename AccDataType,
          bool AcoFlag,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t FilterSizeY,
          index_t FilterSizeX,
          index_t DilationY,
          index_t DilationX,
          index_t NumIter>
struct WarpConvIntrinsic
    : WarpConvIntrinsicBase<AcoFlag, FilterSizeY, FilterSizeX, DilationY, DilationX, NumIter>
{
    static_assert(sizeof(ImgDataType) == 0,
                  "Unsupported data type combination for warp conv intrinsic");
};

// ============================================================
// fp16, 4x2 tile, 1x1 filter
// ============================================================

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<half_t, float, AcoFlag, 4, 2, 1, 1, DilationY, DilationX, 1>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 1>
{
    using WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 1>::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto call(AccVec acc, WeiVec wei, ImgVec a0)
    {
        return __builtin_amdgcn_convolve_f32_f16_4x2(acc, wei, a0, GetMods(), false);
    }
};

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<half_t, float, AcoFlag, 4, 2, 1, 1, DilationY, DilationX, 2>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 2>
{
    using WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 2>::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto call(AccVec acc, WeiVec wei, ImgVec a0, ImgVec a1)
    {
        return __builtin_amdgcn_convolve_f32_f16_4x2(acc, wei, a0, a1, GetMods(), false);
    }
};

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<half_t, float, AcoFlag, 4, 2, 1, 1, DilationY, DilationX, 3>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 3>
{
    using WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 3>::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto call(AccVec acc, WeiVec wei, ImgVec a0, ImgVec a1, ImgVec a2)
    {
        return __builtin_amdgcn_convolve_f32_f16_4x2(acc, wei, a0, a1, a2, GetMods(), false);
    }
};

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<half_t, float, AcoFlag, 4, 2, 1, 1, DilationY, DilationX, 4>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 4>
{
    using WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 4>::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto
    call(AccVec acc, WeiVec wei, ImgVec a0, ImgVec a1, ImgVec a2, ImgVec a3)
    {
        return __builtin_amdgcn_convolve_f32_f16_4x2(acc, wei, a0, a1, a2, a3, GetMods(), false);
    }
};

// ============================================================
// bf16, 4x2 tile, 1x1 filter
// ============================================================

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<bf16_t, float, AcoFlag, 4, 2, 1, 1, DilationY, DilationX, 1>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 1>
{
    using WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 1>::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto call(AccVec acc, WeiVec wei, ImgVec a0)
    {
        return __builtin_amdgcn_convolve_f32_bf16_4x2(acc, wei, a0, GetMods(), false);
    }
};

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<bf16_t, float, AcoFlag, 4, 2, 1, 1, DilationY, DilationX, 2>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 2>
{
    using WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 2>::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto call(AccVec acc, WeiVec wei, ImgVec a0, ImgVec a1)
    {
        return __builtin_amdgcn_convolve_f32_bf16_4x2(acc, wei, a0, a1, GetMods(), false);
    }
};

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<bf16_t, float, AcoFlag, 4, 2, 1, 1, DilationY, DilationX, 3>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 3>
{
    using WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 3>::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto call(AccVec acc, WeiVec wei, ImgVec a0, ImgVec a1, ImgVec a2)
    {
        return __builtin_amdgcn_convolve_f32_bf16_4x2(acc, wei, a0, a1, a2, GetMods(), false);
    }
};

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<bf16_t, float, AcoFlag, 4, 2, 1, 1, DilationY, DilationX, 4>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 4>
{
    using WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 4>::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto
    call(AccVec acc, WeiVec wei, ImgVec a0, ImgVec a1, ImgVec a2, ImgVec a3)
    {
        return __builtin_amdgcn_convolve_f32_bf16_4x2(acc, wei, a0, a1, a2, a3, GetMods(), false);
    }
};

// ============================================================
// fp16, 4x4 tile, 1x1 filter
// ============================================================

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<half_t, half_t, AcoFlag, 4, 4, 1, 1, DilationY, DilationX, 1>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 1>
{
    using WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 1>::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto call(AccVec acc, WeiVec wei, ImgVec a0)
    {
        return __builtin_amdgcn_convolve_f16_f16_4x4(acc, wei, a0, GetMods(), false);
    }
};

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<half_t, half_t, AcoFlag, 4, 4, 1, 1, DilationY, DilationX, 2>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 2>
{
    using WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 2>::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto call(AccVec acc, WeiVec wei, ImgVec a0, ImgVec a1)
    {
        return __builtin_amdgcn_convolve_f16_f16_4x4(acc, wei, a0, a1, GetMods(), false);
    }
};

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<half_t, half_t, AcoFlag, 4, 4, 1, 1, DilationY, DilationX, 3>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 3>
{
    using WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 3>::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto call(AccVec acc, WeiVec wei, ImgVec a0, ImgVec a1, ImgVec a2)
    {
        return __builtin_amdgcn_convolve_f16_f16_4x4(acc, wei, a0, a1, a2, GetMods(), false);
    }
};

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<half_t, half_t, AcoFlag, 4, 4, 1, 1, DilationY, DilationX, 4>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 4>
{
    using WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 4>::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto
    call(AccVec acc, WeiVec wei, ImgVec a0, ImgVec a1, ImgVec a2, ImgVec a3)
    {
        return __builtin_amdgcn_convolve_f16_f16_4x4(acc, wei, a0, a1, a2, a3, GetMods(), false);
    }
};

// ============================================================
// fp16, 8x4 tile, 1x1 filter
// ============================================================

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<half_t, half_t, AcoFlag, 8, 4, 1, 1, DilationY, DilationX, 1>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 1>
{
    using Base1 = WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 1>;
    using Base1::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto call(AccVec acc, WeiVec wei, ImgVec a0)
    {
        return __builtin_amdgcn_convolve_f16_f16_8x4(
            acc, wei, a0, Base1::template GetMods<false, false, HighLane>(), false);
    }
};

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<half_t, half_t, AcoFlag, 8, 4, 1, 1, DilationY, DilationX, 2>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 2>
{
    using Base2 = WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 2>;
    using Base2::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto call(AccVec acc, WeiVec wei, ImgVec a0, ImgVec a1)
    {
        return __builtin_amdgcn_convolve_f16_f16_8x4(
            acc, wei, a0, a1, Base2::template GetMods<false, false, HighLane>(), false);
    }
};

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<half_t, half_t, AcoFlag, 8, 4, 1, 1, DilationY, DilationX, 3>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 3>
{
    using Base3 = WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 3>;
    using Base3::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto call(AccVec acc, WeiVec wei, ImgVec a0, ImgVec a1, ImgVec a2)
    {
        return __builtin_amdgcn_convolve_f16_f16_8x4(
            acc, wei, a0, a1, a2, Base3::template GetMods<false, false, HighLane>(), false);
    }
};

template <bool AcoFlag, index_t DilationY, index_t DilationX>
struct WarpConvIntrinsic<half_t, half_t, AcoFlag, 8, 4, 1, 1, DilationY, DilationX, 4>
    : WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 4>
{
    using Base4 = WarpConvIntrinsicBase<AcoFlag, 1, 1, DilationY, DilationX, 4>;
    using Base4::GetMods;
    template <bool HighLane = false, typename AccVec, typename WeiVec, typename ImgVec>
    CK_TILE_DEVICE static auto
    call(AccVec acc, WeiVec wei, ImgVec a0, ImgVec a1, ImgVec a2, ImgVec a3)
    {
        return __builtin_amdgcn_convolve_f16_f16_8x4(
            acc, wei, a0, a1, a2, a3, Base4::template GetMods<false, false, HighLane>(), false);
    }
};

} // namespace ck_tile
