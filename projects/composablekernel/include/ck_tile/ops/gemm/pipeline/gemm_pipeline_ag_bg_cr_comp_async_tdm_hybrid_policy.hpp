// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

// Default policy for GemmPipelineAgBgCrCompAsyncTDMHybrid (Task 9,
// docs_gfx1250/GFX1250_WRW_REMAINING_TASKS.md).
//
// A operand: loaded via `global_load_async` (async copy straight to LDS).
// B operand: loaded via TDM (tensor data mover hardware descriptor straight to LDS).
//
// This struct is written as a *fresh* CRTP leaf directly extending
// `UniversalGemmBasePolicy<Self>` (not by inheriting from either
// `GemmPipelineAgBgCrCompAsyncDefaultPolicy` or
// `GemmPipelineAgBgCrCompTDMDefaultPolicy`). That matters: several
// `UniversalGemmBasePolicy` methods (`GetSmemSize`/`GetSmemSizeA`/`GetSmemSizeB`,
// `GetVectorSizeC`) call back into `Derived::template Make*LdsBlockDescriptor<Problem>()`
// / `Derived::template GetBlockGemm<Problem>()` via the CRTP parameter fixed at the
// point `UniversalGemmBasePolicy<Derived>` is instantiated. If this policy instead
// inherited from `GemmPipelineAgBgCrCompTDMDefaultPolicy<...>`, that CRTP parameter
// would stay bound to the TDM policy type, silently ignoring any override placed in a
// subclass, causing e.g. `GetSmemSize()` to size shared memory from a *different* A LDS
// descriptor than the one the pipeline body actually uses. Defining every method
// directly on this leaf type (copying bodies from the two donor policies as needed)
// avoids that trap entirely.
//
// A-side methods are intentionally *not* overridden here: for bf16 (this task's
// dtype) `GemmPipelineAgBgCrCompAsyncDefaultPolicy` itself forwards
// `MakeADramTileDistribution`/`MakeALdsBlockDescriptor` straight through to
// `UniversalGemmBasePolicy`'s generic implementation on gfx125 (its XOR-swizzle path
// is fp8/bf8-only, see `IsSupportedXorSwizzleDataType`), so inheriting the Base
// defaults unchanged reproduces Async's own bf16 behavior exactly.
//
// B-side methods (`MakeBDramTileDistribution`, `MakeBLdsBlockDescriptor`) are copied
// from `GemmPipelineAgBgCrCompTDMDefaultPolicy` verbatim: B keeps using TDM's coarse,
// hardware-descriptor-oriented DRAM distribution and its bank-conflict-avoiding LDS
// layout, unrelated to how A is loaded.
//
// Cluster-launch / multicast (TDM's cross-CU broadcast feature) is out of scope here:
// only B uses TDM, so multicast would only ever apply to B, and this first hybrid
// implementation does not wire it up. `GemmPipelineAgBgCrCompAsyncTDMHybrid` hardcodes
// `UseClusterLaunch = false`.
//
// K-subtiling (TDM v1's VGPR-budget-driven `sub_tile_num` split) is also out of scope:
// this hybrid follows `GemmPipelineAgBgCrCompAsync`'s simple ping-pong loop shape
// (single `block_gemm()` call per iteration, whole KPerBlock reduced at once), so
// `GetBlockGemm` below always selects `sub_tile_num = 1`.
struct GemmPipelineAgBgCrCompAsyncTDMHybridDefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncTDMHybridDefaultPolicy>
{
    using Base = UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncTDMHybridDefaultPolicy>;
    using Base::GetSmemPackA;
    using Base::GetSmemPackB;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::is_a_load_tr;
    using Base::is_b_load_tr;

    // B LDS data type: same default rule TDM's policy uses (no pk_int4_t swap needed
    // for the bf16/f16 configs this hybrid targets, kept for formula fidelity).
    template <typename Problem>
    using LdsBDataType = typename Problem::BDataType;

    static constexpr index_t VecByteSize = 16;

    // ---- A operand: plain async-load DRAM window wrapper ----
    // This hybrid never uses XOR swizzle (that path in
    // GemmPipelineAgBgCrCompAsyncDefaultPolicy is fp8/bf8-only), so this is always the
    // plain passthrough async_load_tile itself expects.
    template <typename Problem, typename Window>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadADramWindow(const Window& window)
    {
        return make_tile_window(window.get_bottom_tensor_view(),
                                window.get_window_lengths(),
                                window.get_window_origin());
    }

    // Copied verbatim from
    // GemmPipelineAgBgCrCompTDMDefaultPolicy<false>::MakeBDramTileDistribution. "currently
    // implement basic situation: the tile is divided into same parts"
    // -- the TDM hardware descriptor performs the fine per-lane addressing
    // internally, so this distribution only needs to describe the coarse
    // per-warp-group split, unlike A's per-lane Async distribution.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t warpNum   = BlockSize / get_warp_size();

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using BLayout =
            remove_cvref_t<std::tuple_element_t<number<0>{}, problem_bs_layout_t<Problem>>>;

        // Tile : KPerBlock X NPerBlock
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            static_assert(KPerBlock % warpNum == 0, "KPerBlock should be divided by warpNum");
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, KPerBlock / warpNum>, sequence<NPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
        // Tile : NPerBlock * KPerBlock
        else
        {
            static_assert(NPerBlock % warpNum == 0, "NPerBlock should be divided by warpNum");
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<>,
                    tuple<sequence<warpNum, NPerBlock / warpNum>, sequence<KPerBlock>>,
                    tuple<sequence<1>>,
                    tuple<sequence<0>>,
                    sequence<1, 2>,
                    sequence<1, 0>>{},
                bool_constant<true>{});
        }
    }

    // ---- B operand: TDM's bank-conflict-avoiding LDS descriptor ----
    // Copied verbatim from GemmPipelineAgBgCrCompTDMDefaultPolicy<false>::MakeBLdsBlockDescriptor.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        if constexpr(Base::template is_b_load_tr<Problem>)
        {
            return Base::template MakeBLdsBlockDescriptorForTrLoad<Problem>();
        }
        else
        {
            constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
            constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

            constexpr auto LdsPaddingConfigB = Base::template GetLdsPaddingConfig<Problem, false>();
            constexpr auto IsNeedPadding     = LdsPaddingConfigB[Base::I0];
            // set to -1 to make sure PaddingDataAmount = 0 when IsNeedPadding = false
            constexpr auto PaddingAmount = IsNeedPadding ? LdsPaddingConfigB[Base::I1] : -1;
            using BDataType              = LdsBDataType<Problem>;
            constexpr index_t PackedSize = numeric_traits<BDataType>::PackedSize;
            constexpr auto DataTypeSize  = sizeof(BDataType);

            constexpr index_t BVectorLen = VecByteSize / DataTypeSize * PackedSize;
            constexpr index_t NLdsLayerRequired =
                get_n_lds_banks() * get_n_dwords_per_128b() / KPerBlock / DataTypeSize * PackedSize;
            constexpr auto NLdsLayer = max(1, NLdsLayerRequired);
            // calculate how many elements to pad to avoid bank conflict
            constexpr index_t BytesPerDword = sizeof(int32_t);
            constexpr auto PaddingDataAmount =
                (PaddingAmount + 1) * BytesPerDword / DataTypeSize * PackedSize;

            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NPerBlock / NLdsLayer>{},
                           number<KPerBlock / BVectorLen * NLdsLayer>{},
                           number<BVectorLen>{}),
                make_tuple(number<KPerBlock * NLdsLayer + PaddingDataAmount>{},
                           number<BVectorLen>{},
                           number<1>{}),
                number<BVectorLen>{},
                number<1>{});

            constexpr auto b_lds_block_desc_1 = transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(make_pass_through_transform(number<NPerBlock / NLdsLayer>{}),
                           make_unmerge_transform(
                               make_tuple(number<NLdsLayer>{}, number<KPerBlock / BVectorLen>{})),
                           make_pass_through_transform(number<BVectorLen>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            constexpr auto b_lds_block_desc = transform_tensor_descriptor(
                b_lds_block_desc_1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<NPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<KPerBlock / BVectorLen>{}, number<BVectorLen>{}))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return b_lds_block_desc;
        }
    }

    // ---- BlockGemm selection: no K-subtiling (sub_tile_num == 1), WGAttrNumAccess ----
    // computed the same way GemmPipelineAgBgCrCompAsyncDefaultPolicy does (this
    // computation is Problem/dtype-derived only -- it depends on each operand's own
    // `is_a_load_tr`/`is_b_load_tr` and warp-tile shape, not on which mechanism moved
    // the data into LDS -- so it is valid unchanged for the hybrid).
    template <bool IsLoadTr, typename DataType, index_t ThreadElements>
    CK_TILE_HOST_DEVICE static constexpr auto CalculateWGAttrNumAccess()
    {
        if constexpr(IsLoadTr)
        {
            constexpr index_t vector_size =
                DS_READ_TR_SIZE() / sizeof(DataType) * numeric_traits<DataType>::PackedSize;
            if constexpr(vector_size == ThreadElements)
                return WGAttrNumAccessEnum::Single;
            else if constexpr(vector_size * 2 == ThreadElements)
                return WGAttrNumAccessEnum::Double;
            else if constexpr(vector_size * 4 == ThreadElements)
                return WGAttrNumAccessEnum::Quad;
            else
                return WGAttrNumAccessEnum::Invalid;
        }
        else
        {
            constexpr index_t bytes_per_lane =
                sizeof(DataType) * ThreadElements / numeric_traits<DataType>::PackedSize;
            constexpr index_t ds_read_b128_width = 16;
            if constexpr(bytes_per_lane <= ds_read_b128_width)
                return WGAttrNumAccessEnum::Single;
            else if constexpr(bytes_per_lane <= ds_read_b128_width * 2)
                return WGAttrNumAccessEnum::Double;
            else if constexpr(bytes_per_lane <= ds_read_b128_width * 4)
                return WGAttrNumAccessEnum::Quad;
            else
                return WGAttrNumAccessEnum::Invalid;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWGAttrNumAccess()
    {
        using WarpTile                      = typename Problem::BlockGemmShape::WarpTile;
        constexpr index_t a_thread_elements = WarpTile::at(I0) * WarpTile::at(I2) / get_warp_size();
        constexpr index_t b_thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();

        constexpr auto num_access_a = CalculateWGAttrNumAccess<Base::template is_a_load_tr<Problem>,
                                                               typename Problem::ADataType,
                                                               a_thread_elements>();
        constexpr auto num_access_b = CalculateWGAttrNumAccess<Base::template is_b_load_tr<Problem>,
                                                               typename Problem::BDataType,
                                                               b_thread_elements>();

        if constexpr(num_access_a == WGAttrNumAccessEnum::Invalid ||
                     num_access_b == WGAttrNumAccessEnum::Invalid)
            return WGAttrNumAccessEnum::Invalid;
        else if constexpr(static_cast<index_t>(num_access_a) >= static_cast<index_t>(num_access_b))
            return num_access_a;
        else
            return num_access_b;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        constexpr auto wg_attr_num_access = GetWGAttrNumAccess<Problem>();
        constexpr index_t sub_tile_num    = 1;

        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(Base::I0),
                                            WarpTile::at(Base::I1),
                                            WarpTile::at(Base::I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm,
                                                                    sub_tile_num>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
