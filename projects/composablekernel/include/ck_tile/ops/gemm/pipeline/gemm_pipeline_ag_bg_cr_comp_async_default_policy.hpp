// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAgBgCrCompAsync
// Customized methods: MakeALdsBlockDescriptor, MakeBLdsBlockDescriptor
// GetBlockGemm implementation is copied from GemmPipelineAgBgCrCompV4DefaultPolicy
template <bool EnableSubTile = false>
struct GemmPipelineAgBgCrCompAsyncDefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncDefaultPolicy<EnableSubTile>>
{
    static constexpr auto ATileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::warp_raked;
    using Base = UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncDefaultPolicy<EnableSubTile>>;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::is_a_load_tr;
    using Base::is_b_load_tr;

    static constexpr index_t kLdsBankBytes = 4;
#if defined(__gfx950__)
    static constexpr index_t kLdsBankCount = 64;
#else
    static constexpr index_t kLdsBankCount = 32;
#endif
    static constexpr index_t kLdsRowBytes = kLdsBankCount * kLdsBankBytes;
    static constexpr index_t MaxVecSize   = get_max_mem_vec_inst_width();

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsStoreBlockDescriptor()
    {
#if defined(__gfx125__)
        return Base::template MakeBLdsBlockDescriptor<Problem>();
#else
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t ROWS      = is_a_load_tr<Problem> ? KPerBlock : MPerBlock;
        constexpr index_t COLS      = is_a_load_tr<Problem> ? MPerBlock : KPerBlock;
        return make_naive_tensor_descriptor(make_tuple(number<ROWS>{}, number<COLS>{}),
                                            make_tuple(number<COLS>{}, number<1>{}));
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsStoreBlockDescriptor()
    {
#if defined(__gfx125__)
        return Base::template MakeBLdsBlockDescriptor<Problem>();
#else
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t ROWS      = is_a_load_tr<Problem> ? KPerBlock : NPerBlock;
        constexpr index_t COLS      = is_a_load_tr<Problem> ? NPerBlock : KPerBlock;
        return make_naive_tensor_descriptor(make_tuple(number<ROWS>{}, number<COLS>{}),
                                            make_tuple(number<COLS>{}, number<1>{}));
#endif
    }

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
#if defined(__gfx125__)
        return Base::template MakeALdsBlockDescriptor<Problem>();
#else
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(Base::template is_a_load_tr<Problem>)
        {
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor( //
                make_tuple(number<KPerBlock>{}, number<MPerBlock>{}),
                make_tuple(number<MPerBlock>{}, number<1>{}),
                number<MPerBlock>{},
                number<1>{});
            return a_lds_block_desc_0;
        }
        else
        {
            constexpr index_t APackedSize = numeric_traits<OverrideADataType>::PackedSize;
            constexpr index_t KPack       = MaxVecSize / sizeof(OverrideADataType) * APackedSize;
            constexpr index_t PacksPerLdsRow =
                (kLdsRowBytes / sizeof(OverrideADataType) * APackedSize) / KPack;

            constexpr index_t L2              = KPack;
            constexpr index_t L1              = PacksPerLdsRow;
            constexpr index_t L0              = MPerBlock * KPerBlock / (L1 * L2);
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<L0>{}, number<L1>{}, number<L2>{}),
                make_tuple(number<L1 * L2>{}, number<L2>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            const auto a_lds_block_desc_1 = transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<L0>{}, number<L1>{})),
                           make_pass_through_transform(L2)),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0, 1>{}, sequence<2>{}));

            constexpr index_t KPacksPerBlock = KPerBlock / KPack;
            constexpr index_t MRowsPerLdsRow = integer_divide_ceil(PacksPerLdsRow, KPacksPerBlock);
            const auto a_lds_block_desc_2    = transform_tensor_descriptor(
                a_lds_block_desc_1,
                make_tuple(make_pass_through_transform(number<L0>{}),
                           make_unmerge_transform(
                               make_tuple(number<MRowsPerLdsRow>{}, number<KPacksPerBlock>{})),
                           make_pass_through_transform(number<L2>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            const auto a_lds_block_desc_3 = transform_tensor_descriptor(
                a_lds_block_desc_2,
                make_tuple(
                    make_merge_transform(make_tuple(number<L0>{}, number<MRowsPerLdsRow>{})),
                    make_merge_transform(make_tuple(number<KPacksPerBlock>{}, number<L2>{}))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
            return a_lds_block_desc_3;
        }
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
#if defined(__gfx125__)
        return Base::template MakeBLdsBlockDescriptor<Problem>();
#else
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        if constexpr(Base::template is_b_load_tr<Problem>)
        {
            constexpr auto b_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<KPerBlock>{}, number<NPerBlock>{}),
                                             make_tuple(number<NPerBlock>{}, number<1>{}),
                                             number<NPerBlock>{},
                                             number<1>{});
            return b_lds_block_desc_0;
        }
        else
        {
            using BDataType               = remove_cvref_t<typename Problem::BDataType>;
            constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;
            constexpr index_t KPack       = MaxVecSize / sizeof(BDataType) * BPackedSize;
            constexpr index_t PacksPerLdsRow =
                (kLdsRowBytes / sizeof(typename Problem::BDataType) * BPackedSize) / KPack;

            constexpr index_t L2              = KPack;
            constexpr index_t L1              = PacksPerLdsRow;
            constexpr index_t L0              = NPerBlock * KPerBlock / (L1 * L2);
            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<L0>{}, number<L1>{}, number<L2>{}),
                make_tuple(number<L1 * L2>{}, number<L2>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            const auto b_lds_block_desc_1 = transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<L0>{}, number<L1>{})),
                           make_pass_through_transform(number<L2>{})),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0, 1>{}, sequence<2>{}));

            constexpr index_t KPacksPerBlock = KPerBlock / KPack;
            constexpr index_t NRowsPerLdsRow = integer_divide_ceil(PacksPerLdsRow, KPacksPerBlock);
            const auto b_lds_block_desc_2    = transform_tensor_descriptor(
                b_lds_block_desc_1,
                make_tuple(make_pass_through_transform(number<L0>{}),
                           make_unmerge_transform(
                               make_tuple(number<NRowsPerLdsRow>{}, number<KPacksPerBlock>{})),
                           make_pass_through_transform(number<L2>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

            const auto b_lds_block_desc_3 = transform_tensor_descriptor(
                b_lds_block_desc_2,
                make_tuple(
                    make_merge_transform(make_tuple(number<L0>{}, number<NRowsPerLdsRow>{})),
                    make_merge_transform(make_tuple(number<KPacksPerBlock>{}, number<L2>{}))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
            return b_lds_block_desc_3;
        }
#endif
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetEstimatedVgprCount()
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;
        using CDataType = remove_cvref_t<typename Problem::CDataType>;

        constexpr index_t MWarps       = Problem::BlockGemmShape::BlockWarps::at(Base::I0);
        constexpr index_t NWarps       = Problem::BlockGemmShape::BlockWarps::at(Base::I1);
        constexpr index_t warpSize     = get_warp_size();
        constexpr index_t BlockSize    = Problem::kBlockSize;
        constexpr index_t BytesPerVGPR = 4;
        constexpr index_t AccVGPRNum =
            sizeof(CDataType) * MPerBlock * NPerBlock / BlockSize / BytesPerVGPR;

        constexpr index_t DoubleBufferFactor = 3;

        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;

        constexpr index_t ALoadVGPRNum = sizeof(ADataType) / APackedSize * MPerBlock * KPerBlock /
                                         MWarps / warpSize / BytesPerVGPR * DoubleBufferFactor;

        constexpr index_t BLoadVGPRNum = sizeof(BDataType) / BPackedSize * NPerBlock * KPerBlock /
                                         NWarps / warpSize / BytesPerVGPR * DoubleBufferFactor;

        constexpr index_t TotalInputVGPRNum = ALoadVGPRNum + BLoadVGPRNum;

        return make_tuple(number<AccVGPRNum>{}, number<TotalInputVGPRNum>{});
    }

    // this function is used to get SubTile Number
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetPipelineSubTileNum()
    {
        constexpr auto estimated_vgpr = GetEstimatedVgprCount<Problem>();

        constexpr auto acc_vgpr_num   = estimated_vgpr.at(number<0>{});
        constexpr auto input_vgpr_num = estimated_vgpr.at(number<1>{});

        constexpr index_t vgpr_capacity = get_max_vgpr_count();
        // sub tile number; have 1, 2, 4 choices
        constexpr index_t sub_tile_num = ((input_vgpr_num + acc_vgpr_num) <= vgpr_capacity) ? 1
                                         : ((input_vgpr_num / 2 + acc_vgpr_num) <= vgpr_capacity)
                                             ? 2
                                             : 4;

        return number<sub_tile_num>{};
    }

    // Methods for async byte-based loading (similar to mx_flatmm)
    template <typename Problem, typename WindowTmp>
    CK_TILE_DEVICE static constexpr auto MakeAAsyncLoadBytesDramWindow(const WindowTmp& window_tmp)
    {
        using ADataType               = remove_cvref_t<typename Problem::ADataType>;
        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        constexpr index_t MPerBlock   = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;

        constexpr auto ndims = std::decay_t<decltype(window_tmp)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");
        auto&& tensor_view_tmp  = window_tmp.get_bottom_tensor_view();
        const auto [rows, cols] = tensor_view_tmp.get_tensor_descriptor().get_lengths();

        constexpr index_t K2 = MaxVecSize / sizeof(ADataType) * APackedSize;
        constexpr index_t K1 = KPerBlock / K2;
        const index_t K0     = cols / (K1 * K2);
        const auto col_lens  = make_tuple(K0, number<K1>{}, number<K2>{});

        constexpr index_t LdsPackPerRow = kLdsRowBytes / MaxVecSize;
        constexpr index_t M1            = integer_divide_ceil(LdsPackPerRow, K1);
        const index_t M0                = integer_divide_ceil(rows, M1);
        const auto row_lens             = make_tuple(M0, number<M1>{});

        const auto d0 = make_naive_tensor_descriptor_packed(container_concat(row_lens, col_lens));
        const auto desc_0 = decltype(d0)(
            d0.get_transforms(), tensor_view_tmp.get_tensor_descriptor().get_element_space_size());
        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_merge_transform(make_tuple(number<M1>{}, number<K1>{})),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<2>{}, sequence<1>{}, sequence<3>{}));
        constexpr index_t M1K1 = M1 * K1;
        const auto desc_2 =
            transform_tensor_descriptor(desc_1,
                                        make_tuple(make_xor_transform(make_tuple(M0, M1K1)),
                                                   make_pass_through_transform(K0),
                                                   make_pass_through_transform(number<K2>{})),
                                        make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}),
                                        make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));
        const auto desc_3 = transform_tensor_descriptor(
            desc_2,
            make_tuple(make_pass_through_transform(M0),
                       make_unmerge_transform(make_tuple(M1, K1)),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(K2)),
            make_tuple(sequence<0>{}, sequence<2>{}, sequence<1>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));
        const auto desc =
            transform_tensor_descriptor(desc_3,
                                        make_tuple(make_merge_transform_v3_division_mod(row_lens),
                                                   make_merge_transform_v3_division_mod(col_lens)),
                                        make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        auto&& byte_ptr         = &(tensor_view_tmp.get_buffer_view()(0));
        auto&& byte_tensor_view = make_tensor_view<address_space_enum::global>(byte_ptr, desc);

        auto&& origin_tmp = window_tmp.get_window_origin();

        // Create tile distribution inline (reuse K2, K1, K0 from above)
        static_assert(KPerBlock <= kLdsRowBytes / sizeof(ADataType));
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t K1_dstr   = K2;
        constexpr index_t K0_dstr   = KPerBlock / K2;
        constexpr index_t M2_dstr   = WaveSize / K0_dstr;
        constexpr index_t M1_dstr   = BlockSize / WaveSize;
        constexpr index_t M0_dstr   = MPerBlock / (M1_dstr * M2_dstr);

        const auto tile_dstr = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<M0_dstr, M1_dstr, M2_dstr>, sequence<K0_dstr, K1_dstr>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});

        return make_tile_window(byte_tensor_view,
                                make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                {origin_tmp[0], origin_tmp[1]},
                                tile_dstr);
    }

    template <typename Problem, typename WindowTmp>
    CK_TILE_DEVICE static auto MakeBAsyncLoadBytesDramWindow(const WindowTmp& window_tmp)
    {
        using BDataType               = remove_cvref_t<typename Problem::BDataType>;
        constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;
        constexpr index_t NPerBlock   = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;

        constexpr auto ndims = std::decay_t<decltype(window_tmp)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");
        auto&& tensor_view_tmp  = window_tmp.get_bottom_tensor_view();
        const auto [rows, cols] = tensor_view_tmp.get_tensor_descriptor().get_lengths();

        constexpr index_t K2 = MaxVecSize / sizeof(BDataType) * BPackedSize;
        constexpr index_t K1 = KPerBlock / K2;
        const index_t K0     = cols / (K1 * K2);
        const auto col_lens  = make_tuple(K0, number<K1>{}, number<K2>{});

        constexpr index_t LdsPackPerRow = kLdsRowBytes / MaxVecSize;
        constexpr index_t N1            = integer_divide_ceil(LdsPackPerRow, K1);
        const index_t N0                = integer_divide_ceil(rows, N1);
        const auto row_lens             = make_tuple(N0, number<N1>{});

        const auto d0 = make_naive_tensor_descriptor_packed(container_concat(row_lens, col_lens));
        const auto desc_0 = decltype(d0)(
            d0.get_transforms(), tensor_view_tmp.get_tensor_descriptor().get_element_space_size());
        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(N0),
                       make_merge_transform(make_tuple(number<N1>{}, number<K1>{})),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<2>{}, sequence<1>{}, sequence<3>{}));
        constexpr index_t N2K1 = N1 * K1;
        const auto desc_2 =
            transform_tensor_descriptor(desc_1,
                                        make_tuple(make_xor_transform(make_tuple(N0, N2K1)),
                                                   make_pass_through_transform(K0),
                                                   make_pass_through_transform(number<K2>{})),
                                        make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}),
                                        make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));
        const auto desc_3 = transform_tensor_descriptor(
            desc_2,
            make_tuple(make_pass_through_transform(N0),
                       make_unmerge_transform(make_tuple(N1, K1)),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(K2)),
            make_tuple(sequence<0>{}, sequence<2>{}, sequence<1>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));
        const auto desc =
            transform_tensor_descriptor(desc_3,
                                        make_tuple(make_merge_transform_v3_division_mod(row_lens),
                                                   make_merge_transform_v3_division_mod(col_lens)),
                                        make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        auto&& byte_ptr         = &(tensor_view_tmp.get_buffer_view()(0));
        auto&& byte_tensor_view = make_tensor_view<address_space_enum::global>(byte_ptr, desc);

        auto&& origin_tmp = window_tmp.get_window_origin();

        // Create tile distribution inline (reuse K2, K1, K0 from above)
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t K1_dstr   = K2;
        constexpr index_t K0_dstr   = KPerBlock / K2;
        constexpr index_t N2_dstr   = WaveSize / K0_dstr;
        constexpr index_t N1_dstr   = BlockSize / WaveSize;
        constexpr index_t N0_dstr   = NPerBlock / (N2_dstr * N1_dstr);

        const auto tile_dstr = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<N0_dstr, N1_dstr, N2_dstr>, sequence<K0_dstr, K1_dstr>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});

        return make_tile_window(byte_tensor_view,
                                make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                {origin_tmp[0], origin_tmp[1]},
                                tile_dstr);
    }

    template <typename Problem, typename ADataType>
    CK_TILE_DEVICE static constexpr auto GetALdsStoreTensorView(void* smem)
    {
        ADataType* __restrict__ p_a_lds = static_cast<ADataType*>(smem);
        constexpr auto a_lds_block_desc = MakeALdsStoreBlockDescriptor<Problem, ADataType>();
        return make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);
    }

    template <typename Problem, typename ADataType>
    CK_TILE_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr index_t APackedSize   = numeric_traits<ADataType>::PackedSize;
        constexpr auto a_lds_block_desc = MakeALdsStoreBlockDescriptor<Problem, ADataType>();
        return integer_least_multiple(
            sizeof(ADataType) * a_lds_block_desc.get_element_space_size() / APackedSize, 16);
    }

    template <typename Problem, typename BDataType>
    CK_TILE_DEVICE static constexpr index_t GetSmemSizeB()
    {
        constexpr index_t BPackedSize   = numeric_traits<BDataType>::PackedSize;
        constexpr auto b_lds_block_desc = MakeBLdsStoreBlockDescriptor<Problem>();
        return integer_least_multiple(
            sizeof(BDataType) * b_lds_block_desc.get_element_space_size() / BPackedSize, 16);
    }

    template <typename Problem, typename BDataType>
    CK_TILE_DEVICE static constexpr auto GetBLdsStoreTensorView(void* smem)
    {
        BDataType* __restrict__ p_b_lds = static_cast<BDataType*>(smem);
        constexpr auto b_lds_block_desc = MakeBLdsStoreBlockDescriptor<Problem>();
        return make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);
    }

    template <typename Problem, typename ADataType>
    CK_TILE_DEVICE static constexpr auto GetALdsLoadTensorView(void* smem)
    {
        ADataType* __restrict__ p_a_lds = static_cast<ADataType*>(smem);
        constexpr auto a_lds_block_desc = MakeALdsBlockDescriptor<Problem, ADataType>();
        return make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);
    }

    template <typename Problem, typename BDataType>
    CK_TILE_DEVICE static constexpr auto GetBLdsLoadTensorView(void* smem)
    {
        BDataType* __restrict__ p_b_lds = static_cast<BDataType*>(smem);
        constexpr auto b_lds_block_desc = MakeBLdsBlockDescriptor<Problem>();
        return make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

#if defined(__gfx950__)
        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::AComputeDataType);
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();
        constexpr auto wg_attr_num_access =
            !(Base::template is_a_load_tr<Problem> || Base::template is_b_load_tr<Problem>)
                ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements     ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements ? WGAttrNumAccessEnum::Quad
                                                 : WGAttrNumAccessEnum::Invalid;
#else
        constexpr auto wg_attr_num_access = WGAttrNumAccessEnum::Default;
#endif

        constexpr auto pipeline_tune_params = GetPipelineSubTileNum<Problem>();
        constexpr index_t sub_tile_num      = EnableSubTile ? pipeline_tune_params.value : 1;

        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(Base::I0),
                                            WarpTile::at(Base::I1),
                                            WarpTile::at(Base::I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
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
