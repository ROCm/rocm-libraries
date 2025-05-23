// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"

namespace ck {

// GEMM:
//   input : A[M, K]
//   input : B[N, K]
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   output : F[M, N0], where N0 is number of blocks along N dimension
//   output : G[M, N0], where N0 is number of blocks along N dimension
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
//   F, G = welford(E)
// Assume:
//   D0, D1, ... and E have the same layout
//   Calculate mean & variance along N dimension for E
template <typename ABDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EMeanVarDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          typename AGridDesc_M_K,
          typename BGridDesc_N_K,
          typename DsGridDesc_M_N,
          typename EGridDesc_M_N,
          typename MeanVarGridDesc_M_NBlock,
          typename CountGridDesc_M_NBlock,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename PostShuffleThreadClusterSize_M_N,
          index_t PostShuffleScalarPerVector,
          LoopScheduler LoopSched,
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct GridwiseGemmMultipleDWelfordFirstHalf_xdl_cshuffle
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // K1 should be Number<...>
    static constexpr auto AK1         = Number<AK1Value>{};
    static constexpr auto BK1         = Number<BK1Value>{};
    static constexpr auto AK0PerBlock = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0PerBlock = Number<KPerBlock / BK1Value>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>())>;

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(AK0PerBlock, Number<MPerBlock>{}, AK1),
            make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1, AK1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(BK0PerBlock, Number<NPerBlock>{}, BK1),
            make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1, BK1, I1));
    }

    __host__ __device__ static constexpr auto
    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    // ck::Tuple<const D0DataType*, const D1DataType*, ...>
    static constexpr auto MakeDsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1, BK1);

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        // LDS allocation for C shuffle in LDS
        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

        constexpr auto c_block_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();

        return math::max((a_block_space_size_aligned + b_block_space_size_aligned) *
                             sizeof(ABDataType),
                         c_block_size * sizeof(CShuffleDataType));
    }

    // A desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeDefaultAGridDescriptor_AK0_M_AK1(const AGridDesc_M_K& a_grid_desc_m_k)
    {
        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);

        const auto AK0 = K / AK1;

        return transform_tensor_descriptor(a_grid_desc_m_k,
                                           make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                      make_pass_through_transform(M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    // B desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeDefaultBGridDescriptor_BK0_N_BK1(const BGridDesc_N_K& b_grid_desc_n_k)
    {
        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = b_grid_desc_n_k.GetLength(I1);

        const auto BK0 = K / BK1;

        return transform_tensor_descriptor(b_grid_desc_n_k,
                                           make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                      make_pass_through_transform(N)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    // E desc for destination in blockwise copy
    template <typename EGridDescriptor_M_N>
    __host__ __device__ static constexpr auto MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const EGridDescriptor_M_N& e_grid_desc_m_n)
    {
        const auto M = e_grid_desc_m_n.GetLength(I0);
        const auto N = e_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        const auto e_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            e_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return e_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // Ds desc for source in blockwise copy
    template <typename DsGridDescriptor_M_N>
    __host__ __device__ static constexpr auto
    MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const DsGridDescriptor_M_N& ds_grid_desc_m_n)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(ds_grid_desc_m_n[i]);
            },
            Number<NumDTensor>{});
    }

    template <typename GridDescriptor_M_N>
    __host__ __device__ static constexpr auto
    MakeMeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock(const GridDescriptor_M_N& grid_desc_m_n)
    {
        const auto M      = grid_desc_m_n.GetLength(I0);
        const auto NBlock = grid_desc_m_n.GetLength(I1);
        const auto MBlock = M / MPerBlock;

        const auto grid_desc_mblock_mperblock_nblock = transform_tensor_descriptor(
            grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_pass_through_transform(NBlock)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}));

        return grid_desc_mblock_mperblock_nblock;
    }

    // return block_id to E matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2ETileMap(const EGridDesc_M_N& e_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, EGridDesc_M_N>(
            e_grid_desc_m_n);
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2ETileMap>
    __host__ __device__ static constexpr bool CheckValidity(const AGridDesc_M_K& a_grid_desc_m_k,
                                                            const BGridDesc_N_K& b_grid_desc_n_k,
                                                            const DsGridDesc_M_N& ds_grid_desc_m_n,
                                                            const EGridDesc_M_N& e_grid_desc_m_n,
                                                            const Block2ETileMap& block_2_etile_map)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);

        // check consistency of desc
        if(!(M == e_grid_desc_m_n.GetLength(I0) && N == e_grid_desc_m_n.GetLength(I1)))
        {
            return false;
        }

        bool valid = true;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            valid = valid && (M == ds_grid_desc_m_n[i].GetLength(I0) &&
                              N == ds_grid_desc_m_n[i].GetLength(I1));
        });

        if(!valid)
        {
            return false;
        }

        // check tile size
        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0))
        {
            return false;
        }

        // check gridwise gemm pipeline
        const auto num_k_loop = K / KPerBlock;

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        // check block-to-E-tile
        if(!block_2_etile_map.CheckValidity(e_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        // check tensor size: cannot be larger than 2GB each
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        if(!(a_grid_desc_m_k.GetElementSpaceSize() * sizeof(ABDataType) <= TwoGB &&
             b_grid_desc_n_k.GetElementSpaceSize() * sizeof(ABDataType) <= TwoGB &&
             e_grid_desc_m_n.GetElementSpaceSize() * sizeof(EMeanVarDataType) <= TwoGB))
        {
            return false;
        }

        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    using DefaultAGridDesc_AK0_M_AK1 =
        remove_cvref_t<decltype(MakeDefaultAGridDescriptor_AK0_M_AK1(AGridDesc_M_K{}))>;
    using DefaultBGridDesc_BK0_N_BK1 =
        remove_cvref_t<decltype(MakeDefaultBGridDescriptor_BK0_N_BK1(BGridDesc_N_K{}))>;
    using EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock =
        remove_cvref_t<decltype(MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}))>;
    using MeanVarGridDescriptor_MBlock_MPerBlock_NBlock =
        remove_cvref_t<decltype(MakeMeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock(
            MeanVarGridDesc_M_NBlock{}))>;
    using CountGridDescriptor_MBlock_MPerBlock_NBlock =
        remove_cvref_t<decltype(MakeMeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock(
            CountGridDesc_M_NBlock{}))>;
    using DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock =
        remove_cvref_t<decltype(MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            DsGridDesc_M_N{}))>;

    using DefaultBlock2ETileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2ETileMap(EGridDesc_M_N{}))>;

    using DsGridPointer = decltype(MakeDsGridPointer());

    template <bool HasMainKBlockLoop,
              typename AGridDesc_AK0_M_AK1,
              typename BGridDesc_BK0_N_BK1,
              typename Block2ETileMap>
    __device__ static void
    Run(const ABDataType* __restrict__ p_a_grid,
        const ABDataType* __restrict__ p_b_grid,
        DsGridPointer p_ds_grid,
        EMeanVarDataType* __restrict__ p_e_grid,
        EMeanVarDataType* __restrict__ p_welford_mean_grid,
        EMeanVarDataType* __restrict__ p_welford_var_grid,
        int32_t* __restrict__ p_welford_count,
        void* __restrict__ p_shared,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op,
        const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
        const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
        const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
            e_grid_desc_mblock_mperblock_nblock_nperblock,
        const MeanVarGridDescriptor_MBlock_MPerBlock_NBlock&
            mean_var_grid_desc_mblock_mperblock_nblock,
        const CountGridDescriptor_MBlock_MPerBlock_NBlock& count_grid_desc_mblock_mperblock_nblock,
        const Block2ETileMap& block_2_etile_map,
        index_t NRaw)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());

        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());

        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[i],
                    ds_grid_desc_mblock_mperblock_nblock_nperblock[i].GetElementSpaceSize());
            },
            Number<NumDTensor>{});

        auto e_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e_grid, e_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        auto mean_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_welford_mean_grid, mean_var_grid_desc_mblock_mperblock_nblock.GetElementSpaceSize());

        auto var_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_welford_var_grid, mean_var_grid_desc_mblock_mperblock_nblock.GetElementSpaceSize());

        auto welford_count_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_welford_count, count_grid_desc_mblock_mperblock_nblock.GetElementSpaceSize());

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_etile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_etile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1, BK1);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0PerBlock, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                ABDataType,
                                                ABDataType,
                                                decltype(a_grid_desc_ak0_m_ak1),
                                                decltype(a_block_desc_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                ABlockTransferSrcVectorDim,
                                                2,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_AK1,
                                                1,
                                                1,
                                                AThreadTransferSrcResetCoordinateAfterRun,
                                                true,
                                                NumGemmKPrefetchStage>(
                a_grid_desc_ak0_m_ak1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<BK0PerBlock, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                ABDataType,
                                                ABDataType,
                                                decltype(b_grid_desc_bk0_n_bk1),
                                                decltype(b_block_desc_bk0_n_bk1),
                                                BBlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                BBlockTransferSrcVectorDim,
                                                2,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                BThreadTransferSrcResetCoordinateAfterRun,
                                                true,
                                                NumGemmKPrefetchStage>(
                b_grid_desc_bk0_n_bk1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check
        constexpr auto lcm_AK1_BK1 = math::lcm(AK1, BK1);
        constexpr bool is_single_rate_mfma =
            (((is_same<ABDataType, half_t>::value || is_same<ABDataType, bhalf_t>::value) &&
              lcm_AK1_BK1 <= 4) ||
             (is_same<ABDataType, int8_t>::value && lcm_AK1_BK1 <= 8) ||
             ((is_same<ABDataType, f8_t>::value || is_same<ABDataType, bf8_t>::value) &&
              lcm_AK1_BK1 < 32))
                ? true
                : false;
        constexpr auto is_scale_mfma = false;
        constexpr index_t KPack      = math::max(lcm_AK1_BK1,
                                            MfmaSelector<ABDataType,
                                                         MPerXdl,
                                                         NPerXdl,
                                                         ABDataType,
                                                         is_single_rate_mfma,
                                                         is_scale_mfma>::selected_mfma.k_per_blk);

        auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
            BlockSize,
            ABDataType,
            ABDataType,
            AccDataType,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_block_desc_bk0_n_bk1),
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            LoopSched>();

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ABDataType*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ABDataType*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1, 0, 0);

        // gridwise GEMM pipeline
        const auto gridwise_gemm_pipeline =
            GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            KPerBlock);

        gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(a_grid_desc_ak0_m_ak1,
                                                               a_block_desc_ak0_m_ak1,
                                                               a_blockwise_copy,
                                                               a_grid_buf,
                                                               a_block_buf,
                                                               a_block_slice_copy_step,
                                                               b_grid_desc_bk0_n_bk1,
                                                               b_block_desc_bk0_n_bk1,
                                                               b_blockwise_copy,
                                                               b_grid_buf,
                                                               b_block_buf,
                                                               b_block_slice_copy_step,
                                                               blockwise_gemm,
                                                               c_thread_buf,
                                                               num_k_block_main_loop);

        // shuffle C, Welford and write out
        {
            static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                              NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
            constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

            // TODO: hacky, fix it!
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            // TODO: hacky, fix it!
            // c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp is only used to get lengths
            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I4);
            constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I5);
            constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I6);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I7);

            constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<CShuffleDataType*>(p_shared),
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 = transform_tensor_descriptor(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per shuffle
                        M1,                                      // M1 = MWave
                        M2,                                      // M2 * M3 * M4 = MPerXdl
                        M3,
                        M4)),
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per shuffle
                        N1,                                      // N1 = NWave
                        N2))),                                   // N2 = NPerXdl
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4, 5, 6>{}, Sequence<>{}, Sequence<1, 3, 7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   CShuffleDataType,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   Sequence<CShuffleMXdlPerWavePerShuffle,
                                                            CShuffleNXdlPerWavePerShuffle,
                                                            I1,
                                                            I1,
                                                            M2,
                                                            I1,
                                                            M4,
                                                            I1>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     m_thread_data_on_block_idx[I3],
                                     m_thread_data_on_block_idx[I4],
                                     n_thread_data_on_block_idx[I2]),
                    ck::tensor_operation::element_wise::PassThrough{}};

            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c_vgpr =
                SpaceFillingCurve<Sequence<MXdlPerWave, NXdlPerWave, 1, 1, M2, 1, M4, 1>,
                                  Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                  Sequence<CShuffleMXdlPerWavePerShuffle,
                                           CShuffleNXdlPerWavePerShuffle,
                                           1,
                                           1,
                                           M2,
                                           1,
                                           M4,
                                           1>,
                                  false>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_der_global =
                SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                           1,
                                           CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>,
                                  false>{};

            // LDS c_shuffle_block_desc_mperblock_nperblock
            constexpr auto c_shuffle_block_desc_mperblock_nperblock = transform_tensor_descriptor(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0),
                    make_pass_through_transform(
                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetLength(I1)),
                    make_freeze_transform(I0),
                    make_pass_through_transform(
                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetLength(I3))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<>{}, Sequence<1>{}));

            static_assert(PostShuffleThreadClusterSize_M_N::At(I0) *
                                  PostShuffleThreadClusterSize_M_N::At(I1) ==
                              BlockSize,
                          "wrong!");

            static_assert((CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl) %
                                      PostShuffleThreadClusterSize_M_N::At(I0) ==
                                  0 &&
                              (CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl) %
                                      PostShuffleThreadClusterSize_M_N::At(I1) ==
                                  0,
                          "wrong!");

            constexpr index_t PostShuffleThreadSliceSize_M =
                (CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl) /
                PostShuffleThreadClusterSize_M_N::At(I0);

            constexpr index_t PostShuffleThreadSliceSize_N =
                (CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl) /
                PostShuffleThreadClusterSize_M_N::At(I1);

            constexpr auto PostShuffleThreadSliceSize_M_N =
                Sequence<PostShuffleThreadSliceSize_M, PostShuffleThreadSliceSize_N>{};

            // VGPR post_shuffle_thread_desc_m_n
            constexpr auto post_shuffle_thread_desc_m_n = make_naive_tensor_descriptor_packed(
                make_tuple(Number<PostShuffleThreadSliceSize_M>{},
                           Number<PostShuffleThreadSliceSize_N>{}));

            auto e_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                post_shuffle_thread_desc_m_n.GetElementSpaceSize());

            // To apply D0, D1, ... and Welford.
            // threadwise copy from LDS to VGPR
            constexpr auto post_shuffle_thread_cluster_desc =
                make_cluster_descriptor(PostShuffleThreadClusterSize_M_N{}, Sequence<0, 1>{});

            const auto post_shuffle_thread_cluster_idx =
                post_shuffle_thread_cluster_desc.CalculateBottomIndex(
                    make_multi_index(get_thread_local_1d_id()));

            const auto post_shuffle_thread_data_idx_begin =
                post_shuffle_thread_cluster_idx * PostShuffleThreadSliceSize_M_N;

            // To apply D0, D1, ... and Welford.
            // Copy c shuffle from LDS back to VGPR
            auto post_shuffle_thread_copy_lds_to_vgpr =
                ThreadwiseTensorSliceTransfer_v2<CShuffleDataType,
                                                 AccDataType,
                                                 decltype(c_shuffle_block_desc_mperblock_nperblock),
                                                 decltype(post_shuffle_thread_desc_m_n),
                                                 decltype(PostShuffleThreadSliceSize_M_N),
                                                 Sequence<0, 1>,
                                                 1,
                                                 PostShuffleScalarPerVector,
                                                 1,
                                                 true>{c_shuffle_block_desc_mperblock_nperblock,
                                                       post_shuffle_thread_data_idx_begin};

            // D0, D1, ..., Dn
            constexpr auto post_shuffle_thread_desc_I1_mperblock_I1_nperblock =
                make_naive_tensor_descriptor_packed(
                    make_tuple(I1,
                               Number<PostShuffleThreadSliceSize_M>{},
                               I1,
                               Number<PostShuffleThreadSliceSize_N>{}));

            // FIXME: Decrease usage of VGPR
            // Apply pointwise lambda function from multi-source (Global and LDS) into VGPR
            auto ds_thread_buf = generate_tuple(
                [&](auto) {
                    return make_static_buffer<AddressSpaceEnum::Vgpr, CShuffleDataType>(
                        post_shuffle_thread_desc_I1_mperblock_I1_nperblock.GetElementSpaceSize());
                },
                Number<NumDTensor>{});

            // Copy D0, D1, ..., Dn from global to VGPR
            auto ds_thread_copy_global_to_vgpr = generate_tuple(
                [&](auto I) {
                    using DDataType = remove_cvref_t<tuple_element_t<I.value, DsDataType>>;
                    return ThreadwiseTensorSliceTransfer_v2<
                        DDataType,
                        AccDataType,
                        decltype(ds_grid_desc_mblock_mperblock_nblock_nperblock[I]),
                        decltype(post_shuffle_thread_desc_I1_mperblock_I1_nperblock),
                        Sequence<I1,
                                 PostShuffleThreadSliceSize_M,
                                 I1,
                                 PostShuffleThreadSliceSize_N>,
                        Sequence<0, 1, 2, 3>,
                        3,
                        PostShuffleScalarPerVector,
                        1,
                        true>(
                        ds_grid_desc_mblock_mperblock_nblock_nperblock[I],
                        make_multi_index(
                            I0,
                            m_block_data_idx_on_grid + post_shuffle_thread_data_idx_begin[I0],
                            I0,
                            n_block_data_idx_on_grid + post_shuffle_thread_data_idx_begin[I1]));
                },
                Number<NumDTensor>{});

            auto e_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
                AccDataType,
                EMeanVarDataType,
                decltype(post_shuffle_thread_desc_I1_mperblock_I1_nperblock),
                decltype(e_grid_desc_mblock_mperblock_nblock_nperblock),
                tensor_operation::element_wise::PassThrough,
                Sequence<I1,
                         PostShuffleThreadSliceSize_M,
                         I1,
                         PostShuffleThreadSliceSize_N>, // SliceLengths
                Sequence<0, 1, 2, 3>,                   // DimAccessOrder
                3,                                      // DstVectorDim
                PostShuffleScalarPerVector,
                InMemoryDataOperationEnum::Set,
                1,
                true>{
                e_grid_desc_mblock_mperblock_nblock_nperblock,
                make_multi_index(I0,
                                 m_block_data_idx_on_grid + post_shuffle_thread_data_idx_begin[I0],
                                 I0,
                                 n_block_data_idx_on_grid + post_shuffle_thread_data_idx_begin[I1]),
                tensor_operation::element_wise::PassThrough{}};

            // Welford
            constexpr auto thread_welford_src_desc_m_k = make_naive_tensor_descriptor_packed(
                make_tuple(Number<PostShuffleThreadSliceSize_M>{},
                           Number<PostShuffleThreadSliceSize_N>{}));

            constexpr auto thread_welford_dst_desc_m = make_naive_tensor_descriptor_packed(
                make_tuple(Number<PostShuffleThreadSliceSize_M>{}));

            using ThreadwiseWelford = ThreadwiseWelford<AccDataType,
                                                        decltype(thread_welford_src_desc_m_k),
                                                        decltype(thread_welford_dst_desc_m)>;

            using BlockwiseWelford = BlockwiseWelford<AccDataType,
                                                      BlockSize,
                                                      PostShuffleThreadClusterSize_M_N,
                                                      Sequence<0, 1>,
                                                      false>;

            constexpr int num_shuffleM =
                MPerBlock / (CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl);

            constexpr int num_shuffleN =
                NPerBlock / (CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl);

            using mean_var_vgpr_type =
                decltype(make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                    thread_welford_dst_desc_m.GetElementSpaceSize()));

            using welford_count_vgpr_type =
                decltype(make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
                    thread_welford_dst_desc_m.GetElementSpaceSize()));

            Array<ThreadwiseWelford, num_shuffleM> threadwise_welfords;
            Array<mean_var_vgpr_type, num_shuffleM> mean_thread_bufs;
            Array<mean_var_vgpr_type, num_shuffleM> var_thread_bufs;
            Array<welford_count_vgpr_type, num_shuffleM> welford_count_thread_bufs;

            int max_count     = PostShuffleThreadSliceSize_N * num_shuffleN;
            const auto nblock = mean_var_grid_desc_mblock_mperblock_nblock.GetLength(I2);

            // tail block
            if(block_work_idx[I1] % nblock == nblock - 1)
            {
                constexpr index_t NPerShuffleBlock =
                    CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl;

                int NPerBlockTail = NRaw - NPerBlock * (nblock - 1);
                int thread_max_len =
                    PostShuffleThreadSliceSize_N * (post_shuffle_thread_cluster_idx[I1] + 1);
                int shuffle_step = 0;
                while(thread_max_len <= NPerBlockTail && shuffle_step < num_shuffleN)
                {
                    ++shuffle_step;
                    thread_max_len += NPerShuffleBlock;
                }

                int delta = 0;
                if(thread_max_len - NPerBlockTail > PostShuffleThreadSliceSize_N)
                    delta = 0;
                else if(NPerBlockTail > thread_max_len)
                    delta = PostShuffleThreadSliceSize_N;
                else
                    delta = PostShuffleThreadSliceSize_N - thread_max_len + NPerBlockTail;

                max_count = shuffle_step * PostShuffleThreadSliceSize_N + delta;
            }

            static_for<0, num_shuffleM, 1>{}([&](auto i) {
                threadwise_welfords(i).max_count_ = max_count;
                mean_thread_bufs(i) = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                    thread_welford_dst_desc_m.GetElementSpaceSize());

                var_thread_bufs(i) = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                    thread_welford_dst_desc_m.GetElementSpaceSize());

                welford_count_thread_bufs(i) = make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
                    thread_welford_dst_desc_m.GetElementSpaceSize());

                static_for<0, PostShuffleThreadSliceSize_M, 1>{}([&](auto j) {
                    mean_thread_bufs(i)(j)          = type_convert<AccDataType>(0.0f);
                    var_thread_bufs(i)(j)           = type_convert<AccDataType>(0.0f);
                    welford_count_thread_bufs(i)(j) = 0;
                });
            });

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_der_global.GetNumOfAccess(), "wrong!");

            int shuffleM_index = __builtin_amdgcn_readfirstlane(0);
            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to read from LDS
                block_sync_lds();

                // each thread shuffle data from VGPR to LDS
                c_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                              sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                              c_thread_buf,
                                              c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                              c_shuffle_block_buf);

                // make sure it's safe to write to LDS
                block_sync_lds();

                // Get shuffle data from LDS to VGPR
                post_shuffle_thread_copy_lds_to_vgpr.Run(c_shuffle_block_desc_mperblock_nperblock,
                                                         c_shuffle_block_buf,
                                                         post_shuffle_thread_desc_m_n,
                                                         make_tuple(I0, I0),
                                                         e_thread_buf);

                // Global read D0, D1, ...
                static_for<0, NumDTensor, 1>{}([&](auto Id) {
                    auto& d_thread_copy_global_to_vgpr = ds_thread_copy_global_to_vgpr(Id);
                    d_thread_copy_global_to_vgpr.Run(
                        ds_grid_desc_mblock_mperblock_nblock_nperblock[Id],
                        ds_grid_buf[Id],
                        post_shuffle_thread_desc_I1_mperblock_I1_nperblock,
                        make_tuple(I0, I0, I0, I0),
                        ds_thread_buf(Id));

                    if constexpr(access_id < num_access - 1)
                    {
                        // move on D0, D1, ...
                        constexpr auto de_global_step = sfc_der_global.GetForwardStep(access_id);
                        d_thread_copy_global_to_vgpr.MoveSrcSliceWindow(
                            ds_grid_desc_mblock_mperblock_nblock_nperblock[Id], de_global_step);
                    }
                });

                // cde_element_op(e, c, d0, d1, ...);
                static_for<0, post_shuffle_thread_desc_m_n.GetElementSize(), 1>{}([&](auto i) {
                    const auto c_ds_src_data_refs = concat_tuple_of_reference(
                        tie(e_thread_buf[i]),
                        generate_tie(
                            [&](auto Id) -> const auto& { return ds_thread_buf[Id][i]; },
                            Number<NumDTensor>{}));
                    auto e_dst_data_refs = tie(e_thread_buf(i));
                    unpack2(cde_element_op, e_dst_data_refs, c_ds_src_data_refs);
                });

                // Global write E
                e_thread_copy_vgpr_to_global.Run(post_shuffle_thread_desc_I1_mperblock_I1_nperblock,
                                                 make_tuple(I0, I0, I0, I0),
                                                 e_thread_buf,
                                                 e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                 e_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    // move on E
                    constexpr auto de_global_step = sfc_der_global.GetForwardStep(access_id);
                    e_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                        e_grid_desc_mblock_mperblock_nblock_nperblock, de_global_step);
                }

                // Threadwise welford
                auto& threadwise_welford = threadwise_welfords(shuffleM_index);
                auto& mean_thread_buf    = mean_thread_bufs(shuffleM_index);
                auto& var_thread_buf     = var_thread_bufs(shuffleM_index);

                threadwise_welford.Run(e_thread_buf, mean_thread_buf, var_thread_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto de_global_step = sfc_der_global.GetForwardStep(access_id);
                    constexpr int shuffleMInc =
                        de_global_step[I1] /
                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetLength(I1);
                    shuffleM_index = __builtin_amdgcn_readfirstlane(shuffleM_index + shuffleMInc);
                }
            }); // copy c, d, e + welford

            // Blockwise welford and write out
            static_for<0, num_shuffleM, 1>{}([&](auto i) {
                auto& mean_thread_buf  = mean_thread_bufs(i);
                auto& var_thread_buf   = var_thread_bufs(i);
                auto& count_thread_buf = welford_count_thread_bufs(i);

                static_for<0, PostShuffleThreadSliceSize_M, 1>{}([&](auto j) {
                    block_sync_lds();
                    count_thread_buf(j) = threadwise_welfords(i).cur_count_;
                    BlockwiseWelford::Run(
                        mean_thread_buf(j), var_thread_buf(j), count_thread_buf(j));
                });

                if(post_shuffle_thread_cluster_idx[I1] == 0)
                {
                    constexpr auto thread_welford_desc_I_m_I = make_naive_tensor_descriptor_packed(
                        make_tuple(I1, Number<PostShuffleThreadSliceSize_M>{}, I1));

                    constexpr int shuffleMPerBlock =
                        c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetLength(I1);

                    auto mean_var_count_thread_copy_index = make_multi_index(
                        block_work_idx[I0],                                            // mblock
                        shuffleMPerBlock * i + post_shuffle_thread_data_idx_begin[I0], // mperblock
                        block_work_idx[I1]);                                           // nblock

                    auto mean_var_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
                        AccDataType,
                        EMeanVarDataType,
                        decltype(thread_welford_desc_I_m_I),
                        decltype(mean_var_grid_desc_mblock_mperblock_nblock),
                        tensor_operation::element_wise::PassThrough,
                        Sequence<1, PostShuffleThreadSliceSize_M, 1>,
                        Sequence<0, 1, 2>,
                        1,
                        1,
                        InMemoryDataOperationEnum::Set,
                        1,
                        true>{mean_var_grid_desc_mblock_mperblock_nblock,
                              mean_var_count_thread_copy_index,
                              tensor_operation::element_wise::PassThrough{}};

                    mean_var_thread_copy_vgpr_to_global.Run(
                        thread_welford_desc_I_m_I,
                        make_tuple(I0, I0, I0),
                        mean_thread_buf,
                        mean_var_grid_desc_mblock_mperblock_nblock,
                        mean_grid_buf); // write mean

                    mean_var_thread_copy_vgpr_to_global.Run(
                        thread_welford_desc_I_m_I,
                        make_tuple(I0, I0, I0),
                        var_thread_buf,
                        mean_var_grid_desc_mblock_mperblock_nblock,
                        var_grid_buf); // write variance

                    // Stride of count is [0, 1]. Only the first row in count[0, 0:nblock] need
                    // to be written.
                    if(i == 0 && block_work_idx[I0] == 0 &&
                       post_shuffle_thread_cluster_idx[I0] == 0)
                    {
                        auto count_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
                            int32_t,
                            int32_t,
                            decltype(thread_welford_desc_I_m_I),
                            decltype(count_grid_desc_mblock_mperblock_nblock),
                            tensor_operation::element_wise::PassThrough,
                            Sequence<1, PostShuffleThreadSliceSize_M, 1>,
                            Sequence<0, 1, 2>,
                            1,
                            1,
                            InMemoryDataOperationEnum::Set,
                            1,
                            false>{count_grid_desc_mblock_mperblock_nblock,
                                   mean_var_count_thread_copy_index,
                                   tensor_operation::element_wise::PassThrough{}};

                        count_thread_copy_vgpr_to_global.Run(
                            thread_welford_desc_I_m_I,
                            make_tuple(I0, I0, I0),
                            count_thread_buf,
                            count_grid_desc_mblock_mperblock_nblock,
                            welford_count_grid_buf); // write count
                    }
                }
            });

        } // shuffle C + Ds + welford + write out
    }     // run
};

} // namespace ck
