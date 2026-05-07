// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_wmma.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_async.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AGridDesc,
          typename BGridDesc,
          typename CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_wmma_gfx13(const ADataType* __restrict__ p_a_grid,
                           const BDataType* __restrict__ p_b_grid,
                           CDataType* __restrict__ p_c_grid,
                           const AGridDesc a_grid_desc,
                           const BGridDesc b_grid_desc,
                           const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                               c_grid_desc_mblock_mperblock_nblock_nperblock,
                           const AElementwiseOperation a_element_op,
                           const BElementwiseOperation b_element_op,
                           const CElementwiseOperation c_element_op,
                           const Block2CTileMap block_2_ctile_map)
{
#if defined(__gfx13__)
    __shared__ char p_shared[GridwiseGemm::SharedMemTrait::lds_size];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                  p_b_grid,
                                                  p_c_grid,
                                                  p_shared,
                                                  nullptr,
                                                  a_grid_desc,
                                                  b_grid_desc,
                                                  c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = a_grid_desc;
    ignore = b_grid_desc;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = block_2_ctile_map;
#endif
}

template <typename GridwiseGemm,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AGridDesc,
          typename BGridDesc,
          typename CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
#if defined(__gfx13__)
__exp_amd_wavegroup_kernel(4, 32, 256, 1, 1)
#endif
    __global__
    void kernel_gemm_wmma_wavegroup(const ADataType* __restrict__ p_a_grid,
                                    const BDataType* __restrict__ p_b_grid,
                                    CDataType* __restrict__ p_c_grid,
                                    const AGridDesc a_grid_desc,
                                    const BGridDesc b_grid_desc,
                                    const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                                        c_grid_desc_mblock_mperblock_nblock_nperblock,
                                    const AElementwiseOperation a_element_op,
                                    const BElementwiseOperation b_element_op,
                                    const CElementwiseOperation c_element_op,
                                    const Block2CTileMap block_2_ctile_map)
{
#if defined(__gfx13__)
    __shared__ char p_shared[GridwiseGemm::SharedMemTrait::lds_size];
    static constexpr index_t lane_shared_size =
        math::max(GridwiseGemm::LaneSharedMemTrait::lane_shared_size, 4);
    static __exp_amd_laneshared__ char p_lane_shared[lane_shared_size];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                  p_b_grid,
                                                  p_c_grid,
                                                  p_shared,
                                                  p_lane_shared,
                                                  a_grid_desc,
                                                  b_grid_desc,
                                                  c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = a_grid_desc;
    ignore = b_grid_desc;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = block_2_ctile_map;
#endif
}

template <typename GridwiseGemm,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AGridDesc,
          typename BGridDesc,
          typename AScaleGridDesc,
          typename BScaleGridDesc,
          typename CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_mx_wmma(const int32_t* __restrict__ p_a_grid,
                        const int32_t* __restrict__ p_b_grid,
                        const int32_t* __restrict__ p_a_scale,
                        const int32_t* __restrict__ p_b_scale,
                        const AScaleGridDesc a_scale_grid_desc,
                        const BScaleGridDesc b_scale_grid_desc,
                        CDataType* __restrict__ p_c_grid,
                        const AGridDesc a_grid_desc,
                        const BGridDesc b_grid_desc,
                        const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                            c_grid_desc_mblock_mperblock_nblock_nperblock,
                        const AElementwiseOperation a_element_op,
                        const BElementwiseOperation b_element_op,
                        const CElementwiseOperation c_element_op,
                        const Block2CTileMap block_2_ctile_map)
{
#if defined(__gfx13__)
    __shared__ char p_shared[GridwiseGemm::SharedMemTrait::lds_size];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                  p_b_grid,
                                                  p_a_scale,
                                                  p_b_scale,
                                                  p_c_grid,
                                                  p_shared,
                                                  nullptr,
                                                  a_grid_desc,
                                                  b_grid_desc,
                                                  a_scale_grid_desc,
                                                  b_scale_grid_desc,
                                                  c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_a_scale;
    ignore = p_b_scale;
    ignore = p_c_grid;
    ignore = a_grid_desc;
    ignore = b_grid_desc;
    ignore = a_scale_grid_desc;
    ignore = b_scale_grid_desc;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = block_2_ctile_map;
#endif // end of if (defined(__gfx13__))
}

template <typename GridwiseGemm,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AGridDesc,
          typename BGridDesc,
          typename AScaleGridDesc,
          typename BScaleGridDesc,
          typename CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
#if defined(__gfx13__)
__exp_amd_wavegroup_kernel(4, 32, 256, 1, 1)
#endif
    __global__
    void kernel_gemm_mx_wmma_wavegroup(const int32_t* __restrict__ p_a_grid,
                                       const int32_t* __restrict__ p_b_grid,
                                       const int32_t* __restrict__ p_a_scale,
                                       const int32_t* __restrict__ p_b_scale,
                                       const AScaleGridDesc a_scale_grid_desc,
                                       const BScaleGridDesc b_scale_grid_desc,
                                       CDataType* __restrict__ p_c_grid,
                                       const AGridDesc a_grid_desc,
                                       const BGridDesc b_grid_desc,
                                       const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                                           c_grid_desc_mblock_mperblock_nblock_nperblock,
                                       const AElementwiseOperation a_element_op,
                                       const BElementwiseOperation b_element_op,
                                       const CElementwiseOperation c_element_op,
                                       const Block2CTileMap block_2_ctile_map)
{
#if defined(__gfx13__)
    __shared__ char p_shared[GridwiseGemm::SharedMemTrait::lds_size];
    static constexpr index_t lane_shared_size =
        math::max(GridwiseGemm::LaneSharedMemTrait::lane_shared_size, 4);
    static __exp_amd_laneshared__ char p_lane_shared[lane_shared_size];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                  p_b_grid,
                                                  p_a_scale,
                                                  p_b_scale,
                                                  p_c_grid,
                                                  p_shared,
                                                  p_lane_shared,
                                                  a_grid_desc,
                                                  b_grid_desc,
                                                  a_scale_grid_desc,
                                                  b_scale_grid_desc,
                                                  c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_a_scale;
    ignore = p_b_scale;
    ignore = p_c_grid;
    ignore = a_grid_desc;
    ignore = b_grid_desc;
    ignore = a_scale_grid_desc;
    ignore = b_scale_grid_desc;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = block_2_ctile_map;
#endif // end of if (defined(__gfx13__))
}

template <typename GridwiseGemm,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AGridDesc,
          typename BGridDesc,
          typename CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          int AClusterSize,
          int BClusterSize,
          bool HasMainKBlockLoop>
#if defined(__gfx13__)
__cluster_dims__(AClusterSize* BClusterSize, 1, 1)
#endif
    __global__ void kernel_gemm_wmma_cluster(const ADataType* __restrict__ p_a_grid,
                                             const BDataType* __restrict__ p_b_grid,
                                             CDataType* __restrict__ p_c_grid,
                                             const AGridDesc a_grid_desc,
                                             const BGridDesc b_grid_desc,
                                             const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                                                 c_grid_desc_mblock_mperblock_nblock_nperblock,
                                             const AElementwiseOperation a_element_op,
                                             const BElementwiseOperation b_element_op,
                                             const CElementwiseOperation c_element_op,
                                             const Block2CTileMap block_2_ctile_map)
{
#if defined(__gfx13__)
    __shared__ char p_shared[GridwiseGemm::SharedMemTrait::lds_size];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                  p_b_grid,
                                                  p_c_grid,
                                                  p_shared,
                                                  nullptr,
                                                  a_grid_desc,
                                                  b_grid_desc,
                                                  c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = a_grid_desc;
    ignore = b_grid_desc;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = block_2_ctile_map;
#endif
}

template <index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename CDataType,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGridDesc,
          typename BGridDesc,
#ifdef CK_EXTENSION_MX_TYPE
          typename AScaleGridDesc,
          typename BScaleGridDesc,
#endif
          typename CGridDesc_M_N,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t KPerWmma,
          index_t K1Value,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_M_K0_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          bool AEnableLds,
          bool ABlockLdsExtraM,
          bool AEnableAsyncCopy,
          bool AEnableTRLoadFromGlobal,
          bool AEnableTiledLoadFromGlobal,
          TensorLoadOption ALoadOption, /* 0x1: ClusterMulticastLoad, 0x2: WGPMulticastLoad,
                                                0x3:DDS */
          index_t AClusterSize,         /* Set when ALoadOption == 1 or 3*/
          typename BBlockTransferThreadClusterLengths_N_K0_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          bool BEnableLds,
          bool BBlockLdsExtraN,
          bool BEnableAsyncCopy,
          bool BEnableTRLoadFromGlobal,
          bool BEnableTiledLoadFromGlobal,
          TensorLoadOption BLoadOption, /* 0x1: ClusterMulticastLoad, 0x2: WGPMulticastLoad */
          index_t BClusterSize,         /* Set when BLoadOption == 1 or 3*/
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          bool CStoreEnableAsync,
          bool EnableWaveGroup,
          index_t NumGemmKPrefetchStage = 1,
          LoopScheduler LoopSched       = make_default_loop_scheduler(),
          PipelineVersion PipelineVer   = PipelineVersion::v5>
struct GridwiseGemm_Wmma_GFX13
{
    static_assert((AEnableLds & (AEnableTRLoadFromGlobal | AEnableTiledLoadFromGlobal)) == false,
                  "these two options will not be enabled at the same time");
    static_assert((BEnableLds & (BEnableTRLoadFromGlobal | BEnableTiledLoadFromGlobal)) == false,
                  "these two options will not be enabled at the same time");
    static_assert((AEnableTRLoadFromGlobal & AEnableTiledLoadFromGlobal) == false,
                  "these two options will not be enabled at the same time");
    static_assert((BEnableTRLoadFromGlobal & BEnableTiledLoadFromGlobal) == false,
                  "these two options will not be enabled at the same time");
    static_assert((!AEnableLds & (ALoadOption == TensorLoadOption::DS_TILED_LOAD)) == false,
                  "these two options will not be enabled at the same time");
    static_assert((!BEnableLds & (BLoadOption == TensorLoadOption::DS_TILED_LOAD)) == false,
                  "these two options will not be enabled at the same time");

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // This is used to layout for A and B when implementing WMMA
    static constexpr auto K1 = Number<K1Value>{};
    // the below is used for vec load for A and B from global memory; this variable is used to
    // guarantee the load is aligned to 16 bytes; where can use b128 load as much as possible
    static constexpr auto AVecAccessNumber = Number<ABlockTransferSrcScalarPerVector>{};
    static constexpr auto BVecAccessNumber = Number<BBlockTransferSrcScalarPerVector>{};

    static constexpr auto MWaves = MPerBlock / (MRepeat * MPerWmma);
    static constexpr auto NWaves = NPerBlock / (NRepeat * NPerWmma);
    static constexpr auto WmmaK  = KPerWmma;

    // for mx data type; we need to calculate how many int32 need to load for A and B
    // and because of mx data type A, B can be different, so we need to calculate separately
    // K1 should be equal to 1, double check
    static constexpr auto mx_type_enable = is_mx_type_t_v<ADataType> || is_mx_type_t_v<BDataType>;

    static constexpr auto AKPerWmma = []() {
        if constexpr(mx_type_enable)
        {
            return KPerWmma * ADataType::BITS / 32;
        }
        else
        {
            return KPerWmma;
        }
    }();
    static constexpr auto BKPerWmma = []() {
        if constexpr(mx_type_enable)
        {
            return KPerWmma * BDataType::BITS / 32;
        }
        else
        {
            return KPerWmma;
        }
    }();
    static constexpr auto AKPerBlock = []() {
        if constexpr(mx_type_enable)
        {
            return KPerBlock * ADataType::BITS / 32;
        }
        else
        {
            return KPerBlock;
        }
    }();
    static constexpr auto BKPerBlock = []() {
        if constexpr(mx_type_enable)
        {
            return KPerBlock * BDataType::BITS / 32;
        }
        else
        {
            return KPerBlock;
        }
    }();

    static constexpr index_t WaveSize     = 32;
    static constexpr index_t NumWaveGroup = EnableWaveGroup ? 4 : 0;
    using ThisThreadBlockGrid =
        typename std::conditional<EnableWaveGroup,
                                  ThisThreadBlockWaveGroup<BlockSize, WaveSize, NumWaveGroup>,
                                  ThisThreadBlock<BlockSize>>::type;

    using GridwiseGemmPipe =
        remove_cvref_t<decltype(GridwiseGemmPipeline_Selector<PipelineVer,
                                                              NumGemmKPrefetchStage,
                                                              LoopSched,
                                                              AEnableLds,
                                                              BEnableLds,
                                                              EnableWaveGroup,
                                                              ALoadOption,
                                                              BLoadOption>())>;
#ifdef CK_EXTENSION_MX_TYPE
    __host__ __device__ static constexpr auto MakeAScaleBlockDescriptor()
    {
        // if constexpr(AEnableLds)
        //{
        //  for KPerBlock, 32 elements in K will use one uint8 scale value
        //  128 elements in K will use uint32 scale value; since each load will fetch 8 scale
        //  values, these 8 scale values will be used in 256 consecutive elements
        constexpr auto ScaleK0PerBlock = math::integer_divide_ceil(KPerBlock, 256);
        constexpr auto ScaleKPerBlock  = ScaleK0PerBlock * 2;

        if constexpr(AEnableLds)
        {
            return make_naive_tensor_descriptor(
                make_tuple(Number<MPerBlock>{}, Number<ScaleKPerBlock>{}),
                make_tuple(Number<ScaleKPerBlock>{}, I1));
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(Number<ScaleK0PerBlock>{}, Number<MRepeat>{}, I1, I1, I1),
                make_tuple(Number<MRepeat>{}, I1, I1, I1, I1));
        }
    }

    __host__ __device__ static constexpr auto MakeBScaleBlockDescriptor()
    {
        constexpr auto ScaleK0PerBlock = math::integer_divide_ceil(KPerBlock, 256);
        constexpr auto ScaleKPerBlock  = ScaleK0PerBlock * 2;

        if constexpr(BEnableLds)
        {
            return make_naive_tensor_descriptor(
                make_tuple(Number<NPerBlock>{}, Number<ScaleKPerBlock>{}),
                make_tuple(Number<ScaleKPerBlock>{}, I1));
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(Number<ScaleK0PerBlock>{}, Number<NRepeat>{}, I1, I1, I1),
                make_tuple(Number<NRepeat>{}, I1, I1, I1, I1));
        }
    }
#endif

    // Describe how data store to (LDS/VGPR) buffer from Global memory
    __host__ __device__ static constexpr auto MakeABlockDescriptor()
    {
        constexpr auto a_block_desc = [&]() {
            if constexpr(AEnableLds)
            {
                constexpr auto K0PerBlock = AKPerBlock / ABlockTransferSrcScalarPerVector;
                // WARNING: gfx13 has changed to M->K0->K1{ABlockTransferSrcScalarPerVector} Per
                // Block; which means first go K dimension, then M dimension
                if constexpr(ABlockLdsExtraM)
                {
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<MPerBlock>{}, Number<K0PerBlock>{}, AVecAccessNumber),
                        make_tuple(
                            Number<K0PerBlock + 1>{} * AVecAccessNumber, AVecAccessNumber, I1));
                }
                else
                {
                    // make_naive_tensor_descriptor_aligned has some compiling errors.
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<MPerBlock>{}, Number<K0PerBlock>{}, AVecAccessNumber),
                        make_tuple(Number<K0PerBlock>{} * AVecAccessNumber, AVecAccessNumber, I1));
                }
            }
            else
            {
                if constexpr(AEnableTRLoadFromGlobal)
                {
                    constexpr auto KWmmaPerblock = AKPerBlock / AKPerWmma;
                    constexpr auto A_KRow        = I2;
                    constexpr auto K1PerWmma     = AKPerWmma / A_KRow;
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<KWmmaPerblock>{},
                                   I1,
                                   Number<MRepeat>{},
                                   I1,
                                   I1,
                                   I1,
                                   I1,
                                   Number<K1PerWmma>{}),
                        make_tuple(Number<K1PerWmma>{},
                                   Number<K1PerWmma>{},
                                   Number<K1PerWmma>{} * Number<KWmmaPerblock>{},
                                   Number<K1PerWmma>{},
                                   Number<K1PerWmma>{},
                                   Number<K1PerWmma>{},
                                   Number<K1PerWmma>{},
                                   I1));
                }
                else
                {
                    // TODO: the logic has not changed for gfx13; need to check
                    constexpr auto A_KRow        = I2;
                    constexpr auto KWmmaPerblock = AKPerBlock / AKPerWmma;
                    constexpr auto K0PerWmma     = AKPerWmma / A_KRow / K1;
                    // KWmma->MRepeat->MWave->K0PerWmma->KRow->MPerWmma->K1 Per Thread
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<KWmmaPerblock>{},
                                   I1,
                                   Number<MRepeat>{},
                                   I1,
                                   Number<K0PerWmma>{},
                                   I1,
                                   I1,
                                   K1),
                        make_tuple(Number<K0PerWmma>{} * K1,
                                   K1,
                                   Number<KWmmaPerblock>{} * Number<K0PerWmma>{} * K1,
                                   K1,
                                   K1,
                                   K1,
                                   K1,
                                   I1));
                }
            }
        }();

        return a_block_desc;
    }

    __host__ __device__ static constexpr auto MakeBBlockDescriptor()
    {
        constexpr auto b_block_desc = [&]() {
            if constexpr(BEnableLds)
            {
                // WARNING: gfx13 has changed to N->K0->K1 Per Block
                constexpr auto K0PerBlock = BKPerBlock / BBlockTransferSrcScalarPerVector;
                if constexpr(BBlockLdsExtraN)
                {
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<NPerBlock>{}, Number<K0PerBlock>{}, BVecAccessNumber),
                        make_tuple(
                            Number<K0PerBlock + 1>{} * BVecAccessNumber, BVecAccessNumber, I1));
                }
                else
                {
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<NPerBlock>{}, Number<K0PerBlock>{}, BVecAccessNumber),
                        make_tuple(Number<K0PerBlock>{} * BVecAccessNumber, BVecAccessNumber, I1));
                }
            }
            else
            {
                if constexpr(BEnableTRLoadFromGlobal)
                {
                    constexpr auto KWmmaPerblock = BKPerBlock / BKPerWmma;
                    constexpr auto B_KRow        = I2;
                    constexpr auto K1PerWmma     = BKPerWmma / B_KRow;
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<KWmmaPerblock>{},
                                   I1,
                                   Number<NRepeat>{},
                                   I1,
                                   I1,
                                   I1,
                                   I1,
                                   Number<K1PerWmma>{}),
                        make_tuple(Number<K1PerWmma>{},
                                   Number<K1PerWmma>{},
                                   Number<K1PerWmma>{} * Number<KWmmaPerblock>{},
                                   Number<K1PerWmma>{},
                                   Number<K1PerWmma>{},
                                   Number<K1PerWmma>{},
                                   Number<K1PerWmma>{},
                                   I1));
                }
                else
                {
                    // TODO: the logic has not changed for gfx13
                    constexpr auto B_KRow        = I2;
                    constexpr auto KWmmaPerblock = BKPerBlock / BKPerWmma;
                    constexpr auto K0PerWmma     = BKPerWmma / B_KRow / K1;
                    // KWmma->NRepeat->MWave->K0PerWmma->KRow->MPerWmma->K1 Per Thread
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<KWmmaPerblock>{},
                                   I1,
                                   Number<NRepeat>{},
                                   I1,
                                   Number<K0PerWmma>{},
                                   I1,
                                   I1,
                                   K1),
                        make_tuple(Number<K0PerWmma>{} * K1,
                                   K1,
                                   Number<KWmmaPerblock>{} * Number<K0PerWmma>{} * K1,
                                   K1,
                                   K1,
                                   K1,
                                   K1,
                                   I1));
                }
            }
        }();

        return b_block_desc;
    }

    // Describe how data load from LDS to VGPR
    __host__ __device__ static constexpr auto MakeABlockForWMMADescriptor()
    {
        constexpr auto a_block_desc = [&]() {
            if constexpr(AEnableLds)
            {
                constexpr auto K0PerBlock = AKPerBlock / K1Value;
                // WARNING: gfx13 has changed to M->K0->K1{ABlockTransferSrcScalarPerVector} Per
                // Block; which means first go K dimension, then M dimension
                if constexpr(ABlockLdsExtraM)
                {
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<MPerBlock>{}, Number<K0PerBlock>{}, K1),
                        make_tuple(Number<K0PerBlock + 1>{} * K1, K1, I1));
                }
                else
                {
                    // make_naive_tensor_descriptor_aligned has some compiling errors.
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<MPerBlock>{}, Number<K0PerBlock>{}, K1),
                        make_tuple(Number<K0PerBlock>{} * K1, K1, I1));
                }
            }
            else
            {
                constexpr auto A_KRow        = I2;
                constexpr auto KWmmaPerblock = AKPerBlock / AKPerWmma;
                constexpr auto K0PerWmma     = AKPerWmma / A_KRow / K1;
                // KWmma->MRepeat->MWave->K0PerWmma->KRow->MPerWmma->K1 Per Thread
                return make_naive_tensor_descriptor(
                    make_tuple(Number<KWmmaPerblock>{},
                               I1,
                               Number<MRepeat>{},
                               I1,
                               Number<K0PerWmma>{},
                               I1,
                               I1,
                               K1),
                    make_tuple(Number<K0PerWmma>{} * K1,
                               K1,
                               Number<KWmmaPerblock>{} * Number<K0PerWmma>{} * K1,
                               K1,
                               K1,
                               K1,
                               K1,
                               I1));
            }
        }();

        return a_block_desc;
    }

    __host__ __device__ static constexpr auto MakeBBlockForWMMADescriptor()
    {
        constexpr auto b_block_desc = [&]() {
            if constexpr(BEnableLds)
            {
                // WARNING: gfx13 has changed to N->K0->K1 Per Block
                constexpr auto K0PerBlock = BKPerBlock / K1Value;
                if constexpr(BBlockLdsExtraN)
                {
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<NPerBlock>{}, Number<K0PerBlock>{}, K1),
                        make_tuple(Number<K0PerBlock + 1>{} * K1, K1, I1));
                }
                else
                {
                    return make_naive_tensor_descriptor(
                        make_tuple(Number<NPerBlock>{}, Number<K0PerBlock>{}, K1),
                        make_tuple(Number<K0PerBlock>{} * K1, K1, I1));
                }
            }
            else
            {
                constexpr auto B_KRow        = I2;
                constexpr auto KWmmaPerblock = BKPerBlock / BKPerWmma;
                constexpr auto K0PerWmma     = BKPerWmma / B_KRow / K1;
                // KWmma->NRepeat->MWave->K0PerWmma->KRow->MPerWmma->K1 Per Thread
                return make_naive_tensor_descriptor(
                    make_tuple(Number<KWmmaPerblock>{},
                               I1,
                               Number<NRepeat>{},
                               I1,
                               Number<K0PerWmma>{},
                               I1,
                               I1,
                               K1),
                    make_tuple(Number<K0PerWmma>{} * K1,
                               K1,
                               Number<KWmmaPerblock>{} * Number<K0PerWmma>{} * K1,
                               K1,
                               K1,
                               K1,
                               K1,
                               I1));
            }
        }();

        return b_block_desc;
    }

    __host__ __device__ static constexpr auto MakeABlockSliceCopyStep()
    {
        constexpr auto a_block_copy_step = [&]() {
            if constexpr(AEnableLds)
            {
                constexpr auto K0PerBlock = AKPerBlock / ABlockTransferSrcScalarPerVector;
                if constexpr(ALoadOption == TensorLoadOption::CLUSTER_DDS_LOAD)
                {
                    return make_multi_index(0, K0PerBlock * AClusterSize, 0);
                }
                else
                {
                    return make_multi_index(0, K0PerBlock, 0);
                }
            }
            else
            {
                // TODO: the logic has not changed for gfx13, need to check
                constexpr auto KWmmaPerBlock = AKPerBlock / AKPerWmma;

                return make_multi_index(KWmmaPerBlock, 0, 0, 0, 0, 0, 0, 0);
            }
        }();

        return a_block_copy_step;
    }

    __host__ __device__ static constexpr auto MakeBBlockSliceCopyStep()
    {
        constexpr auto b_block_copy_step = [&]() {
            if constexpr(BEnableLds)
            {
                constexpr auto K0PerBlock = BKPerBlock / BBlockTransferSrcScalarPerVector;
                if constexpr(BLoadOption == TensorLoadOption::CLUSTER_DDS_LOAD)
                {
                    return make_multi_index(0, K0PerBlock * BClusterSize, 0);
                }
                else
                {
                    return make_multi_index(0, K0PerBlock, 0);
                }
            }
            else
            {
                constexpr auto KWmmaPerBlock = BKPerBlock / BKPerWmma;

                return make_multi_index(KWmmaPerBlock, 0, 0, 0, 0, 0, 0, 0);
            }
        }();

        return b_block_copy_step;
    }

    // Describe how data read from (LDS/VGPR) buffer
    template <typename ABlockDesc_>
    __host__ __device__ static constexpr auto MakeAWaveDescriptor(const ABlockDesc_&)
    {

        constexpr auto a_wave_desc = [&]() {
            if constexpr(AEnableLds)
            {
                // AK0_M_AK1 -> AK0_MRepeat_Mwaves_AKRow_MPerWmma_AK1
                constexpr auto A_K        = AKPerBlock;
                constexpr auto A_LayoutK1 = K1Value;
                constexpr auto A_LayoutK0 = A_K / K1Value;
                constexpr auto A_KRow     = I2;

                return transform_tensor_descriptor(
                    ABlockDesc_{},
                    make_tuple(
                        make_unmerge_transform(
                            make_tuple(Number<MRepeat>{}, Number<MWaves>{}, Number<MPerWmma>{})),
                        make_unmerge_transform(make_tuple(Number<A_LayoutK0 / A_KRow>{}, A_KRow)),
                        make_pass_through_transform(Number<A_LayoutK1>{})),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3>{}, Sequence<5>{}));
            }
            else
            {
                // not changed for gfx13
                // KWmma_MRepeat_MWave_K0PerWmma_KRow_MPerWmma_K1 ->
                // K0_MRepeat_Mwaves_MPerWmma_K1
                constexpr auto KWmma     = ABlockDesc_{}.GetLength(I0);
                constexpr auto K0PerWmma = ABlockDesc_{}.GetLength(I4);
                constexpr auto A_KRow    = ABlockDesc_{}.GetLength(I5);
                constexpr auto A_K1      = ABlockDesc_{}.GetLength(I7);
                return make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                                      Number<KWmma * K0PerWmma>{},
                                                                      I1,
                                                                      Number<A_KRow>{},
                                                                      I1,
                                                                      Number<A_K1>{}));
            }
        }();

        return a_wave_desc;
    }

    template <typename BBlockDesc_>
    __host__ __device__ static constexpr auto MakeBWaveDescriptor(const BBlockDesc_&)
    {
        constexpr auto b_wave_desc = [&]() {
            if constexpr(BEnableLds)
            {
                constexpr auto B_K        = BKPerBlock;
                constexpr auto B_LayoutK1 = K1Value;
                constexpr auto B_LayoutK0 = B_K / K1Value;
                constexpr auto B_KRow     = I2;
                return transform_tensor_descriptor(
                    BBlockDesc_{},
                    make_tuple(
                        make_unmerge_transform(
                            make_tuple(Number<NRepeat>{}, Number<NWaves>{}, Number<NPerWmma>{})),
                        make_unmerge_transform(make_tuple(Number<B_LayoutK0 / B_KRow>{}, B_KRow)),
                        make_pass_through_transform(Number<B_LayoutK1>{})),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3>{}, Sequence<5>{}));
            }
            else
            {
                // KWmma_MRepeat_MWave_K0PerWmma_KRow_MPerWmma_K1 ->
                // K0_MRepeat_Mwaves_MPerWmma_K1
                constexpr auto KWmma     = BBlockDesc_{}.GetLength(I0);
                constexpr auto K0PerWmma = BBlockDesc_{}.GetLength(I4);
                constexpr auto B_KRow    = BBlockDesc_{}.GetLength(I5);
                constexpr auto B_K1      = BBlockDesc_{}.GetLength(I7);

                // Workaround, Freeze transform
                return make_naive_tensor_descriptor_packed(make_tuple(Number<NRepeat>{},
                                                                      Number<KWmma * K0PerWmma>{},
                                                                      I1,
                                                                      Number<B_KRow>{},
                                                                      I1,
                                                                      Number<B_K1>{}));
            }
        }();

        return b_wave_desc;
    }

#ifdef CK_EXTENSION_MX_TYPE
    template <typename ABlockDesc_>
    __host__ __device__ static constexpr auto MakeAScaleWaveDescriptor(const ABlockDesc_&)
    {

        constexpr auto a_wave_desc = [&]() {
            if constexpr(AEnableLds)
            {
                constexpr auto A_K1 = 2;
                constexpr auto A_K0 = math::integer_divide_ceil(KPerBlock, 256);

                return transform_tensor_descriptor(
                    ABlockDesc_{},
                    make_tuple(make_unmerge_transform(make_tuple(
                                   Number<MRepeat>{}, Number<MWaves>{}, Number<MPerWmma>{})),
                               make_unmerge_transform(make_tuple(Number<A_K0>{}, Number<A_K1>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<1, 2, 3>{}, Sequence<0, 4>{}));
            }
            else
            {
                return ABlockDesc_{};
            }
        }();

        return a_wave_desc;
    }

    template <typename BBlockDesc_>
    __host__ __device__ static constexpr auto MakeBScaleWaveDescriptor(const BBlockDesc_&)
    {

        constexpr auto b_wave_desc = [&]() {
            if constexpr(BEnableLds)
            {
                constexpr auto B_K1 = 2;
                constexpr auto B_K0 = math::integer_divide_ceil(KPerBlock, 256);
                return transform_tensor_descriptor(
                    BBlockDesc_{},
                    make_tuple(make_unmerge_transform(make_tuple(
                                   Number<NRepeat>{}, Number<NWaves>{}, Number<NPerWmma>{})),
                               make_unmerge_transform(make_tuple(Number<B_K0>{}, Number<B_K1>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<1, 2, 3>{}, Sequence<0, 4>{}));
            }
            else
            {
                return BBlockDesc_{};
            }
        }();

        return b_wave_desc;
    }

    __host__ __device__ static constexpr auto MakeAScaleBlockSliceCopyStep()
    {
        constexpr auto K0PerBlock        = math::integer_divide_ceil(KPerBlock, 256);
        constexpr auto a_block_copy_step = [&]() {
            if constexpr(AEnableLds)
            {

                return make_multi_index(0, K0PerBlock * 2);
            }
            else
            {
                return make_multi_index(K0PerBlock, 0, 0, 0, 0);
            }
        }();

        return a_block_copy_step;
    }

    __host__ __device__ static constexpr auto MakeBScaleBlockSliceCopyStep()
    {
        constexpr auto K0PerBlock        = math::integer_divide_ceil(KPerBlock, 256);
        constexpr auto b_block_copy_step = [&]() {
            if constexpr(BEnableLds)
            {
                return make_multi_index(0, K0PerBlock * 2);
            }
            else
            {
                return make_multi_index(K0PerBlock, 0, 0, 0, 0);
            }
        }();

        return b_block_copy_step;
    }
#endif

    __host__ __device__ static constexpr auto
    // *Caution Here repeat is shuffle repeat
    GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat()
    {
        constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMRepeatPerShuffle * MWaves * MPerWmma>{},
                           I1,
                           Number<CShuffleNRepeatPerShuffle * NWaves * NPerWmma>{}));

        return c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat;
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool CheckValidity(const AGridDesc& a_grid_desc,
                                                            const BGridDesc& b_grid_desc,
                                                            const CGridDesc_M_N& c_grid_desc_m_n,
                                                            const Block2CTileMap& block_2_ctile_map)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerWmma * MRepeat) == 0) &&
                          (NPerBlock % (NRepeat * NPerWmma)) == 0,
                      "Invalid tuning param!");

        static_assert((ALoadOption != TensorLoadOption::CLUSTER_MULTICAST_LOAD) ||
                          (BLoadOption != TensorLoadOption::CLUSTER_MULTICAST_LOAD) ||
                          (ALoadOption != TensorLoadOption::CLUSTER_ASYNC_MULTICAST_LDS_LOAD) ||
                          (BLoadOption != TensorLoadOption::CLUSTER_ASYNC_MULTICAST_LDS_LOAD),
                      "Both A and B cannot enable MultiCastLoad at the same time.");

        static_assert((ALoadOption != TensorLoadOption::CLUSTER_DDS_LOAD) ||
                          (BLoadOption != TensorLoadOption::CLUSTER_DDS_LOAD),
                      "Both A and B cannot enable DdsLoad at the same time.");

        if constexpr(is_mx_type_t_v<ADataType> && is_mx_type_t_v<BDataType>)
        {
            static_assert((ABlockTransferSrcScalarPerVector <= 4));
            static_assert((BBlockTransferSrcScalarPerVector <= 4));
            static_assert((ABlockTransferDstScalarPerVector_K1 <= 4));
            static_assert((BBlockTransferDstScalarPerVector_K1 <= 4));
            static_assert((K1Value == 1), "for MX type K1 should be 1, because of int32 based");
        }

        if constexpr(ALoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD ||
                     ALoadOption == TensorLoadOption::WGP_MULTICAST_LOAD)
        {
            static_assert((AEnableLds == false) && (AEnableTRLoadFromGlobal == false) &&
                              (AEnableTiledLoadFromGlobal == false),
                          "A use multicast load and other load methods should be unused.");
            if constexpr(ALoadOption == TensorLoadOption::WGP_MULTICAST_LOAD)
            {
                static_assert(EnableWaveGroup == true,
                              "A WGPMultiCastLoad should be used with WaveGroup enabled.");
            }
        }
        if constexpr(BLoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD ||
                     BLoadOption == TensorLoadOption::WGP_MULTICAST_LOAD)
        {
            static_assert((BEnableLds == false) && (BEnableTRLoadFromGlobal == false) &&
                              (BEnableTiledLoadFromGlobal == false),
                          "A use multicast load and other load methods should be unused.");
            if constexpr(BLoadOption == TensorLoadOption::WGP_MULTICAST_LOAD)
            {
                static_assert(EnableWaveGroup == true,
                              "B WGPMultiCastLoad should be used with WaveGroup enabled.");
            }
        }

        if constexpr(ALoadOption == TensorLoadOption::CLUSTER_ASYNC_MULTICAST_LDS_LOAD)
        {
            static_assert(AEnableLds == true, "A use async multicast lds load should enable lds.");
        }
        if constexpr(BLoadOption == TensorLoadOption::CLUSTER_ASYNC_MULTICAST_LDS_LOAD)
        {
            static_assert(BEnableLds == true, "B use async multicast lds load should enable lds.");
        }

        const auto GetAProblemsizeMK = [&]() {
            if constexpr(AEnableLds)
            {
                return make_tuple(a_grid_desc.GetLength(I0),
                                  a_grid_desc.GetLength(I1) * a_grid_desc.GetLength(I2));
            }
            else
            {
                if constexpr(AEnableTRLoadFromGlobal)
                {
                    return make_tuple(a_grid_desc.GetLength(I1) * a_grid_desc.GetLength(I2) *
                                          a_grid_desc.GetLength(I3) * a_grid_desc.GetLength(I4) *
                                          a_grid_desc.GetLength(I7),
                                      a_grid_desc.GetLength(I0) * a_grid_desc.GetLength(I5) *
                                          a_grid_desc.GetLength(I6));
                }
                else
                {
                    return make_tuple(a_grid_desc.GetLength(I1) * a_grid_desc.GetLength(I2) *
                                          a_grid_desc.GetLength(I3) * a_grid_desc.GetLength(I6),
                                      a_grid_desc.GetLength(I0) * a_grid_desc.GetLength(I4) *
                                          a_grid_desc.GetLength(I5) * a_grid_desc.GetLength(I7));
                }
            }
        };

        const auto GetBProblemsizeNK = [&]() {
            if constexpr(BEnableLds)
            {
                return make_tuple(b_grid_desc.GetLength(I0),
                                  b_grid_desc.GetLength(I1) * b_grid_desc.GetLength(I2));
            }
            else
            {
                if constexpr(BEnableTRLoadFromGlobal)
                {
                    return make_tuple(b_grid_desc.GetLength(I1) * b_grid_desc.GetLength(I2) *
                                          b_grid_desc.GetLength(I3) * b_grid_desc.GetLength(I4) *
                                          b_grid_desc.GetLength(I7),
                                      b_grid_desc.GetLength(I0) * b_grid_desc.GetLength(I5) *
                                          b_grid_desc.GetLength(I6));
                }
                else
                {
                    return make_tuple(b_grid_desc.GetLength(I1) * b_grid_desc.GetLength(I2) *
                                          b_grid_desc.GetLength(I3) * b_grid_desc.GetLength(I6),
                                      b_grid_desc.GetLength(I0) * b_grid_desc.GetLength(I4) *
                                          b_grid_desc.GetLength(I5) * b_grid_desc.GetLength(I7));
                }
            }
        };

        const auto M = GetAProblemsizeMK()[I0];
        const auto N = GetBProblemsizeNK()[I0];
        const auto K = GetAProblemsizeMK()[I1];

        if(!(M == c_grid_desc_m_n.GetLength(I0) && N == c_grid_desc_m_n.GetLength(I1) &&
             K == GetBProblemsizeNK()[I1]))
        {
            // for MX Type, int32 based K can be different from A, B
            if constexpr(is_mx_type_t_v<ADataType> && is_mx_type_t_v<BDataType>)
            {
                // K's number convert to MX data type
                const auto A_K = K * 32 / ADataType::BITS;
                const auto B_K = GetBProblemsizeNK()[I1] * 32 / BDataType::BITS;
                if(!(M == c_grid_desc_m_n.GetLength(I0) && N == c_grid_desc_m_n.GetLength(I1) &&
                     A_K == B_K))
                    return false;
                return true;
            }
            else
            {
                printf("A: MxK = %d x %d, B: NxK = %d x %d, C: MxN = %d x %d\n",
                       GetAProblemsizeMK()[I0],
                       GetAProblemsizeMK()[I1],
                       GetBProblemsizeNK()[I0],
                       GetBProblemsizeNK()[I1],
                       c_grid_desc_m_n.GetLength(I0),
                       c_grid_desc_m_n.GetLength(I1));
                printf("GridwiseOp err: ProblemSize check");
                return false;
            }
        }

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % AKPerBlock == 0))
        {
            printf("GridwiseOp err: ProblemSize division");
            return false;
        }

        // check gridwise gemm pipeline because K is from A, so use AKPerBlock
        const auto num_k_loop = K / AKPerBlock;

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            printf("GridwiseOp err: Pipeline not support this k_loop");
            return false;
        }

        if(!block_2_ctile_map.CheckValidity(c_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        if(!(a_grid_desc.GetElementSpaceSize() * sizeof(ADataType) <= TwoGB &&
             b_grid_desc.GetElementSpaceSize() * sizeof(BDataType) <= TwoGB))
        {
            return false;
        }
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        index_t ClusterSize = 1;
        if constexpr(ALoadOption == TensorLoadOption::CLUSTER_DDS_LOAD)
        {
            ClusterSize = AClusterSize;
        }
        else if constexpr(BLoadOption == TensorLoadOption::CLUSTER_DDS_LOAD)
        {
            ClusterSize = BClusterSize;
        }

        const index_t num_loop = K / AKPerBlock / ClusterSize;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return c_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto MakeDefaultBlock2CTileMap(
        const CGridDesc_M_N& c_grid_desc_m_n, index_t /* M01 */, index_t /* N01 */)
    {
        // Current it should align on the DDS block2tile firstly as mcast can fetch data
        // from different address
        if constexpr(AClusterSize > 1)
        {
            const auto N = c_grid_desc_m_n.GetLength(I1);
            if((N / NPerBlock) % AClusterSize != 0)
            {
                printf("GridwiseOp err: NBlocks must be multiple of AClusterSize\n");
            }

            if constexpr((BClusterSize > 1) && (BLoadOption == TensorLoadOption::CLUSTER_DDS_LOAD))
            {
                const auto M = c_grid_desc_m_n.GetLength(I0);
                if((M / MPerBlock) % BClusterSize != 0)
                {
                    printf("GridwiseOp err: MBlocks must be multiple of BClusterSize\n");
                }
                return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>(
                    c_grid_desc_m_n, BClusterSize);
            }
            else
            {
                return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>(
                    c_grid_desc_m_n, 1);
            }
        }
        else if constexpr(BClusterSize > 1)
        {
            const auto M = c_grid_desc_m_n.GetLength(I0);
            if((M / MPerBlock) % BClusterSize != 0)
            {
                printf("GridwiseOp err: MBlocks must be multiple of BClusterSize\n");
            }
            if constexpr((AClusterSize > 1) && (ALoadOption == TensorLoadOption::CLUSTER_DDS_LOAD))
            {
                const auto N = c_grid_desc_m_n.GetLength(I1);
                if((N / NPerBlock) % AClusterSize != 0)
                {
                    printf("GridwiseOp err: NBlocks must be multiple of AClusterSize\n");
                }
                return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>(
                    c_grid_desc_m_n, 1);
            }
            else
            {
                return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>(
                    c_grid_desc_m_n, BClusterSize);
            }
        }
        else
        {
            return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>(
                c_grid_desc_m_n);
        }
    }

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment

        // static constexpr auto max_lds_align = ABlockTransferSrcScalarPerVector;

        static constexpr auto a_block_space_size_aligned =
            AEnableLds ? math::integer_least_multiple(MakeABlockDescriptor().GetElementSpaceSize(),
                                                      ABlockTransferSrcScalarPerVector)
                       : 0;
        static constexpr auto b_block_space_size_aligned =
            BEnableLds ? math::integer_least_multiple(MakeBBlockDescriptor().GetElementSpaceSize(),
                                                      BBlockTransferSrcScalarPerVector)
                       : 0;

        static constexpr auto a_block_space_offset = 0;
        static constexpr auto b_block_space_offset = a_block_space_size_aligned;

#ifdef CK_EXTENSION_MX_TYPE
        static constexpr auto a_scale_block_space_size_aligned =
            (AEnableLds && mx_type_enable)
                ? math::integer_least_multiple(MakeAScaleBlockDescriptor().GetElementSpaceSize(), 1)
                : 0;

        static constexpr auto b_scale_block_space_size_aligned =
            (BEnableLds && mx_type_enable)
                ? math::integer_least_multiple(MakeBScaleBlockDescriptor().GetElementSpaceSize(), 1)
                : 0;

        static constexpr auto a_scale_block_space_offset =
            a_block_space_size_aligned + b_block_space_size_aligned;

        static constexpr auto b_scale_block_space_offset = a_block_space_size_aligned +
                                                           b_block_space_size_aligned +
                                                           a_scale_block_space_size_aligned;
#endif
        // LDS allocation for C shuffle in LDS
        static constexpr auto c_shuffle_block_space_size =
            GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat()
                .GetElementSpaceSize();

        static constexpr auto c_shuffle_block_space_offset = 0;

#ifdef CK_EXTENSION_MX_TYPE
        static constexpr auto lds_size =
            math::max(c_shuffle_block_space_size * sizeof(CShuffleDataType),
                      a_block_space_size_aligned * sizeof(view_type_t<ADataType>) +
                          b_block_space_size_aligned * sizeof(view_type_t<BDataType>) +
                          a_scale_block_space_size_aligned * sizeof(int32_t) +
                          b_scale_block_space_size_aligned * sizeof(int32_t));
#else
        static constexpr auto lds_size =
            math::max(c_shuffle_block_space_size * sizeof(CShuffleDataType),
                      a_block_space_size_aligned * sizeof(ADataType) +
                          b_block_space_size_aligned * sizeof(BDataType));
#endif
    };

    struct LaneSharedMemTrait
    {
        // Laneshared VGPR allocation for A and B
        static constexpr index_t max_lane_shared_align = 4;
#ifdef CK_EXTENSION_MX_TYPE
        static constexpr index_t sizeof_a_data_type =
            mx_type_enable ? sizeof(int32_t) : sizeof(ADataType);
        static constexpr index_t sizeof_b_data_type =
            mx_type_enable ? sizeof(int32_t) : sizeof(BDataType);
#else
        static constexpr index_t sizeof_a_data_type = sizeof(ADataType);
        static constexpr index_t sizeof_b_data_type = sizeof(BDataType);
#endif
        static constexpr index_t a_block_space_size_aligned =
            EnableWaveGroup && (AEnableLds == false)
                ? math::integer_least_multiple(MakeABlockDescriptor().GetElementSpaceSize() *
                                                   sizeof_a_data_type * 2,
                                               max_lane_shared_align)
                : 0;
        static constexpr index_t b_block_space_size_aligned =
            EnableWaveGroup && (BEnableLds == false)
                ? math::integer_least_multiple(MakeBBlockDescriptor().GetElementSpaceSize() *
                                                   sizeof_b_data_type * 2,
                                               max_lane_shared_align)
                : 0;

        static constexpr index_t a_block_space_offset = 0;
        static constexpr index_t b_block_space_offset = a_block_space_size_aligned;

#ifdef CK_EXTENSION_MX_TYPE
        static constexpr index_t a_scale_block_space_size_aligned =
            (EnableWaveGroup && (AEnableLds == false) && mx_type_enable)
                ? math::integer_least_multiple(MakeAScaleBlockDescriptor().GetElementSpaceSize() *
                                                   sizeof(uint32_t) * 2,
                                               max_lane_shared_align)
                : 0;

        static constexpr index_t b_scale_block_space_size_aligned =
            (EnableWaveGroup && (BEnableLds == false) && mx_type_enable)
                ? math::integer_least_multiple(MakeBScaleBlockDescriptor().GetElementSpaceSize() *
                                                   sizeof(uint32_t) * 2,
                                               max_lane_shared_align)
                : 0;

        static constexpr index_t a_scale_block_space_offset =
            a_block_space_size_aligned + b_block_space_size_aligned;

        static constexpr index_t b_scale_block_space_offset = a_block_space_size_aligned +
                                                              b_block_space_size_aligned +
                                                              a_scale_block_space_size_aligned;
#endif

#ifdef CK_EXTENSION_MX_TYPE
        static constexpr index_t lane_shared_size =
            a_block_space_size_aligned + b_block_space_size_aligned +
            a_scale_block_space_size_aligned + b_scale_block_space_size_aligned;
#else
        static constexpr index_t lane_shared_size =
            a_block_space_size_aligned + b_block_space_size_aligned;
#endif
    };

    using CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock =
        remove_cvref_t<decltype(MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            CGridDesc_M_N{}))>;
    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(CGridDesc_M_N{}, 1, 1))>;

#ifdef CK_EXTENSION_MX_TYPE
    // this Run function is only used in MX data type.
    template <bool HasMainKBlockLoop, typename Block2CTileMap = DefaultBlock2CTileMap>
    __device__ static void Run(const int32_t* __restrict__ p_a_grid,
                               const int32_t* __restrict__ p_b_grid,
                               const int32_t* __restrict__ p_a_scale,
                               const int32_t* __restrict__ p_b_scale,
                               CDataType* __restrict__ p_c_grid,
                               void* __restrict__ p_shared,
                               void* __restrict__ p_lane_shared,
                               const AGridDesc& a_grid_desc,
                               const BGridDesc& b_grid_desc,
                               const AScaleGridDesc& a_scale_grid_desc,
                               const BScaleGridDesc& b_scale_grid_desc,
                               const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const CElementwiseOperation& c_element_op,
                               const Block2CTileMap& block_2_ctile_map)
    {
        /*******************************************************************************/
        // Memory buffer zone.
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        const auto a_scale_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_scale, a_scale_grid_desc.GetElementSpaceSize());
        const auto b_scale_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_scale, b_scale_grid_desc.GetElementSpaceSize());

        /*******************************************************************************/
        // BlockIdx.x -> [BlockId.m, BlockId.n]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));
        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        // Store BlockId into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);
        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        /*******************************************************************************/
        // BlockLevel, A/B Matrix ThreadMapping in WMMA Source buffer, As Destinaion of
        // BlockWise_Copy
        const auto K = [&]() {
            if constexpr(AEnableLds)
            {
                return a_grid_desc.GetLength(I1) * a_grid_desc.GetLength(I2);
            }
            else
            {
                // TODO not changed for gfx13
                return a_grid_desc.GetLength(I0) * a_grid_desc.GetLength(I4) *
                       a_grid_desc.GetLength(I5) * a_grid_desc.GetLength(I7);
            }
        }();
        // the first two block descriptors are used for load; and the second two block
        // descriptors are used for calculation these two can be different, but need to make
        // sure the dimension between these can be tranformed; for example; a_block_desc =
        // [16(M), 2(K0), 8(K1)]; a_block_wmma_desc = [16(M), 8(K0), 2(K1)] these two can be
        // tranfered to each other
        constexpr auto a_block_desc = MakeABlockDescriptor();
        constexpr auto b_block_desc = MakeBBlockDescriptor();

        // the below block descriptor is used for calculation;
        constexpr auto a_block_wmma_desc  = MakeABlockForWMMADescriptor();
        constexpr auto b_block_wmma_desc  = MakeBBlockForWMMADescriptor();
        constexpr auto a_scale_block_desc = MakeAScaleBlockDescriptor();
        constexpr auto b_scale_block_desc = MakeBScaleBlockDescriptor();
        auto a_block_trait                = [&]() {
            // A matrix blockwise copy
            if constexpr(AEnableLds)
            {
                constexpr auto K0PerBlock = AKPerBlock / ABlockTransferSrcScalarPerVector;
                auto a_block_buf          = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<int32_t*>(p_shared), SharedMemTrait::a_block_space_size_aligned);
                auto a_scale_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<int32_t*>(p_shared) + SharedMemTrait::a_scale_block_space_offset,
                    SharedMemTrait::a_scale_block_space_size_aligned);
                // clang-format off
               if constexpr(AEnableAsyncCopy)
               {
                auto a_blockwise_copy =
                          ThreadGroupTensorSliceTransferAsync<ThisThreadBlockGrid,
 /* typename BlockSliceLengths,                    */         Sequence<MPerBlock, K0PerBlock, ABlockTransferSrcScalarPerVector>,
 /* typename ThreadClusterLengths,                 */         ABlockTransferThreadClusterLengths_M_K0_K1,
 /* typename ThreadClusterArrangeOrder,            */         ABlockTransferThreadClusterArrangeOrder,
                                                              int32_t,
                                                              int32_t,
 /* typename SrcDesc,                              */         decltype(a_grid_desc),
 /* typename DstDesc,                              */         decltype(a_block_desc),
 /* index_t SrcVectorDim,                          */         ABlockTransferSrcVectorDim,
 /* index_t DstVectorDim,                          */         2,
 /* index_t SrcScalarPerVector,                    */         ABlockTransferSrcScalarPerVector,
                                                              false,
                                                              true>(
                a_grid_desc,
                make_multi_index(m_block_data_idx_on_grid, 0, 0),
                a_block_desc,
                make_multi_index(0, 0, 0));
                constexpr auto ScaleKPerBlock = math::integer_divide_ceil(KPerBlock, 256) * 2;
                auto a_scale_blockwise_copy =
                         ThreadGroupTensorSliceTransferAsync<ThisThreadBlockGrid,
                                                             Sequence<MPerBlock, ScaleKPerBlock>,
                                                             Sequence<ABlockTransferThreadClusterLengths_M_K0_K1::At(0),
                                                                      (ScaleKPerBlock >> 1)>,
                                                             Sequence<0, 1>,
                                                             uint32_t,
                                                             uint32_t,
                                                             decltype(a_scale_grid_desc),
                                                             decltype(a_scale_block_desc),
                                                             1,
                                                             1,
                                                             1,
                                                             false,
                                                             true>(
                a_scale_grid_desc,
                make_multi_index(m_block_data_idx_on_grid, 0),
                a_scale_block_desc,
                make_multi_index(0, 0));

                return make_tuple(a_block_buf, a_blockwise_copy, a_scale_block_buf, a_scale_blockwise_copy);
               }
               else
               {
                auto a_blockwise_copy =
                    ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlockGrid,
/* typename SrcElementwiseOperation,              */    AElementwiseOperation,
/* typename DstElementwiseOperation,              */    ck::tensor_operation::element_wise::PassThrough,
/* InMemoryDataOperationEnum DstInMemOp,          */    InMemoryDataOperationEnum::Set,
/* typename BlockSliceLengths,                    */    Sequence<MPerBlock, K0PerBlock, ABlockTransferSrcScalarPerVector>,
/* typename ThreadClusterLengths,                 */    ABlockTransferThreadClusterLengths_M_K0_K1,
/* typename ThreadClusterArrangeOrder,            */    ABlockTransferThreadClusterArrangeOrder,
                                                        int32_t,
                                                        int32_t,
/* typename SrcDesc,                              */    decltype(a_grid_desc),
/* typename DstDesc,                              */    decltype(a_block_desc),
/* typename SrcDimAccessOrder,                    */    ABlockTransferSrcAccessOrder,
/* typename DstDimAccessOrder,                    */    Sequence<0, 1, 2>,
/* index_t SrcVectorDim,                          */    ABlockTransferSrcVectorDim,
/* index_t DstVectorDim,                          */    2,
/* index_t SrcScalarPerVector,                    */    ABlockTransferSrcScalarPerVector,
/* index_t DstScalarPerVector,                    */    ABlockTransferDstScalarPerVector_K1,
/* index_t SrcScalarStrideInVector,               */    1,
/* index_t DstScalarStrideInVector,               */    1,
/* bool ThreadTransferSrcResetCoordinateAfterRun, */    AThreadTransferSrcResetCoordinateAfterRun,
/* bool ThreadTransferDstResetCoordinateAfterRun, */    true,
                                                        NumGemmKPrefetchStage>(
                a_grid_desc,
                make_multi_index(m_block_data_idx_on_grid, 0, 0),
                a_element_op,
                a_block_desc,

                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

                constexpr auto ScaleKPerBlock =  math::integer_divide_ceil(KPerBlock, 256) * 2;
                auto a_scale_blockwise_copy =
                    ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlockGrid,
/* typename SrcElementwiseOperation,              */    ck::tensor_operation::element_wise::PassThrough,
/* typename DstElementwiseOperation,              */    ck::tensor_operation::element_wise::PassThrough,
/* InMemoryDataOperationEnum DstInMemOp,          */    InMemoryDataOperationEnum::Set,
/* typename BlockSliceLengths,                    */    Sequence<MPerBlock, ScaleKPerBlock>,
/* typename ThreadClusterLengths,                 */    Sequence<ABlockTransferThreadClusterLengths_M_K0_K1::At(0),
                                                                 (ScaleKPerBlock >> 1)>,
/* typename ThreadClusterArrangeOrder,            */    Sequence<0, 1>,
                                                        int32_t,
                                                        int32_t,
/* typename SrcDesc,                              */    decltype(a_scale_grid_desc),
/* typename DstDesc,                              */    decltype(a_scale_block_desc),
/* typename SrcDimAccessOrder,                    */    Sequence<0, 1>,
/* typename DstDimAccessOrder,                    */    Sequence<0, 1>,
/* index_t SrcVectorDim,                          */    1,
/* index_t DstVectorDim,                          */    1,
/* index_t SrcScalarPerVector,                    */    1,
/* index_t DstScalarPerVector,                    */    1,
/* index_t SrcScalarStrideInVector,               */    1,
/* index_t DstScalarStrideInVector,               */    1,
/* bool ThreadTransferSrcResetCoordinateAfterRun, */    AThreadTransferSrcResetCoordinateAfterRun,
/* bool ThreadTransferDstResetCoordinateAfterRun, */    true,
                                                        NumGemmKPrefetchStage>(
                a_scale_grid_desc,
                make_multi_index(m_block_data_idx_on_grid, 0),
                ck::tensor_operation::element_wise::PassThrough{},
                a_scale_block_desc,
                make_multi_index(0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

                return make_tuple(a_block_buf, a_blockwise_copy, a_scale_block_buf, a_scale_blockwise_copy);
               }
                // clang-format on
            }
            else
            {
                // Thread-wise copy
                // KPerBlock/WmmaK -> MRepeat -> MWaves -> K0PerWmma -> KRow -> MPerWmma -> K1
                // TODO not fixed
                constexpr auto KWmmaPerBlock   = AKPerBlock / AKPerWmma;
                constexpr auto K0PerWmma       = AKPerWmma / 2 / K1Value;
                constexpr auto ScaleK0PerBlock = math::integer_divide_ceil(KPerBlock, 256);

                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto a_blockwise_copy =
                    ThreadwiseTensorSliceTransfer_v2<int32_t,
                                                                    int32_t,
                                                                    decltype(a_grid_desc),
                                                                    decltype(a_block_desc),
                                                                    Sequence<Number<KWmmaPerBlock>{},
                                                                             I1,
                                                                             Number<MRepeat>{},
                                                                             I1,
                                                                             Number<K0PerWmma>{},
                                                                             I1,
                                                                             I1,
                                                                             Number<K1Value>{}>,
                                                                    Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                                    7,
                                                                    1,
                                                                    AThreadTransferSrcResetCoordinateAfterRun,
                                                                    true>(
                        a_grid_desc,
                        make_multi_index(0,
                                         m_block_data_idx_on_grid / (MWaves * MPerWmma * MRepeat),
                                         /*MRepeat*/ 0,
                                         /*MWaves*/ (ThisThreadBlockGrid::GetThreadId() / 32) /
                                             NWaves,
                                         /*A_K0PerWmma*/ 0,
                                         /*A_KRow*/ (ThisThreadBlockGrid::GetThreadId() % 2),
                                         /*MPerWmma*/ (ThisThreadBlockGrid::GetThreadId() % 32) / 2,
                                         /*K1Row*/ 0));
                auto a_scale_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
                                   int32_t,
                                   int32_t,
                                   decltype(a_scale_grid_desc),  // K0, MRepeat, MWaves, MPerWmma, 2
                                   decltype(a_scale_block_desc), // K0, MRepeat, 1, 1, 1
                                   Sequence<Number<ScaleK0PerBlock>{}, Number<MRepeat>{}, I1, I1, I1>,
                                   Sequence<0, 1, 2, 3, 4>,
                                   4,
                                   1,
                                   AThreadTransferSrcResetCoordinateAfterRun,
                                   true>(a_scale_grid_desc,
                          make_multi_index(0,
                                           m_block_data_idx_on_grid / (MWaves * MPerWmma),
                                           (ThisThreadBlockGrid::GetThreadId() / 32) / NWaves,
                                           (ThisThreadBlockGrid::GetThreadId() % 32) / 2,
                                           (ThisThreadBlockGrid::GetThreadId() % 2)));
                if constexpr(EnableWaveGroup)
                {
                    auto a_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, int32_t>(
                        a_block_desc.GetElementSpaceSize(),
                        static_cast<int32_t*>(p_lane_shared) +
                            LaneSharedMemTrait::a_block_space_offset / sizeof(int32_t));
                    auto a_scale_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, int32_t>(
                        a_scale_block_desc.GetElementSpaceSize(),
                        static_cast<int32_t*>(p_lane_shared) +
                            LaneSharedMemTrait::a_scale_block_space_offset / sizeof(int32_t));
                    return make_tuple(
                        a_block_buf, a_blockwise_copy, a_scale_block_buf, a_scale_blockwise_copy);
                }
                else
                {
                    auto a_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
                        a_block_desc.GetElementSpaceSize());
                    auto a_scale_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
                        a_scale_block_desc.GetElementSpaceSize());
                    return make_tuple(
                        a_block_buf, a_blockwise_copy, a_scale_block_buf, a_scale_blockwise_copy);
                }
            }
        };

        auto b_block_trait = [&]() {
            if constexpr(BEnableLds)
            {
                constexpr auto K0PerBlock = BKPerBlock / BBlockTransferSrcScalarPerVector;
                auto b_block_buf          = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<int32_t*>(p_shared) + SharedMemTrait::b_block_space_offset,
                    SharedMemTrait::b_block_space_size_aligned);
                auto b_scale_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<int32_t*>(p_shared) + SharedMemTrait::b_scale_block_space_offset,
                    SharedMemTrait::b_scale_block_space_size_aligned);
                // clang-format off
                if constexpr(BEnableAsyncCopy)
                {
                auto b_blockwise_copy =
                          ThreadGroupTensorSliceTransferAsync<ThisThreadBlockGrid,
 /* typename BlockSliceLengths,                    */         Sequence<NPerBlock, K0PerBlock, BBlockTransferSrcScalarPerVector>,
 /* typename ThreadClusterLengths,                 */         BBlockTransferThreadClusterLengths_N_K0_K1,
 /* typename ThreadClusterArrangeOrder,            */         BBlockTransferThreadClusterArrangeOrder,
                                                              int32_t,
                                                              int32_t,
 /* typename SrcDesc,                              */         decltype(b_grid_desc),
 /* typename DstDesc,                              */         decltype(b_block_desc),
 /* index_t SrcVectorDim,                          */         BBlockTransferSrcVectorDim,
 /* index_t DstVectorDim,                          */         2,
 /* index_t SrcScalarPerVector,                    */         BBlockTransferSrcScalarPerVector,
                                                              false,
                                                              true>(
                   b_grid_desc,
                   make_multi_index(n_block_data_idx_on_grid, 0, 0),
                   b_block_desc,
                   make_multi_index(0, 0, 0));

                constexpr auto ScaleKPerBlock =  math::integer_divide_ceil(KPerBlock, 256) * 2;
                auto b_scale_blockwise_copy =
                         ThreadGroupTensorSliceTransferAsync<ThisThreadBlockGrid,
                                                             Sequence<NPerBlock, ScaleKPerBlock>,
                                                             Sequence<BBlockTransferThreadClusterLengths_N_K0_K1::At(0),
                                                                      (ScaleKPerBlock >> 1)>,
                                                             Sequence<0, 1>,
                                                             int32_t,
                                                             int32_t,
                                                             decltype(b_scale_grid_desc),
                                                             decltype(b_scale_block_desc),
                                                             1,
                                                             1,
                                                             1,
                                                             false,
                                                             true>(
                   b_scale_grid_desc,
                   make_multi_index(n_block_data_idx_on_grid, 0),
                   b_scale_block_desc,
                   make_multi_index(0, 0));
                return make_tuple(b_block_buf, b_blockwise_copy, b_scale_block_buf, b_scale_blockwise_copy);
                }
                else
                {
                auto b_blockwise_copy =
                    ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlockGrid,
                                                        BElementwiseOperation,
                                                        ck::tensor_operation::element_wise::PassThrough,
                                                        InMemoryDataOperationEnum::Set,
                                                        Sequence<NPerBlock, K0PerBlock, BBlockTransferSrcScalarPerVector>,
                                                        BBlockTransferThreadClusterLengths_N_K0_K1,
                                                        BBlockTransferThreadClusterArrangeOrder,
                                                        int32_t,
                                                        int32_t,
                                                        decltype(b_grid_desc),
                                                        decltype(b_block_desc),
                                                        BBlockTransferSrcAccessOrder,
                                                        Sequence<0, 1, 2>,
                                                        BBlockTransferSrcVectorDim,
                                                        2,
                                                        BBlockTransferSrcScalarPerVector,
                                                        BBlockTransferDstScalarPerVector_K1,
                                                        1,
                                                        1,
                                                        BThreadTransferSrcResetCoordinateAfterRun,
                                                        true,
                                                        NumGemmKPrefetchStage>(
                    b_grid_desc,
                    make_multi_index(n_block_data_idx_on_grid, 0, 0),
                    b_element_op,
                    b_block_desc,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});
                constexpr auto ScaleKPerBlock =  math::integer_divide_ceil(KPerBlock, 256) * 2;
                auto b_scale_blockwise_copy =
                    ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlockGrid,
/* typename SrcElementwiseOperation,              */    ck::tensor_operation::element_wise::PassThrough,
/* typename DstElementwiseOperation,              */    ck::tensor_operation::element_wise::PassThrough,
/* InMemoryDataOperationEnum DstInMemOp,          */    InMemoryDataOperationEnum::Set,
/* typename BlockSliceLengths,                    */    Sequence<NPerBlock, ScaleKPerBlock>,
/* typename ThreadClusterLengths,                 */    Sequence<BBlockTransferThreadClusterLengths_N_K0_K1::At(0),
                                                                 (ScaleKPerBlock >> 1)>,
/* typename ThreadClusterArrangeOrder,            */    Sequence<0, 1>,
                                                        int32_t,
                                                        int32_t,
/* typename SrcDesc,                              */    decltype(b_scale_grid_desc),
/* typename DstDesc,                              */    decltype(b_scale_block_desc),
/* typename SrcDimAccessOrder,                    */    Sequence<0, 1>,
/* typename DstDimAccessOrder,                    */    Sequence<0, 1>,
/* index_t SrcVectorDim,                          */    1,
/* index_t DstVectorDim,                          */    1,
/* index_t SrcScalarPerVector,                    */    1,
/* index_t DstScalarPerVector,                    */    1,
/* index_t SrcScalarStrideInVector,               */    1,
/* index_t DstScalarStrideInVector,               */    1,
/* bool ThreadTransferSrcResetCoordinateAfterRun, */    AThreadTransferSrcResetCoordinateAfterRun,
/* bool ThreadTransferDstResetCoordinateAfterRun, */    true,
                                                        NumGemmKPrefetchStage>(
                b_scale_grid_desc,
                make_multi_index(n_block_data_idx_on_grid, 0),
                ck::tensor_operation::element_wise::PassThrough{},
                b_scale_block_desc,
                make_multi_index(0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

                return make_tuple(b_block_buf, b_blockwise_copy, b_scale_block_buf, b_scale_blockwise_copy);
                }
                // clang-format on
            }
            else
            {
                // Thread-wise copy
                // KPerBlock/WmmaK -> NRepeat -> NWaves -> WmmaK/K1 -> NPerWmma -> K1
                constexpr auto ScaleK0PerBlock = math::integer_divide_ceil(KPerBlock, 256);
                constexpr auto KWmmaPerBlock   = BKPerBlock / BKPerWmma;
                constexpr auto K0PerWmma       = BKPerWmma / 2 / K1Value;

                // Limitation: NumDim of Src and Dst descriptor should be identical
                auto b_blockwise_copy =
                    ThreadwiseTensorSliceTransfer_v2<int32_t,
                                                     int32_t,
                                                     decltype(b_grid_desc),
                                                     decltype(b_block_desc),
                                                     Sequence<Number<KWmmaPerBlock>{},
                                                              I1,
                                                              Number<NRepeat>{},
                                                              I1,
                                                              Number<K0PerWmma>{},
                                                              I1,
                                                              I1,
                                                              Number<K1Value>{}>,
                                                     Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                     7,
                                                     1,
                                                     BThreadTransferSrcResetCoordinateAfterRun,
                                                     true>(
                        b_grid_desc,
                        make_multi_index(0,
                                         n_block_data_idx_on_grid / (NWaves * NPerWmma * NRepeat),
                                         0,
                                         (ThisThreadBlockGrid::GetThreadId() / 32) % NWaves,
                                         0,
                                         (ThisThreadBlockGrid::GetThreadId() % 2),
                                         (ThisThreadBlockGrid::GetThreadId() % 32) / 2,
                                         0));
                auto b_scale_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
                    int32_t,
                    int32_t,
                    decltype(b_scale_grid_desc),  // K0, MRepeat, MWaves, MPerWmma, 2
                    decltype(b_scale_block_desc), // K0, MRepeat, 1, 1, 1
                    Sequence<Number<ScaleK0PerBlock>{}, Number<NRepeat>{}, I1, I1, I1>,
                    Sequence<0, 1, 2, 3, 4>,
                    4,
                    1,
                    AThreadTransferSrcResetCoordinateAfterRun,
                    true>(b_scale_grid_desc,
                          make_multi_index(0,
                                           n_block_data_idx_on_grid / (NWaves * NPerWmma),
                                           (ThisThreadBlockGrid::GetThreadId() / 32) % NWaves,
                                           (ThisThreadBlockGrid::GetThreadId() % 32) / 2,
                                           (ThisThreadBlockGrid::GetThreadId() % 2)));
                if constexpr(EnableWaveGroup)
                {
                    auto b_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, int32_t>(
                        b_block_desc.GetElementSpaceSize(),
                        static_cast<int32_t*>(p_lane_shared) +
                            LaneSharedMemTrait::b_block_space_offset / sizeof(int32_t));
                    auto b_scale_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, int32_t>(
                        b_scale_block_desc.GetElementSpaceSize(),
                        static_cast<int32_t*>(p_lane_shared) +
                            LaneSharedMemTrait::b_scale_block_space_offset / sizeof(int32_t));
                    return make_tuple(
                        b_block_buf, b_blockwise_copy, b_scale_block_buf, b_scale_blockwise_copy);
                }
                else
                {
                    auto b_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
                        b_block_desc.GetElementSpaceSize());
                    auto b_scale_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
                        b_scale_block_desc.GetElementSpaceSize());
                    return make_tuple(
                        b_block_buf, b_blockwise_copy, b_scale_block_buf, b_scale_blockwise_copy);
                }
            }
        };

        auto a_block_buf            = a_block_trait()[I0];
        auto a_blockwise_copy       = a_block_trait()[I1];
        auto a_scale_block_buf      = a_block_trait()[I2];
        auto a_scale_blockwise_copy = a_block_trait()[I3];

        auto b_block_buf            = b_block_trait()[I0];
        auto b_blockwise_copy       = b_block_trait()[I1];
        auto b_scale_block_buf      = b_block_trait()[I2];
        auto b_scale_blockwise_copy = b_block_trait()[I3];
        /*******************************************************************************/
        // GEMM
        constexpr auto KPack = math::integer_least_multiple(K1, WmmaK);

#if(defined(__gfx13__))
        auto blockwise_gemm =
            BlockwiseMXGemmWMMA<ThisThreadBlockGrid,
                                ADataType,
                                BDataType,
                                AccDataType,
                                decltype(MakeAWaveDescriptor(a_block_wmma_desc)),
                                decltype(MakeBWaveDescriptor(b_block_wmma_desc)),
                                decltype(MakeAScaleWaveDescriptor(a_scale_block_desc)),
                                decltype(MakeBScaleWaveDescriptor(b_scale_block_desc)),
                                MPerBlock,
                                NPerBlock,
                                KPerBlock,
                                MPerWmma,
                                NPerWmma,
                                KPerWmma,
                                MRepeat,
                                NRepeat,
                                KPack,
                                AEnableLds,
                                BEnableLds>{};
#else
        auto blockwise_gemm = BlockwiseGemmWMMA<BlockSize,
                                                ADataType,
                                                BDataType,
                                                AccDataType,
                                                decltype(MakeAWaveDescriptor(a_block_wmma_desc)),
                                                decltype(MakeBWaveDescriptor(b_block_wmma_desc)),
                                                MPerBlock,
                                                NPerBlock,
                                                KPerBlock,
                                                MPerWmma,
                                                NPerWmma,
                                                MRepeat,
                                                NRepeat,
                                                KPack,
                                                AEnableLds,
                                                BEnableLds>{};
#endif
        // Prepare Register for C matrix
        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        /*******************************************************************************/
        // Shift Per SUB_K
        constexpr auto a_block_slice_copy_step = MakeABlockSliceCopyStep();
        constexpr auto b_block_slice_copy_step = MakeBBlockSliceCopyStep();

        constexpr auto a_scale_block_slice_copy_step = MakeAScaleBlockSliceCopyStep();
        constexpr auto b_scale_block_slice_copy_step = MakeBScaleBlockSliceCopyStep();
        // gridwise GEMM pipeline
        const index_t KBlockMainLoop = __builtin_amdgcn_readfirstlane(K / AKPerBlock);
        __shared__ NamedBarrier<4> barrier_output;
        if constexpr(EnableWaveGroup)
        {
            if(get_wave_id_in_wavegroup() == WaveIdRun)
            {
                barrier_output.init();
                barrier_output.join();
            }
        }
        GridwiseGemmPipe::template Run<HasMainKBlockLoop>(a_grid_desc,
                                                          a_block_desc,
                                                          a_blockwise_copy,
                                                          a_grid_buf,
                                                          a_block_buf,
                                                          a_block_slice_copy_step,
                                                          b_grid_desc,
                                                          b_block_desc,
                                                          b_blockwise_copy,
                                                          b_grid_buf,
                                                          b_block_buf,
                                                          b_block_slice_copy_step,
                                                          blockwise_gemm,
                                                          c_thread_buf,
                                                          KBlockMainLoop,
                                                          a_scale_grid_desc,
                                                          a_scale_grid_buf,
                                                          a_scale_block_desc,
                                                          a_scale_block_buf,
                                                          a_scale_blockwise_copy,
                                                          a_scale_block_slice_copy_step,
                                                          b_scale_grid_desc,
                                                          b_scale_grid_buf,
                                                          b_scale_block_desc,
                                                          b_scale_block_buf,
                                                          b_scale_blockwise_copy,
                                                          b_scale_block_slice_copy_step);
        /*******************************************************************************/
        if(EnableWaveGroup == false || get_wave_id_in_wavegroup() == WaveIdRun)
        {
            store_to_global(blockwise_gemm,
                            c_grid_desc_mblock_mperblock_nblock_nperblock,
                            block_work_idx,
                            c_element_op,
                            c_thread_buf,
                            p_shared,
                            barrier_output,
                            c_grid_buf);
        }
    }
#endif // CK_EXTENSION_MX_TYPE
    template <bool HasMainKBlockLoop, typename Block2CTileMap = DefaultBlock2CTileMap>
    __device__ static void Run(const ADataType* __restrict__ p_a_grid,
                               const BDataType* __restrict__ p_b_grid,
                               CDataType* __restrict__ p_c_grid,
                               void* __restrict__ p_shared,
                               void* __restrict__ p_lane_shared,
                               const AGridDesc& a_grid_desc,
                               const BGridDesc& b_grid_desc,
                               const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const CElementwiseOperation& c_element_op,
                               const Block2CTileMap& block_2_ctile_map)
    {
        /*******************************************************************************/
        // Memory buffer zone.
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        /*******************************************************************************/
        // BlockIdx.x -> [BlockId.m, BlockId.n]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));
        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        // Store BlockId into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);
        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        /*******************************************************************************/
        // BlockLevel, A/B Matrix ThreadMapping in WMMA Source buffer, As Destinaion of
        // BlockWise_Copy
        const auto K = [&]() {
            if constexpr(AEnableLds)
            {
                return a_grid_desc.GetLength(I1) * a_grid_desc.GetLength(I2);
            }
            else
            {
                if constexpr(AEnableTRLoadFromGlobal)
                {
                    return a_grid_desc.GetLength(I0) * a_grid_desc.GetLength(I5);
                }
                else
                {
                    // TODO not changed for gfx13
                    return a_grid_desc.GetLength(I0) * a_grid_desc.GetLength(I4) *
                           a_grid_desc.GetLength(I5) * a_grid_desc.GetLength(I7);
                }
            }
        }();
        // the first two block descriptors are used for load; and the second two block
        // descriptors are used for calculation these two can be different, but need to make
        // sure the dimension between these can be tranformed; for example; a_block_desc =
        // [16(M), 2(K0), 8(K1)]; a_block_wmma_desc = [16(M), 8(K0), 2(K1)] these two can be
        // tranfered to each other
        constexpr auto a_block_desc = MakeABlockDescriptor();
        constexpr auto b_block_desc = MakeBBlockDescriptor();

        // the below block descriptor is used for calculation;
        constexpr auto a_block_wmma_desc = MakeABlockForWMMADescriptor();
        constexpr auto b_block_wmma_desc = MakeBBlockForWMMADescriptor();
        auto a_block_trait               = [&]() {
            // A matrix blockwise copy
            if constexpr(AEnableLds)
            {
                constexpr auto K0PerBlock = AKPerBlock / ABlockTransferSrcScalarPerVector;
                auto a_block_buf          = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<ADataType*>(p_shared), SharedMemTrait::a_block_space_size_aligned);
                // clang-format off
                if constexpr(AEnableAsyncCopy)
                {
                    auto a_blockwise_copy =
                            ThreadGroupTensorSliceTransferAsync<ThisThreadBlockGrid,
    /* typename BlockSliceLengths,                    */         Sequence<MPerBlock, K0PerBlock, ABlockTransferSrcScalarPerVector>,
    /* typename ThreadClusterLengths,                 */         ABlockTransferThreadClusterLengths_M_K0_K1,
    /* typename ThreadClusterArrangeOrder,            */         ABlockTransferThreadClusterArrangeOrder,
    /* typename SrcData,                              */         ADataType,
    /* typename DstData,                              */         ADataType,
    /* typename SrcDesc,                              */         decltype(a_grid_desc),
    /* typename DstDesc,                              */         decltype(a_block_desc),
    /* index_t SrcVectorDim,                          */         ABlockTransferSrcVectorDim,
    /* index_t DstVectorDim,                          */         2,
    /* index_t SrcScalarPerVector,                    */         ABlockTransferSrcScalarPerVector,
                                                                false,
                                                                true,
                                                                ALoadOption == TensorLoadOption::CLUSTER_ASYNC_MULTICAST_LDS_LOAD>(
                    a_grid_desc,
                    make_multi_index(m_block_data_idx_on_grid, 0, 0),
                    a_block_desc,
                    make_multi_index(0, 0, 0));
                    return make_tuple(a_block_buf, a_blockwise_copy);
                }
                else if constexpr(ALoadOption == TensorLoadOption::CLUSTER_DDS_LOAD)
                {
                    const index_t nk_block_data_idx_on_grid = __builtin_amdgcn_readfirstlane((block_work_idx[I1] % AClusterSize) * NPerBlock);
                    auto k1 = a_grid_desc.GetLength(I2);
                    const index_t k0_start_index_on_block = nk_block_data_idx_on_grid / k1;
                    auto a_blockwise_copy =
                        ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlockGrid,
    /* typename SrcElementwiseOperation,              */    AElementwiseOperation,
    /* typename DstElementwiseOperation,              */    ck::tensor_operation::element_wise::PassThrough,
    /* InMemoryDataOperationEnum DstInMemOp,          */    InMemoryDataOperationEnum::Set,
    /* typename BlockSliceLengths,                    */    Sequence<MPerBlock, K0PerBlock, ABlockTransferSrcScalarPerVector>,
    /* typename ThreadClusterLengths,                 */    ABlockTransferThreadClusterLengths_M_K0_K1,
    /* typename ThreadClusterArrangeOrder,            */    ABlockTransferThreadClusterArrangeOrder,
    /* typename SrcData,                              */    ADataType,
    /* typename DstData,                              */    ADataType,
    /* typename SrcDesc,                              */    decltype(a_grid_desc),
    /* typename DstDesc,                              */    decltype(a_block_desc),
    /* typename SrcDimAccessOrder,                    */    ABlockTransferSrcAccessOrder,
    /* typename DstDimAccessOrder,                    */    Sequence<0, 1, 2>,
    /* index_t SrcVectorDim,                          */    ABlockTransferSrcVectorDim,
    /* index_t DstVectorDim,                          */    2,
    /* index_t SrcScalarPerVector,                    */    ABlockTransferSrcScalarPerVector,
    /* index_t DstScalarPerVector,                    */    ABlockTransferDstScalarPerVector_K1,
    /* index_t SrcScalarStrideInVector,               */    1,
    /* index_t DstScalarStrideInVector,               */    1,
    /* bool ThreadTransferSrcResetCoordinateAfterRun, */    AThreadTransferSrcResetCoordinateAfterRun,
    /* bool ThreadTransferDstResetCoordinateAfterRun, */    true,
                                                            NumGemmKPrefetchStage>(
                    a_grid_desc,
                    make_multi_index(m_block_data_idx_on_grid, k0_start_index_on_block, 0),
                    a_element_op,
                    a_block_desc,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});

                    return make_tuple(a_block_buf, a_blockwise_copy);
                }
                else
                {
                    auto a_blockwise_copy =
                        ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlockGrid,
    /* typename SrcElementwiseOperation,              */    AElementwiseOperation,
    /* typename DstElementwiseOperation,              */    ck::tensor_operation::element_wise::PassThrough,
    /* InMemoryDataOperationEnum DstInMemOp,          */    InMemoryDataOperationEnum::Set,
    /* typename BlockSliceLengths,                    */    Sequence<MPerBlock, K0PerBlock, ABlockTransferSrcScalarPerVector>,
    /* typename ThreadClusterLengths,                 */    ABlockTransferThreadClusterLengths_M_K0_K1,
    /* typename ThreadClusterArrangeOrder,            */    ABlockTransferThreadClusterArrangeOrder,
    /* typename SrcData,                              */    ADataType,
    /* typename DstData,                              */    ADataType,
    /* typename SrcDesc,                              */    decltype(a_grid_desc),
    /* typename DstDesc,                              */    decltype(a_block_desc),
    /* typename SrcDimAccessOrder,                    */    ABlockTransferSrcAccessOrder,
    /* typename DstDimAccessOrder,                    */    Sequence<0, 1, 2>,
    /* index_t SrcVectorDim,                          */    ABlockTransferSrcVectorDim,
    /* index_t DstVectorDim,                          */    2,
    /* index_t SrcScalarPerVector,                    */    ABlockTransferSrcScalarPerVector,
    /* index_t DstScalarPerVector,                    */    ABlockTransferDstScalarPerVector_K1,
    /* index_t SrcScalarStrideInVector,               */    1,
    /* index_t DstScalarStrideInVector,               */    1,
    /* bool ThreadTransferSrcResetCoordinateAfterRun, */    AThreadTransferSrcResetCoordinateAfterRun,
    /* bool ThreadTransferDstResetCoordinateAfterRun, */    true,
                                                            NumGemmKPrefetchStage>(
                    a_grid_desc,
                    make_multi_index(m_block_data_idx_on_grid, 0, 0),
                    a_element_op,
                    a_block_desc,
                    make_multi_index(0, 0, 0),
                    ck::tensor_operation::element_wise::PassThrough{});

                    return make_tuple(a_block_buf, a_blockwise_copy);
                }
                // clang-format on
            }
            else
            {
                if constexpr(AEnableTRLoadFromGlobal)
                {
                    constexpr auto KWmmaPerBlock = AKPerBlock / AKPerWmma;
                    // clang-format off
                    auto a_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
/*SrcData                    */               ADataType,
/*DstData                    */               ADataType,
/*SrcDesc                    */               decltype(a_grid_desc),
/*DstDesc                    */               decltype(a_block_desc),
/*SliceLengths               */               Sequence<Number<KWmmaPerBlock>{}, I1, Number<MRepeat>{}, I1, I1, I1, I1, Number<8>{}>,
/*DimAccessOrder             */               Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
/*SrcVectorDim               */               7,
/*SrcScalarPerVector         */               8,
/*SrcScalarStrideInVector    */               1,
/*SrcResetCoordinateAfterRun */               true,
/*InvalidElementAsNaN        */               false,
/*UseTrLoad                  */               true>(
                        a_grid_desc,
                        make_multi_index(0,
                                         m_block_data_idx_on_grid / (MWaves * MPerWmma * MRepeat),
                                         0,
                                         (ThisThreadBlockGrid::GetThreadId() / 32) / NWaves,
                                         (get_lane_id() >> 2) & 1,
                                         ((get_lane_id() >> 3) << 2) + (get_lane_id() & 3),
                                         0,
                                         0));
                    // clang-format on
                    if constexpr(EnableWaveGroup)
                    {
                        auto a_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, ADataType>(
                            a_block_desc.GetElementSpaceSize(),
                            static_cast<ADataType*>(p_lane_shared) +
                                LaneSharedMemTrait::a_block_space_offset / sizeof(ADataType));

                        return make_tuple(a_block_buf, a_blockwise_copy);
                    }
                    else
                    {
                        auto a_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ADataType>(
                            a_block_desc.GetElementSpaceSize());
                        return make_tuple(a_block_buf, a_blockwise_copy);
                    }
                }
                else if constexpr(AEnableTiledLoadFromGlobal)
                {
                    // Thread-wise copy
                    // KPerBlock/WmmaK -> MRepeat -> MWaves -> K0PerWmma -> KRow -> MPerWmma -> K1
                    // TODO not fixed
                    constexpr auto KWmmaPerBlock  = AKPerBlock / AKPerWmma;
                    constexpr auto K0PerWmma      = AKPerWmma / 2 / K1Value;
                    constexpr auto ThreadsPerTile = 2;
                    constexpr auto VgprsPerTile   = 1;
                    constexpr auto AddrCnt        = 32 / ThreadsPerTile;
                    // Limitation: NumDim of Src and Dst descriptor should be identical
                    auto a_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
                                      ADataType,
                                      ADataType,
                                      decltype(a_grid_desc),
                                      decltype(a_block_desc),
                                      Sequence<Number<KWmmaPerBlock>{},
                                               I1,
                                               Number<MRepeat>{},
                                               I1,
                                               Number<K0PerWmma>{},
                                               I1,
                                               I1,
                                               Number<K1Value>{}>,
                                      Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                      7,
                                      ABlockTransferSrcScalarPerVector,
                                      1,
                                      AThreadTransferSrcResetCoordinateAfterRun,
                                      /*InvalidElementAsNaN       */ true,
                                      /*UseTrLoad                 */ false,
                                      /*ForceAlignToUint32        */ false,
                                      /*UseTileLoad               */ true,
                                      /*ThreadsPerTile            */ ThreadsPerTile,
                                      /*VgprsPerTile              */ VgprsPerTile>(
                        a_grid_desc,

                        make_multi_index(
                            0,
                            m_block_data_idx_on_grid / (MWaves * MPerWmma * MRepeat),
                            /*MRepeat     */ 0,
                            /*MWaves      */ (ThisThreadBlockGrid::GetThreadId() / 32) / NWaves,
                            /*A_K0PerWmma */ 0,
                            /*A_KRow      */ 0,
                            /*MPerWmma    */ (ThisThreadBlockGrid::GetThreadId() % AddrCnt),
                            /*K1Row       */ 0));

                    if constexpr(EnableWaveGroup)
                    {
                        auto a_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, ADataType>(
                            a_block_desc.GetElementSpaceSize(),
                            static_cast<ADataType*>(p_lane_shared) +
                                LaneSharedMemTrait::a_block_space_offset / sizeof(ADataType));

                        return make_tuple(a_block_buf, a_blockwise_copy);
                    }
                    else
                    {
                        auto a_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ADataType>(
                            a_block_desc.GetElementSpaceSize());
                        return make_tuple(a_block_buf, a_blockwise_copy);
                    }
                }
                else if constexpr(ALoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD ||
                                  ALoadOption == TensorLoadOption::WGP_MULTICAST_LOAD)
                {
                    // Thread-wise copy
                    // KPerBlock/WmmaK -> MRepeat -> MWaves -> K0PerWmma -> KRow -> MPerWmma -> K1
                    // TODO not fixed
                    constexpr auto KWmmaPerBlock = AKPerBlock / AKPerWmma;
                    constexpr auto K0PerWmma     = AKPerWmma / 2 / K1Value;
                    // Limitation: NumDim of Src and Dst descriptor should be identical
                    auto a_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
                                      ADataType,
                                      ADataType,
                                      decltype(a_grid_desc),
                                      decltype(a_block_desc),
                                      Sequence<Number<KWmmaPerBlock>{},
                                               I1,
                                               Number<MRepeat>{},
                                               I1,
                                               Number<K0PerWmma>{},
                                               I1,
                                               I1,
                                               Number<K1Value>{}>,
                                      Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                      7,
                                      ABlockTransferSrcScalarPerVector,
                                      1,
                                      AThreadTransferSrcResetCoordinateAfterRun,
                                      /*InvalidElementAsNaN       */ true,
                                      /*UseTrLoad                 */ false,
                                      /*ForceAlignToUint32        */ false,
                                      /*UseTileLoad               */ false,
                                      /*ThreadsPerTile            */ 1,
                                      /*VgprsPerTile              */ 1,
                                      /*ClusterMulticastLoad      */ ALoadOption>(
                        a_grid_desc,

                        make_multi_index(
                            0,
                            m_block_data_idx_on_grid / (MWaves * MPerWmma * MRepeat),
                            /*MRepeat     */ 0,
                            /*MWaves      */ (ThisThreadBlockGrid::GetThreadId() / 32) / NWaves,
                            /*A_K0PerWmma */ 0,
                            /*A_KRow      */ (ThisThreadBlockGrid::GetThreadId() % 2),
                            /*MPerWmma    */ (ThisThreadBlockGrid::GetThreadId() % 32) / 2,
                            /*K1Row       */ 0));

                    if constexpr(EnableWaveGroup)
                    {
                        auto a_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, ADataType>(
                            a_block_desc.GetElementSpaceSize(),
                            static_cast<ADataType*>(p_lane_shared) +
                                LaneSharedMemTrait::a_block_space_offset / sizeof(ADataType));

                        return make_tuple(a_block_buf, a_blockwise_copy);
                    }
                    else
                    {
                        auto a_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ADataType>(
                            a_block_desc.GetElementSpaceSize());
                        return make_tuple(a_block_buf, a_blockwise_copy);
                    }
                }
                else
                {
                    // Thread-wise copy
                    // KPerBlock/WmmaK -> MRepeat -> MWaves -> K0PerWmma -> KRow -> MPerWmma -> K1
                    // TODO not fixed
                    constexpr auto KWmmaPerBlock = AKPerBlock / AKPerWmma;
                    constexpr auto K0PerWmma     = AKPerWmma / 2 / K1Value;

                    // Limitation: NumDim of Src and Dst descriptor should be identical
                    auto a_blockwise_copy =
                        ThreadwiseTensorSliceTransfer_v2<ADataType,
                                                                       ADataType,
                                                                       decltype(a_grid_desc),
                                                                       decltype(a_block_desc),
                                                                       Sequence<Number<KWmmaPerBlock>{},
                                                                                I1,
                                                                                Number<MRepeat>{},
                                                                                I1,
                                                                                Number<K0PerWmma>{},
                                                                                I1,
                                                                                I1,
                                                                                Number<K1Value>{}>,
                                                                       Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                                       7,
                                                                       ABlockTransferSrcScalarPerVector,
                                                                       1,
                                                                       AThreadTransferSrcResetCoordinateAfterRun,
                                                                       true>(
                            a_grid_desc,
                            make_multi_index(
                                0,
                                m_block_data_idx_on_grid / (MWaves * MPerWmma * MRepeat),
                                /*MRepeat     */ 0,
                                /*MWaves      */ (ThisThreadBlockGrid::GetThreadId() / 32) / NWaves,
                                /*A_K0PerWmma */ 0,
                                /*A_KRow      */ (ThisThreadBlockGrid::GetThreadId() % 2),
                                /*MPerWmma    */ (ThisThreadBlockGrid::GetThreadId() % 32) / 2,
                                /*K1Row       */ 0));
                    if constexpr(EnableWaveGroup)
                    {
                        auto a_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, ADataType>(
                            a_block_desc.GetElementSpaceSize(),
                            static_cast<ADataType*>(p_lane_shared) +
                                LaneSharedMemTrait::a_block_space_offset / sizeof(ADataType));

                        return make_tuple(a_block_buf, a_blockwise_copy);
                    }
                    else
                    {
                        auto a_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ADataType>(
                            a_block_desc.GetElementSpaceSize());
                        return make_tuple(a_block_buf, a_blockwise_copy);
                    }
                }
            }
        };

        auto b_block_trait = [&]() {
            if constexpr(BEnableLds)
            {
                constexpr auto K0PerBlock = BKPerBlock / BBlockTransferSrcScalarPerVector;
                auto b_block_buf          = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                    static_cast<BDataType*>(p_shared) + SharedMemTrait::b_block_space_offset,
                    SharedMemTrait::b_block_space_size_aligned);
                // clang-format off
                if constexpr(BEnableAsyncCopy)
                {
                    auto b_blockwise_copy =
                            ThreadGroupTensorSliceTransferAsync<ThisThreadBlockGrid,
    /* typename BlockSliceLengths,                    */         Sequence<NPerBlock, K0PerBlock, BBlockTransferSrcScalarPerVector>,
    /* typename ThreadClusterLengths,                 */         BBlockTransferThreadClusterLengths_N_K0_K1,
    /* typename ThreadClusterArrangeOrder,            */         BBlockTransferThreadClusterArrangeOrder,
    /* typename SrcData,                              */         BDataType,
    /* typename DstData,                              */         BDataType,
    /* typename SrcDesc,                              */         decltype(b_grid_desc),
    /* typename DstDesc,                              */         decltype(b_block_desc),
    /* index_t SrcVectorDim,                          */         BBlockTransferSrcVectorDim,
    /* index_t DstVectorDim,                          */         2,
    /* index_t SrcScalarPerVector,                    */         BBlockTransferSrcScalarPerVector,
                                                                false,
                                                                true,
                                                                BLoadOption == TensorLoadOption::CLUSTER_ASYNC_MULTICAST_LDS_LOAD>(
                    b_grid_desc,
                    make_multi_index(n_block_data_idx_on_grid, 0, 0),
                    b_block_desc,
                    make_multi_index(0, 0, 0));

                    return make_tuple(b_block_buf, b_blockwise_copy);
                }
                else
                {
                    if constexpr(BLoadOption == TensorLoadOption::CLUSTER_DDS_LOAD)
                    {
                        auto k1 = b_grid_desc.GetLength(I2);
                        const index_t mk_block_data_idx_on_grid = __builtin_amdgcn_readfirstlane((block_work_idx[I0] % BClusterSize) * MPerBlock);
                        const index_t k0_start_index_on_block = mk_block_data_idx_on_grid / k1;

                        auto b_blockwise_copy =
                            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlockGrid,
                                                                BElementwiseOperation,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                InMemoryDataOperationEnum::Set,
                                                                Sequence<NPerBlock, K0PerBlock, BBlockTransferSrcScalarPerVector>,
                                                                BBlockTransferThreadClusterLengths_N_K0_K1,
                                                                BBlockTransferThreadClusterArrangeOrder,
                                                                BDataType,
                                                                BDataType,
                                                                decltype(b_grid_desc),
                                                                decltype(b_block_desc),
                                                                BBlockTransferSrcAccessOrder,
                                                                Sequence<0, 1, 2>,
                                                                BBlockTransferSrcVectorDim,
                                                                2,
                                                                BBlockTransferSrcScalarPerVector,
                                                                BBlockTransferDstScalarPerVector_K1,
                                                                1,
                                                                1,
                                                                BThreadTransferSrcResetCoordinateAfterRun,
                                                                true,
                                                                NumGemmKPrefetchStage>(
                            b_grid_desc,
                            make_multi_index(n_block_data_idx_on_grid, k0_start_index_on_block, 0),
                            b_element_op,
                            b_block_desc,
                            make_multi_index(0, 0, 0),
                            ck::tensor_operation::element_wise::PassThrough{});

                        return make_tuple(b_block_buf, b_blockwise_copy);
                    }
                    else
                    {
                        auto b_blockwise_copy =
                            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlockGrid,
                                                                BElementwiseOperation,
                                                                ck::tensor_operation::element_wise::PassThrough,
                                                                InMemoryDataOperationEnum::Set,
                                                                Sequence<NPerBlock, K0PerBlock, BBlockTransferSrcScalarPerVector>,
                                                                BBlockTransferThreadClusterLengths_N_K0_K1,
                                                                BBlockTransferThreadClusterArrangeOrder,
                                                                BDataType,
                                                                BDataType,
                                                                decltype(b_grid_desc),
                                                                decltype(b_block_desc),
                                                                BBlockTransferSrcAccessOrder,
                                                                Sequence<0, 1, 2>,
                                                                BBlockTransferSrcVectorDim,
                                                                2,
                                                                BBlockTransferSrcScalarPerVector,
                                                                BBlockTransferDstScalarPerVector_K1,
                                                                1,
                                                                1,
                                                                BThreadTransferSrcResetCoordinateAfterRun,
                                                                true,
                                                                NumGemmKPrefetchStage>(
                            b_grid_desc,
                            make_multi_index(n_block_data_idx_on_grid, 0, 0),
                            b_element_op,
                            b_block_desc,
                            make_multi_index(0, 0, 0),
                            ck::tensor_operation::element_wise::PassThrough{});

                        return make_tuple(b_block_buf, b_blockwise_copy);
                    }
                    // clang-format on
                }
            }
            else
            {
                if constexpr(BEnableTRLoadFromGlobal)
                {
                    constexpr auto KWmmaPerBlock = BKPerBlock / BKPerWmma;
                    // clang-format off
                    auto b_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
/*SrcData                     */                    BDataType,
/*DstData                     */                    BDataType,
/*SrcDesc                     */                    decltype(b_grid_desc),
/*DstDesc                     */                    decltype(b_block_desc),
/*SliceLengths                */                    Sequence<Number<KWmmaPerBlock>{}, I1, Number<NRepeat>{}, I1, I1, I1, I1, Number<8>{}>,
/*DimAccessOrder              */                    Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
/*SrcVectorDim                */                    7,
/*SrcScalarPerVector          */                    8,
/*SrcScalarStrideInVector     */                    1,
/*SrcResetCoordinateAfterRun  */                    BThreadTransferSrcResetCoordinateAfterRun,
/*InvalidElementAsNaN         */                    false,
/*UseTrLoad                   */                    true>(
                        b_grid_desc,
                        make_multi_index(0,
                                         n_block_data_idx_on_grid / (NWaves * NPerWmma * NRepeat),
                                         0,
                                         (ThisThreadBlockGrid::GetThreadId() / 32) % NWaves,
                                         (get_lane_id() >> 2) & 1,
                                         ((get_lane_id() >> 3) << 2) + (get_lane_id() & 3),
                                         0,
                                         0));
                    // clang-format on
                    if constexpr(EnableWaveGroup)
                    {
                        auto b_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, BDataType>(
                            b_block_desc.GetElementSpaceSize(),
                            static_cast<BDataType*>(p_lane_shared) +
                                LaneSharedMemTrait::b_block_space_offset / sizeof(BDataType));
                        return make_tuple(b_block_buf, b_blockwise_copy);
                    }
                    else
                    {
                        auto b_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, BDataType>(
                            b_block_desc.GetElementSpaceSize());
                        return make_tuple(b_block_buf, b_blockwise_copy);
                    }
                }
                else if constexpr(BEnableTiledLoadFromGlobal)
                {
                    // Thread-wise copy
                    // KPerBlock/WmmaK -> NRepeat -> NWaves -> WmmaK/K1 -> NPerWmma -> K1
                    constexpr auto KWmmaPerBlock  = BKPerBlock / BKPerWmma;
                    constexpr auto K0PerWmma      = BKPerWmma / 2 / K1Value;
                    constexpr auto ThreadsPerTile = 2;
                    constexpr auto VgprsPerTile   = 1;
                    constexpr auto AddrCnt        = 32 / ThreadsPerTile;
                    // Limitation: NumDim of Src and Dst descriptor should be identical
                    auto b_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
                        BDataType,
                        BDataType,
                        decltype(b_grid_desc),
                        decltype(b_block_desc),
                        Sequence<Number<KWmmaPerBlock>{},
                                 I1,
                                 Number<NRepeat>{},
                                 I1,
                                 Number<K0PerWmma>{},
                                 I1,
                                 I1,
                                 Number<K1Value>{}>,
                        Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                        7,
                        BBlockTransferSrcScalarPerVector,
                        1,
                        BThreadTransferSrcResetCoordinateAfterRun,
                        /*InvalidElementAsNaN        */ true,
                        /*UseTrLoad                  */ false,
                        /*ForceAlignToUint32         */ false,
                        /*UseTileLoad                */ true,
                        /*ThreadsPerTile             */ ThreadsPerTile,
                        /*VgprsPerTile               */ VgprsPerTile>(
                        b_grid_desc,
                        make_multi_index(0,
                                         n_block_data_idx_on_grid / (NWaves * NPerWmma * NRepeat),
                                         0,
                                         (ThisThreadBlockGrid::GetThreadId() / 32) % NWaves,
                                         0,
                                         0,
                                         (ThisThreadBlockGrid::GetThreadId() % AddrCnt),
                                         0));
                    if constexpr(EnableWaveGroup)
                    {
                        auto b_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, BDataType>(
                            b_block_desc.GetElementSpaceSize(),
                            static_cast<BDataType*>(p_lane_shared) +
                                LaneSharedMemTrait::b_block_space_offset / sizeof(BDataType));
                        return make_tuple(b_block_buf, b_blockwise_copy);
                    }
                    else
                    {
                        auto b_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, BDataType>(
                            b_block_desc.GetElementSpaceSize());
                        return make_tuple(b_block_buf, b_blockwise_copy);
                    }
                }
                else if constexpr(BLoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD ||
                                  BLoadOption == TensorLoadOption::WGP_MULTICAST_LOAD)
                {
                    // Thread-wise copy
                    // KPerBlock/WmmaK -> NRepeat -> NWaves -> WmmaK/K1 -> NPerWmma -> K1
                    constexpr auto KWmmaPerBlock = BKPerBlock / BKPerWmma;
                    constexpr auto K0PerWmma     = BKPerWmma / 2 / K1Value;
                    // Limitation: NumDim of Src and Dst descriptor should be identical
                    auto b_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
                        BDataType,
                        BDataType,
                        decltype(b_grid_desc),
                        decltype(b_block_desc),
                        Sequence<Number<KWmmaPerBlock>{},
                                 I1,
                                 Number<NRepeat>{},
                                 I1,
                                 Number<K0PerWmma>{},
                                 I1,
                                 I1,
                                 Number<K1Value>{}>,
                        Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                        7,
                        BBlockTransferSrcScalarPerVector,
                        1,
                        BThreadTransferSrcResetCoordinateAfterRun,
                        /*InvalidElementAsNaN        */ false,
                        /*UseTrLoad                  */ false,
                        /*ForceAlignToUint32         */ true,
                        /*UseTileLoad                */ false,
                        /*ThreadsPerTile             */ 1,
                        /*VgprsPerTile               */ 1,
                        /*ClusterMulticastLoad       */ BLoadOption>(
                        b_grid_desc,
                        make_multi_index(0,
                                         n_block_data_idx_on_grid / (NWaves * NPerWmma * NRepeat),
                                         0,
                                         (ThisThreadBlockGrid::GetThreadId() / 32) % NWaves,
                                         0,
                                         (ThisThreadBlockGrid::GetThreadId() % 2),
                                         (ThisThreadBlockGrid::GetThreadId() % 32) / 2,
                                         0));
                    if constexpr(EnableWaveGroup)
                    {
                        auto b_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, BDataType>(
                            b_block_desc.GetElementSpaceSize(),
                            static_cast<BDataType*>(p_lane_shared) +
                                LaneSharedMemTrait::b_block_space_offset / sizeof(BDataType));
                        return make_tuple(b_block_buf, b_blockwise_copy);
                    }
                    else
                    {
                        auto b_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, BDataType>(
                            b_block_desc.GetElementSpaceSize());
                        return make_tuple(b_block_buf, b_blockwise_copy);
                    }
                }
                else
                {
                    // Thread-wise copy
                    // KPerBlock/WmmaK -> NRepeat -> NWaves -> WmmaK/K1 -> NPerWmma -> K1
                    constexpr auto KWmmaPerBlock = BKPerBlock / BKPerWmma;
                    constexpr auto K0PerWmma     = BKPerWmma / 2 / K1Value;
                    // Limitation: NumDim of Src and Dst descriptor should be identical
                    auto b_blockwise_copy =
                        ThreadwiseTensorSliceTransfer_v2<BDataType,
                                                         BDataType,
                                                         decltype(b_grid_desc),
                                                         decltype(b_block_desc),
                                                         Sequence<Number<KWmmaPerBlock>{},
                                                                  I1,
                                                                  Number<NRepeat>{},
                                                                  I1,
                                                                  Number<K0PerWmma>{},
                                                                  I1,
                                                                  I1,
                                                                  Number<K1Value>{}>,
                                                         Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                         7,
                                                         BBlockTransferSrcScalarPerVector,
                                                         1,
                                                         BThreadTransferSrcResetCoordinateAfterRun,
                                                         true>(
                            b_grid_desc,
                            make_multi_index(0,
                                             n_block_data_idx_on_grid /
                                                 (NWaves * NPerWmma * NRepeat),
                                             0,
                                             (ThisThreadBlockGrid::GetThreadId() / 32) % NWaves,
                                             0,
                                             (ThisThreadBlockGrid::GetThreadId() % 2),
                                             (ThisThreadBlockGrid::GetThreadId() % 32) / 2,
                                             0));
                    if constexpr(EnableWaveGroup)
                    {
                        auto b_block_buf = make_static_buffer_v4<AddressSpaceEnum::Vgpr, BDataType>(
                            b_block_desc.GetElementSpaceSize(),
                            static_cast<BDataType*>(p_lane_shared) +
                                LaneSharedMemTrait::b_block_space_offset / sizeof(BDataType));
                        return make_tuple(b_block_buf, b_blockwise_copy);
                    }
                    else
                    {
                        auto b_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, BDataType>(
                            b_block_desc.GetElementSpaceSize());
                        return make_tuple(b_block_buf, b_blockwise_copy);
                    }
                }
            }
        };

        auto a_block_buf      = a_block_trait()[I0];
        auto a_blockwise_copy = a_block_trait()[I1];

        auto b_block_buf      = b_block_trait()[I0];
        auto b_blockwise_copy = b_block_trait()[I1];

        /*******************************************************************************/
        // GEMM
        constexpr auto KPack = math::integer_least_multiple(K1, WmmaK);
#if(defined(__gfx13__))
        auto blockwise_gemm = BlockwiseGemmWMMA<ThisThreadBlockGrid,
                                                ADataType,
                                                BDataType,
                                                AccDataType,
                                                decltype(MakeAWaveDescriptor(a_block_wmma_desc)),
                                                decltype(MakeBWaveDescriptor(b_block_wmma_desc)),
                                                MPerBlock,
                                                NPerBlock,
                                                KPerBlock,
                                                MPerWmma,
                                                NPerWmma,
                                                KPerWmma,
                                                MRepeat,
                                                NRepeat,
                                                KPack,
                                                AEnableLds,
                                                BEnableLds,
                                                false, // APermute
                                                false, // BPermute
                                                false, // Transpose C
                                                ALoadOption,
                                                BLoadOption>{};
#else
        auto blockwise_gemm = BlockwiseGemmWMMA<BlockSize,
                                                ADataType,
                                                BDataType,
                                                AccDataType,
                                                decltype(MakeAWaveDescriptor(a_block_wmma_desc)),
                                                decltype(MakeBWaveDescriptor(b_block_wmma_desc)),
                                                MPerBlock,
                                                NPerBlock,
                                                KPerBlock,
                                                MPerWmma,
                                                NPerWmma,
                                                MRepeat,
                                                NRepeat,
                                                KPack,
                                                AEnableLds,
                                                BEnableLds>{};
#endif

        // Prepare Register for C matrix
        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        /*******************************************************************************/
        // Shift Per SUB_K
        constexpr auto a_block_slice_copy_step = MakeABlockSliceCopyStep();
        constexpr auto b_block_slice_copy_step = MakeBBlockSliceCopyStep();
        // gridwise GEMM pipeline
        const index_t KBlockMainLoop = __builtin_amdgcn_readfirstlane(K / AKPerBlock);

        __shared__ NamedBarrier<4> barrier_output;
        if constexpr(EnableWaveGroup)
        {
            if(get_wave_id_in_wavegroup() == WaveIdRun)
            {
                barrier_output.init();
                barrier_output.join();
            }
        }

        if constexpr(ALoadOption == TensorLoadOption::CLUSTER_DDS_LOAD ||
                     ALoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD ||
                     BLoadOption == TensorLoadOption::CLUSTER_DDS_LOAD ||
                     BLoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD)
        {
            GridwiseGemmPipe::template Run<HasMainKBlockLoop>(a_grid_desc,
                                                              a_block_desc,
                                                              a_blockwise_copy,
                                                              a_grid_buf,
                                                              a_block_buf,
                                                              a_block_slice_copy_step,
                                                              b_grid_desc,
                                                              b_block_desc,
                                                              b_blockwise_copy,
                                                              b_grid_buf,
                                                              b_block_buf,
                                                              b_block_slice_copy_step,
                                                              blockwise_gemm,
                                                              c_thread_buf,
                                                              KBlockMainLoop,
                                                              AClusterSize,
                                                              BClusterSize);
        }
        else
        {
            GridwiseGemmPipe::template Run<HasMainKBlockLoop>(a_grid_desc,
                                                              a_block_desc,
                                                              a_blockwise_copy,
                                                              a_grid_buf,
                                                              a_block_buf,
                                                              a_block_slice_copy_step,
                                                              b_grid_desc,
                                                              b_block_desc,
                                                              b_blockwise_copy,
                                                              b_grid_buf,
                                                              b_block_buf,
                                                              b_block_slice_copy_step,
                                                              blockwise_gemm,
                                                              c_thread_buf,
                                                              KBlockMainLoop);
        }

        /*******************************************************************************/
        if(EnableWaveGroup == false || get_wave_id_in_wavegroup() == WaveIdRun)
        {
            store_to_global(blockwise_gemm,
                            c_grid_desc_mblock_mperblock_nblock_nperblock,
                            block_work_idx,
                            c_element_op,
                            c_thread_buf,
                            p_shared,
                            barrier_output,
                            c_grid_buf);
        }
    }

    template <typename BlockwiseGemm,
              typename CTileIdx,
              typename CThreadBuffer,
              typename NamedBarrier,
              typename CGridBuffer>
    __device__ static void store_to_global(const BlockwiseGemm& blockwise_gemm,
                                           const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                               c_grid_desc_mblock_mperblock_nblock_nperblock,
                                           const CTileIdx& block_work_idx,
                                           const CElementwiseOperation& c_element_op,
                                           const CThreadBuffer& c_thread_buf,
                                           void* __restrict__ p_shared,
                                           NamedBarrier& barrier_output,
                                           CGridBuffer& c_grid_buf)
    {
#if defined(__gfx13__)
        // C mapping in single thread.
        // |  MRepeat  |  MWave  |  MLoopAcc  | MSubGroup  |  NRepeat  |  NWave  |
        // NThreadPerSubGroup  |  MLoopAcc  |
        constexpr auto c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs =
            BlockwiseGemm::
                GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs();

        // C mapping in single block
        // | MRepeat | MWave | MsubGroup | NRepeat | NWave | NThreadPerSubGroup | MLoopAcc |
        // MConsecutiveVgprs |
        constexpr auto c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp =
            BlockwiseGemm::
                GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs();

        constexpr auto MWave =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I1);
        constexpr auto MSubGroup =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I2);
        constexpr auto NWave =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I4);
        constexpr auto NThreadPerSubGroup =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I5);
        constexpr auto LoopAccPerThread =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I6);
        constexpr auto MConsecutiveAccs =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I7);

        // LDS descriptor, shuffle and write out in MRepeat x NRepeat times
        constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
            GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat();

        auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<CShuffleDataType*>(p_shared) + SharedMemTrait::c_shuffle_block_space_offset,
            SharedMemTrait::c_shuffle_block_space_size);

        constexpr auto
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs =
                transform_tensor_descriptor(
                    c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
                    make_tuple(make_freeze_transform(I0),
                               make_unmerge_transform(
                                   make_tuple(Number<CShuffleMRepeatPerShuffle>{}, // MRepeat per
                                                                                   // shuffle repeat
                                              MWave,                               // MWave
                                              LoopAccPerThread,
                                              MSubGroup,
                                              MConsecutiveAccs)), // MConsecutiveAccs = MPerWmma /
                                                                  // MSubGroup / MAccPerThread
                               make_freeze_transform(I0),
                               make_unmerge_transform(make_tuple(
                                   Number<CShuffleNRepeatPerShuffle>{}, // NRepeat per
                                                                        // shuffle repeat
                                   NWave,                               // NWave
                                   NThreadPerSubGroup))), // NThreadPerSubGroup = NPerWmma
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<>{},
                               Sequence<0, 1, 2, 3, 7>{},
                               Sequence<>{},
                               Sequence<4, 5, 6>{}));

        // calculate origin of thread output tensor on global memory
        //     blockwise GEMM c matrix starting index
        const auto c_thread_mtx_on_block = blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0);

        const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
        const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

        const auto m_thread_data_on_block_to_mrepeat_mwave_msubgroup_maccvgprs_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(
                    make_tuple(MRepeat, MWave, LoopAccPerThread, MSubGroup, MConsecutiveAccs))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto n_thread_data_on_block_to_nrepeat_nwave_nthreadpersubgroup_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(NRepeat, NWave, NThreadPerSubGroup))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));

        const auto m_thread_data_on_block_idx =
            m_thread_data_on_block_to_mrepeat_mwave_msubgroup_maccvgprs_adaptor
                .CalculateBottomIndex(make_multi_index(m_thread_data_on_block));

        const auto n_thread_data_on_block_idx =
            n_thread_data_on_block_to_nrepeat_nwave_nthreadpersubgroup_adaptor.CalculateBottomIndex(
                make_multi_index(n_thread_data_on_block));

        // shuffle: threadwise copy C from VGPR to LDS
        auto c_thread_copy_vgpr_to_lds = ThreadwiseTensorSliceTransfer_v1r3<
            AccDataType,
            CShuffleDataType,
            decltype(c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs),
            decltype(c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs),
            ck::tensor_operation::element_wise::PassThrough,
            Sequence<CShuffleMRepeatPerShuffle,
                     I1,
                     LoopAccPerThread,
                     I1,
                     CShuffleNRepeatPerShuffle,
                     I1,
                     I1,
                     MConsecutiveAccs>,
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
            7,
            1, // vector write pixel
            InMemoryDataOperationEnum::Set,
            1,
            true>{c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                  make_multi_index(0,
                                   m_thread_data_on_block_idx[I1],
                                   m_thread_data_on_block_idx[I2],
                                   m_thread_data_on_block_idx[I3],
                                   0,
                                   n_thread_data_on_block_idx[I1],
                                   n_thread_data_on_block_idx[I2],
                                   m_thread_data_on_block_idx[I4]),
                  ck::tensor_operation::element_wise::PassThrough{}};

        // shuffle: blockwise copy C from LDS to global

        auto c_shuffle_block_copy_lds_to_global = [&]() {
            if constexpr(CStoreEnableAsync)
            {
                return ThreadGroupTensorSliceTransferAsync<
                    ThisThreadBlockGrid,
                    Sequence<1,
                             CShuffleMRepeatPerShuffle * MWave * MPerWmma,
                             1,
                             CShuffleNRepeatPerShuffle * NWave * NPerWmma>, // BlockSliceLengths,
                    CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                    Sequence<0, 1, 2, 3>,
                    CShuffleDataType, // typename SrcData,
                    CDataType,        // typename DstData,
                    decltype(c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat),
                    decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                    3,
                    3,
                    CShuffleBlockTransferScalarPerVector_NPerBlock,
                    true,
                    false>{c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
                           make_multi_index(0, 0, 0, 0),
                           c_grid_desc_mblock_mperblock_nblock_nperblock,
                           make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0)};
            }
            else
            {
                return ThreadGroupTensorSliceTransfer_v6r1<
                    ThisThreadBlockGrid,        // ThreadGroup
                    CElementwiseOperation,      // ElementwiseOperation,
                    CGlobalMemoryDataOperation, // DstInMemOp,
                    Sequence<1,
                             CShuffleMRepeatPerShuffle * MWave * MPerWmma,
                             1,
                             CShuffleNRepeatPerShuffle * NWave * NPerWmma>, // BlockSliceLengths,
                    CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                    Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                    CShuffleDataType,     // typename SrcData,
                    CDataType,            // typename DstData,
                    decltype(c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat),
                    decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                    Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                    3,                                              // index_t VectorDim,
                    CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                    true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                    false> // bool ThreadTransferDstResetCoordinateAfterRun>
                    {c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
                     make_multi_index(0, 0, 0, 0),
                     c_grid_desc_mblock_mperblock_nblock_nperblock,
                     make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0),
                     c_element_op};
            }
        }();

        // space filling curve for local reg & global memory
        // space filling curve for threadwise C in VGPR
        constexpr auto sfc_c_vgpr =
            SpaceFillingCurve<Sequence<MRepeat, 1, 1, 1, NRepeat, 1, 1, MConsecutiveAccs>,
                              Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                              Sequence<CShuffleMRepeatPerShuffle,
                                       1,
                                       1,
                                       1,
                                       CShuffleNRepeatPerShuffle,
                                       1,
                                       1,
                                       MConsecutiveAccs>>{};

        // space filling curve for shuffled blockwise C in global mem
        constexpr auto sfc_c_global =
            SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                              Sequence<0, 2, 1, 3>,
                              Sequence<1,
                                       CShuffleMRepeatPerShuffle * MWave * MPerWmma,
                                       1,
                                       CShuffleNRepeatPerShuffle * NWave * NPerWmma>>{};

        constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

        static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

        static_for<0, num_access, 1>{}([&](auto access_id) {
            // make sure it's safe to write to LDS
            if constexpr(EnableWaveGroup == false)
            {
                block_sync_lds();
            }
            else
            {
                barrier_output.signal();
                barrier_output.wait();
            }

            // each thread write its data from VGPR to LDS
            c_thread_copy_vgpr_to_lds.Run(
                c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                c_thread_buf,
                c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                c_shuffle_block_buf);

            // make sure it's safe to read from LDS
            if constexpr(EnableWaveGroup == false)
            {
                block_sync_lds();
            }
            else
            {
                barrier_output.template sync_lds<false>();
            }

            // each block copy its data from LDS to global
            c_shuffle_block_copy_lds_to_global.Run(
                c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
                c_shuffle_block_buf,
                c_grid_desc_mblock_mperblock_nblock_nperblock,
                c_grid_buf);

            if constexpr(access_id < num_access - 1)
            {
                constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                // move on C
                c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                    c_grid_desc_mblock_mperblock_nblock_nperblock, c_global_step);
            }
        });
#else
        ignore = blockwise_gemm;
        ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
        ignore = block_work_idx;
        ignore = c_element_op;
        ignore = c_thread_buf;
        ignore = p_shared;
        ignore = c_grid_buf;
        ignore = barrier_output;
#endif
    }
};

} // namespace ck
