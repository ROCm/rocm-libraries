// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_gfx13.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t NumPrefetch,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t K1,
          ck::index_t MPerWmma,
          ck::index_t NPerWmma,
          ck::index_t KPerWmma,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_M_K0_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          bool ABlockLdsAsyncCopy,
          bool AEnableGlobalTRLoad,
          bool AEnableGlobalTiledLoad,
          ck::TensorLoadOption ALoadOption,
          index_t AClusterSize, /* Set when ALoadOption == 1*/
          typename BBlockTransferThreadClusterLengths_N_K0_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          bool BBlockLdsAsyncCopy,
          bool BEnableGlobalTRLoad,
          bool BEnableGlobalTiledLoad,
          ck::TensorLoadOption BLoadOption,
          index_t BClusterSize, /* Set when BLoadOption == 1*/
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          bool CStoreEnableAsync,
          bool EnableWaveGroup,
          ck::LoopScheduler LoopSched     = make_default_loop_scheduler(),
          ck::PipelineVersion PipelineVer = ck::PipelineVersion::v5>
struct DeviceGemmWmma_GFX13 : public DeviceGemm<ALayout,
                                                BLayout,
                                                CLayout,
                                                ADataType,
                                                BDataType,
                                                CDataType,
                                                AElementwiseOperation,
                                                BElementwiseOperation,
                                                CElementwiseOperation>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};
    // K1 in gfx13 is used for wmma layout
    static constexpr auto K1Number = Number<K1>{};
    // AVecAccessNumber/BVecAccessNumber is Vector Access Number for A/B
    static constexpr auto AVecAccessNumber = Number<ABlockTransferSrcScalarPerVector>{};
    static constexpr auto BVecAccessNumber = Number<BBlockTransferSrcScalarPerVector>{};

    static constexpr auto MWaves = MPerBlock / (MRepeat * MPerWmma);
    static constexpr auto NWaves = NPerBlock / (NRepeat * NPerWmma);

    static constexpr auto MaxVectorLoadA =
        ABlockTransferSrcScalarPerVector * sizeof(view_type_t<ADataType>) == 16 ? true : false;
    static constexpr auto MaxVectorLoadB =
        BBlockTransferSrcScalarPerVector * sizeof(view_type_t<ADataType>) == 16 ? true : false;

    // If true, LDS is disabled unconditionally
#ifdef GEMM_A_DISABLE_LDS
    static constexpr auto ADisableLds_manu = true;
#else
    static constexpr auto ADisableLds_manu = false;
#endif
#ifdef GEMM_B_DISABLE_LDS
    static constexpr auto BDisableLds_manu = true;
#else
    static constexpr auto BDisableLds_manu = false;
#endif
    static constexpr auto AEnableLds_auto_tmp =
        (NWaves == 1 && (MaxVectorLoadA || MRepeat == 1) &&
         is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            ? false
            : (AEnableGlobalTRLoad ? false : !ADisableLds_manu);
    static constexpr auto AEnableLds_auto =
        (AEnableGlobalTiledLoad || ((ALoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD) ||
                                    (ALoadOption == TensorLoadOption::WGP_MULTICAST_LOAD)))
            ? false
            : AEnableLds_auto_tmp;

    static constexpr auto BEnableLds_auto_tmp =
        (MWaves == 1 && (MaxVectorLoadB || NRepeat == 1) &&
         is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            ? false
            : (BEnableGlobalTRLoad ? false : !BDisableLds_manu);
    static constexpr auto BEnableLds_auto =
        (BEnableGlobalTiledLoad || ((BLoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD) ||
                                    (BLoadOption == TensorLoadOption::WGP_MULTICAST_LOAD)))
            ? false
            : BEnableLds_auto_tmp;

    // If true, LDS is used unconditionally
    // if enable lds async load, should always enable lds
    // TODO : manually set lds enable
    static constexpr auto AEnableLds_manu = ABlockLdsAsyncCopy;
    static constexpr auto BEnableLds_manu = BBlockLdsAsyncCopy;

    static constexpr auto AEnableLds_dds = (ALoadOption == TensorLoadOption::CLUSTER_DDS_LOAD);
    static constexpr auto BEnableLds_dds = (BLoadOption == TensorLoadOption::CLUSTER_DDS_LOAD);
    static constexpr auto AEnableLds_dsTiledload = (ALoadOption == TensorLoadOption::DS_TILED_LOAD);
    static constexpr auto BEnableLds_dsTiledload = (BLoadOption == TensorLoadOption::DS_TILED_LOAD);

    static constexpr auto AEnableLds = AEnableLds_auto || AEnableLds_manu || (NumPrefetch > 1) ||
                                       AEnableLds_dds || AEnableLds_dsTiledload;
    static constexpr auto BEnableLds = BEnableLds_auto || BEnableLds_manu || (NumPrefetch > 1) ||
                                       BEnableLds_dds || BEnableLds_dsTiledload;

    static constexpr auto AEnableTRLoadFromGlobal =
        !AEnableLds && AEnableGlobalTRLoad && is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>;
    static constexpr auto BEnableTRLoadFromGlobal =
        !BEnableLds && BEnableGlobalTRLoad && is_same_v<tensor_layout::gemm::RowMajor, BLayout>;

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};
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
    // Describe how data read from Global memory
#ifdef CK_EXTENSION_MX_TYPE
    static constexpr auto scale_matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerWmma, NPerWmma, 2};
    static auto MakeAScaleGridDescriptor(index_t MRaw,
                                         index_t KRaw) // dimension is (M, ceil(K/128))
    {
        const auto a_scale_grid_desc_m_k = [&]() {
            const auto a_scale_desc_raw =
                make_naive_tensor_descriptor(make_tuple(MRaw, KRaw), make_tuple(KRaw, I1));
            // return scale_matrix_padder.PadADescriptor_M_K(a_scale_desc_raw);
            return a_scale_desc_raw;
        }();

        const auto M = a_scale_grid_desc_m_k.GetLength(I0);
        const auto K = a_scale_grid_desc_m_k.GetLength(I1);
        if constexpr(AEnableLds)
        {
            return a_scale_grid_desc_m_k;
        }
        else
        {
            const auto M0 = M / MPerBlock;
            const auto K0 = K / 2;
            return transform_tensor_descriptor(
                a_scale_grid_desc_m_k,
                make_tuple(make_unmerge_transform(
                               make_tuple(M0 * MRepeat, Number<MWaves>{}, Number<MPerWmma>{})),
                           make_unmerge_transform(make_tuple(K0, I2))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1, 2, 3>{}, Sequence<0, 4>{}));
        }
    }

    static auto MakeBScaleGridDescriptor(index_t KRaw, index_t NRaw)
    {
        const auto b_scale_grid_desc_n_k = [&]() {
            const auto b_scale_desc_raw =
                make_naive_tensor_descriptor(make_tuple(NRaw, KRaw), make_tuple(KRaw, I1));
            // return scale_matrix_padder.PadBDescriptor_N_K(b_scale_desc_raw);
            return b_scale_desc_raw;
        }();

        const auto N = b_scale_grid_desc_n_k.GetLength(I0);
        const auto K = b_scale_grid_desc_n_k.GetLength(I1);
        if constexpr(BEnableLds)
        {
            return b_scale_grid_desc_n_k;
        }
        else
        {
            const auto N0 = N / NPerBlock;
            const auto K0 = K / 2;
            return transform_tensor_descriptor(
                b_scale_grid_desc_n_k,
                make_tuple(make_unmerge_transform(
                               make_tuple(N0 * NRepeat, Number<NWaves>{}, Number<NPerWmma>{})),
                           make_unmerge_transform(make_tuple(K0, I2))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1, 2, 3>{}, Sequence<0, 4>{}));
        }
    }
#endif

    static auto MakeAGridDescriptor(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                if constexpr(mx_type_enable) // for mx type, the desriptor is
                                             // int32 based
                {
                    const auto a_grid_desc_mraw_kraw = make_naive_tensor_descriptor(
                        make_tuple(MRaw, KRaw * ADataType::BITS / 32),
                        make_tuple(StrideA * ADataType::BITS / 32, I1));
                    // for mx data type, no need to pad which assume data always aligned
                    return a_grid_desc_mraw_kraw;
                }
                else
                {
                    const auto a_grid_desc_mraw_kraw = make_naive_tensor_descriptor(
                        make_tuple(MRaw, KRaw), make_tuple(StrideA, I1));

                    return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
                }
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                if constexpr(mx_type_enable) // for mx type, the desriptor is
                                             // int32 based
                {
                    const auto a_grid_desc_mraw_kraw = make_naive_tensor_descriptor(
                        make_tuple(MRaw * ADataType::BITS / 32, KRaw),
                        make_tuple(I1, StrideA * ADataType::BITS / 32));

                    return a_grid_desc_mraw_kraw;
                }
                else
                {
                    const auto a_grid_desc_mraw_kraw = make_naive_tensor_descriptor(
                        make_tuple(MRaw, KRaw), make_tuple(I1, StrideA));

                    return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
                }
            }
        }();

        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);
        assert(K % ABlockTransferSrcScalarPerVector == 0);

        // for gfx13, the layout changes to M->K0->K1
        if constexpr(AEnableLds)
        {
            const index_t K0 = K / ABlockTransferSrcScalarPerVector;
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_pass_through_transform(M),
                           make_unmerge_transform(make_tuple(K0, AVecAccessNumber))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}));
        }
        else
        {
            if constexpr(AEnableTRLoadFromGlobal)
            {
                constexpr auto M1        = 8; // this value is based on SPG
                const auto A_KWmma       = K / AKPerWmma;
                const auto M0            = M / MPerBlock;
                constexpr auto M1PerWmma = MPerWmma / M1;
                return transform_tensor_descriptor(
                    a_grid_desc_m_k,
                    make_tuple(
                        make_unmerge_transform(make_tuple(M0, MRepeat, MWaves, M1PerWmma, M1)),
                        make_unmerge_transform(make_tuple(A_KWmma, AKPerWmma, I1))),
                    // this I1 is used to match the dimension in other branch where no TR
                    // load; no real meaning, many codes use 7 dimensions
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<1, 2, 3, 4, 7>{}, Sequence<0, 5, 6>{}));
            }
            else
            {
                // TODO, the logic not changed
                constexpr auto A_KRow      = 2;
                constexpr auto A_K0PerWmma = AKPerWmma / A_KRow / K1Number;
                const auto A_KWmma         = K / AKPerWmma;

                const auto M0 = M / MPerBlock;
                // 0   1     0         1                2        3             4        5 6
                // M - K <-> A_KWmma - MBlock*MRepeat - MWaves - A_K0PerWmma - A_KRow -
                // MPerWmma - A_K1
                return transform_tensor_descriptor(
                    a_grid_desc_m_k,
                    make_tuple(make_unmerge_transform(make_tuple(
                                   A_KWmma, Number<A_K0PerWmma>{}, Number<A_KRow>{}, K1Number)),
                               make_unmerge_transform(make_tuple(
                                   M0, Number<MRepeat>{}, Number<MWaves>{}, Number<MPerWmma>{}))),
                    make_tuple(Sequence<1>{}, Sequence<0>{}),
                    make_tuple(Sequence<0, 4, 5, 7>{}, Sequence<1, 2, 3, 6>{}));
            }
        }
    }

    static auto MakeBGridDescriptor(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b_grid_desc_n_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {

                if constexpr(mx_type_enable) // for mx type, the desriptor is
                                             // int32 based
                {
                    const auto b_grid_desc_nraw_kraw = make_naive_tensor_descriptor(
                        make_tuple(NRaw * BDataType::BITS / 32, KRaw),
                        make_tuple(I1, StrideB * BDataType::BITS / 32));

                    return b_grid_desc_nraw_kraw;
                }
                else
                {
                    const auto b_grid_desc_nraw_kraw = make_naive_tensor_descriptor(
                        make_tuple(NRaw, KRaw), make_tuple(I1, StrideB));

                    return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
                }
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                if constexpr(mx_type_enable) // for mx type, the desriptor is
                                             // int32 based
                {
                    const auto b_grid_desc_nraw_kraw = make_naive_tensor_descriptor(
                        make_tuple(NRaw, KRaw * BDataType::BITS / 32),
                        make_tuple(StrideB * BDataType::BITS / 32, I1));

                    return b_grid_desc_nraw_kraw;
                }
                else
                {

                    const auto b_grid_desc_nraw_kraw = make_naive_tensor_descriptor(
                        make_tuple(NRaw, KRaw), make_tuple(StrideB, I1));

                    return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
                }
            }
        }();

        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = b_grid_desc_n_k.GetLength(I1);
        assert(K % BBlockTransferSrcScalarPerVector == 0);
        if constexpr(BEnableLds)
        {
            const index_t K0 = K / BBlockTransferSrcScalarPerVector;
            return transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_pass_through_transform(N),
                           make_unmerge_transform(make_tuple(K0, BVecAccessNumber))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}));
        }
        else
        {
            if constexpr(BEnableTRLoadFromGlobal)
            {
                constexpr auto N1        = 8;
                const auto B_KWmma       = K / BKPerWmma;
                const auto N0            = N / NPerBlock;
                constexpr auto N1PerWmma = NPerWmma / N1;
                return transform_tensor_descriptor(
                    b_grid_desc_n_k,
                    make_tuple(
                        make_unmerge_transform(make_tuple(N0, NRepeat, NWaves, N1PerWmma, N1)),
                        make_unmerge_transform(make_tuple(B_KWmma, BKPerWmma, I1))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<1, 2, 3, 4, 7>{}, Sequence<0, 5, 6>{}));
            }
            else
            {
                // TODO, not changed
                constexpr auto B_KRow      = 2;
                constexpr auto B_K0PerWmma = BKPerWmma / B_KRow / K1Number;
                const auto B_KWmma         = K / BKPerWmma;

                const auto N0 = N / NPerBlock;
                // 0   1     0         1                2        3             4 5 6 M -
                // K <-> A_KWmma - MBlock*MRepeat - MWaves - A_K0PerWmma - A_KRow -
                // MPerWmma - A_K1
                return transform_tensor_descriptor(
                    b_grid_desc_n_k,
                    make_tuple(make_unmerge_transform(make_tuple(
                                   B_KWmma, Number<B_K0PerWmma>{}, Number<B_KRow>{}, K1Number)),
                               make_unmerge_transform(make_tuple(
                                   N0, Number<NRepeat>{}, Number<NWaves>{}, Number<NPerWmma>{}))),
                    make_tuple(Sequence<1>{}, Sequence<0>{}),
                    make_tuple(Sequence<0, 4, 5, 7>{}, Sequence<1, 2, 3, 6>{}));
            }
        }
    }

    static auto MakeCGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideC)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideC));
            }
        }();

        return matrix_padder.PadCDescriptor_M_N(c_grid_desc_mraw_nraw);
    }

    // Gridwise descriptor, mapping to whole given provblem.
    using AGridDesc     = decltype(MakeAGridDescriptor(1, 1, 1));
    using BGridDesc     = decltype(MakeBGridDescriptor(1, 1, 1));
    using CGridDesc_M_N = decltype(MakeCGridDescriptor_M_N(1, 1, 1));
#ifdef CK_EXTENSION_MX_TYPE
    using AScaleGridDesc = decltype(MakeAScaleGridDescriptor(1, 1));
    using BScaleGridDesc = decltype(MakeBScaleGridDescriptor(1, 1));
#endif
    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_Wmma_GFX13<
        BlockSize,
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        CDataType,
        InMemoryDataOperationEnum::Set,
        AGridDesc,
        BGridDesc,
#ifdef CK_EXTENSION_MX_TYPE
        AScaleGridDesc,
        BScaleGridDesc,
#endif
        CGridDesc_M_N,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        MPerWmma,
        NPerWmma,
        KPerWmma,
        K1,
        MRepeat,
        NRepeat,
        ABlockTransferThreadClusterLengths_M_K0_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector, // for MX type is used in int32 data type
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        AEnableLds,
        ABlockLdsAddExtraM,
        ABlockLdsAsyncCopy,
        AEnableTRLoadFromGlobal,
        AEnableGlobalTiledLoad,
        ALoadOption,
        AClusterSize,
        BBlockTransferThreadClusterLengths_N_K0_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BEnableLds,
        BBlockLdsAddExtraN,
        BBlockLdsAsyncCopy,
        BEnableTRLoadFromGlobal,
        BEnableGlobalTiledLoad,
        BLoadOption,
        BClusterSize,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        CStoreEnableAsync,
        EnableWaveGroup,
        NumPrefetch,
        LoopSched,
        PipelineVer>;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const src_type_t<ADataType>* p_a_grid,
                 const src_type_t<BDataType>* p_b_grid,
                 CDataType* p_c_grid,
                 index_t M,
                 index_t N,
                 index_t K,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 index_t M01,
                 index_t N01,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : p_a_grid_{reinterpret_cast<const view_type_t<ADataType>*>(p_a_grid)},
              p_b_grid_{reinterpret_cast<const view_type_t<BDataType>*>(p_b_grid)},
              p_c_grid_{p_c_grid},
              a_grid_desc_{},
              b_grid_desc_k0_n_k1_{},
              c_grid_desc_m_n_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              MRaw_{M},
              NRaw_{N},
              KRaw_{K}
        {
            a_grid_desc_         = DeviceGemmWmma_GFX13::MakeAGridDescriptor(M, K, StrideA);
            b_grid_desc_k0_n_k1_ = DeviceGemmWmma_GFX13::MakeBGridDescriptor(K, N, StrideB);
            c_grid_desc_m_n_     = DeviceGemmWmma_GFX13::MakeCGridDescriptor_M_N(M, N, StrideC);
            block_2_ctile_map_ =
                GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_, M01, N01);

            if(GridwiseGemm::CheckValidity(
                   a_grid_desc_, b_grid_desc_k0_n_k1_, c_grid_desc_m_n_, block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n_);
            }
        }

#ifdef CK_EXTENSION_MX_TYPE
        Argument(const src_type_t<ADataType>* p_a_grid,
                 const src_type_t<BDataType>* p_b_grid,
                 const int32_t* p_a_scale,
                 const int32_t* p_b_scale,
                 CDataType* p_c_grid,
                 index_t M,
                 index_t N,
                 index_t K,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 index_t M01,
                 index_t N01,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : p_a_grid_{reinterpret_cast<const view_type_t<ADataType>*>(p_a_grid)}, // cast to int32
              p_b_grid_{reinterpret_cast<const view_type_t<BDataType>*>(p_b_grid)},
              p_a_scale_{p_a_scale},
              p_b_scale_{p_b_scale},
              p_c_grid_{p_c_grid},
              a_grid_desc_{},
              b_grid_desc_k0_n_k1_{},
              c_grid_desc_m_n_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              MRaw_{M},
              NRaw_{N},
              KRaw_{K}
        {
            a_grid_desc_         = DeviceGemmWmma_GFX13::MakeAGridDescriptor(M, K, StrideA);
            b_grid_desc_k0_n_k1_ = DeviceGemmWmma_GFX13::MakeBGridDescriptor(K, N, StrideB);
            c_grid_desc_m_n_     = DeviceGemmWmma_GFX13::MakeCGridDescriptor_M_N(M, N, StrideC);
            a_scale_grid_desc_   = DeviceGemmWmma_GFX13::MakeAScaleGridDescriptor(
                M, math::integer_divide_ceil(K, 256) * 2);
            b_scale_grid_desc_ = DeviceGemmWmma_GFX13::MakeBScaleGridDescriptor(
                math::integer_divide_ceil(K, 256) * 2, N);
            block_2_ctile_map_ =
                GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_, M01, N01);

            if(GridwiseGemm::CheckValidity(
                   a_grid_desc_, b_grid_desc_k0_n_k1_, c_grid_desc_m_n_, block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n_);
            }
        }
#endif
#pragma clang diagnostic pop

        //  private:
        const view_type_t<ADataType>* p_a_grid_;
        const view_type_t<BDataType>* p_b_grid_;
        const int32_t* p_a_scale_;
        const int32_t* p_b_scale_;
#ifdef CK_EXTENSION_MX_TYPE
        AScaleGridDesc a_scale_grid_desc_;
        BScaleGridDesc b_scale_grid_desc_;
#endif
        CDataType* p_c_grid_;
        AGridDesc a_grid_desc_;
        BGridDesc b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock;
        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
        // for checking vector load/store
        index_t MRaw_;
        index_t NRaw_;
        index_t KRaw_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGemmWmma_GFX13::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_,
                                            arg.b_grid_desc_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_k0mk1_k0nk1_m0nm1_wmma_v1r1 has invalid "
                    "setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K = [&]() {
                if constexpr(AEnableLds)
                {
                    return arg.a_grid_desc_.GetLength(I1) * arg.a_grid_desc_.GetLength(I2);
                }
                else
                {
                    if constexpr(AEnableTRLoadFromGlobal)
                    {
                        return arg.a_grid_desc_.GetLength(I0) * arg.a_grid_desc_.GetLength(I5) *
                               arg.a_grid_desc_.GetLength(I6);
                    }
                    else
                    {
                        return arg.a_grid_desc_.GetLength(I0) * arg.a_grid_desc_.GetLength(I4) *
                               arg.a_grid_desc_.GetLength(I5) * arg.a_grid_desc_.GetLength(I7);
                    }
                }
            }();
            auto launch_kernel = [&](auto has_main_k_block_loop) {
                const auto kernel_final = [&]() {
#ifdef CK_EXTENSION_MX_TYPE
                    if constexpr(is_mx_type_t_v<ADataType>)
                    {
                        if constexpr(EnableWaveGroup)
                        {
                            const auto kernel = kernel_gemm_mx_wmma_wavegroup<
                                GridwiseGemm,
                                ADataType,
                                BDataType,
                                CDataType,
                                remove_reference_t<DeviceGemmWmma_GFX13::AGridDesc>,
                                remove_reference_t<DeviceGemmWmma_GFX13::BGridDesc>,
                                remove_reference_t<DeviceGemmWmma_GFX13::AScaleGridDesc>,
                                remove_reference_t<DeviceGemmWmma_GFX13::BScaleGridDesc>,
                                remove_reference_t<
                                    typename GridwiseGemm::
                                        CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock>,
                                AElementwiseOperation,
                                BElementwiseOperation,
                                CElementwiseOperation,
                                remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                                has_main_k_block_loop>;

                            return launch_and_time_kernel(
                                stream_config,
                                kernel,
                                dim3(grid_size),
                                dim3(BlockSize),
                                0,
                                arg.p_a_grid_,
                                arg.p_b_grid_,
                                arg.p_a_scale_,
                                arg.p_b_scale_,
                                arg.a_scale_grid_desc_,
                                arg.b_scale_grid_desc_,
                                arg.p_c_grid_,
                                arg.a_grid_desc_,
                                arg.b_grid_desc_k0_n_k1_,
                                arg.c_grid_desc_mblock_mperblock_nblock_nperblock,
                                arg.a_element_op_,
                                arg.b_element_op_,
                                arg.c_element_op_,
                                arg.block_2_ctile_map_);
                        }
                        else
                        {
                            const auto kernel = kernel_gemm_mx_wmma<
                                GridwiseGemm,
                                ADataType,
                                BDataType,
                                CDataType,
                                remove_reference_t<DeviceGemmWmma_GFX13::AGridDesc>,
                                remove_reference_t<DeviceGemmWmma_GFX13::BGridDesc>,
                                remove_reference_t<DeviceGemmWmma_GFX13::AScaleGridDesc>,
                                remove_reference_t<DeviceGemmWmma_GFX13::BScaleGridDesc>,
                                remove_reference_t<
                                    typename GridwiseGemm::
                                        CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock>,
                                AElementwiseOperation,
                                BElementwiseOperation,
                                CElementwiseOperation,
                                remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                                has_main_k_block_loop>;

                            return launch_and_time_kernel(
                                stream_config,
                                kernel,
                                dim3(grid_size),
                                dim3(BlockSize),
                                0,
                                arg.p_a_grid_,
                                arg.p_b_grid_,
                                arg.p_a_scale_,
                                arg.p_b_scale_,
                                arg.a_scale_grid_desc_,
                                arg.b_scale_grid_desc_,
                                arg.p_c_grid_,
                                arg.a_grid_desc_,
                                arg.b_grid_desc_k0_n_k1_,
                                arg.c_grid_desc_mblock_mperblock_nblock_nperblock,
                                arg.a_element_op_,
                                arg.b_element_op_,
                                arg.c_element_op_,
                                arg.block_2_ctile_map_);
                        }
                    }
                    else
#endif
                    {

                        if constexpr(EnableWaveGroup)
                        {
                            /* WGP multicast load handled in this kernel as it needs to enable
                             * wavegroup */
                            const auto kernel = kernel_gemm_wmma_wavegroup<
                                GridwiseGemm,
                                ADataType,
                                BDataType,
                                CDataType,
                                remove_reference_t<DeviceGemmWmma_GFX13::AGridDesc>,
                                remove_reference_t<DeviceGemmWmma_GFX13::BGridDesc>,
                                remove_reference_t<
                                    typename GridwiseGemm::
                                        CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock>,
                                AElementwiseOperation,
                                BElementwiseOperation,
                                CElementwiseOperation,
                                remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                                has_main_k_block_loop>;

                            return launch_and_time_kernel(
                                stream_config,
                                kernel,
                                dim3(grid_size),
                                dim3(BlockSize),
                                0,
                                arg.p_a_grid_,
                                arg.p_b_grid_,
                                arg.p_c_grid_,
                                arg.a_grid_desc_,
                                arg.b_grid_desc_k0_n_k1_,
                                arg.c_grid_desc_mblock_mperblock_nblock_nperblock,
                                arg.a_element_op_,
                                arg.b_element_op_,
                                arg.c_element_op_,
                                arg.block_2_ctile_map_);
                        }
                        else if constexpr((ALoadOption ==
                                           TensorLoadOption::CLUSTER_MULTICAST_LOAD) ||
                                          (ALoadOption == TensorLoadOption::CLUSTER_DDS_LOAD) ||
                                          (ALoadOption ==
                                           TensorLoadOption::CLUSTER_ASYNC_MULTICAST_LDS_LOAD) ||
                                          (BLoadOption ==
                                           TensorLoadOption::CLUSTER_MULTICAST_LOAD) ||
                                          (BLoadOption == TensorLoadOption::CLUSTER_DDS_LOAD) ||
                                          (BLoadOption ==
                                           TensorLoadOption::CLUSTER_ASYNC_MULTICAST_LDS_LOAD))
                        {
                            /* Cluster MulticastLoad */
                            static_assert(ALoadOption == TensorLoadOption::DEFAULT_LOAD ||
                                              ALoadOption == TensorLoadOption::CLUSTER_DDS_LOAD ||
                                              BLoadOption == TensorLoadOption::DEFAULT_LOAD ||
                                              BLoadOption == TensorLoadOption::CLUSTER_DDS_LOAD,
                                          "Either A or B should not be Cluster MulticastLoad.");

                            const auto kernel = kernel_gemm_wmma_cluster<
                                GridwiseGemm,
                                ADataType,
                                BDataType,
                                CDataType,
                                remove_reference_t<DeviceGemmWmma_GFX13::AGridDesc>,
                                remove_reference_t<DeviceGemmWmma_GFX13::BGridDesc>,
                                remove_reference_t<
                                    typename GridwiseGemm::
                                        CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock>,
                                AElementwiseOperation,
                                BElementwiseOperation,
                                CElementwiseOperation,
                                remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                                AClusterSize,
                                BClusterSize,
                                has_main_k_block_loop>;

                            return launch_and_time_kernel(
                                stream_config,
                                kernel,
                                dim3(grid_size),
                                dim3(BlockSize),
                                0,
                                arg.p_a_grid_,
                                arg.p_b_grid_,
                                arg.p_c_grid_,
                                arg.a_grid_desc_,
                                arg.b_grid_desc_k0_n_k1_,
                                arg.c_grid_desc_mblock_mperblock_nblock_nperblock,
                                arg.a_element_op_,
                                arg.b_element_op_,
                                arg.c_element_op_,
                                arg.block_2_ctile_map_);
                        }
                        else
                        {
                            const auto kernel = kernel_gemm_wmma_gfx13<
                                GridwiseGemm,
                                ADataType,
                                BDataType,
                                CDataType,
                                remove_reference_t<DeviceGemmWmma_GFX13::AGridDesc>,
                                remove_reference_t<DeviceGemmWmma_GFX13::BGridDesc>,
                                remove_reference_t<
                                    typename GridwiseGemm::
                                        CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock>,
                                AElementwiseOperation,
                                BElementwiseOperation,
                                CElementwiseOperation,
                                remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                                has_main_k_block_loop>;

                            return launch_and_time_kernel(
                                stream_config,
                                kernel,
                                dim3(grid_size),
                                dim3(BlockSize),
                                0,
                                arg.p_a_grid_,
                                arg.p_b_grid_,
                                arg.p_c_grid_,
                                arg.a_grid_desc_,
                                arg.b_grid_desc_k0_n_k1_,
                                arg.c_grid_desc_mblock_mperblock_nblock_nperblock,
                                arg.a_element_op_,
                                arg.b_element_op_,
                                arg.c_element_op_,
                                arg.block_2_ctile_map_);
                        }
                    }
                };

                return kernel_final();
            };

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(ck::is_gfx11_supported() || ck::is_gfx12_supported() || ck::is_gfx13_supported())
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, ck::half_t> ||
                           is_same_v<AccDataType, int32_t>))
            {
                printf("DeviceOp err: AccDataType\n");
                return false;
            }
        }
        else
        {
            printf("DeviceOp err: Arch\n");
            return false;
        }

        // check vector load/store
        {
            using Row = ck::tensor_layout::gemm::RowMajor;
            using Col = ck::tensor_layout::gemm::ColumnMajor;

            // check vector load of A
            if constexpr(is_same_v<ALayout, Row> && ABlockTransferSrcVectorDim == 2)
            {
                if(arg.KRaw_ % ABlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
            // because change to gfx13, the layout changes to M->K0->K1
            else if constexpr(is_same_v<ALayout, Col> && ABlockTransferSrcVectorDim == 0)
            {
                // FIXME: not rigorous
                if(arg.MRaw_ % ABlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            // check vector laod of B
            if constexpr(is_same_v<BLayout, Col> && BBlockTransferSrcVectorDim == 2)
            {
                if(arg.KRaw_ % BBlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
            else if constexpr(is_same_v<BLayout, Row> && BBlockTransferSrcVectorDim == 0)
            {
                // FIXME: not rigorous
                if(arg.NRaw_ % BBlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            // check vector store of C
            // only support RowMajor for now
            if constexpr(is_same_v<CLayout, Row>)
            {
                if(arg.NRaw_ % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }

        return GridwiseGemm::CheckValidity(arg.a_grid_desc_,
                                           arg.b_grid_desc_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }
#ifdef CK_EXTENSION_MX_TYPE
    static auto MakeArgument(const src_type_t<ADataType>* p_a,
                             const src_type_t<BDataType>* p_b,
                             const int32_t* p_a_scale,
                             const int32_t* p_b_scale,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_a_scale,
                        p_b_scale,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        1,
                        1,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }
#endif
    static auto MakeArgument(const src_type_t<ADataType>* p_a,
                             const src_type_t<BDataType>* p_b,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        1,
                        1,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const src_type_t<ADataType>*>(p_a),
                                          static_cast<const src_type_t<BDataType>*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          1,
                                          1,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
    }

#ifdef CK_EXTENSION_MX_TYPE
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      const int32_t* p_a_scale,
                                                      const int32_t* p_b_scale,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const src_type_t<ADataType>*>(p_a),
                                          static_cast<const src_type_t<BDataType>*>(p_b),
                                          static_cast<const int32_t*>(p_a_scale),
                                          static_cast<const int32_t*>(p_b_scale),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          1,
                                          1,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
    }
#endif
    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<LoopScheduler, std::string> LoopSchedToString{
            {LoopScheduler::Default, "Default"}, {LoopScheduler::Interwave, "Interwave"}};

        std::map<PipelineVersion, std::string> PipelineVersionToString{
            {PipelineVersion::v1, "v1"}, {PipelineVersion::v2, "v2"}, {PipelineVersion::v5, "v5"}};

        std::map<TensorLoadOption, std::string> LoadMethodToString{
            {TensorLoadOption::DEFAULT_LOAD, "default"},
            {TensorLoadOption::CLUSTER_MULTICAST_LOAD, "cluster_multicast"},
            {TensorLoadOption::WGP_MULTICAST_LOAD, "wgp_multicast"},
            {TensorLoadOption::CLUSTER_DDS_LOAD, "cluster_dds_load"},
            {TensorLoadOption::CLUSTER_ASYNC_MULTICAST_LDS_LOAD, "cluster_async_multicast_load"}};

        // clang-format off
        str << "DeviceGemmWmma_GFX13"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << K1 << ", "
            << MPerWmma << ", "
            << NPerWmma << ", "
            << KPerWmma << ", "
            << MRepeat << ", "
            << NRepeat
            << ">"
            << " AEnableLds: "
            << AEnableLds << ", "
            << "BEnableLds: "
            << BEnableLds << ", "
            << "AEnableTRload: "
            << AEnableTRLoadFromGlobal << ", "
            << "AEnableTiledload: "
            << AEnableGlobalTiledLoad << ", "
            << "ALoadOption: "
            << LoadMethodToString[ALoadOption] << ", "
            << "BEnableTRload: "
            << BEnableTRLoadFromGlobal << ", "
            << "BEnableTiledload: "
            << BEnableGlobalTiledLoad << ", "
            << "BLoadOption: "
            << LoadMethodToString[BLoadOption] << ", "
            << "CStoreEnableAsync: "
            << CStoreEnableAsync << ", "
            << "EnableWaveGroup: "
            << EnableWaveGroup << ", "
            << "NumPrefetch: "
            << NumPrefetch << ", "
            << "LoopScheduler: "
            << LoopSchedToString[LoopSched] << ", "
            << "PipelineVersion: "
            << PipelineVersionToString[PipelineVer];
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
