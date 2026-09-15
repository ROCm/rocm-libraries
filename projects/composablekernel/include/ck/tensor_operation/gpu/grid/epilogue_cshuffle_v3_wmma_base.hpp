// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r3.hpp"

namespace ck {

template <typename DsDataType,
          typename EDataType,
          typename AccDataType,
          typename CShuffleDataType,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          typename CDEElementwiseOperation,
          typename ThisThreadBlock,
          typename BlockwiseGemmPipe>
struct EpilogueCShuffleBase
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};

    static constexpr index_t NumDTensor = DsDataType::Size();
    static constexpr auto EShuffleBlockTransferScalarPerVector =
        CDEShuffleBlockTransferScalarPerVectors{}[I0];

    using SpaceFillingCurveVgpr =
        SpaceFillingCurve<Sequence<MRepeat, 1, 1, NRepeat, 1, 1, BlockwiseGemmPipe::MAccVgprs>,
                          Sequence<0, 1, 2, 3, 4, 5, 6>,
                          Sequence<CShuffleMRepeatPerShuffle,
                                   1,
                                   1,
                                   CShuffleNRepeatPerShuffle,
                                   1,
                                   1,
                                   BlockwiseGemmPipe::MAccVgprs>>;

    using SpaceFillingCurveVmem = SpaceFillingCurve<
        Sequence<1, MPerBlock, 1, NPerBlock>,
        Sequence<0, 2, 1, 3>,
        Sequence<1,
                 CShuffleMRepeatPerShuffle * BlockwiseGemmPipe::MWaves * MPerWmma,
                 1,
                 CShuffleNRepeatPerShuffle * BlockwiseGemmPipe::NWaves * NPerWmma>>;

    __device__ static constexpr bool IsLDSNeeded() { return true; }

    // *Caution Here repeat is shuffle repeat
    __device__ static constexpr auto
    GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat()
    {
        constexpr index_t MWaves = MPerBlock / (MRepeat * MPerWmma);
        constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWmma);

        constexpr index_t NPerShRepeat = CShuffleNRepeatPerShuffle * NWaves * NPerWmma;
        constexpr index_t MPerShRepeat = CShuffleMRepeatPerShuffle * MWaves * MPerWmma;

        // T1-01 (GFX1250_CONV_OPTIMIZATION_ROADMAP.md): an unpadded (tile-linear) row stride of
        // NPerShRepeat elements collapses to the same LDS bank for every row whenever
        // NPerShRepeat * sizeof(CShuffleDataType) is a multiple of the gfx1250 LDS bank period
        // (64 banks x 4 bytes = 256 bytes) -- NPerShRepeat is typically a power-of-2 multiple of
        // 64, so this is the common case, producing a maximal 64-way bank conflict on every
        // VGPR<->LDS access in this epilogue. Pad the row stride by 16 bytes in that case
        // (MISA-measured conflict-free pad: gcd(stride_dwords, 64) == 4) so consecutive rows land
        // on different banks. Only the row stride is padded; the logical extents (and therefore
        // every downstream unmerge into MRepeat/MWave/.../NThreadPerSubGroup) are unchanged.
        // The pad is gated on the conflict actually being present: an unconditional pad turns a
        // free (shift-based) row-stride multiply into an explicit multiply instruction even for
        // tile configs whose natural stride was never bank-period-aligned, which is a pure cost
        // with no offsetting benefit for those configs (measured regression without this gate).
        constexpr index_t LdsBankPeriodBytes = 256; // 64 banks x 4 bytes on gfx1250
        constexpr index_t RowStrideBytes     = NPerShRepeat * sizeof(CShuffleDataType);
        constexpr bool RowStrideConflictProne = (RowStrideBytes % LdsBankPeriodBytes) == 0;
        constexpr index_t LdsRowPadElems =
            RowStrideConflictProne ? (16 / sizeof(CShuffleDataType)) : 0;
        constexpr index_t NPerShRepeatPadded = NPerShRepeat + LdsRowPadElems;

        constexpr auto c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
            make_naive_tensor_descriptor(
                make_tuple(I1, Number<MPerShRepeat>{}, I1, Number<NPerShRepeat>{}),
                make_tuple(Number<MPerShRepeat * NPerShRepeatPadded>{},
                           Number<NPerShRepeatPadded>{},
                           Number<NPerShRepeatPadded>{},
                           I1));

        return c_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat;
    }

    __device__ static constexpr auto GetCShuffleLDSDescriptor()
    {
        // C mapping in single block
        constexpr auto c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp =
            BlockwiseGemmPipe::
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
        constexpr auto MAccVgprs =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I6);

        return transform_tensor_descriptor(
            GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat(),
            make_tuple(make_freeze_transform(I0),
                       make_unmerge_transform(make_tuple(
                           Number<CShuffleMRepeatPerShuffle>{}, // MRepeat per shuffle repeat
                           MWave,                               // MWave
                           MSubGroup,                           // MSubGroup * MAccVgprs = MPerWmma
                           MAccVgprs)),
                       make_freeze_transform(I0),
                       make_unmerge_transform(make_tuple(
                           Number<CShuffleNRepeatPerShuffle>{}, // NRepeat per shuffle repeat
                           NWave,                               // NWave
                           NThreadPerSubGroup))),               // NThreadPerSubGroup = NPerWmma
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<>{}, Sequence<0, 1, 2, 6>{}, Sequence<>{}, Sequence<3, 4, 5>{}));
    }

    __device__ static auto GetVgprToLDSEpilogueDescriptor()
    {
        // C mapping in single block
        constexpr auto c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp =
            BlockwiseGemmPipe::
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
        constexpr auto MAccVgprs =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I6);

        // calculate origin of thread output tensor on global memory
        //     blockwise GEMM c matrix starting index
        const auto c_thread_mtx_on_block =
            BlockwiseGemmPipe::CalculateCThreadOriginDataIndex(I0, I0);

        const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
        const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

        const auto m_thread_data_on_block_to_mrepeat_mwave_msubgroup_maccvgprs_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(MRepeat, MWave, MSubGroup, MAccVgprs))),
                make_tuple(Sequence<0, 1, 2, 3>{}),
                make_tuple(Sequence<0>{}));

        const auto m_thread_data_on_block_idx =
            m_thread_data_on_block_to_mrepeat_mwave_msubgroup_maccvgprs_adaptor
                .CalculateBottomIndex(make_multi_index(m_thread_data_on_block));

        const auto n_thread_data_on_block_to_nrepeat_nwave_nthreadpersubgroup_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(NRepeat, NWave, NThreadPerSubGroup))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));

        const auto n_thread_data_on_block_idx =
            n_thread_data_on_block_to_nrepeat_nwave_nthreadpersubgroup_adaptor.CalculateBottomIndex(
                make_multi_index(n_thread_data_on_block));

        return ThreadwiseTensorSliceTransfer_v1r3<
            AccDataType,
            CShuffleDataType,
            decltype(BlockwiseGemmPipe::
                         GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs()),
            decltype(GetCShuffleLDSDescriptor()),
            ck::tensor_operation::element_wise::PassThrough,
            Sequence<CShuffleMRepeatPerShuffle,
                     I1,
                     I1,
                     CShuffleNRepeatPerShuffle,
                     I1,
                     I1,
                     MAccVgprs>,
            Sequence<0, 1, 2, 3, 4, 5, 6>,
            6,
            1,
            InMemoryDataOperationEnum::Set,
            1,
            true>{GetCShuffleLDSDescriptor(),
                  make_multi_index(0,
                                   m_thread_data_on_block_idx[I1],
                                   m_thread_data_on_block_idx[I2],
                                   0,
                                   n_thread_data_on_block_idx[I1],
                                   n_thread_data_on_block_idx[I2],
                                   m_thread_data_on_block_idx[I3]),
                  ck::tensor_operation::element_wise::PassThrough{}};
    }

    template <InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              typename InterDataType,
              typename CDsDescRefs,
              typename EGridDesc>
    __device__ static auto
    GetLDSToVmemEpilogueDescriptor(CDsDescRefs& c_ds_desc_refs,
                                   EGridDesc& e_grid_desc_mblock_mperblock_nblock_nperblock,
                                   CDEElementwiseOperation& cde_element_op,
                                   const index_t& block_m_id,
                                   const index_t& block_n_id)
    {
        // tuple of starting index of C/Ds blockwise copy
        const auto idx_c_ds_block_begin = container_concat(
            make_tuple(make_multi_index(0, 0, 0, 0)),
            generate_tuple([&](auto) { return make_multi_index(block_m_id, 0, block_n_id, 0); },
                           Number<NumDTensor>{}));

        // blockwise copy which loads C from LDS, D from global, applies elementwise
        // operation and stores result E to global
        return ThreadGroupTensorSliceTransfer_v7r3<
            ThisThreadBlock, // ThreadGroup
            decltype(container_concat(make_tuple(CShuffleDataType{}), DsDataType{})),
            Tuple<EDataType>,
            CDsDescRefs,
            decltype(tie(e_grid_desc_mblock_mperblock_nblock_nperblock)),
            CDEElementwiseOperation,                                    // ElementwiseOperation,
            Sequence<static_cast<index_t>(EGlobalMemoryDataOperation)>, // DstInMemOps,
            Sequence<1,
                     CShuffleMRepeatPerShuffle * BlockwiseGemmPipe::MWaves * MPerWmma,
                     1,
                     CShuffleNRepeatPerShuffle * BlockwiseGemmPipe::NWaves *
                         NPerWmma>, // BlockSliceLengths,
            CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
            Sequence<0, 1, 2, 3>,                    // ThreadClusterArrangeOrder,
            Sequence<0, 1, 2, 3>,                    // SrcDimAccessOrder,
            Sequence<0, 1, 2, 3>,                    // DstDimAccessOrder,
            3,                                       // SrcVectorDim,
            3,                                       // DstVectorDim,
            CDEShuffleBlockTransferScalarPerVectors, // SrcScalarPerVectors
            EShuffleBlockTransferScalarPerVector,    // DstScalarPerVector
            sequence_merge_t<
                Sequence<true>,
                uniform_sequence_gen_t<NumDTensor,
                                       false>>, // ThreadTransferSrcResetCoordinateAfterRunFlags
            Sequence<false>,                    // ThreadTransferDstResetCoordinateAfterRunFlags
            1,
            Tuple<InterDataType>>{c_ds_desc_refs,
                                  idx_c_ds_block_begin,
                                  tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                                  make_tuple(make_multi_index(block_m_id, 0, block_n_id, 0)),
                                  cde_element_op};
    }
};

} // namespace ck
