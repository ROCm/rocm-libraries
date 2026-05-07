// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/wmma_gemm.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#define CK_MNK_LOOP

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck {

#if defined(__gfx13__)
template <typename ThisThreadBlock,
          typename FloatA,
          typename FloatB,
          typename FloatAcc,
          typename ABlockDesc,
          typename BBlockDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWMMA,
          index_t NPerWMMA,
          index_t KPerWMMA,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          bool AEnableLds              = true,
          bool BEnableLds              = true,
          bool TransposeC              = false,
          bool APermute                = false, // APermute and BPermute are used for gfx13
          bool BPermute                = false,
          TensorLoadOption ALoadOption = TensorLoadOption::DEFAULT_LOAD,
          TensorLoadOption BLoadOption = TensorLoadOption::DEFAULT_LOAD>
/* Option: Read from LDS, big buffer hold all threads required data
 * Source
 * A: K0PerBlock x MPerBlock x K1
 * B: K0PerBlock x NPerBlock x K1
 * Destination
 * C, non-transpose
 * thread level: MRepeat x NRepeat x MAccVgprs
 * block  level: MRepeat x MWave x MSubGroup x NRepeat x NWave x NThreadPerSubGroup x MAccVgprs
 * KPACK == WMMA_K = 16
 *
 * Option: Read from VMEM, small buffer hold each thread own required data (Skip LDS)
 * Source:
 * A(if skip LDS): MRepeat x KPack
 * B(if skip LDS): NRepeat x KPack
 * Destination
 * C, non-transpose
 * block level: MRepeat x MWave x MSubGroup x NRepeat x NWave x NThreadPerSubGroup x MAccVgprs
 */
struct BlockwiseGemmWMMA
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto WmmaK = Number<KPerWMMA>{};

    // Hardcode of WaveSize, since current HIP Runtime(5.4.0-10984) could not return correct one.
    static constexpr index_t WaveSize = 32;

    // When use LDS, each Row(16 consecutive lanes) read whole data from source buffer
    // When not use LDS, each Row read half of whole data from source buffer, exchange the data via
    // permutation
    static constexpr index_t A_KRow = 2;
    static constexpr index_t B_KRow = 2;

    static constexpr index_t A_K1 = ABlockDesc{}.GetLength(I5);
    static constexpr index_t B_K1 = BBlockDesc{}.GetLength(I5);

    static constexpr bool EnableWaveGroup = ThisThreadBlock::InWaveGroup();
    static constexpr auto wmma_gemm       = WmmaGemm<FloatA,
                                                     FloatB,
                                                     FloatAcc,
                                                     MPerWMMA,
                                                     NPerWMMA,
                                                     KPerWMMA,
                                                     KPack,
                                                     TransposeC,
                                                     EnableWaveGroup>{};

    static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerWMMA);
    static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWMMA);

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              FloatAcc,
                              MRepeat * NRepeat,
                              wmma_gemm.GetRegSizePerWmma(),
                              true>
        c_thread_buf_;

    __host__ __device__ constexpr auto& GetCThreadBuffer() { return c_thread_buf_; }

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = ThisThreadBlock::GetThreadId();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MWaves, NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    // Default, Block buffer in LDS, thread level offset enabled
    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        if constexpr(AEnableLds)
        {
            const auto wave_idx = GetWaveIdx();
            const auto waveId_m = wave_idx[I0];

            if constexpr(ALoadOption == TensorLoadOption::DS_TILED_LOAD)
            {
                return make_tuple(
                    0, 0, waveId_m, 0, get_lane_id() % 16, 0); // 16==32/ThreadsPerTile
            }
            else
            {
                const auto WMMA_a_idx = wmma_gemm.CalculateAThreadOriginDataIndex();
                //               |KRepeat   |MRepeat|MWave    |KRow                       |MLane
                //               |KPack
                return make_tuple(0, 0, waveId_m, wmma_gemm.GetSubGroupId(), WMMA_a_idx, 0);
            }
        }
        else
        {
            return make_tuple(0, 0, 0, 0, 0, 0);
        }
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        if constexpr(BEnableLds)
        {
            const auto wave_idx = GetWaveIdx();
            const auto waveId_n = wave_idx[I1];

            if constexpr(BLoadOption == TensorLoadOption::DS_TILED_LOAD)
            {
                return make_tuple(
                    0, 0, waveId_n, 0, get_lane_id() % 16, 0); // 16==32/ThreadsPerTile
            }
            else
            {
                const auto WMMA_b_idx = wmma_gemm.CalculateBThreadOriginDataIndex();
                //  |KRepeat   |NRepeat|Nwave     |KRow  |NLane  |KPack
                return make_tuple(0, 0, waveId_n, wmma_gemm.GetSubGroupId(), WMMA_b_idx, 0);
            }
        }
        else
        {
            return make_tuple(0, 0, 0, 0, 0, 0);
        }
    }

    template <index_t m0, index_t n0>
    __device__ static auto CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = wmma_gemm.GetBeginOfThreadBlk();

        constexpr auto mrepeat_mwave_mperWMMA_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerWMMA))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto nrepeat_nwave_nperWMMA_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerWMMA))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        const index_t c_thread_m = mrepeat_mwave_mperWMMA_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_idx[I0]))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperWMMA_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_idx[I1]))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    template <index_t m0, index_t n0>
    __device__ static auto CalculateCThreadOriginDataIndex7D(Number<m0>, Number<n0>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = wmma_gemm.GetBeginOfThreadBlk3D();

        return make_tuple(Number<m0>{},
                          waveId_m,
                          blk_idx[I0],
                          Number<n0>{},
                          waveId_n,
                          I0,
                          blk_idx[I1],
                          blk_idx[I2]);
    }

    using Tuple6 = decltype(CalculateAThreadOriginDataIndex());
    __host__ __device__ BlockwiseGemmWMMA(Tuple6 a_origin = CalculateAThreadOriginDataIndex(),
                                          Tuple6 b_origin = CalculateBThreadOriginDataIndex())
        : a_thread_copy_(a_origin), b_thread_copy_(b_origin)
    {
        static_assert(ABlockDesc::IsKnownAtCompileTime() && BBlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerWMMA * MRepeat) == 0 &&
                          NPerBlock % (NPerWMMA * NRepeat) == 0,
                      "wrong!");
    }

    // transposed WMMA output C' = B' * A'
    __host__ __device__ static constexpr auto
    GetCThreadDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs()
    {
        constexpr auto c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens =
            wmma_gemm.GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths();
        constexpr auto NAccVgprsLoops   = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I1];
        constexpr auto NAccVgprsPerLoop = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I2];
        return make_naive_tensor_descriptor_packed(
            //        |MRepeat            |MWave |MSubGroup |NRepeat           |NWave
            //        |NThreadPerSubGroup |MAccVgprs
            make_tuple(Number<MRepeat>{},
                       I1,
                       I1,
                       Number<NRepeat>{},
                       I1,
                       NAccVgprsLoops,
                       I1,
                       NAccVgprsPerLoop));
    }

    // Thread level, register decriptor. Vector-write

    __host__ __device__ static constexpr auto
    GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs()
    {
        constexpr auto c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens =
            wmma_gemm.GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths();
        constexpr auto MLoopAcc  = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I1];
        constexpr auto MAccVgprs = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I2];
        constexpr auto AccStride = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I3];
        return make_naive_tensor_descriptor(
            // |  MRepeat  |  MWave  |  MLoopAcc  | MSubGroup  |  NRepeat  |
            // |  NWave  |  NThreadPerSubGroup  |  MLoopAcc  |

            make_tuple(Number<MRepeat>{}, I1, MLoopAcc, I1, Number<NRepeat>{}, I1, I1, MAccVgprs),
            make_tuple(Number<NRepeat>{} * MLoopAcc * MAccVgprs * AccStride,
                       Number<NRepeat>{} * MLoopAcc * MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       Number<NRepeat>{} * MLoopAcc * MAccVgprs * AccStride,
                       MLoopAcc * MAccVgprs * AccStride,
                       MLoopAcc * MAccVgprs * AccStride,
                       MLoopAcc * MAccVgprs * AccStride,
                       AccStride));
    }

    template <typename CGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
        const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto c_grid_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma =
            transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(
                    make_unmerge_transform(make_tuple(M / (MWaves * MPerWMMA), MWaves, MPerWMMA)),
                    make_unmerge_transform(make_tuple(N / (NWaves * NPerWMMA), NWaves, NPerWMMA))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5>{}));

        return wmma_gemm
            .MakeCDesc_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
                c_grid_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma);
    }

    // transposed WMMA output C' = B' * A'
    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs()
    {
        constexpr auto c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<MPerWMMA>{},
                                                           Number<NRepeat>{},
                                                           Number<NWaves>{},
                                                           Number<NPerWMMA>{}));

        return wmma_gemm
            .MakeCDesc_MBlockxRepeat_MWave_MThreadPerSubGroup_NBlockxRepeat_NWave_NSubGroup_NAccVgprs(
                c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma);
    }

    // Provide dimension size
    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs()
    {
        constexpr auto c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<MPerWMMA>{},
                                                           Number<NRepeat>{},
                                                           Number<NWaves>{},
                                                           Number<NPerWMMA>{}));

        return wmma_gemm
            .MakeCDesc_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
                c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma);
    }

    // Describe how data allocated in thread copy src buffer
    // M0_M1_M2 = MRepeat_MWave_MPerWmma, N0_N1_N2 = NRepeat_NWave_NPerWmma
    static constexpr ABlockDesc a_block_desc_k0_m0_m1_m2_k1;
    static constexpr BBlockDesc b_block_desc_k0_n0_n1_n2_k1;

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>

    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf,
                        const index_t a_share_map_rank_id = 0,
                        const index_t b_share_map_rank_id = 0) const

    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatB>(
            b_thread_desc_.GetElementSpaceSize());

        static_assert(KPack % (A_K1 * A_KRow) == 0, "");
        static_assert(KPack % (B_K1 * B_KRow) == 0, "");

        // basic intrinsic to determine loopover direction
        if constexpr(MRepeat < NRepeat)
        {
            static_for<0, KPerBlock / KPack, 1>{}(
                [&](auto k) { // k=0,1,2 instead of k=0,kpack*1, ...
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        // read A
                        if constexpr(ALoadOption == TensorLoadOption::CLUSTER_DDS_LOAD)
                        {
                            a_thread_copy_.Run(
                                a_block_desc_k0_m0_m1_m2_k1,
                                make_tuple(m0, Number<k * KPack / A_K1 / A_KRow>{}, I0, I0, I0, I0),
                                a_block_buf,
                                a_thread_desc_,
                                make_tuple(m0, I0, I0, I0, I0, I0),
                                a_thread_buf,
                                a_share_map_rank_id);
                        }
                        else
                        {
                            a_thread_copy_.Run(
                                a_block_desc_k0_m0_m1_m2_k1,
                                make_tuple(m0, Number<k * KPack / A_K1 / A_KRow>{}, I0, I0, I0, I0),
                                a_block_buf,
                                a_thread_desc_,
                                make_tuple(m0, I0, I0, I0, I0, I0),
                                a_thread_buf);
                        }

                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            // read B
                            vector_type<FloatA, KPack / A_KRow> a_thread_vec;
                            vector_type<FloatB, KPack / B_KRow> b_thread_vec;

                            if constexpr(BLoadOption == TensorLoadOption::CLUSTER_DDS_LOAD)
                            {
                                b_thread_copy_.Run(
                                    b_block_desc_k0_n0_n1_n2_k1,
                                    make_tuple(
                                        n0, Number<k * KPack / B_K1 / B_KRow>{}, I0, I0, I0, I0),
                                    b_block_buf,
                                    b_thread_desc_,
                                    make_tuple(n0, I0, I0, I0, I0, I0),
                                    b_thread_buf,
                                    b_share_map_rank_id);
                            }
                            else
                            {
                                b_thread_copy_.Run(
                                    b_block_desc_k0_n0_n1_n2_k1,
                                    make_tuple(
                                        n0, Number<k * KPack / B_K1 / B_KRow>{}, I0, I0, I0, I0),
                                    b_block_buf,
                                    b_thread_desc_,
                                    make_tuple(n0, I0, I0, I0, I0, I0),
                                    b_thread_buf);
                            }

                            static_for<0, KPack / A_KRow, 1>{}([&](auto i) {
                                a_thread_vec.template AsType<FloatA>()(i) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0, i / A_K1, 0, 0, 0, i % A_K1))>{}];
                            });

                            static_for<0, KPack / B_KRow, 1>{}([&](auto i) {
                                b_thread_vec.template AsType<FloatB>()(i) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0, i / B_K1, 0, 0, 0, i % B_K1))>{}];
                            });

                            if constexpr(APermute)
                            {
                                vector_type<FloatA, KPack / A_KRow> a_thread_vec_permuted;
                                const uint32_t* pIn = reinterpret_cast<const uint32_t*>(
                                    &(a_thread_vec.template AsType<FloatA>()));
                                uint32_t* pOut = reinterpret_cast<uint32_t*>(
                                    &(a_thread_vec_permuted.template AsType<FloatA>()));
                                constexpr index_t dataInVgpr = DataPerVGPR<FloatA>::value;
                                constexpr index_t sizeInVgpr = KPack / A_KRow / dataInVgpr;
                                // currently only support less or equal to 4(sizeInVgpr)
                                static_for<0, sizeInVgpr, 2>{}([&](auto i) {
                                    uint32_t tmp;
                                    pOut[(i >> 1)] = __builtin_amdgcn_permute_pack_tensor_2src_b64(
                                        &tmp, pIn[i], pIn[i + 1], 0);
                                    pOut[(i >> 1) + 2] = tmp;
                                });

                                static_for<0, KPack / A_KRow, 1>{}([&](auto i) {
                                    a_thread_vec.template AsType<FloatA>()(i) =
                                        a_thread_vec_permuted.template AsType<FloatA>()(i);
                                });
                            }

                            if constexpr(BPermute)
                            {
                                vector_type<FloatB, KPack / B_KRow> b_thread_vec_permuted;
                                const uint32_t* pIn = reinterpret_cast<const uint32_t*>(
                                    &(b_thread_vec.template AsType<FloatB>()));
                                uint32_t* pOut = reinterpret_cast<uint32_t*>(
                                    &(b_thread_vec_permuted.template AsType<FloatB>()));
                                constexpr index_t dataInVgpr = DataPerVGPR<FloatB>::value;
                                constexpr index_t sizeInVgpr = KPack / B_KRow / dataInVgpr;
                                // currently only support less or equal to 4(sizeInVgpr)
                                static_for<0, sizeInVgpr, 2>{}([&](auto i) {
                                    uint32_t tmp;
                                    pOut[(i >> 1)] = __builtin_amdgcn_permute_pack_tensor_2src_b64(
                                        &tmp, pIn[i], pIn[i + 1], 0);
                                    pOut[(i >> 1) + 2] = tmp;
                                });

                                static_for<0, KPack / B_KRow, 1>{}([&](auto i) {
                                    b_thread_vec.template AsType<FloatB>()(i) =
                                        b_thread_vec_permuted.template AsType<FloatB>()(i);
                                });
                            }

                            using wmma_input_type_a =
                                typename vector_type<FloatA, WmmaK / A_KRow>::type;
                            using wmma_input_type_b =
                                typename vector_type<FloatB, WmmaK / B_KRow>::type;

                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                            wmma_gemm.template Run<>(
                                a_thread_vec.template AsType<wmma_input_type_a>(),
                                b_thread_vec.template AsType<wmma_input_type_b>(),
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        });
                    });
                });
        }
        else
        {
            static_for<0, NRepeat, 1>{}([&](auto n0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, KPerBlock / KPack, 1>{}([&](auto k) { // k=0,1,2 instead of
                                                                        // k=0,kpack*1, ..
                        // read B
                        if constexpr(BLoadOption == TensorLoadOption::CLUSTER_DDS_LOAD)
                        {
                            b_thread_copy_.Run(
                                b_block_desc_k0_n0_n1_n2_k1,
                                make_tuple(n0, Number<k * KPack / B_K1 / B_KRow>{}, I0, I0, I0, I0),
                                b_block_buf,
                                b_thread_desc_,
                                make_tuple(n0, I0, I0, I0, I0, I0),
                                b_thread_buf,
                                b_share_map_rank_id);
                        }
                        else
                        {
                            b_thread_copy_.Run(
                                b_block_desc_k0_n0_n1_n2_k1,
                                make_tuple(n0, Number<k * KPack / B_K1 / B_KRow>{}, I0, I0, I0, I0),
                                b_block_buf,
                                b_thread_desc_,
                                make_tuple(n0, I0, I0, I0, I0, I0),
                                b_thread_buf);
                        }

                        // read A
                        if constexpr(ALoadOption == TensorLoadOption::CLUSTER_DDS_LOAD)
                        {
                            a_thread_copy_.Run(
                                a_block_desc_k0_m0_m1_m2_k1,
                                make_tuple(m0, Number<k * KPack / A_K1 / A_KRow>{}, I0, I0, I0, I0),
                                a_block_buf,
                                a_thread_desc_,
                                make_tuple(m0, I0, I0, I0, I0, I0),
                                a_thread_buf,
                                a_share_map_rank_id);
                        }
                        else
                        {
                            a_thread_copy_.Run(
                                a_block_desc_k0_m0_m1_m2_k1,
                                make_tuple(m0, Number<k * KPack / A_K1 / A_KRow>{}, I0, I0, I0, I0),
                                a_block_buf,
                                a_thread_desc_,
                                make_tuple(m0, I0, I0, I0, I0, I0),
                                a_thread_buf);
                        }

                        vector_type<FloatA, KPack / A_KRow> a_thread_vec;
                        vector_type<FloatB, KPack / B_KRow> b_thread_vec;

                        static_for<0, KPack / A_KRow, 1>{}([&](auto i) {
                            a_thread_vec.template AsType<FloatA>()(i) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, i / A_K1, 0, 0, 0, i % A_K1))>{}];
                        });

                        static_for<0, KPack / B_KRow, 1>{}([&](auto i) {
                            b_thread_vec.template AsType<FloatB>()(i) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, i / B_K1, 0, 0, 0, i % B_K1))>{}];
                        });

                        if constexpr(APermute)
                        {
                            vector_type<FloatA, KPack / A_KRow> a_thread_vec_permuted;
                            const uint32_t* pIn = reinterpret_cast<const uint32_t*>(
                                &(a_thread_vec.template AsType<FloatA>()));
                            uint32_t* pOut = reinterpret_cast<uint32_t*>(
                                &(a_thread_vec_permuted.template AsType<FloatA>()));
                            constexpr index_t dataInVgpr = DataPerVGPR<FloatA>::value;
                            constexpr index_t sizeInVgpr = KPack / A_KRow / dataInVgpr;
                            // currently only support less or equal to 4(sizeInVgpr)
                            static_for<0, sizeInVgpr, 2>{}([&](auto i) {
                                uint32_t tmp;
                                pOut[(i >> 1)] = __builtin_amdgcn_permute_pack_tensor_2src_b64(
                                    &tmp, pIn[i], pIn[i + 1], 0);
                                pOut[(i >> 1) + 2] = tmp;
                            });

                            static_for<0, KPack / A_KRow, 1>{}([&](auto i) {
                                a_thread_vec.template AsType<FloatA>()(i) =
                                    a_thread_vec_permuted.template AsType<FloatA>()(i);
                            });
                        }

                        if constexpr(BPermute)
                        {
                            vector_type<FloatB, KPack / B_KRow> b_thread_vec_permuted;
                            const uint32_t* pIn = reinterpret_cast<const uint32_t*>(
                                &(b_thread_vec.template AsType<FloatB>()));
                            uint32_t* pOut = reinterpret_cast<uint32_t*>(
                                &(b_thread_vec_permuted.template AsType<FloatB>()));
                            constexpr index_t dataInVgpr = DataPerVGPR<FloatB>::value;
                            constexpr index_t sizeInVgpr = KPack / B_KRow / dataInVgpr;
                            // currently only support less or equal to 4(sizeInVgpr)
                            static_for<0, sizeInVgpr, 2>{}([&](auto i) {
                                uint32_t tmp;
                                pOut[(i >> 1)] = __builtin_amdgcn_permute_pack_tensor_2src_b64(
                                    &tmp, pIn[i], pIn[i + 1], 0);
                                pOut[(i >> 1) + 2] = tmp;
                            });

                            static_for<0, KPack / B_KRow, 1>{}([&](auto i) {
                                b_thread_vec.template AsType<FloatB>()(i) =
                                    b_thread_vec_permuted.template AsType<FloatB>()(i);
                            });
                        }

                        using wmma_input_type_a =
                            typename vector_type<FloatA, WmmaK / A_KRow>::type;
                        using wmma_input_type_b =
                            typename vector_type<FloatB, WmmaK / B_KRow>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        wmma_gemm.template Run<>(
                            a_thread_vec.template AsType<wmma_input_type_a>(),
                            b_thread_vec.template AsType<wmma_input_type_b>(),
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });
        }
    }

    protected:
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor(
        make_tuple(Number<MRepeat>{}, Number<KPack / A_K1 / A_KRow>{}, I1, I1, I1, Number<A_K1>{}),
        make_tuple(Number<KPack / A_KRow>{},
                   Number<A_K1>{},
                   Number<A_K1>{},
                   Number<A_K1>{},
                   Number<A_K1>{},
                   Number<1>{}));

    static constexpr auto b_thread_desc_ = make_naive_tensor_descriptor(
        make_tuple(Number<NRepeat>{}, Number<KPack / B_K1 / B_KRow>{}, I1, I1, I1, Number<B_K1>{}),
        make_tuple(Number<KPack / B_KRow>{},
                   Number<B_K1>{},
                   Number<B_K1>{},
                   Number<B_K1>{},
                   Number<B_K1>{},
                   Number<1>{}));

    // C[M, N, NumRegWMMA]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, wmma_gemm.GetRegSizePerWmma()));

    template <bool EnableLds, TensorLoadOption LoadOption>
    struct AThreadCopySelector;

    template <TensorLoadOption LoadOption>
    struct AThreadCopySelector<true, LoadOption>
    {
        using type =
            ThreadwiseTensorSliceTransfer_v4<FloatA,
                                             FloatA,
                                             decltype(a_block_desc_k0_m0_m1_m2_k1),
                                             decltype(a_thread_desc_),
                                             Sequence<1, KPack / A_K1 / A_KRow, 1, 1, 1, A_K1>,
                                             Sequence<0, 1, 2, 3, 4, 5>,
                                             5,
                                             A_K1,
                                             A_K1>;
    };

    template <TensorLoadOption LoadOption>
    struct AThreadCopySelector<false, LoadOption>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic_IntraRow<
            FloatA,
            FloatA,
            decltype(a_block_desc_k0_m0_m1_m2_k1),
            decltype(a_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<1, KPack / A_K1 / A_KRow, 1, 1, 1, A_K1>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            A_K1,
            false>;
    };

    template <>
    struct AThreadCopySelector<true, TensorLoadOption::CLUSTER_DDS_LOAD>
    {
        using type = ThreadwiseTensorSliceTransfer_DdsToVgpr<
            FloatA,
            FloatA,
            decltype(a_block_desc_k0_m0_m1_m2_k1),
            decltype(a_thread_desc_),
            Sequence<1, KPack / A_K1 / A_KRow, 1, 1, 1, A_K1>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            A_K1,
            A_K1>;
    };

    template <>
    struct AThreadCopySelector<true, TensorLoadOption::DS_TILED_LOAD>
    {
        static constexpr auto ThreadsPerTile = 2;
        static constexpr auto VgprsPerTile   = 1;
        using type                           = ThreadwiseTensorSliceTransfer_DsTiledLoad<
                                      FloatA,
                                      FloatA,
                                      decltype(a_block_desc_k0_m0_m1_m2_k1),
                                      decltype(a_thread_desc_),
                                      Sequence<1, KPack / A_K1 / A_KRow, 1, 1, 1, A_K1>,
                                      Sequence<0, 1, 2, 3, 4, 5>,
                                      5,
                                      A_K1,
                                      A_K1,
                                      ThreadsPerTile,
                                      VgprsPerTile>;
    };

    template <bool EnableLds, TensorLoadOption LoadOption>
    struct BThreadCopySelector;

    template <TensorLoadOption LoadOption>
    struct BThreadCopySelector<true, LoadOption>
    {
        using type =
            ThreadwiseTensorSliceTransfer_v4<FloatB,
                                             FloatB,
                                             decltype(b_block_desc_k0_n0_n1_n2_k1),
                                             decltype(b_thread_desc_),
                                             Sequence<1, KPack / B_K1 / B_KRow, 1, 1, 1, B_K1>,
                                             Sequence<0, 1, 2, 3, 4, 5>,
                                             5,
                                             B_K1,
                                             B_K1>;
    };

    template <TensorLoadOption LoadOption>
    struct BThreadCopySelector<false, LoadOption>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic_IntraRow<
            FloatB,
            FloatB,
            decltype(b_block_desc_k0_n0_n1_n2_k1),
            decltype(b_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<1, KPack / B_K1 / B_KRow, 1, 1, 1, B_K1>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            B_K1,
            false>;
    };

    template <>
    struct BThreadCopySelector<true, TensorLoadOption::CLUSTER_DDS_LOAD>
    {
        using type = ThreadwiseTensorSliceTransfer_DdsToVgpr<
            FloatB,
            FloatB,
            decltype(b_block_desc_k0_n0_n1_n2_k1),
            decltype(b_thread_desc_),
            Sequence<1, KPack / B_K1 / B_KRow, 1, 1, 1, B_K1>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            B_K1,
            B_K1>;
    };

    typename AThreadCopySelector<AEnableLds, ALoadOption>::type a_thread_copy_;
    typename BThreadCopySelector<BEnableLds, BLoadOption>::type b_thread_copy_;
};

#ifdef CK_EXTENSION_MX_TYPE
template <typename ThisThreadBlock,
          typename FloatA,
          typename FloatB,
          typename FloatAcc,
          typename ABlockDesc,
          typename BBlockDesc,
          typename AScaleBlockDesc,
          typename BScaleBlockDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWMMA,
          index_t NPerWMMA,
          index_t KPerWMMA,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          bool AEnableLds = true,
          bool BEnableLds = true,
          bool TransposeC = false>
struct BlockwiseMXGemmWMMA : public BlockwiseGemmWMMA<ThisThreadBlock,
                                                      FloatA,
                                                      FloatB,
                                                      FloatAcc,
                                                      ABlockDesc,
                                                      BBlockDesc,
                                                      MPerBlock,
                                                      NPerBlock,
                                                      KPerBlock,
                                                      MPerWMMA,
                                                      NPerWMMA,
                                                      KPerWMMA,
                                                      MRepeat,
                                                      NRepeat,
                                                      KPack,
                                                      AEnableLds,
                                                      BEnableLds,
                                                      false,
                                                      false,
                                                      TransposeC>
{
    using PARENT = BlockwiseGemmWMMA<ThisThreadBlock,
                                     FloatA,
                                     FloatB,
                                     FloatAcc,
                                     ABlockDesc,
                                     BBlockDesc,
                                     MPerBlock,
                                     NPerBlock,
                                     KPerBlock,
                                     MPerWMMA,
                                     NPerWMMA,
                                     KPerWMMA,
                                     MRepeat,
                                     NRepeat,
                                     KPack,
                                     AEnableLds,
                                     BEnableLds,
                                     false,
                                     false,
                                     TransposeC>;

    static constexpr auto ScaleK0PerBlock = math::integer_divide_ceil(KPerBlock, 256);

    static constexpr auto AKPack = []() {
        if constexpr(is_mx_type_t_v<FloatA>)
        {
            return FloatA::dwords_per_wmmak;
        }
        else
        {
            return KPack;
        }
    }();
    static constexpr auto BKPack = []() {
        if constexpr(is_mx_type_t_v<FloatB>)
        {
            return FloatB::dwords_per_wmmak;
        }
        else
        {
            return KPack;
        }
    }();

    __device__ static auto CalculateAScaleThreadOriginDataIndex()
    {
        if constexpr(AEnableLds)
        {
            const auto wave_idx   = PARENT::GetWaveIdx();
            const auto waveId_m   = wave_idx[PARENT::I0];
            const auto WMMA_a_idx = PARENT::wmma_gemm.CalculateAThreadOriginDataIndex();
            const auto k1_value   = PARENT::wmma_gemm.GetSubGroupId();
            const auto k0_value   = ThisThreadBlock::GetThreadId() / (64 * 4);
            //   |KRepeat   |MRepeat|MWave  |MLane  |KPack
            return make_tuple(k0_value, 0, waveId_m, WMMA_a_idx, k1_value);
        }
        else
        {
            return make_tuple(0, 0, 0, 0, 0);
        }
    }

    __device__ static auto CalculateBScaleThreadOriginDataIndex()
    {
        if constexpr(BEnableLds)
        {
            const auto wave_idx   = PARENT::GetWaveIdx();
            const auto waveId_n   = wave_idx[PARENT::I1];
            const auto WMMA_b_idx = PARENT::wmma_gemm.CalculateBThreadOriginDataIndex();
            const auto k1_value   = PARENT::wmma_gemm.GetSubGroupId();
            const auto k0_value   = ThisThreadBlock::GetThreadId() / (64 * 4);
            //  |KRepeat   |NRepeat|Nwave  |NLane  |KPack
            return make_tuple(k0_value, 0, waveId_n, WMMA_b_idx, k1_value);
        }
        else
        {
            return make_tuple(0, 0, 0, 0, 0);
        }
    }

    using Tuple5 = decltype(CalculateAScaleThreadOriginDataIndex());
    __host__ __device__
    BlockwiseMXGemmWMMA(Tuple5 a_scale_origin = CalculateAScaleThreadOriginDataIndex(),
                        Tuple5 b_scale_origin = CalculateBScaleThreadOriginDataIndex())
        : BlockwiseGemmWMMA<ThisThreadBlock,
                            FloatA,
                            FloatB,
                            FloatAcc,
                            ABlockDesc,
                            BBlockDesc,
                            MPerBlock,
                            NPerBlock,
                            KPerBlock,
                            MPerWMMA,
                            NPerWMMA,
                            KPerWMMA,
                            MRepeat,
                            NRepeat,
                            KPack,
                            AEnableLds,
                            BEnableLds,
                            false,
                            false,
                            TransposeC>{},
          a_thread_copy_(PARENT::CalculateAThreadOriginDataIndex()),
          b_thread_copy_(PARENT::CalculateBThreadOriginDataIndex()),
          a_scale_thread_copy_(a_scale_origin),
          b_scale_thread_copy_(b_scale_origin)
    {
    }

    static constexpr AScaleBlockDesc a_scale_block_desc_k0_m0_m1_m2_k1;
    static constexpr BScaleBlockDesc b_scale_block_desc_k0_n0_n1_n2_k1;

    template <typename ABlockBuffer,
              typename BBlockBuffer,
              typename AScaleBlockBuffer,
              typename BScaleBlockBuffer,
              typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        const AScaleBlockBuffer& a_scale_block_buf,
                        const BScaleBlockBuffer& b_scale_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {

        // for mx data type use int32 data type
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
            b_thread_desc_.GetElementSpaceSize());

        auto a_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
            a_scale_thread_desc_.GetElementSpaceSize());

        auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
            b_scale_thread_desc_.GetElementSpaceSize());

        static_for<0, ScaleK0PerBlock, 1>{}([&](auto k0) {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                a_scale_thread_copy_.Run(a_scale_block_desc_k0_m0_m1_m2_k1,
                                         make_tuple(k0, m0, PARENT::I0, PARENT::I0, PARENT::I0),
                                         a_scale_block_buf,
                                         a_scale_thread_desc_,
                                         make_tuple(k0, m0, PARENT::I0, PARENT::I0, PARENT::I0),
                                         a_scale_thread_buf);
            });

            static_for<0, NRepeat, 1>{}([&](auto n0) {
                b_scale_thread_copy_.Run(b_scale_block_desc_k0_n0_n1_n2_k1,
                                         make_tuple(k0, n0, PARENT::I0, PARENT::I0, PARENT::I0),
                                         b_scale_block_buf,
                                         b_scale_thread_desc_,
                                         make_tuple(k0, n0, PARENT::I0, PARENT::I0, PARENT::I0),
                                         b_scale_thread_buf);
            });
        });

        if constexpr(MRepeat < NRepeat)
        {
            static_for<0, KPerBlock / KPack, 1>{}([&](auto k) { // k=0,1,2 instead of
                                                                // k=0,kpack*1, ...
                constexpr index_t scale_select_idx = k & 3;
                constexpr index_t k_idx            = k >> 2;
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    // read A
                    a_thread_copy_.Run(
                        PARENT::a_block_desc_k0_m0_m1_m2_k1,
                        make_tuple(m0,
                                   Number<k * AKPack / PARENT::A_K1 / PARENT::A_KRow>{},
                                   PARENT::I0,
                                   PARENT::I0,
                                   PARENT::I0,
                                   PARENT::I0),
                        a_block_buf,
                        a_thread_desc_,
                        make_tuple(m0, PARENT::I0, PARENT::I0, PARENT::I0, PARENT::I0, PARENT::I0),
                        a_thread_buf);

                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        // read B
                        b_thread_copy_.Run(
                            PARENT::b_block_desc_k0_n0_n1_n2_k1,
                            make_tuple(n0,
                                       Number<k * BKPack / PARENT::B_K1 / PARENT::B_KRow>{},
                                       PARENT::I0,
                                       PARENT::I0,
                                       PARENT::I0,
                                       PARENT::I0),
                            b_block_buf,
                            b_thread_desc_,
                            make_tuple(
                                n0, PARENT::I0, PARENT::I0, PARENT::I0, PARENT::I0, PARENT::I0),
                            b_thread_buf);

                        vector_type<int32_t, 8> a_thread_vec{};
                        vector_type<int32_t, 8> b_thread_vec{};

                        uint32_t a_scale_value =
                            a_scale_thread_buf[Number<a_scale_thread_desc_.CalculateOffset(
                                make_tuple(k_idx, m0, PARENT::I0, PARENT::I0, PARENT::I0))>{}];
                        uint32_t b_scale_value =
                            b_scale_thread_buf[Number<b_scale_thread_desc_.CalculateOffset(
                                make_tuple(k_idx, n0, PARENT::I0, PARENT::I0, PARENT::I0))>{}];

                        static_for<0, AKPack / PARENT::A_KRow, 1>{}([&](auto i) {
                            a_thread_vec.template AsType<int32_t>()(i) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(make_tuple(
                                    m0, i / PARENT::A_K1, 0, 0, 0, i % PARENT::A_K1))>{}];
                        });

                        static_for<0, BKPack / PARENT::B_KRow, 1>{}([&](auto i) {
                            b_thread_vec.template AsType<int32_t>()(i) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(make_tuple(
                                    n0, i / PARENT::B_K1, 0, 0, 0, i % PARENT::B_K1))>{}];
                        });

                        using wmma_input_type_a = typename vector_type<int32_t, 8>::type;
                        using wmma_input_type_b = typename vector_type<int32_t, 8>::type;

                        constexpr index_t c_offset =
                            PARENT::c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        if constexpr(scale_select_idx == 0)
                        {
                            PARENT::wmma_gemm.template Run<0, 0>(
                                a_thread_vec.template AsType<wmma_input_type_a>(),
                                b_thread_vec.template AsType<wmma_input_type_b>(),
                                a_scale_value,
                                b_scale_value,
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        }
                        else if constexpr(scale_select_idx == 1)
                        {
                            PARENT::wmma_gemm.template Run<1, 1>(
                                a_thread_vec.template AsType<wmma_input_type_a>(),
                                b_thread_vec.template AsType<wmma_input_type_b>(),
                                a_scale_value,
                                b_scale_value,
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        }
                        else if constexpr(scale_select_idx == 2)
                        {
                            PARENT::wmma_gemm.template Run<2, 2>(
                                a_thread_vec.template AsType<wmma_input_type_a>(),
                                b_thread_vec.template AsType<wmma_input_type_b>(),
                                a_scale_value,
                                b_scale_value,
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        }
                        else
                        {
                            PARENT::wmma_gemm.template Run<3, 3>(
                                a_thread_vec.template AsType<wmma_input_type_a>(),
                                b_thread_vec.template AsType<wmma_input_type_b>(),
                                a_scale_value,
                                b_scale_value,
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        }
                    });
                });
            });
        }
        else
        {
            static_for<0, NRepeat, 1>{}([&](auto n0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, KPerBlock / KPack, 1>{}([&](auto k) { // k=0,1,2 instead of
                                                                        // k=0,kpack*1, ..
                        constexpr index_t k_idx            = k >> 2;
                        constexpr index_t scale_select_idx = k & 0x3;
                        // read B
                        b_thread_copy_.Run(
                            PARENT::b_block_desc_k0_n0_n1_n2_k1,
                            make_tuple(n0,
                                       Number<k * BKPack / PARENT::B_K1 / PARENT::B_KRow>{},
                                       PARENT::I0,
                                       PARENT::I0,
                                       PARENT::I0,
                                       PARENT::I0),
                            b_block_buf,
                            b_thread_desc_,
                            make_tuple(
                                n0, PARENT::I0, PARENT::I0, PARENT::I0, PARENT::I0, PARENT::I0),
                            b_thread_buf);
                        // read A
                        a_thread_copy_.Run(
                            PARENT::a_block_desc_k0_m0_m1_m2_k1,
                            make_tuple(m0,
                                       Number<k * AKPack / PARENT::A_K1 / PARENT::A_KRow>{},
                                       PARENT::I0,
                                       PARENT::I0,
                                       PARENT::I0,
                                       PARENT::I0),
                            a_block_buf,
                            a_thread_desc_,
                            make_tuple(
                                m0, PARENT::I0, PARENT::I0, PARENT::I0, PARENT::I0, PARENT::I0),
                            a_thread_buf);

                        vector_type<int32_t, 8> a_thread_vec{};
                        vector_type<int32_t, 8> b_thread_vec{};

                        static_for<0, AKPack / PARENT::A_KRow, 1>{}([&](auto i) {
                            a_thread_vec.template AsType<int32_t>()(i) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(make_tuple(
                                    m0, i / PARENT::A_K1, 0, 0, 0, i % PARENT::A_K1))>{}];
                        });

                        static_for<0, BKPack / PARENT::B_KRow, 1>{}([&](auto i) {
                            b_thread_vec.template AsType<int32_t>()(i) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(make_tuple(
                                    n0, i / PARENT::B_K1, 0, 0, 0, i % PARENT::B_K1))>{}];
                        });

                        uint32_t a_scale_value =
                            a_scale_thread_buf[Number<a_scale_thread_desc_.CalculateOffset(
                                make_tuple(k_idx, m0, PARENT::I0, PARENT::I0, PARENT::I0))>{}];
                        uint32_t b_scale_value =
                            b_scale_thread_buf[Number<b_scale_thread_desc_.CalculateOffset(
                                make_tuple(k_idx, n0, PARENT::I0, PARENT::I0, PARENT::I0))>{}];

                        using wmma_input_type_a = typename vector_type<int32_t, 8>::type;
                        using wmma_input_type_b = typename vector_type<int32_t, 8>::type;

                        constexpr index_t c_offset =
                            PARENT::c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        if constexpr(scale_select_idx == 0)
                        {
                            PARENT::wmma_gemm.template Run<0, 0>(
                                a_thread_vec.template AsType<wmma_input_type_a>(),
                                b_thread_vec.template AsType<wmma_input_type_b>(),
                                a_scale_value,
                                b_scale_value,
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        }
                        else if constexpr(scale_select_idx == 1)
                        {
                            PARENT::wmma_gemm.template Run<1, 1>(
                                a_thread_vec.template AsType<wmma_input_type_a>(),
                                b_thread_vec.template AsType<wmma_input_type_b>(),
                                a_scale_value,
                                b_scale_value,
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        }
                        else if constexpr(scale_select_idx == 2)
                        {
                            PARENT::wmma_gemm.template Run<2, 2>(
                                a_thread_vec.template AsType<wmma_input_type_a>(),
                                b_thread_vec.template AsType<wmma_input_type_b>(),
                                a_scale_value,
                                b_scale_value,
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        }
                        else
                        {
                            PARENT::wmma_gemm.template Run<3, 3>(
                                a_thread_vec.template AsType<wmma_input_type_a>(),
                                b_thread_vec.template AsType<wmma_input_type_b>(),
                                a_scale_value,
                                b_scale_value,
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        }
                    });
                });
            });
        }
    }

    protected:
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor(make_tuple(Number<MRepeat>{},
                                                Number<AKPack / PARENT::A_K1 / PARENT::A_KRow>{},
                                                PARENT::I1,
                                                PARENT::I1,
                                                PARENT::I1,
                                                Number<PARENT::A_K1>{}),
                                     make_tuple(Number<AKPack / PARENT::A_KRow>{},
                                                Number<PARENT::A_K1>{},
                                                Number<PARENT::A_K1>{},
                                                Number<PARENT::A_K1>{},
                                                Number<PARENT::A_K1>{},
                                                Number<1>{}));

    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor(make_tuple(Number<NRepeat>{},
                                                Number<BKPack / PARENT::B_K1 / PARENT::B_KRow>{},
                                                PARENT::I1,
                                                PARENT::I1,
                                                PARENT::I1,
                                                Number<PARENT::B_K1>{}),
                                     make_tuple(Number<BKPack / PARENT::B_KRow>{},
                                                Number<PARENT::B_K1>{},
                                                Number<PARENT::B_K1>{},
                                                Number<PARENT::B_K1>{},
                                                Number<PARENT::B_K1>{},
                                                Number<1>{}));

    static constexpr auto a_scale_thread_desc_ = make_naive_tensor_descriptor(
        make_tuple(
            Number<ScaleK0PerBlock>{}, Number<MRepeat>{}, PARENT::I1, PARENT::I1, PARENT::I1),
        make_tuple(PARENT::I1, Number<ScaleK0PerBlock>{}, PARENT::I1, PARENT::I1, PARENT::I1));
    static constexpr auto b_scale_thread_desc_ = make_naive_tensor_descriptor(
        make_tuple(
            Number<ScaleK0PerBlock>{}, Number<NRepeat>{}, PARENT::I1, PARENT::I1, PARENT::I1),
        make_tuple(PARENT::I1, Number<ScaleK0PerBlock>{}, PARENT::I1, PARENT::I1, PARENT::I1));

    template <bool EnableLds, TensorLoadOption ALoadOption = TensorLoadOption::DEFAULT_LOAD>
    struct AThreadCopySelector;

    template <>
    struct AThreadCopySelector<true>
    {
        using type = ThreadwiseTensorSliceTransfer_v4<
            int32_t,
            int32_t,
            decltype(PARENT::a_block_desc_k0_m0_m1_m2_k1),
            decltype(a_thread_desc_),
            Sequence<1, AKPack / PARENT::A_K1 / PARENT::A_KRow, 1, 1, 1, PARENT::A_K1>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            PARENT::A_K1,
            PARENT::A_K1>;
    };

    template <>
    struct AThreadCopySelector<false>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic_IntraRow<
            int32_t,
            int32_t,
            decltype(PARENT::a_block_desc_k0_m0_m1_m2_k1),
            decltype(a_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<1, AKPack / PARENT::A_K1 / PARENT::A_KRow, 1, 1, 1, PARENT::A_K1>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            PARENT::A_K1,
            false>;
    };

    template <bool EnableLds, TensorLoadOption BLoadOption = TensorLoadOption::DEFAULT_LOAD>
    struct BThreadCopySelector;

    template <>
    struct BThreadCopySelector<true>
    {
        using type = ThreadwiseTensorSliceTransfer_v4<
            int32_t,
            int32_t,
            decltype(PARENT::b_block_desc_k0_n0_n1_n2_k1),
            decltype(b_thread_desc_),
            Sequence<1, BKPack / PARENT::B_K1 / PARENT::B_KRow, 1, 1, 1, PARENT::B_K1>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            PARENT::B_K1,
            PARENT::B_K1>;
    };

    template <>
    struct BThreadCopySelector<false>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic_IntraRow<
            int32_t,
            int32_t,
            decltype(PARENT::b_block_desc_k0_n0_n1_n2_k1),
            decltype(b_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<1, BKPack / PARENT::B_K1 / PARENT::B_KRow, 1, 1, 1, PARENT::B_K1>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            PARENT::B_K1,
            false>;
    };

    typename AThreadCopySelector<AEnableLds>::type a_thread_copy_;
    typename BThreadCopySelector<BEnableLds>::type b_thread_copy_;

    template <bool EnableLds>
    struct AScaleThreadCopySelector;

    template <>
    struct AScaleThreadCopySelector<true>
    {
        using type = ThreadwiseTensorSliceTransfer_v4<int32_t,
                                                      int32_t,
                                                      decltype(a_scale_block_desc_k0_m0_m1_m2_k1),
                                                      decltype(a_scale_thread_desc_),
                                                      Sequence<1, 1, 1, 1, 1>,
                                                      Sequence<0, 1, 2, 3, 4>,
                                                      4,
                                                      1,
                                                      1>;
    };

    template <>
    struct AScaleThreadCopySelector<false>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic_IntraRow<
            int32_t,
            int32_t,
            decltype(a_scale_block_desc_k0_m0_m1_m2_k1),
            decltype(a_scale_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<1, 1, 1, 1, 1>,
            Sequence<0, 1, 2, 3, 4>,
            4,
            1,
            false>;
    };

    template <bool EnableLds>
    struct BScaleThreadCopySelector;

    template <>
    struct BScaleThreadCopySelector<true>
    {
        using type = ThreadwiseTensorSliceTransfer_v4<int32_t,
                                                      int32_t,
                                                      decltype(b_scale_block_desc_k0_n0_n1_n2_k1),
                                                      decltype(b_scale_thread_desc_),
                                                      Sequence<1, 1, 1, 1, 1>,
                                                      Sequence<0, 1, 2, 3, 4>,
                                                      4,
                                                      1,
                                                      1>;
    };

    template <>
    struct BScaleThreadCopySelector<false>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic_IntraRow<
            int32_t,
            int32_t,
            decltype(b_scale_block_desc_k0_n0_n1_n2_k1),
            decltype(b_scale_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<1, 1, 1, 1, 1>,
            Sequence<0, 1, 2, 3, 4>,
            4,
            1,
            false>;
    };

    typename AScaleThreadCopySelector<AEnableLds>::type a_scale_thread_copy_;
    typename BScaleThreadCopySelector<BEnableLds>::type b_scale_thread_copy_;
};
#endif

#elif defined(__gfx12__)
template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatAcc,
          typename ABlockDesc,
          typename BBlockDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWMMA,
          index_t NPerWMMA,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          bool AEnableLds              = true,
          bool BEnableLds              = true,
          bool TransposeC              = false,
          TensorLoadOption ALoadOption = TensorLoadOption::DEFAULT_LOAD,
          TensorLoadOption BLoadOption = TensorLoadOption::DEFAULT_LOAD>
/* Option: Read from LDS, big buffer hold all threads required data
 * Source
 * A: K0PerBlock x MPerBlock x K1
 * B: K0PerBlock x NPerBlock x K1
 * Destination
 * C, non-transpose
 * thread level: MRepeat x NRepeat x MAccVgprs
 * block  level: MRepeat x MWave x MSubGroup x NRepeat x NWave x NThreadPerSubGroup x MAccVgprs
 * KPACK == WMMA_K = 16
 *
 * Option: Read from VMEM, small buffer hold each thread own required data (Skip LDS)
 * Source:
 * A(if skip LDS): MRepeat x KPack
 * B(if skip LDS): NRepeat x KPack
 * Destination
 * C, non-transpose
 * block level: MRepeat x MWave x MSubGroup x NRepeat x NWave x NThreadPerSubGroup x MAccVgprs
 */
struct BlockwiseGemmWMMA
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    // Hardcode of WaveSize, since current HIP Runtime(5.4.0-10984) could not return correct one.
    static constexpr index_t WaveSize = 32;

    // When use LDS, each Row(16 consecutive lanes) read whole data from source buffer
    // When not use LDS, each Row read half of whole data from source buffer, exchange the data via
    // permutation
    static constexpr index_t A_KRow = 2;
    static constexpr index_t B_KRow = 2;

    static constexpr index_t A_K1 = ABlockDesc{}.GetLength(I5);
    static constexpr index_t B_K1 = BBlockDesc{}.GetLength(I5);

    static constexpr auto wmma_gemm =
        WmmaGemm<FloatA, FloatB, FloatAcc, MPerWMMA, NPerWMMA, KPack, TransposeC>{};

    static constexpr auto WmmaK = Number<wmma_gemm.wmma_instr.k_per_wmma>{};

    static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerWMMA);
    static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWMMA);

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              FloatAcc,
                              MRepeat * NRepeat,
                              wmma_gemm.GetRegSizePerWmma(),
                              true>
        c_thread_buf_;

    __host__ __device__ constexpr auto& GetCThreadBuffer() { return c_thread_buf_; }

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = ThisThreadBlock::GetThreadId();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MWaves, NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    // Default, Block buffer in LDS, thread level offset enabled
    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        if constexpr(AEnableLds)
        {
            const auto wave_idx = GetWaveIdx();
            const auto waveId_m = wave_idx[I0];

            const auto WMMA_a_idx = wmma_gemm.CalculateAThreadOriginDataIndex();
            //               |KRepeat   |MRepeat|MWave    |KRow                       |MLane
            //               |KPack
            return make_tuple(0, 0, waveId_m, wmma_gemm.GetSubGroupId(), WMMA_a_idx, 0);
        }
        else
        {
            return make_tuple(0, 0, 0, 0, 0, 0);
        }
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        if constexpr(BEnableLds)
        {
            const auto wave_idx = GetWaveIdx();
            const auto waveId_n = wave_idx[I1];

            const auto WMMA_b_idx = wmma_gemm.CalculateBThreadOriginDataIndex();
            //  |KRepeat   |NRepeat|Nwave     |KRow  |NLane  |KPack
            return make_tuple(0, 0, waveId_n, wmma_gemm.GetSubGroupId(), WMMA_b_idx, 0);
        }
        else
        {
            return make_tuple(0, 0, 0, 0, 0, 0);
        }
    }

    template <index_t m0, index_t n0>
    __device__ static auto CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = wmma_gemm.GetBeginOfThreadBlk();

        constexpr auto mrepeat_mwave_mperWMMA_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerWMMA))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto nrepeat_nwave_nperWMMA_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerWMMA))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        const index_t c_thread_m = mrepeat_mwave_mperWMMA_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_idx[I0]))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperWMMA_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_idx[I1]))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    template <index_t m0, index_t n0>
    __device__ static auto CalculateCThreadOriginDataIndex7D(Number<m0>, Number<n0>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = wmma_gemm.GetBeginOfThreadBlk3D();

        return make_tuple(
            Number<m0>{}, waveId_m, blk_idx[I0], Number<n0>{}, waveId_n, blk_idx[I1], blk_idx[I2]);
    }

    using Tuple6 = decltype(CalculateAThreadOriginDataIndex());
    __host__ __device__ BlockwiseGemmWMMA(Tuple6 a_origin = CalculateAThreadOriginDataIndex(),
                                          Tuple6 b_origin = CalculateBThreadOriginDataIndex())
        : a_thread_copy_(a_origin), b_thread_copy_(b_origin)
    {
        static_assert(ABlockDesc::IsKnownAtCompileTime() && BBlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerWMMA * MRepeat) == 0 &&
                          NPerBlock % (NPerWMMA * NRepeat) == 0,
                      "wrong!");
    }

    // transposed WMMA output C' = B' * A'
    __host__ __device__ static constexpr auto
    GetCThreadDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs()
    {
        constexpr auto c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens =
            wmma_gemm.GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths();

        constexpr auto NAccVgprs = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I2];
        return make_naive_tensor_descriptor_packed(
            //        |MRepeat            |MWave |MSubGroup |NRepeat           |NWave
            //        |NThreadPerSubGroup |MAccVgprs
            make_tuple(Number<MRepeat>{}, I1, I1, Number<NRepeat>{}, I1, I1, NAccVgprs));
    }

    // Thread level, register decriptor. Vector-write
    __host__ __device__ static constexpr auto
    GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs()
    {
        constexpr auto c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens =
            wmma_gemm.GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths();

        constexpr auto MAccVgprs = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I2];
        constexpr auto AccStride = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I3];
        return make_naive_tensor_descriptor(
            //        |MRepeat           |MWave |MSubGroup |NRepeat           |NWave
            //        |NThreadPerSubGroup |MAccVgprs
            make_tuple(Number<MRepeat>{}, I1, I1, Number<NRepeat>{}, I1, I1, MAccVgprs),
            make_tuple(Number<NRepeat>{} * MAccVgprs * AccStride,
                       Number<NRepeat>{} * MAccVgprs * AccStride,
                       Number<NRepeat>{} * MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       AccStride));
    }

    template <typename CGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
        const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto c_grid_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma =
            transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(
                    make_unmerge_transform(make_tuple(M / (MWaves * MPerWMMA), MWaves, MPerWMMA)),
                    make_unmerge_transform(make_tuple(N / (NWaves * NPerWMMA), NWaves, NPerWMMA))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5>{}));

        return wmma_gemm
            .MakeCDesc_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
                c_grid_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma);
    }

    // transposed WMMA output C' = B' * A'
    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs()
    {
        constexpr auto c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<MPerWMMA>{},
                                                           Number<NRepeat>{},
                                                           Number<NWaves>{},
                                                           Number<NPerWMMA>{}));

        return wmma_gemm
            .MakeCDesc_MBlockxRepeat_MWave_MThreadPerSubGroup_NBlockxRepeat_NWave_NSubGroup_NAccVgprs(
                c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma);
    }

    // Provide dimension size
    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs()
    {
        constexpr auto c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<MPerWMMA>{},
                                                           Number<NRepeat>{},
                                                           Number<NWaves>{},
                                                           Number<NPerWMMA>{}));

        return wmma_gemm
            .MakeCDesc_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
                c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma);
    }

    // Describe how data allocated in thread copy src buffer
    // M0_M1_M2 = MRepeat_MWave_MPerWmma, N0_N1_N2 = NRepeat_NWave_NPerWmma
    static constexpr ABlockDesc a_block_desc_k0_m0_m1_m2_k1;
    static constexpr BBlockDesc b_block_desc_k0_n0_n1_n2_k1;

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatB>(
            b_thread_desc_.GetElementSpaceSize());

        static_assert(KPack % (A_K1 * A_KRow) == 0, "");
        static_assert(KPack % (B_K1 * B_KRow) == 0, "");

        // basic intrinsic to determine loopover direction
        if constexpr(MRepeat < NRepeat)
        {
            static_for<0, KPerBlock / KPack, 1>{}(
                [&](auto k) { // k=0,1,2 instead of k=0,kpack*1, ...
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        // read A
                        a_thread_copy_.Run(
                            a_block_desc_k0_m0_m1_m2_k1,
                            make_tuple(Number<k * KPack / A_K1 / A_KRow>{}, m0, I0, I0, I0, I0),
                            a_block_buf,
                            a_thread_desc_,
                            make_tuple(I0, m0, I0, I0, I0, I0),
                            a_thread_buf);
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            // read B
                            vector_type<FloatA, KPack / A_KRow> a_thread_vec;
                            vector_type<FloatB, KPack / B_KRow> b_thread_vec;

                            b_thread_copy_.Run(
                                b_block_desc_k0_n0_n1_n2_k1,
                                make_tuple(Number<k * KPack / B_K1 / B_KRow>{}, n0, I0, I0, I0, I0),
                                b_block_buf,
                                b_thread_desc_,
                                make_tuple(I0, n0, I0, I0, I0, I0),
                                b_thread_buf);

                            static_for<0, KPack / A_KRow, 1>{}([&](auto i) {
                                a_thread_vec.template AsType<FloatA>()(i) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(i / A_K1, m0, 0, 0, 0, i % A_K1))>{}];
                            });

                            static_for<0, KPack / B_KRow, 1>{}([&](auto i) {
                                b_thread_vec.template AsType<FloatB>()(i) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(i / B_K1, n0, 0, 0, 0, i % B_K1))>{}];
                            });

                            using wmma_input_type_a =
                                typename vector_type<FloatA, WmmaK / A_KRow>::type;
                            using wmma_input_type_b =
                                typename vector_type<FloatB, WmmaK / B_KRow>::type;

                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                            wmma_gemm.template Run<>(
                                a_thread_vec.template AsType<wmma_input_type_a>(),
                                b_thread_vec.template AsType<wmma_input_type_b>(),
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        });
                    });
                });
        }
        else
        {
            static_ford<Sequence<NRepeat, MRepeat, KPerBlock / KPack>>{}([&](auto nmk) {
                constexpr auto n0 = Number<nmk[Number<0>{}]>{};
                constexpr auto m0 = Number<nmk[Number<1>{}]>{};
                constexpr auto k = Number<nmk[Number<2>{}]>{}; // k=0,1,2 instead of k=0,kpack*1, ..
                // read B
                b_thread_copy_.Run(
                    b_block_desc_k0_n0_n1_n2_k1,
                    make_tuple(Number<k * KPack / B_K1 / B_KRow>{}, n0, I0, I0, I0, I0),
                    b_block_buf,
                    b_thread_desc_,
                    make_tuple(I0, n0, I0, I0, I0, I0),
                    b_thread_buf);
                // read A
                a_thread_copy_.Run(
                    a_block_desc_k0_m0_m1_m2_k1,
                    make_tuple(Number<k * KPack / A_K1 / A_KRow>{}, m0, I0, I0, I0, I0),
                    a_block_buf,
                    a_thread_desc_,
                    make_tuple(I0, m0, I0, I0, I0, I0),
                    a_thread_buf);

                vector_type<FloatA, KPack / A_KRow> a_thread_vec;
                vector_type<FloatB, KPack / B_KRow> b_thread_vec;

                static_for<0, KPack / A_KRow, 1>{}([&](auto i) {
                    a_thread_vec.template AsType<FloatA>()(i) =
                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                            make_tuple(i / A_K1, m0, 0, 0, 0, i % A_K1))>{}];
                });

                static_for<0, KPack / B_KRow, 1>{}([&](auto i) {
                    b_thread_vec.template AsType<FloatB>()(i) =
                        b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                            make_tuple(i / B_K1, n0, 0, 0, 0, i % B_K1))>{}];
                });

                using wmma_input_type_a = typename vector_type<FloatA, WmmaK / A_KRow>::type;
                using wmma_input_type_b = typename vector_type<FloatB, WmmaK / B_KRow>::type;

                constexpr index_t c_offset = c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                wmma_gemm.template Run<>(a_thread_vec.template AsType<wmma_input_type_a>(),
                                         b_thread_vec.template AsType<wmma_input_type_b>(),
                                         c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
            });
        }
    }

    protected:
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor(
        make_tuple(Number<KPack / A_K1 / A_KRow>{}, Number<MRepeat>{}, I1, I1, I1, Number<A_K1>{}),
        make_tuple(Number<A_K1>{},
                   Number<KPack / A_KRow>{},
                   Number<A_K1>{},
                   Number<A_K1>{},
                   Number<A_K1>{},
                   Number<1>{}));

    static constexpr auto b_thread_desc_ = make_naive_tensor_descriptor(
        make_tuple(Number<KPack / B_K1 / B_KRow>{}, Number<NRepeat>{}, I1, I1, I1, Number<B_K1>{}),
        make_tuple(Number<B_K1>{},
                   Number<KPack / B_KRow>{},
                   Number<B_K1>{},
                   Number<B_K1>{},
                   Number<B_K1>{},
                   Number<1>{}));

    // C[M, N, NumRegWMMA]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, wmma_gemm.GetRegSizePerWmma()));

    template <bool EnableLds, TensorLoadOption LoadOption>
    struct AThreadCopySelector;

    template <TensorLoadOption LoadOption>
    struct AThreadCopySelector<true, LoadOption>
    {
        using type =
            ThreadwiseTensorSliceTransfer_v4<FloatA,
                                             FloatA,
                                             decltype(a_block_desc_k0_m0_m1_m2_k1),
                                             decltype(a_thread_desc_),
                                             Sequence<KPack / A_K1 / A_KRow, 1, 1, 1, 1, A_K1>,
                                             Sequence<0, 1, 2, 3, 4, 5>,
                                             5,
                                             A_K1,
                                             A_K1>;
    };

    template <TensorLoadOption LoadOption>
    struct AThreadCopySelector<false, LoadOption>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic_IntraRow<
            FloatA,
            FloatA,
            decltype(a_block_desc_k0_m0_m1_m2_k1),
            decltype(a_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<KPack / A_K1 / A_KRow, 1, 1, 1, 1, A_K1>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            A_K1,
            false>;
    };

    template <bool EnableLds, TensorLoadOption LoadOption>
    struct BThreadCopySelector;

    template <TensorLoadOption LoadOption>
    struct BThreadCopySelector<true, LoadOption>
    {
        using type =
            ThreadwiseTensorSliceTransfer_v4<FloatB,
                                             FloatB,
                                             decltype(b_block_desc_k0_n0_n1_n2_k1),
                                             decltype(b_thread_desc_),
                                             Sequence<KPack / B_K1 / B_KRow, 1, 1, 1, 1, B_K1>,
                                             Sequence<0, 1, 2, 3, 4, 5>,
                                             5,
                                             B_K1,
                                             B_K1>;
    };

    template <TensorLoadOption LoadOption>
    struct BThreadCopySelector<false, LoadOption>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic_IntraRow<
            FloatB,
            FloatB,
            decltype(b_block_desc_k0_n0_n1_n2_k1),
            decltype(b_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<KPack / B_K1 / B_KRow, 1, 1, 1, 1, B_K1>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            B_K1,
            false>;
    };

    typename AThreadCopySelector<AEnableLds, ALoadOption>::type a_thread_copy_;
    typename BThreadCopySelector<BEnableLds, BLoadOption>::type b_thread_copy_;
};
#else
template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatAcc,
          typename ABlockDesc,
          typename BBlockDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWMMA,
          index_t NPerWMMA,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          bool AEnableLds = true,
          bool BEnableLds = true,
          bool TransposeC = false>
/* Option: Read from LDS, big buffer hold all threads required data
 * Source
 * A: K0PerBlock x MPerBlock x K1
 * B: K0PerBlock x NPerBlock x K1
 * Destination
 * C, non-transpose
 * thread level: MRepeat x NRepeat x MAccVgprs
 * block  level: MRepeat x MWave x MSubGroup x NRepeat x NWave x NThreadPerSubGroup x MAccVgprs
 * KPACK == WMMA_K = 16
 *
 * Option: Read from VMEM, small buffer hold each thread own required data (Skip LDS)
 * Source:
 * A(if skip LDS): MRepeat x KPack
 * B(if skip LDS): NRepeat x KPack
 * Destination
 * C, non-transpose
 * block level: MRepeat x MWave x MSubGroup x NRepeat x NWave x NThreadPerSubGroup x MAccVgprs
 */
struct BlockwiseGemmWMMA
{
    static constexpr auto I0    = Number<0>{};
    static constexpr auto I1    = Number<1>{};
    static constexpr auto I2    = Number<2>{};
    static constexpr auto I3    = Number<3>{};
    static constexpr auto I4    = Number<4>{};
    static constexpr auto I5    = Number<5>{};
    static constexpr auto WmmaK = Number<16>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    // Hardcode of WaveSize, since current HIP Runtime(5.4.0-10984) could not return correct one.
    static constexpr index_t WaveSize = 32;

    // When use LDS, each Row(16 consecutive lanes) read whole data from source buffer
    // When not use LDS, each Row read half of whole data from source buffer, exchange the data via
    // permutation
    static constexpr index_t A_KRow = AEnableLds ? 1 : 2;
    static constexpr index_t B_KRow = BEnableLds ? 1 : 2;
    static constexpr index_t A_K1   = ABlockDesc{}.GetLength(I5);
    static constexpr index_t B_K1   = BBlockDesc{}.GetLength(I5);

    static constexpr auto wmma_gemm =
        WmmaGemm<FloatA, FloatB, FloatAcc, MPerWMMA, NPerWMMA, KPack, TransposeC>{};

    static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerWMMA);
    static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWMMA);

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              FloatAcc,
                              MRepeat * NRepeat,
                              wmma_gemm.GetRegSizePerWmma(),
                              true>
        c_thread_buf_;

    __host__ __device__ constexpr auto& GetCThreadBuffer() { return c_thread_buf_; }

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = ThisThreadBlock::GetThreadId();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MWaves, NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    // Default, Block buffer in LDS, thread level offset enabled
    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        if constexpr(AEnableLds)
        {
            const auto wave_idx   = GetWaveIdx();
            const auto waveId_m   = wave_idx[I0];
            const auto WMMA_a_idx = wmma_gemm.CalculateAThreadOriginDataIndex();

            //  |KRepeat   |MRepeat|MWave    |KRow  |MLane  |KPack
            return make_tuple(0, 0, waveId_m, 0, WMMA_a_idx, 0);
        }
        else
        {
            return make_tuple(0, 0, 0, 0, 0, 0);
        }
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        if constexpr(BEnableLds)
        {
            const auto wave_idx   = GetWaveIdx();
            const auto waveId_n   = wave_idx[I1];
            const auto WMMA_b_idx = wmma_gemm.CalculateBThreadOriginDataIndex();

            //  |KRepeat   |NRepeat|Nwave     |KRow  |NLane  |KPack
            return make_tuple(0, 0, waveId_n, 0, WMMA_b_idx, 0);
        }
        else
        {
            return make_tuple(0, 0, 0, 0, 0, 0);
        }
    }

    template <index_t m0, index_t n0>
    __device__ static auto CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = wmma_gemm.GetBeginOfThreadBlk();

        constexpr auto mrepeat_mwave_mperWMMA_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerWMMA))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto nrepeat_nwave_nperWMMA_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerWMMA))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        const index_t c_thread_m = mrepeat_mwave_mperWMMA_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_idx[I0]))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperWMMA_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_idx[I1]))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    template <index_t m0, index_t n0>
    __device__ static auto CalculateCThreadOriginDataIndex7D(Number<m0>, Number<n0>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = wmma_gemm.GetBeginOfThreadBlk3D();

        return make_tuple(
            Number<m0>{}, waveId_m, blk_idx[I0], Number<n0>{}, waveId_n, blk_idx[I1], blk_idx[I2]);
    }

    using Tuple6 = decltype(CalculateAThreadOriginDataIndex());
    __host__ __device__ BlockwiseGemmWMMA(Tuple6 a_origin = CalculateAThreadOriginDataIndex(),
                                          Tuple6 b_origin = CalculateBThreadOriginDataIndex())
        : a_thread_copy_(a_origin), b_thread_copy_(b_origin)
    {
        static_assert(ABlockDesc::IsKnownAtCompileTime() && BBlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerWMMA * MRepeat) == 0 &&
                          NPerBlock % (NPerWMMA * NRepeat) == 0,
                      "wrong!");
    }

    // transposed WMMA output C' = B' * A'
    __host__ __device__ static constexpr auto
    GetCThreadDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs()
    {
        constexpr auto c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens =
            wmma_gemm.GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths();

        constexpr auto NAccVgprs = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I2];

        return make_naive_tensor_descriptor_packed(
            //        |MRepeat            |MWave |MSubGroup |NRepeat           |NWave
            //        |NThreadPerSubGroup |MAccVgprs
            make_tuple(Number<MRepeat>{}, I1, I1, Number<NRepeat>{}, I1, I1, NAccVgprs));
    }

    // Thread level, register decriptor. Vector-write
    __host__ __device__ static constexpr auto
    GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs()
    {
        constexpr auto c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens =
            wmma_gemm.GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths();

        constexpr auto MAccVgprs = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I2];
        constexpr auto AccStride = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I3];
        return make_naive_tensor_descriptor(
            //        |MRepeat           |MWave |MSubGroup |NRepeat           |NWave
            //        |NThreadPerSubGroup |MAccVgprs
            make_tuple(Number<MRepeat>{}, I1, I1, Number<NRepeat>{}, I1, I1, MAccVgprs),
            make_tuple(Number<NRepeat>{} * MAccVgprs * AccStride,
                       Number<NRepeat>{} * MAccVgprs * AccStride,
                       Number<NRepeat>{} * MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       AccStride));
    }

    template <typename CGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
        const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto c_grid_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma =
            transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(
                    make_unmerge_transform(make_tuple(M / (MWaves * MPerWMMA), MWaves, MPerWMMA)),
                    make_unmerge_transform(make_tuple(N / (NWaves * NPerWMMA), NWaves, NPerWMMA))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5>{}));

        return wmma_gemm
            .MakeCDesc_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
                c_grid_desc_mblockxrepeat_mwave_mperwmma_nblockxrepeat_nwave_nperwmma);
    }

    // transposed WMMA output C' = B' * A'
    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs()
    {
        constexpr auto c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<MPerWMMA>{},
                                                           Number<NRepeat>{},
                                                           Number<NWaves>{},
                                                           Number<NPerWMMA>{}));

        return wmma_gemm
            .MakeCDesc_MBlockxRepeat_MWave_MThreadPerSubGroup_NBlockxRepeat_NWave_NSubGroup_NAccVgprs(
                c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma);
    }

    // Provide dimension size
    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs()
    {
        constexpr auto c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<MPerWMMA>{},
                                                           Number<NRepeat>{},
                                                           Number<NWaves>{},
                                                           Number<NPerWMMA>{}));

        return wmma_gemm
            .MakeCDesc_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
                c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma);
    }

    // Describe how data allocated in thread copy src buffer
    // M0_M1_M2 = MRepeat_MWave_MPerWmma, N0_N1_N2 = NRepeat_NWave_NPerWmma
    static constexpr ABlockDesc a_block_desc_k0_m0_m1_m2_k1;
    static constexpr BBlockDesc b_block_desc_k0_n0_n1_n2_k1;

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatB>(
            b_thread_desc_.GetElementSpaceSize());

        // basic intrinsic to determine loopover direction
        if constexpr(MRepeat < NRepeat)
        {
            static_for<0, KPerBlock / KPack, 1>{}(
                [&](auto k) { // k=0,1,2 instead of k=0,kpack*1, ...
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        // read A
                        a_thread_copy_.Run(
                            a_block_desc_k0_m0_m1_m2_k1,
                            make_tuple(Number<k * KPack / A_K1 / A_KRow>{}, m0, I0, I0, I0, I0),
                            a_block_buf,
                            a_thread_desc_,
                            make_tuple(I0, m0, I0, I0, I0, I0),
                            a_thread_buf);

                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            // read B
                            b_thread_copy_.Run(
                                b_block_desc_k0_n0_n1_n2_k1,
                                make_tuple(Number<k * KPack / B_K1 / B_KRow>{}, n0, I0, I0, I0, I0),
                                b_block_buf,
                                b_thread_desc_,
                                make_tuple(I0, n0, I0, I0, I0, I0),
                                b_thread_buf);

                            vector_type<FloatA, KPack> a_thread_vec;
                            vector_type<FloatB, KPack> b_thread_vec;

                            static_for<0, KPack, 1>{}([&](auto i) {
                                a_thread_vec.template AsType<FloatA>()(i) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(i / A_K1 / A_KRow,
                                                   m0,
                                                   0,
                                                   (i / A_K1) % A_KRow,
                                                   0,
                                                   i % A_K1))>{}];
                                b_thread_vec.template AsType<FloatB>()(i) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(i / B_K1 / B_KRow,
                                                   n0,
                                                   0,
                                                   (i / B_K1) % B_KRow,
                                                   0,
                                                   i % B_K1))>{}];
                            });

                            using wmma_input_type_a = typename vector_type<FloatA, WmmaK>::type;
                            using wmma_input_type_b = typename vector_type<FloatB, WmmaK>::type;

                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                            wmma_gemm.Run(a_thread_vec.template AsType<wmma_input_type_a>(),
                                          b_thread_vec.template AsType<wmma_input_type_b>(),
                                          c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        });
                    });
                });
        }
        else
        {
            static_ford<Sequence<NRepeat, MRepeat, KPerBlock / KPack>>{}([&](auto nmk) {
                constexpr auto n0 = Number<nmk[Number<0>{}]>{};
                constexpr auto m0 = Number<nmk[Number<1>{}]>{};
                constexpr auto k = Number<nmk[Number<2>{}]>{}; // k=0,1,2 instead of k=0,kpack*1, ..
                // read B
                b_thread_copy_.Run(
                    b_block_desc_k0_n0_n1_n2_k1,
                    make_tuple(Number<k * KPack / B_K1 / B_KRow>{}, n0, I0, I0, I0, I0),
                    b_block_buf,
                    b_thread_desc_,
                    make_tuple(I0, n0, I0, I0, I0, I0),
                    b_thread_buf);
                // read A
                a_thread_copy_.Run(
                    a_block_desc_k0_m0_m1_m2_k1,
                    make_tuple(Number<k * KPack / A_K1 / A_KRow>{}, m0, I0, I0, I0, I0),
                    a_block_buf,
                    a_thread_desc_,
                    make_tuple(I0, m0, I0, I0, I0, I0),
                    a_thread_buf);

                vector_type<FloatA, KPack> a_thread_vec;
                vector_type<FloatB, KPack> b_thread_vec;

                static_for<0, KPack, 1>{}([&](auto i) {
                    b_thread_vec.template AsType<FloatB>()(i) =
                        b_thread_buf[Number<b_thread_desc_.CalculateOffset(make_tuple(
                            i / B_K1 / B_KRow, n0, 0, (i / B_K1) % B_KRow, 0, i % B_K1))>{}];
                    a_thread_vec.template AsType<FloatA>()(i) =
                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(make_tuple(
                            i / A_K1 / A_KRow, m0, 0, (i / A_K1) % A_KRow, 0, i % A_K1))>{}];
                });

                using wmma_input_type_a = typename vector_type<FloatA, WmmaK>::type;
                using wmma_input_type_b = typename vector_type<FloatB, WmmaK>::type;

                constexpr index_t c_offset = c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                wmma_gemm.Run(a_thread_vec.template AsType<wmma_input_type_a>(),
                              b_thread_vec.template AsType<wmma_input_type_b>(),
                              c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
            });
        }
    }

    protected:
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor(make_tuple(Number<KPack / A_K1 / A_KRow>{},
                                                Number<MRepeat>{},
                                                I1,
                                                Number<A_KRow>{},
                                                I1,
                                                Number<A_K1>{}),
                                     make_tuple(Number<A_K1 * A_KRow>{},
                                                Number<KPack>{},
                                                Number<A_K1 * A_KRow>{},
                                                Number<A_K1>{},
                                                Number<A_K1>{},
                                                Number<1>{}));

    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor(make_tuple(Number<KPack / B_K1 / B_KRow>{},
                                                Number<NRepeat>{},
                                                I1,
                                                Number<B_KRow>{},
                                                I1,
                                                Number<B_K1>{}),
                                     make_tuple(Number<B_K1 * B_KRow>{},
                                                Number<KPack>{},
                                                Number<B_K1 * B_KRow>{},
                                                Number<B_K1>{},
                                                Number<B_K1>{},
                                                Number<1>{}));

    // C[M, N, NumRegWMMA]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, wmma_gemm.GetRegSizePerWmma()));

    template <bool EnableLds>
    struct AThreadCopySelector;

    template <>
    struct AThreadCopySelector<true>
    {
        using type =
            ThreadwiseTensorSliceTransfer_v4<FloatA,
                                             FloatA,
                                             decltype(a_block_desc_k0_m0_m1_m2_k1),
                                             decltype(a_thread_desc_),
                                             Sequence<KPack / A_K1 / A_KRow, 1, 1, A_KRow, 1, A_K1>,
                                             Sequence<0, 1, 2, 3, 4, 5>,
                                             5,
                                             A_K1,
                                             A_K1>;
    };

    template <>
    struct AThreadCopySelector<false>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic_InterRow<
            FloatA,
            FloatA,
            decltype(a_block_desc_k0_m0_m1_m2_k1),
            decltype(a_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<KPack / A_K1 / A_KRow, 1, 1, 1, 1, A_K1>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            A_K1,
            0x76543210,
            0xfedcba98,
            TransposeC ? false : true>;
    };

    template <bool EnableLds>
    struct BThreadCopySelector;

    template <>
    struct BThreadCopySelector<true>
    {
        using type =
            ThreadwiseTensorSliceTransfer_v4<FloatB,
                                             FloatB,
                                             decltype(b_block_desc_k0_n0_n1_n2_k1),
                                             decltype(b_thread_desc_),
                                             Sequence<KPack / B_K1 / B_KRow, 1, 1, B_KRow, 1, B_K1>,
                                             Sequence<0, 1, 2, 3, 4, 5>,
                                             5,
                                             B_K1,
                                             B_K1>;
    };

    template <>
    struct BThreadCopySelector<false>
    {
        using type = ThreadwiseTensorSliceTransfer_StaticToStatic_InterRow<
            FloatB,
            FloatB,
            decltype(b_block_desc_k0_n0_n1_n2_k1),
            decltype(b_thread_desc_),
            tensor_operation::element_wise::PassThrough,
            Sequence<KPack / B_K1 / B_KRow, 1, 1, 1, 1, B_K1>,
            Sequence<0, 1, 2, 3, 4, 5>,
            5,
            B_K1,
            0x76543210,
            0xfedcba98,
            TransposeC ? true : false>;
    };

    typename AThreadCopySelector<AEnableLds>::type a_thread_copy_;
    typename BThreadCopySelector<BEnableLds>::type b_thread_copy_;
};

#ifdef CK_EXTENSION_MX_TYPE
// dummy class to fix compiling error
template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatAcc,
          typename ABlockDesc,
          typename BBlockDesc,
          typename AScaleBlockDesc,
          typename BScaleBlockDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWMMA,
          index_t NPerWMMA,
          index_t KPerWMMA,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          bool AEnableLds = true,
          bool BEnableLds = true,
          bool TransposeC = false>
struct BlockwiseMXGemmWMMA : public BlockwiseGemmWMMA<BlockSize,
                                                      FloatA,
                                                      FloatB,
                                                      FloatAcc,
                                                      ABlockDesc,
                                                      BBlockDesc,
                                                      MPerBlock,
                                                      NPerBlock,
                                                      KPerBlock,
                                                      MPerWMMA,
                                                      NPerWMMA,
                                                      MRepeat,
                                                      NRepeat,
                                                      KPack,
                                                      AEnableLds,
                                                      BEnableLds,
                                                      TransposeC>
{
    template <typename ABlockBuffer,
              typename BBlockBuffer,
              typename AScaleBlockBuffer,
              typename BScaleBlockBuffer,
              typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        const AScaleBlockBuffer& a_scale_block_buf,
                        const BScaleBlockBuffer& b_scale_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        ignore = a_block_buf;
        ignore = b_block_buf;
        ignore = a_scale_block_buf;
        ignore = b_scale_block_buf;
        ignore = c_thread_buf;
    }
};
#endif

#endif

} // namespace ck
#pragma clang diagnostic pop
