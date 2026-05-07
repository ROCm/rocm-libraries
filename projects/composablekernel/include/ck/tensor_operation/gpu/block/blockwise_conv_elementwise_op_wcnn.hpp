// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/warp/wcnn_conv.hpp"
#include "ck/tensor_operation/gpu/warp/wcnn_sba.hpp"
#include "ck/tensor_operation/gpu/warp/wcnn_cvt_tensor.hpp"
#include "ck/tensor_operation/gpu/warp/wcnn_fma_tensor.hpp"

namespace ck {

template <index_t ActiveFunc, bool UniformScale, bool ScaleBiasPacked>
struct BlockwiseElementSba
{
    static constexpr const char* name   = "BlockwiseElementSba_";
    static constexpr bool IsSuba        = true;
    static constexpr bool IsFma         = false;
    static constexpr index_t activeFunc = ActiveFunc;
    static constexpr index_t uniform    = UniformScale;
    static constexpr index_t packed     = ScaleBiasPacked;
    __host__ __device__ BlockwiseElementSba(const float scale = 1.0f) : scale_(scale) {}
    static constexpr index_t GetNumDTensor()
    {
        return ScaleBiasPacked ? 2 : (UniformScale ? 0 : 1);
    }
    float scale_ = 1.0f;
};

template <bool UniformScale>
struct BlockwiseElementFma
{
    static constexpr const char* name = "BlockwiseOp_Fma";
    static constexpr bool IsSuba      = false;
    static constexpr bool IsFma       = true;
    static constexpr bool uniform     = UniformScale;
    float scale_                      = 1.0f;
    __host__ __device__ BlockwiseElementFma(const float scale = 1.0f) : scale_(scale) {}
    static constexpr index_t GetNumDTensor() { return UniformScale ? 1 : 2; }
};

template <index_t ActiveFunc, bool UniformScale0, bool ScaleBiasPacked, bool UniformScale1>
struct BlockwiseElementSbaFma
{
    static constexpr const char* name = "BlockwiseOp_Sba_Fma";
    static constexpr bool IsSuba      = true;
    static constexpr bool IsFma       = true;
    using Sba = BlockwiseElementSba<ActiveFunc, UniformScale0, ScaleBiasPacked>;
    using Fma = BlockwiseElementFma<UniformScale1>;
    Sba sba_;
    Fma fma_;
    __host__ __device__ BlockwiseElementSbaFma(const float scale0 = 1.0f, const float scale1 = 1.0f)
        : sba_(scale0), fma_(scale1)
    {
    }
    static constexpr index_t GetNumDTensor() { return Sba::GetNumDTensor() + Fma::GetNumDTensor(); }
};

template <bool convert_to_tensor, index_t ActiveFunc, index_t convertScale, bool Clamp>
struct BlockwiseElementCvtTensor
{
    static constexpr const char* name   = "BlockwiseOp_cvt_tensor_";
    static constexpr index_t activeFunc = ActiveFunc;
    static constexpr index_t scale      = convertScale;
    static constexpr bool cvt_to_tensor = convert_to_tensor;
    static constexpr bool clamp         = Clamp;
};

struct BlockwiseElementPassThrough
{
    static constexpr const char* name   = "BlockwiseOp_passthrough";
    static constexpr bool IsSuba        = false;
    static constexpr bool IsFma         = false;
    static constexpr bool cvt_to_tensor = false;
};

template <typename ThisThreadBlock,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t HRepeat,
          index_t WRepeat,
          index_t HPerWcnn,
          index_t WPerWcnn>
__device__ static auto GetWcnnWaveIdx()
{
    static constexpr index_t WaveSize = 32;
    constexpr index_t HWaves          = HPerBlock / (HRepeat * HPerWcnn);
    constexpr index_t WWaves          = WPerBlock / (WRepeat * WPerWcnn);
    constexpr index_t KWaves = ThisThreadBlock::GetNumOfThread() / WaveSize / HWaves / WWaves;

    const index_t thread_id = ThisThreadBlock::GetThreadId();

    constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
        make_tuple(make_merge_transform(make_tuple(HWaves, WWaves, KWaves, WaveSize))),
        make_tuple(Sequence<0, 1, 2, 3>{}),
        make_tuple(Sequence<0>{}));

    return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
}

template <typename WeiDataType,
          typename InDataType,
          typename AccDataType,
          index_t CPerBlock,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t FilterSize,
          bool Transposed>
static constexpr index_t GetFilterIters()
{
    if constexpr(FilterSize == 2)
    {
        return Transposed ? 1 : 4;
    }
    else if constexpr(FilterSize == 3)
    {
        return 1;
    }
    else if constexpr(FilterSize == 1)
    {
        // Enable Iters if CPerBlock is multiple of CPerWcnn
        constexpr index_t CPerWcnn =
            WcnnConv<WeiDataType, InDataType, AccDataType, HPerWcnn, WPerWcnn, FilterSize, 1, 1>::
                GetNumInputChannels();
        constexpr index_t CRepeat = CPerBlock / CPerWcnn;
        if constexpr(CRepeat % 4 == 0)
        {
            return 4;
        }
        else if constexpr(CRepeat % 2 == 0)
        {
            return 2;
        }
        else
        {
            return 1;
        }
    }
    else
    {
        static_assert(false, "never called");
    }
}

template <typename ThisThreadBlock,
          typename AccDataType,
          typename DsBlockDesc,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t KPerBlock,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t KPerWcnn,
          index_t HRepeat,
          index_t WRepeat,
          index_t KRepeat,
          bool DsEnableLds,
          bool Aco,
          typename BlockwiseElementOp>
struct BlockwiseElementSbaWcnn
{
    static constexpr auto I0              = Number<0>{};
    static constexpr auto I1              = Number<1>{};
    static constexpr auto I2              = Number<2>{};
    static constexpr index_t WaveSize     = 32;
    static constexpr index_t NumDTensor   = DsBlockDesc::Size();
    static constexpr bool EnableWaveGroup = ThisThreadBlock::InWaveGroup();
    static constexpr auto wcnn_sba        = WcnnSba<AccDataType,
                                                    HPerWcnn,
                                                    WPerWcnn,
                                                    BlockwiseElementOp::activeFunc,
                                                    BlockwiseElementOp::packed,
                                                    BlockwiseElementOp::uniform,
                                                    Aco>{};
    static constexpr auto NumBias         = wcnn_sba.GetNumBiasComponents();
    static constexpr auto NumScale        = wcnn_sba.GetNumScaleComponents();

    __device__ static auto GetWaveIdx()
    {
        return GetWcnnWaveIdx<ThisThreadBlock,
                              HPerBlock,
                              WPerBlock,
                              HRepeat,
                              WRepeat,
                              HPerWcnn,
                              WPerWcnn>();
    }

    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        auto ds_grid_packed_desc = generate_tuple(
            [&](auto i) {
                const auto K = ds_grid_desc[i].GetLength(I2);
                return transform_tensor_descriptor(
                    ds_grid_desc[i],
                    make_tuple(make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_pass_through_transform(K)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<>{}, Sequence<>{}, Sequence<0>{}));
            },
            Number<NumDTensor>{});

        if constexpr(DsEnableLds)
        {
            return ds_grid_packed_desc;
        }
        else
        {
            // K0 x K1 X K2
            constexpr auto DsPerThread = wcnn_sba.GetNumBiasComponents();
            return generate_tuple(
                [&](auto i) {
                    const auto K = ds_grid_packed_desc[i].GetLength(I0);
                    return transform_tensor_descriptor(ds_grid_packed_desc[i],
                                                       make_tuple(make_unmerge_transform(make_tuple(
                                                           K / KPerWcnn,
                                                           Number<KPerWcnn / DsPerThread>{},
                                                           Number<DsPerThread>{}))),
                                                       make_tuple(Sequence<0>{}),
                                                       make_tuple(Sequence<0, 1, 2>{}));
                },
                Number<NumDTensor>{});
        }
    }

    template <index_t conv_phase, typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeSingleDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        auto ds_grid_packed_desc = generate_tuple(
            [&](auto i) {
                const auto K = ds_grid_desc[Number<conv_phase>{}][i].GetLength(I2);
                return transform_tensor_descriptor(
                    ds_grid_desc[Number<conv_phase>{}][i],
                    make_tuple(make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_pass_through_transform(K)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<>{}, Sequence<>{}, Sequence<0>{}));
            },
            Number<NumDTensor>{});

        if constexpr(DsEnableLds)
        {
            return ds_grid_packed_desc;
        }
        else
        {
            // K0 x K1 X K2
            constexpr auto DsPerThread = wcnn_sba.GetNumBiasComponents();
            return generate_tuple(
                [&](auto i) {
                    const auto K = ds_grid_packed_desc[i].GetLength(I0);
                    return transform_tensor_descriptor(ds_grid_packed_desc[i],
                                                       make_tuple(make_unmerge_transform(make_tuple(
                                                           K / KPerWcnn,
                                                           Number<KPerWcnn / DsPerThread>{},
                                                           Number<DsPerThread>{}))),
                                                       make_tuple(Sequence<0>{}),
                                                       make_tuple(Sequence<0, 1, 2>{}));
                },
                Number<NumDTensor>{});
        }
    }

    // Describe how data read from (LDS/VGPR) buffer, used by Block level classes
    template <typename DsBlockDesc_>
    __host__ __device__ static constexpr auto MakeDsWaveDescriptor(const DsBlockDesc_&)
    {
        constexpr auto DsPerThread  = wcnn_sba.GetNumBiasComponents();
        constexpr auto ds_wave_desc = [&]() {
            if constexpr(DsEnableLds)
            {
                return generate_tuple(
                    [&](auto i) {
                        return transform_tensor_descriptor(
                            DsBlockDesc_{}[Number<i>{}],
                            make_tuple(
                                make_unmerge_transform(make_tuple(Number<KPerBlock / KPerWcnn>{},
                                                                  Number<KPerWcnn / DsPerThread>{},
                                                                  Number<DsPerThread>{}))),
                            make_tuple(Sequence<0>{}),
                            make_tuple(Sequence<0, 1, 2>{}));
                    },
                    Number<NumDTensor>{});
            }
            else
            {
                return generate_tuple(
                    [&](auto) {
                        // KRepeat x K1 x K2
                        return make_naive_tensor_descriptor_packed(
                            make_tuple(Number<KRepeat>{}, I1, Number<DsPerThread>{}));
                    },
                    Number<NumDTensor>{});
            }
        }();

        return ds_wave_desc;
    }

    using DsWaveDesc = decltype(MakeDsWaveDescriptor(DsBlockDesc{}));

    static constexpr DsWaveDesc ds_wave_desc_;

    __device__ __host__ static auto CalculateDsThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx    = GetWaveIdx();
        const auto waveId_k    = wave_idx[I2];
        const auto wcnn_ds_idx = wcnn_sba.CalculateDsThreadOriginDataIndex(KPerWcnn);

        return make_tuple(waveId_k * KRepeat, wcnn_ds_idx[I0], wcnn_ds_idx[I1]);
#else
        return make_tuple(0, 0, 0);
#endif
    }
    using TupleDsData = decltype(CalculateDsThreadOriginDataIndex());
    __host__ __device__
    BlockwiseElementSbaWcnn(const BlockwiseElementOp& blockOp,
                            TupleDsData dsData_origin = CalculateDsThreadOriginDataIndex())
        : ds_thread_copy_(dsData_origin), scale_(type_convert<AccDataType>(blockOp.scale_))
    {
    }

    __host__ __device__ static auto CalculateThreadOriginDataIndex()
    {
        return generate_tuple([&](auto) { return CalculateDsThreadOriginDataIndex(); },
                              Number<NumDTensor>{});
    }

    __host__ __device__ static constexpr auto GetDsWaveDescLength()
    {
        constexpr auto DsPerThread = wcnn_sba.GetNumBiasComponents();
        return generate_tuple(
            [&](auto) { return Sequence<Number<KRepeat>{}, I1, Number<DsPerThread>{}>{}; },
            Number<NumDTensor>{});
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
    template <typename DsBlockBuffer,
              typename AccDataVec,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void
    Run(const DsBlockBuffer& ds_block_buf, AccDataVec& acc_vec, NumberH0, NumberW0, NumberK0) const
    {
        using BiasVec      = typename decltype(wcnn_sba)::BiasVec::type;
        using BiasScaleVec = typename decltype(wcnn_sba)::BiasScaleVec::type;

        // TOOD: move ds_thread_buf as class member and avoid copy ds data for each Run.
        auto ds_thread_buf = generate_tuple(
            [&](auto i) {
                return make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                    ds_thread_desc_[Number<i>{}].GetElementSpaceSize());
            },
            Number<NumDTensor>{});

        constexpr auto calcOffset = [&](auto i) {
            if constexpr(DsEnableLds)
            {
                return ds_thread_desc_[i].CalculateOffset(make_tuple(I0, I0, I0));
            }
            else
            {
                return ds_wave_desc_[i].CalculateOffset(make_tuple(NumberK0{}, I0, I0));
            }
        };
        auto getBufPtr = [&](auto i, auto offset) {
            if constexpr(DsEnableLds)
            {
                return &ds_thread_buf[i][offset];
            }
            else
            {
                return &ds_block_buf[i][offset];
            }
        };

        // Load scale/bias data
        if constexpr(DsEnableLds)
        {
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                ds_thread_copy_.Run(ds_wave_desc_[Number<i>{}],
                                    make_tuple(NumberK0{}, I0, I0),
                                    ds_block_buf[Number<i>{}],
                                    ds_thread_desc_[Number<i>{}],
                                    make_tuple(I0, I0, I0),
                                    ds_thread_buf(i));
            });
        }

        if constexpr(BlockwiseElementOp::packed)
        {
            constexpr index_t scale_offset = calcOffset(I0);
            constexpr index_t bias_offset  = calcOffset(I1);

            if constexpr(std::is_same<float, AccDataType>::value)
            {
                BiasVec bias;
                auto laneId = ThisThreadBlock::GetThreadId() & (WaveSize - 1);
                if(laneId % 2)
                {
                    bias = *reinterpret_cast<const BiasVec*>(getBufPtr(I0, Number<scale_offset>{}));
                }
                else
                {
                    bias = *reinterpret_cast<const BiasVec*>(getBufPtr(I1, Number<bias_offset>{}));
                }
                wcnn_sba.sba_instr.Run(acc_vec, 0, bias, acc_vec);
            }
            else if constexpr(std::is_same<ck::half_t, AccDataType>::value ||
                              std::is_same<ck::bhalf_t, AccDataType>::value)
            {
                BiasScaleVec bias_scale;
                bias_scale[0] =
                    *reinterpret_cast<const AccDataType*>(getBufPtr(I1, Number<bias_offset>{}));
                bias_scale[1] =
                    *reinterpret_cast<const AccDataType*>(getBufPtr(I0, Number<scale_offset>{}));

                if constexpr((HPerWcnn == 4) && (WPerWcnn == 2))
                {
                    // 4xhalf for output of convolve
                    // 4xhalf for input of sba/uba
                    using sba_out_vector_t = typename vector_type_maker_t<AccDataType, 2>::type;
                    sba_out_vector_t& sba_uba_output0 =
                        acc_vec.template AsType<sba_out_vector_t>()(Number<0>{});
                    sba_out_vector_t& sba_uba_output1 =
                        acc_vec.template AsType<sba_out_vector_t>()(Number<1>{});
                    wcnn_sba.sba_instr.Run(
                        acc_vec, 0, bias_scale, sba_uba_output0, sba_uba_output1);
                }
                else
                {
                    wcnn_sba.sba_instr.Run(acc_vec, 0, bias_scale, acc_vec);
                }
            }
        }
        else
        {
            constexpr index_t bias_offset = calcOffset(I0);
            BiasVec bias = *reinterpret_cast<const BiasVec*>(getBufPtr(I0, Number<bias_offset>{}));
            if constexpr(std::is_same<float, AccDataType>::value)
            {
                wcnn_sba.sba_instr.Run(acc_vec, scale_, bias, acc_vec);
            }
            else if constexpr(std::is_same<half_t, AccDataType>::value ||
                              std::is_same<bhalf_t, AccDataType>::value)
            {
                if constexpr((HPerWcnn == 4) && (WPerWcnn == 2))
                {
                    // 4xhalf for output of convolve
                    // 4xhalf for input of sba/uba
                    using sba_out_vector_t = typename vector_type_maker_t<AccDataType, 2>::type;
                    sba_out_vector_t& sba_uba_output0 =
                        acc_vec.template AsType<sba_out_vector_t>()(Number<0>{});
                    sba_out_vector_t& sba_uba_output1 =
                        acc_vec.template AsType<sba_out_vector_t>()(Number<1>{});
                    wcnn_sba.sba_instr.Run(acc_vec, scale_, bias, sba_uba_output0, sba_uba_output1);
                }
                else
                {
                    wcnn_sba.sba_instr.Run(acc_vec, scale_, bias, acc_vec);
                }
            }
        }
    }
#pragma clang diagnostic pop
    // D[H, W, K, NumDsComp]
    template <index_t Components>
    static constexpr auto MakeDsDescriptor()
    {
        return make_naive_tensor_descriptor_packed(make_tuple(I1, I1, Number<Components>{}));
    }

    static constexpr auto ds_bias_thread_desc_  = MakeDsDescriptor<NumBias>();
    static constexpr auto ds_scale_thread_desc_ = MakeDsDescriptor<NumScale>();
    static constexpr auto ds_thread_desc_ = make_tuple(ds_scale_thread_desc_, ds_bias_thread_desc_);

    static_assert(NumBias == NumScale);
    using DsDataThreadLdsCopy =
        ThreadwiseTensorSliceTransfer_v4<AccDataType,
                                         AccDataType,
                                         remove_cvref_t<decltype(ds_wave_desc_[I0])>,
                                         remove_cvref_t<decltype(ds_thread_desc_[I0])>,
                                         Sequence<1, 1, NumBias>, // ToDo for NumScale
                                         Sequence<0, 1, 2>,
                                         2,
                                         1,
                                         1>;
    DsDataThreadLdsCopy ds_thread_copy_;
    float scale_;
    template <typename DsDesc, typename Index>
    static constexpr auto GetDsElementSpaceSize(const DsDesc&, Index i)
    {
        if constexpr(i < NumDTensor)
        {
            return DsDesc{}[i].GetElementSpaceSize();
        }
        else
        {
            return 0;
        }
    }
    struct SharedMemTrait
    {
        static constexpr auto max_lds_align = 8;

        static constexpr auto d0_block_space_size_aligned =
            DsEnableLds ? math::integer_least_multiple(GetDsElementSpaceSize(DsBlockDesc{}, I0),
                                                       max_lds_align)
                        : 0;

        static constexpr auto d1_block_space_size_aligned =
            DsEnableLds ? math::integer_least_multiple(GetDsElementSpaceSize(DsBlockDesc{}, I1),
                                                       max_lds_align)
                        : 0;

        static constexpr auto ds_block_space_size_aligned =
            make_tuple(d0_block_space_size_aligned, d1_block_space_size_aligned);

        // LDS allocation for Ds in LDS
        static constexpr auto d0_block_space_offset = 0;
        static constexpr auto d1_block_space_offset = d0_block_space_size_aligned;

        static constexpr auto ds_block_space_offset =
            make_tuple(d0_block_space_offset, d1_block_space_offset);

        static constexpr auto lds_size = d0_block_space_size_aligned * sizeof(AccDataType) +
                                         d1_block_space_size_aligned * sizeof(AccDataType);
    };

    struct LaneSharedMemTrait
    {
        static constexpr index_t max_lane_shared_align = 4;

        static constexpr index_t d0_block_space_size_aligned =
            EnableWaveGroup && (DsEnableLds == false)
                ? math::integer_least_multiple(GetDsElementSpaceSize(DsWaveDesc{}, I0) *
                                                   sizeof(AccDataType),
                                               max_lane_shared_align)
                : 0;
        static constexpr index_t d1_block_space_size_aligned =
            EnableWaveGroup && (DsEnableLds == false)
                ? math::integer_least_multiple(GetDsElementSpaceSize(DsWaveDesc{}, I1) *
                                                   sizeof(AccDataType),
                                               max_lane_shared_align)
                : 0;

        static constexpr index_t d0_block_space_offset = 0;
        static constexpr index_t d1_block_space_offset = d0_block_space_size_aligned;
        static constexpr auto ds_block_space_offset =
            make_tuple(d0_block_space_offset, d1_block_space_offset);
        static constexpr index_t lane_shared_size =
            d0_block_space_size_aligned + d1_block_space_size_aligned;
    };
};

template <typename ThisThreadBlock,
          typename AccDataType,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t KPerBlock,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t KPerWcnn,
          index_t HRepeat,
          index_t WRepeat,
          index_t KRepeat,
          bool DsEnableLds,
          bool Aco,
          typename BlockwiseElementOp>
struct BlockwiseElementSbaWcnn<ThisThreadBlock,
                               AccDataType,
                               Tuple<>,
                               HPerBlock,
                               WPerBlock,
                               KPerBlock,
                               HPerWcnn,
                               WPerWcnn,
                               KPerWcnn,
                               HRepeat,
                               WRepeat,
                               KRepeat,
                               DsEnableLds,
                               Aco,
                               BlockwiseElementOp>
{
    static constexpr auto wcnn_sba = WcnnSba<AccDataType,
                                             HPerWcnn,
                                             WPerWcnn,
                                             BlockwiseElementOp::activeFunc,
                                             BlockwiseElementOp::packed,
                                             BlockwiseElementOp::uniform,
                                             Aco>{};

    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto MakeDsGridBlockDescriptor(const DsGridDesc&)
    {
        return Tuple<>{};
    }

    template <index_t conv_phase, typename DsGridDesc>
    __host__ __device__ static constexpr auto MakeSingleDsGridBlockDescriptor(const DsGridDesc&)
    {
        return Tuple<>{};
    }

    template <typename DsBlockDesc_>
    __host__ __device__ static constexpr auto MakeDsWaveDescriptor(const DsBlockDesc_&)
    {
        return Tuple<>{};
    }

    __host__ __device__ static auto CalculateThreadOriginDataIndex() { return make_tuple(); }

    __host__ __device__ static constexpr auto GetDsWaveDescLength() { return make_tuple(); }

    __host__ __device__ BlockwiseElementSbaWcnn(const BlockwiseElementOp& blockOp)
        : scale_(type_convert<AccDataType>(blockOp.scale_))
    {
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
    template <typename DsBlockBuffer,
              typename AccDataVec,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void
    Run(const DsBlockBuffer&, AccDataVec& acc_vec, NumberH0, NumberW0, NumberK0) const
    {
        if constexpr(std::is_same<float, AccDataType>::value)
        {
            wcnn_sba.sba_instr.Run(acc_vec, scale_, 0, acc_vec);
        }
        else if constexpr(std::is_same<half_t, AccDataType>::value ||
                          std::is_same<bhalf_t, AccDataType>::value)
        {
            if constexpr((HPerWcnn == 4) && (WPerWcnn == 2))
            {
                using sba_out_vector_t = typename vector_type_maker_t<AccDataType, 2>::type;
                sba_out_vector_t& sba_uba_output0 =
                    acc_vec.template AsType<sba_out_vector_t>()(Number<0>{});
                sba_out_vector_t& sba_uba_output1 =
                    acc_vec.template AsType<sba_out_vector_t>()(Number<1>{});
                wcnn_sba.sba_instr.Run(acc_vec, scale_, 0, sba_uba_output0, sba_uba_output1);
            }
            else
            {
                wcnn_sba.sba_instr.Run(acc_vec, scale_, 0, acc_vec);
            }
        }
    }
#pragma clang diagnostic pop

    struct SharedMemTrait
    {
        static constexpr auto max_lds_align               = 8;
        static constexpr auto ds_block_space_size_aligned = make_tuple(0, 0);

        // LDS allocation for Ds in LDS
        static constexpr auto ds_block_space_offset = make_tuple(0, 0);
        static constexpr auto lds_size              = 0;
    };

    struct LaneSharedMemTrait
    {
        static constexpr auto max_lane_shared_align = 4;
        static constexpr auto ds_block_space_offset = make_tuple(0, 0);
        static constexpr auto lane_shared_size      = 0;
    };

    AccDataType scale_;
    static constexpr auto ds_wave_desc_ = make_tuple();
};

template <typename ThisThreadBlock,
          typename DsDataType,
          typename AccDataType,
          typename DsBlockDescs, // {Scale, Residual}
          index_t HPerBlock,
          index_t WPerBlock,
          index_t KPerBlock,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t HRepeat,
          index_t WRepeat,
          index_t KRepeat,
          bool DsEnableLds,
          bool Aco,
          typename BlockwiseElementOp>
struct BlockwiseElementFmaWcnn
{
    static constexpr auto I0              = Number<0>{};
    static constexpr auto I1              = Number<1>{};
    static constexpr auto I2              = Number<2>{};
    static constexpr auto I3              = Number<3>{};
    static constexpr auto I4              = Number<4>{};
    static constexpr index_t WaveSize     = 32;
    static constexpr bool EnableWaveGroup = ThisThreadBlock::InWaveGroup();
    static_assert((DsDataType::Size() == 2) ||
                  (DsDataType::Size() == 1 && BlockwiseElementOp::uniform));
    static constexpr bool UniformScale     = BlockwiseElementOp::uniform;
    static constexpr index_t ResidualIndex = UniformScale ? 0 : 1;
    static constexpr index_t ScaleIndex    = UniformScale ? -1 : 0;
    static constexpr auto ResidualId       = Number<ResidualIndex>{};
    static constexpr auto ScaleId          = Number<ScaleIndex>{};
    using ResidualDataType = remove_cvref_t<tuple_element_t<ResidualIndex, DsDataType>>;
    using WcnnFma          = WcnnFmaFromTensor<ResidualDataType, AccDataType, HPerWcnn, WPerWcnn>;

    static constexpr WcnnFma wcnn_fma;
    static_assert(wcnn_fma.GetAccumChannelOrder() == Aco);
    using WcnnConv = WcnnConv<ResidualDataType,
                              ResidualDataType,
                              AccDataType,
                              HPerWcnn,
                              WPerWcnn,
                              1,
                              1,
                              1,
                              1,
                              EnableWaveGroup,
                              Aco>;
    static constexpr WcnnConv wcnn_conv;
    using ScaleDataVec = typename WcnnConv::AccDataVec;

    static constexpr index_t CPerWcnn                = wcnn_conv.GetNumInputChannels();
    static constexpr index_t KPerWcnn                = wcnn_conv.GetNumOutputChannels();
    static constexpr index_t NumSubTilePerImage      = wcnn_conv.GetNumSubTilesPerImageTile();
    static constexpr index_t NumSubTilesPerWeightTap = wcnn_conv.GetNumSubTilesPerWeightTap();
    static constexpr index_t NumWeightCompPerTile    = wcnn_conv.GetNumWeightCompPerTile();
    static constexpr index_t NumWeightTap            = wcnn_conv.GetNumWeightTap();
    static constexpr index_t HPerWave                = HRepeat * HPerWcnn;
    static constexpr index_t WPerWave                = WRepeat * WPerWcnn;
    static constexpr index_t KPerWave                = KPerBlock;
    static constexpr auto NumDataCompPerTile         = wcnn_conv.GetNumDataCompPerTile();

    // Accum descriptor info
    static constexpr auto NumAccComp        = wcnn_conv.GetNumAccumComponents();
    static constexpr auto NumAccCompPerTile = NumAccComp / NumSubTilePerImage;
    static constexpr auto NumAccSwizzleComp = Aco ? 2 : 4;
    static constexpr auto NumAccCompSubTile =
        NumAccCompPerTile > NumAccSwizzleComp ? NumAccCompPerTile / NumAccSwizzleComp : 1;

    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        if constexpr(DsEnableLds)
        {
            return ds_grid_desc;
        }
        else
        {
            // Residual: H x W x C -> W0 x C0 x H0 x H1 x H2 x W1 x C1
            const auto H0                 = ds_grid_desc[ResidualId].GetLength(I0);
            const auto W0                 = ds_grid_desc[ResidualId].GetLength(I1);
            const auto C0                 = ds_grid_desc[ResidualId].GetLength(I2);
            auto residual_grid_block_desc = transform_tensor_descriptor(
                ds_grid_desc[ResidualId],
                make_tuple(
                    make_unmerge_transform(make_tuple(H0 / HPerWcnn,
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWcnn / NumSubTilePerImage>{})),
                    make_unmerge_transform(make_tuple(W0 / WPerWcnn, Number<WPerWcnn>{})),
                    make_unmerge_transform(make_tuple(C0 / CPerWcnn, Number<CPerWcnn>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<2, 3, 4>{}, Sequence<0, 5>{}, Sequence<1, 6>{}));
            if constexpr(UniformScale)
            {
                return make_tuple(residual_grid_block_desc);
            }
            else
            {
                // Scale: H x W x K -> H0 x W0 x K0 x H1 x H2 x W1 x K1 x K2
                const auto H1              = ds_grid_desc[ScaleId].GetLength(I0);
                const auto W1              = ds_grid_desc[ScaleId].GetLength(I1);
                const auto C1              = ds_grid_desc[ScaleId].GetLength(I2);
                auto scale_grid_block_desc = transform_tensor_descriptor(
                    ds_grid_desc[ScaleId],
                    make_tuple(
                        make_unmerge_transform(make_tuple(H1 / HPerWcnn,
                                                          Number<NumSubTilePerImage>{},
                                                          Number<HPerWcnn / NumSubTilePerImage>{})),
                        make_unmerge_transform(make_tuple(W1 / WPerWcnn, Number<WPerWcnn>{})),
                        make_unmerge_transform(make_tuple(C1 / KPerWcnn,
                                                          Number<NumAccCompSubTile>{},
                                                          Number<KPerWcnn / NumAccCompSubTile>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0, 3, 4>{}, Sequence<1, 5>{}, Sequence<2, 6, 7>{}));
                return make_tuple(scale_grid_block_desc, residual_grid_block_desc);
            }
        }
    }

    template <index_t conv_phase, typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeSingleDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        if constexpr(DsEnableLds)
        {
            return ds_grid_desc;
        }
        else
        {
            // Residual: H x W x C -> W0 x C0 x H0 x H1 x H2 x W1 x C1
            const auto H0 = ds_grid_desc[Number<conv_phase>{}][ResidualId].GetLength(I0);
            const auto W0 = ds_grid_desc[Number<conv_phase>{}][ResidualId].GetLength(I1);
            const auto C0 = ds_grid_desc[Number<conv_phase>{}][ResidualId].GetLength(I2);
            auto residual_grid_block_desc = transform_tensor_descriptor(
                ds_grid_desc[Number<conv_phase>{}][ResidualId],
                make_tuple(
                    make_unmerge_transform(make_tuple(H0 / HPerWcnn,
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWcnn / NumSubTilePerImage>{})),
                    make_unmerge_transform(make_tuple(W0 / WPerWcnn, Number<WPerWcnn>{})),
                    make_unmerge_transform(make_tuple(C0 / CPerWcnn, Number<CPerWcnn>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<2, 3, 4>{}, Sequence<0, 5>{}, Sequence<1, 6>{}));
            if constexpr(UniformScale)
            {
                return make_tuple(residual_grid_block_desc);
            }
            else
            {
                // Scale: H x W x K -> H0 x W0 x K0 x H1 x H2 x W1 x K1 x K2
                const auto H1 = ds_grid_desc[Number<conv_phase>{}][ScaleId].GetLength(I0);
                const auto W1 = ds_grid_desc[Number<conv_phase>{}][ScaleId].GetLength(I1);
                const auto C1 = ds_grid_desc[Number<conv_phase>{}][ScaleId].GetLength(I2);
                auto scale_grid_block_desc = transform_tensor_descriptor(
                    ds_grid_desc[Number<conv_phase>{}][ScaleId],
                    make_tuple(
                        make_unmerge_transform(make_tuple(H1 / HPerWcnn,
                                                          Number<NumSubTilePerImage>{},
                                                          Number<HPerWcnn / NumSubTilePerImage>{})),
                        make_unmerge_transform(make_tuple(W1 / WPerWcnn, Number<WPerWcnn>{})),
                        make_unmerge_transform(make_tuple(C1 / KPerWcnn,
                                                          Number<NumAccCompSubTile>{},
                                                          Number<KPerWcnn / NumAccCompSubTile>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0, 3, 4>{}, Sequence<1, 5>{}, Sequence<2, 6, 7>{}));
                return make_tuple(scale_grid_block_desc, residual_grid_block_desc);
            }
        }
    }

    // Describe how data read from (LDS/VGPR) buffer, used by Block level classes
    template <typename DsBlockDesc_>
    __host__ __device__ static constexpr auto MakeScaleWaveDescriptor(const DsBlockDesc_&)
    {
        // H x W x K -> H0 x W0 x K0 x H1 x H2 x W1 x K1 x K2
        if constexpr(DsEnableLds)
        {
            // NOTE: always use I0 to create scale wave desc
            // In uniform mode, this descriptor isn't really used.
            return transform_tensor_descriptor(
                DsBlockDesc_{}[I0],
                make_tuple(
                    make_unmerge_transform(make_tuple(Number<HPerBlock / HPerWcnn>{},
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWcnn / NumSubTilePerImage>{})),
                    make_unmerge_transform(
                        make_tuple(Number<WPerBlock / WPerWcnn>{}, Number<WPerWcnn>{})),
                    make_unmerge_transform(make_tuple(Number<KPerBlock / KPerWcnn>{},
                                                      Number<NumAccCompSubTile>{},
                                                      Number<KPerWcnn / NumAccCompSubTile>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 3, 4>{}, Sequence<1, 5>{}, Sequence<2, 6, 7>{}));
        }
        else
        {
            // TODO: adjust stride according to DsBlockDesc_{}[I1];
            return make_naive_tensor_descriptor_packed(
                make_tuple(Number<HPerWave / HPerWcnn>{},
                           Number<WPerWave / WPerWcnn>{},
                           Number<KPerWave / KPerWcnn>{},
                           Number<NumSubTilePerImage>{},
                           I1,
                           I1,
                           Number<NumAccCompSubTile>{},
                           Number<NumAccCompPerTile / NumAccCompSubTile>{}));
        }
    }

    template <typename DsBlockDesc_>
    __host__ __device__ static constexpr auto MakeResidualWaveDescriptor(const DsBlockDesc_&)
    {
        if constexpr(DsEnableLds)
        {
            // H x W x C -> W0 x C0 x H0 x H1 x H2 x W1 x C1
            return transform_tensor_descriptor(
                DsBlockDesc_{}[ResidualId],
                make_tuple(
                    make_unmerge_transform(make_tuple(Number<HPerBlock / HPerWcnn>{},
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWcnn / NumSubTilePerImage>{})),
                    make_unmerge_transform(
                        make_tuple(Number<WPerBlock / WPerWcnn>{}, Number<WPerWcnn>{})),
                    make_unmerge_transform(
                        make_tuple(Number<KPerBlock / CPerWcnn>{}, Number<CPerWcnn>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<2, 3, 4>{}, Sequence<0, 5>{}, Sequence<1, 6>{}));
        }
        else
        {
            // TODO: adjust stride according to DsBlockDesc_{}[I0];
            return make_naive_tensor_descriptor_packed(make_tuple(Number<WPerWave / WPerWcnn>{},
                                                                  Number<KPerWave / CPerWcnn>{},
                                                                  Number<HPerWave / HPerWcnn>{},
                                                                  Number<NumSubTilePerImage>{},
                                                                  I1,
                                                                  I1,
                                                                  Number<NumDataCompPerTile>{}));
        }
    }

    template <typename DsBlockDesc_>
    __host__ __device__ static constexpr auto MakeDsWaveDescriptor(const DsBlockDesc_&)
    {
        if constexpr(UniformScale)
        {
            return make_tuple(MakeResidualWaveDescriptor(DsBlockDescs{}));
        }
        else
        {
            return make_tuple(MakeScaleWaveDescriptor(DsBlockDescs{}),
                              MakeResidualWaveDescriptor(DsBlockDescs{}));
        }
    }

    using ScaleWaveDesc    = decltype(MakeScaleWaveDescriptor(DsBlockDescs{}));
    using ResidualWaveDesc = decltype(MakeResidualWaveDescriptor(DsBlockDescs{}));

    static constexpr ScaleWaveDesc scale_wave_desc_;
    static constexpr ResidualWaveDesc residual_wave_desc_;
    static constexpr auto ds_wave_desc_ = MakeDsWaveDescriptor(DsBlockDescs{});

    static constexpr auto NumDataSubTiles = wcnn_conv.GetNumSubTilesPerImageTile();
    static constexpr auto NumDataTiles    = wcnn_conv.GetNumImageTilesInVertical();
    static constexpr auto residual_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                       I1,
                                                       Number<NumDataTiles>{},
                                                       Number<NumDataSubTiles>{},
                                                       I1,
                                                       I1,
                                                       Number<NumDataCompPerTile>{}));

    static constexpr auto scale_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(I1,
                   I1,
                   I1,
                   Number<NumSubTilePerImage>{},
                   I1,
                   I1,
                   Number<NumAccCompSubTile>{},
                   Number<NumAccCompPerTile / NumAccCompSubTile>{}));

    using ResidualDataThreadLdsCopy = ThreadwiseTensorSliceTransfer_v4<
        ResidualDataType,
        ResidualDataType,
        ResidualWaveDesc,
        decltype(residual_thread_desc_),
        Sequence<1, 1, NumDataTiles, NumDataSubTiles, 1, 1, NumDataCompPerTile>,
        Sequence<0, 1, 2, 3, 4, 5, 6>,
        6,
        NumDataCompPerTile,
        NumDataCompPerTile>;
    ResidualDataThreadLdsCopy residual_thread_copy_;

    using ScaleDataThreadLdsCopy =
        ThreadwiseTensorSliceTransfer_v4<AccDataType,
                                         AccDataType,
                                         ScaleWaveDesc,
                                         decltype(scale_thread_desc_),
                                         Sequence<1,
                                                  1,
                                                  1,
                                                  NumSubTilePerImage,
                                                  1,
                                                  1,
                                                  NumAccCompSubTile,
                                                  NumAccCompPerTile / NumAccCompSubTile>,
                                         Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                         7,
                                         1,
                                         1>;
    ScaleDataThreadLdsCopy scale_thread_copy_;

    __device__ static auto GetWaveIdx()
    {
        return GetWcnnWaveIdx<ThisThreadBlock,
                              HPerBlock,
                              WPerBlock,
                              HRepeat,
                              WRepeat,
                              HPerWcnn,
                              WPerWcnn>();
    }

    __device__ static auto CalculateScaleThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx    = GetWaveIdx();
        const auto waveId_h    = wave_idx[I0];
        const auto waveId_w    = wave_idx[I1];
        const auto waveId_k    = wave_idx[I2];
        const auto wcnn_ds_idx = wcnn_conv.template CalculateAccThreadOriginDataIndex<false>();

        return make_tuple(waveId_h * HRepeat,
                          waveId_w * WRepeat,
                          waveId_k * KRepeat,
                          wcnn_ds_idx[I0],
                          wcnn_ds_idx[I1],
                          wcnn_ds_idx[I2],
                          wcnn_ds_idx[I3],
                          wcnn_ds_idx[I4]);
#else
        return make_tuple(0, 0, 0, 0, 0, 0, 0, 0);
#endif
    }
    __device__ static auto CalculateResidualThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx = GetWaveIdx();
        const auto waveId_h = wave_idx[I0];
        const auto waveId_w = wave_idx[I1];
        const auto wcnn_in_data_idx =
            wcnn_conv.template CalculateInDataThreadOriginDataIndex<false>();
        // W0 x C0 x H0 x H1 x H2 x W1 x C1
        return make_tuple(waveId_w * WRepeat,
                          0,
                          waveId_h * HRepeat,
                          0,
                          wcnn_in_data_idx[I0],
                          wcnn_in_data_idx[I1],
                          wcnn_in_data_idx[I2]);
#else
        return make_tuple(0, 0, 0, 0, 0, 0, 0);
#endif
    }

    using TupleScaleData    = decltype(CalculateScaleThreadOriginDataIndex());
    using TupleResidualData = decltype(CalculateResidualThreadOriginDataIndex());
    __host__ __device__ BlockwiseElementFmaWcnn(
        const BlockwiseElementOp& blockOp,
        TupleResidualData residualData_origin = CalculateResidualThreadOriginDataIndex(),
        TupleScaleData scaleData_origin       = CalculateScaleThreadOriginDataIndex())
        : residual_thread_copy_(residualData_origin), scale_thread_copy_(scaleData_origin)
    {
        if constexpr(UniformScale)
        {
            static_for<0, wcnn_conv.GetNumAccumComponents(), 1>{}(
                [&](auto i) { scale_[i.value] = type_convert<AccDataType>(blockOp.scale_); });
        }
    }

    __host__ __device__ static auto CalculateThreadOriginDataIndex()
    {
        if constexpr(UniformScale)
        {
            return make_tuple(CalculateResidualThreadOriginDataIndex());
        }
        else
        {
            return make_tuple(CalculateScaleThreadOriginDataIndex(),
                              CalculateResidualThreadOriginDataIndex());
        }
    }

    __host__ __device__ static constexpr auto GetDsWaveDescLength()
    {
        using ScaleWaveDescLength = Sequence<HPerWave / HPerWcnn,
                                             WPerWave / WPerWcnn,
                                             KPerWave / KPerWcnn,
                                             NumSubTilePerImage,
                                             1,
                                             1,
                                             NumAccCompSubTile,
                                             NumAccCompPerTile / NumAccCompSubTile>;
        using ResidualDescLength  = Sequence<WPerWave / WPerWcnn,
                                             KPerWave / CPerWcnn,
                                             HPerWave / HPerWcnn,
                                             NumSubTilePerImage,
                                             1,
                                             1,
                                             NumDataCompPerTile>;
        if constexpr(UniformScale)
        {
            return make_tuple(ResidualDescLength{});
        }
        else
        {
            return make_tuple(ScaleWaveDescLength{}, ResidualDescLength{});
        }
    }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
    template <typename DsBlockBuffer,
              typename AccDataVec,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void
    Run(const DsBlockBuffer& ds_block_buf, AccDataVec& acc_vec, NumberH0, NumberW0, NumberK0) const
    {
        static_assert(std::is_same<AccDataVec, typename decltype(wcnn_conv)::AccDataVec>::value);
        using ResidualDataVec = typename decltype(wcnn_conv)::InDataVec;

        auto load_residual_data = [&](auto h0, auto w0, auto c0) {
            if constexpr(DsEnableLds)
            {
                // Load residual tensor data
                auto residual_thread_buf =
                    make_static_buffer<AddressSpaceEnum::Vgpr, ResidualDataType>(
                        residual_thread_desc_.GetElementSpaceSize());
                residual_thread_copy_.Run(residual_wave_desc_,
                                          make_tuple(w0, c0, h0, I0, I0, I0, I0),
                                          ds_block_buf[ResidualId],
                                          residual_thread_desc_,
                                          make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                          residual_thread_buf);
                return bit_cast<typename ResidualDataVec::type>(residual_thread_buf);
            }
            else
            {
                constexpr index_t residual_offset =
                    residual_wave_desc_.CalculateOffset(make_tuple(w0, c0, h0, I0, I0, I0, I0));
                return *reinterpret_cast<const typename ResidualDataVec::type*>(
                    &ds_block_buf[ResidualId][Number<residual_offset>{}]);
            }
        };

        auto load_scale_data = [&](auto h0, auto w0, auto k0) {
            if constexpr(UniformScale)
            {
                return scale_;
            }
            else if constexpr(DsEnableLds)
            {
                auto scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                    scale_thread_desc_.GetElementSpaceSize());

                scale_thread_copy_.Run(scale_wave_desc_,
                                       make_tuple(h0, w0, k0, I0, I0, I0, I0, I0),
                                       ds_block_buf[ScaleId],
                                       scale_thread_desc_,
                                       make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                       scale_thread_buf);
                return bit_cast<typename ScaleDataVec::type>(scale_thread_buf);
            }
            else
            {
                constexpr index_t scale_offset =
                    scale_wave_desc_.CalculateOffset(make_tuple(h0, w0, k0, I0, I0, I0, I0, I0));
                return *reinterpret_cast<const typename ScaleDataVec::type*>(
                    &ds_block_buf[ScaleId][Number<scale_offset>{}]);
            }
        };

        constexpr NumberH0 h0;
        constexpr NumberW0 w0;
        constexpr NumberK0 k0;

        auto scale = load_scale_data(h0, w0, k0);

        if constexpr(wcnn_fma.GetNumResidual() == 4)
        {
            constexpr index_t c0 = k0 * 4;
            auto residual_data0  = load_residual_data(h0, w0, Number<c0>{});
            auto residual_data1  = load_residual_data(h0, w0, Number<c0 + 1>{});
            auto residual_data2  = load_residual_data(h0, w0, Number<c0 + 2>{});
            auto residual_data3  = load_residual_data(h0, w0, Number<c0 + 3>{});
            wcnn_fma.fma_instr.Run(
                acc_vec.template AsType<typename AccDataVec::type>()(Number<0>{}),
                residual_data0,
                residual_data1,
                residual_data2,
                residual_data3,
                scale,
                acc_vec);
        }
        else if constexpr(wcnn_fma.GetNumResidual() == 2)
        {
            constexpr index_t c0 = k0 * 2;
            auto residual_data0  = load_residual_data(h0, w0, Number<c0>{});
            auto residual_data1  = load_residual_data(h0, w0, Number<c0 + 1>{});
            wcnn_fma.fma_instr.Run(
                acc_vec.template AsType<typename AccDataVec::type>()(Number<0>{}),
                residual_data0,
                residual_data1,
                scale,
                acc_vec);
        }
        else if constexpr(wcnn_fma.GetNumResidual() == 1)
        {
            constexpr index_t numAccum = wcnn_fma.GetNumAccum();
            constexpr index_t c0       = k0 / numAccum;
            auto residual_data0        = load_residual_data(h0, w0, Number<c0>{});
            if constexpr(k0 % numAccum == 0)
            {
                wcnn_fma.fma_instr.Run(
                    acc_vec.template AsType<typename AccDataVec::type>()(Number<0>{}),
                    residual_data0,
                    scale,
                    acc_vec);
            }
            else
            {
                static_assert((numAccum == 2) && (k0 % numAccum == 1));
                constexpr index_t ChanOff = (HPerWcnn == 8) && (WPerWcnn == 4) ? 8 : 16;
                constexpr WcnnFmaFromTensor<ResidualDataType,
                                            AccDataType,
                                            HPerWcnn,
                                            WPerWcnn,
                                            ChanOff>
                    wcnn_fma2;
                wcnn_fma2.fma_instr.Run(acc_vec.template AsType<AccDataVec::type>()(Number<0>{}),
                                        residual_data0,
                                        scale,
                                        acc_vec);
            }
        }
    }
#pragma clang diagnostic pop

    template <typename DsDescs>
    static constexpr auto GetScaleElementSpaceSize(const DsDescs&)
    {
        if constexpr(UniformScale)
        {
            return 0;
        }
        else
        {
            return DsDescs{}[ScaleId].GetElementSpaceSize();
        }
    }

    static constexpr auto GetDsLayoutTuple(index_t scale, index_t residual)
    {
        if constexpr(UniformScale)
        {
            return make_tuple(residual, 0);
        }
        else
        {
            return make_tuple(scale, residual);
        }
    }

    struct SharedMemTrait
    {
        static constexpr auto max_lds_align = 8;

        static constexpr auto scale_block_space_size_aligned =
            DsEnableLds ? math::integer_least_multiple(
                              DsBlockDescs{}[ResidualId].GetElementSpaceSize(), max_lds_align)
                        : 0;

        static constexpr auto residual_block_space_size_aligned =
            DsEnableLds ? math::integer_least_multiple(GetScaleElementSpaceSize(DsBlockDescs{}),
                                                       max_lds_align) *
                              WcnnFma::template SizeOfBits<ResidualDataType>() /
                              WcnnFma::template SizeOfBits<AccDataType>()
                        : 0;

        static constexpr auto ds_block_space_size_aligned =
            GetDsLayoutTuple(scale_block_space_size_aligned, residual_block_space_size_aligned);

        // LDS allocation for Ds in LDS
        static constexpr auto scale_block_space_offset = 0;
        static constexpr auto residual_block_space_offset =
            scale_block_space_offset + scale_block_space_size_aligned;

        static constexpr auto ds_block_space_offset =
            GetDsLayoutTuple(scale_block_space_offset, residual_block_space_offset);

        static constexpr auto lds_size = residual_block_space_size_aligned * sizeof(AccDataType) +
                                         scale_block_space_size_aligned * sizeof(AccDataType);
    };

    struct LaneSharedMemTrait
    {
        static constexpr index_t max_lane_shared_align = 4;

        static constexpr index_t residual_block_space_size_aligned =
            EnableWaveGroup && (DsEnableLds == false)
                ? math::integer_least_multiple(
                      ResidualWaveDesc{}.GetElementSpaceSize() *
                          WcnnFma::template SizeOfBits<ResidualDataType>() / 8,
                      max_lane_shared_align)
                : 0;
        static constexpr index_t scale_block_space_size_aligned =
            EnableWaveGroup && (DsEnableLds == false)
                ? math::integer_least_multiple(GetScaleElementSpaceSize(ds_wave_desc_) *
                                                   sizeof(AccDataType),
                                               max_lane_shared_align)
                : 0;

        static constexpr index_t scale_block_space_offset = 0;
        static constexpr index_t residual_block_space_offset =
            scale_block_space_offset + scale_block_space_size_aligned;
        static constexpr auto ds_block_space_offset =
            GetDsLayoutTuple(scale_block_space_offset, residual_block_space_offset);
        static constexpr index_t lane_shared_size =
            residual_block_space_size_aligned + scale_block_space_size_aligned;
    };
    typename ScaleDataVec::type scale_;
};

template <typename ThisThreadBlock,
          typename DsDataType,
          typename AccDataType,
          typename DsBlockDescs,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t KPerBlock,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t KPerWcnn,
          index_t HRepeat,
          index_t WRepeat,
          index_t KRepeat,
          bool DsEnableLds,
          bool Aco,
          typename BlockwiseElementOp>
struct BlockwiseElementSbaFmaWcnn
{
    static constexpr index_t NumDTensor    = DsBlockDescs::Size();
    static constexpr index_t NumSbaDTensor = BlockwiseElementOp::Sba::GetNumDTensor();
    static constexpr index_t NumFmaDTensor = BlockwiseElementOp::Fma::GetNumDTensor();
    static_assert(NumDTensor == NumSbaDTensor + NumFmaDTensor);
    static_assert(NumDTensor == 2); // todo remove this limitation

    static constexpr auto GetSbaDsBlockDesc()
    {
        return generate_tuple([&](auto i) { return DsBlockDescs{}[i]; }, Number<NumSbaDTensor>{});
    }

    static constexpr auto GetFmaDsBlockDesc()
    {
        return generate_tuple([&](auto i) { return DsBlockDescs{}[Number<i + NumSbaDTensor>{}]; },
                              Number<NumFmaDTensor>{});
    }

    static constexpr auto GetFmaDsDataType()
    {
        return generate_tuple([&](auto i) { return DsDataType{}[Number<i + NumSbaDTensor>{}]; },
                              Number<NumFmaDTensor>{});
    }

    using BlockwiseSuba = BlockwiseElementSbaWcnn<ThisThreadBlock,
                                                  AccDataType,
                                                  decltype(GetSbaDsBlockDesc()), // { Bias, Scale}
                                                  HPerBlock,
                                                  WPerBlock,
                                                  KPerBlock,
                                                  HPerWcnn,
                                                  WPerWcnn,
                                                  KPerWcnn,
                                                  HRepeat,
                                                  WRepeat,
                                                  KRepeat,
                                                  DsEnableLds,
                                                  Aco,
                                                  typename BlockwiseElementOp::Sba>;
    using BlockwiseFma  = BlockwiseElementFmaWcnn<ThisThreadBlock,
                                                  decltype(GetFmaDsDataType()),
                                                  AccDataType,
                                                  decltype(GetFmaDsBlockDesc()), // {Residual, Scale}
                                                  HPerBlock,
                                                  WPerBlock,
                                                  KPerBlock,
                                                  HPerWcnn,
                                                  WPerWcnn,
                                                  HRepeat,
                                                  WRepeat,
                                                  KRepeat,
                                                  DsEnableLds,
                                                  Aco,
                                                  typename BlockwiseElementOp::Fma>;

    BlockwiseSuba blockwise_suba_;
    BlockwiseFma blockwise_fma_;

    static constexpr auto ds_wave_desc_ =
        container_concat(BlockwiseSuba::ds_wave_desc_, BlockwiseFma::ds_wave_desc_);

    __host__ __device__ BlockwiseElementSbaFmaWcnn(const BlockwiseElementOp& blockOp)
        : blockwise_suba_(blockOp.sba_), blockwise_fma_(blockOp.fma_)

    {
    }

    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        auto sba_grid_desc =
            generate_tuple([&](auto i) { return ds_grid_desc[i]; }, Number<NumSbaDTensor>{});

        auto fma_grid_desc =
            generate_tuple([&](auto i) { return ds_grid_desc[Number<i + NumSbaDTensor>{}]; },
                           Number<NumFmaDTensor>{});
        return container_concat(BlockwiseSuba::MakeDsGridBlockDescriptor(sba_grid_desc),
                                BlockwiseFma::MakeDsGridBlockDescriptor(fma_grid_desc));
    }

    template <index_t conv_phase, typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeSingleDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        auto sba_grid_desc =
            generate_tuple([&](auto i) { return ds_grid_desc[i]; }, Number<NumSbaDTensor>{});

        auto fma_grid_desc =
            generate_tuple([&](auto i) { return ds_grid_desc[Number<i + NumSbaDTensor>{}]; },
                           Number<NumFmaDTensor>{});
        return container_concat(
            BlockwiseSuba::MakeSingleDsGridBlockDescriptor<conv_phase>(sba_grid_desc),
            BlockwiseFma::MakeSingleDsGridBlockDescriptor<conv_phase>(fma_grid_desc));
    }

    __host__ __device__ static auto CalculateThreadOriginDataIndex()
    {
        return container_concat(BlockwiseSuba::CalculateThreadOriginDataIndex(),
                                BlockwiseFma::CalculateThreadOriginDataIndex());
    }

    __host__ __device__ static constexpr auto GetDsWaveDescLength()
    {
        return container_concat(BlockwiseSuba::GetDsWaveDescLength(),
                                BlockwiseFma::GetDsWaveDescLength());
    }

    template <typename DsBlockBuffer,
              typename AccDataVec,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void
    Run(const DsBlockBuffer& ds_block_buf, AccDataVec& acc_vec, NumberH0, NumberW0, NumberK0) const
    {
        auto sba_block_buf =
            generate_tuple([&](auto i) { return ds_block_buf[i]; }, Number<NumSbaDTensor>{});

        auto fma_block_buf =
            generate_tuple([&](auto i) { return ds_block_buf[Number<i + NumSbaDTensor>{}]; },
                           Number<NumFmaDTensor>{});

        blockwise_suba_.Run(sba_block_buf, acc_vec, NumberH0{}, NumberW0{}, NumberK0{});
        blockwise_fma_.Run(fma_block_buf, acc_vec, NumberH0{}, NumberW0{}, NumberK0{});
    }

    template <typename SbaMemTrait, typename FmaMemTrait>
    static constexpr auto
    MergeDsMemTraid(const SbaMemTrait& sba, const FmaMemTrait& fma, const index_t offset)
    {
        auto sba_ = generate_tuple([&](auto i) { return sba[i]; }, Number<NumSbaDTensor>{});

        auto fma_ =
            generate_tuple([&](auto i) { return fma[i] + offset; }, Number<NumFmaDTensor>{});
        return container_concat(sba_, fma_);
    }

    struct SharedMemTrait
    {
        static constexpr auto max_lds_align = 8;
        static constexpr auto ds_block_space_size_aligned =
            MergeDsMemTraid(BlockwiseSuba::SharedMemTrait::ds_block_space_size_aligned,
                            BlockwiseFma::SharedMemTrait::ds_block_space_size_aligned,
                            0);

        // LDS allocation for Ds in LDS
        static constexpr index_t fma_block_space_offset =
            BlockwiseSuba::SharedMemTrait::lds_size / sizeof(AccDataType);
        static constexpr auto ds_block_space_offset =
            MergeDsMemTraid(BlockwiseSuba::SharedMemTrait::ds_block_space_offset,
                            BlockwiseFma::SharedMemTrait::ds_block_space_offset,
                            fma_block_space_offset);
        static constexpr auto lds_size =
            BlockwiseSuba::SharedMemTrait::lds_size + BlockwiseFma::SharedMemTrait::lds_size;
    };

    struct LaneSharedMemTrait
    {
        static constexpr auto max_lane_shared_align = 4;
        static constexpr index_t fma_block_space_offset =
            BlockwiseSuba::LaneSharedMemTrait::lane_shared_size;
        static constexpr auto ds_block_space_offset =
            MergeDsMemTraid(BlockwiseSuba::LaneSharedMemTrait::ds_block_space_offset,
                            BlockwiseFma::LaneSharedMemTrait::ds_block_space_offset,
                            fma_block_space_offset);
        static constexpr auto lane_shared_size =
            BlockwiseSuba::LaneSharedMemTrait::lane_shared_size +
            BlockwiseFma::LaneSharedMemTrait::lane_shared_size;
    };
};

template <typename ThisThreadBlock,
          typename OutTensorDataType,
          typename AccDataType,
          typename OutTensorBlockDesc,
          typename AccBlockDesc,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t KPerBlock,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t KPerWcnn,
          index_t CPerWcnn,
          index_t HRepeat,
          index_t WRepeat,
          index_t KRepeat,
          bool Aco,
          typename BlockwiseNextElementOp>
struct BlockwiseElementCvtTensorWcnn
{
    static constexpr auto I0            = Number<0>{};
    static constexpr auto I1            = Number<1>{};
    static constexpr auto I2            = Number<2>{};
    static constexpr index_t WaveSize   = 32;
    static constexpr index_t NumDTensor = 2;

    __device__ static auto GetWaveIdx()
    {
        return GetWcnnWaveIdx<ThisThreadBlock,
                              HPerBlock,
                              WPerBlock,
                              HRepeat,
                              WRepeat,
                              HPerWcnn,
                              WPerWcnn>();
    }

    static constexpr AccBlockDesc accum_thread_desc_;
    static constexpr OutTensorBlockDesc out_tensor_thread_desc_;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
    template <typename AccDataVec,
              typename OutTensorThreadBuffer,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void Run(const AccDataVec& acc_vec,
                        OutTensorThreadBuffer& out_thread_buf,
                        NumberH0,
                        NumberW0,
                        NumberK0) const
    {
        constexpr NumberH0 h0;
        constexpr NumberW0 w0;
        constexpr NumberK0 k0;

        constexpr auto wcnn_cvt_tensor = ck::WcnnCvtTensor<OutTensorDataType,
                                                           AccDataType,
                                                           HPerWcnn,
                                                           WPerWcnn,
                                                           BlockwiseNextElementOp::activeFunc,
                                                           BlockwiseNextElementOp::clamp>();
        constexpr index_t indata_offset =
            out_tensor_thread_desc_.CalculateOffset(make_tuple(h0, w0, k0, I0));
        auto& outVec = out_thread_buf.GetVectorTypeReference(Number<indata_offset>{});
        constexpr index_t convert_scale = BlockwiseNextElementOp::scale;
        wcnn_cvt_tensor.cvt_tensor_instr.Run(acc_vec, convert_scale, outVec);
    }
#pragma clang diagnostic pop
};

template <typename AccBlockDesc>
struct BlockwiseElementOutPassThroughWcnn
{
    template <typename AccDataVec,
              typename OutTensorThreadBuffer,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void Run(const AccDataVec& acc_vec,
                        OutTensorThreadBuffer& out_thread_buf,
                        NumberH0,
                        NumberW0,
                        NumberK0) const
    {
        constexpr auto accum_offset = accum_thread_desc_.CalculateOffset(
            make_tuple(NumberH0{}, NumberW0{}, NumberK0{}, Number<0>{}));
        out_thread_buf.GetVectorTypeReference(Number<accum_offset>{}) = acc_vec;
    }
    static constexpr AccBlockDesc accum_thread_desc_;
};

struct BlockwiseElementPassThroughWcnn
{
    static constexpr auto ds_wave_desc_ = make_tuple();
    __host__ __device__ BlockwiseElementPassThroughWcnn(const BlockwiseElementPassThrough&) {}

    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        return ds_grid_desc;
    }

    template <index_t conv_phase, typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeSingleDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        return ds_grid_desc;
    }

    __host__ __device__ static auto CalculateThreadOriginDataIndex() { return make_tuple(); }

    __host__ __device__ static constexpr auto GetDsWaveDescLength() { return make_tuple(); }

    template <typename DsBlockBuffer,
              typename AccDataVec,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void Run(const DsBlockBuffer&, AccDataVec&, NumberH0, NumberW0, NumberK0) const
    {
    }

    struct SharedMemTrait
    {
        static constexpr auto max_lds_align               = 8;
        static constexpr auto ds_block_space_size_aligned = make_tuple(0, 0);

        // LDS allocation for Ds in LDS
        static constexpr auto ds_block_space_offset = make_tuple(0, 0);
        static constexpr auto lds_size              = 0;
    };

    struct LaneSharedMemTrait
    {
        static constexpr auto max_lane_shared_align = 4;
        static constexpr auto ds_block_space_offset = make_tuple(0, 0);
        static constexpr auto lane_shared_size      = 0;
    };
};

} // namespace ck
