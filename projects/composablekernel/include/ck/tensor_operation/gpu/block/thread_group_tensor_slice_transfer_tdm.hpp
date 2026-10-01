// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <index_t X>
__host__ __device__ constexpr index_t integer_log2_exact()
{
    static_assert(X > 0 && (X & (X - 1)) == 0,
                  "integer_log2_exact requires a positive power of two");

    index_t value  = X;
    index_t result = 0;
    while(value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

struct TDMLdsPaddingConfig
{
    uint32_t pad_interval{0};
    uint32_t pad_amount{0};
};

// context switch is not supported in MI450
struct TDMConfig
{
    // 0 - not a context-restore descriptor. 1 - from a context-restore
    bool is_restore{false};
    // 0 - don't send an atomic barrier op. 1 - send an atomic barrier op.
    bool atomic_barrier_enable{false};
    uint16_t atomic_barrier_address{0}; // only consumed when atomic_barrier_enable
    // 0 - padding disabled. 1 - add padding to lds destination address.
    bool pad_enable{false};
    TDMLdsPaddingConfig pad_config; // padding config
};

namespace detail {

union TDM_GROUP0
{
    __device__ TDM_GROUP0(uintptr_t lds_addr_in, uintptr_t global_addr_in) : bitfield{0, 0, 0, 0}
    {
        count    = 1;
        lds_addr = lds_addr_in;
        globalAddr(global_addr_in);
        type              = 2; // set to 2 for spg
        gather_index_size = 0;
        gather_mode       = 0;
    }

    __device__ void globalAddr(uintptr_t value)
    {
        global_addr_lo = value & 0xFFFFFFFF;
        global_addr_hi = (value >> 32);
    }

    struct
    {
        union
        {
            struct
            {
                uint32_t count : 2;
                uint32_t is_restore : 1;
                uint32_t is_store : 1;
                uint32_t nv : 1;
                uint32_t scope_trait : 2;
                uint32_t th : 3;
                uint32_t reserved_space : 20;
                uint32_t gather_index_size : 1;
                uint32_t gather_mode : 1;
            };
            uint32_t reserved0;
        };
        uint32_t lds_addr;
        uint32_t global_addr_lo;
        union
        {
            struct
            {
                uint32_t global_addr_hi : 25;
                uint32_t reserved2 : 5;
                uint32_t type : 2;
            };
            uint32_t sgpr3;
        };
    };
    int32x4_t bitfield;
};

union TDM_GROUP1
{
    __device__ constexpr TDM_GROUP1() : bitfield{0, 0, 0, 0, 0, 0, 0, 0} {}

    struct
    {
        union
        {
            struct
            {
                uint32_t workgroup_mask : 16;
                uint32_t data_size : 2;
                uint32_t atomic_barrier_enable : 1;
                uint32_t iterate_enable : 1;
                uint32_t pad_enable : 1;
                uint32_t early_timeout : 1;
                uint32_t pad_interval : 3;
                uint32_t pad_amount : 7;
            };
            uint32_t sgpr0;
        };
        union
        {
            struct
            {
                uint32_t atomic_barrier_address : 16;
                uint32_t tensor_dim0_lo : 16;
            };
            uint32_t sgpr1;
        };
        union
        {
            struct
            {
                uint32_t tensor_dim0_hi : 16;
                uint32_t tensor_dim1_lo : 16;
            };
            uint32_t sgpr2;
        };
        union
        {
            struct
            {
                uint32_t tensor_dim1_hi : 16;
                uint32_t tile_dim0 : 16;
            };
            uint32_t sgpr3;
        };
        union
        {
            struct
            {
                uint32_t tile_dim1 : 16;
                uint32_t tile_dim2 : 16;
            };
            uint32_t sgpr4;
        };
        union
        {
            uint32_t tensor_dim0_stride_lo;
            uint32_t sgpr5;
        };
        union
        {
            struct
            {
                uint32_t tensor_dim0_stride_hi : 16;
                uint32_t tensor_dim1_stride_lo : 16;
            };
            uint32_t sgpr6;
        };
        union
        {
            uint32_t tensor_dim1_stride_hi;
            uint32_t sgpr7;
        };
    };
    int32x8_t bitfield;

    __device__ void tensorDim0(uint32_t value)
    {
        tensor_dim0_lo = value & 0xFFFF;
        tensor_dim0_hi = (value >> 16);
    }
    __device__ void tensorDim1(uint32_t value)
    {
        tensor_dim1_lo = value & 0xFFFF;
        tensor_dim1_hi = (value >> 16);
    }
    __device__ void tensorDim0Stride(uint64_t value)
    {
        tensor_dim0_stride_lo = value & 0xFFFFFFFF;
        tensor_dim0_stride_hi = (value >> 32);
    }
    __device__ void tensorDim1Stride(uint64_t value)
    {
        tensor_dim1_stride_lo = value & 0xFFFFFFFF;
        tensor_dim1_stride_hi = (value >> 32);
    }
    __device__ void tensorDim(uint32_t index, uint32_t value)
    {
        if(index == 0)
            tensorDim0(value);
        else if(index == 1)
            tensorDim1(value);
    }
    __device__ void tensorDimStride(uint32_t index, uint64_t value)
    {
        if(index == 0)
            tensorDim0Stride(value);
        else if(index == 1)
            tensorDim1Stride(value);
    }
    __device__ void tileDim(uint32_t index, uint16_t value)
    {
        if(index == 0)
            tile_dim0 = value;
        else if(index == 1)
            tile_dim1 = value;
        else if(index == 2)
            tile_dim2 = value;
    }
};

union TDM_GROUP2
{
    __device__ constexpr TDM_GROUP2() : bitfield{0, 0, 0, 0} {}

    struct
    {
        uint32_t tensor_dim2;
        uint32_t tensor_dim3;
        uint32_t tensor_dim2_stride_lo;
        union
        {
            struct
            {
                uint32_t tensor_dim2_stride_hi : 16;
                uint32_t tile_dim3 : 16;
            };
            uint32_t sgpr3;
        };
    };

    __device__ void tensorDim2Stride(uint64_t value)
    {
        tensor_dim2_stride_lo = value & 0xFFFFFFFF;
        tensor_dim2_stride_hi = value >> 32;
    }

    int32x4_t bitfield;
};

union TDM_GROUP3
{
    __device__ constexpr TDM_GROUP3() : bitfield{0, 0, 0, 0} {}

    struct
    {
        uint32_t tensor_dim3_stride_lo;
        union
        {
            struct
            {
                uint32_t tensor_dim3_stride_hi : 16;
                uint32_t tensor_dim_4_lo : 16;
            };
            uint32_t sgpr1;
        };
        union
        {
            struct
            {
                uint32_t tensor_dim_4_hi : 16;
                uint32_t tile_dim4 : 16;
            };
            uint32_t sgpr2;
        };
        uint32_t sgpr3_reserved;
    };

    __device__ void tensorDim3Stride(uint64_t value)
    {
        tensor_dim3_stride_lo = value & 0xFFFFFFFF;
        tensor_dim3_stride_hi = value >> 32;
    }

    __device__ void tensorDim4(uint32_t value)
    {
        tensor_dim_4_lo = value & 0xFFFF;
        tensor_dim_4_hi = value >> 16;
    }

    int32x4_t bitfield;
};

struct TDMDescriptorPack
{
    int32x4_t g0;
    int32x8_t g1;
    int32x4_t g2;
    int32x4_t g3;
    int32x8_t g4;
};

template <typename DataType, index_t TensorRank>
__device__ TDMDescriptorPack MakeTDMDescriptorPack(const void* global_address,
                                                   void* local_address,
                                                   const uint32_t* tensor_dims,
                                                   const uint64_t* global_strides,
                                                   const uint16_t* box_dims,
                                                   const TDMConfig& tdm_config)
{
    TDM_GROUP0 group0{reinterpret_cast<uintptr_t>(local_address),
                      reinterpret_cast<uintptr_t>(global_address)};

    TDM_GROUP1 group1;
    group1.workgroup_mask = 0xFFFF;
    group1.data_size =
        sizeof(DataType) == 8 ? 3 : (sizeof(DataType) == 4 ? 2 : (sizeof(DataType) == 2 ? 1 : 0));
    group1.atomic_barrier_enable  = tdm_config.atomic_barrier_enable;
    group1.atomic_barrier_address = tdm_config.atomic_barrier_address;
    group1.iterate_enable         = 0;
    group1.pad_enable             = tdm_config.pad_enable;
    group1.early_timeout          = 0;
    group1.pad_interval           = tdm_config.pad_config.pad_interval;
    group1.pad_amount             = tdm_config.pad_config.pad_amount;

    static_for<0, 2, 1>{}([&](auto i) {
        if constexpr(i < TensorRank)
        {
            group1.tensorDim(i, tensor_dims[i]);
            group1.tensorDimStride(i, global_strides[i]);
        }
    });

    static_for<0, 3, 1>{}([&](auto i) {
        if constexpr(i < TensorRank)
        {
            group1.tileDim(i, box_dims[i]);
        }
    });

    TDM_GROUP2 group2;
    if constexpr(TensorRank > 2)
    {
        group2.tensor_dim2 = tensor_dims[2];
        group2.tensorDim2Stride(global_strides[2]);
    }
    if constexpr(TensorRank > 3)
    {
        group2.tensor_dim3 = tensor_dims[3];
        group2.tile_dim3   = box_dims[3];
    }

    TDM_GROUP3 group3;
    if constexpr(TensorRank > 3)
    {
        group3.tensorDim3Stride(global_strides[3]);
    }
    if constexpr(TensorRank > 4)
    {
        group3.tensorDim4(tensor_dims[4]);
        group3.tile_dim4 = box_dims[4];
    }

    int32x8_t zeros = {};
    return TDMDescriptorPack{
        group0.bitfield, group1.bitfield, group2.bitfield, group3.bitfield, zeros};
}

} // namespace detail

template <typename ThreadGroup,
          typename BlockSliceLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          index_t PadInterval,
          index_t PadAmount>
struct ThreadGroupTensorSliceTransfer_TDM
{
    static constexpr index_t nDim = remove_reference_t<SrcDesc>::GetNumOfDimension();
    static_assert(nDim == 2);
    using Index = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));
    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    static constexpr bool PadEnabled          = (PadInterval > 0) && (PadAmount > 0);
    static constexpr auto I0                  = Number<0>{};
    static constexpr auto I1                  = Number<1>{};
    static constexpr auto block_slice_lengths = BlockSliceLengths{};
    static constexpr index_t wave_size        = 32;
    static constexpr index_t wave_num         = ThreadGroup::GetNumOfThread() / wave_size;
    static constexpr auto wave_box_dims_      = []() {
        Array<uint16_t, nDim> dims{};
        static_for<0, nDim, 1>{}([&](auto i) {
            if constexpr(i == I0)
            {
                dims(i) = static_cast<uint16_t>(block_slice_lengths[i] / wave_num);
            }
            else
            {
                dims(i) = static_cast<uint16_t>(block_slice_lengths[i]);
            }
        });
        return dims;
    }();
    static constexpr auto tdm_wave_box_dims_ = []() {
        Array<uint16_t, nDim> dims{};
        static_for<0, nDim, 1>{}([&](auto i) {
            constexpr index_t ck_i = nDim - 1 - i;
            dims(i)                = wave_box_dims_[ck_i];
        });
        return dims;
    }();

    __host__
        __device__ constexpr ThreadGroupTensorSliceTransfer_TDM(const SrcDesc& src_desc,
                                                                const Index& src_block_slice_origin,
                                                                const DstDesc& dst_desc,
                                                                const Index& dst_block_slice_origin)
    {
        static_assert(ck::is_same_v<SrcData, DstData>,
                      "Direct load transfer does not support datatypes conversion. Source and "
                      "destination data types must be the same.");

        static_assert(nDim == remove_cvref_t<SrcDesc>::GetNumOfDimension() &&
                          nDim == remove_cvref_t<DstDesc>::GetNumOfDimension(),
                      "Inconsistent number of dimensions across lengths and descriptors.");

        static_assert(ThreadClusterArrangeOrder{}.At(I0) == I0,
                      "TDM POC expects dim-0 as the wave-split dimension.");

        static_assert(ThreadGroup::GetNumOfThread() % wave_size == 0,
                      "Thread count must be divisible by wave size (32).");
        static_assert(block_slice_lengths[I0] % wave_num == 0,
                      "Block dim-0 must be divisible by number of waves.");

        const auto wave_id = ThreadGroup::GetThreadId() / wave_size;

        Index wave_data_idx_begin{};
        static_for<0, nDim, 1>{}([&](auto i) {
            if constexpr(i == I0)
            {
                wave_data_idx_begin(i) = wave_id * (block_slice_lengths[i] / wave_num);
            }
            else
            {
                wave_data_idx_begin(i) = 0;
            }
        });

        SetSrcSliceOrigin(src_desc, src_block_slice_origin + wave_data_idx_begin);
        SetDstSliceOrigin(dst_desc, dst_block_slice_origin + wave_data_idx_begin);

        // TDM uses dim-0 as fastest, CK uses last dim as fastest.
        // Cache metadata in TDM order (reversed from CK order).
        static_for<0, nDim, 1>{}([&](auto i) {
            constexpr index_t ck_i = nDim - 1 - i;
            cached_tensor_dims_[i] = static_cast<uint32_t>(src_desc.GetLength(Number<ck_i>{}) -
                                                           src_slice_origin_[Number<ck_i>{}]);
        });

        const auto zero_idx = []() {
            Index idx{};
            static_for<0, nDim, 1>{}([&](auto i) { idx(i) = 0; });
            return idx;
        }();

        const auto src_base_offset = src_desc.CalculateOffset(zero_idx);

        static_for<0, nDim, 1>{}([&](auto i) {
            Index unit_idx{};
            constexpr index_t ck_i = nDim - 1 - i;
            static_for<0, nDim, 1>{}([&](auto j) { unit_idx(j) = j == Number<ck_i>{} ? 1 : 0; });
            const auto dim_stride     = src_desc.CalculateOffset(unit_idx) - src_base_offset;
            cached_global_strides_[i] = static_cast<uint64_t>(dim_stride * cached_tensor_dims_[i]);
        });
    }

    __host__ __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc,
                                               const Index& src_slice_origin_idx)
    {
        src_coord_        = make_tensor_coordinate(src_desc, src_slice_origin_idx);
        src_slice_origin_ = src_slice_origin_idx;
    }

    __host__ __device__ void SetDstSliceOrigin(const DstDesc& dst_desc,
                                               const Index& dst_slice_origin_idx)
    {
        dst_coord_        = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
        dst_slice_origin_ = dst_slice_origin_idx;
    }

    __host__ __device__ void ResetDstSliceWindow(const DstDesc& dst_desc)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_);
    }

    __device__ void PrecomputeIdx(const SrcDesc& src_desc)
    {
        static_for<0, nDim, 1>{}([&](auto i) {
            constexpr index_t ck_i = nDim - 1 - i;
            cached_tensor_dims_[i] = static_cast<uint32_t>(math::max(
                0, src_desc.GetLength(Number<ck_i>{}) - src_slice_origin_[Number<ck_i>{}]));
        });
    }

    template <typename SrcBuffer, typename DstBuffer>
    __device__ void Load(const SrcBuffer& src_buf, const DstDesc&, DstBuffer& dst_buf)
    {
        static_assert(SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Global,
                      "Source data must come from a global memory buffer.");
        static_assert(DstBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "Destination data must be stored in an LDS memory buffer.");

        static_assert(
            ck::is_same_v<remove_cvref_t<typename SrcBuffer::type>, remove_cvref_t<SrcData>>,
            "SrcBuffer and SrcData data types must be consistent.");
        static_assert(
            ck::is_same_v<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>,
            "DstBuffer and DstData data types must be consistent.");
        static_assert(2 <= nDim && nDim <= 5,
                      "TDM proof-of-concept currently supports tensor rank in [2, 5].");
        static_assert(ThreadClusterArrangeOrder{}.At(I0) == I0,
                      "TDM POC expects dim-0 as the wave-split dimension.");

        static_assert(ThreadGroup::GetNumOfThread() % wave_size == 0,
                      "Thread count must be divisible by wave size (32).");
        static_assert(block_slice_lengths[I0] % wave_num == 0,
                      "Block dim-0 must be divisible by number of waves.");

        // POC: load one wave tile with a single TDM instruction.
        // cached_tensor_dims_ describe the full source tensor shape.
        // wave_box_dims_ describe the tile shape loaded by this wave.
        TDMConfig tdm_config{};

        if constexpr(PadEnabled)
        {
            tdm_config.pad_enable              = PadEnabled;
            tdm_config.pad_config.pad_interval = integer_log2_exact<PadInterval / 4>() - 1;
            tdm_config.pad_config.pad_amount   = PadAmount / 4 - 1;
        }

        auto tdm_desc = detail::MakeTDMDescriptorPack<remove_cvref_t<SrcData>, nDim>(
            src_buf.p_data_ + src_coord_.GetOffset(),
            dst_buf.p_data_ + dst_coord_.GetOffset(),
            cached_tensor_dims_,
            cached_global_strides_,
            tdm_wave_box_dims_.mData,
            tdm_config);

        __builtin_amdgcn_tensor_load_to_lds(bit_cast<uint32x4_t>(tdm_desc.g0),
                                            tdm_desc.g1,
                                            tdm_desc.g2,
                                            tdm_desc.g3,
                                            tdm_desc.g4,
                                            static_cast<index_t>(0));
    }

    __host__ __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& step)
    {
        src_slice_origin_ = src_slice_origin_ + step;
        src_coord_        = make_tensor_coordinate(src_desc, src_slice_origin_);
    }

    private:
    SrcCoord src_coord_;
    DstCoord dst_coord_;
    Index src_slice_origin_;
    Index dst_slice_origin_;
    uint32_t cached_tensor_dims_[nDim];
    uint64_t cached_global_strides_[nDim];
};

} // namespace ck
