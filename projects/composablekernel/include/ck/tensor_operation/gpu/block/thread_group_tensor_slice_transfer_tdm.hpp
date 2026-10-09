// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"

namespace ck {

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
    static_assert(1 <= nDim && nDim <= 5, "TDM supports tensor rank in [1, 5].");
    static_assert(BlockSliceLengths::Size() == nDim,
                  "Block slice lengths must match the tensor rank.");
    static_assert(ThreadClusterArrangeOrder::Size() == nDim,
                  "Thread cluster arrange order must match the tensor rank.");
    static_assert(is_valid_sequence_map<ThreadClusterArrangeOrder>::value,
                  "Thread cluster arrange order must be a dimension permutation.");
    using Index = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));
    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    static constexpr bool PadEnabled          = (PadInterval > 0) && (PadAmount > 0);
    static constexpr auto I0                  = Number<0>{};
    static constexpr auto wave_split_dim      = ThreadClusterArrangeOrder{}.At(I0);
    static constexpr auto block_slice_lengths = BlockSliceLengths{};
    static constexpr index_t wave_size        = 32;
    static constexpr index_t wave_num         = ThreadGroup::GetNumOfThread() / wave_size;
    static_assert(ThreadGroup::GetNumOfThread() > 0 &&
                      ThreadGroup::GetNumOfThread() % wave_size == 0,
                  "Thread count must be a positive multiple of wave size (32).");
    static_assert(block_slice_lengths[wave_split_dim] % wave_num == 0,
                  "Block slowest dimension must be divisible by number of waves.");
    static constexpr auto wave_box_dims_ = []() {
        Array<uint16_t, nDim> dims{};
        static_for<0, nDim, 1>{}([&](auto i) {
            if constexpr(i == wave_split_dim)
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
            constexpr index_t ck_i = ThreadClusterArrangeOrder{}.At(Number<nDim - 1 - i>{});
            dims(i)                = wave_box_dims_[ck_i];
        });
        return dims;
    }();

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

        auto wave_id = amd_wave_read_first_lane(ThreadGroup::GetThreadId() / wave_size);

        Index wave_data_idx_begin{};
        static_for<0, nDim, 1>{}([&](auto i) {
            if constexpr(i == wave_split_dim)
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

        // Cache tensor dims in TDM order (reversed from arrange order).
        static_for<0, nDim, 1>{}([&](auto i) {
            constexpr index_t ck_i = ThreadClusterArrangeOrder{}.At(Number<nDim - 1 - i>{});
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
            constexpr index_t ck_i = ThreadClusterArrangeOrder{}.At(Number<nDim - 1 - i>{});
            static_for<0, nDim, 1>{}([&](auto j) { unit_idx(j) = j == Number<ck_i>{} ? 1 : 0; });
            const auto dim_stride     = src_desc.CalculateOffset(unit_idx) - src_base_offset;
            cached_global_strides_[i] = static_cast<uint64_t>(dim_stride * cached_tensor_dims_[i]);
        });
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_coord_        = make_tensor_coordinate(src_desc, src_slice_origin_idx);
        src_slice_origin_ = src_slice_origin_idx;
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_        = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
        dst_slice_origin_ = dst_slice_origin_idx;
    }

    __device__ void ResetDstSliceWindow(const DstDesc& dst_desc)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_);
    }

    __device__ void PrecomputeIdx(const SrcDesc& src_desc)
    {
        static_for<0, nDim, 1>{}([&](auto i) {
            constexpr index_t ck_i = ThreadClusterArrangeOrder{}.At(Number<nDim - 1 - i>{});
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
        // load one wave tile with a single TDM instruction.
        // cached_tensor_dims_ describe the full source tensor shape.
        // wave_box_dims_ describe the tile shape loaded by this wave.
        TDMConfig tdm_config{};

        if constexpr(PadEnabled)
        {
            tdm_config.pad_enable              = PadEnabled;
            tdm_config.pad_config.pad_interval = math::integer_log2_exact<PadInterval / 4>() - 1;
            tdm_config.pad_config.pad_amount   = PadAmount / 4 - 1;
        }

        src_buf.template TdmCopyToLds<nDim, remove_cvref_t<decltype(dst_buf)>>(
            dst_buf,
            src_coord_.GetOffset(),
            dst_coord_.GetOffset(),
            cached_tensor_dims_,
            cached_global_strides_,
            tdm_wave_box_dims_.mData,
            tdm_config);
    }

    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& step)
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
