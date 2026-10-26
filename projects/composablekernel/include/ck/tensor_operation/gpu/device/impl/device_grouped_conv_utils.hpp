// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

#ifndef CK_CODE_GEN_RTC
// Optional capability, separate from existing convolution operator vtables.
// A promise covers the supported invocation's preprocessing, internal clears and
// final conversion: every logical output is independent of its prior contents.
struct GroupedConvOutputDefinition
{
    virtual ~GroupedConvOutputDefinition()                                = default;
    virtual bool IsOutputFullyDefined(const BaseArgument* argument) const = 0;
};

inline bool IsOutputFullyDefined(const BaseOperator& operation, const BaseArgument* argument)
{
    const auto* capability = dynamic_cast<const GroupedConvOutputDefinition*>(&operation);
    return capability != nullptr && capability->IsOutputFullyDefined(argument);
}
#endif

// Packed channel-last image addressing in logical [G, N, C/K, spatial...] order.
// Deliberately excludes transposed and merged layouts: a contiguous clear and the
// per-group GEMM stores must refer to the same distinct logical elements.
template <index_t NDimSpatial, typename Layout, typename Lengths, typename Strides>
bool is_packed_grouped_conv_output(const Lengths& lengths, const Strides& strides)
{
    using namespace tensor_layout::convolution;
    constexpr bool group_interleaved =
        (NDimSpatial == 1 && (is_same_v<Layout, NWGC> || is_same_v<Layout, NWGK>)) ||
        (NDimSpatial == 2 && (is_same_v<Layout, NHWGC> || is_same_v<Layout, NHWGK>)) ||
        (NDimSpatial == 3 && (is_same_v<Layout, NDHWGC> || is_same_v<Layout, NDHWGK>));
    constexpr bool group_major =
        (NDimSpatial == 1 && (is_same_v<Layout, GNWC> || is_same_v<Layout, GNWK>)) ||
        (NDimSpatial == 2 && (is_same_v<Layout, GNHWC> || is_same_v<Layout, GNHWK>)) ||
        (NDimSpatial == 3 && (is_same_v<Layout, GNDHWC> || is_same_v<Layout, GNDHWK>));
    if constexpr(!group_interleaved && !group_major)
        return false;

    long_index_t packed_stride = 1;
    const auto consume         = [&](index_t i) {
        if(lengths[i] <= 0 || strides[i] != packed_stride ||
           lengths[i] > 0x7fffffffffffffffLL / packed_stride)
            return false;
        packed_stride *= lengths[i];
        return true;
    };
    if(!consume(2))
        return false;
    if constexpr(group_interleaved)
        if(!consume(0))
            return false;
    for(index_t d = NDimSpatial; d > 0; --d)
        if(!consume(d + 2))
            return false;
    if(!consume(1))
        return false;
    if constexpr(group_major)
        if(!consume(0))
            return false;
    return true;
}

// 1d
template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NWGC_GKXC_NWGK()
{
    return is_same_v<InLayout, tensor_layout::convolution::NWGC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NWGK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_GNWC_GKXC_GNWK()
{
    return is_same_v<InLayout, tensor_layout::convolution::GNWC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::GNWK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCW_GKXC_NGKW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCW> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKW>;
}

// 2d
template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NHWGC_GKYXC_NHWGK()
{
    return is_same_v<InLayout, tensor_layout::convolution::NHWGC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NHWGK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_GNHWC_GKYXC_GNHWK()
{
    return is_same_v<InLayout, tensor_layout::convolution::GNHWC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::GNHWK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCHW_GKYXC_NGKHW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCHW> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKHW>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCHW_GKCYX_NGKHW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCHW> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKCYX> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKHW>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCHW_NGKHW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCHW> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKHW>;
}

// 3d
template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NDHWGC_GKZYXC_NDHWGK()
{
    return is_same_v<InLayout, tensor_layout::convolution::NDHWGC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKZYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NDHWGK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_GNDHWC_GKZYXC_GNDHWK()
{
    return is_same_v<InLayout, tensor_layout::convolution::GNDHWC> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKZYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::GNDHWK>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCDHW_GKZYXC_NGKDHW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCDHW> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKZYXC> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKDHW>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCDHW_GKCZYX_NGKDHW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCDHW> &&
           is_same_v<WeiLayout, tensor_layout::convolution::GKCZYX> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKDHW>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCDHW_NGKDHW()
{
    return is_same_v<InLayout, tensor_layout::convolution::NGCDHW> &&
           is_same_v<OutLayout, tensor_layout::convolution::NGKDHW>;
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NSpatialGC_GKSpatial_NSpatialGK()
{
    return is_NWGC_GKXC_NWGK<InLayout, WeiLayout, OutLayout>() ||
           is_NHWGC_GKYXC_NHWGK<InLayout, WeiLayout, OutLayout>() ||
           is_NDHWGC_GKZYXC_NDHWGK<InLayout, WeiLayout, OutLayout>();
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_GNSpatialC_GKSpatial_GNSpatialK()
{
    return is_GNWC_GKXC_GNWK<InLayout, WeiLayout, OutLayout>() ||
           is_GNHWC_GKYXC_GNHWK<InLayout, WeiLayout, OutLayout>() ||
           is_GNDHWC_GKZYXC_GNDHWK<InLayout, WeiLayout, OutLayout>();
}

template <typename InLayout, typename WeiLayout, typename OutLayout>
constexpr bool is_NGCSpatial_GKSpatial_NGKSpatial()
{
    return is_NGCW_GKXC_NGKW<InLayout, WeiLayout, OutLayout>() ||
           is_NGCHW_GKYXC_NGKHW<InLayout, WeiLayout, OutLayout>() ||
           is_NGCDHW_GKZYXC_NGKDHW<InLayout, WeiLayout, OutLayout>();
}

template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0, typename = void>
struct ComputePtrOffsetOfStridedBatch
{
};

template <index_t NumATensor, index_t NumBTensor, index_t NumDTensor>
struct ComputePtrOffsetOfStridedBatch<NumATensor,
                                      NumBTensor,
                                      NumDTensor,
                                      enable_if_t<(NumATensor > 1 || NumBTensor > 1)>>
{
    ComputePtrOffsetOfStridedBatch() = default;

    ComputePtrOffsetOfStridedBatch(Array<long_index_t, NumATensor>& BatchStrideAs,
                                   Array<long_index_t, NumBTensor>& BatchStrideBs,
                                   Array<long_index_t, NumDTensor>& BatchStrideDs,
                                   long_index_t BatchStrideE)
        : BatchStrideA_(BatchStrideAs),
          BatchStrideB_(BatchStrideBs),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE)
    {
    }

    __host__ __device__ constexpr auto GetAsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumATensor> as_offset;
        static_for<0, NumATensor, 1>{}(
            [&](auto i) { as_offset(i) = static_cast<long_index_t>(g_idx) * BatchStrideA_[i]; });
        return as_offset;
    }

    __host__ __device__ constexpr auto GetBsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumBTensor> bs_offset;
        static_for<0, NumBTensor, 1>{}(
            [&](auto i) { bs_offset(i) = static_cast<long_index_t>(g_idx) * BatchStrideB_[i]; });
        return bs_offset;
    }

    __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumDTensor> ds_offset;
        static_for<0, NumDTensor, 1>{}(
            [&](auto i) { ds_offset(i) = static_cast<long_index_t>(g_idx) * BatchStrideDs_[i]; });
        return ds_offset;
    }

    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetEPtrOffset(index_t g_idx) const
    {
        return static_cast<long_index_t>(g_idx) * BatchStrideE_;
    }

    // alias for kernels without multiple D
    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
    {
        return static_cast<long_index_t>(g_idx) * BatchStrideE_;
    }

    Array<long_index_t, NumATensor> BatchStrideA_{};
    Array<long_index_t, NumBTensor> BatchStrideB_{};
    Array<long_index_t, NumDTensor> BatchStrideDs_{};
    long_index_t BatchStrideE_{};
    long_index_t& BatchStrideC_ = BatchStrideE_; // alias for kernels without multiple D
};

template <index_t NumATensor, index_t NumBTensor, index_t NumDTensor>
struct ComputePtrOffsetOfStridedBatch<NumATensor,
                                      NumBTensor,
                                      NumDTensor,
                                      enable_if_t<(NumATensor == 1 && NumBTensor == 1)>>
{
    ComputePtrOffsetOfStridedBatch() = default;

    ComputePtrOffsetOfStridedBatch(long_index_t BatchStrideA,
                                   long_index_t BatchStrideB,
                                   Array<long_index_t, NumDTensor> BatchStrideDs,
                                   long_index_t BatchStrideE)
        : BatchStrideA_(BatchStrideA),
          BatchStrideB_(BatchStrideB),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE)
    {
    }

    __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
    {
        return static_cast<long_index_t>(g_idx) * BatchStrideA_;
    }

    __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
    {
        return static_cast<long_index_t>(g_idx) * BatchStrideB_;
    }

    __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
    {
        Array<long_index_t, NumDTensor> ds_offset;
        static_for<0, NumDTensor, 1>{}(
            [&](auto i) { ds_offset(i) = static_cast<long_index_t>(g_idx) * BatchStrideDs_[i]; });
        return ds_offset;
    }

    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetEPtrOffset(index_t g_idx) const
    {
        return static_cast<long_index_t>(g_idx) * BatchStrideE_;
    }

    // alias for kernels without multiple D
    [[maybe_unused]] __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
    {
        return static_cast<long_index_t>(g_idx) * BatchStrideE_;
    }

    long_index_t BatchStrideA_{};
    long_index_t BatchStrideB_{};
    Array<long_index_t, NumDTensor> BatchStrideDs_{};
    long_index_t BatchStrideE_{};
    long_index_t& BatchStrideC_ = BatchStrideE_; // alias for kernels without multiple D
};

template <bool isTuple, typename Tensors>
constexpr static auto GetNumABTensors()
{
    if constexpr(isTuple)
    {
        return Number<Tensors::Size()>{};
    }
    else
    {
        return Number<1>{};
    }
}

template <bool isTuple, typename GridwiseGemm, typename DataType>
constexpr static auto GetAGridPointer()
{
    if constexpr(isTuple)
    {
        return typename GridwiseGemm::AsGridPointer{};
    }
    else
    {
        return Tuple<const DataType*>{};
    }
}

template <bool isTuple, typename GridwiseGemm, typename DataType>
constexpr static auto GetBGridPointer()
{
    if constexpr(isTuple)
    {
        return typename GridwiseGemm::BsGridPointer{};
    }
    else
    {
        return Tuple<const DataType*>{};
    }
}

template <bool isTuple, typename Id, typename Type>
constexpr static auto UnpackDataType()
{
    if constexpr(isTuple)
    {
        // unpack if tuple
        return tuple_element_t<Id{}, Type>{};
    }
    else
    {
        // if no, return Type
        return Type{};
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
