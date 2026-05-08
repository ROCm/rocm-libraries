// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/grouped_convolution/warp/warp_conv.hpp"

namespace ck_tile {

namespace impl {
namespace warp_conv_dispatcher {

// Primary template — static_assert on unsupported configurations
template <typename DeviceArch,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t FilterSizeY,
          index_t FilterSizeX,
          index_t DilationY,
          index_t DilationX,
          index_t NumIter>
struct Dispatcher
{
    static_assert(HPerWcnn == 0,
                  "Unsupported WarpConv configuration: no specialization for "
                  "the given tile size, filter size, or dilation");
};

// 1x1 filter — tile size and dilation forwarded to WcnnConvImpl
template <index_t HPerWcnn, index_t WPerWcnn, index_t DilationY, index_t DilationX, index_t NumIter>
struct Dispatcher<gfx13_t, HPerWcnn, WPerWcnn, 1, 1, DilationY, DilationX, NumIter>
{
    template <typename ImgDataType,
              typename WeiDataType,
              typename InAccDataType,
              typename OutAccDataType,
              typename OutDataType>
    using Type = Wcnn1x1ConvImpl<ImgDataType,
                                 WeiDataType,
                                 InAccDataType,
                                 OutAccDataType,
                                 wcnn_mods::GetAcoFlag<OutDataType>(),
                                 HPerWcnn,
                                 WPerWcnn,
                                 NumIter>;
};

} // namespace warp_conv_dispatcher
} // namespace impl

template <typename ImgDataType,
          typename WeiDataType,
          typename InAccDataType,
          typename OutAccDataType,
          typename OutDataType,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t FilterSizeY,
          index_t FilterSizeX,
          index_t DilationY = 1,
          index_t DilationX = 1,
          index_t NumIter   = 1>
using WarpConvDispatcher = typename impl::warp_conv_dispatcher::Dispatcher<
    decltype(get_device_arch()),
    HPerWcnn,
    WPerWcnn,
    FilterSizeY,
    FilterSizeX,
    DilationY,
    DilationX,
    NumIter>::template Type<ImgDataType, WeiDataType, InAccDataType, OutAccDataType, OutDataType>;

} // namespace ck_tile
