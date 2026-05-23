// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/grouped_convolution/warp/warp_conv_impl.hpp"

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
    template <typename ADataType, typename BDataType, typename AccDataType, bool AcoFlag>
    using Type =
        Wcnn1x1ConvImpl<ADataType, BDataType, AccDataType, AcoFlag, HPerWcnn, WPerWcnn, NumIter>;
};

} // namespace warp_conv_dispatcher
} // namespace impl

// Detect device arch type at compile time
// get_device_arch() returns gfx120_t during host compilation pass,
// so we use preprocessor macros instead
#if defined(__gfx13__)
using wcnn_device_arch_t = gfx13_t;
#else
using wcnn_device_arch_t = gfx13_t; // default to gfx13 for WCNN
#endif

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          bool AcoFlag,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t FilterSizeY,
          index_t FilterSizeX,
          index_t DilationY = 1,
          index_t DilationX = 1,
          index_t NumIter   = 1>
using WarpConvDispatcher = typename impl::warp_conv_dispatcher::Dispatcher<
    wcnn_device_arch_t,
    HPerWcnn,
    WPerWcnn,
    FilterSizeY,
    FilterSizeX,
    DilationY,
    DilationX,
    NumIter>::template Type<ADataType, BDataType, AccDataType, AcoFlag>;

} // namespace ck_tile
