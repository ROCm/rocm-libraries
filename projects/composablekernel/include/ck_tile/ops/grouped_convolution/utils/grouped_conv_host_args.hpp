// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {

enum class GroupedConvDirection
{
    FORWARD,
    BACKWARD_DATA,
    BACKWARD_WEIGHT
};

/// @brief The Grouped Conv kernel host arguments.
///
/// @par Overview
///      This structure is passed to Grouped Convolution Kernels when creating kernel
///      arguments object. It contain all necessary information required to
///      build proper kernel argument and launch kernel on GPU.
template <typename InPtr, typename WeiPtr, typename OutPtr, typename CDElementwise>
struct GroupedConvHostArgs : public conv::ConvParam
{
    CK_TILE_HOST GroupedConvHostArgs() = delete;
    CK_TILE_HOST GroupedConvHostArgs(ConvParam conv_param,
                                     InPtr in_ptr_,
                                     WeiPtr wei_ptr_,
                                     const std::vector<const void*> ds_ptr_,
                                     OutPtr out_ptr_,
                                     index_t k_batch_,
                                     CDElementwise elfunc_ = CDElementwise{})
        : conv::ConvParam(conv_param),
          in_ptr(in_ptr_),
          wei_ptr(wei_ptr_),
          ds_ptr(ds_ptr_),
          out_ptr(out_ptr_),
          k_batch(k_batch_),
          elfunc(elfunc_)
    {
    }

    InPtr in_ptr;
    WeiPtr wei_ptr;
    const std::vector<const void*> ds_ptr;
    OutPtr out_ptr;
    index_t k_batch;
    const CDElementwise elfunc;
};

using PassThrough = ck_tile::element_wise::PassThrough;

template <typename CDElementwise = PassThrough>
using GroupedConvFwdHostArgs = GroupedConvHostArgs<const void*, const void*, void*, CDElementwise>;
using GroupedConvBwdWeightHostArgs =
    GroupedConvHostArgs<const void*, void*, const void*, PassThrough>;
using GroupedConvBwdDataHostArgs =
    GroupedConvHostArgs<void*, const void*, const void*, PassThrough>;

} // namespace ck_tile
