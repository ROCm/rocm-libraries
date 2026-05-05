// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

template <typename DOutDataType,
          typename IndexDataType,
          typename ComputeDataType,
          typename DInDataType>
CK_TILE_HOST void reference_pool_bwd(const HostTensor<DOutDataType>& dout,
                                     const HostTensor<IndexDataType>& indices,
                                     HostTensor<DInDataType>& din)
{
    const std::size_t din_length  = din.get_element_space_size();
    const std::size_t dout_length = dout.get_element_space_size();

    std::vector<ComputeDataType> buf(din_length, ComputeDataType{0});

    for(std::size_t i = 0; i < dout_length; ++i)
    {
        const IndexDataType idx = indices.mData[i];
        if(idx >= 0 && static_cast<std::size_t>(idx) < din_length)
        {
            const float dout_v = type_convert<float>(dout.mData[i]);
            const float buf_v  = type_convert<float>(buf[static_cast<std::size_t>(idx)]);
            buf[static_cast<std::size_t>(idx)] = type_convert<ComputeDataType>(buf_v + dout_v);
        }
    }

    for(std::size_t i = 0; i < din_length; ++i)
    {
        din.mData[i] = type_convert<DInDataType>(buf[i]);
    }
}

} // namespace ck_tile
