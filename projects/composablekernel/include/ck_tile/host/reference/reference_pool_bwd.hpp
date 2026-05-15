// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

// Host reference for max-pool backward, used by the GPU tests.
//
// IMPORTANT (packing): this reference treats `dout`, `indices`, and `din` as
// linear flat buffers via `HostTensor::mData[i]`. It does NOT honor the
// HostTensor strides. Callers must pass packed-memory tensors (the natural
// NHWC/NDHWC layouts produced by the example, the host test, and the
// tile-engine test all satisfy this). The GPU bwd kernel makes the same
// assumption: `indices[i]` is interpreted as a flat offset into the packed
// `din` buffer.
//
// The accumulation is always performed in float regardless of
// `ComputeDataType`, matching the GPU's fp32-workspace path for fp16/bf16
// inputs and being a no-op for fp32 inputs. Using fp32 here also keeps the
// reference deterministic and free of intermediate quantization that would
// otherwise be needed for half precision.
template <typename DOutDataType,
          typename IndexDataType,
          typename ComputeDataType,
          typename DInDataType>
CK_TILE_HOST void reference_pool_bwd(const HostTensor<DOutDataType>& dout,
                                     const HostTensor<IndexDataType>& indices,
                                     HostTensor<DInDataType>& din)
{
    static_assert(
        std::is_same_v<ComputeDataType, float>,
        "reference_pool_bwd currently accumulates in float and only supports ComputeDataType=float. "
        "Pass ComputeDataType=float to keep the reference deterministic.");

    const std::size_t din_length  = din.get_element_space_size();
    const std::size_t dout_length = dout.get_element_space_size();

    std::vector<float> buf(din_length, 0.0f);

    for(std::size_t i = 0; i < dout_length; ++i)
    {
        const IndexDataType idx = indices.mData[i];
        if(idx >= 0 && static_cast<std::size_t>(idx) < din_length)
        {
            const float dout_v = type_convert<float>(dout.mData[i]);
            buf[static_cast<std::size_t>(idx)] += dout_v;
        }
    }

    for(std::size_t i = 0; i < din_length; ++i)
    {
        din.mData[i] = type_convert<DInDataType>(buf[i]);
    }
}

} // namespace ck_tile
