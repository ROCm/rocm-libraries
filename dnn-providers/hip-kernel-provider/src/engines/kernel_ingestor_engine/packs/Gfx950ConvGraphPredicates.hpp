// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvGeometryCommon.hpp"

/// @file Gfx950ConvGraphPredicates.hpp
/// The tensor and attribute checks of the gfx950 conv AOT engines' graph contracts, the
/// same for every direction: which tensors the kernels can address, packed NHWC strides,
/// 2-D attributes.
namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv
{

/// A tensor the kernels address: non-virtual, passed by pointer, not ragged, 16-byte
/// aligned, fp16 or bf16.
inline bool isDeviceTensor(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor)
{
    return tensor != nullptr && !tensor->virtual_()
           && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor)
           && !tensor->ragged_offset_tensor_uid().has_value() && tensor->alignment() >= 16
           && tensor->alignment() % 16 == 0
           && (tensor->data_type() == hipdnn_flatbuffers_sdk::data_objects::DataType::HALF
               || tensor->data_type() == hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
}

/// hipDNN [N, C, H, W] dims stored packed NHWC (an activation as NHWC or NHWK, a filter
/// [K, C/G, Y, X] as KYXC), whose 16-bit elements span less than 2^62 bytes.
inline bool
    hasPackedNhwcStrides(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor)
{
    if(tensor == nullptr || tensor->dims() == nullptr || tensor->strides() == nullptr
       || tensor->dims()->size() != 4 || tensor->strides()->size() != 4)
    {
        return false;
    }
    const auto* dims = tensor->dims();
    const std::array<int64_t, 4> extents{dims->Get(0), dims->Get(1), dims->Get(2), dims->Get(3)};
    if(std::any_of(extents.begin(), extents.end(), [](int64_t e) { return e <= 0; })
       || !wideProduct({extents[0], extents[1], extents[2], extents[3], 2}))
    {
        return false;
    }
    // The byte-size guard above bounds every multiplication here.
    const std::array<int64_t, 4> expected{
        extents[1] * extents[2] * extents[3], 1, extents[1] * extents[3], extents[1]};
    for(flatbuffers::uoffset_t axis = 0; axis < 4; ++axis)
    {
        // A unit-extent axis never contributes to an address, whatever its stride.
        if(extents[axis] != 1 && tensor->strides()->Get(axis) != expected[axis])
        {
            return false;
        }
    }
    return true;
}

/// A 2-D convolution attribute (stride, dilation, padding): two values.
inline bool hasSpatialRankTwo(const flatbuffers::Vector<int64_t>* values)
{
    return values != nullptr && values->size() == 2;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv
