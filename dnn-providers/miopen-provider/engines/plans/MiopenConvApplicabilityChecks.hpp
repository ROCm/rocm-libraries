// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace miopen_plugin::conv_applicability
{

using TensorMap
    = std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>;

// Convolution and fused convolution graphs carry only device-buffer tensors. A tensor with a
// pass-by-value role (runtime scalar or baked constant) has no device buffer for the plan to
// look up, so any such tensor makes the graph unservable.
bool hasPassByValueTensor(const TensorMap& tensorMap);

// Layout checks for conv -> [bias add] -> activation graphs. Throws BAD_PARAM when:
//  - the bias is not a per-channel tensor: dims [1, C, 1, ...] with C equal to the convolution
//    output channels, the same rank as the convolution output, and a unit channel stride;
//  - the final activation output differs from the descriptor MIOpen derives for the fused
//    convolution: the input data type and dims equal to the convolution output, in the same
//    packed memory layout as the input.
// `biasAttr` is null for conv -> activation graphs.
void checkConvFwdBiasActivLayouts(
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr,
    const TensorMap& tensorMap);

} // namespace miopen_plugin::conv_applicability
