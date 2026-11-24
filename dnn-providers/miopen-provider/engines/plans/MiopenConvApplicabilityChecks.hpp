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
//  - the bias is not a per-channel tensor: of rank at most the convolution output's (a lower-rank
//    bias is right-aligned against the output, its missing leading dimensions being units), dims
//    [1, C, 1, ...] once so aligned, with C equal to the convolution output channels and, when
//    C > 1, a unit channel stride (a single-channel bias is accepted with any channel stride);
//  - the bias data type differs from the convolution input data type, which is the io data type
//    of the fused graph (the fused MIOpen solver does not validate it and computes garbage);
//  - the final activation output differs from the descriptor MIOpen derives for the fused
//    convolution: the input data type and dims equal to the convolution output, in the same
//    packed memory layout as the input. MIOpen derives that layout from the input strides
//    (TensorDescriptor::find_permutation, GetLayout and GetLayout_str in
//    projects/miopen/src/tensor.cpp; ConvolutionDescriptor::GetForwardOutputTensor[WithLayout] in
//    projects/miopen/src/convolution.cpp) and FusionPlanDescriptor::Execute rejects a mismatching
//    output ("The output descriptors dont match.", projects/miopen/src/fusion.cpp). No public API
//    returns the fused output strides, so the check reproduces them; MIOpen's rejection of a
//    layout it does not name is left to the device check.
// `biasAttr` is null for conv -> activation graphs.
void checkConvFwdBiasActivLayouts(
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr,
    const TensorMap& tensorMap);

// Declines convolution tensor combinations that MIOpen accepts (it reports a solution) but then
// computes wrongly, writes out of bounds or faults on. Each rule is a workaround for a MIOpen
// defect and is documented at its check in MiopenConvApplicabilityChecks.cpp together with the
// tickets that remove it.
//
// Forward convolution: x, w and y must
//   - share one data type, one of FLOAT, HALF, BFLOAT16;
//   - share one layout class: all channels-first or all channels-last (see the .cpp for how a
//     tensor is classified, including tensors with extent-1 dimensions). Three packed tensors
//     of rank 5 or below whose strides MIOpen's strict layout check places in one layout are
//     accepted as well, even when the strides of their extent-1 dimensions differ from the
//     canonical ones;
//   - for a non-packed tensor: belong to an ungrouped convolution that is not 3D channels-last;
//   - packed or not, have strides that provably keep every element at its own address (a
//     conservative check: it also declines some strides whose elements are all distinct).
//
// Backward data (dy, w, dx) and backward weights (x, dy, dw): the three tensors must
//   - share one data type, one of FLOAT, HALF, BFLOAT16;
//   - share one layout class, classified as for forward (without the strict-layout acceptance);
//   - none be both channels-last and non-packed. A channels-first non-packed tensor is accepted:
//     MIOpen computes it correctly.
// Backward data additionally has a size limit: dy and dx must each have at most 2^31 - 1 elements.
//
// Throws hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, reason) when the
// tensors must be declined; `reason` names the operation, the offending tensor and the rule. Also
// throws when a tensor has null dims() or strides(), differing dims()/strides() lengths, rank
// below 3, a negative dim, dims whose product (a zero dim counted as 1, so a zero extent does
// not hide dims that overflow) reaches INT64_MAX, or strides for which the signed 64-bit
// arithmetic of the SDK packing check, the sum of (dim - 1) * stride over the dims plus 1,
// overflows. These are declined before the packing check and before the canonical strides are
// formed, so the io validators never do arithmetic on the dims or strides they are given that
// could overflow (the size check only needs non-null, non-negative dims()). Strides that do not
// overflow are never declined for their value, negative ones included. The tensors are the
// graph's own (unpadded rank 3, 4 or 5), not the padded MIOpen descriptors.

// Forward convolution: data type, layout and packing rules above.
void validateConvFwdIoTensors(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& x,
                              const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& w,
                              const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& y);

// Backward data: data type, layout and channels-last packing rules for dy, w and dx.
void validateConvBwdDataIoTensors(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dy,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& w,
                                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dx);

// Backward weights: data type, layout and channels-last packing rules for x, dy and dw.
void validateConvWrwIoTensors(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& x,
                              const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dy,
                              const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dw);

// Backward data size limit: dy and dx must each have at most 2^31 - 1 elements. Reads only dims(),
// with an overflow-safe product, so it is safe on any dims. It runs before the io-tensor
// validators because it enforces the 2^31 - 1 limit, which the io validators' overflow guard (a
// far larger bound) does not.
void validateConvBwdDataSize(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dy,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dx);

} // namespace miopen_plugin::conv_applicability
