// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace miopen_plugin::conv_applicability
{

// Declines convolution tensor combinations that MIOpen accepts (it reports a solution) but then
// computes wrongly, writes out of bounds or faults on. Each rule is a workaround for a MIOpen
// defect and is documented at its check in MiopenConvTensorChecks.cpp together with the tickets
// that remove it.
//
// Forward convolution: x, w and y must
//   - share one data type, one of FLOAT, HALF, BFLOAT16;
//   - share one layout class: all channels-first or all channels-last (see the .cpp for how a
//     tensor is classified, including tensors with extent-1 dimensions);
//   - all be packed.
//
// Backward data (dy, w, dx) and backward weights (x, dy, dw): the three tensors must
//   - share one data type, one of FLOAT, HALF, BFLOAT16;
//   - share one layout class, classified as for forward;
//   - none be both channels-last and non-packed. Unlike forward, a channels-first non-packed
//     tensor is accepted: MIOpen computes it correctly.
// Backward data additionally has a size limit: dy and dx must each have at most 2^31 - 1 elements.
//
// Fused convolution + bias + activation (forward): the ADD operand that is not the convolution
// output must
//   - be a per-channel bias: dims [1, K, 1, .., 1] with K the convolution output's dims[1] and the
//     same rank as the convolution output;
//   - have the data type of the convolution input x (the io data type of the fused graph).
//
// Throws hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, reason) when the
// tensors must be declined; `reason` names the operation, the offending tensor and the rule. Also
// throws when a tensor has null dims() or strides(), differing dims()/strides() lengths, or rank
// below 3 (the size check only needs non-null, non-negative dims()). The tensors are the graph's
// own (unpadded rank 3, 4 or 5), not the padded MIOpen descriptors.

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
// with an overflow-safe product, so it is safe on any dims and must run before the io-tensor
// validators when the dims are not yet known to be sane.
void validateConvBwdDataSize(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dy,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dx);

/// Throws HipdnnPluginException(BAD_PARAM) unless `bias` (the ADD operand that is not the
/// convolution output) is a per-channel bias of the fused convolution whose output tensor is
/// `yConv`, with the data type of `x`. Reads only dims() (non-null, rank of `yConv` at least 3)
/// and data_type() of the three tensors; strides are not checked.
void validateFusedConvBiasOperand(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& x,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yConv,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& bias);

} // namespace miopen_plugin::conv_applicability
