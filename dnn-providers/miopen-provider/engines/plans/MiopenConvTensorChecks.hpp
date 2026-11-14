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
// Throws hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, reason) when the
// tensors must be declined; `reason` names the offending tensor and the rule. Also throws when a
// tensor has null dims() or strides(), differing dims()/strides() lengths, or rank below 3. The
// tensors are the graph's own (unpadded rank 3, 4 or 5), not the padded MIOpen descriptors.
void validateConvFwdIoTensors(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& x,
                              const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& w,
                              const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& y);

} // namespace miopen_plugin::conv_applicability
