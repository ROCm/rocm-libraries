// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend
// (include/cudnn_frontend/graph_properties.h), used under the MIT license.

/**
 * @file graph_properties.h
 * @brief Graph attribute-type aliases for the hipDNN cuDNN-compatibility shim.
 *
 * Brings hipDNN's v9 graph attribute types into `<shim_ns>::graph` by aliasing
 * them with `using` declarations — zero overhead, no shim-side state. A tensor
 * configured through the shim therefore *is* a hipDNN `TensorAttributes` and
 * flows into a wrapped hipDNN graph with no conversion (UID handling stays on
 * the hipDNN side).
 *
 * @note Internal-to-shim; pulled in by the umbrella `cudnn_frontend.h`.
 */

#pragma once

#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulBwdAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#include <hipdnn_frontend/attributes/ResampleFwdAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{

using hipdnn_frontend::half;
using hipdnn_frontend::graph::ScalarType;
using hipdnn_frontend::graph::Tensor_attributes;
using hipdnn_frontend::graph::TensorAttributes;
using nv_bfloat16 = hipdnn_frontend::bfloat16;

// Tier-1 node attribute types: each cuDNN v9
// *_attributes class with an exact hipDNN counterpart is aliased 1:1 — zero
// overhead, no wrapper. The matching Graph::* node method forwards straight to
// the wrapped hipDNN graph (see detail/graph_wrapper.h).
using hipdnn_frontend::graph::Batchnorm_attributes;
using hipdnn_frontend::graph::Batchnorm_backward_attributes;
using hipdnn_frontend::graph::Batchnorm_inference_attributes;
using hipdnn_frontend::graph::Block_scale_dequantize_attributes;
using hipdnn_frontend::graph::Block_scale_quantize_attributes;
using hipdnn_frontend::graph::Conv_dgrad_attributes;
using hipdnn_frontend::graph::Conv_fprop_attributes;
using hipdnn_frontend::graph::Conv_wgrad_attributes;
using hipdnn_frontend::graph::Layernorm_attributes;
using hipdnn_frontend::graph::Layernorm_backward_attributes;
using hipdnn_frontend::graph::Matmul_attributes;
using hipdnn_frontend::graph::Moe_grouped_matmul_attributes;
using hipdnn_frontend::graph::Moe_grouped_matmul_bwd_attributes;
using hipdnn_frontend::graph::Pointwise_attributes;
using hipdnn_frontend::graph::Reduction_attributes;
using hipdnn_frontend::graph::Resample_attributes;
using hipdnn_frontend::graph::Rmsnorm_attributes;
using hipdnn_frontend::graph::Rmsnorm_backward_attributes;

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
