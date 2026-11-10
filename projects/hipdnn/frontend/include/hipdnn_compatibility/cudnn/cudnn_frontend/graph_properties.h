// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend/graph_properties.h
//     Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file graph_properties.h
 * @brief cuDNN FE tensor and node attribute types, aliased 1:1 to hipDNN types.
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
#include <hipdnn_frontend/attributes/RopeAttributes.hpp>
#include <hipdnn_frontend/attributes/SoftmaxAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{

/// @brief Alias of hipdnn_frontend::half.
using hipdnn_frontend::half;
using hipdnn_frontend::graph::ScalarType;
using hipdnn_frontend::graph::Tensor_attributes;
/// @brief hipDNN spelling of Tensor_attributes.
using hipdnn_frontend::graph::TensorAttributes;
/// @brief Alias of hipdnn_frontend::bfloat16.
using nv_bfloat16 = hipdnn_frontend::bfloat16;

// Node attribute types with an exact hipDNN counterpart, aliased 1:1.
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
using hipdnn_frontend::graph::RoPE_attributes;

// NOLINTBEGIN(readability-identifier-naming): mirrors cuDNN's snake_case public spelling.
/// @brief cuDNN Softmax_attributes over hipDNN's SoftmaxAttributes (stats_use_log2 and the
/// optional Stats output carry over). cuDNN's SINK input has no hipDNN equivalent: a set
/// sink makes Graph::softmax record GRAPH_NOT_SUPPORTED. cuDNN's Max / Sum_exp outputs are
/// reachable only through its internal INode API and have no shim spelling.
class Softmax_attributes : public hipdnn_frontend::graph::SoftmaxAttributes
{
public:
    Softmax_attributes& set_sink(std::shared_ptr<Tensor_attributes> value)
    {
        _sink = std::move(value);
        return *this;
    }

    const std::shared_ptr<Tensor_attributes>& get_sink() const
    {
        return _sink;
    }

    Softmax_attributes& set_stats_use_log2(bool value)
    {
        hipdnn_frontend::graph::SoftmaxAttributes::set_stats_use_log2(value);
        return *this;
    }

private:
    std::shared_ptr<Tensor_attributes> _sink;
};
// NOLINTEND(readability-identifier-naming)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
