// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

namespace hipdnn_frontend::graph
{

/// Identifies the concrete type of a graph node without RTTI.
/// Each node subclass carries its NodeType as a compile-time template
/// parameter through BaseNode, enabling type-based dispatch when
/// visiting the graph's node tree via INode::visit().
enum class NodeType
{
    UNKNOWN,
    CONVOLUTION_FPROP,
    CONVOLUTION_DGRAD,
    CONVOLUTION_WGRAD,
    BATCHNORM,
    BATCHNORM_INFERENCE,
    BATCHNORM_BACKWARD,
    BATCHNORM_INFERENCE_VARIANCE_EXT,
    POINTWISE,
    MATMUL,
    LAYER_NORM,
    RMS_NORM,
    SDPA_FPROP,
    BLOCK_SCALE_QUANTIZE,
    BLOCK_SCALE_DEQUANTIZE
};

} // namespace hipdnn_frontend::graph
