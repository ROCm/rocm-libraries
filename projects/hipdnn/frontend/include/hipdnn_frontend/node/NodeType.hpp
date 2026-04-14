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
    UNKNOWN = 0,
    CONVOLUTION_FPROP = 1,
    CONVOLUTION_DGRAD = 2,
    CONVOLUTION_WGRAD = 3,
    BATCHNORM = 4,
    BATCHNORM_INFERENCE = 5,
    BATCHNORM_BACKWARD = 6,
    BATCHNORM_INFERENCE_VARIANCE_EXT = 7,
    POINTWISE = 8,
    MATMUL = 9,
    LAYER_NORM = 10,
    RMS_NORM = 11,
    SDPA_FWD = 12,
    SDPA_BWD = 13,
    BLOCK_SCALE_QUANTIZE = 14,
    BLOCK_SCALE_DEQUANTIZE = 15,
    CUSTOM_OP = 16,
    REDUCTION = 17
};

} // namespace hipdnn_frontend::graph
