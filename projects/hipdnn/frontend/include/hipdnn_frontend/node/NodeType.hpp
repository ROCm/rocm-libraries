// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <ostream>

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

/// @brief Get a human-readable string for a NodeType value
// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const NodeType& type)
{
    switch(type)
    {
    case NodeType::UNKNOWN:
        return "Unknown";
    case NodeType::CONVOLUTION_FPROP:
        return "ConvFprop";
    case NodeType::CONVOLUTION_DGRAD:
        return "ConvDgrad";
    case NodeType::CONVOLUTION_WGRAD:
        return "ConvWgrad";
    case NodeType::BATCHNORM:
        return "Batchnorm";
    case NodeType::BATCHNORM_INFERENCE:
        return "BatchnormInference";
    case NodeType::BATCHNORM_BACKWARD:
        return "BatchnormBackward";
    case NodeType::BATCHNORM_INFERENCE_VARIANCE_EXT:
        return "BatchnormInferenceVarianceExt";
    case NodeType::POINTWISE:
        return "Pointwise";
    case NodeType::MATMUL:
        return "Matmul";
    case NodeType::LAYER_NORM:
        return "LayerNorm";
    case NodeType::RMS_NORM:
        return "RmsNorm";
    case NodeType::SDPA_FWD:
        return "SdpaFwd";
    case NodeType::SDPA_BWD:
        return "SdpaBwd";
    case NodeType::BLOCK_SCALE_QUANTIZE:
        return "BlockScaleQuantize";
    case NodeType::BLOCK_SCALE_DEQUANTIZE:
        return "BlockScaleDequantize";
    case NodeType::CUSTOM_OP:
        return "CustomOp";
    case NodeType::REDUCTION:
        return "Reduction";
    default:
        return "Unknown";
    }
}

/// @brief Stream insertion operator for NodeType
inline std::ostream& operator<<(std::ostream& os, const NodeType& type)
{
    return os << to_string(type);
}

} // namespace hipdnn_frontend::graph
