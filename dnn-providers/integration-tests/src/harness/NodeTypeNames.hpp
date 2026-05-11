// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/node/NodeType.hpp>
#include <ostream>

namespace hipdnn_integration_tests
{

// Get a human-readable string for a NodeType value.
// Used by GraphDescription and the support matrix to label graph operations.
// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const hipdnn_frontend::graph::NodeType& type)
{
    using hipdnn_frontend::graph::NodeType;

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

inline std::ostream& operator<<(std::ostream& os, const hipdnn_frontend::graph::NodeType& type)
{
    return os << to_string(type);
}

} // namespace hipdnn_integration_tests
