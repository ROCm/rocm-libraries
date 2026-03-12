// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

namespace hipdnn_frontend::graph
{

/// Identifies the concrete type of a graph node without RTTI.
/// Each node subclass carries its NodeType as a compile-time template
/// parameter through BaseNode, enabling type-based dispatch (e.g.,
/// engine override selection) via switch on getNodeType().
enum class NodeType
{
    Unknown,
    ConvolutionFprop,
    ConvolutionDgrad,
    ConvolutionWgrad,
    Batchnorm,
    BatchnormInference,
    BatchnormBackward,
    BatchnormInferenceVarianceExt,
    Pointwise,
    Matmul,
    LayerNorm,
    RMSNorm,
    SdpaFprop,
    BlockScaleQuantize,
    BlockScaleDequantize
};

} // namespace hipdnn_frontend::graph
