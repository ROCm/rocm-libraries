// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/index_selection.hpp>
#include <variant>

namespace roc::hostnumerics {
// Operation-specific option types describe numerical policy independently of
// tensor ownership. Owning entry points return newly allocated tensors;
// corresponding ...Into entry points write caller-owned destinations.

// Stable activation identifiers used by product adapters and compatibility
// bindings. Tensor-centric C++ callers use the typed ActivationFunction below.
enum class Activation {
    None,
    Absolute,
    ClippedRelu,
    Relu,
    Gelu,
    GeluDerivative,
    GeluScaling,
    LeakyRelu,
    ReluDerivative,
    Sigmoid,
    Tanh,
    Silu,
    Swish,
    Clamp,
};

struct IdentityActivation {};
struct AbsoluteActivation {};
struct ReluActivation {};
struct GeluActivation {};
struct GeluDerivativeActivation {};
struct ReluDerivativeActivation {};
struct SigmoidActivation {};
struct SiluActivation {};

struct ClippedReluActivation {
    double lower = 0.0;
    double upper = 0.0;
};

struct GeluScalingActivation {
    double scale = 1.0;
};

struct LeakyReluActivation {
    double negativeSlope = 0.0;
};

struct TanhActivation {
    double inputScale = 1.0;
    double outputScale = 1.0;
};

struct SwishActivation {
    double beta = 1.0;
};

struct ClampActivation {
    double minimum = 0.0;
    double maximum = 0.0;
};

using ActivationFunction =
    std::variant<IdentityActivation, AbsoluteActivation, ClippedReluActivation, ReluActivation,
                 GeluActivation, GeluDerivativeActivation, GeluScalingActivation,
                 LeakyReluActivation, ReluDerivativeActivation, SigmoidActivation, TanhActivation,
                 SiluActivation, SwishActivation, ClampActivation>;

// Selects operand arithmetic applied after input quantization and before
// multiplication.
enum class MathMode {
    Default,
    XFloat32,  // Truncate Float32 operands to the xfloat32 representation.
};

// Selects the final conversion performed before writing an output tensor.
enum class OutputConversion {
    Default,
    SaturatingInt8,  // Round and clamp to [-128, 127].
};

}  // namespace roc::hostnumerics
