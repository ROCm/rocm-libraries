// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <roc/host_numerics/operation_types.hpp>
#include <roc/host_numerics/tensor.hpp>
#include <string>
#include <utility>
#include <vector>

namespace roc::host_numerics {
// Selects the host implementation strategy. Pointwise computes exactly the
// selected D coordinates. Blocked reuses operand blocks and may accumulate
// unselected coordinates in every touched output block, but writes only the
// selected D coordinates. Blas delegates to a supplied BLAS implementation.
enum class GemmBackend {
    Automatic,  // Selects among available BLAS, built-in Blocked, and Pointwise execution.
    Pointwise,  // Computes and writes exactly the selected D coordinates.
    Blocked,    // Accumulates complete touched output blocks; writes selected D coordinates.
    Blas,       // Uses the optional BLAS component's transforming implementation.
    Mixed,      // Reporting-only value for an aggregate that used multiple concrete backends.
};

// Selects when low-precision accumulator types are rounded.
enum class AccumulationRounding {
    TypeDefault,         // Stepwise rounding for F16/BF16 accumulators; full precision otherwise.
    FullPrecision,       // Keeps the host register type through the complete dot product.
    AfterProductAndSum,  // Quantizes every product and accumulated sum.
};

// Associates a rank-two scale tensor with its reduction-dimension block width.
struct BlockScaleBinding {
    BlockScaleBinding(Tensor tensor, size_t reductionBlockSize)
        : values(std::move(tensor)), blockSize(reductionBlockSize) {}

    Tensor values;     // [free dimension, reduction block].
    size_t blockSize;  // Number of consecutive K elements sharing one scale.
};

// Describes one normalized rank-two GEMM operand.
struct GemmOperand {
    explicit GemmOperand(Tensor tensor) : values(std::move(tensor)) {}

    Tensor values;                                // A is [M,K]; B is [K,N].
    std::optional<ScalarType> computeType;        // Optional per-element input quantization.
    std::vector<Tensor> preQuantizationScales;    // Ordered broadcast factors before quantization.
    std::optional<BlockScaleBinding> blockScale;  // Independent per-reduction-block factor.
    bool conjugate = false;  // Conjugates values after loading and before scaling.
};

// Describes alpha/beta/C-scale combination and the fused D finalization program.
struct GemmEpilogue {
    explicit GemmEpilogue(ScalarType coefficientType)
        : alpha(Scalar::one(coefficientType)),
          beta(Scalar::zero(coefficientType)),
          scaleC(Scalar::one(coefficientType)),
          outputScale(Scalar::one(coefficientType)),
          activationParameter0(Scalar::zero(coefficientType)),
          activationParameter1(Scalar::zero(coefficientType)) {}

    Scalar alpha;                      // Multiplies the accumulated A*B term.
    Scalar beta;                       // Multiplies C.
    Scalar scaleC;                     // Multiplies C before beta.
    std::optional<Tensor> bias;        // Broadcast addend after alpha*A*B + beta*scaleC*C.
    std::optional<Tensor> scaleAlpha;  // Broadcast factor applied to alpha.
    std::optional<Tensor> scaleA;      // Broadcast factor applied to alpha.
    std::optional<Tensor> scaleB;      // Broadcast factor applied to alpha.
    Scalar outputScale;                // Applied after activation.
    OutputConversion outputConversion = OutputConversion::Default;  // Final D encoding.
    Activation activation = Activation::None;                       // Applied before outputScale.
    Scalar activationParameter0;  // First activation-specific scalar.
    Scalar activationParameter1;  // Second activation-specific scalar.
};

// Arithmetic, epilogue, and output-selection policy for one GEMM.
struct GemmOptions {
    explicit GemmOptions(ScalarType accumulator = ScalarType::Float32)
        : accumulatorType(accumulator), epilogue(accumulator) {}

    ScalarType accumulatorType;  // Dot-product and epilogue arithmetic type.
    AccumulationRounding accumulationRounding = AccumulationRounding::TypeDefault;
    MathMode mathMode = MathMode::Default;  // Operand transform after compute-type quantization.
    GemmEpilogue epilogue;
    OutputSelection outputSelection = OutputSelection::all();  // Logical D coordinates to write.
};

// Report from validating one GEMM invocation against one execution policy.
struct GemmSupportInfo {
    bool supported = false;  // True only when validation and backend restrictions pass.
    std::string reason;      // Empty when supported; rejection text otherwise.
    // A supported optional backend may still be more expensive than Pointwise
    // for this request. Automatic execution consults this component-owned hint.
    bool preferredForAutomaticExecution = true;

    explicit operator bool() const {
        return supported;
    }
};

// Validates the complete invocation and selected built-in strategy without
// mutating any tensor. Backend support can depend on D's layout and aliases.
GemmSupportInfo queryGemmSupport(const GemmOperand& a, const GemmOperand& b, const Tensor& c,
                                 const Tensor& d, const GemmOptions& options = GemmOptions{},
                                 GemmBackend backend = GemmBackend::Automatic);

// Writes selected coordinates into caller-owned D and reports the concrete
// backend used. Exact same-layout C/D aliasing is supported.
GemmBackend referenceGemmInto(GemmOperand a, GemmOperand b, Tensor c, Tensor d,
                              const GemmOptions& options = GemmOptions{},
                              GemmBackend backend = GemmBackend::Automatic);

// Allocates and zero-initializes D, then executes the owning GEMM. Unselected
// logical coordinates remain zero.
Tensor referenceGemm(GemmOperand a, GemmOperand b, Tensor c, ScalarType outputType,
                     const GemmOptions& options = GemmOptions{},
                     std::optional<Layout> outputLayout = std::nullopt,
                     GemmBackend backend = GemmBackend::Automatic);
}  // namespace roc::host_numerics
