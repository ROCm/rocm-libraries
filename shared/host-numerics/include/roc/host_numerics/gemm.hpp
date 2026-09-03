// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <roc/host_numerics/operation_types.hpp>
#include <roc/host_numerics/tensor.hpp>
#include <vector>

namespace roc::host_numerics {
// Selects the host implementation strategy. Blocked reuses operand blocks and
// may accumulate unselected coordinates in every touched output block, but
// writes only the selected D coordinates. Blas delegates to a supplied BLAS
// implementation.
enum class GemmBackend {
    Automatic,  // Selects between available BLAS and built-in Blocked execution.
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

// Operand transforms, arithmetic, epilogue, and output-selection policy for one GEMM.
struct GemmOptions {
    explicit GemmOptions(ScalarType accumulator = ScalarType::Float32)
        : accumulatorType(accumulator),
          alpha(Scalar::one(accumulator)),
          beta(Scalar::zero(accumulator)),
          scaleC(Scalar::one(accumulator)),
          outputScale(Scalar::one(accumulator)),
          activationParameter0(Scalar::zero(accumulator)),
          activationParameter1(Scalar::zero(accumulator)) {}

    ScalarType accumulatorType;  // Dot-product and epilogue arithmetic type.
    AccumulationRounding accumulationRounding = AccumulationRounding::TypeDefault;
    MathMode mathMode = MathMode::Default;  // Operand transform after compute-type quantization.

    std::optional<ScalarType> computeTypeA;
    std::optional<ScalarType> computeTypeB;
    std::vector<Tensor> preQuantizationScalesA;  // Ordered factors broadcast to A.
    std::vector<Tensor> preQuantizationScalesB;  // Ordered factors broadcast to B.
    std::optional<Tensor> blockScaleA;           // [M, ceil(K / blockSizeA)].
    std::optional<Tensor> blockScaleB;           // [N, ceil(K / blockSizeB)].
    size_t blockSizeA = 0;
    size_t blockSizeB = 0;
    bool conjugateA = false;
    bool conjugateB = false;

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

    OutputSelection outputSelection = OutputSelection::all();  // Logical D coordinates to write.
};

// Writes selected coordinates into caller-owned D and reports the concrete
// backend used. Exact same-layout C/D aliasing is supported.
GemmBackend referenceGemmInto(Tensor a, Tensor b, Tensor c, Tensor d,
                              const GemmOptions& options = GemmOptions{},
                              GemmBackend backend = GemmBackend::Automatic);

// Allocates and zero-initializes D, then executes the owning GEMM. Unselected
// logical coordinates remain zero.
Tensor referenceGemm(Tensor a, Tensor b, Tensor c, ScalarType outputType,
                     const GemmOptions& options = GemmOptions{},
                     std::optional<Layout> outputLayout = std::nullopt,
                     GemmBackend backend = GemmBackend::Automatic);
}  // namespace roc::host_numerics
