// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_numerics/operation_types.hpp>
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

    Tensor values;                                     // A is [M,K]; B is [K,N].
    std::optional<ScalarType> computeType;             // Optional per-element input quantization.
    std::vector<VectorBinding> preQuantizationScales;  // Ordered factors before quantization.
    std::optional<BlockScaleBinding> blockScale;       // Independent per-reduction-block factor.
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

    Scalar alpha;                             // Multiplies the accumulated A*B term.
    Scalar beta;                              // Multiplies C.
    Scalar scaleC;                            // Multiplies C before beta.
    std::optional<VectorBinding> bias;        // Added after alpha*A*B + beta*scaleC*C.
    std::optional<VectorBinding> scaleAlpha;  // Row/column factor applied to alpha.
    std::optional<Tensor> scaleA;  // Rank-one row factor, or length-one broadcast, on alpha.
    std::optional<Tensor> scaleB;  // Rank-one column factor, or length-one broadcast, on alpha.
    Scalar outputScale;            // Applied after activation.
    OutputConversion outputConversion = OutputConversion::Default;  // Final D encoding.
    Activation activation = Activation::None;                       // Applied before outputScale.
    Scalar activationParameter0;  // First activation-specific scalar.
    Scalar activationParameter1;  // Second activation-specific scalar.
};

// Reusable numerical GEMM descriptor. It contains A, B, C, arithmetic policy,
// and D's scalar type, but no destination tensor, output selection, or backend
// policy.
struct GemmProblem {
    GemmProblem(GemmOperand aOperand, GemmOperand bOperand, Tensor cTensor, ScalarType output,
                ScalarType accumulator)
        : a(std::move(aOperand)),
          b(std::move(bOperand)),
          c(std::move(cTensor)),
          outputType(output),
          accumulatorType(accumulator),
          epilogue(accumulator) {}

    GemmOperand a;               // Rank-two [M,K] operand.
    GemmOperand b;               // Rank-two [K,N] operand.
    Tensor c;                    // Rank-two [M,N] addend read when beta is nonzero.
    ScalarType outputType;       // Required scalar type of a request's D tensor.
    ScalarType accumulatorType;  // Dot-product and epilogue arithmetic type.
    AccumulationRounding accumulationRounding = AccumulationRounding::TypeDefault;
    MathMode mathMode = MathMode::Default;  // Operand transform after compute-type quantization.
    GemmEpilogue epilogue;
};

// One GEMM invocation. It extends GemmProblem with the caller-owned D
// destination and the coordinates allowed to change.
struct GemmRequest : GemmProblem {
    GemmRequest(GemmOperand aOperand, GemmOperand bOperand, Tensor cTensor, Tensor dTensor,
                ScalarType accumulator)
        : GemmProblem(std::move(aOperand), std::move(bOperand), std::move(cTensor), dTensor.type(),
                      accumulator),
          d(std::move(dTensor)) {}

    GemmRequest(GemmProblem problem, Tensor dTensor,
                OutputSelection selection = OutputSelection::all())
        : GemmProblem(std::move(problem)),
          d(std::move(dTensor)),
          outputSelection(std::move(selection)) {}

    // Rank-two [M,N] destination with distinct logical element offsets. D must
    // not overlap operands or scales. Exact same-type, same-layout C=D is allowed.
    Tensor d;
    OutputSelection outputSelection = OutputSelection::all();  // Logical D coordinates to write.
};

// Result of validating one request against one execution policy.
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

// Reports the strategy used and its completed output work.
struct GemmRunInfo {
    // Concrete strategy for one request. A product-level aggregate may report
    // Mixed when its independently dispatched requests used different strategies.
    GemmBackend backendUsed = GemmBackend::Pointwise;
    std::optional<std::string> fallbackReason;  // Rejection that caused Automatic fallback.
    size_t outputElementsWritten = 0;           // Logical D coordinates overwritten.
    // Logical output coordinates covered by the strategy: selected coordinates
    // for Pointwise, touched output blocks for Blocked, and full D for Blas.
    size_t outputElementsCovered = 0;
};

// Output allocation and logical-write policy for an owning GEMM call.
struct GemmOutputOptions {
    std::optional<Layout> layout;  // Null selects contiguous [M,N] storage.
    OutputSelection selection = OutputSelection::all();
};

// Owning GEMM output and completed-work metadata.
struct GemmResult {
    Tensor output;
    GemmRunInfo runInfo;
};

// Validates the request and selected built-in strategy without mutating any
// tensor. The optional BLAS component provides queryGemmSupportWithBlasBackend.
GemmSupportInfo queryGemmSupport(const GemmRequest& request,
                                 GemmBackend backend = GemmBackend::Automatic);

// Executes the request, mutates selected D coordinates, and returns completed-work metadata.
// The optional BLAS component provides referenceGemmWithBlasBackend.
GemmRunInfo referenceGemm(const GemmRequest& request, GemmBackend backend = GemmBackend::Automatic);

// Allocates and zero-initializes D, then delegates to the caller-owned request
// path. Unselected logical coordinates remain zero.
GemmResult referenceGemm(const GemmProblem& problem, const GemmOutputOptions& output = {},
                         GemmBackend backend = GemmBackend::Automatic);
}  // namespace roc::host_numerics
