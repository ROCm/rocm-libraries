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
};

// Selects when low-precision accumulator types are rounded.
enum class AccumulationRounding {
    TypeDefault,         // Stepwise rounding for F16/BF16 accumulators; full precision otherwise.
    FullPrecision,       // Keeps the host register type through the complete dot product.
    AfterProductAndSum,  // Quantizes every product and accumulated sum.
};

// Numerical and execution policy intrinsic to one matrix multiplication.
struct MatmulOptions {
    explicit MatmulOptions(ScalarType accumulator = ScalarType::Float32)
        : accumulatorType(accumulator) {}

    ScalarType accumulatorType;  // Dot-product arithmetic type.
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

    OutputSelection outputSelection = OutputSelection::all();
};

// NumPy-style matrix multiplication. The owning form allocates output; the
// `Into` form writes selected coordinates of caller-owned output.
void matmulInto(Tensor a, Tensor b, Tensor output, const MatmulOptions& options = MatmulOptions{},
                GemmBackend backend = GemmBackend::Automatic);
Tensor matmul(Tensor a, Tensor b, ScalarType outputType,
              const MatmulOptions& options = MatmulOptions{},
              std::optional<Layout> outputLayout = std::nullopt,
              GemmBackend backend = GemmBackend::Automatic);
}  // namespace roc::host_numerics
