// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <roc/hostnumerics/operation_types.hpp>
#include <roc/hostnumerics/permutation.hpp>
#include <roc/hostnumerics/tensor.hpp>
#include <set>
#include <vector>

namespace roc::hostnumerics {
// Selects the host implementation strategy. Blocked adapts between direct
// selected-output work and operand-block reuse, but writes only the selected D
// coordinates. Blas delegates to a supplied BLAS implementation.
enum class GemmBackend {
    Automatic,  // Selects between available BLAS and built-in Blocked execution.
    Blocked,    // Uses built-in density-aware blocked or selected-output execution.
    Blas,       // Uses the optional BLAS component's transforming implementation.
};

// Selects when low-precision accumulator types are rounded.
enum class AccumulationRounding {
    TypeDefault,         // Stepwise rounding for F16/BF16 accumulators; full precision otherwise.
    FullPrecision,       // Keeps the host register type through the complete dot product.
    AfterProductAndSum,  // Quantizes every product and accumulated sum.
};

// Real scales shared by `blockSize` consecutive K elements of one operand.
// `scales` has shape [M for A or N for B, at least ceil(K / blockSize)];
// `blockSize` must be nonzero.
struct MatmulBlockScale {
    Tensor scales;
    size_t blockSize = 0;
};

// Numerical and execution policy intrinsic to one matrix multiplication.
struct MatmulOptions {
    explicit MatmulOptions(ScalarType accumulator) : accumulatorType(accumulator) {}

    ScalarType accumulatorType;  // Dot-product arithmetic type.
    AccumulationRounding accumulationRounding = AccumulationRounding::TypeDefault;
    MathMode mathMode = MathMode::Default;  // Operand transform after compute-type quantization.

    std::optional<ScalarType> computeTypeA;
    std::optional<ScalarType> computeTypeB;
    std::vector<Tensor> preQuantizationScalesA;  // Ordered factors broadcast to A.
    std::vector<Tensor> preQuantizationScalesB;  // Ordered factors broadcast to B.
    std::optional<MatmulBlockScale> blockScaleA;
    std::optional<MatmulBlockScale> blockScaleB;
    bool conjugateA = false;
    bool conjugateB = false;
};

// NumPy-style matrix multiplication. The owning form allocates output; the
// `Into` form writes selected coordinates of caller-owned output.
void matmulInto(const Tensor& a, const Tensor& b, Tensor output, const MatmulOptions& options,
                const IndexRange& range, const Permutation& order, GemmBackend backend);
void matmulInto(const Tensor& a, const Tensor& b, Tensor output, const MatmulOptions& options,
                const std::set<size_t>& indices, const Permutation& order, GemmBackend backend);
Tensor matmul(const Tensor& a, const Tensor& b, ScalarType outputType, const MatmulOptions& options,
              std::optional<Layout> outputLayout, GemmBackend backend);

template <typename Indices>
void matmulInto(const Tensor& a, const Tensor& b, Tensor output, const MatmulOptions& options,
                const Indices& indices, LastDimensionFastest, GemmBackend backend) {
    matmulInto(a, b, output, options, indices, Permutation::identity(output.shape().rank()),
               backend);
}

}  // namespace roc::hostnumerics
