// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_numerics/gemm.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gemm_invocation.hpp"
#include "reference_common.hpp"
#include "threading.hpp"

namespace roc::host_numerics {
namespace detail {
inline bool isRuntimeGemmAccumulator(ScalarType type) {
    switch (type) {
        case ScalarType::Float32:
        case ScalarType::Float64:
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Int32:
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
            return true;
        default:
            return false;
    }
}

inline void validateRuntimeGemm(const GemmInvocation& problem) {
    requireRank(problem.a.shape(), 2, "Reference GEMM", "A");
    requireRank(problem.b.shape(), 2, "Reference GEMM", "B");
    requireRank(problem.d.shape(), 2, "Reference GEMM", "output");

    const size_t m = problem.a.shape()[0];
    const size_t k = problem.a.shape()[1];
    const size_t n = problem.b.shape()[1];
    if (problem.b.shape()[0] != k)
        throw std::invalid_argument("Reference GEMM K dimension mismatch.");
    if (problem.d.shape() != Shape{m, n})
        throw std::invalid_argument("Reference GEMM output shape mismatch.");
    if (!isRuntimeGemmAccumulator(problem.accumulatorType))
        throw std::invalid_argument(
            "Runtime reference GEMM currently supports F16, BF16, F32, F64, I32, C64, and "
            "C128 accumulators.");
    if (problem.accumulationRounding == AccumulationRounding::AfterProductAndSum &&
        problem.accumulatorType != ScalarType::Float16 &&
        problem.accumulatorType != ScalarType::BFloat16)
        throw std::invalid_argument(
            "Product-and-sum accumulator rounding currently requires an F16 or BF16 "
            "accumulator type.");

    const bool complexAccumulator = isComplexScalarType(problem.accumulatorType);
    auto validateOperandType = [&](ScalarType type, const char* name) {
        if (type == ScalarType::Count || type == ScalarType::Boolean || isScaleScalarType(type))
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " has an unsupported scalar type.");
        if (!complexAccumulator && isComplexScalarType(type))
            throw std::invalid_argument(
                std::string("Reference GEMM real accumulator cannot consume complex ") + name +
                ".");
    };
    validateOperandType(problem.a.type(), "A");
    validateOperandType(problem.b.type(), "B");
    validateOperandType(problem.d.type(), "output");
    if (complexAccumulator != isComplexScalarType(problem.d.type()))
        throw std::invalid_argument("Reference GEMM complex accumulator/output mismatch.");

    auto validateComputeType = [&](const Tensor& operand,
                                   const std::optional<ScalarType>& computeType, const char* name) {
        if (!computeType) return;
        validateOperandType(*computeType, name);
        if (isComplexScalarType(operand.type()) && !isComplexScalarType(*computeType))
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " compute-input type has incompatible complexity.");
    };
    validateComputeType(problem.a, problem.computeTypeA, "A");
    validateComputeType(problem.b, problem.computeTypeB, "B");
    auto validatePreQuantizationScales = [&](const Tensor& operand,
                                             const std::vector<Tensor>& scales, const char* name) {
        for (const Tensor& scale : scales) {
            try {
                (void)scale.broadcastTo(operand.shape());
            } catch (const std::invalid_argument&) {
                throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                            " is not broadcast-compatible with its operand.");
            }
            if (!complexAccumulator && isComplexScalarType(scale.type()))
                throw std::invalid_argument(
                    std::string("Reference GEMM real accumulator cannot consume complex ") + name +
                    ".");
        }
    };
    validatePreQuantizationScales(problem.a, problem.preQuantizationScalesA,
                                  "A pre-quantization scale");
    validatePreQuantizationScales(problem.b, problem.preQuantizationScalesB,
                                  "B pre-quantization scale");

    if (problem.mathMode == MathMode::XFloat32 && problem.accumulatorType != ScalarType::Float32)
        throw std::invalid_argument("XFloat32 math mode requires a Float32 accumulator.");

    auto validateBlockScale = [&](const std::optional<Tensor>& scale, size_t blockSize,
                                  size_t freeExtent, const char* name) {
        if (!scale) {
            if (blockSize != 0)
                throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                            " block size requires a scale tensor.");
            return;
        }
        if (blockSize == 0)
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " block size must be nonzero.");
        requireRank(scale->shape(), 2, "Reference GEMM", name);
        const size_t blockCount = k / blockSize + (k % blockSize != 0 ? 1 : 0);
        if (scale->shape()[0] != freeExtent || scale->shape()[1] < blockCount)
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " block-scale shape mismatch.");
        if (isComplexScalarType(scale->type()))
            throw std::invalid_argument(std::string("Reference GEMM ") + name +
                                        " block scales must be real.");
    };
    validateBlockScale(problem.blockScaleA, problem.blockSizeA, m, "A");
    validateBlockScale(problem.blockScaleB, problem.blockSizeB, n, "B");
    if (problem.blockScaleA) {
        if (complexAccumulator)
            throw std::invalid_argument("Complex reference GEMM does not support block scaling.");
    }
    if (problem.blockScaleB) {
        if (complexAccumulator)
            throw std::invalid_argument("Complex reference GEMM does not support block scaling.");
    }
    if (!hasProvablyDistinctElementOffsets(problem.d.layout()))
        throw std::invalid_argument(
            "Reference GEMM requires distinct logical destination elements.");

    if (storageOverlaps(problem.d, problem.a) || storageOverlaps(problem.d, problem.b))
        throw std::invalid_argument("Reference GEMM destination must not overlap A or B.");

    for (const Tensor& scale : problem.preQuantizationScalesA)
        if (storageOverlaps(problem.d, scale))
            throw std::invalid_argument(
                "Reference GEMM destination must not overlap an A pre-quantization scale.");
    for (const Tensor& scale : problem.preQuantizationScalesB)
        if (storageOverlaps(problem.d, scale))
            throw std::invalid_argument(
                "Reference GEMM destination must not overlap a B pre-quantization scale.");
    if (problem.blockScaleA && storageOverlaps(problem.d, *problem.blockScaleA))
        throw std::invalid_argument(
            "Reference GEMM destination must not overlap the A block scale.");
    if (problem.blockScaleB && storageOverlaps(problem.d, *problem.blockScaleB))
        throw std::invalid_argument(
            "Reference GEMM destination must not overlap the B block scale.");
    (void)problem.outputSelection.selectedCount(problem.d.shape().elementCount());
}

inline bool canParallelizeGemmOutput(const GemmInvocation& problem) {
    return hasProvablyIndependentElements(problem.d);
}

template <typename Accumulator>
RuntimeQuantizer<Accumulator> gemmAccumulatorQuantizer(const MatmulOptions& problem) {
    const bool typeRoundsAfterEachStep = problem.accumulatorType == ScalarType::Float16 ||
                                         problem.accumulatorType == ScalarType::BFloat16;
    const bool roundAfterEachStep =
        problem.accumulationRounding == AccumulationRounding::AfterProductAndSum ||
        (problem.accumulationRounding == AccumulationRounding::TypeDefault &&
         typeRoundsAfterEachStep);
    return RuntimeQuantizer<Accumulator>(
        roundAfterEachStep ? std::optional<ScalarType>(problem.accumulatorType) : std::nullopt);
}

template <typename Accumulator>
class RuntimeGemmArithmetic {
   public:
    explicit RuntimeGemmArithmetic(
        const MatmulOptions&,
        RuntimeQuantizer<Accumulator> quantizeAccumulator = RuntimeQuantizer<Accumulator>())
        : m_quantizeAccumulator(std::move(quantizeAccumulator)) {}

    Accumulator multiply(Accumulator left, Accumulator right) const {
        return m_quantizeAccumulator(wrappingMultiply(left, right));
    }

    Accumulator add(Accumulator left, Accumulator right) const {
        return m_quantizeAccumulator(wrappingAdd(left, right));
    }

   private:
    RuntimeQuantizer<Accumulator> m_quantizeAccumulator;
};

}  // namespace detail
}  // namespace roc::host_numerics
