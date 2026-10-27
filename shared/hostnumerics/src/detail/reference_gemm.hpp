// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <optional>
#include <roc/hostnumerics/detail/shape_access.hpp>
#include <roc/hostnumerics/gemm.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gemm_invocation.hpp"
#include "reference_common.hpp"
#include "threading.hpp"

namespace roc::hostnumerics {
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

inline bool requiresInputQuantization(ScalarType storageType,
                                      const std::optional<ScalarType>& computeType,
                                      ScalarType arithmeticType) {
    // Loading has already applied storageType, and arithmetic takes place in arithmeticType.
    // Only a distinct intermediate compute type can introduce another quantization boundary.
    return computeType && *computeType != storageType && *computeType != arithmeticType;
}

inline GemmSupportInfo checkRuntimeGemm(const GemmInvocation& problem) {
    if (problem.a.shape().rank() != 2) return {.reason = "Reference GEMM A must have rank 2."};
    if (problem.b.shape().rank() != 2) return {.reason = "Reference GEMM B must have rank 2."};
    if (problem.d.shape().rank() != 2) return {.reason = "Reference GEMM output must have rank 2."};

    const size_t m = problem.a.shape()[0];
    const size_t k = problem.a.shape()[1];
    const size_t n = problem.b.shape()[1];
    if (problem.b.shape()[0] != k) return {.reason = "Reference GEMM K dimension mismatch."};
    if (problem.d.shape()[0] != m || problem.d.shape()[1] != n)
        return {.reason = "Reference GEMM output shape mismatch."};
    if (!isRuntimeGemmAccumulator(problem.accumulatorType))
        return {.reason =
                    "Runtime reference GEMM currently supports F16, BF16, F32, F64, I32, C64, and "
                    "C128 accumulators."};
    if (problem.accumulationRounding == AccumulationRounding::AfterProductAndSum &&
        problem.accumulatorType != ScalarType::Float16 &&
        problem.accumulatorType != ScalarType::BFloat16)
        return {.reason =
                    "Product-and-sum accumulator rounding currently requires an F16 or BF16 "
                    "accumulator type."};

    const bool complexAccumulator = isComplexScalarType(problem.accumulatorType);
    auto validateOperandType = [&](ScalarType type, const char* name) -> GemmSupportInfo {
        if (!isConcreteScalarType(type) || type == ScalarType::Boolean ||
            type == ScalarType::E4M3 || type == ScalarType::E5M3 || type == ScalarType::E8M0 ||
            type == ScalarType::E8M0Zero)
            return {.reason =
                        std::string("Reference GEMM ") + name + " has an unsupported scalar type."};
        if (!complexAccumulator && isComplexScalarType(type))
            return {.reason =
                        std::string("Reference GEMM real accumulator cannot consume complex ") +
                        name + "."};
        return {.supported = true};
    };
    if (auto result = validateOperandType(problem.a.type(), "A"); !result) return result;
    if (auto result = validateOperandType(problem.b.type(), "B"); !result) return result;
    if (auto result = validateOperandType(problem.d.type(), "output"); !result) return result;
    if (complexAccumulator != isComplexScalarType(problem.d.type()))
        return {.reason = "Reference GEMM complex accumulator/output mismatch."};

    auto validateComputeType = [&](const Tensor& operand,
                                   const std::optional<ScalarType>& computeType,
                                   const char* name) -> GemmSupportInfo {
        if (!computeType) return {.supported = true};
        if (auto result = validateOperandType(*computeType, name); !result) return result;
        if (isComplexScalarType(operand.type()) && !isComplexScalarType(*computeType))
            return {.reason = std::string("Reference GEMM ") + name +
                              " compute-input type has incompatible complexity."};
        return {.supported = true};
    };
    if (auto result = validateComputeType(problem.a, problem.computeTypeA, "A"); !result)
        return result;
    if (auto result = validateComputeType(problem.b, problem.computeTypeB, "B"); !result)
        return result;
    auto validatePreQuantizationScales = [&](const Tensor& operand,
                                             const std::vector<Tensor>& scales,
                                             const char* name) -> GemmSupportInfo {
        for (const Tensor& scale : scales) {
            if (!canBroadcastTo(scale.shape().dimensions(), operand.shape().dimensions())) {
                return {.reason = std::string("Reference GEMM ") + name +
                                  " is not broadcast-compatible with its operand."};
            }
            if (!complexAccumulator && isComplexScalarType(scale.type()))
                return {.reason =
                            std::string("Reference GEMM real accumulator cannot consume complex ") +
                            name + "."};
        }
        return {.supported = true};
    };
    if (auto result = validatePreQuantizationScales(problem.a, problem.preQuantizationScalesA,
                                                    "A pre-quantization scale");
        !result)
        return result;
    if (auto result = validatePreQuantizationScales(problem.b, problem.preQuantizationScalesB,
                                                    "B pre-quantization scale");
        !result)
        return result;

    if (problem.mathMode == MathMode::XFloat32 && problem.accumulatorType != ScalarType::Float32)
        return {.reason = "XFloat32 math mode requires a Float32 accumulator."};

    auto validateBlockScale = [&](const std::optional<Tensor>& scale, size_t blockSize,
                                  size_t freeExtent, const char* name) -> GemmSupportInfo {
        if (!scale) {
            if (blockSize != 0)
                return {.reason = std::string("Reference GEMM ") + name +
                                  " block size requires a scale tensor."};
            return {.supported = true};
        }
        if (blockSize == 0)
            return {.reason =
                        std::string("Reference GEMM ") + name + " block size must be nonzero."};
        if (scale->shape().rank() != 2)
            return {.reason = std::string("Reference GEMM ") + name + " must have rank 2."};
        const size_t blockCount = k / blockSize + (k % blockSize != 0 ? 1 : 0);
        if (scale->shape()[0] != freeExtent || scale->shape()[1] < blockCount)
            return {.reason =
                        std::string("Reference GEMM ") + name + " block-scale shape mismatch."};
        if (isComplexScalarType(scale->type()))
            return {.reason =
                        std::string("Reference GEMM ") + name + " block scales must be real."};
        return {.supported = true};
    };
    if (auto result = validateBlockScale(problem.blockScaleA, problem.blockSizeA, m, "A"); !result)
        return result;
    if (auto result = validateBlockScale(problem.blockScaleB, problem.blockSizeB, n, "B"); !result)
        return result;
    if (problem.blockScaleA) {
        if (complexAccumulator)
            return {.reason = "Complex reference GEMM does not support block scaling."};
    }
    if (problem.blockScaleB) {
        if (complexAccumulator)
            return {.reason = "Complex reference GEMM does not support block scaling."};
    }
    if (!hasProvablyDistinctElementOffsets(problem.d.layout()))
        return {.reason = "Reference GEMM requires distinct logical destination elements."};

    if (storageOverlaps(problem.d, problem.a) || storageOverlaps(problem.d, problem.b))
        return {.reason = "Reference GEMM destination must not overlap A or B."};

    for (const Tensor& scale : problem.preQuantizationScalesA)
        if (storageOverlaps(problem.d, scale))
            return {.reason =
                        "Reference GEMM destination must not overlap an A pre-quantization scale."};
    for (const Tensor& scale : problem.preQuantizationScalesB)
        if (storageOverlaps(problem.d, scale))
            return {.reason =
                        "Reference GEMM destination must not overlap a B pre-quantization scale."};
    if (problem.blockScaleA && storageOverlaps(problem.d, *problem.blockScaleA))
        return {.reason = "Reference GEMM destination must not overlap the A block scale."};
    if (problem.blockScaleB && storageOverlaps(problem.d, *problem.blockScaleB))
        return {.reason = "Reference GEMM destination must not overlap the B block scale."};
    return {.supported = true};
}

inline void validateRuntimeGemm(const GemmInvocation& problem) {
    const auto result = checkRuntimeGemm(problem);
    if (!result) throw std::runtime_error(result.reason);
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
    explicit RuntimeGemmArithmetic(const MatmulOptions&,
                                   RuntimeQuantizer<Accumulator> quantizeAccumulator)
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
}  // namespace roc::hostnumerics
