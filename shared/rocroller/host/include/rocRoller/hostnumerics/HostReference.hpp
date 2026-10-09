// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <roc/hostnumerics/comparison.hpp>
#include <roc/hostnumerics/gemm.hpp>
#include <roc/hostnumerics/tensor.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>

#include <rocRoller/hostnumerics/HostDataGeneration.hpp>

namespace rocRoller::hostnumerics
{
    struct AcceptableGEMMError
    {
        double      relativeL2Tolerance = 0.0;
        std::string reasoning;
    };

    struct HostComparisonResult
    {
        AcceptableGEMMError                 acceptableError;
        roc::hostnumerics::ComparisonReport statistics;

        bool        ok() const;
        std::string message() const;
    };

    roc::hostnumerics::Tensor hostScaleTensor(DataType                 type,
                                              std::span<const uint8_t> values,
                                              size_t                   freeExtent,
                                              size_t                   reductionExtent,
                                              size_t                   blockSize);

    roc::hostnumerics::Tensor hostScaleTensor(DataType                 type,
                                              std::span<const uint8_t> values,
                                              TensorDescriptor const&  dataDescriptor,
                                              size_t                   blockedDimension,
                                              size_t                   blockSize);

    roc::hostnumerics::Tensor computeHostReference(roc::hostnumerics::Tensor                a,
                                                   roc::hostnumerics::Tensor                b,
                                                   roc::hostnumerics::Tensor                c,
                                                   std::optional<roc::hostnumerics::Tensor> scaleA,
                                                   std::optional<roc::hostnumerics::Tensor> scaleB,
                                                   size_t scaleBlockSize,
                                                   float  alpha,
                                                   float  beta);

    roc::hostnumerics::Tensor
        computeHostReference(GeneratedGEMMInputs const&               inputs,
                             std::optional<roc::hostnumerics::Tensor> runtimeScaleA,
                             std::optional<roc::hostnumerics::Tensor> runtimeScaleB,
                             size_t                                   scaleBlockSize,
                             float                                    alpha,
                             float                                    beta);

    HostComparisonResult compareHostReference(roc::hostnumerics::Tensor observed,
                                              roc::hostnumerics::Tensor expected,
                                              AcceptableGEMMError       acceptableError);

    namespace HostReferenceDetail
    {
        template <typename>
        inline constexpr bool alwaysFalse = false;

        template <typename T>
        constexpr roc::hostnumerics::ScalarType outputScalarType()
        {
            using Output = std::remove_cvref_t<T>;
            if constexpr(std::is_same_v<Output, float>)
                return roc::hostnumerics::ScalarType::Float32;
            else if constexpr(std::is_same_v<Output, Half>)
                return roc::hostnumerics::ScalarType::Float16;
            else if constexpr(std::is_same_v<Output, BFloat16>)
                return roc::hostnumerics::ScalarType::BFloat16;
            else
                static_assert(alwaysFalse<Output>,
                              "rocroller-gemm host reference supports only F32, F16, and BF16 "
                              "outputs.");
        }
    }

    inline roc::hostnumerics::Layout hostOutputLayout(size_t rows, size_t columns)
    {
        if(columns != 0 && rows > std::numeric_limits<size_t>::max() / columns)
            throw std::overflow_error("rocRoller output matrix element count overflow.");
        if(rows > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::overflow_error("rocRoller output matrix stride exceeds ptrdiff_t.");
        return roc::hostnumerics::Layout(roc::hostnumerics::Shape{rows, columns},
                                         {1, static_cast<ptrdiff_t>(rows)});
    }

    template <typename T>
    roc::hostnumerics::Tensor
        hostOutputTensor(std::span<const T> values, size_t rows, size_t columns)
    {
        const auto layout = hostOutputLayout(rows, columns);
        if(values.size() != rows * columns)
            throw std::invalid_argument(
                "rocRoller output storage does not match the matrix dimensions.");
        return roc::hostnumerics::Tensor::copyEncodedBackingStorage(
            HostReferenceDetail::outputScalarType<T>(), layout, std::as_bytes(values));
    }

    template <typename Output>
    roc::hostnumerics::Tensor convertHostReferenceTensor(roc::hostnumerics::Tensor floatOutput)
    {
        static_assert(
            std::is_same_v<
                Output,
                float> || std::is_same_v<Output, Half> || std::is_same_v<Output, BFloat16>);

        if(floatOutput.type() != roc::hostnumerics::ScalarType::Float32
           || floatOutput.shape().rank() != 2)
            throw std::invalid_argument(
                "rocRoller output conversion requires a rank-two F32 tensor.");

        const size_t                               rows    = floatOutput.shape()[0];
        const size_t                               columns = floatOutput.shape()[1];
        roc::hostnumerics::ScalarConversionOptions conversion;
        if constexpr(std::is_same_v<Output, BFloat16>)
            conversion.bfloat16Rounding = roc::hostnumerics::BFloat16Rounding::Truncate;
        return floatOutput.copyConvertedTo(HostReferenceDetail::outputScalarType<Output>(),
                                           hostOutputLayout(rows, columns),
                                           conversion);
    }

    template <typename Output>
    std::vector<Output> convertHostReference(roc::hostnumerics::Tensor floatOutput)
    {
        const roc::hostnumerics::Tensor converted
            = convertHostReferenceTensor<Output>(std::move(floatOutput));
        const size_t rows    = converted.shape()[0];
        const size_t columns = converted.shape()[1];
        const auto   storage = converted.rawEncodedBackingStorage();
        if(storage.size() != rows * columns * sizeof(Output))
            throw std::invalid_argument(
                "rocRoller output conversion requires contiguous column-major storage.");
        std::vector<Output> result(rows * columns);
        if(!storage.empty())
            std::memcpy(result.data(), storage.data(), storage.size());
        return result;
    }

    template <typename T>
    double hostReferenceEpsilon()
    {
        if constexpr(std::is_same_v<T, Half>)
            return std::pow(2.0, -10);
        else if constexpr(std::is_same_v<T, BFloat16>)
            return std::pow(2.0, -5);
        else if constexpr(std::is_same_v<T, FP8>)
            return std::pow(2.0, -3);
        else if constexpr(std::is_same_v<T, BF8>)
            return std::pow(2.0, -2);
        else
            return std::numeric_limits<T>::epsilon();
    }

    template <typename A, typename B, typename D>
    AcceptableGEMMError acceptableGEMMError(size_t                       reductionSize,
                                            GPUArchitectureTarget const& architecture)
    {
        const double epsilon = hostReferenceEpsilon<D>();
        double       tolerance;

        std::ostringstream reasoning;
        reasoning << "Output epsilon: " << std::scientific << epsilon;

        if constexpr(std::is_same_v<D, FP8> || std::is_same_v<D, BF8>)
        {
            tolerance = epsilon;
            reasoning << " Error expected to be dominated by conversion.";
        }
        else
        {
            const double scale                     = std::sqrt(static_cast<double>(reductionSize));
            double       fudge                     = 5;
            double       extraArchitectureFudgeBF8 = 0.0;
            double       extraArchitectureFudgeBF6 = 0.0;
            double       extraArchitectureFudgeFP4 = 0.0;

            if(architecture.gfx == GPUArchitectureGFX::GFX1250)
            {
                extraArchitectureFudgeBF8 = 5.8;
                extraArchitectureFudgeBF6 = 5.8;
                extraArchitectureFudgeFP4 = 2.8;
            }
            else if(architecture.gfx == GPUArchitectureGFX::GFX950)
            {
                extraArchitectureFudgeBF8 = 2.58;
            }

            reasoning << " K: " << reductionSize;
            reasoning << " Problem size scaling: " << scale;
            reasoning << " Fudge: " << fudge;

            if constexpr(std::is_same_v<A, B>)
            {
                if constexpr(std::is_same_v<A, BF6>)
                {
                    fudge *= 3 + extraArchitectureFudgeBF6;
                    reasoning << " Increase fudge for BF6: " << fudge;
                }
                if constexpr(std::is_same_v<A, BF8>)
                {
                    fudge *= 5 + extraArchitectureFudgeBF8;
                    reasoning << " Increase fudge for BF8: " << fudge;
                }
                if constexpr(std::is_same_v<A, FP8>)
                {
                    fudge *= 6.0 + (architecture.gfx == GPUArchitectureGFX::GFX950 ? 0.5 : 0.0);
                    reasoning << " Increase fudge for FP8: " << fudge;
                }
            }
            else
            {
                if constexpr(std::is_same_v<A, BF8> || std::is_same_v<B, BF8>)
                {
                    fudge *= 5 + extraArchitectureFudgeBF8;
                    reasoning << " Increase fudge for mixed BF8: " << fudge;
                }
                else if constexpr(std::is_same_v<A, FP8> || std::is_same_v<B, FP8>)
                {
                    fudge *= 4.95;
                    reasoning << " Increase fudge for mixed FP8: " << fudge;
                }
                else if constexpr(std::is_same_v<A, BF6> || std::is_same_v<B, BF6>)
                {
                    fudge *= 3;
                    reasoning << " Increase fudge for mixed BF6: " << fudge;
                }

                if constexpr(std::is_same_v<A, FP4> || std::is_same_v<B, FP4>)
                {
                    fudge *= 3 + extraArchitectureFudgeFP4;
                    reasoning << " Increase fudge for mixed FP4: " << fudge;
                }
            }
            tolerance = fudge * epsilon * scale;
        }

        return {tolerance, reasoning.str()};
    }
}
