/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

// Product-private TensileLite adapter.

// TensileLite reference API owned by the shared host-validation component.

#include <roc/host_validation/comparison.hpp>

#include <Tensile/ContractionProblem.hpp>

#include <Tensile/DataTypes.hpp>

namespace TensileLite
{
    namespace Client
    {
        // AlmostEqual tolerance constants per type.
        // Formula: |a - b| < tolerance * (|a| + |b| + 1)
        constexpr float AlmostEqualTolerance_Half = static_cast<float>(
            roc::host_validation::defaultSymmetricRelativeTolerance(
                roc::host_validation::ScalarType::Float16));
        constexpr float AlmostEqualTolerance_BFloat16 = static_cast<float>(
            roc::host_validation::defaultSymmetricRelativeTolerance(
                roc::host_validation::ScalarType::BFloat16));
        // tolerance * epsilon = 2 * 0.0625; 2*eps needed for SR
        constexpr float AlmostEqualTolerance_Float8 = static_cast<float>(
            roc::host_validation::defaultSymmetricRelativeTolerance(
                roc::host_validation::ScalarType::Float8E4M3));
        // tolerance * epsilon = 2 * 0.125; 2*eps needed for SR
        constexpr float AlmostEqualTolerance_BFloat8 = static_cast<float>(
            roc::host_validation::defaultSymmetricRelativeTolerance(
                roc::host_validation::ScalarType::Float8E5M2));
        // 7 digits precision - 2
        constexpr float AlmostEqualTolerance_Float = static_cast<float>(
            roc::host_validation::defaultSymmetricRelativeTolerance(
                roc::host_validation::ScalarType::Float32));
        // 15 digits precision - 2
        constexpr double AlmostEqualTolerance_Double
            = roc::host_validation::defaultSymmetricRelativeTolerance(
                roc::host_validation::ScalarType::Float64);

        // threshold is largest allowed delta. -1 uses default for each type
        template <typename T>
        inline bool AlmostEqual(T a, T b, double threshold = -1.0);

        template <>
        inline bool AlmostEqual(Half a, Half b, double threshold)
        {
            return roc::host_validation::valuesClose(
                static_cast<float>(a),
                static_cast<float>(b),
                roc::host_validation::defaultComparisonOptions(
                    roc::host_validation::ScalarType::Float16));
        }

        template <>
        inline bool AlmostEqual(Float8 a, Float8 b, double threshold)
        {
            return roc::host_validation::valuesClose(
                static_cast<float>(a),
                static_cast<float>(b),
                roc::host_validation::defaultComparisonOptions(
                    roc::host_validation::ScalarType::Float8E4M3));
        }

        template <>
        inline bool AlmostEqual(BFloat8 a, BFloat8 b, double threshold)
        {
            return roc::host_validation::valuesClose(
                static_cast<float>(a),
                static_cast<float>(b),
                roc::host_validation::defaultComparisonOptions(
                    roc::host_validation::ScalarType::Float8E5M2));
        }

        template <>
        inline bool AlmostEqual(Float8_fnuz a, Float8_fnuz b, double threshold)
        {
            return roc::host_validation::valuesClose(
                static_cast<float>(a),
                static_cast<float>(b),
                roc::host_validation::defaultComparisonOptions(
                    roc::host_validation::ScalarType::Float8E4M3Fnuz));
        }

        template <>
        inline bool AlmostEqual(BFloat8_fnuz a, BFloat8_fnuz b, double threshold)
        {
            return roc::host_validation::valuesClose(
                static_cast<float>(a),
                static_cast<float>(b),
                roc::host_validation::defaultComparisonOptions(
                    roc::host_validation::ScalarType::Float8E5M2Fnuz));
        }

        template <>
        inline bool AlmostEqual(BFloat16 a, BFloat16 b, double threshold)
        {
            return roc::host_validation::valuesClose(
                static_cast<float>(a),
                static_cast<float>(b),
                roc::host_validation::defaultComparisonOptions(
                    roc::host_validation::ScalarType::BFloat16));
        }

        template <>
        inline bool AlmostEqual(float a, float b, double threshold)
        {
            const std::optional<double> override =
                threshold > 0.0 ? std::optional<double>(threshold)
                                : std::nullopt;
            return roc::host_validation::valuesClose(
                a,
                b,
                roc::host_validation::defaultComparisonOptions(
                    roc::host_validation::ScalarType::Float32,
                    override));
        }

        template <>
        inline bool AlmostEqual(double a, double b, double threshold)
        {
            return roc::host_validation::valuesClose(
                a,
                b,
                roc::host_validation::defaultComparisonOptions(
                    roc::host_validation::ScalarType::Float64));
        }
        template <>
        inline bool AlmostEqual(int8_t a, int8_t b, double threshold)
        {
            return roc::host_validation::valuesClose(a, b);
        }
        template <>
        inline bool AlmostEqual(int a, int b, double threshold)
        {
            return roc::host_validation::valuesClose(a, b);
        }
        template <>
        inline bool AlmostEqual(unsigned int a, unsigned int b, double threshold)
        {
            return roc::host_validation::valuesClose(a, b);
        }
        template <>
        inline bool AlmostEqual(std::complex<float> a, std::complex<float> b, double threshold)
        {
            return roc::host_validation::valuesClose(
                a,
                b,
                roc::host_validation::defaultComparisonOptions(
                    roc::host_validation::ScalarType::ComplexFloat32));
        }

        template <>
        inline bool AlmostEqual(std::complex<double> a, std::complex<double> b, double threshold)
        {
            return roc::host_validation::valuesClose(
                a,
                b,
                roc::host_validation::defaultComparisonOptions(
                    roc::host_validation::ScalarType::ComplexFloat64));
        }

        void SolveCPU(ContractionProblem const* contraction,
                      ProblemInputs const*      inputs,
                      size_t                    elementsToValidate);

        // Specialized solver for ungrouped GEMM problems. The tiled backend is
        // attempted for eligible dense work; all supported descriptors then
        // use the canonical runtime-typed component implementation.
        void SolveGemmCPU(ContractionProblemGemm const& problem,
                          ContractionInputs const&      inputs,
                          size_t                        elementsToValidate,
                          bool                          tryFastPath = true);

        // Product-private descriptor adapters. Returns false without modifying
        // outputs when a descriptor is not representable.
        bool tryRuntimeCanonicalGemm(ContractionProblemGemm const& problem,
                                     ContractionInputs const&      inputs,
                                     size_t                        elementsToValidate);
        bool tryRuntimeTiledGemm(ContractionProblemGemm const& problem,
                                 ContractionInputs const&      inputs,
                                 size_t                        elementsToValidate);

        // Check whether a given contraction problem is eligible for the fast CPU GEMM path.
        // This inspects problem geometry, data types, and feature flags but does not
        // look at runtime input buffers.
        bool isFastPathEligible(ContractionProblemGemm const& problem);

    } // namespace Client
} // namespace TensileLite
