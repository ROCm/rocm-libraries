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

#include <Tensile/ContractionProblem.hpp>

#include <Tensile/DataTypes.hpp>

namespace TensileLite
{
    namespace Client
    {
        template <typename T>
        inline bool AlmostEqual(T a, T b);

        template <>
        inline bool AlmostEqual(Half a, Half b)
        {
            Half absA = (a > 0) ? a : -a;
            Half absB = (b > 0) ? b : -b;
            // this avoids NaN when inf is compared against inf in the alternative code
            // path
            if(static_cast<float>(absA) == std::numeric_limits<float>::infinity()
               || // numeric_limits is yet to
               // support _Float16 type
               // properly;
               static_cast<float>(absB)
                   == std::numeric_limits<float>::infinity()) // however promoting it to
            // float works just as fine
            {
                return a == b;
            }
            Half absDiff = (a - b > 0) ? a - b : b - a;
            return absDiff / (absA + absB + 1) < 0.01;
        }

        template <>
        inline bool AlmostEqual(Float8 a, Float8 b)
        {
            Float8 absA    = (a > static_cast<Float8>(0.0f)) ? a : static_cast<Float8>(0.0f) - a;
            Float8 absB    = (b > static_cast<Float8>(0.0f)) ? b : static_cast<Float8>(0.0f) - b;
            Float8 absDiff = (a - b > static_cast<Float8>(0.0f)) ? a - b : b - a;
            return absDiff / (absA + absB + static_cast<Float8>(1.0f)) < static_cast<Float8>(
                       0.125f); // tolerance * eps = 2 * 0.0625; 2*eps needed for SR
        }

        template <>
        inline bool AlmostEqual(BFloat8 a, BFloat8 b)
        {
            BFloat8 absA    = (a > static_cast<BFloat8>(0.0f)) ? a : static_cast<BFloat8>(0.0f) - a;
            BFloat8 absB    = (b > static_cast<BFloat8>(0.0f)) ? b : static_cast<BFloat8>(0.0f) - b;
            BFloat8 absDiff = (a - b > static_cast<BFloat8>(0.0f)) ? a - b : b - a;
            return absDiff / (absA + absB + static_cast<BFloat8>(1.0f)) < static_cast<BFloat8>(
                       0.25f); // tolerance * epsilon = 2 * 0.125; 2*eps needed for SR
        }

        template <>
        inline bool AlmostEqual(Float8_fnuz a, Float8_fnuz b)
        {
            Float8_fnuz absA    = (a > static_cast<Float8_fnuz>(0.0f)) ? a : static_cast<Float8_fnuz>(0.0f) - a;
            Float8_fnuz absB    = (b > static_cast<Float8_fnuz>(0.0f)) ? b : static_cast<Float8_fnuz>(0.0f) - b;
            Float8_fnuz absDiff = (a - b > static_cast<Float8_fnuz>(0.0f)) ? a - b : b - a;
            return absDiff / (absA + absB + static_cast<Float8_fnuz>(1.0f)) < static_cast<Float8_fnuz>(
                       0.125f); // tolerance * eps = 2 * 0.0625; 2*eps needed for SR
        }

        template <>
        inline bool AlmostEqual(BFloat8_fnuz a, BFloat8_fnuz b)
        {
            BFloat8_fnuz absA    = (a > static_cast<BFloat8_fnuz>(0.0f)) ? a : static_cast<BFloat8_fnuz>(0.0f) - a;
            BFloat8_fnuz absB    = (b > static_cast<BFloat8_fnuz>(0.0f)) ? b : static_cast<BFloat8_fnuz>(0.0f) - b;
            BFloat8_fnuz absDiff = (a - b > static_cast<BFloat8_fnuz>(0.0f)) ? a - b : b - a;
            return absDiff / (absA + absB + static_cast<BFloat8_fnuz>(1.0f)) < static_cast<BFloat8_fnuz>(
                       0.25f); // tolerance * epsilon = 2 * 0.125; 2*eps needed for SR
        }

    template <>
        inline bool AlmostEqual(BFloat16 a, BFloat16 b)
        {
            BFloat16 absA = (a > static_cast<BFloat16>(0.0f)) ? a : static_cast<BFloat16>(0.0f) - a;
            BFloat16 absB = (b > static_cast<BFloat16>(0.0f)) ? b : static_cast<BFloat16>(0.0f) - b;
            BFloat16 absDiff = (a - b > static_cast<BFloat16>(0.0f)) ? a - b : b - a;
            return absDiff / (absA + absB + static_cast<BFloat16>(1.0f))
                   < static_cast<BFloat16>(0.1f);
        }

        template <>
        inline bool AlmostEqual(float a, float b)
        {
            return std::fabs(a - b) / (std::fabs(a) + std::fabs(b) + 1)
                   < 0.0001; // 7 digits of precision - 2
        }

        template <>
        inline bool AlmostEqual(double a, double b)
        {
            return std::fabs(a - b) / (std::fabs(a) + std::fabs(b) + 1)
                   < 0.000000000001; // 15 digits of precision - 2
        }
        template <>
        inline bool AlmostEqual(int8_t a, int8_t b)
        {
            return a == b;
        }
        template <>
        inline bool AlmostEqual(int a, int b)
        {
            return a == b;
        }
        template <>
        inline bool AlmostEqual(unsigned int a, unsigned int b)
        {
            return a == b;
        }
        template <>
        inline bool AlmostEqual(std::complex<float> a, std::complex<float> b)
        {
            return AlmostEqual(a.real(), b.real()) && AlmostEqual(a.imag(), b.imag());
        }

        template <>
        inline bool AlmostEqual(std::complex<double> a, std::complex<double> b)
        {
            return AlmostEqual(a.real(), b.real()) && AlmostEqual(a.imag(), b.imag());
        }

        template <typename Inputs,
                  typename Accumulator = typename Inputs::DType,
                  typename MathOpAccum = Accumulator>
        struct ReferenceSolution
        {
            static void SolveCPU(ContractionProblemGemm const& contraction,
                                 ContractionInputs const&      inputs,
                                 size_t                        elementsToValidate);
            static void SolveCPU(ContractionProblemGroupedGemm const& contractions,
                                 ContractionGroupedInputs const&      inputs,
                                 size_t                               elementsToValidate);
        };

        void SolveCPU(ContractionProblem const* contraction,
                      ProblemInputs const*      inputs,
                      size_t                    elementsToValidate);

    } // namespace Client
} // namespace TensileLite
