/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_ELEMENT_WISE_OPERATION_COMPLEX_HPP
#define HIPTENSOR_ELEMENT_WISE_OPERATION_COMPLEX_HPP

#include <binary_element_wise_operation.hpp>
#include <hip/hip_complex.h>
#include <hiptensor_unary_element_wise_operation.hpp>

namespace ck
{
    namespace tensor_operation
    {
        namespace element_wise
        {

            struct ScaleComplex : public Scale
            {
                __host__ __device__ ScaleComplex(hipDoubleComplex scale)
                    : Scale(hipCreal(scale))
                {
                    scale_ = scale;
                }

                template <typename Y, typename X>
                __host__ __device__ void operator()(Y& y, const X& x) const;

                template <>
                __host__ __device__ void
                    operator()<hipFloatComplex, hipFloatComplex>(hipFloatComplex&       y,
                                                                 const hipFloatComplex& x) const
                {
                    y = hipCmulf(hipComplexDoubleToFloat(scale_), x);
                };

                template <>
                __host__ __device__ void
                    operator()<hipDoubleComplex, hipDoubleComplex>(hipDoubleComplex&       y,
                                                                   const hipDoubleComplex& x) const
                {
                    y = hipCmul(scale_, x);
                };

                // complex * float
                hipDoubleComplex scale_;
            };

            struct BilinearComplex : public Bilinear
            {
                BilinearComplex(hipDoubleComplex alpha, hipDoubleComplex beta)
                    : Bilinear(hipCreal(alpha), hipCreal(beta))
                {
                    alpha_ = alpha;
                    beta_  = beta;
                }

                template <typename Y, typename X0, typename X1>
                __host__ __device__ constexpr void operator()(Y&, const X0&, const X1&) const;

                template <>
                __host__ __device__ constexpr void
                    operator()<hipDoubleComplex, hipDoubleComplex, hipDoubleComplex>(
                        hipDoubleComplex&       y,
                        const hipDoubleComplex& x0,
                        const hipDoubleComplex& x1) const
                {
                    y = hipCadd(hipCmul(alpha_, x0), hipCmul(beta_, x1));
                };

                template <>
                __host__ __device__ constexpr void
                    operator()<hipFloatComplex, hipFloatComplex, hipFloatComplex>(
                        hipFloatComplex&       y,
                        const hipFloatComplex& x0,
                        const hipFloatComplex& x1) const
                {
                    y = hipCaddf(hipCmulf(hipComplexDoubleToFloat(alpha_), x0),
                                 hipCmulf(hipComplexDoubleToFloat(beta_), x1));
                };

                hipDoubleComplex alpha_;
                hipDoubleComplex beta_;
            };

        } // namespace element_wise
    } // namespace tensor_operation
} // namespace ck

#endif // HIPTENSOR_ELEMENT_WISE_OPERATION_COMPLEX_HPP
