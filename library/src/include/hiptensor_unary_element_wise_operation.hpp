/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_UNARY_ELEMENT_WISE_OPERATION
#define HIPTENSOR_UNARY_ELEMENT_WISE_OPERATION
#include <cassert>

#include <ck/utility/data_type.hpp>
#include <ck/utility/math.hpp>
#include <ck/utility/math_v2.hpp>
#include <ck/utility/type_convert.hpp>

#include <hiptensor/hiptensor_types.hpp>

namespace ck
{
    namespace math
    {
        template <typename T>
        inline __device__ T cos(T x)
        {
            return ck::type_convert<T>(::cosf(ck::type_convert<float>(x)));
        };
        template <>
        inline __device__ float cos<float>(float x)
        {
            return ::cosf(x);
        };
        template <>
        inline __device__ double cos<double>(double x)
        {
            return ::cos(x);
        };
        template <>
        inline __device__ half_t cos<half_t>(half_t x)
        {
            return hcos(static_cast<__half>(x));
        };
    }
    namespace tensor_operation
    {
        namespace element_wise
        {
            using FunctionPtr = void (*)(float& y, float const& x);

            __host__ __device__ static void hiptensor_identity(float& y, float const& x)
            {
                y = x;
            };
            __host__ __device__ static void hiptensor_sqrt(float& y, float const& x)
            {
                y = ck::math::sqrt(x);
            };
            __host__ __device__ static void hiptensor_relu(float& y, float const& x)
            {
                y = x > 0 ? x : 0;
            };
            __host__ __device__ static void hiptensor_conj(float& y, float const& x)
            {
                y = x;
            };
            __host__ __device__ static void hiptensor_rcp(float& y, float const& x)
            {
                y = 1 / x;
            }
            __host__ __device__ static void hiptensor_sigmoid(float& y, float const& x)
            {
                y = 1 / (1 + ck::math::exp(-x));
            }
            __host__ __device__ static void hiptensor_tanh(float& y, float const& x)
            {
                y = ck::math::tanh(x);
            }
            __host__ __device__ static void hiptensor_exp(float& y, float const& x)
            {
                y = ck::math::exp(x);
            }
            __host__ __device__ static void hiptensor_log(float& y, float const& x)
            {
                y = ck::math::log(x);
            }
            __host__ __device__ static void hiptensor_abs(float& y, float const& x)
            {
                y = ck::math::abs(x);
            }
            __host__ __device__ static void hiptensor_neg(float& y, float const& x)
            {
                y = -x;
            }
            __host__ __device__ static void hiptensor_sin(float& y, float const& x)
            {
                y = ck::math::sin(x);
            }
            __host__ __device__ static void hiptensor_cos(float& y, float const& x)
            {
                y = ck::math::cos(x);
            }
            __host__ __device__ static void hiptensor_tan(float& y, float const& x)
            {
                y = ck::math::tan(x);
            }
            __host__ __device__ static void hiptensor_sinh(float& y, float const& x)
            {
                y = ck::math::sinh(x);
            }
            __host__ __device__ static void hiptensor_cosh(float& y, float const& x)
            {
                y = ck::math::cosh(x);
            }
            __host__ __device__ static void hiptensor_asin(float& y, float const& x)
            {
                y = ck::math::asin(x);
            }
            __host__ __device__ static void hiptensor_acos(float& y, float const& x)
            {
                y = ck::math::acos(x);
            }
            __host__ __device__ static void hiptensor_atan(float& y, float const& x)
            {
                y = ck::math::atan(x);
            }
            __host__ __device__ static void hiptensor_asinh(float& y, float const& x)
            {
                y = ck::math::asinh(x);
            }
            __host__ __device__ static void hiptensor_acosh(float& y, float const& x)
            {
                y = ck::math::acosh(x);
            }
            __host__ __device__ static void hiptensor_atanh(float& y, float const& x)
            {
                y = ck::math::atanh(x);
            }
            __host__ __device__ static void hiptensor_ceil(float& y, float const& x)
            {
                y = ck::math::ceil(x);
            }
            __host__ __device__ static void hiptensor_floor(float& y, float const& x)
            {
                y = ck::math::floor(x);
            }

            struct HiptensorUnaryOp
            {
                __host__ __device__ HiptensorUnaryOp(hiptensorOperator_t operator_type)
                    : op_type(operator_type)
                {
                }
                __host__ __device__ HiptensorUnaryOp(const HiptensorUnaryOp& dynamic_op) = default;
                __host__            __device__ ~HiptensorUnaryOp()                       = default;
                __host__ __device__ HiptensorUnaryOp& operator=(const HiptensorUnaryOp& other)
                    = default;

                __host__ __device__ void operator()(float& y, const float& x) const
                {
                    ops[op_type](y, x);
                }

                __host__ __device__ void operator()(half_t& y, const half_t& x) const
                {
                    float tempX = static_cast<float>(x);
                    float tempY;
                    ops[op_type](tempY, tempX);
                    y = static_cast<float>(tempY);
                }

            public:
                hiptensorOperator_t          op_type = HIPTENSOR_OP_IDENTITY;
                static constexpr FunctionPtr ops[]   = {
                    hiptensor_identity, // placeholder 0
                    hiptensor_identity, //HIPTENSOR_OP_IDENTITY = 1, ///< Identity operator (i.e., elements are not changed)
                    hiptensor_sqrt, //HIPTENSOR_OP_SQRT     = 2, ///< Square root
                    hiptensor_identity, // placeholder 3
                    hiptensor_identity, // placeholder 4
                    hiptensor_identity, // placeholder 5
                    hiptensor_identity, // placeholder 6
                    hiptensor_identity, // placeholder 7
                    hiptensor_relu, //HIPTENSOR_OP_RELU     = 8, ///< Rectified linear unit
                    hiptensor_conj, //HIPTENSOR_OP_CONJ     = 9, ///< Complex conjugate
                    hiptensor_rcp, //HIPTENSOR_OP_RCP      = 10, ///< Reciprocal
                    hiptensor_sigmoid, //HIPTENSOR_OP_SIGMOID  = 11, ///< y=1/(1+exp(-x))
                    hiptensor_tanh, //HIPTENSOR_OP_TANH     = 12, ///< y=tanh(x)
                    hiptensor_identity, // placeholder 13
                    hiptensor_identity, // placeholder 14
                    hiptensor_identity, // placeholder 15
                    hiptensor_identity, // placeholder 16
                    hiptensor_identity, // placeholder 17
                    hiptensor_identity, // placeholder 18
                    hiptensor_identity, // placeholder 19
                    hiptensor_identity, // placeholder 20
                    hiptensor_identity, // placeholder 21
                    hiptensor_exp, //HIPTENSOR_OP_EXP      = 22, ///< Exponentiation.
                    hiptensor_log, //HIPTENSOR_OP_LOG      = 23, ///< Log (base e).
                    hiptensor_abs, //HIPTENSOR_OP_ABS      = 24, ///< Absolute value.
                    hiptensor_neg, //HIPTENSOR_OP_NEG      = 25, ///< Negation.
                    hiptensor_sin, //HIPTENSOR_OP_SIN      = 26, ///< Sine.
                    hiptensor_cos, //HIPTENSOR_OP_COS      = 27, ///< Cosine.
                    hiptensor_tan, //HIPTENSOR_OP_TAN      = 28, ///< Tangent.
                    hiptensor_sinh, //HIPTENSOR_OP_SINH     = 29, ///< Hyperbolic sine.
                    hiptensor_cosh, //HIPTENSOR_OP_COSH     = 30, ///< Hyperbolic cosine.
                    hiptensor_asin, //HIPTENSOR_OP_ASIN     = 31, ///< Inverse sine.
                    hiptensor_acos, //HIPTENSOR_OP_ACOS     = 32, ///< Inverse cosine.
                    hiptensor_atan, //HIPTENSOR_OP_ATAN     = 33, ///< Inverse tangent.
                    hiptensor_asinh, //HIPTENSOR_OP_ASINH    = 34, ///< Inverse hyperbolic sine.
                    hiptensor_acosh, //HIPTENSOR_OP_ACOSH    = 35, ///< Inverse hyperbolic cosine.
                    hiptensor_atanh, //HIPTENSOR_OP_ATANH    = 36, ///< Inverse hyperbolic tangent.
                    hiptensor_ceil, //HIPTENSOR_OP_CEIL     = 37, ///< Ceiling.
                    hiptensor_floor, //HIPTENSOR_OP_FLOOR    = 38, ///< Floor.
                };
            };

        } // namespace element_wise
    } // namespace tensor_operation
} // namespace ck
#endif // HIPTENSOR_UNARY_ELEMENT_WISE_OPERATION
