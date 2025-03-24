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

namespace hiptensor
{
    // This hiptensor::math::cos is a workaround for a CK missing function issue.
    // CK has fixed the issue in develop branch, but it has not been made into all
    // CI pipelines.
    // TODO Remove hiptensor::math::cos when `__device__ ck::math::cos` is available in all pipelines.
    namespace math
    {
        template <typename T>
        inline __host__ __device__ T cos(T x)
        {
            return ck::type_convert<T>(::cosf(ck::type_convert<float>(x)));
        };
        template <>
        inline __host__ __device__ float cos<float>(float x)
        {
            return ::cosf(x);
        };
        template <>
        inline __host__ __device__ double cos<double>(double x)
        {
            return ::cos(x);
        };
        template <>
        inline __host__ __device__ ck::half_t cos<ck::half_t>(ck::half_t x)
        {
            return hcos(static_cast<__half>(x));
        };
    }
}

namespace ck
{
    namespace tensor_operation
    {
        namespace element_wise
        {
            using FloatFunctionPtr  = void (*)(float& y, float const& x);
            using DoubleFunctionPtr = void (*)(double& y, double const& x);

            template <typename T>
            __host__ __device__ static void hiptensor_identity(T& y, T const& x)
            {
                y = x;
            };
            template <typename T>
            __host__ __device__ static void hiptensor_sqrt(T& y, T const& x)
            {
                y = ck::math::sqrt(x);
            };
            template <typename T>
            __host__ __device__ static void hiptensor_relu(T& y, T const& x)
            {
                y = x > 0 ? x : 0;
            };
            template <typename T>
            __host__ __device__ static void hiptensor_conj(T& y, T const& x)
            {
                y = x;
            };
            template <typename T>
            __host__ __device__ static void hiptensor_rcp(T& y, T const& x)
            {
                y = 1 / x;
            }
            template <typename T>
            __host__ __device__ static void hiptensor_sigmoid(T& y, T const& x)
            {
                y = 1 / (1 + ck::math::exp(-x));
            }
            template <typename T>
            __host__ __device__ static void hiptensor_tanh(T& y, T const& x)
            {
                y = ck::math::tanh(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_exp(T& y, T const& x)
            {
                y = ck::math::exp(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_log(T& y, T const& x)
            {
                y = ck::math::log(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_abs(T& y, T const& x)
            {
                y = ck::math::abs(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_neg(T& y, T const& x)
            {
                y = -x;
            }
            template <typename T>
            __host__ __device__ static void hiptensor_sin(T& y, T const& x)
            {
                y = ck::math::sin(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_cos(T& y, T const& x)
            {
                y = hiptensor::math::cos(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_tan(T& y, T const& x)
            {
                y = ck::math::tan(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_sinh(T& y, T const& x)
            {
                y = ck::math::sinh(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_cosh(T& y, T const& x)
            {
                y = ck::math::cosh(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_asin(T& y, T const& x)
            {
                y = ck::math::asin(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_acos(T& y, T const& x)
            {
                y = ck::math::acos(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_atan(T& y, T const& x)
            {
                y = ck::math::atan(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_asinh(T& y, T const& x)
            {
                y = ck::math::asinh(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_acosh(T& y, T const& x)
            {
                y = ck::math::acosh(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_atanh(T& y, T const& x)
            {
                y = ck::math::atanh(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_ceil(T& y, T const& x)
            {
                y = ck::math::ceil(x);
            }
            template <typename T>
            __host__ __device__ static void hiptensor_floor(T& y, T const& x)
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

                __host__ __device__ void operator()(double& y, const double& x) const
                {
                    double_ops[op_type](y, x);
                }

                __host__ __device__ void operator()(float& y, const float& x) const
                {
                    float_ops[op_type](y, x);
                }

                __host__ __device__ void operator()(half_t& y, const half_t& x) const
                {
                    float tempX = static_cast<float>(x);
                    float tempY;
                    float_ops[op_type](tempY, tempX);
                    y = static_cast<float>(tempY);
                }

            public:
                hiptensorOperator_t               op_type     = HIPTENSOR_OP_IDENTITY;
                static constexpr FloatFunctionPtr float_ops[] = {
                    hiptensor_identity<float>, // placeholder 0
                    hiptensor_identity<
                        float>, //HIPTENSOR_OP_IDENTITY = 1, ///< Identity operator (i.e., elements are not changed)
                    hiptensor_sqrt<float>, //HIPTENSOR_OP_SQRT     = 2, ///< Square root
                    hiptensor_identity<float>, // placeholder 3
                    hiptensor_identity<float>, // placeholder 4
                    hiptensor_identity<float>, // placeholder 5
                    hiptensor_identity<float>, // placeholder 6
                    hiptensor_identity<float>, // placeholder 7
                    hiptensor_relu<float>, //HIPTENSOR_OP_RELU     = 8, ///< Rectified linear unit
                    hiptensor_conj<float>, //HIPTENSOR_OP_CONJ     = 9, ///< Complex conjugate
                    hiptensor_rcp<float>, //HIPTENSOR_OP_RCP      = 10, ///< Reciprocal
                    hiptensor_sigmoid<float>, //HIPTENSOR_OP_SIGMOID  = 11, ///< y=1/(1+exp(-x))
                    hiptensor_tanh<float>, //HIPTENSOR_OP_TANH     = 12, ///< y=tanh(x)
                    hiptensor_identity<float>, // placeholder 13
                    hiptensor_identity<float>, // placeholder 14
                    hiptensor_identity<float>, // placeholder 15
                    hiptensor_identity<float>, // placeholder 16
                    hiptensor_identity<float>, // placeholder 17
                    hiptensor_identity<float>, // placeholder 18
                    hiptensor_identity<float>, // placeholder 19
                    hiptensor_identity<float>, // placeholder 20
                    hiptensor_identity<float>, // placeholder 21
                    hiptensor_exp<float>, //HIPTENSOR_OP_EXP      = 22, ///< Exponentiation.
                    hiptensor_log<float>, //HIPTENSOR_OP_LOG      = 23, ///< Log (base e).
                    hiptensor_abs<float>, //HIPTENSOR_OP_ABS      = 24, ///< Absolute value.
                    hiptensor_neg<float>, //HIPTENSOR_OP_NEG      = 25, ///< Negation.
                    hiptensor_sin<float>, //HIPTENSOR_OP_SIN      = 26, ///< Sine.
                    hiptensor_cos<float>, //HIPTENSOR_OP_COS      = 27, ///< Cosine.
                    hiptensor_tan<float>, //HIPTENSOR_OP_TAN      = 28, ///< Tangent.
                    hiptensor_sinh<float>, //HIPTENSOR_OP_SINH     = 29, ///< Hyperbolic sine.
                    hiptensor_cosh<float>, //HIPTENSOR_OP_COSH     = 30, ///< Hyperbolic cosine.
                    hiptensor_asin<float>, //HIPTENSOR_OP_ASIN     = 31, ///< Inverse sine.
                    hiptensor_acos<float>, //HIPTENSOR_OP_ACOS     = 32, ///< Inverse cosine.
                    hiptensor_atan<float>, //HIPTENSOR_OP_ATAN     = 33, ///< Inverse tangent.
                    hiptensor_asinh<
                        float>, //HIPTENSOR_OP_ASINH    = 34, ///< Inverse hyperbolic sine.
                    hiptensor_acosh<
                        float>, //HIPTENSOR_OP_ACOSH    = 35, ///< Inverse hyperbolic cosine.
                    hiptensor_atanh<
                        float>, //HIPTENSOR_OP_ATANH    = 36, ///< Inverse hyperbolic tangent.
                    hiptensor_ceil<float>, //HIPTENSOR_OP_CEIL     = 37, ///< Ceiling.
                    hiptensor_floor<float>, //HIPTENSOR_OP_FLOOR    = 38, ///< Floor.
                };

                static constexpr DoubleFunctionPtr double_ops[] = {
                    hiptensor_identity<double>, // placeholder 0
                    hiptensor_identity<
                        double>, //HIPTENSOR_OP_IDENTITY = 1, ///< Identity operator (i.e., elements are not changed)
                    hiptensor_sqrt<double>, //HIPTENSOR_OP_SQRT     = 2, ///< Square root
                    hiptensor_identity<double>, // placeholder 3
                    hiptensor_identity<double>, // placeholder 4
                    hiptensor_identity<double>, // placeholder 5
                    hiptensor_identity<double>, // placeholder 6
                    hiptensor_identity<double>, // placeholder 7
                    hiptensor_relu<double>, //HIPTENSOR_OP_RELU     = 8, ///< Rectified linear unit
                    hiptensor_conj<double>, //HIPTENSOR_OP_CONJ     = 9, ///< Complex conjugate
                    hiptensor_rcp<double>, //HIPTENSOR_OP_RCP      = 10, ///< Reciprocal
                    hiptensor_sigmoid<double>, //HIPTENSOR_OP_SIGMOID  = 11, ///< y=1/(1+exp(-x))
                    hiptensor_tanh<double>, //HIPTENSOR_OP_TANH     = 12, ///< y=tanh(x)
                    hiptensor_identity<double>, // placeholder 13
                    hiptensor_identity<double>, // placeholder 14
                    hiptensor_identity<double>, // placeholder 15
                    hiptensor_identity<double>, // placeholder 16
                    hiptensor_identity<double>, // placeholder 17
                    hiptensor_identity<double>, // placeholder 18
                    hiptensor_identity<double>, // placeholder 19
                    hiptensor_identity<double>, // placeholder 20
                    hiptensor_identity<double>, // placeholder 21
                    hiptensor_exp<double>, //HIPTENSOR_OP_EXP      = 22, ///< Exponentiation.
                    hiptensor_log<double>, //HIPTENSOR_OP_LOG      = 23, ///< Log (base e).
                    hiptensor_abs<double>, //HIPTENSOR_OP_ABS      = 24, ///< Absolute value.
                    hiptensor_neg<double>, //HIPTENSOR_OP_NEG      = 25, ///< Negation.
                    hiptensor_sin<double>, //HIPTENSOR_OP_SIN      = 26, ///< Sine.
                    hiptensor_cos<double>, //HIPTENSOR_OP_COS      = 27, ///< Cosine.
                    hiptensor_tan<double>, //HIPTENSOR_OP_TAN      = 28, ///< Tangent.
                    hiptensor_sinh<double>, //HIPTENSOR_OP_SINH     = 29, ///< Hyperbolic sine.
                    hiptensor_cosh<double>, //HIPTENSOR_OP_COSH     = 30, ///< Hyperbolic cosine.
                    hiptensor_asin<double>, //HIPTENSOR_OP_ASIN     = 31, ///< Inverse sine.
                    hiptensor_acos<double>, //HIPTENSOR_OP_ACOS     = 32, ///< Inverse cosine.
                    hiptensor_atan<double>, //HIPTENSOR_OP_ATAN     = 33, ///< Inverse tangent.
                    hiptensor_asinh<
                        double>, //HIPTENSOR_OP_ASINH    = 34, ///< Inverse hyperbolic sine.
                    hiptensor_acosh<
                        double>, //HIPTENSOR_OP_ACOSH    = 35, ///< Inverse hyperbolic cosine.
                    hiptensor_atanh<
                        double>, //HIPTENSOR_OP_ATANH    = 36, ///< Inverse hyperbolic tangent.
                    hiptensor_ceil<double>, //HIPTENSOR_OP_CEIL     = 37, ///< Ceiling.
                    hiptensor_floor<double>, //HIPTENSOR_OP_FLOOR    = 38, ///< Floor.
                };
            };

            struct HiptensorBinaryOp
            {
                __host__ __device__ HiptensorBinaryOp(hiptensorOperator_t operator_type)
                    : op_type(operator_type)
                {
                }
                __host__ __device__ HiptensorBinaryOp(const HiptensorBinaryOp& dynamic_op)
                    = default;
                __host__            __device__ ~HiptensorBinaryOp() = default;
                __host__ __device__ HiptensorBinaryOp& operator=(const HiptensorBinaryOp& other)
                    = default;

                __host__ __device__ void
                    operator()(double& y, const double& x1, const double& x2) const
                {
                    switch(op_type)
                    {
                    case HIPTENSOR_OP_ADD:
                        y = x1 + x2;
                        break;
                    case HIPTENSOR_OP_MUL:
                        y = x1 * x2;
                        break;
                    case HIPTENSOR_OP_MAX:
                        y = x1 > x2 ? x1 : x2;
                        break;
                    case HIPTENSOR_OP_MIN:
                        y = x1 < x2 ? x1 : x2;
                        break;
                    default:
                        y = y;
                        break;
                    }
                }

                __host__ __device__ void
                    operator()(float& y, const float& x1, const float& x2) const
                {
                    switch(op_type)
                    {
                    case HIPTENSOR_OP_ADD:
                        y = x1 + x2;
                        break;
                    case HIPTENSOR_OP_MUL:
                        y = x1 * x2;
                        break;
                    case HIPTENSOR_OP_MAX:
                        y = x1 > x2 ? x1 : x2;
                        break;
                    case HIPTENSOR_OP_MIN:
                        y = x1 < x2 ? x1 : x2;
                        break;
                    default:
                        y = y;
                        break;
                    }
                }

                __host__ __device__ void
                    operator()(half_t& y, const half_t& x1, const half_t& x2) const
                {
                    float tempX1 = static_cast<float>(x1);
                    float tempX2 = static_cast<float>(x2);
                    float tempY;
                    this->operator()(tempY, tempX1, tempX2);
                    y = static_cast<float>(tempY);
                }

            public:
                hiptensorOperator_t op_type = HIPTENSOR_OP_IDENTITY;
            };
        } // namespace element_wise
    } // namespace tensor_operation
} // namespace ck
#endif // HIPTENSOR_UNARY_ELEMENT_WISE_OPERATION
