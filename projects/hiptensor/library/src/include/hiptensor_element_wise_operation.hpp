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

#pragma once

#include <cassert>

#include <ck/utility/data_type.hpp>
#include <ck/utility/math.hpp>
#include <ck/utility/math_v2.hpp>
#include <ck/utility/type_convert.hpp>

#include <ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp>
#include <hiptensor/hiptensor_types.h>

#if HIPTENSOR_INLINE_UNARY_OPS
#define HIPTENSOR_UNARY_INLINE
#else
#define HIPTENSOR_UNARY_INLINE __attribute__((noinline))
#endif

namespace ck
{
    namespace tensor_operation
    {
        namespace element_wise
        {
            using FloatFunctionPtr  = void (*)(float& y, float const& x);
            using DoubleFunctionPtr = void (*)(double& y, double const& x);

            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_identity(T& y, T const& x)
            {
                y = x;
            };
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_sqrt(T& y, T const& x)
            {
                y = std::sqrt(x);
            };
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_relu(T& y, T const& x)
            {
                y = x > 0 ? x : 0;
            };
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_conj(T& y, T const& x)
            {
                y = x;
            };
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_rcp(T& y, T const& x)
            {
                y = 1 / x;
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_sigmoid(T& y, T const& x)
            {
                y = 1 / (1 + std::exp(-x));
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_tanh(T& y, T const& x)
            {
                y = std::tanh(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_exp(T& y, T const& x)
            {
                y = std::exp(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_log(T& y, T const& x)
            {
                y = std::log(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_abs(T& y, T const& x)
            {
                y = std::abs(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_neg(T& y, T const& x)
            {
                y = -x;
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_sin(T& y, T const& x)
            {
                y = std::sin(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_cos(T& y, T const& x)
            {
                y = std::cos(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_tan(T& y, T const& x)
            {
                y = std::tan(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_sinh(T& y, T const& x)
            {
                y = std::sinh(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_cosh(T& y, T const& x)
            {
                y = std::cosh(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_asin(T& y, T const& x)
            {
                y = std::asin(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_acos(T& y, T const& x)
            {
                y = std::acos(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_atan(T& y, T const& x)
            {
                y = std::atan(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_asinh(T& y, T const& x)
            {
                y = std::asinh(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_acosh(T& y, T const& x)
            {
                y = std::acosh(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_atanh(T& y, T const& x)
            {
                y = std::atanh(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_ceil(T& y, T const& x)
            {
                y = std::ceil(x);
            }
            template <typename T>
            __host__ __device__ HIPTENSOR_UNARY_INLINE static void hiptensor_floor(T& y, T const& x)
            {
                y = std::floor(x);
            }

            struct HiptensorUnaryOp
            {
                __host__ __device__ HiptensorUnaryOp() = default;
                __host__ __device__ explicit HiptensorUnaryOp(hiptensorOperator_t operator_type)
                    : op_type(operator_type)
                {
                }
                __host__ __device__ HiptensorUnaryOp(const HiptensorUnaryOp& dynamic_op) = default;
                __host__            __device__ ~HiptensorUnaryOp()                       = default;
                __host__ __device__ HiptensorUnaryOp& operator=(const HiptensorUnaryOp& other)
                    = default;

                template <typename T>
                __host__ __device__ void switch_op(T& y, T const& x) const
                {
                    switch(op_type)
                    {
                    case HIPTENSOR_OP_IDENTITY:
                        hiptensor_identity(y, x);
                        break;
                    case HIPTENSOR_OP_SQRT:
                        hiptensor_sqrt(y, x);
                        break;
                    case HIPTENSOR_OP_RELU:
                        hiptensor_relu(y, x);
                        break;
                    case HIPTENSOR_OP_CONJ:
                        hiptensor_conj(y, x);
                        break;
                    case HIPTENSOR_OP_RCP:
                        hiptensor_rcp(y, x);
                        break;
                    case HIPTENSOR_OP_SIGMOID:
                        hiptensor_sigmoid(y, x);
                        break;
                    case HIPTENSOR_OP_TANH:
                        hiptensor_tanh(y, x);
                        break;
                    case HIPTENSOR_OP_EXP:
                        hiptensor_exp(y, x);
                        break;
                    case HIPTENSOR_OP_LOG:
                        hiptensor_log(y, x);
                        break;
                    case HIPTENSOR_OP_ABS:
                        hiptensor_abs(y, x);
                        break;
                    case HIPTENSOR_OP_NEG:
                        hiptensor_neg(y, x);
                        break;
                    case HIPTENSOR_OP_SIN:
                        hiptensor_sin(y, x);
                        break;
                    case HIPTENSOR_OP_COS:
                        hiptensor_cos(y, x);
                        break;
                    case HIPTENSOR_OP_TAN:
                        hiptensor_tan(y, x);
                        break;
                    case HIPTENSOR_OP_SINH:
                        hiptensor_sinh(y, x);
                        break;
                    case HIPTENSOR_OP_COSH:
                        hiptensor_cosh(y, x);
                        break;
                    case HIPTENSOR_OP_ASIN:
                        hiptensor_asin(y, x);
                        break;
                    case HIPTENSOR_OP_ACOS:
                        hiptensor_acos(y, x);
                        break;
                    case HIPTENSOR_OP_ATAN:
                        hiptensor_atan(y, x);
                        break;
                    case HIPTENSOR_OP_ASINH:
                        hiptensor_asinh(y, x);
                        break;
                    case HIPTENSOR_OP_ACOSH:
                        hiptensor_acosh(y, x);
                        break;
                    case HIPTENSOR_OP_ATANH:
                        hiptensor_atanh(y, x);
                        break;
                    case HIPTENSOR_OP_CEIL:
                        hiptensor_ceil(y, x);
                        break;
                    case HIPTENSOR_OP_FLOOR:
                        hiptensor_floor(y, x);
                        break;
                    default:
                        hiptensor_identity(y, x);
                        break;
                    }
                }

                __host__ __device__ void operator()(double& y, const double& x) const
                {
                    switch_op(y, x);
                }

                __host__ __device__ void operator()(float& y, const float& x) const
                {
                    switch_op(y, x);
                }

                __host__ __device__ void operator()(half_t& y, const half_t& x) const
                {
                    float tempX = static_cast<float>(x);
                    float tempY;
                    switch_op(tempY, tempX);
                    y = static_cast<float>(tempY);
                }

                __host__ __device__ void operator()(bhalf_t& y, const bhalf_t& x) const
                {
                    float tempX = ck::type_convert<float, bhalf_t>(x);
                    float tempY;
                    switch_op(tempY, tempX);
                    y = ck::type_convert<bhalf_t, float>(tempY);
                }

                // Generic overload for CK contraction kernels where ComputeDataType differs from DataType (e.g. BF16 data and F32 compute).
                // CK calls a_element_op(ComputeType& y, const DataType& x) during the global->LDS transfer. 
                // Without this, bhalf_t (unsigned short) silently integer-promotes to float for op(float& y, const bhalf_t& x), causing incorrect results.
                // As template is less preferred than the explicit same-type overloads, (float,float), (double,double), etc. still be used firstly.
                template <typename Y, typename X>
                __host__ __device__ void operator()(Y& y, const X& x) const
                {
                    float tempX = ck::type_convert<float>(x);
                    float tempY;
                    switch_op(tempY, tempX);
                    y = ck::type_convert<Y>(tempY);
                }

            public:
                hiptensorOperator_t op_type = HIPTENSOR_OP_IDENTITY;
            };

            struct HiptensorBinaryOp
            {
                __host__ __device__ HiptensorBinaryOp(hiptensorOperator_t operator_type)
                    : op_type(operator_type)
                {
                }
                __host__ __device__ HiptensorBinaryOp(const HiptensorBinaryOp& dynamic_op)
                    = default;
                __host__                               __device__ ~HiptensorBinaryOp() = default;
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

                __host__ __device__ void
                    operator()(bhalf_t& y, const bhalf_t& x1, const bhalf_t& x2) const
                {
                    float tempX1 = type_convert<float, bhalf_t>(x1);
                    float tempX2 = type_convert<float, bhalf_t>(x2);
                    float tempY;
                    this->operator()(tempY, tempX1, tempX2);
                    y = type_convert<bhalf_t, float>(tempY);
                }

            public:
                hiptensorOperator_t op_type = HIPTENSOR_OP_IDENTITY;
            };

            struct BilinearUnary
            {
                static constexpr const char* name = "BilinearUnary";

                __host__ __device__ BilinearUnary()
                    : bilinear_op_()
                    , unary_op_()
                {
                }

                __host__ __device__ BilinearUnary(Bilinear bilinear_op, HiptensorUnaryOp unary_op)
                    : bilinear_op_(bilinear_op)
                    , unary_op_(unary_op)
                {
                }

                template <typename Y, typename X0, typename X1>
                __host__ __device__ void operator()(Y& y, const X0& x0, const X1& x1) const
                {
                    X1 x1_tmp;
                    unary_op_(x1_tmp, x1);
                    bilinear_op_(y, x0, x1_tmp);
                }

            private:
                Bilinear         bilinear_op_;
                HiptensorUnaryOp unary_op_;
            };
        } // namespace element_wise
    } // namespace tensor_operation
} // namespace ck
