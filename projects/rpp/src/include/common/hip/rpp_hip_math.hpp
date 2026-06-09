/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_HIP_MATH_HPP
#define RPP_HIP_MATH_HPP
#define RPP_HIP_MATH_DEPENDENCIES

// Arithmetic operations: Add (+), Subtract (-), Multiply (*)
enum class ArithmeticOp { Add, Subtract, Multiply };

template<ArithmeticOp Op>
struct Arithmetic
{
    template<typename T>
    __device__ __forceinline__ static T op(T a, T b)
    {
        if constexpr (Op == ArithmeticOp::Add)      return a + b;
        if constexpr (Op == ArithmeticOp::Subtract) return a - b;
        if constexpr (Op == ArithmeticOp::Multiply) return a * b;
    }
};

using ArithmeticAdd      = Arithmetic<ArithmeticOp::Add>;
using ArithmeticSubtract = Arithmetic<ArithmeticOp::Subtract>;
using ArithmeticMultiply = Arithmetic<ArithmeticOp::Multiply>;

// Arithmetic DIVIDE operation
template <typename T>
struct ArithmeticDivide
{
    __device__ __forceinline__ static float op(T a, T b)
    {
        return static_cast<float>(a) / static_cast<float>(b);
    }
};

// -------------------- Set 1 - scalar helper kernels --------------------

// Functor for bitwise AND operation
struct BitwiseAnd {
    template <typename T>
    __device__ __forceinline__ static T op(T a, T b)
    {
        return a & b;
    }
};

// Functor for bitwise OR operation
struct BitwiseOr {
    template <typename T>
    __device__ __forceinline__ static T op(T a, T b) 
    {
        return a | b;
    }
};

// Functor for bitwise XOR operation
struct BitwiseXor {
    template <typename T>
    __device__ __forceinline__ static T op(T a, T b)
    {
        return a ^ b;
    }
};

// /******************** DEVICE MATH HELPER FUNCTIONS ********************/

// float8 min

__device__ __forceinline__ void rpp_hip_math_min8(d_float8 *srcPtr_f8, float *dstPtr)
{
    *dstPtr = fminf(fminf(fminf(fminf(fminf(fminf(fminf(srcPtr_f8->f1[0], srcPtr_f8->f1[1]), srcPtr_f8->f1[2]), srcPtr_f8->f1[3]), srcPtr_f8->f1[4]), srcPtr_f8->f1[5]), srcPtr_f8->f1[6]), srcPtr_f8->f1[7]);
}

// float8 max

__device__ __forceinline__ void rpp_hip_math_max8(d_float8 *srcPtr_f8, float *dstPtr)
{
    *dstPtr = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(srcPtr_f8->f1[0], srcPtr_f8->f1[1]), srcPtr_f8->f1[2]), srcPtr_f8->f1[3]), srcPtr_f8->f1[4]), srcPtr_f8->f1[5]), srcPtr_f8->f1[6]), srcPtr_f8->f1[7]);
}

// d_float8 floor

__device__ __forceinline__ void rpp_hip_math_floor8(d_float8 *srcPtr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f1[0] = floorf(srcPtr_f8->f1[0]);
    dstPtr_f8->f1[1] = floorf(srcPtr_f8->f1[1]);
    dstPtr_f8->f1[2] = floorf(srcPtr_f8->f1[2]);
    dstPtr_f8->f1[3] = floorf(srcPtr_f8->f1[3]);
    dstPtr_f8->f1[4] = floorf(srcPtr_f8->f1[4]);
    dstPtr_f8->f1[5] = floorf(srcPtr_f8->f1[5]);
    dstPtr_f8->f1[6] = floorf(srcPtr_f8->f1[6]);
    dstPtr_f8->f1[7] = floorf(srcPtr_f8->f1[7]);
}

// d_float16 floor

__device__ __forceinline__ void rpp_hip_math_floor16(d_float16 *srcPtr_f16, d_float16 *dstPtr_f16)
{
    dstPtr_f16->f1[ 0] = floorf(srcPtr_f16->f1[ 0]);
    dstPtr_f16->f1[ 1] = floorf(srcPtr_f16->f1[ 1]);
    dstPtr_f16->f1[ 2] = floorf(srcPtr_f16->f1[ 2]);
    dstPtr_f16->f1[ 3] = floorf(srcPtr_f16->f1[ 3]);
    dstPtr_f16->f1[ 4] = floorf(srcPtr_f16->f1[ 4]);
    dstPtr_f16->f1[ 5] = floorf(srcPtr_f16->f1[ 5]);
    dstPtr_f16->f1[ 6] = floorf(srcPtr_f16->f1[ 6]);
    dstPtr_f16->f1[ 7] = floorf(srcPtr_f16->f1[ 7]);
    dstPtr_f16->f1[ 8] = floorf(srcPtr_f16->f1[ 8]);
    dstPtr_f16->f1[ 9] = floorf(srcPtr_f16->f1[ 9]);
    dstPtr_f16->f1[10] = floorf(srcPtr_f16->f1[10]);
    dstPtr_f16->f1[11] = floorf(srcPtr_f16->f1[11]);
    dstPtr_f16->f1[12] = floorf(srcPtr_f16->f1[12]);
    dstPtr_f16->f1[13] = floorf(srcPtr_f16->f1[13]);
    dstPtr_f16->f1[14] = floorf(srcPtr_f16->f1[14]);
    dstPtr_f16->f1[15] = floorf(srcPtr_f16->f1[15]);
}

// d_float8 nearbyintf

__device__ __forceinline__ void rpp_hip_math_nearbyintf8(d_float8 *srcPtr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f1[0] = nearbyintf(srcPtr_f8->f1[0]);
    dstPtr_f8->f1[1] = nearbyintf(srcPtr_f8->f1[1]);
    dstPtr_f8->f1[2] = nearbyintf(srcPtr_f8->f1[2]);
    dstPtr_f8->f1[3] = nearbyintf(srcPtr_f8->f1[3]);
    dstPtr_f8->f1[4] = nearbyintf(srcPtr_f8->f1[4]);
    dstPtr_f8->f1[5] = nearbyintf(srcPtr_f8->f1[5]);
    dstPtr_f8->f1[6] = nearbyintf(srcPtr_f8->f1[6]);
    dstPtr_f8->f1[7] = nearbyintf(srcPtr_f8->f1[7]);
}

// Generic math operation function for d_float8
template<typename Operation>
__device__ __forceinline__ void rpp_hip_math_op8(d_float8 *src1Ptr_f8, d_float8 *src2Ptr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f4[0] = Operation::op(src1Ptr_f8->f4[0], src2Ptr_f8->f4[0]);
    dstPtr_f8->f4[1] = Operation::op(src1Ptr_f8->f4[1], src2Ptr_f8->f4[1]);
}

// Generic math operation function for d_uint8
template<typename Operation>
__device__ __forceinline__ void rpp_hip_math_op8(d_uint8 *src1Ptr_f8, d_uint8 *src2Ptr_f8, d_uint8 *dstPtr_f8)
{
    dstPtr_f8->ui4[0] = Operation::op(src1Ptr_f8->ui4[0], src2Ptr_f8->ui4[0]);
    dstPtr_f8->ui4[1] = Operation::op(src1Ptr_f8->ui4[1], src2Ptr_f8->ui4[1]);
}

// Generic math operation function for d_int8
template<typename Operation>
__device__ __forceinline__ void rpp_hip_math_op8(d_int8 *src1Ptr_f8, d_int8 *src2Ptr_f8, d_int8 *dstPtr_f8)
{
    dstPtr_f8->i4[0] = Operation::op(src1Ptr_f8->i4[0], src2Ptr_f8->i4[0]);
    dstPtr_f8->i4[1] = Operation::op(src1Ptr_f8->i4[1], src2Ptr_f8->i4[1]);
}

// Generic math operation function for d_ushort8
template<typename Operation>
__device__ __forceinline__ void rpp_hip_math_op8(d_ushort8 *src1Ptr_f8, d_ushort8 *src2Ptr_f8, d_ushort8 *dstPtr_f8)
{
    dstPtr_f8->us4[0] = Operation::op(src1Ptr_f8->us4[0], src2Ptr_f8->us4[0]);
    dstPtr_f8->us4[1] = Operation::op(src1Ptr_f8->us4[1], src2Ptr_f8->us4[1]);
}

// Generic math operation function for d_short8
template<typename Operation>
__device__ __forceinline__ void rpp_hip_math_op8(d_short8 *src1Ptr_f8, d_short8 *src2Ptr_f8, d_short8 *dstPtr_f8)
{
    dstPtr_f8->s4[0] = Operation::op(src1Ptr_f8->s4[0], src2Ptr_f8->s4[0]);
    dstPtr_f8->s4[1] = Operation::op(src1Ptr_f8->s4[1], src2Ptr_f8->s4[1]);
}

// Generic math operation function for d_uchar8
template<typename Operation>
__device__ __forceinline__ void rpp_hip_math_op8(d_uchar8 *src1Ptr_f8, d_uchar8 *src2Ptr_f8, d_uchar8 *dstPtr_f8)
{
    dstPtr_f8->uc4[0] = Operation::op(src1Ptr_f8->uc4[0], src2Ptr_f8->uc4[0]);
    dstPtr_f8->uc4[1] = Operation::op(src1Ptr_f8->uc4[1], src2Ptr_f8->uc4[1]);
}

// Generic math operation function for d_schar8
template<typename Operation>
__device__ __forceinline__ void rpp_hip_math_op8(d_schar8 *src1Ptr_f8, d_schar8 *src2Ptr_f8, d_schar8 *dstPtr_f8)
{
    dstPtr_f8->sc4[0] = Operation::op(src1Ptr_f8->sc4[0], src2Ptr_f8->sc4[0]);
    dstPtr_f8->sc4[1] = Operation::op(src1Ptr_f8->sc4[1], src2Ptr_f8->sc4[1]);
}

// d_float8 subtract

template<typename Vec8>
__device__ __forceinline__ void rpp_hip_math_subtract8(Vec8 *src1Ptr_f8, Vec8 *src2Ptr_f8, Vec8 *dstPtr_f8)
{
    rpp_hip_math_op8<ArithmeticSubtract>(src1Ptr_f8, src2Ptr_f8, dstPtr_f8);
}

// d_float8 add

template<typename Vec8>
__device__ __forceinline__ void rpp_hip_math_add8(Vec8 *src1Ptr_f8, Vec8 *src2Ptr_f8, Vec8 *dstPtr_f8)
{
    rpp_hip_math_op8<ArithmeticAdd>(src1Ptr_f8, src2Ptr_f8, dstPtr_f8);
}

// d_float24 add

__device__ __forceinline__ void rpp_hip_math_add24(d_float24 *src1Ptr_f24, d_float24 *src2Ptr_f24, d_float24 *dstPtr_f24)
{
    dstPtr_f24->f4[0] = src1Ptr_f24->f4[0] + src2Ptr_f24->f4[0];
    dstPtr_f24->f4[1] = src1Ptr_f24->f4[1] + src2Ptr_f24->f4[1];
    dstPtr_f24->f4[2] = src1Ptr_f24->f4[2] + src2Ptr_f24->f4[2];
    dstPtr_f24->f4[3] = src1Ptr_f24->f4[3] + src2Ptr_f24->f4[3];
    dstPtr_f24->f4[4] = src1Ptr_f24->f4[4] + src2Ptr_f24->f4[4];
    dstPtr_f24->f4[5] = src1Ptr_f24->f4[5] + src2Ptr_f24->f4[5];
}

// d_float8 add with constant

__device__ __forceinline__ void rpp_hip_math_add8_const(d_float8 *src_f8, d_float8 *dst_f8, float4 addend_f4)
{
    dst_f8->f4[0] = src_f8->f4[0] + addend_f4;
    dst_f8->f4[1] = src_f8->f4[1] + addend_f4;
}

// d_float24 add with constant

__device__ __forceinline__ void rpp_hip_math_add24_const(d_float24 *src_f24, d_float24 *dst_f24, float4 addend_f4)
{
    dst_f24->f4[0] = src_f24->f4[0] + addend_f4;
    dst_f24->f4[1] = src_f24->f4[1] + addend_f4;
    dst_f24->f4[2] = src_f24->f4[2] + addend_f4;
    dst_f24->f4[3] = src_f24->f4[3] + addend_f4;
    dst_f24->f4[4] = src_f24->f4[4] + addend_f4;
    dst_f24->f4[5] = src_f24->f4[5] + addend_f4;
}

// d_float16 subtract

__device__ __forceinline__ void rpp_hip_math_subtract16(d_float16 *src1Ptr_f16, d_float16 *src2Ptr_f16, d_float16 *dstPtr_f16)
{
    dstPtr_f16->f4[0] = src1Ptr_f16->f4[0] - src2Ptr_f16->f4[0];
    dstPtr_f16->f4[1] = src1Ptr_f16->f4[1] - src2Ptr_f16->f4[1];
    dstPtr_f16->f4[2] = src1Ptr_f16->f4[2] - src2Ptr_f16->f4[2];
    dstPtr_f16->f4[3] = src1Ptr_f16->f4[3] - src2Ptr_f16->f4[3];
}

// d_float8 subtract with constant

__device__ __forceinline__ void rpp_hip_math_subtract8_const(d_float8 *src_f8, d_float8 *dst_f8, float4 subtrahend_f4)
{
    dst_f8->f4[0] = src_f8->f4[0] - subtrahend_f4;
    dst_f8->f4[1] = src_f8->f4[1] - subtrahend_f4;
}

// d_float24 subtract with constant

__device__ __forceinline__ void rpp_hip_math_subtract24_const(d_float24 *src_f24, d_float24 *dst_f24, float4 subtrahend_f4)
{
    dst_f24->f4[0] = src_f24->f4[0] - subtrahend_f4;
    dst_f24->f4[1] = src_f24->f4[1] - subtrahend_f4;
    dst_f24->f4[2] = src_f24->f4[2] - subtrahend_f4;
    dst_f24->f4[3] = src_f24->f4[3] - subtrahend_f4;
    dst_f24->f4[4] = src_f24->f4[4] - subtrahend_f4;
    dst_f24->f4[5] = src_f24->f4[5] - subtrahend_f4;
}

// d_float8 multiply

template<typename Vec8>
__device__ __forceinline__ void rpp_hip_math_multiply8(Vec8 *src1Ptr_f8, Vec8 *src2Ptr_f8, Vec8 *dstPtr_f8)
{
    rpp_hip_math_op8<ArithmeticMultiply>(src1Ptr_f8, src2Ptr_f8, dstPtr_f8);
}

// d_float24 multiply

__device__ __forceinline__ void rpp_hip_math_multiply24(d_float24 *src1Ptr_f24, d_float24 *src2Ptr_f24, d_float24 *dstPtr_f24)
{
    dstPtr_f24->f4[0] = src1Ptr_f24->f4[0] * src2Ptr_f24->f4[0];
    dstPtr_f24->f4[1] = src1Ptr_f24->f4[1] * src2Ptr_f24->f4[1];
    dstPtr_f24->f4[2] = src1Ptr_f24->f4[2] * src2Ptr_f24->f4[2];
    dstPtr_f24->f4[3] = src1Ptr_f24->f4[3] * src2Ptr_f24->f4[3];
    dstPtr_f24->f4[4] = src1Ptr_f24->f4[4] * src2Ptr_f24->f4[4];
    dstPtr_f24->f4[5] = src1Ptr_f24->f4[5] * src2Ptr_f24->f4[5];
}

// d_float8 multiply with constant

__device__ __forceinline__ void rpp_hip_math_multiply8_const(d_float8 *src_f8, d_float8 *dst_f8, float4 multiplier_f4)
{
    dst_f8->f4[0] = src_f8->f4[0] * multiplier_f4;
    dst_f8->f4[1] = src_f8->f4[1] * multiplier_f4;
}

// d_float24 multiply with constant

__device__ __forceinline__ void rpp_hip_math_multiply24_const(d_float24 *src_f24, d_float24 *dst_f24, float4 multiplier_f4)
{
    dst_f24->f4[0] = src_f24->f4[0] * multiplier_f4;
    dst_f24->f4[1] = src_f24->f4[1] * multiplier_f4;
    dst_f24->f4[2] = src_f24->f4[2] * multiplier_f4;
    dst_f24->f4[3] = src_f24->f4[3] * multiplier_f4;
    dst_f24->f4[4] = src_f24->f4[4] * multiplier_f4;
    dst_f24->f4[5] = src_f24->f4[5] * multiplier_f4;
}

// d_float8 divide

__device__ __forceinline__ void rpp_hip_math_divide8(d_float8 *src1Ptr_f8, d_float8 *src2Ptr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f4[0] = src1Ptr_f8->f4[0] / src2Ptr_f8->f4[0];
    dstPtr_f8->f4[1] = src1Ptr_f8->f4[1] / src2Ptr_f8->f4[1];
}

__device__ __forceinline__ void rpp_hip_math_divide8(d_uint8 *src1Ptr_f8, d_uint8 *src2Ptr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f1[0] = static_cast<float>(src1Ptr_f8->ui1[0]) / static_cast<float>(src2Ptr_f8->ui1[0]);
    dstPtr_f8->f1[1] = static_cast<float>(src1Ptr_f8->ui1[1]) / static_cast<float>(src2Ptr_f8->ui1[1]);
    dstPtr_f8->f1[2] = static_cast<float>(src1Ptr_f8->ui1[2]) / static_cast<float>(src2Ptr_f8->ui1[2]);
    dstPtr_f8->f1[3] = static_cast<float>(src1Ptr_f8->ui1[3]) / static_cast<float>(src2Ptr_f8->ui1[3]);
    dstPtr_f8->f1[4] = static_cast<float>(src1Ptr_f8->ui1[4]) / static_cast<float>(src2Ptr_f8->ui1[4]);
    dstPtr_f8->f1[5] = static_cast<float>(src1Ptr_f8->ui1[5]) / static_cast<float>(src2Ptr_f8->ui1[5]);
    dstPtr_f8->f1[6] = static_cast<float>(src1Ptr_f8->ui1[6]) / static_cast<float>(src2Ptr_f8->ui1[6]);
    dstPtr_f8->f1[7] = static_cast<float>(src1Ptr_f8->ui1[7]) / static_cast<float>(src2Ptr_f8->ui1[7]);
}

__device__ __forceinline__ void rpp_hip_math_divide8(d_int8 *src1Ptr_f8, d_int8 *src2Ptr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f1[0] = static_cast<float>(src1Ptr_f8->i1[0]) / static_cast<float>(src2Ptr_f8->i1[0]);
    dstPtr_f8->f1[1] = static_cast<float>(src1Ptr_f8->i1[1]) / static_cast<float>(src2Ptr_f8->i1[1]);
    dstPtr_f8->f1[2] = static_cast<float>(src1Ptr_f8->i1[2]) / static_cast<float>(src2Ptr_f8->i1[2]);
    dstPtr_f8->f1[3] = static_cast<float>(src1Ptr_f8->i1[3]) / static_cast<float>(src2Ptr_f8->i1[3]);
    dstPtr_f8->f1[4] = static_cast<float>(src1Ptr_f8->i1[4]) / static_cast<float>(src2Ptr_f8->i1[4]);
    dstPtr_f8->f1[5] = static_cast<float>(src1Ptr_f8->i1[5]) / static_cast<float>(src2Ptr_f8->i1[5]);
    dstPtr_f8->f1[6] = static_cast<float>(src1Ptr_f8->i1[6]) / static_cast<float>(src2Ptr_f8->i1[6]);
    dstPtr_f8->f1[7] = static_cast<float>(src1Ptr_f8->i1[7]) / static_cast<float>(src2Ptr_f8->i1[7]);
}

__device__ __forceinline__ void rpp_hip_math_divide8(d_ushort8 *src1Ptr_f8, d_ushort8 *src2Ptr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f1[0] = static_cast<float>(src1Ptr_f8->us1[0]) / static_cast<float>(src2Ptr_f8->us1[0]);
    dstPtr_f8->f1[1] = static_cast<float>(src1Ptr_f8->us1[1]) / static_cast<float>(src2Ptr_f8->us1[1]);
    dstPtr_f8->f1[2] = static_cast<float>(src1Ptr_f8->us1[2]) / static_cast<float>(src2Ptr_f8->us1[2]);
    dstPtr_f8->f1[3] = static_cast<float>(src1Ptr_f8->us1[3]) / static_cast<float>(src2Ptr_f8->us1[3]);
    dstPtr_f8->f1[4] = static_cast<float>(src1Ptr_f8->us1[4]) / static_cast<float>(src2Ptr_f8->us1[4]);
    dstPtr_f8->f1[5] = static_cast<float>(src1Ptr_f8->us1[5]) / static_cast<float>(src2Ptr_f8->us1[5]);
    dstPtr_f8->f1[6] = static_cast<float>(src1Ptr_f8->us1[6]) / static_cast<float>(src2Ptr_f8->us1[6]);
    dstPtr_f8->f1[7] = static_cast<float>(src1Ptr_f8->us1[7]) / static_cast<float>(src2Ptr_f8->us1[7]);
}

__device__ __forceinline__ void rpp_hip_math_divide8(d_short8 *src1Ptr_f8, d_short8 *src2Ptr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f1[0] = static_cast<float>(src1Ptr_f8->s1[0]) / static_cast<float>(src2Ptr_f8->s1[0]);
    dstPtr_f8->f1[1] = static_cast<float>(src1Ptr_f8->s1[1]) / static_cast<float>(src2Ptr_f8->s1[1]);
    dstPtr_f8->f1[2] = static_cast<float>(src1Ptr_f8->s1[2]) / static_cast<float>(src2Ptr_f8->s1[2]);
    dstPtr_f8->f1[3] = static_cast<float>(src1Ptr_f8->s1[3]) / static_cast<float>(src2Ptr_f8->s1[3]);
    dstPtr_f8->f1[4] = static_cast<float>(src1Ptr_f8->s1[4]) / static_cast<float>(src2Ptr_f8->s1[4]);
    dstPtr_f8->f1[5] = static_cast<float>(src1Ptr_f8->s1[5]) / static_cast<float>(src2Ptr_f8->s1[5]);
    dstPtr_f8->f1[6] = static_cast<float>(src1Ptr_f8->s1[6]) / static_cast<float>(src2Ptr_f8->s1[6]);
    dstPtr_f8->f1[7] = static_cast<float>(src1Ptr_f8->s1[7]) / static_cast<float>(src2Ptr_f8->s1[7]);
}

__device__ __forceinline__ void rpp_hip_math_divide8(d_schar8 *src1Ptr_f8, d_schar8 *src2Ptr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f1[0] = static_cast<float>(src1Ptr_f8->sc1[0]) / static_cast<float>(src2Ptr_f8->sc1[0]);
    dstPtr_f8->f1[1] = static_cast<float>(src1Ptr_f8->sc1[1]) / static_cast<float>(src2Ptr_f8->sc1[1]);
    dstPtr_f8->f1[2] = static_cast<float>(src1Ptr_f8->sc1[2]) / static_cast<float>(src2Ptr_f8->sc1[2]);
    dstPtr_f8->f1[3] = static_cast<float>(src1Ptr_f8->sc1[3]) / static_cast<float>(src2Ptr_f8->sc1[3]);
    dstPtr_f8->f1[4] = static_cast<float>(src1Ptr_f8->sc1[4]) / static_cast<float>(src2Ptr_f8->sc1[4]);
    dstPtr_f8->f1[5] = static_cast<float>(src1Ptr_f8->sc1[5]) / static_cast<float>(src2Ptr_f8->sc1[5]);
    dstPtr_f8->f1[6] = static_cast<float>(src1Ptr_f8->sc1[6]) / static_cast<float>(src2Ptr_f8->sc1[6]);
    dstPtr_f8->f1[7] = static_cast<float>(src1Ptr_f8->sc1[7]) / static_cast<float>(src2Ptr_f8->sc1[7]);
}

__device__ __forceinline__ void rpp_hip_math_divide8(d_uchar8 *src1Ptr_f8, d_uchar8 *src2Ptr_f8, d_float8 *dstPtr_f8)
{
    dstPtr_f8->f1[0] = static_cast<float>(src1Ptr_f8->uc1[0]) / static_cast<float>(src2Ptr_f8->uc1[0]);
    dstPtr_f8->f1[1] = static_cast<float>(src1Ptr_f8->uc1[1]) / static_cast<float>(src2Ptr_f8->uc1[1]);
    dstPtr_f8->f1[2] = static_cast<float>(src1Ptr_f8->uc1[2]) / static_cast<float>(src2Ptr_f8->uc1[2]);
    dstPtr_f8->f1[3] = static_cast<float>(src1Ptr_f8->uc1[3]) / static_cast<float>(src2Ptr_f8->uc1[3]);
    dstPtr_f8->f1[4] = static_cast<float>(src1Ptr_f8->uc1[4]) / static_cast<float>(src2Ptr_f8->uc1[4]);
    dstPtr_f8->f1[5] = static_cast<float>(src1Ptr_f8->uc1[5]) / static_cast<float>(src2Ptr_f8->uc1[5]);
    dstPtr_f8->f1[6] = static_cast<float>(src1Ptr_f8->uc1[6]) / static_cast<float>(src2Ptr_f8->uc1[6]);
    dstPtr_f8->f1[7] = static_cast<float>(src1Ptr_f8->uc1[7]) / static_cast<float>(src2Ptr_f8->uc1[7]);
}

// d_float8 divide with constant

__device__ __forceinline__ void rpp_hip_math_divide8_const(d_float8 *src_f8, d_float8 *dst_f8, float4 divisor_f4)
{
    dst_f8->f4[0] = divisor_f4 / src_f8->f4[0];
    dst_f8->f4[1] = divisor_f4 / src_f8->f4[1];
}

// Generic bitwise operation function for d_uchar8
template<typename Operation>
__device__ __forceinline__ void rpp_hip_math_bitwise_op8(d_uchar8 *src1_uc8, d_uchar8 *src2_uc8, d_uchar8 *dst_uc8)
{
        dst_uc8->uc1[0] = Operation::op(src1_uc8->uc1[0], src2_uc8->uc1[0]);
        dst_uc8->uc1[1] = Operation::op(src1_uc8->uc1[1], src2_uc8->uc1[1]);
        dst_uc8->uc1[2] = Operation::op(src1_uc8->uc1[2], src2_uc8->uc1[2]);
        dst_uc8->uc1[3] = Operation::op(src1_uc8->uc1[3], src2_uc8->uc1[3]);
        dst_uc8->uc1[4] = Operation::op(src1_uc8->uc1[4], src2_uc8->uc1[4]);
        dst_uc8->uc1[5] = Operation::op(src1_uc8->uc1[5], src2_uc8->uc1[5]);
        dst_uc8->uc1[6] = Operation::op(src1_uc8->uc1[6], src2_uc8->uc1[6]);
        dst_uc8->uc1[7] = Operation::op(src1_uc8->uc1[7], src2_uc8->uc1[7]);
}

// Generic bitwise operation function for d_ushort8
template<typename Operation>
__device__ __forceinline__ void rpp_hip_math_bitwise_op8(d_ushort8 *src1_us8, d_ushort8 *src2_us8, d_ushort8 *dst_us8)
{
        dst_us8->us1[0] = Operation::op(src1_us8->us1[0], src2_us8->us1[0]);
        dst_us8->us1[1] = Operation::op(src1_us8->us1[1], src2_us8->us1[1]);
        dst_us8->us1[2] = Operation::op(src1_us8->us1[2], src2_us8->us1[2]);
        dst_us8->us1[3] = Operation::op(src1_us8->us1[3], src2_us8->us1[3]);
        dst_us8->us1[4] = Operation::op(src1_us8->us1[4], src2_us8->us1[4]);
        dst_us8->us1[5] = Operation::op(src1_us8->us1[5], src2_us8->us1[5]);
        dst_us8->us1[6] = Operation::op(src1_us8->us1[6], src2_us8->us1[6]);
        dst_us8->us1[7] = Operation::op(src1_us8->us1[7], src2_us8->us1[7]);
}

// Generic bitwise operation function for d_uint8
template<typename Operation>
__device__ __forceinline__ void rpp_hip_math_bitwise_op8(d_uint8 *src1_ui8, d_uint8 *src2_ui8, d_uint8 *dst_ui8)
{
        dst_ui8->ui1[0] = Operation::op(src1_ui8->ui1[0], src2_ui8->ui1[0]);
        dst_ui8->ui1[1] = Operation::op(src1_ui8->ui1[1], src2_ui8->ui1[1]);
        dst_ui8->ui1[2] = Operation::op(src1_ui8->ui1[2], src2_ui8->ui1[2]);
        dst_ui8->ui1[3] = Operation::op(src1_ui8->ui1[3], src2_ui8->ui1[3]);
        dst_ui8->ui1[4] = Operation::op(src1_ui8->ui1[4], src2_ui8->ui1[4]);
        dst_ui8->ui1[5] = Operation::op(src1_ui8->ui1[5], src2_ui8->ui1[5]);
        dst_ui8->ui1[6] = Operation::op(src1_ui8->ui1[6], src2_ui8->ui1[6]);
        dst_ui8->ui1[7] = Operation::op(src1_ui8->ui1[7], src2_ui8->ui1[7]);
}

// Used to do bitwise and of the scaled float image representations - Values scaled from 0 to 255 with constant mask
__device__ __forceinline__ void rpp_hip_math_scaled_bitwiseAnd8(d_float8 *src_f8, d_uchar8 *src_mask_u8, d_float8 *dst_f8)
{
        dst_f8->f1[0] = (float)((uchar)nearbyintf(src_f8->f1[0]) & src_mask_u8->uc1[0]);
        dst_f8->f1[1] = (float)((uchar)nearbyintf(src_f8->f1[1]) & src_mask_u8->uc1[1]);
        dst_f8->f1[2] = (float)((uchar)nearbyintf(src_f8->f1[2]) & src_mask_u8->uc1[2]);
        dst_f8->f1[3] = (float)((uchar)nearbyintf(src_f8->f1[3]) & src_mask_u8->uc1[3]);
        dst_f8->f1[4] = (float)((uchar)nearbyintf(src_f8->f1[4]) & src_mask_u8->uc1[4]);
        dst_f8->f1[5] = (float)((uchar)nearbyintf(src_f8->f1[5]) & src_mask_u8->uc1[5]);
        dst_f8->f1[6] = (float)((uchar)nearbyintf(src_f8->f1[6]) & src_mask_u8->uc1[6]);
        dst_f8->f1[7] = (float)((uchar)nearbyintf(src_f8->f1[7]) & src_mask_u8->uc1[7]);
}

// d_uchar8 bitwiseNOT

__device__ __forceinline__ void rpp_hip_math_bitwiseNot8(d_uchar8 *src_uc8, d_uchar8 *dst_uc8)
{
    dst_uc8->uc1[0] = ~src_uc8->uc1[0];
    dst_uc8->uc1[1] = ~src_uc8->uc1[1];
    dst_uc8->uc1[2] = ~src_uc8->uc1[2];
    dst_uc8->uc1[3] = ~src_uc8->uc1[3];
    dst_uc8->uc1[4] = ~src_uc8->uc1[4];
    dst_uc8->uc1[5] = ~src_uc8->uc1[5];
    dst_uc8->uc1[6] = ~src_uc8->uc1[6];
    dst_uc8->uc1[7] = ~src_uc8->uc1[7];
}

__device__ __forceinline__ float rpp_hip_math_inverse_sqrt1(float x)
{
    float xHalf = 0.5f * x;
    int i = *(int*)&x;                              // float bits in int
    i = NEWTON_METHOD_INITIAL_GUESS - (i >> 1);     // initial guess for Newton's method
    x = *(float*)&i;                                // new bits to float
    x = x * (1.5f - xHalf * x * x);                 // One round of Newton's method

    return x;
}

__device__ __forceinline__ float4 rpp_hip_math_inverse_sqrt4(float4 val_f4)
{
    float4 xHalf_f4 = MAKE_FLOAT4(0.5f) * val_f4;
    int4 val_i4 = *(int4 *)&val_f4;                                     // float bits in int
    val_i4 = MAKE_INT4(NEWTON_METHOD_INITIAL_GUESS) - (val_i4 >> MAKE_INT4(1));  // initial guess for Newton's method
    val_f4 = *(float4 *)&val_i4;                                        // new bits to float
    val_f4 = val_f4 * (MAKE_FLOAT4(1.5f) - xHalf_f4 * val_f4 * val_f4);          // One round of Newton's method

    return val_f4;
}

__device__ __forceinline__ void rpp_hip_math_sqrt8(d_float8 *pix_f8, d_float8 *pixSqrt_f8)
{
    pixSqrt_f8->f4[0] = rpp_hip_math_inverse_sqrt4(pix_f8->f4[0]);
    pixSqrt_f8->f4[1] = rpp_hip_math_inverse_sqrt4(pix_f8->f4[1]);

    float4 one_f4 = MAKE_FLOAT4(1.0f);
    pixSqrt_f8->f4[0] = one_f4 / pixSqrt_f8->f4[0];
    pixSqrt_f8->f4[1] = one_f4 / pixSqrt_f8->f4[1];
}

__device__ __forceinline__ void rpp_hip_math_sqrt24(d_float24 *pix_f24, d_float24 *pixSqrt_f24)
{
    pixSqrt_f24->f4[0] = rpp_hip_math_inverse_sqrt4(pix_f24->f4[0]);
    pixSqrt_f24->f4[1] = rpp_hip_math_inverse_sqrt4(pix_f24->f4[1]);
    pixSqrt_f24->f4[2] = rpp_hip_math_inverse_sqrt4(pix_f24->f4[2]);
    pixSqrt_f24->f4[3] = rpp_hip_math_inverse_sqrt4(pix_f24->f4[3]);
    pixSqrt_f24->f4[4] = rpp_hip_math_inverse_sqrt4(pix_f24->f4[4]);
    pixSqrt_f24->f4[5] = rpp_hip_math_inverse_sqrt4(pix_f24->f4[5]);

    float4 one_f4 = MAKE_FLOAT4(1.0f);
    pixSqrt_f24->f4[0] = one_f4 / pixSqrt_f24->f4[0];
    pixSqrt_f24->f4[1] = one_f4 / pixSqrt_f24->f4[1];
    pixSqrt_f24->f4[2] = one_f4 / pixSqrt_f24->f4[2];
    pixSqrt_f24->f4[3] = one_f4 / pixSqrt_f24->f4[3];
    pixSqrt_f24->f4[4] = one_f4 / pixSqrt_f24->f4[4];
    pixSqrt_f24->f4[5] = one_f4 / pixSqrt_f24->f4[5];
}

__device__ __forceinline__ float rpp_hip_math_exp_lim256approx(float x)
{
  x = 1.0 + x * ONE_OVER_256;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;

  return x;
}

__device__ __forceinline__ void rpp_hip_math_log(d_float8 *src_f8, d_float8 *dst_f8)
{
    for(int i = 0; i < 8; i++)
        src_f8->f1[i] = (!src_f8->f1[i]) ? std::nextafter(0.0f, 1.0f) : fabsf(src_f8->f1[i]);

    dst_f8->f1[0] = __logf(src_f8->f1[0]);
    dst_f8->f1[1] = __logf(src_f8->f1[1]);
    dst_f8->f1[2] = __logf(src_f8->f1[2]);
    dst_f8->f1[3] = __logf(src_f8->f1[3]);
    dst_f8->f1[4] = __logf(src_f8->f1[4]);
    dst_f8->f1[5] = __logf(src_f8->f1[5]);
    dst_f8->f1[6] = __logf(src_f8->f1[6]);
    dst_f8->f1[7] = __logf(src_f8->f1[7]);
}

__device__ __forceinline__ void rpp_hip_math_log1p(d_float8 *src_f8, d_float8 *dst_f8)
{
    dst_f8->f1[0] = __logf((src_f8->f1[0]));
    dst_f8->f1[1] = __logf((src_f8->f1[1]));
    dst_f8->f1[2] = __logf((src_f8->f1[2]));
    dst_f8->f1[3] = __logf((src_f8->f1[3]));
    dst_f8->f1[4] = __logf((src_f8->f1[4]));
    dst_f8->f1[5] = __logf((src_f8->f1[5]));
    dst_f8->f1[6] = __logf((src_f8->f1[6]));
    dst_f8->f1[7] = __logf((src_f8->f1[7]));
}

// Maximum of three float values
__device__ __forceinline__ float rpp_hip_max3(const float3 &src_f3)
{
    return fmaxf(src_f3.x, fmaxf(src_f3.y, src_f3.z));
}

// Minimum of three float values
__device__ __forceinline__ float rpp_hip_min3(const float3 &src_f3)
{
    return fminf(src_f3.x, fminf(src_f3.y, src_f3.z));
}

// Median of three float values
__device__ __forceinline__ float rpp_hip_median3(const float3 &src_f3)
{
    return __builtin_amdgcn_fmed3f(src_f3.x, src_f3.y, src_f3.z);
}

#endif // RPP_HIP_MATH_HPP