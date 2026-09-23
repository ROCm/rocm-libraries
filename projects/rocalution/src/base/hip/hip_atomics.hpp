/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_HIP_ATOMICS_HPP_
#define ROCALUTION_HIP_HIP_ATOMICS_HPP_

#include <hip/hip_runtime.h>

#ifdef SUPPORT_COMPLEX
#include <complex>
#include <hip/hip_complex.h>
#endif

namespace rocalution
{
    // atomicCAS
    __device__ __forceinline__ int atomicCAS(int* address, int compare, int val)
    {
        return ::atomicCAS(address, compare, val);
    }

    __device__ __forceinline__ unsigned long long int atomicCAS(unsigned long long int* address,
                                                                unsigned long long int  compare,
                                                                unsigned long long int  val)
    {
        return ::atomicCAS(address, compare, val);
    }

    __device__ __forceinline__ int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val)
    {
        return ::atomicCAS((unsigned long long*)address, (int64_t)compare, (int64_t)val);
    }

    __device__ __forceinline__ int atomicAdd(int* ptr, int val)
    {
        return ::atomicAdd(ptr, val);
    }

    // atomicAdd
    __device__ __forceinline__ int64_t atomicAdd(int64_t* ptr, int64_t val)
    {
        return ::atomicAdd((unsigned long long*)ptr, val);
    }

    __device__ __forceinline__ float atomicAdd(float* ptr, float val)
    {
        return ::atomicAdd(ptr, val);
    }

    __device__ __forceinline__ double atomicAdd(double* ptr, double val)
    {
        return ::atomicAdd(ptr, val);
    }

#ifdef SUPPORT_COMPLEX
    __device__ __forceinline__ hipComplex atomicAdd(hipComplex* ptr, hipComplex val)
    {
        return make_hipFloatComplex(atomicAdd((float*)ptr, hipCrealf(val)),
                                    atomicAdd((float*)ptr + 1, hipCimagf(val)));
    }

    __device__ __forceinline__ hipDoubleComplex atomicAdd(hipDoubleComplex* ptr,
                                                          hipDoubleComplex  val)
    {
        return make_hipDoubleComplex(atomicAdd((double*)ptr, hipCreal(val)),
                                     atomicAdd((double*)ptr + 1, hipCimag(val)));
    }

    __device__ __forceinline__ std::complex<float> atomicAdd(std::complex<float>* ptr,
                                                             std::complex<float>  val)
    {
        return std::complex<float>(atomicAdd((float*)ptr, val.real()),
                                   atomicAdd((float*)ptr + 1, val.imag()));
    }

    __device__ __forceinline__ std::complex<double> atomicAdd(std::complex<double>* ptr,
                                                              std::complex<double>  val)
    {
        return std::complex<double>(atomicAdd((double*)ptr, val.real()),
                                    atomicAdd((double*)ptr + 1, val.imag()));
    }
#endif

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_ATOMICS_HPP_
