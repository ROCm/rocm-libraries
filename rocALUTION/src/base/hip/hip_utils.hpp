/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_HIP_UTILS_HPP_
#define ROCALUTION_HIP_HIP_UTILS_HPP_

#include "../../utils/log.hpp"
#include "../../utils/type_traits.hpp"
#include "../backend_manager.hpp"
#include "backend_hip.hpp"
#include "hip_atomics.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>

#ifdef SUPPORT_COMPLEX
#include <complex>
#include <hip/hip_complex.h>
#endif

// clang-format off
#ifndef ROCALUTION_USE_MOVE_DPP
#if defined(__GFX8__) || defined(__GFX9__)
#define ROCALUTION_USE_MOVE_DPP 1
#else
#define ROCALUTION_USE_MOVE_DPP 0
#endif
#endif
// clang-format on

#define ROCBLAS_HANDLE(handle) *static_cast<rocblas_handle*>(handle)
#define ROCSPARSE_HANDLE(handle) *static_cast<rocsparse_handle*>(handle)
#define HIPSTREAM(handle) *static_cast<hipStream_t*>(handle)

#define CHECK_HIP_ERROR(file, line)                              \
    {                                                            \
        hipError_t err_t;                                        \
        if((err_t = hipGetLastError()) != hipSuccess)            \
        {                                                        \
            LOG_INFO("HIP error: " << hipGetErrorString(err_t)); \
            LOG_INFO("File: " << file << "; line: " << line);    \
            exit(1);                                             \
        }                                                        \
    }

#define CHECK_ROCBLAS_ERROR(stat_t, file, line)               \
    {                                                         \
        if(stat_t != rocblas_status_success)                  \
        {                                                     \
            LOG_INFO("rocBLAS error " << stat_t);             \
            if(stat_t == rocblas_status_invalid_handle)       \
                LOG_INFO("rocblas_status_invalid_handle");    \
            if(stat_t == rocblas_status_not_implemented)      \
                LOG_INFO("rocblas_status_not_implemented");   \
            if(stat_t == rocblas_status_invalid_pointer)      \
                LOG_INFO("rocblas_status_invalid_pointer");   \
            if(stat_t == rocblas_status_invalid_size)         \
                LOG_INFO("rocblas_status_invalid_size");      \
            if(stat_t == rocblas_status_memory_error)         \
                LOG_INFO("rocblas_status_memory_error");      \
            if(stat_t == rocblas_status_internal_error)       \
                LOG_INFO("rocblas_status_internal_error");    \
            LOG_INFO("File: " << file << "; line: " << line); \
            exit(1);                                          \
        }                                                     \
    }

#define CHECK_ROCSPARSE_ERROR(status, file, line)             \
    {                                                         \
        if(status != rocsparse_status_success)                \
        {                                                     \
            LOG_INFO("rocSPARSE error " << status);           \
            if(status == rocsparse_status_invalid_handle)     \
                LOG_INFO("rocsparse_status_invalid_handle");  \
            if(status == rocsparse_status_not_implemented)    \
                LOG_INFO("rocsparse_status_not_implemented"); \
            if(status == rocsparse_status_invalid_pointer)    \
                LOG_INFO("rocsparse_status_invalid_pointer"); \
            if(status == rocsparse_status_invalid_size)       \
                LOG_INFO("rocsparse_status_invalid_size");    \
            if(status == rocsparse_status_memory_error)       \
                LOG_INFO("rocsparse_status_memory_error");    \
            if(status == rocsparse_status_internal_error)     \
                LOG_INFO("rocsparse_status_internal_error");  \
            if(status == rocsparse_status_invalid_value)      \
                LOG_INFO("rocsparse_status_invalid_value");   \
            if(status == rocsparse_status_arch_mismatch)      \
                LOG_INFO("rocsparse_status_arch_mismatch");   \
            LOG_INFO("File: " << file << "; line: " << line); \
            exit(1);                                          \
        }                                                     \
    }

namespace rocalution
{
    static __device__ __forceinline__ float hip_nontemporal_load(const float* ptr)
    {
        return __builtin_nontemporal_load(ptr);
    }
    static __device__ __forceinline__ double hip_nontemporal_load(const double* ptr)
    {
        return __builtin_nontemporal_load(ptr);
    }
    static __device__ __forceinline__ std::complex<float>
                                      hip_nontemporal_load(const std::complex<float>* ptr)
    {
        return std::complex<float>(__builtin_nontemporal_load((const float*)ptr),
                                   __builtin_nontemporal_load((const float*)ptr + 1));
    }
    static __device__ __forceinline__ std::complex<double>
                                      hip_nontemporal_load(const std::complex<double>* ptr)
    {
        return std::complex<double>(__builtin_nontemporal_load((const double*)ptr),
                                    __builtin_nontemporal_load((const double*)ptr + 1));
    }
    static __device__ __forceinline__ int32_t hip_nontemporal_load(const int32_t* ptr)
    {
        return __builtin_nontemporal_load(ptr);
    }
    static __device__ __forceinline__ int64_t hip_nontemporal_load(const int64_t* ptr)
    {
        return __builtin_nontemporal_load(ptr);
    }

    template <typename ValueType>
    static __device__ __forceinline__ bool operator<(const std::complex<ValueType>& lhs,
                                                     const std::complex<ValueType>& rhs)
    {
        return std::real(lhs) < std::real(rhs);
    }

    template <typename ValueType>
    static __device__ __forceinline__ bool operator>(const std::complex<ValueType>& lhs,
                                                     const std::complex<ValueType>& rhs)
    {
        return std::real(lhs) > std::real(rhs);
    }

    template <typename ValueType>
    static __device__ __forceinline__ bool operator<=(const std::complex<ValueType>& lhs,
                                                      const std::complex<ValueType>& rhs)
    {
        return std::real(lhs) <= std::real(rhs);
    }

    template <typename ValueType>
    static __device__ __forceinline__ bool operator>=(const std::complex<ValueType>& lhs,
                                                      const std::complex<ValueType>& rhs)
    {
        return std::real(lhs) >= std::real(rhs);
    }

    // abs()
    static __device__ __forceinline__ float hip_abs(float val)
    {
        return fabsf(val);
    }

    static __device__ __forceinline__ double hip_abs(double val)
    {
        return fabs(val);
    }

#ifdef SUPPORT_COMPLEX
    static __device__ __forceinline__ float hip_abs(std::complex<float> val)
    {
        return sqrtf(val.real() * val.real() + val.imag() * val.imag());
    }

    static __device__ __forceinline__ double hip_abs(std::complex<double> val)
    {
        return sqrt(val.real() * val.real() + val.imag() * val.imag());
    }
#endif

    // real()
    static __device__ __forceinline__ float hip_real(float val)
    {
        return val;
    }

    static __device__ __forceinline__ double hip_real(double val)
    {
        return val;
    }

#ifdef SUPPORT_COMPLEX
    static __device__ __forceinline__ float hip_real(std::complex<float> val)
    {
        return val.real();
    }

    static __device__ __forceinline__ double hip_real(std::complex<double> val)
    {
        return val.real();
    }
#endif

#if ROCALUTION_USE_MOVE_DPP
    template <unsigned int WFSIZE>
    static __device__ __forceinline__ void wf_reduce_sum(int* sum)
    {
        if(WFSIZE > 1)
            *sum += __hip_move_dpp(*sum, 0x111, 0xf, 0xf, 0);
        if(WFSIZE > 2)
            *sum += __hip_move_dpp(*sum, 0x112, 0xf, 0xf, 0);
        if(WFSIZE > 4)
            *sum += __hip_move_dpp(*sum, 0x114, 0xf, 0xe, 0);
        if(WFSIZE > 8)
            *sum += __hip_move_dpp(*sum, 0x118, 0xf, 0xc, 0);
        if(WFSIZE > 16)
            *sum += __hip_move_dpp(*sum, 0x142, 0xa, 0xf, 0);
        if(WFSIZE > 32)
            *sum += __hip_move_dpp(*sum, 0x143, 0xc, 0xf, 0);
    }
#else /* ROCALUTION_USE_MOVE_DPP */
    template <unsigned int WFSIZE>
    static __device__ __forceinline__ void wf_reduce_sum(int* sum)
    {
        for(int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            *sum += __shfl_xor(*sum, i);
        }
    }
#endif /* ROCALUTION_USE_MOVE_DPP */

    template <unsigned int WF_SIZE>
    static __device__ __forceinline__ void wf_reduce_sum(float* sum)
    {
        typedef union flt_b32
        {
            float    val;
            uint32_t b32;
        } flt_b32_t;

        flt_b32_t upper_sum;
        flt_b32_t temp_sum;
        temp_sum.val = *sum;

        if(WF_SIZE > 1)
        {
            upper_sum.b32 = __hip_ds_swizzle(temp_sum.b32, 0x80b1);
            temp_sum.val += upper_sum.val;
        }

        if(WF_SIZE > 2)
        {
            upper_sum.b32 = __hip_ds_swizzle(temp_sum.b32, 0x804e);
            temp_sum.val += upper_sum.val;
        }

        if(WF_SIZE > 4)
        {
            upper_sum.b32 = __hip_ds_swizzle(temp_sum.b32, 0x101f);
            temp_sum.val += upper_sum.val;
        }

        if(WF_SIZE > 8)
        {
            upper_sum.b32 = __hip_ds_swizzle(temp_sum.b32, 0x201f);
            temp_sum.val += upper_sum.val;
        }

        if(WF_SIZE > 16)
        {
            upper_sum.b32 = __hip_ds_swizzle(temp_sum.b32, 0x401f);
            temp_sum.val += upper_sum.val;
        }

        if(WF_SIZE > 32)
        {
            upper_sum.b32 = __builtin_amdgcn_readlane(temp_sum.b32, 32);
            temp_sum.val += upper_sum.val;
        }

        *sum = temp_sum.val;
    }

    template <unsigned int WF_SIZE>
    static __device__ __forceinline__ void wf_reduce_sum(double* sum)
    {
        typedef union dbl_b32
        {
            double   val;
            uint32_t b32[2];
        } dbl_b32_t;

        dbl_b32_t upper_sum;
        dbl_b32_t temp_sum;
        temp_sum.val = *sum;

        if(WF_SIZE > 1)
        {
            upper_sum.b32[0] = __hip_ds_swizzle(temp_sum.b32[0], 0x80b1);
            upper_sum.b32[1] = __hip_ds_swizzle(temp_sum.b32[1], 0x80b1);
            temp_sum.val += upper_sum.val;
        }

        if(WF_SIZE > 2)
        {
            upper_sum.b32[0] = __hip_ds_swizzle(temp_sum.b32[0], 0x804e);
            upper_sum.b32[1] = __hip_ds_swizzle(temp_sum.b32[1], 0x804e);
            temp_sum.val += upper_sum.val;
        }

        if(WF_SIZE > 4)
        {
            upper_sum.b32[0] = __hip_ds_swizzle(temp_sum.b32[0], 0x101f);
            upper_sum.b32[1] = __hip_ds_swizzle(temp_sum.b32[1], 0x101f);
            temp_sum.val += upper_sum.val;
        }

        if(WF_SIZE > 8)
        {
            upper_sum.b32[0] = __hip_ds_swizzle(temp_sum.b32[0], 0x201f);
            upper_sum.b32[1] = __hip_ds_swizzle(temp_sum.b32[1], 0x201f);
            temp_sum.val += upper_sum.val;
        }

        if(WF_SIZE > 16)
        {
            upper_sum.b32[0] = __hip_ds_swizzle(temp_sum.b32[0], 0x401f);
            upper_sum.b32[1] = __hip_ds_swizzle(temp_sum.b32[1], 0x401f);
            temp_sum.val += upper_sum.val;
        }

        if(WF_SIZE > 32)
        {
            upper_sum.b32[0] = __builtin_amdgcn_readlane(temp_sum.b32[0], 32);
            upper_sum.b32[1] = __builtin_amdgcn_readlane(temp_sum.b32[1], 32);
            temp_sum.val += upper_sum.val;
        }

        *sum = temp_sum.val;
    }

#ifdef SUPPORT_COMPLEX
    template <unsigned int WF_SIZE>
    static __device__ __forceinline__ void wf_reduce_sum(hipFloatComplex* sum)
    {
        float real = hipCrealf(*sum);
        float imag = hipCimagf(*sum);

        wf_reduce_sum<WF_SIZE>(&real);
        wf_reduce_sum<WF_SIZE>(&imag);

        *sum = make_hipFloatComplex(real, imag);
    }

    template <unsigned int WF_SIZE>
    static __device__ __forceinline__ void wf_reduce_sum(hipDoubleComplex* sum)
    {
        double real = hipCreal(*sum);
        double imag = hipCimag(*sum);

        wf_reduce_sum<WF_SIZE>(&real);
        wf_reduce_sum<WF_SIZE>(&imag);

        *sum = make_hipDoubleComplex(real, imag);
    }

    template <unsigned int WF_SIZE>
    static __device__ __forceinline__ void wf_reduce_sum(std::complex<float>* sum)
    {
        wf_reduce_sum<WF_SIZE>((hipComplex*)sum);
    }

    template <unsigned int WF_SIZE>
    static __device__ __forceinline__ void wf_reduce_sum(std::complex<double>* sum)
    {
        wf_reduce_sum<WF_SIZE>((hipDoubleComplex*)sum);
    }
#endif

    template <unsigned int WF_SIZE, typename ValueType>
    static __device__ __forceinline__ void wf_reduce_min(ValueType* val)
    {
        for(unsigned int i = WF_SIZE >> 1; i > 0; i >>= 1)
        {
            *val = min(*val, __shfl_xor(*val, i));
        }
    }

#ifdef SUPPORT_COMPLEX
    template <unsigned int WF_SIZE, typename ValueType>
    static __device__ __forceinline__ void wf_reduce_min(std::complex<ValueType>* val)
    {
        ValueType real = std::real(*val);

        wf_reduce_min<WF_SIZE>(&real);

        *val = std::complex<ValueType>(real, static_cast<ValueType>(0));
    }
#endif

    template <unsigned int WF_SIZE, typename ValueType>
    static __device__ __forceinline__ void wf_reduce_max(ValueType* val)
    {
        for(unsigned int i = WF_SIZE >> 1; i > 0; i >>= 1)
        {
            *val = max(*val, __shfl_xor(*val, i));
        }
    }

#ifdef SUPPORT_COMPLEX
    template <unsigned int WF_SIZE, typename ValueType>
    static __device__ __forceinline__ void wf_reduce_max(std::complex<ValueType>* val)
    {
        ValueType real = std::real(*val);

        wf_reduce_max<WF_SIZE>(&real);

        *val = std::complex<ValueType>(real, static_cast<ValueType>(0));
    }
#endif

    static __device__ __forceinline__ float shfl(float var, int src_lane, int width = warpSize)
    {
        return __shfl(var, src_lane, width);
    }
    static __device__ __forceinline__ double shfl(double var, int src_lane, int width = warpSize)
    {
        return __shfl(var, src_lane, width);
    }
#ifdef SUPPORT_COMPLEX
    static __device__ __forceinline__ hipComplex shfl(hipComplex var,
                                                      int        src_lane,
                                                      int        width = warpSize)
    {
        return make_hipFloatComplex(__shfl(hipCrealf(var), src_lane, width),
                                    __shfl(hipCimagf(var), src_lane, width));
    }
    static __device__ __forceinline__ hipDoubleComplex shfl(hipDoubleComplex var,
                                                            int              src_lane,
                                                            int              width = warpSize)
    {
        return make_hipDoubleComplex(__shfl(hipCreal(var), src_lane, width),
                                     __shfl(hipCimag(var), src_lane, width));
    }

    static __device__ __forceinline__ std::complex<float>
        shfl(std::complex<float> var, int src_lane, int width = warpSize)
    {
        return std::complex<float>(__shfl(var.real(), src_lane, width),
                                   __shfl(var.imag(), src_lane, width));
    }
    static __device__ __forceinline__ std::complex<double>
        shfl(std::complex<double> var, int src_lane, int width = warpSize)
    {
        return std::complex<double>(__shfl(var.real(), src_lane, width),
                                    __shfl(var.imag(), src_lane, width));
    }

#endif

    // Block reduce kernel computing block sum
    template <unsigned int BLOCKSIZE, typename ValueType>
    static __device__ __forceinline__ void block_reduce_sum(int i, ValueType* data)
    {
        if(BLOCKSIZE > 512)
        {
            if(i < 512 && i + 512 < BLOCKSIZE)
            {
                data[i] = data[i] + data[i + 512];
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 256)
        {
            if(i < 256 && i + 256 < BLOCKSIZE)
            {
                data[i] = data[i] + data[i + 256];
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 128)
        {
            if(i < 128 && i + 128 < BLOCKSIZE)
            {
                data[i] = data[i] + data[i + 128];
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 64)
        {
            if(i < 64 && i + 64 < BLOCKSIZE)
            {
                data[i] = data[i] + data[i + 64];
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 32)
        {
            if(i < 32 && i + 32 < BLOCKSIZE)
            {
                data[i] = data[i] + data[i + 32];
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 16)
        {
            if(i < 16 && i + 16 < BLOCKSIZE)
            {
                data[i] = data[i] + data[i + 16];
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 8)
        {
            if(i < 8 && i + 8 < BLOCKSIZE)
            {
                data[i] = data[i] + data[i + 8];
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 4)
        {
            if(i < 4 && i + 4 < BLOCKSIZE)
            {
                data[i] = data[i] + data[i + 4];
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 2)
        {
            if(i < 2 && i + 2 < BLOCKSIZE)
            {
                data[i] = data[i] + data[i + 2];
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 1)
        {
            if(i < 1 && i + 1 < BLOCKSIZE)
            {
                data[i] = data[i] + data[i + 1];
            }
            __syncthreads();
        }
    }

    // Block reduce kernel computing blockwide maximum entry
    template <unsigned int BLOCKSIZE, typename T>
    static __device__ __forceinline__ void blockreduce_max(int i, T* data)
    {
        if(BLOCKSIZE > 512)
        {
            if(i < 512 && i + 512 < BLOCKSIZE)
            {
                data[i] = max(data[i], data[i + 512]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 256)
        {
            if(i < 256 && i + 256 < BLOCKSIZE)
            {
                data[i] = max(data[i], data[i + 256]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 128)
        {
            if(i < 128 && i + 128 < BLOCKSIZE)
            {
                data[i] = max(data[i], data[i + 128]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 64)
        {
            if(i < 64 && i + 64 < BLOCKSIZE)
            {
                data[i] = max(data[i], data[i + 64]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 32)
        {
            if(i < 32 && i + 32 < BLOCKSIZE)
            {
                data[i] = max(data[i], data[i + 32]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 16)
        {
            if(i < 16 && i + 16 < BLOCKSIZE)
            {
                data[i] = max(data[i], data[i + 16]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 8)
        {
            if(i < 8 && i + 8 < BLOCKSIZE)
            {
                data[i] = max(data[i], data[i + 8]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 4)
        {
            if(i < 4 && i + 4 < BLOCKSIZE)
            {
                data[i] = max(data[i], data[i + 4]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 2)
        {
            if(i < 2 && i + 2 < BLOCKSIZE)
            {
                data[i] = max(data[i], data[i + 2]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 1)
        {
            if(i < 1 && i + 1 < BLOCKSIZE)
            {
                data[i] = max(data[i], data[i + 1]);
            }
            __syncthreads();
        }
    }

    template <unsigned int BLOCKSIZE, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_find_maximum_blockreduce(int64_t m,
                                             const IndexType* __restrict__ array,
                                             IndexType* __restrict__ workspace)
    {
        unsigned int tid = hipThreadIdx_x;
        int64_t      gid = hipBlockIdx_x * BLOCKSIZE + tid;

        __shared__ IndexType smax[BLOCKSIZE];

        IndexType t_max = -2;

        for(int64_t idx = gid; idx < m; idx += hipGridDim_x * BLOCKSIZE)
        {
            t_max = max(t_max, array[idx]);
        }

        smax[tid] = t_max;

        __syncthreads();

        blockreduce_max<BLOCKSIZE>(tid, smax);

        if(tid == 0)
        {
            workspace[hipBlockIdx_x] = smax[0];
        }
    }

    template <unsigned int BLOCKSIZE, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_find_maximum_finalreduce(IndexType* __restrict__ workspace)
    {
        unsigned int tid = hipThreadIdx_x;

        __shared__ IndexType sdata[BLOCKSIZE];

        sdata[tid] = workspace[tid];

        __syncthreads();

        blockreduce_max<BLOCKSIZE>(tid, sdata);

        if(tid == 0)
        {
            workspace[0] = sdata[0] + 1;
        }
    }

    template <unsigned int BLOCKSIZE, typename ValueType>
    __launch_bounds__(BLOCKSIZE) __global__ void kernel_axpy(int64_t   size,
                                                             ValueType alpha,
                                                             const ValueType* __restrict__ x,
                                                             ValueType* __restrict__ y)
    {
        unsigned int tid = hipThreadIdx_x;
        int64_t      gid = hipBlockIdx_x * BLOCKSIZE + tid;

        if(gid < size)
        {
            y[gid] += alpha * x[gid];
        }
    }

    static __device__ __forceinline__ float hip_shfl_xor(float val, int i)
    {
        return __shfl_xor(val, i);
    }
    static __device__ __forceinline__ double hip_shfl_xor(double val, int i)
    {
        return __shfl_xor(val, i);
    }
#ifdef SUPPORT_COMPLEX
    static __device__ __forceinline__ std::complex<float> hip_shfl_xor(std::complex<float> val,
                                                                       int                 i)
    {
        return std::complex<float>(__shfl_xor(std::real(val), i), __shfl_xor(std::imag(val), i));
    }
    static __device__ __forceinline__ std::complex<double> hip_shfl_xor(std::complex<double> val,
                                                                        int                  i)
    {
        return std::complex<double>(__shfl_xor(std::real(val), i), __shfl_xor(std::imag(val), i));
    }
#endif

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_UTILS_HPP_
