/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "rocsparse-types.h"
#ifdef WIN32
#include <intrin.h>
#endif
#include <hip/hip_runtime.h>

#include "rocsparse_assert.hpp"

#ifndef ROCSPARSE_USE_MOVE_DPP
#if defined(__GFX8__) || defined(__GFX9__)
#define ROCSPARSE_USE_MOVE_DPP 1
#else
#define ROCSPARSE_USE_MOVE_DPP 0
#endif
#endif

// clang-format off
// ELL indexing macros
#define ELL_IND_ROW(i, el, m, width) (el) * (m) + (i)
#define ELL_IND_EL(i, el, m, width) (el) + (width) * (i)
#define ELL_IND(i, el, m, width) ELL_IND_ROW(i, el, m, width)

// BSR indexing macros
#define BSR_IND(j, bi, bj, dir) ((dir == rocsparse_direction_row) ? BSR_IND_R(j, bi, bj) : BSR_IND_C(j, bi, bj))
#define BSR_IND_R(j, bi, bj) (block_dim * block_dim * (j) + (bi) * block_dim + (bj))
#define BSR_IND_C(j, bi, bj) (block_dim * block_dim * (j) + (bi) + (bj) * block_dim)

#define GEBSR_IND(j, bi, bj, dir) ((dir == rocsparse_direction_row) ? GEBSR_IND_R(j, bi, bj) : GEBSR_IND_C(j, bi, bj))
#define GEBSR_IND_R(j, bi, bj) (row_block_dim * col_block_dim * (j) + (bi) * col_block_dim + (bj))
#define GEBSR_IND_C(j, bi, bj) (row_block_dim * col_block_dim * (j) + (bi) + (bj) * row_block_dim)
// clang-format on

namespace rocsparse
{

    // find next power of 2
    __device__ __host__ __forceinline__ uint32_t fnp2(uint32_t x)
    {
        x--;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        x++;

        return x;
    }

    __device__ __forceinline__ _Float16 ldg(const _Float16* ptr)
    {
        return *ptr;
    }
    __device__ __forceinline__ rocsparse_bfloat16 ldg(const rocsparse_bfloat16* ptr)
    {
        return *ptr;
    }
    __device__ __forceinline__ int8_t ldg(const int8_t* ptr)
    {
        return __ldg(ptr);
    }
    __device__ __forceinline__ int32_t ldg(const int32_t* ptr)
    {
        return __ldg(ptr);
    }
    __device__ __forceinline__ int64_t ldg(const int64_t* ptr)
    {
        return __ldg(ptr);
    }
    __device__ __forceinline__ float ldg(const float* ptr)
    {
        return __ldg(ptr);
    }
    __device__ __forceinline__ double ldg(const double* ptr)
    {
        return __ldg(ptr);
    }
    __device__ __forceinline__ rocsparse_float_complex ldg(const rocsparse_float_complex* ptr)
    {
        return rocsparse_float_complex(__ldg((const float*)ptr), __ldg((const float*)ptr + 1));
    }
    __device__ __forceinline__ rocsparse_double_complex ldg(const rocsparse_double_complex* ptr)
    {
        return rocsparse_double_complex(__ldg((const double*)ptr), __ldg((const double*)ptr + 1));
    }

    template <typename T>
    __device__ __forceinline__ T fma(T p, T q, T r);

    __device__ __forceinline__ int32_t shfl_up(int32_t var, int src_lane, int width = warpSize)
    {
        return __shfl_up(var, src_lane, width);
    }
    __device__ __forceinline__ int64_t shfl_up(int64_t var, int src_lane, int width = warpSize)
    {
        return __shfl_up(var, src_lane, width);
    }
    __device__ __forceinline__ float shfl_up(float var, int src_lane, int width = warpSize)
    {
        return __shfl_up(var, src_lane, width);
    }
    __device__ __forceinline__ double shfl_up(double var, int src_lane, int width = warpSize)
    {
        return __shfl_up(var, src_lane, width);
    }
    __device__ __forceinline__ rocsparse_float_complex shfl_up(rocsparse_float_complex var,
                                                               int                     src_lane,
                                                               int width = warpSize)
    {
        return rocsparse_float_complex(__shfl_up(std::real(var), src_lane, width),
                                       __shfl_up(std::imag(var), src_lane, width));
    }
    __device__ __forceinline__ rocsparse_double_complex shfl_up(rocsparse_double_complex var,
                                                                int                      src_lane,
                                                                int width = warpSize)
    {
        return rocsparse_double_complex(__shfl_up(std::real(var), src_lane, width),
                                        __shfl_up(std::imag(var), src_lane, width));
    }

    __device__ __forceinline__ int32_t shfl_down(int32_t var, int src_lane, int width = warpSize)
    {
        return __shfl_down(var, src_lane, width);
    }
    __device__ __forceinline__ int64_t shfl_down(int64_t var, int src_lane, int width = warpSize)
    {
        return __shfl_down(var, src_lane, width);
    }
    __device__ __forceinline__ float shfl_down(float var, int src_lane, int width = warpSize)
    {
        return __shfl_down(var, src_lane, width);
    }
    __device__ __forceinline__ double shfl_down(double var, int src_lane, int width = warpSize)
    {
        return __shfl_down(var, src_lane, width);
    }
    __device__ __forceinline__ rocsparse_float_complex shfl_down(rocsparse_float_complex var,
                                                                 int                     src_lane,
                                                                 int width = warpSize)
    {
        return rocsparse_float_complex(__shfl_down(std::real(var), src_lane, width),
                                       __shfl_down(std::imag(var), src_lane, width));
    }
    __device__ __forceinline__ rocsparse_double_complex shfl_down(rocsparse_double_complex var,
                                                                  int                      src_lane,
                                                                  int width = warpSize)
    {
        return rocsparse_double_complex(__shfl_down(std::real(var), src_lane, width),
                                        __shfl_down(std::imag(var), src_lane, width));
    }

    __device__ __forceinline__ _Float16 read_first_lane(_Float16 var)
    {
        return __builtin_amdgcn_readfirstlane(var);
    }
    __device__ __forceinline__ float read_first_lane(float var)
    {
        return __builtin_amdgcn_readfirstlane(var);
    }
    __device__ __forceinline__ double read_first_lane(double var)
    {
        return __builtin_amdgcn_readfirstlane(var);
    }
    __device__ __forceinline__ int8_t read_first_lane(int8_t var)
    {
        return __builtin_amdgcn_readfirstlane(var);
    }
    __device__ __forceinline__ int32_t read_first_lane(int32_t var)
    {
        return __builtin_amdgcn_readfirstlane(var);
    }
    __device__ __forceinline__ int64_t read_first_lane(int64_t var)
    {
        return __builtin_amdgcn_readfirstlane(var);
    }
    __device__ __forceinline__ uint8_t read_first_lane(uint8_t var)
    {
        return __builtin_amdgcn_readfirstlane(var);
    }
    __device__ __forceinline__ uint32_t read_first_lane(uint32_t var)
    {
        return __builtin_amdgcn_readfirstlane(var);
    }
    __device__ __forceinline__ uint64_t read_first_lane(uint64_t var)
    {
        return __builtin_amdgcn_readfirstlane(var);
    }

    __device__ __forceinline__ int any(int predicate)
    {
        return __any(predicate);
    }
    __device__ __forceinline__ int all(int predicate)
    {
        return __all(predicate);
    }

    template <>
    __device__ __forceinline__ _Float16 fma(_Float16 p, _Float16 q, _Float16 r)
    {
        return std::fma(p, q, r);
    }

    template <>
    __device__ __forceinline__ int32_t fma(int32_t p, int32_t q, int32_t r)
    {
        return p * q + r;
    }

    template <>
    __device__ __forceinline__ int64_t fma(int64_t p, int64_t q, int64_t r)
    {
        return p * q + r;
    }

    template <>
    __device__ __forceinline__ float fma(float p, float q, float r)
    {
        return std::fma(p, q, r);
    }

    template <>
    __device__ __forceinline__ double fma(double p, double q, double r)
    {
        return std::fma(p, q, r);
    }

    template <typename T>
    __device__ __forceinline__ T atomic_add_check(T* ptr, T val)
    {
        if(val != static_cast<T>(0))
            return atomicAdd(ptr, val);
        return *ptr;
    }

    template <>
    __device__ __forceinline__ rocsparse_float_complex fma(rocsparse_float_complex p,
                                                           rocsparse_float_complex q,
                                                           rocsparse_float_complex r)
    {
        return std::fma(p, q, r);
    }

    template <>
    __device__ __forceinline__ rocsparse_double_complex fma(rocsparse_double_complex p,
                                                            rocsparse_double_complex q,
                                                            rocsparse_double_complex r)
    {
        return std::fma(p, q, r);
    }

    __device__ __forceinline__ float abs(float x)
    {
        return x < 0.0f ? -x : x;
    }
    __device__ __forceinline__ double abs(double x)
    {
        return x < 0.0 ? -x : x;
    }
    __device__ __forceinline__ float abs(rocsparse_float_complex x)
    {
        return std::abs(x);
    }
    __device__ __forceinline__ double abs(rocsparse_double_complex x)
    {
        return std::abs(x);
    }

    __device__ __host__ __forceinline__ float ceil(float x)
    {
        return std::ceilf(x);
    }
    __device__ __host__ __forceinline__ double ceil(double x)
    {
        return std::ceil(x);
    }

    __device__ __host__ __forceinline__ float max(float x, float y)
    {
        return std::max(x, y);
    }
    __device__ __host__ __forceinline__ double max(double x, double y)
    {
        return std::max(x, y);
    }
    __device__ __host__ __forceinline__ int32_t max(int32_t x, int32_t y)
    {
        return std::max(x, y);
    }
    __device__ __host__ __forceinline__ int64_t max(int64_t x, int64_t y)
    {
        return std::max(x, y);
    }
    __device__ __host__ __forceinline__ uint32_t max(uint32_t x, uint32_t y)
    {
        return std::max(x, y);
    }
    __device__ __host__ __forceinline__ uint64_t max(uint64_t x, uint64_t y)
    {
        return std::max(x, y);
    }

    __device__ __host__ __forceinline__ float min(float x, float y)
    {
        return std::min(x, y);
    }
    __device__ __host__ __forceinline__ double min(double x, double y)
    {
        return std::min(x, y);
    }
    __device__ __host__ __forceinline__ int32_t min(int32_t x, int32_t y)
    {
        return std::min(x, y);
    }
    __device__ __host__ __forceinline__ int64_t min(int64_t x, int64_t y)
    {
        return std::min(x, y);
    }
    __device__ __host__ __forceinline__ uint32_t min(uint32_t x, uint32_t y)
    {
        return std::min(x, y);
    }
    __device__ __host__ __forceinline__ uint64_t min(uint64_t x, uint64_t y)
    {
        return std::min(x, y);
    }

    __device__ __forceinline__ float sqrt(float val)
    {
        return std::sqrt(val);
    }
    __device__ __forceinline__ double sqrt(double val)
    {
        return std::sqrt(val);
    }

    __device__ __host__ __forceinline__ float log2(float val)
    {
        return std::log2f(val);
    }
    __device__ __host__ __forceinline__ double log2(double val)
    {
        return std::log2(val);
    }
    __device__ __host__ __forceinline__ float log2(int32_t val)
    {
        return std::log2f(val);
    }
    __device__ __host__ __forceinline__ double log2(int64_t val)
    {
        return std::log2(val);
    }
    __device__ __host__ __forceinline__ float log2(uint32_t val)
    {
        return std::log2f(val);
    }
    __device__ __host__ __forceinline__ double log2(uint64_t val)
    {
        return std::log2(val);
    }

    template <>
    __device__ __forceinline__ int64_t atomic_add_check<int64_t>(int64_t* ptr, int64_t val)
    {
        if(val != static_cast<int64_t>(0))
            return atomicAdd((uint64_t*)ptr, val);
        return *ptr;
    }

    template <>
    __device__ __forceinline__ rocsparse_float_complex
        atomic_add_check(rocsparse_float_complex* ptr, rocsparse_float_complex val)
    {
        if(std::real(val) != static_cast<float>(0) || std::imag(val) != static_cast<float>(0))
            return rocsparse_float_complex(atomicAdd((float*)ptr, std::real(val)),
                                           atomicAdd((float*)ptr + 1, std::imag(val)));
        return *ptr;
    }

    template <>
    __device__ __forceinline__ rocsparse_double_complex
        atomic_add_check(rocsparse_double_complex* ptr, rocsparse_double_complex val)
    {
        if(std::real(val) != static_cast<double>(0) || std::imag(val) != static_cast<double>(0))
            return rocsparse_double_complex(atomicAdd((double*)ptr, std::real(val)),
                                            atomicAdd((double*)ptr + 1, std::imag(val)));
        return *ptr;
    }

    __device__ __forceinline__ rocsparse_float_complex sqrt(rocsparse_float_complex val)
    {
        float x = std::real(val);
        float y = std::imag(val);

        float sgnp = (y < 0.0f) ? -1.0f : 1.0f;
        float absz = rocsparse::abs(val);

        return rocsparse_float_complex(std::sqrt((absz + x) * 0.5f),
                                       sgnp * std::sqrt((absz - x) * 0.5f));
    }
    __device__ __forceinline__ rocsparse_double_complex sqrt(rocsparse_double_complex val)
    {
        double x = std::real(val);
        double y = std::imag(val);

        double sgnp = (y < 0.0) ? -1.0 : 1.0;
        double absz = rocsparse::abs(val);

        return rocsparse_double_complex(std::sqrt((absz + x) * 0.5),
                                        sgnp * std::sqrt((absz - x) * 0.5));
    }

    __device__ __forceinline__ _Float16 conj(const _Float16& x)
    {
        return x;
    }
    __device__ __forceinline__ rocsparse_bfloat16 conj(const rocsparse_bfloat16& x)
    {
        return x;
    }
    __device__ __forceinline__ int32_t conj(const int32_t& x)
    {
        return x;
    }
    __device__ __forceinline__ float conj(const float& x)
    {
        return x;
    }
    __device__ __forceinline__ double conj(const double& x)
    {
        return x;
    }
    __device__ __forceinline__ rocsparse_float_complex conj(const rocsparse_float_complex& x)
    {
        return std::conj(x);
    }
    __device__ __forceinline__ rocsparse_double_complex conj(const rocsparse_double_complex& x)
    {
        return std::conj(x);
    }

    __device__ __forceinline__ float real(const float& x)
    {
        return x;
    }
    __device__ __forceinline__ double real(const double& x)
    {
        return x;
    }
    __device__ __forceinline__ float real(const rocsparse_float_complex& x)
    {
        return std::real(x);
    }
    __device__ __forceinline__ double real(const rocsparse_double_complex& x)
    {
        return std::real(x);
    }

    __device__ __forceinline__ float imag(const float& x)
    {
        return static_cast<float>(0);
    }
    __device__ __forceinline__ double imag(const double& x)
    {
        return static_cast<double>(0);
    }
    __device__ __forceinline__ float imag(const rocsparse_float_complex& x)
    {
        return std::imag(x);
    }
    __device__ __forceinline__ double imag(const rocsparse_double_complex& x)
    {
        return std::imag(x);
    }

    __device__ __forceinline__ bool gt(const float& x, const float& y)
    {
        return x > y;
    }
    __device__ __forceinline__ bool gt(const double& x, const double& y)
    {
        return x > y;
    }
    __device__ __forceinline__ bool gt(const rocsparse_float_complex& x,
                                       const rocsparse_float_complex& y)
    {
        if(&x == &y)
        {
            return false;
        }

        return std::real(x) > std::real(y);
    }

    __device__ __forceinline__ bool gt(const rocsparse_double_complex& x,
                                       const rocsparse_double_complex& y)
    {
        if(&x == &y)
        {
            return false;
        }

        return std::real(x) > std::real(y);
    }

    __device__ __forceinline__ _Float16 nontemporal_load(const _Float16* ptr)
    {
        return *ptr;
    }
    __device__ __forceinline__ rocsparse_bfloat16 nontemporal_load(const rocsparse_bfloat16* ptr)
    {
        return *ptr;
    }
    __device__ __forceinline__ float nontemporal_load(const float* ptr)
    {
        return __builtin_nontemporal_load(ptr);
    }
    __device__ __forceinline__ double nontemporal_load(const double* ptr)
    {
        return __builtin_nontemporal_load(ptr);
    }
    __device__ __forceinline__ rocsparse_float_complex
        nontemporal_load(const rocsparse_float_complex* ptr)
    {
        return rocsparse_float_complex(__builtin_nontemporal_load((const float*)ptr),
                                       __builtin_nontemporal_load((const float*)ptr + 1));
    }
    __device__ __forceinline__ rocsparse_double_complex
        nontemporal_load(const rocsparse_double_complex* ptr)
    {
        return rocsparse_double_complex(__builtin_nontemporal_load((const double*)ptr),
                                        __builtin_nontemporal_load((const double*)ptr + 1));
    }
    __device__ __forceinline__ int8_t nontemporal_load(const int8_t* ptr)
    {
        return __builtin_nontemporal_load(ptr);
    }
    __device__ __forceinline__ int32_t nontemporal_load(const int32_t* ptr)
    {
        return __builtin_nontemporal_load(ptr);
    }
    __device__ __forceinline__ int64_t nontemporal_load(const int64_t* ptr)
    {
        return __builtin_nontemporal_load(ptr);
    }

    __device__ __forceinline__ void nontemporal_store(float val, float* ptr)
    {
        __builtin_nontemporal_store(val, ptr);
    }
    __device__ __forceinline__ void nontemporal_store(double val, double* ptr)
    {
        __builtin_nontemporal_store(val, ptr);
    }
    __device__ __forceinline__ void nontemporal_store(rocsparse_float_complex  val,
                                                      rocsparse_float_complex* ptr)
    {
        __builtin_nontemporal_store(std::real(val), (float*)ptr);
        __builtin_nontemporal_store(std::imag(val), (float*)ptr + 1);
    }
    __device__ __forceinline__ void nontemporal_store(rocsparse_double_complex  val,
                                                      rocsparse_double_complex* ptr)
    {
        __builtin_nontemporal_store(std::real(val), (double*)ptr);
        __builtin_nontemporal_store(std::imag(val), (double*)ptr + 1);
    }
    __device__ __forceinline__ void nontemporal_store(int8_t val, int8_t* ptr)
    {
        __builtin_nontemporal_store(val, ptr);
    }
    __device__ __forceinline__ void nontemporal_store(int32_t val, int32_t* ptr)
    {
        __builtin_nontemporal_store(val, ptr);
    }
    __device__ __forceinline__ void nontemporal_store(int64_t val, int64_t* ptr)
    {
        __builtin_nontemporal_store(val, ptr);
    }

    __device__ __forceinline__ int32_t shfl(int32_t var, int src_lane, int width = warpSize)
    {
        return __shfl(var, src_lane, width);
    }
    __device__ __forceinline__ int64_t shfl(int64_t var, int src_lane, int width = warpSize)
    {
        return __shfl(var, src_lane, width);
    }
    __device__ __forceinline__ float shfl(float var, int src_lane, int width = warpSize)
    {
        return __shfl(var, src_lane, width);
    }
    __device__ __forceinline__ double shfl(double var, int src_lane, int width = warpSize)
    {
        return __shfl(var, src_lane, width);
    }
    __device__ __forceinline__ rocsparse_float_complex shfl(rocsparse_float_complex var,
                                                            int                     src_lane,
                                                            int width = warpSize)
    {
        return rocsparse_float_complex(__shfl(std::real(var), src_lane, width),
                                       __shfl(std::imag(var), src_lane, width));
    }
    __device__ __forceinline__ rocsparse_double_complex shfl(rocsparse_double_complex var,
                                                             int                      src_lane,
                                                             int width = warpSize)
    {
        return rocsparse_double_complex(__shfl(std::real(var), src_lane, width),
                                        __shfl(std::imag(var), src_lane, width));
    }

    // Count the number of bits that are set to 1 up to and including bit `lid`
    // For example, if mask = 10010111011010001111001000010101 then
    // popc<32>(mask, 0) = 1
    // popc<32>(mask, 1) = 1
    // popc<32>(mask, 2) = 1
    // popc<32>(mask, 3) = 2
    // popc<32>(mask, 4) = 2
    // popc<32>(mask, 5) = 3
    // popc<32>(mask, 6) = 4
    // popc<32>(mask, 7) = 5
    // ...
    // popc<32>(mask, 31) = 16
    // In other words this performs an inclusive scan on the bits and returns the
    // sum up to and including bit `lid`
    template <uint32_t WFSIZE>
    __device__ __forceinline__ uint32_t popc(uint64_t mask, uint32_t lid)
    {
        if(WFSIZE == 32)
        {
            return __popc(mask & (0xffffffff >> (WFSIZE - 1 - lid)));
        }
        else
        {
            return __popcll(mask & (0xffffffffffffffff >> (WFSIZE - 1 - lid)));
        }
    }

    template <typename T>
    __device__ __forceinline__ T atomic_min(T* ptr, T val)
    {
        return atomicMin(ptr, val);
    }

    template <typename T>
    __device__ __forceinline__ T atomic_max(T* ptr, T val)
    {
        return atomicMax(ptr, val);
    }

    template <typename T>
    __device__ __forceinline__ T atomic_add(T* ptr, T val)
    {
        return atomicAdd(ptr, val);
    }

    template <typename T>
    __device__ __forceinline__ T atomic_cas(T* ptr, T cmp, T val)
    {
        return atomicCAS(ptr, cmp, val);
    }

    template <>
    __device__ __forceinline__ int64_t atomic_min<int64_t>(int64_t* ptr, int64_t val)
    {
        return atomicMin((uint64_t*)ptr, (uint64_t)val);
    }

    template <>
    __device__ __forceinline__ int64_t atomic_max<int64_t>(int64_t* ptr, int64_t val)
    {
        return atomicMax((uint64_t*)ptr, val);
    }

    template <>
    __device__ __forceinline__ int64_t atomic_add<int64_t>(int64_t* ptr, int64_t val)
    {
        return atomicAdd((uint64_t*)ptr, val);
    }

    template <>
    __device__ __forceinline__ rocsparse_float_complex atomic_add(rocsparse_float_complex* ptr,
                                                                  rocsparse_float_complex  val)
    {
        return rocsparse_float_complex(atomicAdd((float*)ptr, std::real(val)),
                                       atomicAdd((float*)ptr + 1, std::imag(val)));
    }

    template <>
    __device__ __forceinline__ rocsparse_double_complex atomic_add(rocsparse_double_complex* ptr,
                                                                   rocsparse_double_complex  val)
    {
        return rocsparse_double_complex(atomicAdd((double*)ptr, std::real(val)),
                                        atomicAdd((double*)ptr + 1, std::imag(val)));
    }

    template <>
    __device__ __forceinline__ int64_t atomic_cas(int64_t* ptr, int64_t cmp, int64_t val)
    {
        return atomicCAS((uint64_t*)ptr, cmp, val);
    }

    __device__ __forceinline__ bool is_inf(float val)
    {
        return (val == std::numeric_limits<float>::infinity());
    }
    __device__ __forceinline__ bool is_inf(double val)
    {
        return (val == std::numeric_limits<double>::infinity());
    }
    __device__ __forceinline__ bool is_inf(rocsparse_float_complex val)
    {
        return (std::real(val) == std::numeric_limits<float>::infinity()
                || std::imag(val) == std::numeric_limits<float>::infinity());
    }
    __device__ __forceinline__ bool is_inf(rocsparse_double_complex val)
    {
        return (std::real(val) == std::numeric_limits<double>::infinity()
                || std::imag(val) == std::numeric_limits<double>::infinity());
    }

    __device__ __forceinline__ bool is_nan(float val)
    {
        return (val != val);
    }
    __device__ __forceinline__ bool is_nan(double val)
    {
        return (val != val);
    }
    __device__ __forceinline__ bool is_nan(rocsparse_float_complex val)
    {
        return (std::real(val) != std::real(val) || std::imag(val) != std::imag(val));
    }
    __device__ __forceinline__ bool is_nan(rocsparse_double_complex val)
    {
        return (std::real(val) != std::real(val) || std::imag(val) != std::imag(val));
    }

    template <typename T, typename I, typename J>
    __device__ __forceinline__ T* load_pointer(T* p, J batch, I stride)
    {
        return p + stride * batch;
    }

    template <typename T, typename I, typename J>
    __device__ __forceinline__ const T* load_pointer(const T* p, J batch, I stride)
    {
        return p + stride * batch;
    }

    template <typename T>
    __device__ __forceinline__ T conj_val(T val, bool conj)
    {
        return conj ? rocsparse::conj(val) : val;
    }

    // Block reduce kernel computing block sum
    template <uint32_t BLOCKSIZE, typename T>
    __device__ __forceinline__ void blockreduce_sum(int i, T* data)
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
    template <uint32_t BLOCKSIZE, typename T>
    __device__ __forceinline__ void blockreduce_max(int i, T* data)
    {
        if(BLOCKSIZE > 512)
        {
            if(i < 512 && i + 512 < BLOCKSIZE)
            {
                data[i] = rocsparse::max(data[i], data[i + 512]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 256)
        {
            if(i < 256 && i + 256 < BLOCKSIZE)
            {
                data[i] = rocsparse::max(data[i], data[i + 256]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 128)
        {
            if(i < 128 && i + 128 < BLOCKSIZE)
            {
                data[i] = rocsparse::max(data[i], data[i + 128]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 64)
        {
            if(i < 64 && i + 64 < BLOCKSIZE)
            {
                data[i] = rocsparse::max(data[i], data[i + 64]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 32)
        {
            if(i < 32 && i + 32 < BLOCKSIZE)
            {
                data[i] = rocsparse::max(data[i], data[i + 32]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 16)
        {
            if(i < 16 && i + 16 < BLOCKSIZE)
            {
                data[i] = rocsparse::max(data[i], data[i + 16]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 8)
        {
            if(i < 8 && i + 8 < BLOCKSIZE)
            {
                data[i] = rocsparse::max(data[i], data[i + 8]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 4)
        {
            if(i < 4 && i + 4 < BLOCKSIZE)
            {
                data[i] = rocsparse::max(data[i], data[i + 4]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 2)
        {
            if(i < 2 && i + 2 < BLOCKSIZE)
            {
                data[i] = rocsparse::max(data[i], data[i + 2]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 1)
        {
            if(i < 1 && i + 1 < BLOCKSIZE)
            {
                data[i] = rocsparse::max(data[i], data[i + 1]);
            }
            __syncthreads();
        }
    }

    // Block reduce kernel computing blockwide minimum entry
    template <uint32_t BLOCKSIZE, typename T>
    __device__ __forceinline__ void blockreduce_min(int i, T* data)
    {
        if(BLOCKSIZE > 512)
        {
            if(i < 512 && i + 512 < BLOCKSIZE)
            {
                data[i] = rocsparse::min(data[i], data[i + 512]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 256)
        {
            if(i < 256 && i + 256 < BLOCKSIZE)
            {
                data[i] = rocsparse::min(data[i], data[i + 256]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 128)
        {
            if(i < 128 && i + 128 < BLOCKSIZE)
            {
                data[i] = rocsparse::min(data[i], data[i + 128]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 64)
        {
            if(i < 64 && i + 64 < BLOCKSIZE)
            {
                data[i] = rocsparse::min(data[i], data[i + 64]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 32)
        {
            if(i < 32 && i + 32 < BLOCKSIZE)
            {
                data[i] = rocsparse::min(data[i], data[i + 32]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 16)
        {
            if(i < 16 && i + 16 < BLOCKSIZE)
            {
                data[i] = rocsparse::min(data[i], data[i + 16]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 8)
        {
            if(i < 8 && i + 8 < BLOCKSIZE)
            {
                data[i] = rocsparse::min(data[i], data[i + 8]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 4)
        {
            if(i < 4 && i + 4 < BLOCKSIZE)
            {
                data[i] = rocsparse::min(data[i], data[i + 4]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 2)
        {
            if(i < 2 && i + 2 < BLOCKSIZE)
            {
                data[i] = rocsparse::min(data[i], data[i + 2]);
            }
            __syncthreads();
        }
        if(BLOCKSIZE > 1)
        {
            if(i < 1 && i + 1 < BLOCKSIZE)
            {
                data[i] = rocsparse::min(data[i], data[i + 1]);
            }
            __syncthreads();
        }
    }

#if ROCSPARSE_USE_MOVE_DPP

    template <uint32_t WFSIZE>
    static __device__ __forceinline__ void wfreduce_max(float* maximum)
    {
        typedef union flt_b32
        {
            float    val;
            uint32_t b32[1];
        } flt_b32_t;

        flt_b32_t upper_max;
        flt_b32_t temp_max;
        temp_max.val = *maximum;

        if(WFSIZE > 1)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x111, 0xf, 0xf, false);
            temp_max.val     = rocsparse::max(temp_max.val, upper_max.val);
        }

        if(WFSIZE > 2)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x112, 0xf, 0xf, false);
            temp_max.val     = rocsparse::max(temp_max.val, upper_max.val);
        }

        if(WFSIZE > 4)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x114, 0xf, 0xe, false);
            temp_max.val     = rocsparse::max(temp_max.val, upper_max.val);
        }

        if(WFSIZE > 8)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x118, 0xf, 0xc, false);
            temp_max.val     = rocsparse::max(temp_max.val, upper_max.val);
        }

        if(WFSIZE > 16)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x142, 0xa, 0xf, false);
            temp_max.val     = rocsparse::max(temp_max.val, upper_max.val);
        }

        if(WFSIZE > 32)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x143, 0xc, 0xf, false);
            temp_max.val     = rocsparse::max(temp_max.val, upper_max.val);
        }

        *maximum = temp_max.val;
    }

    // DPP-based wavefront reduction maximum
    template <uint32_t WFSIZE>
    static __device__ __forceinline__ void wfreduce_max(double* maximum)
    {
        typedef union i64_b32
        {
            double   i64;
            uint32_t b32[2];
        } i64_b32_t;

        i64_b32_t upper_max;
        i64_b32_t temp_max;
        temp_max.i64 = *maximum;

        if(WFSIZE > 1)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x111, 0xf, 0xf, false);
            upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x111, 0xf, 0xf, false);
            temp_max.i64     = rocsparse::max(temp_max.i64, upper_max.i64);
        }

        if(WFSIZE > 2)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x112, 0xf, 0xf, false);
            upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x112, 0xf, 0xf, false);
            temp_max.i64     = rocsparse::max(temp_max.i64, upper_max.i64);
        }

        if(WFSIZE > 4)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x114, 0xf, 0xe, false);
            upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x114, 0xf, 0xe, false);
            temp_max.i64     = rocsparse::max(temp_max.i64, upper_max.i64);
        }

        if(WFSIZE > 8)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x118, 0xf, 0xc, false);
            upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x118, 0xf, 0xc, false);
            temp_max.i64     = rocsparse::max(temp_max.i64, upper_max.i64);
        }

        if(WFSIZE > 16)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x142, 0xa, 0xf, false);
            upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x142, 0xa, 0xf, false);
            temp_max.i64     = rocsparse::max(temp_max.i64, upper_max.i64);
        }

        if(WFSIZE > 32)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x143, 0xc, 0xf, false);
            upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x143, 0xc, 0xf, false);
            temp_max.i64     = rocsparse::max(temp_max.i64, upper_max.i64);
        }

        *maximum = temp_max.i64;
    }

    // DPP-based wavefront reduction maximum
    template <uint32_t WFSIZE>
    __device__ __forceinline__ void wfreduce_max(int* maximum)
    {
        if(WFSIZE > 1)
            *maximum = rocsparse::max(*maximum, __hip_move_dpp(*maximum, 0x111, 0xf, 0xf, 0));
        if(WFSIZE > 2)
            *maximum = rocsparse::max(*maximum, __hip_move_dpp(*maximum, 0x112, 0xf, 0xf, 0));
        if(WFSIZE > 4)
            *maximum = rocsparse::max(*maximum, __hip_move_dpp(*maximum, 0x114, 0xf, 0xe, 0));
        if(WFSIZE > 8)
            *maximum = rocsparse::max(*maximum, __hip_move_dpp(*maximum, 0x118, 0xf, 0xc, 0));
        if(WFSIZE > 16)
            *maximum = rocsparse::max(*maximum, __hip_move_dpp(*maximum, 0x142, 0xa, 0xf, 0));
        if(WFSIZE > 32)
            *maximum = rocsparse::max(*maximum, __hip_move_dpp(*maximum, 0x143, 0xc, 0xf, 0));
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ void wfreduce_max(int64_t* maximum)
    {
        typedef union i64_b32
        {
            int64_t  i64;
            uint32_t b32[2];
        } i64_b32_t;

        i64_b32_t upper_max;
        i64_b32_t temp_max;
        temp_max.i64 = *maximum;

        if(WFSIZE > 1)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x111, 0xf, 0xf, false);
            upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x111, 0xf, 0xf, false);
            temp_max.i64     = max(temp_max.i64, upper_max.i64);
        }

        if(WFSIZE > 2)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x112, 0xf, 0xf, false);
            upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x112, 0xf, 0xf, false);
            temp_max.i64     = max(temp_max.i64, upper_max.i64);
        }

        if(WFSIZE > 4)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x114, 0xf, 0xe, false);
            upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x114, 0xf, 0xe, false);
            temp_max.i64     = max(temp_max.i64, upper_max.i64);
        }

        if(WFSIZE > 8)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x118, 0xf, 0xc, false);
            upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x118, 0xf, 0xc, false);
            temp_max.i64     = max(temp_max.i64, upper_max.i64);
        }

        if(WFSIZE > 16)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x142, 0xa, 0xf, false);
            upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x142, 0xa, 0xf, false);
            temp_max.i64     = max(temp_max.i64, upper_max.i64);
        }

        if(WFSIZE > 32)
        {
            upper_max.b32[0] = __hip_move_dpp(temp_max.b32[0], 0x143, 0xc, 0xf, false);
            upper_max.b32[1] = __hip_move_dpp(temp_max.b32[1], 0x143, 0xc, 0xf, false);
            temp_max.i64     = max(temp_max.i64, upper_max.i64);
        }

        *maximum = temp_max.i64;
    }

    // DPP-based wavefront reduction minimum
    template <uint32_t WFSIZE>
    __device__ __forceinline__ void wfreduce_min(int* minimum)
    {
        if(WFSIZE > 1)
            *minimum = rocsparse::min(*minimum, __hip_move_dpp(*minimum, 0x111, 0xf, 0xf, 0));
        if(WFSIZE > 2)
            *minimum = rocsparse::min(*minimum, __hip_move_dpp(*minimum, 0x112, 0xf, 0xf, 0));
        if(WFSIZE > 4)
            *minimum = rocsparse::min(*minimum, __hip_move_dpp(*minimum, 0x114, 0xf, 0xe, 0));
        if(WFSIZE > 8)
            *minimum = rocsparse::min(*minimum, __hip_move_dpp(*minimum, 0x118, 0xf, 0xc, 0));
        if(WFSIZE > 16)
            *minimum = rocsparse::min(*minimum, __hip_move_dpp(*minimum, 0x142, 0xa, 0xf, 0));
        if(WFSIZE > 32)
            *minimum = rocsparse::min(*minimum, __hip_move_dpp(*minimum, 0x143, 0xc, 0xf, 0));
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ void wfreduce_min(int64_t* minimum)
    {
        typedef union i64_b32
        {
            int64_t  i64;
            uint32_t b32[2];
        } i64_b32_t;

        i64_b32_t upper_min;
        i64_b32_t temp_min;
        temp_min.i64 = *minimum;

        if(WFSIZE > 1)
        {
            upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x111, 0xf, 0xf, false);
            upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x111, 0xf, 0xf, false);
            temp_min.i64     = rocsparse::min(temp_min.i64, upper_min.i64);
        }

        if(WFSIZE > 2)
        {
            upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x112, 0xf, 0xf, false);
            upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x112, 0xf, 0xf, false);
            temp_min.i64     = rocsparse::min(temp_min.i64, upper_min.i64);
        }

        if(WFSIZE > 4)
        {
            upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x114, 0xf, 0xe, false);
            upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x114, 0xf, 0xe, false);
            temp_min.i64     = rocsparse::min(temp_min.i64, upper_min.i64);
        }

        if(WFSIZE > 8)
        {
            upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x118, 0xf, 0xc, false);
            upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x118, 0xf, 0xc, false);
            temp_min.i64     = rocsparse::min(temp_min.i64, upper_min.i64);
        }

        if(WFSIZE > 16)
        {
            upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x142, 0xa, 0xf, false);
            upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x142, 0xa, 0xf, false);
            temp_min.i64     = rocsparse::min(temp_min.i64, upper_min.i64);
        }

        if(WFSIZE > 32)
        {
            upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x143, 0xc, 0xf, false);
            upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x143, 0xc, 0xf, false);
            temp_min.i64     = rocsparse::min(temp_min.i64, upper_min.i64);
        }

        *minimum = temp_min.i64;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ int32_t wfreduce_sum(int32_t sum)
    {
        if(WFSIZE > 1)
            sum += __hip_move_dpp(sum, 0x111, 0xf, 0xf, 0);
        if(WFSIZE > 2)
            sum += __hip_move_dpp(sum, 0x112, 0xf, 0xf, 0);
        if(WFSIZE > 4)
            sum += __hip_move_dpp(sum, 0x114, 0xf, 0xe, 0);
        if(WFSIZE > 8)
            sum += __hip_move_dpp(sum, 0x118, 0xf, 0xc, 0);
        if(WFSIZE > 16)
            sum += __hip_move_dpp(sum, 0x142, 0xa, 0xf, 0);
        if(WFSIZE > 32)
            sum += __hip_move_dpp(sum, 0x143, 0xc, 0xf, 0);

        return sum;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ int64_t wfreduce_sum(int64_t sum)
    {
        typedef union i64_b32
        {
            int64_t  i64;
            uint32_t b32[2];
        } i64_b32_t;

        i64_b32_t upper_sum;
        i64_b32_t temp_sum;
        temp_sum.i64 = sum;

        if(WFSIZE > 1)
        {
            upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x111, 0xf, 0xf, false);
            upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x111, 0xf, 0xf, false);
            temp_sum.i64 += upper_sum.i64;
        }

        if(WFSIZE > 2)
        {
            upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x112, 0xf, 0xf, false);
            upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x112, 0xf, 0xf, false);
            temp_sum.i64 += upper_sum.i64;
        }

        if(WFSIZE > 4)
        {
            upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
            upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
            temp_sum.i64 += upper_sum.i64;
        }

        if(WFSIZE > 8)
        {
            upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
            upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
            temp_sum.i64 += upper_sum.i64;
        }

        if(WFSIZE > 16)
        {
            upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x142, 0xa, 0xf, false);
            upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x142, 0xa, 0xf, false);
            temp_sum.i64 += upper_sum.i64;
        }

        if(WFSIZE > 32)
        {
            upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x143, 0xc, 0xf, false);
            upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x143, 0xc, 0xf, false);
            temp_sum.i64 += upper_sum.i64;
        }

        sum = temp_sum.i64;
        return sum;
    }

    // DPP-based float wavefront reduction sum
    template <uint32_t WFSIZE>
    __device__ __forceinline__ float wfreduce_sum(float sum)
    {
        typedef union flt_b32
        {
            float    val;
            uint32_t b32;
        } flt_b32_t;

        flt_b32_t upper_sum;
        flt_b32_t temp_sum;
        temp_sum.val = sum;

        if(WFSIZE > 1)
        {
            upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x111, 0xf, 0xf, false);
            temp_sum.val += upper_sum.val;
        }

        if(WFSIZE > 2)
        {
            upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x112, 0xf, 0xf, false);
            temp_sum.val += upper_sum.val;
        }

        if(WFSIZE > 4)
        {
            upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x114, 0xf, 0xe, false);
            temp_sum.val += upper_sum.val;
        }

        if(WFSIZE > 8)
        {
            upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x118, 0xf, 0xc, false);
            temp_sum.val += upper_sum.val;
        }

        if(WFSIZE > 16)
        {
            upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x142, 0xa, 0xf, false);
            temp_sum.val += upper_sum.val;
        }

        if(WFSIZE > 32)
        {
            upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x143, 0xc, 0xf, false);
            temp_sum.val += upper_sum.val;
        }

        sum = temp_sum.val;
        return sum;
    }

    // DPP-based double wavefront reduction
    template <uint32_t WFSIZE>
    __device__ __forceinline__ double wfreduce_sum(double sum)
    {
        typedef union dbl_b32
        {
            double   val;
            uint32_t b32[2];
        } dbl_b32_t;

        dbl_b32_t upper_sum;
        dbl_b32_t temp_sum;
        temp_sum.val = sum;

        if(WFSIZE > 1)
        {
            upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x111, 0xf, 0xf, false);
            upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x111, 0xf, 0xf, false);
            temp_sum.val += upper_sum.val;
        }

        if(WFSIZE > 2)
        {
            upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x112, 0xf, 0xf, false);
            upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x112, 0xf, 0xf, false);
            temp_sum.val += upper_sum.val;
        }

        if(WFSIZE > 4)
        {
            upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
            upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
            temp_sum.val += upper_sum.val;
        }

        if(WFSIZE > 8)
        {
            upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
            upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
            temp_sum.val += upper_sum.val;
        }

        if(WFSIZE > 16)
        {
            upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x142, 0xa, 0xf, false);
            upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x142, 0xa, 0xf, false);
            temp_sum.val += upper_sum.val;
        }

        if(WFSIZE > 32)
        {
            upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x143, 0xc, 0xf, false);
            upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x143, 0xc, 0xf, false);
            temp_sum.val += upper_sum.val;
        }

        sum = temp_sum.val;
        return sum;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ _Float16 wfreduce_sum(_Float16 sum)
    {
        return wfreduce_sum<WFSIZE>((float)sum);
    }

    // DPP-based double wavefront partial reduction
    template <uint32_t WFSIZE, uint32_t SUB_WFSIZE>
    __device__ __forceinline__ int32_t wfreduce_partial_sum(int32_t sum)
    {
        if(SUB_WFSIZE == 1)
        {
            return wfreduce_sum<WFSIZE>(sum);
        }
        if(SUB_WFSIZE == 2)
        {
            if(WFSIZE > 2)
                sum += __hip_move_dpp(sum, 0x112, 0xf, 0xf, 0);
            if(WFSIZE > 4)
                sum += __hip_move_dpp(sum, 0x114, 0xf, 0xe, 0);
            if(WFSIZE > 8)
                sum += __hip_move_dpp(sum, 0x118, 0xf, 0xc, 0);
            if(WFSIZE > 16)
                sum += __shfl_xor(sum, 16);
            if(WFSIZE > 32)
                sum += __shfl_xor(sum, 32);
            return sum;
        }
        if(SUB_WFSIZE == 4)
        {
            if(WFSIZE > 4)
                sum += __hip_move_dpp(sum, 0x114, 0xf, 0xe, 0);
            if(WFSIZE > 8)
                sum += __hip_move_dpp(sum, 0x118, 0xf, 0xc, 0);
            if(WFSIZE > 16)
                sum += __shfl_xor(sum, 16);
            if(WFSIZE > 32)
                sum += __shfl_xor(sum, 32);
            return sum;
        }
        if(SUB_WFSIZE == 8)
        {
            if(WFSIZE > 8)
                sum += __hip_move_dpp(sum, 0x118, 0xf, 0xc, 0);
            if(WFSIZE > 16)
                sum += __shfl_xor(sum, 16);
            if(WFSIZE > 32)
                sum += __shfl_xor(sum, 32);
            return sum;
        }
        if(SUB_WFSIZE == 16)
        {
            if(WFSIZE > 16)
                sum += __shfl_xor(sum, 16);
            if(WFSIZE > 32)
                sum += __shfl_xor(sum, 32);
            return sum;
        }
        if(SUB_WFSIZE == 32)
        {
            if(WFSIZE > 32)
                sum += __shfl_xor(sum, 32);
            return sum;
        }
        return sum;
    }

    // DPP-based double wavefront partial reduction
    template <uint32_t WFSIZE, uint32_t SUB_WFSIZE>
    __device__ __forceinline__ int64_t wfreduce_partial_sum(int64_t sum)
    {
        typedef union i64_b32
        {
            int64_t  i64;
            uint32_t b32[2];
        } i64_b32_t;
        i64_b32_t upper_sum;
        i64_b32_t temp_sum;
        temp_sum.i64 = sum;
        if(SUB_WFSIZE == 1)
        {
            return wfreduce_sum<WFSIZE>(sum);
        }
        if(SUB_WFSIZE == 2)
        {
            if(WFSIZE > 2)
            {
                upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x112, 0xf, 0xf, false);
                upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x112, 0xf, 0xf, false);
                temp_sum.i64 += upper_sum.i64;
            }
            if(WFSIZE > 4)
            {
                upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
                upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
                temp_sum.i64 += upper_sum.i64;
            }
            if(WFSIZE > 8)
            {
                upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
                upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
                temp_sum.i64 += upper_sum.i64;
            }
            if(WFSIZE > 16)
            {
                temp_sum.i64 += __shfl_xor(temp_sum.i64, 16);
            }
            if(WFSIZE > 32)
            {
                temp_sum.i64 += __shfl_xor(temp_sum.i64, 32);
            }
            sum = temp_sum.i64;
            return sum;
        }
        if(SUB_WFSIZE == 4)
        {
            if(WFSIZE > 4)
            {
                upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
                upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
                temp_sum.i64 += upper_sum.i64;
            }
            if(WFSIZE > 8)
            {
                upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
                upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
                temp_sum.i64 += upper_sum.i64;
            }
            if(WFSIZE > 16)
            {
                temp_sum.i64 += __shfl_xor(temp_sum.i64, 16);
            }
            if(WFSIZE > 32)
            {
                temp_sum.i64 += __shfl_xor(temp_sum.i64, 32);
            }
            sum = temp_sum.i64;
            return sum;
        }
        if(SUB_WFSIZE == 8)
        {
            if(WFSIZE > 8)
            {
                upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
                upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
                temp_sum.i64 += upper_sum.i64;
            }
            if(WFSIZE > 16)
            {
                temp_sum.i64 += __shfl_xor(temp_sum.i64, 16);
            }
            if(WFSIZE > 32)
            {
                temp_sum.i64 += __shfl_xor(temp_sum.i64, 32);
            }
            sum = temp_sum.i64;
            return sum;
        }
        if(SUB_WFSIZE == 16)
        {
            if(WFSIZE > 16)
            {
                temp_sum.i64 += __shfl_xor(temp_sum.i64, 16);
            }
            if(WFSIZE > 32)
            {
                temp_sum.i64 += __shfl_xor(temp_sum.i64, 32);
            }
            sum = temp_sum.i64;
            return sum;
        }
        if(SUB_WFSIZE == 32)
        {
            if(WFSIZE > 32)
            {
                temp_sum.i64 += __shfl_xor(temp_sum.i64, 32);
            }
            sum = temp_sum.i64;
            return sum;
        }
        sum = temp_sum.i64;
        return sum;
    }
    // DPP-based float wavefront partial reduction sum
    template <uint32_t WFSIZE, uint32_t SUB_WFSIZE>
    __device__ __forceinline__ float wfreduce_partial_sum(float sum)
    {
        typedef union flt_b32
        {
            float    val;
            uint32_t b32;
        } flt_b32_t;
        flt_b32_t upper_sum;
        flt_b32_t temp_sum;
        temp_sum.val = sum;
        if(SUB_WFSIZE == 1)
        {
            return wfreduce_sum<WFSIZE>(sum);
        }
        if(SUB_WFSIZE == 2)
        {
            if(WFSIZE > 2)
            {
                upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x112, 0xf, 0xf, false);
                temp_sum.val += upper_sum.val;
            }
            if(WFSIZE > 4)
            {
                upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x114, 0xf, 0xe, false);
                temp_sum.val += upper_sum.val;
            }
            if(WFSIZE > 8)
            {
                upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x118, 0xf, 0xc, false);
                temp_sum.val += upper_sum.val;
            }
            if(WFSIZE > 16)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 16);
            }
            if(WFSIZE > 32)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 32);
            }
            sum = temp_sum.val;
            return sum;
        }
        if(SUB_WFSIZE == 4)
        {
            if(WFSIZE > 4)
            {
                upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x114, 0xf, 0xe, false);
                temp_sum.val += upper_sum.val;
            }
            if(WFSIZE > 8)
            {
                upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x118, 0xf, 0xc, false);
                temp_sum.val += upper_sum.val;
            }
            if(WFSIZE > 16)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 16);
            }
            if(WFSIZE > 32)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 32);
            }
            sum = temp_sum.val;
            return sum;
        }
        if(SUB_WFSIZE == 8)
        {
            if(WFSIZE > 8)
            {
                upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x118, 0xf, 0xc, false);
                temp_sum.val += upper_sum.val;
            }
            if(WFSIZE > 16)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 16);
            }
            if(WFSIZE > 32)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 32);
            }
            sum = temp_sum.val;
            return sum;
        }
        if(SUB_WFSIZE == 16)
        {
            if(WFSIZE > 16)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 16);
            }
            if(WFSIZE > 32)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 32);
            }
            sum = temp_sum.val;
            return sum;
        }
        if(SUB_WFSIZE == 32)
        {
            if(WFSIZE > 32)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 32);
            }
            sum = temp_sum.val;
            return sum;
        }
        sum = temp_sum.val;
        return sum;
    }

    // DPP-based double wavefront partial reduction
    template <uint32_t WFSIZE, uint32_t SUB_WFSIZE>
    __device__ __forceinline__ double wfreduce_partial_sum(double sum)
    {
        typedef union dbl_b32
        {
            double   val;
            uint32_t b32[2];
        } dbl_b32_t;
        dbl_b32_t upper_sum;
        dbl_b32_t temp_sum;
        temp_sum.val = sum;
        if(SUB_WFSIZE == 1)
        {
            return wfreduce_sum<WFSIZE>(sum);
        }
        if(SUB_WFSIZE == 2)
        {
            if(WFSIZE > 2)
            {
                upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x112, 0xf, 0xf, false);
                upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x112, 0xf, 0xf, false);
                temp_sum.val += upper_sum.val;
            }
            if(WFSIZE > 4)
            {
                upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
                upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
                temp_sum.val += upper_sum.val;
            }
            if(WFSIZE > 8)
            {
                upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
                upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
                temp_sum.val += upper_sum.val;
            }
            if(WFSIZE > 16)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 16);
            }
            if(WFSIZE > 32)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 32);
            }
            sum = temp_sum.val;
            return sum;
        }
        if(SUB_WFSIZE == 4)
        {
            if(WFSIZE > 4)
            {
                upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
                upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
                temp_sum.val += upper_sum.val;
            }
            if(WFSIZE > 8)
            {
                upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
                upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
                temp_sum.val += upper_sum.val;
            }
            if(WFSIZE > 16)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 16);
            }
            if(WFSIZE > 32)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 32);
            }
            sum = temp_sum.val;
            return sum;
        }
        if(SUB_WFSIZE == 8)
        {
            if(WFSIZE > 8)
            {
                upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
                upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
                temp_sum.val += upper_sum.val;
            }
            if(WFSIZE > 16)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 16);
            }
            if(WFSIZE > 32)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 32);
            }
            sum = temp_sum.val;
            return sum;
        }
        if(SUB_WFSIZE == 16)
        {
            if(WFSIZE > 16)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 16);
            }
            if(WFSIZE > 32)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 32);
            }
            sum = temp_sum.val;
            return sum;
        }
        if(SUB_WFSIZE == 32)
        {
            if(WFSIZE > 32)
            {
                temp_sum.val += __shfl_xor(temp_sum.val, 32);
            }
            sum = temp_sum.val;
            return sum;
        }
        sum = temp_sum.val;
        return sum;
    }
#else /* ROCSPARSE_USE_MOVE_DPP */

    template <uint32_t WFSIZE>
    static __device__ __forceinline__ void wfreduce_max(double* maximum)
    {
        for(int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            *maximum = rocsparse::max(*maximum, __shfl_xor(*maximum, i));
        }
    }
    template <uint32_t WFSIZE>
    static __device__ __forceinline__ void wfreduce_max(float* maximum)
    {
        for(int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            *maximum = rocsparse::max(*maximum, __shfl_xor(*maximum, i));
        }
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ void wfreduce_max(int* maximum)
    {
        for(int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            *maximum = rocsparse::max(*maximum, __shfl_xor(*maximum, i));
        }
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ void wfreduce_max(int64_t* maximum)
    {
        for(int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            *maximum = max(*maximum, __shfl_xor(*maximum, i));
        }
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ void wfreduce_min(int* minimum)
    {
        for(int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            *minimum = rocsparse::min(*minimum, __shfl_xor(*minimum, i));
        }
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ void wfreduce_min(int64_t* minimum)
    {
        for(int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            *minimum = rocsparse::min(*minimum, __shfl_xor(*minimum, i));
        }
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ int32_t wfreduce_sum(int32_t sum)
    {
        for(int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            sum += __shfl_xor(sum, i);
        }
        return sum;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ int64_t wfreduce_sum(int64_t sum)
    {
        for(int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            sum += __shfl_xor(sum, i);
        }
        return sum;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ float wfreduce_sum(float sum)
    {
        for(int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            sum += __shfl_xor(sum, i);
        }

        return sum;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ double wfreduce_sum(double sum)
    {
        for(int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            sum += __shfl_xor(sum, i);
        }

        return sum;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ _Float16 wfreduce_sum(_Float16 sum)
    {
        return wfreduce_sum<WFSIZE>((float)sum);
    }

    template <uint32_t WFSIZE, uint32_t SUB_WF_SIZE>
    __device__ __forceinline__ int32_t wfreduce_partial_sum(int32_t sum)
    {
        for(int i = WFSIZE >> 1; i >= SUB_WF_SIZE; i >>= 1)
        {
            sum += __shfl_xor(sum, i);
        }
        return sum;
    }

    template <uint32_t WFSIZE, uint32_t SUB_WF_SIZE>
    __device__ __forceinline__ int64_t wfreduce_partial_sum(int64_t sum)
    {
        for(int i = WFSIZE >> 1; i >= SUB_WF_SIZE; i >>= 1)
        {
            sum += __shfl_xor(sum, i);
        }
        return sum;
    }

    template <uint32_t WFSIZE, uint32_t SUB_WF_SIZE>
    __device__ __forceinline__ float wfreduce_partial_sum(float sum)
    {
        for(int i = WFSIZE >> 1; i >= SUB_WF_SIZE; i >>= 1)
        {
            sum += __shfl_xor(sum, i);
        }

        return sum;
    }

    template <uint32_t WFSIZE, uint32_t SUB_WF_SIZE>
    __device__ __forceinline__ double wfreduce_partial_sum(double sum)
    {
        for(int i = WFSIZE >> 1; i >= SUB_WF_SIZE; i >>= 1)
        {
            sum += __shfl_xor(sum, i);
        }

        return sum;
    }
#endif /* ROCSPARSE_USE_MOVE_DPP */

    // DPP-based complex float wavefront reduction sum
    template <uint32_t WFSIZE>
    __device__ __forceinline__ rocsparse_float_complex wfreduce_sum(rocsparse_float_complex sum)
    {
        return rocsparse_float_complex(rocsparse::wfreduce_sum<WFSIZE>(std::real(sum)),
                                       rocsparse::wfreduce_sum<WFSIZE>(std::imag(sum)));
    }

    // DPP-based complex double wavefront reduction
    template <uint32_t WFSIZE>
    __device__ __forceinline__ rocsparse_double_complex wfreduce_sum(rocsparse_double_complex sum)
    {
        return rocsparse_double_complex(rocsparse::wfreduce_sum<WFSIZE>(std::real(sum)),
                                        rocsparse::wfreduce_sum<WFSIZE>(std::imag(sum)));
    }

    // DPP-based complex float wavefront reduction sum
    template <uint32_t WFSIZE, uint32_t SUB_WF_SIZE>
    __device__ __forceinline__ rocsparse_float_complex
        wfreduce_partial_sum(rocsparse_float_complex sum)
    {
        return rocsparse_float_complex(
            rocsparse::wfreduce_partial_sum<WFSIZE, SUB_WF_SIZE>(std::real(sum)),
            rocsparse::wfreduce_partial_sum<WFSIZE, SUB_WF_SIZE>(std::imag(sum)));
    }

    // DPP-based complex double wavefront reduction
    template <uint32_t WFSIZE, uint32_t SUB_WF_SIZE>
    __device__ __forceinline__ rocsparse_double_complex
        wfreduce_partial_sum(rocsparse_double_complex sum)
    {
        return rocsparse_double_complex(
            rocsparse::wfreduce_partial_sum<WFSIZE, SUB_WF_SIZE>(std::real(sum)),
            rocsparse::wfreduce_partial_sum<WFSIZE, SUB_WF_SIZE>(std::imag(sum)));
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ int32_t wfsegmented_reduce(const int32_t row, int32_t val)
    {
        const uint32_t lid = hipThreadIdx_x & (WFSIZE - 1);

        for(uint32_t j = 1; j < WFSIZE; j <<= 1)
        {
            const int32_t left_row = __shfl_up(row, j);
            const int32_t left_val = __shfl_up(val, j);

            if(row == left_row)
            {
                if(lid >= j)
                {
                    val += left_val;
                }
            }
        }

        return val;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ int32_t wfsegmented_reduce(const int64_t row, int32_t val)
    {
        const uint32_t lid = hipThreadIdx_x & (WFSIZE - 1);

        for(uint32_t j = 1; j < WFSIZE; j <<= 1)
        {
            const int64_t left_row = __shfl_up(row, j);
            const int32_t left_val = __shfl_up(val, j);

            if(row == left_row)
            {
                if(lid >= j)
                {
                    val += left_val;
                }
            }
        }

        return val;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ float wfsegmented_reduce(const int32_t row, float val)
    {
        const uint32_t lid = hipThreadIdx_x & (WFSIZE - 1);

        for(uint32_t j = 1; j < WFSIZE; j <<= 1)
        {
            const int32_t left_row = __shfl_up(row, j);
            const float   left_val = __shfl_up(val, j);

            if(row == left_row)
            {
                if(lid >= j)
                {
                    val += left_val;
                }
            }
        }

        return val;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ float wfsegmented_reduce(const int64_t row, float val)
    {
        const uint32_t lid = hipThreadIdx_x & (WFSIZE - 1);

        for(uint32_t j = 1; j < WFSIZE; j <<= 1)
        {
            const int64_t left_row = __shfl_up(row, j);
            const float   left_val = __shfl_up(val, j);

            if(row == left_row)
            {
                if(lid >= j)
                {
                    val += left_val;
                }
            }
        }

        return val;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ double wfsegmented_reduce(const int32_t row, double val)
    {
        const uint32_t lid = hipThreadIdx_x & (WFSIZE - 1);

        for(uint32_t j = 1; j < WFSIZE; j <<= 1)
        {
            const int32_t left_row = __shfl_up(row, j);
            const double  left_val = __shfl_up(val, j);

            if(row == left_row)
            {
                if(lid >= j)
                {
                    val += left_val;
                }
            }
        }

        return val;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ double wfsegmented_reduce(const int64_t row, double val)
    {
        const uint32_t lid = hipThreadIdx_x & (WFSIZE - 1);

        for(uint32_t j = 1; j < WFSIZE; j <<= 1)
        {
            const int64_t left_row = __shfl_up(row, j);
            const double  left_val = __shfl_up(val, j);

            if(row == left_row)
            {
                if(lid >= j)
                {
                    val += left_val;
                }
            }
        }

        return val;
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ rocsparse_float_complex
        wfsegmented_reduce(const int32_t row, rocsparse_float_complex val)
    {
        return rocsparse_float_complex(rocsparse::wfsegmented_reduce<WFSIZE>(row, std::real(val)),
                                       rocsparse::wfsegmented_reduce<WFSIZE>(row, std::imag(val)));
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ rocsparse_float_complex
        wfsegmented_reduce(const int64_t row, rocsparse_float_complex val)
    {
        return rocsparse_float_complex(rocsparse::wfsegmented_reduce<WFSIZE>(row, std::real(val)),
                                       rocsparse::wfsegmented_reduce<WFSIZE>(row, std::imag(val)));
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ rocsparse_double_complex
        wfsegmented_reduce(const int32_t row, rocsparse_double_complex val)
    {
        return rocsparse_double_complex(rocsparse::wfsegmented_reduce<WFSIZE>(row, std::real(val)),
                                        rocsparse::wfsegmented_reduce<WFSIZE>(row, std::imag(val)));
    }

    template <uint32_t WFSIZE>
    __device__ __forceinline__ rocsparse_double_complex
        wfsegmented_reduce(const int64_t row, rocsparse_double_complex val)
    {
        return rocsparse_double_complex(rocsparse::wfsegmented_reduce<WFSIZE>(row, std::real(val)),
                                        rocsparse::wfsegmented_reduce<WFSIZE>(row, std::imag(val)));
    }

    // BSR gather functionality to permute the BSR values array
    template <uint32_t WFSIZE, uint32_t DIMY, uint32_t BSRDIM, typename I, typename T>
    ROCSPARSE_KERNEL(WFSIZE* DIMY)
    void bsr_gather(rocsparse_direction dir,
                    I                   nnzb,
                    const I* __restrict__ perm,
                    const T* __restrict__ bsr_val_A,
                    T* __restrict__ bsr_val_T,
                    I block_dim)
    {
        int lid = threadIdx.x & (BSRDIM - 1);
        int wid = threadIdx.x / BSRDIM;

        // Non-permuted nnz index
        I j = blockIdx.x * DIMY + threadIdx.y;

        // Do not exceed the number of elements
        if(j >= nnzb)
        {
            return;
        }

        // Load the permuted nnz index
        I p = perm[j];

        // Gather values from A and store them to T with respect to the
        // given row / column permutation
        for(I bi = lid; bi < block_dim; bi += BSRDIM)
        {
            for(I bj = wid; bj < block_dim; bj += BSRDIM)
            {
                bsr_val_T[BSR_IND(j, bi, bj, dir)] = bsr_val_A[BSR_IND(p, bj, bi, dir)];
            }
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csr_max_nnz_per_row(J m, const I* __restrict__ csr_row_ptr, J* __restrict__ max_nnz)
    {
        int tid = hipThreadIdx_x;
        J   gid = tid + BLOCKSIZE * hipBlockIdx_x;

        __shared__ I shared[BLOCKSIZE];

        if(gid < m)
        {
            shared[tid] = csr_row_ptr[gid + 1] - csr_row_ptr[gid];
        }
        else
        {
            shared[tid] = 0;
        }

        __syncthreads();

        rocsparse::blockreduce_max<BLOCKSIZE>(tid, shared);

        if(tid == 0)
        {
            rocsparse::atomic_max(max_nnz, shared[0]);
        }
    }

    template <bool SLEEP>
    __device__ __forceinline__ int spin_loop(int* __restrict__ done, int scope)
    {
        int      local_done    = __hip_atomic_load(done, __ATOMIC_RELAXED, scope);
        uint32_t times_through = 0;
        while(!local_done)
        {
            if(SLEEP)
            {
                for(uint32_t i = 0; i < times_through; ++i)
                {
                    __builtin_amdgcn_s_sleep(1);
                }

                if(times_through < 3907)
                {
                    ++times_through;
                }
            }
            local_done = __hip_atomic_load(done, __ATOMIC_RELAXED, scope);
        }
        return local_done;
    }
}
