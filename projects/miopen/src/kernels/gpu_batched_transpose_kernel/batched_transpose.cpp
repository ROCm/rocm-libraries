/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

#include "miopen_cstdint.hpp"

#ifndef BATCHED_TRANSPOSE_BLOCK_SIZE
#define BATCHED_TRANSPOSE_BLOCK_SIZE 256
#endif

#ifndef BATCHED_TRANSPOSE_OCCUPANCY
#define BATCHED_TRANSPOSE_OCCUPANCY 4
#endif

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

inline __device__ uint32_t magic_div_u32(const uint32_t& numer,
                                         const uint32_t& magic,
                                         const uint32_t& shift)
{
    uint32_t tmp = __umulhi(numer, magic);
    return (tmp + numer) >> shift;
}

inline __device__ void v_pack_b32_f16_00(float& c, const float& a, const float& b)
{
#if 0
    asm volatile("v_pack_b32_f16 %0, %1, %2\n"
                 : "=v"(c)
                 : "v"(a), "v"(b));
#else
    // cppcheck-suppress invalidPointerCast
    const uint32_t x = *reinterpret_cast<const uint32_t*>(&a);
    // cppcheck-suppress invalidPointerCast
    const uint32_t y = *reinterpret_cast<const uint32_t*>(&b);
    uint32_t z       = (x & 0xffff) | ((y & 0xffff) << 16);
    // cppcheck-suppress invalidPointerCast
    c = *reinterpret_cast<float*>(&z);
#endif
}

inline __device__ void v_pack_b32_f16_11(float& c, const float& a, const float& b)
{
#if 0
    asm volatile("v_pack_b32_f16 %0, %1, %2 op_sel:[1, 1]\n"
                 : "=v"(c)
                 : "v"(a), "v"(b));
#else
    // cppcheck-suppress invalidPointerCast
    const uint32_t x = *reinterpret_cast<const uint32_t*>(&a);
    // cppcheck-suppress invalidPointerCast
    const uint32_t y = *reinterpret_cast<const uint32_t*>(&b);
    uint32_t z       = ((x & 0xffff0000) >> 16) | (y & 0xffff0000);
    // cppcheck-suppress invalidPointerCast
    c = *reinterpret_cast<float*>(&z);
#endif
}

inline __device__ void v_pack_b32_f16_2x2(float& y0, float& y1, const float& x0, const float& x1)
{
#if 0
    asm volatile("\n \
                    v_pack_b32_f16 %0, %2, %3\n \
                    v_pack_b32_f16 %1, %2, %3 op_sel:[1, 1]\n"
                 : "=v"(y0), "=v"(y1)
                 : "v"(x0), "v"(x1), "0"(y0), "1"(y1));
#else
    // cppcheck-suppress invalidPointerCast
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&x0);
    // cppcheck-suppress invalidPointerCast
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&x1);
    uint32_t b0       = (a0 & 0xffff) | ((a1 & 0xffff) << 16);
    uint32_t b1       = ((a0 & 0xffff0000) >> 16) | (a1 & 0xffff0000);
    // cppcheck-suppress invalidPointerCast
    y0 = *reinterpret_cast<float*>(&b0);
    // cppcheck-suppress invalidPointerCast
    y1 = *reinterpret_cast<float*>(&b1);
#endif
}

inline __device__ void v_pack_b32_f16_2x2_half_x0(
    float& y0, float& y1, const ushort& x0_lo, const ushort& x0_hi, const float& x1)
{
    // cppcheck-suppress invalidPointerCast
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&x1);
    uint32_t b0       = x0_lo | ((a1 & 0xffff) << 16);
    uint32_t b1       = x0_hi | (a1 & 0xffff0000);
    // cppcheck-suppress invalidPointerCast
    y0 = *reinterpret_cast<float*>(&b0);
    // cppcheck-suppress invalidPointerCast
    y1 = *reinterpret_cast<float*>(&b1);
}

inline __device__ void v_pack_b32_f16_2x2_half_x1(
    float& y0, float& y1, const float& x0, const ushort& x1_lo, const ushort& x1_hi)
{
    // cppcheck-suppress invalidPointerCast
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&x0);
    uint32_t b0       = (a0 & 0xffff) | (x1_lo << 16);
    uint32_t b1       = ((a0 & 0xffff0000) >> 16) | (x1_hi << 16);
    // cppcheck-suppress invalidPointerCast
    y0 = *reinterpret_cast<float*>(&b0);
    // cppcheck-suppress invalidPointerCast
    y1 = *reinterpret_cast<float*>(&b1);
}

inline __device__ void v_pack_b32_f16_2x2_half_x0_half_x1(float& y0,
                                                          float& y1,
                                                          const ushort& x0_lo,
                                                          const ushort& x0_hi,
                                                          const ushort& x1_lo,
                                                          const ushort& x1_hi)
{
    uint32_t b0 = x0_lo | (x1_lo << 16);
    uint32_t b1 = x0_hi | (x1_hi << 16);
    // cppcheck-suppress invalidPointerCast
    y0 = *reinterpret_cast<float*>(&b0);
    // cppcheck-suppress invalidPointerCast
    y1 = *reinterpret_cast<float*>(&b1);
}

template <typename T, int N>
struct mapped_vector_type
{
};

template <>
struct mapped_vector_type<float, 4>
{
    using type = float4;
};

template <>
struct mapped_vector_type<float, 2>
{
    using type = float2;
};

template <>
struct mapped_vector_type<float, 1>
{
    using type = float;
};

template <>
struct mapped_vector_type<ushort, 4>
{
    using type = ushort4;
};

template <>
struct mapped_vector_type<ushort, 2>
{
    using type = ushort2;
};

template <>
struct mapped_vector_type<ushort, 1>
{
    using type = ushort;
};

template <>
struct mapped_vector_type<uchar, 4>
{
    using type = uchar4;
};

template <>
struct mapped_vector_type<uchar, 2>
{
    using type = uchar2;
};

template <>
struct mapped_vector_type<uchar, 1>
{
    using type = uchar;
};

template <typename T>
inline __device__ void batched_transpose_16x16(T* dst,
                                               T* src,
                                               uint32_t height,
                                               uint32_t width,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_h,
                                               uint32_t shift_h,
                                               uint32_t magic_w,
                                               uint32_t shift_w)
{
    /*
     * assume src is batch * height * width, dst is batch * width * height
     */
    constexpr auto element_byte    = sizeof(T);
    constexpr auto padding_element = 4 / element_byte;
    constexpr auto smem_stride     = 16 + padding_element;
    __shared__ T smem[16 * smem_stride];

    uint32_t h_dim = (height + 15) >> 4;
    uint32_t w_dim = (width + 15) >> 4;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 4) + i_src_h;

        __syncthreads();
        if(g_src_h < height && g_src_w < width)
        {
            size_t src_index = static_cast<size_t>(dim_in) * height * width +
                               static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
            smem[i_src_h * smem_stride + i_src_w] = src[src_index];
        }
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 4) + i_dst_w;

        if(g_dst_h < height && g_dst_w < width)
        {
            size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                               static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);
            dst[dst_index] = smem[i_dst_h * smem_stride + i_dst_w];
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_32x16(T* dst,
                                               T* src,
                                               uint32_t height,
                                               uint32_t width,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_h,
                                               uint32_t shift_h,
                                               uint32_t magic_w,
                                               uint32_t shift_w)
{
    /*
     * assume src is batch * height * width, dst is batch * width * height
     */
    constexpr auto element_byte    = sizeof(T);
    constexpr auto padding_element = 4 / element_byte;
    constexpr auto smem_stride     = 16 + padding_element;
    __shared__ T smem[32 * smem_stride];

    uint32_t h_dim = (height + 15) >> 4;
    uint32_t w_dim = (width + 31) >> 5;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 5) + i_src_w;
        uint32_t g_src_h = (dim_ih << 4) + i_src_h;

        __syncthreads();
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        T v_src[2];
        if(g_src_h < height && g_src_w < width)
        {
            v_src[0] = src[src_index];
        }
        if(g_src_h < height && (g_src_w + 16) < width)
        {
            v_src[1] = src[src_index + 16];
        }
        smem[i_src_h * smem_stride + i_src_w]                    = v_src[0];
        smem[i_src_h * smem_stride + i_src_w + 16 * smem_stride] = v_src[1];
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 5) + i_dst_w;

        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);
        T v_dst[2];
        v_dst[0] = smem[i_dst_h * smem_stride + i_dst_w];
        v_dst[1] = smem[i_dst_h * smem_stride + i_dst_w + 16 * smem_stride];

        if(g_dst_h < height && g_dst_w < width)
        {
            dst[dst_index] = v_dst[0];
        }
        if(g_dst_h < height && (g_dst_w + 16) < width)
        {
            dst[dst_index + 16 * height] = v_dst[1];
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_16x32(T* dst,
                                               T* src,
                                               uint32_t height,
                                               uint32_t width,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_h,
                                               uint32_t shift_h,
                                               uint32_t magic_w,
                                               uint32_t shift_w)
{
    /*
     * assume src is batch * height * width, dst is batch * width * height
     */
    constexpr auto element_byte    = sizeof(T);
    constexpr auto padding_element = 4 / element_byte;
    constexpr auto smem_stride     = 16 + padding_element;
    __shared__ T smem[32 * smem_stride];

    uint32_t h_dim = (height + 31) >> 5;
    uint32_t w_dim = (width + 15) >> 4;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 5) + i_src_h;

        __syncthreads();
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        T v_src[2];
        if(g_src_h < height && g_src_w < width)
        {
            v_src[0] = src[src_index];
        }
        if((g_src_h + 16) < height && g_src_w < width)
        {
            v_src[1] = src[src_index + 16 * width];
        }
        smem[i_src_h * smem_stride + i_src_w]                    = v_src[0];
        smem[i_src_h * smem_stride + i_src_w + 16 * smem_stride] = v_src[1];
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 5) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 4) + i_dst_w;

        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);
        T v_dst[2];
        v_dst[0] = smem[i_dst_h * smem_stride + i_dst_w];
        v_dst[1] = smem[i_dst_h * smem_stride + i_dst_w + 16 * smem_stride];

        if(g_dst_h < height && g_dst_w < width)
        {
            dst[dst_index] = v_dst[0];
        }
        if((g_dst_h + 16) < height && g_dst_w < width)
        {
            dst[dst_index + 16] = v_dst[1];
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_32x32(T* dst,
                                               T* src,
                                               uint32_t height,
                                               uint32_t width,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_h,
                                               uint32_t shift_h,
                                               uint32_t magic_w,
                                               uint32_t shift_w)
{
    /*
     * assume src is batch * height * width, dst is batch * width * height
     */
    constexpr auto smem_stride = 17;
    using vec_t                = typename mapped_vector_type<T, 4>::type;
    __shared__ vec_t smem[16 * smem_stride];

    uint32_t h_dim = (height + 31) >> 5;
    uint32_t w_dim = (width + 31) >> 5;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 5) + i_src_w;
        uint32_t g_src_h = (dim_ih << 5) + i_src_h;

        __syncthreads();
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        vec_t v_src;
        if(g_src_h < height && g_src_w < width)
        {
            v_src.x = src[src_index];
        }
        if(g_src_h < height && (g_src_w + 16) < width)
        {
            v_src.z = src[src_index + 16];
        }
        if((g_src_h + 16) < height && g_src_w < width)
        {
            v_src.y = src[src_index + 16 * width];
        }
        if((g_src_h + 16) < height && (g_src_w + 16) < width)
        {
            v_src.w = src[src_index + 16 * width + 16];
        }
        smem[i_src_h * smem_stride + i_src_w] = v_src;
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 5) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 5) + i_dst_w;

        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);
        vec_t v_dst = smem[i_dst_h * smem_stride + i_dst_w];

        if(g_dst_h < height && g_dst_w < width)
        {
            dst[dst_index] = v_dst.x;
        }
        if((g_dst_h + 16) < height && g_dst_w < width)
        {
            dst[dst_index + 16] = v_dst.y;
        }
        if(g_dst_h < height && (g_dst_w + 16) < width)
        {
            dst[dst_index + 16 * height] = v_dst.z;
        }
        if((g_dst_h + 16) < height && (g_dst_w + 16) < width)
        {
            dst[dst_index + 16 * height + 16] = v_dst.w;
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_32x32_pack_2x2_ediv_2x2(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_32x32_pack_2x2_ediv_2x2<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    __shared__ float smem[32 * smem_stride];

    float* p_dst = reinterpret_cast<float*>(dst);
    float* p_src = reinterpret_cast<float*>(src);

    uint32_t height_2 = height >> 1;
    uint32_t width_2  = width >> 1;

    uint32_t h_dim = (height + 31) >> 5;
    uint32_t w_dim = (width + 31) >> 5;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 5) + (i_src_h << 1);

        __syncthreads();
        if(g_src_h < height && g_src_w < width_2)
        {
            float v_a, v_b, v_a_pack, v_b_pack;
            size_t src_index = static_cast<size_t>(dim_in) * height * width_2 +
                               static_cast<size_t>(g_src_h) * width_2 +
                               static_cast<size_t>(g_src_w);
            v_a = p_src[src_index];
            v_b = p_src[src_index + width_2];
            v_pack_b32_f16_2x2(v_a_pack, v_b_pack, v_a, v_b);

            smem[i_src_w * smem_stride + i_src_h]                    = v_a_pack;
            smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] = v_b_pack;
        }
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 5) + (i_dst_w << 1);

        if(g_dst_h < height_2 && g_dst_w < width)
        {
            float v_a, v_b;
            v_a              = smem[i_dst_w * smem_stride + i_dst_h];
            v_b              = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];
            size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 +
                               static_cast<size_t>(g_dst_w) * height_2 +
                               static_cast<size_t>(g_dst_h);
            p_dst[dst_index]            = v_a;
            p_dst[dst_index + height_2] = v_b;
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_32x32_pack_2x2_ediv_1x2(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_32x32_pack_2x2_ediv_1x2<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    __shared__ float smem[32 * smem_stride];

    ushort* p_src = src;
    float* p_dst  = reinterpret_cast<float*>(dst);

    uint32_t height_2 = height >> 1;

    uint32_t h_dim = (height + 31) >> 5;
    uint32_t w_dim = (width + 31) >> 5;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 5) + i_src_w;
        uint32_t g_src_h = (dim_ih << 5) + (i_src_h << 1);

        ushort v_src[4];
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        __syncthreads();
        if(g_src_h < height && g_src_w < width)
        {
            v_src[0] = p_src[src_index];
            v_src[2] = p_src[src_index + width];
        }
        if(g_src_h < height && (g_src_w + 16) < width)
        {
            v_src[1] = p_src[src_index + 16];
            v_src[3] = p_src[src_index + width + 16];
        }

        float v_pack[2];
        v_pack_b32_f16_2x2_half_x0_half_x1(
            v_pack[0], v_pack[1], v_src[0], v_src[1], v_src[2], v_src[3]);

        smem[i_src_w * smem_stride + i_src_h]                    = v_pack[0];
        smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] = v_pack[1];

        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 5) + i_dst_w;

        size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 +
                           static_cast<size_t>(g_dst_w) * height_2 + static_cast<size_t>(g_dst_h);

        float v_a, v_b;
        v_a = smem[i_dst_w * smem_stride + i_dst_h];
        v_b = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];
        if(g_dst_h < height_2 && g_dst_w < width)
        {
            p_dst[dst_index] = v_a;
        }

        if(g_dst_h < height_2 && (g_dst_w + 16) < width)
        {
            p_dst[dst_index + 16 * height_2] = v_b;
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_32x32_pack_2x2_ediv_2x1(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_32x32_pack_2x2_ediv_2x1<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    __shared__ float smem[32 * smem_stride];

    float* p_src  = reinterpret_cast<float*>(src);
    ushort* p_dst = dst;

    uint32_t width_2 = width >> 1;

    uint32_t h_dim = (height + 31) >> 5;
    uint32_t w_dim = (width + 31) >> 5;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 5) + i_src_h;

        float v_src[2];
        size_t src_index = static_cast<size_t>(dim_in) * height * width_2 +
                           static_cast<size_t>(g_src_h) * width_2 + static_cast<size_t>(g_src_w);
        __syncthreads();

        if(g_src_h < height && g_src_w < width_2)
        {
            v_src[0] = p_src[src_index];
        }
        if((g_src_h + 16) < height && g_src_w < width_2)
        {
            v_src[1] = p_src[src_index + 16 * width_2];
        }

        float v_pack[2];
        v_pack_b32_f16_2x2(v_pack[0], v_pack[1], v_src[0], v_src[1]);

        smem[i_src_w * smem_stride + i_src_h]                    = v_pack[0];
        smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] = v_pack[1];

        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 5) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 5) + (i_dst_w << 1);

        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);

        float v_dst[2];
        v_dst[0] = smem[i_dst_w * smem_stride + i_dst_h];
        v_dst[1] = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];

        ushort2 v_dst_buf[2];
        v_dst_buf[0] = *reinterpret_cast<ushort2*>(&v_dst[0]);
        v_dst_buf[1] = *reinterpret_cast<ushort2*>(&v_dst[1]);
        if(g_dst_h < height && g_dst_w < width)
        {
            p_dst[dst_index]          = v_dst_buf[0].x;
            p_dst[dst_index + height] = v_dst_buf[1].x;
        }

        if((g_dst_h + 16) < height && g_dst_w < width)
        {
            p_dst[dst_index + 16]          = v_dst_buf[0].y;
            p_dst[dst_index + height + 16] = v_dst_buf[1].y;
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_32x32_pack_2x2_ediv_1x1(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_32x32_pack_2x2_ediv_1x1<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    __shared__ float smem[32 * smem_stride];

    ushort* p_src = src;
    ushort* p_dst = dst;

    uint32_t h_dim = (height + 31) >> 5;
    uint32_t w_dim = (width + 31) >> 5;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 5) + i_src_w;
        uint32_t g_src_h = (dim_ih << 5) + i_src_h;

        ushort v_src[4];
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        __syncthreads();
        /*
         * 4x4 -> 4x4 transpose: (0, 1, 2, 3 is in ushort, a, b in float)
         *        lo hi
         *        |0|1|      lo |a|b|
         *        |2|3|  ->  hi |_|_|
         */

        if(g_src_h < height && g_src_w < width)
        {
            v_src[0] = p_src[src_index];
        }
        if(g_src_h < height && (g_src_w + 16) < width)
        {
            v_src[1] = p_src[src_index + 16];
        }
        if((g_src_h + 16) < height && g_src_w < width)
        {
            v_src[2] = p_src[src_index + 16 * width];
        }
        if((g_src_h + 16) < height && (g_src_w + 16) < width)
        {
            v_src[3] = p_src[src_index + 16 * width + 16];
        }

        float v_pack[2];
        v_pack_b32_f16_2x2_half_x0_half_x1(
            v_pack[0], v_pack[1], v_src[0], v_src[1], v_src[2], v_src[3]);

        smem[i_src_w * smem_stride + i_src_h]                    = v_pack[0];
        smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] = v_pack[1];

        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 5) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 5) + i_dst_w;

        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);

        float v_dst[2];
        v_dst[0] = smem[i_dst_w * smem_stride + i_dst_h];
        v_dst[1] = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];

        ushort2 v_dst_buf[2];
        v_dst_buf[0] = *reinterpret_cast<ushort2*>(&v_dst[0]);
        v_dst_buf[1] = *reinterpret_cast<ushort2*>(&v_dst[1]);
        if(g_dst_h < height && g_dst_w < width)
        {
            p_dst[dst_index] = v_dst_buf[0].x;
        }
        if((g_dst_h + 16) < height && g_dst_w < width)
        {
            p_dst[dst_index + 16] = v_dst_buf[0].y;
        }
        if(g_dst_h < height && (g_dst_w + 16) < width)
        {
            p_dst[dst_index + 16 * height] = v_dst_buf[1].x;
        }
        if((g_dst_h + 16) < height && (g_dst_w + 16) < width)
        {
            p_dst[dst_index + 16 * height + 16] = v_dst_buf[1].y;
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_64x32_pack_4x2_ediv_4x2(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_64x32_pack_4x2_ediv_4x2<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    __shared__ float smem[64 * smem_stride];
    //__shared__ float4 smem[16 * smem_stride];

    float* p_dst  = reinterpret_cast<float*>(dst);
    float2* p_src = reinterpret_cast<float2*>(src);

    uint32_t height_2 = height >> 1;
    uint32_t width_4  = width >> 2;

    uint32_t h_dim = (height + 31) >> 5;
    uint32_t w_dim = (width + 63) >> 6;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 5) + (i_src_h << 1);

        __syncthreads();
        if(g_src_h < height && g_src_w < width_4)
        {
#if 1
            float2 v_a, v_b;
            float v_pack[4];
            size_t src_index = static_cast<size_t>(dim_in) * height * width_4 +
                               static_cast<size_t>(g_src_h) * width_4 +
                               static_cast<size_t>(g_src_w);
            v_a = p_src[src_index];
            v_b = p_src[src_index + width_4];
            v_pack_b32_f16_2x2(v_pack[0], v_pack[1], v_a.x, v_b.x);
            v_pack_b32_f16_2x2(v_pack[2], v_pack[3], v_a.y, v_b.y);

            smem[i_src_w * smem_stride + i_src_h]                    = v_pack[0];
            smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] = v_pack[1];
            smem[i_src_w * smem_stride + i_src_h + 32 * smem_stride] = v_pack[2];
            smem[i_src_w * smem_stride + i_src_h + 48 * smem_stride] = v_pack[3];
#else
            float2 v_a, v_b;
            float4 v_pack;
            size_t src_index = static_cast<size_t>(dim_in) * height * width_4 +
                               static_cast<size_t>(g_src_h) * width_4 +
                               static_cast<size_t>(g_src_w);
            v_a = p_src[src_index];
            v_b = p_src[src_index + width_4];
            v_pack_b32_f16_2x2(v_pack.x, v_pack.y, v_a.x, v_b.x);
            v_pack_b32_f16_2x2(v_pack.z, v_pack.w, v_a.y, v_b.y);

            smem[i_src_w * smem_stride + i_src_h] = v_pack;
#endif
        }
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 6) + (i_dst_w << 2);

        if(g_dst_h < height_2 && g_dst_w < width)
        {
#if 1
            float v[4];
            v[0]             = smem[i_dst_w * smem_stride + i_dst_h];
            v[1]             = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];
            v[2]             = smem[i_dst_w * smem_stride + i_dst_h + 32 * smem_stride];
            v[3]             = smem[i_dst_w * smem_stride + i_dst_h + 48 * smem_stride];
            size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 +
                               static_cast<size_t>(g_dst_w) * height_2 +
                               static_cast<size_t>(g_dst_h);
            p_dst[dst_index]                = v[0];
            p_dst[dst_index + height_2]     = v[1];
            p_dst[dst_index + 2 * height_2] = v[2];
            p_dst[dst_index + 3 * height_2] = v[3];
#else
            float4 v;
            v                = smem[i_dst_w * smem_stride + i_dst_h];
            size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 +
                               static_cast<size_t>(g_dst_w) * height_2 +
                               static_cast<size_t>(g_dst_h);
            p_dst[dst_index]                = v.x;
            p_dst[dst_index + height_2]     = v.y;
            p_dst[dst_index + 2 * height_2] = v.z;
            p_dst[dst_index + 3 * height_2] = v.w;
#endif
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_64x32_pack_4x2_ediv_2x2(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_64x32_pack_4x2_ediv_2x2<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    //__shared__ float smem[64 * smem_stride];
    __shared__ float4 smem[16 * smem_stride];

    float* p_dst = reinterpret_cast<float*>(dst);
    float* p_src = reinterpret_cast<float*>(src);

    uint32_t height_2 = height >> 1;
    uint32_t width_2  = width >> 1;

    uint32_t h_dim = (height + 31) >> 5;
    uint32_t w_dim = (width + 63) >> 6;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 5) + i_src_w;
        uint32_t g_src_h = (dim_ih << 5) + (i_src_h << 1);

        __syncthreads();
        /*
         * 4x2 -> 2x4 transpose
         *        lo hi
         *        |0|_|2|_|      lo |0|1|2|3|
         *        |1|_|3|_|  ->  hi |_|_|_|_|
         */
        float v_src[4];
        size_t src_index = static_cast<size_t>(dim_in) * height * width_2 +
                           static_cast<size_t>(g_src_h) * width_2 + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width_2)
        {
            v_src[0] = p_src[src_index];
            v_src[1] = p_src[src_index + width_2];
        }
        if(g_src_h < height && (g_src_w + 16) < width_2)
        {
            v_src[2] = p_src[src_index + 16];
            v_src[3] = p_src[src_index + width_2 + 16];
        }

        float4 v_pack;
        v_pack_b32_f16_2x2(v_pack.x, v_pack.y, v_src[0], v_src[1]);
        v_pack_b32_f16_2x2(v_pack.z, v_pack.w, v_src[2], v_src[3]);

        smem[i_src_w * smem_stride + i_src_h] = v_pack;
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 6) + (i_dst_w << 1);

        size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 +
                           static_cast<size_t>(g_dst_w) * height_2 + static_cast<size_t>(g_dst_h);

        float4 v_dst = smem[i_dst_w * smem_stride + i_dst_h];
        if(g_dst_h < height_2 && g_dst_w < width)
        {
            p_dst[dst_index]            = v_dst.x;
            p_dst[dst_index + height_2] = v_dst.y;
        }
        if(g_dst_h < height_2 && (g_dst_w + 32) < width)
        {
            p_dst[dst_index + 32 * height_2] = v_dst.z;
            p_dst[dst_index + 33 * height_2] = v_dst.w;
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_64x32_pack_4x2_ediv_2x1(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_64x32_pack_4x2_ediv_2x1<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    __shared__ float4 smem[16 * smem_stride];

    ushort* p_dst = reinterpret_cast<ushort*>(dst);
    float* p_src  = reinterpret_cast<float*>(src);

    uint32_t width_2 = width >> 1;

    uint32_t h_dim = (height + 31) >> 5;
    uint32_t w_dim = (width + 63) >> 6;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 5) + i_src_w;
        uint32_t g_src_h = (dim_ih << 5) + i_src_h;

        __syncthreads();
        /*
         * 4x2 -> 2x4 transpose
         *        lo hi
         *        |0|_|2|_|      lo |0|1|2|3|
         *        |1|_|3|_|  ->  hi |_|_|_|_|
         */
        float v_src[4];
        size_t src_index = static_cast<size_t>(dim_in) * height * width_2 +
                           static_cast<size_t>(g_src_h) * width_2 + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width_2)
        {
            v_src[0] = p_src[src_index];
        }
        if(g_src_h < height && (g_src_w + 16) < width_2)
        {
            v_src[2] = p_src[src_index + 16];
        }
        if((g_src_h + 16) < height && g_src_w < width_2)
        {
            v_src[1] = p_src[src_index + 16 * width_2];
        }
        if((g_src_h + 16) < height && (g_src_w + 16) < width_2)
        {
            v_src[3] = p_src[src_index + 16 * width_2 + 16];
        }

        float4 v_pack;
        v_pack_b32_f16_2x2(v_pack.x, v_pack.y, v_src[0], v_src[1]);
        v_pack_b32_f16_2x2(v_pack.z, v_pack.w, v_src[2], v_src[3]);

        smem[i_src_w * smem_stride + i_src_h] = v_pack;
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 5) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 6) + (i_dst_w << 1);

        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);

        float4 v_dst = smem[i_dst_w * smem_stride + i_dst_h];
        ushort2 v_dst_buf[4];
        v_dst_buf[0] = *reinterpret_cast<ushort2*>(&v_dst.x);
        v_dst_buf[1] = *reinterpret_cast<ushort2*>(&v_dst.y);
        v_dst_buf[2] = *reinterpret_cast<ushort2*>(&v_dst.z);
        v_dst_buf[3] = *reinterpret_cast<ushort2*>(&v_dst.w);
        if(g_dst_h < height && g_dst_w < width)
        {
            p_dst[dst_index]          = v_dst_buf[0].x;
            p_dst[dst_index + height] = v_dst_buf[1].x;
        }
        if((g_dst_h + 16) < height && g_dst_w < width)
        {
            p_dst[dst_index + 16]          = v_dst_buf[0].y;
            p_dst[dst_index + height + 16] = v_dst_buf[1].y;
        }
        if(g_dst_h < height && (g_dst_w + 32) < width)
        {
            p_dst[dst_index + 32 * height] = v_dst_buf[2].x;
            p_dst[dst_index + 33 * height] = v_dst_buf[3].x;
        }
        if((g_dst_h + 16) < height && (g_dst_w + 32) < width)
        {
            p_dst[dst_index + 32 * height + 16] = v_dst_buf[2].y;
            p_dst[dst_index + 33 * height + 16] = v_dst_buf[3].y;
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_32x64_pack_2x4_ediv_2x4(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_32x64_pack_2x4_ediv_2x4<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    //__shared__ float smem[64 * smem_stride];
    __shared__ float4 smem[16 * smem_stride];

    float2* p_dst = reinterpret_cast<float2*>(dst);
    float* p_src  = reinterpret_cast<float*>(src);

    uint32_t height_4 = height >> 2;
    uint32_t width_2  = width >> 1;

    uint32_t h_dim = (height + 63) >> 6;
    uint32_t w_dim = (width + 31) >> 5;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 6) + (i_src_h << 2);

        __syncthreads();
        /*
         * 4x2 -> 2x4 transpose: (0, 1, 2, 3 is in float)
         *        lo hi
         *      0 |_|_|      lo |0|2|
         *      1 |_|_|  ->  hi |_|_|
         *      2 |_|_|      lo |1|3|
         *      3 |_|_|      hi |_|_|
         */
        if(g_src_h < height && g_src_w < width_2)
        {
#if 0
            float v[4];
            float v_pack[4];
            size_t src_index = static_cast<size_t>(dim_in) * height * width_2 + static_cast<size_t>(g_src_h) * width_2 + static_cast<size_t>(g_src_w);
            v[0] = p_src[src_index];
            v[1] = p_src[src_index + width_2];
            v[2] = p_src[src_index + 2 * width_2];
            v[3] = p_src[src_index + 3 * width_2];
            v_pack_b32_f16_2x2(v_pack[0], v_pack[2], v[0], v[1]);
            v_pack_b32_f16_2x2(v_pack[1], v_pack[3], v[2], v[3]);

            smem[i_src_w * smem_stride + i_src_h] = v_pack[0];
            smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] = v_pack[1];
            smem[i_src_w * smem_stride + i_src_h + 32 * smem_stride] = v_pack[2];
            smem[i_src_w * smem_stride + i_src_h + 48 * smem_stride] = v_pack[3];
#else
            float v[4];
            float4 v_pack;
            size_t src_index = static_cast<size_t>(dim_in) * height * width_2 +
                               static_cast<size_t>(g_src_h) * width_2 +
                               static_cast<size_t>(g_src_w);
            v[0] = p_src[src_index];
            v[1] = p_src[src_index + width_2];
            v[2] = p_src[src_index + 2 * width_2];
            v[3] = p_src[src_index + 3 * width_2];
            v_pack_b32_f16_2x2(v_pack.x, v_pack.z, v[0], v[1]);
            v_pack_b32_f16_2x2(v_pack.y, v_pack.w, v[2], v[3]);

            smem[i_src_w * smem_stride + i_src_h] = v_pack;
#endif
        }
        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 5) + (i_dst_w << 1);

        if(g_dst_h < height_4 && g_dst_w < width)
        {
#if 0
            float v[4];
            v[0] = smem[i_dst_w * smem_stride + i_dst_h];
            v[1] = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];
            v[2] = smem[i_dst_w * smem_stride + i_dst_h + 32 * smem_stride];
            v[3] = smem[i_dst_w * smem_stride + i_dst_h + 48 * smem_stride];
            size_t dst_index = static_cast<size_t>(dim_in) * width * height_4 + static_cast<size_t>(g_dst_w) * height_4 + static_cast<size_t>(g_dst_h);
            p_dst[dst_index] = v[0];
            p_dst[dst_index + height_4] = v[1];
            p_dst[dst_index + 2 * height_4] = v[2];
            p_dst[dst_index + 3 * height_4] = v[3];
#else
            float4 v;
            v                = smem[i_dst_w * smem_stride + i_dst_h];
            size_t dst_index = static_cast<size_t>(dim_in) * width * height_4 +
                               static_cast<size_t>(g_dst_w) * height_4 +
                               static_cast<size_t>(g_dst_h);
            p_dst[dst_index]            = make_float2(v.x, v.y);
            p_dst[dst_index + height_4] = make_float2(v.z, v.w);
#endif
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_32x64_pack_2x4_ediv_2x2(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_32x64_pack_2x4_ediv_2x2<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    //__shared__ float smem[64 * smem_stride];
    __shared__ float4 smem[16 * smem_stride];

    float* p_dst = reinterpret_cast<float*>(dst);
    float* p_src = reinterpret_cast<float*>(src);

    uint32_t height_2 = height >> 1;
    uint32_t width_2  = width >> 1;

    uint32_t h_dim = (height + 63) >> 6;
    uint32_t w_dim = (width + 31) >> 5;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 6) + (i_src_h << 1);

        __syncthreads();

        float v_src[4];
        size_t src_index = static_cast<size_t>(dim_in) * height * width_2 +
                           static_cast<size_t>(g_src_h) * width_2 + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width_2)
        {
            v_src[0] = p_src[src_index];
            v_src[1] = p_src[src_index + width_2];
        }
        if((g_src_h + 32) < height && g_src_w < width_2)
        {
            v_src[2] = p_src[src_index + 32 * width_2];
            v_src[3] = p_src[src_index + 33 * width_2];
        }

        float4 v_pack;
        v_pack_b32_f16_2x2(v_pack.x, v_pack.z, v_src[0], v_src[1]);
        v_pack_b32_f16_2x2(v_pack.y, v_pack.w, v_src[2], v_src[3]);

        smem[i_src_w * smem_stride + i_src_h] = v_pack;

        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 5) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 5) + (i_dst_w << 1);

        size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 +
                           static_cast<size_t>(g_dst_w) * height_2 + static_cast<size_t>(g_dst_h);

        float4 v_dst = smem[i_dst_w * smem_stride + i_dst_h];
        if(g_dst_h < height_2 && g_dst_w < width)
        {
            p_dst[dst_index]            = v_dst.x;
            p_dst[dst_index + height_2] = v_dst.z;
        }
        if((g_dst_h + 16) < height_2 && g_dst_w < width)
        {
            p_dst[dst_index + 16]            = v_dst.y;
            p_dst[dst_index + height_2 + 16] = v_dst.w;
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_32x64_pack_2x4_ediv_1x2(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_32x64_pack_2x4_ediv_1x2<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    //__shared__ float smem[64 * smem_stride];
    __shared__ float4 smem[16 * smem_stride];

    float* p_dst  = reinterpret_cast<float*>(dst);
    ushort* p_src = src;

    uint32_t height_2 = height >> 1;

    uint32_t h_dim = (height + 63) >> 6;
    uint32_t w_dim = (width + 31) >> 5;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 5) + i_src_w;
        uint32_t g_src_h = (dim_ih << 6) + (i_src_h << 1);

        __syncthreads();
        /*
         * 4x2 -> 2x4 transpose:
         *        lo hi
         *        |0|1|      lo |0|2|
         *        |2|3|  ->  hi |_|_|
         *        |4|5|      lo |1|3|
         *        |6|7|      hi |_|_|
         */

        ushort v_src[8];
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width)
        {
            v_src[0] = p_src[src_index];
            v_src[2] = p_src[src_index + width];
        }
        if(g_src_h < height && (g_src_w + 16) < width)
        {
            v_src[1] = p_src[src_index + 16];
            v_src[3] = p_src[src_index + width + 16];
        }
        if((g_src_h + 32) < height && g_src_w < width)
        {
            v_src[4] = p_src[src_index + 32 * width];
            v_src[6] = p_src[src_index + 33 * width];
        }
        if((g_src_h + 32) < height && (g_src_w + 16) < width)
        {
            v_src[5] = p_src[src_index + 32 * width + 16];
            v_src[7] = p_src[src_index + 33 * width + 16];
        }

        float4 v_pack;
        v_pack_b32_f16_2x2_half_x0_half_x1(
            v_pack.x, v_pack.z, v_src[0], v_src[1], v_src[2], v_src[3]);
        v_pack_b32_f16_2x2_half_x0_half_x1(
            v_pack.y, v_pack.w, v_src[4], v_src[5], v_src[6], v_src[7]);

        smem[i_src_w * smem_stride + i_src_h] = v_pack;

        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 5) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 5) + i_dst_w;

        size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 +
                           static_cast<size_t>(g_dst_w) * height_2 + static_cast<size_t>(g_dst_h);

        float4 v_dst = smem[i_dst_w * smem_stride + i_dst_h];
        if(g_dst_h < height_2 && g_dst_w < width)
        {
            p_dst[dst_index] = v_dst.x;
        }
        if((g_dst_h + 16) < height_2 && g_dst_w < width)
        {
            p_dst[dst_index + 16] = v_dst.y;
        }
        if(g_dst_h < height_2 && (g_dst_w + 16) < width)
        {
            p_dst[dst_index + 16 * height_2] = v_dst.z;
        }
        if((g_dst_h + 16) < height_2 && (g_dst_w + 16) < width)
        {
            p_dst[dst_index + 16 * height_2 + 16] = v_dst.w;
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_16x64_pack_1x4_ediv_1x2(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_16x64_pack_1x4_ediv_1x2<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    __shared__ float smem[32 * smem_stride];

    float* p_dst  = reinterpret_cast<float*>(dst);
    ushort* p_src = src;

    uint32_t height_2 = height >> 1;

    uint32_t h_dim = (height + 63) >> 6;
    uint32_t w_dim = (width + 15) >> 4;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 6) + (i_src_h << 1);

        __syncthreads();
        /*
         * 4x1 -> 1x4 transpose:
         *        lo hi
         *        |0|       lo |0|
         *        |1|   ->  hi |_|
         *        |2|       lo |1|
         *        |3|       hi |_|
         */

        ushort v_src[4];
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width)
        {
            v_src[0] = p_src[src_index];
            v_src[1] = p_src[src_index + width];
        }
        if((g_src_h + 32) < height && g_src_w < width)
        {
            v_src[2] = p_src[src_index + 32 * width];
            v_src[3] = p_src[src_index + 33 * width];
        }

        ushort2 v_pack_tmp[2];
        v_pack_tmp[0] = make_ushort2(v_src[0], v_src[1]);
        v_pack_tmp[1] = make_ushort2(v_src[2], v_src[3]);

        float v_pack[2];
        v_pack[0] = *reinterpret_cast<float*>(&v_pack_tmp[0]);
        v_pack[1] = *reinterpret_cast<float*>(&v_pack_tmp[1]);

        smem[i_src_w * smem_stride + i_src_h]                    = v_pack[0];
        smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] = v_pack[1];

        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 5) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 4) + i_dst_w;

        size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 +
                           static_cast<size_t>(g_dst_w) * height_2 + static_cast<size_t>(g_dst_h);

        float v_dst[2];
        v_dst[0] = smem[i_dst_w * smem_stride + i_dst_h];
        v_dst[1] = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];
        if(g_dst_h < height_2 && g_dst_w < width)
        {
            p_dst[dst_index] = v_dst[0];
        }
        if((g_dst_h + 16) < height_2 && g_dst_w < width)
        {
            p_dst[dst_index + 16] = v_dst[1];
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_64x16_pack_4x1_ediv_2x1(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_64x16_pack_4x1_ediv_2x1<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    __shared__ float smem[32 * smem_stride];

    ushort* p_dst = dst;
    float* p_src  = reinterpret_cast<float*>(src);

    uint32_t width_2 = width >> 1;

    uint32_t h_dim = (height + 15) >> 4;
    uint32_t w_dim = (width + 63) >> 6;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 5) + i_src_w;
        uint32_t g_src_h = (dim_ih << 4) + i_src_h;

        __syncthreads();
        float v_src[2];
        size_t src_index = static_cast<size_t>(dim_in) * height * width_2 +
                           static_cast<size_t>(g_src_h) * width_2 + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width_2)
        {
            v_src[0] = p_src[src_index];
        }
        if(g_src_h < height && (g_src_w + 16) < width_2)
        {
            v_src[1] = p_src[src_index + 16];
        }

        smem[i_src_w * smem_stride + i_src_h]                    = v_src[0];
        smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] = v_src[1];

        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 6) + (i_dst_w << 1);

        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);

        float v_dst[2];
        v_dst[0] = smem[i_dst_w * smem_stride + i_dst_h];
        v_dst[1] = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];

        ushort2 v_dst_buf[2];
        v_dst_buf[0] = *reinterpret_cast<ushort2*>(&v_dst[0]);
        v_dst_buf[1] = *reinterpret_cast<ushort2*>(&v_dst[1]);
        if(g_dst_h < height && g_dst_w < width)
        {
            p_dst[dst_index]          = v_dst_buf[0].x;
            p_dst[dst_index + height] = v_dst_buf[0].y;
        }
        if(g_dst_h < height && (g_dst_w + 32) < width)
        {
            p_dst[dst_index + 32 * height] = v_dst_buf[1].x;
            p_dst[dst_index + 33 * height] = v_dst_buf[1].y;
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_64x64_pack_4x4_ediv_4x4(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_64x64_pack_4x4_ediv_4x4<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    //__shared__ float smem[64 * smem_stride];
    __shared__ float4 smem[32 * smem_stride];

    float2* p_dst = reinterpret_cast<float2*>(dst);
    float2* p_src = reinterpret_cast<float2*>(src);

    uint32_t height_4 = height >> 2;
    uint32_t width_4  = width >> 2;

    uint32_t h_dim = (height + 63) >> 6;
    uint32_t w_dim = (width + 63) >> 6;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 4) + i_src_w;
        uint32_t g_src_h = (dim_ih << 6) + (i_src_h << 2);

        __syncthreads();
        /*
         * 4x2 -> 2x4 transpose: (0, 1, 2, 3 is in float2)
         *        lo hi
         *      0 |_|_|_|_|      lo |0|1|2|3|
         *      1 |_|_|_|_|  ->  hi |_|_|_|_|
         *      2 |_|_|_|_|      lo | | | | |
         *      3 |_|_|_|_|      hi |_|_|_|_|
         */

        float2 v_src[4];
        size_t src_index = static_cast<size_t>(dim_in) * height * width_4 +
                           static_cast<size_t>(g_src_h) * width_4 + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width_4)
        {
            v_src[0] = p_src[src_index];
            v_src[1] = p_src[src_index + width_4];
            v_src[2] = p_src[src_index + 2 * width_4];
            v_src[3] = p_src[src_index + 3 * width_4];
        }

        float2 v_pack[4];
        v_pack_b32_f16_2x2(v_pack[0].x, v_pack[1].x, v_src[0].x, v_src[1].x);
        v_pack_b32_f16_2x2(v_pack[2].x, v_pack[3].x, v_src[0].y, v_src[1].y);
        v_pack_b32_f16_2x2(v_pack[0].y, v_pack[1].y, v_src[2].x, v_src[3].x);
        v_pack_b32_f16_2x2(v_pack[2].y, v_pack[3].y, v_src[2].y, v_src[3].y);

        smem[i_src_w * smem_stride + i_src_h] =
            make_float4(v_pack[0].x, v_pack[0].y, v_pack[1].x, v_pack[1].y);
        smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] =
            make_float4(v_pack[2].x, v_pack[2].y, v_pack[3].x, v_pack[3].y);

        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 4) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 6) + (i_dst_w << 2);

        size_t dst_index = static_cast<size_t>(dim_in) * width * height_4 +
                           static_cast<size_t>(g_dst_w) * height_4 + static_cast<size_t>(g_dst_h);

        float4 v_dst[2];
        v_dst[0] = smem[i_dst_w * smem_stride + i_dst_h];
        v_dst[1] = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];
        if(g_dst_h < height_4 && g_dst_w < width)
        {
            p_dst[dst_index]                = make_float2(v_dst[0].x, v_dst[0].y);
            p_dst[dst_index + height_4]     = make_float2(v_dst[0].z, v_dst[0].w);
            p_dst[dst_index + 2 * height_4] = make_float2(v_dst[1].x, v_dst[1].y);
            p_dst[dst_index + 3 * height_4] = make_float2(v_dst[1].z, v_dst[1].w);
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_64x64_pack_4x4_ediv_2x2(T* /*dst*/,
                                                                 T* /*src*/,
                                                                 uint32_t /*height*/,
                                                                 uint32_t /*width*/,
                                                                 uint32_t /*dim_stride*/,
                                                                 uint32_t /*dim_total*/,
                                                                 uint32_t /*magic_h*/,
                                                                 uint32_t /*shift_h*/,
                                                                 uint32_t /*magic_w*/,
                                                                 uint32_t /*shift_w*/)
{
}

template <>
inline __device__ void batched_transpose_64x64_pack_4x4_ediv_2x2<ushort>(ushort* dst,
                                                                         ushort* src,
                                                                         uint32_t height,
                                                                         uint32_t width,
                                                                         uint32_t dim_stride,
                                                                         uint32_t dim_total,
                                                                         uint32_t magic_h,
                                                                         uint32_t shift_h,
                                                                         uint32_t magic_w,
                                                                         uint32_t shift_w)
{
    constexpr auto smem_stride = 17;
    __shared__ float4 smem[32 * smem_stride];

    float* p_dst = reinterpret_cast<float*>(dst);
    float* p_src = reinterpret_cast<float*>(src);

    uint32_t height_2 = height >> 1;
    uint32_t width_2  = width >> 1;

    uint32_t h_dim = (height + 63) >> 6;
    uint32_t w_dim = (width + 63) >> 6;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t i_src_w = threadIdx.x & 15;
        uint32_t i_src_h = threadIdx.x >> 4;
        uint32_t g_src_w = (dim_iw << 5) + i_src_w;
        uint32_t g_src_h = (dim_ih << 6) + (i_src_h << 1);

        __syncthreads();
        /*
         * 4x4 -> 4x4 transpose: (0, 1, 2, 3 is in float, a, b, c, d is in float2)
         *        lo hi
         *       |0|_|1|_|      lo |a|b|c|d|
         *       |2|_|3|_|  ->  hi |_|_|_|_|
         *       |4|_|5|_|      lo | | | | |
         *       |6|_|7|_|      hi |_|_|_|_|
         */

        float v_src[8];
        size_t src_index = static_cast<size_t>(dim_in) * height * width_2 +
                           static_cast<size_t>(g_src_h) * width_2 + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width_2)
        {
            v_src[0] = p_src[src_index];
            v_src[2] = p_src[src_index + width_2];
        }
        if(g_src_h < height && (g_src_w + 16) < width_2)
        {
            v_src[1] = p_src[src_index + 16];
            v_src[3] = p_src[src_index + width_2 + 16];
        }
        if((g_src_h + 32) < height && g_src_w < width_2)
        {
            v_src[4] = p_src[src_index + 32 * width_2];
            v_src[6] = p_src[src_index + 33 * width_2];
        }
        if((g_src_h + 32) < height && (g_src_w + 16) < width_2)
        {
            v_src[5] = p_src[src_index + 32 * width_2 + 16];
            v_src[7] = p_src[src_index + 33 * width_2 + 16];
        }

        float2 v_pack[4];
        v_pack_b32_f16_2x2(v_pack[0].x, v_pack[1].x, v_src[0], v_src[2]);
        v_pack_b32_f16_2x2(v_pack[2].x, v_pack[3].x, v_src[1], v_src[3]);
        v_pack_b32_f16_2x2(v_pack[0].y, v_pack[1].y, v_src[4], v_src[6]);
        v_pack_b32_f16_2x2(v_pack[2].y, v_pack[3].y, v_src[5], v_src[7]);

        smem[i_src_w * smem_stride + i_src_h] =
            make_float4(v_pack[0].x, v_pack[0].y, v_pack[1].x, v_pack[1].y);
        smem[i_src_w * smem_stride + i_src_h + 16 * smem_stride] =
            make_float4(v_pack[2].x, v_pack[2].y, v_pack[3].x, v_pack[3].y);

        __syncthreads();

        uint32_t i_dst_h = threadIdx.x & 15;
        uint32_t i_dst_w = threadIdx.x >> 4;
        uint32_t g_dst_h = (dim_ih << 5) + i_dst_h;
        uint32_t g_dst_w = (dim_iw << 6) + (i_dst_w << 1);

        size_t dst_index = static_cast<size_t>(dim_in) * width * height_2 +
                           static_cast<size_t>(g_dst_w) * height_2 + static_cast<size_t>(g_dst_h);

        float4 v_dst[2];
        v_dst[0] = smem[i_dst_w * smem_stride + i_dst_h];
        v_dst[1] = smem[i_dst_w * smem_stride + i_dst_h + 16 * smem_stride];
        if(g_dst_h < height_2 && g_dst_w < width)
        {
            p_dst[dst_index]            = v_dst[0].x;
            p_dst[dst_index + height_2] = v_dst[0].z;
        }
        if((g_dst_h + 16) < height_2 && g_dst_w < width)
        {
            p_dst[dst_index + 16]            = v_dst[0].y;
            p_dst[dst_index + height_2 + 16] = v_dst[0].w;
        }
        if(g_dst_h < height_2 && (g_dst_w + 32) < width)
        {
            p_dst[dst_index + 32 * height_2] = v_dst[1].x;
            p_dst[dst_index + 33 * height_2] = v_dst[1].z;
        }
        if((g_dst_h + 16) < height_2 && (g_dst_w + 32) < width)
        {
            p_dst[dst_index + 32 * height_2 + 16] = v_dst[1].y;
            p_dst[dst_index + 33 * height_2 + 16] = v_dst[1].w;
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_4x256(T* dst,
                                               T* src,
                                               uint32_t height,
                                               uint32_t width,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_h,
                                               uint32_t shift_h,
                                               uint32_t magic_w,
                                               uint32_t shift_w)
{
    uint32_t h_dim = (height + 255) >> 8;
    uint32_t w_dim = (width + 3) >> 2;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t g_src_w = (dim_iw << 2);
        uint32_t g_src_h = (dim_ih << 8) + threadIdx.x;

        T v_buf[4];
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width)
        {
            v_buf[0] = src[src_index];
        }
        if(g_src_h < height && (g_src_w + 1) < width)
        {
            v_buf[1] = src[src_index + 1];
        }
        if(g_src_h < height && (g_src_w + 2) < width)
        {
            v_buf[2] = src[src_index + 2];
        }
        if(g_src_h < height && (g_src_w + 3) < width)
        {
            v_buf[3] = src[src_index + 3];
        }

        uint32_t g_dst_h = (dim_ih << 8) + threadIdx.x;
        uint32_t g_dst_w = (dim_iw << 2);
        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);

        if(g_dst_h < height && g_dst_w < width)
        {
            dst[dst_index] = v_buf[0];
        }
        if(g_dst_h < height && (g_dst_w + 1) < width)
        {
            dst[dst_index + height] = v_buf[1];
        }
        if(g_dst_h < height && (g_dst_w + 2) < width)
        {
            dst[dst_index + 2 * height] = v_buf[2];
        }
        if(g_dst_h < height && (g_dst_w + 3) < width)
        {
            dst[dst_index + 3 * height] = v_buf[3];
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_256x4(T* dst,
                                               T* src,
                                               uint32_t height,
                                               uint32_t width,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_h,
                                               uint32_t shift_h,
                                               uint32_t magic_w,
                                               uint32_t shift_w)
{
    uint32_t h_dim = (height + 3) >> 2;
    uint32_t w_dim = (width + 255) >> 8;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t g_src_w = (dim_iw << 8) + threadIdx.x;
        uint32_t g_src_h = (dim_ih << 2);

        T v_buf[4];
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width)
        {
            v_buf[0] = src[src_index];
        }
        if((g_src_h + 1) < height && g_src_w < width)
        {
            v_buf[1] = src[src_index + width];
        }
        if((g_src_h + 2) < height && g_src_w < width)
        {
            v_buf[2] = src[src_index + 2 * width];
        }
        if((g_src_h + 3) < height && g_src_w < width)
        {
            v_buf[3] = src[src_index + 3 * width];
        }

        uint32_t g_dst_h = (dim_ih << 2);
        uint32_t g_dst_w = (dim_iw << 8) + threadIdx.x;
        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);

        if(g_dst_h < height && g_dst_w < width)
        {
            dst[dst_index] = v_buf[0];
        }
        if((g_dst_h + 1) < height && g_dst_w < width)
        {
            dst[dst_index + 1] = v_buf[1];
        }
        if((g_dst_h + 2) < height && g_dst_w < width)
        {
            dst[dst_index + 2] = v_buf[2];
        }
        if((g_dst_h + 3) < height && g_dst_w < width)
        {
            dst[dst_index + 3] = v_buf[3];
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_4x128(T* dst,
                                               T* src,
                                               uint32_t height,
                                               uint32_t width,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_h,
                                               uint32_t shift_h,
                                               uint32_t magic_w,
                                               uint32_t shift_w)
{
    uint32_t h_dim = (height + 127) >> 7;
    uint32_t w_dim = (width + 3) >> 2;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t g_src_w = (dim_iw << 2) + (threadIdx.x >> 7);
        uint32_t g_src_h = (dim_ih << 7) + (threadIdx.x & 127);

        T v_buf[2];
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width)
        {
            v_buf[0] = src[src_index];
        }
        if(g_src_h < height && (g_src_w + 2) < width)
        {
            v_buf[1] = src[src_index + 2];
        }

        uint32_t g_dst_h = (dim_ih << 7) + (threadIdx.x & 127);
        uint32_t g_dst_w = (dim_iw << 2) + (threadIdx.x >> 7);
        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);

        if(g_dst_h < height && g_dst_w < width)
        {
            dst[dst_index] = v_buf[0];
        }
        if(g_dst_h < height && (g_dst_w + 2) < width)
        {
            dst[dst_index + 2 * height] = v_buf[1];
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_128x4(T* dst,
                                               T* src,
                                               uint32_t height,
                                               uint32_t width,
                                               uint32_t dim_stride,
                                               uint32_t dim_total,
                                               uint32_t magic_h,
                                               uint32_t shift_h,
                                               uint32_t magic_w,
                                               uint32_t shift_w)
{
    uint32_t h_dim = (height + 3) >> 2;
    uint32_t w_dim = (width + 127) >> 7;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t g_src_w = (dim_iw << 7) + (threadIdx.x & 127);
        uint32_t g_src_h = (dim_ih << 2) + (threadIdx.x >> 7);

        T v_buf[2];
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width)
        {
            v_buf[0] = src[src_index];
        }
        if((g_src_h + 2) < height && g_src_w < width)
        {
            v_buf[1] = src[src_index + 2 * width];
        }

        uint32_t g_dst_h = (dim_ih << 2) + (threadIdx.x >> 7);
        uint32_t g_dst_w = (dim_iw << 7) + (threadIdx.x & 127);
        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);

        if(g_dst_h < height && g_dst_w < width)
        {
            dst[dst_index] = v_buf[0];
        }
        if((g_dst_h + 2) < height && g_dst_w < width)
        {
            dst[dst_index + 2] = v_buf[1];
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_4x64(T* dst,
                                              T* src,
                                              uint32_t height,
                                              uint32_t width,
                                              uint32_t dim_stride,
                                              uint32_t dim_total,
                                              uint32_t magic_h,
                                              uint32_t shift_h,
                                              uint32_t magic_w,
                                              uint32_t shift_w)
{
    uint32_t h_dim = (height + 63) >> 6;
    uint32_t w_dim = (width + 3) >> 2;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t g_src_w = (dim_iw << 2) + (threadIdx.x >> 6);
        uint32_t g_src_h = (dim_ih << 6) + (threadIdx.x & 63);

        T v_buf;
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width)
        {
            v_buf = src[src_index];
        }

        uint32_t g_dst_h = (dim_ih << 6) + (threadIdx.x & 63);
        uint32_t g_dst_w = (dim_iw << 2) + (threadIdx.x >> 6);
        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);

        if(g_dst_h < height && g_dst_w < width)
        {
            dst[dst_index] = v_buf;
        }
    }
}

template <typename T>
inline __device__ void batched_transpose_64x4(T* dst,
                                              T* src,
                                              uint32_t height,
                                              uint32_t width,
                                              uint32_t dim_stride,
                                              uint32_t dim_total,
                                              uint32_t magic_h,
                                              uint32_t shift_h,
                                              uint32_t magic_w,
                                              uint32_t shift_w)
{
    uint32_t h_dim = (height + 3) >> 2;
    uint32_t w_dim = (width + 63) >> 6;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        uint32_t g_src_w = (dim_iw << 6) + (threadIdx.x & 63);
        uint32_t g_src_h = (dim_ih << 2) + (threadIdx.x >> 6);

        T v_buf;
        size_t src_index = static_cast<size_t>(dim_in) * height * width +
                           static_cast<size_t>(g_src_h) * width + static_cast<size_t>(g_src_w);
        if(g_src_h < height && g_src_w < width)
        {
            v_buf = src[src_index];
        }

        uint32_t g_dst_h = (dim_ih << 2) + (threadIdx.x >> 6);
        uint32_t g_dst_w = (dim_iw << 6) + (threadIdx.x & 63);
        size_t dst_index = static_cast<size_t>(dim_in) * width * height +
                           static_cast<size_t>(g_dst_w) * height + static_cast<size_t>(g_dst_h);

        if(g_dst_h < height && g_dst_w < width)
        {
            dst[dst_index] = v_buf;
        }
    }
}

// The helpers of the gfx1250 transposes added below live in namespace impl; the kernels
// (batched_transpose_reg / _lds / _lds_thin), the traits and the extern "C" instances are in the
// global namespace.
namespace impl {

typedef unsigned bt32_t;

// Pointer to the raw dwords of 4-byte elements (float, int32). The cast goes through void* so that
// it is one explicit reinterpretation (and not reported by cppcheck as a float* -> unsigned* cast).
template <typename T>
inline __device__ bt32_t* bt_raw_ptr(T* p)
{
    static_assert(sizeof(T) == sizeof(bt32_t), "4-byte elements");
    return static_cast<bt32_t*>(static_cast<void*>(p));
}

// ---- gfx1250 register transposes with 16-byte accesses (half / bfloat16 only).
// A thread owns a VS x VL block of the source (VS rows, VL columns), transposes it in registers (no
// LDS, no barriers) and stores VL destination row segments of VS elements. All blocks of a tile are
// handled by the 256 threads. A block is either completely inside or completely outside of the
// tensor because width % VL == 0 and height % VS == 0 (ediv), so one test per block is enough.
// Consecutive threads take consecutive blocks along the height (strided loads, coalesced stores).
#if defined(__gfx1250__)
// Buffer instructions: 32-bit byte offsets, an offset beyond the range makes a load return 0 and a
// store be dropped (that is how invalid blocks are skipped without branches). The resource covers
// one batch, so height * width * 2 has to be below 2^31 (checked by the heuristic).
typedef unsigned bt_u2_t __attribute__((ext_vector_type(2)));
typedef unsigned bt_u4_t __attribute__((ext_vector_type(4)));
constexpr uint32_t bt_oob = 0x7fffffffu;
template <typename T>
inline __device__ __amdgpu_buffer_rsrc_t bt_make_rsrc(const T* p, uint32_t bytes)
{
    // Last argument is gfx1250 specific
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<T*>(p), static_cast<short>(0), static_cast<int>(bytes), 0);
}

// Vector accesses of N elements of T (N * sizeof(T) = 2, 4, 8 or 16 bytes), one family for every
// kind of address:
//   bt_ld / bt_st   global memory or LDS (a generic pointer)
//   bt_bld / bt_bst buffer instructions (32-bit byte offset, an offset beyond the range makes a
//   load return 0 and a store be dropped)
// The data is in "raw" form: the bt_nw<T, N> dwords as they are in memory (a single 2-byte element
// is one zero-extended dword). The same form is used for all of them, so the data of a vector can
// be moved between global memory, buffers and LDS without conversions (LDS dwords: T = bt32_t);
// bt_unpack / bt_pack convert from / to one 32-bit register value per element (2-byte elements in
// the low half). Keeping the elements in 32-bit values makes the compiler use 32-bit operations
// instead of 16-bit register half moves, which are slower. The loads do not unpack, so all the
// loads of a block can be issued before the first one is used.
template <typename T, int N>
constexpr int bt_nw = (N * sizeof(T) + 3) / 4;
template <typename T, int N>
inline __device__ void bt_ld(const T* p, bt32_t* raw)
{
    constexpr int B = N * sizeof(T);
    static_assert(B == 2 || B == 4 || B == 8 || B == 16, "2, 4, 8 or 16 byte accesses");
    if constexpr(B == 2)
    {
        raw[0] =
            *reinterpret_cast<const ushort*>(p); // zero-extended: the upper 16 bits have to be 0
    }
    else if constexpr(B == 4)
    {
        raw[0] = *reinterpret_cast<const bt32_t*>(p);
    }
    else if constexpr(B == 8)
    {
        const bt_u2_t x = *reinterpret_cast<const bt_u2_t*>(p);
        raw[0]          = x.x;
        raw[1]          = x.y;
    }
    else if constexpr(B == 16)
    {
        const bt_u4_t x = *reinterpret_cast<const bt_u4_t*>(p);
        raw[0]          = x.x;
        raw[1]          = x.y;
        raw[2]          = x.z;
        raw[3]          = x.w;
    }
}
template <typename T, int N>
inline __device__ void bt_st(T* p, const bt32_t* raw)
{
    constexpr int B = N * sizeof(T);
    static_assert(B == 2 || B == 4 || B == 8 || B == 16, "2, 4, 8 or 16 byte accesses");
    if constexpr(B == 2)
    {
        *reinterpret_cast<ushort*>(p) = static_cast<ushort>(raw[0]);
    }
    else if constexpr(B == 4)
    {
        *reinterpret_cast<bt32_t*>(p) = raw[0];
    }
    else if constexpr(B == 8)
    {
        *reinterpret_cast<bt_u2_t*>(p) = bt_u2_t{raw[0], raw[1]};
    }
    else if constexpr(B == 16)
    {
        *reinterpret_cast<bt_u4_t*>(p) = bt_u4_t{raw[0], raw[1], raw[2], raw[3]};
    }
}
template <typename T, int N>
inline __device__ void bt_bld(__amdgpu_buffer_rsrc_t r, uint32_t off, bt32_t* raw)
{
    constexpr int B = N * sizeof(T);
    static_assert(B == 2 || B == 4 || B == 8 || B == 16, "2, 4, 8 or 16 byte accesses");
    if constexpr(B == 2)
    {
        raw[0] = static_cast<ushort>(__builtin_amdgcn_raw_buffer_load_b16(r, off, 0, 0));
    }
    else if constexpr(B == 4)
    {
        raw[0] = static_cast<bt32_t>(__builtin_amdgcn_raw_buffer_load_b32(r, off, 0, 0));
    }
    else if constexpr(B == 8)
    {
        const bt_u2_t x =
            __builtin_bit_cast(bt_u2_t, __builtin_amdgcn_raw_buffer_load_b64(r, off, 0, 0));
        raw[0] = x.x;
        raw[1] = x.y;
    }
    else if constexpr(B == 16)
    {
        const bt_u4_t x =
            __builtin_bit_cast(bt_u4_t, __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 0));
        raw[0] = x.x;
        raw[1] = x.y;
        raw[2] = x.z;
        raw[3] = x.w;
    }
}
template <typename T, int N>
inline __device__ void bt_bst(__amdgpu_buffer_rsrc_t r, uint32_t off, const bt32_t* raw)
{
    constexpr int B = N * sizeof(T);
    static_assert(B == 2 || B == 4 || B == 8 || B == 16, "2, 4, 8 or 16 byte accesses");
    if constexpr(B == 2)
    {
        __builtin_amdgcn_raw_buffer_store_b16(static_cast<short>(raw[0]), r, off, 0, 0);
    }
    else if constexpr(B == 4)
    {
        __builtin_amdgcn_raw_buffer_store_b32(static_cast<int>(raw[0]), r, off, 0, 0);
    }
    else if constexpr(B == 8)
    {
        __builtin_amdgcn_raw_buffer_store_b64(bt_u2_t{raw[0], raw[1]}, r, off, 0, 0);
    }
    else if constexpr(B == 16)
    {
        __builtin_amdgcn_raw_buffer_store_b128(
            bt_u4_t{raw[0], raw[1], raw[2], raw[3]}, r, off, 0, 0);
    }
}
// Splits raw dwords into one 32-bit register value per element. 4-byte elements and a single 2-byte
// element are already one value per dword. Otherwise a dword holds two consecutive 2-byte elements
// of a row (the one with the lower address in bits 15:0), so every dword gives two values.
template <typename T, int N>
inline __device__ void bt_unpack(const bt32_t* raw, bt32_t* a)
{
    if constexpr(sizeof(T) == 4 || N == 1)
    {
#pragma unroll
        for(int k = 0; k < N; k++)
            a[k] = raw[k];
    }
    else
    {
#pragma unroll
        for(int k = 0; k < N / 2; k++)
        {
            a[2 * k]     = raw[k] & 0xffffu; // element 2k: low half
            a[2 * k + 1] = raw[k] >> 16;     // element 2k + 1: high half
        }
    }
}
// Inverse of bt_unpack: packs one 32-bit value per element into raw dwords.
template <typename T, int N>
inline __device__ void bt_pack(const bt32_t* a, bt32_t* raw)
{
    if constexpr(sizeof(T) == 4 || N == 1)
    {
#pragma unroll
        for(int k = 0; k < N; k++)
            raw[k] = a[k];
    }
    else
    {
#pragma unroll
        for(int k = 0; k < N / 2; k++)
            raw[k] = a[2 * k] | (a[2 * k + 1] << 16);
    }
}
// Packs the 2-byte elements of one column from two rows into one dword (v_perm_b32), without
// unpacking them: lo / hi are dwords of the two rows, each holding two consecutive columns (even
// column in bits 15:0). odd = 0 takes the even column of both rows (lo[15:0] | hi[15:0] << 16), odd
// = 1 the odd column (lo[31:16] | hi[31:16] << 16). The result is the destination dword for rows
// (r, r + 1) of that column. odd is a constant after unrolling, so the selector is an immediate.
inline __device__ bt32_t bt_perm2(bt32_t lo, bt32_t hi, int odd)
{
    return __builtin_amdgcn_perm(hi, lo, odd ? 0x07060302u : 0x05040100u);
}
#endif

} // namespace impl

// Register transpose (no LDS, no barriers) of a TH x TW tile for 2 or 4 byte elements T. A thread
// owns a VS x VL block (VS rows, VL columns): it loads VS row segments of VL elements, transposes
// them in registers and stores VL destination row segments of VS elements. Consecutive threads take
// consecutive row blocks. Requires width % VL == 0 and height % VS == 0 (ediv), so a block is
// either completely inside or completely outside of the tensor. BUF selects buffer instructions
// (out of range blocks are dropped by the hardware, no branch); they need height * width * ES <
// 2^31 (checked by the heuristic). BLOCK is the number of threads per block.
template <typename T, int VL, int VS, int TH, int TW, bool BUF, int BLOCK>
inline __device__ void batched_transpose_reg(T* dst,
                                             T* src,
                                             uint32_t height,
                                             uint32_t width,
                                             uint32_t dim_stride,
                                             uint32_t dim_total,
                                             uint32_t magic_h,
                                             uint32_t shift_h,
                                             uint32_t magic_w,
                                             uint32_t shift_w)
{
#if defined(__gfx1250__)
    using namespace impl;
    static_assert(sizeof(T) == 2 || sizeof(T) == 4, "2 or 4 byte elements");
    static_assert(TH % VS == 0 && TW % VL == 0, "the tile has to be a multiple of the blocks");
    constexpr uint32_t ES   = sizeof(T); // 32-bit offsets (no 64-bit size_t arithmetic)
    constexpr uint32_t nrb  = TH / VS;
    constexpr uint32_t nblk = nrb * (TW / VL);
    const uint32_t h_dim    = (height + TH - 1) / TH;
    const uint32_t w_dim    = (width + TW - 1) / TW;
    const uint32_t bytes    = height * width * ES;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        const uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        const uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        const uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        const uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        const T* src_b                  = src + static_cast<size_t>(dim_in) * height * width;
        T* dst_b                        = dst + static_cast<size_t>(dim_in) * width * height;
        const __amdgpu_buffer_rsrc_t rs = bt_make_rsrc(src_b, bytes);
        const __amdgpu_buffer_rsrc_t rd = bt_make_rsrc(dst_b, bytes);

        for(uint32_t bi = threadIdx.x; bi < nblk; bi += BLOCK)
        {
            const uint32_t r0 = dim_ih * TH + (bi % nrb) * VS;
            const uint32_t c0 = dim_iw * TW + (bi / nrb) * VL;
            const bool valid  = (r0 < height) & (c0 < width);
            if constexpr(!BUF)
            {
                if(!valid)
                    continue;
            }
            bt32_t raw[VS][bt_nw<T, VL>];
#pragma unroll
            for(int j = 0; j < VS; j++)
            {
                if constexpr(BUF)
                    bt_bld<T, VL>(rs, valid ? ((r0 + j) * width + c0) * ES : bt_oob, raw[j]);
                else
                    bt_ld<T, VL>(src_b + static_cast<size_t>(r0 + j) * width + c0, raw[j]);
            }
            if constexpr(sizeof(T) == 2 && VL % 2 == 0 && VS % 2 == 0)
            {
                // 2-byte elements stay packed: every output dword takes one half of two row dwords
#pragma unroll
                for(int i = 0; i < VL; i++)
                {
                    bt32_t w[VS / 2];
#pragma unroll
                    for(int k = 0; k < VS / 2; k++)
                        w[k] = bt_perm2(raw[2 * k][i / 2], raw[2 * k + 1][i / 2], i % 2);
                    if constexpr(BUF)
                        bt_bst<T, VS>(rd, valid ? ((c0 + i) * height + r0) * ES : bt_oob, w);
                    else
                        bt_st<T, VS>(dst_b + static_cast<size_t>(c0 + i) * height + r0, w);
                    // The compiler groups consecutive stores into a hardware clause that issues
                    // them back to back. On tensors that stay in the cache this is up to 2x slower
                    // than stores that other waves can interleave with, a no-op in between prevents
                    // it.
                    asm volatile("s_nop 0" ::: "memory");
                }
            }
            else
            {
                bt32_t a[VS][VL];
#pragma unroll
                for(int j = 0; j < VS; j++)
                    bt_unpack<T, VL>(raw[j], a[j]);
#pragma unroll
                for(int i = 0; i < VL; i++)
                {
                    bt32_t o[VS];
#pragma unroll
                    for(int j = 0; j < VS; j++)
                        o[j] = a[j][i];
                    bt32_t w[bt_nw<T, VS>];
                    bt_pack<T, VS>(o, w);
                    if constexpr(BUF)
                        bt_bst<T, VS>(rd, valid ? ((c0 + i) * height + r0) * ES : bt_oob, w);
                    else
                        bt_st<T, VS>(dst_b + static_cast<size_t>(c0 + i) * height + r0, w);
                }
            }
        }
    }
#else
    (void)dst;
    (void)src;
    (void)height;
    (void)width;
    (void)dim_stride;
    (void)dim_total;
    (void)magic_h;
    (void)shift_h;
    (void)magic_w;
    (void)shift_w;
#endif
}

// LDS transpose of a TH x TW tile for 2 or 4 byte elements T: rows are read as VL-element vectors,
// columns are written as VS-element vectors; same requirements and meaning of BUF / BLOCK as
// batched_transpose_reg. 2-byte elements are kept in LDS as dwords holding two consecutive rows (TH
// and VS have to be even).
template <typename T, int VL, int VS, int TH, int TW, bool BUF, int BLOCK>
inline __device__ void batched_transpose_lds(T* dst,
                                             T* src,
                                             uint32_t height,
                                             uint32_t width,
                                             uint32_t dim_stride,
                                             uint32_t dim_total,
                                             uint32_t magic_h,
                                             uint32_t shift_h,
                                             uint32_t magic_w,
                                             uint32_t shift_w)
{
#if defined(__gfx1250__)
    using namespace impl;
    static_assert(sizeof(T) == 2 || sizeof(T) == 4, "2 or 4 byte elements");
    constexpr bool PAIR   = sizeof(T) == 2;
    constexpr uint32_t ES = sizeof(T); // 32-bit offsets (no 64-bit size_t arithmetic)
    static_assert(!PAIR || (TH % 2 == 0 && VS % 2 == 0), "2-byte elements are stored as row pairs");
    static_assert(TH % VS == 0 && TW % VL == 0, "the tile has to be a multiple of the vectors");
    constexpr int smem_stride = (PAIR ? TH / 2 : TH) + 4; // dwords per tile column (+ padding)
    constexpr int W           = PAIR ? VS / 2 : VS;       // dwords per column vector
    __shared__ __attribute__((aligned(16))) bt32_t smem[TW * smem_stride];

    const uint32_t h_dim = (height + TH - 1) / TH;
    const uint32_t w_dim = (width + TW - 1) / TW;
    const uint32_t bytes = height * width * ES;

    for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
    {
        const uint32_t dim_ih_tmp = magic_div_u32(dim_id, magic_w, shift_w);
        const uint32_t dim_iw     = dim_id - dim_ih_tmp * w_dim;
        const uint32_t dim_in     = magic_div_u32(dim_ih_tmp, magic_h, shift_h);
        const uint32_t dim_ih     = dim_ih_tmp - dim_in * h_dim;

        const uint32_t h0               = dim_ih * TH;
        const uint32_t w0               = dim_iw * TW;
        const T* src_b                  = src + static_cast<size_t>(dim_in) * height * width;
        T* dst_b                        = dst + static_cast<size_t>(dim_in) * width * height;
        const __amdgpu_buffer_rsrc_t rs = bt_make_rsrc(src_b, bytes);
        const __amdgpu_buffer_rsrc_t rd = bt_make_rsrc(dst_b, bytes);

        __syncthreads();

        constexpr uint32_t n_cg = TW / VL;
        constexpr uint32_t n_ld = (PAIR ? TH / 2 : TH) * n_cg;
        for(uint32_t it = threadIdx.x; it < n_ld; it += BLOCK)
        {
            const uint32_t cg = it % n_cg;
            const uint32_t rp = it / n_cg; // row (pair) index
            const uint32_t r  = h0 + (PAIR ? 2 * rp : rp);
            const uint32_t c  = w0 + cg * VL;
            constexpr int NW  = bt_nw<T, VL>;
            bt32_t ra[NW];
            bt32_t rb[NW];
            const bool va = (r < height) & (c < width);
            if constexpr(BUF)
                bt_bld<T, VL>(rs, va ? (r * width + c) * ES : bt_oob, ra);
            else if(va)
                bt_ld<T, VL>(src_b + static_cast<size_t>(r) * width + c, ra);
            else
            {
#pragma unroll
                for(int e = 0; e < NW; e++)
                    ra[e] = 0;
            }
            if constexpr(PAIR)
            {
                const bool vb = (r + 1 < height) & (c < width);
                if constexpr(BUF)
                    bt_bld<T, VL>(rs, vb ? ((r + 1) * width + c) * ES : bt_oob, rb);
                else if(vb)
                    bt_ld<T, VL>(src_b + static_cast<size_t>(r + 1) * width + c, rb);
                else
                {
#pragma unroll
                    for(int e = 0; e < NW; e++)
                        rb[e] = 0;
                }
            }
            bt32_t ea[VL];
            bt32_t eb[VL];
            bt_unpack<T, VL>(ra, ea);
            if constexpr(PAIR)
            {
                bt_unpack<T, VL>(rb, eb);
#pragma unroll
                for(int e = 0; e < VL; e++)
                    smem[(cg * VL + e) * smem_stride + rp] = ea[e] | (eb[e] << 16);
            }
            else
            {
#pragma unroll
                for(int e = 0; e < VL; e++)
                    smem[(cg * VL + e) * smem_stride + rp] = ea[e];
            }
        }

        __syncthreads();

        constexpr uint32_t n_rg = TH / VS;
        for(uint32_t it = threadIdx.x; it < TW * n_rg; it += BLOCK)
        {
            const uint32_t rg = it % n_rg;
            const uint32_t c  = it / n_rg;
            const uint32_t dc = w0 + c;
            const uint32_t dr = h0 + rg * VS;
            const bool valid  = (dc < width) & (dr < height);
            bt32_t a[W];
#pragma unroll
            for(int k = 0; k < W; k++)
                a[k] = smem[c * smem_stride + rg * W + k];
            if constexpr(BUF)
            {
                bt_bst<T, VS>(rd, valid ? (dc * height + dr) * ES : bt_oob, a);
            }
            else
            {
                if(valid)
                    bt_st<T, VS>(dst_b + static_cast<size_t>(dc) * height + dr, a);
            }
        }
    }
#else
    (void)dst;
    (void)src;
    (void)height;
    (void)width;
    (void)dim_stride;
    (void)dim_total;
    (void)magic_h;
    (void)shift_h;
    (void)magic_w;
    (void)shift_w;
#endif
}

#define DEFINE_BATCHED_TRANSPOSE_REG_TRAIT(trait, ...)                                           \
    template <typename T>                                                                        \
    inline __device__ void batched_transpose_##trait(T* /*dst*/,                                 \
                                                     T* /*src*/,                                 \
                                                     uint32_t /*height*/,                        \
                                                     uint32_t /*width*/,                         \
                                                     uint32_t /*dim_stride*/,                    \
                                                     uint32_t /*dim_total*/,                     \
                                                     uint32_t /*magic_h*/,                       \
                                                     uint32_t /*shift_h*/,                       \
                                                     uint32_t /*magic_w*/,                       \
                                                     uint32_t /*shift_w*/)                       \
    {                                                                                            \
    }                                                                                            \
    template <>                                                                                  \
    inline __device__ void batched_transpose_##trait<ushort>(ushort * dst,                       \
                                                             ushort * src,                       \
                                                             uint32_t height,                    \
                                                             uint32_t width,                     \
                                                             uint32_t dim_stride,                \
                                                             uint32_t dim_total,                 \
                                                             uint32_t magic_h,                   \
                                                             uint32_t shift_h,                   \
                                                             uint32_t magic_w,                   \
                                                             uint32_t shift_w)                   \
    {                                                                                            \
        __VA_ARGS__(                                                                             \
            dst, src, height, width, dim_stride, dim_total, magic_h, shift_h, magic_w, shift_w); \
    }

#define DEFINE_BATCHED_TRANSPOSE_F32_TRAIT(trait, ...)                           \
    template <typename T>                                                        \
    inline __device__ void batched_transpose_##trait(T* /*dst*/,                 \
                                                     T* /*src*/,                 \
                                                     uint32_t /*height*/,        \
                                                     uint32_t /*width*/,         \
                                                     uint32_t /*dim_stride*/,    \
                                                     uint32_t /*dim_total*/,     \
                                                     uint32_t /*magic_h*/,       \
                                                     uint32_t /*shift_h*/,       \
                                                     uint32_t /*magic_w*/,       \
                                                     uint32_t /*shift_w*/)       \
    {                                                                            \
    }                                                                            \
    template <>                                                                  \
    inline __device__ void batched_transpose_##trait<float>(float* dst,          \
                                                            float* src,          \
                                                            uint32_t height,     \
                                                            uint32_t width,      \
                                                            uint32_t dim_stride, \
                                                            uint32_t dim_total,  \
                                                            uint32_t magic_h,    \
                                                            uint32_t shift_h,    \
                                                            uint32_t magic_w,    \
                                                            uint32_t shift_w)    \
    {                                                                            \
        __VA_ARGS__(impl::bt_raw_ptr(dst),                                       \
                    impl::bt_raw_ptr(src),                                       \
                    height,                                                      \
                    width,                                                       \
                    dim_stride,                                                  \
                    dim_total,                                                   \
                    magic_h,                                                     \
                    shift_h,                                                     \
                    magic_w,                                                     \
                    shift_w);                                                    \
    }

// ---- gfx1250 LDS transposes for 2-byte elements with one position per thread (position-per-thread
// family). One dimension (the "small" one) is moved as 16-byte or smaller vectors through LDS, the
// other one (the "positions", any size) is processed with coalesced 2-byte accesses, one position
// per thread, so an odd size needs no special handling. Used e.g. for the channel dimension of an
// NCHW <-> NHWC transpose.
//   VL == 1 (cols): src[b][H][W] -> dst[b][W][H], tiles of TW positions (columns, a multiple of
//   BLOCK) x TH rows, loads 1 element, stores VS elements; requires height % TH == 0 (ediv 1 x TH),
//   TH % VS == 0
//   VS == 1 (rows): tiles of TW columns x TH positions (rows, a multiple of BLOCK), loads VL
//   elements, stores 1 element; requires width % TW == 0 (ediv TW x 1), TW % VL == 0
// BUF selects buffer instructions for the global accesses (they need height * width * 2 < 2^31).
// There are no branches between a fast and a general path: the tile has to divide the small
// dimension. LDS transpose for thin matrices: one side is a small channel count (TH, 8..64) that is
// kept whole in the tile, the other side is long (TW, many positions). Each thread owns one
// position and moves TH contiguous elements of it. Exactly one of VL/VS is 1:
//   VL == 1: tile = TW positions (columns) x TH rows,  requires height % TH == 0
//   VS == 1: tile = TW columns x TH positions (rows),  requires width % TW == 0
template <int VL, int VS, int TH, int TW, bool BUF, int BLOCK>
inline __device__ void batched_transpose_lds_thin(ushort* dst,
                                                  ushort* src,
                                                  uint32_t height,
                                                  uint32_t width,
                                                  uint32_t dim_stride,
                                                  uint32_t dim_total,
                                                  uint32_t magic_h,
                                                  uint32_t shift_h,
                                                  uint32_t magic_w,
                                                  uint32_t shift_w)
{
    static_assert((VL == 1) != (VS == 1), "exactly one of the vector sizes has to be 1");
#if defined(__gfx1250__)
    using namespace impl;
    constexpr int VEC    = (VL == 1) ? VS : VL;
    constexpr int NW     = bt_nw<ushort, VEC>; // dwords per vector
    const uint32_t bytes = height * width * 2;
    if constexpr(VL == 1)
    {
        constexpr int G = TH / VS; // vectors per position
        static_assert(TW % BLOCK == 0, "TW has to be a multiple of the block size");
        constexpr int P = TW / BLOCK; // positions per thread
        __shared__ __attribute__((aligned(16))) bt32_t smem[TW * G * NW];
        const uint32_t dim_w = (width + TW - 1) / TW;
        const uint32_t dim_h = height / TH;
        for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
        {
            const uint32_t t  = magic_div_u32(dim_id, magic_w, shift_w);
            const uint32_t iw = dim_id - t * dim_w;
            const uint32_t b  = magic_div_u32(t, magic_h, shift_h);
            const uint32_t ih = t - b * dim_h;
            const uint32_t w0 = iw * TW;
            const uint32_t r0 = ih * TH;
            const uint32_t n  = (width - w0) < static_cast<uint32_t>(TW) ? (width - w0) : TW;
            const __amdgpu_buffer_rsrc_t rs =
                bt_make_rsrc(src + static_cast<size_t>(b) * height * width, bytes);
            const __amdgpu_buffer_rsrc_t rd =
                bt_make_rsrc(dst + static_cast<size_t>(b) * width * height, bytes);
            __syncthreads();
#pragma unroll
            for(int p = 0; p < P; p++)
            {
                const uint32_t pos = p * BLOCK + threadIdx.x;
                const uint32_t hw  = w0 + pos;
                if constexpr(!BUF)
                {
                    if(hw >= width)
                        continue;
                }
                const ushort* s_base = src + (static_cast<size_t>(b) * height + r0) * width + hw;
#pragma unroll
                for(int g = 0; g < G; g++)
                {
                    // VS elements of one position, one per row
                    bt32_t a[VS];
#pragma unroll
                    for(int k = 0; k < VS; k++)
                    {
                        if constexpr(BUF)
                            bt_bld<ushort, 1>(rs,
                                              hw < width ? ((r0 + g * VS + k) * width + hw) * 2
                                                         : bt_oob,
                                              &a[k]);
                        else
                            bt_ld<ushort, 1>(s_base + static_cast<size_t>(g * VS + k) * width,
                                             &a[k]);
                    }
                    bt32_t w[NW];
                    bt_pack<ushort, VS>(a, w);
                    bt_st<bt32_t, NW>(&smem[(pos * G + g) * NW], w);
                }
            }
            __syncthreads();
            for(uint32_t i = threadIdx.x; i < n * G; i += BLOCK)
            {
                const uint32_t pos = i / G;
                const uint32_t g   = i - pos * G;
                bt32_t w[NW];
                bt_ld<bt32_t, NW>(&smem[i * NW], w);
                if constexpr(BUF)
                    bt_bst<ushort, VS>(rd, ((w0 + pos) * height + r0 + g * VS) * 2, w);
                else
                    bt_st<ushort, VS>(dst + (static_cast<size_t>(b) * width + w0 + pos) * height +
                                          r0 + g * VS,
                                      w);
            }
        }
    }
    else
    {
        constexpr int G = TW / VL; // vectors per position
        static_assert(TH % BLOCK == 0, "TH has to be a multiple of the block size");
        constexpr int P = TH / BLOCK; // positions per thread
        __shared__ __attribute__((aligned(16))) bt32_t smem[TH * G * NW];
        const uint32_t dim_w = width / TW;
        const uint32_t dim_h = (height + TH - 1) / TH;
        for(uint32_t dim_id = blockIdx.x; dim_id < dim_total; dim_id += dim_stride)
        {
            const uint32_t t  = magic_div_u32(dim_id, magic_w, shift_w);
            const uint32_t iw = dim_id - t * dim_w;
            const uint32_t b  = magic_div_u32(t, magic_h, shift_h);
            const uint32_t ih = t - b * dim_h;
            const uint32_t h0 = ih * TH;
            const uint32_t c0 = iw * TW;
            const uint32_t n  = (height - h0) < static_cast<uint32_t>(TH) ? (height - h0) : TH;
            const __amdgpu_buffer_rsrc_t rs =
                bt_make_rsrc(src + static_cast<size_t>(b) * height * width, bytes);
            const __amdgpu_buffer_rsrc_t rd =
                bt_make_rsrc(dst + static_cast<size_t>(b) * width * height, bytes);
            __syncthreads();
            for(uint32_t i = threadIdx.x; i < n * G; i += BLOCK)
            {
                const uint32_t pos = i / G;
                const uint32_t g   = i - pos * G;
                bt32_t w[NW];
                if constexpr(BUF)
                    bt_bld<ushort, VL>(rs, ((h0 + pos) * width + c0 + g * VL) * 2, w);
                else
                    bt_ld<ushort, VL>(src + (static_cast<size_t>(b) * height + h0 + pos) * width +
                                          c0 + g * VL,
                                      w);
                bt_st<bt32_t, NW>(&smem[i * NW], w);
            }
            __syncthreads();
#pragma unroll
            for(int p = 0; p < P; p++)
            {
                const uint32_t pos = p * BLOCK + threadIdx.x;
                if constexpr(!BUF)
                {
                    if(pos >= n)
                        continue;
                }
                ushort* d_base = dst + (static_cast<size_t>(b) * width + c0) * height + h0 + pos;
#pragma unroll
                for(int g = 0; g < G; g++)
                {
                    // VL elements of one position, one per destination row
                    bt32_t w[NW];
                    bt_ld<bt32_t, NW>(&smem[(pos * G + g) * NW], w);
                    bt32_t a[VL];
                    bt_unpack<ushort, VL>(w, a);
#pragma unroll
                    for(int k = 0; k < VL; k++)
                    {
                        if constexpr(BUF)
                            bt_bst<ushort, 1>(rd,
                                              pos < n ? ((c0 + g * VL + k) * height + h0 + pos) * 2
                                                      : bt_oob,
                                              &a[k]);
                        else
                            bt_st<ushort, 1>(d_base + static_cast<size_t>(g * VL + k) * height,
                                             &a[k]);
                    }
                }
            }
        }
    }
#else
    (void)dst;
    (void)src;
    (void)height;
    (void)width;
    (void)dim_stride;
    (void)dim_total;
    (void)magic_h;
    (void)shift_h;
    (void)magic_w;
    (void)shift_w;
#endif
}

// ---- instances. Template parameters of the kernels: element type, VL / VS = elements per global
// load (along width) / store (along height), TH x TW = tile (rows x columns), BUF = buffer
// instructions, BLOCK = threads. Instance names:
// batched_transpose_<TW>x<TH>_pack_<VL>x<VS>_ediv_<width divisor>x<height divisor>. 2-byte (half /
// bfloat16)
DEFINE_BATCHED_TRANSPOSE_REG_TRAIT(
    32x64_pack_1x8_ediv_1x8,
    batched_transpose_reg<ushort, 1, 8, 64, 32, false, BATCHED_TRANSPOSE_BLOCK_SIZE>)
DEFINE_BATCHED_TRANSPOSE_REG_TRAIT(
    128x128_pack_8x8_ediv_8x8,
    batched_transpose_reg<ushort, 8, 8, 128, 128, true, BATCHED_TRANSPOSE_BLOCK_SIZE>)
DEFINE_BATCHED_TRANSPOSE_REG_TRAIT(
    128x64_pack_8x8_ediv_8x8,
    batched_transpose_reg<ushort, 8, 8, 64, 128, true, BATCHED_TRANSPOSE_BLOCK_SIZE>)
DEFINE_BATCHED_TRANSPOSE_REG_TRAIT(
    64x128_pack_8x4_ediv_8x4,
    batched_transpose_lds<ushort, 8, 4, 128, 64, true, BATCHED_TRANSPOSE_BLOCK_SIZE>)

// 4-byte (float); the 64x64 trait already exists for half: float specialization
template <>
inline __device__ void batched_transpose_64x64_pack_4x4_ediv_4x4<float>(float* dst,
                                                                        float* src,
                                                                        uint32_t height,
                                                                        uint32_t width,
                                                                        uint32_t dim_stride,
                                                                        uint32_t dim_total,
                                                                        uint32_t magic_h,
                                                                        uint32_t shift_h,
                                                                        uint32_t magic_w,
                                                                        uint32_t shift_w)
{
    batched_transpose_lds<impl::bt32_t, 4, 4, 64, 64, false, BATCHED_TRANSPOSE_BLOCK_SIZE>(
        impl::bt_raw_ptr(dst),
        impl::bt_raw_ptr(src),
        height,
        width,
        dim_stride,
        dim_total,
        magic_h,
        shift_h,
        magic_w,
        shift_w);
}
DEFINE_BATCHED_TRANSPOSE_F32_TRAIT(
    64x128_pack_4x4_ediv_4x4,
    batched_transpose_lds<impl::bt32_t, 4, 4, 128, 64, false, BATCHED_TRANSPOSE_BLOCK_SIZE>)
DEFINE_BATCHED_TRANSPOSE_F32_TRAIT(
    64x32_pack_4x4_ediv_4x4,
    batched_transpose_reg<impl::bt32_t, 4, 4, 32, 64, true, BATCHED_TRANSPOSE_BLOCK_SIZE>)
DEFINE_BATCHED_TRANSPOSE_F32_TRAIT(
    32x256_pack_4x2_ediv_4x2,
    batched_transpose_lds<impl::bt32_t, 4, 2, 256, 32, true, BATCHED_TRANSPOSE_BLOCK_SIZE>)
DEFINE_BATCHED_TRANSPOSE_F32_TRAIT(
    16x32_pack_1x4_ediv_1x4,
    batched_transpose_reg<impl::bt32_t, 1, 4, 32, 16, true, BATCHED_TRANSPOSE_BLOCK_SIZE>)

// instances (tile width x height, load / store elements, ediv = required divisibility of width x
// height)
DEFINE_BATCHED_TRANSPOSE_REG_TRAIT(
    256x32_pack_1x8_ediv_1x32,
    batched_transpose_lds_thin<1, 8, 32, 256, false, BATCHED_TRANSPOSE_BLOCK_SIZE>)
DEFINE_BATCHED_TRANSPOSE_REG_TRAIT(
    512x64_pack_1x8_ediv_1x64,
    batched_transpose_lds_thin<1, 8, 64, 512, false, BATCHED_TRANSPOSE_BLOCK_SIZE>)
DEFINE_BATCHED_TRANSPOSE_REG_TRAIT(
    32x512_pack_8x1_ediv_32x1,
    batched_transpose_lds_thin<8, 1, 512, 32, false, BATCHED_TRANSPOSE_BLOCK_SIZE>)

#define DEFINE_BATCHED_TRANSPOSE_KERNEL(                                                       \
    tile_trait, accept_data_type, cast_data_type, lb_threads_per_block, lb_blocks_per_cu)      \
    extern "C" __global__ void __launch_bounds__(lb_threads_per_block, lb_blocks_per_cu)       \
        batched_transpose_##tile_trait##_##accept_data_type(void* dst,                         \
                                                            void* src,                         \
                                                            uint32_t height,                   \
                                                            uint32_t width,                    \
                                                            uint32_t dim_stride,               \
                                                            uint32_t dim_total,                \
                                                            uint32_t magic_h,                  \
                                                            uint32_t shift_h,                  \
                                                            uint32_t magic_w,                  \
                                                            uint32_t shift_w)                  \
    {                                                                                          \
        batched_transpose_##tile_trait<cast_data_type>(reinterpret_cast<cast_data_type*>(dst), \
                                                       reinterpret_cast<cast_data_type*>(src), \
                                                       height,                                 \
                                                       width,                                  \
                                                       dim_stride,                             \
                                                       dim_total,                              \
                                                       magic_h,                                \
                                                       shift_h,                                \
                                                       magic_w,                                \
                                                       shift_w);                               \
    }

DEFINE_BATCHED_TRANSPOSE_KERNEL(16x16, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(16x16, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(16x16, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(32x16, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(32x16, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(32x16, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(16x32, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(16x32, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(16x32, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(32x32, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(32x32, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(32x32, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(4x256, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(4x256, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(4x256, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(256x4, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(256x4, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(256x4, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(4x128, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(4x128, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(4x128, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(128x4, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(128x4, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(128x4, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(4x64, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(4x64, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(4x64, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(64x4, dword, float, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(64x4, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(64x4, byte, uchar, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(
    32x32_pack_2x2_ediv_2x2, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(
    32x32_pack_2x2_ediv_1x2, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(
    32x32_pack_2x2_ediv_2x1, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(
    32x32_pack_2x2_ediv_1x1, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(
    64x32_pack_4x2_ediv_4x2, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(
    64x32_pack_4x2_ediv_2x2, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(
    64x32_pack_4x2_ediv_2x1, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(
    32x64_pack_2x4_ediv_2x4, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(
    32x64_pack_2x4_ediv_2x2, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(
    32x64_pack_2x4_ediv_1x2, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(
    16x64_pack_1x4_ediv_1x2, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(
    64x16_pack_4x1_ediv_2x1, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(
    64x64_pack_4x4_ediv_4x4, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(
    64x64_pack_4x4_ediv_2x2, half, ushort, 256, BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(32x64_pack_1x8_ediv_1x8,
                                half,
                                ushort,
                                BATCHED_TRANSPOSE_BLOCK_SIZE,
                                BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(128x128_pack_8x8_ediv_8x8,
                                half,
                                ushort,
                                BATCHED_TRANSPOSE_BLOCK_SIZE,
                                BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(128x64_pack_8x8_ediv_8x8,
                                half,
                                ushort,
                                BATCHED_TRANSPOSE_BLOCK_SIZE,
                                BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(64x128_pack_8x4_ediv_8x4,
                                half,
                                ushort,
                                BATCHED_TRANSPOSE_BLOCK_SIZE,
                                BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(64x64_pack_4x4_ediv_4x4,
                                dword,
                                float,
                                BATCHED_TRANSPOSE_BLOCK_SIZE,
                                BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(64x128_pack_4x4_ediv_4x4,
                                dword,
                                float,
                                BATCHED_TRANSPOSE_BLOCK_SIZE,
                                BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(64x32_pack_4x4_ediv_4x4,
                                dword,
                                float,
                                BATCHED_TRANSPOSE_BLOCK_SIZE,
                                BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(32x256_pack_4x2_ediv_4x2,
                                dword,
                                float,
                                BATCHED_TRANSPOSE_BLOCK_SIZE,
                                BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(16x32_pack_1x4_ediv_1x4,
                                dword,
                                float,
                                BATCHED_TRANSPOSE_BLOCK_SIZE,
                                BATCHED_TRANSPOSE_OCCUPANCY)

DEFINE_BATCHED_TRANSPOSE_KERNEL(256x32_pack_1x8_ediv_1x32,
                                half,
                                ushort,
                                BATCHED_TRANSPOSE_BLOCK_SIZE,
                                BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(512x64_pack_1x8_ediv_1x64,
                                half,
                                ushort,
                                BATCHED_TRANSPOSE_BLOCK_SIZE,
                                BATCHED_TRANSPOSE_OCCUPANCY)
DEFINE_BATCHED_TRANSPOSE_KERNEL(32x512_pack_8x1_ediv_32x1,
                                half,
                                ushort,
                                BATCHED_TRANSPOSE_BLOCK_SIZE,
                                BATCHED_TRANSPOSE_OCCUPANCY)
