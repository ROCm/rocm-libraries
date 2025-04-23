
/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef MIVISIONX_HIP_COMMON_FUNCS_H
#define MIVISIONX_HIP_COMMON_FUNCS_H

#include "hip/hip_runtime.h"

#define MASK_EARLY_EXIT 4369
#define HIPSELECT(a, b, c)  (c ? b : a)

const float pi = 0x1.921fb6p+1f;
const float inv_pi = 1 / pi;

typedef struct d_uint6 {
  uint data[6];
} d_uint6;

typedef struct d_uint8 {
  uint data[8];
} d_uint8;

typedef struct d_float8 {
  float data[8];
} d_float8;

typedef struct d_affine_matrix_t {
    float m[3][2];
} d_affine_matrix_t;

typedef struct d_perspective_matrix_t {
    float m[3][3];
} d_perspective_matrix_t;

typedef struct d_KeyPt {
    int x;
    int y;
    float strength;
    float scale;
    float orientation;
    int tracking_status;
    float error;
} d_KeyPt;

// common device kernels

__device__ __forceinline__ uint hip_pack(float4 src) {
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

__device__ __forceinline__ float hip_unpack0(uint src) {
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float hip_unpack1(uint src) {
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float hip_unpack2(uint src) {
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float hip_unpack3(uint src) {
    return (float)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float4 hip_unpack(uint src) {
    return make_float4(hip_unpack0(src), hip_unpack1(src), hip_unpack2(src), hip_unpack3(src));
}

__device__ __forceinline__ float hip_dot2(float2 src0, float2 src1) {
    return fmaf(src0.y, src1.y, src0.x * src1.x);
}

__device__ __forceinline__ float hip_dot3(float3 src0, float3 src1) {
    return fmaf(src0.z, src1.z, fmaf(src0.y, src1.y, src0.x * src1.x));
}

__device__ __forceinline__ float hip_dot4(float4 src0, float4 src1) {
    return fmaf(src0.w, src1.w, fmaf(src0.z, src1.z, fmaf(src0.y, src1.y, src0.x * src1.x)));
}

__device__ __forceinline__ float hip_max3(float src0, float src1, float src2) {
    return fmaxf(src0, fmaxf(src1, src2));
}

__device__ __forceinline__ float hip_min3(float src0, float src1, float src2) {
    return fminf(src0, fminf(src1, src2));
}

__device__ __forceinline__ float hip_median3(float src0, float src1, float src2) {
    return __builtin_amdgcn_fmed3f(src0, src1, src2);
}

__device__ __forceinline__ uint hip_lerp(uint src0, uint src1, uint src2) {
    return __builtin_amdgcn_lerp(src0, src1, src2);
}

__device__ __forceinline__ uint hip_sad(uint a, uint b, uint c) {
    return __builtin_amdgcn_sad_u8(a, b, c);
}

__device__ __forceinline__ uint hip_bytealign(uint a, uint b, uint c) {
    return __builtin_amdgcn_alignbyte(a, b, c);
}

__device__ __forceinline__ float4 hip_fabs4(float4 src) {
    return make_float4(fabsf(src.x), fabsf(src.y), fabsf(src.z), fabsf(src.w));
}

__device__ __forceinline__ float4 hip_floorf4(float4 src) {
    return make_float4(floorf(src.x), floorf(src.y), floorf(src.z), floorf(src.w));
}

__device__ __forceinline__ uint hip_mul24(uint a, uint b) {
    return __ockl_mul24_u32(a, b);
}

__device__ __forceinline__ uint hip_mad24(uint a, uint b, uint c) {
    return ((a << 8) >> 8) * ((b << 8) >> 8) + c;
}

template<class T>
__device__ __forceinline__ T hip_clamp(T v, T lo, T hi) {
    return min(max(v, lo), hi);
}

__device__ __forceinline__ short hip_convert_short_rte(float a) {
    a = rint(a);
    return (short)a;
}

__device__ __forceinline__ short hip_convert_short_sat_rte(float a) {
    a = rint(a);
    return (short)hip_clamp(a, (float)INT16_MIN, (float)INT16_MAX);
}

__device__ __forceinline__ void hip_convert_U8_U1(uint2 *p0, unsigned char p1) {
    uint2 r;
    r.x  = (-(p1 &   1)) & 0x000000ff;
    r.x |= (-(p1 &   2)) & 0x0000ff00;
    r.x |= (-(p1 &   4)) & 0x00ff0000;
    r.x |= (-(p1 &   8)) & 0xff000000;
    r.y  = (-((p1 >> 4) & 1)) & 0x000000ff;
    r.y |= (-(p1 &  32)) & 0x0000ff00;
    r.y |= (-(p1 &  64)) & 0x00ff0000;
    r.y |= (-(p1 & 128)) & 0xff000000;
    *p0 = r;
}

__device__ __forceinline__ void hip_convert_U1_U8(unsigned char *p0, uint2 p1) {
    unsigned char r;
    r  =  p1.x        &   1;
    r |= (p1.x >>  7) &   2;
    r |= (p1.x >> 14) &   4;
    r |= (p1.x >> 21) &   8;
    r |= (p1.y <<  4) &  16;
    r |= (p1.y >>  3) &  32;
    r |= (p1.y >> 10) &  64;
    r |= (p1.y >> 17) & 128;
    *p0 = r;
}

__device__ __forceinline__ void hip_convert_U8_S16(uint2 *p0, int4 p1) {
    uint2 r;
    uint p2 = 16;
    r.x  = ((((int)p1.x)  << 16) >> p2) & 0xff;
    r.x |= ((((int)p1.x)         >> p2) & 0xff) <<  8;
    r.x |= (((((int)p1.y) << 16) >> p2) & 0xff) << 16;
    r.x |= ((((int)p1.y)         >> p2) & 0xff) << 24;
    r.y  = ((((int)p1.z)  << 16) >> p2) & 0xff;
    r.y |= ((((int)p1.z)         >> p2) & 0xff) <<  8;
    r.y |= (((((int)p1.w) << 16) >> p2) & 0xff) << 16;
    r.y |= ((((int)p1.w)         >> p2) & 0xff) << 24;
    *p0 = r;
}

__device__ __forceinline__ float hip_fract(float x, float *itpr) {
    *itpr = floorf(x);
    return fminf(x - floorf(x), 0x1.fffffep-1f);
}

__device__ __forceinline__ float hip_bilinear_sample(const uchar *p, uint ystride, uint xstride,
    float fy0, float fy1, int x, float fx0, float fx1) {
    float4 f;
    p += x;
    f.x = hip_unpack0((uint)p[0]);
    f.y = hip_unpack0((uint)p[xstride]);
    p += ystride;
    f.z = hip_unpack0((uint)p[0]);
    f.w = hip_unpack0((uint)p[xstride]);
    f.x = fmaf(f.x, fx0, f.y * fx1);
    f.z = fmaf(f.z, fx0, f.w * fx1);
    f.x = fmaf(f.x, fy0, f.z * fy1);
    return f.x;
}

__device__ __forceinline__ float hip_bilinear_sample_FXY(const uchar *p, uint pSize, uint stride, float sx, float sy) {
    float fx0, fx1, fy0, fy1, ii;
    uint x, y;
    fx1 = hip_fract(sx, &ii);
    fx0 = 1.0f - fx1;
    x = (uint)ii;
    fy1 = hip_fract(sy, &ii);
    fy0 = 1.0f - fy1;
    y = (uint)ii;
    uint srcIdx = hip_mad24(stride, y, x);
    if (srcIdx < pSize)
        p += srcIdx;
    return hip_bilinear_sample(p, stride, 1, fy0, fy1, 0, fx0, fx1);
}

__device__ __forceinline__ float hip_bilinear_sample_FXY_constant_for_remap(const uchar *p, uint stride, uint width,
    uint height, float sx, float sy, uint borderValue) {

    float fx0, fx1, fy0, fy1, ii;
    int x, y;
    fx1 = hip_fract(sx, &ii);
    fx0 = 1.0f - fx1;
    x = (int)floorf(sx);
    fy1 = hip_fract(sy, &ii);
    fy0 = 1.0f - fy1;
    y = (int)floorf(sy);
    if (((uint)x) < width - 1 && ((uint)y) < height - 1) {
        p += y * stride;
        return hip_bilinear_sample(p, stride, 1, fy0, fy1, x, fx0, fx1);
    }
    else {
        return hip_unpack0(borderValue);
    }
}

__device__ __forceinline__ uint2 hip_clamp_pixel_coordinates_to_border(float f, uint limit, uint stride) {
    uint2 vstride;
    vstride.x = HIPSELECT((uint)f, 0u, f < 0);
    vstride.y = HIPSELECT(stride, 0u, f < 0);
    vstride.x = HIPSELECT(vstride.x, limit, f >= limit);
    vstride.y = HIPSELECT(vstride.y, 0u, f >= limit);
    return vstride;
}

__device__ __forceinline__ uint hip_sample_with_constant_border(const uchar *p, int x, int y, uint width,
    uint height, uint stride, uint borderValue) {

    uint pixelValue = borderValue;
    if (x >= 0 && y >= 0 && x < width && y < height) {
        pixelValue = p[y * stride + x];
    }
    return pixelValue;
}

__device__ __forceinline__ float hip_bilinear_sample_with_constant_border(const uchar *p, int x, int y, uint width,
    uint height, uint stride, float fx0, float fx1, float fy0, float fy1, uint borderValue) {

    float4 f;
    f.x = hip_unpack0(hip_sample_with_constant_border(p, x, y, width, height, stride, borderValue));
    f.y = hip_unpack0(hip_sample_with_constant_border(p, x + 1, y, width, height, stride, borderValue));
    f.z = hip_unpack0(hip_sample_with_constant_border(p, x, y + 1, width, height, stride, borderValue));
    f.w = hip_unpack0(hip_sample_with_constant_border(p, x + 1, y + 1, width, height, stride, borderValue));
    f.x = fmaf(f.x, fx0, f.y * fx1);
    f.z = fmaf(f.z, fx0, f.w * fx1);
    f.x = fmaf(f.x, fy0, f.z * fy1);
    return f.x;
}

__device__ __forceinline__ float hip_bilinear_sample_FXY_constant(const uchar *p, uint stride, uint width, uint height,
    float sx, float sy, uint borderValue) {

    float fx0, fx1, fy0, fy1, ii;
    int x, y;
    fx1 = hip_fract(sx, &ii);
    fx0 = 1.0f - fx1;
    x = (int)ii;
    fy1 = hip_fract(sy, &ii);
    fy0 = 1.0f - fy1;
    y = (int)ii;
    if (((uint)x) < width && ((uint)y) < height) {
        p += y * stride;
        return hip_bilinear_sample(p, stride, 1, fy0, fy1, x, fx0, fx1);
    }
    else {
        return hip_bilinear_sample_with_constant_border(p, x, y, width, height, stride, fx0, fx1, fy0, fy1, borderValue);
    }
}

__device__ __forceinline__ uint hip_canny_mag_phase_L1(float gx, float gy) {
    float dx = fabsf(gx);
    float dy = fabsf(gy);
    float dr = hip_min3((dx + dy), 16383.0f, 16383.0f);
    float d1 = dx * 0.4142135623730950488016887242097f;
    float d2 = dx * 2.4142135623730950488016887242097f;
    uint mp = HIPSELECT(1u, 3u, (gx * gy) < 0.0f);
    mp = HIPSELECT(mp, 0u, dy <= d1);
    mp = HIPSELECT(mp, 2u, dy >= d2);
    mp += (((uint)dr) << 2);
    return mp;
}

__device__ __forceinline__ uint hip_canny_mag_phase_L2(float gx, float gy) {
    float dx = fabsf(gx);
    float dy = fabsf(gy);
    float dr = hip_min3(__fsqrt_rn(fmaf(gy, gy, gx * gx)), 16383.0f, 16383.0f);
    float d1 = dx * 0.4142135623730950488016887242097f;
    float d2 = dx * 2.4142135623730950488016887242097f;
    uint mp = HIPSELECT(1u, 3u, (gx * gy) < 0.0f);
    mp = HIPSELECT(mp, 0u, dy <= d1);
    mp = HIPSELECT(mp, 2u, dy >= d2);
    mp += (((uint)dr) << 2);
    return mp;
}

__device__ __forceinline__ uint hip_canny_mag_phase_L1_7x7(float gx, float gy) {
    float dx = fabsf(gx);
    float dy = fabsf(gy);
    float dr = hip_min3((dx + dy) * 0.25f, 16383.0f, 16383.0f);
    float d1 = dx * 0.4142135623730950488016887242097f;
    float d2 = dx * 2.4142135623730950488016887242097f;
    uint mp = HIPSELECT(1u, 3u, (gx * gy) < 0.0f);
    mp = HIPSELECT(mp, 0u, dy <= d1);
    mp = HIPSELECT(mp, 2u, dy >= d2);
    mp += (((uint)dr) << 2);
    return mp;
}

__device__ __forceinline__ uint hip_canny_mag_phase_L2_7x7(float gx, float gy) {
    float dx = fabsf(gx);
    float dy = fabsf(gy);
    float dr = hip_min3(__fsqrt_rn(fmaf(gy, gy, gx * gx) * 0.0625f), 16383.0f, 16383.0f);
    float d1 = dx * 0.4142135623730950488016887242097f;
    float d2 = dx * 2.4142135623730950488016887242097f;
    uint mp = HIPSELECT(1u, 3u, (gx * gy) < 0.0f);
    mp = HIPSELECT(mp, 0u, dy <= d1);
    mp = HIPSELECT(mp, 2u, dy >= d2);
    mp += (((uint)dr) << 2);
    return mp;
}

__device__ __forceinline__ uint hip_bfe(uint src0, uint src1, uint src2) {
    return __builtin_amdgcn_ubfe(src0, src1, src2);
}

#endif //MIVISIONX_HIP_COMMON_FUNCS_H