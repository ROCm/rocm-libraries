/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <half/half.hpp>
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "rocal_hip_kernels.h"

typedef union { uint ui1[6];    uint2 ui2[3];                                                   }   d_uint6;
typedef union { float f1[8];    float2 f2[4];   float4 f4[2];                                   }   d_float8;
typedef union { float f1[12];   float4 f4[3];                                                   }   d_float12;
typedef union { float f1[16];   float4 f4[4];   d_float8 f8[2];                                 }   d_float16;
typedef union { float f1[24];   float2 f2[12];  float3 f3[8];   float4 f4[6];   d_float8 f8[3]; }   d_float24;
typedef struct { uint   data[ 6]; } d_uint6_s;

#define LOCAL_THREADS_X                 16                  // default rpp hip thread launch config - local threads x = 16
#define LOCAL_THREADS_Y                 16                  // default rpp hip thread launch config - local threads x = 16
#define LOCAL_THREADS_Z                  1                  // default rpp hip thread launch config - local threads x = 16

__global__ void __attribute__((visibility("default")))
Hip_CopyInt8ToNHWC_fp32(
    const unsigned char *inp_image_u8,
    void *output_tensor,
    unsigned int dst_buf_offset,
    uint4 nchw,
    uint2 out_dims,
    float3 multiplier,
    float3 offset,
    unsigned int reverse_channels) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const int W = nchw.w;
    const int H = nchw.z;
    const int C = nchw.y;
    const int max_roi_height = out_dims.x;
    const int max_roi_width = out_dims.y;
    const int img_offset = C * W * H;
    const int out_img_offset = C * max_roi_width * max_roi_height;

    if ((x >= max_roi_width) || (y >= max_roi_height))
        return;
    for (unsigned int n = 0; n < nchw.x; n++) {
        unsigned int src_idx = (y * W + x) * C;  // src is RGB
        unsigned int dst_idx = (y * max_roi_width + x) * C;
        // copy float3  pixels to dst
        if (C == 3) {
            float3 dst;
            const uchar *inp_img = &inp_image_u8[n * img_offset];
            float *out_tensor = (float *)((float *)output_tensor + dst_buf_offset + n * out_img_offset);
            if (reverse_channels)
                dst = make_float3((float)inp_img[src_idx + 2], (float)inp_img[src_idx + 1], (float)inp_img[src_idx]) * multiplier + offset;
            else
                dst = make_float3((float)inp_img[src_idx], (float)inp_img[src_idx + 1], (float)inp_img[src_idx + 2]) * multiplier + offset;
            out_tensor[dst_idx] = dst.x;
            out_tensor[dst_idx + 1] = dst.y;
            out_tensor[dst_idx + 2] = dst.z;
        } else {
            const uchar *inp_img = &inp_image_u8[n * img_offset + dst_buf_offset];
            float *out_tensor = (float *)output_tensor + dst_buf_offset + n * out_img_offset;
            out_tensor[dst_idx] = (float)inp_img[src_idx] * multiplier.x + offset.x;
        }
    }
}

__global__ void __attribute__((visibility("default")))
Hip_CopyInt8ToNHWC_fp16(
    const unsigned char *inp_image_u8,
    void *output_tensor,
    unsigned int dst_buf_offset,
    uint4 nchw,
    uint2 out_dims,
    float3 multiplier,
    float3 offset,
    const unsigned int reverse_channels) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const int W = nchw.w;
    const int H = nchw.z;
    const int C = nchw.y;
    const int max_roi_height = out_dims.x;
    const int max_roi_width = out_dims.y;
    const int img_offset = C * W * H;
    const int out_img_offset = C * max_roi_width * max_roi_height;

    if ((x >= max_roi_width) || (y >= max_roi_height))
        return;
    for (unsigned int n = 0; n < nchw.x; n++) {
        __half *out_tensor = (__half *)output_tensor + dst_buf_offset + n * out_img_offset;
        unsigned int src_idx = (y * W + x) * C;
        // copy float3  pixels to dst
        if (C == 3) {
            unsigned int dst_idx = y * max_roi_width + x * 3;
            const uchar *inp_img = &inp_image_u8[n * img_offset];
            float3 dst;
            if (reverse_channels)
                dst = make_float3((float)inp_img[src_idx + 2], (float)inp_img[src_idx + 1], (float)inp_img[src_idx]) * multiplier + offset;
            else
                dst = make_float3((float)inp_img[src_idx], (float)inp_img[src_idx + 1], (float)inp_img[src_idx + 2]) * multiplier + offset;
            out_tensor[dst_idx] = __float2half(dst.x);
            out_tensor[dst_idx + 1] = __float2half(dst.y);
            out_tensor[dst_idx + 2] = __float2half(dst.z);
        } else {
            unsigned int dst_idx = y * max_roi_width + x;
            const uchar *inp_img = &inp_image_u8[n * img_offset];
            float *out_tensor = (float *)output_tensor + n * out_img_offset;
            out_tensor[dst_idx] = __float2half((float)inp_img[src_idx] * multiplier.x + offset.x);
        }
    }
}

__global__ void __attribute__((visibility("default")))
Hip_CopyInt8ToNCHW_fp32(
    const uchar *inp_image_u8,
    void *output_tensor,
    unsigned int dst_buf_offset,
    uint4 nchw,
    uint2 out_dims,
    float3 multiplier,
    float3 offset,
    unsigned int reverse_channels) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const int W = nchw.w;
    const int H = nchw.z;
    const int C = nchw.y;
    const int max_roi_height = out_dims.x;
    const int max_roi_width = out_dims.y;
    const int img_offset = C * W * H;
    const int out_img_offset = C * max_roi_width * max_roi_height;
    unsigned int cstride = max_roi_width * max_roi_height;

    if ((x >= max_roi_width) || (y >= max_roi_height))
        return;
    for (unsigned int n = 0; n < nchw.x; n++) {
        unsigned int src_idx = (y * W + x) * C;
        unsigned int dst_idx = y * max_roi_width + x;
        // copy float3  pixels to dst
        const uchar *inp_img = &inp_image_u8[n * img_offset];
        float *out_tensor = (float *)output_tensor + n * out_img_offset + dst_buf_offset;
        if (C == 3) {
            float3 dst;
            if (reverse_channels)
                dst = make_float3((float)inp_img[src_idx + 2], (float)inp_img[src_idx + 1], (float)inp_img[src_idx]) * multiplier + offset;
            else
                dst = make_float3((float)inp_img[src_idx], (float)inp_img[src_idx + 1], (float)inp_img[src_idx + 2]) * multiplier + offset;
            out_tensor[dst_idx] = dst.x;
            out_tensor[dst_idx + cstride] = dst.y;
            out_tensor[dst_idx + cstride * 2] = dst.z;
        } else {
            out_tensor[dst_idx] = (float)inp_img[src_idx] * multiplier.x + offset.x;
        }
    }
}

__global__ void __attribute__((visibility("default")))
Hip_CopyInt8ToNCHW_fp16(
    const uchar *inp_image_u8,
    void *output_tensor,
    unsigned int dst_buf_offset,
    uint4 nchw,
    uint2 out_dims,
    float3 multiplier,
    float3 offset,
    const unsigned int reverse_channels) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const int W = nchw.w;
    const int H = nchw.z;
    const int C = nchw.y;
    const int max_roi_height = out_dims.x;
    const int max_roi_width = out_dims.y;
    const int img_offset = C * W * H;
    const int out_img_offset = C * max_roi_width * max_roi_height;
    unsigned int cstride = max_roi_width * max_roi_height;

    if ((x >= max_roi_width) || (y >= max_roi_height))
        return;
    for (unsigned int n = 0; n < nchw.x; n++) {
        __half *out_tensor = (__half *)output_tensor + n * out_img_offset + dst_buf_offset;
        const uchar *inp_img = &inp_image_u8[n * img_offset];
        unsigned int src_idx = (y * W + x) * C;
        // copy float3  pixels to dst
        unsigned int dst_idx = y * max_roi_width + x;
        if (C == 3) {
            float3 dst;
            if (reverse_channels)
                dst = make_float3((float)inp_img[src_idx + 2], (float)inp_img[src_idx + 1], (float)inp_img[src_idx]) * multiplier + offset;
            else
                dst = make_float3((float)inp_img[src_idx], (float)inp_img[src_idx + 1], (float)inp_img[src_idx + 2]) * multiplier + offset;
            out_tensor[dst_idx] = __float2half(dst.x);
            out_tensor[dst_idx + cstride] = __float2half(dst.y);
            out_tensor[dst_idx + cstride * 2] = __float2half(dst.z);
        } else {
            out_tensor[dst_idx] = __float2half((float)inp_img[src_idx] * multiplier.x + offset.x);
        }
    }
}

int HipExecCopyInt8ToNHWC(
    hipStream_t stream,
    const void *inp_image_u8,
    void *output_tensor,
    unsigned int dst_buf_offset,
    const unsigned int n,
    const unsigned int c,
    const unsigned int h,
    const unsigned int w,
    float multiplier0,
    float multiplier1,
    float multiplier2,
    float offset0,
    float offset1,
    float offset2,
    unsigned int reverse_channels,
    unsigned int fp16,
    const unsigned max_output_height,
    const unsigned max_output_width) {
    int local_threads_x = 16, local_threads_y = 16;
    uint2 out_dims;
    if ((max_output_height == 0) || (max_output_width == 0))
        out_dims = make_uint2(h, w);
    else
        out_dims = make_uint2(max_output_height, max_output_width);
    int global_threads_x = w, global_threads_y = h;
    if (!fp16) {
        hipLaunchKernelGGL(Hip_CopyInt8ToNHWC_fp32,
                           dim3(ceil((float)global_threads_x / local_threads_x), ceil((float)global_threads_y / local_threads_y)),
                           dim3(local_threads_x, local_threads_y),
                           0, stream, (const uchar *)inp_image_u8, output_tensor, dst_buf_offset,
                           make_uint4(n, c, h, w), out_dims,
                           make_float3(multiplier0, multiplier1, multiplier2), make_float3(offset0, offset1, offset2),
                           reverse_channels);
    } else {
        hipLaunchKernelGGL(Hip_CopyInt8ToNHWC_fp16,
                           dim3(ceil((float)global_threads_x / local_threads_x), ceil((float)global_threads_y / local_threads_y)),
                           dim3(local_threads_x, local_threads_y),
                           0, stream, (const uchar *)inp_image_u8, output_tensor, dst_buf_offset,
                           make_uint4(n, c, h, w), out_dims,
                           make_float3(multiplier0, multiplier1, multiplier2), make_float3(offset0, offset1, offset2),
                           reverse_channels);
    }
    return 0;
}

int HipExecCopyInt8ToNCHW(
    hipStream_t stream,
    const void *inp_image_u8,
    void *output_tensor,
    unsigned int dst_buf_offset,
    const unsigned int n,
    const unsigned int c,
    const unsigned int h,
    const unsigned int w,
    float multiplier0,
    float multiplier1,
    float multiplier2,
    float offset0,
    float offset1,
    float offset2,
    unsigned int reverse_channels,
    unsigned int fp16,
    const unsigned max_output_height,
    const unsigned max_output_width) {
    int local_threads_x = 16, local_threads_y = 16;
    uint2 out_dims;
    if ((max_output_height == 0) || (max_output_width == 0))
        out_dims = make_uint2(h, w);
    else
        out_dims = make_uint2(max_output_height, max_output_width);
    int global_threads_x = w, global_threads_y = h;
    if (!fp16) {
        hipLaunchKernelGGL(Hip_CopyInt8ToNCHW_fp32,
                           dim3(ceil((float)global_threads_x / local_threads_x), ceil((float)global_threads_y / local_threads_y)),
                           dim3(local_threads_x, local_threads_y),
                           0, stream, (const uchar *)inp_image_u8, output_tensor, dst_buf_offset,
                           make_uint4(n, c, h, w), out_dims,
                           make_float3(multiplier0, multiplier1, multiplier2), make_float3(offset0, offset1, offset2),
                           reverse_channels);
    } else {
        hipLaunchKernelGGL(Hip_CopyInt8ToNCHW_fp16,
                           dim3(ceil((float)global_threads_x / local_threads_x), ceil((float)global_threads_y / local_threads_y)),
                           dim3(local_threads_x, local_threads_y),
                           0, stream, (const uchar *)inp_image_u8, output_tensor, dst_buf_offset,
                           make_uint4(n, c, h, w), out_dims,
                           make_float3(multiplier0, multiplier1, multiplier2), make_float3(offset0, offset1, offset2),
                           reverse_channels);
    }
    return 0;
}

__device__ void resize_roi_and_srclocs_hip_compute(int4 *src_roi_ptr_i4, uint2 *dst_dims_wh, int id_x, int id_y, d_float16 *loc_src_f16) {
    float w_ratio = (float)(src_roi_ptr_i4->z - src_roi_ptr_i4->x + 1) / dst_dims_wh->x;
    float h_ratio = (float)(src_roi_ptr_i4->w - src_roi_ptr_i4->y + 1) / dst_dims_wh->y;
    float4 w_offset_f4 = (float4)((w_ratio - 1) * 0.5f);
    float4 h_offset_f4 = (float4)((h_ratio - 1) * 0.5f);

    d_float8 increment_f8, loc_dst_f8x, loc_dst_f8y;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    loc_dst_f8x.f4[0] = (float4)id_x + increment_f8.f4[0];
    loc_dst_f8x.f4[1] = (float4)id_x + increment_f8.f4[1];
    loc_dst_f8y.f4[0] = (float4)id_y;
    loc_dst_f8y.f4[1] = (float4)id_y;

    loc_src_f16->f8[0].f4[0] = (loc_dst_f8x.f4[0] * (float4)w_ratio) + w_offset_f4 + (float4)src_roi_ptr_i4->x;  // Compute src x locations in float for dst x locations [0-3]
    loc_src_f16->f8[0].f4[1] = (loc_dst_f8x.f4[1] * (float4)w_ratio) + w_offset_f4 + (float4)src_roi_ptr_i4->x;  // Compute src x locations in float for dst x locations [4-7]
    loc_src_f16->f8[1].f4[0] = (loc_dst_f8y.f4[0] * (float4)h_ratio) + h_offset_f4 + (float4)src_roi_ptr_i4->y;  // Compute src y locations in float for dst y locations [0-3]
    loc_src_f16->f8[1].f4[1] = (loc_dst_f8y.f4[1] * (float4)h_ratio) + h_offset_f4 + (float4)src_roi_ptr_i4->y;  // Compute src y locations in float for dst y locations [4-7]
}

// float bilinear interpolation computation
__device__ __forceinline__ void rpp_hip_interpolate_bilinear(float4 *src_neighborhood_f4, float2 *weighted_wh_f2, float2 *one_minus_weighted_wh_f2, float *dst) {
    *dst = fmaf(src_neighborhood_f4->x, one_minus_weighted_wh_f2->y * one_minus_weighted_wh_f2->x,
                fmaf(src_neighborhood_f4->y, one_minus_weighted_wh_f2->y * weighted_wh_f2->x,
                    fmaf(src_neighborhood_f4->z, weighted_wh_f2->y * one_minus_weighted_wh_f2->x,
                        src_neighborhood_f4->w * weighted_wh_f2->y * weighted_wh_f2->x)));
}

// ROI range check for source locations calculated
__device__ __forceinline__ void rpp_hip_roi_range_check(float2 *loc_src_floor_f2, int4 *roi_ptr_src_i4, int2 *loc_src_i2) {
    loc_src_i2->x = (int)fminf(fmaxf(loc_src_floor_f2->x, roi_ptr_src_i4->x), roi_ptr_src_i4->z - 1);
    loc_src_i2->y = (int)fminf(fmaxf(loc_src_floor_f2->y, roi_ptr_src_i4->y), roi_ptr_src_i4->w - 1);
}
__device__ __forceinline__ float rpp_hip_unpack0(uint src) {
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack1(uint src) {
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack2(uint src) {
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_load_pkd3(uchar *src_ptr, uint src_stride_h, float2 *loc_src_floor_f2, int4 *roi_ptr_src_i4, d_float12 *src_neighborhood_f12) {
    uint2 src_u2;
    int2 loc_src1_i2, loc_src2_i2;
    rpp_hip_roi_range_check(loc_src_floor_f2, roi_ptr_src_i4, &loc_src1_i2);
    *loc_src_floor_f2 = *loc_src_floor_f2 + (float2)1.0f;
    rpp_hip_roi_range_check(loc_src_floor_f2, roi_ptr_src_i4, &loc_src2_i2);
    int2 src_inter_row_loc_i2, src_inter_col_loc_i2;
    src_inter_row_loc_i2.x = loc_src1_i2.y * src_stride_h;
    src_inter_row_loc_i2.y = loc_src2_i2.y * src_stride_h;
    src_inter_col_loc_i2.x = loc_src1_i2.x * 3;
    src_inter_col_loc_i2.y = loc_src2_i2.x * 3;

    int src_idx1 = src_inter_row_loc_i2.x + src_inter_col_loc_i2.x;   // Top Left
    int src_idx2 = src_inter_row_loc_i2.x + src_inter_col_loc_i2.y;   // Top Right
    src_u2.x = *(uint *)&src_ptr[src_idx1];
    src_u2.y = *(uint *)&src_ptr[src_idx2];
    src_neighborhood_f12->f1[0] = rpp_hip_unpack0(src_u2.x);
    src_neighborhood_f12->f1[1] = rpp_hip_unpack0(src_u2.y);
    src_neighborhood_f12->f1[4] = rpp_hip_unpack1(src_u2.x);
    src_neighborhood_f12->f1[5] = rpp_hip_unpack1(src_u2.y);
    src_neighborhood_f12->f1[8] = rpp_hip_unpack2(src_u2.x);
    src_neighborhood_f12->f1[9] = rpp_hip_unpack2(src_u2.y);
    src_idx1 = src_inter_row_loc_i2.y + src_inter_col_loc_i2.x;   // Bottom left
    src_idx2 = src_inter_row_loc_i2.y + src_inter_col_loc_i2.y;   // Bottom right
    src_u2.x = *(uint *)&src_ptr[src_idx1];
    src_u2.y = *(uint *)&src_ptr[src_idx2];
    src_neighborhood_f12->f1[ 2] = rpp_hip_unpack0(src_u2.x);
    src_neighborhood_f12->f1[ 3] = rpp_hip_unpack0(src_u2.y);
    src_neighborhood_f12->f1[ 6] = rpp_hip_unpack1(src_u2.x);
    src_neighborhood_f12->f1[ 7] = rpp_hip_unpack1(src_u2.y);
    src_neighborhood_f12->f1[10] = rpp_hip_unpack2(src_u2.x);
    src_neighborhood_f12->f1[11] = rpp_hip_unpack2(src_u2.y);
}

// float3 bilinear interpolation pkd3
template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_pkd3(T *src_ptr, uint src_stride_h, float loc_src_x, float loc_src_y, int4 *roi_ptr_src_i4, float3 *dst_f3, bool check_range) {
    float2 loc_src_floor_f2, weighted_wh_f2, one_minus_weighted_wh_f2;
    loc_src_floor_f2.x = floorf(loc_src_x);
    loc_src_floor_f2.y = floorf(loc_src_y);
    if (check_range && ((loc_src_floor_f2.x < roi_ptr_src_i4->x) || (loc_src_floor_f2.y < roi_ptr_src_i4->y) || (loc_src_floor_f2.x > roi_ptr_src_i4->z) || (loc_src_floor_f2.y > roi_ptr_src_i4->w))) {
        *dst_f3 = (float3) 0.0f;
    } else {
        weighted_wh_f2.x = loc_src_x - loc_src_floor_f2.x;
        weighted_wh_f2.y = loc_src_y - loc_src_floor_f2.y;
        one_minus_weighted_wh_f2.x = 1.0f - weighted_wh_f2.x;
        one_minus_weighted_wh_f2.y = 1.0f - weighted_wh_f2.y;
        d_float12 src_neighborhood_f12;
        rpp_hip_interpolate3_bilinear_load_pkd3(src_ptr, src_stride_h, &loc_src_floor_f2, roi_ptr_src_i4, &src_neighborhood_f12);
        rpp_hip_interpolate_bilinear(&src_neighborhood_f12.f4[0], &weighted_wh_f2, &one_minus_weighted_wh_f2, &(dst_f3->x));
        rpp_hip_interpolate_bilinear(&src_neighborhood_f12.f4[1], &weighted_wh_f2, &one_minus_weighted_wh_f2, &(dst_f3->y));
        rpp_hip_interpolate_bilinear(&src_neighborhood_f12.f4[2], &weighted_wh_f2, &one_minus_weighted_wh_f2, &(dst_f3->z));
    }
}

// d_float24 bilinear interpolation in pkd3
template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate24_bilinear_pkd3(T *src_ptr, uint src_stride_h, d_float16 *loc_ptr_src_f16, int4 *roi_ptr_src_i4, d_float24 *dst_f24, bool check_range = true) {
    rpp_hip_interpolate3_bilinear_pkd3(src_ptr, src_stride_h, loc_ptr_src_f16->f1[0], loc_ptr_src_f16->f1[ 8], roi_ptr_src_i4, &(dst_f24->f3[0]), check_range);
    rpp_hip_interpolate3_bilinear_pkd3(src_ptr, src_stride_h, loc_ptr_src_f16->f1[1], loc_ptr_src_f16->f1[ 9], roi_ptr_src_i4, &(dst_f24->f3[1]), check_range);
    rpp_hip_interpolate3_bilinear_pkd3(src_ptr, src_stride_h, loc_ptr_src_f16->f1[2], loc_ptr_src_f16->f1[10], roi_ptr_src_i4, &(dst_f24->f3[2]), check_range);
    rpp_hip_interpolate3_bilinear_pkd3(src_ptr, src_stride_h, loc_ptr_src_f16->f1[3], loc_ptr_src_f16->f1[11], roi_ptr_src_i4, &(dst_f24->f3[3]), check_range);
    rpp_hip_interpolate3_bilinear_pkd3(src_ptr, src_stride_h, loc_ptr_src_f16->f1[4], loc_ptr_src_f16->f1[12], roi_ptr_src_i4, &(dst_f24->f3[4]), check_range);
    rpp_hip_interpolate3_bilinear_pkd3(src_ptr, src_stride_h, loc_ptr_src_f16->f1[5], loc_ptr_src_f16->f1[13], roi_ptr_src_i4, &(dst_f24->f3[5]), check_range);
    rpp_hip_interpolate3_bilinear_pkd3(src_ptr, src_stride_h, loc_ptr_src_f16->f1[6], loc_ptr_src_f16->f1[14], roi_ptr_src_i4, &(dst_f24->f3[6]), check_range);
    rpp_hip_interpolate3_bilinear_pkd3(src_ptr, src_stride_h, loc_ptr_src_f16->f1[7], loc_ptr_src_f16->f1[15], roi_ptr_src_i4, &(dst_f24->f3[7]), check_range);
}

// Packing to U8s
__device__ __forceinline__ uint rpp_hip_pack(float4 src) {
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

// U8 stores without layout toggle PKD3 to PKD3 (24 U8 pixels)
__device__ __forceinline__ void rpp_hip_pack_float24_pkd3_and_store24_pkd3(uchar *dst_ptr, d_float24 *dst_ptr_f24) {
    d_uint6 dst_ui6;

    dst_ui6.ui1[0] = rpp_hip_pack(dst_ptr_f24->f4[0]);    // write R00G00B00R01
    dst_ui6.ui1[1] = rpp_hip_pack(dst_ptr_f24->f4[1]);    // write G01B01R02G02
    dst_ui6.ui1[2] = rpp_hip_pack(dst_ptr_f24->f4[2]);    // write B02R03G03B03
    dst_ui6.ui1[3] = rpp_hip_pack(dst_ptr_f24->f4[3]);    // write R04G04B04R05
    dst_ui6.ui1[4] = rpp_hip_pack(dst_ptr_f24->f4[4]);    // write G05B05R06G06
    dst_ui6.ui1[5] = rpp_hip_pack(dst_ptr_f24->f4[5]);    // write B06R07G07B07

    *(d_uint6_s *)dst_ptr = *(d_uint6_s *)&dst_ui6;
}

template <typename T>
__global__ void resize_bilinear_pkd_hip_tensor(T *src_ptr,
                                               uint2 src_strides,
                                               T *dst_ptr,
                                               uint2 dst_strides,
                                               size_t *src_width,
                                               size_t *src_height,
                                               size_t *dst_width,
                                               size_t *dst_height,
                                               size_t *src_height_stride,
                                               size_t *src_img_offset) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dst_dims_wh;
    dst_dims_wh.x = dst_width[id_z];
    dst_dims_wh.y = dst_height[id_z];

    if ((id_y >= dst_dims_wh.y) || (id_x >= dst_dims_wh.x))
    {
        return;
    }

    uint src_idx = src_img_offset[id_z];
    uint dst_idx = (id_z * dst_strides.x) + (id_y * dst_strides.y) + id_x * 3;

    int4 src_roi_i4;
    src_roi_i4.x = 0;
    src_roi_i4.y = 0;
    src_roi_i4.z = src_width[id_z] - 1;
    src_roi_i4.w = src_height[id_z] - 1;

    d_float16 loc_src_f16;
    resize_roi_and_srclocs_hip_compute(&src_roi_i4, &dst_dims_wh, id_x, id_y, &loc_src_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(src_ptr + src_idx, src_height_stride[id_z], &loc_src_f16, &src_roi_i4, &dst_f24, false);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dst_ptr + dst_idx, &dst_f24);
}

__device__ void resize_roi_generic_srcloc_and_weight_hip_compute(int roi_loc, int dst_location, float scale, int limit, int *src_loc, float *weight, float offset, int src_stride) {
    float src_location_raw = ((float) dst_location) * scale + offset + (float)roi_loc;
    int src_location_rounded = (int)ceilf(src_location_raw);
    *weight = src_location_rounded - src_location_raw;
    *src_loc = ((src_location_rounded > limit) ? limit : src_location_rounded) * src_stride;
}

__device__ __forceinline__ void rpp_hip_compute_triangular_coefficient(float weight, float *coeff) {
    *coeff = 1 - fabsf(weight);
    *coeff = *coeff < 0 ? 0 : *coeff;
}

__device__ __forceinline__ void rpp_hip_pixel_check_and_store(float pixel, uchar* dst) {
    pixel = fmax(fminf(pixel, 255), 0);
    *dst = (uchar)pixel;
}

__device__ void rpp_hip_compute_interpolation_scale_and_radius(float *scale, float *radius, float scale_ratio) {
    if(scale_ratio > 1.0f) {
        *radius = scale_ratio;
        *scale = (1 / scale_ratio);
    } else {
        *radius = 1.0f;
        *scale = 1.0f;
    }
}

template <typename T>
__global__ void resize_generic_pln1_hip_tensor(T *src_ptr,
                                               uint3 src_strides,
                                               T *dst_ptr,
                                               uint3 dst_strides,
                                               size_t *src_width,
                                               size_t *src_height,
                                               size_t *dst_width,
                                               size_t *dst_height,
                                               size_t *src_height_stride,
                                               size_t *src_img_offset) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dst_dims_wh;
    dst_dims_wh.x = dst_width[id_z];
    dst_dims_wh.y = dst_height[id_z];

    if ((id_y >= dst_dims_wh.y) || (id_x >= dst_dims_wh.x)) {
        return;
    }

    int4 src_roi_i4;
    src_roi_i4.x = 0;
    src_roi_i4.y = 0;
    src_roi_i4.z = src_width[id_z] - 1;
    src_roi_i4.w = src_height[id_z] - 1;

    uint2 src_dims_wh;
    src_dims_wh.x = src_width[id_z];
    src_dims_wh.y = src_height[id_z];

    int width_limit = src_roi_i4.z;
    int height_limit = src_roi_i4.w;
    float w_ratio = (float)src_dims_wh.x / (float)dst_dims_wh.x;
    float h_ratio = (float)src_dims_wh.y / (float)dst_dims_wh.y;
    float h_scale = 1.0f, w_scale = 1.0f, h_radius = 1.0f, w_radius = 1.0f;

    rpp_hip_compute_interpolation_scale_and_radius(&w_scale, &w_radius, w_ratio);
    rpp_hip_compute_interpolation_scale_and_radius(&h_scale, &h_radius, h_ratio);
    float w_offset = (w_ratio - 1) * 0.5f - w_radius;
    float h_offset = (h_ratio - 1) * 0.5f - h_radius;
    int w_kernel_size = ceilf(w_radius * 2);
    int h_kernel_size = ceilf(h_radius * 2);

    float row_weight, col_weight, row_coeff, col_coeff;
    int src_location_row_floor, src_location_column_floor;
    resize_roi_generic_srcloc_and_weight_hip_compute(src_roi_i4.x, id_x, w_ratio, width_limit, &src_location_column_floor, &col_weight, w_offset, 1);
    resize_roi_generic_srcloc_and_weight_hip_compute(src_roi_i4.y, id_y, h_ratio, height_limit, &src_location_row_floor, &row_weight, h_offset, 1);

    T *src_ptr_temp = src_ptr + src_img_offset[id_z];
    float out_pixel = 0.0f;
    float row_coeff_sum = 0.0f, col_coeff_sum = 0.0f, inv_coeff_sum = 0.0f;
    for (int j = 0; j < h_kernel_size; j++) {
        int row_index = fminf(fmaxf((int)(src_location_row_floor + j), 0), height_limit);
        T *src_row_ptrs_for_interp = src_ptr_temp + row_index * src_height_stride[id_z];
        rpp_hip_compute_triangular_coefficient((row_weight - h_radius + j) * h_scale, &row_coeff);
        row_coeff_sum += row_coeff;

        col_coeff_sum = 0;
        for (int k = 0; k < w_kernel_size; k++) {
            int col_index = fminf(fmaxf((int)(src_location_column_floor + k), 0), width_limit);
            rpp_hip_compute_triangular_coefficient((col_weight - w_radius + k) * w_scale, &col_coeff);
            col_coeff_sum += col_coeff;
            float coeff = col_coeff * row_coeff;
            out_pixel += (float)src_row_ptrs_for_interp[col_index] * coeff;
        }
    }
    row_coeff_sum = (row_coeff_sum == 0.0f) ? 1.0f : row_coeff_sum;
    col_coeff_sum = (col_coeff_sum == 0.0f) ? 1.0f : col_coeff_sum;
    inv_coeff_sum = 1 / (row_coeff_sum * col_coeff_sum);
    out_pixel *= inv_coeff_sum;
    uint dst_idx = (id_z * dst_strides.x) + (id_y * dst_strides.z) + id_x;
    rpp_hip_pixel_check_and_store(out_pixel, &dst_ptr[dst_idx]);
}

void HipExecResizeTensor(hipStream_t stream,
                         void *src_ptr,
                         void *dst_ptr,
                         unsigned batch_size,
                         size_t *src_width,
                         size_t *src_height,
                         size_t *dst_width,
                         size_t *dst_height,
                         size_t *src_height_stride,
                         size_t *src_img_offset,
                         unsigned channels,
                         const size_t max_src_width,
                         const size_t max_src_height,
                         const size_t max_dst_width,
                         const size_t max_dst_height) {
    unsigned globalThreads_x = (max_dst_width + 7) >> 3;
    unsigned globalThreads_y = max_dst_height;
    unsigned globalThreads_z = batch_size;

    globalThreads_x = max_dst_width;
    if (channels == 3) {    // For RGB images
        hipLaunchKernelGGL(resize_bilinear_pkd_hip_tensor,
                            dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                            0,
                            stream,
                            static_cast<unsigned char *>(src_ptr),
                            make_uint2(max_src_width * max_src_height * channels, max_src_width * channels),
                            static_cast<unsigned char *>(dst_ptr),
                            make_uint2(max_dst_width * max_dst_height * channels, max_dst_width * channels),
                            src_width,
                            src_height,
                            dst_width,
                            dst_height,
                            src_height_stride,
                            src_img_offset);
    } else if (channels == 1) {
        hipLaunchKernelGGL(resize_generic_pln1_hip_tensor,
                    dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                    0,
                    stream,
                    static_cast<unsigned char *>(src_ptr),
                    make_uint3(max_src_width * max_src_height * channels, max_src_width * max_src_height, max_src_width),
                    static_cast<unsigned char *>(dst_ptr),
                    make_uint3(max_dst_width * max_dst_height * channels, max_src_width * max_src_height, max_dst_width),
                    src_width,
                    src_height,
                    dst_width,
                    dst_height,
                    src_height_stride,
                    src_img_offset);  
    }

}
