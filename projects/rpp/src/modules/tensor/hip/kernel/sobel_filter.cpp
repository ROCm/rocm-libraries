/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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

#include "hip_tensor_executors.hpp"
#include "rpp_hip_load_store.hpp"
#include "rpp_hip_math.hpp"

#define HIP_VECTOR_MAX_INDEX 7  // Maximum index when processing 8 pixels (0-7 inclusive)

__device__ __constant__ float sobel3x3XHip[9] = {-1, 0, 1,
                                                 -2, 0, 2,
                                                 -1, 0, 1};
__device__ __constant__ float sobel3x3YHip[9] = {-1, -2, -1,
                                                  0, 0, 0,
                                                  1, 2, 1};
__device__ __constant__ float sobel5x5XHip[25] = {-1,  -2,   0,   2,   1,
                                                  -4,  -8,   0,   8,   4,
                                                  -6, -12,   0,  12,   6,
                                                  -4,  -8,   0,   8,   4,
                                                  -1,  -2,   0,   2,   1};
__device__ __constant__ float sobel5x5YHip[25] = {-1,  -4,  -6,  -4,  -1,
                                                  -2,  -8, -12,  -8,  -2,
                                                   0,   0,   0,   0,   0,
                                                   2,   8,  12,   8,   2,
                                                   1,   4,   6,   4,   1};
__device__ __constant__ float sobel7x7XHip[49] = {-1,   -4,   -5,    0,    5,    4,    1,
                                                  -6,  -24,  -30,    0,   30,   24,    6,
                                                  -15,  -60,  -75,    0,   75,   60,   15,
                                                  -20,  -80, -100,    0,  100,   80,   20,
                                                  -15,  -60,  -75,    0,   75,   60,   15,
                                                  -6,  -24,  -30,    0,   30,   24,    6,
                                                  -1,   -4,   -5,    0,    5,    4,    1};
__device__ __constant__ float sobel7x7YHip[49] = {-1,   -6,  -15,  -20,  -15,   -6,   -1,
                                                  -4,  -24,  -60,  -80,  -60,  -24,   -4,
                                                  -5,  -30,  -75, -100,  -75,  -30,   -5,
                                                   0,    0,    0,    0,    0,    0,    0,
                                                   5,   30,   75,  100,   75,   30,    5,
                                                   4,   24,   60,   80,   60,   24,    4,
                                                   1,    6,   15,   20,   15,    6,    1};

// -------------------- sobel_filter device helpers --------------------

__device__ __forceinline__ void sobel_filter_bidirection_hip_compute(d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8)
{
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 x = src1_f8->f2[i];
        float2 y = src2_f8->f2[i];

        dst_f8->f2[i].x = sqrtf(fmaf(x.x, x.x, y.x * y.x));
        dst_f8->f2[i].y = sqrtf(fmaf(x.y, x.y, y.y * y.y));
    }
}

template <int filterSize, typename T>
__device__ void sobel_row_hip_compute(T *srcPtr, d_float8 *dst_f8, float *filter)
{
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        #pragma unroll
        for(int k = 0; k < filterSize; ++k)
            dst_f8->f1[j] = fmaf(srcPtr[j + k], filter[k], dst_f8->f1[j]);
    }
}

template <typename T>
__global__ void sobel_filter_3x3_pln_bidirection_tensor(T *srcPtr,
                                                        uint3 srcStridesNCH,
                                                        T *dstPtr,
                                                        uint3 dstStridesNCH,
                                                        int channelsDst,
                                                        uint padLength,
                                                        uint2 tileSize,
                                                        RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int roiBeginX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiBeginY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    int clampedY = roiBeginY + max(0, min(id_y_i, (roiHeight - 1)));

    d_float8 sum_f8x, sum_f8y, sum_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filterRowX1 = &sobel3x3XHip[0];
    float *filterRowX2 = &filterRowX1[3];
    float *filterRowX3 = &filterRowX1[6];
    float *filterRowY1 = &sobel3x3YHip[0];
    float *filterRowY2 = &filterRowY1[3];
    float *filterRowY3 = &filterRowY1[6];
    sum_f8x.f4[0] = FLOAT4_ZERO;
    sum_f8x.f4[1] = FLOAT4_ZERO;
    sum_f8y.f4[0] = FLOAT4_ZERO;
    sum_f8y.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + HIP_VECTOR_MAX_INDEX) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
    {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++)
        {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    {
        sobel_row_hip_compute<3>(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8x, filterRowX1);
        sobel_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8x, filterRowX2);
        sobel_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8x, filterRowX3);
        sobel_row_hip_compute<3>(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8y, filterRowY1);
        sobel_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8y, filterRowY2);
        sobel_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8y, filterRowY3);
        if constexpr (std::is_same<T, Rpp8u>::value)
        {
            rpp_hip_pixel_check_0to255(&sum_f8x);
            rpp_hip_pixel_check_0to255(&sum_f8y);
        }
        else
        {
            rpp_hip_pixel_check_0to1(&sum_f8x);
            rpp_hip_pixel_check_0to1(&sum_f8y);
        }
        sobel_filter_bidirection_hip_compute(&sum_f8x, &sum_f8y, &sum_f8);
        if constexpr (std::is_same<T, Rpp32f>::value)
            rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
__global__ void sobel_filter_5x5_pln_bidirection_tensor(T *srcPtr,
                                                        uint3 srcStridesNCH,
                                                        T *dstPtr,
                                                        uint3 dstStridesNCH,
                                                        int channelsDst,
                                                        uint padLength,
                                                        uint2 tileSize,
                                                        RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    int roiBeginX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiBeginY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    int clampedY = roiBeginY + max(0, min(id_y_i, (roiHeight - 1)));
    d_float8 sum_f8x, sum_f8y, sum_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filterRowX1 = &sobel5x5XHip[0];
    float *filterRowX2 = &filterRowX1[5];
    float *filterRowX3 = &filterRowX1[10];
    float *filterRowX4 = &filterRowX1[15];
    float *filterRowX5 = &filterRowX1[20];
    float *filterRowY1 = &sobel5x5YHip[0];
    float *filterRowY2 = &filterRowY1[5];
    float *filterRowY3 = &filterRowY1[10];
    float *filterRowY4 = &filterRowY1[15];
    float *filterRowY5 = &filterRowY1[20];
    sum_f8x.f4[0] = FLOAT4_ZERO;
    sum_f8x.f4[1] = FLOAT4_ZERO;
    sum_f8y.f4[0] = FLOAT4_ZERO;
    sum_f8y.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + HIP_VECTOR_MAX_INDEX) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
    {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++)
        {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    {
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8x, filterRowX1);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8x, filterRowX2);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8x, filterRowX3);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8x, filterRowX4);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8x, filterRowX5);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8y, filterRowY1);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8y, filterRowY2);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8y, filterRowY3);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8y, filterRowY4);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8y, filterRowY5);
        if constexpr (std::is_same<T, Rpp8u>::value)
        {
            rpp_hip_pixel_check_0to255(&sum_f8x);
            rpp_hip_pixel_check_0to255(&sum_f8y);
        }
        else
        {
            rpp_hip_pixel_check_0to1(&sum_f8x);
            rpp_hip_pixel_check_0to1(&sum_f8y);
        }
        sobel_filter_bidirection_hip_compute(&sum_f8x, &sum_f8y, &sum_f8);
        if constexpr (std::is_same<T, Rpp32f>::value)
            rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
__global__ void sobel_filter_7x7_pln_bidirection_tensor(T *srcPtr,
                                                        uint3 srcStridesNCH,
                                                        T *dstPtr,
                                                        uint3 dstStridesNCH,
                                                        int channelsDst,
                                                        uint padLength,
                                                        uint2 tileSize,
                                                        RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    int roiBeginX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiBeginY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    int clampedY = roiBeginY + max(0, min(id_y_i, (roiHeight - 1)));
    d_float8 sum_f8x, sum_f8y, sum_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filterRowX1 = &sobel7x7XHip[0];
    float *filterRowX2 = &filterRowX1[7];
    float *filterRowX3 = &filterRowX1[14];
    float *filterRowX4 = &filterRowX1[21];
    float *filterRowX5 = &filterRowX1[28];
    float *filterRowX6 = &filterRowX1[35];
    float *filterRowX7 = &filterRowX1[42];
    float *filterRowY1 = &sobel7x7YHip[0];
    float *filterRowY2 = &filterRowY1[7];
    float *filterRowY3 = &filterRowY1[14];
    float *filterRowY4 = &filterRowY1[21];
    float *filterRowY5 = &filterRowY1[28];
    float *filterRowY6 = &filterRowY1[35];
    float *filterRowY7 = &filterRowY1[42];
    sum_f8x.f4[0] = FLOAT4_ZERO;
    sum_f8x.f4[1] = FLOAT4_ZERO;
    sum_f8y.f4[0] = FLOAT4_ZERO;
    sum_f8y.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + HIP_VECTOR_MAX_INDEX) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
    {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++)
        {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    {
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8x, filterRowX1);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8x, filterRowX2);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8x, filterRowX3);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8x, filterRowX4);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8x, filterRowX5);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8x, filterRowX6);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8x, filterRowX7);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8y, filterRowY1);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8y, filterRowY2);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8y, filterRowY3);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8y, filterRowY4);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8y, filterRowY5);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8y, filterRowY6);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8y, filterRowY7);
        if constexpr (std::is_same<T, Rpp8u>::value)
        {
            rpp_hip_pixel_check_0to255(&sum_f8x);
            rpp_hip_pixel_check_0to255(&sum_f8y);
        }
        else
        {
            rpp_hip_pixel_check_0to1(&sum_f8x);
            rpp_hip_pixel_check_0to1(&sum_f8y);
        }
        sobel_filter_bidirection_hip_compute(&sum_f8x, &sum_f8y, &sum_f8);
        if constexpr (std::is_same<T, Rpp32f>::value)
            rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
__global__ void sobel_filter_3x3_pln_x_gradient_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       int channelsDst,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int roiBeginX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiBeginY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    int clampedY = roiBeginY + max(0, min(id_y_i, (roiHeight - 1)));

    d_float8 sum_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filter_row1 = &sobel3x3XHip[0];
    float *filter_row2 = &filter_row1[3];
    float *filter_row3 = &filter_row1[6];
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + HIP_VECTOR_MAX_INDEX) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
    {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++)
        {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    {
        sobel_row_hip_compute<3>(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        sobel_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        sobel_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        if constexpr (std::is_same<T, Rpp8u>::value)
            rpp_hip_pixel_check_0to255(&sum_f8);
        else if constexpr (std::is_same<T, Rpp32f>::value)
            rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
__global__ void sobel_filter_5x5_pln_x_gradient_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       int channelsDst,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    int roiBeginX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiBeginY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    int clampedY = roiBeginY + max(0, min(id_y_i, (roiHeight - 1)));

    d_float8 sum_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filter_row1 = &sobel5x5XHip[0];
    float *filter_row2 = &filter_row1[5];
    float *filter_row3 = &filter_row1[10];
    float *filter_row4 = &filter_row1[15];
    float *filter_row5 = &filter_row1[20];
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + HIP_VECTOR_MAX_INDEX) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
    {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++)
        {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    {
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
        if constexpr (std::is_same<T, Rpp8u>::value)
            rpp_hip_pixel_check_0to255(&sum_f8);
        else if constexpr (std::is_same<T, Rpp32f>::value)
            rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
__global__ void sobel_filter_7x7_pln_x_gradient_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       int channelsDst,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    int roiBeginX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiBeginY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    int clampedY = roiBeginY + max(0, min(id_y_i, (roiHeight - 1)));
    d_float8 sum_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filter_row1 = &sobel7x7XHip[0];
    float *filter_row2 = &filter_row1[7];
    float *filter_row3 = &filter_row1[14];
    float *filter_row4 = &filter_row1[21];
    float *filter_row5 = &filter_row1[28];
    float *filter_row6 = &filter_row1[35];
    float *filter_row7 = &filter_row1[42];
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + HIP_VECTOR_MAX_INDEX) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
    {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++)
        {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    {
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, filter_row6);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, filter_row7);
        if constexpr (std::is_same<T, Rpp8u>::value)
            rpp_hip_pixel_check_0to255(&sum_f8);
        else if constexpr (std::is_same<T, Rpp32f>::value)
            rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
__global__ void sobel_filter_3x3_pln_y_gradient_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       int channelsDst,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    int roiBeginX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiBeginY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    int clampedY = roiBeginY + max(0, min(id_y_i, (roiHeight - 1)));
    
    d_float8 sum_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filter_row1 = &sobel3x3YHip[0];
    float *filter_row2 = &filter_row1[3];
    float *filter_row3 = &filter_row1[6];
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + HIP_VECTOR_MAX_INDEX) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
    {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++)
        {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    {
        sobel_row_hip_compute<3>(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        sobel_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        sobel_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        if constexpr (std::is_same<T, Rpp8u>::value)
            rpp_hip_pixel_check_0to255(&sum_f8);
        if constexpr (std::is_same<T, Rpp32f>::value)
            rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
__global__ void sobel_filter_5x5_pln_y_gradient_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       int channelsDst,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    int roiBeginX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiBeginY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    int clampedY = roiBeginY + max(0, min(id_y_i, (roiHeight - 1)));
    d_float8 sum_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filter_row1 = &sobel5x5YHip[0];
    float *filter_row2 = &filter_row1[5];
    float *filter_row3 = &filter_row1[10];
    float *filter_row4 = &filter_row1[15];
    float *filter_row5 = &filter_row1[20];
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + HIP_VECTOR_MAX_INDEX) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
    {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++)
        {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    {
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
        sobel_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
        if constexpr (std::is_same<T, Rpp8u>::value)
            rpp_hip_pixel_check_0to255(&sum_f8);
        else if constexpr (std::is_same<T, Rpp32f>::value)
            rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
__global__ void sobel_filter_7x7_pln_y_gradient_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       int channelsDst,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    int roiBeginX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiBeginY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    int clampedY = roiBeginY + max(0, min(id_y_i, (roiHeight - 1)));
    d_float8 sum_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filter_row1 = &sobel7x7YHip[0];
    float *filter_row2 = &filter_row1[7];
    float *filter_row3 = &filter_row1[14];
    float *filter_row4 = &filter_row1[21];
    float *filter_row5 = &filter_row1[28];
    float *filter_row6 = &filter_row1[35];
    float *filter_row7 = &filter_row1[42];
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + HIP_VECTOR_MAX_INDEX) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else
    {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++)
        {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    {
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, filter_row6);
        sobel_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, filter_row7);
        if constexpr (std::is_same<T, Rpp8u>::value)
            rpp_hip_pixel_check_0to255(&sum_f8);
        else if constexpr (std::is_same<T, Rpp32f>::value)
            rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
RppStatus hip_exec_sobel_filter_tensor(T *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          T *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32u sobelType,
                                          Rpp32u kernelSize,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (SMEM_LENGTH_X - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;
    bool combined = (sobelType == 2);

    if (kernelSize == 3)
    {
        if(combined)
        {
            hipLaunchKernelGGL(sobel_filter_3x3_pln_bidirection_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
        else
        {
            if(!sobelType)
            {
                hipLaunchKernelGGL(sobel_filter_3x3_pln_x_gradient_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   dstDescPtr->c,
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
                HIP_CHECK_LAUNCH_RETURN();
            }
            else
            {
                hipLaunchKernelGGL(sobel_filter_3x3_pln_y_gradient_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   dstDescPtr->c,
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
                HIP_CHECK_LAUNCH_RETURN();
            }
        }
    }
    else if (kernelSize == 5)
    {
        if(combined)
        {
            hipLaunchKernelGGL(sobel_filter_5x5_pln_bidirection_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
        else
        {
            if(!sobelType)
            {
                hipLaunchKernelGGL(sobel_filter_5x5_pln_x_gradient_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   dstDescPtr->c,
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
                HIP_CHECK_LAUNCH_RETURN();
            }
            else
            {
                hipLaunchKernelGGL(sobel_filter_5x5_pln_y_gradient_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   dstDescPtr->c,
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
                HIP_CHECK_LAUNCH_RETURN();
            }
        }
    }
    else if (kernelSize == 7)
    {
        if(combined)
        {
            hipLaunchKernelGGL(sobel_filter_7x7_pln_bidirection_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
        else
        {
            if(!sobelType)
            {
                hipLaunchKernelGGL(sobel_filter_7x7_pln_x_gradient_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   dstDescPtr->c,
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
                HIP_CHECK_LAUNCH_RETURN();
            }
            else
            {
                hipLaunchKernelGGL(sobel_filter_7x7_pln_y_gradient_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   dstDescPtr->c,
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
                HIP_CHECK_LAUNCH_RETURN();
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_sobel_filter_tensor<Rpp8u>(Rpp8u*,
                                                       RpptDescPtr,
                                                       Rpp8u*,
                                                       RpptDescPtr,
                                                       Rpp32u,
                                                       Rpp32u,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);

template RppStatus hip_exec_sobel_filter_tensor<half>(half*,
                                                      RpptDescPtr,
                                                      half*,
                                                      RpptDescPtr,
                                                      Rpp32u,
                                                      Rpp32u,
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      rpp::Handle&);

template RppStatus hip_exec_sobel_filter_tensor<Rpp8s>(Rpp8s*,
                                                       RpptDescPtr,
                                                       Rpp8s*,
                                                       RpptDescPtr,
                                                       Rpp32u,
                                                       Rpp32u,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);

template RppStatus hip_exec_sobel_filter_tensor<Rpp32f>(Rpp32f*,
                                                        RpptDescPtr,
                                                        Rpp32f*,
                                                        RpptDescPtr,
                                                        Rpp32u,
                                                        Rpp32u,
                                                        RpptROIPtr,
                                                        RpptRoiType,
                                                        rpp::Handle&);
