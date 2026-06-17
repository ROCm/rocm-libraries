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

#include "hip_tensor_executors.hpp"

// -------------------- Set 0 - emboss device helpers --------------------

template <int filterSize, typename T>
__device__ void emboss_row_hip_compute(T* srcPtr, d_float8* dst_f8, float* filter) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int k = 0; k < filterSize; ++k)
            dst_f8->f1[j] = fmaf(srcPtr[j + k], filter[k], dst_f8->f1[j]);
    }
}

// -------------------- Set 1 - PKD3->PKD3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void emboss_3x3_pkd_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr, uint2 dstStridesNH,
                                      uint padLength, uint2 tileSize, RpptROIPtr roiTensorPtrSrc,
                                      float* filterTensor) {
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

    d_float24 sum_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    float* filter_row1 = &filterTensor[id_z * 9];
    float* filter_row2 = &filter_row1[3];
    float* filter_row3 = &filter_row1[6];

    sum_f24.f4[0] = FLOAT4_ZERO;
    sum_f24.f4[1] = FLOAT4_ZERO;
    sum_f24.f4[2] = FLOAT4_ZERO;
    sum_f24.f4[3] = FLOAT4_ZERO;
    sum_f24.f4[4] = FLOAT4_ZERO;
    sum_f24.f4[5] = FLOAT4_ZERO;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = max(roiBeginX, min(id_x_i + i, (roiBeginX + roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }
    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row1);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row1);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row1);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row2);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row2);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row2);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row3);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row3);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row3);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f24);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void emboss_5x5_pkd_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr, uint2 dstStridesNH,
                                      uint padLength, uint2 tileSize, RpptROIPtr roiTensorPtrSrc,
                                      float* filterTensor) {
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

    d_float24 sum_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    float* filter_row1 = &filterTensor[id_z * 25];
    float* filter_row2 = &filter_row1[5];
    float* filter_row3 = &filter_row1[10];
    float* filter_row4 = &filter_row1[15];
    float* filter_row5 = &filter_row1[20];
    sum_f24.f4[0] = FLOAT4_ZERO;
    sum_f24.f4[1] = FLOAT4_ZERO;
    sum_f24.f4[2] = FLOAT4_ZERO;
    sum_f24.f4[3] = FLOAT4_ZERO;
    sum_f24.f4[4] = FLOAT4_ZERO;
    sum_f24.f4[5] = FLOAT4_ZERO;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row1);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row1);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row1);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row2);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row2);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row2);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row3);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row3);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row3);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row4);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row4);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row4);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row5);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row5);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row5);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f24);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void emboss_7x7_pkd_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr, uint2 dstStridesNH,
                                      uint padLength, uint2 tileSize, RpptROIPtr roiTensorPtrSrc,
                                      float* filterTensor) {
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

    d_float24 sum_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    float* filter_row1 = &filterTensor[id_z * 49];
    float* filter_row2 = &filter_row1[7];
    float* filter_row3 = &filter_row1[14];
    float* filter_row4 = &filter_row1[21];
    float* filter_row5 = &filter_row1[28];
    float* filter_row6 = &filter_row1[35];
    float* filter_row7 = &filter_row1[42];
    sum_f24.f4[0] = FLOAT4_ZERO;
    sum_f24.f4[1] = FLOAT4_ZERO;
    sum_f24.f4[2] = FLOAT4_ZERO;
    sum_f24.f4[3] = FLOAT4_ZERO;
    sum_f24.f4[4] = FLOAT4_ZERO;
    sum_f24.f4[5] = FLOAT4_ZERO;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = max(roiBeginX, min(id_x_i + i, (roiBeginX + roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row1);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row1);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row1);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row2);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row2);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row2);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row3);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row3);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row3);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row4);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row4);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row4);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row5);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row5);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row5);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row6);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row6);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row6);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row7);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row7);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row7);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f24);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void emboss_9x9_pkd_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr, uint2 dstStridesNH,
                                      uint padLength, uint2 tileSize, RpptROIPtr roiTensorPtrSrc,
                                      float* filterTensor) {
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

    d_float24 sum_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    float* filter_row1 = &filterTensor[id_z * 81];
    float* filter_row2 = &filter_row1[9];
    float* filter_row3 = &filter_row1[18];
    float* filter_row4 = &filter_row1[27];
    float* filter_row5 = &filter_row1[36];
    float* filter_row6 = &filter_row1[45];
    float* filter_row7 = &filter_row1[54];
    float* filter_row8 = &filter_row1[63];
    float* filter_row9 = &filter_row1[72];
    sum_f24.f4[0] = FLOAT4_ZERO;
    sum_f24.f4[1] = FLOAT4_ZERO;
    sum_f24.f4[2] = FLOAT4_ZERO;
    sum_f24.f4[3] = FLOAT4_ZERO;
    sum_f24.f4[4] = FLOAT4_ZERO;
    sum_f24.f4[5] = FLOAT4_ZERO;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row1);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row1);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row1);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row2);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row2);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row2);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row3);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row3);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row3);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row4);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row4);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row4);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row5);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row5);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row5);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row6);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row6);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row6);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row7);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row7);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row7);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row8);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row8);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row8);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row9);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row9);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row9);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f24);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// -------------------- Set 2 - PLN1->PLN1, PLN3->PLN3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void emboss_3x3_pln_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                      uint3 dstStridesNCH, int channelsDst, uint padLength,
                                      uint2 tileSize, RpptROIPtr roiTensorPtrSrc,
                                      float* filterTensor) {
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
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float* filter_row1 = &filterTensor[id_z * 9];
    float* filter_row2 = &filter_row1[3];
    float* filter_row3 = &filter_row1[6];
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                         &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                srcPtr[clampedIdx];  // Load nearest pixel
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8, filter_row1);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8,
                                  filter_row2);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8,
                                  filter_row3);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }

    if (channelsDst == 3) {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = FLOAT4_ZERO;
        sum_f8.f4[1] = FLOAT4_ZERO;
        if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else {
            // Nearest-neighbor padding
            for (int i = 0; i < 8; i++) {
                int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + srcStridesNCH.y +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }

        __syncthreads();

        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8,
                                      filter_row1);
            emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8,
                                      filter_row2);
            emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8,
                                      filter_row3);
            if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f8);
            if constexpr (std::is_same<T, Rpp8s>::value)
                rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
            else
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = FLOAT4_ZERO;
        sum_f8.f4[1] = FLOAT4_ZERO;
        if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else {
            // Nearest-neighbor padding
            for (int i = 0; i < 8; i++) {
                int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }

        __syncthreads();

        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8,
                                      filter_row1);
            emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8,
                                      filter_row2);
            emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8,
                                      filter_row3);
            if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f8);
            if constexpr (std::is_same<T, Rpp8s>::value)
                rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
            else
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// kernelSize = 5
template <typename T>
__global__ void emboss_5x5_pln_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                      uint3 dstStridesNCH, int channelsDst, uint padLength,
                                      uint2 tileSize, RpptROIPtr roiTensorPtrSrc,
                                      float* filterTensor) {
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

    int srcIdx = (id_z * srcStridesNCH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float* filter_row1 = &filterTensor[id_z * 25];
    float* filter_row2 = &filter_row1[5];
    float* filter_row3 = &filter_row1[10];
    float* filter_row4 = &filter_row1[15];
    float* filter_row5 = &filter_row1[20];
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                         &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                srcPtr[clampedIdx];  // Load nearest pixel
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8, filter_row1);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8,
                                  filter_row2);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8,
                                  filter_row3);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8,
                                  filter_row4);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8,
                                  filter_row5);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }

    if (channelsDst == 3) {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = FLOAT4_ZERO;
        sum_f8.f4[1] = FLOAT4_ZERO;
        if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else {
            // Nearest-neighbor padding
            for (int i = 0; i < 8; i++) {
                int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + srcStridesNCH.y +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }

        __syncthreads();

        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8,
                                      filter_row1);
            emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8,
                                      filter_row2);
            emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8,
                                      filter_row3);
            emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8,
                                      filter_row4);
            emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8,
                                      filter_row5);
            if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f8);
            if constexpr (std::is_same<T, Rpp8s>::value)
                rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
            else
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = FLOAT4_ZERO;
        sum_f8.f4[1] = FLOAT4_ZERO;
        if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else {
            // Nearest-neighbor padding
            for (int i = 0; i < 8; i++) {
                int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }

        __syncthreads();

        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8,
                                      filter_row1);
            emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8,
                                      filter_row2);
            emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8,
                                      filter_row3);
            emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8,
                                      filter_row4);
            emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8,
                                      filter_row5);
            if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f8);
            if constexpr (std::is_same<T, Rpp8s>::value)
                rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
            else
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// kernelSize = 7
template <typename T>
__global__ void emboss_7x7_pln_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                      uint3 dstStridesNCH, int channelsDst, uint padLength,
                                      uint2 tileSize, RpptROIPtr roiTensorPtrSrc,
                                      float* filterTensor) {
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

    int srcIdx = (id_z * srcStridesNCH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float* filter_row1 = &filterTensor[id_z * 49];
    float* filter_row2 = &filter_row1[7];
    float* filter_row3 = &filter_row1[14];
    float* filter_row4 = &filter_row1[21];
    float* filter_row5 = &filter_row1[28];
    float* filter_row6 = &filter_row1[35];
    float* filter_row7 = &filter_row1[42];
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                         &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                srcPtr[clampedIdx];  // Load nearest pixel
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8, filter_row1);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8,
                                  filter_row2);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8,
                                  filter_row3);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8,
                                  filter_row4);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8,
                                  filter_row5);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8,
                                  filter_row6);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8,
                                  filter_row7);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }

    if (channelsDst == 3) {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = FLOAT4_ZERO;
        sum_f8.f4[1] = FLOAT4_ZERO;
        if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else {
            // Nearest-neighbor padding
            for (int i = 0; i < 8; i++) {
                int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }

        __syncthreads();

        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8,
                                      filter_row1);
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8,
                                      filter_row2);
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8,
                                      filter_row3);
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8,
                                      filter_row4);
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8,
                                      filter_row5);
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8,
                                      filter_row6);
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8,
                                      filter_row7);
            if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f8);
            if constexpr (std::is_same<T, Rpp8s>::value)
                rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
            else
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = FLOAT4_ZERO;
        sum_f8.f4[1] = FLOAT4_ZERO;
        if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else {
            // Nearest-neighbor padding
            for (int i = 0; i < 8; i++) {
                int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }

        __syncthreads();

        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8,
                                      filter_row1);
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8,
                                      filter_row2);
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8,
                                      filter_row3);
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8,
                                      filter_row4);
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8,
                                      filter_row5);
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8,
                                      filter_row6);
            emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8,
                                      filter_row7);
            if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f8);
            if constexpr (std::is_same<T, Rpp8s>::value)
                rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
            else
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// kernelSize = 9
template <typename T>
__global__ void emboss_9x9_pln_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                      uint3 dstStridesNCH, int channelsDst, uint padLength,
                                      uint2 tileSize, RpptROIPtr roiTensorPtrSrc,
                                      float* filterTensor) {
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

    int srcIdx = (id_z * srcStridesNCH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float* filter_row1 = &filterTensor[id_z * 81];
    float* filter_row2 = &filter_row1[9];
    float* filter_row3 = &filter_row1[18];
    float* filter_row4 = &filter_row1[27];
    float* filter_row5 = &filter_row1[36];
    float* filter_row6 = &filter_row1[45];
    float* filter_row7 = &filter_row1[54];
    float* filter_row8 = &filter_row1[63];
    float* filter_row9 = &filter_row1[72];
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                         &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                srcPtr[clampedIdx];  // Load nearest pixel
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8, filter_row1);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8,
                                  filter_row2);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8,
                                  filter_row3);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8,
                                  filter_row4);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8,
                                  filter_row5);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8,
                                  filter_row6);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8,
                                  filter_row7);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8,
                                  filter_row8);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8,
                                  filter_row9);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f8);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
        else
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }

    if (channelsDst == 3) {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = FLOAT4_ZERO;
        sum_f8.f4[1] = FLOAT4_ZERO;
        if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else {
            // Nearest-neighbor padding
            for (int i = 0; i < 8; i++) {
                int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }

        __syncthreads();

        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8,
                                      filter_row1);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8,
                                      filter_row2);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8,
                                      filter_row3);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8,
                                      filter_row4);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8,
                                      filter_row5);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8,
                                      filter_row6);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8,
                                      filter_row7);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8,
                                      filter_row8);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8,
                                      filter_row9);
            if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f8);
            if constexpr (std::is_same<T, Rpp8s>::value)
                rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
            else
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        sum_f8.f4[0] = FLOAT4_ZERO;
        sum_f8.f4[1] = FLOAT4_ZERO;
        if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else {
            // Nearest-neighbor padding
            for (int i = 0; i < 8; i++) {
                int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }

        __syncthreads();

        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8,
                                      filter_row1);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8,
                                      filter_row2);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8,
                                      filter_row3);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8,
                                      filter_row4);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8,
                                      filter_row5);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8,
                                      filter_row6);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8,
                                      filter_row7);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8,
                                      filter_row8);
            emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8,
                                      filter_row9);
            if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f8);
            if constexpr (std::is_same<T, Rpp8s>::value)
                rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
            else
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// -------------------- Set 3 - PKD3->PLN3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void emboss_3x3_pkd3_pln3_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                            uint3 dstStridesNCH, uint padLength, uint2 tileSize,
                                            RpptROIPtr roiTensorPtrSrc, float* filterTensor) {
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

    d_float24 sum_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float* filter_row1 = &filterTensor[id_z * 9];
    float* filter_row2 = &filter_row1[3];
    float* filter_row3 = &filter_row1[6];
    sum_f24.f4[0] = FLOAT4_ZERO;
    sum_f24.f4[1] = FLOAT4_ZERO;
    sum_f24.f4[2] = FLOAT4_ZERO;
    sum_f24.f4[3] = FLOAT4_ZERO;
    sum_f24.f4[4] = FLOAT4_ZERO;
    sum_f24.f4[5] = FLOAT4_ZERO;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row1);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row1);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row1);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row2);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row2);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row2);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row3);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row3);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row3);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f24);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pln3<RoundToNearest>(dstPtr + dstIdx,
                                                                       dstStridesNCH.y, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void emboss_5x5_pkd3_pln3_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                            uint3 dstStridesNCH, uint padLength, uint2 tileSize,
                                            RpptROIPtr roiTensorPtrSrc, float* filterTensor) {
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

    d_float24 sum_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float* filter_row1 = &filterTensor[id_z * 25];
    float* filter_row2 = &filter_row1[5];
    float* filter_row3 = &filter_row1[10];
    float* filter_row4 = &filter_row1[15];
    float* filter_row5 = &filter_row1[20];
    sum_f24.f4[0] = FLOAT4_ZERO;
    sum_f24.f4[1] = FLOAT4_ZERO;
    sum_f24.f4[2] = FLOAT4_ZERO;
    sum_f24.f4[3] = FLOAT4_ZERO;
    sum_f24.f4[4] = FLOAT4_ZERO;
    sum_f24.f4[5] = FLOAT4_ZERO;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row1);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row1);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row1);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row2);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row2);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row2);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row3);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row3);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row3);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row4);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row4);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row4);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row5);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row5);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row5);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f24);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pln3<RoundToNearest>(dstPtr + dstIdx,
                                                                       dstStridesNCH.y, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void emboss_7x7_pkd3_pln3_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                            uint3 dstStridesNCH, uint padLength, uint2 tileSize,
                                            RpptROIPtr roiTensorPtrSrc, float* filterTensor) {
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

    d_float24 sum_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float* filter_row1 = &filterTensor[id_z * 49];
    float* filter_row2 = &filter_row1[7];
    float* filter_row3 = &filter_row1[14];
    float* filter_row4 = &filter_row1[21];
    float* filter_row5 = &filter_row1[28];
    float* filter_row6 = &filter_row1[35];
    float* filter_row7 = &filter_row1[42];
    sum_f24.f4[0] = FLOAT4_ZERO;
    sum_f24.f4[1] = FLOAT4_ZERO;
    sum_f24.f4[2] = FLOAT4_ZERO;
    sum_f24.f4[3] = FLOAT4_ZERO;
    sum_f24.f4[4] = FLOAT4_ZERO;
    sum_f24.f4[5] = FLOAT4_ZERO;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row1);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row1);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row1);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row2);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row2);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row2);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row3);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row3);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row3);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row4);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row4);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row4);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row5);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row5);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row5);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row6);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row6);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row6);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row7);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row7);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row7);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f24);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pln3<RoundToNearest>(dstPtr + dstIdx,
                                                                       dstStridesNCH.y, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void emboss_9x9_pkd3_pln3_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                            uint3 dstStridesNCH, uint padLength, uint2 tileSize,
                                            RpptROIPtr roiTensorPtrSrc, float* filterTensor) {
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

    d_float24 sum_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float* filter_row1 = &filterTensor[id_z * 81];
    float* filter_row2 = &filter_row1[9];
    float* filter_row3 = &filter_row1[18];
    float* filter_row4 = &filter_row1[27];
    float* filter_row5 = &filter_row1[36];
    float* filter_row6 = &filter_row1[45];
    float* filter_row7 = &filter_row1[54];
    float* filter_row8 = &filter_row1[63];
    float* filter_row9 = &filter_row1[72];
    sum_f24.f4[0] = FLOAT4_ZERO;
    sum_f24.f4[1] = FLOAT4_ZERO;
    sum_f24.f4[2] = FLOAT4_ZERO;
    sum_f24.f4[3] = FLOAT4_ZERO;
    sum_f24.f4[4] = FLOAT4_ZERO;
    sum_f24.f4[5] = FLOAT4_ZERO;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row1);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row1);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row1);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row2);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row2);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row2);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row3);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row3);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row3);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row4);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row4);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row4);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row5);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row5);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row5);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row6);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row6);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row6);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row7);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row7);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row7);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row8);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row8);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row8);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row9);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row9);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row9);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f24);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pln3<RoundToNearest>(dstPtr + dstIdx,
                                                                       dstStridesNCH.y, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// -------------------- Set 4 - PLN3->PKD3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void emboss_3x3_pln3_pkd3_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                            uint2 dstStridesNH, uint padLength, uint2 tileSize,
                                            RpptROIPtr roiTensorPtrSrc, float* filterTensor) {
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

    d_float24 sum_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) +
               ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
               (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    float* filter_row1 = &filterTensor[id_z * 9];
    float* filter_row2 = &filter_row1[3];
    float* filter_row3 = &filter_row1[6];
    sum_f24.f4[0] = FLOAT4_ZERO;
    sum_f24.f4[1] = FLOAT4_ZERO;
    sum_f24.f4[2] = FLOAT4_ZERO;
    sum_f24.f4[3] = FLOAT4_ZERO;
    sum_f24.f4[4] = FLOAT4_ZERO;
    sum_f24.f4[5] = FLOAT4_ZERO;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight)) {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.x,
                                         &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.y,
                                         &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.z,
                                         &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx0 = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            int clampedIdx1 = clampedIdx0 + srcStridesNCH.y;
            int clampedIdx2 = clampedIdx1 + srcStridesNCH.y;

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx0];  // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx2];  // B
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row1);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row1);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row1);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row2);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row2);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row2);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row3);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row3);
        emboss_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row3);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f24);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void emboss_5x5_pln3_pkd3_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                            uint2 dstStridesNH, uint padLength, uint2 tileSize,
                                            RpptROIPtr roiTensorPtrSrc, float* filterTensor) {
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

    d_float24 sum_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) +
               ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
               (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    float* filter_row1 = &filterTensor[id_z * 25];
    float* filter_row2 = &filter_row1[5];
    float* filter_row3 = &filter_row1[10];
    float* filter_row4 = &filter_row1[15];
    float* filter_row5 = &filter_row1[20];
    sum_f24.f4[0] = FLOAT4_ZERO;
    sum_f24.f4[1] = FLOAT4_ZERO;
    sum_f24.f4[2] = FLOAT4_ZERO;
    sum_f24.f4[3] = FLOAT4_ZERO;
    sum_f24.f4[4] = FLOAT4_ZERO;
    sum_f24.f4[5] = FLOAT4_ZERO;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight)) {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.x,
                                         &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.y,
                                         &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.z,
                                         &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx0 = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            int clampedIdx1 = clampedIdx0 + srcStridesNCH.y;
            int clampedIdx2 = clampedIdx1 + srcStridesNCH.y;

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx0];  // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx2];  // B
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row1);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row1);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row1);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row2);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row2);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row2);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row3);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row3);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row3);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row4);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row4);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row4);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row5);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row5);
        emboss_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row5);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f24);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void emboss_7x7_pln3_pkd3_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                            uint2 dstStridesNH, uint padLength, uint2 tileSize,
                                            RpptROIPtr roiTensorPtrSrc, float* filterTensor) {
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

    d_float24 sum_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) +
               ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
               (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    float* filter_row1 = &filterTensor[id_z * 49];
    float* filter_row2 = &filter_row1[7];
    float* filter_row3 = &filter_row1[14];
    float* filter_row4 = &filter_row1[21];
    float* filter_row5 = &filter_row1[28];
    float* filter_row6 = &filter_row1[35];
    float* filter_row7 = &filter_row1[42];
    sum_f24.f4[0] = FLOAT4_ZERO;
    sum_f24.f4[1] = FLOAT4_ZERO;
    sum_f24.f4[2] = FLOAT4_ZERO;
    sum_f24.f4[3] = FLOAT4_ZERO;
    sum_f24.f4[4] = FLOAT4_ZERO;
    sum_f24.f4[5] = FLOAT4_ZERO;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight)) {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.x,
                                         &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.y,
                                         &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.z,
                                         &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx0 = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            int clampedIdx1 = clampedIdx0 + srcStridesNCH.y;
            int clampedIdx2 = clampedIdx1 + srcStridesNCH.y;

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx0];  // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx2];  // B
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row1);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row1);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row1);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row2);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row2);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row2);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row3);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row3);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row3);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row4);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row4);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row4);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row5);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row5);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row5);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row6);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row6);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row6);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row7);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row7);
        emboss_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row7);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f24);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void emboss_9x9_pln3_pkd3_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                            uint2 dstStridesNH, uint padLength, uint2 tileSize,
                                            RpptROIPtr roiTensorPtrSrc, float* filterTensor) {
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

    d_float24 sum_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) +
               ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
               (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
    float* filter_row1 = &filterTensor[id_z * 81];
    float* filter_row2 = &filter_row1[9];
    float* filter_row3 = &filter_row1[18];
    float* filter_row4 = &filter_row1[27];
    float* filter_row5 = &filter_row1[36];
    float* filter_row6 = &filter_row1[45];
    float* filter_row7 = &filter_row1[54];
    float* filter_row8 = &filter_row1[63];
    float* filter_row9 = &filter_row1[72];
    sum_f24.f4[0] = FLOAT4_ZERO;
    sum_f24.f4[1] = FLOAT4_ZERO;
    sum_f24.f4[2] = FLOAT4_ZERO;
    sum_f24.f4[3] = FLOAT4_ZERO;
    sum_f24.f4[4] = FLOAT4_ZERO;
    sum_f24.f4[5] = FLOAT4_ZERO;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight)) {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.x,
                                         &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.y,
                                         &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.z,
                                         &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx0 = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            int clampedIdx1 = clampedIdx0 + srcStridesNCH.y;
            int clampedIdx2 = clampedIdx1 + srcStridesNCH.y;

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx0];  // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx2];  // B
        }
    }

    __syncthreads();

    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row1);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row1);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row1);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row2);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row2);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row2);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row3);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row3);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row3);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row4);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row4);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row4);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row5);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row5);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row5);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row6);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row6);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row6);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row7);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row7);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row7);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row8);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row8);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row8);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8],
                                  &sum_f24.f8[0], filter_row9);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8],
                                  &sum_f24.f8[1], filter_row9);
        emboss_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8],
                                  &sum_f24.f8[2], filter_row9);
        if constexpr (std::is_same<T, float>::value) rpp_hip_pixel_check_0to1(&sum_f24);
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

__global__ void create_emboss_kernel_3x3(float* filterTensor, float* strengthTensor,
                                         int batchSize) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (id_x >= batchSize) return;

    float* filter = &filterTensor[id_x * 9];  // Each filter is 3x3 = 9 elements
    float strength = strengthTensor[id_x];
    float clampedStrength = (strength > 2.0f) ? 2.0f : strength;

    // 3x3 emboss kernel
    const float baseKernel[9] = {2.0f, 1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 0.0f, -1.0f, -2.0f};

    // Apply strength scaling
    for (int i = 0; i < 9; i++) filter[i] = baseKernel[i] * clampedStrength;
}

__global__ void create_emboss_kernel_5x5(float* filterTensor, float* strengthTensor,
                                         int batchSize) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (id_x >= batchSize) return;

    float* filter = &filterTensor[id_x * 25];
    float strength = strengthTensor[id_x];
    float clampedStrength = (strength > 2.0f) ? 2.0f : strength;

    // 5x5 emboss kernel
    const float baseKernel[25] = {3,  3,  2, 1, 0,  3,  2,  1, 0,  -1, 2,  1, 1,
                                  -1, -2, 1, 0, -1, -2, -3, 0, -1, -2, -3, -3};

    // Apply strength scaling
    for (int i = 0; i < 25; i++) filter[i] = baseKernel[i] * clampedStrength;
}

__global__ void create_emboss_kernel_7x7(float* filterTensor, float* strengthTensor,
                                         int batchSize) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (id_x >= batchSize) return;

    float* filter = &filterTensor[id_x * 49];
    float strength = strengthTensor[id_x];
    float clampedStrength = (strength > 2.0f) ? 2.0f : strength;

    // 7x7 emboss kernel
    const float baseKernel[49] = {4,  5, 4,  3,  2,  1,  0,  5,  3,  3,  2,  1,  0,  -1, 4,  3,  2,
                                  1,  0, -1, -2, 3,  2,  1,  1,  -1, -2, -3, 2,  1,  0,  -1, -2, -3,
                                  -4, 1, 0,  -1, -2, -3, -3, -5, 0,  -1, -2, -3, -4, -5, -4};

    // Apply strength
    for (int i = 0; i < 49; i++) filter[i] = baseKernel[i] * clampedStrength;
}

__global__ void create_emboss_kernel_9x9(float* filterTensor, float* strengthTensor,
                                         int batchSize) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (id_x >= batchSize) return;

    float* filter = &filterTensor[id_x * 81];
    float strength = strengthTensor[id_x];
    float clampedStrength = (strength > 2.0f) ? 2.0f : strength;

    // 9x9 emboss kernel
    const float baseKernel[81] = {5,  7,  6,  5,  4, 3,  2,  1,  0,  7,  4,  5,  4, 3, 2,  1,  0,
                                  -1, 6,  5,  3,  3, 2,  1,  0,  -1, -2, 5,  4,  3, 2, 1,  0,  -1,
                                  -2, -3, 4,  3,  2, 1,  1,  -1, -2, -3, -4, 3,  2, 1, 0,  -1, -2,
                                  -3, -4, -5, 2,  1, 0,  -1, -2, -3, -3, -5, -6, 1, 0, -1, -2, -3,
                                  -4, -5, -4, -7, 0, -1, -2, -3, -4, -5, -6, -7, -5};

    // Apply strength scaling
    for (int i = 0; i < 81; i++) filter[i] = baseKernel[i] * clampedStrength;
}

static RppStatus hip_exec_create_emboss_kernel(Rpp32f* filterTensor, Rpp32s kernelSize,
                                               Rpp32f* strength, rpp::Handle& handle) {
    int globalThreads_x = handle.GetBatchSize();
    int globalThreads_y = 1;
    int globalThreads_z = 1;

    if (kernelSize == 3) {
        hipLaunchKernelGGL(create_emboss_kernel_3x3,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X_1DIM),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y_1DIM),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z_1DIM)),
                           dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           0, handle.GetStream(), filterTensor, strength, handle.GetBatchSize());
    } else if (kernelSize == 5) {
        hipLaunchKernelGGL(create_emboss_kernel_5x5,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X_1DIM),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y_1DIM),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z_1DIM)),
                           dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           0, handle.GetStream(), filterTensor, strength, handle.GetBatchSize());
    } else if (kernelSize == 7) {
        hipLaunchKernelGGL(create_emboss_kernel_7x7,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X_1DIM),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y_1DIM),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z_1DIM)),
                           dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           0, handle.GetStream(), filterTensor, strength, handle.GetBatchSize());
    } else if (kernelSize == 9) {
        hipLaunchKernelGGL(create_emboss_kernel_9x9,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X_1DIM),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y_1DIM),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z_1DIM)),
                           dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           0, handle.GetStream(), filterTensor, strength, handle.GetBatchSize());
    }

    return RPP_SUCCESS;
}

// -------------------- Set 5 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_emboss_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                 RpptDescPtr dstDescPtr, Rpp32f* strength, Rpp32u kernelSize,
                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                 rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB) hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (SMEM_LENGTH_X - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;

    // Create a filter of size (kernel size x kernel size)
    float* filterTensor = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    hip_exec_create_emboss_kernel(filterTensor, kernelSize, strength, handle);

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC)) {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;

        if (kernelSize == 3) {
            hipLaunchKernelGGL(
                emboss_3x3_pkd_tensor,
                dim3(ceil((float)globalThreads_x / tileSize.x),
                     ceil((float)globalThreads_y / tileSize.y),
                     ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                padLength, tileSize, roiTensorPtrSrc, filterTensor);
        } else if (kernelSize == 5) {
            hipLaunchKernelGGL(
                emboss_5x5_pkd_tensor,
                dim3(ceil((float)globalThreads_x / tileSize.x),
                     ceil((float)globalThreads_y / tileSize.y),
                     ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                padLength, tileSize, roiTensorPtrSrc, filterTensor);
        } else if (kernelSize == 7) {
            hipLaunchKernelGGL(
                emboss_7x7_pkd_tensor,
                dim3(ceil((float)globalThreads_x / tileSize.x),
                     ceil((float)globalThreads_y / tileSize.y),
                     ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                padLength, tileSize, roiTensorPtrSrc, filterTensor);
        } else if (kernelSize == 9) {
            hipLaunchKernelGGL(
                emboss_9x9_pkd_tensor,
                dim3(ceil((float)globalThreads_x / tileSize.x),
                     ceil((float)globalThreads_y / tileSize.y),
                     ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                padLength, tileSize, roiTensorPtrSrc, filterTensor);
        }
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        if (kernelSize == 3) {
            hipLaunchKernelGGL(emboss_3x3_pln_tensor,
                               dim3(ceil((float)globalThreads_x / tileSize.x),
                                    ceil((float)globalThreads_y / tileSize.y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               dstDescPtr->c, padLength, tileSize, roiTensorPtrSrc, filterTensor);
        } else if (kernelSize == 5) {
            hipLaunchKernelGGL(emboss_5x5_pln_tensor,
                               dim3(ceil((float)globalThreads_x / tileSize.x),
                                    ceil((float)globalThreads_y / tileSize.y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               dstDescPtr->c, padLength, tileSize, roiTensorPtrSrc, filterTensor);
        } else if (kernelSize == 7) {
            hipLaunchKernelGGL(emboss_7x7_pln_tensor,
                               dim3(ceil((float)globalThreads_x / tileSize.x),
                                    ceil((float)globalThreads_y / tileSize.y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               dstDescPtr->c, padLength, tileSize, roiTensorPtrSrc, filterTensor);
        } else if (kernelSize == 9) {
            hipLaunchKernelGGL(emboss_9x9_pln_tensor,
                               dim3(ceil((float)globalThreads_x / tileSize.x),
                                    ceil((float)globalThreads_y / tileSize.y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               dstDescPtr->c, padLength, tileSize, roiTensorPtrSrc, filterTensor);
        }
    } else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3)) {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            if (kernelSize == 3) {
                hipLaunchKernelGGL(
                    emboss_3x3_pkd3_pln3_tensor,
                    dim3(ceil((float)globalThreads_x / tileSize.x),
                         ceil((float)globalThreads_y / tileSize.y),
                         ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                    srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                    dstPtr,
                    make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                               dstDescPtr->strides.hStride),
                    padLength, tileSize, roiTensorPtrSrc, filterTensor);
            } else if (kernelSize == 5) {
                hipLaunchKernelGGL(
                    emboss_5x5_pkd3_pln3_tensor,
                    dim3(ceil((float)globalThreads_x / tileSize.x),
                         ceil((float)globalThreads_y / tileSize.y),
                         ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                    srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                    dstPtr,
                    make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                               dstDescPtr->strides.hStride),
                    padLength, tileSize, roiTensorPtrSrc, filterTensor);
            } else if (kernelSize == 7) {
                hipLaunchKernelGGL(
                    emboss_7x7_pkd3_pln3_tensor,
                    dim3(ceil((float)globalThreads_x / tileSize.x),
                         ceil((float)globalThreads_y / tileSize.y),
                         ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                    srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                    dstPtr,
                    make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                               dstDescPtr->strides.hStride),
                    padLength, tileSize, roiTensorPtrSrc, filterTensor);
            } else if (kernelSize == 9) {
                hipLaunchKernelGGL(
                    emboss_9x9_pkd3_pln3_tensor,
                    dim3(ceil((float)globalThreads_x / tileSize.x),
                         ceil((float)globalThreads_y / tileSize.y),
                         ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                    srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                    dstPtr,
                    make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                               dstDescPtr->strides.hStride),
                    padLength, tileSize, roiTensorPtrSrc, filterTensor);
            }
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;

            if (kernelSize == 3) {
                hipLaunchKernelGGL(
                    emboss_3x3_pln3_pkd3_tensor,
                    dim3(ceil((float)globalThreads_x / tileSize.x),
                         ceil((float)globalThreads_y / tileSize.y),
                         ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                    srcPtr,
                    make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                               srcDescPtr->strides.hStride),
                    dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                    padLength, tileSize, roiTensorPtrSrc, filterTensor);
            } else if (kernelSize == 5) {
                hipLaunchKernelGGL(
                    emboss_5x5_pln3_pkd3_tensor,
                    dim3(ceil((float)globalThreads_x / tileSize.x),
                         ceil((float)globalThreads_y / tileSize.y),
                         ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                    srcPtr,
                    make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                               srcDescPtr->strides.hStride),
                    dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                    padLength, tileSize, roiTensorPtrSrc, filterTensor);
            } else if (kernelSize == 7) {
                hipLaunchKernelGGL(
                    emboss_7x7_pln3_pkd3_tensor,
                    dim3(ceil((float)globalThreads_x / tileSize.x),
                         ceil((float)globalThreads_y / tileSize.y),
                         ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                    srcPtr,
                    make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                               srcDescPtr->strides.hStride),
                    dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                    padLength, tileSize, roiTensorPtrSrc, filterTensor);
            } else if (kernelSize == 9) {
                hipLaunchKernelGGL(
                    emboss_9x9_pln3_pkd3_tensor,
                    dim3(ceil((float)globalThreads_x / tileSize.x),
                         ceil((float)globalThreads_y / tileSize.y),
                         ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                    srcPtr,
                    make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                               srcDescPtr->strides.hStride),
                    dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                    padLength, tileSize, roiTensorPtrSrc, filterTensor);
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_emboss_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr, Rpp32f*,
                                                 Rpp32u, RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_emboss_tensor<half>(half*, RpptDescPtr, half*, RpptDescPtr, Rpp32f*,
                                                Rpp32u, RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_emboss_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                  Rpp32f*, Rpp32u, RpptROIPtr, RpptRoiType,
                                                  rpp::Handle&);

template RppStatus hip_exec_emboss_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr, Rpp32f*,
                                                 Rpp32u, RpptROIPtr, RpptRoiType, rpp::Handle&);
