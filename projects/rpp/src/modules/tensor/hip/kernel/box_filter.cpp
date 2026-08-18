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

// -------------------- Set 0 - box_filter device helpers --------------------

// Precomputed inverse square values for different kernel sizes (3, 5, 7, 9)
__device__ const float4 kernelSize3InverseSquare = {0.1111111f, 0.1111111f, 0.1111111f, 0.1111111f};
__device__ const float4 kernelSize5InverseSquare = {0.04f, 0.04f, 0.04f, 0.04f};
__device__ const float4 kernelSize7InverseSquare = {0.02040816f, 0.02040816f, 0.02040816f,
                                                    0.02040816f};
__device__ const float4 kernelSize9InverseSquare = {0.01234568f, 0.01234568f, 0.01234568f,
                                                    0.01234568f};

// Box Filter implementation for U8 and I8 datatypes.
template <int filterSize, typename T>
__device__ void box_filter_row_hip_compute(T* srcPtr, d_float8* dst_f8) {
    int sum = 0;
// Initialize the sum with the first 'filterSize' elements
#pragma unroll
    for (int j = 0; j < filterSize; ++j) sum += srcPtr[j];

    // Store the first output by adding the running sum to dst_f8->f1[0]
    dst_f8->f1[0] += sum;

// Slide the window by one element and update the sum:
// add the new element on the right and subtract the old element on the left.
#pragma unroll
    for (int k = 1; k < 8; ++k) {
        sum += srcPtr[k + filterSize - 1];  // Add new rightmost element
        sum -= srcPtr[k - 1];               // Remove old leftmost element
        dst_f8->f1[k] += sum;               // Store updated sum in output
    }
}

// box filter implementation for F16 and F32 datatypes.
template <const int filterSize>
__device__ void box_filter_row_hip_compute(float* srcPtr, d_float8* dst_f8) {
    // source vector type based on kernel size for efficient loading.
    using srcVecType =
        typename std::conditional<(filterSize == 3 || filterSize == 5), d_float12, d_float16>::type;

    // Interpret the srcPtr memory as a vector of floats (srcVecType)
    const srcVecType& src_f = *reinterpret_cast<const srcVecType*>(srcPtr);

    // Load current destination values into local variable for accumulation
    d_float8 localDst = *dst_f8;

// Accumulate sum over kernel window for each of the 8 output elements
#pragma unroll
    for (int i = 0; i < filterSize; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) localDst.f1[j] += src_f.f1[i + j];
    }

    // Write the accumulated sums back to the destination pointer
    *dst_f8 = localDst;
}

// -------------------- Set 1 - PKD3->PKD3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void box_filter_3x3_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                              uint2 dstStridesNH, uint padLength, uint2 tileSize,
                                              RpptROIPtr roiTensorPtrSrc) {
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

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiBeginY) * srcStridesNH.y) +
                 ((id_x_i + roiBeginX) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
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
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 6; i++) sum_f24.f4[i] *= kernelSize3InverseSquare;
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void box_filter_5x5_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                              uint2 dstStridesNH, uint padLength, uint2 tileSize,
                                              RpptROIPtr roiTensorPtrSrc) {
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

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiBeginY) * srcStridesNH.y) +
                 ((id_x_i + roiBeginX) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
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
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 6; i++) sum_f24.f4[i] *= kernelSize5InverseSquare;
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void box_filter_7x7_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                              uint2 dstStridesNH, uint padLength, uint2 tileSize,
                                              RpptROIPtr roiTensorPtrSrc) {
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

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiBeginY) * srcStridesNH.y) +
                 ((id_x_i + roiBeginX) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
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
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 6; i++) sum_f24.f4[i] *= kernelSize7InverseSquare;
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void box_filter_9x9_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                              uint2 dstStridesNH, uint padLength, uint2 tileSize,
                                              RpptROIPtr roiTensorPtrSrc) {
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

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiBeginY) * srcStridesNH.y) +
                 ((id_x_i + roiBeginX) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
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
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 6; i++) sum_f24.f4[i] *= kernelSize9InverseSquare;
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// -------------------- Set 2 - PLN1->PLN1, PLN3->PLN3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void box_filter_3x3_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                              uint3 dstStridesNCH, int channelsDst, uint padLength,
                                              uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
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

    int srcIdx =
        (id_z * srcStridesNCH.x) + ((id_y_i + roiBeginY) * srcStridesNCH.z) + (id_x_i + roiBeginX);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                         &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 2; i++) sum_f8.f4[i] *= kernelSize3InverseSquare;
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
                int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
                int clampedIdx = (id_z * srcStridesNCH.x) + srcStridesNCH.y +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }
        __syncthreads();
        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
// Normalize sum by kernel size inverse factor
#pragma unroll
            for (int i = 0; i < 2; i++) sum_f8.f4[i] *= kernelSize3InverseSquare;
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
                int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }
        __syncthreads();
        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
// Normalize sum by kernel size inverse factor
#pragma unroll
            for (int i = 0; i < 2; i++) sum_f8.f4[i] *= kernelSize3InverseSquare;
            if constexpr (std::is_same<T, Rpp8s>::value)
                rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
            else
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// kernelSize = 5
template <typename T>
__global__ void box_filter_5x5_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                              uint3 dstStridesNCH, int channelsDst, uint padLength,
                                              uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
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

    int srcIdx =
        (id_z * srcStridesNCH.x) + ((id_y_i + roiBeginY) * srcStridesNCH.z) + (id_x_i + roiBeginX);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                         &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 2; i++) sum_f8.f4[i] *= kernelSize5InverseSquare;
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
                int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
                int clampedIdx = (id_z * srcStridesNCH.x) + srcStridesNCH.y +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }
        __syncthreads();
        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
// Normalize sum by kernel size inverse factor
#pragma unroll
            for (int i = 0; i < 2; i++) sum_f8.f4[i] *= kernelSize5InverseSquare;
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
                int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }
        __syncthreads();
        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
// Normalize sum by kernel size inverse factor
#pragma unroll
            for (int i = 0; i < 2; i++) sum_f8.f4[i] *= kernelSize5InverseSquare;
            if constexpr (std::is_same<T, Rpp8s>::value)
                rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
            else
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// kernelSize = 7
template <typename T>
__global__ void box_filter_7x7_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                              uint3 dstStridesNCH, int channelsDst, uint padLength,
                                              uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
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

    int srcIdx =
        (id_z * srcStridesNCH.x) + ((id_y_i + roiBeginY) * srcStridesNCH.z) + (id_x_i + roiBeginX);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                         &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 2; i++) sum_f8.f4[i] *= kernelSize7InverseSquare;
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
                int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
                int clampedIdx = (id_z * srcStridesNCH.x) + (srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }
        __syncthreads();
        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
// Normalize sum by kernel size inverse factor
#pragma unroll
            for (int i = 0; i < 2; i++) sum_f8.f4[i] *= kernelSize7InverseSquare;
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
                int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }
        __syncthreads();
        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
// Normalize sum by kernel size inverse factor
#pragma unroll
            for (int i = 0; i < 2; i++) sum_f8.f4[i] *= kernelSize7InverseSquare;
            if constexpr (std::is_same<T, Rpp8s>::value)
                rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr + dstIdx, &sum_f8);
            else
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
        }
    }
}

// kernelSize = 9
template <typename T>
__global__ void box_filter_9x9_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                              uint3 dstStridesNCH, int channelsDst, uint padLength,
                                              uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
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

    int srcIdx =
        (id_z * srcStridesNCH.x) + ((id_y_i + roiBeginY) * srcStridesNCH.z) + (id_x_i + roiBeginX);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    sum_f8.f4[0] = FLOAT4_ZERO;
    sum_f8.f4[1] = FLOAT4_ZERO;
    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                         &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                srcPtr[clampedIdx];  // Load nearest pixel
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 2; i++) sum_f8.f4[i] *= kernelSize9InverseSquare;
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
                int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
                int clampedIdx = (id_z * srcStridesNCH.x) + (srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }
        __syncthreads();
        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8);
// Normalize sum by kernel size inverse factor
#pragma unroll
            for (int i = 0; i < 2; i++) sum_f8.f4[i] *= kernelSize9InverseSquare;
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
                int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx];  // Load nearest pixel
            }
        }
        __syncthreads();
        if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8);
            box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8);
// Normalize sum by kernel size inverse factor
#pragma unroll
            for (int i = 0; i < 2; i++) sum_f8.f4[i] *= kernelSize9InverseSquare;
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
__global__ void box_filter_3x3_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                    uint3 dstStridesNCH, uint padLength,
                                                    uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
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

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiBeginY) * srcStridesNH.y) +
                 ((id_x_i + roiBeginX) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
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

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight)) {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 6; i++) sum_f24.f4[i] *= kernelSize3InverseSquare;
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pln3<RoundToNearest>(dstPtr + dstIdx,
                                                                       dstStridesNCH.y, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void box_filter_5x5_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                    uint3 dstStridesNCH, uint padLength,
                                                    uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
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

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiBeginY) * srcStridesNH.y) +
                 ((id_x_i + roiBeginX) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
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

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight)) {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 6; i++) sum_f24.f4[i] *= kernelSize5InverseSquare;
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pln3<RoundToNearest>(dstPtr + dstIdx,
                                                                       dstStridesNCH.y, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void box_filter_7x7_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                    uint3 dstStridesNCH, uint padLength,
                                                    uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
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

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiBeginY) * srcStridesNH.y) +
                 ((id_x_i + roiBeginX) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
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

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight)) {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 6; i++) sum_f24.f4[i] *= kernelSize7InverseSquare;
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pln3<RoundToNearest>(dstPtr + dstIdx,
                                                                       dstStridesNCH.y, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &sum_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void box_filter_9x9_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                    uint3 dstStridesNCH, uint padLength,
                                                    uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
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

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiBeginY) * srcStridesNH.y) +
                 ((id_x_i + roiBeginX) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
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

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight)) {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];      // R
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 1];  // G
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[clampedIdx + 2];  // B
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 6; i++) sum_f24.f4[i] *= kernelSize9InverseSquare;
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
__global__ void box_filter_3x3_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                    uint2 dstStridesNH, uint padLength,
                                                    uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
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
    srcIdx.x =
        (id_z * srcStridesNCH.x) + ((id_y_i + roiBeginY) * srcStridesNCH.z) + (id_x_i + roiBeginX);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
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
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
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
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 6; i++) sum_f24.f4[i] *= kernelSize3InverseSquare;
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void box_filter_5x5_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                    uint2 dstStridesNH, uint padLength,
                                                    uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
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
    srcIdx.x =
        (id_z * srcStridesNCH.x) + ((id_y_i + roiBeginY) * srcStridesNCH.z) + (id_x_i + roiBeginX);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
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
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
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
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 6; i++) sum_f24.f4[i] *= kernelSize5InverseSquare;
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void box_filter_7x7_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                    uint2 dstStridesNH, uint padLength,
                                                    uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
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
    srcIdx.x =
        (id_z * srcStridesNCH.x) + ((id_y_i + roiBeginY) * srcStridesNCH.z) + (id_x_i + roiBeginX);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
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
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
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
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 6; i++) sum_f24.f4[i] *= kernelSize7InverseSquare;
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void box_filter_9x9_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                    uint2 dstStridesNH, uint padLength,
                                                    uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
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
    srcIdx.x =
        (id_z * srcStridesNCH.x) + ((id_y_i + roiBeginY) * srcStridesNCH.z) + (id_x_i + roiBeginX);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;
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
            int clampedX = roiBeginX + max(0, min(id_x_i + i, (roiWidth - 1)));
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
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 1][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 2][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 3][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 4][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 5][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 6][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 7][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 7][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 7][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x + 8][hipThreadIdx_x8],
                                      &sum_f24.f8[0]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y + 8][hipThreadIdx_x8],
                                      &sum_f24.f8[1]);
        box_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z + 8][hipThreadIdx_x8],
                                      &sum_f24.f8[2]);
// Normalize sum by kernel size inverse factor
#pragma unroll
        for (int i = 0; i < 6; i++) sum_f24.f4[i] *= kernelSize9InverseSquare;
        if constexpr (std::is_same<T, Rpp8s>::value)
            rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr + dstIdx, &sum_f24);
        else
            rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &sum_f24);
    }
}

// -------------------- Set 5 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_box_filter_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB) hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (128 - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC)) {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;

        using KernelType = void (*)(T*, uint2, T*, uint2, uint, uint2, RpptROIPtr);
        KernelType kernelFn = nullptr;
        switch (kernelSize) {
            case 3:
                kernelFn = box_filter_3x3_pkd_hip_tensor<T>;
                break;
            case 5:
                kernelFn = box_filter_5x5_pkd_hip_tensor<T>;
                break;
            case 7:
                kernelFn = box_filter_7x7_pkd_hip_tensor<T>;
                break;
            case 9:
                kernelFn = box_filter_9x9_pkd_hip_tensor<T>;
                break;
            default:
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        hipLaunchKernelGGL(
            kernelFn,
            dim3(ceil((float)globalThreads_x / tileSize.x),
                 ceil((float)globalThreads_y / tileSize.y),
                 ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(), srcPtr,
            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride), dstPtr,
            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride), padLength,
            tileSize, roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        using KernelType = void (*)(T*, uint3, T*, uint3, int, uint, uint2, RpptROIPtr);
        KernelType kernelFn = nullptr;
        switch (kernelSize) {
            case 3:
                kernelFn = box_filter_3x3_pln_hip_tensor<T>;
                break;
            case 5:
                kernelFn = box_filter_5x5_pln_hip_tensor<T>;
                break;
            case 7:
                kernelFn = box_filter_7x7_pln_hip_tensor<T>;
                break;
            case 9:
                kernelFn = box_filter_9x9_pln_hip_tensor<T>;
                break;
            default:
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        hipLaunchKernelGGL(kernelFn,
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
                           dstDescPtr->c, padLength, tileSize, roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3)) {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            using KernelType = void (*)(T*, uint2, T*, uint3, uint, uint2, RpptROIPtr);
            KernelType kernelFn = nullptr;
            switch (kernelSize) {
                case 3:
                    kernelFn = box_filter_3x3_pkd3_pln3_hip_tensor<T>;
                    break;
                case 5:
                    kernelFn = box_filter_5x5_pkd3_pln3_hip_tensor<T>;
                    break;
                case 7:
                    kernelFn = box_filter_7x7_pkd3_pln3_hip_tensor<T>;
                    break;
                case 9:
                    kernelFn = box_filter_9x9_pkd3_pln3_hip_tensor<T>;
                    break;
                default:
                    return RPP_ERROR_NOT_IMPLEMENTED;
            }
            hipLaunchKernelGGL(kernelFn,
                               dim3(ceil((float)globalThreads_x / tileSize.x),
                                    ceil((float)globalThreads_y / tileSize.y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               padLength, tileSize, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;

            using KernelType = void (*)(T*, uint3, T*, uint2, uint, uint2, RpptROIPtr);
            KernelType kernelFn = nullptr;
            switch (kernelSize) {
                case 3:
                    kernelFn = box_filter_3x3_pln3_pkd3_hip_tensor<T>;
                    break;
                case 5:
                    kernelFn = box_filter_5x5_pln3_pkd3_hip_tensor<T>;
                    break;
                case 7:
                    kernelFn = box_filter_7x7_pln3_pkd3_hip_tensor<T>;
                    break;
                case 9:
                    kernelFn = box_filter_9x9_pln3_pkd3_hip_tensor<T>;
                    break;
                default:
                    return RPP_ERROR_NOT_IMPLEMENTED;
            }
            hipLaunchKernelGGL(kernelFn,
                               dim3(ceil((float)globalThreads_x / tileSize.x),
                                    ceil((float)globalThreads_y / tileSize.y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength, tileSize, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus hip_exec_box_filter_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                           RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                           RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                           rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB) hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = 1;

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (128 - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC)) {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;

        using KernelType = void (*)(T*, uint2, T*, uint2, uint, uint2, RpptROIPtr);
        KernelType kernelFn = nullptr;
        switch (kernelSize) {
            case 3:
                kernelFn = box_filter_3x3_pkd_hip_tensor<T>;
                break;
            case 5:
                kernelFn = box_filter_5x5_pkd_hip_tensor<T>;
                break;
            case 7:
                kernelFn = box_filter_7x7_pkd_hip_tensor<T>;
                break;
            case 9:
                kernelFn = box_filter_9x9_pkd_hip_tensor<T>;
                break;
            default:
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        hipLaunchKernelGGL(
            kernelFn,
            dim3(ceil((float)globalThreads_x / tileSize.x),
                 ceil((float)globalThreads_y / tileSize.y),
                 ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(), srcPtr,
            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride), dstPtr,
            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride), padLength,
            tileSize, roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        using KernelType = void (*)(T*, uint3, T*, uint3, int, uint, uint2, RpptROIPtr);
        KernelType kernelFn = nullptr;
        switch (kernelSize) {
            case 3:
                kernelFn = box_filter_3x3_pln_hip_tensor<T>;
                break;
            case 5:
                kernelFn = box_filter_5x5_pln_hip_tensor<T>;
                break;
            case 7:
                kernelFn = box_filter_7x7_pln_hip_tensor<T>;
                break;
            case 9:
                kernelFn = box_filter_9x9_pln_hip_tensor<T>;
                break;
            default:
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        hipLaunchKernelGGL(kernelFn,
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
                           dstDescPtr->c, padLength, tileSize, roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3)) {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            using KernelType = void (*)(T*, uint2, T*, uint3, uint, uint2, RpptROIPtr);
            KernelType kernelFn = nullptr;
            switch (kernelSize) {
                case 3:
                    kernelFn = box_filter_3x3_pkd3_pln3_hip_tensor<T>;
                    break;
                case 5:
                    kernelFn = box_filter_5x5_pkd3_pln3_hip_tensor<T>;
                    break;
                case 7:
                    kernelFn = box_filter_7x7_pkd3_pln3_hip_tensor<T>;
                    break;
                case 9:
                    kernelFn = box_filter_9x9_pkd3_pln3_hip_tensor<T>;
                    break;
                default:
                    return RPP_ERROR_NOT_IMPLEMENTED;
            }
            hipLaunchKernelGGL(kernelFn,
                               dim3(ceil((float)globalThreads_x / tileSize.x),
                                    ceil((float)globalThreads_y / tileSize.y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               padLength, tileSize, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;

            using KernelType = void (*)(T*, uint3, T*, uint2, uint, uint2, RpptROIPtr);
            KernelType kernelFn = nullptr;
            switch (kernelSize) {
                case 3:
                    kernelFn = box_filter_3x3_pln3_pkd3_hip_tensor<T>;
                    break;
                case 5:
                    kernelFn = box_filter_5x5_pln3_pkd3_hip_tensor<T>;
                    break;
                case 7:
                    kernelFn = box_filter_7x7_pln3_pkd3_hip_tensor<T>;
                    break;
                case 9:
                    kernelFn = box_filter_9x9_pln3_pkd3_hip_tensor<T>;
                    break;
                default:
                    return RPP_ERROR_NOT_IMPLEMENTED;
            }
            hipLaunchKernelGGL(kernelFn,
                               dim3(ceil((float)globalThreads_x / tileSize.x),
                                    ceil((float)globalThreads_y / tileSize.y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength, tileSize, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_box_filter_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                     Rpp32u, RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_box_filter_tensor<half>(half*, RpptDescPtr, half*, RpptDescPtr, Rpp32u,
                                                    RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_box_filter_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                      Rpp32u, RpptROIPtr, RpptRoiType,
                                                      rpp::Handle&);

template RppStatus hip_exec_box_filter_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                     Rpp32u, RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_box_filter_single_image<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                           Rpp32u, RpptROIPtr, RpptRoiType,
                                                           rpp::Handle&);

template RppStatus hip_exec_box_filter_single_image<half>(half*, RpptDescPtr, half*, RpptDescPtr,
                                                          Rpp32u, RpptROIPtr, RpptRoiType,
                                                          rpp::Handle&);

template RppStatus hip_exec_box_filter_single_image<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                            RpptDescPtr, Rpp32u, RpptROIPtr,
                                                            RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_box_filter_single_image<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                           Rpp32u, RpptROIPtr, RpptRoiType,
                                                           rpp::Handle&);
