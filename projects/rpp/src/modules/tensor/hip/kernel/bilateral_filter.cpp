/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

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

#include <cfloat>

#include "hip_tensor_executors.hpp"

// -------------------- Set 0 - bilateral_filter device helpers --------------------

// Filters 8 consecutive output pixels of one channel. srcSmem is window row 0, column 0 of the
// first pixel; shared memory rows are SMEM_LENGTH_X apart.
template <int kernelSize, typename T>
__device__ void bilateral_filter_row8_hip_compute(T* srcSmem, d_float8* dst_f8, float* spatial,
                                                  float rangeMulFactor) {
    constexpr int padLength = kernelSize / 2;
    // The pixel loop stays rolled: unrolling 8 x kernelSize^2 taps spills registers to scratch
#pragma unroll 1
    for (int p = 0; p < 8; p++) {
        float centre = (float)srcSmem[padLength * SMEM_LENGTH_X + padLength + p];
        float sum = 0.0f, weightSum = 0.0f;
#pragma unroll
        for (int i = 0; i < kernelSize; i++) {
#pragma unroll
            for (int j = 0; j < kernelSize; j++) {
                float pixel = (float)srcSmem[i * SMEM_LENGTH_X + j + p];
                float diff = pixel - centre;
                float weight = spatial[i * kernelSize + j] * expf(-(diff * diff) * rangeMulFactor);
                sum += weight * pixel;
                weightSum += weight;
            }
        }
        // Constant indices keep dst_f8 in registers
        float result = sum / weightSum;
#pragma unroll
        for (int q = 0; q < 8; q++)
            if (q == p) dst_f8->f1[q] = result;
    }
}

// U8/I8 round to nearest and F16/F32 clamp to [0, 1], as on HOST
template <typename T>
__device__ __forceinline__ void bilateral_filter_store1(float pixel, T* dstPtr) {
    if constexpr (std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value)
        pixel = nearbyintf(pixel);
    rpp_hip_pixel_check_and_store(pixel, dstPtr);
}

// Stores the first `count` (< 8) pixels of each channel one at a time, so a row that does not end
// on a multiple of 8 pixels is not written past the ROI
template <int channels, typename T>
__device__ void bilateral_filter_store_tail(T* dstPtr, d_float24* dst_f24, int count, uint wStride,
                                            uint cStride) {
#pragma unroll
    for (int p = 0; p < 7; p++) {
        if (p < count) {
#pragma unroll
            for (int c = 0; c < channels; c++)
                bilateral_filter_store1(dst_f24->f8[c].f1[p], &dstPtr[p * wStride + c * cStride]);
        }
    }
}

template <typename T>
__device__ __forceinline__ void bilateral_filter_store8(T* dstPtr, d_float8* dst_f8) {
    if constexpr (std::is_same<T, Rpp8s>::value) {
        rpp_hip_pack_float8_and_store8<RoundToNearest>(dstPtr, dst_f8);
    } else {
        if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, half>::value)
            rpp_hip_pixel_check_0to1(dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr, dst_f8);
    }
}

template <typename T>
__device__ __forceinline__ void bilateral_filter_store24_pkd3(T* dstPtr, d_float24* dst_f24) {
    if constexpr (std::is_same<T, Rpp8s>::value) {
        rpp_hip_pack_float24_pln3_and_store24_pkd3<RoundToNearest>(dstPtr, dst_f24);
    } else {
        if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, half>::value)
            rpp_hip_pixel_check_0to1(dst_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr, dst_f24);
    }
}

template <typename T>
__device__ __forceinline__ void bilateral_filter_store24_pln3(T* dstPtr, uint cStride,
                                                              d_float24* dst_f24) {
    if constexpr (std::is_same<T, Rpp8s>::value) {
        rpp_hip_pack_float24_pln3_and_store24_pln3<RoundToNearest>(dstPtr, cStride, dst_f24);
    } else {
        if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, half>::value)
            rpp_hip_pixel_check_0to1(dst_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr, cStride, dst_f24);
    }
}

// -------------------- Set 1 - PKD3->PKD3 for T = U8/F32/F16/I8 --------------------

template <typename T, int kernelSize>
__global__ void bilateral_filter_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                uint2 dstStridesNH, uint2 tileSize,
                                                RpptROIPtr roiTensorPtrSrc, float* spatialTensor,
                                                float* rangeMulFactorTensor) {
    constexpr int padLength = kernelSize / 2;
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

    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];
    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiBeginY) * srcStridesNH.y) +
                 ((id_x_i + roiBeginX) * 3);
    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);
            src_smem_channel[0][i] = srcPtr[clampedIdx];
            src_smem_channel[1][i] = srcPtr[clampedIdx + 1];
            src_smem_channel[2][i] = srcPtr[clampedIdx + 2];
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        float* spatial = &spatialTensor[id_z * kernelSize * kernelSize];
        float rangeMulFactor = rangeMulFactorTensor[id_z];
        d_float24 dst_f24;
        bilateral_filter_row8_hip_compute<kernelSize>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],
                                                      &dst_f24.f8[0], spatial, rangeMulFactor);
        bilateral_filter_row8_hip_compute<kernelSize>(
            &src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8], &dst_f24.f8[1], spatial,
            rangeMulFactor);
        bilateral_filter_row8_hip_compute<kernelSize>(
            &src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8], &dst_f24.f8[2], spatial,
            rangeMulFactor);
        int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + (id_x_o * 3);
        if (id_x_o + 8 <= roiWidth)
            bilateral_filter_store24_pkd3(dstPtr + dstIdx, &dst_f24);
        else
            bilateral_filter_store_tail<3>(dstPtr + dstIdx, &dst_f24, roiWidth - id_x_o, 3, 1);
    }
}

// -------------------- Set 2 - PLN1->PLN1, PLN3->PLN3 for T = U8/F32/F16/I8 --------------------

template <typename T, int kernelSize>
__global__ void bilateral_filter_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                uint3 dstStridesNCH, int channelsDst,
                                                uint2 tileSize, RpptROIPtr roiTensorPtrSrc,
                                                float* spatialTensor, float* rangeMulFactorTensor) {
    constexpr int padLength = kernelSize / 2;
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
    bool isOutputThread = (id_x_o < roiWidth) && (id_y_o < roiHeight) &&
                          (hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y);

    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx =
        (id_z * srcStridesNCH.x) + ((id_y_i + roiBeginY) * srcStridesNCH.z) + (id_x_i + roiBeginX);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float* spatial = &spatialTensor[id_z * kernelSize * kernelSize];
    float rangeMulFactor = rangeMulFactorTensor[id_z];

    // One channel at a time through the same tile; the barrier at the top of each later channel
    // keeps its loads from overwriting pixels the previous channel is still reading
    for (int c = 0; c < channelsDst; c++) {
        if (c > 0) __syncthreads();
        if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx + (c * srcStridesNCH.y),
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else {
            // Nearest-neighbor padding
            for (int i = 0; i < 8; i++) {
                int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (c * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[clampedIdx];
            }
        }
        __syncthreads();
        if (isOutputThread) {
            d_float24 dst_f24;
            bilateral_filter_row8_hip_compute<kernelSize>(
                &src_smem[hipThreadIdx_y][hipThreadIdx_x8], &dst_f24.f8[0], spatial,
                rangeMulFactor);
            int dstIdxChannel = dstIdx + (c * dstStridesNCH.y);
            if (id_x_o + 8 <= roiWidth)
                bilateral_filter_store8(dstPtr + dstIdxChannel, &dst_f24.f8[0]);
            else
                bilateral_filter_store_tail<1>(dstPtr + dstIdxChannel, &dst_f24, roiWidth - id_x_o,
                                               1, 0);
        }
    }
}

// -------------------- Set 3 - PKD3->PLN3 for T = U8/F32/F16/I8 --------------------

template <typename T, int kernelSize>
__global__ void bilateral_filter_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                      uint3 dstStridesNCH, uint2 tileSize,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      float* spatialTensor,
                                                      float* rangeMulFactorTensor) {
    constexpr int padLength = kernelSize / 2;
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

    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];
    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiBeginY) * srcStridesNH.y) +
                 ((id_x_i + roiBeginX) * 3);
    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight))
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);
            src_smem_channel[0][i] = srcPtr[clampedIdx];
            src_smem_channel[1][i] = srcPtr[clampedIdx + 1];
            src_smem_channel[2][i] = srcPtr[clampedIdx + 2];
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        float* spatial = &spatialTensor[id_z * kernelSize * kernelSize];
        float rangeMulFactor = rangeMulFactorTensor[id_z];
        d_float24 dst_f24;
        bilateral_filter_row8_hip_compute<kernelSize>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],
                                                      &dst_f24.f8[0], spatial, rangeMulFactor);
        bilateral_filter_row8_hip_compute<kernelSize>(
            &src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8], &dst_f24.f8[1], spatial,
            rangeMulFactor);
        bilateral_filter_row8_hip_compute<kernelSize>(
            &src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8], &dst_f24.f8[2], spatial,
            rangeMulFactor);
        int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
        if (id_x_o + 8 <= roiWidth)
            bilateral_filter_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
        else
            bilateral_filter_store_tail<3>(dstPtr + dstIdx, &dst_f24, roiWidth - id_x_o, 1,
                                           dstStridesNCH.y);
    }
}

// -------------------- Set 4 - PLN3->PKD3 for T = U8/F32/F16/I8 --------------------

template <typename T, int kernelSize>
__global__ void bilateral_filter_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                      uint2 dstStridesNH, uint2 tileSize,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      float* spatialTensor,
                                                      float* rangeMulFactorTensor) {
    constexpr int padLength = kernelSize / 2;
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

    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx =
        (id_z * srcStridesNCH.x) + ((id_y_i + roiBeginY) * srcStridesNCH.z) + (id_x_i + roiBeginX);
    if ((id_x_i >= 0) && ((id_x_i + 7) < roiWidth) && (id_y_i >= 0) && (id_y_i < roiHeight)) {
        for (int c = 0; c < 3; c++)
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx + (c * srcStridesNCH.y),
                                             &src_smem[hipThreadIdx_y + 16 * c][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        for (int i = 0; i < 8; i++) {
            int clampedX = roiBeginX + max(0, min(id_x_i + i, roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            for (int c = 0; c < 3; c++)
                src_smem[hipThreadIdx_y + 16 * c][hipThreadIdx_x8 + i] =
                    srcPtr[clampedIdx + (c * srcStridesNCH.y)];
        }
    }
    __syncthreads();
    if ((id_x_o < roiWidth) && (id_y_o < roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        float* spatial = &spatialTensor[id_z * kernelSize * kernelSize];
        float rangeMulFactor = rangeMulFactorTensor[id_z];
        d_float24 dst_f24;
        bilateral_filter_row8_hip_compute<kernelSize>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],
                                                      &dst_f24.f8[0], spatial, rangeMulFactor);
        bilateral_filter_row8_hip_compute<kernelSize>(
            &src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8], &dst_f24.f8[1], spatial,
            rangeMulFactor);
        bilateral_filter_row8_hip_compute<kernelSize>(
            &src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8], &dst_f24.f8[2], spatial,
            rangeMulFactor);
        int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + (id_x_o * 3);
        if (id_x_o + 8 <= roiWidth)
            bilateral_filter_store24_pkd3(dstPtr + dstIdx, &dst_f24);
        else
            bilateral_filter_store_tail<3>(dstPtr + dstIdx, &dst_f24, roiWidth - id_x_o, 3, 1);
    }
}

// -------------------- Set 5 - Kernel Executors --------------------

// The range factor is capped at FLT_MAX, as on HOST: for tiny sigma it overflows float, and the
// centre tap's 0 * inf would be NaN instead of a weight of 1.
__global__ void bilateral_filter_weights_tensor(float* spatialTensor, float* rangeMulFactorTensor,
                                                float* stdDevSpaceTensor, float* stdDevRangeTensor,
                                                int kernelSize, int batchSize) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;  // one thread per image
    if (id_x >= batchSize) return;

    int padLength = kernelSize / 2;
    double spaceStdDev = stdDevSpaceTensor[id_x];
    double spaceMulFactor = 1.0 / (2.0 * spaceStdDev * spaceStdDev);
    float* spatial = &spatialTensor[id_x * kernelSize * kernelSize];
    for (int i = -padLength; i <= padLength; i++)
        for (int j = -padLength; j <= padLength; j++)
            spatial[(i + padLength) * kernelSize + (j + padLength)] =
                exp(-(i * i + j * j) * spaceMulFactor);

    double rangeStdDev = stdDevRangeTensor[id_x];
    rangeMulFactorTensor[id_x] = fmin(1.0 / (2.0 * rangeStdDev * rangeStdDev), (double)FLT_MAX);
}

template <typename T, int kernelSize>
static RppStatus hip_exec_bilateral_filter_layout(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  RpptROIPtr roiTensorPtrSrc, float* spatialTensor,
                                                  float* rangeMulFactorTensor, int batchSize,
                                                  rpp::Handle& handle) {
    uint padLength = kernelSize / 2;
    uint2 tileSize;
    tileSize.x = (SMEM_LENGTH_X - 2 * padLength) / 8;
    tileSize.y = 16 - 2 * padLength;

    int globalThreads_x = (dstDescPtr->w + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = batchSize;
    dim3 grid(ceil((float)globalThreads_x / tileSize.x), ceil((float)globalThreads_y / tileSize.y),
              ceil((float)globalThreads_z / LOCAL_THREADS_Z));
    dim3 block(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z);

    uint2 srcStridesNH = make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride);
    uint2 dstStridesNH = make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride);
    uint3 srcStridesNCH = make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                     srcDescPtr->strides.hStride);
    uint3 dstStridesNCH = make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                     dstDescPtr->strides.hStride);

    // A single channel is laid out the same way in NCHW and NHWC, so it always takes PLN1->PLN1
    if ((srcDescPtr->c == 1) ||
        ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))) {
        hipLaunchKernelGGL((bilateral_filter_pln_hip_tensor<T, kernelSize>), grid, block, 0,
                           handle.GetStream(), srcPtr, srcStridesNCH, dstPtr, dstStridesNCH,
                           (int)dstDescPtr->c, tileSize, roiTensorPtrSrc, spatialTensor,
                           rangeMulFactorTensor);
    } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NHWC)) {
        hipLaunchKernelGGL((bilateral_filter_pkd_hip_tensor<T, kernelSize>), grid, block, 0,
                           handle.GetStream(), srcPtr, srcStridesNH, dstPtr, dstStridesNH, tileSize,
                           roiTensorPtrSrc, spatialTensor, rangeMulFactorTensor);
    } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        hipLaunchKernelGGL((bilateral_filter_pkd3_pln3_hip_tensor<T, kernelSize>), grid, block, 0,
                           handle.GetStream(), srcPtr, srcStridesNH, dstPtr, dstStridesNCH,
                           tileSize, roiTensorPtrSrc, spatialTensor, rangeMulFactorTensor);
    } else {
        hipLaunchKernelGGL((bilateral_filter_pln3_pkd3_hip_tensor<T, kernelSize>), grid, block, 0,
                           handle.GetStream(), srcPtr, srcStridesNCH, dstPtr, dstStridesNH,
                           tileSize, roiTensorPtrSrc, spatialTensor, rangeMulFactorTensor);
    }
    HIP_CHECK_LAUNCH_RETURN();

    return RPP_SUCCESS;
}

template <typename T>
RppStatus hip_exec_bilateral_filter_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                           RpptDescPtr dstDescPtr, Rpp32f* stdDevSpaceTensor,
                                           Rpp32f* stdDevRangeTensor, Rpp32u kernelSize,
                                           RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                           rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB)
        RPP_RETURN_IF_ERROR(hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle));

    // Images in this call, not the handle's batch size, so no read past the given ROIs
    int batchSize = dstDescPtr->n;

    float* spatialTensor = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    float* rangeMulFactorTensor = spatialTensor + batchSize * kernelSize * kernelSize;
    hipLaunchKernelGGL(
        bilateral_filter_weights_tensor, dim3(ceil((float)batchSize / LOCAL_THREADS_X_1DIM)),
        dim3(LOCAL_THREADS_X_1DIM), 0, handle.GetStream(), spatialTensor, rangeMulFactorTensor,
        stdDevSpaceTensor, stdDevRangeTensor, (int)kernelSize, batchSize);
    HIP_CHECK_LAUNCH_RETURN();

    switch (kernelSize) {
        case 3:
            return hip_exec_bilateral_filter_layout<T, 3>(srcPtr, srcDescPtr, dstPtr, dstDescPtr,
                                                          roiTensorPtrSrc, spatialTensor,
                                                          rangeMulFactorTensor, batchSize, handle);
        case 5:
            return hip_exec_bilateral_filter_layout<T, 5>(srcPtr, srcDescPtr, dstPtr, dstDescPtr,
                                                          roiTensorPtrSrc, spatialTensor,
                                                          rangeMulFactorTensor, batchSize, handle);
        case 7:
            return hip_exec_bilateral_filter_layout<T, 7>(srcPtr, srcDescPtr, dstPtr, dstDescPtr,
                                                          roiTensorPtrSrc, spatialTensor,
                                                          rangeMulFactorTensor, batchSize, handle);
        case 9:
            return hip_exec_bilateral_filter_layout<T, 9>(srcPtr, srcDescPtr, dstPtr, dstDescPtr,
                                                          roiTensorPtrSrc, spatialTensor,
                                                          rangeMulFactorTensor, batchSize, handle);
        default:
            return RPP_ERROR_NOT_IMPLEMENTED;
    }
}

template RppStatus hip_exec_bilateral_filter_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                           Rpp32f*, Rpp32f*, Rpp32u, RpptROIPtr,
                                                           RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_tensor<half>(half*, RpptDescPtr, half*, RpptDescPtr,
                                                          Rpp32f*, Rpp32f*, Rpp32u, RpptROIPtr,
                                                          RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                            RpptDescPtr, Rpp32f*, Rpp32f*, Rpp32u,
                                                            RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                           Rpp32f*, Rpp32f*, Rpp32u, RpptROIPtr,
                                                           RpptRoiType, rpp::Handle&);
