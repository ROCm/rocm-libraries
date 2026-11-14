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

// -------------------- Set 0 - bilateral_filter weight kernel --------------------

// The range factor is capped at FLT_MAX, as on HOST: for tiny sigma it overflows float, and the
// centre tap's 0 * inf would be NaN instead of a weight of 1.
__global__ void bilateral_filter_weights_tensor(float* spatialTensor, float* rangeMulFactorTensor,
                                                float* stdDevSpaceTensor, float* stdDevRangeTensor,
                                                int kernelSize, int batchSize) {
    int id_z = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (id_z >= batchSize) return;

    int padLength = kernelSize / 2;
    double spaceStdDev = stdDevSpaceTensor[id_z];
    double spaceMulFactor = 1.0 / (2.0 * spaceStdDev * spaceStdDev);
    float* spatial = &spatialTensor[id_z * kernelSize * kernelSize];
    for (int i = -padLength; i <= padLength; i++)
        for (int j = -padLength; j <= padLength; j++)
            spatial[(i + padLength) * kernelSize + (j + padLength)] =
                exp(-(i * i + j * j) * spaceMulFactor);

    double rangeStdDev = stdDevRangeTensor[id_z];
    rangeMulFactorTensor[id_z] = fmin(1.0 / (2.0 * rangeStdDev * rangeStdDev), (double)FLT_MAX);
}

// -------------------- Set 1 - bilateral_filter tiled kernel --------------------

// Must be launched with LOCAL_THREADS_X x LOCAL_THREADS_Y blocks: the tile is sized from them
template <typename T, int kernelSize>
__global__ void bilateral_filter_tiled_tensor(T* srcPtr, uint4 srcStridesNCHW, T* dstPtr,
                                              uint4 dstStridesNCHW, int channels,
                                              RpptROIPtr roiTensorPtrSrc, float* spatialTensor,
                                              float* rangeMulFactorTensor) {
    constexpr int padLength = kernelSize / 2;
    constexpr int tileWidth = LOCAL_THREADS_X + 2 * padLength;
    constexpr int tileHeight = LOCAL_THREADS_Y + 2 * padLength;
    constexpr int threadsPerBlock = LOCAL_THREADS_X * LOCAL_THREADS_Y;

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int roiBeginX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiBeginY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;

    // Uniform across the block, so no thread is left waiting at __syncthreads()
    int blockBeginX = hipBlockIdx_x * hipBlockDim_x;
    int blockBeginY = hipBlockIdx_y * hipBlockDim_y;
    if ((blockBeginX >= roiWidth) || (blockBeginY >= roiHeight)) return;

    __shared__ float src_smem[3][tileHeight][tileWidth];
    T* srcPtrImage = srcPtr + (id_z * srcStridesNCHW.x);

#pragma unroll
    for (int idx = hipThreadIdx_y * LOCAL_THREADS_X + hipThreadIdx_x; idx < tileWidth * tileHeight;
         idx += threadsPerBlock) {
        int ty = idx / tileWidth;
        int tx = idx % tileWidth;
        int srcY = roiBeginY + max(0, min(blockBeginY - padLength + ty, roiHeight - 1));
        int srcX = roiBeginX + max(0, min(blockBeginX - padLength + tx, roiWidth - 1));
        int srcIdx = (srcY * srcStridesNCHW.z) + (srcX * srcStridesNCHW.w);
        for (int c = 0; c < channels; c++)
            src_smem[c][ty][tx] = (float)srcPtrImage[srcIdx + c * srcStridesNCHW.y];
    }

    __syncthreads();

    // Guard only after the barrier, which every thread must reach
    if ((id_x >= roiWidth) || (id_y >= roiHeight)) return;

    float* spatial = &spatialTensor[id_z * kernelSize * kernelSize];
    float rangeMulFactor = rangeMulFactorTensor[id_z];
    int dstIdx = (id_z * dstStridesNCHW.x) + (id_y * dstStridesNCHW.z) + (id_x * dstStridesNCHW.w);

    for (int c = 0; c < channels; c++) {
        float centre = src_smem[c][hipThreadIdx_y + padLength][hipThreadIdx_x + padLength];
        float sum = 0.0f, weightSum = 0.0f;
#pragma unroll
        for (int i = 0; i < kernelSize; i++) {
#pragma unroll
            for (int j = 0; j < kernelSize; j++) {
                float pixel = src_smem[c][hipThreadIdx_y + i][hipThreadIdx_x + j];
                float diff = pixel - centre;
                float weight = spatial[i * kernelSize + j] * expf(-(diff * diff) * rangeMulFactor);
                sum += weight * pixel;
                weightSum += weight;
            }
        }

        float result = sum / weightSum;
        if constexpr (std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value)
            result = nearbyintf(result);
        rpp_hip_pixel_check_and_store(result, &dstPtr[dstIdx + c * dstStridesNCHW.y]);
    }
}

// -------------------- Set 2 - bilateral_filter executors --------------------

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

    using TiledKernelType = void (*)(T*, uint4, T*, uint4, int, RpptROIPtr, float*, float*);
    TiledKernelType kernelFn = nullptr;
    switch (kernelSize) {
        case 3:
            kernelFn = bilateral_filter_tiled_tensor<T, 3>;
            break;
        case 5:
            kernelFn = bilateral_filter_tiled_tensor<T, 5>;
            break;
        case 7:
            kernelFn = bilateral_filter_tiled_tensor<T, 7>;
            break;
        case 9:
            kernelFn = bilateral_filter_tiled_tensor<T, 9>;
            break;
        default:
            return RPP_ERROR_NOT_IMPLEMENTED;
    }
    hipLaunchKernelGGL(kernelFn,
                       dim3(ceil((float)dstDescPtr->w / LOCAL_THREADS_X),
                            ceil((float)dstDescPtr->h / LOCAL_THREADS_Y),
                            ceil((float)batchSize / LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                       handle.GetStream(), srcPtr,
                       make_uint4(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                  srcDescPtr->strides.hStride, srcDescPtr->strides.wStride),
                       dstPtr,
                       make_uint4(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                  dstDescPtr->strides.hStride, dstDescPtr->strides.wStride),
                       (int)dstDescPtr->c, roiTensorPtrSrc, spatialTensor, rangeMulFactorTensor);
    HIP_CHECK_LAUNCH_RETURN();

    return RPP_SUCCESS;
}

template <typename T>
RppStatus hip_exec_bilateral_filter_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                                 RpptDescPtr dstDescPtr, Rpp32f* stdDevSpaceTensor,
                                                 Rpp32f* stdDevRangeTensor, Rpp32u kernelSize,
                                                 RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                                 rpp::Handle& handle) {
    return hip_exec_bilateral_filter_tensor(srcPtr, srcDescPtr, dstPtr, dstDescPtr,
                                            stdDevSpaceTensor, stdDevRangeTensor, kernelSize,
                                            roiTensorPtrSrc, roiType, handle);
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

template RppStatus hip_exec_bilateral_filter_single_image<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*,
                                                                 RpptDescPtr, Rpp32f*, Rpp32f*,
                                                                 Rpp32u, RpptROIPtr, RpptRoiType,
                                                                 rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_single_image<half>(half*, RpptDescPtr, half*,
                                                                RpptDescPtr, Rpp32f*, Rpp32f*,
                                                                Rpp32u, RpptROIPtr, RpptRoiType,
                                                                rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_single_image<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                                  RpptDescPtr, Rpp32f*, Rpp32f*,
                                                                  Rpp32u, RpptROIPtr, RpptRoiType,
                                                                  rpp::Handle&);

template RppStatus hip_exec_bilateral_filter_single_image<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*,
                                                                 RpptDescPtr, Rpp32f*, Rpp32f*,
                                                                 Rpp32u, RpptROIPtr, RpptRoiType,
                                                                 rpp::Handle&);
