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

// One thread per image: the spatial Gaussian (kernelSize x kernelSize, row-major) and the range
// exponent factor 1 / (2 * stdDevRange^2). Both are computed in double, and the range factor is
// saturated to FLT_MAX, as on HOST: in float, 2 * sigma^2 underflows for sigma below ~4e-20, the
// factor becomes inf, and the centre tap's 0 * inf gives NaN instead of a weight of 1.
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

// -------------------- Set 1 - bilateral_filter generic kernel --------------------

// One thread per output pixel (id_x, id_y) of image id_z, looping over the channels. Every layout
// pair is addressed through the descriptor strides (n, c, h, w), so one kernel covers PKD3, PLN3,
// PLN1 and the PKD3 <-> PLN3 conversions. The source is read at the ROI offset with the window
// clamped to the ROI on both axes (REPLICATE), and the output is written from the destination
// origin, as on HOST.
template <typename T>
__global__ void bilateral_filter_generic_tensor(T* srcPtr, uint4 srcStridesNCHW, T* dstPtr,
                                                uint4 dstStridesNCHW, int channels, int kernelSize,
                                                RpptROIPtr roiTensorPtrSrc, float* spatialTensor,
                                                float* rangeMulFactorTensor) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int roiBeginX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiBeginY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    if ((id_x >= roiWidth) || (id_y >= roiHeight)) return;

    int padLength = kernelSize / 2;
    float* spatial = &spatialTensor[id_z * kernelSize * kernelSize];
    float rangeMulFactor = rangeMulFactorTensor[id_z];

    int centreIdx =
        ((roiBeginY + id_y) * srcStridesNCHW.z) + ((roiBeginX + id_x) * srcStridesNCHW.w);
    int dstIdx = (id_z * dstStridesNCHW.x) + (id_y * dstStridesNCHW.z) + (id_x * dstStridesNCHW.w);

    for (int c = 0; c < channels; c++) {
        T* srcPtrChannel = srcPtr + (id_z * srcStridesNCHW.x) + (c * srcStridesNCHW.y);
        float centre = (float)srcPtrChannel[centreIdx];
        float sum = 0.0f, weightSum = 0.0f;
        for (int i = 0; i < kernelSize; i++) {
            int rowIdx =
                (roiBeginY + max(0, min(id_y - padLength + i, roiHeight - 1))) * srcStridesNCHW.z;
            for (int j = 0; j < kernelSize; j++) {
                int colIdx = (roiBeginX + max(0, min(id_x - padLength + j, roiWidth - 1))) *
                             srcStridesNCHW.w;
                float pixel = (float)srcPtrChannel[rowIdx + colIdx];
                float diff = pixel - centre;
                float weight = spatial[i * kernelSize + j] * expf(-(diff * diff) * rangeMulFactor);
                sum += weight * pixel;
                weightSum += weight;
            }
        }

        // The centre tap has weight 1, so weightSum >= 1. U8/I8 round to nearest as on HOST.
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

    // One grid layer per image of this call, not per image the handle was created for, so a call
    // with fewer images than the handle's batch size never reads past the given ROIs.
    int batchSize = dstDescPtr->n;

    // Per-image spatial weights, followed by the per-image range factors
    float* spatialTensor = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    float* rangeMulFactorTensor = spatialTensor + batchSize * kernelSize * kernelSize;
    hipLaunchKernelGGL(
        bilateral_filter_weights_tensor, dim3(ceil((float)batchSize / LOCAL_THREADS_X_1DIM)),
        dim3(LOCAL_THREADS_X_1DIM), 0, handle.GetStream(), spatialTensor, rangeMulFactorTensor,
        stdDevSpaceTensor, stdDevRangeTensor, (int)kernelSize, batchSize);
    HIP_CHECK_LAUNCH_RETURN();

    hipLaunchKernelGGL(
        bilateral_filter_generic_tensor<T>,
        dim3(ceil((float)dstDescPtr->w / LOCAL_THREADS_X),
             ceil((float)dstDescPtr->h / LOCAL_THREADS_Y),
             ceil((float)batchSize / LOCAL_THREADS_Z)),
        dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(), srcPtr,
        make_uint4(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                   srcDescPtr->strides.hStride, srcDescPtr->strides.wStride),
        dstPtr,
        make_uint4(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                   dstDescPtr->strides.hStride, dstDescPtr->strides.wStride),
        (int)dstDescPtr->c, (int)kernelSize, roiTensorPtrSrc, spatialTensor, rangeMulFactorTensor);
    HIP_CHECK_LAUNCH_RETURN();

    return RPP_SUCCESS;
}

// The tensor executor sizes its grid by dstDescPtr->n, so it already launches a single grid layer
// for a single image.
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
