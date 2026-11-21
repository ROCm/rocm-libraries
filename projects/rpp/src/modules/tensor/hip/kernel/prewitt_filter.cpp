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

#include "hip_tensor_executors.hpp"

// -------------------- Set 0 - prewitt_filter tiled kernel --------------------

// Must be launched with LOCAL_THREADS_X x LOCAL_THREADS_Y blocks: the tile is sized from them
template <typename T>
__global__ void prewitt_filter_tiled_tensor(T* srcPtr, uint4 srcStridesNCHW, T* dstPtr,
                                            uint4 dstStridesNCHW, int channels, int prewittType,
                                            RpptROIPtr roiTensorPtrSrc) {
    constexpr int padLength = 1;
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

    int blockBeginX = hipBlockIdx_x * hipBlockDim_x;
    int blockBeginY = hipBlockIdx_y * hipBlockDim_y;
    if ((blockBeginX >= roiWidth) || (blockBeginY >= roiHeight)) return;

    __shared__ float src_smem[3][tileHeight][tileWidth];
    T* srcPtrImage = srcPtr + (id_z * srcStridesNCHW.x);

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

    if ((id_x >= roiWidth) || (id_y >= roiHeight)) return;

    int dstIdx = (id_z * dstStridesNCHW.x) + (id_y * dstStridesNCHW.z) + (id_x * dstStridesNCHW.w);
    int ty = hipThreadIdx_y;
    int tx = hipThreadIdx_x;

    for (int c = 0; c < channels; c++) {
        float topLeft = src_smem[c][ty][tx];
        float top = src_smem[c][ty][tx + 1];
        float topRight = src_smem[c][ty][tx + 2];
        float left = src_smem[c][ty + 1][tx];
        float right = src_smem[c][ty + 1][tx + 2];
        float bottomLeft = src_smem[c][ty + 2][tx];
        float bottom = src_smem[c][ty + 2][tx + 1];
        float bottomRight = src_smem[c][ty + 2][tx + 2];

        float gx = (topRight + right + bottomRight) - (topLeft + left + bottomLeft);
        float gy = (bottomLeft + bottom + bottomRight) - (topLeft + top + topRight);

        // prewittType 0 = X, 1 = Y, 2 = XY
        float result = (prewittType == 0) ? gx : (prewittType == 1) ? gy : sqrtf(gx * gx + gy * gy);

        if constexpr (std::is_same<T, Rpp8u>::value) {
            result = nearbyintf(result);
        } else if constexpr (std::is_same<T, Rpp8s>::value) {
            // The HIP i8 store only clamps, so shift back to stored space here
            result = nearbyintf(result) - 128.0f;
        }
        rpp_hip_pixel_check_and_store(result, &dstPtr[dstIdx + c * dstStridesNCHW.y]);
    }
}

// -------------------- Set 1 - prewitt_filter executor --------------------

template <typename T>
RppStatus hip_exec_prewitt_filter_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                         RpptDescPtr dstDescPtr, Rpp32u prewittType,
                                         RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                         rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB)
        RPP_RETURN_IF_ERROR(hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle));

    int batchSize = dstDescPtr->n;

    hipLaunchKernelGGL(prewitt_filter_tiled_tensor<T>,
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
                       (int)dstDescPtr->c, (int)prewittType, roiTensorPtrSrc);
    HIP_CHECK_LAUNCH_RETURN();

    return RPP_SUCCESS;
}

template RppStatus hip_exec_prewitt_filter_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                         Rpp32u, RpptROIPtr, RpptRoiType,
                                                         rpp::Handle&);

template RppStatus hip_exec_prewitt_filter_tensor<half>(half*, RpptDescPtr, half*, RpptDescPtr,
                                                        Rpp32u, RpptROIPtr, RpptRoiType,
                                                        rpp::Handle&);

template RppStatus hip_exec_prewitt_filter_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                          RpptDescPtr, Rpp32u, RpptROIPtr,
                                                          RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_prewitt_filter_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                         Rpp32u, RpptROIPtr, RpptRoiType,
                                                         rpp::Handle&);
