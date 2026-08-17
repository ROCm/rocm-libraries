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
#include "rpp_hip_interpolation.hpp"

// -------------------- Set 0 - warp_affine device helpers --------------------

__device__ void warp_affine_srclocs_hip_compute(float affineMatrixElement,
                                                float4 locSrcComponent_f4, d_float8* locSrcPtr_f8) {
    d_float8 increment_f8;
    increment_f8.f4[0] =
        make_float4(0, affineMatrixElement, affineMatrixElement + affineMatrixElement,
                    affineMatrixElement + affineMatrixElement + affineMatrixElement);
    increment_f8.f4[1] =
        MAKE_FLOAT4(affineMatrixElement + increment_f8.f4[0].w) + increment_f8.f4[0];
    locSrcPtr_f8->f4[0] = locSrcComponent_f4 + increment_f8.f4[0];
    locSrcPtr_f8->f4[1] = locSrcComponent_f4 + increment_f8.f4[1];
}

__device__ void warp_affine_roi_and_srclocs_hip_compute(int4* srcRoiPtr_i4, int id_x, int id_y,
                                                        d_float6* affineMatrix_f6,
                                                        d_float16* locSrc_f16) {
    float2 locDst_f2, locSrc_f2;
    int roiHalfWidth = (srcRoiPtr_i4->z - srcRoiPtr_i4->x + 1) >> 1;
    int roiHalfHeight = (srcRoiPtr_i4->w - srcRoiPtr_i4->y + 1) >> 1;
    locDst_f2.x = (float)(id_x - roiHalfWidth);
    locDst_f2.y = (float)(id_y - roiHalfHeight);
    locSrc_f2.x = fmaf(locDst_f2.x, affineMatrix_f6->f1[0],
                       fmaf(locDst_f2.y, affineMatrix_f6->f1[1], affineMatrix_f6->f1[2])) +
                  roiHalfWidth;
    locSrc_f2.y = fmaf(locDst_f2.x, affineMatrix_f6->f1[3],
                       fmaf(locDst_f2.y, affineMatrix_f6->f1[4], affineMatrix_f6->f1[5])) +
                  roiHalfHeight;
    warp_affine_srclocs_hip_compute(affineMatrix_f6->f1[0], MAKE_FLOAT4(locSrc_f2.x),
                                    &(locSrc_f16->f8[0]));  // Compute 8 locSrcX
    warp_affine_srclocs_hip_compute(affineMatrix_f6->f1[3], MAKE_FLOAT4(locSrc_f2.y),
                                    &(locSrc_f16->f8[1]));  // Compute 8 locSrcY
}

// A thread computes 8 pixels per invocation (id_x, id_x+1, ..., id_x+7), and the store helpers
// below always write all 8 at once with no way to skip individual lanes. When roiWidth is not a
// multiple of 8, the last thread's octet can extend past the packed ROI region into pixels that
// must stay exactly as the caller left them. Save those specific pixels immediately before the
// store and write them straight back immediately after, so the vectorized store is otherwise
// untouched and pixels outside the ROI are never actually modified.
template <typename T>
__device__ __forceinline__ void warp_affine_save_restore_tail(T* dstPixel0, int pixelStride,
                                                              int numChannels, int channelStride,
                                                              int id_x, int roiWidth, T* saved,
                                                              bool save) {
    for (int i = 0; i < 8; i++) {
        if (id_x + i >= roiWidth) {
            for (int c = 0; c < numChannels; c++) {
                T* addr = dstPixel0 + i * pixelStride + c * channelStride;
                T* slot = saved + i * numChannels + c;
                if (save)
                    *slot = *addr;
                else
                    *addr = *slot;
            }
        }
    }
}

// -------------------- Set 1 - Bilinear Interpolation --------------------

template <typename T>
__global__ void warp_affine_bilinear_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                    uint2 dstStridesNH, uint2 dstDimsWH,
                                                    d_float6* affineTensorPtr,
                                                    RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Output is packed at the ROI's own width/height (not the full dst buffer), matching
    // the golden model's convention: read source at the ROI's absolute offset, write dst
    // packed at the origin. dstDimsWH (the full dst buffer size) is not the right bound
    // when the ROI is smaller than the buffer (partial ROI).
    int roiWidth = roiTensorPtrSrc[id_z].ltrbROI.rb.x - roiTensorPtrSrc[id_z].ltrbROI.lt.x + 1;
    int roiHeight = roiTensorPtrSrc[id_z].ltrbROI.rb.y - roiTensorPtrSrc[id_z].ltrbROI.lt.y + 1;
    if ((id_y >= roiHeight) || (id_x >= roiWidth)) {
        return;
    }
    // True when this thread's 8-pixel octet extends past roiWidth (roiWidth not a multiple of 8);
    // the trailing pixels must be saved/restored around the vectorized store below.
    const bool hasTail = (id_x + 7 >= roiWidth);

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float6 affineMatrix_f6 = affineTensorPtr[id_z];
    int4 srcRoi_i4 = *(int4*)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4,
                                        &dst_f24);
    T tailSaved24[24];
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 3, 3, 1, id_x, roiWidth, tailSaved24, true);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 3, 3, 1, id_x, roiWidth, tailSaved24, false);
}

template <typename T>
__global__ void warp_affine_bilinear_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                    uint3 dstStridesNCH, uint2 dstDimsWH,
                                                    int channelsDst, d_float6* affineTensorPtr,
                                                    RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Output is packed at the ROI's own width/height (not the full dst buffer), matching
    // the golden model's convention: read source at the ROI's absolute offset, write dst
    // packed at the origin. dstDimsWH (the full dst buffer size) is not the right bound
    // when the ROI is smaller than the buffer (partial ROI).
    int roiWidth = roiTensorPtrSrc[id_z].ltrbROI.rb.x - roiTensorPtrSrc[id_z].ltrbROI.lt.x + 1;
    int roiHeight = roiTensorPtrSrc[id_z].ltrbROI.rb.y - roiTensorPtrSrc[id_z].ltrbROI.lt.y + 1;
    if ((id_y >= roiHeight) || (id_x >= roiWidth)) {
        return;
    }
    // True when this thread's 8-pixel octet extends past roiWidth (roiWidth not a multiple of 8);
    // the trailing pixels must be saved/restored around the vectorized store below.
    const bool hasTail = (id_x + 7 >= roiWidth);

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float6 affineMatrix_f6 = affineTensorPtr[id_z];
    int4 srcRoi_i4 = *(int4*)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4,
                                       &dst_f8);
    T tailSaved8[8];
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 1, 0, id_x, roiWidth, tailSaved8, true);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 1, 0, id_x, roiWidth, tailSaved8, false);

    if (channelsDst == 3) {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16,
                                           &srcRoi_i4, &dst_f8);
        if (hasTail)
            warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 1, 0, id_x, roiWidth, tailSaved8,
                                          true);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
        if (hasTail)
            warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 1, 0, id_x, roiWidth, tailSaved8,
                                          false);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16,
                                           &srcRoi_i4, &dst_f8);
        if (hasTail)
            warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 1, 0, id_x, roiWidth, tailSaved8,
                                          true);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
        if (hasTail)
            warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 1, 0, id_x, roiWidth, tailSaved8,
                                          false);
    }
}

template <typename T>
__global__ void warp_affine_bilinear_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                          uint3 dstStridesNCH, uint2 dstDimsWH,
                                                          d_float6* affineTensorPtr,
                                                          RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Output is packed at the ROI's own width/height (not the full dst buffer), matching
    // the golden model's convention: read source at the ROI's absolute offset, write dst
    // packed at the origin. dstDimsWH (the full dst buffer size) is not the right bound
    // when the ROI is smaller than the buffer (partial ROI).
    int roiWidth = roiTensorPtrSrc[id_z].ltrbROI.rb.x - roiTensorPtrSrc[id_z].ltrbROI.lt.x + 1;
    int roiHeight = roiTensorPtrSrc[id_z].ltrbROI.rb.y - roiTensorPtrSrc[id_z].ltrbROI.lt.y + 1;
    if ((id_y >= roiHeight) || (id_x >= roiWidth)) {
        return;
    }
    // True when this thread's 8-pixel octet extends past roiWidth (roiWidth not a multiple of 8);
    // the trailing pixels must be saved/restored around the vectorized store below.
    const bool hasTail = (id_x + 7 >= roiWidth);

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float6 affineMatrix_f6 = affineTensorPtr[id_z];
    int4 srcRoi_i4 = *(int4*)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4,
                                        &dst_f24);
    T tailSaved24[24];
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 3, (int)dstStridesNCH.y, id_x, roiWidth,
                                      tailSaved24, true);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 3, (int)dstStridesNCH.y, id_x, roiWidth,
                                      tailSaved24, false);
}

template <typename T>
__global__ void warp_affine_bilinear_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                          uint2 dstStridesNH, uint2 dstDimsWH,
                                                          d_float6* affineTensorPtr,
                                                          RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Output is packed at the ROI's own width/height (not the full dst buffer), matching
    // the golden model's convention: read source at the ROI's absolute offset, write dst
    // packed at the origin. dstDimsWH (the full dst buffer size) is not the right bound
    // when the ROI is smaller than the buffer (partial ROI).
    int roiWidth = roiTensorPtrSrc[id_z].ltrbROI.rb.x - roiTensorPtrSrc[id_z].ltrbROI.lt.x + 1;
    int roiHeight = roiTensorPtrSrc[id_z].ltrbROI.rb.y - roiTensorPtrSrc[id_z].ltrbROI.lt.y + 1;
    if ((id_y >= roiHeight) || (id_x >= roiWidth)) {
        return;
    }
    // True when this thread's 8-pixel octet extends past roiWidth (roiWidth not a multiple of 8);
    // the trailing pixels must be saved/restored around the vectorized store below.
    const bool hasTail = (id_x + 7 >= roiWidth);

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float6 affineMatrix_f6 = affineTensorPtr[id_z];
    int4 srcRoi_i4 = *(int4*)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4,
                                        &dst_f24);
    T tailSaved24[24];
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 3, 3, 1, id_x, roiWidth, tailSaved24, true);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 3, 3, 1, id_x, roiWidth, tailSaved24, false);
}

// -------------------- Set 2 - Nearest Neighbor Interpolation --------------------

template <typename T>
__global__ void warp_affine_nearest_neighbor_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH,
                                                            T* dstPtr, uint2 dstStridesNH,
                                                            uint2 dstDimsWH,
                                                            d_float6* affineTensorPtr,
                                                            RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Output is packed at the ROI's own width/height (not the full dst buffer), matching
    // the golden model's convention: read source at the ROI's absolute offset, write dst
    // packed at the origin. dstDimsWH (the full dst buffer size) is not the right bound
    // when the ROI is smaller than the buffer (partial ROI).
    int roiWidth = roiTensorPtrSrc[id_z].ltrbROI.rb.x - roiTensorPtrSrc[id_z].ltrbROI.lt.x + 1;
    int roiHeight = roiTensorPtrSrc[id_z].ltrbROI.rb.y - roiTensorPtrSrc[id_z].ltrbROI.lt.y + 1;
    if ((id_y >= roiHeight) || (id_x >= roiWidth)) {
        return;
    }
    // True when this thread's 8-pixel octet extends past roiWidth (roiWidth not a multiple of 8);
    // the trailing pixels must be saved/restored around the vectorized store below.
    const bool hasTail = (id_x + 7 >= roiWidth);

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float6 affineMatrix_f6 = affineTensorPtr[id_z];
    int4 srcRoi_i4 = *(int4*)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16,
                                                &srcRoi_i4, &dst_f24);
    T tailSaved24[24];
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 3, 3, 1, id_x, roiWidth, tailSaved24, true);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 3, 3, 1, id_x, roiWidth, tailSaved24, false);
}

template <typename T>
__global__ void warp_affine_nearest_neighbor_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH,
                                                            T* dstPtr, uint3 dstStridesNCH,
                                                            uint2 dstDimsWH, int channelsDst,
                                                            d_float6* affineTensorPtr,
                                                            RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Output is packed at the ROI's own width/height (not the full dst buffer), matching
    // the golden model's convention: read source at the ROI's absolute offset, write dst
    // packed at the origin. dstDimsWH (the full dst buffer size) is not the right bound
    // when the ROI is smaller than the buffer (partial ROI).
    int roiWidth = roiTensorPtrSrc[id_z].ltrbROI.rb.x - roiTensorPtrSrc[id_z].ltrbROI.lt.x + 1;
    int roiHeight = roiTensorPtrSrc[id_z].ltrbROI.rb.y - roiTensorPtrSrc[id_z].ltrbROI.lt.y + 1;
    if ((id_y >= roiHeight) || (id_x >= roiWidth)) {
        return;
    }
    // True when this thread's 8-pixel octet extends past roiWidth (roiWidth not a multiple of 8);
    // the trailing pixels must be saved/restored around the vectorized store below.
    const bool hasTail = (id_x + 7 >= roiWidth);

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float6 affineMatrix_f6 = affineTensorPtr[id_z];
    int4 srcRoi_i4 = *(int4*)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16,
                                               &srcRoi_i4, &dst_f8);
    T tailSaved8[8];
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 1, 0, id_x, roiWidth, tailSaved8, true);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 1, 0, id_x, roiWidth, tailSaved8, false);

    if (channelsDst == 3) {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16,
                                                   &srcRoi_i4, &dst_f8);
        if (hasTail)
            warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 1, 0, id_x, roiWidth, tailSaved8,
                                          true);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
        if (hasTail)
            warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 1, 0, id_x, roiWidth, tailSaved8,
                                          false);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16,
                                                   &srcRoi_i4, &dst_f8);
        if (hasTail)
            warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 1, 0, id_x, roiWidth, tailSaved8,
                                          true);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
        if (hasTail)
            warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 1, 0, id_x, roiWidth, tailSaved8,
                                          false);
    }
}

template <typename T>
__global__ void warp_affine_nearest_neighbor_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH,
                                                                  T* dstPtr, uint3 dstStridesNCH,
                                                                  uint2 dstDimsWH,
                                                                  d_float6* affineTensorPtr,
                                                                  RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Output is packed at the ROI's own width/height (not the full dst buffer), matching
    // the golden model's convention: read source at the ROI's absolute offset, write dst
    // packed at the origin. dstDimsWH (the full dst buffer size) is not the right bound
    // when the ROI is smaller than the buffer (partial ROI).
    int roiWidth = roiTensorPtrSrc[id_z].ltrbROI.rb.x - roiTensorPtrSrc[id_z].ltrbROI.lt.x + 1;
    int roiHeight = roiTensorPtrSrc[id_z].ltrbROI.rb.y - roiTensorPtrSrc[id_z].ltrbROI.lt.y + 1;
    if ((id_y >= roiHeight) || (id_x >= roiWidth)) {
        return;
    }
    // True when this thread's 8-pixel octet extends past roiWidth (roiWidth not a multiple of 8);
    // the trailing pixels must be saved/restored around the vectorized store below.
    const bool hasTail = (id_x + 7 >= roiWidth);

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float6 affineMatrix_f6 = affineTensorPtr[id_z];
    int4 srcRoi_i4 = *(int4*)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16,
                                                &srcRoi_i4, &dst_f24);
    T tailSaved24[24];
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 3, (int)dstStridesNCH.y, id_x, roiWidth,
                                      tailSaved24, true);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 1, 3, (int)dstStridesNCH.y, id_x, roiWidth,
                                      tailSaved24, false);
}

template <typename T>
__global__ void warp_affine_nearest_neighbor_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH,
                                                                  T* dstPtr, uint2 dstStridesNH,
                                                                  uint2 dstDimsWH,
                                                                  d_float6* affineTensorPtr,
                                                                  RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Output is packed at the ROI's own width/height (not the full dst buffer), matching
    // the golden model's convention: read source at the ROI's absolute offset, write dst
    // packed at the origin. dstDimsWH (the full dst buffer size) is not the right bound
    // when the ROI is smaller than the buffer (partial ROI).
    int roiWidth = roiTensorPtrSrc[id_z].ltrbROI.rb.x - roiTensorPtrSrc[id_z].ltrbROI.lt.x + 1;
    int roiHeight = roiTensorPtrSrc[id_z].ltrbROI.rb.y - roiTensorPtrSrc[id_z].ltrbROI.lt.y + 1;
    if ((id_y >= roiHeight) || (id_x >= roiWidth)) {
        return;
    }
    // True when this thread's 8-pixel octet extends past roiWidth (roiWidth not a multiple of 8);
    // the trailing pixels must be saved/restored around the vectorized store below.
    const bool hasTail = (id_x + 7 >= roiWidth);

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float6 affineMatrix_f6 = affineTensorPtr[id_z];
    int4 srcRoi_i4 = *(int4*)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    warp_affine_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16,
                                                &srcRoi_i4, &dst_f24);
    T tailSaved24[24];
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 3, 3, 1, id_x, roiWidth, tailSaved24, true);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
    if (hasTail)
        warp_affine_save_restore_tail(dstPtr + dstIdx, 3, 3, 1, id_x, roiWidth, tailSaved24, false);
}

// -------------------- Set 3 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_warp_affine_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                      RpptDescPtr dstDescPtr, Rpp32f* affineTensorPtr,
                                      RpptInterpolationType interpolationType,
                                      RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                      rpp::Handle& handle) {
    if (roiType == RpptRoiType::XYWH) hip_exec_roi_conversion_xywh_to_ltrb(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if (interpolationType == RpptInterpolationType::BILINEAR) {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC)) {
            hipLaunchKernelGGL(
                warp_affine_bilinear_pkd_hip_tensor,
                dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                     ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                     ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                make_uint2(dstDescPtr->w, dstDescPtr->h), (d_float6*)affineTensorPtr,
                roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            hipLaunchKernelGGL(warp_affine_bilinear_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h), dstDescPtr->c,
                               (d_float6*)affineTensorPtr, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3)) {
            if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                (dstDescPtr->layout == RpptLayout::NCHW)) {
                hipLaunchKernelGGL(
                    warp_affine_bilinear_pkd3_pln3_hip_tensor,
                    dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                         ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                         ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                    srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                    dstPtr,
                    make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                               dstDescPtr->strides.hStride),
                    make_uint2(dstDescPtr->w, dstDescPtr->h), (d_float6*)affineTensorPtr,
                    roiTensorPtrSrc);
                HIP_CHECK_LAUNCH_RETURN();
            } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                       (dstDescPtr->layout == RpptLayout::NHWC)) {
                globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
                hipLaunchKernelGGL(
                    warp_affine_bilinear_pln3_pkd3_hip_tensor,
                    dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                         ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                         ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                    srcPtr,
                    make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                               srcDescPtr->strides.hStride),
                    dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                    make_uint2(dstDescPtr->w, dstDescPtr->h), (d_float6*)affineTensorPtr,
                    roiTensorPtrSrc);
                HIP_CHECK_LAUNCH_RETURN();
            }
        }
    } else if (interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR) {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC)) {
            hipLaunchKernelGGL(
                warp_affine_nearest_neighbor_pkd_hip_tensor,
                dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                     ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                     ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                make_uint2(dstDescPtr->w, dstDescPtr->h), (d_float6*)affineTensorPtr,
                roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            hipLaunchKernelGGL(warp_affine_nearest_neighbor_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h), dstDescPtr->c,
                               (d_float6*)affineTensorPtr, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3)) {
            if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                (dstDescPtr->layout == RpptLayout::NCHW)) {
                hipLaunchKernelGGL(
                    warp_affine_nearest_neighbor_pkd3_pln3_hip_tensor,
                    dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                         ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                         ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                    srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                    dstPtr,
                    make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                               dstDescPtr->strides.hStride),
                    make_uint2(dstDescPtr->w, dstDescPtr->h), (d_float6*)affineTensorPtr,
                    roiTensorPtrSrc);
                HIP_CHECK_LAUNCH_RETURN();
            } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                       (dstDescPtr->layout == RpptLayout::NHWC)) {
                globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
                hipLaunchKernelGGL(
                    warp_affine_nearest_neighbor_pln3_pkd3_hip_tensor,
                    dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                         ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                         ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                    srcPtr,
                    make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                               srcDescPtr->strides.hStride),
                    dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                    make_uint2(dstDescPtr->w, dstDescPtr->h), (d_float6*)affineTensorPtr,
                    roiTensorPtrSrc);
                HIP_CHECK_LAUNCH_RETURN();
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_warp_affine_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                      Rpp32f*, RpptInterpolationType, RpptROIPtr,
                                                      RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_warp_affine_tensor<half>(half*, RpptDescPtr, half*, RpptDescPtr,
                                                     Rpp32f*, RpptInterpolationType, RpptROIPtr,
                                                     RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_warp_affine_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                       Rpp32f*, RpptInterpolationType, RpptROIPtr,
                                                       RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_warp_affine_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                      Rpp32f*, RpptInterpolationType, RpptROIPtr,
                                                      RpptRoiType, rpp::Handle&);
