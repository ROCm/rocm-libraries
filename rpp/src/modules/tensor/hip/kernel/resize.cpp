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

// -------------------- Set 0 - resize device helpers --------------------

__device__ void resize_roi_and_srclocs_hip_compute(int4 *srcRoiPtr_i4, uint2 *dstDimsWH, int id_x, int id_y, d_float16 *locSrc_f16)
{
    float wRatio = (float)(srcRoiPtr_i4->z - srcRoiPtr_i4->x + 1) / dstDimsWH->x;
    float hRatio = (float)(srcRoiPtr_i4->w - srcRoiPtr_i4->y + 1) / dstDimsWH->y;
    float4 wOffset_f4 = (float4)((wRatio - 1) * 0.5f);
    float4 hOffset_f4 = (float4)((hRatio - 1) * 0.5f);

    d_float8 increment_f8, locDst_f8x, locDst_f8y;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    locDst_f8x.f4[0] = (float4)id_x + increment_f8.f4[0];
    locDst_f8x.f4[1] = (float4)id_x + increment_f8.f4[1];
    locDst_f8y.f4[0] = (float4)id_y;
    locDst_f8y.f4[1] = (float4)id_y;

    locSrc_f16->f8[0].f4[0] = (locDst_f8x.f4[0] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute src x locations in float for dst x locations [0-3]
    locSrc_f16->f8[0].f4[1] = (locDst_f8x.f4[1] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute src x locations in float for dst x locations [4-7]
    locSrc_f16->f8[1].f4[0] = (locDst_f8y.f4[0] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute src y locations in float for dst y locations [0-3]
    locSrc_f16->f8[1].f4[1] = (locDst_f8y.f4[1] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute src y locations in float for dst y locations [4-7]
}

__device__ void resize_roi_generic_srcloc_and_weight_hip_compute(int roiLoc, int dstLocation, float scale, int limit, int *srcLoc, float *weight, float offset, int srcStride)
{
    float srcLocationRaw = ((float) dstLocation) * scale + offset + (float)roiLoc;
    int srcLocationRounded = (int)ceilf(srcLocationRaw);
    *weight = srcLocationRounded - srcLocationRaw;
    *srcLoc = ((srcLocationRounded > limit) ? limit : srcLocationRounded) * srcStride;
}

// -------------------- Set 1 - Nearest Neighbor Interpolation --------------------

template <typename T>
__global__ void resize_nearest_neighbor_pkd_hip_tensor(T *srcPtr,
                                                       uint2 srcStridesNH,
                                                       T *dstPtr,
                                                       uint2 dstStridesNH,
                                                       RpptImagePatchPtr dstImgSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void resize_nearest_neighbor_pln_hip_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       int channelsDst,
                                                       RpptImagePatchPtr dstImgSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void resize_nearest_neighbor_pkd3_pln3_hip_tensor(T *srcPtr,
                                                             uint2 srcStridesNH,
                                                             T *dstPtr,
                                                             uint3 dstStridesNCH,
                                                             RpptImagePatchPtr dstImgSize,
                                                             RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void resize_nearest_neighbor_pln3_pkd3_hip_tensor(T *srcPtr,
                                                             uint3 srcStridesNCH,
                                                             T *dstPtr,
                                                             uint2 dstStridesNH,
                                                             RpptImagePatchPtr dstImgSize,
                                                             RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

// -------------------- Set 2 - Bilinear Interpolation --------------------

template <typename T>
__global__ void resize_bilinear_pkd_hip_tensor(T *srcPtr,
                                               uint2 srcStridesNH,
                                               T *dstPtr,
                                               uint2 dstStridesNH,
                                               RpptImagePatchPtr dstImgSize,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void resize_bilinear_pln_hip_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               int channelsDst,
                                               RpptImagePatchPtr dstImgSize,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void resize_bilinear_pkd3_pln3_hip_tensor(T *srcPtr,
                                                     uint2 srcStridesNH,
                                                     T *dstPtr,
                                                     uint3 dstStridesNCH,
                                                     RpptImagePatchPtr dstImgSize,
                                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void resize_bilinear_pln3_pkd3_hip_tensor(T *srcPtr,
                                                     uint3 srcStridesNCH,
                                                     T *dstPtr,
                                                     uint2 dstStridesNH,
                                                     RpptImagePatchPtr dstImgSize,
                                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void resize_generic_pkd_hip_tensor(T *srcPtr,
                                              uint2 srcStridesNH,
                                              T *dstPtr,
                                              uint2 dstStridesNH,
                                              RpptImagePatchPtr dstImgSize,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptInterpolationType interpolationType)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;

    int widthLimit = srcRoi_i4.z * 3;
    int heightLimit = srcRoi_i4.w;
    float wRatio = (float)srcDimsWH.x / (float)dstDimsWH.x;
    float hRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float hScale = 1.0f, wScale = 1.0f, hRadius = 1.0f, wRadius = 1.0f;

    rpp_hip_compute_interpolation_scale_and_radius(interpolationType, &wScale, &wRadius, wRatio);
    rpp_hip_compute_interpolation_scale_and_radius(interpolationType, &hScale, &hRadius, hRatio);
    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceilf(wRadius * 2);
    int hKernelSize = ceilf(hRadius * 2);

    float rowWeight, colWeight, rowCoeff, colCoeff;
    int srcLocationRowFloor, srcLocationColumnFloor;
    resize_roi_generic_srcloc_and_weight_hip_compute(srcRoi_i4.x, id_x, wRatio, widthLimit, &srcLocationColumnFloor, &colWeight, wOffset, 3);
    resize_roi_generic_srcloc_and_weight_hip_compute(srcRoi_i4.y, id_y, hRatio, heightLimit, &srcLocationRowFloor, &rowWeight, hOffset, 1);

    T *srcPtrTemp = srcPtr + (id_z * srcStridesNH.x);
    float3 outPixel_f3 = (float3)0.0f;
    float rowCoeffSum = 0.0f, colCoeffSum = 0.0f;
    for(int j = 0; j < hKernelSize; j++)
    {
        int rowIndex = fminf(fmaxf((int)(srcLocationRowFloor + j), 0), heightLimit);
        T *srcRowPtrsForInterp = srcPtrTemp + rowIndex * srcStridesNH.y;
        rpp_hip_compute_interpolation_coefficient(interpolationType, (rowWeight - hRadius + j) * hScale , &rowCoeff);
        rowCoeffSum += rowCoeff;

        colCoeffSum = 0;
        for(int k = 0; k < wKernelSize; k++)
        {
            int colIndex = fminf(fmaxf((int)(srcLocationColumnFloor + (k * 3)), 0), widthLimit);
            rpp_hip_compute_interpolation_coefficient(interpolationType, (colWeight - wRadius + k) * wScale , &colCoeff);
            colCoeffSum += colCoeff;
            float3 coeff_f3 = (float3)(colCoeff * rowCoeff);
            outPixel_f3 += (make_float3(srcRowPtrsForInterp[colIndex], srcRowPtrsForInterp[colIndex + 1], srcRowPtrsForInterp[colIndex + 2]) * coeff_f3);
        }
    }
    rowCoeffSum = (rowCoeffSum == 0.0f) ? 1.0f : rowCoeffSum;
    colCoeffSum = (colCoeffSum == 0.0f) ? 1.0f : colCoeffSum;
    outPixel_f3 *= (float3)(1 / (rowCoeffSum * colCoeffSum));
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    rpp_hip_pixel_check_and_store(outPixel_f3.x, &dstPtr[dstIdx]);
    rpp_hip_pixel_check_and_store(outPixel_f3.y, &dstPtr[dstIdx + 1]);
    rpp_hip_pixel_check_and_store(outPixel_f3.z, &dstPtr[dstIdx + 2]);
}

template <typename T>
__global__ void resize_generic_pln3_hip_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               RpptImagePatchPtr dstImgSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptInterpolationType interpolationType)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    int widthLimit = srcRoi_i4.z;
    int heightLimit = srcRoi_i4.w;
    float wRatio = (float)srcDimsWH.x / (float)dstDimsWH.x;
    float hRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float hScale = 1.0f, wScale = 1.0f, hRadius = 1.0f, wRadius = 1.0f;

    rpp_hip_compute_interpolation_scale_and_radius(interpolationType, &wScale, &wRadius, wRatio);
    rpp_hip_compute_interpolation_scale_and_radius(interpolationType, &hScale, &hRadius, hRatio);
    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceilf(wRadius * 2);
    int hKernelSize = ceilf(hRadius * 2);

    float rowWeight, colWeight, rowCoeff, colCoeff;
    int srcLocationRowFloor, srcLocationColumnFloor;
    resize_roi_generic_srcloc_and_weight_hip_compute(srcRoi_i4.x, id_x, wRatio, widthLimit, &srcLocationColumnFloor, &colWeight, wOffset, 1);
    resize_roi_generic_srcloc_and_weight_hip_compute(srcRoi_i4.y, id_y, hRatio, heightLimit, &srcLocationRowFloor, &rowWeight, hOffset, 1);

    T *srcPtrTemp[3];
    srcPtrTemp[0] = srcPtr + (id_z * srcStridesNCH.x);
    srcPtrTemp[1] = srcPtrTemp[0] + srcStridesNCH.y;
    srcPtrTemp[2] = srcPtrTemp[1] + srcStridesNCH.y;

    T *srcRowPtrsForInterp[3];
    float3 outPixel_f3 = (float3)0.0f;
    float rowCoeffSum = 0.0f, colCoeffSum = 0.0f;
    for(int j = 0; j < hKernelSize; j++)
    {
        int rowIndex = fminf(fmaxf((int)(srcLocationRowFloor + j), 0), heightLimit);
        srcRowPtrsForInterp[0] = srcPtrTemp[0] + rowIndex * srcStridesNCH.z;
        srcRowPtrsForInterp[1] = srcPtrTemp[1] + rowIndex * srcStridesNCH.z;
        srcRowPtrsForInterp[2] = srcPtrTemp[2] + rowIndex * srcStridesNCH.z;
        rpp_hip_compute_interpolation_coefficient(interpolationType, (rowWeight - hRadius + j) * hScale , &rowCoeff);
        rowCoeffSum += rowCoeff;

        colCoeffSum = 0;
        for(int k = 0; k < wKernelSize; k++)
        {
            int colIndex = fminf(fmaxf((int)(srcLocationColumnFloor + k), 0), widthLimit);
            rpp_hip_compute_interpolation_coefficient(interpolationType, (colWeight - wRadius + k) * wScale , &colCoeff);
            colCoeffSum += colCoeff;
            float3 coeff_f3 = (float3)(colCoeff * rowCoeff);
            outPixel_f3 += (make_float3(srcRowPtrsForInterp[0][colIndex], srcRowPtrsForInterp[1][colIndex], srcRowPtrsForInterp[2][colIndex]) * coeff_f3);
        }
    }
    rowCoeffSum = (rowCoeffSum == 0.0f) ? 1.0f : rowCoeffSum;
    colCoeffSum = (colCoeffSum == 0.0f) ? 1.0f : colCoeffSum;
    outPixel_f3 *= (float3)(1 / (rowCoeffSum * colCoeffSum));
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    rpp_hip_pixel_check_and_store(outPixel_f3.x, &dstPtr[dstIdx]);
    rpp_hip_pixel_check_and_store(outPixel_f3.y, &dstPtr[dstIdx + dstStridesNCH.y]);
    rpp_hip_pixel_check_and_store(outPixel_f3.z, &dstPtr[dstIdx + 2 * dstStridesNCH.y]);
}

template <typename T>
__global__ void resize_generic_pln1_hip_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               RpptImagePatchPtr dstImgSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptInterpolationType interpolationType)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    int widthLimit = srcRoi_i4.z;
    int heightLimit = srcRoi_i4.w;
    float wRatio = (float)srcDimsWH.x / (float)dstDimsWH.x;
    float hRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float hScale = 1.0f, wScale = 1.0f, hRadius = 1.0f, wRadius = 1.0f;

    rpp_hip_compute_interpolation_scale_and_radius(interpolationType, &wScale, &wRadius, wRatio);
    rpp_hip_compute_interpolation_scale_and_radius(interpolationType, &hScale, &hRadius, hRatio);
    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceilf(wRadius * 2);
    int hKernelSize = ceilf(hRadius * 2);

    float rowWeight, colWeight, rowCoeff, colCoeff;
    int srcLocationRowFloor, srcLocationColumnFloor;
    resize_roi_generic_srcloc_and_weight_hip_compute(srcRoi_i4.x, id_x, wRatio, widthLimit, &srcLocationColumnFloor, &colWeight, wOffset, 1);
    resize_roi_generic_srcloc_and_weight_hip_compute(srcRoi_i4.y, id_y, hRatio, heightLimit, &srcLocationRowFloor, &rowWeight, hOffset, 1);

    T *srcPtrTemp = srcPtr + (id_z * srcStridesNCH.x);
    float outPixel = 0.0f;
    float rowCoeffSum = 0.0f, colCoeffSum = 0.0f, invCoeffSum = 0.0f;
    for(int j = 0; j < hKernelSize; j++)
    {
        int rowIndex = fminf(fmaxf((int)(srcLocationRowFloor + j), 0), heightLimit);
        T *srcRowPtrsForInterp = srcPtrTemp + rowIndex * srcStridesNCH.z;
        rpp_hip_compute_interpolation_coefficient(interpolationType, (rowWeight - hRadius + j) * hScale , &rowCoeff);
        rowCoeffSum += rowCoeff;

        colCoeffSum = 0;
        for(int k = 0; k < wKernelSize; k++)
        {
            int colIndex = fminf(fmaxf((int)(srcLocationColumnFloor + k), 0), widthLimit);
            rpp_hip_compute_interpolation_coefficient(interpolationType, (colWeight - wRadius + k) * wScale , &colCoeff);
            colCoeffSum += colCoeff;
            float coeff = colCoeff * rowCoeff;
            outPixel += (float) srcRowPtrsForInterp[colIndex] * coeff;
        }
    }
    rowCoeffSum = (rowCoeffSum == 0.0f) ? 1.0f : rowCoeffSum;
    colCoeffSum = (colCoeffSum == 0.0f) ? 1.0f : colCoeffSum;
    invCoeffSum = 1 / (rowCoeffSum * colCoeffSum);
    outPixel *= invCoeffSum;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    rpp_hip_pixel_check_and_store(outPixel, &dstPtr[dstIdx]);
}

template <typename T>
__global__ void resize_generic_pkd3_pln3_hip_tensor(T *srcPtr,
                                                    uint2 srcStridesNH,
                                                    T *dstPtr,
                                                    uint3 dstStridesNCH,
                                                    RpptImagePatchPtr dstImgSize,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptInterpolationType interpolationType)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    int widthLimit = srcRoi_i4.z * 3;
    int heightLimit = srcRoi_i4.w;
    float wRatio = (float)srcDimsWH.x / (float)dstDimsWH.x;
    float hRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float hScale = 1.0f, wScale = 1.0f, hRadius = 1.0f, wRadius = 1.0f;

    rpp_hip_compute_interpolation_scale_and_radius(interpolationType, &wScale, &wRadius, wRatio);
    rpp_hip_compute_interpolation_scale_and_radius(interpolationType, &hScale, &hRadius, hRatio);
    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceilf(wRadius * 2);
    int hKernelSize = ceilf(hRadius * 2);

    float rowWeight, colWeight, rowCoeff, colCoeff;
    int srcLocationRowFloor, srcLocationColumnFloor;
    resize_roi_generic_srcloc_and_weight_hip_compute(srcRoi_i4.x, id_x, wRatio, widthLimit, &srcLocationColumnFloor, &colWeight, wOffset, 3);
    resize_roi_generic_srcloc_and_weight_hip_compute(srcRoi_i4.y, id_y, hRatio, heightLimit, &srcLocationRowFloor, &rowWeight, hOffset, 1);

    T *srcPtrTemp = srcPtr + (id_z * srcStridesNH.x);
    float3 outPixel_f3 = (float3)0.0f;
    float rowCoeffSum = 0.0f, colCoeffSum = 0.0f;
    for(int j = 0; j < hKernelSize; j++)
    {
        int rowIndex = fminf(fmaxf((int)(srcLocationRowFloor + j), 0), heightLimit);
        T *srcRowPtrsForInterp = srcPtrTemp + rowIndex * srcStridesNH.y;
        rpp_hip_compute_interpolation_coefficient(interpolationType, (rowWeight - hRadius + j) * hScale , &rowCoeff);
        rowCoeffSum += rowCoeff;

        colCoeffSum = 0;
        for(int k = 0; k < wKernelSize; k++)
        {
            int colIndex = fminf(fmaxf((int)(srcLocationColumnFloor + (k * 3)), 0), widthLimit);
            rpp_hip_compute_interpolation_coefficient(interpolationType, (colWeight - wRadius + k) * wScale , &colCoeff);
            colCoeffSum += colCoeff;
            float3 coeff_f3 = (float3)(colCoeff * rowCoeff);
            outPixel_f3 += (make_float3(srcRowPtrsForInterp[colIndex], srcRowPtrsForInterp[colIndex + 1], srcRowPtrsForInterp[colIndex + 2]) * coeff_f3);
        }
    }
    rowCoeffSum = (rowCoeffSum == 0.0f) ? 1.0f : rowCoeffSum;
    colCoeffSum = (colCoeffSum == 0.0f) ? 1.0f : colCoeffSum;
    outPixel_f3 *= 1 / (rowCoeffSum * colCoeffSum);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    rpp_hip_pixel_check_and_store(outPixel_f3.x, &dstPtr[dstIdx]);
    rpp_hip_pixel_check_and_store(outPixel_f3.y, &dstPtr[dstIdx + dstStridesNCH.y]);
    rpp_hip_pixel_check_and_store(outPixel_f3.z, &dstPtr[dstIdx + 2 * dstStridesNCH.y]);
}

template <typename T>
__global__ void resize_generic_pln3_pkd3_hip_tensor(T *srcPtr,
                                                    uint3 srcStridesNCH,
                                                    T *dstPtr,
                                                    uint2 dstStridesNH,
                                                    RpptImagePatchPtr dstImgSize,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptInterpolationType interpolationType)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    int widthLimit = srcRoi_i4.z;
    int heightLimit = srcRoi_i4.w;
    float wRatio = (float)srcDimsWH.x / (float)dstDimsWH.x;
    float hRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float hScale = 1.0f, wScale = 1.0f, hRadius = 1.0f, wRadius = 1.0f;

    rpp_hip_compute_interpolation_scale_and_radius(interpolationType, &wScale, &wRadius, wRatio);
    rpp_hip_compute_interpolation_scale_and_radius(interpolationType, &hScale, &hRadius, hRatio);
    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceilf(wRadius * 2);
    int hKernelSize = ceilf(hRadius * 2);

    float rowWeight, colWeight, rowCoeff, colCoeff;
    int srcLocationRowFloor, srcLocationColumnFloor;
    resize_roi_generic_srcloc_and_weight_hip_compute(srcRoi_i4.x, id_x, wRatio, widthLimit, &srcLocationColumnFloor, &colWeight, wOffset, 1);
    resize_roi_generic_srcloc_and_weight_hip_compute(srcRoi_i4.y, id_y, hRatio, heightLimit, &srcLocationRowFloor, &rowWeight, hOffset, 1);

    T *srcPtrTemp[3];
    srcPtrTemp[0] = srcPtr + (id_z * srcStridesNCH.x);
    srcPtrTemp[1] = srcPtrTemp[0] + srcStridesNCH.y;
    srcPtrTemp[2] = srcPtrTemp[1] + srcStridesNCH.y;

    T *srcRowPtrsForInterp[3];
    float3 outPixel_f3 = (float3)0.0f;
    float rowCoeffSum = 0.0f, colCoeffSum = 0.0f;
    for(int j = 0; j < hKernelSize; j++)
    {
        int rowIndex = fminf(fmaxf((int)(srcLocationRowFloor + j), 0), heightLimit);
        srcRowPtrsForInterp[0] = srcPtrTemp[0] + rowIndex * srcStridesNCH.z;
        srcRowPtrsForInterp[1] = srcPtrTemp[1] + rowIndex * srcStridesNCH.z;
        srcRowPtrsForInterp[2] = srcPtrTemp[2] + rowIndex * srcStridesNCH.z;
        rpp_hip_compute_interpolation_coefficient(interpolationType, (rowWeight - hRadius + j) * hScale , &rowCoeff);
        rowCoeffSum += rowCoeff;

        colCoeffSum = 0;
        for(int k = 0; k < wKernelSize; k++)
        {
            int colIndex = fminf(fmaxf((int)(srcLocationColumnFloor + k), 0), widthLimit);
            rpp_hip_compute_interpolation_coefficient(interpolationType, (colWeight - wRadius + k) * wScale , &colCoeff);
            colCoeffSum += colCoeff;
            float3 coeff_f3 = (float3)(colCoeff * rowCoeff);
            outPixel_f3 += (make_float3(srcRowPtrsForInterp[0][colIndex], srcRowPtrsForInterp[1][colIndex], srcRowPtrsForInterp[2][colIndex]) * coeff_f3);
        }
    }
    rowCoeffSum = (rowCoeffSum == 0.0f) ? 1.0f : rowCoeffSum;
    colCoeffSum = (colCoeffSum == 0.0f) ? 1.0f : colCoeffSum;
    outPixel_f3 *= (float3)(1 / (rowCoeffSum * colCoeffSum));
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    rpp_hip_pixel_check_and_store(outPixel_f3.x, &dstPtr[dstIdx]);
    rpp_hip_pixel_check_and_store(outPixel_f3.y, &dstPtr[dstIdx + 1]);
    rpp_hip_pixel_check_and_store(outPixel_f3.z, &dstPtr[dstIdx + 2]);
}

// -------------------- Set 3 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_resize_tensor(T *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 T *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RpptImagePatchPtr dstImgSize,
                                 RpptInterpolationType interpolationType,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rpp::Handle& handle)
{
    if (roiType == RpptRoiType::XYWH)
        hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);

    if (interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(resize_nearest_neighbor_pkd_hip_tensor,
                            dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                            dstPtr,
                            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                            dstImgSize,
                            roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(resize_nearest_neighbor_pln_hip_tensor,
                            dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                            dstPtr,
                            make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                            dstDescPtr->c,
                            dstImgSize,
                            roiTensorPtrSrc);
        }
        else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
        {
            if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(resize_nearest_neighbor_pkd3_pln3_hip_tensor,
                                dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                dstImgSize,
                                roiTensorPtrSrc);
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                hipLaunchKernelGGL(resize_nearest_neighbor_pln3_pkd3_hip_tensor,
                                dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                dstImgSize,
                                roiTensorPtrSrc);
            }
        }
    }
    else if (interpolationType == RpptInterpolationType::BILINEAR)
    {
        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(resize_bilinear_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               dstImgSize,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(resize_bilinear_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               dstImgSize,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
        {
            if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(resize_bilinear_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   dstImgSize,
                                   roiTensorPtrSrc);
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                hipLaunchKernelGGL(resize_bilinear_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   dstImgSize,
                                   roiTensorPtrSrc);
            }
        }
    }
    else
    {
        int globalThreads_x = dstDescPtr->w;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();

        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(resize_generic_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               dstImgSize,
                               roiTensorPtrSrc,
                               interpolationType);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if (srcDescPtr->c == 3)
            {
                hipLaunchKernelGGL(resize_generic_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   dstImgSize,
                                   roiTensorPtrSrc,
                                   interpolationType);
            }
            else if (srcDescPtr->c == 1)
            {
                hipLaunchKernelGGL(resize_generic_pln1_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   dstImgSize,
                                   roiTensorPtrSrc,
                                   interpolationType);
            }
        }
        else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
        {
            if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(resize_generic_pkd3_pln3_hip_tensor,
                                    dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                    dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                    0,
                                    handle.GetStream(),
                                    srcPtr,
                                    make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                    dstPtr,
                                    make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                    dstImgSize,
                                    roiTensorPtrSrc,
                                    interpolationType);
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                hipLaunchKernelGGL(resize_generic_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   dstImgSize,
                                   roiTensorPtrSrc,
                                   interpolationType);
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_resize_tensor<Rpp8u>(Rpp8u*,
                                                 RpptDescPtr,
                                                 Rpp8u*,
                                                 RpptDescPtr,
                                                 RpptImagePatchPtr,
                                                 RpptInterpolationType,
                                                 RpptROIPtr,
                                                 RpptRoiType,
                                                 rpp::Handle&);

template RppStatus hip_exec_resize_tensor<half>(half*,
                                                RpptDescPtr,
                                                half*,
                                                RpptDescPtr,
                                                RpptImagePatchPtr,
                                                RpptInterpolationType,
                                                RpptROIPtr,
                                                RpptRoiType,
                                                rpp::Handle&);

template RppStatus hip_exec_resize_tensor<Rpp32f>(Rpp32f*,
                                                  RpptDescPtr,
                                                  Rpp32f*,
                                                  RpptDescPtr,
                                                  RpptImagePatchPtr,
                                                  RpptInterpolationType,
                                                  RpptROIPtr,
                                                  RpptRoiType,
                                                  rpp::Handle&);

template RppStatus hip_exec_resize_tensor<Rpp8s>(Rpp8s*,
                                                 RpptDescPtr,
                                                 Rpp8s*,
                                                 RpptDescPtr,
                                                 RpptImagePatchPtr,
                                                 RpptInterpolationType,
                                                 RpptROIPtr,
                                                 RpptRoiType,
                                                 rpp::Handle&);
