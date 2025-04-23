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

__device__ float4 rpp_hip_load4(float *table, uint4 &tableValLoc)
{
    return make_float4(table[tableValLoc.x], table[tableValLoc.y], table[tableValLoc.z], table[tableValLoc.w]);
}

__device__ void remap_srclocs_hip_compute(int4 *srcRoiPtr_i4, float *rowRemapTable, float *colRemapTable, int id_x, int id_y, int id_z, d_float16 *locSrc_f16)
{
    d_uint8 increment_ui8, locSrc_ui8;

    increment_ui8.ui4[0] = make_uint4(0, 1, 2, 3);                                // 8 element vectorized kernel needs 8 increments - creating uint4 for increments 0, 1, 2, 3
    increment_ui8.ui4[1] = make_uint4(4, 5, 6, 7);                                // 8 element vectorized kernel needs 8 increments - creating uint4 for increments 4, 5, 6, 7
    uint4 locSrc_ui4 = (uint4)(id_x);                                             // getting current id_x into a uint4
    locSrc_ui8.ui4[0] = locSrc_ui4 + increment_ui8.ui4[0];                        // computing vectorized locs (id_x + 0, id_x + 1, id_x + 2, id_x + 3)
    locSrc_ui8.ui4[1] = locSrc_ui4 + increment_ui8.ui4[1];                        // computing vectorized locs (id_x + 4, id_x + 5, id_x + 6, id_x + 7)

    locSrc_f16->f8[0].f4[0] = rpp_hip_load4(colRemapTable, locSrc_ui8.ui4[0]);    // writes 4 src location col values
    locSrc_f16->f8[0].f4[1] = rpp_hip_load4(colRemapTable, locSrc_ui8.ui4[1]);    // writes 4 src location col values
    locSrc_f16->f8[1].f4[0] = rpp_hip_load4(rowRemapTable, locSrc_ui8.ui4[0]);    // writes 4 src location row values
    locSrc_f16->f8[1].f4[1] = rpp_hip_load4(rowRemapTable, locSrc_ui8.ui4[1]);    // writes 4 src location row values
}

// -------------------- Set 2 - Nearest Neighbor Interpolation --------------------

template <typename T>
__global__ void remap_nearest_neighbor_pkd_hip_tensor(T *srcPtr,
                                                      uint2 srcStridesNH,
                                                      T *dstPtr,
                                                      uint2 dstStridesNH,
                                                      uint2 dstDimsWH,
                                                      float *rowRemapTable,
                                                      float *colRemapTable,
                                                      uint2 remapTableStridesNH,
                                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
        return;

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void remap_nearest_neighbor_pln_hip_tensor(T *srcPtr,
                                                      uint3 srcStridesNCH,
                                                      T *dstPtr,
                                                      uint3 dstStridesNCH,
                                                      uint2 dstDimsWH,
                                                      int channelsDst,
                                                      float *rowRemapTable,
                                                      float *colRemapTable,
                                                      uint2 remapTableStridesNH,
                                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

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
__global__ void remap_nearest_neighbor_pkd3_pln3_hip_tensor(T *srcPtr,
                                                            uint2 srcStridesNH,
                                                            T *dstPtr,
                                                            uint3 dstStridesNCH,
                                                            uint2 dstDimsWH,
                                                            int channelsDst,
                                                            float *rowRemapTable,
                                                            float *colRemapTable,
                                                            uint2 remapTableStridesNH,
                                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
        return;

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void remap_nearest_neighbor_pln3_pkd3_hip_tensor(T *srcPtr,
                                                            uint3 srcStridesNCH,
                                                            T *dstPtr,
                                                            uint2 dstStridesNH,
                                                            uint2 dstDimsWH,
                                                            float *rowRemapTable,
                                                            float *colRemapTable,
                                                            uint2 remapTableStridesNH,
                                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

// -------------------- Set 2 - Bilinear Interpolation --------------------

template <typename T>
__global__ void remap_bilinear_pkd_hip_tensor(T *srcPtr,
                                              uint2 srcStridesNH,
                                              T *dstPtr,
                                              uint2 dstStridesNH,
                                              uint2 dstDimsWH,
                                              float *rowRemapTable,
                                              float *colRemapTable,
                                              uint2 remapTableStridesNH,
                                              RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
        return;

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void remap_bilinear_pln_hip_tensor(T *srcPtr,
                                              uint3 srcStridesNCH,
                                              T *dstPtr,
                                              uint3 dstStridesNCH,
                                              uint2 dstDimsWH,
                                              int channelsDst,
                                              float *rowRemapTable,
                                              float *colRemapTable,
                                              uint2 remapTableStridesNH,
                                              RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void remap_bilinear_pkd3_pln3_hip_tensor(T *srcPtr,
                                                    uint2 srcStridesNH,
                                                    T *dstPtr,
                                                    uint3 dstStridesNCH,
                                                    uint2 dstDimsWH,
                                                    int channelsDst,
                                                    float *rowRemapTable,
                                                    float *colRemapTable,
                                                    uint2 remapTableStridesNH,
                                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
        return;

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void remap_bilinear_pln3_pkd3_hip_tensor(T *srcPtr,
                                                    uint3 srcStridesNCH,
                                                    T *dstPtr,
                                                    uint2 dstStridesNH,
                                                    uint2 dstDimsWH,
                                                    float *rowRemapTable,
                                                    float *colRemapTable,
                                                    uint2 remapTableStridesNH,
                                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    float *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;
    float *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y;

    d_float16 locSrc_f16;
    remap_srclocs_hip_compute(&srcRoi_i4, rowRemapTableTemp, colRemapTableTemp, id_x, id_y, id_z, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

// -------------------- Set 3 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_remap_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *rowRemapTable,
                                Rpp32f *colRemapTable,
                                RpptDescPtr remapTableDescPtr,
                                RpptInterpolationType interpolationType,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rpp::Handle& handle)
{
    if (roiType == RpptRoiType::XYWH)
        hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if (interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(remap_nearest_neighbor_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(remap_nearest_neighbor_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               dstDescPtr->c,
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(remap_nearest_neighbor_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               dstDescPtr->c,
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(remap_nearest_neighbor_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
    }
    else if (interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(remap_bilinear_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(remap_bilinear_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               dstDescPtr->c,
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(remap_bilinear_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               dstDescPtr->c,
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(remap_bilinear_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h),
                               rowRemapTable,
                               colRemapTable,
                               make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
    }
    return RPP_SUCCESS;
}

template RppStatus hip_exec_remap_tensor<Rpp8u>(Rpp8u*,
                                                RpptDescPtr,
                                                Rpp8u*,
                                                RpptDescPtr,
                                                Rpp32f*,
                                                Rpp32f*,
                                                RpptDescPtr,
                                                RpptInterpolationType,
                                                RpptROIPtr,
                                                RpptRoiType,
                                                rpp::Handle&);

template RppStatus hip_exec_remap_tensor<half>(half*,
                                               RpptDescPtr,
                                               half*,
                                               RpptDescPtr,
                                               Rpp32f*,
                                               Rpp32f*,
                                               RpptDescPtr,
                                               RpptInterpolationType,
                                               RpptROIPtr,
                                               RpptRoiType,
                                               rpp::Handle&);

template RppStatus hip_exec_remap_tensor<Rpp8s>(Rpp8s*,
                                                RpptDescPtr,
                                                Rpp8s*,
                                                RpptDescPtr,
                                                Rpp32f*,
                                                Rpp32f*,
                                                RpptDescPtr,
                                                RpptInterpolationType,
                                                RpptROIPtr,
                                                RpptRoiType,
                                                rpp::Handle&);

template RppStatus hip_exec_remap_tensor<Rpp32f>(Rpp32f*,
                                                 RpptDescPtr,
                                                 Rpp32f*,
                                                 RpptDescPtr,
                                                 Rpp32f*,
                                                 Rpp32f*,
                                                 RpptDescPtr,
                                                 RpptInterpolationType,
                                                 RpptROIPtr,
                                                 RpptRoiType,
                                                 rpp::Handle&);
