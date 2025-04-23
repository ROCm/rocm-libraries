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

// -------------------- Set 0 - crop and patch device helpers --------------------

__device__ void update_data(int startLoc, d_float24 *dst_f24, d_float24 *temp_f24)
{
    for(uint i = startLoc, j = 0; i < 8; i++, j++)
    {
        dst_f24->f8[0].f1[i] = temp_f24->f8[0].f1[j];
        dst_f24->f8[1].f1[i] = temp_f24->f8[1].f1[j];
        dst_f24->f8[2].f1[i] = temp_f24->f8[2].f1[j];
    }
}

__device__ void update_data(int startLoc, d_float8 *dst_f8, d_float8 *temp_f8)
{
    for(uint i = startLoc, j = 0; i < 8; i++, j++)
        dst_f8->f1[i] = temp_f8->f1[j];
}

// -------------------- Set 1 - crop and patch main kernels --------------------
template <typename T>
__global__ void crop_and_patch_pkd_hip_tensor(T *srcPtr1,
                                              T *srcPtr2,
                                              uint2 srcStridesNH,
                                              T *dstPtr,
                                              uint2 dstStridesNH,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptROIPtr cropTensorPtrSrc,
                                              RpptROIPtr patchTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    // check if the co-ordinates is within the patch region
    d_float24 dst_f24;
    bool rowCheck = (id_y >= patchTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y < (patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.roiHeight));
    bool colCheck = (id_x >= patchTensorPtrSrc[id_z].xywhROI.xy.x) && (id_x < (patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth));
    uint patchStart = patchTensorPtrSrc[id_z].xywhROI.xy.x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    if(rowCheck && colCheck)
    {
        uint srcIdx1 = (id_z * srcStridesNH.x) + (id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y + (id_x - patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
        uint patchEnd = patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx1, &dst_f24);
        // to handle the case when loaded data goes beyond the bounds of patch region
        if((id_x + 8) >= patchEnd)
        {
            uint srcIdx2 = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (patchEnd + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
            d_float24 temp_f24;
            rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx2, &temp_f24);
            update_data(patchEnd - id_x, &dst_f24, &temp_f24);
        }
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
    }
    // to handle the case when loaded data goes beyond the input region and enters the patch region
    else if(rowCheck && (id_x + 8) >= patchStart)
    {
        uint srcIdx2 = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx2, &dst_f24);
        uint srcIdx1 = (id_z * srcStridesNH.x) + ((id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + cropTensorPtrSrc[id_z].xywhROI.xy.x * 3;
        d_float24 temp_f24;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx1, &temp_f24);
        update_data(patchStart - id_x, &dst_f24, &temp_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
    }
}

template <typename T>
__global__ void crop_and_patch_pln3_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptROIPtr cropTensorPtrSrc,
                                               RpptROIPtr patchTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    d_float24 dst_f24;
    // check if the co-ordinates is within the patch region
    bool rowCheck = (id_y >= patchTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y < (patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.roiHeight));
    bool colCheck = (id_x >= patchTensorPtrSrc[id_z].xywhROI.xy.x) && (id_x < (patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth));
    uint patchStart = patchTensorPtrSrc[id_z].xywhROI.xy.x;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    if(rowCheck && colCheck)
    {
        uint srcIdx1 = (id_z * srcStridesNCH.x) + (id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z + (id_x - patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.xy.x);
        uint patchEnd = patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth;
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx1, srcStridesNCH.y, &dst_f24);
        // to handle the case when loaded data goes beyond the bounds of patch region
        if((id_x + 8) >= patchEnd)
        {
            uint srcIdx2 = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (patchEnd + roiTensorPtrSrc[id_z].xywhROI.xy.x);
            d_float24 temp_f24;
            rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx2, srcStridesNCH.y, &temp_f24);
            update_data(patchEnd - id_x, &dst_f24, &temp_f24);
        }
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
    }
    // to handle the case when loaded data goes beyond the input region and enters the patch region
    else if(rowCheck && (id_x + 8) >= patchStart)
    {
        uint srcIdx2 = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx2, srcStridesNCH.y, &dst_f24);
        uint srcIdx1 = (id_z * srcStridesNCH.x) + ((id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + cropTensorPtrSrc[id_z].xywhROI.xy.x;
        d_float24 temp_f24;
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx1, srcStridesNCH.y, &temp_f24);
        update_data(patchStart - id_x, &dst_f24, &temp_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
    }
}

template <typename T>
__global__ void crop_and_patch_pln1_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint2 srcStridesNH,
                                               T *dstPtr,
                                               uint2 dstStridesNH,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptROIPtr cropTensorPtrSrc,
                                               RpptROIPtr patchTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    // check if the co-ordinates is within the patch region
    bool rowCheck = (id_y >= patchTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y < (patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.roiHeight));
    bool colCheck = (id_x >= patchTensorPtrSrc[id_z].xywhROI.xy.x) && (id_x < (patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth));
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
    uint patchStart = patchTensorPtrSrc[id_z].xywhROI.xy.x;
    if(rowCheck && colCheck)
    {
        uint srcIdx1 = (id_z * srcStridesNH.x) + (id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y + (id_x - patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.xy.x);
        uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
        uint patchEnd = patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth;

        d_float8 dst_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &dst_f8);
        uint srcIdx2;
        // to handle the case when loaded data goes beyond the bounds of patch region
        if((id_x + 8) >= patchEnd)
        {
            d_float8 temp_f8;
            srcIdx2 = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (patchEnd + roiTensorPtrSrc[id_z].xywhROI.xy.x);
            rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &temp_f8);
            update_data(patchEnd - id_x, &dst_f8, &temp_f8);
        }
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
    // to handle the case when loaded data goes beyond the input region and enters the patch region
    else if(rowCheck && (id_x + 8) >= patchStart)
    {
        d_float8 dst_f8, temp_f8;
        uint srcIdx2 = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &dst_f8);

        uint srcIdx1 = (id_z * srcStridesNH.x) + ((id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + cropTensorPtrSrc[id_z].xywhROI.xy.x;
        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &temp_f8);
        update_data(patchStart - id_x, &dst_f8, &temp_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void crop_and_patch_pkd3_pln3_hip_tensor(T *srcPtr1,
                                                    T *srcPtr2,
                                                    uint2 srcStridesNH,
                                                    T *dstPtr,
                                                    uint3 dstStridesNCH,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptROIPtr cropTensorPtrSrc,
                                                    RpptROIPtr patchTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    // check if the co-ordinates is within the patch region
    d_float24 dst_f24;
    bool rowCheck = (id_y >= patchTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y < (patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.roiHeight));
    bool colCheck = (id_x >= patchTensorPtrSrc[id_z].xywhROI.xy.x) && (id_x < (patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth));
    uint patchStart = patchTensorPtrSrc[id_z].xywhROI.xy.x;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    if(rowCheck && colCheck)
    {
        uint srcIdx1 = (id_z * srcStridesNH.x) + (id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y + (id_x - patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
        uint patchEnd = patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx1, &dst_f24);
        // to handle the case when loaded data goes beyond the bounds of patch region
        if((id_x + 8) >= patchEnd)
        {
            uint srcIdx2 = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (patchEnd + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
            d_float24 temp_f24;
            rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx2, &temp_f24);
            update_data(patchEnd - id_x, &dst_f24, &temp_f24);
        }
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
    }
    // to handle the case when loaded data goes beyond the input region and enters the patch region
    else if(rowCheck && (id_x + 8) >= patchStart)
    {
        uint srcIdx2 = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx2, &dst_f24);
        uint srcIdx1 = (id_z * srcStridesNH.x) + ((id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + cropTensorPtrSrc[id_z].xywhROI.xy.x * 3;
        d_float24 temp_f24;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx1, &temp_f24);
        update_data(patchStart - id_x, &dst_f24, &temp_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
    }
}

template <typename T>
__global__ void crop_and_patch_pln3_pkd3_hip_tensor(T *srcPtr1,
                                                    T *srcPtr2,
                                                    uint3 srcStridesNCH,
                                                    T *dstPtr,
                                                    uint2 dstStridesNH,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptROIPtr cropTensorPtrSrc,
                                                    RpptROIPtr patchTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    d_float24 dst_f24;
    // check if the co-ordinates is within the patch region
    bool rowCheck = (id_y >= patchTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y < (patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.roiHeight));
    bool colCheck = (id_x >= patchTensorPtrSrc[id_z].xywhROI.xy.x) && (id_x < (patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth));
    uint patchStart = patchTensorPtrSrc[id_z].xywhROI.xy.x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    if(rowCheck && colCheck)
    {
        uint srcIdx1 = (id_z * srcStridesNCH.x) + (id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z + (id_x - patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.xy.x);
        uint patchEnd = patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth;
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx1, srcStridesNCH.y, &dst_f24);
        // to handle the case when loaded data goes beyond the bounds of patch region
        if((id_x + 8) >= patchEnd)
        {
            uint srcIdx2 = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (patchEnd + roiTensorPtrSrc[id_z].xywhROI.xy.x);
            d_float24 temp_f24;
            rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx2, srcStridesNCH.y, &temp_f24);
            update_data(patchEnd - id_x, &dst_f24, &temp_f24);
        }
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
    }
    // to handle the case when loaded data goes beyond the input region and enters the patch region
    else if(rowCheck && (id_x + 8) >= patchStart)
    {
        uint srcIdx2 = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx2, srcStridesNCH.y, &dst_f24);
        uint srcIdx1 = (id_z * srcStridesNCH.x) + ((id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + cropTensorPtrSrc[id_z].xywhROI.xy.x;
        d_float24 temp_f24;
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx1, srcStridesNCH.y, &temp_f24);
        update_data(patchStart - id_x, &dst_f24, &temp_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
    }
}

// -------------------- Set 2 - Kernel Executors --------------------
template <typename T>
RppStatus hip_exec_crop_and_patch_tensor(T *srcPtr1,
                                         T *srcPtr2,
                                         RpptDescPtr srcDescPtr,
                                         T *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptROIPtr cropTensorPtr,
                                         RpptROIPtr patchTensorPtr,
                                         RpptRoiType roiType,
                                         rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->w + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipMemcpyAsync(dstPtr, srcPtr2, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(crop_and_patch_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc,
                           cropTensorPtr,
                           patchTensorPtr);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipMemcpyAsync(dstPtr, srcPtr2, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
        {
            hipLaunchKernelGGL(crop_and_patch_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc,
                               cropTensorPtr,
                               patchTensorPtr);
        }
        else
        {
            hipLaunchKernelGGL(crop_and_patch_pln1_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc,
                               cropTensorPtr,
                               patchTensorPtr);
        }
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(convert_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr2,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(crop_and_patch_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc,
                               cropTensorPtr,
                               patchTensorPtr);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(convert_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr2,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(crop_and_patch_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc,
                               cropTensorPtr,
                               patchTensorPtr);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_crop_and_patch_tensor<Rpp8u>(Rpp8u*,
                                                         Rpp8u*,
                                                         RpptDescPtr,
                                                         Rpp8u*,
                                                         RpptDescPtr,
                                                         RpptROIPtr,
                                                         RpptROIPtr,
                                                         RpptROIPtr,
                                                         RpptRoiType,
                                                         rpp::Handle&);

template RppStatus hip_exec_crop_and_patch_tensor<half>(half*,
                                                        half*,
                                                        RpptDescPtr,
                                                        half*,
                                                        RpptDescPtr,
                                                        RpptROIPtr,
                                                        RpptROIPtr,
                                                        RpptROIPtr,
                                                        RpptRoiType,
                                                        rpp::Handle&);

template RppStatus hip_exec_crop_and_patch_tensor<Rpp32f>(Rpp32f*,
                                                          Rpp32f*,
                                                          RpptDescPtr,
                                                          Rpp32f*,
                                                          RpptDescPtr,
                                                          RpptROIPtr,
                                                          RpptROIPtr,
                                                          RpptROIPtr,
                                                          RpptRoiType,
                                                          rpp::Handle&);

template RppStatus hip_exec_crop_and_patch_tensor<Rpp8s>(Rpp8s*,
                                                         Rpp8s*,
                                                         RpptDescPtr,
                                                         Rpp8s*,
                                                         RpptDescPtr,
                                                         RpptROIPtr,
                                                         RpptROIPtr,
                                                         RpptROIPtr,
                                                         RpptRoiType,
                                                         rpp::Handle&);
