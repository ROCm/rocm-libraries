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

// Flips a NCDHW tensor along x direction
// Also Flips along y direction if mirrorYZ.x is set to 1
// Also Flips along z direction if mirrorYZ.y is set to 1
template <typename T>
__global__ void flip_xyz_ncdhw_hip_tensor(T *srcPtr,
                                          uint3 srcStridesCDH,
                                          T *dstPtr,
                                          uint3 dstStridesCDH,
                                          int channels,
                                          uint2 mirrorYZ,
                                          RpptROI3DPtr roiGenericPtrSrc,
                                          int batchIndex)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= roiGenericPtrSrc->xyzwhdROI.roiDepth) || (id_y >= roiGenericPtrSrc->xyzwhdROI.roiHeight) || (id_x >= roiGenericPtrSrc->xyzwhdROI.roiWidth))
    {
        return;
    }

    int dstIdx = (id_z * dstStridesCDH.y) + (id_y * dstStridesCDH.z) + id_x;
    int zFactor = ((mirrorYZ.y) ? (roiGenericPtrSrc->xyzwhdROI.xyz.z + roiGenericPtrSrc->xyzwhdROI.roiDepth - 1 - id_z) : (roiGenericPtrSrc->xyzwhdROI.xyz.z + id_z)) * srcStridesCDH.y;
    int yFactor = ((mirrorYZ.x) ? (roiGenericPtrSrc->xyzwhdROI.xyz.y + roiGenericPtrSrc->xyzwhdROI.roiHeight - 1 - id_y) : (roiGenericPtrSrc->xyzwhdROI.xyz.y + id_y)) * srcStridesCDH.z;
    int xFactor = (roiGenericPtrSrc->xyzwhdROI.xyz.x + roiGenericPtrSrc->xyzwhdROI.roiWidth - id_x - 8);

    // To handle the case when trying to load from invalid memory location when width is not a multiple of 8
    if((!batchIndex) && (id_x + 8 > roiGenericPtrSrc->xyzwhdROI.roiWidth))
    {
        bool yCheck = ((mirrorYZ.x && id_y == roiGenericPtrSrc->xyzwhdROI.roiHeight - 1) || (!mirrorYZ.x && id_y == 0));
        bool zCheck = ((mirrorYZ.y && id_z == roiGenericPtrSrc->xyzwhdROI.roiDepth - 1) || (!mirrorYZ.y && id_z == 0));
        if(yCheck && zCheck)
        {
            xFactor = roiGenericPtrSrc->xyzwhdROI.xyz.x;
            dstIdx -= (id_x + 8 - roiGenericPtrSrc->xyzwhdROI.roiWidth);
        }
    }

    int srcIdx = zFactor + yFactor + xFactor;
    d_float8 pix_f8;
    for(int c = 0; c < channels; c++)
    {
        rpp_hip_load8_and_unpack_to_float8_mirror(srcPtr + srcIdx, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        srcIdx += srcStridesCDH.x;
        dstIdx += dstStridesCDH.x;
    }
}

// Flips a NCDHW tensor along y direction if mirrorYZ.x is set to 1
// Also Flips along z direction if mirrorYZ.y is set to 1
template <typename T>
__global__ void flip_yz_ncdhw_hip_tensor(T *srcPtr,
                                         uint3 srcStridesCDH,
                                         T *dstPtr,
                                         uint3 dstStridesCDH,
                                         int channels,
                                         uint2 mirrorYZ,
                                         RpptROI3DPtr roiGenericPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= roiGenericPtrSrc->xyzwhdROI.roiDepth) || (id_y >= roiGenericPtrSrc->xyzwhdROI.roiHeight) || (id_x >= roiGenericPtrSrc->xyzwhdROI.roiWidth))
    {
        return;
    }

    int dstIdx = (id_z * dstStridesCDH.y) + (id_y * dstStridesCDH.z) + id_x;
    int zFactor = ((mirrorYZ.y) ? (roiGenericPtrSrc->xyzwhdROI.xyz.z + roiGenericPtrSrc->xyzwhdROI.roiDepth - 1 - id_z) : (roiGenericPtrSrc->xyzwhdROI.xyz.z + id_z)) * srcStridesCDH.y;
    int yFactor = ((mirrorYZ.x) ? (roiGenericPtrSrc->xyzwhdROI.xyz.y + roiGenericPtrSrc->xyzwhdROI.roiHeight - 1 - id_y) : (roiGenericPtrSrc->xyzwhdROI.xyz.y + id_y)) * srcStridesCDH.z;
    int xFactor = (id_x + roiGenericPtrSrc->xyzwhdROI.xyz.x);
    int srcIdx = zFactor + yFactor + xFactor;
    d_float8 pix_f8;
    for(int c = 0; c < channels; c++)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        srcIdx += srcStridesCDH.x;
        dstIdx += dstStridesCDH.x;
    }
}

// Flips a NDHWC tensor along x direction
// Also Flips along y direction if mirrorYZ.x is set to 1
// Also Flips along z direction if mirrorYZ.y is set to 1
template <typename T>
__global__ void flip_xyz_ndhwc_hip_tensor(T *srcPtr,
                                          uint2 srcStridesDH,
                                          T *dstPtr,
                                          uint2 dstStridesDH,
                                          uint2 mirrorYZ,
                                          RpptROI3DPtr roiGenericPtrSrc,
                                          int batchIndex)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // WC - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= roiGenericPtrSrc->xyzwhdROI.roiDepth) || (id_y >= roiGenericPtrSrc->xyzwhdROI.roiHeight) || (id_x >= roiGenericPtrSrc->xyzwhdROI.roiWidth))
    {
        return;
    }

    int dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x * 3;
    int zFactor = ((mirrorYZ.y) ? (roiGenericPtrSrc->xyzwhdROI.xyz.z + roiGenericPtrSrc->xyzwhdROI.roiDepth - 1 - id_z) : (roiGenericPtrSrc->xyzwhdROI.xyz.z + id_z)) * srcStridesDH.x;
    int yFactor = ((mirrorYZ.x) ? (roiGenericPtrSrc->xyzwhdROI.xyz.y + roiGenericPtrSrc->xyzwhdROI.roiHeight - 1 - id_y) : (roiGenericPtrSrc->xyzwhdROI.xyz.y + id_y)) * srcStridesDH.y;
    int xFactor =  (roiGenericPtrSrc->xyzwhdROI.xyz.x + roiGenericPtrSrc->xyzwhdROI.roiWidth - id_x - 8) * 3;

    // To handle the case when trying to load from invalid memory location when width is not a multiple of 8
    if((!batchIndex) && (id_x + 8 > roiGenericPtrSrc->xyzwhdROI.roiWidth))
    {
        bool yCheck = ((mirrorYZ.x && id_y == roiGenericPtrSrc->xyzwhdROI.roiHeight - 1) || (!mirrorYZ.x && id_y == 0));
        bool zCheck = ((mirrorYZ.y && id_z == roiGenericPtrSrc->xyzwhdROI.roiDepth - 1) || (!mirrorYZ.y && id_z == 0));
        if(yCheck && zCheck)
        {
            xFactor = roiGenericPtrSrc->xyzwhdROI.xyz.x * 3;
            dstIdx -= (id_x + 8 - roiGenericPtrSrc->xyzwhdROI.roiWidth) * 3;
        }
    }
    int srcIdx = zFactor + yFactor + xFactor;
    d_float24 pix_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

// Flips a NDHWC tensor along y direction if mirrorYZ.x is set to 1
// Also Flips along z direction if mirrorYZ.y is set to 1
template <typename T>
__global__ void flip_yz_ndhwc_hip_tensor(T *srcPtr,
                                         uint2 srcStridesDH,
                                         T *dstPtr,
                                         uint2 dstStridesDH,
                                         uint2 mirrorYZ,
                                         RpptROI3DPtr roiGenericPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // WC - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= roiGenericPtrSrc->xyzwhdROI.roiDepth) || (id_y >= roiGenericPtrSrc->xyzwhdROI.roiHeight) || (id_x >= roiGenericPtrSrc->xyzwhdROI.roiWidth))
    {
        return;
    }

    int dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x * 3;
    int zFactor = ((mirrorYZ.y) ? (roiGenericPtrSrc->xyzwhdROI.xyz.z + roiGenericPtrSrc->xyzwhdROI.roiDepth - 1 - id_z) : (roiGenericPtrSrc->xyzwhdROI.xyz.z + id_z))  * srcStridesDH.x;
    int yFactor = ((mirrorYZ.x) ? (roiGenericPtrSrc->xyzwhdROI.xyz.y + roiGenericPtrSrc->xyzwhdROI.roiHeight - 1 - id_y) : (roiGenericPtrSrc->xyzwhdROI.xyz.y + id_y)) * srcStridesDH.y;
    int xFactor =  (id_x + roiGenericPtrSrc->xyzwhdROI.xyz.x) * 3;
    int srcIdx = zFactor + yFactor + xFactor;
    d_float24 pix_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_flip_voxel_tensor(T *srcPtr,
                                     RpptGenericDescPtr srcGenericDescPtr,
                                     T *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     RpptROI3DPtr roiGenericPtrSrc,
                                     Rpp32u *horizontalTensor,
                                     Rpp32u *verticalTensor,
                                     Rpp32u *depthTensor,
                                     RpptRoi3DType roiType,
                                     rpp::Handle& handle)
{
    if (dstGenericDescPtr->layout == RpptLayout::NCDHW)
    {
        int globalThreads_x = (dstGenericDescPtr->strides[3] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[3];               // H - height (y direction)
        int globalThreads_z = dstGenericDescPtr->dims[2];               // D - depth (z direction)

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            if(horizontalTensor[batchCount] == 1)
            {
                hipLaunchKernelGGL(flip_xyz_ncdhw_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                                   make_uint3(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2], srcGenericDescPtr->strides[3]),
                                   dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                   make_uint3(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2], dstGenericDescPtr->strides[3]),
                                   dstGenericDescPtr->dims[1],
                                   make_uint2(verticalTensor[batchCount], depthTensor[batchCount]),
                                   &roiGenericPtrSrc[batchCount],
                                   batchCount);
            }
            else
            {
                hipLaunchKernelGGL(flip_yz_ncdhw_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                                   make_uint3(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2], srcGenericDescPtr->strides[3]),
                                   dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                   make_uint3(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2], dstGenericDescPtr->strides[3]),
                                   dstGenericDescPtr->dims[1],
                                   make_uint2(verticalTensor[batchCount], depthTensor[batchCount]),
                                   &roiGenericPtrSrc[batchCount]);
            }
        }
    }
    else if (dstGenericDescPtr->layout == RpptLayout::NDHWC)
    {
        int globalThreads_x = (dstGenericDescPtr->strides[2] / 3 + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[2];                   // H - height (y direction)
        int globalThreads_z = dstGenericDescPtr->dims[1];                   // D - depth (z direction)
        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            if(horizontalTensor[batchCount] == 1)
            {
                hipLaunchKernelGGL(flip_xyz_ndhwc_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                                   make_uint2(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                                   dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                   make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                                   make_uint2(verticalTensor[batchCount], depthTensor[batchCount]),
                                   &roiGenericPtrSrc[batchCount],
                                   batchCount);
            }
            else
            {
                hipLaunchKernelGGL(flip_yz_ndhwc_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                                   make_uint2(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                                   dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                   make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                                   make_uint2(verticalTensor[batchCount], depthTensor[batchCount]),
                                   &roiGenericPtrSrc[batchCount]);
            }
        }
    }

    return RPP_SUCCESS;
}
template RppStatus hip_exec_flip_voxel_tensor<Rpp8u>(Rpp8u*,
                                                     RpptGenericDescPtr,
                                                     Rpp8u*,
                                                     RpptGenericDescPtr,
                                                     RpptROI3DPtr,
                                                     Rpp32u*,
                                                     Rpp32u*,
                                                     Rpp32u*,
                                                     RpptRoi3DType,
                                                     rpp::Handle&);
template RppStatus hip_exec_flip_voxel_tensor<half>(half*,
                                                    RpptGenericDescPtr,
                                                    half*,
                                                    RpptGenericDescPtr,
                                                    RpptROI3DPtr,
                                                    Rpp32u*,
                                                    Rpp32u*,
                                                    Rpp32u*,
                                                    RpptRoi3DType,
                                                    rpp::Handle&);
template RppStatus hip_exec_flip_voxel_tensor<Rpp32f>(Rpp32f*,
                                                      RpptGenericDescPtr,
                                                      Rpp32f*,
                                                      RpptGenericDescPtr,
                                                      RpptROI3DPtr,
                                                      Rpp32u*,
                                                      Rpp32u*,
                                                      Rpp32u*,
                                                      RpptRoi3DType,
                                                      rpp::Handle&);
template RppStatus hip_exec_flip_voxel_tensor<Rpp8s>(Rpp8s*,
                                                     RpptGenericDescPtr,
                                                     Rpp8s*,
                                                     RpptGenericDescPtr,
                                                     RpptROI3DPtr,
                                                     Rpp32u*,
                                                     Rpp32u*,
                                                     Rpp32u*,
                                                     RpptRoi3DType,
                                                     rpp::Handle&);
