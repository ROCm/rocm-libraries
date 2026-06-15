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

template <typename T>
__global__ void concat_generic_hip_tensor(T *srcPtr1,
                                          T *srcPtr2,
                                          uint *srcTensor1Strides,
                                          uint *srcTensor2Strides,
                                          T *dstPtr,
                                          uint *dstStrides,
                                          uint axis,
                                          uint numDims,
                                          Rpp32u *roiTensor,
                                          Rpp32u *srcOffsets,
                                          Rpp32u *dstOffsets)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Get ROI information for both source tensors
    uint *roi1 = roiTensor + id_z * numDims * 4;  // First tensor ROI
    uint *roi2 = roi1 + numDims * 2;              // Second tensor ROI
    uint *begin1 = roi1;
    uint *length1 = &roi1[numDims];
    uint *begin2 = roi2;
    uint *length2 = &roi2[numDims];
    
    // Calculate total elements for this batch
    uint totalElements = 1;
    for (int i = 0; i < numDims; i++)
    {
        if (i == axis)
            totalElements *= (length1[i] + length2[i]);
        else
            totalElements *= length1[i];
    }
    
    if(id_x >= totalElements)
        return;

    // Get batch-specific offsets
    uint srcOffset1 = srcOffsets[id_z * 2];
    uint srcOffset2 = srcOffsets[id_z * 2 + 1];
    uint dstOffset = dstOffsets[id_z];
    
    uint dstIdx = dstOffset;
    uint srcIdx1 = srcOffset1;
    uint srcIdx2 = srcOffset2;
    uint coords[RPPT_MAX_DIMS];

    // Calculate strides for this specific batch
    uint src1Strides[RPPT_MAX_DIMS];
    uint src2Strides[RPPT_MAX_DIMS];
    uint dstStridesLocal[RPPT_MAX_DIMS];
    
    if (numDims == 0)
        return;
    
    for (int i = (int)numDims - 1; i >= 0; i--)
    {
        if (i == (int)numDims - 1)
        {
            src1Strides[i] = 1;
            src2Strides[i] = 1;
            dstStridesLocal[i] = 1;
        }
        else
        {
            src1Strides[i] = src1Strides[i + 1] * length1[i + 1];
            src2Strides[i] = src2Strides[i + 1] * length2[i + 1];
            if (i + 1 == axis)
                dstStridesLocal[i] = dstStridesLocal[i + 1] * (length1[i + 1] + length2[i + 1]);
            else
                dstStridesLocal[i] = dstStridesLocal[i + 1] * length1[i + 1];
        }
    }

    uint temp = id_x;
    for (int i = 0; i < numDims; i++)
    {
        coords[i] = temp / dstStridesLocal[i];
        temp %= dstStridesLocal[i];
        if(i < axis)
        {
            dstIdx += coords[i] * dstStridesLocal[i];
            srcIdx1 += (coords[i] + begin1[i]) * src1Strides[i];
            srcIdx2 += (coords[i] + begin2[i]) * src2Strides[i];
        }
        else if(i == axis)
        {
            if(coords[i] < length1[i])
            {
                dstIdx += coords[i] * dstStridesLocal[i];
                srcIdx1 += (coords[i] + begin1[i]) * src1Strides[i];
            }
            else
            {
                uint shifted_coord = coords[i] - length1[i];
                dstIdx += coords[i] * dstStridesLocal[i];
                srcIdx2 += (shifted_coord + begin2[i]) * src2Strides[i];
            }
        }
        else
        {
            dstIdx += coords[i] * dstStridesLocal[i];
            srcIdx1 += coords[i] * src1Strides[i];
            srcIdx2 += coords[i] * src2Strides[i];
        }
    }

    // Write to output tensor
    if(coords[axis] < length1[axis])
        dstPtr[dstIdx] = srcPtr1[srcIdx1];
    else
        dstPtr[dstIdx] = srcPtr2[srcIdx2];
}

template <typename T>
__global__ void concat_2d_hip_tensor(T *srcPtr1,
                                     T *srcPtr2,
                                     uint *srcTensor1Strides,
                                     uint *srcTensor2Strides,
                                     T *dstPtr,
                                     uint *dstStrides,
                                     uint *srcDims1,
                                     uint *srcDims2)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    uint maxWidth = (srcDims1[1] > srcDims2[1]) ? srcDims1[1] : srcDims2[1];
    if(id_x >= maxWidth || id_y >= srcDims1[0])
        return;

    uint dstIdx = id_y * dstStrides[1] + id_x * dstStrides[0];
    uint srcIdx1 = id_y * srcTensor1Strides[1] + id_x * srcTensor1Strides[0];
    uint srcIdx2 = id_y * srcTensor2Strides[1] + id_x * srcTensor2Strides[0];
    uint dstIdx2 = dstIdx + srcDims1[1] * dstStrides[0];

    d_float8 src_f8;
    // copy src1
    if(id_x < srcDims1[1])
    {
        if((srcDims1[1] - id_x) >= 8)
        {
            rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &src_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &src_f8);
        }
        else
        {
            for(int i = 0; i < (srcDims1[1] - id_x); i++)
                dstPtr[dstIdx + i * dstStrides[0]] = srcPtr1[srcIdx1 + i * srcTensor1Strides[0]];
        }
    }

    // copy src2
    if(id_x < srcDims2[1])
    {
        if((srcDims2[1] - id_x) >= 8)
        {
            rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx2, &src_f8);
        }
        else
        {
            for(int i = 0; i < (srcDims2[1] - id_x); i++)
                dstPtr[dstIdx2 + i * dstStrides[0]] = srcPtr2[srcIdx2 + i * srcTensor2Strides[0]];
        }
    }
}

template <typename T>
__global__ void concat_3d_hip_tensor(T *srcPtr1,
                                     T *srcPtr2,
                                     uint *srcTensor1Strides,
                                     uint *srcTensor2Strides,
                                     T *dstPtr,
                                     uint *dstStrides,
                                     uint *srcDims1,
                                     uint *srcDims2)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint maxLength = (srcDims1[2] > srcDims2[2]) ? srcDims1[2] : srcDims2[2];
    if(id_x >= maxLength || id_y >= srcDims1[1] || id_z >= srcDims1[0])
        return;

    uint dstIdx = id_z * dstStrides[1] + id_y * dstStrides[2] + id_x;
    uint srcIdx1 = id_z * srcTensor1Strides[1] + id_y * srcTensor1Strides[2] + id_x;
    uint srcIdx2 = id_z * srcTensor2Strides[1] + id_y * srcTensor2Strides[2] + id_x;
    uint dstIdx2 = dstIdx + srcDims1[2];

    d_float8 src_f8;
    // copy src1
    if(id_x < srcDims1[2])
    {
        if((srcDims1[2] - id_x) >= 8)
        {
            rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &src_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &src_f8);
        }
        else
        {
            for(int i = 0; i < (srcDims1[2] - id_x); i++)
                dstPtr[dstIdx + i] = srcPtr1[srcIdx1 + i];
        }
    }

    // copy src2
    if(id_x < srcDims2[2])
    {
        if((srcDims2[2] - id_x) >= 8)
        {
            rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx2, &src_f8);
        }
        else
        {
            for(int i = 0; i < (srcDims2[2] - id_x); i++)
                dstPtr[dstIdx2 + i] = srcPtr2[srcIdx2 + i];
        }
    }
}

template <typename T>
RppStatus hip_exec_concat_tensor(T *srcPtr1,
                                 RpptGenericDescPtr srcPtr1GenericDescPtr,
                                 T *srcPtr2,
                                 RpptGenericDescPtr srcPtr2GenericDescPtr,
                                 T *dstPtr,
                                 RpptGenericDescPtr dstGenericDescPtr,
                                 Rpp32u axis,
                                 Rpp32u *roiTensor,
                                 Rpp32u *roiTensor2,
                                 rpp::Handle& handle)
{
    int globalThreads_x = dstGenericDescPtr->strides[0];
    int globalThreads_y = 1;
    int globalThreads_z = dstGenericDescPtr->dims[0];

    int numDims = dstGenericDescPtr->numDims - 1;

    if(numDims == 2)
    {
        // Calculate cumulative offsets for variable-sized tensors
        Rpp32u batchSize = dstGenericDescPtr->dims[0];
        Rpp32u *offsetBuffer = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);

        // Scratch buffer layout for 2D concat:
        // - src1Offsets: batchSize elements
        // - src2Offsets: batchSize elements
        // - dstOffsets: batchSize elements
        // - dimsBuffer: batchSize * (2 * numDims) elements
        // Total: batchSize * (3 + 2 * numDims) elements
        Rpp32u *src1Offsets = offsetBuffer;
        Rpp32u *src2Offsets = offsetBuffer + batchSize;
        Rpp32u *dstOffsets = offsetBuffer + batchSize * 2;
        Rpp32u *dimsBuffer = offsetBuffer + batchSize * 3;
        
        Rpp32u cumOffset1 = 0, cumOffset2 = 0, cumDstOffset = 0;
        for (int i = 0; i < batchSize; i++)
        {
            src1Offsets[i] = cumOffset1;
            src2Offsets[i] = cumOffset2;
            dstOffsets[i] = cumDstOffset;
            
            Rpp32u *roi1 = roiTensor + i * numDims * 2;
            Rpp32u *roi2 = roiTensor2 + i * numDims * 2;
            Rpp32u *length1 = &roi1[numDims];
            Rpp32u *length2 = &roi2[numDims];
            
            Rpp32u size1 = 1, size2 = 1, dstSize = 1;
            for (int j = 0; j < numDims; j++)
            {
                size1 *= length1[j];
                size2 *= length2[j];
                if (j == axis)
                    dstSize *= (length1[j] + length2[j]);
                else
                    dstSize *= length1[j];
            }
            cumOffset1 += size1;
            cumOffset2 += size2;
            cumDstOffset += dstSize;
        }
        
        if(axis == 0)
        {
            srcPtr1GenericDescPtr->strides[1] = srcPtr1GenericDescPtr->strides[0];
            srcPtr1GenericDescPtr->strides[0] = 1;
            srcPtr2GenericDescPtr->strides[1] = srcPtr2GenericDescPtr->strides[0];
            srcPtr2GenericDescPtr->strides[0] = 1;
            dstGenericDescPtr->strides[1] = dstGenericDescPtr->strides[0];
            dstGenericDescPtr->strides[0] = 1;
        }
        else if(axis == 1)
        {
            srcPtr1GenericDescPtr->strides[0] = srcPtr1GenericDescPtr->strides[2];
            srcPtr2GenericDescPtr->strides[0] = srcPtr2GenericDescPtr->strides[2];
            dstGenericDescPtr->strides[0] = dstGenericDescPtr->strides[2];
        }
        
        for(int batchCount = 0; batchCount < batchSize; batchCount++)
        {
            Rpp32u *roi1 = roiTensor + batchCount * numDims * 2;
            Rpp32u *roi2 = roiTensor2 + batchCount * numDims * 2;
            Rpp32u *length1 = &roi1[numDims];
            Rpp32u *length2 = &roi2[numDims];
            Rpp32u *srcDims1 = dimsBuffer + batchCount * numDims * 2;
            Rpp32u *srcDims2 = srcDims1 + numDims;
            if(axis == 0)
            {
                srcDims1[0] = 1;
                srcDims1[1] = length1[0] * length1[1];
                srcDims2[0] = 1;
                srcDims2[1] = length2[0] * length2[1];
            }
            else if(axis == 1)
            {
                srcDims1[0] = length1[0];
                srcDims1[1] = length1[1];
                srcDims2[0] = length2[0];
                srcDims2[1] = length2[1];
            }
            globalThreads_x = (srcDims1[1] > srcDims2[1]) ? srcDims1[1] : srcDims2[1];
            globalThreads_y = srcDims1[0];
            hipLaunchKernelGGL(concat_2d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1 + src1Offsets[batchCount],
                               srcPtr2 + src2Offsets[batchCount],
                               srcPtr1GenericDescPtr->strides,
                               srcPtr2GenericDescPtr->strides,
                               dstPtr + dstOffsets[batchCount],
                               dstGenericDescPtr->strides,
                               srcDims1,
                               srcDims2);
            HIP_CHECK_LAUNCH_RETURN();
        }
    }
    else if(numDims == 3)
    {
        // Calculate cumulative offsets for variable-sized tensors
        Rpp32u batchSize = dstGenericDescPtr->dims[0];
        Rpp32u *offsetBuffer = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);

        // Scratch buffer layout for 3D concat:
        // - src1Offsets: batchSize elements
        // - src2Offsets: batchSize elements
        // - dstOffsets: batchSize elements
        // - dimsBuffer: batchSize * (2 * numDims) elements
        // Total: batchSize * (3 + 2 * numDims) elements
        Rpp32u *src1Offsets = offsetBuffer;
        Rpp32u *src2Offsets = offsetBuffer + batchSize;
        Rpp32u *dstOffsets = offsetBuffer + batchSize * 2;
        Rpp32u *dimsBuffer = offsetBuffer + batchSize * 3;
        
        Rpp32u cumOffset1 = 0, cumOffset2 = 0, cumDstOffset = 0;
        for (int i = 0; i < batchSize; i++)
        {
            src1Offsets[i] = cumOffset1;
            src2Offsets[i] = cumOffset2;
            dstOffsets[i] = cumDstOffset;
            
            Rpp32u *roi1 = roiTensor + i * numDims * 2;
            Rpp32u *roi2 = roiTensor2 + i * numDims * 2;
            Rpp32u *length1 = &roi1[numDims];
            Rpp32u *length2 = &roi2[numDims];
            
            Rpp32u size1 = 1, size2 = 1, dstSize = 1;
            for (int j = 0; j < numDims; j++)
            {
                size1 *= length1[j];
                size2 *= length2[j];
                if (j == axis)
                    dstSize *= (length1[j] + length2[j]);
                else
                    dstSize *= length1[j];
            }
            cumOffset1 += size1;
            cumOffset2 += size2;
            cumDstOffset += dstSize;
        }
        
        if(axis == 0)
        {
            srcPtr1GenericDescPtr->strides[2] = srcPtr1GenericDescPtr->strides[0];
            srcPtr1GenericDescPtr->strides[0] = srcPtr1GenericDescPtr->strides[1] = 1;
            srcPtr2GenericDescPtr->strides[2] = srcPtr2GenericDescPtr->strides[0];
            srcPtr2GenericDescPtr->strides[0] = srcPtr2GenericDescPtr->strides[1] = 1;
            dstGenericDescPtr->strides[2] = dstGenericDescPtr->strides[0];
            dstGenericDescPtr->strides[0] = dstGenericDescPtr->strides[1] = 1;
        }
        else if(axis == 1)
        {
            srcPtr1GenericDescPtr->strides[2] = srcPtr1GenericDescPtr->strides[1];
            srcPtr1GenericDescPtr->strides[0] = srcPtr1GenericDescPtr->strides[1] = 1;
            srcPtr2GenericDescPtr->strides[2] = srcPtr2GenericDescPtr->strides[1];
            srcPtr2GenericDescPtr->strides[0] = srcPtr2GenericDescPtr->strides[1] = 1;
            dstGenericDescPtr->strides[2] = dstGenericDescPtr->strides[1];
            dstGenericDescPtr->strides[0] = dstGenericDescPtr->strides[1] = 1;
        }
        
        for(int batchCount = 0; batchCount < batchSize; batchCount++)
        {
            Rpp32u *roi1 = roiTensor + batchCount * numDims * 2;
            Rpp32u *roi2 = roiTensor2 + batchCount * numDims * 2;
            Rpp32u *length1 = &roi1[numDims];
            Rpp32u *length2 = &roi2[numDims];
            Rpp32u *srcDims1 = dimsBuffer + batchCount * numDims * 2;
            Rpp32u *srcDims2 = srcDims1 + numDims;
            if(axis == 0)
            {
                srcDims1[0] = srcDims1[1] = 1;
                srcDims1[2] = length1[0] * length1[1] * length1[2];
                srcDims2[0] = srcDims2[1] = 1;
                srcDims2[2] = length2[0] * length2[1] * length2[2];
            }
            else if(axis == 1)
            {
                srcDims1[0] = 1;
                srcDims1[1] = length1[0];
                srcDims1[2] = length1[1] * length1[2];
                srcDims2[0] = 1;
                srcDims2[1] = length2[0];
                srcDims2[2] = length2[1] * length2[2];
            }
            else if(axis == 2)
            {
                srcDims1[0] = length1[0];
                srcDims1[1] = length1[1];
                srcDims1[2] = length1[2];
                srcDims2[0] = length2[0];
                srcDims2[1] = length2[1];
                srcDims2[2] = length2[2];
            }
            globalThreads_x = (srcDims1[2] > srcDims2[2]) ? srcDims1[2] : srcDims2[2];
            globalThreads_y = srcDims1[1];
            globalThreads_z = srcDims1[0];
            hipLaunchKernelGGL(concat_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1 + src1Offsets[batchCount],
                               srcPtr2 + src2Offsets[batchCount],
                               srcPtr1GenericDescPtr->strides,
                               srcPtr2GenericDescPtr->strides,
                               dstPtr + dstOffsets[batchCount],
                               dstGenericDescPtr->strides,
                               srcDims1,
                               srcDims2);
            HIP_CHECK_LAUNCH_RETURN();
        }
    }
    else
    {
        // Calculate offsets for each batch
        Rpp32u batchSize = dstGenericDescPtr->dims[0];
        
        // scratchBufferPinned layout (Rpp32u entries):
        // [mergedRoiTensor: 4 * numDims * Batchsize] [srcOffsets: 2 * BatchSize] [dstOffsets: BatchSize]
        Rpp32u *mergedRoiTensor = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
        Rpp32u *srcOffsets = mergedRoiTensor + numDims * 4 * batchSize;
        Rpp32u *dstOffsets = srcOffsets + batchSize * 2;
        
        // Merge the two separate roiTensor arrays
        for(int i = 0; i < batchSize; i++)
        {
            Rpp32u *srcRoi1 = roiTensor + i * numDims * 2;
            Rpp32u *srcRoi2 = roiTensor2 + i * numDims * 2;
            Rpp32u *dstRoi = mergedRoiTensor + i * numDims * 4;
            
            // Copy roi1_begin and roi1_length
            memcpy(dstRoi, srcRoi1, numDims * 2 * sizeof(Rpp32u));
            // Copy roi2_begin and roi2_length
            memcpy(dstRoi + numDims * 2, srcRoi2, numDims * 2 * sizeof(Rpp32u));
        }
        
        // Calculate offsets for each batch element
        Rpp32u srcOffset1 = 0, srcOffset2 = 0, dstOffset = 0;
        Rpp32u maxElements = 0;
        for (int i = 0; i < batchSize; i++)
        {
            srcOffsets[i * 2] = srcOffset1;
            srcOffsets[i * 2 + 1] = srcOffset2;
            dstOffsets[i] = dstOffset;
            
            Rpp32u *batchRoiBase = mergedRoiTensor + i * numDims * 4;
            Rpp32u *roi1 = batchRoiBase;
            Rpp32u *length1 = roi1 + numDims;
            Rpp32u *roi2 = length1 + numDims;
            Rpp32u *length2 = roi2 + numDims;
            
            Rpp32u src1Size = 1, src2Size = 1, dstSize = 1;
            for (int j = 0; j < numDims; j++)
            {
                src1Size *= length1[j];
                src2Size *= length2[j];
                if (j == axis)
                    dstSize *= (length1[j] + length2[j]);
                else
                    dstSize *= length1[j];
            }
            srcOffset1 += src1Size;
            srcOffset2 += src2Size;
            dstOffset += dstSize;
            maxElements = (dstSize > maxElements) ? dstSize : maxElements;
        }
        
        globalThreads_x = maxElements;
        
        hipLaunchKernelGGL(concat_generic_hip_tensor,
                       dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr1,
                       srcPtr2,
                       srcPtr1GenericDescPtr->strides,
                       srcPtr2GenericDescPtr->strides,
                       dstPtr,
                       dstGenericDescPtr->strides,
                       axis,
                       dstGenericDescPtr->numDims - 1,
                       mergedRoiTensor,
                       srcOffsets,
                       dstOffsets);
        HIP_CHECK_LAUNCH_RETURN();
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_concat_tensor<Rpp8u>(Rpp8u*,
                                                 RpptGenericDescPtr,
                                                 Rpp8u*,
                                                 RpptGenericDescPtr,
                                                 Rpp8u*,
                                                 RpptGenericDescPtr,
                                                 Rpp32u,
                                                 Rpp32u*,
                                                 Rpp32u*,
                                                 rpp::Handle&);

template RppStatus hip_exec_concat_tensor<half>(half*,
                                                RpptGenericDescPtr,
                                                half*,
                                                RpptGenericDescPtr,
                                                half*,
                                                RpptGenericDescPtr,
                                                Rpp32u,
                                                Rpp32u*,
                                                Rpp32u*,
                                                rpp::Handle&);

template RppStatus hip_exec_concat_tensor<Rpp32f>(Rpp32f*,
                                                  RpptGenericDescPtr,
                                                  Rpp32f*,
                                                  RpptGenericDescPtr,
                                                  Rpp32f*,
                                                  RpptGenericDescPtr,
                                                  Rpp32u,
                                                  Rpp32u*,
                                                  Rpp32u*,
                                                  rpp::Handle&);

template RppStatus hip_exec_concat_tensor<Rpp8s>(Rpp8s*,
                                                 RpptGenericDescPtr,
                                                 Rpp8s*,
                                                 RpptGenericDescPtr,
                                                 Rpp8s*,
                                                 RpptGenericDescPtr,
                                                 Rpp32u,
                                                 Rpp32u*,
                                                 Rpp32u*,
                                                 rpp::Handle&);
