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
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR O
THER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "host_tensor_executors.hpp"

inline void updateStridesAndDims(Rpp32u tensorDims, Rpp32u axisMask,
                                 Rpp32u* src1TensorStrides, Rpp32u* src2TensorStrides, Rpp32u* dstTensorStrides,
                                 Rpp32u* src1ReductionDims, Rpp32u* srcTensor1Strides, Rpp32u* src2ReductionDims, Rpp32u* srcTensor2Strides, Rpp32u* dstStride,
                                 Rpp32u* length1, Rpp32u* length2)
{
    if (tensorDims == 2)
    {
        if (axisMask == 0)
        {
            src1ReductionDims[0] = 1;
            src1ReductionDims[1] = length1[0] * length1[1];
            srcTensor1Strides[0] = 1;
            srcTensor1Strides[1] = src1TensorStrides[1];
            src2ReductionDims[0] = 1;
            src2ReductionDims[1] = length2[0] * length2[1];
            srcTensor2Strides[0] = 1;
            srcTensor2Strides[1] = src2TensorStrides[1];
            dstStride[0] = 1;
            dstStride[1] = dstTensorStrides[1];
        }
        else if (axisMask == 1)
        {
            src1ReductionDims[0] = length1[0];
            src1ReductionDims[1] = length1[1];
            srcTensor1Strides[0] = src1TensorStrides[1];
            srcTensor1Strides[1] = src1TensorStrides[2];
            src2ReductionDims[0] = length2[0];
            src2ReductionDims[1] = length2[1];
            srcTensor2Strides[0] = src2TensorStrides[1];
            srcTensor2Strides[1] = src2TensorStrides[2];
            dstStride[0] = dstTensorStrides[1];
            dstStride[1] = dstTensorStrides[2];
        }
    }
    else if (tensorDims == 3)
    {
        if (axisMask == 0)
        {
            src1ReductionDims[0] = 1;
            src1ReductionDims[1] = 1;
            src1ReductionDims[2] = length1[0] * length1[1] * length1[2];
            srcTensor1Strides[0] = 1;
            srcTensor1Strides[1] = 1;
            srcTensor1Strides[2] = src1TensorStrides[1];
            src2ReductionDims[0] = 1;
            src2ReductionDims[1] = 1;
            src2ReductionDims[2] = length2[0] * length2[1] * length2[2];
            srcTensor2Strides[0] = 1;
            srcTensor2Strides[1] = 1;
            srcTensor2Strides[2] = src2TensorStrides[1];
            dstStride[0] = 1;
            dstStride[1] = 1;
            dstStride[2] = dstTensorStrides[1];
        }
        else if (axisMask == 1)
        {
            src1ReductionDims[0] = 1;
            src1ReductionDims[1] = length1[0];
            src1ReductionDims[2] = length1[1] * length1[2];
            srcTensor1Strides[0] = 1;
            srcTensor1Strides[1] = src1TensorStrides[1];
            srcTensor1Strides[2] = src1TensorStrides[2];
            src2ReductionDims[0] = 1;
            src2ReductionDims[1] = length2[0];
            src2ReductionDims[2] = length2[1] * length2[2];
            srcTensor2Strides[0] = 1;
            srcTensor2Strides[1] = src2TensorStrides[1];
            srcTensor2Strides[2] = src2TensorStrides[2];
            dstStride[0] = 1;
            dstStride[1] = dstTensorStrides[1];
            dstStride[2] = dstTensorStrides[2];
        }
        else if (axisMask == 2)
        {
            src1ReductionDims[0] = length1[0];
            src1ReductionDims[1] = length1[1];
            src1ReductionDims[2] = length1[2];
            srcTensor1Strides[0] = src1TensorStrides[1];
            srcTensor1Strides[1] = src1TensorStrides[2];
            srcTensor1Strides[2] = src1TensorStrides[3];
            src2ReductionDims[0] = length2[0];
            src2ReductionDims[1] = length2[1];
            src2ReductionDims[2] = length2[2];
            srcTensor2Strides[0] = src2TensorStrides[1];
            srcTensor2Strides[1] = src2TensorStrides[2];
            srcTensor2Strides[2] = src2TensorStrides[3];
            dstStride[0] = dstTensorStrides[1];
            dstStride[1] = dstTensorStrides[2];
            dstStride[2] = dstTensorStrides[3];
        }
    }
}

// Computes concatenation for 2D tensors (Supports Rpp32f and Rpp8u)
template <typename T, typename SIMD_LOAD, typename SIMD_STORE>
void concat_2D_tensor(T *srcPtr1, T *srcPtr2, SIMD_LOAD simd_load, SIMD_STORE simd_store, RpptGenericDescPtr srcDescPtr, RpptGenericDescPtr srcDescPtr1, T *dstPtr, RpptGenericDescPtr dstDescPtr, Rpp32u *dims1, Rpp32u *strides1, Rpp32u *dims2, Rpp32u *strides2, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    Rpp32u bufferLength = (dims1[1] < dims2[1]) ? dims1[1] : dims2[1];
    Rpp32u alignedLength = (bufferLength / 8) * 8;
    Rpp32u dstRowStride = dims1[1] + dims2[1]; // Calculate the destination row stride based on concatenated size
    for(Rpp32u i = 0; i < dims1[0]; i++)
    {
        T *srcPtrTemp1 = srcPtr1 + i * strides1[0];
        T *srcPtrTemp2 = srcPtr2 + i * strides2[0];
        T *dstPtrTemp = dstPtr + (i * dstRowStride);

        Rpp32u vectorLoopCount = 0;
        __m256 pDst;
        for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
        {
            simd_load(srcPtrTemp1, &pDst);
            simd_store(dstPtrTemp, &pDst);
            simd_load(srcPtrTemp2, &pDst);
            simd_store(dstPtrTemp + dims1[1], &pDst);
            srcPtrTemp1 += vectorIncrement;
            srcPtrTemp2 += vectorIncrement;
            dstPtrTemp += vectorIncrement;
        }
        dstPtrTemp = dstPtr + (i * dstRowStride);
        for(int j = vectorLoopCount; j < dims1[1]; j++)
            *(dstPtrTemp + j) = *srcPtrTemp1++;
        for(int j = vectorLoopCount; j < dims2[1]; j++)
            *(dstPtrTemp + dims1[1] + j) = *srcPtrTemp2++;
    }
}

// Computes concatenation for 3D tensors (Supports Rpp32f and Rpp8u)
template <typename T, typename SIMD_LOAD, typename SIMD_STORE>
void concat_3D_tensor(T *srcPtr1, T *srcPtr2, SIMD_LOAD simd_load, SIMD_STORE simd_store, RpptGenericDescPtr srcPtr1GenericDescPtr, RpptGenericDescPtr srcPtr2GenericDescPtr, T *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *dims1, Rpp32u *strides1, Rpp32u *dims2, Rpp32u *strides2, Rpp32u *dstStrides, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    Rpp32u bufferLength = (dims1[2] < dims2[2]) ? dims1[2] : dims2[2];
    Rpp32u alignedLength = (bufferLength / 8) * 8;
    for(Rpp32u i = 0; i < dims1[0]; i++)
    {
        T *srcPtrRow1 = srcPtr1;
        T *srcPtrRow2 = srcPtr2;
        T *dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < dims1[1]; j++)
        {
            T *srcPtrRowTemp1 = srcPtrRow1;
            T *srcPtrRowTemp2 = srcPtrRow2;
            T *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst;
            for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
            {
                simd_load(srcPtrRowTemp1, &pDst);
                simd_store(dstPtrRowTemp, &pDst);
                simd_load(srcPtrRowTemp2, &pDst);
                simd_store(dstPtrRowTemp + dims1[2], &pDst);
                srcPtrRowTemp1 += vectorIncrement;
                srcPtrRowTemp2 += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(Rpp32u k = vectorLoopCount; k < dims1[2]; k++)
                *(dstPtrRowTemp + k - vectorLoopCount) = *srcPtrRowTemp1++;
            for(Rpp32u k = vectorLoopCount; k < dims2[2]; k++)
                *(dstPtrRowTemp + dims1[2] + k - vectorLoopCount) = *srcPtrRowTemp2++;
            srcPtrRow1 += strides1[1];
            srcPtrRow2 += strides2[1];
            dstPtrRow += dstStrides[1];
        }
        srcPtr1 += strides1[0];
        srcPtr2 += strides2[0];
        dstPtr += dstStrides[0];
    }
}

// Computes concat for 3D variants
void concat_3D_axismask0_tensor(Rpp8u *srcPtr1, Rpp8u *srcPtr2, RpptGenericDescPtr srcPtr1GenericDescPtr, RpptGenericDescPtr srcPtr2GenericDescPtr, Rpp8u *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *dims1, Rpp32u *strides1, Rpp32u *dims2, Rpp32u *strides2, Rpp32u *dstStrides, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    Rpp32u dstHeightStride = dims1[1] + dims2[1]; // Concatenated height size
    for(Rpp32u i = 0; i < dims1[0]; i++)
    {
        Rpp32u bufferLength1 = dims1[2];
        Rpp32u alignedLength1 = (bufferLength1 / 8) * 8;
        Rpp32u bufferLength2 = dims2[2];
        Rpp32u alignedLength2 = (bufferLength2 / 8) * 8;
        Rpp8u *srcPtrRow = srcPtr1;
        Rpp8u *dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < dims1[1]; j++)
        {
            Rpp8u *srcPtrRowTemp = srcPtrRow;
            Rpp8u *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst;
            for(; vectorLoopCount < alignedLength1; vectorLoopCount += vectorIncrement)
            {
                rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp, &pDst);
                rpp_simd_store(rpp_store8_f32_to_u8_avx, dstPtrRowTemp, &pDst);
                srcPtrRowTemp += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(; vectorLoopCount < dims1[2]; vectorLoopCount++)
                *dstPtrRowTemp++ = *srcPtrRowTemp++;
            srcPtrRow += strides1[1];
            dstPtrRow += dstStrides[1];
        }
        Rpp8u *srcPtrRow2 = srcPtr2;
        dstPtrRow = dstPtr + dims1[1] * dstStrides[1];
        for(Rpp32u j = 0; j < dims2[1]; j++)
        {
            Rpp8u *srcPtrRowTemp2 = srcPtrRow2;
            Rpp8u *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst;
            for(; vectorLoopCount < alignedLength2; vectorLoopCount += vectorIncrement)
            {
                rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp2, &pDst);
                rpp_simd_store(rpp_store8_f32_to_u8_avx, dstPtrRowTemp, &pDst);
                srcPtrRowTemp2 += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(; vectorLoopCount < dims2[2]; vectorLoopCount++)
            {
                *dstPtrRowTemp++ = *srcPtrRowTemp2++;
            }
            srcPtrRow2 += strides2[1];
            dstPtrRow += dstStrides[1];
        }
        
        srcPtr1 += strides1[0];
        srcPtr2 += strides2[0];
        dstPtr += dstHeightStride * dstStrides[1];
    }
}

// Computes concat for 3D variants
void concat_3D_axismask0_pln_tensor(Rpp8u *srcPtr1, Rpp8u *srcPtr2, RpptGenericDescPtr srcPtr1GenericDescPtr, RpptGenericDescPtr srcPtr2GenericDescPtr, Rpp8u *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *dims1, Rpp32u *strides1, Rpp32u *dims2, Rpp32u *strides2, Rpp32u *dstStrides, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    
    // Copy all channels from srcPtr1
    for(Rpp32u i = 0; i < dims1[0]; i++)
    {
        Rpp32u bufferLength = dims1[2];
        Rpp32u alignedLength = (bufferLength / 8) * 8;
        Rpp8u *srcPtrRow = srcPtr1;
        Rpp8u *dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < dims1[1]; j++)
        {
            Rpp8u *srcPtrRowTemp = srcPtrRow;
            Rpp8u *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst;
            for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
            {
                rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp, &pDst);
                rpp_simd_store(rpp_store8_f32_to_u8_avx, dstPtrRowTemp, &pDst);
                srcPtrRowTemp += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(; vectorLoopCount < dims1[2]; vectorLoopCount++)
                *dstPtrRowTemp++ = *srcPtrRowTemp++;
            srcPtrRow += strides1[1];
            dstPtrRow += dstStrides[1];
        }
        srcPtr1 += strides1[0];
        dstPtr += dstStrides[0];
    }
    
    // Copy all channels from srcPtr2
    for(Rpp32u i = 0; i < dims2[0]; i++)
    {
        Rpp32u bufferLength = dims2[2];
        Rpp32u alignedLength = (bufferLength / 8) * 8;
        Rpp8u *srcPtrRow = srcPtr2;
        Rpp8u *dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < dims2[1]; j++)
        {
            Rpp8u *srcPtrRowTemp = srcPtrRow;
            Rpp8u *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst;
            for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
            {
                rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp, &pDst);
                rpp_simd_store(rpp_store8_f32_to_u8_avx, dstPtrRowTemp, &pDst);
                srcPtrRowTemp += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(; vectorLoopCount < dims2[2]; vectorLoopCount++)
                *dstPtrRowTemp++ = *srcPtrRowTemp++;
            srcPtrRow += strides2[1];
            dstPtrRow += dstStrides[1];
        }
        srcPtr2 += strides2[0];
        dstPtr += dstStrides[0];
    }
}

// Computes concatenation for N-Dimensional tensors recursively
template<typename T1, typename T2>
void concat_recursive_ND_tensor(T1 *srcPtr, Rpp32u *srcTensor1Strides, T2 *dstPtr, Rpp32u *dims, Rpp32u tensorDim, Rpp32u level, Rpp32u axisMask, Rpp32u maxDims)
{
    if(level == (tensorDim - 1))
    {
        for (Rpp32u i = 0; i < dims[level]; i++)
        {
            *(dstPtr + i) = *srcPtr;
            srcPtr++;
        }
    }
    else
    {
        int size = 1;
        for(int i = level + 1; i < tensorDim; i++)
            size = size * dims[i];
        for (Rpp32u i = 0; i < dims[level]; i++)
        {
            concat_recursive_ND_tensor(srcPtr, srcTensor1Strides, dstPtr, dims, tensorDim, level + 1, axisMask, maxDims);
            dstPtr += srcTensor1Strides[level + 1];
            srcPtr += srcTensor1Strides[level + 1];
        }
    }
}

// Computes concatenation for N-Dimensional tensors
template<typename T1, typename T2>
void concat_ND_tensor(T1 *srcPtr1, T1 *srcPtr2, Rpp32u *srcTensor1Strides, Rpp32u *srcTensor2Strides, Rpp32u *dstStride, T2 *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *dims1, Rpp32u *dims2, Rpp32u tensorDim, Rpp32u level, Rpp32u axisMask, Rpp32u maxDims)
{

    if(level >= axisMask)
    {
        concat_recursive_ND_tensor(srcPtr1, srcTensor1Strides, dstPtr, dims1, tensorDim, level, axisMask, maxDims);
        dstPtr += srcTensor1Strides[level];
        concat_recursive_ND_tensor(srcPtr2, srcTensor2Strides, dstPtr, dims2, tensorDim, level, axisMask, maxDims);
    }
    else
    {
        for (Rpp32u i = 0; i < dims1[level]; i++)
        {
            concat_ND_tensor(srcPtr1, srcPtr2, srcTensor1Strides, srcTensor2Strides, dstStride,  dstPtr, dstGenericDescPtr, dims1, dims2, tensorDim, level + 1, axisMask, maxDims);
            dstPtr += dstStride[level + 1];
            srcPtr1 += srcTensor1Strides[level + 1];
            srcPtr2 += srcTensor2Strides[level + 1];
        }
    }
}

RppStatus concat_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                     Rpp32f *srcPtr2,
                                     RpptGenericDescPtr srcPtr1GenericDescPtr,
                                     RpptGenericDescPtr srcPtr2GenericDescPtr,
                                     Rpp32f *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     Rpp32u axisMask,
                                     Rpp32u *roiTensorSrc1,
                                     Rpp32u *roiTensorSrc2,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    Rpp32u tensorDims = srcPtr1GenericDescPtr->numDims - 1;  // Ignoring batchSize here to get tensor dimensions.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    // Calculate cumulative offsets for variable-sized tensors
    Rpp32u *src1Offsets = reinterpret_cast<Rpp32u*>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    Rpp32u *src2Offsets = src1Offsets + batchSize + 1;
    Rpp32u *dstOffsets = src2Offsets + batchSize + 1;
    
    src1Offsets[0] = 0;
    src2Offsets[0] = 0;
    dstOffsets[0] = 0;
    
    for(int i = 0; i < batchSize; i++)
    {
        Rpp32u *roi1 = roiTensorSrc1 + i * tensorDims * 2;
        Rpp32u *roi2 = roiTensorSrc2 + i * tensorDims * 2;
        Rpp32u *length1 = &roi1[tensorDims];
        Rpp32u *length2 = &roi2[tensorDims];
        
        // Calculate size of current tensor
        Rpp32u size1 = 1, size2 = 1, dstSize = 1;
        for(int j = 0; j < tensorDims; j++)
        {
            size1 *= length1[j];
            size2 *= length2[j];
            if(j == axisMask)
                dstSize *= (length1[j] + length2[j]);
            else
                dstSize *= length1[j];
        }
        
        src1Offsets[i + 1] = src1Offsets[i] + size1;
        src2Offsets[i + 1] = src2Offsets[i] + size2;
        dstOffsets[i + 1] = dstOffsets[i] + dstSize;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        Rpp32u *roi1 = roiTensorSrc1 + batchCount * tensorDims * 2;
        Rpp32u *roi2 = roiTensorSrc2 + batchCount * tensorDims * 2;
        Rpp32u *length1 = &roi1[tensorDims];
        Rpp32u *length2 = &roi2[tensorDims];

        // Use cumulative offsets for variable-sized tensors
        Rpp32f *srcPtrTemp = srcPtr1 + src1Offsets[batchCount];
        Rpp32f *srcPtrTemp1 = srcPtr2 + src2Offsets[batchCount];
        Rpp32f *dstPtrTemp = dstPtr + dstOffsets[batchCount];

        Rpp32u src1ReductionDims[3], srcTensor1Strides[3], src2ReductionDims[3], srcTensor2Strides[3], dstStride[3];

        // Compute stride arrays for the current tensor based on its actual ROI dimensions
        // These strides are specific to this single tensor and account for variable sizes within the batch
        Rpp32u src1TensorStrides[RPPT_MAX_DIMS], src2TensorStrides[RPPT_MAX_DIMS], dstTensorStrides[RPPT_MAX_DIMS];
        src1TensorStrides[tensorDims] = 1;
        src2TensorStrides[tensorDims] = 1;
        dstTensorStrides[tensorDims] = 1;
        
        if (tensorDims > 0)
        {
            for(int i = tensorDims - 1; i >= 0; i--)
            {
                src1TensorStrides[i] = src1TensorStrides[i + 1] * length1[i];
                src2TensorStrides[i] = src2TensorStrides[i + 1] * length2[i];
                if(i == axisMask)
                    dstTensorStrides[i] = dstTensorStrides[i + 1] * (length1[i] + length2[i]);
                else
                    dstTensorStrides[i] = dstTensorStrides[i + 1] * length1[i];
            }
        }

        // Use the helper function to update strides and dimensions
        updateStridesAndDims(tensorDims, axisMask, src1TensorStrides, src2TensorStrides, dstTensorStrides,
                             src1ReductionDims, srcTensor1Strides, src2ReductionDims, srcTensor2Strides, dstStride, length1, length2);

        if (tensorDims == 2) // Called for 2D tensor cases
        {
            concat_2D_tensor(srcPtrTemp, srcPtrTemp1, rpp_load8_f32_to_f32_avx, rpp_store8_f32_to_f32_avx,
            srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtrTemp, dstGenericDescPtr,
            src1ReductionDims, srcTensor1Strides, src2ReductionDims, srcTensor2Strides, axisMask);
        }
        else if (tensorDims == 3) // Called for 3D tensor cases
        {
            concat_3D_tensor(srcPtrTemp, srcPtrTemp1, rpp_load8_f32_to_f32_avx, rpp_store8_f32_to_f32_avx,
            srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtrTemp, dstGenericDescPtr,
            src1ReductionDims, srcTensor1Strides, src2ReductionDims, srcTensor2Strides, dstStride, axisMask);
        }
        else // Handle ND tensors
        {
            concat_ND_tensor(srcPtrTemp, srcPtrTemp1, src1TensorStrides, src2TensorStrides,
            dstTensorStrides, dstPtrTemp, dstGenericDescPtr, length1, length2, tensorDims, 0, axisMask, tensorDims);
        }
    }
    return RPP_SUCCESS;
}

RppStatus concat_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                   Rpp8u *srcPtr2,
                                   RpptGenericDescPtr srcPtr1GenericDescPtr,
                                   RpptGenericDescPtr srcPtr2GenericDescPtr,
                                   Rpp8u *dstPtr,
                                   RpptGenericDescPtr dstGenericDescPtr,
                                   Rpp32u axisMask,
                                   Rpp32u *roiTensorSrc1,
                                   Rpp32u *roiTensorSrc2,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle)
{
    Rpp32u tensorDims = srcPtr1GenericDescPtr->numDims - 1;  // Ignoring batchSize here to get tensor dimensions.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    // Calculate cumulative offsets for variable-sized tensors
    Rpp32u *src1Offsets = reinterpret_cast<Rpp32u*>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    Rpp32u *src2Offsets = src1Offsets + batchSize + 1;
    Rpp32u *dstOffsets = src2Offsets + batchSize + 1;
    
    src1Offsets[0] = 0;
    src2Offsets[0] = 0;
    dstOffsets[0] = 0;
    
    for(int i = 0; i < batchSize; i++)
    {
        Rpp32u *roi1 = roiTensorSrc1 + i * tensorDims * 2;
        Rpp32u *roi2 = roiTensorSrc2 + i * tensorDims * 2;
        Rpp32u *length1 = &roi1[tensorDims];
        Rpp32u *length2 = &roi2[tensorDims];
        
        // Calculate size of current tensor
        Rpp32u size1 = 1, size2 = 1, dstSize = 1;
        for(int j = 0; j < tensorDims; j++)
        {
            size1 *= length1[j];
            size2 *= length2[j];
            if(j == axisMask)
                dstSize *= (length1[j] + length2[j]);
            else
                dstSize *= length1[j];
        }
        
        src1Offsets[i + 1] = src1Offsets[i] + size1;
        src2Offsets[i + 1] = src2Offsets[i] + size2;
        dstOffsets[i + 1] = dstOffsets[i] + dstSize;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        Rpp32u *roi1 = roiTensorSrc1 + batchCount * tensorDims * 2;
        Rpp32u *roi2 = roiTensorSrc2 + batchCount * tensorDims * 2;
        Rpp32u *length1 = &roi1[tensorDims];
        Rpp32u *length2 = &roi2[tensorDims];

        // Use cumulative offsets for variable-sized tensors
        Rpp8u *srcPtrTemp = srcPtr1 + src1Offsets[batchCount];
        Rpp8u *srcPtrTemp1 = srcPtr2 + src2Offsets[batchCount];
        Rpp8u *dstPtrTemp = dstPtr + dstOffsets[batchCount];

        Rpp32u src1ReductionDims[3], srcTensor1Strides[3], src2ReductionDims[3], srcTensor2Strides[3], dstStride[3];

        // Compute stride arrays for the current tensor based on its actual ROI dimensions
        // These strides are specific to this single tensor and account for variable sizes within the batch
        Rpp32u src1TensorStrides[RPPT_MAX_DIMS], src2TensorStrides[RPPT_MAX_DIMS], dstTensorStrides[RPPT_MAX_DIMS];
        
        src1TensorStrides[tensorDims] = 1;
        src2TensorStrides[tensorDims] = 1;
        dstTensorStrides[tensorDims] = 1;
        
        if (tensorDims > 0)
        {
            for(int i = tensorDims - 1; i >= 0; i--)
            {
                src1TensorStrides[i] = src1TensorStrides[i + 1] * length1[i];
                src2TensorStrides[i] = src2TensorStrides[i + 1] * length2[i];
                if(i == axisMask)
                    dstTensorStrides[i] = dstTensorStrides[i + 1] * (length1[i] + length2[i]);
                else
                    dstTensorStrides[i] = dstTensorStrides[i + 1] * length1[i];
            }
        }

        // Use the helper function to update strides and dimensions
        updateStridesAndDims(tensorDims, axisMask, src1TensorStrides, src2TensorStrides, dstTensorStrides,
                             src1ReductionDims, srcTensor1Strides, src2ReductionDims, srcTensor2Strides, dstStride, length1, length2);

        if (tensorDims == 2) // Called for 2D tensor cases
        {
            concat_2D_tensor(srcPtrTemp, srcPtrTemp1, rpp_load8_u8_to_f32_avx, rpp_store8_f32_to_u8_avx,
                             srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtrTemp, dstGenericDescPtr,
                             src1ReductionDims, srcTensor1Strides, src2ReductionDims, srcTensor2Strides, axisMask);
        }
        else if (tensorDims == 3) // Called for 3D tensor cases
        {
            concat_3D_tensor(srcPtrTemp, srcPtrTemp1, rpp_load8_u8_to_f32_avx, rpp_store8_f32_to_u8_avx,
                             srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtrTemp, dstGenericDescPtr,
                             src1ReductionDims, srcTensor1Strides, src2ReductionDims, srcTensor2Strides, dstStride, axisMask);
        }
        else // Handle ND tensors
        {
            concat_ND_tensor(srcPtrTemp, srcPtrTemp1, src1TensorStrides, src2TensorStrides,
                             dstTensorStrides, dstPtrTemp, dstGenericDescPtr, length1, length2, tensorDims, 0, axisMask, tensorDims);
        }
    }
    return RPP_SUCCESS;
}

template<typename T1, typename T2>
RppStatus concat_generic_host_tensor(T1 *srcPtr1,
                                     T1 *srcPtr2,
                                     RpptGenericDescPtr srcPtr1GenericDescPtr,
                                     RpptGenericDescPtr srcPtr2GenericDescPtr,
                                     T2 *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     Rpp32u axisMask,
                                     Rpp32u *roiTensorSrc1,
                                     Rpp32u *roiTensorSrc2,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    Rpp32u tensorDims = srcPtr1GenericDescPtr->numDims - 1; // Ignoring batchSize here to get tensor dimensions.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    // Calculate cumulative offsets for variable-sized tensors
    Rpp32u *src1Offsets = reinterpret_cast<Rpp32u*>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    Rpp32u *src2Offsets = src1Offsets + batchSize + 1;
    Rpp32u *dstOffsets = src2Offsets + batchSize + 1;
    
    src1Offsets[0] = 0;
    src2Offsets[0] = 0;
    dstOffsets[0] = 0;
    
    for(int i = 0; i < batchSize; i++)
    {
        Rpp32u *roi1 = roiTensorSrc1 + i * tensorDims * 2;
        Rpp32u *roi2 = roiTensorSrc2 + i * tensorDims * 2;
        Rpp32u *length1 = &roi1[tensorDims];
        Rpp32u *length2 = &roi2[tensorDims];
        
        // Calculate size of current tensor
        Rpp32u size1 = 1, size2 = 1, dstSize = 1;
        for(int j = 0; j < tensorDims; j++)
        {
            size1 *= length1[j];
            size2 *= length2[j];
            if(j == axisMask)
                dstSize *= (length1[j] + length2[j]);
            else
                dstSize *= length1[j];
        }
        
        src1Offsets[i + 1] = src1Offsets[i] + size1;
        src2Offsets[i + 1] = src2Offsets[i] + size2;
        dstOffsets[i + 1] = dstOffsets[i] + dstSize;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        Rpp32u *roi1 = roiTensorSrc1 + batchCount * tensorDims * 2;
        Rpp32u *roi2 = roiTensorSrc2 + batchCount * tensorDims * 2;
        Rpp32u *length1 = &roi1[tensorDims];
        Rpp32u *length2 = &roi2[tensorDims];

        // Use cumulative offsets for variable-sized tensors
        T1 *srcPtrTemp = srcPtr1 + src1Offsets[batchCount];
        T1 *srcPtrTemp1 = srcPtr2 + src2Offsets[batchCount];
        T2 *dstPtrTemp = dstPtr + dstOffsets[batchCount];

        // Compute stride arrays for the current tensor based on its actual ROI dimensions
        // These strides are specific to this single tensor and account for variable sizes within the batch
        Rpp32u src1TensorStrides[RPPT_MAX_DIMS], src2TensorStrides[RPPT_MAX_DIMS], dstTensorStrides[RPPT_MAX_DIMS];
        
        src1TensorStrides[tensorDims] = 1;
        src2TensorStrides[tensorDims] = 1;
        dstTensorStrides[tensorDims] = 1;
        
        if (tensorDims > 0)
        {
            for(int i = tensorDims - 1; i >= 0; i--)
            {
                src1TensorStrides[i] = src1TensorStrides[i + 1] * length1[i];
                src2TensorStrides[i] = src2TensorStrides[i + 1] * length2[i];
                if(i == axisMask)
                    dstTensorStrides[i] = dstTensorStrides[i + 1] * (length1[i] + length2[i]);
                else
                    dstTensorStrides[i] = dstTensorStrides[i + 1] * length1[i];
            }
        }

        concat_ND_tensor(srcPtrTemp, srcPtrTemp1, src1TensorStrides, src2TensorStrides, dstTensorStrides, dstPtrTemp, dstGenericDescPtr, length1, length2, tensorDims, 0, axisMask, tensorDims);
    }

    return RPP_SUCCESS;
}

template RppStatus concat_generic_host_tensor<Rpp16f, Rpp16f>(Rpp16f*,
                                                              Rpp16f*,
                                                              RpptGenericDescPtr,
                                                              RpptGenericDescPtr,
                                                              Rpp16f*,
                                                              RpptGenericDescPtr,
                                                              Rpp32u,
                                                              Rpp32u*,
                                                              Rpp32u*,
                                                              RppLayoutParams,
                                                              rpp::Handle&);

template RppStatus concat_generic_host_tensor<Rpp8s, Rpp8s>(Rpp8s*,
                                                            Rpp8s*,
                                                            RpptGenericDescPtr,
                                                            RpptGenericDescPtr,
                                                            Rpp8s*,
                                                            RpptGenericDescPtr,
                                                            Rpp32u,
                                                            Rpp32u*,
                                                            Rpp32u*,
                                                            RppLayoutParams,
                                                            rpp::Handle&);
