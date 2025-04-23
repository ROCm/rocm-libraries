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
                                 RpptGenericDescPtr srcPtr1GenericDescPtr, RpptGenericDescPtr srcPtr2GenericDescPtr, RpptGenericDescPtr dstGenericDescPtr,
                                 Rpp32u* src1ReductionDims, Rpp32u* srcTensor1Strides, Rpp32u* src2ReductionDims, Rpp32u* srcTensor2Strides, Rpp32u* dstStride,
                                 Rpp32u* length, Rpp32u* length1)
{
    if (tensorDims == 2)
    {
        if (axisMask == 0)
        {
            src1ReductionDims[0] = 1;
            src1ReductionDims[1] = length[0] * length[1];
            srcTensor1Strides[0] = 1;
            srcTensor1Strides[1] = srcPtr1GenericDescPtr->strides[0];
            src2ReductionDims[0] = 1;
            src2ReductionDims[1] = length1[0] * length1[1];
            srcTensor2Strides[0] = 1;
            srcTensor2Strides[1] = srcPtr2GenericDescPtr->strides[0];
            dstStride[0] = 1;
            dstStride[1] = dstGenericDescPtr->strides[0];
        }
        else if (axisMask == 1)
        {
            src1ReductionDims[0] = length[0];
            src1ReductionDims[1] = length[1];
            srcTensor1Strides[0] = srcPtr1GenericDescPtr->strides[0];
            srcTensor1Strides[1] = srcPtr1GenericDescPtr->strides[1];
            src2ReductionDims[0] = length1[0];
            src2ReductionDims[1] = length1[1];
            srcTensor2Strides[0] = srcPtr2GenericDescPtr->strides[0];
            srcTensor2Strides[1] = srcPtr2GenericDescPtr->strides[1];
            dstStride[0] = dstGenericDescPtr->strides[0];
            dstStride[1] = dstGenericDescPtr->strides[1];
        }
    }
    else if (tensorDims == 3)
    {
        if (axisMask == 0)
        {
            src1ReductionDims[0] = 1;
            src1ReductionDims[1] = 1;
            src1ReductionDims[2] = length[0] * length[1] * length[2];
            srcTensor1Strides[0] = 1;
            srcTensor1Strides[1] = 1;
            srcTensor1Strides[2] = srcPtr1GenericDescPtr->strides[0];
            src2ReductionDims[0] = 1;
            src2ReductionDims[1] = 1;
            src2ReductionDims[2] = length1[0] * length1[1] * length1[2];
            srcTensor2Strides[0] = 1;
            srcTensor2Strides[1] = 1;
            srcTensor2Strides[2] = srcPtr2GenericDescPtr->strides[0];
            dstStride[0] = 1;
            dstStride[1] = 1;
            dstStride[2] = dstGenericDescPtr->strides[0];
        }
        else if (axisMask == 1)
        {
            src1ReductionDims[0] = 1;
            src1ReductionDims[1] = length[0];
            src1ReductionDims[2] = length[1] * length[2];
            srcTensor1Strides[0] = 1;
            srcTensor1Strides[1] = srcPtr1GenericDescPtr->strides[0];
            srcTensor1Strides[2] = srcPtr1GenericDescPtr->strides[1];
            src2ReductionDims[0] = 1;
            src2ReductionDims[1] = length1[0];
            src2ReductionDims[2] = length1[1] * length1[2];
            srcTensor2Strides[0] = 1;
            srcTensor2Strides[1] = srcPtr2GenericDescPtr->strides[0];
            srcTensor2Strides[2] = srcPtr2GenericDescPtr->strides[1];
            dstStride[0] = 1;
            dstStride[1] = dstGenericDescPtr->strides[0];
            dstStride[2] = dstGenericDescPtr->strides[1];
        }
        else if (axisMask == 2)
        {
            src1ReductionDims[0] = length[0];
            src1ReductionDims[1] = length[1];
            src1ReductionDims[2] = length[2];
            srcTensor1Strides[0] = srcPtr1GenericDescPtr->strides[0];
            srcTensor1Strides[1] = srcPtr1GenericDescPtr->strides[1];
            srcTensor1Strides[2] = srcPtr1GenericDescPtr->strides[2];
            src2ReductionDims[0] = length1[0];
            src2ReductionDims[1] = length1[1];
            src2ReductionDims[2] = length1[2];
            srcTensor2Strides[0] = srcPtr2GenericDescPtr->strides[0];
            srcTensor2Strides[1] = srcPtr2GenericDescPtr->strides[1];
            srcTensor2Strides[2] = srcPtr2GenericDescPtr->strides[2];
            dstStride[0] = dstGenericDescPtr->strides[0];
            dstStride[1] = dstGenericDescPtr->strides[1];
            dstStride[2] = dstGenericDescPtr->strides[2];
        }
    }
}

// Computes concatenation for 2D tensors (Supports Rpp32f and Rpp8u)
template <typename T, typename SIMD_LOAD, typename SIMD_STORE>
void concat_2D_tensor(T *srcPtr1, T *srcPtr2, SIMD_LOAD simd_load, SIMD_STORE simd_store, RpptGenericDescPtr srcDescPtr, RpptGenericDescPtr srcDescPtr1, T *dstPtr, RpptGenericDescPtr dstDescPtr, Rpp32u *dims, Rpp32u *strides,Rpp32u *dims1, Rpp32u *strides1, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    Rpp32u bufferLength = (dims[1] < dims1[1]) ? dims[1] : dims1[1];
    Rpp32u alignedLength = (bufferLength / 8) * 8;
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        T *srcPtrTemp1 = srcPtr1 + i * strides[1];
        T *srcPtrTemp2 = srcPtr2 + i * strides[1];
        T *dstPtrTemp = dstPtr + (i * strides[1] * 2);

        Rpp32u vectorLoopCount = 0;
        __m256 pDst;
        for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
        {
            simd_load(srcPtrTemp1, &pDst);
            simd_store(dstPtrTemp, &pDst);
            simd_load(srcPtrTemp2, &pDst);
            simd_store(dstPtrTemp + strides[1], &pDst);
            srcPtrTemp1 += vectorIncrement;
            srcPtrTemp2 += vectorIncrement;
            dstPtrTemp += vectorIncrement;
        }
        dstPtrTemp = dstPtr + (i * strides[1] * 2);
        for(int j = vectorLoopCount; j < dims[1] ; j ++)
            *(dstPtrTemp + j) = *srcPtrTemp1++;
        for(int j = vectorLoopCount; j < dims1[1] ; j ++)
            *(dstPtrTemp + strides[1] + j)  = *srcPtrTemp2++;
    }
}

// Computes concatenation for 3D tensors (Supports Rpp32f and Rpp8u)
template <typename T, typename SIMD_LOAD, typename SIMD_STORE>
void concat_3D_tensor(T *srcPtr1, T *srcPtr2, SIMD_LOAD simd_load, SIMD_STORE simd_store, RpptGenericDescPtr srcPtr1GenericDescPtr, RpptGenericDescPtr srcPtr2GenericDescPtr, T *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *dims, Rpp32u *strides, Rpp32u *dims1, Rpp32u *strides1, Rpp32u *dstStrides, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    Rpp32u bufferLength = (dims[2] < dims1[2]) ? dims[2] : dims1[2];
    Rpp32u alignedLength = (bufferLength / 8) * 8;
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        T *srcPtrRow1 = srcPtr1;
        T *srcPtrRow2 = srcPtr2;
        T *dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < dims[1]; j++)
        {
            T *srcPtrRowTemp1 = srcPtrRow1;
            T *srcPtrRowTemp2 = srcPtrRow2;
            T *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst;
            for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
            {
                simd_load(srcPtrRowTemp1, &pDst);
                simd_store(dstPtrRowTemp, &pDst);
                simd_load(srcPtrRowTemp2, &pDst);
                simd_store(dstPtrRowTemp + strides[2], &pDst);
                srcPtrRowTemp1 += vectorIncrement;
                srcPtrRowTemp2 += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(Rpp32u k = vectorLoopCount ; k < dims[2] ; k++)
                *(dstPtrRowTemp + k - vectorLoopCount) = *srcPtrRowTemp1++;
            for(Rpp32u k = vectorLoopCount ; k < dims1[2] ; k++)
                *(dstPtrRowTemp + strides[2] + k - vectorLoopCount) = *srcPtrRowTemp2++;
            srcPtrRow1 += strides[2];
            srcPtrRow2 += strides1[2];
            dstPtrRow += dstStrides[2];
        }
        srcPtr1 += strides[1];
        srcPtr2 += strides1[1];
        dstPtr += dstStrides[1];
    }
}

// Computes concat for 3D variants
void concat_3D_axismask0_tensor(Rpp8u *srcPtr1, Rpp8u *srcPtr2, RpptGenericDescPtr srcPtr1GenericDescPtr, RpptGenericDescPtr srcPtr2GenericDescPtr, Rpp8u *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *dims, Rpp32u *strides, Rpp32u *dims1, Rpp32u *strides1, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        Rpp32u bufferLength = dims[2];
        Rpp32u alignedLength = (bufferLength / 8) * 8;
        Rpp32u bufferLength1 = dims1[2];
        Rpp32u alignedLength1 = (bufferLength1 / 8) * 8;
        Rpp8u *srcPtrRow = srcPtr1;
        Rpp8u *srcPtrRow1 = srcPtr2;
        Rpp8u *dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < dims[1]; j++)
        {
            Rpp8u *srcPtrRowTemp = srcPtrRow;
            Rpp8u *srcPtrRowTemp1 = srcPtrRow1;
            Rpp8u *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst ;
            for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
            {
                rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp, &pDst);
                rpp_simd_store(rpp_store8_f32_to_u8_avx, dstPtrRowTemp, &pDst);
                srcPtrRowTemp += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(; vectorLoopCount < dims[2] ; vectorLoopCount ++)
            {
                *dstPtrRowTemp++ = *srcPtrRowTemp++;
            }
            srcPtrRow += strides[2];
            dstPtrRow += strides[2];
        }
        dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < dims1[1]; j++)
        {
            Rpp8u *srcPtrRowTemp1 = srcPtrRow1;
            Rpp8u *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst ;
            for(; vectorLoopCount < alignedLength1; vectorLoopCount += vectorIncrement)
            {
                rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp1, &pDst);
                rpp_simd_store(rpp_store8_f32_to_u8_avx, (dstPtrRowTemp + strides[1]) , &pDst);
                srcPtrRowTemp1 += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(; vectorLoopCount < dims1[2] ; vectorLoopCount ++)
            {
                *(dstPtrRowTemp + strides[1]) = *srcPtrRowTemp1++;
                dstPtrRowTemp++;
            }
            srcPtrRow1 += strides1[2];
            dstPtrRow += strides1[2];
        }
        srcPtr1 += strides[1];
        srcPtr2 += strides1[1];
        dstPtr += strides[1] * 2;
    }
}

// Computes concat for 3D variants
void concat_3D_axismask0_pln_tensor(Rpp8u *srcPtr1, Rpp8u *srcPtr2, RpptGenericDescPtr srcPtr1GenericDescPtr, RpptGenericDescPtr srcPtr2GenericDescPtr, Rpp8u *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *dims, Rpp32u *strides, Rpp32u *dims1, Rpp32u *strides1, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        Rpp32u bufferLength = dims[2];
        Rpp32u alignedLength = (bufferLength / 8) * 8;
        Rpp32u bufferLength1 = dims1[2];
        Rpp32u alignedLength1 = (bufferLength1 / 8) * 8;
        Rpp8u *srcPtrRow = srcPtr1;
        Rpp8u *dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < dims[1]; j++)
        {
            Rpp8u *srcPtrRowTemp = srcPtrRow;
            Rpp8u *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst ;
            for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
            {
                rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp, &pDst);
                rpp_simd_store(rpp_store8_f32_to_u8_avx, dstPtrRowTemp, &pDst);
                srcPtrRowTemp += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(; vectorLoopCount < dims[2] ; vectorLoopCount ++)
            {
                *dstPtrRowTemp++ = *srcPtrRowTemp++;
            }
            srcPtrRow += strides[2];
            dstPtrRow += strides[2];
        }
        srcPtr1 += strides[1];
        dstPtr += strides[1];
    }
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        Rpp32u bufferLength1 = dims1[2];
        Rpp32u alignedLength = (bufferLength1 / 8) * 8;
        Rpp8u *srcPtrRow1 = srcPtr2;
        Rpp8u *dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < dims[1]; j++)
        {
            Rpp8u *srcPtrRowTemp1 = srcPtrRow1;
            Rpp8u *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst ;
            for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
            {
                rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp1, &pDst);
                rpp_simd_store(rpp_store8_f32_to_u8_avx, dstPtrRowTemp, &pDst);
                srcPtrRowTemp1 += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(; vectorLoopCount < dims[2] ; vectorLoopCount ++)
            {
                *dstPtrRowTemp++ = *srcPtrRowTemp1++;
            }
            srcPtrRow1 += strides[2];
            dstPtrRow += strides[2];
        }
        srcPtr2 += strides1[1];
        dstPtr += strides1[1];
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
void concat_ND_tensor(T1 *srcPtr1, T1 *srcPtr2, Rpp32u *srcTensor1Strides, Rpp32u *srcTensor2Strides, Rpp32u *dstStride, T2 *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *dims, Rpp32u *dims1, Rpp32u tensorDim, Rpp32u level, Rpp32u axisMask, Rpp32u maxDims)
{

    if(level >= axisMask)
    {
        concat_recursive_ND_tensor(srcPtr1, srcTensor1Strides, dstPtr, dims, tensorDim, level, axisMask, maxDims);
        dstPtr += srcTensor1Strides[level];
        concat_recursive_ND_tensor(srcPtr2, srcTensor2Strides, dstPtr, dims1, tensorDim, level, axisMask, maxDims);
    }
    else
    {
        for (Rpp32u i = 0; i < dims[level]; i++)
        {
            concat_ND_tensor(srcPtr1, srcPtr2, srcTensor1Strides, srcTensor2Strides, dstStride,  dstPtr, dstGenericDescPtr, dims, dims1, tensorDim, level + 1, axisMask, maxDims);
            dstPtr += dstStride[level + 1];
            srcPtr1 += srcTensor1Strides[level + 1];
            srcPtr2 += srcTensor1Strides[level + 1];
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
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u tensorDims = srcPtr1GenericDescPtr->numDims - 1;  // Ignoring batchSize here to get tensor dimensions.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    Rpp32u maxSize = 1;
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        Rpp32u *roi = roiTensorSrc1 + batchCount * tensorDims * 2;
        Rpp32u *roi1 = roiTensorSrc2 + batchCount * tensorDims * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[tensorDims];
        Rpp32u *length1 = &roi1[tensorDims];

        Rpp32f *srcPtrTemp = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        Rpp32f *srcPtrTemp1 = srcPtr2 + batchCount * srcPtr2GenericDescPtr->strides[0];
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u src1ReductionDims[3], srcTensor1Strides[3], src2ReductionDims[3], srcTensor2Strides[3], dstStride[3];

        // Use the helper function to update strides and dimensions
        updateStridesAndDims(tensorDims, axisMask, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstGenericDescPtr,
                             src1ReductionDims, srcTensor1Strides, src2ReductionDims, srcTensor2Strides, dstStride, length, length1);

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
            concat_ND_tensor(srcPtrTemp, srcPtrTemp1, srcPtr1GenericDescPtr->strides, srcPtr2GenericDescPtr->strides,
            dstGenericDescPtr->strides, dstPtrTemp, dstGenericDescPtr, length, length1, tensorDims, 0, axisMask, tensorDims);
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
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u tensorDims = srcPtr1GenericDescPtr->numDims - 1;  // Ignoring batchSize here to get tensor dimensions.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    Rpp32u maxSize = 1;
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        Rpp32u *roi = roiTensorSrc1 + batchCount * tensorDims * 2;
        Rpp32u *roi1 = roiTensorSrc2 + batchCount * tensorDims * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[tensorDims];
        Rpp32u *length1 = &roi1[tensorDims];

        Rpp8u *srcPtrTemp = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        Rpp8u *srcPtrTemp1 = srcPtr2 + batchCount * srcPtr2GenericDescPtr->strides[0];
        Rpp8u *dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u src1ReductionDims[3], srcTensor1Strides[3], src2ReductionDims[3], srcTensor2Strides[3], dstStride[3];
        updateStridesAndDims(tensorDims, axisMask, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstGenericDescPtr,
                             src1ReductionDims, srcTensor1Strides, src2ReductionDims, srcTensor2Strides, dstStride, length, length1);

        if (tensorDims == 2)
        {
            concat_2D_tensor(srcPtrTemp, srcPtrTemp1, rpp_load8_u8_to_f32_avx, rpp_store8_f32_to_u8_avx, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtrTemp, dstGenericDescPtr,
            src1ReductionDims, srcTensor1Strides, src2ReductionDims, srcTensor2Strides, axisMask);
        }
        else if (tensorDims == 3)
        {
            if (srcPtr1GenericDescPtr->layout == RpptLayout::NCHW && axisMask == 0)
                concat_3D_axismask0_pln_tensor(srcPtrTemp, srcPtrTemp1, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                                dstPtrTemp, dstGenericDescPtr, src1ReductionDims, srcTensor1Strides,
                                src2ReductionDims, srcTensor2Strides, axisMask);
            else if ((srcPtr1GenericDescPtr->layout == RpptLayout::NHWC && axisMask == 0) ||
                (srcPtr1GenericDescPtr->layout == RpptLayout::NCHW && axisMask == 1))
                concat_3D_axismask0_tensor(srcPtrTemp, srcPtrTemp1, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                            dstPtrTemp, dstGenericDescPtr, src1ReductionDims, srcTensor1Strides,
                            src2ReductionDims, srcTensor2Strides, axisMask);
            else
                concat_3D_tensor(srcPtrTemp, srcPtrTemp1, rpp_load8_u8_to_f32_avx, rpp_store8_f32_to_u8_avx,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtrTemp, dstGenericDescPtr,
                src1ReductionDims, srcTensor1Strides, src2ReductionDims, srcTensor2Strides, dstStride, axisMask);
        }
        else
        {
            concat_ND_tensor(srcPtrTemp, srcPtrTemp1, srcPtr1GenericDescPtr->strides, srcPtr2GenericDescPtr->strides,
            dstGenericDescPtr->strides, dstPtrTemp, dstGenericDescPtr, length, length1, tensorDims, 0, axisMask, tensorDims);
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
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u tensorDims = srcPtr1GenericDescPtr->numDims - 1; // Ignoring batchSize here to get tensor dimensions.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        T1 *srcPtrTemp;
        T1 *srcPtrTemp1;
        T2 *dstPtrTemp;
        srcPtrTemp = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        srcPtrTemp1 = srcPtr2 + batchCount * srcPtr1GenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * (dstGenericDescPtr->strides[0]);
        int size = 1;
        Rpp32u *roi = roiTensorSrc1 + batchCount * tensorDims * 2;
        Rpp32u *roi1 = roiTensorSrc2 + batchCount * tensorDims * 2;
        Rpp32u *begin = roi;
        Rpp32u *begin1 = roi1;
        Rpp32u *length = &roi[tensorDims];
        Rpp32u *length1 = &roi1[tensorDims];
        Rpp32u level = 0;
        T1 *srcPtrChannel = srcPtrTemp;
        T1 *srcPtrChannel1 = srcPtrTemp1;
        for(int i = 0; i < tensorDims; i++)
        {
            srcPtrChannel += begin[i] * srcPtr1GenericDescPtr->strides[i + 1];
            srcPtrChannel1 += begin1[i] * srcPtr2GenericDescPtr->strides[i + 1];
        }
        concat_ND_tensor(srcPtrChannel, srcPtrChannel1, srcPtr1GenericDescPtr->strides, srcPtr2GenericDescPtr->strides, dstGenericDescPtr->strides, dstPtrTemp, dstGenericDescPtr, length, length1, tensorDims, level, axisMask, tensorDims);
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