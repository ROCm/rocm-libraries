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

#include "host_tensor_executors.hpp"

template<typename T>
RppStatus slice_host_tensor(T *srcPtr,
                            RpptGenericDescPtr srcGenericDescPtr,
                            T *dstPtr,
                            RpptGenericDescPtr dstGenericDescPtr,
                            Rpp32s *anchorTensor,
                            Rpp32s *shapeTensor,
                            T* fillValue,
                            bool enablePadding,
                            Rpp32u *roiTensor,
                            RppLayoutParams layoutParams,
                            rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u numDims = srcGenericDescPtr->numDims - 1; // exclude batchsize from input dims

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
    {
        T *srcPtrTemp, *dstPtrTemp;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        T *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrTemp;

        Rpp32s *anchor = &anchorTensor[batchCount * numDims];
        Rpp32s *shape = &shapeTensor[batchCount * numDims];

        // get the starting address of length values from roiTensor
        Rpp32u *roi = roiTensor + batchCount * numDims * 2;
        Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);

        if (numDims == 4)
        {
            // order of dims
            Rpp32s dimsOrder[3];
            if (dstGenericDescPtr->layout == RpptLayout::NCDHW)
            {
                dimsOrder[0] = 1;  // depth
                dimsOrder[1] = 2;  // height
                dimsOrder[2] = 3;  // width
            }
            else
            {
                dimsOrder[0] = 0;  // depth
                dimsOrder[1] = 1;  // height
                dimsOrder[2] = 2;  // width
            }
            Rpp32u maxDepth = std::min(shape[dimsOrder[0]], length[dimsOrder[0]] - anchor[dimsOrder[0]]);
            Rpp32u maxHeight = std::min(shape[dimsOrder[1]], length[dimsOrder[1]] - anchor[dimsOrder[1]]);
            Rpp32u maxWidth = std::min(shape[dimsOrder[2]], length[dimsOrder[2]] - anchor[dimsOrder[2]]);
            Rpp32u bufferLength = maxWidth * layoutParams.bufferMultiplier;
            Rpp32u copyLengthInBytes = bufferLength * sizeof(T);

            // if padding is required, fill the buffer with fill value specified
            bool needPadding = (((anchor[dimsOrder[0]] + shape[dimsOrder[0]]) > length[dimsOrder[0]]) ||
                                ((anchor[dimsOrder[1]] + shape[dimsOrder[1]]) > length[dimsOrder[1]]) ||
                                ((anchor[dimsOrder[2]] + shape[dimsOrder[2]]) > length[dimsOrder[2]]));
            if (needPadding && enablePadding)
                std::fill(dstPtrChannel, dstPtrChannel + dstGenericDescPtr->strides[0] - 1, *fillValue);

            // slice without fused output-layout toggle (NCDHW -> NCDHW)
            if (dstGenericDescPtr->layout == RpptLayout::NCDHW)
            {
                srcPtrChannel = srcPtrTemp + (anchor[1] * srcGenericDescPtr->strides[2]) + (anchor[2] * srcGenericDescPtr->strides[3]) + (anchor[3] * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
                {
                    T *srcPtrDepth, *dstPtrDepth;
                    srcPtrDepth = srcPtrChannel;
                    dstPtrDepth = dstPtrChannel;
                    for(int i = 0; i < maxDepth; i++)
                    {
                        T *srcPtrRow, *dstPtrRow;
                        srcPtrRow = srcPtrDepth;
                        dstPtrRow = dstPtrDepth;
                        for(int j = 0; j < maxHeight; j++)
                        {
                            memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                            srcPtrRow += srcGenericDescPtr->strides[3];
                            dstPtrRow += dstGenericDescPtr->strides[3];
                        }
                        srcPtrDepth += srcGenericDescPtr->strides[2];
                        dstPtrDepth += dstGenericDescPtr->strides[2];
                    }
                    srcPtrChannel += srcGenericDescPtr->strides[1];
                    dstPtrChannel += srcGenericDescPtr->strides[1];
                }
            }

            // slice without fused output-layout toggle (NDHWC -> NDHWC)
            else if (dstGenericDescPtr->layout == RpptLayout::NDHWC)
            {
                srcPtrChannel = srcPtrTemp + (anchor[0] * srcGenericDescPtr->strides[1]) + (anchor[1] * srcGenericDescPtr->strides[2]) + (anchor[2] * layoutParams.bufferMultiplier);
                T *srcPtrDepth = srcPtrChannel;
                T *dstPtrDepth = dstPtrChannel;
                for(int i = 0; i < maxDepth; i++)
                {
                    T *srcPtrRow, *dstPtrRow;
                    srcPtrRow = srcPtrDepth;
                    dstPtrRow = dstPtrDepth;
                    for(int j = 0; j < maxHeight; j++)
                    {
                        memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                        srcPtrRow += srcGenericDescPtr->strides[2];
                        dstPtrRow += dstGenericDescPtr->strides[2];
                    }
                    srcPtrDepth += srcGenericDescPtr->strides[1];
                    dstPtrDepth += dstGenericDescPtr->strides[1];
                }
            }
        }
        else if (numDims == 3)
        {
            // order of dims
            Rpp32s dimsOrder[2];
            if (dstGenericDescPtr->layout == RpptLayout::NCHW)
            {
                dimsOrder[0] = 1;  // height
                dimsOrder[1] = 2;  // width
            }
            else
            {
                dimsOrder[0] = 0;  // height
                dimsOrder[1] = 1;  // width
            }

            Rpp32u maxHeight = std::min(shape[dimsOrder[0]], length[dimsOrder[0]] - anchor[dimsOrder[0]]);
            Rpp32u maxWidth = std::min(shape[dimsOrder[1]], length[dimsOrder[1]] - anchor[dimsOrder[1]]);
            Rpp32u bufferLength = maxWidth * layoutParams.bufferMultiplier;
            Rpp32u copyLengthInBytes = bufferLength * sizeof(T);

            // if padding is required, fill the buffer with fill value specified
            bool needPadding = ((anchor[dimsOrder[0]] + shape[dimsOrder[0]]) > length[dimsOrder[0]]) ||
                               ((anchor[dimsOrder[1]] + shape[dimsOrder[1]]) > length[dimsOrder[1]]);
            if (needPadding && enablePadding)
                std::fill(dstPtrChannel, dstPtrChannel + dstGenericDescPtr->strides[0] - 1, *fillValue);

            // slice without fused output-layout toggle (NCHW -> NCHW)
            if (dstGenericDescPtr->layout == RpptLayout::NCHW)
            {
                srcPtrChannel = srcPtrTemp + (anchor[1] * srcGenericDescPtr->strides[2]) + (anchor[2] * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
                {
                    T *srcPtrRow, *dstPtrRow;
                    srcPtrRow = srcPtrChannel;
                    dstPtrRow = dstPtrChannel;
                    for(int j = 0; j < maxHeight; j++)
                    {
                        memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                        srcPtrRow += srcGenericDescPtr->strides[2];
                        dstPtrRow += dstGenericDescPtr->strides[2];
                    }
                    srcPtrChannel += srcGenericDescPtr->strides[1];
                    dstPtrChannel += srcGenericDescPtr->strides[1];
                }
            }

            // slice without fused output-layout toggle (NHWC -> NHWC)
            else if (dstGenericDescPtr->layout == RpptLayout::NHWC)
            {
                srcPtrChannel = srcPtrTemp + (anchor[0] * srcGenericDescPtr->strides[1]) + (anchor[1] * layoutParams.bufferMultiplier);
                T *srcPtrRow = srcPtrChannel;
                T *dstPtrRow = dstPtrChannel;
                for(int j = 0; j < maxHeight; j++)
                {
                    memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                    srcPtrRow += srcGenericDescPtr->strides[1];
                    dstPtrRow += dstGenericDescPtr->strides[1];
                }
            }
        }
        else if (numDims == 2)
        {
            srcPtrChannel = srcPtrTemp + (anchor[0] * srcGenericDescPtr->strides[1]) + anchor[1];
            Rpp32u maxHeight = std::min(shape[0], length[0] - anchor[0]);
            Rpp32u maxWidth = std::min(shape[1], length[1] - anchor[1]);
            Rpp32u copyLengthInBytes = maxWidth * sizeof(T);

            // if padding is required, fill the buffer with fill value specified
            bool needPadding = ((anchor[0] + shape[0]) > length[0]) ||
                                ((anchor[1] + shape[1]) > length[1]);
            if (needPadding && enablePadding)
                std::fill(dstPtrChannel, dstPtrChannel + dstGenericDescPtr->strides[0] - 1, *fillValue);

            T *srcPtrRow = srcPtrChannel;
            T *dstPtrRow = dstPtrChannel;
            for(int j = 0; j < maxHeight; j++)
            {
                memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                srcPtrRow += srcGenericDescPtr->strides[1];
                dstPtrRow += dstGenericDescPtr->strides[1];
            }
        }
        else if (numDims == 1)
        {
            srcPtrChannel = srcPtrTemp + anchor[0];
            Rpp32u maxLength = std::min(shape[0], length[0] - anchor[0]);
            Rpp32u copyLengthInBytes = maxLength * sizeof(T);

            // if padding is required, fill the buffer with fill value specified
            bool needPadding = ((anchor[0] + shape[0]) > length[0]);
            if (needPadding && enablePadding)
                std::fill(dstPtrTemp, dstPtrTemp + dstGenericDescPtr->strides[0] - 1, *fillValue);
            memcpy(dstPtrChannel, srcPtrChannel, copyLengthInBytes);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus slice_host_tensor<Rpp32f>(Rpp32f*,
                                             RpptGenericDescPtr,
                                             Rpp32f*,
                                             RpptGenericDescPtr,
                                             Rpp32s*,
                                             Rpp32s*,
                                             Rpp32f*,
                                             bool,
                                             Rpp32u*,
                                             RppLayoutParams,
                                             rpp::Handle&);

template RppStatus slice_host_tensor<Rpp8u>(Rpp8u*,
                                            RpptGenericDescPtr,
                                            Rpp8u*,
                                            RpptGenericDescPtr,
                                            Rpp32s*,
                                            Rpp32s*,
                                            Rpp8u*,
                                            bool,
                                            Rpp32u*,
                                            RppLayoutParams,
                                            rpp::Handle&);
