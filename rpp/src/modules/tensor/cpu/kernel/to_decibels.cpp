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

RppStatus to_decibels_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptImagePatchPtr srcDims,
                                  Rpp32f cutOffDB,
                                  Rpp32f multiplier,
                                  Rpp32f referenceMagnitude,
                                  rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();

    // Calculate the intermediate values needed for DB conversion
    Rpp32f minRatio = std::pow(10, cutOffDB / multiplier);
    if(minRatio == 0.0f)
        minRatio = std::nextafter(0.0f, 1.0f);

    const Rpp32f log10Factor = 0.3010299956639812;      //1 / std::log(10);
    multiplier *= log10Factor;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrCurrent = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrCurrent = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u height = srcDims[batchCount].height;
        Rpp32u width = srcDims[batchCount].width;
        Rpp32f refMag = referenceMagnitude;

        // Compute maximum value in the input buffer
        if(!referenceMagnitude)
        {
            refMag = -std::numeric_limits<Rpp32f>::max();
            Rpp32f *srcPtrTemp = srcPtrCurrent;
            if(width == 1)
                refMag = std::max(refMag, *(std::max_element(srcPtrTemp, srcPtrTemp + height)));
            else
            {
                for(int i = 0; i < height; i++)
                {
                    refMag = std::max(refMag, *(std::max_element(srcPtrTemp, srcPtrTemp + width)));
                    srcPtrTemp += srcDescPtr->strides.hStride;
                }
            }
        }

        Rpp32f invReferenceMagnitude = (refMag) ? (1.f / refMag) : 1.0f;
        // Interpret as 1D array
        if(width == 1)
        {
            for(Rpp32s vectorLoopCount = 0; vectorLoopCount < height; vectorLoopCount++)
                *dstPtrCurrent++ = multiplier * std::log2(std::max(minRatio, (*srcPtrCurrent++) * invReferenceMagnitude));
        }
        else
        {
            for(int i = 0; i < height; i++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrCurrent;
                dstPtrRow = dstPtrCurrent;
                for(Rpp32s vectorLoopCount = 0; vectorLoopCount < width; vectorLoopCount++)
                    *dstPtrRow++ = multiplier * std::log2(std::max(minRatio, (*srcPtrRow++) * invReferenceMagnitude));

                srcPtrCurrent += srcDescPtr->strides.hStride;
                dstPtrCurrent += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
