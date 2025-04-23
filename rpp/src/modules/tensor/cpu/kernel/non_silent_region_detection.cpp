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

/* Non Silent Region Detection requires Moving Mean Square (MMS) computation on input audio data
MMS buffer is a 1D buffer having same length as input audio. The algorithm used for MMS computation is explained with a sample use case

Example:
Input: [1, 2, 3, 4, 5, 6, 7, 8]
audio_length = 8
window_length = 3
reset_interval_length = 4

window_begin = -window_length + 1 = -2
window_factor = 1 / window_length = 1/3

MMS computation is divided into blocks of reset interval length
num_blocks = audio_length / reset_interval_length
For the above example we will have
    - 2 blocks (8 / 4)
    - each block runs for 4 iterations
    - in each iteration window begin value is increment by 1

Block1
window begin = -2
Iteration 0:    sum_of_squares = 1*1                              // window begin = -2
                store sum_of_squares * window_factor in MMS[0]

Iteration 1:    sum_of_squares = 1*1 + 2*2                        // window begin = -1
                store sum_of_squares * window_factor in MMS[1]

Iteration 2:    sum_of_squares = 1*1 + 2*2 + 3*3                  // window begin =  0
                store sum_of_squares * window_factor in MMS[2]
                sum_of_squares -= 1*1

Iteration 3:    sum_of_squares = 2*2 + 3*3 + 4*4                  // window begin =  1
                store sum_of_squares * window_factor in MMS[3]
                sum_of_squares -= 2*2

Block2
Iteration 0:    sum_of_squares = 3*3 + 4*4 + 5*5                  // window begin = 2
                store sum_of_squares * window_factor in MMS[4]
                sum_of_squares -= 3*3

Iteration 1:    sum_of_squares = 4*4 + 5*5 + 6*6                 // window begin = 3
                store sum_of_squares * window_factor in MMS[5]
                sum_of_squares -= 4*4

Iteration 2:    sum_of_squares = 5*5 + 6*6 + 7*7                 // window begin = 4
                store sum_of_squares * window_factor in MMS[6]
                sum_of_squares -= 5*5

Iteration 3:    sum_of_squares  = 6*6 + 7*7 + 8*8                // window begin = 5
                store sum_of_squares * window_factor in MMS[7]
                sum_of_squares -= 6*6

For computing beginning index and length of Non Silent Region in audio data we traverse over
the entire MMS buffer and compare these values with the calculated cutoff value
    - For beginning index, traverse over MMS buffer from 0 to audio_length - 1 and compare if any value
      is greater than or equal to cutoff value. if yes, that is the beginning index
    - For length, traverse over MMS buffer from audio_length - 1 to beginning index and compare if any value
      is greater than or equal to cutoff value. if yes, that is the ending index of Non Silent Region. From this
      data compute length with the formulae, length = ending index - beginning index + 1
*/

#include "host_tensor_executors.hpp"

Rpp32f getSquare(Rpp32f &value)
{
    return (value * value);
}

RppStatus non_silent_region_detection_host_tensor(Rpp32f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp32s *srcLengthTensor,
                                                  Rpp32s *detectedIndexTensor,
                                                  Rpp32s *detectionLengthTensor,
                                                  Rpp32f cutOffDB,
                                                  Rpp32s windowLength,
                                                  Rpp32f referencePower,
                                                  Rpp32s resetInterval,
                                                  rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    const Rpp32f cutOff = std::pow(10.0f, cutOffDB * 0.1f);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32s srcLength = srcLengthTensor[batchCount];

        // mmsBuffer length is equal to input audio length and can vary dynamically for each input in a batch
        // preallocating a static buffer for entire batchsize will be too big, so allocate mmsBuffer for each sample dynamically
        Rpp32f *mmsBuffer = new Rpp32f[srcLength]{};
        bool referenceMax = (referencePower == 0.0f);

        // set reset interval based on the user input
        Rpp32s resetLength = (resetInterval == -1) ? srcLength : resetInterval;

        // calculate moving mean square of input
        Rpp32f meanFactor = 1.0f / windowLength;
        Rpp32s windowBegin = -windowLength + 1;
        for (Rpp32s outPos = 0; outPos < srcLength;)
        {
            // reset the sumOfSquares values to 0 and recompute the starting value required for next block
            Rpp32f sumOfSquares = 0.0f;
            for (Rpp32s i = std::max<Rpp32s>(windowBegin, 0); i < outPos; i++)
                sumOfSquares += getSquare(srcPtrTemp[i]);

            Rpp32s intervalEndIdx = std::min<Rpp32s>(srcLength, outPos + resetLength);
            for (; outPos < intervalEndIdx; outPos++, windowBegin++)
            {
                sumOfSquares += getSquare(srcPtrTemp[outPos]);
                mmsBuffer[outPos] = sumOfSquares * meanFactor;
                if (windowBegin >= 0)
                    sumOfSquares -= getSquare(srcPtrTemp[windowBegin]);
            }
        }

        // convert cutoff from DB to magnitude
        Rpp32f base = (referenceMax) ? *std::max_element(mmsBuffer, mmsBuffer + srcLength) : referencePower;
        Rpp32f cutOffMag = base * cutOff;

        // calculate begining index, length of non silent region from the mms buffer
        Rpp32s endIdx = srcLength;
        Rpp32s beginIdx = endIdx;
        Rpp32s detectBegin, detectEnd;
        for(int i = 0; i < endIdx; i++)
        {
            if(mmsBuffer[i] >= cutOffMag)
            {
                beginIdx = i;
                break;
            }
        }
        if(beginIdx == endIdx)
        {
            detectBegin = 0;
            detectEnd = 0;
        }
        else
        {
            for(int i = endIdx - 1; i >= beginIdx; i--)
            {
                if(mmsBuffer[i] >= cutOffMag)
                {
                    endIdx = i;
                    break;
                }
            }
            detectBegin = beginIdx;
            detectEnd = endIdx - beginIdx + 1;
        }

        // if both starting index and length of nonsilent region is not 0
        // adjust the values as per the windowLength
        if(detectBegin != 0 && detectEnd != 0)
        {
            Rpp32s newBegin = std::max<Rpp32s>(detectBegin - (windowLength - 1), 0);
            detectEnd += detectBegin - newBegin;
            detectBegin = newBegin;
        }

        detectedIndexTensor[batchCount] = detectBegin;
        detectionLengthTensor[batchCount] = detectEnd;
        delete[] mmsBuffer;
    }
    return RPP_SUCCESS;
}
