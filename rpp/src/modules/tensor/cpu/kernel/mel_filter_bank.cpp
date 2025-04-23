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

RppStatus mel_filter_bank_host_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcDimsTensor,
                                      Rpp32f maxFreqVal,    // check unused
                                      Rpp32f minFreqVal,
                                      RpptMelScaleFormula melFormula,
                                      Rpp32s numFilter,
                                      Rpp32f sampleRate,
                                      bool normalize,
                                      rpp::Handle& handle)
{
    BaseMelScale *melScalePtr;
    switch(melFormula)
    {
        case RpptMelScaleFormula::HTK:
            melScalePtr = new HtkMelScale;
            break;
        case RpptMelScaleFormula::SLANEY:
        default:
            melScalePtr = new SlaneyMelScale();
            break;
    }
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u batchSize = srcDescPtr->n;
    Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;

    Rpp32f maxFreq = (maxFreqVal == 0) ? sampleRate / 2 : maxFreqVal;
    Rpp32f minFreq = minFreqVal;

    // Convert lower, higher frequencies to mel scale and find melStep
    Rpp64f melLow = melScalePtr->hz_to_mel(minFreq);
    Rpp64f melHigh = melScalePtr->hz_to_mel(maxFreq);
    Rpp64f melStep = (melHigh - melLow) / (numFilter + 1);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        // Extract nfft, number of Frames, numBins
        Rpp32s nfft = (srcDimsTensor[batchCount * 2] - 1) * 2;
        Rpp32s numBins = nfft / 2 + 1;
        Rpp32s numFrames = srcDimsTensor[batchCount * 2 + 1];

        // Find hzStep
        Rpp64f hzStep = static_cast<Rpp64f>(sampleRate) / nfft;
        Rpp64f invHzStep = 1.0 / hzStep;

        // Find fftBinStart and fftBinEnd
        Rpp32s fftBinStart = std::ceil(minFreq * invHzStep);
        Rpp32s fftBinEnd = std::ceil(maxFreq * invHzStep);
        fftBinEnd = std::min(fftBinEnd, numBins);

        // Set/Fill normFactors, weightsDown and intervals
        Rpp32f *normFactors = scratchMem + (batchCount * numFilter);
        std::fill(normFactors, normFactors + numFilter, 1.f);           // normFactors contain numFilter values of type float
        Rpp32f *weightsDown = scratchMem + (batchSize * numFilter) + (batchCount * numBins);
        memset(weightsDown, 0, sizeof(numBins * sizeof(Rpp32f)));       // weightsDown contain numBins values of type float
        Rpp32s *intervals = reinterpret_cast<Rpp32s *>(weightsDown + (batchSize * numBins));
        std::fill(intervals, intervals + numBins, -1);                  // intervals contain numBins values of type integer

        Rpp32s fftBin = fftBinStart;
        Rpp64f mel0 = melLow, mel1 = melLow + melStep;
        Rpp64f fIter = fftBin * hzStep;
        for (int interval = 0; interval < numFilter + 1; interval++, mel0 = mel1, mel1 += melStep)
        {
            Rpp64f f0 = melScalePtr->mel_to_hz(mel0);
            Rpp64f f1 = melScalePtr->mel_to_hz(interval == numFilter ? melHigh : mel1);
            Rpp64f slope = 1. / (f1 - f0);

            if (normalize && interval < numFilter)
            {
                Rpp64f f2 = melScalePtr->mel_to_hz(mel1 + melStep);
                normFactors[interval] = 2.0 / (f2 - f0);
            }

            for (; fftBin < fftBinEnd && fIter < f1; fftBin++, fIter = fftBin * hzStep)
            {
                weightsDown[fftBin] = (f1 - fIter) * slope;
                intervals[fftBin] = interval;
            }
        }

        Rpp32u maxFrames = std::min(static_cast<Rpp32u>(numFrames + 8), dstDescPtr->strides.hStride);
        Rpp32u maxAlignedLength = maxFrames & ~7;
        Rpp32u vectorIncrement = 8;

        // Set ROI values in dst buffer to 0.0
        for(int i = 0; i < numFilter; i++)
        {
            Rpp32f *dstPtrRow = dstPtrTemp + i * dstDescPtr->strides.hStride;
            Rpp32u vectorLoopCount = 0;
            for(; vectorLoopCount < maxAlignedLength; vectorLoopCount += 8)
            {
                _mm256_storeu_ps(dstPtrRow, avx_p0);
                dstPtrRow += 8;
            }
            for(; vectorLoopCount < maxFrames; vectorLoopCount++)
                *dstPtrRow++ = 0.0f;
        }

        Rpp32u alignedLength = numFrames & ~7;
        __m256 pSrc, pDst;
        Rpp32f *srcRowPtr = srcPtrTemp + fftBinStart * srcDescPtr->strides.hStride;
        for (int64_t fftBin = fftBinStart; fftBin < fftBinEnd; fftBin++)
        {
            auto filterUp = intervals[fftBin];
            auto weightUp = 1.0f - weightsDown[fftBin];
            auto filterDown = filterUp - 1;
            auto weightDown = weightsDown[fftBin];

            if (filterDown >= 0)
            {
                Rpp32f *dstRowPtrTemp = dstPtrTemp + filterDown * dstDescPtr->strides.hStride;
                Rpp32f *srcRowPtrTemp = srcRowPtr;

                if (normalize)
                    weightDown *= normFactors[filterDown];
                __m256 pWeightDown = _mm256_set1_ps(weightDown);

                int vectorLoopCount = 0;
                for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    pSrc = _mm256_loadu_ps(srcRowPtrTemp);
                    pSrc = _mm256_mul_ps(pSrc, pWeightDown);
                    pDst = _mm256_loadu_ps(dstRowPtrTemp);
                    pDst = _mm256_add_ps(pDst, pSrc);
                    _mm256_storeu_ps(dstRowPtrTemp, pDst);
                    dstRowPtrTemp += vectorIncrement;
                    srcRowPtrTemp += vectorIncrement;
                }

                for (; vectorLoopCount < numFrames; vectorLoopCount++)
                    (*dstRowPtrTemp++) += weightDown * (*srcRowPtrTemp++);
            }

            if (filterUp >= 0 && filterUp < numFilter)
            {
                Rpp32f *dstRowPtrTemp = dstPtrTemp + filterUp *  dstDescPtr->strides.hStride;
                Rpp32f *srcRowPtrTemp = srcRowPtr;

                if (normalize)
                    weightUp *= normFactors[filterUp];
                __m256 pWeightUp = _mm256_set1_ps(weightUp);

                int vectorLoopCount = 0;
                for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    pSrc = _mm256_loadu_ps(srcRowPtrTemp);
                    pSrc = _mm256_mul_ps(pSrc, pWeightUp);
                    pDst = _mm256_loadu_ps(dstRowPtrTemp);
                    pDst = _mm256_add_ps(pDst, pSrc);
                    _mm256_storeu_ps(dstRowPtrTemp, pDst);
                    dstRowPtrTemp += vectorIncrement;
                    srcRowPtrTemp += vectorIncrement;
                }

                for (; vectorLoopCount < numFrames; vectorLoopCount++)
                    (*dstRowPtrTemp++) += weightUp * (*srcRowPtrTemp++);
            }

            srcRowPtr += srcDescPtr->strides.hStride;
        }
    }
    delete melScalePtr;

    return RPP_SUCCESS;
}
