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
#include "ffts.h"
#include "ffts_attributes.h"
#include <complex>

inline bool is_pow2(Rpp64s n) { return (n & (n-1)) == 0; }
inline bool can_use_real_impl(Rpp64s n) { return is_pow2(n); }
inline Rpp64s size_in_buf(Rpp64s n) { return can_use_real_impl(n) ? n : 2 * n; }
inline Rpp64s size_out_buf(Rpp64s n) { return can_use_real_impl(n) ? n + 2 : 2 * n; }

// Compute hanning window
inline void hann_window(Rpp32f *output, Rpp32s windowSize)
{
    Rpp64f a = (2.0 * M_PI) / windowSize;
    for (Rpp32s t = 0; t < windowSize; t++)
    {
        Rpp64f phase = a * (t + 0.5);
        output[t] = (0.5 * (1.0 - std::cos(phase)));
    }
}

// Compute number of spectrogram windows
inline Rpp32s get_num_windows(Rpp32s length, Rpp32s windowLength, Rpp32s windowStep, bool centerWindows)
{
    if (!centerWindows)
        length -= windowLength;
    return ((length / windowStep) + 1);
}

// Compute reflect start idx to pad
inline RPP_HOST_DEVICE Rpp32s get_idx_reflect(Rpp32s loc, Rpp32s minLoc, Rpp32s maxLoc)
{
    if (maxLoc - minLoc < 2)
        return maxLoc - 1;
    for (;;)
    {
        if (loc < minLoc)
            loc = 2 * minLoc - loc;
        else if (loc >= maxLoc)
            loc = 2 * maxLoc - 2 - loc;
        else
            break;
    }
    return loc;
}

RppStatus spectrogram_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcLengthTensor,
                                  bool centerWindows,
                                  bool reflectPadding,
                                  Rpp32f *windowFunction,
                                  Rpp32s nfft,
                                  Rpp32s power,
                                  Rpp32s windowLength,
                                  Rpp32s windowStep,
                                  rpp::Handle& handle)
{
    Rpp32s windowCenterOffset = 0;
    bool vertical = (dstDescPtr->layout == RpptLayout::NFT);
    if (centerWindows) windowCenterOffset = windowLength / 2;
    if (nfft == 0) nfft = windowLength;
    const Rpp32s numBins = nfft / 2 + 1;
    const Rpp32f mulFactor = (2.0 * M_PI) / nfft;
    const Rpp32u hStride = dstDescPtr->strides.hStride;
    const Rpp32s alignedNfftLength = nfft & ~7;
    const Rpp32s alignedNbinsLength = numBins & ~7;
    const Rpp32s alignedWindowLength = windowLength & ~7;
    const Rpp32s maxNumWindows = (vertical) ? dstDescPtr->w : dstDescPtr->h;
    const Rpp32u windowOutputStride = maxNumWindows * nfft;
    if (windowOutputStride > 99532800)
        return RPP_ERROR_OUT_OF_BOUND_SCRATCH_MEMORY_SIZE;

    bool useRealImpl = can_use_real_impl(nfft);
    const auto fftInSize = size_in_buf(nfft);
    const auto fftOutSize = size_out_buf(nfft);

    Rpp32f *windowFn = static_cast<Rpp32f *>(calloc(windowLength, sizeof(Rpp32f)));
    Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;

    // Generate hanning window
    if (windowFunction == NULL)
        hann_window(windowFn, windowLength);
    else
        memcpy(windowFn, windowFunction, windowLength * sizeof(Rpp32f));
    Rpp32u numThreads = handle.GetNumThreads();

    // Get windows output
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (Rpp32s batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32s bufferLength = srcLengthTensor[batchCount];
        Rpp32s numWindows = get_num_windows(bufferLength, windowLength, windowStep, centerWindows);
        Rpp32f *windowOutput = scratchMem + batchCount * windowOutputStride;
        std::fill_n(windowOutput, numWindows * nfft, 0);
        for (Rpp32s w = 0; w < numWindows; w++)
        {
            Rpp32s windowStart = w * windowStep - windowCenterOffset;
            Rpp32f *windowOutputTemp = windowOutput + (w * nfft);
            // Pad when either windowStart less than zero or length greater than input srclength
            if (windowStart < 0 || (windowStart + windowLength) > bufferLength)
            {
                for (Rpp32s t = 0; t < windowLength; t++)
                {
                    Rpp32s inIdx = windowStart + t;
                    if (reflectPadding)
                    {
                        inIdx = get_idx_reflect(inIdx, 0, bufferLength);
                        *windowOutputTemp++ = windowFn[t] * srcPtrTemp[inIdx];
                    }
                    else
                    {
                        if (inIdx >= 0 && inIdx < bufferLength)
                            *windowOutputTemp++ = windowFn[t] * srcPtrTemp[inIdx];
                        else
                            *windowOutputTemp++ = 0;
                    }
                }
            }
            else
            {
                Rpp32f *srcPtrWindowTemp = srcPtrTemp + windowStart;
                Rpp32f *windowFnTemp = windowFn;
                Rpp32s t = 0;
                for (; t < alignedWindowLength; t += 8)
                {
                    __m256 pSrc, pWindowFn;
                    pSrc = _mm256_loadu_ps(srcPtrWindowTemp);
                    pWindowFn = _mm256_loadu_ps(windowFnTemp);
                    pSrc = _mm256_mul_ps(pSrc, pWindowFn);
                    _mm256_storeu_ps(windowOutputTemp, pSrc);
                    srcPtrWindowTemp += 8;
                    windowFnTemp += 8;
                    windowOutputTemp += 8;
                }
                for (; t < windowLength; t++)
                    *windowOutputTemp++ = (*windowFnTemp++) * (*srcPtrWindowTemp++);
            }
        }

        // Generate FFT output
        ffts_plan_t *p;
        if(useRealImpl)
            p = ffts_init_1d_real(nfft, FFTS_FORWARD);
        else
            p = ffts_init_1d(nfft, FFTS_FORWARD);

        if (!p)
        {
            printf("FFT Plan is unsupported. Exiting the code\n");
            exit(0);
        }

        // Set temporary buffers to 0
        Rpp32f FFTS_ALIGN(32) *fftInBuf = static_cast<Rpp32f*>(_mm_malloc(fftInSize * sizeof(Rpp32f), 32)); // ffts requires 32-byte aligned memory
        Rpp32f FFTS_ALIGN(32) *fftOutBuf = static_cast<Rpp32f*>(_mm_malloc(fftOutSize * sizeof(Rpp32f), 32)); // ffts requires 32-byte aligned memory

        for (Rpp32s w = 0; w < numWindows; w++)
        {
            Rpp32f *dstPtrBinTemp = dstPtrTemp + (w * hStride);
            Rpp32f *windowOutputTemp = windowOutput + (w * nfft);
            for(int k = 0; k < fftInSize; k++)
                fftInBuf[k] = 0.0f;

            for(int k = 0; k < fftOutSize; k++)
                fftOutBuf[k] = 0.0f;

            Rpp32s inWindowStart = windowLength < nfft ? (nfft - windowLength) / 2 : 0;
            // Copy the window input to fftInBuf
            if (useRealImpl)
            {
                for (int i = 0; i < windowLength; i++)
                    fftInBuf[inWindowStart + i] = windowOutputTemp[i];
            }
            else
            {
                for (int i = 0; i < windowLength; i++)
                {
                    Rpp32s off = 2 * (inWindowStart + i);
                    fftInBuf[off] = windowOutputTemp[i];
                    fftInBuf[off + 1] = 0.0f;
                }
            }

            ffts_execute(p, fftInBuf, fftOutBuf);
            auto *complexFft = reinterpret_cast<std::complex<Rpp32f> *>(fftOutBuf);
            Rpp32s outIdx = w;
            if (vertical)
            {
                if (power == 1)
                {
                    for (int i = 0; i < numBins; i++, outIdx += hStride)
                        dstPtrTemp[outIdx] = std::abs(complexFft[i]);
                }
                else
                {
                    for (int i = 0; i < numBins; i++, outIdx += hStride)
                        dstPtrTemp[outIdx] = std::norm(complexFft[i]);
                }
            }
            else
            {
                if (power == 1)
                {
                    for (int i = 0; i < numBins; i++)
                        *dstPtrBinTemp++ = std::abs(complexFft[i]);
                }
                else
                {
                    for (int i = 0; i < numBins; i++)
                        *dstPtrBinTemp++ = std::norm(complexFft[i]);
                }
            }
        }
        ffts_free(p);
        _mm_free(fftInBuf);
        _mm_free(fftOutBuf);
    }
    if(windowFn)
        free(windowFn);
    return RPP_SUCCESS;
}
