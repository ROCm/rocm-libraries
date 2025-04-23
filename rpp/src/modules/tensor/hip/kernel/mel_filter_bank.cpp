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

/*
This kernel transforms the 2D spectrogram output into a Mel-scaled output based on the number of filters (numFilter) and applies optional normalization.

Mel Filter Bank Transformation:

Input: A 2D spectrogram of dimensions (numBins, numTimeFrames), where numBins is the number of FFT frequency bins (typically nfft / 2 + 1), and numTimeFrames represents the temporal frames.
Output: A 2D Mel-scaled output of dimensions (numFilter, numTimeFrames), where numFilter is the number of desired Mel filter banks, each corresponding to a range of FFT frequency bins.

Key Parameters:
numFilter: Number of Mel filter banks.
normalize flag: Whether to apply normalization to the filter bank.
melFormula: Choice of Mel scale formula (HTK or Slaney).
maxFreq and minFreq: Frequency range for the Mel filter banks.

Preprocessing:
Before the kernel is launched, Three arrays are precomputed to store the filter intervals, normalization factors, and weights:

Compute Intervals:
For each Mel filter, compute the frequency intervals (start and end FFT bins) that the filter spans. This is based on the Mel scale conversion of frequency ranges and the relationship between FFT bin indices and actual frequencies.
            interval = ceil(f1 / hzStep),
where hzStep is the frequency of the FFT bins (based on the sample rate and nfft).

Compute Normalization Factors:
If normalize is enabled, compute normalization factors for each filter. This ensures that each filter captures a normalized energy from its frequency interval.
            normFactor = 2 / (f2 - f0),
where f0 and f2 are adjacent frequencies defining the boundaries of the filter.

Compute Filter Weights:
The weights applied to FFT bins in each interval are precomputed, separated into two phases: weights up and weights down.
Weights up increase linearly from the start of the interval to the center.
Weights down decrease linearly from the center of the interval to the end.
            weightsUp = (f1 - fftBinStart * hzStep) / (f1 - f0),
            weightsDown = (f1 - fIter) * slope,
Kernel Logic:
The kernel applies the Mel filter bank transformation to the spectrogram data for each time frame and each Mel filter.

Steps in Kernel:
In the first interval, the weights increase linearly from 0 to 1. Apply these weights up to the corresponding FFT bins and accumulate the results into the destination value dstVal.
            dstVal += srcVal * weightUp,
            where weightUp = (1.0 - weightDown).

In the second interval, the weights decrease linearly from 1 to 0. Apply these weights down to the FFT bins and accumulate the results into dstVal.
            dstVal += srcVal * weightDown,

Once both intervals have been processed, store the accumulated value dstVal in the output buffer for the current (Mel filter, time frame).
*/

__device__ __forceinline__ void compute_mel(float *srcPtr, int melBin, float *weightsDown, int *intervals, int2 fftStrides, float normFactor, float &dstVal)
{
    dstVal = 0;
    //start and end FFT bin indices for the current mel bin
    int fftbin = intervals[melBin];
    int fftBinEnd = intervals[melBin + 1];

    float *srcPtrTemp = srcPtr + fftbin * fftStrides.x + fftStrides.y;
    // Process the first interval of FFT bins, applying the weights up
    for (; fftbin < fftBinEnd; fftbin++, srcPtrTemp += fftStrides.x)
    {
        float weightUp = 1.0f - weightsDown[fftbin];
        weightUp *= normFactor;
        dstVal += *srcPtrTemp * weightUp;
    }

    fftBinEnd = intervals[melBin + 2];    // Update the end FFT bin index for the next interval
    srcPtrTemp = srcPtr + fftbin * fftStrides.x + fftStrides.y;

    // Process the second interval of FFT bins, applying the weights down
    for (; fftbin < fftBinEnd; fftbin++, srcPtrTemp += fftStrides.x)
    {
        float weightDown = weightsDown[fftbin];
        weightDown *= normFactor;
        dstVal += *srcPtrTemp * weightDown;
    }
}

__global__ void mel_filter_bank_tensor(float *srcPtr,
                                       uint2 srcStridesNH,
                                       float *dstPtr,
                                       uint2 dstStridesNH,
                                       int *srcDimsTensor,
                                       int numFilter,
                                       bool normalize,
                                       float *normFactors,
                                       float *weightsDown,
                                       int *intervals)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= srcDimsTensor[id_z * 2 + 1] || id_y >= numFilter)
        return;

    uint dstIdx = id_z * dstStridesNH.x + id_y * dstStridesNH.y + id_x;
    uint srcIdx = id_z * srcStridesNH.x;

    float normFactor = (normalize) ? normFactors[id_y] : 1;
    compute_mel(srcPtr + srcIdx, id_y, weightsDown, intervals, make_int2(srcStridesNH.y, id_x), normFactor, dstPtr[dstIdx]);
}

RppStatus hip_exec_mel_filter_bank_tensor(Rpp32f *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32s* srcDimsTensor,
                                          Rpp32f maxFreqVal,
                                          Rpp32f minFreqVal,
                                          RpptMelScaleFormula melFormula,
                                          Rpp32s numFilter,
                                          Rpp32f sampleRate,
                                          bool normalize,
                                          rpp::Handle& handle)
{
    // Create an instance of the MelScale class based on the chosen formula
    BaseMelScale *melScalePtr;
    switch (melFormula)
    {
        case RpptMelScaleFormula::HTK:
            melScalePtr = new HtkMelScale;
            break;
        case RpptMelScaleFormula::SLANEY:
        default:
            melScalePtr = new SlaneyMelScale();
            break;
    }

    Rpp32f maxFreq = (maxFreqVal == 0) ? sampleRate / 2 : maxFreqVal;
    Rpp32f minFreq = minFreqVal;

    // Convert the frequency range to Mel scale and compute Mel step size
    Rpp64f melLow = melScalePtr->hz_to_mel(minFreq);
    Rpp64f melHigh = melScalePtr->hz_to_mel(maxFreq);
    Rpp64f melStep = (melHigh - melLow) / (numFilter + 1);

    Rpp32f *scratchMem = handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem;
    Rpp32f *normFactors = scratchMem;
    Rpp32f *weightsDown = scratchMem + numFilter;
    Rpp32s *intervals = reinterpret_cast<Rpp32s *>(weightsDown + srcDescPtr->h);

    // parameters for FFT and frequency bins
    Rpp32s nfft = (srcDescPtr->h - 1) * 2;
    Rpp32s numBins = nfft / 2 + 1;
    Rpp64f hzStep = static_cast<Rpp64f>(sampleRate) / nfft;
    Rpp64f invHzStep = 1.0 / hzStep;

    // start and end bins for the Mel filter bank
    Rpp32s fftBinStart = std::ceil(minFreq * invHzStep);
    Rpp32s fftBinEnd = std::ceil(maxFreq * invHzStep);
    fftBinEnd = std::min(fftBinEnd, numBins);

    // Initialize arrays used for Mel filter bank computation
    std::fill(normFactors, normFactors + numFilter, 1.0f);
    memset(weightsDown, 0, sizeof(srcDescPtr->h * sizeof(Rpp32f)));
    std::fill(intervals, intervals + numFilter + 2, -1);

    // Compute Mel filter weights and intervals
    Rpp32s fftBin = fftBinStart;
    Rpp64f mel0 = melLow, mel1 = melLow + melStep;
    Rpp64f fIter = fftBin * hzStep;

    intervals[0] = fftBinStart;
    intervals[numFilter + 1] = fftBinEnd;

    for (int interval = 1, index = 0; index < numFilter + 1; interval++, index++, mel0 = mel1, mel1 += melStep)
    {
        Rpp64f f0 = melScalePtr->mel_to_hz(mel0);
        Rpp64f f1 = melScalePtr->mel_to_hz(index == numFilter ? melHigh : mel1);
        Rpp64f slope = 1.0 / (f1 - f0);
        intervals[interval] = std::ceil(f1 / hzStep);

        if (normalize && index < numFilter)
        {
            Rpp64f f2 = melScalePtr->mel_to_hz(mel1 + melStep);
            normFactors[index] = 2.0 / (f2 - f0);
        }

         // Compute weights for each filter bank
        for (; fftBin < fftBinEnd && fIter < f1; fftBin++, fIter = fftBin * hzStep) {
            weightsDown[fftBin] = (f1 - fIter) * slope;
        }
    }

    Rpp32s globalThreads_x = dstDescPtr->w;     // number of frequency bins (numBins)
    Rpp32s globalThreads_y = dstDescPtr->h;     // number of time frames
    Rpp32s globalThreads_z = dstDescPtr->n;     // batch size
    hipLaunchKernelGGL(mel_filter_bank_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                       dstPtr,
                       make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                       srcDimsTensor,
                       numFilter,
                       normalize,
                       normFactors,
                       weightsDown,
                       intervals);

    delete melScalePtr;
    return RPP_SUCCESS;
}
