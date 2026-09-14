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

#include <rocfft/rocfft.h>

#include "hip_tensor_executors.hpp"

/* Spectrogram kernel working overview
1D Input -> 2D Output
Output can be in 2 layouts
1. TF layout (Time-Frequency)
2. FT layout (Frequency-Time)

Input parameters
1. nfft
2. windowLength
3. windowStep
4. centerWindows
5. reflectPadding
6. windowFunction
7. power

For input audio sample of length N
      Spectrogram             TF layout
(N) --------------->  (numWindows, nfft / 2 + 1)

      Spectrogram             FT layout
(N) --------------->  (nfft / 2 + 1, numWindows)

Where
windowOffset = (centerWindows) ? (windowLength / 2) : 0;
numWindows = ((N - windowOffset) / windowStep) + 1

Spectrogram output computation is divided into 4 steps as below:
1. Compute window output. Applies a filter for a chunks of input to get the window output of shape
(numWindows, nfft)
2. Compute sine and cosine factors required (nfft, nfft / 2 + 1)
3. Do matrix muliplication to get final output of shape (numWindows, nfft / 2 + 1) for the TF
layout. For the FT layout is just transposed real part      - windowOutput (numWindows, nfft) .
cosFactor (nfft, nfft/2 + 1) imaginary part - windowOutput (numWindows, nfft) . sinFactor (nfft,
nfft/2 + 1)
4. Compute final result using the real and imaginary part */

// Compute hanning window
inline RPP_HOST_DEVICE void hann_window(Rpp32f* output, Rpp32s windowSize) {
    constexpr Rpp64f TWO_PI_VAL = 6.28318530717958647692;
    Rpp64f a = TWO_PI_VAL / windowSize;
    for (Rpp32s t = 0; t < windowSize; t++) {
        Rpp64f phase = a * (t + 0.5);
        output[t] = (0.5 * (1.0 - std::cos(phase)));
    }
}

// Compute number of spectrogram windows
inline RPP_HOST_DEVICE Rpp32s get_num_windows(Rpp32s length, Rpp32s windowLength, Rpp32s windowStep,
                                              bool centerWindows) {
    if (!centerWindows) length -= windowLength;
    return ((length / windowStep) + 1);
}

// Compute reflect start idx to pad
inline RPP_HOST_DEVICE Rpp32s get_idx_reflect(Rpp32s loc, Rpp32s minLoc, Rpp32s maxLoc) {
    if (maxLoc - minLoc < 2) return maxLoc - 1;
    for (;;) {
        if (loc < minLoc)
            loc = 2 * minLoc - loc;
        else if (loc >= maxLoc)
            loc = 2 * maxLoc - 2 - loc;
        else
            break;
    }
    return loc;
}

// -------------------- Set 0 -  spectrogram hip kernels --------------------

// compute magnitude from rocFFT complex output with shared memory transpose for vertical layout
__global__ void compute_magnitude_from_complex_hip_tensor(float2* srcPtr, uint srcStride,
                                                          float* dstPtr, uint2 dstStrideNH,
                                                          int* numWindowsTensor, int2 params_i2,
                                                          bool vertical) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int numWindows = numWindowsTensor[id_z];
    int numBins = params_i2.x;
    int power = params_i2.y;

    if (!vertical) {
        // Non-vertical path: direct coalesced write (TF layout: [numWindows, numBins])
        if ((id_y >= numWindows) || (id_x >= numBins)) return;

        int srcIdx = id_z * srcStride + id_y * numBins + id_x;
        float2 complexVal = srcPtr[srcIdx];
        float magnitudeSquare = (complexVal.x * complexVal.x) + (complexVal.y * complexVal.y);

        int dstIdx = id_z * dstStrideNH.x + id_y * dstStrideNH.y + id_x;
        dstPtr[dstIdx] = (power == 2) ? magnitudeSquare : sqrtf(magnitudeSquare);
    } else {
        // Vertical path: use shared memory transpose for coalesced writes (FT layout: [numBins,
        // numWindows])
        constexpr int MAG_TILE_DIM =
            (LOCAL_THREADS_X > LOCAL_THREADS_Y) ? LOCAL_THREADS_X : LOCAL_THREADS_Y;
        __shared__ float magnitude_smem[MAG_TILE_DIM][MAG_TILE_DIM];

        // Load and compute magnitude in coalesced fashion
        // Read: srcPtr[batch][window][bin] - threads read consecutive bins (coalesced)
        if ((id_y < numWindows) && (id_x < numBins)) {
            int srcIdx = id_z * srcStride + id_y * numBins + id_x;
            float2 complexVal = srcPtr[srcIdx];
            float magnitudeSquare = (complexVal.x * complexVal.x) + (complexVal.y * complexVal.y);
            magnitude_smem[hipThreadIdx_y][hipThreadIdx_x] =
                (power == 2) ? magnitudeSquare : sqrtf(magnitudeSquare);
        } else {
            magnitude_smem[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;
        }
        __syncthreads();

        // Transpose indices for output
        // Original position: (id_y, id_x) = (window, bin)
        // Transposed position: (id_x, id_y) = (bin, window)
        int out_row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_y;  // bin dimension
        int out_col = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_x;  // window dimension

        // Write transposed data in coalesced fashion
        // Write: dstPtr[batch][bin][window] - threads write consecutive windows (coalesced)
        if ((out_row < numBins) && (out_col < numWindows)) {
            int dstIdx = id_z * dstStrideNH.x + out_row * dstStrideNH.y + out_col;
            dstPtr[dstIdx] = magnitude_smem[hipThreadIdx_x][hipThreadIdx_y];
        }
    }
}

// compute window output by applying hanning window
__global__ void window_output_hip_tensor(float* srcPtr, uint srcStride, float* dstPtr,
                                         uint dstStride, float* windowFn, int* srcLengthTensor,
                                         int* numWindowsTensor, int4 params_i4,
                                         bool reflectPadding) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int srcLength = srcLengthTensor[id_z];
    int numWindows = numWindowsTensor[id_z];
    int nfft = params_i4.x;
    int windowLength = params_i4.y;
    int windowStep = params_i4.z;
    int windowCenterOffset = params_i4.w;

    if ((id_x >= windowLength) || (id_y >= numWindows)) return;

    int dstIdx = id_z * dstStride + id_y * nfft + id_x;
    int srcIdx = id_z * srcStride;
    int windowStart = id_y * windowStep - windowCenterOffset;
    int inIdx = windowStart + id_x;

    // check if windowStart is beyond the bounds of input
    if ((windowStart < 0) || ((windowStart + windowLength) > srcLength)) {
        if (reflectPadding) {
            inIdx = get_idx_reflect(inIdx, 0, srcLength);
            dstPtr[dstIdx] = windowFn[id_x] * srcPtr[srcIdx + inIdx];
        } else if ((inIdx >= 0) && (inIdx < srcLength))
            dstPtr[dstIdx] = windowFn[id_x] * srcPtr[srcIdx + inIdx];
    } else {
        dstPtr[dstIdx] = windowFn[id_x] * srcPtr[srcIdx + inIdx];
    }
}

// -------------------- Set 1 - kernel executor --------------------

RppStatus hip_exec_spectrogram_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                      RpptDescPtr dstDescPtr, Rpp32s* srcLengthTensor,
                                      bool centerWindows, bool reflectPadding,
                                      Rpp32f* windowFunction, Rpp32s nfft, Rpp32s power,
                                      Rpp32s windowLength, Rpp32s windowStep, rpp::Handle& handle) {
    bool vertical = (dstDescPtr->layout == RpptLayout::NFT);
    if (!nfft) nfft = windowLength;  // Apply default before computing numBins
    Rpp32s numBins = (nfft / 2 + 1);

    Rpp32s maxNumWindows = (vertical) ? dstDescPtr->w : dstDescPtr->h;
    size_t windowOutputStride = static_cast<size_t>(maxNumWindows) * static_cast<size_t>(nfft);
    size_t fftOutputStride = static_cast<size_t>(maxNumWindows) * static_cast<size_t>(numBins);
    size_t windowOutputFloats = static_cast<size_t>(dstDescPtr->n) * windowOutputStride;
    // Align fftOutput to 8 bytes (float2) accounting for total offset from base pointer.
    // fftOutput is placed after windowOutput, and we need (windowLength + alignedOffset) to be
    // even.
    size_t totalBaseOffset = static_cast<size_t>(windowLength) + windowOutputFloats;
    size_t alignedOffset = windowOutputFloats + (totalBaseOffset & 1);  // add 1 if odd
    size_t rocfftScratchSize = static_cast<size_t>(windowLength) + alignedOffset +
                               static_cast<size_t>(dstDescPtr->n) * fftOutputStride * 2;
    if (rocfftScratchSize > static_cast<size_t>(SPECTROGRAM_MAX_SCRATCH_MEMORY))
        return RPP_ERROR_OUT_OF_BOUND_SCRATCH_MEMORY_SIZE;

    // Ensure scratch buffer is large enough for audio workload (lazily reallocated if needed)
    RppStatus scratchStatus = handle.EnsureAudioScratchBuffer(rocfftScratchSize);
    if (scratchStatus != RPP_SUCCESS) return scratchStatus;

    // Generate hanning window
    Rpp32f* windowFn;
    if (windowFunction == NULL) {
        windowFn = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;
        hann_window(windowFn, windowLength);
    } else {
        windowFn = windowFunction;
    }

    // Copy the hanning window values to HIP memory
    Rpp32f* d_windowFn = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(d_windowFn, windowFn, windowLength * sizeof(Rpp32f),
                                           hipMemcpyHostToDevice, handle.GetStream()));

    // Compute the number of windows required for each input in the batch
    Rpp32s* numWindowsTensor =
        reinterpret_cast<Rpp32s*>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
    for (Rpp32u i = 0; i < dstDescPtr->n; i++)
        numWindowsTensor[i] =
            get_num_windows(srcLengthTensor[i], windowLength, windowStep, centerWindows);

    Rpp32s windowCenterOffset = (centerWindows) ? (windowLength / 2) : 0;

    // Allocate window output buffer (after d_windowFn)
    Rpp32f* windowOutput = d_windowFn + windowLength;
    RPP_HIP_RETURN_IF_ERROR(hipMemsetAsync(
        windowOutput, 0, windowOutputStride * dstDescPtr->n * sizeof(Rpp32f), handle.GetStream()));

    // Compute the windowOutput for all samples in a batch. Each sample will be of shape
    // (numWindows, nfft)
    Rpp32s globalThreads_x = windowLength;
    Rpp32s globalThreads_y = maxNumWindows;
    Rpp32s globalThreads_z = dstDescPtr->n;
    hipLaunchKernelGGL(window_output_hip_tensor,
                       dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                            ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                            ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                       handle.GetStream(), srcPtr, srcDescPtr->strides.nStride, windowOutput,
                       windowOutputStride, d_windowFn, srcLengthTensor, numWindowsTensor,
                       make_int4(nfft, windowLength, windowStep, windowCenterOffset),
                       reflectPadding);
    HIP_CHECK_LAUNCH_RETURN();

    // Allocate complex output buffer for rocFFT (after windowOutput, with 8-byte alignment for
    // float2)
    float2* fftOutput = reinterpret_cast<float2*>(windowOutput + alignedOffset);

    // Get or create cached rocFFT plan for this (nfft, totalWindows) combination
    int totalWindows = maxNumWindows * dstDescPtr->n;
    rocfft_plan plan = nullptr;
    rocfft_plan_description desc = nullptr;
    RppStatus status = rpp::get_rocfft_plan(handle, nfft, totalWindows, &plan, &desc);
    if (status != RPP_SUCCESS) return status;

    // Get work buffer size
    size_t workBufferSize = 0;
    if (rocfft_plan_get_work_buffer_size(plan, &workBufferSize) != rocfft_status_success)
        return RPP_ERROR;

    // Create execution info and set stream unconditionally (required for correct
    // synchronization)
    void* workBuffer = nullptr;
    rocfft_execution_info execInfo = nullptr;
    if (rocfft_execution_info_create(&execInfo) != rocfft_status_success)
        return RPP_ERROR_NOT_ENOUGH_MEMORY;

    if (rocfft_execution_info_set_stream(execInfo, handle.GetStream()) != rocfft_status_success) {
        rocfft_execution_info_destroy(execInfo);
        return RPP_ERROR_HIP_RUNTIME;
    }

    // Allocate work buffer if needed
    if (workBufferSize > 0) {
        if (hipMalloc(&workBuffer, workBufferSize) != hipSuccess ||
            rocfft_execution_info_set_work_buffer(execInfo, workBuffer, workBufferSize) !=
                rocfft_status_success) {
            if (workBuffer) (void)hipFree(workBuffer);
            rocfft_execution_info_destroy(execInfo);
            return RPP_ERROR_NOT_ENOUGH_MEMORY;  // rocFFT work buffer allocation failed
        }
    }

    // Execute rocFFT for the entire batch (plan includes correct batch count). This enqueues
    // work on handle.GetStream() using workBuffer, so neither workBuffer nor execInfo may be
    // released until the stream has completed. From here every exit path must fall through to
    // the shared cleanup below (synchronize, then free) rather than returning early.
    void* inBuffers[1] = {windowOutput};
    void* outBuffers[1] = {fftOutput};
    RppStatus retStatus = RPP_SUCCESS;
    if (rocfft_execute(plan, inBuffers, outBuffers, execInfo) != rocfft_status_success) {
        retStatus = RPP_ERROR_HIP_RUNTIME;  // rocFFT execution failed
    } else {
        // Compute magnitude from complex FFT output
        // For NTF (vertical=false): stride.hStride = width (numBins), for NFT (vertical=true):
        // maxNumWindows
        Rpp32u dstHStride = vertical ? maxNumWindows : dstDescPtr->strides.hStride;
        globalThreads_x = numBins;
        hipLaunchKernelGGL(compute_magnitude_from_complex_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), fftOutput, fftOutputStride, dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstHStride), numWindowsTensor,
                           make_int2(numBins, power), vertical);
        if (hipGetLastError() != hipSuccess) retStatus = RPP_ERROR_HIP_LAUNCH;
    }

    // Synchronize before releasing rocFFT resources so the stream is no longer using
    // workBuffer, then clean up temporary resources (plan is cached and reused).
    hipError_t syncErr = hipStreamSynchronize(handle.GetStream());
    if (workBuffer) (void)hipFree(workBuffer);
    if (execInfo) rocfft_execution_info_destroy(execInfo);

    if (retStatus != RPP_SUCCESS) return retStatus;
    if (syncErr != hipSuccess) return RPP_ERROR_HIP_RUNTIME;
    return RPP_SUCCESS;
}
