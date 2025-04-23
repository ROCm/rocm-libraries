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
1. Compute window output. Applies a filter for a chunks of input to get the window output of shape (numWindows, nfft)
2. Compute sine and cosine factors required (nfft, nfft / 2 + 1)
3. Do matrix muliplication to get final output of shape (numWindows, nfft / 2 + 1) for the TF layout. For the FT layout is just transposed
real part      - windowOutput (numWindows, nfft) . cosFactor (nfft, nfft/2 + 1)
imaginary part - windowOutput (numWindows, nfft) . sinFactor (nfft, nfft/2 + 1)
4. Compute final result using the real and imaginary part */

// Compute hanning window
inline RPP_HOST_DEVICE void hann_window(Rpp32f *output, Rpp32s windowSize)
{
    Rpp64f a = (2.0 * M_PI) / windowSize;
    for (Rpp32s t = 0; t < windowSize; t++)
    {
        Rpp64f phase = a * (t + 0.5);
        output[t] = (0.5 * (1.0 - std::cos(phase)));
    }
}

// Compute number of spectrogram windows
inline RPP_HOST_DEVICE Rpp32s get_num_windows(Rpp32s length, Rpp32s windowLength, Rpp32s windowStep, bool centerWindows)
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

// -------------------- Set 0 -  spectrogram hip kernels --------------------

// compute window output by applying hanning window
__global__ void window_output_hip_tensor(float *srcPtr,
                                         uint srcStride,
                                         float *dstPtr,
                                         uint dstStride,
                                         float *windowFn,
                                         int *srcLengthTensor,
                                         int *numWindowsTensor,
                                         int4 params_i4,
                                         bool reflectPadding)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int srcLength = srcLengthTensor[id_z];
    int numWindows = numWindowsTensor[id_z];
    int nfft = params_i4.x;
    int windowLength = params_i4.y;
    int windowStep = params_i4.z;
    int windowCenterOffset = params_i4.w;

    if ((id_x >= windowLength) || (id_y >= numWindows))
        return;

    int dstIdx = id_z * dstStride + id_y * nfft + id_x;
    int srcIdx = id_z * srcStride;
    int windowStart = id_y * windowStep - windowCenterOffset;
    int inIdx = windowStart + id_x;

    // check if windowStart is beyond the bounds of input
    if ((windowStart < 0) || ((windowStart + windowLength) > srcLength))
    {
        if (reflectPadding)
        {
            inIdx = get_idx_reflect(inIdx, 0, srcLength);
            dstPtr[dstIdx] = windowFn[id_x] * srcPtr[srcIdx + inIdx];
        }
        else if ((inIdx >= 0) && (inIdx < srcLength))
            dstPtr[dstIdx] = windowFn[id_x] * srcPtr[srcIdx + inIdx];
    }
    else
    {
        dstPtr[dstIdx] = windowFn[id_x] * srcPtr[srcIdx + inIdx];
    }
}

// compute factors required for fourier transform
__global__ void compute_coefficients_hip_tensor(float *cosFactor,
                                                float *sinFactor,
                                                int numBins,
                                                int nfft)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if ((id_y >= numBins) || (id_x >= nfft))
        return;

    float factor = id_x * ((TWO_PI * id_y) / nfft);
    int dstIdx = id_x * numBins + id_y;
    cosFactor[dstIdx] = cosf(factor);
    sinFactor[dstIdx] = sinf(factor);
}

/* compute fourier transform on windowed output
   it internally computes a matrix multiplication of
   - windowOutput of size (numWindows, nfft) with cosFactor of size (nfft, nfft/2 + 1) for real part of output
   - windowOutput of size (numWindows, nfft) with sinFactor of size (nfft, nfft/2 + 1) for imaginary part of output */
__global__ void fourier_transform_hip_tensor(float *srcPtr,
                                             uint2 srcStrideNH,
                                             float *dstPtr,
                                             uint2 dstStrideNH,
                                             int *numWindowsTensor,
                                             float *cosFactor,
                                             float *sinFactor,
                                             int4 params_i4,
                                             bool vertical)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int numWindows = numWindowsTensor[id_z];
    int nfft = params_i4.x;
    int numBins = params_i4.y;
    int power = params_i4.z;
    int numTiles = params_i4.w;

    __shared__ float input_smem[16][16];        // 16 rows of src, 16 cols of src in a 16 x 16 thread block
    __shared__ float cosFactor_smem[16][16];    // 16 rows of cosFactor, 16 cols of cosFactor in a 16 x 16 thread block
    __shared__ float sinFactor_smem[16][16];    // 16 rows of sinFactor, 16 cols of sinFactor in a 16 x 16 thread block
    input_smem[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;      // initialization of shared memory to 0 using all 16 x 16 threads
    cosFactor_smem[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;  // initialization of shared memory to 0 using all 16 x 16 threads
    sinFactor_smem[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;  // initialization of shared memory to 0 using all 16 x 16 threads
    __syncthreads();

    int srcIdx = id_z * srcStrideNH.x + id_y * srcStrideNH.y;
    float realVal = 0.0f, imaginaryVal = 0.0f;
    for(int t = 0, offset = 0; t < numTiles; t++, offset += hipBlockDim_x)
    {
        // load input values to shared memory if (id_y, srcCol) < (numWindows, nfft) - range of input
        int srcCol = (offset + hipThreadIdx_x);
        int factorRow = (offset + hipThreadIdx_y);
        if ((id_y < numWindows) && (srcCol < nfft))
            input_smem[hipThreadIdx_y][hipThreadIdx_x] = srcPtr[srcIdx + srcCol];

        // load cosfactor and sinfactor values to shared memory if (factorRow, id_x) < (nfft, numBins) - range of sinFactor and cosFactor
        if ((factorRow < nfft) && (id_x < numBins))
        {
            factorRow *= numBins;
            cosFactor_smem[hipThreadIdx_y][hipThreadIdx_x] = cosFactor[factorRow + id_x];
            sinFactor_smem[hipThreadIdx_y][hipThreadIdx_x] = sinFactor[factorRow + id_x];
        }

        // wait for all threads to load input, cosFactor and sinFactor values to shared memory
        __syncthreads();

        // do matrix multiplication on the small matrix
        for (int j = 0; j < hipBlockDim_x; j++)
        {
            realVal += (input_smem[hipThreadIdx_y][j] * cosFactor_smem[j][hipThreadIdx_x]);
            imaginaryVal += (-input_smem[hipThreadIdx_y][j] * sinFactor_smem[j][hipThreadIdx_x]);
        }
        __syncthreads();
    }

    // final store to dst
    if ((id_y < numWindows) && (id_x < numBins))
    {
        float magnitudeSquare = ((realVal * realVal) + (imaginaryVal * imaginaryVal));

        /* if vertical is set to true, then get the transposed output index (id_x * dstStrideNH.y + id_y)
           else get the normal output index (id_y * dstStrideNH.y + id_x) */
        int dstIdx = (vertical) ? (id_z * dstStrideNH.x + id_x * dstStrideNH.y + id_y) :
                                  (id_z * dstStrideNH.x + id_y * dstStrideNH.y + id_x);
        dstPtr[dstIdx] = (power == 2) ? magnitudeSquare : sqrtf(magnitudeSquare);
    }
}

// -------------------- Set 1 - kernel executor --------------------

RppStatus hip_exec_spectrogram_tensor(Rpp32f* srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f* dstPtr,
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
    bool vertical = (dstDescPtr->layout == RpptLayout::NFT);
    Rpp32s numBins = (nfft / 2 + 1);

    // find the maximum windows required across all inputs in batch and stride required for window output
    Rpp32s maxNumWindows = (vertical) ? dstDescPtr->w : dstDescPtr->h;
    uint scratchMemorySize = nfft * ((maxNumWindows * dstDescPtr->n) + (numBins * 2));

    // check if scratch memory size required for spectrogram is within the limits
    if (scratchMemorySize > SPECTROGRAM_MAX_SCRATCH_MEMORY)
        return RPP_ERROR_OUT_OF_BOUND_SCRATCH_MEMORY_SIZE;

    // generate hanning window
    Rpp32f *windowFn;
    if (windowFunction == NULL)
    {
        windowFn = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;
        hann_window(windowFn, windowLength);
    }
    else
    {
        windowFn = windowFunction;
    }

    // copy the hanning window values to hip memory
    Rpp32f *d_windowFn = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    CHECK_RETURN_STATUS(hipMemcpyAsync(d_windowFn, windowFn, windowLength * sizeof(Rpp32f), hipMemcpyHostToDevice, handle.GetStream()));

    // compute the number of windows required for each input in the batch
    Rpp32s *numWindowsTensor = reinterpret_cast<Rpp32s*>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
    for (Rpp32u i = 0; i < dstDescPtr->n; i++)
        numWindowsTensor[i] = get_num_windows(srcLengthTensor[i], windowLength, windowStep, centerWindows);

    Rpp32s windowCenterOffset = (centerWindows) ? (windowLength / 2) : 0;
    if (!nfft) nfft = windowLength;
    Rpp32u windowOutputStride = maxNumWindows * nfft;

    Rpp32f *windowOutput = d_windowFn + windowLength;
    CHECK_RETURN_STATUS(hipMemsetAsync(windowOutput, 0, windowOutputStride * dstDescPtr->n * sizeof(Rpp32f), handle.GetStream()));

    // compute the windowOutput for all samples in a batch. Each sample will be of shape (numWindows, nfft)
    Rpp32s globalThreads_x = windowLength;
    Rpp32s globalThreads_y = maxNumWindows;
    Rpp32s globalThreads_z = dstDescPtr->n;
    hipLaunchKernelGGL(window_output_hip_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       srcDescPtr->strides.nStride,
                       windowOutput,
                       windowOutputStride,
                       d_windowFn,
                       srcLengthTensor,
                       numWindowsTensor,
                       make_int4(nfft, windowLength, windowStep, windowCenterOffset),
                       reflectPadding);

    // compute the sin and cos factors required for FFT
    Rpp32f *cosTensor, *sinTensor;
    cosTensor = windowOutput + dstDescPtr->n * windowOutputStride;
    sinTensor = cosTensor + (nfft * numBins);
    hipLaunchKernelGGL(compute_coefficients_hip_tensor,
                       dim3(ceil((float)nfft/LOCAL_THREADS_X), ceil((float)numBins/LOCAL_THREADS_Y), ceil((float)1/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       cosTensor,
                       sinTensor,
                       numBins,
                       nfft);

    // compute the final output
    globalThreads_x = numBins;
    Rpp32s numTiles = static_cast<int>(ceil((static_cast<float>(nfft) / LOCAL_THREADS_X)));
    hipLaunchKernelGGL(fourier_transform_hip_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       windowOutput,
                       make_uint2(windowOutputStride, nfft),
                       dstPtr,
                       make_uint2(dstDescPtr->strides.nStride, maxNumWindows),
                       numWindowsTensor,
                       cosTensor,
                       sinTensor,
                       make_int4(nfft, numBins, power, numTiles),
                       vertical);
    CHECK_RETURN_STATUS(hipStreamSynchronize(handle.GetStream()));

    return RPP_SUCCESS;
}
