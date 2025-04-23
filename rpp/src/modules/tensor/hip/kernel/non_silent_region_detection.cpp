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
#include "rpp_hip_math.hpp"

// -------------------- Set 0 -  moving mean square kernel device helpers --------------------

// calculate the position in shared memory to avoid bank conflicts
__host__ __device__ __forceinline__ int compute_pos_in_smem(int pos)
{
    return pos + (pos >> 5); // since shared memory banks considered is 32
}

/* compute prefix sum on the input buffer passed
   prefix sum of an array is an array where each element is the sum of all previous elements in the input array, inclusive of the current element */
__device__ __forceinline__ void compute_prefix_sum(float *input, uint bufferLength)
{
    int offset = 1;
    int2 offset_i2 = static_cast<int2>(offset);
    int2 offsetAB_i2 = make_int2(offset - 1, 2 * offset - 1);
    int threadIdxMul2 = 2 * hipThreadIdx_x;
    int blockDimMul2 = 2 * hipBlockDim_x;

    /* compute intermediate prefix sums in a up sweep manner
       (each level in the hierarchy doubles the distance between the pairs of elements being added) */
    for (int d = bufferLength >> 1; d > 0; d >>= 1)
    {
        // syncthreads before proceeding to next iteration
        __syncthreads();
        int dMul2 = 2 * d;
        for (int idxMul2 = threadIdxMul2; idxMul2 < dMul2; idxMul2 += blockDimMul2)
        {
            int2 pos_i2 = (offset_i2 * static_cast<int2>(idxMul2)) + offsetAB_i2;
            input[compute_pos_in_smem(pos_i2.y)] += input[compute_pos_in_smem(pos_i2.x)];
        }
        offset <<= 1;
        offset_i2 =  static_cast<int2>(offset);
        offsetAB_i2 = make_int2(offset - 1, 2 * offset - 1);
    }

    if (hipThreadIdx_x == 0)
    {
        int last = bufferLength - 1;
        input[compute_pos_in_smem(last)] = 0;
    }

    /* compute final prefix sums in a down sweep manner
       (each level in the hierarchy halves the distance between the pairs of elements being added) */
    for (int d = 1; d < bufferLength; d <<= 1)
    {
        offset >>= 1;
        offset_i2 = static_cast<int2>(offset);
        offsetAB_i2 = make_int2(offset - 1, 2 * offset - 1);
        __syncthreads();
        // syncthreads before proceeding to next iteration

        int dMul2 = 2 * d;
        for (int idxMul2 = threadIdxMul2; idxMul2 < dMul2; idxMul2 += blockDimMul2)
        {
            int2 pos_i2 = offset_i2 * static_cast<int2>(idxMul2) + offsetAB_i2;
            int posA = compute_pos_in_smem(pos_i2.x);
            int posB = compute_pos_in_smem(pos_i2.y);
            float t = input[posA];
            input[posA] = input[posB];
            input[posB] += t;
        }
    }
    __syncthreads();
}

// -------------------- Set 1 -  moving mean square compute kernel --------------------

__global__ void moving_mean_square_hip_tensor(float *srcPtr,
                                              uint nStride,
                                              float *mmsArr,
                                              int *srcLengthTensor,
                                              int outputTileLength,
                                              int windowLength,
                                              float windowFactor,
                                              int inputTileLength)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    uint srcLength = srcLengthTensor[id_z];
    uint batchStride = id_z * nStride;
    int blockStart = hipBlockIdx_x * outputTileLength;

    if (blockStart >= srcLength)
        return;

    float *input = srcPtr + batchStride;
    extern __shared__ float squaredPrefixSum_smem[];

    float *inBlockPtr = srcPtr + batchStride + blockStart;
    float *outBlockPtr = mmsArr + batchStride + blockStart;

    // find the valid output tile length values needed for given block
    int validOutputTileLength = std::min<int>(outputTileLength, srcLength - blockStart);

    // assign pointers that points to block begin and block end locations
    float *extendedBlockStart = inBlockPtr - windowLength;
    float *extendedBlockEnd = inBlockPtr + validOutputTileLength;

    // load input data to shared memory
    for(int pos = hipThreadIdx_x; pos < inputTileLength; pos += hipBlockDim_x)
    {
        float val = 0.0f;
        auto extendedBlockPtr = extendedBlockStart + pos;

        /* check if extendedBlockPtr is within the valid region of input
           and load the value from extendedBlockPtr if it is within valid region */
        if (extendedBlockPtr >= input && extendedBlockPtr < extendedBlockEnd)
            val = *extendedBlockPtr;
        squaredPrefixSum_smem[compute_pos_in_smem(pos)] = val * val;
    }

    // compute prefix sum
    compute_prefix_sum(squaredPrefixSum_smem, inputTileLength);

    // compute the mms value here
    for(int pos = hipThreadIdx_x; pos < validOutputTileLength; pos += hipBlockDim_x)
        outBlockPtr[pos] = windowFactor * ((inBlockPtr[pos] * inBlockPtr[pos]) + squaredPrefixSum_smem[compute_pos_in_smem(windowLength + pos)] - squaredPrefixSum_smem[compute_pos_in_smem(pos + 1)]);
}

// -------------------- Set 2 -  kernels for finding cutoffmag value  --------------------

__global__ void max_reduction_hip_tensor(float *srcPtr,
                                         uint nStride,
                                         float *maxArr,
                                         int *srcLengthTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    uint srcLength = srcLengthTensor[id_z];

    uint srcIdx = id_z * nStride;
    __shared__ float max_smem[256];                                     // 256 values of src in a 256 x 1 thread block
    max_smem[hipThreadIdx_x] = srcPtr[srcIdx];                          // initialization of LDS to start value using all 256 threads

    if (id_x >= srcLength)
        return;

    if (id_x + 8 > srcLength)
        id_x -= (id_x + 8 - srcLength);

    srcIdx += id_x;
    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);       // load 8 pixels to local memory
    rpp_hip_math_max8(&src_f8, &max_smem[hipThreadIdx_x]);
    __syncthreads();                                                    // syncthreads after max compute

    // Reduction of 256 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            max_smem[hipThreadIdx_x] = fmaxf(max_smem[hipThreadIdx_x], max_smem[hipThreadIdx_x + threadMax]);
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int dstIdx = id_z * hipGridDim_x + hipBlockIdx_x;
        maxArr[dstIdx] = max_smem[0];
    }
}

__global__ void cutoffmag_hip_tensor(float *srcPtr,
                                     int maxLength,
                                     float *cutOffMagPtr,
                                     float cutOff,
                                     float referencePower,
                                     bool referenceMax)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // if referenceMax is set to true, perform final max reduction on srcPtr and compute cutOffMag
    if(referenceMax)
    {
        uint srcIdx = id_z * maxLength;
        __shared__ float max_smem[256];                                     // 256 values of src in a 256 x 1 thread block
        max_smem[hipThreadIdx_x] = srcPtr[srcIdx];                          // initialization of LDS to start value using all 256 threads

        if (id_x >= maxLength)
            return;

        srcIdx += id_x;
        float maxVal = srcPtr[srcIdx];
        while (id_x < maxLength)
        {
            maxVal = fmaxf(maxVal, srcPtr[srcIdx]);
            id_x += hipBlockDim_x;
            srcIdx += hipBlockDim_x;
        }
        max_smem[hipThreadIdx_x] = maxVal;
        __syncthreads();                                                    // syncthreads after max compute

        // Reduction of 256 floats on 256 threads per block in x dimension
        for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
        {
            if (hipThreadIdx_x < threadMax)
                max_smem[hipThreadIdx_x] = max(max_smem[hipThreadIdx_x], max_smem[hipThreadIdx_x + threadMax]);
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_x == 0)
            cutOffMagPtr[id_z] = max_smem[0] * cutOff;
    }
    else
    {
        if (hipThreadIdx_x == 0)
            cutOffMagPtr[id_z] = referencePower * cutOff;
    }
}

// -------------------- Set 3 -  kernels for finding begin and length of NSR in inputs --------------------

__global__ void find_region_hip_tensor(float *srcPtr,
                                       uint nStride,
                                       int *beginTensor,
                                       int *lengthTensor,
                                       float *cutOffMagPtr,
                                       int *srcLengthTensor,
                                       float windowLength)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    uint srcLength = srcLengthTensor[id_z];
    float cutOffMag = cutOffMagPtr[id_z];

    __shared__ int beginResult;
    __shared__ int endResult;
    beginResult = srcLength;
    endResult = 0;
    __syncthreads();

    int beginIdx = srcLength;
    int endIdx = 0;
    uint stridePerSample = id_z * nStride;

    // Find the begin index in src whose value is >= cutOffMag
    for (int i = id_x; i < srcLength; i += hipBlockDim_x)
    {
        uint srcIdx = stridePerSample + i;
        if (srcPtr[srcIdx] >= cutOffMag)
        {
            beginIdx = i;
            atomicMin(&beginResult, beginIdx);
            if(beginResult != srcLength)
                break;
        }
    }

    // Find the end index in src whose value is >= cutOffMag
    for (int i = id_x; i < srcLength; i += hipBlockDim_x)
    {
        uint srcIdx = stridePerSample + srcLength - 1 - i;
        if (srcPtr[srcIdx] >= cutOffMag)
        {
            endIdx = srcLength - 1 - i;
            atomicMax(&endResult, endIdx);
            if(endResult != 0)
                break;
        }
    }

    // Final store to dst
    if(hipThreadIdx_x == 0)
    {
        if(beginResult == srcLength || endResult == 0)
        {
            beginTensor[id_z] = 0;
            lengthTensor[id_z] = 0;
        }
        else
        {
            int detectBegin = beginResult;
            int detectEnd = endResult - beginResult + 1;

            // if both starting index and length of nonsilent region is not 0
            // adjust the values as per the windowLength
            if(detectBegin != 0 && detectEnd != 0)
            {
                int newBegin = max(detectBegin - (windowLength - 1), 0);
                detectEnd += detectBegin - newBegin;
                detectBegin = newBegin;
            }
            beginTensor[id_z] = detectBegin;
            lengthTensor[id_z] = detectEnd;
        }
    }
}

// -------------------- Set 4 -  host helpers for kernel executor --------------------

// return the nearest previous power of 2 for the given number
inline Rpp32s prev_pow2(Rpp32s n)
{
    Rpp32s pow2 = 1;
    while (n - pow2 > pow2)
        pow2 += pow2;

    return pow2;
}

// return the nearest next power of 2 for the given number
inline Rpp32s next_pow2(Rpp32s n)
{
    Rpp32s pow2 = 1;
    while (n > pow2)
        pow2 += pow2;

    return pow2;
}

// -------------------- Set 5 - non silent region kernels executor --------------------

RppStatus hip_exec_non_silent_region_detection_tensor(Rpp32f *srcPtr,
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
    // check if scratch memory size required for moving mean square is within the limits
    if ((srcDescPtr->n * srcDescPtr->strides.nStride) > MMS_MAX_SCRATCH_MEMORY)
        return RPP_ERROR_OUT_OF_BOUND_SCRATCH_MEMORY_SIZE;

    Rpp32f *mmsArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    Rpp32s maxSharedMemoryInBytes = handle.GetLocalMemorySize();
    Rpp32s maxSharedMemoryElements = maxSharedMemoryInBytes / sizeof(Rpp32f);
    Rpp32s kSharedMemBanks = 32;
    Rpp32s inputTileLength = prev_pow2(maxSharedMemoryElements * kSharedMemBanks / (kSharedMemBanks + 1));

    if (resetInterval > 0 && resetInterval < inputTileLength)
    {
        Rpp32s p = prev_pow2(resetInterval);
        Rpp32s n = next_pow2(resetInterval);
        if (p > windowLength)
            inputTileLength = p;
        else if (n < inputTileLength)
            inputTileLength = n;
    }

    Rpp32s sharedMemorySizeInBytes = compute_pos_in_smem(inputTileLength) * sizeof(Rpp32f);
    Rpp32s outputTileLength = inputTileLength - windowLength;
    Rpp32f windowFactor = 1.0f / windowLength;

    if (outputTileLength <= 0)
        return RPP_ERROR_INVALID_OUTPUT_TILE_LENGTH;

    if (sharedMemorySizeInBytes > maxSharedMemoryInBytes)
        return RPP_ERROR_OUT_OF_BOUND_SHARED_MEMORY_SIZE;

    // launch kernel to compute the values needed for MMS Array
    Rpp32s globalThreads_x = ceil(static_cast<Rpp32f>(srcDescPtr->strides.nStride) / outputTileLength);
    Rpp32s globalThreads_y = 1;
    Rpp32s globalThreads_z = srcDescPtr->n;

    hipLaunchKernelGGL(moving_mean_square_hip_tensor,
                       dim3(globalThreads_x, globalThreads_y, globalThreads_z),
                       dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       sharedMemorySizeInBytes,
                       handle.GetStream(),
                       srcPtr,
                       srcDescPtr->strides.nStride,
                       mmsArr,
                       srcLengthTensor,
                       outputTileLength,
                       windowLength,
                       windowFactor,
                       inputTileLength);

    const Rpp32f cutOff = std::pow(10.0f, cutOffDB * 0.1f);
    bool referenceMax = (!referencePower);
    Rpp32f *partialMaxArr = mmsArr + srcDescPtr->n * srcDescPtr->strides.nStride;

    Rpp32s numBlocksPerSample = ceil(static_cast<Rpp32f>(srcDescPtr->strides.nStride) / (LOCAL_THREADS_X_1DIM * 8));
    Rpp32s cutOffMagKernelBlockSize = 1;
    if (referenceMax)
    {
        // compute max value in MMS buffer
        hipLaunchKernelGGL(max_reduction_hip_tensor,
                           dim3(numBlocksPerSample, 1, globalThreads_z),
                           dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           0,
                           handle.GetStream(),
                           mmsArr,
                           srcDescPtr->strides.nStride,
                           partialMaxArr,
                           srcLengthTensor);
        cutOffMagKernelBlockSize = 256;
    }
    // find the cutoff value in magnitude
    Rpp32f *cutOffMagPtr = partialMaxArr + globalThreads_z * numBlocksPerSample;
    hipLaunchKernelGGL(cutoffmag_hip_tensor,
                       dim3(1, 1, globalThreads_z),
                       dim3(cutOffMagKernelBlockSize, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       partialMaxArr,
                       numBlocksPerSample,
                       cutOffMagPtr,
                       cutOff,
                       referencePower,
                       referenceMax);

    // find the begin and length values of NSR in inputs
    hipLaunchKernelGGL(find_region_hip_tensor,
                       dim3(1, 1, globalThreads_z),
                       dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       mmsArr,
                       srcDescPtr->strides.nStride,
                       detectedIndexTensor,
                       detectionLengthTensor,
                       cutOffMagPtr,
                       srcLengthTensor,
                       windowLength);
    return RPP_SUCCESS;
}
