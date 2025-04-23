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

// -------------------- Set 0 - resample kernel device helpers  --------------------

__device__ __forceinline__ float resample_hip_compute(float &x, float &scale, float &center, float *lookup, int &lookupSize)
{
    float locRaw = x * scale + center;
    int locFloor = std::floor(locRaw);
    float weight = locRaw - locFloor;
    locFloor = std::max(std::min(locFloor, lookupSize - 2), 0);
    float current = lookup[locFloor];
    float next = lookup[locFloor + 1];
    return current + weight * (next - current);
}

__device__ __forceinline__ void resample_hip_compute(float4 *src_f4, float4 *dst_f4, const float4 *scale_f4, const float4 *center_f4, float *lookup)
{
    float4 locRaw_f4 = (*src_f4) * (*scale_f4) + (*center_f4);
    int4 locFloor_i4;
    FLOOR4(locRaw_f4, locFloor_i4);
    float4 weight_f4 = make_float4(locRaw_f4.x - locFloor_i4.x, locRaw_f4.y - locFloor_i4.y, locRaw_f4.z - locFloor_i4.z, locRaw_f4.w - locFloor_i4.w);
    float4 current_f4 = make_float4(lookup[locFloor_i4.x], lookup[locFloor_i4.y], lookup[locFloor_i4.z], lookup[locFloor_i4.w]);
    float4 next_f4 = make_float4(lookup[locFloor_i4.x + 1], lookup[locFloor_i4.y + 1], lookup[locFloor_i4.z + 1], lookup[locFloor_i4.w + 1]);
    *dst_f4 = current_f4 + weight_f4 * (next_f4 - current_f4);
}

// -------------------- Set 1 - resample kernel host helpers  --------------------

inline void compute_output_dims(Rpp32f *inRateTensor,
                                Rpp32f *outRateTensor,
                                Rpp32s *srcLengthTensor,
                                Rpp32s *dstLengthTensor,
                                Rpp32u batchSize)
{
    for (Rpp32s i = 0, j = 0; i < batchSize; i++, j += 2)
    {
        dstLengthTensor[j] = std::ceil(srcLengthTensor[j] * outRateTensor[i] / inRateTensor[i]);
        dstLengthTensor[j + 1] = srcLengthTensor[j + 1];
    }
}

// -------------------- Set 2 - resample kernels --------------------

__global__ void resample_single_channel_hip_tensor(float *srcPtr,
                                                   float *dstPtr,
                                                   uint2 strides,
                                                   int2 *srcDimsTensor,
                                                   int2 *dstDimsTensor,
                                                   float *inRateTensor,
                                                   float *outRateTensor,
                                                   RpptResamplingWindow *window)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int srcLength = srcDimsTensor[id_z].x;
    int dstLength = dstDimsTensor[id_z].x;
    int outBlock = id_x * hipBlockDim_x;
    int blockEnd = std::min(outBlock + static_cast<int>(hipBlockDim_x), dstLength);

    if (dstLength != srcLength)
    {
        double scale = static_cast<double>(inRateTensor[id_z]) / outRateTensor[id_z];
        extern __shared__ float lookup_smem[];

        // copy all values from window lookup table to shared memory lookup table
        for (int k = hipThreadIdx_x; k < window->lookupSize; k += hipBlockDim_x)
            lookup_smem[k] = window->lookupPinned[k];
        __syncthreads();

        if (outBlock >= dstLength)
            return;

        // extract the window scale, center and lookup size values from window
        float windowScale = window->scale;
        float windowCenter = window->center;
        int lookupSize = window->lookupSize;
        float4 windowScale_f4 = static_cast<float4>(windowScale);
        float4 windowCenter_f4 = static_cast<float4>(windowCenter);
        float4 increment_f4 = static_cast<float4>(8.0f);
        d_float8 locInit_f8;
        locInit_f8.f4[0] = make_float4(0, 1, 2, 3);
        locInit_f8.f4[1] = make_float4(4, 5, 6, 7);

        // compute block wise values required for processing
        double inBlockRaw = outBlock * scale;
        int inBlockRounded = static_cast<int>(inBlockRaw);
        float inPos = inBlockRaw - inBlockRounded;
        float fscale = scale;
        uint dstIdx = id_z * strides.y + outBlock;
        float *inBlockPtr = srcPtr + id_z * strides.x + inBlockRounded;

        // process block size (256) elements in single thread
        for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale, dstIdx++)
        {
            int loc0, loc1;
            window->input_range(inPos, &loc0, &loc1);

            // check if computed loc0, loc1 values are beyond the input dimensions and update accordingly
            if (loc0 + inBlockRounded < 0)
                loc0 = -inBlockRounded;
            if (loc1 + inBlockRounded > srcLength)
                loc1 = srcLength - inBlockRounded;
            int locInWindow = loc0;
            float locBegin = locInWindow - inPos;
            float accum = 0.0f;

            d_float8 locInWindow_f8, accum_f8;
            locInWindow_f8.f4[0] = static_cast<float4>(locBegin) + locInit_f8.f4[0];
            locInWindow_f8.f4[1] = static_cast<float4>(locBegin) + locInit_f8.f4[1];
            accum_f8.f4[0] = static_cast<float4>(0.0f);
            accum_f8.f4[1] = static_cast<float4>(0.0f);
            for (; locInWindow + 7 < loc1; locInWindow += 8)
            {
                d_float8 weights_f8;
                resample_hip_compute(&locInWindow_f8.f4[0], &weights_f8.f4[0], &windowScale_f4, &windowCenter_f4, lookup_smem);
                resample_hip_compute(&locInWindow_f8.f4[1], &weights_f8.f4[1], &windowScale_f4, &windowCenter_f4, lookup_smem);

                d_float8 src_f8;
                rpp_hip_load8_and_unpack_to_float8(inBlockPtr + locInWindow, &src_f8);
                accum_f8.f4[0] +=  src_f8.f4[0] * weights_f8.f4[0];
                accum_f8.f4[1] +=  src_f8.f4[1] * weights_f8.f4[1];

                locInWindow_f8.f4[0] += increment_f4;
                locInWindow_f8.f4[1] += increment_f4;
            }
            accum_f8.f4[0] += accum_f8.f4[1];
            accum += (accum_f8.f1[0] + accum_f8.f1[1] + accum_f8.f1[2] + accum_f8.f1[3]);   // perform small work of reducing float4 to float

            float x = locInWindow - inPos;
            for (; locInWindow < loc1; locInWindow++, x++)
            {
                float w = resample_hip_compute(x, windowScale, windowCenter, lookup_smem, lookupSize);
                accum += inBlockPtr[locInWindow] * w;
            }

            // Final store to dst
            dstPtr[dstIdx] = accum;
        }
    }
    // copy input to output if dstLength is same as srcLength
    else
    {
        if (outBlock >= dstLength)
            return;

        uint srcIdx = id_z * strides.x + outBlock;
        uint dstIdx = id_z * strides.y + outBlock;
        for (int outPos = outBlock; outPos < blockEnd; outPos++, dstIdx++, srcIdx++)
            dstPtr[dstIdx] = srcPtr[srcIdx];
    }
}

__global__ void resample_multi_channel_hip_tensor(float *srcPtr,
                                                  float *dstPtr,
                                                  uint2 strides,
                                                  int2 *srcDimsTensor,
                                                  int2 *dstDimsTensor,
                                                  float *inRateTensor,
                                                  float *outRateTensor,
                                                  RpptResamplingWindow *window)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int srcLength = srcDimsTensor[id_z].x;
    int numChannels = srcDimsTensor[id_z].y;
    int dstLength = dstDimsTensor[id_z].x;
    int outBlock = id_x * hipBlockDim_x;
    int blockEnd = std::min(outBlock + static_cast<int>(hipBlockDim_x), dstLength);

    if (dstLength != srcLength)
    {
        double scale = static_cast<double>(inRateTensor[id_z]) / outRateTensor[id_z];
        extern __shared__ float lookup_smem[];

        // copy all values from window lookup table to shared memory lookup table
        for (int k = hipThreadIdx_x; k < window->lookupSize; k += hipBlockDim_x)
            lookup_smem[k] = window->lookupPinned[k];
        __syncthreads();

        if (outBlock >= dstLength)
            return;

        // extract the window scale, center and lookup size values from window
        float windowScale = window->scale;
        float windowCenter = window->center;
        int lookupSize = window->lookupSize;

        // compute block wise values required for processing
        double inBlockRaw = outBlock * scale;
        int inBlockRounded = static_cast<int>(inBlockRaw);
        float inPos = inBlockRaw - inBlockRounded;
        float fscale = scale;
        uint dstIdx = id_z * strides.y + outBlock * numChannels;
        float *inBlockPtr = srcPtr + id_z * strides.x + (inBlockRounded * numChannels);

        // process block size * channels (256 * channels) elements in single thread
        for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale, dstIdx += numChannels)
        {
            int loc0, loc1;
            window->input_range(inPos, &loc0, &loc1);

            // check if computed loc0, loc1 values are beyond the input dimensions and update accordingly
            if (loc0 + inBlockRounded < 0)
                loc0 = -inBlockRounded;
            if (loc1 + inBlockRounded > srcLength)
                loc1 = srcLength - inBlockRounded;
            float locInWindow = loc0 - inPos;
            int2 offsetLocs_i2 = make_int2(loc0, loc1) * static_cast<int2>(numChannels);    // offsetted loc0, loc1 values for multi channel case

            float accum[RPPT_MAX_AUDIO_CHANNELS] = {0.0f};
            for (int offsetLoc = offsetLocs_i2.x; offsetLoc < offsetLocs_i2.y; offsetLoc += numChannels, locInWindow++)
            {
                float w = resample_hip_compute(locInWindow, windowScale, windowCenter, lookup_smem, lookupSize);
                for (int c = 0; c < numChannels; c++)
                    accum[c] += inBlockPtr[offsetLoc + c] * w;
            }

            // Final store to dst
            for (int c = 0; c < numChannels; c++)
                dstPtr[dstIdx + c] = accum[c];
        }
    }
    else
    {
        if (outBlock >= dstLength)
            return;

        uint srcIdx = id_z * strides.x + outBlock * numChannels;
        uint dstIdx = id_z * strides.y + outBlock * numChannels;
        for (int outPos = outBlock; outPos < blockEnd; outPos++, dstIdx += numChannels, srcIdx += numChannels)
        {
            for (int c = 0; c < numChannels; c++)
                dstPtr[dstIdx + c] = srcPtr[srcIdx + c];
        }
    }
}

// -------------------- Set 3 - resample kernels executor --------------------

RppStatus hip_exec_resample_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f *inRateTensor,
                                   Rpp32f *outRateTensor,
                                   Rpp32s *srcDimsTensor,
                                   RpptResamplingWindow &window,
                                   rpp::Handle& handle)
{
    Rpp32s globalThreads_x = dstDescPtr->strides.hStride;
    Rpp32s globalThreads_y = 1;
    Rpp32s globalThreads_z = dstDescPtr->n;
    Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    RppSize_t sharedMemorySizeInBytes = (window.lookupSize * sizeof(Rpp32f)); // shared memory size needed for resample kernel

    // using the input sampling rate, output sampling rate compute the output dims
    Rpp32s *dstDimsTensor = reinterpret_cast<Rpp32s *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
    compute_output_dims(inRateTensor, outRateTensor, srcDimsTensor, dstDimsTensor, dstDescPtr->n);

    // For 1D audio tensors (channels = 1)
    if (tensorDims == 1)
    {
        hipLaunchKernelGGL(resample_single_channel_hip_tensor,
                           dim3(ceil((Rpp32f)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((Rpp32f)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((Rpp32f)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                           dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           sharedMemorySizeInBytes,
                           handle.GetStream(),
                           srcPtr,
                           dstPtr,
                           make_uint2(srcDescPtr->strides.nStride, dstDescPtr->strides.nStride),
                           reinterpret_cast<int2 *>(srcDimsTensor),
                           reinterpret_cast<int2 *>(dstDimsTensor),
                           inRateTensor,
                           outRateTensor,
                           &window);
    }
    // For 2D audio tensors (channels > 1)
    else if (tensorDims == 2)
    {
        hipLaunchKernelGGL(resample_multi_channel_hip_tensor,
                           dim3(ceil((Rpp32f)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((Rpp32f)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((Rpp32f)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                           dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           sharedMemorySizeInBytes,
                           handle.GetStream(),
                           srcPtr,
                           dstPtr,
                           make_uint2(srcDescPtr->strides.nStride, dstDescPtr->strides.nStride),
                           reinterpret_cast<int2 *>(srcDimsTensor),
                           reinterpret_cast<int2 *>(dstDimsTensor),
                           inRateTensor,
                           outRateTensor,
                           &window);
    }

    return RPP_SUCCESS;
}
