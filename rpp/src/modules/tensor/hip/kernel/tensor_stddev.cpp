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
#include "kernel_dims.hpp"

// ----------------------- Helper Functions --------------------------

__device__ __forceinline__ void stddev_hip_compute(uchar *srcPtr, float *src, float *dst, int numValues) { *dst =  sqrt(*src / numValues); }
__device__ __forceinline__ void stddev_hip_compute(float *srcPtr, float *src, float *dst, int numValues) { *dst =  sqrt(*src / numValues) * 255; }
__device__ __forceinline__ void stddev_hip_compute(signed char *srcPtr, float *src, float *dst, int numValues) { *dst = sqrt(*src / numValues); }
__device__ __forceinline__ void stddev_hip_compute(half *srcPtr, float *src, float *dst, int numValues) { *dst = sqrt(*src / numValues) * 255; }

__device__ __forceinline__ void mean_subtracted_square_3channel_hip_compute(d_float24 *src_f24, d_float24 *dst_f24,
                                                                            float4 &meanR_f4, float4 &meanG_f4, float4 &meanB_f4)
{
    rpp_hip_math_subtract8_const(&src_f24->f8[0], &dst_f24->f8[0], meanR_f4);
    rpp_hip_math_subtract8_const(&src_f24->f8[1], &dst_f24->f8[1], meanG_f4);
    rpp_hip_math_subtract8_const(&src_f24->f8[2], &dst_f24->f8[2], meanB_f4);
    rpp_hip_math_multiply8(&dst_f24->f8[0], &dst_f24->f8[0], &dst_f24->f8[0]);
    rpp_hip_math_multiply8(&dst_f24->f8[1], &dst_f24->f8[1], &dst_f24->f8[1]);
    rpp_hip_math_multiply8(&dst_f24->f8[2], &dst_f24->f8[2], &dst_f24->f8[2]);
}

// perform reduction on shared memory and store the result in output
__device__ __forceinline__ void reduce_variance_3channel_hip(d_float24 *tempChannelSquared_f24, d_float24 *tempSquared_f24,
                                                             float *partialRVarianceRowPtr_smem, float *partialGVarianceRowPtr_smem, float *partialBVarianceRowPtr_smem,
                                                             float *partialTensorVarianceRowPtr_smem, float *dstPtr)
{
    // channel wise addition
    tempChannelSquared_f24->f8[0].f4[0] += tempChannelSquared_f24->f8[0].f4[1];                 // perform small work of vectorized float4 addition
    tempChannelSquared_f24->f8[1].f4[0] += tempChannelSquared_f24->f8[1].f4[1];
    tempChannelSquared_f24->f8[2].f4[0] += tempChannelSquared_f24->f8[2].f4[1];

    partialRVarianceRowPtr_smem[hipThreadIdx_x] = (tempChannelSquared_f24->f8[0].f1[0] +
                                                   tempChannelSquared_f24->f8[0].f1[1] +
                                                   tempChannelSquared_f24->f8[0].f1[2] +
                                                   tempChannelSquared_f24->f8[0].f1[3]);        // perform small work of reducing R float4s to float using 16 x 16 threads and store in _smem
    partialGVarianceRowPtr_smem[hipThreadIdx_x] = (tempChannelSquared_f24->f8[1].f1[0] +
                                                   tempChannelSquared_f24->f8[1].f1[1] +
                                                   tempChannelSquared_f24->f8[1].f1[2] +
                                                   tempChannelSquared_f24->f8[1].f1[3]);        // perform small work of reducing G float4s to float using 16 x 16 threads and store in _smem
    partialBVarianceRowPtr_smem[hipThreadIdx_x] = (tempChannelSquared_f24->f8[2].f1[0] +
                                                   tempChannelSquared_f24->f8[2].f1[1] +
                                                   tempChannelSquared_f24->f8[2].f1[2] +
                                                   tempChannelSquared_f24->f8[2].f1[3]);        // perform small work of reducing B float4s to float using 16 x 16 threads and store in _smem

    // tensor wise addition
    tempSquared_f24->f8[0].f4[0] += tempSquared_f24->f8[0].f4[1];                               // perform small work of vectorized float4 addition
    tempSquared_f24->f8[1].f4[0] += tempSquared_f24->f8[1].f4[1];
    tempSquared_f24->f8[2].f4[0] += tempSquared_f24->f8[2].f4[1];
    tempSquared_f24->f8[0].f4[0] += tempSquared_f24->f8[1].f4[0];
    tempSquared_f24->f8[0].f4[0] += tempSquared_f24->f8[2].f4[0];

    partialTensorVarianceRowPtr_smem[hipThreadIdx_x] = (tempSquared_f24->f8[0].f1[0] +
                                                        tempSquared_f24->f8[0].f1[1] +
                                                        tempSquared_f24->f8[0].f1[2] +
                                                        tempSquared_f24->f8[0].f1[3]);          // perform small work of reducing B float4s to float using 16 x 16 threads and store in _smem
    __syncthreads();                                                                            // syncthreads after _smem load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRVarianceRowPtr_smem[hipThreadIdx_x] += partialRVarianceRowPtr_smem[hipThreadIdx_x + threadMax];
            partialGVarianceRowPtr_smem[hipThreadIdx_x] += partialGVarianceRowPtr_smem[hipThreadIdx_x + threadMax];
            partialBVarianceRowPtr_smem[hipThreadIdx_x] += partialBVarianceRowPtr_smem[hipThreadIdx_x + threadMax];
            partialTensorVarianceRowPtr_smem[hipThreadIdx_x] += partialTensorVarianceRowPtr_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
            {
                partialRVarianceRowPtr_smem[0] += partialRVarianceRowPtr_smem[increment];
                partialGVarianceRowPtr_smem[0] += partialGVarianceRowPtr_smem[increment];
                partialBVarianceRowPtr_smem[0] += partialBVarianceRowPtr_smem[increment];
                partialTensorVarianceRowPtr_smem[0] += partialTensorVarianceRowPtr_smem[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 4;
            dstPtr[idx] = partialRVarianceRowPtr_smem[0];
            dstPtr[idx + 1] = partialGVarianceRowPtr_smem[0];
            dstPtr[idx + 2] = partialBVarianceRowPtr_smem[0];
            dstPtr[idx + 3] = partialTensorVarianceRowPtr_smem[0];
        }
    }
}

// -------------------- Set 0 - Reduction Stage 2 --------------------

template <typename T>
__global__ void tensor_stddev_grid_result_hip(T *inputSrcPtr,
                                              float *srcPtr,
                                              uint xBufferLength,
                                              float *dstPtr,
                                              RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialVariance_smem[1024];                    // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    partialVariance_smem[hipThreadIdx_x] = 0.0f;                    // initialization of _smem to 0 using all 1024 x 1 threads

    if (id_x >= xBufferLength)
        return;

    int xDiff = xBufferLength - (xBufferLength & ~7);               // difference between roiWidth and alignedLength, where alignedLength = roiWidth & ~7
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local memory
    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                    // local memory reset of invalid values (from the vectorized global load) to 0.0f
    src_f8.f4[0] += src_f8.f4[1];                                   // perform small work of vectorized float4 addition
    partialVariance_smem[hipThreadIdx_x] += (src_f8.f1[0] +
                                             src_f8.f1[1] +
                                             src_f8.f1[2] +
                                             src_f8.f1[3]);         // perform small work of reducing float4s to float using 1024 x 1 threads and store in _smem
    __syncthreads();                                                // syncthreads after _smem load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialVariance_smem[hipThreadIdx_x] += partialVariance_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[id_z].xywhROI.roiHeight * roiTensorPtrSrc[id_z].xywhROI.roiWidth;
        stddev_hip_compute(inputSrcPtr, &partialVariance_smem[0] , &dstPtr[id_z], totalElements);
    }
}

template <typename T>
__global__ void tensor_stddev_grid_3channel_result_hip(T *inputSrcPtr,
                                                       float *srcPtr,
                                                       uint xBufferLength,
                                                       float *dstPtr,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x;
    int id_z = hipBlockIdx_z;

    /* Stores individual channel Variations computed from channel Means to compute Stddev of individual channels*/
    __shared__ float partialRVariance_smem[1024];                                     // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    __shared__ float partialGVariance_smem[1024];
    __shared__ float partialBVariance_smem[1024];
    __shared__ float partialTensorVariance_smem[1024];
    partialRVariance_smem[hipThreadIdx_x] = 0.0f;                                     // initialization of _smem to 0 using all 1024 x 1 threads
    partialGVariance_smem[hipThreadIdx_x] = 0.0f;
    partialBVariance_smem[hipThreadIdx_x] = 0.0f;
    partialTensorVariance_smem[hipThreadIdx_x] = 0.0f;

    if (id_x >= xBufferLength)
        return;

    float4 accum_f4 = static_cast<float4>(0.0f);
    while (id_x < xBufferLength)
    {
        uint srcIdx = ((id_z * xBufferLength) + id_x) * 4;
        float4 temp_f4 = *(float4 *)(srcPtr + srcIdx);
        accum_f4 += temp_f4;
        id_x += hipBlockDim_x;
    }

    partialRVariance_smem[hipThreadIdx_x] = accum_f4.x;
    partialGVariance_smem[hipThreadIdx_x] = accum_f4.y;
    partialBVariance_smem[hipThreadIdx_x] = accum_f4.z;
    partialTensorVariance_smem[hipThreadIdx_x] = accum_f4.w;
    __syncthreads();                                                                  // syncthreads after _smem load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRVariance_smem[hipThreadIdx_x] += partialRVariance_smem[hipThreadIdx_x + threadMax];
            partialGVariance_smem[hipThreadIdx_x] += partialGVariance_smem[hipThreadIdx_x + threadMax];
            partialBVariance_smem[hipThreadIdx_x] += partialBVariance_smem[hipThreadIdx_x + threadMax];
            partialTensorVariance_smem[hipThreadIdx_x] += partialTensorVariance_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[id_z].xywhROI.roiHeight * roiTensorPtrSrc[id_z].xywhROI.roiWidth;
        uint dstIdx = id_z * 4;
        stddev_hip_compute(inputSrcPtr, &partialRVariance_smem[0], &dstPtr[dstIdx], totalElements);
        stddev_hip_compute(inputSrcPtr, &partialGVariance_smem[0], &dstPtr[dstIdx + 1], totalElements);
        stddev_hip_compute(inputSrcPtr, &partialBVariance_smem[0], &dstPtr[dstIdx + 2], totalElements);
        stddev_hip_compute(inputSrcPtr, &partialTensorVariance_smem[0], &dstPtr[dstIdx + 3], totalElements * 3);
    }
}

// -------------------- Set 1 - Reduction Stage 1 --------------------

template <typename T>
__global__ void tensor_variance_pln1_hip(T *srcPtr,
                                         uint2 srcStridesNH,
                                         float *tensorVarianceArr,
                                         Rpp32f *mean,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float4 mean_f4 = static_cast<float4>(mean[id_z]);

    __shared__ float partialVariance_smem[16][16];                                   // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    float *partialVarianceRowPtr_smem = &partialVariance_smem[hipThreadIdx_y][0];    // float pointer to beginning of each row in _smem
    partialVarianceRowPtr_smem[hipThreadIdx_x] = 0.0f;                               // initialization of _smem to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - (roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7);    // difference between roiWidth and alignedLength, where alignedLength = roiWidth & ~7
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float8 src_f8, temp_f8, tempSquared_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);                    // load 8 pixels to local memory
    rpp_hip_math_subtract8_const(&src_f8, &temp_f8, mean_f4);                        // subtract mean from each pixel
    rpp_hip_math_multiply8(&temp_f8, &temp_f8, &tempSquared_f8);                     // square the temporary values

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            tempSquared_f8.f1[i] = 0.0f;
    tempSquared_f8.f4[0] += tempSquared_f8.f4[1];                                    // perform small work of vectorized float4 addition
    partialVarianceRowPtr_smem[hipThreadIdx_x] = (tempSquared_f8.f1[0] +
                                                  tempSquared_f8.f1[1] +
                                                  tempSquared_f8.f1[2] +
                                                  tempSquared_f8.f1[3]);
    __syncthreads();                                                                 // syncthreads after _smem load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialVarianceRowPtr_smem[hipThreadIdx_x] += partialVarianceRowPtr_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialVarianceRowPtr_smem[0] += partialVarianceRowPtr_smem[increment];
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
            tensorVarianceArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialVarianceRowPtr_smem[0];
    }
}

template <typename T>
__global__ void tensor_variance_pln3_hip(T *srcPtr,
                                         uint3 srcStridesNCH,
                                         float *tensorVarianceArr,
                                         float4 *meanPtr_f4,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    /* Stores individual channel Variations computed from channel Means to compute Stddev of individual channels*/
    __shared__ float partialRVariance_smem[16][16];                                                       // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGVariance_smem[16][16];
    __shared__ float partialBVariance_smem[16][16];
    __shared__ float partialTensorVariance_smem[16][16];
    float *partialRVarianceRowPtr_smem = &partialRVariance_smem[hipThreadIdx_y][0];                       // float pointer to beginning of each row in _smem
    float *partialGVarianceRowPtr_smem = &partialGVariance_smem[hipThreadIdx_y][0];
    float *partialBVarianceRowPtr_smem = &partialBVariance_smem[hipThreadIdx_y][0];
    float *partialTensorVarianceRowPtr_smem = &partialTensorVariance_smem[hipThreadIdx_y][0];
    partialRVarianceRowPtr_smem[hipThreadIdx_x] = 0.0f;                                                   // initialization of _smem to 0 using all 16 x 16 threads
    partialGVarianceRowPtr_smem[hipThreadIdx_x] = 0.0f;
    partialBVarianceRowPtr_smem[hipThreadIdx_x] = 0.0f;
    partialTensorVarianceRowPtr_smem[hipThreadIdx_x] = 0.0f;
    float4 meanR_f4 = static_cast<float4>(meanPtr_f4[id_z].x);
    float4 meanG_f4 = static_cast<float4>(meanPtr_f4[id_z].y);
    float4 meanB_f4 = static_cast<float4>(meanPtr_f4[id_z].z);
    float4 meanTensor_f4 = static_cast<float4>(meanPtr_f4[id_z].w);

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - (roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7);   // difference between roiWidth and alignedLength, where alignedLength = roiWidth & ~7
    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float24 src_f24, tempChannelSquared_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);           // load 24 pixels to local memory
    mean_subtracted_square_3channel_hip_compute(&src_f24, &tempChannelSquared_f24, meanR_f4, meanG_f4, meanB_f4);

    d_float24 tempSquared_f24;
    mean_subtracted_square_3channel_hip_compute(&src_f24, &tempSquared_f24, meanTensor_f4, meanTensor_f4, meanTensor_f4);

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                                                // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            tempChannelSquared_f24.f8[0].f1[i] = 0.0f;
            tempChannelSquared_f24.f8[1].f1[i] = 0.0f;
            tempChannelSquared_f24.f8[2].f1[i] = 0.0f;
            tempSquared_f24.f8[0].f1[i] = 0.0f;
            tempSquared_f24.f8[1].f1[i] = 0.0f;
            tempSquared_f24.f8[2].f1[i] = 0.0f;
        }
    }

    reduce_variance_3channel_hip(&tempChannelSquared_f24, &tempSquared_f24,
                                 partialRVarianceRowPtr_smem, partialGVarianceRowPtr_smem,
                                 partialBVarianceRowPtr_smem, partialTensorVarianceRowPtr_smem, tensorVarianceArr);
}

template <typename T>
__global__ void tensor_variance_pkd3_hip(T *srcPtr,
                                         uint2 srcStridesNH,
                                         float *tensorVarianceArr,
                                         float4 *meanPtr_f4,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRVariance_smem[16][16];                                                       // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGVariance_smem[16][16];
    __shared__ float partialBVariance_smem[16][16];
    __shared__ float partialTensorVariance_smem[16][16];
    float *partialRVarianceRowPtr_smem = &partialRVariance_smem[hipThreadIdx_y][0];                       // float pointer to beginning of each row in _smem
    float *partialGVarianceRowPtr_smem = &partialGVariance_smem[hipThreadIdx_y][0];
    float *partialBVarianceRowPtr_smem = &partialBVariance_smem[hipThreadIdx_y][0];
    float *partialTensorVarianceRowPtr_smem = &partialTensorVariance_smem[hipThreadIdx_y][0];
    partialRVarianceRowPtr_smem[hipThreadIdx_x] = 0.0f;                                                   // initialization of _smem to 0 using all 16 x 16 threads
    partialGVarianceRowPtr_smem[hipThreadIdx_x] = 0.0f;
    partialBVarianceRowPtr_smem[hipThreadIdx_x] = 0.0f;
    partialTensorVarianceRowPtr_smem[hipThreadIdx_x] = 0.0f;
    float4 meanR_f4 = static_cast<float4>(meanPtr_f4[id_z].x);
    float4 meanG_f4 = static_cast<float4>(meanPtr_f4[id_z].y);
    float4 meanB_f4 = static_cast<float4>(meanPtr_f4[id_z].z);
    float4 meanTensor_f4 = static_cast<float4>(meanPtr_f4[id_z].w);

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - (roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7);   // difference between roiWidth and alignedLength, where alignedLength = roiWidth & ~7
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);

    d_float24 src_f24, tempChannelSquared_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);                            // load 24 pixels to local memory
    mean_subtracted_square_3channel_hip_compute(&src_f24, &tempChannelSquared_f24, meanR_f4, meanG_f4, meanB_f4);

    d_float24 tempSquared_f24;
    mean_subtracted_square_3channel_hip_compute(&src_f24, &tempSquared_f24, meanTensor_f4, meanTensor_f4, meanTensor_f4);

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                                                // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            tempChannelSquared_f24.f8[0].f1[i] = 0.0f;
            tempChannelSquared_f24.f8[1].f1[i] = 0.0f;
            tempChannelSquared_f24.f8[2].f1[i] = 0.0f;
            tempSquared_f24.f8[0].f1[i] = 0.0f;
            tempSquared_f24.f8[1].f1[i] = 0.0f;
            tempSquared_f24.f8[2].f1[i] = 0.0f;
        }
    }

    reduce_variance_3channel_hip(&tempChannelSquared_f24, &tempSquared_f24,
                                 partialRVarianceRowPtr_smem, partialGVarianceRowPtr_smem,
                                 partialBVarianceRowPtr_smem, partialTensorVarianceRowPtr_smem, tensorVarianceArr);
}

// -------------------- Set 2 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_tensor_stddev(T *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp32f *imageStddevArr,
                                 Rpp32f *meanTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (srcDescPtr->w + 7) >> 3;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = srcDescPtr->n;
    int gridDim_x = (int) ceil((float)globalThreads_x/LOCAL_THREADS_X);
    int gridDim_y = (int) ceil((float)globalThreads_y/LOCAL_THREADS_Y);
    int gridDim_z = (int) ceil((float)globalThreads_z/LOCAL_THREADS_Z);

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u tensorPartialVarArrLength = gridDim_x * gridDim_y * gridDim_z;
        float *tensorPartialVarArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        hipMemsetAsync(tensorPartialVarArr, 0, tensorPartialVarArrLength * sizeof(float), handle.GetStream());
        hipLaunchKernelGGL(tensor_variance_pln1_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialVarArr,
                           meanTensor,
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(tensor_stddev_grid_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           tensorPartialVarArr,
                           gridDim_x * gridDim_y,
                           imageStddevArr,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u tensorPartialVarArrLength = gridDim_x * gridDim_y * gridDim_z * 4;
        float *tensorPartialVarArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        hipMemsetAsync(tensorPartialVarArr, 0, tensorPartialVarArrLength * sizeof(float), handle.GetStream());
        hipLaunchKernelGGL(tensor_variance_pln3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           tensorPartialVarArr,
                           reinterpret_cast<float4 *>(meanTensor),
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(tensor_stddev_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           tensorPartialVarArr,
                           gridDim_x * gridDim_y,
                           imageStddevArr,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u tensorPartialVarArrLength = gridDim_x * gridDim_y * gridDim_z * 4;
        float *tensorPartialVarArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        hipMemsetAsync(tensorPartialVarArr, 0, tensorPartialVarArrLength * sizeof(float), handle.GetStream());
        hipLaunchKernelGGL(tensor_variance_pkd3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialVarArr,
                           reinterpret_cast<float4 *>(meanTensor),
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(tensor_stddev_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           tensorPartialVarArr,
                           gridDim_x * gridDim_y,
                           imageStddevArr,
                           roiTensorPtrSrc);
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_tensor_stddev<Rpp8u>(Rpp8u*,
                                                 RpptDescPtr,
                                                 Rpp32f*,
                                                 Rpp32f*,
                                                 RpptROIPtr,
                                                 RpptRoiType,
                                                 rpp::Handle&);

template RppStatus hip_exec_tensor_stddev<half>(half*,
                                                RpptDescPtr,
                                                Rpp32f*,
                                                Rpp32f*,
                                                RpptROIPtr,
                                                RpptRoiType,
                                                rpp::Handle&);

template RppStatus hip_exec_tensor_stddev<Rpp32f>(Rpp32f*,
                                                  RpptDescPtr,
                                                  Rpp32f*,
                                                  Rpp32f*,
                                                  RpptROIPtr,
                                                  RpptRoiType,
                                                  rpp::Handle&);

template RppStatus hip_exec_tensor_stddev<Rpp8s>(Rpp8s*,
                                                 RpptDescPtr,
                                                 Rpp32f*,
                                                 Rpp32f*,
                                                 RpptROIPtr,
                                                 RpptRoiType,
                                                 rpp::Handle&);
