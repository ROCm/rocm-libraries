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
#include "kernel_dims.hpp"

// -------------------- Set 0 - Reduction Stage 2 --------------------
__global__ void tensor_sum_grid_result_hip(Rpp32u *srcPtr,
                                           uint xBufferLength,
                                           Rpp64u *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ uint partialSum_smem[1024];                          // 8192 uints of src reduced to 1024 in a 1024 x 1 thread block
    partialSum_smem[hipThreadIdx_x] = 0;                            // initialization of Shared to 0 using all 1024 x 1 threads

    if (id_x >= xBufferLength)
    {
        return;
    }

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_uint8 src_ui8;
    *reinterpret_cast<d_uint8_s *>(&src_ui8) = *reinterpret_cast<d_uint8_s *>(srcPtr + srcIdx);

    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_ui8.ui1[i] = 0;                                     // local memory reset of invalid values (from the vectorized global load) to 0

    src_ui8.ui4[0] += src_ui8.ui4[1];                               // perform small work of vectorized uint4 addition
    partialSum_smem[hipThreadIdx_x] += (src_ui8.ui1[0] +
                                        src_ui8.ui1[1] +
                                        src_ui8.ui1[2] +
                                        src_ui8.ui1[3]);            // perform small work of reducing uint32s to uint using 1024 x 1 threads and store in Shared
    __syncthreads();                                                // syncthreads after Shared load

    // Reduction of 1024 uints on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSum_smem[hipThreadIdx_x] += partialSum_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
        dstPtr[hipBlockIdx_z] = (Rpp64u)partialSum_smem[0];
}

__global__ void tensor_sum_grid_result_hip(Rpp32s *srcPtr,
                                           uint xBufferLength,
                                           Rpp64s *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ int partialSum_smem[1024];                           // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    partialSum_smem[hipThreadIdx_x] = 0;                            // initialization of Shared to 0 using all 1024 x 1 threads

    if (id_x >= xBufferLength)
    {
        return;
    }

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_int8 src_i8;
    *reinterpret_cast<d_int8_s *>(&src_i8) = *reinterpret_cast<d_int8_s *>(srcPtr + srcIdx);
    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_i8.i1[i] = 0;                                       // local memory reset of invalid values (from the vectorized global load) to 0

    src_i8.i4[0] += src_i8.i4[1];
    partialSum_smem[hipThreadIdx_x] += (src_i8.i1[0] +
                                        src_i8.i1[1] +
                                        src_i8.i1[2] +
                                        src_i8.i1[3]);              // perform small work of reducing uint4s to uint using 1024 x 1 threads and store in Shared
    __syncthreads();                                                // syncthreads after Shared load

    // Reduction of 1024 ints on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSum_smem[hipThreadIdx_x] += partialSum_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
        dstPtr[hipBlockIdx_z] = (Rpp64s)partialSum_smem[0];
}

__global__ void tensor_sum_grid_result_hip(float *srcPtr,
                                       uint xBufferLength,
                                       float *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialSum_smem[1024];                         // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    partialSum_smem[hipThreadIdx_x] = 0.0f;                         // initialization of Shared to 0 using all 1024 x 1 threads

    if (id_x >= xBufferLength)
    {
        return;
    }

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local memory
    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                    // local memory reset of invalid values (from the vectorized global load) to 0.0f

    src_f8.f4[0] += src_f8.f4[1];                                   // perform small work of vectorized float4 addition
    partialSum_smem[hipThreadIdx_x] += (src_f8.f1[0] +
                                        src_f8.f1[1] +
                                        src_f8.f1[2] +
                                        src_f8.f1[3]);              // perform small work of reducing float4s to float using 1024 x 1 threads and store in Shared
    __syncthreads();                                                // syncthreads after Shared load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSum_smem[hipThreadIdx_x] += partialSum_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
        dstPtr[hipBlockIdx_z] = partialSum_smem[0];
}

__global__ void tensor_sum_grid_3channel_result_hip(Rpp32u *srcPtr,
                                                    uint xBufferLength,
                                                    Rpp64u *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ uint partialRSum_smem[1024];                                      // 8192 uints of src reduced to 1024 in a 1024 x 1 thread block
    __shared__ uint partialGSum_smem[1024];
    __shared__ uint partialBSum_smem[1024];
    partialRSum_smem[hipThreadIdx_x] = 0;                                        // initialization of Shared to 0 using all 1024 x 1 threads
    partialGSum_smem[hipThreadIdx_x] = 0;
    partialBSum_smem[hipThreadIdx_x] = 0;

    if (id_x >= xBufferLength)
    {
        return;
    }

    int xAlignedLength = xBufferLength & ~7;                                     // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                                  // difference between bufferLength and alignedLength
    uint srcIdx = ((id_z * xBufferLength) + id_x) * 3;

    d_uint24 src_ui24;
    *reinterpret_cast<d_uint24_s *>(&src_ui24) = *reinterpret_cast<d_uint24_s *>(srcPtr + srcIdx);
    rpp_hip_layouttoggle24_pkd3_to_pln3(reinterpret_cast<d_uint24_s *>(&src_ui24));

    if (id_x + 8 > xBufferLength)                                                // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_ui24.ui8[0].ui1[i] = 0;
            src_ui24.ui8[1].ui1[i] = 0;
            src_ui24.ui8[2].ui1[i] = 0;
        }
    }

    src_ui24.ui8[0].ui4[0] += src_ui24.ui8[0].ui4[1];
    src_ui24.ui8[1].ui4[0] += src_ui24.ui8[1].ui4[1];
    src_ui24.ui8[2].ui4[0] += src_ui24.ui8[2].ui4[1];
    partialRSum_smem[hipThreadIdx_x] = (src_ui24.ui8[0].ui1[0] +
                                        src_ui24.ui8[0].ui1[1] +
                                        src_ui24.ui8[0].ui1[2] +
                                        src_ui24.ui8[0].ui1[3]);                 // perform small work of reducing R uint32s to uint using 1024 threads and store in Shared
    partialGSum_smem[hipThreadIdx_x] = (src_ui24.ui8[1].ui1[0] +
                                        src_ui24.ui8[1].ui1[1] +
                                        src_ui24.ui8[1].ui1[2] +
                                        src_ui24.ui8[1].ui1[3]);                 // perform small work of reducing G uint32s to uint using 1024 threads and store in Shared
    partialBSum_smem[hipThreadIdx_x] = (src_ui24.ui8[2].ui1[0] +
                                        src_ui24.ui8[2].ui1[1] +
                                        src_ui24.ui8[2].ui1[2] +
                                        src_ui24.ui8[2].ui1[3]);                 // perform small work of reducing B uint32s to uint using 1024 threads and store in Shared

    __syncthreads();                                                             // syncthreads after Shared load

    // Reduction of 1024 uints on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSum_smem[hipThreadIdx_x] += partialRSum_smem[hipThreadIdx_x + threadMax];
            partialGSum_smem[hipThreadIdx_x] += partialGSum_smem[hipThreadIdx_x + threadMax];
            partialBSum_smem[hipThreadIdx_x] += partialBSum_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        Rpp64u sum = (Rpp64u)partialRSum_smem[0] + (Rpp64u)partialGSum_smem[0] + (Rpp64u)partialBSum_smem[0];
        int idx = hipBlockIdx_z * 4;
        dstPtr[idx] = (Rpp64u)partialRSum_smem[0];
        dstPtr[idx + 1] = (Rpp64u)partialGSum_smem[0];
        dstPtr[idx + 2] = (Rpp64u)partialBSum_smem[0];
        dstPtr[idx + 3] = sum;
    }
}

__global__ void tensor_sum_grid_3channel_result_hip(Rpp32s *srcPtr,
                                                    uint xBufferLength,
                                                    Rpp64s *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ int partialRSum_smem[1024];                                       // 8192 uints of src reduced to 1024 in a 1024 x 1 thread block
    __shared__ int partialGSum_smem[1024];
    __shared__ int partialBSum_smem[1024];
    partialRSum_smem[hipThreadIdx_x] = 0;                                        // initialization of Shared to 0 using all 1024 x 1 threads
    partialGSum_smem[hipThreadIdx_x] = 0;
    partialBSum_smem[hipThreadIdx_x] = 0;

    if (id_x >= xBufferLength)
    {
        return;
    }

    int xAlignedLength = xBufferLength & ~7;                                     // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                                  // difference between bufferLength and alignedLength
    uint srcIdx = ((id_z * xBufferLength) + id_x) * 3;

    d_int24 src_i24;
    *reinterpret_cast<d_int24_s *>(&src_i24) = *reinterpret_cast<d_int24_s *>(srcPtr + srcIdx);
    rpp_hip_layouttoggle24_pkd3_to_pln3(reinterpret_cast<d_int24_s *>(&src_i24));

    if (id_x + 8 > xBufferLength)                                                // local memory reset of invalid values (from the vectorized global load) to 0
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_i24.i8[0].i1[i] = 0;
            src_i24.i8[1].i1[i] = 0;
            src_i24.i8[2].i1[i] = 0;
        }
    }

    src_i24.i8[0].i4[0] += src_i24.i8[0].i4[1];
    src_i24.i8[1].i4[0] += src_i24.i8[1].i4[1];
    src_i24.i8[2].i4[0] += src_i24.i8[2].i4[1];
    partialRSum_smem[hipThreadIdx_x] = (src_i24.i8[0].i1[0] +
                                        src_i24.i8[0].i1[1] +
                                        src_i24.i8[0].i1[2] +
                                        src_i24.i8[0].i1[3]);                    // perform small work of reducing R int32s to int using 1024 threads and store in Shared
    partialGSum_smem[hipThreadIdx_x] = (src_i24.i8[1].i1[0] +
                                        src_i24.i8[1].i1[1] +
                                        src_i24.i8[1].i1[2] +
                                        src_i24.i8[1].i1[3]);                    // perform small work of reducing G int32s to int using 1024 threads and store in Shared
    partialBSum_smem[hipThreadIdx_x] = (src_i24.i8[2].i1[0] +
                                        src_i24.i8[2].i1[1] +
                                        src_i24.i8[2].i1[2] +
                                        src_i24.i8[2].i1[3]);                    // perform small work of reducing B int32s to int using 1024 threads and store in Shared

    __syncthreads();                                                             // syncthreads after Shared load

    // Reduction of 1024 ints on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSum_smem[hipThreadIdx_x] += partialRSum_smem[hipThreadIdx_x + threadMax];
            partialGSum_smem[hipThreadIdx_x] += partialGSum_smem[hipThreadIdx_x + threadMax];
            partialBSum_smem[hipThreadIdx_x] += partialBSum_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        Rpp64s sum = (Rpp64s)partialRSum_smem[0] + (Rpp64u)partialGSum_smem[0] + (Rpp64u)partialBSum_smem[0];
        int idx = hipBlockIdx_z * 4;
        dstPtr[idx] = (Rpp64s)partialRSum_smem[0];
        dstPtr[idx + 1] = (Rpp64s)partialGSum_smem[0];
        dstPtr[idx + 2] = (Rpp64s)partialBSum_smem[0];
        dstPtr[idx + 3] = sum;
    }
}

__global__ void tensor_sum_grid_3channel_result_hip(float *srcPtr,
                                                    uint xBufferLength,
                                                    float *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialRSum_smem[1024];                                     // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    __shared__ float partialGSum_smem[1024];
    __shared__ float partialBSum_smem[1024];
    partialRSum_smem[hipThreadIdx_x] = 0.0f;                                     // initialization of Shared to 0 using all 1024 x 1 threads
    partialGSum_smem[hipThreadIdx_x] = 0.0f;
    partialBSum_smem[hipThreadIdx_x] = 0.0f;

    if (id_x >= xBufferLength)
    {
        return;
    }

    int xAlignedLength = xBufferLength & ~7;                                     // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                                  // difference between bufferLength and alignedLength
    uint srcIdx = ((id_z * xBufferLength) + id_x) * 3;

    d_float24 src_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);   // load 24 pixels to local memory
    if (id_x + 8 > xBufferLength)                                                // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = 0.0f;
            src_f24.f8[1].f1[i] = 0.0f;
            src_f24.f8[2].f1[i] = 0.0f;
        }
    }
    src_f24.f8[0].f4[0] += src_f24.f8[0].f4[1];                                  // perform small work of vectorized float4 addition
    src_f24.f8[1].f4[0] += src_f24.f8[1].f4[1];
    src_f24.f8[2].f4[0] += src_f24.f8[2].f4[1];
    partialRSum_smem[hipThreadIdx_x] = (src_f24.f8[0].f1[0] +
                                        src_f24.f8[0].f1[1] +
                                        src_f24.f8[0].f1[2] +
                                        src_f24.f8[0].f1[3]);                    // perform small work of reducing R float4s to float using 1024 threads and store in Shared
    partialGSum_smem[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                        src_f24.f8[1].f1[1] +
                                        src_f24.f8[1].f1[2] +
                                        src_f24.f8[1].f1[3]);                    // perform small work of reducing G float4s to float using 1024 threads and store in Shared
    partialBSum_smem[hipThreadIdx_x] = (src_f24.f8[2].f1[0] +
                                        src_f24.f8[2].f1[1] +
                                        src_f24.f8[2].f1[2] +
                                        src_f24.f8[2].f1[3]);                    // perform small work of reducing B float4s to float using 1024 threads and store in Shared

    __syncthreads();                                                             // syncthreads after Shared load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSum_smem[hipThreadIdx_x] += partialRSum_smem[hipThreadIdx_x + threadMax];
            partialGSum_smem[hipThreadIdx_x] += partialGSum_smem[hipThreadIdx_x + threadMax];
            partialBSum_smem[hipThreadIdx_x] += partialBSum_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        float sum = partialRSum_smem[0] + partialGSum_smem[0] + partialBSum_smem[0];
        int idx = hipBlockIdx_z * 4;
        dstPtr[idx] = partialRSum_smem[0];
        dstPtr[idx + 1] = partialGSum_smem[0];
        dstPtr[idx + 2] = partialBSum_smem[0];
        dstPtr[idx + 3] = sum;
    }
}

// Handle f16/f32
template <typename T, typename U>
__global__ void tensor_sum_pln1_hip(T *srcPtr,
                                    uint2 srcStridesNH,
                                    U *tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialSum_smem[16][16];                               // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    float *partialSumRowPtr_smem = &partialSum_smem[hipThreadIdx_y][0];     // float pointer to beginning of each row in Shared
    partialSumRowPtr_smem[hipThreadIdx_x] = 0.0f;                           // initialization of Shared to 0.0f using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local memory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
    src_f8.f4[0] += src_f8.f4[1];                                           // perform small work of vectorized float4 addition
    partialSumRowPtr_smem[hipThreadIdx_x] = (src_f8.f1[0] +
                                             src_f8.f1[1] +
                                             src_f8.f1[2] +
                                             src_f8.f1[3]);                 // perform small work of reducing float4s to float using 16 x 16 threads and store in Shared
    __syncthreads();                                                        // syncthreads after Shared load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSumRowPtr_smem[hipThreadIdx_x] += partialSumRowPtr_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialSumRowPtr_smem[0] += partialSumRowPtr_smem[increment];
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
            tensorSumArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialSumRowPtr_smem[0];
    }
}

template __global__ void tensor_sum_pln1_hip<half, Rpp32f>(half*,
                                                           uint2,
                                                           Rpp32f*,
                                                           RpptROIPtr);

template __global__ void tensor_sum_pln1_hip<Rpp32f, Rpp32f>(Rpp32f*,
                                                             uint2,
                                                             Rpp32f*,
                                                             RpptROIPtr);

// -------------------- Set 1 - Reduction Stage 1 --------------------
// Handle U8
template<>
__global__ void tensor_sum_pln1_hip(Rpp8u *srcPtr,
                                    uint2 srcStridesNH,
                                    Rpp32u *tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ uint partialSum_smem[16][16];                                // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    uint *partialSumRowPtr_smem = &partialSum_smem[hipThreadIdx_y][0];      // uint pointer to beginning of each row in Shared
    partialSumRowPtr_smem[hipThreadIdx_x] = 0;                              // initialization of Shared to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_uint8 src_ui8;
    rpp_hip_load8_to_uint8(srcPtr + srcIdx, &src_ui8);                     // load 8 pixels to local memory

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            src_ui8.ui1[i] = 0;                                            // local memory reset of invalid values (from the vectorized global load) to 0
    src_ui8.ui4[0] += src_ui8.ui4[1];                                      // perform small work of vectorized uint4 addition
    partialSumRowPtr_smem[hipThreadIdx_x] += (src_ui8.ui1[0] +
                                              src_ui8.ui1[1] +
                                              src_ui8.ui1[2] +
                                              src_ui8.ui1[3]);             // perform small work of reducing uint8s to uint using 16 x 16 threads and store in Shared
    __syncthreads();                                                       // syncthreads after Shared load

    // Reduction of 16 uints on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSumRowPtr_smem[hipThreadIdx_x] += partialSumRowPtr_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 uints on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialSumRowPtr_smem[0] += partialSumRowPtr_smem[increment];
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
            tensorSumArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialSumRowPtr_smem[0];
    }
}

// Handle I8
template<>
__global__ void tensor_sum_pln1_hip(Rpp8s *srcPtr,
                                    uint2 srcStridesNH,
                                    Rpp32s *tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ int partialSum_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    int *partialSumRowPtr_smem = &partialSum_smem[hipThreadIdx_y][0];       // int pointer to beginning of each row in Shared
    partialSumRowPtr_smem[hipThreadIdx_x] = 0;                              // initialization of Shared to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_int8 src_i8;
    rpp_hip_load8_to_int8(srcPtr + srcIdx, &src_i8);                        // load 8 pixels to local memory

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            src_i8.i1[i] = 0;                                               // local memory reset of invalid values (from the vectorized global load) to 0
    src_i8.i4[0] += src_i8.i4[1];
    partialSumRowPtr_smem[hipThreadIdx_x] += (src_i8.i1[0] +
                                              src_i8.i1[1] +
                                              src_i8.i1[2] +
                                              src_i8.i1[3]);                // perform small work of reducing int4s to int using 16 x 16 threads and store in Shared
    __syncthreads();                                                        // syncthreads after Shared load

    // Reduction of 16 ints on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSumRowPtr_smem[hipThreadIdx_x] += partialSumRowPtr_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 ints on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialSumRowPtr_smem[0] += partialSumRowPtr_smem[increment];
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
            tensorSumArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialSumRowPtr_smem[0];
    }
}

// Handle f16/f32
template <typename T, typename U>
__global__ void tensor_sum_pln3_hip(T *srcPtr,
                                    uint3 srcStridesNCH,
                                    U *tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRSum_smem[16][16];                                                  // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGSum_smem[16][16];
    __shared__ float partialBSum_smem[16][16];
    float *partialRSumRowPtr_smem = &partialRSum_smem[hipThreadIdx_y][0];                       // float pointer to beginning of each row in Shared
    float *partialGSumRowPtr_smem = &partialGSum_smem[hipThreadIdx_y][0];
    float *partialBSumRowPtr_smem = &partialBSum_smem[hipThreadIdx_y][0];
    partialRSumRowPtr_smem[hipThreadIdx_x] = 0.0f;                                              // initialization of Shared to 0 using all 16 x 16 threads
    partialGSumRowPtr_smem[hipThreadIdx_x] = 0.0f;
    partialBSumRowPtr_smem[hipThreadIdx_x] = 0.0f;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;                        // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float24 src_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24); // load 24 pixels to local memory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                                      // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = 0.0f;
            src_f24.f8[1].f1[i] = 0.0f;
            src_f24.f8[2].f1[i] = 0.0f;
        }
    }
    src_f24.f8[0].f4[0] += src_f24.f8[0].f4[1];                                                 // perform small work of vectorized float4 addition
    src_f24.f8[1].f4[0] += src_f24.f8[1].f4[1];
    src_f24.f8[2].f4[0] += src_f24.f8[2].f4[1];
    partialRSumRowPtr_smem[hipThreadIdx_x] = (src_f24.f8[0].f1[0] +
                                              src_f24.f8[0].f1[1] +
                                              src_f24.f8[0].f1[2] +
                                              src_f24.f8[0].f1[3]);                             // perform small work of reducing R float4s to float using 16 x 16 threads and store in Shared
    partialGSumRowPtr_smem[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                              src_f24.f8[1].f1[1] +
                                              src_f24.f8[1].f1[2] +
                                              src_f24.f8[1].f1[3]);                             // perform small work of reducing G float4s to float using 16 x 16 threads and store in Shared
    partialBSumRowPtr_smem[hipThreadIdx_x] = (src_f24.f8[2].f1[0] +
                                              src_f24.f8[2].f1[1] +
                                              src_f24.f8[2].f1[2] +
                                              src_f24.f8[2].f1[3]);                             // perform small work of reducing B float4s to float using 16 x 16 threads and store in Shared

    __syncthreads();                                                                            // syncthreads after Shared load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSumRowPtr_smem[hipThreadIdx_x] += partialRSumRowPtr_smem[hipThreadIdx_x + threadMax];
            partialGSumRowPtr_smem[hipThreadIdx_x] += partialGSumRowPtr_smem[hipThreadIdx_x + threadMax];
            partialBSumRowPtr_smem[hipThreadIdx_x] += partialBSumRowPtr_smem[hipThreadIdx_x + threadMax];
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
                partialRSumRowPtr_smem[0] += partialRSumRowPtr_smem[increment];
                partialGSumRowPtr_smem[0] += partialGSumRowPtr_smem[increment];
                partialBSumRowPtr_smem[0] += partialBSumRowPtr_smem[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            tensorSumArr[idx] = partialRSumRowPtr_smem[0];
            tensorSumArr[idx + 1] = partialGSumRowPtr_smem[0];
            tensorSumArr[idx + 2] = partialBSumRowPtr_smem[0];
        }
    }
}

template __global__ void tensor_sum_pln3_hip<half, Rpp32f>(half*,
                                                           uint3,
                                                           Rpp32f*,
                                                           RpptROIPtr);

template __global__ void tensor_sum_pln3_hip<Rpp32f, Rpp32f>(Rpp32f*,
                                                             uint3,
                                                             Rpp32f*,
                                                             RpptROIPtr);

// Handle U8
template<>
__global__ void tensor_sum_pln3_hip(Rpp8u *srcPtr,
                                    uint3 srcStridesNCH,
                                    Rpp32u *tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ uint partialRSum_smem[16][16];                                                   // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ uint partialGSum_smem[16][16];
    __shared__ uint partialBSum_smem[16][16];
    uint *partialRSumRowPtr_smem = &partialRSum_smem[hipThreadIdx_y][0];                        // uint pointer to beginning of each row in Shared
    uint *partialGSumRowPtr_smem = &partialGSum_smem[hipThreadIdx_y][0];
    uint *partialBSumRowPtr_smem = &partialBSum_smem[hipThreadIdx_y][0];
    partialRSumRowPtr_smem[hipThreadIdx_x] = 0;                                                 // initialization of Shared to 0 using all 16 x 16 threads
    partialGSumRowPtr_smem[hipThreadIdx_x] = 0;
    partialBSumRowPtr_smem[hipThreadIdx_x] = 0;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;                        // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_uint24 src_ui24;
    rpp_hip_load24_pln3_to_uint24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_ui24);

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                                      // local memory reset of invalid values (from the vectorized global load) to 0
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_ui24.ui8[0].ui1[i] = 0;
            src_ui24.ui8[1].ui1[i] = 0;
            src_ui24.ui8[2].ui1[i] = 0;
        }
    }
    src_ui24.ui8[0].ui4[0] += src_ui24.ui8[0].ui4[1];
    src_ui24.ui8[1].ui4[0] += src_ui24.ui8[1].ui4[1];
    src_ui24.ui8[2].ui4[0] += src_ui24.ui8[2].ui4[1];
    partialRSumRowPtr_smem[hipThreadIdx_x] = (src_ui24.ui8[0].ui1[0] +
                                              src_ui24.ui8[0].ui1[1] +
                                              src_ui24.ui8[0].ui1[2] +
                                              src_ui24.ui8[0].ui1[3]);                           // perform small work of reducing R uint4s to uint using 16 x 16 threads and store in Shared
    partialGSumRowPtr_smem[hipThreadIdx_x] = (src_ui24.ui8[1].ui1[0] +
                                              src_ui24.ui8[1].ui1[1] +
                                              src_ui24.ui8[1].ui1[2] +
                                              src_ui24.ui8[1].ui1[3]);                           // perform small work of reducing G uint4s to uint using 16 x 16 threads and store in Shared
    partialBSumRowPtr_smem[hipThreadIdx_x] = (src_ui24.ui8[2].ui1[0] +
                                              src_ui24.ui8[2].ui1[1] +
                                              src_ui24.ui8[2].ui1[2] +
                                              src_ui24.ui8[2].ui1[3]);                           // perform small work of reducing B uint4s to uint using 16 x 16 threads and store in Shared

    __syncthreads();                                                                             // syncthreads after Shared load

    // Reduction of 16 uints on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSumRowPtr_smem[hipThreadIdx_x] += partialRSumRowPtr_smem[hipThreadIdx_x + threadMax];
            partialGSumRowPtr_smem[hipThreadIdx_x] += partialGSumRowPtr_smem[hipThreadIdx_x + threadMax];
            partialBSumRowPtr_smem[hipThreadIdx_x] += partialBSumRowPtr_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 uints on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
            {
                partialRSumRowPtr_smem[0] += partialRSumRowPtr_smem[increment];
                partialGSumRowPtr_smem[0] += partialGSumRowPtr_smem[increment];
                partialBSumRowPtr_smem[0] += partialBSumRowPtr_smem[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            tensorSumArr[idx] = partialRSumRowPtr_smem[0];
            tensorSumArr[idx + 1] = partialGSumRowPtr_smem[0];
            tensorSumArr[idx + 2] = partialBSumRowPtr_smem[0];
        }
    }
}

// Handle I8
template<>
__global__ void tensor_sum_pln3_hip(Rpp8s *srcPtr,
                                    uint3 srcStridesNCH,
                                    Rpp32s *tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ int partialRSum_smem[16][16];                                                    // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ int partialGSum_smem[16][16];
    __shared__ int partialBSum_smem[16][16];
    int *partialRSumRowPtr_smem = &partialRSum_smem[hipThreadIdx_y][0];                         // int pointer to beginning of each row in Shared
    int *partialGSumRowPtr_smem = &partialGSum_smem[hipThreadIdx_y][0];
    int *partialBSumRowPtr_smem = &partialBSum_smem[hipThreadIdx_y][0];
    partialRSumRowPtr_smem[hipThreadIdx_x] = 0;                                                 // initialization of Shared to 0 using all 16 x 16 threads
    partialGSumRowPtr_smem[hipThreadIdx_x] = 0;
    partialBSumRowPtr_smem[hipThreadIdx_x] = 0;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;                        // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_int24 src_i24;
    rpp_hip_load24_pln3_to_int24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_i24);

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                                      // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_i24.i8[0].i1[i] = 0;
            src_i24.i8[1].i1[i] = 0;
            src_i24.i8[2].i1[i] = 0;
        }
    }
    src_i24.i8[0].i4[0] += src_i24.i8[0].i4[1];
    src_i24.i8[1].i4[0] += src_i24.i8[1].i4[1];
    src_i24.i8[2].i4[0] += src_i24.i8[2].i4[1];
    partialRSumRowPtr_smem[hipThreadIdx_x] = (src_i24.i8[0].i1[0] +
                                              src_i24.i8[0].i1[1] +
                                              src_i24.i8[0].i1[2] +
                                              src_i24.i8[0].i1[3]);                             // perform small work of reducing R int4s to int using 16 x 16 threads and store in Shared
    partialGSumRowPtr_smem[hipThreadIdx_x] = (src_i24.i8[1].i1[0] +
                                              src_i24.i8[1].i1[1] +
                                              src_i24.i8[1].i1[2] +
                                              src_i24.i8[1].i1[3]);                             // perform small work of reducing G int4s to int using 16 x 16 threads and store in Shared
    partialBSumRowPtr_smem[hipThreadIdx_x] = (src_i24.i8[2].i1[0] +
                                              src_i24.i8[2].i1[1] +
                                              src_i24.i8[2].i1[2] +
                                              src_i24.i8[2].i1[3]);                             // perform small work of reducing B int4s to int using 16 x 16 threads and store in Shared

    __syncthreads();                                                                            // syncthreads after Shared load

    // Reduction of 16 ints on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSumRowPtr_smem[hipThreadIdx_x] += partialRSumRowPtr_smem[hipThreadIdx_x + threadMax];
            partialGSumRowPtr_smem[hipThreadIdx_x] += partialGSumRowPtr_smem[hipThreadIdx_x + threadMax];
            partialBSumRowPtr_smem[hipThreadIdx_x] += partialBSumRowPtr_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 ints on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
            {
                partialRSumRowPtr_smem[0] += partialRSumRowPtr_smem[increment];
                partialGSumRowPtr_smem[0] += partialGSumRowPtr_smem[increment];
                partialBSumRowPtr_smem[0] += partialBSumRowPtr_smem[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            tensorSumArr[idx] = partialRSumRowPtr_smem[0];
            tensorSumArr[idx + 1] = partialGSumRowPtr_smem[0];
            tensorSumArr[idx + 2] = partialBSumRowPtr_smem[0];
        }
    }
}

// Handle f16/f32
template <typename T, typename U>
__global__ void tensor_sum_pkd3_hip(T *srcPtr,
                                    uint2 srcStridesNH,
                                    U *tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRSum_smem[16][16];                                      // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGSum_smem[16][16];
    __shared__ float partialBSum_smem[16][16];
    float *partialRSumRowPtr_smem = &partialRSum_smem[hipThreadIdx_y][0];           // float pointer to beginning of each row in Shared
    float *partialGSumRowPtr_smem = &partialGSum_smem[hipThreadIdx_y][0];
    float *partialBSumRowPtr_smem = &partialBSum_smem[hipThreadIdx_y][0];
    partialRSumRowPtr_smem[hipThreadIdx_x] = 0.0f;                                  // initialization of Shared to 0 using all 16 x 16 threads
    partialGSumRowPtr_smem[hipThreadIdx_x] = 0.0f;
    partialBSumRowPtr_smem[hipThreadIdx_x] = 0.0f;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;               // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;            // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);

    d_float24 src_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);      // load 24 pixels to local memory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                          // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = 0.0f;
            src_f24.f8[1].f1[i] = 0.0f;
            src_f24.f8[2].f1[i] = 0.0f;
        }
    }
    src_f24.f8[0].f4[0] += src_f24.f8[0].f4[1];                                     // perform small work of vectorized float4 addition
    src_f24.f8[1].f4[0] += src_f24.f8[1].f4[1];
    src_f24.f8[2].f4[0] += src_f24.f8[2].f4[1];
    partialRSumRowPtr_smem[hipThreadIdx_x] = (src_f24.f8[0].f1[0] +
                                              src_f24.f8[0].f1[1] +
                                              src_f24.f8[0].f1[2] +
                                              src_f24.f8[0].f1[3]);                 // perform small work of reducing R float4s to float using 16 x 16 threads and store in Shared
    partialGSumRowPtr_smem[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                              src_f24.f8[1].f1[1] +
                                              src_f24.f8[1].f1[2] +
                                              src_f24.f8[1].f1[3]);                 // perform small work of reducing G float4s to float using 16 x 16 threads and store in Shared
    partialBSumRowPtr_smem[hipThreadIdx_x] = (src_f24.f8[2].f1[0] +
                                              src_f24.f8[2].f1[1] +
                                              src_f24.f8[2].f1[2] +
                                              src_f24.f8[2].f1[3]);                 // perform small work of reducing B float4s to float using 16 x 16 threads and store in Shared

    __syncthreads();                                                                // syncthreads after Shared load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSumRowPtr_smem[hipThreadIdx_x] += partialRSumRowPtr_smem[hipThreadIdx_x + threadMax];
            partialGSumRowPtr_smem[hipThreadIdx_x] += partialGSumRowPtr_smem[hipThreadIdx_x + threadMax];
            partialBSumRowPtr_smem[hipThreadIdx_x] += partialBSumRowPtr_smem[hipThreadIdx_x + threadMax];
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
                partialRSumRowPtr_smem[0] += partialRSumRowPtr_smem[increment];
                partialGSumRowPtr_smem[0] += partialGSumRowPtr_smem[increment];
                partialBSumRowPtr_smem[0] += partialBSumRowPtr_smem[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            tensorSumArr[idx] = partialRSumRowPtr_smem[0];
            tensorSumArr[idx + 1] = partialGSumRowPtr_smem[0];
            tensorSumArr[idx + 2] = partialBSumRowPtr_smem[0];
        }
    }
}

template __global__ void tensor_sum_pkd3_hip<half, Rpp32f>(half*,
                                                           uint2,
                                                           Rpp32f*,
                                                           RpptROIPtr);

template __global__ void tensor_sum_pkd3_hip<Rpp32f, Rpp32f>(Rpp32f*,
                                                             uint2,
                                                             Rpp32f*,
                                                             RpptROIPtr);

// Handle U8
template<>
__global__ void tensor_sum_pkd3_hip(Rpp8u *srcPtr,
                                    uint2 srcStridesNH,
                                    Rpp32u *tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ uint partialRSum_smem[16][16];                                       // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ uint partialGSum_smem[16][16];
    __shared__ uint partialBSum_smem[16][16];
    uint *partialRSumRowPtr_smem = &partialRSum_smem[hipThreadIdx_y][0];            // uint pointer to beginning of each row in Shared
    uint *partialGSumRowPtr_smem = &partialGSum_smem[hipThreadIdx_y][0];
    uint *partialBSumRowPtr_smem = &partialBSum_smem[hipThreadIdx_y][0];
    partialRSumRowPtr_smem[hipThreadIdx_x] = 0;                                     // initialization of Shared to 0 using all 16 x 16 threads
    partialGSumRowPtr_smem[hipThreadIdx_x] = 0;
    partialBSumRowPtr_smem[hipThreadIdx_x] = 0;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;               // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;            // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);

    d_uint24 src_ui24;
    rpp_hip_load24_pkd3_to_uint24_pln3(srcPtr + srcIdx, &src_ui24);

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                          // local memory reset of invalid values (from the vectorized global load) to 0
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_ui24.ui8[0].ui1[i] = 0;
            src_ui24.ui8[1].ui1[i] = 0;
            src_ui24.ui8[2].ui1[i] = 0;
        }
    }
    src_ui24.ui8[0].ui4[0] += src_ui24.ui8[0].ui4[1];
    src_ui24.ui8[1].ui4[0] += src_ui24.ui8[1].ui4[1];
    src_ui24.ui8[2].ui4[0] += src_ui24.ui8[2].ui4[1];
    partialRSumRowPtr_smem[hipThreadIdx_x] = (src_ui24.ui8[0].ui1[0] +
                                              src_ui24.ui8[0].ui1[1] +
                                              src_ui24.ui8[0].ui1[2] +
                                              src_ui24.ui8[0].ui1[3]);              // perform small work of reducing R uchar8s to uint using 16 x 16 threads and store in Shared
    partialGSumRowPtr_smem[hipThreadIdx_x] = (src_ui24.ui8[1].ui1[0] +
                                              src_ui24.ui8[1].ui1[1] +
                                              src_ui24.ui8[1].ui1[2] +
                                              src_ui24.ui8[1].ui1[3]);              // perform small work of reducing G uchar8s to uint using 16 x 16 threads and store in Shared
    partialBSumRowPtr_smem[hipThreadIdx_x] = (src_ui24.ui8[2].ui1[0] +
                                              src_ui24.ui8[2].ui1[1] +
                                              src_ui24.ui8[2].ui1[2] +
                                              src_ui24.ui8[2].ui1[3]);              // perform small work of reducing B uchar8s to uint using 16 x 16 threads and store in Shared

    __syncthreads();                                                                // syncthreads after Shared load
    // Reduction of 16 uints on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSumRowPtr_smem[hipThreadIdx_x] += partialRSumRowPtr_smem[hipThreadIdx_x + threadMax];
            partialGSumRowPtr_smem[hipThreadIdx_x] += partialGSumRowPtr_smem[hipThreadIdx_x + threadMax];
            partialBSumRowPtr_smem[hipThreadIdx_x] += partialBSumRowPtr_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 uints on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
            {
                partialRSumRowPtr_smem[0] += partialRSumRowPtr_smem[increment];
                partialGSumRowPtr_smem[0] += partialGSumRowPtr_smem[increment];
                partialBSumRowPtr_smem[0] += partialBSumRowPtr_smem[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            tensorSumArr[idx] = partialRSumRowPtr_smem[0];
            tensorSumArr[idx + 1] = partialGSumRowPtr_smem[0];
            tensorSumArr[idx + 2] = partialBSumRowPtr_smem[0];
        }
    }
}

// Handle I8
template<>
__global__ void tensor_sum_pkd3_hip(Rpp8s *srcPtr,
                                    uint2 srcStridesNH,
                                    Rpp32s *tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ int partialRSum_smem[16][16];                                        // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ int partialGSum_smem[16][16];
    __shared__ int partialBSum_smem[16][16];
    int *partialRSumRowPtr_smem = &partialRSum_smem[hipThreadIdx_y][0];             // int pointer to beginning of each row in Shared
    int *partialGSumRowPtr_smem = &partialGSum_smem[hipThreadIdx_y][0];
    int *partialBSumRowPtr_smem = &partialBSum_smem[hipThreadIdx_y][0];
    partialRSumRowPtr_smem[hipThreadIdx_x] = 0;                                     // initialization of Shared to 0 using all 16 x 16 threads
    partialGSumRowPtr_smem[hipThreadIdx_x] = 0;
    partialBSumRowPtr_smem[hipThreadIdx_x] = 0;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;               // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;            // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);

    d_int24 src_i24;
    rpp_hip_load24_pkd3_to_int24_pln3(srcPtr + srcIdx, &src_i24);

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                          // local memory reset of invalid values (from the vectorized global load) to 0
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_i24.i8[0].i1[i] = 0;
            src_i24.i8[1].i1[i] = 0;
            src_i24.i8[2].i1[i] = 0;
        }
    }
    src_i24.i8[0].i4[0] += src_i24.i8[0].i4[1];
    src_i24.i8[1].i4[0] += src_i24.i8[1].i4[1];
    src_i24.i8[2].i4[0] += src_i24.i8[2].i4[1];
    partialRSumRowPtr_smem[hipThreadIdx_x] = (src_i24.i8[0].i1[0] +
                                              src_i24.i8[0].i1[1] +
                                              src_i24.i8[0].i1[2] +
                                              src_i24.i8[0].i1[3]);                 // perform small work of reducing R schar4s to int using 16 x 16 threads and store in Shared
    partialGSumRowPtr_smem[hipThreadIdx_x] = (src_i24.i8[1].i1[0] +
                                              src_i24.i8[1].i1[1] +
                                              src_i24.i8[1].i1[2] +
                                              src_i24.i8[1].i1[3]);                 // perform small work of reducing G schar4s to int using 16 x 16 threads and store in Shared
    partialBSumRowPtr_smem[hipThreadIdx_x] = (src_i24.i8[2].i1[0] +
                                              src_i24.i8[2].i1[1] +
                                              src_i24.i8[2].i1[2] +
                                              src_i24.i8[2].i1[3]);                 // perform small work of reducing B schar4s to int using 16 x 16 threads and store in Shared

    __syncthreads();                                                                // syncthreads after Shared load

    // Reduction of 16 ints on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSumRowPtr_smem[hipThreadIdx_x] += partialRSumRowPtr_smem[hipThreadIdx_x + threadMax];
            partialGSumRowPtr_smem[hipThreadIdx_x] += partialGSumRowPtr_smem[hipThreadIdx_x + threadMax];
            partialBSumRowPtr_smem[hipThreadIdx_x] += partialBSumRowPtr_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 ints on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
            {
                partialRSumRowPtr_smem[0] += partialRSumRowPtr_smem[increment];
                partialGSumRowPtr_smem[0] += partialGSumRowPtr_smem[increment];
                partialBSumRowPtr_smem[0] += partialBSumRowPtr_smem[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            tensorSumArr[idx] = partialRSumRowPtr_smem[0];
            tensorSumArr[idx + 1] = partialGSumRowPtr_smem[0];
            tensorSumArr[idx + 2] = partialBSumRowPtr_smem[0];
        }
    }
}

// Handle f16/f32 datatype
template <typename T, typename U>
RppStatus hip_exec_tensor_sum(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              U *tensorSumArr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (srcDescPtr->w + 7) >> 3;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    int gridDim_x = (int) ceil((float)globalThreads_x/LOCAL_THREADS_X);
    int gridDim_y = (int) ceil((float)globalThreads_y/LOCAL_THREADS_Y);
    int gridDim_z = (int) ceil((float)globalThreads_z/LOCAL_THREADS_Z);

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u partialSumArrLength = gridDim_x * gridDim_y * gridDim_z;
        float *partialSumArr;
        partialSumArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(float), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln1_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           partialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_grid_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           partialSumArr,
                           gridDim_x * gridDim_y,
                           tensorSumArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u partialSumArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *partialSumArr;
        partialSumArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(float), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           partialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           partialSumArr,
                           gridDim_x * gridDim_y,
                           tensorSumArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u partialSumArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *partialSumArr;
        partialSumArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(float), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pkd3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           partialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           partialSumArr,
                           gridDim_x * gridDim_y,
                           tensorSumArr);
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_tensor_sum<half, Rpp32f>(half*,
                                                     RpptDescPtr,
                                                     Rpp32f*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);

template RppStatus hip_exec_tensor_sum<Rpp32f, Rpp32f>(Rpp32f*,
                                                       RpptDescPtr,
                                                       Rpp32f*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);

// -------------------- Set 2 - Kernel Executors --------------------
// Handle U8 datatype
template<>
RppStatus hip_exec_tensor_sum(Rpp8u *srcPtr,
                              RpptDescPtr srcDescPtr,
                              Rpp64u *tensorSumArr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (srcDescPtr->w + 7) >> 3;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    int gridDim_x = (int) ceil((float)globalThreads_x/LOCAL_THREADS_X);
    int gridDim_y = (int) ceil((float)globalThreads_y/LOCAL_THREADS_Y);
    int gridDim_z = (int) ceil((float)globalThreads_z/LOCAL_THREADS_Z);

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u partialSumArrLength = gridDim_x * gridDim_y * gridDim_z;
        Rpp32u *partialSumArr;
        partialSumArr = reinterpret_cast<Rpp32u*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(uint), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln1_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           partialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_grid_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           partialSumArr,
                           gridDim_x * gridDim_y,
                           tensorSumArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u partialSumArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        Rpp32u *partialSumArr;
        partialSumArr = reinterpret_cast<Rpp32u*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32u), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           partialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           partialSumArr,
                           gridDim_x * gridDim_y,
                           tensorSumArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u partialSumArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        Rpp32u *partialSumArr;
        partialSumArr = reinterpret_cast<Rpp32u*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32u), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pkd3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           partialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           partialSumArr,
                           gridDim_x * gridDim_y,
                           tensorSumArr);
    }

    return RPP_SUCCESS;
}

// Handle I8 datatype
template<>
RppStatus hip_exec_tensor_sum(Rpp8s *srcPtr,
                              RpptDescPtr srcDescPtr,
                              Rpp64s *tensorSumArr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (srcDescPtr->w + 7) >> 3;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    int gridDim_x = (int) ceil((float)globalThreads_x/LOCAL_THREADS_X);
    int gridDim_y = (int) ceil((float)globalThreads_y/LOCAL_THREADS_Y);
    int gridDim_z = (int) ceil((float)globalThreads_z/LOCAL_THREADS_Z);

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u partialSumArrLength = gridDim_x * gridDim_y * gridDim_z;
        Rpp32s *partialSumArr;
        partialSumArr = reinterpret_cast<Rpp32s*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32s), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln1_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           partialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_grid_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           partialSumArr,
                           gridDim_x * gridDim_y,
                           tensorSumArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u partialSumArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        Rpp32s *partialSumArr;
        partialSumArr = reinterpret_cast<Rpp32s*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32s), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           partialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           partialSumArr,
                           gridDim_x * gridDim_y,
                           tensorSumArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u partialSumArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        Rpp32s *partialSumArr;
        partialSumArr = reinterpret_cast<Rpp32s*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32s), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pkd3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           partialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           partialSumArr,
                           gridDim_x * gridDim_y,
                           tensorSumArr);
    }

    return RPP_SUCCESS;
}
