/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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

// Coefficients for RGB to YCbCr Conversion
__device__ constexpr float coeffYR = 0.299f;
__device__ constexpr float coeffYG = 0.587f;
__device__ constexpr float coeffYB = 0.114f;

__device__ constexpr float coeffCbR = -0.168736f;
__device__ constexpr float coeffCbG = -0.331264f;
__device__ constexpr float coeffCbB = 0.5f;

__device__ constexpr float coeffCrR = 0.5f;
__device__ constexpr float coeffCrG = -0.418688f;
__device__ constexpr float coeffCrB = -0.081312f;

// Coefficients for YCbCr to RGB Conversion
__device__ constexpr float coeffRCr = 1.402f;
__device__ constexpr float coeffGCb = 0.344136f;
__device__ constexpr float coeffGCr = 0.714136f;
__device__ constexpr float coeffBCb = 1.772f;

__device__ constexpr float maxPixelVal = 255.0f;
__device__ constexpr float chromaOffset = 128.0f;
__device__ constexpr int HISTOGRAM_BINS = 256;

// Vectorized coefficients for RGB to YCbCr
__device__ const float4 pCoeffYR_f4 = {coeffYR, coeffYR, coeffYR, coeffYR};
__device__ const float4 pCoeffYG_f4 = {coeffYG, coeffYG, coeffYG, coeffYG};
__device__ const float4 pCoeffYB_f4 = {coeffYB, coeffYB, coeffYB, coeffYB};

__device__ const float4 pCoeffCbR_f4 = {coeffCbR, coeffCbR, coeffCbR, coeffCbR};
__device__ const float4 pCoeffCbG_f4 = {coeffCbG, coeffCbG, coeffCbG, coeffCbG};
__device__ const float4 pCoeffCbB_f4 = {coeffCbB, coeffCbB, coeffCbB, coeffCbB};

__device__ const float4 pCoeffCrR_f4 = {coeffCrR, coeffCrR, coeffCrR, coeffCrR};
__device__ const float4 pCoeffCrG_f4 = {coeffCrG, coeffCrG, coeffCrG, coeffCrG};
__device__ const float4 pCoeffCrB_f4 = {coeffCrB, coeffCrB, coeffCrB, coeffCrB};

__device__ const float4 pChromaOffset_f4 = {chromaOffset, chromaOffset, chromaOffset, chromaOffset};

// Clamp float4 values to specified range
__device__ inline float4 clamp_f4(float4 v, float lo, float hi)
{
    v.x = fminf(fmaxf(v.x, lo), hi);
    v.y = fminf(fmaxf(v.y, lo), hi);
    v.z = fminf(fmaxf(v.z, lo), hi);
    v.w = fminf(fmaxf(v.w, lo), hi);
    return v;
}

// RGB to YCbCr conversion for 8 pixels
__device__ inline void rgb_to_ycbcr_hip_compute(d_float24 &rgb_f24, d_float8 &y_f8, d_float8 &cb_f8, d_float8 &cr_f8)
{
    y_f8.f4[0] = clamp_f4((rgb_f24.f4[0] * pCoeffYR_f4) + (rgb_f24.f4[2] * pCoeffYG_f4) + (rgb_f24.f4[4] * pCoeffYB_f4), 0.0f, maxPixelVal);
    y_f8.f4[1] = clamp_f4((rgb_f24.f4[1] * pCoeffYR_f4) + (rgb_f24.f4[3] * pCoeffYG_f4) + (rgb_f24.f4[5] * pCoeffYB_f4), 0.0f, maxPixelVal);

    cb_f8.f4[0] = clamp_f4((rgb_f24.f4[0] * pCoeffCbR_f4) + (rgb_f24.f4[2] * pCoeffCbG_f4) + (rgb_f24.f4[4] * pCoeffCbB_f4) + pChromaOffset_f4, 0.0f, maxPixelVal);
    cb_f8.f4[1] = clamp_f4((rgb_f24.f4[1] * pCoeffCbR_f4) + (rgb_f24.f4[3] * pCoeffCbG_f4) + (rgb_f24.f4[5] * pCoeffCbB_f4) + pChromaOffset_f4, 0.0f, maxPixelVal);

    cr_f8.f4[0] = clamp_f4((rgb_f24.f4[0] * pCoeffCrR_f4) + (rgb_f24.f4[2] * pCoeffCrG_f4) + (rgb_f24.f4[4] * pCoeffCrB_f4) + pChromaOffset_f4, 0.0f, maxPixelVal);
    cr_f8.f4[1] = clamp_f4((rgb_f24.f4[1] * pCoeffCrR_f4) + (rgb_f24.f4[3] * pCoeffCrG_f4) + (rgb_f24.f4[5] * pCoeffCrB_f4) + pChromaOffset_f4, 0.0f, maxPixelVal);
}

// YCbCr to RGB conversion for 8 pixels
__device__ inline void ycbcr_to_rgb_hip_compute(d_float24 &rgb_f24, d_float8 &y_f8, d_float8 &cb_f8, d_float8 &cr_f8)
{
    cb_f8.f4[0] -= MAKE_FLOAT4(chromaOffset);
    cb_f8.f4[1] -= MAKE_FLOAT4(chromaOffset);
    cr_f8.f4[0] -= MAKE_FLOAT4(chromaOffset);
    cr_f8.f4[1] -= MAKE_FLOAT4(chromaOffset);

    rgb_f24.f4[0] = clamp_f4((y_f8.f4[0] + MAKE_FLOAT4(coeffRCr) * cr_f8.f4[0]), 0.0f, maxPixelVal);
    rgb_f24.f4[1] = clamp_f4((y_f8.f4[1] + MAKE_FLOAT4(coeffRCr) * cr_f8.f4[1]), 0.0f, maxPixelVal);

    rgb_f24.f4[2] = clamp_f4((y_f8.f4[0] - (MAKE_FLOAT4(coeffGCb) * cb_f8.f4[0]) - (MAKE_FLOAT4(coeffGCr) * cr_f8.f4[0])), 0.0f, maxPixelVal);
    rgb_f24.f4[3] = clamp_f4((y_f8.f4[1] - (MAKE_FLOAT4(coeffGCb) * cb_f8.f4[1]) - (MAKE_FLOAT4(coeffGCr) * cr_f8.f4[1])), 0.0f, maxPixelVal);

    rgb_f24.f4[4] = clamp_f4((y_f8.f4[0] + MAKE_FLOAT4(coeffBCb) * cb_f8.f4[0]), 0.0f, maxPixelVal);
    rgb_f24.f4[5] = clamp_f4((y_f8.f4[1] + MAKE_FLOAT4(coeffBCb) * cb_f8.f4[1]), 0.0f, maxPixelVal);
}

// Histogram collection kernel
__global__ void collect_hist_pln_hip_tensor(const unsigned char *__restrict__ srcPtr,
                                            RpptROIPtr roiTensorPtrSrc,
                                            ulong2 srcStridesNH,
                                            unsigned int *__restrict__ hist,
                                            int batchSize)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(id_z >= batchSize)
        return;

    uint histOffset = id_z * HISTOGRAM_BINS;
    RpptRoiXywh roi = roiTensorPtrSrc[id_z].xywhROI;

    int totalThreads = blockDim.x * blockDim.y * blockDim.z;
    int linearTid = (hipThreadIdx_z * blockDim.y * blockDim.x) + (hipThreadIdx_y * blockDim.x) + hipThreadIdx_x;

    __shared__ unsigned int histShared[HISTOGRAM_BINS];

    for(int i = linearTid; i < HISTOGRAM_BINS; i += totalThreads)
        histShared[i] = 0;
    __syncthreads();

    bool withinBounds = (id_y < roi.roiHeight) && (id_x < roi.roiWidth);
    if(withinBounds)
    {
        uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roi.xy.y) * srcStridesNH.y) + (id_x + roi.xy.x);
        uint8_t pixVal = srcPtr[srcIdx];
        atomicAdd(&histShared[pixVal], 1);
    }
    __syncthreads();

    for(int i = linearTid; i < HISTOGRAM_BINS; i += totalThreads)
    {
        unsigned int count = histShared[i];
        if(count)
            atomicAdd(&hist[histOffset + i], count);
    }
}

// LUT building kernel
__global__ void build_lut_from_hist_kernel(const unsigned int *__restrict__ hist,
                                           unsigned char *__restrict__ lut,
                                           RpptROIPtr roiTensorPtrSrc,
                                           int batchSize)
{
    int batch = blockIdx.x;
    if(batch >= batchSize)
        return;

    int tid = threadIdx.x;
    __shared__ unsigned int cdfShared[HISTOGRAM_BINS];
    __shared__ unsigned int minCdfShared;

    if(tid == 0)
        minCdfShared = 0;
    __syncthreads();

    // Load histogram values directly into shared memory (no need to zero first since all values are overwritten)
    for(int i = tid; i < HISTOGRAM_BINS; i += blockDim.x)
        cdfShared[i] = hist[batch * HISTOGRAM_BINS + i];
    __syncthreads();

    if(tid == 0)
    {
        unsigned int cdfAccum = 0;
        for(int i = 0; i < HISTOGRAM_BINS; i++)
        {
            cdfAccum += cdfShared[i];
            cdfShared[i] = cdfAccum;
            if(!minCdfShared && cdfShared[i])
                minCdfShared = cdfShared[i];
        }
    }
    __syncthreads();

    int roiWidth = roiTensorPtrSrc[batch].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[batch].xywhROI.roiHeight;
    int numPixels = roiWidth * roiHeight;

    for(int i = tid; i < HISTOGRAM_BINS; i += blockDim.x)
    {
        float denominator = fmaxf(static_cast<float>(numPixels - minCdfShared), 1.0f);
        unsigned char equalizedVal = static_cast<unsigned char>(roundf(static_cast<float>((cdfShared[i] - minCdfShared) * maxPixelVal) / denominator));

        unsigned char isUniform = (minCdfShared == numPixels);
        lut[batch * HISTOGRAM_BINS + i] = isUniform * i + (1 - isUniform) * equalizedVal;
    }
}

// LUT application kernel
__global__ void apply_lut_pln1_hip_tensor(const unsigned char *__restrict__ srcPtr,
                                          ulong2 srcStridesNH,
                                          unsigned char *__restrict__ dstPtr,
                                          ulong2 dstStridesNH,
                                          const unsigned char *__restrict__ lut,
                                          RpptROIPtr roiTensorPtrSrc,
                                          int batchSize)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(id_z >= batchSize)
        return;

    if((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    unsigned char pixVal = srcPtr[srcIdx];
    dstPtr[dstIdx] = lut[id_z * HISTOGRAM_BINS + pixVal];
}

// RGB PKD3 to YCbCr PLN3 conversion kernel
// Note: This kernel processes 8 pixels per thread. The buffer is sized for the full image dimensions,
// ensuring sufficient memory even when roiWidth is not a multiple of 8. Edge pixels beyond roiWidth
// but within buffer bounds are processed but will be overwritten or unused.
__global__ void convert_rgb_pkd3_to_ycbcr_pln3(unsigned char *__restrict__ srcPtr,
                                               uint2 srcStridesNH,
                                               unsigned char *__restrict__ yPtr,
                                               unsigned char *__restrict__ cbPtr,
                                               unsigned char *__restrict__ crPtr,
                                               uint2 dstStridesWH,
                                               RpptROIPtr roiTensorPtrSrc,
                                               int batchSize)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(id_z >= batchSize)
        return;

    // Early exit if starting position is outside ROI bounds
    if((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesWH.y * dstStridesWH.x) + (id_y * dstStridesWH.x) + id_x;

    d_float24 rgb_f24;
    d_float8 y_f8, cb_f8, cr_f8;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &rgb_f24);
    rgb_to_ycbcr_hip_compute(rgb_f24, y_f8, cb_f8, cr_f8);

    rpp_hip_pack_float8_and_store8(yPtr + dstIdx, &y_f8);
    rpp_hip_pack_float8_and_store8(cbPtr + dstIdx, &cb_f8);
    rpp_hip_pack_float8_and_store8(crPtr + dstIdx, &cr_f8);
}

// RGB PLN3 to YCbCr PLN3 conversion kernel
// Note: This kernel processes 8 pixels per thread. The buffer is sized for the full image dimensions,
// ensuring sufficient memory even when roiWidth is not a multiple of 8.
__global__ void convert_rgb_pln3_to_ycbcr_pln3(unsigned char *__restrict__ srcPtr,
                                               uint3 srcStridesNCH,
                                               unsigned char *__restrict__ yPtr,
                                               unsigned char *__restrict__ cbPtr,
                                               unsigned char *__restrict__ crPtr,
                                               ulong2 dstStridesNH,
                                               RpptROIPtr roiTensorPtrSrc,
                                               int batchSize)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(id_z >= batchSize)
        return;

    // Early exit if starting position is outside ROI bounds
    if((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    d_float24 rgb_f24;
    d_float8 y_f8, cb_f8, cr_f8;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &rgb_f24);
    rgb_to_ycbcr_hip_compute(rgb_f24, y_f8, cb_f8, cr_f8);

    rpp_hip_pack_float8_and_store8(yPtr + dstIdx, &y_f8);
    rpp_hip_pack_float8_and_store8(cbPtr + dstIdx, &cb_f8);
    rpp_hip_pack_float8_and_store8(crPtr + dstIdx, &cr_f8);
}

// YCbCr PLN3 to RGB PLN3 conversion kernel
// Note: This kernel processes 8 pixels per thread. The buffer is sized for the full image dimensions,
// ensuring sufficient memory even when roiWidth is not a multiple of 8.
__global__ void convert_ycbcr_pln3_to_rgb_pln3(unsigned char *__restrict__ yPtr,
                                               unsigned char *__restrict__ cbPtr,
                                               unsigned char *__restrict__ crPtr,
                                               uint2 srcStridesWH,
                                               unsigned char *__restrict__ dstPtr,
                                               uint3 dstStridesNCH,
                                               RpptROIPtr roiTensorPtrSrc,
                                               int batchSize)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(id_z >= batchSize)
        return;

    // Early exit if starting position is outside ROI bounds
    if((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesWH.y * srcStridesWH.x) + (id_y * srcStridesWH.x) + id_x;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 rgb_f24;
    d_float8 y_f8, cb_f8, cr_f8;

    rpp_hip_load8_and_unpack_to_float8(yPtr + srcIdx, &y_f8);
    rpp_hip_load8_and_unpack_to_float8(cbPtr + srcIdx, &cb_f8);
    rpp_hip_load8_and_unpack_to_float8(crPtr + srcIdx, &cr_f8);

    ycbcr_to_rgb_hip_compute(rgb_f24, y_f8, cb_f8, cr_f8);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &rgb_f24);
}

// YCbCr PLN3 to RGB PKD3 conversion kernel
// Note: This kernel processes 8 pixels per thread. The buffer is sized for the full image dimensions,
// ensuring sufficient memory even when roiWidth is not a multiple of 8.
__global__ void convert_ycbcr_pln3_to_rgb_pkd3(unsigned char *__restrict__ yPtr,
                                               unsigned char *__restrict__ cbPtr,
                                               unsigned char *__restrict__ crPtr,
                                               uint2 srcStridesWH,
                                               unsigned char *__restrict__ dstPtr,
                                               uint2 dstStridesNH,
                                               RpptROIPtr roiTensorPtrSrc,
                                               int batchSize)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(id_z >= batchSize)
        return;

    // Early exit if starting position is outside ROI bounds
    if((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesWH.y * srcStridesWH.x) + (id_y * srcStridesWH.x) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    d_float24 rgb_f24;
    d_float8 y_f8, cb_f8, cr_f8;

    rpp_hip_load8_and_unpack_to_float8(yPtr + srcIdx, &y_f8);
    rpp_hip_load8_and_unpack_to_float8(cbPtr + srcIdx, &cb_f8);
    rpp_hip_load8_and_unpack_to_float8(crPtr + srcIdx, &cr_f8);

    ycbcr_to_rgb_hip_compute(rgb_f24, y_f8, cb_f8, cr_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &rgb_f24);
}

RppStatus hip_exec_histogram_equalize_tensor(Rpp8u *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8u *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             rpp::Handle& handle)
{
    if(roiType == RpptRoiType::LTRB)
        hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int batchSize = dstDescPtr->n;

    // Calculate required scratch buffer size
    // Layout: [d_hist | d_lut | yuvBuf (for 3-channel only)]
    size_t histSize = batchSize * HISTOGRAM_BINS * sizeof(unsigned int);
    size_t lutSize = batchSize * HISTOGRAM_BINS * sizeof(unsigned char);
    size_t yuvSize = (srcDescPtr->c == 3) ? (3 * static_cast<size_t>(srcDescPtr->w) * static_cast<size_t>(srcDescPtr->h) * static_cast<size_t>(srcDescPtr->n)) : 0;
    size_t requiredSize = histSize + lutSize + yuvSize;

    // Pre-allocated scratch buffer size from handle (sizeof(Rpp32f) * 8294400)
    constexpr size_t SCRATCH_BUFFER_SIZE = sizeof(Rpp32f) * 8294400;

    // Use handle's pre-allocated scratch buffer if sufficient, otherwise reallocate overflow buffer
    Rpp8u *scratchBuffer;
    if(requiredSize <= SCRATCH_BUFFER_SIZE)
        scratchBuffer = reinterpret_cast<Rpp8u*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
    else
        // Reallocate overflow buffer if needed
        RPP_HIP_RETURN_IF_ERROR(hipMalloc(&scratchBuffer, requiredSize));

    unsigned int *d_hist = reinterpret_cast<unsigned int*>(scratchBuffer);
    unsigned char *d_lut = reinterpret_cast<unsigned char*>(scratchBuffer + histSize);

    if(srcDescPtr->c == 3)
    {
        // Use size_t for all intermediate calculations to prevent overflow for large images
        const size_t planeSize = static_cast<size_t>(srcDescPtr->w) * static_cast<size_t>(srcDescPtr->h) * static_cast<size_t>(srcDescPtr->n);

        Rpp8u *yuvBuf = scratchBuffer + histSize + lutSize;

        Rpp8u *yBuf = yuvBuf;
        Rpp8u *cbBuf = yuvBuf + planeSize;
        Rpp8u *crBuf = yuvBuf + (planeSize * 2);

        if(srcDescPtr->layout == RpptLayout::NHWC)
        {
            int globalThreads_x = (srcDescPtr->w + 7) >> 3;
            int globalThreads_y = dstDescPtr->h;
            int globalThreads_z = handle.GetBatchSize();

            // Compute YCbCr buffer strides using 64-bit types to prevent overflow for large images
            size_t yuvNStride = static_cast<size_t>(srcDescPtr->w) * static_cast<size_t>(srcDescPtr->h);
            size_t yuvHStride = static_cast<size_t>(srcDescPtr->w);

            hipLaunchKernelGGL(convert_rgb_pkd3_to_ycbcr_pln3,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0, handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               yBuf, cbBuf, crBuf,
                               make_uint2(srcDescPtr->w, srcDescPtr->h),
                               roiTensorPtrSrc,
                               batchSize);

            RPP_HIP_RETURN_IF_ERROR(hipMemsetAsync(d_hist, 0, batchSize * HISTOGRAM_BINS * sizeof(unsigned int), handle.GetStream()));

            globalThreads_x = srcDescPtr->w;
            hipLaunchKernelGGL(collect_hist_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0, handle.GetStream(),
                               yBuf, roiTensorPtrSrc,
                               make_ulong2(yuvNStride, yuvHStride),
                               d_hist,
                               batchSize);

            hipLaunchKernelGGL(build_lut_from_hist_kernel,
                               dim3(batchSize), dim3(HISTOGRAM_BINS),
                               0, handle.GetStream(),
                               d_hist, d_lut, roiTensorPtrSrc, batchSize);

            globalThreads_x = dstDescPtr->w;
            globalThreads_y = dstDescPtr->h;
            globalThreads_z = dstDescPtr->n;
            hipLaunchKernelGGL(apply_lut_pln1_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0, handle.GetStream(),
                               yBuf,
                               make_ulong2(yuvNStride, yuvHStride),
                               yBuf,
                               make_ulong2(yuvNStride, yuvHStride),
                               d_lut,
                               roiTensorPtrSrc,
                               batchSize);

            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            globalThreads_y = dstDescPtr->h;
            globalThreads_z = handle.GetBatchSize();

            if(dstDescPtr->layout == RpptLayout::NHWC)
            {
                hipLaunchKernelGGL(convert_ycbcr_pln3_to_rgb_pkd3,
                                   dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0, handle.GetStream(),
                                   yBuf, cbBuf, crBuf,
                                   make_uint2(srcDescPtr->w, srcDescPtr->h),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   roiTensorPtrSrc,
                                   batchSize);
            }
            else if(dstDescPtr->layout == RpptLayout::NCHW)
            {
                hipLaunchKernelGGL(convert_ycbcr_pln3_to_rgb_pln3,
                                   dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0, handle.GetStream(),
                                   yBuf, cbBuf, crBuf,
                                   make_uint2(srcDescPtr->w, srcDescPtr->h),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   roiTensorPtrSrc,
                                   batchSize);
            }
        }
        else if(srcDescPtr->layout == RpptLayout::NCHW)
        {
            int globalThreads_x = (srcDescPtr->w + 7) >> 3;
            int globalThreads_y = srcDescPtr->h;
            int globalThreads_z = srcDescPtr->n;

            // Compute YCbCr buffer strides using 64-bit types to prevent overflow for large images
            size_t yuvNStride = static_cast<size_t>(srcDescPtr->w) * static_cast<size_t>(srcDescPtr->h);
            size_t yuvHStride = static_cast<size_t>(srcDescPtr->w);

            hipLaunchKernelGGL(convert_rgb_pln3_to_ycbcr_pln3,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0, handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               yBuf, cbBuf, crBuf,
                               make_ulong2(yuvNStride, yuvHStride),
                               roiTensorPtrSrc, batchSize);

            RPP_HIP_RETURN_IF_ERROR(hipMemsetAsync(d_hist, 0, batchSize * HISTOGRAM_BINS * sizeof(unsigned int), handle.GetStream()));

            globalThreads_x = srcDescPtr->w;
            hipLaunchKernelGGL(collect_hist_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0, handle.GetStream(),
                               yBuf, roiTensorPtrSrc,
                               make_ulong2(yuvNStride, yuvHStride),
                               d_hist,
                               batchSize);

            hipLaunchKernelGGL(build_lut_from_hist_kernel,
                               dim3(batchSize), dim3(HISTOGRAM_BINS),
                               0, handle.GetStream(),
                               d_hist, d_lut, roiTensorPtrSrc, batchSize);

            globalThreads_x = dstDescPtr->w;
            globalThreads_y = dstDescPtr->h;
            globalThreads_z = dstDescPtr->n;
            hipLaunchKernelGGL(apply_lut_pln1_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0, handle.GetStream(),
                               yBuf,
                               make_ulong2(yuvNStride, yuvHStride),
                               yBuf,
                               make_ulong2(yuvNStride, yuvHStride),
                               d_lut, roiTensorPtrSrc,
                               batchSize);

            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            globalThreads_y = dstDescPtr->h;
            globalThreads_z = dstDescPtr->n;

            if(dstDescPtr->layout == RpptLayout::NHWC)
            {
                hipLaunchKernelGGL(convert_ycbcr_pln3_to_rgb_pkd3,
                                   dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0, handle.GetStream(),
                                   yBuf, cbBuf, crBuf,
                                   make_uint2(srcDescPtr->w, srcDescPtr->h),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   roiTensorPtrSrc,
                                   batchSize);
            }
            else if(dstDescPtr->layout == RpptLayout::NCHW)
            {
                hipLaunchKernelGGL(convert_ycbcr_pln3_to_rgb_pln3,
                                   dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0, handle.GetStream(),
                                   yBuf, cbBuf, crBuf,
                                   make_uint2(srcDescPtr->w, srcDescPtr->h),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   roiTensorPtrSrc,
                                   batchSize);
            }
        }

        return RPP_SUCCESS;
    }

    // Single channel processing
    RPP_HIP_RETURN_IF_ERROR(hipMemsetAsync(d_hist, 0, batchSize * HISTOGRAM_BINS * sizeof(unsigned int), handle.GetStream()));

    int globalThreads_x = srcDescPtr->w;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = srcDescPtr->n;

    hipLaunchKernelGGL(collect_hist_pln_hip_tensor,
                       dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0, handle.GetStream(),
                       srcPtr, roiTensorPtrSrc,
                       make_ulong2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                       d_hist,
                       batchSize);

    hipLaunchKernelGGL(build_lut_from_hist_kernel,
                       dim3(batchSize), dim3(HISTOGRAM_BINS),
                       0, handle.GetStream(),
                       d_hist, d_lut, roiTensorPtrSrc, batchSize);

    globalThreads_x = dstDescPtr->w;
    globalThreads_y = dstDescPtr->h;
    globalThreads_z = dstDescPtr->n;

    hipLaunchKernelGGL(apply_lut_pln1_hip_tensor,
                       dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0, handle.GetStream(),
                       srcPtr,
                       make_ulong2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                       dstPtr,
                       make_ulong2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                       d_lut,
                       roiTensorPtrSrc,
                       batchSize);

    return RPP_SUCCESS;
}
