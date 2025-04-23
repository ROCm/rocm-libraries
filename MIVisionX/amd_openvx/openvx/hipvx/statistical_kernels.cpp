/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/



#include "hip_common_funcs.h"
#include "hip_host_decls.h"

// ----------------------------------------------------------------------------
// VxThreshold kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U8_U8_Binary(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    int thresholdValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src = *((uint2 *)(&pSrcImage[srcIdx]));
    uint2 dst;

    float4 thr = (float4)hip_unpack0(thresholdValue);
    dst.x = hip_pack((hip_unpack(src.x) - thr) * (float4)256.0f);
    dst.y = hip_pack((hip_unpack(src.y) - thr) * (float4)256.0f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Threshold_U8_U8_Binary(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_int32 thresholdValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U8_U8_Binary, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                       thresholdValue);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U8_U8_Range(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    int thresholdLower, int thresholdUpper) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src = *((uint2 *)(&pSrcImage[srcIdx]));
    uint2 dst;

    float4 thr0 = (float4)(hip_unpack0(thresholdLower) - 1.0f);
    float4 thr1 = (float4)(hip_unpack0(thresholdUpper) + 1.0f);
    float4 pix0 = hip_unpack(src.x);
    float4 pix1 = hip_unpack(src.y);
    dst.x  = hip_pack((pix0 - thr0) * (float4)256.0f);
    dst.x &= hip_pack((thr1 - pix0) * (float4)256.0f);
    dst.y  = hip_pack((pix1 - thr0) * (float4)256.0f);
    dst.y &= hip_pack((thr1 - pix1) * (float4)256.0f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Threshold_U8_U8_Range(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_int32 thresholdLower, vx_int32 thresholdUpper) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U8_U8_Range, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                       thresholdLower, thresholdUpper);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U1_U8_Binary(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    int thresholdValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uint2 src = *((uint2 *)(&pSrcImage[srcIdx]));
    uint2 dst;

    float4 thr = (float4)hip_unpack0(thresholdValue);
    dst.x = hip_pack((hip_unpack(src.x) - thr) * (float4)256.0f);
    dst.y = hip_pack((hip_unpack(src.y) - thr) * (float4)256.0f);

    hip_convert_U1_U8((uchar *)(&pDstImage[dstIdx]), dst);
}
int HipExec_Threshold_U1_U8_Binary(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_int32 thresholdValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U1_U8_Binary, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                       thresholdValue);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U1_U8_Range(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    int thresholdLower, int thresholdUpper) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uint2 src = *((uint2 *)(&pSrcImage[srcIdx]));
    uint2 dst;

    float4 thr0 = (float4)(hip_unpack0(thresholdLower) - 1.0f);
    float4 thr1 = (float4)(hip_unpack0(thresholdUpper) + 1.0f);
    float4 pix0 = hip_unpack(src.x);
    float4 pix1 = hip_unpack(src.y);
    dst.x  = hip_pack((pix0 - thr0) * (float4)256.0f);
    dst.x &= hip_pack((thr1 - pix0) * (float4)256.0f);
    dst.y  = hip_pack((pix1 - thr0) * (float4)256.0f);
    dst.y &= hip_pack((thr1 - pix1) * (float4)256.0f);

    hip_convert_U1_U8((uchar *)(&pDstImage[dstIdx]), dst);
}
int HipExec_Threshold_U1_U8_Range(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_int32 thresholdLower, vx_int32 thresholdUpper) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U1_U8_Range, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                       thresholdLower, thresholdUpper);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U8_S16_Binary(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint thresholdValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    int4 src = *((int4 *)(&pSrcImage[srcIdx]));
    int4 dst;

    short2 p;
    float4 thr = (float4)hip_unpack0(thresholdValue);
    p.x = ((((int)src.x)  << 16) >> 16) & 0xffff;
    p.y = ((int)src.x >> 16) & 0xffff;
    dst.x = (p.x > thr.x) ? 0xffff:0;
    dst.x |= ((p.y > thr.x) ? 0xffff0000:0);
    p.x = ((((int)src.y)  << 16) >> 16) & 0xffff;
    p.y = ((int)src.y >> 16) & 0xffff;
    dst.y = (p.x > thr.x) ? 0xffff:0;
    dst.y |= ((p.y > thr.x) ? 0xffff0000:0);
    p.x = ((((int)src.z)  << 16) >> 16) & 0xffff;
    p.y = ((int)src.z >> 16) & 0xffff;
    dst.z = (p.x > thr.x) ? 0xffff:0;
    dst.z |= ((p.y > thr.x) ? 0xffff0000:0);
    p.x = ((((int)src.w)  << 16) >> 16) & 0xffff;
    p.y = ((int)src.w >> 16) & 0xffff;
    dst.w = (p.x > thr.x) ? 0xffff:0;
    dst.w |= ((p.y > thr.x) ? 0xffff0000:0);

    hip_convert_U8_S16((uint2 *)(&pDstImage[dstIdx]), dst);
}
int HipExec_Threshold_U8_S16_Binary(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_int16 thresholdValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U8_S16_Binary, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                       (uint)thresholdValue);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Threshold_U8_S16_Range(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    int thresholdLower, int thresholdUpper) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    int4 src = *((int4 *)(&pSrcImage[srcIdx]));
    int4 dst;

    float4 thr0 = (float4)(hip_unpack0(thresholdLower) - 1.0f);
    float4 thr1 = (float4)(hip_unpack0(thresholdUpper) + 1.0f);
    short2 p;
    p.x = ((((int)src.x)  << 16) >> 16) & 0xffff;
    p.y = ((int)src.x >> 16) & 0xffff;
    dst.x = (p.x > thr0.x && thr1.x > p.x) ? 0xffff:0;
    dst.x |= ((p.y > thr0.x && thr1.x > p.y) ? 0xffff0000:0);
    p.x = ((((int)src.y)  << 16) >> 16) & 0xffff;
    p.y = ((int)src.y >> 16) & 0xffff;
    dst.y = (p.x > thr0.x && thr1.x > p.x) ? 0xffff:0;
    dst.y |= ((p.y > thr0.x && thr1.x > p.y) ? 0xffff0000:0);
    p.x = ((((int)src.z)  << 16) >> 16) & 0xffff;
    p.y = ((int)src.z >> 16) & 0xffff;
    dst.z = (p.x > thr0.x && thr1.x > p.x) ? 0xffff:0;
    dst.z |= ((p.y > thr0.x && thr1.x > p.y) ? 0xffff0000:0);
    p.x = ((((int)src.w)  << 16) >> 16) & 0xffff;
    p.y = ((int)src.w >> 16) & 0xffff;
    dst.w = (p.x > thr0.x && thr1.x > p.x) ? 0xffff:0;
    dst.w |= ((p.y > thr0.x && thr1.x > p.y) ? 0xffff0000:0);

    hip_convert_U8_S16((uint2 *)(&pDstImage[dstIdx]), dst);
}
int HipExec_Threshold_U8_S16_Range(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_int16 thresholdLower, vx_int16 thresholdUpper) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Threshold_U8_S16_Range, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                       (int)thresholdLower, (int)thresholdUpper);

    return VX_SUCCESS;
}