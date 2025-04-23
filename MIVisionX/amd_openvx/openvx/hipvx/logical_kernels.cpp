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
// VxAnd kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_And_U8_U8U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));

    *((uint2 *)(&pDstImage[dstIdx])) = src1 & src2;
}
int HipExec_And_U8_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U8_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U8_U8U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uchar src2 = *((uchar *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    hip_convert_U8_U1(&dst, src2);
    dst = src1 & dst;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_And_U8_U8U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U8_U8U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U8_U1U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    hip_convert_U8_U1(&dst, src1);
    dst = src2 & dst;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_And_U8_U1U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U8_U1U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U8_U1U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint src2Idx = y * srcImage2StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uchar src2 = *((uchar *)(&pSrcImage2[src2Idx]));
    uint2 dst1, dst2;

    hip_convert_U8_U1(&dst1, src1);
    hip_convert_U8_U1(&dst2, src2);
    dst1 = dst1 & dst2;

    *((uint2 *)(&pDstImage[dstIdx])) = dst1;
}
int HipExec_And_U8_U1U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U8_U1U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U1_U8U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uchar dst;

    src1 = src1 & src2;
    hip_convert_U1_U8(&dst, src1);

    *((uchar *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_And_U1_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U1_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U1_U8U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uchar src2 = *((uchar *)(&pSrcImage2[src2Idx]));
    uchar dst;

    hip_convert_U1_U8(&dst, src1);
    dst = dst & src2;

    *((uchar *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_And_U1_U8U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U1_U8U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U1_U1U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uchar dst;

    hip_convert_U1_U8(&dst, src2);
    dst = dst & src1;

    *((uchar *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_And_U1_U1U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U1_U1U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_And_U1_U1U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint src2Idx = y * srcImage2StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uchar src2 = *((uchar *)(&pSrcImage2[src2Idx]));

    *((uchar *)(&pDstImage[dstIdx])) = src1 & src2;
}
int HipExec_And_U1_U1U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_And_U1_U1U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxOr kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Or_U8_U8U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));

    *((uint2 *)(&pDstImage[dstIdx])) = src1 | src2;
}
int HipExec_Or_U8_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U8_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U8_U8U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uchar src2 = *((uchar *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    hip_convert_U8_U1(&dst, src2);
    dst = src1 | dst;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Or_U8_U8U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U8_U8U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U8_U1U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    hip_convert_U8_U1(&dst, src1);
    dst = src2 | dst;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Or_U8_U1U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U8_U1U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U8_U1U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint src2Idx = y * srcImage2StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uchar src2 = *((uchar *)(&pSrcImage2[src2Idx]));
    uint2 dst1, dst2;

    hip_convert_U8_U1(&dst1, src1);
    hip_convert_U8_U1(&dst2, src2);
    dst1 = dst1 | dst2;

    *((uint2 *)(&pDstImage[dstIdx])) = dst1;
}
int HipExec_Or_U8_U1U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U8_U1U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U1_U8U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uchar dst;

    src1 = src1 | src2;
    hip_convert_U1_U8(&dst, src1);

    *((uchar *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Or_U1_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U1_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U1_U8U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uchar src2 = *((uchar *)(&pSrcImage2[src2Idx]));
    uchar dst;

    hip_convert_U1_U8(&dst, src1);
    dst = dst | src2;

    *((uchar *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Or_U1_U8U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U1_U8U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U1_U1U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uchar dst;

    hip_convert_U1_U8(&dst, src2);
    dst = dst | src1;

    *((uchar *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Or_U1_U1U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U1_U1U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Or_U1_U1U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint src2Idx = y * srcImage2StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uchar src2 = *((uchar *)(&pSrcImage2[src2Idx]));

    *((uchar *)(&pDstImage[dstIdx])) = src1 | src2;
}
int HipExec_Or_U1_U1U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Or_U1_U1U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxXor kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Xor_U8_U8U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));

    *((uint2 *)(&pDstImage[dstIdx])) = src1 ^ src2;
}
int HipExec_Xor_U8_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U8_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U8_U8U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uchar src2 = *((uchar *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    hip_convert_U8_U1(&dst, src2);
    dst = src1 ^ dst;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Xor_U8_U8U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U8_U8U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U8_U1U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 dst;

    hip_convert_U8_U1(&dst, src1);
    dst = src2 ^ dst;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Xor_U8_U1U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U8_U1U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U8_U1U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint src2Idx = y * srcImage2StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uchar src2 = *((uchar *)(&pSrcImage2[src2Idx]));
    uint2 dst1, dst2;

    hip_convert_U8_U1(&dst1, src1);
    hip_convert_U8_U1(&dst2, src2);
    dst1 = dst1 ^ dst2;

    *((uint2 *)(&pDstImage[dstIdx])) = dst1;
}
int HipExec_Xor_U8_U1U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U8_U1U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U1_U8U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uchar dst;

    src1 = src1 ^ src2;
    hip_convert_U1_U8(&dst, src1);

    *((uchar *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Xor_U1_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U1_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U1_U8U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uchar src2 = *((uchar *)(&pSrcImage2[src2Idx]));
    uchar dst;

    hip_convert_U1_U8(&dst, src1);
    dst = dst ^ src2;

    *((uchar *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Xor_U1_U8U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U1_U8U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U1_U1U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uchar dst;

    hip_convert_U1_U8(&dst, src2);
    dst = dst ^ src1;

    *((uchar *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Xor_U1_U1U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U1_U1U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Xor_U1_U1U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint src2Idx = y * srcImage2StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uchar src2 = *((uchar *)(&pSrcImage2[src2Idx]));

    *((uchar *)(&pDstImage[dstIdx])) = src1 ^ src2;
}
int HipExec_Xor_U1_U1U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Xor_U1_U1U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxNot kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Not_U8_U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    *((uint2 *)(&pDstImage[dstIdx])) = ~src1;
}
int HipExec_Not_U8_U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Not_U8_U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Not_U8_U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uint2 dst;

    hip_convert_U8_U1(&dst, src1);
    dst = ~dst;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Not_U8_U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Not_U8_U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Not_U1_U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uchar dst;

    hip_convert_U1_U8(&dst, src1);
    dst = ~dst;

    *((uchar *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Not_U1_U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Not_U1_U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Not_U1_U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uchar src1 = *((uchar *)(&pSrcImage1[src1Idx]));
    uchar dst;

    dst = ~src1;

    *((uchar *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Not_U1_U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Not_U1_U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}