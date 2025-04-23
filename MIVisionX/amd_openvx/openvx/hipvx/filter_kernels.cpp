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
// VxBox kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Box_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, 1.111111119390e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, 1.111111119390e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, 1.111111119390e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, 1.111111119390e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, 1.111111119390e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, 1.111111119390e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, 1.111111119390e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, 1.111111119390e-01f, sum.data[7]);

    sum.data[0] += -4.999899864197e-01f;
    sum.data[1] += -4.999899864197e-01f;
    sum.data[2] += -4.999899864197e-01f;
    sum.data[3] += -4.999899864197e-01f;
    sum.data[4] += -4.999899864197e-01f;
    sum.data[5] += -4.999899864197e-01f;
    sum.data[6] += -4.999899864197e-01f;
    sum.data[7] += -4.999899864197e-01f;

    uint2 dst;
    dst.x = hip_pack(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = hip_pack(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx =  y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Box_U8_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Box_U8_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxDilate kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Dilate_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum;
    uint4 pix;
    float4 val;
    uint2 *pixLoc0 = (uint2*)&pix.x;
    uint2 *pixLoc2 = (uint2*)&pix.z;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    *pixLoc0 = lbufptr[0];
    *pixLoc2 = lbufptr[1];
    val.x = hip_unpack3(pix.x);
    val.y = hip_unpack0(pix.y);
    val.z = hip_unpack1(pix.y);
    sum.data[0] = hip_max3(val.x, val.y, val.z);
    val.x = hip_unpack2(pix.y);
    sum.data[1] = hip_max3(val.x, val.y, val.z);
    val.y = hip_unpack3(pix.y);
    sum.data[2] = hip_max3(val.x, val.y, val.z);
    val.z = hip_unpack0(pix.z);
    sum.data[3] = hip_max3(val.x, val.y, val.z);
    val.x = hip_unpack1(pix.z);
    sum.data[4] = hip_max3(val.x, val.y, val.z);
    val.y = hip_unpack2(pix.z);
    sum.data[5] = hip_max3(val.x, val.y, val.z);
    val.z = hip_unpack3(pix.z);
    sum.data[6] = hip_max3(val.x, val.y, val.z);
    val.x = hip_unpack0(pix.w);
    sum.data[7] = hip_max3(val.x, val.y, val.z);
    *pixLoc0 = lbufptr[17];
    *pixLoc2 = lbufptr[18];
    val.x = hip_unpack3(pix.x);
    val.y = hip_unpack0(pix.y);
    val.z = hip_unpack1(pix.y);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[0] = max(sum.data[0], val.w);
    val.x = hip_unpack2(pix.y);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[1] = max(sum.data[1], val.w);
    val.y = hip_unpack3(pix.y);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[2] = max(sum.data[2], val.w);
    val.z = hip_unpack0(pix.z);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[3] = max(sum.data[3], val.w);
    val.x = hip_unpack1(pix.z);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[4] = max(sum.data[4], val.w);
    val.y = hip_unpack2(pix.z);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[5] = max(sum.data[5], val.w);
    val.z = hip_unpack3(pix.z);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[6] = max(sum.data[6], val.w);
    val.x = hip_unpack0(pix.w);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[7] = max(sum.data[7], val.w);
    *pixLoc0 = lbufptr[34];
    *pixLoc2 = lbufptr[35];
    val.x = hip_unpack3(pix.x);
    val.y = hip_unpack0(pix.y);
    val.z = hip_unpack1(pix.y);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[0] = max(sum.data[0], val.w);
    val.x = hip_unpack2(pix.y);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[1] = max(sum.data[1], val.w);
    val.y = hip_unpack3(pix.y);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[2] = max(sum.data[2], val.w);
    val.z = hip_unpack0(pix.z);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[3] = max(sum.data[3], val.w);
    val.x = hip_unpack1(pix.z);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[4] = max(sum.data[4], val.w);
    val.y = hip_unpack2(pix.z);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[5] = max(sum.data[5], val.w);
    val.z = hip_unpack3(pix.z);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[6] = max(sum.data[6], val.w);
    val.x = hip_unpack0(pix.w);
    val.w = hip_max3(val.x, val.y, val.z);
    sum.data[7] = max(sum.data[7], val.w);

    uint2 dst;
    dst.x = hip_pack(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = hip_pack(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx =  y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Dilate_U8_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Dilate_U8_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxErode kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Erode_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum;
    uint4 pix;
    float4 val;
    uint2 *pixLoc0 = (uint2*)&pix.x;
    uint2 *pixLoc2 = (uint2*)&pix.z;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    *pixLoc0 = lbufptr[0];
    *pixLoc2 = lbufptr[1];
    val.x = hip_unpack3(pix.x);
    val.y = hip_unpack0(pix.y);
    val.z = hip_unpack1(pix.y);
    sum.data[0] = hip_min3(val.x, val.y, val.z);
    val.x = hip_unpack2(pix.y);
    sum.data[1] = hip_min3(val.x, val.y, val.z);
    val.y = hip_unpack3(pix.y);
    sum.data[2] = hip_min3(val.x, val.y, val.z);
    val.z = hip_unpack0(pix.z);
    sum.data[3] = hip_min3(val.x, val.y, val.z);
    val.x = hip_unpack1(pix.z);
    sum.data[4] = hip_min3(val.x, val.y, val.z);
    val.y = hip_unpack2(pix.z);
    sum.data[5] = hip_min3(val.x, val.y, val.z);
    val.z = hip_unpack3(pix.z);
    sum.data[6] = hip_min3(val.x, val.y, val.z);
    val.x = hip_unpack0(pix.w);
    sum.data[7] = hip_min3(val.x, val.y, val.z);
    *pixLoc0 = lbufptr[17];
    *pixLoc2 = lbufptr[18];
    val.x = hip_unpack3(pix.x);
    val.y = hip_unpack0(pix.y);
    val.z = hip_unpack1(pix.y);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[0] = min(sum.data[0], val.w);
    val.x = hip_unpack2(pix.y);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[1] = min(sum.data[1], val.w);
    val.y = hip_unpack3(pix.y);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[2] = min(sum.data[2], val.w);
    val.z = hip_unpack0(pix.z);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[3] = min(sum.data[3], val.w);
    val.x = hip_unpack1(pix.z);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[4] = min(sum.data[4], val.w);
    val.y = hip_unpack2(pix.z);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[5] = min(sum.data[5], val.w);
    val.z = hip_unpack3(pix.z);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[6] = min(sum.data[6], val.w);
    val.x = hip_unpack0(pix.w);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[7] = min(sum.data[7], val.w);
    *pixLoc0 = lbufptr[34];
    *pixLoc2 = lbufptr[35];
    val.x = hip_unpack3(pix.x);
    val.y = hip_unpack0(pix.y);
    val.z = hip_unpack1(pix.y);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[0] = min(sum.data[0], val.w);
    val.x = hip_unpack2(pix.y);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[1] = min(sum.data[1], val.w);
    val.y = hip_unpack3(pix.y);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[2] = min(sum.data[2], val.w);
    val.z = hip_unpack0(pix.z);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[3] = min(sum.data[3], val.w);
    val.x = hip_unpack1(pix.z);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[4] = min(sum.data[4], val.w);
    val.y = hip_unpack2(pix.z);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[5] = min(sum.data[5], val.w);
    val.z = hip_unpack3(pix.z);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[6] = min(sum.data[6], val.w);
    val.x = hip_unpack0(pix.w);
    val.w = hip_min3(val.x, val.y, val.z);
    sum.data[7] = min(sum.data[7], val.w);

    uint2 dst;
    dst.x = hip_pack(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = hip_pack(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx =  y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Erode_U8_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Erode_U8_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxMedian kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Median_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum;
    float4 val0, val1, val2, valz;
    uint4 pix0, pix1, pix2;
    uint2 *pix0Loc0 = (uint2*)&pix0.x;
    uint2 *pix0Loc2 = (uint2*)&pix0.z;
    uint2 *pix1Loc0 = (uint2*)&pix1.x;
    uint2 *pix1Loc2 = (uint2*)&pix1.z;
    uint2 *pix2Loc0 = (uint2*)&pix2.x;
    uint2 *pix2Loc2 = (uint2*)&pix2.z;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    *pix0Loc0 = lbufptr[0];
    *pix0Loc2 = lbufptr[1];
    *pix1Loc0 = lbufptr[17];
    *pix1Loc2 = lbufptr[18];
    *pix2Loc0 = lbufptr[34];
    *pix2Loc2 = lbufptr[35];
    // pixel 0
    valz.x = hip_unpack3(pix0.x);
    valz.y = hip_unpack0(pix0.y);
    valz.z = hip_unpack1(pix0.y);
    val0.x = hip_min3(valz.x, valz.y, valz.z);
    val0.y = hip_median3(valz.x, valz.y, valz.z);
    val0.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack3(pix1.x);
    valz.y = hip_unpack0(pix1.y);
    valz.z = hip_unpack1(pix1.y);
    val1.x = hip_min3(valz.x, valz.y, valz.z);
    val1.y = hip_median3(valz.x, valz.y, valz.z);
    val1.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack3(pix2.x);
    valz.y = hip_unpack0(pix2.y);
    valz.z = hip_unpack1(pix2.y);
    val2.x = hip_min3(valz.x, valz.y, valz.z);
    val2.y = hip_median3(valz.x, valz.y, valz.z);
    val2.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_max3(val0.x, val1.x, val2.x);
    valz.y = hip_median3(val0.y, val1.y, val2.y);
    valz.z = hip_min3(val0.z, val1.z, val2.z);
    sum.data[0] = hip_median3(valz.x, valz.y, valz.z);
    // pixel 1
    valz.x = hip_unpack0(pix0.y);
    valz.y = hip_unpack1(pix0.y);
    valz.z = hip_unpack2(pix0.y);
    val0.x = hip_min3(valz.x, valz.y, valz.z);
    val0.y = hip_median3(valz.x, valz.y, valz.z);
    val0.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack0(pix1.y);
    valz.y = hip_unpack1(pix1.y);
    valz.z = hip_unpack2(pix1.y);
    val1.x = hip_min3(valz.x, valz.y, valz.z);
    val1.y = hip_median3(valz.x, valz.y, valz.z);
    val1.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack0(pix2.y);
    valz.y = hip_unpack1(pix2.y);
    valz.z = hip_unpack2(pix2.y);
    val2.x = hip_min3(valz.x, valz.y, valz.z);
    val2.y = hip_median3(valz.x, valz.y, valz.z);
    val2.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_max3(val0.x, val1.x, val2.x);
    valz.y = hip_median3(val0.y, val1.y, val2.y);
    valz.z = hip_min3(val0.z, val1.z, val2.z);
    sum.data[1] = hip_median3(valz.x, valz.y, valz.z);
    // pixel 2
    valz.x = hip_unpack1(pix0.y);
    valz.y = hip_unpack2(pix0.y);
    valz.z = hip_unpack3(pix0.y);
    val0.x = hip_min3(valz.x, valz.y, valz.z);
    val0.y = hip_median3(valz.x, valz.y, valz.z);
    val0.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack1(pix1.y);
    valz.y = hip_unpack2(pix1.y);
    valz.z = hip_unpack3(pix1.y);
    val1.x = hip_min3(valz.x, valz.y, valz.z);
    val1.y = hip_median3(valz.x, valz.y, valz.z);
    val1.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack1(pix2.y);
    valz.y = hip_unpack2(pix2.y);
    valz.z = hip_unpack3(pix2.y);
    val2.x = hip_min3(valz.x, valz.y, valz.z);
    val2.y = hip_median3(valz.x, valz.y, valz.z);
    val2.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_max3(val0.x, val1.x, val2.x);
    valz.y = hip_median3(val0.y, val1.y, val2.y);
    valz.z = hip_min3(val0.z, val1.z, val2.z);
    sum.data[2] = hip_median3(valz.x, valz.y, valz.z);
    // pixel 3
    valz.x = hip_unpack2(pix0.y);
    valz.y = hip_unpack3(pix0.y);
    valz.z = hip_unpack0(pix0.z);
    val0.x = hip_min3(valz.x, valz.y, valz.z);
    val0.y = hip_median3(valz.x, valz.y, valz.z);
    val0.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack2(pix1.y);
    valz.y = hip_unpack3(pix1.y);
    valz.z = hip_unpack0(pix1.z);
    val1.x = hip_min3(valz.x, valz.y, valz.z);
    val1.y = hip_median3(valz.x, valz.y, valz.z);
    val1.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack2(pix2.y);
    valz.y = hip_unpack3(pix2.y);
    valz.z = hip_unpack0(pix2.z);
    val2.x = hip_min3(valz.x, valz.y, valz.z);
    val2.y = hip_median3(valz.x, valz.y, valz.z);
    val2.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_max3(val0.x, val1.x, val2.x);
    valz.y = hip_median3(val0.y, val1.y, val2.y);
    valz.z = hip_min3(val0.z, val1.z, val2.z);
    sum.data[3] = hip_median3(valz.x, valz.y, valz.z);
    // pixel 4
    valz.x = hip_unpack3(pix0.y);
    valz.y = hip_unpack0(pix0.z);
    valz.z = hip_unpack1(pix0.z);
    val0.x = hip_min3(valz.x, valz.y, valz.z);
    val0.y = hip_median3(valz.x, valz.y, valz.z);
    val0.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack3(pix1.y);
    valz.y = hip_unpack0(pix1.z);
    valz.z = hip_unpack1(pix1.z);
    val1.x = hip_min3(valz.x, valz.y, valz.z);
    val1.y = hip_median3(valz.x, valz.y, valz.z);
    val1.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack3(pix2.y);
    valz.y = hip_unpack0(pix2.z);
    valz.z = hip_unpack1(pix2.z);
    val2.x = hip_min3(valz.x, valz.y, valz.z);
    val2.y = hip_median3(valz.x, valz.y, valz.z);
    val2.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_max3(val0.x, val1.x, val2.x);
    valz.y = hip_median3(val0.y, val1.y, val2.y);
    valz.z = hip_min3(val0.z, val1.z, val2.z);
    sum.data[4] = hip_median3(valz.x, valz.y, valz.z);
    // pixel 5
    valz.x = hip_unpack0(pix0.z);
    valz.y = hip_unpack1(pix0.z);
    valz.z = hip_unpack2(pix0.z);
    val0.x = hip_min3(valz.x, valz.y, valz.z);
    val0.y = hip_median3(valz.x, valz.y, valz.z);
    val0.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack0(pix1.z);
    valz.y = hip_unpack1(pix1.z);
    valz.z = hip_unpack2(pix1.z);
    val1.x = hip_min3(valz.x, valz.y, valz.z);
    val1.y = hip_median3(valz.x, valz.y, valz.z);
    val1.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack0(pix2.z);
    valz.y = hip_unpack1(pix2.z);
    valz.z = hip_unpack2(pix2.z);
    val2.x = hip_min3(valz.x, valz.y, valz.z);
    val2.y = hip_median3(valz.x, valz.y, valz.z);
    val2.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_max3(val0.x, val1.x, val2.x);
    valz.y = hip_median3(val0.y, val1.y, val2.y);
    valz.z = hip_min3(val0.z, val1.z, val2.z);
    sum.data[5] = hip_median3(valz.x, valz.y, valz.z);
    // pixel 6
    valz.x = hip_unpack1(pix0.z);
    valz.y = hip_unpack2(pix0.z);
    valz.z = hip_unpack3(pix0.z);
    val0.x = hip_min3(valz.x, valz.y, valz.z);
    val0.y = hip_median3(valz.x, valz.y, valz.z);
    val0.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack1(pix1.z);
    valz.y = hip_unpack2(pix1.z);
    valz.z = hip_unpack3(pix1.z);
    val1.x = hip_min3(valz.x, valz.y, valz.z);
    val1.y = hip_median3(valz.x, valz.y, valz.z);
    val1.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack1(pix2.z);
    valz.y = hip_unpack2(pix2.z);
    valz.z = hip_unpack3(pix2.z);
    val2.x = hip_min3(valz.x, valz.y, valz.z);
    val2.y = hip_median3(valz.x, valz.y, valz.z);
    val2.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_max3(val0.x, val1.x, val2.x);
    valz.y = hip_median3(val0.y, val1.y, val2.y);
    valz.z = hip_min3(val0.z, val1.z, val2.z);
    sum.data[6] = hip_median3(valz.x, valz.y, valz.z);
    // pixel 7
    valz.x = hip_unpack2(pix0.z);
    valz.y = hip_unpack3(pix0.z);
    valz.z = hip_unpack0(pix0.w);
    val0.x = hip_min3(valz.x, valz.y, valz.z);
    val0.y = hip_median3(valz.x, valz.y, valz.z);
    val0.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack2(pix1.z);
    valz.y = hip_unpack3(pix1.z);
    valz.z = hip_unpack0(pix1.w);
    val1.x = hip_min3(valz.x, valz.y, valz.z);
    val1.y = hip_median3(valz.x, valz.y, valz.z);
    val1.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_unpack2(pix2.z);
    valz.y = hip_unpack3(pix2.z);
    valz.z = hip_unpack0(pix2.w);
    val2.x = hip_min3(valz.x, valz.y, valz.z);
    val2.y = hip_median3(valz.x, valz.y, valz.z);
    val2.z = hip_max3(valz.x, valz.y, valz.z);
    valz.x = hip_max3(val0.x, val1.x, val2.x);
    valz.y = hip_median3(val0.y, val1.y, val2.y);
    valz.z = hip_min3(val0.z, val1.z, val2.z);
    sum.data[7] = hip_median3(valz.x, valz.y, valz.z);

    uint2 dst;
    dst.x = hip_pack(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = hip_pack(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx =  y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Median_U8_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Median_U8_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxGaussian kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Gaussian_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, 6.250000000000e-02f, sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, 1.250000000000e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 6.250000000000e-02f, sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, 6.250000000000e-02f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.250000000000e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 6.250000000000e-02f, sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, 6.250000000000e-02f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.250000000000e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 6.250000000000e-02f, sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, 6.250000000000e-02f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.250000000000e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 6.250000000000e-02f, sum.data[4]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, 6.250000000000e-02f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.250000000000e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 6.250000000000e-02f, sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, 6.250000000000e-02f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.250000000000e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 6.250000000000e-02f, sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, 6.250000000000e-02f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.250000000000e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 6.250000000000e-02f, sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, 6.250000000000e-02f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.250000000000e-01f, sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, 6.250000000000e-02f, sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, 1.250000000000e-01f, sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, 2.500000000000e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.250000000000e-01f, sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, 1.250000000000e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 2.500000000000e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.250000000000e-01f, sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, 1.250000000000e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 2.500000000000e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.250000000000e-01f, sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, 1.250000000000e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 2.500000000000e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.250000000000e-01f, sum.data[4]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, 1.250000000000e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 2.500000000000e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.250000000000e-01f, sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, 1.250000000000e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 2.500000000000e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.250000000000e-01f, sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, 1.250000000000e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 2.500000000000e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.250000000000e-01f, sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, 1.250000000000e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 2.500000000000e-01f, sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, 1.250000000000e-01f, sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, 6.250000000000e-02f, sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, 1.250000000000e-01f, sum.data[0]);
    sum.data[1] = fmaf(fval, 6.250000000000e-02f, sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, 6.250000000000e-02f, sum.data[0]);
    sum.data[1] = fmaf(fval, 1.250000000000e-01f, sum.data[1]);
    sum.data[2] = fmaf(fval, 6.250000000000e-02f, sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, 6.250000000000e-02f, sum.data[1]);
    sum.data[2] = fmaf(fval, 1.250000000000e-01f, sum.data[2]);
    sum.data[3] = fmaf(fval, 6.250000000000e-02f, sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, 6.250000000000e-02f, sum.data[2]);
    sum.data[3] = fmaf(fval, 1.250000000000e-01f, sum.data[3]);
    sum.data[4] = fmaf(fval, 6.250000000000e-02f, sum.data[4]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, 6.250000000000e-02f, sum.data[3]);
    sum.data[4] = fmaf(fval, 1.250000000000e-01f, sum.data[4]);
    sum.data[5] = fmaf(fval, 6.250000000000e-02f, sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, 6.250000000000e-02f, sum.data[4]);
    sum.data[5] = fmaf(fval, 1.250000000000e-01f, sum.data[5]);
    sum.data[6] = fmaf(fval, 6.250000000000e-02f, sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, 6.250000000000e-02f, sum.data[5]);
    sum.data[6] = fmaf(fval, 1.250000000000e-01f, sum.data[6]);
    sum.data[7] = fmaf(fval, 6.250000000000e-02f, sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, 6.250000000000e-02f, sum.data[6]);
    sum.data[7] = fmaf(fval, 1.250000000000e-01f, sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, 6.250000000000e-02f, sum.data[7]);

    sum.data[0] += -4.999899864197e-01f;
    sum.data[1] += -4.999899864197e-01f;
    sum.data[2] += -4.999899864197e-01f;
    sum.data[3] += -4.999899864197e-01f;
    sum.data[4] += -4.999899864197e-01f;
    sum.data[5] += -4.999899864197e-01f;
    sum.data[6] += -4.999899864197e-01f;
    sum.data[7] += -4.999899864197e-01f;

    uint2 dst;
    dst.x = hip_pack(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = hip_pack(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx =  y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Gaussian_U8_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Gaussian_U8_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxConvolve kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Convolve_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes,
    int srcImageBufferSize, float *conv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[0], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[1], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[0], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[2], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[1], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[0], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[2], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[1], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[0], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[2], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[1], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[0], sum.data[4]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[2], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[1], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[0], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[2], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[1], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[0], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[2], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[1], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[0], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[2], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[1], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[2], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[3], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[4], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[3], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[5], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[4], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[3], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[5], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[4], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[3], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[5], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[4], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[3], sum.data[4]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[5], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[4], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[3], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[5], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[4], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[3], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[5], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[4], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[3], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[5], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[4], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[5], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[6], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[7], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[6], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[8], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[7], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[6], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[8], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[7], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[6], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[8], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[7], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[6], sum.data[4]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[8], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[7], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[6], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[8], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[7], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[6], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[8], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[7], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[6], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[8], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[7], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[8], sum.data[7]);

    sum.data[0] += -4.999899864197e-01f;
    sum.data[1] += -4.999899864197e-01f;
    sum.data[2] += -4.999899864197e-01f;
    sum.data[3] += -4.999899864197e-01f;
    sum.data[4] += -4.999899864197e-01f;
    sum.data[5] += -4.999899864197e-01f;
    sum.data[6] += -4.999899864197e-01f;
    sum.data[7] += -4.999899864197e-01f;

    uint2 dst;
    dst.x = hip_pack(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = hip_pack(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx = y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_U8_U8_5x5(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes, int srcImageBufferSize,
    float *conv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2720]; // 136x20 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x20 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 2) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 4) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 4) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 2) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 20) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[ 0], sum.data[0]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 1], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 0], sum.data[1]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 2], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 1], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 0], sum.data[2]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 3], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 2], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 1], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 0], sum.data[3]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[ 4], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 3], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 2], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 1], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 0], sum.data[4]);
    fval = hip_unpack3(pix.y);
    sum.data[1] = fmaf(fval, conv[ 4], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 3], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 2], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 1], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 0], sum.data[5]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[2] = fmaf(fval, conv[ 4], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 3], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 2], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 1], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 0], sum.data[6]);
    fval = hip_unpack1(pix.x);
    sum.data[3] = fmaf(fval, conv[ 4], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 3], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 2], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 1], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 0], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[4] = fmaf(fval, conv[ 4], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 3], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 2], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 1], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[5] = fmaf(fval, conv[ 4], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 3], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 2], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[6] = fmaf(fval, conv[ 4], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 3], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[7] = fmaf(fval, conv[ 4], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[ 5], sum.data[0]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 6], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 5], sum.data[1]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 7], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 6], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 5], sum.data[2]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 8], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 7], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 6], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 5], sum.data[3]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[ 9], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 8], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 7], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 6], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 5], sum.data[4]);
    fval = hip_unpack3(pix.y);
    sum.data[1] = fmaf(fval, conv[ 9], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 8], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 7], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 6], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 5], sum.data[5]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[2] = fmaf(fval, conv[ 9], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 8], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 7], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 6], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 5], sum.data[6]);
    fval = hip_unpack1(pix.x);
    sum.data[3] = fmaf(fval, conv[ 9], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 8], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 7], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 6], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 5], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[4] = fmaf(fval, conv[ 9], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 8], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 7], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 6], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[5] = fmaf(fval, conv[ 9], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 8], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 7], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[6] = fmaf(fval, conv[ 9], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 8], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[7] = fmaf(fval, conv[ 9], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[10], sum.data[0]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[11], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[10], sum.data[1]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[12], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[11], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[10], sum.data[2]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[13], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[12], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[11], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[10], sum.data[3]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[14], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[13], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[12], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[11], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[10], sum.data[4]);
    fval = hip_unpack3(pix.y);
    sum.data[1] = fmaf(fval, conv[14], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[13], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[12], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[11], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[10], sum.data[5]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[2] = fmaf(fval, conv[14], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[13], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[12], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[11], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[10], sum.data[6]);
    fval = hip_unpack1(pix.x);
    sum.data[3] = fmaf(fval, conv[14], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[13], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[12], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[11], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[10], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[4] = fmaf(fval, conv[14], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[13], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[12], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[11], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[5] = fmaf(fval, conv[14], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[13], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[12], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[6] = fmaf(fval, conv[14], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[13], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[7] = fmaf(fval, conv[14], sum.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[15], sum.data[0]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[16], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[15], sum.data[1]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[17], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[16], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[15], sum.data[2]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[18], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[17], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[16], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[15], sum.data[3]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[19], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[18], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[17], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[16], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[15], sum.data[4]);
    fval = hip_unpack3(pix.y);
    sum.data[1] = fmaf(fval, conv[19], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[18], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[17], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[16], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[15], sum.data[5]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum.data[2] = fmaf(fval, conv[19], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[18], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[17], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[16], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[15], sum.data[6]);
    fval = hip_unpack1(pix.x);
    sum.data[3] = fmaf(fval, conv[19], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[18], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[17], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[16], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[15], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[4] = fmaf(fval, conv[19], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[18], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[17], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[16], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[5] = fmaf(fval, conv[19], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[18], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[17], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[6] = fmaf(fval, conv[19], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[18], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[7] = fmaf(fval, conv[19], sum.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[20], sum.data[0]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[21], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[20], sum.data[1]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[22], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[21], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[20], sum.data[2]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[23], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[22], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[21], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[20], sum.data[3]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[24], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[23], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[22], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[21], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[20], sum.data[4]);
    fval = hip_unpack3(pix.y);
    sum.data[1] = fmaf(fval, conv[24], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[23], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[22], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[21], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[20], sum.data[5]);
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum.data[2] = fmaf(fval, conv[24], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[23], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[22], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[21], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[20], sum.data[6]);
    fval = hip_unpack1(pix.x);
    sum.data[3] = fmaf(fval, conv[24], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[23], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[22], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[21], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[20], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[4] = fmaf(fval, conv[24], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[23], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[22], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[21], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[5] = fmaf(fval, conv[24], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[23], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[22], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[6] = fmaf(fval, conv[24], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[23], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[7] = fmaf(fval, conv[24], sum.data[7]);

    sum.data[0] += -4.999899864197e-01f;
    sum.data[1] += -4.999899864197e-01f;
    sum.data[2] += -4.999899864197e-01f;
    sum.data[3] += -4.999899864197e-01f;
    sum.data[4] += -4.999899864197e-01f;
    sum.data[5] += -4.999899864197e-01f;
    sum.data[6] += -4.999899864197e-01f;
    sum.data[7] += -4.999899864197e-01f;

    uint2 dst;
    dst.x = hip_pack(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = hip_pack(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx = y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_U8_U8_7x7(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes,
    int srcImageBufferSize, float *conv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2992]; // 136x22 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x22 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 3) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 6) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 6) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 3) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 22) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[ 0], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[ 1], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 0], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 2], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 1], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 0], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 3], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 2], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 1], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 0], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 4], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 3], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 2], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 1], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 0], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[ 5], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 4], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 3], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 2], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 1], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 0], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[ 6], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 5], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 4], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 3], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 2], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 1], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 0], sum.data[6]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[ 6], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 5], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 4], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 3], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 2], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 1], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 0], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[ 6], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 5], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 4], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 3], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 2], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 1], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[ 6], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 5], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 4], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 3], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 2], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[ 6], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 5], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 4], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 3], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[ 6], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 5], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 4], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[ 6], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 5], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[ 6], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[ 7], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[ 8], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 7], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 9], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 8], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 7], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[10], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 9], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 8], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 7], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[11], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[10], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 9], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 8], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 7], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[12], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[11], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[10], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 9], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 8], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 7], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[13], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[12], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[11], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[10], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 9], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 8], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 7], sum.data[6]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[13], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[12], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[11], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[10], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 9], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 8], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 7], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[13], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[12], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[11], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[10], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 9], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 8], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[13], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[12], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[11], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[10], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 9], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[13], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[12], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[11], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[10], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[13], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[12], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[11], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[13], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[12], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[13], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[14], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[15], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[14], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[16], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[15], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[14], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[17], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[16], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[15], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[14], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[18], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[17], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[16], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[15], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[14], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[19], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[18], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[17], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[16], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[15], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[14], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[20], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[19], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[18], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[17], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[16], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[15], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[14], sum.data[6]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[20], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[19], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[18], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[17], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[16], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[15], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[14], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[20], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[19], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[18], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[17], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[16], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[15], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[20], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[19], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[18], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[17], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[16], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[20], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[19], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[18], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[17], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[20], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[19], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[18], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[20], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[19], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[20], sum.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[21], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[22], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[21], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[23], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[22], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[21], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[24], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[23], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[22], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[21], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[25], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[24], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[23], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[22], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[21], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[26], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[25], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[24], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[23], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[22], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[21], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[27], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[26], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[25], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[24], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[23], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[22], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[21], sum.data[6]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[27], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[26], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[25], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[24], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[23], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[22], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[21], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[27], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[26], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[25], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[24], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[23], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[22], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[27], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[26], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[25], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[24], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[23], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[27], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[26], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[25], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[24], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[27], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[26], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[25], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[27], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[26], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[27], sum.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[28], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[29], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[28], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[30], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[29], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[28], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[31], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[30], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[29], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[28], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[32], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[31], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[30], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[29], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[28], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[33], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[32], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[31], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[30], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[29], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[28], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[34], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[33], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[32], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[31], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[30], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[29], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[28], sum.data[6]);
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[34], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[33], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[32], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[31], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[30], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[29], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[28], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[34], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[33], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[32], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[31], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[30], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[29], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[34], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[33], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[32], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[31], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[30], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[34], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[33], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[32], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[31], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[34], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[33], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[32], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[34], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[33], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[34], sum.data[7]);
    // filterRow = 5
    pix = lbufptr[85];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[35], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[36], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[35], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[37], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[36], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[35], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[38], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[37], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[36], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[35], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[39], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[38], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[37], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[36], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[35], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[40], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[39], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[38], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[37], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[36], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[35], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[41], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[40], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[39], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[38], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[37], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[36], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[35], sum.data[6]);
    pix = lbufptr[86];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[41], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[40], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[39], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[38], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[37], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[36], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[35], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[41], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[40], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[39], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[38], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[37], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[36], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[41], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[40], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[39], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[38], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[37], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[41], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[40], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[39], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[38], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[41], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[40], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[39], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[41], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[40], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[41], sum.data[7]);
    // filterRow = 6
    pix = lbufptr[102];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[42], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[43], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[42], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[44], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[43], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[42], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[45], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[44], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[43], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[42], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[46], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[45], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[44], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[43], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[42], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[47], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[46], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[45], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[44], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[43], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[42], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[48], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[47], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[46], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[45], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[44], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[43], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[42], sum.data[6]);
    pix = lbufptr[103];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[48], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[47], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[46], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[45], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[44], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[43], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[42], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[48], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[47], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[46], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[45], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[44], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[43], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[48], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[47], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[46], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[45], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[44], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[48], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[47], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[46], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[45], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[48], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[47], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[46], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[48], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[47], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[48], sum.data[7]);

    sum.data[0] += -4.999899864197e-01f;
    sum.data[1] += -4.999899864197e-01f;
    sum.data[2] += -4.999899864197e-01f;
    sum.data[3] += -4.999899864197e-01f;
    sum.data[4] += -4.999899864197e-01f;
    sum.data[5] += -4.999899864197e-01f;
    sum.data[6] += -4.999899864197e-01f;
    sum.data[7] += -4.999899864197e-01f;

    uint2 dst;
    dst.x = hip_pack(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = hip_pack(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx = y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_U8_U8_3x9(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes,
    int srcImageBufferSize, float *conv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[3264]; // 136x24 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x24 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 4) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 8) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 8) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 4) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 24) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 0], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 1], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 0], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 2], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 1], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 0], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[ 2], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 1], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 0], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[ 2], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 1], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 0], sum.data[4]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[ 2], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 1], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 0], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[ 2], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 1], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 0], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[ 2], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 1], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 0], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[ 2], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 1], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[ 2], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 3], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 4], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 3], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 5], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 4], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 3], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[ 5], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 4], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 3], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[ 5], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 4], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 3], sum.data[4]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[ 5], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 4], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 3], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[ 5], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 4], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 3], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[ 5], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 4], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 3], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[ 5], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 4], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[ 5], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 6], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 7], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 6], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 8], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 7], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 6], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[ 8], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 7], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 6], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[ 8], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 7], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 6], sum.data[4]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[ 8], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 7], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 6], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[ 8], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 7], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 6], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[ 8], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 7], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 6], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[ 8], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 7], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[ 8], sum.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 9], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[10], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 9], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[11], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[10], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 9], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[11], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[10], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 9], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[11], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[10], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 9], sum.data[4]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[11], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[10], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 9], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[11], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[10], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 9], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[11], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[10], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 9], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[11], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[10], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[11], sum.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[12], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[13], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[12], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[14], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[13], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[12], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[14], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[13], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[12], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[14], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[13], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[12], sum.data[4]);
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[14], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[13], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[12], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[14], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[13], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[12], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[14], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[13], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[12], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[14], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[13], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[14], sum.data[7]);
    // filterRow = 5
    pix = lbufptr[85];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[15], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[16], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[15], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[17], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[16], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[15], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[17], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[16], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[15], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[17], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[16], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[15], sum.data[4]);
    pix = lbufptr[86];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[17], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[16], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[15], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[17], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[16], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[15], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[17], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[16], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[15], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[17], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[16], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[17], sum.data[7]);
    // filterRow = 6
    pix = lbufptr[102];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[18], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[19], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[18], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[20], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[19], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[18], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[20], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[19], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[18], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[20], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[19], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[18], sum.data[4]);
    pix = lbufptr[103];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[20], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[19], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[18], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[20], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[19], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[18], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[20], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[19], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[18], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[20], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[19], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[20], sum.data[7]);
    // filterRow = 7
    pix = lbufptr[119];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[21], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[22], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[21], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[23], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[22], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[21], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[23], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[22], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[21], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[23], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[22], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[21], sum.data[4]);
    pix = lbufptr[120];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[23], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[22], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[21], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[23], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[22], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[21], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[23], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[22], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[21], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[23], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[22], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[23], sum.data[7]);
    // filterRow = 8
    pix = lbufptr[136];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[24], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[25], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[24], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[26], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[25], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[24], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[26], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[25], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[24], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[26], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[25], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[24], sum.data[4]);
    pix = lbufptr[137];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[26], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[25], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[24], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[26], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[25], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[24], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[26], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[25], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[24], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[26], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[25], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[26], sum.data[7]);

    sum.data[0] += -4.999899864197e-01f;
    sum.data[1] += -4.999899864197e-01f;
    sum.data[2] += -4.999899864197e-01f;
    sum.data[3] += -4.999899864197e-01f;
    sum.data[4] += -4.999899864197e-01f;
    sum.data[5] += -4.999899864197e-01f;
    sum.data[6] += -4.999899864197e-01f;
    sum.data[7] += -4.999899864197e-01f;

    uint2 dst;
    dst.x = hip_pack(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = hip_pack(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx = y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_U8_U8_9x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes,
    int srcImageBufferSize, float *conv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
      pix = lbufptr[0];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[ 0], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[ 1], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 0], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[ 2], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 1], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 0], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 3], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 2], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 1], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 0], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 4], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 3], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 2], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 1], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 0], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 5], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 4], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 3], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 2], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 1], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 0], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[ 6], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 5], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 4], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 3], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 2], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 1], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 0], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[ 7], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 6], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 5], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 4], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 3], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 2], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 1], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 0], sum.data[7]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[ 8], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 7], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 6], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 5], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 4], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 3], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 2], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 1], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[ 8], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 7], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 6], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 5], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 4], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 3], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 2], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[ 8], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 7], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 6], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 5], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 4], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 3], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[ 8], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 7], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 6], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 5], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 4], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[ 8], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 7], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 6], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 5], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[ 8], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 7], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 6], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[ 8], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 7], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[ 8], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[ 9], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[10], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 9], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[11], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[10], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 9], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[12], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[11], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[10], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 9], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[13], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[12], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[11], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[10], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 9], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[14], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[13], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[12], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[11], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[10], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 9], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[15], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[14], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[13], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[12], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[11], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[10], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 9], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[16], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[15], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[14], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[13], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[12], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[11], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[10], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 9], sum.data[7]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[17], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[16], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[15], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[14], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[13], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[12], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[11], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[10], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[17], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[16], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[15], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[14], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[13], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[12], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[11], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[17], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[16], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[15], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[14], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[13], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[12], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[17], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[16], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[15], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[14], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[13], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[17], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[16], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[15], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[14], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[17], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[16], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[15], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[17], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[16], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[17], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[18], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[19], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[18], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[20], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[19], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[18], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[21], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[20], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[19], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[18], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[22], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[21], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[20], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[19], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[18], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[23], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[22], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[21], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[20], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[19], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[18], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[24], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[23], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[22], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[21], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[20], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[19], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[18], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[25], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[24], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[23], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[22], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[21], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[20], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[19], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[18], sum.data[7]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[26], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[25], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[24], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[23], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[22], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[21], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[20], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[19], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[26], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[25], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[24], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[23], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[22], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[21], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[20], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[26], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[25], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[24], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[23], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[22], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[21], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[26], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[25], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[24], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[23], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[22], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[26], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[25], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[24], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[23], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[26], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[25], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[24], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[26], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[25], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[26], sum.data[7]);

    sum.data[0] += -4.999899864197e-01f;
    sum.data[1] += -4.999899864197e-01f;
    sum.data[2] += -4.999899864197e-01f;
    sum.data[3] += -4.999899864197e-01f;
    sum.data[4] += -4.999899864197e-01f;
    sum.data[5] += -4.999899864197e-01f;
    sum.data[6] += -4.999899864197e-01f;
    sum.data[7] += -4.999899864197e-01f;

    uint2 dst;
    dst.x = hip_pack(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = hip_pack(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx = y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_U8_U8_9x9(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes,
    int srcImageBufferSize, int srcImageBufferOffset, float *conv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[3264]; // 136x24 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x24 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 4) * srcImageStrideInBytes + x - 4;
        if (goffset > -srcImageBufferOffset && goffset < srcImageBufferSize)
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 8) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 8) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 4) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 24) ? true : false;
        }
        if (doExtraLoad && goffset > -srcImageBufferOffset && goffset < srcImageBufferSize) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
       }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[ 0], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[ 1], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 0], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[ 2], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 1], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 0], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 3], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 2], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 1], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 0], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 4], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 3], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 2], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 1], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 0], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 5], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 4], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 3], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 2], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 1], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 0], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[ 6], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 5], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 4], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 3], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 2], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 1], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 0], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[ 7], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 6], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 5], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 4], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 3], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 2], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 1], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 0], sum.data[7]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[ 8], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 7], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 6], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 5], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 4], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 3], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 2], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 1], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[ 8], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 7], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 6], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 5], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 4], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 3], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 2], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[ 8], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 7], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 6], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 5], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 4], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 3], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[ 8], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 7], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 6], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 5], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 4], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[ 8], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 7], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 6], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 5], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[ 8], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 7], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 6], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[ 8], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 7], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[ 8], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[ 9], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[10], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 9], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[11], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[10], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 9], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[12], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[11], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[10], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 9], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[13], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[12], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[11], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[10], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 9], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[14], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[13], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[12], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[11], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[10], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 9], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[15], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[14], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[13], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[12], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[11], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[10], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 9], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[16], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[15], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[14], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[13], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[12], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[11], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[10], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 9], sum.data[7]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[17], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[16], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[15], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[14], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[13], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[12], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[11], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[10], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[17], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[16], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[15], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[14], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[13], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[12], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[11], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[17], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[16], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[15], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[14], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[13], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[12], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[17], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[16], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[15], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[14], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[13], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[17], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[16], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[15], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[14], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[17], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[16], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[15], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[17], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[16], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[17], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[18], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[19], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[18], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[20], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[19], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[18], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[21], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[20], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[19], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[18], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[22], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[21], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[20], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[19], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[18], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[23], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[22], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[21], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[20], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[19], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[18], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[24], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[23], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[22], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[21], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[20], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[19], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[18], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[25], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[24], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[23], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[22], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[21], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[20], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[19], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[18], sum.data[7]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[26], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[25], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[24], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[23], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[22], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[21], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[20], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[19], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[26], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[25], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[24], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[23], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[22], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[21], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[20], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[26], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[25], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[24], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[23], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[22], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[21], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[26], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[25], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[24], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[23], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[22], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[26], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[25], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[24], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[23], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[26], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[25], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[24], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[26], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[25], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[26], sum.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[27], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[28], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[27], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[29], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[28], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[27], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[30], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[29], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[28], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[27], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[31], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[30], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[29], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[28], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[27], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[32], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[31], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[30], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[29], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[28], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[27], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[33], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[32], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[31], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[30], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[29], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[28], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[27], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[34], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[33], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[32], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[31], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[30], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[29], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[28], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[27], sum.data[7]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[35], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[34], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[33], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[32], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[31], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[30], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[29], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[28], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[35], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[34], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[33], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[32], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[31], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[30], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[29], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[35], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[34], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[33], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[32], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[31], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[30], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[35], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[34], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[33], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[32], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[31], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[35], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[34], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[33], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[32], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[35], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[34], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[33], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[35], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[34], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[35], sum.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[36], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[37], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[36], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[38], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[37], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[36], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[39], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[38], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[37], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[36], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[40], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[39], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[38], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[37], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[36], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[41], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[40], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[39], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[38], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[37], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[36], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[42], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[41], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[40], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[39], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[38], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[37], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[36], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[43], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[42], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[41], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[40], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[39], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[38], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[37], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[36], sum.data[7]);
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[44], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[43], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[42], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[41], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[40], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[39], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[38], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[37], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[44], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[43], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[42], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[41], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[40], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[39], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[38], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[44], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[43], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[42], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[41], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[40], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[39], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[44], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[43], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[42], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[41], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[40], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[44], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[43], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[42], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[41], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[44], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[43], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[42], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[44], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[43], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[44], sum.data[7]);
    // filterRow = 5
    pix = lbufptr[85];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[45], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[46], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[45], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[47], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[46], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[45], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[48], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[47], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[46], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[45], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[49], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[48], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[47], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[46], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[45], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[50], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[49], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[48], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[47], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[46], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[45], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[51], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[50], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[49], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[48], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[47], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[46], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[45], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[52], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[51], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[50], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[49], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[48], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[47], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[46], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[45], sum.data[7]);
    pix = lbufptr[86];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[53], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[52], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[51], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[50], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[49], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[48], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[47], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[46], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[53], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[52], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[51], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[50], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[49], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[48], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[47], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[53], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[52], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[51], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[50], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[49], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[48], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[53], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[52], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[51], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[50], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[49], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[53], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[52], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[51], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[50], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[53], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[52], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[51], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[53], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[52], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[53], sum.data[7]);
    // filterRow = 6
    pix = lbufptr[102];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[54], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[55], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[54], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[56], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[55], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[54], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[57], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[56], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[55], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[54], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[58], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[57], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[56], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[55], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[54], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[59], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[58], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[57], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[56], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[55], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[54], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[60], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[59], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[58], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[57], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[56], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[55], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[54], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[61], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[60], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[59], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[58], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[57], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[56], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[55], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[54], sum.data[7]);
    pix = lbufptr[103];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[62], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[61], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[60], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[59], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[58], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[57], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[56], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[55], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[62], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[61], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[60], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[59], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[58], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[57], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[56], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[62], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[61], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[60], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[59], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[58], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[57], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[62], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[61], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[60], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[59], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[58], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[62], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[61], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[60], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[59], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[62], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[61], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[60], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[62], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[61], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[62], sum.data[7]);
    // filterRow = 7
    pix = lbufptr[119];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[63], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[64], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[63], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[65], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[64], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[63], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[66], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[65], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[64], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[63], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[67], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[66], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[65], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[64], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[63], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[68], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[67], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[66], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[65], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[64], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[63], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[69], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[68], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[67], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[66], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[65], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[64], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[63], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[70], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[69], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[68], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[67], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[66], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[65], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[64], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[63], sum.data[7]);
    pix = lbufptr[120];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[71], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[70], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[69], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[68], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[67], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[66], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[65], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[64], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[71], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[70], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[69], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[68], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[67], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[66], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[65], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[71], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[70], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[69], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[68], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[67], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[66], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[71], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[70], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[69], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[68], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[67], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[71], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[70], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[69], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[68], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[71], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[70], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[69], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[71], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[70], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[71], sum.data[7]);
    // filterRow = 8
    pix = lbufptr[136];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[72], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[73], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[72], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[74], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[73], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[72], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[75], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[74], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[73], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[72], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[76], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[75], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[74], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[73], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[72], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[77], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[76], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[75], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[74], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[73], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[72], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[78], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[77], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[76], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[75], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[74], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[73], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[72], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[79], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[78], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[77], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[76], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[75], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[74], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[73], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[72], sum.data[7]);
    pix = lbufptr[137];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[80], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[79], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[78], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[77], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[76], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[75], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[74], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[73], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[80], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[79], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[78], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[77], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[76], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[75], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[74], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[80], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[79], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[78], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[77], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[76], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[75], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[80], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[79], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[78], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[77], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[76], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[80], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[79], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[78], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[77], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[80], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[79], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[78], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[80], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[79], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[80], sum.data[7]);

    sum.data[0] += -4.999899864197e-01f;
    sum.data[1] += -4.999899864197e-01f;
    sum.data[2] += -4.999899864197e-01f;
    sum.data[3] += -4.999899864197e-01f;
    sum.data[4] += -4.999899864197e-01f;
    sum.data[5] += -4.999899864197e-01f;
    sum.data[6] += -4.999899864197e-01f;
    sum.data[7] += -4.999899864197e-01f;

    uint2 dst;
    dst.x = hip_pack(make_float4(sum.data[0], sum.data[1], sum.data[2], sum.data[3]));
    dst.y = hip_pack(make_float4(sum.data[4], sum.data[5], sum.data[6], sum.data[7]));

    uint dstIdx = y * dstImageStrideInBytes + x;

    if (valid) {
        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}

int HipExec_Convolve_U8_U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, vx_uint32 srcImageBufferSize,
    vx_uint32 srcImageBufferOffset, float *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight) {

    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    dim3 gridDim = dim3(ceil((float)globalThreads_x/localThreads_x),
                        ceil((float)globalThreads_y/localThreads_y));

    dim3 blockDim = dim3(localThreads_x, localThreads_y);

    if (convolutionWidth == 3 && convolutionHeight == 3) {
        hipLaunchKernelGGL(Hip_Convolve_U8_U8_3x3, gridDim, blockDim, 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                            srcImageBufferSize, conv);
    } else if (convolutionWidth == 5 && convolutionHeight == 5) {
        hipLaunchKernelGGL(Hip_Convolve_U8_U8_5x5, gridDim, blockDim, 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                            srcImageBufferSize, conv);
    } else if (convolutionWidth == 7 && convolutionHeight == 7) {
        hipLaunchKernelGGL(Hip_Convolve_U8_U8_7x7, gridDim, blockDim, 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                            srcImageBufferSize, conv);
    } else if (convolutionWidth == 9 && convolutionHeight == 9) {
        hipLaunchKernelGGL(Hip_Convolve_U8_U8_9x9, gridDim, blockDim, 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                            srcImageBufferSize, srcImageBufferOffset, conv);
    } else if (convolutionWidth == 3 && convolutionHeight == 9) {
        hipLaunchKernelGGL(Hip_Convolve_U8_U8_3x9, gridDim, blockDim, 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                            srcImageBufferSize, conv);
    } else if (convolutionWidth == 9 && convolutionHeight == 3) {
        hipLaunchKernelGGL(Hip_Convolve_U8_U8_9x3, gridDim, blockDim, 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                            srcImageBufferSize, conv);
    } else {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_S16_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes,
    int srcImageBufferSize, float *conv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[0], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[1], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[0], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[2], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[1], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[0], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[2], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[1], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[0], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[2], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[1], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[0], sum.data[4]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[2], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[1], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[0], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[2], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[1], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[0], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[2], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[1], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[0], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[2], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[1], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[2], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[3], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[4], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[3], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[5], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[4], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[3], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[5], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[4], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[3], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[5], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[4], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[3], sum.data[4]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[5], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[4], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[3], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[5], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[4], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[3], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[5], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[4], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[3], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[5], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[4], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[5], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[6], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[7], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[6], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[8], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[7], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[6], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[8], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[7], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[6], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[8], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[7], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[6], sum.data[4]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[8], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[7], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[6], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[8], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[7], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[6], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[8], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[7], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[6], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[8], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[7], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[8], sum.data[7]);

    int4 dst;
    dst.x = ((int)hip_clamp(sum.data[0], -32768.0f, 32767.0f)) & 0xffff;
    dst.x |= ((int)hip_clamp(sum.data[1], -32768.0f, 32767.0f)) << 16;
    dst.y = ((int)hip_clamp(sum.data[2], -32768.0f, 32767.0f)) & 0xffff;
    dst.y |= ((int)hip_clamp(sum.data[3], -32768.0f, 32767.0f)) << 16;
    dst.z = ((int)hip_clamp(sum.data[4], -32768.0f, 32767.0f)) & 0xffff;
    dst.z |= ((int)hip_clamp(sum.data[5], -32768.0f, 32767.0f)) << 16;
    dst.w = ((int)hip_clamp(sum.data[6], -32768.0f, 32767.0f)) & 0xffff;
    dst.w |= ((int)hip_clamp(sum.data[7], -32768.0f, 32767.0f)) << 16;

    uint dstIdx = y * dstImageStrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage[dstIdx])) = dst;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_S16_U8_5x5(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes,
    int srcImageBufferSize, float *conv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2720]; // 136x20 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x20 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 2) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 4) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 4) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 2) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 20) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[ 0], sum.data[0]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 1], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 0], sum.data[1]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 2], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 1], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 0], sum.data[2]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 3], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 2], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 1], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 0], sum.data[3]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[ 4], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 3], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 2], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 1], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 0], sum.data[4]);
    fval = hip_unpack3(pix.y);
    sum.data[1] = fmaf(fval, conv[ 4], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 3], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 2], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 1], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 0], sum.data[5]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[2] = fmaf(fval, conv[ 4], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 3], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 2], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 1], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 0], sum.data[6]);
    fval = hip_unpack1(pix.x);
    sum.data[3] = fmaf(fval, conv[ 4], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 3], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 2], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 1], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 0], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[4] = fmaf(fval, conv[ 4], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 3], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 2], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 1], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[5] = fmaf(fval, conv[ 4], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 3], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 2], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[6] = fmaf(fval, conv[ 4], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 3], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[7] = fmaf(fval, conv[ 4], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[ 5], sum.data[0]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 6], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 5], sum.data[1]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 7], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 6], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 5], sum.data[2]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 8], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 7], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 6], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 5], sum.data[3]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[ 9], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 8], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 7], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 6], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 5], sum.data[4]);
    fval = hip_unpack3(pix.y);
    sum.data[1] = fmaf(fval, conv[ 9], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 8], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 7], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 6], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 5], sum.data[5]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[2] = fmaf(fval, conv[ 9], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 8], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 7], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 6], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 5], sum.data[6]);
    fval = hip_unpack1(pix.x);
    sum.data[3] = fmaf(fval, conv[ 9], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 8], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 7], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 6], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 5], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[4] = fmaf(fval, conv[ 9], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 8], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 7], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 6], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[5] = fmaf(fval, conv[ 9], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 8], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 7], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[6] = fmaf(fval, conv[ 9], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 8], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[7] = fmaf(fval, conv[ 9], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[10], sum.data[0]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[11], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[10], sum.data[1]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[12], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[11], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[10], sum.data[2]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[13], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[12], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[11], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[10], sum.data[3]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[14], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[13], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[12], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[11], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[10], sum.data[4]);
    fval = hip_unpack3(pix.y);
    sum.data[1] = fmaf(fval, conv[14], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[13], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[12], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[11], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[10], sum.data[5]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[2] = fmaf(fval, conv[14], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[13], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[12], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[11], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[10], sum.data[6]);
    fval = hip_unpack1(pix.x);
    sum.data[3] = fmaf(fval, conv[14], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[13], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[12], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[11], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[10], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[4] = fmaf(fval, conv[14], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[13], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[12], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[11], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[5] = fmaf(fval, conv[14], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[13], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[12], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[6] = fmaf(fval, conv[14], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[13], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[7] = fmaf(fval, conv[14], sum.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[15], sum.data[0]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[16], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[15], sum.data[1]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[17], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[16], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[15], sum.data[2]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[18], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[17], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[16], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[15], sum.data[3]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[19], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[18], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[17], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[16], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[15], sum.data[4]);
    fval = hip_unpack3(pix.y);
    sum.data[1] = fmaf(fval, conv[19], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[18], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[17], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[16], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[15], sum.data[5]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum.data[2] = fmaf(fval, conv[19], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[18], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[17], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[16], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[15], sum.data[6]);
    fval = hip_unpack1(pix.x);
    sum.data[3] = fmaf(fval, conv[19], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[18], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[17], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[16], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[15], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[4] = fmaf(fval, conv[19], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[18], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[17], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[16], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[5] = fmaf(fval, conv[19], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[18], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[17], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[6] = fmaf(fval, conv[19], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[18], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[7] = fmaf(fval, conv[19], sum.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[20], sum.data[0]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[21], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[20], sum.data[1]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[22], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[21], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[20], sum.data[2]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[23], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[22], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[21], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[20], sum.data[3]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[24], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[23], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[22], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[21], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[20], sum.data[4]);
    fval = hip_unpack3(pix.y);
    sum.data[1] = fmaf(fval, conv[24], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[23], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[22], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[21], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[20], sum.data[5]);
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum.data[2] = fmaf(fval, conv[24], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[23], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[22], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[21], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[20], sum.data[6]);
    fval = hip_unpack1(pix.x);
    sum.data[3] = fmaf(fval, conv[24], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[23], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[22], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[21], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[20], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[4] = fmaf(fval, conv[24], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[23], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[22], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[21], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[5] = fmaf(fval, conv[24], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[23], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[22], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[6] = fmaf(fval, conv[24], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[23], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[7] = fmaf(fval, conv[24], sum.data[7]);

    sum.data[0] = sum.data[0] + -4.999899864197e-01f;
    sum.data[1] = sum.data[1] + -4.999899864197e-01f;
    sum.data[2] = sum.data[2] + -4.999899864197e-01f;
    sum.data[3] = sum.data[3] + -4.999899864197e-01f;
    sum.data[4] = sum.data[4] + -4.999899864197e-01f;
    sum.data[5] = sum.data[5] + -4.999899864197e-01f;
    sum.data[6] = sum.data[6] + -4.999899864197e-01f;
    sum.data[7] = sum.data[7] + -4.999899864197e-01f;

    int4 dst;
    dst.x = ((int)hip_clamp(sum.data[0], -32768.0f, 32767.0f)) & 0xffff;
    dst.x |= ((int)hip_clamp(sum.data[1], -32768.0f, 32767.0f)) << 16;
    dst.y = ((int)hip_clamp(sum.data[2], -32768.0f, 32767.0f)) & 0xffff;
    dst.y |= ((int)hip_clamp(sum.data[3], -32768.0f, 32767.0f)) << 16;
    dst.z = ((int)hip_clamp(sum.data[4], -32768.0f, 32767.0f)) & 0xffff;
    dst.z |= ((int)hip_clamp(sum.data[5], -32768.0f, 32767.0f)) << 16;
    dst.w = ((int)hip_clamp(sum.data[6], -32768.0f, 32767.0f)) & 0xffff;
    dst.w |= ((int)hip_clamp(sum.data[7], -32768.0f, 32767.0f)) << 16;

    uint dstIdx = y * dstImageStrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage[dstIdx])) = dst;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_S16_U8_7x7(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes,
    int srcImageBufferSize, float *conv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2992]; // 136x22 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x22 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 3) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 6) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 6) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 3) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 22) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[ 0], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[ 1], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 0], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 2], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 1], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 0], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 3], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 2], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 1], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 0], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 4], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 3], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 2], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 1], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 0], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[ 5], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 4], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 3], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 2], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 1], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 0], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[ 6], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 5], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 4], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 3], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 2], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 1], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 0], sum.data[6]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[ 6], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 5], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 4], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 3], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 2], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 1], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 0], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[ 6], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 5], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 4], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 3], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 2], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 1], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[ 6], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 5], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 4], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 3], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 2], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[ 6], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 5], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 4], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 3], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[ 6], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 5], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 4], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[ 6], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 5], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[ 6], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[ 7], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[ 8], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 7], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 9], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 8], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 7], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[10], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 9], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 8], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 7], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[11], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[10], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 9], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 8], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 7], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[12], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[11], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[10], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 9], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 8], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 7], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[13], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[12], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[11], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[10], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 9], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 8], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 7], sum.data[6]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[13], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[12], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[11], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[10], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 9], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 8], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 7], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[13], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[12], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[11], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[10], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 9], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 8], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[13], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[12], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[11], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[10], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 9], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[13], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[12], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[11], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[10], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[13], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[12], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[11], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[13], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[12], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[13], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[14], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[15], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[14], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[16], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[15], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[14], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[17], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[16], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[15], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[14], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[18], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[17], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[16], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[15], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[14], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[19], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[18], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[17], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[16], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[15], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[14], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[20], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[19], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[18], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[17], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[16], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[15], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[14], sum.data[6]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[20], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[19], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[18], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[17], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[16], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[15], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[14], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[20], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[19], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[18], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[17], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[16], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[15], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[20], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[19], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[18], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[17], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[16], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[20], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[19], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[18], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[17], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[20], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[19], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[18], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[20], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[19], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[20], sum.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[21], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[22], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[21], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[23], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[22], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[21], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[24], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[23], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[22], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[21], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[25], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[24], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[23], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[22], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[21], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[26], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[25], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[24], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[23], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[22], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[21], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[27], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[26], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[25], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[24], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[23], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[22], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[21], sum.data[6]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[27], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[26], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[25], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[24], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[23], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[22], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[21], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[27], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[26], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[25], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[24], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[23], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[22], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[27], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[26], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[25], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[24], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[23], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[27], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[26], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[25], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[24], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[27], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[26], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[25], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[27], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[26], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[27], sum.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[28], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[29], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[28], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[30], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[29], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[28], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[31], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[30], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[29], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[28], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[32], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[31], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[30], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[29], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[28], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[33], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[32], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[31], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[30], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[29], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[28], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[34], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[33], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[32], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[31], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[30], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[29], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[28], sum.data[6]);
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[34], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[33], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[32], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[31], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[30], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[29], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[28], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[34], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[33], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[32], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[31], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[30], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[29], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[34], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[33], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[32], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[31], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[30], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[34], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[33], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[32], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[31], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[34], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[33], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[32], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[34], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[33], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[34], sum.data[7]);
    // filterRow = 5
    pix = lbufptr[85];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[35], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[36], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[35], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[37], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[36], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[35], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[38], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[37], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[36], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[35], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[39], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[38], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[37], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[36], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[35], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[40], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[39], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[38], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[37], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[36], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[35], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[41], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[40], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[39], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[38], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[37], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[36], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[35], sum.data[6]);
    pix = lbufptr[86];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[41], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[40], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[39], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[38], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[37], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[36], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[35], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[41], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[40], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[39], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[38], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[37], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[36], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[41], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[40], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[39], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[38], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[37], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[41], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[40], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[39], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[38], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[41], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[40], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[39], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[41], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[40], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[41], sum.data[7]);
    // filterRow = 6
    pix = lbufptr[102];
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[42], sum.data[0]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[43], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[42], sum.data[1]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[44], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[43], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[42], sum.data[2]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[45], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[44], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[43], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[42], sum.data[3]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[46], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[45], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[44], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[43], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[42], sum.data[4]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[47], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[46], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[45], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[44], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[43], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[42], sum.data[5]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[48], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[47], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[46], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[45], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[44], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[43], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[42], sum.data[6]);
    pix = lbufptr[103];
    fval = hip_unpack0(pix.x);
    sum.data[1] = fmaf(fval, conv[48], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[47], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[46], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[45], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[44], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[43], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[42], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[2] = fmaf(fval, conv[48], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[47], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[46], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[45], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[44], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[43], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[3] = fmaf(fval, conv[48], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[47], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[46], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[45], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[44], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[4] = fmaf(fval, conv[48], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[47], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[46], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[45], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[5] = fmaf(fval, conv[48], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[47], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[46], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[6] = fmaf(fval, conv[48], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[47], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[7] = fmaf(fval, conv[48], sum.data[7]);
    sum.data[0] += -4.999899864197e-01f;
    sum.data[1] += -4.999899864197e-01f;
    sum.data[2] += -4.999899864197e-01f;
    sum.data[3] += -4.999899864197e-01f;
    sum.data[4] += -4.999899864197e-01f;
    sum.data[5] += -4.999899864197e-01f;
    sum.data[6] += -4.999899864197e-01f;
    sum.data[7] += -4.999899864197e-01f;

    int4 dst;
    dst.x = ((int)hip_clamp(sum.data[0], -32768.0f, 32767.0f)) & 0xffff;
    dst.x |= ((int)hip_clamp(sum.data[1], -32768.0f, 32767.0f)) << 16;
    dst.y = ((int)hip_clamp(sum.data[2], -32768.0f, 32767.0f)) & 0xffff;
    dst.y |= ((int)hip_clamp(sum.data[3], -32768.0f, 32767.0f)) << 16;
    dst.z = ((int)hip_clamp(sum.data[4], -32768.0f, 32767.0f)) & 0xffff;
    dst.z |= ((int)hip_clamp(sum.data[5], -32768.0f, 32767.0f)) << 16;
    dst.w = ((int)hip_clamp(sum.data[6], -32768.0f, 32767.0f)) & 0xffff;
    dst.w |= ((int)hip_clamp(sum.data[7], -32768.0f, 32767.0f)) << 16;

    uint dstIdx = y * dstImageStrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage[dstIdx])) = dst;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_S16_U8_3x9(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes,
    int srcImageBufferSize, float *conv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[3264]; // 136x24 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x24 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 4) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 8) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 8) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 4) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 24) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 0], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 1], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 0], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 2], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 1], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 0], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[ 2], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 1], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 0], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[ 2], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 1], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 0], sum.data[4]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[ 2], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 1], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 0], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[ 2], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 1], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 0], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[ 2], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 1], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 0], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[ 2], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 1], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[ 2], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 3], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 4], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 3], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 5], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 4], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 3], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[ 5], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 4], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 3], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[ 5], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 4], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 3], sum.data[4]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[ 5], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 4], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 3], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[ 5], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 4], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 3], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[ 5], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 4], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 3], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[ 5], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 4], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[ 5], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 6], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 7], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 6], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 8], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 7], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 6], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[ 8], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 7], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 6], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[ 8], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 7], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 6], sum.data[4]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[ 8], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 7], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 6], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[ 8], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 7], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 6], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[ 8], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 7], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 6], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[ 8], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 7], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[ 8], sum.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 9], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[10], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 9], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[11], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[10], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 9], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[11], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[10], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 9], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[11], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[10], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 9], sum.data[4]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[11], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[10], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 9], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[11], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[10], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 9], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[11], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[10], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 9], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[11], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[10], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[11], sum.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[12], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[13], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[12], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[14], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[13], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[12], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[14], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[13], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[12], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[14], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[13], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[12], sum.data[4]);
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[14], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[13], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[12], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[14], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[13], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[12], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[14], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[13], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[12], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[14], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[13], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[14], sum.data[7]);
    // filterRow = 5
    pix = lbufptr[85];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[15], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[16], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[15], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[17], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[16], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[15], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[17], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[16], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[15], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[17], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[16], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[15], sum.data[4]);
    pix = lbufptr[86];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[17], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[16], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[15], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[17], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[16], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[15], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[17], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[16], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[15], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[17], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[16], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[17], sum.data[7]);
    // filterRow = 6
    pix = lbufptr[102];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[18], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[19], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[18], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[20], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[19], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[18], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[20], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[19], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[18], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[20], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[19], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[18], sum.data[4]);
    pix = lbufptr[103];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[20], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[19], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[18], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[20], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[19], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[18], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[20], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[19], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[18], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[20], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[19], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[20], sum.data[7]);
    // filterRow = 7
    pix = lbufptr[119];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[21], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[22], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[21], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[23], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[22], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[21], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[23], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[22], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[21], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[23], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[22], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[21], sum.data[4]);
    pix = lbufptr[120];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[23], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[22], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[21], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[23], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[22], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[21], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[23], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[22], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[21], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[23], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[22], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[23], sum.data[7]);
    // filterRow = 8
    pix = lbufptr[136];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[24], sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[25], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[24], sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[26], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[25], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[24], sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, conv[26], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[25], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[24], sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, conv[26], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[25], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[24], sum.data[4]);
    pix = lbufptr[137];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, conv[26], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[25], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[24], sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, conv[26], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[25], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[24], sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, conv[26], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[25], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[24], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, conv[26], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[25], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, conv[26], sum.data[7]);
    sum.data[0] += -4.999899864197e-01f;
    sum.data[1] += -4.999899864197e-01f;
    sum.data[2] += -4.999899864197e-01f;
    sum.data[3] += -4.999899864197e-01f;
    sum.data[4] += -4.999899864197e-01f;
    sum.data[5] += -4.999899864197e-01f;
    sum.data[6] += -4.999899864197e-01f;
    sum.data[7] += -4.999899864197e-01f;

    int4 dst;
    dst.x = ((int)hip_clamp(sum.data[0], -32768.0f, 32767.0f)) & 0xffff;
    dst.x |= ((int)hip_clamp(sum.data[1], -32768.0f, 32767.0f)) << 16;
    dst.y = ((int)hip_clamp(sum.data[2], -32768.0f, 32767.0f)) & 0xffff;
    dst.y |= ((int)hip_clamp(sum.data[3], -32768.0f, 32767.0f)) << 16;
    dst.z = ((int)hip_clamp(sum.data[4], -32768.0f, 32767.0f)) & 0xffff;
    dst.z |= ((int)hip_clamp(sum.data[5], -32768.0f, 32767.0f)) << 16;
    dst.w = ((int)hip_clamp(sum.data[6], -32768.0f, 32767.0f)) & 0xffff;
    dst.w |= ((int)hip_clamp(sum.data[7], -32768.0f, 32767.0f)) << 16;

    uint dstIdx = y * dstImageStrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage[dstIdx])) = dst;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_S16_U8_9x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes,
    int srcImageBufferSize, float *conv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
      pix = lbufptr[0];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[ 0], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[ 1], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 0], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[ 2], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 1], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 0], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 3], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 2], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 1], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 0], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 4], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 3], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 2], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 1], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 0], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 5], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 4], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 3], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 2], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 1], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 0], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[ 6], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 5], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 4], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 3], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 2], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 1], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 0], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[ 7], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 6], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 5], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 4], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 3], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 2], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 1], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 0], sum.data[7]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[ 8], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 7], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 6], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 5], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 4], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 3], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 2], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 1], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[ 8], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 7], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 6], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 5], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 4], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 3], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 2], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[ 8], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 7], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 6], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 5], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 4], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 3], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[ 8], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 7], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 6], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 5], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 4], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[ 8], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 7], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 6], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 5], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[ 8], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 7], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 6], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[ 8], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 7], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[ 8], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[ 9], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[10], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 9], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[11], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[10], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 9], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[12], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[11], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[10], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 9], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[13], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[12], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[11], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[10], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 9], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[14], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[13], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[12], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[11], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[10], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 9], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[15], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[14], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[13], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[12], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[11], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[10], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 9], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[16], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[15], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[14], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[13], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[12], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[11], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[10], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 9], sum.data[7]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[17], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[16], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[15], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[14], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[13], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[12], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[11], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[10], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[17], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[16], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[15], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[14], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[13], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[12], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[11], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[17], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[16], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[15], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[14], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[13], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[12], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[17], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[16], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[15], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[14], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[13], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[17], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[16], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[15], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[14], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[17], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[16], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[15], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[17], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[16], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[17], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[18], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[19], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[18], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[20], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[19], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[18], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[21], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[20], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[19], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[18], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[22], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[21], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[20], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[19], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[18], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[23], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[22], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[21], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[20], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[19], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[18], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[24], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[23], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[22], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[21], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[20], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[19], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[18], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[25], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[24], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[23], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[22], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[21], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[20], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[19], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[18], sum.data[7]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[26], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[25], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[24], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[23], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[22], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[21], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[20], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[19], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[26], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[25], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[24], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[23], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[22], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[21], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[20], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[26], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[25], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[24], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[23], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[22], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[21], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[26], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[25], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[24], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[23], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[22], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[26], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[25], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[24], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[23], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[26], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[25], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[24], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[26], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[25], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[26], sum.data[7]);

    sum.data[0] += -4.999899864197e-01f;
    sum.data[1] += -4.999899864197e-01f;
    sum.data[2] += -4.999899864197e-01f;
    sum.data[3] += -4.999899864197e-01f;
    sum.data[4] += -4.999899864197e-01f;
    sum.data[5] += -4.999899864197e-01f;
    sum.data[6] += -4.999899864197e-01f;
    sum.data[7] += -4.999899864197e-01f;

    int4 dst;
    dst.x = ((int)hip_clamp(sum.data[0], -32768.0f, 32767.0f)) & 0xffff;
    dst.x |= ((int)hip_clamp(sum.data[1], -32768.0f, 32767.0f)) << 16;
    dst.y = ((int)hip_clamp(sum.data[2], -32768.0f, 32767.0f)) & 0xffff;
    dst.y |= ((int)hip_clamp(sum.data[3], -32768.0f, 32767.0f)) << 16;
    dst.z = ((int)hip_clamp(sum.data[4], -32768.0f, 32767.0f)) & 0xffff;
    dst.z |= ((int)hip_clamp(sum.data[5], -32768.0f, 32767.0f)) << 16;
    dst.w = ((int)hip_clamp(sum.data[6], -32768.0f, 32767.0f)) & 0xffff;
    dst.w |= ((int)hip_clamp(sum.data[7], -32768.0f, 32767.0f)) << 16;

    uint dstIdx = y * dstImageStrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage[dstIdx])) = dst;
    }
}

__global__ void __attribute__((visibility("default")))
Hip_Convolve_S16_U8_9x9(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes,
    int srcImageBufferSize, int srcImageBufferOffset, float *conv) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[3264]; // 136x24 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x24 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 4) * srcImageStrideInBytes + x - 4;
        if (goffset > -srcImageBufferOffset && goffset < srcImageBufferSize)
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 8) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 8) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 4) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 24) ? true : false;
        }
        if (doExtraLoad && goffset > -srcImageBufferOffset && goffset < srcImageBufferSize) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[ 0], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[ 1], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 0], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[ 2], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 1], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 0], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[ 3], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 2], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 1], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 0], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[ 4], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 3], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 2], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 1], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 0], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[ 5], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 4], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 3], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 2], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 1], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 0], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[ 6], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 5], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 4], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 3], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 2], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 1], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 0], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[ 7], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 6], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 5], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 4], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 3], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 2], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 1], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 0], sum.data[7]);
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[ 8], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 7], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 6], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 5], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 4], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 3], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 2], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 1], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[ 8], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 7], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 6], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 5], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 4], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 3], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 2], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[ 8], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 7], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 6], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 5], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 4], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 3], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[ 8], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 7], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 6], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 5], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 4], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[ 8], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 7], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 6], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 5], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[ 8], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 7], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 6], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[ 8], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 7], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[ 8], sum.data[7]);
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[ 9], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[10], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[ 9], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[11], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[10], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[ 9], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[12], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[11], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[10], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[ 9], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[13], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[12], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[11], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[10], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[ 9], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[14], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[13], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[12], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[11], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[10], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[ 9], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[15], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[14], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[13], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[12], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[11], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[10], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[ 9], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[16], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[15], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[14], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[13], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[12], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[11], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[10], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[ 9], sum.data[7]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[17], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[16], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[15], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[14], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[13], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[12], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[11], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[10], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[17], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[16], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[15], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[14], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[13], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[12], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[11], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[17], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[16], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[15], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[14], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[13], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[12], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[17], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[16], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[15], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[14], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[13], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[17], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[16], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[15], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[14], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[17], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[16], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[15], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[17], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[16], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[17], sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[18], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[19], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[18], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[20], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[19], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[18], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[21], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[20], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[19], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[18], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[22], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[21], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[20], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[19], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[18], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[23], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[22], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[21], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[20], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[19], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[18], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[24], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[23], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[22], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[21], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[20], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[19], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[18], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[25], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[24], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[23], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[22], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[21], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[20], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[19], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[18], sum.data[7]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[26], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[25], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[24], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[23], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[22], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[21], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[20], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[19], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[26], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[25], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[24], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[23], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[22], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[21], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[20], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[26], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[25], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[24], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[23], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[22], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[21], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[26], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[25], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[24], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[23], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[22], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[26], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[25], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[24], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[23], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[26], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[25], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[24], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[26], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[25], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[26], sum.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[27], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[28], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[27], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[29], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[28], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[27], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[30], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[29], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[28], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[27], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[31], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[30], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[29], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[28], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[27], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[32], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[31], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[30], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[29], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[28], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[27], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[33], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[32], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[31], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[30], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[29], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[28], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[27], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[34], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[33], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[32], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[31], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[30], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[29], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[28], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[27], sum.data[7]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[35], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[34], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[33], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[32], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[31], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[30], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[29], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[28], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[35], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[34], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[33], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[32], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[31], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[30], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[29], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[35], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[34], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[33], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[32], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[31], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[30], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[35], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[34], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[33], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[32], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[31], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[35], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[34], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[33], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[32], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[35], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[34], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[33], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[35], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[34], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[35], sum.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[36], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[37], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[36], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[38], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[37], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[36], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[39], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[38], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[37], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[36], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[40], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[39], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[38], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[37], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[36], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[41], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[40], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[39], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[38], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[37], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[36], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[42], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[41], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[40], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[39], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[38], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[37], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[36], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[43], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[42], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[41], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[40], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[39], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[38], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[37], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[36], sum.data[7]);
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[44], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[43], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[42], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[41], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[40], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[39], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[38], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[37], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[44], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[43], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[42], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[41], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[40], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[39], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[38], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[44], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[43], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[42], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[41], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[40], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[39], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[44], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[43], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[42], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[41], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[40], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[44], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[43], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[42], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[41], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[44], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[43], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[42], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[44], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[43], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[44], sum.data[7]);
    // filterRow = 5
    pix = lbufptr[85];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[45], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[46], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[45], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[47], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[46], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[45], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[48], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[47], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[46], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[45], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[49], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[48], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[47], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[46], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[45], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[50], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[49], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[48], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[47], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[46], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[45], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[51], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[50], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[49], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[48], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[47], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[46], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[45], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[52], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[51], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[50], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[49], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[48], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[47], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[46], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[45], sum.data[7]);
    pix = lbufptr[86];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[53], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[52], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[51], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[50], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[49], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[48], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[47], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[46], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[53], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[52], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[51], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[50], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[49], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[48], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[47], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[53], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[52], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[51], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[50], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[49], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[48], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[53], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[52], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[51], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[50], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[49], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[53], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[52], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[51], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[50], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[53], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[52], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[51], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[53], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[52], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[53], sum.data[7]);
    // filterRow = 6
    pix = lbufptr[102];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[54], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[55], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[54], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[56], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[55], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[54], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[57], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[56], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[55], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[54], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[58], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[57], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[56], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[55], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[54], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[59], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[58], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[57], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[56], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[55], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[54], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[60], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[59], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[58], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[57], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[56], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[55], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[54], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[61], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[60], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[59], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[58], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[57], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[56], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[55], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[54], sum.data[7]);
    pix = lbufptr[103];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[62], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[61], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[60], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[59], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[58], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[57], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[56], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[55], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[62], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[61], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[60], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[59], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[58], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[57], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[56], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[62], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[61], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[60], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[59], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[58], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[57], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[62], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[61], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[60], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[59], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[58], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[62], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[61], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[60], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[59], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[62], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[61], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[60], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[62], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[61], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[62], sum.data[7]);
    // filterRow = 7
    pix = lbufptr[119];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[63], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[64], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[63], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[65], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[64], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[63], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[66], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[65], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[64], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[63], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[67], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[66], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[65], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[64], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[63], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[68], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[67], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[66], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[65], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[64], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[63], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[69], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[68], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[67], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[66], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[65], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[64], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[63], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[70], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[69], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[68], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[67], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[66], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[65], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[64], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[63], sum.data[7]);
    pix = lbufptr[120];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[71], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[70], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[69], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[68], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[67], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[66], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[65], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[64], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[71], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[70], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[69], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[68], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[67], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[66], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[65], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[71], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[70], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[69], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[68], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[67], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[66], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[71], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[70], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[69], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[68], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[67], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[71], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[70], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[69], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[68], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[71], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[70], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[69], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[71], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[70], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[71], sum.data[7]);
    // filterRow = 8
    pix = lbufptr[136];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[72], sum.data[0]);
    fval = hip_unpack1(pix.x);
    sum.data[0] = fmaf(fval, conv[73], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[72], sum.data[1]);
    fval = hip_unpack2(pix.x);
    sum.data[0] = fmaf(fval, conv[74], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[73], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[72], sum.data[2]);
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, conv[75], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[74], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[73], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[72], sum.data[3]);
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, conv[76], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[75], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[74], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[73], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[72], sum.data[4]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, conv[77], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[76], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[75], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[74], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[73], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[72], sum.data[5]);
    fval = hip_unpack2(pix.y);
    sum.data[0] = fmaf(fval, conv[78], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[77], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[76], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[75], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[74], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[73], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[72], sum.data[6]);
    fval = hip_unpack3(pix.y);
    sum.data[0] = fmaf(fval, conv[79], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[78], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[77], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[76], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[75], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[74], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[73], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[72], sum.data[7]);
    pix = lbufptr[137];
    fval = hip_unpack0(pix.x);
    sum.data[0] = fmaf(fval, conv[80], sum.data[0]);
    sum.data[1] = fmaf(fval, conv[79], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[78], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[77], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[76], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[75], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[74], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[73], sum.data[7]);
    fval = hip_unpack1(pix.x);
    sum.data[1] = fmaf(fval, conv[80], sum.data[1]);
    sum.data[2] = fmaf(fval, conv[79], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[78], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[77], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[76], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[75], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[74], sum.data[7]);
    fval = hip_unpack2(pix.x);
    sum.data[2] = fmaf(fval, conv[80], sum.data[2]);
    sum.data[3] = fmaf(fval, conv[79], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[78], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[77], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[76], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[75], sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[3] = fmaf(fval, conv[80], sum.data[3]);
    sum.data[4] = fmaf(fval, conv[79], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[78], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[77], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[76], sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[4] = fmaf(fval, conv[80], sum.data[4]);
    sum.data[5] = fmaf(fval, conv[79], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[78], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[77], sum.data[7]);
    fval = hip_unpack1(pix.y);
    sum.data[5] = fmaf(fval, conv[80], sum.data[5]);
    sum.data[6] = fmaf(fval, conv[79], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[78], sum.data[7]);
    fval = hip_unpack2(pix.y);
    sum.data[6] = fmaf(fval, conv[80], sum.data[6]);
    sum.data[7] = fmaf(fval, conv[79], sum.data[7]);
    fval = hip_unpack3(pix.y);
    sum.data[7] = fmaf(fval, conv[80], sum.data[7]);

    int4 dst;
    dst.x  = ((int)hip_clamp(sum.data[0], -32768.0f, 32767.0f)) & 0xffff;
    dst.x |= ((int)hip_clamp(sum.data[1], -32768.0f, 32767.0f)) << 16;
    dst.y  = ((int)hip_clamp(sum.data[2], -32768.0f, 32767.0f)) & 0xffff;
    dst.y |= ((int)hip_clamp(sum.data[3], -32768.0f, 32767.0f)) << 16;
    dst.z  = ((int)hip_clamp(sum.data[4], -32768.0f, 32767.0f)) & 0xffff;
    dst.z |= ((int)hip_clamp(sum.data[5], -32768.0f, 32767.0f)) << 16;
    dst.w  = ((int)hip_clamp(sum.data[6], -32768.0f, 32767.0f)) & 0xffff;
    dst.w |= ((int)hip_clamp(sum.data[7], -32768.0f, 32767.0f)) << 16;

    uint dstIdx = y * dstImageStrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage[dstIdx])) = dst;
    }
}

int HipExec_Convolve_S16_U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, vx_uint32 srcImageBufferSize, vx_uint32 srcImageBufferOffset,
    float *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight) {

    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    dim3 gridDim = dim3(ceil((float)globalThreads_x/localThreads_x),
                        ceil((float)globalThreads_y/localThreads_y));

    dim3 blockDim = dim3(localThreads_x, localThreads_y);

    if ((convolutionWidth == 3) && (convolutionHeight == 3)) {
        hipLaunchKernelGGL(Hip_Convolve_S16_U8_3x3, gridDim, blockDim, 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                            conv);
    } else if (convolutionWidth == 5 && convolutionHeight == 5) {
        hipLaunchKernelGGL(Hip_Convolve_S16_U8_5x5, gridDim, blockDim, 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                            conv);
    } else if (convolutionWidth == 7 && convolutionHeight == 7) {
        hipLaunchKernelGGL(Hip_Convolve_S16_U8_7x7, gridDim, blockDim, 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                            conv);
    } else if (convolutionWidth == 9 && convolutionHeight == 9) {
        hipLaunchKernelGGL(Hip_Convolve_S16_U8_9x9, gridDim, blockDim, 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize, srcImageBufferOffset,
                            conv);
    } else if (convolutionWidth == 3 && convolutionHeight == 9) {
        hipLaunchKernelGGL(Hip_Convolve_S16_U8_3x9, gridDim, blockDim, 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                            conv);
    } else if (convolutionWidth == 9 && convolutionHeight == 3) {
        hipLaunchKernelGGL(Hip_Convolve_S16_U8_9x3, gridDim, blockDim, 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                            conv);
    } else {
        return VX_ERROR_NOT_IMPLEMENTED;
    }

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxSobel kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Sobel_S16_U8_3x3_GX(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack3(pix.x);
    sum.data[0] -= fval;
    fval = hip_unpack0(pix.y);
    sum.data[1] -= fval;
    fval = hip_unpack1(pix.y);
    sum.data[0] += fval;
    sum.data[2] -= fval;
    fval = hip_unpack2(pix.y);
    sum.data[1] += fval;
    sum.data[3] -= fval;
    fval = hip_unpack3(pix.y);
    sum.data[2] += fval;
    sum.data[4] -= fval;
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[3] += fval;
    sum.data[5] -= fval;
    fval = hip_unpack1(pix.x);
    sum.data[4] += fval;
    sum.data[6] -= fval;
    fval = hip_unpack2(pix.x);
    sum.data[5] += fval;
    sum.data[7] -= fval;
    fval = hip_unpack3(pix.x);
    sum.data[6] += fval;
    fval = hip_unpack0(pix.y);
    sum.data[7] += fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack3(pix.x);
    sum.data[0] = fmaf(fval, -2.000000000000e+00f, sum.data[0]);
    fval = hip_unpack0(pix.y);
    sum.data[1] = fmaf(fval, -2.000000000000e+00f, sum.data[1]);
    fval = hip_unpack1(pix.y);
    sum.data[0] = fmaf(fval, 2.000000000000e+00f, sum.data[0]);
    sum.data[2] = fmaf(fval, -2.000000000000e+00f, sum.data[2]);
    fval = hip_unpack2(pix.y);
    sum.data[1] = fmaf(fval, 2.000000000000e+00f, sum.data[1]);
    sum.data[3] = fmaf(fval, -2.000000000000e+00f, sum.data[3]);
    fval = hip_unpack3(pix.y);
    sum.data[2] = fmaf(fval, 2.000000000000e+00f, sum.data[2]);
    sum.data[4] = fmaf(fval, -2.000000000000e+00f, sum.data[4]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum.data[3] = fmaf(fval, 2.000000000000e+00f, sum.data[3]);
    sum.data[5] = fmaf(fval, -2.000000000000e+00f, sum.data[5]);
    fval = hip_unpack1(pix.x);
    sum.data[4] = fmaf(fval, 2.000000000000e+00f, sum.data[4]);
    sum.data[6] = fmaf(fval, -2.000000000000e+00f, sum.data[6]);
    fval = hip_unpack2(pix.x);
    sum.data[5] = fmaf(fval, 2.000000000000e+00f, sum.data[5]);
    sum.data[7] = fmaf(fval, -2.000000000000e+00f, sum.data[7]);
    fval = hip_unpack3(pix.x);
    sum.data[6] = fmaf(fval, 2.000000000000e+00f, sum.data[6]);
    fval = hip_unpack0(pix.y);
    sum.data[7] = fmaf(fval, 2.000000000000e+00f, sum.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack3(pix.x);
    sum.data[0] -= fval;
    fval = hip_unpack0(pix.y);
    sum.data[1] -= fval;
    fval = hip_unpack1(pix.y);
    sum.data[0] += fval;
    sum.data[2] -= fval;
    fval = hip_unpack2(pix.y);
    sum.data[1] += fval;
    sum.data[3] -= fval;
    fval = hip_unpack3(pix.y);
    sum.data[2] += fval;
    sum.data[4] -= fval;
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[3] += fval;
    sum.data[5] -= fval;
    fval = hip_unpack1(pix.x);
    sum.data[4] += fval;
    sum.data[6] -= fval;
    fval = hip_unpack2(pix.x);
    sum.data[5] += fval;
    sum.data[7] -= fval;
    fval = hip_unpack3(pix.x);
    sum.data[6] += fval;
    fval = hip_unpack0(pix.y);
    sum.data[7] += fval;

    int4 dst;
    dst.x = ((int)sum.data[0]) & 0xffff;
    dst.x |= ((int)sum.data[1]) << 16;
    dst.y = ((int)sum.data[2]) & 0xffff;
    dst.y |= ((int)sum.data[3]) << 16;
    dst.z = ((int)sum.data[4]) & 0xffff;
    dst.z |= ((int)sum.data[5]) << 16;
    dst.w = ((int)sum.data[6]) & 0xffff;
    dst.w |= ((int)sum.data[7]) << 16;

    uint dstIdx = y * dstImageStrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Sobel_S16_U8_3x3_GX(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sobel_S16_U8_3x3_GX, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sobel_S16_U8_3x3_GY(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack3(pix.x);
    sum.data[0] -= fval;
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, -2.000000000000e+00f, sum.data[0]);
    sum.data[1] -= fval;
    fval = hip_unpack1(pix.y);
    sum.data[0] -= fval;
    sum.data[1] = fmaf(fval, -2.000000000000e+00f, sum.data[1]);
    sum.data[2] -= fval;
    fval = hip_unpack2(pix.y);
    sum.data[1] -= fval;
    sum.data[2] = fmaf(fval, -2.000000000000e+00f, sum.data[2]);
    sum.data[3] -= fval;
    fval = hip_unpack3(pix.y);
    sum.data[2] -= fval;
    sum.data[3] = fmaf(fval, -2.000000000000e+00f, sum.data[3]);
    sum.data[4] -= fval;
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum.data[3] -= fval;
    sum.data[4] = fmaf(fval, -2.000000000000e+00f, sum.data[4]);
    sum.data[5] -= fval;
    fval = hip_unpack1(pix.x);
    sum.data[4] -= fval;
    sum.data[5] = fmaf(fval, -2.000000000000e+00f, sum.data[5]);
    sum.data[6] -= fval;
    fval = hip_unpack2(pix.x);
    sum.data[5] -= fval;
    sum.data[6] = fmaf(fval, -2.000000000000e+00f, sum.data[6]);
    sum.data[7] -= fval;
    fval = hip_unpack3(pix.x);
    sum.data[6] -= fval;
    sum.data[7] = fmaf(fval, -2.000000000000e+00f, sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] -= fval;
    // filterRow = 1
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack3(pix.x);
    sum.data[0] += fval;
    fval = hip_unpack0(pix.y);
    sum.data[0] = fmaf(fval, 2.000000000000e+00f, sum.data[0]);
    sum.data[1] += fval;
    fval = hip_unpack1(pix.y);
    sum.data[0] += fval;
    sum.data[1] = fmaf(fval, 2.000000000000e+00f, sum.data[1]);
    sum.data[2] += fval;
    fval = hip_unpack2(pix.y);
    sum.data[1] += fval;
    sum.data[2] = fmaf(fval, 2.000000000000e+00f, sum.data[2]);
    sum.data[3] += fval;
    fval = hip_unpack3(pix.y);
    sum.data[2] += fval;
    sum.data[3] = fmaf(fval, 2.000000000000e+00f, sum.data[3]);
    sum.data[4] += fval;
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum.data[3] += fval;
    sum.data[4] = fmaf(fval, 2.000000000000e+00f, sum.data[4]);
    sum.data[5] += fval;
    fval = hip_unpack1(pix.x);
    sum.data[4] += fval;
    sum.data[5] = fmaf(fval, 2.000000000000e+00f, sum.data[5]);
    sum.data[6] += fval;
    fval = hip_unpack2(pix.x);
    sum.data[5] += fval;
    sum.data[6] = fmaf(fval, 2.000000000000e+00f, sum.data[6]);
    sum.data[7] += fval;
    fval = hip_unpack3(pix.x);
    sum.data[6] += fval;
    sum.data[7] = fmaf(fval, 2.000000000000e+00f, sum.data[7]);
    fval = hip_unpack0(pix.y);
    sum.data[7] += fval;

    int4 dst;
    dst.x = ((int)sum.data[0]) & 0xffff;
    dst.x |= ((int)sum.data[1]) << 16;
    dst.y = ((int)sum.data[2]) & 0xffff;
    dst.y |= ((int)sum.data[3]) << 16;
    dst.z = ((int)sum.data[4]) & 0xffff;
    dst.z |= ((int)sum.data[5]) << 16;
    dst.w = ((int)sum.data[6]) & 0xffff;
    dst.w |= ((int)sum.data[7]) << 16;

    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_Sobel_S16_U8_3x3_GY(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sobel_S16_U8_3x3_GY, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Sobel_S16S16_U8_3x3_GXY(uint dstWidth, uint dstHeight,
    uchar *pDstImage1, uint dstImage1StrideInBytes,
    uchar *pDstImage2, uint dstImage2StrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidth) && (y < dstHeight);

    __shared__ uchar lbuf[2448]; // 136x18 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + x - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + (((x >> 3) - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum1 = {0.0f};
    d_float8 sum2 = {0.0f};
    uint2 pix;
    float fval;
    uint2 *lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack3(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] -= fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0] = fmaf(fval, -2.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] -= fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] -= fval;
    sum2.data[1] = fmaf(fval, -2.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] -= fval;
    fval = hip_unpack2(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] -= fval;
    sum2.data[2] = fmaf(fval, -2.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] -= fval;
    fval = hip_unpack3(pix.y);
    sum1.data[2] += fval;
    sum2.data[2] -= fval;
    sum2.data[3] = fmaf(fval, -2.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] -= fval;
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] -= fval;
    sum2.data[4] = fmaf(fval, -2.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] -= fval;
    fval = hip_unpack1(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] -= fval;
    sum2.data[5] = fmaf(fval, -2.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] -= fval;
    fval = hip_unpack2(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] -= fval;
    sum2.data[6] = fmaf(fval, -2.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] -= fval;
    fval = hip_unpack3(pix.x);
    sum1.data[6] += fval;
    sum2.data[6] -= fval;
    sum2.data[7] = fmaf(fval, -2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] -= fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack3(pix.x);
    sum1.data[0] = fmaf(fval, -2.000000000000e+00f, sum1.data[0]);
    fval = hip_unpack0(pix.y);
    sum1.data[1] = fmaf(fval, -2.000000000000e+00f, sum1.data[1]);
    fval = hip_unpack1(pix.y);
    sum1.data[0] = fmaf(fval, 2.000000000000e+00f, sum1.data[0]);
    sum1.data[2] = fmaf(fval, -2.000000000000e+00f, sum1.data[2]);
    fval = hip_unpack2(pix.y);
    sum1.data[1] = fmaf(fval, 2.000000000000e+00f, sum1.data[1]);
    sum1.data[3] = fmaf(fval, -2.000000000000e+00f, sum1.data[3]);
    fval = hip_unpack3(pix.y);
    sum1.data[2] = fmaf(fval, 2.000000000000e+00f, sum1.data[2]);
    sum1.data[4] = fmaf(fval, -2.000000000000e+00f, sum1.data[4]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum1.data[3] = fmaf(fval, 2.000000000000e+00f, sum1.data[3]);
    sum1.data[5] = fmaf(fval, -2.000000000000e+00f, sum1.data[5]);
    fval = hip_unpack1(pix.x);
    sum1.data[4] = fmaf(fval, 2.000000000000e+00f, sum1.data[4]);
    sum1.data[6] = fmaf(fval, -2.000000000000e+00f, sum1.data[6]);
    fval = hip_unpack2(pix.x);
    sum1.data[5] = fmaf(fval, 2.000000000000e+00f, sum1.data[5]);
    sum1.data[7] = fmaf(fval, -2.000000000000e+00f, sum1.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[6] = fmaf(fval, 2.000000000000e+00f, sum1.data[6]);
    fval = hip_unpack0(pix.y);
    sum1.data[7] = fmaf(fval, 2.000000000000e+00f, sum1.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack3(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] += fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0] = fmaf(fval, 2.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] += fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] += fval;
    sum2.data[1] = fmaf(fval, 2.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] += fval;
    fval = hip_unpack2(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] += fval;
    sum2.data[2] = fmaf(fval, 2.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] += fval;
    fval = hip_unpack3(pix.y);
    sum1.data[2] += fval;
    sum2.data[2] += fval;
    sum2.data[3] = fmaf(fval, 2.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] += fval;
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] += fval;
    sum2.data[4] = fmaf(fval, 2.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] += fval;
    fval = hip_unpack1(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] += fval;
    sum2.data[5] = fmaf(fval, 2.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] += fval;
    fval = hip_unpack2(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] += fval;
    sum2.data[6] = fmaf(fval, 2.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] += fval;
    fval = hip_unpack3(pix.x);
    sum1.data[6] += fval;
    sum2.data[6] += fval;
    sum2.data[7] = fmaf(fval, 2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] += fval;

    int4 dst1;
    dst1.x = ((int)sum1.data[0]) & 0xffff;
    dst1.x |= ((int)sum1.data[1]) << 16;
    dst1.y = ((int)sum1.data[2]) & 0xffff;
    dst1.y |= ((int)sum1.data[3]) << 16;
    dst1.z = ((int)sum1.data[4]) & 0xffff;
    dst1.z |= ((int)sum1.data[5]) << 16;
    dst1.w = ((int)sum1.data[6]) & 0xffff;
    dst1.w |= ((int)sum1.data[7]) << 16;

    int4 dst2;
    dst2.x = ((int)sum2.data[0]) & 0xffff;
    dst2.x |= ((int)sum2.data[1]) << 16;
    dst2.y = ((int)sum2.data[2]) & 0xffff;
    dst2.y |= ((int)sum2.data[3]) << 16;
    dst2.z = ((int)sum2.data[4]) & 0xffff;
    dst2.z |= ((int)sum2.data[5]) << 16;
    dst2.w = ((int)sum2.data[6]) & 0xffff;
    dst2.w |= ((int)sum2.data[7]) << 16;

    uint dst1Idx =  y * dstImage1StrideInBytes + x + x;
    uint dst2Idx =  y * dstImage2StrideInBytes + x + x;

    if (valid) {
        *((int4 *)(&pDstImage1[dst1Idx])) = dst1;
        *((int4 *)(&pDstImage2[dst2Idx])) = dst2;
    }
}
int HipExec_Sobel_S16S16_U8_3x3_GXY(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage1, vx_uint32 dstImage1StrideInBytes,
    vx_int16 *pHipDstImage2, vx_uint32 dstImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Sobel_S16S16_U8_3x3_GXY, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage1, dstImage1StrideInBytes, (uchar *)pHipDstImage2, dstImage2StrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxScaleGaussianHalf kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ScaleGaussianHalf_U8_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    uint srcWidth, uint srcHeight,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp) {

    __shared__ uchar lbuf[4488]; // 136x33 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    uint dstIdx =  y * dstImageStrideInBytes + (x << 2);
    int srcIdx =  (((y - ly) << 1) + 1) * srcImageStrideInBytes + ((x - lx) << 3);
    bool valid = ((x < dstWidthComp) && (y < dstHeight)) ? true : false;

    { // load 136x33 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (ly - 1) * srcImageStrideInBytes + (lx << 3) - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[srcIdx + goffset]));
        loffset += 16 * 136;
        goffset += 16 * srcImageStrideInBytes;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[srcIdx + goffset]));
        if (ly < 1) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[srcIdx + goffset]));
        }
        uchar *lbufptr;
        lbufptr = lbuf + 128;
        goffset = -srcImageStrideInBytes + 124;
        int id = ly * 16 + lx;
        if (id < 33) {
            *((uint2 *)(&lbufptr[id * 136])) = *((uint2 *)(&pSrcImage[srcIdx + goffset + id * srcImageStrideInBytes]));
        }
        __syncthreads();
    }

    uchar *lbuf_ptr;
    lbuf_ptr = lbuf + ly * 272 + (lx << 3);
    uint3 L0 = *((uint3 *)(&lbuf_ptr[4]));
    uint3 L1 = *((uint3 *)(&lbuf_ptr[140]));
    uint3 L2 = *((uint3 *)(&lbuf_ptr[276]));
    float4 sum;
    float v;
    v = hip_unpack0(L0.x);
    v = fmaf(hip_unpack0(L1.x), 2.0f, v);
    v += hip_unpack0(L2.x);
    sum.x = v;
    v = hip_unpack1(L0.x);
    v = fmaf(hip_unpack1(L1.x), 2.0f, v);
    v += hip_unpack1(L2.x);
    sum.x = fmaf(v, 2.0f, sum.x);
    v = hip_unpack2(L0.x);
    v = fmaf(hip_unpack2(L1.x), 2.0f, v);
    v += hip_unpack2(L2.x);
    sum.y = v;
    sum.x += v;
    v = hip_unpack3(L0.x);
    v = fmaf(hip_unpack3(L1.x), 2.0f, v);
    v += hip_unpack3(L2.x);
    sum.y = fmaf(v, 2.0f, sum.y);
    v = hip_unpack0(L0.y);
    v = fmaf(hip_unpack0(L1.y), 2.0f, v);
    v += hip_unpack0(L2.y);
    sum.z = v;
    sum.y += v;
    v = hip_unpack1(L0.y);
    v = fmaf(hip_unpack1(L1.y), 2.0f, v);
    v += hip_unpack1(L2.y);
    sum.z = fmaf(v, 2.0f, sum.z);
    v = hip_unpack2(L0.y);
    v = fmaf(hip_unpack2(L1.y), 2.0f, v);
    v += hip_unpack2(L2.y);
    sum.w = v;
    sum.z += v;
    v = hip_unpack3(L0.y);
    v = fmaf(hip_unpack3(L1.y), 2.0f, v);
    v += hip_unpack3(L2.y);
    sum.w = fmaf(v, 2.0f, sum.w);
    v = hip_unpack0(L0.z);
    v = fmaf(hip_unpack0(L1.z), 2.0f, v);
    v += hip_unpack0(L2.z);
    sum.w += v;
    sum = sum * (float4)0.0625f;

    if (valid) {
        *((uint *)(&pDstImage[dstIdx])) = hip_pack(sum);
    }
}
int HipExec_ScaleGaussianHalf_U8_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2;
    int globalThreads_y = dstHeight;

    vx_uint32 dstWidthComp = (dstWidth + 3) / 4;

    hipLaunchKernelGGL(Hip_ScaleGaussianHalf_U8_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        srcWidth, srcHeight, (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleGaussianHalf_U8_U8_5x5(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    uint srcWidth, uint srcHeight,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    int srcImageBufferSize, uint dstWidthComp) {

    __shared__ uchar lbuf[4760]; // 136x35 bytes
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    int srcStride = srcImageStrideInBytes;
    uint dstIdx =  y * dstImageStrideInBytes + (x << 2);
    int srcIdx =  (((y - ly) << 1) + 1) * srcStride + ((x - lx) << 3);
    bool valid = ((x < dstWidthComp) && (y < dstHeight)) ? true : false;

    { // load 136x35 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (ly - 2) * srcStride + (lx << 3) - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[srcIdx + goffset]));
        loffset += 16 * 136;
        goffset += 16 * srcStride;
        int sidx = srcIdx + goffset;
        if (sidx < srcImageBufferSize)
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[sidx]));
        if (ly < 3) {
            loffset += 16 * 136;
            goffset += 16 * srcStride;
            sidx = srcIdx + goffset;
            if (sidx < srcImageBufferSize)
                *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[sidx]));
        }
        uchar *lbufptr;
        lbufptr = lbuf + 128;
        goffset = -2 * srcStride + 124;
        int id = ly * 16 + lx;
        sidx = srcIdx + goffset + id * srcStride;
        if (id < 35 && sidx < srcImageBufferSize) {
            *((uint2 *)(&lbufptr[id * 136])) = *((uint2 *)(&pSrcImage[sidx]));
        }
        __syncthreads();
    }

    uchar *lbuf_ptr;
    lbuf_ptr = lbuf + ly * 136 + (lx << 3);
    float4 sum;
    float v;

    uint4 L0 = *((uint4 *)(&lbuf_ptr[0]));

    v = hip_unpack3(L0.x);
    sum.x = v;

    v = hip_unpack0(L0.y);
    sum.x = fmaf(v, 4.0f, sum.x);

    v = hip_unpack1(L0.y);
    sum.x = fmaf(v, 6.0f, sum.x);
    sum.y = v;

    v = hip_unpack2(L0.y);
    sum.x = fmaf(v, 4.0f, sum.x);
    sum.y = fmaf(v, 4.0f, sum.y);

    v = hip_unpack3(L0.y);
    sum.x += v;
    sum.y = fmaf(v, 6.0f, sum.y);
    sum.z = v;

    v = hip_unpack0(L0.z);
    sum.y = fmaf(v, 4.0f, sum.y);
    sum.z = fmaf(v, 4.0f, sum.z);

    v = hip_unpack1(L0.z);
    sum.y += v;
    sum.z = fmaf(v, 6.0f, sum.z);
    sum.w = v;

    v = hip_unpack2(L0.z);
    sum.z = fmaf(v, 4.0f, sum.z);
    sum.w = fmaf(v, 4.0f, sum.w);

    v = hip_unpack3(L0.z);
    sum.z += v;
    sum.w = fmaf(v, 6.0f, sum.w);

    v = hip_unpack0(L0.w);
    sum.w = fmaf(v, 4.0f, sum.w);

    v = hip_unpack1(L0.w);
    sum.w += v;

    L0.x = (uint)sum.x + (((uint)sum.y) << 16);
    L0.y = (uint)sum.z + (((uint)sum.w) << 16);
    *(uint2 *)lbuf_ptr = make_uint2(L0.x, L0.y);

    L0 = *((uint4 *)(&lbuf_ptr[2176]));

    v = hip_unpack3(L0.x);
    sum.x = v;

    v = hip_unpack0(L0.y);
    sum.x = fmaf(v, 4.0f, sum.x);

    v = hip_unpack1(L0.y);
    sum.x = fmaf(v, 6.0f, sum.x);
    sum.y = v;

    v = hip_unpack2(L0.y);
    sum.x = fmaf(v, 4.0f, sum.x);
    sum.y = fmaf(v, 4.0f, sum.y);

    v = hip_unpack3(L0.y);
    sum.x += v;
    sum.y = fmaf(v, 6.0f, sum.y);
    sum.z = v;

    v = hip_unpack0(L0.z);
    sum.y = fmaf(v, 4.0f, sum.y);
    sum.z = fmaf(v, 4.0f, sum.z);

    v = hip_unpack1(L0.z);
    sum.y += v;
    sum.z = fmaf(v, 6.0f, sum.z);
    sum.w = v;

    v = hip_unpack2(L0.z);
    sum.z = fmaf(v, 4.0f, sum.z);
    sum.w = fmaf(v, 4.0f, sum.w);

    v = hip_unpack3(L0.z);
    sum.z += v;
    sum.w = fmaf(v, 6.0f, sum.w);

    v = hip_unpack0(L0.w);
    sum.w = fmaf(v, 4.0f, sum.w);

    v = hip_unpack1(L0.w);
    sum.w += v;

    L0.x = (uint)sum.x + (((uint)sum.y) << 16);
    L0.y = (uint)sum.z + (((uint)sum.w) << 16);
    *(uint2 *)&lbuf_ptr[2176] = make_uint2(L0.x, L0.y);

    if (ly < 3) {
        L0 = *((uint4 *)(&lbuf_ptr[4352]));

        v = hip_unpack3(L0.x);
        sum.x = v;

        v = hip_unpack0(L0.y);
        sum.x = fmaf(v, 4.0f, sum.x);

        v = hip_unpack1(L0.y);
        sum.x = fmaf(v, 6.0f, sum.x);
        sum.y = v;

        v = hip_unpack2(L0.y);
        sum.x = fmaf(v, 4.0f, sum.x);
        sum.y = fmaf(v, 4.0f, sum.y);

        v = hip_unpack3(L0.y);
        sum.x += v;
        sum.y = fmaf(v, 6.0f, sum.y);
        sum.z = v;

        v = hip_unpack0(L0.z);
        sum.y = fmaf(v, 4.0f, sum.y);
        sum.z = fmaf(v, 4.0f, sum.z);

        v = hip_unpack1(L0.z);
        sum.y += v;
        sum.z = fmaf(v, 6.0f, sum.z);
        sum.w = v;

        v = hip_unpack2(L0.z);
        sum.z = fmaf(v, 4.0f, sum.z);
        sum.w = fmaf(v, 4.0f, sum.w);

        v = hip_unpack3(L0.z);
        sum.z += v;
        sum.w = fmaf(v, 6.0f, sum.w);

        v = hip_unpack0(L0.w);
        sum.w = fmaf(v, 4.0f, sum.w);

        v = hip_unpack1(L0.w);
        sum.w += v;

        L0.x = (uint)sum.x + (((uint)sum.y) << 16);
        L0.y = (uint)sum.z + (((uint)sum.w) << 16);
        *(uint2 *)&lbuf_ptr[4352] = make_uint2(L0.x, L0.y);
    }
    __syncthreads();

    lbuf_ptr += ly * 136;
    uint2 L0_01;

    L0_01 = *((uint2 *)(&lbuf_ptr[0]));
    sum.x = (float)(L0_01.x & 0xffff);
    sum.y = (float)(L0_01.x >> 16);
    sum.z = (float)(L0_01.y & 0xffff);
    sum.w = (float)(L0_01.y >> 16);

    L0_01 = *((uint2 *)(&lbuf_ptr[136]));
    sum.x = fmaf((float)(L0_01.x & 0xffff), 4.0f, sum.x);
    sum.y = fmaf((float)(L0_01.x >> 16), 4.0f, sum.y);
    sum.z = fmaf((float)(L0_01.y & 0xffff), 4.0f, sum.z);
    sum.w = fmaf((float)(L0_01.y >> 16), 4.0f, sum.w);

    L0_01 = *((uint2 *)(&lbuf_ptr[272]));
    sum.x = fmaf((float)(L0_01.x & 0xffff), 6.0f, sum.x);
    sum.y = fmaf((float)(L0_01.x >> 16), 6.0f, sum.y);
    sum.z = fmaf((float)(L0_01.y & 0xffff), 6.0f, sum.z);
    sum.w = fmaf((float)(L0_01.y >> 16), 6.0f, sum.w);

    L0_01 = *((uint2 *)(&lbuf_ptr[408]));
    sum.x = fmaf((float)(L0_01.x & 0xffff), 4.0f, sum.x);
    sum.y = fmaf((float)(L0_01.x >> 16), 4.0f, sum.y);
    sum.z = fmaf((float)(L0_01.y & 0xffff), 4.0f, sum.z);
    sum.w = fmaf((float)(L0_01.y >> 16), 4.0f, sum.w);

    L0_01 = *((uint2 *)(&lbuf_ptr[544]));
    sum.x += (float)(L0_01.x & 0xffff);
    sum.y += (float)(L0_01.x >> 16);
    sum.z += (float)(L0_01.y & 0xffff);
    sum.w += (float)(L0_01.y >> 16);

    sum = sum * (float4)0.00390625f;
    if (valid) {
        *((uint *)(&pDstImage[dstIdx])) = hip_pack(sum);
    }
}
int HipExec_ScaleGaussianHalf_U8_U8_5x5(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, vx_uint32 srcImageBufferSize) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2;
    int globalThreads_y = dstHeight;

    vx_uint32 dstWidthComp = (dstWidth + 3) / 4;

    hipLaunchKernelGGL(Hip_ScaleGaussianHalf_U8_U8_5x5, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        srcWidth, srcHeight, (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                        dstWidthComp);

    return VX_SUCCESS;
}
