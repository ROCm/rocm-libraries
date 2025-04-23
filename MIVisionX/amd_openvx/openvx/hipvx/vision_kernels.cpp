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
// VxCannyEdgeDetector kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_3x3_L1NORM(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes) {

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
    __shared__ uint2 * lbufptr;
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

    uint mask = HIPSELECT(0xffffu, 0u, y < 1);
    mask = HIPSELECT(0u, mask, y < (dstHeight - 1));
    uint4 dst;
    uint mp;

    mp = hip_canny_mag_phase_L1(sum1.data[0], sum2.data[0]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 1);
    dst.x = mp;
    mp = hip_canny_mag_phase_L1(sum1.data[1], sum2.data[1]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 0);
    dst.x |= (mp << 16);

    mp = hip_canny_mag_phase_L1(sum1.data[2], sum2.data[2]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 0);
    dst.y = mp;
    mp = hip_canny_mag_phase_L1(sum1.data[3], sum2.data[3]) & mask;
    dst.y |= (mp << 16);

    mp = hip_canny_mag_phase_L1(sum1.data[4], sum2.data[4]) & mask;
    dst.z = mp;
    mp = hip_canny_mag_phase_L1(sum1.data[5], sum2.data[5]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 6));
    dst.z |= (mp << 16);

    mp = hip_canny_mag_phase_L1(sum1.data[6], sum2.data[6]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 7));
    dst.w  =  mp;
    mp = hip_canny_mag_phase_L1(sum1.data[7], sum2.data[7]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 8));
    dst.w |= (mp << 16);

    uint dstIdx =  y * dstImageStrideInBytes + x + x;
    if (valid) {
        *((uint4 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_CannySobel_U16_U8_3x3_L1NORM(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_3x3_L1NORM, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_5x5_L1NORM(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes) {

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
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum1 = {0.0f};
    d_float8 sum2 = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack2(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] -= fval;
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -2.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -4.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] -= fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -2.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -4.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] -= fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 2.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -4.000000000000e+00f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -2.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -4.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] -= fval;
    fval = hip_unpack2(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] -= fval;
    sum1.data[1]  = fmaf(fval, 2.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -4.000000000000e+00f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -2.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -4.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] -= fval;
    fval = hip_unpack3(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] -= fval;
    sum1.data[2]  = fmaf(fval, 2.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -4.000000000000e+00f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -2.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -4.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] -= fval;
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum1.data[2] += fval;
    sum2.data[2] -= fval;
    sum1.data[3]  = fmaf(fval, 2.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -4.000000000000e+00f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -2.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -4.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] -= fval;
    fval = hip_unpack1(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] -= fval;
    sum1.data[4]  = fmaf(fval, 2.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -4.000000000000e+00f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -2.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -4.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] -= fval;
    fval = hip_unpack2(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] -= fval;
    sum1.data[5]  = fmaf(fval, 2.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -4.000000000000e+00f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -2.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] -= fval;
    sum1.data[6]  = fmaf(fval, 2.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -4.000000000000e+00f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[6] += fval;
    sum2.data[6] -= fval;
    sum1.data[7]  = fmaf(fval, 2.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] -= fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -2.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -8.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -8.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -2.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -1.200000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -8.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -8.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -2.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 8.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -8.000000000000e+00f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -1.200000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -8.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -8.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -2.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -2.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 8.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -8.000000000000e+00f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -1.200000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -8.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -8.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -2.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack3(pix.y);
    sum1.data[1]  = fmaf(fval, 4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -2.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 8.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -8.000000000000e+00f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -1.200000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -8.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -8.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -2.000000000000e+00f, sum2.data[5]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum1.data[2]  = fmaf(fval, 4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -2.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 8.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -8.000000000000e+00f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -1.200000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -8.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -8.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -2.000000000000e+00f, sum2.data[6]);
    fval = hip_unpack1(pix.x);
    sum1.data[3]  = fmaf(fval, 4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -2.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 8.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -8.000000000000e+00f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -1.200000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -8.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -8.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[4]  = fmaf(fval, 4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -2.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 8.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -8.000000000000e+00f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -1.200000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -8.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -8.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5]  = fmaf(fval, 4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -2.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 8.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -8.000000000000e+00f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -1.200000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[6]  = fmaf(fval, 4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -2.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 8.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -8.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7]  = fmaf(fval, 4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -2.000000000000e+00f, sum2.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+00f, sum1.data[0]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -1.200000000000e+01f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+00f, sum1.data[1]);
    fval = hip_unpack0(pix.y);
    sum1.data[1]  = fmaf(fval, -1.200000000000e+01f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+00f, sum1.data[2]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 1.200000000000e+01f, sum1.data[0]);
    sum1.data[2]  = fmaf(fval, -1.200000000000e+01f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+00f, sum1.data[3]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+00f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, 1.200000000000e+01f, sum1.data[1]);
    sum1.data[3]  = fmaf(fval, -1.200000000000e+01f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+00f, sum1.data[4]);
    fval = hip_unpack3(pix.y);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+00f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, 1.200000000000e+01f, sum1.data[2]);
    sum1.data[4]  = fmaf(fval, -1.200000000000e+01f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+00f, sum1.data[5]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+00f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, 1.200000000000e+01f, sum1.data[3]);
    sum1.data[5]  = fmaf(fval, -1.200000000000e+01f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+00f, sum1.data[6]);
    fval = hip_unpack1(pix.x);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+00f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, 1.200000000000e+01f, sum1.data[4]);
    sum1.data[6]  = fmaf(fval, -1.200000000000e+01f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+00f, sum1.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+00f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, 1.200000000000e+01f, sum1.data[5]);
    sum1.data[7]  = fmaf(fval, -1.200000000000e+01f, sum1.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+00f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, 1.200000000000e+01f, sum1.data[6]);
    fval = hip_unpack0(pix.y);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+00f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, 1.200000000000e+01f, sum1.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+00f, sum1.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 2.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -8.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 8.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 2.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 1.200000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -8.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 8.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 2.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 8.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 8.000000000000e+00f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 1.200000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -8.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 8.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 2.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 2.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 8.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 8.000000000000e+00f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 1.200000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -8.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 8.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 2.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack3(pix.y);
    sum1.data[1]  = fmaf(fval, 4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 2.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 8.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 8.000000000000e+00f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 1.200000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -8.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 8.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 2.000000000000e+00f, sum2.data[5]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum1.data[2]  = fmaf(fval, 4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 2.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 8.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 8.000000000000e+00f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 1.200000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -8.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 8.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 2.000000000000e+00f, sum2.data[6]);
    fval = hip_unpack1(pix.x);
    sum1.data[3]  = fmaf(fval, 4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 2.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 8.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 8.000000000000e+00f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 1.200000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -8.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 8.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[4]  = fmaf(fval, 4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 2.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 8.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 8.000000000000e+00f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 1.200000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -8.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 8.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5]  = fmaf(fval, 4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 2.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 8.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 8.000000000000e+00f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 1.200000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[6]  = fmaf(fval, 4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 2.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 8.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 8.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7]  = fmaf(fval, 4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 2.000000000000e+00f, sum2.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack2(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] += fval;
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -2.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 4.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] += fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -2.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 4.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] += fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 2.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 4.000000000000e+00f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -2.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 4.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] += fval;
    fval = hip_unpack2(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] += fval;
    sum1.data[1]  = fmaf(fval, 2.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 4.000000000000e+00f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -2.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 4.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] += fval;
    fval = hip_unpack3(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] += fval;
    sum1.data[2]  = fmaf(fval, 2.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 4.000000000000e+00f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -2.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 4.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] += fval;
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum1.data[2] += fval;
    sum2.data[2] += fval;
    sum1.data[3]  = fmaf(fval, 2.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 4.000000000000e+00f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -2.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 4.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] += fval;
    fval = hip_unpack1(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] += fval;
    sum1.data[4]  = fmaf(fval, 2.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 4.000000000000e+00f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -2.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 4.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] += fval;
    fval = hip_unpack2(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] += fval;
    sum1.data[5]  = fmaf(fval, 2.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 4.000000000000e+00f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -2.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] += fval;
    sum1.data[6]  = fmaf(fval, 2.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 4.000000000000e+00f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[6] += fval;
    sum2.data[6] += fval;
    sum1.data[7]  = fmaf(fval, 2.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] += fval;

    uint mask = HIPSELECT(0xffffu, 0u, y < 2);
    mask = HIPSELECT(0u, mask, y < (dstHeight - 2));
    uint4 dst;
    uint mp;

    mp = hip_canny_mag_phase_L1(sum1.data[0], sum2.data[0]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 2);
    dst.x = mp;

    mp = hip_canny_mag_phase_L1(sum1.data[1], sum2.data[1]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 1);
    dst.x |= (mp << 16);

    mp = hip_canny_mag_phase_L1(sum1.data[2], sum2.data[2]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 0);
    dst.y = mp;

    mp = hip_canny_mag_phase_L1(sum1.data[3], sum2.data[3]) & mask;
    dst.y |= (mp << 16);

    mp = hip_canny_mag_phase_L1(sum1.data[4], sum2.data[4]) & mask;
    dst.z = mp;

    mp = hip_canny_mag_phase_L1(sum1.data[5], sum2.data[5]) & mask;

    mp = HIPSELECT(0u, mp, x < (dstWidth - 7));
    dst.z |= (mp << 16);

    mp = hip_canny_mag_phase_L1(sum1.data[6], sum2.data[6]) & mask;

    mp = HIPSELECT(0u, mp, x < (dstWidth - 8));
    dst.w  =  mp;

    mp = hip_canny_mag_phase_L1(sum1.data[7], sum2.data[7]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 9));
    dst.w |= (mp << 16);

    uint dstIdx =  y * dstImageStrideInBytes + x + x;
    if (valid) {
        *((uint4 *)(&pDstImage[dstIdx])) = dst;
    }
}

int HipExec_CannySobel_U16_U8_5x5_L1NORM(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_5x5_L1NORM, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_7x7_L1NORM(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes) {

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
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum1 = {0.0f};
    d_float8 sum2 = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack1(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] -= fval;
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] -= fval;
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -5.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -1.500000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] -= fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -2.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -5.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -1.500000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] -= fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 5.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -1.500000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -2.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -5.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -1.500000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] -= fval;
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 5.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -1.500000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -2.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -5.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -1.500000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] -= fval;
    fval = hip_unpack3(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] -= fval;
    sum1.data[1]  = fmaf(fval, 4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 5.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -1.500000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -2.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -5.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -1.500000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] -= fval;
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum1.data[1] += fval;
    sum2.data[1] -= fval;
    sum1.data[2]  = fmaf(fval, 4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 5.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -1.500000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -2.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -5.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -1.500000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] -= fval;
    fval = hip_unpack1(pix.x);
    sum1.data[2] += fval;
    sum2.data[2] -= fval;
    sum1.data[3]  = fmaf(fval, 4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 5.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -1.500000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -2.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -5.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -1.500000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] -= fval;
    sum1.data[4]  = fmaf(fval, 4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 5.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -1.500000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -2.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -5.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -1.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] -= fval;
    sum1.data[5]  = fmaf(fval, 4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 5.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -1.500000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -2.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5] += fval;
    sum2.data[5] -= fval;
    sum1.data[6]  = fmaf(fval, 4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 5.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -1.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6] += fval;
    sum2.data[6] -= fval;
    sum1.data[7]  = fmaf(fval, 4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] -= fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -4.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -2.400000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -2.400000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -4.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -3.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -2.400000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -2.400000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -4.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -8.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -3.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -2.400000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -2.400000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -4.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 3.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -8.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -3.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -2.400000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -2.400000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -4.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 2.400000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -2.400000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 3.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -8.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -3.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -2.400000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -2.400000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -4.000000000000e+00f, sum2.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -4.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 2.400000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -2.400000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 3.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -8.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -3.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -2.400000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -2.400000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -4.000000000000e+00f, sum2.data[6]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -4.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 2.400000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -2.400000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 3.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -8.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -3.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -2.400000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -2.400000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -4.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 2.400000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -2.400000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 3.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -8.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -3.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -2.400000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -2.400000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -4.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 2.400000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -2.400000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 3.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -8.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -3.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -4.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 2.400000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -2.400000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 3.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -8.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -4.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 2.400000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -2.400000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 3.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -4.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 2.400000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -2.400000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -4.000000000000e+00f, sum2.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -1.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -5.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -3.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -1.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -5.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -7.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -7.500000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -3.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -1.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -5.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -1.000000000000e+02f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -7.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -7.500000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -3.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -1.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -5.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 7.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -7.500000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -1.000000000000e+02f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -7.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -7.500000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -3.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -1.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -5.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -3.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 7.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -7.500000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -1.000000000000e+02f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -7.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -7.500000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -3.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -1.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -5.000000000000e+00f, sum2.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 1.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -5.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -3.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 7.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -7.500000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -1.000000000000e+02f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -7.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -7.500000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -3.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -1.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -5.000000000000e+00f, sum2.data[6]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 1.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -5.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -3.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 7.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -7.500000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -1.000000000000e+02f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -7.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -7.500000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -3.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -1.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -5.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 1.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -5.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -3.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 7.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -7.500000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -1.000000000000e+02f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -7.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -7.500000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -3.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 1.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -5.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -3.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 7.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -7.500000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -1.000000000000e+02f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -7.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -7.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 1.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -5.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -3.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 7.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -7.500000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -1.000000000000e+02f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 1.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -5.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -3.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 7.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -7.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 1.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -5.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -3.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 1.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -5.000000000000e+00f, sum2.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -2.000000000000e+01f, sum1.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -8.000000000000e+01f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, -2.000000000000e+01f, sum1.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -1.000000000000e+02f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, -8.000000000000e+01f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, -2.000000000000e+01f, sum1.data[2]);
    fval = hip_unpack0(pix.y);
    sum1.data[1]  = fmaf(fval, -1.000000000000e+02f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, -8.000000000000e+01f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, -2.000000000000e+01f, sum1.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 1.000000000000e+02f, sum1.data[0]);
    sum1.data[2]  = fmaf(fval, -1.000000000000e+02f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, -8.000000000000e+01f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, -2.000000000000e+01f, sum1.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 8.000000000000e+01f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, 1.000000000000e+02f, sum1.data[1]);
    sum1.data[3]  = fmaf(fval, -1.000000000000e+02f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, -8.000000000000e+01f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, -2.000000000000e+01f, sum1.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 2.000000000000e+01f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, 8.000000000000e+01f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, 1.000000000000e+02f, sum1.data[2]);
    sum1.data[4]  = fmaf(fval, -1.000000000000e+02f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, -8.000000000000e+01f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, -2.000000000000e+01f, sum1.data[6]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 2.000000000000e+01f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, 8.000000000000e+01f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, 1.000000000000e+02f, sum1.data[3]);
    sum1.data[5]  = fmaf(fval, -1.000000000000e+02f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, -8.000000000000e+01f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, -2.000000000000e+01f, sum1.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 2.000000000000e+01f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, 8.000000000000e+01f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, 1.000000000000e+02f, sum1.data[4]);
    sum1.data[6]  = fmaf(fval, -1.000000000000e+02f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, -8.000000000000e+01f, sum1.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 2.000000000000e+01f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, 8.000000000000e+01f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, 1.000000000000e+02f, sum1.data[5]);
    sum1.data[7]  = fmaf(fval, -1.000000000000e+02f, sum1.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 2.000000000000e+01f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, 8.000000000000e+01f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, 1.000000000000e+02f, sum1.data[6]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 2.000000000000e+01f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, 8.000000000000e+01f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, 1.000000000000e+02f, sum1.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 2.000000000000e+01f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, 8.000000000000e+01f, sum1.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 2.000000000000e+01f, sum1.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -1.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 5.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 3.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -1.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 5.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -7.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 7.500000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 3.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -1.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 5.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 1.000000000000e+02f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -7.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 7.500000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 3.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -1.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 5.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 7.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 7.500000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 1.000000000000e+02f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -7.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 7.500000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 3.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -1.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 5.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 3.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 7.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 7.500000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 1.000000000000e+02f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -7.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 7.500000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 3.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -1.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 5.000000000000e+00f, sum2.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 1.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 5.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 3.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 7.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 7.500000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 1.000000000000e+02f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -7.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 7.500000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 3.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -1.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 5.000000000000e+00f, sum2.data[6]);
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 1.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 5.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 3.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 7.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 7.500000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 1.000000000000e+02f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -7.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 7.500000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 3.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -1.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 5.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 1.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 5.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 3.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 7.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 7.500000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 1.000000000000e+02f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -7.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 7.500000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 3.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 1.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 5.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 3.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 7.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 7.500000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 1.000000000000e+02f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -7.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 7.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 1.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 5.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 3.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 7.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 7.500000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 1.000000000000e+02f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 1.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 5.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 3.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 7.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 7.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 1.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 5.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 3.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 1.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 5.000000000000e+00f, sum2.data[7]);
    // filterRow = 5
    pix = lbufptr[85];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 4.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -2.400000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 2.400000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 4.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -3.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -2.400000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 2.400000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 4.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 8.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -3.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -2.400000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 2.400000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 4.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 3.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 8.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -3.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -2.400000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 2.400000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 4.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 2.400000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 2.400000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 3.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 8.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -3.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -2.400000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 2.400000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 4.000000000000e+00f, sum2.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 4.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 2.400000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 2.400000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 3.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 8.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -3.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -2.400000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 2.400000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 4.000000000000e+00f, sum2.data[6]);
    pix = lbufptr[86];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 4.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 2.400000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 2.400000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 3.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 8.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -3.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -2.400000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 2.400000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 4.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 2.400000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 2.400000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 3.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 8.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -3.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -2.400000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 2.400000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 4.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 2.400000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 2.400000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 3.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 8.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -3.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 4.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 2.400000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 2.400000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 3.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 8.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 4.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 2.400000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 2.400000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 3.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 4.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 2.400000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 2.400000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 4.000000000000e+00f, sum2.data[7]);
    // filterRow = 6
    pix = lbufptr[102];
    fval = hip_unpack1(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] += fval;
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] += fval;
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -5.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 1.500000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] += fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 2.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -5.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 1.500000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] += fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 5.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 1.500000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 2.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -5.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 1.500000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] += fval;
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 5.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 1.500000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 2.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -5.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 1.500000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] += fval;
    fval = hip_unpack3(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] += fval;
    sum1.data[1]  = fmaf(fval, 4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 5.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 1.500000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 2.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -5.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 1.500000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] += fval;
    pix = lbufptr[103];
    fval = hip_unpack0(pix.x);
    sum1.data[1] += fval;
    sum2.data[1] += fval;
    sum1.data[2]  = fmaf(fval, 4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 5.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 1.500000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 2.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -5.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 1.500000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] += fval;
    fval = hip_unpack1(pix.x);
    sum1.data[2] += fval;
    sum2.data[2] += fval;
    sum1.data[3]  = fmaf(fval, 4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 5.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 1.500000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 2.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -5.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 1.500000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] += fval;
    sum1.data[4]  = fmaf(fval, 4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 5.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 1.500000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 2.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -5.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 1.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] += fval;
    sum1.data[5]  = fmaf(fval, 4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 5.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 1.500000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 2.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5] += fval;
    sum2.data[5] += fval;
    sum1.data[6]  = fmaf(fval, 4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 5.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 1.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6] += fval;
    sum2.data[6] += fval;
    sum1.data[7]  = fmaf(fval, 4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] += fval;

    uint mask = HIPSELECT(0xffffu, 0u, y < 3);
    mask = HIPSELECT(0u, mask, y < (dstHeight - 3));
    uint4 dst;
    uint mp;

    mp = hip_canny_mag_phase_L1_7x7(sum1.data[0], sum2.data[0]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 3);
    dst.x = mp;

    mp = hip_canny_mag_phase_L1_7x7(sum1.data[1], sum2.data[1]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 2);
    dst.x |= (mp << 16);

    mp = hip_canny_mag_phase_L1_7x7(sum1.data[2], sum2.data[2]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 1);
    dst.y = mp;

    mp = hip_canny_mag_phase_L1_7x7(sum1.data[3], sum2.data[3]) & mask;
    dst.y |= (mp << 16);

    mp = hip_canny_mag_phase_L1_7x7(sum1.data[4], sum2.data[4]) & mask;
    dst.z = mp;

    mp = hip_canny_mag_phase_L1_7x7(sum1.data[5], sum2.data[5]) & mask;

    mp = HIPSELECT(0u, mp, x < (dstWidth - 8));
    dst.z |= (mp << 16);

    mp = hip_canny_mag_phase_L1_7x7(sum1.data[6], sum2.data[6]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 9));
    dst.w  =  mp;

    mp = hip_canny_mag_phase_L1_7x7(sum1.data[7], sum2.data[7]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 10));
    dst.w |= (mp << 16);

    uint dstIdx =  y * dstImageStrideInBytes + x + x;
    if (valid) {
        *((uint4 *)(&pDstImage[dstIdx])) = dst;
    }
}

int HipExec_CannySobel_U16_U8_7x7_L1NORM(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_7x7_L1NORM, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_3x3_L2NORM(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes) {

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
    __shared__ uint2 * lbufptr;
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

    uint mask = HIPSELECT(0xffffu, 0u, y < 1);
    mask = HIPSELECT(0u, mask, y < (dstHeight - 1));
    uint4 dst;
    uint mp;

    mp = hip_canny_mag_phase_L2(sum1.data[0], sum2.data[0]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 1);
    dst.x = mp;
    mp = hip_canny_mag_phase_L2(sum1.data[1], sum2.data[1]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 0);
    dst.x |= (mp << 16);

    mp = hip_canny_mag_phase_L2(sum1.data[2], sum2.data[2]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 0);
    dst.y = mp;
    mp = hip_canny_mag_phase_L2(sum1.data[3], sum2.data[3]) & mask;
    dst.y |= (mp << 16);

    mp = hip_canny_mag_phase_L2(sum1.data[4], sum2.data[4]) & mask;
    dst.z = mp;
    mp = hip_canny_mag_phase_L2(sum1.data[5], sum2.data[5]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 6));
    dst.z |= (mp << 16);

    mp = hip_canny_mag_phase_L2(sum1.data[6], sum2.data[6]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 7));
    dst.w  =  mp;
    mp = hip_canny_mag_phase_L2(sum1.data[7], sum2.data[7]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 8));
    dst.w |= (mp << 16);

    uint dstIdx =  y * dstImageStrideInBytes + x + x;
    if (valid) {
        *((uint4 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_CannySobel_U16_U8_3x3_L2NORM(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_3x3_L2NORM, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_5x5_L2NORM(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes) {

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
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum1 = {0.0f};
    d_float8 sum2 = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
      pix = lbufptr[0];
    fval = hip_unpack2(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] -= fval;
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -2.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -4.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] -= fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -2.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -4.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] -= fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 2.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -4.000000000000e+00f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -2.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -4.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] -= fval;
    fval = hip_unpack2(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] -= fval;
    sum1.data[1]  = fmaf(fval, 2.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -4.000000000000e+00f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -2.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -4.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] -= fval;
    fval = hip_unpack3(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] -= fval;
    sum1.data[2]  = fmaf(fval, 2.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -4.000000000000e+00f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -2.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -4.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] -= fval;
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum1.data[2] += fval;
    sum2.data[2] -= fval;
    sum1.data[3]  = fmaf(fval, 2.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -4.000000000000e+00f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -2.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -4.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] -= fval;
    fval = hip_unpack1(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] -= fval;
    sum1.data[4]  = fmaf(fval, 2.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -4.000000000000e+00f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -2.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -4.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] -= fval;
    fval = hip_unpack2(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] -= fval;
    sum1.data[5]  = fmaf(fval, 2.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -4.000000000000e+00f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -2.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] -= fval;
    sum1.data[6]  = fmaf(fval, 2.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -4.000000000000e+00f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[6] += fval;
    sum2.data[6] -= fval;
    sum1.data[7]  = fmaf(fval, 2.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] -= fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -2.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -8.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -8.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -2.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -1.200000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -8.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -8.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -2.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 8.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -8.000000000000e+00f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -1.200000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -8.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -8.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -2.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -2.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 8.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -8.000000000000e+00f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -1.200000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -8.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -8.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -2.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack3(pix.y);
    sum1.data[1]  = fmaf(fval, 4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -2.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 8.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -8.000000000000e+00f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -1.200000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -8.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -8.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -2.000000000000e+00f, sum2.data[5]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum1.data[2]  = fmaf(fval, 4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -2.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 8.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -8.000000000000e+00f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -1.200000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -8.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -8.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -2.000000000000e+00f, sum2.data[6]);
    fval = hip_unpack1(pix.x);
    sum1.data[3]  = fmaf(fval, 4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -2.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 8.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -8.000000000000e+00f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -1.200000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -8.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -8.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[4]  = fmaf(fval, 4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -2.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 8.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -8.000000000000e+00f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -1.200000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -8.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -8.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5]  = fmaf(fval, 4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -2.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 8.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -8.000000000000e+00f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -1.200000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[6]  = fmaf(fval, 4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -2.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 8.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -8.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7]  = fmaf(fval, 4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -2.000000000000e+00f, sum2.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+00f, sum1.data[0]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -1.200000000000e+01f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+00f, sum1.data[1]);
    fval = hip_unpack0(pix.y);
    sum1.data[1]  = fmaf(fval, -1.200000000000e+01f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+00f, sum1.data[2]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 1.200000000000e+01f, sum1.data[0]);
    sum1.data[2]  = fmaf(fval, -1.200000000000e+01f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+00f, sum1.data[3]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+00f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, 1.200000000000e+01f, sum1.data[1]);
    sum1.data[3]  = fmaf(fval, -1.200000000000e+01f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+00f, sum1.data[4]);
    fval = hip_unpack3(pix.y);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+00f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, 1.200000000000e+01f, sum1.data[2]);
    sum1.data[4]  = fmaf(fval, -1.200000000000e+01f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+00f, sum1.data[5]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+00f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, 1.200000000000e+01f, sum1.data[3]);
    sum1.data[5]  = fmaf(fval, -1.200000000000e+01f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+00f, sum1.data[6]);
    fval = hip_unpack1(pix.x);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+00f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, 1.200000000000e+01f, sum1.data[4]);
    sum1.data[6]  = fmaf(fval, -1.200000000000e+01f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+00f, sum1.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+00f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, 1.200000000000e+01f, sum1.data[5]);
    sum1.data[7]  = fmaf(fval, -1.200000000000e+01f, sum1.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+00f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, 1.200000000000e+01f, sum1.data[6]);
    fval = hip_unpack0(pix.y);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+00f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, 1.200000000000e+01f, sum1.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+00f, sum1.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 2.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -8.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 8.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 2.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 1.200000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -8.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 8.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 2.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 8.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 8.000000000000e+00f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 1.200000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -8.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 8.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 2.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 2.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 8.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 8.000000000000e+00f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 1.200000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -8.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 8.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 2.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack3(pix.y);
    sum1.data[1]  = fmaf(fval, 4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 2.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 8.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 8.000000000000e+00f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 1.200000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -8.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 8.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 2.000000000000e+00f, sum2.data[5]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum1.data[2]  = fmaf(fval, 4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 2.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 8.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 8.000000000000e+00f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 1.200000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -8.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 8.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 2.000000000000e+00f, sum2.data[6]);
    fval = hip_unpack1(pix.x);
    sum1.data[3]  = fmaf(fval, 4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 2.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 8.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 8.000000000000e+00f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 1.200000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -8.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 8.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[4]  = fmaf(fval, 4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 2.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 8.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 8.000000000000e+00f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 1.200000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -8.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 8.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5]  = fmaf(fval, 4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 2.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 8.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 8.000000000000e+00f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 1.200000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[6]  = fmaf(fval, 4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 2.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 8.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 8.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7]  = fmaf(fval, 4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 2.000000000000e+00f, sum2.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack2(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] += fval;
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -2.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 4.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] += fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -2.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 4.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] += fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 2.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 4.000000000000e+00f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -2.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 4.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] += fval;
    fval = hip_unpack2(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] += fval;
    sum1.data[1]  = fmaf(fval, 2.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 4.000000000000e+00f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -2.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 4.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] += fval;
    fval = hip_unpack3(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] += fval;
    sum1.data[2]  = fmaf(fval, 2.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 4.000000000000e+00f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -2.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 4.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] += fval;
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum1.data[2] += fval;
    sum2.data[2] += fval;
    sum1.data[3]  = fmaf(fval, 2.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 4.000000000000e+00f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -2.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 4.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] += fval;
    fval = hip_unpack1(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] += fval;
    sum1.data[4]  = fmaf(fval, 2.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 4.000000000000e+00f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -2.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 4.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] += fval;
    fval = hip_unpack2(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] += fval;
    sum1.data[5]  = fmaf(fval, 2.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 4.000000000000e+00f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -2.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] += fval;
    sum1.data[6]  = fmaf(fval, 2.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 4.000000000000e+00f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[6] += fval;
    sum2.data[6] += fval;
    sum1.data[7]  = fmaf(fval, 2.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] += fval;

    uint mask = HIPSELECT(0xffffu, 0u, y < 2);
    mask = HIPSELECT(0u, mask, y < (dstHeight - 2));
    uint4 dst;
    uint mp;

    mp = hip_canny_mag_phase_L2(sum1.data[0], sum2.data[0]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 2);
    dst.x = mp;

    mp = hip_canny_mag_phase_L2(sum1.data[1], sum2.data[1]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 1);
    dst.x |= (mp << 16);

    mp = hip_canny_mag_phase_L2(sum1.data[2], sum2.data[2]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 0);
    dst.y = mp;

    mp = hip_canny_mag_phase_L2(sum1.data[3], sum2.data[3]) & mask;
    dst.y |= (mp << 16);

    mp = hip_canny_mag_phase_L2(sum1.data[4], sum2.data[4]) & mask;
    dst.z = mp;

    mp = hip_canny_mag_phase_L2(sum1.data[5], sum2.data[5]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 7));
    dst.z |= (mp << 16);

    mp = hip_canny_mag_phase_L2(sum1.data[6], sum2.data[6]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 8));
    dst.w  =  mp;

    mp = hip_canny_mag_phase_L2(sum1.data[7], sum2.data[7]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 9));
    dst.w |= (mp << 16);

    uint dstIdx =  y * dstImageStrideInBytes + x + x;
    if (valid) {
        *((uint4 *)(&pDstImage[dstIdx])) = dst;
    }
}

int HipExec_CannySobel_U16_U8_5x5_L2NORM(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_5x5_L2NORM, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySobel_U16_U8_7x7_L2NORM(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes) {

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
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum1 = {0.0f};
    d_float8 sum2 = {0.0f};
    uint2 pix;
    float fval;
    __shared__ uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);
    // filterRow = 0
      pix = lbufptr[0];
    fval = hip_unpack1(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] -= fval;
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] -= fval;
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -5.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -1.500000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] -= fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -2.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -5.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -1.500000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] -= fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 5.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -1.500000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -2.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -5.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -1.500000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] -= fval;
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 5.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -1.500000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -2.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -5.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -1.500000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] -= fval;
    fval = hip_unpack3(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] -= fval;
    sum1.data[1]  = fmaf(fval, 4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 5.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -1.500000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -2.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -5.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -1.500000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] -= fval;
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum1.data[1] += fval;
    sum2.data[1] -= fval;
    sum1.data[2]  = fmaf(fval, 4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 5.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -1.500000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -2.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -5.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -1.500000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] -= fval;
    fval = hip_unpack1(pix.x);
    sum1.data[2] += fval;
    sum2.data[2] -= fval;
    sum1.data[3]  = fmaf(fval, 4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 5.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -1.500000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -2.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -5.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -1.500000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] -= fval;
    sum1.data[4]  = fmaf(fval, 4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 5.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -1.500000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -2.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -5.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -1.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] -= fval;
    sum1.data[5]  = fmaf(fval, 4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 5.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -1.500000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -2.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5] += fval;
    sum2.data[5] -= fval;
    sum1.data[6]  = fmaf(fval, 4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 5.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -1.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6] += fval;
    sum2.data[6] -= fval;
    sum1.data[7]  = fmaf(fval, 4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] -= fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -4.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -2.400000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -2.400000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -4.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -3.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -2.400000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -2.400000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -4.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -8.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -3.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -2.400000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -2.400000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -4.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 3.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -8.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -3.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -2.400000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -2.400000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -4.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 2.400000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -2.400000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 3.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -8.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -3.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -2.400000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -2.400000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -4.000000000000e+00f, sum2.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -4.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 2.400000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -2.400000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 3.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -8.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -3.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -2.400000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -2.400000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -4.000000000000e+00f, sum2.data[6]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -4.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 2.400000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -2.400000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 3.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -8.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -3.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -2.400000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -2.400000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -4.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 2.400000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -2.400000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 3.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -8.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -3.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -2.400000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -2.400000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -4.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 2.400000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -2.400000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 3.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -8.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -3.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -4.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 2.400000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -2.400000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 3.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -8.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -4.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 2.400000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -2.400000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 3.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -4.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 2.400000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -2.400000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -4.000000000000e+00f, sum2.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -1.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -5.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -3.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -1.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -5.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -7.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -7.500000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -3.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -1.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -5.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -1.000000000000e+02f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -7.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -7.500000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -3.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -1.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -5.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 7.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -7.500000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -1.000000000000e+02f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -7.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -7.500000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -3.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -1.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -5.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -3.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 7.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -7.500000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -1.000000000000e+02f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -7.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -7.500000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -3.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -1.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -5.000000000000e+00f, sum2.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 1.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -5.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -3.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 7.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -7.500000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -1.000000000000e+02f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -7.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -7.500000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -3.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -1.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -5.000000000000e+00f, sum2.data[6]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 1.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -5.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -3.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 7.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -7.500000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -1.000000000000e+02f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -7.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -7.500000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -3.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -1.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -5.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 1.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -5.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -3.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 7.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -7.500000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -1.000000000000e+02f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -7.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -7.500000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -3.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 1.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -5.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -3.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 7.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -7.500000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -1.000000000000e+02f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -7.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -7.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 1.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -5.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -3.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 7.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -7.500000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -1.000000000000e+02f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 1.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -5.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -3.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 7.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -7.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 1.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -5.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -3.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 1.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -5.000000000000e+00f, sum2.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -2.000000000000e+01f, sum1.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -8.000000000000e+01f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, -2.000000000000e+01f, sum1.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -1.000000000000e+02f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, -8.000000000000e+01f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, -2.000000000000e+01f, sum1.data[2]);
    fval = hip_unpack0(pix.y);
    sum1.data[1]  = fmaf(fval, -1.000000000000e+02f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, -8.000000000000e+01f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, -2.000000000000e+01f, sum1.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 1.000000000000e+02f, sum1.data[0]);
    sum1.data[2]  = fmaf(fval, -1.000000000000e+02f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, -8.000000000000e+01f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, -2.000000000000e+01f, sum1.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 8.000000000000e+01f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, 1.000000000000e+02f, sum1.data[1]);
    sum1.data[3]  = fmaf(fval, -1.000000000000e+02f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, -8.000000000000e+01f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, -2.000000000000e+01f, sum1.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 2.000000000000e+01f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, 8.000000000000e+01f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, 1.000000000000e+02f, sum1.data[2]);
    sum1.data[4]  = fmaf(fval, -1.000000000000e+02f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, -8.000000000000e+01f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, -2.000000000000e+01f, sum1.data[6]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 2.000000000000e+01f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, 8.000000000000e+01f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, 1.000000000000e+02f, sum1.data[3]);
    sum1.data[5]  = fmaf(fval, -1.000000000000e+02f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, -8.000000000000e+01f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, -2.000000000000e+01f, sum1.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 2.000000000000e+01f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, 8.000000000000e+01f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, 1.000000000000e+02f, sum1.data[4]);
    sum1.data[6]  = fmaf(fval, -1.000000000000e+02f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, -8.000000000000e+01f, sum1.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 2.000000000000e+01f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, 8.000000000000e+01f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, 1.000000000000e+02f, sum1.data[5]);
    sum1.data[7]  = fmaf(fval, -1.000000000000e+02f, sum1.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 2.000000000000e+01f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, 8.000000000000e+01f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, 1.000000000000e+02f, sum1.data[6]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 2.000000000000e+01f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, 8.000000000000e+01f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, 1.000000000000e+02f, sum1.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 2.000000000000e+01f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, 8.000000000000e+01f, sum1.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 2.000000000000e+01f, sum1.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -1.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 5.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 3.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -1.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 5.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -7.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 7.500000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 3.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -1.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 5.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 1.000000000000e+02f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -7.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 7.500000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 3.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -1.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 5.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 7.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 7.500000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 1.000000000000e+02f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -7.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 7.500000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 3.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -1.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 5.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 3.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 7.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 7.500000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 1.000000000000e+02f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -7.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 7.500000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 3.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -1.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 5.000000000000e+00f, sum2.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 1.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 5.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 3.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 7.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 7.500000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 1.000000000000e+02f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -7.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 7.500000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 3.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -1.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 5.000000000000e+00f, sum2.data[6]);
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 1.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 5.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 3.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 7.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 7.500000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 1.000000000000e+02f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -7.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 7.500000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 3.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -1.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 5.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 1.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 5.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 3.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 7.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 7.500000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 1.000000000000e+02f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -7.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 7.500000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 3.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 1.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 5.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 3.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 7.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 7.500000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 1.000000000000e+02f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -7.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 7.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 1.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 5.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 3.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 7.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 7.500000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 1.000000000000e+02f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 1.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 5.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 3.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 7.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 7.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 1.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 5.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 3.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 1.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 5.000000000000e+00f, sum2.data[7]);
    // filterRow = 5
    pix = lbufptr[85];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 4.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -2.400000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 2.400000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 4.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -3.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -2.400000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 2.400000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 4.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 8.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -3.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -2.400000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 2.400000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 4.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 3.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 8.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -3.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -2.400000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 2.400000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 4.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 2.400000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 2.400000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 3.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 8.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -3.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -2.400000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 2.400000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 4.000000000000e+00f, sum2.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 4.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 2.400000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 2.400000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 3.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 8.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -3.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -2.400000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 2.400000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 4.000000000000e+00f, sum2.data[6]);
    pix = lbufptr[86];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 4.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 2.400000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 2.400000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 3.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 8.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -3.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -2.400000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 2.400000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 4.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 2.400000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 2.400000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 3.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 8.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -3.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -2.400000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 2.400000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 4.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 2.400000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 2.400000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 3.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 8.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -3.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 4.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 2.400000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 2.400000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 3.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 8.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 4.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 2.400000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 2.400000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 3.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 4.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 2.400000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 2.400000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 4.000000000000e+00f, sum2.data[7]);
    // filterRow = 6
    pix = lbufptr[102];
    fval = hip_unpack1(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] += fval;
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] += fval;
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -5.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 1.500000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] += fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 2.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -5.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 1.500000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] += fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 5.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 1.500000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 2.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -5.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 1.500000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] += fval;
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 5.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 1.500000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 2.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -5.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 1.500000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] += fval;
    fval = hip_unpack3(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] += fval;
    sum1.data[1]  = fmaf(fval, 4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 5.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 1.500000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 2.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -5.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 1.500000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] += fval;
    pix = lbufptr[103];
    fval = hip_unpack0(pix.x);
    sum1.data[1] += fval;
    sum2.data[1] += fval;
    sum1.data[2]  = fmaf(fval, 4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 5.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 1.500000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 2.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -5.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 1.500000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] += fval;
    fval = hip_unpack1(pix.x);
    sum1.data[2] += fval;
    sum2.data[2] += fval;
    sum1.data[3]  = fmaf(fval, 4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 5.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 1.500000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 2.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -5.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 1.500000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] += fval;
    sum1.data[4]  = fmaf(fval, 4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 5.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 1.500000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 2.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -5.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 1.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] += fval;
    sum1.data[5]  = fmaf(fval, 4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 5.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 1.500000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 2.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5] += fval;
    sum2.data[5] += fval;
    sum1.data[6]  = fmaf(fval, 4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 5.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 1.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6] += fval;
    sum2.data[6] += fval;
    sum1.data[7]  = fmaf(fval, 4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] += fval;

    uint mask = HIPSELECT(0xffffu, 0u, y < 3);
    mask = HIPSELECT(0u, mask, y < (dstHeight - 3));
    uint4 dst;
    uint mp;

    mp = hip_canny_mag_phase_L2_7x7(sum1.data[0], sum2.data[0]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 3);
    dst.x = mp;

    mp = hip_canny_mag_phase_L2_7x7(sum1.data[1], sum2.data[1]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 2);
    dst.x |= (mp << 16);

    mp = hip_canny_mag_phase_L2_7x7(sum1.data[2], sum2.data[2]) & mask;
    mp = HIPSELECT(mp, 0u, (int)x < 1);
    dst.y = mp;

    mp = hip_canny_mag_phase_L2_7x7(sum1.data[3], sum2.data[3]) & mask;
    dst.y |= (mp << 16);

    mp = hip_canny_mag_phase_L2_7x7(sum1.data[4], sum2.data[4]) & mask;
    dst.z = mp;

    mp = hip_canny_mag_phase_L2_7x7(sum1.data[5], sum2.data[5]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 8));
    dst.z |= (mp << 16);

    mp = hip_canny_mag_phase_L2_7x7(sum1.data[6], sum2.data[6]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 9));
    dst.w  =  mp;

    mp = hip_canny_mag_phase_L2_7x7(sum1.data[7], sum2.data[7]) & mask;
    mp = HIPSELECT(0u, mp, x < (dstWidth - 10));
    dst.w |= (mp << 16);

    uint dstIdx =  y * dstImageStrideInBytes + x + x;
    if (valid) {
        *((uint4 *)(&pDstImage[dstIdx])) = dst;
    }
}

int HipExec_CannySobel_U16_U8_7x7_L2NORM(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_CannySobel_U16_U8_7x7_L2NORM, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_CannySuppThreshold_U8XY_U16_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstImage, int dstImageStrideInBytes,
    const uchar *pSrcImage, int srcImageStrideInBytes,
    const uchar *xyStack, uint xyStackOffset, uint capacityOfXY, uint2 hyst,
    uint dstWidthComp) {

    __shared__ uchar lbuf[2448];
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    bool valid = (x < dstWidthComp) && (y < dstHeight);

    uint dstIdx =  y * dstImageStrideInBytes + (x << 2);

    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (y - 1) * srcImageStrideInBytes + (x << 3) - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (y - ly + id - 1) * srcImageStrideInBytes + ((x - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }

    uchar *lbuf_ptr = lbuf + ly * 136 + (lx << 3);
    uint4 L0 = *((uint4 *)(&lbuf_ptr[0]));
    uint4 L1 = *((uint4 *)(&lbuf_ptr[136]));
    uint4 L2 = *((uint4 *)(&lbuf_ptr[272]));
    uint3 NA, NB, NC;
    uint T, M1, M2;
    uint4 M;

    NA.x = L0.x >> 18;
    NA.y = L1.x >> 18;
    NA.z = L2.x >> 18;
    NB.x = hip_bfe(L0.y, 2, 14);
    NB.y = hip_bfe(L1.y, 2, 14);
    NB.z = hip_bfe(L2.y, 2, 14);
    NC.x = L0.y >> 18;
    NC.y = L1.y >> 18;
    NC.z = L2.y >> 18;
    T = hip_bfe(L1.y,  0, 2);
    M1 = HIPSELECT(NA.y, NA.x, T > 0);
    M1 = HIPSELECT(M1, NB.x, T > 1);
    M1 = HIPSELECT(M1, NA.z, T > 2);
    M2 = HIPSELECT(NC.y, NC.z+1, T > 0);
    M2 = HIPSELECT(M2, NB.z, T > 1);
    M2 = HIPSELECT(M2, NC.x+1, T > 2);
    M.x = HIPSELECT(0u, NB.y, NB.y > M1);
    M.x = HIPSELECT(0u, M.x, NB.y >= M2);

    NA.x = hip_bfe(L0.z, 2, 14);
    NA.y = hip_bfe(L1.z, 2, 14);
    NA.z = hip_bfe(L2.z, 2, 14);
    T = hip_bfe(L1.y, 16, 2);
    M1 = HIPSELECT(NB.y, NB.x, T > 0);
    M1 = HIPSELECT(M1, NC.x, T > 1);
    M1 = HIPSELECT(M1, NB.z, T > 2);
    M2 = HIPSELECT(NA.y, NA.z+1, T > 0);
    M2 = HIPSELECT(M2, NC.z, T > 1);
    M2 = HIPSELECT(M2, NA.x+1, T > 2);
    M.y = HIPSELECT(0u, NC.y, NC.y > M1);
    M.y = HIPSELECT(0u, M.y, NC.y >= M2);
    NB.x = L0.z >> 18;
    NB.y = L1.z >> 18;
    NB.z = L2.z >> 18;
    T = hip_bfe(L1.z, 0, 2);
    M1 = HIPSELECT(NC.y, NC.x, T > 0);
    M1 = HIPSELECT(M1, NA.x, T > 1);
    M1 = HIPSELECT(M1, NC.z, T > 2);
    M2 = HIPSELECT(NB.y, NB.z+1, T > 0);
    M2 = HIPSELECT(M2, NA.z, T > 1);
    M2 = HIPSELECT(M2, NB.x+1, T > 2);
    M.z = HIPSELECT(0u, NA.y, NA.y > M1);
    M.z = HIPSELECT(0u, M.z, NA.y >= M2);
    NC.x = hip_bfe(L0.w, 2, 14);
    NC.y = hip_bfe(L1.w, 2, 14);
    NC.z = hip_bfe(L2.w, 2, 14);
    T = hip_bfe(L1.z, 16, 2);
    M1 = HIPSELECT(NA.y, NA.x, T > 0);
    M1 = HIPSELECT(M1, NB.x, T > 1);
    M1 = HIPSELECT(M1, NA.z, T > 2);
    M2 = HIPSELECT(NC.y, NC.z+1, T > 0);
    M2 = HIPSELECT(M2, NB.z, T > 1);
    M2 = HIPSELECT(M2, NC.x+1, T > 2);
    M.w = HIPSELECT(0u, NB.y, NB.y > M1);
    M.w = HIPSELECT(0u, M.w, NB.y >= M2);

    uint mask = HIPSELECT(0u, 0xffffffffu, x < dstWidthComp);
    mask = HIPSELECT(0u, mask, y < dstHeight);
    M.x &= mask;
    M.y &= mask;
    M.z &= mask;
    M.w &= mask;
    uint4 P;
    P.x = HIPSELECT(0u, 127u, M.x > hyst.x);
    P.y = HIPSELECT(0u, 127u, M.y > hyst.x);
    P.z = HIPSELECT(0u, 127u, M.z > hyst.x);
    P.w = HIPSELECT(0u, 127u, M.w > hyst.x);
    P.x = HIPSELECT(P.x, 255u, M.x > hyst.y);
    P.y = HIPSELECT(P.y, 255u, M.y > hyst.y);
    P.z = HIPSELECT(P.z, 255u, M.z > hyst.y);
    P.w = HIPSELECT(P.w, 255u, M.w > hyst.y);
    uint p0 = P.x;
    p0 += P.y << 8;
    p0 += P.z << 16;
    p0 += P.w << 24;

    if (valid) {
        *((uint *)(&pDstImage[dstIdx])) = p0;
        uint stack_icount;
        stack_icount  = HIPSELECT(0u, 1u, P.x == 255u);
        stack_icount += HIPSELECT(0u, 1u, P.y == 255u);
        stack_icount += HIPSELECT(0u, 1u, P.z == 255u);
        stack_icount += HIPSELECT(0u, 1u, P.w == 255u);
        if (stack_icount > 0) {
            uint pos = atomicAdd((uint *)xyStack, stack_icount);
            uint *xyStackPtr = (uint *)&xyStack[xyStackOffset];
            uint xyloc = (y << 16) + (x << 2);
            if(pos < capacityOfXY && P.x == 255u)
                xyStackPtr[pos++] = xyloc;
            if(pos < capacityOfXY && P.y == 255u)
                xyStackPtr[pos++] = xyloc + 1;
            if(pos < capacityOfXY && P.z == 255u)
                xyStackPtr[pos++] = xyloc + 2;
            if(pos < capacityOfXY && P.w == 255u)
                xyStackPtr[pos++] = xyloc + 3;
        }
    }
}
int HipExec_CannySuppThreshold_U8XY_U16_3x3(hipStream_t stream,
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint8* xyStack, vx_uint32 xyStackOffset, vx_uint32 capacityOfXY,
    vx_uint16 hyst_lower, vx_uint16 hyst_upper) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2;
    int globalThreads_y = dstHeight;

    uint2 hyst;
    hyst.x = (uint) hyst_lower;
    hyst.y = (uint) hyst_upper;

    uint dstWidthComp = (dstWidth + 3) / 4;

    hipLaunchKernelGGL(Hip_CannySuppThreshold_U8XY_U16_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (const uchar *)xyStack, xyStackOffset, capacityOfXY, hyst,
                        dstWidthComp);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxFastCorners kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_FastCorners_XY_U8_NoSupression(uint capacityOfDstCorner, char *pDstCorners, uint cornerBufferOffset,
    uint srcWidth, uint srcHeight, const uchar *pSrcImage, uint srcImageStrideInBytes, float strength_threshold) {

    int idx = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) + 3;
    int idy = (hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y) + 3;
    int stride = (int) srcImageStrideInBytes;
    if((idx > (int)srcWidth - 3) || (idy > (int)srcHeight - 3))
        return;
    const uchar *pTempImg = pSrcImage + hip_mad24(idy, stride, idx);
    int centerPixel_neg = pTempImg[0];
    int centerPixel_pos = centerPixel_neg + (int)strength_threshold;
    centerPixel_neg -= (int)strength_threshold;
    int candp, candn, pos_mask, neg_mask;
    candp = pTempImg[3];
    candn = pTempImg[-3];
    neg_mask = (candp < centerPixel_neg) | ((candn < centerPixel_neg) << 8);
    pos_mask = (candp > centerPixel_pos) | ((candn > centerPixel_pos) << 8);
    int offs = -stride * 3;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 4) | ((candn < centerPixel_neg) << 12));
    pos_mask |= (((candp > centerPixel_pos) << 4) | ((candn > centerPixel_pos) << 12));
    if(((pos_mask | neg_mask) & MASK_EARLY_EXIT) == 0)
        return;

    offs = -stride * 3 + 1;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 3) | ((candn < centerPixel_neg) << 11));
    pos_mask |= (((candp > centerPixel_pos) << 3) | ((candn > centerPixel_pos) << 11));

    offs = -stride * 3 - 1;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 5) | ((candn < centerPixel_neg) << 13));
    pos_mask |= (((candp > centerPixel_pos) << 5) | ((candn > centerPixel_pos) << 13));

    offs = -(stride << 1) + 2;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 2) | ((candn < centerPixel_neg) << 10));
    pos_mask |= (((candp > centerPixel_pos) << 2) | ((candn > centerPixel_pos) << 10));

    offs = -(stride << 1) - 2;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 6) | ((candn < centerPixel_neg) << 14));
    pos_mask |= (((candp > centerPixel_pos) << 6) | ((candn > centerPixel_pos) << 14));

    offs = -stride + 3;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 1) | ((candn < centerPixel_neg) << 9));
    pos_mask |= (((candp > centerPixel_pos) << 1) | ((candn > centerPixel_pos) << 9));

    offs = -stride - 3;
    candp = pTempImg[offs];
    candn = pTempImg[-offs];
    neg_mask |= (((candp < centerPixel_neg) << 7) | ((candn < centerPixel_neg) << 15));
    pos_mask |= (((candp > centerPixel_pos) << 7) | ((candn > centerPixel_pos) << 15));

    pos_mask |= (pos_mask << 16);
    neg_mask |= (neg_mask << 16);

    int cornerMask = 511, isCorner = 0;

    for(int i = 0; i < 16; i++) {
        isCorner += ((pos_mask & cornerMask) == cornerMask);
        isCorner += ((neg_mask & cornerMask) == cornerMask);
        pos_mask >>= 1;
        neg_mask >>= 1;
    }

    int *numKeypoints = (int *) pDstCorners;
    d_KeyPt *keypt_list = (d_KeyPt *) (pDstCorners + cornerBufferOffset);
    if (isCorner) {
        int old_idx = atomicAdd(numKeypoints, 1);
        if (old_idx < capacityOfDstCorner) {
            keypt_list[old_idx].x = idx;
            keypt_list[old_idx].y = idy;
            keypt_list[old_idx].strength = strength_threshold;
            keypt_list[old_idx].scale = 0;
            keypt_list[old_idx].orientation = 0;
            keypt_list[old_idx].tracking_status = 1;
            keypt_list[old_idx].error = 0;
        }
    }
}
int HipExec_FastCorners_XY_U8_NoSupression(hipStream_t stream, vx_uint32 capacityOfDstCorner, vx_uint8 *pDstCorner, vx_uint32 cornerBufferOffset,
    vx_uint32 srcWidth, vx_uint32 srcHeight, const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_float32 strength_threshold) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = ceil((srcWidth - 4) / 14) * 16;
    int globalThreads_y = ceil((srcHeight - 4) / 14) * 16;

    hipLaunchKernelGGL(Hip_FastCorners_XY_U8_NoSupression, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, capacityOfDstCorner, (char *) pDstCorner, cornerBufferOffset,
                        srcWidth, srcHeight, (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        strength_threshold);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FastCorners_XY_U8_Supression(uint capacityOfDstCorner, char *pDstCorners, uint cornerBufferOffset,
    uint srcWidth, uint srcHeight, const uchar *pSrcImage, int srcImageStrideInBytes, float strength_threshold) {

    int lidx = hipThreadIdx_x;
    int lidy = hipThreadIdx_y;
    int gidx = hipBlockIdx_x;
    int gidy = hipBlockIdx_y;
    int xoffset = gidx * 14 + lidx + 2;
    int yoffset = gidy * 14 + lidy + 2;
    const uchar *pTempImg = pSrcImage + hip_mad24(yoffset, srcImageStrideInBytes, xoffset);
    __shared__ int pLocalStrengthShare[16][16];
    bool doCompute = true;
    if((xoffset > (int)srcWidth - 3) || (yoffset > (int)srcHeight - 3) || (xoffset < 3) || (yoffset < 3)) {
        doCompute = false;
        pLocalStrengthShare[lidy][lidx] = 0;
    }

    int local_strength = 0;
    if (doCompute) {
        int boundary[16];
        int pos_mask, neg_mask, offs;
        int centerPixel_neg = pTempImg[0];
        for (int i = 0; i < 16; i++)
            boundary[i] = centerPixel_neg;
        int centerPixel_pos = centerPixel_neg + (int)strength_threshold;
        centerPixel_neg -= (int) strength_threshold;
        int candp = pTempImg[3];
        int candn = pTempImg[-3];
        neg_mask = (candp < centerPixel_neg) | ((candn < centerPixel_neg) << 8);
        pos_mask = (candp > centerPixel_pos) | ((candn > centerPixel_pos) << 8);
        boundary[0] -= candp;
        boundary[8] -= candn;
        offs = -srcImageStrideInBytes * 3;
        candp = pTempImg[offs];
        candn = pTempImg[-offs];
        neg_mask |= (((candp < centerPixel_neg) << 4) | ((candn < centerPixel_neg) << 12));
        pos_mask |= (((candp > centerPixel_pos) << 4) | ((candn > centerPixel_pos) << 12));
        boundary[4] -= candp;
        boundary[12] -= candn;
        if(((pos_mask | neg_mask) & MASK_EARLY_EXIT) == 0) {
            pLocalStrengthShare[lidy][lidx] = 0;
            doCompute = false;
        } else {
            offs = -srcImageStrideInBytes * 3 + 1;
            candp = pTempImg[offs];
            candn = pTempImg[-offs];
            neg_mask |= (((candp < centerPixel_neg) << 3) | ((candn < centerPixel_neg) << 11));
            pos_mask |= (((candp > centerPixel_pos) << 3) | ((candn > centerPixel_pos) << 11));
            boundary[3] -= candp;
            boundary[11] -= candn;

            offs = -srcImageStrideInBytes * 3 - 1;
            candp = pTempImg[offs];
            candn = pTempImg[-offs];
            neg_mask |= (((candp < centerPixel_neg) << 5) | ((candn < centerPixel_neg) << 13));
            pos_mask |= (((candp > centerPixel_pos) << 5) | ((candn > centerPixel_pos) << 13));
            boundary[5] -= candp;
            boundary[13] -= candn;

            offs = -(srcImageStrideInBytes << 1) + 2;
            candp = pTempImg[offs];
            candn = pTempImg[-offs];
            neg_mask |= (((candp < centerPixel_neg) << 2) | ((candn < centerPixel_neg) << 10));
            pos_mask |= (((candp > centerPixel_pos) << 2) | ((candn > centerPixel_pos) << 10));
            boundary[2] -= candp;
            boundary[10] -= candn;

            offs = -(srcImageStrideInBytes << 1) - 2;
            candp = pTempImg[offs];
            candn = pTempImg[-offs];
            neg_mask |= (((candp < centerPixel_neg) << 6) | ((candn < centerPixel_neg) << 14));
            pos_mask |= (((candp > centerPixel_pos) << 6) | ((candn > centerPixel_pos) << 14));
            boundary[6] -= candp;
            boundary[14] -= candn;

            offs = -srcImageStrideInBytes + 3;
            candp = pTempImg[offs];
            candn = pTempImg[-offs];
            neg_mask |= (((candp < centerPixel_neg) << 1) | ((candn < centerPixel_neg) << 9));
            pos_mask |= (((candp > centerPixel_pos) << 1) | ((candn > centerPixel_pos) << 9));
            boundary[1] -= candp;
            boundary[9] -= candn;

            offs = -srcImageStrideInBytes - 3;
            candp = pTempImg[offs];
            candn = pTempImg[-offs];
            neg_mask |= (((candp < centerPixel_neg) << 7) | ((candn < centerPixel_neg) << 15));
            pos_mask |= (((candp > centerPixel_pos) << 7) | ((candn > centerPixel_pos) << 15));
            boundary[7] -= candp;
            boundary[15] -= candn;

            pos_mask |= (pos_mask << 16);
            neg_mask |= (neg_mask << 16);

            int cornerMask = 511;
            int isCorner = 0;

            for (int i = 0; i < 16; i++) {
                isCorner += ((pos_mask & cornerMask) == cornerMask);
                isCorner += ((neg_mask & cornerMask) == cornerMask);
                pos_mask >>= 1;
                neg_mask >>= 1;
            }

            if(isCorner == 0) {
                pLocalStrengthShare[lidy][lidx] = 0;
                doCompute = false;
            } else {
                int strength;
                int tmp = 0;
                for (int i = 0; i < 16; i += 2)	{
                    int s = min(boundary[(i + 1) & 15], boundary[(i + 2) & 15]);
                    s = min(s, boundary[(i + 3) & 15]);
                    s = min(s, boundary[(i + 4) & 15]);
                    s = min(s, boundary[(i + 5) & 15]);
                    s = min(s, boundary[(i + 6) & 15]);
                    s = min(s, boundary[(i + 7) & 15]);
                    s = min(s, boundary[(i + 8) & 15]);
                    tmp = max(tmp, min(s, boundary[i & 15]));
                    tmp = max(tmp, min(s, boundary[(i + 9) & 15]));
                }
                strength = -tmp;
                for (int i = 0; i < 16; i += 2)	{
                    int s = max(boundary[(i + 1) & 15], boundary[(i + 2) & 15]);
                    s = max(s, boundary[(i + 3) & 15]);
                    s = max(s, boundary[(i + 4) & 15]);
                    s = max(s, boundary[(i + 5) & 15]);
                    s = max(s, boundary[(i + 6) & 15]);
                    s = max(s, boundary[(i + 7) & 15]);
                    s = max(s, boundary[(i + 8) & 15]);
                    strength = min(strength, max(s, boundary[i & 15]));
                    strength = min(strength, max(s, boundary[(i + 9) & 15]));
                }

                local_strength = -strength - 1;
                pLocalStrengthShare[lidy][lidx] = local_strength;
            }
        }
    }
    __syncthreads();

    bool writeCorner = doCompute &&
                       (lidx > 0) &&
                       (lidy > 0) &&
                       (lidx < 15) &&
                       (lidy < 15) &&
                       (local_strength >= pLocalStrengthShare[lidy - 1][lidx - 1]) &&
                       (local_strength >= pLocalStrengthShare[lidy - 1][lidx]) &&
                       (local_strength >= pLocalStrengthShare[lidy - 1][lidx + 1]) &&
                       (local_strength >= pLocalStrengthShare[lidy][lidx - 1]) &&
                       (local_strength >  pLocalStrengthShare[lidy][lidx + 1]) &&
                       (local_strength >  pLocalStrengthShare[lidy + 1][lidx - 1]) &&
                       (local_strength >  pLocalStrengthShare[lidy + 1][lidx]) &&
                       (local_strength >= pLocalStrengthShare[lidy + 1][lidx + 1]);

    int *numKeypoints = (int *) pDstCorners;
    d_KeyPt *keypt_list = (d_KeyPt *) (pDstCorners + cornerBufferOffset);
    if (writeCorner) {
        int old_idx = atomicAdd(numKeypoints, 1);
        if (old_idx < capacityOfDstCorner) {
            keypt_list[old_idx].x = xoffset;
            keypt_list[old_idx].y = yoffset;
            keypt_list[old_idx].strength = (float) local_strength;
            keypt_list[old_idx].scale = 0;
            keypt_list[old_idx].orientation = 0;
            keypt_list[old_idx].tracking_status = 1;
            keypt_list[old_idx].error = 0;
        }
    }
}
int HipExec_FastCorners_XY_U8_Supression(hipStream_t stream, vx_uint32 capacityOfDstCorner, vx_uint8* pDstCorner, vx_uint32 cornerBufferOffset,
    vx_uint32 srcWidth, vx_uint32 srcHeight, const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, vx_float32 strength_threshold) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = ceil((srcWidth - 4) / 14) * 16;
    int globalThreads_y = ceil((srcHeight - 4) / 14) * 16;

    hipLaunchKernelGGL(Hip_FastCorners_XY_U8_Supression, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, capacityOfDstCorner, (char *) pDstCorner, cornerBufferOffset,
                        srcWidth, srcHeight, (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        strength_threshold);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxHarrisCorners kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_HarrisSobel_HG3_U8_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstGxy, uint dstGxyStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp1, uint dstWidthComp2) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) << 3;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

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
    uint2 * lbufptr;
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

    if ((x < dstWidth) && (y < dstHeight)) {
        uint dstIdx =  y * dstGxyStrideInBytes + (x << 2);

        float4 sum1X, sum1Y, sum2X, sum2Y;
        sum1X = make_float4(sum1.data[0], sum1.data[1], sum1.data[2], sum1.data[3]);
        sum1Y = make_float4(sum1.data[4], sum1.data[5], sum1.data[6], sum1.data[7]);
        sum2X = make_float4(sum2.data[0], sum2.data[1], sum2.data[2], sum2.data[3]);
        sum2Y = make_float4(sum2.data[4], sum2.data[5], sum2.data[6], sum2.data[7]);

        d_float8 dst = {0};

        *(float4 *)(&dst.data[0]) = sum1X * sum1X;
        *(float4 *)(&dst.data[4]) = sum1Y * sum1Y;
        *((d_float8 *)(&pDstGxy[dstIdx])) = dst;

        *(float4 *)(&dst.data[0]) = sum1X * sum2X;
        *(float4 *)(&dst.data[4]) = sum1Y * sum2Y;
        *((d_float8 *)(&pDstGxy[dstIdx + dstWidthComp1])) = dst;

        *(float4 *)(&dst.data[0]) = sum2X * sum2X;
        *(float4 *)(&dst.data[4]) = sum2Y * sum2Y;
        *((d_float8 *)(&pDstGxy[dstIdx + dstWidthComp2])) = dst;
    }

}
int HipExec_HarrisSobel_HG3_U8_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 *pHipDstGxy, vx_uint32 dstGxyStrideInBytes,
    vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_uint32 dstWidthComp1 = dstWidth * 4;
    vx_uint32 dstWidthComp2 = dstWidth * 8;

    hipLaunchKernelGGL(Hip_HarrisSobel_HG3_U8_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstGxy, dstGxyStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp1, dstWidthComp2);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_HarrisSobel_HG3_U8_5x5(uint dstWidth, uint dstHeight,
    uchar *pDstGxy, uint dstGxyStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp1, uint dstWidthComp2) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) << 3;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    __shared__ uchar lbuf[2720]; // 136x18 bytes
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
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum1 = {0.0f};
    d_float8 sum2 = {0.0f};
    uint2 pix;
    float fval;
    uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);

    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack2(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] -= fval;
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -2.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -4.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] -= fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -2.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -4.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] -= fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 2.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -4.000000000000e+00f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -2.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -4.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] -= fval;
    fval = hip_unpack2(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] -= fval;
    sum1.data[1]  = fmaf(fval, 2.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -4.000000000000e+00f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -2.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -4.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] -= fval;
    fval = hip_unpack3(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] -= fval;
    sum1.data[2]  = fmaf(fval, 2.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -4.000000000000e+00f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -2.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -4.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] -= fval;
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum1.data[2] += fval;
    sum2.data[2] -= fval;
    sum1.data[3]  = fmaf(fval, 2.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -4.000000000000e+00f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -2.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -4.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] -= fval;
    fval = hip_unpack1(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] -= fval;
    sum1.data[4]  = fmaf(fval, 2.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -4.000000000000e+00f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -2.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -4.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] -= fval;
    fval = hip_unpack2(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] -= fval;
    sum1.data[5]  = fmaf(fval, 2.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -4.000000000000e+00f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -2.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] -= fval;
    sum1.data[6]  = fmaf(fval, 2.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -4.000000000000e+00f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[6] += fval;
    sum2.data[6] -= fval;
    sum1.data[7]  = fmaf(fval, 2.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] -= fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -2.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -8.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -8.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -2.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -1.200000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -8.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -8.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -2.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 8.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -8.000000000000e+00f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -1.200000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -8.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -8.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -2.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -2.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 8.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -8.000000000000e+00f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -1.200000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -8.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -8.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -2.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack3(pix.y);
    sum1.data[1]  = fmaf(fval, 4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -2.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 8.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -8.000000000000e+00f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -1.200000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -8.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -8.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -2.000000000000e+00f, sum2.data[5]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum1.data[2]  = fmaf(fval, 4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -2.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 8.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -8.000000000000e+00f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -1.200000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -8.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -8.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -2.000000000000e+00f, sum2.data[6]);
    fval = hip_unpack1(pix.x);
    sum1.data[3]  = fmaf(fval, 4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -2.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 8.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -8.000000000000e+00f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -1.200000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -8.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -8.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[4]  = fmaf(fval, 4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -2.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 8.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -8.000000000000e+00f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -1.200000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -8.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -8.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5]  = fmaf(fval, 4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -2.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 8.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -8.000000000000e+00f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -1.200000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[6]  = fmaf(fval, 4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -2.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 8.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -8.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7]  = fmaf(fval, 4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -2.000000000000e+00f, sum2.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+00f, sum1.data[0]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -1.200000000000e+01f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+00f, sum1.data[1]);
    fval = hip_unpack0(pix.y);
    sum1.data[1]  = fmaf(fval, -1.200000000000e+01f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+00f, sum1.data[2]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 1.200000000000e+01f, sum1.data[0]);
    sum1.data[2]  = fmaf(fval, -1.200000000000e+01f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+00f, sum1.data[3]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+00f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, 1.200000000000e+01f, sum1.data[1]);
    sum1.data[3]  = fmaf(fval, -1.200000000000e+01f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+00f, sum1.data[4]);
    fval = hip_unpack3(pix.y);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+00f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, 1.200000000000e+01f, sum1.data[2]);
    sum1.data[4]  = fmaf(fval, -1.200000000000e+01f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+00f, sum1.data[5]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+00f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, 1.200000000000e+01f, sum1.data[3]);
    sum1.data[5]  = fmaf(fval, -1.200000000000e+01f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+00f, sum1.data[6]);
    fval = hip_unpack1(pix.x);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+00f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, 1.200000000000e+01f, sum1.data[4]);
    sum1.data[6]  = fmaf(fval, -1.200000000000e+01f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+00f, sum1.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+00f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, 1.200000000000e+01f, sum1.data[5]);
    sum1.data[7]  = fmaf(fval, -1.200000000000e+01f, sum1.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+00f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, 1.200000000000e+01f, sum1.data[6]);
    fval = hip_unpack0(pix.y);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+00f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, 1.200000000000e+01f, sum1.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+00f, sum1.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 2.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -8.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 8.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 2.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 1.200000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -8.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 8.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 2.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 8.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 8.000000000000e+00f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 1.200000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -8.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 8.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 2.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 2.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 8.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 8.000000000000e+00f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 1.200000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -8.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 8.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 2.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack3(pix.y);
    sum1.data[1]  = fmaf(fval, 4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 2.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 8.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 8.000000000000e+00f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 1.200000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -8.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 8.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 2.000000000000e+00f, sum2.data[5]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum1.data[2]  = fmaf(fval, 4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 2.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 8.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 8.000000000000e+00f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 1.200000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -8.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 8.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 2.000000000000e+00f, sum2.data[6]);
    fval = hip_unpack1(pix.x);
    sum1.data[3]  = fmaf(fval, 4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 2.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 8.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 8.000000000000e+00f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 1.200000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -8.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 8.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 2.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[4]  = fmaf(fval, 4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 2.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 8.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 8.000000000000e+00f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 1.200000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -8.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 8.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5]  = fmaf(fval, 4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 2.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 8.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 8.000000000000e+00f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 1.200000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[6]  = fmaf(fval, 4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 2.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 8.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 8.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7]  = fmaf(fval, 4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 2.000000000000e+00f, sum2.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack2(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] += fval;
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -2.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 4.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] += fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -2.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 4.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] += fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 2.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 4.000000000000e+00f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -2.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 4.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] += fval;
    fval = hip_unpack2(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] += fval;
    sum1.data[1]  = fmaf(fval, 2.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 4.000000000000e+00f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -2.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 4.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] += fval;
    fval = hip_unpack3(pix.y);
    sum1.data[1] += fval;
    sum2.data[1] += fval;
    sum1.data[2]  = fmaf(fval, 2.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 4.000000000000e+00f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -2.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 4.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] += fval;
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum1.data[2] += fval;
    sum2.data[2] += fval;
    sum1.data[3]  = fmaf(fval, 2.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 4.000000000000e+00f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -2.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 4.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] += fval;
    fval = hip_unpack1(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] += fval;
    sum1.data[4]  = fmaf(fval, 2.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 4.000000000000e+00f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -2.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 4.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] += fval;
    fval = hip_unpack2(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] += fval;
    sum1.data[5]  = fmaf(fval, 2.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 4.000000000000e+00f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -2.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[5] += fval;
    sum2.data[5] += fval;
    sum1.data[6]  = fmaf(fval, 2.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 4.000000000000e+00f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[6] += fval;
    sum2.data[6] += fval;
    sum1.data[7]  = fmaf(fval, 2.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] += fval;
    if ((x < dstWidth) && (y < dstHeight)) {
        uint dstIdx =  y * dstGxyStrideInBytes + (x << 2);

        float4 sum1X, sum1Y, sum2X, sum2Y;
        sum1X = make_float4(sum1.data[0], sum1.data[1], sum1.data[2], sum1.data[3]);
        sum1Y = make_float4(sum1.data[4], sum1.data[5], sum1.data[6], sum1.data[7]);
        sum2X = make_float4(sum2.data[0], sum2.data[1], sum2.data[2], sum2.data[3]);
        sum2Y = make_float4(sum2.data[4], sum2.data[5], sum2.data[6], sum2.data[7]);

        d_float8 dst = {0};

        *(float4 *)(&dst.data[0]) = sum1X * sum1X;
        *(float4 *)(&dst.data[4]) = sum1Y * sum1Y;
        *((d_float8 *)(&pDstGxy[dstIdx])) = dst;

        *(float4 *)(&dst.data[0]) = sum1X * sum2X;
        *(float4 *)(&dst.data[4]) = sum1Y * sum2Y;
        *((d_float8 *)(&pDstGxy[dstIdx + dstWidthComp1])) = dst;

        *(float4 *)(&dst.data[0]) = sum2X * sum2X;
        *(float4 *)(&dst.data[4]) = sum2Y * sum2Y;
        *((d_float8 *)(&pDstGxy[dstIdx + dstWidthComp2])) = dst;
    }
}

int HipExec_HarrisSobel_HG3_U8_5x5(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 *pHipDstGxy, vx_uint32 dstGxyStrideInBytes,
    vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_uint32 dstWidthComp1 = dstWidth * 4;
    vx_uint32 dstWidthComp2 = dstWidth * 8;

    hipLaunchKernelGGL(Hip_HarrisSobel_HG3_U8_5x5, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstGxy, dstGxyStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp1, dstWidthComp2);


    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_HarrisSobel_HG3_U8_7x7(uint dstWidth, uint dstHeight,
    uchar *pDstGxy, uint dstGxyStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp1, uint dstWidthComp2) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) << 3;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    __shared__ uchar lbuf[2992]; // 136x18 bytes
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
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    d_float8 sum1 = {0.0f};
    d_float8 sum2 = {0.0f};
    uint2 pix;
    float fval;
    uint2 * lbufptr;
    lbufptr = (uint2 *) (&lbuf[ly * 136 + (lx << 3)]);

    // filterRow = 0
    pix = lbufptr[0];
    fval = hip_unpack1(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] -= fval;
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] -= fval;
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -5.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -1.500000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] -= fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -2.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -5.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -1.500000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] -= fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 5.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -1.500000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -2.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -5.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -1.500000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] -= fval;
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 5.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -1.500000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -2.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -5.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -1.500000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] -= fval;
    fval = hip_unpack3(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] -= fval;
    sum1.data[1]  = fmaf(fval, 4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 5.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -1.500000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -2.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -5.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -1.500000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] -= fval;
    pix = lbufptr[1];
    fval = hip_unpack0(pix.x);
    sum1.data[1] += fval;
    sum2.data[1] -= fval;
    sum1.data[2]  = fmaf(fval, 4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 5.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -1.500000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -2.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -5.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -1.500000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] -= fval;
    fval = hip_unpack1(pix.x);
    sum1.data[2] += fval;
    sum2.data[2] -= fval;
    sum1.data[3]  = fmaf(fval, 4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 5.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -1.500000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -2.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -5.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -1.500000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] -= fval;
    sum1.data[4]  = fmaf(fval, 4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 5.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -1.500000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -2.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -5.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -1.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] -= fval;
    sum1.data[5]  = fmaf(fval, 4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 5.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -1.500000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -2.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5] += fval;
    sum2.data[5] -= fval;
    sum1.data[6]  = fmaf(fval, 4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 5.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -1.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6] += fval;
    sum2.data[6] -= fval;
    sum1.data[7]  = fmaf(fval, 4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] -= fval;
    // filterRow = 1
    pix = lbufptr[17];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -4.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -2.400000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -2.400000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -4.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -3.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -2.400000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -2.400000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -4.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -8.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -3.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -2.400000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -2.400000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -4.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 3.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -6.000000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -8.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -3.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -2.400000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -2.400000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -4.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 2.400000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -2.400000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 3.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -6.000000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -8.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -3.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -2.400000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -2.400000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -4.000000000000e+00f, sum2.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -4.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 2.400000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -2.400000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 3.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -6.000000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -8.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -3.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -2.400000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -2.400000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -4.000000000000e+00f, sum2.data[6]);
    pix = lbufptr[18];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -4.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 2.400000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -2.400000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 3.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -6.000000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -8.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -3.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -2.400000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -2.400000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -4.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 2.400000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -2.400000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 3.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -6.000000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -8.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -3.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -2.400000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -2.400000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -4.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 2.400000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -2.400000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 3.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -6.000000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -8.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -3.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -4.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 2.400000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -2.400000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 3.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -6.000000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -8.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -4.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 2.400000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -2.400000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 3.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -6.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -4.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 2.400000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -2.400000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -4.000000000000e+00f, sum2.data[7]);
    // filterRow = 2
    pix = lbufptr[34];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -1.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -5.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -3.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -1.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -5.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -7.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -7.500000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -3.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -1.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -5.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, -1.000000000000e+02f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -7.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -7.500000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -3.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -1.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -5.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 7.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -7.500000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, -1.000000000000e+02f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -7.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -7.500000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -3.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -1.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -5.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -3.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 7.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -7.500000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, -1.000000000000e+02f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -7.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -7.500000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -3.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -1.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -5.000000000000e+00f, sum2.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 1.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, -5.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -3.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 7.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -7.500000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, -1.000000000000e+02f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -7.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -7.500000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -3.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -1.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -5.000000000000e+00f, sum2.data[6]);
    pix = lbufptr[35];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 1.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, -5.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -3.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 7.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -7.500000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, -1.000000000000e+02f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -7.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -7.500000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -3.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -1.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -5.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 1.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, -5.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -3.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 7.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -7.500000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, -1.000000000000e+02f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -7.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -7.500000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -3.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 1.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, -5.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -3.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 7.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -7.500000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, -1.000000000000e+02f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -7.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -7.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 1.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, -5.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -3.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 7.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -7.500000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, -1.000000000000e+02f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 1.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, -5.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -3.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 7.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -7.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 1.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, -5.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -3.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 1.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, -5.000000000000e+00f, sum2.data[7]);
    // filterRow = 3
    pix = lbufptr[51];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -2.000000000000e+01f, sum1.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -8.000000000000e+01f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, -2.000000000000e+01f, sum1.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -1.000000000000e+02f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, -8.000000000000e+01f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, -2.000000000000e+01f, sum1.data[2]);
    fval = hip_unpack0(pix.y);
    sum1.data[1]  = fmaf(fval, -1.000000000000e+02f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, -8.000000000000e+01f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, -2.000000000000e+01f, sum1.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 1.000000000000e+02f, sum1.data[0]);
    sum1.data[2]  = fmaf(fval, -1.000000000000e+02f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, -8.000000000000e+01f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, -2.000000000000e+01f, sum1.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 8.000000000000e+01f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, 1.000000000000e+02f, sum1.data[1]);
    sum1.data[3]  = fmaf(fval, -1.000000000000e+02f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, -8.000000000000e+01f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, -2.000000000000e+01f, sum1.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 2.000000000000e+01f, sum1.data[0]);
    sum1.data[1]  = fmaf(fval, 8.000000000000e+01f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, 1.000000000000e+02f, sum1.data[2]);
    sum1.data[4]  = fmaf(fval, -1.000000000000e+02f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, -8.000000000000e+01f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, -2.000000000000e+01f, sum1.data[6]);
    pix = lbufptr[52];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 2.000000000000e+01f, sum1.data[1]);
    sum1.data[2]  = fmaf(fval, 8.000000000000e+01f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, 1.000000000000e+02f, sum1.data[3]);
    sum1.data[5]  = fmaf(fval, -1.000000000000e+02f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, -8.000000000000e+01f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, -2.000000000000e+01f, sum1.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 2.000000000000e+01f, sum1.data[2]);
    sum1.data[3]  = fmaf(fval, 8.000000000000e+01f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, 1.000000000000e+02f, sum1.data[4]);
    sum1.data[6]  = fmaf(fval, -1.000000000000e+02f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, -8.000000000000e+01f, sum1.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 2.000000000000e+01f, sum1.data[3]);
    sum1.data[4]  = fmaf(fval, 8.000000000000e+01f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, 1.000000000000e+02f, sum1.data[5]);
    sum1.data[7]  = fmaf(fval, -1.000000000000e+02f, sum1.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 2.000000000000e+01f, sum1.data[4]);
    sum1.data[5]  = fmaf(fval, 8.000000000000e+01f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, 1.000000000000e+02f, sum1.data[6]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 2.000000000000e+01f, sum1.data[5]);
    sum1.data[6]  = fmaf(fval, 8.000000000000e+01f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, 1.000000000000e+02f, sum1.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 2.000000000000e+01f, sum1.data[6]);
    sum1.data[7]  = fmaf(fval, 8.000000000000e+01f, sum1.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 2.000000000000e+01f, sum1.data[7]);
    // filterRow = 4
    pix = lbufptr[68];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -1.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 5.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 3.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -1.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 5.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -7.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 7.500000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 3.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -1.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 5.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 1.000000000000e+02f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -7.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 7.500000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 3.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -1.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 5.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 7.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 7.500000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 1.000000000000e+02f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -7.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 7.500000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 3.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -1.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 5.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 3.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 7.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 7.500000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 1.000000000000e+02f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -7.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 7.500000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 3.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -1.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 5.000000000000e+00f, sum2.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 1.500000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 5.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 3.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 7.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 7.500000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 1.000000000000e+02f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -7.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 7.500000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 3.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -1.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 5.000000000000e+00f, sum2.data[6]);
    pix = lbufptr[69];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 1.500000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 5.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 3.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 7.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 7.500000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 1.000000000000e+02f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -7.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 7.500000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 3.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -1.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 5.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 1.500000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 5.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 3.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 7.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 7.500000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 1.000000000000e+02f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -7.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 7.500000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 3.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 1.500000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 5.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 3.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 7.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 7.500000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 1.000000000000e+02f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -7.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 7.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 1.500000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 5.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 3.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 7.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 7.500000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 1.000000000000e+02f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 1.500000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 5.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 3.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 7.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 7.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 1.500000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 5.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 3.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 1.500000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 5.000000000000e+00f, sum2.data[7]);
    // filterRow = 5
    pix = lbufptr[85];
    fval = hip_unpack1(pix.x);
    sum1.data[0]  = fmaf(fval, -6.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 4.000000000000e+00f, sum2.data[0]);
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -2.400000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 2.400000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -6.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 4.000000000000e+00f, sum2.data[1]);
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -3.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -2.400000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 2.400000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -6.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 4.000000000000e+00f, sum2.data[2]);
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 8.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -3.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -2.400000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 2.400000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -6.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 4.000000000000e+00f, sum2.data[3]);
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 3.000000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 8.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -3.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -2.400000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 2.400000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -6.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 4.000000000000e+00f, sum2.data[4]);
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 2.400000000000e+01f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 2.400000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 3.000000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 8.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -3.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -2.400000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 2.400000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -6.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 4.000000000000e+00f, sum2.data[5]);
    fval = hip_unpack3(pix.y);
    sum1.data[0]  = fmaf(fval, 6.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 4.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 2.400000000000e+01f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 2.400000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 3.000000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 8.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -3.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -2.400000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 2.400000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -6.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 4.000000000000e+00f, sum2.data[6]);
    pix = lbufptr[86];
    fval = hip_unpack0(pix.x);
    sum1.data[1]  = fmaf(fval, 6.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 4.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 2.400000000000e+01f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 2.400000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 3.000000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 8.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -3.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -2.400000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 2.400000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -6.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 4.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack1(pix.x);
    sum1.data[2]  = fmaf(fval, 6.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 4.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 2.400000000000e+01f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 2.400000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 3.000000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 8.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -3.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -2.400000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 2.400000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3]  = fmaf(fval, 6.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 4.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 2.400000000000e+01f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 2.400000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 3.000000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 8.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -3.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4]  = fmaf(fval, 6.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 4.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 2.400000000000e+01f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 2.400000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 3.000000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 8.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5]  = fmaf(fval, 6.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 4.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 2.400000000000e+01f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 2.400000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 3.000000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6]  = fmaf(fval, 6.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 4.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 2.400000000000e+01f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 2.400000000000e+01f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7]  = fmaf(fval, 6.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 4.000000000000e+00f, sum2.data[7]);
    // filterRow = 6
    pix = lbufptr[102];
    fval = hip_unpack1(pix.x);
    sum1.data[0] -= fval;
    sum2.data[0] += fval;
    fval = hip_unpack2(pix.x);
    sum1.data[0]  = fmaf(fval, -4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+00f, sum2.data[0]);
    sum1.data[1] -= fval;
    sum2.data[1] += fval;
    fval = hip_unpack3(pix.x);
    sum1.data[0]  = fmaf(fval, -5.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 1.500000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+00f, sum2.data[1]);
    sum1.data[2] -= fval;
    sum2.data[2] += fval;
    fval = hip_unpack0(pix.y);
    sum2.data[0]  = fmaf(fval, 2.000000000000e+01f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, -5.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 1.500000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+00f, sum2.data[2]);
    sum1.data[3] -= fval;
    sum2.data[3] += fval;
    fval = hip_unpack1(pix.y);
    sum1.data[0]  = fmaf(fval, 5.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 1.500000000000e+01f, sum2.data[0]);
    sum2.data[1]  = fmaf(fval, 2.000000000000e+01f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, -5.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 1.500000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+00f, sum2.data[3]);
    sum1.data[4] -= fval;
    sum2.data[4] += fval;
    fval = hip_unpack2(pix.y);
    sum1.data[0]  = fmaf(fval, 4.000000000000e+00f, sum1.data[0]);
    sum2.data[0]  = fmaf(fval, 6.000000000000e+00f, sum2.data[0]);
    sum1.data[1]  = fmaf(fval, 5.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 1.500000000000e+01f, sum2.data[1]);
    sum2.data[2]  = fmaf(fval, 2.000000000000e+01f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, -5.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 1.500000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+00f, sum2.data[4]);
    sum1.data[5] -= fval;
    sum2.data[5] += fval;
    fval = hip_unpack3(pix.y);
    sum1.data[0] += fval;
    sum2.data[0] += fval;
    sum1.data[1]  = fmaf(fval, 4.000000000000e+00f, sum1.data[1]);
    sum2.data[1]  = fmaf(fval, 6.000000000000e+00f, sum2.data[1]);
    sum1.data[2]  = fmaf(fval, 5.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 1.500000000000e+01f, sum2.data[2]);
    sum2.data[3]  = fmaf(fval, 2.000000000000e+01f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, -5.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 1.500000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+00f, sum2.data[5]);
    sum1.data[6] -= fval;
    sum2.data[6] += fval;
    pix = lbufptr[103];
    fval = hip_unpack0(pix.x);
    sum1.data[1] += fval;
    sum2.data[1] += fval;
    sum1.data[2]  = fmaf(fval, 4.000000000000e+00f, sum1.data[2]);
    sum2.data[2]  = fmaf(fval, 6.000000000000e+00f, sum2.data[2]);
    sum1.data[3]  = fmaf(fval, 5.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 1.500000000000e+01f, sum2.data[3]);
    sum2.data[4]  = fmaf(fval, 2.000000000000e+01f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, -5.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 1.500000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+00f, sum2.data[6]);
    sum1.data[7] -= fval;
    sum2.data[7] += fval;
    fval = hip_unpack1(pix.x);
    sum1.data[2] += fval;
    sum2.data[2] += fval;
    sum1.data[3]  = fmaf(fval, 4.000000000000e+00f, sum1.data[3]);
    sum2.data[3]  = fmaf(fval, 6.000000000000e+00f, sum2.data[3]);
    sum1.data[4]  = fmaf(fval, 5.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 1.500000000000e+01f, sum2.data[4]);
    sum2.data[5]  = fmaf(fval, 2.000000000000e+01f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, -5.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 1.500000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.x);
    sum1.data[3] += fval;
    sum2.data[3] += fval;
    sum1.data[4]  = fmaf(fval, 4.000000000000e+00f, sum1.data[4]);
    sum2.data[4]  = fmaf(fval, 6.000000000000e+00f, sum2.data[4]);
    sum1.data[5]  = fmaf(fval, 5.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 1.500000000000e+01f, sum2.data[5]);
    sum2.data[6]  = fmaf(fval, 2.000000000000e+01f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, -5.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 1.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack3(pix.x);
    sum1.data[4] += fval;
    sum2.data[4] += fval;
    sum1.data[5]  = fmaf(fval, 4.000000000000e+00f, sum1.data[5]);
    sum2.data[5]  = fmaf(fval, 6.000000000000e+00f, sum2.data[5]);
    sum1.data[6]  = fmaf(fval, 5.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 1.500000000000e+01f, sum2.data[6]);
    sum2.data[7]  = fmaf(fval, 2.000000000000e+01f, sum2.data[7]);
    fval = hip_unpack0(pix.y);
    sum1.data[5] += fval;
    sum2.data[5] += fval;
    sum1.data[6]  = fmaf(fval, 4.000000000000e+00f, sum1.data[6]);
    sum2.data[6]  = fmaf(fval, 6.000000000000e+00f, sum2.data[6]);
    sum1.data[7]  = fmaf(fval, 5.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 1.500000000000e+01f, sum2.data[7]);
    fval = hip_unpack1(pix.y);
    sum1.data[6] += fval;
    sum2.data[6] += fval;
    sum1.data[7]  = fmaf(fval, 4.000000000000e+00f, sum1.data[7]);
    sum2.data[7]  = fmaf(fval, 6.000000000000e+00f, sum2.data[7]);
    fval = hip_unpack2(pix.y);
    sum1.data[7] += fval;
    sum2.data[7] += fval;
    if ((x < dstWidth) && (y < dstHeight)) {
        uint dstIdx =  y * dstGxyStrideInBytes + (x << 2);

        float4 sum1X, sum1Y, sum2X, sum2Y;
        sum1X = make_float4(sum1.data[0], sum1.data[1], sum1.data[2], sum1.data[3]);
        sum1Y = make_float4(sum1.data[4], sum1.data[5], sum1.data[6], sum1.data[7]);
        sum2X = make_float4(sum2.data[0], sum2.data[1], sum2.data[2], sum2.data[3]);
        sum2Y = make_float4(sum2.data[4], sum2.data[5], sum2.data[6], sum2.data[7]);

        d_float8 dst = {0};

        *(float4 *)(&dst.data[0]) = sum1X * sum1X;
        *(float4 *)(&dst.data[4]) = sum1Y * sum1Y;
        *((d_float8 *)(&pDstGxy[dstIdx])) = dst;

        *(float4 *)(&dst.data[0]) = sum1X * sum2X;
        *(float4 *)(&dst.data[4]) = sum1Y * sum2Y;
        *((d_float8 *)(&pDstGxy[dstIdx + dstWidthComp1])) = dst;

        *(float4 *)(&dst.data[0]) = sum2X * sum2X;
        *(float4 *)(&dst.data[4]) = sum2Y * sum2Y;
        *((d_float8 *)(&pDstGxy[dstIdx + dstWidthComp2])) = dst;
    }
}

int HipExec_HarrisSobel_HG3_U8_7x7(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 *pHipDstGxy, vx_uint32 dstGxyStrideInBytes,
    vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_uint32 dstWidthComp1 = dstWidth * 4;
    vx_uint32 dstWidthComp2 = dstWidth * 8;

    hipLaunchKernelGGL(Hip_HarrisSobel_HG3_U8_7x7, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstGxy, dstGxyStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp1, dstWidthComp2);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_HarrisScore_HVC_HG3_3x3(uint dstWidth, uint dstHeight,
    uchar *pDstVc, uint dstVcStrideInBytes,
    uchar *pSrcGxy, uint srcGxyStrideInBytes,
    float sensitivity, float strength_threshold,
    int border, float normFactor,
    uint dstWidthComp1, uint dstWidthComp2) {

    int gx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int gy = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    __shared__ uchar lbuf[4896];
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;

    int gstride = srcGxyStrideInBytes;
    uint dstIdx =  gy * dstVcStrideInBytes + (gx << 4);

    uchar *gbuf;
    gbuf = pSrcGxy;
    uchar *lbuf_ptr;
    float2 v2;

    { // load 272x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 272 + (lx << 4);
        int goffset = (gy - 1) * gstride + (gx << 4) - 8;
        *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 272;
            goffset += 16 * gstride;
            doExtraLoad = true;
        }
        else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 272 + 256;
            goffset = (gy - ly + id - 1) * gstride + ((gx - lx) << 4) + 248;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        }
        __syncthreads();
    }

    float4 sum0;
    lbuf_ptr = &lbuf[ly * 272 + (lx << 4)];
    v2 = *(float2 *)(&lbuf_ptr[4]);
    sum0.x  = v2.x;
    sum0.x += v2.y;
    sum0.y  = v2.y;
    v2 = *(float2 *)(&lbuf_ptr[12]);
    sum0.x += v2.x;
    sum0.y += v2.x;
    sum0.z  = v2.x;
    sum0.y += v2.y;
    sum0.z += v2.y;
    sum0.w  = v2.y;
    v2 = *(float2 *)(&lbuf_ptr[20]);
    sum0.z += v2.x;
    sum0.w += v2.x;
    sum0.w += v2.y;
    *(float4 *)lbuf_ptr = sum0;
    if (ly < 2) {
        v2 = *(float2 *)(&lbuf_ptr[4356]);
        sum0.x  = v2.x;
        sum0.x += v2.y;
        sum0.y  = v2.y;
        v2 = *(float2 *)(&lbuf_ptr[4364]);
        sum0.x += v2.x;
        sum0.y += v2.x;
        sum0.z  = v2.x;
        sum0.y += v2.y;
        sum0.z += v2.y;
        sum0.w  = v2.y;
        v2 = *(float2 *)(&lbuf_ptr[4372]);
        sum0.z += v2.x;
        sum0.w += v2.x;
        sum0.w += v2.y;
        *((float4 *)&lbuf_ptr[4352]) = sum0;
    }

    __syncthreads();

    sum0  = *(float4 *)lbuf_ptr;
    sum0 += *(float4 *)&lbuf_ptr[272];
    sum0 += *(float4 *)&lbuf_ptr[544];

    __syncthreads();

    gbuf = pSrcGxy + dstWidthComp1;

    { // load 272x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 272 + (lx << 4);
        int goffset = (gy - 1) * gstride + (gx << 4) - 8;
        *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 272;
            goffset += 16 * gstride;
            doExtraLoad = true;
        }
        else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 272 + 256;
            goffset = (gy - ly + id - 1) * gstride + ((gx - lx) << 4) + 248;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        }
        __syncthreads();
    }

    float4 sum1;
    lbuf_ptr = &lbuf[ly * 272 + (lx << 4)];
    v2 = *(float2 *)(&lbuf_ptr[4]);
    sum1.x  = v2.x;
    sum1.x += v2.y;
    sum1.y  = v2.y;
    v2 = *(float2 *)(&lbuf_ptr[12]);
    sum1.x += v2.x;
    sum1.y += v2.x;
    sum1.z  = v2.x;
    sum1.y += v2.y;
    sum1.z += v2.y;
    sum1.w  = v2.y;
    v2 = *(float2 *)(&lbuf_ptr[20]);
    sum1.z += v2.x;
    sum1.w += v2.x;
    sum1.w += v2.y;
    *(float4 *)lbuf_ptr = sum1;
    if (ly < 2) {
        v2 = *(float2 *)(&lbuf_ptr[4356]);
        sum1.x  = v2.x;
        sum1.x += v2.y;
        sum1.y  = v2.y;
        v2 = *(float2 *)(&lbuf_ptr[4364]);
        sum1.x += v2.x;
        sum1.y += v2.x;
        sum1.z  = v2.x;
        sum1.y += v2.y;
        sum1.z += v2.y;
        sum1.w  = v2.y;
        v2 = *(float2 *)(&lbuf_ptr[4372]);
        sum1.z += v2.x;
        sum1.w += v2.x;
        sum1.w += v2.y;
        *(float4 *)&lbuf_ptr[4352] = sum1;
    }

    __syncthreads();

    sum1  = *(float4 *)lbuf_ptr;
    sum1 += *(float4 *)&lbuf_ptr[272];
    sum1 += *(float4 *)&lbuf_ptr[544];

    __syncthreads();

    gbuf = pSrcGxy + dstWidthComp2;

    { // load 272x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 272 + (lx << 4);
        int goffset = (gy - 1) * gstride + (gx << 4) - 8;
        *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 272;
            goffset += 16 * gstride;
            doExtraLoad = true;
        }
        else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 272 + 256;
            goffset = (gy - ly + id - 1) * gstride + ((gx - lx) << 4) + 248;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        }
        __syncthreads();
    }

    float4 sum2;
    lbuf_ptr = &lbuf[ly * 272 + (lx << 4)];
    v2 = *(float2 *)(&lbuf_ptr[4]);
    sum2.x  = v2.x;
    sum2.x += v2.y;
    sum2.y  = v2.y;
    v2 = *(float2 *)(&lbuf_ptr[12]);
    sum2.x += v2.x;
    sum2.y += v2.x;
    sum2.z  = v2.x;
    sum2.y += v2.y;
    sum2.z += v2.y;
    sum2.w  = v2.y;
    v2 = *(float2 *)(&lbuf_ptr[20]);
    sum2.z += v2.x;
    sum2.w += v2.x;
    sum2.w += v2.y;
    *(float4 *)lbuf_ptr = sum2;
    if (ly < 2) {
        v2 = *(float2 *)(&lbuf_ptr[4356]);
        sum2.x  = v2.x;
        sum2.x += v2.y;
        sum2.y  = v2.y;
        v2 = *(float2 *)(&lbuf_ptr[4364]);
        sum2.x += v2.x;
        sum2.y += v2.x;
        sum2.z  = v2.x;
        sum2.y += v2.y;
        sum2.z += v2.y;
        sum2.w  = v2.y;
        v2 = *(float2 *)(&lbuf_ptr[4372]);
        sum2.z += v2.x;
        sum2.w += v2.x;
        sum2.w += v2.y;
        *(float4 *)&lbuf_ptr[4352] = sum2;
    }

    __syncthreads();

    sum2  = *(float4 *)lbuf_ptr;
    sum2 += *(float4 *)&lbuf_ptr[272];
    sum2 += *(float4 *)&lbuf_ptr[544];

    gx = gx << 2;
    if ((gx < dstWidth) && (gy < dstHeight)) {
        float4 score = (float4)0.0f;
        if ((gy >= border) && (gy < dstHeight - border)) {
            score = sum0 * sum2 - sum1 * sum1;
            sum0 += sum2;
            sum0 *= sum0;
            score.x = fmaf(sum0.x, -sensitivity, score.x);
            score.y = fmaf(sum0.y, -sensitivity, score.y);
            score.z = fmaf(sum0.z, -sensitivity, score.z);
            score.w = fmaf(sum0.w, -sensitivity, score.w);
            score *= (float4)normFactor;
            score.x = HIPSELECT(0.0f, score.x, score.x > strength_threshold);
            score.y = HIPSELECT(0.0f, score.y, score.y > strength_threshold);
            score.z = HIPSELECT(0.0f, score.z, score.z > strength_threshold);
            score.w = HIPSELECT(0.0f, score.w, score.w > strength_threshold);
            score.x = HIPSELECT(score.x, 0.0f, gx < border);
            score.y = HIPSELECT(score.y, 0.0f, gx < border - 1);
            score.z = HIPSELECT(score.z, 0.0f, gx < border - 2);
            score.w = HIPSELECT(score.w, 0.0f, gx < border - 3);
            score.x = HIPSELECT(score.x, 0.0f, gx > dstWidth - 1 - border);
            score.y = HIPSELECT(score.y, 0.0f, gx > dstWidth - 2 - border);
            score.z = HIPSELECT(score.z, 0.0f, gx > dstWidth - 3 - border);
            score.w = HIPSELECT(score.w, 0.0f, gx > dstWidth - 4 - border);
        }
        *((float4 *)(&pDstVc[dstIdx])) = score;
    }
}
int HipExec_HarrisScore_HVC_HG3_3x3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 *pHipDstVc, vx_uint32 dstVcStrideInBytes, vx_float32 *pHipSrcGxy, vx_uint32 srcGxyStrideInBytes,
    vx_float32 sensitivity, vx_float32 strength_threshold, vx_int32 border, vx_float32 normFactor) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2;
    int globalThreads_y = dstHeight;

    vx_uint32 dstWidthComp1 = dstWidth * 4;
    vx_uint32 dstWidthComp2 = dstWidth * 8;

    hipLaunchKernelGGL(Hip_HarrisScore_HVC_HG3_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstVc, dstVcStrideInBytes,
                        (uchar *)pHipSrcGxy, srcGxyStrideInBytes, sensitivity, strength_threshold, border, normFactor,
                        dstWidthComp1, dstWidthComp2);

    return VX_SUCCESS;
}


__global__ void __attribute__((visibility("default")))
Hip_HarrisScore_HVC_HG3_5x5(uint dstWidth, uint dstHeight,
    uchar *pDstVc, uint dstVcStrideInBytes,
    uchar *pSrcGxy, uint srcGxyStrideInBytes, int srcImageBufferSize,
    float sensitivity, float strength_threshold,
    int border, float normFactor,
    uint dstWidthComp1, uint dstWidthComp2) {

    int gx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int gy = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    __shared__ uchar lbuf[5440];
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;

    int gstride = srcGxyStrideInBytes;
    uint dstIdx =  gy * dstVcStrideInBytes + (gx << 4);

    uchar *gbuf;
    gbuf = pSrcGxy;
    uchar *lbuf_ptr;
    float2 v2;

    { // load 272x20 bytes into local memory using 16x16 workgroup
        int loffset = ly * 272 + (lx << 4);
        int goffset = (gy - 2) * gstride + (gx << 4) - 8;
        *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        bool doExtraLoad = false;
        if (ly < 4) {
            loffset += 16 * 272;
            goffset += 16 * gstride;
            doExtraLoad = true;
        }
        else {
            int id = (ly - 4) * 16 + lx;
            loffset = id * 272 + 256;
            goffset = (gy - ly + id - 2) * gstride + ((gx - lx) << 4) + 248;
            doExtraLoad = (id < 20) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        }
        __syncthreads();
    }

    float4 sum0;
    lbuf_ptr = &lbuf[ly * 272 + (lx << 4)];
    v2 = *(float2 *)&lbuf_ptr[0];
    sum0.x  = v2.x;
    sum0.x += v2.y;
    sum0.y  = v2.y;
    v2 = *(float2 *)&lbuf_ptr[8];
    sum0.x += v2.x;
    sum0.y += v2.x;
    sum0.z  = v2.x;
    sum0.x += v2.y;
    sum0.y += v2.y;
    sum0.z += v2.y;
    sum0.w  = v2.y;
    v2 = *(float2 *)&lbuf_ptr[16];
    sum0.x += v2.x;
    sum0.y += v2.x;
    sum0.z += v2.x;
    sum0.w += v2.x;
    sum0.y += v2.y;
    sum0.z += v2.y;
    sum0.w += v2.y;
    v2 = *(float2 *)&lbuf_ptr[24];
    sum0.z += v2.x;
    sum0.w += v2.x;
    sum0.w += v2.y;

    *(float4 *)lbuf_ptr = sum0;
    if (ly < 4) {
        v2 = *(float2 *)&lbuf_ptr[4352];
        sum0.x  = v2.x;
        sum0.x += v2.y;
        sum0.y  = v2.y;
        v2 = *(float2 *)&lbuf_ptr[4360];
        sum0.x += v2.x;
        sum0.y += v2.x;
        sum0.z  = v2.x;
        sum0.x += v2.y;
        sum0.y += v2.y;
        sum0.z += v2.y;
        sum0.w  = v2.y;
        v2 = *(float2 *)&lbuf_ptr[4368];
        sum0.x += v2.x;
        sum0.y += v2.x;
        sum0.z += v2.x;
        sum0.w += v2.x;
        sum0.y += v2.y;
        sum0.z += v2.y;
        sum0.w += v2.y;
        v2 = *(float2 *)&lbuf_ptr[4376];
        sum0.z += v2.x;
        sum0.w += v2.x;
        sum0.w += v2.y;
        *((float4 *)&lbuf_ptr[4352]) = sum0;
    }

    __syncthreads();

    sum0  = *(float4 *)lbuf_ptr;
    sum0 += *(float4 *)&lbuf_ptr[272];
    sum0 += *(float4 *)&lbuf_ptr[544];
    sum0 += *(float4 *)&lbuf_ptr[816];
    sum0 += *(float4 *)&lbuf_ptr[1088];

    __syncthreads();

    gbuf = pSrcGxy + dstWidthComp1;

    { // load 272x20 bytes into local memory using 16x16 workgroup
        int loffset = ly * 272 + (lx << 4);
        int goffset = (gy - 2) * gstride + (gx << 4) - 8;
        *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        bool doExtraLoad = false;
        if (ly < 4) {
            loffset += 16 * 272;
            goffset += 16 * gstride;
            doExtraLoad = true;
        }
        else {
            int id = (ly - 4) * 16 + lx;
            loffset = id * 272 + 256;
            goffset = (gy - ly + id - 2) * gstride + ((gx - lx) << 4) + 248;
            doExtraLoad = (id < 20) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        }
        __syncthreads();
    }

    float4 sum1;
    lbuf_ptr = &lbuf[ly * 272 + (lx << 4)];
    v2 = *(float2 *)&lbuf_ptr[0];
    sum1.x  = v2.x;
    sum1.x += v2.y;
    sum1.y  = v2.y;
    v2 = *(float2 *)&lbuf_ptr[8];
    sum1.x += v2.x;
    sum1.y += v2.x;
    sum1.z  = v2.x;
    sum1.x += v2.y;
    sum1.y += v2.y;
    sum1.z += v2.y;
    sum1.w  = v2.y;
    v2 = *(float2 *)&lbuf_ptr[16];
    sum1.x += v2.x;
    sum1.y += v2.x;
    sum1.z += v2.x;
    sum1.w += v2.x;
    sum1.y += v2.y;
    sum1.z += v2.y;
    sum1.w += v2.y;
    v2 = *(float2 *)&lbuf_ptr[24];
    sum1.z += v2.x;
    sum1.w += v2.x;
    sum1.w += v2.y;
    *(float4 *)lbuf_ptr = sum1;
    if (ly < 4) {
        v2 = *(float2 *)&lbuf_ptr[4352];
        sum1.x  = v2.x;
        sum1.x += v2.y;
        sum1.y  = v2.y;
        v2 = *(float2 *)&lbuf_ptr[4360];
        sum1.x += v2.x;
        sum1.y += v2.x;
        sum1.z  = v2.x;
        sum1.x += v2.y;
        sum1.y += v2.y;
        sum1.z += v2.y;
        sum1.w  = v2.y;
        v2 = *(float2 *)&lbuf_ptr[4368];
        sum1.x += v2.x;
        sum1.y += v2.x;
        sum1.z += v2.x;
        sum1.w += v2.x;
        sum1.y += v2.y;
        sum1.z += v2.y;
        sum1.w += v2.y;
        v2 = *(float2 *)&lbuf_ptr[4376];
        sum1.z += v2.x;
        sum1.w += v2.x;
        sum1.w += v2.y;
        *(float4 *)&lbuf_ptr[4352] = sum1;
    }

    __syncthreads();

    sum1  = *(float4 *)lbuf_ptr;
    sum1 += *(float4 *)&lbuf_ptr[272];
    sum1 += *(float4 *)&lbuf_ptr[544];
    sum1 += *(float4 *)&lbuf_ptr[816];
    sum1 += *(float4 *)&lbuf_ptr[1088];

    __syncthreads();

    gbuf = pSrcGxy + dstWidthComp2;

    { // load 272x20 bytes into local memory using 16x16 workgroup
        int loffset = ly * 272 + (lx << 4);
        int goffset = (gy - 2) * gstride + (gx << 4) - 8;
        *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        bool doExtraLoad = false;
        if (ly < 4) {
            loffset += 16 * 272;
            goffset += 16 * gstride;
            doExtraLoad = true;
        }
        else {
            int id = (ly - 4) * 16 + lx;
            loffset = id * 272 + 256;
            goffset = (gy - ly + id - 2) * gstride + ((gx - lx) << 4) + 248;
            doExtraLoad = (id < 20) ? true : false;
        }
        if (doExtraLoad && goffset < srcImageBufferSize) {
            *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        }
        __syncthreads();
    }

    float4 sum2;
    lbuf_ptr = &lbuf[ly * 272 + (lx << 4)];
    v2 = *(float2 *)&lbuf_ptr[0];
    sum2.x  = v2.x;
    sum2.x += v2.y;
    sum2.y  = v2.y;
    v2 = *(float2 *)&lbuf_ptr[8];
    sum2.x += v2.x;
    sum2.y += v2.x;
    sum2.z  = v2.x;
    sum2.x += v2.y;
    sum2.y += v2.y;
    sum2.z += v2.y;
    sum2.w  = v2.y;
    v2 = *(float2 *)&lbuf_ptr[16];
    sum2.x += v2.x;
    sum2.y += v2.x;
    sum2.z += v2.x;
    sum2.w += v2.x;
    sum2.y += v2.y;
    sum2.z += v2.y;
    sum2.w += v2.y;
    v2 = *(float2 *)&lbuf_ptr[24];
    sum2.z += v2.x;
    sum2.w += v2.x;
    sum2.w += v2.y;
    *(float4 *)lbuf_ptr = sum2;
    if (ly < 4) {
        v2 = *(float2 *)&lbuf_ptr[4352];
        sum2.x  = v2.x;
        sum2.x += v2.y;
        sum2.y  = v2.y;
        v2 = *(float2 *)&lbuf_ptr[4360];
        sum2.x += v2.x;
        sum2.y += v2.x;
        sum2.z  = v2.x;
        sum2.x += v2.y;
        sum2.y += v2.y;
        sum2.z += v2.y;
        sum2.w  = v2.y;
        v2 = *(float2 *)&lbuf_ptr[4368];
        sum2.x += v2.x;
        sum2.y += v2.x;
        sum2.z += v2.x;
        sum2.w += v2.x;
        sum2.y += v2.y;
        sum2.z += v2.y;
        sum2.w += v2.y;
        v2 = *(float2 *)&lbuf_ptr[4376];
        sum2.z += v2.x;
        sum2.w += v2.x;
        sum2.w += v2.y;
        *(float4 *)&lbuf_ptr[4352] = sum2;
    }

    __syncthreads();

    sum2  = *(float4 *)lbuf_ptr;
    sum2 += *(float4 *)&lbuf_ptr[272];
    sum2 += *(float4 *)&lbuf_ptr[544];
    sum2 += *(float4 *)&lbuf_ptr[816];
    sum2 += *(float4 *)&lbuf_ptr[1088];

    gx = gx << 2;
    if ((gx < dstWidth) && (gy < dstHeight)) {
        float4 score = (float4)0.0f;
        if ((gy >= border) && (gy < dstHeight - border)) {
            score = sum0 * sum2 - sum1 * sum1;
            sum0 += sum2;
            sum0 *= sum0;
            score.x = fmaf(sum0.x, -sensitivity, score.x);
            score.y = fmaf(sum0.y, -sensitivity, score.y);
            score.z = fmaf(sum0.z, -sensitivity, score.z);
            score.w = fmaf(sum0.w, -sensitivity, score.w);
            score *= (float4)normFactor;
            score.x = HIPSELECT(0.0f, score.x, score.x > strength_threshold);
            score.y = HIPSELECT(0.0f, score.y, score.y > strength_threshold);
            score.z = HIPSELECT(0.0f, score.z, score.z > strength_threshold);
            score.w = HIPSELECT(0.0f, score.w, score.w > strength_threshold);
            score.x = HIPSELECT(score.x, 0.0f, gx < border);
            score.y = HIPSELECT(score.y, 0.0f, gx < border - 1);
            score.z = HIPSELECT(score.z, 0.0f, gx < border - 2);
            score.w = HIPSELECT(score.w, 0.0f, gx < border - 3);
            score.x = HIPSELECT(score.x, 0.0f, gx > dstWidth - 1 - border);
            score.y = HIPSELECT(score.y, 0.0f, gx > dstWidth - 2 - border);
            score.z = HIPSELECT(score.z, 0.0f, gx > dstWidth - 3 - border);
            score.w = HIPSELECT(score.w, 0.0f, gx > dstWidth - 4 - border);
        }
        *((float4 *)(&pDstVc[dstIdx])) = score;
    }
}

int HipExec_HarrisScore_HVC_HG3_5x5(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 *pHipDstVc, vx_uint32 dstVcStrideInBytes, vx_float32 *pHipSrcGxy, vx_uint32 srcGxyStrideInBytes, vx_int32 srcImageBufferSize,
    vx_float32 sensitivity, vx_float32 strength_threshold, vx_int32 border, vx_float32 normFactor) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2;
    int globalThreads_y = dstHeight;

    vx_uint32 dstWidthComp1 = dstWidth * 4;
    vx_uint32 dstWidthComp2 = dstWidth * 8;

    hipLaunchKernelGGL(Hip_HarrisScore_HVC_HG3_5x5, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstVc, dstVcStrideInBytes,
                        (uchar *)pHipSrcGxy, srcGxyStrideInBytes, srcImageBufferSize, sensitivity, strength_threshold, border, normFactor,
                        dstWidthComp1, dstWidthComp2);

    return VX_SUCCESS;

}

__global__ void __attribute__((visibility("default")))
Hip_HarrisScore_HVC_HG3_7x7(uint dstWidth, uint dstHeight,
    uchar *pDstVc, uint dstVcStrideInBytes,
    uchar *pSrcGxy, uint srcGxyStrideInBytes,
    float sensitivity, float strength_threshold,
    int border, float normFactor,
    uint dstWidthComp1, uint dstWidthComp2) {

    int gx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int gy = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    __shared__ uchar lbuf[6336];
    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;

    int gstride = srcGxyStrideInBytes;
    uint dstIdx =  gy * dstVcStrideInBytes + (gx << 4);

    uchar *gbuf;
    gbuf = pSrcGxy;
    uchar *lbuf_ptr;
    float2 v2;

    { // load 288x22 bytes into local memory using 16x16 workgroup
        int loffset = ly * 288 + (lx << 4);
        int goffset = (gy - 3) * gstride + (gx << 4) - 16;
        *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        bool doExtraLoad = false;
        if (ly < 6) {
            loffset += 16 * 288;
            goffset += 16 * gstride;
            doExtraLoad = true;
        }
        else {
            int id = (ly - 6) * 16 + lx;
            int ry = id >> 1;
            int rx = id & 1;
            loffset = ry * 288 + (rx << 4) + 256;
            goffset = (gy - ly + ry - 3) * gstride + ((gx - lx + rx) << 4) + 240;
            doExtraLoad = (ry < 22) ? true : false;
        }
        if (doExtraLoad) {
            *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        }
        __syncthreads();
    }

    float4 sum0;
    lbuf_ptr = &lbuf[ly * 288 + (lx << 4)];
    v2 = *(float2 *)&lbuf_ptr[4];
    sum0.x  = v2.x;
    sum0.x += v2.y;
    sum0.y  = v2.y;
    v2 = *(float2 *)&lbuf_ptr[12];
    sum0.x += v2.x;
    sum0.y += v2.x;
    sum0.z  = v2.x;
    sum0.x += v2.y;
    sum0.y += v2.y;
    sum0.z += v2.y;
    sum0.w  = v2.y;
    v2 = *(float2 *)&lbuf_ptr[20];
    sum0.x += v2.x;
    sum0.y += v2.x;
    sum0.z += v2.x;
    sum0.w += v2.x;
    sum0.x += v2.y;
    sum0.y += v2.y;
    sum0.z += v2.y;
    sum0.w += v2.y;
    v2 = *(float2 *)&lbuf_ptr[28];
    sum0.x += v2.x;
    sum0.y += v2.x;
    sum0.z += v2.x;
    sum0.w += v2.x;
    sum0.y += v2.y;
    sum0.z += v2.y;
    sum0.w += v2.y;
    v2 = *(float2 *)&lbuf_ptr[36];
    sum0.z += v2.x;
    sum0.w += v2.x;
    sum0.w += v2.y;

    *(float4 *)lbuf_ptr = sum0;
    if (ly < 6) {
        v2 = *(float2 *)&lbuf_ptr[4612];
        sum0.x  = v2.x;
        sum0.x += v2.y;
        sum0.y  = v2.y;
        v2 = *(float2 *)&lbuf_ptr[4620];
        sum0.x += v2.x;
        sum0.y += v2.x;
        sum0.z  = v2.x;
        sum0.x += v2.y;
        sum0.y += v2.y;
        sum0.z += v2.y;
        sum0.w  = v2.y;
        v2 = *(float2 *)&lbuf_ptr[4628];
        sum0.x += v2.x;
        sum0.y += v2.x;
        sum0.z += v2.x;
        sum0.w += v2.x;
        sum0.x += v2.y;
        sum0.y += v2.y;
        sum0.z += v2.y;
        sum0.w += v2.y;
        v2 = *(float2 *)&lbuf_ptr[4636];
        sum0.x += v2.x;
        sum0.y += v2.x;
        sum0.z += v2.x;
        sum0.w += v2.x;
        sum0.y += v2.y;
        sum0.z += v2.y;
        sum0.w += v2.y;
        v2 = *(float2 *)&lbuf_ptr[4644];
        sum0.z += v2.x;
        sum0.w += v2.x;
        sum0.w += v2.y;
        *(float4 *)&lbuf_ptr[4608] = sum0;
    }

    __syncthreads();

    sum0  = *(float4 *)lbuf_ptr;
    sum0 += *(float4 *)&lbuf_ptr[288];
    sum0 += *(float4 *)&lbuf_ptr[576];
    sum0 += *(float4 *)&lbuf_ptr[864];
    sum0 += *(float4 *)&lbuf_ptr[1152];
    sum0 += *(float4 *)&lbuf_ptr[1440];
    sum0 += *(float4 *)&lbuf_ptr[1728];

    __syncthreads();

    gbuf = pSrcGxy + dstWidthComp1;

    { // load 288x22 bytes into local memory using 16x16 workgroup
        int loffset = ly * 288 + (lx << 4);
        int goffset = (gy - 3) * gstride + (gx << 4) - 16;
        *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        bool doExtraLoad = false;
        if (ly < 6) {
            loffset += 16 * 288;
            goffset += 16 * gstride;
            doExtraLoad = true;
        }
        else {
            int id = (ly - 6) * 16 + lx;
            int ry = id >> 1;
            int rx = id & 1;
            loffset = ry * 288 + (rx << 4) + 256;
            goffset = (gy - ly + ry - 3) * gstride + ((gx - lx + rx) << 4) + 240;
            doExtraLoad = (ry < 22) ? true : false;
        }
        if (doExtraLoad) {
            *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        }
        __syncthreads();
    }

    float4 sum1;
    lbuf_ptr = &lbuf[ly * 288 + (lx << 4)];
    v2 = *(float2 *)&lbuf_ptr[4];
    sum1.x  = v2.x;
    sum1.x += v2.y;
    sum1.y  = v2.y;
    v2 = *(float2 *)&lbuf_ptr[12];
    sum1.x += v2.x;
    sum1.y += v2.x;
    sum1.z  = v2.x;
    sum1.x += v2.y;
    sum1.y += v2.y;
    sum1.z += v2.y;
    sum1.w  = v2.y;
    v2 = *(float2 *)&lbuf_ptr[20];
    sum1.x += v2.x;
    sum1.y += v2.x;
    sum1.z += v2.x;
    sum1.w += v2.x;
    sum1.x += v2.y;
    sum1.y += v2.y;
    sum1.z += v2.y;
    sum1.w += v2.y;
    v2 = *(float2 *)&lbuf_ptr[28];
    sum1.x += v2.x;
    sum1.y += v2.x;
    sum1.z += v2.x;
    sum1.w += v2.x;
    sum1.y += v2.y;
    sum1.z += v2.y;
    sum1.w += v2.y;
    v2 = *(float2 *)&lbuf_ptr[36];
    sum1.z += v2.x;
    sum1.w += v2.x;
    sum1.w += v2.y;
    *(float4 *)lbuf_ptr = sum1;
    if (ly < 6) {
        v2 = *(float2 *)&lbuf_ptr[4612];
        sum1.x  = v2.x;
        sum1.x += v2.y;
        sum1.y  = v2.y;
        v2 = *(float2 *)&lbuf_ptr[4620];
        sum1.x += v2.x;
        sum1.y += v2.x;
        sum1.z  = v2.x;
        sum1.x += v2.y;
        sum1.y += v2.y;
        sum1.z += v2.y;
        sum1.w  = v2.y;
        v2 = *(float2 *)&lbuf_ptr[4628];
        sum1.x += v2.x;
        sum1.y += v2.x;
        sum1.z += v2.x;
        sum1.w += v2.x;
        sum1.x += v2.y;
        sum1.y += v2.y;
        sum1.z += v2.y;
        sum1.w += v2.y;
        v2 = *(float2 *)&lbuf_ptr[4636];
        sum1.x += v2.x;
        sum1.y += v2.x;
        sum1.z += v2.x;
        sum1.w += v2.x;
        sum1.y += v2.y;
        sum1.z += v2.y;
        sum1.w += v2.y;
        v2 = *(float2 *)&lbuf_ptr[4644];
        sum1.z += v2.x;
        sum1.w += v2.x;
        sum1.w += v2.y;
        *(float4 *)&lbuf_ptr[4608] = sum1;
    }

    __syncthreads();

    sum1  = *(float4 *)lbuf_ptr;
    sum1 += *(float4 *)&lbuf_ptr[288];
    sum1 += *(float4 *)&lbuf_ptr[576];
    sum1 += *(float4 *)&lbuf_ptr[864];
    sum1 += *(float4 *)&lbuf_ptr[1152];
    sum1 += *(float4 *)&lbuf_ptr[1440];
    sum1 += *(float4 *)&lbuf_ptr[1728];

    __syncthreads();

    gbuf = pSrcGxy + dstWidthComp2;

    { // load 288x22 bytes into local memory using 16x16 workgroup
        int loffset = ly * 288 + (lx << 4);
        int goffset = (gy - 3) * gstride + (gx << 4) - 16;
        *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        bool doExtraLoad = false;
        if (ly < 6) {
            loffset += 16 * 288;
            goffset += 16 * gstride;
            doExtraLoad = true;
        }
        else {
            int id = (ly - 6) * 16 + lx;
            int ry = id >> 1;
            int rx = id & 1;
            loffset = ry * 288 + (rx << 4) + 256;
            goffset = (gy - ly + ry - 3) * gstride + ((gx - lx + rx) << 4) + 240;
            doExtraLoad = (ry < 22) ? true : false;
        }
        if (doExtraLoad) {
            *((uint4 *)(&lbuf[loffset])) = *((uint4 *)(&gbuf[goffset]));
        }
        __syncthreads();
    }

    float4 sum2;
    lbuf_ptr = &lbuf[ly * 288 + (lx << 4)];
    v2 = *(float2 *)&lbuf_ptr[4];
    sum2.x  = v2.x;
    sum2.x += v2.y;
    sum2.y  = v2.y;
    v2 = *(float2 *)&lbuf_ptr[12];
    sum2.x += v2.x;
    sum2.y += v2.x;
    sum2.z  = v2.x;
    sum2.x += v2.y;
    sum2.y += v2.y;
    sum2.z += v2.y;
    sum2.w  = v2.y;
    v2 = *(float2 *)&lbuf_ptr[20];
    sum2.x += v2.x;
    sum2.y += v2.x;
    sum2.z += v2.x;
    sum2.w += v2.x;
    sum2.x += v2.y;
    sum2.y += v2.y;
    sum2.z += v2.y;
    sum2.w += v2.y;
    v2 = *(float2 *)&lbuf_ptr[28];
    sum2.x += v2.x;
    sum2.y += v2.x;
    sum2.z += v2.x;
    sum2.w += v2.x;
    sum2.y += v2.y;
    sum2.z += v2.y;
    sum2.w += v2.y;
    v2 = *(float2 *)&lbuf_ptr[36];
    sum2.z += v2.x;
    sum2.w += v2.x;
    sum2.w += v2.y;
    *(float4 *)lbuf_ptr = sum2;
    if (ly < 6) {
        v2 = *(float2 *)&lbuf_ptr[4612];
        sum2.x  = v2.x;
        sum2.x += v2.y;
        sum2.y  = v2.y;
        v2 = *(float2 *)&lbuf_ptr[4620];
        sum2.x += v2.x;
        sum2.y += v2.x;
        sum2.z  = v2.x;
        sum2.x += v2.y;
        sum2.y += v2.y;
        sum2.z += v2.y;
        sum2.w  = v2.y;
        v2 = *(float2 *)&lbuf_ptr[4628];
        sum2.x += v2.x;
        sum2.y += v2.x;
        sum2.z += v2.x;
        sum2.w += v2.x;
        sum2.x += v2.y;
        sum2.y += v2.y;
        sum2.z += v2.y;
        sum2.w += v2.y;
        v2 = *(float2 *)&lbuf_ptr[4636];
        sum2.x += v2.x;
        sum2.y += v2.x;
        sum2.z += v2.x;
        sum2.w += v2.x;
        sum2.y += v2.y;
        sum2.z += v2.y;
        sum2.w += v2.y;
        v2 = *(float2 *)&lbuf_ptr[4644];
        sum2.z += v2.x;
        sum2.w += v2.x;
        sum2.w += v2.y;
        *(float4 *)&lbuf_ptr[4608] = sum2;
    }

    __syncthreads();

    sum2  = *(float4 *)lbuf_ptr;
    sum2 += *(float4 *)&lbuf_ptr[288];
    sum2 += *(float4 *)&lbuf_ptr[576];
    sum2 += *(float4 *)&lbuf_ptr[864];
    sum2 += *(float4 *)&lbuf_ptr[1152];
    sum2 += *(float4 *)&lbuf_ptr[1440];
    sum2 += *(float4 *)&lbuf_ptr[1728];

    gx = gx << 2;
    if ((gx < dstWidth) && (gy < dstHeight)) {
        float4 score = (float4)0.0f;
        if ((gy >= border) && (gy < dstHeight - border)) {
            score = sum0 * sum2 - sum1 * sum1;
            sum0 += sum2;
            sum0 *= sum0;
            score.x = fmaf(sum0.x, -sensitivity, score.x);
            score.y = fmaf(sum0.y, -sensitivity, score.y);
            score.z = fmaf(sum0.z, -sensitivity, score.z);
            score.w = fmaf(sum0.w, -sensitivity, score.w);
            score *= (float4)normFactor;
            score.x = HIPSELECT(0.0f, score.x, score.x > strength_threshold);
            score.y = HIPSELECT(0.0f, score.y, score.y > strength_threshold);
            score.z = HIPSELECT(0.0f, score.z, score.z > strength_threshold);
            score.w = HIPSELECT(0.0f, score.w, score.w > strength_threshold);
            score.x = HIPSELECT(score.x, 0.0f, gx < border);
            score.y = HIPSELECT(score.y, 0.0f, gx < border - 1);
            score.z = HIPSELECT(score.z, 0.0f, gx < border - 2);
            score.w = HIPSELECT(score.w, 0.0f, gx < border - 3);
            score.x = HIPSELECT(score.x, 0.0f, gx > dstWidth - 1 - border);
            score.y = HIPSELECT(score.y, 0.0f, gx > dstWidth - 2 - border);
            score.z = HIPSELECT(score.z, 0.0f, gx > dstWidth - 3 - border);
            score.w = HIPSELECT(score.w, 0.0f, gx > dstWidth - 4 - border);
        }
        *((float4 *)(&pDstVc[dstIdx])) = score;
    }
}

int HipExec_HarrisScore_HVC_HG3_7x7(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_float32 *pHipDstVc, vx_uint32 dstVcStrideInBytes, vx_float32 *pHipSrcGxy, vx_uint32 srcGxyStrideInBytes,
    vx_float32 sensitivity, vx_float32 strength_threshold, vx_int32 border, vx_float32 normFactor) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 3) >> 2;
    int globalThreads_y = dstHeight;

    vx_uint32 dstWidthComp1 = dstWidth * 4;
    vx_uint32 dstWidthComp2 = dstWidth * 8;

    hipLaunchKernelGGL(Hip_HarrisScore_HVC_HG3_7x7, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstVc, dstVcStrideInBytes,
                        (uchar *)pHipSrcGxy, srcGxyStrideInBytes, sensitivity, strength_threshold, border, normFactor,
                        dstWidthComp1, dstWidthComp2);

    return VX_SUCCESS;

}

__global__ void __attribute__((visibility("default")))
Hip_NonMaxSupp_XY_ANY_3x3(char *pDstList, uint dstListOffset, uint capacityOfList,
    uint srcWidth, uint srcHeight,
    uchar *pSrcImage, uint srcImageStrideInBytes,
    uint srcWidthComp1, uint srcWidthComp2) {

    int lx = hipThreadIdx_x;
    int ly = hipThreadIdx_y;
    int gx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int gy = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    __shared__ uchar lbuf[2448];
    { // load 136x18 bytes into local memory using 16x16 workgroup
        int loffset = ly * 136 + (lx << 3);
        int goffset = (gy - 1) * srcImageStrideInBytes + (gx << 3) - 4;
        *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        bool doExtraLoad = false;
        if (ly < 2) {
            loffset += 16 * 136;
            goffset += 16 * srcImageStrideInBytes;
            doExtraLoad = true;
        } else {
            int id = (ly - 2) * 16 + lx;
            loffset = id * 136 + 128;
            goffset = (gy - ly + id - 1) * srcImageStrideInBytes + ((gx  - lx) << 3) + 124;
            doExtraLoad = (id < 18) ? true : false;
        }
        if (doExtraLoad) {
            *((uint2 *)(&lbuf[loffset])) = *((uint2 *)(&pSrcImage[goffset]));
        }
        __syncthreads();
    }
    uchar *lbuf_ptr;
    lbuf_ptr = lbuf + ly * 136 + (lx << 3);
    float4 L0 = *(float4 *)lbuf_ptr;
    float4 L1 = *(float4 *)&lbuf_ptr[136];
    float4 L2 = *(float4 *)&lbuf_ptr[272];
    float2 T;
    T.x = L1.y;
    T.y = L1.z;
    T.x = HIPSELECT(0.0f, T.x, T.x >= L1.x);
    T.x = HIPSELECT(0.0f, T.x, T.x >  L1.z);
    T.x = HIPSELECT(0.0f, T.x, T.x >= L0.x);
    T.x = HIPSELECT(0.0f, T.x, T.x >= L0.y);
    T.x = HIPSELECT(0.0f, T.x, T.x >= L0.z);
    T.x = HIPSELECT(0.0f, T.x, T.x >  L2.x);
    T.x = HIPSELECT(0.0f, T.x, T.x >  L2.y);
    T.x = HIPSELECT(0.0f, T.x, T.x >  L2.z);
    T.y = HIPSELECT(0.0f, T.y, T.y >= L1.y);
    T.y = HIPSELECT(0.0f, T.y, T.y >  L1.w);
    T.y = HIPSELECT(0.0f, T.y, T.y >= L0.y);
    T.y = HIPSELECT(0.0f, T.y, T.y >= L0.z);
    T.y = HIPSELECT(0.0f, T.y, T.y >= L0.w);
    T.y = HIPSELECT(0.0f, T.y, T.y >  L2.y);
    T.y = HIPSELECT(0.0f, T.y, T.y >  L2.z);
    T.y = HIPSELECT(0.0f, T.y, T.y >  L2.w);
    T.x = HIPSELECT(0.0f, T.x, gx < srcWidthComp1);
    T.y = HIPSELECT(0.0f, T.y, gx < srcWidthComp2);
    T.x = HIPSELECT(0.0f, T.x, gy < srcHeight);
    T.y = HIPSELECT(0.0f, T.y, gy < srcHeight);
    gx = gx + gx + HIPSELECT(0, 1, T.y > 0.0f);
    T.x = HIPSELECT(T.x, T.y, T.y > 0.0f);
    if (T.x > 0.0f) {
        int pos = atomicAdd((int *)pDstList, 1);
        if(pos < capacityOfList) {
            *((uint2 *)(&pDstList[dstListOffset + (pos << 3)])) = make_uint2(gx | (gy << 16), __float_as_uint(T.x));
        }
    }
}
int HipExec_NonMaxSupp_XY_ANY_3x3(hipStream_t stream, vx_uint32 capacityOfList, vx_uint8* pHipDstList, vx_uint32 dstListOffset,
    vx_uint32 srcWidth, vx_uint32 srcHeight, vx_float32 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (srcWidth + 1) / 2;
    int globalThreads_y = srcHeight;

    vx_uint32 srcWidthComp1 = (srcWidth + 1) / 2;
    vx_uint32 srcWidthComp2 = srcWidth / 2;

    hipLaunchKernelGGL(Hip_NonMaxSupp_XY_ANY_3x3, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, (char *)pHipDstList, dstListOffset, capacityOfList,
                        srcWidth, srcHeight, (uchar *)pHipSrcImage, srcImageStrideInBytes,
                        srcWidthComp1, srcWidthComp2);

    return VX_SUCCESS;
}