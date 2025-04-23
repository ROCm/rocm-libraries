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
// VxScaleImage kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Nearest(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    float xscale, float yscale, float xoffset, float yoffset) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    float4 scaleInfo = make_float4(xscale, yscale, xoffset, yoffset);

    uint2 dst;
    pSrcImage += srcImageStrideInBytes * (uint)fmaf((float)y, scaleInfo.y, scaleInfo.w);
    float fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);

    dst.x  = pSrcImage[(int)fx];
    fx += scaleInfo.x;
    dst.x |= pSrcImage[(int)fx] << 8;
    fx += scaleInfo.x;
    dst.x |= pSrcImage[(int)fx] << 16;
    fx += scaleInfo.x;
    dst.x |= pSrcImage[(int)fx] << 24;

    fx += scaleInfo.x;

    dst.y  = pSrcImage[(int)fx];
    fx += scaleInfo.x;
    dst.y |= pSrcImage[(int)fx] << 8;
    fx += scaleInfo.x;
    dst.y |= pSrcImage[(int)fx] << 16;
    fx += scaleInfo.x;
    dst.y |= pSrcImage[(int)fx] << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ScaleImage_U8_U8_Nearest(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_float32 xscale = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth);
    vx_float32 yscale = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight);
    vx_float32 xoffset = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth * 0.5);
    vx_float32 yoffset = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight * 0.5);

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Nearest, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        xscale, yscale, xoffset, yoffset);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Bilinear(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    float xscale, float yscale, float xoffset, float yoffset) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    float4 scaleInfo = make_float4(xscale, yscale, xoffset, yoffset);

    uint2 dst;
    float fx, fy, fint, frac, fy0, fy1;
    float4 f;
    fy = fmaf((float)y, scaleInfo.y, scaleInfo.w);
    fy0 = floorf(fy);
    fy1 = fy - fy0;
    fy0 = 1.0f - fy1;
    pSrcImage += hip_mul24((uint)fy, srcImageStrideInBytes);

    fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);
    fint = floorf(fx);
    frac = fx - fint;
    f.x = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    fx += scaleInfo.x;
    fint = floorf(fx);
    frac = fx - fint;
    f.y = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    fx += scaleInfo.x;
    fint = floorf(fx);
    frac = fx - fint;
    f.z = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    fx += scaleInfo.x;
    fint = floorf(fx);
    frac = fx - fint;
    f.w = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    dst.x = hip_pack(f);

    fx += scaleInfo.x;
    fint = floorf(fx);
    frac = fx - fint;
    f.x = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    fx += scaleInfo.x;
    fint = floorf(fx);
    frac = fx - fint;
    f.y = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    fx += scaleInfo.x;
    fint = floorf(fx);
    frac = fx - fint;
    f.z = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    fx += scaleInfo.x;
    fint = floorf(fx);
    frac = fx - fint;
    f.w = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
    dst.y = hip_pack(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ScaleImage_U8_U8_Bilinear(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_float32 xscale = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth);
    vx_float32 yscale = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight);
    vx_float32 xoffset = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth * 0.5 - 0.5);
    vx_float32 yoffset = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight * 0.5 - 0.5);

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Bilinear, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        xscale, yscale, xoffset, yoffset);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Bilinear_Replicate(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes, uint srcWidth, uint srcHeight,
    float xscale, float yscale, float xoffset, float yoffset) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    float4 scaleInfo = make_float4(xscale, yscale, xoffset, yoffset);

    float fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);
    float fy = fmaf((float)y, scaleInfo.y, scaleInfo.w);

    if (fx >= 0.0f && fy >= 0.0f && fmaf(8.0f, scaleInfo.x, fx) < (srcWidth - 1) && fmaf(1.0f, scaleInfo.y, fy) < (srcHeight - 1)) {
        uint2 dst;
        float fint, frac, fy0, fy1;
        float4 f;
        fy = fmaf((float)y, scaleInfo.y, scaleInfo.w);
        fy0 = floorf(fy);
        fy1 = fy - fy0;
        fy0 = 1.0f - fy1;
        pSrcImage += hip_mul24((uint)fy, srcImageStrideInBytes);

        fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);
        fint = floorf(fx);
        frac = fx - fint;
        f.x = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.y = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.z = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.w = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        dst.x = hip_pack(f);

        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.x = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.y = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.z = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.w = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        dst.y = hip_pack(f);

        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    } else {
        float fy0, fy1;
        fy0 = floorf(fy);
        fy1 = fy - fy0;
        fy0 = 1.0f - fy1;
        uint2 ycoord = hip_clamp_pixel_coordinates_to_border(fy, srcHeight - 1, srcImageStrideInBytes);
        pSrcImage += hip_mul24(ycoord.x, srcImageStrideInBytes);
        float frac;
        uint2 xcoord;
        uint xlimit = srcWidth - 1;

        uint2 dst;
        float4 f;

        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floorf(fx);
        f.x = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floorf(fx);
        f.y = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floorf(fx);
        f.z = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floorf(fx);
        f.w = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        dst.x = hip_pack(f);

        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floorf(fx);
        f.x = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floorf(fx);
        f.y = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floorf(fx);
        f.z = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        fx += scaleInfo.x;
        xcoord = hip_clamp_pixel_coordinates_to_border(fx, xlimit, 1);
        frac = fx - floorf(fx);
        f.w = hip_bilinear_sample(pSrcImage, ycoord.y, xcoord.y, fy0, fy1, xcoord.x, 1.0f - frac, frac);
        dst.y = hip_pack(f);

        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_ScaleImage_U8_U8_Bilinear_Replicate(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_float32 xscale = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth);
    vx_float32 yscale = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight);
    vx_float32 xoffset = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth * 0.5 - 0.5);
    vx_float32 yoffset = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight * 0.5 - 0.5);

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Bilinear_Replicate, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcWidth, srcHeight,
                        xscale, yscale, xoffset, yoffset);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Bilinear_Constant(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes, uint srcWidth, uint srcHeight,
    float xscale, float yscale, float xoffset, float yoffset, uint borderValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    float4 scaleInfo = make_float4(xscale, yscale, xoffset, yoffset);

    float fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);
    float fy = fmaf((float)y, scaleInfo.y, scaleInfo.w);

    if (fx >= 0.0f && fy >= 0.0f && fmaf(8.0f, scaleInfo.x, fx) < (srcWidth - 1) && fmaf(1.0f, scaleInfo.y, fy) < (srcHeight - 1)) {
        uint2 dst;
        float fint, frac, fy0, fy1;
        float4 f;
        fy = fmaf((float)y, scaleInfo.y, scaleInfo.w);
        fy0 = floorf(fy);
        fy1 = fy - fy0;
        fy0 = 1.0f - fy1;
        pSrcImage += hip_mul24((uint)fy, srcImageStrideInBytes);

        fx = fmaf((float)x, scaleInfo.x, scaleInfo.z);
        fint = floorf(fx);
        frac = fx - fint;
        f.x = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.y = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.z = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.w = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        dst.x = hip_pack(f);

        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.x = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.y = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.z = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        fx += scaleInfo.x;
        fint = floorf(fx);
        frac = fx - fint;
        f.w = hip_bilinear_sample(pSrcImage, srcImageStrideInBytes, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);
        dst.y = hip_pack(f);

        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    } else {
        float fy1 = fy - floorf(fy);
        float fy0 = 1.0f - fy1;
        int sy = (int) floorf(fy);
        float frac;
        uint2 dst;
        float4 f;
        frac = fx - floorf(fx);

        f.x = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floorf(fx), sy, srcWidth, srcHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        fx += scaleInfo.x;
        frac = fx - floorf(fx);
        f.y = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floorf(fx), sy, srcWidth, srcHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        fx += scaleInfo.x;
        frac = fx - floorf(fx);
        f.z = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floorf(fx), sy, srcWidth, srcHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        fx += scaleInfo.x;
        frac = fx - floorf(fx);
        f.w = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floorf(fx), sy, srcWidth, srcHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        dst.x = hip_pack(f);

        fx += scaleInfo.x;
        frac = fx - floorf(fx);
        f.x = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floorf(fx), sy, srcWidth, srcHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        fx += scaleInfo.x;
        frac = fx - floorf(fx);
        f.y = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floorf(fx), sy, srcWidth, srcHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        fx += scaleInfo.x;
        frac = fx - floorf(fx);
        f.z = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floorf(fx), sy, srcWidth, srcHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        fx += scaleInfo.x;
        frac = fx - floorf(fx);
        f.w = hip_bilinear_sample_with_constant_border(pSrcImage, (int)floorf(fx), sy, srcWidth, srcHeight, srcImageStrideInBytes, 1.0f - frac, frac, fy0, fy1, borderValue);
        dst.y = hip_pack(f);

        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }

}
int HipExec_ScaleImage_U8_U8_Bilinear_Constant(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_uint8 borderValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_float32 xscale = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth);
    vx_float32 yscale = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight);
    vx_float32 xoffset = (vx_float32)((vx_float64)srcWidth / (vx_float64)dstWidth * 0.5 - 0.5);
    vx_float32 yoffset = (vx_float32)((vx_float64)srcHeight / (vx_float64)dstHeight * 0.5 - 0.5);

    hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Bilinear_Constant, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcWidth, srcHeight,
                        xscale, yscale, xoffset, yoffset, borderValue);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Area(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    int Nx, int Ny, float iSxSy) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint offset = srcImageStrideInBytes * (y * Ny) + (x * Nx);
    pSrcImage += offset;

    d_float8 f = {0.0f};
    for (uint iy = 0; iy < 2; iy++) {
        uint4 dw;
        dw = *((uint4 *)&pSrcImage[0]);
        f.data[0] += hip_unpack0(dw.x);
        f.data[0] += hip_unpack1(dw.x);
        f.data[1] += hip_unpack2(dw.x);
        f.data[1] += hip_unpack3(dw.x);
        f.data[2] += hip_unpack0(dw.y);
        f.data[2] += hip_unpack1(dw.y);
        f.data[3] += hip_unpack2(dw.y);
        f.data[3] += hip_unpack3(dw.y);
        f.data[4] += hip_unpack0(dw.z);
        f.data[4] += hip_unpack1(dw.z);
        f.data[5] += hip_unpack2(dw.z);
        f.data[5] += hip_unpack3(dw.z);
        f.data[6] += hip_unpack0(dw.w);
        f.data[6] += hip_unpack1(dw.w);
        f.data[7] += hip_unpack2(dw.w);
        f.data[7] += hip_unpack3(dw.w);
        pSrcImage += srcImageStrideInBytes;
    }

    uint2 dst;
    dst.x = hip_pack(make_float4(f.data[0], f.data[1], f.data[2], f.data[3]) * (float4)(iSxSy));
    dst.y = hip_pack(make_float4(f.data[4], f.data[5], f.data[6], f.data[7]) * (float4)(iSxSy));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Area_Sad(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    int Nx, int Ny, float iSxSy) {
    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint offset = srcImageStrideInBytes * (y * Ny) + (x * Nx);
    pSrcImage += offset;

    d_uint8 sum = {0};
    for (uint iy = 0; iy < 4; iy++) {
        uint4 dw;
        dw = *((uint4 *)&pSrcImage[0]);
        sum.data[0] = hip_sad(dw.x, 0u, sum.data[0]);
        sum.data[1] = hip_sad(dw.y, 0u, sum.data[1]);
        sum.data[2] = hip_sad(dw.z, 0u, sum.data[2]);
        sum.data[3] = hip_sad(dw.w, 0u, sum.data[3]);
        dw = *((uint4 *)&pSrcImage[16]);
        sum.data[4] = hip_sad(dw.x, 0u, sum.data[4]);
        sum.data[5] = hip_sad(dw.y, 0u, sum.data[5]);
        sum.data[6] = hip_sad(dw.z, 0u, sum.data[6]);
        sum.data[7] = hip_sad(dw.w, 0u, sum.data[7]);
        pSrcImage += srcImageStrideInBytes;
    }

    d_float8 f;
    uint2 dst;
    f.data[0] = (float)sum.data[0];
    f.data[1] = (float)sum.data[1];
    f.data[2] = (float)sum.data[2];
    f.data[3] = (float)sum.data[3];
    f.data[4] = (float)sum.data[4];
    f.data[5] = (float)sum.data[5];
    f.data[6] = (float)sum.data[6];
    f.data[7] = (float)sum.data[7];

    dst.x = hip_pack(make_float4(f.data[0], f.data[1], f.data[2], f.data[3]) * (float4)(iSxSy));
    dst.y = hip_pack(make_float4(f.data[4], f.data[5], f.data[6], f.data[7]) * (float4)(iSxSy));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleImage_U8_U8_Area_Bytealign(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    float SX, float SY, float factorc, float iSxSy) {
    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }
    uint dstIdx =  y * dstImageStrideInBytes + x;

    float X = (float)x * SX;
    float Y = (float)y * SY;
    float fX = hip_fract(X, &X);
    float fY = hip_fract(Y, &Y);
    uint offset = srcImageStrideInBytes * (int)Y + (int)X;
    uint align = offset & 3;
    offset -= align;
    pSrcImage += offset;
    d_float8 ftotal{0.0f};
    float Sy = SY;
    float Syf = 1.0f - fY;
    for (uint iy = 0; iy < 2; iy++) {
        uint4 dw;
        uint2 d2;
        d_float8 f {0.0f};
        float Xs = fX, factor, Xi, Xf;
        uint offset, align;
        Xf = hip_fract(Xs, &Xi);
        offset = (uint)Xi;
        align = offset & 3;
        offset -= align;
        Xs += SX;

        d2 = *((uint2 *)(&pSrcImage[offset]));
        dw.x = d2.x;
        dw.y = d2.y;

        dw.x = hip_bytealign(dw.y, dw.x, align);
        f.data[0] += hip_unpack0(dw.x) * (1.0f - Xf);
        factor = factorc + Xf;
        f.data[0] += hip_unpack1(dw.x) * hip_clamp(factor, 0.0f, 1.0f) + hip_unpack2(dw.x) * hip_clamp(factor - 1.0f, 0.0f, 1.0f);
        Xf = hip_fract(Xs, &Xi);
        offset = (uint)Xi;
        align = offset & 3;
        offset -= align;
        Xs += SX;

        d2 = *((uint2 *)(&pSrcImage[offset]));
        dw.x = d2.x;
        dw.y = d2.y;

        dw.x = hip_bytealign(dw.y, dw.x, align);
        f.data[1] += hip_unpack0(dw.x) * (1.0f - Xf);
        factor = factorc + Xf;
        f.data[1] += hip_unpack1(dw.x) * hip_clamp(factor, 0.0f, 1.0f) + hip_unpack2(dw.x) * hip_clamp(factor - 1.0f, 0.0f, 1.0f);
        Xf = hip_fract(Xs, &Xi);
        offset = (uint)Xi;
        align = offset & 3;
        offset -= align;
        Xs += SX;

        d2 = *((uint2 *)(&pSrcImage[offset]));
        dw.x = d2.x;
        dw.y = d2.y;

        dw.x = hip_bytealign(dw.y, dw.x, align);
        f.data[2] += hip_unpack0(dw.x) * (1.0f - Xf);
        factor = factorc + Xf;
        f.data[2] += hip_unpack1(dw.x) * hip_clamp(factor, 0.0f, 1.0f) + hip_unpack2(dw.x) * hip_clamp(factor - 1.0f, 0.0f, 1.0f);
        Xf = hip_fract(Xs, &Xi); offset = (uint)Xi;
        align = offset & 3;
        offset -= align;
        Xs += SX;

        d2 = *((uint2 *)(&pSrcImage[offset]));
        dw.x = d2.x;
        dw.y = d2.y;

        dw.x = hip_bytealign(dw.y, dw.x, align);
        f.data[3] += hip_unpack0(dw.x) * (1.0f - Xf);
        factor = factorc + Xf;
        f.data[3] += hip_unpack1(dw.x) * hip_clamp(factor, 0.0f, 1.0f) + hip_unpack2(dw.x) * hip_clamp(factor - 1.0f, 0.0f, 1.0f);
        Xf = hip_fract(Xs, &Xi);
        offset = (uint)Xi;
        align = offset & 3;
        offset -= align;
        Xs += SX;

        d2 = *((uint2 *)(&pSrcImage[offset]));
        dw.x = d2.x;
        dw.y = d2.y;

        dw.x = hip_bytealign(dw.y, dw.x, align);
        f.data[4] += hip_unpack0(dw.x) * (1.0f - Xf);
        factor = factorc + Xf;
        f.data[4] += hip_unpack1(dw.x) * hip_clamp(factor, 0.0f, 1.0f) + hip_unpack2(dw.x) * hip_clamp(factor - 1.0f, 0.0f, 1.0f);
        Xf = hip_fract(Xs, &Xi);
        offset = (uint)Xi;
        align = offset & 3;
        offset -= align;
        Xs += SX;

        d2 = *((uint2 *)(&pSrcImage[offset]));
        dw.x = d2.x;
        dw.y = d2.y;

        dw.x = hip_bytealign(dw.y, dw.x, align);
        f.data[5] += hip_unpack0(dw.x) * (1.0f - Xf);
        factor = factorc + Xf;
        f.data[5] += hip_unpack1(dw.x) * hip_clamp(factor, 0.0f, 1.0f) + hip_unpack2(dw.x) * hip_clamp(factor - 1.0f, 0.0f, 1.0f);
        Xf = hip_fract(Xs, &Xi);
        offset = (uint)Xi;
        align = offset & 3;
        offset -= align;
        Xs += SX;

        d2 = *((uint2 *)(&pSrcImage[offset]));
        dw.x = d2.x;
        dw.y = d2.y;

        dw.x = hip_bytealign(dw.y, dw.x, align);
        f.data[6] += hip_unpack0(dw.x) * (1.0f - Xf);
        factor = factorc + Xf;
        f.data[6] += hip_unpack1(dw.x) * hip_clamp(factor, 0.0f, 1.0f) + hip_unpack2(dw.x) * hip_clamp(factor - 1.0f, 0.0f, 1.0f);
        Xf = hip_fract(Xs, &Xi);
        offset = (uint)Xi;
        align = offset & 3;
        offset -= align;

        d2 = *((uint2 *)(&pSrcImage[offset]));
        dw.x = d2.x;
        dw.y = d2.y;

        dw.x = hip_bytealign(dw.y, dw.x, align);
        f.data[7] += hip_unpack0(dw.x) * (1.0f - Xf);
        factor = factorc + Xf;
        f.data[7] += hip_unpack1(dw.x) * hip_clamp(factor, 0.0f, 1.0f) + hip_unpack2(dw.x) * hip_clamp(factor - 1.0f, 0.0f, 1.0f);

        f.data[0] *= Syf;
        f.data[1] *= Syf;
        f.data[2] *= Syf;
        f.data[3] *= Syf;
        f.data[4] *= Syf;
        f.data[5] *= Syf;
        f.data[6] *= Syf;
        f.data[7] *= Syf;

        ftotal.data[0] += f.data[0];
        ftotal.data[1] += f.data[1];
        ftotal.data[2] += f.data[2];
        ftotal.data[3] += f.data[3];
        ftotal.data[4] += f.data[4];
        ftotal.data[5] += f.data[5];
        ftotal.data[6] += f.data[6];
        ftotal.data[7] += f.data[7];

        Sy -= Syf;
        Syf = hip_clamp(Sy, 0.0f, 1.0f);
        pSrcImage += srcImageStrideInBytes;
    }

    uint2 dst;
    dst.x = hip_pack(make_float4(ftotal.data[0], ftotal.data[1], ftotal.data[2], ftotal.data[3]) * (float4)(iSxSy));
    dst.y = hip_pack(make_float4(ftotal.data[4], ftotal.data[5], ftotal.data[6], ftotal.data[7]) * (float4)(iSxSy));
    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}

int HipExec_ScaleImage_U8_U8_Area(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    float Sx = (float)srcWidth / (float)dstWidth;
    float Sy = (float)srcHeight / (float)dstHeight;
    int Nx = (int)ceilf(Sx);
    int Ny = (int)ceilf(Sy);

    bool need_align = ((Sx * 2.0f) != floorf(Sx * 2.0f)) ? true : false;
    bool use_sad = (Nx % 4) ? false : true;
    float iSxSy = 1.0 / (double)(Sx * Sy);
    float factorc = Sx - (Nx - 1);

    if ((srcWidth % dstWidth) > 0 || (srcHeight % dstHeight) > 0) {
        use_sad = false;
    }

    if (use_sad) {
        hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Area_Sad, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        Nx, Ny, iSxSy);
    } else if (need_align) {
        hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Area_Bytealign, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        Sx, Sy, factorc, iSxSy);
    } else {
        hipLaunchKernelGGL(Hip_ScaleImage_U8_U8_Area, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        Nx, Ny, iSxSy);
    }

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxWarpAffine kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_WarpAffine_U8_U8_Nearest(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint srcImageBufferSize, d_affine_matrix_t *affineMatrix) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst = (uint2)0;
    float sx, sy;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, affineMatrix->m[1][0], affineMatrix->m[2][0]);
    sx = fmaf(dx, affineMatrix->m[0][0], sx);
    sy = fmaf(dy, affineMatrix->m[1][1], affineMatrix->m[2][1]);
    sy = fmaf(dx, affineMatrix->m[0][1], sy);

    uint srcIdx = hip_mad24(srcImageStrideInBytes, (uint)sy, (uint)sx);
    if (srcIdx < srcImageBufferSize)
        dst.x = pSrcImage[srcIdx];
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)sy, (uint)sx);
    if (srcIdx < srcImageBufferSize)
        dst.x |= pSrcImage[srcIdx] << 8;
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)sy, (uint)sx);
    if (srcIdx < srcImageBufferSize)
        dst.x |= pSrcImage[srcIdx] << 16;
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)sy, (uint)sx);
    if (srcIdx < srcImageBufferSize)
        dst.x |= pSrcImage[srcIdx] << 24;

    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];

    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)sy, (uint)sx);
    if (srcIdx < srcImageBufferSize)
        dst.y  = pSrcImage[srcIdx];
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)sy, (uint)sx);
    if (srcIdx < srcImageBufferSize)
        dst.y |= pSrcImage[srcIdx] << 8;
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)sy, (uint)sx);
    if (srcIdx < srcImageBufferSize)
        dst.y |= pSrcImage[srcIdx] << 16;
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)sy, (uint)sx);
    if (srcIdx < srcImageBufferSize)
        dst.y |= pSrcImage[srcIdx] << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpAffine_U8_U8_Nearest(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes, vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, vx_uint32 srcImageBufferSize,
    ago_affine_matrix_t *affineMatrix) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpAffine_U8_U8_Nearest, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                        (d_affine_matrix_t *) affineMatrix);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpAffine_U8_U8_Nearest_Constant(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    d_affine_matrix_t *affineMatrix, uint borderValue, vx_rectangle_t rect_valid) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float sx, sy;
    uint mask, v;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, affineMatrix->m[1][0], affineMatrix->m[2][0]);
    sx = fmaf(dx, affineMatrix->m[0][0], sx);
    sy = fmaf(dy, affineMatrix->m[1][1], affineMatrix->m[2][1]);
    sy = fmaf(dx, affineMatrix->m[0][1], sy);

	uint vl = rect_valid.start_x;
	uint vr = rect_valid.end_x;
	uint vt = rect_valid.start_y;
	uint vb = rect_valid.end_y;

    x = (uint)(int)sx;
    y = (uint)(int)sy;
    dstWidth -= vl;
    dstHeight -= vt;

    mask = ((int)((x - vl) | (vr - 1 - x) | (y - vt) | (vb - 1 - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.x = v;

    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    x = (uint)(int)sx;
    y = (uint)(int)sy;
    mask = ((int)((x - vl) | (vr - 1 - x) | (y - vt) | (vb - 1 - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.x |= v << 8;

    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    x = (uint)(int)sx;
    y = (uint)(int)sy;
    mask = ((int)((x - vl) | (vr - 1 - x) | (y - vt) | (vb - 1 - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.x |= v << 16;

    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    x = (uint)(int)sx;
    y = (uint)(int)sy;
    mask = ((int)((x - vl) | (vr - 1 - x) | (y - vt) | (vb - 1 - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.x |= v << 24;

    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    x = (uint)(int)sx;
    y = (uint)(int)sy;

    mask = ((int)((x - vl) | (vr - 1 - x) | (y - vt) | (vb - 1 - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.y = v;

    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    x = (uint)(int)sx;
    y = (uint)(int)sy;
    mask = ((int)((x - vl) | (vr - 1 - x) | (y - vt) | (vb - 1 - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.y |= v << 8;

    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    x = (uint)(int)sx;
    y = (uint)(int)sy;
    mask = ((int)((x - vl) | (vr - 1 - x) | (y - vt) | (vb - 1 - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.y |= v << 16;

    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    x = (uint)(int)sx;
    y = (uint)(int)sy;
    mask = ((int)((x - vl) | (vr - 1 - x) | (y - vt) | (vb - 1 - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.y |= v << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpAffine_U8_U8_Nearest_Constant(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_affine_matrix_t *affineMatrix, vx_uint8 borderValue, vx_rectangle_t rect_valid) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpAffine_U8_U8_Nearest_Constant, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (d_affine_matrix_t *) affineMatrix, (uint) borderValue, rect_valid);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpAffine_U8_U8_Bilinear(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint srcImageBufferSize, d_affine_matrix_t *affineMatrix) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float4 f;
    float sx, sy;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, affineMatrix->m[1][0], affineMatrix->m[2][0]);
    sx = fmaf(dx, affineMatrix->m[0][0], sx);
    sy = fmaf(dy, affineMatrix->m[1][1], affineMatrix->m[2][1]);
    sy = fmaf(dx, affineMatrix->m[0][1], sy);

    f.x = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx, sy);
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    f.y = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx, sy);
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    f.z = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx, sy);
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    f.w = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx, sy);
    dst.x = hip_pack(f);

    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];

    f.x = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx, sy);
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    f.y = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx, sy);
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    f.z = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx, sy);
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    f.w = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx, sy);
    dst.y = hip_pack(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpAffine_U8_U8_Bilinear(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint32 srcImageBufferSize, ago_affine_matrix_t *affineMatrix) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpAffine_U8_U8_Bilinear, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                        (d_affine_matrix_t *) affineMatrix);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpAffine_U8_U8_Bilinear_Constant(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    d_affine_matrix_t *affineMatrix, uint borderValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float4 f;
    float sx, sy;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, affineMatrix->m[1][0], affineMatrix->m[2][0]);
    sx = fmaf(dx, affineMatrix->m[0][0], sx);
    sy = fmaf(dy, affineMatrix->m[1][1], affineMatrix->m[2][1]);
    sy = fmaf(dx, affineMatrix->m[0][1], sy);

    f.x = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    f.y = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    f.z = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    f.w = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    dst.x = hip_pack(f);

    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];

    f.x = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    f.y = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    f.z = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    sx += affineMatrix->m[0][0];
    sy += affineMatrix->m[0][1];
    f.w = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, dstWidth, dstHeight, sx, sy, borderValue);
    dst.y = hip_pack(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpAffine_U8_U8_Bilinear_Constant(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_affine_matrix_t *affineMatrix, vx_uint8 borderValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpAffine_U8_U8_Bilinear_Constant, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (d_affine_matrix_t *) affineMatrix, (uint) borderValue);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxWarpPerspective kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_WarpPerspective_U8_U8_Nearest(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint srcImageBufferSize, d_perspective_matrix_t *perspectiveMatrix) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst = (uint2)0;
    float sx, sy, sz, isz;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, perspectiveMatrix->m[1][0], perspectiveMatrix->m[2][0]);
    sx = fmaf(dx, perspectiveMatrix->m[0][0], sx);
    sy = fmaf(dy, perspectiveMatrix->m[1][1], perspectiveMatrix->m[2][1]);
    sy = fmaf(dx, perspectiveMatrix->m[0][1], sy);
    sz = fmaf(dy, perspectiveMatrix->m[1][2], perspectiveMatrix->m[2][2]);
    sz = fmaf(dx, perspectiveMatrix->m[0][2], sz);

    isz = 1.0f / sz;

    uint srcIdx = hip_mad24(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz));
    if (srcIdx < srcImageBufferSize)
        dst.x = pSrcImage[srcIdx];
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz));
    if (srcIdx < srcImageBufferSize)
        dst.x |= pSrcImage[srcIdx] << 8;
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz));
    if (srcIdx < srcImageBufferSize)
        dst.x |= pSrcImage[srcIdx] << 16;
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz));
    if (srcIdx < srcImageBufferSize)
        dst.x |= pSrcImage[srcIdx] << 24;
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz));
    if (srcIdx < srcImageBufferSize)
        dst.y  = pSrcImage[srcIdx];
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz));
    if (srcIdx < srcImageBufferSize)
        dst.y |= pSrcImage[srcIdx] << 8;
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz));
    if (srcIdx < srcImageBufferSize)
        dst.y |= pSrcImage[srcIdx] << 16;
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    srcIdx = hip_mad24(srcImageStrideInBytes, (uint)(sy * isz), (uint)(sx * isz));
    if (srcIdx < srcImageBufferSize)
        dst.y |= pSrcImage[srcIdx] << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpPerspective_U8_U8_Nearest(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, vx_uint32 srcImageBufferSize,
    ago_perspective_matrix_t *perspectiveMatrix) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpPerspective_U8_U8_Nearest, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                        (d_perspective_matrix_t *) perspectiveMatrix);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpPerspective_U8_U8_Nearest_Constant(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    uint srcWidth, uint srcHeight, const uchar *pSrcImage, uint srcImageStrideInBytes,
    d_perspective_matrix_t *perspectiveMatrix, uint borderValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    srcWidth -= 2;
    srcHeight -= 2;
    uint2 dst;
    float sx, sy, sz, isz;
    uint mask, v;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, perspectiveMatrix->m[1][0], perspectiveMatrix->m[2][0]);
    sx = fmaf(dx, perspectiveMatrix->m[0][0], sx);
    sy = fmaf(dy, perspectiveMatrix->m[1][1], perspectiveMatrix->m[2][1]);
    sy = fmaf(dx, perspectiveMatrix->m[0][1], sy);
    sz = fmaf(dy, perspectiveMatrix->m[1][2], perspectiveMatrix->m[2][2]);
    sz = fmaf(dx, perspectiveMatrix->m[0][2], sz);

    isz = 1.0f / sz;

    x = (uint)(int)(sx * isz);
    y = (uint)(int)(sy * isz);

    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.x = v;

    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    x = (uint)(int)(sx * isz);
    y = (uint)(int)(sy * isz);
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.x |= (v << 8);

    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    x = (uint)(int)(sx * isz);
    y = (uint)(int)(sy * isz);
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.x |= (v << 16);

    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    x = (uint)(int)(sx * isz);
    y = (uint)(int)(sy * isz);
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.x |= (v << 24);

    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    x = (uint)(int)(sx * isz);
    y = (uint)(int)(sy * isz);

    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.y = v;

    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    x = (uint)(int)(sx * isz);
    y = (uint)(int)(sy * isz);
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.y |= (v << 8);

    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    x = (uint)(int)(sx * isz);
    y = (uint)(int)(sy * isz);
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.y |= (v << 16);

    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    x = (uint)(int)(sx * isz);
    y = (uint)(int)(sy * isz);
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    v = pSrcImage[hip_mad24(srcImageStrideInBytes, y, x)];
    v = HIPSELECT(borderValue, v, mask);
    dst.y |= (v << 24);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpPerspective_U8_U8_Nearest_Constant(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_perspective_matrix_t *perspectiveMatrix, vx_uint8 borderValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpPerspective_U8_U8_Nearest_Constant, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        srcWidth, srcHeight, (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (d_perspective_matrix_t *) perspectiveMatrix, (uint) borderValue);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpPerspective_U8_U8_Bilinear(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint srcImageBufferSize, d_perspective_matrix_t *perspectiveMatrix) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float4 f;
    float sx, sy, sz, isz;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, perspectiveMatrix->m[1][0], perspectiveMatrix->m[2][0]);
    sx = fmaf(dx, perspectiveMatrix->m[0][0], sx);
    sy = fmaf(dy, perspectiveMatrix->m[1][1], perspectiveMatrix->m[2][1]);
    sy = fmaf(dx, perspectiveMatrix->m[0][1], sy);
    sz = fmaf(dy, perspectiveMatrix->m[1][2], perspectiveMatrix->m[2][2]);
    sz = fmaf(dx, perspectiveMatrix->m[0][2], sz);

    isz = 1.0f / sz;

    f.x = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx * isz, sy * isz);
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    f.y = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx * isz, sy * isz);
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    f.z = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx * isz, sy * isz);
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    f.w = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx * isz, sy * isz);
    dst.x = hip_pack(f);

    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;

    f.x = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx * isz, sy * isz);
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    f.y = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx * isz, sy * isz);
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    f.z = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx * isz, sy * isz);
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    f.w = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, sx * isz, sy * isz);
    dst.y = hip_pack(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpPerspective_U8_U8_Bilinear(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint32 srcImageBufferSize, ago_perspective_matrix_t *perspectiveMatrix) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpPerspective_U8_U8_Bilinear, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                        (d_perspective_matrix_t *) perspectiveMatrix);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_WarpPerspective_U8_U8_Bilinear_Constant(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    uint srcWidth, uint srcHeight, const uchar *pSrcImage, uint srcImageStrideInBytes,
    d_perspective_matrix_t *perspectiveMatrix, uint borderValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 dst;
    float4 f;
    float sx, sy, sz, isz;
    float dx = (float)x;
    float dy = (float)y;
    sx = fmaf(dy, perspectiveMatrix->m[1][0], perspectiveMatrix->m[2][0]);
    sx = fmaf(dx, perspectiveMatrix->m[0][0], sx);
    sy = fmaf(dy, perspectiveMatrix->m[1][1], perspectiveMatrix->m[2][1]);
    sy = fmaf(dx, perspectiveMatrix->m[0][1], sy);
    sz = fmaf(dy, perspectiveMatrix->m[1][2], perspectiveMatrix->m[2][2]);
    sz = fmaf(dx, perspectiveMatrix->m[0][2], sz);

    isz = 1.0f / sz;

    f.x = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, sx * isz, sy * isz, borderValue);
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    f.y = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, sx * isz, sy * isz, borderValue);
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    f.z = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, sx * isz, sy * isz, borderValue);
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    f.w = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, sx * isz, sy * isz, borderValue);
    dst.x = hip_pack(f);

    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;

    f.x = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, sx * isz, sy * isz, borderValue);
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    f.y = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, sx * isz, sy * isz, borderValue);
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    f.z = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, sx * isz, sy * isz, borderValue);
    sx += perspectiveMatrix->m[0][0];
    sy += perspectiveMatrix->m[0][1];
    sz += perspectiveMatrix->m[0][2];
    isz = 1.0f / sz;
    f.w = hip_bilinear_sample_FXY_constant(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, sx * isz, sy * isz, borderValue);
    dst.y = hip_pack(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_WarpPerspective_U8_U8_Bilinear_Constant(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_perspective_matrix_t *perspectiveMatrixLoc, vx_uint8 borderValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_WarpPerspective_U8_U8_Bilinear_Constant, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        srcWidth, srcHeight, (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (d_perspective_matrix_t *) perspectiveMatrixLoc, (uint) borderValue);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxRemap kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Remap_U8_U8_Nearest(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint srcImageBufferSize, uchar *remap_, uint remapStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    int *remap = (int *) (remap_ + y * remapStrideInBytes + (x << 2));
    uint2 dst = (uint2)0;
    int map;
    uint v = 0;

    map = remap[0];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    uint srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    dst.x = v;

    map = remap[1];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    dst.x |= v << 8;

    map = remap[2];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    dst.x |= v << 16;

    map = remap[3];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    dst.x |= v << 24;

    map = remap[4];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    dst.y  = v;

    map = remap[5];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    dst.y |= v << 8;

    map = remap[6];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    dst.y |= v << 16;

    map = remap[7];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    dst.y |= v << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Remap_U8_U8_Nearest(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes, vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, vx_uint32 srcImageBufferSize,
    ago_coord2d_ushort_t *remap, vx_uint32 remapStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Remap_U8_U8_Nearest, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                        (uchar *) remap, remapStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Remap_U8_U8_Nearest_Constant(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    uint srcWidth, uint srcHeight, const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint srcImageBufferSize, uchar *remap_, uint remapStrideInBytes, uint borderValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    int *remap = (int *) (remap_ + y * remapStrideInBytes + (x << 2));
    uint2 dst = (uint2)0;
    int map;
    uint mask, v;
    srcWidth -= 1;
    srcHeight -= 1;

    map = remap[0];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    uint srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    v = HIPSELECT(borderValue, v, mask);
    dst.x  = v;

    map = remap[1];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    v = HIPSELECT(borderValue, v, mask);
    dst.x |= v << 8;

    map = remap[2];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    v = HIPSELECT(borderValue, v, mask);
    dst.x |= v << 16;

    map = remap[3];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    v = HIPSELECT(borderValue, v, mask);
    dst.x |= v << 24;

    map = remap[4];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    v = HIPSELECT(borderValue, v, mask);
    dst.y  = v;

    map = remap[5];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    v = HIPSELECT(borderValue, v, mask);
    dst.y |= v << 8;

    map = remap[6];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    v = HIPSELECT(borderValue, v, mask);
    dst.y |= v << 16;

    map = remap[7];
    x = ((map & 0xffff) + 4) >> 3;
    y = (map + 0x00040000) >> 19;
    mask = ((int)(x | (srcWidth - x) | y | (srcHeight - y))) >> 31;
    mask = ~mask;
    x &= mask;
    y &= mask;
    srcIdx = hip_mad24(srcImageStrideInBytes, y, x);
    if (srcIdx < srcImageBufferSize)
        v = pSrcImage[srcIdx];
    v = HIPSELECT(borderValue, v, mask);
    dst.y |= v << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Remap_U8_U8_Nearest_Constant(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes, vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, vx_uint32 srcImageBufferSize,
    ago_coord2d_ushort_t *remap, vx_uint32 remapStrideInBytes, const vx_uint8 borderValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Remap_U8_U8_Nearest_Constant, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        srcWidth, srcHeight, (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                        (uchar *) remap, remapStrideInBytes, (uint) borderValue);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Remap_U8_U8_Bilinear(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint srcImageBufferSize, uchar *remap_, uint remapStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    int *remap = (int *) (remap_ + y * remapStrideInBytes + (x << 2));
    uint2 dst;
    float4 f;
    int map;

    map = remap[0];
    f.x = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);
    map = remap[1];
    f.y = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);
    map = remap[2];
    f.z = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);
    map = remap[3];
    f.w = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);
    dst.x = hip_pack(f);

    map = remap[4];
    f.x = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);
    map = remap[5];
    f.y = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);
    map = remap[6];
    f.z = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);
    map = remap[7];
    f.w = hip_bilinear_sample_FXY(pSrcImage, srcImageBufferSize, srcImageStrideInBytes, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);
    dst.y = hip_pack(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Remap_U8_U8_Bilinear(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint32 srcImageBufferSize, ago_coord2d_ushort_t *remap, vx_uint32 remapStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Remap_U8_U8_Bilinear, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageBufferSize,
                        (uchar *) remap, remapStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_Remap_U8_U8_Bilinear_Constant(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    uint srcWidth, uint srcHeight, const uchar *pSrcImage, uint srcImageStrideInBytes,
    uchar *remap_, uint remapStrideInBytes, uint borderValue) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint dstIdx =  y * dstImageStrideInBytes + x;

    int *remap = (int *) (remap_ + y * remapStrideInBytes + (x << 2));
    uint2 dst;
    float4 f;
    int map;

    map = remap[0];
    f.x = hip_bilinear_sample_FXY_constant_for_remap(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);
    map = remap[1];
    f.y = hip_bilinear_sample_FXY_constant_for_remap(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);
    map = remap[2];
    f.z = hip_bilinear_sample_FXY_constant_for_remap(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);
    map = remap[3];
    f.w = hip_bilinear_sample_FXY_constant_for_remap(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);
    dst.x = hip_pack(f);

    map = remap[4];
    f.x = hip_bilinear_sample_FXY_constant_for_remap(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);
    map = remap[5];
    f.y = hip_bilinear_sample_FXY_constant_for_remap(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);
    map = remap[6];
    f.z = hip_bilinear_sample_FXY_constant_for_remap(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);
    map = remap[7];
    f.w = hip_bilinear_sample_FXY_constant_for_remap(pSrcImage, srcImageStrideInBytes, srcWidth, srcHeight, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);
    dst.y = hip_pack(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Remap_U8_U8_Bilinear_Constant(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    vx_uint32 srcWidth, vx_uint32 srcHeight,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    ago_coord2d_ushort_t *remap, vx_uint32 remapStrideInBytes, const vx_uint8 borderValue) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Remap_U8_U8_Bilinear_Constant, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        srcWidth, srcHeight, (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        (uchar *) remap, remapStrideInBytes, (uint) borderValue);

    return VX_SUCCESS;
}