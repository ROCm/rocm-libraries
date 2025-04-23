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
// VxLut kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_Lut_U8_U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uchar *lut) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uint2 src = *((uint2 *)(&pSrcImage[srcIdx]));
    uint2 dst;

    float4 f;
    f.x = (float) lut[(int)( src.x        & 255)];
    f.y = (float) lut[(int)((src.x >>  8) & 255)];
    f.z = (float) lut[(int)((src.x >> 16) & 255)];
    f.w = (float) lut[(int)( src.x >> 24       )];
    dst.x = hip_pack(f);
    f.x = (float) lut[(int)( src.y        & 255)];
    f.y = (float) lut[(int)((src.y >>  8) & 255)];
    f.z = (float) lut[(int)((src.y >> 16) & 255)];
    f.w = (float) lut[(int)( src.y >> 24       )];
    dst.y = hip_pack(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_Lut_U8_U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint8 *lut) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_Lut_U8_U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                       lut);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxChannelCopy kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ChannelCopy_U8_U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + x;
    uint dstIdx = y * dstImageStrideInBytes + x;

    *((uint2 *)(&pDstImage[dstIdx])) = *((uint2 *)(&pSrcImage[srcIdx]));
}
int HipExec_ChannelCopy_U8_U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelCopy_U8_U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCopy_U8_U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + x;

    uchar src = pSrcImage[srcIdx];
    uint2 dst;

    hip_convert_U8_U1(&dst, src);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ChannelCopy_U8_U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelCopy_U8_U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCopy_U1_U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    uint2 src = *((uint2 *)(&pSrcImage[srcIdx]));
    uchar dst;

    hip_convert_U1_U8(&dst, src);

    pDstImage[dstIdx] = dst;
}
int HipExec_ChannelCopy_U1_U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelCopy_U1_U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCopy_U1_U1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + (x >> 3);
    uint dstIdx =  y * dstImageStrideInBytes + (x >> 3);

    pDstImage[dstIdx] = pSrcImage[srcIdx];
}
int HipExec_ChannelCopy_U1_U1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelCopy_U1_U1, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                       dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                       (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxColorDepth kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ColorDepth_U8_S16_Wrap(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    const int shift) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx =  y * srcImageStrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    int4 src = *((int4 *)(&pSrcImage[srcIdx]));
    uint2 dst;

    int sr = shift;
    sr += 16;
    dst.x  = ((src.x   << 16) >> sr) & 0xff;
    dst.x |= ((src.x          >> sr) & 0xff) <<  8;
    dst.x |= (((src.y  << 16) >> sr) & 0xff) << 16;
    dst.x |= ((src.y          >> sr) & 0xff) << 24;
    dst.y  = ((src.z   << 16) >> sr) & 0xff;
    dst.y |= ((src.z          >> sr) & 0xff) <<  8;
    dst.y |= (((src.w  << 16) >> sr) & 0xff) << 16;
    dst.y |= ((src.w          >> sr) & 0xff) << 24;

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ColorDepth_U8_S16_Wrap(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int32 shift) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorDepth_U8_S16_Wrap, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, shift);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorDepth_U8_S16_Sat(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    const int shift) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx =  y * srcImageStrideInBytes + x + x;
    uint dstIdx =  y * dstImageStrideInBytes + x;

    int4 src = *((int4 *)(&pSrcImage[srcIdx]));
    uint2 dst;

    int sr = shift;
    sr += 16;
    float4 f;
    f.x = (float)((src.x << 16) >> sr);
    f.y = (float)( src.x        >> sr);
    f.z = (float)((src.y << 16) >> sr);
    f.w = (float)( src.y        >> sr);
    dst.x = hip_pack(f);
    f.x = (float)((src.z << 16) >> sr);
    f.y = (float)( src.z        >> sr);
    f.z = (float)((src.w << 16) >> sr);
    f.w = (float)( src.w        >> sr);
    dst.y = hip_pack(f);

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ColorDepth_U8_S16_Sat(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int32 shift) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorDepth_U8_S16_Sat, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, shift);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorDepth_S16_U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    const int shift) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx =  y * srcImageStrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src = *((uint2 *)(&pSrcImage[srcIdx]));
    int4 dst;

    dst.x  =  (src.x & 0x000000ff) <<       shift ;
    dst.x |=  (src.x & 0x0000ff00) << ( 8 + shift);
    dst.y  =  (src.x & 0x00ff0000) >> (16 - shift);
    dst.y |=  (src.x & 0xff000000) >> ( 8 - shift);
    dst.z  =  (src.y & 0x000000ff) <<       shift ;
    dst.z |=  (src.y & 0x0000ff00) << ( 8 + shift);
    dst.w  =  (src.y & 0x00ff0000) >> (16 - shift);
    dst.w |=  (src.y & 0xff000000) >> ( 8 - shift);

    *((int4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ColorDepth_S16_U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    const vx_int32 shift) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorDepth_S16_U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, shift);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxChannelExtract kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U16_Pos0(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    unsigned int srcIdx = y * srcImageStrideInBytes + x + x;
    unsigned int dstIdx = y * dstImageStrideInBytes + x;

    uint4 src = *((uint4 *)(&pSrcImage[srcIdx]));
    uint2 dst;

    dst.x = hip_pack(make_float4(hip_unpack0(src.x), hip_unpack2(src.x), hip_unpack0(src.y), hip_unpack2(src.y)));
    dst.y = hip_pack(make_float4(hip_unpack0(src.z), hip_unpack2(src.z), hip_unpack0(src.w), hip_unpack2(src.w)));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ChannelExtract_U8_U16_Pos0(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U16_Pos0, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U16_Pos1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    unsigned int srcIdx = y * srcImageStrideInBytes + x + x;
    unsigned int dstIdx = y * dstImageStrideInBytes + x;

    uint4 src = *((uint4 *)(&pSrcImage[srcIdx]));
    uint2 dst;

    dst.x = hip_pack(make_float4(hip_unpack1(src.x), hip_unpack3(src.x), hip_unpack1(src.y), hip_unpack3(src.y)));
    dst.y = hip_pack(make_float4(hip_unpack1(src.z), hip_unpack3(src.z), hip_unpack1(src.w), hip_unpack3(src.w)));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ChannelExtract_U8_U16_Pos1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16, localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U16_Pos1, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U24_Pos0(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x * 3);
    uint dstIdx  = y * dstImageStrideInBytes + x;

    d_uint6 src1 = *((d_uint6 *)(&pSrcImage1[src1Idx]));
    uint2 dst;

    dst.x = hip_pack(make_float4(hip_unpack0(src1.data[0]), hip_unpack3(src1.data[0]), hip_unpack2(src1.data[1]), hip_unpack1(src1.data[2])));
    dst.y = hip_pack(make_float4(hip_unpack0(src1.data[3]), hip_unpack3(src1.data[3]), hip_unpack2(src1.data[4]), hip_unpack1(src1.data[5])));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ChannelExtract_U8_U24_Pos0(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U24_Pos0, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U24_Pos1(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x * 3);
    uint dstIdx  = y * dstImageStrideInBytes + x;

    d_uint6 src1 = *((d_uint6 *)(&pSrcImage1[src1Idx]));
    uint2 dst;

    dst.x = hip_pack(make_float4(hip_unpack1(src1.data[0]), hip_unpack0(src1.data[1]), hip_unpack3(src1.data[1]), hip_unpack2(src1.data[2])));
    dst.y = hip_pack(make_float4(hip_unpack1(src1.data[3]), hip_unpack0(src1.data[4]), hip_unpack3(src1.data[4]), hip_unpack2(src1.data[5])));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ChannelExtract_U8_U24_Pos1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U24_Pos1, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U24_Pos2(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x * 3);
    uint dstIdx  = y * dstImageStrideInBytes + x;

    d_uint6 src1 = *((d_uint6 *)(&pSrcImage1[src1Idx]));
    uint2 dst;

    dst.x = hip_pack(make_float4(hip_unpack2(src1.data[0]), hip_unpack1(src1.data[1]), hip_unpack0(src1.data[2]), hip_unpack3(src1.data[2])));
    dst.y = hip_pack(make_float4(hip_unpack2(src1.data[3]), hip_unpack1(src1.data[4]), hip_unpack0(src1.data[5]), hip_unpack3(src1.data[5])));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ChannelExtract_U8_U24_Pos2(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8_U24_Pos2, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos0_RGBX(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x << 2);
    uint dstIdx  = y * dstImageStrideInBytes + x;

    d_uint8 src1 = *((d_uint8 *)(&pSrcImage1[src1Idx]));
    uint2 dst;

    dst.x = hip_pack(make_float4(hip_unpack0(src1.data[0]), hip_unpack0(src1.data[1]), hip_unpack0(src1.data[2]), hip_unpack0(src1.data[3])));
    dst.y = hip_pack(make_float4(hip_unpack0(src1.data[4]), hip_unpack0(src1.data[5]), hip_unpack0(src1.data[6]), hip_unpack0(src1.data[7])));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos0_UYVY(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp && y < dstHeight)) {
        uint srcIdx = y * srcImageStrideInBytes + (x << 4);
        uint dstIdx = y * dstImageStrideInBytes + (x << 2);

        d_uint8 src = *((d_uint8 *)(&pSrcImage[srcIdx]));
        uint2 dst;

        dst.x = hip_pack(make_float4(hip_unpack0(src.data[0]), hip_unpack0(src.data[1]), hip_unpack0(src.data[2]), hip_unpack0(src.data[3])));
        dst.y = hip_pack(make_float4(hip_unpack0(src.data[4]), hip_unpack0(src.data[5]), hip_unpack0(src.data[6]), hip_unpack0(src.data[7])));

        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_ChannelExtract_U8_U32_Pos0(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_df_image srcType) {

    if (srcType == VX_DF_IMAGE_RGBX) {
        int localThreads_x = 16;
        int localThreads_y = 16;
        int globalThreads_x = (dstWidth + 7) >> 3;
        int globalThreads_y = dstHeight;

        hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos0_RGBX, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                            dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);
    }
    else if (srcType == VX_DF_IMAGE_UYVY) {
        int localThreads_x = 16;
        int localThreads_y = 4;
        int globalThreads_x = (dstWidth + 3) >> 2;
        int globalThreads_y = dstHeight;

        vx_uint32 dstWidthComp = (dstWidth + 3) / 4;

        hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos0_UYVY, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                            dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage1, srcImage1StrideInBytes,
                            dstWidthComp);
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos1_RGBX(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x << 2);
    uint dstIdx  = y * dstImageStrideInBytes + x;

    d_uint8 src1 = *((d_uint8 *)(&pSrcImage1[src1Idx]));
    uint2 dst;

    dst.x = hip_pack(make_float4(hip_unpack1(src1.data[0]), hip_unpack1(src1.data[1]), hip_unpack1(src1.data[2]), hip_unpack1(src1.data[3])));
    dst.y = hip_pack(make_float4(hip_unpack1(src1.data[4]), hip_unpack1(src1.data[5]), hip_unpack1(src1.data[6]), hip_unpack1(src1.data[7])));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos1_YUYV(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp && y < dstHeight)) {
        uint srcIdx = y * srcImageStrideInBytes + (x << 4);
        uint dstIdx = y * dstImageStrideInBytes + (x << 2);

        d_uint8 src = *((d_uint8 *)(&pSrcImage[srcIdx]));
        uint2 dst;

        dst.x = hip_pack(make_float4(hip_unpack1(src.data[0]), hip_unpack1(src.data[1]), hip_unpack1(src.data[2]), hip_unpack1(src.data[3])));
        dst.y = hip_pack(make_float4(hip_unpack1(src.data[4]), hip_unpack1(src.data[5]), hip_unpack1(src.data[6]), hip_unpack1(src.data[7])));

        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_ChannelExtract_U8_U32_Pos1(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_df_image srcType) {

    if (srcType == VX_DF_IMAGE_RGBX) {
        int localThreads_x = 16;
        int localThreads_y = 16;
        int globalThreads_x = (dstWidth + 7) >> 3;
        int globalThreads_y = dstHeight;

        hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos1_RGBX, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                            dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);
    }
    else if (srcType == VX_DF_IMAGE_YUYV) {
        int localThreads_x = 16;
        int localThreads_y = 4;
        int globalThreads_x = (dstWidth + 3) >> 2;
        int globalThreads_y = dstHeight;

        vx_uint32 dstWidthComp = (dstWidth + 3) / 4;

        hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos1_YUYV, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                            dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage1, srcImage1StrideInBytes,
                            dstWidthComp);
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos2_RGBX(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x << 2);
    uint dstIdx  = y * dstImageStrideInBytes + x;

    d_uint8 src1 = *((d_uint8 *)(&pSrcImage1[src1Idx]));
    uint2 dst;

    dst.x = hip_pack(make_float4(hip_unpack2(src1.data[0]), hip_unpack2(src1.data[1]), hip_unpack2(src1.data[2]), hip_unpack2(src1.data[3])));
    dst.y = hip_pack(make_float4(hip_unpack2(src1.data[4]), hip_unpack2(src1.data[5]), hip_unpack2(src1.data[6]), hip_unpack2(src1.data[7])));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos2_UYVY(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp && y < dstHeight)) {
        uint srcIdx = y * srcImageStrideInBytes + (x << 4);
        uint dstIdx = y * dstImageStrideInBytes + (x << 2);

        d_uint8 src = *((d_uint8 *)(&pSrcImage[srcIdx]));
        uint2 dst;

        dst.x = hip_pack(make_float4(hip_unpack2(src.data[0]), hip_unpack2(src.data[1]), hip_unpack2(src.data[2]), hip_unpack2(src.data[3])));
        dst.y = hip_pack(make_float4(hip_unpack2(src.data[4]), hip_unpack2(src.data[5]), hip_unpack2(src.data[6]), hip_unpack2(src.data[7])));

        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_ChannelExtract_U8_U32_Pos2(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_df_image srcType) {

    if (srcType == VX_DF_IMAGE_RGBX) {
        int localThreads_x = 16;
        int localThreads_y = 16;
        int globalThreads_x = (dstWidth + 7) >> 3;
        int globalThreads_y = dstHeight;

        hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos2_RGBX, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                            dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);
    }
    else if (srcType == VX_DF_IMAGE_UYVY) {
        int localThreads_x = 16;
        int localThreads_y = 4;
        int globalThreads_x = (dstWidth + 3) >> 2;
        int globalThreads_y = dstHeight;

        vx_uint32 dstWidthComp = (dstWidth + 3) / 4;

        hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos2_UYVY, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                            dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage1, srcImage1StrideInBytes,
                            dstWidthComp);
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos3_RGBX(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x << 2);
    uint dstIdx  = y * dstImageStrideInBytes + x;

    d_uint8 src1 = *((d_uint8 *)(&pSrcImage1[src1Idx]));
    uint2 dst;

    dst.x = hip_pack(make_float4(hip_unpack3(src1.data[0]), hip_unpack3(src1.data[1]), hip_unpack3(src1.data[2]), hip_unpack3(src1.data[3])));
    dst.y = hip_pack(make_float4(hip_unpack3(src1.data[4]), hip_unpack3(src1.data[5]), hip_unpack3(src1.data[6]), hip_unpack3(src1.data[7])));

    *((uint2 *)(&pDstImage[dstIdx])) = dst;
}
__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8_U32_Pos3_YUYV(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp && y < dstHeight)) {
        uint srcIdx = y * srcImageStrideInBytes + (x << 4);
        uint dstIdx = y * dstImageStrideInBytes + (x << 2);

        d_uint8 src = *((d_uint8 *)(&pSrcImage[srcIdx]));
        uint2 dst;

        dst.x = hip_pack(make_float4(hip_unpack3(src.data[0]), hip_unpack3(src.data[1]), hip_unpack3(src.data[2]), hip_unpack3(src.data[3])));
        dst.y = hip_pack(make_float4(hip_unpack3(src.data[4]), hip_unpack3(src.data[5]), hip_unpack3(src.data[6]), hip_unpack3(src.data[7])));

        *((uint2 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_ChannelExtract_U8_U32_Pos3(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    vx_df_image srcType) {

    if (srcType == VX_DF_IMAGE_RGBX) {
        int localThreads_x = 16;
        int localThreads_y = 16;
        int globalThreads_x = (dstWidth + 7) >> 3;
        int globalThreads_y = dstHeight;

        hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos3_RGBX, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                            dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);
    }
    else if (srcType == VX_DF_IMAGE_YUYV) {
        int localThreads_x = 16;
        int localThreads_y = 4;
        int globalThreads_x = (dstWidth + 3) >> 2;
        int globalThreads_y = dstHeight;

        vx_uint32 dstWidthComp = (dstWidth + 3) / 4;

        hipLaunchKernelGGL(Hip_ChannelExtract_U8_U32_Pos3_YUYV, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                            dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                            (const uchar *)pHipSrcImage1, srcImage1StrideInBytes,
                            dstWidthComp);
    }

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8U8U8_U24(uint dstWidth, uint dstHeight,
    uchar *pDstImage1, uchar *pDstImage2, uchar *pDstImage3, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x * 3);
    uint dstIdx  = y * dstImageStrideInBytes + x;

    d_uint6 src1 = *((d_uint6 *)(&pSrcImage1[src1Idx]));
    uint2 dst1, dst2, dst3;

    dst1.x = hip_pack(make_float4(hip_unpack0(src1.data[0]), hip_unpack3(src1.data[0]), hip_unpack2(src1.data[1]), hip_unpack1(src1.data[2])));
    dst1.y = hip_pack(make_float4(hip_unpack0(src1.data[3]), hip_unpack3(src1.data[3]), hip_unpack2(src1.data[4]), hip_unpack1(src1.data[5])));
    dst2.x = hip_pack(make_float4(hip_unpack1(src1.data[0]), hip_unpack0(src1.data[1]), hip_unpack3(src1.data[1]), hip_unpack2(src1.data[2])));
    dst2.y = hip_pack(make_float4(hip_unpack1(src1.data[3]), hip_unpack0(src1.data[4]), hip_unpack3(src1.data[4]), hip_unpack2(src1.data[5])));
    dst3.x = hip_pack(make_float4(hip_unpack2(src1.data[0]), hip_unpack1(src1.data[1]), hip_unpack0(src1.data[2]), hip_unpack3(src1.data[2])));
    dst3.y = hip_pack(make_float4(hip_unpack2(src1.data[3]), hip_unpack1(src1.data[4]), hip_unpack0(src1.data[5]), hip_unpack3(src1.data[5])));

    *((uint2 *)(&pDstImage1[dstIdx])) = dst1;
    *((uint2 *)(&pDstImage2[dstIdx])) = dst2;
    *((uint2 *)(&pDstImage3[dstIdx])) = dst3;
}
int HipExec_ChannelExtract_U8U8U8_U24(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage1, vx_uint8 *pHipDstImage2, vx_uint8 *pHipDstImage3, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8U8U8_U24, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage1, (uchar *)pHipDstImage2, (uchar *)pHipDstImage3, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8U8U8_U32(uint dstWidth, uint dstHeight,
    uchar *pDstImage1, uchar *pDstImage2, uchar *pDstImage3, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x << 2);
    uint dstIdx  = y * dstImageStrideInBytes + x;

    d_uint8 src1 = *((d_uint8 *)(&pSrcImage1[src1Idx]));
    uint2 dst1, dst2, dst3;

    dst1.x = hip_pack(make_float4(hip_unpack0(src1.data[0]), hip_unpack0(src1.data[1]), hip_unpack0(src1.data[2]), hip_unpack0(src1.data[3])));
    dst1.y = hip_pack(make_float4(hip_unpack0(src1.data[4]), hip_unpack0(src1.data[5]), hip_unpack0(src1.data[6]), hip_unpack0(src1.data[7])));
    dst2.x = hip_pack(make_float4(hip_unpack1(src1.data[0]), hip_unpack1(src1.data[1]), hip_unpack1(src1.data[2]), hip_unpack1(src1.data[3])));
    dst2.y = hip_pack(make_float4(hip_unpack1(src1.data[4]), hip_unpack1(src1.data[5]), hip_unpack1(src1.data[6]), hip_unpack1(src1.data[7])));
    dst3.x = hip_pack(make_float4(hip_unpack2(src1.data[0]), hip_unpack2(src1.data[1]), hip_unpack2(src1.data[2]), hip_unpack2(src1.data[3])));
    dst3.y = hip_pack(make_float4(hip_unpack2(src1.data[4]), hip_unpack2(src1.data[5]), hip_unpack2(src1.data[6]), hip_unpack2(src1.data[7])));

    *((uint2 *)(&pDstImage1[dstIdx])) = dst1;
    *((uint2 *)(&pDstImage2[dstIdx])) = dst2;
    *((uint2 *)(&pDstImage3[dstIdx])) = dst3;
}
int HipExec_ChannelExtract_U8U8U8_U32(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage1, vx_uint8 *pHipDstImage2, vx_uint8 *pHipDstImage3, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8U8U8_U32, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage1, (uchar *)pHipDstImage2, (uchar *)pHipDstImage3, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelExtract_U8U8U8U8_U32(uint dstWidth, uint dstHeight,
    uchar *pDstImage1, uchar *pDstImage2, uchar *pDstImage3, uchar *pDstImage4, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + (x << 2);
    uint dstIdx  = y * dstImageStrideInBytes + x;

    d_uint8 src1 = *((d_uint8 *)(&pSrcImage1[src1Idx]));
    uint2 dst1, dst2, dst3, dst4;

    dst1.x = hip_pack(make_float4(hip_unpack0(src1.data[0]), hip_unpack0(src1.data[1]), hip_unpack0(src1.data[2]), hip_unpack0(src1.data[3])));
    dst1.y = hip_pack(make_float4(hip_unpack0(src1.data[4]), hip_unpack0(src1.data[5]), hip_unpack0(src1.data[6]), hip_unpack0(src1.data[7])));
    dst2.x = hip_pack(make_float4(hip_unpack1(src1.data[0]), hip_unpack1(src1.data[1]), hip_unpack1(src1.data[2]), hip_unpack1(src1.data[3])));
    dst2.y = hip_pack(make_float4(hip_unpack1(src1.data[4]), hip_unpack1(src1.data[5]), hip_unpack1(src1.data[6]), hip_unpack1(src1.data[7])));
    dst3.x = hip_pack(make_float4(hip_unpack2(src1.data[0]), hip_unpack2(src1.data[1]), hip_unpack2(src1.data[2]), hip_unpack2(src1.data[3])));
    dst3.y = hip_pack(make_float4(hip_unpack2(src1.data[4]), hip_unpack2(src1.data[5]), hip_unpack2(src1.data[6]), hip_unpack2(src1.data[7])));
    dst4.x = hip_pack(make_float4(hip_unpack3(src1.data[0]), hip_unpack3(src1.data[1]), hip_unpack3(src1.data[2]), hip_unpack3(src1.data[3])));
    dst4.y = hip_pack(make_float4(hip_unpack3(src1.data[4]), hip_unpack3(src1.data[5]), hip_unpack3(src1.data[6]), hip_unpack3(src1.data[7])));

    *((uint2 *)(&pDstImage1[dstIdx])) = dst1;
    *((uint2 *)(&pDstImage2[dstIdx])) = dst2;
    *((uint2 *)(&pDstImage3[dstIdx])) = dst3;
    *((uint2 *)(&pDstImage4[dstIdx])) = dst4;
}
int HipExec_ChannelExtract_U8U8U8U8_U32(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage1, vx_uint8 *pHipDstImage2, vx_uint8 *pHipDstImage3, vx_uint8 *pHipDstImage4, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelExtract_U8U8U8U8_U32, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage1, (uchar *)pHipDstImage2, (uchar *)pHipDstImage3, (uchar *)pHipDstImage4, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxChannelCombine kernels for hip backend
// ----------------------------------------------------------------------------

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U16_U8U8(uint dstWidth, uint dstHeight,
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
    uint dstIdx =  y * dstImageStrideInBytes + x + x;

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint4 dst;

    dst.x = hip_pack(make_float4(hip_unpack0(src1.x), hip_unpack0(src2.x), hip_unpack1(src1.x), hip_unpack1(src2.x)));
    dst.y = hip_pack(make_float4(hip_unpack2(src1.x), hip_unpack2(src2.x), hip_unpack3(src1.x), hip_unpack3(src2.x)));
    dst.z = hip_pack(make_float4(hip_unpack0(src1.y), hip_unpack0(src2.y), hip_unpack1(src1.y), hip_unpack1(src2.y)));
    dst.w = hip_pack(make_float4(hip_unpack2(src1.y), hip_unpack2(src2.y), hip_unpack3(src1.y), hip_unpack3(src2.y)));

    *((uint4 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ChannelCombine_U16_U8U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelCombine_U16_U8U8, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U24_U8U8U8_RGB(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes,
    const uchar *pSrcImage3, uint srcImage3StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint src3Idx = y * srcImage3StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + (x * 3);

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 src3 = *((uint2 *)(&pSrcImage3[src3Idx]));
    d_uint6 dst;

    dst.data[0] = hip_pack(make_float4(hip_unpack0(src1.x), hip_unpack0(src2.x), hip_unpack0(src3.x), hip_unpack1(src1.x)));
    dst.data[1] = hip_pack(make_float4(hip_unpack1(src2.x), hip_unpack1(src3.x), hip_unpack2(src1.x), hip_unpack2(src2.x)));
    dst.data[2] = hip_pack(make_float4(hip_unpack2(src3.x), hip_unpack3(src1.x), hip_unpack3(src2.x), hip_unpack3(src3.x)));
    dst.data[3] = hip_pack(make_float4(hip_unpack0(src1.y), hip_unpack0(src2.y), hip_unpack0(src3.y), hip_unpack1(src1.y)));
    dst.data[4] = hip_pack(make_float4(hip_unpack1(src2.y), hip_unpack1(src3.y), hip_unpack2(src1.y), hip_unpack2(src2.y)));
    dst.data[5] = hip_pack(make_float4(hip_unpack2(src3.y), hip_unpack3(src1.y), hip_unpack3(src2.y), hip_unpack3(src3.y)));

    *((d_uint6 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ChannelCombine_U24_U8U8U8_RGB(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelCombine_U24_U8U8U8_RGB, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, (const uchar *)pHipSrcImage3, srcImage3StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U32_U8U8U8_UYVY(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes,
    const uchar *pSrcImage3, uint srcImage3StrideInBytes,
    uint dstWidthComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeight)) {
        uint src1Idx = y * srcImage1StrideInBytes + (x << 3);
        uint src2Idx = y * srcImage2StrideInBytes + (x << 2);
        uint src3Idx = y * srcImage3StrideInBytes + (x << 2);
        uint dstIdx =  y * dstImageStrideInBytes + (x << 4);

        uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
        uint src2 = *((uint *)(&pSrcImage2[src2Idx]));
        uint src3 = *((uint *)(&pSrcImage3[src3Idx]));
        uint4 dst;

        dst.x = hip_pack(make_float4(hip_unpack0(src2), hip_unpack0(src1.x), hip_unpack0(src3), hip_unpack1(src1.x)));
        dst.y = hip_pack(make_float4(hip_unpack1(src2), hip_unpack2(src1.x), hip_unpack1(src3), hip_unpack3(src1.x)));
        dst.z = hip_pack(make_float4(hip_unpack2(src2), hip_unpack0(src1.y), hip_unpack2(src3), hip_unpack1(src1.y)));
        dst.w = hip_pack(make_float4(hip_unpack3(src2), hip_unpack2(src1.y), hip_unpack3(src3), hip_unpack3(src1.y)));

        *((uint4 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_ChannelCombine_U32_U8U8U8_UYVY(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;

    hipLaunchKernelGGL(Hip_ChannelCombine_U32_U8U8U8_UYVY, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, (const uchar *)pHipSrcImage3, srcImage3StrideInBytes,
                        dstWidthComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U32_U8U8U8_YUYV(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes,
    const uchar *pSrcImage3, uint srcImage3StrideInBytes,
    uint dstWidthComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeight)) {
        uint src1Idx = y * srcImage1StrideInBytes + (x << 3);
        uint src2Idx = y * srcImage2StrideInBytes + (x << 2);
        uint src3Idx = y * srcImage3StrideInBytes + (x << 2);
        uint dstIdx =  y * dstImageStrideInBytes + (x << 4);

        uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
        uint src2 = *((uint *)(&pSrcImage2[src2Idx]));
        uint src3 = *((uint *)(&pSrcImage3[src3Idx]));
        uint4 dst;

        dst.x = hip_pack(make_float4(hip_unpack0(src1.x), hip_unpack0(src2), hip_unpack1(src1.x), hip_unpack0(src3)));
        dst.y = hip_pack(make_float4(hip_unpack2(src1.x), hip_unpack1(src2), hip_unpack3(src1.x), hip_unpack1(src3)));
        dst.z = hip_pack(make_float4(hip_unpack0(src1.y), hip_unpack2(src2), hip_unpack1(src1.y), hip_unpack2(src3)));
        dst.w = hip_pack(make_float4(hip_unpack2(src1.y), hip_unpack3(src2), hip_unpack3(src1.y), hip_unpack3(src3)));

        *((uint4 *)(&pDstImage[dstIdx])) = dst;
    }
}
int HipExec_ChannelCombine_U32_U8U8U8_YUYV(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;

    hipLaunchKernelGGL(Hip_ChannelCombine_U32_U8U8U8_YUYV, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, (const uchar *)pHipSrcImage3, srcImage3StrideInBytes,
                        dstWidthComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ChannelCombine_U32_U8U8U8U8_RGBX(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage1, uint srcImage1StrideInBytes,
    const uchar *pSrcImage2, uint srcImage2StrideInBytes,
    const uchar *pSrcImage3, uint srcImage3StrideInBytes,
    const uchar *pSrcImage4, uint srcImage4StrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint src1Idx = y * srcImage1StrideInBytes + x;
    uint src2Idx = y * srcImage2StrideInBytes + x;
    uint src3Idx = y * srcImage3StrideInBytes + x;
    uint src4Idx = y * srcImage4StrideInBytes + x;
    uint dstIdx =  y * dstImageStrideInBytes + (x << 2);

    uint2 src1 = *((uint2 *)(&pSrcImage1[src1Idx]));
    uint2 src2 = *((uint2 *)(&pSrcImage2[src2Idx]));
    uint2 src3 = *((uint2 *)(&pSrcImage3[src3Idx]));
    uint2 src4 = *((uint2 *)(&pSrcImage4[src4Idx]));
    d_uint8 dst;

    dst.data[0] = hip_pack(make_float4(hip_unpack0(src1.x), hip_unpack0(src2.x), hip_unpack0(src3.x), hip_unpack0(src4.x)));
    dst.data[1] = hip_pack(make_float4(hip_unpack1(src1.x), hip_unpack1(src2.x), hip_unpack1(src3.x), hip_unpack1(src4.x)));
    dst.data[2] = hip_pack(make_float4(hip_unpack2(src1.x), hip_unpack2(src2.x), hip_unpack2(src3.x), hip_unpack2(src4.x)));
    dst.data[3] = hip_pack(make_float4(hip_unpack3(src1.x), hip_unpack3(src2.x), hip_unpack3(src3.x), hip_unpack3(src4.x)));
    dst.data[4] = hip_pack(make_float4(hip_unpack0(src1.y), hip_unpack0(src2.y), hip_unpack0(src3.y), hip_unpack0(src4.y)));
    dst.data[5] = hip_pack(make_float4(hip_unpack1(src1.y), hip_unpack1(src2.y), hip_unpack1(src3.y), hip_unpack1(src4.y)));
    dst.data[6] = hip_pack(make_float4(hip_unpack2(src1.y), hip_unpack2(src2.y), hip_unpack2(src3.y), hip_unpack2(src4.y)));
    dst.data[7] = hip_pack(make_float4(hip_unpack3(src1.y), hip_unpack3(src2.y), hip_unpack3(src3.y), hip_unpack3(src4.y)));

    *((d_uint8 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ChannelCombine_U32_U8U8U8U8_RGBX(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
    const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
    const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes,
    const vx_uint8 *pHipSrcImage4, vx_uint32 srcImage4StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ChannelCombine_U32_U8U8U8U8_RGBX, dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage , dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes, (const uchar *)pHipSrcImage2, srcImage2StrideInBytes, (const uchar *)pHipSrcImage3, srcImage3StrideInBytes, (const uchar *)pHipSrcImage4, srcImage4StrideInBytes);

    return VX_SUCCESS;
}

// ----------------------------------------------------------------------------
// VxColorConvert kernels for hip backend
// ----------------------------------------------------------------------------





// Group 1 - Destination RGB, Source Packed

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_RGBX(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + (x << 2);
    uint dstIdx  = y * dstImageStrideInBytes + (x * 3);

    d_uint8 src = *((d_uint8 *)(&pSrcImage[srcIdx]));
    d_uint6 dst;

    dst.data[0] = hip_pack(make_float4(hip_unpack0(src.data[0]), hip_unpack1(src.data[0]), hip_unpack2(src.data[0]), hip_unpack0(src.data[1])));
    dst.data[1] = hip_pack(make_float4(hip_unpack1(src.data[1]), hip_unpack2(src.data[1]), hip_unpack0(src.data[2]), hip_unpack1(src.data[2])));
    dst.data[2] = hip_pack(make_float4(hip_unpack2(src.data[2]), hip_unpack0(src.data[3]), hip_unpack1(src.data[3]), hip_unpack2(src.data[3])));
    dst.data[3] = hip_pack(make_float4(hip_unpack0(src.data[4]), hip_unpack1(src.data[4]), hip_unpack2(src.data[4]), hip_unpack0(src.data[5])));
    dst.data[4] = hip_pack(make_float4(hip_unpack1(src.data[5]), hip_unpack2(src.data[5]), hip_unpack0(src.data[6]), hip_unpack1(src.data[6])));
    dst.data[5] = hip_pack(make_float4(hip_unpack2(src.data[6]), hip_unpack0(src.data[7]), hip_unpack1(src.data[7]), hip_unpack2(src.data[7])));

    *((d_uint6 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ColorConvert_RGB_RGBX(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage1, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGB_RGBX, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage1, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}
__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_UYVY(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes, uint dstImageStrideInBytesComp,
    const uchar *pSrcImage, uint srcImageStrideInBytes, uint srcImageStrideInBytesComp,
    uint dstWidthComp, uint dstHeightComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint L0Idx = y * srcImageStrideInBytesComp + (x << 4);
        uint L1Idx = L0Idx + srcImageStrideInBytes;
        uint4 L0 = *((uint4 *)(&pSrcImage[L0Idx]));
        uint4 L1 = *((uint4 *)(&pSrcImage[L1Idx]));

        uint RGB0Idx = y * dstImageStrideInBytesComp + (x * 24);
        uint RGB1Idx = RGB0Idx + dstImageStrideInBytes;

        float4 f;

        uint2 pY0, pY1;
        uint2 pU0, pU1;
        uint2 pV0, pV1;

        pY0.x = hip_pack(make_float4(hip_unpack1(L0.x), hip_unpack3(L0.x), hip_unpack1(L0.y), hip_unpack3(L0.y)));
        pY0.y = hip_pack(make_float4(hip_unpack1(L0.z), hip_unpack3(L0.z), hip_unpack1(L0.w), hip_unpack3(L0.w)));
        pY1.x = hip_pack(make_float4(hip_unpack1(L1.x), hip_unpack3(L1.x), hip_unpack1(L1.y), hip_unpack3(L1.y)));
        pY1.y = hip_pack(make_float4(hip_unpack1(L1.z), hip_unpack3(L1.z), hip_unpack1(L1.w), hip_unpack3(L1.w)));
        pU0.x = hip_pack(make_float4(hip_unpack0(L0.x), hip_unpack0(L0.x), hip_unpack0(L0.y), hip_unpack0(L0.y)));
        pU0.y = hip_pack(make_float4(hip_unpack0(L0.z), hip_unpack0(L0.z), hip_unpack0(L0.w), hip_unpack0(L0.w)));
        pU1.x = hip_pack(make_float4(hip_unpack0(L1.x), hip_unpack0(L1.x), hip_unpack0(L1.y), hip_unpack0(L1.y)));
        pU1.y = hip_pack(make_float4(hip_unpack0(L1.z), hip_unpack0(L1.z), hip_unpack0(L1.w), hip_unpack0(L1.w)));
        pV0.x = hip_pack(make_float4(hip_unpack2(L0.x), hip_unpack2(L0.x), hip_unpack2(L0.y), hip_unpack2(L0.y)));
        pV0.y = hip_pack(make_float4(hip_unpack2(L0.z), hip_unpack2(L0.z), hip_unpack2(L0.w), hip_unpack2(L0.w)));
        pV1.x = hip_pack(make_float4(hip_unpack2(L1.x), hip_unpack2(L1.x), hip_unpack2(L1.y), hip_unpack2(L1.y)));
        pV1.y = hip_pack(make_float4(hip_unpack2(L1.z), hip_unpack2(L1.z), hip_unpack2(L1.w), hip_unpack2(L1.w)));

        float2 cR = make_float2( 0.0000f,  1.5748f);
        float2 cG = make_float2(-0.1873f, -0.4681f);
        float2 cB = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        d_uint6 pRGB0, pRGB1;

        yuv.x = hip_unpack0(pY0.x);
        yuv.y = hip_unpack0(pU0.x);
        yuv.z = hip_unpack0(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY0.x);
        yuv.y = hip_unpack1(pU0.x);
        yuv.z = hip_unpack1(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB0.data[0] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY0.x);
        yuv.y = hip_unpack2(pU0.x);
        yuv.z = hip_unpack2(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB0.data[1] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY0.x);
        yuv.y = hip_unpack3(pU0.x);
        yuv.z = hip_unpack3(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[2] = hip_pack(f);

        yuv.x = hip_unpack0(pY0.y);
        yuv.y = hip_unpack0(pU0.y);
        yuv.z = hip_unpack0(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY0.y);
        yuv.y = hip_unpack1(pU0.y);
        yuv.z = hip_unpack1(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB0.data[3] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY0.y);
        yuv.y = hip_unpack2(pU0.y);
        yuv.z = hip_unpack2(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB0.data[4] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY0.y);
        yuv.y = hip_unpack3(pU0.y);
        yuv.z = hip_unpack3(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[5] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.x);
        yuv.y = hip_unpack0(pU1.x);
        yuv.z = hip_unpack0(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY1.x);
        yuv.y = hip_unpack1(pU1.x);
        yuv.z = hip_unpack1(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB1.data[0] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY1.x);
        yuv.y = hip_unpack2(pU1.x);
        yuv.z = hip_unpack2(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB1.data[1] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY1.x);
        yuv.y = hip_unpack3(pU1.x);
        yuv.z = hip_unpack3(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[2] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.y);
        yuv.y = hip_unpack0(pU1.y);
        yuv.z = hip_unpack0(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY1.y);
        yuv.y = hip_unpack1(pU1.y);
        yuv.z = hip_unpack1(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB1.data[3] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY1.y);
        yuv.y = hip_unpack2(pU1.y);
        yuv.z = hip_unpack2(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB1.data[4] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY1.y);
        yuv.y = hip_unpack3(pU1.y);
        yuv.z = hip_unpack3(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[5] = hip_pack(f);

        *((d_uint6 *)(&pDstImage[RGB0Idx])) = pRGB0;
        *((d_uint6 *)(&pDstImage[RGB1Idx])) = pRGB1;
    }
}
int HipExec_ColorConvert_RGB_UYVY(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;
    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstImageStrideInBytesComp = dstImageStrideInBytes * 2;
    vx_uint32 srcImageStrideInBytesComp = srcImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_RGB_UYVY, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes, dstImageStrideInBytesComp,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageStrideInBytesComp,
                        dstWidthComp, dstHeightComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_YUYV(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes, uint dstImageStrideInBytesComp,
    const uchar *pSrcImage, uint srcImageStrideInBytes, uint srcImageStrideInBytesComp,
    uint dstWidthComp, uint dstHeightComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint L0Idx = y * srcImageStrideInBytesComp + (x << 4);
        uint L1Idx = L0Idx + srcImageStrideInBytes;
        uint4 L0 = *((uint4 *)(&pSrcImage[L0Idx]));
        uint4 L1 = *((uint4 *)(&pSrcImage[L1Idx]));

        uint RGB0Idx = y * dstImageStrideInBytesComp + (x * 24);
        uint RGB1Idx = RGB0Idx + dstImageStrideInBytes;

        float4 f;

        uint2 pY0, pY1;
        uint2 pU0, pU1;
        uint2 pV0, pV1;

        pY0.x = hip_pack(make_float4(hip_unpack0(L0.x), hip_unpack2(L0.x), hip_unpack0(L0.y), hip_unpack2(L0.y)));
        pY0.y = hip_pack(make_float4(hip_unpack0(L0.z), hip_unpack2(L0.z), hip_unpack0(L0.w), hip_unpack2(L0.w)));
        pY1.x = hip_pack(make_float4(hip_unpack0(L1.x), hip_unpack2(L1.x), hip_unpack0(L1.y), hip_unpack2(L1.y)));
        pY1.y = hip_pack(make_float4(hip_unpack0(L1.z), hip_unpack2(L1.z), hip_unpack0(L1.w), hip_unpack2(L1.w)));
        pU0.x = hip_pack(make_float4(hip_unpack1(L0.x), hip_unpack1(L0.x), hip_unpack1(L0.y), hip_unpack1(L0.y)));
        pU0.y = hip_pack(make_float4(hip_unpack1(L0.z), hip_unpack1(L0.z), hip_unpack1(L0.w), hip_unpack1(L0.w)));
        pU1.x = hip_pack(make_float4(hip_unpack1(L1.x), hip_unpack1(L1.x), hip_unpack1(L1.y), hip_unpack1(L1.y)));
        pU1.y = hip_pack(make_float4(hip_unpack1(L1.z), hip_unpack1(L1.z), hip_unpack1(L1.w), hip_unpack1(L1.w)));
        pV0.x = hip_pack(make_float4(hip_unpack3(L0.x), hip_unpack3(L0.x), hip_unpack3(L0.y), hip_unpack3(L0.y)));
        pV0.y = hip_pack(make_float4(hip_unpack3(L0.z), hip_unpack3(L0.z), hip_unpack3(L0.w), hip_unpack3(L0.w)));
        pV1.x = hip_pack(make_float4(hip_unpack3(L1.x), hip_unpack3(L1.x), hip_unpack3(L1.y), hip_unpack3(L1.y)));
        pV1.y = hip_pack(make_float4(hip_unpack3(L1.z), hip_unpack3(L1.z), hip_unpack3(L1.w), hip_unpack3(L1.w)));

        float2 cR = make_float2( 0.0000f,  1.5748f);
        float2 cG = make_float2(-0.1873f, -0.4681f);
        float2 cB = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        d_uint6 pRGB0, pRGB1;

        yuv.x = hip_unpack0(pY0.x);
        yuv.y = hip_unpack0(pU0.x);
        yuv.z = hip_unpack0(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY0.x);
        yuv.y = hip_unpack1(pU0.x);
        yuv.z = hip_unpack1(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB0.data[0] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY0.x);
        yuv.y = hip_unpack2(pU0.x);
        yuv.z = hip_unpack2(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB0.data[1] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY0.x);
        yuv.y = hip_unpack3(pU0.x);
        yuv.z = hip_unpack3(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[2] = hip_pack(f);

        yuv.x = hip_unpack0(pY0.y);
        yuv.y = hip_unpack0(pU0.y);
        yuv.z = hip_unpack0(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY0.y);
        yuv.y = hip_unpack1(pU0.y);
        yuv.z = hip_unpack1(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB0.data[3] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY0.y);
        yuv.y = hip_unpack2(pU0.y);
        yuv.z = hip_unpack2(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB0.data[4] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY0.y);
        yuv.y = hip_unpack3(pU0.y);
        yuv.z = hip_unpack3(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[5] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.x);
        yuv.y = hip_unpack0(pU1.x);
        yuv.z = hip_unpack0(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY1.x);
        yuv.y = hip_unpack1(pU1.x);
        yuv.z = hip_unpack1(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB1.data[0] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY1.x);
        yuv.y = hip_unpack2(pU1.x);
        yuv.z = hip_unpack2(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB1.data[1] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY1.x);
        yuv.y = hip_unpack3(pU1.x);
        yuv.z = hip_unpack3(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[2] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.y);
        yuv.y = hip_unpack0(pU1.y);
        yuv.z = hip_unpack0(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY1.y);
        yuv.y = hip_unpack1(pU1.y);
        yuv.z = hip_unpack1(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB1.data[3] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY1.y);
        yuv.y = hip_unpack2(pU1.y);
        yuv.z = hip_unpack2(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB1.data[4] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY1.y);
        yuv.y = hip_unpack3(pU1.y);
        yuv.z = hip_unpack3(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[5] = hip_pack(f);

        *((d_uint6 *)(&pDstImage[RGB0Idx])) = pRGB0;
        *((d_uint6 *)(&pDstImage[RGB1Idx])) = pRGB1;
    }
}
int HipExec_ColorConvert_RGB_YUYV(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstImageStrideInBytesComp = dstImageStrideInBytes * 2;
    vx_uint32 srcImageStrideInBytesComp = srcImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_RGB_YUYV, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes, dstImageStrideInBytesComp,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageStrideInBytesComp,
                        dstWidthComp, dstHeightComp);

    return VX_SUCCESS;
}





// Group 2 - Destination RGBX, Source Packed

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_RGB(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    uint srcIdx = y * srcImageStrideInBytes + (x * 3);
    uint dstIdx  = y * dstImageStrideInBytes + (x << 2);

    d_uint6 src = *((d_uint6 *)(&pSrcImage[srcIdx]));
    d_uint8 dst;

    dst.data[0] = hip_pack(make_float4(hip_unpack0(src.data[0]), hip_unpack1(src.data[0]), hip_unpack2(src.data[0]), 255.0f));
    dst.data[1] = hip_pack(make_float4(hip_unpack3(src.data[0]), hip_unpack0(src.data[1]), hip_unpack1(src.data[1]), 255.0f));
    dst.data[2] = hip_pack(make_float4(hip_unpack2(src.data[1]), hip_unpack3(src.data[1]), hip_unpack0(src.data[2]), 255.0f));
    dst.data[3] = hip_pack(make_float4(hip_unpack1(src.data[2]), hip_unpack2(src.data[2]), hip_unpack3(src.data[2]), 255.0f));
    dst.data[4] = hip_pack(make_float4(hip_unpack0(src.data[3]), hip_unpack1(src.data[3]), hip_unpack2(src.data[3]), 255.0f));
    dst.data[5] = hip_pack(make_float4(hip_unpack3(src.data[3]), hip_unpack0(src.data[4]), hip_unpack1(src.data[4]), 255.0f));
    dst.data[6] = hip_pack(make_float4(hip_unpack2(src.data[4]), hip_unpack3(src.data[4]), hip_unpack0(src.data[5]), 255.0f));
    dst.data[7] = hip_pack(make_float4(hip_unpack1(src.data[5]), hip_unpack2(src.data[5]), hip_unpack3(src.data[5]), 255.0f));

    *((d_uint8 *)(&pDstImage[dstIdx])) = dst;
}
int HipExec_ColorConvert_RGBX_RGB(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage1, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_RGB, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage1, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage1, srcImage1StrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_UYVY(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes, uint dstImageStrideInBytesComp,
    const uchar *pSrcImage, uint srcImageStrideInBytes, uint srcImageStrideInBytesComp,
    uint dstWidthComp, uint dstHeightComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint L0Idx = y * srcImageStrideInBytesComp + (x << 4);
        uint L1Idx = L0Idx + srcImageStrideInBytes;
        uint4 L0 = *((uint4 *)(&pSrcImage[L0Idx]));
        uint4 L1 = *((uint4 *)(&pSrcImage[L1Idx]));

        uint RGB0Idx = y * dstImageStrideInBytesComp + (x << 5);
        uint RGB1Idx = RGB0Idx + dstImageStrideInBytes;

        float4 f;

        uint2 pY0, pY1;
        uint2 pU0, pU1;
        uint2 pV0, pV1;

        pY0.x = hip_pack(make_float4(hip_unpack1(L0.x), hip_unpack3(L0.x), hip_unpack1(L0.y), hip_unpack3(L0.y)));
        pY0.y = hip_pack(make_float4(hip_unpack1(L0.z), hip_unpack3(L0.z), hip_unpack1(L0.w), hip_unpack3(L0.w)));
        pY1.x = hip_pack(make_float4(hip_unpack1(L1.x), hip_unpack3(L1.x), hip_unpack1(L1.y), hip_unpack3(L1.y)));
        pY1.y = hip_pack(make_float4(hip_unpack1(L1.z), hip_unpack3(L1.z), hip_unpack1(L1.w), hip_unpack3(L1.w)));
        pU0.x = hip_pack(make_float4(hip_unpack0(L0.x), hip_unpack0(L0.x), hip_unpack0(L0.y), hip_unpack0(L0.y)));
        pU0.y = hip_pack(make_float4(hip_unpack0(L0.z), hip_unpack0(L0.z), hip_unpack0(L0.w), hip_unpack0(L0.w)));
        pU1.x = hip_pack(make_float4(hip_unpack0(L1.x), hip_unpack0(L1.x), hip_unpack0(L1.y), hip_unpack0(L1.y)));
        pU1.y = hip_pack(make_float4(hip_unpack0(L1.z), hip_unpack0(L1.z), hip_unpack0(L1.w), hip_unpack0(L1.w)));
        pV0.x = hip_pack(make_float4(hip_unpack2(L0.x), hip_unpack2(L0.x), hip_unpack2(L0.y), hip_unpack2(L0.y)));
        pV0.y = hip_pack(make_float4(hip_unpack2(L0.z), hip_unpack2(L0.z), hip_unpack2(L0.w), hip_unpack2(L0.w)));
        pV1.x = hip_pack(make_float4(hip_unpack2(L1.x), hip_unpack2(L1.x), hip_unpack2(L1.y), hip_unpack2(L1.y)));
        pV1.y = hip_pack(make_float4(hip_unpack2(L1.z), hip_unpack2(L1.z), hip_unpack2(L1.w), hip_unpack2(L1.w)));

        float2 cR = make_float2( 0.0000f,  1.5748f);
        float2 cG = make_float2(-0.1873f, -0.4681f);
        float2 cB = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        f.w = 255.0f;
        d_uint8 pRGB0, pRGB1;

        yuv.x = hip_unpack0(pY0.x);
        yuv.y = hip_unpack0(pU0.x);
        yuv.z = hip_unpack0(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[0] = hip_pack(f);

        yuv.x = hip_unpack1(pY0.x);
        yuv.y = hip_unpack1(pU0.x);
        yuv.z = hip_unpack1(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[1] = hip_pack(f);

        yuv.x = hip_unpack2(pY0.x);
        yuv.y = hip_unpack2(pU0.x);
        yuv.z = hip_unpack2(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[2] = hip_pack(f);

        yuv.x = hip_unpack3(pY0.x);
        yuv.y = hip_unpack3(pU0.x);
        yuv.z = hip_unpack3(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[3] = hip_pack(f);

        yuv.x = hip_unpack0(pY0.y);
        yuv.y = hip_unpack0(pU0.y);
        yuv.z = hip_unpack0(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[4] = hip_pack(f);

        yuv.x = hip_unpack1(pY0.y);
        yuv.y = hip_unpack1(pU0.y);
        yuv.z = hip_unpack1(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[5] = hip_pack(f);

        yuv.x = hip_unpack2(pY0.y);
        yuv.y = hip_unpack2(pU0.y);
        yuv.z = hip_unpack2(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[6] = hip_pack(f);

        yuv.x = hip_unpack3(pY0.y);
        yuv.y = hip_unpack3(pU0.y);
        yuv.z = hip_unpack3(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[7] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.x);
        yuv.y = hip_unpack0(pU1.x);
        yuv.z = hip_unpack0(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[0] = hip_pack(f);

        yuv.x = hip_unpack1(pY1.x);
        yuv.y = hip_unpack1(pU1.x);
        yuv.z = hip_unpack1(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[1] = hip_pack(f);

        yuv.x = hip_unpack2(pY1.x);
        yuv.y = hip_unpack2(pU1.x);
        yuv.z = hip_unpack2(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[2] = hip_pack(f);

        yuv.x = hip_unpack3(pY1.x);
        yuv.y = hip_unpack3(pU1.x);
        yuv.z = hip_unpack3(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[3] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.y);
        yuv.y = hip_unpack0(pU1.y);
        yuv.z = hip_unpack0(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[4] = hip_pack(f);

        yuv.x = hip_unpack1(pY1.y);
        yuv.y = hip_unpack1(pU1.y);
        yuv.z = hip_unpack1(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[5] = hip_pack(f);

        yuv.x = hip_unpack2(pY1.y);
        yuv.y = hip_unpack2(pU1.y);
        yuv.z = hip_unpack2(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[6] = hip_pack(f);

        yuv.x = hip_unpack3(pY1.y);
        yuv.y = hip_unpack3(pU1.y);
        yuv.z = hip_unpack3(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[7] = hip_pack(f);

        *((d_uint8 *)(&pDstImage[RGB0Idx])) = pRGB0;
        *((d_uint8 *)(&pDstImage[RGB1Idx])) = pRGB1;
    }
}
int HipExec_ColorConvert_RGBX_UYVY(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstImageStrideInBytesComp = dstImageStrideInBytes * 2;
    vx_uint32 srcImageStrideInBytesComp = srcImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_UYVY, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes, dstImageStrideInBytesComp,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageStrideInBytesComp,
                        dstWidthComp, dstHeightComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_YUYV(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes, uint dstImageStrideInBytesComp,
    const uchar *pSrcImage, uint srcImageStrideInBytes, uint srcImageStrideInBytesComp,
    uint dstWidthComp, uint dstHeightComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint L0Idx = y * srcImageStrideInBytesComp + (x << 4);
        uint L1Idx = L0Idx + srcImageStrideInBytes;
        uint4 L0 = *((uint4 *)(&pSrcImage[L0Idx]));
        uint4 L1 = *((uint4 *)(&pSrcImage[L1Idx]));

        uint RGB0Idx = y * dstImageStrideInBytesComp + (x << 5);
        uint RGB1Idx = RGB0Idx + dstImageStrideInBytes;

        float4 f;

        uint2 pY0, pY1;
        uint2 pU0, pU1;
        uint2 pV0, pV1;

        pY0.x = hip_pack(make_float4(hip_unpack0(L0.x), hip_unpack2(L0.x), hip_unpack0(L0.y), hip_unpack2(L0.y)));
        pY0.y = hip_pack(make_float4(hip_unpack0(L0.z), hip_unpack2(L0.z), hip_unpack0(L0.w), hip_unpack2(L0.w)));
        pY1.x = hip_pack(make_float4(hip_unpack0(L1.x), hip_unpack2(L1.x), hip_unpack0(L1.y), hip_unpack2(L1.y)));
        pY1.y = hip_pack(make_float4(hip_unpack0(L1.z), hip_unpack2(L1.z), hip_unpack0(L1.w), hip_unpack2(L1.w)));
        pU0.x = hip_pack(make_float4(hip_unpack1(L0.x), hip_unpack1(L0.x), hip_unpack1(L0.y), hip_unpack1(L0.y)));
        pU0.y = hip_pack(make_float4(hip_unpack1(L0.z), hip_unpack1(L0.z), hip_unpack1(L0.w), hip_unpack1(L0.w)));
        pU1.x = hip_pack(make_float4(hip_unpack1(L1.x), hip_unpack1(L1.x), hip_unpack1(L1.y), hip_unpack1(L1.y)));
        pU1.y = hip_pack(make_float4(hip_unpack1(L1.z), hip_unpack1(L1.z), hip_unpack1(L1.w), hip_unpack1(L1.w)));
        pV0.x = hip_pack(make_float4(hip_unpack3(L0.x), hip_unpack3(L0.x), hip_unpack3(L0.y), hip_unpack3(L0.y)));
        pV0.y = hip_pack(make_float4(hip_unpack3(L0.z), hip_unpack3(L0.z), hip_unpack3(L0.w), hip_unpack3(L0.w)));
        pV1.x = hip_pack(make_float4(hip_unpack3(L1.x), hip_unpack3(L1.x), hip_unpack3(L1.y), hip_unpack3(L1.y)));
        pV1.y = hip_pack(make_float4(hip_unpack3(L1.z), hip_unpack3(L1.z), hip_unpack3(L1.w), hip_unpack3(L1.w)));

        float2 cR = make_float2( 0.0000f,  1.5748f);
        float2 cG = make_float2(-0.1873f, -0.4681f);
        float2 cB = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        f.w = 255.0f;
        d_uint8 pRGB0, pRGB1;

        yuv.x = hip_unpack0(pY0.x);
        yuv.y = hip_unpack0(pU0.x);
        yuv.z = hip_unpack0(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[0] = hip_pack(f);

        yuv.x = hip_unpack1(pY0.x);
        yuv.y = hip_unpack1(pU0.x);
        yuv.z = hip_unpack1(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[1] = hip_pack(f);

        yuv.x = hip_unpack2(pY0.x);
        yuv.y = hip_unpack2(pU0.x);
        yuv.z = hip_unpack2(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[2] = hip_pack(f);

        yuv.x = hip_unpack3(pY0.x);
        yuv.y = hip_unpack3(pU0.x);
        yuv.z = hip_unpack3(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[3] = hip_pack(f);

        yuv.x = hip_unpack0(pY0.y);
        yuv.y = hip_unpack0(pU0.y);
        yuv.z = hip_unpack0(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[4] = hip_pack(f);

        yuv.x = hip_unpack1(pY0.y);
        yuv.y = hip_unpack1(pU0.y);
        yuv.z = hip_unpack1(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[5] = hip_pack(f);

        yuv.x = hip_unpack2(pY0.y);
        yuv.y = hip_unpack2(pU0.y);
        yuv.z = hip_unpack2(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[6] = hip_pack(f);

        yuv.x = hip_unpack3(pY0.y);
        yuv.y = hip_unpack3(pU0.y);
        yuv.z = hip_unpack3(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[7] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.x);
        yuv.y = hip_unpack0(pU1.x);
        yuv.z = hip_unpack0(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[0] = hip_pack(f);

        yuv.x = hip_unpack1(pY1.x);
        yuv.y = hip_unpack1(pU1.x);
        yuv.z = hip_unpack1(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[1] = hip_pack(f);

        yuv.x = hip_unpack2(pY1.x);
        yuv.y = hip_unpack2(pU1.x);
        yuv.z = hip_unpack2(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[2] = hip_pack(f);

        yuv.x = hip_unpack3(pY1.x);
        yuv.y = hip_unpack3(pU1.x);
        yuv.z = hip_unpack3(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[3] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.y);
        yuv.y = hip_unpack0(pU1.y);
        yuv.z = hip_unpack0(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[4] = hip_pack(f);

        yuv.x = hip_unpack1(pY1.y);
        yuv.y = hip_unpack1(pU1.y);
        yuv.z = hip_unpack1(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[5] = hip_pack(f);

        yuv.x = hip_unpack2(pY1.y);
        yuv.y = hip_unpack2(pU1.y);
        yuv.z = hip_unpack2(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[6] = hip_pack(f);

        yuv.x = hip_unpack3(pY1.y);
        yuv.y = hip_unpack3(pU1.y);
        yuv.z = hip_unpack3(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[7] = hip_pack(f);

        *((d_uint8 *)(&pDstImage[RGB0Idx])) = pRGB0;
        *((d_uint8 *)(&pDstImage[RGB1Idx])) = pRGB1;
    }
}
int HipExec_ColorConvert_RGBX_YUYV(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstImageStrideInBytesComp = dstImageStrideInBytes * 2;
    vx_uint32 srcImageStrideInBytesComp = srcImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_YUYV, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes, dstImageStrideInBytesComp,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes, srcImageStrideInBytesComp,
                        dstWidthComp, dstHeightComp);

    return VX_SUCCESS;
}





// Group 3 - Destination RGB, Source Planar

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_IYUV(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes, uint dstImageStrideInBytesComp,
    const uchar *pSrcYImage, uint srcYImageStrideInBytes, const uchar *pSrcUImage, uint srcUImageStrideInBytes, const uchar *pSrcVImage, uint srcVImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcYImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint srcY0Idx = y * srcYImageStrideInBytesComp + (x << 3);
        uint srcY1Idx = srcY0Idx + srcYImageStrideInBytes;
        uint srcUIdx = y * srcUImageStrideInBytes + (x << 2);
        uint srcVIdx = y * srcVImageStrideInBytes + (x << 2);
        uint2 pY0 = *((uint2 *)(&pSrcYImage[srcY0Idx]));
        uint2 pY1 = *((uint2 *)(&pSrcYImage[srcY1Idx]));
        uint2 pUV;
        pUV.x = *((uint *)(&pSrcUImage[srcUIdx]));
        pUV.y = *((uint *)(&pSrcVImage[srcVIdx]));

        uint RGB0Idx = y * dstImageStrideInBytesComp + (x * 24);
        uint RGB1Idx = RGB0Idx + dstImageStrideInBytes;

        float4 f;

        uint2 pU0, pU1;
        uint2 pV0, pV1;

        f.x = hip_unpack0(pUV.x);
        f.y = f.x;
        f.z = hip_unpack1(pUV.x);
        f.w = f.z;
        pU0.x = hip_pack(f);

        f.x = hip_unpack2(pUV.x);
        f.y = f.x;
        f.z = hip_unpack3(pUV.x);
        f.w = f.z;
        pU0.y = hip_pack(f);

        pU1.x = pU0.x;
        pU1.y = pU0.y;

        f.x = hip_unpack0(pUV.y);
        f.y = f.x;
        f.z = hip_unpack1(pUV.y);
        f.w = f.z;
        pV0.x = hip_pack(f);

        f.x = hip_unpack2(pUV.y);
        f.y = f.x;
        f.z = hip_unpack3(pUV.y);
        f.w = f.z;
        pV0.y = hip_pack(f);

        pV1.x = pV0.x;
        pV1.y = pV0.y;

        float2 cR = make_float2( 0.0000f,  1.5748f);
        float2 cG = make_float2(-0.1873f, -0.4681f);
        float2 cB = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        d_uint6 pRGB0, pRGB1;

        yuv.x = hip_unpack0(pY0.x);
        yuv.y = hip_unpack0(pU0.x);
        yuv.z = hip_unpack0(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY0.x);
        yuv.y = hip_unpack1(pU0.x);
        yuv.z = hip_unpack1(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB0.data[0] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY0.x);
        yuv.y = hip_unpack2(pU0.x);
        yuv.z = hip_unpack2(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB0.data[1] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY0.x);
        yuv.y = hip_unpack3(pU0.x);
        yuv.z = hip_unpack3(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[2] = hip_pack(f);

        yuv.x = hip_unpack0(pY0.y);
        yuv.y = hip_unpack0(pU0.y);
        yuv.z = hip_unpack0(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY0.y);
        yuv.y = hip_unpack1(pU0.y);
        yuv.z = hip_unpack1(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB0.data[3] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY0.y);
        yuv.y = hip_unpack2(pU0.y);
        yuv.z = hip_unpack2(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB0.data[4] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY0.y);
        yuv.y = hip_unpack3(pU0.y);
        yuv.z = hip_unpack3(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[5] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.x);
        yuv.y = hip_unpack0(pU1.x);
        yuv.z = hip_unpack0(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY1.x);
        yuv.y = hip_unpack1(pU1.x);
        yuv.z = hip_unpack1(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB1.data[0] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY1.x);
        yuv.y = hip_unpack2(pU1.x);
        yuv.z = hip_unpack2(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB1.data[1] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY1.x);
        yuv.y = hip_unpack3(pU1.x);
        yuv.z = hip_unpack3(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[2] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.y);
        yuv.y = hip_unpack0(pU1.y);
        yuv.z = hip_unpack0(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY1.y);
        yuv.y = hip_unpack1(pU1.y);
        yuv.z = hip_unpack1(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB1.data[3] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY1.y);
        yuv.y = hip_unpack2(pU1.y);
        yuv.z = hip_unpack2(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB1.data[4] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY1.y);
        yuv.y = hip_unpack3(pU1.y);
        yuv.z = hip_unpack3(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[5] = hip_pack(f);

        *((d_uint6 *)(&pDstImage[RGB0Idx])) = pRGB0;
        *((d_uint6 *)(&pDstImage[RGB1Idx])) = pRGB1;
    }
}
int HipExec_ColorConvert_RGB_IYUV(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcYImage, vx_uint32 srcYImageStrideInBytes,
    const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
    const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstImageStrideInBytesComp = dstImageStrideInBytes * 2;
    vx_uint32 srcYImageStrideInBytesComp = srcYImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_RGB_IYUV, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes, dstImageStrideInBytesComp,
                        (const uchar *)pHipSrcYImage, srcYImageStrideInBytes, (const uchar *)pHipSrcUImage, srcUImageStrideInBytes, (const uchar *)pHipSrcVImage, srcVImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcYImageStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_NV12(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes, uint dstImageStrideInBytesComp,
    const uchar *pSrcLumaImage, uint srcLumaImageStrideInBytes, const uchar *pSrcChromaImage, uint srcChromaImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcLumaImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint srcY0Idx = y * srcLumaImageStrideInBytesComp + (x << 3);
        uint srcY1Idx = srcY0Idx + srcLumaImageStrideInBytes;
        uint srcUVIdx = y * srcChromaImageStrideInBytes + (x << 3);
        uint2 pY0 = *((uint2 *)(&pSrcLumaImage[srcY0Idx]));
        uint2 pY1 = *((uint2 *)(&pSrcLumaImage[srcY1Idx]));
        uint2 pUV = *((uint2 *)(&pSrcChromaImage[srcUVIdx]));

        uint RGB0Idx = y * dstImageStrideInBytesComp + (x * 24);
        uint RGB1Idx = RGB0Idx + dstImageStrideInBytes;

        float4 f;

        uint2 pU0, pU1;
        uint2 pV0, pV1;

        f.x = hip_unpack0(pUV.x);
        f.y = f.x;
        f.z = hip_unpack2(pUV.x);
        f.w = f.z;
        pU0.x = hip_pack(f);

        f.x = hip_unpack0(pUV.y);
        f.y = f.x;
        f.z = hip_unpack2(pUV.y);
        f.w = f.z;
        pU0.y = hip_pack(f);

        pU1.x = pU0.x;
        pU1.y = pU0.y;

        f.x = hip_unpack1(pUV.x);
        f.y = f.x;
        f.z = hip_unpack3(pUV.x);
        f.w = f.z;
        pV0.x = hip_pack(f);

        f.x = hip_unpack1(pUV.y);
        f.y = f.x;
        f.z = hip_unpack3(pUV.y);
        f.w = f.z;
        pV0.y = hip_pack(f);

        pV1.x = pV0.x;
        pV1.y = pV0.y;

        float2 cR = make_float2( 0.0000f,  1.5748f);
        float2 cG = make_float2(-0.1873f, -0.4681f);
        float2 cB = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        d_uint6 pRGB0, pRGB1;

        yuv.x = hip_unpack0(pY0.x);
        yuv.y = hip_unpack0(pU0.x);
        yuv.z = hip_unpack0(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY0.x);
        yuv.y = hip_unpack1(pU0.x);
        yuv.z = hip_unpack1(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB0.data[0] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY0.x);
        yuv.y = hip_unpack2(pU0.x);
        yuv.z = hip_unpack2(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB0.data[1] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY0.x);
        yuv.y = hip_unpack3(pU0.x);
        yuv.z = hip_unpack3(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[2] = hip_pack(f);

        yuv.x = hip_unpack0(pY0.y);
        yuv.y = hip_unpack0(pU0.y);
        yuv.z = hip_unpack0(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY0.y);
        yuv.y = hip_unpack1(pU0.y);
        yuv.z = hip_unpack1(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB0.data[3] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY0.y);
        yuv.y = hip_unpack2(pU0.y);
        yuv.z = hip_unpack2(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB0.data[4] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY0.y);
        yuv.y = hip_unpack3(pU0.y);
        yuv.z = hip_unpack3(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[5] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.x);
        yuv.y = hip_unpack0(pU1.x);
        yuv.z = hip_unpack0(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY1.x);
        yuv.y = hip_unpack1(pU1.x);
        yuv.z = hip_unpack1(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB1.data[0] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY1.x);
        yuv.y = hip_unpack2(pU1.x);
        yuv.z = hip_unpack2(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB1.data[1] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY1.x);
        yuv.y = hip_unpack3(pU1.x);
        yuv.z = hip_unpack3(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[2] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.y);
        yuv.y = hip_unpack0(pU1.y);
        yuv.z = hip_unpack0(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY1.y);
        yuv.y = hip_unpack1(pU1.y);
        yuv.z = hip_unpack1(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB1.data[3] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY1.y);
        yuv.y = hip_unpack2(pU1.y);
        yuv.z = hip_unpack2(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB1.data[4] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY1.y);
        yuv.y = hip_unpack3(pU1.y);
        yuv.z = hip_unpack3(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[5] = hip_pack(f);

        *((d_uint6 *)(&pDstImage[RGB0Idx])) = pRGB0;
        *((d_uint6 *)(&pDstImage[RGB1Idx])) = pRGB1;
    }
}
int HipExec_ColorConvert_RGB_NV12(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstImageStrideInBytesComp = dstImageStrideInBytes * 2;
    vx_uint32 srcLumaImageStrideInBytesComp = srcLumaImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_RGB_NV12, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes, dstImageStrideInBytesComp,
                        (const uchar *)pHipSrcLumaImage, srcLumaImageStrideInBytes, (const uchar *)pHipSrcChromaImage, srcChromaImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcLumaImageStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGB_NV21(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes, uint dstImageStrideInBytesComp,
    const uchar *pSrcLumaImage, uint srcLumaImageStrideInBytes, const uchar *pSrcChromaImage, uint srcChromaImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcLumaImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint srcY0Idx = y * srcLumaImageStrideInBytesComp + (x << 3);
        uint srcY1Idx = srcY0Idx + srcLumaImageStrideInBytes;
        uint srcUVIdx = y * srcChromaImageStrideInBytes + (x << 3);
        uint2 pY0 = *((uint2 *)(&pSrcLumaImage[srcY0Idx]));
        uint2 pY1 = *((uint2 *)(&pSrcLumaImage[srcY1Idx]));
        uint2 pUV = *((uint2 *)(&pSrcChromaImage[srcUVIdx]));

        uint RGB0Idx = y * dstImageStrideInBytesComp + (x * 24);
        uint RGB1Idx = RGB0Idx + dstImageStrideInBytes;

        float4 f;

        uint2 pU0, pU1;
        uint2 pV0, pV1;

        f.x = hip_unpack1(pUV.x);
        f.y = f.x;
        f.z = hip_unpack3(pUV.x);
        f.w = f.z;
        pU0.x = hip_pack(f);

        f.x = hip_unpack1(pUV.y);
        f.y = f.x;
        f.z = hip_unpack3(pUV.y);
        f.w = f.z;
        pU0.y = hip_pack(f);

        pU1.x = pU0.x;
        pU1.y = pU0.y;

        f.x = hip_unpack0(pUV.x);
        f.y = f.x;
        f.z = hip_unpack2(pUV.x);
        f.w = f.z;
        pV0.x = hip_pack(f);

        f.x = hip_unpack0(pUV.y);
        f.y = f.x;
        f.z = hip_unpack2(pUV.y);
        f.w = f.z;
        pV0.y = hip_pack(f);

        pV1.x = pV0.x;
        pV1.y = pV0.y;

        float2 cR = make_float2( 0.0000f,  1.5748f);
        float2 cG = make_float2(-0.1873f, -0.4681f);
        float2 cB = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        d_uint6 pRGB0, pRGB1;

        yuv.x = hip_unpack0(pY0.x);
        yuv.y = hip_unpack0(pU0.x);
        yuv.z = hip_unpack0(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY0.x);
        yuv.y = hip_unpack1(pU0.x);
        yuv.z = hip_unpack1(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB0.data[0] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY0.x);
        yuv.y = hip_unpack2(pU0.x);
        yuv.z = hip_unpack2(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB0.data[1] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY0.x);
        yuv.y = hip_unpack3(pU0.x);
        yuv.z = hip_unpack3(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[2] = hip_pack(f);

        yuv.x = hip_unpack0(pY0.y);
        yuv.y = hip_unpack0(pU0.y);
        yuv.z = hip_unpack0(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY0.y);
        yuv.y = hip_unpack1(pU0.y);
        yuv.z = hip_unpack1(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB0.data[3] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY0.y);
        yuv.y = hip_unpack2(pU0.y);
        yuv.z = hip_unpack2(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB0.data[4] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY0.y);
        yuv.y = hip_unpack3(pU0.y);
        yuv.z = hip_unpack3(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[5] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.x);
        yuv.y = hip_unpack0(pU1.x);
        yuv.z = hip_unpack0(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY1.x);
        yuv.y = hip_unpack1(pU1.x);
        yuv.z = hip_unpack1(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB1.data[0] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY1.x);
        yuv.y = hip_unpack2(pU1.x);
        yuv.z = hip_unpack2(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB1.data[1] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY1.x);
        yuv.y = hip_unpack3(pU1.x);
        yuv.z = hip_unpack3(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[2] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.y);
        yuv.y = hip_unpack0(pU1.y);
        yuv.z = hip_unpack0(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack1(pY1.y);
        yuv.y = hip_unpack1(pU1.y);
        yuv.z = hip_unpack1(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.w = fmaf(cR.y, yuv.z, yuv.x);
        pRGB1.data[3] = hip_pack(f);

        f.x = fmaf(cG.x, yuv.y, yuv.x);
        f.x = fmaf(cG.y, yuv.z, f.x);
        f.y = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack2(pY1.y);
        yuv.y = hip_unpack2(pU1.y);
        yuv.z = hip_unpack2(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.z = fmaf(cR.y, yuv.z, yuv.x);
        f.w = fmaf(cG.x, yuv.y, yuv.x);
        f.w = fmaf(cG.y, yuv.z, f.w);
        pRGB1.data[4] = hip_pack(f);

        f.x = fmaf(cB.x, yuv.y, yuv.x);
        yuv.x = hip_unpack3(pY1.y);
        yuv.y = hip_unpack3(pU1.y);
        yuv.z = hip_unpack3(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.y = fmaf(cR.y, yuv.z, yuv.x);
        f.z = fmaf(cG.x, yuv.y, yuv.x);
        f.z = fmaf(cG.y, yuv.z, f.z);
        f.w = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[5] = hip_pack(f);

        *((d_uint6 *)(&pDstImage[RGB0Idx])) = pRGB0;
        *((d_uint6 *)(&pDstImage[RGB1Idx])) = pRGB1;
    }
}
int HipExec_ColorConvert_RGB_NV21(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstImageStrideInBytesComp = dstImageStrideInBytes * 2;
    vx_uint32 srcLumaImageStrideInBytesComp = srcLumaImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_RGB_NV21, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes, dstImageStrideInBytesComp,
                        (const uchar *)pHipSrcLumaImage, srcLumaImageStrideInBytes, (const uchar *)pHipSrcChromaImage, srcChromaImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcLumaImageStrideInBytesComp);

    return VX_SUCCESS;
}





// Group 4 - Destination RGBX, Source Planar

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_IYUV(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes, uint dstImageStrideInBytesComp,
    const uchar *pSrcYImage, uint srcYImageStrideInBytes, const uchar *pSrcUImage, uint srcUImageStrideInBytes, const uchar *pSrcVImage, uint srcVImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcYImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint srcY0Idx = y * srcYImageStrideInBytesComp + (x << 3);
        uint srcY1Idx = srcY0Idx + srcYImageStrideInBytes;
        uint srcUIdx = y * srcUImageStrideInBytes + (x << 2);
        uint srcVIdx = y * srcVImageStrideInBytes + (x << 2);
        uint2 pY0 = *((uint2 *)(&pSrcYImage[srcY0Idx]));
        uint2 pY1 = *((uint2 *)(&pSrcYImage[srcY1Idx]));
        uint2 pUV;
        pUV.x = *((uint *)(&pSrcUImage[srcUIdx]));
        pUV.y = *((uint *)(&pSrcVImage[srcVIdx]));

        uint RGB0Idx = y * dstImageStrideInBytesComp + (x << 5);
        uint RGB1Idx = RGB0Idx + dstImageStrideInBytes;

        float4 f;

        uint2 pU0, pU1;
        uint2 pV0, pV1;

        f.x = hip_unpack0(pUV.x);
        f.y = f.x;
        f.z = hip_unpack1(pUV.x);
        f.w = f.z;
        pU0.x = hip_pack(f);

        f.x = hip_unpack2(pUV.x);
        f.y = f.x;
        f.z = hip_unpack3(pUV.x);
        f.w = f.z;
        pU0.y = hip_pack(f);

        pU1.x = pU0.x;
        pU1.y = pU0.y;

        f.x = hip_unpack0(pUV.y);
        f.y = f.x;
        f.z = hip_unpack1(pUV.y);
        f.w = f.z;
        pV0.x = hip_pack(f);

        f.x = hip_unpack2(pUV.y);
        f.y = f.x;
        f.z = hip_unpack3(pUV.y);
        f.w = f.z;
        pV0.y = hip_pack(f);

        pV1.x = pV0.x;
        pV1.y = pV0.y;

        float2 cR = make_float2( 0.0000f,  1.5748f);
        float2 cG = make_float2(-0.1873f, -0.4681f);
        float2 cB = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        f.w = 255.0f;
        d_uint8 pRGB0, pRGB1;

        yuv.x = hip_unpack0(pY0.x);
        yuv.y = hip_unpack0(pU0.x);
        yuv.z = hip_unpack0(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[0] = hip_pack(f);

        yuv.x = hip_unpack1(pY0.x);
        yuv.y = hip_unpack1(pU0.x);
        yuv.z = hip_unpack1(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[1] = hip_pack(f);

        yuv.x = hip_unpack2(pY0.x);
        yuv.y = hip_unpack2(pU0.x);
        yuv.z = hip_unpack2(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[2] = hip_pack(f);

        yuv.x = hip_unpack3(pY0.x);
        yuv.y = hip_unpack3(pU0.x);
        yuv.z = hip_unpack3(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[3] = hip_pack(f);

        yuv.x = hip_unpack0(pY0.y);
        yuv.y = hip_unpack0(pU0.y);
        yuv.z = hip_unpack0(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[4] = hip_pack(f);

        yuv.x = hip_unpack1(pY0.y);
        yuv.y = hip_unpack1(pU0.y);
        yuv.z = hip_unpack1(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[5] = hip_pack(f);

        yuv.x = hip_unpack2(pY0.y);
        yuv.y = hip_unpack2(pU0.y);
        yuv.z = hip_unpack2(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[6] = hip_pack(f);

        yuv.x = hip_unpack3(pY0.y);
        yuv.y = hip_unpack3(pU0.y);
        yuv.z = hip_unpack3(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[7] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.x);
        yuv.y = hip_unpack0(pU1.x);
        yuv.z = hip_unpack0(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[0] = hip_pack(f);

        yuv.x = hip_unpack1(pY1.x);
        yuv.y = hip_unpack1(pU1.x);
        yuv.z = hip_unpack1(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[1] = hip_pack(f);

        yuv.x = hip_unpack2(pY1.x);
        yuv.y = hip_unpack2(pU1.x);
        yuv.z = hip_unpack2(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[2] = hip_pack(f);

        yuv.x = hip_unpack3(pY1.x);
        yuv.y = hip_unpack3(pU1.x);
        yuv.z = hip_unpack3(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[3] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.y);
        yuv.y = hip_unpack0(pU1.y);
        yuv.z = hip_unpack0(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[4] = hip_pack(f);

        yuv.x = hip_unpack1(pY1.y);
        yuv.y = hip_unpack1(pU1.y);
        yuv.z = hip_unpack1(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[5] = hip_pack(f);

        yuv.x = hip_unpack2(pY1.y);
        yuv.y = hip_unpack2(pU1.y);
        yuv.z = hip_unpack2(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[6] = hip_pack(f);

        yuv.x = hip_unpack3(pY1.y);
        yuv.y = hip_unpack3(pU1.y);
        yuv.z = hip_unpack3(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[7] = hip_pack(f);

        *((d_uint8 *)(&pDstImage[RGB0Idx])) = pRGB0;
        *((d_uint8 *)(&pDstImage[RGB1Idx])) = pRGB1;
    }
}
int HipExec_ColorConvert_RGBX_IYUV(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcYImage, vx_uint32 srcYImageStrideInBytes,
    const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
    const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstImageStrideInBytesComp = dstImageStrideInBytes * 2;
    vx_uint32 srcYImageStrideInBytesComp = srcYImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_IYUV, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes, dstImageStrideInBytesComp,
                        (const uchar *)pHipSrcYImage, srcYImageStrideInBytes, (const uchar *)pHipSrcUImage, srcUImageStrideInBytes, (const uchar *)pHipSrcVImage, srcVImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcYImageStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_NV12(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes, uint dstImageStrideInBytesComp,
    const uchar *pSrcLumaImage, uint srcLumaImageStrideInBytes, const uchar *pSrcChromaImage, uint srcChromaImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcLumaImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint srcY0Idx = y * srcLumaImageStrideInBytesComp + (x << 3);
        uint srcY1Idx = srcY0Idx + srcLumaImageStrideInBytes;
        uint srcUVIdx = y * srcChromaImageStrideInBytes + (x << 3);
        uint2 pY0 = *((uint2 *)(&pSrcLumaImage[srcY0Idx]));
        uint2 pY1 = *((uint2 *)(&pSrcLumaImage[srcY1Idx]));
        uint2 pUV = *((uint2 *)(&pSrcChromaImage[srcUVIdx]));

        uint RGB0Idx = y * dstImageStrideInBytesComp + (x << 5);
        uint RGB1Idx = RGB0Idx + dstImageStrideInBytes;

        float4 f;

        uint2 pU0, pU1;
        uint2 pV0, pV1;

        f.x = hip_unpack0(pUV.x);
        f.y = f.x;
        f.z = hip_unpack2(pUV.x);
        f.w = f.z;
        pU0.x = hip_pack(f);

        f.x = hip_unpack0(pUV.y);
        f.y = f.x;
        f.z = hip_unpack2(pUV.y);
        f.w = f.z;
        pU0.y = hip_pack(f);

        pU1.x = pU0.x;
        pU1.y = pU0.y;

        f.x = hip_unpack1(pUV.x);
        f.y = f.x;
        f.z = hip_unpack3(pUV.x);
        f.w = f.z;
        pV0.x = hip_pack(f);

        f.x = hip_unpack1(pUV.y);
        f.y = f.x;
        f.z = hip_unpack3(pUV.y);
        f.w = f.z;
        pV0.y = hip_pack(f);

        pV1.x = pV0.x;
        pV1.y = pV0.y;

        float2 cR = make_float2( 0.0000f,  1.5748f);
        float2 cG = make_float2(-0.1873f, -0.4681f);
        float2 cB = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        f.w = 255.0f;
        d_uint8 pRGB0, pRGB1;

        yuv.x = hip_unpack0(pY0.x);
        yuv.y = hip_unpack0(pU0.x);
        yuv.z = hip_unpack0(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[0] = hip_pack(f);

        yuv.x = hip_unpack1(pY0.x);
        yuv.y = hip_unpack1(pU0.x);
        yuv.z = hip_unpack1(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[1] = hip_pack(f);

        yuv.x = hip_unpack2(pY0.x);
        yuv.y = hip_unpack2(pU0.x);
        yuv.z = hip_unpack2(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[2] = hip_pack(f);

        yuv.x = hip_unpack3(pY0.x);
        yuv.y = hip_unpack3(pU0.x);
        yuv.z = hip_unpack3(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[3] = hip_pack(f);

        yuv.x = hip_unpack0(pY0.y);
        yuv.y = hip_unpack0(pU0.y);
        yuv.z = hip_unpack0(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[4] = hip_pack(f);

        yuv.x = hip_unpack1(pY0.y);
        yuv.y = hip_unpack1(pU0.y);
        yuv.z = hip_unpack1(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[5] = hip_pack(f);

        yuv.x = hip_unpack2(pY0.y);
        yuv.y = hip_unpack2(pU0.y);
        yuv.z = hip_unpack2(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[6] = hip_pack(f);

        yuv.x = hip_unpack3(pY0.y);
        yuv.y = hip_unpack3(pU0.y);
        yuv.z = hip_unpack3(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[7] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.x);
        yuv.y = hip_unpack0(pU1.x);
        yuv.z = hip_unpack0(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[0] = hip_pack(f);

        yuv.x = hip_unpack1(pY1.x);
        yuv.y = hip_unpack1(pU1.x);
        yuv.z = hip_unpack1(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[1] = hip_pack(f);

        yuv.x = hip_unpack2(pY1.x);
        yuv.y = hip_unpack2(pU1.x);
        yuv.z = hip_unpack2(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[2] = hip_pack(f);

        yuv.x = hip_unpack3(pY1.x);
        yuv.y = hip_unpack3(pU1.x);
        yuv.z = hip_unpack3(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[3] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.y);
        yuv.y = hip_unpack0(pU1.y);
        yuv.z = hip_unpack0(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[4] = hip_pack(f);

        yuv.x = hip_unpack1(pY1.y);
        yuv.y = hip_unpack1(pU1.y);
        yuv.z = hip_unpack1(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[5] = hip_pack(f);

        yuv.x = hip_unpack2(pY1.y);
        yuv.y = hip_unpack2(pU1.y);
        yuv.z = hip_unpack2(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[6] = hip_pack(f);

        yuv.x = hip_unpack3(pY1.y);
        yuv.y = hip_unpack3(pU1.y);
        yuv.z = hip_unpack3(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[7] = hip_pack(f);

        *((d_uint8 *)(&pDstImage[RGB0Idx])) = pRGB0;
        *((d_uint8 *)(&pDstImage[RGB1Idx])) = pRGB1;
    }
}
int HipExec_ColorConvert_RGBX_NV12(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstImageStrideInBytesComp = dstImageStrideInBytes * 2;
    vx_uint32 srcLumaImageStrideInBytesComp = srcLumaImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_NV12, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes, dstImageStrideInBytesComp,
                        (const uchar *)pHipSrcLumaImage, srcLumaImageStrideInBytes, (const uchar *)pHipSrcChromaImage, srcChromaImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcLumaImageStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_RGBX_NV21(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes, uint dstImageStrideInBytesComp,
    const uchar *pSrcLumaImage, uint srcLumaImageStrideInBytes, const uchar *pSrcChromaImage, uint srcChromaImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcLumaImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint srcY0Idx = y * srcLumaImageStrideInBytesComp + (x << 3);
        uint srcY1Idx = srcY0Idx + srcLumaImageStrideInBytes;
        uint srcUVIdx = y * srcChromaImageStrideInBytes + (x << 3);
        uint2 pY0 = *((uint2 *)(&pSrcLumaImage[srcY0Idx]));
        uint2 pY1 = *((uint2 *)(&pSrcLumaImage[srcY1Idx]));
        uint2 pUV = *((uint2 *)(&pSrcChromaImage[srcUVIdx]));

        uint RGB0Idx = y * dstImageStrideInBytesComp + (x << 5);
        uint RGB1Idx = RGB0Idx + dstImageStrideInBytes;

        float4 f;

        uint2 pU0, pU1;
        uint2 pV0, pV1;

        f.x = hip_unpack1(pUV.x);
        f.y = f.x;
        f.z = hip_unpack3(pUV.x);
        f.w = f.z;
        pU0.x = hip_pack(f);

        f.x = hip_unpack1(pUV.y);
        f.y = f.x;
        f.z = hip_unpack3(pUV.y);
        f.w = f.z;
        pU0.y = hip_pack(f);

        pU1.x = pU0.x;
        pU1.y = pU0.y;

        f.x = hip_unpack0(pUV.x);
        f.y = f.x;
        f.z = hip_unpack2(pUV.x);
        f.w = f.z;
        pV0.x = hip_pack(f);

        f.x = hip_unpack0(pUV.y);
        f.y = f.x;
        f.z = hip_unpack2(pUV.y);
        f.w = f.z;
        pV0.y = hip_pack(f);

        pV1.x = pV0.x;
        pV1.y = pV0.y;

        float2 cR = make_float2( 0.0000f,  1.5748f);
        float2 cG = make_float2(-0.1873f, -0.4681f);
        float2 cB = make_float2( 1.8556f,  0.0000f);
        float3 yuv;
        f.w = 255.0f;
        d_uint8 pRGB0, pRGB1;

        yuv.x = hip_unpack0(pY0.x);
        yuv.y = hip_unpack0(pU0.x);
        yuv.z = hip_unpack0(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[0] = hip_pack(f);

        yuv.x = hip_unpack1(pY0.x);
        yuv.y = hip_unpack1(pU0.x);
        yuv.z = hip_unpack1(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[1] = hip_pack(f);

        yuv.x = hip_unpack2(pY0.x);
        yuv.y = hip_unpack2(pU0.x);
        yuv.z = hip_unpack2(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[2] = hip_pack(f);

        yuv.x = hip_unpack3(pY0.x);
        yuv.y = hip_unpack3(pU0.x);
        yuv.z = hip_unpack3(pV0.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[3] = hip_pack(f);

        yuv.x = hip_unpack0(pY0.y);
        yuv.y = hip_unpack0(pU0.y);
        yuv.z = hip_unpack0(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[4] = hip_pack(f);

        yuv.x = hip_unpack1(pY0.y);
        yuv.y = hip_unpack1(pU0.y);
        yuv.z = hip_unpack1(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[5] = hip_pack(f);

        yuv.x = hip_unpack2(pY0.y);
        yuv.y = hip_unpack2(pU0.y);
        yuv.z = hip_unpack2(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[6] = hip_pack(f);

        yuv.x = hip_unpack3(pY0.y);
        yuv.y = hip_unpack3(pU0.y);
        yuv.z = hip_unpack3(pV0.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB0.data[7] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.x);
        yuv.y = hip_unpack0(pU1.x);
        yuv.z = hip_unpack0(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[0] = hip_pack(f);

        yuv.x = hip_unpack1(pY1.x);
        yuv.y = hip_unpack1(pU1.x);
        yuv.z = hip_unpack1(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[1] = hip_pack(f);

        yuv.x = hip_unpack2(pY1.x);
        yuv.y = hip_unpack2(pU1.x);
        yuv.z = hip_unpack2(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[2] = hip_pack(f);

        yuv.x = hip_unpack3(pY1.x);
        yuv.y = hip_unpack3(pU1.x);
        yuv.z = hip_unpack3(pV1.x);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[3] = hip_pack(f);

        yuv.x = hip_unpack0(pY1.y);
        yuv.y = hip_unpack0(pU1.y);
        yuv.z = hip_unpack0(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[4] = hip_pack(f);

        yuv.x = hip_unpack1(pY1.y);
        yuv.y = hip_unpack1(pU1.y);
        yuv.z = hip_unpack1(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[5] = hip_pack(f);

        yuv.x = hip_unpack2(pY1.y);
        yuv.y = hip_unpack2(pU1.y);
        yuv.z = hip_unpack2(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[6] = hip_pack(f);

        yuv.x = hip_unpack3(pY1.y);
        yuv.y = hip_unpack3(pU1.y);
        yuv.z = hip_unpack3(pV1.y);
        yuv.y -= 128.0f;
        yuv.z -= 128.0f;
        f.x = fmaf(cR.y, yuv.z, yuv.x);
        f.y = fmaf(cG.x, yuv.y, yuv.x);
        f.y = fmaf(cG.y, yuv.z, f.y);
        f.z = fmaf(cB.x, yuv.y, yuv.x);
        pRGB1.data[7] = hip_pack(f);

        *((d_uint8 *)(&pDstImage[RGB0Idx])) = pRGB0;
        *((d_uint8 *)(&pDstImage[RGB1Idx])) = pRGB1;
    }
}
int HipExec_ColorConvert_RGBX_NV21(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstImageStrideInBytesComp = dstImageStrideInBytes * 2;
    vx_uint32 srcLumaImageStrideInBytesComp = srcLumaImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_RGBX_NV21, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes, dstImageStrideInBytesComp,
                        (const uchar *)pHipSrcLumaImage, srcLumaImageStrideInBytes, (const uchar *)pHipSrcChromaImage, srcChromaImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcLumaImageStrideInBytesComp);

    return VX_SUCCESS;
}





// Group 5 - Destination IYUV, Source Packed

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_IYUV_RGB(uint dstWidth, uint dstHeight,
    uchar *pDstYImage, uint dstYImageStrideInBytes, uchar *pDstUImage, uint dstUImageStrideInBytes, uchar *pDstVImage, uint dstVImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcImageStrideInBytesComp, uint dstYImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint srcIdx0 = y * srcImageStrideInBytesComp + (x * 24);
        uint srcIdx1 = srcIdx0 + srcImageStrideInBytes;
        d_uint6 pRGB0 = *((d_uint6 *)(&pSrcImage[srcIdx0]));
        d_uint6 pRGB1 = *((d_uint6 *)(&pSrcImage[srcIdx1]));

        uint dstY0Idx  = y * dstYImageStrideInBytesComp + (x << 3);
        uint dstY1Idx  = dstY0Idx + dstYImageStrideInBytes;
        uint dstUIdx  = y * dstUImageStrideInBytes + (x << 2);
        uint dstVIdx  = y * dstVImageStrideInBytes + (x << 2);

        float4 f;

        float4 cY = make_float4(0.2126f, 0.7152f, 0.0722f, 0.0f);
        float3 cY3 = make_float3(0.2126f, 0.7152f, 0.0722f);
        uint2 pY0, pY1;
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[0]), hip_unpack1(pRGB0.data[0]), hip_unpack2(pRGB0.data[0])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack3(pRGB0.data[0]), hip_unpack0(pRGB0.data[1]), hip_unpack1(pRGB0.data[1])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack2(pRGB0.data[1]), hip_unpack3(pRGB0.data[1]), hip_unpack0(pRGB0.data[2])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack1(pRGB0.data[2]), hip_unpack2(pRGB0.data[2]), hip_unpack3(pRGB0.data[2])));
        pY0.x = hip_pack(f);
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[3]), hip_unpack1(pRGB0.data[3]), hip_unpack2(pRGB0.data[3])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack3(pRGB0.data[3]), hip_unpack0(pRGB0.data[4]), hip_unpack1(pRGB0.data[4])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack2(pRGB0.data[4]), hip_unpack3(pRGB0.data[4]), hip_unpack0(pRGB0.data[5])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack1(pRGB0.data[5]), hip_unpack2(pRGB0.data[5]), hip_unpack3(pRGB0.data[5])));
        pY0.y = hip_pack(f);
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[0]), hip_unpack1(pRGB1.data[0]), hip_unpack2(pRGB1.data[0])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack3(pRGB1.data[0]), hip_unpack0(pRGB1.data[1]), hip_unpack1(pRGB1.data[1])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack2(pRGB1.data[1]), hip_unpack3(pRGB1.data[1]), hip_unpack0(pRGB1.data[2])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack1(pRGB1.data[2]), hip_unpack2(pRGB1.data[2]), hip_unpack3(pRGB1.data[2])));
        pY1.x = hip_pack(f);
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[3]), hip_unpack1(pRGB1.data[3]), hip_unpack2(pRGB1.data[3])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack3(pRGB1.data[3]), hip_unpack0(pRGB1.data[4]), hip_unpack1(pRGB1.data[4])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack2(pRGB1.data[4]), hip_unpack3(pRGB1.data[4]), hip_unpack0(pRGB1.data[5])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack1(pRGB1.data[5]), hip_unpack2(pRGB1.data[5]), hip_unpack3(pRGB1.data[5])));
        pY1.y = hip_pack(f);

        float3 cU = make_float3(-0.1146f, -0.3854f, 0.5f);
        uint2 pU0, pU1;
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[0]), hip_unpack1(pRGB0.data[0]), hip_unpack2(pRGB0.data[0])));
        f.y = hip_dot3(cU, make_float3(hip_unpack2(pRGB0.data[1]), hip_unpack3(pRGB0.data[1]), hip_unpack0(pRGB0.data[2])));
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[3]), hip_unpack1(pRGB0.data[3]), hip_unpack2(pRGB0.data[3])));
        f.w = hip_dot3(cU, make_float3(hip_unpack2(pRGB0.data[4]), hip_unpack3(pRGB0.data[4]), hip_unpack0(pRGB0.data[5])));
        pU0.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cU, make_float3(hip_unpack3(pRGB0.data[0]), hip_unpack0(pRGB0.data[1]), hip_unpack1(pRGB0.data[1])));
        f.y = hip_dot3(cU, make_float3(hip_unpack1(pRGB0.data[2]), hip_unpack2(pRGB0.data[2]), hip_unpack3(pRGB0.data[2])));
        f.z = hip_dot3(cU, make_float3(hip_unpack3(pRGB0.data[3]), hip_unpack0(pRGB0.data[4]), hip_unpack1(pRGB0.data[4])));
        f.w = hip_dot3(cU, make_float3(hip_unpack1(pRGB0.data[5]), hip_unpack2(pRGB0.data[5]), hip_unpack3(pRGB0.data[5])));
        pU0.y = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[0]), hip_unpack1(pRGB1.data[0]), hip_unpack2(pRGB1.data[0])));
        f.y = hip_dot3(cU, make_float3(hip_unpack2(pRGB1.data[1]), hip_unpack3(pRGB1.data[1]), hip_unpack0(pRGB1.data[2])));
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[3]), hip_unpack1(pRGB1.data[3]), hip_unpack2(pRGB1.data[3])));
        f.w = hip_dot3(cU, make_float3(hip_unpack2(pRGB1.data[4]), hip_unpack3(pRGB1.data[4]), hip_unpack0(pRGB1.data[5])));
        pU1.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cU, make_float3(hip_unpack3(pRGB1.data[0]), hip_unpack0(pRGB1.data[1]), hip_unpack1(pRGB1.data[1])));
        f.y = hip_dot3(cU, make_float3(hip_unpack1(pRGB1.data[2]), hip_unpack2(pRGB1.data[2]), hip_unpack3(pRGB1.data[2])));
        f.z = hip_dot3(cU, make_float3(hip_unpack3(pRGB1.data[3]), hip_unpack0(pRGB1.data[4]), hip_unpack1(pRGB1.data[4])));
        f.w = hip_dot3(cU, make_float3(hip_unpack1(pRGB1.data[5]), hip_unpack2(pRGB1.data[5]), hip_unpack3(pRGB1.data[5])));
        pU1.y = hip_pack(f + (float4)(128));
        pU0.x = hip_lerp(pU0.x, pU0.y, 0);
        pU1.x = hip_lerp(pU1.x, pU1.y, 0);
        pU0.x = hip_lerp(pU0.x, pU1.x, 0);

        float3 cV = make_float3(0.5f, -0.4542f, -0.0458f);
        uint2 pV0, pV1;
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[0]), hip_unpack1(pRGB0.data[0]), hip_unpack2(pRGB0.data[0])));
        f.y = hip_dot3(cV, make_float3(hip_unpack2(pRGB0.data[1]), hip_unpack3(pRGB0.data[1]), hip_unpack0(pRGB0.data[2])));
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[3]), hip_unpack1(pRGB0.data[3]), hip_unpack2(pRGB0.data[3])));
        f.w = hip_dot3(cV, make_float3(hip_unpack2(pRGB0.data[4]), hip_unpack3(pRGB0.data[4]), hip_unpack0(pRGB0.data[5])));
        pV0.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cV, make_float3(hip_unpack3(pRGB0.data[0]), hip_unpack0(pRGB0.data[1]), hip_unpack1(pRGB0.data[1])));
        f.y = hip_dot3(cV, make_float3(hip_unpack1(pRGB0.data[2]), hip_unpack2(pRGB0.data[2]), hip_unpack3(pRGB0.data[2])));
        f.z = hip_dot3(cV, make_float3(hip_unpack3(pRGB0.data[3]), hip_unpack0(pRGB0.data[4]), hip_unpack1(pRGB0.data[4])));
        f.w = hip_dot3(cV, make_float3(hip_unpack1(pRGB0.data[5]), hip_unpack2(pRGB0.data[5]), hip_unpack3(pRGB0.data[5])));
        pV0.y = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[0]), hip_unpack1(pRGB1.data[0]), hip_unpack2(pRGB1.data[0])));
        f.y = hip_dot3(cV, make_float3(hip_unpack2(pRGB1.data[1]), hip_unpack3(pRGB1.data[1]), hip_unpack0(pRGB1.data[2])));
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[3]), hip_unpack1(pRGB1.data[3]), hip_unpack2(pRGB1.data[3])));
        f.w = hip_dot3(cV, make_float3(hip_unpack2(pRGB1.data[4]), hip_unpack3(pRGB1.data[4]), hip_unpack0(pRGB1.data[5])));
        pV1.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cV, make_float3(hip_unpack3(pRGB1.data[0]), hip_unpack0(pRGB1.data[1]), hip_unpack1(pRGB1.data[1])));
        f.y = hip_dot3(cV, make_float3(hip_unpack1(pRGB1.data[2]), hip_unpack2(pRGB1.data[2]), hip_unpack3(pRGB1.data[2])));
        f.z = hip_dot3(cV, make_float3(hip_unpack3(pRGB1.data[3]), hip_unpack0(pRGB1.data[4]), hip_unpack1(pRGB1.data[4])));
        f.w = hip_dot3(cV, make_float3(hip_unpack1(pRGB1.data[5]), hip_unpack2(pRGB1.data[5]), hip_unpack3(pRGB1.data[5])));
        pV1.y = hip_pack(f + (float4)(128));
        pV0.x = hip_lerp(pV0.x, pV0.y, 0);
        pV1.x = hip_lerp(pV1.x, pV1.y, 0);
        pV0.x = hip_lerp(pV0.x, pV1.x, 0);

        *((uint2 *)(&pDstYImage[dstY0Idx])) = pY0;
        *((uint2 *)(&pDstYImage[dstY1Idx])) = pY1;
        *((uint *)(&pDstUImage[dstUIdx])) = pU0.x;
        *((uint *)(&pDstVImage[dstVIdx])) = pV0.x;
    }
}
int HipExec_ColorConvert_IYUV_RGB(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 srcImageStrideInBytesComp = srcImageStrideInBytes * 2;
    vx_uint32 dstYImageStrideInBytesComp = dstYImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_IYUV_RGB, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstYImage, dstYImageStrideInBytes, (uchar *)pHipDstUImage, dstUImageStrideInBytes, (uchar *)pHipDstVImage, dstVImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcImageStrideInBytesComp, dstYImageStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_IYUV_RGBX(uint dstWidth, uint dstHeight,
    uchar *pDstYImage, uint dstYImageStrideInBytes, uchar *pDstUImage, uint dstUImageStrideInBytes, uchar *pDstVImage, uint dstVImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcImageStrideInBytesComp, uint dstYImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint srcIdx0 = y * srcImageStrideInBytesComp + (x << 5);
        uint srcIdx1 = srcIdx0 + srcImageStrideInBytes;
        d_uint8 pRGB0 = *((d_uint8 *)(&pSrcImage[srcIdx0]));
        d_uint8 pRGB1 = *((d_uint8 *)(&pSrcImage[srcIdx1]));

        uint dstY0Idx  = y * dstYImageStrideInBytesComp + (x << 3);
        uint dstY1Idx  = dstY0Idx + dstYImageStrideInBytes;
        uint dstUIdx  = y * dstUImageStrideInBytes + (x << 2);
        uint dstVIdx  = y * dstVImageStrideInBytes + (x << 2);

        float4 f;

        float4 cY = make_float4(0.2126f, 0.7152f, 0.0722f, 0.0f);
        float3 cY3 = make_float3(0.2126f, 0.7152f, 0.0722f);
        uint2 pY0, pY1;
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[0]), hip_unpack1(pRGB0.data[0]), hip_unpack2(pRGB0.data[0])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[1]), hip_unpack1(pRGB0.data[1]), hip_unpack2(pRGB0.data[1])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[2]), hip_unpack1(pRGB0.data[2]), hip_unpack2(pRGB0.data[2])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[3]), hip_unpack1(pRGB0.data[3]), hip_unpack2(pRGB0.data[3])));
        pY0.x = hip_pack(f);
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[4]), hip_unpack1(pRGB0.data[4]), hip_unpack2(pRGB0.data[4])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[5]), hip_unpack1(pRGB0.data[5]), hip_unpack2(pRGB0.data[5])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[6]), hip_unpack1(pRGB0.data[6]), hip_unpack2(pRGB0.data[6])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[7]), hip_unpack1(pRGB0.data[7]), hip_unpack2(pRGB0.data[7])));
        pY0.y = hip_pack(f);
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[0]), hip_unpack1(pRGB1.data[0]), hip_unpack2(pRGB1.data[0])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[1]), hip_unpack1(pRGB1.data[1]), hip_unpack2(pRGB1.data[1])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[2]), hip_unpack1(pRGB1.data[2]), hip_unpack2(pRGB1.data[2])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[3]), hip_unpack1(pRGB1.data[3]), hip_unpack2(pRGB1.data[3])));
        pY1.x = hip_pack(f);
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[4]), hip_unpack1(pRGB1.data[4]), hip_unpack2(pRGB1.data[4])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[5]), hip_unpack1(pRGB1.data[5]), hip_unpack2(pRGB1.data[5])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[6]), hip_unpack1(pRGB1.data[6]), hip_unpack2(pRGB1.data[6])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[7]), hip_unpack1(pRGB1.data[7]), hip_unpack2(pRGB1.data[7])));
        pY1.y = hip_pack(f);

        float3 cU = make_float3(-0.1146f, -0.3854f, 0.5f);
        uint2 pU0, pU1;
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[0]), hip_unpack1(pRGB0.data[0]), hip_unpack2(pRGB0.data[0])));
        f.y = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[2]), hip_unpack1(pRGB0.data[2]), hip_unpack2(pRGB0.data[2])));
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[4]), hip_unpack1(pRGB0.data[4]), hip_unpack2(pRGB0.data[4])));
        f.w = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[6]), hip_unpack1(pRGB0.data[6]), hip_unpack2(pRGB0.data[6])));
        pU0.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[1]), hip_unpack1(pRGB0.data[1]), hip_unpack2(pRGB0.data[1])));
        f.y = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[3]), hip_unpack1(pRGB0.data[3]), hip_unpack2(pRGB0.data[3])));
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[5]), hip_unpack1(pRGB0.data[5]), hip_unpack2(pRGB0.data[5])));
        f.w = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[7]), hip_unpack1(pRGB0.data[7]), hip_unpack2(pRGB0.data[7])));
        pU0.y = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[0]), hip_unpack1(pRGB1.data[0]), hip_unpack2(pRGB1.data[0])));
        f.y = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[2]), hip_unpack1(pRGB1.data[2]), hip_unpack2(pRGB1.data[2])));
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[4]), hip_unpack1(pRGB1.data[4]), hip_unpack2(pRGB1.data[4])));
        f.w = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[6]), hip_unpack1(pRGB1.data[6]), hip_unpack2(pRGB1.data[6])));
        pU1.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[1]), hip_unpack1(pRGB1.data[1]), hip_unpack2(pRGB1.data[1])));
        f.y = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[3]), hip_unpack1(pRGB1.data[3]), hip_unpack2(pRGB1.data[3])));
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[5]), hip_unpack1(pRGB1.data[5]), hip_unpack2(pRGB1.data[5])));
        f.w = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[7]), hip_unpack1(pRGB1.data[7]), hip_unpack2(pRGB1.data[7])));
        pU1.y = hip_pack(f + (float4)(128));
        pU0.x = hip_lerp(pU0.x, pU0.y, 0);
        pU1.x = hip_lerp(pU1.x, pU1.y, 0);
        pU0.x = hip_lerp(pU0.x, pU1.x, 0);

        float3 cV = make_float3(0.5f, -0.4542f, -0.0458f);
        uint2 pV0, pV1;
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[0]), hip_unpack1(pRGB0.data[0]), hip_unpack2(pRGB0.data[0])));
        f.y = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[2]), hip_unpack1(pRGB0.data[2]), hip_unpack2(pRGB0.data[2])));
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[4]), hip_unpack1(pRGB0.data[4]), hip_unpack2(pRGB0.data[4])));
        f.w = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[6]), hip_unpack1(pRGB0.data[6]), hip_unpack2(pRGB0.data[6])));
        pV0.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[1]), hip_unpack1(pRGB0.data[1]), hip_unpack2(pRGB0.data[1])));
        f.y = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[3]), hip_unpack1(pRGB0.data[3]), hip_unpack2(pRGB0.data[3])));
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[5]), hip_unpack1(pRGB0.data[5]), hip_unpack2(pRGB0.data[5])));
        f.w = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[7]), hip_unpack1(pRGB0.data[7]), hip_unpack2(pRGB0.data[7])));
        pV0.y = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[0]), hip_unpack1(pRGB1.data[0]), hip_unpack2(pRGB1.data[0])));
        f.y = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[2]), hip_unpack1(pRGB1.data[2]), hip_unpack2(pRGB1.data[2])));
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[4]), hip_unpack1(pRGB1.data[4]), hip_unpack2(pRGB1.data[4])));
        f.w = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[6]), hip_unpack1(pRGB1.data[6]), hip_unpack2(pRGB1.data[6])));
        pV1.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[1]), hip_unpack1(pRGB1.data[1]), hip_unpack2(pRGB1.data[1])));
        f.y = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[3]), hip_unpack1(pRGB1.data[3]), hip_unpack2(pRGB1.data[3])));
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[5]), hip_unpack1(pRGB1.data[5]), hip_unpack2(pRGB1.data[5])));
        f.w = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[7]), hip_unpack1(pRGB1.data[7]), hip_unpack2(pRGB1.data[7])));
        pV1.y = hip_pack(f + (float4)(128));
        pV0.x = hip_lerp(pV0.x, pV0.y, 0);
        pV1.x = hip_lerp(pV1.x, pV1.y, 0);
        pV0.x = hip_lerp(pV0.x, pV1.x, 0);

        *((uint2 *)(&pDstYImage[dstY0Idx])) = pY0;
        *((uint2 *)(&pDstYImage[dstY1Idx])) = pY1;
        *((uint *)(&pDstUImage[dstUIdx])) = pU0.x;
        *((uint *)(&pDstVImage[dstVIdx])) = pV0.x;
    }
}
int HipExec_ColorConvert_IYUV_RGBX(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 srcImageStrideInBytesComp = srcImageStrideInBytes * 2;
    vx_uint32 dstYImageStrideInBytesComp = dstYImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_IYUV_RGBX, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstYImage, dstYImageStrideInBytes, (uchar *)pHipDstUImage, dstUImageStrideInBytes, (uchar *)pHipDstVImage, dstVImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcImageStrideInBytesComp, dstYImageStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_IYUV_UYVY(uint dstWidth, uint dstHeight,
    uchar *pDstYImage, uint dstYImageStrideInBytes, uchar *pDstUImage, uint dstUImageStrideInBytes, uchar *pDstVImage, uint dstVImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcImageStrideInBytesComp, uint dstYImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint L0Idx = y * srcImageStrideInBytesComp + (x << 4);
        uint L1Idx = L0Idx + srcImageStrideInBytes;
        uint4 L0 = *((uint4 *)(&pSrcImage[L0Idx]));
        uint4 L1 = *((uint4 *)(&pSrcImage[L1Idx]));

        uint dstY0Idx = y * dstYImageStrideInBytesComp + (x << 3);
        uint dstY1Idx = dstY0Idx + dstYImageStrideInBytes;
        uint dstUIdx = y * dstUImageStrideInBytes + (x << 2);
        uint dstVIdx = y * dstVImageStrideInBytes + (x << 2);

        uint2 pY0, pY1;
        uint pU, pV;
        pY0.x = hip_pack(make_float4(hip_unpack1(L0.x), hip_unpack3(L0.x), hip_unpack1(L0.y), hip_unpack3(L0.y)));
        pY0.y = hip_pack(make_float4(hip_unpack1(L0.z), hip_unpack3(L0.z), hip_unpack1(L0.w), hip_unpack3(L0.w)));
        pY1.x = hip_pack(make_float4(hip_unpack1(L1.x), hip_unpack3(L1.x), hip_unpack1(L1.y), hip_unpack3(L1.y)));
        pY1.y = hip_pack(make_float4(hip_unpack1(L1.z), hip_unpack3(L1.z), hip_unpack1(L1.w), hip_unpack3(L1.w)));
        L0.x  = hip_lerp(L0.x, L1.x, 0);
        L0.y  = hip_lerp(L0.y, L1.y, 0);
        L0.z  = hip_lerp(L0.z, L1.z, 0);
        L0.w  = hip_lerp(L0.w, L1.w, 0);
        pU = hip_pack(make_float4(hip_unpack0(L0.x), hip_unpack0(L0.y), hip_unpack0(L0.z), hip_unpack0(L0.w)));
        pV = hip_pack(make_float4(hip_unpack2(L0.x), hip_unpack2(L0.y), hip_unpack2(L0.z), hip_unpack2(L0.w)));

        *((uint2 *)(&pDstYImage[dstY0Idx])) = pY0;
        *((uint2 *)(&pDstYImage[dstY1Idx])) = pY1;
        *((uint *)(&pDstUImage[dstUIdx])) = pU;
        *((uint *)(&pDstVImage[dstVIdx])) = pV;
    }
}
int HipExec_FormatConvert_IYUV_UYVY(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstYImageStrideInBytesComp = dstYImageStrideInBytes * 2;
    vx_uint32 srcImageStrideInBytesComp = srcImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_FormatConvert_IYUV_UYVY, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstYImage, dstYImageStrideInBytes, (uchar *)pHipDstUImage, dstUImageStrideInBytes, (uchar *)pHipDstVImage, dstVImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcImageStrideInBytesComp, dstYImageStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_IYUV_YUYV(uint dstWidth, uint dstHeight,
    uchar *pDstYImage, uint dstYImageStrideInBytes, uchar *pDstUImage, uint dstUImageStrideInBytes, uchar *pDstVImage, uint dstVImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcImageStrideInBytesComp, uint dstYImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint L0Idx = y * srcImageStrideInBytesComp + (x << 4);
        uint L1Idx = L0Idx + srcImageStrideInBytes;
        uint4 L0 = *((uint4 *)(&pSrcImage[L0Idx]));
        uint4 L1 = *((uint4 *)(&pSrcImage[L1Idx]));

        uint dstY0Idx = y * dstYImageStrideInBytesComp + (x << 3);
        uint dstY1Idx = dstY0Idx + dstYImageStrideInBytes;
        uint dstUIdx = y * dstUImageStrideInBytes + (x << 2);
        uint dstVIdx = y * dstVImageStrideInBytes + (x << 2);

        uint2 pY0, pY1;
        uint pU, pV;
        pY0.x = hip_pack(make_float4(hip_unpack0(L0.x), hip_unpack2(L0.x), hip_unpack0(L0.y), hip_unpack2(L0.y)));
        pY0.y = hip_pack(make_float4(hip_unpack0(L0.z), hip_unpack2(L0.z), hip_unpack0(L0.w), hip_unpack2(L0.w)));
        pY1.x = hip_pack(make_float4(hip_unpack0(L1.x), hip_unpack2(L1.x), hip_unpack0(L1.y), hip_unpack2(L1.y)));
        pY1.y = hip_pack(make_float4(hip_unpack0(L1.z), hip_unpack2(L1.z), hip_unpack0(L1.w), hip_unpack2(L1.w)));
        L0.x  = hip_lerp(L0.x, L1.x, 0);
        L0.y  = hip_lerp(L0.y, L1.y, 0);
        L0.z  = hip_lerp(L0.z, L1.z, 0);
        L0.w  = hip_lerp(L0.w, L1.w, 0);
        pU = hip_pack(make_float4(hip_unpack1(L0.x), hip_unpack1(L0.y), hip_unpack1(L0.z), hip_unpack1(L0.w)));
        pV = hip_pack(make_float4(hip_unpack3(L0.x), hip_unpack3(L0.y), hip_unpack3(L0.z), hip_unpack3(L0.w)));

        *((uint2 *)(&pDstYImage[dstY0Idx])) = pY0;
        *((uint2 *)(&pDstYImage[dstY1Idx])) = pY1;
        *((uint *)(&pDstUImage[dstUIdx])) = pU;
        *((uint *)(&pDstVImage[dstVIdx])) = pV;
    }
}
int HipExec_FormatConvert_IYUV_YUYV(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstYImageStrideInBytesComp = dstYImageStrideInBytes * 2;
    vx_uint32 srcImageStrideInBytesComp = srcImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_FormatConvert_IYUV_YUYV, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstYImage, dstYImageStrideInBytes, (uchar *)pHipDstUImage, dstUImageStrideInBytes, (uchar *)pHipDstVImage, dstVImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcImageStrideInBytesComp, dstYImageStrideInBytesComp);

    return VX_SUCCESS;
}





// Group 6 - Destination NV12, Source Packed

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_NV12_RGB(uint dstWidth, uint dstHeight,
    uchar *pDstImageLuma, uint dstImageLumaStrideInBytes, uchar *pDstImageChroma, uint dstImageChromaStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcImageStrideInBytesComp, uint dstImageLumaStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint srcIdx0 = y * srcImageStrideInBytesComp + (x * 24);
        uint srcIdx1 = srcIdx0 + srcImageStrideInBytes;
        d_uint6 pRGB0 = *((d_uint6 *)(&pSrcImage[srcIdx0]));
        d_uint6 pRGB1 = *((d_uint6 *)(&pSrcImage[srcIdx1]));

        uint dstY0Idx  = y * dstImageLumaStrideInBytesComp + (x << 3);
        uint dstY1Idx  = dstY0Idx + dstImageLumaStrideInBytes;
        uint dstUVIdx  = y * dstImageChromaStrideInBytes + (x << 3);

        float4 f;

        float4 cY = make_float4(0.2126f, 0.7152f, 0.0722f, 0.0f);
        float3 cY3 = make_float3(0.2126f, 0.7152f, 0.0722f);
        uint2 pY0, pY1;
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[0]), hip_unpack1(pRGB0.data[0]), hip_unpack2(pRGB0.data[0])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack3(pRGB0.data[0]), hip_unpack0(pRGB0.data[1]), hip_unpack1(pRGB0.data[1])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack2(pRGB0.data[1]), hip_unpack3(pRGB0.data[1]), hip_unpack0(pRGB0.data[2])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack1(pRGB0.data[2]), hip_unpack2(pRGB0.data[2]), hip_unpack3(pRGB0.data[2])));
        pY0.x = hip_pack(f);
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[3]), hip_unpack1(pRGB0.data[3]), hip_unpack2(pRGB0.data[3])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack3(pRGB0.data[3]), hip_unpack0(pRGB0.data[4]), hip_unpack1(pRGB0.data[4])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack2(pRGB0.data[4]), hip_unpack3(pRGB0.data[4]), hip_unpack0(pRGB0.data[5])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack1(pRGB0.data[5]), hip_unpack2(pRGB0.data[5]), hip_unpack3(pRGB0.data[5])));
        pY0.y = hip_pack(f);
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[0]), hip_unpack1(pRGB1.data[0]), hip_unpack2(pRGB1.data[0])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack3(pRGB1.data[0]), hip_unpack0(pRGB1.data[1]), hip_unpack1(pRGB1.data[1])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack2(pRGB1.data[1]), hip_unpack3(pRGB1.data[1]), hip_unpack0(pRGB1.data[2])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack1(pRGB1.data[2]), hip_unpack2(pRGB1.data[2]), hip_unpack3(pRGB1.data[2])));
        pY1.x = hip_pack(f);
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[3]), hip_unpack1(pRGB1.data[3]), hip_unpack2(pRGB1.data[3])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack3(pRGB1.data[3]), hip_unpack0(pRGB1.data[4]), hip_unpack1(pRGB1.data[4])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack2(pRGB1.data[4]), hip_unpack3(pRGB1.data[4]), hip_unpack0(pRGB1.data[5])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack1(pRGB1.data[5]), hip_unpack2(pRGB1.data[5]), hip_unpack3(pRGB1.data[5])));
        pY1.y = hip_pack(f);

        float3 cU = make_float3(-0.1146f, -0.3854f, 0.5f);
        uint2 pU0, pU1;
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[0]), hip_unpack1(pRGB0.data[0]), hip_unpack2(pRGB0.data[0])));
        f.y = hip_dot3(cU, make_float3(hip_unpack2(pRGB0.data[1]), hip_unpack3(pRGB0.data[1]), hip_unpack0(pRGB0.data[2])));
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[3]), hip_unpack1(pRGB0.data[3]), hip_unpack2(pRGB0.data[3])));
        f.w = hip_dot3(cU, make_float3(hip_unpack2(pRGB0.data[4]), hip_unpack3(pRGB0.data[4]), hip_unpack0(pRGB0.data[5])));
        pU0.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cU, make_float3(hip_unpack3(pRGB0.data[0]), hip_unpack0(pRGB0.data[1]), hip_unpack1(pRGB0.data[1])));
        f.y = hip_dot3(cU, make_float3(hip_unpack1(pRGB0.data[2]), hip_unpack2(pRGB0.data[2]), hip_unpack3(pRGB0.data[2])));
        f.z = hip_dot3(cU, make_float3(hip_unpack3(pRGB0.data[3]), hip_unpack0(pRGB0.data[4]), hip_unpack1(pRGB0.data[4])));
        f.w = hip_dot3(cU, make_float3(hip_unpack1(pRGB0.data[5]), hip_unpack2(pRGB0.data[5]), hip_unpack3(pRGB0.data[5])));
        pU0.y = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[0]), hip_unpack1(pRGB1.data[0]), hip_unpack2(pRGB1.data[0])));
        f.y = hip_dot3(cU, make_float3(hip_unpack2(pRGB1.data[1]), hip_unpack3(pRGB1.data[1]), hip_unpack0(pRGB1.data[2])));
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[3]), hip_unpack1(pRGB1.data[3]), hip_unpack2(pRGB1.data[3])));
        f.w = hip_dot3(cU, make_float3(hip_unpack2(pRGB1.data[4]), hip_unpack3(pRGB1.data[4]), hip_unpack0(pRGB1.data[5])));
        pU1.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cU, make_float3(hip_unpack3(pRGB1.data[0]), hip_unpack0(pRGB1.data[1]), hip_unpack1(pRGB1.data[1])));
        f.y = hip_dot3(cU, make_float3(hip_unpack1(pRGB1.data[2]), hip_unpack2(pRGB1.data[2]), hip_unpack3(pRGB1.data[2])));
        f.z = hip_dot3(cU, make_float3(hip_unpack3(pRGB1.data[3]), hip_unpack0(pRGB1.data[4]), hip_unpack1(pRGB1.data[4])));
        f.w = hip_dot3(cU, make_float3(hip_unpack1(pRGB1.data[5]), hip_unpack2(pRGB1.data[5]), hip_unpack3(pRGB1.data[5])));
        pU1.y = hip_pack(f + (float4)(128));
        pU0.x = hip_lerp(pU0.x, pU0.y, 0);
        pU1.x = hip_lerp(pU1.x, pU1.y, 0);
        pU0.x = hip_lerp(pU0.x, pU1.x, 0);

        float3 cV = make_float3(0.5f, -0.4542f, -0.0458f);
        uint2 pV0, pV1;
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[0]), hip_unpack1(pRGB0.data[0]), hip_unpack2(pRGB0.data[0])));
        f.y = hip_dot3(cV, make_float3(hip_unpack2(pRGB0.data[1]), hip_unpack3(pRGB0.data[1]), hip_unpack0(pRGB0.data[2])));
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[3]), hip_unpack1(pRGB0.data[3]), hip_unpack2(pRGB0.data[3])));
        f.w = hip_dot3(cV, make_float3(hip_unpack2(pRGB0.data[4]), hip_unpack3(pRGB0.data[4]), hip_unpack0(pRGB0.data[5])));
        pV0.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cV, make_float3(hip_unpack3(pRGB0.data[0]), hip_unpack0(pRGB0.data[1]), hip_unpack1(pRGB0.data[1])));
        f.y = hip_dot3(cV, make_float3(hip_unpack1(pRGB0.data[2]), hip_unpack2(pRGB0.data[2]), hip_unpack3(pRGB0.data[2])));
        f.z = hip_dot3(cV, make_float3(hip_unpack3(pRGB0.data[3]), hip_unpack0(pRGB0.data[4]), hip_unpack1(pRGB0.data[4])));
        f.w = hip_dot3(cV, make_float3(hip_unpack1(pRGB0.data[5]), hip_unpack2(pRGB0.data[5]), hip_unpack3(pRGB0.data[5])));
        pV0.y = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[0]), hip_unpack1(pRGB1.data[0]), hip_unpack2(pRGB1.data[0])));
        f.y = hip_dot3(cV, make_float3(hip_unpack2(pRGB1.data[1]), hip_unpack3(pRGB1.data[1]), hip_unpack0(pRGB1.data[2])));
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[3]), hip_unpack1(pRGB1.data[3]), hip_unpack2(pRGB1.data[3])));
        f.w = hip_dot3(cV, make_float3(hip_unpack2(pRGB1.data[4]), hip_unpack3(pRGB1.data[4]), hip_unpack0(pRGB1.data[5])));
        pV1.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cV, make_float3(hip_unpack3(pRGB1.data[0]), hip_unpack0(pRGB1.data[1]), hip_unpack1(pRGB1.data[1])));
        f.y = hip_dot3(cV, make_float3(hip_unpack1(pRGB1.data[2]), hip_unpack2(pRGB1.data[2]), hip_unpack3(pRGB1.data[2])));
        f.z = hip_dot3(cV, make_float3(hip_unpack3(pRGB1.data[3]), hip_unpack0(pRGB1.data[4]), hip_unpack1(pRGB1.data[4])));
        f.w = hip_dot3(cV, make_float3(hip_unpack1(pRGB1.data[5]), hip_unpack2(pRGB1.data[5]), hip_unpack3(pRGB1.data[5])));
        pV1.y = hip_pack(f + (float4)(128));
        pV0.x = hip_lerp(pV0.x, pV0.y, 0);
        pV1.x = hip_lerp(pV1.x, pV1.y, 0);
        pV0.x = hip_lerp(pV0.x, pV1.x, 0);

        uint2 pUV;
        f.x = hip_unpack0(pU0.x);
        f.y = hip_unpack0(pV0.x);
        f.z = hip_unpack1(pU0.x);
        f.w = hip_unpack1(pV0.x);
        pUV.x = hip_pack(f);
        f.x = hip_unpack2(pU0.x);
        f.y = hip_unpack2(pV0.x);
        f.z = hip_unpack3(pU0.x);
        f.w = hip_unpack3(pV0.x);
        pUV.y = hip_pack(f);

        *((uint2 *)(&pDstImageLuma[dstY0Idx])) = pY0;
        *((uint2 *)(&pDstImageLuma[dstY1Idx])) = pY1;
        *((uint2 *)(&pDstImageChroma[dstUVIdx])) = pUV;
    }
}
int HipExec_ColorConvert_NV12_RGB(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImageLuma, vx_uint32 dstImageLumaStrideInBytes,
    vx_uint8 *pHipDstImageChroma, vx_uint32 dstImageChromaStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 srcImageStrideInBytesComp = srcImageStrideInBytes * 2;
    vx_uint32 dstImageLumaStrideInBytesComp = dstImageLumaStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_NV12_RGB, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImageLuma, dstImageLumaStrideInBytes, (uchar *)pHipDstImageChroma, dstImageChromaStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcImageStrideInBytesComp, dstImageLumaStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_NV12_RGBX(uint dstWidth, uint dstHeight,
    uchar *pDstImageLuma, uint dstImageLumaStrideInBytes, uchar *pDstImageChroma, uint dstImageChromaStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcImageStrideInBytesComp, uint dstImageLumaStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint srcIdx0 = y * srcImageStrideInBytesComp + (x << 5);
        uint srcIdx1 = srcIdx0 + srcImageStrideInBytes;
        d_uint8 pRGB0 = *((d_uint8 *)(&pSrcImage[srcIdx0]));
        d_uint8 pRGB1 = *((d_uint8 *)(&pSrcImage[srcIdx1]));

        uint dstY0Idx  = y * dstImageLumaStrideInBytesComp + (x << 3);
        uint dstY1Idx  = dstY0Idx + dstImageLumaStrideInBytes;
        uint dstUVIdx  = y * dstImageChromaStrideInBytes + (x << 3);

        float4 f;

        float4 cY = make_float4(0.2126f, 0.7152f, 0.0722f, 0.0f);
        float3 cY3 = make_float3(0.2126f, 0.7152f, 0.0722f);
        uint2 pY0, pY1;
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[0]), hip_unpack1(pRGB0.data[0]), hip_unpack2(pRGB0.data[0])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[1]), hip_unpack1(pRGB0.data[1]), hip_unpack2(pRGB0.data[1])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[2]), hip_unpack1(pRGB0.data[2]), hip_unpack2(pRGB0.data[2])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[3]), hip_unpack1(pRGB0.data[3]), hip_unpack2(pRGB0.data[3])));
        pY0.x = hip_pack(f);
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[4]), hip_unpack1(pRGB0.data[4]), hip_unpack2(pRGB0.data[4])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[5]), hip_unpack1(pRGB0.data[5]), hip_unpack2(pRGB0.data[5])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[6]), hip_unpack1(pRGB0.data[6]), hip_unpack2(pRGB0.data[6])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB0.data[7]), hip_unpack1(pRGB0.data[7]), hip_unpack2(pRGB0.data[7])));
        pY0.y = hip_pack(f);
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[0]), hip_unpack1(pRGB1.data[0]), hip_unpack2(pRGB1.data[0])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[1]), hip_unpack1(pRGB1.data[1]), hip_unpack2(pRGB1.data[1])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[2]), hip_unpack1(pRGB1.data[2]), hip_unpack2(pRGB1.data[2])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[3]), hip_unpack1(pRGB1.data[3]), hip_unpack2(pRGB1.data[3])));
        pY1.x = hip_pack(f);
        f.x = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[4]), hip_unpack1(pRGB1.data[4]), hip_unpack2(pRGB1.data[4])));
        f.y = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[5]), hip_unpack1(pRGB1.data[5]), hip_unpack2(pRGB1.data[5])));
        f.z = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[6]), hip_unpack1(pRGB1.data[6]), hip_unpack2(pRGB1.data[6])));
        f.w = cY.w + hip_dot3(cY3, make_float3(hip_unpack0(pRGB1.data[7]), hip_unpack1(pRGB1.data[7]), hip_unpack2(pRGB1.data[7])));
        pY1.y = hip_pack(f);

        float3 cU = make_float3(-0.1146f, -0.3854f, 0.5f);
        uint2 pU0, pU1;
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[0]), hip_unpack1(pRGB0.data[0]), hip_unpack2(pRGB0.data[0])));
        f.y = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[2]), hip_unpack1(pRGB0.data[2]), hip_unpack2(pRGB0.data[2])));
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[4]), hip_unpack1(pRGB0.data[4]), hip_unpack2(pRGB0.data[4])));
        f.w = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[6]), hip_unpack1(pRGB0.data[6]), hip_unpack2(pRGB0.data[6])));
        pU0.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[1]), hip_unpack1(pRGB0.data[1]), hip_unpack2(pRGB0.data[1])));
        f.y = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[3]), hip_unpack1(pRGB0.data[3]), hip_unpack2(pRGB0.data[3])));
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[5]), hip_unpack1(pRGB0.data[5]), hip_unpack2(pRGB0.data[5])));
        f.w = hip_dot3(cU, make_float3(hip_unpack0(pRGB0.data[7]), hip_unpack1(pRGB0.data[7]), hip_unpack2(pRGB0.data[7])));
        pU0.y = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[0]), hip_unpack1(pRGB1.data[0]), hip_unpack2(pRGB1.data[0])));
        f.y = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[2]), hip_unpack1(pRGB1.data[2]), hip_unpack2(pRGB1.data[2])));
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[4]), hip_unpack1(pRGB1.data[4]), hip_unpack2(pRGB1.data[4])));
        f.w = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[6]), hip_unpack1(pRGB1.data[6]), hip_unpack2(pRGB1.data[6])));
        pU1.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[1]), hip_unpack1(pRGB1.data[1]), hip_unpack2(pRGB1.data[1])));
        f.y = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[3]), hip_unpack1(pRGB1.data[3]), hip_unpack2(pRGB1.data[3])));
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[5]), hip_unpack1(pRGB1.data[5]), hip_unpack2(pRGB1.data[5])));
        f.w = hip_dot3(cU, make_float3(hip_unpack0(pRGB1.data[7]), hip_unpack1(pRGB1.data[7]), hip_unpack2(pRGB1.data[7])));
        pU1.y = hip_pack(f + (float4)(128));
        pU0.x = hip_lerp(pU0.x, pU0.y, 0);
        pU1.x = hip_lerp(pU1.x, pU1.y, 0);
        pU0.x = hip_lerp(pU0.x, pU1.x, 0);

        float3 cV = make_float3(0.5f, -0.4542f, -0.0458f);
        uint2 pV0, pV1;
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[0]), hip_unpack1(pRGB0.data[0]), hip_unpack2(pRGB0.data[0])));
        f.y = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[2]), hip_unpack1(pRGB0.data[2]), hip_unpack2(pRGB0.data[2])));
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[4]), hip_unpack1(pRGB0.data[4]), hip_unpack2(pRGB0.data[4])));
        f.w = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[6]), hip_unpack1(pRGB0.data[6]), hip_unpack2(pRGB0.data[6])));
        pV0.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[1]), hip_unpack1(pRGB0.data[1]), hip_unpack2(pRGB0.data[1])));
        f.y = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[3]), hip_unpack1(pRGB0.data[3]), hip_unpack2(pRGB0.data[3])));
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[5]), hip_unpack1(pRGB0.data[5]), hip_unpack2(pRGB0.data[5])));
        f.w = hip_dot3(cV, make_float3(hip_unpack0(pRGB0.data[7]), hip_unpack1(pRGB0.data[7]), hip_unpack2(pRGB0.data[7])));
        pV0.y = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[0]), hip_unpack1(pRGB1.data[0]), hip_unpack2(pRGB1.data[0])));
        f.y = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[2]), hip_unpack1(pRGB1.data[2]), hip_unpack2(pRGB1.data[2])));
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[4]), hip_unpack1(pRGB1.data[4]), hip_unpack2(pRGB1.data[4])));
        f.w = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[6]), hip_unpack1(pRGB1.data[6]), hip_unpack2(pRGB1.data[6])));
        pV1.x = hip_pack(f + (float4)(128));
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[1]), hip_unpack1(pRGB1.data[1]), hip_unpack2(pRGB1.data[1])));
        f.y = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[3]), hip_unpack1(pRGB1.data[3]), hip_unpack2(pRGB1.data[3])));
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[5]), hip_unpack1(pRGB1.data[5]), hip_unpack2(pRGB1.data[5])));
        f.w = hip_dot3(cV, make_float3(hip_unpack0(pRGB1.data[7]), hip_unpack1(pRGB1.data[7]), hip_unpack2(pRGB1.data[7])));
        pV1.y = hip_pack(f + (float4)(128));
        pV0.x = hip_lerp(pV0.x, pV0.y, 0);
        pV1.x = hip_lerp(pV1.x, pV1.y, 0);
        pV0.x = hip_lerp(pV0.x, pV1.x, 0);

        uint2 pUV;
        f.x = hip_unpack0(pU0.x);
        f.y = hip_unpack0(pV0.x);
        f.z = hip_unpack1(pU0.x);
        f.w = hip_unpack1(pV0.x);
        pUV.x = hip_pack(f);
        f.x = hip_unpack2(pU0.x);
        f.y = hip_unpack2(pV0.x);
        f.z = hip_unpack3(pU0.x);
        f.w = hip_unpack3(pV0.x);
        pUV.y = hip_pack(f);

        *((uint2 *)(&pDstImageLuma[dstY0Idx])) = pY0;
        *((uint2 *)(&pDstImageLuma[dstY1Idx])) = pY1;
        *((uint2 *)(&pDstImageChroma[dstUVIdx])) = pUV;
    }
}
int HipExec_ColorConvert_NV12_RGBX(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImageLuma, vx_uint32 dstImageLumaStrideInBytes,
    vx_uint8 *pHipDstImageChroma, vx_uint32 dstImageChromaStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 srcImageStrideInBytesComp = srcImageStrideInBytes * 2;
    vx_uint32 dstImageLumaStrideInBytesComp = dstImageLumaStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ColorConvert_NV12_RGBX, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImageLuma, dstImageLumaStrideInBytes, (uchar *)pHipDstImageChroma, dstImageChromaStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcImageStrideInBytesComp, dstImageLumaStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_NV12_UYVY(uint dstWidth, uint dstHeight,
    uchar *pDstImageLuma, uint dstImageLumaStrideInBytes, uchar *pDstImageChroma, uint dstImageChromaStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcImageStrideInBytesComp, uint dstImageLumaStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint L0Idx = y * srcImageStrideInBytesComp + (x << 4);
        uint L1Idx = L0Idx + srcImageStrideInBytes;
        uint4 L0 = *((uint4 *)(&pSrcImage[L0Idx]));
        uint4 L1 = *((uint4 *)(&pSrcImage[L1Idx]));

        uint dstY0Idx = y * dstImageLumaStrideInBytesComp + (x << 3);
        uint dstY1Idx = dstY0Idx + dstImageLumaStrideInBytes;
        uint dstUVIdx = y * dstImageChromaStrideInBytes + (x << 3);

        uint2 pY0, pY1, pUV;
        pY0.x = hip_pack(make_float4(hip_unpack1(L0.x), hip_unpack3(L0.x), hip_unpack1(L0.y), hip_unpack3(L0.y)));
        pY0.y = hip_pack(make_float4(hip_unpack1(L0.z), hip_unpack3(L0.z), hip_unpack1(L0.w), hip_unpack3(L0.w)));
        pY1.x = hip_pack(make_float4(hip_unpack1(L1.x), hip_unpack3(L1.x), hip_unpack1(L1.y), hip_unpack3(L1.y)));
        pY1.y = hip_pack(make_float4(hip_unpack1(L1.z), hip_unpack3(L1.z), hip_unpack1(L1.w), hip_unpack3(L1.w)));
        L0.x  = hip_lerp(L0.x, L1.x, 0);
        L0.y  = hip_lerp(L0.y, L1.y, 0);
        L0.z  = hip_lerp(L0.z, L1.z, 0);
        L0.w  = hip_lerp(L0.w, L1.w, 0);
        pUV.x = hip_pack(make_float4(hip_unpack0(L0.x), hip_unpack2(L0.x), hip_unpack0(L0.y), hip_unpack2(L0.y)));
        pUV.y = hip_pack(make_float4(hip_unpack0(L0.z), hip_unpack2(L0.z), hip_unpack0(L0.w), hip_unpack2(L0.w)));

        *((uint2 *)(&pDstImageLuma[dstY0Idx])) = pY0;
        *((uint2 *)(&pDstImageLuma[dstY1Idx])) = pY1;
        *((uint2 *)(&pDstImageChroma[dstUVIdx])) = pUV;
    }
}
int HipExec_FormatConvert_NV12_UYVY(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImageLuma, vx_uint32 dstImageLumaStrideInBytes,
    vx_uint8 *pHipDstImageChroma, vx_uint32 dstImageChromaStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 srcImageStrideInBytesComp = srcImageStrideInBytes * 2;
    vx_uint32 dstImageLumaStrideInBytesComp = dstImageLumaStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_FormatConvert_NV12_UYVY, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImageLuma, dstImageLumaStrideInBytes, (uchar *)pHipDstImageChroma, dstImageChromaStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcImageStrideInBytesComp, dstImageLumaStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_NV12_YUYV(uint dstWidth, uint dstHeight,
    uchar *pDstImageLuma, uint dstImageLumaStrideInBytes, uchar *pDstImageChroma, uint dstImageChromaStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcImageStrideInBytesComp, uint dstImageLumaStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint L0Idx = y * srcImageStrideInBytesComp + (x << 4);
        uint L1Idx = L0Idx + srcImageStrideInBytes;
        uint4 L0 = *((uint4 *)(&pSrcImage[L0Idx]));
        uint4 L1 = *((uint4 *)(&pSrcImage[L1Idx]));

        uint dstY0Idx = y * dstImageLumaStrideInBytesComp + (x << 3);
        uint dstY1Idx = dstY0Idx + dstImageLumaStrideInBytes;
        uint dstUVIdx = y * dstImageChromaStrideInBytes + (x << 3);

        uint2 pY0, pY1, pUV;
        pY0.x = hip_pack(make_float4(hip_unpack0(L0.x), hip_unpack2(L0.x), hip_unpack0(L0.y), hip_unpack2(L0.y)));
        pY0.y = hip_pack(make_float4(hip_unpack0(L0.z), hip_unpack2(L0.z), hip_unpack0(L0.w), hip_unpack2(L0.w)));
        pY1.x = hip_pack(make_float4(hip_unpack0(L1.x), hip_unpack2(L1.x), hip_unpack0(L1.y), hip_unpack2(L1.y)));
        pY1.y = hip_pack(make_float4(hip_unpack0(L1.z), hip_unpack2(L1.z), hip_unpack0(L1.w), hip_unpack2(L1.w)));
        L0.x  = hip_lerp(L0.x, L1.x, 0);
        L0.y  = hip_lerp(L0.y, L1.y, 0);
        L0.z  = hip_lerp(L0.z, L1.z, 0);
        L0.w  = hip_lerp(L0.w, L1.w, 0);
        pUV.x = hip_pack(make_float4(hip_unpack1(L0.x), hip_unpack3(L0.x), hip_unpack1(L0.y), hip_unpack3(L0.y)));
        pUV.y = hip_pack(make_float4(hip_unpack1(L0.z), hip_unpack3(L0.z), hip_unpack1(L0.w), hip_unpack3(L0.w)));

        *((uint2 *)(&pDstImageLuma[dstY0Idx])) = pY0;
        *((uint2 *)(&pDstImageLuma[dstY1Idx])) = pY1;
        *((uint2 *)(&pDstImageChroma[dstUVIdx])) = pUV;
    }
}
int HipExec_FormatConvert_NV12_YUYV(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImageLuma, vx_uint32 dstImageLumaStrideInBytes,
    vx_uint8 *pHipDstImageChroma, vx_uint32 dstImageChromaStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 srcImageStrideInBytesComp = srcImageStrideInBytes * 2;
    vx_uint32 dstImageLumaStrideInBytesComp = dstImageLumaStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_FormatConvert_NV12_YUYV, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImageLuma, dstImageLumaStrideInBytes, (uchar *)pHipDstImageChroma, dstImageChromaStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcImageStrideInBytesComp, dstImageLumaStrideInBytesComp);

    return VX_SUCCESS;
}





// Group 7 - Destination YUV4, Source Packed

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_YUV4_RGB(uint dstWidth, uint dstHeight,
    uchar *pDstYImage, uint dstYImageStrideInBytes, uchar *pDstUImage, uint dstUImageStrideInBytes, uchar *pDstVImage, uint dstVImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if( x < dstWidth && y < dstHeight) {
        uint srcIdx = y * srcImageStrideInBytes + (x * 3);
        d_uint6 pRGB = *((d_uint6 *)(&pSrcImage[srcIdx]));

        uint dstYIdx  = y * dstYImageStrideInBytes + x;
        uint dstUIdx  = y * dstUImageStrideInBytes + x;
        uint dstVIdx  = y * dstVImageStrideInBytes + x;

        float4 f;
        uint2 yChannel, uChannel, vChannel;

        float3 cY = make_float3(0.2126f, 0.7152f, 0.0722f);
        f.x = hip_dot3(cY, make_float3(hip_unpack0(pRGB.data[0]), hip_unpack1(pRGB.data[0]), hip_unpack2(pRGB.data[0])));
        f.y = hip_dot3(cY, make_float3(hip_unpack3(pRGB.data[0]), hip_unpack0(pRGB.data[1]), hip_unpack1(pRGB.data[1])));
        f.z = hip_dot3(cY, make_float3(hip_unpack2(pRGB.data[1]), hip_unpack3(pRGB.data[1]), hip_unpack0(pRGB.data[2])));
        f.w = hip_dot3(cY, make_float3(hip_unpack1(pRGB.data[2]), hip_unpack2(pRGB.data[2]), hip_unpack3(pRGB.data[2])));
        yChannel.x = hip_pack(f);
        f.x = hip_dot3(cY, make_float3(hip_unpack0(pRGB.data[3]), hip_unpack1(pRGB.data[3]), hip_unpack2(pRGB.data[3])));
        f.y = hip_dot3(cY, make_float3(hip_unpack3(pRGB.data[3]), hip_unpack0(pRGB.data[4]), hip_unpack1(pRGB.data[4])));
        f.z = hip_dot3(cY, make_float3(hip_unpack2(pRGB.data[4]), hip_unpack3(pRGB.data[4]), hip_unpack0(pRGB.data[5])));
        f.w = hip_dot3(cY, make_float3(hip_unpack1(pRGB.data[5]), hip_unpack2(pRGB.data[5]), hip_unpack3(pRGB.data[5])));
        yChannel.y = hip_pack(f);

        float3 cU = make_float3(-0.1146f, -0.3854f, 0.5f);
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB.data[0]), hip_unpack1(pRGB.data[0]), hip_unpack2(pRGB.data[0]))) + 128.0f;
        f.y = hip_dot3(cU, make_float3(hip_unpack3(pRGB.data[0]), hip_unpack0(pRGB.data[1]), hip_unpack1(pRGB.data[1]))) + 128.0f;
        f.z = hip_dot3(cU, make_float3(hip_unpack2(pRGB.data[1]), hip_unpack3(pRGB.data[1]), hip_unpack0(pRGB.data[2]))) + 128.0f;
        f.w = hip_dot3(cU, make_float3(hip_unpack1(pRGB.data[2]), hip_unpack2(pRGB.data[2]), hip_unpack3(pRGB.data[2]))) + 128.0f;
        uChannel.x = hip_pack(f);
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB.data[3]), hip_unpack1(pRGB.data[3]), hip_unpack2(pRGB.data[3]))) + 128.0f;
        f.y = hip_dot3(cU, make_float3(hip_unpack3(pRGB.data[3]), hip_unpack0(pRGB.data[4]), hip_unpack1(pRGB.data[4]))) + 128.0f;
        f.z = hip_dot3(cU, make_float3(hip_unpack2(pRGB.data[4]), hip_unpack3(pRGB.data[4]), hip_unpack0(pRGB.data[5]))) + 128.0f;
        f.w = hip_dot3(cU, make_float3(hip_unpack1(pRGB.data[5]), hip_unpack2(pRGB.data[5]), hip_unpack3(pRGB.data[5]))) + 128.0f;
        uChannel.y = hip_pack(f);

        float3 cV = make_float3(0.5f, -0.4542f, -0.0458f);
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB.data[0]), hip_unpack1(pRGB.data[0]), hip_unpack2(pRGB.data[0]))) + 128.0f;
        f.y = hip_dot3(cV, make_float3(hip_unpack3(pRGB.data[0]), hip_unpack0(pRGB.data[1]), hip_unpack1(pRGB.data[1]))) + 128.0f;
        f.z = hip_dot3(cV, make_float3(hip_unpack2(pRGB.data[1]), hip_unpack3(pRGB.data[1]), hip_unpack0(pRGB.data[2]))) + 128.0f;
        f.w = hip_dot3(cV, make_float3(hip_unpack1(pRGB.data[2]), hip_unpack2(pRGB.data[2]), hip_unpack3(pRGB.data[2]))) + 128.0f;
        vChannel.x = hip_pack(f);
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB.data[3]), hip_unpack1(pRGB.data[3]), hip_unpack2(pRGB.data[3]))) + 128.0f;
        f.y = hip_dot3(cV, make_float3(hip_unpack3(pRGB.data[3]), hip_unpack0(pRGB.data[4]), hip_unpack1(pRGB.data[4]))) + 128.0f;
        f.z = hip_dot3(cV, make_float3(hip_unpack2(pRGB.data[4]), hip_unpack3(pRGB.data[4]), hip_unpack0(pRGB.data[5]))) + 128.0f;
        f.w = hip_dot3(cV, make_float3(hip_unpack1(pRGB.data[5]), hip_unpack2(pRGB.data[5]), hip_unpack3(pRGB.data[5]))) + 128.0f;
        vChannel.y = hip_pack(f);

        *((uint2 *)(&pDstYImage[dstYIdx])) = yChannel;
        *((uint2 *)(&pDstUImage[dstUIdx])) = uChannel;
        *((uint2 *)(&pDstVImage[dstVIdx])) = vChannel;
    }
}
int HipExec_ColorConvert_YUV4_RGB(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_YUV4_RGB, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstYImage, dstYImageStrideInBytes, (uchar *)pHipDstUImage, dstUImageStrideInBytes, (uchar *)pHipDstVImage, dstVImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ColorConvert_YUV4_RGBX(uint dstWidth, uint dstHeight,
    uchar *pDstYImage, uint dstYImageStrideInBytes, uchar *pDstUImage, uint dstUImageStrideInBytes, uchar *pDstVImage, uint dstVImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes) {

    int x = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 8;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if( x < dstWidth && y < dstHeight) {
        uint srcIdx = y * srcImageStrideInBytes + (x << 2);
        d_uint8 pRGB = *((d_uint8 *)(&pSrcImage[srcIdx]));

        uint dstYIdx  = y * dstYImageStrideInBytes + x;
        uint dstUIdx  = y * dstUImageStrideInBytes + x;
        uint dstVIdx  = y * dstVImageStrideInBytes + x;

        float4 f;
        uint2 yChannel, uChannel, vChannel;

        float3 cY = make_float3(0.2126f, 0.7152f, 0.0722f);
        f.x = hip_dot3(cY, make_float3(hip_unpack0(pRGB.data[0]), hip_unpack1(pRGB.data[0]), hip_unpack2(pRGB.data[0])));
        f.y = hip_dot3(cY, make_float3(hip_unpack0(pRGB.data[1]), hip_unpack1(pRGB.data[1]), hip_unpack2(pRGB.data[1])));
        f.z = hip_dot3(cY, make_float3(hip_unpack0(pRGB.data[2]), hip_unpack1(pRGB.data[2]), hip_unpack2(pRGB.data[2])));
        f.w = hip_dot3(cY, make_float3(hip_unpack0(pRGB.data[3]), hip_unpack1(pRGB.data[3]), hip_unpack2(pRGB.data[3])));
        yChannel.x = hip_pack(f);
        f.x = hip_dot3(cY, make_float3(hip_unpack0(pRGB.data[4]), hip_unpack1(pRGB.data[4]), hip_unpack2(pRGB.data[4])));
        f.y = hip_dot3(cY, make_float3(hip_unpack0(pRGB.data[5]), hip_unpack1(pRGB.data[5]), hip_unpack2(pRGB.data[5])));
        f.z = hip_dot3(cY, make_float3(hip_unpack0(pRGB.data[6]), hip_unpack1(pRGB.data[6]), hip_unpack2(pRGB.data[6])));
        f.w = hip_dot3(cY, make_float3(hip_unpack0(pRGB.data[7]), hip_unpack1(pRGB.data[7]), hip_unpack2(pRGB.data[7])));
        yChannel.y = hip_pack(f);

        float3 cU = make_float3(-0.1146f, -0.3854f, 0.5f);
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB.data[0]), hip_unpack1(pRGB.data[0]), hip_unpack2(pRGB.data[0]))) + 128.0f;
        f.y = hip_dot3(cU, make_float3(hip_unpack0(pRGB.data[1]), hip_unpack1(pRGB.data[1]), hip_unpack2(pRGB.data[1]))) + 128.0f;
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB.data[2]), hip_unpack1(pRGB.data[2]), hip_unpack2(pRGB.data[2]))) + 128.0f;
        f.w = hip_dot3(cU, make_float3(hip_unpack0(pRGB.data[3]), hip_unpack1(pRGB.data[3]), hip_unpack2(pRGB.data[3]))) + 128.0f;
        uChannel.x = hip_pack(f);
        f.x = hip_dot3(cU, make_float3(hip_unpack0(pRGB.data[4]), hip_unpack1(pRGB.data[4]), hip_unpack2(pRGB.data[4]))) + 128.0f;
        f.y = hip_dot3(cU, make_float3(hip_unpack0(pRGB.data[5]), hip_unpack1(pRGB.data[5]), hip_unpack2(pRGB.data[5]))) + 128.0f;
        f.z = hip_dot3(cU, make_float3(hip_unpack0(pRGB.data[6]), hip_unpack1(pRGB.data[6]), hip_unpack2(pRGB.data[6]))) + 128.0f;
        f.w = hip_dot3(cU, make_float3(hip_unpack0(pRGB.data[7]), hip_unpack1(pRGB.data[7]), hip_unpack2(pRGB.data[7]))) + 128.0f;
        uChannel.y = hip_pack(f);

        float3 cV = make_float3(0.5f, -0.4542f, -0.0458f);
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB.data[0]), hip_unpack1(pRGB.data[0]), hip_unpack2(pRGB.data[0]))) + 128.0f;
        f.y = hip_dot3(cV, make_float3(hip_unpack0(pRGB.data[1]), hip_unpack1(pRGB.data[1]), hip_unpack2(pRGB.data[1]))) + 128.0f;
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB.data[2]), hip_unpack1(pRGB.data[2]), hip_unpack2(pRGB.data[2]))) + 128.0f;
        f.w = hip_dot3(cV, make_float3(hip_unpack0(pRGB.data[3]), hip_unpack1(pRGB.data[3]), hip_unpack2(pRGB.data[3]))) + 128.0f;
        vChannel.x = hip_pack(f);
        f.x = hip_dot3(cV, make_float3(hip_unpack0(pRGB.data[4]), hip_unpack1(pRGB.data[4]), hip_unpack2(pRGB.data[4]))) + 128.0f;
        f.y = hip_dot3(cV, make_float3(hip_unpack0(pRGB.data[5]), hip_unpack1(pRGB.data[5]), hip_unpack2(pRGB.data[5]))) + 128.0f;
        f.z = hip_dot3(cV, make_float3(hip_unpack0(pRGB.data[6]), hip_unpack1(pRGB.data[6]), hip_unpack2(pRGB.data[6]))) + 128.0f;
        f.w = hip_dot3(cV, make_float3(hip_unpack0(pRGB.data[7]), hip_unpack1(pRGB.data[7]), hip_unpack2(pRGB.data[7]))) + 128.0f;
        vChannel.y = hip_pack(f);

        *((uint2 *)(&pDstYImage[dstYIdx])) = yChannel;
        *((uint2 *)(&pDstUImage[dstUIdx])) = uChannel;
        *((uint2 *)(&pDstVImage[dstVIdx])) = vChannel;
    }
}
int HipExec_ColorConvert_YUV4_RGBX(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 16;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = dstHeight;

    hipLaunchKernelGGL(Hip_ColorConvert_YUV4_RGBX, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstYImage, dstYImageStrideInBytes, (uchar *)pHipDstUImage, dstUImageStrideInBytes, (uchar *)pHipDstVImage, dstVImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes);

    return VX_SUCCESS;
}





// Group 8 - Helper kernels with vision functions

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_IUV_UV12(uint dstWidth, uint dstHeight,
    uchar *pDstUImage, uint dstUImageStrideInBytes, uchar *pDstVImage, uint dstVImageStrideInBytes,
    const uchar *pSrcChromaImage, uint srcChromaImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcChromaImageStrideInBytesComp, uint dstUImageStrideInBytesComp, uint dstVImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint L0Idx = y * srcChromaImageStrideInBytesComp + (x << 4);
        uint L1Idx = L0Idx + srcChromaImageStrideInBytes;
        uint4 L0 = *((uint4 *)(&pSrcChromaImage[L0Idx]));
        uint4 L1 = *((uint4 *)(&pSrcChromaImage[L1Idx]));

        uint dstU0Idx = y * dstUImageStrideInBytesComp + (x << 3);
        uint dstU1Idx = dstU0Idx + dstUImageStrideInBytes;
        uint dstV0Idx = y * dstVImageStrideInBytesComp + (x << 3);
        uint dstV1Idx = dstV0Idx + dstVImageStrideInBytes;

        uint2 pU0, pV0, pU1, pV1;
        pU0.x = hip_pack(make_float4(hip_unpack0(L0.x), hip_unpack2(L0.x), hip_unpack0(L0.y), hip_unpack2(L0.y)));
        pU0.y = hip_pack(make_float4(hip_unpack0(L0.z), hip_unpack2(L0.z), hip_unpack0(L0.w), hip_unpack2(L0.w)));
        pV0.x = hip_pack(make_float4(hip_unpack1(L0.x), hip_unpack3(L0.x), hip_unpack1(L0.y), hip_unpack3(L0.y)));
        pV0.y = hip_pack(make_float4(hip_unpack1(L0.z), hip_unpack3(L0.z), hip_unpack1(L0.w), hip_unpack3(L0.w)));
        pU1.x = hip_pack(make_float4(hip_unpack0(L1.x), hip_unpack2(L1.x), hip_unpack0(L1.y), hip_unpack2(L1.y)));
        pU1.y = hip_pack(make_float4(hip_unpack0(L1.z), hip_unpack2(L1.z), hip_unpack0(L1.w), hip_unpack2(L1.w)));
        pV1.x = hip_pack(make_float4(hip_unpack1(L1.x), hip_unpack3(L1.x), hip_unpack1(L1.y), hip_unpack3(L1.y)));
        pV1.y = hip_pack(make_float4(hip_unpack1(L1.z), hip_unpack3(L1.z), hip_unpack1(L1.w), hip_unpack3(L1.w)));

        *((uint2 *)(&pDstUImage[dstU0Idx])) = pU0;
        *((uint2 *)(&pDstUImage[dstU1Idx])) = pU1;
        *((uint2 *)(&pDstVImage[dstV0Idx])) = pV0;
        *((uint2 *)(&pDstVImage[dstV1Idx])) = pV1;
    }
}
int HipExec_FormatConvert_IUV_UV12(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 srcChromaImageStrideInBytesComp = srcChromaImageStrideInBytes * 2;
    vx_uint32 dstUImageStrideInBytesComp = dstUImageStrideInBytes * 2;
    vx_uint32 dstVImageStrideInBytesComp = dstVImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_FormatConvert_IUV_UV12, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstUImage, dstUImageStrideInBytes, (uchar *)pHipDstVImage, dstVImageStrideInBytes,
                        (const uchar *)pHipSrcChromaImage, srcChromaImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcChromaImageStrideInBytesComp, dstUImageStrideInBytesComp, dstVImageStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_UV12_IUV(uint dstWidth, uint dstHeight,
    uchar *pDstChromaImage, uint dstChromaImageStrideInBytes,
    const uchar *pSrcUImage, uint srcUImageStrideInBytes, const uchar *pSrcVImage, uint srcVImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint srcUImageStrideInBytesComp, uint srcVImageStrideInBytesComp, uint dstChromaImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint srcU0Idx = y * srcUImageStrideInBytesComp + (x << 3);
        uint srcU1Idx = srcU0Idx + srcUImageStrideInBytes;
        uint srcV0Idx = y * srcVImageStrideInBytesComp + (x << 3);
        uint srcV1Idx = srcV0Idx + srcVImageStrideInBytes;
        uint2 pU0 = *((uint2 *)(&pSrcUImage[srcU0Idx]));
        uint2 pU1 = *((uint2 *)(&pSrcUImage[srcU1Idx]));
        uint2 pV0 = *((uint2 *)(&pSrcVImage[srcV0Idx]));
        uint2 pV1 = *((uint2 *)(&pSrcVImage[srcV1Idx]));

        uint L0Idx = y * dstChromaImageStrideInBytesComp + (x << 4);
        uint L1Idx = L0Idx + dstChromaImageStrideInBytes;

        uint4 L0, L1;
        L0.x = hip_pack(make_float4(hip_unpack0(pU0.x), hip_unpack0(pV0.x), hip_unpack1(pU0.x), hip_unpack1(pV0.x)));
        L0.y = hip_pack(make_float4(hip_unpack2(pU0.x), hip_unpack2(pV0.x), hip_unpack3(pU0.x), hip_unpack3(pV0.x)));
        L0.z = hip_pack(make_float4(hip_unpack0(pU0.y), hip_unpack0(pV0.y), hip_unpack1(pU0.y), hip_unpack1(pV0.y)));
        L0.w = hip_pack(make_float4(hip_unpack2(pU0.y), hip_unpack2(pV0.y), hip_unpack3(pU0.y), hip_unpack3(pV0.y)));
        L1.x = hip_pack(make_float4(hip_unpack0(pU1.x), hip_unpack0(pV1.x), hip_unpack1(pU1.x), hip_unpack1(pV1.x)));
        L1.y = hip_pack(make_float4(hip_unpack2(pU1.x), hip_unpack2(pV1.x), hip_unpack3(pU1.x), hip_unpack3(pV1.x)));
        L1.z = hip_pack(make_float4(hip_unpack0(pU1.y), hip_unpack0(pV1.y), hip_unpack1(pU1.y), hip_unpack1(pV1.y)));
        L1.w = hip_pack(make_float4(hip_unpack2(pU1.y), hip_unpack2(pV1.y), hip_unpack3(pU1.y), hip_unpack3(pV1.y)));

        *((uint4 *)(&pDstChromaImage[L0Idx])) = L0;
        *((uint4 *)(&pDstChromaImage[L1Idx])) = L1;
    }
}
int HipExec_FormatConvert_UV12_IUV(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstChromaImage, vx_uint32 dstChromaImageStrideInBytes,
    vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
    vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 srcUImageStrideInBytesComp = srcUImageStrideInBytes * 2;
    vx_uint32 srcVImageStrideInBytesComp = srcVImageStrideInBytes * 2;
    vx_uint32 dstChromaImageStrideInBytesComp = dstChromaImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_FormatConvert_UV12_IUV, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstChromaImage, dstChromaImageStrideInBytes,
                        (const uchar *)pHipSrcUImage, srcUImageStrideInBytes, (const uchar *)pHipSrcVImage, srcVImageStrideInBytes,
                        dstWidthComp, dstHeightComp, srcUImageStrideInBytesComp, srcVImageStrideInBytesComp, dstChromaImageStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_FormatConvert_UV_UV12(uint dstWidth, uint dstHeight,
    uchar *pDstUImage, uint dstUImageStrideInBytes, uchar *pDstVImage, uint dstVImageStrideInBytes,
    const uchar *pSrcChromaImage, uint srcChromaImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint dstUImageStrideInBytesComp, uint dstVImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint L0Idx = y * srcChromaImageStrideInBytes + (x << 3);
        uint2 L0 = *((uint2 *)(&pSrcChromaImage[L0Idx]));

        uint dstU0Idx = y * dstUImageStrideInBytesComp + (x << 3);
        uint dstU1Idx = dstU0Idx + dstUImageStrideInBytes;
        uint dstV0Idx = y * dstVImageStrideInBytesComp + (x << 3);
        uint dstV1Idx = dstV0Idx + dstVImageStrideInBytes;

        uint2 pU, pV;
        pU.x = hip_pack(make_float4(hip_unpack0(L0.x), hip_unpack0(L0.x), hip_unpack2(L0.x), hip_unpack2(L0.x)));
        pU.y = hip_pack(make_float4(hip_unpack0(L0.y), hip_unpack0(L0.y), hip_unpack2(L0.y), hip_unpack2(L0.y)));
        pV.x = hip_pack(make_float4(hip_unpack1(L0.x), hip_unpack1(L0.x), hip_unpack3(L0.x), hip_unpack3(L0.x)));
        pV.y = hip_pack(make_float4(hip_unpack1(L0.y), hip_unpack1(L0.y), hip_unpack3(L0.y), hip_unpack3(L0.y)));

        *((uint2 *)(&pDstUImage[dstU0Idx])) = pU;
        *((uint2 *)(&pDstUImage[dstU1Idx])) = pU;
        *((uint2 *)(&pDstVImage[dstV0Idx])) = pV;
        *((uint2 *)(&pDstVImage[dstV1Idx])) = pV;
    }
}
int HipExec_FormatConvert_UV_UV12(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
    vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
    const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstUImageStrideInBytesComp = dstUImageStrideInBytes * 2;
    vx_uint32 dstVImageStrideInBytesComp = dstVImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_FormatConvert_UV_UV12, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstUImage, dstUImageStrideInBytes, (uchar *)pHipDstVImage, dstVImageStrideInBytes,
                        (const uchar *)pHipSrcChromaImage, srcChromaImageStrideInBytes,
                        dstWidthComp, dstHeightComp, dstUImageStrideInBytesComp, dstVImageStrideInBytesComp);

    return VX_SUCCESS;
}

__global__ void __attribute__((visibility("default")))
Hip_ScaleUp2x2_U8_U8(uint dstWidth, uint dstHeight,
    uchar *pDstImage, uint dstImageStrideInBytes,
    const uchar *pSrcImage, uint srcImageStrideInBytes,
    uint dstWidthComp, uint dstHeightComp, uint dstImageStrideInBytesComp) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ((x < dstWidthComp) && (y < dstHeightComp)) {
        uint srcIdx = y * srcImageStrideInBytes + (x << 2);
        uint src = *((uint *)(&pSrcImage[srcIdx]));

        uint dstIdx0 = y * dstImageStrideInBytesComp + (x << 3);
        uint dstIdx1 = dstIdx0 + dstImageStrideInBytes;
        uint2 dst;

        dst.x = hip_pack(make_float4(hip_unpack0(src), hip_unpack0(src), hip_unpack1(src), hip_unpack1(src)));
        dst.y = hip_pack(make_float4(hip_unpack2(src), hip_unpack2(src), hip_unpack3(src), hip_unpack3(src)));

        *((uint2 *)(&pDstImage[dstIdx0])) = dst;
        *((uint2 *)(&pDstImage[dstIdx1])) = dst;
    }
}
int HipExec_ScaleUp2x2_U8_U8(hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes) {
    int localThreads_x = 16;
    int localThreads_y = 4;
    int globalThreads_x = (dstWidth + 7) >> 3;
    int globalThreads_y = (dstHeight + 1) >> 1;

    vx_uint32 dstWidthComp = (dstWidth + 7) / 8;
    vx_uint32 dstHeightComp = (dstHeight + 1) / 2;
    vx_uint32 dstImageStrideInBytesComp = dstImageStrideInBytes * 2;

    hipLaunchKernelGGL(Hip_ScaleUp2x2_U8_U8, dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y)),
                        dim3(localThreads_x, localThreads_y), 0, stream, dstWidth, dstHeight, (uchar *)pHipDstImage, dstImageStrideInBytes,
                        (const uchar *)pHipSrcImage, srcImageStrideInBytes,
                        dstWidthComp, dstHeightComp, dstImageStrideInBytesComp);

    return VX_SUCCESS;
}