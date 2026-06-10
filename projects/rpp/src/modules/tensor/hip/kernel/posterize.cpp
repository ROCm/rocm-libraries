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

// Helper functions that compute mask/factor based on individual image's posterize level bits
__device__ __forceinline__ uchar compute_posterizeBitsMask(uchar imgPosterizeLevelBits)
{
    return ((1 << imgPosterizeLevelBits) - 1) << (8 - imgPosterizeLevelBits);
}

__device__ __forceinline__ float compute_posterizeBitsFactor(uchar imgPosterizeLevelBits)
{
    return 255.0/(1 << (8 - imgPosterizeLevelBits));
}


// Helper for U8 and I8 data type images - Bitwise AND with mask to represent with lesser number of bits
__device__ void posterize_hip_compute(d_uchar8 *src_uc8, d_uchar8* src_mask_u8, d_uchar8 *dst_uc8)
{
    rpp_hip_math_bitwise_op8<BitwiseAnd>(src_uc8, src_mask_u8, dst_uc8);
}

// Helper for F16 data type images - Scaled up to 0-255, bitwise and is performed, and normalized back to 0-1
// Pixel values are scaled up to the range 0–255 before a bitwise AND is performed.
// The same method used for F32 is not applied here as it leads to precision mismatches.
//
// Example:
// The equivalence of 96 in the 0–255 range (U8 representation) is either:
//   - 0.376471 in F32 (normalized to [0, 1])
//   - 0.376465 in F16 (normalized to [0, 1])
//
// Multiplying these values by the posterize factor for 3 bits (7.968750) results in:
//   - F32: 0.376471 × 7.968750 = 3.000000
//   - F16: 0.376465 × 7.968750 = 2.999954
//
// Taking the floor of these values gives different results, which causes significant pixel mismatches.

__device__ void posterize_hip_compute(d_float8 *src_f8, d_uchar8* src_mask_u8, d_float8 *dst_f8)
{
    rpp_hip_math_multiply8_const(src_f8, src_f8, MAKE_FLOAT4(255));
    rpp_hip_math_scaled_bitwiseAnd8(src_f8, src_mask_u8, dst_f8);
    rpp_hip_math_multiply8_const(dst_f8, dst_f8, MAKE_FLOAT4(ONE_OVER_255));
}

// Helper for F32 data type images - Scaled up by posterize factor, floored and normalized back to 0-1
__device__ void posterize_hip_compute(d_float8 *src_f8, d_float8* srcFactor_f8, d_float8 *dst_f8)
{
    d_float8 scaled_src_f8, floored_src_f8;
    rpp_hip_math_multiply8(src_f8, srcFactor_f8, &scaled_src_f8);
    rpp_hip_math_floor8(&scaled_src_f8, &floored_src_f8);
    rpp_hip_math_divide8(&floored_src_f8, srcFactor_f8, dst_f8);
}

__global__ void posterize_pkd_hip_tensor(Rpp8u *srcPtr,
                                         uint2 srcStridesNH,
                                         Rpp8u *dstPtr,
                                         uint2 dstStridesNH,
                                         Rpp8u *posterizeLevelBits,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_uchar8 src_mask_u8;

    uchar posterizeBitsMask = compute_posterizeBitsMask(posterizeLevelBits[id_z]);

    src_mask_u8.uc4[0] = MAKE_UCHAR4(posterizeBitsMask);
    src_mask_u8.uc4[1] = MAKE_UCHAR4(posterizeBitsMask);

    d_uchar24 src_uc24, dst_uc24;

    rpp_hip_load24_pkd3_and_unpack_to_uchar24_pkd3(srcPtr + srcIdx, &src_uc24);
    posterize_hip_compute(&src_uc24.uc8[0], &src_mask_u8, &dst_uc24.uc8[0]);
    posterize_hip_compute(&src_uc24.uc8[1], &src_mask_u8, &dst_uc24.uc8[1]);
    posterize_hip_compute(&src_uc24.uc8[2], &src_mask_u8, &dst_uc24.uc8[2]);
    rpp_hip_pack_uchar24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_uc24);
}

__global__ void posterize_pkd_hip_tensor(half *srcPtr,
                                         uint2 srcStridesNH,
                                         half *dstPtr,
                                         uint2 dstStridesNH,
                                         Rpp8u *posterizeLevelBits,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_uchar8 src_mask_u8;

    uchar posterizeBitsMask = compute_posterizeBitsMask(posterizeLevelBits[id_z]);

    src_mask_u8.uc4[0] = MAKE_UCHAR4(posterizeBitsMask);
    src_mask_u8.uc4[1] = MAKE_UCHAR4(posterizeBitsMask);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    posterize_hip_compute(&src_f24.f8[0], &src_mask_u8, &dst_f24.f8[0]);
    posterize_hip_compute(&src_f24.f8[1], &src_mask_u8, &dst_f24.f8[1]);
    posterize_hip_compute(&src_f24.f8[2], &src_mask_u8, &dst_f24.f8[2]);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

__global__ void posterize_pkd_hip_tensor(Rpp32f *srcPtr,
                                         uint2 srcStridesNH,
                                         Rpp32f *dstPtr,
                                         uint2 dstStridesNH,
                                         Rpp8u *posterizeLevelBits,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float posterizeBitsFactor = compute_posterizeBitsFactor(posterizeLevelBits[id_z]);

    d_float8 srcFactor_f8;
    srcFactor_f8.f4[0] = MAKE_FLOAT4(posterizeBitsFactor);
    srcFactor_f8.f4[1] = MAKE_FLOAT4(posterizeBitsFactor);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    posterize_hip_compute(&src_f24.f8[0], &srcFactor_f8, &dst_f24.f8[0]);
    posterize_hip_compute(&src_f24.f8[1], &srcFactor_f8, &dst_f24.f8[1]);
    posterize_hip_compute(&src_f24.f8[2], &srcFactor_f8, &dst_f24.f8[2]);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

__global__ void posterize_pln_hip_tensor(Rpp8u *srcPtr,
                                         uint3 srcStridesNCH,
                                         Rpp8u *dstPtr,
                                         uint3 dstStridesNCH,
                                         int channelsDst,
                                         Rpp8u *posterizeLevelBits,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    uchar posterizeBitsMask = compute_posterizeBitsMask(posterizeLevelBits[id_z]);

    d_uchar8 src_mask_u8;
    src_mask_u8.uc4[0] = MAKE_UCHAR4(posterizeBitsMask);
    src_mask_u8.uc4[1] = MAKE_UCHAR4(posterizeBitsMask);

    d_uchar8 src_uc8, dst_uc8;
    uchar* srcPtr_uc8 = (uchar*)&src_uc8;

    rpp_hip_load8_to_uchar8(srcPtr + srcIdx, srcPtr_uc8);
    posterize_hip_compute(&src_uc8, &src_mask_u8, &dst_uc8);
    rpp_hip_pack_uchar8_and_store8(dstPtr + dstIdx, &dst_uc8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, srcPtr_uc8);
        posterize_hip_compute(&src_uc8, &src_mask_u8, &dst_uc8);
        rpp_hip_pack_uchar8_and_store8(dstPtr + dstIdx, &dst_uc8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, srcPtr_uc8);
        posterize_hip_compute(&src_uc8, &src_mask_u8, &dst_uc8);
        rpp_hip_pack_uchar8_and_store8(dstPtr + dstIdx, &dst_uc8);
    }
}

__global__ void posterize_pln_hip_tensor(half *srcPtr,
                                         uint3 srcStridesNCH,
                                         half *dstPtr,
                                         uint3 dstStridesNCH,
                                         int channelsDst,
                                         Rpp8u *posterizeLevelBits,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    uchar posterizeBitsMask = compute_posterizeBitsMask(posterizeLevelBits[id_z]);

    d_uchar8 src_mask_u8;
    src_mask_u8.uc4[0] = MAKE_UCHAR4(posterizeBitsMask);
    src_mask_u8.uc4[1] = MAKE_UCHAR4(posterizeBitsMask);

    d_float8 src_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    posterize_hip_compute(&src_f8, &src_mask_u8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        posterize_hip_compute(&src_f8, &src_mask_u8, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        posterize_hip_compute(&src_f8, &src_mask_u8, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

__global__ void posterize_pln_hip_tensor(Rpp32f *srcPtr,
                                         uint3 srcStridesNCH,
                                         Rpp32f *dstPtr,
                                         uint3 dstStridesNCH,
                                         int channelsDst,
                                         Rpp8u *posterizeLevelBits,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float posterizeBitsFactor = compute_posterizeBitsFactor(posterizeLevelBits[id_z]);

    d_float8 srcFactor_f8;
    srcFactor_f8.f4[0] = MAKE_FLOAT4(posterizeBitsFactor);
    srcFactor_f8.f4[1] = MAKE_FLOAT4(posterizeBitsFactor);

    d_float8 src_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    posterize_hip_compute(&src_f8, &srcFactor_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        posterize_hip_compute(&src_f8, &srcFactor_f8, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        posterize_hip_compute(&src_f8, &srcFactor_f8, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

__global__ void posterize_pkd3_pln3_hip_tensor(Rpp8u *srcPtr,
                                               uint2 srcStridesNH,
                                               Rpp8u *dstPtr,
                                               uint3 dstStridesNCH,
                                               Rpp8u *posterizeLevelBits,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    uchar posterizeBitsMask = compute_posterizeBitsMask(posterizeLevelBits[id_z]);

    d_uchar8 src_mask_u8;
    src_mask_u8.uc4[0] = MAKE_UCHAR4(posterizeBitsMask);
    src_mask_u8.uc4[1] = MAKE_UCHAR4(posterizeBitsMask);

    d_uchar24 src_uc24, dst_uc24;

    rpp_hip_load24_pkd3_and_unpack_to_uchar24_pln3(srcPtr + srcIdx, &src_uc24);
    posterize_hip_compute(&src_uc24.uc8[0], &src_mask_u8, &dst_uc24.uc8[0]);
    posterize_hip_compute(&src_uc24.uc8[1], &src_mask_u8, &dst_uc24.uc8[1]);
    posterize_hip_compute(&src_uc24.uc8[2], &src_mask_u8, &dst_uc24.uc8[2]);
    rpp_hip_pack_uchar24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_uc24);
}

__global__ void posterize_pkd3_pln3_hip_tensor(half *srcPtr,
                                               uint2 srcStridesNH,
                                               half *dstPtr,
                                               uint3 dstStridesNCH,
                                               Rpp8u *posterizeLevelBits,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    uchar posterizeBitsMask = compute_posterizeBitsMask(posterizeLevelBits[id_z]);

    d_uchar8 src_mask_u8;
    src_mask_u8.uc4[0] = MAKE_UCHAR4(posterizeBitsMask);
    src_mask_u8.uc4[1] = MAKE_UCHAR4(posterizeBitsMask);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    posterize_hip_compute(&src_f24.f8[0], &src_mask_u8, &dst_f24.f8[0]);
    posterize_hip_compute(&src_f24.f8[1], &src_mask_u8, &dst_f24.f8[1]);
    posterize_hip_compute(&src_f24.f8[2], &src_mask_u8, &dst_f24.f8[2]);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

__global__ void posterize_pkd3_pln3_hip_tensor(Rpp32f *srcPtr,
                                               uint2 srcStridesNH,
                                               Rpp32f *dstPtr,
                                               uint3 dstStridesNCH,
                                               Rpp8u *posterizeLevelBits,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float posterizeBitsFactor = compute_posterizeBitsFactor(posterizeLevelBits[id_z]);

    d_float8 srcFactor_f8;
    srcFactor_f8.f4[0] = MAKE_FLOAT4(posterizeBitsFactor);
    srcFactor_f8.f4[1] = MAKE_FLOAT4(posterizeBitsFactor);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    posterize_hip_compute(&src_f24.f8[0], &srcFactor_f8, &dst_f24.f8[0]);
    posterize_hip_compute(&src_f24.f8[1], &srcFactor_f8, &dst_f24.f8[1]);
    posterize_hip_compute(&src_f24.f8[2], &srcFactor_f8, &dst_f24.f8[2]);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

__global__ void posterize_pln3_pkd3_hip_tensor(Rpp8u *srcPtr,
                                               uint3 srcStridesNCH,
                                               Rpp8u *dstPtr,
                                               uint2 dstStridesNH,
                                               Rpp8u *posterizeLevelBits,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    uchar posterizeBitsMask = compute_posterizeBitsMask(posterizeLevelBits[id_z]);

    d_uchar8 src_mask_u8;
    src_mask_u8.uc4[0] = MAKE_UCHAR4(posterizeBitsMask);
    src_mask_u8.uc4[1] = MAKE_UCHAR4(posterizeBitsMask);

    d_uchar24 src_uc24, dst_uc24;

    rpp_hip_load24_pln3_and_unpack_to_uchar24_pkd3(srcPtr + srcIdx, srcStridesNCH.y, &src_uc24);
    posterize_hip_compute(&src_uc24.uc8[0], &src_mask_u8, &dst_uc24.uc8[0]);
    posterize_hip_compute(&src_uc24.uc8[1], &src_mask_u8, &dst_uc24.uc8[1]);
    posterize_hip_compute(&src_uc24.uc8[2], &src_mask_u8, &dst_uc24.uc8[2]);
    rpp_hip_pack_uchar24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_uc24);
}

__global__ void posterize_pln3_pkd3_hip_tensor(half *srcPtr,
                                               uint3 srcStridesNCH,
                                               half *dstPtr,
                                               uint2 dstStridesNH,
                                               Rpp8u *posterizeLevelBits,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    uchar posterizeBitsMask = compute_posterizeBitsMask(posterizeLevelBits[id_z]);

    d_uchar8 src_mask_u8;
    src_mask_u8.uc4[0] = MAKE_UCHAR4(posterizeBitsMask);
    src_mask_u8.uc4[1] = MAKE_UCHAR4(posterizeBitsMask);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    posterize_hip_compute(&src_f24.f8[0], &src_mask_u8, &dst_f24.f8[0]);
    posterize_hip_compute(&src_f24.f8[1], &src_mask_u8, &dst_f24.f8[1]);
    posterize_hip_compute(&src_f24.f8[2], &src_mask_u8, &dst_f24.f8[2]);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

__global__ void posterize_pln3_pkd3_hip_tensor(Rpp32f *srcPtr,
                                               uint3 srcStridesNCH,
                                               Rpp32f *dstPtr,
                                               uint2 dstStridesNH,
                                               Rpp8u *posterizeLevelBits,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float posterizeBitsFactor = compute_posterizeBitsFactor(posterizeLevelBits[id_z]);

    d_float8 srcFactor_f8;
    srcFactor_f8.f4[0] = MAKE_FLOAT4(posterizeBitsFactor);
    srcFactor_f8.f4[1] = MAKE_FLOAT4(posterizeBitsFactor);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    posterize_hip_compute(&src_f24.f8[0], &srcFactor_f8, &dst_f24.f8[0]);
    posterize_hip_compute(&src_f24.f8[1], &srcFactor_f8, &dst_f24.f8[1]);
    posterize_hip_compute(&src_f24.f8[2], &srcFactor_f8, &dst_f24.f8[2]);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}


template <typename T>
RppStatus hip_exec_posterize_tensor(T *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    T *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp8u *posterizeLevelBits,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->w + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = dstDescPtr->n;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(posterize_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           posterizeLevelBits,
                           roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(posterize_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           posterizeLevelBits,
                           roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(posterize_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               posterizeLevelBits,
                               roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(posterize_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               posterizeLevelBits,
                               roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_posterize_tensor<Rpp8u>(Rpp8u*,
                                                    RpptDescPtr,
                                                    Rpp8u*,
                                                    RpptDescPtr,
                                                    Rpp8u*,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    rpp::Handle&);

template RppStatus hip_exec_posterize_tensor<half>(half*,
                                                   RpptDescPtr,
                                                   half*,
                                                   RpptDescPtr,
                                                   Rpp8u*,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   rpp::Handle&);

template RppStatus hip_exec_posterize_tensor<Rpp32f>(Rpp32f*,
                                                     RpptDescPtr,
                                                     Rpp32f*,
                                                     RpptDescPtr,
                                                     Rpp8u*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);
