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

#ifndef RPP_HIP_INTERPOLATION_HPP
#define RPP_HIP_INTERPOLATION_HPP

// /******************** DEVICE INTERPOLATION HELPER FUNCTIONS ********************/

// ROI range check for source locations calculated

__device__ __forceinline__ void rpp_hip_roi_range_check(float2* locSrcFloor_f2, int4* roiPtrSrc_i4,
                                                        int2* locSrc_i2) {
    locSrc_i2->x = (int)fminf(fmaxf(locSrcFloor_f2->x, roiPtrSrc_i4->x), roiPtrSrc_i4->z - 1);
    locSrc_i2->y = (int)fminf(fmaxf(locSrcFloor_f2->y, roiPtrSrc_i4->y), roiPtrSrc_i4->w - 1);
}

__device__ __forceinline__ float rpp_hip_math_sinc(float x) {
    x *= PI;
    return (fabsf(x) < 1e-5f) ? (1.0f - x * x * ONE_OVER_6) : sinf(x) / x;
}

__device__ __forceinline__ void rpp_hip_compute_bicubic_coefficient(float weight, float* coeff) {
    Rpp32f x = fabsf(weight);
    *coeff = (x >= 2) ? 0
                      : ((x > 1) ? (x * x * (-0.5f * x + 2.5f) - 4.0f * x + 2.0f)
                                 : (x * x * (1.5f * x - 2.5f) + 1.0f));
}

__device__ __forceinline__ void rpp_hip_compute_lanczos3_coefficient(float weight, float* coeff) {
    *coeff = fabsf(weight) >= 3
                 ? 0.0f
                 : (rpp_hip_math_sinc(weight) * rpp_hip_math_sinc(weight * ONE_OVER_3));
}

__device__ __forceinline__ void rpp_hip_compute_gaussian_coefficient(float weight, float* coeff) {
    *coeff = expf(weight * weight * -4.0f);
}

__device__ __forceinline__ void rpp_hip_compute_triangular_coefficient(float weight, float* coeff) {
    *coeff = 1 - fabsf(weight);
    *coeff = *coeff < 0 ? 0 : *coeff;
}

__device__ __forceinline__ void rpp_hip_compute_interpolation_coefficient(
    RpptInterpolationType interpolationType, float weight, float* coeff) {
    switch (interpolationType) {
        case RpptInterpolationType::BICUBIC: {
            rpp_hip_compute_bicubic_coefficient(weight, coeff);
            break;
        }
        case RpptInterpolationType::LANCZOS: {
            rpp_hip_compute_lanczos3_coefficient(weight, coeff);
            break;
        }
        case RpptInterpolationType::GAUSSIAN: {
            rpp_hip_compute_gaussian_coefficient(weight, coeff);
            break;
        }
        case RpptInterpolationType::TRIANGULAR: {
            rpp_hip_compute_triangular_coefficient(weight, coeff);
            break;
        }
        default:
            break;
    }
}

__device__ void rpp_hip_compute_interpolation_scale_and_radius(
    RpptInterpolationType interpolationType, float* scale, float* radius, float scaleRatio) {
    switch (interpolationType) {
        case RpptInterpolationType::BICUBIC: {
            *radius = 2.0f;
            break;
        }
        case RpptInterpolationType::LANCZOS: {
            if (scaleRatio > 1.0f) {
                *radius = 3.0f * scaleRatio;
                *scale = (1 / scaleRatio);
            } else
                *radius = 3.0f;
            break;
        }
        case RpptInterpolationType::GAUSSIAN: {
            if (scaleRatio > 1.0f) {
                *radius = scaleRatio;
                *scale = (1 / scaleRatio);
            }
            break;
        }
        case RpptInterpolationType::TRIANGULAR: {
            if (scaleRatio > 1.0f) {
                *radius = scaleRatio;
                *scale = (1 / scaleRatio);
            }
            break;
        }
        default: {
            *radius = 1.0f;
            *scale = 1.0f;
            break;
        }
    }
}

// BILINEAR INTERPOLATION LOAD HELPERS (separate load routines for each bit depth)

// Per-corner ROI validity for the four bilinear taps, computed from the *unclamped* top-left
// location (locSrcFloor_f2, before it gets advanced to the bottom-right location by the callers
// below). When checkRange is false (e.g. resize-family ops that pre-clamp their coordinates and
// expect edge-replication rather than border-substitution), every corner is reported valid and
// rpp_hip_roi_range_check's clamp is solely responsible for addressing.
__device__ __forceinline__ void rpp_hip_compute_bilinear_corner_validity(
    float2* locSrcFloor_f2, int4* roiPtrSrc_i4, bool checkRange, bool* validTL, bool* validTR,
    bool* validBL, bool* validBR) {
    if (!checkRange) {
        *validTL = *validTR = *validBL = *validBR = true;
        return;
    }
    float left = locSrcFloor_f2->x;
    float top = locSrcFloor_f2->y;
    float right = left + 1.0f;
    float bottom = top + 1.0f;
    bool leftValid = (left >= roiPtrSrc_i4->x) && (left <= roiPtrSrc_i4->z);
    bool rightValid = (right >= roiPtrSrc_i4->x) && (right <= roiPtrSrc_i4->z);
    bool topValid = (top >= roiPtrSrc_i4->y) && (top <= roiPtrSrc_i4->w);
    bool bottomValid = (bottom >= roiPtrSrc_i4->y) && (bottom <= roiPtrSrc_i4->w);
    *validTL = topValid && leftValid;
    *validTR = topValid && rightValid;
    *validBL = bottomValid && leftValid;
    *validBR = bottomValid && rightValid;
}

// U8 loads for bilinear interpolation (4 U8 pixels)

__device__ __forceinline__ void rpp_hip_interpolate1_bilinear_load_pln1(
    uchar* srcPtr, uint srcStrideH, float2* locSrcFloor_f2, int4* roiPtrSrc_i4,
    float4* srcNeighborhood_f4, bool checkRange) {
    uint2 src_u2;
    int2 locSrc1_i2, locSrc2_i2;
    bool validTL, validTR, validBL, validBR;
    rpp_hip_compute_bilinear_corner_validity(locSrcFloor_f2, roiPtrSrc_i4, checkRange, &validTL,
                                             &validTR, &validBL, &validBR);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc1_i2);
    *locSrcFloor_f2 = *locSrcFloor_f2 + MAKE_FLOAT2(1.0f);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc2_i2);
    int2 srcInterRowLoc_i2;
    srcInterRowLoc_i2.x = locSrc1_i2.y * srcStrideH;
    srcInterRowLoc_i2.y = locSrc2_i2.y * srcStrideH;

    int srcIdx1 = srcInterRowLoc_i2.x + locSrc1_i2.x;  // Top Left
    int srcIdx2 = srcInterRowLoc_i2.x + locSrc2_i2.x;  // Top Right
    src_u2.x = *(uint*)&srcPtr[srcIdx1];
    src_u2.y = *(uint*)&srcPtr[srcIdx2];
    srcNeighborhood_f4->x = validTL ? rpp_hip_unpack0(src_u2.x) : 0.0f;
    srcNeighborhood_f4->y = validTR ? rpp_hip_unpack0(src_u2.y) : 0.0f;
    srcIdx1 = srcInterRowLoc_i2.y + locSrc1_i2.x;  // Bottom left
    srcIdx2 = srcInterRowLoc_i2.y + locSrc2_i2.x;  // Bottom right
    src_u2.x = *(uint*)&srcPtr[srcIdx1];
    src_u2.y = *(uint*)&srcPtr[srcIdx2];
    srcNeighborhood_f4->z = validBL ? rpp_hip_unpack0(src_u2.x) : 0.0f;
    srcNeighborhood_f4->w = validBR ? rpp_hip_unpack0(src_u2.y) : 0.0f;
}

// F32 loads for bilinear interpolation (4 F32 pixels)

__device__ __forceinline__ void rpp_hip_interpolate1_bilinear_load_pln1(
    float* srcPtr, uint srcStrideH, float2* locSrcFloor_f2, int4* roiPtrSrc_i4,
    float4* srcNeighborhood_f4, bool checkRange) {
    int2 locSrc1_i2, locSrc2_i2;
    bool validTL, validTR, validBL, validBR;
    rpp_hip_compute_bilinear_corner_validity(locSrcFloor_f2, roiPtrSrc_i4, checkRange, &validTL,
                                             &validTR, &validBL, &validBR);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc1_i2);
    *locSrcFloor_f2 = *locSrcFloor_f2 + MAKE_FLOAT2(1.0f);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc2_i2);
    int2 srcInterRowLoc_i2;
    srcInterRowLoc_i2.x = locSrc1_i2.y * srcStrideH;
    srcInterRowLoc_i2.y = locSrc2_i2.y * srcStrideH;

    int srcIdx1 = srcInterRowLoc_i2.x + locSrc1_i2.x;  // Top Left
    int srcIdx2 = srcInterRowLoc_i2.x + locSrc2_i2.x;  // Top Right
    srcNeighborhood_f4->x = validTL ? *(float*)&srcPtr[srcIdx1] : 0.0f;
    srcNeighborhood_f4->y = validTR ? *(float*)&srcPtr[srcIdx2] : 0.0f;
    srcIdx1 = srcInterRowLoc_i2.y + locSrc1_i2.x;  // Bottom left
    srcIdx2 = srcInterRowLoc_i2.y + locSrc2_i2.x;  // Bottom right
    srcNeighborhood_f4->z = validBL ? *(float*)&srcPtr[srcIdx1] : 0.0f;
    srcNeighborhood_f4->w = validBR ? *(float*)&srcPtr[srcIdx2] : 0.0f;
}

// I8 loads for bilinear interpolation (4 I8 pixels)

__device__ __forceinline__ void rpp_hip_interpolate1_bilinear_load_pln1(
    schar* srcPtr, uint srcStrideH, float2* locSrcFloor_f2, int4* roiPtrSrc_i4,
    float4* srcNeighborhood_f4, bool checkRange) {
    int2 src_i2, locSrc1_i2, locSrc2_i2;
    bool validTL, validTR, validBL, validBR;
    rpp_hip_compute_bilinear_corner_validity(locSrcFloor_f2, roiPtrSrc_i4, checkRange, &validTL,
                                             &validTR, &validBL, &validBR);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc1_i2);
    *locSrcFloor_f2 = *locSrcFloor_f2 + MAKE_FLOAT2(1.0f);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc2_i2);
    int2 srcInterRowLoc_i2;
    srcInterRowLoc_i2.x = locSrc1_i2.y * srcStrideH;
    srcInterRowLoc_i2.y = locSrc2_i2.y * srcStrideH;

    int srcIdx1 = srcInterRowLoc_i2.x + locSrc1_i2.x;  // Top Left
    int srcIdx2 = srcInterRowLoc_i2.x + locSrc2_i2.x;  // Top Right
    src_i2.x = *(int*)&srcPtr[srcIdx1];
    src_i2.y = *(int*)&srcPtr[srcIdx2];
    srcNeighborhood_f4->x = validTL ? rpp_hip_unpack0(src_i2.x) : 0.0f;
    srcNeighborhood_f4->y = validTR ? rpp_hip_unpack0(src_i2.y) : 0.0f;
    srcIdx1 = srcInterRowLoc_i2.y + locSrc1_i2.x;  // Bottom left
    srcIdx2 = srcInterRowLoc_i2.y + locSrc2_i2.x;  // Bottom right
    src_i2.x = *(int*)&srcPtr[srcIdx1];
    src_i2.y = *(int*)&srcPtr[srcIdx2];
    srcNeighborhood_f4->z = validBL ? rpp_hip_unpack0(src_i2.x) : 0.0f;
    srcNeighborhood_f4->w = validBR ? rpp_hip_unpack0(src_i2.y) : 0.0f;
}

// F16 loads for bilinear interpolation (4 F16 pixels)

__device__ __forceinline__ void rpp_hip_interpolate1_bilinear_load_pln1(
    half* srcPtr, uint srcStrideH, float2* locSrcFloor_f2, int4* roiPtrSrc_i4,
    float4* srcNeighborhood_f4, bool checkRange) {
    float2 srcUpper_f2, srcLower_f2;
    int2 locSrc1_i2, locSrc2_i2;
    bool validTL, validTR, validBL, validBR;
    rpp_hip_compute_bilinear_corner_validity(locSrcFloor_f2, roiPtrSrc_i4, checkRange, &validTL,
                                             &validTR, &validBL, &validBR);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc1_i2);
    *locSrcFloor_f2 = *locSrcFloor_f2 + MAKE_FLOAT2(1.0f);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc2_i2);
    int2 srcInterRowLoc_i2;
    srcInterRowLoc_i2.x = locSrc1_i2.y * srcStrideH;
    srcInterRowLoc_i2.y = locSrc2_i2.y * srcStrideH;

    int srcIdx1 = srcInterRowLoc_i2.x + locSrc1_i2.x;  // Top Left
    int srcIdx2 = srcInterRowLoc_i2.x + locSrc2_i2.x;  // Top Right
    srcUpper_f2.x = validTL ? __half2float(*(half*)&srcPtr[srcIdx1]) : 0.0f;
    srcUpper_f2.y = validTR ? __half2float(*(half*)&srcPtr[srcIdx2]) : 0.0f;
    srcIdx1 = srcInterRowLoc_i2.y + locSrc1_i2.x;  // Bottom left
    srcIdx2 = srcInterRowLoc_i2.y + locSrc2_i2.x;  // Bottom right
    srcLower_f2.x = validBL ? __half2float(*(half*)&srcPtr[srcIdx1]) : 0.0f;
    srcLower_f2.y = validBR ? __half2float(*(half*)&srcPtr[srcIdx2]) : 0.0f;
    *srcNeighborhood_f4 = make_float4(srcUpper_f2.x, srcUpper_f2.y, srcLower_f2.x, srcLower_f2.y);
}

// U8 loads for bilinear interpolation (12 U8 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_load_pkd3(
    uchar* srcPtr, uint srcStrideH, float2* locSrcFloor_f2, int4* roiPtrSrc_i4,
    d_float12* srcNeighborhood_f12, bool checkRange) {
    uint2 src_u2;
    int2 locSrc1_i2, locSrc2_i2;
    bool validTL, validTR, validBL, validBR;
    rpp_hip_compute_bilinear_corner_validity(locSrcFloor_f2, roiPtrSrc_i4, checkRange, &validTL,
                                             &validTR, &validBL, &validBR);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc1_i2);
    *locSrcFloor_f2 = *locSrcFloor_f2 + MAKE_FLOAT2(1.0f);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc2_i2);
    int2 srcInterRowLoc_i2, srcInterColLoc_i2;
    srcInterRowLoc_i2.x = locSrc1_i2.y * srcStrideH;
    srcInterRowLoc_i2.y = locSrc2_i2.y * srcStrideH;
    srcInterColLoc_i2.x = locSrc1_i2.x * 3;
    srcInterColLoc_i2.y = locSrc2_i2.x * 3;

    int srcIdx1 = srcInterRowLoc_i2.x + srcInterColLoc_i2.x;  // Top Left
    int srcIdx2 = srcInterRowLoc_i2.x + srcInterColLoc_i2.y;  // Top Right
    src_u2.x = *(uint*)&srcPtr[srcIdx1];
    src_u2.y = *(uint*)&srcPtr[srcIdx2];
    srcNeighborhood_f12->f1[0] = validTL ? rpp_hip_unpack0(src_u2.x) : 0.0f;
    srcNeighborhood_f12->f1[1] = validTR ? rpp_hip_unpack0(src_u2.y) : 0.0f;
    srcNeighborhood_f12->f1[4] = validTL ? rpp_hip_unpack1(src_u2.x) : 0.0f;
    srcNeighborhood_f12->f1[5] = validTR ? rpp_hip_unpack1(src_u2.y) : 0.0f;
    srcNeighborhood_f12->f1[8] = validTL ? rpp_hip_unpack2(src_u2.x) : 0.0f;
    srcNeighborhood_f12->f1[9] = validTR ? rpp_hip_unpack2(src_u2.y) : 0.0f;
    srcIdx1 = srcInterRowLoc_i2.y + srcInterColLoc_i2.x;  // Bottom left
    srcIdx2 = srcInterRowLoc_i2.y + srcInterColLoc_i2.y;  // Bottom right
    src_u2.x = *(uint*)&srcPtr[srcIdx1];
    src_u2.y = *(uint*)&srcPtr[srcIdx2];
    srcNeighborhood_f12->f1[2] = validBL ? rpp_hip_unpack0(src_u2.x) : 0.0f;
    srcNeighborhood_f12->f1[3] = validBR ? rpp_hip_unpack0(src_u2.y) : 0.0f;
    srcNeighborhood_f12->f1[6] = validBL ? rpp_hip_unpack1(src_u2.x) : 0.0f;
    srcNeighborhood_f12->f1[7] = validBR ? rpp_hip_unpack1(src_u2.y) : 0.0f;
    srcNeighborhood_f12->f1[10] = validBL ? rpp_hip_unpack2(src_u2.x) : 0.0f;
    srcNeighborhood_f12->f1[11] = validBR ? rpp_hip_unpack2(src_u2.y) : 0.0f;
}

// F32 loads for bilinear interpolation (12 F32 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_load_pkd3(
    float* srcPtr, uint srcStrideH, float2* locSrcFloor_f2, int4* roiPtrSrc_i4,
    d_float12* srcNeighborhood_f12, bool checkRange) {
    float3 src1_f3, src2_f3;
    int2 locSrc1_i2, locSrc2_i2;
    bool validTL, validTR, validBL, validBR;
    rpp_hip_compute_bilinear_corner_validity(locSrcFloor_f2, roiPtrSrc_i4, checkRange, &validTL,
                                             &validTR, &validBL, &validBR);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc1_i2);
    *locSrcFloor_f2 = *locSrcFloor_f2 + MAKE_FLOAT2(1.0f);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc2_i2);
    int2 srcInterRowLoc_i2, srcInterColLoc_i2;
    srcInterRowLoc_i2.x = locSrc1_i2.y * srcStrideH;
    srcInterRowLoc_i2.y = locSrc2_i2.y * srcStrideH;
    srcInterColLoc_i2.x = locSrc1_i2.x * 3;
    srcInterColLoc_i2.y = locSrc2_i2.x * 3;

    int srcIdx1 = srcInterRowLoc_i2.x + srcInterColLoc_i2.x;  // Top Left
    int srcIdx2 = srcInterRowLoc_i2.x + srcInterColLoc_i2.y;  // Top Right
    src1_f3 = *(float3*)&srcPtr[srcIdx1];
    src2_f3 = *(float3*)&srcPtr[srcIdx2];
    srcNeighborhood_f12->f1[0] = validTL ? src1_f3.x : 0.0f;
    srcNeighborhood_f12->f1[1] = validTR ? src2_f3.x : 0.0f;
    srcNeighborhood_f12->f1[4] = validTL ? src1_f3.y : 0.0f;
    srcNeighborhood_f12->f1[5] = validTR ? src2_f3.y : 0.0f;
    srcNeighborhood_f12->f1[8] = validTL ? src1_f3.z : 0.0f;
    srcNeighborhood_f12->f1[9] = validTR ? src2_f3.z : 0.0f;
    srcIdx1 = srcInterRowLoc_i2.y + srcInterColLoc_i2.x;  // Bottom left
    srcIdx2 = srcInterRowLoc_i2.y + srcInterColLoc_i2.y;  // Bottom right
    src1_f3 = *(float3*)&srcPtr[srcIdx1];
    src2_f3 = *(float3*)&srcPtr[srcIdx2];
    srcNeighborhood_f12->f1[2] = validBL ? src1_f3.x : 0.0f;
    srcNeighborhood_f12->f1[3] = validBR ? src2_f3.x : 0.0f;
    srcNeighborhood_f12->f1[6] = validBL ? src1_f3.y : 0.0f;
    srcNeighborhood_f12->f1[7] = validBR ? src2_f3.y : 0.0f;
    srcNeighborhood_f12->f1[10] = validBL ? src1_f3.z : 0.0f;
    srcNeighborhood_f12->f1[11] = validBR ? src2_f3.z : 0.0f;
}

// I8 loads for bilinear interpolation (12 I8 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_load_pkd3(
    schar* srcPtr, uint srcStrideH, float2* locSrcFloor_f2, int4* roiPtrSrc_i4,
    d_float12* srcNeighborhood_f12, bool checkRange) {
    int2 src_u2, locSrc1_i2, locSrc2_i2;
    bool validTL, validTR, validBL, validBR;
    rpp_hip_compute_bilinear_corner_validity(locSrcFloor_f2, roiPtrSrc_i4, checkRange, &validTL,
                                             &validTR, &validBL, &validBR);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc1_i2);
    *locSrcFloor_f2 = *locSrcFloor_f2 + MAKE_FLOAT2(1.0f);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc2_i2);
    int2 srcInterRowLoc_i2, srcInterColLoc_i2;
    srcInterRowLoc_i2.x = locSrc1_i2.y * srcStrideH;
    srcInterRowLoc_i2.y = locSrc2_i2.y * srcStrideH;
    srcInterColLoc_i2.x = locSrc1_i2.x * 3;
    srcInterColLoc_i2.y = locSrc2_i2.x * 3;

    int srcIdx1 = srcInterRowLoc_i2.x + srcInterColLoc_i2.x;  // Top Left
    int srcIdx2 = srcInterRowLoc_i2.x + srcInterColLoc_i2.y;  // Top Right
    src_u2.x = *(int*)&srcPtr[srcIdx1];
    src_u2.y = *(int*)&srcPtr[srcIdx2];
    srcNeighborhood_f12->f1[0] = validTL ? rpp_hip_unpack0(src_u2.x) : 0.0f;
    srcNeighborhood_f12->f1[1] = validTR ? rpp_hip_unpack0(src_u2.y) : 0.0f;
    srcNeighborhood_f12->f1[4] = validTL ? rpp_hip_unpack1(src_u2.x) : 0.0f;
    srcNeighborhood_f12->f1[5] = validTR ? rpp_hip_unpack1(src_u2.y) : 0.0f;
    srcNeighborhood_f12->f1[8] = validTL ? rpp_hip_unpack2(src_u2.x) : 0.0f;
    srcNeighborhood_f12->f1[9] = validTR ? rpp_hip_unpack2(src_u2.y) : 0.0f;
    srcIdx1 = srcInterRowLoc_i2.y + srcInterColLoc_i2.x;  // Bottom left
    srcIdx2 = srcInterRowLoc_i2.y + srcInterColLoc_i2.y;  // Bottom right
    src_u2.x = *(int*)&srcPtr[srcIdx1];
    src_u2.y = *(int*)&srcPtr[srcIdx2];
    srcNeighborhood_f12->f1[2] = validBL ? rpp_hip_unpack0(src_u2.x) : 0.0f;
    srcNeighborhood_f12->f1[3] = validBR ? rpp_hip_unpack0(src_u2.y) : 0.0f;
    srcNeighborhood_f12->f1[6] = validBL ? rpp_hip_unpack1(src_u2.x) : 0.0f;
    srcNeighborhood_f12->f1[7] = validBR ? rpp_hip_unpack1(src_u2.y) : 0.0f;
    srcNeighborhood_f12->f1[10] = validBL ? rpp_hip_unpack2(src_u2.x) : 0.0f;
    srcNeighborhood_f12->f1[11] = validBR ? rpp_hip_unpack2(src_u2.y) : 0.0f;
}

// F16 loads for bilinear interpolation (12 F16 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_load_pkd3(
    half* srcPtr, uint srcStrideH, float2* locSrcFloor_f2, int4* roiPtrSrc_i4,
    d_float12* srcNeighborhood_f12, bool checkRange) {
    d_half3_s src1_h3, src2_h3;
    int2 locSrc1, locSrc2;
    bool validTL, validTR, validBL, validBR;
    rpp_hip_compute_bilinear_corner_validity(locSrcFloor_f2, roiPtrSrc_i4, checkRange, &validTL,
                                             &validTR, &validBL, &validBR);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc1);
    *locSrcFloor_f2 = *locSrcFloor_f2 + MAKE_FLOAT2(1.0f);
    rpp_hip_roi_range_check(locSrcFloor_f2, roiPtrSrc_i4, &locSrc2);
    int2 srcInterRowLoc_i2, srcInterColLoc_i2;
    srcInterRowLoc_i2.x = locSrc1.y * srcStrideH;
    srcInterRowLoc_i2.y = locSrc2.y * srcStrideH;
    srcInterColLoc_i2.x = locSrc1.x * 3;
    srcInterColLoc_i2.y = locSrc2.x * 3;

    int srcIdx1 = srcInterRowLoc_i2.x + srcInterColLoc_i2.x;  // Top Left
    int srcIdx2 = srcInterRowLoc_i2.x + srcInterColLoc_i2.y;  // Top Right
    src1_h3 = *(d_half3_s*)&srcPtr[srcIdx1];
    src2_h3 = *(d_half3_s*)&srcPtr[srcIdx2];
    srcNeighborhood_f12->f1[0] = validTL ? __half2float(src1_h3.h1[0]) : 0.0f;
    srcNeighborhood_f12->f1[1] = validTR ? __half2float(src2_h3.h1[0]) : 0.0f;
    srcNeighborhood_f12->f1[4] = validTL ? __half2float(src1_h3.h1[1]) : 0.0f;
    srcNeighborhood_f12->f1[5] = validTR ? __half2float(src2_h3.h1[1]) : 0.0f;
    srcNeighborhood_f12->f1[8] = validTL ? __half2float(src1_h3.h1[2]) : 0.0f;
    srcNeighborhood_f12->f1[9] = validTR ? __half2float(src2_h3.h1[2]) : 0.0f;
    srcIdx1 = srcInterRowLoc_i2.y + srcInterColLoc_i2.x;  // Top Left
    srcIdx2 = srcInterRowLoc_i2.y + srcInterColLoc_i2.y;  // Top Right
    src1_h3 = *(d_half3_s*)&srcPtr[srcIdx1];
    src2_h3 = *(d_half3_s*)&srcPtr[srcIdx2];
    srcNeighborhood_f12->f1[2] = validBL ? __half2float(src1_h3.h1[0]) : 0.0f;
    srcNeighborhood_f12->f1[3] = validBR ? __half2float(src2_h3.h1[0]) : 0.0f;
    srcNeighborhood_f12->f1[6] = validBL ? __half2float(src1_h3.h1[1]) : 0.0f;
    srcNeighborhood_f12->f1[7] = validBR ? __half2float(src2_h3.h1[1]) : 0.0f;
    srcNeighborhood_f12->f1[10] = validBL ? __half2float(src1_h3.h1[2]) : 0.0f;
    srcNeighborhood_f12->f1[11] = validBR ? __half2float(src2_h3.h1[2]) : 0.0f;
}

// BILINEAR INTERPOLATION EXECUTION HELPERS (templated execution routines for all bit depths)

// float bilinear interpolation computation

__device__ __forceinline__ void rpp_hip_interpolate_bilinear(float4* srcNeighborhood_f4,
                                                             float2* weightedWH_f2,
                                                             float2* oneMinusWeightedWH_f2,
                                                             float* dst) {
    *dst = fmaf(srcNeighborhood_f4->x, oneMinusWeightedWH_f2->y * oneMinusWeightedWH_f2->x,
                fmaf(srcNeighborhood_f4->y, oneMinusWeightedWH_f2->y * weightedWH_f2->x,
                     fmaf(srcNeighborhood_f4->z, weightedWH_f2->y * oneMinusWeightedWH_f2->x,
                          srcNeighborhood_f4->w * weightedWH_f2->y * weightedWH_f2->x)));
}

// float bilinear interpolation pln1

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate1_bilinear_pln1(T* srcPtr, uint srcStrideH,
                                                                   float locSrcX, float locSrcY,
                                                                   int4* roiPtrSrc_i4, float* dst,
                                                                   bool checkRange) {
    // Per-corner ROI validity (and the black-tap substitution for any corner outside it) is
    // handled inside the load function, so this always produces the right answer whether the
    // bilinear window is fully inside, straddling, or fully outside the ROI.
    float2 locSrcFloor_f2, weightedWH_f2, oneMinusWeightedWH_f2;
    locSrcFloor_f2.x = floorf(locSrcX);
    locSrcFloor_f2.y = floorf(locSrcY);
    weightedWH_f2.x = locSrcX - locSrcFloor_f2.x;
    weightedWH_f2.y = locSrcY - locSrcFloor_f2.y;
    oneMinusWeightedWH_f2.x = 1.0f - weightedWH_f2.x;
    oneMinusWeightedWH_f2.y = 1.0f - weightedWH_f2.y;
    float4 srcNeighborhood_f4;
    rpp_hip_interpolate1_bilinear_load_pln1(srcPtr, srcStrideH, &locSrcFloor_f2, roiPtrSrc_i4,
                                            &srcNeighborhood_f4, checkRange);
    rpp_hip_interpolate_bilinear(&srcNeighborhood_f4, &weightedWH_f2, &oneMinusWeightedWH_f2, dst);
}

// float3 bilinear interpolation pkd3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate3_bilinear_pkd3(T* srcPtr, uint srcStrideH,
                                                                   float locSrcX, float locSrcY,
                                                                   int4* roiPtrSrc_i4,
                                                                   float3* dst_f3,
                                                                   bool checkRange) {
    // Per-corner ROI validity (and the black-tap substitution for any corner outside it) is
    // handled inside the load function, so this always produces the right answer whether the
    // bilinear window is fully inside, straddling, or fully outside the ROI.
    float2 locSrcFloor_f2, weightedWH_f2, oneMinusWeightedWH_f2;
    locSrcFloor_f2.x = floorf(locSrcX);
    locSrcFloor_f2.y = floorf(locSrcY);
    weightedWH_f2.x = locSrcX - locSrcFloor_f2.x;
    weightedWH_f2.y = locSrcY - locSrcFloor_f2.y;
    oneMinusWeightedWH_f2.x = 1.0f - weightedWH_f2.x;
    oneMinusWeightedWH_f2.y = 1.0f - weightedWH_f2.y;
    d_float12 srcNeighborhood_f12;
    rpp_hip_interpolate3_bilinear_load_pkd3(srcPtr, srcStrideH, &locSrcFloor_f2, roiPtrSrc_i4,
                                            &srcNeighborhood_f12, checkRange);
    rpp_hip_interpolate_bilinear(&srcNeighborhood_f12.f4[0], &weightedWH_f2, &oneMinusWeightedWH_f2,
                                 &(dst_f3->x));
    rpp_hip_interpolate_bilinear(&srcNeighborhood_f12.f4[1], &weightedWH_f2, &oneMinusWeightedWH_f2,
                                 &(dst_f3->y));
    rpp_hip_interpolate_bilinear(&srcNeighborhood_f12.f4[2], &weightedWH_f2, &oneMinusWeightedWH_f2,
                                 &(dst_f3->z));
}

// d_float8 bilinear interpolation in pln1

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate8_bilinear_pln1(T* srcPtr, uint srcStrideH,
                                                                   d_float16* locPtrSrc_f16,
                                                                   int4* roiPtrSrc_i4,
                                                                   d_float8* dst_f8,
                                                                   bool checkRange = true) {
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[0],
                                       locPtrSrc_f16->f1[8], roiPtrSrc_i4, &(dst_f8->f1[0]),
                                       checkRange);
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[1],
                                       locPtrSrc_f16->f1[9], roiPtrSrc_i4, &(dst_f8->f1[1]),
                                       checkRange);
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[2],
                                       locPtrSrc_f16->f1[10], roiPtrSrc_i4, &(dst_f8->f1[2]),
                                       checkRange);
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[3],
                                       locPtrSrc_f16->f1[11], roiPtrSrc_i4, &(dst_f8->f1[3]),
                                       checkRange);
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[4],
                                       locPtrSrc_f16->f1[12], roiPtrSrc_i4, &(dst_f8->f1[4]),
                                       checkRange);
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[5],
                                       locPtrSrc_f16->f1[13], roiPtrSrc_i4, &(dst_f8->f1[5]),
                                       checkRange);
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[6],
                                       locPtrSrc_f16->f1[14], roiPtrSrc_i4, &(dst_f8->f1[6]),
                                       checkRange);
    rpp_hip_interpolate1_bilinear_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[7],
                                       locPtrSrc_f16->f1[15], roiPtrSrc_i4, &(dst_f8->f1[7]),
                                       checkRange);
}

// d_float24 bilinear interpolation in pln3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate24_bilinear_pln3(T* srcPtr, uint3* srcStridesNCH,
                                                                    d_float16* locPtrSrc_f16,
                                                                    int4* roiPtrSrc_i4,
                                                                    d_float24* dst_f24,
                                                                    bool checkRange = true) {
    rpp_hip_interpolate8_bilinear_pln1(srcPtr, srcStridesNCH->z, locPtrSrc_f16, roiPtrSrc_i4,
                                       &(dst_f24->f8[0]), checkRange);
    srcPtr += srcStridesNCH->y;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr, srcStridesNCH->z, locPtrSrc_f16, roiPtrSrc_i4,
                                       &(dst_f24->f8[1]), checkRange);
    srcPtr += srcStridesNCH->y;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr, srcStridesNCH->z, locPtrSrc_f16, roiPtrSrc_i4,
                                       &(dst_f24->f8[2]), checkRange);
}

// d_float24 bilinear interpolation in pkd3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate24_bilinear_pkd3(T* srcPtr, uint srcStrideH,
                                                                    d_float16* locPtrSrc_f16,
                                                                    int4* roiPtrSrc_i4,
                                                                    d_float24* dst_f24,
                                                                    bool checkRange = true) {
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[0],
                                       locPtrSrc_f16->f1[8], roiPtrSrc_i4, &(dst_f24->f3[0]),
                                       checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[1],
                                       locPtrSrc_f16->f1[9], roiPtrSrc_i4, &(dst_f24->f3[1]),
                                       checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[2],
                                       locPtrSrc_f16->f1[10], roiPtrSrc_i4, &(dst_f24->f3[2]),
                                       checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[3],
                                       locPtrSrc_f16->f1[11], roiPtrSrc_i4, &(dst_f24->f3[3]),
                                       checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[4],
                                       locPtrSrc_f16->f1[12], roiPtrSrc_i4, &(dst_f24->f3[4]),
                                       checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[5],
                                       locPtrSrc_f16->f1[13], roiPtrSrc_i4, &(dst_f24->f3[5]),
                                       checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[6],
                                       locPtrSrc_f16->f1[14], roiPtrSrc_i4, &(dst_f24->f3[6]),
                                       checkRange);
    rpp_hip_interpolate3_bilinear_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[7],
                                       locPtrSrc_f16->f1[15], roiPtrSrc_i4, &(dst_f24->f3[7]),
                                       checkRange);
}

// NEAREST NEIGHBOR INTERPOLATION LOAD HELPERS (separate load routines for each bit depth)

// U8 loads for nearest_neighbor interpolation (1 U8 pixel)

__device__ __forceinline__ void rpp_hip_interpolate1_nearest_neighbor_load_pln1(uchar* srcPtr,
                                                                                float* dstPtr) {
    uint src = *(uint*)srcPtr;
    *dstPtr = rpp_hip_unpack0(src);
}

// F32 loads for nearest_neighbor interpolation (1 F32 pixel)

__device__ __forceinline__ void rpp_hip_interpolate1_nearest_neighbor_load_pln1(float* srcPtr,
                                                                                float* dstPtr) {
    *dstPtr = *srcPtr;
}

// I8 loads for nearest_neighbor interpolation (1 I8 pixel)

__device__ __forceinline__ void rpp_hip_interpolate1_nearest_neighbor_load_pln1(schar* srcPtr,
                                                                                float* dstPtr) {
    int src = *(int*)srcPtr;
    *dstPtr = rpp_hip_unpack0(src);
}

// F16 loads for nearest_neighbor interpolation (1 F16 pixel)

__device__ __forceinline__ void rpp_hip_interpolate1_nearest_neighbor_load_pln1(half* srcPtr,
                                                                                float* dstPtr) {
    *dstPtr = __half2float(*srcPtr);
}

// U8 loads for nearest_neighbor interpolation (3 U8 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_nearest_neighbor_load_pkd3(uchar* srcPtr,
                                                                                float3* dstPtr_f3) {
    uint src = *(uint*)srcPtr;
    *dstPtr_f3 = make_float3(rpp_hip_unpack0(src), rpp_hip_unpack1(src), rpp_hip_unpack2(src));
}

// F32 loads for nearest_neighbor interpolation (3 F32 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_nearest_neighbor_load_pkd3(float* srcPtr,
                                                                                float3* dstPtr_f3) {
    float3 src_f3 = *(float3*)srcPtr;
    *dstPtr_f3 = src_f3;
}

// I8 loads for nearest_neighbor interpolation (3 I8 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_nearest_neighbor_load_pkd3(schar* srcPtr,
                                                                                float3* dstPtr_f3) {
    int src = *(int*)srcPtr;
    *dstPtr_f3 = make_float3(rpp_hip_unpack0(src), rpp_hip_unpack1(src), rpp_hip_unpack2(src));
}

// F16 loads for nearest_neighbor interpolation (3 F16 pixels)

__device__ __forceinline__ void rpp_hip_interpolate3_nearest_neighbor_load_pkd3(half* srcPtr,
                                                                                float3* dstPtr_f3) {
    d_half3_s src_h3 = *(d_half3_s*)srcPtr;
    dstPtr_f3->x = __half2float(src_h3.h1[0]);
    dstPtr_f3->y = __half2float(src_h3.h1[1]);
    dstPtr_f3->z = __half2float(src_h3.h1[2]);
}

// NEAREST NEIGHBOR INTERPOLATION EXECUTION HELPERS (templated execution routines for all bit
// depths)

// float nearest neighbor interpolation pln1

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate1_nearest_neighbor_pln1(
    T* srcPtr, uint srcStrideH, float locSrcX, float locSrcY, int4* roiPtrSrc_i4, float* dst) {
    int2 locSrc;
    locSrc.x = roundf(locSrcX);
    locSrc.y = roundf(locSrcY);

    if ((locSrc.x < roiPtrSrc_i4->x) || (locSrc.y < roiPtrSrc_i4->y) ||
        (locSrc.x > roiPtrSrc_i4->z) || (locSrc.y > roiPtrSrc_i4->w)) {
        *dst = 0.0f;
    } else {
        int srcIdx = locSrc.y * srcStrideH + locSrc.x;
        rpp_hip_interpolate1_nearest_neighbor_load_pln1(srcPtr + srcIdx, dst);
    }
}

// float3 nearest neighbor interpolation pkd3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate3_nearest_neighbor_pkd3(
    T* srcPtr, uint srcStrideH, float locSrcX, float locSrcY, int4* roiPtrSrc_i4, float3* dst_f3) {
    int2 locSrc;
    locSrc.x = roundf(locSrcX);
    locSrc.y = roundf(locSrcY);

    if ((locSrc.x < roiPtrSrc_i4->x) || (locSrc.y < roiPtrSrc_i4->y) ||
        (locSrc.x > roiPtrSrc_i4->z) || (locSrc.y > roiPtrSrc_i4->w)) {
        *dst_f3 = MAKE_FLOAT3(0.0f);
    } else {
        int srcIdx = locSrc.y * srcStrideH + locSrc.x * 3;
        rpp_hip_interpolate3_nearest_neighbor_load_pkd3(srcPtr + srcIdx, dst_f3);
    }
}

// d_float8 nearest neighbor interpolation in pln1

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate8_nearest_neighbor_pln1(
    T* srcPtr, uint srcStrideH, d_float16* locPtrSrc_f16, int4* roiPtrSrc_i4, d_float8* dst_f8) {
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[0],
                                               locPtrSrc_f16->f1[8], roiPtrSrc_i4,
                                               &(dst_f8->f1[0]));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[1],
                                               locPtrSrc_f16->f1[9], roiPtrSrc_i4,
                                               &(dst_f8->f1[1]));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[2],
                                               locPtrSrc_f16->f1[10], roiPtrSrc_i4,
                                               &(dst_f8->f1[2]));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[3],
                                               locPtrSrc_f16->f1[11], roiPtrSrc_i4,
                                               &(dst_f8->f1[3]));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[4],
                                               locPtrSrc_f16->f1[12], roiPtrSrc_i4,
                                               &(dst_f8->f1[4]));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[5],
                                               locPtrSrc_f16->f1[13], roiPtrSrc_i4,
                                               &(dst_f8->f1[5]));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[6],
                                               locPtrSrc_f16->f1[14], roiPtrSrc_i4,
                                               &(dst_f8->f1[6]));
    rpp_hip_interpolate1_nearest_neighbor_pln1(srcPtr, srcStrideH, locPtrSrc_f16->f1[7],
                                               locPtrSrc_f16->f1[15], roiPtrSrc_i4,
                                               &(dst_f8->f1[7]));
}

// d_float24 nearest neighbor interpolation in pln3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate24_nearest_neighbor_pln3(
    T* srcPtr, uint3* srcStridesNCH, d_float16* locPtrSrc_f16, int4* roiPtrSrc_i4,
    d_float24* dst_f24) {
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr, srcStridesNCH->z, locPtrSrc_f16,
                                               roiPtrSrc_i4, &(dst_f24->f8[0]));
    srcPtr += srcStridesNCH->y;
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr, srcStridesNCH->z, locPtrSrc_f16,
                                               roiPtrSrc_i4, &(dst_f24->f8[1]));
    srcPtr += srcStridesNCH->y;
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr, srcStridesNCH->z, locPtrSrc_f16,
                                               roiPtrSrc_i4, &(dst_f24->f8[2]));
}

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate24_nearest_neighbor_pln3(
    T* srcPtr, uint2* srcStridesCH, d_float16* locPtrSrc_f16, int4* roiPtrSrc_i4,
    d_float24* dst_f24) {
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr, srcStridesCH->y, locPtrSrc_f16, roiPtrSrc_i4,
                                               &(dst_f24->f8[0]));
    srcPtr += srcStridesCH->x;
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr, srcStridesCH->y, locPtrSrc_f16, roiPtrSrc_i4,
                                               &(dst_f24->f8[1]));
    srcPtr += srcStridesCH->x;
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr, srcStridesCH->y, locPtrSrc_f16, roiPtrSrc_i4,
                                               &(dst_f24->f8[2]));
}

// d_float24 nearest neighbor interpolation in pkd3

template <typename T>
__device__ __forceinline__ void rpp_hip_interpolate24_nearest_neighbor_pkd3(
    T* srcPtr, uint srcStrideH, d_float16* locPtrSrc_f16, int4* roiPtrSrc_i4, d_float24* dst_f24) {
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[0],
                                               locPtrSrc_f16->f1[8], roiPtrSrc_i4,
                                               &(dst_f24->f3[0]));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[1],
                                               locPtrSrc_f16->f1[9], roiPtrSrc_i4,
                                               &(dst_f24->f3[1]));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[2],
                                               locPtrSrc_f16->f1[10], roiPtrSrc_i4,
                                               &(dst_f24->f3[2]));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[3],
                                               locPtrSrc_f16->f1[11], roiPtrSrc_i4,
                                               &(dst_f24->f3[3]));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[4],
                                               locPtrSrc_f16->f1[12], roiPtrSrc_i4,
                                               &(dst_f24->f3[4]));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[5],
                                               locPtrSrc_f16->f1[13], roiPtrSrc_i4,
                                               &(dst_f24->f3[5]));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[6],
                                               locPtrSrc_f16->f1[14], roiPtrSrc_i4,
                                               &(dst_f24->f3[6]));
    rpp_hip_interpolate3_nearest_neighbor_pkd3(srcPtr, srcStrideH, locPtrSrc_f16->f1[7],
                                               locPtrSrc_f16->f1[15], roiPtrSrc_i4,
                                               &(dst_f24->f3[7]));
}

#endif  // RPP_HIP_INTERPOLATION_HPP
