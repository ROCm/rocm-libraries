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

// Gridmask helper - Computing row and column ratios

__device__ void gridmask_ratio_hip_compute(int id_x, int id_y, float2 *rotateRatios_f2, float2 *translateRatios_f2, float2 *gridRowRatio_f2, d_float16 *gridColRatio_f16)
{
    gridRowRatio_f2->x = fmaf(id_y, -rotateRatios_f2->y, -translateRatios_f2->x);
    gridRowRatio_f2->y = fmaf(id_y, rotateRatios_f2->x, -translateRatios_f2->y);

    int id_x_vector[8];
    id_x_vector[0] = id_x;
    id_x_vector[1] = id_x + 1;
    id_x_vector[2] = id_x + 2;
    id_x_vector[3] = id_x + 3;
    id_x_vector[4] = id_x + 4;
    id_x_vector[5] = id_x + 5;
    id_x_vector[6] = id_x + 6;
    id_x_vector[7] = id_x + 7;

    gridColRatio_f16->f1[ 0] = fmaf(id_x_vector[0], rotateRatios_f2->x, gridRowRatio_f2->x);
    gridColRatio_f16->f1[ 1] = fmaf(id_x_vector[1], rotateRatios_f2->x, gridRowRatio_f2->x);
    gridColRatio_f16->f1[ 2] = fmaf(id_x_vector[2], rotateRatios_f2->x, gridRowRatio_f2->x);
    gridColRatio_f16->f1[ 3] = fmaf(id_x_vector[3], rotateRatios_f2->x, gridRowRatio_f2->x);
    gridColRatio_f16->f1[ 4] = fmaf(id_x_vector[4], rotateRatios_f2->x, gridRowRatio_f2->x);
    gridColRatio_f16->f1[ 5] = fmaf(id_x_vector[5], rotateRatios_f2->x, gridRowRatio_f2->x);
    gridColRatio_f16->f1[ 6] = fmaf(id_x_vector[6], rotateRatios_f2->x, gridRowRatio_f2->x);
    gridColRatio_f16->f1[ 7] = fmaf(id_x_vector[7], rotateRatios_f2->x, gridRowRatio_f2->x);

    gridColRatio_f16->f1[ 8] = fmaf(id_x_vector[0], rotateRatios_f2->y, gridRowRatio_f2->y);
    gridColRatio_f16->f1[ 9] = fmaf(id_x_vector[1], rotateRatios_f2->y, gridRowRatio_f2->y);
    gridColRatio_f16->f1[10] = fmaf(id_x_vector[2], rotateRatios_f2->y, gridRowRatio_f2->y);
    gridColRatio_f16->f1[11] = fmaf(id_x_vector[3], rotateRatios_f2->y, gridRowRatio_f2->y);
    gridColRatio_f16->f1[12] = fmaf(id_x_vector[4], rotateRatios_f2->y, gridRowRatio_f2->y);
    gridColRatio_f16->f1[13] = fmaf(id_x_vector[5], rotateRatios_f2->y, gridRowRatio_f2->y);
    gridColRatio_f16->f1[14] = fmaf(id_x_vector[6], rotateRatios_f2->y, gridRowRatio_f2->y);
    gridColRatio_f16->f1[15] = fmaf(id_x_vector[7], rotateRatios_f2->y, gridRowRatio_f2->y);
}

// Gridmask helpers - Vector masked store computes

__device__ void gridmask_vector_masked_store8_hip_compute(d_uchar8 *srcPtr_uc8, d_uchar8 *dstPtr_uc8, d_float16 *gridColRatio_f16, float gridRatio)
{
    dstPtr_uc8->uc1[0] = ((gridColRatio_f16->f1[0] >= gridRatio) || (gridColRatio_f16->f1[ 8] >= gridRatio)) ? srcPtr_uc8->uc1[0] : (uchar)0;
    dstPtr_uc8->uc1[1] = ((gridColRatio_f16->f1[1] >= gridRatio) || (gridColRatio_f16->f1[ 9] >= gridRatio)) ? srcPtr_uc8->uc1[1] : (uchar)0;
    dstPtr_uc8->uc1[2] = ((gridColRatio_f16->f1[2] >= gridRatio) || (gridColRatio_f16->f1[10] >= gridRatio)) ? srcPtr_uc8->uc1[2] : (uchar)0;
    dstPtr_uc8->uc1[3] = ((gridColRatio_f16->f1[3] >= gridRatio) || (gridColRatio_f16->f1[11] >= gridRatio)) ? srcPtr_uc8->uc1[3] : (uchar)0;
    dstPtr_uc8->uc1[4] = ((gridColRatio_f16->f1[4] >= gridRatio) || (gridColRatio_f16->f1[12] >= gridRatio)) ? srcPtr_uc8->uc1[4] : (uchar)0;
    dstPtr_uc8->uc1[5] = ((gridColRatio_f16->f1[5] >= gridRatio) || (gridColRatio_f16->f1[13] >= gridRatio)) ? srcPtr_uc8->uc1[5] : (uchar)0;
    dstPtr_uc8->uc1[6] = ((gridColRatio_f16->f1[6] >= gridRatio) || (gridColRatio_f16->f1[14] >= gridRatio)) ? srcPtr_uc8->uc1[6] : (uchar)0;
    dstPtr_uc8->uc1[7] = ((gridColRatio_f16->f1[7] >= gridRatio) || (gridColRatio_f16->f1[15] >= gridRatio)) ? srcPtr_uc8->uc1[7] : (uchar)0;
}
__device__ void gridmask_vector_masked_store8_hip_compute(d_float8 *srcPtr_f8, d_float8 *dstPtr_f8, d_float16 *gridColRatio_f16, float gridRatio)
{
    dstPtr_f8->f1[0] = ((gridColRatio_f16->f1[0] >= gridRatio) || (gridColRatio_f16->f1[ 8] >= gridRatio)) ? srcPtr_f8->f1[0] : 0.0f;
    dstPtr_f8->f1[1] = ((gridColRatio_f16->f1[1] >= gridRatio) || (gridColRatio_f16->f1[ 9] >= gridRatio)) ? srcPtr_f8->f1[1] : 0.0f;
    dstPtr_f8->f1[2] = ((gridColRatio_f16->f1[2] >= gridRatio) || (gridColRatio_f16->f1[10] >= gridRatio)) ? srcPtr_f8->f1[2] : 0.0f;
    dstPtr_f8->f1[3] = ((gridColRatio_f16->f1[3] >= gridRatio) || (gridColRatio_f16->f1[11] >= gridRatio)) ? srcPtr_f8->f1[3] : 0.0f;
    dstPtr_f8->f1[4] = ((gridColRatio_f16->f1[4] >= gridRatio) || (gridColRatio_f16->f1[12] >= gridRatio)) ? srcPtr_f8->f1[4] : 0.0f;
    dstPtr_f8->f1[5] = ((gridColRatio_f16->f1[5] >= gridRatio) || (gridColRatio_f16->f1[13] >= gridRatio)) ? srcPtr_f8->f1[5] : 0.0f;
    dstPtr_f8->f1[6] = ((gridColRatio_f16->f1[6] >= gridRatio) || (gridColRatio_f16->f1[14] >= gridRatio)) ? srcPtr_f8->f1[6] : 0.0f;
    dstPtr_f8->f1[7] = ((gridColRatio_f16->f1[7] >= gridRatio) || (gridColRatio_f16->f1[15] >= gridRatio)) ? srcPtr_f8->f1[7] : 0.0f;
}
__device__ void gridmask_vector_masked_store8_hip_compute(d_schar8_s *srcPtr_sc8, d_schar8_s *dstPtr_sc8, d_float16 *gridColRatio_f16, float gridRatio)
{
    dstPtr_sc8->sc1[0] = ((gridColRatio_f16->f1[0] >= gridRatio) || (gridColRatio_f16->f1[ 8] >= gridRatio)) ? srcPtr_sc8->sc1[0] : (schar)-128;
    dstPtr_sc8->sc1[1] = ((gridColRatio_f16->f1[1] >= gridRatio) || (gridColRatio_f16->f1[ 9] >= gridRatio)) ? srcPtr_sc8->sc1[1] : (schar)-128;
    dstPtr_sc8->sc1[2] = ((gridColRatio_f16->f1[2] >= gridRatio) || (gridColRatio_f16->f1[10] >= gridRatio)) ? srcPtr_sc8->sc1[2] : (schar)-128;
    dstPtr_sc8->sc1[3] = ((gridColRatio_f16->f1[3] >= gridRatio) || (gridColRatio_f16->f1[11] >= gridRatio)) ? srcPtr_sc8->sc1[3] : (schar)-128;
    dstPtr_sc8->sc1[4] = ((gridColRatio_f16->f1[4] >= gridRatio) || (gridColRatio_f16->f1[12] >= gridRatio)) ? srcPtr_sc8->sc1[4] : (schar)-128;
    dstPtr_sc8->sc1[5] = ((gridColRatio_f16->f1[5] >= gridRatio) || (gridColRatio_f16->f1[13] >= gridRatio)) ? srcPtr_sc8->sc1[5] : (schar)-128;
    dstPtr_sc8->sc1[6] = ((gridColRatio_f16->f1[6] >= gridRatio) || (gridColRatio_f16->f1[14] >= gridRatio)) ? srcPtr_sc8->sc1[6] : (schar)-128;
    dstPtr_sc8->sc1[7] = ((gridColRatio_f16->f1[7] >= gridRatio) || (gridColRatio_f16->f1[15] >= gridRatio)) ? srcPtr_sc8->sc1[7] : (schar)-128;
}
__device__ void gridmask_vector_masked_store8_hip_compute(d_half8 *srcPtr_h8, d_half8 *dstPtr_h8, d_float16 *gridColRatio_f16, float gridRatio)
{
    dstPtr_h8->h1[0] = ((gridColRatio_f16->f1[0] >= gridRatio) || (gridColRatio_f16->f1[ 8] >= gridRatio)) ? srcPtr_h8->h1[0] : (half)0.0f;
    dstPtr_h8->h1[1] = ((gridColRatio_f16->f1[1] >= gridRatio) || (gridColRatio_f16->f1[ 9] >= gridRatio)) ? srcPtr_h8->h1[1] : (half)0.0f;
    dstPtr_h8->h1[2] = ((gridColRatio_f16->f1[2] >= gridRatio) || (gridColRatio_f16->f1[10] >= gridRatio)) ? srcPtr_h8->h1[2] : (half)0.0f;
    dstPtr_h8->h1[3] = ((gridColRatio_f16->f1[3] >= gridRatio) || (gridColRatio_f16->f1[11] >= gridRatio)) ? srcPtr_h8->h1[3] : (half)0.0f;
    dstPtr_h8->h1[4] = ((gridColRatio_f16->f1[4] >= gridRatio) || (gridColRatio_f16->f1[12] >= gridRatio)) ? srcPtr_h8->h1[4] : (half)0.0f;
    dstPtr_h8->h1[5] = ((gridColRatio_f16->f1[5] >= gridRatio) || (gridColRatio_f16->f1[13] >= gridRatio)) ? srcPtr_h8->h1[5] : (half)0.0f;
    dstPtr_h8->h1[6] = ((gridColRatio_f16->f1[6] >= gridRatio) || (gridColRatio_f16->f1[14] >= gridRatio)) ? srcPtr_h8->h1[6] : (half)0.0f;
    dstPtr_h8->h1[7] = ((gridColRatio_f16->f1[7] >= gridRatio) || (gridColRatio_f16->f1[15] >= gridRatio)) ? srcPtr_h8->h1[7] : (half)0.0f;
}

// Gridmask helpers for different data layouts

// PKD3 -> PKD3
__device__ void gridmask_result_pkd3_pkd3_hip_compute(uchar *srcPtr, uchar *dstPtr, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_uchar24 src_uc24, dst_uc24;
    *(d_uchar24_s *)&src_uc24 = *(d_uchar24_s *)srcPtr;
    rpp_hip_layouttoggle24_pkd3_to_pln3((d_uchar24_s *)&src_uc24);
    gridmask_vector_masked_store8_hip_compute(&src_uc24.uc8[0], &dst_uc24.uc8[0], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_uc24.uc8[1], &dst_uc24.uc8[1], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_uc24.uc8[2], &dst_uc24.uc8[2], gridColRatio_f16, gridRatio);
    rpp_hip_layouttoggle24_pln3_to_pkd3((d_uchar24_s *)&dst_uc24);
    *(d_uchar24_s *)dstPtr = *(d_uchar24_s *)&dst_uc24;
}
__device__ void gridmask_result_pkd3_pkd3_hip_compute(float *srcPtr, float *dstPtr, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_float24 src_f24, dst_f24;
    *(d_float24_s *)&src_f24 = *(d_float24_s *)srcPtr;
    rpp_hip_layouttoggle24_pkd3_to_pln3((d_float24_s *)&src_f24);
    gridmask_vector_masked_store8_hip_compute(&src_f24.f8[0], &dst_f24.f8[0], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_f24.f8[1], &dst_f24.f8[1], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_f24.f8[2], &dst_f24.f8[2], gridColRatio_f16, gridRatio);
    rpp_hip_layouttoggle24_pln3_to_pkd3((d_float24_s *)&dst_f24);
    *(d_float24_s *)dstPtr = *(d_float24_s *)&dst_f24;
}
__device__ void gridmask_result_pkd3_pkd3_hip_compute(schar *srcPtr, schar *dstPtr, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_schar24_s src_sc24, dst_sc24;
    src_sc24 = *(d_schar24_s *)srcPtr;
    rpp_hip_layouttoggle24_pkd3_to_pln3((d_schar24sc1s_s *)&src_sc24);
    gridmask_vector_masked_store8_hip_compute(&src_sc24.sc8[0], &dst_sc24.sc8[0], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_sc24.sc8[1], &dst_sc24.sc8[1], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_sc24.sc8[2], &dst_sc24.sc8[2], gridColRatio_f16, gridRatio);
    rpp_hip_layouttoggle24_pln3_to_pkd3((d_schar24sc1s_s *)&dst_sc24);
    *(d_schar24_s *)dstPtr = dst_sc24;
}
__device__ void gridmask_result_pkd3_pkd3_hip_compute(half *srcPtr, half *dstPtr, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_half24 src_h24, dst_h24;
    src_h24 = *(d_half24 *)srcPtr;
    rpp_hip_layouttoggle24_pkd3_to_pln3((d_half24_s *)&src_h24);
    gridmask_vector_masked_store8_hip_compute(&src_h24.h8[0], &dst_h24.h8[0], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_h24.h8[1], &dst_h24.h8[1], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_h24.h8[2], &dst_h24.h8[2], gridColRatio_f16, gridRatio);
    rpp_hip_layouttoggle24_pln3_to_pkd3((d_half24_s *)&dst_h24);
    *(d_half24 *)dstPtr = dst_h24;
}

// PLN3 -> PLN3
__device__ void gridmask_result_pln3_pln3_hip_compute(uchar *srcPtr, uint srcStrideC, uchar *dstPtr, uint dstStrideC, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_uchar8 src_uc8, dst_uc8;
    *(uint2 *)&src_uc8 = *(uint2 *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_uc8, &dst_uc8, gridColRatio_f16, gridRatio);
    *(uint2 *)dstPtr = *(uint2 *)&dst_uc8;
    srcPtr += srcStrideC;
    dstPtr += dstStrideC;
    *(uint2 *)&src_uc8 = *(uint2 *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_uc8, &dst_uc8, gridColRatio_f16, gridRatio);
    *(uint2 *)dstPtr = *(uint2 *)&dst_uc8;
    srcPtr += srcStrideC;
    dstPtr += dstStrideC;
    *(uint2 *)&src_uc8 = *(uint2 *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_uc8, &dst_uc8, gridColRatio_f16, gridRatio);
    *(uint2 *)dstPtr = *(uint2 *)&dst_uc8;
}
__device__ void gridmask_result_pln3_pln3_hip_compute(float *srcPtr, uint srcStrideC, float *dstPtr, uint dstStrideC, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_float8 src_f8, dst_f8;
    *(d_float8_s *)&src_f8 = *(d_float8_s *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_f8, &dst_f8, gridColRatio_f16, gridRatio);
    *(d_float8_s *)dstPtr = *(d_float8_s *)&dst_f8;
    srcPtr += srcStrideC;
    dstPtr += dstStrideC;
    *(d_float8_s *)&src_f8 = *(d_float8_s *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_f8, &dst_f8, gridColRatio_f16, gridRatio);
    *(d_float8_s *)dstPtr = *(d_float8_s *)&dst_f8;
    srcPtr += srcStrideC;
    dstPtr += dstStrideC;
    *(d_float8_s *)&src_f8 = *(d_float8_s *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_f8, &dst_f8, gridColRatio_f16, gridRatio);
    *(d_float8_s *)dstPtr = *(d_float8_s *)&dst_f8;
}
__device__ void gridmask_result_pln3_pln3_hip_compute(schar *srcPtr, uint srcStrideC, schar *dstPtr, uint dstStrideC, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_schar8_s src_sc8, dst_sc8;
    src_sc8 = *(d_schar8_s *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_sc8, &dst_sc8, gridColRatio_f16, gridRatio);
    *(d_schar8_s *)dstPtr = dst_sc8;
    srcPtr += srcStrideC;
    dstPtr += dstStrideC;
    src_sc8 = *(d_schar8_s *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_sc8, &dst_sc8, gridColRatio_f16, gridRatio);
    *(d_schar8_s *)dstPtr = dst_sc8;
    srcPtr += srcStrideC;
    dstPtr += dstStrideC;
    src_sc8 = *(d_schar8_s *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_sc8, &dst_sc8, gridColRatio_f16, gridRatio);
    *(d_schar8_s *)dstPtr = dst_sc8;
}
__device__ void gridmask_result_pln3_pln3_hip_compute(half *srcPtr, uint srcStrideC, half *dstPtr, uint dstStrideC, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_half8 src_h8, dst_h8;
    src_h8 = *(d_half8 *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_h8, &dst_h8, gridColRatio_f16, gridRatio);
    *(d_half8 *)dstPtr = dst_h8;
    srcPtr += srcStrideC;
    dstPtr += dstStrideC;
    src_h8 = *(d_half8 *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_h8, &dst_h8, gridColRatio_f16, gridRatio);
    *(d_half8 *)dstPtr = dst_h8;
    srcPtr += srcStrideC;
    dstPtr += dstStrideC;
    src_h8 = *(d_half8 *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_h8, &dst_h8, gridColRatio_f16, gridRatio);
    *(d_half8 *)dstPtr = dst_h8;
}

// PLN1 -> PLN1
__device__ void gridmask_result_pln1_pln1_hip_compute(uchar *srcPtr, uchar *dstPtr, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_uchar8 src_uc8, dst_uc8;
    *(uint2 *)&src_uc8 = *(uint2 *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_uc8, &dst_uc8, gridColRatio_f16, gridRatio);
    *(uint2 *)dstPtr = *(uint2 *)&dst_uc8;
}
__device__ void gridmask_result_pln1_pln1_hip_compute(float *srcPtr, float *dstPtr, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_float8 src_f8, dst_f8;
    *(d_float8_s *)&src_f8 = *(d_float8_s *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_f8, &dst_f8, gridColRatio_f16, gridRatio);
    *(d_float8_s *)dstPtr = *(d_float8_s *)&dst_f8;
}
__device__ void gridmask_result_pln1_pln1_hip_compute(schar *srcPtr, schar *dstPtr, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_schar8_s src_sc8, dst_sc8;
    src_sc8 = *(d_schar8_s *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_sc8, &dst_sc8, gridColRatio_f16, gridRatio);
    *(d_schar8_s *)dstPtr = dst_sc8;
}
__device__ void gridmask_result_pln1_pln1_hip_compute(half *srcPtr, half *dstPtr, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_half8 src_h8, dst_h8;
    src_h8 = *(d_half8 *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_h8, &dst_h8, gridColRatio_f16, gridRatio);
    *(d_half8 *)dstPtr = dst_h8;
}

// PKD3 -> PLN3
__device__ void gridmask_result_pkd3_pln3_hip_compute(uchar *srcPtr, uchar *dstPtr, uint dstStrideC, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_uchar24 src_uc24, dst_uc24;
    *(d_uchar24_s *)&src_uc24 = *(d_uchar24_s *)srcPtr;
    rpp_hip_layouttoggle24_pkd3_to_pln3((d_uchar24_s *)&src_uc24);
    gridmask_vector_masked_store8_hip_compute(&src_uc24.uc8[0], &dst_uc24.uc8[0], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_uc24.uc8[1], &dst_uc24.uc8[1], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_uc24.uc8[2], &dst_uc24.uc8[2], gridColRatio_f16, gridRatio);
    *(uint2 *)dstPtr = *(uint2 *)&dst_uc24.uc8[0];
    dstPtr += dstStrideC;
    *(uint2 *)dstPtr = *(uint2 *)&dst_uc24.uc8[1];
    dstPtr += dstStrideC;
    *(uint2 *)dstPtr = *(uint2 *)&dst_uc24.uc8[2];
}
__device__ void gridmask_result_pkd3_pln3_hip_compute(float *srcPtr, float *dstPtr, uint dstStrideC, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_float24 src_f24, dst_f24;
    *(d_float24_s *)&src_f24 = *(d_float24_s *)srcPtr;
    rpp_hip_layouttoggle24_pkd3_to_pln3((d_float24_s *)&src_f24);
    gridmask_vector_masked_store8_hip_compute(&src_f24.f8[0], &dst_f24.f8[0], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_f24.f8[1], &dst_f24.f8[1], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_f24.f8[2], &dst_f24.f8[2], gridColRatio_f16, gridRatio);
    *(d_float8_s *)dstPtr = *(d_float8_s *)&dst_f24.f8[0];
    dstPtr += dstStrideC;
    *(d_float8_s *)dstPtr = *(d_float8_s *)&dst_f24.f8[1];
    dstPtr += dstStrideC;
    *(d_float8_s *)dstPtr = *(d_float8_s *)&dst_f24.f8[2];
}
__device__ void gridmask_result_pkd3_pln3_hip_compute(schar *srcPtr, schar *dstPtr, uint dstStrideC, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_schar24_s src_sc24, dst_sc24;
    src_sc24 = *(d_schar24_s *)srcPtr;
    rpp_hip_layouttoggle24_pkd3_to_pln3((d_schar24sc1s_s *)&src_sc24);
    gridmask_vector_masked_store8_hip_compute(&src_sc24.sc8[0], &dst_sc24.sc8[0], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_sc24.sc8[1], &dst_sc24.sc8[1], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_sc24.sc8[2], &dst_sc24.sc8[2], gridColRatio_f16, gridRatio);
    *(d_schar8_s *)dstPtr = dst_sc24.sc8[0];
    dstPtr += dstStrideC;
    *(d_schar8_s *)dstPtr = dst_sc24.sc8[1];
    dstPtr += dstStrideC;
    *(d_schar8_s *)dstPtr = dst_sc24.sc8[2];
}
__device__ void gridmask_result_pkd3_pln3_hip_compute(half *srcPtr, half *dstPtr, uint dstStrideC, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_half24 src_h24, dst_h24;
    src_h24 = *(d_half24 *)srcPtr;
    rpp_hip_layouttoggle24_pkd3_to_pln3((d_half24_s *)&src_h24);
    gridmask_vector_masked_store8_hip_compute(&src_h24.h8[0], &dst_h24.h8[0], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_h24.h8[1], &dst_h24.h8[1], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_h24.h8[2], &dst_h24.h8[2], gridColRatio_f16, gridRatio);
    *(d_half8 *)dstPtr = dst_h24.h8[0];
    dstPtr += dstStrideC;
    *(d_half8 *)dstPtr = dst_h24.h8[1];
    dstPtr += dstStrideC;
    *(d_half8 *)dstPtr = dst_h24.h8[2];
}

// PLN3 -> PKD3
__device__ void gridmask_result_pln3_pkd3_hip_compute(uchar *srcPtr, uint srcStrideC, uchar *dstPtr, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_uchar24 src_uc24, dst_uc24;
    *(uint2 *)&src_uc24.uc8[0] = *(uint2 *)srcPtr;
    srcPtr += srcStrideC;
    *(uint2 *)&src_uc24.uc8[1] = *(uint2 *)srcPtr;
    srcPtr += srcStrideC;
    *(uint2 *)&src_uc24.uc8[2] = *(uint2 *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_uc24.uc8[0], &dst_uc24.uc8[0], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_uc24.uc8[1], &dst_uc24.uc8[1], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_uc24.uc8[2], &dst_uc24.uc8[2], gridColRatio_f16, gridRatio);
    rpp_hip_layouttoggle24_pln3_to_pkd3((d_uchar24_s *)&dst_uc24);
    *(d_uchar24_s *)dstPtr = *(d_uchar24_s *)&dst_uc24;
}
__device__ void gridmask_result_pln3_pkd3_hip_compute(float *srcPtr, uint srcStrideC, float *dstPtr, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_float24 src_f24, dst_f24;
    *(d_float8_s *)&src_f24.f8[0] = *(d_float8_s *)srcPtr;
    srcPtr += srcStrideC;
    *(d_float8_s *)&src_f24.f8[1] = *(d_float8_s *)srcPtr;
    srcPtr += srcStrideC;
    *(d_float8_s *)&src_f24.f8[2] = *(d_float8_s *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_f24.f8[0], &dst_f24.f8[0], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_f24.f8[1], &dst_f24.f8[1], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_f24.f8[2], &dst_f24.f8[2], gridColRatio_f16, gridRatio);
    rpp_hip_layouttoggle24_pln3_to_pkd3((d_float24_s *)&dst_f24);
    *(d_float24_s *)dstPtr = *(d_float24_s *)&dst_f24;
}
__device__ void gridmask_result_pln3_pkd3_hip_compute(schar *srcPtr, uint srcStrideC, schar *dstPtr, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_schar24_s src_sc24, dst_sc24;
    src_sc24.sc8[0] = *(d_schar8_s *)srcPtr;
    srcPtr += srcStrideC;
    src_sc24.sc8[1] = *(d_schar8_s *)srcPtr;
    srcPtr += srcStrideC;
    src_sc24.sc8[2] = *(d_schar8_s *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_sc24.sc8[0], &dst_sc24.sc8[0], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_sc24.sc8[1], &dst_sc24.sc8[1], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_sc24.sc8[2], &dst_sc24.sc8[2], gridColRatio_f16, gridRatio);
    rpp_hip_layouttoggle24_pln3_to_pkd3((d_schar24sc1s_s *)&dst_sc24);
    *(d_schar24_s *)dstPtr = dst_sc24;
}
__device__ void gridmask_result_pln3_pkd3_hip_compute(half *srcPtr, uint srcStrideC, half *dstPtr, d_float16 *gridColRatio_f16, float gridRatio)
{
    d_half24 src_h24, dst_h24;
    src_h24.h8[0] = *(d_half8 *)srcPtr;
    srcPtr += srcStrideC;
    src_h24.h8[1] = *(d_half8 *)srcPtr;
    srcPtr += srcStrideC;
    src_h24.h8[2] = *(d_half8 *)srcPtr;
    gridmask_vector_masked_store8_hip_compute(&src_h24.h8[0], &dst_h24.h8[0], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_h24.h8[1], &dst_h24.h8[1], gridColRatio_f16, gridRatio);
    gridmask_vector_masked_store8_hip_compute(&src_h24.h8[2], &dst_h24.h8[2], gridColRatio_f16, gridRatio);
    rpp_hip_layouttoggle24_pln3_to_pkd3((d_half24_s *)&dst_h24);
    *(d_half24 *)dstPtr = dst_h24;
}

// Gridmask kernels

template <typename T>
__global__ void gridmask_pkd_hip_tensor(T *srcPtr,
                                        uint2 srcStridesNH,
                                        T *dstPtr,
                                        uint2 dstStridesNH,
                                        float2 rotateRatios,
                                        float2 translateRatios,
                                        float gridRatio,
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
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    float2 gridRowRatio_f2;
    d_float16 gridColRatio_f16, gridColRatioFloor_f16;
    gridmask_ratio_hip_compute(id_x, id_y, &rotateRatios, &translateRatios, &gridRowRatio_f2, &gridColRatio_f16);
    rpp_hip_math_floor16(&gridColRatio_f16, &gridColRatioFloor_f16);
    rpp_hip_math_subtract16(&gridColRatio_f16, &gridColRatioFloor_f16, &gridColRatio_f16);
    gridmask_result_pkd3_pkd3_hip_compute(srcPtr + srcIdx, dstPtr + dstIdx, &gridColRatio_f16, gridRatio);
}

template <typename T>
__global__ void gridmask_pln_hip_tensor(T *srcPtr,
                                        uint3 srcStridesNCH,
                                        T *dstPtr,
                                        uint3 dstStridesNCH,
                                        int channelsDst,
                                        float2 rotateRatios,
                                        float2 translateRatios,
                                        float gridRatio,
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

    float2 gridRowRatio_f2;
    d_float16 gridColRatio_f16, gridColRatioFloor_f16;
    gridmask_ratio_hip_compute(id_x, id_y, &rotateRatios, &translateRatios, &gridRowRatio_f2, &gridColRatio_f16);
    rpp_hip_math_floor16(&gridColRatio_f16, &gridColRatioFloor_f16);
    rpp_hip_math_subtract16(&gridColRatio_f16, &gridColRatioFloor_f16, &gridColRatio_f16);

    if (channelsDst == 3)
        gridmask_result_pln3_pln3_hip_compute(srcPtr + srcIdx, srcStridesNCH.y, dstPtr + dstIdx, dstStridesNCH.y, &gridColRatio_f16, gridRatio);
    else
        gridmask_result_pln1_pln1_hip_compute(srcPtr + srcIdx, dstPtr + dstIdx, &gridColRatio_f16, gridRatio);
}

template <typename T>
__global__ void gridmask_pkd3_pln3_hip_tensor(T *srcPtr,
                                              uint2 srcStridesNH,
                                              T *dstPtr,
                                              uint3 dstStridesNCH,
                                              float2 rotateRatios,
                                              float2 translateRatios,
                                              float gridRatio,
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

    float2 gridRowRatio_f2;
    d_float16 gridColRatio_f16, gridColRatioFloor_f16;
    gridmask_ratio_hip_compute(id_x, id_y, &rotateRatios, &translateRatios, &gridRowRatio_f2, &gridColRatio_f16);
    rpp_hip_math_floor16(&gridColRatio_f16, &gridColRatioFloor_f16);
    rpp_hip_math_subtract16(&gridColRatio_f16, &gridColRatioFloor_f16, &gridColRatio_f16);
    gridmask_result_pkd3_pln3_hip_compute(srcPtr + srcIdx, dstPtr + dstIdx, dstStridesNCH.y, &gridColRatio_f16, gridRatio);
}

template <typename T>
__global__ void gridmask_pln3_pkd3_hip_tensor(T *srcPtr,
                                              uint3 srcStridesNCH,
                                              T *dstPtr,
                                              uint2 dstStridesNH,
                                              float2 rotateRatios,
                                              float2 translateRatios,
                                              float gridRatio,
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
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    float2 gridRowRatio_f2;
    d_float16 gridColRatio_f16, gridColRatioFloor_f16;
    gridmask_ratio_hip_compute(id_x, id_y, &rotateRatios, &translateRatios, &gridRowRatio_f2, &gridColRatio_f16);
    rpp_hip_math_floor16(&gridColRatio_f16, &gridColRatioFloor_f16);
    rpp_hip_math_subtract16(&gridColRatio_f16, &gridColRatioFloor_f16, &gridColRatio_f16);
    gridmask_result_pln3_pkd3_hip_compute(srcPtr + srcIdx, srcStridesNCH.y, dstPtr + dstIdx, &gridColRatio_f16, gridRatio);
}

template <typename T>
RppStatus hip_exec_gridmask_tensor(T *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   T *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u tileWidth,
                                   Rpp32f gridRatio,
                                   Rpp32f gridAngle,
                                   RpptUintVector2D translateVector,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->w + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32f tileWidthInv = 1.0f / (Rpp32f)tileWidth;
    float2 rotateRatios = make_float2((cos(gridAngle) * tileWidthInv), (sin(gridAngle) * tileWidthInv));
    float2 translateRatios = make_float2((translateVector.x * tileWidthInv), (translateVector.y * tileWidthInv));

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(gridmask_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           rotateRatios,
                           translateRatios,
                           gridRatio,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(gridmask_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           rotateRatios,
                           translateRatios,
                           gridRatio,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(gridmask_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               rotateRatios,
                               translateRatios,
                               gridRatio,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(gridmask_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               rotateRatios,
                               translateRatios,
                               gridRatio,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_gridmask_tensor<Rpp8u>(Rpp8u*,
                                                   RpptDescPtr,
                                                   Rpp8u*,
                                                   RpptDescPtr,
                                                   Rpp32u,
                                                   Rpp32f,
                                                   Rpp32f,
                                                   RpptUintVector2D,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   rpp::Handle&);

template RppStatus hip_exec_gridmask_tensor<half>(half*,
                                                  RpptDescPtr,
                                                  half*,
                                                  RpptDescPtr,
                                                  Rpp32u,
                                                  Rpp32f,
                                                  Rpp32f,
                                                  RpptUintVector2D,
                                                  RpptROIPtr,
                                                  RpptRoiType,
                                                  rpp::Handle&);

template RppStatus hip_exec_gridmask_tensor<Rpp32f>(Rpp32f*,
                                                    RpptDescPtr,
                                                    Rpp32f*,
                                                    RpptDescPtr,
                                                    Rpp32u,
                                                    Rpp32f,
                                                    Rpp32f,
                                                    RpptUintVector2D,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    rpp::Handle&);

template RppStatus hip_exec_gridmask_tensor<Rpp8s>(Rpp8s*,
                                                   RpptDescPtr,
                                                   Rpp8s*,
                                                   RpptDescPtr,
                                                   Rpp32u,
                                                   Rpp32f,
                                                   Rpp32f,
                                                   RpptUintVector2D,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   rpp::Handle&);
