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

// -------------------- Set 0 - lens_correction device helpers --------------------

__device__ __forceinline__ void camera_coordinates_hip_compute(d_float8 *cameraCoords_f8, int id_y, d_float8 *locDst_f8x, float3 *inverseMatrix)
{
    float4 inverseCoord1_f4 = static_cast<float4>(id_y * inverseMatrix->y + inverseMatrix->z);
    float4 inverseCoord2_f4 = static_cast<float4>(inverseMatrix->x);
    cameraCoords_f8->f4[0] = inverseCoord1_f4 + locDst_f8x->f4[0] * inverseCoord2_f4;
    cameraCoords_f8->f4[1] = inverseCoord1_f4 + locDst_f8x->f4[1] * inverseCoord2_f4;
}

// -------------------- Set 1 - lens_correction kernels --------------------

// compute inverse of 3x3 camera matrix
__global__ void compute_inverse_matrix_hip_tensor(d_float9 *matTensor, d_float9 *invMatTensor)
{
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    d_float9 *mat_f9 = &matTensor[id_z];
    d_float9 *invMat_f9 = &invMatTensor[id_z];

    // initialize all values in invMat_f9 to zero
    invMat_f9->f3[0] = static_cast<float3>(0.0f);
    invMat_f9->f3[1] = invMat_f9->f3[0];
    invMat_f9->f3[2] = invMat_f9->f3[0];

    // compute determinant mat_f9
    float det =  (mat_f9->f1[0] * ((mat_f9->f1[4] * mat_f9->f1[8]) - (mat_f9->f1[7] * mat_f9->f1[5])))
               - (mat_f9->f1[1] * ((mat_f9->f1[3] * mat_f9->f1[8]) - (mat_f9->f1[5] * mat_f9->f1[6])))
               + (mat_f9->f1[2] * ((mat_f9->f1[3] * mat_f9->f1[7]) - (mat_f9->f1[4] * mat_f9->f1[6])));
    if(det != 0)
    {
        float invDet = 1 / det;
        invMat_f9->f1[0] = (mat_f9->f1[4] * mat_f9->f1[8] - mat_f9->f1[7] * mat_f9->f1[5]) * invDet;
        invMat_f9->f1[1] = (mat_f9->f1[2] * mat_f9->f1[7] - mat_f9->f1[1] * mat_f9->f1[8]) * invDet;
        invMat_f9->f1[2] = (mat_f9->f1[1] * mat_f9->f1[5] - mat_f9->f1[2] * mat_f9->f1[4]) * invDet;
        invMat_f9->f1[3] = (mat_f9->f1[5] * mat_f9->f1[6] - mat_f9->f1[3] * mat_f9->f1[8]) * invDet;
        invMat_f9->f1[4] = (mat_f9->f1[0] * mat_f9->f1[8] - mat_f9->f1[2] * mat_f9->f1[6]) * invDet;
        invMat_f9->f1[5] = (mat_f9->f1[3] * mat_f9->f1[2] - mat_f9->f1[0] * mat_f9->f1[5]) * invDet;
        invMat_f9->f1[6] = (mat_f9->f1[3] * mat_f9->f1[7] - mat_f9->f1[6] * mat_f9->f1[4]) * invDet;
        invMat_f9->f1[7] = (mat_f9->f1[6] * mat_f9->f1[1] - mat_f9->f1[0] * mat_f9->f1[7]) * invDet;
        invMat_f9->f1[8] = (mat_f9->f1[0] * mat_f9->f1[4] - mat_f9->f1[3] * mat_f9->f1[1]) * invDet;
    }
}

// compute remap tables from the camera matrix and distortion coefficients
__global__ void compute_remap_tables_hip_tensor(float *rowRemapTable,
                                                float *colRemapTable,
                                                d_float9 *cameraMatrixTensor,
                                                d_float9 *inverseMatrixTensor,
                                                d_float8 *distortionCoeffsTensor,
                                                uint2 remapTableStridesNH,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    d_float9 cameraMatrix_f9 = cameraMatrixTensor[id_z];
    d_float9 inverseMatrix_f9 = inverseMatrixTensor[id_z];
    d_float8 distortionCoeffs_f8 = distortionCoeffsTensor[id_z];

    // Get radial and tangential distortion coefficients
    float radialCoeff[6] = {distortionCoeffs_f8.f1[0], distortionCoeffs_f8.f1[1], distortionCoeffs_f8.f1[4], distortionCoeffs_f8.f1[5], distortionCoeffs_f8.f1[6], distortionCoeffs_f8.f1[7]};
    float tangentialCoeff[2] = {distortionCoeffs_f8.f1[2], distortionCoeffs_f8.f1[3]};

    uint dstIdx = id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y + id_x;
    d_float8 locDst_f8x;
    locDst_f8x.f4[0] = static_cast<float4>(id_x) + make_float4(0, 1, 2, 3);
    locDst_f8x.f4[1] = static_cast<float4>(id_x) + make_float4(4, 5, 6, 7);

    float4 one_f4 = static_cast<float4>(1.0f);
    float4 two_f4 = static_cast<float4>(2.0f);
    d_float8 z_f8, y_f8, x_f8;
    camera_coordinates_hip_compute(&z_f8, id_y, &locDst_f8x, &inverseMatrix_f9.f3[2]);          // float zCamera = id_y * inverseMatrix.f1[7] + inverseMatrix.f1[8] + id_x * inverseMatrix.f1[6]
    camera_coordinates_hip_compute(&y_f8, id_y, &locDst_f8x, &inverseMatrix_f9.f3[1]);          // float yCamera = id_y * inverseMatrix.f1[4] + inverseMatrix.f1[5] + id_x * inverseMatrix.f1[3]
    camera_coordinates_hip_compute(&x_f8, id_y, &locDst_f8x, &inverseMatrix_f9.f3[0]);          // float xCamera = id_y * inverseMatrix.f1[1] + inverseMatrix.f1[2] + id_x * inverseMatrix.f1[0]
    rpp_hip_math_divide8_const(&z_f8, &z_f8, one_f4);                                           // float z = 1./zCamera
    rpp_hip_math_multiply8(&y_f8, &z_f8, &y_f8);                                                // float y = yCamera * z;
    rpp_hip_math_multiply8(&x_f8, &z_f8, &x_f8);                                                // float x = xCamera * z;

    d_float8 ySquare_f8, xSquare_f8;
    rpp_hip_math_multiply8(&y_f8, &y_f8, &ySquare_f8);                                          // float ySquare = x * x
    rpp_hip_math_multiply8(&x_f8, &x_f8, &xSquare_f8);                                          // float xSquare = x * x

    d_float8 r2_f8, kr_f8, kr1_f8, kr2_f8;
    rpp_hip_math_add8(&xSquare_f8, &ySquare_f8, &r2_f8);                                        // float r2 = xSquare + ySquare

    d_float8 r2Cube_f8, r2Square_f8;
    rpp_hip_math_multiply8(&r2_f8, &r2_f8, &r2Square_f8);                                       // float r2Square = r2 * r2;
    rpp_hip_math_multiply8(&r2Square_f8, &r2_f8, &r2Cube_f8);                                   // float r2Cube = r2Square * r2;

    d_float24 radialCoeff_f24;
    radialCoeff_f24.f4[0] = static_cast<float4>(radialCoeff[0]);
    radialCoeff_f24.f4[1] = static_cast<float4>(radialCoeff[1]);
    radialCoeff_f24.f4[2] = static_cast<float4>(radialCoeff[2]);
    radialCoeff_f24.f4[3] = static_cast<float4>(radialCoeff[3]);
    radialCoeff_f24.f4[4] = static_cast<float4>(radialCoeff[4]);
    radialCoeff_f24.f4[5] = static_cast<float4>(radialCoeff[5]);

    // float kr = (1 + (radialCoeff[2] * r2Cube) + (radialCoeff[1] * r2Square) + (radialCoeff[0]) * r2)) / (1 + (radialCoeff[5] * r2Cube) + (radialCoeff[4] * r2Square) + (radialCoeff[3]) *r2))
    kr1_f8.f4[0] = (one_f4 + (radialCoeff_f24.f4[2] * r2Cube_f8.f4[0]) + (radialCoeff_f24.f4[1] *  r2Square_f8.f4[0]) + (radialCoeff_f24.f4[0] *  r2_f8.f4[0]));
    kr1_f8.f4[1] = (one_f4 + (radialCoeff_f24.f4[2] * r2Cube_f8.f4[1]) + (radialCoeff_f24.f4[1] *  r2Square_f8.f4[1]) + (radialCoeff_f24.f4[0] *  r2_f8.f4[1]));
    kr2_f8.f4[0] = (one_f4 + (radialCoeff_f24.f4[5] * r2Cube_f8.f4[0]) + (radialCoeff_f24.f4[4] *  r2Square_f8.f4[0]) + (radialCoeff_f24.f4[3] *  r2_f8.f4[0]));
    kr2_f8.f4[1] = (one_f4 + (radialCoeff_f24.f4[5] * r2Cube_f8.f4[1]) + (radialCoeff_f24.f4[4] *  r2Square_f8.f4[1]) + (radialCoeff_f24.f4[3] *  r2_f8.f4[1]));
    rpp_hip_math_divide8(&kr1_f8, &kr2_f8, &kr_f8);

    d_float8 xyMul2_f8;
    rpp_hip_math_multiply8(&x_f8, &y_f8, &xyMul2_f8);
    rpp_hip_math_multiply8_const(&xyMul2_f8, &xyMul2_f8, two_f4);                               // float xyMul2 = 2 * x * y

    d_float8 colLoc_f8, rowLoc_f8;
    rpp_hip_math_multiply8_const(&xSquare_f8, &xSquare_f8, two_f4);                             // xSquare = xSquare * 2;
    rpp_hip_math_multiply8_const(&ySquare_f8, &ySquare_f8, two_f4);                             // ySquare = ySquare * 2;

    d_float16 cameraMatrix_f16;
    cameraMatrix_f16.f4[0] = static_cast<float4>(cameraMatrix_f9.f1[0]);
    cameraMatrix_f16.f4[1] = static_cast<float4>(cameraMatrix_f9.f1[2]);
    cameraMatrix_f16.f4[2] = static_cast<float4>(cameraMatrix_f9.f1[4]);
    cameraMatrix_f16.f4[3] = static_cast<float4>(cameraMatrix_f9.f1[5]);

    d_float8 tangentialCoeff_f8;
    tangentialCoeff_f8.f4[0] = static_cast<float4>(tangentialCoeff[0]);
    tangentialCoeff_f8.f4[1] = static_cast<float4>(tangentialCoeff[1]);

    // float colLoc = cameraMatrix[0] * (x * kr + tangentialCoeff[0] * xyMul2 + tangentialCoeff[1] * (r2 + 2 * xSquare)) + cameraMatrix[2];
    colLoc_f8.f4[0] = cameraMatrix_f16.f4[0] * ((x_f8.f4[0] * kr_f8.f4[0]) + (tangentialCoeff_f8.f4[0] * xyMul2_f8.f4[0]) + (tangentialCoeff_f8.f4[1] * (r2_f8.f4[0] + xSquare_f8.f4[0]))) + cameraMatrix_f16.f4[1];
    colLoc_f8.f4[1] = cameraMatrix_f16.f4[0] * ((x_f8.f4[1] * kr_f8.f4[1]) + (tangentialCoeff_f8.f4[0] * xyMul2_f8.f4[1]) + (tangentialCoeff_f8.f4[1] * (r2_f8.f4[1] + xSquare_f8.f4[1]))) + cameraMatrix_f16.f4[1];

    // float rowLoc = cameraMatrix[4] * (y * kr + tangentialCoeff[1] * xyMul2 + tangentialCoeff[0] * (r2 + 2 * ySquare)) + cameraMatrix[4];
    rowLoc_f8.f4[0] = cameraMatrix_f16.f4[2] * ((y_f8.f4[0] * kr_f8.f4[0]) + (tangentialCoeff_f8.f4[1] * xyMul2_f8.f4[0]) + (tangentialCoeff_f8.f4[0] * (r2_f8.f4[0] + ySquare_f8.f4[0]))) + cameraMatrix_f16.f4[3];
    rowLoc_f8.f4[1] = cameraMatrix_f16.f4[2] * ((y_f8.f4[1] * kr_f8.f4[1]) + (tangentialCoeff_f8.f4[1] * xyMul2_f8.f4[1]) + (tangentialCoeff_f8.f4[0] * (r2_f8.f4[1] + ySquare_f8.f4[1]))) + cameraMatrix_f16.f4[3];

    rpp_hip_pack_float8_and_store8(colRemapTable + dstIdx, &colLoc_f8);
    rpp_hip_pack_float8_and_store8(rowRemapTable + dstIdx, &rowLoc_f8);
}

// -------------------- Set 2 - Kernel Executors --------------------

RppStatus hip_exec_lens_correction_tensor(RpptDescPtr dstDescPtr,
                                          Rpp32f *rowRemapTable,
                                          Rpp32f *colRemapTable,
                                          RpptDescPtr remapTableDescPtr,
                                          Rpp32f *cameraMatrix,
                                          Rpp32f *distanceCoeffs,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->w + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = dstDescPtr->n;

    float *inverseMatrix = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    hipLaunchKernelGGL(compute_inverse_matrix_hip_tensor,
                       dim3(1, 1, ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(1, 1, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       reinterpret_cast<d_float9 *>(cameraMatrix),
                       reinterpret_cast<d_float9 *>(inverseMatrix));
    hipLaunchKernelGGL(compute_remap_tables_hip_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       rowRemapTable,
                       colRemapTable,
                       reinterpret_cast<d_float9 *>(cameraMatrix),
                       reinterpret_cast<d_float9 *>(inverseMatrix),
                       reinterpret_cast<d_float8 *>(distanceCoeffs),
                       make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                       roiTensorPtrSrc);

    return RPP_SUCCESS;
}
