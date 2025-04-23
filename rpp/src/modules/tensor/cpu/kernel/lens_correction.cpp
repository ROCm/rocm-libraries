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

#include "host_tensor_executors.hpp"

// Compute Inverse matrix (3x3)
inline void get_inverse(float *mat, float *invMat)
{
    float det = mat[0] * (mat[4] * mat[8] - mat[7] * mat[5]) - mat[1] * (mat[3] * mat[8] - mat[5] * mat[6]) + mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);
    if(det != 0)
    {
        float invDet = 1 / det;
        invMat[0] = (mat[4] * mat[8] - mat[7] * mat[5]) * invDet;
        invMat[1] = (mat[2] * mat[7] - mat[1] * mat[8]) * invDet;
        invMat[2] = (mat[1] * mat[5] - mat[2] * mat[4]) * invDet;
        invMat[3] = (mat[5] * mat[6] - mat[3] * mat[8]) * invDet;
        invMat[4] = (mat[0] * mat[8] - mat[2] * mat[6]) * invDet;
        invMat[5] = (mat[3] * mat[2] - mat[0] * mat[5]) * invDet;
        invMat[6] = (mat[3] * mat[7] - mat[6] * mat[4]) * invDet;
        invMat[7] = (mat[6] * mat[1] - mat[0] * mat[7]) * invDet;
        invMat[8] = (mat[0] * mat[4] - mat[3] * mat[1]) * invDet;
    }
}

void compute_lens_correction_remap_tables_host_tensor(RpptDescPtr srcDescPtr,
                                                      Rpp32f *rowRemapTable,
                                                      Rpp32f *colRemapTable,
                                                      RpptDescPtr tableDescPtr,
                                                      Rpp32f *cameraMatrixTensor,
                                                      Rpp32f *distortionCoeffsTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
        rowRemapTableTemp = rowRemapTable + batchCount * tableDescPtr->strides.nStride;
        colRemapTableTemp = colRemapTable + batchCount * tableDescPtr->strides.nStride;

        // cameraMatrix is a 3x3 matrix thus increment by 9 to iterate from one tensor in a batch to another
        Rpp32f *cameraMatrix = cameraMatrixTensor + batchCount * 9;
        Rpp32f *distortionCoeffs = distortionCoeffsTensor + batchCount * 8;
        Rpp32s height = roiTensorPtrSrc[batchCount].xywhROI.roiHeight;
        Rpp32s width = roiTensorPtrSrc[batchCount].xywhROI.roiWidth;
        Rpp32u alignedLength = width & ~7;
        Rpp32s vectorIncrement = 8;

        Rpp32f invCameraMatrix[9];
        std::fill(invCameraMatrix, invCameraMatrix + 9, 0.0f);  // initialize all values in invCameraMatrix to zero
        get_inverse(cameraMatrix, invCameraMatrix);
        Rpp32f *invMat = &invCameraMatrix[0];

        // Get radial and tangential distortion coefficients
        Rpp32f rCoeff[6] = { distortionCoeffs[0], distortionCoeffs[1], distortionCoeffs[4], distortionCoeffs[5], distortionCoeffs[6], distortionCoeffs[7] };
        Rpp32f tCoeff[2] = { distortionCoeffs[2], distortionCoeffs[3] };

        __m256 pRCoeff[6], pTCoeff[2];
        pRCoeff[0] = _mm256_set1_ps(rCoeff[0]);
        pRCoeff[1] = _mm256_set1_ps(rCoeff[1]);
        pRCoeff[2] = _mm256_set1_ps(rCoeff[2]);
        pRCoeff[3] = _mm256_set1_ps(rCoeff[3]);
        pRCoeff[4] = _mm256_set1_ps(rCoeff[4]);
        pRCoeff[5] = _mm256_set1_ps(rCoeff[5]);
        pTCoeff[0] = _mm256_set1_ps(tCoeff[0]);
        pTCoeff[1] = _mm256_set1_ps(tCoeff[1]);

        Rpp32f u0 = cameraMatrix[2],  v0 = cameraMatrix[5];
        Rpp32f fx = cameraMatrix[0],  fy = cameraMatrix[4];
        __m256 pFx, pFy, pU0, pV0;
        pFx = _mm256_set1_ps(fx);
        pFy = _mm256_set1_ps(fy);
        pU0 = _mm256_set1_ps(u0);
        pV0 = _mm256_set1_ps(v0);

        __m256 pInvMat0, pInvMat3, pInvMat6;
        pInvMat0 = _mm256_set1_ps(invMat[0]);
        pInvMat3 = _mm256_set1_ps(invMat[3]);
        pInvMat6 = _mm256_set1_ps(invMat[6]);

        __m256 pXCameraInit, pYCameraInit, pZCameraInit;
        __m256 pXCameraIncrement, pYCameraIncrement, pZCameraIncrement;
        pXCameraInit = _mm256_mul_ps(avx_pDstLocInit, pInvMat0);
        pYCameraInit = _mm256_mul_ps(avx_pDstLocInit, pInvMat3);
        pZCameraInit = _mm256_mul_ps(avx_pDstLocInit, pInvMat6);
        pXCameraIncrement = _mm256_mul_ps(pInvMat0, avx_p8);
        pYCameraIncrement = _mm256_mul_ps(pInvMat3, avx_p8);
        pZCameraIncrement = _mm256_mul_ps(pInvMat6, avx_p8);
        for(int i = 0; i < height; i++)
        {
            Rpp32f *rowRemapTableRow = rowRemapTableTemp + i * tableDescPtr->strides.hStride;
            Rpp32f *colRemapTableRow = colRemapTableTemp + i * tableDescPtr->strides.hStride;
            Rpp32f xCamera = i * invMat[1] + invMat[2];
            Rpp32f yCamera = i * invMat[4] + invMat[5];
            Rpp32f zCamera = i * invMat[7] + invMat[8];
            __m256 pXCamera = _mm256_add_ps(_mm256_set1_ps(xCamera), pXCameraInit);
            __m256 pYCamera = _mm256_add_ps(_mm256_set1_ps(yCamera), pYCameraInit);
            __m256 pZCamera = _mm256_add_ps(_mm256_set1_ps(zCamera), pZCameraInit);
            int vectorLoopCount = 0;
            for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
            {
                // float z = 1./zCamera, x = xCamera*z, y = yCamera*z;
                __m256 pZ = _mm256_div_ps(avx_p1, pZCamera);
                __m256 pX = _mm256_mul_ps(pXCamera, pZ);
                __m256 pY = _mm256_mul_ps(pYCamera, pZ);

                // float xSquare = x*x, ySquare = y*y, r2 = xSquare + ySquare;
                __m256 pXSquare = _mm256_mul_ps(pX, pX);
                __m256 pYSquare = _mm256_mul_ps(pY, pY);
                __m256 pR2 = _mm256_add_ps(pXSquare, pYSquare);

                // float xyMul2 = 2*x*y;
                __m256 p2xy = _mm256_mul_ps(avx_p2, _mm256_mul_ps(pX, pY));

                // float kr = std::fmaf(std::fmaf(std::fmaf(rCoeff[2], r2, rCoeff[1]), r2, rCoeff[0]), r2, 1) / std::fmaf(std::fmaf(std::fmaf(rCoeff[5], r2, rCoeff[4]), r2, rCoeff[3]), r2, 1);
                __m256 pNum = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(pRCoeff[2], pR2, pRCoeff[1]), pR2, pRCoeff[0]), pR2, avx_p1);
                __m256 pDen = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(pRCoeff[5], pR2, pRCoeff[4]), pR2, pRCoeff[3]), pR2, avx_p1);
                __m256 pKR = _mm256_div_ps(pNum, pDen);

                // float colLoc = std::fmaf(fx, (std::fmaf(tCoeff[1], (std::fmaf(2, xSquare, r2)), std::fmaf(x, kr, (tCoeff[0] * xyMul2)))), u0);
                __m256 pColLoc = _mm256_fmadd_ps(pFx, _mm256_fmadd_ps(pTCoeff[1], _mm256_fmadd_ps(avx_p2, pXSquare, pR2), _mm256_fmadd_ps(pX, pKR,  _mm256_mul_ps(pTCoeff[0], p2xy))), pU0);

                // float rowLoc = std::fmaf(fy, (std::fmaf(tCoeff[0], (std::fmaf(2, ySquare, r2)), std::fmaf(y, kr, (tCoeff[1] * xyMul2)))), v0);
                __m256 pRowLoc = _mm256_fmadd_ps(pFy, _mm256_fmadd_ps(pTCoeff[0], _mm256_fmadd_ps(avx_p2, pYSquare, pR2), _mm256_fmadd_ps(pY, pKR,  _mm256_mul_ps(pTCoeff[1], p2xy))), pV0);

                _mm256_storeu_ps(rowRemapTableRow, pRowLoc);
                _mm256_storeu_ps(colRemapTableRow, pColLoc);
                rowRemapTableRow += vectorIncrement;
                colRemapTableRow += vectorIncrement;

                // xCamera += invMat[0], yCamera += invMat[3], zCamera += invMat[6]
                pXCamera = _mm256_add_ps(pXCamera, pXCameraIncrement);
                pYCamera = _mm256_add_ps(pYCamera, pYCameraIncrement);
                pZCamera = _mm256_add_ps(pZCamera, pZCameraIncrement);
            }
            for(; vectorLoopCount < width; vectorLoopCount++)
            {
                Rpp32f z = 1./zCamera, x = xCamera * z, y = yCamera * z;
                Rpp32f xSquare = x * x, ySquare = y * y, r2 = xSquare + ySquare;
                Rpp32f xyMul2 = 2 * x * y;
                Rpp32f kr = std::fmaf(std::fmaf(std::fmaf(rCoeff[2], r2, rCoeff[1]), r2, rCoeff[0]), r2, 1) / std::fmaf(std::fmaf(std::fmaf(rCoeff[5], r2, rCoeff[4]), r2, rCoeff[3]), r2, 1);
                Rpp32f colLoc = std::fmaf(fx, (std::fmaf(tCoeff[1], (std::fmaf(2, xSquare, r2)), std::fmaf(x, kr, (tCoeff[0] * xyMul2)))), u0);
                Rpp32f rowLoc = std::fmaf(fy, (std::fmaf(tCoeff[0], (std::fmaf(2, ySquare, r2)), std::fmaf(y, kr, (tCoeff[1] * xyMul2)))), v0);
                *rowRemapTableRow++ = rowLoc;
                *colRemapTableRow++ = colLoc;
                xCamera += invMat[0];
                yCamera += invMat[3];
                zCamera += invMat[6];
            }
        }
    }
}
