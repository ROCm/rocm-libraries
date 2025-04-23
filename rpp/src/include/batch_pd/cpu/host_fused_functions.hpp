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

#ifndef HOST_FUSED_FUNCTIONS_H
#define HOST_FUSED_FUNCTIONS_H

#include "rpp_cpu_batchpd.hpp"

/**************** color_twist ***************/

template <typename T>
RppStatus color_twist_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                         Rpp32f *batch_alpha, Rpp32f *batch_beta,
                         Rpp32f *batch_hueShift, Rpp32f *batch_saturationFactor,
                         RppiROI *roiPoints, Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp64u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f hueShift = batch_hueShift[batchCount];
            Rpp32f saturationFactor = batch_saturationFactor[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            color_twist_host(srcPtrImage, batch_srcSizeMax[batchCount], dstPtrImage, alpha, beta, hueShift, saturationFactor, chnFormat, channel);

            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp64u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f hueShift = batch_hueShift[batchCount];
            Rpp32f saturationFactor = batch_saturationFactor[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            color_twist_host(srcPtrImage, batch_srcSizeMax[batchCount], dstPtrImage, alpha, beta, hueShift, saturationFactor, chnFormat, channel);

            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus color_twist_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f alpha, Rpp32f beta,
                    Rpp32f hueShift, Rpp32f saturationFactor,
                    RppiChnFormat chnFormat, Rpp32u channel)
{

    hueShift = (int)hueShift % 360;
    Rpp32f hueShiftAngle = hueShift;
    hueShift *= 0.002778f;

    if (hueShift < 0)
    {
        hueShift += 1;
        hueShiftAngle += 360;
    }

    //Rpp64u totalImageDim = channel * srcSize.height * srcSize.width;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
        T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

        Rpp64u imageDim = srcSize.height * srcSize.width;
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + (imageDim);
        srcPtrTempB = srcPtr + (2 * imageDim);
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + (imageDim);
        dstPtrTempB = dstPtr + (2 * imageDim);

        Rpp64u bufferLength = srcSize.height * srcSize.width;
        Rpp64u alignedLength = bufferLength & ~3;

        __m128i const zero = _mm_setzero_si128();
        __m128 pZeros = _mm_set1_ps(0.0);
        __m128 pOnes = _mm_set1_ps(1.0);
        __m128 pFactor = _mm_set1_ps(255.0);
        __m128 pHueShift = _mm_set1_ps(hueShift);
        __m128 pSaturationFactor = _mm_set1_ps(saturationFactor);
        __m128i px0, px1, px2;
        __m128 xR, xG, xB;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;
        __m128 h0, h1, h2, h3;
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        h0 = _mm_set1_ps(1.0);

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp8u arrayR[4];
        Rpp8u arrayG[4];
        Rpp8u arrayB[4];

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
        {
            px0 =  _mm_loadu_si128((__m128i *)srcPtrTempR);
            px1 =  _mm_loadu_si128((__m128i *)srcPtrTempG);
            px2 =  _mm_loadu_si128((__m128i *)srcPtrTempB);

            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
            px1 = _mm_unpacklo_epi8(px1, zero);    // pixels 0-7
            px2 = _mm_unpacklo_epi8(px2, zero);    // pixels 0-7

            xR = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
            xG = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 0-3
            xB = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px2, zero));    // pixels 0-3

            xR = _mm_div_ps(xR, pFactor);
            xG = _mm_div_ps(xG, pFactor);
            xB = _mm_div_ps(xB, pFactor);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]

            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, pOnes);                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction
            xH = _mm_cmple_ps(pOnes, xG);

            xH = _mm_and_ps(xH, pOnes);
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Modify Hue Values and re-normalize H to fraction
            xG = _mm_add_ps(xG, pHueShift);

            xH = _mm_cmple_ps(pOnes, xG);
            xH = _mm_and_ps(xH, pOnes);

            xG = _mm_sub_ps(xG, xH);

            // Modify Saturation Values
            xS = _mm_mul_ps(xS, pSaturationFactor);
            xS = _mm_min_ps(xS, pOnes);
            xS = _mm_max_ps(xS, pZeros);

            _MM_TRANSPOSE4_PS (h0, xG, xS, xV);

            __m128 h1, h2, h3;

            h1 = xG;
            h2 = xS;
            h3 = xV;

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            _MM_TRANSPOSE4_PS (x0, x1, x2, x3);

            x1 = _mm_mul_ps(x1, pFactor);
            x2 = _mm_mul_ps(x2, pFactor);
            x3 = _mm_mul_ps(x3, pFactor);

            x1 = _mm_mul_ps(x1, pMul);
            x2 = _mm_mul_ps(x2, pMul);
            x3 = _mm_mul_ps(x3, pMul);

            x1 = _mm_add_ps(x1, pAdd);
            x2 = _mm_add_ps(x2, pAdd);
            x3 = _mm_add_ps(x3, pAdd);

            px0 = _mm_cvtps_epi32(x1);
            px1 = _mm_cvtps_epi32(x2);
            px2 = _mm_cvtps_epi32(x3);

            px0 = _mm_packs_epi32(px0, px0);
            px0 = _mm_packus_epi16(px0, px0);
            *((int*)arrayR) = _mm_cvtsi128_si32(px0);

            px1 = _mm_packs_epi32(px1, px1);
            px1 = _mm_packus_epi16(px1, px1);
            *((int*)arrayG) = _mm_cvtsi128_si32(px1);

            px2 = _mm_packs_epi32(px2, px2);
            px2 = _mm_packus_epi16(px2, px2);
            *((int*)arrayB) = _mm_cvtsi128_si32(px2);

            memcpy(dstPtrTempR, arrayR, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempG, arrayG, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempB, arrayB, 4 * sizeof(Rpp8u));

            srcPtrTempR += 4;
            srcPtrTempG += 4;
            srcPtrTempB += 4;
            dstPtrTempR += 4;
            dstPtrTempG += 4;
            dstPtrTempB += 4;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            // RGB to HSV

            Rpp32f hue, sat, val;
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                hue = 0;
            }
            else if (cmax == rf)
            {
                hue = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                hue = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                hue = round(60 * (((rf - gf) / delta) + 4));
            }

            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            if (cmax == 0)
            {
                sat = 0;
            }
            else
            {
                sat = delta / cmax;
            }

            val = cmax;

            // Modify Hue and Saturation

            hue = hue + hueShiftAngle;
            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            sat *= saturationFactor;
            sat = (sat < (Rpp32f) 1) ? sat : ((Rpp32f) 1);
            sat = (sat > (Rpp32f) 0) ? sat : ((Rpp32f) 0);

            // HSV to RGB

            Rpp32f c, x, m;
            c = val * sat;
            x = c * (1 - abs(int(fmod((hue / 60), 2)) - 1));
            m = val - c;

            if ((0 <= hue) && (hue < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= hue) && (hue < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= hue) && (hue < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= hue) && (hue < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= hue) && (hue < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= hue) && (hue < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempR++;
            srcPtrTempG++;
            srcPtrTempB++;
            dstPtrTempR++;
            dstPtrTempG++;
            dstPtrTempB++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        T *srcPtrTemp, *dstPtrTemp;
        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;

        Rpp32u bufferLength = srcSize.height * srcSize.width;
        Rpp32u alignedLength = bufferLength & ~3;
        __m128i mask_R = _mm_setr_epi8(0, 0x80, 0x80, 0x80, 3, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80, 9, 0x80, 0x80, 0x80);
        __m128i mask_G = _mm_setr_epi8(1, 0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80, 7, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80);
        __m128i mask_B = _mm_setr_epi8(2, 0x80, 0x80, 0x80, 5, 0x80, 0x80, 0x80, 8, 0x80, 0x80, 0x80, 11, 0x80, 0x80, 0x80);
        __m128i mask2 = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15);
        __m128 pZeros = _mm_set1_ps(0.0);
        __m128 pOnes = _mm_set1_ps(1.0);
        __m128 pMultiplier1 = _mm_set1_ps(1 / 255.0);
        __m128 pHueShift = _mm_set1_ps(hueShift);
        __m128 pSaturationFactor = _mm_set1_ps(saturationFactor);
        __m128 pMul = _mm_set1_ps(alpha*255);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp32u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
        {
          // todo:: tomany registers used: try to reuse and reduce
          __m128i px0, px1, px2, px3;
          __m128 xR, xG, xB, xA;
          __m128 xH, xS, xV, xC;
          __m128 xX, xY, xZ;
          __m128 h0, h1, h2, h3;
          __m128 x0, x1, x2, x3;
          __m128 a0, a1;
            // Load -> Shuffle -> Unpack
            px0 = _mm_loadu_si128((__m128i *)srcPtrTemp);           // load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04
            xR = _mm_cvtepi32_ps(_mm_shuffle_epi8(px0, mask_R));    // XR - Contains R01-04
            xG = _mm_cvtepi32_ps(_mm_shuffle_epi8(px0, mask_G));    // XG - Contains G01-04
            xB = _mm_cvtepi32_ps(_mm_shuffle_epi8(px0, mask_B));    // XB - Contains B01-04

            // Normalize 0-255 -> 0-1
            xR = _mm_mul_ps(xR, pMultiplier1);
            xG = _mm_mul_ps(xG, pMultiplier1);
            xB = _mm_mul_ps(xB, pMultiplier1);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m], delta
            xC = xS;                                               // Xc <- delta
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ], delta/max

            //xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]

            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            //xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xC = _mm_mul_ps(xC, xmm_p6);                            // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, pOnes);                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);
            xG = _mm_add_ps(xG, pHueShift);

            // Normalize H to fraction
            xH = _mm_cmple_ps(pOnes, xG);

            xH = _mm_and_ps(xH, pOnes);
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Modify Saturation Values
            xS = _mm_mul_ps(xS, pSaturationFactor);
            xS = _mm_min_ps(xS, pOnes);
            xS = _mm_max_ps(xS, pZeros);

            x0 = SIMD_SHUFFLE_PS(xG, _MM_SHUFFLE(0, 0, 0, 0));    // x0 <- [H           |H           |H           |H          ]
            x1 = SIMD_SHUFFLE_PS(xG, _MM_SHUFFLE(1, 1, 1, 1));    // x1 <- [H           |H           |H           |H          ]
            x2 = SIMD_SHUFFLE_PS(xG, _MM_SHUFFLE(2, 2, 2, 2));    // x2 <- [H           |H           |H           |H          ]
            x3 = SIMD_SHUFFLE_PS(xG, _MM_SHUFFLE(3, 3, 3, 3));    // x3 <- [H           |H           |H           |H          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(xS, _MM_SHUFFLE(0, 0, 0, 0));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(xS, _MM_SHUFFLE(1, 1, 1, 1));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(xV, _MM_SHUFFLE(0, 0, 0, 0));     // h0 <- [V           |V           |V           |V          ]
            h1 = SIMD_SHUFFLE_PS(xV, _MM_SHUFFLE(1, 1, 1, 1));     // h1 <- [V           |V           |V           |V          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(xS, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(xS, _MM_SHUFFLE(3, 3, 3, 3));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(xV, _MM_SHUFFLE(2, 2, 2, 2));     // h2 <- [V           |V           |V           |V          ]
            h3 = SIMD_SHUFFLE_PS(xV, _MM_SHUFFLE(3, 3, 3, 3));     // h3 <- [V           |V           |V           |V          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |V          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |V          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |V          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |V          ]

            // Store into ps
            x0 = _mm_shuffle_ps(x0,x0, _MM_SHUFFLE(0,3,2,1));
            x1 = _mm_shuffle_ps(x1,x1, _MM_SHUFFLE(0,3,2,1));
            x2 = _mm_shuffle_ps(x2,x2, _MM_SHUFFLE(0,3,2,1));
            x3 = _mm_shuffle_ps(x3,x3, _MM_SHUFFLE(0,3,2,1));

            // Un-normalize + Brightness Change
            x0 = _mm_mul_ps(x0, pMul);
            x1 = _mm_mul_ps(x1, pMul);
            x2 = _mm_mul_ps(x2, pMul);
            x3 = _mm_mul_ps(x3, pMul);
            x0 = _mm_add_ps(x0, pAdd);
            x1 = _mm_add_ps(x1, pAdd);
            x2 = _mm_add_ps(x2, pAdd);
            x3 = _mm_add_ps(x3, pAdd);

            // Pack -> Shuffle -> Store 0-1 -> 0-255
            px0 = _mm_cvtps_epi32(x0);
            px1 = _mm_cvtps_epi32(x1);
            px2 = _mm_cvtps_epi32(x2);
            px3 = _mm_cvtps_epi32(x3);
            px0 = _mm_packus_epi32(px0, px1);    // pack pixels 0-7
            px1 = _mm_packus_epi32(px2, px3);    // pack pixels 8-15
            px0 = _mm_packus_epi16(px0, px1);    // pack pixels 0-15 as [R01|G01|B01|A01|R02|G02|B02|A02|R03|G03|B03|A03|R04|G04|B04|A04]
            px0 = _mm_shuffle_epi8(px0, mask2);    // shuffle to get [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|A01|A02|A03|A04]
            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);    // store [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|A01|A02|A03|A04]

            srcPtrTemp += 12;
            dstPtrTemp += 12;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            // RGB to HSV

            Rpp32f hue, sat, val;
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) srcPtrTemp[0]) / 255;
            gf = ((Rpp32f) srcPtrTemp[1]) / 255;
            bf = ((Rpp32f) srcPtrTemp[2]) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;
            hue = 0.0f;
            sat = 0.0f;
            float add = 0.0;
            if (delta) {
                if (cmax) sat = delta / cmax;
                if (cmax == rf)
                {
                    hue = gf - bf;
                    add = 0.0f;
                }
                else if (cmax == gf)
                {
                    hue = bf - rf;
                    add = 1.0f/3.0f;
                }
                else
                {
                    hue = rf - gf;
                    add = 2.0f/3.0f;
                }
                hue /= (6.0*delta);
                //hue += add;
            }


            // Modify Hue and Saturation

            hue += hueShift + add;
            //if (hue >= 1.f) hue -= 1.0f;
            hue = hue - (int)hue;
            if (hue < 0) hue += 1.0;

            sat *= saturationFactor;
            sat = (sat < (Rpp32f) 1) ? sat : ((Rpp32f) 1);
            sat = (sat > (Rpp32f) 0) ? sat : ((Rpp32f) 0);

            // HSV to RGB
            val = cmax*255;
            hue *= 6.0f;
            int index = static_cast<int>(hue);

            float f = (hue - static_cast<float>(index));
            float p = val * (1.0f - sat);
            float q = val * (1.0f - sat * f);
            float t = val * (1.0f - sat * (1.0f - f));

            switch (index) {
              case 0: dstPtrTemp[0] = val; dstPtrTemp[1] = t; dstPtrTemp[2] = p; break;
              case 1: dstPtrTemp[0] = q; dstPtrTemp[1] = val; dstPtrTemp[2] = p; break;
              case 2: dstPtrTemp[0] = p; dstPtrTemp[1] = val; dstPtrTemp[2] = t; break;
              case 3: dstPtrTemp[0] = p; dstPtrTemp[1] = q; dstPtrTemp[2] = val; break;
              case 4: dstPtrTemp[0] = t; dstPtrTemp[1] = p; dstPtrTemp[2] = val; break;
              case 5: dstPtrTemp[0] = val; dstPtrTemp[1] = p; dstPtrTemp[2] = q; break;
            }

            srcPtrTemp += 3;
            dstPtrTemp += 3;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus color_twist_f32_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                         Rpp32f *batch_alpha, Rpp32f *batch_beta,
                         Rpp32f *batch_hueShift, Rpp32f *batch_saturationFactor,
                         RppiROI *roiPoints, Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp64u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f hueShift = batch_hueShift[batchCount];
            Rpp32f saturationFactor = batch_saturationFactor[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            color_twist_f32_host(srcPtrImage, batch_srcSizeMax[batchCount], dstPtrImage, alpha, beta, hueShift, saturationFactor, chnFormat, channel);

            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp64u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f hueShift = batch_hueShift[batchCount];
            Rpp32f saturationFactor = batch_saturationFactor[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            color_twist_f32_host(srcPtrImage, batch_srcSizeMax[batchCount], dstPtrImage, alpha, beta, hueShift, saturationFactor, chnFormat, channel);

            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_twist_f32_host(Rpp32f* srcPtr, RppiSize srcSize, Rpp32f* dstPtr,
                    Rpp32f alpha, Rpp32f beta,
                    Rpp32f hueShift, Rpp32f saturationFactor,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    hueShift = (int)hueShift % 360;
    Rpp32f hueShiftAngle = hueShift;
    hueShift *= 0.002778f;

    if (hueShift < 0)
    {
        hueShift += 1;
        hueShiftAngle += 360;
    }

    Rpp64u totalImageDim = channel * srcSize.height * srcSize.width;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
        Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

        Rpp64u imageDim = srcSize.height * srcSize.width;
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + (imageDim);
        srcPtrTempB = srcPtr + (2 * imageDim);
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + (imageDim);
        dstPtrTempB = dstPtr + (2 * imageDim);

        Rpp64u bufferLength = srcSize.height * srcSize.width;
        Rpp64u alignedLength = bufferLength & ~3;

        __m128 pZeros = _mm_set1_ps(0.0);
        __m128 pOnes = _mm_set1_ps(1.0);
        __m128 pFactor = _mm_set1_ps(255.0);
        __m128 pHueShift = _mm_set1_ps(hueShift);
        __m128 pSaturationFactor = _mm_set1_ps(saturationFactor);
        __m128 xR, xG, xB;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;
        __m128 h0, h1, h2, h3;
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        h0 = _mm_set1_ps(1.0);

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
        {
            xR = _mm_loadu_ps(srcPtrTempR);
            xG = _mm_loadu_ps(srcPtrTempG);
            xB = _mm_loadu_ps(srcPtrTempB);

            xR = _mm_div_ps(xR, pFactor);
            xG = _mm_div_ps(xG, pFactor);
            xB = _mm_div_ps(xB, pFactor);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]

            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Modify Hue Values and re-normalize H to fraction
            xG = _mm_add_ps(xG, pHueShift);

            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);
            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));

            xG = _mm_sub_ps(xG, xH);

            // Modify Saturation Values
            xS = _mm_mul_ps(xS, pSaturationFactor);
            xS = _mm_min_ps(xS, pOnes);
            xS = _mm_max_ps(xS, pZeros);

            _MM_TRANSPOSE4_PS (h0, xG, xS, xV);

            __m128 h1, h2, h3;

            h1 = xG;
            h2 = xS;
            h3 = xV;

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            _MM_TRANSPOSE4_PS (x0, x1, x2, x3);

            x1 = _mm_mul_ps(x1, pFactor);
            x2 = _mm_mul_ps(x2, pFactor);
            x3 = _mm_mul_ps(x3, pFactor);

            x1 = _mm_mul_ps(x1, pMul);
            x2 = _mm_mul_ps(x2, pMul);
            x3 = _mm_mul_ps(x3, pMul);

            x1 = _mm_add_ps(x1, pAdd);
            x2 = _mm_add_ps(x2, pAdd);
            x3 = _mm_add_ps(x3, pAdd);

            _mm_storeu_ps(dstPtrTempR, x1);
            _mm_storeu_ps(dstPtrTempG, x2);
            _mm_storeu_ps(dstPtrTempB, x3);

            srcPtrTempR += 4;
            srcPtrTempG += 4;
            srcPtrTempB += 4;
            dstPtrTempR += 4;
            dstPtrTempG += 4;
            dstPtrTempB += 4;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            // RGB to HSV

            Rpp32f hue, sat, val;
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = *srcPtrTempR / 255;
            gf = *srcPtrTempG / 255;
            bf = *srcPtrTempB / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                hue = 0;
            }
            else if (cmax == rf)
            {
                hue = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                hue = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                hue = round(60 * (((rf - gf) / delta) + 4));
            }

            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            if (cmax == 0)
            {
                sat = 0;
            }
            else
            {
                sat = delta / cmax;
            }

            val = cmax;

            // Modify Hue and Saturation

            hue = hue + hueShiftAngle;
            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            sat *= saturationFactor;
            sat = (sat < (Rpp32f) 1) ? sat : ((Rpp32f) 1);
            sat = (sat > (Rpp32f) 0) ? sat : ((Rpp32f) 0);

            // HSV to RGB

            Rpp32f c, x, m;
            c = val * sat;
            x = c * (1 - abs(int(fmod((hue / 60), 2)) - 1));
            m = val - c;

            if ((0 <= hue) && (hue < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= hue) && (hue < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= hue) && (hue < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= hue) && (hue < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= hue) && (hue < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= hue) && (hue < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = round((rf + m) * 255);
            *dstPtrTempG = round((gf + m) * 255);
            *dstPtrTempB = round((bf + m) * 255);

            srcPtrTempR++;
            srcPtrTempG++;
            srcPtrTempB++;
            dstPtrTempR++;
            dstPtrTempG++;
            dstPtrTempB++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32f *srcPtrTempPx0, *srcPtrTempPx1, *srcPtrTempPx2, *srcPtrTempPx3;
        Rpp32f *dstPtrTempPx0, *dstPtrTempPx1, *dstPtrTempPx2, *dstPtrTempPx3;

        srcPtrTempPx0 = srcPtr;
        srcPtrTempPx1 = srcPtr + 3;
        srcPtrTempPx2 = srcPtr + 6;
        srcPtrTempPx3 = srcPtr + 9;
        dstPtrTempPx0 = dstPtr;
        dstPtrTempPx1 = dstPtr + 3;
        dstPtrTempPx2 = dstPtr + 6;
        dstPtrTempPx3 = dstPtr + 9;

        Rpp64u bufferLength = totalImageDim;
        Rpp64u alignedLength = (bufferLength / 12) * 12;

        __m128 pZeros = _mm_set1_ps(0.0);
        __m128 pOnes = _mm_set1_ps(1.0);
        __m128 pFactor = _mm_set1_ps(255.0);
        __m128 pHueShift = _mm_set1_ps(hueShift);
        __m128 pSaturationFactor = _mm_set1_ps(saturationFactor);
        __m128 xR, xG, xB, xA;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;
        __m128 h0, h1, h2, h3;
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        h0 = _mm_set1_ps(1.0);

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
        {
            xR = _mm_loadu_ps(srcPtrTempPx0);
            xG = _mm_loadu_ps(srcPtrTempPx1);
            xB = _mm_loadu_ps(srcPtrTempPx2);
            xA = _mm_loadu_ps(srcPtrTempPx3);

            _MM_TRANSPOSE4_PS (xR, xG, xB, xA);

            xR = _mm_div_ps(xR, pFactor);
            xG = _mm_div_ps(xG, pFactor);
            xB = _mm_div_ps(xB, pFactor);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]

            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Modify Hue Values and re-normalize H to fraction
            xG = _mm_add_ps(xG, pHueShift);

            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);
            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));

            xG = _mm_sub_ps(xG, xH);

            // Modify Saturation Values
            xS = _mm_mul_ps(xS, pSaturationFactor);
            xS = _mm_min_ps(xS, pOnes);
            xS = _mm_max_ps(xS, pZeros);

            _MM_TRANSPOSE4_PS (h0, xG, xS, xV);

            __m128 h1, h2, h3;

            h1 = xG;
            h2 = xS;
            h3 = xV;

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            x0 = _mm_shuffle_ps(x0,x0, _MM_SHUFFLE(0,3,2,1));
            x1 = _mm_shuffle_ps(x1,x1, _MM_SHUFFLE(0,3,2,1));
            x2 = _mm_shuffle_ps(x2,x2, _MM_SHUFFLE(0,3,2,1));
            x3 = _mm_shuffle_ps(x3,x3, _MM_SHUFFLE(0,3,2,1));

            x0 = _mm_mul_ps(x0, pFactor);
            x1 = _mm_mul_ps(x1, pFactor);
            x2 = _mm_mul_ps(x2, pFactor);
            x3 = _mm_mul_ps(x3, pFactor);

            x0 = _mm_mul_ps(x0, pMul);
            x1 = _mm_mul_ps(x1, pMul);
            x2 = _mm_mul_ps(x2, pMul);
            x3 = _mm_mul_ps(x3, pMul);

            x0 = _mm_add_ps(x0, pAdd);
            x1 = _mm_add_ps(x1, pAdd);
            x2 = _mm_add_ps(x2, pAdd);
            x3 = _mm_add_ps(x3, pAdd);

            _mm_storeu_ps(dstPtrTempPx0, x0);
            _mm_storeu_ps(dstPtrTempPx1, x1);
            _mm_storeu_ps(dstPtrTempPx2, x2);
            _mm_storeu_ps(dstPtrTempPx3, x3);

            srcPtrTempPx0 += 12;
            srcPtrTempPx1 += 12;
            srcPtrTempPx2 += 12;
            srcPtrTempPx3 += 12;
            dstPtrTempPx0 += 12;
            dstPtrTempPx1 += 12;
            dstPtrTempPx2 += 12;
            dstPtrTempPx3 += 12;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

            srcPtrTempR = srcPtrTempPx0;
            srcPtrTempG = srcPtrTempPx0 + 1;
            srcPtrTempB = srcPtrTempPx0 + 2;
            dstPtrTempR = dstPtrTempPx0;
            dstPtrTempG = dstPtrTempPx0 + 1;
            dstPtrTempB = dstPtrTempPx0 + 2;

            // RGB to HSV

            Rpp32f hue, sat, val;
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = *srcPtrTempR / 255;
            gf = *srcPtrTempG / 255;
            bf = *srcPtrTempB / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                hue = 0;
            }
            else if (cmax == rf)
            {
                hue = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                hue = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                hue = round(60 * (((rf - gf) / delta) + 4));
            }

            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            if (cmax == 0)
            {
                sat = 0;
            }
            else
            {
                sat = delta / cmax;
            }

            val = cmax;

            // Modify Hue and Saturation

            hue = hue + hueShiftAngle;
            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            sat *= saturationFactor;
            sat = (sat < (Rpp32f) 1) ? sat : ((Rpp32f) 1);
            sat = (sat > (Rpp32f) 0) ? sat : ((Rpp32f) 0);

            // HSV to RGB

            Rpp32f c, x, m;
            c = val * sat;
            x = c * (1 - abs(int(fmod((hue / 60), 2)) - 1));
            m = val - c;

            if ((0 <= hue) && (hue < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= hue) && (hue < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= hue) && (hue < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= hue) && (hue < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= hue) && (hue < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= hue) && (hue < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = round((rf + m) * 255);
            *dstPtrTempG = round((gf + m) * 255);
            *dstPtrTempB = round((bf + m) * 255);

            srcPtrTempPx0 += 3;
            dstPtrTempPx0 += 3;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus color_twist_f16_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                         Rpp32f *batch_alpha, Rpp32f *batch_beta,
                         Rpp32f *batch_hueShift, Rpp32f *batch_saturationFactor,
                         RppiROI *roiPoints, Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp64u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f hueShift = batch_hueShift[batchCount];
            Rpp32f saturationFactor = batch_saturationFactor[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            color_twist_f16_host(srcPtrImage, batch_srcSizeMax[batchCount], dstPtrImage, alpha, beta, hueShift, saturationFactor, chnFormat, channel);

            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, imageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp64u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f hueShift = batch_hueShift[batchCount];
            Rpp32f saturationFactor = batch_saturationFactor[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            color_twist_f16_host(srcPtrImage, batch_srcSizeMax[batchCount], dstPtrImage, alpha, beta, hueShift, saturationFactor, chnFormat, channel);

            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_twist_f16_host(Rpp16f* srcPtr, RppiSize srcSize, Rpp16f* dstPtr,
                    Rpp32f alpha, Rpp32f beta,
                    Rpp32f hueShift, Rpp32f saturationFactor,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    hueShift = (int)hueShift % 360;
    Rpp32f hueShiftAngle = hueShift;
    hueShift *= 0.002778f;

    if (hueShift < 0)
    {
        hueShift += 1;
        hueShiftAngle += 360;
    }

    Rpp64u totalImageDim = channel * srcSize.height * srcSize.width;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
        Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

        Rpp32f srcPtrTempRps[4], srcPtrTempGps[4], srcPtrTempBps[4];
        Rpp32f dstPtrTempRps[4], dstPtrTempGps[4], dstPtrTempBps[4];

        Rpp64u imageDim = srcSize.height * srcSize.width;
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + (imageDim);
        srcPtrTempB = srcPtr + (2 * imageDim);
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + (imageDim);
        dstPtrTempB = dstPtr + (2 * imageDim);

        Rpp64u bufferLength = srcSize.height * srcSize.width;
        Rpp64u alignedLength = bufferLength & ~3;

        __m128 pZeros = _mm_set1_ps(0.0);
        __m128 pOnes = _mm_set1_ps(1.0);
        __m128 pFactor = _mm_set1_ps(255.0);
        __m128 pHueShift = _mm_set1_ps(hueShift);
        __m128 pSaturationFactor = _mm_set1_ps(saturationFactor);
        __m128 xR, xG, xB;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;
        __m128 h0, h1, h2, h3;
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        h0 = _mm_set1_ps(1.0);

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
        {
            for(int cnt = 0; cnt < 4; cnt++)
            {
                *(srcPtrTempRps + cnt) = (Rpp32f) (*(srcPtrTempR + cnt));
                *(srcPtrTempGps + cnt) = (Rpp32f) (*(srcPtrTempG + cnt));
                *(srcPtrTempBps + cnt) = (Rpp32f) (*(srcPtrTempB + cnt));
            }

            xR = _mm_loadu_ps(srcPtrTempRps);
            xG = _mm_loadu_ps(srcPtrTempGps);
            xB = _mm_loadu_ps(srcPtrTempBps);

            xR = _mm_div_ps(xR, pFactor);
            xG = _mm_div_ps(xG, pFactor);
            xB = _mm_div_ps(xB, pFactor);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]

            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Modify Hue Values and re-normalize H to fraction
            xG = _mm_add_ps(xG, pHueShift);

            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);
            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));

            xG = _mm_sub_ps(xG, xH);

            // Modify Saturation Values
            xS = _mm_mul_ps(xS, pSaturationFactor);
            xS = _mm_min_ps(xS, pOnes);
            xS = _mm_max_ps(xS, pZeros);

            _MM_TRANSPOSE4_PS (h0, xG, xS, xV);

            __m128 h1, h2, h3;

            h1 = xG;
            h2 = xS;
            h3 = xV;

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            _MM_TRANSPOSE4_PS (x0, x1, x2, x3);

            x1 = _mm_mul_ps(x1, pFactor);
            x2 = _mm_mul_ps(x2, pFactor);
            x3 = _mm_mul_ps(x3, pFactor);

            x1 = _mm_mul_ps(x1, pMul);
            x2 = _mm_mul_ps(x2, pMul);
            x3 = _mm_mul_ps(x3, pMul);

            x1 = _mm_add_ps(x1, pAdd);
            x2 = _mm_add_ps(x2, pAdd);
            x3 = _mm_add_ps(x3, pAdd);

            _mm_storeu_ps(dstPtrTempRps, x1);
            _mm_storeu_ps(dstPtrTempGps, x2);
            _mm_storeu_ps(dstPtrTempBps, x3);

            for(int cnt = 0; cnt < 4; cnt++)
            {
                *(dstPtrTempR + cnt) = (Rpp16f) (*(dstPtrTempRps + cnt));
                *(dstPtrTempG + cnt) = (Rpp16f) (*(dstPtrTempGps + cnt));
                *(dstPtrTempB + cnt) = (Rpp16f) (*(dstPtrTempBps + cnt));
            }

            srcPtrTempR += 4;
            srcPtrTempG += 4;
            srcPtrTempB += 4;
            dstPtrTempR += 4;
            dstPtrTempG += 4;
            dstPtrTempB += 4;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            // RGB to HSV

            Rpp32f hue, sat, val;
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = *srcPtrTempR / 255;
            gf = *srcPtrTempG / 255;
            bf = *srcPtrTempB / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                hue = 0;
            }
            else if (cmax == rf)
            {
                hue = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                hue = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                hue = round(60 * (((rf - gf) / delta) + 4));
            }

            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            if (cmax == 0)
            {
                sat = 0;
            }
            else
            {
                sat = delta / cmax;
            }

            val = cmax;

            // Modify Hue and Saturation

            hue = hue + hueShiftAngle;
            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            sat *= saturationFactor;
            sat = (sat < (Rpp32f) 1) ? sat : ((Rpp32f) 1);
            sat = (sat > (Rpp32f) 0) ? sat : ((Rpp32f) 0);

            // HSV to RGB

            Rpp32f c, x, m;
            c = val * sat;
            x = c * (1 - abs(int(fmod((hue / 60), 2)) - 1));
            m = val - c;

            if ((0 <= hue) && (hue < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= hue) && (hue < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= hue) && (hue < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= hue) && (hue < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= hue) && (hue < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= hue) && (hue < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = round((rf + m) * 255);
            *dstPtrTempG = round((gf + m) * 255);
            *dstPtrTempB = round((bf + m) * 255);

            srcPtrTempR++;
            srcPtrTempG++;
            srcPtrTempB++;
            dstPtrTempR++;
            dstPtrTempG++;
            dstPtrTempB++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp16f *srcPtrTempPx0, *srcPtrTempPx1, *srcPtrTempPx2, *srcPtrTempPx3;
        Rpp16f *dstPtrTempPx0, *dstPtrTempPx1, *dstPtrTempPx2, *dstPtrTempPx3;

        Rpp32f srcPtrTempPx0ps[4], srcPtrTempPx1ps[4], srcPtrTempPx2ps[4], srcPtrTempPx3ps[4];
        Rpp32f dstPtrTempPx0ps[4], dstPtrTempPx1ps[4], dstPtrTempPx2ps[4], dstPtrTempPx3ps[4];

        srcPtrTempPx0 = srcPtr;
        srcPtrTempPx1 = srcPtr + 3;
        srcPtrTempPx2 = srcPtr + 6;
        srcPtrTempPx3 = srcPtr + 9;
        dstPtrTempPx0 = dstPtr;
        dstPtrTempPx1 = dstPtr + 3;
        dstPtrTempPx2 = dstPtr + 6;
        dstPtrTempPx3 = dstPtr + 9;

        Rpp64u bufferLength = totalImageDim;
        Rpp64u alignedLength = (bufferLength / 12) * 12;

        __m128 pZeros = _mm_set1_ps(0.0);
        __m128 pOnes = _mm_set1_ps(1.0);
        __m128 pFactor = _mm_set1_ps(255.0);
        __m128 pHueShift = _mm_set1_ps(hueShift);
        __m128 pSaturationFactor = _mm_set1_ps(saturationFactor);
        __m128 xR, xG, xB, xA;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;
        __m128 h0, h1, h2, h3;
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        h0 = _mm_set1_ps(1.0);

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
        {
            for(int cnt = 0; cnt < 4; cnt++)
            {
                *(srcPtrTempPx0ps + cnt) = (Rpp32f) (*(srcPtrTempPx0 + cnt));
                *(srcPtrTempPx1ps + cnt) = (Rpp32f) (*(srcPtrTempPx1 + cnt));
                *(srcPtrTempPx2ps + cnt) = (Rpp32f) (*(srcPtrTempPx2 + cnt));
                *(srcPtrTempPx3ps + cnt) = (Rpp32f) (*(srcPtrTempPx3 + cnt));
            }

            xR = _mm_loadu_ps(srcPtrTempPx0ps);
            xG = _mm_loadu_ps(srcPtrTempPx1ps);
            xB = _mm_loadu_ps(srcPtrTempPx2ps);
            xA = _mm_loadu_ps(srcPtrTempPx3ps);

            _MM_TRANSPOSE4_PS (xR, xG, xB, xA);

            xR = _mm_div_ps(xR, pFactor);
            xG = _mm_div_ps(xG, pFactor);
            xB = _mm_div_ps(xB, pFactor);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]

            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Modify Hue Values and re-normalize H to fraction
            xG = _mm_add_ps(xG, pHueShift);

            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);
            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));

            xG = _mm_sub_ps(xG, xH);

            // Modify Saturation Values
            xS = _mm_mul_ps(xS, pSaturationFactor);
            xS = _mm_min_ps(xS, pOnes);
            xS = _mm_max_ps(xS, pZeros);

            _MM_TRANSPOSE4_PS (h0, xG, xS, xV);

            __m128 h1, h2, h3;

            h1 = xG;
            h2 = xS;
            h3 = xV;

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            x0 = _mm_shuffle_ps(x0,x0, _MM_SHUFFLE(0,3,2,1));
            x1 = _mm_shuffle_ps(x1,x1, _MM_SHUFFLE(0,3,2,1));
            x2 = _mm_shuffle_ps(x2,x2, _MM_SHUFFLE(0,3,2,1));
            x3 = _mm_shuffle_ps(x3,x3, _MM_SHUFFLE(0,3,2,1));

            x0 = _mm_mul_ps(x0, pFactor);
            x1 = _mm_mul_ps(x1, pFactor);
            x2 = _mm_mul_ps(x2, pFactor);
            x3 = _mm_mul_ps(x3, pFactor);

            x0 = _mm_mul_ps(x0, pMul);
            x1 = _mm_mul_ps(x1, pMul);
            x2 = _mm_mul_ps(x2, pMul);
            x3 = _mm_mul_ps(x3, pMul);

            x0 = _mm_add_ps(x0, pAdd);
            x1 = _mm_add_ps(x1, pAdd);
            x2 = _mm_add_ps(x2, pAdd);
            x3 = _mm_add_ps(x3, pAdd);

            _mm_storeu_ps(dstPtrTempPx0ps, x0);
            _mm_storeu_ps(dstPtrTempPx1ps, x1);
            _mm_storeu_ps(dstPtrTempPx2ps, x2);
            _mm_storeu_ps(dstPtrTempPx3ps, x3);

            for(int cnt = 0; cnt < 3; cnt++)
            {
                *(dstPtrTempPx0 + cnt) = (Rpp16f) (*(dstPtrTempPx0ps + cnt));
                *(dstPtrTempPx1 + cnt) = (Rpp16f) (*(dstPtrTempPx1ps + cnt));
                *(dstPtrTempPx2 + cnt) = (Rpp16f) (*(dstPtrTempPx2ps + cnt));
                *(dstPtrTempPx3 + cnt) = (Rpp16f) (*(dstPtrTempPx3ps + cnt));
            }

            srcPtrTempPx0 += 12;
            srcPtrTempPx1 += 12;
            srcPtrTempPx2 += 12;
            srcPtrTempPx3 += 12;
            dstPtrTempPx0 += 12;
            dstPtrTempPx1 += 12;
            dstPtrTempPx2 += 12;
            dstPtrTempPx3 += 12;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

            srcPtrTempR = srcPtrTempPx0;
            srcPtrTempG = srcPtrTempPx0 + 1;
            srcPtrTempB = srcPtrTempPx0 + 2;
            dstPtrTempR = dstPtrTempPx0;
            dstPtrTempG = dstPtrTempPx0 + 1;
            dstPtrTempB = dstPtrTempPx0 + 2;

            // RGB to HSV

            Rpp32f hue, sat, val;
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = *srcPtrTempR / 255;
            gf = *srcPtrTempG / 255;
            bf = *srcPtrTempB / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                hue = 0;
            }
            else if (cmax == rf)
            {
                hue = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                hue = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                hue = round(60 * (((rf - gf) / delta) + 4));
            }

            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            if (cmax == 0)
            {
                sat = 0;
            }
            else
            {
                sat = delta / cmax;
            }

            val = cmax;

            // Modify Hue and Saturation

            hue = hue + hueShiftAngle;
            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            sat *= saturationFactor;
            sat = (sat < (Rpp32f) 1) ? sat : ((Rpp32f) 1);
            sat = (sat > (Rpp32f) 0) ? sat : ((Rpp32f) 0);

            // HSV to RGB

            Rpp32f c, x, m;
            c = val * sat;
            x = c * (1 - abs(int(fmod((hue / 60), 2)) - 1));
            m = val - c;

            if ((0 <= hue) && (hue < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= hue) && (hue < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= hue) && (hue < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= hue) && (hue < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= hue) && (hue < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= hue) && (hue < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = round((rf + m) * 255);
            *dstPtrTempG = round((gf + m) * 255);
            *dstPtrTempB = round((bf + m) * 255);

            srcPtrTempPx0 += 3;
            dstPtrTempPx0 += 3;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus color_twist_i8_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                         Rpp32f *batch_alpha, Rpp32f *batch_beta,
                         Rpp32f *batch_hueShift, Rpp32f *batch_saturationFactor,
                         RppiROI *roiPoints, Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp64u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f hueShift = batch_hueShift[batchCount];
            Rpp32f saturationFactor = batch_saturationFactor[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp8u *srcPtrImage8u = (Rpp8u*) calloc(channel * imageDimMax, sizeof(Rpp8u));
            Rpp8u *dstPtrImage8u = (Rpp8u*) calloc(channel * imageDimMax, sizeof(Rpp8u));

            T *srcPtrImageTemp;
            srcPtrImageTemp = srcPtrImage;

            Rpp8u *srcPtrImage8uTemp;
            srcPtrImage8uTemp = srcPtrImage8u;

            for (int i = 0; i < channel * imageDimMax; i++)
            {
                *srcPtrImage8uTemp = (Rpp8u) RPPPIXELCHECK(((Rpp32s) *srcPtrImageTemp) + 128);
                srcPtrImageTemp++;
                srcPtrImage8uTemp++;
            }

            color_twist_host(srcPtrImage8u, batch_srcSizeMax[batchCount], dstPtrImage8u, alpha, beta, hueShift, saturationFactor, chnFormat, channel);

            if (outputFormatToggle == 1)
            {
                Rpp8u *dstPtrImageUnpadded = (Rpp8u*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp8u));
                Rpp8u *dstPtrImageUnpaddedCopy = (Rpp8u*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp8u));

                compute_unpadded_from_padded_host(dstPtrImage8u, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(Rpp8u));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage8u, (Rpp8u) 0, imageDimMax * channel * sizeof(Rpp8u));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage8u, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }

            T *dstPtrImageTemp;
            dstPtrImageTemp = dstPtrImage;

            Rpp8u *dstPtrImage8uTemp;
            dstPtrImage8uTemp = dstPtrImage8u;

            for (int i = 0; i < channel * imageDimMax; i++)
            {
                *dstPtrImageTemp = (Rpp8s) (((Rpp32s) *dstPtrImage8uTemp) - 128);
                dstPtrImageTemp++;
                dstPtrImage8uTemp++;
            }

            free(srcPtrImage8u);
            free(dstPtrImage8u);
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp64u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32f hueShift = batch_hueShift[batchCount];
            Rpp32f saturationFactor = batch_saturationFactor[batchCount];
            Rpp32f alpha = batch_alpha[batchCount];
            Rpp32f beta = batch_beta[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp8u *srcPtrImage8u = (Rpp8u*) calloc(channel * imageDimMax, sizeof(Rpp8u));
            Rpp8u *dstPtrImage8u = (Rpp8u*) calloc(channel * imageDimMax, sizeof(Rpp8u));

            T *srcPtrImageTemp;
            srcPtrImageTemp = srcPtrImage;

            Rpp8u *srcPtrImage8uTemp;
            srcPtrImage8uTemp = srcPtrImage8u;

            for (int i = 0; i < channel * imageDimMax; i++)
            {
                *srcPtrImage8uTemp = (Rpp8u) RPPPIXELCHECK(((Rpp32s) *srcPtrImageTemp) + 128);
                srcPtrImageTemp++;
                srcPtrImage8uTemp++;
            }

            color_twist_host(srcPtrImage8u, batch_srcSizeMax[batchCount], dstPtrImage8u, alpha, beta, hueShift, saturationFactor, chnFormat, channel);

            if (outputFormatToggle == 1)
            {
                Rpp8u *dstPtrImageUnpadded = (Rpp8u*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp8u));
                Rpp8u *dstPtrImageUnpaddedCopy = (Rpp8u*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(Rpp8u));

                compute_unpadded_from_padded_host(dstPtrImage8u, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width * sizeof(Rpp8u));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_srcSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtrImage8u, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }

            T *dstPtrImageTemp;
            dstPtrImageTemp = dstPtrImage;

            Rpp8u *dstPtrImage8uTemp;
            dstPtrImage8uTemp = dstPtrImage8u;

            for (int i = 0; i < channel * imageDimMax; i++)
            {
                *dstPtrImageTemp = (Rpp8s) (((Rpp32s) *dstPtrImage8uTemp) - 128);
                dstPtrImageTemp++;
                dstPtrImage8uTemp++;
            }

            free(srcPtrImage8u);
            free(dstPtrImage8u);
        }
    }

    return RPP_SUCCESS;
}

/**************** crop_mirror ***************/

template <typename T, typename U>
RppStatus crop_mirror_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32u *batch_mirrorFlag, Rpp32u outputFormatToggle,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            if (mirrorFlag == 0)
            {
                for(int c = 0; c < channel; c++)
                {
                    T *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x1;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        memcpy(dstPtrChannel, srcPtrChannelROI, batch_dstSize[batchCount].width * sizeof(T));
                        dstPtrChannel += batch_dstSizeMax[batchCount].width;
                        srcPtrChannelROI += batch_srcSizeMax[batchCount].width;
                    }
                }
            }
            else if (mirrorFlag == 1)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width + batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    T *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x2;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        T *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 16) * 16;

                        __m128i px0;
                        __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            srcPtrChannelROI -= 15;
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                            px0 = _mm_shuffle_epi8(px0, vMask);
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            srcPtrChannelROI -= 1;
                            dstPtrTemp += 16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = (T) *srcPtrChannelROI--;
                        }

                        srcPtrChannelROI += srcROIIncrement;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * batch_dstSize[batchCount].width;

            if (mirrorFlag == 0)
            {
                T *srcPtrImageROI, *dstPtrImageTemp;
                srcPtrImageROI = srcPtrImage + (y1 * srcElementsInRowMax) + (x1 * channel);
                dstPtrImageTemp = dstPtrImage;


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    memcpy(dstPtrImageTemp, srcPtrImageROI, elementsInRowROI * sizeof(T));
                    dstPtrImageTemp += dstElementsInRowMax;
                    srcPtrImageROI += srcElementsInRowMax;
                }
            }
            else if (mirrorFlag == 1)
            {
                T  *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + ((x2 - 1) * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax + elementsInRowROI;


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    T *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;
                    Rpp32u alignedLength = ((bufferLength / 15) * 15) - 1;

                    __m128i px0;
                    __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                    {
                        srcPtrROI -= 13;
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);
                        px0 = _mm_shuffle_epi8(px0, vMask);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrROI -= 2;
                        dstPtrTemp += 15;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                    {
                        memcpy(dstPtrTemp, srcPtrROI, channel * sizeof(T));
                        dstPtrTemp += channel;
                        srcPtrROI -= channel;
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus crop_mirror_host(T* srcPtr, RppiSize srcSize, U* dstPtr, RppiSize dstSize,
                                     Rpp32u crop_pos_x, Rpp32u crop_pos_y,
                                     Rpp32u mirrorFlag, Rpp32u outputFormatToggle,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    U *dstPtrTemp;
    dstPtrTemp = dstPtr;

    Rpp32u srcImageDim = srcSize.height * srcSize.width;

    if(chnFormat == RPPI_CHN_PLANAR)
    {
        if (mirrorFlag == 0)
        {
            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannelROI;
                srcPtrChannelROI = srcPtr + (c * srcImageDim) + (crop_pos_y * srcSize.width) + crop_pos_x;

                for(int i = 0; i < dstSize.height; i++)
                {
                    memcpy(dstPtrTemp, srcPtrChannelROI, dstSize.width * sizeof(T));
                    dstPtrTemp += dstSize.width;
                    srcPtrChannelROI += srcSize.width;
                }
            }
        }
        else if (mirrorFlag == 1)
        {
            Rpp32u srcROIIncrement = srcSize.width + dstSize.width;
            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannelROI;
                srcPtrChannelROI = srcPtr + (c * srcImageDim) + (crop_pos_y * srcSize.width) + crop_pos_x + dstSize.width - 1;

                for(int i = 0; i < dstSize.height; i++)
                {
                    Rpp32u bufferLength = dstSize.width;
                    Rpp32u alignedLength = (bufferLength / 16) * 16;

                    __m128i px0;
                    __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        srcPtrChannelROI -= 15;
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                        px0 = _mm_shuffle_epi8(px0, vMask);

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrChannelROI -= 1;
                        dstPtrTemp += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = *srcPtrChannelROI--;
                    }

                    srcPtrChannelROI += srcROIIncrement;
                }
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u srcElementsInRow = channel * srcSize.width;
        Rpp32u dstElementsInRow = channel * dstSize.width;

        if (mirrorFlag == 0)
        {
            T *srcPtrROI;
            srcPtrROI = srcPtr + (crop_pos_y * srcElementsInRow) + (crop_pos_x * channel);

            for(int i = 0; i < dstSize.height; i++)
            {
                memcpy(dstPtrTemp, srcPtrROI, dstElementsInRow * sizeof(T));
                dstPtrTemp += dstElementsInRow;
                srcPtrROI += srcElementsInRow;
            }
        }
        else if (mirrorFlag == 1)
        {
            T  *srcPtrROI;
            srcPtrROI = srcPtr + (crop_pos_y * srcElementsInRow) + ((crop_pos_x + dstSize.width - 1) * channel);

            Rpp32u srcROIIncrement = srcElementsInRow + dstElementsInRow;

            for(int i = 0; i < dstSize.height; i++)
            {
                Rpp32u bufferLength = dstElementsInRow;
                Rpp32u alignedLength = ((bufferLength / 15) * 15) - 1;

                __m128i px0;
                __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                {
                    srcPtrROI -= 13;
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);
                    px0 = _mm_shuffle_epi8(px0, vMask);

                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrROI -= 2;
                    dstPtrTemp += 15;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                {
                    memcpy(dstPtrTemp, srcPtrROI, channel * sizeof(T));
                    dstPtrTemp += channel;
                    srcPtrROI -= channel;
                }

                srcPtrROI += srcROIIncrement;
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** crop_mirror_normalize ***************/

template <typename T, typename U>
RppStatus crop_mirror_normalize_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32f *batch_mean, Rpp32f *batch_stdDev,
                                     Rpp32u *batch_mirrorFlag, Rpp32u outputFormatToggle,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = batch_mean[batchCount];
            Rpp32f stdDev = batch_stdDev[batchCount];
            Rpp32f invStdDev = 1.0 / stdDev;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            if (mirrorFlag == 0)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width - batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    T *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x1;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        T *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 16) * 16;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i px0, px1, px2, px3;
                        __m128 p0, p1, p2, p3;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            p0 = _mm_sub_ps(p0, vMean);
                            p1 = _mm_sub_ps(p1, vMean);
                            p2 = _mm_sub_ps(p2, vMean);
                            p3 = _mm_sub_ps(p3, vMean);
                            px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                            px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                            px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                            px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));

                            px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                            px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            srcPtrChannelROI += 16;
                            dstPtrTemp += 16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrChannelROI) - mean) * invStdDev);
                            dstPtrTemp++;
                            srcPtrChannelROI++;
                        }

                        srcPtrChannelROI += srcROIIncrement;

                    }
                }
            }
            else if (mirrorFlag == 1)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width + batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    T *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x2;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        T *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 16) * 16;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i px0, px1, px2, px3;
                        __m128 p0, p1, p2, p3;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(invStdDev);
                        __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            srcPtrChannelROI -= 15;
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                            px0 = _mm_shuffle_epi8(px0, vMask);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            p0 = _mm_sub_ps(p0, vMean);
                            p1 = _mm_sub_ps(p1, vMean);
                            p2 = _mm_sub_ps(p2, vMean);
                            p3 = _mm_sub_ps(p3, vMean);
                            px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                            px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                            px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                            px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));

                            px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                            px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            srcPtrChannelROI -= 1;
                            dstPtrTemp += 16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrChannelROI) - mean) * invStdDev);
                            dstPtrTemp++;
                            srcPtrChannelROI--;
                        }

                        srcPtrChannelROI += srcROIIncrement;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(T));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (T) 0, dstImageDimMax * channel * sizeof(T));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = batch_mean[batchCount];
            Rpp32f stdDev = batch_stdDev[batchCount];
            Rpp32f invStdDev = 1.0 / stdDev;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * batch_dstSize[batchCount].width;

            if (mirrorFlag == 0)
            {
                T  *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + (x1 * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax - elementsInRowROI;


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    T *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;
                    Rpp32u alignedLength = (bufferLength / 16) * 16;

                    __m128i const zero = _mm_setzero_si128();
                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1, p2, p3;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        p0 = _mm_sub_ps(p0, vMean);
                        p1 = _mm_sub_ps(p1, vMean);
                        p2 = _mm_sub_ps(p2, vMean);
                        p3 = _mm_sub_ps(p3, vMean);
                        px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                        px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                        px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                        px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));

                        px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                        px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrROI += 16;
                        dstPtrTemp += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrROI) - mean) * invStdDev);
                        dstPtrTemp++;
                        srcPtrROI++;
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            else if (mirrorFlag == 1)
            {
                T  *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + ((x2 - 1) * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax + elementsInRowROI;


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    T *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;
                    Rpp32u alignedLength = ((bufferLength / 15) * 15) - 1;

                    __m128i const zero = _mm_setzero_si128();
                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1, p2, p3;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(invStdDev);
                    __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                    {
                        srcPtrROI -= 13;
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);
                        px0 = _mm_shuffle_epi8(px0, vMask);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        p0 = _mm_sub_ps(p0, vMean);
                        p1 = _mm_sub_ps(p1, vMean);
                        p2 = _mm_sub_ps(p2, vMean);
                        p3 = _mm_sub_ps(p3, vMean);
                        px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                        px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                        px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                        px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));

                        px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                        px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrROI -= 2;
                        dstPtrTemp += 15;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrROI) - mean) * invStdDev);
                            dstPtrTemp++;
                            srcPtrROI++;
                        }
                        srcPtrROI -= (2 * channel);
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(T));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus crop_mirror_normalize_host(T* srcPtr, RppiSize srcSize, U* dstPtr, RppiSize dstSize,
                                     Rpp32u crop_pos_x, Rpp32u crop_pos_y,
                                     Rpp32f mean, Rpp32f stdDev,
                                     Rpp32u mirrorFlag, Rpp32u outputFormatToggle,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    U *dstPtrTemp;
    dstPtrTemp = dstPtr;

    Rpp32u srcImageDim = srcSize.height * srcSize.width;
    Rpp32f invStdDev = 1.0 / stdDev;

    if(chnFormat == RPPI_CHN_PLANAR)
    {
        if (mirrorFlag == 0)
        {
            Rpp32u srcROIIncrement = srcSize.width - dstSize.width;
            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannelROI;
                srcPtrChannelROI = srcPtr + (c * srcImageDim) + (crop_pos_y * srcSize.width) + crop_pos_x;

                for(int i = 0; i < dstSize.height; i++)
                {
                    Rpp32u bufferLength = dstSize.width;
                    Rpp32u alignedLength = (bufferLength / 16) * 16;

                    __m128i const zero = _mm_setzero_si128();
                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1, p2, p3;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        p0 = _mm_sub_ps(p0, vMean);
                        p1 = _mm_sub_ps(p1, vMean);
                        p2 = _mm_sub_ps(p2, vMean);
                        p3 = _mm_sub_ps(p3, vMean);
                        px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                        px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                        px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                        px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));

                        px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                        px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrChannelROI += 16;
                        dstPtrTemp += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrChannelROI) - mean) * invStdDev);
                        dstPtrTemp++;
                        srcPtrChannelROI++;
                    }

                    srcPtrChannelROI += srcROIIncrement;
                }
            }
        }
        else if (mirrorFlag == 1)
        {
            Rpp32u srcROIIncrement = srcSize.width + dstSize.width;
            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannelROI;
                srcPtrChannelROI = srcPtr + (c * srcImageDim) + (crop_pos_y * srcSize.width) + crop_pos_x + dstSize.width - 1;

                for(int i = 0; i < dstSize.height; i++)
                {
                    Rpp32u bufferLength = dstSize.width;
                    Rpp32u alignedLength = (bufferLength / 16) * 16;

                    __m128i const zero = _mm_setzero_si128();
                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1, p2, p3;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(invStdDev);
                    __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        srcPtrChannelROI -= 15;
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                        px0 = _mm_shuffle_epi8(px0, vMask);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        p0 = _mm_sub_ps(p0, vMean);
                        p1 = _mm_sub_ps(p1, vMean);
                        p2 = _mm_sub_ps(p2, vMean);
                        p3 = _mm_sub_ps(p3, vMean);
                        px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                        px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                        px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                        px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));

                        px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                        px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                        srcPtrChannelROI -= 1;
                        dstPtrTemp += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrChannelROI) - mean) * invStdDev);
                        dstPtrTemp++;
                        srcPtrChannelROI--;
                    }

                    srcPtrChannelROI += srcROIIncrement;
                }
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u srcElementsInRow = channel * srcSize.width;
        Rpp32u dstElementsInRow = channel * dstSize.width;

        if (mirrorFlag == 0)
        {
            T  *srcPtrROI;
            srcPtrROI = srcPtr + (crop_pos_y * srcElementsInRow) + (crop_pos_x * channel);

            Rpp32u srcROIIncrement = srcElementsInRow - dstElementsInRow;

            for(int i = 0; i < dstSize.height; i++)
            {
                Rpp32u bufferLength = dstElementsInRow;
                Rpp32u alignedLength = (bufferLength / 16) * 16;

                __m128i const zero = _mm_setzero_si128();
                __m128i px0, px1, px2, px3;
                __m128 p0, p1, p2, p3;
                __m128 vMean = _mm_set1_ps(mean);
                __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);

                    px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                    px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                    p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                    p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                    p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                    p0 = _mm_sub_ps(p0, vMean);
                    p1 = _mm_sub_ps(p1, vMean);
                    p2 = _mm_sub_ps(p2, vMean);
                    p3 = _mm_sub_ps(p3, vMean);
                    px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                    px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                    px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                    px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));

                    px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                    px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                    px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrROI += 16;
                    dstPtrTemp += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrROI) - mean) * invStdDev);
                    dstPtrTemp++;
                    srcPtrROI++;
                }

                srcPtrROI += srcROIIncrement;
            }
        }
        else if (mirrorFlag == 1)
        {
            T  *srcPtrROI;
            srcPtrROI = srcPtr + (crop_pos_y * srcElementsInRow) + ((crop_pos_x + dstSize.width - 1) * channel);

            Rpp32u srcROIIncrement = srcElementsInRow + dstElementsInRow;

            for(int i = 0; i < dstSize.height; i++)
            {
                Rpp32u bufferLength = dstElementsInRow;
                Rpp32u alignedLength = ((bufferLength / 15) * 15) - 1;

                __m128i const zero = _mm_setzero_si128();
                __m128i px0, px1, px2, px3;
                __m128 p0, p1, p2, p3;
                __m128 vMean = _mm_set1_ps(mean);
                __m128 vInvStdDev = _mm_set1_ps(invStdDev);
                __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                {
                    srcPtrROI -= 13;
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);
                    px0 = _mm_shuffle_epi8(px0, vMask);

                    px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                    px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                    p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                    p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                    p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                    p0 = _mm_sub_ps(p0, vMean);
                    p1 = _mm_sub_ps(p1, vMean);
                    p2 = _mm_sub_ps(p2, vMean);
                    p3 = _mm_sub_ps(p3, vMean);
                    px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                    px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                    px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                    px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));

                    px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                    px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                    px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrROI -= 2;
                    dstPtrTemp += 15;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                {
                    for(int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*srcPtrROI) - mean) * invStdDev);
                        dstPtrTemp++;
                        srcPtrROI++;
                    }
                    srcPtrROI -= (2 * channel);
                }

                srcPtrROI += srcROIIncrement;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_f32_host_batch(Rpp32f* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp32f* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32f *batch_mean, Rpp32f *batch_stdDev,
                                     Rpp32u *batch_mirrorFlag, Rpp32u outputFormatToggle,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = batch_mean[batchCount];
            Rpp32f stdDev = batch_stdDev[batchCount];
            Rpp32f invStdDev = 1.0 / stdDev;

            Rpp32f *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            if (mirrorFlag == 0)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width - batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    Rpp32f *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x1;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        Rpp32f *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 4) * 4;

                        __m128 p0;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                        {
                            p0 = _mm_loadu_ps(srcPtrChannelROI);
                            p0 = _mm_sub_ps(p0, vMean);
                            p0 = _mm_mul_ps(p0, vInvStdDev);
                            _mm_storeu_ps(dstPtrTemp, p0);

                            srcPtrChannelROI += 4;
                            dstPtrTemp += 4;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (*srcPtrChannelROI - mean) * invStdDev;
                            dstPtrTemp++;
                            srcPtrChannelROI++;
                        }

                        srcPtrChannelROI += srcROIIncrement;

                    }
                }
            }
            else if (mirrorFlag == 1)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width + batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    Rpp32f *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x2;

                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        Rpp32f *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 4) * 4;

                        __m128 p0;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                        {
                            srcPtrChannelROI -= 3;
                            p0 = _mm_loadu_ps(srcPtrChannelROI);
                            p0 = _mm_shuffle_ps(p0, p0, _MM_SHUFFLE(0,1,2,3));
                            p0 = _mm_sub_ps(p0, vMean);
                            p0 = _mm_mul_ps(p0, vInvStdDev);
                            _mm_storeu_ps(dstPtrTemp, p0);

                            srcPtrChannelROI -= 1;
                            dstPtrTemp += 4;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (*srcPtrChannelROI - mean) * invStdDev;
                            dstPtrTemp++;
                            srcPtrChannelROI--;
                        }

                        srcPtrChannelROI += srcROIIncrement;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                Rpp32f *dstPtrImageUnpadded = (Rpp32f*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(Rpp32f));
                Rpp32f *dstPtrImageUnpaddedCopy = (Rpp32f*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(Rpp32f));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(Rpp32f));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (Rpp32f) 0, dstImageDimMax * channel * sizeof(Rpp32f));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = batch_mean[batchCount];
            Rpp32f stdDev = batch_stdDev[batchCount];
            Rpp32f invStdDev = 1.0 / stdDev;

            Rpp32f *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * batch_dstSize[batchCount].width;

            if (mirrorFlag == 0)
            {
                Rpp32f *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + (x1 * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax - elementsInRowROI;


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp32f *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;
                    Rpp32u alignedLength = (bufferLength / 4) * 4;

                    __m128 p0;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        p0 = _mm_loadu_ps(srcPtrROI);
                        p0 = _mm_sub_ps(p0, vMean);
                        p0 = _mm_mul_ps(p0, vInvStdDev);
                        _mm_storeu_ps(dstPtrTemp, p0);

                        srcPtrROI += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (*srcPtrROI - mean) * invStdDev;
                        dstPtrTemp++;
                        srcPtrROI++;
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            else if (mirrorFlag == 1)
            {
                Rpp32f *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + ((x2 - 1) * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax + elementsInRowROI;


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp32f *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;

                    __m128 p0;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < 3; vectorLoopCount+=3)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp = (*srcPtrROI - mean) * invStdDev;
                            dstPtrTemp++;
                            srcPtrROI++;
                        }
                        srcPtrROI -= (2 * channel);
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        p0 = _mm_loadu_ps(srcPtrROI);
                        p0 = _mm_sub_ps(p0, vMean);
                        p0 = _mm_mul_ps(p0, vInvStdDev);
                        _mm_storeu_ps(dstPtrTemp, p0);

                        srcPtrROI -= 3;
                        dstPtrTemp += 3;
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            if (outputFormatToggle == 1)
            {
                Rpp32f *dstPtrImageUnpadded = (Rpp32f*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(Rpp32f));
                Rpp32f *dstPtrImageUnpaddedCopy = (Rpp32f*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(Rpp32f));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(Rpp32f));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_f16_host_batch(Rpp16f* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp16f* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32f *batch_mean, Rpp32f *batch_stdDev,
                                     Rpp32u *batch_mirrorFlag, Rpp32u outputFormatToggle,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = batch_mean[batchCount];
            Rpp32f stdDev = batch_stdDev[batchCount];
            Rpp32f invStdDev = 1.0 / stdDev;

            Rpp16f *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32f srcPtrChannelROIps[4], dstPtrTempps[4];

            if (mirrorFlag == 0)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width - batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    Rpp16f *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x1;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        Rpp16f *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 4) * 4;

                        __m128 p0;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                        {
                            for(int cnt = 0; cnt < 4; cnt++)
                            {
                                *(srcPtrChannelROIps + cnt) = (Rpp32f) (*(srcPtrChannelROI + cnt));
                            }

                            p0 = _mm_loadu_ps(srcPtrChannelROIps);
                            p0 = _mm_sub_ps(p0, vMean);
                            p0 = _mm_mul_ps(p0, vInvStdDev);
                            _mm_storeu_ps(dstPtrTempps, p0);

                            for(int cnt = 0; cnt < 4; cnt++)
                            {
                                *(dstPtrTemp + cnt) = (Rpp16f) (*(dstPtrTempps + cnt));
                            }

                            srcPtrChannelROI += 4;
                            dstPtrTemp += 4;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (*srcPtrChannelROI - mean) * invStdDev;
                            dstPtrTemp++;
                            srcPtrChannelROI++;
                        }

                        srcPtrChannelROI += srcROIIncrement;

                    }
                }
            }
            else if (mirrorFlag == 1)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width + batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    Rpp16f *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x2;

                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        Rpp16f *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 4) * 4;

                        __m128 p0;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                        {
                            srcPtrChannelROI -= 3;

                            for(int cnt = 0; cnt < 4; cnt++)
                            {
                                *(srcPtrChannelROIps + cnt) = (Rpp32f) (*(srcPtrChannelROI + cnt));
                            }

                            p0 = _mm_loadu_ps(srcPtrChannelROIps);
                            p0 = _mm_shuffle_ps(p0, p0, _MM_SHUFFLE(0,1,2,3));
                            p0 = _mm_sub_ps(p0, vMean);
                            p0 = _mm_mul_ps(p0, vInvStdDev);
                            _mm_storeu_ps(dstPtrTempps, p0);

                            for(int cnt = 0; cnt < 4; cnt++)
                            {
                                *(dstPtrTemp + cnt) = (Rpp16f) (*(dstPtrTempps + cnt));
                            }

                            srcPtrChannelROI -= 1;
                            dstPtrTemp += 4;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (*srcPtrChannelROI - mean) * invStdDev;
                            dstPtrTemp++;
                            srcPtrChannelROI--;
                        }

                        srcPtrChannelROI += srcROIIncrement;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                Rpp16f *dstPtrImageUnpadded = (Rpp16f*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(Rpp16f));
                Rpp16f *dstPtrImageUnpaddedCopy = (Rpp16f*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(Rpp16f));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(Rpp16f));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (Rpp16f) 0, dstImageDimMax * channel * sizeof(Rpp16f));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = batch_mean[batchCount];
            Rpp32f stdDev = batch_stdDev[batchCount];
            Rpp32f invStdDev = 1.0 / stdDev;

            Rpp16f *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * batch_dstSize[batchCount].width;

            Rpp32f srcPtrROIps[4], dstPtrTempps[4];

            if (mirrorFlag == 0)
            {
                Rpp16f *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + (x1 * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax - elementsInRowROI;


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp16f *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;
                    Rpp32u alignedLength = (bufferLength / 4) * 4;

                    __m128 p0;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtrROIps + cnt) = (Rpp32f) (*(srcPtrROI + cnt));
                        }

                        p0 = _mm_loadu_ps(srcPtrROIps);
                        p0 = _mm_sub_ps(p0, vMean);
                        p0 = _mm_mul_ps(p0, vInvStdDev);
                        _mm_storeu_ps(dstPtrTempps, p0);

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTemp + cnt) = (Rpp16f) (*(dstPtrTempps + cnt));
                        }

                        srcPtrROI += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (*srcPtrROI - mean) * invStdDev;
                        dstPtrTemp++;
                        srcPtrROI++;
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            else if (mirrorFlag == 1)
            {
                Rpp16f *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + ((x2 - 1) * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax + elementsInRowROI;


                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    Rpp16f *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;

                    __m128 p0;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < 3; vectorLoopCount+=3)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp = (*srcPtrROI - mean) * invStdDev;
                            dstPtrTemp++;
                            srcPtrROI++;
                        }
                        srcPtrROI -= (2 * channel);
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtrROIps + cnt) = (Rpp32f) (*(srcPtrROI + cnt));
                        }

                        p0 = _mm_loadu_ps(srcPtrROIps);
                        p0 = _mm_sub_ps(p0, vMean);
                        p0 = _mm_mul_ps(p0, vInvStdDev);
                        _mm_storeu_ps(dstPtrTempps, p0);

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTemp + cnt) = (Rpp16f) (*(dstPtrTempps + cnt));
                        }

                        srcPtrROI -= 3;
                        dstPtrTemp += 3;
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            if (outputFormatToggle == 1)
            {
                Rpp16f *dstPtrImageUnpadded = (Rpp16f*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(Rpp16f));
                Rpp16f *dstPtrImageUnpaddedCopy = (Rpp16f*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(Rpp16f));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(Rpp16f));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus crop_mirror_normalize_u8_f_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32f *batch_mean, Rpp32f *batch_stdDev,
                                     Rpp32u *batch_mirrorFlag, Rpp32u outputFormatToggle,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = batch_mean[batchCount];
            Rpp32f stdDev = 255.0 * batch_stdDev[batchCount];
            Rpp32f invStdDev = 1.0 / stdDev;

            T *srcPtrImage;
            U *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32f dst[16];
            Rpp32f *dst0, *dst1, *dst2, *dst3;
            dst0 = dst;
            dst1 = dst + 4;
            dst2 = dst + 8;
            dst3 = dst + 12;

            if (mirrorFlag == 0)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width - batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannelROI;
                    U *dstPtrChannel;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x1;

                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        U *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 16) * 16;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i px0, px1, px2, px3;
                        __m128 p0, p1, p2, p3;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            p0 = _mm_sub_ps(p0, vMean);
                            p1 = _mm_sub_ps(p1, vMean);
                            p2 = _mm_sub_ps(p2, vMean);
                            p3 = _mm_sub_ps(p3, vMean);

                            p0 = _mm_mul_ps(p0, vInvStdDev);
                            p1 = _mm_mul_ps(p1, vInvStdDev);
                            p2 = _mm_mul_ps(p2, vInvStdDev);
                            p3 = _mm_mul_ps(p3, vInvStdDev);

                            _mm_storeu_ps(dst0, p0);
                            _mm_storeu_ps(dst1, p1);
                            _mm_storeu_ps(dst2, p2);
                            _mm_storeu_ps(dst3, p3);

                            for (int j = 0; j < 16; j++)
                            {
                                *dstPtrTemp = (U) (*(dst + j));
                                dstPtrTemp++;
                            }

                            srcPtrChannelROI += 16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (U) RPPPIXELCHECK(((Rpp32f)(*srcPtrChannelROI) - mean) * invStdDev);
                            dstPtrTemp++;
                            srcPtrChannelROI++;
                        }

                        srcPtrChannelROI += srcROIIncrement;

                    }
                }
            }
            else if (mirrorFlag == 1)
            {
                Rpp32u srcROIIncrement = batch_srcSizeMax[batchCount].width + batch_dstSize[batchCount].width;
                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannelROI;
                    U *dstPtrChannel;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x2;

                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        U *dstPtrTemp;
                        dstPtrTemp = dstPtrChannel + (i * batch_dstSizeMax[batchCount].width);

                        Rpp32u bufferLength = batch_dstSize[batchCount].width;
                        Rpp32u alignedLength = (bufferLength / 16) * 16;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i px0, px1, px2, px3;
                        __m128 p0, p1, p2, p3;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(invStdDev);
                        __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            srcPtrChannelROI -= 15;
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrChannelROI);
                            px0 = _mm_shuffle_epi8(px0, vMask);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            p0 = _mm_sub_ps(p0, vMean);
                            p1 = _mm_sub_ps(p1, vMean);
                            p2 = _mm_sub_ps(p2, vMean);
                            p3 = _mm_sub_ps(p3, vMean);

                            p0 = _mm_mul_ps(p0, vInvStdDev);
                            p1 = _mm_mul_ps(p1, vInvStdDev);
                            p2 = _mm_mul_ps(p2, vInvStdDev);
                            p3 = _mm_mul_ps(p3, vInvStdDev);

                            _mm_storeu_ps(dst0, p0);
                            _mm_storeu_ps(dst1, p1);
                            _mm_storeu_ps(dst2, p2);
                            _mm_storeu_ps(dst3, p3);

                            for (int j = 0; j < 16; j++)
                            {
                                *dstPtrTemp = (U) (*(dst + j));
                                dstPtrTemp++;
                            }

                            srcPtrChannelROI -= 1;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (U) RPPPIXELCHECK(((Rpp32f)(*srcPtrChannelROI) - mean) * invStdDev);
                            dstPtrTemp++;
                            srcPtrChannelROI--;
                        }

                        srcPtrChannelROI += srcROIIncrement;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                U *dstPtrImageUnpadded = (U*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(U));
                U *dstPtrImageUnpaddedCopy = (U*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(U));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(U));

                compute_planar_to_packed_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                memset(dstPtrImage, (U) 0, dstImageDimMax * channel * sizeof(U));

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = batch_mean[batchCount];
            Rpp32f stdDev = 255.0 * batch_stdDev[batchCount];
            Rpp32f invStdDev = 1.0 / stdDev;

            T *srcPtrImage;
            U *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * batch_dstSize[batchCount].width;

            Rpp32f dst[16];
            Rpp32f *dst0, *dst1, *dst2, *dst3;
            dst0 = dst;
            dst1 = dst + 4;
            dst2 = dst + 8;
            dst3 = dst + 12;

            if (mirrorFlag == 0)
            {
                T *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + (x1 * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax - elementsInRowROI;

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    U *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;
                    Rpp32u alignedLength = (bufferLength / 16) * 16;

                    __m128i const zero = _mm_setzero_si128();
                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1, p2, p3;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        p0 = _mm_sub_ps(p0, vMean);
                        p1 = _mm_sub_ps(p1, vMean);
                        p2 = _mm_sub_ps(p2, vMean);
                        p3 = _mm_sub_ps(p3, vMean);

                        p0 = _mm_mul_ps(p0, vInvStdDev);
                        p1 = _mm_mul_ps(p1, vInvStdDev);
                        p2 = _mm_mul_ps(p2, vInvStdDev);
                        p3 = _mm_mul_ps(p3, vInvStdDev);

                        _mm_storeu_ps(dst0, p0);
                        _mm_storeu_ps(dst1, p1);
                        _mm_storeu_ps(dst2, p2);
                        _mm_storeu_ps(dst3, p3);

                        for (int j = 0; j < 16; j++)
                        {
                            *dstPtrTemp = (U) (*(dst + j));
                            dstPtrTemp++;
                        }

                        srcPtrROI += 16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (U) RPPPIXELCHECK(((Rpp32f)(*srcPtrROI) - mean) * invStdDev);
                        dstPtrTemp++;
                        srcPtrROI++;
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            else if (mirrorFlag == 1)
            {
                T *srcPtrROI;
                srcPtrROI = srcPtrImage + (y1 * srcElementsInRowMax) + ((x2 - 1) * channel);

                Rpp32u srcROIIncrement = srcElementsInRowMax + elementsInRowROI;

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    U *dstPtrTemp;
                    dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                    Rpp32u bufferLength = elementsInRowROI;
                    Rpp32u alignedLength = ((bufferLength / 15) * 15) - 1;

                    __m128i const zero = _mm_setzero_si128();
                    __m128i px0, px1, px2, px3;
                    __m128 p0, p1, p2, p3;
                    __m128 vMean = _mm_set1_ps(mean);
                    __m128 vInvStdDev = _mm_set1_ps(invStdDev);
                    __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                    {
                        srcPtrROI -= 13;
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrROI);
                        px0 = _mm_shuffle_epi8(px0, vMask);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        p0 = _mm_sub_ps(p0, vMean);
                        p1 = _mm_sub_ps(p1, vMean);
                        p2 = _mm_sub_ps(p2, vMean);
                        p3 = _mm_sub_ps(p3, vMean);

                        p0 = _mm_mul_ps(p0, vInvStdDev);
                        p1 = _mm_mul_ps(p1, vInvStdDev);
                        p2 = _mm_mul_ps(p2, vInvStdDev);
                        p3 = _mm_mul_ps(p3, vInvStdDev);

                        _mm_storeu_ps(dst0, p0);
                        _mm_storeu_ps(dst1, p1);
                        _mm_storeu_ps(dst2, p2);
                        _mm_storeu_ps(dst3, p3);

                        for (int j = 0; j < 16; j++)
                        {
                            *dstPtrTemp = (U) (*(dst + j));
                            dstPtrTemp++;
                        }

                        srcPtrROI -= 2;
                        dstPtrTemp -= 1;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp = (U) RPPPIXELCHECK(((Rpp32f)(*srcPtrROI) - mean) * invStdDev);
                            dstPtrTemp++;
                            srcPtrROI++;
                        }
                        srcPtrROI -= (2 * channel);
                    }

                    srcPtrROI += srcROIIncrement;
                }
            }
            if (outputFormatToggle == 1)
            {
                U *dstPtrImageUnpadded = (U*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(U));
                U *dstPtrImageUnpaddedCopy = (U*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(U));

                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);

                memcpy(dstPtrImageUnpaddedCopy, dstPtrImageUnpadded, channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width * sizeof(U));

                compute_packed_to_planar_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], dstPtrImageUnpadded, channel);

                compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);

                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_u8_i8_host_batch(Rpp8u* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp8s* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32f *batch_mean, Rpp32f *batch_stdDev,
                                     Rpp32u *batch_mirrorFlag, Rpp32u outputFormatToggle,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u srcBufferSize = nbatchSize * batch_srcSizeMax[0].height * batch_srcSizeMax[0].width * channel;
    Rpp32u dstBufferSize = nbatchSize * batch_dstSizeMax[0].height * batch_dstSizeMax[0].width * channel;

    Rpp32f *srcPtrf32 = (Rpp32f *)calloc(srcBufferSize, sizeof(Rpp32f));
    Rpp32f *dstPtrf32 = (Rpp32f *)calloc(dstBufferSize, sizeof(Rpp32f));

    Rpp8u *srcPtrTemp;
    Rpp32f *srcPtrf32Temp;
    srcPtrTemp = srcPtr;
    srcPtrf32Temp = srcPtrf32;

    for (int i = 0; i < srcBufferSize; i++)
    {
        *srcPtrf32Temp = (Rpp32f) (*srcPtrTemp) / 255.0;
        srcPtrTemp++;
        srcPtrf32Temp++;
    }

    crop_mirror_normalize_f32_host_batch(srcPtrf32, batch_srcSize, batch_srcSizeMax, dstPtrf32, batch_dstSize, batch_dstSizeMax,
                                    batch_crop_pos_x, batch_crop_pos_y, batch_mean, batch_stdDev, batch_mirrorFlag, outputFormatToggle,
                                    nbatchSize, chnFormat, channel, handle);

    Rpp8s *dstPtrTemp;
    Rpp32f *dstPtrf32Temp;
    dstPtrTemp = dstPtr;
    dstPtrf32Temp = dstPtrf32;

    for (int i = 0; i < srcBufferSize; i++)
    {
        *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((*dstPtrf32Temp * 255.0) - 128);
        dstPtrTemp++;
        dstPtrf32Temp++;
    }

    free(srcPtrf32);
    free(dstPtrf32);

    return RPP_SUCCESS;
}

/**************** crop ***************/

template <typename T, typename U>
RppStatus crop_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            if (outputFormatToggle == 1)
            {
                T *srcPtrImageRoiR, *srcPtrImageRoiG, *srcPtrImageRoiB;
                T *dstPtrImageTemp;

                srcPtrImageRoiR = srcPtrImage + (y1 * batch_srcSizeMax[batchCount].width) + x1;
                srcPtrImageRoiG = srcPtrImageRoiR + srcImageDimMax;
                srcPtrImageRoiB = srcPtrImageRoiG + srcImageDimMax;

                dstPtrImageTemp = dstPtrImage;

                Rpp32u incrementSrc = batch_srcSizeMax[batchCount].width - batch_dstSize[batchCount].width;
                Rpp32u incrementDst = (batch_dstSizeMax[batchCount].width - batch_dstSize[batchCount].width) * channel;

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        *dstPtrImageTemp = *srcPtrImageRoiR;
                        srcPtrImageRoiR++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = *srcPtrImageRoiG;
                        srcPtrImageRoiG++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = *srcPtrImageRoiB;
                        srcPtrImageRoiB++;
                        dstPtrImageTemp++;
                    }
                    dstPtrImageTemp += incrementDst;
                    srcPtrImageRoiR += incrementSrc;
                    srcPtrImageRoiG += incrementSrc;
                    srcPtrImageRoiB += incrementSrc;
                }
            }
            else
            {
                for(int c = 0; c < channel; c++)
                {
                    T *dstPtrChannel, *srcPtrChannelROI;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x1;

                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        memcpy(dstPtrChannel, srcPtrChannelROI, batch_dstSize[batchCount].width * sizeof(T));
                        dstPtrChannel += batch_dstSizeMax[batchCount].width;
                        srcPtrChannelROI += batch_srcSizeMax[batchCount].width;
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * batch_dstSize[batchCount].width;

            T *srcPtrImageROI, *dstPtrImageTemp;
            srcPtrImageROI = srcPtrImage + (y1 * srcElementsInRowMax) + (x1 * channel);
            dstPtrImageTemp = dstPtrImage;

            if (outputFormatToggle == 1)
            {
                T *srcPtrImageROITemp;

                T *dstPtrImageTempG, *dstPtrImageTempB;
                dstPtrImageTempG = dstPtrImageTemp + (batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width);
                dstPtrImageTempB = dstPtrImageTempG + (batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width);

                Rpp32u increment = batch_dstSizeMax[batchCount].width - batch_dstSize[batchCount].width;

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    srcPtrImageROITemp = srcPtrImageROI;
                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        *dstPtrImageTemp = *srcPtrImageROITemp;
                        srcPtrImageROITemp++;
                        dstPtrImageTemp++;

                        *dstPtrImageTempG = *srcPtrImageROITemp;
                        srcPtrImageROITemp++;
                        dstPtrImageTempG++;

                        *dstPtrImageTempB = *srcPtrImageROITemp;
                        srcPtrImageROITemp++;
                        dstPtrImageTempB++;
                    }
                    dstPtrImageTemp += increment;
                    dstPtrImageTempG += increment;
                    dstPtrImageTempB += increment;
                    srcPtrImageROI += srcElementsInRowMax;
                }
            }
            else
            {
                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    memcpy(dstPtrImageTemp, srcPtrImageROI, elementsInRowROI * sizeof(T));
                    dstPtrImageTemp += dstElementsInRowMax;
                    srcPtrImageROI += srcElementsInRowMax;
                }
            }

        }
    }
    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus crop_host_u_f_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            T *srcPtrImage;
            U *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            U multiplier = (U) (1.0 / 255.0);

            if (outputFormatToggle == 1)
            {
                T *srcPtrImageRoiR, *srcPtrImageRoiG, *srcPtrImageRoiB;
                U *dstPtrImageTemp;

                srcPtrImageRoiR = srcPtrImage + (y1 * batch_srcSizeMax[batchCount].width) + x1;
                srcPtrImageRoiG = srcPtrImageRoiR + srcImageDimMax;
                srcPtrImageRoiB = srcPtrImageRoiG + srcImageDimMax;

                dstPtrImageTemp = dstPtrImage;

                Rpp32u incrementSrc = batch_srcSizeMax[batchCount].width - batch_dstSize[batchCount].width;
                Rpp32u incrementDst = (batch_dstSizeMax[batchCount].width - batch_dstSize[batchCount].width) * channel;

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        *dstPtrImageTemp = (U) *srcPtrImageRoiR * multiplier;
                        srcPtrImageRoiR++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = (U) *srcPtrImageRoiG * multiplier;
                        srcPtrImageRoiG++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = (U) *srcPtrImageRoiB * multiplier;
                        srcPtrImageRoiB++;
                        dstPtrImageTemp++;
                    }
                    dstPtrImageTemp += incrementDst;
                    srcPtrImageRoiR += incrementSrc;
                    srcPtrImageRoiG += incrementSrc;
                    srcPtrImageRoiB += incrementSrc;
                }
            }
            else
            {
                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannelROI, *srcPtrChannelROITemp;
                    U *dstPtrChannel, *dstPtrChannelTemp;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x1;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        dstPtrChannelTemp = dstPtrChannel;
                        srcPtrChannelROITemp = srcPtrChannelROI;

                        for (int j = 0; j < batch_dstSize[batchCount].width; j++)
                        {
                            *dstPtrChannelTemp = (U) *srcPtrChannelROITemp * multiplier;
                            dstPtrChannelTemp++;
                            srcPtrChannelROITemp++;
                        }

                        dstPtrChannel += batch_dstSizeMax[batchCount].width;
                        srcPtrChannelROI += batch_srcSizeMax[batchCount].width;
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            T *srcPtrImage;
            U *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            U multiplier = (U) (1.0 / 255.0);

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * batch_dstSize[batchCount].width;

            T *srcPtrImageROI, *srcPtrImageROITemp;
            U *dstPtrImageRow, *dstPtrImageRowTemp;
            srcPtrImageROI = srcPtrImage + (y1 * srcElementsInRowMax) + (x1 * channel);
            dstPtrImageRow = dstPtrImage;

            if (outputFormatToggle == 1)
            {
                T *srcPtrImageROITemp;

                U *dstPtrImageTempG, *dstPtrImageTempB;
                dstPtrImageTempG = dstPtrImageRow + (batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width);
                dstPtrImageTempB = dstPtrImageTempG + (batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width);

                Rpp32u increment = batch_dstSizeMax[batchCount].width - batch_dstSize[batchCount].width;

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    srcPtrImageROITemp = srcPtrImageROI;
                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        *dstPtrImageRow = (U) *srcPtrImageROITemp * multiplier;
                        srcPtrImageROITemp++;
                        dstPtrImageRow++;

                        *dstPtrImageTempG = (U) *srcPtrImageROITemp * multiplier;
                        srcPtrImageROITemp++;
                        dstPtrImageTempG++;

                        *dstPtrImageTempB = (U) *srcPtrImageROITemp * multiplier;
                        srcPtrImageROITemp++;
                        dstPtrImageTempB++;
                    }
                    dstPtrImageRow += increment;
                    dstPtrImageTempG += increment;
                    dstPtrImageTempB += increment;
                    srcPtrImageROI += srcElementsInRowMax;
                }
            }
            else
            {
                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    srcPtrImageROITemp = srcPtrImageROI;
                    dstPtrImageRowTemp = dstPtrImageRow;

                    for (int j = 0; j < elementsInRowROI; j++)
                    {
                        *dstPtrImageRowTemp = (U) *srcPtrImageROITemp * multiplier;
                        dstPtrImageRowTemp++;
                        srcPtrImageROITemp++;
                    }

                    dstPtrImageRow += dstElementsInRowMax;
                    srcPtrImageROI += srcElementsInRowMax;
                }
            }
        }
    }
    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus crop_host_u_i_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                     Rpp32u *batch_crop_pos_x, Rpp32u *batch_crop_pos_y,
                                     Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            T *srcPtrImage;
            U *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            if (outputFormatToggle == 1)
            {
                T *srcPtrImageRoiR, *srcPtrImageRoiG, *srcPtrImageRoiB;
                U *dstPtrImageTemp;

                srcPtrImageRoiR = srcPtrImage + (y1 * batch_srcSizeMax[batchCount].width) + x1;
                srcPtrImageRoiG = srcPtrImageRoiR + srcImageDimMax;
                srcPtrImageRoiB = srcPtrImageRoiG + srcImageDimMax;

                dstPtrImageTemp = dstPtrImage;

                Rpp32u incrementSrc = batch_srcSizeMax[batchCount].width - batch_dstSize[batchCount].width;
                Rpp32u incrementDst = (batch_dstSizeMax[batchCount].width - batch_dstSize[batchCount].width) * channel;

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        *dstPtrImageTemp = (U) (((Rpp32s) *srcPtrImageRoiR) - 128);
                        srcPtrImageRoiR++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = (U) (((Rpp32s) *srcPtrImageRoiG) - 128);
                        srcPtrImageRoiG++;
                        dstPtrImageTemp++;

                        *dstPtrImageTemp = (U) (((Rpp32s) *srcPtrImageRoiB) - 128);
                        srcPtrImageRoiB++;
                        dstPtrImageTemp++;
                    }
                    dstPtrImageTemp += incrementDst;
                    srcPtrImageRoiR += incrementSrc;
                    srcPtrImageRoiG += incrementSrc;
                    srcPtrImageRoiB += incrementSrc;
                }
            }
            else
            {
                for(int c = 0; c < channel; c++)
                {
                    T *srcPtrChannelROI, *srcPtrChannelROITemp;
                    U *dstPtrChannel, *dstPtrChannelTemp;
                    dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                    srcPtrChannelROI = srcPtrImage + (c * srcImageDimMax) + (y1 * batch_srcSizeMax[batchCount].width) + x1;


                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        dstPtrChannelTemp = dstPtrChannel;
                        srcPtrChannelROITemp = srcPtrChannelROI;

                        for (int j = 0; j < batch_dstSize[batchCount].width; j++)
                        {
                            *dstPtrChannelTemp = (U) (((Rpp32s) *srcPtrChannelROITemp) - 128);
                            dstPtrChannelTemp++;
                            srcPtrChannelROITemp++;
                        }

                        dstPtrChannel += batch_dstSizeMax[batchCount].width;
                        srcPtrChannelROI += batch_srcSizeMax[batchCount].width;
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u x1 = batch_crop_pos_x[batchCount];
            Rpp32u y1 = batch_crop_pos_y[batchCount];
            Rpp32u x2 = x1 + batch_dstSize[batchCount].width - 1;
            Rpp32u y2 = y1 + batch_dstSize[batchCount].height - 1;

            T *srcPtrImage;
            U *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u srcElementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * batch_dstSize[batchCount].width;

            T *srcPtrImageROI, *srcPtrImageROITemp;
            U *dstPtrImageRow, *dstPtrImageRowTemp;
            srcPtrImageROI = srcPtrImage + (y1 * srcElementsInRowMax) + (x1 * channel);
            dstPtrImageRow = dstPtrImage;

            if (outputFormatToggle == 1)
            {
                T *srcPtrImageROITemp;

                U *dstPtrImageTempG, *dstPtrImageTempB;
                dstPtrImageTempG = dstPtrImageRow + (batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width);
                dstPtrImageTempB = dstPtrImageTempG + (batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width);

                Rpp32u increment = batch_dstSizeMax[batchCount].width - batch_dstSize[batchCount].width;

                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    srcPtrImageROITemp = srcPtrImageROI;
                    for(int j = 0; j < batch_dstSize[batchCount].width; j++)
                    {
                        *dstPtrImageRow = (U) (((Rpp32s) *srcPtrImageROITemp) - 128);
                        srcPtrImageROITemp++;
                        dstPtrImageRow++;

                        *dstPtrImageTempG = (U) (((Rpp32s) *srcPtrImageROITemp) - 128);
                        srcPtrImageROITemp++;
                        dstPtrImageTempG++;

                        *dstPtrImageTempB = (U) (((Rpp32s) *srcPtrImageROITemp) - 128);
                        srcPtrImageROITemp++;
                        dstPtrImageTempB++;
                    }
                    dstPtrImageRow += increment;
                    dstPtrImageTempG += increment;
                    dstPtrImageTempB += increment;
                    srcPtrImageROI += srcElementsInRowMax;
                }
            }
            else
            {
                for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                {
                    srcPtrImageROITemp = srcPtrImageROI;
                    dstPtrImageRowTemp = dstPtrImageRow;

                    for (int j = 0; j < elementsInRowROI; j++)
                    {
                        *dstPtrImageRowTemp = (U) (((Rpp32s) *srcPtrImageROITemp) - 128);
                        dstPtrImageRowTemp++;
                        srcPtrImageROITemp++;
                    }

                    dstPtrImageRow += dstElementsInRowMax;
                    srcPtrImageROI += srcElementsInRowMax;
                }
            }
        }
    }
    return RPP_SUCCESS;
}

// /**************** resize_crop_mirror ***************/

template <typename T>
RppStatus resize_crop_mirror_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                        Rpp32u *batch_x1, Rpp32u *batch_x2, Rpp32u *batch_y1, Rpp32u *batch_y2, Rpp32u *batch_mirrorFlag,
                                        Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                                        RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u y2 = batch_y2[batchCount];
            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];

            Rpp64u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp64u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = RPPABS(y2 - y1) + 1;
            srcSizeROI.width = RPPABS(x2 - x1) + 1;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            T *dstPtrROI = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
            T *srcPtrImageTemp, *srcPtrROITemp;

            srcPtrROITemp = srcPtrROI;
            for (int c = 0; c < channel; c++)
            {
                srcPtrImageTemp = srcPtrImage + (c * srcImageDimMax) + ((Rpp32u) y1 * batch_srcSizeMax[batchCount].width) + (Rpp32u) x1;
                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    memcpy(srcPtrROITemp, srcPtrImageTemp, srcSizeROI.width * sizeof(T));
                    srcPtrImageTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrROITemp += srcSizeROI.width;
                }
            }

            if (mirrorFlag == 0)
            {
                resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);
            }
            else if (mirrorFlag == 1)
            {
                T *srcPtrROIMirrorred = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
                T *srcPtrROITemp2, *srcPtrROIMirrorredTemp;
                srcPtrROIMirrorredTemp = srcPtrROIMirrorred;
                Rpp32u bufferLength = srcSizeROI.width;
                Rpp32u alignedLength = (bufferLength / 16) * 16;

                for (int c = 0; c < channel; c++)
                {
                    srcPtrROITemp = srcPtrROI + (c * srcSizeROI.height * srcSizeROI.width) + srcSizeROI.width - 1;
                    for (int i = 0; i < srcSizeROI.height; i++)
                    {
                        srcPtrROITemp2 = srcPtrROITemp;

                        __m128i px0;
                        __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            srcPtrROITemp2 -= 15;
                            px0 = _mm_loadu_si128((__m128i *)srcPtrROITemp2);
                            px0 = _mm_shuffle_epi8(px0, vMask);
                            _mm_storeu_si128((__m128i *)srcPtrROIMirrorredTemp, px0);
                            srcPtrROITemp2 -= 1;
                            srcPtrROIMirrorredTemp += 16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *srcPtrROIMirrorredTemp++ = (T) *srcPtrROITemp2--;
                        }
                        srcPtrROITemp = srcPtrROITemp + srcSizeROI.width;
                    }
                }

                resize_kernel_host(srcPtrROIMirrorred, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

                free(srcPtrROIMirrorred);
            }

            if (outputFormatToggle == 1)
            {
                T *dstPtrROICopy = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
                compute_planar_to_packed_host(dstPtrROI, dstSize, dstPtrROICopy, channel);
                compute_padded_from_unpadded_host(dstPtrROICopy, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);
                free(dstPtrROICopy);
            }
            else
            {
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            }

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u y2 = batch_y2[batchCount];
            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = RPPABS(y2 - y1) + 1;
            srcSizeROI.width = RPPABS(x2 - x1) + 1;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * srcSizeROI.width;

            T *srcPtrROI = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
            T *dstPtrROI = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
            T *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;

            srcPtrImageTemp = srcPtrImage + ((Rpp32u) y1 * elementsInRowMax) + (channel * (Rpp32u) x1);
            for (int i = 0; i < srcSizeROI.height; i++)
            {
                memcpy(srcPtrROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(T));
                srcPtrImageTemp += elementsInRowMax;
                srcPtrROITemp += elementsInRowROI;
            }

            if (mirrorFlag == 0)
            {
                resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);
            }
            else if (mirrorFlag == 1)
            {
                T *srcPtrROIMirrorred = (T *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(T));
                T *srcPtrROIMirrorredTemp;
                srcPtrROIMirrorredTemp = srcPtrROIMirrorred;
                Rpp32u bufferLength = channel * srcSizeROI.width;
                Rpp32u alignedLength = (bufferLength / 15) * 15;

                srcPtrROITemp = srcPtrROI + (channel * (srcSizeROI.width - 1));

                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    __m128i px0;
                    __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                    {
                        srcPtrROITemp -= 13;
                        px0 = _mm_loadu_si128((__m128i *)srcPtrROITemp);
                        px0 = _mm_shuffle_epi8(px0, vMask);
                        _mm_storeu_si128((__m128i *)srcPtrROIMirrorredTemp, px0);
                        srcPtrROITemp -= 2;
                        srcPtrROIMirrorredTemp += 15;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                    {
                        memcpy(srcPtrROIMirrorredTemp, srcPtrROITemp, channel * sizeof(T));
                        srcPtrROIMirrorredTemp += channel;
                        srcPtrROITemp -= channel;
                    }

                    srcPtrROITemp = srcPtrROITemp + (channel * (2 * srcSizeROI.width));
                }

                resize_kernel_host(srcPtrROIMirrorred, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

                free(srcPtrROIMirrorred);
            }

            if (outputFormatToggle == 1)
            {
                T *dstPtrROICopy = (T *)calloc(dstSize.height * dstSize.width * channel, sizeof(T));
                compute_packed_to_planar_host(dstPtrROI, dstSize, dstPtrROICopy, channel);
                compute_padded_from_unpadded_host(dstPtrROICopy, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);
                free(dstPtrROICopy);
            }
            else
            {
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            }

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }

    return RPP_SUCCESS;
}

RppStatus resize_crop_mirror_f32_host_batch(Rpp32f* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp32f* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                        Rpp32u *batch_x1, Rpp32u *batch_x2, Rpp32u *batch_y1, Rpp32u *batch_y2, Rpp32u *batch_mirrorFlag,
                                        Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                                        RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u y2 = batch_y2[batchCount];
            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];

            Rpp64u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp64u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            Rpp32f *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = RPPABS(y2 - y1) + 1;
            srcSizeROI.width = RPPABS(x2 - x1) + 1;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            Rpp32f *srcPtrROI = (Rpp32f *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(Rpp32f));
            Rpp32f *dstPtrROI = (Rpp32f *)calloc(dstSize.height * dstSize.width * channel, sizeof(Rpp32f));
            Rpp32f *srcPtrImageTemp, *srcPtrROITemp;

            srcPtrROITemp = srcPtrROI;
            for (int c = 0; c < channel; c++)
            {
                srcPtrImageTemp = srcPtrImage + (c * srcImageDimMax) + ((Rpp32u) y1 * batch_srcSizeMax[batchCount].width) + (Rpp32u) x1;
                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    memcpy(srcPtrROITemp, srcPtrImageTemp, srcSizeROI.width * sizeof(Rpp32f));
                    srcPtrImageTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrROITemp += srcSizeROI.width;
                }
            }

            if (mirrorFlag == 0)
            {
                resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);
            }
            else if (mirrorFlag == 1)
            {
                Rpp32f *srcPtrROIMirrorred = (Rpp32f *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(Rpp32f));
                Rpp32f *srcPtrROITemp2, *srcPtrROIMirrorredTemp;
                srcPtrROIMirrorredTemp = srcPtrROIMirrorred;
                Rpp32u bufferLength = srcSizeROI.width;
                Rpp32u alignedLength = (bufferLength / 4) * 4;

                for (int c = 0; c < channel; c++)
                {
                    srcPtrROITemp = srcPtrROI + (c * srcSizeROI.height * srcSizeROI.width) + srcSizeROI.width - 1;
                    for (int i = 0; i < srcSizeROI.height; i++)
                    {
                        srcPtrROITemp2 = srcPtrROITemp;

                        __m128 p0;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                        {
                            srcPtrROITemp2 -= 3;
                            p0 = _mm_loadu_ps(srcPtrROITemp2);
                            p0 = _mm_shuffle_ps(p0, p0, _MM_SHUFFLE(0,1,2,3));
                            _mm_storeu_ps(srcPtrROIMirrorredTemp, p0);

                            srcPtrROITemp2 -= 1;
                            srcPtrROIMirrorredTemp += 4;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *srcPtrROIMirrorredTemp++ = *srcPtrROITemp2--;
                        }
                        srcPtrROITemp = srcPtrROITemp + srcSizeROI.width;
                    }
                }

                resize_kernel_host(srcPtrROIMirrorred, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

                free(srcPtrROIMirrorred);
            }

            if (outputFormatToggle == 1)
            {
                Rpp32f *dstPtrROICopy = (Rpp32f *)calloc(dstSize.height * dstSize.width * channel, sizeof(Rpp32f));
                compute_planar_to_packed_host(dstPtrROI, dstSize, dstPtrROICopy, channel);
                compute_padded_from_unpadded_host(dstPtrROICopy, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);
                free(dstPtrROICopy);
            }
            else
            {
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            }

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u y2 = batch_y2[batchCount];
            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            Rpp32f *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = RPPABS(y2 - y1) + 1;
            srcSizeROI.width = RPPABS(x2 - x1) + 1;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * srcSizeROI.width;

            Rpp32f *srcPtrROI = (Rpp32f *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(Rpp32f));
            Rpp32f *dstPtrROI = (Rpp32f *)calloc(dstSize.height * dstSize.width * channel, sizeof(Rpp32f));
            Rpp32f *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;

            srcPtrImageTemp = srcPtrImage + ((Rpp32u) y1 * elementsInRowMax) + (channel * (Rpp32u) x1);
            for (int i = 0; i < srcSizeROI.height; i++)
            {
                memcpy(srcPtrROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(Rpp32f));
                srcPtrImageTemp += elementsInRowMax;
                srcPtrROITemp += elementsInRowROI;
            }

            if (mirrorFlag == 0)
            {
                resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);
            }
            else if (mirrorFlag == 1)
            {
                Rpp32f *srcPtrROIMirrorred = (Rpp32f *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(Rpp32f));
                Rpp32f *srcPtrROIMirrorredTemp;
                srcPtrROIMirrorredTemp = srcPtrROIMirrorred;
                Rpp32u bufferLength = channel * srcSizeROI.width;

                srcPtrROITemp = srcPtrROI + (channel * (srcSizeROI.width - 1));

                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    __m128 p0;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < 3; vectorLoopCount+=3)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *srcPtrROIMirrorredTemp = *srcPtrROITemp;
                            srcPtrROIMirrorredTemp++;
                            srcPtrROITemp++;
                        }
                        srcPtrROITemp -= (2 * channel);
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        p0 = _mm_loadu_ps(srcPtrROITemp);
                        _mm_storeu_ps(srcPtrROIMirrorredTemp, p0);

                        srcPtrROITemp -= 3;
                        srcPtrROIMirrorredTemp += 3;
                    }

                    srcPtrROITemp = srcPtrROITemp + (channel * (2 * srcSizeROI.width));
                }

                resize_kernel_host(srcPtrROIMirrorred, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

                free(srcPtrROIMirrorred);
            }

            if (outputFormatToggle == 1)
            {
                Rpp32f *dstPtrROICopy = (Rpp32f *)calloc(dstSize.height * dstSize.width * channel, sizeof(Rpp32f));
                compute_packed_to_planar_host(dstPtrROI, dstSize, dstPtrROICopy, channel);
                compute_padded_from_unpadded_host(dstPtrROICopy, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);
                free(dstPtrROICopy);
            }
            else
            {
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            }

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }

    return RPP_SUCCESS;
}

RppStatus resize_crop_mirror_f16_host_batch(Rpp16f* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, Rpp16f* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                        Rpp32u *batch_x1, Rpp32u *batch_x2, Rpp32u *batch_y1, Rpp32u *batch_y2, Rpp32u *batch_mirrorFlag,
                                        Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                                        RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u y2 = batch_y2[batchCount];
            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];

            Rpp64u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp64u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            Rpp16f *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = RPPABS(y2 - y1) + 1;
            srcSizeROI.width = RPPABS(x2 - x1) + 1;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            Rpp16f *srcPtrROI = (Rpp16f *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(Rpp16f));
            Rpp16f *dstPtrROI = (Rpp16f *)calloc(dstSize.height * dstSize.width * channel, sizeof(Rpp16f));
            Rpp16f *srcPtrImageTemp, *srcPtrROITemp;

            srcPtrROITemp = srcPtrROI;
            for (int c = 0; c < channel; c++)
            {
                srcPtrImageTemp = srcPtrImage + (c * srcImageDimMax) + ((Rpp32u) y1 * batch_srcSizeMax[batchCount].width) + (Rpp32u) x1;
                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    memcpy(srcPtrROITemp, srcPtrImageTemp, srcSizeROI.width * sizeof(Rpp16f));
                    srcPtrImageTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrROITemp += srcSizeROI.width;
                }
            }

            if (mirrorFlag == 0)
            {
                resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);
            }
            else if (mirrorFlag == 1)
            {
                Rpp16f *srcPtrROIMirrorred = (Rpp16f *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(Rpp16f));
                Rpp16f *srcPtrROITemp2, *srcPtrROIMirrorredTemp;
                srcPtrROIMirrorredTemp = srcPtrROIMirrorred;
                Rpp32u bufferLength = srcSizeROI.width;
                Rpp32u alignedLength = (bufferLength / 4) * 4;

                for (int c = 0; c < channel; c++)
                {
                    srcPtrROITemp = srcPtrROI + (c * srcSizeROI.height * srcSizeROI.width) + srcSizeROI.width - 1;
                    for (int i = 0; i < srcSizeROI.height; i++)
                    {
                        srcPtrROITemp2 = srcPtrROITemp;

                        __m128 p0;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                        {
                            srcPtrROITemp2 -= 3;

                            *srcPtrROIMirrorredTemp = *(srcPtrROITemp2 + 3);
                            *(srcPtrROIMirrorredTemp + 1) = *(srcPtrROITemp2 + 2);
                            *(srcPtrROIMirrorredTemp + 2) = *(srcPtrROITemp2 + 1);
                            *(srcPtrROIMirrorredTemp + 3) = *srcPtrROITemp2;

                            srcPtrROITemp2 -= 1;
                            srcPtrROIMirrorredTemp += 4;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *srcPtrROIMirrorredTemp++ = *srcPtrROITemp2--;
                        }
                        srcPtrROITemp = srcPtrROITemp + srcSizeROI.width;
                    }
                }

                resize_kernel_host(srcPtrROIMirrorred, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

                free(srcPtrROIMirrorred);
            }

            if (outputFormatToggle == 1)
            {
                Rpp16f *dstPtrROICopy = (Rpp16f *)calloc(dstSize.height * dstSize.width * channel, sizeof(Rpp16f));
                compute_planar_to_packed_host(dstPtrROI, dstSize, dstPtrROICopy, channel);
                compute_padded_from_unpadded_host(dstPtrROICopy, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);
                free(dstPtrROICopy);
            }
            else
            {
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            }

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u x1 = batch_x1[batchCount];
            Rpp32u x2 = batch_x2[batchCount];
            Rpp32u y1 = batch_y1[batchCount];
            Rpp32u y2 = batch_y2[batchCount];
            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];

            Rpp32f hRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].height - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].height - 1)));
            Rpp32f wRatio = (((Rpp32f) (batch_dstSizeMax[batchCount].width - 1)) / ((Rpp32f) (batch_srcSizeMax[batchCount].width - 1)));

            Rpp16f *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            RppiSize srcSizeROI, dstSize;
            srcSizeROI.height = RPPABS(y2 - y1) + 1;
            srcSizeROI.width = RPPABS(x2 - x1) + 1;
            dstSize.height = batch_dstSize[batchCount].height;
            dstSize.width = batch_dstSize[batchCount].width;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowROI = channel * srcSizeROI.width;

            Rpp16f *srcPtrROI = (Rpp16f *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(Rpp16f));
            Rpp16f *dstPtrROI = (Rpp16f *)calloc(dstSize.height * dstSize.width * channel, sizeof(Rpp16f));
            Rpp16f *srcPtrImageTemp, *srcPtrROITemp;
            srcPtrROITemp = srcPtrROI;

            srcPtrImageTemp = srcPtrImage + ((Rpp32u) y1 * elementsInRowMax) + (channel * (Rpp32u) x1);
            for (int i = 0; i < srcSizeROI.height; i++)
            {
                memcpy(srcPtrROITemp, srcPtrImageTemp, elementsInRowROI * sizeof(Rpp16f));
                srcPtrImageTemp += elementsInRowMax;
                srcPtrROITemp += elementsInRowROI;
            }

            if (mirrorFlag == 0)
            {
                resize_kernel_host(srcPtrROI, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);
            }
            else if (mirrorFlag == 1)
            {
                Rpp16f *srcPtrROIMirrorred = (Rpp16f *)calloc(srcSizeROI.height * srcSizeROI.width * channel, sizeof(Rpp16f));
                Rpp16f *srcPtrROIMirrorredTemp;
                srcPtrROIMirrorredTemp = srcPtrROIMirrorred;
                Rpp32u bufferLength = channel * srcSizeROI.width;

                srcPtrROITemp = srcPtrROI + (channel * (srcSizeROI.width - 1));

                for (int i = 0; i < srcSizeROI.height; i++)
                {
                    __m128 p0;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < 3; vectorLoopCount+=3)
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            *srcPtrROIMirrorredTemp = *srcPtrROITemp;
                            srcPtrROIMirrorredTemp++;
                            srcPtrROITemp++;
                        }
                        srcPtrROITemp -= (2 * channel);
                    }

                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        memcpy(srcPtrROIMirrorredTemp, srcPtrROITemp, 3 * sizeof(Rpp16f));

                        srcPtrROITemp -= 3;
                        srcPtrROIMirrorredTemp += 3;
                    }

                    srcPtrROITemp = srcPtrROITemp + (channel * (2 * srcSizeROI.width));
                }

                resize_kernel_host(srcPtrROIMirrorred, srcSizeROI, dstPtrROI, dstSize, chnFormat, channel);

                free(srcPtrROIMirrorred);
            }

            if (outputFormatToggle == 1)
            {
                Rpp16f *dstPtrROICopy = (Rpp16f *)calloc(dstSize.height * dstSize.width * channel, sizeof(Rpp16f));
                compute_packed_to_planar_host(dstPtrROI, dstSize, dstPtrROICopy, channel);
                compute_padded_from_unpadded_host(dstPtrROICopy, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);
                free(dstPtrROICopy);
            }
            else
            {
                compute_padded_from_unpadded_host(dstPtrROI, dstSize, batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            }

            free(srcPtrROI);
            free(dstPtrROI);
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus resize_mirror_normalize_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiSize *batch_dstSize, RppiSize *batch_dstSizeMax,
                                        Rpp32f *batch_mean, Rpp32f *batch_stdDev, Rpp32u *batch_mirrorFlag,
                                        Rpp32u outputFormatToggle, Rpp32u nbatchSize,
                                        RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        T *dstPtrCopy = (T*) calloc(channel * batch_dstSizeMax[0].height * batch_dstSizeMax[0].width * nbatchSize, sizeof(T));
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = batch_mean[batchCount];
            Rpp32f stdDev = batch_stdDev[batchCount];
            Rpp32f invStdDev = 1.0 / stdDev;

            T *srcPtrImage, *dstPtrImage, *dstPtrCopyImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;
            dstPtrCopyImage = dstPtrCopy + dstLoc;

            T *srcPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
            T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
            compute_unpadded_from_padded_host(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtrImageUnpadded, chnFormat, channel);
            resize_kernel_host(srcPtrImageUnpadded, batch_srcSize[batchCount], dstPtrImageUnpadded, batch_dstSize[batchCount], chnFormat, channel);
            compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            free(srcPtrImageUnpadded);
            free(dstPtrImageUnpadded);

            if (mirrorFlag == 0)
            {
                if ((mean != 0) || (stdDev != 1))
                {
                    memcpy(dstPtrCopyImage, dstPtrImage, dstImageDimMax * channel);
                    Rpp32u dstElementsInRowMax = batch_dstSizeMax[batchCount].width;
                    Rpp32u dstElementsInRow = batch_dstSize[batchCount].width;

                    Rpp32u dstROIIncrement = dstElementsInRowMax - dstElementsInRow;
                    for(int c = 0; c < channel; c++)
                    {
                        T *dstPtrChannel, *dstPtrCopyChannel;
                        dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                        dstPtrCopyChannel = dstPtrCopyImage + (c * dstImageDimMax);


                        for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                        {
                            T *dstPtrTemp;
                            dstPtrTemp = dstPtrChannel + (i * dstElementsInRowMax);

                            Rpp32u bufferLength = dstElementsInRow;
                            Rpp32u alignedLength = (bufferLength / 16) * 16;

                            __m128i const zero = _mm_setzero_si128();
                            __m128i px0, px1, px2, px3;
                            __m128 p0, p1, p2, p3;
                            __m128 vMean = _mm_set1_ps(mean);
                            __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                            int vectorLoopCount = 0;
                            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                            {
                                px0 =  _mm_loadu_si128((__m128i *)dstPtrCopyChannel);

                                px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                                px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                                p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                                p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                                p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                                p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                                p0 = _mm_sub_ps(p0, vMean);
                                p1 = _mm_sub_ps(p1, vMean);
                                p2 = _mm_sub_ps(p2, vMean);
                                p3 = _mm_sub_ps(p3, vMean);
                                px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                                px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                                px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                                px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));

                                px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                                px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                                px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                                dstPtrCopyChannel += 16;
                                dstPtrTemp += 16;
                            }
                            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                            {
                                *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*dstPtrCopyChannel) - mean) * invStdDev);
                                dstPtrTemp++;
                                dstPtrCopyChannel++;
                            }

                            dstPtrCopyChannel += dstROIIncrement;

                        }
                    }
                }
            }
            else if (mirrorFlag == 1)
            {
                memcpy(dstPtrCopyImage, dstPtrImage, dstImageDimMax * channel);
                Rpp32u dstElementsInRowMax = batch_dstSizeMax[batchCount].width;
                Rpp32u dstElementsInRow = batch_dstSize[batchCount].width;

                if ((mean != 0) || (stdDev != 1))
                {
                    Rpp32u dstROIIncrement = dstElementsInRowMax + dstElementsInRow;
                    for(int c = 0; c < channel; c++)
                    {
                        T *dstPtrChannel, *dstPtrCopyChannel;
                        dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                        dstPtrCopyChannel = dstPtrCopyImage + (c * dstImageDimMax) + dstElementsInRow - 1;

                        for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                        {
                            T *dstPtrTemp;
                            dstPtrTemp = dstPtrChannel + (i * dstElementsInRowMax);

                            Rpp32u bufferLength = dstElementsInRow;
                            Rpp32u alignedLength = (bufferLength / 16) * 16;

                            __m128i const zero = _mm_setzero_si128();
                            __m128i px0, px1, px2, px3;
                            __m128 p0, p1, p2, p3;
                            __m128 vMean = _mm_set1_ps(mean);
                            __m128 vInvStdDev = _mm_set1_ps(invStdDev);
                            __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

                            int vectorLoopCount = 0;
                            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                            {
                                dstPtrCopyChannel -= 15;
                                px0 =  _mm_loadu_si128((__m128i *)dstPtrCopyChannel);
                                px0 = _mm_shuffle_epi8(px0, vMask);

                                px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                                px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                                p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                                p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                                p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                                p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                                p0 = _mm_sub_ps(p0, vMean);
                                p1 = _mm_sub_ps(p1, vMean);
                                p2 = _mm_sub_ps(p2, vMean);
                                p3 = _mm_sub_ps(p3, vMean);
                                px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                                px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                                px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                                px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));

                                px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                                px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                                px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                                dstPtrCopyChannel -= 1;
                                dstPtrTemp += 16;
                            }
                            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                            {
                                *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*dstPtrCopyChannel) - mean) * invStdDev);
                                dstPtrTemp++;
                                dstPtrCopyChannel--;
                            }

                            dstPtrCopyChannel += dstROIIncrement;
                        }
                    }
                }
                else if ((mean == 0) && (stdDev == 1))
                {
                    Rpp32u dstROIIncrement = dstElementsInRowMax + dstElementsInRow;
                    for(int c = 0; c < channel; c++)
                    {
                        T *dstPtrChannel, *dstPtrCopyChannel;
                        dstPtrChannel = dstPtrImage + (c * dstImageDimMax);
                        dstPtrCopyChannel = dstPtrCopyImage + (c * dstImageDimMax) + dstElementsInRow - 1;

                        for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                        {
                            T *dstPtrTemp;
                            dstPtrTemp = dstPtrChannel + (i * dstElementsInRowMax);

                            Rpp32u bufferLength = dstElementsInRow;
                            Rpp32u alignedLength = (bufferLength / 16) * 16;

                            __m128i vMask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                            __m128i px0;

                            int vectorLoopCount = 0;
                            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                            {
                                dstPtrCopyChannel -= 15;
                                px0 =  _mm_loadu_si128((__m128i *)dstPtrCopyChannel);
                                px0 = _mm_shuffle_epi8(px0, vMask);

                                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                                dstPtrCopyChannel -= 1;
                                dstPtrTemp += 16;
                            }
                            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                            {
                                *dstPtrTemp = (T) RPPPIXELCHECK(*dstPtrCopyChannel);
                                dstPtrTemp++;
                                dstPtrCopyChannel--;
                            }

                            dstPtrCopyChannel += dstROIIncrement;
                        }
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);
                compute_planar_to_packed_host(dstPtrImageUnpadded, batch_dstSize[batchCount], dstPtrImageUnpaddedCopy, channel);
                memset(dstPtrImage, (T) 0, dstImageDimMax * channel * sizeof(T));
                compute_padded_from_unpadded_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PACKED, channel);
                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }

        free(dstPtrCopy);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        T *dstPtrCopy = (T*) calloc(channel * batch_dstSizeMax[0].height * batch_dstSizeMax[0].width * nbatchSize, sizeof(T));
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u srcImageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u dstImageDimMax = batch_dstSizeMax[batchCount].height * batch_dstSizeMax[batchCount].width;

            Rpp32u mirrorFlag = batch_mirrorFlag[batchCount];
            Rpp32f mean = batch_mean[batchCount];
            Rpp32f stdDev = batch_stdDev[batchCount];
            Rpp32f invStdDev = 1.0 / stdDev;

            T *srcPtrImage, *dstPtrImage, *dstPtrCopyImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_dstSizeMax, batchCount, &dstLoc, channel);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;
            dstPtrCopyImage = dstPtrCopy + dstLoc;

            T *srcPtrImageUnpadded = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
            T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
            compute_unpadded_from_padded_host(srcPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtrImageUnpadded, chnFormat, channel);
            resize_kernel_host(srcPtrImageUnpadded, batch_srcSize[batchCount], dstPtrImageUnpadded, batch_dstSize[batchCount], chnFormat, channel);
            compute_padded_from_unpadded_host(dstPtrImageUnpadded, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, chnFormat, channel);
            free(srcPtrImageUnpadded);
            free(dstPtrImageUnpadded);

            if (mirrorFlag == 0)
            {
                if ((mean != 0) || (stdDev != 1))
                {
                    memcpy(dstPtrCopyImage, dstPtrImage, dstImageDimMax * channel);
                    Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
                    Rpp32u dstElementsInRow = channel * batch_dstSize[batchCount].width;

                    T  *dstPtrCopyImageTemp;
                    dstPtrCopyImageTemp = dstPtrCopyImage;

                    Rpp32u dstROIIncrement = dstElementsInRowMax - dstElementsInRow;

                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        T *dstPtrTemp;
                        dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                        Rpp32u bufferLength = dstElementsInRow;
                        Rpp32u alignedLength = (bufferLength / 16) * 16;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i px0, px1, px2, px3;
                        __m128 p0, p1, p2, p3;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(invStdDev);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)dstPtrCopyImageTemp);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            p0 = _mm_sub_ps(p0, vMean);
                            p1 = _mm_sub_ps(p1, vMean);
                            p2 = _mm_sub_ps(p2, vMean);
                            p3 = _mm_sub_ps(p3, vMean);
                            px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                            px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                            px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                            px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));

                            px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                            px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            dstPtrCopyImageTemp += 16;
                            dstPtrTemp += 16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*dstPtrCopyImageTemp) - mean) * invStdDev);
                            dstPtrTemp++;
                            dstPtrCopyImageTemp++;
                        }

                        dstPtrCopyImageTemp += dstROIIncrement;
                    }
                }
            }
            else if (mirrorFlag == 1)
            {
                memcpy(dstPtrCopyImage, dstPtrImage, dstImageDimMax * channel);
                Rpp32u dstElementsInRowMax = channel * batch_dstSizeMax[batchCount].width;
                Rpp32u dstElementsInRow = channel * batch_dstSize[batchCount].width;

                if ((mean != 0) || (stdDev != 1))
                {
                    T  *dstPtrCopyImageTemp;
                    dstPtrCopyImageTemp = dstPtrCopyImage + dstElementsInRow - channel;
                    Rpp32u dstROIIncrement = dstElementsInRowMax + dstElementsInRow;

                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        T *dstPtrTemp;
                        dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                        Rpp32u bufferLength = dstElementsInRow;
                        Rpp32u alignedLength = ((bufferLength / 15) * 15) - 1;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i px0, px1, px2, px3;
                        __m128 p0, p1, p2, p3;
                        __m128 vMean = _mm_set1_ps(mean);
                        __m128 vInvStdDev = _mm_set1_ps(invStdDev);
                        __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                        {
                            dstPtrCopyImageTemp -= 13;
                            px0 =  _mm_loadu_si128((__m128i *)dstPtrCopyImageTemp);
                            px0 = _mm_shuffle_epi8(px0, vMask);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            p0 = _mm_sub_ps(p0, vMean);
                            p1 = _mm_sub_ps(p1, vMean);
                            p2 = _mm_sub_ps(p2, vMean);
                            p3 = _mm_sub_ps(p3, vMean);
                            px0 = _mm_cvtps_epi32(_mm_mul_ps(p0, vInvStdDev));
                            px1 = _mm_cvtps_epi32(_mm_mul_ps(p1, vInvStdDev));
                            px2 = _mm_cvtps_epi32(_mm_mul_ps(p2, vInvStdDev));
                            px3 = _mm_cvtps_epi32(_mm_mul_ps(p3, vInvStdDev));

                            px0 = _mm_packus_epi32(px0, px1);    // pixels 0-7
                            px1 = _mm_packus_epi32(px2, px3);    // pixels 8-15
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            dstPtrCopyImageTemp -= 2;
                            dstPtrTemp += 15;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                        {
                            for(int c = 0; c < channel; c++)
                            {
                                *dstPtrTemp = (T) RPPPIXELCHECK(((Rpp32f)(*dstPtrCopyImageTemp) - mean) * invStdDev);
                                dstPtrTemp++;
                                dstPtrCopyImageTemp++;
                            }
                            dstPtrCopyImageTemp -= (2 * channel);
                        }

                        dstPtrCopyImageTemp += dstROIIncrement;
                    }
                }
                else if ((mean == 0) && (stdDev == 1))
                {
                    T  *dstPtrCopyImageTemp;
                    dstPtrCopyImageTemp = dstPtrCopyImage + dstElementsInRow - channel;
                    Rpp32u dstROIIncrement = dstElementsInRowMax + dstElementsInRow;

                    for(int i = 0; i < batch_dstSize[batchCount].height; i++)
                    {
                        T *dstPtrTemp;
                        dstPtrTemp = dstPtrImage + (i * dstElementsInRowMax);

                        Rpp32u bufferLength = dstElementsInRow;
                        Rpp32u alignedLength = ((bufferLength / 15) * 15) - 1;

                        __m128i vMask = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);
                        __m128i px0;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                        {
                            dstPtrCopyImageTemp -= 13;
                            px0 =  _mm_loadu_si128((__m128i *)dstPtrCopyImageTemp);
                            px0 = _mm_shuffle_epi8(px0, vMask);
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                            dstPtrCopyImageTemp -= 2;
                            dstPtrTemp += 15;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel)
                        {
                            for(int c = 0; c < channel; c++)
                            {
                                *dstPtrTemp = (T) RPPPIXELCHECK(*dstPtrCopyImageTemp);
                                dstPtrTemp++;
                                dstPtrCopyImageTemp++;
                            }
                            dstPtrCopyImageTemp -= (2 * channel);
                        }

                        dstPtrCopyImageTemp += dstROIIncrement;
                    }
                }
            }
            if (outputFormatToggle == 1)
            {
                T *dstPtrImageUnpadded = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                T *dstPtrImageUnpaddedCopy = (T*) calloc(channel * batch_dstSize[batchCount].height * batch_dstSize[batchCount].width, sizeof(T));
                compute_unpadded_from_padded_host(dstPtrImage, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImageUnpadded, chnFormat, channel);
                compute_packed_to_planar_host(dstPtrImageUnpadded, batch_dstSize[batchCount], dstPtrImageUnpaddedCopy, channel);
                compute_padded_from_unpadded_host(dstPtrImageUnpaddedCopy, batch_dstSize[batchCount], batch_dstSizeMax[batchCount], dstPtrImage, RPPI_CHN_PLANAR, channel);
                free(dstPtrImageUnpadded);
                free(dstPtrImageUnpaddedCopy);
            }
        }

        free(dstPtrCopy);
    }

    return RPP_SUCCESS;
}

#endif
