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
#include "rpp_cpu_simd_math.hpp"

Rpp32s BLOCK_SIZE = 8;

const Rpp32f dctCoeff1 = 1.387039845322148f;  // sqrt(2) * cos(pi / 16)
const Rpp32f dctCoeff2 = 1.306562964876377f;  // sqrt(2) * cos(pi / 8)
const Rpp32f dctCoeff3 = 1.175875602419359f;  // sqrt(2) * cos(3 * pi / 16)
const Rpp32f dctCoeff4 = 0.785694958387102f;  // sqrt(2) * cos(5 * pi / 16)
const Rpp32f dctCoeff5 = 0.541196100146197f;  // sqrt(2) * cos(3 * pi / 8)
const Rpp32f dctCoeff6 = 0.275899379282943f;  // sqrt(2) * cos(7 * pi / 16)
const Rpp32f dctNormFactor = 0.3535533905932737f; // 1 / sqrt(8)

// Coefficients for Y channel
const __m256 pCoeffYR = _mm256_set1_ps(0.299f);
const __m256 pCoeffYG = _mm256_set1_ps(0.587f);
const __m256 pCoeffYB = _mm256_set1_ps(0.114f);

// Coefficients for Cb channel
const __m256 pCoeffCbR = _mm256_set1_ps(-0.168736f);
const __m256 pCoeffCbG = _mm256_set1_ps(-0.331264f);
const __m256 pCoeffCbB = _mm256_set1_ps(0.5f);

// Coefficients for Cr channel
const __m256 pCoeffCrR = _mm256_set1_ps(0.5f);
const __m256 pCoeffCrG = _mm256_set1_ps(-0.418688f);
const __m256 pCoeffCrB = _mm256_set1_ps(-0.081312f);

// Coefficients for YCbCr to RGB conversion
const __m256 pCoeffRCr = _mm256_set1_ps(1.402f);

const __m256 pCoeffGCb = _mm256_set1_ps(-0.344136f);
const __m256 pCoeffGCr = _mm256_set1_ps(-0.714136f);

const __m256 pCoeffBCb = _mm256_set1_ps(1.772f);

const __m256i pxMask = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
const __m256 pQuarterFactor = _mm256_set1_ps(0.25f);

alignas(32) const Rpp32f chromaQuantTable[64] = {
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
};

alignas(32) const Rpp32f lumaQuantTable[64] = {
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
};

// Clamp a value between minVal and maxVal
template <typename T>
inline T clamp(T val, T minVal, T maxVal)
{
    return (val < minVal) ? minVal : (val > maxVal) ? maxVal : val;
}

// Get multiplication and addition factors based on data type
template <typename T>
constexpr std::pair<float, float> get_mul_add()
{
    if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
        return {255.0f, 0.0f};  // Scale float types by 255
    else if constexpr (std::is_same<T, Rpp8s>::value)
        return {1.0f, 128.0f};  // Offset signed 8-bit data by 128
    else
        return {1.0f, 0.0f};    // No transformation for other types
}

// Compute quality factor for JPEG quantization based on quality parameter
inline Rpp32f get_quality_factor(Rpp32s quality)
{
    Rpp32f qualityFactor;
    if (quality < 50)
        qualityFactor = 50.0f / quality;             // Higher compression
    else
        qualityFactor = 2.0f - (quality / 50.0f);    // Lower compression
    return qualityFactor;
}

// Upsample 8 values in AVX2 register by duplicating adjacent pairs
inline void upsample_avx2(__m256 input, __m256 *p)
{
    p[0] = _mm256_permutevar8x32_ps(input, _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3));
    p[1] = _mm256_permutevar8x32_ps(input, _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7));
}

// Transpose an 8x8 matrix represented by 8 __m256 rows
inline void transpose_8x8_avx(__m256* p)
{
    __m256 temp[16];

    temp[0] = _mm256_unpacklo_ps(p[0], p[2]);
    temp[1] = _mm256_unpackhi_ps(p[0], p[2]);
    temp[2] = _mm256_unpacklo_ps(p[1], p[3]);
    temp[3] = _mm256_unpackhi_ps(p[1], p[3]);
    temp[4] = _mm256_unpacklo_ps(p[4], p[6]);
    temp[5] = _mm256_unpackhi_ps(p[4], p[6]);
    temp[6] = _mm256_unpacklo_ps(p[5], p[7]);
    temp[7] = _mm256_unpackhi_ps(p[5], p[7]);

    temp[8]  = _mm256_unpacklo_ps(temp[0], temp[2]);
    temp[9]  = _mm256_unpackhi_ps(temp[0], temp[2]);
    temp[10] = _mm256_unpacklo_ps(temp[1], temp[3]);
    temp[11] = _mm256_unpackhi_ps(temp[1], temp[3]);
    temp[12] = _mm256_unpacklo_ps(temp[4], temp[6]);
    temp[13] = _mm256_unpackhi_ps(temp[4], temp[6]);
    temp[14] = _mm256_unpacklo_ps(temp[5], temp[7]);
    temp[15] = _mm256_unpackhi_ps(temp[5], temp[7]);

    // Rearrange for final transposed layout
    p[0] = _mm256_permute2f128_ps(temp[8], temp[12], (2 << 4) | 0);
    p[4] = _mm256_permute2f128_ps(temp[8], temp[12], (3 << 4) | 1);
    p[1] = _mm256_permute2f128_ps(temp[9], temp[13], (2 << 4) | 0);
    p[5] = _mm256_permute2f128_ps(temp[9], temp[13], (3 << 4) | 1);
    p[2] = _mm256_permute2f128_ps(temp[10], temp[14], (2 << 4) | 0);
    p[6] = _mm256_permute2f128_ps(temp[10], temp[14], (3 << 4) | 1);
    p[3] = _mm256_permute2f128_ps(temp[11], temp[15], (2 << 4) | 0);
    p[7] = _mm256_permute2f128_ps(temp[11], temp[15], (3 << 4) | 1);
}

// Apply quantization to an 8x8 block with a given quantization table and quality
inline void quantize_block(Rpp32f *block, const Rpp32f *quantTable, Rpp32s stride, Rpp32s qualityParam)
{
    Rpp32f qualityFactor = get_quality_factor(qualityParam);
    for (Rpp32s row = 0; row < 8; row++)
    {
        Rpp32s rowIdx = row * stride;
        Rpp32s rowQuantIdx = row * 8;
        for (Rpp32s col = 0; col < 8; col++)
        {
            Rpp32f qCoeff = quantTable[rowQuantIdx + col] * qualityFactor;
            block[rowIdx + col] = qCoeff * roundf(block[rowIdx + col] / qCoeff);
        }
    }
}

// 1D forward DCT core used for 8x8 DCT
template <typename T>
inline void dct_8x8_1d_core(T *x)
{
    T temp[8];

    // Odd part
    temp[0] = x[0] - x[7];
    temp[1] = x[6] - x[1];
    temp[2] = x[2] - x[5];
    temp[3] = x[4] - x[3];

    // Even part
    temp[4] = (x[0] + x[7]) + (x[3] + x[4]);
    temp[5] = (x[0] + x[7]) - (x[3] + x[4]);
    temp[6] = (x[1] + x[6]) + (x[2] + x[5]);
    temp[7] = (x[1] + x[6]) - (x[2] + x[5]);

    // Final output coefficients
    x[0] = dctNormFactor * (temp[4] + temp[6]);
    x[2] = dctNormFactor * ((dctCoeff2 * temp[5]) + (dctCoeff5 * temp[7]));
    x[4] = dctNormFactor * (temp[4] - temp[6]);
    x[6] = dctNormFactor * ((dctCoeff5 * temp[5]) - (dctCoeff2 * temp[7]));

    x[1] = dctNormFactor * ((dctCoeff1 * temp[0]) - (dctCoeff3 * temp[1]) + (dctCoeff4 * temp[2]) - (dctCoeff6 * temp[3]));
    x[3] = dctNormFactor * ((dctCoeff3 * temp[0]) + (dctCoeff6 * temp[1]) - (dctCoeff1 * temp[2]) + (dctCoeff4 * temp[3]));
    x[5] = dctNormFactor * ((dctCoeff4 * temp[0]) + (dctCoeff1 * temp[1]) + (dctCoeff6 * temp[2]) - (dctCoeff3 * temp[3]));
    x[7] = dctNormFactor * ((dctCoeff6 * temp[0]) + (dctCoeff4 * temp[1]) + (dctCoeff3 * temp[2]) + (dctCoeff1 * temp[3]));
}

// 1D inverse DCT core used for reconstructing 8x8 blocks
template <typename T>
inline void dct_inv_8x8_1d_core(T *x)
{
    T temp[12];

    // Even part
    temp[0] = x[0] + x[4];
    temp[1] = (dctCoeff2 * x[2]) + (dctCoeff5 * x[6]);
    temp[2] = temp[0] + temp[1];
    temp[3] = temp[0] - temp[1];
    temp[4] = (dctCoeff6 * x[7]) + (dctCoeff1 * x[1]) + (dctCoeff3 * x[3]) + (dctCoeff4 * x[5]);
    temp[5] = (dctCoeff1 * x[7]) - (dctCoeff6 * x[1]) + (dctCoeff4 * x[3]) - (dctCoeff3 * x[5]);

    // Odd part
    temp[6] = x[0] - x[4];
    temp[7] = (dctCoeff5 * x[2]) - (dctCoeff2 * x[6]);
    temp[8] = temp[6] + temp[7];
    temp[9] = temp[6] - temp[7];
    temp[10] = (dctCoeff3 * x[1]) - (dctCoeff4 * x[7]) - (dctCoeff6 * x[3]) - (dctCoeff1 * x[5]);
    temp[11] = (dctCoeff4 * x[1]) + (dctCoeff3 * x[7]) - (dctCoeff1 * x[3]) + (dctCoeff6 * x[5]);

    // Final reconstructed output
    x[0] = dctNormFactor * (temp[2] + temp[4]);
    x[7] = dctNormFactor * (temp[2] - temp[4]);
    x[4] = dctNormFactor * (temp[3] + temp[5]);
    x[3] = dctNormFactor * (temp[3] - temp[5]);

    x[1] = dctNormFactor * (temp[8] + temp[10]);
    x[5] = dctNormFactor * (temp[9] - temp[11]);
    x[2] = dctNormFactor * (temp[9] + temp[11]);
    x[6] = dctNormFactor * (temp[8] - temp[10]);
}

// Performs 1D forward DCT on an 8x8 block, using a specified stride
template <Rpp32s stride>
inline void dct_fwd_8x8_1d(Rpp32f* data)
{
    Rpp32f x[8];

    // Load data from memory into temporary array
    for (int i = 0; i < 8; i++)
        x[i] = data[i * stride];

    // Perform the core 1D DCT transformation
    dct_8x8_1d_core(x);

    // Store the result back to the original memory location
    for (int i = 0; i < 8; i++)
        data[i * stride] = x[i];
}

// AVX2-optimized 1D DCT for 8 elements
inline void dct_8x8_1d_avx2(__m256 *pVecDct)
{
    Rpp32f x[8];

    // Load from AVX register
    for (int i = 0; i < 8; i++)
        x[i] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_set1_epi32(i)));

    // Apply DCT core
    dct_8x8_1d_core(x);

    // Pack results back into AVX register
    pVecDct[0] = _mm256_setr_ps(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
}

// Performs 1D inverse DCT on an 8x8 block, using a specified stride
template <Rpp32s stride>
inline void dct_inv_8x8_1d(Rpp32f *data)
{
    Rpp32f x[8];

    // Load data
    for (int i = 0; i < 8; i++)
        x[i] = data[i * stride];

    // Apply inverse DCT
    dct_inv_8x8_1d_core(x);

    // Store results
    for (int i = 0; i < 8; i++)
        data[i * stride] = x[i];
}

// AVX2-optimized inverse DCT for 8 elements
inline void dct_inv_8x8_1d_avx2(__m256 *pVecDct)
{
    Rpp32f x[8];

    // Extract AVX values to array
    for (int i = 0; i < 8; i++)
        x[i] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_set1_epi32(i)));

    // Compute inverse DCT
    dct_inv_8x8_1d_core(x);

    // Store result into AVX register
    pVecDct[0] = _mm256_setr_ps(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
}

inline __m256 accurate_quant_round(__m256 val, __m256 quant)
{
    alignas(32) float valArr[8], quantArr[8];
    _mm256_store_ps(valArr, val);
    _mm256_store_ps(quantArr, quant);

    float result[8];
    for (int i = 0; i < 8; i++)
    {
        float quotient = valArr[i] / quantArr[i];
        float rounded = std::round(quotient);
        result[i] = rounded * quantArr[i];
    }
    return _mm256_setr_ps(result[0], result[1], result[2], result[3],
                          result[4], result[5], result[6], result[7]);
}

// Applies quantization
inline void quantize_block_avx2(__m256 *p, const float *quantTable, int qualityParam)
{
    float qualityFactor = get_quality_factor(qualityParam);
    __m256 pQualityFactor = _mm256_set1_ps(qualityFactor);

    for (int i = 0; i < 8; i++) {
        __m256 quantRow = _mm256_loadu_ps(&quantTable[i * 8]);
        quantRow = _mm256_mul_ps(quantRow, pQualityFactor);  // Scale by quality
        p[i] = accurate_quant_round(p[i], quantRow);  // Quantize
    }
}

// Converts RGB image block to YCbCr with 4:2:0 chroma subsampling
template<typename T>
inline void rgb_to_ycbcr_generic(T *srcPtr, Rpp32s rowLimit, Rpp32s colLimit, Rpp32f *y, Rpp32f *cb, Rpp32f *cr, RpptDescPtr srcDescPtr)
{
    // Split RGB pointers based on channel stride
    T *srcPtrR = srcPtr;
    T *srcPtrG = srcPtrR + srcDescPtr->strides.cStride;
    T *srcPtrB = srcPtrG + srcDescPtr->strides.cStride;

    Rpp32s wStride = srcDescPtr->strides.wStride;
    Rpp32s hStride = srcDescPtr->strides.hStride;

    Rpp32f r[256], g[256], b[256], val;

    // Luma (Y) computation with padding
    for (Rpp32s row = 0; row < 16; row++)
    {
        Rpp32s srcRowIdx = row * hStride;
        Rpp32s blockRowOffset = row * 16;
        for (Rpp32s col = 0; col < 16; col++)
        {
            Rpp32s rgbIdx = blockRowOffset + col;
            if (row < colLimit && col < rowLimit)
            {
                Rpp32s idx = srcRowIdx + col * wStride;
                auto [mul, add] = get_mul_add<T>();
                r[rgbIdx] = static_cast<Rpp32f>(srcPtrR[idx]) * mul + add;
                g[rgbIdx] = static_cast<Rpp32f>(srcPtrG[idx]) * mul + add;
                b[rgbIdx] = static_cast<Rpp32f>(srcPtrB[idx]) * mul + add;
                val = (0.299f * r[rgbIdx]) + (0.587f * g[rgbIdx]) + (0.114f * b[rgbIdx]);
                val = clamp(val, 0.0f, 255.0f);
                y[rgbIdx] = val - 128.0f;
            }
            else  // Pad by replicating nearest pixel
            {
                r[rgbIdx] = r[(std::min(row, colLimit - 1)) * 16 + std::min(col, rowLimit - 1)];
                g[rgbIdx] = g[(std::min(row, colLimit - 1)) * 16 + std::min(col, rowLimit - 1)];
                b[rgbIdx] = b[(std::min(row, colLimit - 1)) * 16 + std::min(col, rowLimit - 1)];
                y[rgbIdx] = y[(std::min(row, colLimit - 1)) * 16 + std::min(col, rowLimit - 1)];
            }
        }
    }

    // Chroma (Cb, Cr) subsampling using 4:2:0 (average over 2x2 blocks)
    for (Rpp32s row = 0; row < 16; row += 2)
    {
        for (Rpp32s col = 0; col < 16; col += 2)
        {
            Rpp32s id1 = (row * 16) + col;
            Rpp32s id2 = id1 + 1;
            Rpp32s id3 = (row + 1) * 16 + col;
            Rpp32s id4 = id3 + 1;

            Rpp32f avgR = (r[id1] + r[id2] + r[id3] + r[id4]) * 0.25f;
            Rpp32f avgG = (g[id1] + g[id2] + g[id3] + g[id4]) * 0.25f;
            Rpp32f avgB = (b[id1] + b[id2] + b[id3] + b[id4]) * 0.25f;

            // Convert to Cb/Cr and center around 0
            Rpp32s chromaIdx = (row >> 1) * 8 + (col >> 1);
            Rpp32f cbValue = (-0.168736f * avgR) - (0.331264f * avgG) + (0.5f * avgB) + 128.0f;
            cb[chromaIdx] = clamp(cbValue, 0.0f, 255.0f) - 128.0f;

            Rpp32f crValue = (0.5f * avgR) - (0.418688f * avgG) - (0.081312f * avgB) + 128.0f;
            cr[chromaIdx] = clamp(crValue, 0.0f, 255.0f) - 128.0f;
        }
    }
}

// Converts YCbCr to RGB, expanding subsampled Cb/Cr back to 16x16 using bilinear
template <typename T>
inline void ycbcr_to_rgb_generic(T *dstPtr, Rpp32s rowLimit, Rpp32s colLimit, Rpp32f *y, Rpp32f *cb, Rpp32f *cr, RpptDescPtr dstDescPtr)
{
    T *dstPtrR = dstPtr;
    T *dstPtrG = dstPtrR + dstDescPtr->strides.cStride;
    T *dstPtrB = dstPtrG + dstDescPtr->strides.cStride;

    Rpp32s hStride = dstDescPtr->strides.hStride;
    Rpp32s wStride = dstDescPtr->strides.wStride;

    // Process 8x8 chroma blocks (mapping 4:2:0 chroma to 16x16 Y pixels)
    for (Rpp32s row = 0; row < 8; row++)
    {
        Rpp32s blockRowOffset = row * 8;
        Rpp32s rowIdx = row << 1;
        for (Rpp32s col = 0; col < 8; col++)
        {
            // Get chroma values for this 2x2 block
            Rpp32s cbcrIdx = blockRowOffset + col;
            Rpp32s colIdx = col << 1;
            Rpp32f currCb = cb[cbcrIdx];
            Rpp32f currCr = cr[cbcrIdx];

            // Process 2x2 Y pixels for each Cb/Cr pair
            for (Rpp32s subRow = 0; subRow < 2; subRow++)
            {
                #pragma unroll
                for (Rpp32s subCol = 0; subCol < 2; subCol++)
                {
                    Rpp32s yIdx = (rowIdx + subRow) * 16 + (colIdx + subCol);
                    Rpp32s dstIdx = (rowIdx + subRow) * hStride + (colIdx + subCol) * wStride;

                    // Prevent out-of-bounds access
                    if ((rowIdx + subRow) >= colLimit || (colIdx + subCol) >= rowLimit)
                        continue;

                    // Convert YCbCr to RGB
                    Rpp32f yVal = y[yIdx] + 128.0f;
                    Rpp32f r = yVal + 1.402f * currCr;
                    Rpp32f g = yVal - 0.344136f * currCb - 0.714136f * currCr;
                    Rpp32f b = yVal + 1.772f * currCb;

                    r = clamp(r, 0.0f, 255.0f);
                    g = clamp(g, 0.0f, 255.0f);
                    b = clamp(b, 0.0f, 255.0f);

                    // Corrected mapping to dstPtr using strides
                    if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
                    {
                        dstPtrR[dstIdx] = static_cast<T>(r * ONE_OVER_255);
                        dstPtrG[dstIdx] = static_cast<T>(g * ONE_OVER_255);
                        dstPtrB[dstIdx] = static_cast<T>(b * ONE_OVER_255);
                    }
                    else if constexpr (std::is_same<T, Rpp8s>::value)
                    {
                        dstPtrR[dstIdx] = static_cast<T>(r - 128);
                        dstPtrG[dstIdx] = static_cast<T>(g - 128);
                        dstPtrB[dstIdx] = static_cast<T>(b - 128);
                    }
                    else
                    {
                        dstPtrR[dstIdx] = static_cast<T>(r);
                        dstPtrG[dstIdx] = static_cast<T>(g);
                        dstPtrB[dstIdx] = static_cast<T>(b);
                    }
                }
            }
        }
    }
}

template <typename T>
inline void jpeg_compression_distortion_generic(T *srcPtr, T *dstPtr, Rpp32f *scratchMem, Rpp32s rowLimit, Rpp32s colLimit, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, Rpp32s qualityParam)
{
    // Allocate memory slices for Y, Cb, and Cr channels within the scratch memory buffer
    Rpp32f *y, *cb, *cr;
    y = scratchMem;
    cb = y + 256;
    cr = cb + 64;

    // Step 1: Convert RGB to YCbCr using the provided input buffer
    rgb_to_ycbcr_generic(srcPtr, rowLimit, colLimit, y, cb, cr, srcDescPtr);

    // Step 2: Process 16x16 Y channel block as four 8x8 DCT blocks
    for (Rpp32s blockRow = 0; blockRow < 16; blockRow += 8)
    {
        for (Rpp32s blockCol = 0; blockCol < 16; blockCol += 8)
        {
            // Extract pointer to current 8x8 block
            Rpp32f *block = y + blockRow * 16 + blockCol;

            // Apply row-wise 1D forward DCT
            for(Rpp32s row = 0; row < 128; row += 16)
                dct_fwd_8x8_1d<1>(block + row);  

            // Apply column-wise 1D forward DCT
            for(Rpp32s row = 0; row < 8; row++)
                dct_fwd_8x8_1d<16>(block + row);

            // Quantize the DCT coefficients using luminance table
            quantize_block(block, lumaQuantTable, 16, qualityParam);

            // Apply inverse column-wise and row-wise DCTs
            for(Rpp32s row = 0; row < 8; row++)
                dct_inv_8x8_1d<16>(block + row);
            for(Rpp32s row = 0; row < 128; row += 16)
                dct_inv_8x8_1d<1>(block + row);
        }
    }

    // Step 3: Perform 8x8 DCT/IDCT + quantization on Cb and Cr channels
    // ----------- Cb Channel -----------
    for(Rpp32s row = 0; row < 64; row += 8)
        dct_fwd_8x8_1d<1>(cb + row);
    for(Rpp32s row = 0; row < 8; row++)
        dct_fwd_8x8_1d<8>(cb + row);
    quantize_block(cb, chromaQuantTable, 8, qualityParam);
    for(Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d<8>(cb + row);
    for(Rpp32s row = 0; row < 64; row += 8)
        dct_inv_8x8_1d<1>(cb + row);

    // ----------- Cr Channel -----------
    for(Rpp32s row = 0; row < 64; row += 8)
        dct_fwd_8x8_1d<1>(cr + row);
    for(Rpp32s row = 0; row < 8; row++)
        dct_fwd_8x8_1d<8>(cr + row);
    quantize_block(cr, chromaQuantTable, 8, qualityParam);
    for(Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d<8>(cr + row);
    for(Rpp32s row = 0; row < 64; row += 8)
        dct_inv_8x8_1d<1>(cr + row);

    // Step 4: Convert YCbCr back to RGB and write to output buffer
    ycbcr_to_rgb_generic(dstPtr, rowLimit, colLimit, y, cb, cr, dstDescPtr);
}

template <typename T>
inline void jpeg_compression_distortion_pln1_generic(T *srcPtr, T *dstPtr, Rpp32f *scratchMem, Rpp32s rowLimit, Rpp32s colLimit, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, Rpp32s qualityParam)
{
    Rpp32f blockData[64];

    // Step 1: Load 8x8 pixel block into blockData with boundary checks
    for (Rpp32s row = 0; row < 8; row++)
    {
        Rpp32s rowIdx = row * 8;
        Rpp32s srcIdx = row * srcDescPtr->strides.hStride;
        for (Rpp32s col = 0; col < 8; col++)
        {
            Rpp32s blockIdx = rowIdx + col;
            if (row < colLimit && col < rowLimit)
            {
                Rpp32s idx = srcIdx + col;
                if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
                    blockData[blockIdx] = (static_cast<Rpp32f>(srcPtr[idx]) * 255.0f) - 128.0f;
                else if constexpr (std::is_same<T, Rpp8s>::value)
                    blockData[blockIdx] = static_cast<Rpp32f>(srcPtr[idx]);
                else
                    blockData[blockIdx] = static_cast<Rpp32f>(srcPtr[idx]) - 128.0f;
            }
            else
            {
                // Pad edge values if out of bounds
                blockData[blockIdx] = blockData[(std::min(row, colLimit - 1)) * 8 + std::min(col, rowLimit - 1)];
            }
        }
    }

    // Step 2: DCT -> Quantization -> IDCT
    for (Rpp32s row = 0; row < 64; row += 8)
        dct_fwd_8x8_1d<1>(blockData + row);  
    for (Rpp32s row = 0; row < 8; row++)
        dct_fwd_8x8_1d<8>(blockData + row);  
    quantize_block(blockData, lumaQuantTable, 8, qualityParam);
    for (Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d<8>(blockData + row);  
    for (Rpp32s row = 0; row < 64; row += 8)
        dct_inv_8x8_1d<1>(blockData + row);  

    // Step 3: Store processed block back to dstPtr with format conversion
    for (Rpp32s row = 0; row < colLimit; row++)
    {
        Rpp32s rowIdx = row * 8;
        Rpp32s rowDstIdx = row * dstDescPtr->strides.hStride;
        for (Rpp32s col = 0; col < rowLimit; col++)
        {
            Rpp32s idx = rowDstIdx + col;
            Rpp32f value = blockData[rowIdx + col] + 128.0f;
            value = clamp(value, 0.0f, 255.0f);

            if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
                dstPtr[idx] = static_cast<T>(value * ONE_OVER_255);
            else if constexpr (std::is_same<T, Rpp8s>::value)
                dstPtr[idx] = static_cast<T>(value - 128.0f);
            else
                dstPtr[idx] = static_cast<T>(value);
        }
    }
}

// convert rgb to ycbcr buffer
inline void rgb_to_ycbcr_subsampled(__m256 *pRgb, __m256 *pY, __m256 *pCb, __m256 *pCr)
{
    __m256 pRavg[16], pGavg[16], pBavg[16];
    for(int idxY = 0, idxRGB = 0; idxY < 16; idxY++, idxRGB += 6)
    {
        // Compute Y
        pY[idxY] = _mm256_fmadd_ps(pRgb[idxRGB + 4], pCoeffYB, _mm256_fmadd_ps(pRgb[idxRGB + 2], pCoeffYG, _mm256_mul_ps(pRgb[idxRGB], pCoeffYR)));
        pY[idxY + 16] = _mm256_fmadd_ps(pRgb[idxRGB + 5], pCoeffYB, _mm256_fmadd_ps(pRgb[idxRGB + 3], pCoeffYG, _mm256_mul_ps(pRgb[idxRGB + 1], pCoeffYR)));

        pY[idxY] = _mm256_max_ps(avx_p0, _mm256_min_ps(pY[idxY], avx_p255));
        pY[idxY + 16] = _mm256_max_ps(avx_p0, _mm256_min_ps(pY[idxY + 16], avx_p255));
        pY[idxY] =_mm256_sub_ps(pY[idxY], avx_p128);
        pY[idxY + 16] =_mm256_sub_ps(pY[idxY + 16], avx_p128);

        pRavg[idxY] = _mm256_hadd_ps(pRgb[idxRGB], pRgb[idxRGB + 1]);
        pGavg[idxY] = _mm256_hadd_ps(pRgb[idxRGB + 2], pRgb[idxRGB + 3]);
        pBavg[idxY] = _mm256_hadd_ps(pRgb[idxRGB + 4], pRgb[idxRGB + 5]);

        pRavg[idxY] = _mm256_permutevar8x32_ps(pRavg[idxY], pxMask);
        pGavg[idxY] = _mm256_permutevar8x32_ps(pGavg[idxY], pxMask);
        pBavg[idxY] = _mm256_permutevar8x32_ps(pBavg[idxY], pxMask);
    }
    for (Rpp32s i = 0; i < 16; i += 2)
    {
        Rpp32s chromaIdx = i >> 1;
        pRavg[i] = _mm256_mul_ps(_mm256_add_ps(pRavg[i], pRavg[i + 1]), pQuarterFactor);
        pGavg[i] = _mm256_mul_ps(_mm256_add_ps(pGavg[i], pGavg[i + 1]), pQuarterFactor);
        pBavg[i] = _mm256_mul_ps(_mm256_add_ps(pBavg[i], pBavg[i + 1]), pQuarterFactor);
        pCb[chromaIdx] = _mm256_fmadd_ps(pBavg[i], pCoeffCbB, _mm256_fmadd_ps(pGavg[i], pCoeffCbG, _mm256_fmadd_ps(pRavg[i], pCoeffCbR, avx_p128)));
        pCr[chromaIdx] = _mm256_fmadd_ps(pBavg[i], pCoeffCrB, _mm256_fmadd_ps(pGavg[i], pCoeffCrG, _mm256_fmadd_ps(pRavg[i], pCoeffCrR, avx_p128)));
        pCb[chromaIdx] = _mm256_max_ps(avx_p0, _mm256_min_ps(pCb[chromaIdx], avx_p255));
        pCr[chromaIdx] = _mm256_max_ps(avx_p0, _mm256_min_ps(pCr[chromaIdx], avx_p255));
        pCb[chromaIdx] = _mm256_sub_ps(pCb[chromaIdx], avx_p128);
        pCr[chromaIdx] = _mm256_sub_ps(pCr[chromaIdx], avx_p128);
    }
}

// convert ycbcr buffer to rgb buffer
inline void ycbcr_to_rgb_subsampled(__m256* pY, __m256* pCb, __m256* pCr, __m256* pRgb)
{
    for (Rpp32s i = 0; i < 8; i++)
    {
        __m256 cb[2];
        upsample_avx2(pCb[i], &cb[0]);
        __m256 cr[2];
        upsample_avx2(pCr[i], &cr[0]);
        for(Rpp32s j = 0; j < 4; j++)
        {
            Rpp32s idxY = (i << 1) + (j >> 1) + ((j & 1) << 4);
            Rpp32s idxRGB = i * 12 + ((j >> 1) * 6) + (j & 1);
            __m256 curCb = cb[j % 2];
            __m256 curCr = cr[j % 2];

            pY[idxY] = _mm256_add_ps(pY[idxY], avx_p128);
            pY[idxY] = _mm256_min_ps(_mm256_max_ps(pY[idxY], avx_p0), avx_p255);
            __m256 pR = _mm256_fmadd_ps(pCoeffRCr, curCr, pY[idxY]);

            __m256 pG = _mm256_fmadd_ps(pCoeffGCr, curCr,
                            _mm256_fmadd_ps(pCoeffGCb, curCb, pY[idxY]));

            __m256 pB = _mm256_fmadd_ps(pCoeffBCb, curCb, pY[idxY]);

            pR = _mm256_max_ps(avx_p0, _mm256_min_ps(pR, avx_p255));
            pG = _mm256_max_ps(avx_p0, _mm256_min_ps(pG, avx_p255));
            pB = _mm256_max_ps(avx_p0, _mm256_min_ps(pB, avx_p255));

            // Store interleaved R, G, B
            pRgb[idxRGB] = pR;
            pRgb[idxRGB + 2] = pG;
            pRgb[idxRGB + 4] = pB;
        }
    }
}

// process JPEG compression by applying DCT, quantization, and inverse DCT to YCbCr image data.
inline void process_jpeg_compression_distortion(__m256* pRgb, __m256* pY, __m256* pCb, __m256* pCr,
                                                const Rpp32f *lumaQuantTable, const Rpp32f *chromaQuantTable, Rpp32s qualityParam)
{
    rgb_to_ycbcr_subsampled(pRgb, pY, pCb, pCr);
    for (Rpp32s idxY = 0; idxY < 32; idxY += 8)
    {
        for (Rpp32s row = 0; row < 8; row++)
            dct_8x8_1d_avx2(&pY[idxY + row]);
        transpose_8x8_avx(&pY[idxY]);
        for (Rpp32s row = 0; row < 8; row++)
            dct_8x8_1d_avx2(&pY[idxY + row]);
        transpose_8x8_avx(&pY[idxY]);

        quantize_block_avx2(&pY[idxY], lumaQuantTable, qualityParam);

        transpose_8x8_avx(&pY[idxY]);
        for (Rpp32s row = 0; row < 8; row++)
            dct_inv_8x8_1d_avx2(&pY[idxY + row]);
        transpose_8x8_avx(&pY[idxY]);
        for (Rpp32s row = 0; row < 8; row++)
            dct_inv_8x8_1d_avx2(&pY[idxY + row]);
    }

    for (Rpp32s row = 0; row < 8; row++)
        dct_8x8_1d_avx2(&pCb[row]);
    transpose_8x8_avx(&pCb[0]);
    for (Rpp32s row = 0; row < 8; row++)
        dct_8x8_1d_avx2(&pCb[row]);
    transpose_8x8_avx(&pCb[0]);

    quantize_block_avx2(pCb, chromaQuantTable, qualityParam);

    transpose_8x8_avx(&pCb[0]);
    for (Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d_avx2(&pCb[row]);
    transpose_8x8_avx(&pCb[0]);
    for (Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d_avx2(&pCb[row]);

    for (Rpp32s row = 0; row < 8; row++)
        dct_8x8_1d_avx2(&pCr[row]);
    transpose_8x8_avx(&pCr[0]);
    for (Rpp32s row = 0; row < 8; row++)
        dct_8x8_1d_avx2(&pCr[row]);
    transpose_8x8_avx(&pCr[0]);

    quantize_block_avx2(pCr, chromaQuantTable, qualityParam);

    transpose_8x8_avx(&pCr[0]);
    for (Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d_avx2(&pCr[row]);
    transpose_8x8_avx(&pCr[0]);
    for (Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d_avx2(&pCr[row]);

    ycbcr_to_rgb_subsampled(pY, pCb, pCr, pRgb);
}

RppStatus jpeg_compression_distortion_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                        RpptDescPtr srcDescPtr,
                                                        Rpp8u *dstPtr,
                                                        RpptDescPtr dstDescPtr,
                                                        Rpp32s *qualityTensor,
                                                        RpptROIPtr roiTensorPtrSrc,
                                                        RpptRoiType roiType,
                                                        RppLayoutParams layoutParams,
                                                        rpp::Handle& handle)
{
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for(Rpp32s batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 16) * 16;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32s qualityParam = qualityTensor[batchCount];
        Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + (batchCount * (16 * 16 * 3));  // (16 * 16) is the block size, and 3 represents the number of channels

        Rpp32s srcIncrement = 16 * srcDescPtr->strides.hStride;
        Rpp32s dstIncrement = 16 * dstDescPtr->strides.hStride;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8u *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8u *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8u *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8u *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
        else if((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            srcIncrement = 8 * srcDescPtr->strides.hStride;
            dstIncrement = 8 * dstDescPtr->strides.hStride;
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 8)
            {
                Rpp32s colLimit = ((i + 8) < roi.xywhROI.roiHeight) ? 8 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
#if __AVX2__
                    __m256 p[8];
                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        Rpp8u *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrTempRow, &p[row]);                                 // simd loads
                            p[row] = _mm256_sub_ps(p[row], avx_p128);
                        }
                        else
                            p[row] = p[colLimit - 1];
                    }

                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    quantize_block_avx2(p, lumaQuantTable, qualityParam);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);

                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        p[row] = _mm256_add_ps(p[row], avx_p128);
                        p[row] = _mm256_max_ps(avx_p0, _mm256_min_ps(p[row], avx_p255));
                        Rpp8u *dstPtrTempRow;
                        dstPtrTempRow = dstPtrTemp + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store8_f32pln1_to_u8pln1_avx, dstPtrTempRow, p[row]);                                 // simd loads
                    }
#endif
                    dstPtrTemp += 8;
                    srcPtrTemp += 8;
                }
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += 8)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 8) < roi.xywhROI.roiWidth) ? 8 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_pln1_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 8)
                    {
                        dstPtrTemp += 8;
                        srcPtrTemp += 8;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus jpeg_compression_distortion_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                          RpptDescPtr srcDescPtr,
                                                          Rpp32f *dstPtr,
                                                          RpptDescPtr dstDescPtr,
                                                          Rpp32s *qualityTensor,
                                                          RpptROIPtr roiTensorPtrSrc,
                                                          RpptRoiType roiType,
                                                          RppLayoutParams layoutParams,
                                                          rpp::Handle& handle)
{
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for(Rpp32s batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 16) * 16;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32s qualityParam = qualityTensor[batchCount];
        Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + (batchCount * (16 * 16 * 3));  // (16 * 16) is the block size, and 3 represents the number of channels
        Rpp32s srcIncrement = 16 * srcDescPtr->strides.hStride;
        Rpp32s dstIncrement = 16 * dstDescPtr->strides.hStride;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp32f *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load48_f32pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p255);  // Scale up
                        }
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp32f *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p1op255);  // Scale down
                            rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                        }
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp32f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load48_f32pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p255);  // Scale up
                        }
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp32f *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p1op255);  // Scale down
                            rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);
                        }
                    }
                    dstPtrTemp += 48;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTemp += 48;
                    }
                }
                    srcPtrRow += srcIncrement;
                    dstPtrRow += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp32f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load48_f32pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p255);  // Scale up
                        }
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp32f *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p1op255);  // Scale down
                            rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);
                        }
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp32f *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load48_f32pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p255);  // Scale up
                        }
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp32f *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p1op255);  // Scale down
                            rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);
                        }
                    }
                    dstPtrTemp += 48;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
        else if((srcDescPtr->c == 1 ) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            srcIncrement = 8 * srcDescPtr->strides.hStride;
            dstIncrement = 8 * dstDescPtr->strides.hStride;
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 8)
            {
                Rpp32s colLimit = ((i + 8) < roi.xywhROI.roiHeight) ? 8 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
#if __AVX2__
                    __m256 p[8];
                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        Rpp32f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTempRow, &p[row]);                                 // simd loads
                            p[row] = _mm256_mul_ps(p[row], avx_p255);
                            p[row] = _mm256_sub_ps(p[row], avx_p128);
                        }
                        else
                            p[row] = p[colLimit - 1];
                    }

                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    quantize_block_avx2(p, lumaQuantTable, qualityParam);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);

                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        p[row] = _mm256_add_ps(p[row], avx_p128);
                        p[row] = _mm256_max_ps(avx_p0, _mm256_min_ps(p[row], avx_p255));
                        Rpp32f *dstPtrTempRow;
                        dstPtrTempRow = dstPtrTemp + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            p[row] = _mm256_mul_ps(p[row], avx_p1op255);
                            rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTempRow, p[row]);                                 // simd loads
                        }
                    }
#endif
                    dstPtrTemp += 8;
                    srcPtrTemp += 8;
                }
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += 8)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 8) < roi.xywhROI.roiWidth) ? 8 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_pln1_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 8)
                    {
                        dstPtrTemp += 8;
                        srcPtrTemp += 8;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus jpeg_compression_distortion_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                          RpptDescPtr srcDescPtr,
                                                          Rpp16f *dstPtr,
                                                          RpptDescPtr dstDescPtr,
                                                          Rpp32s *qualityTensor,
                                                          RpptROIPtr roiTensorPtrSrc,
                                                          RpptRoiType roiType,
                                                          RppLayoutParams layoutParams,
                                                          rpp::Handle& handle)
{
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for(Rpp32s batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 16) * 16;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32s qualityParam = qualityTensor[batchCount];
        Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + (batchCount * (16 * 16 * 3));  // (16 * 16) is the block size, and 3 represents the number of channels
        Rpp32s srcIncrement = 16 * srcDescPtr->strides.hStride;
        Rpp32s dstIncrement = 16 * dstDescPtr->strides.hStride;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load48_f16pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p255);  // Scale up
                        }
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p1op255);  // Scale down
                            rpp_simd_store(rpp_store48_f32pln3_to_f16pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                        }
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load48_f16pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p255);  // Scale up
                        }
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p1op255);  // Scale down
                            rpp_simd_store(rpp_store48_f32pln3_to_f16pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                        }
                    }
                    dstPtrTemp += 48;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTemp += 48;
                    }
                }
                    srcPtrRow += srcIncrement;
                    dstPtrRow += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load48_f16pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p255);  // Scale up
                        }
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p1op255);  // Scale down
                            rpp_simd_store(rpp_store48_f32pln3_to_f16pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                        }
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load48_f16pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p255);  // Scale up
                        }
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = _mm256_mul_ps(pRgb[row * 6 + col], avx_p1op255);  // Scale down
                            rpp_simd_store(rpp_store48_f32pln3_to_f16pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                        }
                    }
                    dstPtrTemp += 48;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
        else if((srcDescPtr->c == 1 ) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            srcIncrement = 8 * srcDescPtr->strides.hStride;
            dstIncrement = 8 * dstDescPtr->strides.hStride;
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 8)
            {
                Rpp32s colLimit = ((i + 8) < roi.xywhROI.roiHeight) ? 8 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
#if __AVX2__
                    __m256 p[8];
                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        Rpp16f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtrTempRow, &p[row]);                                 // simd loads
                            p[row] = _mm256_mul_ps(p[row], avx_p255);
                            p[row] = _mm256_sub_ps(p[row], avx_p128);
                        }
                        else
                            p[row] = p[colLimit - 1];
                    }

                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    quantize_block_avx2(p, lumaQuantTable, qualityParam);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);

                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        p[row] = _mm256_add_ps(p[row], avx_p128);
                        p[row] = _mm256_max_ps(avx_p0, _mm256_min_ps(p[row], avx_p255));
                        Rpp16f *dstPtrTempRow;
                        dstPtrTempRow = dstPtrTemp + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            p[row] = _mm256_mul_ps(p[row], avx_p1op255);
                            rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTempRow, &p[row]);                                 // simd loads
                        }
                    }
#endif
                    dstPtrTemp += 8;
                    srcPtrTemp += 8;
                }
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += 8)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 8) < roi.xywhROI.roiWidth) ? 8 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_pln1_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 8)
                    {
                        dstPtrTemp += 8;
                        srcPtrTemp += 8;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus jpeg_compression_distortion_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                        RpptDescPtr srcDescPtr,
                                                        Rpp8s *dstPtr,
                                                        RpptDescPtr dstDescPtr,
                                                        Rpp32s *qualityTensor,
                                                        RpptROIPtr roiTensorPtrSrc,
                                                        RpptRoiType roiType,
                                                        RppLayoutParams layoutParams,
                                                        rpp::Handle& handle)
{
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(1)
    for(Rpp32s batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 16) * 16;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32s qualityParam = qualityTensor[batchCount];
        Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + (batchCount * (16 * 16 * 3));  // (16 * 16) is the block size, and 3 represents the number of channels
        Rpp32s srcIncrement = 16 * srcDescPtr->strides.hStride;
        Rpp32s dstIncrement = 16 * dstDescPtr->strides.hStride;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8s *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8s *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8s *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8s *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8s *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8s *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8s *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                        else
                        {
                            for (int col = 0; col < 6; col++)
                                pRgb[row * 6 + col] = pRgb[(colLimit - 1) * 6 + col];
                        }
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8s *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
        else if((srcDescPtr->c == 1 ) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            srcIncrement = 8 * srcDescPtr->strides.hStride;
            dstIncrement = 8 * dstDescPtr->strides.hStride;
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 8)
            {
                Rpp32s colLimit = ((i + 8) < roi.xywhROI.roiHeight) ? 8 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
#if __AVX2__
                    __m256 p[8];
                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        Rpp8s *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load8_i8_to_f32_avx, srcPtrTempRow, &p[row]);                                // simd loads
                            p[row] = _mm256_sub_ps(p[row], avx_p128);
                        }
                        else
                            p[row] = p[colLimit - 1];
                    }

                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    quantize_block_avx2(p, lumaQuantTable, qualityParam);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);

                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        p[row] = _mm256_add_ps(p[row], avx_p128);
                        p[row] = _mm256_max_ps(avx_p0, _mm256_min_ps(p[row], avx_p255));
                        Rpp8s *dstPtrTempRow;
                        dstPtrTempRow = dstPtrTemp + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store8_f32pln1_to_i8pln1_avx, dstPtrTempRow, p[row]);                                 // simd loads
                    }
#endif
                    dstPtrTemp += 8;
                    srcPtrTemp += 8;
                }
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += 8)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 8) < roi.xywhROI.roiWidth) ? 8 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_pln1_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 8)
                    {
                        dstPtrTemp += 8;
                        srcPtrTemp += 8;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
    }
    return RPP_SUCCESS;
}
