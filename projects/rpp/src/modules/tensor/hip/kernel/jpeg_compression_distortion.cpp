/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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

// JPEG standard quantization tables
static constexpr float lumaQuantTable[64] = {
    16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};

static constexpr float chromaQuantTable[64] = {
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99,
    99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};

// DCT Constants
__device__ const float normCoeff = 0.3535533905932737f;
__device__ const float4 dctACDF = {1.387039845322148f, 1.175875602419359f, 0.785694958387102f,
                                   0.275899379282943f};
__device__ const float2 dctBE = {1.306562964876377f, 0.541196100146197f};

__device__ const float4 yR_f4 = {0.299f, 0.299f, 0.299f, 0.299f};
__device__ const float4 yG_f4 = {0.587f, 0.587f, 0.587f, 0.587f};
__device__ const float4 yB_f4 = {0.114f, 0.114f, 0.114f, 0.114f};
__device__ const float4 maxVal255_f4 = {255.0f, 255.0f, 255.0f, 255.0f};
__device__ const float4 maxVal128_f4 = {128.0f, 128.0f, 128.0f, 128.0f};

// Clamping for signed char (schar)
__device__ inline void clamp_range(schar* src, float* values) {
    for (int j = 0; j < 8; j++) values[j] = __builtin_amdgcn_fmed3f(-128, values[j], 127);
}

// Clamping for float
__device__ inline void clamp_range(float* src, float* values) {
    for (int j = 0; j < 8; j++) values[j] = __builtin_amdgcn_fmed3f(0.0f, values[j], 1.0f);
}

// Clamping for half
__device__ inline void clamp_range(half* src, float* values) {
    for (int j = 0; j < 8; j++) values[j] = __builtin_amdgcn_fmed3f(0.0f, values[j], 1.0f);
}

// Clamping for unsigned char (uchar)
__device__ inline void clamp_range(uchar* src, float* values) {
    for (int j = 0; j < 8; j++) values[j] = __builtin_amdgcn_fmed3f(0, values[j], 255);
}

// Generic clamping when no specific type is provided (default to 0-255)
__device__ inline void clamp_range(float* values) {
    for (int j = 0; j < 8; j++) values[j] = __builtin_amdgcn_fmed3f(0.0f, values[j], 255.0f);
}

// Computing Y from R G B
__device__ inline void y_hip_compute(float* src, d_float24* rgb_f24, d_float8* y_f8) {
    y_f8->f4[0] = (rgb_f24[0].f8[0].f4[0] * yR_f4) + (rgb_f24[0].f8[1].f4[0] * yG_f4) +
                  (rgb_f24[0].f8[2].f4[0] * yB_f4);
    y_f8->f4[1] = (rgb_f24[0].f8[0].f4[1] * yR_f4) + (rgb_f24[0].f8[1].f4[1] * yG_f4) +
                  (rgb_f24[0].f8[2].f4[1] * yB_f4);
}

__device__ inline void y_hip_compute(half* src, d_float24* rgb_f24, d_float8* y_f8) {
    y_f8->f4[0] = (rgb_f24[0].f8[0].f4[0] * yR_f4) + (rgb_f24[0].f8[1].f4[0] * yG_f4) +
                  (rgb_f24[0].f8[2].f4[0] * yB_f4);
    y_f8->f4[1] = (rgb_f24[0].f8[0].f4[1] * yR_f4) + (rgb_f24[0].f8[1].f4[1] * yG_f4) +
                  (rgb_f24[0].f8[2].f4[1] * yB_f4);
}

__device__ inline void y_hip_compute(schar* src, d_float24* rgb_f24, d_float8* y_f8) {
    y_f8->f4[0] = (rgb_f24[0].f8[0].f4[0] * yR_f4) + (rgb_f24[0].f8[1].f4[0] * yG_f4) +
                  (rgb_f24[0].f8[2].f4[0] * yB_f4);
    y_f8->f4[1] = (rgb_f24[0].f8[0].f4[1] * yR_f4) + (rgb_f24[0].f8[1].f4[1] * yG_f4) +
                  (rgb_f24[0].f8[2].f4[1] * yB_f4);
}

__device__ inline void y_hip_compute(uchar* src, d_float24* rgb_f24, d_float8* y_f8) {
    // RGB values are already in 0-255 range
    y_f8->f4[0] = (rgb_f24[0].f8[0].f4[0] * yR_f4) + (rgb_f24[0].f8[1].f4[0] * yG_f4) +
                  (rgb_f24[0].f8[2].f4[0] * yB_f4);
    y_f8->f4[1] = (rgb_f24[0].f8[0].f4[1] * yR_f4) + (rgb_f24[0].f8[1].f4[1] * yG_f4) +
                  (rgb_f24[0].f8[2].f4[1] * yB_f4);
}

// Downsampling and computing Cb and Cr
__device__ inline void downsample_cbcr_hip_compute(d_float8* r1_f8, d_float8* r2_f8,
                                                   d_float8* g1_f8, d_float8* g2_f8,
                                                   d_float8* b1_f8, d_float8* b2_f8, float4* cb_f4,
                                                   float4* cr_f4) {
    // Vertical downsampling
    d_float8 avgR_f8, avgG_f8, avgB_f8;
    avgR_f8.f4[0] = (r1_f8->f4[0] + r2_f8->f4[0]);
    avgR_f8.f4[1] = (r1_f8->f4[1] + r2_f8->f4[1]);
    avgG_f8.f4[0] = (g1_f8->f4[0] + g2_f8->f4[0]);
    avgG_f8.f4[1] = (g1_f8->f4[1] + g2_f8->f4[1]);
    avgB_f8.f4[0] = (b1_f8->f4[0] + b2_f8->f4[0]);
    avgB_f8.f4[1] = (b1_f8->f4[1] + b2_f8->f4[1]);

    // Horizontal downsampling
    float4 avgR_f4 =
        make_float4((avgR_f8.f4[0].x + avgR_f8.f4[0].y), (avgR_f8.f4[0].z + avgR_f8.f4[0].w),
                    (avgR_f8.f4[1].x + avgR_f8.f4[1].y), (avgR_f8.f4[1].z + avgR_f8.f4[1].w));

    float4 avgG_f4 =
        make_float4((avgG_f8.f4[0].x + avgG_f8.f4[0].y), (avgG_f8.f4[0].z + avgG_f8.f4[0].w),
                    (avgG_f8.f4[1].x + avgG_f8.f4[1].y), (avgG_f8.f4[1].z + avgG_f8.f4[1].w));

    float4 avgB_f4 =
        make_float4((avgB_f8.f4[0].x + avgB_f8.f4[0].y), (avgB_f8.f4[0].z + avgB_f8.f4[0].w),
                    (avgB_f8.f4[1].x + avgB_f8.f4[1].y), (avgB_f8.f4[1].z + avgB_f8.f4[1].w));

    *cb_f4 = (avgR_f4 * MAKE_FLOAT4(-0.042184f)) + (avgG_f4 * MAKE_FLOAT4(-0.082816f)) +
             (avgB_f4 * MAKE_FLOAT4(0.125000f)) + maxVal128_f4;
    *cr_f4 = (avgR_f4 * MAKE_FLOAT4(0.125000f)) + (avgG_f4 * MAKE_FLOAT4(-0.104672f)) +
             (avgB_f4 * MAKE_FLOAT4(-0.020328f)) + maxVal128_f4;
}

// DCT forward 1D implementation
__device__ inline void dct_fwd_8x8_1d(float* vec, bool offset128) {
    int val = (-128.0f * offset128);
    float4 val4 = MAKE_FLOAT4((float)val);

    // Load data into float4 vectors
    float4 vec1_f4 = *(float4*)&vec[0];
    float4 vec2_f4 = *(float4*)&vec[4];

    // Perform the vectorized addition
    float4 inp1_f4 = vec1_f4 + val4;
    float4 inp2_f4 = vec2_f4 + val4;

    float4 temp1_f4 = make_float4(inp1_f4.x + inp2_f4.w, inp1_f4.y + inp2_f4.z,
                                  inp1_f4.z + inp2_f4.y, inp1_f4.w + inp2_f4.x);
    float4 temp2_f4 = make_float4(inp1_f4.x - inp2_f4.w, inp2_f4.z - inp1_f4.y,
                                  inp1_f4.z - inp2_f4.y, inp2_f4.x - inp1_f4.w);
    float2 temp3_f2 = make_float2(temp1_f4.x + temp1_f4.w, temp1_f4.y + temp1_f4.z);

    float4 temp4_f4, temp5_f4;

    // Apply DCT transformation
    temp4_f4.x = temp3_f2.x + temp3_f2.y;
    temp5_f4.x = temp3_f2.x - temp3_f2.y;
    temp4_f4.z = fmaf(dctBE.x, (temp1_f4.x - temp1_f4.w), dctBE.y * (temp1_f4.y - temp1_f4.z));
    temp5_f4.z = fmaf(dctBE.y, (temp1_f4.x - temp1_f4.w), -dctBE.x * (temp1_f4.y - temp1_f4.z));

    temp4_f4.y =
        fmaf(dctACDF.x, temp2_f4.x,
             fmaf(-dctACDF.y, temp2_f4.y, fmaf(dctACDF.z, temp2_f4.z, -dctACDF.w * temp2_f4.w)));
    temp4_f4.w =
        fmaf(dctACDF.y, temp2_f4.x,
             fmaf(dctACDF.w, temp2_f4.y, fmaf(-dctACDF.x, temp2_f4.z, dctACDF.z * temp2_f4.w)));
    temp5_f4.y =
        fmaf(dctACDF.z, temp2_f4.x,
             fmaf(dctACDF.x, temp2_f4.y, fmaf(dctACDF.w, temp2_f4.z, -dctACDF.y * temp2_f4.w)));
    temp5_f4.w =
        fmaf(dctACDF.w, temp2_f4.x,
             fmaf(dctACDF.z, temp2_f4.y, fmaf(dctACDF.y, temp2_f4.z, dctACDF.x * temp2_f4.w)));

    *(float4*)&vec[0] = (temp4_f4 * MAKE_FLOAT4(normCoeff));  // stores to vec[0] through vec[3]
    *(float4*)&vec[4] = (temp5_f4 * MAKE_FLOAT4(normCoeff));  // stores to vec[4] through vec[7]
}

// Inverse 1D DCT
__device__ inline void dct_inv_8x8_1d(float* vec, bool offset128) {
    int val = (128.0f * offset128);

    // Load data into float4 vectors
    float4 vec1_f4 = *(float4*)&vec[0];
    float4 vec2_f4 = *(float4*)&vec[4];

    float temp0 = vec1_f4.x + vec2_f4.x;
    float temp1 = fmaf(dctBE.x, vec1_f4.z, (dctBE.y * vec2_f4.z));
    float temp2 = temp0 + temp1;
    float temp3 = temp0 - temp1;

    float temp4 =
        fmaf(dctACDF.w, vec2_f4.w,
             fmaf(dctACDF.x, vec1_f4.y, fmaf(dctACDF.y, vec1_f4.w, (dctACDF.z * vec2_f4.y))));
    float temp5 =
        fmaf(dctACDF.x, vec2_f4.w,
             fmaf(-dctACDF.w, vec1_f4.y, fmaf(dctACDF.z, vec1_f4.w, (-dctACDF.y * vec2_f4.y))));

    float temp6 = vec1_f4.x - vec2_f4.x;
    float temp7 = fmaf(dctBE.y, vec1_f4.z, (-dctBE.x * vec2_f4.z));

    float temp8 = temp6 + temp7;
    float temp9 = temp6 - temp7;
    float temp10 =
        fmaf(dctACDF.y, vec1_f4.y,
             fmaf(-dctACDF.z, vec2_f4.w, fmaf(-dctACDF.w, vec1_f4.w, (-dctACDF.x * vec2_f4.y))));
    float temp11 =
        fmaf(dctACDF.z, vec1_f4.y,
             fmaf(dctACDF.y, vec2_f4.w, fmaf(-dctACDF.x, vec1_f4.w, (dctACDF.w * vec2_f4.y))));

    vec[0] = fmaf(normCoeff, (temp2 + temp4), val);
    vec[7] = fmaf(normCoeff, (temp2 - temp4), val);
    vec[4] = fmaf(normCoeff, (temp3 + temp5), val);
    vec[3] = fmaf(normCoeff, (temp3 - temp5), val);

    vec[1] = fmaf(normCoeff, (temp8 + temp10), val);
    vec[5] = fmaf(normCoeff, (temp9 - temp11), val);
    vec[2] = fmaf(normCoeff, (temp9 + temp11), val);
    vec[6] = fmaf(normCoeff, (temp8 - temp10), val);
}

// Quantization
__device__ inline void quantize(float* value, float* coeff, float qScale) {
    for (int i = 0; i < 8; i++) {
        float qCoeff = coeff[i] * qScale;               // Runtime multiplication (matches HOST)
        value[i] = qCoeff * roundf(value[i] / qCoeff);  // Uses roundf; HOST AVX2 uses std::round
                                                        // with equivalent behavior (no clamping)
    }
}

// Compute quality factor for JPEG quantization based on quality parameter
__device__ inline Rpp32f get_quality_factor(Rpp32s quality) {
    quality = max(1, min(100, quality));
    const Rpp32f lowerCompression = 2.0f - (quality / 50.0f);
    const Rpp32f higherCompression = 50.0f / quality;
    return (quality < 50) ? higherCompression : lowerCompression;
}

// Scale input values to 0-255 range for DCT processing
__device__ inline void scale_to_dct_range(uchar* srcPtr, d_float8* values) {
    // uchar is already in 0-255 range, no scaling needed
}

__device__ inline void scale_to_dct_range(float* srcPtr, d_float8* values) {
    // Scale float from 0-1 to 0-255
    values->f4[0] = values->f4[0] * FLOAT4_255;
    values->f4[1] = values->f4[1] * FLOAT4_255;
}

__device__ inline void scale_to_dct_range(half* srcPtr, d_float8* values) {
    // Scale half from 0-1 to 0-255
    values->f4[0] = values->f4[0] * FLOAT4_255;
    values->f4[1] = values->f4[1] * FLOAT4_255;
}

__device__ inline void scale_to_dct_range(schar* srcPtr, d_float8* values) {
    // Scale schar from -128,127 to 0-255
    values->f4[0] = values->f4[0] + FLOAT4_128;
    values->f4[1] = values->f4[1] + FLOAT4_128;
}

// Scale RGB values (d_float24) to 0-255 range for color kernels
__device__ inline void scale_rgb_to_dct_range(uchar* srcPtr, d_float24* rgb_f24) {
    // uchar is already in 0-255 range, no scaling needed
}

__device__ inline void scale_rgb_to_dct_range(float* srcPtr, d_float24* rgb_f24) {
    // Scale float from 0-1 to 0-255 for all RGB channels
    rpp_hip_math_multiply24_const(rgb_f24, rgb_f24, maxVal255_f4);
}

__device__ inline void scale_rgb_to_dct_range(half* srcPtr, d_float24* rgb_f24) {
    // Scale half from 0-1 to 0-255 for all RGB channels
    rpp_hip_math_multiply24_const(rgb_f24, rgb_f24, maxVal255_f4);
}

__device__ inline void scale_rgb_to_dct_range(schar* srcPtr, d_float24* rgb_f24) {
    // Scale schar from -128,127 to 0-255 for all RGB channels
    rpp_hip_math_add24_const(rgb_f24, rgb_f24, maxVal128_f4);
}

// Horizontal Upsampling and Color conversion to RGB
__device__ inline void upsample_and_RGB_hip_compute(float4 cb_f4, float4 cr_f4, d_float8* ch1_f8,
                                                    d_float8* ch2_f8, d_float8* ch3_f8) {
    d_float8 cb_f8, cr_f8, r_f8, g_f8, b_f8;
    // Copy Y values
    d_float8 y_f8 = *((d_float8*)ch1_f8);

    // Subtract 128 from cb_f4 and Cr_f4 before expanding
    cb_f4 = cb_f4 - FLOAT4_128;
    cr_f4 = cr_f4 - FLOAT4_128;

    // Expand each value
    cb_f8.f4[0] = make_float4(cb_f4.x, cb_f4.x, cb_f4.y, cb_f4.y);
    cb_f8.f4[1] = make_float4(cb_f4.z, cb_f4.z, cb_f4.w, cb_f4.w);
    cr_f8.f4[0] = make_float4(cr_f4.x, cr_f4.x, cr_f4.y, cr_f4.y);
    cr_f8.f4[1] = make_float4(cr_f4.z, cr_f4.z, cr_f4.w, cr_f4.w);

    // Now use the offset-adjusted values in the conversion
    r_f8.f4[0] = y_f8.f4[0] + (MAKE_FLOAT4(1.402f) * cr_f8.f4[0]);
    r_f8.f4[1] = y_f8.f4[1] + (MAKE_FLOAT4(1.402f) * cr_f8.f4[1]);

    g_f8.f4[0] = y_f8.f4[0] - (MAKE_FLOAT4(0.344136285f) * cb_f8.f4[0]) -
                 (MAKE_FLOAT4(0.714136285f) * cr_f8.f4[0]);
    g_f8.f4[1] = y_f8.f4[1] - (MAKE_FLOAT4(0.344136285f) * cb_f8.f4[1]) -
                 (MAKE_FLOAT4(0.714136285f) * cr_f8.f4[1]);

    b_f8.f4[0] = y_f8.f4[0] + (MAKE_FLOAT4(1.772f) * cb_f8.f4[0]);
    b_f8.f4[1] = y_f8.f4[1] + (MAKE_FLOAT4(1.772f) * cb_f8.f4[1]);

    // Write back the results
    *((d_float8*)ch1_f8) = r_f8;
    *((d_float8*)ch2_f8) = g_f8;
    *((d_float8*)ch3_f8) = b_f8;
}

template <typename T>
__device__ __forceinline__ void process_jpeg_distortion(float src_smem[48][128],
                                                        int hipThreadIdx_x8, int hipThreadIdx_x4,
                                                        int localThreadIdx_y, int alignedWidth,
                                                        int3 hipThreadIdx_y_channel, float* tableY,
                                                        float* tableCbCr, float qScale, T* srcPtr,
                                                        T* dstPtr) {
    // ----------- Step 1: RGB to YCbCr Conversion -----------
    d_float8 y_f8;
    d_float24 rgb_f24;
    rgb_f24.f8[0] = *((d_float8*)&src_smem[localThreadIdx_y][hipThreadIdx_x8]);
    rgb_f24.f8[1] = *((d_float8*)&src_smem[localThreadIdx_y + 16][hipThreadIdx_x8]);
    rgb_f24.f8[2] = *((d_float8*)&src_smem[localThreadIdx_y + 32][hipThreadIdx_x8]);

    // Scale RGB values to 0-255 range for proper YCbCr conversion
    scale_rgb_to_dct_range(srcPtr, &rgb_f24);

    // Update shared memory with scaled values
    *((d_float8*)&src_smem[localThreadIdx_y][hipThreadIdx_x8]) = rgb_f24.f8[0];
    *((d_float8*)&src_smem[localThreadIdx_y + 16][hipThreadIdx_x8]) = rgb_f24.f8[1];
    *((d_float8*)&src_smem[localThreadIdx_y + 32][hipThreadIdx_x8]) = rgb_f24.f8[2];
    __syncthreads();

    int cbcrY = localThreadIdx_y * 2;
    y_hip_compute(srcPtr, &rgb_f24, &y_f8);
    __syncthreads();

    // ----------- Step 2: Downsample CbCr -----------
    float4 cb_f4 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 cr_f4 = {0.0f, 0.0f, 0.0f, 0.0f};
    if (localThreadIdx_y < 8) {
        downsample_cbcr_hip_compute((d_float8*)&src_smem[cbcrY][hipThreadIdx_x8],
                                    (d_float8*)&src_smem[cbcrY + 1][hipThreadIdx_x8],
                                    (d_float8*)&src_smem[cbcrY + 16][hipThreadIdx_x8],
                                    (d_float8*)&src_smem[cbcrY + 17][hipThreadIdx_x8],
                                    (d_float8*)&src_smem[cbcrY + 32][hipThreadIdx_x8],
                                    (d_float8*)&src_smem[cbcrY + 33][hipThreadIdx_x8], &cb_f4,
                                    &cr_f4);
    }
    __syncthreads();  // all smem reads must complete before any thread writes Cb/Cr back

    if (localThreadIdx_y < 8) {
        *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = cb_f4;
        *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = cr_f4;
    }

    *(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = y_f8;
    __syncthreads();

    // ----------- Step 3: Clamp + Forward DCT -----------
    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    __syncthreads();

    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], true);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], true);
    __syncthreads();

    // ----------- Step4 Column-wise DCT -----------
    int col = ((hipThreadIdx_x8 / 8) * 16) + localThreadIdx_y;
    if ((col < 128) && (col < alignedWidth)) {
        float colVec[32];
        for (int i = 0; i < 32; i++) colVec[i] = src_smem[i][col];

        dct_fwd_8x8_1d(&colVec[0], false);
        dct_fwd_8x8_1d(&colVec[8], false);
        dct_fwd_8x8_1d(&colVec[16], false);
        dct_fwd_8x8_1d(&colVec[24], false);

        for (int i = 0; i < 32; i++) src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // ----------- Step 5: Quantization -----------
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
             &tableY[(localThreadIdx_y % 8) * 8], qScale);
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
             &tableCbCr[(localThreadIdx_y % 8) * 8], qScale);
    __syncthreads();

    // ----------- Step 6: Inverse DCT -----------
    if ((col < 128) && (col < alignedWidth)) {
        float colVec[32];
#pragma unroll
        for (int i = 0; i < 32; i++) colVec[i] = src_smem[i][col];

        dct_inv_8x8_1d(&colVec[0], false);
        dct_inv_8x8_1d(&colVec[8], false);
        dct_inv_8x8_1d(&colVec[16], false);
        dct_inv_8x8_1d(&colVec[24], false);

#pragma unroll
        for (int i = 0; i < 32; i++) src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // ----------- Step 7: Row-wise IDCT -----------
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], true);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], true);
    __syncthreads();

    // ----------- Step 8: Clamp & Upsample -----------
    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    __syncthreads();

    cbcrY = localThreadIdx_y / 2;
    cb_f4 = *(float4*)&src_smem[cbcrY + 16][hipThreadIdx_x4];
    cr_f4 = *(float4*)&src_smem[cbcrY + 24][hipThreadIdx_x4];
    __syncthreads();

    upsample_and_RGB_hip_compute(cb_f4, cr_f4,
                                 (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                 (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                 (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    // ----------- Step 9: Final Clamp -----------
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);

    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
}

template <typename T>
__global__ void jpeg_compression_distortion_pkd3_hip_tensor(T* srcPtr, uint2 srcStridesNH,
                                                            T* dstPtr, uint2 dstStridesNH,
                                                            RpptROIPtr roiTensorPtrSrc,
                                                            float* tableY, float* tableCbCr,
                                                            Rpp32s* qualityTensor) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float qScale = get_quality_factor(qualityTensor[id_z]);

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = (roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) & ~15;
    int alignedHeight = (roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) & ~15;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth)) return;

    // ROI parameters
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;

    __shared__ float src_smem[48][128];
    int3 hipThreadIdx_y_channel = {hipThreadIdx_y, hipThreadIdx_y + 16, hipThreadIdx_y + 32};

    float* src_smem_channel[3] = {&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                  &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                  &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]};

    // ----------- Step 1: Load from Global Memory to Shared Memory -----------
    int srcIdx;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    // Check if we need special handling for image edges
    if (id_y < roiHeight)
        srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiY) * srcStridesNH.y) + ((id_x + roiX) * 3);
    else  // All out-of-bounds threads use the last valid row
        srcIdx = (id_z * srcStridesNH.x) + ((roiHeight - 1 + roiY) * srcStridesNH.y) +
                 ((id_x + roiX) * 3);

    bool isEdge = ((id_x + 8) > roiWidth) && (id_x < alignedWidth);

    if (!isEdge) {
        rpp_hip_load24_pkd3_to_float24_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Partial block load with edge pixel replication
        int validPixels = roiWidth - id_x;
        if (validPixels > 0) {
            for (int i = 0, idx = srcIdx; i < validPixels; i++, idx += 3) {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[idx];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[idx + 1];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[idx + 2];
            }
        }

        int lastValidIdx = srcIdx + ((validPixels - 1) * 3);
        for (int i = validPixels; i < min(validPixels + 16, 8); i++) {
            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[lastValidIdx];
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[lastValidIdx + 1];
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[lastValidIdx + 2];
        }
    }
    __syncthreads();

    // Perform JPEG compression distortion on the shared memory block:
    // Includes RGB to YCbCr conversion, chroma downsampling, DCT, quantization, inverse DCT,
    // upsampling, and final RGB reconstruction.
    process_jpeg_distortion(src_smem, hipThreadIdx_x8, hipThreadIdx_x4, hipThreadIdx_y,
                            alignedWidth, hipThreadIdx_y_channel, tableY, tableCbCr, qScale, srcPtr,
                            dstPtr);

    __syncthreads();

    if ((id_x < roiWidth) && (id_y < roiHeight))
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, src_smem_channel);
}

template <typename T>
__global__ void jpeg_compression_distortion_pln3_hip_tensor(T* srcPtr, uint3 srcStridesNCH,
                                                            T* dstPtr, uint3 dstStridesNCH,
                                                            RpptROIPtr roiTensorPtrSrc,
                                                            float* tableY, float* tableCbCr,
                                                            Rpp32s* qualityTensor) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float qScale = get_quality_factor(qualityTensor[id_z]);

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = (roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) & ~15;
    int alignedHeight = (roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) & ~15;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth)) return;

    // ROI parameters
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;

    // Shared memory declaration
    __shared__ float src_smem[48][128];  // Assuming 48 rows (aligned height for 3 channels)
    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    float* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    // ----------- Step 1: Load from Global Memory to Shared Memory -----------
    uint3 srcIdx, dstIdx;
    dstIdx.x = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    dstIdx.y = dstIdx.x + dstStridesNCH.y;
    dstIdx.z = dstIdx.y + dstStridesNCH.y;

    // Determine actual source coordinates based on ROI bounds
    int srcY = (id_y < roiHeight) ? (id_y + roiY) : (roiHeight - 1 + roiY);

    int srcX = id_x + roiX;
    srcIdx.x = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;

    // Horizontal loading with proper padding to 16-pixel multiples
    bool isEdge = ((id_x + 8) > roiWidth);
    if (!isEdge) {
        // Load 8 pixels at once for each channel
        rpp_hip_load8_and_unpack_to_float8(
            srcPtr + srcIdx.x, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(
            srcPtr + srcIdx.y, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(
            srcPtr + srcIdx.z, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } else {
        // Edge case: Need to pad horizontally
        int validPixels = (id_x < roiWidth) ? (roiWidth - id_x) : 0;
        if (validPixels > 0) {
            // Load valid pixels for each channel separately
            for (int i = 0; i < validPixels; i++) {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x + i];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y + i];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z + i];
            }
            // Pad remaining pixels in the block by duplicating the last valid pixel for each
            // channel
            for (int i = validPixels; i < 8; i++) {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] =
                    srcPtr[srcIdx.x + validPixels - 1];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] =
                    srcPtr[srcIdx.y + validPixels - 1];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] =
                    srcPtr[srcIdx.z + validPixels - 1];
            }
        } else {
            // Completely outside ROI width but within aligned width - duplicate last pixel of the
            // row
            int srcX = (roiWidth - 1) + roiX;  // Last valid pixel in the row
            srcIdx.x = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;
            srcIdx.y = srcIdx.x + srcStridesNCH.y;
            srcIdx.z = srcIdx.y + srcStridesNCH.y;
            // Duplicate the last valid pixel across the entire 8-pixel block
            for (int i = 0; i < 8; i++) {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z];
            }
        }
    }
    __syncthreads();

    // Perform JPEG compression distortion on the shared memory block:
    // Includes RGB to YCbCr conversion, chroma downsampling, DCT, quantization, inverse DCT,
    // upsampling, and final RGB reconstruction.
    process_jpeg_distortion(src_smem, hipThreadIdx_x8, hipThreadIdx_x4, hipThreadIdx_y,
                            alignedWidth, hipThreadIdx_y_channel, tableY, tableCbCr, qScale, srcPtr,
                            dstPtr);

    __syncthreads();

    if ((id_x < roiWidth) && (id_y < roiHeight)) {
        rpp_hip_pack_float8_and_store8(
            dstPtr + dstIdx.x, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_pack_float8_and_store8(
            dstPtr + dstIdx.y, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_pack_float8_and_store8(
            dstPtr + dstIdx.z, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
}

template <typename T>
__global__ void jpeg_compression_distortion_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH,
                                                                 T* dstPtr, uint3 dstStridesNCH,
                                                                 RpptROIPtr roiTensorPtrSrc,
                                                                 float* tableY, float* tableCbCr,
                                                                 Rpp32s* qualityTensor) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float qScale = get_quality_factor(qualityTensor[id_z]);

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = (roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) & ~15;
    int alignedHeight = (roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) & ~15;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth)) return;

    // ROI parameters
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;

    // Shared memory declaration
    __shared__ float src_smem[48][128];  // Assuming 48 rows (aligned height for 3 channels)
    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    float* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    // ----------- Step 1: Load from Global Memory to Shared Memory -----------
    int srcIdx;
    uint3 dstIdx;
    dstIdx.x = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    dstIdx.y = dstIdx.x + dstStridesNCH.y;
    dstIdx.z = dstIdx.y + dstStridesNCH.y;

    // Check if we need special handling for image edges
    if (id_y < roiHeight)
        srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiY) * srcStridesNH.y) + ((id_x + roiX) * 3);
    else  // All out-of-bounds threads use the last valid row
        srcIdx = (id_z * srcStridesNH.x) + ((roiHeight - 1 + roiY) * srcStridesNH.y) +
                 ((id_x + roiX) * 3);

    bool isEdge = ((id_x + 8) > roiWidth) && (id_x < alignedWidth);
    if (!isEdge)
        rpp_hip_load24_pkd3_to_float24_pln3(srcPtr + srcIdx, src_smem_channel);
    else {
        int validPixels = roiWidth - id_x;
        // Load valid pixels (only if id_x is within valid range)
        if (validPixels > 0) {
            for (int i = 0, idx = srcIdx; i < validPixels; i++, idx += 3) {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[idx];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[idx + 1];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[idx + 2];
            }
        }
        // Pad 16 pixels by duplicating the last valid pixel
        int lastValidIdx = srcIdx + ((validPixels - 1) * 3);
        for (int i = validPixels; i < min(validPixels + 16, 8); i++) {
            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[lastValidIdx];
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[lastValidIdx + 1];
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[lastValidIdx + 2];
        }
    }
    __syncthreads();

    // Perform JPEG compression distortion on the shared memory block:
    // Includes RGB to YCbCr conversion, chroma downsampling, DCT, quantization, inverse DCT,
    // upsampling, and final RGB reconstruction.
    process_jpeg_distortion(src_smem, hipThreadIdx_x8, hipThreadIdx_x4, hipThreadIdx_y,
                            alignedWidth, hipThreadIdx_y_channel, tableY, tableCbCr, qScale, srcPtr,
                            dstPtr);

    __syncthreads();

    if ((id_x < roiWidth) && (id_y < roiHeight)) {
        rpp_hip_pack_float8_and_store8(
            dstPtr + dstIdx.x, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_pack_float8_and_store8(
            dstPtr + dstIdx.y, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_pack_float8_and_store8(
            dstPtr + dstIdx.z, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
}

template <typename T>
__global__ void jpeg_compression_distortion_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH,
                                                                 T* dstPtr, uint2 dstStridesNH,
                                                                 RpptROIPtr roiTensorPtrSrc,
                                                                 float* tableY, float* tableCbCr,
                                                                 Rpp32s* qualityTensor) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float qScale = get_quality_factor(qualityTensor[id_z]);

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = (roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) & ~15;
    int alignedHeight = (roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) & ~15;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth)) return;

    // ROI parameters
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;

    // Shared memory declaration
    __shared__ float src_smem[48][128];  // Assuming 48 rows (aligned height for 3 channels)
    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    float* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    // ----------- Step 1: Load from Global Memory to Shared Memory -----------
    uint3 srcIdx;

    // Check if we need special handling for image edges
    int id_y_clamped;
    id_y_clamped = (id_y < roiHeight) ? id_y : (roiHeight - 1);

    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_clamped + roiY) * srcStridesNCH.z) + (id_x + roiX);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;

    int dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    // Determine actual source coordinates based on ROI bounds
    int srcY = (id_y < roiHeight) ? (id_y + roiY) : (roiHeight - 1 + roiY);

    // Horizontal loading with proper padding to 16-pixel multiples
    bool isEdge = (id_x + 8 > roiWidth);
    if (!isEdge) {
        // Normal case: The entire 8-pixel block fits within the ROI
        int srcX = id_x + roiX;
        srcIdx.x = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;
        srcIdx.y = srcIdx.x + srcStridesNCH.y;
        srcIdx.z = srcIdx.y + srcStridesNCH.y;

        // Load 8 pixels at once for each channel
        rpp_hip_load8_and_unpack_to_float8(
            srcPtr + srcIdx.x, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(
            srcPtr + srcIdx.y, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(
            srcPtr + srcIdx.z, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } else {
        // Edge case: Need to pad horizontally
        int validPixels = (id_x < roiWidth) ? (roiWidth - id_x) : 0;

        if (validPixels > 0) {
            // Load valid pixels and pad the rest
            int srcX = id_x + roiX;
            srcIdx.x = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;
            srcIdx.y = srcIdx.x + srcStridesNCH.y;
            srcIdx.z = srcIdx.y + srcStridesNCH.y;

            // Load valid pixels for each channel separately
            for (int i = 0; i < validPixels; i++) {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x + i];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y + i];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z + i];
            }
            // Pad remaining pixels in the block by duplicating the last valid pixel for each
            // channel
            for (int i = validPixels; i < 8; i++) {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] =
                    srcPtr[srcIdx.x + validPixels - 1];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] =
                    srcPtr[srcIdx.y + validPixels - 1];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] =
                    srcPtr[srcIdx.z + validPixels - 1];
            }
        } else {
            // Completely outside ROI width but within aligned width - duplicate last pixel of the
            // row
            int srcX = (roiWidth - 1) + roiX;  // Last valid pixel in the row
            srcIdx.x = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;
            srcIdx.y = srcIdx.x + srcStridesNCH.y;
            srcIdx.z = srcIdx.y + srcStridesNCH.y;
            // Duplicate the last valid pixel across the entire 8-pixel block
            for (int i = 0; i < 8; i++) {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z];
            }
        }
    }
    __syncthreads();

    // Perform JPEG compression distortion on the shared memory block:
    // Includes RGB to YCbCr conversion, chroma downsampling, DCT, quantization, inverse DCT,
    // upsampling, and final RGB reconstruction.
    process_jpeg_distortion(src_smem, hipThreadIdx_x8, hipThreadIdx_x4, hipThreadIdx_y,
                            alignedWidth, hipThreadIdx_y_channel, tableY, tableCbCr, qScale, srcPtr,
                            dstPtr);

    __syncthreads();

    if ((id_x < roiWidth) && (id_y < roiHeight))
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, src_smem_channel);
}

template <typename T>
__global__ void jpeg_compression_distortion_pln1_hip_tensor(T* srcPtr, uint3 srcStridesNCH,
                                                            T* dstPtr, uint3 dstStridesNCH,
                                                            RpptROIPtr roiTensorPtrSrc,
                                                            float* tableY, Rpp32s* qualityTensor) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float qScale = get_quality_factor(qualityTensor[id_z]);

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;

    int alignedWidth = (roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) & ~15;
    int alignedHeight = (roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) & ~15;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth)) return;

    // ROI parameters
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;

    // Shared memory declaration
    __shared__ float src_smem[16][128];  // Assuming 16 rows (aligned height for 1 channel)
    d_float8 zeroes_f8 = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    *(d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = zeroes_f8;
    __syncthreads();

    // ----------- Step 1: Load from Global Memory to Shared Memory -----------
    uint srcIdx, dstIdx;

    // Calculate destination indices first
    dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    // Determine actual source coordinates based on ROI bounds
    int srcY = (id_y < roiHeight) ? (id_y + roiY) : ((roiHeight - 1) + roiY);

    int srcX = id_x + roiX;
    srcIdx = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;

    // Horizontal loading with proper padding to 16-pixel multiples
    bool isEdge = ((id_x + 8) > roiWidth);
    if (!isEdge) {
        // Load 8 pixels at once for each channel
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx,
                                           (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    } else {
        // Edge case: Need to pad horizontally
        int validPixels = (id_x < roiWidth) ? (roiWidth - id_x) : 0;

        if (validPixels > 0) {
            // Load valid pixels for each channel separately
            for (int i = 0; i < validPixels; i++)
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[srcIdx + i];

            // Pad remaining pixels in the block by duplicating the last valid pixel for each
            // channel
            for (int i = validPixels; i < 8; i++)
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[srcIdx + validPixels - 1];
        } else {
            // Completely outside ROI width but within aligned width - duplicate last pixel of the
            // row
            int srcX = (roiWidth - 1) + roiX;  // Last valid pixel in the row
            srcIdx = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;
            // Duplicate the last valid pixel across the entire 8-pixel block
            for (int i = 0; i < 8; i++)
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[srcIdx];
        }
    }
    __syncthreads();

    // Scale input values to 0-255 range for DCT processing
    scale_to_dct_range(srcPtr, (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);

    // Doing -128 as part of DCT,
    // // ----------- Step 2: Forward DCT -----------
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], true);
    __syncthreads();

    // // ----------- Step 3: Column-wise DCT -----------
    int col = (hipThreadIdx_x * 16) + hipThreadIdx_y;
    // Process all 128 columns
    if ((col < 128) && (col < alignedWidth)) {
        // Load column into temporary array
        float colVec[16];

        for (int i = 0; i < 16; i++) colVec[i] = src_smem[i][col];

        dct_fwd_8x8_1d(&colVec[0], false);
        dct_fwd_8x8_1d(&colVec[8], false);

        for (int i = 0; i < 16; i++) src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // // ----------- Step 4: Quantization -----------
    quantize(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &tableY[(hipThreadIdx_y % 8) * 8], qScale);
    __syncthreads();

    //// ----------- Step 5: Inverse DCT -----------
    // Process all 128 columns
    // Adding back 128 as part of DCT
    if ((col < 128) && (col < alignedWidth)) {
        // Load column into temporary array
        float colVec[16];

        for (int i = 0; i < 16; i++) colVec[i] = src_smem[i][col];

        dct_inv_8x8_1d(&colVec[0], false);
        dct_inv_8x8_1d(&colVec[8], false);

        for (int i = 0; i < 16; i++) src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // Inverse DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], true);
    __syncthreads();

    clamp_range((float*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);

    // Clamp values and store results
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);

    // ----------- Step 6: Final Clamp & Store -----------
    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    __syncthreads();

    if ((id_x < roiWidth) && (id_y < roiHeight))
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx,
                                       (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
}

template <typename T>
RppStatus hip_exec_jpeg_compression_distortion(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                               RpptDescPtr dstDescPtr, Rpp32s* qualityTensor,
                                               RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                               rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB) hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    // Allocate pinned memory and copy tables (no prescaling - matches HOST reference)
    Rpp32f* tableY =
        reinterpret_cast<Rpp32f*>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
    Rpp32f* tableCbCr = tableY + 64;
    std::memcpy(tableY, lumaQuantTable, 64 * sizeof(float));
    std::memcpy(tableCbCr, chromaQuantTable, 64 * sizeof(float));

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC)) {
        hipLaunchKernelGGL(
            jpeg_compression_distortion_pkd3_hip_tensor,
            dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                 ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                 ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(), srcPtr,
            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride), dstPtr,
            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride), roiTensorPtrSrc,
            tableY, tableCbCr, qualityTensor);
        HIP_CHECK_LAUNCH_RETURN();
    }

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) &&
        (srcDescPtr->c == 3)) {
        hipLaunchKernelGGL(jpeg_compression_distortion_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                      srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           roiTensorPtrSrc, tableY, tableCbCr, qualityTensor);
        HIP_CHECK_LAUNCH_RETURN();
    }

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) &&
        (srcDescPtr->c == 1)) {
        hipLaunchKernelGGL(jpeg_compression_distortion_pln1_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                      srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           roiTensorPtrSrc, tableY, qualityTensor);
        HIP_CHECK_LAUNCH_RETURN();
    }

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW)) {
        hipLaunchKernelGGL(
            jpeg_compression_distortion_pkd3_pln3_hip_tensor,
            dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                 ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                 ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(), srcPtr,
            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride), dstPtr,
            make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                       dstDescPtr->strides.hStride),
            roiTensorPtrSrc, tableY, tableCbCr, qualityTensor);
        HIP_CHECK_LAUNCH_RETURN();
    }

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC)) {
        hipLaunchKernelGGL(
            jpeg_compression_distortion_pln3_pkd3_hip_tensor,
            dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                 ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                 ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(), srcPtr,
            make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                       srcDescPtr->strides.hStride),
            dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
            roiTensorPtrSrc, tableY, tableCbCr, qualityTensor);
        HIP_CHECK_LAUNCH_RETURN();
    }
    return RPP_SUCCESS;
}

template RppStatus hip_exec_jpeg_compression_distortion<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*,
                                                               RpptDescPtr, Rpp32s*, RpptROIPtr,
                                                               RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_jpeg_compression_distortion<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                                RpptDescPtr, Rpp32s*, RpptROIPtr,
                                                                RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_jpeg_compression_distortion<half>(half*, RpptDescPtr, half*,
                                                              RpptDescPtr, Rpp32s*, RpptROIPtr,
                                                              RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_jpeg_compression_distortion<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*,
                                                               RpptDescPtr, Rpp32s*, RpptROIPtr,
                                                               RpptRoiType, rpp::Handle&);
