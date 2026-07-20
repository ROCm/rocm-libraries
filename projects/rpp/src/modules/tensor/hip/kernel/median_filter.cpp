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

// -------------------- median_filter device helpers --------------------

template <typename T>
__device__ void median_filter_3x3_row_hip_compute(T* src_smem, d_float8* median_f8) {
    using VectorType = typename FilterDispatch<T>::VectorType;
    // Load 3 rows of shared memory into vectorized uint3/ int3 format
    VectorType row0 = *((VectorType*)&src_smem[0 * SMEM_LENGTH_X]);
    VectorType row1 = *((VectorType*)&src_smem[1 * SMEM_LENGTH_X]);
    VectorType row2 = *((VectorType*)&src_smem[2 * SMEM_LENGTH_X]);

    float3 val0_f3, val1_f3, val2_f3, valz_f3;

    // ======================
    // Pixel : 3x3 window
    // [ row0.x[0], row0.x[1], row0.x[2] ]
    // [ row1.x[0], row1.x[1], row1.x[2] ]
    // [ row2.x[0], row2.x[1], row2.x[2] ]
    // ======================

    // pixel 0
    valz_f3.x = rpp_hip_unpack0(row0.x);
    valz_f3.y = rpp_hip_unpack1(row0.x);
    valz_f3.z = rpp_hip_unpack2(row0.x);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack0(row1.x);
    valz_f3.y = rpp_hip_unpack1(row1.x);
    valz_f3.z = rpp_hip_unpack2(row1.x);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack0(row2.x);
    valz_f3.y = rpp_hip_unpack1(row2.x);
    valz_f3.z = rpp_hip_unpack2(row2.x);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    // Compute 3x3 median:
    // - Take max of row mins, median of row medians, min of row maxes
    // - Then, compute median of those three
    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[0] = rpp_hip_median3(valz_f3);

    // pixel 1
    valz_f3.x = rpp_hip_unpack1(row0.x);
    valz_f3.y = rpp_hip_unpack2(row0.x);
    valz_f3.z = rpp_hip_unpack3(row0.x);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack1(row1.x);
    valz_f3.y = rpp_hip_unpack2(row1.x);
    valz_f3.z = rpp_hip_unpack3(row1.x);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack1(row2.x);
    valz_f3.y = rpp_hip_unpack2(row2.x);
    valz_f3.z = rpp_hip_unpack3(row2.x);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[1] = rpp_hip_median3(valz_f3);

    // pixel 2
    valz_f3.x = rpp_hip_unpack2(row0.x);
    valz_f3.y = rpp_hip_unpack3(row0.x);
    valz_f3.z = rpp_hip_unpack0(row0.y);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack2(row1.x);
    valz_f3.y = rpp_hip_unpack3(row1.x);
    valz_f3.z = rpp_hip_unpack0(row1.y);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack2(row2.x);
    valz_f3.y = rpp_hip_unpack3(row2.x);
    valz_f3.z = rpp_hip_unpack0(row2.y);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[2] = rpp_hip_median3(valz_f3);

    // pixel 3
    valz_f3.x = rpp_hip_unpack3(row0.x);
    valz_f3.y = rpp_hip_unpack0(row0.y);
    valz_f3.z = rpp_hip_unpack1(row0.y);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack3(row1.x);
    valz_f3.y = rpp_hip_unpack0(row1.y);
    valz_f3.z = rpp_hip_unpack1(row1.y);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack3(row2.x);
    valz_f3.y = rpp_hip_unpack0(row2.y);
    valz_f3.z = rpp_hip_unpack1(row2.y);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[3] = rpp_hip_median3(valz_f3);

    // pixel 4
    valz_f3.x = rpp_hip_unpack0(row0.y);
    valz_f3.y = rpp_hip_unpack1(row0.y);
    valz_f3.z = rpp_hip_unpack2(row0.y);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack0(row1.y);
    valz_f3.y = rpp_hip_unpack1(row1.y);
    valz_f3.z = rpp_hip_unpack2(row1.y);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack0(row2.y);
    valz_f3.y = rpp_hip_unpack1(row2.y);
    valz_f3.z = rpp_hip_unpack2(row2.y);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[4] = rpp_hip_median3(valz_f3);

    // pixel 5
    valz_f3.x = rpp_hip_unpack1(row0.y);
    valz_f3.y = rpp_hip_unpack2(row0.y);
    valz_f3.z = rpp_hip_unpack3(row0.y);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack1(row1.y);
    valz_f3.y = rpp_hip_unpack2(row1.y);
    valz_f3.z = rpp_hip_unpack3(row1.y);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack1(row2.y);
    valz_f3.y = rpp_hip_unpack2(row2.y);
    valz_f3.z = rpp_hip_unpack3(row2.y);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[5] = rpp_hip_median3(valz_f3);

    // pixel 6
    valz_f3.x = rpp_hip_unpack2(row0.y);
    valz_f3.y = rpp_hip_unpack3(row0.y);
    valz_f3.z = rpp_hip_unpack0(row0.z);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack2(row1.y);
    valz_f3.y = rpp_hip_unpack3(row1.y);
    valz_f3.z = rpp_hip_unpack0(row1.z);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack2(row2.y);
    valz_f3.y = rpp_hip_unpack3(row2.y);
    valz_f3.z = rpp_hip_unpack0(row2.z);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[6] = rpp_hip_median3(valz_f3);

    // pixel 7
    valz_f3.x = rpp_hip_unpack3(row0.y);
    valz_f3.y = rpp_hip_unpack0(row0.z);
    valz_f3.z = rpp_hip_unpack1(row0.z);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack3(row1.y);
    valz_f3.y = rpp_hip_unpack0(row1.z);
    valz_f3.z = rpp_hip_unpack1(row1.z);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack3(row2.y);
    valz_f3.y = rpp_hip_unpack0(row2.z);
    valz_f3.z = rpp_hip_unpack1(row2.z);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[7] = rpp_hip_median3(valz_f3);
}

template <>
__device__ void median_filter_3x3_row_hip_compute<float>(float* src_smem, d_float8* median_f8) {
    float* row0Ptr = src_smem;
    float* row1Ptr = row0Ptr + SMEM_LENGTH_X;
    float* row2Ptr = row1Ptr + SMEM_LENGTH_X;
    float3 minVal_f3, maxVal_f3, medianVal_f3;

    for (int px = 0; px < 8; ++px) {
        float3 row0_f3 = make_float3(row0Ptr[px], row0Ptr[px + 1], row0Ptr[px + 2]);
        float3 row1_f3 = make_float3(row1Ptr[px], row1Ptr[px + 1], row1Ptr[px + 2]);
        float3 row2_f3 = make_float3(row2Ptr[px], row2Ptr[px + 1], row2Ptr[px + 2]);

        minVal_f3.x = rpp_hip_min3(row0_f3);
        medianVal_f3.x = rpp_hip_median3(row0_f3);
        maxVal_f3.x = rpp_hip_max3(row0_f3);

        minVal_f3.y = rpp_hip_min3(row1_f3);
        medianVal_f3.y = rpp_hip_median3(row1_f3);
        maxVal_f3.y = rpp_hip_max3(row1_f3);

        minVal_f3.z = rpp_hip_min3(row2_f3);
        medianVal_f3.z = rpp_hip_median3(row2_f3);
        maxVal_f3.z = rpp_hip_max3(row2_f3);

        float maxOfMin = rpp_hip_max3(make_float3(minVal_f3.x, minVal_f3.y, minVal_f3.z));
        float median = rpp_hip_median3(make_float3(medianVal_f3.x, medianVal_f3.y, medianVal_f3.z));
        float minOfMax = rpp_hip_min3(make_float3(maxVal_f3.x, maxVal_f3.y, maxVal_f3.z));

        median_f8->f1[px] = rpp_hip_median3(make_float3(minOfMax, median, maxOfMin));
    }
}

// 5×5 sorting network helper - applies sorting network to find median at p[12]
__device__ __forceinline__ float apply_sorting_network_5x5(float p[25]) {
#define OP(a, b)                   \
    {                              \
        float t = min(p[a], p[b]); \
        p[b] = max(p[a], p[b]);    \
        p[a] = t;                  \
    }
    OP(1, 2);
    OP(0, 1);
    OP(1, 2);
    OP(4, 5);
    OP(3, 4);
    OP(4, 5);
    OP(0, 3);
    OP(2, 5);
    OP(2, 3);
    OP(1, 4);
    OP(1, 2);
    OP(3, 4);
    OP(7, 8);
    OP(6, 7);
    OP(7, 8);
    OP(10, 11);
    OP(9, 10);
    OP(10, 11);
    OP(6, 9);
    OP(8, 11);
    OP(8, 9);
    OP(7, 10);
    OP(7, 8);
    OP(9, 10);
    OP(0, 6);
    OP(4, 10);
    OP(4, 6);
    OP(2, 8);
    OP(2, 4);
    OP(6, 8);
    OP(1, 7);
    OP(5, 11);
    OP(5, 7);
    OP(3, 9);
    OP(3, 5);
    OP(7, 9);
    OP(1, 2);
    OP(3, 4);
    OP(5, 6);
    OP(7, 8);
    OP(9, 10);
    OP(13, 14);
    OP(12, 13);
    OP(13, 14);
    OP(16, 17);
    OP(15, 16);
    OP(16, 17);
    OP(12, 15);
    OP(14, 17);
    OP(14, 15);
    OP(13, 16);
    OP(13, 14);
    OP(15, 16);
    OP(19, 20);
    OP(18, 19);
    OP(19, 20);
    OP(21, 22);
    OP(23, 24);
    OP(21, 23);
    OP(22, 24);
    OP(22, 23);
    OP(18, 21);
    OP(20, 23);
    OP(20, 21);
    OP(19, 22);
    OP(22, 24);
    OP(19, 20);
    OP(21, 22);
    OP(23, 24);
    OP(12, 18);
    OP(16, 22);
    OP(16, 18);
    OP(14, 20);
    OP(20, 24);
    OP(14, 16);
    OP(18, 20);
    OP(22, 24);
    OP(13, 19);
    OP(17, 23);
    OP(17, 19);
    OP(15, 21);
    OP(15, 17);
    OP(19, 21);
    OP(13, 14);
    OP(15, 16);
    OP(17, 18);
    OP(19, 20);
    OP(21, 22);
    OP(23, 24);
    OP(0, 12);
    OP(8, 20);
    OP(8, 12);
    OP(4, 16);
    OP(16, 24);
    OP(12, 16);
    OP(2, 14);
    OP(10, 22);
    OP(10, 14);
    OP(6, 18);
    OP(6, 10);
    OP(10, 12);
    OP(1, 13);
    OP(9, 21);
    OP(9, 13);
    OP(5, 17);
    OP(13, 17);
    OP(3, 15);
    OP(11, 23);
    OP(11, 15);
    OP(7, 19);
    OP(7, 11);
    OP(11, 13);
    OP(11, 12);
#undef OP
    return p[12];
}

// Optimized 5×5 median using sorting network (processes 8 pixels simultaneously)
template <typename T>
__device__ void median_filter_5x5_row_hip_compute(T* src_smem, d_float8* median_f8) {
    // Process 8 pixels using optimized sorting network
    for (int px = 0; px < 8; ++px) {
        // Extract 5×5 window for current pixel
        float p[25];
        int baseOffset = px;

        // Unpack from row0
        p[0] = static_cast<float>(src_smem[0 * SMEM_LENGTH_X + baseOffset]);
        p[1] = static_cast<float>(src_smem[0 * SMEM_LENGTH_X + baseOffset + 1]);
        p[2] = static_cast<float>(src_smem[0 * SMEM_LENGTH_X + baseOffset + 2]);
        p[3] = static_cast<float>(src_smem[0 * SMEM_LENGTH_X + baseOffset + 3]);
        p[4] = static_cast<float>(src_smem[0 * SMEM_LENGTH_X + baseOffset + 4]);

        // Unpack from row1
        p[5] = static_cast<float>(src_smem[1 * SMEM_LENGTH_X + baseOffset]);
        p[6] = static_cast<float>(src_smem[1 * SMEM_LENGTH_X + baseOffset + 1]);
        p[7] = static_cast<float>(src_smem[1 * SMEM_LENGTH_X + baseOffset + 2]);
        p[8] = static_cast<float>(src_smem[1 * SMEM_LENGTH_X + baseOffset + 3]);
        p[9] = static_cast<float>(src_smem[1 * SMEM_LENGTH_X + baseOffset + 4]);

        // Unpack from row2
        p[10] = static_cast<float>(src_smem[2 * SMEM_LENGTH_X + baseOffset]);
        p[11] = static_cast<float>(src_smem[2 * SMEM_LENGTH_X + baseOffset + 1]);
        p[12] = static_cast<float>(src_smem[2 * SMEM_LENGTH_X + baseOffset + 2]);
        p[13] = static_cast<float>(src_smem[2 * SMEM_LENGTH_X + baseOffset + 3]);
        p[14] = static_cast<float>(src_smem[2 * SMEM_LENGTH_X + baseOffset + 4]);

        // Unpack from row3
        p[15] = static_cast<float>(src_smem[3 * SMEM_LENGTH_X + baseOffset]);
        p[16] = static_cast<float>(src_smem[3 * SMEM_LENGTH_X + baseOffset + 1]);
        p[17] = static_cast<float>(src_smem[3 * SMEM_LENGTH_X + baseOffset + 2]);
        p[18] = static_cast<float>(src_smem[3 * SMEM_LENGTH_X + baseOffset + 3]);
        p[19] = static_cast<float>(src_smem[3 * SMEM_LENGTH_X + baseOffset + 4]);

        // Unpack from row4
        p[20] = static_cast<float>(src_smem[4 * SMEM_LENGTH_X + baseOffset]);
        p[21] = static_cast<float>(src_smem[4 * SMEM_LENGTH_X + baseOffset + 1]);
        p[22] = static_cast<float>(src_smem[4 * SMEM_LENGTH_X + baseOffset + 2]);
        p[23] = static_cast<float>(src_smem[4 * SMEM_LENGTH_X + baseOffset + 3]);
        p[24] = static_cast<float>(src_smem[4 * SMEM_LENGTH_X + baseOffset + 4]);

        // Apply sorting network
        median_f8->f1[px] = apply_sorting_network_5x5(p);
    }
}

template <>
__device__ void median_filter_5x5_row_hip_compute<float>(float* src_smem, d_float8* median_f8) {
    float* row0Ptr = src_smem;
    float* row1Ptr = row0Ptr + SMEM_LENGTH_X;
    float* row2Ptr = row1Ptr + SMEM_LENGTH_X;
    float* row3Ptr = row2Ptr + SMEM_LENGTH_X;
    float* row4Ptr = row3Ptr + SMEM_LENGTH_X;

    for (int px = 0; px < 8; ++px) {
        // Load 5×5 window
        float p[25];
        p[0] = row0Ptr[px];
        p[1] = row0Ptr[px + 1];
        p[2] = row0Ptr[px + 2];
        p[3] = row0Ptr[px + 3];
        p[4] = row0Ptr[px + 4];
        p[5] = row1Ptr[px];
        p[6] = row1Ptr[px + 1];
        p[7] = row1Ptr[px + 2];
        p[8] = row1Ptr[px + 3];
        p[9] = row1Ptr[px + 4];
        p[10] = row2Ptr[px];
        p[11] = row2Ptr[px + 1];
        p[12] = row2Ptr[px + 2];
        p[13] = row2Ptr[px + 3];
        p[14] = row2Ptr[px + 4];
        p[15] = row3Ptr[px];
        p[16] = row3Ptr[px + 1];
        p[17] = row3Ptr[px + 2];
        p[18] = row3Ptr[px + 3];
        p[19] = row3Ptr[px + 4];
        p[20] = row4Ptr[px];
        p[21] = row4Ptr[px + 1];
        p[22] = row4Ptr[px + 2];
        p[23] = row4Ptr[px + 3];
        p[24] = row4Ptr[px + 4];

        // Apply sorting network
        median_f8->f1[px] = apply_sorting_network_5x5(p);
    }
}

// Quick-Select based median computation
template <int kernelSize, typename T>
__device__ __forceinline__ float compute_median_quickselect(T* window) {
    constexpr int windowSize = kernelSize * kernelSize;
    constexpr int medianIndex = windowSize / 2;

    int leftIdx = 0;
    int rightIdx = windowSize - 1;

    // Quick-Select partitioning algorithm with proper duplicate handling
    while (leftIdx < rightIdx) {
        // Choose pivot using median-of-3 (first, mid, last)
        int midIdx = (leftIdx + rightIdx) >> 1;
        float3 val_f3 =
            make_float3(static_cast<float>(window[leftIdx]), static_cast<float>(window[midIdx]),
                        static_cast<float>(window[rightIdx]));
        T pivotVal = static_cast<T>(rpp_hip_median3(val_f3));

        // Three-way partition (Dutch National Flag) to handle duplicates
        int lt = leftIdx;   // window[leftIdx..lt-1] < pivot
        int gt = rightIdx;  // window[gt+1..rightIdx] > pivot
        int i = leftIdx;    // window[lt..i-1] == pivot, window[i..gt] unexamined

        while (i <= gt) {
            if (window[i] < pivotVal) {
                T tmp = window[lt];
                window[lt] = window[i];
                window[i] = tmp;
                ++lt;
                ++i;
            } else if (window[i] > pivotVal) {
                T tmp = window[gt];
                window[gt] = window[i];
                window[i] = tmp;
                --gt;
            } else {
                ++i;
            }
        }

        // After partition: window[leftIdx..lt-1] < pivot, window[lt..gt] == pivot,
        // window[gt+1..rightIdx] > pivot Shrink search interval toward median position
        if (medianIndex < lt)
            rightIdx = lt - 1;
        else if (medianIndex > gt)
            leftIdx = gt + 1;
        else
            break;  // medianIndex is in the equal-to-pivot region
    }

    return static_cast<float>(window[medianIndex]);
}

// Histogram-based median computation using shared memory (for U8 types with large kernels)
template <int kernelSize>
__device__ __forceinline__ float compute_median_histogram(uchar* window, int* hist_smem) {
    constexpr int windowSize = kernelSize * kernelSize;
    constexpr int medianIndex = windowSize / 2;

    // Clear histogram
    for (int i = 0; i < 256; i++) hist_smem[i] = 0;

    // Build histogram - count occurrences of each value
    for (int i = 0; i < windowSize; ++i) hist_smem[window[i]]++;

    // Find median by accumulating counts
    int count = 0;
    for (int v = 0; v < 256; ++v) {
        count += hist_smem[v];
        if (count > medianIndex) return static_cast<float>(v);
    }
    return 0.0f;
}

template <int kernelSize, typename T>
__device__ __forceinline__ float compute_median(T* window, int* hist_smem = nullptr) {
    // Use histogram method for U8 types with large kernels (7×7, 9×9) when shared memory is
    // available
    if constexpr (std::is_same<T, uchar>::value && kernelSize >= 7) {
        if (hist_smem != nullptr) return compute_median_histogram<kernelSize>(window, hist_smem);
    }
    // Fall back to Quick-Select for other types or when no shared memory provided
    return compute_median_quickselect<kernelSize>(window);
}

template <int kernelSize, typename T>
__device__ void median_filter_row_hip_compute(T* srcPtr, d_float8* median_f8) {
    const int paddedKernelWidth = kernelSize + 7;
    const int windowSize = kernelSize * kernelSize;

    // Handle differently for float vs integer types
    if constexpr (std::is_same<T, float>::value) {
        // Float path - direct array access
        float src[kernelSize * paddedKernelWidth];

        for (int i = 0; i < kernelSize; ++i) {
            float* srcRow = srcPtr + i * SMEM_LENGTH_X;
#pragma unroll
            for (int j = 0; j < paddedKernelWidth; ++j) src[i * paddedKernelWidth + j] = srcRow[j];
        }

        for (int filter = 0; filter < 8; ++filter) {
            float window[windowSize];
            for (int k = 0; k < windowSize; ++k) {
                int row = k / kernelSize;
                int col = k % kernelSize;
                window[k] = src[row * paddedKernelWidth + filter + col];
            }
            median_f8->f1[filter] = compute_median<kernelSize>(window);
        }
    } else {
        // Integer types path - unpack from uint32_t
        using bufType =
            typename std::conditional<std::is_same<T, uchar>::value, uint32_t, int32_t>::type;
        const int loadCountPerRow = (kernelSize + 10) / 4;

        T src[kernelSize * paddedKernelWidth];

        for (int i = 0; i < kernelSize; ++i) {
            bufType* srcPtrRowUint = (bufType*)(srcPtr + i * SMEM_LENGTH_X);
            for (int j = 0; j < loadCountPerRow; ++j) {
                bufType val = srcPtrRowUint[j];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int posInRow = (j << 2) + k;
                    if (posInRow >= paddedKernelWidth) break;
                    src[i * paddedKernelWidth + posInRow] =
                        static_cast<T>((val >> (k << 3)) & 0xFF);
                }
            }
        }

        for (int filter = 0; filter < 8; ++filter) {
            T window[windowSize];
            for (int i = 0; i < windowSize; ++i)
                window[i] = src[(i / kernelSize) * paddedKernelWidth + filter + (i % kernelSize)];

            median_f8->f1[filter] = compute_median<kernelSize>(window);
        }
    }
}

template <typename T>
__global__ void median_filter_3x3_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                 uint2 dstStridesNH, uint padLength, uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float24 median_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Nearest-neighbor padding
        T tempBuffer[24];  // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];          // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1];  // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2];  // B
        }

        // Use helper function to load padded data into shared memory
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                          &median_f24.f8[0]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                          &median_f24.f8[1]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                          &median_f24.f8[2]);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void median_filter_5x5_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                 uint2 dstStridesNH, uint padLength, uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float24 median_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Nearest-neighbor padding
        T tempBuffer[24];  // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];          // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1];  // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2];  // B
        }

        // Use helper function to load padded data into shared memory
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                          &median_f24.f8[0]);
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                          &median_f24.f8[1]);
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                          &median_f24.f8[2]);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void median_filter_7x7_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                 uint2 dstStridesNH, uint padLength, uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float24 median_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Nearest-neighbor padding
        T tempBuffer[24];  // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];          // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1];  // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2];  // B
        }

        // Use helper function to load padded data into shared memory
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                         &median_f24.f8[0]);
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                         &median_f24.f8[1]);
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                         &median_f24.f8[2]);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void median_filter_9x9_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                 uint2 dstStridesNH, uint padLength, uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float24 median_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Nearest-neighbor padding
        T tempBuffer[24];  // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];          // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1];  // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2];  // B
        }

        // Use helper function to load padded data into shared memory
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                         &median_f24.f8[0]);
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                         &median_f24.f8[1]);
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                         &median_f24.f8[2]);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 3
template <typename T>
__global__ void median_filter_3x3_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                 uint3 dstStridesNCH, int channelsDst,
                                                 uint padLength, uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Compute input pixel coordinates with edge replication
    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float8 median_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                         &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        T tempBuffer[8];  // Temporary storage for 8 pixels
        for (int i = 0; i < 8; i++) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
        FilterDispatch<T>::rpp_hip_load8(
            tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
    }

    // Process the remaining 2 channels for RGB images
    if (channelsDst == 3) {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        } else {
            // Nearest-neighbor padding
            T tempBuffer[8];  // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++) {
                int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                               max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + srcStridesNCH.y +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            FilterDispatch<T>::rpp_hip_load8(
                tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],
                                              &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        } else {
            // Nearest-neighbor padding
            T tempBuffer[8];  // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++) {
                int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                               max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            FilterDispatch<T>::rpp_hip_load8(
                tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],
                                              &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }
    }
}

// kernelSize = 5
template <typename T>
__global__ void median_filter_5x5_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                 uint3 dstStridesNCH, int channelsDst,
                                                 uint padLength, uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float8 median_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                         &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        T tempBuffer[8];  // Temporary storage for 8 pixels
        for (int i = 0; i < 8; i++) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
        FilterDispatch<T>::rpp_hip_load8(
            tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
    }

    if (channelsDst == 3) {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        } else {
            // Nearest-neighbor padding
            T tempBuffer[8];  // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++) {
                int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                               max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + srcStridesNCH.y +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            FilterDispatch<T>::rpp_hip_load8(
                tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],
                                              &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        } else {
            // Nearest-neighbor padding
            T tempBuffer[8];  // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++) {
                int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                               max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            FilterDispatch<T>::rpp_hip_load8(
                tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],
                                              &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }
    }
}

// kernelSize = 7
template <typename T>
__global__ void median_filter_7x7_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                 uint3 dstStridesNCH, int channelsDst,
                                                 uint padLength, uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float8 median_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                         &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        T tempBuffer[8];  // Temporary storage for 8 pixels
        for (int i = 0; i < 8; i++) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
        FilterDispatch<T>::rpp_hip_load8(
            tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
    }

    if (channelsDst == 3) {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        } else {
            // Nearest-neighbor padding
            T tempBuffer[8];  // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++) {
                int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                               max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            FilterDispatch<T>::rpp_hip_load8(
                tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],
                                             &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        } else {
            // Nearest-neighbor padding
            T tempBuffer[8];  // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++) {
                int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                               max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            FilterDispatch<T>::rpp_hip_load8(
                tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],
                                             &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }
    }
}

// kernelSize = 9
template <typename T>
__global__ void median_filter_9x9_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                 uint3 dstStridesNCH, int channelsDst,
                                                 uint padLength, uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float8 median_f8;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                         &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        T tempBuffer[8];  // Temporary storage for 8 pixels
        for (int i = 0; i < 8; i++) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
        FilterDispatch<T>::rpp_hip_load8(
            tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
    }

    if (channelsDst == 3) {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        } else {
            // Nearest-neighbor padding
            T tempBuffer[8];  // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++) {
                int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                               max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            FilterDispatch<T>::rpp_hip_load8(
                tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],
                                             &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx,
                                             &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        } else {
            // Nearest-neighbor padding
            T tempBuffer[8];  // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++) {
                int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                               max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) +
                                 (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            FilterDispatch<T>::rpp_hip_load8(
                tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y)) {
            median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],
                                             &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }
    }
}

// -------------------- Set 3 - PKD3->PLN3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void median_filter_3x3_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                       uint3 dstStridesNCH, uint padLength,
                                                       uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float24 median_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Nearest-neighbor padding
        T tempBuffer[24];  // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];          // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1];  // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2];  // B
        }

        // Use helper function to load padded data into shared memory
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                          &median_f24.f8[0]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                          &median_f24.f8[1]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                          &median_f24.f8[2]);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &median_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void median_filter_5x5_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                       uint3 dstStridesNCH, uint padLength,
                                                       uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float24 median_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Nearest-neighbor padding
        T tempBuffer[24];  // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];          // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1];  // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2];  // B
        }

        // Use helper function to load padded data into shared memory
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                          &median_f24.f8[0]);
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                          &median_f24.f8[1]);
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                          &median_f24.f8[2]);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &median_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void median_filter_7x7_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                       uint3 dstStridesNCH, uint padLength,
                                                       uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float24 median_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Nearest-neighbor padding
        T tempBuffer[24];  // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];          // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1];  // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2];  // B
        }

        // Use helper function to load padded data into shared memory
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                         &median_f24.f8[0]);
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                         &median_f24.f8[1]);
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                         &median_f24.f8[2]);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &median_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void median_filter_9x9_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                                       uint3 dstStridesNCH, uint padLength,
                                                       uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float24 median_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) +
                 ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                 ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType* src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
    } else {
        // Nearest-neighbor padding
        T tempBuffer[24];  // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];          // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1];  // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2];  // B
        }

        // Use helper function to load padded data into shared memory
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                         &median_f24.f8[0]);
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                         &median_f24.f8[1]);
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                         &median_f24.f8[2]);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &median_f24);
    }
}

// -------------------- Set 4 - PLN3->PKD3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void median_filter_3x3_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                       uint2 dstStridesNH, uint padLength,
                                                       uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float24 median_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) +
               ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
               (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.x,
                                         &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.y,
                                         &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.z,
                                         &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        T tempBuffer0[8], tempBuffer1[8], tempBuffer2[8];

        for (int i = 0; i < 8; i++) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx0 = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            int clampedIdx1 = clampedIdx0 + srcStridesNCH.y;
            int clampedIdx2 = clampedIdx1 + srcStridesNCH.y;

            tempBuffer0[i] = srcPtr[clampedIdx0];
            tempBuffer1[i] = srcPtr[clampedIdx1];
            tempBuffer2[i] = srcPtr[clampedIdx2];
        }

        FilterDispatch<T>::rpp_hip_load8(tempBuffer0,
                                         &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer1,
                                         &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer2,
                                         &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                          &median_f24.f8[0]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                          &median_f24.f8[1]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                          &median_f24.f8[2]);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void median_filter_5x5_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                       uint2 dstStridesNH, uint padLength,
                                                       uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float24 median_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) +
               ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
               (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.x,
                                         &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.y,
                                         &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.z,
                                         &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        T tempBuffer0[8], tempBuffer1[8], tempBuffer2[8];

        for (int i = 0; i < 8; i++) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx0 = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            int clampedIdx1 = clampedIdx0 + srcStridesNCH.y;
            int clampedIdx2 = clampedIdx1 + srcStridesNCH.y;

            tempBuffer0[i] = srcPtr[clampedIdx0];
            tempBuffer1[i] = srcPtr[clampedIdx1];
            tempBuffer2[i] = srcPtr[clampedIdx2];
        }

        FilterDispatch<T>::rpp_hip_load8(tempBuffer0,
                                         &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer1,
                                         &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer2,
                                         &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                          &median_f24.f8[0]);
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                          &median_f24.f8[1]);
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                          &median_f24.f8[2]);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void median_filter_7x7_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                       uint2 dstStridesNH, uint padLength,
                                                       uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float24 median_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) +
               ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
               (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.x,
                                         &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.y,
                                         &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.z,
                                         &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        T tempBuffer0[8], tempBuffer1[8], tempBuffer2[8];

        for (int i = 0; i < 8; i++) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx0 = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            int clampedIdx1 = clampedIdx0 + srcStridesNCH.y;
            int clampedIdx2 = clampedIdx1 + srcStridesNCH.y;

            tempBuffer0[i] = srcPtr[clampedIdx0];
            tempBuffer1[i] = srcPtr[clampedIdx1];
            tempBuffer2[i] = srcPtr[clampedIdx2];
        }

        FilterDispatch<T>::rpp_hip_load8(tempBuffer0,
                                         &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer1,
                                         &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer2,
                                         &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                         &median_f24.f8[0]);
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                         &median_f24.f8[1]);
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                         &median_f24.f8[2]);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void median_filter_9x9_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                                       uint2 dstStridesNH, uint padLength,
                                                       uint2 tileSize, RpptROIPtr roiTensorPtrSrc) {
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    int clampedY = roiTensorPtrSrc[id_z].xywhROI.xy.y +
                   max(0, min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

    d_float24 median_f24;
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) +
               ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
               (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= 0) && ((id_x_i + 7) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) &&
        (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.x,
                                         &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.y,
                                         &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.z,
                                         &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } else {
        // Nearest-neighbor padding
        T tempBuffer0[8], tempBuffer1[8], tempBuffer2[8];

        for (int i = 0; i < 8; i++) {
            int clampedX = roiTensorPtrSrc[id_z].xywhROI.xy.x +
                           max(0, min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedIdx0 = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            int clampedIdx1 = clampedIdx0 + srcStridesNCH.y;
            int clampedIdx2 = clampedIdx1 + srcStridesNCH.y;

            tempBuffer0[i] = srcPtr[clampedIdx0];
            tempBuffer1[i] = srcPtr[clampedIdx1];
            tempBuffer2[i] = srcPtr[clampedIdx2];
        }

        FilterDispatch<T>::rpp_hip_load8(tempBuffer0,
                                         &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer1,
                                         &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer2,
                                         &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y)) {
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                         &median_f24.f8[0]);
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                         &median_f24.f8[1]);
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
                                         &median_f24.f8[2]);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// -------------------- Set 5 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_median_filter_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                        RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                        RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                        rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB) hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + kernelSize + 7) >> 3;
    int globalThreads_y = dstDescPtr->h + kernelSize;
    int globalThreads_z = handle.GetBatchSize();

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (128 - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC)) {
        globalThreads_x = ((dstDescPtr->strides.hStride + kernelSize) / 3 + 7) >> 3;

        using PkdKernelType = void (*)(T*, uint2, T*, uint2, uint, uint2, RpptROIPtr);
        PkdKernelType kernelFn = nullptr;
        switch (kernelSize) {
            case 3:
                kernelFn = median_filter_3x3_pkd_hip_tensor<T>;
                break;
            case 5:
                kernelFn = median_filter_5x5_pkd_hip_tensor<T>;
                break;
            case 7:
                kernelFn = median_filter_7x7_pkd_hip_tensor<T>;
                break;
            case 9:
                kernelFn = median_filter_9x9_pkd_hip_tensor<T>;
                break;
            default:
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        hipLaunchKernelGGL(
            kernelFn,
            dim3(ceil((float)globalThreads_x / tileSize.x),
                 ceil((float)globalThreads_y / tileSize.y),
                 ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(), srcPtr,
            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride), dstPtr,
            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride), padLength,
            tileSize, roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        using PlnKernelType = void (*)(T*, uint3, T*, uint3, int, uint, uint2, RpptROIPtr);
        PlnKernelType kernelFn = nullptr;
        switch (kernelSize) {
            case 3:
                kernelFn = median_filter_3x3_pln_hip_tensor<T>;
                break;
            case 5:
                kernelFn = median_filter_5x5_pln_hip_tensor<T>;
                break;
            case 7:
                kernelFn = median_filter_7x7_pln_hip_tensor<T>;
                break;
            case 9:
                kernelFn = median_filter_9x9_pln_hip_tensor<T>;
                break;
            default:
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        hipLaunchKernelGGL(kernelFn,
                           dim3(ceil((float)globalThreads_x / tileSize.x),
                                ceil((float)globalThreads_y / tileSize.y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                      srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           dstDescPtr->c, padLength, tileSize, roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3)) {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            using Pkd3Pln3KernelType = void (*)(T*, uint2, T*, uint3, uint, uint2, RpptROIPtr);
            Pkd3Pln3KernelType kernelFn = nullptr;
            switch (kernelSize) {
                case 3:
                    kernelFn = median_filter_3x3_pkd3_pln3_hip_tensor<T>;
                    break;
                case 5:
                    kernelFn = median_filter_5x5_pkd3_pln3_hip_tensor<T>;
                    break;
                case 7:
                    kernelFn = median_filter_7x7_pkd3_pln3_hip_tensor<T>;
                    break;
                case 9:
                    kernelFn = median_filter_9x9_pkd3_pln3_hip_tensor<T>;
                    break;
                default:
                    return RPP_ERROR_NOT_IMPLEMENTED;
            }
            hipLaunchKernelGGL(kernelFn,
                               dim3(ceil((float)globalThreads_x / tileSize.x),
                                    ceil((float)globalThreads_y / tileSize.y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               padLength, tileSize, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;

            using Pln3Pkd3KernelType = void (*)(T*, uint3, T*, uint2, uint, uint2, RpptROIPtr);
            Pln3Pkd3KernelType kernelFn = nullptr;
            switch (kernelSize) {
                case 3:
                    kernelFn = median_filter_3x3_pln3_pkd3_hip_tensor<T>;
                    break;
                case 5:
                    kernelFn = median_filter_5x5_pln3_pkd3_hip_tensor<T>;
                    break;
                case 7:
                    kernelFn = median_filter_7x7_pln3_pkd3_hip_tensor<T>;
                    break;
                case 9:
                    kernelFn = median_filter_9x9_pln3_pkd3_hip_tensor<T>;
                    break;
                default:
                    return RPP_ERROR_NOT_IMPLEMENTED;
            }
            hipLaunchKernelGGL(kernelFn,
                               dim3(ceil((float)globalThreads_x / tileSize.x),
                                    ceil((float)globalThreads_y / tileSize.y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength, tileSize, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    }

    return RPP_SUCCESS;
}

// -------------------- Set 5 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_median_filter_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                              RpptDescPtr dstDescPtr, Rpp32u kernelSize,
                                              RpptROIPtr roiPtrSrc, RpptRoiType roiType,
                                              rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB) hip_exec_roi_conversion_ltrb_to_xywh(roiPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + kernelSize + 7) >> 3;
    int globalThreads_y = dstDescPtr->h + kernelSize;
    int globalThreads_z = 1;

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (128 - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC)) {
        globalThreads_x = ((dstDescPtr->strides.hStride + kernelSize) / 3 + 7) >> 3;

        using PkdKernelType = void (*)(T*, uint2, T*, uint2, uint, uint2, RpptROIPtr);
        PkdKernelType kernelFn = nullptr;
        switch (kernelSize) {
            case 3:
                kernelFn = median_filter_3x3_pkd_hip_tensor<T>;
                break;
            case 5:
                kernelFn = median_filter_5x5_pkd_hip_tensor<T>;
                break;
            case 7:
                kernelFn = median_filter_7x7_pkd_hip_tensor<T>;
                break;
            case 9:
                kernelFn = median_filter_9x9_pkd_hip_tensor<T>;
                break;
            default:
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        hipLaunchKernelGGL(
            kernelFn,
            dim3(ceil((float)globalThreads_x / tileSize.x),
                 ceil((float)globalThreads_y / tileSize.y),
                 ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(), srcPtr,
            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride), dstPtr,
            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride), padLength,
            tileSize, roiPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        using PlnKernelType = void (*)(T*, uint3, T*, uint3, int, uint, uint2, RpptROIPtr);
        PlnKernelType kernelFn = nullptr;
        switch (kernelSize) {
            case 3:
                kernelFn = median_filter_3x3_pln_hip_tensor<T>;
                break;
            case 5:
                kernelFn = median_filter_5x5_pln_hip_tensor<T>;
                break;
            case 7:
                kernelFn = median_filter_7x7_pln_hip_tensor<T>;
                break;
            case 9:
                kernelFn = median_filter_9x9_pln_hip_tensor<T>;
                break;
            default:
                return RPP_ERROR_NOT_IMPLEMENTED;
        }
        hipLaunchKernelGGL(kernelFn,
                           dim3(ceil((float)globalThreads_x / tileSize.x),
                                ceil((float)globalThreads_y / tileSize.y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                      srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           dstDescPtr->c, padLength, tileSize, roiPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3)) {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            using Pkd3Pln3KernelType = void (*)(T*, uint2, T*, uint3, uint, uint2, RpptROIPtr);
            Pkd3Pln3KernelType kernelFn = nullptr;
            switch (kernelSize) {
                case 3:
                    kernelFn = median_filter_3x3_pkd3_pln3_hip_tensor<T>;
                    break;
                case 5:
                    kernelFn = median_filter_5x5_pkd3_pln3_hip_tensor<T>;
                    break;
                case 7:
                    kernelFn = median_filter_7x7_pkd3_pln3_hip_tensor<T>;
                    break;
                case 9:
                    kernelFn = median_filter_9x9_pkd3_pln3_hip_tensor<T>;
                    break;
                default:
                    return RPP_ERROR_NOT_IMPLEMENTED;
            }
            hipLaunchKernelGGL(kernelFn,
                               dim3(ceil((float)globalThreads_x / tileSize.x),
                                    ceil((float)globalThreads_y / tileSize.y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               padLength, tileSize, roiPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;

            using Pln3Pkd3KernelType = void (*)(T*, uint3, T*, uint2, uint, uint2, RpptROIPtr);
            Pln3Pkd3KernelType kernelFn = nullptr;
            switch (kernelSize) {
                case 3:
                    kernelFn = median_filter_3x3_pln3_pkd3_hip_tensor<T>;
                    break;
                case 5:
                    kernelFn = median_filter_5x5_pln3_pkd3_hip_tensor<T>;
                    break;
                case 7:
                    kernelFn = median_filter_7x7_pln3_pkd3_hip_tensor<T>;
                    break;
                case 9:
                    kernelFn = median_filter_9x9_pln3_pkd3_hip_tensor<T>;
                    break;
                default:
                    return RPP_ERROR_NOT_IMPLEMENTED;
            }
            hipLaunchKernelGGL(kernelFn,
                               dim3(ceil((float)globalThreads_x / tileSize.x),
                                    ceil((float)globalThreads_y / tileSize.y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength, tileSize, roiPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_median_filter_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                        Rpp32u, RpptROIPtr, RpptRoiType,
                                                        rpp::Handle&);

template RppStatus hip_exec_median_filter_tensor<half>(half*, RpptDescPtr, half*, RpptDescPtr,
                                                       Rpp32u, RpptROIPtr, RpptRoiType,
                                                       rpp::Handle&);

template RppStatus hip_exec_median_filter_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                         Rpp32u, RpptROIPtr, RpptRoiType,
                                                         rpp::Handle&);

template RppStatus hip_exec_median_filter_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                        Rpp32u, RpptROIPtr, RpptRoiType,
                                                        rpp::Handle&);

template RppStatus hip_exec_median_filter_single_image<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*,
                                                              RpptDescPtr, Rpp32u, RpptROIPtr,
                                                              RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_median_filter_single_image<half>(half*, RpptDescPtr, half*, RpptDescPtr,
                                                             Rpp32u, RpptROIPtr, RpptRoiType,
                                                             rpp::Handle&);

template RppStatus hip_exec_median_filter_single_image<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*,
                                                               RpptDescPtr, Rpp32u, RpptROIPtr,
                                                               RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_median_filter_single_image<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*,
                                                              RpptDescPtr, Rpp32u, RpptROIPtr,
                                                              RpptRoiType, rpp::Handle&);
