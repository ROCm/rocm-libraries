/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef ROCWMMA_TEST_DEVICE_COMMON_HPP
#define ROCWMMA_TEST_DEVICE_COMMON_HPP

#include <rocwmma/internal/types.hpp>
#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    template <typename T>
    __device__ inline float64_t toDouble(T const& val)
    {
        // TODO: deprecate. Not needed anymore because bfloat16_t has float operator
        return static_cast<double>(val);
    }

    __device__ inline uint32_t rowMjr(uint32_t row, uint32_t col, uint32_t ld)
    {
        return row * ld + col;
    }

    __device__ inline uint32_t colMjr(uint32_t row, uint32_t col, uint32_t ld)
    {
        return col * ld + row;
    }

    __device__ inline float64_t maxDouble(float64_t a, float64_t b)
    {
        if(std::isinf(a) || std::isinf(b))
        {
            return std::numeric_limits<float64_t>::infinity();
        }
        // Check for NaN
        else if(std::isnan(a) || std::isnan(b))
        {
            return std::numeric_limits<float64_t>::signaling_NaN();
        }
        return a > b ? a : b;
    }

    __global__ static void maxReduceKernel(float64_t* relativeError,
                                           uint32_t   elements,
                                           uint32_t   offset,
                                           uint32_t   maxIdx)
    {
        float64_t* localRelativeError = relativeError + (offset * elements * blockIdx.x);

        for(int i = elements >> 1; i > 0; i = i >> 1)
        {
            if(threadIdx.x < i && offset * (elements * blockIdx.x + threadIdx.x + i) < maxIdx)
            {
                localRelativeError[offset * threadIdx.x]
                    = maxDouble(localRelativeError[offset * threadIdx.x],
                                localRelativeError[offset * (threadIdx.x + i)]);
            }
            synchronize_workgroup();
        }
    }

    // Comparitive kernel for batched matrix outputs as used in gemm tests
    // Compares all values of two M x N matrices
    template <typename TypeA, typename TypeB, typename LayoutA, typename LayoutB>
    __global__ void compareEqualKernel(TypeA*     matrixA,
                                       TypeB*     matrixB,
                                       float64_t* relativeError,
                                       uint32_t   m,
                                       uint32_t   n,
                                       uint32_t   lda,
                                       uint32_t   ldb)
    {
        uint32_t errorIdx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t colIdx   = errorIdx % n;
        uint32_t rowIdx   = errorIdx / n;

        uint32_t indexA = std::is_same<LayoutA, row_major>::value ? rowMjr(rowIdx, colIdx, lda)
                                                                  : colMjr(rowIdx, colIdx, lda);
        uint32_t indexB = std::is_same<LayoutB, row_major>::value ? rowMjr(rowIdx, colIdx, ldb)
                                                                  : colMjr(rowIdx, colIdx, ldb);

        if(rowIdx < m && colIdx < n)
        {
            TypeA valA = matrixA[indexA];
            TypeB valB = matrixB[indexB];

            // Determine relative error for each element of matrix A/B
            auto numerator = fabs(toDouble(valA) - toDouble(valB));
            auto divisor   = fabs(toDouble(valA)) + fabs(toDouble(valB)) + 1.0;
            if(std::isinf(numerator) || std::isinf(divisor))
            {
                relativeError[errorIdx] = std::numeric_limits<float64_t>::infinity();
            }
            else if(std::isnan(numerator) || std::isnan(divisor))
            {
                relativeError[errorIdx] = std::numeric_limits<float64_t>::signaling_NaN();
            }
            else
            {
                relativeError[errorIdx] = numerator / divisor;
            }
        }
    }

    // Comparitive kernel for batched matrix outputs as used in DLRM tests
    // Compares all values of two M x N matrices over B batches
    template <typename TypeA, typename TypeB>
    __global__ void compareEqualKernel(TypeA*     matrixA,
                                       TypeB*     matrixB,
                                       float64_t* relativeError,
                                       uint32_t   m,
                                       uint32_t   n,
                                       uint32_t   b)
    {
        uint32_t errorIdx    = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t colIdx      = errorIdx % n;
        uint32_t rowIdx      = errorIdx / n;
        uint32_t matrixIdx   = rowMjr(rowIdx, colIdx, n);
        uint32_t batchOffset = blockIdx.z * m * n;

        if(rowIdx < m && colIdx < n && blockIdx.z < b)
        {
            TypeA valA = matrixA[batchOffset + matrixIdx];
            TypeB valB = matrixB[batchOffset + matrixIdx];

            // Determine relative error for each element of matrix A/B
            auto numerator = fabs(toDouble(valA) - toDouble(valB));
            auto divisor   = fabs(toDouble(valA)) + fabs(toDouble(valB)) + 1.0;
            if(std::isinf(numerator) || std::isinf(divisor))
            {
                relativeError[batchOffset + errorIdx] = std::numeric_limits<float64_t>::infinity();
            }
            else if(std::isnan(numerator) || std::isnan(divisor))
            {
                relativeError[batchOffset + errorIdx]
                    = std::numeric_limits<float64_t>::signaling_NaN();
            }
            else
            {
                relativeError[batchOffset + errorIdx] = numerator / divisor;
            }
        }
    }

    // fill kernel for M x N matrix with padding
    template <typename DataT, typename Layout>
    __global__ void fillWithPaddingKernel(
        DataT* mat, uint32_t m, uint32_t n, uint32_t padM, uint32_t padN, DataT padValue)
    {
        const auto limitM = m + 2 * padM;
        const auto limitN = n + 2 * padN;

        uint32_t rowIdx = (blockIdx.x * blockDim.x + threadIdx.x) / limitN;
        uint32_t colIdx = (blockIdx.x * blockDim.x + threadIdx.x) % limitN;

        auto ld    = std::is_same<Layout, row_major>::value ? limitN : limitM;
        auto index = std::is_same<Layout, row_major>::value ? rowMjr(rowIdx, colIdx, ld)
                                                            : colMjr(rowIdx, colIdx, ld);

        if(rowIdx < limitM && colIdx < limitN)
        {
            // fill padding
            if(rowIdx < padM || rowIdx >= (limitM - padM) || colIdx < padN
               || colIdx >= (limitN - padN))
            {
                mat[index] = padValue;
            }
            // fill interior
            else
            {
                auto value = ((rowIdx - padM) * n + (colIdx - padN)) % 5;
                mat[index] = ((value % 5) && std::is_signed<DataT>::value)
                                 ? -static_cast<DataT>(value)
                                 : static_cast<DataT>(value);
            }
        }
    }

    // fill kernel for M x N matrix
    template <typename DataT, typename Layout>
    __global__ void fillKernel(DataT* mat, uint32_t m, uint32_t n)
    {
        uint32_t rowIdx = (blockIdx.x * blockDim.x + threadIdx.x) / n;
        uint32_t colIdx = (blockIdx.x * blockDim.x + threadIdx.x) % n;

        auto ld    = std::is_same<Layout, row_major>::value ? n : m;
        auto index = std::is_same<Layout, row_major>::value ? rowMjr(rowIdx, colIdx, ld)
                                                            : colMjr(rowIdx, colIdx, ld);

        if(rowIdx < m && colIdx < n)
        {
            auto value = (rowIdx * n + colIdx) % 3;
            mat[index] = ((value % 3) && std::is_signed<DataT>::value) ? -static_cast<DataT>(value)
                                                                       : static_cast<DataT>(value);
        }
    }

    // fill kernel for batched M x K matrices
    template <typename DataT, typename Layout>
    __global__ void fillKernel(DataT* mat, uint32_t m, uint32_t k, uint32_t b)
    {
        uint32_t rowIdx      = (blockIdx.x * blockDim.x + threadIdx.x) / k;
        uint32_t colIdx      = (blockIdx.x * blockDim.x + threadIdx.x) % k;
        uint32_t batchOffset = m * k * blockIdx.z;

        auto ld    = std::is_same<Layout, row_major>::value ? k : m;
        auto index = std::is_same<Layout, row_major>::value ? rowMjr(rowIdx, colIdx, ld)
                                                            : colMjr(rowIdx, colIdx, ld);
        index += batchOffset;

        if(rowIdx < m && colIdx < k)
        {
            auto value = (rowIdx * k + colIdx) % 5;
            mat[index] = ((value % 3) && std::is_signed<DataT>::value) ? -static_cast<DataT>(value)
                                                                       : static_cast<DataT>(value);
        }
    }

    // fill kernel for batched M x K matrices for a specific value
    template <typename DataT, typename Layout>
    __global__ void fillKernel(DataT* mat, uint32_t m, uint32_t k, uint32_t b, DataT value)
    {
        uint32_t rowIdx      = (blockIdx.x * blockDim.x + threadIdx.x) / k;
        uint32_t colIdx      = (blockIdx.x * blockDim.x + threadIdx.x) % k;
        uint32_t batchOffset = m * k * blockIdx.z;

        auto ld    = std::is_same<Layout, row_major>::value ? k : m;
        auto index = std::is_same<Layout, row_major>::value ? rowMjr(rowIdx, colIdx, ld)
                                                            : colMjr(rowIdx, colIdx, ld);
        index += batchOffset;

        if(rowIdx < m && colIdx < k)
        {
            mat[index] = value;
        }
    }

    // fill kernel for M x N matrix for a specific value
    template <typename DataT, typename Layout>
    __global__ void fillValKernel(DataT* mat, uint32_t m, uint32_t n, DataT value)
    {
        uint32_t rowIdx = (blockIdx.x * blockDim.x + threadIdx.x) / n;
        uint32_t colIdx = (blockIdx.x * blockDim.x + threadIdx.x) % n;

        auto ld    = std::is_same<Layout, row_major>::value ? n : m;
        auto index = std::is_same<Layout, row_major>::value ? rowMjr(rowIdx, colIdx, ld)
                                                            : colMjr(rowIdx, colIdx, ld);

        if(rowIdx < m && colIdx < n)
        {
            mat[index] = value;
        }
    }

    // fill kernel for M x N matrix for a mat[i] = i
    template <typename DataT, typename Layout>
    __global__ void fillIdxKernel(DataT* mat, uint32_t m, uint32_t n)
    {
        uint32_t rowIdx = (blockIdx.x * blockDim.x + threadIdx.x) / n;
        uint32_t colIdx = (blockIdx.x * blockDim.x + threadIdx.x) % n;

        auto ld    = std::is_same<Layout, row_major>::value ? n : m;
        auto index = std::is_same<Layout, row_major>::value ? rowMjr(rowIdx, colIdx, ld)
                                                            : colMjr(rowIdx, colIdx, ld);

        if(rowIdx < m && colIdx < n)
        {
            mat[index] = index % 64;
        }
    }
} // namespace rocwmma

#endif // ROCWMMA_TEST_DEVICE_COMMON_HPP
