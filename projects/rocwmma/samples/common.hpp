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

#ifndef ROCWMMA_SAMPLES_COMMON_HPP
#define ROCWMMA_SAMPLES_COMMON_HPP

#include <iostream>
#include <mutex>

// Helper macro for HIP errors
#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(expression)                      \
    if(auto status = (expression); status != hipSuccess) \
    {                                                    \
        fprintf(stderr,                                  \
                "hip error: '%s'(%d) at %s:%d\n",        \
                hipGetErrorString(status),               \
                status,                                  \
                __FILE__,                                \
                __LINE__);                               \
        exit(EXIT_FAILURE);                              \
    }
#endif

#ifndef CHECK_HIPRTC_ERROR
#define CHECK_HIPRTC_ERROR(expression)                       \
    if(auto status = (expression); status != HIPRTC_SUCCESS) \
    {                                                        \
        fprintf(stderr,                                      \
                "hipRTC error: '%s'(%d) at %s:%d\n",         \
                hiprtcGetErrorString(status),                \
                status,                                      \
                __FILE__,                                    \
                __LINE__);                                   \
        exit(EXIT_FAILURE);                                  \
    }
#endif

#include <rocwmma/internal/type_traits.hpp>

// HIP Host functions to determine the gfx architecture
bool isGfx9()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return ((deviceName.find("gfx908") != std::string::npos)
            || (deviceName.find("gfx90a") != std::string::npos)
            || (deviceName.find("gfx942") != std::string::npos)
            || (deviceName.find("gfx950") != std::string::npos));
}

bool isGfx11()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return ((deviceName.find("gfx1100") != std::string::npos)
            || (deviceName.find("gfx1101") != std::string::npos)
            || (deviceName.find("gfx1102") != std::string::npos)
            || (deviceName.find("gfx1150") != std::string::npos)
            || (deviceName.find("gfx1151") != std::string::npos));
}

bool isGfx12()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return ((deviceName.find("gfx1200") != std::string::npos)
            || (deviceName.find("gfx1201") != std::string::npos));
}

// HIP Host function to find if the device supports f64
bool isF64Supported()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return ((deviceName.find("gfx90a") != std::string::npos)
            || (deviceName.find("gfx942") != std::string::npos)
            || (deviceName.find("gfx950") != std::string::npos));
}

bool isF32Supported()
{
    return isGfx9();
}

inline double calculateGFlops(uint32_t m, uint32_t n, uint32_t k)
{
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) * 1.0e-9;
}

inline double calculateTFlopsPerSec(
    uint32_t m, uint32_t n, uint32_t k, double elapsedTimeMs, uint32_t repeats = 1u)
{
    // elapsedTimeMs is over all iterations
    return calculateGFlops(m, n, k) / elapsedTimeMs * static_cast<double>(repeats);
}

// HIP Host function to retrieve the warp size
enum hipWarpSize_t : uint32_t
{
    Wave32 = 32,
    Wave64 = 64,
    UNSUPPORTED_WARP_SIZE,
};

uint32_t getWarpSize()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;
    uint32_t        mWarpSize = hipWarpSize_t::UNSUPPORTED_WARP_SIZE;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    switch(mProps.warpSize)
    {
    case hipWarpSize_t::Wave32:
    case hipWarpSize_t::Wave64:
        mWarpSize = mProps.warpSize;
    default:;
    }

    if(mWarpSize == hipWarpSize_t::UNSUPPORTED_WARP_SIZE)
    {
        std::cerr << "Cannot proceed: unsupported warp sizev detected. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    return mWarpSize;
}

// Batched matrix data initialization
template <typename DataT>
__host__ static inline void
    fill(DataT* mat, uint32_t m, uint32_t k, uint32_t b, uint32_t normalization = 1)
{
    auto batchOffset = m * k;
    for(int t = 0; t < b; ++t)
    {
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < k; ++j)
            {
                // Random values normalized such that output is between 0 and 1
                auto value = __float2half(static_cast<float>(rand() / normalization)
                                          / static_cast<float>(RAND_MAX));

                mat[t * batchOffset + i * k + j] = rocwmma::convert<DataT>(value);
            }
        }
    }
}

template <typename DataT>
__host__ static inline void fillVal(DataT* mat, uint32_t m, uint32_t n, DataT val = 1)
{
    for(int i = 0; i < m * n; ++i)
    {
        mat[i] = val;
    }
}

// Host matrix data random initialization
template <typename DataT>
__host__ static inline void fillRand(DataT* mat, uint32_t m, uint32_t n)
{
    auto randInit = []() {
        srand(time(0));
        return 0u;
    };
    static auto init = randInit();
#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
        auto rando = rand() % 5u;
        for(int j = 0; j < n; j++)
        {
            // Assign random integer values within 0-64, alternating
            // sign if the value is a multiple of 3
            auto value     = (rando + j) % 5u;
            mat[i * n + j] = ((value % 3u == 0u) && std::is_signed<DataT>::value)
                                 ? -static_cast<DataT>(value)
                                 : static_cast<DataT>(value);
        }
    }
}

#include <iomanip>

template <typename DataLayoutT, typename DataT>
__host__ static inline void fillEnc(DataT* mat, uint32_t m, uint32_t n)
{
    using EncT = std::conditional_t<sizeof(DataT) == 2, uint16_t, uint32_t>;
    //#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; j++)
        {
            // Use binary encoding for the row / col coords
            // 0x MMMM NNNN
            EncT enc = ((i & 0xFF) << (sizeof(DataT) * 4)) | (j & 0xFF);
            //std::cout << "row: " << i << " col: " << j << " :";
            //std::cout << "0x" << std::setfill('0') << std::setw(sizeof(DataT)*2) << std::right << std::hex << ((i & 0xFF) << (sizeof(DataT)*4)) << " " << (j & 0xFF) << std::endl;
            //std::cout << "0x" << std::setfill('0') << std::setw(sizeof(DataT)*2) << std::right << std::hex << enc << std::endl;
            auto idx = std::is_same_v<DataLayoutT, rocwmma::row_major> ? (i * n + j) : (i + m * j);
            mat[idx] = reinterpret_cast<DataT&>(enc);
        }
    }
}

template <typename DataLayoutT, typename DataT>
__host__ static inline void printEnc(DataT* mat, uint32_t m, uint32_t n)
{
    using EncT = std::conditional_t<sizeof(DataT) == 2, uint16_t, uint32_t>;
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; j++)
        {
            // Use binary encoding for the row / col coords
            // 0x MMMM NNNN
            auto idx = std::is_same_v<DataLayoutT, rocwmma::row_major> ? (i * n + j) : (i + m * j);
            std::cout << "0x" << std::setfill('0') << std::setw(sizeof(DataT) * 2) << std::right
                      << std::hex << reinterpret_cast<EncT&>(mat[idx]) << " ";
            //std::cout << reinterpret_cast<uint16_t&>(mat[idx]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename DataLayoutT, typename DataT>
__host__ static inline void printData(DataT* mat, uint32_t m, uint32_t n)
{
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; j++)
        {
            // Use binary encoding for the row / col coords
            // 0x MMMM NNNN
            auto idx = std::is_same_v<DataLayoutT, rocwmma::row_major> ? (i * n + j) : (i + m * j);
            std::cout << std::setw(8) << std::right << float(mat[idx]) << " ";
            //std::cout << reinterpret_cast<uint16_t&>(mat[idx]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Host GEMM validation
template <typename InputT,
          typename OutputT,
          typename ComputeT,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename LayoutD = LayoutC>
__host__ void gemm_cpu_h(uint32_t       m,
                         uint32_t       n,
                         uint32_t       k,
                         InputT const*  a,
                         InputT const*  b,
                         OutputT const* c,
                         OutputT*       d,
                         uint32_t       lda,
                         uint32_t       ldb,
                         uint32_t       ldc,
                         uint32_t       ldd,
                         ComputeT       alpha,
                         ComputeT       beta)
{
    auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
    auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

    auto aIndex = std::is_same<LayoutA, rocwmma::row_major>::value ? rowMjr : colMjr;
    auto bIndex = std::is_same<LayoutB, rocwmma::row_major>::value ? rowMjr : colMjr;
    auto cIndex = std::is_same<LayoutC, rocwmma::row_major>::value ? rowMjr : colMjr;
    auto dIndex = std::is_same<LayoutD, rocwmma::row_major>::value ? rowMjr : colMjr;

#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
#pragma omp parallel for
        for(int j = 0; j < n; ++j)
        {
            ComputeT accum = static_cast<ComputeT>(0);
            for(int h = 0; h < k; ++h)
            {
                accum += static_cast<ComputeT>(a[aIndex(i, h, lda)])
                         * static_cast<ComputeT>(b[bIndex(h, j, ldb)]);
            }
            d[dIndex(i, j, ldd)] = static_cast<OutputT>(
                alpha * accum + beta * static_cast<ComputeT>(c[cIndex(i, j, ldc)]));
        }
    }
}

// Element-wise comparison
template <typename DataT>
__host__ std::pair<bool, double>
         compareEqual(DataT const* a, DataT const* b, uint32_t size, double tolerance = 10.0)
{
    bool   retval             = true;
    double max_relative_error = 0.0;

    // Some types don't have direct conversion to double.
    // Convert to float first then to double.
    auto toDouble = [](DataT const& val) { return static_cast<double>(static_cast<float>(val)); };

    bool       isInf = false;
    bool       isNaN = false;
    std::mutex writeMutex;

#pragma omp parallel for
    for(int i = 0; i < size; ++i)
    {
        auto valA = a[i];
        auto valB = b[i];

        auto numerator = fabs(toDouble(valA) - toDouble(valB));
        auto divisor   = fabs(toDouble(valA)) + fabs(toDouble(valB)) + 1.0;

        if(std::isinf(numerator) || std::isinf(divisor))
        {
#pragma omp atomic
            isInf |= true;
        }
        else
        {
            auto relative_error = numerator / divisor;
            if(std::isnan(relative_error))
            {
#pragma omp atomic
                isNaN |= true;
            }
            else if(relative_error > max_relative_error)
            {
                const std::lock_guard<std::mutex> guard(writeMutex);
                // Double check in case of stall
                if(relative_error > max_relative_error)
                {
                    max_relative_error = relative_error;
                }
            }
        }

        if(isInf || isNaN)
        {
            i = size;
        }
    }

    auto eps = toDouble(std::numeric_limits<DataT>::epsilon());
    if(isInf)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DataT>::infinity();
    }
    else if(isNaN)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DataT>::signaling_NaN();
    }
    else if(max_relative_error > (eps * tolerance))
    {
        retval = false;
    }

    return std::make_pair(retval, max_relative_error);
}

#endif // ROCWMMA_SAMPLES_COMMON_HPP
