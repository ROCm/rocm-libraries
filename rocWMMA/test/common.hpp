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

#ifndef ROCWMMA_TEST_COMMON_HPP
#define ROCWMMA_TEST_COMMON_HPP

#if ROCWMMA_TESTS_NO_HALF
#warning("Building tests with hfloat16_t requires !HIP_NO_HALF && !__HIP_NO_HALF_CONVERSIONS__. Proceeding without hfloat16_t")
#endif // !ROCWMMA_NO_HALF && __HIP_NO_HALF_CONVERSIONS__

#include <iostream>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <vector>

#include <rocwmma/internal/types.hpp>

#include "test_config.hpp"

#include "device/common.hpp"

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

#if ROCWMMA_BENCHMARK_TESTS
#ifndef CHECK_RSMI_ERROR
#define CHECK_RSMI_ERROR(expression, smiErrorFlag)                                               \
    if(auto status = (expression); status != RSMI_STATUS_SUCCESS)                                \
    {                                                                                            \
        const char* errName = nullptr;                                                           \
        rsmi_status_string(status, &errName);                                                    \
        fprintf(stderr, "rsmi error: '%s'(%d) at %s:%d\n", errName, status, __FILE__, __LINE__); \
        smiErrorFlag = true;                                                                     \
    }
#endif
#endif // ROCWMMA_BENCHMARK_TESTS

namespace rocwmma
{
    ROCWMMA_DEVICE inline bool isFirstThread()
    {
        return (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0) && (blockIdx.x == 0)
                && (blockIdx.y == 0) && (blockIdx.z == 0);
    }

    template <typename Lhs, typename Rhs>
    ROCWMMA_DEVICE bool expectEqual(Lhs&& testL, Rhs&& testR, bool debugOnFail, const char* file, uint32_t line)
    {
        bool compare_result = (testL == testR);
        if(!compare_result && debugOnFail && isFirstThread())
        {
            printf("<TestBegin>\n");
            printf("File: %s L:%d\n", file, line);
            if constexpr(is_same_v<decay_t<Lhs>, hfloat16_t>)
            {
                printf("Lhs: %f, Rhs: %f\n", static_cast<float>(testL), static_cast<float>(testR));
            }
            printf("FAILED");
            printf("<Test End>\n");
        }

        return compare_result;
    }

    template <typename Lhs, typename Rhs>
    ROCWMMA_DEVICE bool expectNotEqual(Lhs&& testL, Rhs&& testR, bool debugOnFail, const char* file, uint32_t line)
    {
        bool compare_result = (testL != testR);
        if(!compare_result && debugOnFail && isFirstThread())
        {
            printf("<TestBegin>\n");
            printf("File: %s L:%d\n", file, line);
            printf("FAILED");
            printf("<Test End>\n");
        }

        return compare_result;
    }

    static constexpr uint32_t ERROR_VALUE   = 7u;
    static constexpr uint32_t SUCCESS_VALUE = 0u;

    namespace quirks
    {
        // rocBLAS does not yet support Ti/To/Tc = bf16/bf16/bf16
        template <typename InputT, typename OutputT, typename ComputeT>
        struct rocblas_supported : std::true_type
        {
        };

        template <>
        struct rocblas_supported<bfloat16_t, bfloat16_t, bfloat16_t> : std::false_type
        {
        };

        template <>
        struct rocblas_supported<int8_t, int8_t, int32_t> : std::false_type
        {
        };

#if !defined(ROCBLAS_DATA_TYPE_FLOAT8)
        template <>
        struct rocblas_supported<float8_fnuz_t, float32_t, float32_t> : std::false_type
        {
        };

        template <>
        struct rocblas_supported<bfloat8_fnuz_t, float32_t, float32_t> : std::false_type
        {
        };
#endif
        template <>
        struct rocblas_supported<float8_t, float32_t, float32_t> : std::false_type
        {
        };

        template <>
        struct rocblas_supported<bfloat8_t, float32_t, float32_t> : std::false_type
        {
        };

    } // namespace quirks

    template <typename Layout>
    struct MatrixUtil
    {
        template <typename DataT>
        __host__ static inline void
            print(DataT const* mat, uint32_t m, uint32_t n, std::ostream& stream = std::cout)
        {
            auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
            auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

            auto index = std::is_same<Layout, row_major>::value ? rowMjr : colMjr;
            auto ld    = std::is_same<Layout, row_major>::value ? n : m;

            for(int i = 0; i < m; ++i) // row
            {
                stream << "[ ";
                for(int j = 0; j < n; ++j) // col
                {
                    // (Row, col)
                    stream << mat[index(i, j, ld)] << " ";
                }
                stream << "]\n";
            }
            stream << "\n";
        }

        template <typename DataT>
        __host__ static inline void print(std::vector<DataT> const& mat,
                                          uint32_t                  m,
                                          uint32_t                  n,
                                          std::ostream&             stream = std::cout)
        {
            assert(mat.size() == n * m);
            print(mat.data(), m, n, stream);
        }

        template <typename DataT, typename PrintT>
        __host__ static inline void
            printAsType(DataT const* mat, uint32_t m, uint32_t n, std::ostream& stream = std::cout)
        {
            auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
            auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

            auto index = std::is_same<Layout, row_major>::value ? rowMjr : colMjr;
            auto ld    = std::is_same<Layout, row_major>::value ? n : m;

            for(int i = 0; i < m; ++i) // row
            {
                stream << "[ ";
                for(int j = 0; j < n; ++j) // col
                {
                    // (Row, col)
                    stream << static_cast<PrintT>(mat[index(i, j, ld)]) << " ";
                }
                stream << "]\n";
            }
            stream << "\n";
        }

        template <typename DataT>
        __host__ static inline void fillWithPadding(
            DataT* mat, uint32_t m, uint32_t n, uint32_t padM, uint32_t padN, DataT padValue)
        {
            auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
            auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

            const auto limitM = m + 2 * padM;
            const auto limitN = n + 2 * padN;
            auto       index  = std::is_same<Layout, row_major>::value ? rowMjr : colMjr;
            auto       ld     = std::is_same<Layout, row_major>::value ? limitN : limitM;

#pragma omp parallel for
            for(int i = 0; i < limitM; ++i) // row
            {
#pragma omp parallel for
                for(int j = 0; j < limitN; ++j) // col
                {
                    auto idx = index(i, j, ld);
                    if(i < padM || i >= (limitM - padM) || j < padN || j >= (limitN - padN))
                    {
                        mat[idx] = padValue;
                    }
                    else
                    {
                        // Count up in integers, in ascending order for each row.
                        auto value = ((i - padM) * n + (j - padN)) % 5;
                        mat[idx]   = ((value % 3) && std::is_signed<DataT>::value)
                                         ? -static_cast<DataT>(value)
                                         : static_cast<DataT>(value);
                    }
                }
            }
        }

        template <typename DataT>
        __host__ static inline void fillWithPadding(std::vector<DataT>& mat,
                                                    uint32_t            m,
                                                    uint32_t            n,
                                                    uint32_t            padM,
                                                    uint32_t            padN,
                                                    DataT               padValue)
        {
            assert(mat.size() == n * m);
            fillWithPadding(mat.data(), m, n, padM, padN, padValue);
        }

        template <typename DataT>
        __host__ static inline void fillWithPaddingLaunchKernel(
            DataT* d_mat, uint32_t m, uint32_t n, uint32_t padM, uint32_t padN, DataT padValue)
        {
            const auto limitM = m + 2 * padM;
            const auto limitN = n + 2 * padN;

            auto blockDim = dim3(1024, 1, 1);
            auto gridDim  = dim3(ceilDiv(limitM * limitN, blockDim.x), 1, 1);
            hipLaunchKernelGGL((fillWithPaddingKernel<DataT, Layout>),
                               gridDim,
                               blockDim,
                               0,
                               0,
                               d_mat,
                               m,
                               n,
                               padM,
                               padN,
                               padValue);
        }

        template <typename DataT>
        __host__ static inline void fill(DataT* mat, uint32_t m, uint32_t n)
        {
            auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
            auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

            auto index = std::is_same<Layout, row_major>::value ? rowMjr : colMjr;
            auto ld    = std::is_same<Layout, row_major>::value ? n : m;

#pragma omp parallel for
            for(int i = 0; i < m; ++i) // row
            {
#pragma omp parallel for
                for(int j = 0; j < n; ++j) // col
                {
                    // Count up in integers, in ascending order for each row.
                    auto value = (i * n + j) % 5;
                    auto idx   = index(i, j, ld);
                    mat[idx]   = ((value % 3) && std::is_signed<DataT>::value)
                                     ? -static_cast<DataT>(value)
                                     : static_cast<DataT>(value);
                }
            }
        }

        template <typename DataT>
        __host__ static inline void fill(std::vector<DataT>& mat, uint32_t m, uint32_t n)
        {
            assert(mat.size() == n * m);
            fill(mat.data(), m, n);
        }

        template <typename DataT>
        __host__ static inline void fillVal(DataT* mat, uint32_t m, uint32_t n, DataT value)
        {
#pragma omp parallel for
            for(int i = 0; i < m * n; ++i) // row
            {
                mat[i] = value;
            }
        }

        template <typename DataT>
        __host__ static inline void
            fillVal(std::vector<DataT>& mat, uint32_t m, uint32_t n, DataT value)
        {
            assert(mat.size() == n * m);
            fillVal(mat.data(), m, n, value);
        }

        // fill kernel wrapper for M x N matrix
        template <typename DataT>
        __host__ static inline void fillLaunchKernel(DataT* d_mat, uint32_t m, uint32_t n)
        {
            auto blockDim = dim3(1024, 1, 1);
            auto gridDim  = dim3(ceilDiv(m * n, blockDim.x), 1, 1);
            hipLaunchKernelGGL((fillKernel<DataT, Layout>), gridDim, blockDim, 0, 0, d_mat, m, n);
        }

        // fill kernel wrapper for batched M x K matrices
        template <typename DataT>
        __host__ static inline void
            fillLaunchKernel(DataT* d_mat, uint32_t m, uint32_t k, uint32_t b)
        {
            auto blockDim = dim3(1024, 1, 1);
            auto gridDim  = dim3(ceilDiv(m * k, blockDim.x), 1, b);
            hipLaunchKernelGGL(
                (fillKernel<DataT, Layout>), gridDim, blockDim, 0, 0, d_mat, m, k, b);
        }

        // fill kernel wrapper for M x N matrix for a specific value
        template <typename DataT>
        __host__ static inline void
            fillValLaunchKernel(DataT* d_mat, uint32_t m, uint32_t n, DataT value)
        {
            auto blockDim = dim3(1024, 1, 1);
            auto gridDim  = dim3(ceilDiv(m * n, blockDim.x), 1, 1);
            hipLaunchKernelGGL(
                (fillValKernel<DataT, Layout>), gridDim, blockDim, 0, 0, d_mat, m, n, value);
        }

        // fill kernel wrapper for M x N matrix for mat[i] = i
        template <typename DataT>
        __host__ static inline void fillIdxLaunchKernel(DataT* d_mat, uint32_t m, uint32_t n)
        {
            auto blockDim = dim3(1024, 1, 1);
            auto gridDim  = dim3(ceilDiv(m * n, blockDim.x), 1, 1);
            hipLaunchKernelGGL(
                (fillIdxKernel<DataT, Layout>), gridDim, blockDim, 0, 0, d_mat, m, n);
        }
    };

    // compareEqual on two different layouts: must calculate index offsets
    template <typename TypeA,
              typename TypeB,
              typename LayoutA,
              typename LayoutB,
              typename std::enable_if_t<!std::is_same<LayoutA, LayoutB>::value, int> = 0>
    std::pair<bool, double> compareEqual(TypeA const* matrixA,
                                         TypeB const* matrixB,
                                         uint32_t     m,
                                         uint32_t     n,
                                         uint32_t     lda,
                                         uint32_t     ldb,
                                         double       tolerance = 10.0)
    {
        bool   retval             = true;
        double max_relative_error = 0.0;

        // Some types don't have direct conversion to double.
        // Convert to float first then to double.
        auto toDoubleA
            = [](TypeA const& val) { return static_cast<double>(static_cast<float>(val)); };
        auto toDoubleB
            = [](TypeB const& val) { return static_cast<double>(static_cast<float>(val)); };

        auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
        auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

        auto indexA = std::is_same<LayoutA, row_major>::value ? rowMjr : colMjr;
        auto indexB = std::is_same<LayoutB, row_major>::value ? rowMjr : colMjr;

        bool       isInf = false;
        bool       isNaN = false;
        std::mutex writeMutex;

#pragma omp parallel for
        for(int i = 0; i < m; ++i) // Row
        {
#pragma omp parallel for
            for(int j = 0; j < n; ++j) // Col
            {
                auto valA = matrixA[indexA(i, j, lda)];
                auto valB = matrixB[indexB(i, j, ldb)];

                auto numerator = fabs(toDoubleA(valA) - toDoubleB(valB));
                auto divisor   = fabs(toDoubleA(valA)) + fabs(toDoubleB(valB)) + 1.0;

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
                    i = m;
                    j = n;
                }
            }
        }

        auto eps = toDoubleA(std::numeric_limits<TypeA>::epsilon());
        if(isInf)
        {
            retval             = false;
            max_relative_error = std::numeric_limits<TypeA>::infinity();
        }
        else if(isNaN)
        {
            retval             = false;
            max_relative_error = double(std::numeric_limits<TypeA>::signaling_NaN());
        }
        else if(max_relative_error > (eps * tolerance))
        {
            retval = false;
        }

        return std::make_pair(retval, max_relative_error);
    }

    // compareEqual on two equal layouts: index offsets are identical
    // can use slightly faster 1D compare
    template <typename TypeA,
              typename TypeB,
              typename LayoutA,
              typename LayoutB,
              typename std::enable_if_t<std::is_same<LayoutA, LayoutB>::value, int> = 0>
    std::pair<bool, double> compareEqual(TypeA const* matrixA,
                                         TypeB const* matrixB,
                                         uint32_t     m,
                                         uint32_t     n,
                                         uint32_t     lda,
                                         uint32_t     ldb,
                                         double       tolerance = 10.0)
    {
        assert(lda == ldb && "Leading dims must match");

        bool   retval             = true;
        double max_relative_error = 0.0;

        // Some types don't have direct conversion to double.
        // Convert to float first then to double.
        auto toDoubleA
            = [](TypeA const& val) { return static_cast<double>(static_cast<float>(val)); };
        auto toDoubleB
            = [](TypeB const& val) { return static_cast<double>(static_cast<float>(val)); };

        bool       isInf = false;
        bool       isNaN = false;
        std::mutex writeMutex;

#pragma omp parallel for
        for(int i = 0; i < m * n; ++i) // Row
        {
            auto valA = matrixA[i];
            auto valB = matrixB[i];

            auto numerator = fabs(toDoubleA(valA) - toDoubleB(valB));
            auto divisor   = fabs(toDoubleA(valA)) + fabs(toDoubleB(valB)) + 1.0;

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
                i = m * n;
            }
        }

        auto eps = toDoubleA(std::numeric_limits<TypeA>::epsilon());
        if(isInf)
        {
            retval             = false;
            max_relative_error = std::numeric_limits<TypeA>::infinity();
        }
        else if(isNaN)
        {
            retval             = false;
            max_relative_error = double(std::numeric_limits<TypeA>::signaling_NaN());
        }
        else if(max_relative_error > (eps * tolerance))
        {
            retval = false;
        }

        return std::make_pair(retval, max_relative_error);
    }

    template <typename TypeA, typename TypeB, typename LayoutA, typename LayoutB>
    inline std::pair<bool, double> compareEqual(
        TypeA const* matrixA, TypeB const* matrixB, uint32_t m, uint32_t n, double tolerance = 10.0)
    {
        uint32_t lda = std::is_same<LayoutA, row_major>::value ? n : m;
        uint32_t ldb = std::is_same<LayoutB, row_major>::value ? n : m;

        return compareEqual<TypeA, TypeB, LayoutA, LayoutB>(
            matrixA, matrixB, m, n, lda, ldb, tolerance);
    }

    template <typename TypeA, typename TypeB, typename LayoutA, typename LayoutB>
    inline std::pair<bool, double> compareEqual(std::vector<TypeA> const& a,
                                                std::vector<TypeB> const& b,
                                                uint32_t                  m,
                                                uint32_t                  n,
                                                double                    tolerance = 10.0)
    {
        assert(a.size() == b.size() && "A and B are not the same size");
        assert(a.size() == m * n && "A and B do not match size M x N");
        return compareEqual<TypeA, TypeB, LayoutA, LayoutB>(a.data(), b.data(), m, n, tolerance);
    }

    template <typename TypeA, typename TypeB, typename LayoutA, typename LayoutB>
    inline std::pair<bool, double> compareEqual(std::vector<TypeA> const& a,
                                                std::vector<TypeB> const& b,
                                                uint32_t                  m,
                                                uint32_t                  n,
                                                uint32_t                  lda,
                                                uint32_t                  ldb,
                                                double                    tolerance = 10.0)
    {
        assert(a.size() == b.size() && "A and B are not the same size");
        assert(a.size() == m * n && "A and B do not match size m x n");
        return compareEqual<TypeA, TypeB, LayoutA, LayoutB>(
            a.data(), b.data(), m, n, lda, ldb, tolerance);
    }

    // compareEqual kernel wrapper for gemm tests
    template <typename TypeA, typename TypeB, typename LayoutA, typename LayoutB>
    std::pair<bool, double> compareEqualLaunchKernel(
        TypeA* matrixA, TypeB* matrixB, uint32_t m, uint32_t n, double tolerance = 10.0)
    {
        uint32_t lda = std::is_same<LayoutA, row_major>::value ? n : m;
        uint32_t ldb = std::is_same<LayoutB, row_major>::value ? n : m;

        auto blockDim = dim3(1024, 1, 1);
        auto gridDim  = dim3(ceilDiv(m * n, blockDim.x), 1, 1);

        double* d_relativeError;
        double  maxRelativeError;
        CHECK_HIP_ERROR(hipMalloc(&d_relativeError, m * n * sizeof(double)));

        hipEvent_t syncEvent;
        CHECK_HIP_ERROR(hipEventCreate(&syncEvent));

        // Calculate the relative error for each element of matrix A/B
        hipLaunchKernelGGL((compareEqualKernel<TypeA, TypeB, LayoutA, LayoutB>),
                           gridDim,
                           blockDim,
                           0,
                           0,
                           matrixA,
                           matrixB,
                           d_relativeError,
                           m,
                           n,
                           lda,
                           ldb);
        CHECK_HIP_ERROR(hipEventRecord(syncEvent));
        CHECK_HIP_ERROR(hipEventSynchronize(syncEvent));

        // Determine the maximum relative error
        blockDim             = dim3(512, 1, 1);
        uint32_t maxElements = 1024;
        uint32_t offset      = 1;

        for(uint32_t i = m * n; i > 1; i = ceilDiv(i, maxElements))
        {
            gridDim       = dim3(ceilDiv(i, maxElements), 1, 1);
            auto elements = i > maxElements ? maxElements : i;

            hipLaunchKernelGGL((maxReduceKernel),
                               gridDim,
                               blockDim,
                               0,
                               0,
                               d_relativeError,
                               elements,
                               offset,
                               m * n);

            CHECK_HIP_ERROR(hipEventRecord(syncEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(syncEvent));
            offset = offset * maxElements;
        }

        CHECK_HIP_ERROR(
            hipMemcpy(&maxRelativeError, d_relativeError, sizeof(double), hipMemcpyDeviceToHost));

        // Free allocated device memory
        CHECK_HIP_ERROR(hipFree(d_relativeError));

        bool retval = true;
        bool isNaN  = std::isnan(maxRelativeError);

        auto toDoubleA
            = [](TypeA const& val) { return static_cast<double>(static_cast<float>(val)); };

        auto eps = toDoubleA(std::numeric_limits<TypeA>::epsilon());
        if(isNaN)
        {
            retval           = false;
            maxRelativeError = double(std::numeric_limits<TypeA>::signaling_NaN());
        }
        else if(maxRelativeError > (eps * tolerance))
        {
            retval = false;
        }

        return std::make_pair(retval, maxRelativeError);
    }

    // compareEqual kernel wrapper for batched matrices
    template <typename TypeA, typename TypeB>
    std::pair<bool, double> compareEqualLaunchKernel(
        TypeA* matrixA, TypeB* matrixB, uint32_t m, uint32_t k, uint32_t b, double tolerance = 10.0)
    {
        auto blockDim = dim3(1024, 1, 1);
        auto gridDim  = dim3(ceilDiv(m * k, blockDim.x), 1, b);

        double* d_relativeError;
        double  maxRelativeError;
        CHECK_HIP_ERROR(hipMalloc(&d_relativeError, m * k * b * sizeof(double)));

        hipEvent_t syncEvent;
        CHECK_HIP_ERROR(hipEventCreate(&syncEvent));

        // Calculate the relative error for each element of matrix A/B
        hipLaunchKernelGGL((compareEqualKernel<TypeA, TypeB>),
                           gridDim,
                           blockDim,
                           0,
                           0,
                           matrixA,
                           matrixB,
                           d_relativeError,
                           m,
                           k,
                           b);
        CHECK_HIP_ERROR(hipEventRecord(syncEvent));
        CHECK_HIP_ERROR(hipEventSynchronize(syncEvent));

        // Determine the maximum relative error
        blockDim             = dim3(512, 1, 1);
        uint32_t maxElements = 1024;
        uint32_t offset      = 1;

        for(uint32_t i = m * k * b; i > 1; i = ceilDiv(i, maxElements))
        {
            gridDim       = dim3(ceilDiv(i, maxElements), 1, 1);
            auto elements = i > maxElements ? maxElements : i;

            hipLaunchKernelGGL((maxReduceKernel),
                               gridDim,
                               blockDim,
                               0,
                               0,
                               d_relativeError,
                               elements,
                               offset,
                               m * k * b);

            CHECK_HIP_ERROR(hipEventRecord(syncEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(syncEvent));
            offset = offset * maxElements;
        }

        CHECK_HIP_ERROR(
            hipMemcpy(&maxRelativeError, d_relativeError, sizeof(double), hipMemcpyDeviceToHost));

        // Free allocated device memory
        CHECK_HIP_ERROR(hipFree(d_relativeError));

        bool retval = true;
        bool isNaN  = std::isnan(maxRelativeError);

        auto toDoubleA
            = [](TypeA const& val) { return static_cast<double>(static_cast<float>(val)); };

        auto eps = toDoubleA(std::numeric_limits<TypeA>::epsilon());
        if(isNaN)
        {
            retval           = false;
            maxRelativeError = double(std::numeric_limits<TypeA>::signaling_NaN());
        }
        else if(maxRelativeError > (eps * tolerance))
        {
            retval = false;
        }

        return std::make_pair(retval, maxRelativeError);
    }

    // Count occurrences of val in the input array
    template <typename DataT>
    uint64_t countVal(DataT const* a, uint64_t size, DataT const& val, double tolerance = 10.0)
    {
        uint64_t count = 0;
#pragma omp parallel for
        for(uint64_t i = 0; i < size; ++i)
        {
            using TestT = double;
            if(fabs(static_cast<TestT>(val) - static_cast<TestT>(a[i]))
               <= (tolerance * static_cast<TestT>(std::numeric_limits<DataT>::epsilon())))
            {
#pragma omp atomic
                ++count;
            }
        }
        return count;
    }

    // Count occurrences of val inside the rectangular padding of input matrix
    template <typename DataT, typename LayoutT>
    __host__ static inline uint64_t countPaddingVal(DataT*   mat,
                                                    uint32_t m,
                                                    uint32_t n,
                                                    uint32_t padM,
                                                    uint32_t padN,
                                                    DataT    padValue,
                                                    double   tolerance = 10.0)
    {
        auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
        auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };

        const auto limitM = m + 2 * padM;
        const auto limitN = n + 2 * padN;
        auto       index  = std::is_same<LayoutT, row_major>::value ? rowMjr : colMjr;
        auto       ld     = std::is_same<LayoutT, row_major>::value ? limitN : limitM;

        uint64_t count = 0;

#pragma omp parallel for
        for(uint32_t i = 0; i < limitM; ++i) // row
        {
#pragma omp parallel for
            for(uint32_t j = 0; j < limitN; ++j) // col
            {
                auto idx = index(i, j, ld);
                if(i < padM || i >= (limitM - padM) || j < padN || j >= (limitN - padN))
                {
                    using TestT = float64_t;
                    if(fabs(static_cast<TestT>(padValue) - static_cast<TestT>(mat[idx]))
                       <= (tolerance * static_cast<TestT>(std::numeric_limits<DataT>::epsilon())))
                    {
#pragma omp atomic
                        ++count;
                    }
                }
            }
        }
        return count;
    }

} // namespace rocwmma

#endif // ROCWMMA_TEST_COMMON_HPP
