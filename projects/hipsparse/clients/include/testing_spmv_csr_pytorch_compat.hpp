/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

/**
 * @file testing_spmv_csr_pytorch_compat.hpp
 * @brief Tests that reproduce PyTorch's test_csr_matvec for various precision types.
 *
 * This replicates the PyTorch tests:
 *   PYTORCH_TEST_WITH_ROCM=1 python test/test_sparse_csr.py \
 *       TestSparseCSRCUDA.test_csr_matvec_cuda_bfloat16
 *   PYTORCH_TEST_WITH_ROCM=1 python test/test_sparse_csr.py \
 *       TestSparseCSRCUDA.test_csr_matvec_cuda_float16
 *
 * The PyTorch test:
 * 1. Generates a random sparse CSR matrix of shape (100, 100) with 1000 non-zeros
 * 2. Creates a random dense vector of size 100
 * 3. Performs sparse matrix-vector multiplication: res = csr.matmul(vec)
 * 4. Compares against dense matmul: expected = csr.to_dense().matmul(vec)
 */

#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <type_traits>
#include <vector>

using namespace hipsparse_test;

// ============================================================================
// Type traits for HIP data types
// ============================================================================

template <typename T>
struct hip_datatype;

template <>
struct hip_datatype<hipsparseBfloat16>
{
    static constexpr hipDataType value = HIP_R_16BF;
};

template <>
struct hip_datatype<_Float16>
{
    static constexpr hipDataType value = HIP_R_16F;
};

template <>
struct hip_datatype<float>
{
    static constexpr hipDataType value = HIP_R_32F;
};

template <>
struct hip_datatype<double>
{
    static constexpr hipDataType value = HIP_R_64F;
};

// Compute type for SpMV (use float32 for half types, native for float/double)
template <typename T>
struct compute_datatype
{
    static constexpr hipDataType value = hip_datatype<T>::value;
    using scalar_type                  = T;
};

template <>
struct compute_datatype<hipsparseBfloat16>
{
    static constexpr hipDataType value = HIP_R_32F;
    using scalar_type                  = float;
};

template <>
struct compute_datatype<_Float16>
{
    static constexpr hipDataType value = HIP_R_32F;
    using scalar_type                  = float;
};

// Tolerance traits for different precision types
template <typename T>
struct tolerance_traits;

template <>
struct tolerance_traits<hipsparseBfloat16>
{
    static constexpr float       atol = 2e-2f;
    static constexpr float       rtol = 1e-2f;
    static constexpr const char* name = "BFloat16";
};

template <>
struct tolerance_traits<_Float16>
{
    static constexpr float       atol = 2e-3f;
    static constexpr float       rtol = 1e-3f;
    static constexpr const char* name = "Float16";
};

template <>
struct tolerance_traits<float>
{
    static constexpr float       atol = 1e-5f;
    static constexpr float       rtol = 1e-5f;
    static constexpr const char* name = "Float32";
};

template <>
struct tolerance_traits<double>
{
    // Note: Reference is computed in float32, so expect float32-level precision
    static constexpr float       atol = 1e-5f;
    static constexpr float       rtol = 1e-5f;
    static constexpr const char* name = "Float64";
};

// ============================================================================
// Type conversion helpers
// ============================================================================

template <typename T>
inline T from_float(float val)
{
    return static_cast<T>(val);
}

template <>
inline hipsparseBfloat16 from_float<hipsparseBfloat16>(float val)
{
    return hipsparseBfloat16(val);
}

template <>
inline _Float16 from_float<_Float16>(float val)
{
    return _Float16(val);
}

template <typename T>
inline float to_float(T val)
{
    return static_cast<float>(val);
}

template <>
inline float to_float<hipsparseBfloat16>(hipsparseBfloat16 val)
{
    return static_cast<float>(val);
}

template <>
inline float to_float<_Float16>(_Float16 val)
{
    return static_cast<float>(val);
}

// ============================================================================
// Generate random sparse CSR matrix (PyTorch style)
// ============================================================================

/**
 * @brief Generate a random sparse CSR matrix similar to PyTorch's genSparseCSRTensor
 */
inline int generate_random_csr_pytorch(int                 m,
                                       int                 n,
                                       int                 target_nnz,
                                       std::vector<int>&   crow_indices,
                                       std::vector<int>&   col_indices,
                                       std::vector<float>& values,
                                       unsigned int        seed = 12345)
{
    std::mt19937                          gen(seed);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int>    col_dist(0, n - 1);

    // Distribute nnz across rows roughly uniformly with some randomness
    std::vector<int> nnz_per_row(m, 0);
    int              nnz_remaining = target_nnz;

    // First pass: assign minimum nnz to each row
    for(int i = 0; i < m && nnz_remaining > 0; i++)
    {
        int                                max_for_row = std::min(nnz_remaining, n);
        std::uniform_int_distribution<int> nnz_dist(0, std::min(max_for_row, target_nnz / m + 5));
        nnz_per_row[i] = nnz_dist(gen);
        nnz_remaining -= nnz_per_row[i];
    }

    // Second pass: distribute remaining nnz
    while(nnz_remaining > 0)
    {
        int row = gen() % m;
        if(nnz_per_row[row] < n)
        {
            nnz_per_row[row]++;
            nnz_remaining--;
        }
    }

    // Generate actual entries
    crow_indices.resize(m + 1);
    crow_indices[0] = 0;
    col_indices.clear();
    values.clear();

    for(int i = 0; i < m; i++)
    {
        std::set<int> row_cols;
        int           row_nnz = std::min(nnz_per_row[i], n);

        // Generate unique column indices for this row
        while(static_cast<int>(row_cols.size()) < row_nnz)
        {
            row_cols.insert(col_dist(gen));
        }

        // Add sorted column indices
        for(int col : row_cols)
        {
            col_indices.push_back(col);
            values.push_back(val_dist(gen));
        }

        crow_indices[i + 1] = static_cast<int>(col_indices.size());
    }

    return static_cast<int>(col_indices.size());
}

// ============================================================================
// Convert sparse CSR to dense matrix
// ============================================================================

inline void csr_to_dense_pytorch(int                       m,
                                 int                       n,
                                 const std::vector<int>&   crow_indices,
                                 const std::vector<int>&   col_indices,
                                 const std::vector<float>& values,
                                 std::vector<float>&       dense)
{
    dense.assign(m * n, 0.0f);

    for(int i = 0; i < m; i++)
    {
        for(int j = crow_indices[i]; j < crow_indices[i + 1]; j++)
        {
            int col            = col_indices[j];
            dense[i * n + col] = values[j];
        }
    }
}

// ============================================================================
// Dense matrix-vector multiplication (reference)
// ============================================================================

inline void dense_matvec_pytorch(
    int m, int n, const std::vector<float>& A, const std::vector<float>& x, std::vector<float>& y)
{
    y.assign(m, 0.0f);

    for(int i = 0; i < m; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < n; j++)
        {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

// ============================================================================
// Compare results with tolerance
// ============================================================================

inline bool compare_results_pytorch(const std::vector<float>& expected,
                                    const std::vector<float>& actual,
                                    float                     atol,
                                    float                     rtol,
                                    int&                      num_errors,
                                    float&                    max_abs_error,
                                    float&                    max_rel_error)
{
    num_errors    = 0;
    max_abs_error = 0.0f;
    max_rel_error = 0.0f;

    if(expected.size() != actual.size())
    {
        return false;
    }

    for(size_t i = 0; i < expected.size(); i++)
    {
        float diff          = std::abs(expected[i] - actual[i]);
        float abs_tolerance = atol + rtol * std::abs(expected[i]);

        if(diff > abs_tolerance)
        {
            num_errors++;
        }

        max_abs_error = std::max(max_abs_error, diff);
        if(std::abs(expected[i]) > 1e-6f)
        {
            max_rel_error = std::max(max_rel_error, diff / std::abs(expected[i]));
        }
    }

    return (num_errors == 0);
}

// ============================================================================
// Templated Test CSR SpMV
// ============================================================================

template <typename T>
inline void testing_spmv_csr_pytorch_compat(int m, int n, int target_nnz, bool use_int64)
{
    // Generate random CSR matrix on host (always in float32)
    std::vector<int>   h_crow_indices;
    std::vector<int>   h_col_indices;
    std::vector<float> h_values_f32;

    int actual_nnz = generate_random_csr_pytorch(
        m, n, target_nnz, h_crow_indices, h_col_indices, h_values_f32);

    // Convert values to target type
    std::vector<T> h_values(actual_nnz);
    for(int i = 0; i < actual_nnz; i++)
    {
        h_values[i] = from_float<T>(h_values_f32[i]);
    }

    // Generate random input vector x
    std::vector<float>                    h_x_f32(n);
    std::vector<T>                        h_x(n);
    std::mt19937                          gen(54321);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(int i = 0; i < n; i++)
    {
        h_x_f32[i] = dist(gen);
        h_x[i]     = from_float<T>(h_x_f32[i]);
    }

    // Compute reference result using dense matmul (in float32)
    std::vector<float> h_dense;
    csr_to_dense_pytorch(m, n, h_crow_indices, h_col_indices, h_values_f32, h_dense);

    std::vector<float> h_y_ref;
    dense_matvec_pytorch(m, n, h_dense, h_x_f32, h_y_ref);

    // Allocate device memory
    auto d_crow_indices_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto d_col_indices_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * actual_nnz), device_free};
    auto d_values_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * actual_nnz), device_free};
    auto d_x_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * n), device_free};
    auto d_y_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};

    int* d_crow_indices = (int*)d_crow_indices_managed.get();
    int* d_col_indices  = (int*)d_col_indices_managed.get();
    T*   d_values       = (T*)d_values_managed.get();
    T*   d_x            = (T*)d_x_managed.get();
    T*   d_y            = (T*)d_y_managed.get();

    // Copy data to device
    CHECK_HIP_ERROR(hipMemcpy(
        d_crow_indices, h_crow_indices.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        d_col_indices, h_col_indices.data(), sizeof(int) * actual_nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(d_values, h_values.data(), sizeof(T) * actual_nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_x, h_x.data(), sizeof(T) * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(d_y, 0, sizeof(T) * m));

    // Setup hipSPARSE
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Create sparse matrix descriptor
    hipsparseSpMatDescr_t matA;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateCsr(&matA,
                                             m,
                                             n,
                                             actual_nnz,
                                             d_crow_indices,
                                             d_col_indices,
                                             d_values,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_32I,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             hip_datatype<T>::value));

    // Create dense vector descriptors
    hipsparseDnVecDescr_t vecX, vecY;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&vecX, n, d_x, hip_datatype<T>::value));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&vecY, m, d_y, hip_datatype<T>::value));

    // Set alpha = 1.0, beta = 0.0 (y = A * x)
    // Use the correct scalar type for the compute type
    using scalar_t = typename compute_datatype<T>::scalar_type;
    scalar_t alpha = static_cast<scalar_t>(1.0);
    scalar_t beta  = static_cast<scalar_t>(0.0);

    // Get buffer size
    size_t bufferSize = 0;
    CHECK_HIPSPARSE_ERROR(hipsparseSpMV_bufferSize(handle,
                                                   HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   matA,
                                                   vecX,
                                                   &beta,
                                                   vecY,
                                                   compute_datatype<T>::value,
                                                   HIPSPARSE_MV_ALG_DEFAULT,
                                                   &bufferSize));

    // Allocate buffer
    auto  d_buffer_managed = hipsparse_unique_ptr{device_malloc(bufferSize), device_free};
    void* d_buffer         = d_buffer_managed.get();

    // Preprocess
    CHECK_HIPSPARSE_ERROR(hipsparseSpMV_preprocess(handle,
                                                   HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   matA,
                                                   vecX,
                                                   &beta,
                                                   vecY,
                                                   compute_datatype<T>::value,
                                                   HIPSPARSE_MV_ALG_DEFAULT,
                                                   d_buffer));

    // Execute SpMV
    CHECK_HIPSPARSE_ERROR(hipsparseSpMV(handle,
                                        HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        matA,
                                        vecX,
                                        &beta,
                                        vecY,
                                        compute_datatype<T>::value,
                                        HIPSPARSE_MV_ALG_DEFAULT,
                                        d_buffer));

    // Wait for completion
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // Copy result back to host
    std::vector<T> h_y(m);
    CHECK_HIP_ERROR(hipMemcpy(h_y.data(), d_y, sizeof(T) * m, hipMemcpyDeviceToHost));

    // Convert result to float for comparison
    std::vector<float> h_y_actual(m);
    for(int i = 0; i < m; i++)
    {
        h_y_actual[i] = to_float(h_y[i]);
    }

    // Compare results using type-specific tolerance
    int   num_errors;
    float max_abs_error, max_rel_error;
    bool  passed = compare_results_pytorch(h_y_ref,
                                          h_y_actual,
                                          tolerance_traits<T>::atol,
                                          tolerance_traits<T>::rtol,
                                          num_errors,
                                          max_abs_error,
                                          max_rel_error);

#ifdef GOOGLE_TEST
    ASSERT_TRUE(passed) << tolerance_traits<T>::name << " SpMV mismatch: " << num_errors
                        << " errors, max_abs=" << max_abs_error << ", max_rel=" << max_rel_error;
#endif

    // Cleanup
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matA));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(vecX));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(vecY));
}

// ============================================================================
// Convenience wrappers for backward compatibility
// ============================================================================

inline void testing_spmv_csr_bfloat16_pytorch_compat(int m, int n, int target_nnz, bool use_int64)
{
    testing_spmv_csr_pytorch_compat<hipsparseBfloat16>(m, n, target_nnz, use_int64);
}

inline void testing_spmv_csr_float16_pytorch_compat(int m, int n, int target_nnz, bool use_int64)
{
    testing_spmv_csr_pytorch_compat<_Float16>(m, n, target_nnz, use_int64);
}

inline void testing_spmv_csr_float32_pytorch_compat(int m, int n, int target_nnz, bool use_int64)
{
    testing_spmv_csr_pytorch_compat<float>(m, n, target_nnz, use_int64);
}

inline void testing_spmv_csr_float64_pytorch_compat(int m, int n, int target_nnz, bool use_int64)
{
    testing_spmv_csr_pytorch_compat<double>(m, n, target_nnz, use_int64);
}
