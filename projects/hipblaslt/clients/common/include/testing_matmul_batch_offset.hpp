/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
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

#pragma once

#include "datatype_interface.hpp"
#include "flops.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_math.hpp"
#include "hipblaslt_random.hpp"
#include "hipblaslt_test.hpp"
#include "hipblaslt_vector.hpp"
#include "near.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include <cmath>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <roc/host_validation/adapters/hipblaslt/Types.hpp>
#include <roc/host_validation/validation.hpp>

/* ============================================================================================ */
/*! \brief  Test for 64-bit batch offset support in general batched GEMM                        */

template <typename Ti, typename To, typename Tc>
void testing_matmul_batch_offset_impl(const Arguments& arg)
{
    hipblasOperation_t transA = char_to_hipblas_operation(arg.transA);
    hipblasOperation_t transB = char_to_hipblas_operation(arg.transB);

    // Use first element from arrays (grouped GEMM uses arrays, we use first element)
    int64_t M = arg.M[0];
    int64_t N = arg.N[0];
    int64_t K = arg.K[0];

    int64_t lda = arg.lda[0];
    int64_t ldb = arg.ldb[0];
    int64_t ldc = arg.ldc[0];
    int64_t ldd = arg.ldd[0];

    int32_t batch_count = arg.batch_count;

    // Batch offsets (from YAML - in ELEMENTS)
    int64_t offset_a = arg.batch_offset_a;
    int64_t offset_b = arg.batch_offset_b;
    int64_t offset_c = arg.batch_offset_c;
    int64_t offset_d = arg.batch_offset_d;

    // Only test general batched mode (pointer array)
    if(arg.batch_mode != 1) // 1 = pointer array
    {
        GTEST_SKIP() << "Batch offset only supported for general batched mode (batch_mode=1)";
    }

    // Calculate matrix sizes
    int64_t A_row = transA == HIPBLAS_OP_N ? M : K;
    int64_t A_col = transA == HIPBLAS_OP_N ? K : M;
    int64_t B_row = transB == HIPBLAS_OP_N ? K : N;
    int64_t B_col = transB == HIPBLAS_OP_N ? N : K;

    // Sub-matrix sizes (actual GEMM size) in elements
    size_t size_A_sub = size_t(lda) * size_t(A_col);
    size_t size_B_sub = size_t(ldb) * size_t(B_col);
    size_t size_C_sub = size_t(ldc) * size_t(N);
    size_t size_D_sub = size_t(ldd) * size_t(N);

    // Calculate padding needed for negative offsets (in elements)
    // If offset is negative, we need padding BEFORE the base pointer
    // If offset is positive, padding is at the beginning (offset space)
    size_t padding_a = (offset_a < 0) ? size_t(-offset_a) : 0;
    size_t padding_b = (offset_b < 0) ? size_t(-offset_b) : 0;
    size_t padding_c = (offset_c < 0) ? size_t(-offset_c) : 0;
    size_t padding_d = (offset_d < 0) ? size_t(-offset_d) : 0;

    // Full buffer sizes: [padding for negative offsets] + [matrix data] + [positive offset space]
    // Layout for negative offset:  [padding|matrix_data]  base points to start of matrix_data
    // Layout for positive offset:  [matrix_data|padding]  base points to start of buffer
    size_t size_A_full = padding_a + size_A_sub + (offset_a > 0 ? size_t(offset_a) : 0);
    size_t size_B_full = padding_b + size_B_sub + (offset_b > 0 ? size_t(offset_b) : 0);
    size_t size_C_full = padding_c + size_C_sub + (offset_c > 0 ? size_t(offset_c) : 0);
    size_t size_D_full = padding_d + size_D_sub + (offset_d > 0 ? size_t(offset_d) : 0);

    // Allocate host memory for full buffers
    host_vector<Ti> h_A_full(size_A_full * batch_count);
    host_vector<Ti> h_B_full(size_B_full * batch_count);
    host_vector<To> h_C_full(size_C_full * batch_count);
    host_vector<To> h_D_full(size_D_full * batch_count);   // GPU result with offset API
    host_vector<To> h_D_gold(size_D_sub * batch_count);    // CPU reference

    // Initialize matrices with known pattern
    // For negative offsets: base points to (padding + 0), data at (base + negative_offset) = padding region
    // For positive offsets: base points to 0, data at (base + positive_offset) = offset region
    for(int b = 0; b < batch_count; b++)
    {
        // Calculate base pointer for this batch (after padding for negative offsets)
        Ti* A_base = h_A_full.data() + b * size_A_full + padding_a;
        Ti* B_base = h_B_full.data() + b * size_B_full + padding_b;
        To* C_base = h_C_full.data() + b * size_C_full + padding_c;

        // Data location is at (base + offset)
        // For negative offset: base + (-N) points backward into padding region
        // For positive offset: base + (+N) points forward into buffer
        Ti* A_batch = A_base + offset_a;
        Ti* B_batch = B_base + offset_b;
        To* C_batch = C_base + offset_c;

        // Simple initialization: A and B with small integers.
        roc::host_validation::generate(
            roc::host_validation::MatrixView<Ti>(
                A_batch, size_t(A_row), size_t(A_col), 1, static_cast<ptrdiff_t>(lda)),
            [b](size_t row, size_t column) {
                return (static_cast<int64_t>(row) + static_cast<int64_t>(column) + b) % 7 + 1;
            });
        roc::host_validation::generate(
            roc::host_validation::MatrixView<Ti>(
                B_batch, size_t(B_row), size_t(B_col), 1, static_cast<ptrdiff_t>(ldb)),
            [b](size_t row, size_t column) {
                return (static_cast<int64_t>(row) - static_cast<int64_t>(column) + b) % 5 + 1;
            });
        roc::host_validation::generate(
            roc::host_validation::MatrixView<To>(
                C_batch, size_t(M), size_t(N), 1, static_cast<ptrdiff_t>(ldc)),
            [](size_t row, size_t column) {
                return (static_cast<int64_t>(row) + static_cast<int64_t>(column)) % 3;
            });
    }

    // Allocate device memory
    device_vector<Ti> d_A_full(size_A_full * batch_count);
    device_vector<Ti> d_B_full(size_B_full * batch_count);
    device_vector<To> d_C_full(size_C_full * batch_count);
    device_vector<To> d_D_full(size_D_full * batch_count);

    // Copy to device
    CHECK_HIP_ERROR(
        hipMemcpy(d_A_full, h_A_full.data(), sizeof(Ti) * size_A_full * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(d_B_full, h_B_full.data(), sizeof(Ti) * size_B_full * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(d_C_full, h_C_full.data(), sizeof(To) * size_C_full * batch_count, hipMemcpyHostToDevice));

    // Setup pointer arrays for base addresses
    // Base pointers must point AFTER the padding (for negative offset support)
    std::vector<Ti*> h_batch_A(batch_count);
    std::vector<Ti*> h_batch_B(batch_count);
    std::vector<To*> h_batch_C(batch_count);
    std::vector<To*> h_batch_D(batch_count);

    for(int b = 0; b < batch_count; b++)
    {
        h_batch_A[b] = d_A_full + b * size_A_full + padding_a;
        h_batch_B[b] = d_B_full + b * size_B_full + padding_b;
        h_batch_C[b] = d_C_full + b * size_C_full + padding_c;
        h_batch_D[b] = d_D_full + b * size_D_full + padding_d;
    }

    // Allocate device memory for pointer arrays
    Ti** d_batch_A;
    Ti** d_batch_B;
    To** d_batch_C;
    To** d_batch_D;

    CHECK_HIP_ERROR(hipMalloc(&d_batch_A, sizeof(Ti*) * batch_count));
    CHECK_HIP_ERROR(hipMalloc(&d_batch_B, sizeof(Ti*) * batch_count));
    CHECK_HIP_ERROR(hipMalloc(&d_batch_C, sizeof(To*) * batch_count));
    CHECK_HIP_ERROR(hipMalloc(&d_batch_D, sizeof(To*) * batch_count));

    CHECK_HIP_ERROR(
        hipMemcpy(d_batch_A, h_batch_A.data(), sizeof(Ti*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(d_batch_B, h_batch_B.data(), sizeof(Ti*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(d_batch_C, h_batch_C.data(), sizeof(To*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(d_batch_D, h_batch_D.data(), sizeof(To*) * batch_count, hipMemcpyHostToDevice));

    // Alpha and beta
    Tc h_alpha = arg.get_alpha<Tc>();
    Tc h_beta  = arg.get_beta<Tc>();

    // Setup hipBLASLt
    hipblasLtHandle_t handle;
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

    hipblasLtMatmulDesc_t matmul_desc;
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulDescCreate(&matmul_desc, arg.compute_type, arg.scale_type));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulDescSetAttribute(
            matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulDescSetAttribute(
            matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Create matrix layouts
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, arg.a_type, A_row, A_col, lda));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, arg.b_type, B_row, B_col, ldb));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, arg.c_type, M, N, ldc));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, arg.d_type, M, N, ldd));

    // Set batch count and mode
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

    int32_t batch_mode = 1; // Pointer array
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));

    // ========================================
    // GPU GEMM with offset API
    // ========================================

    // Set offsets for all matrices
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matA, HIPBLASLT_MATRIX_LAYOUT_OFFSET, &offset_a, sizeof(offset_a)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matB, HIPBLASLT_MATRIX_LAYOUT_OFFSET, &offset_b, sizeof(offset_b)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matC, HIPBLASLT_MATRIX_LAYOUT_OFFSET, &offset_c, sizeof(offset_c)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        matD, HIPBLASLT_MATRIX_LAYOUT_OFFSET, &offset_d, sizeof(offset_d)));

    // Find algorithm
    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    size_t prefMaxWorkspaceSize = 128 * 1024 * 1024;  // 128 MB
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &prefMaxWorkspaceSize, sizeof(prefMaxWorkspaceSize)));

    // Support testing multiple solutions via requested_solution_num parameter
    // Default to 1 (test only best solution) for backward compatibility
    // Use HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM when -1 to get all available solutions
    int32_t requestedAlgos = (arg.requested_solution_num < 0) ? HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM
                           : (arg.requested_solution_num == 0) ? 1
                           : arg.requested_solution_num;

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult(requestedAlgos);
    int                                           numAlgos = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul_desc, matA, matB, matC, matD, pref, requestedAlgos, heuristicResult.data(), &numAlgos));

    if(numAlgos == 0)
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
        GTEST_SKIP() << "No algorithm found for this configuration";
    }

    // Find maximum workspace size across all solutions
    size_t maxWorkspaceSize = 0;
    for(int i = 0; i < numAlgos; i++)
    {
        maxWorkspaceSize = std::max(maxWorkspaceSize, heuristicResult[i].workspaceSize);
    }

    // Allocate workspace
    void* d_workspace = nullptr;
    if(maxWorkspaceSize > 0)
    {
        CHECK_HIP_ERROR(hipMalloc(&d_workspace, maxWorkspaceSize));
    }

    // ========================================
    // CPU Reference: D = alpha * A * B + beta * C
    // Computed once before testing any solutions
    // ========================================
    for(int b = 0; b < batch_count; b++)
    {
        // Get pointers to sub-matrices (with element offset applied)
        // Base is at padding, then add offset (which may be negative)
        Ti* A_sub = h_A_full.data() + b * size_A_full + padding_a + offset_a;
        Ti* B_sub = h_B_full.data() + b * size_B_full + padding_b + offset_b;
        To* C_sub = h_C_full.data() + b * size_C_full + padding_c + offset_c;
        To* D_sub = h_D_gold.data() + b * size_D_sub;

        const ptrdiff_t aRowStride    = transA == HIPBLAS_OP_N ? 1 : static_cast<ptrdiff_t>(lda);
        const ptrdiff_t aColumnStride = transA == HIPBLAS_OP_N ? static_cast<ptrdiff_t>(lda) : 1;
        const ptrdiff_t bRowStride    = transB == HIPBLAS_OP_N ? 1 : static_cast<ptrdiff_t>(ldb);
        const ptrdiff_t bColumnStride = transB == HIPBLAS_OP_N ? static_cast<ptrdiff_t>(ldb) : 1;

        using namespace roc::host_validation;
        using namespace roc::host_validation::hipblaslt_adapter;
        auto storageElements
            = [](size_t rows, size_t columns, ptrdiff_t rowStride, ptrdiff_t columnStride) {
                  if(rows == 0 || columns == 0)
                      return size_t(0);
                  return size_t(1) + (rows - 1) * size_t(std::abs(rowStride))
                         + (columns - 1) * size_t(std::abs(columnStride));
              };

        GemmProblem problem(
            GemmOperand(
                tensorView(A_sub,
                           storageElements(size_t(M), size_t(K), aRowStride, aColumnStride),
                           Layout(Shape{size_t(M), size_t(K)}, {aRowStride, aColumnStride}))),
            GemmOperand(
                tensorView(B_sub,
                           storageElements(size_t(K), size_t(N), bRowStride, bColumnStride),
                           Layout(Shape{size_t(K), size_t(N)}, {bRowStride, bColumnStride}))),
            tensorView(C_sub,
                       storageElements(size_t(M), size_t(N), 1, static_cast<ptrdiff_t>(ldc)),
                       Layout(Shape{size_t(M), size_t(N)}, {1, static_cast<ptrdiff_t>(ldc)})),
            mutableTensorView(
                D_sub,
                storageElements(size_t(M), size_t(N), 1, static_cast<ptrdiff_t>(ldd)),
                Layout(Shape{size_t(M), size_t(N)}, {1, static_cast<ptrdiff_t>(ldd)})),
            scalarType<Tc>());
        problem.epilogue.alpha = {static_cast<double>(h_alpha), 0.0};
        problem.epilogue.beta  = {static_cast<double>(h_beta), 0.0};
        referenceGemm(problem);
    }

    // Tolerance: epsilon * factor * K (accumulation over K elements)
    double tol = std::numeric_limits<Tc>::epsilon() * 100 * K;

    // Track validation results across all solutions
    int numPassed = 0;
    int numFailed = 0;

    // Test each solution
    for(int algoIdx = 0; algoIdx < numAlgos; algoIdx++)
    {
        // Reset only D output buffer to sentinel values before testing each solution
        // A, B, C are inputs and don't change, so we can reuse them
        CHECK_HIP_ERROR(hipMemcpy(d_D_full,
                                  h_D_full.data(),
                                  sizeof(To) * size_D_full * batch_count,
                                  hipMemcpyHostToDevice));

        CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                              matmul_desc,
                                              &h_alpha,
                                              d_batch_A,
                                              matA,
                                              d_batch_B,
                                              matB,
                                              &h_beta,
                                              d_batch_C,
                                              matC,
                                              d_batch_D,
                                              matD,
                                              &heuristicResult[algoIdx].algo,
                                              d_workspace,
                                              heuristicResult[algoIdx].workspaceSize,
                                              0));

        // Ensure kernel completes before reading result
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Copy GPU result back for verification
        CHECK_HIP_ERROR(hipMemcpy(h_D_full.data(),
                                  d_D_full,
                                  sizeof(To) * size_D_full * batch_count,
                                  hipMemcpyDeviceToHost));

        // ========================================
        // VALIDATION: Compare GPU vs CPU
        // ========================================
        double max_error = 0.0;
        bool   all_close = true;
        for(int b = 0; b < batch_count; b++)
        {
            // GPU result is at (base + offset) within each batch's buffer
            // Base is at padding_d, then add offset_d (which may be negative)
            To* result_gpu = h_D_full.data() + b * size_D_full + padding_d + offset_d;
            To* result_cpu = h_D_gold.data() + b * size_D_sub;

            const auto comparison = roc::host_validation::compare(
                roc::host_validation::ConstMatrixView<To>(
                    result_gpu, size_t(M), size_t(N), 1, static_cast<ptrdiff_t>(ldd)),
                roc::host_validation::ConstMatrixView<To>(
                    result_cpu, size_t(M), size_t(N), 1, static_cast<ptrdiff_t>(ldd)),
                {.absoluteTolerance = std::nextafter(tol, 0.0),
                 .maxReportedMismatches = 0});
            max_error = std::max(max_error, comparison.maxAbsoluteDifference);
            all_close = all_close && comparison.passed();
        }

        // Check and count per-solution results
        if(arg.unit_check)
        {
            if(!all_close)
            {
                numFailed++;
                EXPECT_LT(max_error, tol)
                    << "Solution " << algoIdx << "/" << numAlgos
                    << " FAILED (error: " << max_error << ", tol: " << tol << ")";
            }
            else
            {
                numPassed++;
            }
        }
    }

    // Report summary when testing multiple solutions
    if(numAlgos > 1 && arg.unit_check)
    {
        hipblaslt_cout << "Tested " << numAlgos << " solutions: "
                       << numPassed << " passed, " << numFailed << " failed" << std::endl;
    }

    // Cleanup
    if(d_workspace)
    {
        CHECK_HIP_ERROR(hipFree(d_workspace));
    }
    CHECK_HIP_ERROR(hipFree(d_batch_A));
    CHECK_HIP_ERROR(hipFree(d_batch_B));
    CHECK_HIP_ERROR(hipFree(d_batch_C));
    CHECK_HIP_ERROR(hipFree(d_batch_D));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul_desc));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
}

// Type dispatcher based on Arguments
void testing_matmul_batch_offset(const Arguments& arg)
{
    // Dispatch based on data types in Arguments
    // For now, support only f32, f16, bf16 with matching input/output types
    if(arg.a_type == HIP_R_32F && arg.b_type == HIP_R_32F &&
       arg.c_type == HIP_R_32F && arg.d_type == HIP_R_32F)
    {
        testing_matmul_batch_offset_impl<float, float, float>(arg);
    }
    else if(arg.a_type == HIP_R_16F && arg.b_type == HIP_R_16F &&
            arg.c_type == HIP_R_16F && arg.d_type == HIP_R_16F)
    {
        testing_matmul_batch_offset_impl<hipblasLtHalf, hipblasLtHalf, float>(arg);
    }
    else if(arg.a_type == HIP_R_16BF && arg.b_type == HIP_R_16BF &&
            arg.c_type == HIP_R_16BF && arg.d_type == HIP_R_16BF)
    {
        testing_matmul_batch_offset_impl<hip_bfloat16, hip_bfloat16, float>(arg);
    }
    else
    {
        GTEST_SKIP() << "Unsupported type combination for batch_offset test";
    }
}
