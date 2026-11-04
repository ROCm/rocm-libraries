// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * Standalone General Batched GEMM Example using hipBLASLt
 *
 * This sample demonstrates how to use hipBLASLt to perform General Batched GEMM operations.
 * General Batched GEMM uses pointer arrays to reference individual matrices for each batch,
 * unlike Strided Batched GEMM which uses a single base pointer with uniform strides.
 *
 * Operation: D = alpha * op(A) * op(B) + beta * C
 *
 * Where:
 *   - Each batch has its own matrix data stored separately
 *   - Matrices are referenced via device pointer arrays
 *   - Batch mode is set to HIPBLASLT_BATCH_MODE_POINTER_ARRAY
 */

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

// Helper macro for checking HIP errors
#define HIP_CHECK(cmd)                                                                           \
    do {                                                                                         \
        hipError_t error = (cmd);                                                                \
        if(error != hipSuccess) {                                                                \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                                  \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while(0)

// Helper macro for checking hipBLASLt errors
#define HIPBLASLT_CHECK(cmd)                                                                     \
    do {                                                                                         \
        hipblasStatus_t status = (cmd);                                                          \
        if(status != HIPBLAS_STATUS_SUCCESS) {                                                   \
            std::cerr << "hipBLASLt error: " << status << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                              \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while(0)

// Simple host GEMM for verification
void host_gemm(hipblasOperation_t        trans_a,
               hipblasOperation_t        trans_b,
               int                       m,
               int                       n,
               int                       k,
               float                     alpha,
               const std::vector<float>& a,
               int                       lda,
               const std::vector<float>& b,
               int                       ldb,
               float                     beta,
               const std::vector<float>& c,
               int                       ldc,
               std::vector<float>&       d,
               int                       ldd)
{
    for(int col = 0; col < n; ++col)
    {
        for(int row = 0; row < m; ++row)
        {
            float accum = 0.0f;
            for(int inner = 0; inner < k; ++inner)
            {
                float a_val, b_val;

                // Fetch A element based on transpose
                if(trans_a == HIPBLAS_OP_N)
                    a_val = a[row + inner * lda];
                else
                    a_val = a[inner + row * lda];

                // Fetch B element based on transpose
                if(trans_b == HIPBLAS_OP_N)
                    b_val = b[inner + col * ldb];
                else
                    b_val = b[col + inner * ldb];

                accum += a_val * b_val;
            }
            d[row + col * ldd] = alpha * accum + beta * c[row + col * ldc];
        }
    }
}

// Initialize matrix with pattern
void initialize_matrix(std::vector<float>& matrix, int rows, int cols, int ld, float seed)
{
    matrix.resize(ld * cols);
    for(int col = 0; col < cols; ++col)
    {
        for(int row = 0; row < rows; ++row)
        {
            matrix[row + col * ld] = seed + static_cast<float>(row + 1) * 0.25f
                                     + static_cast<float>(col + 1) * 0.5f;
        }
    }
}

// Compare two vectors with tolerance
bool verify_results(const std::vector<float>& result,
                    const std::vector<float>& reference,
                    float                     tolerance = 1.0e-4f)
{
    if(result.size() != reference.size())
        return false;

    for(size_t i = 0; i < result.size(); ++i)
    {
        if(std::fabs(result[i] - reference[i]) > tolerance)
        {
            std::cerr << "Mismatch at index " << i << ": result=" << result[i]
                      << ", reference=" << reference[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    // ============================================================================
    // Problem configuration
    // ============================================================================
    const hipblasOperation_t trans_a = HIPBLAS_OP_N;  // No transpose for A
    const hipblasOperation_t trans_b = HIPBLAS_OP_N;  // No transpose for B

    const int m = 128;                // Rows of op(A) and D
    const int n = 64;                 // Columns of op(B) and D
    const int k = 96;                 // Columns of op(A) and rows of op(B)

    const int lda = m;                // Leading dimension of A
    const int ldb = k;                // Leading dimension of B
    const int ldc = m;                // Leading dimension of C
    const int ldd = m;                // Leading dimension of D

    const int batch_count = 4;        // Number of batches
    const hipblasLtBatchMode_t batch_mode = HIPBLASLT_BATCH_MODE_POINTER_ARRAY;

    const float alpha = 1.5f;         // Scalar alpha
    const float beta  = -0.5f;        // Scalar beta

    const size_t max_workspace_bytes = 64 * 1024 * 1024;  // 64 MB workspace limit

    std::cout << "=== general batched GEMM example ===" << std::endl;
    std::cout << "Problem size: M=" << m << ", N=" << n << ", K=" << k << std::endl;
    std::cout << "Batch count: " << batch_count << std::endl;
    std::cout << "Alpha=" << alpha << ", Beta=" << beta << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // Initialize hipBLASLt
    // ============================================================================
    hipblasLtHandle_t handle;
    HIPBLASLT_CHECK(hipblasLtCreate(&handle));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // ============================================================================
    // Allocate and initialize host matrices for each batch
    // ============================================================================
    std::vector<std::vector<float>> h_a(batch_count);
    std::vector<std::vector<float>> h_b(batch_count);
    std::vector<std::vector<float>> h_c(batch_count);
    std::vector<std::vector<float>> h_d(batch_count);
    std::vector<std::vector<float>> h_d_ref(batch_count);

    const size_t size_a = lda * k;
    const size_t size_b = ldb * n;
    const size_t size_c = ldc * n;
    const size_t size_d = ldd * n;

    for(int i = 0; i < batch_count; ++i)
    {
        initialize_matrix(h_a[i], m, k, lda, 10.0f * i + 1.0f);
        initialize_matrix(h_b[i], k, n, ldb, 20.0f * i + 2.0f);
        initialize_matrix(h_c[i], m, n, ldc, 30.0f * i + 3.0f);

        h_d[i].resize(size_d, 0.0f);
        h_d_ref[i].resize(size_d, 0.0f);

        // Compute reference on CPU
        host_gemm(trans_a, trans_b, m, n, k, alpha, h_a[i], lda, h_b[i], ldb,
                  beta, h_c[i], ldc, h_d_ref[i], ldd);
    }

    // ============================================================================
    // Allocate device memory for each batch
    // ============================================================================
    std::vector<float*> d_a(batch_count);
    std::vector<float*> d_b(batch_count);
    std::vector<float*> d_c(batch_count);
    std::vector<float*> d_d(batch_count);

    for(int i = 0; i < batch_count; ++i)
    {
        HIP_CHECK(hipMalloc(&d_a[i], sizeof(float) * size_a));
        HIP_CHECK(hipMalloc(&d_b[i], sizeof(float) * size_b));
        HIP_CHECK(hipMalloc(&d_c[i], sizeof(float) * size_c));
        HIP_CHECK(hipMalloc(&d_d[i], sizeof(float) * size_d));

        // Copy host data to device
        HIP_CHECK(hipMemcpy(d_a[i], h_a[i].data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b[i], h_b[i].data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_c[i], h_c[i].data(), sizeof(float) * size_c, hipMemcpyHostToDevice));
    }

    // ============================================================================
    // Create device pointer arrays (key for General Batched GEMM)
    // ============================================================================
    float** d_ptr_array_a;
    float** d_ptr_array_b;
    float** d_ptr_array_c;
    float** d_ptr_array_d;

    HIP_CHECK(hipMalloc(&d_ptr_array_a, sizeof(float*) * batch_count));
    HIP_CHECK(hipMalloc(&d_ptr_array_b, sizeof(float*) * batch_count));
    HIP_CHECK(hipMalloc(&d_ptr_array_c, sizeof(float*) * batch_count));
    HIP_CHECK(hipMalloc(&d_ptr_array_d, sizeof(float*) * batch_count));

    // Copy pointer arrays to device
    HIP_CHECK(hipMemcpy(d_ptr_array_a, d_a.data(), sizeof(float*) * batch_count, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ptr_array_b, d_b.data(), sizeof(float*) * batch_count, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ptr_array_c, d_c.data(), sizeof(float*) * batch_count, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ptr_array_d, d_d.data(), sizeof(float*) * batch_count, hipMemcpyHostToDevice));

    // ============================================================================
    // Create matrix layouts
    // ============================================================================
    hipblasLtMatrixLayout_t mat_a, mat_b, mat_c, mat_d;

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_a, HIP_R_32F, m, k, lda));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_b, HIP_R_32F, k, n, ldb));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_c, HIP_R_32F, m, n, ldc));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_d, HIP_R_32F, m, n, ldd));

    // Set batch count for all matrices
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        mat_a, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        mat_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        mat_c, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        mat_d, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

    // Set batch mode to General Batched (pointer array mode)
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        mat_a, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        mat_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        mat_c, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        mat_d, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batch_mode, sizeof(batch_mode)));

    // Note: Stride attributes are not used in General Batched mode,
    // as each batch has its own separate memory location

    // ============================================================================
    // Create matmul descriptor
    // ============================================================================
    hipblasLtMatmulDesc_t matmul_desc;

    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // ============================================================================
    // Create matmul preference
    // ============================================================================
    hipblasLtMatmulPreference_t pref;

    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes, sizeof(max_workspace_bytes)));

    // ============================================================================
    // Get heuristic for algorithm selection
    // ============================================================================
    hipblasLtMatmulHeuristicResult_t heuristic_result;
    int returned_algo_count = 0;

    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle,
        matmul_desc,
        mat_a,
        mat_b,
        mat_c,
        mat_d,
        pref,
        1,  // Request one algorithm
        &heuristic_result,
        &returned_algo_count));

    if(returned_algo_count == 0)
    {
        std::cerr << "Error: No suitable algorithm found!" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Algorithm selected with workspace size: " << heuristic_result.workspaceSize
              << " bytes" << std::endl;

    // ============================================================================
    // Allocate workspace if needed
    // ============================================================================
    void* workspace = nullptr;
    if(heuristic_result.workspaceSize > 0)
    {
        HIP_CHECK(hipMalloc(&workspace, heuristic_result.workspaceSize));
    }

    // ============================================================================
    // Perform General Batched GEMM
    // ============================================================================
    std::cout << "Executing general batched GEMM..." << std::endl;

    HIPBLASLT_CHECK(hipblasLtMatmul(
        handle,
        matmul_desc,
        &alpha,
        d_ptr_array_a,  // Pointer array for A matrices
        mat_a,
        d_ptr_array_b,  // Pointer array for B matrices
        mat_b,
        &beta,
        d_ptr_array_c,  // Pointer array for C matrices
        mat_c,
        d_ptr_array_d,  // Pointer array for D matrices
        mat_d,
        &heuristic_result.algo,
        workspace,
        heuristic_result.workspaceSize,
        stream));

    HIP_CHECK(hipStreamSynchronize(stream));
    std::cout << "GEMM execution completed." << std::endl;

    // ============================================================================
    // Copy results back to host and verify
    // ============================================================================
    bool all_passed = true;

    for(int i = 0; i < batch_count; ++i)
    {
        HIP_CHECK(hipMemcpy(h_d[i].data(), d_d[i], sizeof(float) * size_d, hipMemcpyDeviceToHost));

        if(!verify_results(h_d[i], h_d_ref[i]))
        {
            std::cerr << "Verification FAILED for batch " << i << std::endl;
            all_passed = false;
        }
        else
        {
            std::cout << "Batch " << i << ": PASSED" << std::endl;
        }
    }

    // ============================================================================
    // Cleanup
    // ============================================================================
    if(workspace)
        HIP_CHECK(hipFree(workspace));

    for(int i = 0; i < batch_count; ++i)
    {
        HIP_CHECK(hipFree(d_a[i]));
        HIP_CHECK(hipFree(d_b[i]));
        HIP_CHECK(hipFree(d_c[i]));
        HIP_CHECK(hipFree(d_d[i]));
    }

    HIP_CHECK(hipFree(d_ptr_array_a));
    HIP_CHECK(hipFree(d_ptr_array_b));
    HIP_CHECK(hipFree(d_ptr_array_c));
    HIP_CHECK(hipFree(d_ptr_array_d));

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_a));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_b));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_c));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_d));

    HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(matmul_desc));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceDestroy(pref));

    HIP_CHECK(hipStreamDestroy(stream));
    HIPBLASLT_CHECK(hipblasLtDestroy(handle));

    // ============================================================================
    // Final result
    // ============================================================================
    std::cout << std::endl;
    if(all_passed)
    {
        std::cout << "SUCCESS: All batches passed verification!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "FAILURE: Some batches failed verification." << std::endl;
        return EXIT_FAILURE;
    }
}
