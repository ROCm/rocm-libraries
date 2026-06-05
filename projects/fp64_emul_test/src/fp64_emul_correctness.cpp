// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * fp64_emul_correctness.cpp
 *
 * Correctness test for the FP64 DGEMM emulation (Ozaki Scheme II).
 *
 * Sweeps over:
 *   transA, transB ∈ {N, T}              → 4 transpose combinations
 *   alpha           ∈ {-2, -1, 0, 1, 2}  → 5 alpha values
 *   beta            ∈ {-2, -1, 0, 1, 2}  → 5 beta values
 *
 * Each combination is run TWICE:
 *   Round 1: leading dimension  ld = N      (contiguous, no padding)
 *   Round 2: leading dimension  ld = N + 3  (non-aligned padding)
 *
 * For each of the 200 cases:
 *   D_ref  = native  DGEMM(alpha, op(A), op(B), beta, C)
 *   D_emul = emulated DGEMM(alpha, op(A), op(B), beta, C)
 *   Compares the N×N active block element-wise:
 *     max_relerr = max_i |D_emul[i] - D_ref[i]| / max(1, |D_ref[i]|) ≤ RELTOL
 *
 * A square N×N matrix (default N=128) is used for all combinations so
 * the active dimensions never change across transpose variants.  The matrices
 * A, B, C are filled with deterministic sine/cosine values in the active
 * region; the padding columns are left uninitialised (they are never read).
 *
 * The ld = N+3 round exercises arbitrary leading dimensions in all four
 * kernels (extraction, scale, finalize) and verifies that the emulation
 * correctly decouples its internal INT8 padded strides from the caller's lda.
 *
 * Uses only the public hipBLASLt API — no internal headers.
 *
 * Exit code: 0 if all PASS, 1 if any FAIL.
 */

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* =========================================================================
 * Error-check macros
 * ========================================================================= */
#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t _e = (expr);                                                \
        if(_e != hipSuccess) {                                                 \
            std::fprintf(stderr, "HIP error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, hipGetErrorString(_e));           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while(0)

#define HLT_CHECK(expr)                                                        \
    do {                                                                       \
        hipblasStatus_t _s = (expr);                                           \
        if(_s != HIPBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "hipBLASLt error %s:%d: status=%d\n",        \
                         __FILE__, __LINE__, static_cast<int>(_s));            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while(0)

/* =========================================================================
 * run_gemm — execute one N×N DGEMM via hipBLASLt.
 *
 * All four matrices use the same leading dimension `ld`.  For a square
 * N×N matrix stored column-major, element (row, col) is at offset
 * row + col * ld.  The physical allocation must be at least ld * N doubles.
 *
 * Uses implicit algo selection (algo=nullptr) — acceptable overhead for a
 * correctness test.
 * ========================================================================= */
static void run_gemm(hipblasLtHandle_t   handle,
                     hipblasOperation_t  transA,
                     hipblasOperation_t  transB,
                     int64_t             N,
                     int64_t             ld,      /* leading dim for all matrices */
                     double              alpha,
                     const double*       d_A,
                     const double*       d_B,
                     double              beta,
                     const double*       d_C,
                     double*             d_D,
                     void*               workspace,
                     size_t              workspace_bytes,
                     hipStream_t         stream)
{
    hipblasLtMatmulDesc_t   desc    = nullptr;
    hipblasLtMatrixLayout_t layoutA = nullptr;
    hipblasLtMatrixLayout_t layoutB = nullptr;
    hipblasLtMatrixLayout_t layoutC = nullptr;
    hipblasLtMatrixLayout_t layoutD = nullptr;

    /* Matmul descriptor: FP64 compute and scale type */
    HLT_CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F));
    HLT_CHECK(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    HLT_CHECK(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    /* Square N×N matrices: N active rows × N active columns, ld ≥ N.    */
    const uint64_t uN = static_cast<uint64_t>(N);
    HLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_64F, uN, uN, ld));
    HLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_64F, uN, uN, ld));
    HLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_64F, uN, uN, ld));
    HLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_64F, uN, uN, ld));

    HLT_CHECK(hipblasLtMatmul(
        handle, desc,
        &alpha, d_A, layoutA,
                d_B, layoutB,
        &beta,  d_C, layoutC,
                d_D, layoutD,
        nullptr, workspace, workspace_bytes, stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    hipblasLtMatrixLayoutDestroy(layoutD);
    hipblasLtMatrixLayoutDestroy(layoutC);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtMatmulDescDestroy(desc);
}

/* =========================================================================
 * run_round — execute one full sweep (100 combos) with a given ld.
 *
 * All device buffers must be allocated for ld×N doubles.
 * The active N×N block of A, B, C must be filled before calling.
 * ========================================================================= */
static int run_round(hipblasLtHandle_t  h_native,
                     hipblasLtHandle_t  h_emul,
                     int64_t            N,
                     int64_t            ld,
                     const double*      d_A,
                     const double*      d_B,
                     const double*      d_C,
                     double*            d_D_ref,
                     double*            d_D_emul,
                     std::vector<double>& h_D_ref,
                     std::vector<double>& h_D_emul,
                     void*              workspace,
                     size_t             workspace_bytes,
                     hipStream_t        stream,
                     int&               n_pass,
                     int&               n_fail)
{
    constexpr double RELTOL = 1e-10;

    const hipblasOperation_t trans_opts[] = { HIPBLAS_OP_N, HIPBLAS_OP_T };
    const double alpha_vals[]             = { -2.0, -1.0, 0.0, 1.0, 2.0 };
    const double beta_vals[]              = { -2.0, -1.0, 0.0, 1.0, 2.0 };

    const size_t N2 = static_cast<size_t>(N) * static_cast<size_t>(N);

    for(auto transA : trans_opts) {
        for(auto transB : trans_opts) {
            for(double alpha : alpha_vals) {
                for(double beta : beta_vals) {

                    run_gemm(h_native, transA, transB, N, ld,
                             alpha, d_A, d_B, beta, d_C, d_D_ref,
                             workspace, workspace_bytes, stream);

                    run_gemm(h_emul, transA, transB, N, ld,
                             alpha, d_A, d_B, beta, d_C, d_D_emul,
                             workspace, workspace_bytes, stream);

                    /* Copy only the active N×N block (first N rows of each
                     * column).  When ld > N the d_D arrays use strided layout
                     * — copy column by column.                               */
                    const size_t col_bytes = static_cast<size_t>(N) * sizeof(double);
                    for(int64_t col = 0; col < N; ++col) {
                        HIP_CHECK(hipMemcpy(
                            h_D_ref.data()  + col * N,
                            d_D_ref         + col * ld,
                            col_bytes, hipMemcpyDeviceToHost));
                        HIP_CHECK(hipMemcpy(
                            h_D_emul.data() + col * N,
                            d_D_emul        + col * ld,
                            col_bytes, hipMemcpyDeviceToHost));
                    }

                    double max_relerr = 0.0;
                    for(size_t k = 0; k < N2; ++k) {
                        const double denom  = std::max(1.0, std::abs(h_D_ref[k]));
                        const double relerr = std::abs(h_D_emul[k] - h_D_ref[k]) / denom;
                        if(relerr > max_relerr) max_relerr = relerr;
                    }

                    const bool pass = (max_relerr <= RELTOL);
                    pass ? ++n_pass : ++n_fail;

                    std::printf("%-6s %-6s %+5.0f %+5.0f  %4lld  %-12.2e  %s\n",
                                (transA == HIPBLAS_OP_N ? "N" : "T"),
                                (transB == HIPBLAS_OP_N ? "N" : "T"),
                                alpha, beta,
                                static_cast<long long>(ld),
                                max_relerr,
                                pass ? "PASS" : "FAIL ***");
                }
            }
        }
    }
    return n_fail;
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(int argc, char** argv)
{
    int64_t N = 128;
    if(argc >= 2) N = static_cast<int64_t>(std::strtol(argv[1], nullptr, 10));
    if(N <= 0 || N > 4096) {
        std::fprintf(stderr, "Usage: %s [N]  (1 ≤ N ≤ 4096, default 128)\n", argv[0]);
        return EXIT_FAILURE;
    }

    /* Round 1: ld = N (contiguous)
     * Round 2: ld = N+3 (non-aligned padding, intentionally not a multiple
     *                    of 128 to stress the INT8 padding logic)          */
    const int64_t LD1 = N;
    const int64_t LD2 = N + 3;

    /* Largest allocation needed: LD2 * N doubles per matrix */
    const size_t alloc_elems = static_cast<size_t>(LD2) * static_cast<size_t>(N);
    const size_t N2          = static_cast<size_t>(N) * static_cast<size_t>(N);

    /* ── Device info ──────────────────────────────────────────────────── */
    {
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, 0));
        std::fprintf(stderr, "Device : %s\n", prop.name);
        std::fprintf(stderr, "N      : %lld  (ld rounds: %lld and %lld)\n\n",
                     static_cast<long long>(N),
                     static_cast<long long>(LD1),
                     static_cast<long long>(LD2));
    }

    /* ── GPU allocations (sized for the larger ld = N+3) ─────────────── */
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    double *d_D_ref = nullptr, *d_D_emul = nullptr;

    HIP_CHECK(hipMalloc(&d_A,      alloc_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B,      alloc_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C,      alloc_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_D_ref,  alloc_elems * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_D_emul, alloc_elems * sizeof(double)));

    /* Zero the full allocation so padding does not affect results.        */
    HIP_CHECK(hipMemset(d_A,      0, alloc_elems * sizeof(double)));
    HIP_CHECK(hipMemset(d_B,      0, alloc_elems * sizeof(double)));
    HIP_CHECK(hipMemset(d_C,      0, alloc_elems * sizeof(double)));

    /* ── Host buffer for filling (sized for the larger ld) ───────────── */
    /* fill_matrices() fills the N×N active block using the given stride
     * so that element (row,col) = sin((row*1 + col*7)*freq + phase).
     * The same formula is used for both rounds so the logical matrix
     * values are identical; only the physical layout changes.            */
    std::vector<double> h_buf(alloc_elems, 0.0);

    auto fill_matrices = [&](int64_t ld) {
        auto upload = [&](double* d_dst, double freq, double phase) {
            std::fill(h_buf.begin(), h_buf.end(), 0.0);
            for(int64_t col = 0; col < N; ++col)
                for(int64_t row = 0; row < N; ++row)
                    h_buf[row + col * ld] =
                        std::sin(static_cast<double>(row + col * 7) * freq + phase);
            HIP_CHECK(hipMemcpy(d_dst, h_buf.data(),
                                static_cast<size_t>(ld) * static_cast<size_t>(N) * sizeof(double),
                                hipMemcpyHostToDevice));
        };
        upload(d_A, 0.300, 1.0);
        upload(d_B, 0.173, 2.0);
        upload(d_C, 0.511, 3.0);
    };

    /* ── Workspace ────────────────────────────────────────────────────── */
    constexpr size_t WS_BYTES = 64ull << 20;   /* 64 MiB */
    void* d_ws = nullptr;
    HIP_CHECK(hipMalloc(&d_ws, WS_BYTES));

    /* ── hipBLASLt handles ────────────────────────────────────────────── */
    hipblasLtHandle_t h_native = nullptr, h_emul = nullptr;

    HLT_CHECK(hipblasLtCreate(&h_native));
    HLT_CHECK(hipblasLtSetEmulationEnabled(h_native, false));

    HLT_CHECK(hipblasLtCreate(&h_emul));
    HLT_CHECK(hipblasLtSetEmulationEnabled(h_emul, true));
    HLT_CHECK(hipblasLtSetEmulationStrategy(h_emul, HIPBLASLT_EMULATION_STRATEGY_EAGER));
    HLT_CHECK(hipblasLtSetEmulationSpecialValuesSupport(h_emul, 0u));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    /* ── Host result buffers (active N×N block only) ─────────────────── */
    std::vector<double> h_D_ref(N2), h_D_emul(N2);

    /* ── Print header ─────────────────────────────────────────────────── */
    std::printf("%-6s %-6s %5s %5s  %4s  %-12s  %s\n",
                "transA", "transB", "alpha", "beta", "ld", "max_relerr", "result");
    std::printf("%s\n", std::string(58, '-').c_str());

    int n_pass = 0, n_fail = 0;

    /* ── Round 1: ld = N  (fill with stride N) ───────────────────────── */
    fill_matrices(LD1);
    run_round(h_native, h_emul, N, LD1,
              d_A, d_B, d_C, d_D_ref, d_D_emul,
              h_D_ref, h_D_emul,
              d_ws, WS_BYTES, stream,
              n_pass, n_fail);

    std::printf("%s\n", std::string(58, '-').c_str());

    /* ── Round 2: ld = N+3  (re-fill with stride N+3) ───────────────── */
    fill_matrices(LD2);
    run_round(h_native, h_emul, N, LD2,
              d_A, d_B, d_C, d_D_ref, d_D_emul,
              h_D_ref, h_D_emul,
              d_ws, WS_BYTES, stream,
              n_pass, n_fail);

    /* ── Summary ──────────────────────────────────────────────────────── */
    const int n_total = n_pass + n_fail;
    std::printf("%s\n", std::string(58, '-').c_str());
    std::printf("Result: %d / %d PASS%s\n",
                n_pass, n_total,
                n_fail ? "  ← FAILURES DETECTED" : "  (all OK)");

    /* ── Cleanup ──────────────────────────────────────────────────────── */
    HIP_CHECK(hipStreamDestroy(stream));
    HLT_CHECK(hipblasLtDestroy(h_emul));
    HLT_CHECK(hipblasLtDestroy(h_native));
    HIP_CHECK(hipFree(d_ws));
    HIP_CHECK(hipFree(d_D_emul));
    HIP_CHECK(hipFree(d_D_ref));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_A));

    return (n_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
