// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <mosaic/layout.hpp>
#include <mosaic/utils.hpp>  // MOSAIC_CHECK, error::exception_t
#include <vector>

using fp16_t = _Float16;
using fp16x4_t = __attribute__((__vector_size__(4 * sizeof(fp16_t)))) fp16_t;
using fp32x4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;

// MFMA shape for v_mfma_f32_16x16x16f16: D[MxN] = A[MxK] * B[KxN]
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;
constexpr int WAVE = 64;

// ===========================================================================
// Each operand has a thread-value (TV) layout: (tid, vid) -> row-major flat
// element index into the operand. The layouts encode exactly the same lane
// pattern that the MFMA instruction physically enforces.
//
// Reading the layouts:
//
//   A (M x K = 16 x 16, row-major). Each lane holds 4 fp16 contiguous in K.
//
//     A_tv = ((16, 4), 4) : ((16, 4), 1)
//
//     - tid decomposes (colex) into (tid_lo, tid_hi) where
//         tid_lo = tid % 16  in [0, 16)   -- selects the M-row
//         tid_hi = tid / 16  in [0,  4)   -- selects which K-chunk of 4
//     - vid in [0, 4)                    -- offset within that K-chunk
//     - flat = tid_lo*16 + tid_hi*4 + vid*1
//            = row * 16   + col   (row=tid_lo, col=4*tid_hi+vid)
//
//   B (K x N = 16 x 16, row-major). Each lane holds 4 fp16 strided by N=16
//   along K.
//
//     B_tv = ((16, 4), 4) : ((1, 64), 16)
//
//     - tid_lo selects N-col, tid_hi selects which K-chunk of 4
//     - flat = tid_lo*1 + tid_hi*64 + vid*16
//            = (4*tid_hi + vid) * 16 + tid_lo
//            = k * 16 + n  (k=4*tid_hi+vid, n=tid_lo)
//
//   D (M x N = 16 x 16, row-major fp32). Same shape/stride pattern as B.
//
//     D_tv = ((16, 4), 4) : ((1, 64), 16)
//
// ===========================================================================

__global__ void mfma_16x16x16_fp16_via_layouts(const fp16_t* __restrict__ A,
                                               const fp16_t* __restrict__ B,
                                               float* __restrict__ D) {
    using namespace mosaic;

    auto const A_tv =
        Layout(make_shape(make_shape(16u, 4u), 4u), make_stride(make_stride(16u, 4u), 1u));
    auto const B_tv =
        Layout(make_shape(make_shape(16u, 4u), 4u), make_stride(make_stride(1u, 64u), 16u));
    auto const D_tv =
        Layout(make_shape(make_shape(16u, 4u), 4u), make_stride(make_stride(1u, 64u), 16u));

    fp16x4_t a_reg;
    fp16x4_t b_reg;
    fp32x4_t c_reg = {0.0f, 0.0f, 0.0f, 0.0f};

    unsigned const tid = threadIdx.x;

#pragma unroll
    for (int v = 0; v < 4; ++v) a_reg[v] = A[A_tv(tid, v)];
#pragma unroll
    for (int v = 0; v < 4; ++v) b_reg[v] = B[B_tv(tid, v)];

    c_reg = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_reg, 0, 0, 0);

#pragma unroll
    for (int v = 0; v < 4; ++v) D[D_tv(tid, v)] = c_reg[v];
}

__global__ void mfma_16x16x16_fp16_reference(const fp16_t* __restrict__ A,
                                             const fp16_t* __restrict__ B, float* __restrict__ D) {
    fp16x4_t a_reg;
    fp16x4_t b_reg;
    fp32x4_t c_reg = {};

    a_reg =
        *reinterpret_cast<const fp16x4_t*>(A + 4 * (threadIdx.x / 16) + 16 * (threadIdx.x % 16));

    for (int i = 0; i < 4; i++) {
        b_reg[i] = *(B + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64);
    }

    c_reg = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_reg, 0, 0, 0);

    for (int i = 0; i < 4; i++) {
        *(D + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64) = c_reg[i];
    }
}

void cpu_matmul(const fp16_t* A, const fp16_t* B, float* D) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<float>(A[m * K + k]) * static_cast<float>(B[k * N + n]);
            }
            D[m * N + n] = acc;
        }
    }
}

int verify(const float* got, const float* ref, char const* label) {
    int mismatches = 0;
    for (int i = 0; i < M * N; ++i) {
        float const diff = std::fabs(got[i] - ref[i]);
        float const tol = 1e-2f * std::max(std::fabs(ref[i]), 1.0f);
        if (diff > tol) {
            if (mismatches < 5) {
                std::fprintf(
                    stderr,
                    "  [%s] mismatch at flat=%d (m=%d, n=%d): got %.4f, ref %.4f  (|diff|=%.4f)\n",
                    label, i, i / N, i % N, got[i], ref[i], diff);
            }
            ++mismatches;
        }
    }
    if (mismatches == 0) {
        std::printf("  [%s] OK  (%d cells, all within tol)\n", label, M * N);
    } else {
        std::printf("  [%s] FAIL: %d / %d cells mismatch\n", label, mismatches, M * N);
    }
    return mismatches;
}

// ---------------------------------------------------------------------------
// Pretty-print the top-left 4x4 corner of D for spot-checking.
// ---------------------------------------------------------------------------

void print_corner(char const* label, const float* D) {
    std::printf("  %-15s ", label);
    for (int n = 0; n < 4; ++n) std::printf("%8.2f ", D[0 * N + n]);
    std::printf("\n");
    for (int m = 1; m < 4; ++m) {
        std::printf("  %-15s ", "");
        for (int n = 0; n < 4; ++n) std::printf("%8.2f ", D[m * N + n]);
        std::printf("\n");
    }
}

int main() {
    std::vector<fp16_t> h_A(M * K), h_B(K * N);
    std::vector<float> h_D_ref(M * N, 0.0f);
    std::vector<float> h_D_gpu_layouts(M * N, 0.0f);
    std::vector<float> h_D_gpu_reference(M * N, 0.0f);

    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) h_A[m * K + k] = static_cast<fp16_t>(((m + k) % 7) * 0.25f);

    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) h_B[k * N + n] = static_cast<fp16_t>(((k * 3 + n) % 5) * 0.5f);

    cpu_matmul(h_A.data(), h_B.data(), h_D_ref.data());

    fp16_t* d_A = nullptr;
    fp16_t* d_B = nullptr;
    float* d_D = nullptr;

    try {
        MOSAIC_CHECK(hipMalloc(&d_A, M * K * sizeof(fp16_t)));
        MOSAIC_CHECK(hipMalloc(&d_B, K * N * sizeof(fp16_t)));
        MOSAIC_CHECK(hipMalloc(&d_D, M * N * sizeof(float)));

        MOSAIC_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(fp16_t), hipMemcpyHostToDevice));
        MOSAIC_CHECK(hipMemcpy(d_B, h_B.data(), K * N * sizeof(fp16_t), hipMemcpyHostToDevice));

        std::printf("MFMA 16x16x16 fp16 via mosaic TV layouts\n");
        std::printf("==========================================\n");
        std::printf("Shapes: A[%dx%d] * B[%dx%d] = D[%dx%d]   (1 wavefront, %d lanes)\n\n", M, K, K,
                    N, M, N, WAVE);

        // -- Kernel A: layout-based --
        MOSAIC_CHECK(hipMemset(d_D, 0, M * N * sizeof(float)));
        mfma_16x16x16_fp16_via_layouts<<<1, WAVE>>>(d_A, d_B, d_D);
        MOSAIC_CHECK(hipGetLastError());
        MOSAIC_CHECK(hipDeviceSynchronize());
        MOSAIC_CHECK(
            hipMemcpy(h_D_gpu_layouts.data(), d_D, M * N * sizeof(float), hipMemcpyDeviceToHost));
        const int fails_layouts = verify(h_D_gpu_layouts.data(), h_D_ref.data(), "via_layouts");

        // -- Kernel B: reference (sanity cross-check) --
        MOSAIC_CHECK(hipMemset(d_D, 0, M * N * sizeof(float)));
        mfma_16x16x16_fp16_reference<<<1, WAVE>>>(d_A, d_B, d_D);
        MOSAIC_CHECK(hipGetLastError());
        MOSAIC_CHECK(hipDeviceSynchronize());
        MOSAIC_CHECK(
            hipMemcpy(h_D_gpu_reference.data(), d_D, M * N * sizeof(float), hipMemcpyDeviceToHost));
        const int fails_reference = verify(h_D_gpu_reference.data(), h_D_ref.data(), "reference");

        std::printf("\nTop-left 4x4 of D for spot-check:\n");
        print_corner("cpu_ref:", h_D_ref.data());
        print_corner("via_layouts:", h_D_gpu_layouts.data());
        print_corner("reference:", h_D_gpu_reference.data());

        MOSAIC_CHECK(hipFree(d_A));
        MOSAIC_CHECK(hipFree(d_B));
        MOSAIC_CHECK(hipFree(d_D));

        return (fails_layouts == 0 && fails_reference == 0) ? 0 : 1;
    } catch (std::exception const& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        if (d_A) (void)hipFree(d_A);
        if (d_B) (void)hipFree(d_B);
        if (d_D) (void)hipFree(d_D);
        return 1;
    }
}
