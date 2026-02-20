/* **************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "rocblas_utility.hpp"

#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define ROCSOLVER_MFMA_ENABLED 1
#else
#define ROCSOLVER_MFMA_ENABLED 0
#endif // defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)

ROCSOLVER_BEGIN_NAMESPACE

#if ROCSOLVER_MFMA_ENABLED

template <typename T>
struct mfma_16x16x4_base
{
    using RegT = T;
    using AccT = __attribute__((__vector_size__(4 * sizeof(T)))) T;
};

template <typename T>
struct mfma_16x16x4;

// float specialization
template <>
struct mfma_16x16x4<float> : public mfma_16x16x4_base<float>
{
    __device__ inline auto operator()(const RegT& a, const RegT& b, const AccT& c) const
    {
        return __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
    }
};

// double specialization
template <>
struct mfma_16x16x4<double> : public mfma_16x16x4_base<double>
{
    __device__ inline auto operator()(const RegT& a, const RegT& b, const AccT& c) const
    {
        return __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
    }
};

// complex specialization
template <typename T>
struct mfma_16x16x4
{
    using RegT = T;
    using AccT = std::array<T, 4>;

    using S = decltype(std::real(T{}));
    using RegS = typename mfma_16x16x4_base<S>::RegT;
    using AccS = typename mfma_16x16x4_base<S>::AccT;

    __device__ inline auto operator()(const RegT& a, const RegT& b, const AccT& c) const
    {
        RegS ar = a.real();
        RegS ai = a.imag();
        RegS br = b.real();
        RegS bi = b.imag();
        AccS cr = {c[0].real(), c[1].real(), c[2].real(), c[3].real()};
        AccS ci = {c[0].imag(), c[1].imag(), c[2].imag(), c[3].imag()};
        AccS zero = {0};

        const auto mfma_S = mfma_16x16x4<S>();

        // real x real
        auto arbr = mfma_S(ar, br, zero);

        // real x imag
        auto arbi = mfma_S(ar, bi, zero);

        // imag x real
        auto aibr = mfma_S(ai, br, zero);

        // imag x imag
        auto aibi = mfma_S(ai, bi, zero);

        // cr += r x r - i x i
        cr += arbr - aibi;
        // ci += r x i + i x r
        ci += arbi + aibr;

        return AccT{rocblas_complex_num<S>(cr[0], ci[0]), rocblas_complex_num<S>(cr[1], ci[1]),
                    rocblas_complex_num<S>(cr[2], ci[2]), rocblas_complex_num<S>(cr[3], ci[3])};
    }
};

template <typename T,
          typename I,
          std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, rocblas_float_complex>, int> = 0>
__device__ inline I get_c_col(I li, I lj, I gpri, I inc_C, I ldc)
{
    return lj;
}

template <typename T,
          typename I,
          std::enable_if_t<std::is_same_v<T, double> || std::is_same_v<T, rocblas_double_complex>, int> = 0>
__device__ inline I get_c_col(I li, I lj, I gpri, I inc_C, I ldc)
{
    return lj;
}

template <typename T,
          typename I,
          std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, rocblas_float_complex>, int> = 0>
__device__ inline I get_c_row(I li, I lj, I gpri, I inc_C, I ldc)
{
    return gpri + li * 4;
}

template <typename T,
          typename I,
          std::enable_if_t<std::is_same_v<T, double> || std::is_same_v<T, rocblas_double_complex>, int> = 0>
__device__ inline I get_c_row(I li, I lj, I gpri, I inc_C, I ldc)
{
    return gpri * 4 + li;
}

/** GEMM device function to compute C = alpha * A * B + beta * C.

    Where C is an m x n matrix, A is an m x p matrix, and B is an
    p x n matrix. This is a wave function, every lane of the wave
    must perform call this function.

    MFMA instruction element and register mapping tool:
    https://github.com/ROCm/amd_matrix_instruction_calculator

    transA      form of op(A).
    transB      form of op(B).
    m           number of rows of matrix C.
                0 < m <= 16
    n           number of columns of matrix C.
                0 < n <= 16
    p           number of columns of matrix op(A) and number of rows of matrix op(B).
                0 < p
    alpha       scalar alpha.
    A           pointer to matrix A.
    inc_A       stride from the start of one row to the next of matrix A.
    lda         leading dimension of A.
    B           pointer to matrix B.
    inc_B       stride from the start of one row to the next of matrix B.
    ldb         leading dimension of B.
    C           pointer to matrix C.
    inc_C       stride from the start of one row to the next of matrix C.
    ldc         leading dimension of C.

**/
// Run with warpSize sized block
template <typename T, typename I>
__device__ void gemm_16x16xp(rocblas_operation transA,
                             rocblas_operation transB,
                             I m,
                             I n,
                             I p,
                             T alpha,
                             const T* A,
                             I inc_A,
                             I lda,
                             const T* B,
                             I inc_B,
                             I ldb,
                             T beta,
                             T* C,
                             I inc_C,
                             I ldc)
{
    using T4 = typename mfma_16x16x4<T>::AccT;

    const I lid = threadIdx.x % warpSize;

    const I cmajor_i_16x4 = lid % 16;
    const I cmajor_j_16x4 = lid / 16;

    const I cmajor_i_4x16 = lid % 4;
    const I cmajor_j_4x16 = lid / 4;

    const I rmajor_i_4x16 = cmajor_j_16x4;
    const I rmajor_j_4x16 = cmajor_i_16x4;

    // addresses to transpose B from col-major to row-major
    // and transpose C from row-major to col-major
    const auto c2r_src = rmajor_j_4x16 * 4 + rmajor_i_4x16;
    const auto r2c_src = cmajor_i_4x16 * 16 + cmajor_j_4x16;

    T4 dmn = {0};

    for(I kb = 0; kb < p; kb += 4)
    {
        // read A and B in col-major
        T amk = 0;
        T bkn = 0;

        // load A
        if(transA == rocblas_operation_none)
        {
            // read col major 16x4 A
            if(cmajor_i_16x4 < m && (kb + cmajor_j_16x4) < p)
                amk = A[(kb + cmajor_j_16x4) * lda + cmajor_i_16x4 * inc_A];
        }
        else
        {
            // read col major 4x16 op(A)
            if(cmajor_j_4x16 < m && (kb + cmajor_i_4x16) < p)
                amk = A[cmajor_j_4x16 * lda + (kb + cmajor_i_4x16) * inc_A];

            // transpose op(A) to 16x4
            amk = shfl(amk, c2r_src);
        }

        // load B
        if(transB == rocblas_operation_none)
        {
            // read col major 4x16 B
            if(cmajor_j_4x16 < n && (kb + cmajor_i_4x16) < p)
                bkn = B[cmajor_j_4x16 * ldb + (kb + cmajor_i_4x16) * inc_B];

            // transpose B to row major
            bkn = shfl(bkn, c2r_src);
        }
        else
        {
            // read col major 16x4 op(B)
            if(cmajor_i_16x4 < n && (kb + cmajor_j_16x4) < p)
                bkn = B[(kb + cmajor_j_16x4) * ldb + cmajor_i_16x4 * inc_B];
        }

        if constexpr(rocblas_is_complex<T>)
        {
            if(transA == rocblas_operation_conjugate_transpose)
                amk = conj(amk);
            if(transB == rocblas_operation_conjugate_transpose)
                bkn = conj(bkn);
        }

        dmn = mfma_16x16x4<T>()(amk, bkn, dmn);
    }

#pragma unroll
    for(I i = 0; i < 4; ++i)
    {
        const I c_col = get_c_col<T>(cmajor_i_4x16, cmajor_j_4x16, i, inc_C, ldc);
        const I c_row = get_c_row<T>(cmajor_i_4x16, cmajor_j_4x16, i, inc_C, ldc);
        const I idx = (c_col * ldc) + (c_row * inc_C);

        // transpose C to col major
        dmn[i] = shfl(dmn[i], r2c_src);

        if(c_col < n && c_row < m)
            C[idx] = alpha * dmn[i] + beta * C[idx];
    }
}

#endif // ROCSOLVER_MFMA_ENABLED

ROCSOLVER_END_NAMESPACE
