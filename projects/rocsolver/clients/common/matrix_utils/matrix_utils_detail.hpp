/* **************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstring>
#include <vector>

#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"

namespace matxu
{
namespace detail
{
    template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 1>
    inline T conj(const T& scalar)
    {
        return scalar;
    }

    template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
    inline T conj(const T& scalar)
    {
        return T{scalar.real(), -scalar.imag()};
    }

    template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 1>
    inline auto norm(const T& scalar)
    {
        using S = decltype(std::real(T{}));
        return S(scalar * scalar);
    }

    template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
    inline auto norm(const T& scalar)
    {
        using S = decltype(std::real(T{}));
        return S(scalar.real() * scalar.real() + scalar.imag() * scalar.imag());
    }

    template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 1>
    inline auto abs(const T& scalar)
    {
        using S = decltype(std::real(T{}));
        return S(std::abs(scalar));
    }

    template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
    inline auto abs(const T& scalar)
    {
        using S = decltype(std::real(T{}));
        return S(std::sqrt(detail::norm(scalar)));
    }

    template <typename T>
    void lapack_gemm(T const* A, const int nrowsA, const int ncolsA, T const* B, const int ncolsB, T* C)
    {
        const int ldA = nrowsA;
        const int ldB = ncolsA;
        const int ldC = nrowsA;

        cpu_gemm(rocblas_operation_none, rocblas_operation_none, nrowsA, ncolsB, ncolsA, T(1.),
                 const_cast<T*>(A), ldA, const_cast<T*>(B), ldB, T(0.), C, ldC);
    }

    //
    // Given input X, return its qr factorization
    //
    //
    // Outputs Q (nrowsX x ncolsX), R (ncolsX x ncolsX)
    //
    template <typename T>
    bool lapack_qr(T const* X, const int nrowsX, const int ncolsX, T* Q, T* R)
    {
        const int rank = std::min(nrowsX, ncolsX);
        if(rank < 1)
        {
            return false;
        }

        [[maybe_unused]] int info;
        int worksize = std::min(rank, 32)
            * std::max(nrowsX, ncolsX); // pick a workspace size that is big enough
        std::vector<T> work(worksize, T(0.)); // lapack workspace
        std::vector<T> tau(rank); // scalar factors of geqrf reflectors
        auto rocblas_operation_adjoint = rocblas_is_complex<T> ? rocblas_operation_conjugate_transpose
                                                               : rocblas_operation_transpose;

        // Copy input into Q
        {
            [[maybe_unused]] auto mptr = memmove(Q, X, sizeof(T) * nrowsX * ncolsX);
        }

        // TODO: let lapack set its preferred worksize
        /* work.resize(worksize, T(0.)); */
        const int nrowsQ = nrowsX;
        const int ncolsQ = ncolsX;
        const int ldQ = nrowsQ;
        cpu_geqrf<T>(nrowsQ, ncolsQ, Q, ldQ, tau.data(), work.data(), worksize);

        // Copy upper triangular part of intermediate result Q into R
        const int nrowsR = ncolsX;
        const int ncolsR = ncolsX;
        const int ldR = nrowsR;
        {
            auto volatile mptr = memset(R, 0, sizeof(T) * nrowsR * ncolsR);
        }
        for(int i = 0; i < rank; ++i)
        {
            R[i + i * static_cast<std::int64_t>(ldR)] = Q[i + i * static_cast<std::int64_t>(ldQ)];
            for(int j = i + 1; j < ncolsR; ++j)
            {
                R[i + j * static_cast<std::int64_t>(ldR)] = Q[i + j * static_cast<std::int64_t>(ldQ)];
            }
        }

        // Extract Q
        cpu_orgqr_ungqr<T>(nrowsQ, rank, rank, Q, ldQ, tau.data(), work.data(), worksize);

        return true;
    }

    // Compute eigenvalues and eigenvectors of A with lapack_*syev
    template <typename T, typename S>
    bool lapack_sym_eig_upper(T const* A, const int n, T* U, S* D)
    {
        const rocblas_fill uplo = rocblas_fill_upper;

        if(A == nullptr || n < 1)
        {
            return false;
        }
        [[maybe_unused]] volatile auto mptr = memcpy(U, A, sizeof(T) * n * n);

        int info;
        int worksize = n * n;
        std::vector<T> work(worksize, T(0.));
        int worksize_real = n * n;
        std::vector<S> work_real(worksize_real, S(0.));
        cpu_syev_heev(rocblas_evect_original, uplo, n, U, n, D, work.data(), worksize,
                      work_real.data(), worksize_real, &info);

        return (info == 0);
    }

    // Compute eigenvalues and eigenvectors of A with lapack_*syev
    template <typename T, typename S>
    bool lapack_sym_eig_lower(T const* A, const int n, T* U, S* D)
    {
        const rocblas_fill uplo = rocblas_fill_lower;

        if(A == nullptr || n < 1)
        {
            return false;
        }
        [[maybe_unused]] volatile auto mptr = memcpy(U, A, sizeof(T) * n * n);

        int info;
        int worksize = n * n;
        std::vector<T> work(worksize, T(0.));
        int worksize_real = n * n;
        std::vector<S> work_real(worksize_real, S(0.));
        cpu_syev_heev(rocblas_evect_original, uplo, n, U, n, D, work.data(), worksize,
                      work_real.data(), worksize_real, &info);

        return (info == 0);
    }

    // Compute singular values and singular vectors of A with lapack_*gesvd
    template <typename T, typename S>
    bool lapack_ge_svd(T const* A, const int nrows, const int ncols, T* U, S* D, T* V)
    {
        if(A == nullptr || nrows < 1 || ncols < 1)
        {
            return false;
        }

        int info;
        int worksize = 32 * std::max(1, 2 * std::min(nrows, ncols) + std::max(nrows, ncols));
        std::vector<T> work(worksize, T(0.));
        int worksize_real = 5 * std::min(nrows, ncols);
        std::vector<S> work_real(worksize_real, S(0.));
        T* Acpy;
        Acpy = (T*)malloc(sizeof(T) * nrows * ncols);
        memcpy(Acpy, A, sizeof(T) * nrows * ncols);
        cpu_gesvd(rocblas_svect_all, rocblas_svect_all, nrows, ncols, Acpy, nrows, D, U, nrows, V,
                  ncols, work.data(), worksize, work_real.data(), &info);
        free(Acpy);

        return (info == 0);
    }

} // namespace detail

} // namespace matxu
