/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#pragma once

#include <cstring>
#include <vector>

#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"

namespace matxu
{
    namespace detail
    {
        template <typename T, std::enable_if_t<!is_complex<T>, int> = 1>
        inline T conj(const T& scalar)
        {
            return scalar;
        }

        template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
        inline T conj(const T& scalar)
        {
            return T{scalar.real(), -scalar.imag()};
        }

        template <typename T, std::enable_if_t<!is_complex<T>, int> = 1>
        inline auto norm(const T& scalar)
        {
            using S = decltype(std::real(T{}));
            return S(scalar * scalar);
        }

        template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
        inline auto norm(const T& scalar)
        {
            using S = decltype(std::real(T{}));
            return S(scalar.real() * scalar.real() + scalar.imag() * scalar.imag());
        }

        template <typename T, std::enable_if_t<!is_complex<T>, int> = 1>
        inline auto abs(const T& scalar)
        {
            using S = decltype(std::real(T{}));
            return S(std::abs(scalar));
        }

        template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
        inline auto abs(const T& scalar)
        {
            using S = decltype(std::real(T{}));
            return S(std::sqrt(detail::norm(scalar)));
        }

        template <typename T>
        void lapack_gemm(
            T const* A, const int nrowsA, const int ncolsA, T const* B, const int ncolsB, T* C)
        {
            const int ldA = nrowsA;
            const int ldB = ncolsA;
            const int ldC = nrowsA;

            cpu_gemm(HIPSOLVER_OP_N,
                     HIPSOLVER_OP_N,
                     nrowsA,
                     ncolsB,
                     ncolsA,
                     T(1.),
                     const_cast<T*>(A),
                     ldA,
                     const_cast<T*>(B),
                     ldB,
                     T(0.),
                     C,
                     ldC);
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
            int                  worksize = std::min(rank, 32)
                           * std::max(nrowsX, ncolsX); // pick a workspace size that is big enough
            std::vector<T> work(worksize, T(0.)); // lapack workspace
            std::vector<T> tau(rank); // scalar factors of geqrf reflectors
            auto           operation_adjoint = is_complex<T> ? HIPSOLVER_OP_C : HIPSOLVER_OP_T;

            // Copy input into Q
            {
                [[maybe_unused]] auto mptr = memmove(Q, X, sizeof(T) * nrowsX * ncolsX);
            }

            // TODO: let lapack set its preferred worksize
            /* work.resize(worksize, T(0.)); */
            const int nrowsQ = nrowsX;
            const int ncolsQ = ncolsX;
            const int ldQ    = nrowsQ;
            cpu_geqrf<T>(nrowsQ, ncolsQ, Q, ldQ, tau.data(), work.data(), worksize);

            // Copy upper triangular part of intermediate result Q into R
            const int nrowsR = ncolsX;
            const int ncolsR = ncolsX;
            const int ldR    = nrowsR;
            {
                auto volatile mptr = memset(R, 0, sizeof(T) * nrowsR * ncolsR);
            }
            for(int i = 0; i < rank; ++i)
            {
                R[i + i * static_cast<std::int64_t>(ldR)]
                    = Q[i + i * static_cast<std::int64_t>(ldQ)];
                for(int j = i + 1; j < ncolsR; ++j)
                {
                    R[i + j * static_cast<std::int64_t>(ldR)]
                        = Q[i + j * static_cast<std::int64_t>(ldQ)];
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
            const hipsolverFillMode_t uplo = HIPSOLVER_FILL_MODE_UPPER;

            if(A == nullptr || n < 1)
            {
                return false;
            }
            [[maybe_unused]] volatile auto mptr = memcpy(U, A, sizeof(T) * n * n);

            int            info;
            int            worksize = n * n;
            std::vector<T> work(worksize, T(0.));
            int            worksize_real = n * n;
            std::vector<S> work_real(worksize_real, S(0.));
            cpu_syev_heev(HIPSOLVER_EIG_MODE_VECTOR,
                          uplo,
                          n,
                          U,
                          n,
                          D,
                          work.data(),
                          worksize,
                          work_real.data(),
                          worksize_real,
                          &info);

            return (info == 0);
        }

        // Compute eigenvalues and eigenvectors of A with lapack_*syev
        template <typename T, typename S>
        bool lapack_sym_eig_lower(T const* A, const int n, T* U, S* D)
        {
            const hipsolverFillMode_t uplo = HIPSOLVER_FILL_MODE_LOWER;

            if(A == nullptr || n < 1)
            {
                return false;
            }
            [[maybe_unused]] volatile auto mptr = memcpy(U, A, sizeof(T) * n * n);

            int            info;
            int            worksize = n * n;
            std::vector<T> work(worksize, T(0.));
            int            worksize_real = n * n;
            std::vector<S> work_real(worksize_real, S(0.));
            cpu_syev_heev(HIPSOLVER_EIG_MODE_VECTOR,
                          uplo,
                          n,
                          U,
                          n,
                          D,
                          work.data(),
                          worksize,
                          work_real.data(),
                          worksize_real,
                          &info);

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
            int            worksize_real = 5 * std::min(nrows, ncols);
            std::vector<S> work_real(worksize_real, S(0.));
            T*             Acpy;
            Acpy = (T*)malloc(sizeof(T) * nrows * ncols);
            memcpy(Acpy, A, sizeof(T) * nrows * ncols);
            cpu_gesvd('A',
                      'A',
                      nrows,
                      ncols,
                      Acpy,
                      nrows,
                      D,
                      U,
                      nrows,
                      V,
                      ncols,
                      work.data(),
                      worksize,
                      work_real.data(),
                      &info);
            free(Acpy);

            return (info == 0);
        }

    } // namespace detail

} // namespace matxu
