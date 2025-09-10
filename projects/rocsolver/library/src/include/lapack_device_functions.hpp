/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib_device_helpers.hpp"
#include "lib_macros.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_logger.hpp"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    common location for device functions and kernels that reproduce LAPACK
 *    and BLAS functionality. Includes some reproduction of rocBLAS
 *    functionality since rocBLAS cannot be called from within a kernel.
 * ===========================================================================
 */

template <typename T>
__device__ void trtri_kernel_upper(const rocblas_diagonal diag,
                                   const rocblas_int n,
                                   T* a,
                                   const rocblas_int lda,
                                   rocblas_int* info,
                                   T* w)
{
    // unblocked trtri kernel assuming upper triangular matrix
    int i = hipThreadIdx_y;

    // diagonal element
    if(diag == rocblas_diagonal_non_unit && i < n)
        a[i + i * lda] = 1.0 / a[i + i * lda];
    __syncthreads();

    // compute element i of each column j
    T ajj, aij;
    for(rocblas_int j = 1; j < n; j++)
    {
        if(i < j && i < n)
            w[i] = a[i + j * lda];
        __syncthreads();

        if(i < j && i < n)
        {
            ajj = (diag == rocblas_diagonal_non_unit ? a[j + j * lda] : 1);
            aij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for(rocblas_int ii = i + 1; ii < j; ii++)
                aij += a[i + ii * lda] * w[ii];

            a[i + j * lda] = -ajj * aij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trtri_kernel_lower(const rocblas_diagonal diag,
                                   const rocblas_int n,
                                   T* a,
                                   const rocblas_int lda,
                                   rocblas_int* info,
                                   T* w)
{
    // unblocked trtri kernel assuming lower triangular matrix
    int i = hipThreadIdx_y;

    // diagonal element
    if(diag == rocblas_diagonal_non_unit && i < n)
        a[i + i * lda] = 1.0 / a[i + i * lda];
    __syncthreads();

    // compute element i of each column j
    T ajj, aij;
    for(rocblas_int j = n - 2; j >= 0; j--)
    {
        if(i > j && i < n)
            w[i] = a[i + j * lda];
        __syncthreads();

        if(i > j && i < n)
        {
            ajj = (diag == rocblas_diagonal_non_unit ? a[j + j * lda] : 1);
            aij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for(rocblas_int ii = i - 1; ii > j; ii--)
                aij += a[i + ii * lda] * w[ii];

            a[i + j * lda] = -ajj * aij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trmm_kernel_left_upper(const rocblas_diagonal diag,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       T* alpha,
                                       T* a,
                                       const rocblas_int lda,
                                       T* b,
                                       const rocblas_int ldb,
                                       T* w)
{
    // trmm kernel assuming no transpose, upper triangular matrix from the left
    // min dim for w is m
    T bij;
    for(int j = 0; j < n; j++)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
            w[i] = b[i + j * ldb];
        __syncthreads();

        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            bij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for(int k = i + 1; k < m; k++)
                bij += a[i + k * lda] * w[k];

            b[i + j * ldb] = *alpha * bij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trmm_kernel_left_lower(const rocblas_diagonal diag,
                                       const rocblas_int m,
                                       const rocblas_int n,
                                       T* alpha,
                                       T* a,
                                       const rocblas_int lda,
                                       T* b,
                                       const rocblas_int ldb,
                                       T* w)
{
    // trmm kernel assuming no transpose, lower triangular matrix from the left
    // min dim for w is m
    T bij;
    for(int j = 0; j < n; j++)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
            w[i] = b[i + j * ldb];
        __syncthreads();

        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            bij = (diag == rocblas_diagonal_non_unit ? a[i + i * lda] : 1) * w[i];

            for(int k = 0; k < i; k++)
                bij += a[i + k * lda] * w[k];

            b[i + j * ldb] = *alpha * bij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trsm_kernel_right_upper(const rocblas_diagonal diag,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        T* alpha,
                                        T* a,
                                        const rocblas_int lda,
                                        T* b,
                                        const rocblas_int ldb)
{
    // trsm kernel assuming no transpose, upper triangular matrix from the right
    T ajj, bij;
    for(int j = 0; j < n; j++)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            ajj = (diag == rocblas_diagonal_non_unit ? 1.0 / a[j + j * lda] : 1);
            bij = *alpha * b[i + j * ldb];

            for(int k = 0; k < j; k++)
                bij -= a[k + j * lda] * b[i + k * ldb];

            b[i + j * ldb] = ajj * bij;
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void trsm_kernel_right_lower(const rocblas_diagonal diag,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        T* alpha,
                                        T* a,
                                        const rocblas_int lda,
                                        T* b,
                                        const rocblas_int ldb)
{
    // trsm kernel assuming no transpose, lower triangular matrix from the right
    T ajj, bij;
    for(int j = n - 1; j >= 0; j--)
    {
        for(int i = hipThreadIdx_y; i < m; i += hipBlockDim_y)
        {
            ajj = (diag == rocblas_diagonal_non_unit ? 1.0 / a[j + j * lda] : 1);
            bij = *alpha * b[i + j * ldb];

            for(int k = j + 1; k < n; k++)
                bij -= a[k + j * lda] * b[i + k * ldb];

            b[i + j * ldb] = ajj * bij;
        }
        __syncthreads();
    }
}

/** LARTG device function computes the sine (s) and cosine (c) values
    to create a givens rotation such that:
    [  c s ]' * [ f ] = [ r ]
    [ -s c ]    [ g ]   [ 0 ] **/
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ __host__ void lartg(T& f, T& g, T& c, T& s, T& r)
{
    if(g == 0)
    {
        c = 1;
        s = 0;
        r = f;
    }
    else if(f == 0)
    {
        c = 0;
        s = 1;
        r = -g;
    }
    else
    {
        T t;
        if(std::abs(g) > std::abs(f))
        {
            t = -f / g;
            s = 1 / std::sqrt(1 + t * t);
            c = s * t;
        }
        else
        {
            t = -g / f;
            c = 1 / std::sqrt(1 + t * t);
            s = c * t;
        }
        r = c * f - s * g;
    }
}

/** LASR device function applies a sequence of rotations P(i) i=1,2,...z
    to a m-by-n matrix A from either the left (P*A with z=m) or the right (A*P'
    with z=n). P = P(z-1)*...*P(1) if forward direction, P = P(1)*...*P(z-1) if
    backward direction. **/
template <typename T, typename W>
__device__ void lasr(const rocblas_side side,
                     const rocblas_direct direc,
                     const rocblas_int m,
                     const rocblas_int n,
                     W* c,
                     W* s,
                     T* A,
                     const rocblas_int lda)
{
    T temp;
    W cs, sn;

    if(side == rocblas_side_left)
    {
        if(direc == rocblas_forward_direction)
        {
            for(rocblas_int i = 0; i < m - 1; ++i)
            {
                for(rocblas_int j = 0; j < n; ++j)
                {
                    temp = A[i + j * lda];
                    cs = c[i];
                    sn = s[i];
                    A[i + j * lda] = cs * temp + sn * A[i + 1 + j * lda];
                    A[i + 1 + j * lda] = cs * A[i + 1 + j * lda] - sn * temp;
                }
            }
        }
        else
        {
            for(rocblas_int i = m - 1; i > 0; --i)
            {
                for(rocblas_int j = 0; j < n; ++j)
                {
                    temp = A[i + j * lda];
                    cs = c[i - 1];
                    sn = s[i - 1];
                    A[i + j * lda] = cs * temp - sn * A[i - 1 + j * lda];
                    A[i - 1 + j * lda] = cs * A[i - 1 + j * lda] + sn * temp;
                }
            }
        }
    }

    else
    {
        if(direc == rocblas_forward_direction)
        {
            for(rocblas_int j = 0; j < n - 1; ++j)
            {
                for(rocblas_int i = 0; i < m; ++i)
                {
                    temp = A[i + j * lda];
                    cs = c[j];
                    sn = s[j];
                    A[i + j * lda] = cs * temp + sn * A[i + (j + 1) * lda];
                    A[i + (j + 1) * lda] = cs * A[i + (j + 1) * lda] - sn * temp;
                }
            }
        }
        else
        {
            for(rocblas_int j = n - 1; j > 0; --j)
            {
                for(rocblas_int i = 0; i < m; ++i)
                {
                    temp = A[i + j * lda];
                    cs = c[j - 1];
                    sn = s[j - 1];
                    A[i + j * lda] = cs * temp - sn * A[i + (j - 1) * lda];
                    A[i + (j - 1) * lda] = cs * A[i + (j - 1) * lda] + sn * temp;
                }
            }
        }
    }
}

/** LAE2 computes the eigenvalues of a 2x2 symmetric matrix
    [ a b ]
    [ b c ] **/
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ __host__ void lae2(T& a, T& b, T& c, T& rt1, T& rt2)
{
    T sm = a + c;
    T adf = abs(a - c);
    T ab = abs(b + b);

    T rt, acmx, acmn;
    if(adf > ab)
    {
        rt = ab / adf;
        rt = adf * sqrt(1 + rt * rt);
    }
    else if(adf < ab)
    {
        rt = adf / ab;
        rt = ab * sqrt(1 + rt * rt);
    }
    else
        rt = ab * sqrt(2);

    // Compute the eigenvalues
    if(abs(a) > abs(c))
    {
        acmx = a;
        acmn = c;
    }
    else
    {
        acmx = c;
        acmn = a;
    }
    if(sm < 0)
    {
        rt1 = T(0.5) * (sm - rt);
        rt2 = T((acmx / (double)rt1) * acmn - (b / (double)rt1) * b);
    }
    else if(sm > 0)
    {
        rt1 = T(0.5) * (sm + rt);
        rt2 = T((acmx / (double)rt1) * acmn - (b / (double)rt1) * b);
    }
    else
    {
        rt1 = T(0.5) * rt;
        rt2 = T(-0.5) * rt;
    }
}

/** LAEV2 computes the eigenvalues and eigenvectors of a 2x2 symmetric matrix
    [ a b ]
    [ b c ] **/
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ __host__ void laev2(T& a, T& b, T& c, T& rt1, T& rt2, T& cs1, T& sn1)
{
    int sgn1, sgn2;

    T sm = a + c;
    T df = a - c;
    T adf = abs(df);
    T tb = b + b;
    T ab = abs(tb);

    T rt, temp1, temp2;
    if(adf > ab)
    {
        rt = ab / adf;
        rt = adf * sqrt(1 + rt * rt);
    }
    else if(adf < ab)
    {
        rt = adf / ab;
        rt = ab * sqrt(1 + rt * rt);
    }
    else
        rt = ab * sqrt(2);

    // Compute the eigenvalues
    if(abs(a) > abs(c))
    {
        temp1 = a;
        temp2 = c;
    }
    else
    {
        temp1 = c;
        temp2 = a;
    }
    if(sm < 0)
    {
        sgn1 = -1;
        rt1 = T(0.5) * (sm - rt);
        rt2 = T((temp1 / (double)rt1) * temp2 - (b / (double)rt1) * b);
    }
    else if(sm > 0)
    {
        sgn1 = 1;
        rt1 = T(0.5) * (sm + rt);
        rt2 = T((temp1 / (double)rt1) * temp2 - (b / (double)rt1) * b);
    }
    else
    {
        sgn1 = 1;
        rt1 = T(0.5) * rt;
        rt2 = T(-0.5) * rt;
    }

    // Compute the eigenvector
    if(df >= 0)
    {
        temp1 = df + rt;
        sgn2 = 1;
    }
    else
    {
        temp1 = df - rt;
        sgn2 = -1;
    }

    if(abs(temp1) > ab)
    {
        // temp2 is cotan
        temp2 = -tb / temp1;
        sn1 = T(1) / sqrt(1 + temp2 * temp2);
        cs1 = temp2 * sn1;
    }
    else
    {
        if(ab == 0)
        {
            cs1 = 1;
            sn1 = 0;
        }
        else
        {
            // temp2 is tan
            temp2 = -temp1 / tb;
            cs1 = T(1) / sqrt(1 + temp2 * temp2);
            sn1 = temp2 * cs1;
        }
    }

    if(sgn1 == sgn2)
    {
        temp1 = cs1;
        cs1 = -sn1;
        sn1 = temp1;
    }
}

/** LASRT_INCREASING sorts an array D in increasing order.
    stack is a 32x2 array of integers on the device. **/
template <typename T>
__device__ void lasrt_increasing(const rocblas_int n, T* D, rocblas_int* stack)
{
    /** (TODO: Current implementation is failling for large sizes. Not removed for now
        as quick-sort methods could be required for performance purposes in the future.
        It should be debugged some time.) **/
    T d1, d2, d3, dmnmx, temp;
    constexpr rocblas_int select = 20;
    constexpr rocblas_int lds = 32;
    rocblas_int i, j, start, endd;
    rocblas_int stackptr = 0;

    // Initialize stack[0, 0] and stack[1, 0]
    stack[0 + 0 * lds] = 0;
    stack[1 + 0 * lds] = n - 1;
    while(stackptr >= 0)
    {
        start = stack[0 + stackptr * lds];
        endd = stack[1 + stackptr * lds];
        stackptr--;

        if(endd - start <= select && endd - start > 0)
        {
            // Insertion sort
            for(i = start + 1; i <= endd; i++)
            {
                for(j = i; j > start; j--)
                {
                    if(D[j] < D[j - 1])
                    {
                        dmnmx = D[j];
                        D[j] = D[j - 1];
                        D[j - 1] = dmnmx;
                    }
                    else
                        break;
                }
            }
        }
        else if(endd - start > select)
        {
            // Partition and add to stack
            d1 = D[start];
            d2 = D[endd];
            i = (start + endd) / 2;
            d3 = D[i];

            if(d1 < d2)
            {
                if(d3 < d1)
                    dmnmx = d1;
                else if(d3 < d2)
                    dmnmx = d3;
                else
                    dmnmx = d2;
            }
            else
            {
                if(d3 < d2)
                    dmnmx = d2;
                else if(d3 < d1)
                    dmnmx = d3;
                else
                    dmnmx = d1;
            }

            i = start;
            j = endd;
            while(i < j)
            {
                while(D[i] < dmnmx)
                    i++;
                while(D[j] > dmnmx)
                    j--;
                if(i < j)
                {
                    temp = D[i];
                    D[i] = D[j];
                    D[j] = temp;
                }
            }
            if(j - start > endd - j - 1)
            {
                stackptr++;
                stack[0 + stackptr * lds] = start;
                stack[1 + stackptr * lds] = j;
                stackptr++;
                stack[0 + stackptr * lds] = j + 1;
                stack[1 + stackptr * lds] = endd;
            }
            else
            {
                stackptr++;
                stack[0 + stackptr * lds] = j + 1;
                stack[1 + stackptr * lds] = endd;
                stackptr++;
                stack[0 + stackptr * lds] = start;
                stack[1 + stackptr * lds] = j;
            }
        }
    }
}

/** IAMAX finds the maximum element of a given vector.
    MAX_THDS should be 128, 256, 512, or 1024, and sval should
    be a shared array of size MAX_THDS. **/
template <int MAX_THDS, typename T, typename I, typename S>
__device__ void iamax(const I tid, const I n, T* A, const I incA, S* sval)
{
    // local memory setup
    S val1, val2;

    // read into shared memory while doing initial step
    // (each thread reduce as many elements as needed to cover the original array)
    val1 = 0;
    for(I i = tid; i < n; i += MAX_THDS)
    {
        val2 = aabs<S>(A[i * incA]);
        if(val1 < val2)
            val1 = val2;
    }
    sval[tid] = val1;
    __syncthreads();

    if(n <= 1)
        return;

        /** <========= Next do the reduction on the shared memory array =========>
        (We halve the number of active threads at each step
        reducing two elements in the shared array. **/

#pragma unroll
    for(I i = MAX_THDS / 2; i > warpSize; i /= 2)
    {
        if(tid < i)
        {
            val2 = sval[tid + i];
            if(val1 < val2)
                sval[tid] = val1 = val2;
        }
        __syncthreads();
    }

    // from this point, as all the active threads will form a single wavefront
    // and work in lock-step, there is no need for synchronizations and barriers
    if(tid < warpSize)
    {
        if(warpSize >= 64)
        {
            val2 = sval[tid + 64];
            if(val1 < val2)
                sval[tid] = val1 = val2;
        }
        val2 = sval[tid + 32];
        if(val1 < val2)
            sval[tid] = val1 = val2;
        val2 = sval[tid + 16];
        if(val1 < val2)
            sval[tid] = val1 = val2;
        val2 = sval[tid + 8];
        if(val1 < val2)
            sval[tid] = val1 = val2;
        val2 = sval[tid + 4];
        if(val1 < val2)
            sval[tid] = val1 = val2;
        val2 = sval[tid + 2];
        if(val1 < val2)
            sval[tid] = val1 = val2;
        val2 = sval[tid + 1];
        if(val1 < val2)
            sval[tid] = val1 = val2;
    }

    // after the reduction, the maximum of the elements is in sval[0]
}

/** IAMAX finds the maximum element of a given vector and its index.
    MAX_THDS should be 64, 128, 256, 512, or 1024, and sval and sidx should
    be shared arrays of size MAX_THDS. **/
template <int MAX_THDS, typename T, typename I, typename S>
__device__ void iamax(const I tid, const I n, T* A, const I incA, S* sval, I* sidx)
{
    // local memory setup
    S val1, val2;
    I idx1, idx2;

    // read into shared memory while doing initial step
    // (each thread reduce as many elements as needed to cover the original array)
    val1 = 0;
    idx1 = INT_MAX;
    for(I i = tid; i < n; i += MAX_THDS)
    {
        val2 = aabs<S>(A[i * incA]);
        idx2 = i + 1; // add one to make it 1-based index
        if(val1 < val2 || idx1 == INT_MAX)
        {
            val1 = val2;
            idx1 = idx2;
        }
    }
    sval[tid] = val1;
    sidx[tid] = idx1;
    __syncthreads();

    if(n <= 1)
        return;

        /** <========= Next do the reduction on the shared memory array =========>
        (We halve the number of active threads at each step
        reducing two elements in the shared array. **/

#pragma unroll
    for(I i = MAX_THDS / 2; i > warpSize; i /= 2)
    {
        if(tid < i)
        {
            val2 = sval[tid + i];
            idx2 = sidx[tid + i];
            if((val1 < val2) || (val1 == val2 && idx1 > idx2))
            {
                sval[tid] = val1 = val2;
                sidx[tid] = idx1 = idx2;
            }
        }
        __syncthreads();
    }

    // from this point, as all the active threads will form a single wavefront
    // and work in lock-step, there is no need for synchronizations and barriers
    if(tid < warpSize)
    {
        if(warpSize >= 64 && MAX_THDS >= 128)
        {
            val2 = sval[tid + 64];
            idx2 = sidx[tid + 64];
            if((val1 < val2) || (val1 == val2 && idx1 > idx2))
            {
                sval[tid] = val1 = val2;
                sidx[tid] = idx1 = idx2;
            }
        }
        val2 = sval[tid + 32];
        idx2 = sidx[tid + 32];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 16];
        idx2 = sidx[tid + 16];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 8];
        idx2 = sidx[tid + 8];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 4];
        idx2 = sidx[tid + 4];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 2];
        idx2 = sidx[tid + 2];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 1];
        idx2 = sidx[tid + 1];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
    }

    // after the reduction, the maximum of the elements is in sval[0] and sidx[0]
}

/** NRM2 finds the euclidean norm of a given vector.
    MAX_THDS should be 128, 256, 512, or 1024, and sval should
    be a shared array of size MAX_THDS. **/
template <int MAX_THDS, typename T>
__device__ void nrm2(const rocblas_int tid, const rocblas_int n, T* A, const rocblas_int incA, T* sval)
{
    // local memory setup
    T val = 0;

    // read into shared memory while doing initial step
    // (each thread reduce as many elements as needed to cover the original array)
    for(int i = tid; i < n; i += MAX_THDS)
        val = val + A[i * incA] * A[i * incA];
    sval[tid] = val;
    __syncthreads();

    if(n <= 1)
    {
        if(tid == 0)
            sval[0] = sqrt(sval[0]);
        return;
    }

    /** <========= Next do the reduction on the shared memory array =========>
        (We halve the number of active threads at each step
        reducing two elements in the shared array. **/

#pragma unroll
    for(int i = MAX_THDS / 2; i > warpSize; i /= 2)
    {
        if(tid < i)
            val = val + sval[tid + i];
        __syncthreads();
        if(tid < i)
            sval[tid] = val;
        __syncthreads();
    }

    // from this point, as all the active threads will form a single wavefront
    // and work in lock-step, there is no need for synchronizations and barriers
    if(tid < warpSize)
    {
        if(warpSize >= 64 && MAX_THDS >= 128)
        {
            sval[tid] = sval[tid] + sval[tid + 64];
            __threadfence();
        }
        sval[tid] = sval[tid] + sval[tid + 32];
        __threadfence();
        sval[tid] = sval[tid] + sval[tid + 16];
        __threadfence();
        sval[tid] = sval[tid] + sval[tid + 8];
        __threadfence();
        sval[tid] = sval[tid] + sval[tid + 4];
        __threadfence();
        sval[tid] = sval[tid] + sval[tid + 2];
        __threadfence();
        sval[tid] = sval[tid] + sval[tid + 1];
        __threadfence();
    }

    // after the reduction, the euclidean norm of the elements is in sval[0]
    if(tid == 0)
        sval[0] = sqrt(sval[0]);
}

/** DOT finds the dot product of vectors x and y (or conj(y)).
    MAX_THDS should be 64, 128, 256, 512, or 1024, and sval should
    be a shared array of size MAX_THDS. **/
template <int MAX_THDS, bool CONJY, typename T>
__device__ void dot(const rocblas_int tid,
                    const rocblas_int n,
                    T* x,
                    const rocblas_int incX,
                    T* y,
                    const rocblas_int incY,
                    T* sval)
{
    // local memory setup
    T val = 0;

    // read into shared memory while doing initial step
    // (each thread reduce as many elements as needed to cover the original array)
    for(int i = tid; i < n; i += MAX_THDS)
        val = val + x[i * incX] * (CONJY ? conj(y[i * incY]) : y[i * incY]);

    if(n <= 1)
    {
        if(tid == 0)
            sval[0] = val;
        return;
    }

    /** <========= Next do the reduction on the shared memory array =========> **/

    val += shift_left(val, 1);
    val += shift_left(val, 2);
    val += shift_left(val, 4);
    val += shift_left(val, 8);
    val += shift_left(val, 16);
    if(warpSize > 32)
        val += shift_left(val, 32);
    if(tid % warpSize == 0)
        sval[tid / warpSize] = val;
    __syncthreads();
    if(tid == 0)
    {
        for(int k = 1; k < MAX_THDS / warpSize; k++)
            val += sval[k];

        sval[0] = val;
    }

    // after the reduction, the dot product is in sval[0]
}

/** LAGTF computes an LU factorization of a matrix T - lambda*I, where T
    is a tridiagonal matrix and lambda is a scalar. **/
template <typename T>
__device__ void lagtf(rocblas_int n, T* a, T lambda, T* b, T* c, T tol, T* d, rocblas_int* in, T eps)
{
    T scale1, scale2, piv1, piv2, mult, temp;

    a[0] = a[0] - lambda;
    in[n - 1] = 0;
    if(n == 1)
    {
        if(a[0] == 0)
            in[0] = 1;
        return;
    }

    tol = std::fmax(tol, eps);
    scale1 = abs(a[0]) + abs(b[0]);
    for(rocblas_int k = 0; k < n - 1; k++)
    {
        temp = a[k + 1] - lambda;
        a[k + 1] = temp;
        scale2 = abs(c[k]) + abs(temp);
        if(k < n - 2)
            scale2 = scale2 + abs(b[k + 1]);
        piv1 = (a[k] == 0 ? 0 : abs(a[k]) / scale1);

        if(c[k] == 0)
        {
            in[k] = 0;
            piv2 = 0;
            scale1 = scale2;
            if(k < n - 2)
                d[k] = 0;
        }
        else
        {
            piv2 = abs(c[k]) / scale2;
            if(piv2 <= piv1)
            {
                in[k] = 0;
                scale1 = scale2;
                mult = c[k] / a[k];
                c[k] = mult;
                a[k + 1] = a[k + 1] - mult * b[k];
                if(k < n - 2)
                    d[k] = 0;
            }
            else
            {
                in[k] = 1;
                mult = a[k] / c[k];
                a[k] = c[k];
                a[k + 1] = b[k] - mult * temp;
                if(k < n - 2)
                {
                    d[k] = b[k + 1];
                    b[k + 1] = -mult * b[k + 1];
                }
                b[k] = temp;
                c[k] = mult;
            }
        }

        if(std::fmax(piv1, piv2) <= tol && in[n - 1] == 0)
            in[n - 1] = k + 1;
    }

    if(abs(a[n - 1]) <= scale1 * tol && in[n - 1] == 0)
        in[n - 1] = n;
}

/** LAGTS_TYPE1_PERTURB solves the system of equations (T - lambda*I)x = y,
    where T is a tridiagonal matrix and lambda is a scalar. If overflow were
    to occur, the diagonal elements are perturbed. **/
template <typename T>
__device__ void
    lagts_type1_perturb(rocblas_int n, T* a, T* b, T* c, T* d, rocblas_int* in, T* y, T tol, T eps, T ssfmin)
{
    rocblas_int k;
    T temp, pert, ak, absak;

    T bignum = T(1) / ssfmin;
    if(tol <= 0)
    {
        tol = abs(a[0]);
        if(n > 1)
            tol = std::fmax(tol, std::fmax(abs(a[1]), abs(b[0])));
        for(k = 2; k < n; k++)
            tol = std::fmax(std::fmax(tol, abs(a[k])), std::fmax(abs(b[k - 1]), abs(d[k - 2])));
        tol = tol * eps;
        if(tol == 0)
            tol = eps;
    }

    for(k = 1; k < n; k++)
    {
        if(in[k - 1] == 0)
            y[k] = y[k] - c[k - 1] * y[k - 1];
        else
        {
            temp = y[k - 1];
            y[k - 1] = y[k];
            y[k] = temp - c[k - 1] * y[k];
        }
    }

    for(k = n - 1; k >= 0; k--)
    {
        temp = y[k];
        if(k < n - 1)
            temp = temp - b[k] * y[k + 1];
        if(k < n - 2)
            temp = temp - d[k] * y[k + 2];

        ak = a[k];
        pert = (ak >= 0 ? abs(tol) : -abs(tol));
        while((absak = abs(ak)) < 1)
        {
            if(absak < ssfmin)
            {
                if(absak == 0 || abs(temp) * ssfmin > absak)
                {
                    ak = ak + pert;
                    pert = 2 * pert;
                }
                else
                {
                    temp = temp * bignum;
                    ak = ak * bignum;
                    break;
                }
            }
            else
            {
                if(abs(temp) > absak * bignum)
                {
                    ak = ak + pert;
                    pert = 2 * pert;
                }
                else
                    break;
            }
        }

        y[k] = temp / ak;
    }
}

/** AXPY computes a constant times a vector plus a vector. **/
template <typename T, typename U, typename V>
ROCSOLVER_KERNEL void axpy_kernel(const rocblas_int n,
                                  T* alpha,
                                  const rocblas_stride stride_alpha,
                                  U X,
                                  const rocblas_int shiftX,
                                  const rocblas_int incx,
                                  const rocblas_stride strideX,
                                  V Y,
                                  const rocblas_int shiftY,
                                  const rocblas_int incy,
                                  const rocblas_stride strideY)
{
    rocblas_int b = hipBlockIdx_x;
    rocblas_int i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(i < n)
    {
        T* x = load_ptr_batch<T>(X, b, shiftX, strideX);
        T* y = load_ptr_batch<T>(Y, b, shiftY, strideY);
        T* a = alpha + b * stride_alpha;

        // axpy
        y[i * incy] = a[0] * x[i * incx] + y[i * incy];
    }
}

/** ROT applies a Givens rotation between to vector x y of dimension n.
    Launch this kernel with a desired number of threads organized in
    NG groups in the x direction with NT threads in the x direction. **/
template <typename S, typename T, typename I>
ROCSOLVER_KERNEL void
    rot_kernel(I const n, T* const x, I const incx, T* const y, I const incy, S const c, S const s)
{
    if(n <= 0)
        return;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    if((incx == 1) && (incy == 1))
    {
        // ------------
        // special case
        // ------------
        for(I i = i_start; i < n; i += i_inc)
        {
            auto const temp = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = temp;
        }
    }
    else
    {
        // ---------------------------
        // code for unequal increments
        // ---------------------------

        for(auto i = i_start; i < n; i += i_inc)
        {
            auto const ix = i * static_cast<int64_t>(incx);
            auto const iy = i * static_cast<int64_t>(incy);
            auto const temp = c * x[ix] + s * y[iy];
            y[iy] = c * y[iy] - s * x[ix];
            x[ix] = temp;
        }
    }
}

/** SCAL scales a vector x of dimension n by a factor da.
    Launch this kernel with a desired number of threads organized in
    NG groups in the x direction with NT threads in the x direction. **/
template <typename S, typename T, typename I>
ROCSOLVER_KERNEL void scal_kernel(I const n, S const da, T* const x, I const incx)
{
    if(n <= 0)
        return;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    S const zero = 0;
    bool const is_da_zero = (da == zero);
    if(incx == 1)
    {
        for(I i = i_start; i < n; i += i_inc)
        {
            x[i] = da * x[i];
        }
    }
    else
    {
        // ---------------------------
        // code for non-unit increments
        // ---------------------------

        for(I i = i_start; i < n; i += i_inc)
        {
            auto const ix = i * static_cast<int64_t>(incx);
            x[ix] = da * x[ix];
        }
    }
}

template <typename S, typename I>
__device__ I slaed6(I kniter,
                    bool orgati,
                    S rho,
                    S* d,
                    S* z,
                    S finit,
                    S& tau,
                    S eps = std::numeric_limits<S>::epsilon() / S(2.),
                    S ssfmin = std::numeric_limits<S>::min(),
                    I MAXIT = 50)
{
    auto lam_abs = [](auto x) -> auto
    {
        return std::abs(x);
    };
    auto lam_sqrt = [](auto x) -> auto
    {
        return std::sqrt(x);
    };
    auto lam_max = [](auto x, auto y, auto z) -> auto
    {
        return std::max(std::max(x, y), z);
    };
    auto lam_min = [](auto x, auto y) -> auto
    {
        return std::min(x, y);
    };

    S dscale[3]{}, zscale[3]{};
    struct X_t
    {
        S* x_;
        __device__ X_t(S* x)
            : x_(x)
        {
        }

        __device__ S& operator()(int j)
        {
            return x_[j - 1];
        }

    } D(d), Z(z), DSCALE(dscale), ZSCALE(zscale);

    bool scale;

    S a, b, c, ddf, df, erretm, eta, f, fc, sclfac, sclinv, temp, temp1, temp2, temp3, temp4, lbd,
        ubd;

    I iter, niter;

    I info = 0;

    if(orgati)
    {
        lbd = D(2);
        ubd = D(3);
    }
    else
    {
        lbd = D(1);
        ubd = D(2);
    }

    if(finit < S(0.))
    {
        lbd = S(0.);
    }
    else
    {
        ubd = S(0.);
    }

    niter = 1;
    tau = S(0.);
    if(kniter == 2)
    {
        if(orgati)
        {
            temp = (D(3) - D(2)) / S(2.);
            c = rho + Z(1) / ((D(1) - D(2)) - temp);
            a = c * (D(2) + D(3)) + Z(2) + Z(3);
            b = c * D(2) * D(3) + Z(2) * D(3) + Z(3) * D(2);
        }
        else
        {
            temp = (D(1) - D(2)) / S(2.);
            c = rho + Z(3) / ((D(3) - D(2)) - temp);
            a = c * (D(1) + D(2)) + Z(1) + Z(2);
            b = c * D(1) * D(2) + Z(1) * D(2) + Z(2) * D(1);
        }

        temp = lam_max(lam_abs(a), lam_abs(b), lam_abs(c));
        a = a / temp;
        b = b / temp;
        c = c / temp;
        if(c == S(0.))
        {
            tau = b / a;
        }
        else if(a <= S(0.))
        {
            tau = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
        }
        else
        {
            tau = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
        }

        if(tau < lbd || tau > ubd)
        {
            tau = (lbd + ubd) / S(2.);
        }
        if(D(1) == tau || D(2) == tau || D(3) == tau)
        {
            tau = S(0.);
        }
        else
        {
            temp = finit + tau * Z(1) / (D(1) * (D(1) - tau)) + tau * Z(2) / (D(2) * (D(2) - tau))
                + tau * Z(3) / (D(3) * (D(3) - tau));
            if(temp <= S(0.))
            {
                lbd = tau;
            }
            else
            {
                ubd = tau;
            }

            if(lam_abs(finit) <= lam_abs(temp))
            {
                tau = S(0.);
            }
        }
    }
    //
    //     get machine parameters for possible scaling to avoid overflow
    //
    const S small1 = std::pow(S(2.), (std::log(ssfmin) / std::log(S(2.))) / S(3.));
    const S sminv1 = S(1.) / small1;
    const S small2 = small1 * small1;
    const S sminv2 = sminv1 * sminv1;
    //
    //     Determine if scaling of inputs necessary to avoid overflow
    //     when computing 1/temp**3
    //
    if(orgati)
    {
        temp = lam_min(lam_abs(D(2) - tau), lam_abs(D(3) - tau));
    }
    else
    {
        temp = lam_min(lam_abs(D(1) - tau), lam_abs(D(2) - tau));
    }

    scale = false;
    if(temp <= small1)
    {
        scale = true;
        if(temp <= small2)
        {
            //
            //        Scale up by power of radix nearest 1/SAFMIN**(2/3)
            //
            sclfac = sminv2;
            sclinv = small2;
        }
        else
        {
            //
            //        Scale up by power of radix nearest 1/SAFMIN**(1/3)
            //
            sclfac = sminv1;
            sclinv = small1;
        }
        //
        //        Scaling up safe because D, Z, tau scaled elsewhere to be O(1)
        //
        for(int i = 1; i <= 3; ++i)
        {
            DSCALE(i) = D(i) * sclfac;
            ZSCALE(i) = Z(i) * sclfac;
        }
        tau = tau * sclfac;
        lbd = lbd * sclfac;
        ubd = ubd * sclfac;
    }
    else
    {
        //
        //        Copy D and Z to DSCALE and ZSCALE
        //
        for(int i = 1; i <= 3; ++i)
        {
            DSCALE(i) = D(i);
            ZSCALE(i) = Z(i);
        }
    }
    fc = S(0.);
    df = S(0.);
    ddf = S(0.);
    for(int i = 1; i <= 3; ++i)
    {
        temp = S(1.) / (DSCALE(i) - tau);
        temp1 = ZSCALE(i) * temp;
        temp2 = temp1 * temp;
        temp3 = temp2 * temp;
        fc = fc + temp1 / DSCALE(i);
        df = df + temp2;
        ddf = ddf + temp3;
    }
    f = finit + tau * fc;
    if(lam_abs(f) <= S(0.))
    {
        if(scale)
        {
            tau = tau * sclinv;
        }
        return info;
    }
    if(f <= S(0.))
    {
        lbd = tau;
    }
    else
    {
        ubd = tau;
    }
    //
    //        Iteration begins -- Use Gragg-Thornton-Warner cubic convergent
    //                            scheme
    //
    //     It is not hard to see that
    //
    //           1) Iterations will go up monotonically
    //              if finit < 0;
    //
    //           2) Iterations will go down monotonically
    //              if finit > 0.
    //
    iter = niter + 1;
    for(int niter = iter; niter <= MAXIT; ++niter)
    {
        if(orgati)
        {
            temp1 = DSCALE(2) - tau;
            temp2 = DSCALE(3) - tau;
        }
        else
        {
            temp1 = DSCALE(1) - tau;
            temp2 = DSCALE(2) - tau;
        }

        a = (temp1 + temp2) * f - temp1 * temp2 * df;
        b = temp1 * temp2 * f;
        c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
        temp = lam_max(lam_abs(a), lam_abs(b), lam_abs(c));
        a = a / temp;
        b = b / temp;
        c = c / temp;
        if(c == S(0.))
        {
            eta = b / a;
        }
        else if(a <= S(0.))
        {
            eta = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
        }
        else
        {
            eta = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
        }

        if(f * eta >= S(0.))
        {
            eta = -f / df;
        }

        tau = tau + eta;
        if(tau < lbd || tau > ubd)
        {
            tau = (lbd + ubd) / S(2.);
        }

        fc = S(0.);
        erretm = S(0.);
        df = S(0.);
        ddf = S(0.);
        for(int i = 1; i <= 3; ++i)
        {
            if((DSCALE(i) - tau) != S(0.))
            {
                temp = S(1.) / (DSCALE(i) - tau);
                temp1 = ZSCALE(i) * temp;
                temp2 = temp1 * temp;
                temp3 = temp2 * temp;
                temp4 = temp1 / DSCALE(i);
                fc = fc + temp4;
                erretm = erretm + lam_abs(temp4);
                df = df + temp2;
                ddf = ddf + temp3;
            }
            else
            {
                if(scale)
                {
                    tau = tau * sclinv;
                }
                return info;
            }
        }
        f = finit + tau * fc;
        erretm = S(8.) * (lam_abs(finit) + lam_abs(tau) * erretm) + lam_abs(tau) * df;
        if((lam_abs(f) <= S(4.) * eps * erretm) || ((ubd - lbd) <= S(4.) * eps * lam_abs(tau)))
        {
            if(scale)
            {
                tau = tau * sclinv;
            }
            return info;
        }
        // REVIEW
        if(f <= S(0.))
        {
            lbd = tau;
        }
        else
        {
            ubd = tau;
        }
    }
    info = 1;
    //
    //     Undo scaling
    //
    if(scale)
    {
        tau = tau * sclinv;
    }

    return info;
}

template <typename S, typename I>
__device__ I slaed4(I n,
                    I i,
                    S* delta,
                    S* z,
                    S rho,
                    S& dlam,
                    S eps = std::numeric_limits<S>::epsilon() / S(2.),
                    S ssfmin = std::numeric_limits<S>::min(),
                    I MAXIT = 50)
{
    auto lam_abs = [](auto x) -> auto
    {
        return std::abs(x);
    };
    auto lam_sqr = [](auto x) -> auto
    {
        return x * x;
    };
    auto lam_sqrt = [](auto x) -> auto
    {
        return std::sqrt(x);
    };
    auto lam_max = [](auto x, auto y) -> auto
    {
        return std::max(x, y);
    };
    auto lam_min = [](auto x, auto y) -> auto
    {
        return std::min(x, y);
    };

    i = i + 1;
    S zz[3]{};
    struct X_t
    {
        S* x_;
        __device__ X_t(S* x)
            : x_(x)
        {
        }

        __device__ S& operator()(int j)
        {
            return x_[j - 1];
        }

    } Z(z), ZZ(zz), DELTA(delta);

    S tau, eta = S(0.), dltlb, dltub;
    S psi, dpsi, phi, dphi, rhoinv, midpt;
    S del, a, b, c, w, erretm, temp, dw, temp1, prew;
    I ii, niter, iter, orgati, iim1, iip1;
    bool swtch3, swtch;

    S d1 = DELTA(1);
    S di = DELTA(i);
    S dnm1 = DELTA(n - 1);
    S dn = DELTA(n);

    rhoinv = S(1.) / rho;
    I info = 0;

    if(n == 1)
    {
        dlam = d1 + rho * Z(1) * Z(1);
        DELTA(1) = S(1.);
    }
    else if(i == n)
    {
        ii = n - 1;
        niter = 1;
        //
        //        Initial guess
        //
        //        If ||Z||_2 is not one, then midpt should be set to
        //        rho * ||Z||_2^2 / S(2.)
        //
        midpt = rho / S(2.);

        psi = S(0.);
        for(int j = 1; j <= n - 2; ++j)
        {
            psi = psi + Z(j) * Z(j) / ((DELTA(j) - di) - midpt);
        }
        c = rhoinv + psi;
        w = c + Z(ii) * Z(ii) / ((DELTA(ii) - di) - midpt) + Z(n) * Z(n) / ((dn - di) - midpt);
        if(w <= S(0.))
        {
            temp = Z(n - 1) * Z(n - 1) / (dn - dnm1 + rho) + Z(n) * Z(n) / rho;
            if(c <= temp)
            {
                tau = rho;
            }
            else
            {
                del = dn - dnm1;
                a = -c * del + Z(n - 1) * Z(n - 1) + Z(n) * Z(n);
                b = Z(n) * Z(n) * del;
                if(a < S(0.))
                {
                    tau = S(2.) * b / (lam_sqrt(a * a + S(4.) * b * c) - a);
                }
                else
                {
                    tau = (a + lam_sqrt(a * a + S(4.) * b * c)) / (S(2.) * c);
                }
            }
            //
            //           It can be proved that
            //               D(n)+rho/2 <= LAMBDA(n) < D(n)+tau <= D(n)+rho
            //
            dltlb = midpt;
            dltub = rho;
        }
        else
        {
            del = dn - dnm1;
            a = -c * del + Z(n - 1) * Z(n - 1) + Z(n) * Z(n);
            b = Z(n) * Z(n) * del;
            if(a < S(0.))
            {
                tau = S(2.) * b / (lam_sqrt(a * a + S(4.) * b * c) - a);
            }
            else
            {
                tau = (a + lam_sqrt(a * a + S(4.) * b * c)) / (S(2.) * c);
            }
            //
            //           It can be proved that
            //               D(n) < D(n)+tau < LAMBDA(n) < D(n)+rho/2
            //
            dltlb = S(0.);
            dltub = midpt;
        }
        for(int j = 1; j <= n; ++j)
        {
            DELTA(j) = (DELTA(j) - di) - tau;
        }
        //
        //        Evaluate psi and the derivative dpsi
        //
        dpsi = S(0.);
        psi = S(0.);
        erretm = S(0.);
        for(int j = 1; j <= ii; ++j)
        {
            temp = Z(j) / DELTA(j);
            psi = psi + Z(j) * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = lam_abs(erretm);
        //
        //        Evaluate phi and the derivative dphi
        //
        temp = Z(n) / DELTA(n);
        phi = Z(n) * temp;
        dphi = temp * temp;
        erretm = S(8.) * (-phi - psi) + erretm - phi + rhoinv + lam_abs(tau) * (dpsi + dphi);
        w = rhoinv + phi + psi;
        //
        //        Test for convergence
        //
        if(lam_abs(w) <= eps * erretm)
        {
            dlam = di + tau;
            return info;
        }
        if(w <= S(0.))
        {
            dltlb = lam_max(dltlb, tau);
        }
        else
        {
            dltub = lam_min(dltub, tau);
        }
        //
        //        Calculate the new step
        //
        niter = niter + 1;
        c = w - DELTA(n - 1) * dpsi - DELTA(n) * dphi;
        a = (DELTA(n - 1) + DELTA(n)) * w - DELTA(n - 1) * DELTA(n) * (dpsi + dphi);
        b = DELTA(n - 1) * DELTA(n) * w;
        // REVIEW
        if(c < S(0.))
        {
            c = lam_abs(c);
        }
        if(c <= S(0.))
        {
            eta = -w / (dpsi + dphi);
        }
        else if(a >= S(0.))
        {
            eta = (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
        }
        else
        {
            eta = S(2.) * b / (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
        }
        //
        //        Note, eta should be positive if w is negative, and
        //        eta should be negative otherwise. However,
        //        if for some reason caused by roundoff, eta*w > 0,
        //        we simply use one Newton step instead. This way
        //        will guarantee eta*w < 0.
        //
        if(w * eta > S(0.))
        {
            eta = -w / (dpsi + dphi);
        }
        temp = tau + eta;
        if(temp > dltub || temp < dltlb)
        {
            if(w < S(0.))
            {
                eta = (dltub - tau) / S(2.);
            }
            else
            {
                eta = (dltlb - tau) / S(2.);
            }
        }
        for(int j = 1; j <= n; ++j)
        {
            DELTA(j) = DELTA(j) - eta;
        }
        tau = tau + eta;
        //
        //        Evaluate psi and the derivative dpsi
        //
        dpsi = S(0.);
        psi = S(0.);
        erretm = S(0.);
        for(int j = 1; j <= ii; ++j)
        {
            temp = Z(j) / DELTA(j);
            psi = psi + Z(j) * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = lam_abs(erretm);
        //
        //        Evaluate phi and the derivative dphi
        //
        temp = Z(n) / DELTA(n);
        phi = Z(n) * temp;
        dphi = temp * temp;
        erretm = S(8.) * (-phi - psi) + erretm - phi + rhoinv + lam_abs(tau) * (dpsi + dphi);
        w = rhoinv + phi + psi;
        //
        //        Main loop to update the values of the array DELTA
        //
        iter = niter + 1;
        for(niter = iter; niter <= MAXIT; ++niter)
        {
            //
            //           Test for convergence
            //
            if(lam_abs(w) <= eps * erretm)
            {
                dlam = di + tau;
                return info;
            }
            if(w <= S(0.))
            {
                dltlb = lam_max(dltlb, tau);
            }
            else
            {
                dltub = lam_min(dltub, tau);
            }
            //
            //           Calculate the new step
            //
            c = w - DELTA(n - 1) * dpsi - DELTA(n) * dphi;
            a = (DELTA(n - 1) + DELTA(n)) * w - DELTA(n - 1) * DELTA(n) * (dpsi + dphi);
            b = DELTA(n - 1) * DELTA(n) * w;
            if(a >= S(0.))
            {
                eta = (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
            }
            else
            {
                eta = S(2.) * b / (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
            }
            //
            //           Note, eta should be positive if w is negative, and
            //           eta should be negative otherwise. However,
            //           if for some reason caused by roundoff, eta*w > 0,
            //           we simply use one Newton step instead. This way
            //           will guarantee eta*w < 0.
            //
            if(w * eta > S(0.))
            {
                eta = -w / (dpsi + dphi);
            }
            temp = tau + eta;
            if(temp > dltub || temp < dltlb)
            {
                if(w < S(0.))
                {
                    eta = (dltub - tau) / S(2.);
                }
                else
                {
                    eta = (dltlb - tau) / S(2.);
                }
            }
            for(int j = 1; j <= n; ++j)
            {
                DELTA(j) = DELTA(j) - eta;
            }
            tau = tau + eta;
            //
            //           Evaluate psi and the derivative dpsi
            //
            dpsi = S(0.);
            psi = S(0.);
            erretm = S(0.);
            for(int j = 1; j <= ii; ++j)
            {
                temp = Z(j) / DELTA(j);
                psi = psi + Z(j) * temp;
                dpsi = dpsi + temp * temp;
                erretm = erretm + psi;
            }
            erretm = lam_abs(erretm);
            //
            //           Evaluate phi and the derivative dphi
            //
            temp = Z(n) / DELTA(n);
            phi = Z(n) * temp;
            dphi = temp * temp;
            erretm = S(8.) * (-phi - psi) + erretm - phi + rhoinv + lam_abs(tau) * (dpsi + dphi);
            w = rhoinv + phi + psi;
        }
        //
        //        Return with info = 1, niter = MAXIT and not converged
        //
        info = 1;
        dlam = di + tau;
    }
    else
    {
        //
        //        The case for i < n
        //
        niter = 1;
        I ip1 = i + 1;
        S dip1 = DELTA(ip1);
        //
        //        Calculate initial guess
        //
        del = dip1 - di;
        midpt = del / S(2.);
        psi = S(0.);
        for(int j = 1; j <= i - 1; ++j)
        {
            S dj = (DELTA(j) - di) - midpt;
            psi = psi + Z(j) * Z(j) / dj;
        }
        phi = S(0.);
        for(int j = n; j >= i + 2; --j)
        {
            S dj = (DELTA(j) - di) - midpt;
            phi = phi + Z(j) * Z(j) / dj;
        }
        c = rhoinv + psi + phi;
        w = c + Z(i) * Z(i) / (-midpt) + Z(ip1) * Z(ip1) / ((dip1 - di) - midpt);
        if(w > S(0.))
        {
            //
            //           d(i) < the ith eigenvalue < (d(i)+d(i+1))/2
            //
            //           We choose d(i) as origin.
            //
            orgati = true;
            a = c * del + Z(i) * Z(i) + Z(ip1) * Z(ip1);
            b = Z(i) * Z(i) * del;
            if(a > S(0.))
            {
                tau = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
            }
            else
            {
                tau = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
            }
            dltlb = S(0.);
            dltub = midpt;
        }
        else
        {
            //
            //           (d(i)+d(i+1))/2 <= the ith eigenvalue < d(i+1)
            //
            //           We choose d(i+1) as origin.
            //
            orgati = false;
            a = c * del - Z(i) * Z(i) - Z(ip1) * Z(ip1);
            b = Z(ip1) * Z(ip1) * del;
            if(a < S(0.))
            {
                tau = S(2.) * b / (a - lam_sqrt(lam_abs(a * a + S(4.) * b * c)));
            }
            else
            {
                tau = -(a + lam_sqrt(lam_abs(a * a + S(4.) * b * c))) / (S(2.) * c);
            }
            dltlb = -midpt;
            dltub = S(0.);
        }
        if(orgati)
        {
            ii = i;
        }
        else
        {
            ii = i + 1;
        }
        iim1 = ii - 1;
        iip1 = ii + 1;
        S diim1 = DELTA(iim1);
        S diip1 = DELTA(iip1);
        if(orgati)
        {
            for(int j = 1; j <= n; ++j)
            {
                DELTA(j) = (DELTA(j) - di) - tau;
            }
        }
        else
        {
            for(int j = 1; j <= n; ++j)
            {
                DELTA(j) = (DELTA(j) - dip1) - tau;
            }
        }
        //
        //        Evaluate psi and the derivative dpsi
        //
        dpsi = S(0.);
        psi = S(0.);
        erretm = S(0.);
        for(int j = 1; j <= iim1; ++j)
        {
            temp = Z(j) / DELTA(j);
            psi = psi + Z(j) * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = lam_abs(erretm);
        //
        //        Evaluate phi and the derivative dphi
        //
        dphi = S(0.);
        phi = S(0.);
        for(int j = n; j >= iip1; --j)
        {
            temp = Z(j) / DELTA(j);
            phi = phi + Z(j) * temp;
            dphi = dphi + temp * temp;
            erretm = erretm + phi;
        }
        w = rhoinv + phi + psi;
        //
        //        w is the value of the secular function with
        //        its ii-th element removed.
        //
        swtch3 = false;
        if(orgati)
        {
            if(w < S(0.))
            {
                swtch3 = true;
            }
        }
        else
        {
            if(w > S(0.))
            {
                swtch3 = true;
            }
        }
        if(ii == 1 || ii == n)
        {
            swtch3 = false;
        }
        temp = Z(ii) / DELTA(ii);
        dw = dpsi + dphi + temp * temp;
        temp = Z(ii) * temp;
        w = w + temp;
        erretm = S(8.) * (phi - psi) + erretm + S(2.) * rhoinv + S(3.) * lam_abs(temp)
            + lam_abs(tau) * dw;
        //
        //        Test for convergence
        //
        if(lam_abs(w) <= eps * erretm)
        {
            if(orgati)
            {
                dlam = di + tau;
            }
            else
            {
                dlam = dip1 + tau;
            }

            return info;
        }
        if(w <= S(0.))
        {
            dltlb = lam_max(dltlb, tau);
        }
        else
        {
            dltub = lam_min(dltub, tau);
        }
        //
        //        Calculate the new step
        //
        niter = niter + 1;
        if(!swtch3)
        {
            if(orgati)
            {
                c = w - DELTA(ip1) * dw - (di - dip1) * lam_sqr(Z(i) / DELTA(i));
            }
            else
            {
                c = w - DELTA(i) * dw - (dip1 - di) * lam_sqr(Z(ip1) / DELTA(ip1));
            }
            a = (DELTA(i) + DELTA(ip1)) * w - DELTA(i) * DELTA(ip1) * dw;
            b = DELTA(i) * DELTA(ip1) * w;
            if(c == S(0.))
            {
                if(a == S(0.))
                {
                    if(orgati)
                    {
                        a = Z(i) * Z(i) + DELTA(ip1) * DELTA(ip1) * (dpsi + dphi);
                    }
                    else
                    {
                        a = Z(ip1) * Z(ip1) + DELTA(i) * DELTA(i) * (dpsi + dphi);
                    }
                }
                eta = b / a;
            }
            else if(a <= S(0.))
            {
                eta = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
            }
            else
            {
                eta = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
            }
        }
        else
        {
            //
            //           Interpolation using THREE most relevant poles
            //
            temp = rhoinv + psi + phi;
            if(orgati)
            {
                temp1 = Z(iim1) / DELTA(iim1);
                temp1 = temp1 * temp1;
                c = temp - DELTA(iip1) * (dpsi + dphi) - (diim1 - diip1) * temp1;
                ZZ(1) = Z(iim1) * Z(iim1);
                ZZ(3) = DELTA(iip1) * DELTA(iip1) * ((dpsi - temp1) + dphi);
            }
            else
            {
                temp1 = Z(iip1) / DELTA(iip1);
                temp1 = temp1 * temp1;
                c = temp - DELTA(iim1) * (dpsi + dphi) - (diip1 - diim1) * temp1;
                ZZ(1) = DELTA(iim1) * DELTA(iim1) * (dpsi + (dphi - temp1));
                ZZ(3) = Z(iip1) * Z(iip1);
            }
            ZZ(2) = Z(ii) * Z(ii);
            info = slaed6(niter, orgati, c, DELTA.x_ + iim1 - 1, ZZ.x_, w, eta, eps, ssfmin, MAXIT);
            if(info != 0)
            {
                return info;
            }
        }
        //
        //        Note, eta should be positive if w is negative, and
        //        eta should be negative otherwise. However,
        //        if for some reason caused by roundoff, eta*w > 0,
        //        we simply use one Newton step instead. This way
        //        will guarantee eta*w < 0.
        //
        if(w * eta >= S(0.))
        {
            eta = -w / dw;
        }
        temp = tau + eta;
        if(temp > dltub || temp < dltlb)
        {
            if(w < S(0.))
            {
                eta = (dltub - tau) / S(2.);
            }
            else
            {
                eta = (dltlb - tau) / S(2.);
            }
        }
        prew = w;
        for(int j = 1; j <= n; ++j)
        {
            DELTA(j) = DELTA(j) - eta;
        }
        //
        //        Evaluate psi and the derivative dpsi
        //
        dpsi = S(0.);
        psi = S(0.);
        erretm = S(0.);
        for(int j = 1; j <= iim1; ++j)
        {
            temp = Z(j) / DELTA(j);
            psi = psi + Z(j) * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = lam_abs(erretm);
        //
        //        Evaluate phi and the derivative dphi
        //
        dphi = S(0.);
        phi = S(0.);
        for(int j = n; j >= iip1; --j)
        {
            temp = Z(j) / DELTA(j);
            phi = phi + Z(j) * temp;
            dphi = dphi + temp * temp;
            erretm = erretm + phi;
        }
        temp = Z(ii) / DELTA(ii);
        dw = dpsi + dphi + temp * temp;
        temp = Z(ii) * temp;
        w = rhoinv + phi + psi + temp;
        erretm = S(8.) * (phi - psi) + erretm + S(2.) * rhoinv + S(3.) * lam_abs(temp)
            + lam_abs(tau + eta) * dw;
        swtch = false;
        if(orgati)
        {
            if(-w > lam_abs(prew) / S(10.))
            {
                swtch = true;
            }
        }
        else
        {
            if(w > lam_abs(prew) / S(10.))
            {
                swtch = true;
            }
        }
        tau = tau + eta;
        //
        //        Main loop to update the values of the array   DELTA
        //
        iter = niter + 1;
        for(niter = iter; niter < MAXIT; ++niter)
        {
            //
            //           Test for convergence
            //
            if(lam_abs(w) <= eps * erretm)
            {
                if(orgati)
                {
                    dlam = di + tau;
                }
                else
                {
                    dlam = dip1 + tau;
                }

                return info;
            }
            if(w <= S(0.))
            {
                dltlb = lam_max(dltlb, tau);
            }
            else
            {
                dltub = lam_min(dltub, tau);
            }
            //
            //           Calculate the new step
            //
            if(!swtch3)
            {
                if(!swtch)
                {
                    if(orgati)
                    {
                        c = w - DELTA(ip1) * dw - (di - dip1) * lam_sqr(Z(i) / DELTA(i));
                    }
                    else
                    {
                        c = w - DELTA(i) * dw - (dip1 - di) * lam_sqr(Z(ip1) / DELTA(ip1));
                    }
                }
                else
                {
                    temp = Z(ii) / DELTA(ii);
                    if(orgati)
                    {
                        dpsi = dpsi + temp * temp;
                    }
                    else
                    {
                        dphi = dphi + temp * temp;
                    }
                    c = w - DELTA(i) * dpsi - DELTA(ip1) * dphi;
                }
                a = (DELTA(i) + DELTA(ip1)) * w - DELTA(i) * DELTA(ip1) * dw;
                b = DELTA(i) * DELTA(ip1) * w;
                if(c == S(0.))
                {
                    if(a == S(0.))
                    {
                        if(!swtch)
                        {
                            if(orgati)
                            {
                                a = Z(i) * Z(i) + DELTA(ip1) * DELTA(ip1) * (dpsi + dphi);
                            }
                            else
                            {
                                a = Z(ip1) * Z(ip1) + DELTA(i) * DELTA(i) * (dpsi + dphi);
                            }
                        }
                        else
                        {
                            a = DELTA(i) * DELTA(i) * dpsi + DELTA(ip1) * DELTA(ip1) * dphi;
                        }
                    }
                    eta = b / a;
                }
                else if(a <= S(0.))
                {
                    eta = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
                }
                else
                {
                    eta = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
                }
            }
            else
            {
                //
                //              Interpolation using 3 most relevant poles
                //
                temp = rhoinv + psi + phi;
                if(swtch)
                {
                    c = temp - DELTA(iim1) * dpsi - DELTA(iip1) * dphi;
                    ZZ(1) = DELTA(iim1) * DELTA(iim1) * dpsi;
                    ZZ(3) = DELTA(iip1) * DELTA(iip1) * dphi;
                }
                else
                {
                    if(orgati)
                    {
                        temp1 = Z(iim1) / DELTA(iim1);
                        temp1 = temp1 * temp1;
                        c = temp - DELTA(iip1) * (dpsi + dphi) - (diim1 - diip1) * temp1;
                        ZZ(1) = Z(iim1) * Z(iim1);
                        ZZ(3) = DELTA(iip1) * DELTA(iip1) * ((dpsi - temp1) + dphi);
                    }
                    else
                    {
                        temp1 = Z(iip1) / DELTA(iip1);
                        temp1 = temp1 * temp1;
                        c = temp - DELTA(iim1) * (dpsi + dphi) - (diip1 - diim1) * temp1;
                        ZZ(1) = DELTA(iim1) * DELTA(iim1) * (dpsi + (dphi - temp1));
                        ZZ(3) = Z(iip1) * Z(iip1);
                    }
                }
                info = slaed6(niter, orgati, c, DELTA.x_ + iim1 - 1, ZZ.x_, w, eta, eps, ssfmin,
                              MAXIT);
                if(info != 0)
                {
                    return info;
                }
            }
            //
            //           Note, eta should be positive if w is negative, and
            //           eta should be negative otherwise. However,
            //           if for some reason caused by roundoff, eta*w > 0,
            //           we simply use one Newton step instead. This way
            //           will guarantee eta*w < 0.
            //
            if(w * eta >= S(0.))
            {
                eta = -w / dw;
            }
            temp = tau + eta;
            if(temp > dltub || temp < dltlb)
            {
                if(w < S(0.))
                {
                    eta = (dltub - tau) / S(2.);
                }
                else
                {
                    eta = (dltlb - tau) / S(2.);
                }
            }
            /* * */
            for(int j = 1; j <= n; ++j)
            {
                DELTA(j) = DELTA(j) - eta;
            }
            tau = tau + eta;
            prew = w;
            //
            //           Evaluate psi and the derivative dpsi
            //
            dpsi = S(0.);
            psi = S(0.);
            erretm = S(0.);
            for(int j = 1; j <= iim1; ++j)
            {
                temp = Z(j) / DELTA(j);
                psi = psi + Z(j) * temp;
                dpsi = dpsi + temp * temp;
                erretm = erretm + psi;
            }
            erretm = lam_abs(erretm);
            //
            //           Evaluate phi and the derivative dphi
            //
            dphi = S(0.);
            phi = S(0.);
            for(int j = n; j >= iip1; --j)
            {
                temp = Z(j) / DELTA(j);
                phi = phi + Z(j) * temp;
                dphi = dphi + temp * temp;
                erretm = erretm + phi;
            }
            temp = Z(ii) / DELTA(ii);
            dw = dpsi + dphi + temp * temp;
            temp = Z(ii) * temp;
            w = rhoinv + phi + psi + temp;
            erretm = S(8.) * (phi - psi) + erretm + S(2.) * rhoinv + S(3.) * lam_abs(temp)
                + lam_abs(tau) * dw;
            if(w * prew > S(0.) && lam_abs(w) > lam_abs(prew) / S(10.))
            {
                swtch = !swtch;
            }
        }
        //
        //        Return with info = 1, niter = MAXIT and not converged
        //
        info = 1;
        if(orgati)
        {
            dlam = di + tau;
        }
        else
        {
            dlam = dip1 + tau;
        }
    }

    return info;
}

#define MAXITERS 50 // Max number of iterations for root finding method

/** SEQ_EVAL evaluates the secular equation at a given point. It accumulates the
    corrections to the elements in D so that distance to poles are computed
   accurately **/
template <typename S>
__device__ void seq_eval(const rocblas_int type,
                         const rocblas_int k,
                         const rocblas_int dd,
                         S* D,
                         const S* z,
                         const S p,
                         const S cor,
                         S* pt_fx,
                         S* pt_fdx,
                         S* pt_gx,
                         S* pt_gdx,
                         S* pt_hx,
                         S* pt_hdx,
                         S* pt_er,
                         bool modif)
{
    S er, fx, gx, hx, fdx, gdx, hdx, zz, tmp;
    rocblas_int gout, hout;

    // prepare computations
    // if type = 0: evaluate secular equation
    if(type == 0)
    {
        gout = k + 1;
        hout = k;
    }
    // if type = 1: evaluate secular equation without the k-th pole
    else if(type == 1)
    {
        if(modif)
        {
            tmp = D[k] - cor;
            D[k] = tmp;
        }
        gout = k;
        hout = k;
    }
    // if type = 2: evaluate secular equation without the k-th and (k+1)-th poles
    else if(type == 2)
    {
        if(modif)
        {
            tmp = D[k] - cor;
            D[k] = tmp;
            tmp = D[k + 1] - cor;
            D[k + 1] = tmp;
        }
        gout = k;
        hout = k + 1;
    }
    else
    {
        // unexpected value for type, something is wrong
        assert(false);
    }

    // computations
    gx = 0;
    gdx = 0;
    er = 0;
    for(int i = 0; i < gout; ++i)
    {
        tmp = D[i] - cor;
        if(modif)
            D[i] = tmp;
        zz = z[i];
        tmp = zz / tmp;
        gx += zz * tmp;
        gdx += tmp * tmp;
        er += gx;
    }
    er = abs(er);

    hx = 0;
    hdx = 0;
    for(int i = dd - 1; i > hout; --i)
    {
        tmp = D[i] - cor;
        if(modif)
            D[i] = tmp;
        zz = z[i];
        tmp = zz / tmp;
        hx += zz * tmp;
        hdx += tmp * tmp;
        er += hx;
    }

    fx = p + gx + hx;
    fdx = gdx + hdx;

    // return results
    *pt_fx = fx;
    *pt_fdx = fdx;
    *pt_gx = gx;
    *pt_gdx = gdx;
    *pt_hx = hx;
    *pt_hdx = hdx;
    *pt_er = er;
}

//--------------------------------------------------------------------------------------//
/** SEQ_SOLVE solves secular equation at point k (i.e. computes kth eigenvalue
   that is within an internal interval). We use rational interpolation and fixed
   weights method between the 2 poles of the interval. (TODO: In the future, we
   could consider using 3 poles for those cases that may need it to reduce the
   number of required iterations to converge. The performance improvements are
   expected to be marginal, though) **/
template <typename S>
__device__ rocblas_int seq_solve(const rocblas_int dd,
                                 S* D,
                                 const S* z,
                                 const S p,
                                 rocblas_int k,
                                 S* ev,
                                 const S tol,
                                 const S ssfmin,
                                 const S ssfmax)
{
    bool converged = false;
    bool up, fixed;
    S lowb, uppb, aa, bb, cc, x;
    S nx, er, fx, fdx, gx, gdx, hx, hdx, oldfx;
    S tau, eta;
    S dk, dk1, ddk, ddk1;
    rocblas_int kk;
    rocblas_int k1 = k + 1;

    // initialize
    dk = D[k];
    dk1 = D[k1];
    x = (dk + dk1) / 2; // midpoint of interval
    tau = (dk1 - dk);
    S pinv = 1 / p;

    // find bounds and initial guess; translate origin
    seq_eval(2, k, dd, D, z, pinv, x, &cc, &fdx, &gx, &gdx, &hx, &hdx, &er, false);
    gdx = z[k] * z[k];
    hdx = z[k1] * z[k1];
    fx = cc + 2 * (hdx - gdx) / tau;
    if(fx > 0)
    {
        // if the secular eq at the midpoint is positive, the root is in between
        // D[k] and the midpoint take D[k] as the origin, i.e. x = D[k] + tau with
        // tau in (0, uppb)
        lowb = 0;
        uppb = tau / 2;
        up = true;
        kk = k; // origin remains the same
        aa = cc * tau + gdx + hdx;
        bb = gdx * tau;
        eta = sqrt(abs(aa * aa - 4 * bb * cc));
        if(aa > 0)
            tau = 2 * bb / (aa + eta);
        else
            tau = (aa - eta) / (2 * cc);
        x = dk + tau; // initial guess
    }
    else
    {
        // otherwise, the root is in between the midpoint and D[k+1]
        // take D[k+1] as the origin, i.e. x = D[k+1] + tau with tau in (lowb, 0)
        lowb = -tau / 2;
        uppb = 0;
        up = false;
        kk = k + 1; // translate the origin
        aa = cc * tau - gdx - hdx;
        bb = hdx * tau;
        eta = sqrt(abs(aa * aa + 4 * bb * cc));
        if(aa < 0)
            tau = 2 * bb / (aa - eta);
        else
            tau = -(aa + eta) / (2 * cc);
        x = dk1 + tau; // initial guess
    }

    // evaluate secular eq and get input values to calculate step correction
    seq_eval(0, kk, dd, D, z, pinv, (up ? dk : dk1), &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
    seq_eval(1, kk, dd, D, z, pinv, tau, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
    bb = z[kk];
    aa = bb / D[kk];
    fdx += aa * aa;
    bb *= aa;
    fx += bb;

    // calculate tolerance er for convergence test
    er += 8 * (hx - gx) + 2 * pinv + 3 * abs(bb) + abs(tau) * fdx;

    // if the value of secular eq is small enough, no point to continue;
    // converged!!!
    if(abs(fx) <= tol * er)
        converged = true;

    // otherwise...
    else
    {
        // update bounds
        lowb = (fx <= 0) ? std::max(lowb, tau) : lowb;
        uppb = (fx > 0) ? std::min(uppb, tau) : uppb;

        // calculate first step correction with fixed weight method
        ddk = D[k];
        ddk1 = D[k1];
        if(up)
            cc = fx - ddk1 * fdx - (dk - dk1) * z[k] * z[k] / ddk / ddk;
        else
            cc = fx - ddk * fdx - (dk1 - dk) * z[k1] * z[k1] / ddk1 / ddk1;
        aa = (ddk + ddk1) * fx - ddk * ddk1 * fdx;
        bb = ddk * ddk1 * fx;
        if(cc == 0)
        {
            if(aa == 0)
            {
                if(up)
                    aa = z[k] * z[k] + ddk1 * ddk1 * (gdx + hdx);
                else
                    aa = z[k1] * z[k1] + ddk * ddk * (gdx + hdx);
            }
            eta = bb / aa;
        }
        else
        {
            eta = sqrt(abs(aa * aa - 4 * bb * cc));
            if(aa <= 0)
                eta = (aa - eta) / (2 * cc);
            else
                eta = (2 * bb) / (aa + eta);
        }

        // verify that the correction eta will get x closer to the root
        // i.e. eta*fx should be negative. If not the case, take a Newton step
        // instead
        if(fx * eta >= 0)
            eta = -fx / fdx;

        // now verify that applying the correction won't get the process out of
        // bounds if that is the case, bisect the interval instead
        if(tau + eta > uppb || tau + eta < lowb)
        {
            if(fx < 0)
                eta = (uppb - tau) / 2;
            else
                eta = (lowb - tau) / 2;
        }

        // take the step
        tau += eta;
        x = (up ? dk : dk1) + tau;

        // evaluate secular eq and get input values to calculate step correction
        oldfx = fx;
        seq_eval(1, kk, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
        bb = z[kk];
        aa = bb / D[kk];
        fdx += aa * aa;
        bb *= aa;
        fx += bb;

        // calculate tolerance er for convergence test
        er += 8 * (hx - gx) + 2 * pinv + 3 * abs(bb) + abs(tau) * fdx;

        // from now on, further step corrections will be calculated either with
        // fixed weights method or with normal interpolation depending on the value
        // of boolean fixed
        cc = up ? -1 : 1;
        fixed = (cc * fx) > (abs(oldfx) / 10);

        // MAIN ITERATION LOOP
        // ==============================================
        for(int i = 1; i < MAXITERS; ++i)
        {
            // if the value of secular eq is small enough, no point to continue;
            // converged!!!
            if(abs(fx) <= tol * er)
            {
                converged = true;
                break;
            }

            // update bounds
            lowb = (fx <= 0) ? std::max(lowb, tau) : lowb;
            uppb = (fx > 0) ? std::min(uppb, tau) : uppb;

            // calculate next step correction with either fixed weight method or
            // simple interpolation
            ddk = D[k];
            ddk1 = D[k1];
            if(fixed)
            {
                if(up)
                    cc = fx - ddk1 * fdx - (dk - dk1) * z[k] * z[k] / ddk / ddk;
                else
                    cc = fx - ddk * fdx - (dk1 - dk) * z[k1] * z[k1] / ddk1 / ddk1;
            }
            else
            {
                if(up)
                    gdx += aa * aa;
                else
                    hdx += aa * aa;
                cc = fx - ddk * gdx - ddk1 * hdx;
            }
            aa = (ddk + ddk1) * fx - ddk * ddk1 * fdx;
            bb = ddk * ddk1 * fx;
            if(cc == 0)
            {
                if(aa == 0)
                {
                    if(fixed)
                    {
                        if(up)
                            aa = z[k] * z[k] + ddk1 * ddk1 * (gdx + hdx);
                        else
                            aa = z[k1] * z[k1] + ddk * ddk * (gdx + hdx);
                    }
                    else
                        aa = ddk * ddk * gdx + ddk1 * ddk1 * hdx;
                }
                eta = bb / aa;
            }
            else
            {
                eta = sqrt(abs(aa * aa - 4 * bb * cc));
                if(aa <= 0)
                    eta = (aa - eta) / (2 * cc);
                else
                    eta = (2 * bb) / (aa + eta);
            }

            // verify that the correction eta will get x closer to the root
            // i.e. eta*fx should be negative. If not the case, take a Newton step
            // instead
            if(fx * eta >= 0)
                eta = -fx / fdx;

            // now verify that applying the correction won't get the process out of
            // bounds if that is the case, bisect the interval instead
            if(tau + eta > uppb || tau + eta < lowb)
            {
                if(fx < 0)
                    eta = (uppb - tau) / 2;
                else
                    eta = (lowb - tau) / 2;
            }

            // take the step
            tau += eta;
            x = (up ? dk : dk1) + tau;

            // evaluate secular eq and get input values to calculate step correction
            oldfx = fx;
            seq_eval(1, kk, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
            bb = z[kk];
            aa = bb / D[kk];
            fdx += aa * aa;
            bb *= aa;
            fx += bb;

            // calculate tolerance er for convergence test
            er += 8 * (hx - gx) + 2 * pinv + 3 * abs(bb) + abs(tau) * fdx;

            // update boolean fixed if necessary
            if(fx * oldfx > 0 && abs(fx) > abs(oldfx) / 10)
                fixed = !fixed;
        }
    }

    *ev = x;
    return converged ? 0 : 1;
}

//--------------------------------------------------------------------------------------//
/** SEQ_SOLVE_EXT solves secular equation at point n (i.e. computes last
   eigenvalue). We use rational interpolation and fixed weights method between
   the (n-1)th and nth poles. (TODO: In the future, we could consider using 3
   poles for those cases that may need it to reduce the number of required
   iterations to converge. The performance improvements are expected to be
   marginal, though) **/
template <typename S>
__device__ rocblas_int seq_solve_ext(const rocblas_int dd,
                                     S* D,
                                     const S* z,
                                     const S p,
                                     S* ev,
                                     const S tol,
                                     const S ssfmin,
                                     const S ssfmax)
{
    bool converged = false;
    S lowb, uppb, aa, bb, cc, x;
    S er, fx, fdx, gx, gdx, hx, hdx;
    S tau, eta;
    S dk, dkm1, ddk, ddkm1;
    rocblas_int k = dd - 1;
    rocblas_int km1 = dd - 2;

    // initialize
    dk = D[k];
    dkm1 = D[km1];
    x = dk + p / 2;
    S pinv = 1 / p;

    // find bounds and initial guess
    seq_eval(2, km1, dd, D, z, pinv, x, &cc, &fdx, &gx, &gdx, &hx, &hdx, &er, false);
    gdx = z[km1] * z[km1];
    hdx = z[k] * z[k];
    fx = cc + gdx / (dkm1 - x) - 2 * hdx * pinv;
    if(fx > 0)
    {
        // if the secular eq at the midpoint is positive, the root is in between
        // D[k] and the midpoint take D[k] as the origin, i.e. x = D[k] + tau with
        // tau in (0, uppb)
        lowb = 0;
        uppb = p / 2;
        tau = dk - dkm1;
        aa = -cc * tau + gdx + hdx;
        bb = hdx * tau;
        eta = sqrt(aa * aa + 4 * bb * cc);
        if(aa < 0)
            tau = 2 * bb / (eta - aa);
        else
            tau = (aa + eta) / (2 * cc);
    }
    else
    {
        // otherwise, the root is in between the midpoint and D[k+1]
        // take D[k+1] as the origin, i.e. x = D[k+1] + tau with tau in (lowb, 0)
        lowb = p / 2;
        uppb = p;
        eta = gdx / (dk - dkm1 + p) + hdx / p;
        if(cc <= eta)
            tau = p;
        else
        {
            tau = dk - dkm1;
            aa = -cc * tau + gdx + hdx;
            bb = hdx * tau;
            eta = sqrt(aa * aa + 4 * bb * cc);
            if(aa < 0)
                tau = 2 * bb / (eta - aa);
            else
                tau = (aa + eta) / (2 * cc);
        }
    }
    x = dk + tau; // initial guess

    // evaluate secular eq and get input values to calculate step correction
    seq_eval(0, km1, dd, D, z, pinv, dk, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
    seq_eval(0, km1, dd, D, z, pinv, tau, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);

    // calculate tolerance er for convergence test
    er += abs(tau) * (hdx + gdx) - 8 * (hx + gx) - hx + pinv;

    // if the value of secular eq is small enough, no point to continue;
    // converged!!!
    if(abs(fx) <= tol * er)
        converged = true;

    // otherwise...
    else
    {
        // update bounds
        lowb = (fx <= 0) ? std::max(lowb, tau) : lowb;
        uppb = (fx > 0) ? std::min(uppb, tau) : uppb;

        // calculate first step correction with fixed weight method
        ddk = D[k];
        ddkm1 = D[km1];
        cc = abs(fx - ddkm1 * gdx - ddk * hdx);
        aa = (ddk + ddkm1) * fx - ddk * ddkm1 * (gdx + hdx);
        bb = ddk * ddkm1 * fx;
        if(cc == 0)
        {
            eta = uppb - tau;
        }
        else
        {
            eta = sqrt(abs(aa * aa - 4 * bb * cc));
            if(aa >= 0)
                eta = (aa + eta) / (2 * cc);
            else
                eta = (2 * bb) / (aa - eta);
        }

        // verify that the correction eta will get x closer to the root
        // i.e. eta*fx should be negative. If not the case, take a Newton step
        // instead
        if(fx * eta > 0)
            eta = -fx / (gdx + hdx);

        // now verify that applying the correction won't get the process out of
        // bounds if that is the case, bisect the interval instead
        if(tau + eta > uppb || tau + eta < lowb)
        {
            if(fx < 0)
                eta = (uppb - tau) / 2;
            else
                eta = (lowb - tau) / 2;
        }

        // take the step
        tau += eta;
        x = dk + tau;

        // evaluate secular eq and get input values to calculate step correction
        seq_eval(0, km1, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);

        // calculate tolerance er for convergence test
        er += abs(tau) * (hdx + gdx) - 8 * (hx + gx) - hx + pinv;

        // MAIN ITERATION LOOP
        // ==============================================
        for(int i = 1; i < MAXITERS; ++i)
        {
            // if the value of secular eq is small enough, no point to continue;
            // converged!!!
            if(abs(fx) <= tol * er)
            {
                converged = true;
                break;
            }

            // update bounds
            lowb = (fx <= 0) ? std::max(lowb, tau) : lowb;
            uppb = (fx > 0) ? std::min(uppb, tau) : uppb;

            // calculate step correction
            ddk = D[k];
            ddkm1 = D[km1];
            cc = fx - ddkm1 * gdx - ddk * hdx;
            aa = (ddk + ddkm1) * fx - ddk * ddkm1 * (gdx + hdx);
            bb = ddk * ddkm1 * fx;
            eta = sqrt(abs(aa * aa - 4 * bb * cc));
            if(aa >= 0)
                eta = (aa + eta) / (2 * cc);
            else
                eta = (2 * bb) / (aa - eta);

            // verify that the correction eta will get x closer to the root
            // i.e. eta*fx should be negative. If not the case, take a Newton step
            // instead
            if(fx * eta > 0)
                eta = -fx / (gdx + hdx);

            // now verify that applying the correction won't get the process out of
            // bounds if that is the case, bisect the interval instead
            if(tau + eta > uppb || tau + eta < lowb)
            {
                if(fx < 0)
                    eta = (uppb - tau) / 2;
                else
                    eta = (lowb - tau) / 2;
            }

            // take the step
            tau += eta;
            x = dk + tau;

            // evaluate secular eq and get input values to calculate step correction
            seq_eval(0, km1, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);

            // calculate tolerance er for convergence test
            er += abs(tau) * (hdx + gdx) - 8 * (hx + gx) - hx + pinv;
        }
    }

    *ev = x;
    return converged ? 0 : 1;
}

/** This local gemm adapts rocblas_gemm to multiply complex*real, and
    overwrite result: A = A*B **/
template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename S,
          typename U,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
void local_gemm(rocblas_handle handle,
                const rocblas_int n,
                U A,
                const rocblas_int shiftA,
                const rocblas_int lda,
                const rocblas_stride strideA,
                S* B,
                S* temp,
                S* work,
                const rocblas_int shiftV,
                const rocblas_int ldv,
                const rocblas_stride strideV,
                const rocblas_int batch_count,
                S** workArr)
{
    S one = 1.0;
    S zero = 0.0;

    // Execute A*B -> temp -> A
    // temp = A*B
    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, A, shiftA,
                   lda, strideA, B, shiftV, ldv, strideV, &zero, temp, shiftV, ldv, strideV,
                   batch_count, workArr);

    // A = temp
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocks = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks, blocks, batch_count), dim3(BS2, BS2), 0,
                            stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, temp);
}

template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename S,
          typename U,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
void local_gemm(rocblas_handle handle,
                const rocblas_int n,
                U A,
                const rocblas_int shiftA,
                const rocblas_int lda,
                const rocblas_stride strideA,
                S* B,
                S* temp,
                S* work,
                const rocblas_int shiftV,
                const rocblas_int ldv,
                const rocblas_stride strideV,
                const rocblas_int batch_count,
                S** workArr)
{
    S one = 1.0;
    S zero = 0.0;

    // Execute A -> work; work*B -> temp -> A

    // work = real(A)
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocks = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, true>), dim3(blocks, blocks, batch_count), dim3(BS2, BS2),
                            0, stream, copymat_to_buffer, n, n, A, shiftA, lda, strideA, work);

    // temp = work*B
    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, work,
                   shiftV, ldv, strideV, B, shiftV, ldv, strideV, &zero, temp, shiftV, ldv, strideV,
                   batch_count, workArr);

    // real(A) = temp
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, true>), dim3(blocks, blocks, batch_count), dim3(BS2, BS2),
                            0, stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, temp);

    // work = imag(A)
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, false>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_to_buffer, n, n, A, shiftA, lda,
                            strideA, work);

    // temp = work*B
    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, work,
                   shiftV, ldv, strideV, B, shiftV, ldv, strideV, &zero, temp, shiftV, ldv, strideV,
                   batch_count, workArr);

    // imag(A) = temp
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, false>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_from_buffer, n, n, A, shiftA, lda,
                            strideA, temp);
}

ROCSOLVER_END_NAMESPACE
