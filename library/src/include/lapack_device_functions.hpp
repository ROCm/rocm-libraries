/* **************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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
__device__ void lartg(T& f, T& g, T& c, T& s, T& r)
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
__device__ void laev2(T& a, T& b, T& c, T& rt1, T& rt2, T& cs1, T& sn1)
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
    MAX_THDS should be 64, 128, 256, 512, or 1024, and sval should
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
    sval[tid] = val;
    __syncthreads();

    if(n <= 1)
        return;

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

ROCSOLVER_END_NAMESPACE
