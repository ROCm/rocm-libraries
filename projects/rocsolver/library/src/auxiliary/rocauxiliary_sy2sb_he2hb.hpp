/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

 #include "lapack/roclapack_gelqf.hpp"
 #include "lapack/roclapack_geqrf.hpp"
 #include "rocblas.hpp"
 #include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
ROCSOLVER_KERNEL void sy2sb_updateAV_kernel(const rocblas_int ii,
                               const rocblas_int m,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               T* V,
                               const rocblas_int shiftV,
                               const rocblas_int ldv,
                               const rocblas_stride strideV)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* Vp = load_ptr_batch<T>(V, b, shiftV, strideV);

    if(i < m && j < n)
    {
/*        if(upper)
        {
            const auto ai = ii + i;
            const auto aj = ii + i + j;

            const auto abi = k - j;
            const auto abj = ii + i + j;

            // copy to AB
            ABp[abi + abj * ldab] = Ap[ai + aj * lda];

            // set V
            if(j == k)
            {
                Ap[ai + aj * lda] = 1;
            }
            else if(i + j >= k)
            {
                Ap[ai + aj * lda] = 0;
            }
        }*/
    }
/*    if(i < k+1 && j < pk)
    {
        if(lower)
        {
            const auto ai = ii + i + j;
            const auto aj = ii + j;

            const auto abi = i;
            const auto abj = ii + j;

            // copy to AB
            ABp[abi + abj * ldab] = Ap[ai + aj * lda];

            // set V
            if(i == k)
            {
                Ap[ai + aj * lda] = 1;
            }
            else if(i + j >= k)
            {
                Ap[ai + aj * lda] = 0;
            }
        }
    }*/
}

template <typename T, typename U>
ROCSOLVER_KERNEL void copyTBand(const rocblas_fill uplo,
                               const rocblas_int k,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               T* AB,
                               const rocblas_int shiftAB,
                               const rocblas_int ldab,
                               const rocblas_stride strideAB)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const bool upper = (uplo == rocblas_fill_upper);
    const bool lower = (uplo == rocblas_fill_lower);

    if(i < k && j < k)
    {
        T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
        T* ABp = load_ptr_batch<T>(AB, b, shiftAB, strideAB);

        if(upper && i <= j)
        {
            const auto ai = i;
            const auto aj = j;

            const auto abi = k + i - j;
            const auto abj = j;

            // copy to AB
            ABp[abi + abj * ldab] = Ap[ai + aj * lda];
        }
        else if(lower && j <= i)
        {
            const auto ai = i;
            const auto aj = j;

            const auto abi = i - j;
            const auto abj = j;

            // copy to AB
            ABp[abi + abj * ldab] = Ap[ai + aj * lda];
        }
    }
}

template <bool BATCHED, typename T>
void rocsolver_sy2sb_he2hb_getMemorySize(const rocblas_int n,
                                         const rocblas_int nb,
                                         const rocblas_int k,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_workT,
                                         size_t* size_workS1,
                                         size_t* size_workS2,
                                         size_t* size_workW,
                                         size_t* size_workArr)
{
    *size_scalars = 0;
    *size_workT = 0;
    *size_workS1 = 0;
    *size_workS2 = 0;
    *size_workW = 0;
    *size_workArr = 0;

    // if quick return no workspace needed
    if(n == 0 || batch_count == 0 || n <= k + 1)
        return;

    *size_workT = sizeof(T) * n * n * batch_count; // ==================== k * k *
    *size_workS1 = sizeof(T) * k * k * batch_count;
    *size_workS2 = sizeof(T) * (n - k) * k * batch_count;
    *size_workW = sizeof(T) * (n - k) * k * batch_count;
    *size_workArr = BATCHED ? sizeof(T*) * 2 * batch_count : false;

    size_t w, wa, s1, s2;
//    if(uplo == rocblas_fill_upper)
//        rocsolver_gelqf_getMemorySize<BATCHED, T>(k, n - k, batch_count, size_scalars, &s1,
//            &s2, &w, &wa);
//    else
        rocsolver_geqrf_getMemorySize<BATCHED, T>(n - k, k, batch_count, size_scalars, &s1,
            &s2, &w, &wa);
    *size_workS1 = std::max(*size_workS1, s1);
    *size_workS2 = std::max(*size_workS2, s2);

    *size_workW = std::max(*size_workW, w);
    *size_workArr = std::max(*size_workArr, wa);

    rocsolver_larft_getMemorySize<BATCHED, T>(n-k, k, batch_count, size_scalars, &w, &wa);

    *size_workW = std::max(*size_workW, w);
    *size_workArr = std::max(*size_workArr, wa);
}

template <typename T, typename S>
rocblas_status rocsolver_sy2sb_he2hb_argCheck(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nb,
                                              const rocblas_int k,
                                              T A,
                                              const rocblas_int lda,
                                              S V,
                                              S W,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    //if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
   //     return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || (n > 0 && nb < 1) || k < nb || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !V) || (n && !W))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_sy2sb_he2hb_template(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nb,
                                              const rocblas_int k,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* V,
                                              const rocblas_int ldv,
                                              const rocblas_stride strideV,
                                              T* W,
                                              const rocblas_int ldw,
                                              const rocblas_stride strideW,
                                              const rocblas_int batch_count,
                                              T* scalars,
//                                              T* workT,
                                              T* Acpy,
                                              T* workS1,
                                              T* workS2,
                                              T* workW,
                                              T** workArr)
{
    ROCSOLVER_ENTER("sy2sb_he2hb", "n:", n, "nb", nb, "k:", k, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    using S = decltype(std::real(T{}));

    // quick return
    if(n == 0 || nb == 0 || k == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    T one = 1;
    T zero = 0;
    T neghalf = -0.5;
    T negone = -1;
    S rone = 1;

/*    const bool upper = (uplo == rocblas_fill_upper);

    if(n <= k + 1)
    {
        const rocblas_int cpy_blks = (n) / 32 + 1;
        if(upper)
        {
            // Copy A to AB
            ROCSOLVER_LAUNCH_KERNEL((copyTBand), dim3(cpy_blks, cpy_blks, batch_count),
                                    dim3(32, 32), 0, stream, uplo, n, A, shiftA, lda, strideA,
                                    AB, shiftAB + idx2D(k - n, 0, ldab), ldab, strideAB);
        }
        else
        {
            // Copy A to AB
            ROCSOLVER_LAUNCH_KERNEL((copyTBand), dim3(cpy_blks, cpy_blks, batch_count),
                                    dim3(32, 32), 0, stream, uplo, n, A, shiftA, lda, strideA,
                                    AB, shiftAB, ldab, strideAB);
        }

        rocblas_set_pointer_mode(handle, old_mode);
        return rocblas_status_success;
    }*/

/*    rocblas_int ldt = k;
    rocblas_int lds1 = k;
    rocblas_int lds2 = upper ? k : (n - k);
    rocblas_int ldw = upper ? k : (n - k);

    rocblas_stride strideT = k * k;
    rocblas_stride strideS1 = k * k;
    rocblas_stride strideS2 = (n - k) * k;
    rocblas_stride strideW = (n - k) * k;*/
    
    // set T to zero
    // const rocblas_int reset_nblks = (k - 1) / 32 + 1;
    // ROCSOLVER_LAUNCH_KERNEL((set_zero<T>), dim3(reset_nblks, reset_nblks, batch_count),
    //                         dim3(32, 32), 0, stream, ldt, k, workT, 0, ldt, strideT,
    //                         rocblas_fill_full);
    
/*    if(upper)
    {
        for(rocblas_int i = 0; i < n - k; i += k)
        {
            rocblas_int pn = n - i - k;
            rocblas_int pk = std::min(pn, k);

            // Compute LQ
            rocsolver_gelqf_template<BATCHED, STRIDED>(handle, k, pn, A, shiftA + idx2D(i, i+k, lda), lda, strideA,
                tau + i, strideP, batch_count, scalars, workS1, workS2, workW, workArr);

            // Copy A to AB
            const rocblas_int cpy_xblks = (k) / 32 + 1;
            const rocblas_int cpy_yblks = (k + 1) / 32 + 1;
            ROCSOLVER_LAUNCH_KERNEL((copyBand_setV), dim3(cpy_xblks, cpy_yblks, batch_count),
                                    dim3(32, 32), 0, stream, uplo, k, pk, i, A, shiftA, lda, strideA,
                                    AB, shiftAB, ldab, strideAB);

            // Form matrix T
            rocsolver_larft_template<T>(handle, rocblas_forward_direction, rocblas_row_wise, pn, pk, A, shiftA + idx2D(i, i+k, lda),
                lda, strideA, tau + i, strideP, workT, ldt, strideT, batch_count, scalars, workW, workArr);

            // Compute W
            rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, pk, pn, pk, &one, workT,
                0, ldt, strideT, A, shiftA + idx2D(i, i+k, lda), lda, strideA, &zero, workS2, 0, lds2,
                strideS2, batch_count, workArr);

            rocblasCall_symm_hemm(handle, rocblas_side_right, uplo, pk, pn, &one, A, shiftA + idx2D(i+k, i+k, lda),
                lda, strideA, workS2, 0, lds2, strideS2, &zero, workW, 0, ldw, strideW, batch_count);
            // rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, pk, pn, pn, &one, workS2,
            //     0, lds2, strideS2, A, shiftA + idx2D(i+k, i+k, lda), lda, strideA, &zero, workW, 0, ldw,
            //     strideW, batch_count, workArr);

            
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, pk, pk, pn, &one, workW,
                0, ldw, strideW, workS2, 0, lds2, strideS2, &zero, workS1, 0, lds1,
                strideS1, batch_count, workArr);
            
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, pk, pn, pk, &neghalf, workS1,
                0, lds1, strideS1, A, shiftA + idx2D(i, i+k, lda), lda, strideA, &one, workW, 0, ldw,
                strideW, batch_count, workArr);

            // Update submatrix A
            rocblasCall_syr2k_her2k<BATCHED>(handle, uplo, rocblas_operation_conjugate_transpose, pn, pk, &negone, A, shiftA + idx2D(i, i+k, lda),
                lda, strideA, workW, 0, ldw, strideW, &rone, A, shiftA + idx2D(i+k, i+k, lda), lda, strideA, batch_count, workArr);
        }
        // Copy A to AB
        const rocblas_int cpy_blks = (k) / 32 + 1;
        ROCSOLVER_LAUNCH_KERNEL((copyTBand), dim3(cpy_blks, cpy_blks, batch_count),
                                dim3(32, 32), 0, stream, uplo, k, A, shiftA + idx2D(n-k, n-k, lda), lda, strideA,
                                AB, shiftAB + idx2D(0, n-k, ldab), ldab, strideAB);
    }
    else
    {*/

//print_device_matrix(std::cout,"A init",n,n,A,lda);
//print_device_matrix(std::cout,"AB init",k+1,n,AB,ldab);

    rocblas_int nk = n - nb;
    rocblas_int nk_blks = (nk - 1) / 32 + 1;
    rocblas_int nb_blks = (nb - 1) / 32 + 1;
    rocblas_int ldacpy = n;
    rocblas_stride strideAcpy = n * n;
    rocblas_stride strideP = strideAcpy;
    rocblas_int j, jj;
    T* tau = Acpy + n - 1;


print_device_matrix(std::cout,"A in",n,n,A,lda);

        for(rocblas_int i = 0; i < nk; i += k)
        {

printf("for i = %d\n",i);

            rocblas_int qn = n - i - nb;
            rocblas_int endb = std::min(i+k, nk); 

            // keep copy of trailing matrix in Acpy to update V and W
            rocblas_int cpy_blks = (qn - 1) / 32 + 1;
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(cpy_blks, cpy_blks, batch_count),
                                    dim3(32, 32), 0, stream, qn, qn, 
                                    A, shiftA + idx2D(i + nb, i + nb, lda), lda, strideA,
                                    Acpy, 0, ldacpy, strideAcpy);

print_device_matrix(std::cout,"Acpy",qn,qn,Acpy,ldacpy);
 
            // reduce first panel in block
            rocsolver_geqrf_template<BATCHED, STRIDED>(handle, qn, nb, A, shiftA + idx2D(i + nb, i, lda), lda, strideA,
                tau, strideP, batch_count, scalars, workS1, workS2, workW, workArr);

print_device_matrix(std::cout,"A",n,n,A,lda);

            // update A and V
            ROCSOLVER_LAUNCH_KERNEL((sy2sb_updateAV_kernel), dim3(nk_blks, nb_blks, batch_count),
                                    dim3(32, 32), 0, stream, i, nk, nb, A, shiftA + idx2D(i + nb, i, lda), lda, strideA,
                                    V, idx2D(0, i, ldv), ldv, strideV);
print_device_matrix(std::cout,"V",nk,nk,V,ldv);

/*            // Form corresponding matrix T
            rocsolver_larft_template<T>(handle, rocblas_forward_direction, rocblas_column_wise, pn, pk, A, shiftA + idx2D(i+k, i, lda),
                lda, strideA, tau + i, strideP, workT, ldt, strideT, batch_count, scalars, workW, workArr);

            // Update W
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, pn, pk, pk, &one, A,
                shiftA + idx2D(i+k, i, lda), lda, strideA, workT, 0, ldt, strideT, &zero, workS2, 0, lds2,
                strideS2, batch_count, workArr);

//            rocblasCall_symm_hemm(handle, rocblas_side_left, uplo, pn, pk, &one, A, shiftA + idx2D(i+k, i+k, lda),
//                lda, strideA, workS2, 0, lds2, strideS2, &zero, workW, 0, ldw, strideW, batch_count);
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, pn, pk, pn, &one, A, shiftA + idx2D(i+k, i+k, lda),
                lda, strideA, workS2, 0, lds2, strideS2, &zero, workW, 0, ldw, strideW, batch_count, workArr);

            rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, pk, pk, pn, &one, workS2,
                0, lds2, strideS2, workW, 0, ldw, strideW, &zero, workS1, 0, lds1,
                strideS1, batch_count, workArr);
            
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, pn, pk, pk, &neghalf, A,
                shiftA + idx2D(i+k, i, lda), lda, strideA, workS1, 0, lds1, strideS1, &one, workW, 0, ldw,
                strideW, batch_count, workArr);

            // Update submatrix A
            rocblasCall_syr2k_her2k<BATCHED>(handle, uplo, rocblas_operation_none, pn, pk, &negone, A, shiftA + idx2D(i+k, i, lda),
                lda, strideA, workW, 0, ldw, strideW, &rone, A, shiftA + idx2D(i+k, i+k, lda), lda, strideA, batch_count, workArr);*/
            
            // reduce all other panels in block
            j = i + nb;
            jj = 0;
            while(j < endb)
            {
printf(">>>>> for j = %d\n",j);
                qn = n - j - nb;
//                rocblas_int cols = std::min(nb, nk - j);
                // update current panel

                // reduce current panel
                rocsolver_geqrf_template<BATCHED, STRIDED>(handle, qn, nb, A, shiftA + idx2D(j + nb, j, lda), lda, strideA,
                tau, strideP, batch_count, scalars, workS1, workS2, workW, workArr);

print_device_matrix(std::cout,"A",n,n,A,lda);

                // update A and V
                ROCSOLVER_LAUNCH_KERNEL((sy2sb_updateAV_kernel), dim3(nk_blks, nb_blks, batch_count),
                                        dim3(32, 32), 0, stream, j, nk, nb, A, shiftA + idx2D(j + nb, j, lda), lda, strideA,
                                        V, idx2D(0, j, ldv), ldv, strideV);

print_device_matrix(std::cout,"V",nk,nk,V,ldv);

                j += nb;
            }
        }
        // Copy A to AB
//        const rocblas_int cpy_blks = (k) / 32 + 1;
//        ROCSOLVER_LAUNCH_KERNEL((copyTBand), dim3(cpy_blks, cpy_blks, batch_count),
//                                dim3(32, 32), 0, stream, uplo, k, A, shiftA + idx2D(n-k, n-k, lda), lda, strideA,
//                                AB, shiftAB + idx2D(0, n-k, ldab), ldab, strideAB);

//print_device_matrix(std::cout,"A fin",n,n,A,lda);
//print_device_matrix(std::cout,"AB fin",k+1,n,AB,ldab);

//    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
