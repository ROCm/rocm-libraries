/* **************************************************************************
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

 #include "rocauxiliary_sb2st_hb2st.hpp"

 ROCSOLVER_BEGIN_NAMESPACE

 template <typename T, typename S, typename U>
 rocblas_status rocsolver_sb2st_hb2st_impl(rocblas_handle handle,
                                           const rocblas_int n,
                                           const rocblas_int nb,
                                           U A,
                                           const rocblas_int lda,
                                           S* D,
                                           S* E)
 {
     ROCSOLVER_ENTER_TOP("sb2st_hb2st", "-n", n, "-nb", nb, "--lda", lda);
 
     if(!handle)
         return rocblas_status_invalid_handle;
 
     // argument checking
     rocblas_status st = rocsolver_sb2st_hb2st_argCheck(handle, n, nb, lda, A, D, E);
     if(st != rocblas_status_continue)
         return st;
 
     // working with unshifted arrays
     rocblas_int shiftA = 0;
     rocblas_int shiftAB = 0;
 
     // normal (non-batched non-strided) execution
     rocblas_stride strideA = 0;
     rocblas_stride strideD = 0;
     rocblas_stride strideE = 0;
     rocblas_int batch_count = 1;
 
     // memory workspace sizes:
     // size for constants in rocblas calls
     size_t size_scalars;
     rocsolver_sb2st_hb2st_getMemorySize<false, T, S>(n, nb, batch_count, &size_scalars);

     if(rocblas_is_device_memory_size_query(handle))
         return rocblas_set_optimal_device_memory_size(handle, size_scalars);
 
     // memory workspace allocation
     void *scalars;
     rocblas_device_malloc mem(handle, size_scalars);
 
     if(!mem)
         return rocblas_status_memory_error;
 
     scalars = mem[0];
     if(size_scalars > 0)
         init_scalars(handle, (T*)scalars);
 
     // execution
     return rocsolver_sb2st_hb2st_template<false, false, T>(handle, n, nb, A, shiftA, lda, strideA,
                                                            D, strideD, E, strideE, batch_count, (T*)scalars);
 }
 
 ROCSOLVER_END_NAMESPACE
 
 /*
  * ===========================================================================
  *    C wrapper
  * ===========================================================================
  */
 
 extern "C" {

 ROCSOLVER_EXPORT rocblas_status rocsolver_ssb2st(rocblas_handle handle,
                                                  const rocblas_int n,
                                                  const rocblas_int nb,
                                                  float* A,
                                                  const rocblas_int lda,
                                                  float* D,
                                                  float* E)
 {
     return rocsolver::rocsolver_sb2st_hb2st_impl<float>(handle, n, nb, A, lda, D, E);
 }

 ROCSOLVER_EXPORT rocblas_status rocsolver_dsb2st(rocblas_handle handle,
                                                  const rocblas_int n,
                                                  const rocblas_int nb,
                                                  double* A,
                                                  const rocblas_int lda,
                                                  double* D,
                                                  double* E)
 {
     return rocsolver::rocsolver_sb2st_hb2st_impl<double>(handle, n, nb, A, lda, D, E);
 }

 ROCSOLVER_EXPORT rocblas_status rocsolver_chb2st(rocblas_handle handle,
                                                  const rocblas_int n,
                                                  const rocblas_int nb,
                                                  rocblas_float_complex* A,
                                                  const rocblas_int lda,
                                                  float* D,
                                                  float* E)
 {
     return rocsolver::rocsolver_sb2st_hb2st_impl<rocblas_float_complex>(handle, n, nb, A, lda, D, E);
 }

 ROCSOLVER_EXPORT rocblas_status rocsolver_zhb2st(rocblas_handle handle,
                                                  const rocblas_int n,
                                                  const rocblas_int nb,
                                                  rocblas_double_complex* A,
                                                  const rocblas_int lda,
                                                  double* D,
                                                  double* E)
 {
     return rocsolver::rocsolver_sb2st_hb2st_impl<rocblas_double_complex>(handle, n, nb, A, lda, D,
                                                                          E);
 }

 } // extern C
 