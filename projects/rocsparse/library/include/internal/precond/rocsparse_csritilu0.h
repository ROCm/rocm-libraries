/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_CSRITILU0_H
#define ROCSPARSE_CSRITILU0_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csritilu0_buffer_size computes the size in bytes of the buffer that has to be allocated by the user.
*  This buffer is then used in \ref rocsparse_csritilu0_preprocess, \ref rocsparse_scsritilu0_compute "rocsparse_Xcsritilu0_compute()",
*  \ref rocsparse_scsritilu0_compute_ex "rocsparse_Xcsritilu0_compute_ex()", and \ref rocsparse_scsritilu0_history "rocsparse_Xcsritilu0_history()".
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  rocsparse_csrsort().
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  alg         algorithm to use, \ref rocsparse_itilu0_alg
*  @param[in]
*  option      combination of enumeration values from \ref rocsparse_itilu0_option.
*  @param[in]
*  nmaxiter     maximum number of iterations.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start
*              of every row of the sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*  @param[in]
*  datatype    Type of numerical values, \ref rocsparse_datatype.
*  @param[out]
*  buffer_size size of the temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_value \p alg, \p base or datatype is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr
*              or \p csr_col_ind pointer is invalid.
*  \retval     rocsparse_status_zero_pivot if nnz is zero.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csritilu0_buffer_size(rocsparse_handle     handle,
                                                 rocsparse_itilu0_alg alg,
                                                 rocsparse_int        option,
                                                 rocsparse_int        nmaxiter,
                                                 rocsparse_int        m,
                                                 rocsparse_int        nnz,
                                                 const rocsparse_int* csr_row_ptr,
                                                 const rocsparse_int* csr_col_ind,
                                                 rocsparse_index_base idx_base,
                                                 rocsparse_datatype   datatype,
                                                 size_t*              buffer_size);

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csritilu0_preprocess computes the information required to run \ref rocsparse_scsritilu0_compute "rocsparse_Xcsritilu0_compute()",
*  and \ref rocsparse_scsritilu0_compute_ex "rocsparse_Xcsritilu0_compute_ex()" and stores it in the buffer.
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  rocsparse_csrsort().
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  alg         algorithm to use, \ref rocsparse_itilu0_alg
*  @param[in]
*  option      combination of enumeration values from \ref rocsparse_itilu0_option.
*  @param[in]
*  nmaxiter    maximum number of iterations.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start
*              of every row of the sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*  @param[in]
*  datatype    Type of numerical values, \ref rocsparse_datatype.
*  @param[in]
*  buffer_size size of the storage buffer allocated by the user.
*  @param[in]
*  buffer      storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_value \p alg, \p base or datatype is invalid.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr
*              or \p csr_col_ind pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_zero_pivot if missing diagonal element is detected.
*
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csritilu0_preprocess(rocsparse_handle     handle,
                                                rocsparse_itilu0_alg alg,
                                                rocsparse_int        option,
                                                rocsparse_int        nmaxiter,
                                                rocsparse_int        m,
                                                rocsparse_int        nnz,
                                                const rocsparse_int* csr_row_ptr,
                                                const rocsparse_int* csr_col_ind,
                                                rocsparse_index_base idx_base,
                                                rocsparse_datatype   datatype,
                                                size_t               buffer_size,
                                                void*                buffer);

/*! \ingroup precond_module
*  \brief Iterative Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format.
*
*  \details
*  \p rocsparse_csritilu0_compute computes iteratively the incomplete LU factorization with 0 fill-ins and no
*  pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
*  \f[
*    A \approx (L + Id)(D + U)
*  \f]
*
*  We use the following notation for the equations below: diag is the diagonal part, lower is the strict lower triangular part and upper is the strict upper triangular part of a given matrix.
*  Starting with \f$L_{0} = lower(\f$ \p ilu0 \f$)\f$, \f$U_{0} = upper(\f$ \p ilu0 \f$)\f$, the method iterates with
*  \f[
*  \begin{eqnarray}
*     R_k &=& A - L_{k} U_{k},\\
*     D_{k+1} &=& diag(R_k),\\
*     L_{k+1} &=& lower(R_k) D_{k+1}^{-1},\\
*     U_{k+1} &=& upper(R_k),
*  \end{eqnarray}
*  \f]
*  if \f$ 0 \le k \lt \f$ \p nmaxiter and if
*  \f[
*     \Vert R_k \Vert_{\infty} \gt \epsilon \Vert A \Vert_{\infty},
*  \f]
*  with \f$\epsilon\f$ = \p tol. Note that the calculation of \f$R_k\f$ is performed with no fill-in.
*
*  Computing the above iterative incomplete LU factorization requires three steps to complete. First,
*  the user determines the size of the required temporary storage buffer by calling \ref rocsparse_csritilu0_buffer_size.
*  Once this buffer size has been determined, the user allocates the buffer and passes it to
*  \ref rocsparse_csritilu0_preprocess. This will perform analysis on the sparsity pattern of the matrix. Finally,
*  the user calls \p rocsparse_scsritilu0_compute, \p rocsparse_dcsritilu0_compute, \p rocsparse_ccsritilu0_compute,
*  or \p rocsparse_zcsritilu0_compute to perform the actual factorization. The calculation
*  of the buffer size and the analysis of the sparse matrix only need to be performed once for a given sparsity pattern
*  while the factorization can be repeatedly applied to multiple matrices having the same sparsity pattern. Once all calls
*  to \ref rocsparse_scsritilu0_compute "rocsparse_Xcsritilu0_compute()" are complete, the temporary buffer can be deallocated.
*
*  \p rocsparse_csritilu0 has a number of options that can be useful for examining the convergence history, easily printing debug
*  information, and for using COO internal format.
*  <table>
*  <caption id="csritilu0 options">Options</caption>
*  <tr><th>Option                                              <th>Notes
*  <tr><td>rocsparse_itilu0_option_verbose</td>                <td>Print to stdout convergence data as the routine runs. Useful for debugging.</td>
*  <tr><td>rocsparse_itilu0_option_stopping_criteria</td>      <td>Enable stopping criteria.</td>
*  <tr><td>rocsparse_itilu0_option_compute_nrm_correction</td> <td>Compute and store normalized correction. The stored data can then be queried later with \ref rocsparse_scsritilu0_history "rocsparse_Xcsritilu0_history".</td>
*  <tr><td>rocsparse_itilu0_option_compute_nrm_residual</td>   <td>Compute and store the normalized residual of the between the approximate solution and the exact solution per iteration. The stored data can then be queried later with \ref rocsparse_scsritilu0_history "rocsparse_Xcsritilu0_history".</td>
*  <tr><td>rocsparse_itilu0_option_convergence_history</td>    <td>Enable collecting convergence history data with \ref rocsparse_scsritilu0_history "rocsparse_Xcsritilu0_history".</td>
*  <tr><td>rocsparse_itilu0_option_coo_format</td>             <td>Use COO format internally.</td>
*  </table>
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  rocsparse_csrsort().
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  alg         algorithm to use, \ref rocsparse_itilu0_alg
*  @param[in]
*  option      combination of enumeration values from \ref rocsparse_itilu0_option.
*  @param[inout]
*  nmaxiter     maximum number of iterations on input and number of iterations on output. If the output number of iterations is strictly less than the input maximum number of iterations, then the algorithm converged.
*  @param[in]
*  tol tolerance to use for stopping criteria.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start
*              of every row of the sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[inout]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[out]
*  ilu0        incomplete factorization.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*  @param[in]
*  buffer_size size of the storage buffer allocated by the user.
*  @param[in]
*  buffer      storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_value \p alg or \p base is invalid.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr
*              or \p csr_col_ind pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*  \par Example
*  \include example_rocsparse_csritilu0.cpp
*/
/**@{*/
__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_scsritilu0_compute_ex instead.")))
ROCSPARSE_EXPORT rocsparse_status
    rocsparse_scsritilu0_compute(rocsparse_handle     handle,
                                 rocsparse_itilu0_alg alg,
                                 rocsparse_int        option,
                                 rocsparse_int*       nmaxiter,
                                 float                tol,
                                 rocsparse_int        m,
                                 rocsparse_int        nnz,
                                 const rocsparse_int* csr_row_ptr,
                                 const rocsparse_int* csr_col_ind,
                                 const float*         csr_val,
                                 float*               ilu0,
                                 rocsparse_index_base idx_base,
                                 size_t               buffer_size,
                                 void*                buffer);

__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_dcsritilu0_compute_ex instead.")))
ROCSPARSE_EXPORT rocsparse_status
    rocsparse_dcsritilu0_compute(rocsparse_handle     handle,
                                 rocsparse_itilu0_alg alg,
                                 rocsparse_int        option,
                                 rocsparse_int*       nmaxiter,
                                 double               tol,
                                 rocsparse_int        m,
                                 rocsparse_int        nnz,
                                 const rocsparse_int* csr_row_ptr,
                                 const rocsparse_int* csr_col_ind,
                                 const double*        csr_val,
                                 double*              ilu0,
                                 rocsparse_index_base idx_base,
                                 size_t               buffer_size,
                                 void*                buffer);

__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_ccsritilu0_compute_ex instead.")))
ROCSPARSE_EXPORT rocsparse_status
    rocsparse_ccsritilu0_compute(rocsparse_handle               handle,
                                 rocsparse_itilu0_alg           alg,
                                 rocsparse_int                  option,
                                 rocsparse_int*                 nmaxiter,
                                 float                          tol,
                                 rocsparse_int                  m,
                                 rocsparse_int                  nnz,
                                 const rocsparse_int*           csr_row_ptr,
                                 const rocsparse_int*           csr_col_ind,
                                 const rocsparse_float_complex* csr_val,
                                 rocsparse_float_complex*       ilu0,
                                 rocsparse_index_base           idx_base,
                                 size_t                         buffer_size,
                                 void*                          buffer);

__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_zcsritilu0_compute_ex instead.")))
ROCSPARSE_EXPORT rocsparse_status
    rocsparse_zcsritilu0_compute(rocsparse_handle                handle,
                                 rocsparse_itilu0_alg            alg,
                                 rocsparse_int                   option,
                                 rocsparse_int*                  nmaxiter,
                                 double                          tol,
                                 rocsparse_int                   m,
                                 rocsparse_int                   nnz,
                                 const rocsparse_int*            csr_row_ptr,
                                 const rocsparse_int*            csr_col_ind,
                                 const rocsparse_double_complex* csr_val,
                                 rocsparse_double_complex*       ilu0,
                                 rocsparse_index_base            idx_base,
                                 size_t                          buffer_size,
                                 void*                           buffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Iterative Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format.
*
*  \details
*  \p rocsparse_csritilu0_compute computes iteratively the incomplete LU factorization with 0 fill-ins and no
*  pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
*  \f[
*    A \approx (L + Id)(D + U)
*  \f]
*
*
*  We use the following notation for the equations below: diag is the diagonal part, lower is the strict lower triangular part and upper is the strict upper triangular part of a given matrix.
*  Starting with \f$L_{0} = lower(\f$ \p ilu0 \f$)\f$, \f$U_{0} = upper(\f$ \p ilu0 \f$)\f$, the method iterates with
*  \f[
*  \begin{eqnarray}
*     R_k &=& A - L_{k} U_{k},\\
*     D_{k+1} &=& diag(R_k),\\
*     L_{k+1} &=& lower(R_k) D_{k+1}^{-1},\\
*     U_{k+1} &=& upper(R_k),
*  \end{eqnarray}
*  \f]
*  if \f$ 0 \le k \lt \f$ \p nmaxiter and if
*  \f[
*     \Vert R_k \Vert_{\infty} \gt \epsilon \Vert A \Vert_{\infty},
*  \f]
*  with \f$\epsilon\f$ = \p tol. Note that the calculation of \f$R_k\f$ is performed with no fill-in.
*
*  The parameter \p nfreeiter is used to control the frequence of the stopping criteria evaluation, thus potentially improving the performance of the algorithm with less norm calculation. Between each iteration of index \f$ k \f$, \p nfreeiter are performed without stopping criteria evaluation. Thus, if the convergence is obtained with \f$ k \f$ This means \f$ (k + 1)( \f$ \p nfreeiter \f$ ) + k \f$ iterations.
*
*  \p rocsparse_csritilu0 requires a user allocated temporary buffer. Its size is returned
*  by rocsparse_csritilu0_buffer_size(). Furthermore,
*  analysis meta data is required. It can be obtained by rocsparse_csritlu0_preprocess().
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  rocsparse_csrsort().
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  alg         algorithm to use, \ref rocsparse_itilu0_alg
*  @param[in]
*  option      combination of enumeration values from \ref rocsparse_itilu0_option.
*  @param[inout]
*  nmaxiter     maximum number of iterations on input and number of iterations on output. If the output number of iterations is strictly less than the input maximum number of iterations, then the algorithm converged.
*  @param[inout]
*  nfreeiter    number of free iterations, i.e. the number of iterations the algorithm will perform without stopping criteria evaluations.
*  @param[in]
*  tol tolerance to use for stopping criteria.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start
*              of every row of the sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[inout]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[out]
*  ilu0        incomplete factorization.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*  @param[in]
*  buffer_size size of the storage buffer allocated by the user.
*  @param[in]
*  buffer      storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_value \p alg or \p base is invalid.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr
*              or \p csr_col_ind pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsritilu0_compute_ex(rocsparse_handle     handle,
                                                 rocsparse_itilu0_alg alg,
                                                 rocsparse_int        option,
                                                 rocsparse_int*       nmaxiter,
                                                 rocsparse_int        nfreeiter,
                                                 float                tol,
                                                 rocsparse_int        m,
                                                 rocsparse_int        nnz,
                                                 const rocsparse_int* csr_row_ptr,
                                                 const rocsparse_int* csr_col_ind,
                                                 const float*         csr_val,
                                                 float*               ilu0,
                                                 rocsparse_index_base idx_base,
                                                 size_t               buffer_size,
                                                 void*                buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsritilu0_compute_ex(rocsparse_handle     handle,
                                                 rocsparse_itilu0_alg alg,
                                                 rocsparse_int        option,
                                                 rocsparse_int*       nmaxiter,
                                                 rocsparse_int        nfreeiter,
                                                 double               tol,
                                                 rocsparse_int        m,
                                                 rocsparse_int        nnz,
                                                 const rocsparse_int* csr_row_ptr,
                                                 const rocsparse_int* csr_col_ind,
                                                 const double*        csr_val,
                                                 double*              ilu0,
                                                 rocsparse_index_base idx_base,
                                                 size_t               buffer_size,
                                                 void*                buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsritilu0_compute_ex(rocsparse_handle               handle,
                                                 rocsparse_itilu0_alg           alg,
                                                 rocsparse_int                  option,
                                                 rocsparse_int*                 nmaxiter,
                                                 rocsparse_int                  nfreeiter,
                                                 float                          tol,
                                                 rocsparse_int                  m,
                                                 rocsparse_int                  nnz,
                                                 const rocsparse_int*           csr_row_ptr,
                                                 const rocsparse_int*           csr_col_ind,
                                                 const rocsparse_float_complex* csr_val,
                                                 rocsparse_float_complex*       ilu0,
                                                 rocsparse_index_base           idx_base,
                                                 size_t                         buffer_size,
                                                 void*                          buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsritilu0_compute_ex(rocsparse_handle                handle,
                                                 rocsparse_itilu0_alg            alg,
                                                 rocsparse_int                   option,
                                                 rocsparse_int*                  nmaxiter,
                                                 rocsparse_int                   nfreeiter,
                                                 double                          tol,
                                                 rocsparse_int                   m,
                                                 rocsparse_int                   nnz,
                                                 const rocsparse_int*            csr_row_ptr,
                                                 const rocsparse_int*            csr_col_ind,
                                                 const rocsparse_double_complex* csr_val,
                                                 rocsparse_double_complex*       ilu0,
                                                 rocsparse_index_base            idx_base,
                                                 size_t                          buffer_size,
                                                 void*                           buffer);
/**@}*/

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csritilu0_history fetches convergence history data if
*  \ref rocsparse_itilu0_option_convergence_history has been set when calling
*  \ref rocsparse_scsritilu0_compute "rocsparse_Xcsritilu0_compute" or
*  \ref rocsparse_scsritilu0_compute_ex "rocsparse_Xcsritilu0_compute_ex":
*
*  \code{.c}
*  int options = 0;
*  options |= rocsparse_itilu0_option_stopping_criteria;
*  options |= rocsparse_itilu0_option_compute_nrm_residual;
*  options |= rocsparse_itilu0_option_convergence_history;
*  rocsparse_scsritilu0_compute(handle,
*                                 alg,
*                                 options,
*                                 &nmaxiter,
*                                 tol,
*                                 m,
*                                 nnz,
*                                 dcsr_row_ptr,
*                                 dcsr_col_ind,
*                                 dcsr_val,
*                                 dilu0,
*                                 idx_base,
*                                 buffer_size,
*                                 dbuffer);
*
*  if((options & rocsparse_itilu0_option_convergence_history) > 0)
*  {
*      std::vector<float> history(nmaxiter * 2);
*      rocsparse_int history_niter = 0;
*      rocsparse_scsritilu0_history(handle, alg, &history_niter, history.data(), buffer_size, dbuffer);
*
*      const bool nrm_residual = (options & rocsparse_itilu0_option_compute_nrm_residual) > 0;
*      for(rocsparse_int i = 0; i < history_niter; ++i)
*      {
*          std::cout << std::setw(12) << i;
*          if(nrm_residual)
*          {
*              std::cout << std::setw(12) << history[history_niter + i];
*          }
*          std::cout << std::endl;
*      }
*  }
*  \endcode
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  rocsparse_csrsort().
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  alg         algorithm to use, \ref rocsparse_itilu0_alg
*  @param[out]
*  niter       number of performed iterations.
*  @param[out]
*  data        norms.
*  @param[in]
*  buffer_size size of the buffer allocated by the user.
*  @param[in]
*  buffer buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p niter or \p data is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsritilu0_history(rocsparse_handle     handle,
                                              rocsparse_itilu0_alg alg,
                                              rocsparse_int*       niter,
                                              float*               data,
                                              size_t               buffer_size,
                                              void*                buffer);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsritilu0_history(rocsparse_handle     handle,
                                              rocsparse_itilu0_alg alg,
                                              rocsparse_int*       niter,
                                              double*              data,
                                              size_t               buffer_size,
                                              void*                buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsritilu0_history(rocsparse_handle     handle,
                                              rocsparse_itilu0_alg alg,
                                              rocsparse_int*       niter,
                                              float*               data,
                                              size_t               buffer_size,
                                              void*                buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsritilu0_history(rocsparse_handle     handle,
                                              rocsparse_itilu0_alg alg,
                                              rocsparse_int*       niter,
                                              double*              data,
                                              size_t               buffer_size,
                                              void*                buffer);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSRITILU0_H */
