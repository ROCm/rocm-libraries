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

#ifndef ROCSPARSE_GPSV_H
#define ROCSPARSE_GPSV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \details
*  \p rocsparse_gpsv_interleaved_batch_buffer_size calculates the required buffer size
*  for \ref rocsparse_sgpsv_interleaved_batch "rocsparse_Xgpsv_interleaved_batch()". It is the user's 
*  responsibility to allocate this buffer.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  alg          algorithm to solve the linear system.
*  @param[in]
*  m            size of the pentadiagonal linear system.
*  @param[in]
*  ds           lower diagonal (distance 2) of pentadiagonal system. First two entries
*               must be zero.
*  @param[in]
*  dl           lower diagonal of pentadiagonal system. First entry must be zero.
*  @param[in]
*  d            main diagonal of pentadiagonal system.
*  @param[in]
*  du           upper diagonal of pentadiagonal system. Last entry must be zero.
*  @param[in]
*  dw           upper diagonal (distance 2) of pentadiagonal system. Last two entries
*               must be zero.
*  @param[in]
*  x            Dense array of right-hand-sides with dimension \p batch_stride by \p m.
*  @param[in]
*  batch_count  The number of systems to solve.
*  @param[in]
*  batch_stride The number of elements that separate consecutive elements in a system.
*               Must satisfy \p batch_stride >= \p batch_count.
*  @param[out]
*  buffer_size  Number of bytes of the temporary storage buffer required.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p alg, \p batch_count or
*              \p batch_stride is invalid.
*  \retval     rocsparse_status_invalid_pointer \p ds, \p dl, \p d, \p du, \p dw, \p x
*              or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgpsv_interleaved_batch_buffer_size(rocsparse_handle handle,
                                                               rocsparse_gpsv_interleaved_alg alg,
                                                               rocsparse_int                  m,
                                                               const float*                   ds,
                                                               const float*                   dl,
                                                               const float*                   d,
                                                               const float*                   du,
                                                               const float*                   dw,
                                                               const float*                   x,
                                                               rocsparse_int batch_count,
                                                               rocsparse_int batch_stride,
                                                               size_t*       buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgpsv_interleaved_batch_buffer_size(rocsparse_handle handle,
                                                               rocsparse_gpsv_interleaved_alg alg,
                                                               rocsparse_int                  m,
                                                               const double*                  ds,
                                                               const double*                  dl,
                                                               const double*                  d,
                                                               const double*                  du,
                                                               const double*                  dw,
                                                               const double*                  x,
                                                               rocsparse_int batch_count,
                                                               rocsparse_int batch_stride,
                                                               size_t*       buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgpsv_interleaved_batch_buffer_size(rocsparse_handle handle,
                                                               rocsparse_gpsv_interleaved_alg alg,
                                                               rocsparse_int                  m,
                                                               const rocsparse_float_complex* ds,
                                                               const rocsparse_float_complex* dl,
                                                               const rocsparse_float_complex* d,
                                                               const rocsparse_float_complex* du,
                                                               const rocsparse_float_complex* dw,
                                                               const rocsparse_float_complex* x,
                                                               rocsparse_int batch_count,
                                                               rocsparse_int batch_stride,
                                                               size_t*       buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgpsv_interleaved_batch_buffer_size(rocsparse_handle handle,
                                                               rocsparse_gpsv_interleaved_alg  alg,
                                                               rocsparse_int                   m,
                                                               const rocsparse_double_complex* ds,
                                                               const rocsparse_double_complex* dl,
                                                               const rocsparse_double_complex* d,
                                                               const rocsparse_double_complex* du,
                                                               const rocsparse_double_complex* dw,
                                                               const rocsparse_double_complex* x,
                                                               rocsparse_int batch_count,
                                                               rocsparse_int batch_stride,
                                                               size_t*       buffer_size);
/**@}*/

/*! \ingroup precond_module
*  \brief Batched Pentadiagonal solver
*
*  \details
*  \p rocsparse_gpsv_interleaved_batch solves a batch of pentadiagonal linear systems
*  \f[
*    P^{i}*x^{i} = x^{i}
*  \f]
*  where for each batch \f$i=0\ldots\f$ \p batch_count, \f$P^{i}\f$ is a sparse pentadiagonal matrix and 
*  \f$x^{i}\f$ is a dense right-hand side vector. All of the pentadiagonal matrices, \f$P^{i}\f$, are 
*  packed in an interleaved fashion into five vectors: \p ds for the lowest diagonals, \p dl for the lower 
*  diagonals, \p d for the main diagonals, \p du for the upper diagonals, and \p dw for the highest digaonals. 
*  See below for a description of what this interleaved memory pattern looks like.
*
*  Solving the batched pentadiagonal system involves two steps. First, the user calls 
*  \ref rocsparse_sgpsv_interleaved_batch_buffer_size "rocsparse_Xgpsv_interleaved_batch_buffer_size()" 
*  in order to determine the size of the required temporary storage buffer. Once determined, the user allocates 
*  this buffer and passes it to \ref rocsparse_sgpsv_interleaved_batch "rocsparse_Xgpsv_interleaved_batch()" 
*  to perform the actual solve. The \f$x^{i}\f$ vectors, which initially stores the right-hand side values, are 
*  overwritten with the solution after the call to 
*  \ref rocsparse_sgpsv_interleaved_batch "rocsparse_Xgpsv_interleaved_batch()".
*
*  Unlike the strided batch routines which write each batch matrix one after the other in memory, the interleaved 
*  routines write the batch matrices such that each element from each matrix is written consecutively one after 
*  the other. For example, consider the following batch matrices:
*
*  \f[
*    \begin{bmatrix}
*    t^{0}_{00} & t^{0}_{01} & t^{0}_{02} \\
*    t^{0}_{10} & t^{0}_{11} & t^{0}_{12} \\
*    t^{0}_{20} & t^{0}_{21} & t^{0}_{22}
*    \end{bmatrix}
*    \begin{bmatrix}
*    t^{1}_{00} & t^{1}_{01} & t^{1}_{02} \\
*    t^{1}_{10} & t^{1}_{11} & t^{1}_{12} \\
*    t^{1}_{20} & t^{1}_{21} & t^{1}_{22}
*    \end{bmatrix}
*    \begin{bmatrix}
*    t^{2}_{00} & t^{2}_{01} & t^{2}_{02} \\
*    t^{2}_{10} & t^{2}_{11} & t^{2}_{12} \\
*    t^{2}_{20} & t^{2}_{21} & t^{2}_{22}
*    \end{bmatrix}
*  \f] 
*
*  In interleaved format, the highest, higher, lowest, lower, and diagonal arrays would look like:
*  \f[
*    \begin{align}
*    \text{lowest} &= \begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 0 & t^{0}_{20} & t^{1}_{20} & t^{2}_{20} \end{bmatrix} \\
*    \text{lower} &= \begin{bmatrix} 0 & 0 & 0 & t^{0}_{10} & t^{1}_{10} & t^{1}_{10} & t^{0}_{21} & t^{1}_{21} & t^{2}_{21} \end{bmatrix} \\
*    \text{diagonal} &= \begin{bmatrix} t^{0}_{00} & t^{1}_{00} & t^{2}_{00} & t^{0}_{11} & t^{1}_{11} & t^{2}_{11} & t^{0}_{22} & t^{1}_{22} & t^{2}_{22} \end{bmatrix} \\
*    \text{higher} &= \begin{bmatrix} t^{0}_{01} & t^{1}_{01} & t^{2}_{01} & t^{0}_{12} & t^{1}_{12} & t^{2}_{12} & 0 & 0 & 0 \end{bmatrix} \\
*    \text{highest} &= \begin{bmatrix} t^{0}_{02} & t^{1}_{02} & t^{2}_{02} & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix} \\
*    \end{align}
*  \f]
*  For the lowest array, the first \p 2*batch_count entries are zero, for the lower array, the first \p batch_count entries are zero, 
*  for the upper array the last \p batch_count entries are zero, and for the highest array, the last \p 2*batch_count entries are zero.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  The routine is numerically stable because it uses QR to solve the linear systems.
*
*  \note
*  m need to be at least 3, to be a valid pentadiagonal matrix.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  alg          algorithm to solve the linear system.
*  @param[in]
*  m            size of the pentadiagonal linear system.
*  @param[inout]
*  ds           lower diagonal (distance 2) of pentadiagonal system. First two entries
*               must be zero.
*  @param[inout]
*  dl           lower diagonal of pentadiagonal system. First entry must be zero.
*  @param[inout]
*  d            main diagonal of pentadiagonal system.
*  @param[inout]
*  du           upper diagonal of pentadiagonal system. Last entry must be zero.
*  @param[inout]
*  dw           upper diagonal (distance 2) of pentadiagonal system. Last two entries
*               must be zero.
*  @param[inout]
*  x            Dense array of right-hand-sides with dimension \p batch_stride by \p m.
*  @param[in]
*  batch_count  The number of systems to solve.
*  @param[in]
*  batch_stride The number of elements that separate consecutive elements in a system.
*               Must satisfy \p batch_stride >= \p batch_count.
*  @param[in]
*  temp_buffer  Temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p alg, \p batch_count or
*              \p batch_stride is invalid.
*  \retval     rocsparse_status_invalid_pointer \p ds, \p dl, \p d, \p du, \p dw, \p x
*              or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*  \par Example
*  \code{.c}
*   // Size of each square pentadiagonal matrix
*   rocsparse_int m = 6;
*
*   // Number of batches
*   rocsparse_int batch_count = 4;
*
*   // Batch stride
*   rocsparse_int batch_stride = batch_count;
*
*   // Host pentadiagonal matrix
*   std::vector<float> hds(m * batch_stride);
*   std::vector<float> hdl(m * batch_stride);
*   std::vector<float> hd(m * batch_stride);
*   std::vector<float> hdu(m * batch_stride);
*   std::vector<float> hdw(m * batch_stride);
*
*   // Solve multiple pentadiagonal matrix systems by interleaving matrices for better memory access:
*   //
*   //      4 2 1 0 0 0        5 3 2 0 0 0        6 4 3 0 0 0        7 5 4 0 0 0
*   //      2 4 2 1 0 0        3 5 3 2 0 0        4 6 4 3 0 0        5 7 5 4 0 0
*   // A1 = 1 2 4 2 1 0   A2 = 2 3 5 3 2 0   A3 = 3 4 6 4 3 0   A4 = 4 5 7 5 4 0
*   //      0 1 2 4 2 1        0 2 3 5 3 2        0 3 4 6 4 3        0 4 5 7 5 4
*   //      0 0 1 2 4 2        0 0 2 3 5 3        0 0 3 4 6 4        0 0 4 5 7 5
*   //      0 0 0 1 2 4        0 0 0 2 3 5        0 0 0 3 4 6        0 0 0 4 5 7
*   //
*   // hds = 0 0 0 0 0 0 0 0 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4
*   // hdl = 0 0 0 0 2 3 4 5 2 3 4 5 2 3 4 5 2 3 4 5 2 3 4 5
*   // hd  = 4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7
*   // hdu = 2 3 4 5 2 3 4 5 2 3 4 5 2 3 4 5 2 3 4 5 0 0 0 0
*   // hdw = 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 0 0 0 0 0 0 0 0
*   for(int b = 0; b < batch_count; ++b)
*   {
*       for(rocsparse_int i = 0; i < m; ++i)
*       {
*           hds[batch_stride * i + b] = 1 + b;
*           hdl[batch_stride * i + b] = 2 + b;
*           hd[batch_stride * i + b]  = 4 + b;
*           hdu[batch_stride * i + b] = 2 + b;
*           hdw[batch_stride * i + b] = 1 + b;
*       }
*
*       hds[batch_stride * 0 + b]       = 0.0f;
*       hds[batch_stride * 1 + b]       = 0.0f;
*       hdl[batch_stride * 0 + b]       = 0.0f;
*       hdu[batch_stride * (m - 1) + b] = 0.0f;
*       hdw[batch_stride * (m - 1) + b] = 0.0f;
*       hdw[batch_stride * (m - 2) + b] = 0.0f;
*   }
*
*   // Host dense rhs
*   std::vector<float> hx(m * batch_stride);
*
*   for(int b = 0; b < batch_count; ++b)
*   {
*       for(int i = 0; i < m; ++i)
*       {
*           hx[batch_stride * i + b] = static_cast<float>(b + 1);
*       }
*   }
*
*   float* dds = nullptr;
*   float* ddl = nullptr;
*   float* dd = nullptr;
*   float* ddu = nullptr;
*   float* ddw = nullptr;
*   float* dx = nullptr;
*   hipMalloc((void**)&dds, sizeof(float) * m * batch_stride);
*   hipMalloc((void**)&ddl, sizeof(float) * m * batch_stride);
*   hipMalloc((void**)&dd, sizeof(float) * m * batch_stride);
*   hipMalloc((void**)&ddu, sizeof(float) * m * batch_stride);
*   hipMalloc((void**)&ddw, sizeof(float) * m * batch_stride);
*   hipMalloc((void**)&dx, sizeof(float) * m * batch_stride);
*
*   hipMemcpy(dds, hds.data(), sizeof(float) * m * batch_stride, hipMemcpyHostToDevice);
*   hipMemcpy(ddl, hdl.data(), sizeof(float) * m * batch_stride, hipMemcpyHostToDevice);
*   hipMemcpy(dd, hd.data(), sizeof(float) * m * batch_stride, hipMemcpyHostToDevice);
*   hipMemcpy(ddu, hdu.data(), sizeof(float) * m * batch_stride, hipMemcpyHostToDevice);
*   hipMemcpy(ddw, hdw.data(), sizeof(float) * m * batch_stride, hipMemcpyHostToDevice);
*   hipMemcpy(dx, hx.data(), sizeof(float) * m * batch_stride, hipMemcpyHostToDevice);
*
*   // rocSPARSE handle
*   rocsparse_handle handle;
*   rocsparse_create_handle(&handle);
*
*   // Obtain required buffer size
*   size_t buffer_size;
*   rocsparse_sgpsv_interleaved_batch_buffer_size(handle,
*                                                 rocsparse_gpsv_interleaved_alg_default,
*                                                 m,
*                                                 dds,
*                                                 ddl,
*                                                 dd,
*                                                 ddu,
*                                                 ddw,
*                                                 dx,
*                                                 batch_count,
*                                                 batch_stride,
*                                                 &buffer_size);
*
*   void* dbuffer;
*   hipMalloc(&dbuffer, buffer_size);
*
*   rocsparse_sgpsv_interleaved_batch(handle,
*                                     rocsparse_gpsv_interleaved_alg_default,
*                                     m,
*                                     dds,
*                                     ddl,
*                                     dd,
*                                     ddu,
*                                     ddw,
*                                     dx,
*                                     batch_count,
*                                     batch_stride,
*                                     dbuffer);
*
*   // Copy right-hand side to host
*   hipMemcpy(hx.data(), dx, sizeof(float) * m * batch_stride, hipMemcpyDeviceToHost);
*
*   // Clear rocSPARSE
*   rocsparse_destroy_handle(handle);
*
*   // Clear device memory
*   hipFree(dds);
*   hipFree(ddl);
*   hipFree(dd);
*   hipFree(ddu);
*   hipFree(ddw);
*   hipFree(dx);
*   hipFree(dbuffer);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgpsv_interleaved_batch(rocsparse_handle               handle,
                                                   rocsparse_gpsv_interleaved_alg alg,
                                                   rocsparse_int                  m,
                                                   float*                         ds,
                                                   float*                         dl,
                                                   float*                         d,
                                                   float*                         du,
                                                   float*                         dw,
                                                   float*                         x,
                                                   rocsparse_int                  batch_count,
                                                   rocsparse_int                  batch_stride,
                                                   void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgpsv_interleaved_batch(rocsparse_handle               handle,
                                                   rocsparse_gpsv_interleaved_alg alg,
                                                   rocsparse_int                  m,
                                                   double*                        ds,
                                                   double*                        dl,
                                                   double*                        d,
                                                   double*                        du,
                                                   double*                        dw,
                                                   double*                        x,
                                                   rocsparse_int                  batch_count,
                                                   rocsparse_int                  batch_stride,
                                                   void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgpsv_interleaved_batch(rocsparse_handle               handle,
                                                   rocsparse_gpsv_interleaved_alg alg,
                                                   rocsparse_int                  m,
                                                   rocsparse_float_complex*       ds,
                                                   rocsparse_float_complex*       dl,
                                                   rocsparse_float_complex*       d,
                                                   rocsparse_float_complex*       du,
                                                   rocsparse_float_complex*       dw,
                                                   rocsparse_float_complex*       x,
                                                   rocsparse_int                  batch_count,
                                                   rocsparse_int                  batch_stride,
                                                   void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgpsv_interleaved_batch(rocsparse_handle               handle,
                                                   rocsparse_gpsv_interleaved_alg alg,
                                                   rocsparse_int                  m,
                                                   rocsparse_double_complex*      ds,
                                                   rocsparse_double_complex*      dl,
                                                   rocsparse_double_complex*      d,
                                                   rocsparse_double_complex*      du,
                                                   rocsparse_double_complex*      dw,
                                                   rocsparse_double_complex*      x,
                                                   rocsparse_int                  batch_count,
                                                   rocsparse_int                  batch_stride,
                                                   void*                          temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_GPSV_H */
