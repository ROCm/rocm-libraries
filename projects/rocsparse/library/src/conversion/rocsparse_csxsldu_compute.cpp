/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "internal/conversion/rocsparse_csr2csc.h"
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csr2csc.hpp"
#include "rocsparse_csxsldu.hpp"
#include "rocsparse_utility.hpp"

#include "csr2csc_device.h"

namespace rocsparse
{
    template <uint32_t            BLOCKSIZE,
              rocsparse_diag_type FDIAG,
              rocsparse_diag_type SDIAG,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csxsldu_fill_kernel(J                    nseq_,
                             const I*             ptr_,
                             const J*             ind_,
                             const T*             val_,
                             rocsparse_index_base base_,
                             const I*             fptr_,
                             J*                   find_,
                             T*                   fval_,
                             rocsparse_index_base fbase_,
                             const I*             sptr_,
                             J*                   sind_,
                             T*                   sval_,
                             rocsparse_index_base sbase_,
                             T*                   diag_)
    {
        I seq = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        if(seq < nseq_)
        {
            I       start  = ptr_[seq] - base_;
            const I end    = ptr_[seq + 1] - base_;
            const I fstart = fptr_[seq] - fbase_;
            const I sstart = sptr_[seq] - sbase_;
            I       k      = start;
            for(; k < end; ++k)
            {
                const J local_k = k - start;
                const J ind     = ind_[k] - base_;

                if(FDIAG == rocsparse_diag_type_unit)
                {
                    if(seq <= ind)
                    {
                        break;
                    }
                }
                else
                {
                    if(seq < ind)
                    {
                        break;
                    }
                }

                find_[fstart + local_k] = ind + fbase_;
                fval_[fstart + local_k] = val_[k];
            }

            if(SDIAG == rocsparse_diag_type_unit && FDIAG == rocsparse_diag_type_unit)
            {
                if((k < end) && ((ind_[k] - base_) == seq))
                {
                    diag_[seq] = val_[k];
                    ++k;
                }
            }

            start = k;
            for(; k < end; ++k)
            {
                const J local_k         = k - start;
                const J ind             = ind_[k] - base_;
                sind_[sstart + local_k] = ind + sbase_;
                sval_[sstart + local_k] = val_[k];
            }
        }
    }

    template <int nthreads_per_block, typename I, typename J, typename... P>
    static rocsparse_status csxsldu_fill_kernel_dispatch(rocsparse_handle    handle_,
                                                         dim3&               blocks,
                                                         dim3&               threads,
                                                         rocsparse_diag_type ldiag_,
                                                         rocsparse_diag_type udiag_,
                                                         P... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(ldiag_)
        {
        case rocsparse_diag_type_unit:
        {
            switch(udiag_)
            {
            case rocsparse_diag_type_non_unit:
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csxsldu_fill_kernel<nthreads_per_block,
                                                    rocsparse_diag_type_unit,
                                                    rocsparse_diag_type_non_unit,
                                                    I,
                                                    J>),
                    blocks,
                    threads,
                    0,
                    handle_->stream,
                    p...);
                break;
            }
            case rocsparse_diag_type_unit:
            {

                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csxsldu_fill_kernel<nthreads_per_block,
                                                    rocsparse_diag_type_unit,
                                                    rocsparse_diag_type_unit,
                                                    I,
                                                    J>),
                    blocks,
                    threads,
                    0,
                    handle_->stream,
                    p...);
                break;
            }
            }
            break;
        }
        case rocsparse_diag_type_non_unit:
        {
            switch(udiag_)
            {
            case rocsparse_diag_type_non_unit:
            {
                // LCOV_EXCL_START
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
                // LCOV_EXCL_STOP
            }
            case rocsparse_diag_type_unit:
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csxsldu_fill_kernel<nthreads_per_block,
                                                    rocsparse_diag_type_non_unit,
                                                    rocsparse_diag_type_unit,
                                                    I,
                                                    J>),
                    blocks,
                    threads,
                    0,
                    handle_->stream,
                    p...);
                break;
            }
            }
            break;
        }
        }
        return rocsparse_status_success;
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse::csxsldu_compute_template(rocsparse_handle handle_,
                                                     //
                                                     rocsparse_direction  dir_,
                                                     J                    m_,
                                                     J                    n_,
                                                     I                    nnz_,
                                                     const I*             ptr_,
                                                     const J*             ind_,
                                                     T*                   val_,
                                                     rocsparse_index_base base_,
                                                     //
                                                     rocsparse_diag_type  ldiag_,
                                                     rocsparse_direction  ldir_,
                                                     I                    lnnz_,
                                                     I*                   lptr_,
                                                     J*                   lind_,
                                                     T*                   lval_,
                                                     rocsparse_index_base lbase_,
                                                     //
                                                     rocsparse_diag_type  udiag_,
                                                     rocsparse_direction  udir_,
                                                     I                    unnz_,
                                                     I*                   uptr_,
                                                     J*                   uind_,
                                                     T*                   uval_,
                                                     rocsparse_index_base ubase_,
                                                     //
                                                     T* diag_,
                                                     //
                                                     void* buffer_)
{
    ROCSPARSE_ROUTINE_TRACE;

    static constexpr uint32_t nthreads_per_block = 1024;
    J                         size               = (dir_ == rocsparse_direction_row) ? m_ : n_;
    J                         sizet              = (dir_ == rocsparse_direction_row) ? n_ : m_;
    J                         nblocks            = (size - 1) / nthreads_per_block + 1;
    dim3                      blocks(nblocks);
    dim3                      threads(nthreads_per_block);

    I* buffer = (I*)buffer_;
    I* lptr   = (ldir_ != dir_) ? buffer : lptr_;
    I* uptr
        = (udir_ != dir_)
              ? ((ldir_ != dir_) ? (buffer + (((rocsparse_direction_row == dir_) ? m_ : n_) + 1))
                                 : buffer)
              : uptr_;

    if(dir_ == rocsparse_direction_row)
    {
        //
        // calculated csr_row_uptr_ is in the buffer.
        //
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csxsldu_fill_kernel_dispatch<nthreads_per_block, T, I, J>)(handle_,
                                                                                   blocks,
                                                                                   threads,
                                                                                   udiag_,
                                                                                   ldiag_,

                                                                                   size,
                                                                                   ptr_,
                                                                                   ind_,
                                                                                   val_,
                                                                                   base_,

                                                                                   lptr,
                                                                                   lind_,
                                                                                   lval_,
                                                                                   lbase_,

                                                                                   uptr,
                                                                                   uind_,
                                                                                   uval_,
                                                                                   ubase_,

                                                                                   diag_));
    }
    else
    {

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csxsldu_fill_kernel_dispatch<nthreads_per_block, T, I, J>)(handle_,
                                                                                   blocks,
                                                                                   threads,
                                                                                   udiag_,
                                                                                   ldiag_,

                                                                                   size,
                                                                                   ptr_,
                                                                                   ind_,
                                                                                   val_,
                                                                                   base_,

                                                                                   uptr,
                                                                                   uind_,
                                                                                   uval_,
                                                                                   ubase_,

                                                                                   lptr,
                                                                                   lind_,
                                                                                   lval_,
                                                                                   lbase_,

                                                                                   diag_));
    }

    if(udir_ != dir_)
    {
        if(unnz_ == 0)
        {
            if(m_ == n_)
            {
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync(uptr_, 0, sizeof(I) * (sizet + 1), handle_->stream));
            }
            else
            {
                if(ubase_ == rocsparse_index_base_zero)
                {
                    RETURN_IF_HIP_ERROR(
                        hipMemsetAsync(uptr_, 0, sizeof(I) * (sizet + 1), handle_->stream));
                }
                else
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::valset(handle_,
                                          sizet + 1,
                                          static_cast<rocsparse_int>(rocsparse_index_base_one),
                                          uptr_));
                }
            }
        }
        else
        {
            size_t buffer_size;
            // convert u to csc.
            //
            // Synchronize.
            //
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2csc_buffer_size(
                handle_, m_, n_, unnz_, uptr, uind_, rocsparse_action_numeric, &buffer_size));

            void* buffer_conversion;
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&buffer_conversion, buffer_size, handle_->stream));

            J* tmp_ind;
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&tmp_ind, sizeof(J) * unnz_, handle_->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                tmp_ind, uind_, sizeof(J) * (unnz_), hipMemcpyDeviceToDevice, handle_->stream));
            T* tmp_val;
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&tmp_val, sizeof(T) * unnz_, handle_->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                tmp_val, uval_, sizeof(T) * (unnz_), hipMemcpyDeviceToDevice, handle_->stream));
            I* tmp_uptr = uptr;
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2csc_template(handle_,
                                                                  m_,
                                                                  n_,
                                                                  unnz_,
                                                                  tmp_val,
                                                                  tmp_uptr,
                                                                  tmp_ind,
                                                                  uval_,
                                                                  uind_,
                                                                  uptr_,
                                                                  rocsparse_action_numeric,
                                                                  ubase_,
                                                                  buffer_conversion));
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(buffer_conversion, handle_->stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(tmp_val, handle_->stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(tmp_ind, handle_->stream));
        }
    }

    if(ldir_ != dir_)
    {
        if(lnnz_ == 0)
        {
            if(m_ == n_)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                    lptr_, lptr, sizeof(I) * (m_ + 1), hipMemcpyDeviceToDevice, handle_->stream));
            }
            else
            {
                if(lbase_ == rocsparse_index_base_zero)
                {
                    RETURN_IF_HIP_ERROR(
                        hipMemsetAsync(lptr_, 0, sizeof(I) * (sizet + 1), handle_->stream));
                }
                else
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::valset(handle_,
                                          sizet + 1,
                                          static_cast<rocsparse_int>(rocsparse_index_base_one),
                                          lptr_));
                }
            }
        }
        else
        {

            size_t buffer_size;
            // convert l to csr.
            // convert u to csc.
            //
            // Synchronize.
            //
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2csc_buffer_size(
                handle_, n_, m_, lnnz_, lptr, lind_, rocsparse_action_numeric, &buffer_size));

            void* buffer_conversion;
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&buffer_conversion, buffer_size, handle_->stream));

            J* tmp_ind;
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&tmp_ind, sizeof(J) * lnnz_, handle_->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                tmp_ind, lind_, sizeof(J) * (lnnz_), hipMemcpyDeviceToDevice, handle_->stream));

            T* tmp_val;
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&tmp_val, sizeof(T) * lnnz_, handle_->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                tmp_val, lval_, sizeof(T) * (lnnz_), hipMemcpyDeviceToDevice, handle_->stream));

            I* tmp_lptr = lptr;
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2csc_template(handle_,
                                                                  n_,
                                                                  m_,
                                                                  lnnz_,
                                                                  tmp_val,
                                                                  tmp_lptr,
                                                                  tmp_ind,
                                                                  lval_,
                                                                  lind_,
                                                                  lptr_,
                                                                  rocsparse_action_numeric,
                                                                  lbase_,
                                                                  buffer_conversion));

            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(buffer_conversion, handle_->stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(tmp_val, handle_->stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(tmp_ind, handle_->stream));
        }
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(TOK, T, I, J)                                        \
    template rocsparse_status rocsparse::csxsldu_compute_##TOK<T, I, J>( \
        rocsparse_handle     handle_,                                    \
        rocsparse_direction  dir_,                                       \
        J                    m_,                                         \
        J                    n_,                                         \
        I                    nnz_,                                       \
        const I*             ptr_,                                       \
        const J*             ind_,                                       \
        T*                   val_,                                       \
        rocsparse_index_base base_,                                      \
        rocsparse_diag_type  ldiag_,                                     \
        rocsparse_direction  ldir_,                                      \
        I                    lnnz_,                                      \
        I*                   lptr_,                                      \
        J*                   lind_,                                      \
        T*                   lval_,                                      \
        rocsparse_index_base lbase_,                                     \
        rocsparse_diag_type  udiag_,                                     \
        rocsparse_direction  udir_,                                      \
        I                    unnz_,                                      \
        I*                   uptr_,                                      \
        J*                   uind_,                                      \
        T*                   uval_,                                      \
        rocsparse_index_base ubase_,                                     \
        T*                   diag_,                                      \
        void*                buffer_)

INSTANTIATE(template, rocsparse_int, rocsparse_int, rocsparse_int);
INSTANTIATE(template, float, rocsparse_int, rocsparse_int);
INSTANTIATE(template, double, rocsparse_int, rocsparse_int);
INSTANTIATE(template, rocsparse_float_complex, rocsparse_int, rocsparse_int);
INSTANTIATE(template, rocsparse_double_complex, rocsparse_int, rocsparse_int);

#undef INSTANTIATE

#define C_IMPL(NAME, TYPE)                                                       \
    extern "C" rocsparse_status NAME(rocsparse_handle     handle_,               \
                                     rocsparse_direction  dir_,                  \
                                     rocsparse_int        m_,                    \
                                     rocsparse_int        n_,                    \
                                     rocsparse_int        nnz_,                  \
                                     const rocsparse_int* ptr_,                  \
                                     const rocsparse_int* ind_,                  \
                                     TYPE*                val_,                  \
                                     rocsparse_index_base base_,                 \
                                     rocsparse_diag_type  ldiag_,                \
                                     rocsparse_direction  ldir_,                 \
                                     rocsparse_int        lnnz_,                 \
                                     rocsparse_int*       lptr_,                 \
                                     rocsparse_int*       lind_,                 \
                                     TYPE*                lval_,                 \
                                     rocsparse_index_base lbase_,                \
                                     rocsparse_diag_type  udiag_,                \
                                     rocsparse_direction  udir_,                 \
                                     rocsparse_int        unnz_,                 \
                                     rocsparse_int*       uptr_,                 \
                                     rocsparse_int*       uind_,                 \
                                     TYPE*                uval_,                 \
                                     rocsparse_index_base ubase_,                \
                                     TYPE*                diag_,                 \
                                     void*                buffer_)               \
    try                                                                          \
    {                                                                            \
        ROCSPARSE_ROUTINE_TRACE;                                                 \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csxsldu_compute_template(handle_,   \
                                                                      dir_,      \
                                                                      m_,        \
                                                                      n_,        \
                                                                      nnz_,      \
                                                                      ptr_,      \
                                                                      ind_,      \
                                                                      val_,      \
                                                                      base_,     \
                                                                      ldiag_,    \
                                                                      ldir_,     \
                                                                      lnnz_,     \
                                                                      lptr_,     \
                                                                      lind_,     \
                                                                      lval_,     \
                                                                      lbase_,    \
                                                                      udiag_,    \
                                                                      udir_,     \
                                                                      unnz_,     \
                                                                      uptr_,     \
                                                                      uind_,     \
                                                                      uval_,     \
                                                                      ubase_,    \
                                                                      diag_,     \
                                                                      buffer_)); \
        return rocsparse_status_success;                                         \
    }                                                                            \
    catch(...)                                                                   \
    {                                                                            \
        RETURN_ROCSPARSE_EXCEPTION();                                            \
    }

C_IMPL(rocsparse_scsxsldu_compute, float);
C_IMPL(rocsparse_dcsxsldu_compute, double);
C_IMPL(rocsparse_ccsxsldu_compute, rocsparse_float_complex);
C_IMPL(rocsparse_zcsxsldu_compute, rocsparse_double_complex);
#undef C_IMPL
