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
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csxsldu.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename I, typename J>
    static rocsparse_status inclusive_scan(rocsparse_handle handle, J m_, I* ptr_)
    {
        ROCSPARSE_ROUTINE_TRACE;

        size_t temp_storage_size_bytes;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::inclusive_scan_buffer_size<I, I>(
            handle, m_ + 1, &temp_storage_size_bytes)));

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= temp_storage_size_bytes)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
                &temp_storage_ptr, temp_storage_size_bytes, handle->stream));
            temp_alloc = true;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(
            handle, ptr_, ptr_, m_ + 1, temp_storage_size_bytes, temp_storage_ptr));

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
        }

        return rocsparse_status_success;
    }

    template <uint32_t            BLOCKSIZE,
              rocsparse_diag_type FDIAG,
              rocsparse_diag_type SDIAG,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csxtril_count_kernel(
        J nseq_, const I* ptr_, const J* ind_, rocsparse_index_base base_, I* fptr_, I* sptr_)
    {
        I seq = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        if(seq < nseq_)
        {
            I fcount = 0, scount = 0;
            for(I k = ptr_[seq] - base_; k < ptr_[seq + 1] - base_; ++k)
            {
                const J ind = ind_[k] - base_;
                if(seq > ind)
                {
                    ++fcount;
                }
                else if(seq < ind)
                {
                    ++scount;
                }
                else
                {
                    if(FDIAG == rocsparse_diag_type_non_unit)
                    {
                        ++fcount;
                    }
                    else if(SDIAG == rocsparse_diag_type_non_unit)
                    {
                        ++scount;
                    }
                }
            }
            fptr_[seq + 1] = fcount;
            sptr_[seq + 1] = scount;
        }
    }

    template <int nthreads_per_block, typename I, typename J, typename... P>
    rocsparse_status csxtril_count_kernel_dispatch(rocsparse_handle    handle_,
                                                   dim3&               blocks,
                                                   dim3&               threads,
                                                   rocsparse_diag_type ldiag_,
                                                   rocsparse_diag_type udiag_,
                                                   P&&... p)
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
                    (rocsparse::csxtril_count_kernel<nthreads_per_block,
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
                    (rocsparse::csxtril_count_kernel<nthreads_per_block,
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
                    (rocsparse::csxtril_count_kernel<nthreads_per_block,
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
rocsparse_status rocsparse::csxsldu_preprocess_template(rocsparse_handle     handle_,
                                                        rocsparse_direction  dir_,
                                                        J                    m_,
                                                        J                    n_,
                                                        I                    nnz_,
                                                        const I*             ptr_,
                                                        const J*             ind_,
                                                        const T*             val_,
                                                        rocsparse_index_base base_,
                                                        rocsparse_diag_type  ldiag_,
                                                        rocsparse_direction  ldir_,
                                                        I*                   host_lnnz_,
                                                        I*                   lptr_,
                                                        rocsparse_index_base lbase_,
                                                        rocsparse_diag_type  udiag_,
                                                        rocsparse_direction  udir_,
                                                        I*                   host_unnz_,
                                                        I*                   uptr_,
                                                        rocsparse_index_base ubase_,
                                                        void*                buffer_)
{
    ROCSPARSE_ROUTINE_TRACE;

    const I              ubase              = static_cast<I>(ubase_);
    const I              lbase              = static_cast<I>(lbase_);
    static constexpr int nthreads_per_block = 1024;
    dim3                 threads(nthreads_per_block);
    I*                   buffer = (I*)buffer_;
    I*                   lptr   = (ldir_ != dir_) ? buffer : lptr_;

    I* uptr
        = (udir_ != dir_)
              ? ((ldir_ != dir_) ? (buffer + (((rocsparse_direction_row == dir_) ? m_ : n_) + 1))
                                 : buffer)
              : uptr_;

    switch(dir_)
    {
    case rocsparse_direction_row:
    {
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(lptr, &lbase, sizeof(I), hipMemcpyHostToDevice, handle_->stream));
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(uptr, &ubase, sizeof(I), hipMemcpyHostToDevice, handle_->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));
        J    nblocks = (m_ - 1) / nthreads_per_block + 1;
        dim3 blocks(nblocks);
        rocsparse::csxtril_count_kernel_dispatch<nthreads_per_block, I, J>(
            handle_, blocks, threads, ldiag_, udiag_, m_, ptr_, ind_, base_, lptr, uptr);
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::inclusive_scan(handle_, m_, lptr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::inclusive_scan(handle_, m_, uptr));
        break;
    }

    case rocsparse_direction_column:
    {
        J    nblocks = (n_ - 1) / nthreads_per_block + 1;
        dim3 blocks(nblocks);
        rocsparse::csxtril_count_kernel_dispatch<nthreads_per_block, I, J>(
            handle_, blocks, threads, udiag_, ldiag_, n_, ptr_, ind_, base_, uptr, lptr);
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::inclusive_scan(handle_, n_, lptr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::inclusive_scan(handle_, n_, uptr));
        break;
    }
    }

    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(host_lnnz_, &lptr[m_], sizeof(I), hipMemcpyDeviceToHost, handle_->stream));
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(host_unnz_, &uptr[m_], sizeof(I), hipMemcpyDeviceToHost, handle_->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));

    host_lnnz_[0] -= lbase;
    host_unnz_[0] -= ubase;
    return rocsparse_status_success;
}

#define INSTANTIATE(TOK, T, I, J)                                           \
    template rocsparse_status rocsparse::csxsldu_preprocess_##TOK<T, I, J>( \
        rocsparse_handle     handle_,                                       \
        rocsparse_direction  dir_,                                          \
        J                    m_,                                            \
        J                    n_,                                            \
        I                    nnz_,                                          \
        const I*             ptr_,                                          \
        const J*             ind_,                                          \
        const T*             val_,                                          \
        rocsparse_index_base base_,                                         \
        rocsparse_diag_type  ldiag_,                                        \
        rocsparse_direction  ldir_,                                         \
        I*                   lnnz_,                                         \
        I*                   lptr_,                                         \
        rocsparse_index_base lbase_,                                        \
        rocsparse_diag_type  udiag_,                                        \
        rocsparse_direction  udir_,                                         \
        I*                   unnz_,                                         \
        I*                   uptr_,                                         \
        rocsparse_index_base ubase_,                                        \
        void*                buffer_)

INSTANTIATE(template, rocsparse_int, rocsparse_int, rocsparse_int);
INSTANTIATE(template, float, rocsparse_int, rocsparse_int);
INSTANTIATE(template, double, rocsparse_int, rocsparse_int);
INSTANTIATE(template, rocsparse_float_complex, rocsparse_int, rocsparse_int);
INSTANTIATE(template, rocsparse_double_complex, rocsparse_int, rocsparse_int);

#undef INSTANTIATE

#define C_IMPL(NAME, TYPE)                                                            \
    extern "C" rocsparse_status NAME(rocsparse_handle     handle_,                    \
                                     rocsparse_direction  dir_,                       \
                                     rocsparse_int        m_,                         \
                                     rocsparse_int        n_,                         \
                                     rocsparse_int        nnz_,                       \
                                     const rocsparse_int* ptr_,                       \
                                     const rocsparse_int* ind_,                       \
                                     const TYPE*          val_,                       \
                                     rocsparse_index_base base_,                      \
                                     rocsparse_diag_type  ldiag_type_,                \
                                     rocsparse_direction  ldir_,                      \
                                     rocsparse_int*       lnnz_,                      \
                                     rocsparse_int*       lptr_,                      \
                                     rocsparse_index_base lbase_,                     \
                                     rocsparse_diag_type  udiag_type_,                \
                                     rocsparse_direction  udir_,                      \
                                     rocsparse_int*       unnz_,                      \
                                     rocsparse_int*       uptr_,                      \
                                     rocsparse_index_base ubase_,                     \
                                     void*                buffer_)                    \
    try                                                                               \
    {                                                                                 \
        ROCSPARSE_ROUTINE_TRACE;                                                      \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csxsldu_preprocess_template(handle_,     \
                                                                         dir_,        \
                                                                         m_,          \
                                                                         n_,          \
                                                                         nnz_,        \
                                                                         ptr_,        \
                                                                         ind_,        \
                                                                         val_,        \
                                                                         base_,       \
                                                                         ldiag_type_, \
                                                                         ldir_,       \
                                                                         lnnz_,       \
                                                                         lptr_,       \
                                                                         lbase_,      \
                                                                         udiag_type_, \
                                                                         udir_,       \
                                                                         unnz_,       \
                                                                         uptr_,       \
                                                                         ubase_,      \
                                                                         buffer_));   \
        return rocsparse_status_success;                                              \
    }                                                                                 \
    catch(...)                                                                        \
    {                                                                                 \
        RETURN_ROCSPARSE_EXCEPTION();                                                 \
    }

C_IMPL(rocsparse_scsxsldu_preprocess, float);
C_IMPL(rocsparse_dcsxsldu_preprocess, double);
C_IMPL(rocsparse_ccsxsldu_preprocess, rocsparse_float_complex);
C_IMPL(rocsparse_zcsxsldu_preprocess, rocsparse_double_complex);

#undef C_IMPL
