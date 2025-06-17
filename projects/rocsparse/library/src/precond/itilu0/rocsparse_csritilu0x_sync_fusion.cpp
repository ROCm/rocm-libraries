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

#include "../conversion/rocsparse_csr2csc.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "common.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_csritilu0x_driver.hpp"
#include <iomanip>

namespace rocsparse
{
    template <int BLOCKSIZE, int WFSIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel(bool               stopping_criteria,
                bool               compute_nrm_corr,
                bool               compute_nrm_residual,
                const J            niter_,
                J*                 niter__,
                floating_data_t<T> tol_,
                const J            m_,
                const I            nnz_,
                const I* __restrict__ ptr_begin_,
                const I* __restrict__ ptr_end_,
                const J* __restrict__ ind_,
                const T* __restrict__ val_,
                const rocsparse_index_base base_,

                const I* __restrict__ lptr_begin_,
                const I* __restrict__ lptr_end_,
                const J* __restrict__ lind_,
                T* __restrict__ lval0_,
                T* __restrict__ lval_,
                const rocsparse_index_base lbase_,

                const I* __restrict__ uptr_begin_,
                const I* __restrict__ uptr_end_,
                const J* __restrict__ uind_,
                T* __restrict__ uval0_,
                T* __restrict__ uval_,
                const rocsparse_index_base ubase_,
                T* __restrict__ dval0_,
                T* __restrict__ dval_,
                floating_data_t<T>* __restrict__ nrms_corr,
                floating_data_t<T>* __restrict__ nrms_residual,
                const floating_data_t<T>* __restrict__ nrm0)
    {

        static constexpr uint32_t nid  = BLOCKSIZE / WFSIZE;
        const J                   lid  = hipThreadIdx_x & (WFSIZE - 1);
        const J                   wid  = hipThreadIdx_x / WFSIZE;
        const J                   row0 = BLOCKSIZE * hipBlockIdx_x + wid;
        J                         iter = 0;
        floating_data_t<T> nrminf = floating_data_t<T>(0), nrminf_residual = floating_data_t<T>(0);
        {
            __shared__ floating_data_t<T> sdata[BLOCKSIZE / WFSIZE];

            sdata[hipThreadIdx_x] = 0;
            __syncthreads();

            if(row0 < m_)
            {
                for(; iter < niter_; ++iter)
                {
                    if(compute_nrm_corr)
                    {
                        nrminf = static_cast<floating_data_t<T>>(0);
                    }
                    if(compute_nrm_residual)
                    {
                        nrminf_residual = static_cast<floating_data_t<T>>(0);
                    }

                    for(J l = 0; l < WFSIZE; ++l)
                    {
                        J row = row0 + nid * l;

                        if(row < m_)
                        {
                            const J nl     = lptr_end_[row] - lptr_begin_[row];
                            const I lshift = lptr_begin_[row] - lbase_;
                            const I begin  = ((ptr_begin_[row] - base_) + lid);
                            const I end    = (ptr_end_[row] - base_);

                            for(I k = begin; k < end; k += WFSIZE)
                            {
                                const J    col    = ind_[k] - base_;
                                const I    ushift = uptr_begin_[col] - ubase_;
                                const bool in_L   = (row > col);
                                const bool in_U   = (row < col);
                                const J    nu     = uptr_end_[col] - uptr_begin_[col];
                                J          i = 0, j = 0;
                                T          sum = rocsparse::sparse_dotproduct(nl,
                                                                     lind_ + lshift,
                                                                     lval0_ + lshift,
                                                                     lbase_,
                                                                     nu,
                                                                     uind_ + ushift,
                                                                     uval0_ + ushift,
                                                                     ubase_,
                                                                     i,
                                                                     j);

                                T s = val_[k] - sum;
                                if(in_L)
                                {
                                    if(std::abs(dval0_[col]) > 0)
                                    {
                                        s /= dval0_[col];
                                    }
                                    else
                                    {
                                        s = 0;
                                    }
                                }

                                //
                                // Assign.
                                //
                                floating_data_t<T> tmp = std::abs(s);
                                const bool assignable  = (!std::isinf(tmp) && !std::isnan(tmp));
                                if(assignable)
                                {
                                    if(in_L)
                                    {
                                        for(J h = i; h < nl; ++h)
                                        {
                                            if((lind_[lshift + h] - lbase_) == col)
                                            {
                                                lval_[lshift + h] = s;
                                                if(compute_nrm_corr)
                                                {
                                                    tmp = std::abs(lval0_[lshift + h] - s);
                                                    if(!std::isinf(tmp) && !std::isnan(tmp))
                                                    {
                                                        nrminf = rocsparse::max(nrminf, tmp);
                                                    }
                                                }
                                                break;
                                            }
                                        }
                                    }
                                    else if(in_U)
                                    {
                                        for(J h = j; h < nu; ++h)
                                        {
                                            if((uind_[ushift + h] - ubase_) == row)
                                            {
                                                uval_[ushift + h] = s;
                                                if(compute_nrm_corr)
                                                {
                                                    tmp = std::abs(uval0_[ushift + h] - s);
                                                    if(!std::isinf(tmp) && !std::isnan(tmp))
                                                    {
                                                        nrminf = rocsparse::max(nrminf, tmp);
                                                    }
                                                }
                                                break;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        dval_[col] = s;
                                        if(compute_nrm_corr)
                                        {
                                            tmp = std::abs(dval0_[col] - s);
                                            if(!std::isinf(tmp) && !std::isnan(tmp))
                                            {
                                                nrminf = rocsparse::max(nrminf, tmp);
                                            }
                                        }
                                    }
                                }

                                if(compute_nrm_residual)
                                {
                                    if(assignable)
                                    {
                                        T sum_residual = sum;
                                        if(j < nu)
                                        {
                                            for(J h = j; h < nu; ++h)
                                            {
                                                if((uind_[ushift + h] - ubase_) == row)
                                                {
                                                    sum_residual += uval0_[ushift + h];
                                                    break;
                                                }
                                            }
                                        }
                                        else if(i < nl)
                                        {
                                            for(J h = i; h < nl; ++h)
                                            {
                                                if((lind_[lshift + h] - lbase_) == col)
                                                {
                                                    sum_residual
                                                        += lval0_[lshift + h] * dval0_[col];
                                                    break;
                                                }
                                            }
                                        }
                                        if(row == col)
                                        {
                                            sum_residual += dval0_[col];
                                        }

                                        tmp = std::abs(val_[k] - sum_residual);
                                        if(!std::isinf(tmp) && !std::isnan(tmp))
                                        {
                                            nrminf_residual = rocsparse::max(nrminf_residual, tmp);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    //
                    // Finalize nrminf from shared memory.
                    //
                    if(compute_nrm_corr)
                    {
                        rocsparse::wfreduce_max<WFSIZE>(&nrminf);
                        if(lid == (WFSIZE - 1))
                        {
                            sdata[wid] = nrminf;
                        }
                        __syncthreads();

                        rocsparse::blockreduce_max<BLOCKSIZE / WFSIZE>(hipThreadIdx_x, sdata);
                        nrminf = sdata[0] / nrm0[0];
                    }

                    if(compute_nrm_residual)
                    {
                        rocsparse::wfreduce_max<WFSIZE>(&nrminf_residual);
                        if(lid == (WFSIZE - 1))
                        {
                            sdata[wid] = nrminf_residual;
                        }
                        __syncthreads();

                        rocsparse::blockreduce_max<BLOCKSIZE / WFSIZE>(hipThreadIdx_x, sdata);
                        nrminf_residual = sdata[0] / nrm0[0];
                    }

                    //
                    // COPY
                    //
                    for(J row = row0; row < BLOCKSIZE * (hipBlockIdx_x + 1); row += nid)
                    {
                        if(row < m_)
                        {
                            for(I ii = lptr_begin_[row] - lbase_ + lid;
                                ii < lptr_end_[row] - lbase_;
                                ii += WFSIZE)
                            {
                                lval0_[ii] = lval_[ii];
                            }
                            for(I ii = uptr_begin_[row] - ubase_ + lid;
                                ii < uptr_end_[row] - ubase_;
                                ii += WFSIZE)
                            {
                                uval0_[ii] = uval_[ii];
                            }
                            if(lid == 0)
                            {
                                dval0_[row] = dval_[row];
                            }
                        }
                    }

                    if(stopping_criteria)
                    {

                        const bool success = (compute_nrm_corr && compute_nrm_residual)
                                                 ? (nrminf <= tol_ && nrminf_residual <= tol_)
                                                 : ((compute_nrm_corr) ? (nrminf <= tol_)
                                                                       : (nrminf_residual <= tol_));
                        if(success)
                        {
                            break;
                        }
                    }
                }
            }
        }

        {
            __shared__ int sdata2[BLOCKSIZE / WFSIZE];

            //    sdata[hipThreadIdx_x] = 0;
            //    __syncthreads();
            //
            if(stopping_criteria)
            {
                rocsparse::wfreduce_max<WFSIZE>(&iter);
                if(lid == (WFSIZE - 1))
                {
                    sdata2[wid] = iter;
                }
                __syncthreads();

                rocsparse::blockreduce_max<BLOCKSIZE / WFSIZE>(hipThreadIdx_x, sdata2);
                iter = sdata2[0];
                if(hipThreadIdx_x == 0)
                {
                    rocsparse::atomic_max(niter__, iter + 1);
                }
            }
        }

        if(compute_nrm_corr)
        {
            if(hipThreadIdx_x == 0)
            {
                rocsparse::atomic_max(nrms_corr, nrminf);
            }
        }
        if(compute_nrm_residual)
        {
            if(hipThreadIdx_x == 0)
            {
                rocsparse::atomic_max(nrms_residual, nrminf_residual);
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              typename T,
              typename I,
              typename J,
              typename... P>
    static void kernel_launch(dim3& blocks_, dim3& threads_, hipStream_t stream_, P... p)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::kernel<BLOCKSIZE, WFSIZE, T, I, J>), blocks_, threads_, 0, stream_, p...);
    }

    template <uint32_t BLOCKSIZE, typename T, typename I, typename J, typename... P>
    static void
        kernel_dispatch(J m_, J mean_nnz_per_row_, int wavefront_size, hipStream_t stream_, P... p)
    {
        dim3 blocks((m_ - 1) / BLOCKSIZE + 1);
        dim3 threads(BLOCKSIZE);

        if(mean_nnz_per_row_ <= 2)
        {
            rocsparse::kernel_launch<BLOCKSIZE, 1, T, I, J>(blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 4)
        {
            rocsparse::kernel_launch<BLOCKSIZE, 2, T, I, J>(blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 8)
        {
            rocsparse::kernel_launch<BLOCKSIZE, 4, T, I, J>(blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 16)
        {
            rocsparse::kernel_launch<BLOCKSIZE, 8, T, I, J>(blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 32)
        {
            rocsparse::kernel_launch<BLOCKSIZE, 16, T, I, J>(blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 64)
        {
            rocsparse::kernel_launch<BLOCKSIZE, 32, T, I, J>(blocks, threads, stream_, p...);
        }
        else
        {
            if(wavefront_size == 32)
            {
                rocsparse::kernel_launch<BLOCKSIZE, 32, T, I, J>(blocks, threads, stream_, p...);
            }
            else if(wavefront_size == 64)
            {
                rocsparse::kernel_launch<BLOCKSIZE, 64, T, I, J>(blocks, threads, stream_, p...);
            }
        }
    }

    template <int BLOCKSIZE, int WFSIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_freerun(const J niter_,
                        const J m_,
                        const I nnz_,
                        const I* __restrict__ ptr_begin_,
                        const I* __restrict__ ptr_end_,
                        const J* __restrict__ ind_,
                        const T* __restrict__ val_,
                        const rocsparse_index_base base_,

                        const I* __restrict__ lptr_begin_,
                        const I* __restrict__ lptr_end_,
                        const J* __restrict__ lind_,
                        T* __restrict__ lval0_,
                        T* __restrict__ lval_,
                        const rocsparse_index_base lbase_,

                        const I* __restrict__ uptr_begin_,
                        const I* __restrict__ uptr_end_,
                        const J* __restrict__ uind_,
                        T* __restrict__ uval0_,
                        T* __restrict__ uval_,
                        const rocsparse_index_base ubase_,
                        T* __restrict__ dval0_,
                        T* __restrict__ dval_)
    {

        static constexpr uint32_t nid  = BLOCKSIZE / WFSIZE;
        const J                   lid  = hipThreadIdx_x & (WFSIZE - 1);
        const J                   wid  = hipThreadIdx_x / WFSIZE;
        const J                   row0 = BLOCKSIZE * hipBlockIdx_x + wid;
        J                         iter = 0;
        if(row0 < m_)
        {
            for(; iter < niter_; ++iter)
            {

                for(J l = 0; l < WFSIZE; ++l)
                {
                    J row = row0 + nid * l;

                    if(row < m_)
                    {
                        const J nl     = lptr_end_[row] - lptr_begin_[row];
                        const I lshift = lptr_begin_[row] - lbase_;
                        const I begin  = ((ptr_begin_[row] - base_) + lid);
                        const I end    = (ptr_end_[row] - base_);

                        for(I k = begin; k < end; k += WFSIZE)
                        {
                            const J    col    = ind_[k] - base_;
                            const I    ushift = uptr_begin_[col] - ubase_;
                            const bool in_L   = (row > col);
                            const bool in_U   = (row < col);
                            const J    nu     = uptr_end_[col] - uptr_begin_[col];
                            J          i = 0, j = 0;
                            T          sum = rocsparse::sparse_dotproduct(nl,
                                                                 lind_ + lshift,
                                                                 lval0_ + lshift,
                                                                 lbase_,
                                                                 nu,
                                                                 uind_ + ushift,
                                                                 uval0_ + ushift,
                                                                 ubase_,
                                                                 i,
                                                                 j);

                            T s = val_[k] - sum;
                            if(in_L)
                            {
                                if(std::abs(dval0_[col]) > 0)
                                {
                                    s /= dval0_[col];
                                }
                                else
                                {
                                    s = 0;
                                }
                            }

                            //
                            // Assign.
                            //
                            floating_data_t<T> tmp        = std::abs(s);
                            const bool         assignable = (!std::isinf(tmp) && !std::isnan(tmp));
                            if(assignable)
                            {
                                if(in_L)
                                {
                                    for(J h = i; h < nl; ++h)
                                    {
                                        if((lind_[lshift + h] - lbase_) == col)
                                        {
                                            lval_[lshift + h] = s;
                                            break;
                                        }
                                    }
                                }
                                else if(in_U)
                                {
                                    for(J h = j; h < nu; ++h)
                                    {
                                        if((uind_[ushift + h] - ubase_) == row)
                                        {
                                            uval_[ushift + h] = s;
                                            break;
                                        }
                                    }
                                }
                                else
                                {
                                    dval_[col] = s;
                                }
                            }
                        }
                    }
                }

                //
                // COPY
                //
                for(J row = row0; row < BLOCKSIZE * (hipBlockIdx_x + 1); row += nid)
                {
                    if(row < m_)
                    {
                        for(I ii = lptr_begin_[row] - lbase_ + lid; ii < lptr_end_[row] - lbase_;
                            ii += WFSIZE)
                        {
                            lval0_[ii] = lval_[ii];
                        }
                        for(I ii = uptr_begin_[row] - ubase_ + lid; ii < uptr_end_[row] - ubase_;
                            ii += WFSIZE)
                        {
                            uval0_[ii] = uval_[ii];
                        }
                        if(lid == 0)
                        {
                            dval0_[row] = dval_[row];
                        }
                    }
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              typename T,
              typename I,
              typename J,
              typename... P>
    static void kernel_freerun_launch(dim3& blocks_, dim3& threads_, hipStream_t stream_, P... p)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::kernel_freerun<BLOCKSIZE, WFSIZE, T, I, J>),
                                          blocks_,
                                          threads_,
                                          0,
                                          stream_,
                                          p...);
    }

    template <uint32_t BLOCKSIZE, typename T, typename I, typename J, typename... P>
    static void kernel_freerun_dispatch(
        J m_, J mean_nnz_per_row_, int wavefront_size, hipStream_t stream_, P... p)
    {
        dim3 blocks((m_ - 1) / BLOCKSIZE + 1);
        dim3 threads(BLOCKSIZE);

        if(mean_nnz_per_row_ <= 2)
        {
            rocsparse::kernel_freerun_launch<BLOCKSIZE, 1, T, I, J>(blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 4)
        {
            rocsparse::kernel_freerun_launch<BLOCKSIZE, 2, T, I, J>(blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 8)
        {
            rocsparse::kernel_freerun_launch<BLOCKSIZE, 4, T, I, J>(blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 16)
        {
            rocsparse::kernel_freerun_launch<BLOCKSIZE, 8, T, I, J>(blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 32)
        {
            rocsparse::kernel_freerun_launch<BLOCKSIZE, 16, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 64)
        {
            rocsparse::kernel_freerun_launch<BLOCKSIZE, 32, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else
        {
            if(wavefront_size == 32)
            {
                rocsparse::kernel_freerun_launch<BLOCKSIZE, 32, T, I, J>(
                    blocks, threads, stream_, p...);
            }
            else if(wavefront_size == 64)
            {
                rocsparse::kernel_freerun_launch<BLOCKSIZE, 64, T, I, J>(
                    blocks, threads, stream_, p...);
            }
        }
    }
}

template <>
struct rocsparse::csritilu0x_driver_t<rocsparse_itilu0_alg_sync_split_fusion>
{
    static constexpr int BLOCKSIZE = 1024;

    template <typename T, typename J>
    struct history
    {
        static rocsparse_status run(rocsparse_handle handle_,
                                    J* __restrict__ niter_,
                                    T* __restrict__ data_,
                                    size_t buffer_size_,
                                    void* __restrict__ buffer_)
        {
            RETURN_IF_ROCSPARSE_ERROR((
                rocsparse::csritilu0x_driver_t<rocsparse_itilu0_alg_sync_split>::history<T, J>::run(
                    handle_, niter_, data_, buffer_size_, buffer_)));
            return rocsparse_status_success;
        }
    };

    template <typename I, typename J>
    struct buffer_size
    {
        static rocsparse_status run(rocsparse_handle handle_,
                                    J                options_,
                                    J                nmaxiter,
                                    J                m_,
                                    I                nnz_,
                                    const I* __restrict__ ptr_begin_,
                                    const I* __restrict__ ptr_end_,
                                    const J* __restrict__ ind_,
                                    rocsparse_index_base base_,
                                    rocsparse_diag_type  ldiag_type_,
                                    rocsparse_direction  ldir_,
                                    rocsparse_diag_type  udiag_type_,
                                    rocsparse_direction  udir_,
                                    rocsparse_datatype   datatype_,
                                    size_t* __restrict__ buffer_size_)
        {
            // Quick return if possible
            if(m_ == 0)
            {
                *buffer_size_ = 0;
                return rocsparse_status_success;
            }

            if(nnz_ == 0)
            {
                *buffer_size_ = 0;
                return rocsparse_status_success;
            }

            //
            //
            // GET OPTIONS.
            //
            //
            size_t buffer_size = 0;
            size_t datasizeof  = 0;
            switch(datatype_)
            {
            case rocsparse_datatype_f32_r:
            {
                datasizeof = sizeof(float);
                break;
            }
            case rocsparse_datatype_f64_r:
            {
                datasizeof = sizeof(double);
                break;
            }
            case rocsparse_datatype_f32_c:
            {
                datasizeof = sizeof(rocsparse_float_complex);
                break;
            }
            case rocsparse_datatype_f64_c:
            {
                datasizeof = sizeof(rocsparse_double_complex);
                break;
            }
            case rocsparse_datatype_i8_r:
            case rocsparse_datatype_u8_r:
            case rocsparse_datatype_i32_r:
            case rocsparse_datatype_u32_r:
            case rocsparse_datatype_f16_r:
            case rocsparse_datatype_bf16_r:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }

            size_t size_convergence_info;
            switch(datatype_)
            {
            case rocsparse_datatype_f32_c:
            case rocsparse_datatype_f32_r:
            {
                size_convergence_info
                    = rocsparse::itilu0x_convergence_info_t<float, J>::size(nmaxiter, options_);
                break;
            }
            case rocsparse_datatype_f64_c:
            case rocsparse_datatype_f64_r:
            {
                size_convergence_info
                    = rocsparse::itilu0x_convergence_info_t<double, J>::size(nmaxiter, options_);
                break;
            }
            case rocsparse_datatype_i8_r:
            case rocsparse_datatype_u8_r:
            case rocsparse_datatype_i32_r:
            case rocsparse_datatype_u32_r:
            case rocsparse_datatype_f16_r:
            case rocsparse_datatype_bf16_r:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
            buffer_size += size_convergence_info;

            //
            // vector unknowns
            //
            buffer_size += datasizeof * nnz_;

            *buffer_size_ = buffer_size;
            return rocsparse_status_success;
        }
    };

    template <typename I, typename J>
    struct preprocess
    {

        static rocsparse_status run(rocsparse_handle handle_,
                                    J                options_,
                                    J                nmaxiter,
                                    J                m_,
                                    I                nnz_,
                                    const I* __restrict__ ptr_begin_,
                                    const I* __restrict__ ptr_end_,
                                    const J* __restrict__ ind_,
                                    rocsparse_index_base base_,
                                    rocsparse_diag_type  ldiag_type_,
                                    rocsparse_direction  ldir_,
                                    I                    lnnz_,
                                    const I* __restrict__ lptr_begin_,
                                    const I* __restrict__ lptr_end_,
                                    const J* __restrict__ lind_,
                                    rocsparse_index_base lbase_,
                                    rocsparse_diag_type  udiag_type_,
                                    rocsparse_direction  udir_,
                                    I                    unnz_,
                                    const I* __restrict__ uptr_begin_,
                                    const I* __restrict__ uptr_end_,
                                    const J* __restrict__ uind_,

                                    rocsparse_index_base ubase_,
                                    rocsparse_datatype   datatype_,
                                    size_t               buffer_size_,
                                    void* __restrict__ buffer_)
        {
            return rocsparse_status_success;
        }
    };

    template <typename T, typename I, typename J>
    struct compute
    {
        static rocsparse_status run(rocsparse_handle handle_,
                                    J                options_,
                                    J* __restrict__ nmaxiter_,
                                    J                  nfreeiter_,
                                    floating_data_t<T> tol_,
                                    J                  m_,
                                    I                  nnz_,
                                    const I* __restrict__ ptr_begin_,
                                    const I* __restrict__ ptr_end_,
                                    const J* __restrict__ ind_,
                                    const T* __restrict__ val_,
                                    rocsparse_index_base base_,
                                    rocsparse_diag_type  ldiag_type_,
                                    rocsparse_direction  ldir_,
                                    I                    lnnz_,
                                    const I* __restrict__ lptr_begin_,
                                    const I* __restrict__ lptr_end_,
                                    const J* __restrict__ lind_,
                                    T* __restrict__ lval_,
                                    rocsparse_index_base lbase_,
                                    rocsparse_diag_type  udiag_type_,
                                    rocsparse_direction  udir_,
                                    I                    unnz_,
                                    const I* __restrict__ uptr_begin_,
                                    const I* __restrict__ uptr_end_,
                                    const J* __restrict__ uind_,

                                    T* __restrict__ uval_,
                                    rocsparse_index_base ubase_,
                                    T* __restrict__ dval_,
                                    size_t buffer_size_,
                                    void* __restrict__ buffer_)
        {
            hipStream_t stream = handle_->stream;

            const bool stopping_criteria
                = (options_ & rocsparse_itilu0_option_stopping_criteria) > 0;
            const bool convergence_history
                = (options_ & rocsparse_itilu0_option_convergence_history) > 0;
            const bool compute_nrm_corr
                = (options_ & rocsparse_itilu0_option_compute_nrm_correction) > 0;
            const bool compute_nrm_residual
                = (options_ & rocsparse_itilu0_option_compute_nrm_residual) > 0;
            const bool verbose = (options_ & rocsparse_itilu0_option_verbose) > 0;

            const I nmaxiter      = nmaxiter_[0];
            void*   buffer        = buffer_;
            J       s_local_niter = 32;
            const I mean          = rocsparse::max(nnz_ / m_, static_cast<I>(1));

            //
            //
            //
            //
            // Initialize the convergence info.
            //

            rocsparse::itilu0x_convergence_info_t<floating_data_t<T>, J> convergence_info;
            buffer = convergence_info.init(handle_, buffer, nmaxiter, options_);

            floating_data_t<T>* p_nrm_matrix = (compute_nrm_residual || compute_nrm_corr)
                                                   ? convergence_info.info.nrm_matrix
                                                   : nullptr;
            floating_data_t<T>* p_nrm_residual
                = (compute_nrm_residual) ? convergence_info.info.nrm_residual : nullptr;
            floating_data_t<T>* p_nrm_corr
                = (compute_nrm_corr) ? convergence_info.info.nrm_corr : nullptr;
            J*                  p_iter       = convergence_info.info.iter;
            J*                  p_local_iter = convergence_info.info.local_iter;
            floating_data_t<T>* log_mxcorr
                = (compute_nrm_corr) ? convergence_info.log_mxcorr : nullptr;
            floating_data_t<T>* log_mxresidual
                = (compute_nrm_residual) ? convergence_info.log_mxresidual : nullptr;

            if(compute_nrm_residual || compute_nrm_corr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::nrminf<BLOCKSIZE>(
                    handle_, nnz_, val_, p_nrm_matrix, nullptr, false));
            }

            //
            // Init layout next.
            //
            rocsparse::itilu0x_layout_t<T, I, J> layout_next;
            layout_next.init(m_, ldiag_type_, lnnz_, udiag_type_, unnz_, buffer);

            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                layout_next.dval, dval_, sizeof(T) * m_, hipMemcpyDeviceToDevice, handle_->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(layout_next.uval,
                                               uval_,
                                               sizeof(T) * unnz_,
                                               hipMemcpyDeviceToDevice,
                                               handle_->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(layout_next.lval,
                                               lval_,
                                               sizeof(T) * lnnz_,
                                               hipMemcpyDeviceToDevice,
                                               handle_->stream));

            //
            // Loop over.
            //
            floating_data_t<T> nrm_indicator_previous = static_cast<floating_data_t<T>>(0);
            bool               converged              = false;
            for(J iter = 0; iter < nmaxiter; ++iter)
            {

                for(J freeiter = 0; freeiter < nfreeiter_; ++freeiter)
                {
                    rocsparse::kernel_freerun_dispatch<BLOCKSIZE, T, I, J>(m_,
                                                                           mean,
                                                                           handle_->wavefront_size,
                                                                           handle_->stream,
                                                                           1,
                                                                           //
                                                                           m_,
                                                                           nnz_,
                                                                           ptr_begin_,
                                                                           ptr_end_,
                                                                           ind_,
                                                                           val_,
                                                                           base_,
                                                                           //
                                                                           lptr_begin_,
                                                                           lptr_end_,
                                                                           lind_,
                                                                           lval_,
                                                                           layout_next.lval,
                                                                           lbase_,
                                                                           //
                                                                           uptr_begin_,
                                                                           uptr_end_,
                                                                           uind_,
                                                                           uval_,
                                                                           layout_next.uval,
                                                                           ubase_,
                                                                           //
                                                                           dval_,
                                                                           layout_next.dval);
                }

                if(!stopping_criteria && !compute_nrm_corr && !compute_nrm_residual)
                {
                    rocsparse::kernel_freerun_dispatch<BLOCKSIZE, T, I, J>(m_,
                                                                           mean,
                                                                           handle_->wavefront_size,
                                                                           handle_->stream,
                                                                           1,
                                                                           //
                                                                           m_,
                                                                           nnz_,
                                                                           ptr_begin_,
                                                                           ptr_end_,
                                                                           ind_,
                                                                           val_,
                                                                           base_,
                                                                           //
                                                                           lptr_begin_,
                                                                           lptr_end_,
                                                                           lind_,
                                                                           lval_,
                                                                           layout_next.lval,
                                                                           lbase_,
                                                                           //
                                                                           uptr_begin_,
                                                                           uptr_end_,
                                                                           uind_,
                                                                           uval_,
                                                                           layout_next.uval,
                                                                           ubase_,
                                                                           //
                                                                           dval_,
                                                                           layout_next.dval);
                }
                else
                {
                    if(p_nrm_corr != nullptr)
                    {
                        RETURN_IF_HIP_ERROR(hipMemsetAsync(
                            p_nrm_corr, 0, sizeof(floating_data_t<T>), handle_->stream));
                    }
                    if(p_nrm_residual != nullptr)
                    {
                        RETURN_IF_HIP_ERROR(hipMemsetAsync(
                            p_nrm_residual, 0, sizeof(floating_data_t<T>), handle_->stream));
                    }

                    RETURN_IF_HIP_ERROR(
                        hipMemsetAsync(p_local_iter, 0, sizeof(J), handle_->stream));

                    rocsparse::kernel_dispatch<BLOCKSIZE, T, I, J>(m_,
                                                                   mean,
                                                                   handle_->wavefront_size,
                                                                   handle_->stream,
                                                                   stopping_criteria,
                                                                   compute_nrm_corr,
                                                                   compute_nrm_residual,
                                                                   s_local_niter,
                                                                   p_local_iter,
                                                                   tol_,
                                                                   //
                                                                   m_,
                                                                   nnz_,
                                                                   ptr_begin_,
                                                                   ptr_end_,
                                                                   ind_,
                                                                   val_,
                                                                   base_,
                                                                   //
                                                                   lptr_begin_,
                                                                   lptr_end_,
                                                                   lind_,
                                                                   lval_,
                                                                   layout_next.lval,
                                                                   lbase_,
                                                                   //
                                                                   uptr_begin_,
                                                                   uptr_end_,
                                                                   uind_,
                                                                   uval_,
                                                                   layout_next.uval,
                                                                   ubase_,
                                                                   //
                                                                   dval_,
                                                                   layout_next.dval,
                                                                   p_nrm_corr,
                                                                   p_nrm_residual,
                                                                   p_nrm_matrix);

                    J local_niter = 0;
                    if(stopping_criteria || verbose)
                    {
                        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&local_niter,
                                                           p_local_iter,
                                                           sizeof(J),
                                                           hipMemcpyDeviceToHost,
                                                           handle_->stream));
                    }

                    //
                    //
                    //
                    if(compute_nrm_corr)
                    {
                        if(convergence_history)
                        {
                            RETURN_IF_HIP_ERROR(hipMemcpyAsync(log_mxcorr + iter,
                                                               p_nrm_corr,
                                                               sizeof(floating_data_t<T>),
                                                               hipMemcpyDeviceToDevice,
                                                               handle_->stream));
                        }
                    }

                    if(compute_nrm_residual)
                    {
                        if(convergence_history)
                        {
                            RETURN_IF_HIP_ERROR(hipMemcpyAsync(log_mxresidual + iter,
                                                               p_nrm_residual,
                                                               sizeof(floating_data_t<T>),
                                                               hipMemcpyDeviceToDevice,
                                                               handle_->stream));
                        }
                    }

                    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));
                    floating_data_t<T> nrm_corr     = static_cast<floating_data_t<T>>(0);
                    floating_data_t<T> nrm_residual = static_cast<floating_data_t<T>>(0);
                    floating_data_t<T> nrm_indicator;
                    if(stopping_criteria || verbose)
                    {
                        if(compute_nrm_corr)
                        {
                            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&nrm_corr,
                                                               p_nrm_corr,
                                                               sizeof(floating_data_t<T>),
                                                               hipMemcpyDeviceToHost,
                                                               handle_->stream));
                        }

                        if(compute_nrm_residual)
                        {
                            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&nrm_residual,
                                                               p_nrm_residual,
                                                               sizeof(floating_data_t<T>),
                                                               hipMemcpyDeviceToHost,
                                                               handle_->stream));
                        }

                        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));

                        if(compute_nrm_residual && compute_nrm_corr)
                        {
                            nrm_indicator = rocsparse::max(nrm_residual, nrm_corr);
                        }
                        else if(compute_nrm_corr)
                        {
                            nrm_indicator = nrm_corr;
                        }
                        else
                        {
                            nrm_indicator = nrm_residual;
                        }
                    }

                    if(verbose)
                    {
                        std::cout << std::setw(16) << "liter";
                        std::cout << std::setw(16) << "iter";
                        if(compute_nrm_corr)
                        {
                            std::cout << std::setw(16) << "corr";
                        }
                        if(compute_nrm_residual)
                        {
                            std::cout << std::setw(16) << "residual";
                        }
                        std::cout << std::setw(16) << "val";
                        std::cout << std::setw(16) << "rate";
                        std::cout << std::endl;

                        std::cout << std::setw(16) << local_niter;
                        std::cout << std::setw(16) << iter;
                        if(compute_nrm_corr)
                        {
                            std::cout << std::setw(16) << nrm_corr;
                        }
                        if(compute_nrm_residual)
                        {
                            std::cout << std::setw(16) << nrm_residual;
                        }
                        std::cout << std::setw(16) << nrm_indicator << std::setw(16)
                                  << (std::abs(nrm_indicator - nrm_indicator_previous)
                                      / nrm_indicator);
                        std::cout << std::endl;
                    }

                    if(stopping_criteria)
                    {
                        if(std::isinf(nrm_indicator) || std::isnan(nrm_indicator))
                        {

                            nmaxiter_[0] = iter + 1;
                            converged    = false;

                            RETURN_IF_HIP_ERROR(rocsparse::on_device(p_iter, nmaxiter_, stream));

                            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
                        }
                        else
                        {

                            if((nrm_indicator <= tol_) && (iter > 0))
                            {
                                nmaxiter_[0] = iter + 1;
                                converged    = true;
                                break;
                            }
                        }

                        {
                            //
                            // To avoid some stagnation.
                            // - if enough iteration
                            // - if rate is slowing down
                            // - if the correction is small enough
                            //
                            static constexpr floating_data_t<T> tol_increment
                                = (sizeof(floating_data_t<T>) == sizeof(float)) ? 1.0e-5 : 1.0e-15;
                            if((iter > 3)
                               && (std::abs(nrm_indicator - nrm_indicator_previous)
                                   <= tol_increment * nrm_indicator)
                               && nrm_indicator <= tol_ * 10)
                            {
                                nmaxiter_[0] = iter + 1;
                                converged    = true;
                                break;
                            }
                        }
                    }
                    nrm_indicator_previous = nrm_indicator;
                }
            }

            RETURN_IF_HIP_ERROR(
                rocsparse::on_device(p_iter, (converged) ? nmaxiter_ : (&nmaxiter), stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
            return rocsparse_status_success;
        }
    };
};

#define INSTANTIATE(T, I, J)                        \
    template struct rocsparse::csritilu0x_driver_t< \
        rocsparse_itilu0_alg_sync_split_fusion>::compute<T, I, J>

INSTANTIATE(float, rocsparse_int, rocsparse_int);
INSTANTIATE(double, rocsparse_int, rocsparse_int);
INSTANTIATE(rocsparse_float_complex, rocsparse_int, rocsparse_int);
INSTANTIATE(rocsparse_double_complex, rocsparse_int, rocsparse_int);

#undef INSTANTIATE

#define INSTANTIATE(I, J)                                          \
    template struct rocsparse::csritilu0x_driver_t<                \
        rocsparse_itilu0_alg_sync_split_fusion>::preprocess<I, J>; \
    template struct rocsparse::csritilu0x_driver_t<                \
        rocsparse_itilu0_alg_sync_split_fusion>::buffer_size<I, J>

INSTANTIATE(rocsparse_int, rocsparse_int);

#undef INSTANTIATE

#define INSTANTIATE(T, J)                           \
    template struct rocsparse::csritilu0x_driver_t< \
        rocsparse_itilu0_alg_sync_split_fusion>::history<T, J>

INSTANTIATE(float, rocsparse_int);
INSTANTIATE(double, rocsparse_int);

#undef INSTANTIATE
