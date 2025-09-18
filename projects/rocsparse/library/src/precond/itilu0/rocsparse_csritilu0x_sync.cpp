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
    void kernel_nrm_residual(const J m_,
                             const I nnz_,
                             const I* __restrict__ ptr_begin_,
                             const I* __restrict__ ptr_end_,
                             const J* __restrict__ ind_,
                             const T* __restrict__ val_,
                             const rocsparse_index_base base_,

                             const I* __restrict__ lptr_begin_,
                             const I* __restrict__ lptr_end_,
                             const J* __restrict__ lind_,
                             const T* __restrict__ lval0_,
                             const rocsparse_index_base lbase_,

                             const I* __restrict__ uptr_begin_,
                             const I* __restrict__ uptr_end_,
                             const J* __restrict__ uind_,
                             const T* __restrict__ uval0_,
                             const rocsparse_index_base ubase_,
                             const T* __restrict__ dval0_,
                             floating_data_t<T>* __restrict__ nrm_,
                             const floating_data_t<T>* __restrict__ nrm0_)
    {
        __shared__ floating_data_t<T> sdata[BLOCKSIZE / WFSIZE];
        floating_data_t<T>            nrm  = static_cast<floating_data_t<T>>(0);
        static constexpr uint32_t     nid  = BLOCKSIZE / WFSIZE;
        const J                       lid  = hipThreadIdx_x & (WFSIZE - 1);
        const J                       wid  = hipThreadIdx_x / WFSIZE;
        const J                       row0 = BLOCKSIZE * hipBlockIdx_x + wid;
        if(row0 < m_)
        {
            for(J row = row0; row < BLOCKSIZE * (hipBlockIdx_x + 1); row += nid)
            {
                if(row < m_)
                {
                    const J nl     = lptr_end_[row] - lptr_begin_[row];
                    const I lshift = lptr_begin_[row] - lbase_;
                    const I begin  = ((ptr_begin_[row] - base_) + lid);
                    const I end    = (ptr_end_[row] - base_);
                    for(I k = begin; k < end; k += WFSIZE)
                    {
                        const J col    = ind_[k] - base_;
                        const I ushift = uptr_begin_[col] - ubase_;
                        const J nu     = uptr_end_[col] - uptr_begin_[col];
                        J       i = 0, j = 0;
                        T       sum = rocsparse::sparse_dotproduct(nl,
                                                             lind_ + lshift,
                                                             lval0_ + lshift,
                                                             lbase_,
                                                             nu,
                                                             uind_ + ushift,
                                                             uval0_ + ushift,
                                                             ubase_,
                                                             i,
                                                             j);

                        if(j < nu)
                        {
                            for(J h = j; h < nu; ++h)
                            {
                                if((uind_[ushift + h] - ubase_) == row)
                                {
                                    sum += uval0_[ushift + h];
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
                                    sum += lval0_[lshift + h] * dval0_[col];
                                    break;
                                }
                            }
                        }

                        if(row == col)
                        {
                            sum += dval0_[col];
                        }

                        {
                            auto tmp = std::abs(val_[k] - sum);
                            if(!std::isinf(tmp) && !std::isnan(tmp))
                            {
                                nrm = (nrm > tmp) ? nrm : tmp;
                            }
                        }
                    }
                }
            }
        }

        rocsparse::wfreduce_max<WFSIZE>(&nrm);
        if(lid == (WFSIZE - 1))
            sdata[wid] = nrm;
        __syncthreads();
        rocsparse::blockreduce_max<BLOCKSIZE / WFSIZE>(hipThreadIdx_x, sdata);
        if(hipThreadIdx_x == 0)
        {
            rocsparse::atomic_max(nrm_, sdata[0] / nrm0_[0]);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              typename T,
              typename I,
              typename J,
              typename... P>
    static void
        kernel_nrm_residual_launch(dim3& blocks_, dim3& threads_, hipStream_t stream_, P... p)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::kernel_nrm_residual<BLOCKSIZE, WFSIZE, T, I, J>),
            blocks_,
            threads_,
            0,
            stream_,
            p...);
    }

    template <uint32_t BLOCKSIZE, typename T, typename I, typename J, typename... P>
    static void kernel_nrm_residual_dispatch(
        J m_, J mean_nnz_per_row_, int wavefront_size, hipStream_t stream_, P... p)
    {
        dim3 blocks((m_ - 1) / BLOCKSIZE + 1);
        dim3 threads(BLOCKSIZE);
        if(mean_nnz_per_row_ <= 2)
        {
            rocsparse::kernel_nrm_residual_launch<BLOCKSIZE, 1, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 4)
        {
            rocsparse::kernel_nrm_residual_launch<BLOCKSIZE, 2, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 8)
        {
            rocsparse::kernel_nrm_residual_launch<BLOCKSIZE, 4, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 16)
        {
            rocsparse::kernel_nrm_residual_launch<BLOCKSIZE, 8, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 32)
        {
            rocsparse::kernel_nrm_residual_launch<BLOCKSIZE, 16, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 64)
        {
            rocsparse::kernel_nrm_residual_launch<BLOCKSIZE, 32, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else
        {
            if(wavefront_size == 32)
            {
                rocsparse::kernel_nrm_residual_launch<BLOCKSIZE, 32, T, I, J>(
                    blocks, threads, stream_, p...);
            }
            else if(wavefront_size == 64)
            {
                rocsparse::kernel_nrm_residual_launch<BLOCKSIZE, 64, T, I, J>(
                    blocks, threads, stream_, p...);
            }
        }
    }

    template <int BLOCKSIZE, int WFSIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_correction(const J m_,
                           const I nnz_,
                           const I* __restrict__ ptr_begin_,
                           const I* __restrict__ ptr_end_,
                           const J* __restrict__ ind_,
                           const T* __restrict__ val_,
                           const rocsparse_index_base base_,

                           const I* __restrict__ lptr_begin_,
                           const I* __restrict__ lptr_end_,
                           const J* __restrict__ lind_,
                           const T* __restrict__ lval0_,
                           T* __restrict__ lval_,
                           const rocsparse_index_base lbase_,

                           const I* __restrict__ uptr_begin_,
                           const I* __restrict__ uptr_end_,
                           const J* __restrict__ uind_,
                           const T* __restrict__ uval0_,
                           T* __restrict__ uval_,
                           const rocsparse_index_base ubase_,
                           const T* __restrict__ dval0_,
                           T* __restrict__ dval_)
    {
        static constexpr uint32_t nid = BLOCKSIZE / WFSIZE;
        const J                   lid = hipThreadIdx_x & (WFSIZE - 1);
        const J                   wid = hipThreadIdx_x / WFSIZE;

        const J row0 = BLOCKSIZE * hipBlockIdx_x + wid;
        if(row0 < m_)
        {
            for(J row = row0; row < BLOCKSIZE * (hipBlockIdx_x + 1); row += nid)
            {
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
                            s /= dval0_[col];
                        }

                        //
                        // Assign.
                        //
                        auto ss = std::abs(s);
                        if(!std::isinf(ss) && !std::isnan(ss))
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
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              typename T,
              typename I,
              typename J,
              typename... P>
    static void kernel_correction_launch(dim3& blocks_, dim3& threads_, hipStream_t stream_, P... p)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::kernel_correction<BLOCKSIZE, WFSIZE, T, I, J>),
            blocks_,
            threads_,
            0,
            stream_,
            p...);
    }

    template <uint32_t BLOCKSIZE, typename T, typename I, typename J, typename... P>
    static void kernel_correction_dispatch(
        J m_, J mean_nnz_per_row_, int wavefront_size, hipStream_t stream_, P... p)
    {
        dim3 blocks((m_ - 1) / BLOCKSIZE + 1);
        dim3 threads(BLOCKSIZE);

        if(mean_nnz_per_row_ <= 2)
        {
            rocsparse::kernel_correction_launch<BLOCKSIZE, 1, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 4)
        {
            rocsparse::kernel_correction_launch<BLOCKSIZE, 2, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 8)
        {
            rocsparse::kernel_correction_launch<BLOCKSIZE, 4, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 16)
        {
            rocsparse::kernel_correction_launch<BLOCKSIZE, 8, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 32)
        {
            rocsparse::kernel_correction_launch<BLOCKSIZE, 16, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 64)
        {
            rocsparse::kernel_correction_launch<BLOCKSIZE, 32, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else
        {
            if(wavefront_size == 32)
            {
                rocsparse::kernel_correction_launch<BLOCKSIZE, 32, T, I, J>(
                    blocks, threads, stream_, p...);
            }
            else if(wavefront_size == 64)
            {
                rocsparse::kernel_correction_launch<BLOCKSIZE, 64, T, I, J>(
                    blocks, threads, stream_, p...);
            }
        }
    }
}

template <>
struct rocsparse::csritilu0x_driver_t<rocsparse_itilu0_alg_sync_split>
{
private:
    static constexpr int BLOCKSIZE = 1024;

    template <typename I>
    static constexpr I get_nblocks(I len_)
    {
        return (len_ - 1) / (BLOCKSIZE) + 1;
    }

public:
    template <typename T, typename J>
    struct history
    {
        static rocsparse_status run(rocsparse_handle handle_,
                                    J* __restrict__ niter_,
                                    T* __restrict__ data_,
                                    size_t buffer_size_,
                                    void* __restrict__ buffer_)
        {
            if(buffer_size_ == 0)
            {
                *niter_ = static_cast<J>(0);
                return rocsparse_status_success;
            }

            rocsparse::itilu0x_convergence_info_t<T, J> convergence_info;
            buffer_ = convergence_info.init(handle_, buffer_);
            J options;

            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&options,
                                               convergence_info.info.options,
                                               sizeof(J),
                                               hipMemcpyDeviceToHost,
                                               handle_->stream));

            RETURN_IF_HIP_ERROR(hipMemcpyAsync(niter_,
                                               convergence_info.info.iter,
                                               sizeof(J),
                                               hipMemcpyDeviceToHost,
                                               handle_->stream));

            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));

            J          niter = niter_[0];
            const bool convergence_history
                = (options & rocsparse_itilu0_option_convergence_history) > 0;
            if(!convergence_history)
            {
                std::cerr << "convergence history has not been activated." << std::endl;
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }

            const bool compute_nrm_residual
                = (options & rocsparse_itilu0_option_compute_nrm_residual) > 0;
            const bool compute_nrm_corr
                = (options & rocsparse_itilu0_option_compute_nrm_correction) > 0;

            if(compute_nrm_corr)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(data_,
                                                   convergence_info.log_mxcorr,
                                                   sizeof(T) * niter,
                                                   hipMemcpyDeviceToHost,
                                                   handle_->stream));
            }

            if(compute_nrm_residual)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(data_ + niter,
                                                   convergence_info.log_mxresidual,
                                                   sizeof(T) * niter,
                                                   hipMemcpyDeviceToHost,
                                                   handle_->stream));
            }

            //
            // No stream synchronization needed here,
            //
            return rocsparse_status_success;
        }
    };

    template <typename I, typename J>
    struct buffer_size
    {
        static rocsparse_status run(rocsparse_handle handle_,
                                    J                options_,
                                    J                nmaxiter_,
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
                    = rocsparse::itilu0x_convergence_info_t<float, J>::size(nmaxiter_, options_);
                break;
            }
            case rocsparse_datatype_f64_c:
            case rocsparse_datatype_f64_r:
            {
                size_convergence_info
                    = rocsparse::itilu0x_convergence_info_t<double, J>::size(nmaxiter_, options_);
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
                                    J                nmaxiter_,
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
        };
    };

    template <typename T, typename I, typename J>
    static rocsparse_status copy_unkwowns(rocsparse_handle handle_,
                                          I                lnnz_,
                                          T* __restrict__& lval0_,
                                          T* __restrict__& lval1_,
                                          I                unnz_,
                                          T* __restrict__& uval0_,
                                          T* __restrict__& uval1_,
                                          J                m_,
                                          T* __restrict__& dval0_,
                                          T* __restrict__& dval1_,
                                          hipMemcpyKind    mode_)
    {
        {
            T* tmp = lval0_;
            lval0_ = lval1_;
            lval1_ = tmp;
        }

        {
            T* tmp = uval0_;
            uval0_ = uval1_;
            uval1_ = tmp;
        }

        {
            T* tmp = dval0_;
            dval0_ = dval1_;
            dval1_ = tmp;
        }
        return rocsparse_status_success;
    }

    template <typename T, typename I, typename J>
    static rocsparse_status calculate_nrm_correction(rocsparse_handle          handle_,
                                                     J                         m_,
                                                     I                         nnz_,
                                                     I                         lnnz_,
                                                     I                         unnz_,
                                                     const T*                  lval0_,
                                                     const T*                  uval0_,
                                                     const T*                  dval0_,
                                                     const T*                  lval1_,
                                                     const T*                  uval1_,
                                                     const T*                  dval1_,
                                                     floating_data_t<T>*       nrm_,
                                                     const floating_data_t<T>* nrm_matrix_)

    {

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::nrminf_diff<BLOCKSIZE>(
            handle_, unnz_, uval0_, uval1_, nrm_, nrm_matrix_, false));

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::nrminf_diff<BLOCKSIZE>(
            handle_, lnnz_, lval0_, lval1_, nrm_, nrm_matrix_, true));

        if(dval0_ != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::nrminf_diff<BLOCKSIZE>(
                handle_, m_, dval0_, dval1_, nrm_, nrm_matrix_, true));
        }
        return rocsparse_status_success;
    }

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
            hipStream_t stream  = handle_->stream;
            const bool  verbose = (options_ & rocsparse_itilu0_option_verbose) > 0;
            const bool  compute_nrm_corr
                = (options_ & rocsparse_itilu0_option_compute_nrm_correction) > 0;
            const bool compute_nrm_residual
                = (options_ & rocsparse_itilu0_option_compute_nrm_residual) > 0;
            const bool stopping_criteria
                = (options_ & rocsparse_itilu0_option_stopping_criteria) > 0;
            const bool convergence_history
                = (options_ & rocsparse_itilu0_option_convergence_history) > 0;

            const I nmaxiter = nmaxiter_[0];

            void*   buffer = buffer_;
            const I mean   = rocsparse::max(nnz_ / m_, static_cast<I>(1));

            //
            // Initialize the convergence info.
            //
            rocsparse::itilu0x_convergence_info_t<floating_data_t<T>, J> setup;

            buffer = setup.init(handle_, buffer, nmaxiter, options_);

            floating_data_t<T>* p_nrm_matrix
                = (compute_nrm_residual || compute_nrm_corr) ? setup.info.nrm_matrix : nullptr;
            floating_data_t<T>* p_nrm_residual
                = (compute_nrm_residual) ? setup.info.nrm_residual : nullptr;
            floating_data_t<T>* p_nrm_corr = (compute_nrm_corr) ? setup.info.nrm_corr : nullptr;
            J*                  p_iter     = setup.info.iter;
            floating_data_t<T>* log_mxcorr = (compute_nrm_corr) ? setup.log_mxcorr : nullptr;
            floating_data_t<T>* log_mxresidual
                = (compute_nrm_residual) ? setup.log_mxresidual : nullptr;

            if(compute_nrm_residual || compute_nrm_corr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::nrminf<BLOCKSIZE>(
                    handle_, nnz_, val_, p_nrm_matrix, nullptr, false));
            }

            //
            // Init layout next.
            //

            rocsparse::itilu0x_layout_t<T, I, J> layout_x_next;
            layout_x_next.init(m_, ldiag_type_, lnnz_, udiag_type_, unnz_, buffer);

            //
            // Loop over.
            //
            floating_data_t<T> nrm_indicator_previous = static_cast<floating_data_t<T>>(0);
            bool               converged              = false;

            for(J iter = 0; iter < nmaxiter; ++iter)
            {
                for(J freeiter = 0; freeiter < nfreeiter_; ++freeiter)
                {
                    rocsparse::kernel_correction_dispatch<BLOCKSIZE, T, I, J>(
                        m_,
                        mean,
                        handle_->wavefront_size,
                        stream,
                        m_,
                        nnz_,
                        ptr_begin_,
                        ptr_end_,
                        ind_,
                        val_,
                        base_,
                        lptr_begin_,
                        lptr_end_,
                        lind_,
                        lval_,
                        layout_x_next.lval,
                        lbase_,
                        uptr_begin_,
                        uptr_end_,
                        uind_,
                        uval_,
                        layout_x_next.uval,
                        ubase_,
                        dval_,
                        layout_x_next.dval);
                    //
                    // Copy (before stopping criteria).
                    //
                    RETURN_IF_ROCSPARSE_ERROR(copy_unkwowns(handle_,
                                                            lnnz_,
                                                            layout_x_next.lval,
                                                            lval_,
                                                            unnz_,
                                                            layout_x_next.uval,
                                                            uval_,
                                                            m_,
                                                            layout_x_next.dval,
                                                            dval_,
                                                            hipMemcpyDeviceToDevice));
                }

                floating_data_t<T> nrm_corr     = static_cast<floating_data_t<T>>(0);
                floating_data_t<T> nrm_residual = static_cast<floating_data_t<T>>(0);

                if(compute_nrm_residual)
                {
                    //
                    // Compute norm of residual.
                    //
                    RETURN_IF_HIP_ERROR(
                        hipMemsetAsync(p_nrm_residual, 0, sizeof(floating_data_t<T>), stream));

                    rocsparse::kernel_nrm_residual_dispatch<BLOCKSIZE, T, I, J>(
                        m_,
                        mean,
                        handle_->wavefront_size,
                        stream,
                        m_,
                        nnz_,
                        ptr_begin_,
                        ptr_end_,
                        ind_,
                        val_,
                        base_,
                        lptr_begin_,
                        lptr_end_,
                        lind_,
                        lval_,
                        lbase_,
                        uptr_begin_,
                        uptr_end_,
                        uind_,
                        uval_,
                        ubase_,
                        dval_,
                        p_nrm_residual,
                        p_nrm_matrix);

                    if(convergence_history)
                    {
                        //
                        // Log convergence of residual.
                        //
                        RETURN_IF_HIP_ERROR(rocsparse::stay_on_device(
                            &log_mxresidual[iter], p_nrm_residual, stream));
                    }
                }

                //
                // CALCULATE CORRECTION.
                //
                rocsparse::kernel_correction_dispatch<BLOCKSIZE, T, I, J>(m_,
                                                                          mean,
                                                                          handle_->wavefront_size,
                                                                          stream,
                                                                          m_,
                                                                          nnz_,
                                                                          ptr_begin_,
                                                                          ptr_end_,
                                                                          ind_,
                                                                          val_,
                                                                          base_,
                                                                          lptr_begin_,
                                                                          lptr_end_,
                                                                          lind_,
                                                                          lval_,
                                                                          layout_x_next.lval,
                                                                          lbase_,
                                                                          uptr_begin_,
                                                                          uptr_end_,
                                                                          uind_,
                                                                          uval_,
                                                                          layout_x_next.uval,
                                                                          ubase_,
                                                                          dval_,
                                                                          layout_x_next.dval);
                if(compute_nrm_corr)
                {
                    //
                    // Calculate the norm of the correction.
                    //
                    RETURN_IF_ROCSPARSE_ERROR(calculate_nrm_correction(handle_,
                                                                       m_,
                                                                       nnz_,
                                                                       lnnz_,
                                                                       unnz_,
                                                                       lval_,
                                                                       uval_,
                                                                       dval_,
                                                                       layout_x_next.lval,
                                                                       layout_x_next.uval,
                                                                       layout_x_next.dval,
                                                                       p_nrm_corr,
                                                                       p_nrm_matrix));

                    if(convergence_history)
                    {
                        //
                        // Log the norm of the correction.
                        //
                        RETURN_IF_HIP_ERROR(
                            rocsparse::stay_on_device(&log_mxcorr[iter], p_nrm_corr, stream));
                    }
                }

                //
                // Copy (before stopping criteria).
                //
                RETURN_IF_ROCSPARSE_ERROR(copy_unkwowns(handle_,
                                                        lnnz_,
                                                        layout_x_next.lval,
                                                        lval_,
                                                        unnz_,
                                                        layout_x_next.uval,
                                                        uval_,
                                                        m_,
                                                        layout_x_next.dval,
                                                        dval_,
                                                        hipMemcpyDeviceToDevice));

                floating_data_t<T> nrm_indicator;
                if(stopping_criteria || verbose)
                {
                    if(compute_nrm_corr)
                    {
                        //
                        // EXTRACT NORM.
                        //
                        RETURN_IF_HIP_ERROR(rocsparse::on_host(&nrm_corr, p_nrm_corr, stream));
                    }

                    if(compute_nrm_residual)
                    {
                        //
                        // EXTRACT NORM.
                        //
                        RETURN_IF_HIP_ERROR(
                            rocsparse::on_host(&nrm_residual, p_nrm_residual, stream));
                    }

                    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
                }

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
                if(verbose)
                {
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
                              << (std::abs(nrm_indicator - nrm_indicator_previous) / nrm_indicator);
                    std::cout << std::endl;
                }

                if(stopping_criteria)
                {
                    if(std::isinf(nrm_indicator) || std::isnan(nrm_indicator))
                    {

                        nmaxiter_[0] = iter + 1;
                        converged    = false;

                        RETURN_IF_HIP_ERROR(rocsparse::on_device(p_iter, nmaxiter_, stream));
                        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
                        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
                    }
                    else
                    {
                        if((nrm_indicator <= tol_))
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

            RETURN_IF_HIP_ERROR(
                rocsparse::on_device(p_iter, (converged) ? nmaxiter_ : (&nmaxiter), stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
            return rocsparse_status_success;
        }
    };
};

#define INSTANTIATE(T, I, J)                        \
    template struct rocsparse::csritilu0x_driver_t< \
        rocsparse_itilu0_alg_sync_split>::compute<T, I, J>

INSTANTIATE(float, rocsparse_int, rocsparse_int);
INSTANTIATE(double, rocsparse_int, rocsparse_int);
INSTANTIATE(rocsparse_float_complex, rocsparse_int, rocsparse_int);
INSTANTIATE(rocsparse_double_complex, rocsparse_int, rocsparse_int);

#undef INSTANTIATE

#define INSTANTIATE(I, J)                                   \
    template struct rocsparse::csritilu0x_driver_t<         \
        rocsparse_itilu0_alg_sync_split>::preprocess<I, J>; \
    template struct rocsparse::csritilu0x_driver_t<         \
        rocsparse_itilu0_alg_sync_split>::buffer_size<I, J>

INSTANTIATE(rocsparse_int, rocsparse_int);

#undef INSTANTIATE

#define INSTANTIATE(T, J) \
    template struct rocsparse::csritilu0x_driver_t<rocsparse_itilu0_alg_sync_split>::history<T, J>

INSTANTIATE(float, rocsparse_int);
INSTANTIATE(double, rocsparse_int);

#undef INSTANTIATE
