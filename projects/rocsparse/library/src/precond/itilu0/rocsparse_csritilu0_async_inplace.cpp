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
#include "internal/conversion/rocsparse_coosort.h"
#include <iomanip>

#include "../conversion/rocsparse_coo2csr.hpp"
#include "../conversion/rocsparse_csr2coo.hpp"
#include "../conversion/rocsparse_csr2csc.hpp"
#include "../conversion/rocsparse_csxsldu.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "../level1/rocsparse_gthr.hpp"
#include "common.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_csritilu0_driver.hpp"
#include "rocsparse_csritilu0x_driver.hpp"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    template <typename I, typename J>
    static rocsparse_status inclusive_scan(rocsparse_handle handle, J m_, I* ptr_)
    {
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

    template <bool RESIDUAL, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void device_calculate(const J i,
                                               const J j,
                                               const T* __restrict__ x_,
                                               T* __restrict__ y_,

                                               const I* __restrict__ lptr_begin_,
                                               const I* __restrict__ lptr_end_,
                                               const J* __restrict__ lind_,

                                               const I* __restrict__ uptr_begin_,
                                               const I* __restrict__ uptr_end_,
                                               const J* __restrict__ uind_,
                                               const I* __restrict__ uperm_,
                                               const rocsparse_index_base base_,
                                               const T* __restrict__ ilu0_,
                                               floating_data_t<T>* __restrict__ nrm_)
    {
        T val = *x_;

        const I uend = uptr_end_[j] - base_;
        const I lend = lptr_end_[i] - base_;
        const T diag = ilu0_[lptr_end_[j] - base_];
        I       kx   = lptr_begin_[i] - base_;
        I       ky   = uptr_begin_[j] - base_;
        T       sum  = static_cast<T>(0);
        while((kx < lend) && (ky < uend))
        {
            const J jx = lind_[kx] - base_;
            const J jy = uind_[ky] - base_;
            if(jx == jy)
            {
                sum = (jx == jy) ? rocsparse::fma(ilu0_[kx], ilu0_[uperm_[ky]], sum) : sum;
            }
            kx = (jx <= jy) ? (kx + 1) : kx;
            ky = (jx >= jy) ? (ky + 1) : ky;
        }
        if(RESIDUAL)
        {
            auto tmp = val;
            val      = val - sum;
            if(i > j)
            {
                val /= diag;
            }
            sum = sum - tmp;
        }
        else
        {
            val = val - sum;
            if(i > j)
            {
                val /= diag;
            }
        }

        floating_data_t<T> absum;
        if(RESIDUAL)
        {
            for(; ky < uend; ++ky)
            {
                if((uind_[ky] - base_) == i)
                {
                    sum += ilu0_[uperm_[ky]];
                    break;
                }
            }
            for(; kx < lend; ++kx)
            {
                if((lind_[kx] - base_) == j)
                {
                    sum = rocsparse::fma(ilu0_[kx], diag, sum);
                    break;
                }
            }
            if(i == j)
            {
                sum += diag;
            }

            absum = std::abs(sum);
            if(!std::isinf(absum) && !std::isnan(absum))
            {
                *nrm_ = (*nrm_ > absum) ? *nrm_ : absum;
            }
        }

        absum = std::abs(val);
        if(!std::isinf(absum) && !std::isnan(absum))
        {
            *y_ = val;
        }
    }

    template <int BLOCKSIZE, int WFSIZE, bool RESIDUAL, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_calculate(const J m_,
                          const I nnz_,
                          const I* __restrict__ ptr_begin_,
                          const I* __restrict__ ptr_end_,
                          const J* __restrict__ ind_,
                          const T* __restrict__ val_,
                          const rocsparse_index_base base_,

                          const I* __restrict__ lptr_begin_,
                          const I* __restrict__ lptr_end_,
                          const J* __restrict__ lind_,

                          const I* __restrict__ uptr_begin_,
                          const I* __restrict__ uptr_end_,
                          const J* __restrict__ uind_,
                          const I* __restrict__ uperm_,
                          T* __restrict__ ilu0_,
                          floating_data_t<T>*       nrm_,
                          const floating_data_t<T>* nrm0_)
    {
        static constexpr uint32_t nid = BLOCKSIZE / WFSIZE;
        const J                   lid = hipThreadIdx_x & (WFSIZE - 1);
        const J                   wid = hipThreadIdx_x / WFSIZE;
        floating_data_t<T>        nrm = static_cast<floating_data_t<T>>(0);
        const J                   i0  = BLOCKSIZE * hipBlockIdx_x + wid;
        __shared__ floating_data_t<T> nrms[BLOCKSIZE / WFSIZE];
        if(i0 < m_)
        {
            for(int l = 0; l < WFSIZE; ++l)
            {
                const J i = i0 + nid * l;
                if(i < m_)
                {
                    const I end = (ptr_end_[i] - base_);
                    for(I k = ((ptr_begin_[i] - base_) + lid); k < end; k += WFSIZE)
                    {
                        const J j = ind_[k] - base_;
                        rocsparse::device_calculate<RESIDUAL>(i,
                                                              j,
                                                              val_ + k,
                                                              ilu0_ + k,

                                                              lptr_begin_,
                                                              lptr_end_,
                                                              lind_,

                                                              uptr_begin_,
                                                              uptr_end_,
                                                              uind_,
                                                              uperm_,
                                                              base_,
                                                              ilu0_,
                                                              &nrm);
                    }
                }
            }
        }

        //
        // REDUCE RESIDUAL.
        //
        if(RESIDUAL)
        {
            //
            // Current Warp reduce.
            //
            rocsparse::wfreduce_max<WFSIZE>(&nrm);
            if(lid == WFSIZE - 1)
            {
                nrms[wid] = nrm;
            }
            __syncthreads();

            //
            // Reduce over warps.
            //
            rocsparse::blockreduce_max<BLOCKSIZE / WFSIZE>(hipThreadIdx_x, nrms);

            //
            // Atomic to reduce over blocks.
            //
            if(hipThreadIdx_x == 0)
            {
                rocsparse::atomic_max(nrm_, nrms[0] / nrm0_[0]);
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              bool     RESIDUAL,
              typename T,
              typename I,
              typename J,
              typename... P>
    static void kernel_calculate_launch(dim3& blocks_, dim3& threads_, hipStream_t stream_, P... p)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::kernel_calculate<BLOCKSIZE, WFSIZE, RESIDUAL, T, I, J>),
            blocks_,
            threads_,
            0,
            stream_,
            p...);
    }

    template <uint32_t BLOCKSIZE, bool RESIDUAL, typename T, typename I, typename J, typename... P>
    static void kernel_calculate_dispatch(
        J m_, J mean_nnz_per_row_, int wavefront_size, hipStream_t stream_, P... p)
    {
        dim3 blocks((m_ - 1) / BLOCKSIZE + 1);
        dim3 threads(BLOCKSIZE);

        if(mean_nnz_per_row_ <= 2)
        {
            rocsparse::kernel_calculate_launch<BLOCKSIZE, 1, RESIDUAL, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 4)
        {
            rocsparse::kernel_calculate_launch<BLOCKSIZE, 2, RESIDUAL, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 8)
        {
            rocsparse::kernel_calculate_launch<BLOCKSIZE, 4, RESIDUAL, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 16)
        {
            rocsparse::kernel_calculate_launch<BLOCKSIZE, 8, RESIDUAL, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 32)
        {
            rocsparse::kernel_calculate_launch<BLOCKSIZE, 16, RESIDUAL, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else if(mean_nnz_per_row_ <= 64)
        {
            rocsparse::kernel_calculate_launch<BLOCKSIZE, 32, RESIDUAL, T, I, J>(
                blocks, threads, stream_, p...);
        }
        else
        {

            if(wavefront_size == 32)
            {
                rocsparse::kernel_calculate_launch<BLOCKSIZE, 32, RESIDUAL, T, I, J>(
                    blocks, threads, stream_, p...);
            }
            else if(wavefront_size == 64)
            {
                rocsparse::kernel_calculate_launch<BLOCKSIZE, 64, RESIDUAL, T, I, J>(
                    blocks, threads, stream_, p...);
            }
        }
    }

    template <int BLOCKSIZE, int WFSIZE, bool RESIDUAL, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_calculate_coo(const J m_,
                              const I nnz_,
                              const J* __restrict__ row_,
                              const J* __restrict__ col_,
                              const T* __restrict__ val_,
                              const rocsparse_index_base base_,

                              const I* __restrict__ lptr_begin_,
                              const I* __restrict__ lptr_end_,
                              const J* __restrict__ lind_,

                              const I* __restrict__ uptr_begin_,
                              const I* __restrict__ uptr_end_,
                              const J* __restrict__ uind_,
                              const I* __restrict__ uperm_,
                              T* __restrict__ ilu0_,
                              floating_data_t<T>*       nrm_,
                              const floating_data_t<T>* nrm0_)
    {
        static constexpr int num = 64;

        const J    lid = hipThreadIdx_x & (WFSIZE - 1);
        const J    wid = hipThreadIdx_x / WFSIZE;
        const I    tid = (hipBlockIdx_x * (BLOCKSIZE * num)) + hipThreadIdx_x;
        __shared__ floating_data_t<T> nrms[BLOCKSIZE / WFSIZE];
        floating_data_t<T>            nrm = 0;
        if(tid < nnz_)
        {
            I id = tid;
            for(int i = 0; i < num; ++i)
            {
                if(id < nnz_)
                {
                    rocsparse::device_calculate<RESIDUAL>(row_[id] - base_,
                                                          col_[id] - base_,
                                                          val_ + id,
                                                          ilu0_ + id,

                                                          lptr_begin_,
                                                          lptr_end_,
                                                          lind_,

                                                          uptr_begin_,
                                                          uptr_end_,
                                                          uind_,
                                                          uperm_,
                                                          base_,
                                                          ilu0_,
                                                          &nrm);
                }
                id += BLOCKSIZE;
            }
        }

        if(RESIDUAL)
        {
            rocsparse::wfreduce_max<WFSIZE>(&nrm);
            if(lid == WFSIZE - 1)
            {
                nrms[wid] = nrm;
            }
            __syncthreads();
            rocsparse::blockreduce_max<BLOCKSIZE / WFSIZE>(hipThreadIdx_x, nrms);
            if(hipThreadIdx_x == 0)
            {
                rocsparse::atomic_max(nrm_, nrms[0] / nrm0_[0]);
            }
        }
    }
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              bool     RESIDUAL,
              typename T,
              typename I,
              typename J,
              typename... P>
    static inline void
        kernel_calculate_coo_launch(dim3& blocks_, dim3& threads_, hipStream_t stream_, P... p)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::kernel_calculate_coo<BLOCKSIZE, WFSIZE, RESIDUAL, T, I, J>),
            blocks_,
            threads_,
            0,
            stream_,
            p...);
    }

    template <uint32_t BLOCKSIZE, bool RESIDUAL, typename T, typename I, typename J, typename... P>
    static inline void
        kernel_calculate_coo_dispatch(J nnz_, int wavefront_size, hipStream_t stream_, P... p)
    {
        static constexpr int num = 64;
        dim3                 blocks((nnz_ - 1) / (BLOCKSIZE * num) + 1);
        dim3                 threads(BLOCKSIZE);
        if(wavefront_size == 32)
            rocsparse::kernel_calculate_coo_launch<BLOCKSIZE, 32, RESIDUAL, T, I, J>(
                blocks, threads, stream_, p...);
        else if(wavefront_size == 64)
            rocsparse::kernel_calculate_coo_launch<BLOCKSIZE, 64, RESIDUAL, T, I, J>(
                blocks, threads, stream_, p...);
    }

    template <uint32_t BLOCKSIZE, typename T, typename I, typename J>
    struct compute_iter
    {
        static rocsparse_status light_run(rocsparse_handle handle_,
                                          J                options_,
                                          J                nmaxiter_,
                                          J                m_,
                                          I                nnz_,
                                          const I* __restrict__ ptr_begin_,
                                          const I* __restrict__ ptr_end_,
                                          const J* __restrict__ row_ind_,
                                          const J* __restrict__ ind_,
                                          const T* __restrict__ val_,
                                          rocsparse_index_base base_,

                                          const I* __restrict__ lptr_begin_,
                                          const I* __restrict__ lptr_end_,
                                          const J* __restrict__ lind_,
                                          const I* __restrict__ uptr_begin_,
                                          const I* __restrict__ uptr_end_,
                                          const J* __restrict__ uind_,
                                          const I* __restrict__ uperm_,
                                          T* __restrict__ ilu0_)
        {
            hipStream_t stream         = handle_->stream;
            const bool  use_coo_format = (options_ & rocsparse_itilu0_option_coo_format) > 0;

            const I mean = rocsparse::max(nnz_ / m_, static_cast<I>(1));

            if(!use_coo_format)
            {
                for(J iter = 0; iter < nmaxiter_; ++iter)
                {
                    rocsparse::kernel_calculate_dispatch<BLOCKSIZE, false, T, I, J>(
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

                        uptr_begin_,
                        uptr_end_,
                        uind_,
                        uperm_,
                        ilu0_,
                        nullptr,
                        nullptr);
                }
            }
            else
            {
                for(J iter = 0; iter < nmaxiter_; ++iter)
                {
                    rocsparse::kernel_calculate_coo_dispatch<BLOCKSIZE, false, T, I, J>(
                        nnz_,
                        handle_->wavefront_size,
                        stream,

                        m_,
                        nnz_,
                        row_ind_,
                        ind_,
                        val_,
                        base_,

                        lptr_begin_,
                        lptr_end_,
                        lind_,

                        uptr_begin_,
                        uptr_end_,
                        uind_,
                        uperm_,
                        ilu0_,
                        nullptr,
                        nullptr);
                }
            }
            return rocsparse_status_success;
        }

        static rocsparse_status run(rocsparse_handle handle_,
                                    J                options_,
                                    J* __restrict__ nmaxiter_,
                                    J                  nfreeiter_,
                                    floating_data_t<T> tol_,
                                    J                  m_,
                                    I                  nnz_,
                                    const I* __restrict__ ptr_begin_,
                                    const I* __restrict__ ptr_end_,
                                    const J* __restrict__ row_ind_,
                                    const J* __restrict__ ind_,
                                    const T* __restrict__ val_,
                                    rocsparse_index_base base_,

                                    const I* __restrict__ lptr_begin_,
                                    const I* __restrict__ lptr_end_,
                                    const J* __restrict__ lind_,

                                    const I* __restrict__ uptr_begin_,
                                    const I* __restrict__ uptr_end_,
                                    const J* __restrict__ uind_,
                                    const I* __restrict__ uperm_,

                                    T* __restrict__ ilu0_,
                                    size_t buffer_size_,
                                    void* __restrict__ buffer_)
        {
            hipStream_t stream  = handle_->stream;
            const bool  verbose = (options_ & rocsparse_itilu0_option_verbose) > 0;
            const bool  stopping_criteria
                = (options_ & rocsparse_itilu0_option_stopping_criteria) > 0;
            const bool convergence_history
                = (options_ & rocsparse_itilu0_option_convergence_history) > 0;
            const bool use_coo_format = (options_ & rocsparse_itilu0_option_coo_format) > 0;

            const I nmaxiter = nmaxiter_[0];

            void* buffer = buffer_;

            const I mean = rocsparse::max(nnz_ / m_, static_cast<I>(1));

            //
            // Initialize the convergence info.
            //

            rocsparse::itilu0x_convergence_info_t<floating_data_t<T>, J> setup;
            buffer = setup.init(handle_, buffer, nmaxiter, options_);

            floating_data_t<T>* p_nrm_matrix   = setup.info.nrm_matrix;
            floating_data_t<T>* p_nrm_residual = setup.info.nrm_residual;
            J*                  p_iter         = setup.info.iter;
            floating_data_t<T>* log_mxresidual = setup.log_mxresidual;

            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::nrminf<BLOCKSIZE>(handle_, nnz_, val_, p_nrm_matrix, nullptr, false));
            //
            // Loop over.
            //
            floating_data_t<T> nrm_residual_previous = static_cast<floating_data_t<T>>(0);
            bool               converged             = false;
            for(J iter = 0; iter < nmaxiter; ++iter)
            {

                if(nfreeiter_ > 0)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (compute_iter<BLOCKSIZE, T, I, J>::light_run)(handle_,
                                                                      options_,
                                                                      nfreeiter_,
                                                                      m_,
                                                                      nnz_,
                                                                      ptr_begin_,
                                                                      ptr_end_,
                                                                      row_ind_,
                                                                      ind_,
                                                                      val_,
                                                                      base_,
                                                                      lptr_begin_,
                                                                      lptr_end_,
                                                                      lind_,

                                                                      uptr_begin_,
                                                                      uptr_end_,
                                                                      uind_,
                                                                      uperm_,
                                                                      ilu0_));
                }

                //
                // Need to set to zero because of atomics.
                // (And absolutely need to be aligned).
                //
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync(p_nrm_residual, 0, sizeof(floating_data_t<T>), handle_->stream));

                if(use_coo_format)
                {
                    //
                    // Compute an iteration of the factorization.
                    //
                    rocsparse::kernel_calculate_coo_dispatch<BLOCKSIZE, true, T, I, J>(
                        nnz_,
                        handle_->wavefront_size,
                        handle_->stream,

                        m_,
                        nnz_,
                        row_ind_,
                        ind_,
                        val_,
                        base_,

                        lptr_begin_,
                        lptr_end_,
                        lind_,

                        uptr_begin_,
                        uptr_end_,
                        uind_,
                        uperm_,

                        ilu0_,
                        p_nrm_residual,
                        p_nrm_matrix);
                }
                else
                {
                    rocsparse::kernel_calculate_dispatch<BLOCKSIZE, true, T, I, J>(
                        m_,
                        mean,
                        handle_->wavefront_size,
                        handle_->stream,

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

                        uptr_begin_,
                        uptr_end_,
                        uind_,
                        uperm_,

                        ilu0_,
                        p_nrm_residual,
                        p_nrm_matrix);
                }
                if(convergence_history)
                {
                    //
                    // Log history of residual.
                    //
                    RETURN_IF_HIP_ERROR(
                        rocsparse::stay_on_device(&log_mxresidual[iter], p_nrm_residual, stream));
                }

                floating_data_t<T> nrm_residual;
                if(stopping_criteria)
                {
                    RETURN_IF_HIP_ERROR(rocsparse::on_host(&nrm_residual, p_nrm_residual, stream));
                    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
                }

                //
                // INFO
                //
                if(verbose)
                {
                    std::cout << std::setw(16) << "iter";
                    std::cout << std::setw(16) << "residual" << std::setw(16) << "rate";
                    std::cout << std::endl;
                    std::cout << std::setw(16) << iter;
                    std::cout << std::setw(16) << nrm_residual << std::setw(16)
                              << std::abs(nrm_residual - nrm_residual_previous) / nrm_residual;
                    std::cout << std::endl;
                }

                if(stopping_criteria)
                {

                    if(std::isinf(nrm_residual))
                    {

                        nmaxiter_[0] = iter + 1;
                        converged    = false;

                        RETURN_IF_HIP_ERROR(rocsparse::on_device(p_iter, nmaxiter_, stream));

                        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
                    }
                    else
                    {
                        if((nrm_residual <= tol_))
                        {
                            nmaxiter_[0] = iter + 1;
                            converged    = true;
                            break;
                        }
                    }

                    //
                    // To avoid some stagnation.
                    // - if enough iteration
                    // - if rate is slowing down
                    // - if the correction is small enough
                    //
                    static constexpr floating_data_t<T> tol_increment
                        = (sizeof(floating_data_t<T>) == sizeof(float)) ? 1.0e-5 : 1.0e-15;

                    if((iter > 3)
                       && (std::abs(nrm_residual - nrm_residual_previous)
                           < tol_increment * nrm_residual)
                       && nrm_residual < 1.0e-5)
                    {
                        nmaxiter_[0] = iter + 1;
                        converged    = true;
                        break;
                    }
                }

                nrm_residual_previous = nrm_residual;
            }
            RETURN_IF_HIP_ERROR(
                rocsparse::on_device(p_iter, (converged) ? nmaxiter_ : (&nmaxiter), stream));

            return rocsparse_status_success;
        }
    };

    struct buffer_layout_inplace_t
    {
    public:
        static size_t get_sizeof_double()
        {
            return ((sizeof(buffer_layout_inplace_t) - 1) / sizeof(double) + 1);
        }

        typedef enum enum_ivalue_type
        {
            lptr_end,
            uperm,
            uptr
        } ivalue_type;

        typedef enum enum_jvalue_type
        {
            uind,
            coo_row_ind,
        } jvalue_type;

        typedef enum enum_tvalue_type
        {
            x,
            buffer
        } tvalue_type;

    public:
        size_t get_size(ivalue_type i)
        {
            return m_isizes[i];
        }
        size_t get_size(jvalue_type i)
        {
            return m_jsizes[i];
        }

        size_t get_size(tvalue_type i)
        {
            return m_tsizes[i];
        }

        void* get_pointer(ivalue_type i)
        {
            return m_ipointers[i];
        }

        void* get_pointer(jvalue_type j)
        {
            return m_jpointers[j];
        }

        void* get_pointer(tvalue_type i)
        {
            return m_tpointers[i];
        }

    protected:
        void set(tvalue_type         v,
                 size_t&             buffer_size_,
                 void* __restrict__& buffer_,
                 size_t              nitems_,
                 size_t              sizelm_)
        {

            m_tpointers[v]
                = (void*)rocsparse::assign_b<char>(buffer_size_, buffer_, nitems_ * sizelm_);
            m_tsizes[v] = sizelm_ * nitems_;
        }

        template <typename J>
        void set(jvalue_type v, size_t& buffer_size_, void* __restrict__& buffer_, size_t nitems_)
        {
            m_jpointers[v]
                = (J*)rocsparse::assign_b<char>(buffer_size_, buffer_, align_size<J>(nitems_));

            m_jsizes[v] = sizeof(J) * nitems_;
        }

        template <typename I>
        void set(ivalue_type v, size_t& buffer_size_, void* __restrict__& buffer_, size_t nitems_)
        {
            m_ipointers[v]
                = (I*)rocsparse::assign_b<char>(buffer_size_, buffer_, align_size<I>(nitems_));
            m_isizes[v] = sizeof(I) * nitems_;
        }
        template <typename T>
        static size_t align_size(size_t nelms_)
        {
            return ((sizeof(T) * nelms_ + sizeof(double) - 1) / sizeof(double)) * sizeof(double);
        }

    public:
        template <typename I, typename J>
        static void buffer_size(J m_, I nnz_, I unnz_, size_t& buffer_size_, bool csrcoo_)
        {
            buffer_size_ += align_size<I>(m_); // lptr_end
            buffer_size_ += align_size<I>(m_ + 1); // uptr
            buffer_size_ += align_size<I>(unnz_); // uperm
            buffer_size_ += align_size<J>(unnz_); // uind
            if(csrcoo_)
            {
                buffer_size_ += align_size<J>(nnz_);
            }

            const size_t sizeof_double = get_sizeof_double();
            buffer_size_ += sizeof_double * sizeof(double);
        }

        template <typename I, typename J>
        void init(J                   m_,
                  I                   nnz_,
                  I                   unnz_,
                  rocsparse_datatype  datatype_,
                  size_t&             buffer_size_,
                  void* __restrict__& buffer_,
                  bool                csrcoo_)
        {
            const size_t parent_sizeof_double = get_sizeof_double();
            m_buffer_size = buffer_size_ - parent_sizeof_double * sizeof(double);
            m_buffer      = (void*)(((double*)buffer_) + parent_sizeof_double);
            buffer_size_  = m_buffer_size;
            buffer_       = m_buffer;
            if(csrcoo_)
            {
                this->set<J>(coo_row_ind, buffer_size_, buffer_, nnz_);
            }
            this->set<I>(lptr_end, buffer_size_, buffer_, m_);
            this->set<I>(uptr, buffer_size_, buffer_, m_ + 1);
            this->set<J>(uind, buffer_size_, buffer_, unnz_);
            this->set<I>(uperm, buffer_size_, buffer_, unnz_);
            m_tpointers[buffer] = buffer_;
            m_tsizes[buffer]    = buffer_size_;
        }

        buffer_layout_inplace_t(){};

    private:
        void*  m_buffer{};
        size_t m_buffer_size{};
        size_t m_isizes[3]{};
        size_t m_jsizes[1]{};
        size_t m_tsizes[2]{};
        void*  m_ipointers[3]{};
        void*  m_jpointers[1]{};
        void*  m_tpointers[2]{};
    };

    //
    // Calculate the array lptr_end.
    //
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_compute_lptr_end(J m_,
                                 const I* __restrict__ ptr_begin_,
                                 const I* __restrict__ ptr_end_,
                                 const J* __restrict__ ind_,
                                 rocsparse_index_base base_,
                                 I* __restrict__ lptr_end_)
    {
        const I i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        if(i < m_)
        {
            const I upper_b = ptr_end_[i] - base_;
            I       b       = upper_b;
            for(I k = ptr_begin_[i] - base_; k < upper_b; ++k)
            {
                if((ind_[k] - base_) >= i)
                {
                    b = k;
                    break;
                }
            }
            lptr_end_[i] = b + base_;
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J, typename... P>
    static void
        kernel_compute_lptr_end_launch(dim3& blocks_, dim3& threads_, hipStream_t stream_, P... p)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::kernel_compute_lptr_end<BLOCKSIZE, WFSIZE, I, J>),
            blocks_,
            threads_,
            0,
            stream_,
            p...);
    }

    template <uint32_t BLOCKSIZE, typename I, typename J, typename... P>
    static void kernel_compute_lptr_end_dispatch(J           target_size_,
                                                 int         wavefront_size,
                                                 hipStream_t stream_,
                                                 P... p)
    {
        dim3 blocks((target_size_ + BLOCKSIZE - 1) / BLOCKSIZE);
        dim3 threads(BLOCKSIZE);
        if(wavefront_size == 32)
        {
            rocsparse::kernel_compute_lptr_end_launch<BLOCKSIZE, 32, I, J>(
                blocks, threads, stream_, p...);
        }
        else
        {
            rocsparse::kernel_compute_lptr_end_launch<BLOCKSIZE, 64, I, J>(
                blocks, threads, stream_, p...);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_compute_coo(J m_,
                            const I* __restrict__ ptr_begin_,
                            const I* __restrict__ ptr_end_,
                            const J* __restrict__ ind_,
                            rocsparse_index_base base_,
                            const I* __restrict__ coo_ptr_,
                            J* __restrict__ coo_row_ind_,
                            J* __restrict__ coo_col_ind_,
                            I* __restrict__ coo_perm_)
    {
        const I i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        if(i < m_)
        {
            const I begin = ptr_begin_[i] - base_;
            const I end   = ptr_end_[i] - base_;
            const I at    = coo_ptr_[i] - base_;
            for(I k = 0; k < (end - begin - 1); ++k)
            {
                const J j            = ind_[begin + 1 + k] - base_;
                coo_row_ind_[at + k] = i + base_;
                coo_col_ind_[at + k] = j + base_;
                coo_perm_[at + k]    = begin + 1 + k;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J, typename... P>
    static void
        kernel_compute_coo_launch(dim3& blocks_, dim3& threads_, hipStream_t stream_, P... p)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::kernel_compute_coo<BLOCKSIZE, WFSIZE, I, J>),
                                          blocks_,
                                          threads_,
                                          0,
                                          stream_,
                                          p...);
    }

    template <uint32_t BLOCKSIZE, typename I, typename J, typename... P>
    static void
        kernel_compute_coo_dispatch(J target_size_, int wavefront_size, hipStream_t stream_, P... p)
    {
        dim3 blocks((target_size_ + BLOCKSIZE - 1) / BLOCKSIZE);
        dim3 threads(BLOCKSIZE);
        if(wavefront_size == 32)
        {
            rocsparse::kernel_compute_coo_launch<BLOCKSIZE, 32, I, J>(
                blocks, threads, stream_, p...);
        }
        else
        {
            rocsparse::kernel_compute_coo_launch<BLOCKSIZE, 64, I, J>(
                blocks, threads, stream_, p...);
        }
    }

    //
    // Calculate the array ucsr_ptr.
    //
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_initialize_ucsr_ptr(J m_,
                                    const I* __restrict__ ptr_begin_,
                                    const I* __restrict__ ptr_end_,
                                    I* __restrict__ ucsr_ptr_,
                                    rocsparse_index_base base_)
    {
        const I i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        if(i < m_)
        {
            if(i == 0)
            {
                ucsr_ptr_[0] = base_;
            }
            ucsr_ptr_[i + 1] = ptr_end_[i] - ptr_begin_[i] - 1;
            //
            // -1 is for the diagonal element
            //
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J, typename... P>
    static void kernel_initialize_ucsr_ptr_launch(dim3&       blocks_,
                                                  dim3&       threads_,
                                                  hipStream_t stream_,
                                                  P... p)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::kernel_initialize_ucsr_ptr<BLOCKSIZE, WFSIZE, I, J>),
            blocks_,
            threads_,
            0,
            stream_,
            p...);
    }

    template <uint32_t BLOCKSIZE, typename I, typename J, typename... P>
    static void kernel_initialize_ucsr_ptr_dispatch(J           target_size_,
                                                    int         wavefront_size,
                                                    hipStream_t stream_,
                                                    P... p)
    {
        dim3 blocks((target_size_ + BLOCKSIZE - 1) / BLOCKSIZE);
        dim3 threads(BLOCKSIZE);
        if(wavefront_size == 32)
        {
            rocsparse::kernel_initialize_ucsr_ptr_launch<BLOCKSIZE, 32, I, J>(
                blocks, threads, stream_, p...);
        }
        else
        {
            rocsparse::kernel_initialize_ucsr_ptr_launch<BLOCKSIZE, 64, I, J>(
                blocks, threads, stream_, p...);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_compute_unnz(J m_,
                             const I* __restrict__ ptr_begin_,
                             const I* __restrict__ ptr_end_,
                             const J* __restrict__ ind_,
                             rocsparse_index_base base_,
                             I* __restrict__ nnz_,
                             I* __restrict__ nnz_diag_)
    {
        __shared__ I data[BLOCKSIZE];
        const I      i        = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        const bool   valid    = (i < m_);
        I            nnz      = 0;
        I            nnz_diag = 0;
        if(valid)
        {
            for(I k = ptr_begin_[i] - base_; k < ptr_end_[i] - base_; ++k)
            {
                const J j = ind_[k] - base_;
                if(j > i)
                {
                    ++nnz;
                }
                else if(nnz_diag_ != nullptr && j == i)
                {
                    ++nnz_diag;
                }
            }
        }
        data[hipThreadIdx_x] = nnz;
        __syncthreads();

        rocsparse::blockreduce_sum<BLOCKSIZE>(hipThreadIdx_x, data);
        if(hipThreadIdx_x == 0)
        {
            rocsparse::atomic_add(nnz_, data[0]);
        }
        if(nnz_diag_ != nullptr)
        {
            data[hipThreadIdx_x] = nnz_diag;
            __syncthreads();

            rocsparse::blockreduce_sum<BLOCKSIZE>(hipThreadIdx_x, data);
            if(hipThreadIdx_x == 0)
            {
                rocsparse::atomic_add(nnz_diag_, data[0]);
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J, typename... P>
    static void
        kernel_compute_unnz_launch(dim3& blocks_, dim3& threads_, hipStream_t stream_, P... p)
    {
        THROW_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::kernel_compute_unnz<BLOCKSIZE, WFSIZE, I, J>),
                                          blocks_,
                                          threads_,
                                          0,
                                          stream_,
                                          p...);
    }

    template <uint32_t BLOCKSIZE, typename I, typename J, typename... P>
    static void kernel_compute_unnz_dispatch(J           target_size_,
                                             int         wavefront_size,
                                             hipStream_t stream_,
                                             P... p)
    {
        dim3 blocks((target_size_ + BLOCKSIZE - 1) / BLOCKSIZE);
        dim3 threads(BLOCKSIZE);
        if(wavefront_size == 32)
        {
            rocsparse::kernel_compute_unnz_launch<BLOCKSIZE, 32, I, J>(
                blocks, threads, stream_, p...);
        }
        else
        {
            rocsparse::kernel_compute_unnz_launch<BLOCKSIZE, 64, I, J>(
                blocks, threads, stream_, p...);
        }
    }
}

template <>
struct rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_async_inplace>
{
    static constexpr uint32_t BLOCKSIZE = 1024;

    template <typename T, typename J>
    struct history
    {

        static rocsparse_status run(buffer_layout_inplace_t& layout_,
                                    rocsparse_handle         handle_,
                                    rocsparse_itilu0_alg     alg_,
                                    J*                       niter_,
                                    T*                       data_,
                                    size_t                   buffer_size_,
                                    void*                    buffer_)
        {

            using layout_t = buffer_layout_inplace_t;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &layout_, buffer_, sizeof(layout_t), hipMemcpyDeviceToHost, handle_->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));
            void*  p_buffer      = layout_.get_pointer(layout_t::buffer);
            size_t p_buffer_size = layout_.get_size(layout_t::buffer);
            if(p_buffer_size == 0)
            {
                *niter_ = static_cast<J>(0);
                return rocsparse_status_success;
            }

            rocsparse::itilu0x_convergence_info_t<floating_data_t<T>, J> convergence_info;
            p_buffer = convergence_info.init(handle_, p_buffer);
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
                                                   sizeof(floating_data_t<T>) * niter,
                                                   hipMemcpyDeviceToHost,
                                                   handle_->stream));
            }

            if(compute_nrm_residual)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(data_ + niter,
                                                   convergence_info.log_mxresidual,
                                                   sizeof(floating_data_t<T>) * niter,
                                                   hipMemcpyDeviceToHost,
                                                   handle_->stream));
            }
            //
            // No synchronization needed here.
            //
            return rocsparse_status_success;
        }

        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J*                   niter_,
                                    T*                   data_,
                                    size_t               buffer_size_,
                                    void*                buffer_)
        {
            using layout_t = buffer_layout_inplace_t;
            layout_t layout;
            RETURN_IF_ROCSPARSE_ERROR(
                run(layout, handle_, alg_, niter_, data_, buffer_size_, buffer_));
            return rocsparse_status_success;
        }
    };

    template <typename I, typename J>
    struct buffer_size
    {
        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J                    options_,
                                    J                    nmaxiter_,
                                    J                    m_,
                                    I                    nnz_,
                                    const I* __restrict__ ptr_,
                                    const J* __restrict__ ind_,
                                    rocsparse_index_base base_,
                                    rocsparse_datatype   datatype_,
                                    size_t* __restrict__ buffer_size_)
        {
            const bool use_coo_format = (options_ & rocsparse_itilu0_option_coo_format) > 0;

            size_t buffer_size = 0;
            // quick compute of unnz.
            RETURN_IF_HIP_ERROR(hipMemsetAsync(handle_->buffer, 0, sizeof(I), handle_->stream));
            rocsparse::kernel_compute_unnz_dispatch<BLOCKSIZE, I, J>(m_,
                                                                     handle_->wavefront_size,
                                                                     handle_->stream,

                                                                     m_,
                                                                     ptr_,
                                                                     ptr_ + 1,
                                                                     ind_,
                                                                     base_,
                                                                     (I*)handle_->buffer,
                                                                     nullptr);
            I unnz;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &unnz, (I*)handle_->buffer, sizeof(I), hipMemcpyDeviceToHost, handle_->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));

            using layout_t = buffer_layout_inplace_t;
            layout_t::buffer_size(m_, nnz_, unnz, buffer_size, use_coo_format);

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
            *buffer_size_ = buffer_size;
            return rocsparse_status_success;
        }
    };

    template <typename I, typename J>
    struct preprocess
    {

        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J                    options_,
                                    J                    nmaxiter_,
                                    J                    m_,
                                    I                    nnz_,
                                    const I* __restrict__ ptr_,
                                    const J* __restrict__ ind_,
                                    rocsparse_index_base base_,
                                    rocsparse_datatype   datatype_,
                                    size_t               buffer_size_,
                                    void* __restrict__ buffer__)

        {
            void* __restrict__ buffer_ = buffer__;
            // quick compute of unnz.
            RETURN_IF_HIP_ERROR(hipMemsetAsync(handle_->buffer, 0, sizeof(I) * 2, handle_->stream));
            rocsparse::kernel_compute_unnz_dispatch<BLOCKSIZE, I, J>(m_,
                                                                     handle_->wavefront_size,
                                                                     handle_->stream,
                                                                     m_,
                                                                     ptr_,
                                                                     ptr_ + 1,
                                                                     ind_,
                                                                     base_,
                                                                     ((I*)handle_->buffer),
                                                                     ((I*)handle_->buffer) + 1);
            I hb[2];
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                hb, (I*)handle_->buffer, sizeof(I) * 2, hipMemcpyDeviceToHost, handle_->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));
            const I unnz     = hb[0];
            const I nnz_diag = hb[1];

            if(nnz_diag != m_)
            {
                std::cerr << "rocsparse_csritilu0_preprocess has detected " << (m_ - nnz_diag)
                          << "/" << m_ << " non-existent diagonal element." << std::endl;
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
            }

            const bool use_coo_format = (options_ & rocsparse_itilu0_option_coo_format) > 0;

            using layout_t = buffer_layout_inplace_t;
            layout_t layout;
            layout.init(m_, nnz_, unnz, datatype_, buffer_size_, buffer_, use_coo_format);

            I* p_lptr_end    = (I*)layout.get_pointer(layout_t::lptr_end);
            I* p_uptr        = (I*)layout.get_pointer(layout_t::uptr);
            J* p_uind        = (J*)layout.get_pointer(layout_t::uind);
            I* p_uperm       = (I*)layout.get_pointer(layout_t::uperm);
            J* p_coo_row_ind = (J*)layout.get_pointer(layout_t::coo_row_ind);

            //
            // Details.
            //
            // 1/ Compute lptr_end (L strictly lower)
            // 2/ Convert the strict upper part to coordinate format.
            // 3/ Sort the coordinates by column
            // 4/ Convert to csc.
            //
            //
            // Compute the vector lptr_end that separates
            // the strict lower part from the rest of the matrix.
            //
            rocsparse::kernel_compute_lptr_end_dispatch<BLOCKSIZE, I, J>(m_,
                                                                         handle_->wavefront_size,
                                                                         handle_->stream,
                                                                         m_,
                                                                         ptr_,
                                                                         ptr_ + 1,
                                                                         ind_,
                                                                         base_,
                                                                         p_lptr_end);

            I* tmp = p_uptr; // we can reuse the memory.
            J* csc_col_ind;
            if(p_coo_row_ind != nullptr)
            {
                csc_col_ind = p_coo_row_ind;
            }
            else
            {
                if(sizeof(J) * unnz > handle_->buffer_size)
                {

                    RETURN_IF_HIP_ERROR(
                        rocsparse_hipMallocAsync(&csc_col_ind, sizeof(J) * unnz, handle_->stream));
                }
                else
                {
                    csc_col_ind = (J*)handle_->buffer;
                }
            }

            //
            // Compute ucsr_ptr to help the extraction into coo format.
            //
            {
                //
                // Count.
                //
                rocsparse::kernel_initialize_ucsr_ptr_dispatch<BLOCKSIZE, I, J>(
                    m_,
                    handle_->wavefront_size,
                    handle_->stream,
                    m_,
                    p_lptr_end,
                    ptr_ + 1,
                    tmp,
                    base_);

                //
                //  Prefix sum.
                //
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::inclusive_scan(handle_, m_, tmp));
            }

            //
            // Extract the upper part into coo format.
            //
            rocsparse::kernel_compute_coo_dispatch<BLOCKSIZE, I, J>(m_,
                                                                    handle_->wavefront_size,
                                                                    handle_->stream,
                                                                    m_,
                                                                    p_lptr_end,
                                                                    ptr_ + 1,
                                                                    ind_,
                                                                    base_,
                                                                    tmp,
                                                                    p_uind,
                                                                    csc_col_ind,
                                                                    p_uperm);

            //
            // Sort the coordinates per column.
            //
            tmp = nullptr;

            void*  buffer = nullptr;
            size_t buffer_size;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_coosort_buffer_size(
                handle_, m_, m_, unnz, (I*)0x4, (I*)0x4, &buffer_size));

            if(buffer_size > 0)
            {
                RETURN_IF_HIP_ERROR(
                    rocsparse_hipMallocAsync(&buffer, buffer_size, handle_->stream));
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_coosort_by_column(
                handle_, m_, m_, unnz, p_uind, csc_col_ind, p_uperm, buffer));

            //
            // Free buffer.
            //
            if(buffer_size > 0)
            {
                RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(buffer, handle_->stream));
            }

            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::coo2csr_template(handle_, csc_col_ind, unnz, m_, p_uptr, base_));

            if(p_coo_row_ind == nullptr && csc_col_ind != handle_->buffer)
            {
                RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(csc_col_ind, handle_->stream));
            }

            if(use_coo_format)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2coo_core(
                    handle_, ptr_, ptr_ + 1, nnz_, m_, p_coo_row_ind, base_));
            }

            //
            // Copy the struct to device.
            //
            using layout_t = buffer_layout_inplace_t;

            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                buffer__, &layout, sizeof(layout_t), hipMemcpyHostToDevice, handle_->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));
            return rocsparse_status_success;
        }
    };

    template <typename T, typename I, typename J>
    struct compute
    {
        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J                    options_,
                                    J*                   nmaxiter_,
                                    J                    nfreeiter_,
                                    floating_data_t<T>   tol_,
                                    J                    m_,
                                    I                    nnz_,
                                    const I* __restrict__ ptr_,
                                    const J* __restrict__ ind_,
                                    const T* __restrict__ val_,
                                    T* __restrict__ sol_,
                                    rocsparse_index_base base_,
                                    size_t               buffer_size_,
                                    void* __restrict__ buffer_)
        {

            //
            // Get the layout from the buffer header.
            //
            using layout_t = buffer_layout_inplace_t;
            layout_t layout;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &layout, buffer_, sizeof(layout), hipMemcpyDeviceToHost, handle_->stream));
            buffer_ = (void*)(((double*)buffer_) + layout_t::get_sizeof_double());
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));

            //
            // Initialize pointers.
            //
            const I* p_lptr_begin = ptr_;
            const I* p_lptr_end   = (const I*)layout.get_pointer(layout_t::lptr_end);

            const I* p_uptr_begin = (const I*)layout.get_pointer(layout_t::uptr);
            const I* p_uptr_end   = p_uptr_begin + 1;
            const J* p_uind       = (const J*)layout.get_pointer(layout_t::uind);
            const I* p_uperm      = (const I*)layout.get_pointer(layout_t::uperm);

            const J* p_coo_row_ind = (const I*)layout.get_pointer(layout_t::coo_row_ind);

            void*  p_buffer      = layout.get_pointer(layout_t::buffer);
            size_t p_buffer_size = layout.get_size(layout_t::buffer);

            //
            // Compute expert routine.
            //

            const bool compute_residual
                = (options_ & rocsparse_itilu0_option_compute_nrm_residual) > 0;

            if(compute_residual)
            {
                RETURN_IF_ROCSPARSE_ERROR((compute_iter<BLOCKSIZE, T, I, J>::run)(handle_,
                                                                                  options_,
                                                                                  nmaxiter_,
                                                                                  nfreeiter_,
                                                                                  tol_,
                                                                                  m_,
                                                                                  nnz_,
                                                                                  ptr_,
                                                                                  ptr_ + 1,
                                                                                  p_coo_row_ind,
                                                                                  ind_,
                                                                                  val_,
                                                                                  base_,
                                                                                  p_lptr_begin,
                                                                                  p_lptr_end,
                                                                                  ind_,

                                                                                  p_uptr_begin,
                                                                                  p_uptr_end,
                                                                                  p_uind,
                                                                                  p_uperm,
                                                                                  sol_,
                                                                                  p_buffer_size,
                                                                                  p_buffer));
            }
            else
            {

                RETURN_IF_ROCSPARSE_ERROR(
                    (compute_iter<BLOCKSIZE, T, I, J>::light_run)(handle_,
                                                                  options_,
                                                                  nmaxiter_[0]
                                                                      + nmaxiter_[0] * nfreeiter_,
                                                                  m_,
                                                                  nnz_,
                                                                  ptr_,
                                                                  ptr_ + 1,
                                                                  p_coo_row_ind,
                                                                  ind_,
                                                                  val_,
                                                                  base_,
                                                                  p_lptr_begin,
                                                                  p_lptr_end,
                                                                  ind_,
                                                                  p_uptr_begin,
                                                                  p_uptr_end,
                                                                  p_uind,
                                                                  p_uperm,
                                                                  sol_));
            }
            return rocsparse_status_success;
        }
    };
};

#define INSTANTIATE(T, I, J)                       \
    template struct rocsparse::csritilu0_driver_t< \
        rocsparse_itilu0_alg_async_inplace>::compute<T, I, J>

INSTANTIATE(float, rocsparse_int, rocsparse_int);
INSTANTIATE(double, rocsparse_int, rocsparse_int);
INSTANTIATE(rocsparse_float_complex, rocsparse_int, rocsparse_int);
INSTANTIATE(rocsparse_double_complex, rocsparse_int, rocsparse_int);

#undef INSTANTIATE

#define INSTANTIATE(T, J) \
    template struct rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_async_inplace>::history<T, J>

INSTANTIATE(float, rocsparse_int);
INSTANTIATE(double, rocsparse_int);

#undef INSTANTIATE

#define INSTANTIATE(I, J)                                       \
    template struct rocsparse::csritilu0_driver_t<              \
        rocsparse_itilu0_alg_async_inplace>::buffer_size<I, J>; \
    template struct rocsparse::csritilu0_driver_t<              \
        rocsparse_itilu0_alg_async_inplace>::preprocess<I, J>;

INSTANTIATE(rocsparse_int, rocsparse_int);

#undef INSTANTIATE
