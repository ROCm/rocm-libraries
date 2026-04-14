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
#pragma once

#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    //
    // Struct to layout the buffer for the LDU factorization.
    //
    template <typename T, typename I, typename J>
    struct itilu0x_layout_t
    {

    public:
        T* __restrict__ dval{};
        T* __restrict__ lval{};
        T* __restrict__ uval{};
        I lnnz{};
        I unnz{};
        J m{};

        itilu0x_layout_t() {}

        void init(J                   m_,
                  rocsparse_diag_type ldiag_type_,
                  I                   lnnz_,
                  rocsparse_diag_type udiag_type_,
                  I                   unnz_,
                  void*               buffer)
        {
            m    = m_;
            lnnz = lnnz_;
            unnz = unnz_;

            lval   = (T* __restrict__)buffer;
            buffer = (void* __restrict__)&lval[lnnz_];
            uval   = (T* __restrict__)buffer;
            buffer = (void* __restrict__)&uval[unnz_];
            if(ldiag_type_ == rocsparse_diag_type_unit && udiag_type_ == rocsparse_diag_type_unit)
            {
                dval   = (T* __restrict__)buffer;
                buffer = (void* __restrict__)&dval[m_];
            }
        }
    };

    template <typename T, typename J>
    struct itilu0x_info_t
    {
        T*    nrm_matrix{};
        T*    nrm_corr{};
        T*    nrm_residual{};
        J*    options;
        J*    nmaxiter;
        J*    local_iter{};
        J*    iter{};
        void* init(void* buffer_)
        {
            void* buffer = buffer_;
            //
            // T first for aligments.
            //
            nrm_matrix = ((T*)buffer);
            buffer     = (void*)&nrm_matrix[1];

            nrm_corr = ((T*)buffer);
            buffer   = (void*)&nrm_corr[1];

            nrm_residual = ((T*)buffer);
            buffer       = (void*)&nrm_residual[1];

            options = ((J*)buffer);
            buffer  = (void*)&options[1];

            nmaxiter = ((J*)buffer);
            buffer   = (void*)&nmaxiter[1];

            local_iter = ((J*)buffer);
            buffer     = (void*)&local_iter[1];

            iter   = ((J*)buffer);
            buffer = (void*)&iter[1];

            return (void*)(((char*)buffer_) + size());
        };

        static size_t size()
        {
            return (((sizeof(T) * 3 + sizeof(J) * 4) - 1) / sizeof(T) + 1) * sizeof(T);
        };
    };

    template <typename T, typename J>
    struct itilu0x_convergence_info_t
    {
        rocsparse::itilu0x_info_t<T, J> info{};
        T* __restrict__ log_mxcorr{};
        T* __restrict__ log_mxresidual{};

        static size_t size(J nsweeps_, J options_)
        {
            const bool compute_nrm_corr
                = (options_ & rocsparse_itilu0_option_compute_nrm_correction) > 0;
            const bool compute_nrm_residual
                = (options_ & rocsparse_itilu0_option_compute_nrm_residual) > 0;
            const bool convergence_history
                = (options_ & rocsparse_itilu0_option_convergence_history) > 0;

            size_t s = 0;
            s += rocsparse::itilu0x_info_t<T, J>::size();
            if(compute_nrm_corr)
            {
                if(convergence_history)
                {
                    s += sizeof(T) * nsweeps_;
                }
            }

            if(compute_nrm_residual)
            {
                if(convergence_history)
                {
                    s += sizeof(T) * nsweeps_;
                }
            }

            return s;
        };

        void* init(rocsparse_handle handle_, void* buffer_, J nsweeps_, J options_)
        {

            void* buffer = buffer_;
            buffer       = info.init(buffer);
            THROW_IF_HIP_ERROR(hipMemcpyAsync(
                info.options, &options_, sizeof(J), hipMemcpyHostToDevice, handle_->stream));
            THROW_IF_HIP_ERROR(hipMemcpyAsync(
                info.nmaxiter, &nsweeps_, sizeof(J), hipMemcpyHostToDevice, handle_->stream));
            THROW_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));

            const bool compute_nrm_corr
                = (options_ & rocsparse_itilu0_option_compute_nrm_correction) > 0;
            const bool compute_nrm_residual
                = (options_ & rocsparse_itilu0_option_compute_nrm_residual) > 0;
            const bool convergence_history
                = (options_ & rocsparse_itilu0_option_convergence_history) > 0;

            if(convergence_history)
            {
                if(compute_nrm_corr)
                {
                    log_mxcorr = ((T* __restrict__)buffer);
                    buffer     = (void* __restrict__)&log_mxcorr[nsweeps_];
                }
                if(compute_nrm_residual)
                {
                    log_mxresidual = ((T* __restrict__)buffer);
                    buffer         = (void* __restrict__)&log_mxresidual[nsweeps_];
                }
            }

            return buffer;
        }

        void* init(rocsparse_handle handle_, void* buffer_)
        {
            void* buffer = buffer_;
            buffer       = info.init(buffer);

            J options_, nsweeps_;
            THROW_IF_HIP_ERROR(hipMemcpyAsync(
                &options_, info.options, sizeof(J), hipMemcpyDeviceToHost, handle_->stream));
            THROW_IF_HIP_ERROR(hipMemcpyAsync(
                &nsweeps_, info.nmaxiter, sizeof(J), hipMemcpyDeviceToHost, handle_->stream));
            THROW_IF_HIP_ERROR(hipStreamSynchronize(handle_->stream));

            const bool compute_nrm_corr
                = (options_ & rocsparse_itilu0_option_compute_nrm_correction) > 0;
            const bool compute_nrm_residual
                = (options_ & rocsparse_itilu0_option_compute_nrm_residual) > 0;
            const bool convergence_history
                = (options_ & rocsparse_itilu0_option_convergence_history) > 0;

            if(convergence_history)
            {
                if(compute_nrm_corr)
                {
                    log_mxcorr = ((T* __restrict__)buffer);
                    buffer     = (void* __restrict__)&log_mxcorr[nsweeps_];
                }
                if(compute_nrm_residual)
                {
                    log_mxresidual = ((T* __restrict__)buffer);
                    buffer         = (void* __restrict__)&log_mxresidual[nsweeps_];
                }
            }

            return buffer;
        }
    };

    template <rocsparse_itilu0_alg alg_>
    struct csritilu0x_driver_t
    {

        template <typename T, typename J>
        struct history
        {

            static rocsparse_status run(rocsparse_handle handle_,
                                        J* __restrict__ niter_,
                                        T* __restrict__ data_,
                                        size_t buffer_size_,
                                        void* __restrict__ buffer_);
        };

        template <typename I, typename J>
        struct buffer_size
        {
            static rocsparse_status run(rocsparse_handle handle_,
                                        J                options_,
                                        J                nsweeps,
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
                                        size_t* __restrict__ buffer_size_);
        };
        template <typename I, typename J>
        struct preprocess
        {
            static rocsparse_status run(rocsparse_handle handle_,
                                        J                options_,
                                        J                nsweeps,
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
                                        void* __restrict__ buffer_);
        };
        template <typename T, typename I, typename J>
        struct compute
        {
            static rocsparse_status run(rocsparse_handle handle_,
                                        J                options_,
                                        J* __restrict__ nsweeps_,
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
                                        void* __restrict__ buffer_);
        };
    };

    template <typename T, typename J>
    inline __device__ T sparse_dotproduct(J                          xn_,
                                          const J*                   xind_,
                                          const T*                   x_,
                                          const rocsparse_index_base xbase_,
                                          J                          yn_,
                                          const J*                   yind_,
                                          const T*                   y_,
                                          const rocsparse_index_base ybase_,
                                          J&                         ix_,
                                          J&                         iy_)
    {
        J jx, jy;
        T s = static_cast<T>(0);
        while((ix_ < xn_) && (iy_ < yn_))
        {
            jx  = xind_[ix_] - xbase_;
            jy  = yind_[iy_] - ybase_;
            s   = (jx == jy) ? rocsparse::fma(x_[ix_], y_[iy_], s) : s;
            ix_ = (jx <= jy) ? (ix_ + 1) : ix_;
            iy_ = (jx >= jy) ? (iy_ + 1) : iy_;
        }
        return s;
    }
}
