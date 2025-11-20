/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level2/rocsparse_csritsv.h"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csritsv.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_csrmv.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status nrminf(rocsparse_handle                     handle_,
                            size_t                               nitems_,
                            const T*                             x_,
                            rocsparse::floating_data_t<T>*       nrm_,
                            const rocsparse::floating_data_t<T>* nrm0_,
                            bool                                 MX);

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status nrminf_diff(rocsparse_handle                     handle_,
                                 size_t                               nitems_,
                                 const T*                             x_,
                                 const T*                             y_,
                                 rocsparse::floating_data_t<T>*       nrm_,
                                 const rocsparse::floating_data_t<T>* nrm0_,
                                 bool                                 MX);

    template <uint32_t BLOCKSIZE, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_add_scaled_residual(J m,
                                    const T* __restrict__ r_,
                                    T* __restrict__ y_,
                                    const T* __restrict__ invdiag)
    {
        const uint32_t tid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
        if(tid < m)
        {
            y_[tid] = y_[tid] + invdiag[tid] * r_[tid];
        }
    }

}

namespace
{
    template <typename T, typename I, typename J>
    struct calculator_inverse_diagonal_t
    {

        template <uint32_t BLOCKSIZE, bool CONJ>
        ROCSPARSE_KERNEL(BLOCKSIZE)
        void kernel_inverse_diagonal(J m,
                                     const J* __restrict__ ind,
                                     const T* __restrict__ val,
                                     rocsparse_index_base base,
                                     T* __restrict__ invdiag,
                                     const I* __restrict__ ptr_diag,
                                     const J              ptr_shift,
                                     rocsparse_index_base ptr_diag_base,
                                     rocsparse_int* __restrict__ zero_pivot)
        {
            const J tid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
            if(tid < m)
            {
                const I k = ptr_diag[tid] - ptr_diag_base + ptr_shift;
                const J j = ind[k] - base;
                if(j == tid)
                {
                    const T local_val = (!CONJ) ? val[k] : rocsparse::conj(val[k]);
                    if(local_val != static_cast<T>(0))
                    {
                        invdiag[tid] = static_cast<T>(1) / local_val;
                    }
                    else
                    {
                        rocsparse::atomic_min<rocsparse_int>(zero_pivot, tid + base);
                        invdiag[tid] = static_cast<T>(1);
                    }
                }
                else
                {
                    rocsparse::atomic_min<rocsparse_int>(zero_pivot, tid + base);
                    invdiag[tid] = static_cast<T>(1);
                }
            }
        }

        static rocsparse_status calculate(rocsparse_handle    handle,
                                          rocsparse_operation trans,
                                          J                   m,
                                          I                   nnz,
                                          const J* __restrict__ csr_ind,
                                          const T* __restrict__ csr_val,
                                          rocsparse_index_base csr_base,
                                          T*                   invdiag,
                                          const I* __restrict__ csr_diag_ind,
                                          J                    ptr_shift,
                                          rocsparse_index_base csr_diag_ind_base,
                                          rocsparse_int* __restrict__ zero_pivot)
        {
            ROCSPARSE_ROUTINE_TRACE;

            //
            // Compute inverse of the diagonal.
            //
            static constexpr uint32_t BLOCKSIZE = 1024;
            dim3                      blocks((m - 1) / BLOCKSIZE + 1);
            dim3                      threads(BLOCKSIZE);
            switch(trans)
            {
            case rocsparse_operation_transpose:
            case rocsparse_operation_none:
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((kernel_inverse_diagonal<BLOCKSIZE, false>),
                                                   blocks,
                                                   threads,
                                                   0,
                                                   handle->stream,
                                                   m,
                                                   csr_ind,
                                                   csr_val,
                                                   csr_base,
                                                   invdiag,
                                                   csr_diag_ind,
                                                   ptr_shift,
                                                   csr_diag_ind_base,
                                                   zero_pivot);
                break;
            }
            case rocsparse_operation_conjugate_transpose:
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((kernel_inverse_diagonal<BLOCKSIZE, true>),
                                                   blocks,
                                                   threads,
                                                   0,
                                                   handle->stream,
                                                   m,
                                                   csr_ind,
                                                   csr_val,
                                                   csr_base,
                                                   invdiag,
                                                   csr_diag_ind,
                                                   ptr_shift,
                                                   csr_diag_ind_base,
                                                   zero_pivot);
                break;
            }
            }

            return rocsparse_status_success;
        }
    };

}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csritsv_solve_ex_template(rocsparse_handle handle,
                                                      rocsparse_int*   host_nmaxiter,
                                                      rocsparse_int    host_nfreeiter,
                                                      const rocsparse::floating_data_t<T>* host_tol,
                                                      rocsparse::floating_data_t<T>* host_history,
                                                      rocsparse_operation            trans,
                                                      J                              m,
                                                      I                              nnz,
                                                      const T*                  alpha_device_host,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  csr_val,
                                                      const I*                  csr_row_ptr,
                                                      const J*                  csr_col_ind,
                                                      rocsparse_mat_info        info,
                                                      const T*                  x,
                                                      T*                        y,
                                                      rocsparse_solve_policy    policy,
                                                      void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    static constexpr bool fallback_algorithm = true;
    static constexpr bool force_conj         = false;

    const bool                    breakable   = (host_tol != nullptr);
    const bool                    recordable  = (host_history != nullptr);
    const bool                    compute_nrm = (recordable || breakable);
    const rocsparse_int           nmaxiter    = host_nmaxiter[0];
    T*                            y_p         = (T*)temp_buffer;
    hipStream_t                   stream      = handle->stream;
    rocsparse::floating_data_t<T> host_nrm[1];

    static constexpr bool verbose = false;
    if(m == 0)
    {
        return rocsparse_status_success;
    }

    rocsparse_csritsv_info    csritsv_info = info->csritsv_info;
    static constexpr uint32_t BLOCKSIZE    = 1024;
    dim3                      blocks((m - 1) / BLOCKSIZE + 1);
    dim3                      threads(BLOCKSIZE);

    //
    // reinitialize zero pivot.
    //
    if(descr->diag_type == rocsparse_diag_type_unit)
    {
        rocsparse_int max = std::numeric_limits<rocsparse_int>::max();
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(info->csritsv_info->get_zero_pivot(),
                                           &max,
                                           sizeof(rocsparse_int),
                                           hipMemcpyHostToDevice,
                                           stream));

        // Wait for device transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    }

    const rocsparse_fill_mode fill_mode = descr->fill_mode;
    const rocsparse_diag_type diag_type = descr->diag_type;
    if(nnz == 0)
    {
        //
        //
        //
        if(diag_type == rocsparse_diag_type_unit)
        {
            //
            // Compute the residual.
            //
            if(compute_nrm)
            {
                RETURN_IF_HIP_ERROR(
                    hipMemcpyAsync(y_p, x, sizeof(T) * m, hipMemcpyDeviceToDevice, stream));
                //
                // Scale.
                //
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::scale_array(handle, m, alpha_device_host, y_p));
                rocsparse::floating_data_t<T>* device_nrm
                    = (rocsparse::floating_data_t<T>*)(y_p + m);
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::nrminf_diff<1024>(handle, m, y_p, y, device_nrm, nullptr, false));
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(host_nrm,
                                                   device_nrm,
                                                   sizeof(rocsparse::floating_data_t<T>),
                                                   hipMemcpyDeviceToHost,
                                                   stream));
                RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
                if(recordable)
                {
                    host_history[0] = host_nrm[0];
                }
            }

            host_nmaxiter[0] = 0;
            if(host_nrm[0] > host_tol[0])
            {
                if(compute_nrm)
                {
                    RETURN_IF_HIP_ERROR(
                        hipMemcpyAsync(y, y_p, sizeof(T) * m, hipMemcpyDeviceToDevice, stream));
                }
                else
                {
                    //
                    // Copy.
                    //
                    RETURN_IF_HIP_ERROR(
                        hipMemcpyAsync(y, x, sizeof(T) * m, hipMemcpyDeviceToDevice, stream));
                    //
                    // Scale.
                    //
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::scale_array(handle, m, alpha_device_host, y));
                }
                host_nmaxiter[0] = 1;
            }
            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_async(
                reinterpret_cast<rocsparse_int*>(info->csritsv_info->get_zero_pivot()),
                (rocsparse_int)descr->base,
                stream));
            return rocsparse_status_success;
        }
    }

    //
    // Use buffer as a vector of size 2xm.
    //
    T* invdiag            = (rocsparse_diag_type_non_unit == diag_type) ? &y_p[m] : nullptr;
    T* csrmv_alpha_device = (rocsparse_diag_type_non_unit == diag_type) ? &y_p[m * 2] : &y_p[m];
    rocsparse::floating_data_t<T>* device_nrm
        = (rocsparse::floating_data_t<T>*)(csrmv_alpha_device + 1);

    //
    // Check if we need to store csrmv_alpha on host or on device.
    //
    const T*       csrmv_alpha_device_host{};
    static const T s_minus_one = static_cast<T>(-1);
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        csrmv_alpha_device_host = csrmv_alpha_device;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            csrmv_alpha_device, &s_minus_one, sizeof(T), hipMemcpyHostToDevice, stream));
    }
    else
    {
        csrmv_alpha_device_host = &s_minus_one;
    }

    const I* ptr_begin;
    const I* ptr_end;
    const I* ptr_diag;
    J        ptr_diag_shift;
    switch(diag_type)
    {
    case rocsparse_diag_type_non_unit:
    {
        if(csritsv_info->is_submatrix)
        {
            switch(fill_mode)
            {
            case rocsparse_fill_mode_lower:
            {
                ptr_begin      = csr_row_ptr;
                ptr_end        = (const I*)csritsv_info->ptr_end;
                ptr_diag       = ptr_end;
                ptr_diag_shift = -1;
                break;
            }
            case rocsparse_fill_mode_upper:
            {
                ptr_begin      = (const I*)csritsv_info->ptr_end;
                ptr_end        = csr_row_ptr + 1;
                ptr_diag       = ptr_begin;
                ptr_diag_shift = 0;
                break;
            }
            }
        }
        else
        {
            ptr_begin = csr_row_ptr;
            ptr_end   = csr_row_ptr + 1;
            switch(fill_mode)
            {
            case rocsparse_fill_mode_lower:
            {
                ptr_diag_shift = -1;
                ptr_diag       = ptr_end;
                break;
            }
            case rocsparse_fill_mode_upper:
            {
                ptr_diag_shift = 0;
                ptr_diag       = ptr_begin;
                break;
            }
            }
        }
        break;
    }

    case rocsparse_diag_type_unit:
    {
        //
        // We can simplify since D is identity and we expect that only T is stored.
        // yk+1 = yk + inv(D) * ( alpha * x - (D + T) yk )
        // yk+1 = alpha * x - T yk
        //
        // yk+1 = alpha * x - T yk
        // rk = alpha * x - (Id+T) yk = yk+1 - yk
        // rk = yk+1 - yk
        if(csritsv_info->is_submatrix)
        {
            switch(fill_mode)
            {
            case rocsparse_fill_mode_lower:
            {
                ptr_begin = csr_row_ptr;
                ptr_end   = (const I*)csritsv_info->ptr_end;
                break;
            }
            case rocsparse_fill_mode_upper:
            {
                ptr_begin = (const I*)csritsv_info->ptr_end;
                ptr_end   = csr_row_ptr + 1;
                break;
            }
            }
        }
        else
        {
            ptr_begin = csr_row_ptr;
            ptr_end   = csr_row_ptr + 1;
        }
        break;
    }
    }
    rocsparse_int ch = 1;
    switch(diag_type)
    {
    case rocsparse_diag_type_non_unit:
    {
        //
        // Compute the inverse of the diagonal.
        //
        RETURN_IF_ROCSPARSE_ERROR(
            (calculator_inverse_diagonal_t<T, I, J>::calculate)(handle,
                                                                trans,
                                                                m,
                                                                nnz,
                                                                csr_col_ind,
                                                                csr_val,
                                                                descr->base,
                                                                invdiag,
                                                                ptr_diag,
                                                                ptr_diag_shift,
                                                                descr->base,
                                                                (rocsparse_int*)info->csritsv_info
                                                                    ->get_zero_pivot()));

        int64_t zero_pivot;
        auto    csritsv_info = info->csritsv_info;
        auto    status       = csritsv_info->copy_zero_pivot_async(rocsparse_pointer_mode_host,
                                                          rocsparse::get_indextype<int64_t>(),
                                                          &zero_pivot,
                                                          handle->stream);
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
        if(status == rocsparse_status_zero_pivot)
        {
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(status);

        //
        // in y out y
        //

        //
        // y_p = y
        // y = r_k
        // y = y_p + invD * (r_k)

        //
        // tmp = (alpha*x - A * y)
        // tmp *= invdiag(tmp)
        // y = y + tmp;
        //

        for(rocsparse_int iter = 0; iter < nmaxiter; ++iter)
        {
            for(rocsparse_int freeiter = 0; freeiter < host_nfreeiter; ++freeiter)
            {
                RETURN_IF_HIP_ERROR(
                    hipMemcpyAsync(y_p, x, sizeof(T) * m, hipMemcpyDeviceToDevice, stream));

                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse::
                         csrmv_template<T, I, J, T, T, T>)(handle,
                                                           trans,
                                                           info != nullptr
                                                               ? rocsparse::csrmv_alg_adaptive
                                                               : rocsparse::csrmv_alg_rowsplit,
                                                           m,
                                                           m,
                                                           nnz,
                                                           csrmv_alpha_device_host,
                                                           descr,
                                                           csr_val,
                                                           ptr_begin,
                                                           ptr_end,
                                                           csr_col_ind,
                                                           info->get_csrmv_info(),
                                                           y,
                                                           alpha_device_host,
                                                           y_p,
                                                           force_conj,
                                                           fallback_algorithm));
                //
                // Add scale the residual
                //
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::kernel_add_scaled_residual<BLOCKSIZE>),
                    blocks,
                    threads,
                    0,
                    stream,
                    m,
                    y_p,
                    y,
                    invdiag);
            }

            //
            // Compute residual.
            //
            RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(y_p, x, sizeof(T) * m, hipMemcpyDeviceToDevice, stream));
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csrmv_template<T, I, J, T, T, T>)(handle,
                                                              trans,
                                                              info != nullptr
                                                                  ? rocsparse::csrmv_alg_adaptive
                                                                  : rocsparse::csrmv_alg_rowsplit,
                                                              m,
                                                              m,
                                                              nnz,
                                                              csrmv_alpha_device_host,
                                                              descr,
                                                              csr_val,
                                                              ptr_begin,
                                                              ptr_end,
                                                              csr_col_ind,
                                                              info->get_csrmv_info(),
                                                              y,
                                                              alpha_device_host,
                                                              y_p,
                                                              force_conj,
                                                              fallback_algorithm));
            bool break_loop = false;
            if(compute_nrm)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::nrminf<1024>(handle, m, y_p, device_nrm, nullptr, false));
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(host_nrm,
                                                   device_nrm,
                                                   sizeof(rocsparse::floating_data_t<T>),
                                                   hipMemcpyDeviceToHost,
                                                   stream));
                RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
                if(verbose)
                {
                    std::cout << "device iter " << iter << ", nrm " << host_nrm[0] << std::endl;
                }
                break_loop = (breakable) ? (host_nrm[0] <= host_tol[0]) : false;
                if(recordable)
                {
                    host_history[iter] = host_nrm[0];
                }
            }

            if(break_loop)
            {
                host_nmaxiter[0] = iter;
                break;
            }

            //
            // Add scale the residual
            //
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::kernel_add_scaled_residual<BLOCKSIZE>),
                                               blocks,
                                               threads,
                                               0,
                                               stream,
                                               m,
                                               y_p,
                                               y,
                                               invdiag);
        }

        break;
    }

    case rocsparse_diag_type_unit:
    {
        bool break_loop = false;
        for(rocsparse_int iter = 0; iter < nmaxiter; ++iter)
        {
            for(rocsparse_int freeiter = 0; freeiter < host_nfreeiter; ++freeiter)
            {
                //
                // swap pointers.
                //
                {
                    auto tmp = y_p;
                    y_p      = y;
                    y        = tmp;
                    ch *= -1;
                }

                //
                // We can simplify since D is identity and we expect that only T is stored.
                //  yk+1 = yk + inv(D) * ( alpha * x - (D + T) yk )
                // becomes
                //  yk+1 = alpha * x - T yk
                //

                //
                // Copy x to y_{k+1}
                //
                RETURN_IF_HIP_ERROR(
                    hipMemcpyAsync(y, x, sizeof(T) * m, hipMemcpyDeviceToDevice, stream));

                //
                // y_{k+1} = -T y_k + alpha * y_{k+1}
                //
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse::
                         csrmv_template<T, I, J, T, T, T>)(handle,
                                                           trans,
                                                           info != nullptr
                                                               ? rocsparse::csrmv_alg_adaptive
                                                               : rocsparse::csrmv_alg_rowsplit,
                                                           m,
                                                           m,
                                                           nnz,
                                                           csrmv_alpha_device_host,
                                                           descr,
                                                           csr_val,
                                                           ptr_begin,
                                                           ptr_end,
                                                           csr_col_ind,
                                                           info->get_csrmv_info(),
                                                           y_p,
                                                           alpha_device_host,
                                                           y,
                                                           force_conj,
                                                           fallback_algorithm));
            }

            //
            // swap pointers.
            //
            {
                auto tmp = y_p;
                y_p      = y;
                y        = tmp;
                ch *= -1;
            }

            //
            // We can simplify since D is identity and we expect that only T is stored.
            //  y_{k+1} = y_k + inv(D) * ( alpha * x - (D + T) y_k )
            // becomes
            //  y_{k+1} = alpha * x - T y_k
            //

            //
            // Copy x to y_{k+1}
            //
            RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(y, x, sizeof(T) * m, hipMemcpyDeviceToDevice, stream));

            //
            // y_{k+1} = -T y_k + alpha * y_{k+1}
            //
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csrmv_template<T, I, J, T, T, T>)(handle,
                                                              trans,
                                                              info != nullptr
                                                                  ? rocsparse::csrmv_alg_adaptive
                                                                  : rocsparse::csrmv_alg_rowsplit,
                                                              m,
                                                              m,
                                                              nnz,
                                                              csrmv_alpha_device_host,
                                                              descr,
                                                              csr_val,
                                                              ptr_begin,
                                                              ptr_end,
                                                              csr_col_ind,
                                                              info->get_csrmv_info(),
                                                              y_p,
                                                              alpha_device_host,
                                                              y,
                                                              force_conj,
                                                              fallback_algorithm));

            if(compute_nrm)
            {
                //
                // nrm.
                //
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::nrminf_diff<1024>(handle, m, y_p, y, device_nrm, nullptr, false));
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(host_nrm,
                                                   device_nrm,
                                                   sizeof(rocsparse::floating_data_t<T>),
                                                   hipMemcpyDeviceToHost,
                                                   stream));
                RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
                if(verbose)
                {
                    std::cout << "device iter " << iter << ", nrm " << host_nrm[0] << std::endl;
                }
                break_loop = (breakable) ? (host_nrm[0] <= host_tol[0]) : false;
                if(recordable)
                {
                    host_history[iter] = host_nrm[0];
                }
            }

            if(break_loop)
            {
                host_nmaxiter[0] = iter;
                break;
            }
        }
        if(ch < 0)
        {
            if(break_loop == false)
            {
                RETURN_IF_HIP_ERROR(
                    hipMemcpyAsync(y_p, y, sizeof(T) * m, hipMemcpyDeviceToDevice, stream));
            }
        }
        else
        {
            if(break_loop == true)
            {
                RETURN_IF_HIP_ERROR(
                    hipMemcpyAsync(y, y_p, sizeof(T) * m, hipMemcpyDeviceToDevice, stream));
            }
        }
        break;
    }
    }
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status csritsv_solve_ex_impl(rocsparse_handle                     handle,
                                           rocsparse_int*                       host_nmaxiter,
                                           rocsparse_int                        host_nfreeiter,
                                           const rocsparse::floating_data_t<T>* host_tol,
                                           rocsparse::floating_data_t<T>*       host_history,
                                           rocsparse_operation                  trans,
                                           J                                    m,
                                           I                                    nnz,
                                           const T*                             alpha_device_host,
                                           const rocsparse_mat_descr            descr,
                                           const T*                             csr_val,
                                           const I*                             csr_row_ptr,
                                           const J*                             csr_col_ind,
                                           rocsparse_mat_info                   info,
                                           const T*                             x,
                                           T*                                   y,
                                           rocsparse_solve_policy               policy,
                                           void*                                temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Check for valid handle and matrix descriptor
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(9, descr);
        ROCSPARSE_CHECKARG_POINTER(13, info);

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xcsritsv_solve_ex"),
                             (const void*&)host_nmaxiter,
                             host_nfreeiter,
                             (const void*&)host_tol,
                             (const void*&)host_history,
                             trans,
                             m,
                             nnz,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                             (const void*&)descr,
                             (const void*&)csr_val,
                             (const void*&)csr_row_ptr,
                             (const void*&)csr_col_ind,
                             (const void*&)info,
                             (const void*&)x,
                             (const void*&)y,
                             policy,
                             (const void*&)temp_buffer);

        ROCSPARSE_CHECKARG_ENUM(5, trans);
        ROCSPARSE_CHECKARG_ENUM(16, policy);

        // Check matrix type
        ROCSPARSE_CHECKARG(9,
                           descr,
                           (descr->type != rocsparse_matrix_type_general
                            && descr->type != rocsparse_matrix_type_triangular),
                           rocsparse_status_not_implemented);

        // Check matrix sorting mode

        ROCSPARSE_CHECKARG(9,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        // Check sizes
        ROCSPARSE_CHECKARG_SIZE(6, m);
        ROCSPARSE_CHECKARG_SIZE(7, nnz);

        ROCSPARSE_CHECKARG_ARRAY(10, nnz, csr_val);

        ROCSPARSE_CHECKARG_ARRAY(11, m, csr_row_ptr);

        ROCSPARSE_CHECKARG_ARRAY(12, nnz, csr_col_ind);

        ROCSPARSE_CHECKARG(17,
                           temp_buffer,
                           (m > 0 && nnz > 0 && temp_buffer == nullptr),
                           rocsparse_status_invalid_pointer);

        ROCSPARSE_CHECKARG_POINTER(1, host_nmaxiter);

        ROCSPARSE_CHECKARG_SIZE(2, host_nfreeiter);

        ROCSPARSE_CHECKARG_POINTER(8, alpha_device_host);

        ROCSPARSE_CHECKARG_ARRAY(14, m, x);

        ROCSPARSE_CHECKARG_ARRAY(15, m, y);

        ROCSPARSE_CHECKARG(
            13, info, (m > 0 && info->csritsv_info == nullptr), rocsparse_status_invalid_pointer);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csritsv_solve_ex_template(handle,
                                                                       host_nmaxiter,
                                                                       host_nfreeiter,
                                                                       host_tol,
                                                                       host_history,
                                                                       trans,
                                                                       m,
                                                                       nnz,
                                                                       alpha_device_host,
                                                                       descr,
                                                                       csr_val,
                                                                       csr_row_ptr,
                                                                       csr_col_ind,
                                                                       info,
                                                                       x,
                                                                       y,
                                                                       policy,
                                                                       temp_buffer));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(I, J, T)                                                 \
    template rocsparse_status rocsparse::csritsv_solve_ex_template<I, J, T>( \
        rocsparse_handle handle,                                             \
        rocsparse_int * host_nmaxiter,                                       \
        rocsparse_int                        host_nfreeiter,                 \
        const rocsparse::floating_data_t<T>* host_tol,                       \
        rocsparse::floating_data_t<T>*       host_history,                   \
        rocsparse_operation                  trans,                          \
        J                                    m,                              \
        I                                    nnz,                            \
        const T*                             alpha_device_host,              \
        const rocsparse_mat_descr            descr,                          \
        const T*                             csr_val,                        \
        const I*                             csr_row_ptr,                    \
        const J*                             csr_col_ind,                    \
        rocsparse_mat_info                   info,                           \
        const T*                             x,                              \
        T*                                   y,                              \
        rocsparse_solve_policy               policy,                         \
        void*                                temp_buffer);                                                  \
    template rocsparse_status rocsparse::csritsv_solve_ex_impl<I, J, T>(     \
        rocsparse_handle handle,                                             \
        rocsparse_int * host_nmaxiter,                                       \
        rocsparse_int                        host_nfreeiter,                 \
        const rocsparse::floating_data_t<T>* host_tol,                       \
        rocsparse::floating_data_t<T>*       host_history,                   \
        rocsparse_operation                  trans,                          \
        J                                    m,                              \
        I                                    nnz,                            \
        const T*                             alpha_device_host,              \
        const rocsparse_mat_descr            descr,                          \
        const T*                             csr_val,                        \
        const I*                             csr_row_ptr,                    \
        const J*                             csr_col_ind,                    \
        rocsparse_mat_info                   info,                           \
        const T*                             x,                              \
        T*                                   y,                              \
        rocsparse_solve_policy               policy,                         \
        void*                                temp_buffer)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, T)                                                                      \
    extern "C" rocsparse_status NAME(rocsparse_handle                     handle,            \
                                     rocsparse_int*                       host_nmaxiter,     \
                                     rocsparse_int                        host_nfreeiter,    \
                                     const rocsparse::floating_data_t<T>* host_tol,          \
                                     rocsparse::floating_data_t<T>*       host_history,      \
                                     rocsparse_operation                  trans,             \
                                     rocsparse_int                        m,                 \
                                     rocsparse_int                        nnz,               \
                                     const T*                             alpha_device_host, \
                                     const rocsparse_mat_descr            descr,             \
                                     const T*                             csr_val,           \
                                     const rocsparse_int*                 csr_row_ptr,       \
                                     const rocsparse_int*                 csr_col_ind,       \
                                     rocsparse_mat_info                   info,              \
                                     const T*                             x,                 \
                                     T*                                   y,                 \
                                     rocsparse_solve_policy               policy,            \
                                     void*                                temp_buffer)       \
    try                                                                                      \
    {                                                                                        \
        ROCSPARSE_ROUTINE_TRACE;                                                             \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csritsv_solve_ex_impl(handle,                   \
                                                                   host_nmaxiter,            \
                                                                   host_nfreeiter,           \
                                                                   host_tol,                 \
                                                                   host_history,             \
                                                                   trans,                    \
                                                                   m,                        \
                                                                   nnz,                      \
                                                                   alpha_device_host,        \
                                                                   descr,                    \
                                                                   csr_val,                  \
                                                                   csr_row_ptr,              \
                                                                   csr_col_ind,              \
                                                                   info,                     \
                                                                   x,                        \
                                                                   y,                        \
                                                                   policy,                   \
                                                                   temp_buffer));            \
        return rocsparse_status_success;                                                     \
    }                                                                                        \
    catch(...)                                                                               \
    {                                                                                        \
        RETURN_ROCSPARSE_EXCEPTION();                                                        \
    }

C_IMPL(rocsparse_scsritsv_solve_ex, float);
C_IMPL(rocsparse_dcsritsv_solve_ex, double);
C_IMPL(rocsparse_ccsritsv_solve_ex, rocsparse_float_complex);
C_IMPL(rocsparse_zcsritsv_solve_ex, rocsparse_double_complex);

#undef C_IMPL
