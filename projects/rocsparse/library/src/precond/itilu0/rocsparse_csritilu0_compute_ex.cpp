/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
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

#include "internal/precond/rocsparse_csritilu0.h"
#include "rocsparse_common.hpp"
#include "rocsparse_csritilu0_driver.hpp"

namespace rocsparse
{
    template <typename T, typename I, typename J, typename... P>
    static rocsparse_status compute_dispatch(rocsparse_itilu0_alg alg_, P&&... parameters)
    {
        switch(alg_)
        {
        case rocsparse_itilu0_alg_default:
        case rocsparse_itilu0_alg_async_inplace:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_async_inplace>::compute<T, I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_async_split:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_async_split>::compute<T, I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_sync_split:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_sync_split>::compute<T, I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_sync_split_fusion:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_sync_split_fusion>::compute<T, I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename T, typename I, typename J>
    static rocsparse_status csritilu0_compute_template(rocsparse_handle              handle_,
                                                       rocsparse_itilu0_alg          alg_,
                                                       J                             options_,
                                                       J*                            nmaxiter_,
                                                       J                             nfreeiter_,
                                                       rocsparse::floating_data_t<T> tol_,
                                                       J                             m_,
                                                       I                             nnz_,
                                                       const I* __restrict__ ptr_,
                                                       const J* __restrict__ ind_,
                                                       const T* __restrict__ val_,
                                                       T* __restrict__ ilu0_,
                                                       rocsparse_index_base base_,
                                                       size_t               buffer_size_,
                                                       void* __restrict__ buffer_)
    {

        // Quick return if possible
        if(m_ == 0)
        {
            nmaxiter_[0] = 0;
            return rocsparse_status_success;
        }
        if(nnz_ == 0)
        {
            nmaxiter_[0] = 0;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
        }

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::compute_dispatch<T, I, J>(alg_,
                                                                        handle_,
                                                                        alg_,
                                                                        options_,
                                                                        nmaxiter_,
                                                                        nfreeiter_,
                                                                        tol_,
                                                                        m_,
                                                                        nnz_,
                                                                        ptr_,
                                                                        ind_,
                                                                        val_,
                                                                        ilu0_,
                                                                        base_,
                                                                        buffer_size_,
                                                                        buffer_)));
        return rocsparse_status_success;
    }

    template <typename T, typename I, typename J>
    rocsparse_status csritilu0_compute_impl(rocsparse_handle              handle, //0
                                            rocsparse_itilu0_alg          alg, //1
                                            J                             options, //2
                                            J*                            nmaxiter, //3
                                            J                             nfreeiter, //4
                                            rocsparse::floating_data_t<T> tol, //5
                                            J                             m, //6
                                            I                             nnz, //7
                                            const I* __restrict__ ptr, //8
                                            const J* __restrict__ ind, //9
                                            const T* __restrict__ val, //10
                                            T* __restrict__ ilu0, //11
                                            rocsparse_index_base base, //12
                                            size_t               buffer_size, //13
                                            void* __restrict__ buffer)
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        // Logging
        log_trace(handle,
                  "rocsparse_csritilu0_compute_ex",
                  alg,
                  options,
                  (const void*&)nmaxiter,
                  nfreeiter,
                  tol,
                  m,
                  nnz,
                  (const void*&)ptr,
                  (const void*&)ind,
                  (const void*&)val,
                  base,
                  buffer_size,
                  (const void*&)buffer);

        ROCSPARSE_CHECKARG_ENUM(1, alg);
        ROCSPARSE_CHECKARG(2, options, (options < 0), rocsparse_status_invalid_value);
        ROCSPARSE_CHECKARG_POINTER(3, nmaxiter);
        ROCSPARSE_CHECKARG_SIZE(4, nfreeiter);
        ROCSPARSE_CHECKARG(5, tol, (tol < 0), rocsparse_status_invalid_value);
        ROCSPARSE_CHECKARG_SIZE(6, m);
        ROCSPARSE_CHECKARG_SIZE(7, nnz);
        ROCSPARSE_CHECKARG_ARRAY(8, m, ptr);
        ROCSPARSE_CHECKARG_ARRAY(9, nnz, ind);
        ROCSPARSE_CHECKARG_ARRAY(10, nnz, val);
        ROCSPARSE_CHECKARG_ARRAY(11, nnz, ilu0);
        ROCSPARSE_CHECKARG_ENUM(12, base);
        ROCSPARSE_CHECKARG_ARRAY(14, buffer_size, buffer);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csritilu0_compute_template(handle,
                                                                        alg,
                                                                        options,
                                                                        nmaxiter,
                                                                        nfreeiter,
                                                                        tol,
                                                                        m,
                                                                        nnz,
                                                                        ptr,
                                                                        ind,
                                                                        val,
                                                                        ilu0,
                                                                        base,
                                                                        buffer_size,
                                                                        buffer));
        return rocsparse_status_success;
    }
}

#define IMPL(NAME, T, I, J)                                                                \
    extern "C" rocsparse_status NAME(rocsparse_handle              handle,                 \
                                     rocsparse_itilu0_alg          alg,                    \
                                     J                             options,                \
                                     J*                            nmaxiter,               \
                                     J                             nfreeiter,              \
                                     rocsparse::floating_data_t<T> tol,                    \
                                     J                             m,                      \
                                     I                             nnz,                    \
                                     const I*                      ptr,                    \
                                     const J*                      ind,                    \
                                     const T*                      val,                    \
                                     T*                            ilu0,                   \
                                     rocsparse_index_base          base,                   \
                                     size_t                        buffer_size,            \
                                     void*                         buffer)                 \
    try                                                                                    \
    {                                                                                      \
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csritilu0_compute_impl<T, I, J>(handle,      \
                                                                              alg,         \
                                                                              options,     \
                                                                              nmaxiter,    \
                                                                              nfreeiter,   \
                                                                              tol,         \
                                                                              m,           \
                                                                              nnz,         \
                                                                              ptr,         \
                                                                              ind,         \
                                                                              val,         \
                                                                              ilu0,        \
                                                                              base,        \
                                                                              buffer_size, \
                                                                              buffer)));   \
        return rocsparse_status_success;                                                   \
    }                                                                                      \
    catch(...)                                                                             \
    {                                                                                      \
        RETURN_ROCSPARSE_EXCEPTION();                                                      \
    }

IMPL(rocsparse_scsritilu0_compute_ex, float, rocsparse_int, rocsparse_int);
IMPL(rocsparse_dcsritilu0_compute_ex, double, rocsparse_int, rocsparse_int);
IMPL(rocsparse_ccsritilu0_compute_ex, rocsparse_float_complex, rocsparse_int, rocsparse_int);
IMPL(rocsparse_zcsritilu0_compute_ex, rocsparse_double_complex, rocsparse_int, rocsparse_int);

#undef IMPL
