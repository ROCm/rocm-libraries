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

#include "rocsparse_csritilu0_history.hpp"
#include "internal/precond/rocsparse_csritilu0.h"
#include "rocsparse_common.hpp"
#include "rocsparse_csritilu0_driver.hpp"

namespace rocsparse
{
    template <typename T, typename J, typename... P>
    static rocsparse_status history_dispatch(rocsparse_itilu0_alg alg_, P&&... parameters)
    {
        switch(alg_)
        {
        case rocsparse_itilu0_alg_default:
        case rocsparse_itilu0_alg_async_inplace:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_async_inplace>::
                     history<rocsparse::floating_data_t<T>, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_async_split:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_async_split>::
                     history<rocsparse::floating_data_t<T>, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_sync_split:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>::
                     history<rocsparse::floating_data_t<T>, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_sync_split_fusion:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_sync_split_fusion>::
                     history<rocsparse::floating_data_t<T>, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename T, typename J>
    static rocsparse_status
        csritilu0_history_template(rocsparse_handle     handle_,
                                   rocsparse_itilu0_alg alg_,
                                   J* __restrict__ niter_,
                                   rocsparse::floating_data_t<T>* __restrict__ nrms_,
                                   size_t buffer_size_,
                                   void* __restrict__ buffer_)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::history_dispatch<T, J>(
            alg_, handle_, alg_, niter_, nrms_, buffer_size_, buffer_)));
        return rocsparse_status_success;
    }
}

template <typename T, typename J>
rocsparse_status rocsparse::csritilu0_history_impl(rocsparse_handle     handle,
                                                   rocsparse_itilu0_alg alg,
                                                   J* __restrict__ niter,
                                                   rocsparse::floating_data_t<T>* __restrict__ nrms,
                                                   size_t buffer_size,
                                                   void* __restrict__ buffer)
{
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_csritilu0_history",
                         alg,
                         (const void*&)niter,
                         (const void*&)nrms,
                         buffer_size,
                         (const void*&)buffer);

    ROCSPARSE_CHECKARG_ENUM(1, alg);
    ROCSPARSE_CHECKARG_POINTER(2, niter);
    ROCSPARSE_CHECKARG_POINTER(3, nrms);
    ROCSPARSE_CHECKARG_ARRAY(5, buffer_size, buffer);
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csritilu0_history_template<T, J>(
        handle, alg, niter, nrms, buffer_size, buffer)));
    return rocsparse_status_success;
}

#define IMPL(NAME, T, J)                                                         \
    extern "C" rocsparse_status NAME(rocsparse_handle               handle,      \
                                     rocsparse_itilu0_alg           alg,         \
                                     J*                             niter,       \
                                     rocsparse::floating_data_t<T>* nrms,        \
                                     size_t                         buffer_size, \
                                     void*                          buffer)      \
    try                                                                          \
    {                                                                            \
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csritilu0_history_impl<T, J>(      \
            handle, alg, niter, nrms, buffer_size, buffer)));                    \
        return rocsparse_status_success;                                         \
    }                                                                            \
    catch(...)                                                                   \
    {                                                                            \
        RETURN_ROCSPARSE_EXCEPTION();                                            \
    }

IMPL(rocsparse_scsritilu0_history, float, rocsparse_int);
IMPL(rocsparse_dcsritilu0_history, double, rocsparse_int);
IMPL(rocsparse_ccsritilu0_history, rocsparse_float_complex, rocsparse_int);
IMPL(rocsparse_zcsritilu0_history, rocsparse_double_complex, rocsparse_int);

#undef IMPL
