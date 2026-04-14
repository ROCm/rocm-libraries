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

#include "rocsparse_csritilu0x_history.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_csritilu0x_driver.hpp"

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
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            return rocsparse_status_success;
        }

        case rocsparse_itilu0_alg_sync_split_fusion:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0x_driver_t<
                    rocsparse_itilu0_alg_sync_split_fusion>::history<T, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_sync_split:
        {
            RETURN_IF_ROCSPARSE_ERROR((
                rocsparse::csritilu0x_driver_t<rocsparse_itilu0_alg_sync_split>::history<T, J>::run(
                    parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_async_split:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0x_driver_t<
                    rocsparse_itilu0_alg_async_split>::history<T, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

template <typename T, typename J>
rocsparse_status rocsparse::csritilu0x_history_template(rocsparse_handle     handle_,
                                                        rocsparse_itilu0_alg alg_,
                                                        J* __restrict__ niter_,
                                                        T* __restrict__ data_,
                                                        size_t buffer_size_,
                                                        void* __restrict__ buffer_)
{
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::history_dispatch<T, J>(alg_, handle_, niter_, data_, buffer_size_, buffer_)));
    return rocsparse_status_success;
}

template rocsparse_status rocsparse::csritilu0x_history_template(rocsparse_handle     handle_,
                                                                 rocsparse_itilu0_alg alg_,
                                                                 rocsparse_int* __restrict__ niter_,
                                                                 float* __restrict__ data_,
                                                                 size_t buffer_size_,
                                                                 void* __restrict__ buffer_);

template rocsparse_status rocsparse::csritilu0x_history_template(rocsparse_handle     handle_,
                                                                 rocsparse_itilu0_alg alg_,
                                                                 rocsparse_int* __restrict__ niter_,
                                                                 double* __restrict__ data_,
                                                                 size_t buffer_size_,
                                                                 void* __restrict__ buffer_);
