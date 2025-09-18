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

#include "rocsparse_csritilu0_preprocess.hpp"
#include "internal/precond/rocsparse_csritilu0.h"
#include "rocsparse_common.hpp"
#include "rocsparse_csritilu0_driver.hpp"

namespace rocsparse
{
    template <typename I, typename J, typename... P>
    static rocsparse_status preprocess_dispatch(rocsparse_itilu0_alg alg_, P&&... parameters)
    {
        switch(alg_)
        {
        case rocsparse_itilu0_alg_default:
        case rocsparse_itilu0_alg_async_inplace:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_async_inplace>::preprocess<I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_async_split:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_async_split>::preprocess<I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_sync_split:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_sync_split>::preprocess<I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_sync_split_fusion:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_sync_split_fusion>::preprocess<I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csritilu0_preprocess_template(rocsparse_handle     handle_,
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
                                                          void* __restrict__ buffer_)
{
    // Quick return if possible
    if(m_ == 0)
    {
        return rocsparse_status_success;
    }
    if(nnz_ == 0)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
    }

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::preprocess_dispatch<I, J>(alg_,
                                                                    handle_,
                                                                    alg_,
                                                                    options_,
                                                                    nmaxiter_,
                                                                    m_,
                                                                    nnz_,
                                                                    ptr_,
                                                                    ind_,
                                                                    base_,
                                                                    datatype_,
                                                                    buffer_size_,
                                                                    buffer_)));
    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::csritilu0_preprocess_impl(rocsparse_handle     handle, //0
                                                      rocsparse_itilu0_alg alg, //1
                                                      J                    options, //2
                                                      J                    nmaxiter, //3
                                                      J                    m, //4
                                                      I                    nnz, //5
                                                      const I* __restrict__ ptr, //6
                                                      const J* __restrict__ ind, //7
                                                      rocsparse_index_base base, //8
                                                      rocsparse_datatype   datatype, //9
                                                      size_t               buffer_size, //10
                                                      void* __restrict__ buffer) //11
{
    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_csritilu0_preprocess",
                         alg,
                         options,
                         nmaxiter,
                         m,
                         nnz,
                         (const void*&)ptr,
                         (const void*&)ind,
                         base,
                         datatype,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_ENUM(1, alg);
    ROCSPARSE_CHECKARG(2, options, (options < 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(3, nmaxiter, (nmaxiter < 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG_SIZE(4, m);
    ROCSPARSE_CHECKARG_SIZE(5, nnz);
    ROCSPARSE_CHECKARG_ARRAY(6, m, ptr);
    ROCSPARSE_CHECKARG_ARRAY(7, nnz, ind);
    ROCSPARSE_CHECKARG_ENUM(8, base);
    ROCSPARSE_CHECKARG_ENUM(9, datatype);
    ROCSPARSE_CHECKARG_ARRAY(11, buffer_size, buffer);

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csritilu0_preprocess_template(
        handle, alg, options, nmaxiter, m, nnz, ptr, ind, base, datatype, buffer_size, buffer)));
    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_csritilu0_preprocess(rocsparse_handle     handle_,
                                                           rocsparse_itilu0_alg alg_,
                                                           rocsparse_int        options_,
                                                           rocsparse_int        nmaxiter_,
                                                           rocsparse_int        m_,
                                                           rocsparse_int        nnz_,
                                                           const rocsparse_int* ptr_,
                                                           const rocsparse_int* ind_,
                                                           rocsparse_index_base base_,
                                                           rocsparse_datatype   datatype_,
                                                           size_t               buffer_size_,
                                                           void*                buffer_)
try
{
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::csritilu0_preprocess_impl<rocsparse_int, rocsparse_int>(handle_,
                                                                            alg_,
                                                                            options_,
                                                                            nmaxiter_,
                                                                            m_,
                                                                            nnz_,
                                                                            ptr_,
                                                                            ind_,
                                                                            base_,
                                                                            datatype_,
                                                                            buffer_size_,
                                                                            buffer_)));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
