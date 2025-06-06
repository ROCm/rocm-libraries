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

#include "rocsparse_csritilu0_buffer_size.hpp"
#include "internal/precond/rocsparse_csritilu0.h"
#include "rocsparse_common.hpp"
#include "rocsparse_csritilu0_driver.hpp"
#include "rocsparse_enum_utils.hpp"

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_itilu0_alg value)
{

    switch(value)
    {
#define CASE(C) \
    case C:     \
        return #C
        CASE(rocsparse_itilu0_alg_default);
        CASE(rocsparse_itilu0_alg_async_inplace);
        CASE(rocsparse_itilu0_alg_async_split);
        CASE(rocsparse_itilu0_alg_sync_split);
        CASE(rocsparse_itilu0_alg_sync_split_fusion);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
};

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_itilu0_alg value)
{
    switch(value)
    {
    case rocsparse_itilu0_alg_default:
    case rocsparse_itilu0_alg_async_inplace:
    case rocsparse_itilu0_alg_async_split:
    case rocsparse_itilu0_alg_sync_split:
    case rocsparse_itilu0_alg_sync_split_fusion:
    {
        return false;
    }
    }
    return true;
}

namespace rocsparse
{

    template <typename I, typename J, typename... P>
    static rocsparse_status buffer_size_dispatch(rocsparse_itilu0_alg alg_, P&&... parameters)
    {
        switch(alg_)
        {
        case rocsparse_itilu0_alg_default:
        case rocsparse_itilu0_alg_async_inplace:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_async_inplace>::buffer_size<I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_async_split:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_async_split>::buffer_size<I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_sync_split:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_sync_split>::buffer_size<I, J>::run(parameters...)));
            return rocsparse_status_success;
        }

        case rocsparse_itilu0_alg_sync_split_fusion:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_sync_split_fusion>::
                     buffer_size<I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csritilu0_buffer_size_template(rocsparse_handle     handle_,
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
    if(m_ == 0)
    {
        *buffer_size_ = 0;
        return rocsparse_status_success;
    }
    if(nnz_ == 0)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
    }

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::buffer_size_dispatch<I, J>(alg_,
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
                                                                     buffer_size_)));
    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::csritilu0_buffer_size_impl(rocsparse_handle     handle,
                                                       rocsparse_itilu0_alg alg,
                                                       J                    options,
                                                       J                    nmaxiter,
                                                       J                    m,
                                                       I                    nnz,
                                                       const I* __restrict__ ptr,
                                                       const J* __restrict__ ind,
                                                       rocsparse_index_base base,
                                                       rocsparse_datatype   datatype,
                                                       size_t* __restrict__ buffer_size)
{
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_csritilu0_buffer_size",
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
    ROCSPARSE_CHECKARG_POINTER(10, buffer_size);
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csritilu0_buffer_size_template(
        handle, alg, options, nmaxiter, m, nnz, ptr, ind, base, datatype, buffer_size));
    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_csritilu0_buffer_size(rocsparse_handle     handle,
                                                            rocsparse_itilu0_alg alg,
                                                            rocsparse_int        options,
                                                            rocsparse_int        nmaxiter,
                                                            rocsparse_int        m,
                                                            rocsparse_int        nnz,
                                                            const rocsparse_int* __restrict__ ptr,
                                                            const rocsparse_int* __restrict__ ind,
                                                            rocsparse_index_base base,
                                                            rocsparse_datatype   datatype,
                                                            size_t* __restrict__ buffer_size)
try
{
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csritilu0_buffer_size_impl<rocsparse_int, rocsparse_int>(
        handle, alg, options, nmaxiter, m, nnz, ptr, ind, base, datatype, buffer_size)));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
