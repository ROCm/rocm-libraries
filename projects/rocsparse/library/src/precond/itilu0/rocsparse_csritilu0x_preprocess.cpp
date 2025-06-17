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

#include "rocsparse_csritilu0x_preprocess.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_csritilu0x_driver.hpp"

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
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_async_split:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0x_driver_t<
                    rocsparse_itilu0_alg_async_split>::preprocess<I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_sync_split:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0x_driver_t<
                    rocsparse_itilu0_alg_sync_split>::preprocess<I, J>::run(parameters...)));
            return rocsparse_status_success;
        }
        case rocsparse_itilu0_alg_sync_split_fusion:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0x_driver_t<
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
rocsparse_status rocsparse::csritilu0x_preprocess_template(rocsparse_handle     handle_,
                                                           rocsparse_itilu0_alg alg_,
                                                           J                    options_,
                                                           J                    nsweeps_,
                                                           J                    m_,
                                                           I                    nnz_,
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
                                                                    options_,
                                                                    nsweeps_,
                                                                    m_,
                                                                    nnz_,
                                                                    ptr_begin_,
                                                                    ptr_end_,
                                                                    ind_,
                                                                    base_,
                                                                    ldiag_type_,
                                                                    ldir_,
                                                                    lnnz_,
                                                                    lptr_begin_,
                                                                    lptr_end_,
                                                                    lind_,
                                                                    lbase_,
                                                                    udiag_type_,
                                                                    udir_,
                                                                    unnz_,
                                                                    uptr_begin_,
                                                                    uptr_end_,
                                                                    uind_,
                                                                    ubase_,
                                                                    datatype_,
                                                                    buffer_size_,
                                                                    buffer_)));
    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::csritilu0x_preprocess_impl(rocsparse_handle     handle_,
                                                       rocsparse_itilu0_alg alg_,
                                                       J                    options_,
                                                       J                    nsweeps_,
                                                       J                    m_,
                                                       I                    nnz_,
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

    if(handle_ == nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_handle);
    }

    // Logging
    rocsparse::log_trace(handle_,
                         "rocsparse_Xcsritilu0x_preprocess",
                         options_,
                         nsweeps_,
                         m_,
                         nnz_,
                         (const void*&)ptr_begin_,
                         (const void*&)ptr_end_,
                         (const void*&)ind_,
                         base_,
                         ldiag_type_,
                         ldir_,
                         lnnz_,
                         (const void*&)lptr_begin_,
                         (const void*&)lptr_end_,
                         (const void*&)lind_,
                         lbase_,
                         udiag_type_,
                         udir_,
                         unnz_,
                         (const void*&)uptr_begin_,
                         (const void*&)uptr_end_,
                         (const void*&)uind_,

                         ubase_,

                         datatype_,
                         buffer_size_,
                         (const void*&)buffer_);

    // Check sizes
    if(m_ < 0 || nnz_ < 0)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_size);
    }

    // Check pointer arguments
    if(m_ > 0 && (ptr_begin_ == nullptr || ptr_end_ == nullptr))
    {

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    if(nnz_ > 0 && ind_ == nullptr)
    {

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    if(rocsparse::enum_utils::is_invalid(base_))
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    if(rocsparse::enum_utils::is_invalid(ubase_))
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
    if(rocsparse::enum_utils::is_invalid(lbase_))
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    if(rocsparse::enum_utils::is_invalid(udiag_type_))
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
    if(rocsparse::enum_utils::is_invalid(udir_))
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
    if(rocsparse::enum_utils::is_invalid(ldiag_type_))
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
    if(rocsparse::enum_utils::is_invalid(ldir_))
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    if((m_ > 0)
       && (uptr_begin_ == nullptr || lptr_begin_ == nullptr || uptr_end_ == nullptr
           || lptr_end_ == nullptr))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    if((unnz_ > 0) && (uind_ == nullptr))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    if((lnnz_ > 0) && (lind_ == nullptr))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    if(ldir_ != rocsparse_direction_row)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    if(udir_ != rocsparse_direction_column)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csritilu0x_preprocess_template(handle_,
                                                                        alg_,
                                                                        options_,
                                                                        nsweeps_,
                                                                        m_,
                                                                        nnz_,
                                                                        ptr_begin_,
                                                                        ptr_end_,
                                                                        ind_,
                                                                        base_,
                                                                        ldiag_type_,
                                                                        ldir_,
                                                                        lnnz_,
                                                                        lptr_begin_,
                                                                        lptr_end_,
                                                                        lind_,
                                                                        lbase_,
                                                                        udiag_type_,
                                                                        udir_,
                                                                        unnz_,
                                                                        uptr_begin_,
                                                                        uptr_end_,
                                                                        uind_,

                                                                        ubase_,
                                                                        datatype_,
                                                                        buffer_size_,
                                                                        buffer_));
    return rocsparse_status_success;
}
#define INSTANTIATE(TOK, I, J)                                              \
    template rocsparse_status rocsparse::csritilu0x_preprocess_##TOK<I, J>( \
        rocsparse_handle     handle_,                                       \
        rocsparse_itilu0_alg alg_,                                          \
        J                    options_,                                      \
        J                    nsweeps_,                                      \
        J                    m_,                                            \
        I                    nnz_,                                          \
        const I* __restrict__ ptr_begin_,                                   \
        const I* __restrict__ ptr_end_,                                     \
        const J* __restrict__ ind_,                                         \
        rocsparse_index_base base_,                                         \
        rocsparse_diag_type  ldiag_type_,                                   \
        rocsparse_direction  ldir_,                                         \
        I                    lnnz_,                                         \
        const I* __restrict__ lptr_begin_,                                  \
        const I* __restrict__ lptr_end_,                                    \
        const J* __restrict__ lind_,                                        \
        rocsparse_index_base lbase_,                                        \
        rocsparse_diag_type  udiag_type_,                                   \
        rocsparse_direction  udir_,                                         \
        I                    unnz_,                                         \
        const I* __restrict__ uptr_begin_,                                  \
        const I* __restrict__ uptr_end_,                                    \
        const J* __restrict__ uind_,                                        \
        rocsparse_index_base ubase_,                                        \
        rocsparse_datatype   datatype_,                                     \
        size_t               buffer_size_,                                  \
        void* __restrict__ buffer_)

INSTANTIATE(template, rocsparse_int, rocsparse_int);
INSTANTIATE(impl, rocsparse_int, rocsparse_int);
#undef INSTANTIATE
