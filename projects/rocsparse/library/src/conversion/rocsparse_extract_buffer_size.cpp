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

#include "rocsparse_extract.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{

    ///
    /// @brief Core of the extract buffer size.
    ///
    static rocsparse_status extract_buffer_size_core(rocsparse_handle            handle,
                                                     rocsparse_extract_descr     descr,
                                                     rocsparse_const_spmat_descr source,
                                                     rocsparse_spmat_descr       target,
                                                     rocsparse_extract_stage     stage,
                                                     size_t* buffer_size_in_bytes)
    {
        ROCSPARSE_ROUTINE_TRACE;

        buffer_size_in_bytes[0] = 0;
        RETURN_IF_ROCSPARSE_ERROR(
            descr->buffer_size(handle, source, target, stage, buffer_size_in_bytes));
        return rocsparse_status_success;
    }

    ///
    /// @brief Quick return decision of the extract buffer size.
    ///
    static rocsparse_status extract_buffer_size_quickreturn(rocsparse_handle            handle,
                                                            rocsparse_extract_descr     descr,
                                                            rocsparse_const_spmat_descr source,
                                                            rocsparse_spmat_descr       target,
                                                            rocsparse_extract_stage     stage,
                                                            size_t* buffer_size_in_bytes)
    {
        ROCSPARSE_ROUTINE_TRACE;
        return rocsparse_status_continue;
    }

    ///
    /// @brief Argument checking of the extract buffer size.
    ///
    static rocsparse_status extract_buffer_size_checkarg(rocsparse_handle            handle,
                                                         rocsparse_extract_descr     descr,
                                                         rocsparse_const_spmat_descr source,
                                                         rocsparse_spmat_descr       target,
                                                         rocsparse_extract_stage     stage,
                                                         size_t* buffer_size_in_bytes)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(1, descr);
        ROCSPARSE_CHECKARG_POINTER(2, source);
        ROCSPARSE_CHECKARG_POINTER(3, target);
        ROCSPARSE_CHECKARG_ENUM(4, stage);
        ROCSPARSE_CHECKARG_POINTER(5, buffer_size_in_bytes);

        const rocsparse_status status = rocsparse::extract_buffer_size_quickreturn(
            handle, descr, source, target, stage, buffer_size_in_bytes);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    ///
    /// @brief Implementation of the extract buffer size.
    ///
    template <typename... P>
    static rocsparse_status extract_buffer_size_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse::extract_buffer_size_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::extract_buffer_size_core(p...));
        return rocsparse_status_success;
    }

}

///
/// @brief C-API implementation of the extract buffer size.
///
extern "C" rocsparse_status rocsparse_extract_buffer_size(rocsparse_handle            handle,
                                                          rocsparse_extract_descr     descr,
                                                          rocsparse_const_spmat_descr source,
                                                          rocsparse_spmat_descr       target,
                                                          rocsparse_extract_stage     stage,
                                                          size_t* buffer_size_in_bytes)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::extract_buffer_size_impl(
        handle, descr, source, target, stage, buffer_size_in_bytes));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
