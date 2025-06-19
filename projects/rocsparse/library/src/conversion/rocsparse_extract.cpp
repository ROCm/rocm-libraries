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
#include "rocsparse_extract_alg_default.hpp"
#include "rocsparse_utility.hpp"

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_extract_stage value)
{
    switch(value)
    {
    case rocsparse_extract_stage_analysis:
    case rocsparse_extract_stage_compute:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_extract_alg value)
{
    switch(value)
    {
    case rocsparse_extract_alg_default:
    {
        return false;
    }
    }
    return true;
}

namespace rocsparse
{

    ///
    /// @brief Core of the extract.
    ///
    static rocsparse_status extract_core(rocsparse_handle            handle,
                                         rocsparse_extract_descr     descr,
                                         rocsparse_const_spmat_descr source,
                                         rocsparse_spmat_descr       target,
                                         rocsparse_extract_stage     stage,
                                         size_t                      buffer_size_in_bytes,
                                         void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(descr->alg())
        {
        case rocsparse_extract_alg_default:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                descr->run(handle, source, target, stage, buffer_size_in_bytes, buffer));
            return rocsparse_status_success;
        }
        }
        return rocsparse_status_success;
    }

    ///
    /// @brief Quick return decision of the extract.
    ///
    static rocsparse_status extract_quickreturn(rocsparse_handle            handle,
                                                rocsparse_extract_descr     descr,
                                                rocsparse_const_spmat_descr source,
                                                rocsparse_spmat_descr       target,
                                                rocsparse_extract_stage     stage,
                                                size_t                      buffer_size_in_bytes,
                                                void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        return rocsparse_status_continue;
    }

    ///
    /// @brief Argument checking of the extract.
    ///
    static rocsparse_status extract_checkarg(rocsparse_handle            handle, //0
                                             rocsparse_extract_descr     descr, //1
                                             rocsparse_const_spmat_descr source, //2
                                             rocsparse_spmat_descr       target, //3
                                             rocsparse_extract_stage     stage, //4
                                             size_t                      buffer_size_in_bytes, //5
                                             void*                       buffer) //6
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(1, descr);
        ROCSPARSE_CHECKARG_POINTER(2, source);
        ROCSPARSE_CHECKARG_POINTER(3, target);
        ROCSPARSE_CHECKARG_ENUM(4, stage);
        ROCSPARSE_CHECKARG_ARRAY(6, buffer_size_in_bytes, buffer);

        const rocsparse_status status = rocsparse::extract_quickreturn(
            handle, descr, source, target, stage, buffer_size_in_bytes, buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    ///
    /// @brief Implementation of the extract.
    ///
    template <typename... P>
    static rocsparse_status extract_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse::extract_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::extract_core(p...));
        return rocsparse_status_success;
    }

}

///
/// @brief C-API implementation of the extract.
///
extern "C" rocsparse_status rocsparse_extract(rocsparse_handle            handle,
                                              rocsparse_extract_descr     descr,
                                              rocsparse_const_spmat_descr source,
                                              rocsparse_spmat_descr       target,
                                              rocsparse_extract_stage     stage,
                                              size_t                      buffer_size_in_bytes,
                                              void*                       buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::extract_impl(
        handle, descr, source, target, stage, buffer_size_in_bytes, buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
