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

namespace rocsparse
{
    ///
    /// @brief Core of the resulting number of non-zeros of the extract.
    ///
    static rocsparse_status
        extract_nnz_core(rocsparse_handle handle, const rocsparse_extract_descr descr, int64_t* nnz)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            nnz, descr->m_device_nnz, sizeof(int64_t), hipMemcpyDeviceToHost, handle->stream));
        return rocsparse_status_success;
    }

    ///
    /// @brief Quick return decision of the resulting number of non-zeros of the extract.
    ///
    static rocsparse_status extract_nnz_quickreturn(rocsparse_handle              handle,
                                                    const rocsparse_extract_descr descr,
                                                    int64_t*                      nnz)
    {
        ROCSPARSE_ROUTINE_TRACE;

        return rocsparse_status_continue;
    }

    ///
    /// @brief Argument checking of the resulting number of non-zeros of the extract.
    ///
    static rocsparse_status extract_nnz_checkarg(rocsparse_handle              handle,
                                                 const rocsparse_extract_descr descr,
                                                 int64_t*                      nnz)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(1, descr);
        ROCSPARSE_CHECKARG_POINTER(2, nnz);

        const rocsparse_status status = rocsparse::extract_nnz_quickreturn(handle, descr, nnz);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    ///
    /// @brief Implementation of the resulting number of non-zeros of the extract.
    ///
    template <typename... P>
    static rocsparse_status extract_nnz_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse::extract_nnz_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::extract_nnz_core(p...));
        return rocsparse_status_success;
    }

}

///
/// @brief C-API implementation of the resulting number of non-zeros of the extract.
///
extern "C" rocsparse_status
    rocsparse_extract_nnz(rocsparse_handle handle, rocsparse_extract_descr descr, int64_t* nnz)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::extract_nnz_impl(handle, descr, nnz));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
