/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/precond/rocsparse_bsrilu0.h"
#include "rocsparse_bsrilu0.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename T>
    rocsparse_status xbsrilu0_numeric_boost(rocsparse_handle   handle,
                                            rocsparse_mat_info info,
                                            int                enable_boost,
                                            rocsparse_datatype boost_tol_datatype,
                                            const void*        boost_tol,
                                            const T*           boost_val)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(1, info);

        // Reset boost
        auto boost = info->get_boost();
        boost->set_enable(0);
        if(enable_boost)
        {
            ROCSPARSE_CHECKARG_POINTER(3, boost_tol);
            ROCSPARSE_CHECKARG_POINTER(4, boost_val);
            boost->define(enable_boost,
                          handle->pointer_mode,
                          boost_tol_datatype,
                          boost_tol,
                          handle->pointer_mode,
                          boost_val);
        }

        return rocsparse_status_success;
    }

}

extern "C" rocsparse_status rocsparse_sbsrilu0_numeric_boost(rocsparse_handle   handle,
                                                             rocsparse_mat_info info,
                                                             int                enable_boost,
                                                             const float*       boost_tol,
                                                             const float*       boost_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xbsrilu0_numeric_boost(
        handle, info, enable_boost, rocsparse_datatype_f32_r, boost_tol, boost_val));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dbsrilu0_numeric_boost(rocsparse_handle   handle,
                                                             rocsparse_mat_info info,
                                                             int                enable_boost,
                                                             const double*      boost_tol,
                                                             const double*      boost_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xbsrilu0_numeric_boost(
        handle, info, enable_boost, rocsparse_datatype_f64_r, boost_tol, boost_val));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_cbsrilu0_numeric_boost(rocsparse_handle               handle,
                                     rocsparse_mat_info             info,
                                     int                            enable_boost,
                                     const float*                   boost_tol,
                                     const rocsparse_float_complex* boost_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xbsrilu0_numeric_boost(
        handle, info, enable_boost, rocsparse_datatype_f32_r, boost_tol, boost_val));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_zbsrilu0_numeric_boost(rocsparse_handle                handle,
                                     rocsparse_mat_info              info,
                                     int                             enable_boost,
                                     const double*                   boost_tol,
                                     const rocsparse_double_complex* boost_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xbsrilu0_numeric_boost(
        handle, info, enable_boost, rocsparse_datatype_f64_r, boost_tol, boost_val));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dsbsrilu0_numeric_boost(rocsparse_handle   handle,
                                                              rocsparse_mat_info info,
                                                              int                enable_boost,
                                                              const double*      boost_tol,
                                                              const float*       boost_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xbsrilu0_numeric_boost(
        handle, info, enable_boost, rocsparse_datatype_f64_r, boost_tol, boost_val));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_dcbsrilu0_numeric_boost(rocsparse_handle               handle,
                                      rocsparse_mat_info             info,
                                      int                            enable_boost,
                                      const double*                  boost_tol,
                                      const rocsparse_float_complex* boost_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xbsrilu0_numeric_boost(
        handle, info, enable_boost, rocsparse_datatype_f64_r, boost_tol, boost_val));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
