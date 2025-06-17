/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_gcreate_identity_permutation.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_identity.hpp"
#include "rocsparse_internal_convert_scalar.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::gcreate_identity_permutation(rocsparse_handle    handle_,
                                                         int64_t             nnz,
                                                         rocsparse_indextype idx_type,
                                                         void*               perm)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(idx_type)
    {
    case rocsparse_indextype_u16:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    case rocsparse_indextype_i32:
    {
        int32_t local_nnz;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::internal_convert_scalar<int64_t, int32_t>)(nnz, local_nnz));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_identity_permutation_template<int32_t>(
            handle_, local_nnz, (int32_t*)perm));
        return rocsparse_status_success;
    }
    case rocsparse_indextype_i64:
    {
        int64_t local_nnz;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::internal_convert_scalar<int64_t, int64_t>)(nnz, local_nnz));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_identity_permutation_template<int64_t>(
            handle_, local_nnz, (int64_t*)perm));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}
