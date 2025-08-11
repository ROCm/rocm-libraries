/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_blas_rocblas.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse::blas_impl value)
{
    switch(value)
    {
    case rocsparse::blas_impl_none:
    case rocsparse::blas_impl_default:
    case rocsparse::blas_impl_rocblas:
    {
        return false;
    }
    }
    return true;
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse::blas_impl value)
{

#define CASE(C) \
    case C:     \
        return #C
    switch(value)
    {
        CASE(rocsparse::blas_impl_none);
        CASE(rocsparse::blas_impl_default);
        CASE(rocsparse::blas_impl_rocblas);
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
// LCOV_EXCL_STOP
#undef case
};

rocsparse_status rocsparse::blas_create_handle(rocsparse::blas_handle* pblas_handle,
                                               rocsparse::blas_impl    blas_impl)
try
{
    ROCSPARSE_CHECKARG_POINTER(0, pblas_handle);
    ROCSPARSE_CHECKARG_ENUM(1, blas_impl);
    *pblas_handle              = new rocsparse::_blas_handle();
    pblas_handle[0]->blas_impl = blas_impl;
    switch(blas_impl)
    {
    case rocsparse::blas_impl_none:
    {
        return rocsparse_status_success;
    }
    case rocsparse::blas_impl_default:
    case rocsparse::blas_impl_rocblas:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::blas_rocblas_create_handle(&pblas_handle[0]->blas_rocblas_handle));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse::blas_destroy_handle(rocsparse::blas_handle blas_handle)
try
{
    if(blas_handle)
    {

        switch(blas_handle->blas_impl)
        {
        case rocsparse::blas_impl_none:
        {
            break;
        }
        case rocsparse::blas_impl_default:
        case rocsparse::blas_impl_rocblas:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::blas_rocblas_destroy_handle(blas_handle->blas_rocblas_handle));
            break;
        }
        }

        delete blas_handle;
    }

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse::blas_set_stream(rocsparse::blas_handle blas_handle, hipStream_t stream)
{
    ROCSPARSE_CHECKARG_POINTER(0, blas_handle);
    switch(blas_handle->blas_impl)
    {
    case rocsparse::blas_impl_none:
    {
        return rocsparse_status_success;
    }
    case rocsparse::blas_impl_default:
    case rocsparse::blas_impl_rocblas:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::blas_rocblas_set_stream(blas_handle->blas_rocblas_handle, stream));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

rocsparse_status rocsparse::blas_set_pointer_mode(rocsparse::blas_handle blas_handle,
                                                  rocsparse_pointer_mode pointer_mode)
{
    ROCSPARSE_CHECKARG_POINTER(0, blas_handle);
    ROCSPARSE_CHECKARG_ENUM(1, pointer_mode);
    switch(blas_handle->blas_impl)
    {
    case rocsparse::blas_impl_none:
    {
        return rocsparse_status_success;
    }
    case rocsparse::blas_impl_default:
    case rocsparse::blas_impl_rocblas:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::blas_rocblas_set_pointer_mode(
            blas_handle->blas_rocblas_handle, pointer_mode));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

rocsparse_status rocsparse::blas_gemm_ex(rocsparse::blas_handle   blas_handle,
                                         rocsparse_operation      transA,
                                         rocsparse_operation      transB,
                                         rocsparse_int            m,
                                         rocsparse_int            n,
                                         rocsparse_int            k,
                                         const void*              alpha,
                                         const void*              a,
                                         rocsparse_datatype       a_type,
                                         rocsparse_int            lda,
                                         const void*              b,
                                         rocsparse_datatype       b_type,
                                         rocsparse_int            ldb,
                                         const void*              beta,
                                         const void*              c,
                                         rocsparse_datatype       c_type,
                                         rocsparse_int            ldc,
                                         void*                    d,
                                         rocsparse_datatype       d_type,
                                         rocsparse_int            ldd,
                                         rocsparse_datatype       compute_type,
                                         rocsparse::blas_gemm_alg algo,
                                         int32_t                  solution_index,
                                         uint32_t                 flags)
{
    ROCSPARSE_CHECKARG_POINTER(0, blas_handle);
    switch(blas_handle->blas_impl)
    {
    case rocsparse::blas_impl_none:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented,
                                               "no blas implementation is selected.");
    }
    case rocsparse::blas_impl_default:
    case rocsparse::blas_impl_rocblas:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::blas_rocblas_gemm_ex(blas_handle->blas_rocblas_handle,
                                                                  transA,
                                                                  transB,
                                                                  m,
                                                                  n,
                                                                  k,
                                                                  alpha,
                                                                  a,
                                                                  a_type,
                                                                  lda,
                                                                  b,
                                                                  b_type,
                                                                  ldb,
                                                                  beta,
                                                                  c,
                                                                  c_type,
                                                                  ldc,
                                                                  d,
                                                                  d_type,
                                                                  ldd,
                                                                  compute_type,
                                                                  algo,
                                                                  solution_index,
                                                                  flags));
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}
