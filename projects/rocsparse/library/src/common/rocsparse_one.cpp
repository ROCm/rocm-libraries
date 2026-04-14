/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_one.hpp"
#include "rocsparse_handle.hpp"

rocsparse_status rocsparse::set_minus_one_async(rocsparse_pointer_mode pointer_mode,
                                                rocsparse_indextype    data_indextype,
                                                void*                  data,
                                                hipStream_t            stream)
{
    switch(pointer_mode)
    {
    case rocsparse_pointer_mode_device:
    {
        RETURN_IF_HIP_ERROR(
            hipMemsetAsync(data, 0xFF, rocsparse::indextype_sizeof(data_indextype), stream));
        return rocsparse_status_success;
    }
    case rocsparse_pointer_mode_host:
    {
        memset(data, 0xFF, rocsparse::indextype_sizeof(data_indextype));
        return rocsparse_status_success;
        // LCOV_EXCL_START
    }
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

rocsparse_status rocsparse::set_minus_one_async(rocsparse_pointer_mode pointer_mode,
                                                rocsparse_indextype    data_indextype,
                                                int64_t                data_size,
                                                void*                  data,
                                                hipStream_t            stream)
{
    switch(pointer_mode)
    {
    case rocsparse_pointer_mode_device:
    {
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            data, 0xFF, rocsparse::indextype_sizeof(data_indextype) * data_size, stream));
        return rocsparse_status_success;
    }
    case rocsparse_pointer_mode_host:
    {
        memset(data, 0xFF, rocsparse::indextype_sizeof(data_indextype) * data_size);
        return rocsparse_status_success;
        // LCOV_EXCL_START
    }
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

void rocsparse::one(const rocsparse_handle handle, float** one)
{
    *one = (float*)handle->sone;
}

void rocsparse::one(const rocsparse_handle handle, double** one)
{
    *one = (double*)handle->done;
}

void rocsparse::one(const rocsparse_handle handle, rocsparse_float_complex** one)
{
    *one = (rocsparse_float_complex*)handle->sone;
}

void rocsparse::one(const rocsparse_handle handle, rocsparse_double_complex** one)
{
    *one = (rocsparse_double_complex*)handle->done;
}
