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

#include "rocsparse_ggthr.hpp"
#include "../level1/rocsparse_gthr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::ggthr(rocsparse_handle     handle_,
                                  int64_t              nnz,
                                  rocsparse_datatype   data_type,
                                  const void*          in,
                                  void*                out,
                                  rocsparse_indextype  perm_type,
                                  void*                perm,
                                  rocsparse_index_base base)
{
    ROCSPARSE_ROUTINE_TRACE;

#define CALL_TEMPLATE(PERM_TYPE, DATA_TYPE)                                    \
    RETURN_IF_ROCSPARSE_ERROR(                                                 \
        (rocsparse::gthr_template<PERM_TYPE, DATA_TYPE>)(handle_,              \
                                                         nnz,                  \
                                                         (const DATA_TYPE*)in, \
                                                         (DATA_TYPE*)out,      \
                                                         (PERM_TYPE*)perm,     \
                                                         base));

#define DISPATCH_DATA_TYPE(PERM_TYPE)                       \
    switch(data_type)                                       \
    {                                                       \
    case rocsparse_datatype_f16_r:                          \
    {                                                       \
        CALL_TEMPLATE(PERM_TYPE, _Float16);                 \
        return rocsparse_status_success;                    \
    }                                                       \
    case rocsparse_datatype_f32_r:                          \
    {                                                       \
        CALL_TEMPLATE(PERM_TYPE, float);                    \
        return rocsparse_status_success;                    \
    }                                                       \
    case rocsparse_datatype_f32_c:                          \
    {                                                       \
        CALL_TEMPLATE(PERM_TYPE, rocsparse_float_complex);  \
        return rocsparse_status_success;                    \
    }                                                       \
    case rocsparse_datatype_f64_r:                          \
    {                                                       \
        CALL_TEMPLATE(PERM_TYPE, double);                   \
        return rocsparse_status_success;                    \
    }                                                       \
    case rocsparse_datatype_f64_c:                          \
    {                                                       \
        CALL_TEMPLATE(PERM_TYPE, rocsparse_double_complex); \
        return rocsparse_status_success;                    \
    }                                                       \
    case rocsparse_datatype_i8_r:                           \
    {                                                       \
        CALL_TEMPLATE(PERM_TYPE, int8_t);                   \
        return rocsparse_status_success;                    \
    }                                                       \
    case rocsparse_datatype_u8_r:                           \
    {                                                       \
        CALL_TEMPLATE(PERM_TYPE, uint8_t);                  \
        return rocsparse_status_success;                    \
    }                                                       \
    case rocsparse_datatype_i32_r:                          \
    {                                                       \
        CALL_TEMPLATE(PERM_TYPE, int32_t);                  \
        return rocsparse_status_success;                    \
    }                                                       \
    case rocsparse_datatype_u32_r:                          \
    {                                                       \
        CALL_TEMPLATE(PERM_TYPE, uint32_t);                 \
        return rocsparse_status_success;                    \
    }                                                       \
    case rocsparse_datatype_bf16_r:                         \
    {                                                       \
        CALL_TEMPLATE(PERM_TYPE, rocsparse_bfloat16);       \
        return rocsparse_status_success;                    \
    }                                                       \
    }                                                       \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value)

    switch(perm_type)
    {
    case rocsparse_indextype_u16:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    case rocsparse_indextype_i32:
    {
        DISPATCH_DATA_TYPE(int32_t);
    }
    case rocsparse_indextype_i64:
    {
        DISPATCH_DATA_TYPE(int64_t);
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP

#undef CALL_TEMPLATE
#undef DISPATCH_DATA_TYPE
}
