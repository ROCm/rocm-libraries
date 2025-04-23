/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef MIGRAPHX_GUARD_C_API_MIGRAPHX_H
#define MIGRAPHX_GUARD_C_API_MIGRAPHX_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#include <migraphx/api/export.h>

// Add new types here
// clang-format off
#define MIGRAPHX_SHAPE_VISIT_TYPES(m) \
    m(bool_type, bool) \
    m(half_type, half) \
    m(float_type, float) \
    m(double_type, double) \
    m(uint8_type, uint8_t) \
    m(int8_type, int8_t) \
    m(uint16_type, uint16_t) \
    m(int16_type, int16_t) \
    m(int32_type, int32_t) \
    m(int64_type, int64_t) \
    m(uint32_type, uint32_t) \
    m(uint64_type, uint64_t) \
    m(fp8e4m3fnuz_type, migraphx::fp8::fp8e4m3fnuz) \
    m(fp8e4m3fn_type, migraphx::fp8::fp8e4m3fn) \
    m(fp8e5m2_type, migraphx::fp8::fp8e5m2) \
    m(bf16_type, bf16) \
    m(fp8e5m2fnuz_type, migraphx::fp8::fp8e5m2fnuz)
// clang-format on

#ifdef __cplusplus
extern "C" {
#endif

// return code, more to be added later
typedef enum
{
    migraphx_status_success        = 0,
    migraphx_status_bad_param      = 1,
    migraphx_status_unknown_target = 3,
    migraphx_status_unknown_error  = 4,

} migraphx_status;

#define MIGRAPHX_SHAPE_GENERATE_ENUM_TYPES(x, t) migraphx_shape_##x,
/// An enum to represent the different data type inputs
typedef enum
{
    migraphx_shape_tuple_type,
    MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_SHAPE_GENERATE_ENUM_TYPES)
} migraphx_shape_datatype_t;
#undef MIGRAPHX_SHAPE_GENERATE_ENUM_TYPES

<%
    generate_c_header()
%>

#ifdef __cplusplus
}
#endif

#endif
