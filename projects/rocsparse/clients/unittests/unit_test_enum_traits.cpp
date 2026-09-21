/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

//
// Tier-0 unit tests: compile-time type-trait maps.
//
// These traits are header-only (library/src/include), so this translation unit
// needs no linking against librocsparse at all -- it validates the compile-time
// contract directly. Any regression here is a build failure, not a runtime one.
//
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_indextype_utils.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

// ---------------------------------------------------------------------------
// indextype_traits<v>::type_t
// ---------------------------------------------------------------------------
static_assert(
    std::is_same<rocsparse::indextype_traits<rocsparse_indextype_i32>::type_t, int32_t>::value,
    "indextype_traits<i32> must map to int32_t");
static_assert(
    std::is_same<rocsparse::indextype_traits<rocsparse_indextype_i64>::type_t, int64_t>::value,
    "indextype_traits<i64> must map to int64_t");
static_assert(std::is_same<rocsparse::indextype_traits<deprecated_rocsparse_indextype_u16>::type_t,
                           uint16_t>::value,
              "indextype_traits<u16> must map to uint16_t");

// ---------------------------------------------------------------------------
// datatype_traits<v>::type_t
// ---------------------------------------------------------------------------
static_assert(
    std::is_same<rocsparse::datatype_traits<rocsparse_datatype_f32_r>::type_t, float>::value,
    "datatype_traits<f32_r> must map to float");
static_assert(
    std::is_same<rocsparse::datatype_traits<rocsparse_datatype_f64_r>::type_t, double>::value,
    "datatype_traits<f64_r> must map to double");
static_assert(
    std::is_same<rocsparse::datatype_traits<rocsparse_datatype_i32_r>::type_t, int32_t>::value,
    "datatype_traits<i32_r> must map to int32_t");
static_assert(
    std::is_same<rocsparse::datatype_traits<rocsparse_datatype_u32_r>::type_t, uint32_t>::value,
    "datatype_traits<u32_r> must map to uint32_t");
static_assert(
    std::is_same<rocsparse::datatype_traits<rocsparse_datatype_i8_r>::type_t, int8_t>::value,
    "datatype_traits<i8_r> must map to int8_t");
static_assert(
    std::is_same<rocsparse::datatype_traits<rocsparse_datatype_u8_r>::type_t, uint8_t>::value,
    "datatype_traits<u8_r> must map to uint8_t");
static_assert(std::is_same<rocsparse::datatype_traits<rocsparse_datatype_f32_c>::type_t,
                           rocsparse_float_complex>::value,
              "datatype_traits<f32_c> must map to rocsparse_float_complex");
static_assert(std::is_same<rocsparse::datatype_traits<rocsparse_datatype_f64_c>::type_t,
                           rocsparse_double_complex>::value,
              "datatype_traits<f64_c> must map to rocsparse_double_complex");

// The static_asserts above are the real test (checked at compile time). This
// runtime case simply records the coverage in the gtest report.
TEST(enum_traits, compile_time_type_maps)
{
    SUCCEED();
}
