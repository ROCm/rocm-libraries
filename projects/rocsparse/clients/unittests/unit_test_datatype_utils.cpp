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
// Tier-0 unit tests: rocsparse::get_datatype<T>() and datatype_sizeof().
//
// Host-only functions defined in
//   library/src/common/rocsparse_datatype_utils.cpp
// compiled directly into the unit-test binary. No library changes required.
//
#include "rocsparse_datatype_utils.hpp"

#include <cstdint>
#include <gtest/gtest.h>

TEST(datatype_utils, get_datatype)
{
    EXPECT_EQ(rocsparse::get_datatype<float>(), rocsparse_datatype_f32_r);
    EXPECT_EQ(rocsparse::get_datatype<double>(), rocsparse_datatype_f64_r);
    EXPECT_EQ(rocsparse::get_datatype<rocsparse_float_complex>(), rocsparse_datatype_f32_c);
    EXPECT_EQ(rocsparse::get_datatype<rocsparse_double_complex>(), rocsparse_datatype_f64_c);
}

TEST(datatype_utils, datatype_sizeof_real)
{
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse_datatype_f32_r), sizeof(float));
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse_datatype_f64_r), sizeof(double));
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse_datatype_i32_r), sizeof(int32_t));
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse_datatype_u32_r), sizeof(uint32_t));
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse_datatype_i8_r), sizeof(int8_t));
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse_datatype_u8_r), sizeof(uint8_t));
}

TEST(datatype_utils, datatype_sizeof_complex)
{
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse_datatype_f32_c),
              sizeof(rocsparse_float_complex));
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse_datatype_f64_c),
              sizeof(rocsparse_double_complex));
    // A complex value is exactly two real components wide.
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse_datatype_f32_c),
              2 * rocsparse::datatype_sizeof(rocsparse_datatype_f32_r));
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse_datatype_f64_c),
              2 * rocsparse::datatype_sizeof(rocsparse_datatype_f64_r));
}

// Round-trip invariant between the two host helpers.
TEST(datatype_utils, get_datatype_sizeof_roundtrip)
{
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse::get_datatype<float>()), sizeof(float));
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse::get_datatype<double>()), sizeof(double));
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse::get_datatype<rocsparse_float_complex>()),
              sizeof(rocsparse_float_complex));
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse::get_datatype<rocsparse_double_complex>()),
              sizeof(rocsparse_double_complex));
}
