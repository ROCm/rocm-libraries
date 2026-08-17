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
// Tier-0 unit tests: rocsparse::get_indextype<T>() and indextype_sizeof().
//
// These are host-only functions defined in
//   library/src/common/rocsparse_indextype_utils.cpp
// which is compiled directly into the unit-test binary (they are not exported
// from librocsparse due to hidden visibility). No library changes required.
//
#include "rocsparse_indextype_utils.hpp"

#include <cstdint>
#include <gtest/gtest.h>

TEST(indextype_utils, get_indextype)
{
    EXPECT_EQ(rocsparse::get_indextype<int32_t>(), rocsparse_indextype_i32);
    EXPECT_EQ(rocsparse::get_indextype<int64_t>(), rocsparse_indextype_i64);
    EXPECT_EQ(rocsparse::get_indextype<uint16_t>(), deprecated_rocsparse_indextype_u16);
}

TEST(indextype_utils, indextype_sizeof)
{
    EXPECT_EQ(rocsparse::indextype_sizeof(rocsparse_indextype_i32), sizeof(int32_t));
    EXPECT_EQ(rocsparse::indextype_sizeof(rocsparse_indextype_i64), sizeof(int64_t));
    EXPECT_EQ(rocsparse::indextype_sizeof(deprecated_rocsparse_indextype_u16), sizeof(uint16_t));
}

// Round-trip: the size reported for the enum returned by get_indextype<T>()
// must equal sizeof(T). This is the kind of cross-component invariant that is
// invisible at the public-API level.
TEST(indextype_utils, get_indextype_sizeof_roundtrip)
{
    EXPECT_EQ(rocsparse::indextype_sizeof(rocsparse::get_indextype<int32_t>()), sizeof(int32_t));
    EXPECT_EQ(rocsparse::indextype_sizeof(rocsparse::get_indextype<int64_t>()), sizeof(int64_t));
    EXPECT_EQ(rocsparse::indextype_sizeof(rocsparse::get_indextype<uint16_t>()), sizeof(uint16_t));
}
