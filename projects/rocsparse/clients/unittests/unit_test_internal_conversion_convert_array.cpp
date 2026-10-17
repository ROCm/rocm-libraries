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
// Unit tests for the host dispatch of library/src/conversion/rocsparse_convert_array.cpp
// (already compiled into rocsparse-unit-test-device):
//
//   - every (target, source) numerical datatype pair of
//     convert_array(handle, n, rocsparse_datatype, ...), with n > 0 and n == 0,
//   - the same-type copy / in-place shortcuts of all convert_array overloads,
//   - out-of-range index conversions on the strided overload,
//   - rocsparse::dnvec_transfer_from argument checks and real/complex rules.
//
// Arrays are a handful of elements; this needs a GPU only because the dispatch
// launches the conversion kernels and reads back the error scalars.
//
#include "unit_test_utils.hpp"

#include "../../library/src/conversion/rocsparse_convert_array.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    using ConvertArray = HandleTest;

    constexpr rocsparse_datatype all_datatypes[] = {rocsparse_datatype_i8_r,
                                                    rocsparse_datatype_u8_r,
                                                    rocsparse_datatype_i32_r,
                                                    rocsparse_datatype_u32_r,
                                                    rocsparse_datatype_f16_r,
                                                    rocsparse_datatype_bf16_r,
                                                    rocsparse_datatype_f32_r,
                                                    rocsparse_datatype_f64_r,
                                                    rocsparse_datatype_f32_c,
                                                    rocsparse_datatype_f64_c};

    // Large enough for nitems elements of the widest type (double complex).
    constexpr size_t nitems    = 4;
    constexpr size_t max_bytes = nitems * 16;
}

TEST_F(ConvertArray, datatype_matrix)
{
    device_vector<uint8_t> source(std::vector<uint8_t>(max_bytes, 0));
    device_vector<uint8_t> target(max_bytes);
    ASSERT_NE(source.ptr, nullptr);
    ASSERT_NE(target.ptr, nullptr);

    for(const auto t : all_datatypes)
    {
        for(const auto s : all_datatypes)
        {
            EXPECT_EQ(rocsparse::convert_array(handle, nitems, t, target.ptr, s, source.ptr),
                      rocsparse_status_success)
                << "target " << t << " source " << s;
            EXPECT_EQ(rocsparse::convert_array(handle, 0, t, target.ptr, s, source.ptr),
                      rocsparse_status_success)
                << "empty, target " << t << " source " << s;
        }
    }
}

TEST_F(ConvertArray, datatype_values)
{
    const std::vector<float> hf{1.0f, -2.0f, 3.5f, 0.25f};
    device_vector<float>     f32(hf);
    device_vector<double>    f64(nitems);

    ASSERT_EQ(
        rocsparse::convert_array(
            handle, nitems, rocsparse_datatype_f64_r, f64.ptr, rocsparse_datatype_f32_r, f32.ptr),
        rocsparse_status_success);
    EXPECT_EQ(to_host(f64), (std::vector<double>{1.0, -2.0, 3.5, 0.25}));

    device_vector<float> back(nitems);
    ASSERT_EQ(
        rocsparse::convert_array(
            handle, nitems, rocsparse_datatype_f32_r, back.ptr, rocsparse_datatype_f64_r, f64.ptr),
        rocsparse_status_success);
    EXPECT_EQ(to_host(back), hf);

    device_vector<rocsparse_float_complex> c32(nitems);
    ASSERT_EQ(
        rocsparse::convert_array(
            handle, nitems, rocsparse_datatype_f32_c, c32.ptr, rocsparse_datatype_f32_r, f32.ptr),
        rocsparse_status_success);
    const auto hc32 = to_host(c32);
    for(size_t i = 0; i < nitems; ++i)
    {
        EXPECT_EQ(std::real(hc32[i]), hf[i]);
        EXPECT_EQ(std::imag(hc32[i]), 0.0f);
    }

    device_vector<rocsparse_double_complex> c64(nitems);
    ASSERT_EQ(
        rocsparse::convert_array(
            handle, nitems, rocsparse_datatype_f64_c, c64.ptr, rocsparse_datatype_f32_c, c32.ptr),
        rocsparse_status_success);
    const auto hc64 = to_host(c64);
    for(size_t i = 0; i < nitems; ++i)
    {
        EXPECT_EQ(std::real(hc64[i]), static_cast<double>(hf[i]));
    }
}

TEST_F(ConvertArray, same_type_copy_and_in_place)
{
    const std::vector<int32_t> h{1, 2, 3, 4};
    device_vector<int32_t>     a(h);
    device_vector<int32_t>     b(nitems);

    // Numerical overload: copy, then in place.
    ASSERT_EQ(rocsparse::convert_array(
                  handle, nitems, rocsparse_datatype_i32_r, b.ptr, rocsparse_datatype_i32_r, a.ptr),
              rocsparse_status_success);
    EXPECT_EQ(to_host(b), h);
    EXPECT_EQ(rocsparse::convert_array(
                  handle, nitems, rocsparse_datatype_i32_r, a.ptr, rocsparse_datatype_i32_r, a.ptr),
              rocsparse_status_success);

    // Index-base overload with matching base: copy, then in place.
    UT_CHECK_HIP(hipMemset(b.ptr, 0, nitems * sizeof(int32_t)));
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       nitems,
                                       rocsparse_indextype_i32,
                                       b.ptr,
                                       rocsparse_index_base_zero,
                                       rocsparse_indextype_i32,
                                       a.ptr,
                                       rocsparse_index_base_zero),
              rocsparse_status_success);
    EXPECT_EQ(to_host(b), h);
    EXPECT_EQ(rocsparse::convert_array(handle,
                                       nitems,
                                       rocsparse_indextype_i32,
                                       a.ptr,
                                       rocsparse_index_base_zero,
                                       rocsparse_indextype_i32,
                                       a.ptr,
                                       rocsparse_index_base_zero),
              rocsparse_status_success);

    // Unit-stride overload: copy, then in place.
    UT_CHECK_HIP(hipMemset(b.ptr, 0, nitems * sizeof(int32_t)));
    ASSERT_EQ(rocsparse::convert_array(
                  handle, nitems, rocsparse_indextype_i32, b.ptr, rocsparse_indextype_i32, a.ptr),
              rocsparse_status_success);
    EXPECT_EQ(to_host(b), h);
    EXPECT_EQ(rocsparse::convert_array(
                  handle, nitems, rocsparse_indextype_i32, a.ptr, rocsparse_indextype_i32, a.ptr),
              rocsparse_status_success);
}

TEST_F(ConvertArray, indexbase_change_all_index_types)
{
    const std::vector<int64_t> h64{0, 1, 2, 3};
    device_vector<int64_t>     s64(h64);
    device_vector<int32_t>     t32(nitems);

    // i64 source, zero based -> i32 target, one based.
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       nitems,
                                       rocsparse_indextype_i32,
                                       t32.ptr,
                                       rocsparse_index_base_one,
                                       rocsparse_indextype_i64,
                                       s64.ptr,
                                       rocsparse_index_base_zero),
              rocsparse_status_success);
    EXPECT_EQ(to_host(t32), (std::vector<int32_t>{1, 2, 3, 4}));

    // i32 one based -> i64 zero based.
    device_vector<int64_t> t64(nitems);
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       nitems,
                                       rocsparse_indextype_i64,
                                       t64.ptr,
                                       rocsparse_index_base_zero,
                                       rocsparse_indextype_i32,
                                       t32.ptr,
                                       rocsparse_index_base_one),
              rocsparse_status_success);
    EXPECT_EQ(to_host(t64), h64);

    // i64 -> i64 with a base change goes through the compute path too.
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       nitems,
                                       rocsparse_indextype_i64,
                                       t64.ptr,
                                       rocsparse_index_base_one,
                                       rocsparse_indextype_i64,
                                       s64.ptr,
                                       rocsparse_index_base_zero),
              rocsparse_status_success);
    EXPECT_EQ(to_host(t64), (std::vector<int64_t>{1, 2, 3, 4}));

    // Empty arrays return early.
    EXPECT_EQ(rocsparse::convert_array(handle,
                                       0,
                                       rocsparse_indextype_i32,
                                       t32.ptr,
                                       rocsparse_index_base_one,
                                       rocsparse_indextype_i64,
                                       s64.ptr,
                                       rocsparse_index_base_zero),
              rocsparse_status_success);
}

TEST_F(ConvertArray, strided_index_conversion)
{
    const std::vector<int32_t> h32{10, -1, 20, -1, 30, -1, 40, -1};
    device_vector<int32_t>     s32(h32);
    device_vector<int64_t>     t64(std::vector<int64_t>(nitems, 0));

    // Source stride 2 -> dense i64 target.
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       nitems,
                                       rocsparse_indextype_i64,
                                       t64.ptr,
                                       1,
                                       rocsparse_indextype_i32,
                                       s32.ptr,
                                       2),
              rocsparse_status_success);
    EXPECT_EQ(to_host(t64), (std::vector<int64_t>{10, 20, 30, 40}));

    // Same index type but non-unit stride uses the compute path.
    device_vector<int32_t> t32(std::vector<int32_t>(nitems, 0));
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       nitems,
                                       rocsparse_indextype_i32,
                                       t32.ptr,
                                       1,
                                       rocsparse_indextype_i32,
                                       s32.ptr,
                                       2),
              rocsparse_status_success);
    EXPECT_EQ(to_host(t32), (std::vector<int32_t>{10, 20, 30, 40}));

    // i64 -> i32 unit-stride (6-argument overload).
    ASSERT_EQ(
        rocsparse::convert_array(
            handle, nitems, rocsparse_indextype_i32, t32.ptr, rocsparse_indextype_i64, t64.ptr),
        rocsparse_status_success);
    EXPECT_EQ(to_host(t32), (std::vector<int32_t>{10, 20, 30, 40}));

    EXPECT_EQ(
        rocsparse::convert_array(
            handle, 0, rocsparse_indextype_i32, t32.ptr, 1, rocsparse_indextype_i64, t64.ptr, 1),
        rocsparse_status_success);
}

TEST_F(ConvertArray, strided_index_out_of_range)
{
    const int64_t              big = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    const std::vector<int64_t> h64{1, big, 2, -big - 1};
    device_vector<int64_t>     s64(h64);
    device_vector<int32_t>     t32(nitems);

    EXPECT_EQ(rocsparse::convert_array(handle,
                                       nitems,
                                       rocsparse_indextype_i32,
                                       t32.ptr,
                                       1,
                                       rocsparse_indextype_i64,
                                       s64.ptr,
                                       1),
              rocsparse_status_type_mismatch);
}

TEST_F(ConvertArray, dnvec_transfer_from)
{
    const std::vector<float>                hf{1.0f, 2.0f, 3.0f, 4.0f};
    device_vector<float>                    f32(hf);
    device_vector<double>                   f64(nitems);
    device_vector<int32_t>                  i32(std::vector<int32_t>(nitems, 0));
    device_vector<rocsparse_float_complex>  c32(nitems);
    device_vector<rocsparse_double_complex> c64(nitems);

    rocsparse_dnvec_descr vf32 = nullptr, vf64 = nullptr, vi32 = nullptr, vc32 = nullptr,
                          vc64 = nullptr, vshort = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vf32, nitems, f32.ptr, rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vf64, nitems, f64.ptr, rocsparse_datatype_f64_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vi32, nitems, i32.ptr, rocsparse_datatype_i32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vc32, nitems, c32.ptr, rocsparse_datatype_f32_c),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vc64, nitems, c64.ptr, rocsparse_datatype_f64_c),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&vshort, nitems - 1, f32.ptr, rocsparse_datatype_f32_r),
              rocsparse_status_success);

    EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, nullptr, vf32),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, vf64, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, vf64, vshort), rocsparse_status_invalid_size);

    // Real target from complex source is rejected.
    EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, vf64, vc32), rocsparse_status_not_implemented);
    EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, vf32, vc64), rocsparse_status_not_implemented);

    // Real from real.
    ASSERT_EQ(rocsparse::dnvec_transfer_from(handle, vf64, vf32), rocsparse_status_success);
    EXPECT_EQ(to_host(f64), (std::vector<double>{1.0, 2.0, 3.0, 4.0}));
    EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, vf32, vi32), rocsparse_status_success);

    // Complex from real, then complex from complex.
    ASSERT_EQ(rocsparse::dnvec_transfer_from(handle, vc32, vf64), rocsparse_status_success);
    ASSERT_EQ(rocsparse::dnvec_transfer_from(handle, vc64, vc32), rocsparse_status_success);
    const auto hc64 = to_host(c64);
    for(size_t i = 0; i < nitems; ++i)
    {
        EXPECT_EQ(std::real(hc64[i]), static_cast<double>(i + 1));
        EXPECT_EQ(std::imag(hc64[i]), 0.0);
    }

    for(auto d : {vf32, vf64, vi32, vc32, vc64, vshort})
    {
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(d), rocsparse_status_success);
    }
}
