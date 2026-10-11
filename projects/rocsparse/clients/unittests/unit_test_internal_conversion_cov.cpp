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
// Device unit tests for conversion translation units the quick/pre_checkin
// rocsparse-test filter never executes:
//   ell2dense / coo2dense_aos templates (also reached from sddmm alg_dense),
//   ggthr and gcoosort generic dispatch (no in-library caller),
//   convert_array index and numeric type switches,
//   csxsldu_compute direction / diag branches,
//   gell2csr via rocsparse_sparse_to_sparse (ELL -> CSR).
//
#include "unit_test_utils.hpp"

#include "rocsparse_convert_array.hpp"
#include "rocsparse_coo2dense_aos.hpp"
#include "rocsparse_csxsldu.hpp"
#include "rocsparse_ell2dense.hpp"
#include "rocsparse_gcoosort.hpp"
#include "rocsparse_ggthr.hpp"

#include <complex>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    struct MatDescr
    {
        rocsparse_mat_descr d = nullptr;
        MatDescr()
        {
            (void)rocsparse_create_mat_descr(&d);
        }
        ~MatDescr()
        {
            if(d)
                (void)rocsparse_destroy_mat_descr(d);
        }
    };

    template <typename T>
    void expect_equal(const std::vector<T>& got, const std::vector<T>& exp)
    {
        ASSERT_EQ(got.size(), exp.size());
        for(size_t i = 0; i < got.size(); ++i)
        {
            if constexpr(std::is_same_v<T, rocsparse_float_complex>
                         || std::is_same_v<T, rocsparse_double_complex>)
            {
                EXPECT_FLOAT_EQ(std::real(got[i]), std::real(exp[i])) << "index " << i;
                EXPECT_FLOAT_EQ(std::imag(got[i]), std::imag(exp[i])) << "index " << i;
            }
            else if constexpr(std::is_same_v<T, _Float16> || std::is_same_v<T, rocsparse_bfloat16>)
            {
                EXPECT_FLOAT_EQ(static_cast<float>(got[i]), static_cast<float>(exp[i]))
                    << "index " << i;
            }
            else
            {
                EXPECT_EQ(got[i], exp[i]) << "index " << i;
            }
        }
    }
} // namespace

class ConversionCov : public HandleTest
{
};

// ---------------------------------------------------------------------------
// ell2dense: column and row order, plus the quick-return and argument checks
// the sddmm alg_dense caller does not take (it always passes column order).
// ---------------------------------------------------------------------------
TEST_F(ConversionCov, ell2dense)
{
    MatDescr descr;
    ASSERT_NE(descr.d, nullptr);

    // Off-diagonal ELL so column-major and row-major dense layouts differ.
    // width 1, rows {col 1 = 3, col 0 = 4}.
    device_vector<int32_t> col{std::vector<int32_t>{1, 0}};
    device_vector<float>   val{std::vector<float>{3.0f, 4.0f}};
    device_vector<float>   dense{size_t(4)};
    ASSERT_TRUE(col.ptr && val.ptr && dense.ptr);

    ASSERT_EQ((rocsparse::ell2dense_template<int32_t, float>(handle,
                                                            2,
                                                            2,
                                                            descr.d,
                                                            1,
                                                            val.ptr,
                                                            col.ptr,
                                                            dense.ptr,
                                                            2,
                                                            rocsparse_order_column)),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    expect_equal(to_host(dense.ptr, 4), std::vector<float>{0.0f, 4.0f, 3.0f, 0.0f});

    ASSERT_EQ(hipMemset(dense.ptr, 0x7f, sizeof(float) * 4), hipSuccess);
    ASSERT_EQ((rocsparse::ell2dense_template<int32_t, float>(handle,
                                                            2,
                                                            2,
                                                            descr.d,
                                                            1,
                                                            val.ptr,
                                                            col.ptr,
                                                            dense.ptr,
                                                            2,
                                                            rocsparse_order_row)),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    expect_equal(to_host(dense.ptr, 4), std::vector<float>{0.0f, 3.0f, 4.0f, 0.0f});

    // int64 index instantiation, column order, same pattern.
    device_vector<int64_t> col64{std::vector<int64_t>{1, 0}};
    ASSERT_TRUE(col64.ptr);
    ASSERT_EQ((rocsparse::ell2dense_template<int64_t, float>(handle,
                                                            2,
                                                            2,
                                                            descr.d,
                                                            1,
                                                            val.ptr,
                                                            col64.ptr,
                                                            dense.ptr,
                                                            2,
                                                            rocsparse_order_column)),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    expect_equal(to_host(dense.ptr, 4), std::vector<float>{0.0f, 4.0f, 3.0f, 0.0f});

    // Quick returns do not touch the dense array.
    EXPECT_EQ((rocsparse::ell2dense_template<int32_t, float>(handle,
                                                            0,
                                                            2,
                                                            descr.d,
                                                            1,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            1,
                                                            rocsparse_order_column)),
              rocsparse_status_success);
    EXPECT_EQ((rocsparse::ell2dense_template<int32_t, float>(handle,
                                                            2,
                                                            0,
                                                            descr.d,
                                                            1,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            2,
                                                            rocsparse_order_column)),
              rocsparse_status_success);
    EXPECT_EQ((rocsparse::ell2dense_template<int32_t, float>(handle,
                                                            2,
                                                            2,
                                                            descr.d,
                                                            0,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            2,
                                                            rocsparse_order_column)),
              rocsparse_status_success);

    EXPECT_EQ((rocsparse::ell2dense_template<int32_t, float>(nullptr,
                                                            2,
                                                            2,
                                                            descr.d,
                                                            1,
                                                            val.ptr,
                                                            col.ptr,
                                                            dense.ptr,
                                                            2,
                                                            rocsparse_order_column)),
              rocsparse_status_invalid_handle);
    EXPECT_EQ((rocsparse::ell2dense_template<int32_t, float>(handle,
                                                            2,
                                                            2,
                                                            nullptr,
                                                            1,
                                                            val.ptr,
                                                            col.ptr,
                                                            dense.ptr,
                                                            2,
                                                            rocsparse_order_column)),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_set_mat_type(descr.d, rocsparse_matrix_type_symmetric),
              rocsparse_status_success);
    EXPECT_EQ((rocsparse::ell2dense_template<int32_t, float>(handle,
                                                            2,
                                                            2,
                                                            descr.d,
                                                            1,
                                                            val.ptr,
                                                            col.ptr,
                                                            dense.ptr,
                                                            2,
                                                            rocsparse_order_column)),
              rocsparse_status_not_implemented);
    EXPECT_EQ(rocsparse_set_mat_type(descr.d, rocsparse_matrix_type_general),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_set_mat_storage_mode(descr.d, rocsparse_storage_mode_unsorted),
              rocsparse_status_success);
    EXPECT_EQ((rocsparse::ell2dense_template<int32_t, float>(handle,
                                                            2,
                                                            2,
                                                            descr.d,
                                                            1,
                                                            val.ptr,
                                                            col.ptr,
                                                            dense.ptr,
                                                            2,
                                                            rocsparse_order_column)),
              rocsparse_status_requires_sorted_storage);
    EXPECT_EQ(rocsparse_set_mat_storage_mode(descr.d, rocsparse_storage_mode_sorted),
              rocsparse_status_success);
    EXPECT_EQ((rocsparse::ell2dense_template<int32_t, float>(
                  handle, 2, 2, descr.d, 1, val.ptr, col.ptr, dense.ptr, 2, (rocsparse_order)99)),
              rocsparse_status_invalid_value);
    EXPECT_EQ((rocsparse::ell2dense_template<int32_t, float>(handle,
                                                            -1,
                                                            2,
                                                            descr.d,
                                                            1,
                                                            val.ptr,
                                                            col.ptr,
                                                            dense.ptr,
                                                            2,
                                                            rocsparse_order_column)),
              rocsparse_status_invalid_size);
    EXPECT_EQ((rocsparse::ell2dense_template<int32_t, float>(handle,
                                                            2,
                                                            2,
                                                            descr.d,
                                                            1,
                                                            val.ptr,
                                                            col.ptr,
                                                            dense.ptr,
                                                            1,
                                                            rocsparse_order_column)),
              rocsparse_status_invalid_size);
    EXPECT_EQ((rocsparse::ell2dense_template<int32_t, float>(
                  handle, 2, 2, descr.d, 1, val.ptr, col.ptr, nullptr, 2, rocsparse_order_column)),
              rocsparse_status_invalid_pointer);
}

// ---------------------------------------------------------------------------
// coo2dense_aos: interleaved (row, col) pairs, both orders.
// ---------------------------------------------------------------------------
TEST_F(ConversionCov, coo2dense_aos)
{
    MatDescr descr;
    ASSERT_NE(descr.d, nullptr);

    device_vector<int32_t> ind{std::vector<int32_t>{0, 1, 1, 0}};
    device_vector<float>   val{std::vector<float>{3.0f, 4.0f}};
    device_vector<float>   dense{size_t(4)};
    ASSERT_TRUE(ind.ptr && val.ptr && dense.ptr);

    ASSERT_EQ((rocsparse::coo2dense_aos_template<int32_t, float>(handle,
                                                                2,
                                                                2,
                                                                2,
                                                                descr.d,
                                                                val.ptr,
                                                                ind.ptr,
                                                                dense.ptr,
                                                                2,
                                                                rocsparse_order_column)),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    expect_equal(to_host(dense.ptr, 4), std::vector<float>{0.0f, 4.0f, 3.0f, 0.0f});

    ASSERT_EQ((rocsparse::coo2dense_aos_template<int32_t, float>(handle,
                                                                2,
                                                                2,
                                                                2,
                                                                descr.d,
                                                                val.ptr,
                                                                ind.ptr,
                                                                dense.ptr,
                                                                2,
                                                                rocsparse_order_row)),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    expect_equal(to_host(dense.ptr, 4), std::vector<float>{0.0f, 3.0f, 4.0f, 0.0f});

    device_vector<int64_t> ind64{std::vector<int64_t>{0, 1, 1, 0}};
    ASSERT_TRUE(ind64.ptr);
    ASSERT_EQ((rocsparse::coo2dense_aos_template<int64_t, float>(handle,
                                                                2,
                                                                2,
                                                                2,
                                                                descr.d,
                                                                val.ptr,
                                                                ind64.ptr,
                                                                dense.ptr,
                                                                2,
                                                                rocsparse_order_column)),
              rocsparse_status_success);

    EXPECT_EQ((rocsparse::coo2dense_aos_template<int32_t, float>(handle,
                                                                0,
                                                                2,
                                                                2,
                                                                descr.d,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                1,
                                                                rocsparse_order_column)),
              rocsparse_status_success);
    EXPECT_EQ((rocsparse::coo2dense_aos_template<int32_t, float>(handle,
                                                                2,
                                                                2,
                                                                0,
                                                                descr.d,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                2,
                                                                rocsparse_order_column)),
              rocsparse_status_success);
    EXPECT_EQ((rocsparse::coo2dense_aos_template<int32_t, float>(nullptr,
                                                                2,
                                                                2,
                                                                2,
                                                                descr.d,
                                                                val.ptr,
                                                                ind.ptr,
                                                                dense.ptr,
                                                                2,
                                                                rocsparse_order_column)),
              rocsparse_status_invalid_handle);
    EXPECT_EQ((rocsparse::coo2dense_aos_template<int32_t, float>(handle,
                                                                2,
                                                                2,
                                                                2,
                                                                nullptr,
                                                                val.ptr,
                                                                ind.ptr,
                                                                dense.ptr,
                                                                2,
                                                                rocsparse_order_column)),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_set_mat_type(descr.d, rocsparse_matrix_type_hermitian),
              rocsparse_status_success);
    EXPECT_EQ((rocsparse::coo2dense_aos_template<int32_t, float>(handle,
                                                                2,
                                                                2,
                                                                2,
                                                                descr.d,
                                                                val.ptr,
                                                                ind.ptr,
                                                                dense.ptr,
                                                                2,
                                                                rocsparse_order_column)),
              rocsparse_status_not_implemented);
    EXPECT_EQ(rocsparse_set_mat_type(descr.d, rocsparse_matrix_type_general),
              rocsparse_status_success);
    EXPECT_EQ((rocsparse::coo2dense_aos_template<int32_t, float>(handle,
                                                                2,
                                                                2,
                                                                2,
                                                                descr.d,
                                                                val.ptr,
                                                                ind.ptr,
                                                                dense.ptr,
                                                                1,
                                                                rocsparse_order_column)),
              rocsparse_status_invalid_size);
    EXPECT_EQ((rocsparse::coo2dense_aos_template<int32_t, float>(
                  handle, 2, 2, 2, descr.d, val.ptr, ind.ptr, nullptr, 2, rocsparse_order_column)),
              rocsparse_status_invalid_pointer);
}

// ---------------------------------------------------------------------------
// ggthr: every non-excluded datatype x i32/i64 permutation.
// ---------------------------------------------------------------------------
namespace
{
    template <typename T>
    void check_ggthr(rocsparse_handle handle, rocsparse_datatype dt, rocsparse_indextype perm_type)
    {
        const std::vector<T> host_in{scalar<T>(10), scalar<T>(20), scalar<T>(30), scalar<T>(40)};
        device_vector<T>     in{host_in};
        device_vector<T>     out{size_t(2)};
        ASSERT_TRUE(in.ptr && out.ptr);

        if(perm_type == rocsparse_indextype_i32)
        {
            device_vector<int32_t> perm{std::vector<int32_t>{2, 0}};
            ASSERT_TRUE(perm.ptr);
            ASSERT_EQ(rocsparse::ggthr(
                          handle, 2, dt, in.ptr, out.ptr, perm_type, perm.ptr, rocsparse_index_base_zero),
                      rocsparse_status_success);
        }
        else
        {
            device_vector<int64_t> perm{std::vector<int64_t>{2, 0}};
            ASSERT_TRUE(perm.ptr);
            ASSERT_EQ(rocsparse::ggthr(
                          handle, 2, dt, in.ptr, out.ptr, perm_type, perm.ptr, rocsparse_index_base_zero),
                      rocsparse_status_success);
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_equal(to_host(out.ptr, 2), std::vector<T>{scalar<T>(30), scalar<T>(10)});
    }
} // namespace

TEST_F(ConversionCov, ggthr)
{
    check_ggthr<float>(handle, rocsparse_datatype_f32_r, rocsparse_indextype_i32);
    check_ggthr<float>(handle, rocsparse_datatype_f32_r, rocsparse_indextype_i64);
    check_ggthr<double>(handle, rocsparse_datatype_f64_r, rocsparse_indextype_i32);
    check_ggthr<double>(handle, rocsparse_datatype_f64_r, rocsparse_indextype_i64);
    check_ggthr<rocsparse_float_complex>(handle, rocsparse_datatype_f32_c, rocsparse_indextype_i32);
    check_ggthr<rocsparse_float_complex>(handle, rocsparse_datatype_f32_c, rocsparse_indextype_i64);
    check_ggthr<rocsparse_double_complex>(handle, rocsparse_datatype_f64_c, rocsparse_indextype_i32);
    check_ggthr<rocsparse_double_complex>(handle, rocsparse_datatype_f64_c, rocsparse_indextype_i64);
    check_ggthr<int8_t>(handle, rocsparse_datatype_i8_r, rocsparse_indextype_i32);
    check_ggthr<int8_t>(handle, rocsparse_datatype_i8_r, rocsparse_indextype_i64);
    check_ggthr<uint8_t>(handle, rocsparse_datatype_u8_r, rocsparse_indextype_i32);
    check_ggthr<uint8_t>(handle, rocsparse_datatype_u8_r, rocsparse_indextype_i64);
    check_ggthr<int32_t>(handle, rocsparse_datatype_i32_r, rocsparse_indextype_i32);
    check_ggthr<int32_t>(handle, rocsparse_datatype_i32_r, rocsparse_indextype_i64);
    check_ggthr<uint32_t>(handle, rocsparse_datatype_u32_r, rocsparse_indextype_i32);
    check_ggthr<uint32_t>(handle, rocsparse_datatype_u32_r, rocsparse_indextype_i64);
    check_ggthr<_Float16>(handle, rocsparse_datatype_f16_r, rocsparse_indextype_i32);
    check_ggthr<_Float16>(handle, rocsparse_datatype_f16_r, rocsparse_indextype_i64);
    check_ggthr<rocsparse_bfloat16>(handle, rocsparse_datatype_bf16_r, rocsparse_indextype_i32);
    check_ggthr<rocsparse_bfloat16>(handle, rocsparse_datatype_bf16_r, rocsparse_indextype_i64);

    // nnz == 0 takes the quick return inside gthr after the datatype switch.
    EXPECT_EQ(rocsparse::ggthr(handle,
                               0,
                               rocsparse_datatype_f32_r,
                               nullptr,
                               nullptr,
                               rocsparse_indextype_i32,
                               nullptr,
                               rocsparse_index_base_zero),
              rocsparse_status_success);

    // Invalid datatype is the non-excluded fall-through of the data switch.
    device_vector<float>   in{std::vector<float>{1.0f}};
    device_vector<float>   out{size_t(1)};
    device_vector<int32_t> perm{std::vector<int32_t>{0}};
    EXPECT_EQ(rocsparse::ggthr(handle,
                               1,
                               (rocsparse_datatype)999,
                               in.ptr,
                               out.ptr,
                               rocsparse_indextype_i32,
                               perm.ptr,
                               rocsparse_index_base_zero),
              rocsparse_status_invalid_value);
}

// ---------------------------------------------------------------------------
// gcoosort: i32 and i64, by row (with perm) and by column (null perm), nnz 0.
// ---------------------------------------------------------------------------
namespace
{
    template <typename J>
    void check_gcoosort(rocsparse_handle handle, rocsparse_indextype idx)
    {
        device_vector<J> rows{std::vector<J>{J(1), J(0), J(1)}};
        device_vector<J> cols{std::vector<J>{J(2), J(1), J(0)}};
        device_vector<J> perm{size_t(3)};
        ASSERT_TRUE(rows.ptr && cols.ptr && perm.ptr);

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse::gcoosort_buffer_size(
                      handle, 2, 3, 3, idx, rows.ptr, cols.ptr, &buffer_size),
                  rocsparse_status_success);
        ASSERT_GT(buffer_size, size_t(0));
        device_vector<char> buffer{buffer_size};
        ASSERT_TRUE(buffer.ptr);

        ASSERT_EQ(rocsparse::gcoosort_by_row(
                      handle, 2, 3, 3, idx, rows.ptr, cols.ptr, perm.ptr, buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_equal(to_host(rows.ptr, 3), std::vector<J>{J(0), J(1), J(1)});
        expect_equal(to_host(cols.ptr, 3), std::vector<J>{J(1), J(0), J(2)});

        // Fresh unsorted pair, null permutation (the other coosort branch).
        device_vector<J> rows2{std::vector<J>{J(1), J(0), J(1)}};
        device_vector<J> cols2{std::vector<J>{J(2), J(1), J(0)}};
        ASSERT_TRUE(rows2.ptr && cols2.ptr);
        ASSERT_EQ(rocsparse::gcoosort_by_column(
                      handle, 2, 3, 3, idx, rows2.ptr, cols2.ptr, nullptr, buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_equal(to_host(rows2.ptr, 3), std::vector<J>{J(1), J(0), J(1)});
        expect_equal(to_host(cols2.ptr, 3), std::vector<J>{J(0), J(1), J(2)});
    }
} // namespace

TEST_F(ConversionCov, gcoosort)
{
    check_gcoosort<int32_t>(handle, rocsparse_indextype_i32);
    check_gcoosort<int64_t>(handle, rocsparse_indextype_i64);

    size_t buffer_size = 1;
    EXPECT_EQ(rocsparse::gcoosort_buffer_size(handle,
                                              0,
                                              3,
                                              0,
                                              rocsparse_indextype_i32,
                                              nullptr,
                                              nullptr,
                                              &buffer_size),
              rocsparse_status_success);
    EXPECT_EQ(buffer_size, size_t(0));
    EXPECT_EQ(rocsparse::gcoosort_by_row(
                  handle, 2, 3, 0, rocsparse_indextype_i64, nullptr, nullptr, nullptr, nullptr),
              rocsparse_status_success);
}

// ---------------------------------------------------------------------------
// convert_array: index-base, strided, and numeric datatype switches.
// The defining TU is already compiled into this binary.
// ---------------------------------------------------------------------------
TEST_F(ConversionCov, convert_array)
{
    // Same type, distinct base: index-base kernel (0-based -> 1-based).
    device_vector<int32_t> src32{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> dst32{size_t(3)};
    ASSERT_TRUE(src32.ptr && dst32.ptr);
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       3,
                                       rocsparse_indextype_i32,
                                       dst32.ptr,
                                       rocsparse_index_base_one,
                                       rocsparse_indextype_i32,
                                       src32.ptr,
                                       rocsparse_index_base_zero),
              rocsparse_status_success);
    expect_equal(to_host(dst32.ptr, 3), std::vector<int32_t>{1, 2, 3});

    // Same type and base, different pointers: device memcpy.
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       3,
                                       rocsparse_indextype_i32,
                                       dst32.ptr,
                                       rocsparse_index_base_zero,
                                       rocsparse_indextype_i32,
                                       src32.ptr,
                                       rocsparse_index_base_zero),
              rocsparse_status_success);
    expect_equal(to_host(dst32.ptr, 3), std::vector<int32_t>{0, 1, 2});

    // Same pointer: no copy.
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       3,
                                       rocsparse_indextype_i32,
                                       src32.ptr,
                                       rocsparse_index_base_zero,
                                       rocsparse_indextype_i32,
                                       src32.ptr,
                                       rocsparse_index_base_zero),
              rocsparse_status_success);

    // i32 -> i64 and i64 -> i32.
    device_vector<int64_t> dst64{size_t(3)};
    device_vector<int64_t> src64{std::vector<int64_t>{4, 5, 6}};
    ASSERT_TRUE(dst64.ptr && src64.ptr);
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       3,
                                       rocsparse_indextype_i64,
                                       dst64.ptr,
                                       rocsparse_index_base_zero,
                                       rocsparse_indextype_i32,
                                       src32.ptr,
                                       rocsparse_index_base_zero),
              rocsparse_status_success);
    expect_equal(to_host(dst64.ptr, 3), std::vector<int64_t>{0, 1, 2});
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       3,
                                       rocsparse_indextype_i32,
                                       dst32.ptr,
                                       rocsparse_index_base_zero,
                                       rocsparse_indextype_i64,
                                       src64.ptr,
                                       rocsparse_index_base_zero),
              rocsparse_status_success);
    expect_equal(to_host(dst32.ptr, 3), std::vector<int32_t>{4, 5, 6});

    // i64 -> i64 with a base change.
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       3,
                                       rocsparse_indextype_i64,
                                       dst64.ptr,
                                       rocsparse_index_base_one,
                                       rocsparse_indextype_i64,
                                       src64.ptr,
                                       rocsparse_index_base_zero),
              rocsparse_status_success);
    expect_equal(to_host(dst64.ptr, 3), std::vector<int64_t>{5, 6, 7});

    // Value that does not fit in int32.
    device_vector<int64_t> big{std::vector<int64_t>{int64_t(1) << 40}};
    device_vector<int32_t> narrow{size_t(1)};
    ASSERT_TRUE(big.ptr && narrow.ptr);
    std::ostringstream err;
    auto*              old_err = std::cerr.rdbuf(err.rdbuf());
    EXPECT_EQ(rocsparse::convert_array(handle,
                                       1,
                                       rocsparse_indextype_i32,
                                       narrow.ptr,
                                       rocsparse_index_base_zero,
                                       rocsparse_indextype_i64,
                                       big.ptr,
                                       rocsparse_index_base_zero),
              rocsparse_status_type_mismatch);
    std::cerr.rdbuf(old_err);

    // Strided copy: source increment 2, target increment 1, mixed types.
    device_vector<int64_t> strided{std::vector<int64_t>{0, 99, 1, 99, 2, 99}};
    device_vector<int32_t> packed{size_t(3)};
    ASSERT_TRUE(strided.ptr && packed.ptr);
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       3,
                                       rocsparse_indextype_i32,
                                       packed.ptr,
                                       int64_t(1),
                                       rocsparse_indextype_i64,
                                       strided.ptr,
                                       int64_t(2)),
              rocsparse_status_success);
    expect_equal(to_host(packed.ptr, 3), std::vector<int32_t>{0, 1, 2});

    // Same type, unit stride: memcpy overload, and the inc-free wrapper.
    device_vector<int32_t> packed_src{std::vector<int32_t>{7, 8, 9}};
    ASSERT_TRUE(packed_src.ptr);
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       3,
                                       rocsparse_indextype_i32,
                                       packed.ptr,
                                       int64_t(1),
                                       rocsparse_indextype_i32,
                                       packed_src.ptr,
                                       int64_t(1)),
              rocsparse_status_success);
    expect_equal(to_host(packed.ptr, 3), std::vector<int32_t>{7, 8, 9});
    ASSERT_EQ(rocsparse::convert_array(
                  handle, 3, rocsparse_indextype_i64, dst64.ptr, rocsparse_indextype_i32, src32.ptr),
              rocsparse_status_success);
    expect_equal(to_host(dst64.ptr, 3), std::vector<int64_t>{0, 1, 2});

    // Numeric conversions. Same-type is a memcpy; every other pair takes the
    // datatype switch. Specialized pairs are checked, the rest only for status
    // (several pairs launch the generic empty conversion kernel).
    const rocsparse_datatype dts[] = {rocsparse_datatype_i8_r,
                                      rocsparse_datatype_u8_r,
                                      rocsparse_datatype_i32_r,
                                      rocsparse_datatype_u32_r,
                                      rocsparse_datatype_f16_r,
                                      rocsparse_datatype_bf16_r,
                                      rocsparse_datatype_f32_r,
                                      rocsparse_datatype_f64_r,
                                      rocsparse_datatype_f32_c,
                                      rocsparse_datatype_f64_c};
    device_vector<char> raw_src{size_t(32)};
    device_vector<char> raw_dst{size_t(32)};
    ASSERT_TRUE(raw_src.ptr && raw_dst.ptr);
    ASSERT_EQ(hipMemset(raw_src.ptr, 0, 32), hipSuccess);
    ASSERT_EQ(hipMemset(raw_dst.ptr, 0, 32), hipSuccess);

    std::ostringstream log;
    auto*              old_out = std::cout.rdbuf(log.rdbuf());
    for(rocsparse_datatype target : dts)
    {
        for(rocsparse_datatype source : dts)
        {
            ASSERT_EQ(rocsparse::convert_array(
                          handle, 2, target, raw_dst.ptr, source, raw_src.ptr),
                      rocsparse_status_success)
                << "target " << int(target) << " source " << int(source);
        }
    }
    std::cout.rdbuf(old_out);

    device_vector<float>  fsrc{std::vector<float>{1.5f, -2.0f}};
    device_vector<double> ddst{size_t(2)};
    ASSERT_TRUE(fsrc.ptr && ddst.ptr);
    old_out = std::cout.rdbuf(log.rdbuf());
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       2,
                                       rocsparse_datatype_f64_r,
                                       ddst.ptr,
                                       rocsparse_datatype_f32_r,
                                       fsrc.ptr),
              rocsparse_status_success);
    std::cout.rdbuf(old_out);
    auto hd = to_host(ddst.ptr, 2);
    EXPECT_DOUBLE_EQ(hd[0], 1.5);
    EXPECT_DOUBLE_EQ(hd[1], -2.0);

    device_vector<double> dsrc{std::vector<double>{1.25, 3.0}};
    device_vector<float>  fdst{size_t(2)};
    ASSERT_TRUE(dsrc.ptr && fdst.ptr);
    old_out = std::cout.rdbuf(log.rdbuf());
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       2,
                                       rocsparse_datatype_f32_r,
                                       fdst.ptr,
                                       rocsparse_datatype_f64_r,
                                       dsrc.ptr),
              rocsparse_status_success);
    device_vector<rocsparse_float_complex> cdst{size_t(2)};
    ASSERT_TRUE(cdst.ptr);
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       2,
                                       rocsparse_datatype_f32_c,
                                       cdst.ptr,
                                       rocsparse_datatype_f32_r,
                                       fsrc.ptr),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       2,
                                       rocsparse_datatype_f32_c,
                                       cdst.ptr,
                                       rocsparse_datatype_f64_r,
                                       dsrc.ptr),
              rocsparse_status_success);
    device_vector<rocsparse_float_complex> csrc{
        std::vector<rocsparse_float_complex>{rocsparse_float_complex(1.0f, 2.0f),
                                             rocsparse_float_complex(-3.0f, 0.5f)}};
    device_vector<rocsparse_double_complex> zdst{size_t(2)};
    ASSERT_TRUE(csrc.ptr && zdst.ptr);
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       2,
                                       rocsparse_datatype_f64_c,
                                       zdst.ptr,
                                       rocsparse_datatype_f32_c,
                                       csrc.ptr),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse::convert_array(handle,
                                       2,
                                       rocsparse_datatype_f32_c,
                                       cdst.ptr,
                                       rocsparse_datatype_f64_c,
                                       zdst.ptr),
              rocsparse_status_success);
    std::cout.rdbuf(old_out);
    auto hz = to_host(zdst.ptr, 2);
    EXPECT_DOUBLE_EQ(std::real(hz[0]), 1.0);
    EXPECT_DOUBLE_EQ(std::imag(hz[0]), 2.0);
    auto hc = to_host(cdst.ptr, 2);
    EXPECT_FLOAT_EQ(std::real(hc[0]), 1.0f);
    EXPECT_FLOAT_EQ(std::imag(hc[0]), 2.0f);

    // dnvec_transfer_from: real->real, complex->real rejected, size mismatch.
    device_vector<float>  dv_f{std::vector<float>{4.0f, 5.0f}};
    device_vector<double> dv_d{size_t(2)};
    ASSERT_TRUE(dv_f.ptr && dv_d.ptr);
    rocsparse_dnvec_descr src_vec = nullptr, dst_vec = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&src_vec, 2, dv_f.ptr, rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&dst_vec, 2, dv_d.ptr, rocsparse_datatype_f64_r),
              rocsparse_status_success);
    old_out = std::cout.rdbuf(log.rdbuf());
    ASSERT_EQ(rocsparse::dnvec_transfer_from(handle, dst_vec, src_vec), rocsparse_status_success);
    std::cout.rdbuf(old_out);
    expect_equal(to_host(dv_d.ptr, 2), std::vector<double>{4.0, 5.0});

    rocsparse_dnvec_descr cvec = nullptr;
    device_vector<rocsparse_float_complex> cv{size_t(2)};
    ASSERT_TRUE(cv.ptr);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&cvec, 2, cv.ptr, rocsparse_datatype_f32_c),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, dst_vec, cvec),
              rocsparse_status_not_implemented);
    EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, nullptr, src_vec),
              rocsparse_status_invalid_pointer);
    rocsparse_dnvec_descr short_vec = nullptr;
    device_vector<float>  one{size_t(1)};
    ASSERT_TRUE(one.ptr);
    ASSERT_EQ(rocsparse_create_dnvec_descr(&short_vec, 1, one.ptr, rocsparse_datatype_f32_r),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse::dnvec_transfer_from(handle, short_vec, src_vec),
              rocsparse_status_invalid_size);

    EXPECT_EQ(rocsparse_destroy_dnvec_descr(src_vec), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(dst_vec), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(cvec), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(short_vec), rocsparse_status_success);
}

// ---------------------------------------------------------------------------
// csxsldu_compute: same-direction split, both diag-type pairs the fill kernel
// implements, column direction, and direction-mismatch (empty and csr2csc).
// ---------------------------------------------------------------------------
namespace
{
    // 3x3 CSR
    //   | 1 2 0 |
    //   | 0 3 0 |
    //   | 4 0 5 |
    const std::vector<int32_t> k_ptr{0, 2, 3, 5};
    const std::vector<int32_t> k_ind{0, 1, 1, 0, 2};
    const std::vector<float>   k_val{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    rocsparse_status run_csxsldu(rocsparse_handle        handle,
                                 rocsparse_direction     dir,
                                 int32_t                 m,
                                 int32_t                 n,
                                 const std::vector<int32_t>& ptr,
                                 const std::vector<int32_t>& ind,
                                 const std::vector<float>&   val,
                                 rocsparse_diag_type     ldiag,
                                 rocsparse_direction     ldir,
                                 int32_t                 lnnz,
                                 std::vector<int32_t>    lptr_init,
                                 rocsparse_index_base    lbase,
                                 rocsparse_diag_type     udiag,
                                 rocsparse_direction     udir,
                                 int32_t                 unnz,
                                 std::vector<int32_t>    uptr_init,
                                 rocsparse_index_base    ubase,
                                 std::vector<int32_t>*   lptr_out,
                                 std::vector<int32_t>*   lind_out,
                                 std::vector<float>*     lval_out,
                                 std::vector<int32_t>*   uptr_out,
                                 std::vector<int32_t>*   uind_out,
                                 std::vector<float>*     uval_out,
                                 std::vector<float>*     diag_out,
                                 void*                   temp_ptrs)
    {
        device_vector<int32_t> dptr{ptr};
        device_vector<int32_t> dind{ind};
        device_vector<float>   dval{val};
        device_vector<int32_t> lptr{lptr_init};
        device_vector<int32_t> lind{size_t(std::max(lnnz, int32_t(1)))};
        device_vector<float>   lval{size_t(std::max(lnnz, int32_t(1)))};
        device_vector<int32_t> uptr{uptr_init};
        device_vector<int32_t> uind{size_t(std::max(unnz, int32_t(1)))};
        device_vector<float>   uval{size_t(std::max(unnz, int32_t(1)))};
        device_vector<float>   diag{size_t(std::max(m, n))};
        if(!dptr.ptr || !dind.ptr || !dval.ptr || !lptr.ptr || !lind.ptr || !lval.ptr || !uptr.ptr
           || !uind.ptr || !uval.ptr || !diag.ptr)
            return rocsparse_status_memory_error;

        // When a factor's direction differs from the input, the kernel reads the
        // temporary pointer array from `temp_ptrs` (the caller placed it there).
        const rocsparse_status st
            = rocsparse::csxsldu_compute_template<float, int32_t, int32_t>(handle,
                                                                           dir,
                                                                           m,
                                                                           n,
                                                                           int32_t(ind.size()),
                                                                           dptr.ptr,
                                                                           dind.ptr,
                                                                           dval.ptr,
                                                                           rocsparse_index_base_zero,
                                                                           ldiag,
                                                                           ldir,
                                                                           lnnz,
                                                                           lptr.ptr,
                                                                           lind.ptr,
                                                                           lval.ptr,
                                                                           lbase,
                                                                           udiag,
                                                                           udir,
                                                                           unnz,
                                                                           uptr.ptr,
                                                                           uind.ptr,
                                                                           uval.ptr,
                                                                           ubase,
                                                                           diag.ptr,
                                                                           temp_ptrs);
        if(st != rocsparse_status_success)
            return st;
        if(hipDeviceSynchronize() != hipSuccess)
            return rocsparse_status_internal_error;
        if(lptr_out)
            *lptr_out = to_host(lptr.ptr, lptr_init.size());
        if(lind_out && lnnz > 0)
            *lind_out = to_host(lind.ptr, lnnz);
        if(lval_out && lnnz > 0)
            *lval_out = to_host(lval.ptr, lnnz);
        if(uptr_out)
            *uptr_out = to_host(uptr.ptr, uptr_init.size());
        if(uind_out && unnz > 0)
            *uind_out = to_host(uind.ptr, unnz);
        if(uval_out && unnz > 0)
            *uval_out = to_host(uval.ptr, unnz);
        if(diag_out)
            *diag_out = to_host(diag.ptr, std::max(m, n));
        return rocsparse_status_success;
    }
} // namespace

TEST_F(ConversionCov, csxsldu_compute)
{
    device_vector<int32_t> scratch{size_t(64)};
    ASSERT_TRUE(scratch.ptr);
    ASSERT_EQ(hipMemset(scratch.ptr, 0, sizeof(int32_t) * 64), hipSuccess);

    // Same direction, both triangles strict, diagonal extracted.
    // L = {(2,0)=4}, U = {(0,1)=2}, diag = {1,3,5}.
    std::vector<int32_t> lptr, lind, uptr, uind;
    std::vector<float>   lval, uval, diag;
    ASSERT_EQ(run_csxsldu(handle,
                          rocsparse_direction_row,
                          3,
                          3,
                          k_ptr,
                          k_ind,
                          k_val,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_row,
                          1,
                          {0, 0, 0, 1},
                          rocsparse_index_base_zero,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_row,
                          1,
                          {0, 1, 1, 1},
                          rocsparse_index_base_zero,
                          &lptr,
                          &lind,
                          &lval,
                          &uptr,
                          &uind,
                          &uval,
                          &diag,
                          scratch.ptr),
              rocsparse_status_success);
    expect_equal(lind, std::vector<int32_t>{0});
    expect_equal(lval, std::vector<float>{4.0f});
    expect_equal(uind, std::vector<int32_t>{1});
    expect_equal(uval, std::vector<float>{2.0f});
    expect_equal(diag, std::vector<float>{1.0f, 3.0f, 5.0f});

    // ldiag non-unit, udiag unit: lower stays strict, upper absorbs the diagonal.
    ASSERT_EQ(run_csxsldu(handle,
                          rocsparse_direction_row,
                          3,
                          3,
                          k_ptr,
                          k_ind,
                          k_val,
                          rocsparse_diag_type_non_unit,
                          rocsparse_direction_row,
                          1,
                          {0, 0, 0, 1},
                          rocsparse_index_base_zero,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_row,
                          4,
                          {0, 2, 3, 4},
                          rocsparse_index_base_zero,
                          &lptr,
                          &lind,
                          &lval,
                          &uptr,
                          &uind,
                          &uval,
                          nullptr,
                          scratch.ptr),
              rocsparse_status_success);
    expect_equal(lind, std::vector<int32_t>{0});
    expect_equal(uind, std::vector<int32_t>{0, 1, 1, 2});
    expect_equal(uval, std::vector<float>{1.0f, 2.0f, 3.0f, 5.0f});

    // ldiag unit, udiag non-unit: lower absorbs the diagonal, upper stays strict.
    ASSERT_EQ(run_csxsldu(handle,
                          rocsparse_direction_row,
                          3,
                          3,
                          k_ptr,
                          k_ind,
                          k_val,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_row,
                          4,
                          {0, 1, 2, 4},
                          rocsparse_index_base_zero,
                          rocsparse_diag_type_non_unit,
                          rocsparse_direction_row,
                          1,
                          {0, 1, 1, 1},
                          rocsparse_index_base_zero,
                          &lptr,
                          &lind,
                          &lval,
                          &uptr,
                          &uind,
                          &uval,
                          nullptr,
                          scratch.ptr),
              rocsparse_status_success);
    expect_equal(lind, std::vector<int32_t>{0, 1, 0, 2});
    expect_equal(lval, std::vector<float>{1.0f, 3.0f, 4.0f, 5.0f});
    expect_equal(uind, std::vector<int32_t>{1});
    expect_equal(uval, std::vector<float>{2.0f});

    // Column direction, diagonal CSC, both unit: diagonal is extracted.
    ASSERT_EQ(run_csxsldu(handle,
                          rocsparse_direction_column,
                          2,
                          2,
                          {0, 1, 2},
                          {0, 1},
                          {10.0f, 20.0f},
                          rocsparse_diag_type_unit,
                          rocsparse_direction_column,
                          0,
                          {0, 0, 0},
                          rocsparse_index_base_zero,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_column,
                          0,
                          {0, 0, 0},
                          rocsparse_index_base_zero,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          &diag,
                          scratch.ptr),
              rocsparse_status_success);
    expect_equal(diag, std::vector<float>{10.0f, 20.0f});

    // Direction mismatch, empty factors, square: U is memset, L is copied.
    ASSERT_EQ(hipMemset(scratch.ptr, 0, sizeof(int32_t) * 64), hipSuccess);
    std::vector<int32_t> uptr_sq, lptr_sq;
    ASSERT_EQ(run_csxsldu(handle,
                          rocsparse_direction_row,
                          2,
                          2,
                          {0, 1, 2},
                          {0, 1},
                          {1.0f, 2.0f},
                          rocsparse_diag_type_unit,
                          rocsparse_direction_column,
                          0,
                          {7, 7, 7},
                          rocsparse_index_base_zero,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_column,
                          0,
                          {7, 7, 7},
                          rocsparse_index_base_zero,
                          &lptr_sq,
                          nullptr,
                          nullptr,
                          &uptr_sq,
                          nullptr,
                          nullptr,
                          nullptr,
                          scratch.ptr),
              rocsparse_status_success);
    expect_equal(uptr_sq, std::vector<int32_t>{0, 0, 0});
    expect_equal(lptr_sq, std::vector<int32_t>{0, 0, 0});

    // Rectangular, base zero: both mismatched empty factors are memset.
    // 2x3 diagonal has no strict lower or upper entry.
    std::vector<int32_t> uptr_r, lptr_r;
    ASSERT_EQ(hipMemset(scratch.ptr, 0, sizeof(int32_t) * 64), hipSuccess);
    ASSERT_EQ(run_csxsldu(handle,
                          rocsparse_direction_row,
                          2,
                          3,
                          {0, 1, 2},
                          {0, 1},
                          {1.0f, 2.0f},
                          rocsparse_diag_type_unit,
                          rocsparse_direction_column,
                          0,
                          {7, 7, 7, 7},
                          rocsparse_index_base_zero,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_column,
                          0,
                          {7, 7, 7, 7},
                          rocsparse_index_base_zero,
                          &lptr_r,
                          nullptr,
                          nullptr,
                          &uptr_r,
                          nullptr,
                          nullptr,
                          nullptr,
                          scratch.ptr),
              rocsparse_status_success);
    expect_equal(uptr_r, std::vector<int32_t>{0, 0, 0, 0});
    expect_equal(lptr_r, std::vector<int32_t>{0, 0, 0, 0});

    // Rectangular, base one: both mismatched empty factors are set to 1.
    ASSERT_EQ(hipMemset(scratch.ptr, 0, sizeof(int32_t) * 64), hipSuccess);
    ASSERT_EQ(run_csxsldu(handle,
                          rocsparse_direction_row,
                          2,
                          3,
                          {0, 1, 2},
                          {0, 1},
                          {1.0f, 2.0f},
                          rocsparse_diag_type_unit,
                          rocsparse_direction_column,
                          0,
                          {7, 7, 7, 7},
                          rocsparse_index_base_one,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_column,
                          0,
                          {7, 7, 7, 7},
                          rocsparse_index_base_one,
                          &lptr_r,
                          nullptr,
                          nullptr,
                          &uptr_r,
                          nullptr,
                          nullptr,
                          nullptr,
                          scratch.ptr),
              rocsparse_status_success);
    expect_equal(uptr_r, std::vector<int32_t>{1, 1, 1, 1});
    expect_equal(lptr_r, std::vector<int32_t>{1, 1, 1, 1});

    // Nonzero direction mismatch: U is CSR in the scratch pointer array and is
    // transposed to CSC. Upper of the 3x3 is the single entry (0,1)=2.
    // Scratch layout when only U differs: uptr temp at the start of scratch.
    std::vector<int32_t> uptr_csr{0, 1, 1, 1};
    ASSERT_EQ(hipMemcpy(scratch.ptr, uptr_csr.data(), sizeof(int32_t) * 4, hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(run_csxsldu(handle,
                          rocsparse_direction_row,
                          3,
                          3,
                          k_ptr,
                          k_ind,
                          k_val,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_row,
                          1,
                          {0, 0, 0, 1},
                          rocsparse_index_base_zero,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_column,
                          1,
                          {9, 9, 9, 9},
                          rocsparse_index_base_zero,
                          &lptr,
                          &lind,
                          &lval,
                          &uptr,
                          &uind,
                          &uval,
                          &diag,
                          scratch.ptr),
              rocsparse_status_success);
    expect_equal(uind, std::vector<int32_t>{0});
    expect_equal(uval, std::vector<float>{2.0f});
    expect_equal(uptr, std::vector<int32_t>{0, 0, 1, 1});

    // Nonzero direction mismatch on L. Lower entry (2,0)=4 becomes CSC.
    // Only L differs, so the temp pointer array is at the start of scratch.
    std::vector<int32_t> lptr_csr{0, 0, 0, 1};
    ASSERT_EQ(hipMemcpy(scratch.ptr, lptr_csr.data(), sizeof(int32_t) * 4, hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(run_csxsldu(handle,
                          rocsparse_direction_row,
                          3,
                          3,
                          k_ptr,
                          k_ind,
                          k_val,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_column,
                          1,
                          {9, 9, 9, 9},
                          rocsparse_index_base_zero,
                          rocsparse_diag_type_unit,
                          rocsparse_direction_row,
                          1,
                          {0, 1, 1, 1},
                          rocsparse_index_base_zero,
                          &lptr,
                          &lind,
                          &lval,
                          &uptr,
                          &uind,
                          &uval,
                          nullptr,
                          scratch.ptr),
              rocsparse_status_success);
    expect_equal(lind, std::vector<int32_t>{2});
    expect_equal(lval, std::vector<float>{4.0f});
    expect_equal(lptr, std::vector<int32_t>{0, 1, 1, 1});
}

// ---------------------------------------------------------------------------
// gell2csr via the public sparse_to_sparse ELL -> CSR path. The legacy
// ell2csr API does not enter this generic dispatcher.
// ---------------------------------------------------------------------------
namespace
{
    template <typename T, typename Icol, typename Irow>
    void ell_to_csr(rocsparse_handle    handle,
                    rocsparse_datatype  dt,
                    rocsparse_indextype col_type,
                    rocsparse_indextype row_type)
    {
        device_vector<Icol> ell_col{std::vector<Icol>{Icol(0), Icol(1), Icol(2)}};
        device_vector<T>    ell_val{std::vector<T>{scalar<T>(1), scalar<T>(2), scalar<T>(3)}};
        device_vector<Irow> row_ptr{size_t(4)};
        device_vector<Icol> col_ind{size_t(3)};
        device_vector<T>    csr_val{size_t(3)};
        ASSERT_TRUE(ell_col.ptr && ell_val.ptr && row_ptr.ptr && col_ind.ptr && csr_val.ptr);

        rocsparse_spmat_descr ell = nullptr, csr = nullptr;
        ASSERT_EQ(rocsparse_create_ell_descr(
                      &ell, 3, 3, ell_col.ptr, ell_val.ptr, 1, col_type, rocsparse_index_base_zero, dt),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&csr,
                                             3,
                                             3,
                                             3,
                                             row_ptr.ptr,
                                             col_ind.ptr,
                                             csr_val.ptr,
                                             row_type,
                                             col_type,
                                             rocsparse_index_base_zero,
                                             dt),
                  rocsparse_status_success);

        rocsparse_sparse_to_sparse_descr s2s = nullptr;
        ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                      &s2s, ell, csr, rocsparse_sparse_to_sparse_alg_default),
                  rocsparse_status_success);
        for(rocsparse_sparse_to_sparse_stage stage :
            {rocsparse_sparse_to_sparse_stage_analysis, rocsparse_sparse_to_sparse_stage_compute})
        {
            size_t buffer_size = 0;
            ASSERT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                          handle, s2s, ell, csr, stage, &buffer_size),
                      rocsparse_status_success);
            device_vector<char> buffer{buffer_size ? buffer_size : size_t(1)};
            ASSERT_TRUE(buffer.ptr);
            ASSERT_EQ(rocsparse_sparse_to_sparse(
                          handle, s2s, ell, csr, stage, buffer_size, buffer.ptr),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }
        expect_equal(to_host(row_ptr.ptr, 4), std::vector<Irow>{Irow(0), Irow(1), Irow(2), Irow(3)});
        expect_equal(to_host(col_ind.ptr, 3), std::vector<Icol>{Icol(0), Icol(1), Icol(2)});
        expect_equal(to_host(csr_val.ptr, 3),
                     std::vector<T>{scalar<T>(1), scalar<T>(2), scalar<T>(3)});

        EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(s2s), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(ell), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);
    }
} // namespace

TEST_F(ConversionCov, gell2csr)
{
    ell_to_csr<float, int32_t, int32_t>(
        handle, rocsparse_datatype_f32_r, rocsparse_indextype_i32, rocsparse_indextype_i32);
    ell_to_csr<float, int32_t, int64_t>(
        handle, rocsparse_datatype_f32_r, rocsparse_indextype_i32, rocsparse_indextype_i64);
    ell_to_csr<float, int64_t, int32_t>(
        handle, rocsparse_datatype_f32_r, rocsparse_indextype_i64, rocsparse_indextype_i32);
    ell_to_csr<float, int64_t, int64_t>(
        handle, rocsparse_datatype_f32_r, rocsparse_indextype_i64, rocsparse_indextype_i64);

    ell_to_csr<double, int32_t, int32_t>(
        handle, rocsparse_datatype_f64_r, rocsparse_indextype_i32, rocsparse_indextype_i32);
    ell_to_csr<double, int64_t, int64_t>(
        handle, rocsparse_datatype_f64_r, rocsparse_indextype_i64, rocsparse_indextype_i64);
    ell_to_csr<double, int32_t, int64_t>(
        handle, rocsparse_datatype_f64_r, rocsparse_indextype_i32, rocsparse_indextype_i64);
    ell_to_csr<double, int64_t, int32_t>(
        handle, rocsparse_datatype_f64_r, rocsparse_indextype_i64, rocsparse_indextype_i32);

    ell_to_csr<rocsparse_float_complex, int32_t, int32_t>(
        handle, rocsparse_datatype_f32_c, rocsparse_indextype_i32, rocsparse_indextype_i32);
    ell_to_csr<rocsparse_float_complex, int64_t, int64_t>(
        handle, rocsparse_datatype_f32_c, rocsparse_indextype_i64, rocsparse_indextype_i64);
    ell_to_csr<rocsparse_float_complex, int32_t, int64_t>(
        handle, rocsparse_datatype_f32_c, rocsparse_indextype_i32, rocsparse_indextype_i64);
    ell_to_csr<rocsparse_float_complex, int64_t, int32_t>(
        handle, rocsparse_datatype_f32_c, rocsparse_indextype_i64, rocsparse_indextype_i32);

    ell_to_csr<rocsparse_double_complex, int32_t, int32_t>(
        handle, rocsparse_datatype_f64_c, rocsparse_indextype_i32, rocsparse_indextype_i32);
    ell_to_csr<rocsparse_double_complex, int64_t, int64_t>(
        handle, rocsparse_datatype_f64_c, rocsparse_indextype_i64, rocsparse_indextype_i64);
    ell_to_csr<rocsparse_double_complex, int32_t, int64_t>(
        handle, rocsparse_datatype_f64_c, rocsparse_indextype_i32, rocsparse_indextype_i64);
    ell_to_csr<rocsparse_double_complex, int64_t, int32_t>(
        handle, rocsparse_datatype_f64_c, rocsparse_indextype_i64, rocsparse_indextype_i32);

    // Datatype mismatch and column-index mismatch return not_implemented.
    device_vector<int32_t> ell_col{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   ell_val{std::vector<float>{1, 2, 3}};
    device_vector<int32_t> row_ptr{size_t(4)};
    device_vector<int32_t> col_ind{size_t(3)};
    device_vector<double>  csr_d{size_t(3)};
    ASSERT_TRUE(ell_col.ptr && ell_val.ptr && row_ptr.ptr && col_ind.ptr && csr_d.ptr);
    rocsparse_spmat_descr ell = nullptr, csr = nullptr;
    ASSERT_EQ(rocsparse_create_ell_descr(&ell,
                                         3,
                                         3,
                                         ell_col.ptr,
                                         ell_val.ptr,
                                         1,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_csr_descr(&csr,
                                         3,
                                         3,
                                         3,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         csr_d.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f64_r),
              rocsparse_status_success);
    rocsparse_sparse_to_sparse_descr s2s = nullptr;
    ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  &s2s, ell, csr, rocsparse_sparse_to_sparse_alg_default),
              rocsparse_status_success);
    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                  handle, s2s, ell, csr, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size),
              rocsparse_status_success);
    device_vector<char> buffer{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(buffer.ptr);
    ASSERT_EQ(rocsparse_sparse_to_sparse(handle,
                                         s2s,
                                         ell,
                                         csr,
                                         rocsparse_sparse_to_sparse_stage_analysis,
                                         buffer_size,
                                         buffer.ptr),
              rocsparse_status_success);
    buffer_size = 0;
    ASSERT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                  handle, s2s, ell, csr, rocsparse_sparse_to_sparse_stage_compute, &buffer_size),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_sparse_to_sparse(handle,
                                         s2s,
                                         ell,
                                         csr,
                                         rocsparse_sparse_to_sparse_stage_compute,
                                         buffer_size,
                                         buffer.ptr),
              rocsparse_status_not_implemented);
    EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(s2s), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);

    device_vector<int64_t> col64{size_t(3)};
    device_vector<float>   csr_f{size_t(3)};
    ASSERT_TRUE(col64.ptr && csr_f.ptr);
    ASSERT_EQ(rocsparse_create_csr_descr(&csr,
                                         3,
                                         3,
                                         3,
                                         row_ptr.ptr,
                                         col64.ptr,
                                         csr_f.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i64,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  &s2s, ell, csr, rocsparse_sparse_to_sparse_alg_default),
              rocsparse_status_success);
    buffer_size = 0;
    ASSERT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                  handle, s2s, ell, csr, rocsparse_sparse_to_sparse_stage_compute, &buffer_size),
              rocsparse_status_success);
    device_vector<char> buffer_col{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(buffer_col.ptr);
    EXPECT_EQ(rocsparse_sparse_to_sparse(handle,
                                         s2s,
                                         ell,
                                         csr,
                                         rocsparse_sparse_to_sparse_stage_compute,
                                         buffer_size,
                                         buffer_col.ptr),
              rocsparse_status_not_implemented);
    EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(s2s), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);

    // Integer / half / bfloat values are rejected inside gell2csr.
    const rocsparse_datatype rejected[] = {rocsparse_datatype_i8_r,
                                           rocsparse_datatype_u8_r,
                                           rocsparse_datatype_i32_r,
                                           rocsparse_datatype_u32_r,
                                           rocsparse_datatype_f16_r,
                                           rocsparse_datatype_bf16_r};
    for(rocsparse_datatype dt : rejected)
    {
        rocsparse_spmat_descr e2 = nullptr, c2 = nullptr;
        ASSERT_EQ(rocsparse_create_ell_descr(&e2,
                                             3,
                                             3,
                                             ell_col.ptr,
                                             ell_val.ptr,
                                             1,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             dt),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_csr_descr(&c2,
                                             3,
                                             3,
                                             3,
                                             row_ptr.ptr,
                                             col_ind.ptr,
                                             ell_val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             dt),
                  rocsparse_status_success);
        rocsparse_sparse_to_sparse_descr d2 = nullptr;
        ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                      &d2, e2, c2, rocsparse_sparse_to_sparse_alg_default),
                  rocsparse_status_success);
        buffer_size = 0;
        ASSERT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                      handle, d2, e2, c2, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size),
                  rocsparse_status_success);
        device_vector<char> buf{buffer_size ? buffer_size : size_t(1)};
        ASSERT_TRUE(buf.ptr);
        ASSERT_EQ(rocsparse_sparse_to_sparse(handle,
                                             d2,
                                             e2,
                                             c2,
                                             rocsparse_sparse_to_sparse_stage_analysis,
                                             buffer_size,
                                             buf.ptr),
                  rocsparse_status_success);
        buffer_size = 0;
        ASSERT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                      handle, d2, e2, c2, rocsparse_sparse_to_sparse_stage_compute, &buffer_size),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_sparse_to_sparse(handle,
                                             d2,
                                             e2,
                                             c2,
                                             rocsparse_sparse_to_sparse_stage_compute,
                                             buffer_size,
                                             buf.ptr),
                  rocsparse_status_not_implemented)
            << "datatype " << int(dt);
        EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(d2), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(e2), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(c2), rocsparse_status_success);
    }
    EXPECT_EQ(rocsparse_destroy_spmat_descr(ell), rocsparse_status_success);
}
