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
// rocsparse_scheck_matrix_ell (the ELL half of rocsparse_check_matrix_hyb) with
// m * ell_width above INT32_MAX, which a HYB matrix can now reach
// (rocsparse_csr2hyb stores ell_nnz as 64-bit). Three 32-bit products used to
// make the check unsafe:
//
//   * the argument checks and the quick return evaluated m * ell_width in
//     32-bit, so the product wrapped to a negative number (null arrays were
//     accepted) or to exactly 0 (the check was skipped and reported success);
//   * the kernel indexed the ELL arrays with a 32-bit m * j + row, which wrapped
//     negative and read before the start of the arrays.
//
#include "unit_test_utils.hpp"

using namespace rocsparse_ut;

class CheckMatrixEllLarge : public HandleTest
{
};

namespace
{
    // m * ell_width = 2^32: the 32-bit product is exactly 0.
    constexpr rocsparse_int m_wrap_zero = 65536;
    constexpr rocsparse_int w_wrap_zero = 65536;
    // m * ell_width = 2,147,549,184 > INT32_MAX: the 32-bit product is negative.
    constexpr rocsparse_int m_wrap_neg = 65536;
    constexpr rocsparse_int w_wrap_neg = 32769;
}

// Host-only: the size is validated against the null arrays, so no large
// allocation is needed. With the 32-bit product the call returned success.
TEST_F(CheckMatrixEllLarge, size_checks_use_64_bit_product)
{
    const rocsparse_int ms[] = {m_wrap_zero, m_wrap_neg};
    const rocsparse_int ws[] = {w_wrap_zero, w_wrap_neg};
    const rocsparse_int n    = 40000;
    device_vector<char> buffer(size_t{256});
    ASSERT_NE(buffer.ptr, nullptr);

    for(int i = 0; i < 2; ++i)
    {
        size_t buffer_size = 0;
        EXPECT_EQ(rocsparse_scheck_matrix_ell_buffer_size(handle,
                                                          ms[i],
                                                          n,
                                                          ws[i],
                                                          nullptr,
                                                          nullptr,
                                                          rocsparse_index_base_zero,
                                                          rocsparse_matrix_type_general,
                                                          rocsparse_fill_mode_lower,
                                                          rocsparse_storage_mode_unsorted,
                                                          &buffer_size),
                  rocsparse_status_invalid_pointer)
            << "case " << i;

        rocsparse_data_status data_status = rocsparse_data_status_success;
        EXPECT_EQ(rocsparse_scheck_matrix_ell(handle,
                                              ms[i],
                                              n,
                                              ws[i],
                                              nullptr,
                                              nullptr,
                                              rocsparse_index_base_zero,
                                              rocsparse_matrix_type_general,
                                              rocsparse_fill_mode_lower,
                                              rocsparse_storage_mode_unsorted,
                                              &data_status,
                                              buffer.ptr),
                  rocsparse_status_invalid_pointer)
            << "case " << i;
    }
}

// An invalid column index planted at flat position 2^31 + 5 (row 5, j = 32768)
// must be found. With a 32-bit m * j + row that position wrapped to
// -2^31 + 5, so the kernel never read it and the check reported success.
// Needs two 2^31-element arrays (about 17 GB): skipped when the device is
// smaller.
TEST_F(CheckMatrixEllLarge, kernel_indexes_beyond_int32_max)
{
    const rocsparse_int m     = m_wrap_neg;
    const rocsparse_int w     = w_wrap_neg;
    const rocsparse_int n     = 40000;
    const int64_t       total = static_cast<int64_t>(m) * w;

    size_t free_bytes = 0, total_bytes = 0;
    ASSERT_EQ(hipMemGetInfo(&free_bytes, &total_bytes), hipSuccess);
    const size_t needed = static_cast<size_t>(total) * (sizeof(rocsparse_int) + sizeof(float));
    if(free_bytes < needed + (size_t{2} << 30))
    {
        GTEST_SKIP() << "needs " << needed / (1 << 20) << " MiB of device memory, have "
                     << free_bytes / (1 << 20) << " MiB free";
    }

    device_vector<rocsparse_int> col(static_cast<size_t>(total));
    device_vector<float>         val(static_cast<size_t>(total));
    ASSERT_NE(col.ptr, nullptr);
    ASSERT_NE(val.ptr, nullptr);
    // Every column index 0 and every value 0 is valid.
    UT_CHECK_HIP(hipMemset(col.ptr, 0, static_cast<size_t>(total) * sizeof(rocsparse_int)));
    UT_CHECK_HIP(hipMemset(val.ptr, 0, static_cast<size_t>(total) * sizeof(float)));

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_scheck_matrix_ell_buffer_size(handle,
                                                      m,
                                                      n,
                                                      w,
                                                      val.ptr,
                                                      col.ptr,
                                                      rocsparse_index_base_zero,
                                                      rocsparse_matrix_type_general,
                                                      rocsparse_fill_mode_lower,
                                                      rocsparse_storage_mode_unsorted,
                                                      &buffer_size),
              rocsparse_status_success);
    device_vector<char> buffer(buffer_size > 0 ? buffer_size : size_t{256});
    ASSERT_NE(buffer.ptr, nullptr);

    auto check = [&]() {
        rocsparse_data_status data_status = rocsparse_data_status_success;
        EXPECT_EQ(rocsparse_scheck_matrix_ell(handle,
                                              m,
                                              n,
                                              w,
                                              val.ptr,
                                              col.ptr,
                                              rocsparse_index_base_zero,
                                              rocsparse_matrix_type_general,
                                              rocsparse_fill_mode_lower,
                                              rocsparse_storage_mode_unsorted,
                                              &data_status,
                                              buffer.ptr),
                  rocsparse_status_success);
        return data_status;
    };

    // All entries valid: success.
    EXPECT_EQ(check(), rocsparse_data_status_success);

    // One column index out of [0, n) at flat position 2^31 + 5.
    const rocsparse_int bad = n + 5;
    UT_CHECK_HIP(hipMemcpy(
        col.ptr + (int64_t(1) << 31) + 5, &bad, sizeof(rocsparse_int), hipMemcpyHostToDevice));
    EXPECT_EQ(check(), rocsparse_data_status_invalid_index);
}
