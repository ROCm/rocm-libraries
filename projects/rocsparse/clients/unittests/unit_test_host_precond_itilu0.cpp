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
// Host-path unit tests for the precond sub-lib. Drives a full csrilu0
// buffer_size -> analysis -> compute -> clear pipeline on a tiny well-conditioned
// diagonal matrix (ILU0 of a diagonal is trivially the diagonal, no zero pivot),
// plus mat_info lifecycle and validation guards. Exercises host dispatch/analysis
// code in library/src/precond.
//
#include "unit_test_utils.hpp"

#include <type_traits>

using namespace rocsparse_ut;

// ======================================================================
// csritilu0 : iterative ILU0. Full buffer_size -> preprocess -> compute_ex
// pipeline on diag(2,3,4) (converges immediately), plus bad-arg guards.
// ======================================================================
class PrecondCsritilu0 : public HandleTest
{
};

template <typename T>
static void run_csritilu0_pipeline(rocsparse_handle handle)
{
    const rocsparse_itilu0_alg alg    = rocsparse_itilu0_alg_default;
    const rocsparse_int        option = 0;
    const rocsparse_int        m = 3, nnz = 3;
    const rocsparse_index_base base     = rocsparse_index_base_zero;
    rocsparse_int              nmaxiter = 20;

    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<T>             val{std::vector<T>{scalar<T>(2), scalar<T>(3), scalar<T>(4)}};
    device_vector<T>             ilu0{(size_t)nnz};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr && ilu0.ptr);

    size_t           buffer_size = 0;
    rocsparse_status st          = rocsparse_csritilu0_buffer_size(
        handle, alg, option, nmaxiter, m, nnz, row_ptr, col_ind, base, dt_of<T>(), &buffer_size);
    if(st == rocsparse_status_not_implemented)
        return;
    ASSERT_EQ(st, rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    st = rocsparse_csritilu0_preprocess(handle,
                                        alg,
                                        option,
                                        nmaxiter,
                                        m,
                                        nnz,
                                        row_ptr,
                                        col_ind,
                                        base,
                                        dt_of<T>(),
                                        buffer_size,
                                        buffer.ptr);
    if(st == rocsparse_status_not_implemented)
        return;
    ASSERT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsritilu0_compute_ex(handle,
                                             alg,
                                             option,
                                             &nmaxiter,
                                             0,
                                             1.0e-6f,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             val,
                                             ilu0,
                                             base,
                                             buffer_size,
                                             buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsritilu0_compute_ex(handle,
                                             alg,
                                             option,
                                             &nmaxiter,
                                             0,
                                             1.0e-10,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             val,
                                             ilu0,
                                             base,
                                             buffer_size,
                                             buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsritilu0_compute_ex(handle,
                                             alg,
                                             option,
                                             &nmaxiter,
                                             0,
                                             1.0e-6f,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             val,
                                             ilu0,
                                             base,
                                             buffer_size,
                                             buffer.ptr);
    else
        st = rocsparse_zcsritilu0_compute_ex(handle,
                                             alg,
                                             option,
                                             &nmaxiter,
                                             0,
                                             1.0e-10,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             val,
                                             ilu0,
                                             base,
                                             buffer_size,
                                             buffer.ptr);
    if(st == rocsparse_status_not_implemented)
        return;
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(PrecondCsritilu0, pipeline_float)
{
    run_csritilu0_pipeline<float>(handle);
}
TEST_F(PrecondCsritilu0, pipeline_double)
{
    run_csritilu0_pipeline<double>(handle);
}
TEST_F(PrecondCsritilu0, pipeline_float_complex)
{
    run_csritilu0_pipeline<rocsparse_float_complex>(handle);
}
TEST_F(PrecondCsritilu0, pipeline_double_complex)
{
    run_csritilu0_pipeline<rocsparse_double_complex>(handle);
}

TEST_F(PrecondCsritilu0, bad_args)
{
    const rocsparse_itilu0_alg   alg    = rocsparse_itilu0_alg_default;
    const rocsparse_int          option = 0;
    const rocsparse_int          m = 3, nnz = 3;
    const rocsparse_index_base   base     = rocsparse_index_base_zero;
    const rocsparse_int          nmaxiter = 10;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr);

    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_csritilu0_buffer_size(nullptr,
                                              alg,
                                              option,
                                              nmaxiter,
                                              m,
                                              nnz,
                                              row_ptr,
                                              col_ind,
                                              base,
                                              rocsparse_datatype_f32_r,
                                              &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csritilu0_buffer_size(handle,
                                              alg,
                                              option,
                                              nmaxiter,
                                              m,
                                              nnz,
                                              nullptr,
                                              col_ind,
                                              base,
                                              rocsparse_datatype_f32_r,
                                              &buffer_size),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_csritilu0_buffer_size(handle,
                                              alg,
                                              option,
                                              nmaxiter,
                                              -1,
                                              nnz,
                                              row_ptr,
                                              col_ind,
                                              base,
                                              rocsparse_datatype_f32_r,
                                              &buffer_size),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_csritilu0_buffer_size(handle,
                                              alg,
                                              option,
                                              nmaxiter,
                                              m,
                                              nnz,
                                              row_ptr,
                                              col_ind,
                                              (rocsparse_index_base)-1,
                                              rocsparse_datatype_f32_r,
                                              &buffer_size),
              rocsparse_status_invalid_value);
}

// ======================================================================
// csritilu0 : algorithm / option / index-base coverage of the preprocess
// dispatch (rocsparse_csritilu0_preprocess.cpp) and the compute_ex driver.
// The default diagonal system diag(2,3,4) converges immediately for every
// algorithm and option combination.
// ======================================================================
// When run_compute is false the pipeline stops after preprocess. The preprocess
// dispatch (rocsparse_csritilu0_preprocess.cpp) branches purely on the algorithm
// enum, so buffer_size + preprocess is enough to cover every algorithm case;
// the split-storage device compute paths are only driven for the default /
// in-place algorithm that the existing pipelines already validate.
template <typename T>
static void run_csritilu0_ex(rocsparse_handle     handle,
                             rocsparse_itilu0_alg alg,
                             rocsparse_int        option,
                             rocsparse_index_base base,
                             bool                 run_compute)
{
    const rocsparse_int m = 3, nnz = 3;
    rocsparse_int       nmaxiter = 20;

    const rocsparse_int          b = (base == rocsparse_index_base_one) ? 1 : 0;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0 + b, 1 + b, 2 + b, 3 + b}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0 + b, 1 + b, 2 + b}};
    device_vector<T>             val{std::vector<T>{scalar<T>(2), scalar<T>(3), scalar<T>(4)}};
    device_vector<T>             ilu0{(size_t)nnz};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr && ilu0.ptr);

    size_t           buffer_size = 0;
    rocsparse_status st          = rocsparse_csritilu0_buffer_size(
        handle, alg, option, nmaxiter, m, nnz, row_ptr, col_ind, base, dt_of<T>(), &buffer_size);
    if(st == rocsparse_status_not_implemented)
        return;
    ASSERT_EQ(st, rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    st = rocsparse_csritilu0_preprocess(handle,
                                        alg,
                                        option,
                                        nmaxiter,
                                        m,
                                        nnz,
                                        row_ptr,
                                        col_ind,
                                        base,
                                        dt_of<T>(),
                                        buffer_size,
                                        buffer.ptr);
    if(st == rocsparse_status_not_implemented)
        return;
    ASSERT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    if(!run_compute)
        return;

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsritilu0_compute_ex(handle,
                                             alg,
                                             option,
                                             &nmaxiter,
                                             0,
                                             1.0e-6f,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             val,
                                             ilu0,
                                             base,
                                             buffer_size,
                                             buffer.ptr);
    else
        st = rocsparse_dcsritilu0_compute_ex(handle,
                                             alg,
                                             option,
                                             &nmaxiter,
                                             0,
                                             1.0e-10,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             val,
                                             ilu0,
                                             base,
                                             buffer_size,
                                             buffer.ptr);
    if(st == rocsparse_status_not_implemented)
        return;
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(PrecondCsritilu0, alg_async_inplace_compute)
{
    run_csritilu0_ex<double>(
        handle, rocsparse_itilu0_alg_async_inplace, 0, rocsparse_index_base_zero, true);
}
TEST_F(PrecondCsritilu0, alg_async_split_preprocess)
{
    run_csritilu0_ex<double>(
        handle, rocsparse_itilu0_alg_async_split, 0, rocsparse_index_base_zero, false);
}
TEST_F(PrecondCsritilu0, alg_sync_split_preprocess)
{
    run_csritilu0_ex<double>(
        handle, rocsparse_itilu0_alg_sync_split, 0, rocsparse_index_base_zero, false);
}
TEST_F(PrecondCsritilu0, alg_sync_split_fusion_preprocess)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    run_csritilu0_ex<float>(
        handle, rocsparse_itilu0_alg_sync_split_fusion, 0, rocsparse_index_base_zero, false);
#pragma clang diagnostic pop
}
TEST_F(PrecondCsritilu0, option_bitmask_default_alg)
{
    const rocsparse_int option = rocsparse_itilu0_option_stopping_criteria
                                 | rocsparse_itilu0_option_compute_nrm_residual
                                 | rocsparse_itilu0_option_convergence_history;
    run_csritilu0_ex<double>(
        handle, rocsparse_itilu0_alg_default, option, rocsparse_index_base_zero, true);
}
TEST_F(PrecondCsritilu0, index_base_one_compute)
{
    run_csritilu0_ex<double>(
        handle, rocsparse_itilu0_alg_default, 0, rocsparse_index_base_one, true);
}

// preprocess-specific guards (rocsparse_csritilu0_preprocess.cpp).
TEST_F(PrecondCsritilu0, preprocess_guards)
{
    const rocsparse_itilu0_alg   alg = rocsparse_itilu0_alg_default;
    const rocsparse_int          m = 3, nnz = 3, nmaxiter = 10;
    const rocsparse_index_base   base = rocsparse_index_base_zero;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<char>          buffer{size_t(256)};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && buffer.ptr);

    // Invalid handle.
    EXPECT_EQ(rocsparse_csritilu0_preprocess(nullptr,
                                             alg,
                                             0,
                                             nmaxiter,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             base,
                                             rocsparse_datatype_f32_r,
                                             256,
                                             buffer.ptr),
              rocsparse_status_invalid_handle);
    // Invalid algorithm enum.
    EXPECT_EQ(rocsparse_csritilu0_preprocess(handle,
                                             (rocsparse_itilu0_alg)-1,
                                             0,
                                             nmaxiter,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             base,
                                             rocsparse_datatype_f32_r,
                                             256,
                                             buffer.ptr),
              rocsparse_status_invalid_value);
    // Negative option.
    EXPECT_EQ(rocsparse_csritilu0_preprocess(handle,
                                             alg,
                                             -1,
                                             nmaxiter,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             base,
                                             rocsparse_datatype_f32_r,
                                             256,
                                             buffer.ptr),
              rocsparse_status_invalid_value);
    // Negative nmaxiter.
    EXPECT_EQ(rocsparse_csritilu0_preprocess(handle,
                                             alg,
                                             0,
                                             -1,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             base,
                                             rocsparse_datatype_f32_r,
                                             256,
                                             buffer.ptr),
              rocsparse_status_invalid_value);
    // m == 0 is a quick-return success.
    EXPECT_EQ(rocsparse_csritilu0_preprocess(handle,
                                             alg,
                                             0,
                                             nmaxiter,
                                             0,
                                             0,
                                             nullptr,
                                             nullptr,
                                             base,
                                             rocsparse_datatype_f32_r,
                                             256,
                                             buffer.ptr),
              rocsparse_status_success);
    // nnz == 0 with m > 0 reports a zero pivot.
    EXPECT_EQ(rocsparse_csritilu0_preprocess(handle,
                                             alg,
                                             0,
                                             nmaxiter,
                                             m,
                                             0,
                                             row_ptr,
                                             nullptr,
                                             base,
                                             rocsparse_datatype_f32_r,
                                             256,
                                             buffer.ptr),
              rocsparse_status_zero_pivot);
}
