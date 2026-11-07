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

// Stage ordering and buffer-size checks for rocsparse_spsort that the
// integration tests do not reach.

#include "unit_test_utils.hpp"

using namespace rocsparse_ut;

namespace
{
    struct SpmatGuard
    {
        rocsparse_spmat_descr mat = nullptr;

        SpmatGuard() = default;

        ~SpmatGuard()
        {
            if(mat != nullptr)
            {
                rocsparse_destroy_spmat_descr(mat);
            }
        }

        SpmatGuard(const SpmatGuard&) = delete;

        SpmatGuard& operator=(const SpmatGuard&) = delete;
    };

    struct SpsortDescrGuard
    {
        rocsparse_handle       handle;
        rocsparse_spsort_descr descr = nullptr;

        explicit SpsortDescrGuard(rocsparse_handle h)
            : handle(h)
        {
        }

        ~SpsortDescrGuard()
        {
            if(descr != nullptr)
            {
                rocsparse_spsort_descr_destroy(handle, descr, nullptr);
            }
        }

        SpsortDescrGuard(const SpsortDescrGuard&) = delete;

        SpsortDescrGuard& operator=(const SpsortDescrGuard&) = delete;
    };

    // Creates the descriptor and sets the default algorithm and, if given, the
    // COO direction. Returns the first failing status.
    rocsparse_status make_spsort_descr(rocsparse_handle           handle,
                                       rocsparse_spsort_descr*    descr,
                                       const rocsparse_direction* dir = nullptr)
    {
        rocsparse_status status = rocsparse_spsort_descr_create(handle, descr, nullptr);
        if(status != rocsparse_status_success)
        {
            return status;
        }

        const rocsparse_spsort_alg alg = rocsparse_spsort_alg_default;
        status                         = rocsparse_spsort_set_input(
            handle, *descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr);
        if(status != rocsparse_status_success || dir == nullptr)
        {
            return status;
        }

        return rocsparse_spsort_set_input(
            handle, *descr, rocsparse_spsort_input_direction, dir, sizeof(*dir), nullptr);
    }
}

class ConversionSpsort : public HandleTest
{
};

TEST_F(ConversionSpsort, analysis_after_compute_is_invalid_value)
{
    // 2x3 CSR with unsorted columns.
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{2, 0, 1}};
    device_vector<float>         val{std::vector<float>{1.0f, 2.0f, 3.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    SpmatGuard mat;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat.mat,
                                         2,
                                         3,
                                         3,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    SpsortDescrGuard spsort(handle);
    ASSERT_EQ(make_spsort_descr(handle, &spsort.descr), rocsparse_status_success);

    ASSERT_EQ(rocsparse_spsort(handle,
                               spsort.descr,
                               mat.mat,
                               mat.mat,
                               rocsparse_spsort_stage_analysis,
                               0,
                               nullptr,
                               nullptr),
              rocsparse_status_success);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_spsort_buffer_size(handle,
                                           spsort.descr,
                                           mat.mat,
                                           mat.mat,
                                           rocsparse_spsort_stage_compute,
                                           &buffer_size,
                                           nullptr),
              rocsparse_status_success);
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer_size == 0 || buffer.ptr != nullptr);
    ASSERT_EQ(rocsparse_spsort(handle,
                               spsort.descr,
                               mat.mat,
                               mat.mat,
                               rocsparse_spsort_stage_compute,
                               buffer_size,
                               buffer_size == 0 ? nullptr : buffer.ptr,
                               nullptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_spsort(handle,
                               spsort.descr,
                               mat.mat,
                               mat.mat,
                               rocsparse_spsort_stage_analysis,
                               0,
                               nullptr,
                               nullptr),
              rocsparse_status_invalid_value);
}

// Change detector: the analysis stage needs no scratch space today, which the
// API documentation does not promise.
TEST_F(ConversionSpsort, analysis_buffer_size_is_zero)
{
    device_vector<rocsparse_int> ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> row_ind{std::vector<rocsparse_int>{1, 0}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1}};
    device_vector<float>         val{std::vector<float>{1.0f, 2.0f}};
    ASSERT_TRUE(ptr.ptr && row_ind.ptr && col_ind.ptr && val.ptr);

    const rocsparse_direction dir = rocsparse_direction_row;

    for(rocsparse_format format :
        {rocsparse_format_csr, rocsparse_format_csc, rocsparse_format_coo})
    {
        SCOPED_TRACE(format);

        SpmatGuard mat;
        switch(format)
        {
        case rocsparse_format_csr:
            ASSERT_EQ(rocsparse_create_csr_descr(&mat.mat,
                                                 2,
                                                 2,
                                                 2,
                                                 ptr.ptr,
                                                 col_ind.ptr,
                                                 val.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            break;
        case rocsparse_format_csc:
            ASSERT_EQ(rocsparse_create_csc_descr(&mat.mat,
                                                 2,
                                                 2,
                                                 2,
                                                 ptr.ptr,
                                                 row_ind.ptr,
                                                 val.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            break;
        default:
            ASSERT_EQ(rocsparse_create_coo_descr(&mat.mat,
                                                 2,
                                                 2,
                                                 2,
                                                 row_ind.ptr,
                                                 col_ind.ptr,
                                                 val.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
            break;
        }

        SpsortDescrGuard spsort(handle);
        ASSERT_EQ(make_spsort_descr(
                      handle, &spsort.descr, format == rocsparse_format_coo ? &dir : nullptr),
                  rocsparse_status_success);

        size_t buffer_size = 123;
        EXPECT_EQ(rocsparse_spsort_buffer_size(handle,
                                               spsort.descr,
                                               mat.mat,
                                               mat.mat,
                                               rocsparse_spsort_stage_analysis,
                                               &buffer_size,
                                               nullptr),
                  rocsparse_status_success);
        EXPECT_EQ(buffer_size, 0u);
    }
}
