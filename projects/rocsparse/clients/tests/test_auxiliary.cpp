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

#include <cstring>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

// =============================================================================
// Handle Tests
// =============================================================================

TEST(auxiliary_pre_checkin, HandleCreateDestroy)
{
    rocsparse_handle handle;
    ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, HandleCreateNullptr)
{
    ASSERT_EQ(rocsparse_create_handle(nullptr), rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, HandleDestroyNull)
{
    ASSERT_EQ(rocsparse_destroy_handle(nullptr), rocsparse_status_invalid_handle);
}

TEST(auxiliary_pre_checkin, PointerMode)
{
    rocsparse_handle handle;
    ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);

    // Set and get host mode
    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);

    rocsparse_pointer_mode mode;
    ASSERT_EQ(rocsparse_get_pointer_mode(handle, &mode), rocsparse_status_success);
    ASSERT_EQ(mode, rocsparse_pointer_mode_host);

    // Set and get device mode
    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_get_pointer_mode(handle, &mode), rocsparse_status_success);
    ASSERT_EQ(mode, rocsparse_pointer_mode_device);

    // Test invalid cases
    ASSERT_EQ(rocsparse_set_pointer_mode(nullptr, rocsparse_pointer_mode_host),
              rocsparse_status_invalid_handle);
    ASSERT_EQ(rocsparse_get_pointer_mode(nullptr, &mode), rocsparse_status_invalid_handle);
    ASSERT_EQ(rocsparse_get_pointer_mode(handle, nullptr), rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, Stream)
{
    rocsparse_handle handle;
    ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);

    // Create a custom stream
    hipStream_t custom_stream;
    ASSERT_EQ(hipStreamCreate(&custom_stream), hipSuccess);

    // Set and get custom stream
    ASSERT_EQ(rocsparse_set_stream(handle, custom_stream), rocsparse_status_success);

    hipStream_t retrieved_stream;
    ASSERT_EQ(rocsparse_get_stream(handle, &retrieved_stream), rocsparse_status_success);
    ASSERT_EQ(retrieved_stream, custom_stream);

    // Test invalid cases
    ASSERT_EQ(rocsparse_set_stream(nullptr, custom_stream), rocsparse_status_invalid_handle);
    ASSERT_EQ(rocsparse_get_stream(nullptr, &retrieved_stream), rocsparse_status_invalid_handle);
    ASSERT_EQ(rocsparse_get_stream(handle, nullptr), rocsparse_status_invalid_pointer);

    ASSERT_EQ(hipStreamDestroy(custom_stream), hipSuccess);
    ASSERT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, Version)
{
    rocsparse_handle handle;
    ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);

    int version;
    ASSERT_EQ(rocsparse_get_version(handle, &version), rocsparse_status_success);
    ASSERT_GT(version, 0);

    // Test invalid cases
    ASSERT_EQ(rocsparse_get_version(nullptr, &version), rocsparse_status_invalid_handle);
    ASSERT_EQ(rocsparse_get_version(handle, nullptr), rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, GitRev)
{
    rocsparse_handle handle;
    ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);

    char rev[64];
    ASSERT_EQ(rocsparse_get_git_rev(handle, rev), rocsparse_status_success);

    // Test invalid cases
    ASSERT_EQ(rocsparse_get_git_rev(nullptr, rev), rocsparse_status_invalid_handle);
    ASSERT_EQ(rocsparse_get_git_rev(handle, nullptr), rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
}

// =============================================================================
// Matrix Descriptor Tests
// =============================================================================

TEST(auxiliary_pre_checkin, MatDescrCreateDestroy)
{
    rocsparse_mat_descr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, MatDescrCreateNullptr)
{
    ASSERT_EQ(rocsparse_create_mat_descr(nullptr), rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, MatDescrCopy)
{
    rocsparse_mat_descr src, dest;
    ASSERT_EQ(rocsparse_create_mat_descr(&src), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_descr(&dest), rocsparse_status_success);

    // Set properties on source
    ASSERT_EQ(rocsparse_set_mat_index_base(src, rocsparse_index_base_one),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_type(src, rocsparse_matrix_type_symmetric),
              rocsparse_status_success);

    // Copy
    ASSERT_EQ(rocsparse_copy_mat_descr(dest, src), rocsparse_status_success);

    // Test invalid cases
    ASSERT_EQ(rocsparse_copy_mat_descr(nullptr, src), rocsparse_status_invalid_pointer);
    ASSERT_EQ(rocsparse_copy_mat_descr(dest, nullptr), rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_mat_descr(src), rocsparse_status_success);
    ASSERT_EQ(rocsparse_destroy_mat_descr(dest), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, MatDescrIndexBase)
{
    rocsparse_mat_descr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    // Test zero-based indexing
    ASSERT_EQ(rocsparse_set_mat_index_base(descr, rocsparse_index_base_zero),
              rocsparse_status_success);

    // Test one-based indexing
    ASSERT_EQ(rocsparse_set_mat_index_base(descr, rocsparse_index_base_one),
              rocsparse_status_success);

    // Test invalid cases
    ASSERT_EQ(rocsparse_set_mat_index_base(nullptr, rocsparse_index_base_zero),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, MatDescrType)
{
    rocsparse_mat_descr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    // Test all matrix types
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_symmetric),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_hermitian),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_triangular),
              rocsparse_status_success);

    // Test invalid cases
    ASSERT_EQ(rocsparse_set_mat_type(nullptr, rocsparse_matrix_type_general),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, MatDescrFillMode)
{
    rocsparse_mat_descr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    // Test fill modes
    ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_upper),
              rocsparse_status_success);

    // Test invalid cases
    ASSERT_EQ(rocsparse_set_mat_fill_mode(nullptr, rocsparse_fill_mode_lower),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, MatDescrDiagType)
{
    rocsparse_mat_descr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    // Test diagonal types
    ASSERT_EQ(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_unit),
              rocsparse_status_success);

    // Test invalid cases
    ASSERT_EQ(rocsparse_set_mat_diag_type(nullptr, rocsparse_diag_type_non_unit),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, MatDescrStorageMode)
{
    rocsparse_mat_descr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    // Test storage modes
    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted),
              rocsparse_status_success);

    // Test invalid cases
    ASSERT_EQ(rocsparse_set_mat_storage_mode(nullptr, rocsparse_storage_mode_sorted),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// HYB Matrix Tests
// =============================================================================

TEST(auxiliary_pre_checkin, HybMatCreateDestroy)
{
    rocsparse_hyb_mat hyb;
    ASSERT_EQ(rocsparse_create_hyb_mat(&hyb), rocsparse_status_success);
    ASSERT_NE(hyb, nullptr);
    ASSERT_EQ(rocsparse_destroy_hyb_mat(hyb), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, HybMatCreateNullptr)
{
    ASSERT_EQ(rocsparse_create_hyb_mat(nullptr), rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, HybMatDestroyNull)
{
    ASSERT_EQ(rocsparse_destroy_hyb_mat(nullptr), rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, HybMatCopy)
{
    rocsparse_hyb_mat src, dest;
    ASSERT_EQ(rocsparse_create_hyb_mat(&src), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_hyb_mat(&dest), rocsparse_status_success);

    // Copy (even though it's empty)
    ASSERT_EQ(rocsparse_copy_hyb_mat(dest, src), rocsparse_status_success);

    // Test invalid cases
    ASSERT_EQ(rocsparse_copy_hyb_mat(nullptr, src), rocsparse_status_invalid_pointer);
    ASSERT_EQ(rocsparse_copy_hyb_mat(dest, nullptr), rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_hyb_mat(src), rocsparse_status_success);
    ASSERT_EQ(rocsparse_destroy_hyb_mat(dest), rocsparse_status_success);
}

// =============================================================================
// Mat Info Tests
// =============================================================================

TEST(auxiliary_pre_checkin, MatInfoCreateDestroy)
{
    rocsparse_mat_info info;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, MatInfoCreateNullptr)
{
    ASSERT_EQ(rocsparse_create_mat_info(nullptr), rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, MatInfoCopy)
{
    rocsparse_mat_info src, dest;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_info(&dest), rocsparse_status_success);

    // Copy
    ASSERT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);

    // Test invalid cases
    ASSERT_EQ(rocsparse_copy_mat_info(nullptr, src), rocsparse_status_invalid_pointer);
    ASSERT_EQ(rocsparse_copy_mat_info(dest, nullptr), rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    ASSERT_EQ(rocsparse_destroy_mat_info(dest), rocsparse_status_success);
}

// =============================================================================
// Color Info Tests
// =============================================================================

TEST(auxiliary_pre_checkin, ColorInfoCreateDestroy)
{
    rocsparse_color_info info;
    ASSERT_EQ(rocsparse_create_color_info(&info), rocsparse_status_success);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(rocsparse_destroy_color_info(info), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ColorInfoCreateNullptr)
{
    ASSERT_EQ(rocsparse_create_color_info(nullptr), rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, ColorInfoCopy)
{
    rocsparse_color_info src, dest;
    ASSERT_EQ(rocsparse_create_color_info(&src), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_color_info(&dest), rocsparse_status_success);

    // Copy
    ASSERT_EQ(rocsparse_copy_color_info(dest, src), rocsparse_status_success);

    // Test invalid cases
    ASSERT_EQ(rocsparse_copy_color_info(nullptr, src), rocsparse_status_invalid_pointer);
    ASSERT_EQ(rocsparse_copy_color_info(dest, nullptr), rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_color_info(src), rocsparse_status_success);
    ASSERT_EQ(rocsparse_destroy_color_info(dest), rocsparse_status_success);
}

// =============================================================================
// Combined Test - Realistic Usage
// =============================================================================

TEST(auxiliary_pre_checkin, CombinedUsage)
{
    // Create handle
    rocsparse_handle handle;
    ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);

    // Configure handle
    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);

    // Create matrix descriptor
    rocsparse_mat_descr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_index_base(descr, rocsparse_index_base_zero),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general),
              rocsparse_status_success);

    // Create mat_info
    rocsparse_mat_info info;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    // Cleanup
    ASSERT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    ASSERT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
}
