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

TEST(auxiliary_pre_checkin, MatInfoDestroyNull)
{
    ASSERT_EQ(rocsparse_destroy_mat_info(nullptr), rocsparse_status_success);
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

TEST(auxiliary_pre_checkin, ColorInfoDestroyNull)
{
    ASSERT_EQ(rocsparse_destroy_color_info(nullptr), rocsparse_status_success);
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

// =============================================================================
// Getter Function Tests
// =============================================================================

TEST(auxiliary_pre_checkin, GetStatusName)
{
    // Test all status codes
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_success), "rocsparse_status_success");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_invalid_handle),
                 "rocsparse_status_invalid_handle");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_not_implemented),
                 "rocsparse_status_not_implemented");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_invalid_pointer),
                 "rocsparse_status_invalid_pointer");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_invalid_size),
                 "rocsparse_status_invalid_size");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_memory_error),
                 "rocsparse_status_memory_error");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_internal_error),
                 "rocsparse_status_internal_error");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_invalid_value),
                 "rocsparse_status_invalid_value");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_arch_mismatch),
                 "rocsparse_status_arch_mismatch");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_zero_pivot),
                 "rocsparse_status_zero_pivot");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_not_initialized),
                 "rocsparse_status_not_initialized");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_type_mismatch),
                 "rocsparse_status_type_mismatch");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_requires_sorted_storage),
                 "rocsparse_status_requires_sorted_storage");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_thrown_exception),
                 "rocsparse_status_thrown_exception");
    EXPECT_STREQ(rocsparse_get_status_name(rocsparse_status_continue), "rocsparse_status_continue");

    // Test unrecognized status code
    EXPECT_STREQ(rocsparse_get_status_name(static_cast<rocsparse_status>(999)),
                 "Unrecognized status code");
}

TEST(auxiliary_pre_checkin, GetStatusDescription)
{
    // Test all status codes
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_success),
                 "rocsparse operation was successful");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_invalid_handle),
                 "handle not initialized, invalid or null");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_not_implemented),
                 "function is not implemented");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_invalid_pointer),
                 "invalid pointer parameter");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_invalid_size),
                 "invalid size parameter");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_memory_error),
                 "failed memory allocation, copy, dealloc");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_internal_error),
                 "other internal library failure");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_invalid_value),
                 "invalid value parameter");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_arch_mismatch),
                 "device arch is not supported");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_zero_pivot),
                 "encountered zero pivot");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_not_initialized),
                 "descriptor has not been initialized");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_type_mismatch),
                 "index types do not match");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_requires_sorted_storage),
                 "sorted storage required");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_thrown_exception),
                 "exception being thrown");
    EXPECT_STREQ(rocsparse_get_status_description(rocsparse_status_continue),
                 "nothing preventing function to proceed");

    // Test unrecognized status code
    EXPECT_STREQ(rocsparse_get_status_description(static_cast<rocsparse_status>(999)),
                 "Unrecognized status code");
}

TEST(auxiliary_pre_checkin, GetMatIndexBase)
{
    rocsparse_mat_descr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    // Test default (should be zero-based)
    EXPECT_EQ(rocsparse_get_mat_index_base(descr), rocsparse_index_base_zero);

    // Set and get zero-based
    ASSERT_EQ(rocsparse_set_mat_index_base(descr, rocsparse_index_base_zero),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_index_base(descr), rocsparse_index_base_zero);

    // Set and get one-based
    ASSERT_EQ(rocsparse_set_mat_index_base(descr, rocsparse_index_base_one),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_index_base(descr), rocsparse_index_base_one);

    // Test nullptr (should return default)
    EXPECT_EQ(rocsparse_get_mat_index_base(nullptr), rocsparse_index_base_zero);

    ASSERT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, GetMatType)
{
    rocsparse_mat_descr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    // Test default (should be general)
    EXPECT_EQ(rocsparse_get_mat_type(descr), rocsparse_matrix_type_general);

    // Test all matrix types
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_type(descr), rocsparse_matrix_type_general);

    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_symmetric),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_type(descr), rocsparse_matrix_type_symmetric);

    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_hermitian),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_type(descr), rocsparse_matrix_type_hermitian);

    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_triangular),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_type(descr), rocsparse_matrix_type_triangular);

    // Test nullptr (should return default)
    EXPECT_EQ(rocsparse_get_mat_type(nullptr), rocsparse_matrix_type_general);

    ASSERT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, GetMatFillMode)
{
    rocsparse_mat_descr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    // Test default (should be lower)
    EXPECT_EQ(rocsparse_get_mat_fill_mode(descr), rocsparse_fill_mode_lower);

    // Test both fill modes
    ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_fill_mode(descr), rocsparse_fill_mode_lower);

    ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_upper),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_fill_mode(descr), rocsparse_fill_mode_upper);

    // Test nullptr (should return default)
    EXPECT_EQ(rocsparse_get_mat_fill_mode(nullptr), rocsparse_fill_mode_lower);

    ASSERT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, GetMatDiagType)
{
    rocsparse_mat_descr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    // Test default (should be non-unit)
    EXPECT_EQ(rocsparse_get_mat_diag_type(descr), rocsparse_diag_type_non_unit);

    // Test both diagonal types
    ASSERT_EQ(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_diag_type(descr), rocsparse_diag_type_non_unit);

    ASSERT_EQ(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_unit),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_diag_type(descr), rocsparse_diag_type_unit);

    // Test nullptr (should return default)
    EXPECT_EQ(rocsparse_get_mat_diag_type(nullptr), rocsparse_diag_type_non_unit);

    ASSERT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, GetMatStorageMode)
{
    rocsparse_mat_descr descr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    // Test default (should be sorted)
    EXPECT_EQ(rocsparse_get_mat_storage_mode(descr), rocsparse_storage_mode_sorted);

    // Test both storage modes
    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_storage_mode(descr), rocsparse_storage_mode_sorted);

    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_get_mat_storage_mode(descr), rocsparse_storage_mode_unsorted);

    // Test nullptr (should return default)
    EXPECT_EQ(rocsparse_get_mat_storage_mode(nullptr), rocsparse_storage_mode_sorted);

    ASSERT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// Sparse Vector Descriptor Tests
// =============================================================================

TEST(auxiliary_pre_checkin, SpvecDescrCreateDestroy)
{
    rocsparse_spvec_descr descr;
    int64_t               size    = 100;
    int64_t               nnz     = 10;
    void*                 indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values  = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spvec_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpvecDescrCreateNullptr)
{
    int64_t size    = 100;
    int64_t nnz     = 10;
    void*   indices = nullptr;
    void*   values  = nullptr;

    ASSERT_EQ(rocsparse_create_spvec_descr(nullptr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, SpvecDescrDestroyNull)
{
    ASSERT_EQ(rocsparse_destroy_spvec_descr(nullptr), rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, ConstSpvecDescrCreateDestroy)
{
    rocsparse_const_spvec_descr descr;
    int64_t                     size = 100;
    int64_t                     nnz  = 10;
    const void* indices = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer
    const void* values  = reinterpret_cast<const void*>(0x2000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_spvec_descr(&descr,
                                                 size,
                                                 nnz,
                                                 indices,
                                                 values,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spvec_descr(descr), rocsparse_status_success);
}

// Validation tests for rocsparse_create_spvec_descr parameter checks (lines 1525-1533)

TEST(auxiliary_pre_checkin, SpvecDescrCreateNullIndices)
{
    rocsparse_spvec_descr descr;
    int64_t               size    = 100;
    int64_t               nnz     = 10; // Non-zero nnz
    void*                 indices = nullptr; // Null indices should fail
    void*                 values  = reinterpret_cast<void*>(0x2000);

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, SpvecDescrCreateNullValues)
{
    rocsparse_spvec_descr descr;
    int64_t               size    = 100;
    int64_t               nnz     = 10; // Non-zero nnz
    void*                 indices = reinterpret_cast<void*>(0x1000);
    void*                 values  = nullptr; // Null values should fail

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, SpvecDescrCreateZeroNnzNullPointers)
{
    rocsparse_spvec_descr descr;
    int64_t               size    = 100;
    int64_t               nnz     = 0; // Zero nnz allows null pointers
    void*                 indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values  = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spvec_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpvecDescrCreateNnzGreaterThanSize)
{
    rocsparse_spvec_descr descr;
    int64_t               size    = 100;
    int64_t               nnz     = 101; // nnz > size should fail
    void*                 indices = reinterpret_cast<void*>(0x1000);
    void*                 values  = reinterpret_cast<void*>(0x2000);

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_invalid_size);
}

TEST(auxiliary_pre_checkin, SpvecDescrCreateNegativeSize)
{
    rocsparse_spvec_descr descr;
    int64_t               size    = -1; // Negative size should fail
    int64_t               nnz     = 10;
    void*                 indices = reinterpret_cast<void*>(0x1000);
    void*                 values  = reinterpret_cast<void*>(0x2000);

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_invalid_size);
}

TEST(auxiliary_pre_checkin, SpvecDescrCreateNegativeNnz)
{
    rocsparse_spvec_descr descr;
    int64_t               size    = 100;
    int64_t               nnz     = -1; // Negative nnz should fail
    void*                 indices = reinterpret_cast<void*>(0x1000);
    void*                 values  = reinterpret_cast<void*>(0x2000);

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_invalid_size);
}

TEST(auxiliary_pre_checkin, SpvecDescrCreateInvalidIndexType)
{
    rocsparse_spvec_descr descr;
    int64_t               size    = 100;
    int64_t               nnz     = 10;
    void*                 indices = reinterpret_cast<void*>(0x1000);
    void*                 values  = reinterpret_cast<void*>(0x2000);

    // Use an invalid enum value (out of range)
    rocsparse_indextype invalid_idx_type = static_cast<rocsparse_indextype>(999);

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           invalid_idx_type,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_invalid_value);
}

TEST(auxiliary_pre_checkin, SpvecDescrCreateInvalidIndexBase)
{
    rocsparse_spvec_descr descr;
    int64_t               size    = 100;
    int64_t               nnz     = 10;
    void*                 indices = reinterpret_cast<void*>(0x1000);
    void*                 values  = reinterpret_cast<void*>(0x2000);

    // Use an invalid enum value (out of range)
    rocsparse_index_base invalid_idx_base = static_cast<rocsparse_index_base>(999);

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           invalid_idx_base,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_invalid_value);
}

TEST(auxiliary_pre_checkin, SpvecDescrCreateInvalidDataType)
{
    rocsparse_spvec_descr descr;
    int64_t               size    = 100;
    int64_t               nnz     = 10;
    void*                 indices = reinterpret_cast<void*>(0x1000);
    void*                 values  = reinterpret_cast<void*>(0x2000);

    // Use an invalid enum value (out of range)
    rocsparse_datatype invalid_data_type = static_cast<rocsparse_datatype>(999);

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           invalid_data_type),
              rocsparse_status_invalid_value);
}

// =============================================================================
// Sparse Matrix COO Descriptor Tests
// =============================================================================

TEST(auxiliary_pre_checkin, CooDescrCreateDestroy)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 100;
    int64_t               nnz         = 10;
    void*                 row_indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_coo_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_indices,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, CooDescrCreateNullptr)
{
    int64_t rows        = 100;
    int64_t cols        = 100;
    int64_t nnz         = 10;
    void*   row_indices = nullptr;
    void*   col_indices = nullptr;
    void*   values      = nullptr;

    ASSERT_EQ(rocsparse_create_coo_descr(nullptr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_indices,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, ConstCooDescrCreateDestroy)
{
    rocsparse_const_spmat_descr descr;
    int64_t                     rows = 100;
    int64_t                     cols = 100;
    int64_t                     nnz  = 10;
    const void* row_indices = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer
    const void* col_indices = reinterpret_cast<const void*>(0x2000); // Non-null dummy pointer
    const void* values      = reinterpret_cast<const void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_coo_descr(&descr,
                                               rows,
                                               cols,
                                               nnz,
                                               row_indices,
                                               col_indices,
                                               values,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, CooAosDescrCreateDestroy)
{
    rocsparse_spmat_descr descr;
    int64_t               rows    = 100;
    int64_t               cols    = 100;
    int64_t               nnz     = 10;
    void*                 indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values  = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_coo_aos_descr(&descr,
                                             rows,
                                             cols,
                                             nnz,
                                             indices,
                                             values,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// Sparse Matrix CSR Descriptor Tests
// =============================================================================

TEST(auxiliary_pre_checkin, CsrDescrCreateDestroy)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 100;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstCsrDescrCreateDestroy)
{
    rocsparse_const_spmat_descr descr;
    int64_t                     rows = 100;
    int64_t                     cols = 100;
    int64_t                     nnz  = 10;
    const void* row_ptrs    = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer
    const void* col_indices = reinterpret_cast<const void*>(0x2000); // Non-null dummy pointer
    const void* values      = reinterpret_cast<const void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_csr_descr(&descr,
                                               rows,
                                               cols,
                                               nnz,
                                               row_ptrs,
                                               col_indices,
                                               values,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// Sparse Matrix CSC Descriptor Tests
// =============================================================================

TEST(auxiliary_pre_checkin, CscDescrCreateDestroy)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 100;
    int64_t               nnz         = 10;
    void*                 col_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 row_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csc_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         col_ptrs,
                                         row_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstCscDescrCreateDestroy)
{
    rocsparse_const_spmat_descr descr;
    int64_t                     rows = 100;
    int64_t                     cols = 100;
    int64_t                     nnz  = 10;
    const void* col_ptrs    = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer
    const void* row_indices = reinterpret_cast<const void*>(0x2000); // Non-null dummy pointer
    const void* values      = reinterpret_cast<const void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_csc_descr(&descr,
                                               rows,
                                               cols,
                                               nnz,
                                               col_ptrs,
                                               row_indices,
                                               values,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// Sparse Matrix ELL Descriptor Tests
// =============================================================================

TEST(auxiliary_pre_checkin, EllDescrCreateDestroy)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 100;
    int64_t               ell_width   = 5;
    void*                 col_indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer

    // Note: parameter order is rows, cols, col_indices, values, ell_width
    ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                         rows,
                                         cols,
                                         col_indices,
                                         values,
                                         ell_width,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// Sparse Matrix BELL Descriptor Tests
// =============================================================================

TEST(auxiliary_pre_checkin, BellDescrCreateDestroy)
{
    rocsparse_spmat_descr descr;
    int64_t               rows            = 100;
    int64_t               cols            = 100;
    rocsparse_direction   dir             = rocsparse_direction_row;
    int64_t               block_dim       = 2;
    int64_t               ell_block_width = 5;
    void*                 col_indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_bell_descr(&descr,
                                          rows,
                                          cols,
                                          dir,
                                          block_dim,
                                          ell_block_width,
                                          col_indices,
                                          values,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// Dense Vector Descriptor Tests
// =============================================================================

TEST(auxiliary_pre_checkin, DnvecDescrCreateDestroy)
{
    rocsparse_dnvec_descr descr;
    int64_t               size   = 100;
    void*                 values = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_dnvec_descr(&descr, size, values, rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_dnvec_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, DnvecDescrCreateNullptr)
{
    int64_t size   = 100;
    void*   values = nullptr;

    ASSERT_EQ(rocsparse_create_dnvec_descr(nullptr, size, values, rocsparse_datatype_f32_r),
              rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, DnvecDescrDestroyNull)
{
    ASSERT_EQ(rocsparse_destroy_dnvec_descr(nullptr), rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, ConstDnvecDescrCreateDestroy)
{
    rocsparse_const_dnvec_descr descr;
    int64_t                     size = 100;
    const void* values = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_dnvec_descr(&descr, size, values, rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_dnvec_descr(descr), rocsparse_status_success);
}

// =============================================================================
// Dense Matrix Descriptor Tests
// =============================================================================

TEST(auxiliary_pre_checkin, DnmatDescrCreateDestroy)
{
    rocsparse_dnmat_descr descr;
    int64_t               rows   = 100;
    int64_t               cols   = 50;
    int64_t               ld     = 100;
    void*                 values = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, DnmatDescrCreateNullptr)
{
    int64_t rows   = 100;
    int64_t cols   = 50;
    int64_t ld     = 100;
    void*   values = nullptr;

    ASSERT_EQ(
        rocsparse_create_dnmat_descr(
            nullptr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_column),
        rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, DnmatDescrDestroyNull)
{
    ASSERT_EQ(rocsparse_destroy_dnmat_descr(nullptr), rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, DnmatDescrCreateDestroyRowMajor)
{
    rocsparse_dnmat_descr descr;
    int64_t               rows   = 100;
    int64_t               cols   = 50;
    int64_t               ld     = 50; // For row-major, ld must be >= cols
    void*                 values = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_row),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, DnmatDescrCreateRowMajorInvalidLd)
{
    rocsparse_dnmat_descr descr;
    int64_t               rows   = 100;
    int64_t               cols   = 50;
    int64_t               ld     = 49; // Invalid: For row-major, ld must be >= cols
    void*                 values = reinterpret_cast<void*>(0x1000);

    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_row),
              rocsparse_status_invalid_size);
}

TEST(auxiliary_pre_checkin, DnmatDescrCreateColumnMajorInvalidLd)
{
    rocsparse_dnmat_descr descr;
    int64_t               rows   = 100;
    int64_t               cols   = 50;
    int64_t               ld     = 99; // Invalid: For column-major, ld must be >= rows
    void*                 values = reinterpret_cast<void*>(0x1000);

    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_invalid_size);
}

TEST(auxiliary_pre_checkin, ConstDnmatDescrCreateDestroy)
{
    rocsparse_const_dnmat_descr descr;
    int64_t                     rows = 100;
    int64_t                     cols = 50;
    int64_t                     ld   = 100;
    const void* values = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstDnmatDescrCreateDestroyRowMajor)
{
    rocsparse_const_dnmat_descr descr;
    int64_t                     rows = 100;
    int64_t                     cols = 50;
    int64_t                     ld   = 50; // For row-major, ld must be >= cols
    const void* values = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_row),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstDnmatDescrCreateRowMajorInvalidLd)
{
    rocsparse_const_dnmat_descr descr;
    int64_t                     rows   = 100;
    int64_t                     cols   = 50;
    int64_t                     ld     = 49; // Invalid: For row-major, ld must be >= cols
    const void*                 values = reinterpret_cast<const void*>(0x1000);

    ASSERT_EQ(rocsparse_create_const_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_row),
              rocsparse_status_invalid_size);
}

TEST(auxiliary_pre_checkin, ConstDnmatDescrCreateColumnMajorInvalidLd)
{
    rocsparse_const_dnmat_descr descr;
    int64_t                     rows   = 100;
    int64_t                     cols   = 50;
    int64_t                     ld     = 99; // Invalid: For column-major, ld must be >= rows
    const void*                 values = reinterpret_cast<const void*>(0x1000);

    ASSERT_EQ(rocsparse_create_const_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_invalid_size);
}

// =============================================================================
// SpGEAM Descriptor Tests
// =============================================================================

TEST(auxiliary_pre_checkin, SpgeamDescrCreateDestroy)
{
    rocsparse_spgeam_descr descr;

    ASSERT_EQ(rocsparse_create_spgeam_descr(&descr), rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spgeam_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpgeamDescrCreateNullptr)
{
    ASSERT_EQ(rocsparse_create_spgeam_descr(nullptr), rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, SpgeamDescrDestroyNull)
{
    ASSERT_EQ(rocsparse_destroy_spgeam_descr(nullptr), rocsparse_status_invalid_pointer);
}

// =============================================================================
// Sparse Vector Get/Set Tests
// =============================================================================

TEST(auxiliary_pre_checkin, SpvecGet)
{
    rocsparse_spvec_descr descr;
    int64_t               size    = 100;
    int64_t               nnz     = 10;
    void*                 indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values  = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              size_out, nnz_out;
    void*                indices_out;
    void*                values_out;
    rocsparse_indextype  idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_spvec_get(descr,
                                  &size_out,
                                  &nnz_out,
                                  &indices_out,
                                  &values_out,
                                  &idx_type,
                                  &idx_base,
                                  &data_type),
              rocsparse_status_success);
    EXPECT_EQ(size_out, size);
    EXPECT_EQ(nnz_out, nnz);

    ASSERT_EQ(rocsparse_destroy_spvec_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstSpvecGet)
{
    rocsparse_const_spvec_descr descr;
    int64_t                     size = 100;
    int64_t                     nnz  = 10;
    const void* indices = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer
    const void* values  = reinterpret_cast<const void*>(0x2000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_spvec_descr(&descr,
                                                 size,
                                                 nnz,
                                                 indices,
                                                 values,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              size_out, nnz_out;
    const void*          indices_out;
    const void*          values_out;
    rocsparse_indextype  idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_spvec_get(descr,
                                        &size_out,
                                        &nnz_out,
                                        &indices_out,
                                        &values_out,
                                        &idx_type,
                                        &idx_base,
                                        &data_type),
              rocsparse_status_success);
    EXPECT_EQ(size_out, size);
    EXPECT_EQ(nnz_out, nnz);

    ASSERT_EQ(rocsparse_destroy_spvec_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpvecGetIndexBase)
{
    rocsparse_spvec_descr descr;
    int64_t               size    = 100;
    int64_t               nnz     = 10;
    void*                 indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values  = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_one,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_index_base idx_base;
    ASSERT_EQ(rocsparse_spvec_get_index_base(descr, &idx_base), rocsparse_status_success);
    EXPECT_EQ(idx_base, rocsparse_index_base_one);

    ASSERT_EQ(rocsparse_destroy_spvec_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpvecGetSetValues)
{
    rocsparse_spvec_descr descr;
    int64_t               size       = 100;
    int64_t               nnz        = 10;
    void*                 indices    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values     = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 new_values = reinterpret_cast<void*>(0x12345678);

    ASSERT_EQ(rocsparse_create_spvec_descr(&descr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           rocsparse_indextype_i32,
                                           rocsparse_index_base_zero,
                                           rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Set values
    ASSERT_EQ(rocsparse_spvec_set_values(descr, new_values), rocsparse_status_success);

    // Get values
    void* values_out;
    ASSERT_EQ(rocsparse_spvec_get_values(descr, &values_out), rocsparse_status_success);
    EXPECT_EQ(values_out, new_values);

    ASSERT_EQ(rocsparse_destroy_spvec_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstSpvecGetValues)
{
    rocsparse_const_spvec_descr descr;
    int64_t                     size = 100;
    int64_t                     nnz  = 10;
    const void* indices = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer
    const void* values  = reinterpret_cast<const void*>(0x12345678);

    ASSERT_EQ(rocsparse_create_const_spvec_descr(&descr,
                                                 size,
                                                 nnz,
                                                 indices,
                                                 values,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
              rocsparse_status_success);

    const void* values_out;
    ASSERT_EQ(rocsparse_const_spvec_get_values(descr, &values_out), rocsparse_status_success);
    EXPECT_EQ(values_out, values);

    ASSERT_EQ(rocsparse_destroy_spvec_descr(descr), rocsparse_status_success);
}

// =============================================================================
// Sparse Matrix Common Get/Set Tests
// =============================================================================

TEST(auxiliary_pre_checkin, SpmatGetSize)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t rows_out, cols_out, nnz_out;
    ASSERT_EQ(rocsparse_spmat_get_size(descr, &rows_out, &cols_out, &nnz_out),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(nnz_out, nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetFormat)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_format format;
    ASSERT_EQ(rocsparse_spmat_get_format(descr, &format), rocsparse_status_success);
    EXPECT_EQ(format, rocsparse_format_csr);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetIndexBase)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_one,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_index_base idx_base;
    ASSERT_EQ(rocsparse_spmat_get_index_base(descr, &idx_base), rocsparse_status_success);
    EXPECT_EQ(idx_base, rocsparse_index_base_one);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetSetValues)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer
    void*                 new_values  = reinterpret_cast<void*>(0x12345678);

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Set values
    ASSERT_EQ(rocsparse_spmat_set_values(descr, new_values), rocsparse_status_success);

    // Get values
    void* values_out;
    ASSERT_EQ(rocsparse_spmat_get_values(descr, &values_out), rocsparse_status_success);
    EXPECT_EQ(values_out, new_values);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstSpmatGetValues)
{
    rocsparse_const_spmat_descr descr;
    int64_t                     rows = 100;
    int64_t                     cols = 50;
    int64_t                     nnz  = 10;
    const void* row_ptrs    = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer
    const void* col_indices = reinterpret_cast<const void*>(0x2000); // Non-null dummy pointer
    const void* values      = reinterpret_cast<const void*>(0x12345678);

    ASSERT_EQ(rocsparse_create_const_csr_descr(&descr,
                                               rows,
                                               cols,
                                               nnz,
                                               row_ptrs,
                                               col_indices,
                                               values,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r),
              rocsparse_status_success);

    const void* values_out;
    ASSERT_EQ(rocsparse_const_spmat_get_values(descr, &values_out), rocsparse_status_success);
    EXPECT_EQ(values_out, values);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetSetNnz)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Get nnz
    int64_t nnz_out;
    ASSERT_EQ(rocsparse_spmat_get_nnz(descr, &nnz_out), rocsparse_status_success);
    EXPECT_EQ(nnz_out, nnz);

    // Set nnz
    int64_t new_nnz = 20;
    ASSERT_EQ(rocsparse_spmat_set_nnz(descr, new_nnz), rocsparse_status_success);

    // Verify
    ASSERT_EQ(rocsparse_spmat_get_nnz(descr, &nnz_out), rocsparse_status_success);
    EXPECT_EQ(nnz_out, new_nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// Tests for rocsparse_spmat_set_nnz with different formats (lines 4311-4360)

TEST(auxiliary_pre_checkin, SpmatSetNnzBsr)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t new_nnz = 10;
    ASSERT_EQ(rocsparse_spmat_set_nnz(descr, new_nnz), rocsparse_status_success);

    int64_t nnz_out;
    ASSERT_EQ(rocsparse_spmat_get_nnz(descr, &nnz_out), rocsparse_status_success);
    EXPECT_EQ(nnz_out, new_nnz * block_dim * block_dim);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatSetNnzCsc)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 col_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 row_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_csc_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         col_ptrs,
                                         row_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t new_nnz = 20;
    ASSERT_EQ(rocsparse_spmat_set_nnz(descr, new_nnz), rocsparse_status_success);

    int64_t nnz_out;
    ASSERT_EQ(rocsparse_spmat_get_nnz(descr, &nnz_out), rocsparse_status_success);
    EXPECT_EQ(nnz_out, new_nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatSetNnzCoo)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_indices = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_coo_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_indices,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t new_nnz = 20;
    ASSERT_EQ(rocsparse_spmat_set_nnz(descr, new_nnz), rocsparse_status_success);

    int64_t nnz_out;
    ASSERT_EQ(rocsparse_spmat_get_nnz(descr, &nnz_out), rocsparse_status_success);
    EXPECT_EQ(nnz_out, new_nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatSetNnzCooAos)
{
    rocsparse_spmat_descr descr;
    int64_t               rows    = 100;
    int64_t               cols    = 50;
    int64_t               nnz     = 10;
    void*                 coo_ind = reinterpret_cast<void*>(0x1000);
    void*                 values  = reinterpret_cast<void*>(0x2000);

    ASSERT_EQ(rocsparse_create_coo_aos_descr(&descr,
                                             rows,
                                             cols,
                                             nnz,
                                             coo_ind,
                                             values,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t new_nnz = 20;
    ASSERT_EQ(rocsparse_spmat_set_nnz(descr, new_nnz), rocsparse_status_success);

    int64_t nnz_out;
    ASSERT_EQ(rocsparse_spmat_get_nnz(descr, &nnz_out), rocsparse_status_success);
    EXPECT_EQ(nnz_out, new_nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatSetNnzSell)
{
    rocsparse_spmat_descr descr;
    int64_t               rows          = 100;
    int64_t               cols          = 50;
    int64_t               nnz           = 10;
    int64_t               slice_size    = 64;
    int64_t               colval_size   = 100;
    void*                 slice_offsets = reinterpret_cast<void*>(0x1000);
    void*                 col_indices   = reinterpret_cast<void*>(0x2000);
    void*                 values        = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_sell_descr(&descr,
                                          rows,
                                          cols,
                                          nnz,
                                          slice_size,
                                          colval_size,
                                          slice_offsets,
                                          col_indices,
                                          values,
                                          rocsparse_indextype_i32,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t new_nnz = 20;
    ASSERT_EQ(rocsparse_spmat_set_nnz(descr, new_nnz), rocsparse_status_success);

    int64_t nnz_out;
    ASSERT_EQ(rocsparse_spmat_get_nnz(descr, &nnz_out), rocsparse_status_success);
    EXPECT_EQ(nnz_out, new_nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatSetNnzBellInvalid)
{
    rocsparse_spmat_descr descr;
    int64_t               rows            = 100;
    int64_t               cols            = 100;
    rocsparse_direction   dir             = rocsparse_direction_row;
    int64_t               block_dim       = 2;
    int64_t               ell_block_width = 5;
    void*                 col_indices     = reinterpret_cast<void*>(0x2000);
    void*                 values          = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bell_descr(&descr,
                                          rows,
                                          cols,
                                          dir,
                                          block_dim,
                                          ell_block_width,
                                          col_indices,
                                          values,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t new_nnz = 20;
    ASSERT_EQ(rocsparse_spmat_set_nnz(descr, new_nnz), rocsparse_status_invalid_value);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatSetNnzEllInvalid)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               ell_width   = 5;
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                         rows,
                                         cols,
                                         col_indices,
                                         values,
                                         ell_width,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t new_nnz = 20;
    ASSERT_EQ(rocsparse_spmat_set_nnz(descr, new_nnz), rocsparse_status_invalid_value);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// Tests for rocsparse_spmat_get_nnz with different formats (lines 4259-4275)

TEST(auxiliary_pre_checkin, SpmatGetNnzBell)
{
    rocsparse_spmat_descr descr;
    int64_t               rows            = 100;
    int64_t               cols            = 100;
    rocsparse_direction   dir             = rocsparse_direction_row;
    int64_t               block_dim       = 2;
    int64_t               ell_block_width = 5;
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_bell_descr(&descr,
                                          rows,
                                          cols,
                                          dir,
                                          block_dim,
                                          ell_block_width,
                                          col_indices,
                                          values,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Get nnz - should be ell_cols * rows * block_dim * block_dim
    int64_t nnz_out;
    ASSERT_EQ(rocsparse_spmat_get_nnz(descr, &nnz_out), rocsparse_status_success);
    int64_t expected_nnz = ell_block_width * rows * block_dim * block_dim;
    EXPECT_EQ(nnz_out, expected_nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetNnzEll)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               ell_width   = 5;
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                         rows,
                                         cols,
                                         col_indices,
                                         values,
                                         ell_width,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Get nnz - should be ell_width * rows
    int64_t nnz_out;
    ASSERT_EQ(rocsparse_spmat_get_nnz(descr, &nnz_out), rocsparse_status_success);
    int64_t expected_nnz = ell_width * rows;
    EXPECT_EQ(nnz_out, expected_nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetNnzBsr)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Get nnz - should be nnz * block_dim * block_dim
    int64_t nnz_out;
    ASSERT_EQ(rocsparse_spmat_get_nnz(descr, &nnz_out), rocsparse_status_success);
    int64_t expected_nnz = block_nnz * block_dim * block_dim;
    EXPECT_EQ(nnz_out, expected_nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetSetStridedBatch)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Set strided batch
    int batch_count = 5;
    ASSERT_EQ(rocsparse_spmat_set_strided_batch(descr, batch_count), rocsparse_status_success);

    // Get strided batch
    int batch_count_out;
    ASSERT_EQ(rocsparse_spmat_get_strided_batch(descr, &batch_count_out), rocsparse_status_success);
    EXPECT_EQ(batch_count_out, batch_count);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetSetAttribute)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Set fill mode attribute
    rocsparse_fill_mode fill_mode = rocsparse_fill_mode_upper;
    ASSERT_EQ(rocsparse_spmat_set_attribute(
                  descr, rocsparse_spmat_fill_mode, &fill_mode, sizeof(fill_mode)),
              rocsparse_status_success);

    // Get fill mode attribute
    rocsparse_fill_mode fill_mode_out;
    ASSERT_EQ(rocsparse_spmat_get_attribute(
                  descr, rocsparse_spmat_fill_mode, &fill_mode_out, sizeof(fill_mode_out)),
              rocsparse_status_success);
    EXPECT_EQ(fill_mode_out, fill_mode);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// Tests for additional rocsparse_spmat_get_attribute attributes (lines 4519-4548)

TEST(auxiliary_pre_checkin, SpmatGetSetAttributeDiagType)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Set diag type attribute
    rocsparse_diag_type diag_type = rocsparse_diag_type_unit;
    ASSERT_EQ(rocsparse_spmat_set_attribute(
                  descr, rocsparse_spmat_diag_type, &diag_type, sizeof(diag_type)),
              rocsparse_status_success);

    // Get diag type attribute
    rocsparse_diag_type diag_type_out;
    ASSERT_EQ(rocsparse_spmat_get_attribute(
                  descr, rocsparse_spmat_diag_type, &diag_type_out, sizeof(diag_type_out)),
              rocsparse_status_success);
    EXPECT_EQ(diag_type_out, diag_type);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetSetAttributeMatrixType)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Set matrix type attribute
    rocsparse_matrix_type matrix_type = rocsparse_matrix_type_symmetric;
    ASSERT_EQ(rocsparse_spmat_set_attribute(
                  descr, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)),
              rocsparse_status_success);

    // Get matrix type attribute
    rocsparse_matrix_type matrix_type_out;
    ASSERT_EQ(rocsparse_spmat_get_attribute(
                  descr, rocsparse_spmat_matrix_type, &matrix_type_out, sizeof(matrix_type_out)),
              rocsparse_status_success);
    EXPECT_EQ(matrix_type_out, matrix_type);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetSetAttributeStorageMode)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Set storage mode attribute
    rocsparse_storage_mode storage_mode = rocsparse_storage_mode_sorted;
    ASSERT_EQ(rocsparse_spmat_set_attribute(
                  descr, rocsparse_spmat_storage_mode, &storage_mode, sizeof(storage_mode)),
              rocsparse_status_success);

    // Get storage mode attribute
    rocsparse_storage_mode storage_mode_out;
    ASSERT_EQ(rocsparse_spmat_get_attribute(
                  descr, rocsparse_spmat_storage_mode, &storage_mode_out, sizeof(storage_mode_out)),
              rocsparse_status_success);
    EXPECT_EQ(storage_mode_out, storage_mode);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetAttributeDiagTypeInvalidSize)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_diag_type diag_type_out;
    // Try with wrong size (should fail)
    ASSERT_EQ(rocsparse_spmat_get_attribute(
                  descr, rocsparse_spmat_diag_type, &diag_type_out, sizeof(diag_type_out) - 1),
              rocsparse_status_invalid_size);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetAttributeMatrixTypeInvalidSize)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_matrix_type matrix_type_out;
    // Try with wrong size (should fail)
    ASSERT_EQ(
        rocsparse_spmat_get_attribute(
            descr, rocsparse_spmat_matrix_type, &matrix_type_out, sizeof(matrix_type_out) + 1),
        rocsparse_status_invalid_size);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatGetAttributeStorageModeInvalidSize)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_storage_mode storage_mode_out;
    // Try with wrong size (should fail)
    ASSERT_EQ(
        rocsparse_spmat_get_attribute(descr, rocsparse_spmat_storage_mode, &storage_mode_out, 1),
        rocsparse_status_invalid_size);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatSetAttributeDiagTypeInvalidSize)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_diag_type diag_type = rocsparse_diag_type_unit;
    // Try with wrong size (should fail)
    ASSERT_EQ(rocsparse_spmat_set_attribute(
                  descr, rocsparse_spmat_diag_type, &diag_type, sizeof(diag_type) + 1),
              rocsparse_status_invalid_size);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatSetAttributeMatrixTypeInvalidSize)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_matrix_type matrix_type = rocsparse_matrix_type_symmetric;
    // Try with wrong size (should fail)
    ASSERT_EQ(rocsparse_spmat_set_attribute(descr, rocsparse_spmat_matrix_type, &matrix_type, 1),
              rocsparse_status_invalid_size);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpmatSetAttributeStorageModeInvalidSize)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_storage_mode storage_mode = rocsparse_storage_mode_sorted;
    // Try with wrong size (should fail)
    ASSERT_EQ(rocsparse_spmat_set_attribute(
                  descr, rocsparse_spmat_storage_mode, &storage_mode, sizeof(storage_mode) - 1),
              rocsparse_status_invalid_size);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// COO Matrix Get/Set Tests
// =============================================================================

TEST(auxiliary_pre_checkin, CooGet)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_coo_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_indices,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, nnz_out;
    void*                row_indices_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_coo_get(descr,
                                &rows_out,
                                &cols_out,
                                &nnz_out,
                                &row_indices_out,
                                &col_indices_out,
                                &values_out,
                                &idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(nnz_out, nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstCooGet)
{
    rocsparse_const_spmat_descr descr;
    int64_t                     rows = 100;
    int64_t                     cols = 50;
    int64_t                     nnz  = 10;
    const void* row_indices          = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    const void* col_indices          = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    const void* values               = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_coo_descr(&descr,
                                               rows,
                                               cols,
                                               nnz,
                                               row_indices,
                                               col_indices,
                                               values,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, nnz_out;
    const void*          row_indices_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_coo_get(descr,
                                      &rows_out,
                                      &cols_out,
                                      &nnz_out,
                                      &row_indices_out,
                                      &col_indices_out,
                                      &values_out,
                                      &idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(nnz_out, nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, CooAosGet)
{
    rocsparse_spmat_descr descr;
    int64_t               rows    = 100;
    int64_t               cols    = 50;
    int64_t               nnz     = 10;
    void*                 indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values  = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_coo_aos_descr(&descr,
                                             rows,
                                             cols,
                                             nnz,
                                             indices,
                                             values,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, nnz_out;
    void*                indices_out;
    void*                values_out;
    rocsparse_indextype  idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_coo_aos_get(descr,
                                    &rows_out,
                                    &cols_out,
                                    &nnz_out,
                                    &indices_out,
                                    &values_out,
                                    &idx_type,
                                    &idx_base,
                                    &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(nnz_out, nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstCooAosGet)
{
    rocsparse_spmat_descr descr;
    int64_t               rows    = 100;
    int64_t               cols    = 50;
    int64_t               nnz     = 10;
    void*                 indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values  = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_coo_aos_descr(&descr,
                                             rows,
                                             cols,
                                             nnz,
                                             indices,
                                             values,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, nnz_out;
    const void*          indices_out;
    const void*          values_out;
    rocsparse_indextype  idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_coo_aos_get(descr,
                                          &rows_out,
                                          &cols_out,
                                          &nnz_out,
                                          &indices_out,
                                          &values_out,
                                          &idx_type,
                                          &idx_base,
                                          &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(nnz_out, nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, CooSetPointers)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_coo_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_indices,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    void* new_row_indices = reinterpret_cast<void*>(0x1000);
    void* new_col_indices = reinterpret_cast<void*>(0x2000);
    void* new_values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_coo_set_pointers(descr, new_row_indices, new_col_indices, new_values),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, CooSetStridedBatch)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_coo_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_indices,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int     batch_count  = 5;
    int64_t batch_stride = 1000;
    ASSERT_EQ(rocsparse_coo_set_strided_batch(descr, batch_count, batch_stride),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, CooAosSetPointers)
{
    rocsparse_spmat_descr descr;
    int64_t               rows    = 100;
    int64_t               cols    = 50;
    int64_t               nnz     = 10;
    void*                 coo_ind = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values  = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_coo_aos_descr(&descr,
                                             rows,
                                             cols,
                                             nnz,
                                             coo_ind,
                                             values,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
              rocsparse_status_success);

    void* new_coo_ind = reinterpret_cast<void*>(0x3000);
    void* new_values  = reinterpret_cast<void*>(0x4000);

    ASSERT_EQ(rocsparse_coo_aos_set_pointers(descr, new_coo_ind, new_values),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, CooAosSetPointersNullDescr)
{
    void* coo_ind = reinterpret_cast<void*>(0x1000);
    void* values  = reinterpret_cast<void*>(0x2000);

    ASSERT_EQ(rocsparse_coo_aos_set_pointers(nullptr, coo_ind, values),
              rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, CooAosSetPointersNullInd)
{
    rocsparse_spmat_descr descr;
    int64_t               rows    = 100;
    int64_t               cols    = 50;
    int64_t               nnz     = 10;
    void*                 coo_ind = reinterpret_cast<void*>(0x1000);
    void*                 values  = reinterpret_cast<void*>(0x2000);

    ASSERT_EQ(rocsparse_create_coo_aos_descr(&descr,
                                             rows,
                                             cols,
                                             nnz,
                                             coo_ind,
                                             values,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
              rocsparse_status_success);

    void* new_values = reinterpret_cast<void*>(0x4000);

    ASSERT_EQ(rocsparse_coo_aos_set_pointers(descr, nullptr, new_values),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, CooAosSetPointersNullVal)
{
    rocsparse_spmat_descr descr;
    int64_t               rows    = 100;
    int64_t               cols    = 50;
    int64_t               nnz     = 10;
    void*                 coo_ind = reinterpret_cast<void*>(0x1000);
    void*                 values  = reinterpret_cast<void*>(0x2000);

    ASSERT_EQ(rocsparse_create_coo_aos_descr(&descr,
                                             rows,
                                             cols,
                                             nnz,
                                             coo_ind,
                                             values,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
              rocsparse_status_success);

    void* new_coo_ind = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_coo_aos_set_pointers(descr, new_coo_ind, nullptr),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// CSR Matrix Get/Set Tests
// =============================================================================

TEST(auxiliary_pre_checkin, CsrGet)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, nnz_out;
    void*                row_ptrs_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_csr_get(descr,
                                &rows_out,
                                &cols_out,
                                &nnz_out,
                                &row_ptrs_out,
                                &col_indices_out,
                                &values_out,
                                &row_idx_type,
                                &col_idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(nnz_out, nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstCsrGet)
{
    rocsparse_const_spmat_descr descr;
    int64_t                     rows = 100;
    int64_t                     cols = 50;
    int64_t                     nnz  = 10;
    const void* row_ptrs             = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    const void* col_indices          = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    const void* values               = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_csr_descr(&descr,
                                               rows,
                                               cols,
                                               nnz,
                                               row_ptrs,
                                               col_indices,
                                               values,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, nnz_out;
    const void*          row_ptrs_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_csr_get(descr,
                                      &rows_out,
                                      &cols_out,
                                      &nnz_out,
                                      &row_ptrs_out,
                                      &col_indices_out,
                                      &values_out,
                                      &row_idx_type,
                                      &col_idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(nnz_out, nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, CsrSetPointers)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    void* new_row_ptrs    = reinterpret_cast<void*>(0x1000);
    void* new_col_indices = reinterpret_cast<void*>(0x2000);
    void* new_values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_csr_set_pointers(descr, new_row_ptrs, new_col_indices, new_values),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, CsrSetStridedBatch)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csr_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int     batch_count                 = 5;
    int64_t offsets_batch_stride        = 500;
    int64_t columns_values_batch_stride = 1000;
    ASSERT_EQ(rocsparse_csr_set_strided_batch(
                  descr, batch_count, offsets_batch_stride, columns_values_batch_stride),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// CSC Matrix Get/Set Tests
// =============================================================================

TEST(auxiliary_pre_checkin, CscGet)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 col_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 row_indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csc_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         col_ptrs,
                                         row_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, nnz_out;
    void*                col_ptrs_out;
    void*                row_indices_out;
    void*                values_out;
    rocsparse_indextype  col_idx_type, row_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_csc_get(descr,
                                &rows_out,
                                &cols_out,
                                &nnz_out,
                                &col_ptrs_out,
                                &row_indices_out,
                                &values_out,
                                &col_idx_type,
                                &row_idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(nnz_out, nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstCscGet)
{
    rocsparse_const_spmat_descr descr;
    int64_t                     rows = 100;
    int64_t                     cols = 50;
    int64_t                     nnz  = 10;
    const void* col_ptrs             = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    const void* row_indices          = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    const void* values               = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_csc_descr(&descr,
                                               rows,
                                               cols,
                                               nnz,
                                               col_ptrs,
                                               row_indices,
                                               values,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, nnz_out;
    const void*          col_ptrs_out;
    const void*          row_indices_out;
    const void*          values_out;
    rocsparse_indextype  col_idx_type, row_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_csc_get(descr,
                                      &rows_out,
                                      &cols_out,
                                      &nnz_out,
                                      &col_ptrs_out,
                                      &row_indices_out,
                                      &values_out,
                                      &col_idx_type,
                                      &row_idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(nnz_out, nnz);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, CscSetPointers)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 col_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 row_indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csc_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         col_ptrs,
                                         row_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    void* new_col_ptrs    = reinterpret_cast<void*>(0x1000);
    void* new_row_indices = reinterpret_cast<void*>(0x2000);
    void* new_values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_csc_set_pointers(descr, new_col_ptrs, new_row_indices, new_values),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, CscSetStridedBatch)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               nnz         = 10;
    void*                 col_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 row_indices = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_csc_descr(&descr,
                                         rows,
                                         cols,
                                         nnz,
                                         col_ptrs,
                                         row_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int     batch_count              = 5;
    int64_t offsets_batch_stride     = 500;
    int64_t rows_values_batch_stride = 1000;
    ASSERT_EQ(rocsparse_csc_set_strided_batch(
                  descr, batch_count, offsets_batch_stride, rows_values_batch_stride),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// ELL Matrix Get Tests
// =============================================================================

TEST(auxiliary_pre_checkin, EllGet)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               ell_width   = 5;
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    // Note: parameter order is rows, cols, col_indices, values, ell_width
    ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                         rows,
                                         cols,
                                         col_indices,
                                         values,
                                         ell_width,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, ell_width_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    // Note: rocsparse_ell_get parameter order is rows, cols, col_indices, values, ell_width
    ASSERT_EQ(rocsparse_ell_get(descr,
                                &rows_out,
                                &cols_out,
                                &col_indices_out,
                                &values_out,
                                &ell_width_out,
                                &idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(ell_width_out, ell_width);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstEllGet)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               ell_width   = 5;
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    // Note: parameter order is rows, cols, col_indices, values, ell_width
    ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                         rows,
                                         cols,
                                         col_indices,
                                         values,
                                         ell_width,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, ell_width_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    // Note: rocsparse_const_ell_get parameter order is rows, cols, col_indices, values, ell_width
    ASSERT_EQ(rocsparse_const_ell_get(descr,
                                      &rows_out,
                                      &cols_out,
                                      &col_indices_out,
                                      &values_out,
                                      &ell_width_out,
                                      &idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(ell_width_out, ell_width);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, EllSetPointers)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               ell_width   = 5;
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                         rows,
                                         cols,
                                         col_indices,
                                         values,
                                         ell_width,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    void* new_col_indices = reinterpret_cast<void*>(0x4000);
    void* new_values      = reinterpret_cast<void*>(0x5000);

    ASSERT_EQ(rocsparse_ell_set_pointers(descr, new_col_indices, new_values),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, EllSetPointersNullDescr)
{
    void* col_indices = reinterpret_cast<void*>(0x2000);
    void* values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_ell_set_pointers(nullptr, col_indices, values),
              rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, EllSetPointersNullColInd)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               ell_width   = 5;
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                         rows,
                                         cols,
                                         col_indices,
                                         values,
                                         ell_width,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    void* new_values = reinterpret_cast<void*>(0x5000);

    ASSERT_EQ(rocsparse_ell_set_pointers(descr, nullptr, new_values),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, EllSetPointersNullVal)
{
    rocsparse_spmat_descr descr;
    int64_t               rows        = 100;
    int64_t               cols        = 50;
    int64_t               ell_width   = 5;
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_ell_descr(&descr,
                                         rows,
                                         cols,
                                         col_indices,
                                         values,
                                         ell_width,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    void* new_col_indices = reinterpret_cast<void*>(0x4000);

    ASSERT_EQ(rocsparse_ell_set_pointers(descr, new_col_indices, nullptr),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// BELL Matrix Create and Get Tests
// =============================================================================

TEST(auxiliary_pre_checkin, ConstBellDescrCreateDestroy)
{
    rocsparse_const_spmat_descr descr;
    int64_t                     rows            = 100;
    int64_t                     cols            = 100;
    rocsparse_direction         dir             = rocsparse_direction_row;
    int64_t                     block_dim       = 2;
    int64_t                     ell_block_width = 5;
    const void* col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    const void* values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_bell_descr(&descr,
                                                rows,
                                                cols,
                                                dir,
                                                block_dim,
                                                ell_block_width,
                                                col_indices,
                                                values,
                                                rocsparse_indextype_i32,
                                                rocsparse_index_base_zero,
                                                rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, BellGet)
{
    rocsparse_spmat_descr descr;
    int64_t               rows            = 100;
    int64_t               cols            = 100;
    rocsparse_direction   dir             = rocsparse_direction_row;
    int64_t               block_dim       = 2;
    int64_t               ell_block_width = 5;
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_bell_descr(&descr,
                                          rows,
                                          cols,
                                          dir,
                                          block_dim,
                                          ell_block_width,
                                          col_indices,
                                          values,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, block_dim_out, ell_block_width_out;
    rocsparse_direction  dir_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_bell_get(descr,
                                 &rows_out,
                                 &cols_out,
                                 &dir_out,
                                 &block_dim_out,
                                 &ell_block_width_out,
                                 &col_indices_out,
                                 &values_out,
                                 &idx_type,
                                 &idx_base,
                                 &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(block_dim_out, block_dim);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstBellGet)
{
    rocsparse_const_spmat_descr descr;
    int64_t                     rows            = 100;
    int64_t                     cols            = 100;
    rocsparse_direction         dir             = rocsparse_direction_row;
    int64_t                     block_dim       = 2;
    int64_t                     ell_block_width = 5;
    const void* col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    const void* values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_bell_descr(&descr,
                                                rows,
                                                cols,
                                                dir,
                                                block_dim,
                                                ell_block_width,
                                                col_indices,
                                                values,
                                                rocsparse_indextype_i32,
                                                rocsparse_index_base_zero,
                                                rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, block_dim_out, ell_block_width_out;
    rocsparse_direction  dir_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_bell_get(descr,
                                       &rows_out,
                                       &cols_out,
                                       &dir_out,
                                       &block_dim_out,
                                       &ell_block_width_out,
                                       &col_indices_out,
                                       &values_out,
                                       &idx_type,
                                       &idx_base,
                                       &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(block_dim_out, block_dim);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// SELL Matrix Create and Get Tests
// =============================================================================

TEST(auxiliary_pre_checkin, SellDescrCreateDestroy)
{
    rocsparse_spmat_descr descr;
    int64_t               rows          = 100;
    int64_t               cols          = 100;
    int64_t               nnz           = 200;
    int64_t               slice_size    = 64;
    int64_t               colval_size   = 256;
    void*                 slice_offsets = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices   = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values        = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_sell_descr(&descr,
                                          rows,
                                          cols,
                                          nnz,
                                          slice_size,
                                          colval_size,
                                          slice_offsets,
                                          col_indices,
                                          values,
                                          rocsparse_indextype_i32,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstSellDescrCreateDestroy)
{
    rocsparse_const_spmat_descr descr;
    int64_t                     rows        = 100;
    int64_t                     cols        = 100;
    int64_t                     nnz         = 200;
    int64_t                     slice_size  = 64;
    int64_t                     colval_size = 256;
    const void* slice_offsets = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer
    const void* col_indices   = reinterpret_cast<const void*>(0x2000); // Non-null dummy pointer
    const void* values        = reinterpret_cast<const void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_sell_descr(&descr,
                                                rows,
                                                cols,
                                                nnz,
                                                slice_size,
                                                colval_size,
                                                slice_offsets,
                                                col_indices,
                                                values,
                                                rocsparse_indextype_i32,
                                                rocsparse_indextype_i32,
                                                rocsparse_index_base_zero,
                                                rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SellGet)
{
    rocsparse_spmat_descr descr;
    int64_t               rows          = 100;
    int64_t               cols          = 100;
    int64_t               nnz           = 200;
    int64_t               slice_size    = 64;
    int64_t               colval_size   = 256;
    void*                 slice_offsets = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices   = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values        = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_sell_descr(&descr,
                                          rows,
                                          cols,
                                          nnz,
                                          slice_size,
                                          colval_size,
                                          slice_offsets,
                                          col_indices,
                                          values,
                                          rocsparse_indextype_i32,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, nnz_out, slice_size_out, colval_size_out;
    void*                slice_offsets_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  slice_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_sell_get(descr,
                                 &rows_out,
                                 &cols_out,
                                 &nnz_out,
                                 &slice_size_out,
                                 &colval_size_out,
                                 &slice_offsets_out,
                                 &col_indices_out,
                                 &values_out,
                                 &slice_idx_type,
                                 &col_idx_type,
                                 &idx_base,
                                 &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(nnz_out, nnz);
    EXPECT_EQ(slice_size_out, slice_size);
    EXPECT_EQ(colval_size_out, colval_size);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstSellGet)
{
    rocsparse_const_spmat_descr descr;
    int64_t                     rows        = 100;
    int64_t                     cols        = 100;
    int64_t                     nnz         = 200;
    int64_t                     slice_size  = 64;
    int64_t                     colval_size = 256;
    const void* slice_offsets = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer
    const void* col_indices   = reinterpret_cast<const void*>(0x2000); // Non-null dummy pointer
    const void* values        = reinterpret_cast<const void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_sell_descr(&descr,
                                                rows,
                                                cols,
                                                nnz,
                                                slice_size,
                                                colval_size,
                                                slice_offsets,
                                                col_indices,
                                                values,
                                                rocsparse_indextype_i32,
                                                rocsparse_indextype_i32,
                                                rocsparse_index_base_zero,
                                                rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              rows_out, cols_out, nnz_out, slice_size_out, colval_size_out;
    const void*          slice_offsets_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  slice_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_sell_get(descr,
                                       &rows_out,
                                       &cols_out,
                                       &nnz_out,
                                       &slice_size_out,
                                       &colval_size_out,
                                       &slice_offsets_out,
                                       &col_indices_out,
                                       &values_out,
                                       &slice_idx_type,
                                       &col_idx_type,
                                       &idx_base,
                                       &data_type),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(nnz_out, nnz);
    EXPECT_EQ(slice_size_out, slice_size);
    EXPECT_EQ(colval_size_out, colval_size);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// BSR Matrix Create, Get, and Set Tests
// =============================================================================

TEST(auxiliary_pre_checkin, BsrDescrCreateDestroy)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_NE(descr, nullptr);
    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, BsrGet)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    void*                row_ptrs_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_bsr_get(descr,
                                &block_rows_out,
                                &block_cols_out,
                                &block_nnz_out,
                                &dir_out,
                                &block_dim_out,
                                &row_ptrs_out,
                                &col_indices_out,
                                &values_out,
                                &row_idx_type,
                                &col_idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_success);
    EXPECT_EQ(block_rows_out, block_rows);
    EXPECT_EQ(block_cols_out, block_cols);
    EXPECT_EQ(block_nnz_out, block_nnz);
    EXPECT_EQ(block_dim_out, block_dim);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstBsrGet)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    const void*          row_ptrs_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_bsr_get(descr,
                                      &block_rows_out,
                                      &block_cols_out,
                                      &block_nnz_out,
                                      &dir_out,
                                      &block_dim_out,
                                      &row_ptrs_out,
                                      &col_indices_out,
                                      &values_out,
                                      &row_idx_type,
                                      &col_idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_success);
    EXPECT_EQ(block_rows_out, block_rows);
    EXPECT_EQ(block_cols_out, block_cols);
    EXPECT_EQ(block_nnz_out, block_nnz);
    EXPECT_EQ(block_dim_out, block_dim);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// Const BSR Get validation tests for lines 3323-3346 in rocsparse_auxiliary.cpp

TEST(auxiliary_pre_checkin, ConstBsrGetNullDescr)
{
    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    const void*          row_ptrs_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_bsr_get(nullptr,
                                      &block_rows_out,
                                      &block_cols_out,
                                      &block_nnz_out,
                                      &dir_out,
                                      &block_dim_out,
                                      &row_ptrs_out,
                                      &col_indices_out,
                                      &values_out,
                                      &row_idx_type,
                                      &col_idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, ConstBsrGetNullBlockRows)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    const void*          row_ptrs_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_bsr_get(descr,
                                      nullptr,
                                      &block_cols_out,
                                      &block_nnz_out,
                                      &dir_out,
                                      &block_dim_out,
                                      &row_ptrs_out,
                                      &col_indices_out,
                                      &values_out,
                                      &row_idx_type,
                                      &col_idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstBsrGetNullBlockCols)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    const void*          row_ptrs_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_bsr_get(descr,
                                      &block_rows_out,
                                      nullptr,
                                      &block_nnz_out,
                                      &dir_out,
                                      &block_dim_out,
                                      &row_ptrs_out,
                                      &col_indices_out,
                                      &values_out,
                                      &row_idx_type,
                                      &col_idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstBsrGetNullBlockNnz)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_dim_out;
    rocsparse_direction  dir_out;
    const void*          row_ptrs_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_bsr_get(descr,
                                      &block_rows_out,
                                      &block_cols_out,
                                      nullptr,
                                      &dir_out,
                                      &block_dim_out,
                                      &row_ptrs_out,
                                      &col_indices_out,
                                      &values_out,
                                      &row_idx_type,
                                      &col_idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstBsrGetNullRowPtrs)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_bsr_get(descr,
                                      &block_rows_out,
                                      &block_cols_out,
                                      &block_nnz_out,
                                      &dir_out,
                                      &block_dim_out,
                                      nullptr,
                                      &col_indices_out,
                                      &values_out,
                                      &row_idx_type,
                                      &col_idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstBsrGetNullColIndices)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    const void*          row_ptrs_out;
    const void*          values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_bsr_get(descr,
                                      &block_rows_out,
                                      &block_cols_out,
                                      &block_nnz_out,
                                      &dir_out,
                                      &block_dim_out,
                                      &row_ptrs_out,
                                      nullptr,
                                      &values_out,
                                      &row_idx_type,
                                      &col_idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstBsrGetNullValues)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    const void*          row_ptrs_out;
    const void*          col_indices_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_bsr_get(descr,
                                      &block_rows_out,
                                      &block_cols_out,
                                      &block_nnz_out,
                                      &dir_out,
                                      &block_dim_out,
                                      &row_ptrs_out,
                                      &col_indices_out,
                                      nullptr,
                                      &row_idx_type,
                                      &col_idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstBsrGetNullRowIdxType)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    const void*          row_ptrs_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_bsr_get(descr,
                                      &block_rows_out,
                                      &block_cols_out,
                                      &block_nnz_out,
                                      &dir_out,
                                      &block_dim_out,
                                      &row_ptrs_out,
                                      &col_indices_out,
                                      &values_out,
                                      nullptr,
                                      &col_idx_type,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstBsrGetNullColIdxType)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    const void*          row_ptrs_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  row_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_const_bsr_get(descr,
                                      &block_rows_out,
                                      &block_cols_out,
                                      &block_nnz_out,
                                      &dir_out,
                                      &block_dim_out,
                                      &row_ptrs_out,
                                      &col_indices_out,
                                      &values_out,
                                      &row_idx_type,
                                      nullptr,
                                      &idx_base,
                                      &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstBsrGetNullIdxBase)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t             block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction dir_out;
    const void*         row_ptrs_out;
    const void*         col_indices_out;
    const void*         values_out;
    rocsparse_indextype row_idx_type, col_idx_type;
    rocsparse_datatype  data_type;

    ASSERT_EQ(rocsparse_const_bsr_get(descr,
                                      &block_rows_out,
                                      &block_cols_out,
                                      &block_nnz_out,
                                      &dir_out,
                                      &block_dim_out,
                                      &row_ptrs_out,
                                      &col_indices_out,
                                      &values_out,
                                      &row_idx_type,
                                      &col_idx_type,
                                      nullptr,
                                      &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstBsrGetNullDataType)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    const void*          row_ptrs_out;
    const void*          col_indices_out;
    const void*          values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;

    ASSERT_EQ(rocsparse_const_bsr_get(descr,
                                      &block_rows_out,
                                      &block_cols_out,
                                      &block_nnz_out,
                                      &dir_out,
                                      &block_dim_out,
                                      &row_ptrs_out,
                                      &col_indices_out,
                                      &values_out,
                                      &row_idx_type,
                                      &col_idx_type,
                                      &idx_base,
                                      nullptr),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, BsrSetPointers)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 col_indices = reinterpret_cast<void*>(0x2000); // Non-null dummy pointer
    void*                 values      = reinterpret_cast<void*>(0x3000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    void* new_row_ptrs    = reinterpret_cast<void*>(0x4000);
    void* new_col_indices = reinterpret_cast<void*>(0x5000);
    void* new_values      = reinterpret_cast<void*>(0x6000);

    ASSERT_EQ(rocsparse_bsr_set_pointers(descr, new_row_ptrs, new_col_indices, new_values),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// BSR Get validation tests for lines 3323-3346 in rocsparse_auxiliary.cpp

TEST(auxiliary_pre_checkin, BsrGetNullDescr)
{
    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    void*                row_ptrs_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_bsr_get(nullptr,
                                &block_rows_out,
                                &block_cols_out,
                                &block_nnz_out,
                                &dir_out,
                                &block_dim_out,
                                &row_ptrs_out,
                                &col_indices_out,
                                &values_out,
                                &row_idx_type,
                                &col_idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_invalid_pointer);
}

TEST(auxiliary_pre_checkin, BsrGetNullBlockRows)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    void*                row_ptrs_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_bsr_get(descr,
                                nullptr,
                                &block_cols_out,
                                &block_nnz_out,
                                &dir_out,
                                &block_dim_out,
                                &row_ptrs_out,
                                &col_indices_out,
                                &values_out,
                                &row_idx_type,
                                &col_idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, BsrGetNullBlockCols)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    void*                row_ptrs_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_bsr_get(descr,
                                &block_rows_out,
                                nullptr,
                                &block_nnz_out,
                                &dir_out,
                                &block_dim_out,
                                &row_ptrs_out,
                                &col_indices_out,
                                &values_out,
                                &row_idx_type,
                                &col_idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, BsrGetNullBlockNnz)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_dim_out;
    rocsparse_direction  dir_out;
    void*                row_ptrs_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_bsr_get(descr,
                                &block_rows_out,
                                &block_cols_out,
                                nullptr,
                                &dir_out,
                                &block_dim_out,
                                &row_ptrs_out,
                                &col_indices_out,
                                &values_out,
                                &row_idx_type,
                                &col_idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, BsrGetNullRowPtrs)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_bsr_get(descr,
                                &block_rows_out,
                                &block_cols_out,
                                &block_nnz_out,
                                &dir_out,
                                &block_dim_out,
                                nullptr,
                                &col_indices_out,
                                &values_out,
                                &row_idx_type,
                                &col_idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, BsrGetNullColIndices)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    void*                row_ptrs_out;
    void*                values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_bsr_get(descr,
                                &block_rows_out,
                                &block_cols_out,
                                &block_nnz_out,
                                &dir_out,
                                &block_dim_out,
                                &row_ptrs_out,
                                nullptr,
                                &values_out,
                                &row_idx_type,
                                &col_idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, BsrGetNullValues)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    void*                row_ptrs_out;
    void*                col_indices_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_bsr_get(descr,
                                &block_rows_out,
                                &block_cols_out,
                                &block_nnz_out,
                                &dir_out,
                                &block_dim_out,
                                &row_ptrs_out,
                                &col_indices_out,
                                nullptr,
                                &row_idx_type,
                                &col_idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, BsrGetNullRowIdxType)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    void*                row_ptrs_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  col_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_bsr_get(descr,
                                &block_rows_out,
                                &block_cols_out,
                                &block_nnz_out,
                                &dir_out,
                                &block_dim_out,
                                &row_ptrs_out,
                                &col_indices_out,
                                &values_out,
                                nullptr,
                                &col_idx_type,
                                &idx_base,
                                &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, BsrGetNullColIdxType)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    void*                row_ptrs_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  row_idx_type;
    rocsparse_index_base idx_base;
    rocsparse_datatype   data_type;

    ASSERT_EQ(rocsparse_bsr_get(descr,
                                &block_rows_out,
                                &block_cols_out,
                                &block_nnz_out,
                                &dir_out,
                                &block_dim_out,
                                &row_ptrs_out,
                                &col_indices_out,
                                &values_out,
                                &row_idx_type,
                                nullptr,
                                &idx_base,
                                &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, BsrGetNullIdxBase)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t             block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction dir_out;
    void*               row_ptrs_out;
    void*               col_indices_out;
    void*               values_out;
    rocsparse_indextype row_idx_type, col_idx_type;
    rocsparse_datatype  data_type;

    ASSERT_EQ(rocsparse_bsr_get(descr,
                                &block_rows_out,
                                &block_cols_out,
                                &block_nnz_out,
                                &dir_out,
                                &block_dim_out,
                                &row_ptrs_out,
                                &col_indices_out,
                                &values_out,
                                &row_idx_type,
                                &col_idx_type,
                                nullptr,
                                &data_type),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, BsrGetNullDataType)
{
    rocsparse_spmat_descr descr;
    int64_t               block_rows  = 10;
    int64_t               block_cols  = 10;
    int64_t               block_nnz   = 5;
    rocsparse_direction   dir         = rocsparse_direction_row;
    int64_t               block_dim   = 4;
    void*                 row_ptrs    = reinterpret_cast<void*>(0x1000);
    void*                 col_indices = reinterpret_cast<void*>(0x2000);
    void*                 values      = reinterpret_cast<void*>(0x3000);

    ASSERT_EQ(rocsparse_create_bsr_descr(&descr,
                                         block_rows,
                                         block_cols,
                                         block_nnz,
                                         dir,
                                         block_dim,
                                         row_ptrs,
                                         col_indices,
                                         values,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t              block_rows_out, block_cols_out, block_nnz_out, block_dim_out;
    rocsparse_direction  dir_out;
    void*                row_ptrs_out;
    void*                col_indices_out;
    void*                values_out;
    rocsparse_indextype  row_idx_type, col_idx_type;
    rocsparse_index_base idx_base;

    ASSERT_EQ(rocsparse_bsr_get(descr,
                                &block_rows_out,
                                &block_cols_out,
                                &block_nnz_out,
                                &dir_out,
                                &block_dim_out,
                                &row_ptrs_out,
                                &col_indices_out,
                                &values_out,
                                &row_idx_type,
                                &col_idx_type,
                                &idx_base,
                                nullptr),
              rocsparse_status_invalid_pointer);

    ASSERT_EQ(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// Dense Vector Get/Set Tests
// =============================================================================

TEST(auxiliary_pre_checkin, DnvecGet)
{
    rocsparse_dnvec_descr descr;
    int64_t               size   = 100;
    void*                 values = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_dnvec_descr(&descr, size, values, rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t            size_out;
    void*              values_out;
    rocsparse_datatype data_type;

    ASSERT_EQ(rocsparse_dnvec_get(descr, &size_out, &values_out, &data_type),
              rocsparse_status_success);
    EXPECT_EQ(size_out, size);

    ASSERT_EQ(rocsparse_destroy_dnvec_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstDnvecGet)
{
    rocsparse_const_dnvec_descr descr;
    int64_t                     size = 100;
    const void* values = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_dnvec_descr(&descr, size, values, rocsparse_datatype_f32_r),
              rocsparse_status_success);

    int64_t            size_out;
    const void*        values_out;
    rocsparse_datatype data_type;

    ASSERT_EQ(rocsparse_const_dnvec_get(descr, &size_out, &values_out, &data_type),
              rocsparse_status_success);
    EXPECT_EQ(size_out, size);

    ASSERT_EQ(rocsparse_destroy_dnvec_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, DnvecGetSetValues)
{
    rocsparse_dnvec_descr descr;
    int64_t               size       = 100;
    void*                 values     = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 new_values = reinterpret_cast<void*>(0x12345678);

    ASSERT_EQ(rocsparse_create_dnvec_descr(&descr, size, values, rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Set values
    ASSERT_EQ(rocsparse_dnvec_set_values(descr, new_values), rocsparse_status_success);

    // Get values
    void* values_out;
    ASSERT_EQ(rocsparse_dnvec_get_values(descr, &values_out), rocsparse_status_success);
    EXPECT_EQ(values_out, new_values);

    ASSERT_EQ(rocsparse_destroy_dnvec_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstDnvecGetValues)
{
    rocsparse_const_dnvec_descr descr;
    int64_t                     size   = 100;
    const void*                 values = reinterpret_cast<const void*>(0x12345678);

    ASSERT_EQ(rocsparse_create_const_dnvec_descr(&descr, size, values, rocsparse_datatype_f32_r),
              rocsparse_status_success);

    const void* values_out;
    ASSERT_EQ(rocsparse_const_dnvec_get_values(descr, &values_out), rocsparse_status_success);
    EXPECT_EQ(values_out, values);

    ASSERT_EQ(rocsparse_destroy_dnvec_descr(descr), rocsparse_status_success);
}

// =============================================================================
// Dense Matrix Get/Set Tests
// =============================================================================

TEST(auxiliary_pre_checkin, DnmatGet)
{
    rocsparse_dnmat_descr descr;
    int64_t               rows   = 100;
    int64_t               cols   = 50;
    int64_t               ld     = 100;
    void*                 values = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);

    int64_t            rows_out, cols_out, ld_out;
    void*              values_out;
    rocsparse_datatype data_type;
    rocsparse_order    order;

    ASSERT_EQ(
        rocsparse_dnmat_get(descr, &rows_out, &cols_out, &ld_out, &values_out, &data_type, &order),
        rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(ld_out, ld);

    ASSERT_EQ(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstDnmatGet)
{
    rocsparse_const_dnmat_descr descr;
    int64_t                     rows = 100;
    int64_t                     cols = 50;
    int64_t                     ld   = 100;
    const void* values = reinterpret_cast<const void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_const_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);

    int64_t            rows_out, cols_out, ld_out;
    const void*        values_out;
    rocsparse_datatype data_type;
    rocsparse_order    order;

    ASSERT_EQ(rocsparse_const_dnmat_get(
                  descr, &rows_out, &cols_out, &ld_out, &values_out, &data_type, &order),
              rocsparse_status_success);
    EXPECT_EQ(rows_out, rows);
    EXPECT_EQ(cols_out, cols);
    EXPECT_EQ(ld_out, ld);

    ASSERT_EQ(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, DnmatGetSetValues)
{
    rocsparse_dnmat_descr descr;
    int64_t               rows       = 100;
    int64_t               cols       = 50;
    int64_t               ld         = 100;
    void*                 values     = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer
    void*                 new_values = reinterpret_cast<void*>(0x12345678);

    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);

    // Set values
    ASSERT_EQ(rocsparse_dnmat_set_values(descr, new_values), rocsparse_status_success);

    // Get values
    void* values_out;
    ASSERT_EQ(rocsparse_dnmat_get_values(descr, &values_out), rocsparse_status_success);
    EXPECT_EQ(values_out, new_values);

    ASSERT_EQ(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, ConstDnmatGetValues)
{
    rocsparse_const_dnmat_descr descr;
    int64_t                     rows   = 100;
    int64_t                     cols   = 50;
    int64_t                     ld     = 100;
    const void*                 values = reinterpret_cast<const void*>(0x12345678);

    ASSERT_EQ(rocsparse_create_const_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);

    const void* values_out;
    ASSERT_EQ(rocsparse_const_dnmat_get_values(descr, &values_out), rocsparse_status_success);
    EXPECT_EQ(values_out, values);

    ASSERT_EQ(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, DnmatGetSetStridedBatch)
{
    rocsparse_dnmat_descr descr;
    int64_t               rows   = 100;
    int64_t               cols   = 50;
    int64_t               ld     = 100;
    void*                 values = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);

    // Set strided batch
    int     batch_count  = 5;
    int64_t batch_stride = 5000;
    ASSERT_EQ(rocsparse_dnmat_set_strided_batch(descr, batch_count, batch_stride),
              rocsparse_status_success);

    // Get strided batch
    int     batch_count_out;
    int64_t batch_stride_out;
    ASSERT_EQ(rocsparse_dnmat_get_strided_batch(descr, &batch_count_out, &batch_stride_out),
              rocsparse_status_success);
    EXPECT_EQ(batch_count_out, batch_count);
    EXPECT_EQ(batch_stride_out, batch_stride);

    ASSERT_EQ(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, DnmatGetSetStridedBatchRowMajor)
{
    rocsparse_dnmat_descr descr;
    int64_t               rows   = 100;
    int64_t               cols   = 50;
    int64_t               ld     = 50; // For row-major, ld must be >= cols
    void*                 values = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_row),
              rocsparse_status_success);

    // Set strided batch - valid: batch_stride >= ld * rows (50 * 100 = 5000)
    int     batch_count  = 5;
    int64_t batch_stride = 5000;
    ASSERT_EQ(rocsparse_dnmat_set_strided_batch(descr, batch_count, batch_stride),
              rocsparse_status_success);

    // Get strided batch
    int     batch_count_out;
    int64_t batch_stride_out;
    ASSERT_EQ(rocsparse_dnmat_get_strided_batch(descr, &batch_count_out, &batch_stride_out),
              rocsparse_status_success);
    EXPECT_EQ(batch_count_out, batch_count);
    EXPECT_EQ(batch_stride_out, batch_stride);

    ASSERT_EQ(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, DnmatSetStridedBatchRowMajorInvalidStride)
{
    rocsparse_dnmat_descr descr;
    int64_t               rows   = 100;
    int64_t               cols   = 50;
    int64_t               ld     = 50; // For row-major, ld must be >= cols
    void*                 values = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_row),
              rocsparse_status_success);

    // Invalid: batch_stride < ld * rows when batch_count > 1
    // ld * rows = 50 * 100 = 5000, so batch_stride = 4999 is invalid
    int     batch_count  = 5;
    int64_t batch_stride = 4999;
    ASSERT_EQ(rocsparse_dnmat_set_strided_batch(descr, batch_count, batch_stride),
              rocsparse_status_invalid_value);

    ASSERT_EQ(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, DnmatSetStridedBatchColumnMajorInvalidStride)
{
    rocsparse_dnmat_descr descr;
    int64_t               rows   = 100;
    int64_t               cols   = 50;
    int64_t               ld     = 100;
    void*                 values = reinterpret_cast<void*>(0x1000); // Non-null dummy pointer

    ASSERT_EQ(rocsparse_create_dnmat_descr(
                  &descr, rows, cols, ld, values, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);

    // Invalid: batch_stride < ld * cols when batch_count > 1
    // ld * cols = 100 * 50 = 5000, so batch_stride = 4999 is invalid
    int     batch_count  = 5;
    int64_t batch_stride = 4999;
    ASSERT_EQ(rocsparse_dnmat_set_strided_batch(descr, batch_count, batch_stride),
              rocsparse_status_invalid_value);

    ASSERT_EQ(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
}

// =============================================================================
// SpGEAM Set Input / Get Output Tests
// =============================================================================

TEST(auxiliary_pre_checkin, SpgeamSetInputGetOutput)
{
    rocsparse_handle handle;
    ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);

    rocsparse_spgeam_descr descr;
    ASSERT_EQ(rocsparse_create_spgeam_descr(&descr), rocsparse_status_success);

    rocsparse_error error = nullptr;

    // Set algorithm
    rocsparse_spgeam_alg alg = rocsparse_spgeam_alg_default;
    ASSERT_EQ(rocsparse_spgeam_set_input(
                  handle, descr, rocsparse_spgeam_input_alg, &alg, sizeof(alg), &error),
              rocsparse_status_success);

    // Set scalar datatype
    rocsparse_datatype scalar_type = rocsparse_datatype_f32_r;
    ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                         descr,
                                         rocsparse_spgeam_input_scalar_datatype,
                                         &scalar_type,
                                         sizeof(scalar_type),
                                         &error),
              rocsparse_status_success);

    // Set compute datatype
    rocsparse_datatype compute_type = rocsparse_datatype_f32_r;
    ASSERT_EQ(rocsparse_spgeam_set_input(handle,
                                         descr,
                                         rocsparse_spgeam_input_compute_datatype,
                                         &compute_type,
                                         sizeof(compute_type),
                                         &error),
              rocsparse_status_success);

    // Set operation A
    rocsparse_operation op_A = rocsparse_operation_none;
    ASSERT_EQ(rocsparse_spgeam_set_input(
                  handle, descr, rocsparse_spgeam_input_operation_A, &op_A, sizeof(op_A), &error),
              rocsparse_status_success);

    // Set operation B
    rocsparse_operation op_B = rocsparse_operation_transpose;
    ASSERT_EQ(rocsparse_spgeam_set_input(
                  handle, descr, rocsparse_spgeam_input_operation_B, &op_B, sizeof(op_B), &error),
              rocsparse_status_success);

    // Set scalar alpha (pointer to scalar)
    float alpha     = 1.0f;
    void* alpha_ptr = &alpha;
    ASSERT_EQ(
        rocsparse_spgeam_set_input(
            handle, descr, rocsparse_spgeam_input_scalar_alpha, &alpha_ptr, sizeof(void*), &error),
        rocsparse_status_success);

    // Set scalar beta (pointer to scalar)
    float beta     = 2.0f;
    void* beta_ptr = &beta;
    ASSERT_EQ(
        rocsparse_spgeam_set_input(
            handle, descr, rocsparse_spgeam_input_scalar_beta, &beta_ptr, sizeof(void*), &error),
        rocsparse_status_success);

    // Get output nnz (even though we haven't computed anything, just test the function)
    int64_t nnz_out;
    ASSERT_EQ(rocsparse_spgeam_get_output(
                  handle, descr, rocsparse_spgeam_output_nnz, &nnz_out, sizeof(nnz_out), &error),
              rocsparse_status_success);

    ASSERT_EQ(rocsparse_destroy_spgeam_descr(descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpgeamSetInputInvalidHandle)
{
    rocsparse_spgeam_descr descr;
    ASSERT_EQ(rocsparse_create_spgeam_descr(&descr), rocsparse_status_success);

    rocsparse_spgeam_alg alg   = rocsparse_spgeam_alg_default;
    rocsparse_error      error = nullptr;

    ASSERT_EQ(rocsparse_spgeam_set_input(
                  nullptr, descr, rocsparse_spgeam_input_alg, &alg, sizeof(alg), &error),
              rocsparse_status_invalid_handle);

    ASSERT_EQ(rocsparse_destroy_spgeam_descr(descr), rocsparse_status_success);
}

TEST(auxiliary_pre_checkin, SpgeamGetOutputInvalidHandle)
{
    rocsparse_spgeam_descr descr;
    ASSERT_EQ(rocsparse_create_spgeam_descr(&descr), rocsparse_status_success);

    int64_t         nnz_out;
    rocsparse_error error = nullptr;

    ASSERT_EQ(rocsparse_spgeam_get_output(
                  nullptr, descr, rocsparse_spgeam_output_nnz, &nnz_out, sizeof(nnz_out), &error),
              rocsparse_status_invalid_handle);

    ASSERT_EQ(rocsparse_destroy_spgeam_descr(descr), rocsparse_status_success);
}
