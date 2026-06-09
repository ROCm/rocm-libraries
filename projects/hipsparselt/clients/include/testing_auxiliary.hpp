/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "flops.hpp"
#include "hipsparselt_datatype2string.hpp"
#include "hipsparselt_init.hpp"
#include "hipsparselt_math.hpp"
#include "hipsparselt_random.hpp"
#include "hipsparselt_test.hpp"
#include "hipsparselt_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include <hipsparselt/hipsparselt.h>

// ============================================================
// hipsparseLtGetVersion
// ============================================================

void testing_aux_get_version_match(const Arguments& arg)
{
    static int version;
    hipsparselt_local_handle handle;
    hipsparseLtGetVersion(handle, &version);
    int major, minor, patch;
    hipsparseLtGetProperty(HIP_LIBRARY_MAJOR_VERSION, &major);
    hipsparseLtGetProperty(HIP_LIBRARY_MINOR_VERSION, &minor);
    hipsparseLtGetProperty(HIP_LIBRARY_PATCH_LEVEL, &patch);
    int version_ = major * 100000 + minor * 100 + patch;
    ASSERT_EQ(version, version_);
}

void testing_aux_get_version_null_handle(const Arguments& arg)
{
    int version;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtGetVersion(nullptr, &version),
                            HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_get_version_null_version(const Arguments& arg)
{
    hipsparselt_local_handle handle;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtGetVersion(handle, nullptr),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

// ============================================================
// hipsparseLtGetProperty
// ============================================================

void testing_aux_get_property_null_value(const Arguments& arg)
{
    EXPECT_HIPSPARSE_STATUS(hipsparseLtGetProperty(HIP_LIBRARY_MAJOR_VERSION, nullptr),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

// ============================================================
// hipsparseLtGetGitRevision
// ============================================================

void testing_aux_get_version_git_rev_null(const Arguments& arg)
{
    hipsparselt_local_handle handle;
    char* rev = nullptr;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtGetGitRevision(handle, rev), HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_get_git_revision_uninit_handle(const Arguments& arg)
{
    hipsparseLtHandle_t handle_;
    char                rev[64];
    EXPECT_HIPSPARSE_STATUS(hipsparseLtGetGitRevision(handle_, rev),
                            HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_get_git_revision_valid(const Arguments& arg)
{
    hipsparselt_local_handle handle;
    char                     rev[64] = {};
    EXPECT_HIPSPARSE_STATUS(hipsparseLtGetGitRevision(handle, rev), HIPSPARSE_STATUS_SUCCESS);
}

// ============================================================
// hipsparseLtGetArchName
// ============================================================

void testing_aux_get_arch_name(const Arguments& arg)
{
    char* archName = nullptr;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtGetArchName(&archName), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_get_arch_name_null(const Arguments& arg)
{
    EXPECT_HIPSPARSE_STATUS(hipsparseLtGetArchName(nullptr), HIPSPARSE_STATUS_INVALID_VALUE);
}

// ============================================================
// hipsparseLtInit
// ============================================================

void testing_aux_handle_init_bad_arg(const Arguments& arg)
{
    EXPECT_HIPSPARSE_STATUS(hipsparseLtInit(nullptr), HIPSPARSE_STATUS_INVALID_VALUE);
}

// ============================================================
// hipsparseLtDestroy
// ============================================================

void testing_aux_handle(const Arguments& arg)
{
    hipsparseLtHandle_t handle;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtInit(&handle), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtDestroy(&handle), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_handle_destroy_bad_arg_uninit(const Arguments& arg)
{
    hipsparseLtHandle_t handle;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtDestroy(&handle), HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_handle_destroy_bad_arg_null(const Arguments& arg)
{
    EXPECT_HIPSPARSE_STATUS(hipsparseLtDestroy(nullptr), HIPSPARSE_STATUS_SUCCESS);
}

// ============================================================
// hipsparseLtDenseDescriptorInit
// ============================================================

void testing_aux_mat_dense_init(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_dense, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_mat_dense_init_row_order(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_dense, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_ROW);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_mat_init_dense_bad_arg_uninit_handle(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparseLtHandle_t        handle;
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            &handle, &m_descr, row, col, ld, 16, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_dense_bad_arg_null_handle(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            nullptr, &m_descr, row, col, ld, 16, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_dense_bad_arg_null_descr(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle handle{arg};
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            handle, nullptr, row, col, ld, 16, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_dense_bad_arg_zero_row(const Arguments& arg)
{
    const int64_t col = 128, ld = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            handle, &m_descr, 0, col, ld, 16, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_dense_bad_arg_zero_col(const Arguments& arg)
{
    const int64_t row = 128, ld = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            handle, &m_descr, row, 0, ld, 16, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_dense_bad_arg_zero_ld(const Arguments& arg)
{
    const int64_t row = 128, col = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            handle, &m_descr, row, col, 0, 16, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_dense_bad_arg_large_ld(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_NVIDIA__
    const int64_t row = 128, col = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            handle, &m_descr, row, col, 129, 16, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif
}

void testing_aux_mat_init_dense_bad_arg_large_alignment(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_NVIDIA__
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtDenseDescriptorInit(
            handle, &m_descr, row, col, ld, 17, arg.a_type, HIPSPARSE_ORDER_COL),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif
}

// ============================================================
// hipsparseLtStructuredDescriptorInit
// ============================================================

void testing_aux_mat_structured_init(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_mat_init_structured_bad_arg_uninit_handle(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparseLtHandle_t        handle_;
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(&handle_, &m_descr, row, col, ld, 16, arg.a_type,
                                            HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_structured_bad_arg_null_handle(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(nullptr, &m_descr, row, col, ld, 16, arg.a_type,
                                            HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_structured_bad_arg_null_descr(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle handle{arg};
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(handle, nullptr, row, col, ld, 16, arg.a_type,
                                            HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_structured_bad_arg_zero_row(const Arguments& arg)
{
    const int64_t col = 128, ld = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(handle, &m_descr, 0, col, ld, 16, arg.a_type,
                                            HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_structured_bad_arg_small_row(const Arguments& arg)
{
    const int64_t col = 128, ld = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(handle, &m_descr, 6, col, ld, 16, arg.a_type,
                                            HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
}

void testing_aux_mat_init_structured_bad_arg_zero_col(const Arguments& arg)
{
    const int64_t row = 128, ld = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(handle, &m_descr, row, 0, ld, 16, arg.a_type,
                                            HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_structured_bad_arg_small_col(const Arguments& arg)
{
    const int64_t row = 128, ld = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(handle, &m_descr, row, 6, ld, 16, arg.a_type,
                                            HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
}

void testing_aux_mat_init_structured_bad_arg_zero_ld(const Arguments& arg)
{
    const int64_t row = 128, col = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(handle, &m_descr, row, col, 0, 16, arg.a_type,
                                            HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_structured_bad_arg_unaligned_ld(const Arguments& arg)
{
    const int64_t row = 128, col = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(handle, &m_descr, row, col, 127, 16, arg.a_type,
                                            HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init_structured_bad_arg_unaligned_row(const Arguments& arg)
{
    const int64_t col = 128, ld = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    int num_elements = 8;
    switch(arg.a_type)
    {
    case HIP_R_8I:
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
    case HIP_R_8F_E5M2:
#endif
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
    case HIP_R_8F_E5M2_FNUZ:
#endif
        num_elements = 16;
        break;
    default:
        break;
    }
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(handle, &m_descr, num_elements * 4 + 4, col, ld, 16,
                                            arg.a_type, HIPSPARSE_ORDER_COL,
                                            HIPSPARSELT_SPARSITY_50_PERCENT),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
}

void testing_aux_mat_init_structured_bad_arg_unsupported_type(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(handle, &m_descr, row, col, ld, 16, HIP_R_64F,
                                            HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
}

void testing_aux_mat_init_structured_bad_arg_large_ld(const Arguments& arg)
{
    const int64_t row = 128, col = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(handle, &m_descr, row, col, 129, 16, arg.a_type,
                                            HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT),
#ifdef __HIP_PLATFORM_NVIDIA__
        HIPSPARSE_STATUS_NOT_SUPPORTED
#else
        HIPSPARSE_STATUS_SUCCESS
#endif
    );
}

void testing_aux_mat_init_structured_bad_arg_large_alignment(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_NVIDIA__
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle   handle{arg};
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtStructuredDescriptorInit(handle, &m_descr, row, col, ld, 17, arg.a_type,
                                            HIPSPARSE_ORDER_COL, HIPSPARSELT_SPARSITY_50_PERCENT),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif
}

// ============================================================
// hipsparseLtMatDescriptorDestroy
// ============================================================

void testing_aux_mat_destroy_bad_arg_uninit(const Arguments& arg)
{
    hipsparseLtMatDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescriptorDestroy(&m_descr), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_mat_destroy_bad_arg_null(const Arguments& arg)
{
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescriptorDestroy(nullptr),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

// ============================================================
// hipsparseLtMatDescSetAttribute / hipsparseLtMatDescGetAttribute
// ============================================================

void testing_aux_mat_set_get_attr_num_batches(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int data = 2, data_r = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data_r, sizeof(int)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data_r == data);
}

void testing_aux_mat_set_get_attr_batch_stride(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    std::vector<int64_t> strides = {0, ld * col};
    int64_t              data64_r = -1;
    for(int64_t stride : strides)
    {
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, &stride, sizeof(int64_t)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescGetAttribute(
                handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, &data64_r, sizeof(int64_t)),
            HIPSPARSE_STATUS_SUCCESS);
        ASSERT_TRUE(data64_r == stride);
    }
}

void testing_aux_mat_assign_copy_value(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int data = 1, data2 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtMatDescriptor_t mat2 = mat;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(
            handle, &mat2, HIPSPARSELT_MAT_NUM_BATCHES, &data2, sizeof(data2)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data == data2);
}

void testing_aux_mat_assign_not_reference(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int data = 1;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtMatDescriptor_t mat2 = mat;
    int data2 = 10;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, &mat2, HIPSPARSELT_MAT_NUM_BATCHES, &data2, sizeof(data2)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(
            handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data != data2);
}

void testing_aux_mat_set_attr_bad_arg_null_handle(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int data;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            nullptr, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg_uninit_handle(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtHandle_t handle_;
    int data;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            &handle_, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg_null_descr(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int data;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, nullptr, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg_uninit_descr(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtMatDescriptor_t mat_;
    int data;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, &mat_, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg_null_data(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, nullptr, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg_zero_batches(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg_wrong_size_batches(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int data = 1;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg_null_stride(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, nullptr, sizeof(int64_t)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg_invalid_stride(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int64_t data64 = 2;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(
            handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, &data64, sizeof(int64_t)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg_wrong_size_stride(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int64_t data64 = ld * col;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescSetAttribute(handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_get_attr_bad_arg_null_handle(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int data;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(
            nullptr, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_get_attr_bad_arg_uninit_handle(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtHandle_t handle_;
    int data;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(
            &handle_, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_get_attr_bad_arg_null_descr(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int data;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(
            handle, nullptr, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_get_attr_bad_arg_uninit_descr(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtMatDescriptor_t mat_;
    int data;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(
            handle, &mat_, HIPSPARSELT_MAT_NUM_BATCHES, &data, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_get_attr_bad_arg_null_data(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(
            handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, nullptr, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_get_attr_bad_arg_wrong_size_batches(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int data;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(handle, mat, HIPSPARSELT_MAT_NUM_BATCHES, &data, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_get_attr_bad_arg_null_stride(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(
            handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, nullptr, sizeof(int64_t)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_mat_get_attr_bad_arg_wrong_size_stride(const Arguments& arg)
{
    const int64_t row = 128, col = 128, ld = 128;
    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr mat(
        hipsparselt_matrix_type_structured, handle, row, col, ld, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(mat.status(), HIPSPARSE_STATUS_SUCCESS);
    int64_t data64;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatDescGetAttribute(
            handle, mat, HIPSPARSELT_MAT_BATCH_STRIDE, &data64, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

// ============================================================
// hipsparseLtMatmulDescriptorInit
// ============================================================

namespace
{
    struct matmul_bad_arg_fixture
    {
        static const int64_t M = 128, N = 128, K = 128;
        static const int64_t lda = 128, ldb = 128, ldc = 128;
        static constexpr hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
        static constexpr hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    };
} // namespace

void testing_aux_matmul_init(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, M, K, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_matmul_init_matB_sparse(const Arguments& arg)
{
    // opA=NON_TRANSPOSE: op(A) = A (M×K), matA is dense M×K
    // opB=TRANSPOSE:     op(B) = B^T (K×N), matB is structured N×K (sparse)
    const int64_t M = 128, N = 128, K = 128;
    const int64_t lda = 128, ldb = 128, ldc = 128;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_dense,      handle, M, K, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_structured, handle, N, K, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense,      handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense,      handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE,
        matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_matmul_init_bad_arg_null_handle(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(nullptr, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_uninit_handle(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtHandle_t handle_;
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(&handle_, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_null_descr(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, nullptr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_conj_opA(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_conj_opB(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
            matA, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_uninit_matA(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatDescriptor_t mat_;
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            &mat_, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_null_matA(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            nullptr, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_uninit_matB(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatDescriptor_t mat_;
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, &mat_, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_null_matB(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, nullptr, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_uninit_matC(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatDescriptor_t mat_;
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, &mat_, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_null_matC(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, nullptr, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_uninit_matD(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatDescriptor_t mat_;
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC, &mat_, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_null_matD(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC, nullptr, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_same_op_int8(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_NVIDIA__
    if(arg.a_type != HIP_R_8I)
        return;
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE,
            matA, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif
}

void testing_aux_matmul_init_bad_arg_two_sparse(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matBS(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matBS, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
}

void testing_aux_matmul_init_bad_arg_wrong_compute_type(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    hipsparseLtComputetype_t tmpComputeType;
    switch(arg.a_type)
    {
    case HIP_R_16F:
    case HIP_R_16BF:
        tmpComputeType = HIPSPARSELT_COMPUTE_32I;
        break;
    default:
#ifdef __HIP_PLATFORM_AMD__
        tmpComputeType = HIPSPARSELT_COMPUTE_32F;
#else
        tmpComputeType = HIPSPARSELT_COMPUTE_16F;
#endif
        break;
    }
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC, matD, tmpComputeType),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
}

void testing_aux_matmul_init_bad_arg_structured_C(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matCS(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matCS, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_mismatched_order(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matDR(hipsparselt_matrix_type_dense,      handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_ROW);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC, matDR, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_mismatched_K(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB_bad(hipsparselt_matrix_type_dense, handle, 112, 128, 112, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB_bad, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_mismatched_N(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB_bad(hipsparselt_matrix_type_dense, handle, 128, 112, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB_bad, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init_bad_arg_mismatched_C_dim(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC_bad(hipsparselt_matrix_type_dense, handle, 112, 112, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC_bad, matD, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

void testing_aux_matmul_init_bad_arg_mismatched_D_dim(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD_bad(hipsparselt_matrix_type_dense, handle, 112, 112, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC, matD_bad, arg.compute_type),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

void testing_aux_matmul_init_bad_arg_unsupported_A_type(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA_bad(hipsparselt_matrix_type_structured, handle, 128, 128, 128, HIP_R_32F, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA_bad, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
}

void testing_aux_matmul_init_bad_arg_mismatched_B_type(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipDataType diffType = (arg.b_type == HIP_R_16BF) ? HIP_R_16F : HIP_R_16BF;
    hipsparselt_local_mat_descr matB_bad(hipsparselt_matrix_type_dense, handle, 128, 128, 128, diffType, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB_bad, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
}

void testing_aux_matmul_init_bad_arg_mismatched_C_type(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    if(arg.a_type == HIP_R_8I)
        return;
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipDataType diffType = (arg.c_type == HIP_R_16BF) ? HIP_R_16F : HIP_R_16BF;
    hipsparselt_local_mat_descr matC_bad(hipsparselt_matrix_type_dense, handle, 128, 128, 128, diffType, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC_bad, matD, arg.compute_type),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif
}

void testing_aux_matmul_init_bad_arg_mismatched_D_type(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    if(arg.a_type == HIP_R_8I)
        return;
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipDataType diffType = (arg.d_type == HIP_R_16BF) ? HIP_R_16F : HIP_R_16BF;
    hipsparselt_local_mat_descr matD_bad(hipsparselt_matrix_type_dense, handle, 128, 128, 128, diffType, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t m_descr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &m_descr,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC, matD_bad, arg.compute_type),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif
}

// ============================================================
// hipsparseLtMatmulDescSetAttribute / hipsparseLtMatmulDescGetAttribute
// ============================================================

void testing_aux_matmul_set_get_attr_relu(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    int data = 1, data_r = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data_r, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data == data_r);
}

void testing_aux_matmul_set_get_attr_relu_upperbound(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    float dataf = 1.0f, dataf_r = 0.0f;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, &dataf, sizeof(dataf)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, &dataf_r, sizeof(dataf)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(dataf == dataf_r);
}

void testing_aux_matmul_set_get_bias_vector(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    device_vector<float> dBias(M, 1);
    CHECK_DEVICE_ALLOCATION(dBias.memcheck());
    host_vector<float> hBias_gold(M);
    host_vector<float> hBias(M);

    hipsparselt_seedrand();
    hipsparselt_init<float>(hBias_gold, M, 1, M, M, 1);
    CHECK_HIP_ERROR(dBias.transfer_from(hBias_gold));

    void* _dBias = dBias;

    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_BIAS_POINTER, &_dBias, sizeof(void*)),
        HIPSPARSE_STATUS_SUCCESS);

    void* dBias_r;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_BIAS_POINTER, &dBias_r, sizeof(void*)),
        HIPSPARSE_STATUS_SUCCESS);

    CHECK_HIP_ERROR(hipMemcpy(hBias, dBias_r, sizeof(float) * M, hipMemcpyDeviceToHost));

    unit_check_general<float>(M, 1, M, M, hBias_gold, hBias, 1);
}

void testing_aux_matmul_set_get_attr_gelu(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    int data = 1, data_r = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_GELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_GELU, &data_r, sizeof(data_r)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data == data_r);

    float scalef = 0.5f, scalef_r = 0.0f;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING, &scalef, sizeof(scalef)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING, &scalef_r, sizeof(scalef_r)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(scalef == scalef_r);
}

void testing_aux_matmul_set_get_attr_abs(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__    
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    int data = 1, data_r = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_ABS, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_ABS, &data_r, sizeof(data_r)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data == data_r);
#endif
}

void testing_aux_matmul_set_get_attr_leakyrelu(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    int data = 1, data_r = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU, &data_r, sizeof(data_r)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data == data_r);

    float alpha = 0.01f, alpha_r = 0.0f;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA, &alpha, sizeof(alpha)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA, &alpha_r, sizeof(alpha_r)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(alpha == alpha_r);
#endif    
}

void testing_aux_matmul_set_get_attr_tanh(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    if(arg.d_type == HIP_R_8I)
        return;

    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    int data = 1, data_r = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH, &data_r, sizeof(data_r)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data == data_r);

    float alpha = 1.0f, alpha_r = 0.0f;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA, &alpha, sizeof(alpha)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA, &alpha_r, sizeof(alpha_r)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(alpha == alpha_r);

    float beta = 1.0f, beta_r = 0.0f;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA, &beta, sizeof(beta)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA, &beta_r, sizeof(beta_r)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(beta == beta_r);
#endif
}

void testing_aux_matmul_set_attr_alpha_vector_scaling(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    // bad arg: null data
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, nullptr, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);

    // bad arg: wrong size
    int data = 1;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &data, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);

    // success: enable alpha vector scaling
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);

    int data_r = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(
            handle, matmul, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &data_r, sizeof(data_r)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data == data_r);
}

void testing_aux_matmul_assign_copy_value(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t matmul;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &matmul,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_SUCCESS);
    int data = 1;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, &matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtMatmulDescriptor_t lMatmul = matmul;
    int data_r = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, &lMatmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data_r, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data == data_r);
}

void testing_aux_matmul_assign_not_reference(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparseLtMatmulDescriptor_t matmul;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescriptorInit(handle, &matmul,
            HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            matA, matB, matC, matD, arg.compute_type),
        HIPSPARSE_STATUS_SUCCESS);
    int data = 1;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, &matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    {
        hipsparseLtMatmulDescriptor_t lMatmul = matmul;
        int data_r = 100;
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatmulDescSetAttribute(handle, &lMatmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data_r, sizeof(data_r)),
            HIPSPARSE_STATUS_SUCCESS);
    }
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, &matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data != 100);
}

void testing_aux_matmul_set_attr_bad_arg_null_handle(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(nullptr, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_set_attr_bad_arg_uninit_handle(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtHandle_t handle_;
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(&handle_, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_set_attr_bad_arg_null_matmul(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, nullptr, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_set_attr_bad_arg_uninit_matmul(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtMatmulDescriptor_t matmul_;
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, &matmul_, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_set_attr_bad_arg_null_data(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, nullptr, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_set_attr_bad_arg_wrong_size(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_set_attr_bad_arg_relu_upperbound_wrong_size(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    int data64 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_set_attr_bad_arg_sigmoid_int8(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    if(arg.d_type != HIP_R_8I)
        return;
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    int dataSigmoid = 1;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID, &dataSigmoid, sizeof(dataSigmoid)),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif
}

void testing_aux_matmul_set_attr_bad_arg_bias_pointer_wrong_size(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    void* dBias = nullptr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, matmul, HIPSPARSELT_MATMUL_BIAS_POINTER, &dBias, sizeof(dBias) - 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_set_attr_bad_arg_bias_stride_invalid(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    int64_t bias_stride = 128 - 1; // M - 1
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, matmul, HIPSPARSELT_MATMUL_BIAS_STRIDE, &bias_stride, sizeof(bias_stride)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_set_attr_bad_arg_bias_type(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    char bias_type = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescSetAttribute(handle, matmul, HIPSPARSELT_MATMUL_BIAS_TYPE, &bias_type, sizeof(char)),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

#define MATMUL_GET_ATTR_SETUP() \
    hipsparselt_local_handle handle{arg}; \
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL); \
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL); \
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL); \
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL); \
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type); \
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS)

void testing_aux_matmul_get_attr_bad_arg_null_handle(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(nullptr, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_uninit_handle(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    hipsparseLtHandle_t handle_;
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(&handle_, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_null_matmul(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, nullptr, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_uninit_matmul(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    hipsparseLtMatmulDescriptor_t matmul_;
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, &matmul_, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_null_data(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, nullptr, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_wrong_size(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU, &data, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_relu_upperbound_null(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    int data64 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, nullptr, sizeof(data64)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_relu_upperbound_wrong_size(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    int data64 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_relu_threshold_null(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD, nullptr, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_relu_threshold_wrong_size(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    int data64 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_leakyrelu_alpha_null(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    MATMUL_GET_ATTR_SETUP();
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA, nullptr, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

void testing_aux_matmul_get_attr_bad_arg_leakyrelu_alpha_wrong_size(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    MATMUL_GET_ATTR_SETUP();
    int data64 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

void testing_aux_matmul_get_attr_bad_arg_tanh_alpha_null(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    MATMUL_GET_ATTR_SETUP();
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA, nullptr, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

void testing_aux_matmul_get_attr_bad_arg_tanh_alpha_wrong_size(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    MATMUL_GET_ATTR_SETUP();
    int data64 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

void testing_aux_matmul_get_attr_bad_arg_tanh_beta_null(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    MATMUL_GET_ATTR_SETUP();
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA, nullptr, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

void testing_aux_matmul_get_attr_bad_arg_tanh_beta_wrong_size(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    MATMUL_GET_ATTR_SETUP();
    int data64 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

void testing_aux_matmul_get_attr_bad_arg_bias_pointer_wrong_size(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    size_t bad_ptr_size = sizeof(void*) - 1;
    void* dBias;
    CHECK_HIP_ERROR(hipMalloc((void**)&dBias, 128 * sizeof(float)));
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_BIAS_POINTER, &dBias, bad_ptr_size),
        HIPSPARSE_STATUS_INVALID_VALUE);
    CHECK_HIP_ERROR(hipFree(dBias));
}

void testing_aux_matmul_get_attr_bad_arg_bias_stride_null(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    int data64 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_BIAS_STRIDE, nullptr, sizeof(data64)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_bias_stride_wrong_size(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    int data64 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_BIAS_STRIDE, &data64, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_bias_type_null(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    MATMUL_GET_ATTR_SETUP();
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_BIAS_TYPE, nullptr, sizeof(hipDataType)),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

void testing_aux_matmul_get_attr_bad_arg_bias_type_wrong_size(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    MATMUL_GET_ATTR_SETUP();
    hipDataType biasType;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_BIAS_TYPE, &biasType, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

void testing_aux_matmul_get_attr_bad_arg_alpha_vector_scaling_null(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, nullptr, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_alpha_vector_scaling_wrong_size(const Arguments& arg)
{
    MATMUL_GET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &data, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg_beta_vector_scaling(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    MATMUL_GET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulDescGetAttribute(handle, matmul, HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING, &data, sizeof(data)),
        HIPSPARSE_STATUS_NOT_SUPPORTED);
#endif
}

#undef MATMUL_GET_ATTR_SETUP

// ============================================================
// hipsparseLtMatmulAlgSelectionInit
// ============================================================

void testing_aux_matmul_alg_init(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_matmul_alg_init_bad_arg_null_handle(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtMatmulAlgSelection_t alg_sel;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSelectionInit(nullptr, &alg_sel, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_init_bad_arg_uninit_handle(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtHandle_t handle_;
    hipsparseLtMatmulAlgSelection_t alg_sel;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSelectionInit(&handle_, &alg_sel, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_init_bad_arg_null_alg_sel(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSelectionInit(handle, nullptr, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_init_bad_arg_null_matmul(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparseLtMatmulAlgSelection_t alg_sel;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSelectionInit(handle, &alg_sel, nullptr, HIPSPARSELT_MATMUL_ALG_DEFAULT),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_init_bad_arg_uninit_matmul(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparseLtMatmulDescriptor_t   matmul_;
    hipsparseLtMatmulAlgSelection_t alg_sel;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSelectionInit(handle, &alg_sel, &matmul_, HIPSPARSELT_MATMUL_ALG_DEFAULT),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

// ============================================================
// hipsparseLtMatmulAlgSelectionDestroy
// ============================================================

void testing_aux_matmul_alg_sel_destroy(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparseLtMatmulAlgSelection_t alg_sel;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSelectionInit(handle, &alg_sel, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulAlgSelectionDestroy(&alg_sel),
                            HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_matmul_alg_sel_destroy_bad_arg_null(const Arguments& arg)
{
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulAlgSelectionDestroy(nullptr),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_sel_destroy_bad_arg_uninit(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    hipsparseLtMatmulAlgSelection_t alg_sel;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulAlgSelectionDestroy(&alg_sel),
                            HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

// ============================================================
// hipsparseLtMatmulAlgSetAttribute / hipsparseLtMatmulAlgGetAttribute
// ============================================================

void testing_aux_matmul_alg_assign_copy_value(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);
    int data = 20, data2 = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtMatmulAlgSelection_t alg_sel2 = alg_sel;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(handle, &alg_sel2, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data2, sizeof(data2)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data2 == data);
}

void testing_aux_matmul_alg_assign_not_reference(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);
    int data = 20;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtMatmulAlgSelection_t alg_sel2 = alg_sel;
    int data2 = 100;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(handle, &alg_sel2, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data2, sizeof(data2)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(data2 != data);
}

#define ALG_SET_ATTR_SETUP() \
    hipsparselt_local_handle handle{arg}; \
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL); \
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL); \
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL); \
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL); \
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type); \
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT); \
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS)

void testing_aux_matmul_alg_set_attr_bad_arg_null_handle(const Arguments& arg)
{
    ALG_SET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(nullptr, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_set_attr_bad_arg_uninit_handle(const Arguments& arg)
{
    ALG_SET_ATTR_SETUP();
    hipsparseLtHandle_t handle_;
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(&handle_, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_set_attr_bad_arg_null_alg_sel(const Arguments& arg)
{
    ALG_SET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(handle, nullptr, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_set_attr_bad_arg_uninit_alg_sel(const Arguments& arg)
{
    ALG_SET_ATTR_SETUP();
    hipsparseLtMatmulAlgSelection_t alg_sel_;
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(handle, &alg_sel_, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_set_attr_bad_arg_config_max_id(const Arguments& arg)
{
    ALG_SET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_set_attr_bad_arg_split_k(const Arguments& arg)
{
    ALG_SET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_SPLIT_K, &data, sizeof(data)),
#ifdef __HIP_PLATFORM_AMD__        
        HIPSPARSE_STATUS_NOT_SUPPORTED
#else
        HIPSPARSE_STATUS_SUCCESS
#endif
    );
}

void testing_aux_matmul_alg_set_attr_bad_arg_null_data(const Arguments& arg)
{
    ALG_SET_ATTR_SETUP();
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, nullptr, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_set_attr_bad_arg_wrong_size(const Arguments& arg)
{
    ALG_SET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_set_attr_bad_arg_config_id_out_of_range(const Arguments& arg)
{
    ALG_SET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_SUCCESS);
    data++;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_set_attr_bad_arg_search_iterations_wrong_size(const Arguments& arg)
{
    ALG_SET_ATTR_SETUP();
    int data = 100;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_set_attr_bad_arg_search_iterations_zero(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    ALG_SET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_SEARCH_ITERATIONS, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

void testing_aux_matmul_alg_set_attr_bad_arg_split_k_mode(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparseLtSplitKMode_t data = HIPSPARSELT_SPLIT_K_MODE_ONE_KERNEL;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_SPLIT_K_MODE, &data, sizeof(data)),    
#ifdef __HIP_PLATFORM_AMD__
        HIPSPARSE_STATUS_NOT_SUPPORTED
#else
        HIPSPARSE_STATUS_SUCCESS
#endif
        );
}

void testing_aux_matmul_alg_set_attr_bad_arg_split_k_buffers(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);

    void* buf = nullptr;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgSetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_SPLIT_K_BUFFERS, &buf, sizeof(buf)),
#ifdef __HIP_PLATFORM_AMD__
        HIPSPARSE_STATUS_NOT_SUPPORTED
#else
        HIPSPARSE_STATUS_SUCCESS
#endif
        );
}

void testing_aux_matmul_alg_get_attr_max_id(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense,       handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);

    int max_id = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(
            handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &max_id, sizeof(max_id)),
        HIPSPARSE_STATUS_SUCCESS);
    ASSERT_TRUE(max_id >= 0);
}

#undef ALG_SET_ATTR_SETUP

#define ALG_GET_ATTR_SETUP() \
    hipsparselt_local_handle handle{arg}; \
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL); \
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL); \
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL); \
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL); \
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type); \
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT); \
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS)

void testing_aux_matmul_alg_get_attr_bad_arg_null_handle(const Arguments& arg)
{
    ALG_GET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(nullptr, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_get_attr_bad_arg_uninit_handle(const Arguments& arg)
{
    ALG_GET_ATTR_SETUP();
    hipsparseLtHandle_t handle_;
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(&handle_, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_get_attr_bad_arg_null_alg_sel(const Arguments& arg)
{
    ALG_GET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(handle, nullptr, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_get_attr_bad_arg_uninit_alg_sel(const Arguments& arg)
{
    ALG_GET_ATTR_SETUP();
    hipsparseLtMatmulAlgSelection_t alg_sel_;
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(handle, &alg_sel_, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, sizeof(data)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_get_attr_bad_arg_null_data(const Arguments& arg)
{
    ALG_GET_ATTR_SETUP();
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, nullptr, sizeof(int)),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_alg_get_attr_bad_arg_wrong_size(const Arguments& arg)
{
    ALG_GET_ATTR_SETUP();
    int data = 0;
    EXPECT_HIPSPARSE_STATUS(
        hipsparseLtMatmulAlgGetAttribute(handle, alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &data, 1),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

#undef ALG_GET_ATTR_SETUP

// ============================================================
// hipsparseLtMatmulGetWorkspace
// ============================================================

void testing_aux_get_workspace_size(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);
    EXPECT_HIPSPARSE_STATUS(plan.status(), HIPSPARSE_STATUS_SUCCESS);

    size_t workspace_size = 0;

    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(handle, plan, &workspace_size),
                            HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_get_workspace_size_bad_arg_null_handle(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);
    size_t workspace_size = 0;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(nullptr, plan, &workspace_size),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_get_workspace_size_bad_arg_uninit_handle(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);
    hipsparseLtHandle_t handle_;
    size_t workspace_size = 0;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(&handle_, plan, &workspace_size),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_get_workspace_size_bad_arg_null_plan(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    size_t workspace_size = 0;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(handle, nullptr, &workspace_size),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_get_workspace_size_bad_arg_null_size(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(handle, plan, nullptr),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_get_workspace_size_bad_arg_uninit_plan(const Arguments& arg)
{
#ifdef __HIP_PLATFORM_AMD__
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);
    hipsparseLtMatmulPlan_t plan_;
    size_t workspace_size = 0;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulGetWorkspace(handle, &plan_, &workspace_size),
                            HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

// ============================================================
// hipsparseLtMatmulPlanInit
// ============================================================

void testing_aux_matmul_plan_init(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const hipsparseOperation_t opA = HIPSPARSE_OPERATION_TRANSPOSE;
    const hipsparseOperation_t opB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    hipsparselt_local_handle handle{arg};

    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, K, M, lda, arg.a_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matA.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matB.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matC.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, HIPSPARSE_ORDER_COL);
    EXPECT_HIPSPARSE_STATUS(matD.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_descr matmul(
        handle, opA, opB, matA, matB, matC, matD, arg.compute_type);
    EXPECT_HIPSPARSE_STATUS(matmul.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(alg_sel.status(), HIPSPARSE_STATUS_SUCCESS);

    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);
    EXPECT_HIPSPARSE_STATUS(plan.status(), HIPSPARSE_STATUS_SUCCESS);
}

void testing_aux_matmul_plan_init_bad_arg_null_handle(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    hipsparseLtMatmulPlan_t plan;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(nullptr, &plan, matmul, alg_sel),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_plan_init_bad_arg_uninit_handle(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    hipsparseLtHandle_t handle_;
    hipsparseLtMatmulPlan_t plan;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(&handle_, &plan, matmul, alg_sel),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_plan_init_bad_arg_null_plan(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, nullptr, matmul, alg_sel),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_plan_init_bad_arg_null_matmul(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    hipsparseLtMatmulPlan_t plan;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, &plan, nullptr, alg_sel),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_plan_init_bad_arg_uninit_matmul(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    hipsparseLtMatmulDescriptor_t matmul_;
    hipsparseLtMatmulPlan_t plan;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, &plan, &matmul_, alg_sel),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_plan_init_bad_arg_null_alg_sel(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    hipsparseLtMatmulPlan_t plan;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, &plan, matmul, nullptr),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_plan_init_bad_arg_uninit_alg_sel(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    hipsparseLtMatmulAlgSelection_t alg_sel_;
    hipsparseLtMatmulPlan_t plan;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, &plan, matmul, &alg_sel_),
                            HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_plan_init_bad_arg_mismatched_batches(const Arguments& arg)
{
    hipsparselt_local_handle handle{arg};
    hipsparselt_local_mat_descr matA(hipsparselt_matrix_type_structured, handle, 128, 128, 128, arg.a_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matB(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.b_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matC(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.c_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_mat_descr matD(hipsparselt_matrix_type_dense, handle, 128, 128, 128, arg.d_type, HIPSPARSE_ORDER_COL);
    hipsparselt_local_matmul_descr matmul(handle, HIPSPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);
    int num_batches_a = 2, num_batches_b = 3;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
            handle, matA, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches_a, sizeof(num_batches_a)),
        HIPSPARSE_STATUS_SUCCESS);
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatDescSetAttribute(
            handle, matB, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches_b, sizeof(num_batches_b)),
        HIPSPARSE_STATUS_SUCCESS);
    hipsparseLtMatmulPlan_t plan;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanInit(handle, &plan, matmul, alg_sel),
        HIPSPARSE_STATUS_INVALID_VALUE);
}

// ============================================================
// hipsparseLtMatmulPlanDestroy
// ============================================================

void testing_aux_matmul_plan_destroy_bad_arg_null(const Arguments& arg)
{
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanDestroy(nullptr), HIPSPARSE_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_plan_destroy_bad_arg_uninit(const Arguments& arg)
{
    hipsparseLtMatmulPlan_t plan;
    EXPECT_HIPSPARSE_STATUS(hipsparseLtMatmulPlanDestroy(&plan), HIPSPARSE_STATUS_INVALID_VALUE);
}
